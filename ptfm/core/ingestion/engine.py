"""
PTFM Ingestion Engine

File:
    ptfm/core/ingestion/engine.py

Description:
    Sequences the ingestion pipeline stages for one run, managing concurrency,
    fault isolation, and cancellation.

What the engine does and does not do
------------------------------------
    The engine sequences stages and manages concurrency. It knows nothing
    about PDFs, filesystems, or any knowledge domain: every collaborator
    arrives through the constructor as an interface, and no stage is
    constructed here.

    It performs no parsing, no scoring, no calibration, and emits no
    confidence value of any kind. It is a pure transformation and dispatch
    layer.

Streaming, not batching
-----------------------
    The public surface yields outcomes one at a time. A corpus is never
    materialized as a list, so a run over half a terabyte of documents holds
    only the documents currently in flight, and a slow consumer applies
    natural back-pressure rather than accumulating an unbounded queue.

Ordering
--------
    Outcomes are yielded in completion order, not discovery order. Documents
    are processed concurrently and vary by orders of magnitude in cost, so
    preserving discovery order would require buffering completed documents
    behind a slow one, which would defeat the memory bound that streaming
    exists to provide.

    Where discovery order matters, a consumer can sort by ``source_uri``, or
    set ``max_concurrent_documents`` to one.

Fault isolation
---------------
    A failure in one document produces a ``DocumentFailed`` outcome and the
    run continues. A corrupt file at position forty thousand must not discard
    the preceding work.

    Two conditions abort a run. Configuration and connector faults are
    run-level and raise immediately, since every subsequent document would
    fail identically. A failure ratio exceeding the configured guard, once
    enough documents have been processed for the ratio to mean anything,
    indicates a systemically bad corpus rather than isolated bad documents.

Cancellation and resource safety
--------------------------------
    Every payload is released in a ``finally`` block, so a cancelled or
    aborted run leaves no spooled temporary files behind. The connector is
    closed through its asynchronous context manager, which releases transport
    resources on every exit path including cancellation.

Dependency contract
-------------------
    Imports only the standard library, ``ptfm.core.schemas`` and the ingestion
    ``exceptions``, ``config``, ``models``, ``interfaces``, ``registry``,
    ``telemetry`` and pipeline stage modules. No connector, extractor, or
    plugin is imported.
"""

from __future__ import annotations

import asyncio
import time
from typing import AsyncIterator, Final, Optional

from ptfm.core.ingestion.config import IngestionConfig
from ptfm.core.ingestion.exceptions import (
    IngestionConfigurationError,
    IngestionStage,
    PTFMIngestionError,
    SourceError,
)
from ptfm.core.ingestion.interfaces import (
    Clock,
    FormatDetector,
    IdGenerator,
    MetricsSink,
    SourceConnector,
    StructuredLogger,
    ThreadOffloader,
)
from ptfm.core.ingestion.models import (
    AcquiredBlob,
    DocumentFailed,
    DocumentIngested,
    DocumentSkipped,
    IngestionFailure,
    IngestionOutcome,
    IngestionReport,
    SourceRecord,
    SourceSpec,
)
from ptfm.core.ingestion.pipeline.acquisition import AcquisitionStage
from ptfm.core.ingestion.pipeline.deduplication import Deduplicator
from ptfm.core.ingestion.pipeline.extraction import ExtractionStage
from ptfm.core.ingestion.pipeline.metadata import MetadataHarvester
from ptfm.core.ingestion.pipeline.normalization import RawDocumentNormalizer
from ptfm.core.ingestion.pipeline.structure import StructureNormalizer
from ptfm.core.ingestion.registry import ConnectorRegistry
from ptfm.core.schemas import RawDocument

__all__ = ["IngestionEngine"]

#: Metric names emitted by the engine. Declared as constants so that a name
#: cannot drift between its emission sites.
_METRIC_DISCOVERED: Final[str] = "ptfm.ingestion.documents.discovered"
_METRIC_INGESTED: Final[str] = "ptfm.ingestion.documents.ingested"
_METRIC_SKIPPED: Final[str] = "ptfm.ingestion.documents.skipped"
_METRIC_FAILED: Final[str] = "ptfm.ingestion.documents.failed"
_METRIC_DURATION: Final[str] = "ptfm.ingestion.document.duration_ms"
_METRIC_BYTES: Final[str] = "ptfm.ingestion.document.size_bytes"

#: Milliseconds per second, for rendering durations.
_MS_PER_SECOND: Final[float] = 1000.0


class IngestionEngine:
    """
    Executes the ingestion pipeline for one source specification.

    Every collaborator is injected. The engine constructs nothing and holds no
    module-level state, so two engines with different configurations can run
    concurrently in one process without interfering.

    Parameters
    ----------
    config:
        Complete ingestion configuration. Supplies concurrency limits, the
        failure guard, and the settings each stage consults.
    connector_registry:
        Resolves a URI scheme to a connector. Dispatch by scheme alone is what
        keeps the engine free of transport-specific branching.
    acquisition:
        Discovery and transfer stage.
    detector:
        Media type and encoding resolution.
    extraction:
        Content extraction stage.
    structure:
        Text and block tree construction.
    metadata:
        Generic metadata harvesting.
    deduplicator:
        Duplicate recognition.
    normalizer:
        Canonical document assembly.
    offloader:
        Executes blocking stage work off the event loop.
    clock:
        Supplies run timestamps.
    id_generator:
        Supplies the run correlation identifier. Never used for document
        identity, which is a deterministic derivation.
    logger:
        Structured logger. Cannot fail a run.
    metrics:
        Metrics sink. Cannot fail a run.
    """

    __slots__ = (
        "_config",
        "_connectors",
        "_acquisition",
        "_detector",
        "_extraction",
        "_structure",
        "_metadata",
        "_deduplicator",
        "_normalizer",
        "_offloader",
        "_clock",
        "_id_generator",
        "_logger",
        "_metrics",
    )

    def __init__(
        self,
        *,
        config: IngestionConfig,
        connector_registry: ConnectorRegistry,
        acquisition: AcquisitionStage,
        detector: FormatDetector,
        extraction: ExtractionStage,
        structure: StructureNormalizer,
        metadata: MetadataHarvester,
        deduplicator: Deduplicator,
        normalizer: RawDocumentNormalizer,
        offloader: ThreadOffloader,
        clock: Clock,
        id_generator: IdGenerator,
        logger: StructuredLogger,
        metrics: MetricsSink,
    ) -> None:
        self._config = config
        self._connectors = connector_registry
        self._acquisition = acquisition
        self._detector = detector
        self._extraction = extraction
        self._structure = structure
        self._metadata = metadata
        self._deduplicator = deduplicator
        self._normalizer = normalizer
        self._offloader = offloader
        self._clock = clock
        self._id_generator = id_generator
        self._logger = logger
        self._metrics = metrics

    # ------------------------------------------------------------------
    # Public surface
    # ------------------------------------------------------------------

    async def ingest(self, spec: SourceSpec) -> AsyncIterator[IngestionOutcome]:
        """
        Ingest a source, yielding one outcome per discovered document.

        Outcomes arrive in completion order rather than discovery order, since
        documents are processed concurrently and vary widely in cost. Ordering
        by discovery would require buffering completed documents behind slow
        ones, defeating the memory bound streaming provides.

        Implemented as an async generator. Nothing is done until the caller
        begins iterating, and abandoning the iterator cancels the run and
        releases every resource it held.

        Parameters
        ----------
        spec:
            Source specification: root URI, selection filters, and an optional
            incremental cursor.

        Yields
        ------
        IngestionOutcome
            A successful document, an intentional skip, or a failure record.

        Raises
        ------
        UnsupportedSourceSchemeError
            If no connector is registered for the specification's scheme.
        SourceError
            If the source could not be reached or enumerated.
        IngestionConfigurationError
            If the run was aborted by the failure guard, or the connector
            reported itself unhealthy.
        """
        run_id = self._id_generator.new_id()
        logger = self._logger.bind(run_id=run_id, root_uri=spec.root_uri)
        started = self._clock.now()

        logger.info(
            "ingestion.run.started",
            scheme=spec.scheme,
            max_concurrent=self._config.max_concurrent_documents,
        )

        counters = _RunCounters()
        connector = self._connectors.create(spec.scheme, self._config)

        try:
            async with connector:
                await self._verify_connector(connector, spec)
                async for outcome in self._run(connector, spec, counters, logger):
                    yield outcome
        finally:
            # Emitted on every exit path, including cancellation, so that an
            # abandoned run still reports what it accomplished.
            logger.info(
                "ingestion.run.finished",
                discovered=counters.discovered,
                ingested=counters.ingested,
                skipped=counters.skipped,
                failed=counters.failed,
                duration_seconds=(self._clock.now() - started).total_seconds(),
            )

    async def ingest_all(self, spec: SourceSpec) -> IngestionReport:
        """
        Ingest a source to completion, discarding documents and returning a
        report.

        Intended for operational runs where documents are persisted by a
        consumer registered elsewhere, or for verifying that a corpus ingests
        cleanly before committing to processing it. A caller needing the
        documents themselves should iterate :meth:`ingest` instead.

        Parameters
        ----------
        spec:
            Source specification.

        Returns
        -------
        IngestionReport
            Run-level aggregate counts and diagnostics.

        Raises
        ------
        PTFMIngestionError
            If the run aborted. The report is not produced in that case, since
            an aborted run's counts describe an incomplete traversal.
        """
        run_id = self._id_generator.new_id()
        started = self._clock.now()
        counters = _RunCounters()
        aborted_reason: Optional[str] = None

        try:
            async for outcome in self.ingest(spec):
                counters.record(outcome)
        except PTFMIngestionError as error:
            aborted_reason = error.message
            raise
        finally:
            del aborted_reason

        return IngestionReport(
            run_id=run_id,
            root_uri=spec.root_uri,
            started_at=started,
            completed_at=self._clock.now(),
            discovered_count=counters.discovered,
            ingested_count=counters.ingested,
            skipped_count=counters.skipped,
            failed_count=counters.failed,
            failures_by_stage=dict(counters.failures_by_stage),
            failures_by_code=dict(counters.failures_by_code),
            documents_by_media_type=dict(counters.by_media_type),
            total_bytes=counters.total_bytes,
        )

    async def ingest_one(
        self,
        record: SourceRecord,
        connector: SourceConnector,
    ) -> IngestionOutcome:
        """
        Ingest a single already-discovered record.

        Provided for embedded use, such as a service handling one upload,
        where discovery has already happened or does not apply. The caller
        owns the connector's lifetime.

        Parameters
        ----------
        record:
            The record to ingest.
        connector:
            The connector to fetch from.

        Returns
        -------
        IngestionOutcome
            The document, a skip, or a failure record. Document-level failures
            are returned rather than raised, so that this method behaves
            consistently with the streaming surface.
        """
        return await self._process(record, connector, self._logger)

    # ------------------------------------------------------------------
    # Run orchestration
    # ------------------------------------------------------------------

    async def _run(
        self,
        connector: SourceConnector,
        spec: SourceSpec,
        counters: _RunCounters,
        logger: StructuredLogger,
    ) -> AsyncIterator[IngestionOutcome]:
        """
        Process discovered records with bounded concurrency.

        A fixed number of documents are kept in flight. As each completes its
        outcome is yielded and another record is drawn from discovery, so
        memory is bounded by the concurrency limit rather than by corpus size,
        and a slow consumer naturally slows discovery.

        Parameters
        ----------
        connector:
            The opened connector.
        spec:
            Source specification.
        counters:
            Run counters, updated as outcomes complete.
        logger:
            Logger bound to this run.

        Yields
        ------
        IngestionOutcome
            Outcomes in completion order.

        Raises
        ------
        IngestionConfigurationError
            If the failure guard aborted the run.
        SourceError
            If discovery failed.
        """
        limit = self._config.max_concurrent_documents
        records = self._acquisition.discover(connector, spec).__aiter__()
        in_flight: set[asyncio.Task[IngestionOutcome]] = set()
        exhausted = False

        try:
            while True:
                while not exhausted and len(in_flight) < limit:
                    try:
                        record = await records.__anext__()
                    except StopAsyncIteration:
                        exhausted = True
                        break

                    counters.discovered += 1
                    self._metrics.increment(_METRIC_DISCOVERED)
                    in_flight.add(
                        asyncio.create_task(
                            self._process(record, connector, logger)
                        )
                    )

                if not in_flight:
                    return

                done, pending = await asyncio.wait(
                    in_flight, return_when=asyncio.FIRST_COMPLETED
                )
                in_flight = set(pending)

                for task in done:
                    outcome = task.result()
                    counters.record(outcome)
                    self._record_metrics(outcome)
                    self._check_failure_guard(counters, logger)
                    yield outcome

        finally:
            # Reached on normal completion, on an abort, and when the consumer
            # abandons the iterator. Every in-flight document is cancelled so
            # that no task outlives the run holding a spooled payload.
            for task in in_flight:
                task.cancel()
            if in_flight:
                await asyncio.gather(*in_flight, return_exceptions=True)
            await _close_iterator(records)

    async def _verify_connector(
        self,
        connector: SourceConnector,
        spec: SourceSpec,
    ) -> None:
        """
        Verify that a connector is operational before discovery begins.

        Checked up front so that a misconfiguration fails immediately rather
        than after partial ingestion, and so that an unreachable source is
        distinguishable from an empty one.

        Parameters
        ----------
        connector:
            The connector to verify.
        spec:
            Source specification, for error provenance.

        Returns
        -------
        None

        Raises
        ------
        IngestionConfigurationError
            If the connector reported itself unhealthy.
        SourceError
            If the health check itself failed.
        """
        try:
            healthy = await connector.health_check()
        except PTFMIngestionError:
            raise
        except Exception as exc:
            raise SourceError(
                f"Health check for {connector.name!r} raised "
                f"{type(exc).__name__}: {exc}",
                source_uri=spec.root_uri,
                context={"connector_name": connector.name},
            ) from exc

        if not healthy:
            raise IngestionConfigurationError(
                f"Connector {connector.name!r} reported that it is not "
                f"operational for {spec.root_uri!r}.",
                context={
                    "connector_name": connector.name,
                    "root_uri": spec.root_uri,
                },
            )

    def _check_failure_guard(
        self,
        counters: _RunCounters,
        logger: StructuredLogger,
    ) -> None:
        """
        Abort a run whose failure rate indicates a systemic problem.

        The guard requires a minimum number of processed documents before it
        can fire. Without that, a run would abort when the first of one
        document failed, which is an isolated fault rather than evidence about
        the corpus.

        This is an operational guard over run health. It carries no truth
        judgement and never influences verification.

        Parameters
        ----------
        counters:
            Current run counters.
        logger:
            Logger bound to this run.

        Returns
        -------
        None

        Raises
        ------
        IngestionConfigurationError
            If the failure ratio exceeds the configured guard.
        """
        if self._config.fail_fast and counters.failed > 0:
            logger.error("ingestion.run.aborted", reason="fail_fast")
            raise IngestionConfigurationError(
                "Run aborted after the first document failure because "
                "fail_fast is enabled.",
                context={"processed": counters.processed, "failed": counters.failed},
            )

        if counters.processed < self._config.failure_ratio_min_samples:
            return

        ratio = counters.failed / counters.processed
        if ratio <= self._config.failure_ratio_threshold:
            return

        logger.error(
            "ingestion.run.aborted",
            reason="failure_ratio",
            failure_ratio=ratio,
            processed=counters.processed,
        )
        raise IngestionConfigurationError(
            f"Run aborted: {counters.failed} of {counters.processed} documents "
            f"failed, exceeding the configured guard of "
            f"{self._config.failure_ratio_threshold}.",
            context={
                "failure_ratio": ratio,
                "processed": counters.processed,
                "failed": counters.failed,
                "threshold": self._config.failure_ratio_threshold,
            },
        )

    def _record_metrics(self, outcome: IngestionOutcome) -> None:
        """
        Emit metrics for one completed outcome.

        Parameters
        ----------
        outcome:
            The completed outcome.

        Returns
        -------
        None
        """
        if isinstance(outcome, DocumentIngested):
            media_type = outcome.document.source.detected_media_type
            self._metrics.increment(_METRIC_INGESTED, media_type=media_type)
            self._metrics.observe(_METRIC_DURATION, outcome.duration_ms)
            self._metrics.observe(
                _METRIC_BYTES, float(outcome.document.source.size_bytes)
            )
        elif isinstance(outcome, DocumentSkipped):
            self._metrics.increment(_METRIC_SKIPPED, reason=outcome.reason.value)
        else:
            self._metrics.increment(
                _METRIC_FAILED,
                stage=outcome.failure.stage.value,
                error_code=outcome.failure.error_code,
            )

    # ------------------------------------------------------------------
    # Per-document pipeline
    # ------------------------------------------------------------------

    async def _process(
        self,
        record: SourceRecord,
        connector: SourceConnector,
        logger: StructuredLogger,
    ) -> IngestionOutcome:
        """
        Run one document through every stage.

        Every failure is captured as an outcome rather than raised, which is
        what isolates one bad document from the rest of a run. Cancellation is
        not captured, so an aborted run terminates promptly.

        Parameters
        ----------
        record:
            The record to process.
        connector:
            The connector to fetch from.
        logger:
            Logger bound to this run.

        Returns
        -------
        IngestionOutcome
            The document, a skip, or a failure record.
        """
        started = time.monotonic()
        document_logger = logger.bind(source_uri=record.source_uri)
        blob: Optional[AcquiredBlob] = None

        try:
            unchanged = await self._deduplicator.check_unchanged(record)
            if unchanged is not None:
                return unchanged

            blob = await self._acquisition.fetch(connector, record)

            duplicate = await self._deduplicator.check_duplicate(blob)
            if duplicate is not None:
                return duplicate

            document = await self._build_document(blob, record)
            await self._deduplicator.record(document)

            duration_ms = (time.monotonic() - started) * _MS_PER_SECOND

            if self._config.telemetry.log_document_events:
                document_logger.info(
                    "ingestion.document.ingested",
                    document_id=document.document_id,
                    media_type=document.source.detected_media_type,
                    char_count=document.metadata.char_count,
                    duration_ms=duration_ms,
                )

            return DocumentIngested(document=document, duration_ms=duration_ms)

        except PTFMIngestionError as error:
            document_logger.warning(
                "ingestion.document.failed",
                stage=error.stage.value,
                error_code=error.error_code,
                error_type=type(error).__name__,
                message=error.message,
            )
            return DocumentFailed(
                failure=IngestionFailure.from_exception(
                    error,
                    source_uri=record.source_uri,
                    failed_at=self._clock.now(),
                )
            )
        finally:
            # Released on every path. Without this, each failed or skipped
            # document would leak a spooled temporary file, which stays
            # invisible until the disk fills.
            if blob is not None:
                blob.close()

    async def _build_document(
        self,
        blob: AcquiredBlob,
        record: SourceRecord,
    ) -> RawDocument:
        """
        Transform an acquired payload into a canonical document.

        Detection, structure normalization, metadata harvesting, and canonical
        assembly are synchronous and CPU-bound, so each is executed off the
        event loop through the injected offloader. Extraction performs its own
        offloading, since it also owns the parse deadline.

        Parameters
        ----------
        blob:
            The acquired payload.
        record:
            The originating record.

        Returns
        -------
        RawDocument
            The canonical document.

        Raises
        ------
        PTFMIngestionError
            If any stage rejected the document.
        """
        header = await self._offloader.run(
            blob.header, self._config.detection.header_bytes
        )
        detected = await self._offloader.run(
            self._detector.detect,
            header,
            filename=record.filename,
            declared_media_type=record.declared_media_type,
        )

        extraction = await self._extraction.extract(blob, detected)

        structure = await self._offloader.run(
            self._structure.normalize, extraction
        )
        metadata = await self._offloader.run(
            self._metadata.harvest, extraction.metadata, structure.text
        )

        near_duplicate = await self._deduplicator.check_near_duplicate(
            structure.text
        )

        return await self._offloader.run(
            self._assemble,
            blob,
            detected,
            structure,
            metadata,
            # Carried through so that the canonical document records which
            # component derived its block tree. Resolution depends on registry
            # priority, exclusions, and dependency availability, so this is not
            # recoverable from the media type afterwards.
            extraction.extractor_name,
            extraction.warnings,
            near_duplicate,
        )

    def _assemble(
        self,
        blob: AcquiredBlob,
        detected: object,
        structure: object,
        metadata: object,
        extractor_name: str,
        extraction_warnings: object,
        near_duplicate: object,
    ) -> RawDocument:
        """
        Assemble the canonical document.

        A thin binding around the normalizer, present so that its
        keyword-only signature can be called through the offloader, which
        forwards positionally.

        Parameters
        ----------
        blob:
            The acquired payload.
        detected:
            The resolved format.
        structure:
            The normalized text and block tree.
        metadata:
            The harvested metadata.
        extractor_name:
            Identifier of the extractor that produced the block tree.
        extraction_warnings:
            Warnings reported by the extractor.
        near_duplicate:
            Near-duplicate observation, if any.

        Returns
        -------
        RawDocument
            The canonical document.
        """
        return self._normalizer.assemble(
            blob=blob,
            detected=detected,  # type: ignore[arg-type]
            structure=structure,  # type: ignore[arg-type]
            metadata=metadata,  # type: ignore[arg-type]
            extractor_name=extractor_name,
            extraction_warnings=extraction_warnings,  # type: ignore[arg-type]
            deduplication_warning=near_duplicate,  # type: ignore[arg-type]
        )

    def __repr__(self) -> str:
        """
        Return a representation naming the engine's concurrency settings.

        Returns
        -------
        str
            Diagnostic representation.
        """
        return (
            f"IngestionEngine(max_concurrent="
            f"{self._config.max_concurrent_documents}, "
            f"schemes={list(self._connectors.supported_schemes)!r})"
        )


# ============================================================================
# RUN STATE
# ============================================================================


class _RunCounters:
    """
    Mutable counters for one run.

    Deliberately mutable and deliberately run-scoped: an engine may execute
    several runs, so counters live per run rather than on the engine, which
    holds no mutable state of its own.
    """

    __slots__ = (
        "discovered",
        "ingested",
        "skipped",
        "failed",
        "total_bytes",
        "failures_by_stage",
        "failures_by_code",
        "by_media_type",
    )

    def __init__(self) -> None:
        self.discovered = 0
        self.ingested = 0
        self.skipped = 0
        self.failed = 0
        self.total_bytes = 0
        self.failures_by_stage: dict[str, int] = {}
        self.failures_by_code: dict[str, int] = {}
        self.by_media_type: dict[str, int] = {}

    @property
    def processed(self) -> int:
        """
        Number of documents that reached a terminal outcome.

        Returns
        -------
        int
            Sum of ingested, skipped, and failed counts.
        """
        return self.ingested + self.skipped + self.failed

    def record(self, outcome: IngestionOutcome) -> None:
        """
        Update the counters for one completed outcome.

        Parameters
        ----------
        outcome:
            The completed outcome.

        Returns
        -------
        None
        """
        if isinstance(outcome, DocumentIngested):
            self.ingested += 1
            self.total_bytes += outcome.document.source.size_bytes
            media_type = outcome.document.source.detected_media_type
            self.by_media_type[media_type] = (
                self.by_media_type.get(media_type, 0) + 1
            )
        elif isinstance(outcome, DocumentSkipped):
            self.skipped += 1
        else:
            self.failed += 1
            stage = outcome.failure.stage.value
            code = outcome.failure.error_code
            self.failures_by_stage[stage] = (
                self.failures_by_stage.get(stage, 0) + 1
            )
            self.failures_by_code[code] = self.failures_by_code.get(code, 0) + 1


async def _close_iterator(iterator: object) -> None:
    """
    Close an async iterator if it supports closing.

    A discovery generator abandoned mid-iteration holds whatever resources it
    had open. Closing it explicitly runs its cleanup rather than leaving that
    to garbage collection at an unpredictable later point.

    Parameters
    ----------
    iterator:
        The iterator to close.

    Returns
    -------
    None
    """
    close = getattr(iterator, "aclose", None)
    if close is None:
        return
    try:
        await close()
    except (RuntimeError, StopAsyncIteration):
        # Already closed, or closing from a different task than the one that
        # created it. Neither is a fault worth propagating during cleanup.
        return
