"""
PTFM Knowledge Processing Engine

File:
    ptfm/core/knowledge/engine.py

Description:
    Sequences the knowledge processing pipeline, managing concurrency, fault
    isolation, and cancellation.

What the engine does and does not do
------------------------------------
    It sequences stages and manages concurrency. It performs no segmentation,
    no derivation, no embedding, and no persistence: every collaborator
    arrives through the constructor as an interface, and no stage, strategy,
    provider, or sink is constructed here.

    It emits no score of any kind, and applies no truth judgement. The only
    threshold it consults is an operational guard over run health, injected
    from configuration.

Two planes, two batching regimes
--------------------------------
    Segmentation is deterministic and cheap; embedding is model-versioned and
    expensive, and wants batches. Reconciling those inside one streaming run
    is the engine's central problem.

    The resolution: a document's nodes are produced in full, then embedded in
    batches bounded by the provider, then emitted. Batching therefore happens
    WITHIN a document rather than across documents.

    Across-document batching would fill a batch faster on a corpus of small
    documents, but it would also mean holding an unbounded number of partially
    completed documents while their nodes waited for batch partners, and a
    document could not be validated until nodes belonging to other documents
    had been embedded. Validation is per-document, so its unit must be the
    document. The chosen bound is therefore concurrency multiplied by document
    size, which is predictable, rather than a quantity that grows with how
    small the documents happen to be.

Ordering
--------
    Outcomes arrive in completion order rather than input order, since
    documents are processed concurrently and vary by orders of magnitude in
    cost. Preserving input order would require buffering completed documents
    behind slow ones, defeating the memory bound streaming exists to provide.

    Within a document, nodes are emitted in reading order.

Fault isolation
---------------
    A failure in one document produces a failure outcome and the run
    continues. Two conditions abort: a configuration or sink fault, which
    would fail identically for every subsequent document, and a failure ratio
    exceeding the configured guard once enough documents have been processed
    for the ratio to mean anything.

Dependency contract
-------------------
    Imports the knowledge ``exceptions``, ``config``, ``models``,
    ``interfaces`` and stage modules, plus ``ptfm.core.schemas``. It imports
    no strategy, provider, or sink implementation.
"""

from __future__ import annotations

import asyncio
import time
from typing import AsyncIterator, Final, Optional, Sequence

from ptfm.core.knowledge.config import KnowledgeConfig
from ptfm.core.knowledge.exceptions import (
    KnowledgeConfigurationError,
    KnowledgeStage,
    PTFMKnowledgeError,
    SinkError,
)
from ptfm.core.knowledge.interfaces import (
    Clock,
    IdGenerator,
    KnowledgeSink,
    MetricsSink,
    StructuredLogger,
    ThreadOffloader,
)
from ptfm.core.knowledge.models import (
    DocumentFailed,
    DocumentSkipped,
    KnowledgeUnit,
    NodeProduced,
    ProcessingFailure,
    ProcessingOutcome,
    ProcessingReport,
    SkipReason,
)
from ptfm.core.knowledge.pipeline.assembly import NodeAssembler
from ptfm.core.knowledge.pipeline.embedding import EmbeddingStage
from ptfm.core.knowledge.pipeline.enrichment import EnrichmentStage
from ptfm.core.knowledge.pipeline.planning import ChunkPlanner
from ptfm.core.knowledge.pipeline.relations import RelationDeriver
from ptfm.core.knowledge.pipeline.validation import ProvenanceValidator
from ptfm.core.schemas import KnowledgeNode, NodeEmbedding, RawDocument

__all__ = ["KnowledgeEngine"]

_METRIC_DOCUMENTS: Final[str] = "ptfm.knowledge.documents.processed"
_METRIC_SKIPPED: Final[str] = "ptfm.knowledge.documents.skipped"
_METRIC_FAILED: Final[str] = "ptfm.knowledge.documents.failed"
_METRIC_NODES: Final[str] = "ptfm.knowledge.nodes.produced"
_METRIC_EMBEDDINGS: Final[str] = "ptfm.knowledge.embeddings.produced"
_METRIC_RELATIONS: Final[str] = "ptfm.knowledge.relations.produced"
_METRIC_DURATION: Final[str] = "ptfm.knowledge.document.duration_ms"
_METRIC_NODE_CHARS: Final[str] = "ptfm.knowledge.node.chars"

_MS_PER_SECOND: Final[float] = 1000.0


class KnowledgeEngine:
    """
    Executes the knowledge processing pipeline over a stream of documents.

    Every collaborator is injected. The engine constructs nothing and holds no
    mutable state, so two engines with different configurations can run
    concurrently in one process without interfering.

    Parameters
    ----------
    config:
        Complete configuration. Supplies concurrency limits, the failure
        guard, and sink behaviour.
    segmentation_fingerprint:
        The fingerprint under which this engine segments. Resolved once at the
        composition root, together with the strategy and segmenter it
        describes, and carried on every node. Passing it in rather than
        computing it here means the engine cannot produce nodes whose
        fingerprint disagrees with the components that made them.
    planner:
        Produces and verifies a chunk plan.
    assembler:
        Materializes a plan into nodes.
    relation_deriver:
        Derives structural relations.
    enrichment:
        Applies domain metadata.
    embedding:
        Produces vectors. ``None`` when embedding is disabled, in which case
        nodes are emitted without vectors and the corpus can be embedded
        later.
    validator:
        Verifies the complete unit set before it leaves the module.
    sink:
        Receives the units. ``None`` when the caller consumes the stream
        directly.
    offloader:
        Executes blocking stage work off the event loop.
    clock:
        Supplies run timestamps.
    id_generator:
        Supplies the run correlation identifier. Never used for node identity,
        which is a deterministic derivation.
    logger:
        Structured logger.
    metrics:
        Metrics sink.
    """

    __slots__ = (
        "_config",
        "_fingerprint",
        "_planner",
        "_assembler",
        "_relations",
        "_enrichment",
        "_embedding",
        "_validator",
        "_sink",
        "_offloader",
        "_clock",
        "_id_generator",
        "_logger",
        "_metrics",
    )

    def __init__(
        self,
        *,
        config: KnowledgeConfig,
        segmentation_fingerprint: str,
        planner: ChunkPlanner,
        assembler: NodeAssembler,
        relation_deriver: RelationDeriver,
        enrichment: EnrichmentStage,
        embedding: Optional[EmbeddingStage],
        validator: ProvenanceValidator,
        sink: Optional[KnowledgeSink],
        offloader: ThreadOffloader,
        clock: Clock,
        id_generator: IdGenerator,
        logger: StructuredLogger,
        metrics: MetricsSink,
    ) -> None:
        self._config = config
        self._fingerprint = segmentation_fingerprint
        self._planner = planner
        self._assembler = assembler
        self._relations = relation_deriver
        self._enrichment = enrichment
        self._embedding = embedding
        self._validator = validator
        self._sink = sink
        self._offloader = offloader
        self._clock = clock
        self._id_generator = id_generator
        self._logger = logger
        self._metrics = metrics

    # ------------------------------------------------------------------
    # Public surface
    # ------------------------------------------------------------------

    @property
    def segmentation_fingerprint(self) -> str:
        """
        The fingerprint under which this engine segments.

        Exposed so that a caller can detect, before a run, that an existing
        corpus was produced under different segmentation and would therefore
        occupy a disjoint identifier space.

        Returns
        -------
        str
            The segmentation fingerprint.
        """
        return self._fingerprint

    @property
    def embedding_model_identity(self) -> Optional[str]:
        """
        Identity of the embedding model, when embedding is enabled.

        Returns
        -------
        Optional[str]
            The model identity, or ``None`` when embedding is disabled.
        """
        if self._embedding is None:
            return None
        return self._embedding.model_identity.identity

    async def process(
        self,
        documents: AsyncIterator[RawDocument],
    ) -> AsyncIterator[ProcessingOutcome]:
        """
        Process a stream of documents, yielding one outcome per node or event.

        Outcomes arrive in completion order rather than input order, since
        documents are processed concurrently and vary widely in cost. Within a
        document, nodes are emitted in reading order.

        Implemented as an async generator: nothing happens until the caller
        iterates, and abandoning the iterator cancels the run and releases
        every resource it held.

        Parameters
        ----------
        documents:
            The source documents.

        Yields
        ------
        ProcessingOutcome
            A produced node, a skipped document, or a failure record.

        Raises
        ------
        KnowledgeConfigurationError
            If the failure guard aborted the run.
        SinkError
            If a sink failed and the configuration treats that as fatal.
        """
        run_id = self._id_generator.new_id()
        logger = self._logger.bind(run_id=run_id)
        started = self._clock.now()
        counters = _RunCounters()

        logger.info(
            "knowledge.run.started",
            segmentation_fingerprint=self._fingerprint,
            embedding_model=self.embedding_model_identity,
            max_concurrent=self._config.max_concurrent_documents,
        )

        try:
            async for outcome in self._run(documents, counters, logger):
                yield outcome
        finally:
            # Emitted on every exit path, including cancellation, so that an
            # abandoned run still reports what it accomplished.
            logger.info(
                "knowledge.run.finished",
                documents=counters.documents_seen,
                nodes=counters.nodes,
                embeddings=counters.embeddings,
                failed=counters.failed,
                duration_seconds=(self._clock.now() - started).total_seconds(),
            )

    async def process_all(
        self,
        documents: AsyncIterator[RawDocument],
    ) -> ProcessingReport:
        """
        Process a stream to completion, returning a run report.

        Intended for operational runs where units are persisted by an injected
        sink. A caller needing the units themselves should iterate
        :meth:`process`.

        Parameters
        ----------
        documents:
            The source documents.

        Returns
        -------
        ProcessingReport
            Run-level aggregate counts and diagnostics.

        Raises
        ------
        PTFMKnowledgeError
            If the run aborted. No report is produced, since an aborted run's
            counts describe an incomplete traversal.
        """
        run_id = self._id_generator.new_id()
        started = self._clock.now()
        counters = _RunCounters()

        async for outcome in self.process(documents):
            counters.record(outcome)

        return ProcessingReport(
            run_id=run_id,
            segmentation_fingerprint=self._fingerprint,
            embedding_model_identity=self.embedding_model_identity,
            started_at=started,
            completed_at=self._clock.now(),
            documents_processed=counters.processed,
            documents_skipped=counters.skipped,
            documents_failed=counters.failed,
            nodes_produced=counters.nodes,
            embeddings_produced=counters.embeddings,
            relations_produced=counters.relations,
            total_node_chars=counters.node_chars,
            nodes_by_type=dict(counters.by_type),
            skips_by_reason=dict(counters.by_skip_reason),
            failures_by_stage=dict(counters.by_stage),
            failures_by_code=dict(counters.by_code),
        )

    async def process_document(
        self,
        document: RawDocument,
    ) -> tuple[ProcessingOutcome, ...]:
        """
        Process one document, returning its outcomes.

        Provided for embedded use, such as a service handling one upload.
        Document-level failures are returned rather than raised, so that this
        method behaves consistently with the streaming surface.

        Parameters
        ----------
        document:
            The document to process.

        Returns
        -------
        tuple[ProcessingOutcome, ...]
            One outcome per node, or a single skip or failure.
        """
        return tuple(await self._process(document, self._logger))

    # ------------------------------------------------------------------
    # Run orchestration
    # ------------------------------------------------------------------

    async def _run(
        self,
        documents: AsyncIterator[RawDocument],
        counters: _RunCounters,
        logger: StructuredLogger,
    ) -> AsyncIterator[ProcessingOutcome]:
        """
        Process documents with bounded concurrency.

        A fixed number of documents are kept in flight. As each completes, its
        outcomes are yielded and another document is drawn from the source, so
        memory is bounded by the concurrency limit rather than by corpus size,
        and a slow consumer naturally slows intake.

        Parameters
        ----------
        documents:
            The source documents.
        counters:
            Run counters, updated as documents complete.
        logger:
            Logger bound to this run.

        Yields
        ------
        ProcessingOutcome
            Outcomes in completion order.

        Raises
        ------
        KnowledgeConfigurationError
            If the failure guard aborted the run.
        """
        limit = self._config.max_concurrent_documents
        source = documents.__aiter__()
        in_flight: set[asyncio.Task[Sequence[ProcessingOutcome]]] = set()
        exhausted = False
        since_flush = 0

        try:
            while True:
                while not exhausted and len(in_flight) < limit:
                    try:
                        document = await source.__anext__()
                    except StopAsyncIteration:
                        exhausted = True
                        break
                    in_flight.add(
                        asyncio.create_task(self._process(document, logger))
                    )

                if not in_flight:
                    break

                done, pending = await asyncio.wait(
                    in_flight, return_when=asyncio.FIRST_COMPLETED
                )
                in_flight = set(pending)

                for task in done:
                    for outcome in task.result():
                        counters.record(outcome)
                        self._record_metrics(outcome)
                        yield outcome

                    since_flush += 1
                    if since_flush >= self._config.sink.flush_interval_units:
                        await self._flush(logger)
                        since_flush = 0

                    self._check_failure_guard(counters, logger)

            await self._flush(logger)

        finally:
            # Reached on completion, on abort, and when the consumer abandons
            # the iterator. Every in-flight document is cancelled so that no
            # task outlives the run.
            for task in in_flight:
                task.cancel()
            if in_flight:
                await asyncio.gather(*in_flight, return_exceptions=True)
            await _close_iterator(source)

    async def _flush(self, logger: StructuredLogger) -> None:
        """
        Commit buffered sink writes.

        Parameters
        ----------
        logger:
            Logger bound to this run.

        Returns
        -------
        None

        Raises
        ------
        SinkError
            If the flush failed and sink failures are fatal.
        """
        if self._sink is None:
            return

        try:
            await self._sink.flush()
        except SinkError as error:
            if self._config.sink.fail_run_on_sink_error:
                raise
            logger.warning(
                "knowledge.sink.flush_failed",
                sink_name=self._sink.name,
                error_code=error.error_code,
                message=error.message,
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

        An operational guard over run health. It carries no truth judgement
        and never influences verification.

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
        KnowledgeConfigurationError
            If the failure ratio exceeds the configured guard.
        """
        if self._config.fail_fast and counters.failed > 0:
            logger.error("knowledge.run.aborted", reason="fail_fast")
            raise KnowledgeConfigurationError(
                "Run aborted after the first document failure because "
                "fail_fast is enabled.",
                context={"failed": counters.failed},
            )

        seen = counters.documents_seen
        if seen < self._config.failure_ratio_min_samples:
            return

        ratio = counters.failed / seen
        if ratio <= self._config.failure_ratio_threshold:
            return

        logger.error(
            "knowledge.run.aborted", reason="failure_ratio", failure_ratio=ratio
        )
        raise KnowledgeConfigurationError(
            f"Run aborted: {counters.failed} of {seen} documents failed, "
            f"exceeding the configured guard of "
            f"{self._config.failure_ratio_threshold}.",
            context={
                "failure_ratio": ratio,
                "documents_seen": seen,
                "failed": counters.failed,
            },
        )

    def _record_metrics(self, outcome: ProcessingOutcome) -> None:
        """
        Emit metrics for one outcome.

        Parameters
        ----------
        outcome:
            The completed outcome.

        Returns
        -------
        None
        """
        if isinstance(outcome, NodeProduced):
            node = outcome.unit.node
            self._metrics.increment(
                _METRIC_NODES, node_type=node.node_type.value
            )
            self._metrics.observe(_METRIC_NODE_CHARS, float(node.char_count))
            self._metrics.observe(_METRIC_DURATION, outcome.duration_ms)
            if outcome.unit.is_embedded:
                self._metrics.increment(_METRIC_EMBEDDINGS)
            if outcome.unit.relations:
                self._metrics.increment(
                    _METRIC_RELATIONS, len(outcome.unit.relations)
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
        document: RawDocument,
        logger: StructuredLogger,
    ) -> Sequence[ProcessingOutcome]:
        """
        Run one document through every stage.

        Every failure is captured as an outcome rather than raised, which is
        what isolates one bad document from the rest of a run. Two exceptions
        propagate deliberately: cancellation, so an aborted run terminates
        promptly, and a fatal sink failure, which would fail identically for
        every subsequent document.

        Parameters
        ----------
        document:
            The document to process.
        logger:
            Logger bound to this run.

        Returns
        -------
        Sequence[ProcessingOutcome]
            One outcome per node, or a single skip or failure.

        Raises
        ------
        SinkError
            If a sink failed and the configuration treats that as fatal.
        """
        started = time.monotonic()
        document_logger = logger.bind(document_id=document.document_id)

        try:
            units = await self._build_units(document)
        except PTFMKnowledgeError as error:
            document_logger.warning(
                "knowledge.document.failed",
                stage=error.stage.value,
                error_code=error.error_code,
                error_type=type(error).__name__,
                message=error.message,
            )
            return (
                DocumentFailed(
                    failure=ProcessingFailure.from_exception(
                        error,
                        document_id=document.document_id,
                        failed_at=self._clock.now(),
                    )
                ),
            )

        if not units:
            self._metrics.increment(_METRIC_DOCUMENTS)
            return (
                DocumentSkipped(
                    document_id=document.document_id,
                    reason=SkipReason.NO_EXTRACTABLE_CONTENT,
                ),
            )

        # Written before emission, so that a fatal sink failure aborts the run
        # rather than being discovered after the consumer has acted on
        # outcomes nothing received.
        await self._write(units, document_logger)

        self._metrics.increment(_METRIC_DOCUMENTS)
        duration_ms = (time.monotonic() - started) * _MS_PER_SECOND
        per_node = duration_ms / len(units)

        if self._config.telemetry.log_node_events:
            document_logger.info(
                "knowledge.document.processed",
                nodes=len(units),
                duration_ms=duration_ms,
            )

        return tuple(
            NodeProduced(unit=unit, duration_ms=per_node) for unit in units
        )

    async def _build_units(
        self,
        document: RawDocument,
    ) -> tuple[KnowledgeUnit, ...]:
        """
        Run the structural and semantic planes for one document.

        Stage order is not arbitrary. Planning precedes assembly because a
        plan is verified before any node exists. Relations are derived from
        assembled nodes. Enrichment follows assembly because an enricher sees
        a complete node, and precedes embedding because composition may
        eventually draw on metadata. Embedding follows enrichment for that
        reason and precedes validation, so that validation sees the complete
        unit set including vectors, which is the only point at which
        cross-unit embedding comparability can be checked.

        Parameters
        ----------
        document:
            The document to process.

        Returns
        -------
        tuple[KnowledgeUnit, ...]
            The document's units, in reading order. Empty for a document with
            no extractable content.

        Raises
        ------
        PTFMKnowledgeError
            If any stage rejected the document.
        """
        plan = await self._offloader.run(self._planner.plan, document)
        if plan.is_empty:
            return ()

        nodes = await self._offloader.run(
            self._assembler.assemble, document, plan, self._fingerprint
        )

        enriched = await self._enrich(nodes)

        relations = await self._offloader.run(
            self._relations.derive, document, enriched
        )
        by_source: dict[str, list[object]] = {}
        for relation in relations:
            by_source.setdefault(relation.source_node_id, []).append(relation)

        embeddings = await self._embed(enriched)

        units = tuple(
            KnowledgeUnit(
                node=node,
                embedding=embeddings.get(node.node_id),
                relations=tuple(by_source.get(node.node_id, ())),  # type: ignore[arg-type]
            )
            for node in enriched
        )

        await self._offloader.run(self._validator.validate, document, units)
        return units

    async def _enrich(
        self,
        nodes: Sequence[KnowledgeNode],
    ) -> tuple[KnowledgeNode, ...]:
        """
        Enrich a document's nodes concurrently, under a bound.

        Enrichment is per-node independent and ordinarily crosses a network or
        database boundary. Running it one node at a time would serialize a
        large document's round trips inside a single document slot while the
        run's concurrency limit held other documents back, which is a
        throughput cliff that appears only once a real enricher is attached.

        Bounded rather than unlimited, because an enricher's backing service
        has its own capacity and a thousand-node document should not present a
        thousand simultaneous requests to it.

        Results are reassembled in reading order regardless of completion
        order, since a node's ordinal and its position in the emitted sequence
        must agree.

        Parameters
        ----------
        nodes:
            The document's nodes, in reading order.

        Returns
        -------
        tuple[KnowledgeNode, ...]
            The enriched nodes, in reading order.

        Raises
        ------
        EnrichmentError
            If an enricher failed and the configured policy is fatal.
        """
        limit = asyncio.Semaphore(self._config.enrichment.max_concurrent_nodes)

        async def _one(node: KnowledgeNode) -> KnowledgeNode:
            async with limit:
                return await self._enrichment.enrich(node)

        return tuple(await asyncio.gather(*(_one(node) for node in nodes)))

    async def _embed(
        self,
        nodes: Sequence[KnowledgeNode],
    ) -> dict[str, NodeEmbedding]:
        """
        Produce vectors for a document's nodes, in provider-sized batches.

        Batching is bounded by the provider's own maximum, since a provider
        knows its limits and exceeding them fails every batch rather than
        merely being inefficient.

        Batches are formed within a document. Filling them across documents
        would hold an unbounded number of partially completed documents while
        their nodes waited for batch partners, and would prevent per-document
        validation until unrelated documents had been embedded.

        Parameters
        ----------
        nodes:
            The document's nodes.

        Returns
        -------
        dict[str, NodeEmbedding]
            Embedding keyed by node identifier, empty when embedding is
            disabled.

        Raises
        ------
        EmbeddingError
            If the provider failed after every permitted attempt.
        """
        if self._embedding is None:
            return {}

        produced: dict[str, NodeEmbedding] = {}
        size = self._embedding.max_batch_size

        for start in range(0, len(nodes), size):
            batch = nodes[start : start + size]
            for embedding in await self._embedding.embed(batch):
                produced[embedding.node_id] = embedding

        return produced

    async def _write(
        self,
        units: Sequence[KnowledgeUnit],
        logger: StructuredLogger,
    ) -> None:
        """
        Write a document's units to the sink.

        A sink failure is treated differently from a document failure by
        default. Unlike a malformed document, an unavailable sink will fail
        identically for every subsequent document, so continuing would produce
        a long run that discards everything it computes.

        Parameters
        ----------
        units:
            The units to write.
        logger:
            Logger bound to this document.

        Returns
        -------
        None

        Raises
        ------
        SinkError
            If the write failed and sink failures are fatal.
        """
        if self._sink is None:
            return

        try:
            for unit in units:
                await self._sink.write(unit)
        except SinkError as error:
            if self._config.sink.fail_run_on_sink_error:
                raise
            logger.warning(
                "knowledge.sink.write_failed",
                sink_name=self._sink.name,
                error_code=error.error_code,
                message=error.message,
            )

    def __repr__(self) -> str:
        """
        Return a representation naming the engine's segmentation and model.

        Returns
        -------
        str
            Diagnostic representation.
        """
        return (
            f"KnowledgeEngine(fingerprint={self._fingerprint!r}, "
            f"model={self.embedding_model_identity!r}, "
            f"max_concurrent={self._config.max_concurrent_documents})"
        )


# ============================================================================
# RUN STATE
# ============================================================================


class _RunCounters:
    """
    Mutable counters for one run.

    Deliberately run-scoped rather than held on the engine, which has no
    mutable state of its own so that several runs may share one engine.
    """

    __slots__ = (
        "processed",
        "skipped",
        "failed",
        "nodes",
        "embeddings",
        "relations",
        "node_chars",
        "by_type",
        "by_skip_reason",
        "by_stage",
        "by_code",
        "_documents",
    )

    def __init__(self) -> None:
        self.processed = 0
        self.skipped = 0
        self.failed = 0
        self.nodes = 0
        self.embeddings = 0
        self.relations = 0
        self.node_chars = 0
        self.by_type: dict[str, int] = {}
        self.by_skip_reason: dict[str, int] = {}
        self.by_stage: dict[str, int] = {}
        self.by_code: dict[str, int] = {}
        self._documents: set[str] = set()

    @property
    def documents_seen(self) -> int:
        """
        Number of documents that reached a terminal outcome.

        Counted by identifier rather than by outcome, because one document
        yields many node outcomes and counting those would understate the
        failure ratio by an arbitrary factor.

        Returns
        -------
        int
            Distinct document count.
        """
        return len(self._documents)

    def record(self, outcome: ProcessingOutcome) -> None:
        """
        Update the counters for one outcome.

        Parameters
        ----------
        outcome:
            The outcome to record.

        Returns
        -------
        None
        """
        if isinstance(outcome, NodeProduced):
            node = outcome.unit.node
            if node.document_id not in self._documents:
                self._documents.add(node.document_id)
                self.processed += 1
            self.nodes += 1
            self.node_chars += node.char_count
            self.by_type[node.node_type.value] = (
                self.by_type.get(node.node_type.value, 0) + 1
            )
            if outcome.unit.is_embedded:
                self.embeddings += 1
            self.relations += len(outcome.unit.relations)
        elif isinstance(outcome, DocumentSkipped):
            self._documents.add(outcome.document_id)
            self.skipped += 1
            reason = outcome.reason.value
            self.by_skip_reason[reason] = self.by_skip_reason.get(reason, 0) + 1
        else:
            self._documents.add(outcome.document_id)
            self.failed += 1
            stage = outcome.failure.stage.value
            code = outcome.failure.error_code
            self.by_stage[stage] = self.by_stage.get(stage, 0) + 1
            self.by_code[code] = self.by_code.get(code, 0) + 1


async def _close_iterator(iterator: object) -> None:
    """
    Close an async iterator if it supports closing.

    A source abandoned mid-iteration holds whatever resources it had open.
    Closing it explicitly runs its cleanup rather than leaving that to garbage
    collection at an unpredictable later point.

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
        return
