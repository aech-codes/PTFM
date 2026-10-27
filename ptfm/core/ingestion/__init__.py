"""
PTFM Data Ingestion Module

File:
    ptfm/core/ingestion/__init__.py

Description:
    Public interface of the Data Ingestion Module, the first stage of the PTFM
    pipeline.

    Documents are acquired from heterogeneous sources, their formats detected,
    their content extracted, their structure normalized, and their metadata
    harvested, producing the canonical ``RawDocument`` that Knowledge
    Processing consumes.

Composition root
----------------
    :func:`build_default_engine` is the single place in the module where
    concrete implementations are chosen and wired together. Every other module
    depends on interfaces alone, which is what allows any component to be
    replaced without editing the pipeline.

    Nothing in this package constructs its own dependencies, and no module
    holds mutable global state, so several engines with different
    configurations can run concurrently in one process without interfering.

Usage
-----
    ::

        from ptfm.core.ingestion import build_default_engine, SourceSpec

        engine = build_default_engine()
        spec = SourceSpec(
            root_uri="file:///corpus",
            include_patterns=("*.pdf", "*.docx"),
        )

        async for outcome in engine.ingest(spec):
            if outcome.kind == "INGESTED":
                consume(outcome.document)

Scope
-----
    This module performs no chunking, embedding, indexing, or domain
    enrichment, and produces no confidence value of any kind. Those belong to
    Knowledge Processing, to Retrieval, and to domain plugins. ``RawDocument``
    is the contract boundary between them.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Iterable, Optional
from uuid import uuid4

from ptfm.core.ingestion.config import (
    AcquisitionConfig,
    DeduplicationConfig,
    DetectionConfig,
    ExtractionConfig,
    IngestionConfig,
    MetadataConfig,
    RetryConfig,
    StructureConfig,
    TelemetryConfig,
    TextNormalizationForm,
)
from ptfm.core.ingestion.connectors.filesystem import LocalFilesystemConnector
from ptfm.core.ingestion.connectors.memory import (
    InMemoryConnector,
    MemoryDocument,
)
from ptfm.core.ingestion.engine import IngestionEngine
from ptfm.core.ingestion.exceptions import (
    IngestionStage,
    PTFMIngestionError,
)
from ptfm.core.ingestion.extractors.docx import DocxExtractor
from ptfm.core.ingestion.extractors.html import HtmlExtractor
from ptfm.core.ingestion.extractors.markdown import MarkdownExtractor
from ptfm.core.ingestion.extractors.pdf import PdfExtractor
from ptfm.core.ingestion.extractors.plaintext import PlainTextExtractor
from ptfm.core.ingestion.extractors.tabular import TabularExtractor
from ptfm.core.ingestion.interfaces import (
    Clock,
    ContentExtractor,
    DocumentFingerprintStore,
    FormatDetector,
    HashProvider,
    LanguageDetector,
    MetricsSink,
    SourceConnector,
    StructuredLogger,
    ThreadOffloader,
)
from ptfm.core.ingestion.models import (
    AcquiredBlob,
    DetectedFormat,
    DocumentFailed,
    DocumentIngested,
    DocumentSkipped,
    ExtractedBlock,
    ExtractedMetadata,
    ExtractionInput,
    ExtractionOutput,
    IngestionFailure,
    IngestionOutcome,
    IngestionReport,
    SkipReason,
    SourceRecord,
    SourceSpec,
)
from ptfm.core.ingestion.pipeline.acquisition import AcquisitionStage
from ptfm.core.ingestion.pipeline.deduplication import (
    Deduplicator,
    InMemoryFingerprintStore,
)
from ptfm.core.ingestion.pipeline.detection import DefaultFormatDetector
from ptfm.core.ingestion.pipeline.extraction import ExtractionStage
from ptfm.core.ingestion.pipeline.metadata import MetadataHarvester
from ptfm.core.ingestion.pipeline.normalization import RawDocumentNormalizer
from ptfm.core.ingestion.pipeline.structure import StructureNormalizer
from ptfm.core.ingestion.registry import (
    ConnectorRegistration,
    ConnectorRegistry,
    ExtractorRegistry,
)
from ptfm.core.ingestion.telemetry import build_logger, build_metrics_sink
from ptfm.core.ingestion.utils.concurrency import (
    BoundedThreadOffloader,
    InlineOffloader,
)
from ptfm.core.ingestion.utils.hashing import Blake2bHashProvider

__all__ = [
    # Engine and composition
    "IngestionEngine",
    "build_default_engine",
    "default_connectors",
    "default_extractors",
    # Specification and outcomes
    "SourceSpec",
    "SourceRecord",
    "IngestionOutcome",
    "DocumentIngested",
    "DocumentSkipped",
    "DocumentFailed",
    "SkipReason",
    "IngestionFailure",
    "IngestionReport",
    "IngestionStage",
    "PTFMIngestionError",
    # Configuration
    "IngestionConfig",
    "AcquisitionConfig",
    "DetectionConfig",
    "ExtractionConfig",
    "StructureConfig",
    "MetadataConfig",
    "DeduplicationConfig",
    "RetryConfig",
    "TelemetryConfig",
    "TextNormalizationForm",
    # Contracts
    "SourceConnector",
    "ContentExtractor",
    "FormatDetector",
    "LanguageDetector",
    "DocumentFingerprintStore",
    "HashProvider",
    "ThreadOffloader",
    "Clock",
    "MetricsSink",
    "StructuredLogger",
    # Extension
    "ConnectorRegistry",
    "ConnectorRegistration",
    "ExtractorRegistry",
    "MemoryDocument",
    "InMemoryConnector",
    "LocalFilesystemConnector",
    "InMemoryFingerprintStore",
    "BoundedThreadOffloader",
    "InlineOffloader",
    "Blake2bHashProvider",
    # Extractor-facing types
    "AcquiredBlob",
    "DetectedFormat",
    "ExtractionInput",
    "ExtractionOutput",
    "ExtractedBlock",
    "ExtractedMetadata",
]


class _SystemClock:
    """
    Clock reading the system time.

    Satisfies the ``Clock`` protocol. Returns timezone-aware instants, since a
    naive one cannot be compared against the timestamps used elsewhere in the
    pipeline.
    """

    __slots__ = ()

    def now(self) -> datetime:
        """
        Return the current instant.

        Returns
        -------
        datetime
            The current instant in UTC.
        """
        return datetime.now(UTC)


class _UuidGenerator:
    """
    Correlation identifier generator.

    Satisfies the ``IdGenerator`` protocol. Used solely for run and trace
    identifiers. Document identity is a deterministic derivation from source
    URI and content digest, never a generated value, which is what keeps
    ingestion idempotent and evidence citations stable across index rebuilds.
    """

    __slots__ = ()

    def new_id(self) -> str:
        """
        Generate a fresh correlation identifier.

        Returns
        -------
        str
            A unique identifier.
        """
        return str(uuid4())


def default_connectors() -> tuple[ConnectorRegistration, ...]:
    """
    Return the connector registrations shipped with this phase.

    Declared as a function rather than a constant so that each caller receives
    an independent tuple and no shared registration object can be mutated
    across engines.

    Returns
    -------
    tuple[ConnectorRegistration, ...]
        Registrations for the local filesystem and in-memory connectors.
    """
    return (
        ConnectorRegistration(
            scheme="file",
            factory=LocalFilesystemConnector,
            origin="builtin",
        ),
        ConnectorRegistration(
            scheme="memory",
            factory=InMemoryConnector,
            origin="builtin",
        ),
    )


def default_extractors() -> tuple[ContentExtractor, ...]:
    """
    Return the extractors shipped with this phase.

    Ordering carries no meaning: the registry resolves by media type
    specificity and declared priority, never by registration order.

    The plain text extractor appears twice. It claims ``text/plain`` at
    ordinary priority, and additionally claims ``text/*`` at negative
    priority, which makes it the fallback for any text format whose dedicated
    extractor is not installed. Negative priority guarantees the fallback can
    never displace a dedicated extractor.

    Returns
    -------
    tuple[ContentExtractor, ...]
        The built-in extractors.
    """
    return (
        PlainTextExtractor(),
        MarkdownExtractor(),
        HtmlExtractor(),
        TabularExtractor(),
        PdfExtractor(),
        DocxExtractor(),
        PlainTextExtractor(["text/*"], priority=-100),
    )


def build_default_engine(
    config: Optional[IngestionConfig] = None,
    *,
    connectors: Optional[Iterable[ConnectorRegistration]] = None,
    extractors: Optional[Iterable[ContentExtractor]] = None,
    fingerprint_store: Optional[DocumentFingerprintStore] = None,
    language_detector: Optional[LanguageDetector] = None,
    hash_provider: Optional[HashProvider] = None,
    offloader: Optional[ThreadOffloader] = None,
    clock: Optional[Clock] = None,
    logger: Optional[StructuredLogger] = None,
    metrics: Optional[MetricsSink] = None,
    discover_entry_points: bool = True,
) -> IngestionEngine:
    """
    Construct an ingestion engine with the default implementations.

    The single composition root of the module. Every concrete class is chosen
    here and injected downward, so that no stage constructs its own
    dependencies and any component can be replaced by passing a substitute.

    A shared thread offloader is constructed unless one is supplied, so that
    the configured limit bounds the process as a whole. Passing an
    ``InlineOffloader`` is supported for tests and embedded single-document
    use, and must not be used for corpus ingestion, since every parse would
    then execute on the event loop.

    The fingerprint store defaults to an in-memory implementation, which does
    not persist between runs. A second run over an unchanged corpus therefore
    rediscovers every document as new. Cross-run incremental ingestion
    requires a persistent store, which also closes the concurrent
    check-then-record window that an in-memory store cannot.

    Parameters
    ----------
    config:
        Ingestion configuration. Constructed from the environment when
        omitted.
    connectors:
        Connector registrations, replacing the built-in set entirely when
        supplied.
    extractors:
        Extractors, replacing the built-in set entirely when supplied.
    fingerprint_store:
        Duplicate detection store.
    language_detector:
        Natural language identifier. Detection is skipped when omitted, so a
        deployment can leave the dependency uninstalled.
    hash_provider:
        Content fingerprint provider. The same provider is given to
        acquisition and to canonical assembly, since digests computed under
        differing algorithms are incomparable.
    offloader:
        Executor for blocking work.
    clock:
        Time source.
    logger:
        Structured logger.
    metrics:
        Metrics sink.
    discover_entry_points:
        Whether separately installed packages may contribute connectors and
        extractors through entry points.

    Returns
    -------
    IngestionEngine
        A fully wired engine.

    Raises
    ------
    IngestionConfigurationError
        If a registration is invalid or two components collide.
    ExtractorUnavailableError
        If an extractor's optional dependency is absent and strict
        availability is in force.
    """
    resolved_config = config if config is not None else IngestionConfig()
    resolved_clock: Clock = clock if clock is not None else _SystemClock()
    resolved_hash: HashProvider = (
        hash_provider if hash_provider is not None else Blake2bHashProvider()
    )
    resolved_offloader: ThreadOffloader = (
        offloader
        if offloader is not None
        else BoundedThreadOffloader(resolved_config.resolve_thread_limit())
    )

    connector_registrations = (
        tuple(connectors) if connectors is not None else default_connectors()
    )
    extractor_instances = (
        tuple(extractors) if extractors is not None else default_extractors()
    )

    if discover_entry_points:
        connector_registry = ConnectorRegistry.from_entry_points(
            additional=connector_registrations
        )
        extractor_registry = ExtractorRegistry.from_entry_points(
            config=resolved_config.extraction, additional=extractor_instances
        )
    else:
        connector_registry = ConnectorRegistry(connector_registrations)
        extractor_registry = ExtractorRegistry(
            extractor_instances, config=resolved_config.extraction
        )

    resolved_logger = build_logger(
        resolved_config.telemetry.log_level, logger=logger
    )
    resolved_metrics = build_metrics_sink(
        resolved_config.telemetry.emit_metrics, sink=metrics
    )

    for skipped in extractor_registry.skipped_extractors:
        # Reported rather than silently dropped, so that an operator learns at
        # startup which formats this deployment cannot read.
        resolved_logger.warning(
            "ingestion.extractor.skipped",
            extractor_name=skipped.extractor_name,
            reason=skipped.reason,
        )

    store = (
        fingerprint_store
        if fingerprint_store is not None
        else InMemoryFingerprintStore()
    )

    return IngestionEngine(
        config=resolved_config,
        connector_registry=connector_registry,
        acquisition=AcquisitionStage.from_config(
            resolved_config, hash_provider=resolved_hash
        ),
        detector=DefaultFormatDetector(resolved_config.detection),
        extraction=ExtractionStage(
            extractor_registry, resolved_offloader, resolved_config.extraction
        ),
        structure=StructureNormalizer(
            resolved_config.structure, resolved_config.extraction
        ),
        metadata=MetadataHarvester(
            resolved_config.metadata, resolved_clock, language_detector
        ),
        deduplicator=Deduplicator(
            store, resolved_config.deduplication, resolved_clock
        ),
        normalizer=RawDocumentNormalizer(resolved_hash, resolved_clock),
        offloader=resolved_offloader,
        clock=resolved_clock,
        id_generator=_UuidGenerator(),
        logger=resolved_logger,
        metrics=resolved_metrics,
    )
