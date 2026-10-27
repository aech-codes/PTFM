"""
PTFM Knowledge Processing Module

File:
    ptfm/core/knowledge/__init__.py

Description:
    Public interface of the Knowledge Processing Module, the second stage of
    the PTFM pipeline.

    Canonical ``RawDocument`` instances are segmented into provenance-
    preserving ``KnowledgeNode`` units, related structurally, enriched by
    domain plugins, and optionally embedded, producing the artifacts that
    Hybrid Retrieval indexes and Verification reasons over.

Composition root
----------------
    :func:`build_default_engine` is the single place in the module where
    concrete implementations are chosen and wired. Every other module depends
    on interfaces alone, which is what allows any component to be replaced
    without editing the pipeline.

    Resolution and fingerprinting happen together, so the engine cannot be
    constructed with a fingerprint that disagrees with the strategy and
    segmenter that produced it.

Scope
-----
    This module builds no index. It emits nodes, embeddings, and relations to
    a ``KnowledgeSink``, and Retrieval supplies implementations backed by
    vector, lexical, metadata, and graph storage whose schemas it owns.

    It produces no confidence value. Similarity is computed at query time and
    calibrated by the conformal layer.

Usage
-----
    ::

        from ptfm.core.knowledge import build_default_engine
        from ptfm.core.knowledge.sinks import InMemoryKnowledgeSink

        sink = InMemoryKnowledgeSink()
        engine = build_default_engine(sink=sink)

        report = await engine.process_all(documents)
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Iterable, Optional
from uuid import uuid4

from ptfm.core.ingestion.utils.concurrency import (
    BoundedThreadOffloader,
    InlineOffloader,
)
from ptfm.core.ingestion.utils.hashing import Blake2bHashProvider
from ptfm.core.knowledge.chunking.baselines import (
    BLOCK_CHUNKER_NAME,
    WHOLE_DOCUMENT_CHUNKER_NAME,
    BlockChunker,
    WholeDocumentChunker,
)
from ptfm.core.knowledge.chunking.segmenters import (
    REGEX_SEGMENTER_NAME,
    RegexSentenceSegmenter,
)
from ptfm.core.knowledge.chunking.structural import (
    STRUCTURAL_CHUNKER_NAME,
    StructuralChunker,
)
from ptfm.core.knowledge.config import (
    ChunkingConfig,
    EmbeddingConfig,
    EnrichmentConfig,
    EnrichmentFailurePolicy,
    KnowledgeConfig,
    RelationConfig,
    SinkConfig,
    TelemetryConfig,
    ValidationConfig,
)
from ptfm.core.knowledge.embedding.builtin import (
    DETERMINISTIC_PROVIDER_NAME,
    NULL_PROVIDER_NAME,
    DeterministicEmbeddingProvider,
    NullEmbeddingProvider,
)
from ptfm.core.knowledge.embedding.local import (
    LOCAL_PROVIDER_NAME,
    SentenceTransformerProvider,
)
from ptfm.core.knowledge.engine import KnowledgeEngine
from ptfm.core.knowledge.exceptions import (
    KnowledgeStage,
    PTFMKnowledgeError,
)
from ptfm.core.knowledge.interfaces import (
    ChunkingStrategy,
    Clock,
    EmbeddingProvider,
    EmbeddingTextComposer,
    KnowledgeEnricher,
    KnowledgeSink,
    MetricsSink,
    ModelIdentity,
    SentenceSegmenter,
    StructuredLogger,
    ThreadOffloader,
)
from ptfm.core.knowledge.models import (
    ChunkPlan,
    DocumentFailed,
    DocumentSkipped,
    KnowledgeUnit,
    NodeProduced,
    PlannedChunk,
    ProcessingFailure,
    ProcessingOutcome,
    ProcessingReport,
    SkipReason,
)
from ptfm.core.knowledge.pipeline.assembly import NodeAssembler
from ptfm.core.knowledge.pipeline.composition import (
    HeadingPathComposer,
    TextOnlyComposer,
)
from ptfm.core.knowledge.pipeline.embedding import EmbeddingStage
from ptfm.core.knowledge.pipeline.enrichment import EnrichmentStage
from ptfm.core.knowledge.pipeline.planning import ChunkPlanner
from ptfm.core.knowledge.pipeline.relations import RelationDeriver
from ptfm.core.knowledge.pipeline.validation import ProvenanceValidator
from ptfm.core.knowledge.registry import (
    EmbeddingRegistry,
    ProviderRegistration,
    SegmentationRegistry,
    SegmenterRegistration,
)
from ptfm.core.knowledge.sinks import (
    CompositeKnowledgeSink,
    InMemoryKnowledgeSink,
    NullKnowledgeSink,
)
from ptfm.core.knowledge.telemetry import build_logger, build_metrics_sink

__all__ = [
    # Engine and composition
    "KnowledgeEngine",
    "build_default_engine",
    "default_strategies",
    "default_segmenters",
    "default_providers",
    "default_composers",
    # Outcomes
    "ProcessingOutcome",
    "NodeProduced",
    "DocumentSkipped",
    "DocumentFailed",
    "SkipReason",
    "ProcessingFailure",
    "ProcessingReport",
    "KnowledgeUnit",
    "ChunkPlan",
    "PlannedChunk",
    "KnowledgeStage",
    "PTFMKnowledgeError",
    # Configuration
    "KnowledgeConfig",
    "ChunkingConfig",
    "EmbeddingConfig",
    "EnrichmentConfig",
    "EnrichmentFailurePolicy",
    "RelationConfig",
    "ValidationConfig",
    "SinkConfig",
    "TelemetryConfig",
    # Contracts
    "ChunkingStrategy",
    "SentenceSegmenter",
    "EmbeddingProvider",
    "EmbeddingTextComposer",
    "KnowledgeEnricher",
    "KnowledgeSink",
    "ModelIdentity",
    "Clock",
    "ThreadOffloader",
    "MetricsSink",
    "StructuredLogger",
    # Extension
    "SegmentationRegistry",
    "EmbeddingRegistry",
    "SegmenterRegistration",
    "ProviderRegistration",
    # Implementations
    "StructuralChunker",
    "WholeDocumentChunker",
    "BlockChunker",
    "RegexSentenceSegmenter",
    "NullEmbeddingProvider",
    "DeterministicEmbeddingProvider",
    "SentenceTransformerProvider",
    "HeadingPathComposer",
    "TextOnlyComposer",
    "NullKnowledgeSink",
    "InMemoryKnowledgeSink",
    "CompositeKnowledgeSink",
    "BoundedThreadOffloader",
    "InlineOffloader",
]


class _SystemClock:
    """
    Clock reading the system time.

    Returns timezone-aware instants, since a naive one cannot be compared
    against the timestamps used elsewhere in the pipeline.
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

    Used solely for run identifiers. Node identity is a deterministic
    derivation from document, span, and segmentation fingerprint, never a
    generated value, which is what keeps processing idempotent and citations
    stable across index rebuilds.
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


# ============================================================================
# DEFAULT REGISTRATIONS
# ============================================================================


def default_strategies(config: ChunkingConfig) -> tuple[ChunkingStrategy, ...]:
    """
    Construct the chunking strategies shipped with this module.

    Declared as a function rather than a constant so that each caller receives
    independent instances configured for its own run.

    Parameters
    ----------
    config:
        Boundary-affecting configuration.

    Returns
    -------
    tuple[ChunkingStrategy, ...]
        The structural strategy and the two baselines.
    """
    return (
        StructuralChunker(config, RegexSentenceSegmenter()),
        WholeDocumentChunker(config),
        BlockChunker(config),
    )


def default_segmenters() -> tuple[SegmenterRegistration, ...]:
    """
    Construct the sentence segmenters shipped with this module.

    Returns
    -------
    tuple[SegmenterRegistration, ...]
        The rule-based segmenter.
    """
    return (
        SegmenterRegistration(
            name=REGEX_SEGMENTER_NAME,
            segmenter=RegexSentenceSegmenter(),
            origin="builtin",
        ),
    )


def default_providers(
    offloader: ThreadOffloader,
) -> tuple[ProviderRegistration, ...]:
    """
    Construct the embedding provider registrations shipped with this module.

    Providers are registered as factories rather than instances, because a
    provider owns resources with a per-run lifecycle: a local model occupies
    gigabytes and takes seconds to load.

    Each factory reads its own settings from ``EmbeddingConfig.provider_options``
    and validates them itself. That is what keeps provider-specific parameters
    such as a model identifier out of the Core contract: naming them as
    configuration fields would require this module to know what every provider
    accepts, so adding a provider would mean editing the configuration.

    Parameters
    ----------
    offloader:
        Executor supplied to providers that perform blocking inference.

    Returns
    -------
    tuple[ProviderRegistration, ...]
        The null, deterministic, and local providers.
    """

    def _null(config: KnowledgeConfig) -> EmbeddingProvider:
        return NullEmbeddingProvider(config.embedding)

    def _deterministic(config: KnowledgeConfig) -> EmbeddingProvider:
        options = config.embedding.provider_options
        return DeterministicEmbeddingProvider(
            config.embedding,
            dimension=_int_option(options, "dimension", default=256),
        )

    def _local(config: KnowledgeConfig) -> EmbeddingProvider:
        options = config.embedding.provider_options
        return SentenceTransformerProvider(
            config.embedding,
            offloader,
            model_id=options.get("model_id") or _DEFAULT_LOCAL_MODEL,
            device=options.get("device") or None,
            max_concurrent_batches=_int_option(
                options, "max_concurrent_batches", default=1
            ),
            query_prefix=options.get("query_prefix", ""),
            passage_prefix=options.get("passage_prefix", ""),
        )

    return (
        ProviderRegistration(name=NULL_PROVIDER_NAME, factory=_null, origin="builtin"),
        ProviderRegistration(
            name=DETERMINISTIC_PROVIDER_NAME,
            factory=_deterministic,
            origin="builtin",
        ),
        ProviderRegistration(
            name=LOCAL_PROVIDER_NAME, factory=_local, origin="builtin"
        ),
    )


def default_composers(config: EmbeddingConfig) -> tuple[EmbeddingTextComposer, ...]:
    """
    Construct the embedding text composers shipped with this module.

    Parameters
    ----------
    config:
        Embedding configuration supplying the input bound.

    Returns
    -------
    tuple[EmbeddingTextComposer, ...]
        The heading-path and text-only compositions.
    """
    return (HeadingPathComposer(config), TextOnlyComposer(config))


#: Model loaded by the local provider when its options name none. Declared
#: here rather than in configuration because it is a property of that one
#: provider, not of the module.
_DEFAULT_LOCAL_MODEL = "BAAI/bge-large-en-v1.5"


def _int_option(
    options: object,
    key: str,
    *,
    default: int,
) -> int:
    """
    Read an integer from a provider's options.

    Options are string-valued so that a setting from code and the same setting
    from an environment variable are indistinguishable. A malformed value is a
    configuration error rather than something to silently default, since
    silently ignoring a dimension an operator set would produce vectors of an
    unexpected width.

    Parameters
    ----------
    options:
        The provider options mapping.
    key:
        Option to read.
    default:
        Value used when the option is absent.

    Returns
    -------
    int
        The parsed value.

    Raises
    ------
    KnowledgeConfigurationError
        If the value is present but not an integer.
    """
    from ptfm.core.knowledge.exceptions import KnowledgeConfigurationError

    raw = options.get(key) if hasattr(options, "get") else None
    if raw is None or raw == "":
        return default

    try:
        return int(raw)
    except (TypeError, ValueError) as exc:
        raise KnowledgeConfigurationError(
            f"Embedding provider option {key!r} must be an integer, got "
            f"{raw!r}.",
            context={"option": key, "value": str(raw)},
        ) from exc


# ============================================================================
# COMPOSITION ROOT
# ============================================================================


def build_default_engine(
    config: Optional[KnowledgeConfig] = None,
    *,
    sink: Optional[KnowledgeSink] = None,
    enrichers: Iterable[KnowledgeEnricher] = (),
    strategies: Optional[Iterable[ChunkingStrategy]] = None,
    segmenters: Optional[Iterable[SegmenterRegistration]] = None,
    providers: Optional[Iterable[ProviderRegistration]] = None,
    composers: Optional[Iterable[EmbeddingTextComposer]] = None,
    offloader: Optional[ThreadOffloader] = None,
    clock: Optional[Clock] = None,
    logger: Optional[StructuredLogger] = None,
    metrics: Optional[MetricsSink] = None,
    discover_entry_points: bool = True,
) -> KnowledgeEngine:
    """
    Construct a knowledge processing engine with the default implementations.

    The single composition root of the module. Every concrete class is chosen
    here and injected downward, so no stage constructs its own dependencies
    and any component can be replaced by passing a substitute.

    Segmentation is resolved and fingerprinted in one operation, and the
    resulting fingerprint is handed to the engine. The engine therefore cannot
    produce nodes whose fingerprint disagrees with the strategy and segmenter
    that placed their boundaries.

    Parameters
    ----------
    config:
        Configuration. Constructed from the environment when omitted.
    sink:
        Destination for produced units. When omitted the engine emits to its
        stream alone, which suits a caller consuming outcomes directly.
    enrichers:
        Domain metadata enrichers. None by default: the Core ships no domain
        knowledge.
    strategies:
        Chunking strategies, replacing the built-in set entirely when
        supplied.
    segmenters:
        Sentence segmenter registrations, replacing the built-in set.
    providers:
        Embedding provider registrations, replacing the built-in set.
    composers:
        Embedding text composers, replacing the built-in set.
    offloader:
        Executor for blocking work. A bounded thread offloader is constructed
        when omitted, so that the configured limit bounds the process as a
        whole.
    clock:
        Time source.
    logger:
        Structured logger.
    metrics:
        Metrics sink.
    discover_entry_points:
        Whether separately installed packages may contribute strategies,
        segmenters, providers, and composers.

    Returns
    -------
    KnowledgeEngine
        A fully wired engine.

    Raises
    ------
    ChunkerRegistrationError
        If a strategy or segmenter is invalid, or the configured one is not
        registered.
    EmbedderRegistrationError
        If a provider or composer is invalid, or the configured one is not
        registered.
    ProviderUnavailableError
        If the configured provider's optional dependency is absent.
    """
    resolved_config = config if config is not None else KnowledgeConfig()
    resolved_clock: Clock = clock if clock is not None else _SystemClock()
    resolved_offloader: ThreadOffloader = (
        offloader
        if offloader is not None
        else BoundedThreadOffloader(resolved_config.resolve_thread_limit())
    )

    resolved_logger = build_logger(
        resolved_config.telemetry.log_level, logger=logger
    )
    resolved_metrics = build_metrics_sink(
        resolved_config.telemetry.emit_metrics, sink=metrics
    )

    segmentation_registry = _build_segmentation_registry(
        resolved_config, strategies, segmenters, discover_entry_points
    )
    segmentation = segmentation_registry.resolve(resolved_config)

    resolved_logger.info(
        "knowledge.segmentation.resolved",
        strategy=segmentation.strategy.name,
        segmenter_version=segmentation.segmenter.version,
        segmentation_fingerprint=segmentation.fingerprint,
    )

    embedding_stage = _build_embedding_stage(
        resolved_config,
        providers,
        composers,
        resolved_offloader,
        resolved_clock,
        resolved_logger,
        discover_entry_points,
    )

    return KnowledgeEngine(
        config=resolved_config,
        segmentation_fingerprint=segmentation.fingerprint,
        planner=ChunkPlanner(segmentation.strategy, resolved_config.validation),
        assembler=NodeAssembler(resolved_config.validation, resolved_clock),
        relation_deriver=RelationDeriver(resolved_config.relations),
        enrichment=EnrichmentStage(
            tuple(enrichers), resolved_config.enrichment, resolved_logger
        ),
        embedding=embedding_stage,
        validator=ProvenanceValidator(resolved_config.validation),
        sink=sink,
        offloader=resolved_offloader,
        clock=resolved_clock,
        id_generator=_UuidGenerator(),
        logger=resolved_logger,
        metrics=resolved_metrics,
    )


def _build_segmentation_registry(
    config: KnowledgeConfig,
    strategies: Optional[Iterable[ChunkingStrategy]],
    segmenters: Optional[Iterable[SegmenterRegistration]],
    discover_entry_points: bool,
) -> SegmentationRegistry:
    """
    Assemble the segmentation registry for a run.

    Parameters
    ----------
    config:
        Configuration naming the components to resolve.
    strategies:
        Strategies replacing the built-in set, when supplied.
    segmenters:
        Segmenter registrations replacing the built-in set, when supplied.
    discover_entry_points:
        Whether installed packages may contribute components.

    Returns
    -------
    SegmentationRegistry
        The assembled registry.
    """
    resolved_strategies = (
        tuple(strategies)
        if strategies is not None
        else default_strategies(config.chunking)
    )
    resolved_segmenters = (
        tuple(segmenters) if segmenters is not None else default_segmenters()
    )

    if discover_entry_points:
        return SegmentationRegistry.from_entry_points(
            strategies=resolved_strategies, segmenters=resolved_segmenters
        )
    return SegmentationRegistry(resolved_strategies, resolved_segmenters)


def _build_embedding_stage(
    config: KnowledgeConfig,
    providers: Optional[Iterable[ProviderRegistration]],
    composers: Optional[Iterable[EmbeddingTextComposer]],
    offloader: ThreadOffloader,
    clock: Clock,
    logger: StructuredLogger,
    discover_entry_points: bool,
) -> Optional[EmbeddingStage]:
    """
    Assemble the embedding stage, or report that embedding is disabled.

    Returns ``None`` rather than a stage wrapping a provider that cannot
    embed. A stage that existed but always failed would turn a deliberate
    configuration choice into a per-batch error, and the engine would have to
    distinguish the two anyway.

    Parameters
    ----------
    config:
        Configuration naming the components to resolve.
    providers:
        Provider registrations replacing the built-in set, when supplied.
    composers:
        Composers replacing the built-in set, when supplied.
    offloader:
        Executor supplied to providers performing blocking inference.
    clock:
        Time source recorded on every embedding.
    logger:
        Structured logger.
    discover_entry_points:
        Whether installed packages may contribute components.

    Returns
    -------
    Optional[EmbeddingStage]
        The stage, or ``None`` when embedding is disabled.
    """
    if not config.embedding.enabled:
        logger.info("knowledge.embedding.disabled")
        return None

    resolved_providers = (
        tuple(providers)
        if providers is not None
        else default_providers(offloader)
    )
    resolved_composers = (
        tuple(composers)
        if composers is not None
        else default_composers(config.embedding)
    )

    registry = (
        EmbeddingRegistry.from_entry_points(
            providers=resolved_providers, composers=resolved_composers
        )
        if discover_entry_points
        else EmbeddingRegistry(resolved_providers, resolved_composers)
    )

    resolved = registry.resolve(config)

    logger.info(
        "knowledge.embedding.resolved",
        provider=config.embedding.provider,
        composition=resolved.composer.composition_id,
    )

    return EmbeddingStage(
        resolved.provider,
        resolved.composer,
        Blake2bHashProvider(),
        config.embedding,
        clock,
    )
