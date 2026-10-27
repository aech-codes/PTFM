"""
PTFM Retrieval Module

File:
    ptfm/core/retrieval/__init__.py

Description:
    The public surface of the retrieval module, and the composition root that
    wires an engine from configuration.

What the composition root is for
--------------------------------
    Every stage in this module names its collaborators by protocol and
    constructs none of them. That is what lets the same pipeline run over the
    in-memory stores in a test and over PostgreSQL in production, and it is what
    keeps a driver import out of the pipeline.

    The cost of that discipline is that something must eventually choose the
    concrete parts. This module is that place, and it is the ONLY place in the
    retrieval module where a concrete store is named. A stage that constructed
    its own collaborator would make the choice unreachable from here and the
    substitution impossible.

What is deliberately absent
---------------------------
    No confidence, no calibration, no verification, no truth. Retrieval ends
    with traceable evidence and an honest account of what it could not
    establish. Deciding what any of it supports belongs to the stages after
    this module, working from the Verifiable Truth Space.

    No reranking stage. The architecture defines where one goes — between
    fusion and hydration, consuming identifiers and returning a reordering —
    and no implementation is wired, because a reranker that scored index copies
    of passages rather than authoritative text would defeat the reason
    hydration precedes everything that reads content.

Backend selection
-----------------
    ``build_default_engine`` builds over the in-memory stores unless given
    stores of its own. That default is not a placeholder: the in-memory stores
    are the module's correctness reference, and an engine over them is the one a
    test should use.

    A persistent deployment passes ``stores=`` built from a backend. The
    PostgreSQL backend is an optional extra, and nothing here imports it, so a
    deployment that installs neither ``psycopg`` nor ``psycopg_pool`` still has
    a working engine.

Dependency contract
-------------------
    Imports the pipeline stages and the in-memory stores. Does NOT import the
    PostgreSQL backend, any driver, or any domain plugin. A caller wanting the
    persistent backend imports it explicitly and hands it over, which keeps the
    dependency visible at the call site rather than implied by an import here.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Optional, Sequence

from ptfm.core.ingestion.interfaces import (
    Clock,
    IdGenerator,
    MetricsSink,
    StructuredLogger,
)
from ptfm.core.retrieval.config import (
    ArmConfig,
    CredibilityConfig,
    DenseConfig,
    DegradationPolicy,
    ExpansionConfig,
    FusionConfig,
    GraphConfig,
    IngestConfig,
    PartitionConfig,
    RetrievalConfig,
    SparseConfig,
    SupersessionConfig,
    TelemetryConfig,
)
from ptfm.core.retrieval.engine import (
    EmbeddingCoverageProbe,
    RetrievalEngine,
    RetrievalOutcome,
)
from ptfm.core.retrieval.exceptions import (
    AllArmsFailedError,
    AnchorViolationError,
    ArmTimeoutError,
    ArmUnavailableError,
    CorroborationIntegrityError,
    CredibilityError,
    EmbeddingSpaceMismatchError,
    EmptyQueryError,
    ExpansionError,
    FusionError,
    HydrationError,
    IncomparableSignalsError,
    IndexWriteError,
    NodeNotFoundError,
    PTFMRetrievalError,
    PartialIndexWriteError,
    PartitionError,
    PartitionNotFoundError,
    ProvenanceIncompleteError,
    QueryEncodingError,
    QueryTooLargeError,
    RetrievalConfigurationError,
    RetrievalStage,
    SearchError,
    SegmentationMismatchError,
    StoreUnavailableError,
    SupersessionError,
)
from ptfm.core.retrieval.interfaces import (
    ContextExpander,
    CredibilityProvider,
    FusionStrategy,
    IndexWriter,
    LexicalIndex,
    NodeStore,
    PartitionCatalog,
    QueryEncoder,
    QueryPreprocessor,
    RelationStore,
    RetrievalStrategy,
    SupersessionStore,
    VectorIndex,
)
from ptfm.core.retrieval.models import (
    ArmDiagnostic,
    ArmResult,
    ArmStatus,
    Candidate,
    EmbeddingPartition,
    FilterField,
    FilterOperator,
    FusedCandidate,
    QueryPlan,
    RetrievalDiagnostics,
    RetrievalFilter,
    SegmentationPartition,
)
from ptfm.core.retrieval.partition import (
    PartitionResolver,
    embedding_for_vector,
    segmentation_for_node,
)
from ptfm.core.retrieval.pipeline.arms import DenseArm, SparseArm, StructuralArm
from ptfm.core.retrieval.pipeline.assembly import (
    AssembledEvidence,
    CredibilityAnnotator,
    EvidenceAssembler,
)
from ptfm.core.retrieval.pipeline.encoding import ProviderQueryEncoder, QueryPreparer
from ptfm.core.retrieval.pipeline.expansion import OrdinalContextExpander
from ptfm.core.retrieval.pipeline.fusion import ReciprocalRankFusion
from ptfm.core.retrieval.pipeline.hydration import (
    HydratedResult,
    NodeHydrator,
    SupersessionFilter,
)
from ptfm.core.retrieval.registry import (
    ArmRegistration,
    ArmRegistry,
    FusionRegistry,
)
from ptfm.core.retrieval.stores.memory import (
    InMemoryIndexWriter,
    InMemoryLexicalIndex,
    InMemoryNodeStore,
    InMemoryPartitionCatalog,
    InMemoryRelationStore,
    InMemorySupersessionStore,
    InMemoryVectorIndex,
)
from ptfm.core.retrieval.stores.tokenization import TOKENIZER_VERSION, tokenize

__all__ = [
    # Engine
    "RetrievalEngine",
    "RetrievalOutcome",
    "EmbeddingCoverageProbe",
    "RetrievalStores",
    "build_default_engine",
    "build_memory_stores",
    "default_arm_registrations",
    "default_fusion_strategies",
    # Configuration
    "RetrievalConfig",
    "ArmConfig",
    "DenseConfig",
    "SparseConfig",
    "GraphConfig",
    "FusionConfig",
    "ExpansionConfig",
    "SupersessionConfig",
    "CredibilityConfig",
    "PartitionConfig",
    "IngestConfig",
    "TelemetryConfig",
    "DegradationPolicy",
    # Models
    "QueryPlan",
    "Candidate",
    "ArmResult",
    "FusedCandidate",
    "ArmStatus",
    "ArmDiagnostic",
    "RetrievalDiagnostics",
    "RetrievalFilter",
    "FilterField",
    "FilterOperator",
    "SegmentationPartition",
    "EmbeddingPartition",
    # Partitions
    "PartitionResolver",
    "segmentation_for_node",
    "embedding_for_vector",
    # Interfaces
    "NodeStore",
    "VectorIndex",
    "LexicalIndex",
    "RelationStore",
    "SupersessionStore",
    "PartitionCatalog",
    "IndexWriter",
    "QueryEncoder",
    "QueryPreprocessor",
    "RetrievalStrategy",
    "FusionStrategy",
    "ContextExpander",
    "CredibilityProvider",
    # Pipeline stages
    "QueryPreparer",
    "ProviderQueryEncoder",
    "DenseArm",
    "SparseArm",
    "StructuralArm",
    "ReciprocalRankFusion",
    "NodeHydrator",
    "SupersessionFilter",
    "HydratedResult",
    "OrdinalContextExpander",
    "CredibilityAnnotator",
    "EvidenceAssembler",
    "AssembledEvidence",
    # Registries
    "ArmRegistration",
    "ArmRegistry",
    "FusionRegistry",
    # In-memory stores
    "InMemoryNodeStore",
    "InMemoryVectorIndex",
    "InMemoryLexicalIndex",
    "InMemoryRelationStore",
    "InMemorySupersessionStore",
    "InMemoryPartitionCatalog",
    "InMemoryIndexWriter",
    # Lexical terms
    "tokenize",
    "TOKENIZER_VERSION",
    # Exceptions
    "PTFMRetrievalError",
    "RetrievalStage",
    "RetrievalConfigurationError",
    "PartitionError",
    "PartitionNotFoundError",
    "SegmentationMismatchError",
    "EmbeddingSpaceMismatchError",
    "QueryEncodingError",
    "QueryTooLargeError",
    "EmptyQueryError",
    "SearchError",
    "ArmUnavailableError",
    "ArmTimeoutError",
    "AllArmsFailedError",
    "StoreUnavailableError",
    "FusionError",
    "IncomparableSignalsError",
    "SupersessionError",
    "HydrationError",
    "NodeNotFoundError",
    "ProvenanceIncompleteError",
    "ExpansionError",
    "AnchorViolationError",
    "CredibilityError",
    "CorroborationIntegrityError",
    "IndexWriteError",
    "PartialIndexWriteError",
]


class _SystemClock:
    """
    Clock reading the system time in UTC.

    UTC rather than local time, because a processing timestamp is compared
    across machines and a local one makes two such comparisons disagree.
    """

    __slots__ = ()

    def now(self) -> datetime:
        """
        Read the current time.

        Returns
        -------
        datetime
            The current time, timezone-aware in UTC.
        """
        return datetime.now(timezone.utc)


class _UuidGenerator:
    """
    Identifier source backed by random UUIDs.

    Random rather than sequential, so that a query identifier reveals nothing
    about how many queries preceded it.
    """

    __slots__ = ()

    def new_id(self) -> str:
        """
        Produce a new identifier.

        Returns
        -------
        str
            A UUID4 in hexadecimal.
        """
        return uuid.uuid4().hex


class _NullMetrics:
    """
    Metrics sink that discards everything.

    The default, so that an engine built without a metrics backend behaves
    identically to one built with it rather than failing on the first
    measurement.
    """

    __slots__ = ()

    def increment(self, name: str, value: int = 1, **labels: str) -> None:
        """
        Discard a counter increment.

        Parameters
        ----------
        name:
            Metric name.
        value:
            Amount to add.
        labels:
            Metric labels.
        """

    def observe(self, name: str, value: float, **labels: str) -> None:
        """
        Discard an observation.

        Parameters
        ----------
        name:
            Metric name.
        value:
            Observed value.
        labels:
            Metric labels.
        """


class _NullLogger:
    """
    Logger that discards everything.

    The default for the same reason as the metrics sink. A deployment passes a
    real logger; nothing in the pipeline behaves differently without one.
    """

    __slots__ = ()

    def bind(self, **values: object) -> _NullLogger:
        """
        Return a logger with additional context.

        Parameters
        ----------
        values:
            Context values, discarded.

        Returns
        -------
        _NullLogger
            This logger, since there is no context to carry.
        """
        return self

    def debug(self, event: str, **values: object) -> None:
        """
        Discard a debug event.

        Parameters
        ----------
        event:
            Event name.
        values:
            Event fields.
        """

    def info(self, event: str, **values: object) -> None:
        """
        Discard an info event.

        Parameters
        ----------
        event:
            Event name.
        values:
            Event fields.
        """

    def warning(self, event: str, **values: object) -> None:
        """
        Discard a warning event.

        Parameters
        ----------
        event:
            Event name.
        values:
            Event fields.
        """

    def error(self, event: str, **values: object) -> None:
        """
        Discard an error event.

        Parameters
        ----------
        event:
            Event name.
        values:
            Event fields.
        """


class RetrievalStores:
    """
    The set of stores an engine reads and writes.

    Grouped rather than passed as eight arguments, because they are not
    independent: the node store, vector index, lexical index and relation store
    must describe the SAME corpus. An engine assembled from stores belonging to
    different corpora would hydrate an identifier from one against text from
    another, and the Evidence-First guarantee would hold within each store while
    failing across them.

    Passing them as one object does not enforce that they agree — nothing can,
    short of a shared identity they do not have — but it makes the grouping the
    unit that is built and handed over, so they are chosen together.

    Attributes
    ----------
    nodes:
        Authoritative node storage.
    vectors:
        Dense vector index.
    lexical:
        Term index.
    relations:
        Structural relations.
    supersession:
        Document currency.
    catalog:
        Partition registry.
    writer:
        Index writer, or ``None`` for a read-only engine.
    """

    __slots__ = (
        "nodes",
        "vectors",
        "lexical",
        "relations",
        "supersession",
        "catalog",
        "writer",
    )

    def __init__(
        self,
        *,
        nodes: NodeStore,
        vectors: VectorIndex,
        lexical: LexicalIndex,
        relations: RelationStore,
        supersession: SupersessionStore,
        catalog: PartitionCatalog,
        writer: Optional[IndexWriter] = None,
    ) -> None:
        """
        Group stores describing one corpus.

        Parameters
        ----------
        nodes:
            Authoritative node storage.
        vectors:
            Dense vector index.
        lexical:
            Term index.
        relations:
            Structural relations.
        supersession:
            Document currency.
        catalog:
            Partition registry.
        writer:
            Index writer, or ``None`` for a read-only engine.
        """
        self.nodes = nodes
        self.vectors = vectors
        self.lexical = lexical
        self.relations = relations
        self.supersession = supersession
        self.catalog = catalog
        self.writer = writer

    def __repr__(self) -> str:
        """
        Describe the stores by name.

        Returns
        -------
        str
            A representation naming each store, and nothing about credentials.
        """
        return (
            f"RetrievalStores(nodes={self.nodes.name!r}, "
            f"vectors={self.vectors.name!r}, lexical={self.lexical.name!r}, "
            f"relations={self.relations.name!r}, "
            f"writer={self.writer.name if self.writer else None!r})"
        )


def build_memory_stores() -> RetrievalStores:
    """
    Build a complete set of in-memory stores.

    The correctness reference, and the right choice for a test. All seven share
    one node store, because the vector, lexical and relation indexes resolve
    identifiers against it: giving them separate node stores would let an index
    return an identifier the hydrator cannot resolve.

    Returns
    -------
    RetrievalStores
        Stores over a single in-memory corpus, with an atomic writer.
    """
    nodes = InMemoryNodeStore()
    vectors = InMemoryVectorIndex(nodes)
    lexical = InMemoryLexicalIndex(nodes)
    relations = InMemoryRelationStore(nodes)
    supersession = InMemorySupersessionStore()

    return RetrievalStores(
        nodes=nodes,
        vectors=vectors,
        lexical=lexical,
        relations=relations,
        supersession=supersession,
        catalog=InMemoryPartitionCatalog(),
        writer=InMemoryIndexWriter(
            nodes, vectors, lexical, relations, supersession
        ),
    )


def default_arm_registrations(stores: RetrievalStores) -> tuple[ArmRegistration, ...]:
    """
    Describe the built-in arms as registrations over a set of stores.

    Registrations rather than instances, so the registry resolves only the arms
    configuration enables. Building all three and discarding two would open
    store connections for arms that never run.

    The structural arm is registered whether or not it is enabled. A
    registration that exists and is disabled is visible in diagnostics as
    ``SKIPPED``; one that was never registered is invisible, and a caller
    reading diagnostics could not tell that graph retrieval had been available.

    Parameters
    ----------
    stores:
        The stores the arms search.

    Returns
    -------
    tuple[ArmRegistration, ...]
        Registrations for the dense, sparse, and structural arms.
    """
    return (
        ArmRegistration(
            name="dense",
            factory=lambda config: DenseArm(stores.vectors, config.dense),
        ),
        ArmRegistration(
            name="sparse",
            factory=lambda config: SparseArm(stores.lexical, config.sparse),
        ),
        ArmRegistration(
            name="graph",
            factory=lambda config: StructuralArm(stores.relations, config.graph),
        ),
    )


def default_fusion_strategies(config: RetrievalConfig) -> tuple[FusionStrategy, ...]:
    """
    Build the fusion strategies available by default.

    Only reciprocal rank fusion. It combines orderings using rank alone, so it
    cannot be misled by one arm's scores being larger than another's — which
    matters here because arm scores are uncalibrated and carry different scales
    by design. A score-combining strategy would require a calibration this
    module deliberately does not perform.

    Parameters
    ----------
    config:
        Retrieval configuration supplying the rank constant.

    Returns
    -------
    tuple[FusionStrategy, ...]
        The available strategies.
    """
    return (ReciprocalRankFusion(config.fusion),)


def build_default_engine(
    config: Optional[RetrievalConfig] = None,
    *,
    stores: Optional[RetrievalStores] = None,
    encoder: Optional[QueryEncoder] = None,
    preprocessor: Optional[QueryPreprocessor] = None,
    credibility: Optional[CredibilityProvider] = None,
    arm_registrations: Sequence[ArmRegistration] = (),
    fusion_strategies: Sequence[FusionStrategy] = (),
    clock: Optional[Clock] = None,
    id_generator: Optional[IdGenerator] = None,
    logger: Optional[StructuredLogger] = None,
    metrics: Optional[MetricsSink] = None,
    discover_plugins: bool = False,
) -> RetrievalEngine:
    """
    Assemble a retrieval engine from configuration.

    Every dependency is optional and every default is the least surprising one:
    in-memory stores, no query encoder, no credibility provider, discarding
    logger and metrics. An engine built with no arguments is therefore complete
    and usable, which is what makes it testable without a fixture.

    The absent encoder deserves note. Without one, no query vector is produced,
    so the dense arm is SKIPPED with that reason recorded rather than failing.
    Sparse and structural retrieval work unchanged. That is a deliberately
    usable configuration — lexical-only retrieval over a corpus with no
    embeddings — and not a broken one.

    Parameters
    ----------
    config:
        Retrieval configuration. Defaults to configuration read from the
        environment.
    stores:
        The stores to read and write. Defaults to a fresh in-memory set.
    encoder:
        Query encoder. ``None`` disables dense retrieval, as described above.
    preprocessor:
        Query preprocessor applied before encoding.
    credibility:
        Source credibility provider. ``None`` leaves every credibility
        unrecorded rather than defaulted, so nothing invented enters a
        computation carrying a statistical guarantee.
    arm_registrations:
        Additional arms beyond the built-in three.
    fusion_strategies:
        Additional fusion strategies beyond reciprocal rank fusion.
    clock:
        Time source. Defaults to the system clock in UTC.
    id_generator:
        Correlation identifier source. Defaults to random UUIDs.
    logger:
        Structured logger. Defaults to discarding.
    metrics:
        Metrics sink. Defaults to discarding.
    discover_plugins:
        Whether to load arms and fusion strategies from entry points.

        Off by default. Entry-point discovery makes the set of arms depend on
        what happens to be installed, so an engine's behaviour would change
        with an unrelated package. A deployment that wants plugin arms opts in.

    Returns
    -------
    RetrievalEngine
        An engine wired from the resolved parts.

    Raises
    ------
    RetrievalConfigurationError
        If configuration and the supplied parts cannot produce a working
        engine — no enabled arm, a non-atomic writer where atomicity is
        required, or a coverage threshold with nothing able to measure it.
    """
    resolved_config = config if config is not None else RetrievalConfig()
    resolved_stores = stores if stores is not None else build_memory_stores()
    resolved_clock: Clock = clock if clock is not None else _SystemClock()
    resolved_ids: IdGenerator = (
        id_generator if id_generator is not None else _UuidGenerator()
    )
    resolved_logger: StructuredLogger = (
        logger if logger is not None else _NullLogger()
    )
    resolved_metrics: MetricsSink = (
        metrics if metrics is not None else _NullMetrics()
    )

    registrations = tuple(default_arm_registrations(resolved_stores)) + tuple(
        arm_registrations
    )
    arm_registry = (
        ArmRegistry.from_entry_points(registrations=registrations)
        if discover_plugins
        else ArmRegistry(registrations)
    )

    strategies = tuple(default_fusion_strategies(resolved_config)) + tuple(
        fusion_strategies
    )
    fusion_registry = (
        FusionRegistry.from_entry_points(strategies=strategies)
        if discover_plugins
        else FusionRegistry(strategies)
    )

    resolver = PartitionResolver(
        resolved_stores.catalog, resolved_config.partitions
    )
    preparer = QueryPreparer(
        resolver,
        encoder,
        resolved_config,
        resolved_ids,
        resolved_logger,
        preprocessor=preprocessor,
    )

    expander = (
        OrdinalContextExpander(
            resolved_stores.nodes, resolved_config.expansion, resolved_logger
        )
        if resolved_config.expansion.enabled
        else None
    )

    coverage = (
        EmbeddingCoverageProbe(resolved_stores.nodes, resolved_stores.vectors)
        if resolved_config.partitions.minimum_embedding_coverage > 0.0
        else None
    )

    return RetrievalEngine(
        preparer=preparer,
        arms=arm_registry.resolve(resolved_config),
        fusion=fusion_registry.resolve(resolved_config),
        hydrator=NodeHydrator(resolved_stores.nodes, resolved_logger),
        supersession=SupersessionFilter(
            resolved_stores.supersession,
            resolved_config.supersession,
            resolved_logger,
        ),
        expander=expander,
        credibility=CredibilityAnnotator(
            credibility, resolved_config.credibility, resolved_logger
        ),
        assembler=EvidenceAssembler(resolved_logger),
        config=resolved_config,
        clock=resolved_clock,
        id_generator=resolved_ids,
        logger=resolved_logger,
        metrics=resolved_metrics,
        writer=resolved_stores.writer,
        coverage=coverage,
    )
