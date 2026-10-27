"""
PTFM Retrieval Contracts

File:
    ptfm/core/retrieval/interfaces.py

Description:
    Defines every abstraction on which the Retrieval Module depends. The
    engine and its stages reference ONLY the types declared here; they never
    import a concrete store, index, strategy, or backend.

The isolation that matters most
-------------------------------
    A retrieval strategy returns IDENTIFIERS AND SIGNALS. It never returns
    text, spans, or provenance.

    Authoritative content comes from the node store alone. An index may hold
    its own copy for its own purposes, and that copy may be stale, partial, or
    transformed for search; none of it reaches an evidence item. A defective
    or compromised index can therefore change which nodes are found, but
    cannot change what a citation resolves to.

    This is the structural equivalent of Module 2's decision that a chunking
    strategy returns a plan rather than nodes: the component with the most
    freedom is given the least authority.

Five stores, not three
----------------------
    Dense, lexical, and structural retrieval imply three indexes. Two more
    stores are load-bearing and are declared as peers rather than folded into
    the others:

        NodeStore
            Authoritative ``KnowledgeNode`` by identifier. The source of every
            span, text, and provenance record that leaves this module, and the
            reason expansion needs no document fetch on the query path.

        SupersessionStore
            Which document currently represents a source. Keyed by source URI
            rather than by node, because supersession is a statement about a
            source over time and a node knows nothing about its successors.

Partitions, not checks
----------------------
    Every store operation takes a partition key. Incompatible artifacts are
    therefore not merely rejected at query time; they cannot occupy the same
    index. Segmentation and embedding partitions are separate keys, because
    re-embedding a corpus must not require rewriting its nodes, lexical
    index, relations, or supersession records.

Execution contract
------------------
    Store and strategy operations are asynchronous: each crosses a process,
    network, or storage boundary.

    Fusion is synchronous. It is pure arithmetic over ranks with no I/O, and
    an asynchronous signature would invite blocking work onto the event loop
    behind a coroutine that conceals it.

    Both halves are enforced at class-definition time rather than left to
    documentation.

Nothing here judges
-------------------
    No contract in this module returns a confidence, a probability, or a
    truth value. A strategy returns an ordering and an uncalibrated signal on
    a named scale. Turning such a signal into a confidence is the conformal
    calibration layer's responsibility and no other component's.

Dependency contract
-------------------
    Imports only the standard library, ``ptfm.core.retrieval.exceptions``,
    and contracts reused from earlier modules. Schema and model types are
    imported under ``TYPE_CHECKING`` exclusively, so this module carries no
    runtime dependency on either.
"""

from __future__ import annotations

import inspect
from types import TracebackType
from typing import (
    TYPE_CHECKING,
    Any,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    runtime_checkable,
)

from ptfm.core.ingestion.interfaces import (
    Clock,
    HashProvider,
    IdGenerator,
    MetricsSink,
    StructuredLogger,
    ThreadOffloader,
)
from ptfm.core.knowledge.interfaces import EmbeddingProvider, ModelIdentity
from ptfm.core.retrieval.exceptions import (
    StoreRegistrationError,
    StrategyRegistrationError,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ptfm.core.knowledge.models import KnowledgeUnit
    from ptfm.core.retrieval.models import (
        ArmResult,
        Candidate,
        EmbeddingPartition,
        FusedCandidate,
        QueryPlan,
        RetrievalFilter,
        SegmentationPartition,
    )
    from ptfm.core.schemas import KnowledgeNode, NodeEmbedding, NodeRelation

__all__ = [
    # Reused from earlier modules
    "Clock",
    "EmbeddingProvider",
    "HashProvider",
    "IdGenerator",
    "MetricsSink",
    "ModelIdentity",
    "StructuredLogger",
    "ThreadOffloader",
    # Stores
    "NodeStore",
    "VectorIndex",
    "LexicalIndex",
    "RelationStore",
    "SupersessionStore",
    "IndexWriter",
    "PartitionCatalog",
    # Query pipeline
    "QueryEncoder",
    "RetrievalStrategy",
    "FusionStrategy",
    "ContextExpander",
    "CredibilityProvider",
    "QueryPreprocessor",
]


# ============================================================================
# STORES
# ============================================================================


@runtime_checkable
class NodeStore(Protocol):
    """
    Authoritative store of knowledge nodes, addressed by identifier.

    The single source of text, span, and provenance for everything this module
    emits. Indexes return identifiers; this store turns them into evidence.

    Two access patterns, both required
    ----------------------------------
    Retrieval resolves nodes by identifier. Expansion resolves them by
    position, because a node's neighbours are its ordinal siblings within a
    document. Module 2 guarantees ordinals contiguous from zero and spans
    disjoint and ascending, so positional lookup is exact rather than
    heuristic, and expansion needs no graph traversal.

    Historical resolution
    ---------------------
    :meth:`get_many` resolves a node whether or not its document is current. A
    stored evidence item cites a node identifier, and that citation must
    remain auditable after the document has been superseded. Filtering to
    current documents is the supersession stage's concern, applied to search
    results rather than to resolution.

    Note that ``runtime_checkable`` verifies member presence only. Signature
    conformance is established by ``mypy --strict``.
    """

    @property
    def name(self) -> str:
        """
        Stable identifier of this store.

        Returns
        -------
        str
            Implementation identifier.
        """
        ...

    async def get_many(
        self,
        node_ids: Sequence[str],
        partition: SegmentationPartition,
    ) -> Mapping[str, KnowledgeNode]:
        """
        Resolve nodes by identifier.

        Batched rather than singular because retrieval resolves a page of
        results at once, and a per-node round trip would dominate query
        latency.

        Resolves regardless of whether a node's document is current, so that
        a historical citation remains auditable.

        Parameters
        ----------
        node_ids:
            Identifiers to resolve.
        partition:
            The segmentation partition to resolve within.

        Returns
        -------
        Mapping[str, KnowledgeNode]
            Resolved nodes keyed by identifier. An identifier absent from the
            result was not found; the caller decides whether that is a fault.

        Raises
        ------
        StoreUnavailableError
            If the store could not be reached.
        """
        ...

    async def get_by_ordinal(
        self,
        document_id: str,
        ordinals: Sequence[int],
        partition: SegmentationPartition,
    ) -> Mapping[int, KnowledgeNode]:
        """
        Resolve a document's nodes by position.

        Serves context expansion. A node's neighbours are its ordinal
        siblings, so expansion is a bounded positional lookup rather than a
        traversal.

        Parameters
        ----------
        document_id:
            The document whose nodes are sought.
        ordinals:
            Positions to resolve. Positions outside the document are omitted
            from the result rather than raising, since asking for a neighbour
            of the first or last node is normal.
        partition:
            The segmentation partition to resolve within.

        Returns
        -------
        Mapping[int, KnowledgeNode]
            Resolved nodes keyed by ordinal.

        Raises
        ------
        StoreUnavailableError
            If the store could not be reached.
        """
        ...

    async def put_many(
        self,
        nodes: Sequence[KnowledgeNode],
        partition: SegmentationPartition,
    ) -> None:
        """
        Store nodes, replacing any existing node with the same identifier.

        Last-one-wins is safe because node identity is deterministic:
        rewriting a node rewrites identical content, so a re-run over an
        unchanged corpus converges rather than duplicating.

        Parameters
        ----------
        nodes:
            The nodes to store.
        partition:
            The segmentation partition to write into.

        Returns
        -------
        None

        Raises
        ------
        IndexWriteError
            If the write was rejected.
        StoreUnavailableError
            If the store could not be reached.
        """
        ...

    async def count(self, partition: SegmentationPartition) -> int:
        """
        Report how many nodes a partition holds.

        Used to compute embedding coverage, since a vector partition holding
        fewer vectors than its segmentation partition holds nodes is only
        partially embedded, and an unembedded node is invisible to dense
        search.

        Parameters
        ----------
        partition:
            The segmentation partition to count.

        Returns
        -------
        int
            Node count.
        """
        ...


@runtime_checkable
class VectorIndex(Protocol):
    """
    Approximate or exact nearest-neighbour index over node embeddings.

    Returns identifiers and similarity signals. It never returns node text:
    an index copy may be stale or transformed for search, and admitting one
    into evidence would let a search structure determine what a citation
    resolves to.

    Keyed by embedding partition, so vectors from differing models,
    dimensions, normalizations, compositions, or query conventions cannot
    occupy one index. Mixing them is not rejected at query time; it is
    inexpressible.
    """

    @property
    def name(self) -> str:
        """
        Stable identifier of this index.

        Returns
        -------
        str
            Implementation identifier.
        """
        ...

    @property
    def supports_exact_search(self) -> bool:
        """
        Whether this index can search exhaustively.

        Approximate search is not deterministic: results vary with index
        construction and insertion order. An exact mode makes a query
        reproducible, which is what an audit of a past verified response
        requires, and is worth its latency for that purpose alone.

        Returns
        -------
        bool
            ``True`` when exact search is available.
        """
        ...

    async def search(
        self,
        vector: Sequence[float],
        partition: EmbeddingPartition,
        top_k: int,
        *,
        filters: Sequence[RetrievalFilter] = (),
        exact: bool = False,
    ) -> Sequence[Candidate]:
        """
        Find the nodes whose vectors are nearest a query vector.

        Parameters
        ----------
        vector:
            The query vector, produced by the same model and under the same
            query convention the partition records.
        partition:
            The embedding partition to search.
        top_k:
            Largest number of results to return.
        filters:
            Structural predicates applied within the index, so that filtering
            does not require over-fetching and post-filtering.
        exact:
            Whether to search exhaustively. Ignored by an index reporting no
            exact support, which reports that through
            :attr:`supports_exact_search` rather than by failing.

        Returns
        -------
        Sequence[Candidate]
            Identifiers with rank and similarity signal, best first.

        Raises
        ------
        StoreUnavailableError
            If the index could not be reached.
        EmbeddingSpaceMismatchError
            If the vector does not belong to the partition's embedding space.
        """
        ...

    async def upsert(
        self,
        embeddings: Sequence[NodeEmbedding],
        partition: EmbeddingPartition,
    ) -> None:
        """
        Store embeddings, replacing any existing vector for the same node.

        Parameters
        ----------
        embeddings:
            The embeddings to store.
        partition:
            The embedding partition to write into.

        Returns
        -------
        None

        Raises
        ------
        IndexWriteError
            If the write was rejected.
        EmbeddingSpaceMismatchError
            If an embedding does not belong to the partition's space.
        """
        ...

    async def count(self, partition: EmbeddingPartition) -> int:
        """
        Report how many vectors a partition holds.

        Returns
        -------
        int
            Vector count.
        """
        ...


@runtime_checkable
class LexicalIndex(Protocol):
    """
    Term-based index over node text.

    Not redundant with dense retrieval. Vector search is weak at exact match,
    and identifiers, quantities, dates, and rare tokens are precisely what a
    verification engine checks. A claim about a specific figure is found
    lexically or not at all.

    Keyed by segmentation partition rather than embedding partition, because
    lexical indexing depends on the text and its analysis, not on any
    embedding model. Re-embedding a corpus therefore leaves this index
    untouched.
    """

    @property
    def name(self) -> str:
        """
        Stable identifier of this index.

        Returns
        -------
        str
            Implementation identifier.
        """
        ...

    async def search(
        self,
        query: str,
        partition: SegmentationPartition,
        top_k: int,
        *,
        filters: Sequence[RetrievalFilter] = (),
    ) -> Sequence[Candidate]:
        """
        Find the nodes whose text best matches a query.

        Parameters
        ----------
        query:
            The query text, after any domain preprocessing.
        partition:
            The segmentation partition to search.
        top_k:
            Largest number of results to return.
        filters:
            Structural predicates applied within the index.

        Returns
        -------
        Sequence[Candidate]
            Identifiers with rank and relevance signal, best first. The signal
            is corpus-relative and unbounded, which is why its scale is
            recorded alongside it.

        Raises
        ------
        StoreUnavailableError
            If the index could not be reached.
        """
        ...

    async def index_nodes(
        self,
        nodes: Sequence[KnowledgeNode],
        partition: SegmentationPartition,
    ) -> None:
        """
        Add nodes to the lexical index.

        Parameters
        ----------
        nodes:
            The nodes to index.
        partition:
            The segmentation partition to write into.

        Returns
        -------
        None

        Raises
        ------
        IndexWriteError
            If the write was rejected.
        """
        ...


@runtime_checkable
class RelationStore(Protocol):
    """
    Store of structural relations between nodes.

    Relations are facts about a document's structure, derived from its block
    tree. They carry no semantic judgement: there is no similarity or
    entailment relation, because those are verification judgements and one
    recorded as structure would enter the knowledge base outside the layer
    that must govern it.

    Consumed by structural retrieval and, as a peer, by verification. Sequence
    expansion deliberately does NOT use this store: a node's neighbours are
    its ordinal siblings, so positional lookup against the node store is exact
    and avoids a traversal per result on the query path.
    """

    @property
    def name(self) -> str:
        """
        Stable identifier of this store.

        Returns
        -------
        str
            Implementation identifier.
        """
        ...

    async def relations_from(
        self,
        node_ids: Sequence[str],
        partition: SegmentationPartition,
        *,
        relation_types: Sequence[str] = (),
    ) -> Mapping[str, Sequence[NodeRelation]]:
        """
        Retrieve the relations originating from given nodes.

        Parameters
        ----------
        node_ids:
            Origin identifiers.
        partition:
            The segmentation partition to read within.
        relation_types:
            Relation types to include. Empty includes every type.

        Returns
        -------
        Mapping[str, Sequence[NodeRelation]]
            Relations keyed by origin identifier.

        Raises
        ------
        StoreUnavailableError
            If the store could not be reached.
        """
        ...

    async def structural_search(
        self,
        filters: Sequence[RetrievalFilter],
        partition: SegmentationPartition,
        top_k: int,
    ) -> Sequence[Candidate]:
        """
        Find nodes satisfying a structural constraint.

        The honest form of structural retrieval. It answers questions a
        similarity cannot: which nodes fall under a heading, belong to a
        table, appear on a page, or carry a metadata value a domain plugin
        attached.

        Ordering is optional, and its absence is reported rather than
        disguised. Where the constraint implies an order, such as position
        within a document, candidates carry ranks reflecting it. Where it
        implies only membership, every candidate carries the same rank and no
        score, and the caller records that this arm contributed membership
        rather than relevance. Inventing a relevance value to justify the
        arm's participation would fabricate a signal, which is worse than
        contributing none.

        Declared on this store rather than the node store because it searches
        STRUCTURE: the heading tree, table and list membership, and pagination
        are all derived from the block tree this store describes. A backend
        will ordinarily satisfy it by querying node attributes alongside
        relations, which is an implementation detail rather than a contract
        one.

        Parameters
        ----------
        filters:
            The structural constraint. An empty sequence matches nothing
            rather than everything: a structural arm with no constraint has no
            question to answer, and returning the corpus would present
            arbitrary nodes as responsive.
        partition:
            The segmentation partition to search.
        top_k:
            Largest number of candidates to return.

        Returns
        -------
        Sequence[Candidate]
            Matching nodes. Ranks are contiguous from zero, and scores are
            absent where the constraint implies no ordering.

        Raises
        ------
        StoreUnavailableError
            If the store could not be reached.
        """
        ...

    async def put_many(
        self,
        relations: Sequence[NodeRelation],
        partition: SegmentationPartition,
    ) -> None:
        """
        Store relations.

        Parameters
        ----------
        relations:
            The relations to store.
        partition:
            The segmentation partition to write into.

        Returns
        -------
        None

        Raises
        ------
        IndexWriteError
            If the write was rejected.
        """
        ...


@runtime_checkable
class SupersessionStore(Protocol):
    """
    Records which document currently represents each source.

    Keyed by source URI, because supersession is a statement about a source
    over time and a node knows nothing about its successors.

    Filter, never delete
    --------------------
    A superseded document's nodes remain resolvable. A stored evidence item
    cites a node identifier, and if that identifier stopped resolving, a past
    verified response would become unauditable, which would forfeit the
    guarantee this system exists to provide.

    Superseded nodes are therefore excluded from SEARCH results while
    remaining available to RESOLUTION. Reclaiming their storage is a retention
    policy, deliberately not a retrieval concern.

    Absence is distinguishable from removal
    ---------------------------------------
    A source that has been deleted is recorded with no current document,
    which is a different statement from a source that was never ingested. A
    consumer that could not tell them apart would report a removed document as
    though it had never existed.
    """

    @property
    def name(self) -> str:
        """
        Stable identifier of this store.

        Returns
        -------
        str
            Implementation identifier.
        """
        ...

    async def current_document_ids(
        self,
        source_uris: Sequence[str],
        partition: SegmentationPartition,
    ) -> Mapping[str, Optional[str]]:
        """
        Report the current document for each source.

        Parameters
        ----------
        source_uris:
            Sources to resolve.
        partition:
            The segmentation partition to read within.

        Returns
        -------
        Mapping[str, Optional[str]]
            Current document identifier per source. ``None`` records a source
            whose content has been removed, which is distinct from a source
            absent from the mapping entirely, meaning never ingested.

        Raises
        ------
        SupersessionError
            If currency could not be determined.
        """
        ...

    async def record_current(
        self,
        source_uri: str,
        document_id: Optional[str],
        version_tag: Optional[str],
        partition: SegmentationPartition,
    ) -> None:
        """
        Record which document now represents a source.

        Parameters
        ----------
        source_uri:
            The source.
        document_id:
            The current document, or ``None`` to record removal.
        version_tag:
            Source-side version marker, giving supersession an ordering
            derived from the source rather than from ingestion order.
        partition:
            The segmentation partition to write into.

        Returns
        -------
        None

        Raises
        ------
        IndexWriteError
            If the write was rejected.
        """
        ...


@runtime_checkable
@runtime_checkable
class IndexWriter(Protocol):
    """
    Writes knowledge units into every store as one operation.

    Declared alongside the per-store write methods rather than instead of
    them, because the two express different things. A store's own write method
    is the MECHANISM by which one store accepts data; this is the UNIT of work
    that must succeed or fail as a whole.

    Why the unit-level contract is needed
    -------------------------------------
    A knowledge unit touches the node, vector, lexical, relation, and
    supersession stores together. If those writes are independent, a crash
    between them leaves a node retrievable by one index while its
    authoritative text is absent: a result that appears valid and cannot be
    traced, which is the worst failure available to an evidence system.

    Expressing the unit of work as a contract means a backend able to offer
    atomicity can do so, and one unable to must say so through
    ``PartialIndexWriteError`` rather than leaving the risk unstated. Recovery
    is possible either way, because node identity is deterministic and
    rewriting a unit reproduces identical content, but recovery should be
    deliberate rather than assumed.

    Two implementation shapes are expected. A transactional backend implements
    this directly, opening one transaction across its stores. A composing
    backend implements it by calling each store in turn and reporting a
    partial write when one fails; that is correct for testing and for
    deployments that accept the weaker guarantee.

    This is the protocol behind the module's implementation of the knowledge
    sink contract, which is how Knowledge Processing delivers its output.
    """

    @property
    def name(self) -> str:
        """
        Stable identifier of this writer.

        Returns
        -------
        str
            Implementation identifier.
        """
        ...

    @property
    def is_atomic(self) -> bool:
        """
        Whether this writer's unit writes succeed or fail as a whole.

        Declared rather than assumed, so that a deployment can discover at
        startup that its backend offers only per-store writes, instead of
        discovering it from an untraceable result months later.

        Returns
        -------
        bool
            ``True`` when a failed unit leaves no store partially written.
        """
        ...

    async def write_units(
        self,
        segmentation: SegmentationPartition,
        embedding: Optional[EmbeddingPartition],
        units: Sequence[KnowledgeUnit],
    ) -> None:
        """
        Write units to every store.

        Parameters
        ----------
        segmentation:
            Partition receiving nodes, lexical entries, relations, and
            supersession records.
        embedding:
            Partition receiving vectors. ``None`` when the units carry none,
            which is a corpus segmented but not yet embedded and is a valid
            intermediate state rather than an error.
        units:
            The units to write.

        Returns
        -------
        None

        Raises
        ------
        IndexWriteError
            If the write failed and no store was left partially written.
        PartialIndexWriteError
            If some stores accepted the units and others did not.
        EmbeddingSpaceMismatchError
            If a unit carries a vector that does not belong to the named
            embedding partition.
        """
        ...

    async def flush(self) -> None:
        """
        Commit any buffered writes.

        Returns
        -------
        None

        Raises
        ------
        IndexWriteError
            If buffered writes could not be committed.
        """
        ...

    async def close(self) -> None:
        """
        Release resources held by this writer. Idempotent.

        Returns
        -------
        None
        """
        ...


class PartitionCatalog(Protocol):
    """
    Records which partitions exist and which serve queries.

    Exactly one segmentation partition is active. Several embedding
    partitions may exist over it: one active, others being built or retired.
    That asymmetry is the whole of the migration story, and it holds because
    re-embedding changes vectors while leaving nodes, spans, identifiers, and
    lexical terms untouched.
    """

    @property
    def name(self) -> str:
        """
        Stable identifier of this catalog.

        Returns
        -------
        str
            Implementation identifier.
        """
        ...

    async def active_segmentation(self) -> Optional[SegmentationPartition]:
        """
        Report the segmentation partition that serves queries.

        Returns
        -------
        Optional[SegmentationPartition]
            The active partition, or ``None`` when nothing has been indexed.

        Raises
        ------
        AmbiguousPartitionError
            If more than one segmentation partition is marked active.
        """
        ...

    async def active_embedding(
        self,
        segmentation: SegmentationPartition,
    ) -> Optional[EmbeddingPartition]:
        """
        Report the embedding partition that serves dense queries.

        Parameters
        ----------
        segmentation:
            The active segmentation partition.

        Returns
        -------
        Optional[EmbeddingPartition]
            The active embedding partition, or ``None`` when the corpus is
            not embedded, in which case dense retrieval is unavailable and
            the other arms serve.
        """
        ...

    async def register(
        self,
        segmentation: SegmentationPartition,
        embedding: Optional[EmbeddingPartition],
        *,
        activate: bool,
    ) -> None:
        """
        Record a partition, optionally making it active.

        Activation is the atomic switch a migration turns. Until it is turned,
        a partition may be written to and is not served.

        Parameters
        ----------
        segmentation:
            The segmentation partition.
        embedding:
            The embedding partition, when one is being registered.
        activate:
            Whether to make the partition serve queries.

        Returns
        -------
        None

        Raises
        ------
        IndexWriteError
            If the registration was rejected.
        """
        ...


# ============================================================================
# QUERY PIPELINE
# ============================================================================


@runtime_checkable
class QueryEncoder(Protocol):
    """
    Converts a query into a vector comparable with an embedding partition.

    Delegates to an embedding provider's own query method rather than
    applying a transformation itself. Which transformation a model expects on
    the query side is the provider's knowledge; a component reconstructing it
    from configuration would be guessing at provider-specific behaviour, and
    guessing wrong degrades retrieval measurably while producing no error.

    Compatibility is verified before encoding, so a query is never embedded
    into a space it will then be compared against incorrectly.
    """

    @property
    def model_identity(self) -> ModelIdentity:
        """
        Identity of the model this encoder encodes with.

        Returns
        -------
        ModelIdentity
            Identifier, dimension, normalization, and query convention.
        """
        ...

    async def encode(
        self,
        query: str,
        partition: EmbeddingPartition,
    ) -> tuple[float, ...]:
        """
        Encode a query for a given embedding partition.

        Parameters
        ----------
        query:
            The query text, after any domain preprocessing.
        partition:
            The partition the resulting vector will be compared against.

        Returns
        -------
        tuple[float, ...]
            A vector comparable with the partition's stored vectors.

        Raises
        ------
        QueryEncodingMismatchError
            If the encoder's query convention differs from the partition's.
        EmbeddingSpaceMismatchError
            If the encoder's model differs from the partition's.
        QueryEncodingError
            If encoding failed.
        EmptyQueryError
            If the query carries no searchable content.
        QueryTooLargeError
            If the query exceeds what the model accepts.
        """
        ...


class RetrievalStrategy:
    """
    One arm of hybrid retrieval.

    Implementation contract
    -----------------------
    - :meth:`search` MUST be asynchronous. Every arm crosses a store
      boundary.

    - :meth:`search` MUST return IDENTIFIERS AND SIGNALS ONLY. Returning text
      or provenance would let a search structure determine what a citation
      resolves to; returning identifiers means a defective index can change
      which nodes are found and nothing else.

    - Results MUST be ordered best-first, and each candidate's rank MUST
      reflect that ordering. Rank is what fusion consumes; a raw score is
      optional, because a structural predicate produces an ordering without
      one.

    - An arm MUST NOT invent a score. A strategy that produces a set rather
      than an ordering reports rank-only candidates, and fusion handles them
      correctly. Synthesizing a number to appear comparable with the other
      arms would fabricate a measurement.

    - An arm MUST be independently failable. The engine catches a failing arm
      and continues with the others, so an arm must not hold state that a
      later arm depends on.

    The asynchronous requirement is enforced at class-definition time.
    """
    #: Declared so that a subclass's own ``__slots__`` takes effect. Without
    #: it, Python gives every subclass a ``__dict__`` regardless of what that
    #: subclass declares, and an implementation believing itself stateless
    #: could silently accumulate per-query state on an instance shared by
    #: every concurrent query.
    __slots__ = ()


    def __init_subclass__(cls, **kwargs: Any) -> None:
        """
        Verify the asynchronous contract of a strategy implementation.

        Parameters
        ----------
        **kwargs:
            Forwarded to ``object.__init_subclass__``.

        Returns
        -------
        None

        Raises
        ------
        StrategyRegistrationError
            If ``search`` or ``health_check`` is defined and is not a
            coroutine function.
        """
        super().__init_subclass__(**kwargs)

        for method_name in ("search", "health_check"):
            member = cls.__dict__.get(method_name)
            if member is None:
                continue
            if not inspect.iscoroutinefunction(member):
                raise StrategyRegistrationError(
                    f"{cls.__name__}.{method_name} must be a coroutine "
                    f"function declared with 'async def'. Every retrieval arm "
                    f"crosses a store boundary and must not block the event "
                    f"loop.",
                    strategy_name=cls.__name__,
                )

    @property
    def name(self) -> str:
        """
        Stable identifier of this arm.

        Returns
        -------
        str
            Implementation identifier.
        """
        raise NotImplementedError

    @property
    def method(self) -> str:
        """
        The discovery method this arm records on its results.

        Distinct from :attr:`name`: several arms could in principle use one
        method, and the method is what appears in a result's retrieval
        history.

        Returns
        -------
        str
            A ``DiscoveryMethod`` value.
        """
        raise NotImplementedError

    async def search(self, plan: QueryPlan) -> Sequence[Candidate]:
        """
        Find candidate nodes for a query.

        Parameters
        ----------
        plan:
            The resolved query: preprocessed text, encoded vector where
            available, partitions, filters, and limits.

        Returns
        -------
        Sequence[Candidate]
            Identifiers with rank and optional signal, best first. Empty
            where the arm found nothing, which is a normal result and
            distinct from failing.

        Raises
        ------
        ArmUnavailableError
            If the arm could not execute.
        ArmTimeoutError
            If the arm exceeded its deadline.
        """
        raise NotImplementedError

    async def health_check(self) -> bool:
        """
        Verify that this arm can execute.

        Returns
        -------
        bool
            ``True`` when the arm is operational.
        """
        raise NotImplementedError


class FusionStrategy:
    """
    Combines the orderings produced by several arms into one.

    Implementation contract
    -----------------------
    - :meth:`fuse` MUST be synchronous. It is pure arithmetic over ranks with
      no I/O, and an asynchronous signature would invite blocking work onto
      the event loop behind a coroutine that conceals it.

    - :meth:`fuse` MUST be deterministic and independent of the order arms
      completed in. Arms run concurrently, so a fusion sensitive to
      completion order would make ranking vary between identical queries.

    - Ties MUST be broken deterministically, by node identifier. An arbitrary
      tie-break would make a query irreproducible for audit.

    - A fused result MUST retain every arm's signal. The fused position is a
      derived summary; the per-arm observations are the record of what was
      actually measured, and discarding them would leave a consumer unable to
      tell why a result ranked where it did.

    - Fusion MUST NOT produce a probability. Its output is an ordinal
      position on its own scale, and any component treating it as a
      likelihood would be reading a rank as a measurement.

    The synchronous requirement is enforced at class-definition time.
    """
    #: Declared so that a subclass's own ``__slots__`` takes effect. Without
    #: it, Python gives every subclass a ``__dict__`` regardless of what that
    #: subclass declares, and an implementation believing itself stateless
    #: could silently accumulate per-query state on an instance shared by
    #: every concurrent query.
    __slots__ = ()


    def __init_subclass__(cls, **kwargs: Any) -> None:
        """
        Verify the synchronous contract of a fusion implementation.

        Parameters
        ----------
        **kwargs:
            Forwarded to ``object.__init_subclass__``.

        Returns
        -------
        None

        Raises
        ------
        StrategyRegistrationError
            If ``fuse`` is declared as a coroutine or async generator
            function.
        """
        super().__init_subclass__(**kwargs)

        fuse = cls.__dict__.get("fuse")
        if fuse is None:
            return
        if inspect.iscoroutinefunction(fuse) or inspect.isasyncgenfunction(fuse):
            raise StrategyRegistrationError(
                f"{cls.__name__}.fuse must be synchronous. Fusion is pure "
                f"arithmetic over ranks; an async signature would conceal "
                f"blocking work on the event loop.",
                strategy_name=cls.__name__,
            )

    @property
    def name(self) -> str:
        """
        Stable identifier of this fusion strategy.

        Returns
        -------
        str
            Implementation identifier.
        """
        raise NotImplementedError

    def fuse(
        self,
        results: Sequence[ArmResult],
        top_k: int,
    ) -> Sequence[FusedCandidate]:
        """
        Combine several arms' orderings into one ranking.

        Parameters
        ----------
        results:
            One result per arm that executed. An arm that failed is absent,
            so fusion operates on what succeeded rather than on placeholder
            entries.
        top_k:
            Largest number of results to return.

        Returns
        -------
        Sequence[FusedCandidate]
            Deduplicated results in fused order, each retaining every signal
            that contributed to it.

        Raises
        ------
        IncomparableSignalsError
            If the strategy combines scores and was given incomparable
            scales. Rank-based fusion cannot raise this, which is why it is
            the default.
        """
        raise NotImplementedError


@runtime_checkable
class ContextExpander(Protocol):
    """
    Adds surrounding context to retrieved results.

    Expansion is what replaces overlapping chunks. Overlap was removed from
    Knowledge Processing because duplicated text enters verification as two
    evidence items, making a claim appear better corroborated than the corpus
    warrants. Expansion supplies the same context by reference instead.

    That only holds if expanded nodes are never counted as independent
    findings. An expander therefore attaches context TO an anchor and never
    produces a new one. An expanded node that also happens to have been
    retrieved independently remains a single result carrying both discovery
    records, so it is counted once.

    Expansion must be deterministic and bounded. Unbounded expansion would
    recreate the index inflation that overlap caused, and a nondeterministic
    one would make a query irreproducible.
    """

    @property
    def name(self) -> str:
        """
        Stable identifier of this expander.

        Returns
        -------
        str
            Implementation identifier.
        """
        ...

    async def expand(
        self,
        anchors: Sequence[KnowledgeNode],
        partition: SegmentationPartition,
    ) -> Mapping[str, Sequence[KnowledgeNode]]:
        """
        Find the context accompanying each anchor.

        Takes HYDRATED anchors rather than fused candidates, because a node's
        neighbours are determined by its document and ordinal, and only a
        hydrated node carries those. An expander given identifiers alone would
        have to resolve them a second time, duplicating work hydration has
        already done and opening the possibility that the two resolutions
        disagree if the store changed between them.

        Parameters
        ----------
        anchors:
            The independently retrieved nodes, in fused order. Order is
            significant: it determines which anchor a shared context node is
            assigned to.
        partition:
            The segmentation partition to resolve within.

        Returns
        -------
        Mapping[str, Sequence[KnowledgeNode]]
            Context nodes keyed by the anchor identifier they accompany.

            Two guarantees hold across the whole mapping, and the implementation
            is responsible for both because only it sees every anchor at once:

            A node appears at most ONCE. A node adjacent to two anchors is
            assigned to exactly one of them, deterministically, so that it
            cannot be emitted twice and counted twice.

            No anchor appears as another anchor's context. An anchor was found
            independently; listing it as context would either duplicate it or
            demote a finding to a companion.

        Raises
        ------
        ExpansionError
            If context could not be resolved. Degradable: the anchors remain
            complete evidence without it.
        """
        ...


@runtime_checkable
class CredibilityProvider(Protocol):
    """
    Supplies source credibility, ordinarily from a domain plugin.

    The one domain-aware component in the query path. What makes a source
    credible is domain knowledge — a regulatory filing, a peer-reviewed
    journal, an internal wiki, and a preprint are not weighed alike, and none
    of that belongs in a generic retrieval engine.

    Credibility describes a SOURCE, not a claim. It is an input to retrieval
    reliability; nothing in this module judges whether a claim is true.

    An unavailable value must be reported as unavailable and never defaulted.
    Retrieval reliability carries a statistical guarantee, and substituting an
    invented number would make it indistinguishable from a measured one.
    """

    @property
    def name(self) -> str:
        """
        Stable identifier of this provider.

        Returns
        -------
        str
            Implementation identifier.
        """
        ...

    async def credibility_for(
        self,
        source_uris: Sequence[str],
    ) -> Mapping[str, float]:
        """
        Report credibility for each source.

        Parameters
        ----------
        source_uris:
            The sources to assess.

        Returns
        -------
        Mapping[str, float]
            Credibility in ``[0, 1]`` per source. A source absent from the
            result has no available credibility, which the caller records as
            an absence rather than replacing with a default.

        Raises
        ------
        CredibilityUnavailableError
            If credibility could not be obtained.
        """
        ...


# ============================================================================
# LIFECYCLE
# ============================================================================


@runtime_checkable
class QueryPreprocessor(Protocol):
    """
    Normalizes a query for a domain before it is searched.

    The second domain boundary of this module, and with source credibility the
    only other point at which domain knowledge enters retrieval.

    Declared here rather than reached through a plugin base class, so that the
    Core imports no plugin. What normalization means is domain knowledge:
    expanding an abbreviation, resolving a symbol to an identifier, or
    normalizing a code are each specific to a corpus, and the retrieval engine
    performs none of them. It knows the result only as text.

    A deployment with no domain rules supplies none, and the engine searches
    the query as written.
    """

    @property
    def name(self) -> str:
        """
        Stable identifier of this preprocessor.

        Recorded in diagnostics, so that a query whose text was rewritten is
        distinguishable from one searched as written.

        Returns
        -------
        str
            Implementation identifier.
        """
        ...

    async def preprocess(self, query: str) -> str:
        """
        Normalize a query for the domain.

        Parameters
        ----------
        query:
            The query text as the caller supplied it.

        Returns
        -------
        str
            The normalized query. Returning the input unchanged is valid, and
            is what a domain with no normalization rules does.

        Raises
        ------
        QueryEncodingError
            If normalization failed. Degradable: a query that could not be
            normalized can still be searched as written, which is a worse
            query rather than no query.
        """
        ...


class _AsyncCloseable(Protocol):
    """
    Common shape of a component owning resources for the life of a service.

    Declared for documentation rather than for structural checking: stores
    and indexes hold connection pools that must be released on shutdown and
    on failure alike.
    """

    async def close(self) -> None:
        """
        Release resources held by this component. Must be idempotent.

        Returns
        -------
        None
        """
        ...

    async def __aenter__(self) -> Any:
        """
        Enter an asynchronous context managing this component's lifetime.

        Returns
        -------
        Any
            This component.
        """
        ...

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """
        Close the component on context exit, including on failure.

        Parameters
        ----------
        exc_type:
            Type of the propagating exception, if any.
        exc:
            The propagating exception, if any.
        traceback:
            Traceback of the propagating exception, if any.

        Returns
        -------
        None
        """
        ...


def _assert_exports(names: Sequence[str]) -> None:
    """
    Verify at import time that every exported name resolves.

    Guards against an entry in ``__all__`` drifting away from a definition
    during refactoring, which would otherwise surface as an ``AttributeError``
    in an unrelated module.

    Parameters
    ----------
    names:
        Exported names to verify.

    Returns
    -------
    None

    Raises
    ------
    StoreRegistrationError
        If an exported name is not defined in this module.
    """
    missing = [name for name in names if name not in globals()]
    if missing:
        raise StoreRegistrationError(
            f"ptfm.core.retrieval.interfaces exports undefined names: "
            f"{missing}"
        )


_assert_exports(__all__)
