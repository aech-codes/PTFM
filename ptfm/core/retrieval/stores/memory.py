"""
PTFM In-Memory Retrieval Stores

File:
    ptfm/core/retrieval/stores/memory.py

Description:
    Complete, working implementations of every retrieval store, holding their
    data in process memory.

    Together they make the whole query path executable with no database, no
    service, and no network, which is what allows the pipeline stages to be
    verified against real behaviour rather than against doubles that agree with
    whatever the caller expects.

Real implementations, not stand-ins
-----------------------------------
    Nothing here is a placeholder. Lexical search is Okapi BM25 with the
    standard parameters, vector search is exact cosine similarity, filters are
    evaluated against the same node attributes a database backend would index,
    and writes are genuinely atomic.

    These are therefore reference implementations as well as test fixtures: they
    define what a backend has to do, and a backend that disagrees with them
    about a query's result is wrong about the contract rather than merely
    different.

NOT FOR PRODUCTION USE
----------------------
    Every store grows without bound and nothing survives a restart. A corpus of
    any size will exhaust memory, and exact search is linear in corpus size, so
    query latency grows with the corpus rather than with the result.

    The limits are properties of holding a corpus in a process, not defects to
    be fixed here. A deployment uses a persistent backend behind the same
    protocols.

Partitions isolate absolutely
-----------------------------
    Every store keys its data by partition. A query against one partition cannot
    observe another's data by any route, so mixing incomparable vectors or
    inconsistent spans is not prevented by a check that could be bypassed: it is
    unreachable.

Dependency contract
-------------------
    Standard library only, plus the retrieval ``exceptions``, ``models`` and
    ``interfaces`` modules and the shared schemas. No third-party dependency of
    any kind, so these stores are importable wherever the Core is.
"""

from __future__ import annotations

import asyncio
import math
from collections import defaultdict
from typing import Final, Iterable, Mapping, Optional, Sequence

from ptfm.core.retrieval.exceptions import (
    EmbeddingSpaceMismatchError,
    IndexWriteError,
    PartialIndexWriteError,
    SegmentationMismatchError,
    StoreUnavailableError,
)
from ptfm.core.retrieval.models import (
    Candidate,
    EmbeddingPartition,
    FilterField,
    FilterOperator,
    RetrievalFilter,
    SegmentationPartition,
)
from ptfm.core.retrieval.stores.tokenization import tokenize
from ptfm.core.schemas import (
    DiscoveryMethod,
    KnowledgeNode,
    NodeEmbedding,
    NodeRelation,
    SignalScale,
)

__all__ = [
    "InMemoryNodeStore",
    "InMemoryVectorIndex",
    "InMemoryLexicalIndex",
    "InMemoryRelationStore",
    "InMemorySupersessionStore",
    "InMemoryPartitionCatalog",
    "InMemoryIndexWriter",
    "matches_filters",
]

#: BM25 term-frequency saturation. The standard value; higher lets repeated
#: terms keep accumulating weight, lower saturates sooner.
_BM25_K1: Final[float] = 1.2

#: BM25 length normalization. The standard value; zero would ignore document
#: length entirely and favour long documents that contain everything.
_BM25_B: Final[float] = 0.75

#: Tokenizer. Word characters only, so punctuation does not become a term and
#: a query's trailing question mark cannot fail to match.


def _tokens(text: str) -> list[str]:
    """
    Split text into lowercase terms.

    Delegates to the shared term rule rather than defining one. Every lexical
    backend must match on the same terms: a rule that differed here would make
    these stores select different nodes than the backend they are the reference
    for, which is a difference in what a query means rather than in how it is
    ranked.

    Parameters
    ----------
    text:
        The text to tokenize.

    Returns
    -------
    list[str]
        Lowercase terms in order of appearance.
    """
    return list(tokenize(text))


# ============================================================================
# FILTER EVALUATION
# ============================================================================


def matches_filters(
    node: KnowledgeNode,
    filters: Sequence[RetrievalFilter],
) -> bool:
    """
    Report whether a node satisfies every filter.

    One implementation, shared by all three searching stores. Three copies would
    drift, and a filter that meant one thing to the vector index and another to
    the lexical index would make a hybrid query return a set no single arm
    agreed with.

    Filters combine conjunctively: a node must satisfy all of them. Disjunction
    is expressed within a filter by ``IN``, which keeps the combinator single
    and a backend's translation of it unambiguous.

    Parameters
    ----------
    node:
        The node to test.
    filters:
        The constraints. An empty sequence matches everything.

    Returns
    -------
    bool
        ``True`` when the node satisfies every filter.
    """
    return all(_matches(node, constraint) for constraint in filters)


def _matches(node: KnowledgeNode, constraint: RetrievalFilter) -> bool:
    """
    Report whether a node satisfies one filter.

    Parameters
    ----------
    node:
        The node to test.
    constraint:
        The constraint to apply.

    Returns
    -------
    bool
        ``True`` when the node satisfies it.
    """
    if constraint.field is FilterField.HEADING_PATH:
        return _matches_sequence(node.heading_path, constraint)

    if constraint.field is FilterField.PAGE_NUMBER:
        return any(
            _matches_scalar(str(page), constraint) for page in node.page_numbers
        ) if node.page_numbers else constraint.operator in _NEGATIVE

    if constraint.field is FilterField.METADATA:
        key = constraint.metadata_key or ""
        if key not in node.metadata:
            # A node lacking the key satisfies a negative constraint and fails a
            # positive one. Treating absence as a match either way would make
            # NOT_EQUALS and EQUALS both true for it.
            return constraint.operator in _NEGATIVE
        return _matches_scalar(node.metadata[key], constraint)

    return _matches_scalar(_attribute(node, constraint.field), constraint)


def _attribute(node: KnowledgeNode, field: FilterField) -> str:
    """
    Read the scalar node attribute a filter field names.

    Parameters
    ----------
    node:
        The node to read.
    field:
        The field to read.

    Returns
    -------
    str
        The attribute rendered as a string, since filter values are strings so
        that a setting from configuration and one from an environment variable
        are indistinguishable.
    """
    if field is FilterField.DOCUMENT_ID:
        return node.document_id
    if field is FilterField.SOURCE_URI:
        return node.source.source_uri
    if field is FilterField.NODE_TYPE:
        return str(node.node_type.value)
    if field is FilterField.ORDINAL:
        return str(node.ordinal)
    return ""


#: Operators whose truth is satisfied by an attribute the node does not carry.
_NEGATIVE: Final[frozenset[FilterOperator]] = frozenset(
    {FilterOperator.NOT_EQUALS, FilterOperator.NOT_IN}
)


def _matches_scalar(value: str, constraint: RetrievalFilter) -> bool:
    """
    Apply an operator to a single attribute value.

    Parameters
    ----------
    value:
        The attribute value.
    constraint:
        The constraint to apply.

    Returns
    -------
    bool
        ``True`` when the value satisfies the operator.
    """
    values = constraint.values

    if constraint.operator is FilterOperator.EQUALS:
        return value == values[0]
    if constraint.operator is FilterOperator.NOT_EQUALS:
        return value != values[0]
    if constraint.operator is FilterOperator.IN:
        return value in values
    if constraint.operator is FilterOperator.NOT_IN:
        return value not in values
    if constraint.operator is FilterOperator.STARTS_WITH:
        return value.startswith(values[0])

    numeric = _as_number(value)
    bound = _as_number(values[0])
    if numeric is None or bound is None:
        # An ordered comparison against something that is not a number is not
        # merely false, it is meaningless. Reporting no match keeps the store
        # from inventing an ordering over strings that a backend would not
        # reproduce.
        return False

    if constraint.operator is FilterOperator.GREATER_THAN_OR_EQUAL:
        return numeric >= bound
    return numeric <= bound


def _matches_sequence(
    sequence: Sequence[str],
    constraint: RetrievalFilter,
) -> bool:
    """
    Apply an operator to a sequence-valued attribute.

    ``STARTS_WITH`` compares element-wise, which is how a heading path selects a
    section together with everything beneath it.

    Parameters
    ----------
    sequence:
        The attribute's elements, in order.
    constraint:
        The constraint to apply.

    Returns
    -------
    bool
        ``True`` when the sequence satisfies the operator.
    """
    values = constraint.values

    if constraint.operator is FilterOperator.STARTS_WITH:
        return (
            len(sequence) >= len(values)
            and tuple(sequence[: len(values)]) == tuple(values)
        )
    if constraint.operator is FilterOperator.EQUALS:
        return tuple(sequence) == tuple(values)
    if constraint.operator is FilterOperator.NOT_EQUALS:
        return tuple(sequence) != tuple(values)
    if constraint.operator is FilterOperator.IN:
        return any(element in values for element in sequence)
    if constraint.operator is FilterOperator.NOT_IN:
        return all(element not in values for element in sequence)

    return False


def _as_number(value: str) -> Optional[float]:
    """
    Interpret a filter value as a number, if it is one.

    Parameters
    ----------
    value:
        The value to interpret.

    Returns
    -------
    Optional[float]
        The number, or ``None`` when the value is not numeric.
    """
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


# ============================================================================
# NODE STORE
# ============================================================================


class InMemoryNodeStore:
    """
    Authoritative node store held in process memory.

    The source of every node's text, span, and provenance. Satisfies the
    ``NodeStore`` protocol.

    Nodes are resolvable regardless of whether their documents are current, so a
    citation recorded before a document was superseded still resolves. Nothing
    is ever removed.

    NOT FOR PRODUCTION USE: unbounded and non-durable.
    """

    __slots__ = ("_by_id", "_by_ordinal", "_lock")

    def __init__(self) -> None:
        self._by_id: dict[str, dict[str, KnowledgeNode]] = defaultdict(dict)
        self._by_ordinal: dict[str, dict[tuple[str, int], KnowledgeNode]] = (
            defaultdict(dict)
        )
        self._lock = asyncio.Lock()

    @property
    def name(self) -> str:
        """
        Stable identifier of this store.

        Returns
        -------
        str
            Implementation identifier.
        """
        return "in-memory-node"

    async def get_many(
        self,
        node_ids: Sequence[str],
        partition: SegmentationPartition,
    ) -> Mapping[str, KnowledgeNode]:
        """
        Resolve node identifiers within a partition.

        Parameters
        ----------
        node_ids:
            Identifiers to resolve.
        partition:
            The segmentation partition to resolve within.

        Returns
        -------
        Mapping[str, KnowledgeNode]
            Resolved nodes by identifier. An identifier the partition does not
            hold is omitted, leaving the caller to decide whether that is a
            fault.
        """
        held = self._by_id[partition.key]
        return {
            node_id: held[node_id] for node_id in node_ids if node_id in held
        }

    async def get_by_ordinal(
        self,
        document_id: str,
        ordinals: Sequence[int],
        partition: SegmentationPartition,
    ) -> Mapping[int, KnowledgeNode]:
        """
        Resolve nodes by position within a document.

        Parameters
        ----------
        document_id:
            The document whose nodes are sought.
        ordinals:
            Positions to resolve.
        partition:
            The segmentation partition to resolve within.

        Returns
        -------
        Mapping[int, KnowledgeNode]
            Resolved nodes by ordinal. A position past a document's last node is
            omitted, which is ordinary at a document's edges.
        """
        held = self._by_ordinal[partition.key]
        return {
            ordinal: held[(document_id, ordinal)]
            for ordinal in ordinals
            if (document_id, ordinal) in held
        }

    async def put_many(
        self,
        nodes: Sequence[KnowledgeNode],
        partition: SegmentationPartition,
    ) -> None:
        """
        Store nodes in a partition.

        Parameters
        ----------
        nodes:
            The nodes to store.
        partition:
            The partition to store them in.

        Returns
        -------
        None

        Raises
        ------
        SegmentationMismatchError
            If a node was segmented under different parameters than the
            partition holds. Accepting it would place spans computed against
            one division of a document beside spans computed against another.
        """
        for node in nodes:
            if not partition.accepts(
                node.segmentation_fingerprint, node.source_schema_version
            ):
                raise SegmentationMismatchError(
                    expected=partition.segmentation_fingerprint,
                    actual=node.segmentation_fingerprint,
                    context={
                        "node_id": node.node_id,
                        "partition_key": partition.key,
                    },
                )

        async with self._lock:
            by_id = self._by_id[partition.key]
            by_ordinal = self._by_ordinal[partition.key]
            for node in nodes:
                by_id[node.node_id] = node
                by_ordinal[(node.document_id, node.ordinal)] = node

    async def count(self, partition: SegmentationPartition) -> int:
        """
        Report how many nodes a partition holds.

        Parameters
        ----------
        partition:
            The partition to measure.

        Returns
        -------
        int
            Node count.
        """
        return len(self._by_id[partition.key])

    async def node_ids(self, partition: SegmentationPartition) -> tuple[str, ...]:
        """
        List every node identifier a partition holds, in sorted order.

        Beyond the ``NodeStore`` protocol, which has no reason to offer a full
        scan: a database backend answers a structural query with an indexed
        lookup rather than by enumerating a corpus. Provided here because
        structural search has no index to use, and sorted so that a scan is
        reproducible.

        Parameters
        ----------
        partition:
            The partition to enumerate.

        Returns
        -------
        tuple[str, ...]
            Node identifiers in sorted order.
        """
        return tuple(sorted(self._by_id[partition.key]))

    def __repr__(self) -> str:
        """
        Return a representation naming the partitions held.

        Returns
        -------
        str
            Diagnostic representation.
        """
        return f"InMemoryNodeStore(partitions={len(self._by_id)})"


# ============================================================================
# VECTOR INDEX
# ============================================================================


class InMemoryVectorIndex:
    """
    Exact nearest-neighbour search over vectors held in memory.

    Satisfies the ``VectorIndex`` protocol.

    Always exact, never approximate. Every query compares against every vector
    in the partition, so a result is reproducible across processes and rebuilds,
    which makes this the reference against which an approximate backend's recall
    can be measured.

    NOT FOR PRODUCTION USE: search is linear in corpus size.
    """

    __slots__ = ("_vectors", "_nodes", "_lock")

    def __init__(self, node_store: InMemoryNodeStore) -> None:
        self._vectors: dict[str, dict[str, NodeEmbedding]] = defaultdict(dict)
        # Filters constrain node attributes, so the index needs the nodes to
        # evaluate them. A database backend has both in one place; holding a
        # reference is how that is expressed in memory.
        self._nodes = node_store
        self._lock = asyncio.Lock()

    @property
    def name(self) -> str:
        """
        Stable identifier of this index.

        Returns
        -------
        str
            Implementation identifier.
        """
        return "in-memory-vector"

    @property
    def supports_exact_search(self) -> bool:
        """
        Whether exact search is available.

        Returns
        -------
        bool
            Always ``True``. This index has no approximate mode, so a caller
            requiring reproducibility is always served.
        """
        return True

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
        Find the nearest vectors to a query vector.

        Parameters
        ----------
        vector:
            The query vector, encoded for this partition.
        partition:
            The embedding partition to search.
        top_k:
            Largest number of results to return.
        filters:
            Structural constraints, applied during search so that a filtered
            query still returns the requested number where the corpus allows.
        exact:
            Accepted for contract conformance. This index is always exact, so
            the flag changes nothing rather than being ignored silently.

        Returns
        -------
        Sequence[Candidate]
            Nodes in descending similarity, ranks contiguous from zero. Ties
            break on identifier, so the ordering is reproducible.

        Raises
        ------
        EmbeddingSpaceMismatchError
            If the query vector's width differs from the partition's.
        StoreUnavailableError
            If the partition's nodes cannot be resolved for filtering.
        """
        if len(vector) != partition.dimension:
            raise EmbeddingSpaceMismatchError(
                f"Query vector has width {len(vector)} but partition "
                f"{partition.key!r} holds vectors of width "
                f"{partition.dimension}.",
                expected_identity=partition.model_identity,
                context={"partition_key": partition.key},
            )

        held = self._vectors[partition.key]
        if not held or top_k < 1:
            return ()

        permitted = await self._permitted(held, partition, filters)

        scored = [
            (
                _similarity(vector, embedding.vector, partition.normalized),
                node_id,
            )
            for node_id, embedding in held.items()
            if node_id in permitted
        ]

        # Descending similarity, then ascending identifier. The identifier
        # tie-break is what makes two equally similar nodes order the same way
        # on every run and in every process.
        scored.sort(key=lambda item: (-item[0], item[1]))

        return tuple(
            Candidate(
                node_id=node_id,
                method=DiscoveryMethod.DENSE,
                rank=rank,
                score=score,
                scale=SignalScale.COSINE,
            )
            for rank, (score, node_id) in enumerate(scored[:top_k])
        )

    async def upsert(
        self,
        embeddings: Sequence[NodeEmbedding],
        partition: EmbeddingPartition,
    ) -> None:
        """
        Store vectors in a partition.

        Parameters
        ----------
        embeddings:
            The vectors to store.
        partition:
            The partition to store them in.

        Returns
        -------
        None

        Raises
        ------
        EmbeddingSpaceMismatchError
            If a vector does not belong to the partition's space. Storing it
            would make every later comparison against it meaningless while
            reporting success.
        """
        for embedding in embeddings:
            if (
                embedding.model_identity != partition.model_identity
                or embedding.dimension != partition.dimension
                or embedding.normalized != partition.normalized
                or embedding.composition_id != partition.composition_id
                or embedding.query_encoding != partition.query_encoding
            ):
                raise EmbeddingSpaceMismatchError(
                    f"Embedding for node {embedding.node_id!r} does not belong "
                    f"to partition {partition.key!r}. A similarity computed "
                    f"across two spaces is arithmetically valid and "
                    f"semantically meaningless.",
                    expected_identity=partition.model_identity,
                    actual_identity=embedding.model_identity,
                    context={"node_id": embedding.node_id},
                )

        async with self._lock:
            held = self._vectors[partition.key]
            for embedding in embeddings:
                held[embedding.node_id] = embedding

    async def count(self, partition: EmbeddingPartition) -> int:
        """
        Report how many vectors a partition holds.

        Parameters
        ----------
        partition:
            The partition to measure.

        Returns
        -------
        int
            Vector count.
        """
        return len(self._vectors[partition.key])

    async def _permitted(
        self,
        held: Mapping[str, NodeEmbedding],
        partition: EmbeddingPartition,
        filters: Sequence[RetrievalFilter],
    ) -> frozenset[str]:
        """
        Determine which of the partition's nodes the filters permit.

        Parameters
        ----------
        held:
            The vectors in the partition.
        partition:
            The embedding partition being searched.
        filters:
            The constraints.

        Returns
        -------
        frozenset[str]
            Identifiers a filtered search may return.

        Raises
        ------
        StoreUnavailableError
            If a vector's node cannot be resolved, which means the vector index
            and the node store have diverged.
        """
        if not filters:
            return frozenset(held)

        nodes = await self._nodes.get_many(
            tuple(held), partition.segmentation
        )
        missing = set(held) - set(nodes)
        if missing:
            raise StoreUnavailableError(
                f"{len(missing)} vectors in partition {partition.key!r} have no "
                f"node, so a filtered search cannot establish whether they are "
                f"permitted.",
                store_name=self.name,
                context={"missing_node_ids": sorted(missing)[:10]},
            )

        return frozenset(
            node_id
            for node_id, node in nodes.items()
            if matches_filters(node, filters)
        )

    def __repr__(self) -> str:
        """
        Return a representation naming the partitions held.

        Returns
        -------
        str
            Diagnostic representation.
        """
        return f"InMemoryVectorIndex(partitions={len(self._vectors)})"


def _similarity(
    query: Sequence[float],
    stored: Sequence[float],
    normalized: bool,
) -> float:
    """
    Compute cosine similarity between two vectors.

    When the partition records its vectors as normalized, the dot product IS the
    cosine, and computing magnitudes would spend effort to divide by one. When
    it does not, the magnitudes are computed, because a dot product over
    unnormalized vectors is not a similarity and treating it as one would rank
    long vectors above relevant ones.

    Parameters
    ----------
    query:
        The query vector.
    stored:
        The stored vector.
    normalized:
        Whether the partition's vectors have unit magnitude.

    Returns
    -------
    float
        Cosine similarity in ``[-1, 1]``, or zero where a vector has no
        magnitude and no direction to compare.
    """
    dot = sum(left * right for left, right in zip(query, stored))

    if normalized:
        return dot

    query_magnitude = math.sqrt(sum(value * value for value in query))
    stored_magnitude = math.sqrt(sum(value * value for value in stored))

    if query_magnitude == 0.0 or stored_magnitude == 0.0:
        return 0.0

    return dot / (query_magnitude * stored_magnitude)


# ============================================================================
# LEXICAL INDEX
# ============================================================================


class InMemoryLexicalIndex:
    """
    Okapi BM25 search over node text held in memory.

    Satisfies the ``LexicalIndex`` protocol.

    BM25 rather than a term-overlap count, because the arm exists to find exact
    matches on identifiers, quantities, and rare proper nouns, and that depends
    on inverse document frequency weighting a rare term above a common one. A
    scorer without it would rank a node containing many common query words above
    one containing the single term that mattered.

    Scores are corpus-relative and unbounded, which is why they are never
    compared against a similarity.

    NOT FOR PRODUCTION USE: unbounded, non-durable, and with no configurable
    analysis.
    """

    __slots__ = ("_postings", "_lengths", "_terms", "_nodes", "_lock")

    def __init__(self, node_store: InMemoryNodeStore) -> None:
        self._postings: dict[str, dict[str, dict[str, int]]] = defaultdict(
            lambda: defaultdict(dict)
        )
        self._lengths: dict[str, dict[str, int]] = defaultdict(dict)
        self._terms: dict[str, int] = {}
        self._nodes = node_store
        self._lock = asyncio.Lock()

    @property
    def name(self) -> str:
        """
        Stable identifier of this index.

        Returns
        -------
        str
            Implementation identifier.
        """
        return "in-memory-lexical"

    async def search(
        self,
        query: str,
        partition: SegmentationPartition,
        top_k: int,
        *,
        filters: Sequence[RetrievalFilter] = (),
    ) -> Sequence[Candidate]:
        """
        Find nodes matching a query's terms.

        Implements Okapi BM25:

            score(q, d) = sum over terms t in q of
                IDF(t) * ( f(t,d) * (k1 + 1) )
                        / ( f(t,d) + k1 * (1 - b + b * |d| / avgdl) )

        with

            IDF(t) = ln( 1 + (N - n(t) + 0.5) / (n(t) + 0.5) )

        where ``f(t,d)`` is the term's frequency in the node, ``|d|`` the node's
        length in terms, ``avgdl`` the partition's mean node length, ``N`` the
        node count, and ``n(t)`` the number of nodes containing the term.

        Parameters
        ----------
        query:
            Query text, after any domain normalization.
        partition:
            The segmentation partition to search.
        top_k:
            Largest number of results to return.
        filters:
            Structural constraints applied during search.

        Returns
        -------
        Sequence[Candidate]
            Nodes in descending relevance, ranks contiguous from zero. Ties
            break on identifier. A node scoring zero is omitted rather than
            returned, since it shares no term with the query.
        """
        terms = _tokens(query)
        lengths = self._lengths[partition.key]

        if not terms or not lengths or top_k < 1:
            return ()

        permitted = await self._permitted(lengths, partition, filters)
        if not permitted:
            return ()

        postings = self._postings[partition.key]
        total = len(lengths)
        average = sum(lengths.values()) / total

        scores: dict[str, float] = defaultdict(float)
        for term in terms:
            holders = postings.get(term)
            if not holders:
                continue
            idf = math.log(
                1.0 + (total - len(holders) + 0.5) / (len(holders) + 0.5)
            )
            for node_id, frequency in holders.items():
                if node_id not in permitted:
                    continue
                length = lengths[node_id]
                denominator = frequency + _BM25_K1 * (
                    1.0 - _BM25_B + _BM25_B * length / average
                )
                scores[node_id] += idf * frequency * (_BM25_K1 + 1.0) / denominator

        ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))

        return tuple(
            Candidate(
                node_id=node_id,
                method=DiscoveryMethod.SPARSE,
                rank=rank,
                score=score,
                scale=SignalScale.LEXICAL,
            )
            for rank, (node_id, score) in enumerate(ranked[:top_k])
            if score > 0.0
        )

    async def index_nodes(
        self,
        nodes: Sequence[KnowledgeNode],
        partition: SegmentationPartition,
    ) -> None:
        """
        Index nodes for term search.

        The heading path is indexed alongside the node's own text, because a
        section title describes its content and a query naming the section
        should reach the nodes inside it.

        Parameters
        ----------
        nodes:
            The nodes to index.
        partition:
            The partition to index them in.

        Returns
        -------
        None
        """
        async with self._lock:
            postings = self._postings[partition.key]
            lengths = self._lengths[partition.key]

            for node in nodes:
                terms = _tokens(node.text) + _tokens(" ".join(node.heading_path))

                # Re-indexing a node must replace its postings, not add to them,
                # or a term's frequency would accumulate across writes and the
                # node would climb the ranking by being written twice.
                previous = [
                    term
                    for term, holders in postings.items()
                    if node.node_id in holders
                ]
                for term in previous:
                    del postings[term][node.node_id]
                    if not postings[term]:
                        del postings[term]

                counts: dict[str, int] = defaultdict(int)
                for term in terms:
                    counts[term] += 1
                for term, frequency in counts.items():
                    postings[term][node.node_id] = frequency

                lengths[node.node_id] = len(terms)

    async def _permitted(
        self,
        lengths: Mapping[str, int],
        partition: SegmentationPartition,
        filters: Sequence[RetrievalFilter],
    ) -> frozenset[str]:
        """
        Determine which indexed nodes the filters permit.

        Parameters
        ----------
        lengths:
            The indexed nodes' lengths, keyed by identifier.
        partition:
            The partition being searched.
        filters:
            The constraints.

        Returns
        -------
        frozenset[str]
            Identifiers a filtered search may return.
        """
        if not filters:
            return frozenset(lengths)

        nodes = await self._nodes.get_many(tuple(lengths), partition)
        return frozenset(
            node_id
            for node_id, node in nodes.items()
            if matches_filters(node, filters)
        )

    def __repr__(self) -> str:
        """
        Return a representation naming the partitions held.

        Returns
        -------
        str
            Diagnostic representation.
        """
        return f"InMemoryLexicalIndex(partitions={len(self._lengths)})"


# ============================================================================
# RELATION STORE
# ============================================================================


class InMemoryRelationStore:
    """
    Structural relations and structural search over data held in memory.

    Satisfies the ``RelationStore`` protocol.

    Structural search evaluates filters against node attributes, so this store
    holds a reference to the node store. A database backend has relations and
    nodes in one place; the reference is how that is expressed in memory, and it
    is the reason structural search can live on this protocol at all.

    NOT FOR PRODUCTION USE: unbounded and non-durable.
    """

    __slots__ = ("_relations", "_nodes", "_lock")

    def __init__(self, node_store: InMemoryNodeStore) -> None:
        self._relations: dict[str, dict[str, list[NodeRelation]]] = defaultdict(
            lambda: defaultdict(list)
        )
        self._nodes = node_store
        self._lock = asyncio.Lock()

    @property
    def name(self) -> str:
        """
        Stable identifier of this store.

        Returns
        -------
        str
            Implementation identifier.
        """
        return "in-memory-relation"

    async def relations_from(
        self,
        node_ids: Sequence[str],
        partition: SegmentationPartition,
        *,
        relation_types: Sequence[str] = (),
    ) -> Mapping[str, Sequence[NodeRelation]]:
        """
        Find the relations originating from given nodes.

        Parameters
        ----------
        node_ids:
            Nodes whose outgoing relations are sought.
        partition:
            The segmentation partition to search.
        relation_types:
            Types to include. Empty includes every type.

        Returns
        -------
        Mapping[str, Sequence[NodeRelation]]
            Relations by originating node. A node with no relations is omitted.
        """
        held = self._relations[partition.key]
        wanted = frozenset(relation_types)

        found: dict[str, tuple[NodeRelation, ...]] = {}
        for node_id in node_ids:
            relations = held.get(node_id, ())
            if wanted:
                relations = [
                    relation
                    for relation in relations
                    if str(relation.relation_type.value) in wanted
                ]
            if relations:
                found[node_id] = tuple(relations)

        return found

    async def structural_search(
        self,
        filters: Sequence[RetrievalFilter],
        partition: SegmentationPartition,
        top_k: int,
    ) -> Sequence[Candidate]:
        """
        Find nodes satisfying a structural constraint.

        Returns matches in document and reading order, all at rank zero and
        with no score.

        A structural constraint implies no relevance ordering: the rows of a
        table have no intrinsic rank with respect to a query, and neither do the
        nodes under a heading. Reporting no score says so, where a fabricated
        one would be indistinguishable from a measured one by the time it
        reached fusion.

        Parameters
        ----------
        filters:
            The structural constraints. Empty returns nothing rather than the
            corpus, since without a constraint there is no question to answer.
        partition:
            The segmentation partition to search.
        top_k:
            Largest number of results to return.

        Returns
        -------
        Sequence[Candidate]
            Matching nodes, ordered by document and then position so that the
            same constraint returns the same set in the same order on every run.
        """
        if not filters or top_k < 1:
            return ()

        held = await self._nodes.get_many(
            await self._nodes.node_ids(partition), partition
        )

        matching = sorted(
            (
                node
                for node in held.values()
                if matches_filters(node, filters)
            ),
            key=lambda node: (node.document_id, node.ordinal, node.node_id),
        )

        return tuple(
            Candidate(
                node_id=node.node_id,
                method=DiscoveryMethod.GRAPH,
                # Position in the deterministic ordering above, not a relevance
                # rank. A structural match is a match, which is why the scale
                # stays RANK_ONLY and no score is attached.
                #
                # Enumerated rather than left at zero. Rank-based fusion reads
                # these as an ordering: giving every match rank 0 would make
                # each one tie at the top and receive the maximum reciprocal
                # rank, so a filter matching a thousand nodes would outweigh
                # every dense and lexical result at once.
                rank=rank,
                score=None,
                scale=SignalScale.RANK_ONLY,
            )
            for rank, node in enumerate(matching[:top_k])
        )

    async def put_many(
        self,
        relations: Sequence[NodeRelation],
        partition: SegmentationPartition,
    ) -> None:
        """
        Store relations in a partition.

        Parameters
        ----------
        relations:
            The relations to store.
        partition:
            The partition to store them in.

        Returns
        -------
        None
        """
        async with self._lock:
            held = self._relations[partition.key]
            for relation in relations:
                existing = held[relation.source_node_id]
                # Relations are re-derived whenever a document is reprocessed,
                # so a rewrite must not accumulate duplicates that would make a
                # traversal visit the same target repeatedly.
                if not any(
                    other.target_node_id == relation.target_node_id
                    and other.relation_type == relation.relation_type
                    for other in existing
                ):
                    existing.append(relation)

    def __repr__(self) -> str:
        """
        Return a representation naming the partitions held.

        Returns
        -------
        str
            Diagnostic representation.
        """
        return f"InMemoryRelationStore(partitions={len(self._relations)})"


# ============================================================================
# SUPERSESSION STORE
# ============================================================================


class InMemorySupersessionStore:
    """
    Source currency records held in memory.

    Satisfies the ``SupersessionStore`` protocol.

    Records which document currently represents each source. A source recorded
    with no document is a tombstone: it was withdrawn, which is distinct from
    having no record at all, and the distinction is preserved because collapsing
    it would make a deletion indistinguishable from an omission.

    NOT FOR PRODUCTION USE: unbounded and non-durable.
    """

    __slots__ = ("_current", "_versions", "_lock")

    def __init__(self) -> None:
        self._current: dict[str, dict[str, Optional[str]]] = defaultdict(dict)
        self._versions: dict[str, dict[str, Optional[str]]] = defaultdict(dict)
        self._lock = asyncio.Lock()

    @property
    def name(self) -> str:
        """
        Stable identifier of this store.

        Returns
        -------
        str
            Implementation identifier.
        """
        return "in-memory-supersession"

    async def current_document_ids(
        self,
        source_uris: Sequence[str],
        partition: SegmentationPartition,
    ) -> Mapping[str, Optional[str]]:
        """
        Report which document currently represents each source.

        Parameters
        ----------
        source_uris:
            The sources to look up.
        partition:
            The segmentation partition to consult.

        Returns
        -------
        Mapping[str, Optional[str]]
            Current document identifier by source. A source with a record but no
            document maps to ``None``, marking it withdrawn. A source with no
            record at all is ABSENT from the result, which the caller reads as
            "not known to be superseded" rather than as withdrawn.
        """
        held = self._current[partition.key]
        return {uri: held[uri] for uri in source_uris if uri in held}

    async def record_current(
        self,
        source_uri: str,
        document_id: Optional[str],
        version_tag: Optional[str],
        partition: SegmentationPartition,
    ) -> None:
        """
        Record the document that currently represents a source.

        Parameters
        ----------
        source_uri:
            The source to record.
        document_id:
            The current document, or ``None`` to record a tombstone marking the
            source withdrawn.
        version_tag:
            Source-side version marker, when the connector supplied one. Gives
            supersession an ordering from the source rather than from the order
            documents happened to be ingested.
        partition:
            The partition to record within.

        Returns
        -------
        None
        """
        async with self._lock:
            self._current[partition.key][source_uri] = document_id
            self._versions[partition.key][source_uri] = version_tag

    async def version_tag(
        self,
        source_uri: str,
        partition: SegmentationPartition,
    ) -> Optional[str]:
        """
        Report the recorded version marker for a source.

        Beyond the protocol, and provided for audit: it answers which source-side
        version a currency record was based on, which is what distinguishes a
        supersession the source declared from one inferred from ingest order.

        Parameters
        ----------
        source_uri:
            The source to look up.
        partition:
            The partition to consult.

        Returns
        -------
        Optional[str]
            The version marker, or ``None`` when none was recorded.
        """
        return self._versions[partition.key].get(source_uri)

    def __repr__(self) -> str:
        """
        Return a representation naming the partitions held.

        Returns
        -------
        str
            Diagnostic representation.
        """
        return f"InMemorySupersessionStore(partitions={len(self._current)})"


# ============================================================================
# PARTITION CATALOG
# ============================================================================


class InMemoryPartitionCatalog:
    """
    Records which partitions exist and which serve queries.

    Satisfies the ``PartitionCatalog`` protocol.

    Exactly one segmentation partition is active. Several embedding partitions
    may exist over it, one active and the others being built or retired, which
    is the whole of the migration story: re-embedding changes vectors while
    leaving nodes, spans, identifiers, and lexical terms untouched.

    NOT FOR PRODUCTION USE: non-durable, so activation does not survive a
    restart.
    """

    __slots__ = ("_segmentations", "_embeddings", "_active", "_active_embedding", "_lock")

    def __init__(self) -> None:
        self._segmentations: dict[str, SegmentationPartition] = {}
        self._embeddings: dict[str, dict[str, EmbeddingPartition]] = defaultdict(dict)
        self._active: Optional[SegmentationPartition] = None
        self._active_embedding: dict[str, EmbeddingPartition] = {}
        self._lock = asyncio.Lock()

    @property
    def name(self) -> str:
        """
        Stable identifier of this catalog.

        Returns
        -------
        str
            Implementation identifier.
        """
        return "in-memory-catalog"

    async def active_segmentation(self) -> Optional[SegmentationPartition]:
        """
        Report the segmentation partition serving queries.

        Returns
        -------
        Optional[SegmentationPartition]
            The active partition, or ``None`` when nothing has been activated.
        """
        return self._active

    async def active_embedding(
        self,
        segmentation: SegmentationPartition,
    ) -> Optional[EmbeddingPartition]:
        """
        Report the embedding partition serving queries over a segmentation.

        Parameters
        ----------
        segmentation:
            The segmentation partition whose active embedding is sought.

        Returns
        -------
        Optional[EmbeddingPartition]
            The active embedding partition, or ``None`` when the corpus has been
            segmented but not embedded, which is a valid intermediate state.
        """
        return self._active_embedding.get(segmentation.key)

    async def register(
        self,
        segmentation: SegmentationPartition,
        embedding: Optional[EmbeddingPartition],
        *,
        activate: bool,
    ) -> None:
        """
        Record that a partition exists, and optionally activate it.

        Activation is the atomic switch a migration turns. Registering without
        activating is how a new partition is built alongside the one serving
        queries.

        Parameters
        ----------
        segmentation:
            The segmentation partition.
        embedding:
            The embedding partition over it, if any.
        activate:
            Whether these partitions should serve queries.

        Returns
        -------
        None

        Raises
        ------
        EmbeddingSpaceMismatchError
            If the embedding partition describes a different segmentation than
            the one it is registered under.
        """
        if (
            embedding is not None
            and embedding.segmentation.key != segmentation.key
        ):
            raise EmbeddingSpaceMismatchError(
                f"Embedding partition {embedding.key!r} describes segmentation "
                f"{embedding.segmentation.key!r}, not {segmentation.key!r}. Its "
                f"vectors would be compared against nodes they were not "
                f"produced from.",
                context={"partition_key": embedding.key},
            )

        async with self._lock:
            self._segmentations[segmentation.key] = segmentation
            if embedding is not None:
                self._embeddings[segmentation.key][embedding.key] = embedding

            if activate:
                self._active = segmentation
                if embedding is not None:
                    self._active_embedding[segmentation.key] = embedding

    def __repr__(self) -> str:
        """
        Return a representation naming the active partition.

        Returns
        -------
        str
            Diagnostic representation.
        """
        active = self._active.key if self._active is not None else None
        return (
            f"InMemoryPartitionCatalog(segmentations="
            f"{len(self._segmentations)}, active={active!r})"
        )


# ============================================================================
# INDEX WRITER
# ============================================================================


class InMemoryIndexWriter:
    """
    Writes knowledge units to every store as one atomic operation.

    Satisfies the ``IndexWriter`` protocol, and reports itself atomic because it
    genuinely is: every unit is validated before anything is applied, and the
    application runs to completion without awaiting, so no other coroutine can
    observe a half-written unit.

    That property is the reason this writer exists rather than callers writing to
    each store directly. A unit whose vector reached the index while its node did
    not would be retrievable with no authoritative text: a result that appears
    valid and cannot be traced, which is the worst failure available to an
    evidence system.

    NOT FOR PRODUCTION USE: nothing it writes survives the process.

    Parameters
    ----------
    nodes:
        The node store.
    vectors:
        The vector index.
    lexical:
        The lexical index.
    relations:
        The relation store.
    supersession:
        The supersession store.
    """

    __slots__ = (
        "_nodes",
        "_vectors",
        "_lexical",
        "_relations",
        "_supersession",
        "_closed",
        "_lock",
    )

    def __init__(
        self,
        nodes: InMemoryNodeStore,
        vectors: InMemoryVectorIndex,
        lexical: InMemoryLexicalIndex,
        relations: InMemoryRelationStore,
        supersession: InMemorySupersessionStore,
    ) -> None:
        self._nodes = nodes
        self._vectors = vectors
        self._lexical = lexical
        self._relations = relations
        self._supersession = supersession
        self._closed = False
        self._lock = asyncio.Lock()

    @property
    def name(self) -> str:
        """
        Stable identifier of this writer.

        Returns
        -------
        str
            Implementation identifier.
        """
        return "in-memory-writer"

    @property
    def is_atomic(self) -> bool:
        """
        Whether a failed unit write leaves no store partially written.

        Returns
        -------
        bool
            Always ``True``. Validation precedes application and application
            does not await, so a failure cannot leave a unit half-written.
        """
        return True

    async def write_units(
        self,
        segmentation: SegmentationPartition,
        embedding: Optional[EmbeddingPartition],
        units: Sequence[object],
    ) -> None:
        """
        Write units to every store.

        Validated in full before anything is applied. That ordering is what
        makes the write atomic: a unit that cannot be written is rejected while
        every store is still untouched.

        Parameters
        ----------
        segmentation:
            Partition receiving nodes, lexical entries, relations, and
            supersession records.
        embedding:
            Partition receiving vectors. ``None`` when the units carry none,
            which is a corpus segmented but not yet embedded.
        units:
            Knowledge units, each carrying a node and optionally an embedding
            and relations.

        Returns
        -------
        None

        Raises
        ------
        IndexWriteError
            If the writer is closed, or a unit cannot be written.
        SegmentationMismatchError
            If a node does not belong to the segmentation partition.
        EmbeddingSpaceMismatchError
            If a vector does not belong to the embedding partition.
        PartialIndexWriteError
            Never raised by this writer, which is atomic. Declared because the
            contract permits it and a composing backend does raise it.
        """
        if self._closed:
            raise IndexWriteError(
                f"Writer {self.name!r} is closed and can no longer write.",
                context={"writer": self.name},
            )

        if not units:
            return

        nodes = [getattr(unit, "node") for unit in units]
        embeddings = [
            vector
            for unit in units
            if (vector := getattr(unit, "embedding", None)) is not None
        ]
        relations = [
            relation
            for unit in units
            for relation in getattr(unit, "relations", ())
        ]

        self._validate(segmentation, embedding, nodes, embeddings)

        async with self._lock:
            # Applied without awaiting, so no other coroutine can observe the
            # corpus between the first store and the last.
            self._apply(segmentation, embedding, nodes, embeddings, relations)

    def _validate(
        self,
        segmentation: SegmentationPartition,
        embedding: Optional[EmbeddingPartition],
        nodes: Sequence[KnowledgeNode],
        embeddings: Sequence[NodeEmbedding],
    ) -> None:
        """
        Verify that every unit can be written before any is.

        Parameters
        ----------
        segmentation:
            The segmentation partition.
        embedding:
            The embedding partition, if any.
        nodes:
            The nodes to write.
        embeddings:
            The vectors to write.

        Returns
        -------
        None

        Raises
        ------
        SegmentationMismatchError
            If a node does not belong to the segmentation partition.
        EmbeddingSpaceMismatchError
            If a vector does not belong to the embedding partition.
        IndexWriteError
            If vectors were supplied with no partition to hold them.
        """
        for node in nodes:
            if not segmentation.accepts(
                node.segmentation_fingerprint, node.source_schema_version
            ):
                raise SegmentationMismatchError(
                    expected=segmentation.segmentation_fingerprint,
                    actual=node.segmentation_fingerprint,
                    context={"node_id": node.node_id},
                )

        if not embeddings:
            return

        if embedding is None:
            raise IndexWriteError(
                f"{len(embeddings)} vectors were supplied with no embedding "
                f"partition to hold them. Writing the nodes alone would leave "
                f"the corpus silently unembedded.",
                context={"segmentation_key": segmentation.key},
            )

        for vector in embeddings:
            if (
                vector.model_identity != embedding.model_identity
                or vector.dimension != embedding.dimension
                or vector.normalized != embedding.normalized
                or vector.composition_id != embedding.composition_id
                or vector.query_encoding != embedding.query_encoding
            ):
                raise EmbeddingSpaceMismatchError(
                    f"Vector for node {vector.node_id!r} does not belong to "
                    f"partition {embedding.key!r}.",
                    expected_identity=embedding.model_identity,
                    actual_identity=vector.model_identity,
                    context={"node_id": vector.node_id},
                )

    def _apply(
        self,
        segmentation: SegmentationPartition,
        embedding: Optional[EmbeddingPartition],
        nodes: Sequence[KnowledgeNode],
        embeddings: Sequence[NodeEmbedding],
        relations: Sequence[NodeRelation],
    ) -> None:
        """
        Apply a validated write to every store, without awaiting.

        Reaches into the stores' own state rather than calling their coroutine
        methods, because awaiting between stores would create exactly the window
        in which a half-written corpus is observable, and closing that window is
        this writer's only reason to exist.

        Parameters
        ----------
        segmentation:
            The segmentation partition.
        embedding:
            The embedding partition, if any.
        nodes:
            The nodes to write.
        embeddings:
            The vectors to write.
        relations:
            The relations to write.

        Returns
        -------
        None
        """
        by_id = self._nodes._by_id[segmentation.key]
        by_ordinal = self._nodes._by_ordinal[segmentation.key]
        postings = self._lexical._postings[segmentation.key]
        lengths = self._lexical._lengths[segmentation.key]
        held_relations = self._relations._relations[segmentation.key]
        current = self._supersession._current[segmentation.key]
        versions = self._supersession._versions[segmentation.key]

        for node in nodes:
            by_id[node.node_id] = node
            by_ordinal[(node.document_id, node.ordinal)] = node

            terms = _tokens(node.text) + _tokens(" ".join(node.heading_path))
            stale = [
                term for term, holders in postings.items() if node.node_id in holders
            ]
            for term in stale:
                del postings[term][node.node_id]
                if not postings[term]:
                    del postings[term]

            counts: dict[str, int] = defaultdict(int)
            for term in terms:
                counts[term] += 1
            for term, frequency in counts.items():
                postings[term][node.node_id] = frequency
            lengths[node.node_id] = len(terms)

            # A document being written becomes the current representative of its
            # source. Recorded here rather than by a separate call, so that a
            # node and the statement that its document is current cannot
            # disagree.
            #
            # Every node of a document carries that document's own source
            # descriptor, so repeating this per node records the same fact each
            # time rather than overwriting it with a different one.
            current[node.source.source_uri] = node.document_id
            versions[node.source.source_uri] = getattr(
                node.source, "version_tag", None
            )

        if embedding is not None:
            vectors = self._vectors._vectors[embedding.key]
            for vector in embeddings:
                vectors[vector.node_id] = vector

        for relation in relations:
            existing = held_relations[relation.source_node_id]
            if not any(
                other.target_node_id == relation.target_node_id
                and other.relation_type == relation.relation_type
                for other in existing
            ):
                existing.append(relation)

    async def flush(self) -> None:
        """
        Commit buffered writes. Nothing is buffered, so this does nothing.

        Returns
        -------
        None
        """
        return None

    async def close(self) -> None:
        """
        Close the writer. Idempotent.

        Data already written remains readable, so a corpus stays queryable after
        ingest finishes.

        Returns
        -------
        None
        """
        self._closed = True

    def __repr__(self) -> str:
        """
        Return a representation naming the writer's state.

        Returns
        -------
        str
            Diagnostic representation.
        """
        return f"InMemoryIndexWriter(atomic=True, closed={self._closed})"
