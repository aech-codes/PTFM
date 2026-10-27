"""
Unit tests for the in-memory retrieval stores.

File:
    tests/core/retrieval/test_memory_stores.py

Emphasis
--------
    These stores are reference implementations, not doubles, so the tests assert
    real behaviour: that BM25 weights a rare term above a common one, that
    cosine similarity is correct for unnormalized vectors, that partitions
    isolate absolutely, and that a rejected write leaves every store untouched.

    A backend that disagrees with these results is wrong about the contract
    rather than merely different.
"""

from __future__ import annotations

import asyncio
import ast
import inspect
import math
from datetime import UTC, datetime
from typing import Any, Optional, Sequence

import pytest

from ptfm.core.knowledge.models import KnowledgeUnit
from ptfm.core.retrieval.exceptions import (
    EmbeddingSpaceMismatchError,
    IndexWriteError,
    SegmentationMismatchError,
    StoreUnavailableError,
)
from ptfm.core.retrieval.models import (
    EmbeddingPartition,
    FilterField,
    FilterOperator,
    RetrievalFilter,
    SegmentationPartition,
)
from ptfm.core.retrieval.stores import memory as memory_module
from ptfm.core.retrieval.stores.memory import (
    InMemoryIndexWriter,
    InMemoryLexicalIndex,
    InMemoryNodeStore,
    InMemoryPartitionCatalog,
    InMemoryRelationStore,
    InMemorySupersessionStore,
    InMemoryVectorIndex,
    matches_filters,
)
from ptfm.core.schemas import (
    DiscoveryMethod,
    KnowledgeNode,
    NodeEmbedding,
    NodeRelation,
    NodeType,
    RelationType,
    SignalScale,
    SourceDescriptor,
    TextSpan,
)

NOW = datetime(2026, 9, 23, tzinfo=UTC)

SOURCE_A = "file:///corpus/a.pdf"
SOURCE_B = "file:///corpus/b.pdf"


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def _segmentation(fingerprint: str = "1:abc") -> SegmentationPartition:
    return SegmentationPartition(
        segmentation_fingerprint=fingerprint, source_schema_version="1.0"
    )


def _embedding(
    segmentation: Optional[SegmentationPartition] = None,
    *,
    model_identity: str = "BAAI/bge@r1",
    dimension: int = 2,
    normalized: bool = True,
    query_encoding: str = "prefix:query: ",
) -> EmbeddingPartition:
    return EmbeddingPartition(
        segmentation=segmentation or _segmentation(),
        model_identity=model_identity,
        dimension=dimension,
        normalized=normalized,
        composition_id="heading_path_prefixed",
        query_encoding=query_encoding,
    )


def _node(
    node_id: str,
    *,
    document_id: str = "doc-1",
    ordinal: int = 0,
    text: str = "alpha beta",
    source_uri: str = SOURCE_A,
    heading_path: Sequence[str] = (),
    node_type: NodeType = NodeType.PROSE,
    metadata: Optional[dict[str, str]] = None,
    page_numbers: Sequence[int] = (),
    fingerprint: str = "1:abc",
    version_tag: Optional[str] = None,
) -> KnowledgeNode:
    return KnowledgeNode(
        node_id=node_id,
        document_id=document_id,
        span=TextSpan(start=0, end=len(text)),
        text=text,
        node_type=node_type,
        block_ids=(str(ordinal),),
        ordinal=ordinal,
        source=SourceDescriptor(
            source_uri=source_uri,
            scheme="file",
            connector_name="LocalFilesystemConnector",
            extractor_name="PdfExtractor",
            acquired_at=NOW,
            size_bytes=1024,
            detected_media_type="application/pdf",
            version_tag=version_tag,
        ),
        heading_path=tuple(heading_path),
        page_numbers=tuple(page_numbers),
        metadata=dict(metadata or {}),
        segmentation_fingerprint=fingerprint,
        source_text_hash="text-hash",
        source_schema_version="1.0",
        processed_at=NOW,
    )


def _vector(
    node_id: str,
    values: Sequence[float] = (1.0, 0.0),
    *,
    model_id: str = "BAAI/bge",
    revision: Optional[str] = "r1",
    dimension: int = 2,
    normalized: bool = True,
    composition_id: str = "heading_path_prefixed",
    query_encoding: str = "prefix:query: ",
) -> NodeEmbedding:
    return NodeEmbedding(
        node_id=node_id,
        document_id="doc-1",
        vector=tuple(values),
        dimension=dimension,
        model_id=model_id,
        model_revision=revision,
        normalized=normalized,
        composition_id=composition_id,
        query_encoding=query_encoding,
        input_text_hash="input-hash",
        embedded_at=NOW,
    )


def _filter(
    field: FilterField,
    operator: FilterOperator,
    *values: str,
    metadata_key: Optional[str] = None,
) -> RetrievalFilter:
    return RetrievalFilter(
        field=field,
        operator=operator,
        values=tuple(values),
        metadata_key=metadata_key,
    )


class _Stores:
    """The six stores wired as a composition root would wire them."""

    def __init__(self) -> None:
        self.nodes = InMemoryNodeStore()
        self.vectors = InMemoryVectorIndex(self.nodes)
        self.lexical = InMemoryLexicalIndex(self.nodes)
        self.relations = InMemoryRelationStore(self.nodes)
        self.supersession = InMemorySupersessionStore()
        self.catalog = InMemoryPartitionCatalog()
        self.writer = InMemoryIndexWriter(
            self.nodes,
            self.vectors,
            self.lexical,
            self.relations,
            self.supersession,
        )


@pytest.fixture()
def stores() -> _Stores:
    return _Stores()


# ===========================================================================
# PARTITION ISOLATION
# ===========================================================================


async def test_a_partition_cannot_observe_another(stores: _Stores) -> None:
    """
    Mixing incomparable data is not prevented by a check that could be
    bypassed; it is unreachable.
    """
    await stores.nodes.put_many([_node("node-1")], _segmentation())

    assert await stores.nodes.get_many(["node-1"], _segmentation("1:other")) == {}
    assert await stores.nodes.count(_segmentation("1:other")) == 0


async def test_a_node_from_another_segmentation_is_refused(
    stores: _Stores,
) -> None:
    """
    Accepting it would place spans computed against one division of a document
    beside spans computed against another.
    """
    with pytest.raises(SegmentationMismatchError):
        await stores.nodes.put_many(
            [_node("node-1", fingerprint="1:other")], _segmentation()
        )


# ===========================================================================
# NODE STORE
# ===========================================================================


async def test_nodes_resolve_by_identifier(stores: _Stores) -> None:
    partition = _segmentation()
    await stores.nodes.put_many([_node("node-1"), _node("node-2", ordinal=1)], partition)

    resolved = await stores.nodes.get_many(["node-1", "unknown"], partition)

    assert set(resolved) == {"node-1"}


async def test_nodes_resolve_by_ordinal(stores: _Stores) -> None:
    """The mechanism behind context expansion."""
    partition = _segmentation()
    await stores.nodes.put_many(
        [_node(f"node-{index}", ordinal=index) for index in range(3)], partition
    )

    resolved = await stores.nodes.get_by_ordinal("doc-1", [0, 1, 9], partition)

    assert set(resolved) == {0, 1}


async def test_a_rewritten_node_replaces_its_predecessor(
    stores: _Stores,
) -> None:
    partition = _segmentation()
    await stores.nodes.put_many([_node("node-1", text="first")], partition)
    await stores.nodes.put_many([_node("node-1", text="second")], partition)

    resolved = await stores.nodes.get_many(["node-1"], partition)

    assert resolved["node-1"].text == "second"
    assert await stores.nodes.count(partition) == 1


async def test_node_identifiers_enumerate_in_sorted_order(
    stores: _Stores,
) -> None:
    """So that a structural scan is reproducible."""
    partition = _segmentation()
    await stores.nodes.put_many(
        [_node("zulu"), _node("alpha", ordinal=1)], partition
    )

    assert await stores.nodes.node_ids(partition) == ("alpha", "zulu")


# ===========================================================================
# VECTOR INDEX
# ===========================================================================


async def test_vector_search_returns_nearest_first(stores: _Stores) -> None:
    partition = _embedding()
    unit = 1 / math.sqrt(2)
    await stores.vectors.upsert(
        [
            _vector("far", (0.0, 1.0)),
            _vector("near", (1.0, 0.0)),
            _vector("middle", (unit, unit)),
        ],
        partition,
    )

    results = await stores.vectors.search((1.0, 0.0), partition, 3)

    assert [candidate.node_id for candidate in results] == ["near", "middle", "far"]
    assert [candidate.rank for candidate in results] == [0, 1, 2]


async def test_vector_candidates_declare_the_cosine_scale(
    stores: _Stores,
) -> None:
    partition = _embedding()
    await stores.vectors.upsert([_vector("node-1")], partition)

    results = await stores.vectors.search((1.0, 0.0), partition, 1)

    assert results[0].scale is SignalScale.COSINE
    assert results[0].method is DiscoveryMethod.DENSE
    assert results[0].score == pytest.approx(1.0)


async def test_search_is_always_exact(stores: _Stores) -> None:
    """
    A caller requiring reproducibility is always served, which makes this index
    the reference against which an approximate backend's recall is measured.
    """
    assert stores.vectors.supports_exact_search is True


async def test_cosine_is_computed_correctly_for_unnormalized_vectors(
    stores: _Stores,
) -> None:
    """
    A dot product over unnormalized vectors is not a similarity, and treating it
    as one would rank long vectors above relevant ones.
    """
    partition = _embedding(normalized=False)
    await stores.vectors.upsert(
        [
            _vector("long_but_aligned", (10.0, 0.0), normalized=False),
            _vector("short_but_orthogonal", (0.0, 1.0), normalized=False),
        ],
        partition,
    )

    results = await stores.vectors.search((1.0, 0.0), partition, 2)

    assert results[0].node_id == "long_but_aligned"
    assert results[0].score == pytest.approx(1.0)
    assert results[1].score == pytest.approx(0.0)


async def test_equally_similar_vectors_break_the_tie_on_identifier(
    stores: _Stores,
) -> None:
    """Reproducible across processes, which arrival order would not be."""
    partition = _embedding()
    await stores.vectors.upsert(
        [_vector("zulu"), _vector("alpha")], partition
    )

    results = await stores.vectors.search((1.0, 0.0), partition, 2)

    assert [candidate.node_id for candidate in results] == ["alpha", "zulu"]


async def test_a_query_vector_of_the_wrong_width_is_refused(
    stores: _Stores,
) -> None:
    partition = _embedding()

    with pytest.raises(EmbeddingSpaceMismatchError):
        await stores.vectors.search((1.0, 0.0, 0.0), partition, 1)


@pytest.mark.parametrize(
    "override",
    [
        {"model_id": "other/model"},
        {"revision": "r2"},
        {"normalized": False},
        {"composition_id": "text_only"},
        {"query_encoding": ""},
    ],
)
async def test_a_vector_from_another_space_is_refused(
    stores: _Stores, override: dict[str, Any]
) -> None:
    """
    Storing it would make every later comparison against it meaningless while
    reporting success.
    """
    with pytest.raises(EmbeddingSpaceMismatchError):
        await stores.vectors.upsert([_vector("node-1", **override)], _embedding())


async def test_a_filtered_vector_search_needs_its_nodes(
    stores: _Stores,
) -> None:
    """
    A vector with no node means the index and the node store have diverged, and
    a filtered search cannot establish whether it is permitted.
    """
    partition = _embedding()
    await stores.vectors.upsert([_vector("orphan")], partition)

    with pytest.raises(StoreUnavailableError):
        await stores.vectors.search(
            (1.0, 0.0),
            partition,
            1,
            filters=(_filter(FilterField.NODE_TYPE, FilterOperator.EQUALS, "PROSE"),),
        )


# ===========================================================================
# LEXICAL INDEX
# ===========================================================================


async def _index_corpus(stores: _Stores) -> SegmentationPartition:
    partition = _segmentation()
    corpus = [
        _node("rare", text="the applicable rate is four percent", ordinal=0),
        _node("common", text="the rate of interest", ordinal=1),
        _node("unrelated", text="entirely different prose about weather", ordinal=2),
        _node("repetitive", text="rate rate rate rate", ordinal=3),
    ]
    await stores.nodes.put_many(corpus, partition)
    await stores.lexical.index_nodes(corpus, partition)
    return partition


async def test_lexical_search_finds_term_matches(stores: _Stores) -> None:
    partition = await _index_corpus(stores)

    results = await stores.lexical.search("applicable rate", partition, 10)

    assert {candidate.node_id for candidate in results} >= {
        "rare",
        "common",
        "repetitive",
    }


async def test_inverse_document_frequency_weights_the_rare_term(
    stores: _Stores,
) -> None:
    """
    The reason this is BM25 and not a term-overlap count. The arm exists to find
    exact matches on identifiers and rare nouns, and without IDF a node
    containing many common query words would outrank the one containing the term
    that mattered.
    """
    partition = await _index_corpus(stores)

    results = await stores.lexical.search("applicable rate", partition, 10)

    assert results[0].node_id == "rare"


async def test_a_node_sharing_no_term_is_omitted(stores: _Stores) -> None:
    partition = await _index_corpus(stores)

    results = await stores.lexical.search("applicable rate", partition, 10)

    assert "unrelated" not in {candidate.node_id for candidate in results}


async def test_lexical_candidates_declare_the_lexical_scale(
    stores: _Stores,
) -> None:
    """Corpus-relative and unbounded, which is why they are never compared
    against a similarity."""
    partition = await _index_corpus(stores)

    results = await stores.lexical.search("rate", partition, 10)

    assert all(candidate.scale is SignalScale.LEXICAL for candidate in results)
    assert all(candidate.method is DiscoveryMethod.SPARSE for candidate in results)


@pytest.mark.parametrize("query", ["", "   ", "zzzznomatch"])
async def test_a_query_that_cannot_match_returns_nothing(
    stores: _Stores, query: str
) -> None:
    partition = await _index_corpus(stores)

    assert await stores.lexical.search(query, partition, 10) == ()


async def test_reindexing_replaces_postings_rather_than_accumulating(
    stores: _Stores,
) -> None:
    """
    Otherwise a node would climb the ranking by being written twice, and a
    reprocessed corpus would rank differently from a freshly ingested one.
    """
    partition = await _index_corpus(stores)
    node = _node("repetitive", text="rate rate rate rate", ordinal=3)

    before = next(
        candidate.score
        for candidate in await stores.lexical.search("rate", partition, 10)
        if candidate.node_id == "repetitive"
    )
    await stores.lexical.index_nodes([node], partition)
    after = next(
        candidate.score
        for candidate in await stores.lexical.search("rate", partition, 10)
        if candidate.node_id == "repetitive"
    )

    assert before == pytest.approx(after)


async def test_the_heading_path_is_searchable(stores: _Stores) -> None:
    """A section title describes its content, so a query naming the section
    should reach the nodes inside it."""
    partition = _segmentation()
    node = _node("inside", text="body text", heading_path=("Chapter 2", "Interest"))
    await stores.nodes.put_many([node], partition)
    await stores.lexical.index_nodes([node], partition)

    results = await stores.lexical.search("interest", partition, 10)

    assert [candidate.node_id for candidate in results] == ["inside"]


# ===========================================================================
# FILTERS
# ===========================================================================


def _rich_node() -> KnowledgeNode:
    return _node(
        "node-1",
        document_id="doc-7",
        ordinal=4,
        source_uri=SOURCE_B,
        node_type=NodeType.TABLE,
        heading_path=("Chapter 2", "Rates", "Table"),
        page_numbers=(3, 4),
        metadata={"instrument": "NASDAQ:AAPL"},
    )


@pytest.mark.parametrize(
    ("constraint", "expected"),
    [
        (_filter(FilterField.DOCUMENT_ID, FilterOperator.EQUALS, "doc-7"), True),
        (_filter(FilterField.DOCUMENT_ID, FilterOperator.EQUALS, "doc-1"), False),
        (_filter(FilterField.DOCUMENT_ID, FilterOperator.NOT_EQUALS, "doc-1"), True),
        (_filter(FilterField.SOURCE_URI, FilterOperator.IN, SOURCE_A, SOURCE_B), True),
        (_filter(FilterField.SOURCE_URI, FilterOperator.NOT_IN, SOURCE_A), True),
        (_filter(FilterField.NODE_TYPE, FilterOperator.EQUALS, "TABLE"), True),
        (_filter(FilterField.ORDINAL, FilterOperator.GREATER_THAN_OR_EQUAL, "4"), True),
        (_filter(FilterField.ORDINAL, FilterOperator.GREATER_THAN_OR_EQUAL, "5"), False),
        (_filter(FilterField.ORDINAL, FilterOperator.LESS_THAN_OR_EQUAL, "4"), True),
        (_filter(FilterField.PAGE_NUMBER, FilterOperator.EQUALS, "3"), True),
        (_filter(FilterField.PAGE_NUMBER, FilterOperator.EQUALS, "9"), False),
        (
            _filter(
                FilterField.HEADING_PATH,
                FilterOperator.STARTS_WITH,
                "Chapter 2",
                "Rates",
            ),
            True,
        ),
        (
            _filter(FilterField.HEADING_PATH, FilterOperator.STARTS_WITH, "Chapter 3"),
            False,
        ),
        (_filter(FilterField.HEADING_PATH, FilterOperator.IN, "Rates"), True),
        (
            _filter(
                FilterField.METADATA,
                FilterOperator.EQUALS,
                "NASDAQ:AAPL",
                metadata_key="instrument",
            ),
            True,
        ),
        (
            _filter(
                FilterField.METADATA,
                FilterOperator.EQUALS,
                "NYSE:IBM",
                metadata_key="instrument",
            ),
            False,
        ),
    ],
)
def test_filter_evaluation(constraint: RetrievalFilter, expected: bool) -> None:
    assert matches_filters(_rich_node(), [constraint]) is expected


def test_a_missing_metadata_key_fails_a_positive_and_passes_a_negative() -> None:
    """
    Treating absence as a match either way would make EQUALS and NOT_EQUALS both
    true for the same node.
    """
    node = _rich_node()

    assert not matches_filters(
        node,
        [
            _filter(
                FilterField.METADATA, FilterOperator.EQUALS, "x", metadata_key="absent"
            )
        ],
    )
    assert matches_filters(
        node,
        [
            _filter(
                FilterField.METADATA,
                FilterOperator.NOT_EQUALS,
                "x",
                metadata_key="absent",
            )
        ],
    )


def test_filters_combine_conjunctively() -> None:
    """Disjunction is expressed within a filter by IN, which keeps a backend's
    translation unambiguous."""
    node = _rich_node()

    assert matches_filters(
        node,
        [
            _filter(FilterField.DOCUMENT_ID, FilterOperator.EQUALS, "doc-7"),
            _filter(FilterField.NODE_TYPE, FilterOperator.EQUALS, "TABLE"),
        ],
    )
    assert not matches_filters(
        node,
        [
            _filter(FilterField.DOCUMENT_ID, FilterOperator.EQUALS, "doc-7"),
            _filter(FilterField.NODE_TYPE, FilterOperator.EQUALS, "PROSE"),
        ],
    )


def test_no_filters_match_everything() -> None:
    assert matches_filters(_rich_node(), [])


def test_an_ordered_comparison_against_a_non_number_is_no_match() -> None:
    """
    Not merely false: meaningless. Inventing an ordering over strings would make
    these stores disagree with a backend that does not.
    """
    assert not matches_filters(
        _rich_node(),
        [_filter(FilterField.SOURCE_URI, FilterOperator.GREATER_THAN_OR_EQUAL, "abc")],
    )


def test_filter_evaluation_has_one_implementation() -> None:
    """
    A filter meaning one thing to the vector index and another to the lexical
    index would make a hybrid query return a set no single arm agreed with.
    """
    source = inspect.getsource(memory_module)

    assert source.count("def matches_filters") == 1
    assert source.count("matches_filters(node, filters)") >= 2


async def test_both_searching_indexes_honour_filters(stores: _Stores) -> None:
    partition = _segmentation()
    nodes = [
        _node("prose", node_type=NodeType.PROSE, text="rate alpha", ordinal=0),
        _node("table", node_type=NodeType.TABLE, text="rate beta", ordinal=1),
    ]
    await stores.nodes.put_many(nodes, partition)
    await stores.lexical.index_nodes(nodes, partition)
    embedding = _embedding(partition)
    await stores.vectors.upsert(
        [_vector("prose"), _vector("table")], embedding
    )

    constraint = (_filter(FilterField.NODE_TYPE, FilterOperator.EQUALS, "TABLE"),)

    dense = await stores.vectors.search(
        (1.0, 0.0), embedding, 10, filters=constraint
    )
    sparse = await stores.lexical.search("rate", partition, 10, filters=constraint)

    assert [candidate.node_id for candidate in dense] == ["table"]
    assert [candidate.node_id for candidate in sparse] == ["table"]


# ===========================================================================
# RELATION STORE
# ===========================================================================


async def test_relations_are_found_from_their_source(stores: _Stores) -> None:
    partition = _segmentation()
    await stores.relations.put_many(
        [
            NodeRelation(
                source_node_id="a",
                target_node_id="b",
                relation_type=RelationType.PRECEDES,
            ),
            NodeRelation(
                source_node_id="a",
                target_node_id="b",
                relation_type=RelationType.SAME_PAGE,
            ),
        ],
        partition,
    )

    found = await stores.relations.relations_from(["a", "b"], partition)

    assert len(found["a"]) == 2
    assert "b" not in found


async def test_relations_can_be_filtered_by_type(stores: _Stores) -> None:
    partition = _segmentation()
    await stores.relations.put_many(
        [
            NodeRelation(
                source_node_id="a",
                target_node_id="b",
                relation_type=RelationType.PRECEDES,
            ),
            NodeRelation(
                source_node_id="a",
                target_node_id="c",
                relation_type=RelationType.SAME_PAGE,
            ),
        ],
        partition,
    )

    found = await stores.relations.relations_from(
        ["a"], partition, relation_types=["PRECEDES"]
    )

    assert len(found["a"]) == 1


async def test_rewriting_relations_does_not_duplicate_them(
    stores: _Stores,
) -> None:
    """
    Relations are re-derived whenever a document is reprocessed, so a rewrite
    must not accumulate duplicates that would make a traversal visit the same
    target repeatedly.
    """
    partition = _segmentation()
    relations = [
        NodeRelation(
            source_node_id="a", target_node_id="b", relation_type=RelationType.PRECEDES
        )
    ]

    await stores.relations.put_many(relations, partition)
    await stores.relations.put_many(relations, partition)

    found = await stores.relations.relations_from(["a"], partition)
    assert len(found["a"]) == 1


async def test_structural_search_finds_matching_nodes(stores: _Stores) -> None:
    partition = _segmentation()
    await stores.nodes.put_many(
        [
            _node("prose", node_type=NodeType.PROSE, ordinal=0),
            _node("table", node_type=NodeType.TABLE, ordinal=1),
        ],
        partition,
    )

    results = await stores.relations.structural_search(
        [_filter(FilterField.NODE_TYPE, FilterOperator.EQUALS, "TABLE")], partition, 10
    )

    assert [candidate.node_id for candidate in results] == ["table"]


async def test_structural_results_are_unranked_and_unscored(
    stores: _Stores,
) -> None:
    """
    A structural constraint implies no relevance ordering. A fabricated score
    would be indistinguishable from a measured one by the time it reached
    fusion.
    """
    partition = _segmentation()
    await stores.nodes.put_many([_node("table", node_type=NodeType.TABLE)], partition)

    results = await stores.relations.structural_search(
        [_filter(FilterField.NODE_TYPE, FilterOperator.EQUALS, "TABLE")], partition, 10
    )

    assert all(
        candidate.rank == 0
        and candidate.score is None
        and candidate.scale is SignalScale.RANK_ONLY
        for candidate in results
    )


async def test_an_unconstrained_structural_search_returns_nothing(
    stores: _Stores,
) -> None:
    """Without a constraint there is no question to answer, and returning the
    corpus would swamp the arms that did answer one."""
    partition = _segmentation()
    await stores.nodes.put_many([_node("node-1")], partition)

    assert await stores.relations.structural_search([], partition, 10) == ()


async def test_structural_results_are_ordered_by_document_then_position(
    stores: _Stores,
) -> None:
    """So the same constraint returns the same set in the same order every
    run."""
    partition = _segmentation()
    await stores.nodes.put_many(
        [
            _node("second", document_id="doc-1", ordinal=1),
            _node("first", document_id="doc-1", ordinal=0),
            _node("other", document_id="doc-2", ordinal=0),
        ],
        partition,
    )

    results = await stores.relations.structural_search(
        [_filter(FilterField.SOURCE_URI, FilterOperator.EQUALS, SOURCE_A)],
        partition,
        10,
    )

    assert [candidate.node_id for candidate in results] == [
        "first",
        "second",
        "other",
    ]


# ===========================================================================
# SUPERSESSION
# ===========================================================================


async def test_currency_is_recorded_and_read_back(stores: _Stores) -> None:
    partition = _segmentation()
    await stores.supersession.record_current(SOURCE_A, "doc-1", "v1", partition)

    current = await stores.supersession.current_document_ids([SOURCE_A], partition)

    assert current[SOURCE_A] == "doc-1"


async def test_a_source_with_no_record_is_absent_from_the_result(
    stores: _Stores,
) -> None:
    """
    Which the caller reads as "not known to be superseded" rather than as
    withdrawn.
    """
    assert (
        await stores.supersession.current_document_ids([SOURCE_A], _segmentation())
        == {}
    )


async def test_a_tombstone_is_recorded_as_a_present_none(
    stores: _Stores,
) -> None:
    """
    Distinct from having no record at all. Collapsing the two would make a
    deletion indistinguishable from an omission.
    """
    partition = _segmentation()
    await stores.supersession.record_current(SOURCE_A, None, None, partition)

    current = await stores.supersession.current_document_ids([SOURCE_A], partition)

    assert SOURCE_A in current
    assert current[SOURCE_A] is None


async def test_the_version_marker_is_retained_for_audit(
    stores: _Stores,
) -> None:
    """
    It answers which source-side version a currency record was based on, which
    distinguishes a supersession the source declared from one inferred from
    ingest order.
    """
    partition = _segmentation()
    await stores.supersession.record_current(SOURCE_A, "doc-1", "v7", partition)

    assert await stores.supersession.version_tag(SOURCE_A, partition) == "v7"


# ===========================================================================
# PARTITION CATALOG
# ===========================================================================


async def test_nothing_serves_queries_until_something_is_activated(
    stores: _Stores,
) -> None:
    assert await stores.catalog.active_segmentation() is None


async def test_activation_makes_a_partition_serve_queries(
    stores: _Stores,
) -> None:
    partition = _segmentation()
    embedding = _embedding(partition)

    await stores.catalog.register(partition, embedding, activate=True)

    active = await stores.catalog.active_segmentation()
    assert active is not None and active.key == partition.key
    active_embedding = await stores.catalog.active_embedding(partition)
    assert active_embedding is not None and active_embedding.key == embedding.key


async def test_registering_without_activating_leaves_the_active_one_serving(
    stores: _Stores,
) -> None:
    """
    How a migration proceeds: build the new partition alongside the one serving
    queries, then turn one atomic switch.
    """
    partition = _segmentation()
    first = _embedding(partition)
    second = _embedding(partition, model_identity="other/model@r1")

    await stores.catalog.register(partition, first, activate=True)
    await stores.catalog.register(partition, second, activate=False)

    active = await stores.catalog.active_embedding(partition)
    assert active is not None and active.key == first.key

    await stores.catalog.register(partition, second, activate=True)

    active = await stores.catalog.active_embedding(partition)
    assert active is not None and active.key == second.key


async def test_an_embedding_under_a_foreign_segmentation_is_refused(
    stores: _Stores,
) -> None:
    """Its vectors would be compared against nodes they were not produced
    from."""
    with pytest.raises(EmbeddingSpaceMismatchError):
        await stores.catalog.register(
            _segmentation("1:other"), _embedding(_segmentation()), activate=False
        )


async def test_a_segmented_but_unembedded_corpus_is_a_valid_state(
    stores: _Stores,
) -> None:
    partition = _segmentation()

    await stores.catalog.register(partition, None, activate=True)

    assert await stores.catalog.active_embedding(partition) is None


# ===========================================================================
# ATOMIC WRITES
# ===========================================================================


def _units(
    count: int = 2, *, fingerprint: str = "1:abc", model_id: str = "BAAI/bge"
) -> list[KnowledgeUnit]:
    return [
        KnowledgeUnit(
            node=_node(
                f"node-{index}",
                ordinal=index,
                fingerprint=fingerprint,
                version_tag="v1",
            ),
            embedding=_vector(f"node-{index}", model_id=model_id),
            relations=(
                [
                    NodeRelation(
                        source_node_id=f"node-{index}",
                        target_node_id=f"node-{index + 1}",
                        relation_type=RelationType.PRECEDES,
                    )
                ]
                if index == 0
                else []
            ),
        )
        for index in range(count)
    ]


def test_the_writer_declares_itself_atomic(stores: _Stores) -> None:
    """
    Validation precedes application and application does not await, so a
    failure cannot leave a unit half-written.
    """
    assert stores.writer.is_atomic is True


async def test_one_write_updates_every_store(stores: _Stores) -> None:
    partition = _segmentation()
    embedding = _embedding(partition)

    await stores.writer.write_units(partition, embedding, _units())

    assert await stores.nodes.count(partition) == 2
    assert await stores.vectors.count(embedding) == 2
    assert len(await stores.lexical.search("alpha", partition, 10)) == 2
    assert (await stores.relations.relations_from(["node-0"], partition))["node-0"]
    current = await stores.supersession.current_document_ids([SOURCE_A], partition)
    assert current[SOURCE_A] == "doc-1"


async def test_the_version_marker_is_captured_from_the_document(
    stores: _Stores,
) -> None:
    """
    Every node of a document carries that document's own source descriptor, so
    recording per node records the same fact each time.
    """
    partition = _segmentation()

    await stores.writer.write_units(partition, _embedding(partition), _units())

    assert await stores.supersession.version_tag(SOURCE_A, partition) == "v1"


async def test_a_rejected_write_leaves_every_store_untouched(
    stores: _Stores,
) -> None:
    """
    The writer's whole reason to exist. A unit whose vector reached the index
    while its node did not would be retrievable with no authoritative text: a
    result that appears valid and cannot be traced.
    """
    partition = _segmentation()
    embedding = _embedding(partition)
    units = _units(1) + [
        KnowledgeUnit(
            node=_node("foreign", ordinal=1, fingerprint="1:other"),
            embedding=_vector("foreign"),
        )
    ]

    with pytest.raises(SegmentationMismatchError):
        await stores.writer.write_units(partition, embedding, units)

    assert await stores.nodes.count(partition) == 0
    assert await stores.vectors.count(embedding) == 0
    assert await stores.lexical.search("alpha", partition, 10) == ()
    assert (
        await stores.supersession.current_document_ids([SOURCE_A], partition) == {}
    )


async def test_a_foreign_vector_also_rolls_the_whole_batch_back(
    stores: _Stores,
) -> None:
    partition = _segmentation()
    embedding = _embedding(partition)
    units = _units(1) + [
        KnowledgeUnit(
            node=_node("other", ordinal=1),
            embedding=_vector("other", model_id="wrong/model"),
        )
    ]

    with pytest.raises(EmbeddingSpaceMismatchError):
        await stores.writer.write_units(partition, embedding, units)

    assert await stores.nodes.count(partition) == 0
    assert await stores.vectors.count(embedding) == 0


async def test_vectors_without_a_partition_are_refused(stores: _Stores) -> None:
    """Writing the nodes alone would leave the corpus silently unembedded."""
    partition = _segmentation()

    with pytest.raises(IndexWriteError):
        await stores.writer.write_units(partition, None, _units(1))

    assert await stores.nodes.count(partition) == 0


async def test_an_unembedded_batch_is_a_valid_write(stores: _Stores) -> None:
    partition = _segmentation()
    unit = KnowledgeUnit(node=_node("node-0"))

    await stores.writer.write_units(partition, None, [unit])

    assert await stores.nodes.count(partition) == 1


async def test_rewriting_a_unit_is_idempotent(stores: _Stores) -> None:
    """
    Node identity is deterministic, so reprocessing a corpus converges rather
    than duplicating it.
    """
    partition = _segmentation()
    embedding = _embedding(partition)
    units = _units(1)

    await stores.writer.write_units(partition, embedding, units)
    await stores.writer.write_units(partition, embedding, units)

    assert await stores.nodes.count(partition) == 1
    assert await stores.vectors.count(embedding) == 1
    assert len((await stores.relations.relations_from(["node-0"], partition))["node-0"]) == 1


async def test_an_empty_write_is_a_no_op(stores: _Stores) -> None:
    partition = _segmentation()

    await stores.writer.write_units(partition, _embedding(partition), [])
    await stores.writer.flush()

    assert await stores.nodes.count(partition) == 0


async def test_a_closed_writer_refuses_but_its_data_remains(
    stores: _Stores,
) -> None:
    """A corpus stays queryable after ingest finishes."""
    partition = _segmentation()
    embedding = _embedding(partition)
    await stores.writer.write_units(partition, embedding, _units(1))

    await stores.writer.close()
    await stores.writer.close()

    with pytest.raises(IndexWriteError):
        await stores.writer.write_units(partition, embedding, _units(1))
    assert await stores.nodes.count(partition) == 1


async def test_concurrent_writes_leave_no_store_behind(stores: _Stores) -> None:
    """
    Every store advances together under interleaving, which is what the lock and
    the await-free application are for.
    """
    partition = _segmentation()
    embedding = _embedding(partition)

    async def write(index: int) -> None:
        await stores.writer.write_units(
            partition,
            embedding,
            [
                KnowledgeUnit(
                    node=_node(f"c{index}", ordinal=index),
                    embedding=_vector(f"c{index}"),
                )
            ],
        )

    await asyncio.gather(*(write(index) for index in range(30)))

    assert await stores.nodes.count(partition) == 30
    assert await stores.vectors.count(embedding) == 30
    assert len(await stores.lexical.search("alpha", partition, 50)) == 30


def test_the_writer_applies_without_awaiting(stores: _Stores) -> None:
    """
    Awaiting between stores would create exactly the window in which a
    half-written corpus is observable, and closing that window is this writer's
    only reason to exist.
    """
    body = _function_body_without_docstring(memory_module.InMemoryIndexWriter._apply)

    assert "await" not in body
    assert not inspect.iscoroutinefunction(memory_module.InMemoryIndexWriter._apply)


# ===========================================================================
# Anti-pattern guards
# ===========================================================================


def _function_body_without_docstring(function: Any) -> str:
    """
    Return a function's source with its docstring removed.

    A docstring that DESCRIBES a property names the very thing a scan looks for:
    ``_apply``'s docstring explains that it does not await, which a naive scan
    reads as an await. Stripping it makes the assertion test the code rather
    than the prose.

    Parameters
    ----------
    function:
        The function to read.

    Returns
    -------
    str
        The source with the docstring removed.
    """
    source = inspect.getsource(function)
    tree = ast.parse(inspect.cleandoc(source) if source.startswith(" ") else source)
    definition = tree.body[0]

    if not isinstance(definition, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return source

    body = definition.body
    if not body:
        return source

    first = body[0]
    if not (
        isinstance(first, ast.Expr)
        and isinstance(first.value, ast.Constant)
        and isinstance(first.value.value, str)
    ):
        return source

    return "\n".join(ast.unparse(statement) for statement in body[1:])


def _code_without_docstrings(module: object) -> str:
    """
    Return a module's source with every docstring removed.

    Line ranges are used rather than text replacement because
    ``ast.get_docstring`` dedents its result, which does not match indented
    source.

    Parameters
    ----------
    module:
        The module to read.

    Returns
    -------
    str
        The source with docstrings stripped.
    """
    source = inspect.getsource(module)  # type: ignore[arg-type]
    tree = ast.parse(source)
    lines = source.splitlines(keepends=True)
    blank: set[int] = set()

    for node in ast.walk(tree):
        if not isinstance(
            node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
        ):
            continue
        body = getattr(node, "body", None)
        if not body:
            continue
        first = body[0]
        if not (
            isinstance(first, ast.Expr)
            and isinstance(first.value, ast.Constant)
            and isinstance(first.value.value, str)
        ):
            continue
        blank.update(range(first.lineno - 1, (first.end_lineno or first.lineno)))

    return "".join("" if i in blank else line for i, line in enumerate(lines))


def test_the_module_has_no_third_party_dependency() -> None:
    """So these stores are importable wherever the Core is."""
    source = inspect.getsource(memory_module)

    for forbidden in (
        "import psycopg",
        "import asyncpg",
        "import qdrant",
        "import numpy",
        "import pydantic",
        "from ptfm.plugins",
    ):
        assert forbidden not in source


def test_the_production_warning_is_prominent() -> None:
    """The class names do not convey that these grow without bound."""
    assert "NOT FOR PRODUCTION USE" in (memory_module.__doc__ or "")


def test_the_module_contains_no_domain_vocabulary() -> None:
    source = inspect.getsource(memory_module).lower()

    for term in ("jurisdiction", "statute", "ticker", " icd", "patient", "court"):
        assert term not in source


def test_the_module_declares_no_calibration_concept() -> None:
    source = _code_without_docstrings(memory_module).lower()

    for term in ("confidence", "calibrat", "entail", "truth space"):
        assert term not in source


def test_the_module_declares_no_mutable_state() -> None:
    mutable = [
        name
        for name, value in vars(memory_module).items()
        if not name.startswith("_") and isinstance(value, (list, dict, set))
    ]

    assert mutable == []


def test_the_module_never_catches_base_exception() -> None:
    assert "except BaseException" not in inspect.getsource(memory_module)


def test_the_bm25_parameters_are_the_standard_ones() -> None:
    """
    Named constants rather than inline numbers, so a deployment comparing these
    results against a backend's can see what they were compared under.
    """
    assert memory_module._BM25_K1 == 1.2
    assert memory_module._BM25_B == 0.75


@pytest.mark.parametrize(
    "store",
    [
        InMemoryNodeStore(),
        InMemorySupersessionStore(),
        InMemoryPartitionCatalog(),
    ],
)
def test_stores_hold_no_assignable_state(store: Any) -> None:
    with pytest.raises(AttributeError):
        store.cache = {}
