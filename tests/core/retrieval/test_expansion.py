"""
Unit and property tests for context expansion.

File:
    tests/core/retrieval/test_expansion.py

Emphasis
--------
    The anchor invariants. Expansion is the designated remedy for the gap that
    removing overlapping nodes left behind, and if it can manufacture
    corroboration then removing overlap bought nothing. Every test below exists
    to show that it cannot.
"""

from __future__ import annotations

import ast
import inspect
from datetime import UTC, datetime
from typing import Any, Mapping, Optional, Sequence

import pytest
from hypothesis import given, settings, strategies as st

from ptfm.core.retrieval.config import ExpansionConfig
from ptfm.core.retrieval.exceptions import AnchorViolationError, ExpansionError
from ptfm.core.retrieval.models import SegmentationPartition
from ptfm.core.retrieval.pipeline import expansion as expansion_module
from ptfm.core.retrieval.pipeline.expansion import (
    ORDINAL_EXPANDER_NAME,
    OrdinalContextExpander,
)
from ptfm.core.schemas import (
    KnowledgeNode,
    NodeType,
    SourceDescriptor,
    TextSpan,
)

NOW = datetime(2026, 9, 23, tzinfo=UTC)

#: Two documents of differing length, so a document's first and last node and a
#: fully covered document are all reachable in tests.
DOCUMENTS: Mapping[str, int] = {"doc-a": 6, "doc-b": 3}


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def _partition() -> SegmentationPartition:
    return SegmentationPartition(
        segmentation_fingerprint="1:abc", source_schema_version="1.0"
    )


def _node(document_id: str, ordinal: int) -> KnowledgeNode:
    text = f"{document_id} node {ordinal}"
    return KnowledgeNode(
        node_id=f"{document_id}-{ordinal}",
        document_id=document_id,
        span=TextSpan(start=0, end=len(text)),
        text=text,
        node_type=NodeType.PROSE,
        block_ids=(str(ordinal),),
        ordinal=ordinal,
        source=SourceDescriptor(
            source_uri=f"file:///corpus/{document_id}.pdf",
            scheme="file",
            connector_name="LocalFilesystemConnector",
            extractor_name="PdfExtractor",
            acquired_at=NOW,
            size_bytes=1024,
            detected_media_type="application/pdf",
        ),
        segmentation_fingerprint="1:abc",
        source_text_hash="text-hash",
        source_schema_version="1.0",
        processed_at=NOW,
    )


class _Logger:
    def __init__(self) -> None:
        self.events: list[str] = []

    def bind(self, **values: Any) -> _Logger:
        return self

    def debug(self, event: str, **values: Any) -> None:
        self.events.append(event)

    def info(self, event: str, **values: Any) -> None:
        self.events.append(event)

    def warning(self, event: str, **values: Any) -> None:
        self.events.append(event)

    def error(self, event: str, **values: Any) -> None:
        self.events.append(event)


class _NodeStore:
    """Serves a fixed corpus by ordinal, recording each call."""

    def __init__(
        self,
        documents: Mapping[str, int] = DOCUMENTS,
        *,
        failure: Optional[Exception] = None,
    ) -> None:
        self._documents = {
            document_id: {
                ordinal: _node(document_id, ordinal) for ordinal in range(count)
            }
            for document_id, count in documents.items()
        }
        self._failure = failure
        self.calls: list[tuple[str, tuple[int, ...]]] = []

    @property
    def name(self) -> str:
        return "node"

    async def get_many(
        self, node_ids: Sequence[str], partition: Any
    ) -> Mapping[str, KnowledgeNode]:
        return {}

    async def get_by_ordinal(
        self, document_id: str, ordinals: Sequence[int], partition: Any
    ) -> Mapping[int, KnowledgeNode]:
        self.calls.append((document_id, tuple(ordinals)))
        if self._failure is not None:
            raise self._failure
        available = self._documents.get(document_id, {})
        return {
            ordinal: available[ordinal]
            for ordinal in ordinals
            if ordinal in available
        }

    async def put_many(self, nodes: Any, partition: Any) -> None:
        return None

    async def count(self, partition: Any) -> int:
        return 0


def _expander(
    store: Optional[_NodeStore] = None,
    *,
    logger: Optional[_Logger] = None,
    **config: Any,
) -> OrdinalContextExpander:
    return OrdinalContextExpander(
        store or _NodeStore(), ExpansionConfig(**config), logger or _Logger()
    )


def _flatten(context: Mapping[str, Sequence[KnowledgeNode]]) -> list[str]:
    return [node.node_id for nodes in context.values() for node in nodes]


# ===========================================================================
# THE ANCHOR INVARIANTS
# ===========================================================================


async def test_an_anchor_is_never_taken_as_context() -> None:
    """
    An anchor was found independently. Listing it as another anchor's companion
    would either duplicate it or demote a finding to a companion, and
    corroboration counts distinct anchors.
    """
    anchors = [_node("doc-a", 1), _node("doc-a", 2)]

    context = await _expander().expand(anchors, _partition())

    anchor_ids = {anchor.node_id for anchor in anchors}
    assert not anchor_ids & set(_flatten(context))


async def test_adjacent_anchors_still_gain_their_outer_context() -> None:
    """Excluding anchors narrows context; it does not disable it."""
    anchors = [_node("doc-a", 1), _node("doc-a", 2)]

    context = await _expander().expand(anchors, _partition())

    assert set(_flatten(context)) == {"doc-a-0", "doc-a-3"}


async def test_a_shared_neighbour_appears_exactly_once() -> None:
    """Emitting it twice would let one passage be counted twice."""
    anchors = [_node("doc-a", 1), _node("doc-a", 3)]

    context = await _expander().expand(anchors, _partition())

    flattened = _flatten(context)
    assert flattened.count("doc-a-2") == 1
    assert len(flattened) == len(set(flattened))


async def test_a_shared_neighbour_goes_to_the_earlier_anchor() -> None:
    """
    Deterministic, and it puts shared context on the higher-ranked result, where
    a consumer reading top-down encounters it soonest.
    """
    context = await _expander().expand(
        [_node("doc-a", 1), _node("doc-a", 3)], _partition()
    )

    assert "doc-a-2" in [node.node_id for node in context["doc-a-1"]]


async def test_anchor_order_decides_assignment() -> None:
    context = await _expander().expand(
        [_node("doc-a", 3), _node("doc-a", 1)], _partition()
    )

    assert "doc-a-2" in [node.node_id for node in context["doc-a-3"]]
    assert _flatten(context).count("doc-a-2") == 1


async def test_the_corroboration_count_is_unchanged_by_expansion() -> None:
    """
    THE test this module exists to pass. A claim straddling two nodes yields ONE
    anchor with context attached, not two findings. If this failed, removing
    overlapping nodes from Knowledge Processing would have bought nothing.
    """
    anchors = [_node("doc-a", 2)]

    context = await _expander().expand(anchors, _partition())

    assert len(anchors) == 1
    assert len(context["doc-a-2"]) == 2
    assert all(node.node_id != "doc-a-2" for node in context["doc-a-2"])


# ===========================================================================
# NEIGHBOURS
# ===========================================================================


async def test_radius_one_yields_both_neighbours() -> None:
    context = await _expander().expand([_node("doc-a", 2)], _partition())

    assert [node.node_id for node in context["doc-a-2"]] == ["doc-a-1", "doc-a-3"]


async def test_context_is_returned_in_reading_order() -> None:
    context = await _expander(neighbour_radius=2).expand(
        [_node("doc-a", 2)], _partition()
    )

    ordinals = [node.ordinal for node in context["doc-a-2"]]
    assert ordinals == sorted(ordinals)


async def test_a_documents_first_node_has_only_a_successor() -> None:
    """A negative ordinal cannot exist, and asking for one is pointless."""
    context = await _expander().expand([_node("doc-a", 0)], _partition())

    assert [node.node_id for node in context["doc-a-0"]] == ["doc-a-1"]


async def test_a_documents_last_node_has_only_a_predecessor() -> None:
    context = await _expander().expand([_node("doc-a", 5)], _partition())

    assert [node.node_id for node in context["doc-a-5"]] == ["doc-a-4"]


async def test_a_wider_radius_reaches_further() -> None:
    context = await _expander(neighbour_radius=2).expand(
        [_node("doc-a", 2)], _partition()
    )

    assert [node.node_id for node in context["doc-a-2"]] == [
        "doc-a-0",
        "doc-a-1",
        "doc-a-3",
        "doc-a-4",
    ]


async def test_context_never_crosses_a_document_boundary() -> None:
    """
    The node before a document's first is not adjacent to it in any meaningful
    sense; it belongs to a different document that happened to be indexed
    nearby.
    """
    context = await _expander().expand(
        [_node("doc-a", 0), _node("doc-b", 0)], _partition()
    )

    for anchor_id, nodes in context.items():
        expected = anchor_id.rsplit("-", 1)[0]
        assert all(node.document_id == expected for node in nodes)


async def test_an_anchor_with_no_context_is_absent_from_the_mapping() -> None:
    """
    A caller iterating the mapping should visit only anchors that gained
    something.
    """
    anchors = [_node("doc-b", ordinal) for ordinal in range(3)]

    assert await _expander().expand(anchors, _partition()) == {}


# ===========================================================================
# BOUNDS
# ===========================================================================


async def test_expansion_can_be_disabled() -> None:
    assert await _expander(enabled=False).expand([_node("doc-a", 2)], _partition()) == {}


async def test_a_radius_of_zero_yields_nothing() -> None:
    assert (
        await _expander(neighbour_radius=0).expand([_node("doc-a", 2)], _partition())
        == {}
    )


async def test_the_budget_bounds_total_context() -> None:
    """
    A query matching many anchors must not multiply its own cost by the radius,
    which would re-create the index inflation that removing overlap prevented.
    """
    context = await _expander(max_context_nodes=1).expand(
        [_node("doc-a", 1), _node("doc-a", 4)], _partition()
    )

    assert sum(len(nodes) for nodes in context.values()) == 1


async def test_a_zero_budget_yields_nothing() -> None:
    assert (
        await _expander(max_context_nodes=0).expand([_node("doc-a", 1)], _partition())
        == {}
    )


async def test_no_anchors_yields_nothing() -> None:
    assert await _expander().expand([], _partition()) == {}


# ===========================================================================
# EFFICIENCY AND FAILURE
# ===========================================================================


async def test_ordinals_are_fetched_once_per_document() -> None:
    """
    Several anchors in one document cost one request rather than one each, and a
    neighbour two anchors share is fetched once.
    """
    store = _NodeStore()

    await _expander(store).expand(
        [_node("doc-a", 1), _node("doc-a", 4), _node("doc-b", 1)], _partition()
    )

    assert len(store.calls) == 2
    for _, ordinals in store.calls:
        assert len(set(ordinals)) == len(ordinals)


async def test_expansion_does_not_traverse_relations() -> None:
    """
    Expansion runs on the hot path of every query, and a graph round trip per
    result would be the dominant cost of a capability that is arithmetic.
    """
    assert "relations_from" not in inspect.getsource(expansion_module)


async def test_a_store_failure_degrades_rather_than_failing_the_query() -> None:
    """The anchors remain complete evidence without context."""
    store = _NodeStore(failure=RuntimeError("store unreachable"))

    with pytest.raises(ExpansionError) as excinfo:
        await _expander(store).expand([_node("doc-a", 2)], _partition())

    assert isinstance(excinfo.value.__cause__, RuntimeError)
    assert excinfo.value.degradable is True


def test_an_anchor_violation_is_not_degradable() -> None:
    """
    It would inflate corroboration, producing confidence that errs toward
    overconfidence, which is the worst outcome available to this system.
    """
    assert AnchorViolationError.degradable is False
    assert issubclass(AnchorViolationError, ExpansionError)


def test_the_invariants_are_verified_after_assignment() -> None:
    """
    Defence in depth. Assignment already excludes anchors and repeats, so the
    check should never fire; it exists because a breach inflates corroboration.
    """
    source = inspect.getsource(expansion_module)

    assert "_verify" in source
    assert "AnchorViolationError" in source


# ===========================================================================
# PROPERTIES
# ===========================================================================

_ANCHORS = st.lists(
    st.tuples(
        st.sampled_from(sorted(DOCUMENTS)),
        st.integers(min_value=0, max_value=5),
    ),
    min_size=0,
    max_size=8,
    unique=True,
)


def _build(pairs: Sequence[tuple[str, int]]) -> list[KnowledgeNode]:
    return [
        _node(document_id, ordinal)
        for document_id, ordinal in pairs
        if ordinal < DOCUMENTS[document_id]
    ]


@settings(max_examples=200, deadline=None)
@given(
    pairs=_ANCHORS,
    radius=st.integers(min_value=1, max_value=3),
    budget=st.integers(min_value=1, max_value=20),
)
async def test_property_an_anchor_is_never_context(
    pairs: Sequence[tuple[str, int]], radius: int, budget: int
) -> None:
    anchors = _build(pairs)

    context = await _expander(
        neighbour_radius=radius, max_context_nodes=budget
    ).expand(anchors, _partition())

    anchor_ids = {anchor.node_id for anchor in anchors}
    assert not anchor_ids & set(_flatten(context))


@settings(max_examples=200, deadline=None)
@given(
    pairs=_ANCHORS,
    radius=st.integers(min_value=1, max_value=3),
    budget=st.integers(min_value=1, max_value=20),
)
async def test_property_no_context_node_appears_twice(
    pairs: Sequence[tuple[str, int]], radius: int, budget: int
) -> None:
    context = await _expander(
        neighbour_radius=radius, max_context_nodes=budget
    ).expand(_build(pairs), _partition())

    flattened = _flatten(context)
    assert len(flattened) == len(set(flattened))


@settings(max_examples=200, deadline=None)
@given(
    pairs=_ANCHORS,
    radius=st.integers(min_value=1, max_value=3),
    budget=st.integers(min_value=1, max_value=20),
)
async def test_property_the_budget_is_never_exceeded(
    pairs: Sequence[tuple[str, int]], radius: int, budget: int
) -> None:
    context = await _expander(
        neighbour_radius=radius, max_context_nodes=budget
    ).expand(_build(pairs), _partition())

    assert sum(len(nodes) for nodes in context.values()) <= budget


@settings(max_examples=200, deadline=None)
@given(pairs=_ANCHORS, radius=st.integers(min_value=1, max_value=3))
async def test_property_context_stays_within_its_anchors_document(
    pairs: Sequence[tuple[str, int]], radius: int
) -> None:
    anchors = _build(pairs)
    by_id = {anchor.node_id: anchor for anchor in anchors}

    context = await _expander(neighbour_radius=radius).expand(anchors, _partition())

    for anchor_id, nodes in context.items():
        assert all(
            node.document_id == by_id[anchor_id].document_id for node in nodes
        )


@settings(max_examples=150, deadline=None)
@given(pairs=_ANCHORS, radius=st.integers(min_value=1, max_value=3))
async def test_property_expansion_is_deterministic(
    pairs: Sequence[tuple[str, int]], radius: int
) -> None:
    anchors = _build(pairs)
    expander = _expander(neighbour_radius=radius)

    first = await expander.expand(anchors, _partition())
    second = await expander.expand(anchors, _partition())

    assert {key: tuple(n.node_id for n in value) for key, value in first.items()} == {
        key: tuple(n.node_id for n in value) for key, value in second.items()
    }


# ===========================================================================
# Anti-pattern guards
# ===========================================================================


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


def test_the_expander_name_is_stable() -> None:
    assert _expander().name == ORDINAL_EXPANDER_NAME


def test_the_module_imports_no_backend() -> None:
    source = inspect.getsource(expansion_module)

    for forbidden in (
        "import psycopg",
        "import asyncpg",
        "import qdrant",
        "import numpy",
        "from ptfm.plugins",
    ):
        assert forbidden not in source


def test_the_module_names_no_storage_technology() -> None:
    source = inspect.getsource(expansion_module).lower()

    for term in ("postgres", "pgvector", "qdrant", "faiss", "elasticsearch", "neo4j"):
        assert term not in source


def test_the_module_contains_no_domain_vocabulary() -> None:
    source = inspect.getsource(expansion_module).lower()

    for term in ("jurisdiction", "statute", "ticker", " icd", "patient", "court"):
        assert term not in source


def test_the_module_declares_no_calibrated_concept() -> None:
    source = _code_without_docstrings(expansion_module).lower()

    for term in ("confidence", "calibrat", "entail", "truth space"):
        assert term not in source


def test_the_module_declares_no_mutable_state() -> None:
    mutable = [
        name
        for name, value in vars(expansion_module).items()
        if not name.startswith("_") and isinstance(value, (list, dict, set))
    ]

    assert mutable == []


def test_the_module_never_catches_base_exception() -> None:
    assert "except BaseException" not in inspect.getsource(expansion_module)


def test_every_bound_comes_from_configuration() -> None:
    source = inspect.getsource(expansion_module)

    assert "self._config.neighbour_radius" in source
    assert "self._config.max_context_nodes" in source


def test_the_expander_holds_no_assignable_state() -> None:
    """One instance serves all concurrent queries."""
    with pytest.raises(AttributeError):
        _expander().cache = {}  # type: ignore[attr-defined]
