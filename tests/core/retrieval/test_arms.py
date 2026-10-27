"""
Unit tests for the three retrieval arms.

File:
    tests/core/retrieval/test_arms.py

Emphasis
--------
    That an arm returns identifiers and signals and nothing else, that no arm
    fabricates a score it did not measure, and that a failing or slow arm
    degrades rather than failing the query.
"""

from __future__ import annotations

import ast
import asyncio
import inspect
from typing import Any, Optional, Sequence

import pytest

from ptfm.core.retrieval.config import DenseConfig, GraphConfig, SparseConfig
from ptfm.core.retrieval.exceptions import (
    ArmTimeoutError,
    ArmUnavailableError,
    StoreUnavailableError,
)
from ptfm.core.retrieval.models import (
    Candidate,
    EmbeddingPartition,
    FilterField,
    FilterOperator,
    QueryPlan,
    RetrievalFilter,
    SegmentationPartition,
)
from ptfm.core.retrieval.pipeline import arms as arms_module
from ptfm.core.retrieval.pipeline.arms import DenseArm, SparseArm, StructuralArm
from ptfm.core.schemas import DiscoveryMethod, SignalScale


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def _segmentation() -> SegmentationPartition:
    return SegmentationPartition(
        segmentation_fingerprint="1:abc", source_schema_version="1.0"
    )


def _embedding() -> EmbeddingPartition:
    return EmbeddingPartition(
        segmentation=_segmentation(),
        model_identity="BAAI/bge-large-en-v1.5@r1",
        dimension=4,
        normalized=True,
        composition_id="heading_path_prefixed",
        query_encoding="prefix:query: ",
    )


def _filter() -> RetrievalFilter:
    return RetrievalFilter(
        field=FilterField.NODE_TYPE,
        operator=FilterOperator.EQUALS,
        values=("PROSE",),
    )


def _plan(
    *,
    vector: Optional[tuple[float, ...]] = (0.5, 0.5, 0.5, 0.5),
    filters: Sequence[RetrievalFilter] = (),
    top_k: int = 5,
    exact: bool = False,
) -> QueryPlan:
    return QueryPlan(
        query_id="q-1",
        text="the applicable rate",
        vector=vector,
        segmentation=_segmentation(),
        embedding=_embedding() if vector is not None else None,
        top_k=top_k,
        filters=tuple(filters),
        exact=exact,
    )


def _dense_candidates(count: int = 3) -> tuple[Candidate, ...]:
    return tuple(
        Candidate(
            node_id=f"node-{index}",
            method=DiscoveryMethod.DENSE,
            rank=index,
            score=0.9 - index * 0.1,
            scale=SignalScale.COSINE,
        )
        for index in range(count)
    )


def _sparse_candidates(count: int = 2) -> tuple[Candidate, ...]:
    return tuple(
        Candidate(
            node_id=f"node-{index}",
            method=DiscoveryMethod.SPARSE,
            rank=index,
            score=7.5 - index,
            scale=SignalScale.LEXICAL,
        )
        for index in range(count)
    )


class _VectorIndex:
    """Records how it was called, so forwarding is observable."""

    def __init__(
        self,
        results: Optional[Sequence[Candidate]] = None,
        *,
        failure: Optional[Exception] = None,
        delay: float = 0.0,
    ) -> None:
        self._results = results
        self._failure = failure
        self._delay = delay
        self.calls: list[dict[str, Any]] = []

    @property
    def name(self) -> str:
        return "vector"

    @property
    def supports_exact_search(self) -> bool:
        return True

    async def search(
        self,
        vector: Sequence[float],
        partition: Any,
        top_k: int,
        *,
        filters: Sequence[RetrievalFilter] = (),
        exact: bool = False,
    ) -> Sequence[Candidate]:
        self.calls.append(
            {
                "width": len(vector),
                "top_k": top_k,
                "filters": tuple(filters),
                "exact": exact,
            }
        )
        if self._delay:
            await asyncio.sleep(self._delay)
        if self._failure is not None:
            raise self._failure
        return _dense_candidates() if self._results is None else tuple(self._results)

    async def upsert(self, embeddings: Any, partition: Any) -> None:
        return None

    async def count(self, partition: Any) -> int:
        return 0


class _LexicalIndex:
    def __init__(
        self,
        results: Optional[Sequence[Candidate]] = None,
        *,
        failure: Optional[Exception] = None,
        delay: float = 0.0,
    ) -> None:
        self._results = results
        self._failure = failure
        self._delay = delay
        self.calls: list[dict[str, Any]] = []

    @property
    def name(self) -> str:
        return "lexical"

    async def search(
        self,
        query: str,
        partition: Any,
        top_k: int,
        *,
        filters: Sequence[RetrievalFilter] = (),
    ) -> Sequence[Candidate]:
        self.calls.append(
            {"query": query, "top_k": top_k, "filters": tuple(filters)}
        )
        if self._delay:
            await asyncio.sleep(self._delay)
        if self._failure is not None:
            raise self._failure
        return _sparse_candidates() if self._results is None else tuple(self._results)

    async def index_nodes(self, nodes: Any, partition: Any) -> None:
        return None


class _RelationStore:
    def __init__(
        self,
        results: Optional[Sequence[Candidate]] = None,
        *,
        failure: Optional[Exception] = None,
    ) -> None:
        self._results = results
        self._failure = failure
        self.calls: list[dict[str, Any]] = []

    @property
    def name(self) -> str:
        return "relation"

    async def relations_from(
        self, node_ids: Any, partition: Any, *, relation_types: Any = ()
    ) -> dict[str, Any]:
        return {}

    async def put_many(self, relations: Any, partition: Any) -> None:
        return None

    async def structural_search(
        self, predicate: Any, partition: Any, top_k: int
    ) -> Sequence[Candidate]:
        self.calls.append({"predicate": predicate, "top_k": top_k})
        if self._failure is not None:
            raise self._failure
        if self._results is not None:
            return tuple(self._results)
        return tuple(
            Candidate(node_id=f"node-{index}", method=DiscoveryMethod.GRAPH, rank=0)
            for index in range(2)
        )


# ===========================================================================
# SHARED CONTRACT
#
# Every arm must satisfy these. An arm that returned content, or fabricated a
# score, would corrupt evidence or ranking in a way no later stage could
# detect.
# ===========================================================================


def _all_arms() -> list[Any]:
    return [
        DenseArm(_VectorIndex(), DenseConfig()),
        SparseArm(_LexicalIndex(), SparseConfig()),
        StructuralArm(_RelationStore(), GraphConfig(enabled=True)),
    ]


@pytest.mark.parametrize("arm", _all_arms())
def test_every_arm_declares_a_name_and_method(arm: Any) -> None:
    """A result's discovery history names the arm that found it."""
    assert isinstance(arm.name, str) and arm.name
    assert isinstance(arm.method, DiscoveryMethod)


@pytest.mark.parametrize("arm", _all_arms())
def test_every_arm_declares_whether_it_needs_a_vector(arm: Any) -> None:
    """
    Declared so the engine can omit an arm cleanly rather than invoking it and
    converting the resulting failure into a diagnostic that misdescribes the
    cause.
    """
    assert isinstance(arm.requires_vector, bool)


@pytest.mark.parametrize("arm", _all_arms())
async def test_every_arm_returns_candidates_only(arm: Any) -> None:
    """
    Authoritative content comes from the node store alone, so an index holding
    a stale copy cannot change what a citation resolves to.
    """
    results = await arm.search(_plan(filters=(_filter(),)))

    for candidate in results:
        assert isinstance(candidate, Candidate)


@pytest.mark.parametrize("arm", _all_arms())
async def test_every_arm_stamps_its_own_method(arm: Any) -> None:
    results = await arm.search(_plan(filters=(_filter(),)))

    assert all(candidate.method is arm.method for candidate in results)


@pytest.mark.parametrize("arm", _all_arms())
async def test_every_arm_reports_health(arm: Any) -> None:
    assert await arm.health_check() is True


# ===========================================================================
# DENSE
# ===========================================================================


async def test_the_dense_arm_returns_similarity_candidates() -> None:
    arm = DenseArm(_VectorIndex(), DenseConfig())

    results = await arm.search(_plan())

    assert len(results) == 3
    assert all(candidate.scale is SignalScale.COSINE for candidate in results)
    assert [candidate.rank for candidate in results] == [0, 1, 2]


def test_the_dense_arm_declares_that_it_needs_a_vector() -> None:
    assert DenseArm(_VectorIndex(), DenseConfig()).requires_vector is True


async def test_the_dense_arm_does_not_search_without_a_vector() -> None:
    """
    A deployment with no embedding provider, or a query whose encoding
    degraded. The arm was never asked, so it returns nothing rather than
    failing.
    """
    index = _VectorIndex()

    results = await DenseArm(index, DenseConfig()).search(_plan(vector=None))

    assert results == ()
    assert index.calls == []


async def test_the_dense_arm_fetches_at_least_the_requested_depth() -> None:
    """Fusion discards candidates no arm ranked highly, so each arm must
    return at least as many as the caller wants."""
    index = _VectorIndex()

    await DenseArm(index, DenseConfig()).search(_plan(top_k=5))

    assert index.calls[0]["top_k"] >= 5


async def test_the_dense_arm_forwards_filters_to_the_index() -> None:
    """
    Applied during search rather than after it, so a backend does not rank
    results the caller will discard.
    """
    index = _VectorIndex()

    await DenseArm(index, DenseConfig()).search(_plan(filters=(_filter(),)))

    assert len(index.calls[0]["filters"]) == 1


async def test_the_dense_arm_forwards_a_request_for_exactness() -> None:
    """Approximate indexes are not reproducible across rebuilds, so a caller
    promising reproducibility must be able to ask for exact search."""
    index = _VectorIndex()

    await DenseArm(index, DenseConfig()).search(_plan(exact=True))

    assert index.calls[0]["exact"] is True


# ===========================================================================
# SPARSE
# ===========================================================================


async def test_the_sparse_arm_returns_lexical_candidates() -> None:
    arm = SparseArm(_LexicalIndex(), SparseConfig())

    results = await arm.search(_plan())

    assert len(results) == 2
    assert all(candidate.scale is SignalScale.LEXICAL for candidate in results)


def test_the_sparse_arm_needs_no_vector() -> None:
    """
    Not a fallback for dense retrieval but a complement to it. Identifiers,
    quantities and dates are what verification checks and what an embedding
    blurs.
    """
    assert SparseArm(_LexicalIndex(), SparseConfig()).requires_vector is False


async def test_the_sparse_arm_serves_a_vectorless_plan() -> None:
    results = await SparseArm(_LexicalIndex(), SparseConfig()).search(
        _plan(vector=None)
    )

    assert len(results) == 2


async def test_the_sparse_arm_searches_the_prepared_text() -> None:
    """Domain normalization has already been applied; the arm searches what it
    was given."""
    index = _LexicalIndex()

    await SparseArm(index, SparseConfig()).search(_plan())

    assert index.calls[0]["query"] == "the applicable rate"


# ===========================================================================
# STRUCTURAL
# ===========================================================================


def test_the_structural_arm_needs_no_vector() -> None:
    assert (
        StructuralArm(_RelationStore(), GraphConfig(enabled=True)).requires_vector
        is False
    )


async def test_an_unconstrained_structural_search_returns_nothing() -> None:
    """
    Without a constraint the arm has no question to answer. Returning the
    corpus would present arbitrary nodes as responsive to a query they have no
    relation to, and would swamp the arms that did answer it.
    """
    store = _RelationStore()

    results = await StructuralArm(store, GraphConfig(enabled=True)).search(_plan())

    assert results == ()
    assert store.calls == []


async def test_a_constraint_drives_a_structural_search() -> None:
    store = _RelationStore()

    results = await StructuralArm(store, GraphConfig(enabled=True)).search(
        _plan(filters=(_filter(),))
    )

    assert len(results) == 2
    assert len(store.calls) == 1


async def test_an_unordered_constraint_contributes_no_score() -> None:
    """
    The rows of a table have no intrinsic rank with respect to a query. A
    fabricated relevance value is indistinguishable from a measured one by the
    time it reaches fusion, so contributing none is strictly better than
    contributing an invented one.
    """
    results = await StructuralArm(_RelationStore(), GraphConfig(enabled=True)).search(
        _plan(filters=(_filter(),))
    )

    assert all(candidate.score is None for candidate in results)
    assert all(candidate.scale is SignalScale.RANK_ONLY for candidate in results)


async def test_an_ordered_constraint_passes_its_score_through() -> None:
    """
    Where a structural predicate does imply an ordering, the arm reports it
    rather than discarding it. Rank-based fusion uses positions, so an honest
    ordering participates fully.
    """
    ordered = (
        Candidate(
            node_id="node-0",
            method=DiscoveryMethod.GRAPH,
            rank=0,
            score=3.0,
            scale=SignalScale.DISTANCE,
        ),
        Candidate(
            node_id="node-1",
            method=DiscoveryMethod.GRAPH,
            rank=1,
            score=7.0,
            scale=SignalScale.DISTANCE,
        ),
    )

    results = await StructuralArm(
        _RelationStore(ordered), GraphConfig(enabled=True)
    ).search(_plan(filters=(_filter(),)))

    assert [candidate.score for candidate in results] == [3.0, 7.0]
    assert all(candidate.scale is SignalScale.DISTANCE for candidate in results)


async def test_the_structural_arm_preserves_the_stores_ordering() -> None:
    results = await StructuralArm(_RelationStore(), GraphConfig(enabled=True)).search(
        _plan(filters=(_filter(),))
    )

    assert [candidate.node_id for candidate in results] == ["node-0", "node-1"]


# ===========================================================================
# FAILURE AND DEGRADATION
# ===========================================================================


async def test_an_empty_result_is_not_a_failure() -> None:
    """Finding nothing is an answer; only being unable to look is a failure."""
    assert await DenseArm(_VectorIndex([]), DenseConfig()).search(_plan()) == ()


@pytest.mark.parametrize(
    ("arm_factory", "plan_kwargs"),
    [
        (
            lambda failure: DenseArm(
                _VectorIndex(failure=failure), DenseConfig()
            ),
            {},
        ),
        (
            lambda failure: SparseArm(
                _LexicalIndex(failure=failure), SparseConfig()
            ),
            {},
        ),
        (
            lambda failure: StructuralArm(
                _RelationStore(failure=failure), GraphConfig(enabled=True)
            ),
            {"filters": (_filter(),)},
        ),
    ],
)
async def test_an_unexpected_store_failure_is_wrapped_with_its_cause(
    arm_factory: Any, plan_kwargs: dict[str, Any]
) -> None:
    """
    A backend driver may raise anything. Wrapping means the engine sees a
    typed, attributable failure it can convert into a diagnostic.
    """
    arm = arm_factory(RuntimeError("driver reset"))

    with pytest.raises(ArmUnavailableError) as excinfo:
        await arm.search(_plan(**plan_kwargs))

    assert isinstance(excinfo.value.__cause__, RuntimeError)
    assert excinfo.value.degradable is True


async def test_a_typed_store_failure_reaches_the_engine() -> None:
    failure = StoreUnavailableError("connection refused", store_name="vector")
    arm = DenseArm(_VectorIndex(failure=failure), DenseConfig())

    with pytest.raises((ArmUnavailableError, StoreUnavailableError)) as excinfo:
        await arm.search(_plan())

    assert excinfo.value.degradable is True


@pytest.mark.parametrize(
    ("arm_factory", "plan_kwargs"),
    [
        (
            lambda: DenseArm(
                _VectorIndex(delay=0.30), DenseConfig(timeout_seconds=0.05)
            ),
            {},
        ),
        (
            lambda: SparseArm(
                _LexicalIndex(delay=0.30), SparseConfig(timeout_seconds=0.05)
            ),
            {},
        ),
    ],
)
async def test_an_arm_that_exceeds_its_deadline_times_out(
    arm_factory: Any, plan_kwargs: dict[str, Any]
) -> None:
    """
    A slow store must not hold the query open. The deadline is per arm, so one
    unhealthy backend cannot delay the arms that are healthy.
    """
    with pytest.raises(ArmTimeoutError) as excinfo:
        await arm_factory().search(_plan(**plan_kwargs))

    assert excinfo.value.degradable is True


async def test_a_timeout_leaves_no_orphaned_task() -> None:
    """
    An abandoned search would continue to hold a connection after the query
    that issued it had been answered.
    """
    arm = DenseArm(_VectorIndex(delay=5.0), DenseConfig(timeout_seconds=0.05))

    with pytest.raises(ArmTimeoutError):
        await arm.search(_plan())

    pending = [
        task for task in asyncio.all_tasks() if task is not asyncio.current_task()
    ]
    assert pending == []


async def test_an_arm_within_its_deadline_is_unaffected() -> None:
    arm = DenseArm(_VectorIndex([], delay=0.0), DenseConfig(timeout_seconds=2.0))

    assert await arm.search(_plan()) == ()


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


def test_no_arm_names_an_authoritative_type() -> None:
    """Arms cannot return content, so they have no reason to name it."""
    source = inspect.getsource(arms_module)

    for forbidden in ("KnowledgeNode", "RetrievedDocument", "EvidenceItem"):
        assert forbidden not in source


def test_the_module_imports_no_backend_or_plugin() -> None:
    source = inspect.getsource(arms_module)

    for forbidden in (
        "import psycopg",
        "import asyncpg",
        "import qdrant",
        "import numpy",
        "from ptfm.plugins",
    ):
        assert forbidden not in source


def test_the_module_names_no_storage_technology() -> None:
    """The arms search protocols; a backend stays replaceable."""
    source = inspect.getsource(arms_module).lower()

    for term in (
        "postgres",
        "pgvector",
        "qdrant",
        "faiss",
        "milvus",
        "elasticsearch",
        "tantivy",
        "neo4j",
        "hnsw",
        "bm25",
    ):
        assert term not in source


def test_the_module_contains_no_domain_vocabulary() -> None:
    source = inspect.getsource(arms_module).lower()

    for term in ("jurisdiction", "statute", "ticker", " icd", "patient", "court"):
        assert term not in source


def test_the_module_declares_no_calibrated_concept() -> None:
    """
    An arm emits an uncalibrated signal on a named scale. Confidence and
    entailment belong to later modules.
    """
    source = _code_without_docstrings(arms_module).lower()

    for term in ("confidence", "calibrat", "entail", "truth space"):
        assert term not in source


def test_the_module_declares_no_mutable_state() -> None:
    mutable = [
        name
        for name, value in vars(arms_module).items()
        if not name.startswith("_") and isinstance(value, (list, dict, set))
    ]

    assert mutable == []


def test_the_module_never_catches_base_exception() -> None:
    source = inspect.getsource(arms_module)

    assert "except BaseException" not in source


def test_arms_hold_no_assignable_state() -> None:
    """One instance serves all concurrent queries."""
    for arm in _all_arms():
        with pytest.raises(AttributeError):
            arm.cache = {}
