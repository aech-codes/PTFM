"""
Unit and property tests for rank fusion.

File:
    tests/core/retrieval/test_fusion.py

Emphasis
--------
    The two guarantees no later stage can recover from: that a node found by
    several arms becomes ONE entry, and that the ranking does not depend on
    which arm happened to finish first.

    The first is what keeps corroboration honest downstream; the second is what
    makes a result reproducible for audit.
"""

from __future__ import annotations

import ast
import inspect
import itertools
from typing import Any, Sequence

import pytest
from hypothesis import assume, given, settings, strategies as st
from pydantic import ValidationError

from ptfm.core.retrieval.config import FusionConfig
from ptfm.core.retrieval.exceptions import FusionError
from ptfm.core.retrieval.models import ArmResult, Candidate
from ptfm.core.retrieval.pipeline import fusion as fusion_module
from ptfm.core.retrieval.pipeline.fusion import (
    RECIPROCAL_RANK_FUSION_NAME,
    ReciprocalRankFusion,
)
from ptfm.core.schemas import DiscoveryMethod, SignalScale

#: Each arm's method paired with the scale its scores live on, so a fused
#: result can be checked for having preserved both.
_ARMS: tuple[tuple[str, DiscoveryMethod, SignalScale], ...] = (
    ("dense", DiscoveryMethod.DENSE, SignalScale.COSINE),
    ("sparse", DiscoveryMethod.SPARSE, SignalScale.LEXICAL),
    ("graph", DiscoveryMethod.GRAPH, SignalScale.RANK_ONLY),
)


def _arm(
    name: str,
    method: DiscoveryMethod,
    node_ids: Sequence[str],
    scale: SignalScale = SignalScale.COSINE,
) -> ArmResult:
    scored = scale is not SignalScale.RANK_ONLY
    return ArmResult(
        arm_name=name,
        method=method,
        candidates=tuple(
            Candidate(
                node_id=node_id,
                method=method,
                rank=rank,
                score=(1.0 - rank * 0.1) if scored else None,
                scale=scale,
            )
            for rank, node_id in enumerate(node_ids)
        ),
    )


def _fusion(**overrides: Any) -> ReciprocalRankFusion:
    return ReciprocalRankFusion(FusionConfig(**overrides))


def _three_arms() -> tuple[ArmResult, ArmResult, ArmResult]:
    """Dense and sparse disagree on order; the graph arm corroborates one node."""
    return (
        _arm("dense", DiscoveryMethod.DENSE, ["a", "b", "c"], SignalScale.COSINE),
        _arm("sparse", DiscoveryMethod.SPARSE, ["b", "a", "d"], SignalScale.LEXICAL),
        _arm("graph", DiscoveryMethod.GRAPH, ["b"], SignalScale.RANK_ONLY),
    )


# ===========================================================================
# ONE NODE, ONE ENTRY
# ===========================================================================


def test_a_node_found_by_several_arms_becomes_one_entry() -> None:
    """
    Three arms finding one passage is one passage corroborated three ways, not
    three passages. Corroboration is counted downstream by distinct anchors, so
    duplication here would inflate it in the quantity the system exists to
    calibrate.
    """
    results = _fusion().fuse(_three_arms(), top_k=10)

    node_ids = [candidate.node_id for candidate in results]
    assert len(node_ids) == len(set(node_ids))

    corroborated = next(c for c in results if c.node_id == "b")
    assert len(corroborated.signals) == 3
    assert corroborated.arm_count == 3


def test_every_node_any_arm_found_survives() -> None:
    """Fusion reorders; it does not discard."""
    results = _fusion().fuse(_three_arms(), top_k=10)

    assert {candidate.node_id for candidate in results} == {"a", "b", "c", "d"}


def test_agreement_across_arms_raises_a_node() -> None:
    """
    The property that makes hybrid retrieval worth running: a node several
    methods agree on outranks one only a single method found, without any arm's
    score being trusted in absolute terms.
    """
    results = _fusion().fuse(_three_arms(), top_k=10)

    assert results[0].node_id == "b"


def test_fused_ranks_are_contiguous_from_zero() -> None:
    results = _fusion().fuse(_three_arms(), top_k=10)

    assert [c.fused_rank for c in results] == list(range(len(results)))


# ===========================================================================
# RANKS, NOT SCORES
# ===========================================================================


def test_each_signal_keeps_its_own_scale() -> None:
    """
    Scores are not discarded from the result, only from the arithmetic. A
    consumer can still inspect what each arm observed and on what scale.
    """
    results = _fusion().fuse(_three_arms(), top_k=10)

    corroborated = next(c for c in results if c.node_id == "b")
    assert {signal.scale for signal in corroborated.signals} == {
        SignalScale.COSINE,
        SignalScale.LEXICAL,
        SignalScale.RANK_ONLY,
    }


def test_each_signal_keeps_its_own_rank() -> None:
    results = _fusion().fuse(_three_arms(), top_k=10)

    corroborated = next(c for c in results if c.node_id == "b")
    assert sorted(signal.rank for signal in corroborated.signals) == [0, 0, 1]


def test_an_unscored_arm_contributes_a_signal_without_a_score() -> None:
    """A structural predicate that implies no ordering still participates by
    rank."""
    results = _fusion().fuse(_three_arms(), top_k=10)

    corroborated = next(c for c in results if c.node_id == "b")
    assert any(signal.raw_score is None for signal in corroborated.signals)
    assert any(signal.raw_score is not None for signal in corroborated.signals)


def test_the_fused_value_is_the_rank_reciprocal_sum() -> None:
    """
    The equation, asserted rather than assumed:

        score(node) = sum over arms of weight / (rank_constant + rank)
    """
    constant = FusionConfig().rank_constant
    results = _fusion().fuse(_three_arms(), top_k=10)

    corroborated = next(c for c in results if c.node_id == "b")
    expected = 1 / (constant + 1) + 1 / (constant + 0) + 1 / (constant + 0)

    assert corroborated.fused_score == pytest.approx(expected, abs=1e-12)


def test_the_fused_value_declares_an_ordinal_scale() -> None:
    """
    A reciprocal rank value is derived from positions alone and carries no
    probabilistic meaning. Its scale says so, one module away from the layer
    that does produce probabilities.
    """
    results = _fusion().fuse(_three_arms(), top_k=10)

    assert all(c.fused_scale is SignalScale.RECIPROCAL_RANK for c in results)


def test_fusion_never_reads_a_score_arithmetically() -> None:
    """
    The reason rank fusion is the default. A cosine similarity and a lexical
    relevance value occupy different ranges with different meanings, so
    combining them numerically produces something that resembles a measurement
    and is not one.
    """
    constant = FusionConfig().rank_constant
    ordinary = _arm("dense", DiscoveryMethod.DENSE, ["x", "y"], SignalScale.COSINE)

    inflated = ArmResult(
        arm_name="dense",
        method=DiscoveryMethod.DENSE,
        candidates=(
            Candidate(
                node_id="x",
                method=DiscoveryMethod.DENSE,
                rank=0,
                score=999.0,
                scale=SignalScale.COSINE,
            ),
            Candidate(
                node_id="y",
                method=DiscoveryMethod.DENSE,
                rank=1,
                score=998.0,
                scale=SignalScale.COSINE,
            ),
        ),
    )

    assert [c.fused_score for c in _fusion().fuse([ordinary], 10)] == [
        pytest.approx(1 / constant),
        pytest.approx(1 / (constant + 1)),
    ]
    assert [c.fused_score for c in _fusion().fuse([inflated], 10)] == [
        pytest.approx(1 / constant),
        pytest.approx(1 / (constant + 1)),
    ]


# ===========================================================================
# DETERMINISM
# ===========================================================================


def test_arm_completion_order_cannot_change_the_ranking() -> None:
    """
    The arms run concurrently and finish in whatever order the network allows.
    If that affected ranking, the same query against an unchanged corpus would
    rank differently run to run and no result could be reproduced for audit.
    """
    arms = _three_arms()
    fusion = _fusion()

    baseline = [
        (c.node_id, round(c.fused_score or 0.0, 12)) for c in fusion.fuse(arms, 10)
    ]

    for permutation in itertools.permutations(arms):
        assert [
            (c.node_id, round(c.fused_score or 0.0, 12))
            for c in fusion.fuse(list(permutation), 10)
        ] == baseline


def test_signal_order_is_deterministic() -> None:
    """
    A consumer diffing two runs should see a difference only where the
    retrieval genuinely differed, not where the arms happened to reply in
    another order.
    """
    arms = _three_arms()
    fusion = _fusion()

    orders = {
        tuple(signal.method for signal in fusion.fuse(list(p), 10)[0].signals)
        for p in itertools.permutations(arms)
    }

    assert len(orders) == 1


def test_equal_values_break_the_tie_on_identifier() -> None:
    """
    Stable across processes and index rebuilds, which a tie broken by arrival
    order would not be.
    """
    first = _arm("dense", DiscoveryMethod.DENSE, ["z", "y"], SignalScale.COSINE)
    second = _arm("sparse", DiscoveryMethod.SPARSE, ["y", "z"], SignalScale.LEXICAL)
    fusion = _fusion()

    assert [c.node_id for c in fusion.fuse([first, second], 10)] == ["y", "z"]
    assert [c.node_id for c in fusion.fuse([second, first], 10)] == ["y", "z"]


# ===========================================================================
# BOUNDS AND EMPTY INPUT
# ===========================================================================


def test_top_k_truncates_from_the_bottom() -> None:
    arms = _three_arms()
    fusion = _fusion()

    full = [c.node_id for c in fusion.fuse(arms, 10)]

    assert [c.node_id for c in fusion.fuse(arms, 2)] == full[:2]


def test_a_top_k_below_one_is_refused() -> None:
    with pytest.raises(FusionError):
        _fusion().fuse([_arm("dense", DiscoveryMethod.DENSE, ["a"])], 0)


def test_no_arms_yields_no_results() -> None:
    """
    Whether an empty result should fail the query is the engine's decision, not
    fusion's. Fusion reports what it was given.
    """
    assert _fusion().fuse([], 10) == ()


def test_an_arm_that_found_nothing_lowers_no_node() -> None:
    """Finding nothing is an answer, and it must not penalise the arms that
    found something."""
    dense = _arm("dense", DiscoveryMethod.DENSE, ["a", "b"], SignalScale.COSINE)
    empty = _arm("sparse", DiscoveryMethod.SPARSE, [], SignalScale.LEXICAL)

    with_empty = _fusion().fuse([dense, empty], 10)
    without = _fusion().fuse([dense], 10)

    assert [c.node_id for c in with_empty] == [c.node_id for c in without]
    assert [c.fused_score for c in with_empty] == [c.fused_score for c in without]


# ===========================================================================
# WEIGHTING
# ===========================================================================


def test_an_unweighted_symmetric_case_ties() -> None:
    dense = _arm("dense", DiscoveryMethod.DENSE, ["a", "b"], SignalScale.COSINE)
    sparse = _arm("sparse", DiscoveryMethod.SPARSE, ["b", "a"], SignalScale.LEXICAL)

    results = _fusion().fuse([dense, sparse], 10)

    assert results[0].fused_score == pytest.approx(results[1].fused_score)


def test_a_weight_raises_its_method() -> None:
    dense = _arm("dense", DiscoveryMethod.DENSE, ["a", "b"], SignalScale.COSINE)
    sparse = _arm("sparse", DiscoveryMethod.SPARSE, ["b", "a"], SignalScale.LEXICAL)

    weighted = _fusion(method_weights={"DENSE": 5.0})

    assert [c.node_id for c in weighted.fuse([dense, sparse], 10)] == ["a", "b"]


def test_weighting_scales_the_rank_contribution_not_a_score() -> None:
    """A weighted deployment remains scale-free."""
    constant = FusionConfig().rank_constant
    dense = _arm("dense", DiscoveryMethod.DENSE, ["a"], SignalScale.COSINE)

    results = _fusion(method_weights={"DENSE": 3.0}).fuse([dense], 10)

    assert results[0].fused_score == pytest.approx(3.0 / constant)


def test_a_zero_weight_excludes_a_method_from_the_ranking() -> None:
    dense = _arm("dense", DiscoveryMethod.DENSE, ["a", "b"], SignalScale.COSINE)
    sparse = _arm("sparse", DiscoveryMethod.SPARSE, ["b", "a"], SignalScale.LEXICAL)

    results = _fusion(method_weights={"SPARSE": 0.0}).fuse([dense, sparse], 10)

    assert [c.node_id for c in results] == ["a", "b"]


def test_a_zero_weight_still_records_the_observation() -> None:
    """
    How an arm can inform a consumer without influencing order. The signal is
    provenance; the weight is policy.
    """
    dense = _arm("dense", DiscoveryMethod.DENSE, ["a"], SignalScale.COSINE)
    sparse = _arm("sparse", DiscoveryMethod.SPARSE, ["a"], SignalScale.LEXICAL)

    results = _fusion(method_weights={"SPARSE": 0.0}).fuse([dense, sparse], 10)

    assert any(
        signal.method is DiscoveryMethod.SPARSE for signal in results[0].signals
    )


def test_a_weight_key_is_case_insensitive() -> None:
    assert _fusion(method_weights={"dense": 2.0}) is not None


def test_a_weight_naming_no_known_method_is_refused() -> None:
    """
    An operator who misspells a method would otherwise believe a weighting was
    in force while the ranking stayed unweighted, with nothing reporting the
    discrepancy.
    """
    with pytest.raises(FusionError, match="not a discovery"):
        _fusion(method_weights={"DENCE": 1.0})


def test_a_negative_weight_is_refused() -> None:
    """It would lower a node because an arm found it, inverting what a
    retrieval observation means."""
    with pytest.raises((FusionError, ValidationError)):
        _fusion(method_weights={"DENSE": -1.0})


def test_a_partial_weight_map_is_accepted() -> None:
    """
    A method absent from the map carries a weight of one, so raising one arm
    does not require restating the others and the map needs no revision when a
    method is added.
    """
    assert FusionConfig(method_weights={"SPARSE": 0.0}) is not None
    assert FusionConfig(method_weights={"DENSE": 2.0}) is not None


def test_zeroing_every_method_is_refused() -> None:
    """A successful search would return an empty ordering."""
    with pytest.raises(ValidationError):
        FusionConfig(
            method_weights={method.value: 0.0 for method in DiscoveryMethod}
        )


# ===========================================================================
# MISCONFIGURATION
# ===========================================================================


def test_two_arms_sharing_a_method_are_refused() -> None:
    """
    A signal records the method that produced it, and a fused candidate admits
    one signal per method. Discarding one arm's observation instead would leave
    the discovery history incomplete while reporting success.
    """
    first = _arm("dense", DiscoveryMethod.DENSE, ["a"], SignalScale.COSINE)
    second = _arm("dense_v2", DiscoveryMethod.DENSE, ["a"], SignalScale.COSINE)

    with pytest.raises(FusionError) as excinfo:
        _fusion().fuse([first, second], 10)

    assert "dense_v2" in excinfo.value.message


def test_the_duplicate_method_guard_fires_even_without_a_shared_node() -> None:
    """
    Caught as a configuration fault rather than waiting for the first query
    whose arms happen to overlap.
    """
    first = _arm("dense", DiscoveryMethod.DENSE, ["a"], SignalScale.COSINE)
    second = _arm("dense_v2", DiscoveryMethod.DENSE, ["z"], SignalScale.COSINE)

    with pytest.raises(FusionError):
        _fusion().fuse([first, second], 10)


# ===========================================================================
# PROPERTIES
# ===========================================================================

_NODE_IDS = st.lists(
    st.sampled_from([f"node-{index}" for index in range(12)]),
    min_size=0,
    max_size=6,
    unique=True,
)


@st.composite
def _arm_results(draw: Any) -> list[ArmResult]:
    chosen = draw(
        st.lists(st.sampled_from(_ARMS), min_size=0, max_size=3, unique_by=lambda a: a[1])
    )
    return [
        _arm(name, method, draw(_NODE_IDS), scale) for name, method, scale in chosen
    ]


@settings(max_examples=250)
@given(results=_arm_results(), top_k=st.integers(min_value=1, max_value=15))
def test_property_fusion_never_duplicates_a_node(
    results: list[ArmResult], top_k: int
) -> None:
    fused = _fusion().fuse(results, top_k)

    node_ids = [candidate.node_id for candidate in fused]
    assert len(node_ids) == len(set(node_ids))


@settings(max_examples=250)
@given(results=_arm_results(), top_k=st.integers(min_value=1, max_value=15))
def test_property_fusion_is_order_independent(
    results: list[ArmResult], top_k: int
) -> None:
    fusion = _fusion()
    baseline = [
        (c.node_id, round(c.fused_score or 0.0, 12)) for c in fusion.fuse(results, top_k)
    ]

    for permutation in itertools.permutations(results):
        assert [
            (c.node_id, round(c.fused_score or 0.0, 12))
            for c in fusion.fuse(list(permutation), top_k)
        ] == baseline


@settings(max_examples=250)
@given(results=_arm_results(), top_k=st.integers(min_value=1, max_value=15))
def test_property_fusion_respects_its_bound(
    results: list[ArmResult], top_k: int
) -> None:
    assert len(_fusion().fuse(results, top_k)) <= top_k


@settings(max_examples=250)
@given(results=_arm_results(), top_k=st.integers(min_value=1, max_value=15))
def test_property_values_are_non_increasing(
    results: list[ArmResult], top_k: int
) -> None:
    values = [c.fused_score or 0.0 for c in _fusion().fuse(results, top_k)]

    assert all(
        values[index] >= values[index + 1] - 1e-15
        for index in range(len(values) - 1)
    )


@settings(max_examples=250)
@given(results=_arm_results(), top_k=st.integers(min_value=1, max_value=15))
def test_property_every_found_node_survives_when_the_bound_allows(
    results: list[ArmResult], top_k: int
) -> None:
    union = {node_id for result in results for node_id in result.node_ids}
    assume(len(union) <= top_k)

    fused = _fusion().fuse(results, top_k)

    assert {candidate.node_id for candidate in fused} == union


@settings(max_examples=200)
@given(results=_arm_results(), top_k=st.integers(min_value=1, max_value=15))
def test_property_ranks_are_contiguous_from_zero(
    results: list[ArmResult], top_k: int
) -> None:
    fused = _fusion().fuse(results, top_k)

    assert [c.fused_rank for c in fused] == list(range(len(fused)))


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


def test_the_strategy_name_matches_the_configuration_default() -> None:
    assert _fusion().name == RECIPROCAL_RANK_FUSION_NAME == FusionConfig().strategy


def test_fusion_is_synchronous() -> None:
    """It performs no I/O, and declaring it async would invite an
    implementation to acquire some."""
    assert "async def" not in inspect.getsource(fusion_module)


def test_fusion_touches_no_store() -> None:
    source = inspect.getsource(fusion_module)

    for forbidden in ("NodeStore", "VectorIndex", "LexicalIndex", "RelationStore"):
        assert forbidden not in source


def test_fusion_handles_no_authoritative_type() -> None:
    """It ranks identifiers; content arrives later, from the node store."""
    source = inspect.getsource(fusion_module)

    for forbidden in ("KnowledgeNode", "RetrievedDocument", "EvidenceItem"):
        assert forbidden not in source


def test_the_module_declares_no_calibrated_concept() -> None:
    source = _code_without_docstrings(fusion_module).lower()

    for term in ("confidence", "calibrat", "entail", "truth space"):
        assert term not in source


def test_the_module_names_no_storage_technology() -> None:
    source = inspect.getsource(fusion_module).lower()

    for term in ("postgres", "qdrant", "faiss", "elasticsearch", "neo4j", "bm25"):
        assert term not in source


def test_the_module_contains_no_domain_vocabulary() -> None:
    source = inspect.getsource(fusion_module).lower()

    for term in ("jurisdiction", "statute", "ticker", " icd", "patient", "court"):
        assert term not in source


def test_the_module_declares_no_mutable_state() -> None:
    mutable = [
        name
        for name, value in vars(fusion_module).items()
        if not name.startswith("_") and isinstance(value, (list, dict, set))
    ]

    assert mutable == []


def test_the_strategy_holds_no_assignable_state() -> None:
    """One instance serves all concurrent queries."""
    with pytest.raises(AttributeError):
        _fusion().cache = {}  # type: ignore[attr-defined]


def test_the_rank_constant_is_not_hardcoded() -> None:
    """Every operational number comes from configuration."""
    source = inspect.getsource(fusion_module)

    assert "config.rank_constant" in source
    assert "+ 60" not in source
