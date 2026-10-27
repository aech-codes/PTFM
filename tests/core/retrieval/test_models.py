"""
Unit and property tests for the Retrieval Module internal models.

File:
    tests/core/retrieval/test_models.py

Emphasis
--------
    The two invariants no later stage can recover from: that partitions make
    an incomparable comparison unexpressible, and that a node found by several
    arms is one finding rather than several.
"""

from __future__ import annotations

import ast
import inspect
from typing import Any

import pytest
from hypothesis import given, settings, strategies as st
from pydantic import ValidationError

from ptfm.core.retrieval import models as models_module
from ptfm.core.retrieval.exceptions import ArmUnavailableError
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
from ptfm.core.schemas import DiscoveryMethod, RetrievalSignal, SignalScale


def _code_without_docstrings(module: object) -> str:
    """
    Return a module's source with every docstring removed.

    A prohibition is documented in docstrings, so a naive substring scan flags
    the sentences stating the rule as though they violated it. Line ranges are
    used rather than text replacement because ``ast.get_docstring`` dedents its
    result, which does not match indented source.

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


def _normalized_doc(obj: object) -> str:
    """
    Return an object's docstring with all whitespace collapsed to single
    spaces.

    Docstrings are wrapped for readability, so a phrase that reads as one line
    may be split across two in the source. A substring check against the raw
    text therefore fails for reasons of formatting rather than of content,
    which makes the assertion test the line width instead of the claim.

    Parameters
    ----------
    obj:
        The object whose docstring is read.

    Returns
    -------
    str
        The docstring with whitespace normalized, empty when there is none.
    """
    return " ".join((inspect.getdoc(obj) or "").split())


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def _segmentation(**overrides: Any) -> SegmentationPartition:
    fields: dict[str, Any] = {
        "segmentation_fingerprint": "1:abc",
        "source_schema_version": "1.0",
    }
    fields.update(overrides)
    return SegmentationPartition(**fields)


def _embedding(**overrides: Any) -> EmbeddingPartition:
    fields: dict[str, Any] = {
        "segmentation": _segmentation(),
        "model_identity": "BAAI/bge-large-en-v1.5@r1",
        "dimension": 1024,
        "normalized": True,
        "composition_id": "heading_path_prefixed",
        "query_encoding": "prefix:query: ",
    }
    fields.update(overrides)
    return EmbeddingPartition(**fields)


def _candidate(rank: int, node_id: str | None = None, **overrides: Any) -> Candidate:
    fields: dict[str, Any] = {
        "node_id": node_id or f"node-{rank}",
        "method": DiscoveryMethod.DENSE,
        "rank": rank,
        "score": 1.0 - rank * 0.1,
        "scale": SignalScale.COSINE,
    }
    fields.update(overrides)
    return Candidate(**fields)


def _signal(method: DiscoveryMethod, rank: int) -> RetrievalSignal:
    return RetrievalSignal(
        method=method, rank=rank, raw_score=None, scale=SignalScale.RANK_ONLY
    )


# ===========================================================================
# PARTITIONS
# ===========================================================================


def test_a_segmentation_key_is_deterministic() -> None:
    assert _segmentation().key == _segmentation().key


def test_a_differing_fingerprint_yields_a_different_partition() -> None:
    """
    Node identifiers derive in part from the fingerprint, so partitions under
    differing fingerprints occupy disjoint identifier spaces.
    """
    assert _segmentation().key != _segmentation(segmentation_fingerprint="1:def").key


def test_the_document_contract_version_participates() -> None:
    """
    A span is an offset into document text. A contract altering normalization
    would shift every offset while the segmentation fingerprint stayed
    identical, so spans from two contracts must not share a partition.
    """
    assert _segmentation().key != _segmentation(source_schema_version="2.0").key


def test_a_partition_accepts_only_matching_nodes() -> None:
    partition = _segmentation()

    assert partition.accepts("1:abc", "1.0") is True
    assert partition.accepts("1:def", "1.0") is False
    assert partition.accepts("1:abc", "2.0") is False


def test_an_embedding_key_is_deterministic() -> None:
    assert _embedding().key == _embedding().key


@pytest.mark.parametrize(
    "override",
    [
        {"model_identity": "intfloat/multilingual-e5-large@r1"},
        {"dimension": 768},
        {"normalized": False},
        {"composition_id": "text_only"},
        {"query_encoding": ""},
    ],
)
def test_every_component_separates_the_embedding_space(
    override: dict[str, Any],
) -> None:
    """
    Each of these is a reason two vectors cannot be compared, so each must
    yield a distinct partition. A similarity across any of them is
    arithmetically valid and semantically meaningless.
    """
    assert not _embedding().describes_same_space(_embedding(**override))


def test_dimension_is_not_redundant_with_model_identity() -> None:
    """
    Several models expose truncated representations at more than one width
    from a single set of weights, so identity alone does not determine
    dimension.
    """
    assert _embedding(dimension=1024).key != _embedding(dimension=256).key


def test_an_embedding_partition_is_subordinate_to_its_segmentation() -> None:
    """Vectors are meaningless without the nodes they belong to."""
    other = _embedding(segmentation=_segmentation(segmentation_fingerprint="1:def"))

    assert not _embedding().describes_same_space(other)


def test_the_query_convention_is_verifiable_before_a_search() -> None:
    """
    An incompatible pairing degrades retrieval measurably and produces no
    error of its own, so it must be checked rather than inferred from results.
    """
    partition = _embedding()

    assert partition.accepts_query_encoding("prefix:query: ") is True
    assert partition.accepts_query_encoding("") is False


def test_partitions_are_frozen() -> None:
    partition = _segmentation()

    with pytest.raises(ValidationError):
        partition.segmentation_fingerprint = "1:other"  # type: ignore[misc]


# ===========================================================================
# FILTERS
# ===========================================================================


def test_a_filter_builds() -> None:
    filter_ = RetrievalFilter(
        field=FilterField.NODE_TYPE,
        operator=FilterOperator.EQUALS,
        values=("PROSE",),
    )

    assert filter_.values == ("PROSE",)


def test_a_metadata_filter_must_name_its_key() -> None:
    with pytest.raises(ValidationError, match="metadata key"):
        RetrievalFilter(
            field=FilterField.METADATA,
            operator=FilterOperator.EQUALS,
            values=("x",),
        )


def test_a_non_metadata_filter_rejects_a_key() -> None:
    with pytest.raises(ValidationError, match="meaningful only"):
        RetrievalFilter(
            field=FilterField.NODE_TYPE,
            operator=FilterOperator.EQUALS,
            values=("PROSE",),
            metadata_key="topic",
        )


def test_a_filter_requires_a_value() -> None:
    with pytest.raises(ValidationError, match="at least one value"):
        RetrievalFilter(
            field=FilterField.NODE_TYPE,
            operator=FilterOperator.EQUALS,
            values=(),
        )


def test_a_single_valued_operator_rejects_several_values() -> None:
    with pytest.raises(ValidationError, match="exactly one value"):
        RetrievalFilter(
            field=FilterField.NODE_TYPE,
            operator=FilterOperator.EQUALS,
            values=("PROSE", "CODE"),
        )


def test_filter_fields_are_enumerated_not_free_form() -> None:
    """
    Enumeration is what keeps filtering safe and domain-neutral: query text
    cannot reach a position where a backend parses it as a field name, and no
    member presumes a document domain.
    """
    assert issubclass(FilterField, __import__("enum").StrEnum)

    values = {member.value for member in FilterField}
    for domain_term in ("JURISDICTION", "TICKER", "ICD_CODE", "PATIENT_ID"):
        assert domain_term not in values


def test_domain_specific_filtering_goes_through_metadata() -> None:
    """The engine treats the key as opaque; a plugin gives it meaning."""
    filter_ = RetrievalFilter(
        field=FilterField.METADATA,
        operator=FilterOperator.EQUALS,
        values=("NASDAQ:AAPL",),
        metadata_key="instrument",
    )

    assert filter_.metadata_key == "instrument"


# ===========================================================================
# QUERY PLAN
# ===========================================================================


def test_a_plan_without_a_vector_is_valid() -> None:
    """A deployment may run without an embedding provider, and lexical and
    structural retrieval continue."""
    plan = QueryPlan(
        query_id="q-1", text="applicable rate", segmentation=_segmentation(), top_k=10
    )

    assert plan.has_vector is False


def test_a_vector_requires_its_partition() -> None:
    """Comparability must be establishable before the search runs."""
    with pytest.raises(ValidationError, match="before the search runs"):
        QueryPlan(
            query_id="q-1",
            text="rate",
            vector=(0.1, 0.2),
            segmentation=_segmentation(),
            top_k=10,
        )


def test_a_partition_requires_a_vector() -> None:
    with pytest.raises(ValidationError, match="cannot run"):
        QueryPlan(
            query_id="q-1",
            text="rate",
            segmentation=_segmentation(),
            embedding=_embedding(),
            top_k=10,
        )


def test_the_partitions_must_describe_one_corpus() -> None:
    """Otherwise vectors would be compared against nodes they were not
    produced from."""
    with pytest.raises(ValidationError, match="were not produced from"):
        QueryPlan(
            query_id="q-1",
            text="rate",
            vector=(0.1,),
            segmentation=_segmentation(segmentation_fingerprint="1:def"),
            embedding=_embedding(),
            top_k=10,
        )


def test_exact_search_is_requestable() -> None:
    """A caller promising reproducibility must be able to ask for the slower
    guarantee."""
    plan = QueryPlan(
        query_id="q-1",
        text="rate",
        segmentation=_segmentation(),
        top_k=10,
        exact=True,
    )

    assert plan.exact is True


# ===========================================================================
# CANDIDATES
# ===========================================================================


def test_a_candidate_carries_no_content() -> None:
    """
    An index must not be able to determine what a citation resolves to.
    Authoritative content arrives later, from the node store.
    """
    fields = set(Candidate.model_fields)

    for forbidden in ("text", "content", "span", "source_uri", "document_id"):
        assert forbidden not in fields


def test_a_score_must_name_its_scale() -> None:
    """Comparing scores from different arms must be visibly wrong rather than
    silently plausible."""
    with pytest.raises(ValidationError, match="scale that score lives on"):
        Candidate(
            node_id="n-1", method=DiscoveryMethod.DENSE, rank=0, score=0.5
        )


def test_a_scale_must_accompany_a_score() -> None:
    with pytest.raises(ValidationError, match="score it does not carry"):
        Candidate(
            node_id="n-1",
            method=DiscoveryMethod.DENSE,
            rank=0,
            scale=SignalScale.COSINE,
        )


def test_an_arm_may_order_without_scoring() -> None:
    """A structural predicate produces a set, not a ranking, and inventing a
    score to justify the arm would fabricate a signal."""
    candidate = Candidate(node_id="n-1", method=DiscoveryMethod.GRAPH, rank=0)

    assert candidate.scale is SignalScale.RANK_ONLY
    assert candidate.score is None


def test_a_candidate_renders_as_a_signal() -> None:
    signal = _candidate(0).as_signal()

    assert signal.method is DiscoveryMethod.DENSE
    assert signal.scale is SignalScale.COSINE


# ===========================================================================
# ARM RESULTS
# ===========================================================================


def test_an_arm_result_builds() -> None:
    result = ArmResult(
        arm_name="dense",
        method=DiscoveryMethod.DENSE,
        candidates=tuple(_candidate(i) for i in range(3)),
    )

    assert len(result) == 3
    assert result.node_ids == ("node-0", "node-1", "node-2")


def test_ranks_must_be_contiguous_from_zero() -> None:
    """Rank-based fusion treats ranks as an ordering, so a gap would distort
    every fused value derived from them."""
    with pytest.raises(ValidationError, match="contiguous from zero"):
        ArmResult(
            arm_name="dense",
            method=DiscoveryMethod.DENSE,
            candidates=(_candidate(0), _candidate(2)),
        )


def test_one_arm_may_not_return_a_node_twice() -> None:
    """Repeating it would weight that node twice in fusion."""
    with pytest.raises(ValidationError, match="twice"):
        ArmResult(
            arm_name="dense",
            method=DiscoveryMethod.DENSE,
            candidates=(_candidate(0), _candidate(1, node_id="node-0")),
        )


def test_a_candidate_method_must_match_its_arm() -> None:
    with pytest.raises(ValidationError, match="declares method"):
        ArmResult(
            arm_name="sparse",
            method=DiscoveryMethod.SPARSE,
            candidates=(_candidate(0),),
        )


def test_an_empty_arm_result_is_valid() -> None:
    """Finding nothing is an answer; only being unable to look is a failure."""
    result = ArmResult(arm_name="graph", method=DiscoveryMethod.GRAPH)

    assert len(result) == 0


# ===========================================================================
# FUSION — one node, one entry
# ===========================================================================


def test_multi_arm_discovery_is_one_entry_with_several_signals() -> None:
    """
    A node found by three arms is ONE finding corroborated by method, not
    three findings. Corroboration is counted downstream by distinct anchors,
    so duplication here would inflate it in the quantity the system exists to
    calibrate.
    """
    fused = FusedCandidate(
        node_id="node-1",
        signals=(
            _signal(DiscoveryMethod.DENSE, 0),
            _signal(DiscoveryMethod.SPARSE, 2),
            _signal(DiscoveryMethod.GRAPH, 5),
        ),
        fused_rank=0,
    )

    assert fused.arm_count == 3
    assert fused.methods == {
        DiscoveryMethod.DENSE,
        DiscoveryMethod.SPARSE,
        DiscoveryMethod.GRAPH,
    }


def test_arm_count_is_not_corroboration() -> None:
    """
    A diagnostic about retrieval agreement. Corroboration counts distinct
    source passages, and several arms finding one passage is one passage.
    """
    doc = _normalized_doc(FusedCandidate.arm_count.fget)

    assert "explicitly NOT a measure of corroboration" in doc


def test_a_fused_candidate_requires_a_signal() -> None:
    with pytest.raises(ValidationError, match="never found by anything"):
        FusedCandidate(node_id="node-1", signals=(), fused_rank=0)


def test_one_method_may_not_signal_twice() -> None:
    with pytest.raises(ValidationError, match="two signals from one method"):
        FusedCandidate(
            node_id="node-1",
            signals=(
                _signal(DiscoveryMethod.DENSE, 0),
                _signal(DiscoveryMethod.DENSE, 1),
            ),
            fused_rank=0,
        )


def test_a_signal_is_retrievable_by_method() -> None:
    fused = FusedCandidate(
        node_id="node-1",
        signals=(_signal(DiscoveryMethod.DENSE, 3),),
        fused_rank=0,
    )

    assert fused.signal_for(DiscoveryMethod.DENSE).rank == 3
    assert fused.signal_for(DiscoveryMethod.SPARSE) is None


def test_the_fused_value_declares_its_scale() -> None:
    """
    A reciprocal rank fusion value is derived from positions alone and carries
    no probabilistic meaning, so treating it as a confidence would be a
    category error.
    """
    fused = FusedCandidate(
        node_id="node-1", signals=(_signal(DiscoveryMethod.DENSE, 0),), fused_rank=0
    )

    assert fused.fused_scale is SignalScale.RECIPROCAL_RANK


# ===========================================================================
# DIAGNOSTICS
# ===========================================================================


def _diagnostics(*arms: ArmDiagnostic, **overrides: Any) -> RetrievalDiagnostics:
    fields: dict[str, Any] = {
        "query_id": "q-1",
        "segmentation_key": _segmentation().key,
        "arms": arms,
    }
    fields.update(overrides)
    return RetrievalDiagnostics(**fields)


def test_degradation_is_visible() -> None:
    """
    A result assembled from two arms because a third was unavailable is not
    the same result as one assembled from three.
    """
    diagnostics = _diagnostics(
        ArmDiagnostic(
            arm_name="dense",
            method=DiscoveryMethod.DENSE,
            status=ArmStatus.COMPLETED,
        ),
        ArmDiagnostic(
            arm_name="graph",
            method=DiscoveryMethod.GRAPH,
            status=ArmStatus.FAILED,
        ),
    )

    assert diagnostics.degraded is True
    assert diagnostics.failed_arms == ("graph",)


def test_an_empty_arm_is_not_degradation() -> None:
    """Finding nothing is an answer."""
    diagnostics = _diagnostics(
        ArmDiagnostic(
            arm_name="sparse", method=DiscoveryMethod.SPARSE, status=ArmStatus.EMPTY
        )
    )

    assert diagnostics.degraded is False
    assert diagnostics.contributing_arms == ("sparse",)


def test_a_skipped_arm_is_degradation_but_not_a_failure() -> None:
    """
    A dense arm cannot run without a query vector. That is a consequence of
    the query rather than a fault, so it reduces the result without being
    reported as an error.
    """
    diagnostics = _diagnostics(
        ArmDiagnostic(
            arm_name="dense", method=DiscoveryMethod.DENSE, status=ArmStatus.SKIPPED
        )
    )

    assert diagnostics.degraded is True
    assert list(diagnostics.iter_failures()) == []


def test_reproducibility_requires_exactness_and_completeness() -> None:
    """
    Approximate indexes are not reproducible across rebuilds, and a degraded
    result depends on which components happened to be reachable.
    """
    complete = ArmDiagnostic(
        arm_name="dense", method=DiscoveryMethod.DENSE, status=ArmStatus.COMPLETED
    )

    assert _diagnostics(complete, exact_search=True).reproducible is True
    assert _diagnostics(complete, exact_search=False).reproducible is False
    assert (
        _diagnostics(
            complete,
            ArmDiagnostic(
                arm_name="graph",
                method=DiscoveryMethod.GRAPH,
                status=ArmStatus.FAILED,
            ),
            exact_search=True,
        ).reproducible
        is False
    )


def test_a_diagnostic_is_built_from_a_failure() -> None:
    try:
        raise ArmUnavailableError("connection refused", arm_name="dense")
    except ArmUnavailableError as error:
        diagnostic = ArmDiagnostic.from_exception(
            error,
            arm_name="dense",
            method=DiscoveryMethod.DENSE,
            status=ArmStatus.FAILED,
        )

    assert diagnostic.error_code == "ARM_UNAVAILABLE"
    assert diagnostic.traceback_digest is not None
    assert diagnostic.contributed is False


def test_unavailable_credibility_is_recorded_not_defaulted() -> None:
    """
    Retrieval reliability carries a statistical guarantee, so an invented
    value would be indistinguishable from a measured one.
    """
    diagnostics = _diagnostics(credibility_unavailable=("file:///a.pdf",))

    assert diagnostics.credibility_unavailable == ("file:///a.pdf",)


def test_incomplete_embedding_coverage_is_visible() -> None:
    """An unembedded node is invisible to dense search however well it
    matches."""
    diagnostics = _diagnostics(embedding_coverage=0.4)

    assert diagnostics.embedding_coverage == pytest.approx(0.4)


def test_expansion_is_counted_separately_from_fusion() -> None:
    """Expanded context is never corroboration, so it is never folded into the
    fused count."""
    diagnostics = _diagnostics(fused_count=10, expanded_count=6)

    assert diagnostics.fused_count == 10
    assert diagnostics.expanded_count == 6


# ===========================================================================
# Properties
# ===========================================================================


@settings(max_examples=200)
@given(
    fingerprint=st.text(min_size=1, max_size=20),
    version=st.text(min_size=1, max_size=8),
)
def test_a_partition_key_is_always_deterministic(
    fingerprint: str, version: str
) -> None:
    first = SegmentationPartition(
        segmentation_fingerprint=fingerprint, source_schema_version=version
    )
    second = SegmentationPartition(
        segmentation_fingerprint=fingerprint, source_schema_version=version
    )

    assert first.key == second.key


@settings(max_examples=200)
@given(
    left=st.tuples(st.text(min_size=1, max_size=12), st.text(min_size=1, max_size=6)),
    right=st.tuples(st.text(min_size=1, max_size=12), st.text(min_size=1, max_size=6)),
)
def test_differing_components_always_yield_differing_keys(
    left: tuple[str, str], right: tuple[str, str]
) -> None:
    """Property: two partitions share a key only when they share components."""
    first = SegmentationPartition(
        segmentation_fingerprint=left[0], source_schema_version=left[1]
    )
    second = SegmentationPartition(
        segmentation_fingerprint=right[0], source_schema_version=right[1]
    )

    assert (first.key == second.key) == (left == right)


@settings(max_examples=100)
@given(count=st.integers(min_value=1, max_value=20))
def test_any_well_formed_arm_result_validates(count: int) -> None:
    result = ArmResult(
        arm_name="dense",
        method=DiscoveryMethod.DENSE,
        candidates=tuple(_candidate(i) for i in range(count)),
    )

    assert len(result) == count


# ===========================================================================
# Anti-pattern guards
# ===========================================================================


def test_models_do_not_import_the_contract_layer() -> None:
    """
    Interfaces imports these models under TYPE_CHECKING; the dependency runs
    in one direction only.
    """
    source = inspect.getsource(models_module)

    assert "from ptfm.core.retrieval.interfaces" not in source


def test_models_import_no_backend() -> None:
    source = inspect.getsource(models_module)

    for forbidden in (
        "import psycopg",
        "import asyncpg",
        "import qdrant",
        "import numpy",
        "from ptfm.plugins",
    ):
        assert forbidden not in source


def test_no_model_exposes_a_confidence_field() -> None:
    forbidden = {"confidence", "probability", "relevance", "truth"}

    for name in models_module.__all__:
        candidate = getattr(models_module, name)
        fields = getattr(candidate, "model_fields", None)
        if fields is not None:
            assert not forbidden & set(fields), name


def test_the_module_declares_no_mutable_state() -> None:
    mutable = [
        name
        for name, value in vars(models_module).items()
        if not name.startswith("_") and isinstance(value, (list, dict, set))
    ]

    assert mutable == []


def test_the_module_names_no_storage_technology() -> None:
    source = inspect.getsource(models_module).lower()

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
    ):
        assert term not in source


def test_the_module_contains_no_domain_vocabulary() -> None:
    source = inspect.getsource(models_module).lower()

    for term in ("jurisdiction", "statute", "ticker", " icd", "patient", "court"):
        assert term not in source


def test_the_module_declares_no_calibration_concept() -> None:
    source = _code_without_docstrings(models_module).lower()

    for term in ("calibrat", "entail", "truth space", "verification result"):
        assert term not in source


def test_the_digest_algorithm_is_not_injectable() -> None:
    """
    A partition key must be reproducible across every deployment
    indefinitely. A configurable digest would let two deployments treat
    compatible data as incompatible.
    """
    source = inspect.getsource(models_module)

    assert "hashlib.blake2b" in source
    assert "HashProvider" not in source
