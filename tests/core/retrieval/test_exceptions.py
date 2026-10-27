"""
Unit tests for the Retrieval Module exception hierarchy.

File:
    tests/core/retrieval/test_exceptions.py

Emphasis
--------
    The degradable/fatal boundary. Retrieval is the one PTFM module expected
    to answer usefully while a component is unavailable, so where that line
    falls determines whether an outage is presented as an absence of evidence.
"""

from __future__ import annotations

import inspect
import pickle

import pytest

from ptfm.core.retrieval import exceptions as retrieval_exceptions
from ptfm.core.retrieval.exceptions import (
    AllArmsFailedError,
    AmbiguousPartitionError,
    AnchorViolationError,
    ArmTimeoutError,
    ArmUnavailableError,
    BackendUnavailableError,
    CorroborationIntegrityError,
    CredibilityError,
    CredibilityUnavailableError,
    EmbeddingSpaceMismatchError,
    EmptyQueryError,
    EvidenceAssemblyError,
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
    QueryEncodingMismatchError,
    QueryTooLargeError,
    RetrievalConfigurationError,
    RetrievalStage,
    SearchError,
    SegmentationMismatchError,
    StoreRegistrationError,
    StoreUnavailableError,
    StrategyRegistrationError,
    SupersessionError,
)
from ptfm.utils.exceptions import PTFMError

#: The only exceptions that permit a query to proceed with a reduced result.
#: Everything else fails the query, because proceeding would misrepresent the
#: result as complete or would compare things that are not comparable.
DEGRADABLE_TYPES: frozenset[type[PTFMRetrievalError]] = frozenset(
    {SearchError, QueryEncodingError, ExpansionError, CredibilityError}
)

#: Exceptions that override a degradable parent because the condition is a
#: property of the request or would corrupt corroboration.
NON_DEGRADABLE_OVERRIDES: frozenset[type[PTFMRetrievalError]] = frozenset(
    {QueryTooLargeError, EmptyQueryError, AllArmsFailedError, AnchorViolationError}
)


def _all_error_types() -> list[type[PTFMRetrievalError]]:
    return [
        obj
        for name in retrieval_exceptions.__all__
        if isinstance(obj := getattr(retrieval_exceptions, name), type)
        and issubclass(obj, PTFMRetrievalError)
    ]


def _leaf_types() -> list[type[PTFMRetrievalError]]:
    every = _all_error_types()
    return [
        candidate
        for candidate in every
        if not any(
            other is not candidate and issubclass(other, candidate)
            for other in every
        )
    ]


# ---------------------------------------------------------------------------
# Hierarchy invariants
# ---------------------------------------------------------------------------


def test_every_error_derives_from_the_project_root() -> None:
    for error_type in _all_error_types():
        assert issubclass(error_type, PTFMError)


def test_every_leaf_declares_a_concrete_stage() -> None:
    for error_type in _leaf_types():
        assert error_type.stage is not RetrievalStage.UNSPECIFIED, (
            error_type.__name__
        )


def test_every_pipeline_stage_has_at_least_one_exception() -> None:
    """A stage with no exception type could not attribute a failure."""
    covered = {t.stage for t in _all_error_types()}
    expected = set(RetrievalStage) - {RetrievalStage.UNSPECIFIED}

    assert expected <= covered, sorted(s.value for s in expected - covered)


def test_error_codes_are_unique() -> None:
    codes = [t.error_code for t in _all_error_types()]

    assert len(codes) == len(set(codes))


def test_all_exports_resolve() -> None:
    for name in retrieval_exceptions.__all__:
        assert hasattr(retrieval_exceptions, name), name


# ---------------------------------------------------------------------------
# The degradable boundary
# ---------------------------------------------------------------------------


def test_only_the_intended_families_are_degradable() -> None:
    for error_type in _all_error_types():
        if error_type in NON_DEGRADABLE_OVERRIDES:
            continue
        expected = any(
            issubclass(error_type, base) for base in DEGRADABLE_TYPES
        )
        assert error_type.degradable is expected, error_type.__name__


@pytest.mark.parametrize("error_type", sorted(NON_DEGRADABLE_OVERRIDES, key=str))
def test_overrides_are_not_degradable(
    error_type: type[PTFMRetrievalError],
) -> None:
    assert error_type.degradable is False


def test_a_single_arm_failure_degrades() -> None:
    """The arms are independent, so one failing leaves the others able to
    serve."""
    error = ArmUnavailableError("connection refused", arm_name="dense")

    assert error.degradable is True
    assert error.stage is RetrievalStage.SEARCH


def test_total_arm_failure_does_not_degrade() -> None:
    """
    The most consequential distinction in the module. An empty result would be
    indistinguishable from an absence of evidence, and a verification engine
    told there is no evidence refuses to answer.
    """
    error = AllArmsFailedError(
        failures={"dense": "timeout", "sparse": "refused", "graph": "refused"}
    )

    assert error.degradable is False
    assert "absence" in error.message
    assert error.context["failed_arms"] == ["dense", "sparse", "graph"]


def test_total_arm_failure_names_every_arm_and_reason() -> None:
    """A diagnosis should not require correlating separate log entries."""
    error = AllArmsFailedError(failures={"dense": "timeout", "sparse": "refused"})

    assert "dense: timeout" in error.message
    assert "sparse: refused" in error.message


def test_hydration_never_degrades() -> None:
    """
    Arms return identifiers, so without the node store there is no text, span,
    or provenance: candidates, not evidence.
    """
    for error_type in _all_error_types():
        if issubclass(error_type, HydrationError):
            assert error_type.degradable is False, error_type.__name__


def test_partition_failures_never_degrade() -> None:
    """Proceeding would compare things that are not comparable."""
    for error_type in _all_error_types():
        if issubclass(error_type, PartitionError):
            assert error_type.degradable is False, error_type.__name__


def test_degradable_is_distinct_from_retryable() -> None:
    """
    A retryable failure may succeed if attempted again; a degradable one need
    not be attempted again at all.
    """
    error = ArmTimeoutError(arm_name="dense", timeout_seconds=2.0)

    assert error.degradable is True
    assert error.retryable is True

    fatal = AllArmsFailedError(failures={"dense": "down"})
    assert fatal.degradable is False
    assert fatal.retryable is True


# ---------------------------------------------------------------------------
# Corroboration integrity
# ---------------------------------------------------------------------------


def test_an_anchor_violation_is_fatal_despite_a_degradable_parent() -> None:
    """
    An expanded node admitted as an anchor makes one finding appear as two,
    reintroducing by another route the harm that removing overlapping chunks
    prevented.
    """
    error = AnchorViolationError(
        "expanded node would be anchored to itself",
        node_id="node-2",
        anchor_node_id="node-1",
    )

    assert issubclass(AnchorViolationError, ExpansionError)
    assert ExpansionError.degradable is True
    assert error.degradable is False


def test_the_anchor_violation_explains_the_consequence() -> None:
    doc = AnchorViolationError.__doc__ or ""

    assert "corroboration" in doc.lower()
    assert "overconfidence" in doc.lower()


def test_corroboration_integrity_names_the_broken_invariant() -> None:
    error = CorroborationIntegrityError(
        "a node appears twice in one evidence set",
        violated_invariant="unique node_id within an evidence set",
    )

    assert error.violated_invariant == "unique node_id within an evidence set"
    assert error.stage is RetrievalStage.ASSEMBLY
    assert error.degradable is False


# ---------------------------------------------------------------------------
# Credibility is never invented
# ---------------------------------------------------------------------------


def test_credibility_unavailability_is_recorded_not_defaulted() -> None:
    """
    Retrieval reliability carries a statistical guarantee, so substituting a
    default would insert an invented number where it is indistinguishable from
    a measured one.
    """
    error = CredibilityUnavailableError(
        "plugin timed out",
        source_uri="file:///corpus/a.pdf",
        plugin_name="FinancePlugin",
    )

    assert error.source_uri == "file:///corpus/a.pdf"
    assert error.context["plugin_name"] == "FinancePlugin"
    assert error.stage is RetrievalStage.CREDIBILITY


def test_credibility_concerns_a_source_not_a_claim() -> None:
    """Nothing in this module judges a claim."""
    doc = CredibilityError.__doc__ or ""

    assert "SOURCE" in doc
    assert "not a judgement about a claim" in doc


# ---------------------------------------------------------------------------
# Compatibility
# ---------------------------------------------------------------------------


def test_a_segmentation_mismatch_explains_disjoint_identifiers() -> None:
    error = SegmentationMismatchError(expected="1:abc", actual="1:def")

    assert error.expected == "1:abc"
    assert "disjoint" in error.message


def test_an_embedding_space_mismatch_is_reported() -> None:
    error = EmbeddingSpaceMismatchError(
        "vectors from two models would be compared",
        expected_identity="bge@r1",
        actual_identity="e5@r2",
    )

    assert error.context["expected_identity"] == "bge@r1"
    assert error.degradable is False


def test_a_query_encoding_mismatch_explains_the_invisibility() -> None:
    """The condition is detected here because it would otherwise surface only
    as poor results."""
    error = QueryEncodingMismatchError(expected="prefix:query: ", actual="")

    assert "without producing any visible error" in error.message


def test_several_segmentation_partitions_are_ambiguous() -> None:
    error = AmbiguousPartitionError(candidates=["1:abc", "1:def"])

    assert error.candidates == ("1:abc", "1:def")
    assert "Exactly one" in error.message


def test_a_missing_partition_names_the_key() -> None:
    error = PartitionNotFoundError("not indexed", partition_key="1:abc/bge")

    assert error.partition_key == "1:abc/bge"


# ---------------------------------------------------------------------------
# Request-shaped failures
# ---------------------------------------------------------------------------


def test_an_oversized_query_is_a_property_of_the_request() -> None:
    """
    Degrading would silently answer a different question from the one asked.
    """
    error = QueryTooLargeError(query_chars=12_000, max_chars=8_192)

    assert error.degradable is False
    assert error.query_chars == 12_000


def test_an_empty_query_is_not_degradable() -> None:
    assert EmptyQueryError("query carries no content").degradable is False


# ---------------------------------------------------------------------------
# Fusion
# ---------------------------------------------------------------------------


def test_incomparable_signals_are_reported() -> None:
    error = IncomparableSignalsError(
        "cannot average a similarity with a lexical score",
        scales=["COSINE", "LEXICAL"],
    )

    assert error.scales == ("COSINE", "LEXICAL")
    assert error.stage is RetrievalStage.FUSION


def test_rank_based_fusion_cannot_raise_it() -> None:
    """
    The reason rank-based fusion is the default.

    Asserted against the fusion module's SOURCE rather than against a phrase in
    this exception's docstring. A prose assertion passes or fails on how a
    sentence happens to be worded and wrapped, and says nothing about what the
    code does; this one fails only if a rank-only strategy gains the ability to
    compare scales.
    """
    import ast
    import inspect

    from ptfm.core.retrieval.pipeline import fusion as fusion_module

    source = inspect.getsource(fusion_module)
    raised = {
        ast.unparse(node.exc.func)
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Raise)
        and isinstance(node.exc, ast.Call)
    }

    assert "IncomparableSignalsError" not in raised, (
        "reciprocal rank fusion compares ranks only, so it has no scale to "
        "find incomparable"
    )


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------


def test_a_missing_node_indicates_store_divergence() -> None:
    """Silently dropping the result would present an inconsistent store as a
    less relevant corpus."""
    error = NodeNotFoundError(
        "index returned unknown identifiers", missing_node_ids=["n1", "n2"]
    )

    assert error.missing_node_ids == ("n1", "n2")
    assert error.degradable is False


def test_incomplete_provenance_names_the_missing_fields() -> None:
    error = ProvenanceIncompleteError(
        "node cannot become evidence", missing_fields=["span", "source_uri"]
    )

    assert error.missing_fields == ("span", "source_uri")
    assert error.stage is RetrievalStage.HYDRATION


def test_a_partial_index_write_names_both_sides() -> None:
    error = PartialIndexWriteError(
        "vector store accepted, node store refused",
        written_stores=["vector"],
        failed_stores=["node"],
    )

    assert error.written_stores == ("vector",)
    assert error.failed_stores == ("node",)
    assert error.retryable is True


def test_a_partial_write_is_recoverable_because_identity_is_deterministic() -> None:
    doc = PartialIndexWriteError.__doc__ or ""

    assert "deterministic" in doc


# ---------------------------------------------------------------------------
# Base behaviour
# ---------------------------------------------------------------------------


def test_stage_and_degradability_are_serialized() -> None:
    error = NodeNotFoundError(
        "unknown identifier", query_id="q-1", node_id="n-1"
    )

    payload = error.to_dict()

    assert payload["stage"] == RetrievalStage.HYDRATION.value
    assert payload["degradable"] is False
    assert payload["query_id"] == "q-1"
    assert payload["node_id"] == "n-1"


def test_identity_is_mirrored_into_context_for_uniform_logging() -> None:
    error = ArmUnavailableError("down", arm_name="dense", query_id="q-2")

    assert error.context["query_id"] == "q-2"
    assert error.context["stage"] == RetrievalStage.SEARCH.value
    assert error.context["degradable"] is True


def test_absent_identity_is_omitted_rather_than_recorded_as_none() -> None:
    error = FusionError("no ranking")

    assert "query_id" not in error.context
    assert error.to_dict()["query_id"] is None


def test_a_third_party_exception_is_preserved_as_cause() -> None:
    class _DriverError(Exception):
        """Stand-in for a store client exception."""

    try:
        try:
            raise _DriverError("connection reset")
        except _DriverError as exc:
            raise StoreUnavailableError(
                "vector store unreachable", store_name="vector"
            ) from exc
    except StoreUnavailableError as error:
        assert isinstance(error.__cause__, _DriverError)
        assert error.to_dict()["cause_type"] == "_DriverError"


@pytest.mark.parametrize(
    "error",
    [
        AllArmsFailedError(failures={"dense": "down"}),
        SegmentationMismatchError(expected="a", actual="b"),
        QueryEncodingMismatchError(expected="a", actual="b"),
        AmbiguousPartitionError(candidates=["a", "b"]),
        QueryTooLargeError(query_chars=10, max_chars=5),
        ArmTimeoutError(arm_name="dense", timeout_seconds=1.0),
        BackendUnavailableError(
            component="c", missing_dependency="d", install_extra="e"
        ),
    ],
)
def test_keyword_only_exceptions_survive_process_boundaries(
    error: PTFMRetrievalError,
) -> None:
    """Retrieval may be served from worker processes."""
    restored = pickle.loads(pickle.dumps(error))

    assert type(restored) is type(error)
    assert restored.message == error.message
    assert restored.stage is error.stage
    assert restored.degradable is error.degradable
    assert dict(restored.context) == dict(error.context)


# ---------------------------------------------------------------------------
# Registration and availability
# ---------------------------------------------------------------------------


def test_strategy_registration_names_the_strategy() -> None:
    error = StrategyRegistrationError("duplicate", strategy_name="dense")

    assert error.strategy_name == "dense"
    assert error.stage is RetrievalStage.CONFIGURATION


def test_store_registration_names_the_store() -> None:
    error = StoreRegistrationError("invalid", store_name="vector")

    assert error.store_name == "vector"


def test_an_unavailable_backend_names_the_install_extra() -> None:
    error = BackendUnavailableError(
        component="vector store 'postgres'",
        missing_dependency="psycopg",
        install_extra="ptfm[postgres]",
    )

    assert error.install_extra == "ptfm[postgres]"
    assert "psycopg" in error.message


def test_configuration_failures_never_degrade() -> None:
    for error_type in _all_error_types():
        if issubclass(error_type, RetrievalConfigurationError):
            assert error_type.degradable is False, error_type.__name__


# ---------------------------------------------------------------------------
# Module hygiene
# ---------------------------------------------------------------------------


def test_the_module_imports_only_the_project_root_exception() -> None:
    source = inspect.getsource(retrieval_exceptions)

    assert "from ptfm.utils.exceptions import PTFMError" in source
    for forbidden in (
        "from ptfm.core.schemas",
        "from ptfm.core.knowledge",
        "from ptfm.core.retrieval.models",
        "from ptfm.core.retrieval.config",
        "from ptfm.plugins",
        "import pydantic",
        "import numpy",
        "import psycopg",
    ):
        assert forbidden not in source


def test_the_module_declares_no_mutable_state() -> None:
    mutable = [
        name
        for name, value in vars(retrieval_exceptions).items()
        if not name.startswith("__") and isinstance(value, (list, dict, set))
    ]

    assert mutable == []


def test_the_module_names_no_storage_technology() -> None:
    """Backends stay behind interfaces; the exception vocabulary must not
    presume one."""
    source = inspect.getsource(retrieval_exceptions).lower()

    for term in (
        "postgres",
        "pgvector",
        "qdrant",
        "faiss",
        "milvus",
        "elasticsearch",
        "opensearch",
        "tantivy",
        "neo4j",
    ):
        assert term not in source


def test_the_module_contains_no_domain_vocabulary() -> None:
    source = inspect.getsource(retrieval_exceptions).lower()

    for term in ("jurisdiction", "statute", "ticker", " icd", "patient", "court"):
        assert term not in source


def _code_without_docstrings(module: object) -> str:
    """
    Return a module's source with every docstring removed.

    A prohibition is DOCUMENTED in docstrings, so a naive substring scan flags
    the sentences that state the rule as though they violated it. Removing
    docstrings keeps the guard a simple substring check while making it
    accurate about what it enforces: that the concept appears in no name,
    declaration, or comment in the code itself.
    """
    import ast

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
        # Line numbers, not the cleaned text: ast.get_docstring dedents its
        # result, so it does not match indented source and a text replacement
        # would silently leave the docstring in place.
        blank.update(range(first.lineno - 1, (first.end_lineno or first.lineno)))

    return "".join(
        "" if index in blank else line for index, line in enumerate(lines)
    )


def test_the_module_declares_no_truth_or_calibration_concept() -> None:
    """
    Retrieval produces uncalibrated signals. Confidence, entailment, and
    calibration belong to later modules.

    Scanned excluding the module docstring, which is where this very
    prohibition is stated.
    """
    source = _code_without_docstrings(retrieval_exceptions).lower()

    for term in ("entail", "calibrat", "truth space", "verification result"):
        assert term not in source


def test_the_module_declares_no_thresholds() -> None:
    source = inspect.getsource(retrieval_exceptions)

    assert "> 0.8" not in source
    assert "0.5" not in source
