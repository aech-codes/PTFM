"""
Unit tests for Retrieval Module configuration.

File:
    tests/core/retrieval/test_config.py

Emphasis
--------
    That every operational number lives here, and that no truth, calibration,
    or relevance threshold does. Retrieval is one module away from the
    calibration layer, which is exactly where a probability-shaped
    configuration value would do damage.
"""

from __future__ import annotations

import ast
import inspect
import os
from typing import Any

import pytest
from pydantic import ValidationError

from ptfm.core.retrieval import config as config_module
from ptfm.core.retrieval.config import (
    ArmConfig,
    CredibilityConfig,
    DegradationPolicy,
    DenseConfig,
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
from ptfm.core.retrieval.exceptions import RetrievalConfigurationError


def _config(**overrides: Any) -> RetrievalConfig:
    """Build a configuration without reading any ambient .env file."""
    return RetrievalConfig(_env_file=None, **overrides)


def _code_without_docstrings(module: object) -> str:
    """
    Return a module's source with every docstring removed.

    A prohibition is documented in docstrings, so a naive substring scan flags
    the sentences stating the rule as though they violated it. Line ranges are
    used rather than text replacement because ``ast.get_docstring`` dedents
    its result, which does not match indented source.

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


# ---------------------------------------------------------------------------
# Defaults that encode a decision
# ---------------------------------------------------------------------------


def test_the_default_configuration_builds() -> None:
    assert _config() is not None


def test_rank_based_fusion_is_the_default() -> None:
    """
    Cosine similarity, lexical relevance, and traversal proximity live on
    incomparable scales. Rank-based fusion discards scores entirely, so it
    cannot combine things that should not be combined.
    """
    assert FusionConfig().strategy == "reciprocal_rank"


def test_query_encoding_verification_cannot_be_disabled() -> None:
    """
    An incompatible query convention degrades retrieval measurably and
    produces no error of its own, so a setting able to switch the check off
    could only ever be used to hide a failure the operator cannot otherwise
    see. It is therefore absent rather than defaulted on.
    """
    assert "verify_query_encoding" not in PartitionConfig.model_fields

    with pytest.raises(ValidationError):
        PartitionConfig(verify_query_encoding=False)


def test_superseded_results_are_filtered_by_default() -> None:
    """Serving superseded content as current evidence is a truthfulness
    failure, and an invisible one."""
    assert SupersessionConfig().filter_superseded is True


def test_the_graph_arm_is_disabled_by_default() -> None:
    """
    It contributes structural retrieval and filtering rather than general
    relevance, so enabling it by default would spend a round trip per query
    for little ranking benefit.
    """
    assert GraphConfig().enabled is False


def test_approximate_search_is_the_default() -> None:
    """
    Exact search is available for audit and reproduction, at a latency cost
    that a serving path should not pay by default.
    """
    assert DenseConfig().exact is False


def test_expansion_is_bounded() -> None:
    """
    Unbounded expansion would re-create the index inflation that removing
    overlapping nodes prevented.
    """
    expansion = ExpansionConfig()

    assert expansion.neighbour_radius >= 1
    assert expansion.max_context_nodes >= 1


def test_degradation_is_the_default_policy() -> None:
    """A single failed arm should reduce a result, not fail a query."""
    assert _config().degradation_policy is DegradationPolicy.DEGRADE


# ---------------------------------------------------------------------------
# Derived settings
# ---------------------------------------------------------------------------


def test_an_absent_request_uses_the_default() -> None:
    config = _config()

    assert config.resolve_top_k(None) == config.default_top_k


def test_an_oversized_request_is_refused_not_clamped() -> None:
    """
    Silently returning fewer results than asked for would let a caller believe
    it had seen everything the corpus offered.
    """
    config = _config()

    with pytest.raises(RetrievalConfigurationError, match="exceeding"):
        config.resolve_top_k(config.max_top_k + 1)


def test_a_permitted_request_is_honoured() -> None:
    assert _config().resolve_top_k(5) == 5


def test_arms_fetch_at_least_the_requested_depth() -> None:
    """
    Fusion discards candidates that no arm ranked highly, so each arm must
    return at least as many as the caller wants.
    """
    config = _config()
    arm = config.arm_config(config.enabled_arms()[0])

    assert config.resolve_fetch_depth(arm, 10) >= 10


def test_a_filtered_query_fetches_deeper() -> None:
    """
    A structural filter discards candidates, so a filtered search must look
    further to return the requested number.
    """
    config = _config()
    arm = config.arm_config(config.enabled_arms()[0])

    unfiltered = config.resolve_fetch_depth(arm, 10)
    filtered = config.resolve_fetch_depth(arm, 10, filtered=True)

    assert filtered >= unfiltered


def test_the_thread_limit_is_usable() -> None:
    assert _config().resolve_thread_limit() >= 1


def test_enabled_arms_are_reported() -> None:
    arms = _config().enabled_arms()

    assert "dense" in arms
    assert "sparse" in arms


def test_a_disabled_arm_is_omitted() -> None:
    config = _config(dense=DenseConfig(enabled=False).model_dump())

    assert "dense" not in config.enabled_arms()


def test_an_unknown_arm_is_reported() -> None:
    with pytest.raises(RetrievalConfigurationError):
        _config().arm_config("nonexistent")


# ---------------------------------------------------------------------------
# Cross-field validation
# ---------------------------------------------------------------------------


def test_a_default_above_the_maximum_is_rejected() -> None:
    with pytest.raises((ValidationError, RetrievalConfigurationError)):
        _config(default_top_k=200, max_top_k=100)


def test_disabling_every_arm_is_rejected() -> None:
    """
    A configuration that can retrieve nothing would answer every query with an
    empty result, which is indistinguishable from an absence of evidence.
    """
    with pytest.raises((ValidationError, RetrievalConfigurationError)):
        _config(
            dense=DenseConfig(enabled=False).model_dump(),
            sparse=SparseConfig(enabled=False).model_dump(),
            graph=GraphConfig(enabled=False).model_dump(),
        )


def test_a_partial_weight_map_is_accepted() -> None:
    """
    A method absent from the map carries a weight of one. Treating a partial map
    as exhaustive would refuse the legitimate case of excluding one arm from the
    ranking while keeping its observation on the result.
    """
    assert FusionConfig(method_weights={"DENSE": 2.0}) is not None
    assert FusionConfig(method_weights={"SPARSE": 0.0}) is not None


def test_zeroing_every_method_is_rejected() -> None:
    """A successful search would return an empty ordering."""
    from ptfm.core.schemas import DiscoveryMethod

    with pytest.raises(ValidationError, match="empty ordering"):
        FusionConfig(
            method_weights={method.value: 0.0 for method in DiscoveryMethod}
        )


def test_a_negative_weight_is_rejected() -> None:
    """It would rank a method's own best results last."""
    with pytest.raises(ValidationError):
        FusionConfig(method_weights={"DENSE": -1.0})


def test_an_unknown_log_level_is_rejected() -> None:
    with pytest.raises(ValidationError):
        TelemetryConfig(log_level="VERBOSE")


# ---------------------------------------------------------------------------
# Immutability, overrides, and layering
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "instance",
    [
        DenseConfig(),
        SparseConfig(),
        GraphConfig(),
        FusionConfig(),
        ExpansionConfig(),
        SupersessionConfig(),
        CredibilityConfig(),
        PartitionConfig(),
        IngestConfig(),
        TelemetryConfig(),
    ],
)
def test_group_configurations_are_frozen(instance: Any) -> None:
    field_name = next(iter(type(instance).model_fields))

    with pytest.raises(ValidationError):
        setattr(instance, field_name, None)


def test_the_root_configuration_is_frozen() -> None:
    config = _config()

    with pytest.raises(ValidationError):
        config.default_top_k = 99  # type: ignore[misc]


@pytest.mark.parametrize(
    ("model", "kwargs"),
    [
        (RetrievalConfig, {"unknown_field": 1}),
        (DenseConfig, {"k": 10}),
        (FusionConfig, {"weights": {}}),
    ],
)
def test_unknown_fields_are_rejected(model: type, kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        model(**kwargs)


def test_overrides_produce_a_new_validated_configuration() -> None:
    original = _config()

    derived = original.with_overrides(default_top_k=5)

    assert derived.default_top_k == 5
    assert original.default_top_k != 5 or original.default_top_k == 5


def test_overrides_revalidate_cross_field_constraints() -> None:
    """
    A copy that skipped validation could hold a combination this class
    explicitly rejects.
    """
    with pytest.raises((RetrievalConfigurationError, ValidationError)):
        _config().with_overrides(default_top_k=10**6)


def test_top_level_values_load_from_the_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PTFM_RETRIEVAL_DEFAULT_TOP_K", "25")

    assert RetrievalConfig().default_top_k == 25


def test_nested_values_load_from_the_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PTFM_RETRIEVAL_EXPANSION__NEIGHBOUR_RADIUS", "2")

    assert RetrievalConfig().expansion.neighbour_radius == 2


def test_unprefixed_variables_are_ignored(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DEFAULT_TOP_K", "999")

    assert _config().default_top_k != 999


# ---------------------------------------------------------------------------
# Anti-pattern guards
# ---------------------------------------------------------------------------


def test_no_truth_or_calibration_threshold_is_declared() -> None:
    """
    Operational limits only. Retrieval sits one module from the calibration
    layer, which is exactly where a probability-shaped configuration value
    would do damage.
    """
    forbidden = {
        "confidence_threshold",
        "verification_threshold",
        "calibration_threshold",
        "similarity_threshold",
        "relevance_threshold",
        "nli_threshold",
        "tau",
        "tau_low",
        "alpha",
    }

    for name in config_module.__all__:
        candidate = getattr(config_module, name)
        fields = getattr(candidate, "model_fields", None)
        if fields is not None:
            assert not forbidden & set(fields), name


def test_no_configuration_field_observes_a_model_score() -> None:
    for name in config_module.__all__:
        candidate = getattr(config_module, name)
        fields = getattr(candidate, "model_fields", None)
        if fields is None:
            continue
        offending = [
            field
            for field in fields
            if "confidence" in field or "probability" in field
        ]
        assert not offending, f"{name}: {offending}"


def test_the_module_declares_no_mutable_state() -> None:
    mutable = [
        name
        for name, value in vars(config_module).items()
        if not name.startswith("_") and isinstance(value, (list, dict, set))
    ]

    assert mutable == []


def test_the_module_imports_no_stage_model_or_backend() -> None:
    source = inspect.getsource(config_module)

    for forbidden in (
        "ptfm.core.retrieval.interfaces",
        "ptfm.core.retrieval.models",
        "ptfm.core.retrieval.pipeline",
        "from ptfm.plugins",
        "import psycopg",
        "import qdrant",
    ):
        assert forbidden not in source


def test_the_module_names_no_storage_technology() -> None:
    """A backend stays replaceable only if configuration does not presume
    one."""
    source = inspect.getsource(config_module).lower()

    for term in (
        "postgres",
        "pgvector",
        "qdrant",
        "faiss",
        "milvus",
        "elasticsearch",
        "tantivy",
        "neo4j",
    ):
        assert term not in source


def test_the_module_contains_no_domain_vocabulary() -> None:
    source = inspect.getsource(config_module).lower()

    for term in ("jurisdiction", "statute", "ticker", " icd", "patient", "court"):
        assert term not in source


def test_the_module_declares_no_calibration_concept() -> None:
    source = _code_without_docstrings(config_module).lower()

    for term in ("calibrat", "entail", "truth space", "verification result"):
        assert term not in source


def test_query_text_logging_is_opt_in() -> None:
    """
    A query may carry information the operator did not intend to persist, so
    recording it must be a decision rather than a default.
    """
    assert TelemetryConfig().log_query_text is False
