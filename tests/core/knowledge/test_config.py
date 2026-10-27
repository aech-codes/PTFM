"""
Unit tests for Knowledge Processing configuration.

File:
    tests/core/knowledge/test_config.py

Emphasis
--------
    The segmentation fingerprint. It determines node identity, so its
    inclusion rule is the single most consequential thing in this module: a
    parameter wrongly included invalidates a corpus for no reason, and one
    wrongly omitted lets an index silently fail to resolve against its corpus
    with no error raised anywhere.
"""

from __future__ import annotations

import inspect
import os
from typing import Any

import pytest
from pydantic import ValidationError

from ptfm.core.knowledge import config as config_module
from ptfm.core.knowledge.config import (
    SEGMENTATION_FINGERPRINT_VERSION,
    ChunkingConfig,
    EmbeddingConfig,
    EnrichmentConfig,
    EnrichmentFailurePolicy,
    KnowledgeConfig,
    RelationConfig,
    SinkConfig,
    TelemetryConfig,
    ValidationConfig,
    compute_segmentation_fingerprint,
)
from ptfm.core.knowledge.exceptions import KnowledgeConfigurationError


def _config(**overrides: Any) -> KnowledgeConfig:
    """Build a configuration without reading any ambient .env file."""
    return KnowledgeConfig(_env_file=None, **overrides)


# ---------------------------------------------------------------------------
# The inclusion rule
# ---------------------------------------------------------------------------


def test_only_chunking_contributes_to_identity() -> None:
    """
    The rule the fingerprint exists to enforce. Every configured component
    must originate in the chunking group and nowhere else.
    """
    components = _config().boundary_components()

    assert components
    assert all(key.startswith("chunking.") for key in components)


@pytest.mark.parametrize(
    ("group", "overrides"),
    [
        ("embedding", {"batch_size": 128, "provider": "other"}),
        ("enrichment", {"timeout_seconds": 99.0}),
        ("relations", {"derive_sequence": False}),
        ("validation", {"max_nodes_per_document": 5}),
        ("sink", {"flush_interval_units": 7}),
        ("telemetry", {"log_level": "DEBUG"}),
    ],
)
def test_non_boundary_groups_never_change_identity(
    group: str, overrides: dict[str, Any]
) -> None:
    """
    Adding an enricher, changing a batch size, or enabling a log level must
    never invalidate a corpus. Each alters no node's text or span.
    """
    baseline = _config()
    group_type = type(getattr(baseline, group))

    altered = baseline.with_overrides(
        **{group: group_type(**overrides).model_dump()}
    )

    assert altered.segmentation_fingerprint() == baseline.segmentation_fingerprint()


def test_concurrency_never_changes_identity() -> None:
    baseline = _config()

    altered = _config(max_concurrent_documents=64, fail_fast=True)

    assert altered.segmentation_fingerprint() == baseline.segmentation_fingerprint()


@pytest.mark.parametrize(
    "override",
    [
        {"strategy": "fixed_window"},
        {"sentence_segmenter": "pysbd"},
        {"target_chunk_chars": 1_500},
        {"max_chunk_chars": 3_000},
        {"min_chunk_chars": 50},
        {"split_oversized_blocks": False},
        {"preserve_table_rows": False},
        {"respect_heading_boundaries": False},
    ],
)
def test_every_chunking_parameter_changes_identity(
    override: dict[str, Any]
) -> None:
    """
    Each of these can move a boundary, so each must produce a disjoint
    identifier space. An omission here is the silent failure the fingerprint
    exists to prevent.
    """
    baseline = _config()

    altered = _config(chunking=ChunkingConfig(**override).model_dump())

    assert altered.segmentation_fingerprint() != baseline.segmentation_fingerprint()


def test_the_chunking_group_holds_only_boundary_parameters() -> None:
    """
    The structural enforcement: the fingerprint is derived from the whole
    model, so a non-boundary field placed here would wrongly invalidate a
    corpus, and a boundary field placed elsewhere would be silently omitted.
    """
    fields = set(ChunkingConfig.model_fields)

    assert fields == {
        "strategy",
        "sentence_segmenter",
        "target_chunk_chars",
        "max_chunk_chars",
        "min_chunk_chars",
        "split_oversized_blocks",
        "preserve_table_rows",
        "respect_heading_boundaries",
    }


def test_components_are_derived_from_the_model_not_a_list() -> None:
    """A maintained list could drift out of step with the fields."""
    components = ChunkingConfig().boundary_components()

    assert len(components) == len(ChunkingConfig.model_fields)


def test_the_inclusion_rule_is_documented_where_it_is_enforced() -> None:
    doc = config_module.__doc__ or ""

    assert "IF AND ONLY IF" in doc
    assert "CORPUS MIGRATION" in doc


# ---------------------------------------------------------------------------
# Fingerprint composition
# ---------------------------------------------------------------------------


def test_the_fingerprint_is_deterministic() -> None:
    config = _config()

    assert config.segmentation_fingerprint() == config.segmentation_fingerprint()


def test_the_fingerprint_is_independent_of_component_order() -> None:
    """Otherwise node identity would depend on wiring order."""
    a = {"strategy.window": "512"}
    b = {"segmenter.version": "pysbd-0.3.4"}

    assert compute_segmentation_fingerprint(a, b) == compute_segmentation_fingerprint(
        b, a
    )


def test_runtime_components_contribute() -> None:
    """
    A segmenter upgrade shifts boundaries within oversized blocks while every
    configured value stays the same. Configuration alone cannot see it.
    """
    config = _config()

    baseline = config.segmentation_fingerprint({"segmenter.version": "1.0"})
    upgraded = config.segmentation_fingerprint({"segmenter.version": "1.1"})

    assert baseline != upgraded


def test_a_conflicting_component_is_rejected() -> None:
    """A conflict would make the fingerprint depend on argument order."""
    with pytest.raises(KnowledgeConfigurationError, match="non-deterministic"):
        compute_segmentation_fingerprint(
            {"strategy.window": "512"}, {"strategy.window": "1024"}
        )


def test_an_identical_duplicate_component_is_permitted() -> None:
    """Two sources agreeing is not a conflict."""
    result = compute_segmentation_fingerprint(
        {"strategy.window": "512"}, {"strategy.window": "512"}
    )

    assert result == compute_segmentation_fingerprint({"strategy.window": "512"})


def test_the_fingerprint_carries_its_scheme_version() -> None:
    """
    A future change to how the fingerprint is composed must produce visibly
    different values rather than colliding with the previous scheme.
    """
    fingerprint = _config().segmentation_fingerprint()

    assert fingerprint.startswith(f"{SEGMENTATION_FINGERPRINT_VERSION}:")


def test_the_fingerprint_of_an_empty_component_set_is_stable() -> None:
    assert compute_segmentation_fingerprint() == compute_segmentation_fingerprint()


@pytest.mark.parametrize(
    ("value", "rendered"),
    [
        (True, "true"),
        (False, "false"),
        (1_000, "1000"),
        ("block_aligned", "block_aligned"),
        (frozenset({"b", "a"}), "a,b"),
    ],
)
def test_values_render_stably_across_processes(
    value: object, rendered: str
) -> None:
    """
    A value supplied through an environment variable must digest identically
    to the same value supplied in code.
    """
    assert config_module._render(value) == rendered


def test_boolean_rendering_is_not_python_specific() -> None:
    """``str(True)`` would render as ``"True"``, which no other language or
    configuration format would produce."""
    assert config_module._render(True) == "true"


def test_the_digest_algorithm_is_not_injectable() -> None:
    """
    A deployment with a different hash provider would compute a different
    fingerprint for identical configuration, producing disjoint identifier
    spaces for no reason.
    """
    source = inspect.getsource(compute_segmentation_fingerprint)

    assert "hashlib.blake2b" in source
    assert "HashProvider" not in source


def test_components_are_exposed_for_diagnosis() -> None:
    """A mismatch must be diagnosable by comparing components, not by
    observing that two opaque digests differ."""
    components = _config().boundary_components()

    assert "chunking.strategy" in components
    assert components["chunking.max_chunk_chars"] == "2000"


# ---------------------------------------------------------------------------
# Chunking validation
# ---------------------------------------------------------------------------


def test_overlap_cannot_be_configured() -> None:
    """
    Overlapping nodes would duplicate text, and duplicated text enters
    verification as two evidence items rather than one, inflating apparent
    corroboration in the quantity the system exists to calibrate. The remedy
    for a claim straddling two nodes is retrieval-time neighbour expansion
    through PRECEDES relations.
    """
    assert "overlap_chars" not in ChunkingConfig.model_fields

    with pytest.raises(ValidationError):
        ChunkingConfig(overlap_chars=50)


def test_the_overlap_decision_is_recorded_where_it_binds() -> None:
    """A future contributor must find the reasoning on the model itself."""
    doc = ChunkingConfig.__doc__ or ""

    assert "Overlap is deliberately absent" in doc
    assert "PRECEDES" in doc


def test_defaults_are_coherent() -> None:
    chunking = ChunkingConfig()

    assert chunking.min_chunk_chars <= chunking.target_chunk_chars
    assert chunking.target_chunk_chars <= chunking.max_chunk_chars


def test_a_minimum_above_the_target_is_rejected() -> None:
    with pytest.raises(ValidationError, match="could satisfy both"):
        ChunkingConfig(min_chunk_chars=2_000, target_chunk_chars=1_000)


def test_a_target_above_the_maximum_is_rejected() -> None:
    with pytest.raises(ValidationError, match="oversized on reaching"):
        ChunkingConfig(target_chunk_chars=3_000, max_chunk_chars=2_000)


# ---------------------------------------------------------------------------
# Cross-group validation
# ---------------------------------------------------------------------------


def test_embedding_input_must_accommodate_the_largest_node() -> None:
    """
    Failing only once a sufficiently large node appears would make a
    configuration error look like a corpus problem.
    """
    with pytest.raises(ValidationError, match="could never be embedded"):
        _config(
            chunking=ChunkingConfig(max_chunk_chars=10_000).model_dump(),
            embedding=EmbeddingConfig(max_input_chars=8_192).model_dump(),
        )


def test_the_capacity_check_is_skipped_when_embedding_is_disabled() -> None:
    config = _config(
        chunking=ChunkingConfig(max_chunk_chars=10_000).model_dump(),
        embedding=EmbeddingConfig(enabled=False, max_input_chars=100).model_dump(),
    )

    assert config.embedding.enabled is False


# ---------------------------------------------------------------------------
# Defaults that encode a decision
# ---------------------------------------------------------------------------


def test_provenance_verification_is_enabled_by_default() -> None:
    """Disabling it makes the Evidence-First guarantee aspirational."""
    assert ValidationConfig().verify_provenance is True


def test_full_coverage_is_required_by_default() -> None:
    """
    Text absent from every node cannot be retrieved, so a claim it would have
    supported becomes unverifiable, invisibly.
    """
    assert ValidationConfig().require_full_coverage is True


def test_inverse_relations_are_not_emitted_by_default() -> None:
    """Storing both directions doubles the count for no information."""
    assert RelationConfig().emit_inverse_relations is False


def test_enrichment_degrades_by_default() -> None:
    """Enrichment produces metadata only, so a failure loses metadata rather
    than a node."""
    assert EnrichmentConfig().failure_policy is EnrichmentFailurePolicy.DEGRADE


def test_a_sink_failure_aborts_the_run_by_default() -> None:
    """
    Unlike a malformed document, an unavailable sink fails identically for
    every subsequent document, so continuing discards everything computed.
    """
    assert SinkConfig().fail_run_on_sink_error is True


def test_normalization_is_required_by_default() -> None:
    assert EmbeddingConfig().require_normalized is True


def test_the_failure_guard_requires_a_minimum_sample() -> None:
    """Otherwise a run aborts when the first of one document fails."""
    assert _config().failure_ratio_min_samples > 1


# ---------------------------------------------------------------------------
# Derived settings
# ---------------------------------------------------------------------------


def test_an_explicit_thread_limit_is_honoured() -> None:
    assert _config(processing_thread_limit=3).resolve_thread_limit() == 3


def test_the_thread_limit_derives_from_cpu_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Segmentation is CPU-bound with no I/O, so oversubscription only adds
    scheduling overhead."""
    monkeypatch.setattr(os, "cpu_count", lambda: 6)

    assert _config().resolve_thread_limit() == 6


def test_the_thread_limit_falls_back_when_cpu_count_is_unknown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(os, "cpu_count", lambda: None)

    assert _config().resolve_thread_limit() >= 1


def test_a_provider_maximum_bounds_the_batch_size() -> None:
    """A provider knows its own limits; exceeding them fails every batch."""
    config = _config(embedding=EmbeddingConfig(batch_size=64).model_dump())

    assert config.resolve_batch_size(provider_maximum=16) == 16
    assert config.resolve_batch_size(provider_maximum=256) == 64


def test_an_unusable_provider_maximum_is_rejected() -> None:
    with pytest.raises(KnowledgeConfigurationError, match="not usable"):
        _config().resolve_batch_size(provider_maximum=0)


def test_an_empty_enricher_selection_runs_every_enricher() -> None:
    config = _config()

    assert config.enricher_is_enabled("AnyEnricher") is True


def test_a_named_selection_runs_only_those_enrichers() -> None:
    config = _config(
        enrichment=EnrichmentConfig(
            enabled_enrichers=frozenset({"LegalEnricher"})
        ).model_dump()
    )

    assert config.enricher_is_enabled("LegalEnricher") is True
    assert config.enricher_is_enabled("OtherEnricher") is False


def test_disabling_enrichment_disables_every_enricher() -> None:
    config = _config(enrichment=EnrichmentConfig(enabled=False).model_dump())

    assert config.enricher_is_enabled("LegalEnricher") is False


# ---------------------------------------------------------------------------
# Immutability and layering
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "instance",
    [
        ChunkingConfig(),
        EmbeddingConfig(),
        EnrichmentConfig(),
        RelationConfig(),
        ValidationConfig(),
        SinkConfig(),
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
        config.max_concurrent_documents = 32  # type: ignore[misc]


@pytest.mark.parametrize(
    ("model", "kwargs"),
    [
        (KnowledgeConfig, {"unknown_field": 1}),
        (ChunkingConfig, {"chunk_size": 100}),
        (EmbeddingConfig, {"model": "x"}),
    ],
)
def test_unknown_fields_are_rejected(model: type, kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        model(**kwargs)


def test_an_unknown_log_level_is_rejected() -> None:
    with pytest.raises(ValidationError, match="log_level"):
        TelemetryConfig(log_level="VERBOSE")


# ---------------------------------------------------------------------------
# Environment layering
# ---------------------------------------------------------------------------


def test_top_level_values_load_from_the_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PTFM_KNOWLEDGE_MAX_CONCURRENT_DOCUMENTS", "32")

    assert _config().max_concurrent_documents == 32


def test_nested_values_load_from_the_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PTFM_KNOWLEDGE_CHUNKING__MAX_CHUNK_CHARS", "1500")
    monkeypatch.setenv("PTFM_KNOWLEDGE_EMBEDDING__BATCH_SIZE", "64")

    config = _config()

    assert config.chunking.max_chunk_chars == 1500
    assert config.embedding.batch_size == 64


def test_an_environment_supplied_boundary_value_changes_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Identity must not depend on how a value reached the configuration."""
    baseline = _config().segmentation_fingerprint()
    monkeypatch.setenv("PTFM_KNOWLEDGE_CHUNKING__MAX_CHUNK_CHARS", "1500")

    assert _config().segmentation_fingerprint() != baseline


def test_an_environment_supplied_value_digests_as_a_code_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from_code = _config(
        chunking=ChunkingConfig(max_chunk_chars=1500).model_dump()
    ).segmentation_fingerprint()

    monkeypatch.setenv("PTFM_KNOWLEDGE_CHUNKING__MAX_CHUNK_CHARS", "1500")

    assert _config().segmentation_fingerprint() == from_code


def test_unprefixed_variables_are_ignored(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MAX_CONCURRENT_DOCUMENTS", "999")

    assert _config().max_concurrent_documents == 8


# ---------------------------------------------------------------------------
# Overrides
# ---------------------------------------------------------------------------


def test_overrides_produce_a_new_validated_configuration() -> None:
    original = _config()

    derived = original.with_overrides(max_concurrent_documents=16)

    assert derived.max_concurrent_documents == 16
    assert original.max_concurrent_documents == 8


def test_overrides_revalidate_cross_field_constraints() -> None:
    with pytest.raises(KnowledgeConfigurationError):
        _config().with_overrides(
            chunking=ChunkingConfig(max_chunk_chars=99_999).model_dump()
        )


def test_an_invalid_override_raises_a_ptfm_exception() -> None:
    with pytest.raises(KnowledgeConfigurationError):
        _config().with_overrides(max_concurrent_documents=0)


def test_the_configuration_round_trips_through_json() -> None:
    original = _config(max_concurrent_documents=4)

    restored = KnowledgeConfig.model_validate_json(original.model_dump_json())

    assert restored == original
    assert restored.segmentation_fingerprint() == original.segmentation_fingerprint()


# ---------------------------------------------------------------------------
# Anti-pattern guards
# ---------------------------------------------------------------------------


def test_no_verification_threshold_is_declared() -> None:
    """
    Operational limits only. A verification or calibration threshold here
    would place a hardcoded truth judgement in the knowledge path.
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

    for group in (
        KnowledgeConfig,
        ChunkingConfig,
        EmbeddingConfig,
        EnrichmentConfig,
        RelationConfig,
        ValidationConfig,
        SinkConfig,
        TelemetryConfig,
    ):
        assert not forbidden & set(group.model_fields), group.__name__


def test_no_configuration_field_observes_a_model_score() -> None:
    """
    Unlike ingestion, which tolerated one language-detection confidence, no
    score reaches configuration here at all.
    """
    for group in (
        KnowledgeConfig,
        ChunkingConfig,
        EmbeddingConfig,
        EnrichmentConfig,
        RelationConfig,
        ValidationConfig,
        SinkConfig,
        TelemetryConfig,
    ):
        assert not [
            name for name in group.model_fields if "confidence" in name
        ], group.__name__


def test_the_module_declares_no_mutable_state() -> None:
    mutable = [
        name
        for name, value in vars(config_module).items()
        if not name.startswith("_") and isinstance(value, (list, dict, set))
    ]

    assert mutable == []


def test_the_module_imports_no_stage_or_implementation() -> None:
    source = inspect.getsource(config_module)

    for forbidden in (
        "from ptfm.core.knowledge.interfaces",
        "from ptfm.core.knowledge.models",
        "from ptfm.core.knowledge.pipeline",
        "from ptfm.core.knowledge.chunking",
        "from ptfm.core.knowledge.embedding",
        "from ptfm.plugins",
    ):
        assert forbidden not in source


def test_the_module_contains_no_domain_vocabulary() -> None:
    source = inspect.getsource(config_module).lower()

    for term in ("jurisdiction", "statute", "ticker", "icd", "patient", "court"):
        assert term not in source
