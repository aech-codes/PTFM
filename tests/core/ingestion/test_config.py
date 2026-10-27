"""Unit tests for Data Ingestion Module configuration."""

from __future__ import annotations

import os
from typing import Any

import pytest
from pydantic import ValidationError

from ptfm.core.ingestion.config import (
    KIB,
    MIB,
    AcquisitionConfig,
    DeduplicationConfig,
    DetectionConfig,
    ExponentialBackoffRetryPolicy,
    ExtractionConfig,
    IngestionConfig,
    MetadataConfig,
    RetryConfig,
    StructureConfig,
    TelemetryConfig,
    TextNormalizationForm,
)
from ptfm.core.ingestion.exceptions import IngestionConfigurationError
from ptfm.core.ingestion.interfaces import RetryPolicy


def _config(**overrides: Any) -> IngestionConfig:
    """Build a configuration without reading any ambient .env file."""
    return IngestionConfig(_env_file=None, **overrides)


# ---------------------------------------------------------------------------
# Defaults and immutability
# ---------------------------------------------------------------------------


def test_defaults_construct_a_valid_configuration() -> None:
    config = _config()

    assert config.max_concurrent_documents == 8
    assert config.fail_fast is False
    assert config.acquisition.max_document_bytes == 100 * MIB
    assert config.structure.normalization_form is TextNormalizationForm.NFC
    assert config.retry.max_attempts == 3


@pytest.mark.parametrize(
    "instance",
    [
        RetryConfig(),
        AcquisitionConfig(),
        DetectionConfig(),
        ExtractionConfig(),
        StructureConfig(),
        MetadataConfig(),
        DeduplicationConfig(),
        TelemetryConfig(),
    ],
)
def test_group_configurations_are_frozen(instance: Any) -> None:
    field_name = next(iter(type(instance).model_fields))

    with pytest.raises(ValidationError):
        setattr(instance, field_name, None)


def test_root_configuration_is_frozen() -> None:
    config = _config()

    with pytest.raises(ValidationError):
        config.max_concurrent_documents = 32  # type: ignore[misc]


@pytest.mark.parametrize(
    ("model", "kwargs"),
    [
        (IngestionConfig, {"unknown_field": 1}),
        (RetryConfig, {"backoff": 2}),
        (AcquisitionConfig, {"max_size": 1}),
        (StructureConfig, {"separator": " "}),
    ],
)
def test_unknown_fields_are_rejected(model: type, kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        model(**kwargs)


# ---------------------------------------------------------------------------
# Cross-field validation
# ---------------------------------------------------------------------------


def test_spool_threshold_above_max_size_is_rejected() -> None:
    with pytest.raises(ValidationError, match="could never engage"):
        AcquisitionConfig(max_document_bytes=1 * MIB, spool_threshold_bytes=8 * MIB)


def test_read_chunk_above_spool_threshold_is_rejected() -> None:
    with pytest.raises(ValidationError, match="defeat the memory bound"):
        AcquisitionConfig(spool_threshold_bytes=32 * KIB, read_chunk_bytes=64 * KIB)


def test_detection_header_above_spool_threshold_is_rejected() -> None:
    """Detection must never force a payload into memory."""
    with pytest.raises(ValidationError, match="header_bytes"):
        _config(
            acquisition=AcquisitionConfig(
                spool_threshold_bytes=4 * KIB, read_chunk_bytes=1 * KIB
            ),
            detection=DetectionConfig(header_bytes=8 * KIB),
        )


def test_inverted_backoff_ceiling_is_rejected() -> None:
    with pytest.raises(ValidationError, match="inverts the backoff curve"):
        RetryConfig(base_delay_seconds=10.0, max_delay_seconds=1.0)


def test_unreachable_language_detection_is_rejected() -> None:
    with pytest.raises(ValidationError, match="could never run"):
        MetadataConfig(
            language_detection_sample_chars=16, language_detection_min_chars=64
        )


def test_near_duplicate_distance_must_be_below_fingerprint_width() -> None:
    with pytest.raises(ValidationError, match="every document matches"):
        DeduplicationConfig(simhash_bits=64, near_duplicate_max_distance=64)


# ---------------------------------------------------------------------------
# Span-invariant protection
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("separator", ["", "---", " | ", "\n\r\n"])
def test_invalid_block_separators_are_rejected(separator: str) -> None:
    """A non-whitespace separator would place content outside the block tree."""
    with pytest.raises(ValidationError):
        StructureConfig(block_separator=separator)


@pytest.mark.parametrize("separator", ["\n", "\n\n", " ", "\t"])
def test_whitespace_block_separators_are_accepted(separator: str) -> None:
    assert StructureConfig(block_separator=separator).block_separator == separator


def test_span_integrity_validation_is_enabled_by_default() -> None:
    """The runtime guarantee of evidence traceability must not be off by default."""
    assert StructureConfig().validate_span_integrity is True


def test_unsupported_encoding_is_rejected() -> None:
    with pytest.raises(ValidationError, match="not a recognized codec"):
        DetectionConfig(default_text_encoding="definitely-not-a-codec")


def test_unknown_log_level_is_rejected() -> None:
    with pytest.raises(ValidationError, match="log_level"):
        TelemetryConfig(log_level="VERBOSE")


# ---------------------------------------------------------------------------
# Retry policy
# ---------------------------------------------------------------------------


def test_policy_satisfies_the_retry_protocol() -> None:
    assert isinstance(_config().build_retry_policy(), RetryPolicy)


def test_first_attempt_is_never_delayed() -> None:
    policy = _config().build_retry_policy(rng=lambda: 0.0)

    assert policy.delay_for(1) == 0.0


def test_backoff_grows_geometrically_without_jitter() -> None:
    config = RetryConfig(
        base_delay_seconds=1.0,
        backoff_multiplier=2.0,
        max_delay_seconds=100.0,
        jitter_ratio=0.0,
    )
    policy = ExponentialBackoffRetryPolicy(config=config, rng=lambda: 0.0)

    assert [policy.delay_for(n) for n in (2, 3, 4, 5)] == [1.0, 2.0, 4.0, 8.0]


def test_backoff_is_clamped_to_the_ceiling() -> None:
    config = RetryConfig(
        base_delay_seconds=1.0,
        backoff_multiplier=10.0,
        max_delay_seconds=5.0,
        jitter_ratio=0.0,
    )
    policy = ExponentialBackoffRetryPolicy(config=config, rng=lambda: 0.0)

    assert policy.delay_for(6) == 5.0


def test_jitter_reduces_delay_within_the_configured_ratio() -> None:
    config = RetryConfig(
        base_delay_seconds=4.0,
        backoff_multiplier=1.0,
        max_delay_seconds=100.0,
        jitter_ratio=0.25,
    )

    full = ExponentialBackoffRetryPolicy(config=config, rng=lambda: 1.0)
    none = ExponentialBackoffRetryPolicy(config=config, rng=lambda: 0.0)

    assert none.delay_for(2) == pytest.approx(4.0)
    assert full.delay_for(2) == pytest.approx(3.0)


def test_delay_is_never_negative_under_maximum_jitter() -> None:
    config = RetryConfig(jitter_ratio=1.0)
    policy = ExponentialBackoffRetryPolicy(config=config, rng=lambda: 1.0)

    assert policy.delay_for(2) >= 0.0


def test_injected_rng_makes_delays_deterministic() -> None:
    policy = _config().build_retry_policy(rng=lambda: 0.5)

    assert policy.delay_for(3) == policy.delay_for(3)


def test_invalid_attempt_index_is_rejected() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        _config().build_retry_policy().delay_for(0)


def test_retrying_can_be_disabled() -> None:
    assert _config(retry=RetryConfig(max_attempts=1)).build_retry_policy().max_attempts == 1


# ---------------------------------------------------------------------------
# Thread limit derivation
# ---------------------------------------------------------------------------


def test_explicit_thread_limit_is_honoured() -> None:
    assert _config(extraction_thread_limit=3).resolve_thread_limit() == 3


def test_thread_limit_is_derived_from_cpu_count(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(os, "cpu_count", lambda: 6)

    assert _config().resolve_thread_limit() == 12


def test_thread_limit_falls_back_when_cpu_count_is_unknown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(os, "cpu_count", lambda: None)

    assert _config().resolve_thread_limit() >= 1


# ---------------------------------------------------------------------------
# Environment layering
# ---------------------------------------------------------------------------


def test_top_level_values_load_from_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PTFM_INGESTION_MAX_CONCURRENT_DOCUMENTS", "32")
    monkeypatch.setenv("PTFM_INGESTION_FAIL_FAST", "true")

    config = _config()

    assert config.max_concurrent_documents == 32
    assert config.fail_fast is True


def test_nested_values_load_from_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PTFM_INGESTION_RETRY__MAX_ATTEMPTS", "7")
    monkeypatch.setenv("PTFM_INGESTION_ACQUISITION__MAX_DOCUMENT_BYTES", "1048576")

    config = _config()

    assert config.retry.max_attempts == 7
    assert config.acquisition.max_document_bytes == 1 * MIB


def test_explicit_arguments_override_the_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PTFM_INGESTION_MAX_CONCURRENT_DOCUMENTS", "32")

    assert _config(max_concurrent_documents=2).max_concurrent_documents == 2


def test_invalid_environment_value_fails_at_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PTFM_INGESTION_MAX_CONCURRENT_DOCUMENTS", "0")

    with pytest.raises(ValidationError):
        _config()


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
    assert derived.retry == original.retry


def test_overrides_revalidate_cross_field_constraints() -> None:
    """A copy that skipped validation could hold a rejected combination."""
    original = _config()

    with pytest.raises(IngestionConfigurationError):
        original.with_overrides(
            acquisition=AcquisitionConfig(
                spool_threshold_bytes=4 * KIB, read_chunk_bytes=1 * KIB
            ).model_dump()
        )


def test_invalid_override_raises_a_ptfm_exception() -> None:
    with pytest.raises(IngestionConfigurationError):
        _config().with_overrides(max_concurrent_documents=0)


def test_configuration_round_trips_through_json() -> None:
    original = _config(max_concurrent_documents=4)

    restored = IngestionConfig.model_validate_json(original.model_dump_json())

    assert restored == original


# ---------------------------------------------------------------------------
# Anti-pattern guards
# ---------------------------------------------------------------------------


def test_config_declares_no_verification_thresholds() -> None:
    """
    Operational limits only. A verification or calibration threshold here
    would place a hardcoded truth judgement in the ingestion path.
    """
    forbidden = {
        "confidence_threshold",
        "verification_threshold",
        "calibration_threshold",
        "nli_threshold",
        "entailment_threshold",
        "tau",
        "tau_low",
        "alpha",
        "conformal_threshold",
        "credibility_threshold",
        "relevance_threshold",
    }

    groups: list[type] = [
        IngestionConfig,
        RetryConfig,
        AcquisitionConfig,
        DetectionConfig,
        ExtractionConfig,
        StructureConfig,
        MetadataConfig,
        DeduplicationConfig,
        TelemetryConfig,
    ]
    for group in groups:
        assert not forbidden & set(group.model_fields), group.__name__


def test_only_the_documented_field_observes_a_model_score() -> None:
    """Language detection confidence is the sole permitted exception."""
    scored = {
        name
        for group in (MetadataConfig, DetectionConfig, ExtractionConfig, StructureConfig)
        for name in group.model_fields
        if "confidence" in name
    }

    assert scored == {"language_detection_min_confidence"}


def test_failure_ratio_guard_requires_a_minimum_sample() -> None:
    """A ratio alone would abort a run when the first of one document fails."""
    assert _config().failure_ratio_min_samples > 1


def test_module_declares_no_mutable_state() -> None:
    from ptfm.core.ingestion import config as module

    mutable = [
        name
        for name, value in vars(module).items()
        if not name.startswith("_") and isinstance(value, (list, dict, set))
    ]
    assert mutable == []


def test_config_does_not_import_models_or_stages() -> None:
    import inspect

    from ptfm.core.ingestion import config as module

    source = inspect.getsource(module)
    for forbidden in (
        "from ptfm.core.ingestion.models",
        "from ptfm.core.ingestion.pipeline",
        "from ptfm.core.ingestion.connectors",
        "from ptfm.core.ingestion.extractors",
    ):
        assert forbidden not in source