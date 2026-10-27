"""Unit tests for the PTFM root exception hierarchy."""

from __future__ import annotations

import pickle

import pytest

from ptfm.utils.exceptions import PTFMError, _derive_error_code


class _SampleError(PTFMError):
    """Fixture subclass used to exercise base-class behaviour."""


class _PinnedCodeError(PTFMError):
    """Fixture subclass that pins an explicit error code."""

    error_code = "LEGACY_CONTRACT_CODE"


@pytest.mark.parametrize(
    ("class_name", "expected"),
    [
        ("SourceUnavailableError", "SOURCE_UNAVAILABLE"),
        ("PTFMIngestionError", "PTFM_INGESTION"),
        ("DocumentTooLargeError", "DOCUMENT_TOO_LARGE"),
        ("SpanIntegrityError", "SPAN_INTEGRITY"),
        ("PTFMError", "PTFM"),
    ],
)
def test_error_code_derivation_handles_acronyms(class_name: str, expected: str) -> None:
    assert _derive_error_code(class_name) == expected


def test_subclass_receives_derived_error_code() -> None:
    assert _SampleError.error_code == "_SAMPLE"


def test_explicit_error_code_is_not_overwritten() -> None:
    assert _PinnedCodeError.error_code == "LEGACY_CONTRACT_CODE"


def test_default_is_not_retryable() -> None:
    assert PTFMError.retryable is False
    assert _SampleError("boom").retryable is False


def test_context_is_copied_and_immutable() -> None:
    supplied = {"key": "value"}
    error = _SampleError("boom", context=supplied)

    supplied["key"] = "mutated"

    assert error.context["key"] == "value"
    with pytest.raises(TypeError):
        error.context["key"] = "forbidden"  # type: ignore[index]


def test_add_context_merges_additional_detail() -> None:
    error = _SampleError("boom", context={"a": 1})
    error.add_context(b=2)

    assert dict(error.context) == {"a": 1, "b": 2}


def test_to_dict_includes_chained_cause() -> None:
    try:
        try:
            raise ValueError("underlying failure")
        except ValueError as exc:
            raise _SampleError("wrapped", context={"k": "v"}) from exc
    except _SampleError as error:
        payload = error.to_dict()

    assert payload["error_type"] == "_SampleError"
    assert payload["error_code"] == "_SAMPLE"
    assert payload["message"] == "wrapped"
    assert payload["retryable"] is False
    assert payload["context"] == {"k": "v"}
    assert payload["cause_type"] == "ValueError"
    assert payload["cause_message"] == "underlying failure"


def test_to_dict_omits_cause_keys_when_unchained() -> None:
    payload = _SampleError("boom").to_dict()

    assert "cause_type" not in payload
    assert "cause_message" not in payload


def test_pickle_roundtrip_preserves_state() -> None:
    original = _SampleError("boom", context={"a": 1})

    restored = pickle.loads(pickle.dumps(original))

    assert type(restored) is _SampleError
    assert restored.message == "boom"
    assert dict(restored.context) == {"a": 1}
    assert str(restored) == "boom"


def test_str_and_repr() -> None:
    error = _SampleError("boom")

    assert str(error) == "boom"
    assert repr(error) == "_SampleError(code='_SAMPLE', message='boom')"