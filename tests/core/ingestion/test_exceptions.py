"""Unit tests for the Data Ingestion Module exception hierarchy."""

from __future__ import annotations

import pickle

import pytest

from ptfm.core.ingestion import exceptions as ingestion_exceptions
from ptfm.core.ingestion.exceptions import (
    ChecksumMismatchError,
    CorruptDocumentError,
    DeduplicationError,
    DocumentTooLargeError,
    ExtractionTimeoutError,
    ExtractorInternalError,
    ExtractorUnavailableError,
    FingerprintStoreError,
    IngestionStage,
    PTFMIngestionError,
    SchemaValidationError,
    SourceAuthenticationError,
    SourceTimeoutError,
    SourceUnavailableError,
    SpanIntegrityError,
    UnsupportedMediaTypeError,
    UnsupportedSourceSchemeError,
)
from ptfm.utils.exceptions import PTFMError

RETRYABLE_TYPES: frozenset[type[PTFMIngestionError]] = frozenset(
    {SourceUnavailableError, SourceTimeoutError}
)


def _all_ingestion_error_types() -> list[type[PTFMIngestionError]]:
    """Collect every exported ingestion exception class."""
    return [
        obj
        for name in ingestion_exceptions.__all__
        if isinstance(obj := getattr(ingestion_exceptions, name), type)
        and issubclass(obj, PTFMIngestionError)
    ]


def _concrete_leaf_types() -> list[type[PTFMIngestionError]]:
    """Collect exception classes that have no subclasses in the hierarchy."""
    all_types = _all_ingestion_error_types()
    return [
        candidate
        for candidate in all_types
        if not any(other is not candidate and issubclass(other, candidate) for other in all_types)
    ]


# ---------------------------------------------------------------------------
# Hierarchy invariants
# ---------------------------------------------------------------------------


def test_every_ingestion_error_derives_from_ptfm_error() -> None:
    for error_type in _all_ingestion_error_types():
        assert issubclass(error_type, PTFMError)


def test_every_leaf_declares_a_concrete_stage() -> None:
    for error_type in _concrete_leaf_types():
        assert error_type.stage is not IngestionStage.UNSPECIFIED, error_type.__name__


def test_only_transport_transients_are_retryable() -> None:
    for error_type in _all_ingestion_error_types():
        expected = any(issubclass(error_type, base) for base in RETRYABLE_TYPES)
        assert error_type.retryable is expected, error_type.__name__


def test_duplicate_documents_are_not_modelled_as_an_exception() -> None:
    assert not hasattr(ingestion_exceptions, "DuplicateDocumentError")


def test_all_exports_resolve() -> None:
    for name in ingestion_exceptions.__all__:
        assert hasattr(ingestion_exceptions, name), name


def test_error_codes_are_unique() -> None:
    codes = [error_type.error_code for error_type in _all_ingestion_error_types()]
    assert len(codes) == len(set(codes))


# ---------------------------------------------------------------------------
# Base behaviour
# ---------------------------------------------------------------------------


def test_stage_and_identity_are_serialized() -> None:
    error = CorruptDocumentError(
        "unreadable payload",
        source_uri="file:///corpus/broken.pdf",
        document_id="doc-1",
    )

    payload = error.to_dict()

    assert payload["stage"] == IngestionStage.EXTRACTION.value
    assert payload["source_uri"] == "file:///corpus/broken.pdf"
    assert payload["document_id"] == "doc-1"
    assert payload["error_code"] == "CORRUPT_DOCUMENT"
    assert payload["retryable"] is False


def test_identity_is_mirrored_into_context_for_uniform_logging() -> None:
    error = CorruptDocumentError("bad", source_uri="file:///a", document_id="doc-2")

    assert error.context["source_uri"] == "file:///a"
    assert error.context["document_id"] == "doc-2"
    assert error.context["stage"] == IngestionStage.EXTRACTION.value


# ---------------------------------------------------------------------------
# Field-carrying subclasses
# ---------------------------------------------------------------------------


def test_document_too_large_generates_actionable_message() -> None:
    error = DocumentTooLargeError(size_bytes=2_048, limit_bytes=1_024)

    assert error.size_bytes == 2_048
    assert error.limit_bytes == 1_024
    assert "2048" in error.message and "1024" in error.message
    assert error.context["limit_bytes"] == 1_024


def test_unsupported_scheme_lists_registered_schemes() -> None:
    error = UnsupportedSourceSchemeError(scheme="s3", supported_schemes=["file", "memory"])

    assert error.scheme == "s3"
    assert error.supported_schemes == ("file", "memory")
    assert "file, memory" in error.message


def test_unsupported_media_type_reports_empty_registry_clearly() -> None:
    error = UnsupportedMediaTypeError(media_type="application/pdf")

    assert "none" in error.message
    assert error.supported_media_types == ()


def test_extractor_unavailable_names_the_install_extra() -> None:
    error = ExtractorUnavailableError(
        media_type="application/pdf",
        missing_dependency="pypdfium2",
        install_extra="ptfm[pdf]",
    )

    assert error.install_extra == "ptfm[pdf]"
    assert "pypdfium2" in error.message
    assert error.context["media_type"] == "application/pdf"


def test_checksum_mismatch_records_both_digests() -> None:
    error = ChecksumMismatchError(
        expected_digest="aaaa", actual_digest="bbbb", algorithm="blake2b"
    )

    assert error.expected_digest == "aaaa"
    assert error.actual_digest == "bbbb"
    assert error.context["algorithm"] == "blake2b"


def test_extraction_timeout_records_deadline_and_is_not_retryable() -> None:
    error = ExtractionTimeoutError("parser deadline exceeded", timeout_seconds=30.0)

    assert error.timeout_seconds == 30.0
    assert error.retryable is False


def test_source_timeout_is_retryable_and_records_deadline() -> None:
    error = SourceTimeoutError("read timed out", timeout_seconds=5.0)

    assert error.retryable is True
    assert error.context["timeout_seconds"] == 5.0


def test_authentication_failure_is_permanent() -> None:
    assert SourceAuthenticationError("403 forbidden").retryable is False


def test_span_integrity_error_carries_reconstruction_lengths() -> None:
    error = SpanIntegrityError(
        "leaf spans do not reconstruct document text",
        block_id="b.3.1",
        expected_length=1_000,
        actual_length=998,
    )

    assert error.block_id == "b.3.1"
    assert error.context["expected_length"] == 1_000
    assert error.context["actual_length"] == 998


def test_schema_validation_error_stores_plain_mappings() -> None:
    error = SchemaValidationError(
        "RawDocument failed validation",
        validation_errors=[{"loc": ("blocks", 0), "msg": "field required"}],
    )

    assert error.validation_errors[0]["msg"] == "field required"
    assert isinstance(error.context["validation_errors"], list)


def test_fingerprint_store_failure_is_not_retryable() -> None:
    error = FingerprintStoreError("store unreachable", operation="lookup")

    assert isinstance(error, DeduplicationError)
    assert error.retryable is False
    assert error.operation == "lookup"


# ---------------------------------------------------------------------------
# Boundary wrapping and process transport
# ---------------------------------------------------------------------------


def test_third_party_exception_is_preserved_as_cause() -> None:
    class _VendorParserError(Exception):
        """Stand-in for a third-party parser exception."""

    try:
        try:
            raise _VendorParserError("xref table damaged")
        except _VendorParserError as exc:
            raise ExtractorInternalError(
                "PDF extraction failed",
                extractor_name="PdfExtractor",
                source_uri="file:///corpus/a.pdf",
            ) from exc
    except ExtractorInternalError as error:
        assert isinstance(error.__cause__, _VendorParserError)
        assert error.extractor_name == "PdfExtractor"
        assert error.to_dict()["cause_type"] == "_VendorParserError"


@pytest.mark.parametrize(
    "error",
    [
        DocumentTooLargeError(size_bytes=10, limit_bytes=5, source_uri="file:///a"),
        UnsupportedSourceSchemeError(scheme="s3", supported_schemes=["file"]),
        ExtractorInternalError("boom", extractor_name="PdfExtractor"),
        SourceTimeoutError("slow", timeout_seconds=1.5),
    ],
)
def test_keyword_only_exceptions_survive_process_boundaries(
    error: PTFMIngestionError,
) -> None:
    restored = pickle.loads(pickle.dumps(error))

    assert type(restored) is type(error)
    assert restored.message == error.message
    assert restored.stage is error.stage
    assert restored.retryable is error.retryable
    assert dict(restored.context) == dict(error.context)