"""Unit tests for the Data Ingestion Module internal models."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from pydantic import TypeAdapter, ValidationError

from ptfm.core.ingestion.exceptions import (
    CorruptDocumentError,
    IngestionStage,
    SourceTimeoutError,
)
from ptfm.core.ingestion.models import (
    AcquiredBlob,
    DetectedFormat,
    DetectionSignal,
    DetectionSignalSource,
    DocumentFailed,
    DocumentIngested,
    DocumentSkipped,
    ExtractedBlock,
    ExtractedMetadata,
    ExtractionInput,
    ExtractionOutput,
    IngestionFailure,
    IngestionOutcome,
    IngestionReport,
    SkipReason,
    SourceRecord,
    SourceSpec,
)
from ptfm.core.schemas import (
    BlockType,
    DocumentMetadata,
    ExtractionWarning,
    ExtractionWarningCode,
    RawDocument,
    SourceDescriptor,
    derive_document_id,
)

FIXED_TIME = datetime(2026, 1, 1, tzinfo=UTC)


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def _record(uri: str = "file:///corpus/a.txt") -> SourceRecord:
    return SourceRecord(
        record_id="r-1",
        source_uri=uri,
        scheme="file",
        connector_name="LocalFilesystemConnector",
        discovered_at=FIXED_TIME,
        filename="a.txt",
        size_bytes=5,
    )


def _blob(data: bytes = b"hello", **overrides: object) -> AcquiredBlob:
    payload: dict[str, object] = {
        "record": _record(),
        "size_bytes": len(data),
        "content_hash": "hash-1",
        "hash_algorithm": "blake2b-256",
        "acquired_at": FIXED_TIME,
        "data": data,
    }
    payload.update(overrides)
    return AcquiredBlob(**payload)  # type: ignore[arg-type]


def _raw_document(text: str = "hello") -> RawDocument:
    return RawDocument(
        document_id=derive_document_id("file:///corpus/a.txt", "hash-1"),
        source=SourceDescriptor(
            source_uri="file:///corpus/a.txt",
            scheme="file",
            connector_name="LocalFilesystemConnector",
            acquired_at=FIXED_TIME,
            size_bytes=len(text),
            detected_media_type="text/plain",
        ),
        metadata=DocumentMetadata(char_count=len(text), word_count=len(text.split())),
        text=text,
        content_hash="hash-1",
        text_hash="hash-2",
        ingested_at=FIXED_TIME,
    )


def _leaf(text: str, parent_index: int | None = None) -> ExtractedBlock:
    return ExtractedBlock(
        block_type=BlockType.PARAGRAPH, text=text, parent_index=parent_index
    )


# ---------------------------------------------------------------------------
# SourceSpec
# ---------------------------------------------------------------------------


def test_spec_exposes_scheme_from_root_uri() -> None:
    assert SourceSpec(root_uri="file:///corpus").scheme == "file"
    assert SourceSpec(root_uri="S3://bucket/prefix").scheme == "s3"


def test_spec_requires_a_scheme() -> None:
    with pytest.raises(ValidationError, match="must include a scheme"):
        SourceSpec(root_uri="/corpus/documents")


def test_spec_is_frozen_and_rejects_unknown_fields() -> None:
    spec = SourceSpec(root_uri="file:///corpus")

    with pytest.raises(ValidationError):
        spec.recursive = False  # type: ignore[misc]
    with pytest.raises(ValidationError):
        SourceSpec(root_uri="file:///c", depth=3)  # type: ignore[call-arg]


def test_spec_options_reject_structured_values() -> None:
    """String-only options block domain objects from entering via config."""
    with pytest.raises(ValidationError):
        SourceSpec(root_uri="file:///c", options={"filters": {"court": "SCOTUS"}})  # type: ignore[dict-item]


def test_max_documents_must_be_positive() -> None:
    with pytest.raises(ValidationError):
        SourceSpec(root_uri="file:///c", max_documents=0)


# ---------------------------------------------------------------------------
# AcquiredBlob
# ---------------------------------------------------------------------------


def test_blob_requires_exactly_one_payload_representation() -> None:
    with pytest.raises(ValueError, match="exactly one"):
        AcquiredBlob(
            record=_record(),
            size_bytes=0,
            content_hash="h",
            hash_algorithm="blake2b-256",
            acquired_at=FIXED_TIME,
        )
    with pytest.raises(ValueError, match="exactly one"):
        AcquiredBlob(
            record=_record(),
            size_bytes=5,
            content_hash="h",
            hash_algorithm="blake2b-256",
            acquired_at=FIXED_TIME,
            data=b"hello",
            spool_path=Path("/tmp/x"),
        )


def test_blob_size_must_match_in_memory_payload() -> None:
    with pytest.raises(ValueError, match="does not match"):
        _blob(b"hello", size_bytes=99)


def test_in_memory_blob_reads_and_slices() -> None:
    blob = _blob(b"hello world")

    assert blob.is_spooled is False
    assert blob.read() == b"hello world"
    assert blob.header(5) == b"hello"
    assert blob.header(1000) == b"hello world"
    assert blob.source_uri == "file:///corpus/a.txt"


def test_header_rejects_negative_size() -> None:
    with pytest.raises(ValueError):
        _blob().header(-1)


def test_spooled_blob_reads_prefix_without_materializing(tmp_path: Path) -> None:
    """Detection must not force a large payload into memory."""
    spool = tmp_path / "payload.bin"
    spool.write_bytes(b"%PDF-1.7" + b"x" * 4096)

    blob = AcquiredBlob(
        record=_record(),
        size_bytes=spool.stat().st_size,
        content_hash="hash-1",
        hash_algorithm="blake2b-256",
        acquired_at=FIXED_TIME,
        spool_path=spool,
    )

    assert blob.is_spooled is True
    assert blob.header(8) == b"%PDF-1.7"
    assert len(blob.read()) == spool.stat().st_size


def test_close_removes_spool_file_and_is_idempotent(tmp_path: Path) -> None:
    spool = tmp_path / "payload.bin"
    spool.write_bytes(b"data")
    blob = AcquiredBlob(
        record=_record(),
        size_bytes=4,
        content_hash="h",
        hash_algorithm="blake2b-256",
        acquired_at=FIXED_TIME,
        spool_path=spool,
    )

    blob.close()
    blob.close()

    assert not spool.exists()


def test_use_after_close_is_rejected() -> None:
    blob = _blob()
    blob.close()

    with pytest.raises(ValueError, match="has been closed"):
        blob.read()
    with pytest.raises(ValueError, match="has been closed"):
        blob.header(4)


def test_context_manager_releases_on_failure(tmp_path: Path) -> None:
    spool = tmp_path / "payload.bin"
    spool.write_bytes(b"data")
    blob = AcquiredBlob(
        record=_record(),
        size_bytes=4,
        content_hash="h",
        hash_algorithm="blake2b-256",
        acquired_at=FIXED_TIME,
        spool_path=spool,
    )

    class _Boom(Exception):
        pass

    with pytest.raises(_Boom):
        with blob:
            raise _Boom

    assert not spool.exists()


def test_blob_repr_never_leaks_payload_bytes() -> None:
    """A stray log line must not emit an entire document."""
    blob = _blob(b"SENSITIVE-CONTENT")

    assert "SENSITIVE" not in repr(blob)
    assert "size_bytes=17" in repr(blob)


def test_blob_is_not_a_value_object() -> None:
    """Resource handles must not compare equal by content."""
    assert _blob() != _blob()


# ---------------------------------------------------------------------------
# ExtractionInput
# ---------------------------------------------------------------------------


def test_extraction_input_repr_never_leaks_payload() -> None:
    payload = ExtractionInput(
        source_uri="file:///a.txt", media_type="text/plain", data=b"SENSITIVE"
    )

    assert "SENSITIVE" not in repr(payload)
    assert payload.size_bytes == 9


def test_extraction_input_is_frozen() -> None:
    payload = ExtractionInput(
        source_uri="file:///a.txt", media_type="text/plain", data=b"x"
    )

    with pytest.raises(AttributeError):
        payload.media_type = "application/pdf"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# DetectedFormat
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("media_type", "is_text"),
    [
        ("text/plain", True),
        ("text/html", True),
        ("application/json", True),
        ("application/xhtml+xml", True),
        ("application/pdf", False),
        ("image/png", False),
    ],
)
def test_text_classification(media_type: str, is_text: bool) -> None:
    assert DetectedFormat(media_type=media_type).is_text is is_text


def test_signals_are_retrievable_by_source() -> None:
    detected = DetectedFormat(
        media_type="application/pdf",
        signals=(
            DetectionSignal(
                source=DetectionSignalSource.MAGIC_BYTES, media_type="application/pdf"
            ),
            DetectionSignal(
                source=DetectionSignalSource.EXTENSION, media_type="text/plain"
            ),
        ),
        warnings=(
            ExtractionWarning(
                code=ExtractionWarningCode.MEDIA_TYPE_MISMATCH,
                message="extension disagrees with magic bytes",
            ),
        ),
    )

    magic = detected.signal_for(DetectionSignalSource.MAGIC_BYTES)

    assert magic is not None and magic.media_type == "application/pdf"
    assert detected.signal_for(DetectionSignalSource.DECLARED) is None
    assert len(detected.warnings) == 1


# ---------------------------------------------------------------------------
# ExtractedBlock and ExtractionOutput
# ---------------------------------------------------------------------------


def test_container_block_must_not_carry_text() -> None:
    with pytest.raises(ValidationError, match="must not carry text"):
        ExtractedBlock(block_type=BlockType.SECTION, text="header text")


def test_leaf_block_must_supply_text() -> None:
    with pytest.raises(ValidationError, match="must supply text"):
        ExtractionOutput(
            blocks=(ExtractedBlock(block_type=BlockType.PARAGRAPH),),
            extractor_name="X",
        )


def test_structural_marker_may_carry_empty_text() -> None:
    output = ExtractionOutput(
        blocks=(ExtractedBlock(block_type=BlockType.PAGE_BREAK, text=""),),
        extractor_name="PdfExtractor",
    )

    assert output.is_empty is True


def test_forward_parent_reference_is_rejected() -> None:
    """A parent after its child breaks pre-order and would permit cycles."""
    with pytest.raises(ValidationError, match="pre-order"):
        ExtractionOutput(
            blocks=(
                _leaf("child", parent_index=1),
                ExtractedBlock(block_type=BlockType.SECTION),
            ),
            extractor_name="X",
        )


def test_self_referencing_parent_is_rejected() -> None:
    with pytest.raises(ValidationError, match="pre-order"):
        ExtractionOutput(blocks=(_leaf("x", parent_index=0),), extractor_name="X")


def test_leaf_type_may_not_have_children() -> None:
    with pytest.raises(ValidationError, match="may not have children"):
        ExtractionOutput(
            blocks=(_leaf("parent"), _leaf("child", parent_index=0)),
            extractor_name="X",
        )


def test_empty_container_is_rejected() -> None:
    with pytest.raises(ValidationError, match="has no children"):
        ExtractionOutput(
            blocks=(ExtractedBlock(block_type=BlockType.SECTION),), extractor_name="X"
        )


def test_valid_nested_output_is_accepted() -> None:
    output = ExtractionOutput(
        blocks=(
            ExtractedBlock(block_type=BlockType.SECTION),
            ExtractedBlock(
                block_type=BlockType.HEADING, text="Title", parent_index=0
            ),
            _leaf("Body text.", parent_index=0),
        ),
        metadata=ExtractedMetadata(title="Title", page_count=1),
        extractor_name="DocxExtractor",
    )

    assert [index for index, _ in output.iter_leaf_blocks()] == [1, 2]
    assert output.is_empty is False


def test_whitespace_only_output_is_empty() -> None:
    output = ExtractionOutput(blocks=(_leaf("   \n  "),), extractor_name="X")

    assert output.is_empty is True


def test_extracted_metadata_rejects_structured_properties() -> None:
    with pytest.raises(ValidationError):
        ExtractedMetadata(properties={"citation": {"court": "SCOTUS"}})  # type: ignore[dict-item]


def test_extracted_metadata_preserves_typed_timestamps() -> None:
    metadata = ExtractedMetadata(created_at=FIXED_TIME)

    assert metadata.created_at is not None
    assert metadata.created_at.tzinfo is not None


# ---------------------------------------------------------------------------
# IngestionFailure
# ---------------------------------------------------------------------------


def test_failure_captures_stage_and_retryability_from_the_exception() -> None:
    try:
        raise CorruptDocumentError(
            "xref damaged", source_uri="file:///a.pdf", document_id="doc-1"
        )
    except CorruptDocumentError as error:
        failure = IngestionFailure.from_exception(
            error, source_uri="file:///a.pdf", failed_at=FIXED_TIME
        )

    assert failure.stage is IngestionStage.EXTRACTION
    assert failure.error_code == "CORRUPT_DOCUMENT"
    assert failure.error_type == "CorruptDocumentError"
    assert failure.retryable is False
    assert failure.document_id == "doc-1"
    assert failure.attempts == 1
    assert failure.context["stage"] == IngestionStage.EXTRACTION.value


def test_failure_records_retryable_exceptions_with_attempt_count() -> None:
    try:
        raise SourceTimeoutError("read timed out", timeout_seconds=5.0)
    except SourceTimeoutError as error:
        failure = IngestionFailure.from_exception(
            error, source_uri="file:///a.txt", failed_at=FIXED_TIME, attempts=3
        )

    assert failure.retryable is True
    assert failure.attempts == 3
    assert failure.stage is IngestionStage.ACQUISITION


def test_traceback_digest_groups_identical_failures() -> None:
    def _raise() -> None:
        raise CorruptDocumentError("same failure")

    digests: list[str | None] = []
    for _ in range(2):
        try:
            _raise()
        except CorruptDocumentError as error:
            digests.append(
                IngestionFailure.from_exception(
                    error, source_uri="file:///a", failed_at=FIXED_TIME
                ).traceback_digest
            )

    assert digests[0] is not None
    assert digests[0] == digests[1]


def test_unraised_exception_yields_no_digest() -> None:
    failure = IngestionFailure.from_exception(
        CorruptDocumentError("never raised"),
        source_uri="file:///a",
        failed_at=FIXED_TIME,
    )

    assert failure.traceback_digest is None


def test_failure_prefers_uri_carried_by_the_exception() -> None:
    failure = IngestionFailure.from_exception(
        CorruptDocumentError("bad", source_uri="file:///actual.pdf"),
        source_uri="file:///fallback.pdf",
        failed_at=FIXED_TIME,
    )

    assert failure.source_uri == "file:///actual.pdf"


# ---------------------------------------------------------------------------
# Outcome union
# ---------------------------------------------------------------------------


def test_duplicate_is_a_skip_not_a_failure() -> None:
    """Duplication is the expected steady state of incremental re-ingestion."""
    outcome = DocumentSkipped(
        source_uri="file:///a.txt",
        reason=SkipReason.DUPLICATE_CONTENT,
        existing_document_id="doc-1",
    )

    assert outcome.kind == "SKIPPED"
    assert not isinstance(outcome, DocumentFailed)


def test_outcome_accessors_expose_source_uri_uniformly() -> None:
    ingested = DocumentIngested(document=_raw_document(), duration_ms=12.5)
    failed = DocumentFailed(
        failure=IngestionFailure.from_exception(
            CorruptDocumentError("bad"), source_uri="file:///b.pdf", failed_at=FIXED_TIME
        )
    )

    assert ingested.source_uri == "file:///corpus/a.txt"
    assert failed.source_uri == "file:///b.pdf"


@pytest.mark.parametrize(
    ("outcome", "expected_kind"),
    [
        (DocumentIngested(document=_raw_document(), duration_ms=1.0), "INGESTED"),
        (
            DocumentSkipped(
                source_uri="file:///a", reason=SkipReason.UNCHANGED_VERSION
            ),
            "SKIPPED",
        ),
        (
            DocumentFailed(
                failure=IngestionFailure.from_exception(
                    CorruptDocumentError("bad"),
                    source_uri="file:///a",
                    failed_at=FIXED_TIME,
                )
            ),
            "FAILED",
        ),
    ],
)
def test_outcome_union_round_trips_by_discriminator(
    outcome: object, expected_kind: str
) -> None:
    adapter: TypeAdapter[IngestionOutcome] = TypeAdapter(IngestionOutcome)

    restored = adapter.validate_json(adapter.dump_json(outcome))

    assert restored.kind == expected_kind
    assert type(restored) is type(outcome)
    assert restored == outcome


def test_unknown_discriminator_is_rejected() -> None:
    adapter: TypeAdapter[IngestionOutcome] = TypeAdapter(IngestionOutcome)

    with pytest.raises(ValidationError):
        adapter.validate_python({"kind": "UNKNOWN"})


def test_negative_duration_is_rejected() -> None:
    with pytest.raises(ValidationError):
        DocumentIngested(document=_raw_document(), duration_ms=-1.0)


# ---------------------------------------------------------------------------
# IngestionReport
# ---------------------------------------------------------------------------


def _report(**overrides: object) -> IngestionReport:
    payload: dict[str, object] = {
        "run_id": "run-1",
        "root_uri": "file:///corpus",
        "started_at": FIXED_TIME,
        "completed_at": FIXED_TIME + timedelta(seconds=90),
    }
    payload.update(overrides)
    return IngestionReport(**payload)  # type: ignore[arg-type]


def test_report_aggregates_counts_and_duration() -> None:
    report = _report(
        discovered_count=100,
        ingested_count=80,
        skipped_count=15,
        failed_count=5,
        failures_by_stage={IngestionStage.EXTRACTION.value: 5},
        documents_by_media_type={"application/pdf": 80},
        total_bytes=1_048_576,
    )

    assert report.processed_count == 100
    assert report.failure_ratio == pytest.approx(0.05)
    assert report.duration_seconds == pytest.approx(90.0)


def test_failure_ratio_is_zero_for_an_empty_run() -> None:
    assert _report().failure_ratio == 0.0


def test_report_rejects_inverted_interval() -> None:
    with pytest.raises(ValidationError, match="must not precede"):
        _report(completed_at=FIXED_TIME - timedelta(seconds=1))


def test_aborted_run_records_its_reason() -> None:
    report = _report(
        ingested_count=10, failed_count=90, aborted=True, abort_reason="failure ratio"
    )

    assert report.aborted is True
    assert report.failure_ratio == pytest.approx(0.9)


def test_report_json_round_trip() -> None:
    report = _report(ingested_count=3, failures_by_code={"CORRUPT_DOCUMENT": 2})

    restored = IngestionReport.model_validate_json(report.model_dump_json())

    assert restored == report


# ---------------------------------------------------------------------------
# Anti-pattern guards
# ---------------------------------------------------------------------------


def test_no_model_exposes_a_confidence_field() -> None:
    from ptfm.core.ingestion import models

    forbidden = {"confidence", "score", "probability", "calibrated_confidence"}
    for name in models.__all__:
        candidate = getattr(models, name)
        fields = getattr(candidate, "model_fields", None)
        if fields is not None:
            assert not forbidden & set(fields), name


def test_module_declares_no_mutable_state() -> None:
    from ptfm.core.ingestion import models

    mutable = [
        name
        for name, value in vars(models).items()
        if not name.startswith("__") and isinstance(value, (list, dict, set))
    ]
    assert mutable == []


def test_models_does_not_import_interfaces_or_stages() -> None:
    """Preserves the acyclic dependency graph and the generation order."""
    import inspect

    from ptfm.core.ingestion import models

    source = inspect.getsource(models)
    for forbidden in (
        "from ptfm.core.ingestion.interfaces",
        "from ptfm.core.ingestion.pipeline",
        "from ptfm.core.ingestion.connectors",
        "from ptfm.core.ingestion.extractors",
    ):
        assert forbidden not in source