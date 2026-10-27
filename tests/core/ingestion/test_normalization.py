"""
Unit tests for the canonical assembly stage.

File:
    tests/core/ingestion/test_normalization.py
"""

from __future__ import annotations

import inspect
from datetime import UTC, datetime
from typing import Any, Optional, Sequence

import pytest

from ptfm.core.ingestion.exceptions import (
    IngestionStage,
    NormalizationError,
    SchemaValidationError,
)
from ptfm.core.ingestion.models import (
    AcquiredBlob,
    DetectedFormat,
    DetectionSignal,
    DetectionSignalSource,
    SourceRecord,
)
from ptfm.core.ingestion.pipeline import normalization as normalization_module
from ptfm.core.ingestion.pipeline.metadata import HarvestedMetadata
from ptfm.core.ingestion.pipeline.normalization import RawDocumentNormalizer
from ptfm.core.ingestion.pipeline.structure import NormalizedStructure
from ptfm.core.ingestion.utils.hashing import Blake2bHashProvider
from ptfm.core.schemas import (
    RAW_DOCUMENT_SCHEMA_VERSION,
    BlockType,
    DocumentBlock,
    DocumentMetadata,
    ExtractionWarning,
    ExtractionWarningCode,
    RawDocument,
    TextSpan,
    derive_document_id,
)

ACQUIRED_AT = datetime(2026, 7, 30, 9, 0, tzinfo=UTC)
INGESTED_AT = datetime(2026, 7, 30, 9, 5, tzinfo=UTC)

SOURCE_URI = "file:///corpus/report.pdf"
CONTENT_HASH = "a1b2c3d4"


# ---------------------------------------------------------------------------
# Doubles and builders
# ---------------------------------------------------------------------------


class _FixedClock:
    """Clock returning a fixed, timezone-aware instant."""

    def __init__(self, instant: datetime = INGESTED_AT) -> None:
        self._instant = instant

    def now(self) -> datetime:
        return self._instant


class _NaiveClock:
    """Clock violating the aware-instant contract."""

    def now(self) -> datetime:
        return datetime(2026, 7, 30, 9, 5)


class _AlternateHashProvider:
    """Hash provider declaring a different algorithm."""

    @property
    def algorithm(self) -> str:
        return "sha256-fake"

    def hash_bytes(self, data: bytes) -> str:
        return data.hex()

    def hash_text(self, text: str) -> str:
        return text.encode("utf-8").hex()

    def new_incremental(self) -> Any:  # pragma: no cover - unused here
        raise NotImplementedError


def _record(
    uri: str = SOURCE_URI,
    *,
    declared_media_type: str | None = "application/pdf",
    version_tag: str | None = "etag-7",
    size_bytes: int | None = 4096,
) -> SourceRecord:
    return SourceRecord(
        record_id="r-1",
        source_uri=uri,
        scheme="file",
        connector_name="LocalFilesystemConnector",
        discovered_at=ACQUIRED_AT,
        filename="report.pdf",
        declared_media_type=declared_media_type,
        size_bytes=size_bytes,
        version_tag=version_tag,
    )


def _blob(
    *,
    record: SourceRecord | None = None,
    content_hash: str = CONTENT_HASH,
    algorithm: str = "blake2b-256",
    size_bytes: int = 2048,
) -> AcquiredBlob:
    return AcquiredBlob(
        record=record or _record(),
        size_bytes=size_bytes,
        content_hash=content_hash,
        hash_algorithm=algorithm,
        acquired_at=ACQUIRED_AT,
        data=b"x" * size_bytes,
    )


def _detected(
    media_type: str = "application/pdf",
    encoding: str | None = None,
    warnings: Sequence[ExtractionWarning] = (),
) -> DetectedFormat:
    return DetectedFormat(
        media_type=media_type,
        encoding=encoding,
        signals=(
            DetectionSignal(
                source=DetectionSignalSource.MAGIC_BYTES, media_type=media_type
            ),
        ),
        warnings=tuple(warnings),
    )


def _structure(
    text: str = "Heading\n\nBody paragraph.",
    *,
    with_blocks: bool = True,
    warnings: Sequence[ExtractionWarning] = (),
) -> NormalizedStructure:
    if not with_blocks or not text:
        return NormalizedStructure(text=text, blocks=(), warnings=tuple(warnings))

    heading, _, body = text.partition("\n\n")
    blocks: list[DocumentBlock] = [
        DocumentBlock(
            block_id="0",
            block_type=BlockType.HEADING,
            span=TextSpan(start=0, end=len(heading)),
            depth=0,
            ordinal=0,
        )
    ]
    if body:
        start = len(heading) + 2
        blocks.append(
            DocumentBlock(
                block_id="1",
                block_type=BlockType.PARAGRAPH,
                span=TextSpan(start=start, end=start + len(body)),
                depth=0,
                ordinal=1,
            )
        )
    return NormalizedStructure(
        text=text, blocks=tuple(blocks), warnings=tuple(warnings)
    )


def _metadata(
    text: str,
    *,
    char_count: int | None = None,
    warnings: Sequence[ExtractionWarning] = (),
    **fields: Any,
) -> HarvestedMetadata:
    return HarvestedMetadata(
        metadata=DocumentMetadata(
            char_count=len(text) if char_count is None else char_count,
            word_count=len(text.split()),
            **fields,
        ),
        warnings=tuple(warnings),
    )


def _normalizer(
    *,
    hash_provider: Any = None,
    clock: Any = None,
    max_warnings: int = 1_000,
) -> RawDocumentNormalizer:
    return RawDocumentNormalizer(
        hash_provider or Blake2bHashProvider(),
        clock or _FixedClock(),
        max_warnings=max_warnings,
    )


def _assemble(
    normalizer: RawDocumentNormalizer | None = None,
    *,
    blob: AcquiredBlob | None = None,
    detected: DetectedFormat | None = None,
    text: str = "Heading\n\nBody paragraph.",
    structure: NormalizedStructure | None = None,
    metadata: HarvestedMetadata | None = None,
    extraction_warnings: Sequence[ExtractionWarning] = (),
    deduplication_warning: Optional[ExtractionWarning] = None,
) -> RawDocument:
    resolved_structure = structure if structure is not None else _structure(text)
    return (normalizer or _normalizer()).assemble(
        blob=blob or _blob(),
        detected=detected or _detected(),
        structure=resolved_structure,
        metadata=metadata
        if metadata is not None
        else _metadata(resolved_structure.text),
        extraction_warnings=extraction_warnings,
        deduplication_warning=deduplication_warning,
    )


def _warning(
    code: ExtractionWarningCode, message: str
) -> ExtractionWarning:
    return ExtractionWarning(code=code, message=message)


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------


def test_assembly_produces_a_valid_canonical_document() -> None:
    document = _assemble()

    assert isinstance(document, RawDocument)
    assert document.text == "Heading\n\nBody paragraph."
    assert len(document.blocks) == 2
    assert document.schema_version == RAW_DOCUMENT_SCHEMA_VERSION


def test_text_and_blocks_are_carried_through_unchanged() -> None:
    structure = _structure("Alpha\n\nBeta")

    document = _assemble(structure=structure, metadata=_metadata(structure.text))

    assert document.text == structure.text
    assert document.blocks == structure.blocks


def test_metadata_record_is_carried_through_unchanged() -> None:
    text = "Body text here."
    metadata = _metadata(
        text, title="Quarterly Report", authors=("Ada Lovelace",), page_count=12
    )

    document = _assemble(text=text, structure=_structure(text), metadata=metadata)

    assert document.metadata is metadata.metadata
    assert document.metadata.title == "Quarterly Report"


def test_empty_document_is_assembled() -> None:
    structure = NormalizedStructure(text="", blocks=())

    document = _assemble(structure=structure, metadata=_metadata(""))

    assert document.text == ""
    assert document.blocks == ()
    assert document.metadata.char_count == 0


def test_assembly_reads_no_payload_bytes() -> None:
    """Only blob metadata is consulted, so a released payload is acceptable."""
    blob = _blob()
    blob.close()

    document = _assemble(blob=blob)

    assert document.content_hash == CONTENT_HASH


# ---------------------------------------------------------------------------
# Identity
# ---------------------------------------------------------------------------


def test_identifier_is_the_deterministic_derivation() -> None:
    document = _assemble()

    assert document.document_id == derive_document_id(SOURCE_URI, CONTENT_HASH)


def test_identical_inputs_produce_identical_identifiers() -> None:
    """Idempotency: re-ingesting the same bytes must not duplicate."""
    first = _assemble()
    second = _assemble()

    assert first.document_id == second.document_id


def test_assembly_is_byte_identical_for_identical_inputs() -> None:
    assert _assemble() == _assemble()


def test_differing_payload_yields_a_differing_identifier() -> None:
    baseline = _assemble()

    altered = _assemble(blob=_blob(content_hash="ffffffff"))

    assert altered.document_id != baseline.document_id


def test_differing_uri_yields_a_differing_identifier() -> None:
    baseline = _assemble()

    altered = _assemble(blob=_blob(record=_record("file:///corpus/other.pdf")))

    assert altered.document_id != baseline.document_id


def test_identifier_is_independent_of_the_ingestion_instant() -> None:
    """Identity must depend on origin and payload alone."""
    early = _assemble(_normalizer(clock=_FixedClock(datetime(2026, 1, 1, tzinfo=UTC))))
    late = _assemble(_normalizer(clock=_FixedClock(datetime(2026, 12, 1, tzinfo=UTC))))

    assert early.document_id == late.document_id
    assert early.ingested_at != late.ingested_at


def test_identifier_is_independent_of_the_document_text() -> None:
    """Identity derives from the payload digest, not from extracted text."""
    baseline = _assemble(text="Heading\n\nBody paragraph.")

    other = _assemble(text="Different\n\nText entirely.")

    assert other.document_id == baseline.document_id


def test_missing_payload_digest_is_rejected() -> None:
    with pytest.raises(NormalizationError, match="no content digest"):
        _assemble(blob=_blob(content_hash="   "))


def test_identity_errors_are_attributed_to_normalization() -> None:
    with pytest.raises(NormalizationError) as excinfo:
        _assemble(blob=_blob(content_hash=" "))

    error = excinfo.value
    assert error.stage is IngestionStage.NORMALIZATION
    assert error.source_uri == SOURCE_URI
    assert error.retryable is False


def test_normalizer_accepts_no_id_generator() -> None:
    """Omitting it removes the opportunity to break idempotency."""
    parameters = inspect.signature(RawDocumentNormalizer.__init__).parameters

    assert "id_generator" not in parameters
    assert "IdGenerator" not in inspect.getsource(normalization_module)


# ---------------------------------------------------------------------------
# Digests
# ---------------------------------------------------------------------------


def test_payload_digest_is_recorded_verbatim() -> None:
    document = _assemble()

    assert document.content_hash == CONTENT_HASH


def test_text_digest_is_computed_from_the_normalized_text() -> None:
    text = "Heading\n\nBody paragraph."
    provider = Blake2bHashProvider()

    document = _assemble(_normalizer(hash_provider=provider), text=text)

    assert document.text_hash == provider.hash_text(text)


def test_text_digest_distinguishes_the_two_digests() -> None:
    document = _assemble()

    assert document.text_hash != document.content_hash


def test_identical_text_from_differing_payloads_shares_a_text_digest() -> None:
    """The point of the text digest: detect re-encoding that changed nothing."""
    first = _assemble(blob=_blob(content_hash="1111"))
    second = _assemble(blob=_blob(content_hash="2222"))

    assert first.text_hash == second.text_hash
    assert first.content_hash != second.content_hash


def test_mismatched_digest_algorithms_are_rejected() -> None:
    """Two digests on one document under different algorithms are incoherent."""
    with pytest.raises(NormalizationError) as excinfo:
        _assemble(blob=_blob(algorithm="sha256"))

    error = excinfo.value
    assert error.context["payload_algorithm"] == "sha256"
    assert error.context["text_algorithm"] == "blake2b-256"


def test_an_alternate_provider_is_honoured_when_consistent() -> None:
    provider = _AlternateHashProvider()

    document = _assemble(
        _normalizer(hash_provider=provider),
        blob=_blob(algorithm="sha256-fake"),
        text="abc",
        structure=_structure("abc"),
        metadata=_metadata("abc"),
    )

    assert document.text_hash == provider.hash_text("abc")


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------


def test_provenance_is_assembled_from_the_record_and_detection() -> None:
    document = _assemble()

    source = document.source
    assert source.source_uri == SOURCE_URI
    assert source.scheme == "file"
    assert source.connector_name == "LocalFilesystemConnector"
    assert source.detected_media_type == "application/pdf"
    assert source.version_tag == "etag-7"


def test_recorded_size_is_the_transferred_size_not_the_declared_size() -> None:
    """A declared size is an assertion; provenance must be truthful."""
    blob = _blob(record=_record(size_bytes=999_999), size_bytes=2048)

    document = _assemble(blob=blob)

    assert document.source.size_bytes == 2048


def test_acquisition_instant_comes_from_the_transfer() -> None:
    document = _assemble()

    assert document.source.acquired_at == ACQUIRED_AT


def test_ingestion_instant_comes_from_the_injected_clock() -> None:
    document = _assemble()

    assert document.ingested_at == INGESTED_AT
    assert document.ingested_at != document.source.acquired_at


def test_declared_media_type_is_retained_even_when_it_disagrees() -> None:
    blob = _blob(record=_record(declared_media_type="text/plain"))

    document = _assemble(blob=blob, detected=_detected("application/pdf"))

    assert document.source.declared_media_type == "text/plain"
    assert document.source.detected_media_type == "application/pdf"


def test_encoding_is_recorded_for_text_payloads() -> None:
    document = _assemble(detected=_detected("text/plain", encoding="utf-8-sig"))

    assert document.source.encoding == "utf-8-sig"


def test_encoding_is_absent_for_binary_payloads() -> None:
    document = _assemble(detected=_detected("application/pdf", encoding=None))

    assert document.source.encoding is None


def test_absent_optional_provenance_is_preserved_as_none() -> None:
    blob = _blob(record=_record(declared_media_type=None, version_tag=None))

    document = _assemble(blob=blob)

    assert document.source.declared_media_type is None
    assert document.source.version_tag is None


# ---------------------------------------------------------------------------
# Warning merging
# ---------------------------------------------------------------------------


def test_warnings_from_every_stage_are_merged_in_pipeline_order() -> None:
    text = "Heading\n\nBody paragraph."
    detection = _warning(ExtractionWarningCode.MEDIA_TYPE_MISMATCH, "detection")
    extraction = _warning(ExtractionWarningCode.PARTIAL_EXTRACTION, "extraction")
    structural = _warning(ExtractionWarningCode.STRUCTURE_INFERRED, "structure")
    meta = _warning(ExtractionWarningCode.METADATA_UNAVAILABLE, "metadata")
    dedup = _warning(ExtractionWarningCode.NEAR_DUPLICATE_DETECTED, "dedup")

    document = _assemble(
        detected=_detected(warnings=(detection,)),
        structure=_structure(text, warnings=(structural,)),
        metadata=_metadata(text, warnings=(meta,)),
        extraction_warnings=(extraction,),
        deduplication_warning=dedup,
    )

    assert [w.message for w in document.warnings] == [
        "detection",
        "extraction",
        "structure",
        "metadata",
        "dedup",
    ]


def test_no_warnings_yields_an_empty_tuple() -> None:
    assert _assemble().warnings == ()


def test_absent_deduplication_warning_contributes_nothing() -> None:
    document = _assemble(deduplication_warning=None)

    assert document.warnings == ()


def test_repeated_warnings_are_not_collapsed() -> None:
    """A repeated observation understates a fidelity problem if collapsed."""
    repeated = [
        _warning(ExtractionWarningCode.EMPTY_PAGE, "empty page") for _ in range(4)
    ]

    document = _assemble(extraction_warnings=repeated)

    assert len(document.warnings) == 4


def test_warnings_are_filterable_by_code() -> None:
    document = _assemble(
        extraction_warnings=(
            _warning(ExtractionWarningCode.EMPTY_PAGE, "page 3"),
            _warning(ExtractionWarningCode.OCR_APPLIED, "page 4"),
            _warning(ExtractionWarningCode.EMPTY_PAGE, "page 7"),
        )
    )

    selected = document.warnings_by_code(ExtractionWarningCode.EMPTY_PAGE)

    assert [w.message for w in selected] == ["page 3", "page 7"]


def test_warning_merge_order_is_deterministic() -> None:
    text = "Heading\n\nBody paragraph."
    kwargs: dict[str, Any] = {
        "detected": _detected(
            warnings=(_warning(ExtractionWarningCode.OTHER, "d"),)
        ),
        "structure": _structure(
            text, warnings=(_warning(ExtractionWarningCode.OTHER, "s"),)
        ),
        "metadata": _metadata(
            text, warnings=(_warning(ExtractionWarningCode.OTHER, "m"),)
        ),
        "extraction_warnings": (_warning(ExtractionWarningCode.OTHER, "e"),),
    }

    first = _assemble(**kwargs)
    second = _assemble(**kwargs)

    assert first.warnings == second.warnings


# ---------------------------------------------------------------------------
# Warning truncation
# ---------------------------------------------------------------------------


def test_excessive_warnings_are_truncated_with_a_notice() -> None:
    warnings = [
        _warning(ExtractionWarningCode.EMPTY_PAGE, f"page {n}") for n in range(50)
    ]

    document = _assemble(
        _normalizer(max_warnings=10), extraction_warnings=warnings
    )

    assert len(document.warnings) == 10
    notice = document.warnings[-1]
    assert notice.code is ExtractionWarningCode.OTHER
    assert notice.detail["total_warnings"] == 50
    assert notice.detail["retained_warnings"] == 10


def test_truncation_retains_the_earliest_warnings() -> None:
    warnings = [
        _warning(ExtractionWarningCode.EMPTY_PAGE, f"page {n}") for n in range(20)
    ]

    document = _assemble(
        _normalizer(max_warnings=5), extraction_warnings=warnings
    )

    assert [w.message for w in document.warnings[:4]] == [
        "page 0",
        "page 1",
        "page 2",
        "page 3",
    ]


def test_a_reader_is_never_shown_a_silently_partial_list() -> None:
    warnings = [
        _warning(ExtractionWarningCode.EMPTY_PAGE, f"page {n}") for n in range(6)
    ]

    document = _assemble(
        _normalizer(max_warnings=3), extraction_warnings=warnings
    )

    assert "truncated" in document.warnings[-1].message


def test_a_count_at_the_limit_is_not_truncated() -> None:
    warnings = [
        _warning(ExtractionWarningCode.EMPTY_PAGE, f"page {n}") for n in range(5)
    ]

    document = _assemble(
        _normalizer(max_warnings=5), extraction_warnings=warnings
    )

    assert len(document.warnings) == 5
    assert document.warnings[-1].message == "page 4"


def test_invalid_warning_limit_is_rejected_at_construction() -> None:
    with pytest.raises(NormalizationError, match="at least 1"):
        RawDocumentNormalizer(Blake2bHashProvider(), _FixedClock(), max_warnings=0)


# ---------------------------------------------------------------------------
# Input verification
# ---------------------------------------------------------------------------


def test_metadata_from_different_text_is_rejected() -> None:
    """A mismatch is a wiring defect, not a schema violation."""
    text = "Heading\n\nBody paragraph."

    with pytest.raises(NormalizationError) as excinfo:
        _assemble(
            text=text,
            structure=_structure(text),
            metadata=_metadata(text, char_count=999),
        )

    error = excinfo.value
    assert error.context["metadata_char_count"] == 999
    assert error.context["text_length"] == len(text)
    assert "different text" in error.message


def test_naive_clock_is_rejected() -> None:
    with pytest.raises(NormalizationError, match="naive instant"):
        _assemble(_normalizer(clock=_NaiveClock()))


def test_every_assembled_timestamp_is_timezone_aware() -> None:
    document = _assemble()

    assert document.ingested_at.tzinfo is not None
    assert document.source.acquired_at.tzinfo is not None


# ---------------------------------------------------------------------------
# Final validation gate
# ---------------------------------------------------------------------------


def test_span_exceeding_the_text_is_rejected() -> None:
    structure = NormalizedStructure(
        text="short",
        blocks=(
            DocumentBlock(
                block_id="0",
                block_type=BlockType.PARAGRAPH,
                span=TextSpan(start=0, end=500),
                depth=0,
                ordinal=0,
            ),
        ),
    )

    with pytest.raises(SchemaValidationError) as excinfo:
        _assemble(structure=structure, metadata=_metadata("short"))

    error = excinfo.value
    assert error.stage is IngestionStage.NORMALIZATION
    assert error.validation_errors
    assert error.context["char_count"] == 5


def test_uncovered_content_between_blocks_is_rejected() -> None:
    """The Evidence-First guard: no text may exist outside the block tree."""
    text = "one HIDDEN two"
    structure = NormalizedStructure(
        text=text,
        blocks=(
            DocumentBlock(
                block_id="0",
                block_type=BlockType.PARAGRAPH,
                span=TextSpan(start=0, end=3),
                depth=0,
                ordinal=0,
            ),
            DocumentBlock(
                block_id="1",
                block_type=BlockType.PARAGRAPH,
                span=TextSpan(start=11, end=14),
                depth=0,
                ordinal=1,
            ),
        ),
    )

    with pytest.raises(SchemaValidationError):
        _assemble(structure=structure, metadata=_metadata(text))


def test_carriage_returns_in_the_text_are_rejected() -> None:
    text = "line one\r\nline two"
    structure = NormalizedStructure(text=text, blocks=())

    with pytest.raises(SchemaValidationError):
        _assemble(structure=structure, metadata=_metadata(text))


def test_duplicate_block_identifiers_are_rejected() -> None:
    text = "ab"
    structure = NormalizedStructure(
        text=text,
        blocks=(
            DocumentBlock(
                block_id="dup",
                block_type=BlockType.PARAGRAPH,
                span=TextSpan(start=0, end=1),
                depth=0,
                ordinal=0,
            ),
            DocumentBlock(
                block_id="dup",
                block_type=BlockType.PARAGRAPH,
                span=TextSpan(start=1, end=2),
                depth=0,
                ordinal=1,
            ),
        ),
    )

    with pytest.raises(SchemaValidationError):
        _assemble(structure=structure, metadata=_metadata(text))


def test_validation_failure_reports_the_derived_identifier() -> None:
    structure = NormalizedStructure(
        text="short",
        blocks=(
            DocumentBlock(
                block_id="0",
                block_type=BlockType.PARAGRAPH,
                span=TextSpan(start=0, end=500),
                depth=0,
                ordinal=0,
            ),
        ),
    )

    with pytest.raises(SchemaValidationError) as excinfo:
        _assemble(structure=structure, metadata=_metadata("short"))

    assert excinfo.value.document_id == derive_document_id(
        SOURCE_URI, CONTENT_HASH
    )


def test_validation_errors_are_plain_mappings() -> None:
    """No Pydantic type may escape into a caller's except clause."""
    structure = NormalizedStructure(
        text="short",
        blocks=(
            DocumentBlock(
                block_id="0",
                block_type=BlockType.PARAGRAPH,
                span=TextSpan(start=0, end=500),
                depth=0,
                ordinal=0,
            ),
        ),
    )

    with pytest.raises(SchemaValidationError) as excinfo:
        _assemble(structure=structure, metadata=_metadata("short"))

    for entry in excinfo.value.validation_errors:
        assert isinstance(entry, dict)


def test_validation_failure_preserves_its_cause() -> None:
    structure = NormalizedStructure(text="a\rb", blocks=())

    with pytest.raises(SchemaValidationError) as excinfo:
        _assemble(structure=structure, metadata=_metadata("a\rb"))

    assert excinfo.value.__cause__ is not None


# ---------------------------------------------------------------------------
# Round trip
# ---------------------------------------------------------------------------


def test_assembled_document_round_trips_through_json() -> None:
    document = _assemble(
        extraction_warnings=(
            _warning(ExtractionWarningCode.OCR_APPLIED, "page 2"),
        )
    )

    restored = RawDocument.model_validate_json(document.model_dump_json())

    assert restored == document


def test_assembled_document_is_immutable() -> None:
    from pydantic import ValidationError as PydanticValidationError

    document = _assemble()

    with pytest.raises(PydanticValidationError):
        document.text = "mutated"  # type: ignore[misc]


def test_leaf_spans_recover_their_text_after_assembly() -> None:
    document = _assemble(text="Heading\n\nBody paragraph.")

    recovered = [document.block_text(b) for b in document.leaf_blocks()]

    assert recovered == ["Heading", "Body paragraph."]


# ---------------------------------------------------------------------------
# Anti-pattern guards
# ---------------------------------------------------------------------------


def test_stage_is_synchronous_and_performs_no_io() -> None:
    source = inspect.getsource(normalization_module)
    for forbidden in ("async def", "await ", "open(", "requests", "httpx"):
        assert forbidden not in source


def test_stage_never_reads_the_wall_clock_directly() -> None:
    source = inspect.getsource(normalization_module)

    assert "self._clock.now()" in source
    assert "datetime.now(" not in source
    assert "utcnow" not in source


def test_stage_never_catches_base_exception() -> None:
    source = inspect.getsource(normalization_module)

    assert "except BaseException" not in source
    assert "except:" not in source


def test_stage_declares_no_verification_thresholds() -> None:
    source = inspect.getsource(RawDocumentNormalizer)

    assert "self._max_warnings" in source
    assert "> 0.8" not in source
    assert "confidence" not in source.lower()


def test_stage_does_not_duplicate_upstream_size_limits() -> None:
    """Payload size limits belong to acquisition, text limits to structure."""
    source = inspect.getsource(normalization_module)

    assert "max_document_bytes" not in source
    assert "max_text_chars" not in source
    assert "max_blocks" not in source


def test_normalizer_holds_no_assignable_state() -> None:
    with pytest.raises(AttributeError):
        _normalizer().cache = {}  # type: ignore[attr-defined]


def test_module_declares_no_mutable_state() -> None:
    mutable = [
        name
        for name, value in vars(normalization_module).items()
        if not name.startswith("__") and isinstance(value, (list, dict, set))
    ]
    assert mutable == []


def test_stage_imports_no_concrete_implementations() -> None:
    source = inspect.getsource(normalization_module)
    for forbidden in (
        "from ptfm.core.ingestion.extractors",
        "from ptfm.core.ingestion.connectors",
        "from ptfm.core.ingestion.registry",
        "from ptfm.plugins",
    ):
        assert forbidden not in source


def test_stage_contains_no_domain_vocabulary() -> None:
    source = inspect.getsource(normalization_module).lower()
    for term in (
        "jurisdiction",
        "statute",
        "ticker",
        "icd",
        "patient",
        "court",
        "diagnosis",
    ):
        assert term not in source


def test_repr_names_the_configured_algorithm() -> None:
    assert "blake2b-256" in repr(_normalizer())
