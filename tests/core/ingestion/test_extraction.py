"""Unit tests for the extraction stage."""

from __future__ import annotations

import asyncio
import inspect
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Optional

import pytest

from ptfm.core.ingestion.config import ExtractionConfig
from ptfm.core.ingestion.exceptions import (
    CorruptDocumentError,
    EncryptedDocumentError,
    ExtractionTimeoutError,
    ExtractorInternalError,
    IngestionStage,
    UnsupportedMediaTypeError,
)
from ptfm.core.ingestion.models import (
    AcquiredBlob,
    DetectedFormat,
    ExtractedBlock,
    ExtractionInput,
    ExtractionOutput,
    SourceRecord,
)
from ptfm.core.ingestion.pipeline import extraction as extraction_module
from ptfm.core.ingestion.pipeline.extraction import ExtractionStage
from ptfm.core.ingestion.registry import ExtractorRegistry
from ptfm.core.schemas import BlockType, ExtractionWarningCode

FIXED_TIME = datetime(2026, 1, 1, tzinfo=UTC)


# ---------------------------------------------------------------------------
# Doubles
# ---------------------------------------------------------------------------


class _Extractor:
    """Extractor whose behaviour is supplied per test."""

    def __init__(
        self,
        name: str = "TestExtractor",
        media_types: tuple[str, ...] = ("text/plain",),
        behaviour: Optional[Callable[[ExtractionInput], Any]] = None,
    ) -> None:
        self._name = name
        self._media_types = frozenset(media_types)
        self._behaviour = behaviour
        self.calls: list[ExtractionInput] = []

    @property
    def name(self) -> str:
        return self._name

    @property
    def supported_media_types(self) -> frozenset[str]:
        return self._media_types

    @property
    def priority(self) -> int:
        return 0

    @classmethod
    def check_availability(cls) -> None:
        return None

    def supports(self, media_type: str) -> bool:
        return media_type in self._media_types

    def extract(self, payload: ExtractionInput) -> Any:
        self.calls.append(payload)
        if self._behaviour is None:
            return _output(self._name, "extracted text")
        return self._behaviour(payload)


class _RecordingOffloader:
    """Offloader running work inline while recording that it was used."""

    def __init__(self) -> None:
        self.calls = 0

    async def run(self, func: Any, *args: Any, **kwargs: Any) -> Any:
        self.calls += 1
        return func(*args, **kwargs)


class _NeverCompletingOffloader:
    """Offloader that never returns, simulating a hung parser."""

    async def run(self, func: Any, *args: Any, **kwargs: Any) -> Any:
        await asyncio.Event().wait()
        raise AssertionError("unreachable")


def _output(extractor_name: str, text: str) -> ExtractionOutput:
    return ExtractionOutput(
        blocks=(ExtractedBlock(block_type=BlockType.PARAGRAPH, text=text),),
        extractor_name=extractor_name,
    )


def _record(filename: str | None = "a.txt") -> SourceRecord:
    return SourceRecord(
        record_id="r-1",
        source_uri="file:///corpus/a.txt",
        scheme="file",
        connector_name="TestConnector",
        discovered_at=FIXED_TIME,
        filename=filename,
    )


def _blob(data: bytes = b"payload", **overrides: Any) -> AcquiredBlob:
    fields: dict[str, Any] = {
        "record": _record(),
        "size_bytes": len(data),
        "content_hash": "hash-1",
        "hash_algorithm": "blake2b-256",
        "acquired_at": FIXED_TIME,
        "data": data,
    }
    fields.update(overrides)
    return AcquiredBlob(**fields)


def _detected(media_type: str = "text/plain", encoding: str | None = "utf-8") -> DetectedFormat:
    return DetectedFormat(media_type=media_type, encoding=encoding)


def _stage(
    *extractors: _Extractor,
    offloader: Any = None,
    **config: Any,
) -> tuple[ExtractionStage, _RecordingOffloader]:
    resolved = offloader or _RecordingOffloader()
    registry = ExtractorRegistry(extractors, config=ExtractionConfig())
    stage = ExtractionStage(registry, resolved, ExtractionConfig(**config))
    return stage, resolved


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


async def test_payload_is_extracted_via_the_resolved_extractor() -> None:
    extractor = _Extractor()
    stage, _ = _stage(extractor)

    output = await stage.extract(_blob(), _detected())

    assert output.extractor_name == "TestExtractor"
    assert len(extractor.calls) == 1


async def test_extraction_input_carries_the_full_context() -> None:
    extractor = _Extractor()
    stage, _ = _stage(extractor)

    await stage.extract(_blob(b"body bytes"), _detected(encoding="utf-16"))

    payload = extractor.calls[0]
    assert payload.source_uri == "file:///corpus/a.txt"
    assert payload.media_type == "text/plain"
    assert payload.data == b"body bytes"
    assert payload.encoding == "utf-16"
    assert payload.filename == "a.txt"


async def test_missing_filename_is_passed_through_as_none() -> None:
    extractor = _Extractor()
    stage, _ = _stage(extractor)

    await stage.extract(_blob(record=_record(filename=None)), _detected())

    assert extractor.calls[0].filename is None


async def test_specificity_ordering_of_the_registry_is_honoured() -> None:
    exact = _Extractor("Exact", ("text/plain",))
    fallback = _Extractor("Fallback", ("*/*",))
    stage, _ = _stage(exact, fallback)

    assert (await stage.extract(_blob(), _detected("text/plain"))).extractor_name == "Exact"
    assert (await stage.extract(_blob(), _detected("image/png"))).extractor_name == "Fallback"


# ---------------------------------------------------------------------------
# Async safety
# ---------------------------------------------------------------------------


async def test_extractor_is_invoked_through_the_offloader() -> None:
    """A blocking parser must never run on the event loop."""
    stage, offloader = _stage(_Extractor())

    await stage.extract(_blob(), _detected())

    assert offloader.calls == 1


async def test_read_and_parse_share_a_single_offload() -> None:
    """Splitting them would double context switches and invite a loop read."""
    stage, offloader = _stage(_Extractor())

    await stage.extract(_blob(), _detected())

    assert offloader.calls == 1


async def test_spooled_payload_is_read_inside_the_worker_thread(
    tmp_path: Path,
) -> None:
    spool = tmp_path / "payload.bin"
    spool.write_bytes(b"spooled body")
    blob = AcquiredBlob(
        record=_record(),
        size_bytes=12,
        content_hash="hash-1",
        hash_algorithm="blake2b-256",
        acquired_at=FIXED_TIME,
        spool_path=spool,
    )
    extractor = _Extractor()
    stage, offloader = _stage(extractor)

    await stage.extract(blob, _detected())

    assert extractor.calls[0].data == b"spooled body"
    assert offloader.calls == 1


def test_stage_module_declares_no_blocking_calls_outside_the_worker() -> None:
    entry = inspect.getsource(extraction_module._read_and_extract)
    stage_source = inspect.getsource(ExtractionStage)

    assert "blob.read()" in entry
    assert "blob.read()" not in stage_source
    assert ".extract(" not in stage_source.replace("async def extract(", "")


# ---------------------------------------------------------------------------
# Resolution failure
# ---------------------------------------------------------------------------


async def test_unregistered_media_type_raises_with_available_types() -> None:
    stage, _ = _stage(_Extractor("Text", ("text/plain",)))

    with pytest.raises(UnsupportedMediaTypeError) as excinfo:
        await stage.extract(_blob(), _detected("application/pdf"))

    assert excinfo.value.supported_media_types == ("text/plain",)
    assert excinfo.value.stage is IngestionStage.DETECTION


async def test_resolution_failure_never_invokes_the_offloader() -> None:
    stage, offloader = _stage(_Extractor("Text", ("text/plain",)))

    with pytest.raises(UnsupportedMediaTypeError):
        await stage.extract(_blob(), _detected("image/png"))

    assert offloader.calls == 0


# ---------------------------------------------------------------------------
# Deadline
# ---------------------------------------------------------------------------


async def test_deadline_expiry_quarantines_the_document() -> None:
    stage, _ = _stage(
        _Extractor(), offloader=_NeverCompletingOffloader(), timeout_seconds=0.01
    )

    with pytest.raises(ExtractionTimeoutError) as excinfo:
        await stage.extract(_blob(), _detected())

    error = excinfo.value
    assert error.timeout_seconds == 0.01
    assert error.retryable is False
    assert error.source_uri == "file:///corpus/a.txt"


async def test_deadline_error_records_that_the_thread_was_not_reclaimed() -> None:
    """The limitation is surfaced in diagnostics, not concealed."""
    stage, _ = _stage(
        _Extractor(), offloader=_NeverCompletingOffloader(), timeout_seconds=0.01
    )

    with pytest.raises(ExtractionTimeoutError) as excinfo:
        await stage.extract(_blob(), _detected())

    assert excinfo.value.context["thread_reclaimed"] is False
    assert "may still be running" in excinfo.value.message


async def test_deadline_releases_the_pipeline_promptly() -> None:
    stage, _ = _stage(
        _Extractor(), offloader=_NeverCompletingOffloader(), timeout_seconds=0.05
    )

    with pytest.raises(ExtractionTimeoutError):
        await asyncio.wait_for(stage.extract(_blob(), _detected()), timeout=2.0)


async def test_work_completing_within_the_deadline_succeeds() -> None:
    stage, _ = _stage(_Extractor(), timeout_seconds=30.0)

    assert (await stage.extract(_blob(), _detected())).extractor_name == "TestExtractor"


# ---------------------------------------------------------------------------
# Exception conversion
# ---------------------------------------------------------------------------


async def test_ptfm_exception_from_an_extractor_passes_through() -> None:
    """Extractors are obliged to raise PTFM types; those are not rewrapped."""

    def _raise(payload: ExtractionInput) -> Any:
        raise CorruptDocumentError("xref table damaged")

    stage, _ = _stage(_Extractor(behaviour=_raise))

    with pytest.raises(CorruptDocumentError) as excinfo:
        await stage.extract(_blob(), _detected())

    assert not isinstance(excinfo.value, ExtractorInternalError)


async def test_passthrough_exception_is_enriched_with_document_context() -> None:
    def _raise(payload: ExtractionInput) -> Any:
        raise CorruptDocumentError("damaged")

    stage, _ = _stage(_Extractor("PdfExtractor", behaviour=_raise))

    with pytest.raises(CorruptDocumentError) as excinfo:
        await stage.extract(_blob(), _detected())

    error = excinfo.value
    assert error.source_uri == "file:///corpus/a.txt"
    assert error.context["extractor_name"] == "PdfExtractor"


async def test_extractor_supplied_uri_is_not_overwritten() -> None:
    def _raise(payload: ExtractionInput) -> Any:
        raise CorruptDocumentError("damaged", source_uri="file:///inner.pdf")

    stage, _ = _stage(_Extractor(behaviour=_raise))

    with pytest.raises(CorruptDocumentError) as excinfo:
        await stage.extract(_blob(), _detected())

    assert excinfo.value.source_uri == "file:///inner.pdf"


async def test_unanticipated_exception_is_wrapped_with_its_cause() -> None:
    class _VendorFailure(Exception):
        pass

    def _raise(payload: ExtractionInput) -> Any:
        raise _VendorFailure("internal parser state invalid")

    stage, _ = _stage(_Extractor("PdfExtractor", behaviour=_raise))

    with pytest.raises(ExtractorInternalError) as excinfo:
        await stage.extract(_blob(), _detected())

    error = excinfo.value
    assert isinstance(error.__cause__, _VendorFailure)
    assert error.extractor_name == "PdfExtractor"
    assert error.context["original_exception"] == "_VendorFailure"


@pytest.mark.parametrize("exception_type", [MemoryError, RecursionError])
async def test_resource_exhaustion_quarantines_rather_than_aborting(
    exception_type: type[Exception],
) -> None:
    """Per-document isolation must survive a hostile document."""

    def _raise(payload: ExtractionInput) -> Any:
        raise exception_type("exhausted")

    stage, _ = _stage(_Extractor(behaviour=_raise))

    with pytest.raises(ExtractorInternalError):
        await stage.extract(_blob(), _detected())


@pytest.mark.parametrize(
    "base_exception", [asyncio.CancelledError, KeyboardInterrupt, SystemExit]
)
async def test_base_exceptions_are_never_caught(
    base_exception: type[BaseException],
) -> None:
    """Catching these would make a cancelled run un-cancellable."""

    def _raise(payload: ExtractionInput) -> Any:
        raise base_exception()

    stage, _ = _stage(_Extractor(behaviour=_raise))

    with pytest.raises(base_exception):
        await stage.extract(_blob(), _detected())


async def test_cancellation_propagates_from_a_pending_offload() -> None:
    stage, _ = _stage(
        _Extractor(), offloader=_NeverCompletingOffloader(), timeout_seconds=60.0
    )
    task = asyncio.create_task(stage.extract(_blob(), _detected()))
    await asyncio.sleep(0)

    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task


# ---------------------------------------------------------------------------
# Heuristic classification
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name", ["FileEncryptedError", "PasswordError", "PDFPasswordIncorrect"]
)
async def test_known_encryption_exception_names_are_classified(name: str) -> None:
    def _raise(payload: ExtractionInput) -> Any:
        raise type(name, (Exception,), {})("cannot open")

    stage, _ = _stage(_Extractor(behaviour=_raise))

    with pytest.raises(EncryptedDocumentError):
        await stage.extract(_blob(), _detected())


@pytest.mark.parametrize(
    "message",
    [
        "file has not been decrypted",
        "Password required to open",
        "document is encrypted",
    ],
)
async def test_encryption_message_tokens_are_classified(message: str) -> None:
    def _raise(payload: ExtractionInput) -> Any:
        raise ValueError(message)

    stage, _ = _stage(_Extractor(behaviour=_raise))

    with pytest.raises(EncryptedDocumentError) as excinfo:
        await stage.extract(_blob(), _detected())

    assert excinfo.value.retryable is False


@pytest.mark.parametrize("name", ["BadZipFile", "PdfiumError", "XMLSyntaxError"])
async def test_known_corruption_exception_names_are_classified(name: str) -> None:
    def _raise(payload: ExtractionInput) -> Any:
        raise type(name, (Exception,), {})("malformed structure")

    stage, _ = _stage(_Extractor(behaviour=_raise))

    with pytest.raises(CorruptDocumentError):
        await stage.extract(_blob(), _detected())


async def test_unrecognized_exception_degrades_safely() -> None:
    """
    A renamed library exception must still quarantine the document with its
    cause preserved, rather than escaping unclassified.
    """

    def _raise(payload: ExtractionInput) -> Any:
        raise RuntimeError("some future library error name")

    stage, _ = _stage(_Extractor(behaviour=_raise))

    with pytest.raises(ExtractorInternalError) as excinfo:
        await stage.extract(_blob(), _detected())

    assert isinstance(excinfo.value.__cause__, RuntimeError)


async def test_classification_does_not_misread_ordinary_failures() -> None:
    def _raise(payload: ExtractionInput) -> Any:
        raise ValueError("unexpected token at offset 42")

    stage, _ = _stage(_Extractor(behaviour=_raise))

    with pytest.raises(ExtractorInternalError):
        await stage.extract(_blob(), _detected())


# ---------------------------------------------------------------------------
# Contract violations
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("returned", [None, "text", 42, {"blocks": []}])
async def test_wrong_return_type_is_rejected(returned: Any) -> None:
    stage, _ = _stage(_Extractor(behaviour=lambda payload: returned))

    with pytest.raises(ExtractorInternalError, match="not an ExtractionOutput"):
        await stage.extract(_blob(), _detected())


async def test_unreadable_payload_is_reported_against_the_extractor() -> None:
    blob = _blob()
    blob.close()
    stage, _ = _stage(_Extractor())

    with pytest.raises(ExtractorInternalError) as excinfo:
        await stage.extract(blob, _detected())

    assert "could not be read" in excinfo.value.message
    assert isinstance(excinfo.value.__cause__, ValueError)


async def test_missing_spool_file_is_reported(tmp_path: Path) -> None:
    blob = AcquiredBlob(
        record=_record(),
        size_bytes=4,
        content_hash="hash-1",
        hash_algorithm="blake2b-256",
        acquired_at=FIXED_TIME,
        spool_path=tmp_path / "absent.bin",
    )
    stage, _ = _stage(_Extractor())

    with pytest.raises(ExtractorInternalError) as excinfo:
        await stage.extract(blob, _detected())

    assert isinstance(excinfo.value.__cause__, OSError)
    assert excinfo.value.context["spooled"] is True


# ---------------------------------------------------------------------------
# Empty extraction
# ---------------------------------------------------------------------------


async def test_empty_extraction_warns_rather_than_failing() -> None:
    """Bytes present but no text is a fidelity signal, not a fault."""
    empty = ExtractionOutput(
        blocks=(ExtractedBlock(block_type=BlockType.PARAGRAPH, text="   "),),
        extractor_name="PdfExtractor",
    )
    stage, _ = _stage(_Extractor("PdfExtractor", behaviour=lambda payload: empty))

    output = await stage.extract(_blob(), _detected())

    assert len(output.warnings) == 1
    warning = output.warnings[0]
    assert warning.code is ExtractionWarningCode.PARTIAL_EXTRACTION
    assert warning.detail["extractor_name"] == "PdfExtractor"


async def test_empty_warning_preserves_extractor_warnings() -> None:
    from ptfm.core.schemas import ExtractionWarning

    existing = ExtractionWarning(
        code=ExtractionWarningCode.EMPTY_PAGE, message="page 3 empty"
    )
    empty = ExtractionOutput(
        blocks=(ExtractedBlock(block_type=BlockType.PARAGRAPH, text=""),),
        warnings=(existing,),
        extractor_name="PdfExtractor",
    )
    stage, _ = _stage(_Extractor("PdfExtractor", behaviour=lambda payload: empty))

    output = await stage.extract(_blob(), _detected())

    assert output.warnings[0] == existing
    assert len(output.warnings) == 2


async def test_non_empty_extraction_is_returned_unaltered() -> None:
    produced = _output("TestExtractor", "real content")
    stage, _ = _stage(_Extractor(behaviour=lambda payload: produced))

    assert await stage.extract(_blob(), _detected()) is produced


async def test_extraction_with_no_blocks_warns() -> None:
    empty = ExtractionOutput(blocks=(), extractor_name="TestExtractor")
    stage, _ = _stage(_Extractor(behaviour=lambda payload: empty))

    output = await stage.extract(_blob(), _detected())

    assert output.warnings[0].code is ExtractionWarningCode.PARTIAL_EXTRACTION


# ---------------------------------------------------------------------------
# Anti-pattern guards
# ---------------------------------------------------------------------------


def test_stage_declares_no_hardcoded_deadline() -> None:
    source = inspect.getsource(ExtractionStage)

    assert "self._config.timeout_seconds" in source
    assert "timeout=30" not in source
    assert "> 0.8" not in source


def test_stage_does_not_duplicate_structure_stage_limits() -> None:
    """max_blocks and max_text_chars belong to structure normalization."""
    source = inspect.getsource(extraction_module)

    assert "max_blocks" not in source
    assert "max_text_chars" not in source


def test_stage_imports_no_parsing_libraries() -> None:
    source = inspect.getsource(extraction_module)
    for forbidden in ("pypdfium2", "import docx", "selectolax", "openpyxl", "pdfplumber"):
        assert forbidden not in source


def test_stage_never_catches_base_exception() -> None:
    source = inspect.getsource(extraction_module)

    assert "except BaseException" not in source
    assert "except:" not in source


def test_stage_holds_no_assignable_state() -> None:
    stage, _ = _stage(_Extractor())

    with pytest.raises(AttributeError):
        stage.cache = {}  # type: ignore[attr-defined]


def test_module_declares_no_mutable_state() -> None:
    mutable = [
        name
        for name, value in vars(extraction_module).items()
        if not name.startswith("__") and isinstance(value, (list, dict, set))
    ]
    assert mutable == []


def test_stage_imports_no_concrete_implementations() -> None:
    source = inspect.getsource(extraction_module)
    for forbidden in (
        "from ptfm.core.ingestion.extractors",
        "from ptfm.core.ingestion.connectors",
        "from ptfm.plugins",
    ):
        assert forbidden not in source