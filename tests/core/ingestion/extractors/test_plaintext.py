"""
Unit tests for the plain text extractor.

File:
    tests/core/ingestion/extractors/test_plaintext.py
"""

from __future__ import annotations

import codecs
import inspect

import pytest

from ptfm.core.ingestion.config import ExtractionConfig
from ptfm.core.ingestion.exceptions import ExtractorInternalError
from ptfm.core.ingestion.extractors import plaintext as plaintext_module
from ptfm.core.ingestion.extractors.plaintext import PlainTextExtractor
from ptfm.core.ingestion.interfaces import ContentExtractor
from ptfm.core.ingestion.models import ExtractionInput
from ptfm.core.ingestion.registry import ExtractorRegistry
from ptfm.core.schemas import BlockType, ExtractionWarningCode


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def _payload(
    data: bytes | str,
    *,
    encoding: str | None = "utf-8",
    media_type: str = "text/plain",
    filename: str | None = "a.txt",
) -> ExtractionInput:
    if isinstance(data, str):
        data = data.encode(encoding or "utf-8")
    return ExtractionInput(
        source_uri="file:///corpus/a.txt",
        media_type=media_type,
        data=data,
        encoding=encoding,
        filename=filename,
    )


def _texts(output: object) -> list[str]:
    return [block.text for block in output.blocks]  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Contract conformance
# ---------------------------------------------------------------------------


def test_extractor_satisfies_the_content_extractor_contract() -> None:
    extractor = PlainTextExtractor()

    assert isinstance(extractor, ContentExtractor)
    assert extractor.name == "PlainTextExtractor"
    assert extractor.supported_media_types == frozenset({"text/plain"})


def test_extract_is_synchronous() -> None:
    """A blocking parser declared async would conceal work on the event loop."""
    assert not inspect.iscoroutinefunction(PlainTextExtractor.extract)
    assert not inspect.isasyncgenfunction(PlainTextExtractor.extract)


def test_extractor_is_always_available() -> None:
    """No optional dependency, so this extractor can serve as a fallback."""
    assert PlainTextExtractor.check_availability() is None


def test_extractor_registers_without_arguments() -> None:
    registry = ExtractorRegistry([PlainTextExtractor()], config=ExtractionConfig())

    assert registry.resolve("text/plain").name == "PlainTextExtractor"


# ---------------------------------------------------------------------------
# Segmentation
# ---------------------------------------------------------------------------


def test_blank_lines_separate_paragraphs() -> None:
    output = PlainTextExtractor().extract(
        _payload("First paragraph.\n\nSecond paragraph.\n\nThird.")
    )

    assert _texts(output) == [
        "First paragraph.",
        "Second paragraph.",
        "Third.",
    ]


def test_single_newlines_do_not_split_a_paragraph() -> None:
    output = PlainTextExtractor().extract(
        _payload("A wrapped line\ncontinues here.")
    )

    assert _texts(output) == ["A wrapped line\ncontinues here."]


def test_every_block_is_a_paragraph() -> None:
    output = PlainTextExtractor().extract(_payload("One.\n\nTwo."))

    assert all(block.block_type is BlockType.PARAGRAPH for block in output.blocks)
    assert all(block.parent_index is None for block in output.blocks)


@pytest.mark.parametrize(
    "separator", ["\n\n", "\n\n\n\n", "\n   \n", "\n\t\n"]
)
def test_whitespace_only_lines_separate_paragraphs(separator: str) -> None:
    output = PlainTextExtractor().extract(_payload(f"One.{separator}Two."))

    assert _texts(output) == ["One.", "Two."]


@pytest.mark.parametrize("line_ending", ["\r\n", "\r", "\u2028"])
def test_alternative_line_endings_segment_identically(line_ending: str) -> None:
    body = f"First.{line_ending}{line_ending}Second."

    output = PlainTextExtractor().extract(_payload(body))

    assert _texts(output) == ["First.", "Second."]


def test_no_structure_is_inferred_beyond_segmentation() -> None:
    """A leading line is not reliably a title; guessing would be a claim."""
    output = PlainTextExtractor().extract(
        _payload("REPORT TITLE\n\nBody text follows.")
    )

    assert all(block.block_type is BlockType.PARAGRAPH for block in output.blocks)


# ---------------------------------------------------------------------------
# Empty and degenerate payloads
# ---------------------------------------------------------------------------


def test_empty_payload_produces_no_blocks() -> None:
    output = PlainTextExtractor().extract(_payload(b""))

    assert output.blocks == ()
    assert output.is_empty is True


@pytest.mark.parametrize("body", ["   ", "\n\n\n", "\t\n \n"])
def test_whitespace_only_payload_produces_no_blocks(body: str) -> None:
    output = PlainTextExtractor().extract(_payload(body))

    assert output.blocks == ()
    assert output.is_empty is True


def test_leading_and_trailing_blank_lines_produce_no_empty_blocks() -> None:
    output = PlainTextExtractor().extract(_payload("\n\n\nBody.\n\n\n"))

    assert _texts(output) == ["Body."]


# ---------------------------------------------------------------------------
# Decoding
# ---------------------------------------------------------------------------


def test_utf8_content_is_decoded_exactly() -> None:
    output = PlainTextExtractor().extract(_payload("café naïve 日本語"))

    assert _texts(output) == ["café naïve 日本語"]
    assert output.warnings == ()


def test_declared_encoding_is_honoured() -> None:
    output = PlainTextExtractor().extract(
        _payload("café".encode("iso-8859-1"), encoding="iso-8859-1")
    )

    assert _texts(output) == ["café"]


def test_undecodable_bytes_degrade_rather_than_fail() -> None:
    """A few malformed trailing bytes must not discard a large document."""
    data = "valid prose ".encode("utf-8") + b"\xff\xfe"

    output = PlainTextExtractor().extract(_payload(data))

    assert output.blocks
    assert "valid prose" in output.blocks[0].text
    assert output.warnings[0].code is ExtractionWarningCode.ENCODING_REPLACEMENT
    assert output.warnings[0].detail["replacement_count"] >= 1


def test_no_second_replacement_ratio_is_judged() -> None:
    """Detection owns that judgement; a second limit could disagree with it."""
    output = PlainTextExtractor().extract(_payload(b"\xff\xfe\xff\xfe abc"))

    assert output.blocks
    assert len(output.warnings) == 1


def test_leading_byte_order_mark_is_removed() -> None:
    """A surviving mark would displace every block span by one character."""
    data = codecs.BOM_UTF8 + "Title text".encode("utf-8")

    output = PlainTextExtractor().extract(_payload(data, encoding="utf-8"))

    assert _texts(output) == ["Title text"]
    assert "\ufeff" not in output.blocks[0].text


def test_bom_stripping_codec_is_also_handled() -> None:
    data = codecs.BOM_UTF8 + "Title text".encode("utf-8")

    output = PlainTextExtractor().extract(_payload(data, encoding="utf-8-sig"))

    assert _texts(output) == ["Title text"]


def test_absent_encoding_falls_back_to_utf8() -> None:
    output = PlainTextExtractor().extract(_payload("café", encoding=None))

    assert _texts(output) == ["café"]


def test_unrecognized_codec_is_reported_as_an_internal_fault() -> None:
    """An unknown codec is a detection fault, not a malformed document."""
    with pytest.raises(ExtractorInternalError, match="not a recognized codec"):
        PlainTextExtractor().extract(_payload(b"data", encoding="not-a-codec"))


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------


def test_line_count_is_recorded() -> None:
    output = PlainTextExtractor().extract(_payload("one\ntwo\nthree"))

    assert output.metadata.properties["line_count"] == "3"


def test_line_count_of_an_empty_payload_is_zero() -> None:
    output = PlainTextExtractor().extract(_payload(b""))

    assert output.metadata.properties["line_count"] == "0"


def test_no_metadata_is_invented() -> None:
    """Plain text declares no title, author, or page count."""
    metadata = PlainTextExtractor().extract(_payload("body")).metadata

    assert metadata.title is None
    assert metadata.authors == ()
    assert metadata.created_at is None
    assert metadata.page_count is None


def test_extractor_name_is_recorded_on_the_result() -> None:
    output = PlainTextExtractor().extract(_payload("body"))

    assert output.extractor_name == "PlainTextExtractor"


# ---------------------------------------------------------------------------
# Fallback registration
# ---------------------------------------------------------------------------


def test_media_types_and_priority_are_configurable() -> None:
    fallback = PlainTextExtractor(["text/*"], priority=-10)

    assert fallback.supported_media_types == frozenset({"text/*"})
    assert fallback.priority == -10


def test_a_fallback_registration_cannot_displace_a_dedicated_extractor() -> None:
    class _Dedicated(PlainTextExtractor):
        @property
        def name(self) -> str:
            return "DedicatedMarkdown"

        @property
        def supported_media_types(self) -> frozenset[str]:
            return frozenset({"text/markdown"})

    registry = ExtractorRegistry(
        [PlainTextExtractor(["text/*"], priority=-10), _Dedicated()],
        config=ExtractionConfig(),
    )

    assert registry.resolve("text/markdown").name == "DedicatedMarkdown"
    assert registry.resolve("text/csv").name == "PlainTextExtractor"


def test_declared_media_types_are_normalized() -> None:
    extractor = PlainTextExtractor(["Text/Plain; charset=utf-8"])

    assert extractor.supported_media_types == frozenset({"text/plain"})


# ---------------------------------------------------------------------------
# Concurrency and determinism
# ---------------------------------------------------------------------------


def test_extraction_is_deterministic() -> None:
    extractor = PlainTextExtractor()
    payload = _payload("One.\n\nTwo.")

    assert extractor.extract(payload) == extractor.extract(payload)


def test_extractor_holds_no_assignable_state() -> None:
    with pytest.raises(AttributeError):
        PlainTextExtractor().cache = {}  # type: ignore[attr-defined]


def test_repeated_extraction_does_not_accumulate_warnings() -> None:
    extractor = PlainTextExtractor()
    payload = _payload(b"\xff\xfe abc")

    first = extractor.extract(payload)
    second = extractor.extract(payload)

    assert len(first.warnings) == len(second.warnings) == 1


# ---------------------------------------------------------------------------
# Anti-pattern guards
# ---------------------------------------------------------------------------


def test_extractor_performs_no_io() -> None:
    source = inspect.getsource(plaintext_module)
    for forbidden in ("open(", "Path(", "requests", "httpx", "async def"):
        assert forbidden not in source


def test_extractor_declares_no_thresholds() -> None:
    source = inspect.getsource(plaintext_module)

    assert "max_replacement_ratio" not in source
    assert "> 0.8" not in source


def test_module_declares_no_mutable_state() -> None:
    mutable = [
        name
        for name, value in vars(plaintext_module).items()
        if not name.startswith("__") and isinstance(value, (list, dict, set))
    ]
    assert mutable == []


def test_extractor_contains_no_domain_vocabulary() -> None:
    source = inspect.getsource(plaintext_module).lower()
    for term in ("jurisdiction", "statute", "ticker", "icd", "patient", "court"):
        assert term not in source
