"""
Unit tests for the PDF extractor.

File:
    tests/core/ingestion/extractors/test_pdf.py

Notes
-----
    Fixtures are authored in memory rather than committed, so that no binary
    artefact and no licensing question enters the repository, and so that the
    layout each test depends on is visible in the test itself.
"""

from __future__ import annotations

import inspect

import pytest

from ptfm.core.ingestion.exceptions import (
    CorruptDocumentError,
    EncryptedDocumentError,
    ExtractorUnavailableError,
)
from ptfm.core.ingestion.extractors import pdf as pdf_module
from ptfm.core.ingestion.extractors.pdf import PdfExtractor
from ptfm.core.ingestion.interfaces import ContentExtractor
from ptfm.core.ingestion.models import (
    ExtractedBlock,
    ExtractionInput,
    ExtractionOutput,
)
from ptfm.core.schemas import BlockType, ExtractionWarningCode

pytest.importorskip("pypdfium2", reason="requires the ptfm[pdf] extra")
reportlab = pytest.importorskip(
    "reportlab", reason="requires reportlab to author PDF fixtures"
)

from reportlab.lib.pagesizes import letter  # noqa: E402
from reportlab.pdfgen import canvas  # noqa: E402


# ---------------------------------------------------------------------------
# Fixture authoring
# ---------------------------------------------------------------------------


def _build_pdf(pages: list[list[tuple[float, float, str]]]) -> bytes:
    """
    Author a PDF from explicit text positions.

    Positions are given rather than flowed, so that each test states exactly
    the layout whose interpretation it is checking.

    Parameters
    ----------
    pages:
        One list per page, each holding ``(x, y, text)`` placements.

    Returns
    -------
    bytes
        The packaged document.
    """
    import io

    buffer = io.BytesIO()
    document = canvas.Canvas(buffer, pagesize=letter)
    document.setFont("Helvetica", 12)

    for placements in pages:
        for x, y, text in placements:
            document.drawString(x, y, text)
        document.showPage()

    document.save()
    return buffer.getvalue()


def _lines(*texts: str, leading: float = 16.0, start: float = 700.0) -> list[tuple[float, float, str]]:
    """Place consecutive lines at uniform leading."""
    return [(72.0, start - index * leading, text) for index, text in enumerate(texts)]


def _payload(data: bytes) -> ExtractionInput:
    return ExtractionInput(
        source_uri="file:///corpus/report.pdf",
        media_type="application/pdf",
        data=data,
        encoding=None,
        filename="report.pdf",
    )


def _extract(data: bytes, **options: bool) -> ExtractionOutput:
    return PdfExtractor(**options).extract(_payload(data))


def _of_type(
    output: ExtractionOutput, block_type: BlockType
) -> list[ExtractedBlock]:
    return [b for b in output.blocks if b.block_type is block_type]


def _texts(output: ExtractionOutput) -> list[str]:
    return [
        block.text
        for _, block in output.iter_leaf_blocks()
        if block.text and block.text.strip()
    ]


# ---------------------------------------------------------------------------
# Contract conformance
# ---------------------------------------------------------------------------


def test_extractor_satisfies_the_content_extractor_contract() -> None:
    extractor = PdfExtractor()

    assert isinstance(extractor, ContentExtractor)
    assert extractor.name == "PdfExtractor"
    assert extractor.supported_media_types == frozenset({"application/pdf"})


def test_extract_is_synchronous() -> None:
    assert not inspect.iscoroutinefunction(PdfExtractor.extract)
    assert not inspect.isasyncgenfunction(PdfExtractor.extract)


def test_availability_passes_when_the_parser_is_installed() -> None:
    assert PdfExtractor.check_availability() is None


def test_missing_parser_names_the_install_extra(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_import = __import__

    def _blocked(name: str, *args: object, **kwargs: object) -> object:
        if name == "pypdfium2":
            raise ImportError("no module named pypdfium2")
        return real_import(name, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr("builtins.__import__", _blocked)

    with pytest.raises(ExtractorUnavailableError) as excinfo:
        PdfExtractor.check_availability()

    assert excinfo.value.install_extra == "ptfm[pdf]"
    assert excinfo.value.missing_dependency == "pypdfium2"


def test_a_permissively_licensed_parser_is_used() -> None:
    """PyMuPDF is faster but AGPL, which is a hazard for enterprise
    deployment."""
    source = inspect.getsource(pdf_module)

    assert "pypdfium2" in source
    assert "fitz" not in source
    assert "PyMuPDF" not in source.replace("``PyMuPDF``", "")


# ---------------------------------------------------------------------------
# What is claimed
# ---------------------------------------------------------------------------


def test_pages_become_section_containers() -> None:
    data = _build_pdf([_lines("Page one text."), _lines("Page two text.")])

    output = _extract(data)
    pages = _of_type(output, BlockType.SECTION)

    assert len(pages) == 2
    assert [p.attributes.page_number for p in pages] == [1, 2]


def test_text_is_recovered_in_reading_order() -> None:
    data = _build_pdf([_lines("First line.", "Second line.", "Third line.")])

    output = _extract(data)

    assert "First line." in " ".join(_texts(output))
    assert " ".join(_texts(output)).index("First") < " ".join(_texts(output)).index(
        "Third"
    )


def test_blocks_carry_their_page_number() -> None:
    data = _build_pdf([_lines("Alpha."), _lines("Beta.")])

    output = _extract(data)
    paragraphs = _of_type(output, BlockType.PARAGRAPH)

    assert {p.attributes.page_number for p in paragraphs} == {1, 2}


def test_page_content_is_parented_to_its_page() -> None:
    data = _build_pdf([_lines("Alpha."), _lines("Beta.")])

    output = _extract(data)
    pages = _of_type(output, BlockType.SECTION)
    page_indices = {output.blocks.index(page) for page in pages}

    for block in _of_type(output, BlockType.PARAGRAPH):
        assert block.parent_index in page_indices


# ---------------------------------------------------------------------------
# What is deliberately not claimed
# ---------------------------------------------------------------------------


def test_no_heading_block_is_ever_emitted() -> None:
    """The format declares no headings; one invented from appearance would
    assert structure the document never made."""
    import io

    buffer = io.BytesIO()
    document = canvas.Canvas(buffer, pagesize=letter)
    document.setFont("Helvetica-Bold", 28)
    document.drawString(72, 700, "LOOKS EXACTLY LIKE A HEADING")
    document.setFont("Helvetica", 11)
    document.drawString(72, 660, "Ordinary body text follows.")
    document.showPage()
    document.save()

    output = _extract(buffer.getvalue())

    assert _of_type(output, BlockType.HEADING) == []


def test_no_table_block_is_ever_emitted() -> None:
    """Aligned text is sometimes a table and sometimes columns of prose;
    a reconstructed grid would cite values incorrectly."""
    data = _build_pdf(
        [
            [
                (72.0, 700.0, "Region"),
                (250.0, 700.0, "Revenue"),
                (72.0, 684.0, "North"),
                (250.0, 684.0, "120"),
            ]
        ]
    )

    output = _extract(data)

    assert _of_type(output, BlockType.TABLE) == []
    assert _of_type(output, BlockType.TABLE_CELL) == []


def test_no_list_block_is_ever_emitted() -> None:
    data = _build_pdf([_lines("• first item", "• second item")])

    output = _extract(data)

    assert _of_type(output, BlockType.LIST) == []
    assert _of_type(output, BlockType.LIST_ITEM) == []


def test_only_pages_and_paragraphs_are_produced() -> None:
    data = _build_pdf([_lines("Alpha.", "Beta.")])

    output = _extract(data)

    assert {b.block_type for b in output.blocks} <= {
        BlockType.SECTION,
        BlockType.PARAGRAPH,
    }


# ---------------------------------------------------------------------------
# Line and word assembly
# ---------------------------------------------------------------------------


def test_words_separated_only_by_position_are_spaced() -> None:
    """Many PDFs position words by coordinate without emitting spaces."""
    data = _build_pdf(
        [[(72.0, 700.0, "Widely"), (140.0, 700.0, "separated"), (240.0, 700.0, "words")]]
    )

    output = _extract(data)
    text = " ".join(_texts(output))

    assert "Widely" in text and "separated" in text and "words" in text
    assert "Widelyseparated" not in text


def test_characters_sharing_a_baseline_form_one_line() -> None:
    data = _build_pdf([[(72.0, 700.0, "Left part"), (200.0, 700.0, "right part")]])

    output = _extract(data)

    assert len(_texts(output)) == 1


def test_layout_whitespace_does_not_become_content() -> None:
    data = _build_pdf([[(72.0, 700.0, "Word"), (400.0, 700.0, "Far")]])

    output = _extract(data)

    assert "   " not in " ".join(_texts(output))


# ---------------------------------------------------------------------------
# Paragraph grouping
# ---------------------------------------------------------------------------


def test_uniformly_spaced_lines_group_into_one_paragraph() -> None:
    data = _build_pdf([_lines("Line one.", "Line two.", "Line three.", "Line four.")])

    output = _extract(data)

    assert len(_of_type(output, BlockType.PARAGRAPH)) == 1


def test_a_wide_vertical_gap_starts_a_new_paragraph() -> None:
    placements = _lines("First para line one.", "First para line two.", "First para line three.")
    placements += [
        (72.0, 700.0 - 3 * 16.0 - 45.0, "Second paragraph line one."),
        (72.0, 700.0 - 4 * 16.0 - 45.0, "Second paragraph line two."),
    ]

    output = _extract(_build_pdf([placements]))

    assert len(_of_type(output, BlockType.PARAGRAPH)) == 2


def test_grouping_adapts_to_the_document_leading() -> None:
    """A fixed measurement would suit one typesetting and not another."""
    tight = _build_pdf([_lines("a one", "a two", "a three", "a four", leading=10.0)])
    loose = _build_pdf([_lines("b one", "b two", "b three", "b four", leading=28.0)])

    assert len(_of_type(_extract(tight), BlockType.PARAGRAPH)) == 1
    assert len(_of_type(_extract(loose), BlockType.PARAGRAPH)) == 1


def test_grouping_can_be_disabled() -> None:
    """The conservative choice for multi-column layouts."""
    data = _build_pdf([_lines("Line one.", "Line two.", "Line three.")])

    output = _extract(data, group_paragraphs=False)

    assert len(_of_type(output, BlockType.PARAGRAPH)) == 3


def test_paragraph_inference_is_stated_in_a_warning() -> None:
    """A downstream reader must know the boundaries were derived, not
    declared."""
    placements = _lines("one", "two", "three")
    placements += [(72.0, 600.0, "four"), (72.0, 584.0, "five")]

    output = _extract(_build_pdf([placements]))

    codes = [w.code for w in output.warnings]
    assert ExtractionWarningCode.STRUCTURE_INFERRED in codes


def test_the_inference_warning_is_recorded_once() -> None:
    pages = [
        _lines("a", "b", "c") + [(72.0, 600.0, "d")],
        _lines("e", "f", "g") + [(72.0, 600.0, "h")],
    ]

    output = _extract(_build_pdf(pages))
    inferred = [
        w
        for w in output.warnings
        if w.code is ExtractionWarningCode.STRUCTURE_INFERRED
    ]

    assert len(inferred) == 1


def test_a_single_line_page_produces_one_paragraph() -> None:
    output = _extract(_build_pdf([[(72.0, 700.0, "Solitary line.")]]))

    assert len(_of_type(output, BlockType.PARAGRAPH)) == 1


# ---------------------------------------------------------------------------
# Empty pages
# ---------------------------------------------------------------------------


def test_an_empty_page_is_retained_with_an_empty_leaf() -> None:
    """Page numbering must stay correct, and a childless container could not
    be assigned a span."""
    data = _build_pdf([_lines("Content."), [], _lines("More content.")])

    output = _extract(data)
    pages = _of_type(output, BlockType.SECTION)

    assert len(pages) == 3
    assert [p.attributes.page_number for p in pages] == [1, 2, 3]

    blank_index = output.blocks.index(pages[1])
    children = [b for b in output.blocks if b.parent_index == blank_index]
    assert [c.text for c in children] == [""]


def test_empty_pages_are_reported_with_their_numbers() -> None:
    data = _build_pdf([_lines("Content."), [], []])

    output = _extract(data)
    empty = [w for w in output.warnings if w.code is ExtractionWarningCode.EMPTY_PAGE]

    assert len(empty) == 1
    assert empty[0].detail["empty_page_count"] == 2
    assert empty[0].detail["empty_pages"] == [2, 3]


def test_a_wholly_empty_document_is_identified_as_scanned() -> None:
    data = _build_pdf([[], []])

    output = _extract(data)
    empty = [w for w in output.warnings if w.code is ExtractionWarningCode.EMPTY_PAGE]

    assert empty[0].detail["wholly_empty"] is True
    assert "optical character recognition" in empty[0].message


def test_pages_with_content_produce_no_empty_page_warning() -> None:
    output = _extract(_build_pdf([_lines("Content.")]))

    assert not [
        w for w in output.warnings if w.code is ExtractionWarningCode.EMPTY_PAGE
    ]


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------


def test_page_count_is_recorded() -> None:
    data = _build_pdf([_lines("one"), _lines("two"), _lines("three")])

    assert _extract(data).metadata.page_count == 3


def test_information_dictionary_is_harvested() -> None:
    import io

    buffer = io.BytesIO()
    document = canvas.Canvas(buffer, pagesize=letter)
    document.setTitle("Quarterly Report")
    document.setAuthor("Ada Lovelace")
    document.drawString(72, 700, "Body.")
    document.showPage()
    document.save()

    metadata = _extract(buffer.getvalue()).metadata

    assert metadata.title == "Quarterly Report"
    assert metadata.authors == ("Ada Lovelace",)


def test_dates_are_retained_verbatim_not_parsed() -> None:
    """PDF date strings frequently lack or misstate a timezone; a wrong
    instant is worse than none."""
    metadata = _extract(_build_pdf([_lines("Body.")])).metadata

    assert metadata.created_at is None
    assert metadata.modified_at is None


def test_a_document_without_information_still_extracts() -> None:
    output = _extract(_build_pdf([_lines("Body.")]))

    assert _texts(output)
    assert output.metadata.page_count == 1


# ---------------------------------------------------------------------------
# Malformed and protected input
# ---------------------------------------------------------------------------


def test_a_non_pdf_payload_is_reported_as_corrupt() -> None:
    with pytest.raises(CorruptDocumentError) as excinfo:
        _extract(b"this is not a pdf at all")

    assert excinfo.value.retryable is False


def test_a_truncated_payload_is_reported_as_corrupt() -> None:
    data = _build_pdf([_lines("Body.")])

    with pytest.raises(CorruptDocumentError):
        _extract(data[: len(data) // 3])


def test_an_encrypted_document_is_distinguished_from_a_corrupt_one() -> None:
    """The remedies differ entirely: a password, or a replacement file."""

    class _PasswordError(Exception):
        pass

    error = _PasswordError("password required to open document")

    classified = pdf_module._classify_parser_error(
        error, "file:///a.pdf", "PdfExtractor"
    )

    assert isinstance(classified, EncryptedDocumentError)


def test_an_unrecognized_parser_error_is_reported_as_corrupt() -> None:
    classified = pdf_module._classify_parser_error(
        RuntimeError("xref table damaged"), "file:///a.pdf", "PdfExtractor"
    )

    assert isinstance(classified, CorruptDocumentError)


# ---------------------------------------------------------------------------
# Structural validity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "pages",
    [
        [_lines("single line")],
        [_lines("a", "b", "c", "d")],
        [_lines("one"), [], _lines("three")],
        [[], []],
        [[(72.0, 700.0, "col a"), (300.0, 700.0, "col b")]],
    ],
)
def test_output_is_always_structurally_valid(pages: list) -> None:
    """Constructing ExtractionOutput enforces pre-order, parentage, and
    container semantics."""
    assert isinstance(_extract(_build_pdf(pages)), ExtractionOutput)


def test_parent_indices_always_precede_their_children() -> None:
    output = _extract(_build_pdf([_lines("a", "b"), _lines("c", "d")]))

    for index, block in enumerate(output.blocks):
        if block.parent_index is not None:
            assert block.parent_index < index


def test_no_page_container_carries_text() -> None:
    output = _extract(_build_pdf([_lines("a", "b")]))

    for block in _of_type(output, BlockType.SECTION):
        assert block.text is None


# ---------------------------------------------------------------------------
# Determinism and resources
# ---------------------------------------------------------------------------


def test_extraction_is_deterministic() -> None:
    data = _build_pdf([_lines("Alpha.", "Beta.")])
    extractor = PdfExtractor()
    payload = _payload(data)

    assert extractor.extract(payload) == extractor.extract(payload)


def test_native_resources_are_released() -> None:
    """The parser holds handles that garbage collection alone does not
    free."""
    source = inspect.getsource(pdf_module)

    assert "document.close()" in source
    assert "page.close()" in source
    assert "textpage.close()" in source


def test_resources_are_released_even_when_extraction_fails() -> None:
    source = inspect.getsource(PdfExtractor.extract)

    assert "finally:" in source


def test_extractor_holds_no_assignable_state() -> None:
    with pytest.raises(AttributeError):
        PdfExtractor().cache = {}  # type: ignore[attr-defined]


def test_repeated_extraction_does_not_accumulate_warnings() -> None:
    data = _build_pdf([_lines("Content."), []])
    extractor = PdfExtractor()
    payload = _payload(data)

    first = extractor.extract(payload)
    second = extractor.extract(payload)

    assert len(first.warnings) == len(second.warnings)


# ---------------------------------------------------------------------------
# Anti-pattern guards
# ---------------------------------------------------------------------------


def test_extractor_performs_no_io() -> None:
    source = inspect.getsource(pdf_module)
    for forbidden in ("open(", "requests", "httpx", "async def"):
        assert forbidden not in source


def test_no_confidence_value_is_produced() -> None:
    """Any OCR confidence would be a raw model score, prohibited outside
    calibration."""
    source = inspect.getsource(pdf_module).lower()

    assert "confidence" not in source


def test_module_declares_no_mutable_state() -> None:
    mutable = [
        name
        for name, value in vars(pdf_module).items()
        if not name.startswith("__") and isinstance(value, (list, set))
    ]
    assert mutable == []


def test_extractor_contains_no_domain_vocabulary() -> None:
    source = inspect.getsource(pdf_module).lower()
    for term in ("jurisdiction", "statute", "ticker", "icd", "patient"):
        assert term not in source
