"""
Unit tests for the DOCX extractor.

File:
    tests/core/ingestion/extractors/test_docx.py
"""

from __future__ import annotations

import inspect
import io

import pytest

from ptfm.core.ingestion.exceptions import (
    CorruptDocumentError,
    EncryptedDocumentError,
    ExtractorUnavailableError,
)
from ptfm.core.ingestion.extractors import docx as docx_module
from ptfm.core.ingestion.extractors.docx import DocxExtractor
from ptfm.core.ingestion.interfaces import ContentExtractor
from ptfm.core.ingestion.models import (
    ExtractedBlock,
    ExtractionInput,
    ExtractionOutput,
)
from ptfm.core.schemas import BlockType, ExtractionWarningCode

docx = pytest.importorskip("docx", reason="requires the ptfm[office] extra")

DOCX_MEDIA_TYPE = (
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
)


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def _build(populate) -> bytes:
    """Author a document in memory and return its packaged bytes."""
    document = docx.Document()
    populate(document)
    buffer = io.BytesIO()
    document.save(buffer)
    return buffer.getvalue()


def _payload(data: bytes) -> ExtractionInput:
    return ExtractionInput(
        source_uri="file:///corpus/report.docx",
        media_type=DOCX_MEDIA_TYPE,
        data=data,
        encoding=None,
        filename="report.docx",
    )


def _extract(populate) -> ExtractionOutput:
    return DocxExtractor().extract(_payload(_build(populate)))


def _of_type(
    output: ExtractionOutput, block_type: BlockType
) -> list[ExtractedBlock]:
    return [b for b in output.blocks if b.block_type is block_type]


def _leaf_texts(output: ExtractionOutput) -> list[str]:
    return [block.text for _, block in output.iter_leaf_blocks()]


# ---------------------------------------------------------------------------
# Contract conformance
# ---------------------------------------------------------------------------


def test_extractor_satisfies_the_content_extractor_contract() -> None:
    extractor = DocxExtractor()

    assert isinstance(extractor, ContentExtractor)
    assert extractor.name == "DocxExtractor"
    assert DOCX_MEDIA_TYPE in extractor.supported_media_types


def test_templates_are_also_claimed() -> None:
    assert (
        "application/vnd.openxmlformats-officedocument.wordprocessingml.template"
        in DocxExtractor().supported_media_types
    )


def test_extract_is_synchronous() -> None:
    assert not inspect.iscoroutinefunction(DocxExtractor.extract)
    assert not inspect.isasyncgenfunction(DocxExtractor.extract)


def test_availability_passes_when_the_parser_is_installed() -> None:
    assert DocxExtractor.check_availability() is None


def test_missing_parser_names_the_install_extra(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_import = __import__

    def _blocked(name: str, *args: object, **kwargs: object) -> object:
        if name == "docx":
            raise ImportError("no module named docx")
        return real_import(name, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr("builtins.__import__", _blocked)

    with pytest.raises(ExtractorUnavailableError) as excinfo:
        DocxExtractor.check_availability()

    assert excinfo.value.install_extra == "ptfm[office]"
    assert excinfo.value.missing_dependency == "python-docx"


# ---------------------------------------------------------------------------
# Declared structure
# ---------------------------------------------------------------------------


def test_heading_styles_yield_heading_blocks_with_levels() -> None:
    def _populate(document) -> None:
        document.add_heading("Chapter One", level=1)
        document.add_heading("Section", level=2)
        document.add_heading("Subsection", level=3)

    output = _extract(_populate)
    headings = _of_type(output, BlockType.HEADING)

    assert [b.text for b in headings] == ["Chapter One", "Section", "Subsection"]
    assert [b.attributes.heading_level for b in headings] == [1, 2, 3]


def test_body_paragraphs_yield_paragraph_blocks() -> None:
    def _populate(document) -> None:
        document.add_paragraph("First paragraph.")
        document.add_paragraph("Second paragraph.")

    output = _extract(_populate)

    assert [b.text for b in _of_type(output, BlockType.PARAGRAPH)] == [
        "First paragraph.",
        "Second paragraph.",
    ]


def test_no_heading_is_inferred_from_appearance() -> None:
    """A level invented from formatting would assert structure the author
    never declared."""

    def _populate(document) -> None:
        paragraph = document.add_paragraph()
        run = paragraph.add_run("LOOKS LIKE A HEADING")
        run.bold = True
        run.font.size = docx.shared.Pt(24)

    output = _extract(_populate)

    assert _of_type(output, BlockType.HEADING) == []
    assert _leaf_texts(output) == ["LOOKS LIKE A HEADING"]


def test_quote_styles_yield_quote_containers() -> None:
    def _populate(document) -> None:
        document.add_paragraph("Quoted passage.", style="Quote")

    output = _extract(_populate)
    quotes = _of_type(output, BlockType.QUOTE)

    assert len(quotes) == 1
    assert quotes[0].text is None
    assert _leaf_texts(output) == ["Quoted passage."]


def test_caption_styles_yield_caption_blocks() -> None:
    def _populate(document) -> None:
        document.add_paragraph("Figure 1: revenue", style="Caption")

    output = _extract(_populate)

    assert [b.text for b in _of_type(output, BlockType.CAPTION)] == [
        "Figure 1: revenue"
    ]


def test_blank_paragraphs_produce_no_blocks() -> None:
    """Word uses them for vertical spacing; they carry no content."""

    def _populate(document) -> None:
        document.add_paragraph("Real text.")
        document.add_paragraph("")
        document.add_paragraph("   ")
        document.add_paragraph("More text.")

    output = _extract(_populate)

    assert _leaf_texts(output) == ["Real text.", "More text."]


def test_an_empty_document_produces_no_blocks() -> None:
    output = _extract(lambda document: None)

    assert output.blocks == ()
    assert output.is_empty is True


# ---------------------------------------------------------------------------
# Document order
# ---------------------------------------------------------------------------


def test_paragraphs_and_tables_retain_their_interleaving() -> None:
    """The parser exposes them as separate collections, which loses order and
    would therefore misplace evidence."""

    def _populate(document) -> None:
        document.add_paragraph("Before the table.")
        table = document.add_table(rows=1, cols=1)
        table.cell(0, 0).text = "Cell value"
        document.add_paragraph("After the table.")

    output = _extract(_populate)

    assert _leaf_texts(output) == [
        "Before the table.",
        "Cell value",
        "After the table.",
    ]


def test_reading_order_is_preserved_across_mixed_content() -> None:
    def _populate(document) -> None:
        document.add_heading("Title", level=1)
        document.add_paragraph("Intro.")
        table = document.add_table(rows=1, cols=2)
        table.cell(0, 0).text = "A"
        table.cell(0, 1).text = "B"
        document.add_paragraph("Conclusion.")

    output = _extract(_populate)

    assert _leaf_texts(output) == ["Title", "Intro.", "A", "B", "Conclusion."]


# ---------------------------------------------------------------------------
# Lists
# ---------------------------------------------------------------------------


def test_list_paragraphs_produce_a_container_tree() -> None:
    def _populate(document) -> None:
        document.add_paragraph("alpha", style="List Bullet")
        document.add_paragraph("beta", style="List Bullet")

    output = _extract(_populate)

    assert len(_of_type(output, BlockType.LIST)) == 1
    assert len(_of_type(output, BlockType.LIST_ITEM)) == 2
    assert _leaf_texts(output) == ["alpha", "beta"]


def test_a_list_item_holds_its_text_in_a_leaf_child() -> None:
    """LIST_ITEM is a container, so a childless one could not be given a
    span."""

    def _populate(document) -> None:
        document.add_paragraph("only item", style="List Bullet")

    output = _extract(_populate)
    item = _of_type(output, BlockType.LIST_ITEM)[0]
    item_index = output.blocks.index(item)
    children = [b for b in output.blocks if b.parent_index == item_index]

    assert item.text is None
    assert [c.text for c in children] == ["only item"]


def test_a_list_is_closed_by_following_body_text() -> None:
    def _populate(document) -> None:
        document.add_paragraph("item", style="List Bullet")
        document.add_paragraph("Body text after the list.")

    output = _extract(_populate)
    paragraphs = [
        b
        for b in _of_type(output, BlockType.PARAGRAPH)
        if b.parent_index is None
    ]

    assert [b.text for b in paragraphs] == ["Body text after the list."]


def test_an_empty_list_item_still_receives_a_leaf() -> None:
    def _populate(document) -> None:
        document.add_paragraph("", style="List Bullet")

    output = _extract(_populate)
    item = _of_type(output, BlockType.LIST_ITEM)[0]
    item_index = output.blocks.index(item)

    assert [b.text for b in output.blocks if b.parent_index == item_index] == [""]


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------


def test_tables_produce_a_row_and_cell_tree() -> None:
    def _populate(document) -> None:
        table = document.add_table(rows=2, cols=2)
        table.cell(0, 0).text = "Region"
        table.cell(0, 1).text = "Revenue"
        table.cell(1, 0).text = "North"
        table.cell(1, 1).text = "120"

    output = _extract(_populate)

    assert len(_of_type(output, BlockType.TABLE)) == 1
    assert len(_of_type(output, BlockType.TABLE_ROW)) == 2
    assert len(_of_type(output, BlockType.TABLE_CELL)) == 4
    assert _leaf_texts(output) == ["Region", "Revenue", "North", "120"]


def test_cell_positions_are_recorded() -> None:
    """Position is what makes a single value citable."""

    def _populate(document) -> None:
        table = document.add_table(rows=2, cols=2)
        for row in range(2):
            for column in range(2):
                table.cell(row, column).text = f"r{row}c{column}"

    output = _extract(_populate)
    cells = _of_type(output, BlockType.TABLE_CELL)
    positions = [
        (c.attributes.row_index, c.attributes.column_index) for c in cells
    ]

    assert positions == [(0, 0), (0, 1), (1, 0), (1, 1)]


def test_an_empty_cell_receives_an_empty_leaf() -> None:
    def _populate(document) -> None:
        table = document.add_table(rows=1, cols=2)
        table.cell(0, 1).text = "value"

    output = _extract(_populate)
    cells = _of_type(output, BlockType.TABLE_CELL)
    empty_index = output.blocks.index(cells[0])

    assert [b.text for b in output.blocks if b.parent_index == empty_index] == [""]


def test_a_multi_paragraph_cell_is_joined() -> None:
    def _populate(document) -> None:
        table = document.add_table(rows=1, cols=1)
        cell = table.cell(0, 0)
        cell.text = "first line"
        cell.add_paragraph("second line")

    output = _extract(_populate)

    assert _leaf_texts(output) == ["first line\nsecond line"]


def test_merged_cells_are_reported_and_not_duplicated() -> None:
    """Reconstructing the logical grid would need span information the model
    cannot carry, so duplication is avoided and the fact recorded."""

    def _populate(document) -> None:
        table = document.add_table(rows=2, cols=2)
        table.cell(0, 0).merge(table.cell(0, 1))
        table.cell(0, 0).text = "merged"
        table.cell(1, 0).text = "a"
        table.cell(1, 1).text = "b"

    output = _extract(_populate)

    assert _leaf_texts(output).count("merged") == 1
    codes = [w.code for w in output.warnings]
    assert ExtractionWarningCode.TABLE_STRUCTURE_UNCERTAIN in codes


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------


def test_core_properties_are_harvested() -> None:
    def _populate(document) -> None:
        properties = document.core_properties
        properties.title = "Quarterly Report"
        properties.author = "Ada Lovelace; Alan Turing"
        properties.subject = "Finance"
        document.add_paragraph("Body.")

    metadata = _extract(_populate).metadata

    assert metadata.title == "Quarterly Report"
    assert metadata.authors == ("Ada Lovelace", "Alan Turing")
    assert metadata.properties["subject"] == "Finance"


def test_page_count_is_not_claimed() -> None:
    """A DOCX has no fixed pagination; reporting one would state as fact
    something that depends on the reader."""

    def _populate(document) -> None:
        document.add_paragraph("Body.")

    assert _extract(_populate).metadata.page_count is None


def test_absent_properties_yield_a_minimal_record() -> None:
    def _populate(document) -> None:
        document.add_paragraph("Body.")

    metadata = _extract(_populate).metadata

    assert metadata.authors == ()
    assert metadata.declared_language is None


# ---------------------------------------------------------------------------
# Unrecognized styles
# ---------------------------------------------------------------------------


def test_an_unrecognized_style_yields_body_text_and_a_warning() -> None:
    def _populate(document) -> None:
        document.add_paragraph("Intense text.", style="Intense Emphasis")

    output = _extract(_populate)

    assert _leaf_texts(output) == ["Intense text."]
    codes = [w.code for w in output.warnings]
    assert ExtractionWarningCode.STRUCTURE_INFERRED in codes


def test_unrecognized_styles_are_reported_once_per_style() -> None:
    def _populate(document) -> None:
        for index in range(20):
            document.add_paragraph(f"line {index}", style="Intense Emphasis")

    output = _extract(_populate)
    inferred = [
        w
        for w in output.warnings
        if w.code is ExtractionWarningCode.STRUCTURE_INFERRED
    ]

    assert len(inferred) == 1


def test_the_default_style_produces_no_warning() -> None:
    def _populate(document) -> None:
        document.add_paragraph("Ordinary text.")

    assert _extract(_populate).warnings == ()


# ---------------------------------------------------------------------------
# Style interpretation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("style", "expected"),
    [
        ("Heading 1", 1),
        ("Heading 3", 3),
        ("heading 9", 9),
        ("Title", 1),
        ("Subtitle", 2),
        ("Titre 2", 2),
        ("Heading", 1),
        ("Normal", None),
        ("Body Text", None),
        ("Heading 12", None),
        ("", None),
    ],
)
def test_heading_levels_derive_only_from_declared_styles(
    style: str, expected: int | None
) -> None:
    assert docx_module._heading_level(style) == expected


@pytest.mark.parametrize(
    ("style", "expected"),
    [
        ("Quote", BlockType.QUOTE),
        ("Intense Quote", BlockType.QUOTE),
        ("Code", BlockType.CODE),
        ("Caption", BlockType.CAPTION),
        ("Normal", None),
        ("", None),
    ],
)
def test_styles_map_to_canonical_block_types(
    style: str, expected: BlockType | None
) -> None:
    assert docx_module._styled_block_type(style) == expected


# ---------------------------------------------------------------------------
# Malformed input
# ---------------------------------------------------------------------------


def test_a_non_package_payload_is_reported_as_corrupt() -> None:
    payload = _payload(b"this is not a zip archive at all")

    with pytest.raises(CorruptDocumentError) as excinfo:
        DocxExtractor().extract(payload)

    assert excinfo.value.retryable is False


def test_an_ole_container_is_reported_as_encrypted() -> None:
    """An encrypted OOXML file is an OLE container, so it presents as a bad
    archive; the remedies differ entirely."""
    payload = _payload(b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1" + b"\x00" * 512)

    with pytest.raises(EncryptedDocumentError):
        DocxExtractor().extract(payload)


def test_a_truncated_package_is_reported_as_corrupt() -> None:
    data = _build(lambda document: document.add_paragraph("Body."))
    payload = _payload(data[: len(data) // 2])

    with pytest.raises(CorruptDocumentError):
        DocxExtractor().extract(payload)


# ---------------------------------------------------------------------------
# Structural validity
# ---------------------------------------------------------------------------


def test_output_is_always_structurally_valid() -> None:
    def _populate(document) -> None:
        document.add_heading("Title", level=1)
        document.add_paragraph("Body.")
        document.add_paragraph("item", style="List Bullet")
        table = document.add_table(rows=1, cols=2)
        table.cell(0, 0).text = "A"
        document.add_paragraph("Quoted.", style="Quote")

    assert isinstance(_extract(_populate), ExtractionOutput)


def test_parent_indices_always_precede_their_children() -> None:
    def _populate(document) -> None:
        document.add_paragraph("item", style="List Bullet")
        table = document.add_table(rows=2, cols=2)
        table.cell(0, 0).text = "A"

    output = _extract(_populate)

    for index, block in enumerate(output.blocks):
        if block.parent_index is not None:
            assert block.parent_index < index


def test_no_container_block_carries_text() -> None:
    def _populate(document) -> None:
        document.add_paragraph("item", style="List Bullet")
        document.add_paragraph("Quoted.", style="Quote")
        table = document.add_table(rows=1, cols=1)
        table.cell(0, 0).text = "A"

    output = _extract(_populate)
    containers = {
        BlockType.LIST,
        BlockType.LIST_ITEM,
        BlockType.QUOTE,
        BlockType.TABLE,
        BlockType.TABLE_ROW,
        BlockType.TABLE_CELL,
    }

    for block in output.blocks:
        if block.block_type in containers:
            assert block.text is None


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_extraction_is_deterministic() -> None:
    data = _build(lambda document: document.add_paragraph("Body."))
    extractor = DocxExtractor()
    payload = _payload(data)

    assert extractor.extract(payload) == extractor.extract(payload)


def test_extractor_holds_no_assignable_state() -> None:
    with pytest.raises(AttributeError):
        DocxExtractor().cache = {}  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Anti-pattern guards
# ---------------------------------------------------------------------------


def test_extractor_performs_no_io() -> None:
    source = inspect.getsource(docx_module)
    for forbidden in ("open(", "requests", "httpx", "async def"):
        assert forbidden not in source


def test_module_declares_no_mutable_state() -> None:
    mutable = [
        name
        for name, value in vars(docx_module).items()
        if not name.startswith("__") and isinstance(value, (list, set))
    ]
    assert mutable == []


def test_extractor_contains_no_domain_vocabulary() -> None:
    source = inspect.getsource(docx_module).lower()
    for term in ("jurisdiction", "statute", "ticker", "icd", "patient"):
        assert term not in source
