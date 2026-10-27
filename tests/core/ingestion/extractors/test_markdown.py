"""
Unit tests for the Markdown extractor.

File:
    tests/core/ingestion/extractors/test_markdown.py
"""

from __future__ import annotations

import inspect
import sys

import pytest

from ptfm.core.ingestion.exceptions import ExtractorUnavailableError
from ptfm.core.ingestion.extractors import markdown as markdown_module
from ptfm.core.ingestion.extractors.markdown import MarkdownExtractor
from ptfm.core.ingestion.interfaces import ContentExtractor
from ptfm.core.ingestion.models import ExtractedBlock, ExtractionInput, ExtractionOutput
from ptfm.core.schemas import BlockType, ExtractionWarningCode

pytest.importorskip("markdown_it", reason="requires the ptfm[markdown] extra")


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def _payload(body: str) -> ExtractionInput:
    return ExtractionInput(
        source_uri="file:///corpus/doc.md",
        media_type="text/markdown",
        data=body.encode("utf-8"),
        encoding="utf-8",
        filename="doc.md",
    )


def _extract(body: str) -> ExtractionOutput:
    return MarkdownExtractor().extract(_payload(body))


def _of_type(
    output: ExtractionOutput, block_type: BlockType
) -> list[ExtractedBlock]:
    return [block for block in output.blocks if block.block_type is block_type]


def _leaf_texts(output: ExtractionOutput) -> list[str]:
    return [text for _, block in output.iter_leaf_blocks() if (text := block.text)]


# ---------------------------------------------------------------------------
# Contract conformance
# ---------------------------------------------------------------------------


def test_extractor_satisfies_the_content_extractor_contract() -> None:
    extractor = MarkdownExtractor()

    assert isinstance(extractor, ContentExtractor)
    assert extractor.name == "MarkdownExtractor"
    assert extractor.supported_media_types == frozenset({"text/markdown"})


def test_extract_is_synchronous() -> None:
    assert not inspect.iscoroutinefunction(MarkdownExtractor.extract)
    assert not inspect.isasyncgenfunction(MarkdownExtractor.extract)


def test_availability_passes_when_the_parser_is_installed() -> None:
    assert MarkdownExtractor.check_availability() is None


def test_missing_parser_names_the_install_extra(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A missing extra must be actionable at startup, not an ImportError
    forty thousand documents into a run."""
    monkeypatch.setitem(sys.modules, "markdown_it", None)
    monkeypatch.setattr(
        markdown_module,
        "__builtins__",
        markdown_module.__builtins__,
        raising=False,
    )

    real_import = __import__

    def _blocked(name: str, *args: object, **kwargs: object) -> object:
        if name == "markdown_it":
            raise ImportError("no module named markdown_it")
        return real_import(name, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr("builtins.__import__", _blocked)

    with pytest.raises(ExtractorUnavailableError) as excinfo:
        MarkdownExtractor.check_availability()

    assert excinfo.value.install_extra == "ptfm[markdown]"
    assert excinfo.value.missing_dependency == "markdown-it-py"


# ---------------------------------------------------------------------------
# Headings and paragraphs
# ---------------------------------------------------------------------------


def test_headings_become_heading_blocks_with_levels() -> None:
    output = _extract("# Title\n\n## Section\n\n### Subsection\n")

    headings = _of_type(output, BlockType.HEADING)

    assert [block.text for block in headings] == ["Title", "Section", "Subsection"]
    assert [block.attributes.heading_level for block in headings] == [1, 2, 3]


def test_setext_headings_are_recognized() -> None:
    output = _extract("Title\n=====\n\nBody.\n")

    headings = _of_type(output, BlockType.HEADING)

    assert [block.text for block in headings] == ["Title"]
    assert headings[0].attributes.heading_level == 1


def test_paragraphs_become_paragraph_blocks() -> None:
    output = _extract("First paragraph.\n\nSecond paragraph.\n")

    assert [b.text for b in _of_type(output, BlockType.PARAGRAPH)] == [
        "First paragraph.",
        "Second paragraph.",
    ]


def test_empty_document_produces_no_blocks() -> None:
    output = _extract("")

    assert output.blocks == ()
    assert output.is_empty is True


# ---------------------------------------------------------------------------
# Inline syntax
# ---------------------------------------------------------------------------


def test_link_syntax_is_reduced_to_its_label() -> None:
    """Leaving the target in place would index URLs as prose."""
    output = _extract("See [the ruling](https://example.com/case) for detail.\n")

    assert _leaf_texts(output) == ["See the ruling for detail."]


def test_emphasis_markers_are_removed() -> None:
    output = _extract("This is *emphasised* and **strong** text.\n")

    assert _leaf_texts(output) == ["This is emphasised and strong text."]


def test_inline_code_content_is_retained() -> None:
    output = _extract("Call `compute_total()` first.\n")

    assert _leaf_texts(output) == ["Call compute_total() first."]


def test_image_alternative_text_is_retained() -> None:
    """Alt text is the only description of the image the document offers."""
    output = _extract("![Quarterly revenue chart](chart.png)\n")

    assert _leaf_texts(output) == ["Quarterly revenue chart"]


def test_reference_links_are_reduced_to_their_label() -> None:
    output = _extract(
        "See [the report][ref] for detail.\n\n[ref]: https://example.com\n"
    )

    assert _leaf_texts(output) == ["See the report for detail."]


def test_line_breaks_within_a_paragraph_are_preserved_as_breaks() -> None:
    output = _extract("First line\nsecond line\n")

    assert _leaf_texts(output) == ["First line\nsecond line"]


# ---------------------------------------------------------------------------
# Lists
# ---------------------------------------------------------------------------


def test_bullet_lists_produce_a_container_tree() -> None:
    output = _extract("- alpha\n- beta\n")

    assert len(_of_type(output, BlockType.LIST)) == 1
    assert len(_of_type(output, BlockType.LIST_ITEM)) == 2
    assert _leaf_texts(output) == ["alpha", "beta"]


def test_ordered_lists_are_also_list_blocks() -> None:
    output = _extract("1. first\n2. second\n")

    assert len(_of_type(output, BlockType.LIST)) == 1
    assert _leaf_texts(output) == ["first", "second"]


def test_list_items_hold_their_text_in_a_leaf_child() -> None:
    """LIST_ITEM is a container, so its text must live in a child block."""
    output = _extract("- alpha\n")

    item = _of_type(output, BlockType.LIST_ITEM)[0]
    item_index = output.blocks.index(item)
    children = [b for b in output.blocks if b.parent_index == item_index]

    assert item.text is None
    assert [child.text for child in children] == ["alpha"]


def test_nested_lists_record_their_depth() -> None:
    output = _extract("- outer\n    - inner\n")

    levels = [b.attributes.list_level for b in _of_type(output, BlockType.LIST)]

    assert levels == [0, 1]


def test_deeply_nested_lists_form_a_valid_tree() -> None:
    body = "- a\n    - b\n        - c\n            - d\n"

    output = _extract(body)

    assert _leaf_texts(output) == ["a", "b", "c", "d"]


# ---------------------------------------------------------------------------
# Quotes and code
# ---------------------------------------------------------------------------


def test_block_quotes_produce_a_quote_container() -> None:
    output = _extract("> quoted passage\n")

    quote = _of_type(output, BlockType.QUOTE)[0]

    assert quote.text is None
    assert _leaf_texts(output) == ["quoted passage"]


def test_fenced_code_becomes_a_code_block() -> None:
    output = _extract("```python\ndef f():\n    return 1\n```\n")

    code = _of_type(output, BlockType.CODE)

    assert len(code) == 1
    assert code[0].text == "def f():\n    return 1\n"


def test_code_indentation_is_preserved_verbatim() -> None:
    """The structure stage exempts CODE from whitespace collapsing."""
    output = _extract("```\n    deeply\n        indented\n```\n")

    assert "    deeply" in _of_type(output, BlockType.CODE)[0].text


def test_indented_code_blocks_are_recognized() -> None:
    output = _extract("    literal code line\n")

    assert len(_of_type(output, BlockType.CODE)) == 1


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------


TABLE = "| Region | Revenue |\n| --- | --- |\n| North | 120 |\n| South | 90 |\n"


def test_tables_produce_a_row_and_cell_tree() -> None:
    output = _extract(TABLE)

    assert len(_of_type(output, BlockType.TABLE)) == 1
    assert len(_of_type(output, BlockType.TABLE_ROW)) == 3
    assert len(_of_type(output, BlockType.TABLE_CELL)) == 6


def test_header_cells_are_marked() -> None:
    output = _extract(TABLE)

    cells = _of_type(output, BlockType.TABLE_CELL)

    assert [cell.attributes.is_header for cell in cells[:2]] == [True, True]
    assert cells[2].attributes.is_header is False


def test_cell_positions_are_recorded() -> None:
    output = _extract(TABLE)

    cells = _of_type(output, BlockType.TABLE_CELL)

    assert [cell.attributes.column_index for cell in cells[:2]] == [0, 1]


def test_cell_text_is_reachable_through_leaf_children() -> None:
    output = _extract(TABLE)

    assert _leaf_texts(output) == [
        "Region",
        "Revenue",
        "North",
        "120",
        "South",
        "90",
    ]


def test_an_empty_cell_receives_an_empty_leaf() -> None:
    """A container with no children could not be assigned a span."""
    output = _extract("| A | B |\n| --- | --- |\n|  | 90 |\n")

    cells = _of_type(output, BlockType.TABLE_CELL)
    empty_index = output.blocks.index(cells[2])
    children = [b for b in output.blocks if b.parent_index == empty_index]

    assert [child.text for child in children] == [""]


def test_a_table_of_only_empty_cells_still_validates() -> None:
    output = _extract("| A | B |\n| --- | --- |\n|  |  |\n")

    assert len(_of_type(output, BlockType.TABLE_CELL)) == 4


# ---------------------------------------------------------------------------
# Front matter
# ---------------------------------------------------------------------------


def test_front_matter_is_removed_from_the_body() -> None:
    """Left in place it parses as a thematic break and stray paragraphs."""
    output = _extract("---\ntitle: Quarterly Report\n---\n\nBody text.\n")

    assert _leaf_texts(output) == ["Body text."]


def test_front_matter_title_is_promoted() -> None:
    output = _extract("---\ntitle: Quarterly Report\n---\n\n# Other Heading\n")

    assert output.metadata.title == "Quarterly Report"


def test_front_matter_authors_are_split() -> None:
    output = _extract("---\nauthors: Ada Lovelace, Alan Turing\n---\n\nBody.\n")

    assert output.metadata.authors == ("Ada Lovelace", "Alan Turing")


def test_front_matter_language_is_promoted() -> None:
    output = _extract("---\nlang: en-GB\n---\n\nBody.\n")

    assert output.metadata.declared_language == "en-GB"


def test_remaining_front_matter_keys_become_properties() -> None:
    output = _extract("---\ntitle: T\ndate: 2025-03-01\ndraft: false\n---\n\nBody.\n")

    assert output.metadata.properties == {"date": "2025-03-01", "draft": "false"}


def test_promoted_keys_are_not_duplicated_into_properties() -> None:
    output = _extract("---\ntitle: T\n---\n\nBody.\n")

    assert "title" not in output.metadata.properties


def test_front_matter_dates_are_not_interpreted_as_timestamps() -> None:
    """A misinterpreted instant is worse than an absent one."""
    output = _extract("---\ndate: 03/01/2025\n---\n\nBody.\n")

    assert output.metadata.created_at is None
    assert output.metadata.properties["date"] == "03/01/2025"


def test_structured_front_matter_values_are_skipped_not_approximated() -> None:
    body = "---\ntitle: T\ntags:\n  - alpha\n  - beta\n---\n\nBody.\n"

    output = _extract(body)

    assert "tags" not in output.metadata.properties
    assert output.metadata.title == "T"


def test_unclosed_front_matter_is_reported_and_kept_as_content() -> None:
    output = _extract("---\ntitle: T\n\nBody text.\n")

    codes = [warning.code for warning in output.warnings]
    assert ExtractionWarningCode.METADATA_UNAVAILABLE in codes


def test_quoted_front_matter_values_are_unquoted() -> None:
    output = _extract('---\ntitle: "Quarterly Report"\n---\n\nBody.\n')

    assert output.metadata.title == "Quarterly Report"


# ---------------------------------------------------------------------------
# Title inference
# ---------------------------------------------------------------------------


def test_title_falls_back_to_the_first_heading() -> None:
    output = _extract("# Document Title\n\nBody text.\n")

    assert output.metadata.title == "Document Title"


def test_first_heading_wins_over_later_headings() -> None:
    output = _extract("# First\n\n## Second\n")

    assert output.metadata.title == "First"


def test_a_document_without_headings_has_no_title() -> None:
    output = _extract("Just body text.\n")

    assert output.metadata.title is None


# ---------------------------------------------------------------------------
# Skipped content
# ---------------------------------------------------------------------------


def test_raw_html_is_skipped_and_reported() -> None:
    """Emitting markup as text would index tag names as prose."""
    output = _extract("<div class='note'>raw</div>\n\nBody.\n")

    codes = [w.code for w in output.warnings]
    assert ExtractionWarningCode.UNSUPPORTED_ELEMENT_SKIPPED in codes
    assert _leaf_texts(output) == ["Body."]


def test_repeated_skips_are_reported_once() -> None:
    body = "\n\n".join("<div>x</div>" for _ in range(20)) + "\n\nBody.\n"

    output = _extract(body)

    skipped = [
        w
        for w in output.warnings
        if w.code is ExtractionWarningCode.UNSUPPORTED_ELEMENT_SKIPPED
    ]
    assert len(skipped) == 1


def test_thematic_breaks_contribute_no_block() -> None:
    output = _extract("Before.\n\n---\n\nAfter.\n")

    assert _leaf_texts(output) == ["Before.", "After."]


# ---------------------------------------------------------------------------
# Structural validity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "body",
    [
        "# H\n\nBody.\n",
        "- a\n    - b\n\n> quote\n\n```\ncode\n```\n",
        TABLE,
        "---\ntitle: T\n---\n\n1. one\n2. two\n",
        "> - quoted list item\n",
    ],
)
def test_output_is_always_structurally_valid(body: str) -> None:
    """Constructing ExtractionOutput enforces pre-order, parentage, and
    container semantics; a malformed tree would raise here."""
    output = _extract(body)

    assert isinstance(output, ExtractionOutput)


def test_parent_indices_always_precede_their_children() -> None:
    output = _extract("> - quoted item\n\n" + TABLE)

    for index, block in enumerate(output.blocks):
        if block.parent_index is not None:
            assert block.parent_index < index


def test_no_container_block_carries_text() -> None:
    output = _extract("- a\n\n> q\n\n" + TABLE)

    for block in output.blocks:
        if block.block_type in {
            BlockType.LIST,
            BlockType.LIST_ITEM,
            BlockType.QUOTE,
            BlockType.TABLE,
            BlockType.TABLE_ROW,
            BlockType.TABLE_CELL,
        }:
            assert block.text is None


# ---------------------------------------------------------------------------
# Determinism and concurrency
# ---------------------------------------------------------------------------


def test_extraction_is_deterministic() -> None:
    extractor = MarkdownExtractor()
    payload = _payload("# T\n\n- a\n- b\n")

    assert extractor.extract(payload) == extractor.extract(payload)


def test_repeated_extraction_does_not_accumulate_warnings() -> None:
    extractor = MarkdownExtractor()
    payload = _payload("<div>x</div>\n\nBody.\n")

    first = extractor.extract(payload)
    second = extractor.extract(payload)

    assert len(first.warnings) == len(second.warnings) == 1


def test_extractor_holds_no_assignable_state() -> None:
    with pytest.raises(AttributeError):
        MarkdownExtractor().cache = {}  # type: ignore[attr-defined]


def test_a_fresh_parser_is_built_per_call() -> None:
    """Concurrent invocation is permitted, so no parser is shared."""
    source = inspect.getsource(MarkdownExtractor.extract)

    assert "MarkdownIt()" in source


# ---------------------------------------------------------------------------
# Decoding
# ---------------------------------------------------------------------------


def test_decoding_is_shared_with_the_plain_text_extractor() -> None:
    source = inspect.getsource(markdown_module)

    assert "decode_payload" in source
    assert "def decode_payload" not in source


def test_undecodable_bytes_degrade_rather_than_fail() -> None:
    payload = ExtractionInput(
        source_uri="file:///corpus/doc.md",
        media_type="text/markdown",
        data="# Title\n\nbody ".encode("utf-8") + b"\xff\xfe",
        encoding="utf-8",
        filename="doc.md",
    )

    output = MarkdownExtractor().extract(payload)

    codes = [w.code for w in output.warnings]
    assert ExtractionWarningCode.ENCODING_REPLACEMENT in codes
    assert output.metadata.title == "Title"


# ---------------------------------------------------------------------------
# Anti-pattern guards
# ---------------------------------------------------------------------------


def test_extractor_performs_no_io() -> None:
    source = inspect.getsource(markdown_module)
    for forbidden in ("open(", "requests", "httpx", "async def"):
        assert forbidden not in source


def test_extractor_declares_no_thresholds() -> None:
    source = inspect.getsource(markdown_module)

    assert "> 0.8" not in source
    assert "confidence" not in source.lower()


def test_module_declares_no_mutable_state() -> None:
    mutable = [
        name
        for name, value in vars(markdown_module).items()
        if not name.startswith("__") and isinstance(value, (list, set))
    ]
    assert mutable == []


def test_extractor_contains_no_domain_vocabulary() -> None:
    source = inspect.getsource(markdown_module).lower()
    for term in ("jurisdiction", "statute", "ticker", "icd", "patient"):
        assert term not in source
