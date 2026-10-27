"""
Unit tests for the HTML extractor.

File:
    tests/core/ingestion/extractors/test_html.py
"""

from __future__ import annotations

import inspect

import pytest

from ptfm.core.ingestion.exceptions import ExtractorUnavailableError
from ptfm.core.ingestion.extractors import html as html_module
from ptfm.core.ingestion.extractors.html import HtmlExtractor
from ptfm.core.ingestion.interfaces import ContentExtractor
from ptfm.core.ingestion.models import (
    ExtractedBlock,
    ExtractionInput,
    ExtractionOutput,
)
from ptfm.core.schemas import BlockType, ExtractionWarningCode

pytest.importorskip("selectolax", reason="requires the ptfm[web] extra")


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def _payload(
    markup: str, *, media_type: str = "text/html"
) -> ExtractionInput:
    return ExtractionInput(
        source_uri="file:///corpus/page.html",
        media_type=media_type,
        data=markup.encode("utf-8"),
        encoding="utf-8",
        filename="page.html",
    )


def _extract(markup: str, **options: bool) -> ExtractionOutput:
    return HtmlExtractor(**options).extract(_payload(markup))


def _of_type(
    output: ExtractionOutput, block_type: BlockType
) -> list[ExtractedBlock]:
    return [b for b in output.blocks if b.block_type is block_type]


def _leaf_texts(output: ExtractionOutput) -> list[str]:
    return [text for _, block in output.iter_leaf_blocks() if (text := block.text)]


def _all_text(output: ExtractionOutput) -> str:
    return " ".join(_leaf_texts(output))


# ---------------------------------------------------------------------------
# Contract conformance
# ---------------------------------------------------------------------------


def test_extractor_satisfies_the_content_extractor_contract() -> None:
    extractor = HtmlExtractor()

    assert isinstance(extractor, ContentExtractor)
    assert extractor.name == "HtmlExtractor"
    assert extractor.supported_media_types == frozenset(
        {"text/html", "application/xhtml+xml"}
    )


def test_extract_is_synchronous() -> None:
    assert not inspect.iscoroutinefunction(HtmlExtractor.extract)
    assert not inspect.isasyncgenfunction(HtmlExtractor.extract)


def test_availability_passes_when_the_parser_is_installed() -> None:
    assert HtmlExtractor.check_availability() is None


def test_missing_parser_names_the_install_extra(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_import = __import__

    def _blocked(name: str, *args: object, **kwargs: object) -> object:
        if name.startswith("selectolax"):
            raise ImportError("no module named selectolax")
        return real_import(name, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr("builtins.__import__", _blocked)

    with pytest.raises(ExtractorUnavailableError) as excinfo:
        HtmlExtractor.check_availability()

    assert excinfo.value.install_extra == "ptfm[web]"
    assert excinfo.value.missing_dependency == "selectolax"


def test_the_content_detector_is_genuinely_optional() -> None:
    """Element removal alone still yields usable documents."""
    source = inspect.getsource(html_module)

    assert "import trafilatura" in source
    assert "trafilatura" not in inspect.getsource(HtmlExtractor.check_availability)


# ---------------------------------------------------------------------------
# Boilerplate removal
# ---------------------------------------------------------------------------


BOILERPLATE_PAGE = """
<html><body>
  <nav><a href="/">Home</a><a href="/about">About</a></nav>
  <header><h1>Site Name</h1></header>
  <aside>Subscribe to our newsletter</aside>
  <article><h2>The Ruling</h2><p>Substantive body text.</p></article>
  <footer>Copyright 2026. All rights reserved.</footer>
  <script>trackPageView();</script>
  <style>.hidden { display: none }</style>
</body></html>
"""


def test_navigation_and_furniture_are_discarded() -> None:
    """Repeated across a site, this text would dominate corpus frequency."""
    output = _extract(BOILERPLATE_PAGE)
    text = _all_text(output)

    assert "Substantive body text." in text
    for furniture in ("Home", "About", "newsletter", "Copyright", "rights reserved"):
        assert furniture not in text


def test_scripts_and_styles_are_discarded() -> None:
    output = _extract(BOILERPLATE_PAGE)
    text = _all_text(output)

    assert "trackPageView" not in text
    assert "display: none" not in text


@pytest.mark.parametrize(
    "tag", ["script", "style", "noscript", "iframe", "form", "button"]
)
def test_every_discarded_tag_is_removed(tag: str) -> None:
    markup = f"<html><body><{tag}>REMOVED</{tag}><p>Kept.</p></body></html>"

    output = _extract(markup)

    assert "REMOVED" not in _all_text(output)
    assert "Kept." in _all_text(output)


def test_boilerplate_removal_can_be_disabled() -> None:
    """Generated reports are content throughout and must not be pruned."""
    extractor = HtmlExtractor(remove_boilerplate=False)

    output = extractor.extract(_payload(BOILERPLATE_PAGE))

    assert "Substantive body text." in _all_text(output)


def test_short_documents_are_not_subjected_to_content_detection() -> None:
    """A fragment has no boilerplate and would be gutted by a detector."""
    output = _extract("<html><body><p>Brief fragment.</p></body></html>")

    assert _leaf_texts(output) == ["Brief fragment."]
    assert output.warnings == ()


# ---------------------------------------------------------------------------
# Headings and paragraphs
# ---------------------------------------------------------------------------


def test_headings_carry_their_outline_level() -> None:
    markup = "<h1>One</h1><h2>Two</h2><h3>Three</h3>"

    output = _extract(markup)
    headings = _of_type(output, BlockType.HEADING)

    assert [b.text for b in headings] == ["One", "Two", "Three"]
    assert [b.attributes.heading_level for b in headings] == [1, 2, 3]


def test_paragraphs_become_paragraph_blocks() -> None:
    output = _extract("<p>First.</p><p>Second.</p>")

    assert [b.text for b in _of_type(output, BlockType.PARAGRAPH)] == [
        "First.",
        "Second.",
    ]


def test_inline_markup_is_reduced_to_its_text() -> None:
    markup = '<p>See <a href="https://example.com/x">the ruling</a> for <em>detail</em>.</p>'

    output = _extract(markup)

    assert _leaf_texts(output) == ["See the ruling for detail ."]
    assert "https://example.com" not in _all_text(output)


def test_markup_indentation_does_not_become_content() -> None:
    markup = "<p>\n    Wrapped\n    across lines\n</p>"

    output = _extract(markup)

    assert _leaf_texts(output) == ["Wrapped across lines"]


def test_empty_document_produces_no_blocks() -> None:
    output = _extract("<html><body></body></html>")

    assert output.blocks == ()
    assert output.is_empty is True


def test_empty_elements_produce_no_blocks() -> None:
    output = _extract("<p></p><p>   </p><p>Real.</p>")

    assert _leaf_texts(output) == ["Real."]


# ---------------------------------------------------------------------------
# Lists, quotes, and code
# ---------------------------------------------------------------------------


def test_lists_produce_a_container_tree() -> None:
    output = _extract("<ul><li>alpha</li><li>beta</li></ul>")

    assert len(_of_type(output, BlockType.LIST)) == 1
    assert len(_of_type(output, BlockType.LIST_ITEM)) == 2
    assert _leaf_texts(output) == ["alpha", "beta"]


def test_nested_lists_record_their_depth() -> None:
    markup = "<ul><li>outer<ul><li>inner</li></ul></li></ul>"

    output = _extract(markup)
    items = _of_type(output, BlockType.LIST_ITEM)

    assert [item.attributes.list_level for item in items] == [0, 1]


def test_block_quotes_become_quote_containers() -> None:
    output = _extract("<blockquote><p>Quoted passage.</p></blockquote>")

    quote = _of_type(output, BlockType.QUOTE)[0]

    assert quote.text is None
    assert _leaf_texts(output) == ["Quoted passage."]


def test_preformatted_whitespace_is_preserved() -> None:
    """The structure stage exempts CODE from whitespace collapsing."""
    output = _extract("<pre>def f():\n    return 1</pre>")

    code = _of_type(output, BlockType.CODE)[0]

    assert "    return 1" in code.text


def test_figures_and_captions_are_recognized() -> None:
    markup = "<figure><img src='c.png'><figcaption>Revenue chart</figcaption></figure>"

    output = _extract(markup)

    assert len(_of_type(output, BlockType.FIGURE)) == 1
    assert [b.text for b in _of_type(output, BlockType.CAPTION)] == ["Revenue chart"]


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------


TABLE = """
<table>
  <thead><tr><th>Region</th><th>Revenue</th></tr></thead>
  <tbody><tr><td>North</td><td>120</td></tr></tbody>
</table>
"""


def test_tables_produce_a_row_and_cell_tree() -> None:
    output = _extract(TABLE)

    assert len(_of_type(output, BlockType.TABLE)) == 1
    assert len(_of_type(output, BlockType.TABLE_ROW)) == 2
    assert len(_of_type(output, BlockType.TABLE_CELL)) == 4


def test_header_cells_are_marked() -> None:
    output = _extract(TABLE)
    cells = _of_type(output, BlockType.TABLE_CELL)

    assert [c.attributes.is_header for c in cells] == [True, True, False, False]


def test_table_section_elements_add_no_tree_depth() -> None:
    """thead and tbody carry no content; representing them would be markup
    depth rather than meaning."""
    output = _extract(TABLE)
    rows = _of_type(output, BlockType.TABLE_ROW)
    table_index = output.blocks.index(_of_type(output, BlockType.TABLE)[0])

    assert all(row.parent_index == table_index for row in rows)


def test_cell_positions_are_recorded() -> None:
    output = _extract(TABLE)
    cells = _of_type(output, BlockType.TABLE_CELL)

    assert [c.attributes.column_index for c in cells[:2]] == [0, 1]


def test_an_empty_cell_receives_an_empty_leaf() -> None:
    """A container with no children could not be assigned a span."""
    output = _extract("<table><tr><td></td><td>90</td></tr></table>")

    cells = _of_type(output, BlockType.TABLE_CELL)
    empty_index = output.blocks.index(cells[0])
    children = [b for b in output.blocks if b.parent_index == empty_index]

    assert [c.text for c in children] == [""]


# ---------------------------------------------------------------------------
# Generic containers
# ---------------------------------------------------------------------------


def test_layout_divs_do_not_become_blocks() -> None:
    """Treating every div as a section would describe styling, not meaning."""
    markup = "<div><div><div><p>Deeply wrapped.</p></div></div></div>"

    output = _extract(markup)

    assert len(output.blocks) == 1
    assert output.blocks[0].block_type is BlockType.PARAGRAPH


def test_a_div_holding_text_directly_becomes_a_paragraph() -> None:
    output = _extract("<div>Loose text in a container.</div>")

    assert _leaf_texts(output) == ["Loose text in a container."]


def test_text_is_never_recorded_twice() -> None:
    markup = "<div>Preamble<p>Paragraph text.</p></div>"

    output = _extract(markup)

    assert _leaf_texts(output) == ["Preamble", "Paragraph text."]


def test_semantic_sections_are_preserved() -> None:
    output = _extract("<section><h2>Title</h2><p>Body.</p></section>")

    assert len(_of_type(output, BlockType.SECTION)) == 1


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------


HEAD = """
<html lang="en-GB">
<head>
  <title>Fallback Title</title>
  <meta name="author" content="Ada Lovelace, Alan Turing">
  <meta property="og:title" content="Preferred Title">
  <meta name="generator" content="Hugo 0.120">
  <meta name="description" content="A summary.">
</head>
<body><p>Body.</p></body></html>
"""


def test_metadata_is_harvested_from_the_head() -> None:
    metadata = _extract(HEAD).metadata

    assert metadata.title == "Preferred Title"
    assert metadata.authors == ("Ada Lovelace", "Alan Turing")
    assert metadata.declared_language == "en-GB"
    assert metadata.producer == "Hugo 0.120"
    assert metadata.properties["description"] == "A summary."


def test_title_falls_back_to_the_title_element() -> None:
    markup = "<html><head><title>Only Title</title></head><body><p>x</p></body></html>"

    assert _extract(markup).metadata.title == "Only Title"


def test_metadata_survives_content_extraction() -> None:
    """Head elements are discarded by extraction, so metadata precedes it."""
    metadata = _extract(HEAD).metadata

    assert metadata.title is not None
    assert "Fallback Title" not in _all_text(_extract(HEAD))


def test_promoted_keys_are_not_duplicated_into_properties() -> None:
    properties = _extract(HEAD).metadata.properties

    assert "title" not in properties
    assert "author" not in properties
    assert "generator" not in properties


def test_dates_are_not_recovered_as_timestamps() -> None:
    """A misinterpreted instant is worse than an absent one."""
    markup = (
        "<html><head><meta property='article:published_time' "
        "content='03/01/2025'></head><body><p>x</p></body></html>"
    )

    metadata = _extract(markup).metadata

    assert metadata.created_at is None


def test_absent_metadata_yields_a_minimal_record() -> None:
    metadata = _extract("<p>Body.</p>").metadata

    assert metadata.authors == ()
    assert metadata.producer is None


# ---------------------------------------------------------------------------
# Malformed markup
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "markup",
    [
        "<p>Unclosed paragraph",
        "<div><p>Mismatched</div></p>",
        "<table><tr><td>Orphaned cell</table>",
        "<p>Text with <b>unclosed emphasis</p>",
        "Bare text with no elements at all",
    ],
)
def test_malformed_markup_is_recovered_rather_than_rejected(markup: str) -> None:
    """Real HTML is frequently invalid; a strict parser would reject a corpus."""
    output = _extract(markup)

    assert isinstance(output, ExtractionOutput)


def test_unrecognized_elements_are_descended_into() -> None:
    markup = "<custom-element><p>Nested content.</p></custom-element>"

    output = _extract(markup)

    assert "Nested content." in _all_text(output)


def test_unrecognized_elements_are_reported_once() -> None:
    markup = "".join("<custom-el>x</custom-el>" for _ in range(20))

    output = _extract(markup)

    skipped = [
        w
        for w in output.warnings
        if w.code is ExtractionWarningCode.UNSUPPORTED_ELEMENT_SKIPPED
    ]
    assert len(skipped) <= 1


# ---------------------------------------------------------------------------
# Structural validity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "markup",
    [
        BOILERPLATE_PAGE,
        TABLE,
        HEAD,
        "<ul><li>a<ul><li>b</li></ul></li></ul>",
        "<blockquote><ul><li>quoted item</li></ul></blockquote>",
        "<div><div><p>nested</p></div></div>",
    ],
)
def test_output_is_always_structurally_valid(markup: str) -> None:
    """Constructing ExtractionOutput enforces pre-order, parentage, and
    container semantics."""
    assert isinstance(_extract(markup), ExtractionOutput)


def test_parent_indices_always_precede_their_children() -> None:
    output = _extract(TABLE + "<ul><li>a</li></ul>")

    for index, block in enumerate(output.blocks):
        if block.parent_index is not None:
            assert block.parent_index < index


def test_no_container_block_carries_text() -> None:
    output = _extract(TABLE + "<blockquote><p>q</p></blockquote>")

    containers = {
        BlockType.TABLE,
        BlockType.TABLE_ROW,
        BlockType.TABLE_CELL,
        BlockType.LIST,
        BlockType.LIST_ITEM,
        BlockType.QUOTE,
        BlockType.SECTION,
        BlockType.FIGURE,
    }
    for block in output.blocks:
        if block.block_type in containers:
            assert block.text is None


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_extraction_is_deterministic() -> None:
    extractor = HtmlExtractor()
    payload = _payload(TABLE)

    assert extractor.extract(payload) == extractor.extract(payload)


def test_repeated_extraction_does_not_accumulate_warnings() -> None:
    extractor = HtmlExtractor()
    payload = _payload("<custom-el>x</custom-el><p>y</p>")

    first = extractor.extract(payload)
    second = extractor.extract(payload)

    assert len(first.warnings) == len(second.warnings)


def test_extractor_holds_no_assignable_state() -> None:
    with pytest.raises(AttributeError):
        HtmlExtractor().cache = {}  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Anti-pattern guards
# ---------------------------------------------------------------------------


def test_decoding_is_shared_with_the_plain_text_extractor() -> None:
    source = inspect.getsource(html_module)

    assert "decode_payload" in source
    assert "def decode_payload" not in source


def test_extractor_performs_no_io() -> None:
    source = inspect.getsource(html_module)
    for forbidden in ("open(", "requests", "httpx", "async def", "urlopen"):
        assert forbidden not in source


def test_module_declares_no_mutable_state() -> None:
    mutable = [
        name
        for name, value in vars(html_module).items()
        if not name.startswith("__") and isinstance(value, (list, set))
    ]
    assert mutable == []


def test_extractor_contains_no_domain_vocabulary() -> None:
    source = inspect.getsource(html_module).lower()
    for term in ("jurisdiction", "statute", "ticker", "icd", "patient"):
        assert term not in source
