"""
Unit tests for the tabular extractor.

File:
    tests/core/ingestion/extractors/test_tabular.py
"""

from __future__ import annotations

import csv
import inspect

import pytest

from ptfm.core.ingestion.config import ExtractionConfig
from ptfm.core.ingestion.exceptions import CorruptDocumentError
from ptfm.core.ingestion.extractors import tabular as tabular_module
from ptfm.core.ingestion.extractors.tabular import TabularExtractor
from ptfm.core.ingestion.interfaces import ContentExtractor
from ptfm.core.ingestion.models import (
    ExtractedBlock,
    ExtractionInput,
    ExtractionOutput,
)
from ptfm.core.ingestion.registry import ExtractorRegistry
from ptfm.core.schemas import BlockType, ExtractionWarningCode

CSV_WITH_HEADER = "region,revenue\nNorth,120\nSouth,90\n"


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def _payload(
    body: str, *, media_type: str = "text/csv", encoding: str = "utf-8"
) -> ExtractionInput:
    return ExtractionInput(
        source_uri="file:///corpus/data.csv",
        media_type=media_type,
        data=body.encode(encoding),
        encoding=encoding,
        filename="data.csv",
    )


def _extract(body: str, *, media_type: str = "text/csv", **options: bool) -> ExtractionOutput:
    return TabularExtractor(**options).extract(_payload(body, media_type=media_type))


def _of_type(
    output: ExtractionOutput, block_type: BlockType
) -> list[ExtractedBlock]:
    return [b for b in output.blocks if b.block_type is block_type]


def _values(output: ExtractionOutput) -> list[str]:
    return [block.text for _, block in output.iter_leaf_blocks()]


# ---------------------------------------------------------------------------
# Contract conformance
# ---------------------------------------------------------------------------


def test_extractor_satisfies_the_content_extractor_contract() -> None:
    extractor = TabularExtractor()

    assert isinstance(extractor, ContentExtractor)
    assert extractor.name == "TabularExtractor"
    assert extractor.supported_media_types == frozenset(
        {"text/csv", "text/tab-separated-values"}
    )


def test_extract_is_synchronous() -> None:
    assert not inspect.iscoroutinefunction(TabularExtractor.extract)
    assert not inspect.isasyncgenfunction(TabularExtractor.extract)


def test_extractor_is_always_available() -> None:
    """Standard library only, so no optional dependency can be missing."""
    assert TabularExtractor.check_availability() is None


def test_extractor_registers_without_arguments() -> None:
    registry = ExtractorRegistry([TabularExtractor()], config=ExtractionConfig())

    assert registry.resolve("text/csv").name == "TabularExtractor"


# ---------------------------------------------------------------------------
# Structure
# ---------------------------------------------------------------------------


def test_a_table_tree_is_produced() -> None:
    output = _extract(CSV_WITH_HEADER)

    assert len(_of_type(output, BlockType.TABLE)) == 1
    assert len(_of_type(output, BlockType.TABLE_ROW)) == 3
    assert len(_of_type(output, BlockType.TABLE_CELL)) == 6


def test_values_are_reachable_in_reading_order() -> None:
    output = _extract(CSV_WITH_HEADER)

    assert _values(output) == ["region", "revenue", "North", "120", "South", "90"]


def test_every_cell_holds_its_value_in_a_leaf_child() -> None:
    """A cell is a container, so a childless one could not be given a span."""
    output = _extract("a,b\n")

    for index, block in enumerate(output.blocks):
        if block.block_type is BlockType.TABLE_CELL:
            children = [b for b in output.blocks if b.parent_index == index]
            assert len(children) == 1
            assert children[0].text is not None


def test_row_and_column_positions_are_recorded() -> None:
    """Position is what makes a single value citable."""
    output = _extract(CSV_WITH_HEADER)
    cells = _of_type(output, BlockType.TABLE_CELL)

    assert (cells[0].attributes.row_index, cells[0].attributes.column_index) == (0, 0)
    assert (cells[3].attributes.row_index, cells[3].attributes.column_index) == (1, 1)


def test_rows_are_parented_to_the_table() -> None:
    output = _extract(CSV_WITH_HEADER)

    assert all(row.parent_index == 0 for row in _of_type(output, BlockType.TABLE_ROW))


def test_empty_cells_are_preserved() -> None:
    """Omitting a blank cell would misalign every subsequent column."""
    output = _extract("a,b,c\n1,,3\n")

    assert _values(output) == ["a", "b", "c", "1", "", "3"]


# ---------------------------------------------------------------------------
# Value fidelity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value",
    ["007", "1.10", "0001234", "1e5", "+44", "0.500", "  padded  "],
)
def test_values_reach_the_corpus_exactly_as_written(value: str) -> None:
    """Type inference would change identifiers and versions irreversibly."""
    output = _extract(f"code\n{value}\n")

    assert _values(output)[-1] == value


def test_quoted_delimiters_are_handled() -> None:
    output = _extract('name,note\n"Smith, John",ok\n')

    assert _values(output) == ["name", "note", "Smith, John", "ok"]


def test_embedded_newlines_are_preserved() -> None:
    output = _extract('name,note\nAda,"line one\nline two"\n')

    assert _values(output)[-1] == "line one\nline two"


def test_escaped_quotes_are_handled() -> None:
    output = _extract('name,note\nAda,"she said ""yes"""\n')

    assert _values(output)[-1] == 'she said "yes"'


def test_unicode_values_survive() -> None:
    output = _extract("name\ncafé\n日本語\n")

    assert _values(output) == ["name", "café", "日本語"]


# ---------------------------------------------------------------------------
# Header detection
# ---------------------------------------------------------------------------


def test_a_textual_first_row_is_identified_as_a_header() -> None:
    output = _extract(CSV_WITH_HEADER)
    cells = _of_type(output, BlockType.TABLE_CELL)

    assert [c.attributes.is_header for c in cells[:2]] == [True, True]
    assert cells[2].attributes.is_header is False
    assert output.metadata.properties["has_header"] == "true"


def test_a_numeric_first_row_is_not_a_header() -> None:
    """Claiming one would silently remove a data row from the corpus."""
    output = _extract("1,2\n3,4\n")

    assert all(
        c.attributes.is_header is False
        for c in _of_type(output, BlockType.TABLE_CELL)
    )
    assert output.metadata.properties["has_header"] == "false"


def test_duplicate_first_row_values_prevent_header_identification() -> None:
    output = _extract("a,a\n1,2\n")

    assert output.metadata.properties["has_header"] == "false"


def test_an_empty_first_field_prevents_header_identification() -> None:
    output = _extract("a,,c\n1,2,3\n")

    assert output.metadata.properties["has_header"] == "false"


def test_header_detection_can_be_disabled() -> None:
    output = _extract(CSV_WITH_HEADER, detect_header=False)

    assert output.metadata.properties["has_header"] == "false"
    assert all(
        c.attributes.is_header is False
        for c in _of_type(output, BlockType.TABLE_CELL)
    )


def test_no_row_is_ever_dropped_by_header_detection() -> None:
    with_header = _extract(CSV_WITH_HEADER)
    without_header = _extract(CSV_WITH_HEADER, detect_header=False)

    assert len(_of_type(with_header, BlockType.TABLE_ROW)) == 3
    assert len(_of_type(without_header, BlockType.TABLE_ROW)) == 3


# ---------------------------------------------------------------------------
# Delimiters
# ---------------------------------------------------------------------------


def test_tab_separated_media_type_is_definitive() -> None:
    """The format is defined by its delimiter; sniffing could contradict it."""
    output = _extract(
        "region\trevenue\nNorth\t120\n", media_type="text/tab-separated-values"
    )

    assert _values(output) == ["region", "revenue", "North", "120"]
    assert output.metadata.properties["delimiter"] == "tab"


@pytest.mark.parametrize(
    ("delimiter", "described"),
    [(";", "semicolon"), ("|", "pipe"), ("\t", "tab")],
)
def test_alternative_delimiters_are_detected(
    delimiter: str, described: str
) -> None:
    body = f"region{delimiter}revenue\nNorth{delimiter}120\nSouth{delimiter}90\n"

    output = _extract(body)

    assert _values(output) == ["region", "revenue", "North", "120", "South", "90"]
    assert output.metadata.properties["delimiter"] == described


def test_a_detected_delimiter_differing_from_the_media_type_is_reported() -> None:
    output = _extract("a;b\n1;2\n3;4\n")

    codes = [w.code for w in output.warnings]
    assert ExtractionWarningCode.STRUCTURE_INFERRED in codes


def test_a_single_column_file_falls_back_to_the_implied_delimiter() -> None:
    output = _extract("value\nalpha\nbeta\n")

    assert _values(output) == ["value", "alpha", "beta"]


# ---------------------------------------------------------------------------
# Ragged rows
# ---------------------------------------------------------------------------


def test_ragged_rows_are_preserved_exactly() -> None:
    """Padding or truncating would fabricate or discard data."""
    output = _extract("a,b,c\n1,2,3\n4,5\n6,7,8,9\n")

    rows = _of_type(output, BlockType.TABLE_ROW)
    cells_per_row = [
        len([b for b in output.blocks if b.parent_index == output.blocks.index(row)])
        for row in rows
    ]

    assert cells_per_row == [3, 3, 2, 4]


def test_ragged_rows_are_reported_once() -> None:
    body = "a,b,c\n" + "".join(f"{n},{n}\n" for n in range(20))

    output = _extract(body)

    ragged = [
        w
        for w in output.warnings
        if w.code is ExtractionWarningCode.TABLE_STRUCTURE_UNCERTAIN
    ]
    assert len(ragged) == 1
    assert ragged[0].detail["ragged_row_count"] == 20
    assert ragged[0].detail["expected_columns"] == 3


def test_a_uniform_file_produces_no_raggedness_warning() -> None:
    output = _extract(CSV_WITH_HEADER)

    assert not [
        w
        for w in output.warnings
        if w.code is ExtractionWarningCode.TABLE_STRUCTURE_UNCERTAIN
    ]


# ---------------------------------------------------------------------------
# Empty and degenerate payloads
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("body", ["", "   ", "\n\n\n", ",,\n,,\n"])
def test_payloads_with_no_records_produce_no_blocks(body: str) -> None:
    """An empty table container could not be assigned a span."""
    output = _extract(body)

    assert output.blocks == ()
    assert output.is_empty is True
    assert output.metadata.properties["row_count"] == "0"


def test_blank_lines_between_records_are_dropped() -> None:
    output = _extract("a,b\n\n1,2\n\n\n3,4\n")

    assert len(_of_type(output, BlockType.TABLE_ROW)) == 3


def test_a_single_row_file_is_extracted() -> None:
    output = _extract("only,row\n")

    assert len(_of_type(output, BlockType.TABLE_ROW)) == 1
    assert _values(output) == ["only", "row"]


# ---------------------------------------------------------------------------
# Malformed input
# ---------------------------------------------------------------------------


def test_an_oversized_field_is_reported_as_corruption() -> None:
    """An unterminated quotation would otherwise consume the whole document."""
    body = 'a,b\n"' + "x" * (5 * 1024 * 1024) + "\n"

    with pytest.raises(CorruptDocumentError) as excinfo:
        _extract(body)

    assert excinfo.value.retryable is False


def test_the_process_global_field_limit_is_restored() -> None:
    """The limit is global; leaving it raised would affect every consumer."""
    before = csv.field_size_limit()

    _extract(CSV_WITH_HEADER)

    assert csv.field_size_limit() == before


def test_the_limit_is_restored_even_when_parsing_fails() -> None:
    before = csv.field_size_limit()

    with pytest.raises(CorruptDocumentError):
        _extract('a,b\n"' + "x" * (5 * 1024 * 1024) + "\n")

    assert csv.field_size_limit() == before


def test_undecodable_bytes_degrade_rather_than_fail() -> None:
    payload = ExtractionInput(
        source_uri="file:///corpus/data.csv",
        media_type="text/csv",
        data="a,b\n1,2\n".encode("utf-8") + b"\xff\xfe",
        encoding="utf-8",
        filename="data.csv",
    )

    output = TabularExtractor().extract(payload)

    codes = [w.code for w in output.warnings]
    assert ExtractionWarningCode.ENCODING_REPLACEMENT in codes
    assert output.blocks


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------


def test_shape_is_recorded_as_metadata() -> None:
    properties = _extract(CSV_WITH_HEADER).metadata.properties

    assert properties["row_count"] == "3"
    assert properties["column_count"] == "2"
    assert properties["delimiter"] == "comma"


def test_no_metadata_is_invented() -> None:
    metadata = _extract(CSV_WITH_HEADER).metadata

    assert metadata.title is None
    assert metadata.authors == ()
    assert metadata.created_at is None
    assert metadata.page_count is None


def test_extractor_name_is_recorded_on_the_result() -> None:
    assert _extract(CSV_WITH_HEADER).extractor_name == "TabularExtractor"


# ---------------------------------------------------------------------------
# Structural validity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "body",
    [
        CSV_WITH_HEADER,
        "1,2\n3,4\n",
        "a,b,c\n1,2\n3,4,5,6\n",
        'name,note\nAda,"line one\nline two"\n',
        "single\n",
    ],
)
def test_output_is_always_structurally_valid(body: str) -> None:
    """Constructing ExtractionOutput enforces pre-order, parentage, and
    container semantics."""
    assert isinstance(_extract(body), ExtractionOutput)


def test_parent_indices_always_precede_their_children() -> None:
    output = _extract(CSV_WITH_HEADER)

    for index, block in enumerate(output.blocks):
        if block.parent_index is not None:
            assert block.parent_index < index


def test_no_container_block_carries_text() -> None:
    output = _extract(CSV_WITH_HEADER)

    for block in output.blocks:
        if block.block_type in {
            BlockType.TABLE,
            BlockType.TABLE_ROW,
            BlockType.TABLE_CELL,
        }:
            assert block.text is None


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_extraction_is_deterministic() -> None:
    extractor = TabularExtractor()
    payload = _payload(CSV_WITH_HEADER)

    assert extractor.extract(payload) == extractor.extract(payload)


def test_extractor_holds_no_assignable_state() -> None:
    with pytest.raises(AttributeError):
        TabularExtractor().cache = {}  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Anti-pattern guards
# ---------------------------------------------------------------------------


def test_no_dataframe_library_is_imported() -> None:
    """Type inference is actively unwanted; values must survive verbatim."""
    source = inspect.getsource(tabular_module)

    assert "pandas" not in source
    assert "pyarrow" not in source
    assert "numpy" not in source


def test_decoding_is_shared_with_the_plain_text_extractor() -> None:
    source = inspect.getsource(tabular_module)

    assert "decode_payload" in source
    assert "def decode_payload" not in source


def test_extractor_performs_no_io() -> None:
    source = inspect.getsource(tabular_module)
    for forbidden in ("open(", "requests", "httpx", "async def"):
        assert forbidden not in source


def test_module_declares_no_mutable_state() -> None:
    mutable = [
        name
        for name, value in vars(tabular_module).items()
        if not name.startswith("__") and isinstance(value, (list, set))
    ]
    assert mutable == []


def test_extractor_contains_no_domain_vocabulary() -> None:
    source = inspect.getsource(tabular_module).lower()
    for term in ("jurisdiction", "statute", "ticker", "icd", "patient"):
        assert term not in source
