"""Unit and property tests for the structure normalization stage."""

from __future__ import annotations

import unicodedata
from datetime import UTC, datetime

import pytest
from hypothesis import given, settings, strategies as st

from ptfm.core.ingestion.config import (
    ExtractionConfig,
    StructureConfig,
    TextNormalizationForm,
)
from ptfm.core.ingestion.exceptions import (
    SpanIntegrityError,
    StructureParsingError,
)
from ptfm.core.ingestion.models import (
    ExtractedBlock,
    ExtractedMetadata,
    ExtractionOutput,
)
from ptfm.core.ingestion.pipeline import structure as structure_module
from ptfm.core.ingestion.pipeline.structure import (
    NormalizedStructure,
    StructureNormalizer,
)
from ptfm.core.schemas import (
    BlockAttributes,
    BlockType,
    DocumentMetadata,
    ExtractionWarningCode,
    RawDocument,
    SourceDescriptor,
    derive_document_id,
)

FIXED_TIME = datetime(2026, 1, 1, tzinfo=UTC)


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def _normalizer(
    structure: StructureConfig | None = None,
    limits: ExtractionConfig | None = None,
) -> StructureNormalizer:
    return StructureNormalizer(
        structure or StructureConfig(), limits or ExtractionConfig()
    )


def _output(*blocks: ExtractedBlock, name: str = "TestExtractor") -> ExtractionOutput:
    return ExtractionOutput(blocks=blocks, extractor_name=name)


def _leaf(
    text: str,
    *,
    block_type: BlockType = BlockType.PARAGRAPH,
    parent_index: int | None = None,
    attributes: BlockAttributes | None = None,
) -> ExtractedBlock:
    return ExtractedBlock(
        block_type=block_type,
        text=text,
        parent_index=parent_index,
        attributes=attributes or BlockAttributes(),
    )


def _container(
    block_type: BlockType = BlockType.SECTION,
    *,
    parent_index: int | None = None,
) -> ExtractedBlock:
    return ExtractedBlock(block_type=block_type, parent_index=parent_index)


def _as_raw_document(result: NormalizedStructure) -> RawDocument:
    """Construct a RawDocument, exercising every canonical invariant."""
    return RawDocument(
        document_id=derive_document_id("file:///a.txt", "hash-1"),
        source=SourceDescriptor(
            source_uri="file:///a.txt",
            scheme="file",
            connector_name="TestConnector",
            acquired_at=FIXED_TIME,
            size_bytes=len(result.text),
            detected_media_type="text/plain",
        ),
        metadata=DocumentMetadata(
            char_count=result.char_count, word_count=len(result.text.split())
        ),
        text=result.text,
        blocks=result.blocks,
        content_hash="hash-1",
        text_hash="hash-2",
        warnings=result.warnings,
        ingested_at=FIXED_TIME,
    )


# ---------------------------------------------------------------------------
# The load-bearing guarantee
# ---------------------------------------------------------------------------


def test_output_always_constructs_a_valid_raw_document() -> None:
    """
    The stage's entire purpose: its output satisfies every RawDocument
    invariant by construction, not by later repair.
    """
    result = _normalizer().normalize(
        _output(
            _container(),
            _leaf("Chapter One", block_type=BlockType.HEADING, parent_index=0),
            _leaf("First paragraph.", parent_index=0),
            _leaf("Second paragraph.", parent_index=0),
            _container(BlockType.TABLE),
            _container(BlockType.TABLE_ROW, parent_index=4),
            _leaf("Cell A", block_type=BlockType.TABLE_CELL, parent_index=5),
            _leaf("Cell B", block_type=BlockType.TABLE_CELL, parent_index=5),
        )
    )

    document = _as_raw_document(result)

    assert len(document.blocks) == 8
    assert len(document.leaf_blocks()) == 5


def test_every_leaf_span_recovers_its_own_text() -> None:
    result = _normalizer().normalize(
        _output(_leaf("Alpha"), _leaf("Beta"), _leaf("Gamma"))
    )
    document = _as_raw_document(result)

    recovered = [document.block_text(b) for b in document.leaf_blocks()]

    assert recovered == ["Alpha", "Beta", "Gamma"]


def test_container_span_envelops_its_entire_subtree() -> None:
    result = _normalizer().normalize(
        _output(
            _container(),
            _leaf("Title", block_type=BlockType.HEADING, parent_index=0),
            _leaf("Body.", parent_index=0),
        )
    )
    document = _as_raw_document(result)

    section = document.blocks[0]

    assert section.span.start == 0
    assert section.span.end == len(result.text)
    assert document.block_text(section) == "Title\n\nBody."


def test_deep_nesting_envelopes_propagate_upward() -> None:
    result = _normalizer().normalize(
        _output(
            _container(),
            _container(BlockType.LIST, parent_index=0),
            _container(BlockType.LIST_ITEM, parent_index=1),
            _leaf("Innermost text", parent_index=2),
        )
    )
    document = _as_raw_document(result)

    for block in document.blocks[:3]:
        assert document.block_text(block) == "Innermost text"
    assert document.blocks[3].depth == 3


# ---------------------------------------------------------------------------
# Empty input
# ---------------------------------------------------------------------------


def test_empty_extraction_yields_an_empty_document() -> None:
    result = _normalizer().normalize(_output())

    assert result.text == ""
    assert result.blocks == ()
    assert result.warnings == ()
    assert _as_raw_document(result).blocks == ()


# ---------------------------------------------------------------------------
# Identifiers, depth, ordinals
# ---------------------------------------------------------------------------


def test_identifiers_derive_from_tree_position() -> None:
    result = _normalizer().normalize(
        _output(
            _container(),
            _leaf("A", parent_index=0),
            _leaf("B", parent_index=0),
            _leaf("C"),
        )
    )

    assert [b.block_id for b in result.blocks] == ["0", "0.0", "0.1", "1"]


def test_parent_links_use_identifiers_not_indices() -> None:
    result = _normalizer().normalize(
        _output(_container(), _leaf("A", parent_index=0))
    )

    assert result.blocks[0].parent_id is None
    assert result.blocks[1].parent_id == "0"


def test_depth_and_ordinals_are_assigned_from_position() -> None:
    result = _normalizer().normalize(
        _output(
            _container(),
            _leaf("A", parent_index=0),
            _leaf("B", parent_index=0),
            _leaf("C", parent_index=0),
        )
    )

    assert [b.depth for b in result.blocks] == [0, 1, 1, 1]
    assert [b.ordinal for b in result.blocks] == [0, 0, 1, 2]


def test_root_ordinals_are_contiguous() -> None:
    result = _normalizer().normalize(_output(_leaf("A"), _leaf("B"), _leaf("C")))

    assert [b.ordinal for b in result.blocks] == [0, 1, 2]


def test_identifiers_are_stable_across_repeated_normalization() -> None:
    """Stable identifiers are what make an evidence citation durable."""
    output = _output(_container(), _leaf("A", parent_index=0), _leaf("B"))
    normalizer = _normalizer()

    first = normalizer.normalize(output)
    second = normalizer.normalize(output)

    assert [b.block_id for b in first.blocks] == [b.block_id for b in second.blocks]
    assert first == second


def test_depth_limit_is_enforced() -> None:
    blocks = [_container()]
    for index in range(4):
        blocks.append(_container(BlockType.SECTION, parent_index=index))
    blocks.append(_leaf("deep", parent_index=len(blocks) - 1))

    normalizer = _normalizer(StructureConfig(max_block_depth=3))

    with pytest.raises(StructureParsingError, match="nesting depth"):
        normalizer.normalize(_output(*blocks))


# ---------------------------------------------------------------------------
# Line endings
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw", ["line one\r\nline two", "line one\rline two", "line one\u2028line two"]
)
def test_line_endings_are_normalized(raw: str) -> None:
    result = _normalizer(
        StructureConfig(collapse_whitespace=False)
    ).normalize(_output(_leaf(raw)))

    assert "\r" not in result.text
    assert result.text == "line one\nline two"


def test_carriage_returns_never_survive_into_the_document() -> None:
    result = _normalizer().normalize(_output(_leaf("a\r\nb"), _leaf("c\rd")))

    _as_raw_document(result)
    assert "\r" not in result.text


# ---------------------------------------------------------------------------
# Whitespace policy
# ---------------------------------------------------------------------------


def test_whitespace_runs_are_collapsed_by_default() -> None:
    result = _normalizer().normalize(_output(_leaf("too    many\n\n\nspaces")))

    assert result.text == "too many spaces"


def test_collapsing_can_be_disabled() -> None:
    result = _normalizer(
        StructureConfig(collapse_whitespace=False, strip_block_text=False)
    ).normalize(_output(_leaf("keep    me")))

    assert result.text == "keep    me"


def test_code_blocks_preserve_internal_whitespace() -> None:
    """Collapsing a code listing destroys indentation, which is content."""
    source = "def f():\n    if x:\n        return 1"
    result = _normalizer().normalize(_output(_leaf(source, block_type=BlockType.CODE)))

    assert result.text == source


def test_code_blocks_are_stripped_of_surrounding_newlines_only() -> None:
    result = _normalizer().normalize(
        _output(_leaf("\n\n    indented\n\n", block_type=BlockType.CODE))
    )

    assert result.text == "    indented"


def test_block_text_is_stripped_by_default() -> None:
    result = _normalizer().normalize(_output(_leaf("   padded   ")))

    assert result.text == "padded"


def test_stripping_can_be_disabled() -> None:
    result = _normalizer(
        StructureConfig(collapse_whitespace=False, strip_block_text=False)
    ).normalize(_output(_leaf("  padded  ")))

    assert result.text == "  padded  "


# ---------------------------------------------------------------------------
# Separators
# ---------------------------------------------------------------------------


def test_blocks_are_joined_by_the_configured_separator() -> None:
    result = _normalizer(StructureConfig(block_separator="\n")).normalize(
        _output(_leaf("A"), _leaf("B"))
    )

    assert result.text == "A\nB"
    assert [(b.span.start, b.span.end) for b in result.blocks] == [(0, 1), (2, 3)]


def test_separator_gaps_contain_only_whitespace() -> None:
    result = _normalizer().normalize(_output(_leaf("A"), _leaf("B")))
    document = _as_raw_document(result)

    leaves = document.leaf_blocks()
    gap = result.text[leaves[0].span.end : leaves[1].span.start]

    assert gap == "\n\n"
    assert gap.isspace()


def test_no_separator_precedes_the_first_block() -> None:
    result = _normalizer().normalize(_output(_leaf("First")))

    assert result.text == "First"
    assert result.blocks[0].span.start == 0


# ---------------------------------------------------------------------------
# Empty leaves
# ---------------------------------------------------------------------------


def test_blank_paragraphs_receive_zero_length_spans() -> None:
    """Blank paragraphs are ordinary; they must not accumulate whitespace."""
    result = _normalizer().normalize(
        _output(_leaf("A"), _leaf("   "), _leaf("B"))
    )

    assert result.text == "A\n\nB"
    assert result.blocks[1].span.length == 0
    _as_raw_document(result)


def test_structural_markers_carry_zero_length_spans() -> None:
    result = _normalizer().normalize(
        _output(
            _leaf("Page one"),
            _leaf("", block_type=BlockType.PAGE_BREAK),
            _leaf("Page two"),
        )
    )
    document = _as_raw_document(result)

    marker = document.blocks[1]

    assert marker.span.length == 0
    assert document.block_text(marker) == ""


def test_many_consecutive_blank_blocks_do_not_accumulate_whitespace() -> None:
    result = _normalizer().normalize(
        _output(_leaf("A"), *[_leaf("") for _ in range(20)], _leaf("B"))
    )

    assert result.text == "A\n\nB"
    _as_raw_document(result)


def test_a_document_of_only_blank_blocks_is_empty() -> None:
    result = _normalizer().normalize(_output(_leaf(""), _leaf("  ")))

    assert result.text == ""
    assert all(b.span.length == 0 for b in result.blocks)
    _as_raw_document(result)


def test_blank_paragraphs_produce_no_warnings() -> None:
    result = _normalizer().normalize(_output(_leaf("A"), _leaf("   ")))

    assert result.warnings == ()


# ---------------------------------------------------------------------------
# Unicode normalization
# ---------------------------------------------------------------------------


def test_decomposed_text_is_composed_under_nfc() -> None:
    decomposed = "cafe\u0301"
    result = _normalizer().normalize(_output(_leaf(decomposed)))

    assert result.text == "café"
    assert len(result.text) == 4


def test_spans_reflect_post_normalization_lengths() -> None:
    """Measuring before normalization would leave every span one too long."""
    result = _normalizer().normalize(_output(_leaf("cafe\u0301"), _leaf("bar")))
    document = _as_raw_document(result)

    leaves = document.leaf_blocks()

    assert document.block_text(leaves[0]) == "café"
    assert leaves[0].span.end == 4
    assert document.block_text(leaves[1]) == "bar"


@pytest.mark.parametrize("form", list(TextNormalizationForm))
def test_every_normalization_form_produces_normalized_text(
    form: TextNormalizationForm,
) -> None:
    result = _normalizer(StructureConfig(normalization_form=form)).normalize(
        _output(_leaf("cafe\u0301 \ufb01n"), _leaf("na\u00efve"))
    )

    assert unicodedata.is_normalized(form.value, result.text)
    _as_raw_document(result)


def test_compatibility_normalization_precedes_collapsing() -> None:
    """Under NFKC a non-breaking space becomes collapsible; ordering matters."""
    result = _normalizer(
        StructureConfig(normalization_form=TextNormalizationForm.NFKC)
    ).normalize(_output(_leaf("a\u00a0\u00a0b")))

    assert result.text == "a b"


def test_normalization_is_reapplied_after_whitespace_removal() -> None:
    """Removing a space can join a base character to a combining mark."""
    result = _normalizer().normalize(_output(_leaf("  \u0301abc  ")))

    assert unicodedata.is_normalized("NFC", result.text)
    _as_raw_document(result)


# ---------------------------------------------------------------------------
# Attributes
# ---------------------------------------------------------------------------


def test_block_attributes_are_carried_through_unchanged() -> None:
    attributes = BlockAttributes(heading_level=2, page_number=7)
    result = _normalizer().normalize(
        _output(_leaf("Title", block_type=BlockType.HEADING, attributes=attributes))
    )

    assert result.blocks[0].attributes == attributes


def test_block_types_are_preserved() -> None:
    result = _normalizer().normalize(
        _output(
            _container(BlockType.QUOTE),
            _leaf("Quoted.", parent_index=0),
            _leaf("code()", block_type=BlockType.CODE),
        )
    )

    assert [b.block_type for b in result.blocks] == [
        BlockType.QUOTE,
        BlockType.PARAGRAPH,
        BlockType.CODE,
    ]


# ---------------------------------------------------------------------------
# Warnings
# ---------------------------------------------------------------------------


def test_container_yielding_no_text_is_reported() -> None:
    result = _normalizer().normalize(
        _output(_container(BlockType.TABLE), _leaf("", parent_index=0))
    )

    assert len(result.warnings) == 1
    warning = result.warnings[0]
    assert warning.code is ExtractionWarningCode.PARTIAL_EXTRACTION
    assert warning.block_id == "0"
    assert warning.detail["block_type"] == BlockType.TABLE.value


def test_containers_with_content_produce_no_warning() -> None:
    result = _normalizer().normalize(
        _output(_container(), _leaf("Body.", parent_index=0))
    )

    assert result.warnings == ()


def test_warnings_survive_into_the_canonical_document() -> None:
    result = _normalizer().normalize(
        _output(_container(BlockType.SLIDE), _leaf("", parent_index=0))
    )

    document = _as_raw_document(result)

    assert len(
        document.warnings_by_code(ExtractionWarningCode.PARTIAL_EXTRACTION)
    ) == 1


# ---------------------------------------------------------------------------
# Limits
# ---------------------------------------------------------------------------


def test_block_count_limit_is_enforced() -> None:
    normalizer = _normalizer(limits=ExtractionConfig(max_blocks=3))

    with pytest.raises(StructureParsingError, match="exceeding the configured"):
        normalizer.normalize(_output(*[_leaf(str(n)) for n in range(4)]))


def test_text_length_limit_is_enforced() -> None:
    normalizer = _normalizer(limits=ExtractionConfig(max_text_chars=10))

    with pytest.raises(StructureParsingError, match="Normalized text length"):
        normalizer.normalize(_output(_leaf("x" * 50)))


def test_limit_errors_name_the_extractor() -> None:
    normalizer = _normalizer(limits=ExtractionConfig(max_blocks=1))

    with pytest.raises(StructureParsingError) as excinfo:
        normalizer.normalize(
            _output(_leaf("a"), _leaf("b"), name="PdfExtractor")
        )

    assert excinfo.value.context["extractor_name"] == "PdfExtractor"


# ---------------------------------------------------------------------------
# Integrity verification
# ---------------------------------------------------------------------------


def test_integrity_verification_is_enabled_by_default() -> None:
    assert StructureConfig().validate_span_integrity is True


def test_integrity_verification_accepts_well_formed_output() -> None:
    strict = _normalizer(StructureConfig(validate_span_integrity=True))
    lax = _normalizer(StructureConfig(validate_span_integrity=False))
    output = _output(_container(), _leaf("A", parent_index=0), _leaf("B"))

    assert strict.normalize(output) == lax.normalize(output)


def test_non_normalized_assembly_is_failed_rather_than_repaired(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Renormalizing at document level would invalidate every computed span,
    so the document is quarantined instead.
    """
    normalizer = _normalizer()
    monkeypatch.setattr(
        structure_module.unicodedata,
        "is_normalized",
        lambda form, text: len(text) < 4,
    )

    with pytest.raises(SpanIntegrityError, match="not in NFC form"):
        normalizer.normalize(_output(_leaf("alpha"), _leaf("beta")))


def test_carriage_return_surviving_normalization_is_detected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    normalizer = _normalizer()
    monkeypatch.setattr(
        structure_module, "_LINE_ENDINGS", structure_module.re.compile(r"(?!x)x")
    )

    with pytest.raises(SpanIntegrityError, match="carriage returns"):
        normalizer.normalize(
            _output(_leaf("a\rb", block_type=BlockType.CODE))
        )


# ---------------------------------------------------------------------------
# Property tests
# ---------------------------------------------------------------------------

_TEXT = st.text(
    alphabet=st.characters(blacklist_categories=("Cs", "Cc")), max_size=40
)


@settings(max_examples=200)
@given(texts=st.lists(_TEXT, min_size=1, max_size=15))
def test_output_is_always_a_valid_raw_document(texts: list[str]) -> None:
    """Property: no flat block sequence can produce an invalid document."""
    result = _normalizer().normalize(_output(*[_leaf(text) for text in texts]))

    _as_raw_document(result)


@settings(max_examples=200)
@given(texts=st.lists(_TEXT, min_size=1, max_size=15))
def test_leaf_spans_always_recover_their_normalized_text(
    texts: list[str],
) -> None:
    """Property: every leaf span indexes exactly the text it represents."""
    normalizer = _normalizer()
    result = normalizer.normalize(_output(*[_leaf(text) for text in texts]))
    document = _as_raw_document(result)

    for block, extracted in zip(document.blocks, texts, strict=True):
        expected = normalizer._normalize_leaf_text(_leaf(extracted))
        assert document.block_text(block) == expected


@settings(max_examples=100)
@given(texts=st.lists(_TEXT, min_size=1, max_size=10))
def test_normalization_is_deterministic(texts: list[str]) -> None:
    """Property: identical input always yields byte-identical output."""
    output = _output(*[_leaf(text) for text in texts])
    normalizer = _normalizer()

    assert normalizer.normalize(output) == normalizer.normalize(output)


@settings(max_examples=100)
@given(
    texts=st.lists(_TEXT, min_size=1, max_size=8),
    heading=_TEXT,
)
def test_nested_output_is_always_valid(texts: list[str], heading: str) -> None:
    """Property: container envelopes always satisfy containment."""
    blocks = [
        _container(),
        _leaf(heading, block_type=BlockType.HEADING, parent_index=0),
        *[_leaf(text, parent_index=0) for text in texts],
    ]

    _as_raw_document(_normalizer().normalize(_output(*blocks)))


# ---------------------------------------------------------------------------
# Anti-pattern guards
# ---------------------------------------------------------------------------


def test_stage_is_synchronous_and_performs_no_io() -> None:
    import inspect

    source = inspect.getsource(structure_module)
    for forbidden in ("async def", "await ", "open(", "requests", "httpx"):
        assert forbidden not in source


def test_normalizer_holds_no_assignable_state() -> None:
    with pytest.raises(AttributeError):
        _normalizer().cache = {}  # type: ignore[attr-defined]


def test_module_declares_no_mutable_state() -> None:
    mutable = [
        name
        for name, value in vars(structure_module).items()
        if not name.startswith("__") and isinstance(value, (list, dict, set))
    ]
    assert mutable == []


def test_stage_declares_no_hardcoded_limits() -> None:
    """Every operational limit originates from injected configuration."""
    import inspect

    source = inspect.getsource(StructureNormalizer)

    assert "self._config.max_block_depth" in source
    assert "self._limits.max_blocks" in source
    assert "self._limits.max_text_chars" in source
    assert "> 0.8" not in source


def test_stage_imports_no_concrete_extractors_or_plugins() -> None:
    import inspect

    source = inspect.getsource(structure_module)
    for forbidden in (
        "from ptfm.core.ingestion.extractors",
        "from ptfm.core.ingestion.connectors",
        "from ptfm.plugins",
    ):
        assert forbidden not in source