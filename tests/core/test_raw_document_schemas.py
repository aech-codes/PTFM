"""Unit and property tests for the RawDocument canonical schema family."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from hypothesis import given, strategies as st
from pydantic import ValidationError

from ptfm.core.schemas import (
    PTFM_DOCUMENT_NAMESPACE,
    RAW_DOCUMENT_SCHEMA_VERSION,
    BlockAttributes,
    BlockIndex,
    BlockType,
    DocumentBlock,
    DocumentMetadata,
    ExtractionWarning,
    ExtractionWarningCode,
    RawDocument,
    SourceDescriptor,
    TextSpan,
    derive_document_id,
)

FIXED_TIME = datetime(2026, 1, 1, tzinfo=UTC)


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def _source(uri: str = "file:///corpus/a.txt") -> SourceDescriptor:
    return SourceDescriptor(
        source_uri=uri,
        scheme="file",
        connector_name="LocalFilesystemConnector",
        acquired_at=FIXED_TIME,
        size_bytes=128,
        detected_media_type="text/plain",
    )


def _metadata(text: str) -> DocumentMetadata:
    return DocumentMetadata(char_count=len(text), word_count=len(text.split()))


def _document(
    text: str,
    blocks: tuple[DocumentBlock, ...] = (),
    **overrides: object,
) -> RawDocument:
    payload: dict[str, object] = {
        "document_id": derive_document_id("file:///corpus/a.txt", "abc123"),
        "source": _source(),
        "metadata": _metadata(text),
        "text": text,
        "blocks": blocks,
        "content_hash": "abc123",
        "text_hash": "def456",
        "ingested_at": FIXED_TIME,
    }
    payload.update(overrides)
    return RawDocument(**payload)  # type: ignore[arg-type]


def _block(
    block_id: str,
    start: int,
    end: int,
    *,
    block_type: BlockType = BlockType.PARAGRAPH,
    parent_id: str | None = None,
    depth: int = 0,
    ordinal: int = 0,
) -> DocumentBlock:
    return DocumentBlock(
        block_id=block_id,
        block_type=block_type,
        span=TextSpan(start=start, end=end),
        parent_id=parent_id,
        depth=depth,
        ordinal=ordinal,
    )


# ---------------------------------------------------------------------------
# Identity
# ---------------------------------------------------------------------------


def test_document_id_is_deterministic() -> None:
    first = derive_document_id("file:///a.pdf", "hash-1")
    second = derive_document_id("file:///a.pdf", "hash-1")

    assert first == second


@pytest.mark.parametrize(
    ("uri", "digest"),
    [("file:///b.pdf", "hash-1"), ("file:///a.pdf", "hash-2")],
)
def test_document_id_varies_with_origin_and_payload(uri: str, digest: str) -> None:
    baseline = derive_document_id("file:///a.pdf", "hash-1")

    assert derive_document_id(uri, digest) != baseline


def test_document_id_separator_prevents_field_ambiguity() -> None:
    """'ab' + 'c' must not collide with 'a' + 'bc'."""
    assert derive_document_id("ab", "c") != derive_document_id("a", "bc")


def test_namespace_is_stable() -> None:
    assert str(PTFM_DOCUMENT_NAMESPACE) == str(PTFM_DOCUMENT_NAMESPACE)
    assert PTFM_DOCUMENT_NAMESPACE.version == 5


@pytest.mark.parametrize(("uri", "digest"), [("", "h"), ("file:///a", "")])
def test_document_id_rejects_empty_components(uri: str, digest: str) -> None:
    with pytest.raises(ValueError):
        derive_document_id(uri, digest)


# ---------------------------------------------------------------------------
# Immutability
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "instance",
    [
        TextSpan(start=0, end=1),
        BlockAttributes(heading_level=1),
        _block("b0", 0, 1),
        _source(),
        _metadata("abc"),
        ExtractionWarning(code=ExtractionWarningCode.EMPTY_PAGE, message="m"),
    ],
)
def test_models_are_frozen(instance: object) -> None:
    with pytest.raises(ValidationError):
        object.__getattribute__(instance, "__class__")
        instance.__setattr__("__dummy_never_a_field__", 1)  # type: ignore[attr-defined]


def test_raw_document_is_frozen() -> None:
    document = _document("hello")

    with pytest.raises(ValidationError):
        document.text = "mutated"  # type: ignore[misc]


def test_unknown_fields_are_rejected() -> None:
    with pytest.raises(ValidationError):
        TextSpan(start=0, end=1, unexpected="x")  # type: ignore[call-arg]


def test_sequence_fields_are_tuples() -> None:
    document = _document("hi", (_block("b0", 0, 2),))

    assert isinstance(document.blocks, tuple)
    assert isinstance(document.metadata.authors, tuple)
    assert isinstance(document.warnings, tuple)


# ---------------------------------------------------------------------------
# TextSpan
# ---------------------------------------------------------------------------


def test_span_rejects_inverted_range() -> None:
    with pytest.raises(ValidationError):
        TextSpan(start=5, end=2)


def test_span_allows_zero_length() -> None:
    assert TextSpan(start=3, end=3).length == 0


def test_span_rejects_negative_offsets() -> None:
    with pytest.raises(ValidationError):
        TextSpan(start=-1, end=2)


# ---------------------------------------------------------------------------
# Block-level validation
# ---------------------------------------------------------------------------


def test_root_block_must_have_zero_depth() -> None:
    with pytest.raises(ValidationError):
        _block("b0", 0, 1, depth=2)


def test_child_block_must_have_nonzero_depth() -> None:
    with pytest.raises(ValidationError):
        _block("b1", 0, 1, parent_id="b0", depth=0)


# ---------------------------------------------------------------------------
# Document text invariants
# ---------------------------------------------------------------------------


def test_carriage_returns_are_rejected() -> None:
    with pytest.raises(ValidationError, match="carriage returns"):
        _document("line one\r\nline two")


def test_char_count_must_match_text() -> None:
    text = "hello"
    with pytest.raises(ValidationError, match="char_count"):
        _document(text, metadata=DocumentMetadata(char_count=99, word_count=1))


def test_empty_document_is_valid() -> None:
    document = _document("")

    assert document.blocks == ()
    assert document.leaf_blocks() == ()
    assert document.schema_version == RAW_DOCUMENT_SCHEMA_VERSION


# ---------------------------------------------------------------------------
# Tree invariants
# ---------------------------------------------------------------------------


def test_valid_nested_tree_is_accepted() -> None:
    text = "Title\n\nBody text."
    blocks = (
        _block("b0", 0, 17, block_type=BlockType.SECTION),
        _block(
            "b0.0", 0, 5, block_type=BlockType.HEADING, parent_id="b0", depth=1
        ),
        _block("b0.1", 7, 17, parent_id="b0", depth=1, ordinal=1),
    )

    document = _document(text, blocks)

    assert len(document.leaf_blocks()) == 2
    assert document.block_text(document.blocks[1]) == "Title"
    assert document.block_text(document.blocks[2]) == "Body text."


def test_duplicate_block_ids_are_rejected() -> None:
    text = "ab"
    blocks = (_block("dup", 0, 1), _block("dup", 1, 2, ordinal=1))

    with pytest.raises(ValidationError, match="Duplicate block_id"):
        _document(text, blocks)


def test_forward_parent_reference_is_rejected() -> None:
    """A parent appearing after its child breaks pre-order and would allow cycles."""
    text = "ab"
    blocks = (
        _block("child", 0, 1, parent_id="parent", depth=1),
        _block("parent", 0, 2, block_type=BlockType.SECTION, ordinal=1),
    )

    with pytest.raises(ValidationError, match="pre-order"):
        _document(text, blocks)


def test_incorrect_depth_is_rejected() -> None:
    text = "ab"
    blocks = (
        _block("b0", 0, 2, block_type=BlockType.SECTION),
        _block("b0.0", 0, 2, parent_id="b0", depth=3),
    )

    with pytest.raises(ValidationError, match="expected 1"):
        _document(text, blocks)


def test_noncontiguous_sibling_ordinals_are_rejected() -> None:
    text = "a b"
    blocks = (_block("b0", 0, 1), _block("b1", 2, 3, ordinal=7))

    with pytest.raises(ValidationError, match="contiguous"):
        _document(text, blocks)


def test_leaf_type_may_not_have_children() -> None:
    text = "ab"
    blocks = (
        _block("b0", 0, 2, block_type=BlockType.PARAGRAPH),
        _block("b0.0", 0, 2, parent_id="b0", depth=1),
    )

    with pytest.raises(ValidationError, match="may not have children"):
        _document(text, blocks)


# ---------------------------------------------------------------------------
# Span invariants (Evidence-First guarantees)
# ---------------------------------------------------------------------------


def test_span_beyond_text_is_rejected() -> None:
    with pytest.raises(ValidationError, match="exceeds"):
        _document("abc", (_block("b0", 0, 99),))


def test_child_span_must_lie_within_parent_span() -> None:
    text = "abcdef"
    blocks = (
        _block("b0", 0, 3, block_type=BlockType.SECTION),
        _block("b0.0", 2, 6, parent_id="b0", depth=1),
    )

    with pytest.raises(ValidationError, match="not contained"):
        _document(text, blocks)


def test_overlapping_leaf_spans_are_rejected() -> None:
    text = "abcdef"
    blocks = (_block("b0", 0, 4), _block("b1", 2, 6, ordinal=1))

    with pytest.raises(ValidationError, match="overlapping"):
        _document(text, blocks)


def test_uncovered_content_between_leaves_is_rejected() -> None:
    """The core Evidence-First guard: no text may hide outside the block tree."""
    text = "one HIDDEN two"
    blocks = (_block("b0", 0, 3), _block("b1", 11, 14, ordinal=1))

    with pytest.raises(ValidationError, match="not covered by any block"):
        _document(text, blocks)


def test_whitespace_separators_between_leaves_are_permitted() -> None:
    text = "one\n\ntwo"
    blocks = (_block("b0", 0, 3), _block("b1", 5, 8, ordinal=1))

    document = _document(text, blocks)

    assert document.block_text(document.blocks[0]) == "one"
    assert document.block_text(document.blocks[1]) == "two"


def test_uncovered_trailing_content_is_rejected() -> None:
    text = "one TRAILING"
    blocks = (_block("b0", 0, 3),)

    with pytest.raises(ValidationError, match="trailing text"):
        _document(text, blocks)


def test_zero_length_leaf_is_permitted() -> None:
    """PAGE_BREAK carries structure but no characters."""
    text = "one\ntwo"
    blocks = (
        _block("b0", 0, 3),
        _block("b1", 4, 4, block_type=BlockType.PAGE_BREAK, ordinal=1),
        _block("b2", 4, 7, ordinal=2),
    )

    document = _document(text, blocks)

    assert document.block_text(document.blocks[1]) == ""


# ---------------------------------------------------------------------------
# Accessors
# ---------------------------------------------------------------------------


def test_accessors_agree_on_tree_shape() -> None:
    text = "Head\n\nBody"
    blocks = (
        _block("b0", 0, 10, block_type=BlockType.SECTION),
        _block("b0.0", 0, 4, block_type=BlockType.HEADING, parent_id="b0", depth=1),
        _block("b0.1", 6, 10, parent_id="b0", depth=1, ordinal=1),
    )
    document = _document(text, blocks)

    assert [b.block_id for b in document.root_blocks()] == ["b0"]
    assert [b.block_id for b in document.leaf_blocks()] == ["b0.0", "b0.1"]
    assert document.find_block("b0.1") is blocks[2]
    assert document.find_block("absent") is None
    assert list(document.iter_blocks_with_text()) == [
        (blocks[1], "Head"),
        (blocks[2], "Body"),
    ]


def test_block_index_matches_linear_accessors() -> None:
    text = "Head\n\nBody"
    blocks = (
        _block("b0", 0, 10, block_type=BlockType.SECTION),
        _block("b0.0", 0, 4, block_type=BlockType.HEADING, parent_id="b0", depth=1),
        _block("b0.1", 6, 10, parent_id="b0", depth=1, ordinal=1),
    )
    document = _document(text, blocks)

    index: BlockIndex = document.build_index()

    assert index.get("b0.0") is document.find_block("b0.0")
    assert index.get("absent") is None
    assert [b.block_id for b in index.children_of("b0")] == ["b0.0", "b0.1"]
    assert [b.block_id for b in index.children_of(None)] == ["b0"]
    assert index.children_of("b0.0") == ()


def test_warnings_filter_by_code() -> None:
    warnings = (
        ExtractionWarning(code=ExtractionWarningCode.EMPTY_PAGE, message="p3"),
        ExtractionWarning(code=ExtractionWarningCode.OCR_APPLIED, message="p4"),
        ExtractionWarning(code=ExtractionWarningCode.EMPTY_PAGE, message="p7"),
    )
    document = _document("x", warnings=warnings)

    selected = document.warnings_by_code(ExtractionWarningCode.EMPTY_PAGE)

    assert [w.message for w in selected] == ["p3", "p7"]


# ---------------------------------------------------------------------------
# Domain-agnosticism guard
# ---------------------------------------------------------------------------


def test_format_properties_reject_structured_values() -> None:
    """String-only values block structured domain objects from leaking through."""
    with pytest.raises(ValidationError):
        DocumentMetadata(
            char_count=0,
            word_count=0,
            format_properties={"citation": {"court": "SCOTUS"}},  # type: ignore[dict-item]
        )


def test_metadata_exposes_no_numeric_confidence_field() -> None:
    forbidden = {"confidence", "score", "probability", "language_confidence"}

    assert not forbidden & set(DocumentMetadata.model_fields)


# ---------------------------------------------------------------------------
# Round-trip and property tests
# ---------------------------------------------------------------------------


def test_json_round_trip_preserves_structure() -> None:
    text = "Head\n\nBody"
    blocks = (
        _block("b0", 0, 10, block_type=BlockType.SECTION),
        _block("b0.0", 0, 4, block_type=BlockType.HEADING, parent_id="b0", depth=1),
        _block("b0.1", 6, 10, parent_id="b0", depth=1, ordinal=1),
    )
    original = _document(text, blocks)

    restored = RawDocument.model_validate_json(original.model_dump_json())

    assert restored == original
    assert restored.blocks == original.blocks


@given(
    segments=st.lists(
        st.text(alphabet=st.characters(blacklist_categories=("Cc", "Zs")), min_size=1),
        min_size=1,
        max_size=25,
    )
)
def test_leaf_spans_always_recover_their_segments(segments: list[str]) -> None:
    """Property: for any paragraph sequence, every leaf span recovers its text."""
    separator = "\n\n"
    text = separator.join(segments)

    blocks: list[DocumentBlock] = []
    cursor = 0
    for ordinal, segment in enumerate(segments):
        blocks.append(_block(f"b{ordinal}", cursor, cursor + len(segment), ordinal=ordinal))
        cursor += len(segment) + len(separator)

    document = _document(text, tuple(blocks))

    recovered = [document.block_text(block) for block in document.leaf_blocks()]
    assert recovered == segments


@given(
    segments=st.lists(
        st.text(alphabet=st.characters(blacklist_categories=("Cc", "Zs")), min_size=1),
        min_size=2,
        max_size=10,
    ),
    shift=st.integers(min_value=1, max_value=3),
)
def test_span_drift_is_always_detected(segments: list[str], shift: int) -> None:
    """Property: any offset drift in a leaf span fails validation."""
    separator = "\n\n"
    text = separator.join(segments)

    blocks: list[DocumentBlock] = []
    cursor = 0
    for ordinal, segment in enumerate(segments):
        blocks.append(_block(f"b{ordinal}", cursor, cursor + len(segment), ordinal=ordinal))
        cursor += len(segment) + len(separator)

    drifted = list(blocks)
    last = drifted[-1]
    drifted[-1] = last.model_copy(
        update={"span": TextSpan(start=last.span.start + shift, end=last.span.end)}
    )

    with pytest.raises(ValidationError):
        _document(text, tuple(drifted))