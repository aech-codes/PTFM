"""
Unit and property tests for the planning and assembly stages.

File:
    tests/core/knowledge/test_planning_assembly.py

Emphasis
--------
    These two stages manufacture the Evidence-First guarantee. Planning
    verifies that a plan describes its document truthfully; assembly produces
    text by slicing at the span, so the characters stored and the offsets
    recorded originate from one operation.

    The tests that matter most are therefore the ones proving that no plan and
    no document can produce a node whose text and span disagree.
"""

from __future__ import annotations

import inspect
from datetime import UTC, datetime
from typing import Any, Mapping, Sequence

import pytest
from hypothesis import given, settings, strategies as st

from ptfm.core.knowledge.config import ValidationConfig
from ptfm.core.knowledge.exceptions import (
    BlockReferenceError,
    CoverageError,
    InvalidChunkPlanError,
    UnsplittableBlockError,
)
from ptfm.core.knowledge.interfaces import ChunkingStrategy
from ptfm.core.knowledge.models import ChunkPlan, PlannedChunk
from ptfm.core.knowledge.pipeline import assembly as assembly_module
from ptfm.core.knowledge.pipeline import planning as planning_module
from ptfm.core.knowledge.pipeline.assembly import (
    NodeAssembler,
    derive_heading_paths,
)
from ptfm.core.knowledge.pipeline.planning import ChunkPlanner
from ptfm.core.schemas import (
    BlockAttributes,
    BlockType,
    DocumentBlock,
    DocumentMetadata,
    NodeType,
    RawDocument,
    SourceDescriptor,
    TextSpan,
    derive_document_id,
)

NOW = datetime(2026, 8, 1, tzinfo=UTC)
FINGERPRINT = "1:seg-fp-001"

TEXT = "Interest Rates\n\nThe applicable rate is 4.5 percent.\n\nSee the annex."


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


class _FixedClock:
    def now(self) -> datetime:
        return NOW


def _block(
    block_id: str,
    start: int,
    end: int,
    *,
    block_type: BlockType = BlockType.PARAGRAPH,
    parent_id: str | None = None,
    depth: int = 0,
    ordinal: int = 0,
    attributes: BlockAttributes | None = None,
) -> DocumentBlock:
    return DocumentBlock(
        block_id=block_id,
        block_type=block_type,
        span=TextSpan(start=start, end=end),
        parent_id=parent_id,
        depth=depth,
        ordinal=ordinal,
        attributes=attributes or BlockAttributes(),
    )


def _document(
    text: str = TEXT,
    blocks: Sequence[DocumentBlock] | None = None,
) -> RawDocument:
    if blocks is None:
        blocks = (
            _block(
                "0",
                0,
                14,
                block_type=BlockType.HEADING,
                attributes=BlockAttributes(heading_level=1),
            ),
            _block("1", 16, 50, ordinal=1),
            _block("2", 52, 68, ordinal=2),
        )
    return RawDocument(
        document_id=derive_document_id("file:///corpus/a.pdf", "hash-abc"),
        source=SourceDescriptor(
            source_uri="file:///corpus/a.pdf",
            scheme="file",
            connector_name="LocalFilesystemConnector",
            extractor_name="PdfExtractor",
            acquired_at=NOW,
            size_bytes=4096,
            detected_media_type="application/pdf",
        ),
        metadata=DocumentMetadata(char_count=len(text), word_count=len(text.split())),
        text=text,
        blocks=tuple(blocks),
        content_hash="hash-abc",
        text_hash="text-hash-abc",
        ingested_at=NOW,
    )


class _Strategy(ChunkingStrategy):
    """Strategy returning a plan supplied per test."""

    def __init__(self, plan_factory: Any = None, name: str = "test_strategy") -> None:
        self._name = name
        self._plan_factory = plan_factory

    @property
    def name(self) -> str:
        return self._name

    def plan(self, document: RawDocument) -> Any:
        if self._plan_factory is None:
            return _one_chunk_per_leaf(document, self._name)
        return self._plan_factory(document)

    def fingerprint_components(self) -> Mapping[str, str]:
        return {"strategy": self._name}


def _one_chunk_per_leaf(document: RawDocument, strategy: str) -> ChunkPlan:
    """Trivially correct plan: one chunk per content-bearing leaf block."""
    return ChunkPlan(
        document_id=document.document_id,
        strategy_name=strategy,
        chunks=tuple(
            PlannedChunk(span=block.span, block_ids=(block.block_id,))
            for block in document.leaf_blocks()
            if block.span.length > 0
        ),
    )


def _planner(
    strategy: ChunkingStrategy | None = None, **config: Any
) -> ChunkPlanner:
    return ChunkPlanner(strategy or _Strategy(), ValidationConfig(**config))


def _assembler(**config: Any) -> NodeAssembler:
    return NodeAssembler(ValidationConfig(**config), _FixedClock())


def _plan_and_assemble(
    document: RawDocument, strategy: ChunkingStrategy | None = None
) -> Any:
    plan = _planner(strategy).plan(document)
    return _assembler().assemble(document, plan, FINGERPRINT)


# ---------------------------------------------------------------------------
# The Evidence-First guarantee
# ---------------------------------------------------------------------------


def test_every_assembled_node_verifies_against_its_document() -> None:
    """The guarantee both stages exist to manufacture."""
    document = _document()

    nodes = _plan_and_assemble(document)

    assert nodes
    for node in nodes:
        assert node.verify_against(document.text) is True


def test_node_text_is_sliced_from_the_document() -> None:
    """
    The characters stored and the offsets recorded originate from one
    operation, so they cannot disagree.
    """
    document = _document()

    nodes = _plan_and_assemble(document)

    for node in nodes:
        assert node.text == document.text[node.span.start : node.span.end]


def test_assembly_derives_provenance_from_the_document() -> None:
    document = _document()

    node = _plan_and_assemble(document)[0]

    assert node.document_id == document.document_id
    assert node.source is document.source
    assert node.source_text_hash == document.text_hash
    assert node.source_schema_version == document.schema_version
    assert node.segmentation_fingerprint == FINGERPRINT


def test_the_fingerprint_participates_in_node_identity() -> None:
    document = _document()
    plan = _planner().plan(document)
    assembler = _assembler()

    first = assembler.assemble(document, plan, "1:fp-a")
    second = assembler.assemble(document, plan, "1:fp-b")

    assert first[0].node_id != second[0].node_id


def test_assembly_is_deterministic() -> None:
    document = _document()
    plan = _planner().plan(document)
    assembler = _assembler()

    assert assembler.assemble(document, plan, FINGERPRINT) == assembler.assemble(
        document, plan, FINGERPRINT
    )


# ---------------------------------------------------------------------------
# Planning: coverage
# ---------------------------------------------------------------------------


def test_uncovered_content_is_rejected() -> None:
    """
    Text absent from every node cannot be retrieved, so a claim it would have
    supported becomes unverifiable with nothing reporting an error.
    """
    document = _document()

    def _skip_middle(doc: RawDocument) -> ChunkPlan:
        leaves = [b for b in doc.leaf_blocks() if b.span.length > 0]
        return ChunkPlan(
            document_id=doc.document_id,
            strategy_name="skipper",
            chunks=(
                PlannedChunk(span=leaves[0].span, block_ids=(leaves[0].block_id,)),
                PlannedChunk(span=leaves[2].span, block_ids=(leaves[2].block_id,)),
            ),
        )

    with pytest.raises(CoverageError) as excinfo:
        _planner(_Strategy(_skip_middle)).plan(document)

    assert excinfo.value.uncovered_start == 14
    assert "unverifiable" in excinfo.value.message


def test_uncovered_trailing_content_is_rejected() -> None:
    document = _document()

    def _drop_last(doc: RawDocument) -> ChunkPlan:
        leaves = [b for b in doc.leaf_blocks() if b.span.length > 0]
        return ChunkPlan(
            document_id=doc.document_id,
            strategy_name="dropper",
            chunks=tuple(
                PlannedChunk(span=b.span, block_ids=(b.block_id,))
                for b in leaves[:-1]
            ),
        )

    with pytest.raises(CoverageError):
        _planner(_Strategy(_drop_last)).plan(document)


def test_whitespace_between_chunks_is_permitted() -> None:
    """A separator belongs to neither of the chunks it separates."""
    document = _document()

    plan = _planner().plan(document)

    assert plan.chunk_count == 3


def test_coverage_checking_can_be_disabled() -> None:
    document = _document()

    def _drop_last(doc: RawDocument) -> ChunkPlan:
        leaves = [b for b in doc.leaf_blocks() if b.span.length > 0]
        return ChunkPlan(
            document_id=doc.document_id,
            strategy_name="dropper",
            chunks=tuple(
                PlannedChunk(span=b.span, block_ids=(b.block_id,))
                for b in leaves[:-1]
            ),
        )

    planner = _planner(_Strategy(_drop_last), require_full_coverage=False)

    assert planner.plan(document).chunk_count == 2


def test_coverage_is_required_by_default() -> None:
    assert ValidationConfig().require_full_coverage is True


# ---------------------------------------------------------------------------
# Planning: block alignment
# ---------------------------------------------------------------------------


def test_a_span_disagreeing_with_its_blocks_is_rejected() -> None:
    """
    A citation would mean one thing or the other depending on which statement
    a consumer trusted.
    """
    document = _document()

    def _misaligned(doc: RawDocument) -> ChunkPlan:
        leaves = [b for b in doc.leaf_blocks() if b.span.length > 0]
        return ChunkPlan(
            document_id=doc.document_id,
            strategy_name="misaligned",
            chunks=(
                PlannedChunk(
                    span=TextSpan(start=0, end=68), block_ids=(leaves[0].block_id,)
                ),
            ),
        )

    with pytest.raises(InvalidChunkPlanError, match="outside|does not name"):
        _planner(_Strategy(_misaligned)).plan(document)


def test_a_chunk_covering_an_unnamed_leaf_is_rejected() -> None:
    """
    The unnamed leaf's text would be present in the node while absent from its
    structural provenance.
    """
    document = _document()

    def _partial(doc: RawDocument) -> ChunkPlan:
        leaves = [b for b in doc.leaf_blocks() if b.span.length > 0]
        # Span lies within the extent of the named blocks, so only the
        # skipped middle leaf makes the chunk invalid.
        return ChunkPlan(
            document_id=doc.document_id,
            strategy_name="partial",
            chunks=(
                PlannedChunk(
                    span=TextSpan(start=0, end=68),
                    block_ids=(leaves[0].block_id, leaves[2].block_id),
                ),
            ),
        )

    with pytest.raises(InvalidChunkPlanError, match="does not name"):
        _planner(_Strategy(_partial)).plan(document)


def test_a_container_block_reference_is_rejected() -> None:
    """
    A container's span is an envelope over its descendants, so naming one
    makes the block reference ambiguous.
    """
    blocks = (
        _block("0", 0, 14, block_type=BlockType.SECTION),
        _block("0.0", 0, 14, parent_id="0", depth=1),
    )
    document = _document("Interest Rates", blocks)

    def _names_container(doc: RawDocument) -> ChunkPlan:
        return ChunkPlan(
            document_id=doc.document_id,
            strategy_name="container",
            chunks=(
                PlannedChunk(span=TextSpan(start=0, end=14), block_ids=("0",)),
            ),
        )

    with pytest.raises(InvalidChunkPlanError, match="container rather than a leaf"):
        _planner(_Strategy(_names_container)).plan(document)


def test_alignment_checking_can_be_disabled() -> None:
    document = _document()

    def _misaligned(doc: RawDocument) -> ChunkPlan:
        leaves = [b for b in doc.leaf_blocks() if b.span.length > 0]
        return ChunkPlan(
            document_id=doc.document_id,
            strategy_name="misaligned",
            chunks=(
                PlannedChunk(
                    span=TextSpan(start=0, end=68),
                    block_ids=tuple(b.block_id for b in leaves),
                ),
            ),
        )

    planner = _planner(_Strategy(_misaligned), verify_block_alignment=False)

    assert planner.plan(document).chunk_count == 1


def test_a_block_may_be_divided_internally() -> None:
    """
    A block larger than the size budget is divided into several chunks, each
    naming that block. Requiring the span to equal the block extent would make
    the default configuration unable to plan any document containing one.
    """
    text = "x" * 90
    document = _document(text, (_block("0", 0, 90),))

    def _thirds(doc: RawDocument) -> ChunkPlan:
        return ChunkPlan(
            document_id=doc.document_id,
            strategy_name="splitter",
            chunks=tuple(
                PlannedChunk(
                    span=TextSpan(start=start, end=start + 30), block_ids=("0",)
                )
                for start in (0, 30, 60)
            ),
        )

    nodes = _plan_and_assemble(document, _Strategy(_thirds))

    assert len(nodes) == 3
    assert all(node.block_ids == ("0",) for node in nodes)
    assert "".join(node.text for node in nodes) == text
    for node in nodes:
        assert node.verify_against(document.text) is True


def test_a_chunk_naming_a_block_it_does_not_touch_is_rejected() -> None:
    """The node would claim structural provenance for text it does not
    contain."""
    document = _document()

    def _spurious(doc: RawDocument) -> ChunkPlan:
        leaves = [b for b in doc.leaf_blocks() if b.span.length > 0]
        return ChunkPlan(
            document_id=doc.document_id,
            strategy_name="spurious",
            chunks=(
                PlannedChunk(
                    span=leaves[0].span,
                    block_ids=(leaves[0].block_id, leaves[2].block_id),
                ),
            ),
        )

    with pytest.raises(InvalidChunkPlanError, match="does not touch"):
        _planner(_Strategy(_spurious)).plan(document)


# ---------------------------------------------------------------------------
# Planning: structural validation
# ---------------------------------------------------------------------------


def test_a_span_beyond_the_document_is_rejected() -> None:
    document = _document()

    def _overrun(doc: RawDocument) -> ChunkPlan:
        return ChunkPlan(
            document_id=doc.document_id,
            strategy_name="overrun",
            chunks=(
                PlannedChunk(span=TextSpan(start=0, end=9_999), block_ids=("0",)),
            ),
        )

    with pytest.raises(InvalidChunkPlanError, match="beyond the document"):
        _planner(_Strategy(_overrun)).plan(document)


def test_an_unknown_block_reference_is_rejected() -> None:
    document = _document()

    def _phantom(doc: RawDocument) -> ChunkPlan:
        return ChunkPlan(
            document_id=doc.document_id,
            strategy_name="phantom",
            chunks=(
                PlannedChunk(span=TextSpan(start=0, end=14), block_ids=("absent",)),
            ),
        )

    with pytest.raises(InvalidChunkPlanError, match="does not exist"):
        _planner(_Strategy(_phantom)).plan(document)


def test_a_plan_for_another_document_is_rejected() -> None:
    """Nodes built from it would carry provenance pointing elsewhere."""
    document = _document()

    def _foreign(doc: RawDocument) -> ChunkPlan:
        return ChunkPlan(
            document_id="a-different-document",
            strategy_name="foreign",
            chunks=(),
        )

    with pytest.raises(InvalidChunkPlanError, match="different document"):
        _planner(_Strategy(_foreign)).plan(document)


def test_a_non_plan_return_is_rejected() -> None:
    document = _document()

    with pytest.raises(InvalidChunkPlanError, match="not a ChunkPlan"):
        _planner(_Strategy(lambda doc: "not a plan")).plan(document)


def test_an_excessive_node_count_is_rejected() -> None:
    document = _document()
    planner = _planner(max_nodes_per_document=2)

    with pytest.raises(InvalidChunkPlanError, match="exceeding the configured"):
        planner.plan(document)


def test_a_strategy_exception_is_wrapped_with_its_cause() -> None:
    def _explode(doc: RawDocument) -> ChunkPlan:
        raise RuntimeError("internal strategy failure")

    with pytest.raises(InvalidChunkPlanError) as excinfo:
        _planner(_Strategy(_explode)).plan(_document())

    assert isinstance(excinfo.value.__cause__, RuntimeError)


def test_a_knowledge_exception_passes_through_enriched() -> None:
    """Strategies raise knowledge exceptions directly; those are not
    rewrapped."""

    def _unsplittable(doc: RawDocument) -> ChunkPlan:
        raise UnsplittableBlockError(block_id="1", block_chars=9_000, max_chars=2_000)

    with pytest.raises(UnsplittableBlockError) as excinfo:
        _planner(_Strategy(_unsplittable)).plan(_document())

    assert excinfo.value.document_id is not None


# ---------------------------------------------------------------------------
# Planning: size is not enforced
# ---------------------------------------------------------------------------


def test_chunk_size_is_not_enforced_by_the_planner() -> None:
    """
    Size is the strategy's contract. An oversized chunk may be a defect, or a
    single unsplittable block the configuration permits to pass through whole,
    and the planner cannot distinguish the two.
    """
    source = inspect.getsource(planning_module)

    assert "max_chunk_chars" not in source
    assert "min_chunk_chars" not in source


def test_an_oversized_chunk_is_accepted() -> None:
    long_text = "x" * 5_000
    document = _document(long_text, (_block("0", 0, 5_000),))

    nodes = _plan_and_assemble(document)

    assert len(nodes) == 1
    assert nodes[0].char_count == 5_000


# ---------------------------------------------------------------------------
# Empty documents
# ---------------------------------------------------------------------------


def test_an_empty_document_yields_an_empty_plan() -> None:
    """A document with no content is a skip, not a failure."""
    document = _document("", ())

    plan = _planner().plan(document)

    assert plan.is_empty is True


def test_a_whitespace_only_document_yields_an_empty_plan() -> None:
    document = _document("   \n\n  ", (_block("0", 0, 7),))

    assert _planner().plan(document).is_empty is True


def test_an_empty_plan_assembles_to_no_nodes() -> None:
    document = _document("", ())
    plan = _planner().plan(document)

    assert _assembler().assemble(document, plan, FINGERPRINT) == ()


# ---------------------------------------------------------------------------
# Assembly: node type derivation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("block_type", "expected"),
    [
        (BlockType.PARAGRAPH, NodeType.PROSE),
        (BlockType.HEADING, NodeType.HEADING),
        (BlockType.CODE, NodeType.CODE),
        (BlockType.QUOTE, NodeType.QUOTE),
        (BlockType.TABLE_CELL, NodeType.TABLE_REGION),
        (BlockType.TABLE_ROW, NodeType.TABLE_REGION),
        (BlockType.LIST_ITEM, NodeType.LIST_REGION),
        (BlockType.CAPTION, NodeType.FIGURE_REGION),
    ],
)
def test_node_type_derives_from_the_covered_blocks(
    block_type: BlockType, expected: NodeType
) -> None:
    """
    Derived from the document rather than declared by a strategy, so the label
    is a fact about the source rather than an assertion that could be wrong.
    """
    document = _document("Some content here", (_block("0", 0, 17, block_type=block_type),))

    nodes = _plan_and_assemble(document)

    assert nodes[0].node_type is expected


def test_a_chunk_covering_several_families_is_mixed() -> None:
    """
    A node spanning the end of a table and the start of the following prose is
    neither; reporting a dominant type would assert a character it does not
    uniformly have.
    """
    text = "Cell\n\nProse."
    blocks = (
        _block("0", 0, 4, block_type=BlockType.TABLE_CELL),
        _block("1", 6, 12, ordinal=1),
    )
    document = _document(text, blocks)

    def _one_chunk(doc: RawDocument) -> ChunkPlan:
        return ChunkPlan(
            document_id=doc.document_id,
            strategy_name="single",
            chunks=(
                PlannedChunk(
                    span=TextSpan(start=0, end=12), block_ids=("0", "1")
                ),
            ),
        )

    nodes = _plan_and_assemble(document, _Strategy(_one_chunk))

    assert nodes[0].node_type is NodeType.MIXED


def test_a_strategy_cannot_declare_a_node_type() -> None:
    assert "node_type" not in PlannedChunk.model_fields


# ---------------------------------------------------------------------------
# Assembly: heading paths
# ---------------------------------------------------------------------------


def test_headings_are_recovered_for_flat_documents() -> None:
    """
    Word processing documents and lightweight markup emit headings as
    siblings, not as ancestors.
    """
    text = "Part I\n\nBody one.\n\nPart II\n\nBody two."
    blocks = (
        _block("0", 0, 6, block_type=BlockType.HEADING,
               attributes=BlockAttributes(heading_level=1)),
        _block("1", 8, 17, ordinal=1),
        _block("2", 19, 26, block_type=BlockType.HEADING, ordinal=2,
               attributes=BlockAttributes(heading_level=1)),
        _block("3", 28, 37, ordinal=3),
    )
    document = _document(text, blocks)

    paths = derive_heading_paths(document)

    assert paths["1"] == ("Part I",)
    assert paths["3"] == ("Part II",)


def test_a_deeper_heading_extends_the_path() -> None:
    text = "Part I\n\nChapter A\n\nBody."
    blocks = (
        _block("0", 0, 6, block_type=BlockType.HEADING,
               attributes=BlockAttributes(heading_level=1)),
        _block("1", 8, 17, block_type=BlockType.HEADING, ordinal=1,
               attributes=BlockAttributes(heading_level=2)),
        _block("2", 19, 24, ordinal=2),
    )
    document = _document(text, blocks)

    assert derive_heading_paths(document)["2"] == ("Part I", "Chapter A")


def test_a_shallower_heading_replaces_the_path_below_it() -> None:
    """A subsequent section's path must exclude the previous section's
    headings."""
    text = "Part I\n\nChapter A\n\nPart II\n\nBody."
    blocks = (
        _block("0", 0, 6, block_type=BlockType.HEADING,
               attributes=BlockAttributes(heading_level=1)),
        _block("1", 8, 17, block_type=BlockType.HEADING, ordinal=1,
               attributes=BlockAttributes(heading_level=2)),
        _block("2", 19, 26, block_type=BlockType.HEADING, ordinal=2,
               attributes=BlockAttributes(heading_level=1)),
        _block("3", 28, 33, ordinal=3),
    )
    document = _document(text, blocks)

    assert derive_heading_paths(document)["3"] == ("Part II",)


def test_headings_are_recovered_for_nested_documents() -> None:
    """Pre-order guarantees a heading precedes its content under either
    model."""
    text = "Section One\n\nBody."
    blocks = (
        _block("0", 0, 18, block_type=BlockType.SECTION),
        _block("0.0", 0, 11, block_type=BlockType.HEADING, parent_id="0", depth=1,
               attributes=BlockAttributes(heading_level=1)),
        _block("0.1", 13, 18, parent_id="0", depth=1, ordinal=1),
    )
    document = _document(text, blocks)

    assert derive_heading_paths(document)["0.1"] == ("Section One",)


def test_a_heading_is_not_its_own_ancestor() -> None:
    document = _document()

    assert derive_heading_paths(document)["0"] == ()


def test_a_document_without_headings_yields_empty_paths() -> None:
    document = _document("Body only.", (_block("0", 0, 10),))

    assert derive_heading_paths(document)["0"] == ()


def test_a_heading_with_no_level_is_treated_as_outermost() -> None:
    text = "Title\n\nBody."
    blocks = (
        _block("0", 0, 5, block_type=BlockType.HEADING),
        _block("1", 7, 12, ordinal=1),
    )
    document = _document(text, blocks)

    assert derive_heading_paths(document)["1"] == ("Title",)


def test_the_heading_path_reaches_the_assembled_node() -> None:
    document = _document()

    nodes = _plan_and_assemble(document)
    body = next(n for n in nodes if n.ordinal == 1)

    assert body.heading_path == ("Interest Rates",)
    assert body.heading_context == "Interest Rates"


def test_heading_context_is_not_stored_in_node_text() -> None:
    """
    Context reaches the vector at compose time; the stored text must remain an
    exact substring of the source.
    """
    document = _document()

    body = next(n for n in _plan_and_assemble(document) if n.ordinal == 1)

    assert "Interest Rates" not in body.text
    assert body.verify_against(document.text) is True


def test_heading_paths_are_computed_once_per_document() -> None:
    """Scanning backwards per chunk would be quadratic in block count."""
    source = inspect.getsource(assembly_module.NodeAssembler.assemble)

    assert "derive_heading_paths(document)" in source


# ---------------------------------------------------------------------------
# Assembly: page numbers and ordinals
# ---------------------------------------------------------------------------


def test_page_numbers_are_collected_from_the_covered_blocks() -> None:
    text = "Page one text.\n\nPage two text."
    blocks = (
        _block("0", 0, 14, attributes=BlockAttributes(page_number=1)),
        _block("1", 16, 30, ordinal=1, attributes=BlockAttributes(page_number=2)),
    )
    document = _document(text, blocks)

    def _one_chunk(doc: RawDocument) -> ChunkPlan:
        return ChunkPlan(
            document_id=doc.document_id,
            strategy_name="single",
            chunks=(
                PlannedChunk(span=TextSpan(start=0, end=30), block_ids=("0", "1")),
            ),
        )

    node = _plan_and_assemble(document, _Strategy(_one_chunk))[0]

    assert node.page_numbers == (1, 2)
    assert node.page_range == (1, 2)
    assert node.spans_multiple_pages is True


def test_an_unpaginated_document_yields_no_page_numbers() -> None:
    node = _plan_and_assemble(_document())[0]

    assert node.page_numbers == ()


def test_ordinals_follow_document_order() -> None:
    nodes = _plan_and_assemble(_document())

    assert [node.ordinal for node in nodes] == [0, 1, 2]


# ---------------------------------------------------------------------------
# Assembly: failure paths
# ---------------------------------------------------------------------------


def test_an_unresolvable_block_reference_is_reported() -> None:
    document = _document()
    plan = ChunkPlan(
        document_id=document.document_id,
        strategy_name="phantom",
        chunks=(
            PlannedChunk(span=TextSpan(start=0, end=14), block_ids=("absent",)),
        ),
    )

    with pytest.raises(BlockReferenceError, match="unrecoverable"):
        _assembler().assemble(document, plan, FINGERPRINT)


def test_provenance_verification_can_be_disabled() -> None:
    document = _document()
    plan = _planner().plan(document)

    strict = NodeAssembler(ValidationConfig(verify_provenance=True), _FixedClock())
    lax = NodeAssembler(ValidationConfig(verify_provenance=False), _FixedClock())

    assert strict.assemble(document, plan, FINGERPRINT) == lax.assemble(
        document, plan, FINGERPRINT
    )


def test_provenance_verification_is_enabled_by_default() -> None:
    assert ValidationConfig().verify_provenance is True


# ---------------------------------------------------------------------------
# Property tests
# ---------------------------------------------------------------------------

_SEGMENT = st.text(
    alphabet=st.characters(blacklist_categories=("Cs", "Cc", "Zs")),
    min_size=1,
    max_size=30,
)


def _document_from(segments: list[str]) -> RawDocument:
    separator = "\n\n"
    text = separator.join(segments)
    blocks = []
    cursor = 0
    for ordinal, segment in enumerate(segments):
        blocks.append(
            _block(str(ordinal), cursor, cursor + len(segment), ordinal=ordinal)
        )
        cursor += len(segment) + len(separator)
    return _document(text, tuple(blocks))


@settings(max_examples=200)
@given(segments=st.lists(_SEGMENT, min_size=1, max_size=12))
def test_any_document_produces_verifiable_nodes(segments: list[str]) -> None:
    """Property: no document produces a node whose text and span disagree."""
    document = _document_from(segments)

    nodes = _plan_and_assemble(document)

    for node in nodes:
        assert node.verify_against(document.text) is True


@settings(max_examples=200)
@given(segments=st.lists(_SEGMENT, min_size=1, max_size=12))
def test_every_segment_is_recovered_exactly(segments: list[str]) -> None:
    """Property: each node recovers the segment it was planned from."""
    document = _document_from(segments)

    nodes = _plan_and_assemble(document)

    assert [node.text for node in nodes] == segments


@settings(max_examples=100)
@given(segments=st.lists(_SEGMENT, min_size=1, max_size=10))
def test_identity_is_stable_across_reprocessing(segments: list[str]) -> None:
    """Property: reprocessing an unchanged document reproduces identifiers."""
    document = _document_from(segments)

    first = _plan_and_assemble(document)
    second = _plan_and_assemble(document)

    assert [n.node_id for n in first] == [n.node_id for n in second]


@settings(max_examples=100)
@given(segments=st.lists(_SEGMENT, min_size=2, max_size=10))
def test_coverage_holds_for_any_document(segments: list[str]) -> None:
    """Property: the nodes collectively account for every non-whitespace
    character."""
    document = _document_from(segments)

    nodes = _plan_and_assemble(document)

    cursor = 0
    for node in nodes:
        gap = document.text[cursor : node.span.start]
        assert not gap or gap.isspace()
        cursor = node.span.end
    assert not document.text[cursor:].strip()


# ---------------------------------------------------------------------------
# Anti-pattern guards
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("module", [planning_module, assembly_module])
def test_the_stages_are_synchronous(module: Any) -> None:
    source = inspect.getsource(module)

    assert "async def" not in source
    assert "await " not in source


@pytest.mark.parametrize("module", [planning_module, assembly_module])
def test_the_stages_perform_no_io(module: Any) -> None:
    source = inspect.getsource(module)

    for forbidden in ("open(", "requests", "httpx", "Path("):
        assert forbidden not in source


def test_assembly_never_reads_the_wall_clock_directly() -> None:
    source = inspect.getsource(assembly_module)

    assert "self._clock.now()" in source
    assert "datetime.now(" not in source
    assert "utcnow" not in source


@pytest.mark.parametrize("module", [planning_module, assembly_module])
def test_the_stages_declare_no_mutable_state(module: Any) -> None:
    mutable = [
        name
        for name, value in vars(module).items()
        if not name.startswith("_") and isinstance(value, (list, dict, set))
    ]

    assert mutable == []


@pytest.mark.parametrize("module", [planning_module, assembly_module])
def test_the_stages_import_no_concrete_implementation(module: Any) -> None:
    source = inspect.getsource(module)

    for forbidden in (
        "from ptfm.core.knowledge.chunking",
        "from ptfm.core.knowledge.embedding",
        "from ptfm.core.knowledge.registry",
        "from ptfm.plugins",
    ):
        assert forbidden not in source


@pytest.mark.parametrize("module", [planning_module, assembly_module])
def test_the_stages_produce_no_confidence_value(module: Any) -> None:
    source = inspect.getsource(module).lower()

    assert "confidence" not in source
    assert "similarity" not in source


@pytest.mark.parametrize("module", [planning_module, assembly_module])
def test_the_stages_contain_no_domain_vocabulary(module: Any) -> None:
    source = inspect.getsource(module).lower()

    for term in ("jurisdiction", "statute", "ticker", "icd", "patient", "court"):
        assert term not in source
