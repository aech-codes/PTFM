"""
PTFM Assembly Stage

File:
    ptfm/core/knowledge/pipeline/assembly.py

Description:
    Materializes a verified chunk plan into canonical knowledge nodes.

    This is the stage at which the Evidence-First guarantee is manufactured.
    Every node's text is produced by slicing the source document at the
    node's own span, so the two cannot disagree: the characters stored and
    the offsets recorded originate from one operation.

What assembly derives, and why the strategy does not
----------------------------------------------------
    A plan carries a span and block references. Everything else on a node is
    derived here from the source document:

        text            sliced at the span
        node_type       from the types of the covered blocks
        heading_path    from the headings preceding the chunk
        page_numbers    from the covered blocks' page attributes
        node_id         from document, span, and segmentation fingerprint
        source          copied from the document's provenance record

    A strategy declaring any of these could mislabel a node, and nothing
    downstream would detect it: a table region asserted to be prose would be
    scored and rendered as prose forever. Deriving them from the document
    makes each unforgeable, and reduces a strategy's responsibility to the one
    decision it is qualified to make.

Heading derivation
------------------
    Two heading models occur in practice, and both must work.

    Some formats nest content beneath a section container holding a heading
    child. Others, including word processing documents and lightweight markup,
    emit headings as siblings interleaved with the content they introduce.

    A single mechanism serves both: because the block tree is in depth-first
    pre-order, a heading always precedes the content it introduces. Assembly
    therefore walks the document once, maintaining a stack of open headings by
    outline level, and records the path in force at each block. A deeper
    heading extends the path; an equal or shallower one replaces its
    counterpart and everything below it.

Execution contract
------------------
    Synchronous and CPU-bound. The engine executes it in a worker thread; it
    must never be invoked from a coroutine.

Dependency contract
-------------------
    Imports only the standard library, ``ptfm.core.schemas`` and the knowledge
    ``exceptions``, ``config``, ``models`` and ``interfaces`` modules.
"""

from __future__ import annotations

from typing import Final, Mapping, Sequence

from pydantic import ValidationError

from ptfm.core.knowledge.config import ValidationConfig
from ptfm.core.knowledge.exceptions import (
    BlockReferenceError,
    NodeSchemaValidationError,
    NodeSpanIntegrityError,
)
from ptfm.core.knowledge.interfaces import Clock
from ptfm.core.knowledge.models import ChunkPlan, PlannedChunk
from ptfm.core.schemas import (
    BlockType,
    DocumentBlock,
    KnowledgeNode,
    NodeType,
    RawDocument,
    TextSpan,
    derive_node_id,
)

__all__ = ["NodeAssembler", "derive_heading_paths"]

#: Outline level assumed for a heading that declares none. Word processing
#: documents permit a heading style carrying no level, and treating it as the
#: outermost is the only interpretation that keeps a path well ordered.
_DEFAULT_HEADING_LEVEL: Final[int] = 1

#: Block types mapped to the node type they imply when a chunk covers them
#: exclusively. A chunk covering several kinds is mixed.
_SINGLE_TYPE_NODES: Final[Mapping[BlockType, NodeType]] = {
    BlockType.PARAGRAPH: NodeType.PROSE,
    BlockType.HEADING: NodeType.HEADING,
    BlockType.CODE: NodeType.CODE,
    BlockType.QUOTE: NodeType.QUOTE,
    BlockType.CAPTION: NodeType.FIGURE_REGION,
    BlockType.FOOTNOTE: NodeType.PROSE,
    BlockType.PAGE_BREAK: NodeType.PROSE,
    BlockType.OTHER: NodeType.PROSE,
}

#: Block types belonging to a table. A chunk covering these exclusively is a
#: table region regardless of which of them it covers.
_TABLE_BLOCK_TYPES: Final[frozenset[BlockType]] = frozenset(
    {BlockType.TABLE, BlockType.TABLE_ROW, BlockType.TABLE_CELL}
)

#: Block types belonging to a list.
_LIST_BLOCK_TYPES: Final[frozenset[BlockType]] = frozenset(
    {BlockType.LIST, BlockType.LIST_ITEM}
)

#: Block types belonging to a figure.
_FIGURE_BLOCK_TYPES: Final[frozenset[BlockType]] = frozenset(
    {BlockType.FIGURE, BlockType.CAPTION}
)

#: Longest text excerpt included in a diagnostic.
_EXCERPT_CHARS: Final[int] = 120


class NodeAssembler:
    """
    Materializes a verified plan into canonical knowledge nodes.

    Stateless: all per-document state is local to a call, so one instance
    serves all concurrent workers.

    Parameters
    ----------
    config:
        Validation configuration governing whether provenance is verified
        after construction.
    clock:
        Supplies the instant recorded on each node. Injected so that assembled
        nodes are byte-identical under test, which is what makes the
        determinism property assertable.
    """

    __slots__ = ("_config", "_clock")

    def __init__(self, config: ValidationConfig, clock: Clock) -> None:
        self._config = config
        self._clock = clock

    def assemble(
        self,
        document: RawDocument,
        plan: ChunkPlan,
        segmentation_fingerprint: str,
    ) -> tuple[KnowledgeNode, ...]:
        """
        Build the knowledge nodes a plan describes.

        Blocking and CPU-bound by contract. The engine executes this in a
        worker thread; it must never be invoked from a coroutine.

        Parameters
        ----------
        document:
            The source document. Its text is the single source of truth for
            every node's content.
        plan:
            A plan already verified against this document by the planning
            stage.
        segmentation_fingerprint:
            The fingerprint under which the plan was produced. Participates in
            node identity, so that differing segmentations yield disjoint
            identifier spaces.

        Returns
        -------
        tuple[KnowledgeNode, ...]
            Nodes in document order.

        Raises
        ------
        BlockReferenceError
            If a chunk references a block absent from the document.
        NodeSpanIntegrityError
            If a node's text is not the exact substring of the document at its
            span.
        NodeSchemaValidationError
            If an assembled node failed canonical validation.
        """
        if plan.is_empty:
            return ()

        blocks_by_id = {block.block_id: block for block in document.blocks}
        heading_paths = derive_heading_paths(document)
        processed_at = self._clock.now()

        nodes = tuple(
            self._assemble_node(
                document=document,
                chunk=chunk,
                ordinal=ordinal,
                blocks_by_id=blocks_by_id,
                heading_paths=heading_paths,
                segmentation_fingerprint=segmentation_fingerprint,
                processed_at=processed_at,
            )
            for ordinal, chunk in enumerate(plan.chunks)
        )

        if self._config.verify_provenance:
            self._verify(document, nodes)

        return nodes

    # ------------------------------------------------------------------
    # Node construction
    # ------------------------------------------------------------------

    def _assemble_node(
        self,
        *,
        document: RawDocument,
        chunk: PlannedChunk,
        ordinal: int,
        blocks_by_id: Mapping[str, DocumentBlock],
        heading_paths: Mapping[str, tuple[str, ...]],
        segmentation_fingerprint: str,
        processed_at: object,
    ) -> KnowledgeNode:
        """
        Build one node from one planned chunk.

        The text is sliced from the document at the chunk's own span, so the
        stored characters and the recorded offsets originate from one
        operation and cannot disagree.

        Parameters
        ----------
        document:
            The source document.
        chunk:
            The planned chunk.
        ordinal:
            Position of the node within the document.
        blocks_by_id:
            The document's blocks, keyed by identifier.
        heading_paths:
            Heading path in force at each block, precomputed for the document.
        segmentation_fingerprint:
            The fingerprint participating in node identity.
        processed_at:
            The instant to record, taken once per document so that every node
            of a document shares it.

        Returns
        -------
        KnowledgeNode
            The assembled node.

        Raises
        ------
        BlockReferenceError
            If a referenced block is absent from the document.
        NodeSchemaValidationError
            If the node failed canonical validation.
        """
        blocks = self._resolve_blocks(document, chunk, blocks_by_id)
        span = TextSpan(start=chunk.span.start, end=chunk.span.end)

        try:
            return KnowledgeNode(
                node_id=derive_node_id(
                    document.document_id, span, segmentation_fingerprint
                ),
                document_id=document.document_id,
                span=span,
                # Sliced from the document, never reconstructed. This is the
                # operation that makes the provenance invariant hold by
                # construction rather than by later verification.
                text=document.text[span.start : span.end],
                node_type=_derive_node_type(blocks),
                block_ids=chunk.block_ids,
                heading_path=heading_paths.get(chunk.block_ids[0], ()),
                ordinal=ordinal,
                page_numbers=_derive_page_numbers(blocks),
                # Reused verbatim rather than reconstructed, so that one
                # provenance vocabulary spans ingestion and knowledge alike.
                source=document.source,
                segmentation_fingerprint=segmentation_fingerprint,
                source_text_hash=document.text_hash,
                source_schema_version=document.schema_version,
                processed_at=processed_at,  # type: ignore[arg-type]
            )
        except ValidationError as exc:
            raise NodeSchemaValidationError(
                f"Assembled node at ordinal {ordinal} failed canonical "
                f"validation. This indicates a defect in assembly rather than "
                f"a malformed document.",
                validation_errors=[dict(item) for item in exc.errors()],
                document_id=document.document_id,
                context={
                    "ordinal": ordinal,
                    "span_start": span.start,
                    "span_end": span.end,
                },
            ) from exc

    def _resolve_blocks(
        self,
        document: RawDocument,
        chunk: PlannedChunk,
        blocks_by_id: Mapping[str, DocumentBlock],
    ) -> tuple[DocumentBlock, ...]:
        """
        Resolve a chunk's block references against the document.

        Parameters
        ----------
        document:
            The source document.
        chunk:
            The planned chunk.
        blocks_by_id:
            The document's blocks, keyed by identifier.

        Returns
        -------
        tuple[DocumentBlock, ...]
            The referenced blocks, in the order named.

        Raises
        ------
        BlockReferenceError
            If a reference does not resolve.
        """
        resolved: list[DocumentBlock] = []

        for block_id in chunk.block_ids:
            block = blocks_by_id.get(block_id)
            if block is None:
                raise BlockReferenceError(
                    f"Chunk references block {block_id!r}, which is absent "
                    f"from the document. The node's structural context would "
                    f"be unrecoverable.",
                    block_ids=chunk.block_ids,
                    document_id=document.document_id,
                )
            resolved.append(block)

        return tuple(resolved)

    # ------------------------------------------------------------------
    # Verification
    # ------------------------------------------------------------------

    def _verify(
        self,
        document: RawDocument,
        nodes: Sequence[KnowledgeNode],
    ) -> None:
        """
        Verify every node against the document it claims to come from.

        Defence in depth. Slicing the text from the span should make a
        mismatch impossible, so a failure here indicates a defect in assembly
        rather than in the document, and the message says so.

        Parameters
        ----------
        document:
            The source document.
        nodes:
            The assembled nodes.

        Returns
        -------
        None

        Raises
        ------
        NodeSpanIntegrityError
            If a node's text is not the substring at its span.
        """
        for node in nodes:
            if node.verify_against(document.text):
                continue

            located = document.text[node.span.start : node.span.end]
            raise NodeSpanIntegrityError(
                f"Node {node.node_id!r} carries text that is not the substring "
                f"of its document at [{node.span.start}, {node.span.end}). "
                f"Assembly slices text from the span, so this indicates a "
                f"defect in assembly rather than a malformed document.",
                expected_text=located,
                actual_text=node.text,
                span_start=node.span.start,
                span_end=node.span.end,
                document_id=document.document_id,
                node_id=node.node_id,
            )

    def __repr__(self) -> str:
        """
        Return a representation of this assembler.

        Returns
        -------
        str
            Diagnostic representation.
        """
        return (
            f"NodeAssembler(verify_provenance="
            f"{self._config.verify_provenance})"
        )


# ============================================================================
# DERIVATION
# ============================================================================


def _derive_node_type(blocks: Sequence[DocumentBlock]) -> NodeType:
    """
    Determine a node's structural character from the blocks it covers.

    Derived from the document rather than declared by a strategy, so the label
    is a fact about the source rather than an assertion that could be wrong.

    A chunk covering one structural family takes that family's type. One
    covering several is mixed, which is honest: a node spanning the end of a
    table and the start of the following prose is neither.

    Parameters
    ----------
    blocks:
        The covered blocks.

    Returns
    -------
    NodeType
        The node's structural character.
    """
    if not blocks:
        return NodeType.PROSE

    types = {block.block_type for block in blocks}

    if types <= _TABLE_BLOCK_TYPES:
        return NodeType.TABLE_REGION
    if types <= _LIST_BLOCK_TYPES:
        return NodeType.LIST_REGION
    if types <= _FIGURE_BLOCK_TYPES:
        return NodeType.FIGURE_REGION

    if len(types) == 1:
        only = next(iter(types))
        mapped = _SINGLE_TYPE_NODES.get(only)
        if mapped is not None:
            return mapped
        return NodeType.MIXED

    # Several structural families in one node. Reporting the dominant one
    # would assert a character the node does not uniformly have.
    return NodeType.MIXED


def _derive_page_numbers(blocks: Sequence[DocumentBlock]) -> tuple[int, ...]:
    """
    Collect the pages a node originates on.

    Ascending and free of duplicates, which the canonical schema requires and
    which also makes a page range directly readable.

    Parameters
    ----------
    blocks:
        The covered blocks.

    Returns
    -------
    tuple[int, ...]
        One-based page numbers, empty for an unpaginated format.
    """
    pages = {
        block.attributes.page_number
        for block in blocks
        if block.attributes.page_number is not None
    }
    return tuple(sorted(pages))


def derive_heading_paths(
    document: RawDocument,
) -> Mapping[str, tuple[str, ...]]:
    """
    Compute the heading path in force at each block of a document.

    Serves both heading models found in practice. Some formats nest content
    beneath a section container holding a heading child; others emit headings
    as siblings interleaved with the content they introduce. Because the block
    tree is in depth-first pre-order, a heading precedes the content it
    introduces under either model, so one forward traversal suffices.

    A stack of open headings is maintained by outline level. A deeper heading
    extends the path; an equal or shallower heading replaces its counterpart
    and everything below it, which is what makes a subsequent section's path
    exclude the previous section's headings.

    Computed once per document rather than per chunk. Scanning backwards from
    each chunk would be quadratic in block count, which on a large document is
    the difference between milliseconds and minutes.

    A heading's own path excludes itself, since a heading is not its own
    ancestor.

    Parameters
    ----------
    document:
        The source document.

    Returns
    -------
    Mapping[str, tuple[str, ...]]
        Block identifier mapped to the heading path in force at that block.
    """
    paths: dict[str, tuple[str, ...]] = {}
    stack: list[tuple[int, str]] = []

    for block in document.blocks:
        paths[block.block_id] = tuple(text for _, text in stack)

        if block.block_type is not BlockType.HEADING:
            continue

        text = document.text[block.span.start : block.span.end].strip()
        if not text:
            continue

        level = block.attributes.heading_level or _DEFAULT_HEADING_LEVEL
        while stack and stack[-1][0] >= level:
            stack.pop()
        stack.append((level, text))

    return paths
