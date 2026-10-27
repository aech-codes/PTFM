"""
PTFM Structural Chunking Strategy

File:
    ptfm/core/knowledge/chunking/structural.py

Description:
    The production chunking strategy: accumulates consecutive leaf blocks
    toward a size target, dividing only where the document itself offers a
    division.

One strategy, several policies
------------------------------
    An earlier sketch of this module proposed separate strategies for
    block-aligned, table-aware, and sentence-based chunking. Building against
    the finished contracts showed that to be the wrong shape.

    A real corpus mixes prose and tables in one document, so a "table
    strategy" would still have to handle prose and a "block strategy" would
    still encounter tables. Both would therefore implement the whole
    algorithm, differing only in a policy decision that configuration already
    expresses: ``preserve_table_rows``. Two classes for one algorithm is a
    duplication that Anti-Pattern 9 exists to prevent.

    Sentence segmentation is likewise not a peer. It applies only where a
    single block exceeds the budget, which is a fallback within this
    algorithm rather than a different way of dividing a document, and it is
    already expressed by ``split_oversized_blocks`` and by an injected
    segmenter.

    What remains is one strategy whose behaviour is governed by the four
    boundary policies already present in configuration. Adding a fifth
    behaviour means adding a policy, not a class.

Boundaries the author made, in preference to boundaries we invent
-----------------------------------------------------------------
    Division happens at block boundaries wherever possible, because a block
    boundary is a paragraph, list item, or table row that the author created
    deliberately. Sentence boundaries are inferred and are used only when no
    authored boundary is available.

    This is also why embedding-based chunking is not offered. Cutting where
    neighbouring sentence vectors diverge is non-deterministic across model
    versions, which would make node identity depend on an embedding model, and
    it replaces a boundary the author wrote with one a model guessed.

Dependency contract
-------------------
    Imports only the standard library, ``ptfm.core.schemas`` and the knowledge
    ``exceptions``, ``config``, ``models`` and ``interfaces`` modules.
"""

from __future__ import annotations

from typing import Final, Iterator, Mapping, Optional, Sequence

from ptfm.core.knowledge.config import ChunkingConfig
from ptfm.core.knowledge.exceptions import UnsplittableBlockError
from ptfm.core.knowledge.interfaces import ChunkingStrategy, SentenceSegmenter
from ptfm.core.knowledge.models import ChunkPlan, PlannedChunk
from ptfm.core.schemas import BlockType, DocumentBlock, RawDocument, TextSpan

__all__ = ["STRUCTURAL_CHUNKER_NAME", "StructuralChunker"]

#: Selector matching ``ChunkingConfig.strategy``.
STRUCTURAL_CHUNKER_NAME: Final[str] = "block_aligned"

#: Container types whose members must not be divided mid-member when the
#: corresponding policy is in force.
_ROW_CONTAINER: Final[BlockType] = BlockType.TABLE_ROW


class StructuralChunker(ChunkingStrategy):
    """
    Accumulates consecutive leaf blocks toward a size target.

    The default strategy, and the only one intended for production use.

    Algorithm
    ---------
    Leaf blocks are visited in document order and accumulated. The
    accumulation is closed when adding the next block would exceed the
    maximum, when the target has been reached, or when a policy boundary
    intervenes. A block larger than the maximum is divided internally at
    sentence boundaries, where that is permitted and possible.

    Policies, all from configuration
    --------------------------------
    ``respect_heading_boundaries``
        Closes an accumulation at a heading, so that a node does not span two
        sections. Without it, a node could carry the end of one section and
        the beginning of the next, and its heading path would describe only
        one of them.

    ``preserve_table_rows``
        Closes an accumulation at a row boundary rather than mid-row. A node
        holding the last cells of one record and the first of the next is
        worse than an oversized node, because it reads as a single record and
        is not one.

    ``split_oversized_blocks``
        Divides a block larger than the maximum at sentence boundaries. When
        disabled, such a block becomes one oversized node, which some corpora
        prefer to fragmentation.

    ``min_chunk_chars``
        Merges a trailing accumulation too small to stand alone into its
        predecessor, provided the result stays within the maximum. A node of a
        few characters carries no retrievable meaning.

    Determinism
    -----------
    A pure function of the document and the configuration. No instance state
    is mutated during planning, so one instance serves all concurrent workers
    and the same document always plans identically.

    Parameters
    ----------
    config:
        Boundary-affecting configuration. Every field participates in node
        identity.
    segmenter:
        Divides an oversized block. Its version participates in the
        segmentation fingerprint.

    Nodes never overlap. Duplicated text would enter verification as two
    evidence items rather than one, making a claim appear better corroborated
    than the corpus warrants, which in a system producing calibrated
    confidence is a correctness defect. Where a claim's support straddles two
    nodes, ``PRECEDES`` relations let retrieval expand a hit into its
    neighbours at query time.
    """

    __slots__ = ("_config", "_segmenter")

    def __init__(
        self,
        config: ChunkingConfig,
        segmenter: SentenceSegmenter,
    ) -> None:
        self._config = config
        self._segmenter = segmenter

    @property
    def name(self) -> str:
        """
        Stable identifier of this strategy.

        Returns
        -------
        str
            Implementation identifier.
        """
        return STRUCTURAL_CHUNKER_NAME

    def fingerprint_components(self) -> Mapping[str, str]:
        """
        Report the parameters of this strategy that affect a boundary.

        The configured chunking parameters are contributed by configuration
        itself, so what remains here is the segmenter's version, which
        determines where an oversized block divides and which configuration
        cannot see.

        Returns
        -------
        Mapping[str, str]
            Component names mapped to their values.
        """
        return {
            "name": STRUCTURAL_CHUNKER_NAME,
            "segmenter_version": self._segmenter.version,
        }

    def plan(self, document: RawDocument) -> ChunkPlan:
        """
        Determine the node boundaries for a document.

        Synchronous and CPU-bound by contract.

        Parameters
        ----------
        document:
            The source document.

        Returns
        -------
        ChunkPlan
            Ordered, disjoint, block-aligned spans covering the document's
            content.

        Raises
        ------
        UnsplittableBlockError
            If a block exceeds the maximum, division is permitted, and the
            block offers no interior boundary.
        """
        leaves = [
            block for block in document.leaf_blocks() if block.span.length > 0
        ]

        chunks = self._merge_undersized(
            list(self._accumulate(document, leaves)),
        )

        return ChunkPlan(
            document_id=document.document_id,
            strategy_name=self.name,
            chunks=tuple(chunks),
        )

    # ------------------------------------------------------------------
    # Accumulation
    # ------------------------------------------------------------------

    def _accumulate(
        self,
        document: RawDocument,
        leaves: Sequence[DocumentBlock],
    ) -> Iterator[PlannedChunk]:
        """
        Group consecutive leaf blocks into chunks.

        Parameters
        ----------
        document:
            The source document, consulted for its block tree.
        leaves:
            Content-bearing leaf blocks in document order.

        Yields
        ------
        PlannedChunk
            Chunks in document order, before undersized merging.

        Raises
        ------
        UnsplittableBlockError
            If an oversized block cannot be divided.
        """
        rows = _row_membership(document)
        pending: list[DocumentBlock] = []

        for block in leaves:
            if self._exceeds_maximum(block):
                yield from _emit(pending)
                pending = []
                yield from self._divide(document, block)
                continue

            if pending and self._closes_before(block, pending[-1], rows):
                yield from _emit(pending)
                pending = []

            pending.append(block)

            if self._reached_target(pending):
                yield from _emit(pending)
                pending = []

        yield from _emit(pending)

    def _closes_before(
        self,
        block: DocumentBlock,
        previous: DocumentBlock,
        rows: Mapping[str, str],
    ) -> bool:
        """
        Report whether an accumulation must close before a block.

        Parameters
        ----------
        block:
            The block about to be added.
        previous:
            The last block accumulated.
        rows:
            Leaf identifier mapped to its enclosing table row, where one
            exists.

        Returns
        -------
        bool
            ``True`` when a policy boundary intervenes.
        """
        if (
            self._config.respect_heading_boundaries
            and block.block_type is BlockType.HEADING
        ):
            return True

        if self._config.preserve_table_rows:
            # Closing when the row changes means a chunk never holds the last
            # cells of one record and the first of the next, which would read
            # as a single record and is not one.
            current_row = rows.get(block.block_id)
            previous_row = rows.get(previous.block_id)
            if current_row != previous_row:
                return True

        return False

    def _reached_target(self, pending: Sequence[DocumentBlock]) -> bool:
        """
        Report whether an accumulation has reached its size target.

        Measured across the accumulated extent rather than by summing block
        lengths, so that the separators between blocks are counted. A node's
        text includes them, so excluding them here would systematically
        underestimate node size.

        Parameters
        ----------
        pending:
            The accumulated blocks.

        Returns
        -------
        bool
            ``True`` when the accumulation should close.
        """
        if not pending:
            return False
        extent = pending[-1].span.end - pending[0].span.start
        return extent >= self._config.target_chunk_chars

    def _exceeds_maximum(self, block: DocumentBlock) -> bool:
        """
        Report whether a single block is larger than the maximum node size.

        Parameters
        ----------
        block:
            The block to measure.

        Returns
        -------
        bool
            ``True`` when the block alone exceeds the maximum.
        """
        return block.span.length > self._config.max_chunk_chars

    # ------------------------------------------------------------------
    # Oversized blocks
    # ------------------------------------------------------------------

    def _divide(
        self,
        document: RawDocument,
        block: DocumentBlock,
    ) -> Iterator[PlannedChunk]:
        """
        Divide a block larger than the maximum node size.

        Each resulting chunk names the block it came from, so structural
        provenance survives division: a consumer can still reach the paragraph
        a node was taken from, and the spans together account for the block
        exactly.

        Parameters
        ----------
        document:
            The source document, consulted for the block's text.
        block:
            The oversized block.

        Yields
        ------
        PlannedChunk
            One chunk per division, in reading order.

        Raises
        ------
        UnsplittableBlockError
            If division is permitted but the block offers no boundary.
        """
        if not self._config.split_oversized_blocks:
            # The operator has chosen an oversized node over fragmentation.
            yield PlannedChunk(span=block.span, block_ids=(block.block_id,))
            return

        text = document.text[block.span.start : block.span.end]
        sentences = self._segmenter.segment(text)

        if not sentences:
            raise UnsplittableBlockError(
                block_id=block.block_id,
                block_chars=block.span.length,
                max_chars=self._config.max_chunk_chars,
                document_id=document.document_id,
                context={"segmenter_version": self._segmenter.version},
            )

        offset = block.span.start
        start = block.span.start
        end = block.span.start

        for sentence in sentences:
            candidate_end = offset + sentence.end

            if end > start and candidate_end - start > self._config.max_chunk_chars:
                yield PlannedChunk(
                    span=TextSpan(start=start, end=end),
                    block_ids=(block.block_id,),
                )
                start = offset + sentence.start

            end = candidate_end

            if end - start >= self._config.target_chunk_chars:
                yield PlannedChunk(
                    span=TextSpan(start=start, end=end),
                    block_ids=(block.block_id,),
                )
                start = end

        # The final division extends to the block's end, so the divisions
        # together account for the block exactly and no text is dropped.
        if start < block.span.end:
            yield PlannedChunk(
                span=TextSpan(start=start, end=block.span.end),
                block_ids=(block.block_id,),
            )

    # ------------------------------------------------------------------
    # Undersized chunks
    # ------------------------------------------------------------------

    def _merge_undersized(
        self,
        chunks: Sequence[PlannedChunk],
    ) -> list[PlannedChunk]:
        """
        Absorb chunks too small to stand alone into their predecessor.

        A node of a few characters carries no retrievable meaning, and in a
        dense index it dilutes the statistics its neighbours depend on.

        Merging is skipped where it would push the result past the maximum,
        since an oversized node is a worse outcome than a small one, and where
        a chunk has no predecessor, since a document beginning with a short
        block has nowhere to put it.

        Parameters
        ----------
        chunks:
            Chunks in document order.

        Returns
        -------
        list[PlannedChunk]
            Chunks after merging.
        """
        minimum = self._config.min_chunk_chars
        if minimum <= 0 or not chunks:
            return list(chunks)

        merged: list[PlannedChunk] = []

        for chunk in chunks:
            if not merged or chunk.span.length >= minimum:
                merged.append(chunk)
                continue

            previous = merged[-1]
            combined = chunk.span.end - previous.span.start

            if combined > self._config.max_chunk_chars:
                merged.append(chunk)
                continue

            merged[-1] = PlannedChunk(
                span=TextSpan(start=previous.span.start, end=chunk.span.end),
                block_ids=_unique(previous.block_ids + chunk.block_ids),
            )

        return merged

    def __repr__(self) -> str:
        """
        Return a representation naming the configured size bounds.

        Returns
        -------
        str
            Diagnostic representation.
        """
        return (
            f"StructuralChunker(target={self._config.target_chunk_chars}, "
            f"max={self._config.max_chunk_chars}, "
            f"segmenter={self._segmenter.version!r})"
        )


# ============================================================================
# HELPERS
# ============================================================================


def _emit(pending: Sequence[DocumentBlock]) -> Iterator[PlannedChunk]:
    """
    Produce a chunk from an accumulation, if it holds anything.

    Parameters
    ----------
    pending:
        The accumulated blocks.

    Yields
    ------
    PlannedChunk
        One chunk spanning the accumulation, or nothing when it is empty.
    """
    if not pending:
        return
    yield PlannedChunk(
        span=TextSpan(start=pending[0].span.start, end=pending[-1].span.end),
        block_ids=tuple(block.block_id for block in pending),
    )


def _row_membership(document: RawDocument) -> Mapping[str, str]:
    """
    Map each leaf block to the table row enclosing it, where one exists.

    Computed once per document. Walking ancestry per block during
    accumulation would repeat the same traversal for every cell of a row.

    Parameters
    ----------
    document:
        The source document.

    Returns
    -------
    Mapping[str, str]
        Leaf identifier mapped to enclosing row identifier. Blocks outside any
        table are absent.
    """
    blocks_by_id = {block.block_id: block for block in document.blocks}
    membership: dict[str, str] = {}

    for block in document.blocks:
        row = _enclosing_row(block, blocks_by_id)
        if row is not None:
            membership[block.block_id] = row

    return membership


def _enclosing_row(
    block: DocumentBlock,
    blocks_by_id: Mapping[str, DocumentBlock],
) -> Optional[str]:
    """
    Find the nearest table row ancestor of a block.

    Parameters
    ----------
    block:
        The block whose ancestry is walked.
    blocks_by_id:
        The document's blocks, keyed by identifier.

    Returns
    -------
    Optional[str]
        Identifier of the nearest enclosing row, or ``None``.
    """
    current: Optional[DocumentBlock] = block

    while current is not None:
        if current.block_type is _ROW_CONTAINER:
            return current.block_id
        current = (
            blocks_by_id.get(current.parent_id)
            if current.parent_id is not None
            else None
        )

    return None


def _unique(block_ids: Sequence[str]) -> tuple[str, ...]:
    """
    Remove repeated identifiers while preserving order.

    Merging two chunks that divide one block would otherwise name that block
    twice, which the plan contract rejects.

    Parameters
    ----------
    block_ids:
        Identifiers in order.

    Returns
    -------
    tuple[str, ...]
        Identifiers without repetition.
    """
    seen: set[str] = set()
    ordered: list[str] = []

    for block_id in block_ids:
        if block_id in seen:
            continue
        seen.add(block_id)
        ordered.append(block_id)

    return tuple(ordered)
