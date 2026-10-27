"""
PTFM Planning Stage

File:
    ptfm/core/knowledge/pipeline/planning.py

Description:
    Invokes a chunking strategy and verifies its plan against the source
    document before any node is constructed.

Why validation lives here and not in the strategy
-------------------------------------------------
    A strategy decides where boundaries fall. This stage decides whether the
    resulting plan is a truthful description of the document it claims to
    describe.

    The separation matters because the checks below require the document, and
    a strategy that performed them itself would be marking its own work. A
    defective strategy fails here, before a single node exists, rather than by
    producing a corpus whose provenance cannot be trusted and whose defect
    surfaces only when a citation is followed months later.

What is verified
----------------
    1. Every span lies within the document text.
    2. Every named block exists in the document.
    3. Named blocks are LEAF blocks, so that provenance is unambiguous.
    4. A chunk's named blocks are exactly the leaves its span covers.
    5. A chunk's span exactly matches the extent of those blocks.
    6. Every non-whitespace character belongs to some chunk.
    7. The plan does not exceed the configured node count.

    Ordering, disjointness, and block exclusivity are enforced by ``ChunkPlan``
    itself, since they need no document.

What is deliberately NOT verified
---------------------------------
    Chunk size. A strategy's size decisions are its own contract: an
    oversized chunk may be a defect, or may be a single unsplittable block
    that the configuration explicitly permits to pass through whole. This
    stage cannot distinguish the two, and rejecting both would make a
    legitimate configuration unusable while duplicating logic the strategy
    already owns.

    Size is therefore governed where it is decided, and this stage confines
    itself to whether the plan describes the document truthfully.

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

from ptfm.core.knowledge.config import ValidationConfig
from ptfm.core.knowledge.exceptions import (
    CoverageError,
    InvalidChunkPlanError,
    PTFMKnowledgeError,
)
from ptfm.core.knowledge.interfaces import ChunkingStrategy
from ptfm.core.knowledge.models import ChunkPlan, PlannedChunk
from ptfm.core.schemas import DocumentBlock, RawDocument

__all__ = ["ChunkPlanner"]

#: Longest text excerpt included in a diagnostic. A document may be very
#: large, and a failure record holding a substantial portion of one would be
#: unusable at corpus scale.
_EXCERPT_CHARS: Final[int] = 120


class ChunkPlanner:
    """
    Produces a verified chunk plan for a document.

    Stateless: no instance state is mutated during planning, so one instance
    serves all concurrent workers.

    Parameters
    ----------
    strategy:
        The chunking strategy to invoke.
    config:
        Validation configuration governing which checks run and how many
        nodes one document may produce.
    """

    __slots__ = ("_strategy", "_config")

    def __init__(
        self,
        strategy: ChunkingStrategy,
        config: ValidationConfig,
    ) -> None:
        self._strategy = strategy
        self._config = config

    def plan(self, document: RawDocument) -> ChunkPlan:
        """
        Produce a plan for a document and verify it against that document.

        Blocking and CPU-bound by contract. The engine executes this in a
        worker thread; it must never be invoked from a coroutine.

        Parameters
        ----------
        document:
            The source document.

        Returns
        -------
        ChunkPlan
            A plan verified to describe the document truthfully. Empty for a
            document with no extractable content, which the engine reports as
            a skipped outcome rather than a failure.

        Raises
        ------
        InvalidChunkPlanError
            If the strategy produced a plan that does not describe the
            document, or raised an unanticipated exception.
        CoverageError
            If document content exists that no chunk covers.
        UnsplittableBlockError
            If the strategy could not divide an oversized block.
        """
        if not document.blocks or not document.text.strip():
            # A document with no content yields no nodes. Reported as a skip
            # rather than a failure: an empty document is a routine corpus
            # occurrence, not a fault.
            return ChunkPlan(
                document_id=document.document_id,
                strategy_name=self._strategy.name,
                chunks=(),
            )

        plan = self._invoke_strategy(document)
        self._verify(document, plan)
        return plan

    # ------------------------------------------------------------------
    # Strategy invocation
    # ------------------------------------------------------------------

    def _invoke_strategy(self, document: RawDocument) -> ChunkPlan:
        """
        Invoke the strategy, converting any unanticipated failure.

        A strategy is contractually obliged to raise knowledge exceptions
        itself, so this handling is a safety net rather than a translation
        layer. Cancellation and interruption are never caught.

        Parameters
        ----------
        document:
            The source document.

        Returns
        -------
        ChunkPlan
            The strategy's plan, unverified.

        Raises
        ------
        InvalidChunkPlanError
            If the strategy raised an unanticipated exception, or returned
            something other than a plan, or a plan for another document.
        PTFMKnowledgeError
            Re-raised, enriched with document context, when the strategy
            raised a knowledge exception directly.
        """
        try:
            plan = self._strategy.plan(document)
        except PTFMKnowledgeError as error:
            if error.document_id is None:
                error.document_id = document.document_id
            error.add_context(
                document_id=document.document_id, strategy_name=self._strategy.name
            )
            raise
        except RecursionError as exc:
            raise InvalidChunkPlanError(
                f"Strategy {self._strategy.name!r} exhausted the stack while "
                f"planning; the document is nested too deeply to traverse.",
                strategy_name=self._strategy.name,
                document_id=document.document_id,
            ) from exc
        except Exception as exc:
            raise InvalidChunkPlanError(
                f"Strategy {self._strategy.name!r} raised "
                f"{type(exc).__name__} while planning: {exc}",
                strategy_name=self._strategy.name,
                document_id=document.document_id,
            ) from exc

        if not isinstance(plan, ChunkPlan):
            raise InvalidChunkPlanError(
                f"Strategy {self._strategy.name!r} returned "
                f"{type(plan).__name__}, which is not a ChunkPlan.",
                strategy_name=self._strategy.name,
                document_id=document.document_id,
            )

        if plan.document_id != document.document_id:
            raise InvalidChunkPlanError(
                f"Strategy {self._strategy.name!r} produced a plan for "
                f"document {plan.document_id!r} while planning "
                f"{document.document_id!r}. Nodes built from it would carry "
                f"provenance pointing at a different document.",
                strategy_name=self._strategy.name,
                document_id=document.document_id,
            )

        return plan

    # ------------------------------------------------------------------
    # Verification
    # ------------------------------------------------------------------

    def _verify(self, document: RawDocument, plan: ChunkPlan) -> None:
        """
        Verify that a plan describes its document truthfully.

        Parameters
        ----------
        document:
            The source document.
        plan:
            The strategy's plan.

        Returns
        -------
        None

        Raises
        ------
        InvalidChunkPlanError
            If the plan exceeds the node limit, or a chunk's spans or block
            references disagree with the document.
        CoverageError
            If document content exists that no chunk covers.
        """
        self._verify_node_count(document, plan)

        leaves = document.leaf_blocks()
        leaves_by_id = {block.block_id: block for block in leaves}
        leaf_ids = set(leaves_by_id)

        for index, chunk in enumerate(plan.chunks):
            self._verify_span_bounds(document, plan, index, chunk)
            self._verify_block_references(
                document, plan, index, chunk, leaves_by_id, leaf_ids
            )
            if self._config.verify_block_alignment:
                self._verify_block_alignment(
                    document, plan, index, chunk, leaves, leaves_by_id
                )

        if self._config.require_full_coverage:
            self._verify_coverage(document, plan)

    def _verify_node_count(self, document: RawDocument, plan: ChunkPlan) -> None:
        """
        Ensure the plan does not exceed the configured node limit.

        A document yielding an enormous number of nodes is a fault rather than
        a corpus, and detecting it at the plan avoids constructing them all
        first.

        Parameters
        ----------
        document:
            The source document.
        plan:
            The strategy's plan.

        Returns
        -------
        None

        Raises
        ------
        InvalidChunkPlanError
            If the chunk count exceeds the limit.
        """
        limit = self._config.max_nodes_per_document
        if plan.chunk_count <= limit:
            return

        raise InvalidChunkPlanError(
            f"Strategy {plan.strategy_name!r} planned {plan.chunk_count} "
            f"chunks, exceeding the configured limit of {limit}.",
            strategy_name=plan.strategy_name,
            document_id=document.document_id,
            context={"chunk_count": plan.chunk_count, "limit": limit},
        )

    def _verify_span_bounds(
        self,
        document: RawDocument,
        plan: ChunkPlan,
        index: int,
        chunk: PlannedChunk,
    ) -> None:
        """
        Ensure a chunk's span lies within the document text.

        Parameters
        ----------
        document:
            The source document.
        plan:
            The strategy's plan.
        index:
            Position of the chunk within the plan.
        chunk:
            The chunk to verify.

        Returns
        -------
        None

        Raises
        ------
        InvalidChunkPlanError
            If the span extends beyond the document text.
        """
        length = len(document.text)
        if chunk.span.end <= length:
            return

        raise InvalidChunkPlanError(
            f"Chunk {index} spans to {chunk.span.end}, beyond the document "
            f"text length of {length}. A node built from it would carry a "
            f"span that resolves to nothing.",
            strategy_name=plan.strategy_name,
            chunk_index=index,
            document_id=document.document_id,
        )

    def _verify_block_references(
        self,
        document: RawDocument,
        plan: ChunkPlan,
        index: int,
        chunk: PlannedChunk,
        leaves_by_id: Mapping[str, DocumentBlock],
        leaf_ids: set[str],
    ) -> None:
        """
        Ensure a chunk names blocks that exist and are content-bearing.

        Named blocks must be leaves. A container block's span is an envelope
        over its descendants, so naming one would make the block reference
        ambiguous: a consumer could not tell whether the node covers the
        container's whole subtree or some part of it, and naming both a
        container and its child would count the same text twice.

        Structure above the leaf remains recoverable through the document's
        block tree, so nothing is lost by requiring leaves.

        Parameters
        ----------
        document:
            The source document.
        plan:
            The strategy's plan.
        index:
            Position of the chunk within the plan.
        chunk:
            The chunk to verify.
        leaves_by_id:
            The document's leaf blocks, keyed by identifier.
        leaf_ids:
            Identifiers of those leaves.

        Returns
        -------
        None

        Raises
        ------
        InvalidChunkPlanError
            If a named block is absent from the document, or is not a leaf.
        """
        known = {block.block_id for block in document.blocks}

        for block_id in chunk.block_ids:
            if block_id not in known:
                raise InvalidChunkPlanError(
                    f"Chunk {index} names block {block_id!r}, which does not "
                    f"exist in the document. The node's structural provenance "
                    f"would be unresolvable.",
                    strategy_name=plan.strategy_name,
                    chunk_index=index,
                    document_id=document.document_id,
                )
            if block_id not in leaf_ids:
                raise InvalidChunkPlanError(
                    f"Chunk {index} names block {block_id!r}, which is a "
                    f"container rather than a leaf. A container's span is an "
                    f"envelope over its descendants, so naming one makes the "
                    f"block reference ambiguous. Name the leaf blocks the "
                    f"chunk covers; ancestors remain recoverable through the "
                    f"document's block tree.",
                    strategy_name=plan.strategy_name,
                    chunk_index=index,
                    document_id=document.document_id,
                )

        del leaves_by_id

    def _verify_block_alignment(
        self,
        document: RawDocument,
        plan: ChunkPlan,
        index: int,
        chunk: PlannedChunk,
        leaves: Sequence[DocumentBlock],
        leaves_by_id: Mapping[str, DocumentBlock],
    ) -> None:
        """
        Ensure a chunk's span and its block references describe the same text.

        Two independent statements are made by every chunk: a span, and a set
        of blocks. If they disagree, the node built from it is internally
        inconsistent, and which of the two a consumer trusts determines what
        the citation means.

        Verified in both directions. The span must lie within the extent of
        the blocks named, and the blocks named must be exactly those leaves
        the span touches.

        Containment rather than equality of extent, because a block larger
        than the size budget is divided internally: each resulting chunk
        covers part of one block and names that block, so its span is a subset
        of the block's rather than equal to it. Requiring equality would make
        the default configuration, which divides oversized blocks, unable to
        produce a valid plan for any document containing one.

        Intersection rather than containment on the block side, for the same
        reason: a partially covered block is genuinely touched by the chunk
        and must be named, while a block merely adjacent to the span must not
        be.

        Parameters
        ----------
        document:
            The source document.
        plan:
            The strategy's plan.
        index:
            Position of the chunk within the plan.
        chunk:
            The chunk to verify.
        leaves:
            The document's leaf blocks, in document order.
        leaves_by_id:
            The same leaves, keyed by identifier.

        Returns
        -------
        None

        Raises
        ------
        InvalidChunkPlanError
            If the span and the block references disagree.
        """
        named = [leaves_by_id[block_id] for block_id in chunk.block_ids]

        # A zero-length chunk names structural markers alone; extent
        # comparison is not meaningful for it.
        if chunk.span.length > 0:
            extent_start = min(block.span.start for block in named)
            extent_end = max(block.span.end for block in named)

            if chunk.span.start < extent_start or chunk.span.end > extent_end:
                raise InvalidChunkPlanError(
                    f"Chunk {index} spans "
                    f"[{chunk.span.start}, {chunk.span.end}), reaching outside "
                    f"the extent of its named blocks, "
                    f"[{extent_start}, {extent_end}). The chunk would carry "
                    f"text that no block it names accounts for.",
                    strategy_name=plan.strategy_name,
                    chunk_index=index,
                    document_id=document.document_id,
                )

        touched = {
            block.block_id
            for block in leaves
            if block.span.length > 0
            and block.span.start < chunk.span.end
            and block.span.end > chunk.span.start
        }
        unnamed = touched - set(chunk.block_ids)

        if unnamed:
            raise InvalidChunkPlanError(
                f"Chunk {index} covers leaf blocks it does not name: "
                f"{sorted(unnamed)}. Their text would be present in the node "
                f"while absent from its structural provenance.",
                strategy_name=plan.strategy_name,
                chunk_index=index,
                document_id=document.document_id,
            )

        spurious = set(chunk.block_ids) - touched
        if spurious and chunk.span.length > 0:
            raise InvalidChunkPlanError(
                f"Chunk {index} names leaf blocks its span does not touch: "
                f"{sorted(spurious)}. The node would claim structural "
                f"provenance for text it does not contain.",
                strategy_name=plan.strategy_name,
                chunk_index=index,
                document_id=document.document_id,
            )

    def _verify_coverage(self, document: RawDocument, plan: ChunkPlan) -> None:
        """
        Ensure every non-whitespace character belongs to some chunk.

        The consequence of a gap is not untidiness. Text absent from every
        node cannot be retrieved, so a claim it would have supported becomes
        unverifiable, and the failure is invisible: the run reports success
        while the evidence is silently missing.

        Whitespace between chunks is permitted, since a separator belongs to
        neither of the chunks it separates.

        Parameters
        ----------
        document:
            The source document.
        plan:
            The strategy's plan.

        Returns
        -------
        None

        Raises
        ------
        CoverageError
            If non-whitespace text falls outside every chunk.
        """
        text = document.text
        cursor = 0

        for chunk in plan.chunks:
            gap = text[cursor : chunk.span.start]
            if gap and not gap.isspace():
                raise self._coverage_error(document, cursor, chunk.span.start, gap)
            cursor = max(cursor, chunk.span.end)

        trailing = text[cursor:]
        if trailing and not trailing.isspace():
            raise self._coverage_error(document, cursor, len(text), trailing)

    def _coverage_error(
        self,
        document: RawDocument,
        start: int,
        end: int,
        excerpt: str,
    ) -> CoverageError:
        """
        Build a coverage failure naming the uncovered region.

        Parameters
        ----------
        document:
            The source document.
        start:
            Start offset of the uncovered region.
        end:
            End offset of the uncovered region.
        excerpt:
            The uncovered text.

        Returns
        -------
        CoverageError
            The failure to raise.
        """
        return CoverageError(
            f"Document content at [{start}, {end}) belongs to no chunk: "
            f"{excerpt[:_EXCERPT_CHARS]!r}. Text absent from every node "
            f"cannot be retrieved, so a claim it would have supported becomes "
            f"unverifiable with nothing reporting an error.",
            document_id=document.document_id,
            uncovered_start=start,
            uncovered_end=end,
            uncovered_chars=len(excerpt.strip()),
            context={"strategy_name": self._strategy.name},
        )

    def __repr__(self) -> str:
        """
        Return a representation naming the strategy in use.

        Returns
        -------
        str
            Diagnostic representation.
        """
        return f"ChunkPlanner(strategy={self._strategy.name!r})"
