"""
PTFM Baseline Chunking Strategies

File:
    ptfm/core/knowledge/chunking/baselines.py

Description:
    Two degenerate strategies at the extremes of granularity: one node per
    document, and one node per block.

Why these exist, and why a fixed-size window does not
-----------------------------------------------------
    Neither of these is a tuning of the structural strategy. Each answers a
    corpus shape for which accumulation toward a size target is simply wrong:

        WholeDocumentChunker
            For corpora of short, self-contained records: catalogue entries,
            support tickets, log lines, question-and-answer pairs. Dividing a
            record that is already the unit of meaning produces fragments that
            retrieve worse than the record did.

        BlockChunker
            For corpora where the block IS the record: a table of
            transactions, a list of definitions, a register. Merging
            neighbouring rows toward a size target would place unrelated
            records in one node.

    Both are also the honest baselines for evaluating the structural
    strategy. A retrieval improvement is only meaningful against a floor, and
    these are the two obvious floors.

    A fixed-size window is deliberately ABSENT. A window cutting at an
    arbitrary character offset produces a chunk whose span reaches into a
    block the chunk does not fully contain and whose neighbours claim the
    remainder, which the plan contract rejects: the blocks a chunk names must
    be exactly those its span touches, and two chunks may not name the same
    block unless one block was divided as a whole. More fundamentally, a
    boundary at an arbitrary offset severs a sentence mid-clause, and the
    resulting node's text is a fragment that means something different from
    what the author wrote. That is a poor baseline and a worse default, so it
    is not offered.

Dependency contract
-------------------
    Imports only the standard library, ``ptfm.core.schemas`` and the knowledge
    ``config``, ``models`` and ``interfaces`` modules.
"""

from __future__ import annotations

from typing import Final, Mapping

from ptfm.core.knowledge.config import ChunkingConfig
from ptfm.core.knowledge.interfaces import ChunkingStrategy
from ptfm.core.knowledge.models import ChunkPlan, PlannedChunk
from ptfm.core.schemas import RawDocument, TextSpan

__all__ = [
    "WHOLE_DOCUMENT_CHUNKER_NAME",
    "BLOCK_CHUNKER_NAME",
    "WholeDocumentChunker",
    "BlockChunker",
]

#: Selectors matching ``ChunkingConfig.strategy``.
WHOLE_DOCUMENT_CHUNKER_NAME: Final[str] = "whole_document"
BLOCK_CHUNKER_NAME: Final[str] = "block"


class WholeDocumentChunker(ChunkingStrategy):
    """
    Produces exactly one node per document.

    Appropriate where a document is already the unit of meaning and dividing
    it would destroy that: a catalogue entry, a support ticket, a log line, a
    question-and-answer pair. Retrieval over such a corpus wants whole
    records, and a fragment of one retrieves worse than the record did.

    Ignores every size parameter, deliberately. A record that happens to be
    long is still one record, and a size bound applied to it would produce
    exactly the fragmentation this strategy exists to avoid. An operator
    choosing it is asserting that document boundaries are the right
    boundaries.

    The span runs from the first content-bearing block to the last, so the
    node's text excludes leading and trailing whitespace while covering
    everything between.

    Parameters
    ----------
    config:
        Accepted for interface uniformity, so that every strategy is
        constructed identically at the composition root. No field is
        consulted.
    """

    __slots__ = ("_config",)

    def __init__(self, config: ChunkingConfig) -> None:
        self._config = config

    @property
    def name(self) -> str:
        """
        Stable identifier of this strategy.

        Returns
        -------
        str
            Implementation identifier.
        """
        return WHOLE_DOCUMENT_CHUNKER_NAME

    def fingerprint_components(self) -> Mapping[str, str]:
        """
        Report the parameters of this strategy that affect a boundary.

        Boundaries are the document's own, so nothing beyond the strategy's
        identity varies. The declaration is nonetheless required: a strategy
        contributing no component could not be distinguished from a
        differently configured instance of itself.

        Returns
        -------
        Mapping[str, str]
            Component names mapped to their values.
        """
        return {"name": WHOLE_DOCUMENT_CHUNKER_NAME}

    def plan(self, document: RawDocument) -> ChunkPlan:
        """
        Produce a single chunk covering the document's content.

        Parameters
        ----------
        document:
            The source document.

        Returns
        -------
        ChunkPlan
            One chunk, or none for a document with no content.
        """
        leaves = [
            block for block in document.leaf_blocks() if block.span.length > 0
        ]

        if not leaves:
            return ChunkPlan(
                document_id=document.document_id,
                strategy_name=self.name,
                chunks=(),
            )

        return ChunkPlan(
            document_id=document.document_id,
            strategy_name=self.name,
            chunks=(
                PlannedChunk(
                    span=TextSpan(
                        start=leaves[0].span.start, end=leaves[-1].span.end
                    ),
                    block_ids=tuple(block.block_id for block in leaves),
                ),
            ),
        )

    def __repr__(self) -> str:
        """
        Return a representation of this strategy.

        Returns
        -------
        str
            Diagnostic representation.
        """
        return "WholeDocumentChunker()"


class BlockChunker(ChunkingStrategy):
    """
    Produces exactly one node per content-bearing block.

    Appropriate where the block is the record: a table of transactions, a
    list of definitions, a register of entries. Accumulating neighbours toward
    a size target would place unrelated records in one node, and a query
    matching one of them would return the others alongside it.

    Maximum granularity, and the finest division the plan contract permits
    without dividing a block internally.

    Ignores every size parameter, for the same reason as
    :class:`WholeDocumentChunker`: an operator choosing it is asserting that
    block boundaries are the right boundaries, and a size bound would override
    that assertion in exactly the cases where it matters.

    Parameters
    ----------
    config:
        Accepted for interface uniformity. No field is consulted.
    """

    __slots__ = ("_config",)

    def __init__(self, config: ChunkingConfig) -> None:
        self._config = config

    @property
    def name(self) -> str:
        """
        Stable identifier of this strategy.

        Returns
        -------
        str
            Implementation identifier.
        """
        return BLOCK_CHUNKER_NAME

    def fingerprint_components(self) -> Mapping[str, str]:
        """
        Report the parameters of this strategy that affect a boundary.

        Returns
        -------
        Mapping[str, str]
            Component names mapped to their values.
        """
        return {"name": BLOCK_CHUNKER_NAME}

    def plan(self, document: RawDocument) -> ChunkPlan:
        """
        Produce one chunk per content-bearing leaf block.

        Zero-length blocks are omitted. A blank paragraph is a structural
        marker rather than content, and a node covering no characters would
        be retrievable by nothing.

        Parameters
        ----------
        document:
            The source document.

        Returns
        -------
        ChunkPlan
            One chunk per content-bearing block, in document order.
        """
        return ChunkPlan(
            document_id=document.document_id,
            strategy_name=self.name,
            chunks=tuple(
                PlannedChunk(span=block.span, block_ids=(block.block_id,))
                for block in document.leaf_blocks()
                if block.span.length > 0
            ),
        )

    def __repr__(self) -> str:
        """
        Return a representation of this strategy.

        Returns
        -------
        str
            Diagnostic representation.
        """
        return "BlockChunker()"
