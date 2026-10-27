"""
PTFM Structure Normalization Stage

File:
    ptfm/core/ingestion/pipeline/structure.py

Description:
    Converts the ``ExtractedBlock`` sequence produced by a content
    extractor into the canonical pair consumed by ``RawDocument``:
    normalized document text, and a ``DocumentBlock`` tree whose spans
    index into that text.

Why this stage exists
---------------------
    An extractor cannot produce ``DocumentBlock`` instances. Those require
    character spans into normalized text that does not yet exist, plus
    hierarchical identifiers, depth, and sibling ordinals that derive from
    tree position. This stage supplies all of it.

Span integrity by construction
------------------------------
    Text assembly and span computation happen in a SINGLE pass. Each leaf's
    text is normalized first, its length measured after every
    transformation, and its span recorded from the running cursor. Because
    the characters appended to the document and the offsets recorded in the
    span originate from the same operation, span drift cannot be
    introduced.

    This is the mechanism by which the ``RawDocument`` span invariants hold
    by construction rather than by inspection, and therefore the mechanism
    by which every downstream ``EvidenceItem`` remains traceable to an
    exact region of an identified source document.

Normalization order
-------------------
    Per leaf, in this order:

        1. line endings collapsed to LF
        2. Unicode normalization applied
        3. whitespace runs collapsed, unless the block type preserves them
        4. leading and trailing whitespace stripped
        5. Unicode normalization reapplied if steps 3 or 4 disturbed it

    Step 2 precedes step 3 because compatibility normalization can convert
    non-breaking spaces into ordinary ones, which collapsing must then see.
    Step 5 exists because removing whitespace can join a base character to
    a following combining mark, creating a newly composable sequence.

Purity contract
---------------
    Synchronous, stateless, and free of I/O. Holds no instance state that
    is mutated, so a single instance serves all concurrent workers.

Dependency contract
-------------------
    Imports only the standard library, ``ptfm.core.schemas`` and the
    ingestion ``exceptions``, ``config`` and ``models`` modules.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Final, Optional, Sequence

from ptfm.core.ingestion.config import ExtractionConfig, StructureConfig
from ptfm.core.ingestion.exceptions import (
    SpanIntegrityError,
    StructureParsingError,
)
from ptfm.core.ingestion.models import ExtractedBlock, ExtractionOutput
from ptfm.core.schemas import (
    CONTAINER_BLOCK_TYPES,
    BlockType,
    DocumentBlock,
    ExtractionWarning,
    ExtractionWarningCode,
    TextSpan,
)

__all__ = [
    "NormalizedStructure",
    "StructureNormalizer",
]

#: Any sequence of CR, LF, or CRLF, collapsed to a single LF. Required
#: unconditionally: ``RawDocument.text`` forbids carriage returns.
_LINE_ENDINGS: Final[re.Pattern[str]] = re.compile(r"\r\n|\r|\u2028|\u2029")

#: A run of whitespace, collapsed to a single space where permitted.
_WHITESPACE_RUN: Final[re.Pattern[str]] = re.compile(r"\s+")

#: Block types whose internal whitespace is semantic content and must
#: survive normalization. Collapsing a code listing destroys its
#: indentation, which is information, not formatting.
_WHITESPACE_PRESERVING_BLOCK_TYPES: Final[frozenset[BlockType]] = frozenset(
    {BlockType.CODE}
)

#: Sentinel marking a container span not yet resolved by the envelope pass.
_UNRESOLVED: Final[int] = -1


# ============================================================================
# RESULT
# ============================================================================


@dataclass(frozen=True, slots=True)
class NormalizedStructure:
    """
    Normalized document text paired with its canonical block tree.

    Stage-local: this type carries results from structure normalization to
    canonical assembly and is not a cross-module contract. Consumers
    outside the ingestion package depend on ``RawDocument`` instead.

    Attributes
    ----------
    text:
        Normalized document text. The single source of truth for all block
        content; blocks reference it by span rather than carrying copies.
    blocks:
        Canonical block tree in depth-first pre-order, satisfying every
        structural and span invariant of ``RawDocument``.
    warnings:
        Non-fatal observations recorded during normalization. Ordinarily
        empty; populated when a container block's entire subtree yielded no
        text.
    """

    text: str
    blocks: tuple[DocumentBlock, ...]
    warnings: tuple[ExtractionWarning, ...] = ()

    @property
    def char_count(self) -> int:
        """
        Length of the normalized document text.

        Returns
        -------
        int
            Character count.
        """
        return len(self.text)


# ============================================================================
# NORMALIZER
# ============================================================================


class StructureNormalizer:
    """
    Assembles normalized text and a canonical block tree from extractor
    output.

    Parameters
    ----------
    config:
        Text and tree construction settings: normalization form, block
        separator, whitespace policy, depth limit, and whether span
        integrity is verified after assembly.
    limits:
        Output size limits. ``max_text_chars`` guards the normalized text,
        which only this stage produces and which therefore cannot be
        bounded at any earlier point in the pipeline.
    """

    __slots__ = ("_config", "_limits")

    def __init__(
        self,
        config: StructureConfig,
        limits: ExtractionConfig,
    ) -> None:
        self._config = config
        self._limits = limits

    # ------------------------------------------------------------------
    # Public contract
    # ------------------------------------------------------------------

    def normalize(self, output: ExtractionOutput) -> NormalizedStructure:
        """
        Convert extractor output into normalized text and a block tree.

        Parameters
        ----------
        output
            Extraction result. Its block sequence is already validated as
            depth-first pre-order with resolvable parentage and correct
            container semantics, so this stage may assume well-formed
            input and concern itself only with text and spans.

        Returns
        -------
        NormalizedStructure
            Normalized text, the canonical block tree, and any non-fatal
            observations.

        Raises
        ------
        StructureParsingError
            If nesting exceeds the configured depth, an output size limit
            is exceeded, or a container span could not be resolved.
        SpanIntegrityError
            If the assembled spans do not faithfully index the assembled
            text, or the text is not in the configured normal form.
        """
        blocks = output.blocks

        if len(blocks) > self._limits.max_blocks:
            raise StructureParsingError(
                f"Extractor {output.extractor_name!r} produced {len(blocks)} "
                f"blocks, exceeding the configured limit of "
                f"{self._limits.max_blocks}.",
                context={
                    "block_count": len(blocks),
                    "limit": self._limits.max_blocks,
                    "extractor_name": output.extractor_name,
                },
            )

        if not blocks:
            return NormalizedStructure(text="", blocks=())

        assembled = self._assemble(blocks)
        text = "".join(assembled.parts)

        if len(text) > self._limits.max_text_chars:
            raise StructureParsingError(
                f"Normalized text length {len(text)} exceeds the configured "
                f"limit of {self._limits.max_text_chars}.",
                context={
                    "char_count": len(text),
                    "limit": self._limits.max_text_chars,
                    "extractor_name": output.extractor_name,
                },
            )

        self._resolve_envelopes(blocks, assembled.spans)
        document_blocks = self._build_blocks(blocks, assembled)
        warnings = _empty_container_warnings(
            blocks, document_blocks, assembled.leaf_flags
        )

        self._verify_text(text)
        if self._config.validate_span_integrity:
            self._verify_spans(text, document_blocks, assembled.leaf_flags)

        return NormalizedStructure(
            text=text, blocks=document_blocks, warnings=warnings
        )

    # ------------------------------------------------------------------
    # Assembly
    # ------------------------------------------------------------------

    def _assemble(self, blocks: Sequence[ExtractedBlock]) -> _Assembly:
        """
        Build document text and leaf spans in a single forward pass.

        Identifiers, depth, and sibling ordinals are derived from tree
        position as the pass proceeds. Leaf text is normalized and appended
        immediately, and its span recorded from the running cursor, so that
        the appended characters and the recorded offsets originate from one
        operation.

        Parameters
        ----------
        blocks
            Extractor blocks in depth-first pre-order.

        Returns
        -------
        _Assembly
            Text parts, per-block spans, identifiers, depths, ordinals, and
            leaf flags.

        Raises
        ------
        StructureParsingError
            If nesting exceeds the configured depth limit.
        """
        has_children = {
            block.parent_index
            for block in blocks
            if block.parent_index is not None
        }

        separator = self._config.block_separator
        parts: list[str] = []
        spans: list[list[int]] = []
        identifiers: list[str] = []
        depths: list[int] = []
        ordinals: list[int] = []
        leaf_flags: list[bool] = []
        child_counts: dict[Optional[int], int] = {}
        cursor = 0

        for index, block in enumerate(blocks):
            parent_index = block.parent_index
            ordinal = child_counts.get(parent_index, 0)
            child_counts[parent_index] = ordinal + 1

            if parent_index is None:
                depth = 0
                identifier = str(ordinal)
            else:
                depth = depths[parent_index] + 1
                identifier = f"{identifiers[parent_index]}.{ordinal}"

            if depth > self._config.max_block_depth:
                raise StructureParsingError(
                    f"Block nesting depth {depth} exceeds the configured "
                    f"limit of {self._config.max_block_depth} at block "
                    f"{identifier!r}.",
                    context={
                        "block_id": identifier,
                        "depth": depth,
                        "limit": self._config.max_block_depth,
                    },
                )

            identifiers.append(identifier)
            depths.append(depth)
            ordinals.append(ordinal)

            is_leaf = index not in has_children
            leaf_flags.append(is_leaf)

            if not is_leaf:
                spans.append([_UNRESOLVED, _UNRESOLVED])
                continue

            text = self._normalize_leaf_text(block)

            if not text:
                # A blank paragraph is ordinary document noise. It receives
                # a zero-length span at the current cursor and contributes
                # no separator, so that empty blocks cannot accumulate
                # runaway whitespace in the document text.
                spans.append([cursor, cursor])
                continue

            if parts:
                parts.append(separator)
                cursor += len(separator)

            start = cursor
            parts.append(text)
            cursor += len(text)
            spans.append([start, cursor])

        return _Assembly(
            parts=parts,
            spans=spans,
            identifiers=identifiers,
            depths=depths,
            ordinals=ordinals,
            leaf_flags=leaf_flags,
        )

    def _normalize_leaf_text(self, block: ExtractedBlock) -> str:
        """
        Normalize the text of one leaf block.

        Line endings are collapsed unconditionally because
        ``RawDocument.text`` forbids carriage returns. Unicode
        normalization precedes whitespace collapsing so that compatibility
        forms which convert non-breaking spaces are visible to the
        collapser, and is reapplied afterwards because removing whitespace
        can join a base character to a following combining mark.

        Whitespace-preserving block types are exempt from collapsing and
        are stripped of leading and trailing newlines only, since their
        internal whitespace is semantic content.

        Parameters
        ----------
        block
            The leaf block whose text is being normalized.

        Returns
        -------
        str
            Normalized text, possibly empty.
        """
        text = _LINE_ENDINGS.sub("\n", block.text or "")
        form = self._config.normalization_form.value
        text = unicodedata.normalize(form, text)

        preserves_whitespace = (
            block.block_type in _WHITESPACE_PRESERVING_BLOCK_TYPES
        )

        if self._config.collapse_whitespace and not preserves_whitespace:
            text = _WHITESPACE_RUN.sub(" ", text)

        if self._config.strip_block_text:
            text = text.strip("\n") if preserves_whitespace else text.strip()

        if not unicodedata.is_normalized(form, text):
            text = unicodedata.normalize(form, text)

        return text

    def _resolve_envelopes(
        self,
        blocks: Sequence[ExtractedBlock],
        spans: list[list[int]],
    ) -> None:
        """
        Resolve container spans as envelopes over their descendants.

        Iterates in reverse. Because the input is depth-first pre-order, a
        parent always precedes its children, so traversing backwards
        guarantees that a child's own envelope is already complete before
        it propagates into its parent. One pass suffices regardless of tree
        depth.

        Parameters
        ----------
        blocks
            Extractor blocks in depth-first pre-order.
        spans
            Mutable per-block spans; container entries are resolved in
            place.

        Returns
        -------
        None

        Raises
        ------
        StructureParsingError
            If a container span remains unresolved, which indicates a
            container with no descendants.
        """
        for index in range(len(blocks) - 1, -1, -1):
            parent_index = blocks[index].parent_index
            if parent_index is None:
                continue

            child = spans[index]
            parent = spans[parent_index]

            if parent[0] == _UNRESOLVED:
                parent[0], parent[1] = child[0], child[1]
            else:
                parent[0] = min(parent[0], child[0])
                parent[1] = max(parent[1], child[1])

        for index, span in enumerate(spans):
            if span[0] == _UNRESOLVED:
                raise StructureParsingError(
                    f"Container block at index {index} of type "
                    f"{blocks[index].block_type.value} has no descendants and "
                    f"could not be assigned a span.",
                    context={
                        "block_index": index,
                        "block_type": blocks[index].block_type.value,
                    },
                )

    def _build_blocks(
        self,
        blocks: Sequence[ExtractedBlock],
        assembly: _Assembly,
    ) -> tuple[DocumentBlock, ...]:
        """
        Construct canonical blocks from the assembled positional data.

        Parameters
        ----------
        blocks
            Extractor blocks in depth-first pre-order.
        assembly
            Positional data produced by the forward pass.

        Returns
        -------
        tuple[DocumentBlock, ...]
            The canonical block tree, preserving input order.
        """
        return tuple(
            DocumentBlock(
                block_id=assembly.identifiers[index],
                block_type=block.block_type,
                span=TextSpan(
                    start=assembly.spans[index][0],
                    end=assembly.spans[index][1],
                ),
                parent_id=(
                    None
                    if block.parent_index is None
                    else assembly.identifiers[block.parent_index]
                ),
                depth=assembly.depths[index],
                ordinal=assembly.ordinals[index],
                attributes=block.attributes,
            )
            for index, block in enumerate(blocks)
        )

    # ------------------------------------------------------------------
    # Verification
    # ------------------------------------------------------------------

    def _verify_text(self, text: str) -> None:
        """
        Verify that the assembled text satisfies the canonical text
        invariants.

        Document-level normalization is checked rather than assumed.
        Canonical composition does not cross a whitespace boundary in any
        realistic document, so the join of normalized leaves is normally
        already normalized. Renormalizing at document level is not an
        available remedy, because it would alter lengths and invalidate
        every span already computed. The document is therefore failed and
        quarantined instead.

        Parameters
        ----------
        text
            The assembled document text.

        Returns
        -------
        None

        Raises
        ------
        SpanIntegrityError
            If the text contains carriage returns, or is not in the
            configured normal form.
        """
        if "\r" in text:
            raise SpanIntegrityError(
                "Assembled document text contains carriage returns despite "
                "line ending normalization.",
                expected_length=len(text),
                actual_length=len(text),
            )

        form = self._config.normalization_form.value
        if not unicodedata.is_normalized(form, text):
            raise SpanIntegrityError(
                f"Assembled document text is not in {form} form. Joining "
                f"normalized blocks produced a composable sequence across a "
                f"separator boundary. Renormalizing would invalidate every "
                f"computed span, so the document cannot be emitted with "
                f"traceable evidence spans.",
                expected_length=len(text),
                actual_length=len(unicodedata.normalize(form, text)),
            )

    def _verify_spans(
        self,
        text: str,
        blocks: Sequence[DocumentBlock],
        leaf_flags: Sequence[bool],
    ) -> None:
        """
        Verify that the assembled spans faithfully index the assembled text.

        Defence in depth. ``RawDocument`` enforces the same invariants at
        construction, but failing here attributes the fault to the
        structure stage and names the offending block, which is what makes
        a quarantined document diagnosable.

        Parameters
        ----------
        text
            The assembled document text.
        blocks
            The canonical block tree.
        leaf_flags
            Whether each block, by position, is a leaf.

        Returns
        -------
        None

        Raises
        ------
        SpanIntegrityError
            If a span falls outside the text, a child escapes its parent,
            leaf spans overlap or are misordered, or any non-whitespace
            text falls outside the block tree.
        """
        length = len(text)
        by_id = {block.block_id: block for block in blocks}

        for block in blocks:
            if block.span.end > length:
                raise SpanIntegrityError(
                    f"Block {block.block_id!r} span end {block.span.end} "
                    f"exceeds document text length {length}.",
                    block_id=block.block_id,
                    expected_length=length,
                    actual_length=block.span.end,
                )

            if block.parent_id is None:
                continue

            parent = by_id[block.parent_id]
            if (
                block.span.start < parent.span.start
                or block.span.end > parent.span.end
            ):
                raise SpanIntegrityError(
                    f"Block {block.block_id!r} span "
                    f"[{block.span.start}, {block.span.end}) escapes its "
                    f"parent {parent.block_id!r} span "
                    f"[{parent.span.start}, {parent.span.end}).",
                    block_id=block.block_id,
                )

        previous_end = 0
        previous_id: Optional[str] = None

        for block, is_leaf in zip(blocks, leaf_flags, strict=True):
            if not is_leaf:
                continue

            if block.span.start < previous_end:
                raise SpanIntegrityError(
                    f"Leaf block {block.block_id!r} starts at "
                    f"{block.span.start}, overlapping leaf {previous_id!r} "
                    f"which ends at {previous_end}.",
                    block_id=block.block_id,
                )

            gap = text[previous_end : block.span.start]
            if gap and not gap.isspace():
                raise SpanIntegrityError(
                    f"Non-whitespace text at [{previous_end}, "
                    f"{block.span.start}) precedes leaf {block.block_id!r} and "
                    f"belongs to no block. All document content must be "
                    f"reachable through the block tree.",
                    block_id=block.block_id,
                )

            previous_end = block.span.end
            previous_id = block.block_id

        trailing = text[previous_end:]
        if trailing and not trailing.isspace():
            raise SpanIntegrityError(
                f"Non-whitespace trailing text at [{previous_end}, {length}) "
                f"belongs to no block.",
                expected_length=length,
                actual_length=previous_end,
            )


# ============================================================================
# INTERNAL
# ============================================================================


@dataclass(frozen=True, slots=True)
class _Assembly:
    """
    Positional data produced by the forward assembly pass.

    Attributes
    ----------
    parts:
        Text fragments whose concatenation is the document text.
    spans:
        Mutable per-block spans. Leaf entries are final; container entries
        are resolved by the envelope pass.
    identifiers:
        Hierarchical block identifiers derived from tree position.
    depths:
        Distance from the tree root, by block position.
    ordinals:
        Position among siblings, by block position.
    leaf_flags:
        Whether each block, by position, is a leaf.
    """

    parts: list[str]
    spans: list[list[int]]
    identifiers: list[str]
    depths: list[int]
    ordinals: list[bool]
    leaf_flags: list[bool]


def _empty_container_warnings(
    extracted: Sequence[ExtractedBlock],
    blocks: Sequence[DocumentBlock],
    leaf_flags: Sequence[bool],
) -> tuple[ExtractionWarning, ...]:
    """
    Record containers whose entire subtree yielded no text.

    A blank paragraph is ordinary document noise and is not reported. A
    container that produced nothing at all, such as a PDF page or a table
    from which no text could be extracted, is a genuine fidelity signal
    and rare enough that reporting it does not flood the record.

    Parameters
    ----------
    extracted
        The originating extractor blocks.
    blocks
        The canonical block tree.
    leaf_flags
        Whether each block, by position, is a leaf.

    Returns
    -------
    tuple[ExtractionWarning, ...]
        One warning per empty container, in document order.
    """
    warnings: list[ExtractionWarning] = []

    for block, is_leaf in zip(blocks, leaf_flags, strict=True):
        if is_leaf or block.span.length > 0:
            continue
        if block.block_type not in CONTAINER_BLOCK_TYPES:
            continue
        warnings.append(
            ExtractionWarning(
                code=ExtractionWarningCode.PARTIAL_EXTRACTION,
                message=(
                    f"Container block of type {block.block_type.value} "
                    f"yielded no text across its entire subtree"
                ),
                block_id=block.block_id,
                detail={"block_type": block.block_type.value},
            )
        )

    del extracted
    return tuple(warnings)