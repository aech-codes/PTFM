"""
PTFM Ingestion Schemas

File:
    ptfm/core/schemas/ingestion.py

Layer:
    Ingestion. May import from ``common`` only.

Description:
    Declares the canonical output contract of the Data Ingestion Module
    (Module 1): the ``RawDocument``, its structural block tree, its generic
    metadata, and its acquisition and interpretation provenance.

Consumers
---------
    Produced by Module 1. Consumed by Knowledge Processing (Module 2),
    Retrieval (Module 3), and Verification (Module 4). These are cross-module
    contracts, not Module 1 internals: the producing module does not own them
    unilaterally.

Stability
---------
    FROZEN. ``RawDocument`` and every type it composes were frozen on
    completion of Module 1 and must not be modified. A change here is a
    corpus migration event for every deployment holding ingested documents.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Dict, Final, Iterator, Optional
from uuid import NAMESPACE_URL, UUID, uuid5

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ptfm.core.schemas.common import TextSpan

__all__ = [
    "RAW_DOCUMENT_SCHEMA_VERSION",
    "PTFM_DOCUMENT_NAMESPACE",
    "derive_document_id",
    "BlockType",
    "CONTAINER_BLOCK_TYPES",
    "ExtractionWarningCode",
    "ExtractionWarning",
    "BlockAttributes",
    "DocumentBlock",
    "SourceDescriptor",
    "DocumentMetadata",
    "BlockIndex",
    "RawDocument",
    "RawDocumentBatch",
]


# ============================================================================
# RAW DOCUMENT (Module 1 — Data Ingestion output contract)
# ============================================================================
#
# RawDocument is the canonical hand-off between the Data Ingestion Module
# and Knowledge Processing. It is deliberately pre-chunking, pre-embedding,
# and pre-enrichment: it carries normalized text, a structural block tree,
# generic metadata, and full provenance, and nothing else.
#
# Evidence-First contract
# -----------------------
# Every DocumentBlock references a character span into RawDocument.text.
# Downstream EvidenceItem instances inherit document_id plus a block span,
# which makes every factual statement produced by PTFM traceable to an exact
# location in an identified source document.
#
# Domain-agnosticism contract
# ---------------------------
# No field in this section may encode legal, medical, financial, or any
# other domain concept. Domain enrichment is performed by Knowledge
# Processing and by plugins, never during ingestion.
# ============================================================================


RAW_DOCUMENT_SCHEMA_VERSION: Final[str] = "1.0"

#: Namespace for deterministic UUIDv5 document identifiers. Derived from a
#: stable URL so that the namespace itself is auditable and reproducible.
PTFM_DOCUMENT_NAMESPACE: Final[UUID] = uuid5(
    NAMESPACE_URL, "https://ptfm.dev/schemas/1.0/raw-document"
)


def derive_document_id(source_uri: str, content_hash: str) -> str:
    """
    Derive the deterministic canonical identifier for a document.

    Identity is a pure function of origin and payload, so re-ingesting the
    same bytes from the same URI always produces the same identifier. This
    makes ingestion idempotent and keeps evidence citations stable across
    index rebuilds.

    Parameters
    ----------
    source_uri:
        Fully qualified URI of the document at its source.
    content_hash:
        Hexadecimal digest of the raw payload bytes.

    Returns
    -------
    str
        String form of a UUIDv5 in :data:`PTFM_DOCUMENT_NAMESPACE`.

    Raises
    ------
    ValueError
        If either argument is empty.
    """
    if not source_uri:
        raise ValueError("source_uri must be a non-empty string")
    if not content_hash:
        raise ValueError("content_hash must be a non-empty string")
    return str(uuid5(PTFM_DOCUMENT_NAMESPACE, f"{source_uri}\x00{content_hash}"))


# ----------------------------------------------------------------------------
# Block taxonomy
# ----------------------------------------------------------------------------


class BlockType(StrEnum):
    """
    Canonical structural role of a document block.

    The taxonomy is format-neutral: PDF pages, DOCX styles, HTML tags,
    Markdown headings, and CSV rows all normalize onto these members.
    """

    SECTION = "SECTION"
    HEADING = "HEADING"
    PARAGRAPH = "PARAGRAPH"
    LIST = "LIST"
    LIST_ITEM = "LIST_ITEM"
    TABLE = "TABLE"
    TABLE_ROW = "TABLE_ROW"
    TABLE_CELL = "TABLE_CELL"
    FIGURE = "FIGURE"
    CAPTION = "CAPTION"
    QUOTE = "QUOTE"
    CODE = "CODE"
    FOOTNOTE = "FOOTNOTE"
    PAGE_BREAK = "PAGE_BREAK"
    SLIDE = "SLIDE"
    OTHER = "OTHER"


#: Block types permitted to have children. Any other type must be a leaf.
#: Membership here is permissive, not mandatory: a SECTION with no children
#: is valid, but a PARAGRAPH with children is not.
CONTAINER_BLOCK_TYPES: Final[frozenset[BlockType]] = frozenset(
    {
        BlockType.SECTION,
        BlockType.LIST,
        BlockType.LIST_ITEM,
        BlockType.TABLE,
        BlockType.TABLE_ROW,
        BlockType.TABLE_CELL,
        BlockType.FIGURE,
        BlockType.QUOTE,
        BlockType.SLIDE,
    }
)


class ExtractionWarningCode(StrEnum):
    """
    Machine-readable classification of a non-fatal extraction observation.

    Warnings record degraded fidelity that did not prevent producing a
    document. They are diagnostic only and must never be interpreted as
    confidence signals or used to gate any decision.
    """

    MEDIA_TYPE_MISMATCH = "MEDIA_TYPE_MISMATCH"
    ENCODING_REPLACEMENT = "ENCODING_REPLACEMENT"
    LANGUAGE_DETECTION_UNCERTAIN = "LANGUAGE_DETECTION_UNCERTAIN"
    STRUCTURE_INFERRED = "STRUCTURE_INFERRED"
    TABLE_STRUCTURE_UNCERTAIN = "TABLE_STRUCTURE_UNCERTAIN"
    UNSUPPORTED_ELEMENT_SKIPPED = "UNSUPPORTED_ELEMENT_SKIPPED"
    PARTIAL_EXTRACTION = "PARTIAL_EXTRACTION"
    EMPTY_PAGE = "EMPTY_PAGE"
    OCR_APPLIED = "OCR_APPLIED"
    OCR_LOW_CONFIDENCE = "OCR_LOW_CONFIDENCE"
    NEAR_DUPLICATE_DETECTED = "NEAR_DUPLICATE_DETECTED"
    METADATA_UNAVAILABLE = "METADATA_UNAVAILABLE"
    OTHER = "OTHER"


class ExtractionWarning(BaseModel):
    """
    Non-fatal observation recorded during ingestion.

    Attributes
    ----------
    code:
        Classification used for aggregation and corpus quality metrics.
    message:
        Human-readable description.
    block_id:
        Identifier of the affected block, when the warning is localized.
    detail:
        Free-form diagnostic payload.

        Any numeric score placed here (for example OCR per-token
        confidence) is operator-facing diagnostics ONLY. It must never be
        propagated into verification, calibration, or any decision path.
        Raw model scores are prohibited outside the conformal calibration
        layer.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    code: ExtractionWarningCode
    message: str
    block_id: Optional[str] = None
    detail: Dict[str, Any] = Field(default_factory=dict)


# ----------------------------------------------------------------------------
# Blocks
#
# Character spans are expressed with ``TextSpan``, declared in the common
# layer because the knowledge layer indexes into the same normalized text.
# ----------------------------------------------------------------------------


class BlockAttributes(BaseModel):
    """
    Optional structural attributes of a block.

    Grouped into a dedicated model so that :class:`DocumentBlock` stays lean
    and new format-neutral attributes can be introduced without altering the
    block contract itself. Every field is format-neutral; nothing here may
    encode domain semantics.

    Attributes
    ----------
    heading_level:
        Outline depth for ``HEADING`` blocks, 1 being the highest level.
    list_level:
        Nesting depth for list structures, 0 being the outermost.
    page_number:
        One-based page, slide, or sheet index within the source document.
    row_index:
        Zero-based row position for table structures.
    column_index:
        Zero-based column position for table cells.
    column_span:
        Number of columns occupied by a merged table cell.
    row_span:
        Number of rows occupied by a merged table cell.
    is_header:
        Whether a table row or cell carries header semantics.
    language:
        BCP 47 tag when this block's language differs from the document's.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    heading_level: Optional[int] = Field(default=None, ge=1)
    list_level: Optional[int] = Field(default=None, ge=0)
    page_number: Optional[int] = Field(default=None, ge=1)
    row_index: Optional[int] = Field(default=None, ge=0)
    column_index: Optional[int] = Field(default=None, ge=0)
    column_span: Optional[int] = Field(default=None, ge=1)
    row_span: Optional[int] = Field(default=None, ge=1)
    is_header: Optional[bool] = None
    language: Optional[str] = None


class DocumentBlock(BaseModel):
    """
    A single node in the canonical document structure tree.

    Blocks do NOT carry their own text. They reference a character span into
    :attr:`RawDocument.text`, which is the single source of truth. This makes
    divergence between a block and the text it claims to cover structurally
    impossible, and halves the memory footprint of a document.

    Use :meth:`RawDocument.block_text` to materialize a block's text.

    Attributes
    ----------
    block_id:
        Stable hierarchical identifier, unique within the document, derived
        deterministically from tree position (for example ``"0.3.1"``).
        Stability across re-ingestion is what makes evidence citations
        durable.
    block_type:
        Structural role of the block.
    span:
        Character range into the document text. For container blocks this is
        the envelope of all descendant spans.
    parent_id:
        Identifier of the parent block, or ``None`` for a root block.
    depth:
        Distance from the tree root; roots have depth zero.
    ordinal:
        Zero-based position among siblings sharing the same parent.
    attributes:
        Optional format-neutral structural attributes.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    block_id: str = Field(min_length=1)
    block_type: BlockType
    span: TextSpan
    parent_id: Optional[str] = None
    depth: int = Field(ge=0)
    ordinal: int = Field(ge=0)
    attributes: BlockAttributes = Field(default_factory=BlockAttributes)

    @model_validator(mode="after")
    def _validate_root_depth(self) -> DocumentBlock:
        """
        Ensure depth and parentage agree.

        Returns
        -------
        DocumentBlock
            The validated instance.

        Raises
        ------
        ValueError
            If a root block has non-zero depth, or a non-root block has
            depth zero.
        """
        if self.parent_id is None and self.depth != 0:
            raise ValueError(
                f"Root block {self.block_id!r} must have depth 0, got {self.depth}"
            )
        if self.parent_id is not None and self.depth == 0:
            raise ValueError(
                f"Non-root block {self.block_id!r} must have depth greater than 0"
            )
        return self


# ----------------------------------------------------------------------------
# Provenance and metadata
# ----------------------------------------------------------------------------


class SourceDescriptor(BaseModel):
    """
    Provenance record describing where a document came from, how it was
    acquired, and how it was interpreted.

    Attributes
    ----------
    source_uri:
        Fully qualified URI of the document at its source.
    scheme:
        URI scheme, matching the registered connector key.
    connector_name:
        Identifier of the connector implementation that performed acquisition.
    extractor_name:
        Identifier of the extractor implementation that produced the block
        tree, when one was involved.

        Completes the provenance chain. The connector determines which bytes
        entered the pipeline; the extractor determines what structure was
        derived from them, and therefore what a downstream span refers to.

        This cannot be inferred from :attr:`detected_media_type`. Resolution
        depends on registry priority, configured exclusions, and optional
        dependency availability, so the same media type may be handled by a
        dedicated extractor in one deployment and by a fallback in another,
        yielding materially different block structures.

        ``None`` where no extractor ran, which occurs only for a document that
        yielded no content.
    acquired_at:
        Timezone-aware instant at which the payload was fetched.
    size_bytes:
        Size of the raw payload.
    detected_media_type:
        Media type resolved by format detection. Authoritative.
    declared_media_type:
        Media type asserted by the source, when supplied. Advisory only; a
        disagreement with the detected type is recorded as a warning.
    encoding:
        Character encoding resolved for text payloads.
    version_tag:
        Opaque source-side version marker such as an ETag, object version
        identifier, or modification timestamp. Enables incremental
        re-ingestion without re-downloading payloads.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    source_uri: str = Field(min_length=1)
    scheme: str = Field(min_length=1)
    connector_name: str = Field(min_length=1)
    extractor_name: Optional[str] = None
    acquired_at: datetime
    size_bytes: int = Field(ge=0)
    detected_media_type: str = Field(min_length=1)
    declared_media_type: Optional[str] = None
    encoding: Optional[str] = None
    version_tag: Optional[str] = None


class DocumentMetadata(BaseModel):
    """
    Format-generic metadata harvested during ingestion.

    Scope is strictly limited to properties that exist independently of any
    knowledge domain. Domain metadata (jurisdiction, ticker symbol,
    diagnostic code, study design) is produced by Knowledge Processing and
    by plugins, never here.

    Attributes
    ----------
    title:
        Document title as declared by the format, when available.
    authors:
        Declared authors.
    created_at:
        Declared creation timestamp.
    modified_at:
        Declared last-modification timestamp.
    declared_language:
        Language asserted by the document format.
    detected_language:
        Language inferred from the text. No detection score is recorded;
        low-confidence detection is reported as an extraction warning
        instead, so that no raw model score becomes consumable as a field.
    page_count:
        Number of pages, slides, or sheets, as applicable to the format.
    char_count:
        Length of the normalized document text. Validated against the text.
    word_count:
        Whitespace-delimited token count. Advisory; tokenization is not
        standardized across extractors.
    producer:
        Application that generated the document, when declared.
    format_properties:
        Format-native properties preserved verbatim, for example
        ``{"pdf_version": "1.7"}``. Values are constrained to strings
        specifically to prevent structured domain objects from being
        smuggled through this field.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    title: Optional[str] = None
    authors: tuple[str, ...] = ()
    created_at: Optional[datetime] = None
    modified_at: Optional[datetime] = None
    declared_language: Optional[str] = None
    detected_language: Optional[str] = None
    page_count: Optional[int] = Field(default=None, ge=0)
    char_count: int = Field(ge=0)
    word_count: int = Field(ge=0)
    producer: Optional[str] = None
    format_properties: Dict[str, str] = Field(default_factory=dict)


# ----------------------------------------------------------------------------
# Block index
# ----------------------------------------------------------------------------


class BlockIndex:
    """
    Precomputed lookup structure over a document's block tree.

    :class:`RawDocument` accessors are linear scans by design, which keeps
    the model free of caching machinery that interacts badly with frozen
    Pydantic instances. Callers performing repeated lookups build one of
    these instead.

    Parameters
    ----------
    blocks:
        The document's blocks in depth-first pre-order.
    """

    __slots__ = ("_by_id", "_children")

    def __init__(self, blocks: tuple[DocumentBlock, ...]) -> None:
        self._by_id: dict[str, DocumentBlock] = {b.block_id: b for b in blocks}
        children: dict[Optional[str], list[DocumentBlock]] = {}
        for block in blocks:
            children.setdefault(block.parent_id, []).append(block)
        self._children: dict[Optional[str], tuple[DocumentBlock, ...]] = {
            key: tuple(value) for key, value in children.items()
        }

    def get(self, block_id: str) -> Optional[DocumentBlock]:
        """
        Retrieve a block by identifier in constant time.

        Parameters
        ----------
        block_id:
            Identifier to resolve.

        Returns
        -------
        Optional[DocumentBlock]
            The block, or ``None`` if no such block exists.
        """
        return self._by_id.get(block_id)

    def children_of(self, block_id: Optional[str]) -> tuple[DocumentBlock, ...]:
        """
        Retrieve the ordered children of a block in constant time.

        Parameters
        ----------
        block_id:
            Parent identifier, or ``None`` for root blocks.

        Returns
        -------
        tuple[DocumentBlock, ...]
            Children in sibling order; empty if the block is a leaf.
        """
        return self._children.get(block_id, ())


# ----------------------------------------------------------------------------
# RawDocument
# ----------------------------------------------------------------------------


class RawDocument(BaseModel):
    """
    Canonical output of the Data Ingestion Module.

    A ``RawDocument`` is a faithful, domain-agnostic, structure-preserving
    representation of a single source document. It is deliberately
    pre-chunking and pre-embedding: Knowledge Processing consumes it to
    produce cleaned, chunked, enriched ``KnowledgeNode`` instances.

    Structural invariants enforced at construction
    ----------------------------------------------
    1. ``blocks`` is in depth-first pre-order, so a parent always precedes
       its descendants. Cycles are therefore inexpressible.
    2. Block identifiers are unique.
    3. Every ``parent_id`` resolves to an earlier block.
    4. ``depth`` equals parent depth plus one.
    5. Sibling ``ordinal`` values are contiguous from zero.
    6. Only container block types have children.
    7. Every span lies within ``[0, len(text)]``.
    8. A child span is contained within its parent's span.
    9. Leaf spans are ordered and non-overlapping.
    10. The gap between consecutive leaf spans contains whitespace only, so
        no document content can exist outside the block tree.
    11. ``metadata.char_count`` equals ``len(text)``.
    12. ``text`` contains no carriage returns; line endings are normalized.

    Invariants 7 through 10 are the machine-checked form of the
    Evidence-First principle: they guarantee that any span cited by a
    downstream ``EvidenceItem`` resolves to an exact, unambiguous region of
    an identified source document.

    Attributes
    ----------
    document_id:
        Deterministic canonical identifier from :func:`derive_document_id`.
        Typed as ``str`` to join directly against ``RetrievedDocument`` and
        ``EvidenceItem``.
    source:
        Acquisition and interpretation provenance.
    metadata:
        Format-generic metadata.
    text:
        NFC-normalized full document text. The single source of truth for
        all block content.
    blocks:
        Structure tree in depth-first pre-order. May be empty for a document
        that yielded no extractable content.
    content_hash:
        Hexadecimal digest of the raw payload bytes. Basis of document
        identity and exact-duplicate detection.
    text_hash:
        Hexadecimal digest of the normalized text. Detects substantive
        change when a payload is re-encoded without semantic modification.
    warnings:
        Non-fatal observations recorded during ingestion.
    ingested_at:
        Timezone-aware instant at which this document was produced.
    schema_version:
        Version of the ``RawDocument`` contract, for corpus migration.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    document_id: str = Field(min_length=1)
    source: SourceDescriptor
    metadata: DocumentMetadata
    text: str
    blocks: tuple[DocumentBlock, ...] = ()
    content_hash: str = Field(min_length=1)
    text_hash: str = Field(min_length=1)
    warnings: tuple[ExtractionWarning, ...] = ()
    ingested_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    schema_version: str = RAW_DOCUMENT_SCHEMA_VERSION

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    @model_validator(mode="after")
    def _validate_text_normalization(self) -> RawDocument:
        """
        Ensure line endings are normalized and metadata agrees with the text.

        Returns
        -------
        RawDocument
            The validated instance.

        Raises
        ------
        ValueError
            If the text contains carriage returns, or ``char_count``
            disagrees with the text length.
        """
        if "\r" in self.text:
            raise ValueError(
                "RawDocument.text must use normalized line endings; "
                "carriage returns are not permitted"
            )
        if self.metadata.char_count != len(self.text):
            raise ValueError(
                f"metadata.char_count ({self.metadata.char_count}) does not "
                f"match len(text) ({len(self.text)})"
            )
        return self

    @model_validator(mode="after")
    def _validate_block_tree(self) -> RawDocument:
        """
        Enforce every structural and span invariant of the block tree.

        Executes a single linear pass, so validation cost is proportional to
        block count regardless of tree depth.

        Returns
        -------
        RawDocument
            The validated instance.

        Raises
        ------
        ValueError
            If any documented structural or span invariant is violated.
        """
        if not self.blocks:
            return self

        text_length = len(self.text)
        seen: dict[str, DocumentBlock] = {}
        child_counts: dict[Optional[str], int] = {}
        has_children: set[str] = set()

        for block in self.blocks:
            if block.block_id in seen:
                raise ValueError(f"Duplicate block_id {block.block_id!r}")

            if block.span.end > text_length:
                raise ValueError(
                    f"Block {block.block_id!r} span end {block.span.end} exceeds "
                    f"text length {text_length}"
                )

            parent: Optional[DocumentBlock] = None
            if block.parent_id is not None:
                parent = seen.get(block.parent_id)
                if parent is None:
                    raise ValueError(
                        f"Block {block.block_id!r} references parent "
                        f"{block.parent_id!r}, which does not precede it. Blocks "
                        f"must be ordered depth-first pre-order."
                    )
                if block.depth != parent.depth + 1:
                    raise ValueError(
                        f"Block {block.block_id!r} has depth {block.depth}; "
                        f"expected {parent.depth + 1}"
                    )
                if parent.block_type not in CONTAINER_BLOCK_TYPES:
                    raise ValueError(
                        f"Block {parent.block_id!r} of type "
                        f"{parent.block_type.value} may not have children"
                    )
                if (
                    block.span.start < parent.span.start
                    or block.span.end > parent.span.end
                ):
                    raise ValueError(
                        f"Block {block.block_id!r} span "
                        f"[{block.span.start}, {block.span.end}) is not contained "
                        f"within parent {parent.block_id!r} span "
                        f"[{parent.span.start}, {parent.span.end})"
                    )
                has_children.add(parent.block_id)

            expected_ordinal = child_counts.get(block.parent_id, 0)
            if block.ordinal != expected_ordinal:
                raise ValueError(
                    f"Block {block.block_id!r} has ordinal {block.ordinal}; "
                    f"expected {expected_ordinal}. Sibling ordinals must be "
                    f"contiguous from zero."
                )
            child_counts[block.parent_id] = expected_ordinal + 1
            seen[block.block_id] = block

        self._validate_leaf_spans(has_children, text_length)
        return self

    def _validate_leaf_spans(
        self,
        has_children: set[str],
        text_length: int,
    ) -> None:
        """
        Validate ordering, disjointness, and gap content of leaf spans.

        Leaf blocks are the content-bearing nodes. Container spans are
        envelopes over their descendants and are excluded here to avoid
        counting the same text twice.

        Parameters
        ----------
        has_children:
            Identifiers of blocks observed to have at least one child.
        text_length:
            Length of the normalized document text.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If leaf spans are out of order, overlap, or are separated by
            non-whitespace text.
        """
        previous_end = 0
        previous_id: Optional[str] = None

        for block in self.blocks:
            if block.block_id in has_children:
                continue
            if block.span.start < previous_end:
                raise ValueError(
                    f"Leaf block {block.block_id!r} span starts at "
                    f"{block.span.start}, overlapping or preceding leaf "
                    f"{previous_id!r} which ends at {previous_end}"
                )
            gap = self.text[previous_end : block.span.start]
            if gap and not gap.isspace():
                raise ValueError(
                    f"Non-whitespace text at [{previous_end}, "
                    f"{block.span.start}) is not covered by any block. All "
                    f"document content must belong to the block tree."
                )
            previous_end = block.span.end
            previous_id = block.block_id

        trailing = self.text[previous_end:text_length]
        if trailing and not trailing.isspace():
            raise ValueError(
                f"Non-whitespace trailing text at [{previous_end}, "
                f"{text_length}) is not covered by any block"
            )

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def block_text(self, block: DocumentBlock) -> str:
        """
        Materialize the text covered by a block.

        Parameters
        ----------
        block:
            A block belonging to this document.

        Returns
        -------
        str
            The substring of :attr:`text` covered by the block's span. For a
            container block this is the full text of its subtree, including
            inter-child separators.

        Raises
        ------
        ValueError
            If the span exceeds the document text.
        """
        if block.span.end > len(self.text):
            raise ValueError(
                f"Block {block.block_id!r} span exceeds document text length"
            )
        return self.text[block.span.start : block.span.end]

    def leaf_blocks(self) -> tuple[DocumentBlock, ...]:
        """
        Return the content-bearing leaf blocks in reading order.

        Complexity is linear in block count.

        Returns
        -------
        tuple[DocumentBlock, ...]
            Blocks having no children, in document order.
        """
        parents = {b.parent_id for b in self.blocks if b.parent_id is not None}
        return tuple(b for b in self.blocks if b.block_id not in parents)

    def root_blocks(self) -> tuple[DocumentBlock, ...]:
        """
        Return the top-level blocks of the document.

        Complexity is linear in block count.

        Returns
        -------
        tuple[DocumentBlock, ...]
            Blocks with no parent, in sibling order.
        """
        return tuple(b for b in self.blocks if b.parent_id is None)

    def find_block(self, block_id: str) -> Optional[DocumentBlock]:
        """
        Locate a block by identifier.

        Complexity is linear in block count. Use :meth:`build_index` when
        performing repeated lookups.

        Parameters
        ----------
        block_id:
            Identifier to resolve.

        Returns
        -------
        Optional[DocumentBlock]
            The block, or ``None`` if absent.
        """
        for block in self.blocks:
            if block.block_id == block_id:
                return block
        return None

    def iter_blocks_with_text(self) -> Iterator[tuple[DocumentBlock, str]]:
        """
        Iterate leaf blocks paired with their text.

        The primary consumption path for Knowledge Processing, which chunks
        over content-bearing blocks.

        Yields
        ------
        tuple[DocumentBlock, str]
            Each leaf block and the text it covers, in reading order.
        """
        for block in self.leaf_blocks():
            yield block, self.text[block.span.start : block.span.end]

    def build_index(self) -> BlockIndex:
        """
        Construct a reusable lookup index over the block tree.

        Returns
        -------
        BlockIndex
            Index supporting constant-time identifier and child lookups.
        """
        return BlockIndex(self.blocks)

    def warnings_by_code(
        self, code: ExtractionWarningCode
    ) -> tuple[ExtractionWarning, ...]:
        """
        Filter recorded warnings by classification.

        Parameters
        ----------
        code:
            Warning classification to select.

        Returns
        -------
        tuple[ExtractionWarning, ...]
            Matching warnings in the order recorded.
        """
        return tuple(w for w in self.warnings if w.code is code)


class RawDocumentBatch(BaseModel):
    """
    An ordered collection of ingested documents.

    Provided for batch hand-off to Knowledge Processing and for fixture
    serialization in tests. The streaming ingestion path does not
    materialize batches; it yields documents individually.

    Attributes
    ----------
    documents:
        The documents in the batch.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    documents: tuple[RawDocument, ...] = ()
