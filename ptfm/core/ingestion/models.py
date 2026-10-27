"""
PTFM Data Ingestion Internal Models

File:
    ptfm/core/ingestion/models.py

Description:
    Defines the models exchanged between stages of the Data Ingestion
    Module. These are INTERNAL to ingestion. The cross-module contract is
    ``ptfm.core.schemas.RawDocument``; nothing outside this package should
    depend on the types declared here, with the exception of the outcome
    union that the engine yields.

Model families
--------------
    Pydantic v2, frozen
        Serializable value records: SourceSpec, SourceRecord,
        DetectedFormat, ExtractedBlock, ExtractionOutput, IngestionFailure,
        the outcome union, IngestionReport.

    Frozen dataclasses with __slots__
        Payload-bearing and resource-owning types: AcquiredBlob,
        ExtractionInput.

    Payload types are deliberately excluded from the Pydantic layer. A
    multi-megabyte ``bytes`` field reachable by ``model_dump_json`` is one
    stray log statement away from emitting an entire document, and
    ``AcquiredBlob`` owns a spooled temporary file whose lifecycle cannot
    be expressed by an immutable value model.

Pipeline correspondence
-----------------------
    SourceSpec       -> what to ingest
    SourceRecord     -> discovered candidate, metadata only
    AcquiredBlob     -> transferred payload with provenance
    DetectedFormat   -> resolved media type and encoding
    ExtractionInput  -> payload handed to an extractor
    ExtractionOutput -> extractor result, pre-normalization
    IngestionOutcome -> per-document result yielded by the engine
    IngestionReport  -> run-level aggregate

Dependency contract
-------------------
    Imports only the standard library, Pydantic, ``ptfm.core.schemas`` and
    ``ptfm.core.ingestion.exceptions``. It must not import ``interfaces``,
    any stage, connector, or extractor.
"""

from __future__ import annotations

import hashlib
import traceback
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from types import TracebackType
from typing import Annotated, Any, Iterator, Literal, Mapping, Optional, TypeAlias
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ptfm.core.ingestion.exceptions import (
    IngestionStage,
    PTFMIngestionError,
)
from ptfm.core.schemas import (
    CONTAINER_BLOCK_TYPES,
    BlockAttributes,
    BlockType,
    ExtractionWarning,
    ExtractionWarningCode,
    RawDocument,
)

__all__ = [
    # Re-exports for a single consumer import site
    "IngestionStage",
    "ExtractionWarning",
    "ExtractionWarningCode",
    # Source specification and discovery
    "SourceSpec",
    "SourceRecord",
    # Acquisition
    "AcquiredBlob",
    # Detection
    "DetectionSignalSource",
    "DetectionSignal",
    "DetectedFormat",
    # Extraction
    "ExtractionInput",
    "ExtractedBlock",
    "ExtractedMetadata",
    "ExtractionOutput",
    # Outcomes
    "SkipReason",
    "IngestionFailure",
    "DocumentIngested",
    "DocumentSkipped",
    "DocumentFailed",
    "IngestionOutcome",
    # Reporting
    "IngestionReport",
]


# ============================================================================
# SOURCE SPECIFICATION AND DISCOVERY
# ============================================================================


class SourceSpec(BaseModel):
    """
    Declarative description of what a connector should ingest.

    Connector-agnostic by design: it carries selection intent, not
    transport mechanics. Anything genuinely specific to one transport goes
    in :attr:`options`.

    Attributes
    ----------
    root_uri:
        Fully qualified URI identifying the source root. Its scheme
        selects the connector.
    include_patterns:
        Glob patterns a candidate must match to be considered. An empty
        tuple accepts everything.
    exclude_patterns:
        Glob patterns that reject a candidate. Applied after
        :attr:`include_patterns`.
    recursive:
        Whether to descend into nested containers such as directories.
    max_documents:
        Optional cap on documents yielded by discovery, for sampling and
        smoke runs.
    cursor:
        Opaque incremental marker from a previous run. Connectors use it
        to enumerate only what changed, which is what makes re-ingestion
        cheap rather than a full re-download.
    options:
        Connector-specific settings. String-valued so that structured
        domain objects cannot be smuggled through a generic escape hatch.

    Raises
    ------
    ValueError
        If ``root_uri`` carries no scheme.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    root_uri: str = Field(min_length=1)
    include_patterns: tuple[str, ...] = ()
    exclude_patterns: tuple[str, ...] = ()
    recursive: bool = True
    max_documents: Optional[int] = Field(default=None, ge=1)
    cursor: Optional[str] = None
    options: Mapping[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_scheme(self) -> SourceSpec:
        """
        Ensure the root URI carries a resolvable scheme.

        Returns
        -------
        SourceSpec
            The validated instance.

        Raises
        ------
        ValueError
            If no scheme is present.
        """
        if not urlparse(self.root_uri).scheme:
            raise ValueError(
                f"SourceSpec.root_uri must include a scheme, got "
                f"{self.root_uri!r}. Use 'file:///path' rather than '/path'."
            )
        return self

    @property
    def scheme(self) -> str:
        """
        URI scheme selecting the connector for this specification.

        Returns
        -------
        str
            Lowercase scheme, for example ``"file"``.
        """
        return urlparse(self.root_uri).scheme.lower()


class SourceRecord(BaseModel):
    """
    Lightweight descriptor of a discovered candidate document.

    Yielded by discovery before any payload is transferred. Carrying size
    and version information ahead of transfer is what enables incremental
    ingestion: an unchanged record is skipped without ever being fetched.

    Attributes
    ----------
    record_id:
        Connector-scoped identifier, unique within one discovery pass.
    source_uri:
        Fully qualified URI of the candidate document.
    scheme:
        URI scheme, matching the discovering connector.
    connector_name:
        Identifier of the connector that produced this record.
    discovered_at:
        Timezone-aware instant of discovery.
    filename:
        Originating filename, when the transport exposes one. Contributes
        the weakest signal to format detection.
    declared_media_type:
        Media type asserted by the source. Advisory only.
    size_bytes:
        Payload size when known ahead of transfer, enabling size limits to
        be enforced before download.
    version_tag:
        Opaque source-side version marker such as an ETag or modification
        timestamp. Basis of incremental skip decisions.
    attributes:
        Additional transport metadata. String-valued by policy.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    record_id: str = Field(min_length=1)
    source_uri: str = Field(min_length=1)
    scheme: str = Field(min_length=1)
    connector_name: str = Field(min_length=1)
    discovered_at: datetime
    filename: Optional[str] = None
    declared_media_type: Optional[str] = None
    size_bytes: Optional[int] = Field(default=None, ge=0)
    version_tag: Optional[str] = None
    attributes: Mapping[str, str] = Field(default_factory=dict)


# ============================================================================
# ACQUISITION
# ============================================================================


@dataclass(slots=True, eq=False)
class AcquiredBlob:
    """
    A transferred payload together with its acquisition provenance.

    Deliberately NOT a Pydantic model and NOT frozen. This type owns a
    resource: payloads exceeding the spool threshold live in a temporary
    file that must be released. A frozen value model cannot express that
    lifecycle without ``object.__setattr__`` workarounds, and a Pydantic
    model would place a multi-megabyte payload one stray ``model_dump_json``
    call away from being logged in full.

    ``eq=False`` prevents this handle from being mistaken for a value.

    Exactly one of :attr:`data` and :attr:`spool_path` is populated.

    Attributes
    ----------
    record:
        The source record this payload was fetched for.
    size_bytes:
        Payload size in bytes.
    content_hash:
        Digest of the payload, computed incrementally during transfer so
        that spooled payloads are never materialized solely to be hashed.
    hash_algorithm:
        Algorithm used for :attr:`content_hash`, recorded so a future
        algorithm change is a detectable, migratable event.
    acquired_at:
        Timezone-aware instant at which transfer completed.
    data:
        In-memory payload, for transfers below the spool threshold.
    spool_path:
        Temporary file holding the payload, for larger transfers.
    """

    record: SourceRecord
    size_bytes: int
    content_hash: str
    hash_algorithm: str
    acquired_at: datetime
    data: Optional[bytes] = None
    spool_path: Optional[Path] = None
    _closed: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        """
        Validate that exactly one payload representation is present.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If both or neither of ``data`` and ``spool_path`` are set, or
            if ``size_bytes`` disagrees with an in-memory payload.
        """
        if (self.data is None) == (self.spool_path is None):
            raise ValueError(
                "AcquiredBlob requires exactly one of 'data' or 'spool_path'"
            )
        if self.data is not None and len(self.data) != self.size_bytes:
            raise ValueError(
                f"AcquiredBlob size_bytes ({self.size_bytes}) does not match "
                f"payload length ({len(self.data)})"
            )

    @property
    def is_spooled(self) -> bool:
        """
        Whether the payload resides in temporary storage rather than memory.

        Returns
        -------
        bool
            ``True`` when the payload is spooled to disk.
        """
        return self.spool_path is not None

    @property
    def source_uri(self) -> str:
        """
        URI of the document this payload was fetched from.

        Returns
        -------
        str
            The source URI.
        """
        return self.record.source_uri

    def header(self, size: int) -> bytes:
        """
        Read the leading bytes of the payload for format detection.

        Reads only the requested prefix from spooled payloads, so a large
        document is never materialized merely to identify its type.

        Parameters
        ----------
        size:
            Maximum number of leading bytes to return.

        Returns
        -------
        bytes
            The payload prefix, shorter than ``size`` if the payload is
            smaller.

        Raises
        ------
        ValueError
            If ``size`` is negative or the blob has been closed.
        OSError
            If the spooled payload cannot be read.
        """
        if size < 0:
            raise ValueError(f"header size must not be negative, got {size}")
        self._ensure_open()
        if self.data is not None:
            return self.data[:size]
        assert self.spool_path is not None
        with self.spool_path.open("rb") as handle:
            return handle.read(size)

    def read(self) -> bytes:
        """
        Materialize the complete payload.

        Blocking for spooled payloads. Call only from a worker thread; the
        extraction stage guarantees this.

        Returns
        -------
        bytes
            The full payload.

        Raises
        ------
        ValueError
            If the blob has been closed.
        OSError
            If the spooled payload cannot be read.
        """
        self._ensure_open()
        if self.data is not None:
            return self.data
        assert self.spool_path is not None
        return self.spool_path.read_bytes()

    def close(self) -> None:
        """
        Release the payload, removing any spooled temporary file.

        Idempotent. Missing spool files are tolerated so that cleanup
        during a cancelled run cannot itself raise.

        Returns
        -------
        None
        """
        if self._closed:
            return
        self._closed = True
        if self.spool_path is not None:
            self.spool_path.unlink(missing_ok=True)

    def _ensure_open(self) -> None:
        """
        Guard against use after release.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If the blob has already been closed.
        """
        if self._closed:
            raise ValueError(
                f"AcquiredBlob for {self.record.source_uri!r} has been closed"
            )

    def __enter__(self) -> AcquiredBlob:
        """
        Enter a context managing this payload's lifetime.

        Returns
        -------
        AcquiredBlob
            This blob.
        """
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        """
        Release the payload on context exit, including on failure.

        Parameters
        ----------
        exc_type:
            Type of the propagating exception, if any.
        exc:
            The propagating exception, if any.
        tb:
            Traceback of the propagating exception, if any.

        Returns
        -------
        None
        """
        self.close()

    def __repr__(self) -> str:
        """
        Return a representation that never includes payload bytes.

        Returns
        -------
        str
            Safe representation for logs and error messages.
        """
        location = "spooled" if self.is_spooled else "memory"
        return (
            f"AcquiredBlob(source_uri={self.record.source_uri!r}, "
            f"size_bytes={self.size_bytes}, storage={location!r}, "
            f"closed={self._closed})"
        )


# ============================================================================
# FORMAT DETECTION
# ============================================================================


class DetectionSignalSource(StrEnum):
    """
    Origin of a format detection signal, ordered by authority.

    Magic bytes are authoritative because they describe the payload
    itself. A declared type and a filename extension are both assertions
    that may be wrong or absent.
    """

    MAGIC_BYTES = "MAGIC_BYTES"
    DECLARED = "DECLARED"
    EXTENSION = "EXTENSION"


class DetectionSignal(BaseModel):
    """
    One contributing observation in format detection.

    Retained on the result so that a disagreement between signals is
    inspectable after the fact rather than silently discarded.

    Attributes
    ----------
    source:
        Where the signal came from.
    media_type:
        Media type this signal indicates.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    source: DetectionSignalSource
    media_type: str = Field(min_length=1)


class DetectedFormat(BaseModel):
    """
    Resolved media type and encoding for a payload.

    Attributes
    ----------
    media_type:
        Normalized media type selected from the contributing signals.
    encoding:
        Character encoding for text payloads; ``None`` for binary formats.
    signals:
        Every contributing signal, in the order evaluated.
    warnings:
        Non-fatal observations, notably a mismatch between the detected
        and declared types.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    media_type: str = Field(min_length=1)
    encoding: Optional[str] = None
    signals: tuple[DetectionSignal, ...] = ()
    warnings: tuple[ExtractionWarning, ...] = ()

    @property
    def is_text(self) -> bool:
        """
        Whether the payload is textual and therefore requires decoding.

        Returns
        -------
        bool
            ``True`` for text media types and known text-bearing
            application subtypes.
        """
        if self.media_type.startswith("text/"):
            return True
        return self.media_type in {
            "application/json",
            "application/xml",
            "application/xhtml+xml",
        }

    def signal_for(self, source: DetectionSignalSource) -> Optional[DetectionSignal]:
        """
        Retrieve the signal contributed by a given source.

        Parameters
        ----------
        source:
            Signal origin to look up.

        Returns
        -------
        Optional[DetectionSignal]
            The matching signal, or ``None`` if that source contributed
            nothing.
        """
        for signal in self.signals:
            if signal.source is source:
                return signal
        return None


# ============================================================================
# EXTRACTION
# ============================================================================


@dataclass(frozen=True, slots=True)
class ExtractionInput:
    """
    The payload handed to a content extractor.

    A frozen dataclass rather than a Pydantic model for the same reason as
    :class:`AcquiredBlob`: it carries the full document bytes, and a
    serializable model would make accidental payload logging possible.

    Attributes
    ----------
    source_uri:
        URI of the document, for error messages and provenance.
    media_type:
        Normalized media type resolved by detection.
    data:
        Complete payload bytes.
    encoding:
        Character encoding for text payloads; ``None`` for binary formats.
    filename:
        Originating filename, when available.
    """

    source_uri: str
    media_type: str
    data: bytes
    encoding: Optional[str] = None
    filename: Optional[str] = None

    @property
    def size_bytes(self) -> int:
        """
        Payload size in bytes.

        Returns
        -------
        int
            Length of the payload.
        """
        return len(self.data)

    def __repr__(self) -> str:
        """
        Return a representation that never includes payload bytes.

        Returns
        -------
        str
            Safe representation for logs and error messages.
        """
        return (
            f"ExtractionInput(source_uri={self.source_uri!r}, "
            f"media_type={self.media_type!r}, size_bytes={len(self.data)})"
        )


class ExtractedBlock(BaseModel):
    """
    A structural block as produced by an extractor, before normalization.

    Distinct from ``ptfm.core.schemas.DocumentBlock`` because an extractor
    cannot supply what that type requires: spans reference normalized text
    that does not yet exist, and identifiers, depth and ordinals derive
    from tree position, which the normalizer assigns.

    ``StructureNormalizer`` consumes a sequence of these to produce the
    document text and its ``DocumentBlock`` tree in a single pass. Because
    the text and the spans are emitted together from the same source, the
    span invariants of ``RawDocument`` hold by construction rather than by
    convention.

    Attributes
    ----------
    block_type:
        Structural role of the block.
    text:
        Content for leaf blocks. MUST be ``None`` for container blocks,
        whose text is the concatenation of their descendants. Extractors
        need not normalize line endings or Unicode form; the structure
        stage performs both while computing spans, so that exactly one
        component owns normalization.
    parent_index:
        Zero-based index of the parent within the same sequence. Must
        reference a strictly earlier position, which makes cycles
        inexpressible and validation a single linear pass.
    attributes:
        Format-neutral structural attributes.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    block_type: BlockType
    text: Optional[str] = None
    parent_index: Optional[int] = Field(default=None, ge=0)
    attributes: BlockAttributes = Field(default_factory=BlockAttributes)

    @model_validator(mode="after")
    def _validate_container_text(self) -> ExtractedBlock:
        """
        Ensure container blocks carry no text of their own.

        Returns
        -------
        ExtractedBlock
            The validated instance.

        Raises
        ------
        ValueError
            If a container block type supplies text.
        """
        if self.block_type in CONTAINER_BLOCK_TYPES and self.text is not None:
            raise ValueError(
                f"Container block of type {self.block_type.value} must not "
                f"carry text; its content is the concatenation of its children"
            )
        return self


class ExtractedMetadata(BaseModel):
    """
    Format-native metadata as reported by an extractor.

    Timestamps are typed rather than stringified. Round-tripping parser
    datetimes through strings and re-parsing them is a reliable source of
    timezone defects.

    ``MetadataHarvester`` merges this with detected language and computed
    counts to produce the canonical ``DocumentMetadata``.

    Attributes
    ----------
    title:
        Title declared by the format.
    authors:
        Authors declared by the format.
    created_at:
        Declared creation timestamp.
    modified_at:
        Declared last-modification timestamp.
    declared_language:
        Language asserted by the format.
    page_count:
        Pages, slides, or sheets, as applicable.
    producer:
        Application that generated the document.
    properties:
        Remaining format-native properties. String-valued so that
        structured domain objects cannot pass through.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    title: Optional[str] = None
    authors: tuple[str, ...] = ()
    created_at: Optional[datetime] = None
    modified_at: Optional[datetime] = None
    declared_language: Optional[str] = None
    page_count: Optional[int] = Field(default=None, ge=0)
    producer: Optional[str] = None
    properties: Mapping[str, str] = Field(default_factory=dict)


class ExtractionOutput(BaseModel):
    """
    The complete result of extracting one document.

    Attributes
    ----------
    blocks:
        Structural blocks in depth-first pre-order.
    metadata:
        Format-native metadata.
    warnings:
        Non-fatal observations recorded during extraction.
    extractor_name:
        Identifier of the extractor that produced this result, retained
        for provenance and for attributing corpus quality issues.

    Raises
    ------
    ValueError
        If block ordering, parentage, or container semantics are violated.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    blocks: tuple[ExtractedBlock, ...] = ()
    metadata: ExtractedMetadata = Field(default_factory=ExtractedMetadata)
    warnings: tuple[ExtractionWarning, ...] = ()
    extractor_name: str = Field(min_length=1)

    @model_validator(mode="after")
    def _validate_block_sequence(self) -> ExtractionOutput:
        """
        Enforce pre-order, parentage, and container invariants.

        Executes one linear pass, so cost is proportional to block count
        regardless of tree depth. Rejecting malformed structure here means
        the normalizer can assume well-formed input.

        Returns
        -------
        ExtractionOutput
            The validated instance.

        Raises
        ------
        ValueError
            If a parent index is not strictly earlier, if a non-container
            block is given children, or if a container block has no
            children.
        """
        has_children: set[int] = set()

        for index, block in enumerate(self.blocks):
            parent_index = block.parent_index
            if parent_index is None:
                continue
            if parent_index >= index:
                raise ValueError(
                    f"Block at index {index} references parent index "
                    f"{parent_index}, which does not precede it. Blocks must "
                    f"be ordered depth-first pre-order."
                )
            parent = self.blocks[parent_index]
            if parent.block_type not in CONTAINER_BLOCK_TYPES:
                raise ValueError(
                    f"Block at index {parent_index} of type "
                    f"{parent.block_type.value} may not have children"
                )
            has_children.add(parent_index)

        for index, block in enumerate(self.blocks):
            if block.block_type in CONTAINER_BLOCK_TYPES and index not in has_children:
                raise ValueError(
                    f"Container block at index {index} of type "
                    f"{block.block_type.value} has no children. An empty "
                    f"container carries no content and would produce an "
                    f"unanchored span."
                )
            if block.block_type not in CONTAINER_BLOCK_TYPES and block.text is None:
                raise ValueError(
                    f"Leaf block at index {index} of type "
                    f"{block.block_type.value} must supply text; use an empty "
                    f"string for structural markers such as PAGE_BREAK"
                )

        return self

    @property
    def is_empty(self) -> bool:
        """
        Whether extraction yielded no content-bearing text.

        Returns
        -------
        bool
            ``True`` when no block carries non-whitespace text.
        """
        return not any(block.text and block.text.strip() for block in self.blocks)

    def iter_leaf_blocks(self) -> Iterator[tuple[int, ExtractedBlock]]:
        """
        Iterate content-bearing blocks with their sequence positions.

        Yields
        ------
        tuple[int, ExtractedBlock]
            Index and block for every block having no children.
        """
        parents = {b.parent_index for b in self.blocks if b.parent_index is not None}
        for index, block in enumerate(self.blocks):
            if index not in parents:
                yield index, block


# ============================================================================
# OUTCOMES
# ============================================================================


class SkipReason(StrEnum):
    """
    Why a discovered document was not ingested.

    A skip is a normal, expected result, not a fault. Incremental
    re-ingestion over an unchanged corpus produces skips for nearly every
    record, which is precisely the intended behaviour.
    """

    DUPLICATE_CONTENT = "DUPLICATE_CONTENT"
    UNCHANGED_VERSION = "UNCHANGED_VERSION"
    EXCLUDED_BY_FILTER = "EXCLUDED_BY_FILTER"


class IngestionFailure(BaseModel):
    """
    A per-document failure record routed to quarantine.

    A single corrupt document must never abort a corpus run, so failures
    are captured as data and the stream continues.

    Attributes
    ----------
    source_uri:
        URI of the document that failed.
    stage:
        Pipeline stage in which the failure occurred.
    error_type:
        Class name of the raised exception.
    error_code:
        Stable machine-readable code from the exception.
    message:
        Human-readable description.
    retryable:
        Whether the failure was eligible for retry.
    attempts:
        Number of attempts made before giving up.
    failed_at:
        Timezone-aware instant of final failure.
    document_id:
        Canonical identifier, when failure occurred after identity
        derivation.
    traceback_digest:
        Short digest of the traceback. Identical failures share a digest,
        which makes triage across a large corpus tractable. The full
        traceback is deliberately not stored: retaining tens of thousands
        of them is unmanageable and adds nothing beyond grouping.
    context:
        Structured detail carried by the exception.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    source_uri: str = Field(min_length=1)
    stage: IngestionStage
    error_type: str = Field(min_length=1)
    error_code: str = Field(min_length=1)
    message: str
    retryable: bool
    attempts: int = Field(ge=1)
    failed_at: datetime
    document_id: Optional[str] = None
    traceback_digest: Optional[str] = None
    context: Mapping[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_exception(
        cls,
        error: PTFMIngestionError,
        *,
        source_uri: str,
        failed_at: datetime,
        attempts: int = 1,
    ) -> IngestionFailure:
        """
        Build a failure record from a raised ingestion exception.

        Parameters
        ----------
        error:
            The exception that terminated processing for this document.
        source_uri:
            URI of the document, used when the exception did not carry one.
        failed_at:
            Timezone-aware instant of final failure, supplied by the
            injected clock so that failure records are deterministic under
            test.
        attempts:
            Number of attempts made before giving up.

        Returns
        -------
        IngestionFailure
            The populated failure record.
        """
        return cls(
            source_uri=error.source_uri or source_uri,
            stage=error.stage,
            error_type=type(error).__name__,
            error_code=error.error_code,
            message=error.message,
            retryable=error.retryable,
            attempts=attempts,
            failed_at=failed_at,
            document_id=error.document_id,
            traceback_digest=_traceback_digest(error),
            context=dict(error.context),
        )


def _traceback_digest(error: BaseException) -> Optional[str]:
    """
    Compute a short digest identifying a traceback shape.

    Uses ``hashlib`` directly rather than the injected ``HashProvider``:
    this is diagnostic grouping, explicitly not document identity, and
    routing it through the identity abstraction would blur that boundary.

    Parameters
    ----------
    error:
        The exception whose traceback should be digested.

    Returns
    -------
    Optional[str]
        Sixteen hexadecimal characters, or ``None`` if the exception
        carries no traceback.
    """
    if error.__traceback__ is None:
        return None
    frames = traceback.format_tb(error.__traceback__)
    joined = "".join(frames).encode("utf-8")
    return hashlib.blake2b(joined, digest_size=8).hexdigest()


class DocumentIngested(BaseModel):
    """
    A document was successfully normalized into the canonical schema.

    Attributes
    ----------
    kind:
        Discriminator for the outcome union.
    document:
        The canonical document.
    duration_ms:
        Wall-clock time spent processing this document.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["INGESTED"] = "INGESTED"
    document: RawDocument
    duration_ms: float = Field(ge=0.0)

    @property
    def source_uri(self) -> str:
        """
        URI of the ingested document.

        Returns
        -------
        str
            The source URI.
        """
        return self.document.source.source_uri


class DocumentSkipped(BaseModel):
    """
    A discovered document was intentionally not ingested.

    Attributes
    ----------
    kind:
        Discriminator for the outcome union.
    source_uri:
        URI of the skipped document.
    reason:
        Why the document was skipped.
    existing_document_id:
        Identifier of the previously ingested document, when the skip was
        caused by duplication or an unchanged version.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["SKIPPED"] = "SKIPPED"
    source_uri: str = Field(min_length=1)
    reason: SkipReason
    existing_document_id: Optional[str] = None


class DocumentFailed(BaseModel):
    """
    A document could not be ingested.

    Attributes
    ----------
    kind:
        Discriminator for the outcome union.
    failure:
        The failure record.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["FAILED"] = "FAILED"
    failure: IngestionFailure

    @property
    def source_uri(self) -> str:
        """
        URI of the document that failed.

        Returns
        -------
        str
            The source URI.
        """
        return self.failure.source_uri


#: Per-document result yielded by the ingestion engine. Discriminated on
#: ``kind`` so that consumers exhaust the union under static analysis and
#: so that outcomes serialize and deserialize without ambiguity.
IngestionOutcome: TypeAlias = Annotated[
    DocumentIngested | DocumentSkipped | DocumentFailed,
    Field(discriminator="kind"),
]


# ============================================================================
# REPORTING
# ============================================================================


class IngestionReport(BaseModel):
    """
    Run-level aggregate produced when an ingestion run completes.

    Attributes
    ----------
    run_id:
        Correlation identifier for the run, from the injected generator.
    root_uri:
        Root URI that was ingested.
    started_at:
        Timezone-aware instant the run began.
    completed_at:
        Timezone-aware instant the run ended.
    discovered_count:
        Candidates yielded by discovery.
    ingested_count:
        Documents successfully normalized.
    skipped_count:
        Documents intentionally not ingested.
    failed_count:
        Documents that could not be ingested.
    failures_by_stage:
        Failure counts keyed by pipeline stage.
    failures_by_code:
        Failure counts keyed by error code.
    documents_by_media_type:
        Successful ingestion counts keyed by media type.
    total_bytes:
        Total payload bytes transferred.
    aborted:
        Whether the run terminated early.
    abort_reason:
        Explanation when the run was aborted.

    Raises
    ------
    ValueError
        If the completion instant precedes the start instant.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    run_id: str = Field(min_length=1)
    root_uri: str = Field(min_length=1)
    started_at: datetime
    completed_at: datetime
    discovered_count: int = Field(default=0, ge=0)
    ingested_count: int = Field(default=0, ge=0)
    skipped_count: int = Field(default=0, ge=0)
    failed_count: int = Field(default=0, ge=0)
    failures_by_stage: Mapping[str, int] = Field(default_factory=dict)
    failures_by_code: Mapping[str, int] = Field(default_factory=dict)
    documents_by_media_type: Mapping[str, int] = Field(default_factory=dict)
    total_bytes: int = Field(default=0, ge=0)
    aborted: bool = False
    abort_reason: Optional[str] = None

    @model_validator(mode="after")
    def _validate_interval(self) -> IngestionReport:
        """
        Ensure the run interval is well ordered.

        Returns
        -------
        IngestionReport
            The validated instance.

        Raises
        ------
        ValueError
            If ``completed_at`` precedes ``started_at``.
        """
        if self.completed_at < self.started_at:
            raise ValueError(
                "IngestionReport.completed_at must not precede started_at"
            )
        return self

    @property
    def processed_count(self) -> int:
        """
        Number of documents that reached a terminal outcome.

        Returns
        -------
        int
            Sum of ingested, skipped, and failed counts.
        """
        return self.ingested_count + self.skipped_count + self.failed_count

    @property
    def failure_ratio(self) -> float:
        """
        Proportion of processed documents that failed.

        This is a computed diagnostic only. The engine compares it against
        a configuration-injected value to decide whether a run is
        systemically bad; no threshold is expressed here.

        Returns
        -------
        float
            Value in ``[0, 1]``; zero when nothing was processed.
        """
        if self.processed_count == 0:
            return 0.0
        return self.failed_count / self.processed_count

    @property
    def duration_seconds(self) -> float:
        """
        Wall-clock duration of the run.

        Returns
        -------
        float
            Elapsed seconds.
        """
        return (self.completed_at - self.started_at).total_seconds()