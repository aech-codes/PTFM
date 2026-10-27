"""
PTFM Knowledge Processing Internal Models

File:
    ptfm/core/knowledge/models.py

Description:
    Defines the models exchanged between stages of the Knowledge Processing
    Module. These are INTERNAL to knowledge processing. The cross-module
    contracts are ``KnowledgeNode``, ``NodeEmbedding`` and ``NodeRelation``,
    declared in ``ptfm.core.schemas.knowledge``; nothing outside this package
    should depend on the types declared here, with the exception of the
    outcome union that the engine yields and the unit that a sink receives.

Pipeline correspondence
-----------------------
    ChunkPlan          where node boundaries will fall
    PlannedChunk       one prospective node, named by span and blocks
    KnowledgeUnit      a produced node with its embedding and relations
    ProcessingOutcome  per-item result yielded by the engine
    ProcessingReport   run-level aggregate

A plan names spans; it does not build nodes
-------------------------------------------
    ``PlannedChunk`` carries a span and the blocks it covers, and nothing
    else. It deliberately does NOT carry the node type, heading path, or
    text.

    Those are derived by the assembly stage from the source document. A
    strategy that declared them could mislabel a chunk, and nothing
    downstream would detect it: a table region asserted to be prose would be
    scored and rendered as prose forever. Deriving them from the document
    makes the label unforgeable, and reduces a strategy's responsibility to
    the one decision it is actually qualified to make.

    The block references are the cross-check. A strategy states which blocks
    it intended to cover, and assembly verifies that those blocks' spans are
    consistent with the chunk span. A disagreement between the two indicates a
    defective strategy and fails the plan rather than producing a node.

Outcome granularity
-------------------
    The stream is node-valued, because a node is what a sink consumes and
    what retrieval indexes. Skips and failures are document-scoped, because a
    document that yields nothing yields no node to attach an outcome to.

    The union is therefore mixed by necessity rather than by oversight, and
    the member names say so: ``NodeProduced`` against ``DocumentSkipped`` and
    ``DocumentFailed``.

Dependency contract
-------------------
    Imports only the standard library, Pydantic,
    ``ptfm.core.knowledge.exceptions`` and ``ptfm.core.schemas``. It must NOT
    import ``interfaces``, which imports these models under ``TYPE_CHECKING``;
    the dependency runs in one direction only.
"""

from __future__ import annotations

import hashlib
import traceback
from datetime import datetime
from enum import StrEnum
from typing import Annotated, Any, Iterator, Literal, Mapping, Optional, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ptfm.core.knowledge.exceptions import KnowledgeStage, PTFMKnowledgeError
from ptfm.core.schemas import (
    KnowledgeNode,
    NodeEmbedding,
    NodeRelation,
    NodeType,
    TextSpan,
)

__all__ = [
    # Re-export for a single consumer import site
    "KnowledgeStage",
    # Planning
    "PlannedChunk",
    "ChunkPlan",
    # Production
    "KnowledgeUnit",
    # Outcomes
    "SkipReason",
    "ProcessingFailure",
    "NodeProduced",
    "DocumentSkipped",
    "DocumentFailed",
    "ProcessingOutcome",
    # Reporting
    "ProcessingReport",
]


# ============================================================================
# PLANNING
# ============================================================================


class PlannedChunk(BaseModel):
    """
    One prospective knowledge node, named by span and originating blocks.

    Produced by a chunking strategy and consumed by the assembly stage. It
    describes where a node will fall, not what the node will contain: the
    text, node type, and heading path are derived by assembly from the source
    document.

    Attributes
    ----------
    span:
        Character range into the source document text. The same span type the
        ingestion and knowledge layers use, so a planned boundary is directly
        comparable to a block boundary.
    block_ids:
        Identifiers of the blocks this chunk covers, in document order. States
        the strategy's intent, which assembly cross-checks against the span. A
        disagreement indicates a defective strategy and fails the plan.

    Raises
    ------
    ValueError
        If no block is named, or a block is named twice.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    span: TextSpan
    block_ids: tuple[str, ...]

    @model_validator(mode="after")
    def _validate_block_references(self) -> PlannedChunk:
        """
        Ensure the chunk names the blocks it intends to cover.

        Returns
        -------
        PlannedChunk
            The validated instance.

        Raises
        ------
        ValueError
            If no block is named, or a block is named twice.
        """
        if not self.block_ids:
            raise ValueError(
                "PlannedChunk must name at least one block; a chunk covering "
                "no block cannot be cross-checked against the document, and "
                "the node it produces could not be traced to structure"
            )
        if len(set(self.block_ids)) != len(self.block_ids):
            raise ValueError(
                f"PlannedChunk block_ids contains duplicates: "
                f"{list(self.block_ids)}"
            )
        return self


class ChunkPlan(BaseModel):
    """
    The complete set of node boundaries for one document.

    Verified before any node is constructed, so that a defective strategy
    fails at the plan rather than by producing a corpus whose provenance
    cannot be trusted.

    Invariants enforced at construction
    -----------------------------------
    1. Chunk spans are in ascending order.
    2. Chunk spans do not overlap.
    3. No block is claimed by two chunks.

    Invariants requiring the source document, namely that spans fall within
    the text, that block references resolve, and that every non-whitespace
    character is covered, are verified by the planning stage.

    Attributes
    ----------
    document_id:
        Identifier of the document being planned.
    strategy_name:
        Identifier of the strategy that produced the plan, recorded so that a
        plan failure names the component responsible.
    chunks:
        The planned boundaries, in document order. May be empty for a document
        with no extractable content, which is reported as a skipped outcome
        rather than a failure.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    document_id: str = Field(min_length=1)
    strategy_name: str = Field(min_length=1)
    chunks: tuple[PlannedChunk, ...] = ()

    @model_validator(mode="after")
    def _validate_chunk_ordering(self) -> ChunkPlan:
        """
        Ensure chunks are ordered and do not overlap.

        Overlapping chunks would duplicate text across nodes, inflating the
        index and distorting the term frequency statistics retrieval depends
        on. Unordered chunks would make document ordinals meaningless.

        Returns
        -------
        ChunkPlan
            The validated instance.

        Raises
        ------
        ValueError
            If spans are unordered or overlap.
        """
        previous_end = 0
        previous_index = -1

        for index, chunk in enumerate(self.chunks):
            if chunk.span.start < previous_end:
                raise ValueError(
                    f"Chunk {index} starts at {chunk.span.start}, overlapping "
                    f"or preceding chunk {previous_index} which ends at "
                    f"{previous_end}. Chunk spans must be ascending and "
                    f"disjoint."
                )
            previous_end = chunk.span.end
            previous_index = index

        return self

    @model_validator(mode="after")
    def _validate_block_exclusivity(self) -> ChunkPlan:
        """
        Ensure no block is claimed by two chunks.

        A block claimed twice would place its text in two nodes, which
        contradicts the disjointness the span ordering already establishes and
        indicates a strategy that has lost track of its own traversal.

        Returns
        -------
        ChunkPlan
            The validated instance.

        Raises
        ------
        ValueError
            If a block appears in more than one chunk.
        """
        seen: dict[str, int] = {}

        for index, chunk in enumerate(self.chunks):
            for block_id in chunk.block_ids:
                if block_id in seen:
                    raise ValueError(
                        f"Block {block_id!r} is claimed by both chunk "
                        f"{seen[block_id]} and chunk {index}"
                    )
                seen[block_id] = index

        return self

    @property
    def chunk_count(self) -> int:
        """
        Number of planned chunks.

        Returns
        -------
        int
            Chunk count; zero for a document with no extractable content.
        """
        return len(self.chunks)

    @property
    def is_empty(self) -> bool:
        """
        Whether the plan names no chunks.

        Returns
        -------
        bool
            ``True`` when the document yields no nodes.
        """
        return not self.chunks

    @property
    def covered_chars(self) -> int:
        """
        Total characters covered by the planned chunks.

        Excludes the whitespace between chunks, so this is content rather than
        document length.

        Returns
        -------
        int
            Sum of chunk span lengths.
        """
        return sum(chunk.span.length for chunk in self.chunks)

    @property
    def block_ids(self) -> tuple[str, ...]:
        """
        Every block claimed by the plan, in document order.

        Returns
        -------
        tuple[str, ...]
            Block identifiers, free of duplicates by invariant.
        """
        return tuple(
            block_id for chunk in self.chunks for block_id in chunk.block_ids
        )


# ============================================================================
# PRODUCTION
# ============================================================================


class KnowledgeUnit(BaseModel):
    """
    A produced node together with its embedding and outgoing relations.

    The artifact a sink receives. The three are carried together so that a
    sink may write them atomically where its storage permits, rather than
    reconstructing their association from three separate streams.

    Invariants enforced at construction
    -----------------------------------
    1. The embedding, when present, belongs to this node.
    2. Every relation originates from this node.

    Both guard the same failure: an artifact attached to the wrong node would
    corrupt a store silently, and no downstream check could detect it. A
    vector belonging to another node would return the wrong text for a query,
    and a relation attributed to the wrong origin would misdirect structural
    traversal.

    Attributes
    ----------
    node:
        The knowledge node.
    embedding:
        The node's vector, when one has been produced. Absent where embedding
        is disabled or deferred, since embeddings are a separate artifact with
        their own lifecycle and a corpus may be embedded later.
    relations:
        Structural relations originating from this node. Incoming relations
        belong to the unit of their own origin, so that each relation is
        written exactly once.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    node: KnowledgeNode
    embedding: Optional[NodeEmbedding] = None
    relations: tuple[NodeRelation, ...] = ()

    @model_validator(mode="after")
    def _validate_artifact_ownership(self) -> KnowledgeUnit:
        """
        Ensure every artifact in the unit belongs to its node.

        Returns
        -------
        KnowledgeUnit
            The validated instance.

        Raises
        ------
        ValueError
            If the embedding belongs to another node, or a relation does not
            originate from this node.
        """
        if self.embedding is not None:
            if self.embedding.node_id != self.node.node_id:
                raise ValueError(
                    f"KnowledgeUnit embedding belongs to node "
                    f"{self.embedding.node_id!r} but the unit's node is "
                    f"{self.node.node_id!r}. A vector attached to the wrong "
                    f"node would return the wrong text for a query, with no "
                    f"downstream check able to detect it."
                )
            if self.embedding.document_id != self.node.document_id:
                raise ValueError(
                    f"KnowledgeUnit embedding names document "
                    f"{self.embedding.document_id!r} but the unit's node "
                    f"belongs to {self.node.document_id!r}"
                )

        for relation in self.relations:
            if relation.source_node_id != self.node.node_id:
                raise ValueError(
                    f"KnowledgeUnit relation originates from "
                    f"{relation.source_node_id!r} but the unit's node is "
                    f"{self.node.node_id!r}. A unit carries only its own "
                    f"outgoing relations, so that each relation is written "
                    f"exactly once."
                )

        return self

    @property
    def node_id(self) -> str:
        """
        Identifier of this unit's node.

        Returns
        -------
        str
            The node identifier.
        """
        return self.node.node_id

    @property
    def document_id(self) -> str:
        """
        Identifier of the source document.

        Returns
        -------
        str
            The document identifier.
        """
        return self.node.document_id

    @property
    def is_embedded(self) -> bool:
        """
        Whether a vector has been produced for this node.

        Returns
        -------
        bool
            ``True`` when an embedding is present.
        """
        return self.embedding is not None


# ============================================================================
# OUTCOMES
# ============================================================================


class SkipReason(StrEnum):
    """
    Why a document produced no knowledge nodes.

    A skip is a normal result, not a fault. A document that yields nothing is
    a routine corpus occurrence, and a repeated run over an unchanged corpus
    produces skips for nearly every document, which is the intended behaviour.
    """

    #: The document contains no extractable content. Ordinarily a scanned
    #: document ingested without optical character recognition.
    NO_EXTRACTABLE_CONTENT = "NO_EXTRACTABLE_CONTENT"
    #: The document has already been processed under the same segmentation, so
    #: reprocessing would reproduce identical nodes and identifiers.
    ALREADY_PROCESSED = "ALREADY_PROCESSED"
    #: The document was rejected by a selection filter.
    EXCLUDED_BY_FILTER = "EXCLUDED_BY_FILTER"


class ProcessingFailure(BaseModel):
    """
    A per-item failure record routed to quarantine.

    A single failing document must never abort a corpus run, so failures are
    captured as data and the stream continues.

    Attributes
    ----------
    document_id:
        Identifier of the document that failed.
    node_id:
        Identifier of the node being produced, when failure occurred after
        identity derivation.
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
    traceback_digest:
        Short digest of the traceback. Identical failures share a digest,
        which makes triage across a large corpus tractable. The full traceback
        is deliberately not stored: retaining tens of thousands of them is
        unmanageable and adds nothing beyond grouping.
    context:
        Structured detail carried by the exception.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    document_id: str = Field(min_length=1)
    node_id: Optional[str] = None
    stage: KnowledgeStage
    error_type: str = Field(min_length=1)
    error_code: str = Field(min_length=1)
    message: str
    retryable: bool
    attempts: int = Field(ge=1)
    failed_at: datetime
    traceback_digest: Optional[str] = None
    context: Mapping[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_exception(
        cls,
        error: PTFMKnowledgeError,
        *,
        document_id: str,
        failed_at: datetime,
        attempts: int = 1,
    ) -> ProcessingFailure:
        """
        Build a failure record from a raised knowledge processing exception.

        Parameters
        ----------
        error:
            The exception that terminated processing.
        document_id:
            Identifier of the document, used when the exception did not carry
            one.
        failed_at:
            Timezone-aware instant of final failure, supplied by the injected
            clock so that failure records are deterministic under test.
        attempts:
            Number of attempts made before giving up.

        Returns
        -------
        ProcessingFailure
            The populated failure record.
        """
        return cls(
            document_id=error.document_id or document_id,
            node_id=error.node_id,
            stage=error.stage,
            error_type=type(error).__name__,
            error_code=error.error_code,
            message=error.message,
            retryable=error.retryable,
            attempts=attempts,
            failed_at=failed_at,
            traceback_digest=_traceback_digest(error),
            context=dict(error.context),
        )


def _traceback_digest(error: BaseException) -> Optional[str]:
    """
    Compute a short digest identifying a traceback shape.

    Uses ``hashlib`` directly rather than an injected hash provider: this is
    diagnostic grouping, explicitly not artifact identity, and routing it
    through the identity abstraction would blur a boundary that matters.

    Parameters
    ----------
    error:
        The exception whose traceback should be digested.

    Returns
    -------
    Optional[str]
        Sixteen hexadecimal characters, or ``None`` if the exception carries
        no traceback.
    """
    if error.__traceback__ is None:
        return None
    joined = "".join(traceback.format_tb(error.__traceback__)).encode("utf-8")
    return hashlib.blake2b(joined, digest_size=8).hexdigest()


class NodeProduced(BaseModel):
    """
    A knowledge node was successfully produced.

    Node-valued rather than document-valued, because a node is what a sink
    consumes and what retrieval indexes, and because materializing every node
    of a document before yielding any would defeat the memory bound streaming
    provides.

    Attributes
    ----------
    kind:
        Discriminator for the outcome union.
    unit:
        The node with its embedding and outgoing relations.
    duration_ms:
        Wall-clock time spent producing this node.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["NODE_PRODUCED"] = "NODE_PRODUCED"
    unit: KnowledgeUnit
    duration_ms: float = Field(ge=0.0)

    @property
    def document_id(self) -> str:
        """
        Identifier of the source document.

        Returns
        -------
        str
            The document identifier.
        """
        return self.unit.document_id

    @property
    def node_id(self) -> str:
        """
        Identifier of the produced node.

        Returns
        -------
        str
            The node identifier.
        """
        return self.unit.node_id


class DocumentSkipped(BaseModel):
    """
    A document was intentionally not processed, or produced no nodes.

    Document-valued rather than node-valued: a document that yields nothing
    yields no node to attach an outcome to.

    Attributes
    ----------
    kind:
        Discriminator for the outcome union.
    document_id:
        Identifier of the skipped document.
    reason:
        Why the document produced no nodes.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["DOCUMENT_SKIPPED"] = "DOCUMENT_SKIPPED"
    document_id: str = Field(min_length=1)
    reason: SkipReason


class DocumentFailed(BaseModel):
    """
    A document could not be processed.

    Document-valued, since a failure may occur before any node exists. Where
    it occurs after identity derivation, the failure record names the node.

    Attributes
    ----------
    kind:
        Discriminator for the outcome union.
    failure:
        The failure record.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["DOCUMENT_FAILED"] = "DOCUMENT_FAILED"
    failure: ProcessingFailure

    @property
    def document_id(self) -> str:
        """
        Identifier of the document that failed.

        Returns
        -------
        str
            The document identifier.
        """
        return self.failure.document_id


#: Per-item result yielded by the knowledge processing engine. Discriminated
#: on ``kind`` so that consumers exhaust the union under static analysis and
#: so that outcomes serialize and deserialize without ambiguity.
#:
#: The union is mixed in granularity by necessity: production is node-valued
#: because a node is the unit a sink consumes, while skips and failures are
#: document-valued because a document yielding nothing has no node to which an
#: outcome could attach.
ProcessingOutcome: TypeAlias = Annotated[
    NodeProduced | DocumentSkipped | DocumentFailed,
    Field(discriminator="kind"),
]


# ============================================================================
# REPORTING
# ============================================================================


class ProcessingReport(BaseModel):
    """
    Run-level aggregate produced when a processing run completes.

    Attributes
    ----------
    run_id:
        Correlation identifier for the run, from the injected generator.
    segmentation_fingerprint:
        The segmentation under which this corpus was produced. Recorded
        because node identity depends on it: a later run under a different
        fingerprint produces a disjoint identifier space, and a report that
        did not name its own segmentation could not be compared against one
        that did.
    embedding_model_identity:
        Identity of the model used, when embedding ran. Absent when embedding
        was disabled or deferred, since a corpus may be embedded later.
    started_at:
        Timezone-aware instant the run began.
    completed_at:
        Timezone-aware instant the run ended.
    documents_processed:
        Documents that produced at least one node.
    documents_skipped:
        Documents that produced no nodes.
    documents_failed:
        Documents that could not be processed.
    nodes_produced:
        Total nodes produced.
    embeddings_produced:
        Total vectors produced.
    relations_produced:
        Total structural relations produced.
    total_node_chars:
        Total characters across all node texts.
    nodes_by_type:
        Node counts keyed by structural type.
    skips_by_reason:
        Skip counts keyed by reason.
    failures_by_stage:
        Failure counts keyed by pipeline stage.
    failures_by_code:
        Failure counts keyed by error code.
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
    segmentation_fingerprint: str = Field(min_length=1)
    embedding_model_identity: Optional[str] = None
    started_at: datetime
    completed_at: datetime
    documents_processed: int = Field(default=0, ge=0)
    documents_skipped: int = Field(default=0, ge=0)
    documents_failed: int = Field(default=0, ge=0)
    nodes_produced: int = Field(default=0, ge=0)
    embeddings_produced: int = Field(default=0, ge=0)
    relations_produced: int = Field(default=0, ge=0)
    total_node_chars: int = Field(default=0, ge=0)
    nodes_by_type: Mapping[str, int] = Field(default_factory=dict)
    skips_by_reason: Mapping[str, int] = Field(default_factory=dict)
    failures_by_stage: Mapping[str, int] = Field(default_factory=dict)
    failures_by_code: Mapping[str, int] = Field(default_factory=dict)
    aborted: bool = False
    abort_reason: Optional[str] = None

    @model_validator(mode="after")
    def _validate_interval(self) -> ProcessingReport:
        """
        Ensure the run interval is well ordered.

        Returns
        -------
        ProcessingReport
            The validated instance.

        Raises
        ------
        ValueError
            If ``completed_at`` precedes ``started_at``.
        """
        if self.completed_at < self.started_at:
            raise ValueError(
                "ProcessingReport.completed_at must not precede started_at"
            )
        return self

    @property
    def documents_seen(self) -> int:
        """
        Number of documents that reached a terminal outcome.

        Returns
        -------
        int
            Sum of processed, skipped, and failed counts.
        """
        return (
            self.documents_processed
            + self.documents_skipped
            + self.documents_failed
        )

    @property
    def failure_ratio(self) -> float:
        """
        Proportion of documents that failed.

        A computed diagnostic only. The engine compares it against a
        configuration-injected value to decide whether a run is systemically
        bad; no threshold is expressed here.

        Returns
        -------
        float
            Value in ``[0, 1]``; zero when no document was seen.
        """
        if self.documents_seen == 0:
            return 0.0
        return self.documents_failed / self.documents_seen

    @property
    def embedding_coverage(self) -> float:
        """
        Proportion of produced nodes that carry a vector.

        A diagnostic over completeness, not a quality measure. A value below
        one means a corpus is partially embedded, which retrieval must know
        because an unembedded node is invisible to dense search.

        Returns
        -------
        float
            Value in ``[0, 1]``; zero when no node was produced.
        """
        if self.nodes_produced == 0:
            return 0.0
        return self.embeddings_produced / self.nodes_produced

    @property
    def mean_node_chars(self) -> float:
        """
        Average node length in characters.

        Returns
        -------
        float
            Mean length; zero when no node was produced.
        """
        if self.nodes_produced == 0:
            return 0.0
        return self.total_node_chars / self.nodes_produced

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

    def iter_node_types(self) -> Iterator[tuple[NodeType, int]]:
        """
        Iterate node counts as typed enum members.

        Counts are stored keyed by string so that the report serializes
        cleanly. This accessor restores the enum for a consumer that wants it.

        Yields
        ------
        tuple[NodeType, int]
            Each recognized node type and its count. Unrecognized keys, which
            a future schema version could introduce, are omitted.
        """
        for name, count in self.nodes_by_type.items():
            try:
                yield NodeType(name), count
            except ValueError:
                continue
