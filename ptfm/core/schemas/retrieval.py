"""
PTFM Retrieval Schemas

File:
    ptfm/core/schemas/retrieval.py

Layer:
    Retrieval. May import from ``common``, ``ingestion`` and ``knowledge``.

Description:
    Declares the request and result contracts of the Hybrid Retrieval Module
    (Module 3), together with ``EvidenceItem``, the unit on which verification
    operates.

Placement rationale
-------------------
    ``EvidenceItem`` is produced by retrieval and evidence fusion and consumed
    by verification, so it belongs to the retrieval layer. Placing it in the
    verification layer would invert the dependency, since retrieval would then
    import from a layer above it.

Evidence-First continuity
-------------------------
    ``RetrievedDocument`` and ``EvidenceItem`` carry the knowledge node
    identity, character span, and originating block references forward from
    the knowledge layer.

    Without them the provenance chain would terminate at retrieval: a node
    located precisely within a document would become an untyped mention of the
    document as a whole, and a citation would degrade from an exact character
    range to a whole-document reference. Verification would then have to
    reconstruct the location from untyped metadata, which no validator or
    compliance test could enforce.

    The fields are optional so that a retrieval strategy operating over whole
    documents rather than nodes, such as one serving a corpus ingested before
    knowledge processing existed, remains expressible. When a result does
    originate from a knowledge node, they must be populated.

Corroboration is counted over anchors, never over items
-------------------------------------------------------
    A retrieved node may be accompanied by neighbouring nodes that supply the
    context needed to interpret it. Those neighbours were not found by
    retrieval; they were included because they adjoin something that was.

    If verification counted them as evidence items, one finding would appear
    as several independent sources agreeing, and a claim would seem better
    corroborated than the corpus warrants. In a system whose output is
    calibrated confidence, that is a correctness defect rather than an
    inefficiency, and it is the same defect that made overlapping knowledge
    nodes unacceptable.

    Every evidence item therefore names an ANCHOR: the independently
    retrieved node it supports. An item is an anchor when
    ``node_id == anchor_node_id``. Corroboration is the number of distinct
    anchors, so context can never masquerade as an independent source.

    This is deliberately not a two-valued flag. A node may be discovered by
    several retrieval arms at once, or reached from more than one anchor, or
    introduced by a mechanism that does not yet exist. A flag would force each
    of those into a single value and would have to be revised whenever a new
    mechanism appeared, whereas counting anchors is correct without revision.

Retrieval signals are uncalibrated
----------------------------------
    Every number this layer carries from retrieval is a raw signal on a named
    scale. None is a probability, a confidence, or a judgement of truth.

    Calibrated confidence originates in the conformal calibration layer alone.
    A retrieval signal that reached a decision without passing through it
    would be a raw model score gating a truth judgement, which the
    architecture prohibits.

Stability
---------
    These are cross-module contracts. ``RetrievalRequest``,
    ``RetrievalResult``, ``EvidenceItem`` and ``AtomicClaim`` additionally
    form the interface every domain plugin implements.
"""

from __future__ import annotations

from enum import Enum, StrEnum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ptfm.core.schemas.common import TextSpan

__all__ = [
    "RetrievalMethod",
    "EvidenceType",
    "DiscoveryMethod",
    "SignalScale",
    "RetrievalSignal",
    "DiscoveryRecord",
    "RetrievalRequest",
    "RetrievedDocument",
    "RetrievalResult",
    "RetrievalStatistics",
    "EvidenceItem",
    "RetrievalMetrics",
]


# ============================================================================
# ENUMERATIONS
# ============================================================================


class RetrievalMethod(str, Enum):
    """Retrieval strategy."""

    DENSE = "DENSE"
    SPARSE = "SPARSE"
    GRAPH = "GRAPH"
    HYBRID = "HYBRID"


class EvidenceType(str, Enum):
    """Type of retrieved evidence."""

    DOCUMENT = "DOCUMENT"
    KNOWLEDGE_GRAPH = "KNOWLEDGE_GRAPH"
    DATABASE = "DATABASE"
    WEB = "WEB"
    OTHER = "OTHER"


class DiscoveryMethod(StrEnum):
    """
    A means by which a node entered a result set.

    Declared separately from :class:`RetrievalMethod` rather than extending
    it. The two describe different things: a retrieval method is a strategy
    that searches an index, while a discovery method includes ways a node can
    arrive without having been searched for at all.

    Merging them would place ``EXPANSION`` among the values a consumer
    switching on ``RetrievalMethod`` expects to be strategies, and such a
    consumer would mishandle it silently.
    """

    #: Found by vector similarity search.
    DENSE = "DENSE"
    #: Found by lexical search.
    SPARSE = "SPARSE"
    #: Found by structural traversal or a structural predicate.
    GRAPH = "GRAPH"
    #: Included because it adjoins a node that was found. Never an anchor.
    EXPANSION = "EXPANSION"
    #: Reordered by a relevance model after fusion. Never introduces a node.
    RERANK = "RERANK"


class SignalScale(StrEnum):
    """
    The scale a raw retrieval signal lives on.

    Recorded because signals from different methods are not comparable. A
    cosine similarity, a lexical relevance score, and a reciprocal rank
    fusion value occupy different ranges with different meanings, and a
    consumer that averaged or thresholded across them would produce a number
    that resembles a measurement and is not one.

    Naming the scale makes such a comparison visibly wrong rather than
    silently plausible.

    No member of this enumeration denotes a probability. Converting any of
    these into a calibrated confidence is the conformal calibration layer's
    responsibility and no other component's.
    """

    #: Cosine similarity, conventionally in [-1, 1].
    COSINE = "COSINE"
    #: Inner product, unbounded.
    INNER_PRODUCT = "INNER_PRODUCT"
    #: A distance, where smaller is nearer.
    DISTANCE = "DISTANCE"
    #: A lexical relevance score such as BM25, unbounded and corpus-relative.
    LEXICAL = "LEXICAL"
    #: A reciprocal rank fusion value, derived from ranks alone.
    RECIPROCAL_RANK = "RECIPROCAL_RANK"
    #: A relevance model's output, on that model's own scale.
    RERANK = "RERANK"
    #: An ordinal position with no underlying score.
    RANK_ONLY = "RANK_ONLY"


# ============================================================================
# SIGNALS
# ============================================================================


class RetrievalSignal(BaseModel):
    """
    One retrieval method's uncalibrated observation about one result.

    Attributes
    ----------
    method:
        The method that produced the observation.
    rank:
        Zero-based position within that method's own ordering. Always
        meaningful, whereas a raw score may be absent.
    raw_score:
        The method's own score, on :attr:`scale`. ``None`` where the method
        produces an ordering without a score, as a structural predicate does.
    scale:
        The scale :attr:`raw_score` lives on. Required whenever a score is
        present, so that a consumer cannot treat an unfamiliar number as
        though it were a probability.

    Prohibition
    -----------
    This is an observation, not a judgement. It must not be interpreted as a
    likelihood that a result is relevant, correct, or true.

    Raises
    ------
    ValueError
        If a score is present without a scale.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    method: DiscoveryMethod
    rank: int = Field(ge=0)
    raw_score: Optional[float] = None
    scale: Optional[SignalScale] = None

    @model_validator(mode="after")
    def _validate_scale_accompanies_score(self) -> RetrievalSignal:
        """
        Ensure a score is never recorded without its scale.

        An unlabelled number invites comparison with numbers from other
        methods, which is the category error this model exists to prevent.

        Returns
        -------
        RetrievalSignal
            The validated instance.

        Raises
        ------
        ValueError
            If a score is present without a scale.
        """
        if self.raw_score is not None and self.scale is None:
            raise ValueError(
                "A retrieval signal carrying a raw_score must name its scale; "
                "an unlabelled score invites comparison with scores from "
                "other methods, which are not comparable."
            )
        return self


class DiscoveryRecord(BaseModel):
    """
    One path by which a node entered a result set.

    A node may carry several of these. Being found by two retrieval arms is
    one node discovered twice, not two findings: recording both paths keeps
    the retrieval history complete while leaving the count of independent
    evidence unchanged.

    Attributes
    ----------
    method:
        How the node was reached.
    rank:
        Position within that method's ordering.
    raw_score:
        The method's own score, where it produces one.
    scale:
        The scale the score lives on.
    via_node_id:
        For an expansion, the node from which this one was reached. ``None``
        for a direct discovery.

        Present so that expansion is auditable: a reader of an evidence set
        can see not merely that a node was context, but which anchor it
        accompanied and by what route.

    Raises
    ------
    ValueError
        If a score is present without a scale, or an expansion does not name
        the node it was reached from.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    method: DiscoveryMethod
    rank: int = Field(default=0, ge=0)
    raw_score: Optional[float] = None
    scale: Optional[SignalScale] = None
    via_node_id: Optional[str] = None

    @model_validator(mode="after")
    def _validate_record(self) -> DiscoveryRecord:
        """
        Ensure the record is internally coherent.

        Returns
        -------
        DiscoveryRecord
            The validated instance.

        Raises
        ------
        ValueError
            If a score lacks a scale, or an expansion does not name its
            origin.
        """
        if self.raw_score is not None and self.scale is None:
            raise ValueError(
                "A discovery record carrying a raw_score must name its scale."
            )
        if self.method is DiscoveryMethod.EXPANSION and not self.via_node_id:
            raise ValueError(
                "An expansion record must name the node it was reached from, "
                "or the resulting evidence could not be audited: a reader "
                "could see that a node was context without being able to "
                "establish which anchor it accompanied."
            )
        return self


# ============================================================================
# RETRIEVAL REQUEST AND RESULT
# ============================================================================


class RetrievalRequest(BaseModel):
    """Request sent to retrieval engine."""

    query: str
    top_k: int = Field(default=10, ge=1)


class RetrievedDocument(BaseModel):
    """
    A unit of content retrieved from a knowledge source.

    Attributes
    ----------
    document_id:
        Identifier of the source document.
    node_id:
        Identifier of the knowledge node this result originates from, when it
        originates from one. Carries node identity forward so that a result
        can be traced to the exact unit that was indexed rather than only to
        the document containing it.
    span:
        Character range of the originating node within the source document
        text. Preserves the exact location, so a citation names a character
        range rather than a whole document.
    block_ids:
        Identifiers of the structural blocks the originating node covered.
        Recovers the document structure the flat content loses.
    signals:
        Every retrieval method's uncalibrated observation about this result.

        The authoritative record of how the result was found. A result found
        by two methods carries two signals, which is what allows a consumer to
        see that agreement without mistaking it for two separate results.
    retrieval_score:
        The fused ordinal signal, derived mechanically from :attr:`signals`.

        UNCALIBRATED. It occupies ``[0, 1]`` because the field predates this
        contract, not because it denotes a probability. It expresses relative
        position within one result set and carries no meaning across queries,
        across corpora, or as a likelihood that the result is relevant or
        true.

        Converting it into confidence is the conformal calibration layer's
        responsibility. Any component that thresholds it directly is treating
        a raw model output as a judgement.
    credibility_score:
        Source credibility as reported by a domain plugin. A property of the
        source, independent of the query, and equally uncalibrated.

        ``None`` when no credibility was available for the source.

        Absence is represented rather than filled in. Retrieval reliability is
        computed as ``r(E) = (1/k) * sum_i r_i * s_i`` and feeds hallucination
        risk, so a substituted value would enter a statistical guarantee
        indistinguishable from a measured one. On a three-item result set with
        one unscoreable source, the choice of substitute moves hallucination
        risk by more than half its value, and dropping the item instead raises
        apparent reliability by keeping only the sources that could be scored.

        A consumer must therefore decide for itself what an unknown credibility
        means. Retrieval does not decide on its behalf.
    """

    document_id: str
    node_id: Optional[str] = None
    span: Optional[TextSpan] = None
    block_ids: tuple[str, ...] = ()
    title: Optional[str] = None
    content: str

    source_uri: str

    retrieval_method: RetrievalMethod

    signals: tuple[RetrievalSignal, ...] = ()

    retrieval_score: float = Field(ge=0.0, le=1.0)

    credibility_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)

    evidence_type: EvidenceType

    metadata: Dict[str, Any] = Field(default_factory=dict)


class RetrievalResult(BaseModel):
    """Output of hybrid retrieval."""

    query: str

    documents: List[RetrievedDocument]

    retrieval_latency_ms: float

    total_documents: int


class RetrievalStatistics(BaseModel):
    """Retrieval diagnostics."""

    dense_hits: int

    sparse_hits: int

    graph_hits: int

    hybrid_score: float = Field(ge=0.0, le=1.0)


# ============================================================================
# EVIDENCE
# ============================================================================


class EvidenceItem(BaseModel):
    """
    Evidence supporting or contradicting a claim.

    The final link of the Evidence-First chain. A verification result cites
    evidence by identifier, so whatever this record fails to carry is
    unavailable to any consumer of a verified response.

    Anchors
    -------
    An item is an ANCHOR when its :attr:`node_id` equals its
    :attr:`anchor_node_id`, meaning retrieval found it. An item whose anchor
    is a different node is CONTEXT: it was included because it adjoins the
    anchor, and it was not independently found.

    Corroboration is the number of distinct anchors, never the number of
    items. Counting items would let the context surrounding one finding
    appear as several independent sources agreeing, which would make a claim
    seem better supported than the corpus warrants.

    Attributes
    ----------
    evidence_id:
        Identifier of this evidence record.
    document_id:
        Identifier of the source document.
    node_id:
        Identifier of the knowledge node this evidence originates from, when
        it originates from one.
    anchor_node_id:
        Identifier of the independently retrieved node this item supports.
        Equal to :attr:`node_id` when this item is itself an anchor.

        ``None`` only for evidence that did not originate from a knowledge
        node, which a whole-document retrieval strategy may produce.
    discovery:
        Every path by which the node entered the result set.

        A node found by two retrieval arms carries two records, which
        preserves the retrieval history without altering the count of
        independent evidence. An expansion record names the node it was
        reached from, so context is auditable rather than merely marked.
    span:
        Character range of the originating node within the source document
        text. What allows a verified claim to be traced to an exact region of
        an identified source document.
    block_ids:
        Identifiers of the structural blocks the originating node covered.
    content:
        The evidence text.
    retrieval_score:
        The fused ordinal signal. UNCALIBRATED; see
        :attr:`RetrievedDocument.retrieval_score`.
    credibility_score:
        Source credibility as reported by a domain plugin. Uncalibrated.

        ``None`` when no credibility was available for the source. It is never
        inferred, defaulted, or substituted: this value is consumed directly by
        retrieval reliability, and an invented number there would be
        indistinguishable from a measured one inside a statistical guarantee.

    Raises
    ------
    ValueError
        If an item names an anchor without naming its own node, or an item
        that is not an anchor records no discovery path.
    """

    model_config = ConfigDict(extra="forbid")

    evidence_id: str

    document_id: str

    node_id: Optional[str] = None

    anchor_node_id: Optional[str] = None

    discovery: tuple[DiscoveryRecord, ...] = ()

    span: Optional[TextSpan] = None

    block_ids: tuple[str, ...] = ()

    content: str

    source_uri: str

    retrieval_score: float = Field(ge=0.0, le=1.0)

    credibility_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _validate_anchor(self) -> EvidenceItem:
        """
        Ensure anchoring is coherent.

        Returns
        -------
        EvidenceItem
            The validated instance.

        Raises
        ------
        ValueError
            If an anchor is named without a node, or context records no path
            by which it was reached.
        """
        if self.anchor_node_id is not None and self.node_id is None:
            raise ValueError(
                "Evidence naming an anchor must also name its own node; "
                "otherwise it could not be established whether the item is "
                "the anchor or context accompanying it."
            )

        if (
            self.anchor_node_id is not None
            and self.node_id != self.anchor_node_id
            and not self.discovery
        ):
            raise ValueError(
                "Evidence that is not an anchor must record how it was "
                "reached. Context with no recorded path cannot be "
                "distinguished from an independent finding, which is what "
                "would let one finding be counted as several."
            )

        return self

    @property
    def is_anchor(self) -> bool:
        """
        Whether retrieval found this node independently.

        The unit of corroboration. An item that is not an anchor accompanies
        one and must never be counted as separate support.

        Returns
        -------
        bool
            ``True`` when this item is an anchor.
        """
        if self.node_id is None or self.anchor_node_id is None:
            # Evidence that did not originate from a knowledge node stands on
            # its own; there is no expansion mechanism that could have
            # introduced it as context.
            return True
        return self.node_id == self.anchor_node_id

    @property
    def discovery_methods(self) -> frozenset[DiscoveryMethod]:
        """
        The distinct methods by which this node was reached.

        Returns
        -------
        frozenset[DiscoveryMethod]
            Methods recorded on this item.
        """
        return frozenset(record.method for record in self.discovery)

    def was_found_by(self, method: DiscoveryMethod) -> bool:
        """
        Report whether a given method reached this node.

        Parameters
        ----------
        method:
            The method to test.

        Returns
        -------
        bool
            ``True`` when the method recorded a discovery.
        """
        return method in self.discovery_methods


# ============================================================================
# METRICS
# ============================================================================


class RetrievalMetrics(BaseModel):
    precision: float = Field(ge=0.0, le=1.0)

    recall: float = Field(ge=0.0, le=1.0)

    latency_ms: float
