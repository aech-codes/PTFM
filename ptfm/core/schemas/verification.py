"""
PTFM Verification Schemas

File:
    ptfm/core/schemas/verification.py

Layer:
    Verification. May import from ``common``, ``ingestion``, ``knowledge`` and
    ``retrieval``.

Description:
    Declares the contracts of claim extraction, neural and symbolic
    verification, the Verifiable Truth Space, conformal calibration, the
    decision engine, and the final verified response.

Placement rationale
-------------------
    ``PipelineMetrics`` composes ``RetrievalMetrics`` and
    ``VerificationMetrics``. Since it spans both, it is placed in the lower of
    the two layers, which is the only placement the acyclic layering permits.

Confidence contract
-------------------
    Every confidence value declared in this layer originates from the
    conformal calibration layer. Raw neural scores are prohibited outside it,
    so no schema below this layer may carry a calibrated confidence, and no
    schema here may be populated from an uncalibrated model output.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from ptfm.core.schemas.retrieval import RetrievalMetrics

__all__ = [
    "DecisionEnum",
    "VerificationStatus",
    "ConfidenceLevel",
    "AtomicClaim",
    "ClaimCollection",
    "VerificationResult",
    "VerificationBatch",
    "TruthSpaceClaim",
    "TruthSpace",
    "SemanticEntropy",
    "CalibrationResult",
    "ConfidenceMetrics",
    "RetryPlan",
    "DecisionOutcome",
    "VerifiedResponse",
    "VerificationMetrics",
    "PipelineMetrics",
]


# ============================================================================
# ENUMERATIONS
# ============================================================================


class DecisionEnum(str, Enum):
    """Final decision returned by the Decision Engine."""

    GENERATE = "GENERATE"
    RETRY = "RETRY"
    REFUSE = "REFUSE"


class VerificationStatus(str, Enum):
    """Verification result for an atomic claim."""

    SUPPORTED = "SUPPORTED"
    REFUTED = "REFUTED"
    INSUFFICIENT_INFO = "INSUFFICIENT_INFO"


class ConfidenceLevel(str, Enum):
    """Human-readable confidence category."""

    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


# ============================================================================
# CLAIM EXTRACTION
# ============================================================================


class AtomicClaim(BaseModel):
    """Single independently verifiable factual claim."""

    claim_id: str

    claim_text: str

    source_sentence: str

    sentence_index: int

    entities: List[str] = Field(default_factory=list)

    relations: List[str] = Field(default_factory=list)


class ClaimCollection(BaseModel):
    """Collection of extracted claims."""

    claims: List[AtomicClaim]


# ============================================================================
# VERIFICATION
# ============================================================================


class VerificationResult(BaseModel):
    """Verification outcome for one claim."""

    claim_id: str

    status: VerificationStatus

    nli_score: float = Field(ge=0.0, le=1.0)

    symbolic_verification: bool

    supporting_evidence: List[str] = Field(default_factory=list)

    contradicting_evidence: List[str] = Field(default_factory=list)

    explanation: Optional[str] = None


class VerificationBatch(BaseModel):
    """Verification results for all claims."""

    results: List[VerificationResult]


# ============================================================================
# TRUTH SPACE
# ============================================================================


class TruthSpaceClaim(BaseModel):
    """Membership information for the Verifiable Truth Space."""

    claim_id: str

    membership_probability: float = Field(ge=0.0, le=1.0)

    is_member: bool

    confidence: float = Field(ge=0.0, le=1.0)

    verification_status: VerificationStatus


class TruthSpace(BaseModel):
    """Complete Verifiable Truth Space."""

    verified_claims: List[TruthSpaceClaim]


# ============================================================================
# CONFIDENCE ENGINE
# ============================================================================


class SemanticEntropy(BaseModel):
    """Semantic uncertainty metrics."""

    entropy: float = Field(ge=0.0)

    num_generations: int = Field(ge=1)


class CalibrationResult(BaseModel):
    """Output of conformal calibration."""

    calibrated_confidence: float = Field(ge=0.0, le=1.0)

    nonconformity_score: float = Field(ge=0.0)

    conformal_threshold: float = Field(ge=0.0, le=1.0)


class ConfidenceMetrics(BaseModel):
    """Aggregated confidence metrics."""

    semantic_entropy: float

    retrieval_reliability: float

    calibrated_confidence: float

    response_confidence: float

    confidence_level: ConfidenceLevel


# ============================================================================
# DECISION ENGINE
# ============================================================================


class RetryPlan(BaseModel):
    """Retry strategy."""

    retry_count: int = Field(ge=0)

    expanded_queries: List[str] = Field(default_factory=list)


class DecisionOutcome(BaseModel):
    """Decision engine output."""

    decision: DecisionEnum

    confidence: float = Field(ge=0.0, le=1.0)

    reason: str

    retry_plan: Optional[RetryPlan] = None


# ============================================================================
# VERIFIED RESPONSE
# ============================================================================


class VerifiedResponse(BaseModel):
    """Final response returned to the user."""

    response_text: Optional[str]

    decision: DecisionEnum

    overall_confidence: float = Field(ge=0.0, le=1.0)

    verified_claims: List[TruthSpaceClaim]

    supporting_sources: List[str]

    diagnostics: Dict[str, Any] = Field(default_factory=dict)

    latency_ms: float


# ============================================================================
# PIPELINE METRICS
# ============================================================================


class VerificationMetrics(BaseModel):
    supported_claims: int

    refuted_claims: int

    insufficient_claims: int

    average_confidence: float = Field(ge=0.0, le=1.0)


class PipelineMetrics(BaseModel):
    retrieval: RetrievalMetrics

    verification: VerificationMetrics

    total_latency_ms: float
