"""
PTFM Core Data Schemas

File:
    ptfm/core/schemas.py

Description:
    This module defines the canonical data schemas exchanged between
    all core modules of the Probably Truthful Foundation Model (PTFM).

Pipeline:

    User Query
        ↓
    Retrieval
        ↓
    LLM Generation
        ↓
    Claim Extraction
        ↓
    Verification
        ↓
    Truth Space
        ↓
    Confidence Engine
        ↓
    Decision Engine
        ↓
    Verified Response

These schemas are intentionally domain-agnostic.
Domain-specific information must be implemented through plugins.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field


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


class ConfidenceLevel(str, Enum):
    """Human-readable confidence category."""

    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


# ============================================================================
# USER INPUT
# ============================================================================


class UserQuery(BaseModel):
    """Incoming user request."""

    model_config = ConfigDict(extra="forbid")

    query_id: UUID = Field(default_factory=uuid4)
    query: str
    language: str = "en"
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = Field(default_factory=dict)


# ============================================================================
# RETRIEVAL MODULE
# ============================================================================


class RetrievalRequest(BaseModel):
    """Request sent to retrieval engine."""

    query: str
    top_k: int = Field(default=10, ge=1)


class RetrievedDocument(BaseModel):
    """Document retrieved from a knowledge source."""

    document_id: str
    title: Optional[str] = None
    content: str

    source_uri: str

    retrieval_method: RetrievalMethod

    retrieval_score: float = Field(ge=0.0, le=1.0)

    credibility_score: float = Field(ge=0.0, le=1.0)

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
# LLM MODULE
# ============================================================================


class LLMRequest(BaseModel):
    """Prompt sent to the language model."""

    prompt: str

    context_documents: List[str]


class GenerationMetadata(BaseModel):
    """Generation diagnostics."""

    model_name: str

    token_count: int

    generation_time_ms: float

    finish_reason: str


class LLMResponse(BaseModel):
    """Raw LLM response before verification."""

    response_text: str

    metadata: GenerationMetadata


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


class EvidenceItem(BaseModel):
    """Evidence supporting or contradicting a claim."""

    evidence_id: str

    document_id: str

    content: str

    source_uri: str

    retrieval_score: float = Field(ge=0.0, le=1.0)

    credibility_score: float = Field(ge=0.0, le=1.0)


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


class RetrievalMetrics(BaseModel):
    precision: float = Field(ge=0.0, le=1.0)

    recall: float = Field(ge=0.0, le=1.0)

    latency_ms: float


class VerificationMetrics(BaseModel):
    supported_claims: int

    refuted_claims: int

    insufficient_claims: int

    average_confidence: float = Field(ge=0.0, le=1.0)


class PipelineMetrics(BaseModel):
    retrieval: RetrievalMetrics

    verification: VerificationMetrics

    total_latency_ms: float


# ============================================================================
# EXECUTION TRACE
# ============================================================================


class ModuleTrace(BaseModel):
    """Execution trace for one pipeline module."""

    module_name: str

    execution_time_ms: float

    status: str


class ExecutionTrace(BaseModel):
    """Complete execution trace."""

    query_id: UUID

    modules: List[ModuleTrace]

    total_execution_time_ms: float