"""
PTFM Core Data Schemas

File:
    ptfm/core/schemas/__init__.py

Description:
    Canonical data schemas exchanged between every module of the Probably
    Truthful Foundation Model. These are cross-module contracts: a schema is
    not owned by the module that produces it, since changing one changes the
    interface between modules.

Pipeline:

    External Sources
        ↓
    Data Ingestion              → RawDocument
        ↓
    Knowledge Processing        → KnowledgeNode, NodeEmbedding, NodeRelation
        ↓
    Hybrid Retrieval            → RetrievalResult, EvidenceItem
        ↓
    Claim Extraction            → AtomicClaim
        ↓
    Verification                → VerificationResult
        ↓
    Truth Space                 → TruthSpace
        ↓
    Confidence Calibration      → CalibrationResult
        ↓
    Decision Engine             → DecisionOutcome
        ↓
    Verified Response

Package layout
--------------
    Declarations are grouped into layers by pipeline position. The dependency
    graph is strictly acyclic and runs in pipeline order:

        common
            ↓
        ingestion
            ↓
        knowledge
            ↓
        retrieval
            ↓
        verification

    A module may import from any layer below it and none above. The rule is
    enforced by an architecture-compliance test rather than left to
    convention, so that a violation fails the build rather than accumulating.

Public interface
----------------
    Every name is re-exported here, so ``from ptfm.core.schemas import X``
    resolves exactly as it did when these declarations lived in a single
    module. Consumers should import from this package rather than from a
    layer module directly, so that future re-layering remains non-breaking.

    Re-exports are explicit rather than wildcard, which keeps the public
    surface auditable: a name added to a layer module does not silently
    become public API.

Domain neutrality
-----------------
    These schemas are domain-agnostic without exception. Legal, medical,
    financial, and every other domain concept is expressed through plugins,
    never here.
"""

from __future__ import annotations

from ptfm.core.schemas.common import (
    ExecutionTrace,
    GenerationMetadata,
    LLMRequest,
    LLMResponse,
    ModuleTrace,
    TextSpan,
    UserQuery,
)
from ptfm.core.schemas.ingestion import (
    CONTAINER_BLOCK_TYPES,
    PTFM_DOCUMENT_NAMESPACE,
    RAW_DOCUMENT_SCHEMA_VERSION,
    BlockAttributes,
    BlockIndex,
    BlockType,
    DocumentBlock,
    DocumentMetadata,
    ExtractionWarning,
    ExtractionWarningCode,
    RawDocument,
    RawDocumentBatch,
    SourceDescriptor,
    derive_document_id,
)
from ptfm.core.schemas.knowledge import (
    KNOWLEDGE_NODE_SCHEMA_VERSION,
    PTFM_NODE_NAMESPACE,
    KnowledgeNode,
    KnowledgeNodeBatch,
    NodeEmbedding,
    NodeRelation,
    NodeType,
    RelationType,
    derive_node_id,
)
from ptfm.core.schemas.retrieval import (
    EvidenceItem,
    EvidenceType,
    RetrievalMethod,
    RetrievalMetrics,
    RetrievalRequest,
    RetrievalResult,
    RetrievalStatistics,
    RetrievedDocument,
)
from ptfm.core.schemas.verification import (
    AtomicClaim,
    CalibrationResult,
    ClaimCollection,
    ConfidenceLevel,
    ConfidenceMetrics,
    DecisionEnum,
    DecisionOutcome,
    PipelineMetrics,
    RetryPlan,
    SemanticEntropy,
    TruthSpace,
    TruthSpaceClaim,
    VerificationBatch,
    VerificationMetrics,
    VerificationResult,
    VerificationStatus,
    VerifiedResponse,
)

__all__ = [
    # ---- common ----------------------------------------------------------
    "TextSpan",
    "UserQuery",
    "LLMRequest",
    "GenerationMetadata",
    "LLMResponse",
    "ModuleTrace",
    "ExecutionTrace",
    # ---- ingestion -------------------------------------------------------
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
    # ---- knowledge -------------------------------------------------------
    "KNOWLEDGE_NODE_SCHEMA_VERSION",
    "PTFM_NODE_NAMESPACE",
    "derive_node_id",
    "NodeType",
    "RelationType",
    "KnowledgeNode",
    "NodeEmbedding",
    "NodeRelation",
    "KnowledgeNodeBatch",
    # ---- retrieval -------------------------------------------------------
    "RetrievalMethod",
    "EvidenceType",
    "RetrievalRequest",
    "RetrievedDocument",
    "RetrievalResult",
    "RetrievalStatistics",
    "EvidenceItem",
    "RetrievalMetrics",
    # ---- verification ----------------------------------------------------
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
