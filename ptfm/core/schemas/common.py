"""
PTFM Common Schemas

File:
    ptfm/core/schemas/common.py

Layer:
    Base. This module MUST NOT import from any other module in the
    ``ptfm.core.schemas`` package.

Description:
    Declares the primitives and cross-cutting records shared by every layer of
    the PTFM pipeline: the character span type on which the Evidence-First
    guarantee rests, the incoming user request, the language model exchange
    records, and the execution trace.

Placement rationale
-------------------
    ``TextSpan`` is the shared primitive of the evidence chain. A
    ``DocumentBlock`` spans into ``RawDocument.text`` and a ``KnowledgeNode``
    spans into the same text, so the type is used by the ingestion and
    knowledge layers alike and belongs at the base rather than in either.

    The language model records are generic transport schemas rather than
    artifacts of retrieval or verification. They are placed here pending a
    dedicated generation layer, should a later module justify one.

Stability
---------
    These schemas are frozen cross-module contracts. Changing a field is a
    change to the contract between modules, not an internal revision.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator

__all__ = [
    "TextSpan",
    "UserQuery",
    "LLMRequest",
    "GenerationMetadata",
    "LLMResponse",
    "ModuleTrace",
    "ExecutionTrace",
]


# ============================================================================
# SPANS
# ============================================================================


class TextSpan(BaseModel):
    """
    Half-open character range ``[start, end)`` into a normalized text.

    Offsets are Python string indices over NFC-normalized text, not byte
    offsets, so they are stable across encodings.

    Used by every layer that must locate content precisely: a
    ``DocumentBlock`` spans into ``RawDocument.text``, and a
    ``KnowledgeNode`` spans into the same text. This shared indexing is what
    makes a claim traceable to an exact region of an identified source
    document.

    Raises
    ------
    ValueError
        If ``end`` precedes ``start``.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    start: int = Field(ge=0)
    end: int = Field(ge=0)

    @model_validator(mode="after")
    def _validate_ordering(self) -> TextSpan:
        """
        Ensure the span is well ordered.

        Returns
        -------
        TextSpan
            The validated instance.

        Raises
        ------
        ValueError
            If ``end < start``.
        """
        if self.end < self.start:
            raise ValueError(
                f"TextSpan end ({self.end}) must not precede start ({self.start})"
            )
        return self

    @property
    def length(self) -> int:
        """
        Number of characters covered by the span.

        Returns
        -------
        int
            ``end - start``.
        """
        return self.end - self.start


# ============================================================================
# USER INPUT
# ============================================================================


class UserQuery(BaseModel):
    """Incoming user request."""

    model_config = ConfigDict(extra="forbid")

    query_id: UUID = Field(default_factory=uuid4)
    query: str
    language: str = "en"
    # Timezone-aware. ``datetime.utcnow`` is deprecated from Python 3.12 and
    # returns a naive datetime, which cannot be compared against the aware
    # timestamps used throughout ingestion and the execution trace.
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    metadata: Dict[str, Any] = Field(default_factory=dict)


# ============================================================================
# LANGUAGE MODEL EXCHANGE
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
