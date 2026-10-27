"""
PTFM Knowledge Processing Exception Hierarchy

File:
    ptfm/core/knowledge/exceptions.py

Description:
    Defines every exception raised by the Knowledge Processing Module, rooted
    in ``PTFMError``. The hierarchy mirrors the pipeline stages so that a
    failure can be attributed to a stage without string parsing.

Pipeline stage mapping
----------------------

    Configuration   -> KnowledgeConfigurationError
    Planning        -> PlanningError
    Assembly        -> AssemblyError
    Relations       -> RelationError
    Enrichment      -> EnrichmentError
    Validation      -> KnowledgeValidationError
    Composition     -> CompositionError
    Embedding       -> EmbeddingError
    Sink            -> SinkError

Design contract
---------------
    - ``stage`` and ``retryable`` are class-level constants. Raise sites never
      supply them, which makes stage attribution unforgeable.

    - Only genuinely transient conditions are retryable. An embedding provider
      being briefly unreachable is transient; a dimension mismatch, a span
      integrity violation, or an invalid plan are deterministic and retrying
      them wastes budget while delaying quarantine.

    - A document that yields no knowledge nodes is NOT an error. An empty or
      whitespace-only document is a normal corpus occurrence and is reported
      as a skipped outcome, following the precedent set by duplicate documents
      in ingestion. Exceptions are for faults, not for control flow.

    - Enrichment failures are recoverable by design. Enrichment is additive
      and produces metadata only, so a failed enricher degrades a node rather
      than losing it. Whether a failure is tolerated or fatal is a
      configuration decision, and both paths need a typed exception.

    - This module imports only from the standard library and from
      ``ptfm.utils.exceptions``. No schema, model, or third-party type may
      leak into a caller's ``except`` clause.

Evidence-First enforcement
--------------------------
    Three exceptions exist solely to make the provenance guarantee
    enforceable at runtime rather than aspirational:

        NodeSpanIntegrityError
            A node's text is not the exact substring of the source document
            at its recorded span.

        BlockReferenceError
            A node references blocks that are absent from, or out of order
            in, the source document.

        CoverageError
            Document content exists that no node covers, meaning a claim
            could be unverifiable because its supporting text was silently
            dropped during segmentation.

    Each fails the affected document rather than emitting a node whose span
    cannot be trusted to locate evidence.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any, ClassVar, Mapping, Sequence

from ptfm.utils.exceptions import PTFMError

__all__ = [
    "KnowledgeStage",
    "PTFMKnowledgeError",
    # Configuration
    "KnowledgeConfigurationError",
    "ChunkerRegistrationError",
    "EmbedderRegistrationError",
    "ProviderUnavailableError",
    "SegmentationFingerprintMismatchError",
    # Planning
    "PlanningError",
    "InvalidChunkPlanError",
    "UnsplittableBlockError",
    "ChunkSizeError",
    # Assembly
    "AssemblyError",
    "NodeSpanIntegrityError",
    "BlockReferenceError",
    "CoverageError",
    # Relations
    "RelationError",
    "UnresolvedNodeReferenceError",
    # Enrichment
    "EnrichmentError",
    "EnricherFailedError",
    "EnricherTimeoutError",
    "EnricherContractViolationError",
    # Validation
    "KnowledgeValidationError",
    "ProvenanceViolationError",
    "NodeSchemaValidationError",
    # Composition
    "CompositionError",
    "InputTooLargeError",
    # Embedding
    "EmbeddingError",
    "EmbeddingProviderError",
    "EmbeddingTimeoutError",
    "DimensionMismatchError",
    "ModelIdentityError",
    # Sink
    "SinkError",
    "SinkWriteError",
    "SinkUnavailableError",
]


class KnowledgeStage(StrEnum):
    """
    Pipeline stage in which a knowledge processing failure originated.

    Used for structured logging, quarantine routing, and per-stage failure
    metrics. Declared here rather than in ``models`` because the exception
    module sits at the base of the module dependency graph.

    The two execution planes are distinguishable from the stage alone:
    planning through validation are structural and deterministic, while
    composition, embedding, and sink writes are model-versioned or external.
    A failure in the latter group does not invalidate the nodes already
    produced.
    """

    UNSPECIFIED = "UNSPECIFIED"
    CONFIGURATION = "CONFIGURATION"
    PLANNING = "PLANNING"
    ASSEMBLY = "ASSEMBLY"
    RELATIONS = "RELATIONS"
    ENRICHMENT = "ENRICHMENT"
    VALIDATION = "VALIDATION"
    COMPOSITION = "COMPOSITION"
    EMBEDDING = "EMBEDDING"
    SINK = "SINK"


# ============================================================================
# BASE
# ============================================================================


class PTFMKnowledgeError(PTFMError):
    """
    Base class for all Knowledge Processing Module failures.

    Attributes
    ----------
    stage:
        Pipeline stage that produced the failure. Declared as a class
        constant by each subtree.

    Parameters
    ----------
    message:
        Human-readable description of the failure.
    document_id:
        Canonical identifier of the source document being processed, when
        known.
    node_id:
        Canonical identifier of the knowledge node being produced, when the
        failure occurred after identity derivation.
    context:
        Additional structured detail.

    Raises
    ------
    None
    """

    stage: ClassVar[KnowledgeStage] = KnowledgeStage.UNSPECIFIED

    def __init__(
        self,
        message: str,
        *,
        document_id: str | None = None,
        node_id: str | None = None,
        context: Mapping[str, Any] | None = None,
    ) -> None:
        merged: dict[str, Any] = dict(context) if context else {}
        merged.setdefault("stage", str(self.stage))
        if document_id is not None:
            merged.setdefault("document_id", document_id)
        if node_id is not None:
            merged.setdefault("node_id", node_id)
        super().__init__(message, context=merged)
        self.document_id: str | None = document_id
        self.node_id: str | None = node_id

    def to_dict(self) -> dict[str, Any]:
        """
        Render the failure as a structured mapping including stage detail.

        Returns
        -------
        dict[str, Any]
            The base payload extended with ``stage``, ``document_id`` and
            ``node_id``.
        """
        payload = super().to_dict()
        payload["stage"] = str(self.stage)
        payload["document_id"] = self.document_id
        payload["node_id"] = self.node_id
        return payload


# ============================================================================
# CONFIGURATION
# ============================================================================


class KnowledgeConfigurationError(PTFMKnowledgeError):
    """
    Invalid or inconsistent knowledge processing configuration.

    Configuration failures are run-level and abort the entire run rather than
    failing a single document, since every subsequent document would fail
    identically.
    """

    stage: ClassVar[KnowledgeStage] = KnowledgeStage.CONFIGURATION


class ChunkerRegistrationError(KnowledgeConfigurationError):
    """
    A chunking strategy could not be registered.

    Raised on a duplicate strategy name, an unresolvable strategy reference,
    or an implementation that does not satisfy the ``ChunkingStrategy``
    contract.

    Parameters
    ----------
    message:
        Description of the registration failure.
    strategy_name:
        The strategy involved, when known.
    """

    def __init__(
        self,
        message: str,
        *,
        strategy_name: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(message, **kwargs)
        self.strategy_name: str | None = strategy_name
        if strategy_name is not None:
            self.add_context(strategy_name=strategy_name)


class EmbedderRegistrationError(KnowledgeConfigurationError):
    """
    An embedding provider could not be registered.

    Parameters
    ----------
    message:
        Description of the registration failure.
    provider_name:
        The provider involved, when known.
    """

    def __init__(
        self,
        message: str,
        *,
        provider_name: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(message, **kwargs)
        self.provider_name: str | None = provider_name
        if provider_name is not None:
            self.add_context(provider_name=provider_name)


class ProviderUnavailableError(KnowledgeConfigurationError):
    """
    A component is registered but its optional dependency is not installed.

    Raised at startup rather than mid-corpus, so that a missing packaging
    extra produces an actionable message instead of an ``ImportError`` after
    an expensive partial run.

    Parameters
    ----------
    component:
        Identifier of the unavailable component.
    missing_dependency:
        Distribution name of the absent package.
    install_extra:
        Name of the packaging extra that provides the dependency.
    """

    def __init__(
        self,
        *,
        component: str,
        missing_dependency: str,
        install_extra: str,
        **kwargs: Any,
    ) -> None:
        message = (
            f"Component {component!r} is unavailable: required dependency "
            f"{missing_dependency!r} is not installed. Install it with the "
            f"{install_extra!r} extra."
        )
        super().__init__(message, **kwargs)
        self.component: str = component
        self.missing_dependency: str = missing_dependency
        self.install_extra: str = install_extra
        self.add_context(
            component=component,
            missing_dependency=missing_dependency,
            install_extra=install_extra,
        )


class SegmentationFingerprintMismatchError(KnowledgeConfigurationError):
    """
    The configured segmentation differs from the one an existing corpus used.

    Node identity is derived in part from the segmentation fingerprint, so a
    change to any boundary-affecting configuration produces a disjoint
    identifier space. Nodes retrieved from an index built under the previous
    configuration would be unresolvable against a corpus processed under the
    new one, and every stored citation would break.

    Raised when the mismatch is detected, so that the condition surfaces as an
    explicit migration decision rather than as silently unresolvable
    references.

    Parameters
    ----------
    expected_fingerprint:
        Fingerprint recorded by the existing corpus.
    actual_fingerprint:
        Fingerprint of the current configuration.
    changed_components:
        Components observed to differ, when identifiable. Naming them turns
        an opaque digest mismatch into an actionable diagnosis.
    """

    def __init__(
        self,
        *,
        expected_fingerprint: str,
        actual_fingerprint: str,
        changed_components: Sequence[str] = (),
        **kwargs: Any,
    ) -> None:
        changed = ", ".join(changed_components) or "unknown"
        message = (
            f"Segmentation fingerprint mismatch: the corpus was processed "
            f"under {expected_fingerprint!r} but the current configuration "
            f"yields {actual_fingerprint!r}. Changed components: {changed}. "
            f"Node identifiers derived under differing segmentation are "
            f"disjoint, so the corpus must be reprocessed or the previous "
            f"configuration restored."
        )
        super().__init__(message, **kwargs)
        self.expected_fingerprint: str = expected_fingerprint
        self.actual_fingerprint: str = actual_fingerprint
        self.changed_components: tuple[str, ...] = tuple(changed_components)
        self.add_context(
            expected_fingerprint=expected_fingerprint,
            actual_fingerprint=actual_fingerprint,
            changed_components=list(changed_components),
        )


# ============================================================================
# PLANNING
# ============================================================================


class PlanningError(PTFMKnowledgeError):
    """Failure while determining where knowledge node boundaries fall."""

    stage: ClassVar[KnowledgeStage] = KnowledgeStage.PLANNING


class InvalidChunkPlanError(PlanningError):
    """
    A chunking strategy produced a plan that violates its contract.

    Raised on unordered or overlapping spans, spans outside the document
    text, or boundaries that do not align with block boundaries.

    Detected before any node is constructed, so that a defective strategy
    fails at the plan rather than producing a corpus whose provenance cannot
    be trusted.

    Parameters
    ----------
    message:
        Description of the violation.
    strategy_name:
        The strategy that produced the plan.
    chunk_index:
        Position of the offending chunk within the plan, when isolable.
    """

    def __init__(
        self,
        message: str,
        *,
        strategy_name: str | None = None,
        chunk_index: int | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(message, **kwargs)
        self.strategy_name: str | None = strategy_name
        self.chunk_index: int | None = chunk_index
        detail: dict[str, Any] = {}
        if strategy_name is not None:
            detail["strategy_name"] = strategy_name
        if chunk_index is not None:
            detail["chunk_index"] = chunk_index
        if detail:
            self.add_context(**detail)


class UnsplittableBlockError(PlanningError):
    """
    A block exceeds the size budget and could not be divided.

    Occurs where a single block is larger than the configured maximum and no
    interior boundary is available, as with an unbroken run of text carrying
    no sentence terminators.

    Distinguished from a general planning failure because the remedy is
    specific: raise the maximum, enable a finer segmenter, or accept the
    oversized node.

    Parameters
    ----------
    block_id:
        Identifier of the block that could not be divided.
    block_chars:
        Length of the block.
    max_chars:
        Configured maximum.
    """

    def __init__(
        self,
        *,
        block_id: str,
        block_chars: int,
        max_chars: int,
        **kwargs: Any,
    ) -> None:
        message = (
            f"Block {block_id!r} contains {block_chars} characters, exceeding "
            f"the configured maximum of {max_chars}, and offers no interior "
            f"boundary at which to divide it."
        )
        super().__init__(message, **kwargs)
        self.block_id: str = block_id
        self.block_chars: int = block_chars
        self.max_chars: int = max_chars
        self.add_context(
            block_id=block_id, block_chars=block_chars, max_chars=max_chars
        )


class ChunkSizeError(PlanningError):
    """
    A planned chunk violates the configured size bounds.

    Parameters
    ----------
    message:
        Description of the violation.
    chunk_chars:
        Length of the offending chunk.
    limit_chars:
        The bound that was violated.
    """

    def __init__(
        self,
        message: str,
        *,
        chunk_chars: int,
        limit_chars: int,
        **kwargs: Any,
    ) -> None:
        super().__init__(message, **kwargs)
        self.chunk_chars: int = chunk_chars
        self.limit_chars: int = limit_chars
        self.add_context(chunk_chars=chunk_chars, limit_chars=limit_chars)


# ============================================================================
# ASSEMBLY
# ============================================================================


class AssemblyError(PTFMKnowledgeError):
    """Failure while materializing planned spans into knowledge nodes."""

    stage: ClassVar[KnowledgeStage] = KnowledgeStage.ASSEMBLY


class NodeSpanIntegrityError(AssemblyError):
    """
    A node's text is not the exact substring of its source at its span.

    This is a correctness violation of the Evidence-First principle. If a
    node's text and span disagree, a citation locating evidence by span
    resolves to text the node does not contain, and the guarantee that a
    claim is traceable to an exact region of an identified source document is
    void.

    Named distinctly from the ingestion module's block-level span exception,
    so that a caller catching one cannot inadvertently catch the other. The
    two guard the same principle at different layers and warrant separate
    handling.

    Parameters
    ----------
    message:
        Description of the violation.
    expected_text:
        The substring the span identifies. Truncated in the recorded context,
        since a full node text may be large.
    actual_text:
        The text the node carries. Truncated likewise.
    span_start:
        Start offset of the node's span.
    span_end:
        End offset of the node's span.
    """

    #: Longest text excerpt retained in the diagnostic context. A node may be
    #: several thousand characters, and a failure record holding two full
    #: copies would be unusable at corpus scale.
    _EXCERPT_CHARS: ClassVar[int] = 120

    def __init__(
        self,
        message: str,
        *,
        expected_text: str | None = None,
        actual_text: str | None = None,
        span_start: int | None = None,
        span_end: int | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(message, **kwargs)
        self.span_start: int | None = span_start
        self.span_end: int | None = span_end
        detail: dict[str, Any] = {}
        if expected_text is not None:
            detail["expected_excerpt"] = expected_text[: self._EXCERPT_CHARS]
            detail["expected_length"] = len(expected_text)
        if actual_text is not None:
            detail["actual_excerpt"] = actual_text[: self._EXCERPT_CHARS]
            detail["actual_length"] = len(actual_text)
        if span_start is not None:
            detail["span_start"] = span_start
        if span_end is not None:
            detail["span_end"] = span_end
        if detail:
            self.add_context(**detail)


class BlockReferenceError(AssemblyError):
    """
    A node references blocks absent from, or misordered in, its source.

    Block references are what let a node recover the structure its flat text
    loses. A reference that does not resolve makes the node's structural
    context unrecoverable, and a misordered one misrepresents reading order.

    Parameters
    ----------
    message:
        Description of the violation.
    block_ids:
        The references involved.
    """

    def __init__(
        self,
        message: str,
        *,
        block_ids: Sequence[str] = (),
        **kwargs: Any,
    ) -> None:
        super().__init__(message, **kwargs)
        self.block_ids: tuple[str, ...] = tuple(block_ids)
        if block_ids:
            self.add_context(block_ids=list(block_ids))


class CoverageError(AssemblyError):
    """
    Document content exists that no knowledge node covers.

    The consequence is not merely untidy. Text absent from every node cannot
    be retrieved, so a claim it would have supported becomes unverifiable, and
    the failure is invisible: the pipeline reports success while the evidence
    is silently missing.

    Parameters
    ----------
    message:
        Description of the gap.
    uncovered_start:
        Start offset of the uncovered region.
    uncovered_end:
        End offset of the uncovered region.
    uncovered_chars:
        Total uncovered non-whitespace characters in the document.
    """

    def __init__(
        self,
        message: str,
        *,
        uncovered_start: int | None = None,
        uncovered_end: int | None = None,
        uncovered_chars: int | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(message, **kwargs)
        self.uncovered_start: int | None = uncovered_start
        self.uncovered_end: int | None = uncovered_end
        self.uncovered_chars: int | None = uncovered_chars
        detail: dict[str, Any] = {}
        if uncovered_start is not None:
            detail["uncovered_start"] = uncovered_start
        if uncovered_end is not None:
            detail["uncovered_end"] = uncovered_end
        if uncovered_chars is not None:
            detail["uncovered_chars"] = uncovered_chars
        if detail:
            self.add_context(**detail)


# ============================================================================
# RELATIONS
# ============================================================================


class RelationError(PTFMKnowledgeError):
    """Failure while deriving structural relations between nodes."""

    stage: ClassVar[KnowledgeStage] = KnowledgeStage.RELATIONS


class UnresolvedNodeReferenceError(RelationError):
    """
    A relation references a node that does not exist.

    Relations are the substrate on which symbolic verification will traverse
    document structure, so a dangling reference would surface as a reasoning
    failure far from its cause.

    Parameters
    ----------
    message:
        Description of the violation.
    source_node_id:
        Identifier of the relation's origin.
    target_node_id:
        Identifier of the relation's destination.
    """

    def __init__(
        self,
        message: str,
        *,
        source_node_id: str | None = None,
        target_node_id: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(message, **kwargs)
        self.source_node_id: str | None = source_node_id
        self.target_node_id: str | None = target_node_id
        detail: dict[str, Any] = {}
        if source_node_id is not None:
            detail["source_node_id"] = source_node_id
        if target_node_id is not None:
            detail["target_node_id"] = target_node_id
        if detail:
            self.add_context(**detail)


# ============================================================================
# ENRICHMENT
# ============================================================================


class EnrichmentError(PTFMKnowledgeError):
    """
    Failure while applying domain metadata enrichment.

    Enrichment is additive: it contributes metadata and cannot alter a node's
    text, span, or identity. A failure therefore degrades a node rather than
    losing it, and whether the run tolerates the degradation or treats it as
    fatal is a configuration decision.
    """

    stage: ClassVar[KnowledgeStage] = KnowledgeStage.ENRICHMENT


class EnricherFailedError(EnrichmentError):
    """
    An enricher raised while processing a node.

    Parameters
    ----------
    message:
        Description of the failure.
    enricher_name:
        Identifier of the enricher that raised.
    """

    def __init__(
        self,
        message: str,
        *,
        enricher_name: str,
        **kwargs: Any,
    ) -> None:
        super().__init__(message, **kwargs)
        self.enricher_name: str = enricher_name
        self.add_context(enricher_name=enricher_name)


class EnricherTimeoutError(EnrichmentError):
    """
    An enricher exceeded its deadline.

    Not retryable. Enrichment is optional metadata, so the proportionate
    response to a slow enricher is to proceed without its contribution rather
    than to spend the run's time budget waiting again. A genuinely transient
    upstream failure is the enricher's own concern to retry internally.

    Parameters
    ----------
    enricher_name:
        Identifier of the enricher that timed out.
    timeout_seconds:
        The deadline that was exceeded.
    """

    def __init__(
        self,
        *,
        enricher_name: str,
        timeout_seconds: float,
        **kwargs: Any,
    ) -> None:
        message = (
            f"Enricher {enricher_name!r} exceeded its {timeout_seconds} second "
            f"deadline."
        )
        super().__init__(message, **kwargs)
        self.enricher_name: str = enricher_name
        self.timeout_seconds: float = timeout_seconds
        self.add_context(
            enricher_name=enricher_name, timeout_seconds=timeout_seconds
        )


class EnricherContractViolationError(EnrichmentError):
    """
    An enricher returned something its contract does not permit.

    An enricher may contribute string-valued metadata and nothing else. This
    is the boundary that keeps domain logic out of provenance: an enricher
    that could return structured objects, or that could reach a node's text or
    span, would be able to alter what a citation resolves to.

    Raised when a return value is of the wrong type or carries a non-string
    value, so that a defective plugin is rejected rather than allowed to
    contaminate the knowledge base.

    Parameters
    ----------
    message:
        Description of the violation.
    enricher_name:
        Identifier of the offending enricher.
    offending_key:
        Metadata key whose value violated the contract, when isolable.
    """

    def __init__(
        self,
        message: str,
        *,
        enricher_name: str,
        offending_key: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(message, **kwargs)
        self.enricher_name: str = enricher_name
        self.offending_key: str | None = offending_key
        self.add_context(enricher_name=enricher_name)
        if offending_key is not None:
            self.add_context(offending_key=offending_key)


# ============================================================================
# VALIDATION
# ============================================================================


class KnowledgeValidationError(PTFMKnowledgeError):
    """
    Failure while verifying a produced node against its source document.

    Distinct from assembly errors: assembly detects a violation while
    building, whereas validation detects one that assembly should have made
    impossible. A failure here therefore indicates a defect in the pipeline
    rather than in the source document.
    """

    stage: ClassVar[KnowledgeStage] = KnowledgeStage.VALIDATION


class ProvenanceViolationError(KnowledgeValidationError):
    """
    A produced node does not faithfully represent its source document.

    The final gate before a node enters the knowledge base. Reaching it means
    an invariant the assembly stage owns was not upheld, so the node is
    rejected rather than emitted with provenance that cannot be trusted.

    Parameters
    ----------
    message:
        Description of the violation.
    violated_invariant:
        Name of the invariant that failed, so that a quarantine record
        identifies the guarantee that was broken rather than only that
        something was.
    """

    def __init__(
        self,
        message: str,
        *,
        violated_invariant: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(message, **kwargs)
        self.violated_invariant: str | None = violated_invariant
        if violated_invariant is not None:
            self.add_context(violated_invariant=violated_invariant)


class NodeSchemaValidationError(KnowledgeValidationError):
    """
    An assembled node failed canonical schema validation.

    Validation errors are passed as plain mappings rather than as a
    third-party validation object, so that no validation library type escapes
    into a caller's ``except`` clause.

    Parameters
    ----------
    message:
        Description of the validation failure.
    validation_errors:
        Normalized validation error records, typically obtained from
        ``pydantic.ValidationError.errors()``.
    """

    def __init__(
        self,
        message: str,
        *,
        validation_errors: Sequence[Mapping[str, Any]] = (),
        **kwargs: Any,
    ) -> None:
        super().__init__(message, **kwargs)
        self.validation_errors: tuple[Mapping[str, Any], ...] = tuple(
            validation_errors
        )
        if validation_errors:
            self.add_context(
                validation_errors=[dict(item) for item in validation_errors]
            )


# ============================================================================
# COMPOSITION
# ============================================================================


class CompositionError(PTFMKnowledgeError):
    """
    Failure while composing the text submitted to an embedding provider.

    Composition assembles embedding input from a node's fields. It never
    alters the node: the text stored on a node remains the exact substring of
    its source, while the text embedded may additionally carry structural
    context such as the heading path.
    """

    stage: ClassVar[KnowledgeStage] = KnowledgeStage.COMPOSITION


class InputTooLargeError(CompositionError):
    """
    Composed embedding input exceeds what the provider accepts.

    Parameters
    ----------
    input_chars:
        Length of the composed input.
    max_chars:
        Provider maximum.
    node_id:
        Identifier of the node whose input was oversized.
    """

    def __init__(
        self,
        *,
        input_chars: int,
        max_chars: int,
        **kwargs: Any,
    ) -> None:
        message = (
            f"Composed embedding input of {input_chars} characters exceeds the "
            f"provider maximum of {max_chars}."
        )
        super().__init__(message, **kwargs)
        self.input_chars: int = input_chars
        self.max_chars: int = max_chars
        self.add_context(input_chars=input_chars, max_chars=max_chars)


# ============================================================================
# EMBEDDING
# ============================================================================


class EmbeddingError(PTFMKnowledgeError):
    """
    Failure while generating vectors for knowledge nodes.

    Embeddings are a separate artifact from nodes, so a failure here does not
    invalidate the nodes already produced. A corpus may be embedded later, or
    re-embedded under a different model, without reprocessing.
    """

    stage: ClassVar[KnowledgeStage] = KnowledgeStage.EMBEDDING


class EmbeddingProviderError(EmbeddingError):
    """
    An embedding provider failed.

    Transient. A provider being briefly unreachable, rate limited, or
    restarting is a normal operational condition, and the batch is eligible
    for bounded retry.

    Parameters
    ----------
    message:
        Description of the failure.
    provider_name:
        Identifier of the provider that failed.
    batch_size:
        Number of inputs in the failed batch, when known.
    """

    retryable: ClassVar[bool] = True

    def __init__(
        self,
        message: str,
        *,
        provider_name: str,
        batch_size: int | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(message, **kwargs)
        self.provider_name: str = provider_name
        self.batch_size: int | None = batch_size
        self.add_context(provider_name=provider_name)
        if batch_size is not None:
            self.add_context(batch_size=batch_size)


class EmbeddingTimeoutError(EmbeddingError):
    """
    An embedding request exceeded its deadline.

    Transient. Unlike a deterministic parse, an embedding call crosses a
    process or network boundary whose latency varies with load, so a retry is
    a reasonable response.

    Parameters
    ----------
    message:
        Description of the failure.
    timeout_seconds:
        The deadline that was exceeded.
    batch_size:
        Number of inputs in the batch, when known.
    """

    retryable: ClassVar[bool] = True

    def __init__(
        self,
        message: str,
        *,
        timeout_seconds: float,
        batch_size: int | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(message, **kwargs)
        self.timeout_seconds: float = timeout_seconds
        self.batch_size: int | None = batch_size
        self.add_context(timeout_seconds=timeout_seconds)
        if batch_size is not None:
            self.add_context(batch_size=batch_size)


class DimensionMismatchError(EmbeddingError):
    """
    A provider returned a vector of unexpected dimension.

    Deterministic and therefore not retryable. A mismatch means the provider
    is not the model it reports itself to be, or that vectors from two models
    have been mixed. Either would corrupt every similarity computation
    downstream, silently, so the condition must fail loudly.

    Parameters
    ----------
    expected_dimension:
        Dimension the provider declares.
    actual_dimension:
        Dimension observed.
    provider_name:
        Identifier of the provider.
    """

    def __init__(
        self,
        *,
        expected_dimension: int,
        actual_dimension: int,
        provider_name: str,
        **kwargs: Any,
    ) -> None:
        message = (
            f"Provider {provider_name!r} declares dimension "
            f"{expected_dimension} but returned a vector of dimension "
            f"{actual_dimension}. Vectors of differing dimension cannot be "
            f"compared, so the corpus would be silently corrupted."
        )
        super().__init__(message, **kwargs)
        self.expected_dimension: int = expected_dimension
        self.actual_dimension: int = actual_dimension
        self.provider_name: str = provider_name
        self.add_context(
            expected_dimension=expected_dimension,
            actual_dimension=actual_dimension,
            provider_name=provider_name,
        )


class ModelIdentityError(EmbeddingError):
    """
    A provider could not report, or misreported, its model identity.

    Model identity is recorded on every embedding so that a vector can be
    attributed to the model that produced it. Without it, a corpus embedded
    across a model upgrade becomes an undifferentiated mixture whose
    similarity scores are meaningless.

    Parameters
    ----------
    message:
        Description of the failure.
    provider_name:
        Identifier of the provider.
    """

    def __init__(
        self,
        message: str,
        *,
        provider_name: str,
        **kwargs: Any,
    ) -> None:
        super().__init__(message, **kwargs)
        self.provider_name: str = provider_name
        self.add_context(provider_name=provider_name)


# ============================================================================
# SINK
# ============================================================================


class SinkError(PTFMKnowledgeError):
    """
    Failure while writing knowledge artifacts to a destination.

    The module writes to an injected sink and remains unaware of index
    implementations, so these exceptions describe the write operation rather
    than any storage technology.
    """

    stage: ClassVar[KnowledgeStage] = KnowledgeStage.SINK


class SinkWriteError(SinkError):
    """
    A sink rejected a write.

    Deterministic. A rejected artifact will be rejected again, so the fault is
    surfaced rather than retried.

    Parameters
    ----------
    message:
        Description of the failure.
    sink_name:
        Identifier of the sink that rejected the write.
    """

    def __init__(
        self,
        message: str,
        *,
        sink_name: str,
        **kwargs: Any,
    ) -> None:
        super().__init__(message, **kwargs)
        self.sink_name: str = sink_name
        self.add_context(sink_name=sink_name)


class SinkUnavailableError(SinkError):
    """
    A sink is temporarily unreachable.

    Transient. A storage backend restarting or briefly unreachable is a normal
    operational condition and is eligible for bounded retry.

    Parameters
    ----------
    message:
        Description of the failure.
    sink_name:
        Identifier of the unreachable sink.
    """

    retryable: ClassVar[bool] = True

    def __init__(
        self,
        message: str,
        *,
        sink_name: str,
        **kwargs: Any,
    ) -> None:
        super().__init__(message, **kwargs)
        self.sink_name: str = sink_name
        self.add_context(sink_name=sink_name)
