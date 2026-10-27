"""
Unit tests for the Knowledge Processing exception hierarchy.

File:
    tests/core/knowledge/test_exceptions.py
"""

from __future__ import annotations

import inspect
import pickle

import pytest

from ptfm.core.knowledge import exceptions as knowledge_exceptions
from ptfm.core.knowledge.exceptions import (
    AssemblyError,
    BlockReferenceError,
    ChunkSizeError,
    ChunkerRegistrationError,
    CompositionError,
    CoverageError,
    DimensionMismatchError,
    EmbedderRegistrationError,
    EmbeddingError,
    EmbeddingProviderError,
    EmbeddingTimeoutError,
    EnricherContractViolationError,
    EnricherFailedError,
    EnricherTimeoutError,
    EnrichmentError,
    InputTooLargeError,
    InvalidChunkPlanError,
    KnowledgeConfigurationError,
    KnowledgeStage,
    KnowledgeValidationError,
    ModelIdentityError,
    NodeSchemaValidationError,
    NodeSpanIntegrityError,
    PTFMKnowledgeError,
    PlanningError,
    ProvenanceViolationError,
    ProviderUnavailableError,
    RelationError,
    SegmentationFingerprintMismatchError,
    SinkError,
    SinkUnavailableError,
    SinkWriteError,
    UnresolvedNodeReferenceError,
    UnsplittableBlockError,
)
from ptfm.utils.exceptions import PTFMError

#: The only exception types eligible for retry. Every other failure in the
#: module is deterministic, so retrying would spend budget while delaying
#: quarantine.
RETRYABLE_TYPES: frozenset[type[PTFMKnowledgeError]] = frozenset(
    {EmbeddingProviderError, EmbeddingTimeoutError, SinkUnavailableError}
)


def _all_knowledge_error_types() -> list[type[PTFMKnowledgeError]]:
    """Collect every exported knowledge exception class."""
    return [
        obj
        for name in knowledge_exceptions.__all__
        if isinstance(obj := getattr(knowledge_exceptions, name), type)
        and issubclass(obj, PTFMKnowledgeError)
    ]


def _concrete_leaf_types() -> list[type[PTFMKnowledgeError]]:
    """Collect exception classes with no subclasses in the hierarchy."""
    every = _all_knowledge_error_types()
    return [
        candidate
        for candidate in every
        if not any(
            other is not candidate and issubclass(other, candidate)
            for other in every
        )
    ]


# ---------------------------------------------------------------------------
# Hierarchy invariants
# ---------------------------------------------------------------------------


def test_every_error_derives_from_the_project_root() -> None:
    for error_type in _all_knowledge_error_types():
        assert issubclass(error_type, PTFMError)


def test_every_leaf_declares_a_concrete_stage() -> None:
    for error_type in _concrete_leaf_types():
        assert error_type.stage is not KnowledgeStage.UNSPECIFIED, (
            error_type.__name__
        )


def test_only_transient_conditions_are_retryable() -> None:
    for error_type in _all_knowledge_error_types():
        expected = any(issubclass(error_type, base) for base in RETRYABLE_TYPES)
        assert error_type.retryable is expected, error_type.__name__


def test_all_exports_resolve() -> None:
    for name in knowledge_exceptions.__all__:
        assert hasattr(knowledge_exceptions, name), name


def test_error_codes_are_unique() -> None:
    codes = [t.error_code for t in _all_knowledge_error_types()]

    assert len(codes) == len(set(codes))


def test_every_pipeline_stage_has_at_least_one_exception() -> None:
    """A stage with no exception type could not attribute a failure."""
    covered = {t.stage for t in _all_knowledge_error_types()}
    expected = set(KnowledgeStage) - {KnowledgeStage.UNSPECIFIED}

    assert expected <= covered, sorted(s.value for s in expected - covered)


# ---------------------------------------------------------------------------
# Base behaviour
# ---------------------------------------------------------------------------


def test_stage_and_identity_are_serialized() -> None:
    error = NodeSpanIntegrityError(
        "text does not match span",
        document_id="doc-1",
        node_id="node-1",
    )

    payload = error.to_dict()

    assert payload["stage"] == KnowledgeStage.ASSEMBLY.value
    assert payload["document_id"] == "doc-1"
    assert payload["node_id"] == "node-1"
    assert payload["error_code"] == "NODE_SPAN_INTEGRITY"
    assert payload["retryable"] is False


def test_identity_is_mirrored_into_context_for_uniform_logging() -> None:
    error = CoverageError("gap", document_id="doc-2", node_id="node-2")

    assert error.context["document_id"] == "doc-2"
    assert error.context["node_id"] == "node-2"
    assert error.context["stage"] == KnowledgeStage.ASSEMBLY.value


def test_absent_identity_is_omitted_rather_than_recorded_as_none() -> None:
    error = PlanningError("no document context")

    assert "document_id" not in error.context
    assert error.to_dict()["document_id"] is None


# ---------------------------------------------------------------------------
# Evidence-First enforcement
# ---------------------------------------------------------------------------


def test_span_integrity_records_both_texts_for_diagnosis() -> None:
    error = NodeSpanIntegrityError(
        "mismatch",
        expected_text="the quick brown fox",
        actual_text="the quick brown dog",
        span_start=10,
        span_end=29,
    )

    assert error.context["expected_excerpt"] == "the quick brown fox"
    assert error.context["actual_excerpt"] == "the quick brown dog"
    assert error.span_start == 10
    assert error.span_end == 29


def test_span_integrity_excerpts_are_truncated() -> None:
    """A failure record holding two full node texts is unusable at scale."""
    error = NodeSpanIntegrityError(
        "mismatch", expected_text="x" * 5000, actual_text="y" * 5000
    )

    assert len(error.context["expected_excerpt"]) == 120
    assert error.context["expected_length"] == 5000


def test_span_integrity_is_distinct_from_the_ingestion_exception() -> None:
    """
    Two identically named exceptions across modules would let a caller catch
    the wrong one. The layers guard the same principle and warrant separate
    handling.
    """
    from ptfm.core.ingestion.exceptions import (
        SpanIntegrityError as IngestionSpanIntegrityError,
    )

    assert NodeSpanIntegrityError is not IngestionSpanIntegrityError
    assert not issubclass(NodeSpanIntegrityError, IngestionSpanIntegrityError)
    assert NodeSpanIntegrityError.__name__ != IngestionSpanIntegrityError.__name__


def test_coverage_error_locates_the_uncovered_region() -> None:
    error = CoverageError(
        "content outside every node",
        uncovered_start=400,
        uncovered_end=520,
        uncovered_chars=118,
    )

    assert error.context["uncovered_start"] == 400
    assert error.context["uncovered_chars"] == 118


def test_block_reference_error_names_the_references() -> None:
    error = BlockReferenceError("unresolvable", block_ids=["0.1", "0.2"])

    assert error.block_ids == ("0.1", "0.2")
    assert error.context["block_ids"] == ["0.1", "0.2"]


def test_provenance_violation_names_the_broken_invariant() -> None:
    error = ProvenanceViolationError(
        "node text is not a substring of its source",
        violated_invariant="node.text == raw.text[span]",
    )

    assert error.violated_invariant == "node.text == raw.text[span]"
    assert error.stage is KnowledgeStage.VALIDATION


# ---------------------------------------------------------------------------
# Segmentation identity
# ---------------------------------------------------------------------------


def test_fingerprint_mismatch_names_the_changed_components() -> None:
    """An opaque digest mismatch is not actionable; a named component is."""
    error = SegmentationFingerprintMismatchError(
        expected_fingerprint="abc123",
        actual_fingerprint="def456",
        changed_components=["max_chunk_chars", "sentence_segmenter_version"],
    )

    assert error.expected_fingerprint == "abc123"
    assert error.changed_components == (
        "max_chunk_chars",
        "sentence_segmenter_version",
    )
    assert "max_chunk_chars" in error.message


def test_fingerprint_mismatch_explains_the_consequence() -> None:
    error = SegmentationFingerprintMismatchError(
        expected_fingerprint="a", actual_fingerprint="b"
    )

    assert "disjoint" in error.message
    assert error.stage is KnowledgeStage.CONFIGURATION
    assert error.retryable is False


def test_fingerprint_mismatch_tolerates_unidentified_components() -> None:
    error = SegmentationFingerprintMismatchError(
        expected_fingerprint="a", actual_fingerprint="b"
    )

    assert "unknown" in error.message


# ---------------------------------------------------------------------------
# Planning
# ---------------------------------------------------------------------------


def test_invalid_plan_records_the_offending_chunk() -> None:
    error = InvalidChunkPlanError(
        "chunk spans overlap",
        strategy_name="BlockAlignedChunker",
        chunk_index=7,
    )

    assert error.context["strategy_name"] == "BlockAlignedChunker"
    assert error.context["chunk_index"] == 7
    assert error.stage is KnowledgeStage.PLANNING


def test_unsplittable_block_states_the_remedy_inputs() -> None:
    error = UnsplittableBlockError(
        block_id="0.3", block_chars=9000, max_chars=2000
    )

    assert error.block_id == "0.3"
    assert "9000" in error.message and "2000" in error.message


def test_chunk_size_error_records_both_bounds() -> None:
    error = ChunkSizeError("chunk too small", chunk_chars=5, limit_chars=100)

    assert error.chunk_chars == 5
    assert error.limit_chars == 100


def test_an_empty_document_is_not_modelled_as_an_exception() -> None:
    """
    A document yielding no nodes is a normal corpus occurrence, reported as a
    skipped outcome. Exceptions are for faults, not control flow.
    """
    assert not hasattr(knowledge_exceptions, "EmptyDocumentError")


# ---------------------------------------------------------------------------
# Enrichment
# ---------------------------------------------------------------------------


def test_enricher_failure_names_the_enricher() -> None:
    error = EnricherFailedError("lookup failed", enricher_name="LegalEnricher")

    assert error.enricher_name == "LegalEnricher"
    assert error.stage is KnowledgeStage.ENRICHMENT


def test_enricher_timeout_is_not_retryable() -> None:
    """Enrichment is optional metadata; proceeding without it is proportionate."""
    error = EnricherTimeoutError(
        enricher_name="SlowEnricher", timeout_seconds=5.0
    )

    assert error.retryable is False
    assert error.timeout_seconds == 5.0


def test_contract_violation_identifies_the_offending_key() -> None:
    error = EnricherContractViolationError(
        "value is not a string",
        enricher_name="BadEnricher",
        offending_key="citation",
    )

    assert error.offending_key == "citation"
    assert error.context["enricher_name"] == "BadEnricher"


def test_a_contract_violation_type_exists_for_plugin_isolation() -> None:
    """
    The boundary keeping domain logic out of provenance needs a typed failure,
    or a defective plugin degrades silently.
    """
    assert issubclass(EnricherContractViolationError, EnrichmentError)


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------


def test_provider_failures_are_retryable() -> None:
    error = EmbeddingProviderError(
        "connection reset", provider_name="LocalProvider", batch_size=32
    )

    assert error.retryable is True
    assert error.context["batch_size"] == 32


def test_embedding_timeouts_are_retryable() -> None:
    """Unlike a deterministic parse, an embedding call crosses a boundary
    whose latency varies with load."""
    error = EmbeddingTimeoutError(
        "deadline exceeded", timeout_seconds=30.0, batch_size=16
    )

    assert error.retryable is True


def test_dimension_mismatch_is_permanent_and_explains_the_risk() -> None:
    error = DimensionMismatchError(
        expected_dimension=1024,
        actual_dimension=768,
        provider_name="LocalProvider",
    )

    assert error.retryable is False
    assert "silently corrupted" in error.message
    assert error.context["expected_dimension"] == 1024


def test_model_identity_failure_is_typed() -> None:
    error = ModelIdentityError(
        "provider reported no model", provider_name="RemoteProvider"
    )

    assert error.stage is KnowledgeStage.EMBEDDING
    assert error.retryable is False


def test_embedding_failures_do_not_invalidate_nodes() -> None:
    """
    Embeddings are a separate artifact, so an embedding stage exists that is
    distinct from assembly. A corpus may be embedded later without
    reprocessing.
    """
    assert EmbeddingError.stage is KnowledgeStage.EMBEDDING
    assert AssemblyError.stage is KnowledgeStage.ASSEMBLY


# ---------------------------------------------------------------------------
# Composition
# ---------------------------------------------------------------------------


def test_oversized_input_records_both_sizes() -> None:
    error = InputTooLargeError(input_chars=12000, max_chars=8192)

    assert error.input_chars == 12000
    assert error.stage is KnowledgeStage.COMPOSITION


def test_composition_is_a_distinct_stage_from_embedding() -> None:
    """Composition assembles input from node fields; it never alters a node."""
    assert CompositionError.stage is not EmbeddingError.stage


# ---------------------------------------------------------------------------
# Sink
# ---------------------------------------------------------------------------


def test_a_rejected_write_is_permanent() -> None:
    error = SinkWriteError("schema rejected", sink_name="VectorSink")

    assert error.retryable is False
    assert error.sink_name == "VectorSink"


def test_an_unreachable_sink_is_transient() -> None:
    error = SinkUnavailableError("connection refused", sink_name="GraphSink")

    assert error.retryable is True
    assert error.stage is KnowledgeStage.SINK


def test_sink_exceptions_name_no_storage_technology() -> None:
    """The module must remain unaware of index implementations."""
    source = inspect.getsource(SinkError) + inspect.getsource(SinkWriteError)
    lowered = source.lower()

    for term in ("faiss", "qdrant", "milvus", "chroma", "elasticsearch", "neo4j"):
        assert term not in lowered


# ---------------------------------------------------------------------------
# Registration and availability
# ---------------------------------------------------------------------------


def test_chunker_registration_names_the_strategy() -> None:
    error = ChunkerRegistrationError(
        "duplicate strategy", strategy_name="BlockAlignedChunker"
    )

    assert error.strategy_name == "BlockAlignedChunker"
    assert error.stage is KnowledgeStage.CONFIGURATION


def test_embedder_registration_names_the_provider() -> None:
    error = EmbedderRegistrationError("invalid", provider_name="LocalProvider")

    assert error.provider_name == "LocalProvider"


def test_unavailable_provider_names_the_install_extra() -> None:
    error = ProviderUnavailableError(
        component="SentenceTransformersProvider",
        missing_dependency="sentence-transformers",
        install_extra="ptfm[embeddings]",
    )

    assert error.install_extra == "ptfm[embeddings]"
    assert "sentence-transformers" in error.message


def test_configuration_failures_are_never_retryable() -> None:
    for error_type in _all_knowledge_error_types():
        if issubclass(error_type, KnowledgeConfigurationError):
            assert error_type.retryable is False, error_type.__name__


# ---------------------------------------------------------------------------
# Boundary wrapping and process transport
# ---------------------------------------------------------------------------


def test_third_party_exception_is_preserved_as_cause() -> None:
    class _VendorModelError(Exception):
        """Stand-in for an embedding library exception."""

    try:
        try:
            raise _VendorModelError("CUDA out of memory")
        except _VendorModelError as exc:
            raise EmbeddingProviderError(
                "embedding failed",
                provider_name="LocalProvider",
                document_id="doc-1",
            ) from exc
    except EmbeddingProviderError as error:
        assert isinstance(error.__cause__, _VendorModelError)
        assert error.to_dict()["cause_type"] == "_VendorModelError"


def test_schema_validation_errors_are_plain_mappings() -> None:
    """No validation library type may escape into a caller's except clause."""
    error = NodeSchemaValidationError(
        "node failed validation",
        validation_errors=[{"loc": ("span",), "msg": "field required"}],
    )

    for entry in error.validation_errors:
        assert isinstance(entry, dict)


@pytest.mark.parametrize(
    "error",
    [
        UnsplittableBlockError(block_id="0.1", block_chars=100, max_chars=50),
        DimensionMismatchError(
            expected_dimension=1024, actual_dimension=768, provider_name="P"
        ),
        ProviderUnavailableError(
            component="C", missing_dependency="d", install_extra="e"
        ),
        SegmentationFingerprintMismatchError(
            expected_fingerprint="a", actual_fingerprint="b"
        ),
        EnricherTimeoutError(enricher_name="E", timeout_seconds=1.0),
        InputTooLargeError(input_chars=10, max_chars=5),
    ],
)
def test_keyword_only_exceptions_survive_process_boundaries(
    error: PTFMKnowledgeError,
) -> None:
    """Embedding may be offloaded to worker processes."""
    restored = pickle.loads(pickle.dumps(error))

    assert type(restored) is type(error)
    assert restored.message == error.message
    assert restored.stage is error.stage
    assert restored.retryable is error.retryable
    assert dict(restored.context) == dict(error.context)


# ---------------------------------------------------------------------------
# Module hygiene
# ---------------------------------------------------------------------------


def test_the_module_imports_only_the_project_root_exception() -> None:
    source = inspect.getsource(knowledge_exceptions)

    assert "from ptfm.utils.exceptions import PTFMError" in source
    for forbidden in (
        "from ptfm.core.schemas",
        "from ptfm.core.knowledge.models",
        "from ptfm.core.knowledge.config",
        "from ptfm.plugins",
        "import pydantic",
        "import numpy",
    ):
        assert forbidden not in source


def test_the_module_declares_no_mutable_state() -> None:
    mutable = [
        name
        for name, value in vars(knowledge_exceptions).items()
        if not name.startswith("__") and isinstance(value, (list, dict, set))
    ]

    assert mutable == []


def test_the_module_contains_no_domain_vocabulary() -> None:
    source = inspect.getsource(knowledge_exceptions).lower()

    for term in ("jurisdiction", "statute", "ticker", "icd", "patient", "court"):
        assert term not in source


def test_the_module_declares_no_thresholds() -> None:
    """
    Operational limits are injected from configuration. The only numeric
    constant here bounds a diagnostic excerpt and gates nothing.
    """
    source = inspect.getsource(knowledge_exceptions)

    assert "> 0.8" not in source
    assert "confidence" not in source.lower()
    assert "similarity" not in source.lower()
