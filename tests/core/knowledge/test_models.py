"""
Unit tests for the Knowledge Processing internal models.

File:
    tests/core/knowledge/test_models.py
"""

from __future__ import annotations

import inspect
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from pydantic import TypeAdapter, ValidationError

from ptfm.core.knowledge import models as models_module
from ptfm.core.knowledge.exceptions import (
    CoverageError,
    EmbeddingProviderError,
    KnowledgeStage,
    NodeSpanIntegrityError,
)
from ptfm.core.knowledge.models import (
    ChunkPlan,
    DocumentFailed,
    DocumentSkipped,
    KnowledgeUnit,
    NodeProduced,
    PlannedChunk,
    ProcessingFailure,
    ProcessingOutcome,
    ProcessingReport,
    SkipReason,
)
from ptfm.core.schemas import (
    KnowledgeNode,
    NodeEmbedding,
    NodeRelation,
    NodeType,
    RelationType,
    SourceDescriptor,
    TextSpan,
    derive_document_id,
    derive_node_id,
)

NOW = datetime(2026, 8, 1, tzinfo=UTC)
DOCUMENT_TEXT = "Interest Rates\n\nThe applicable rate is 4.5 percent."
DOCUMENT_ID = derive_document_id("file:///corpus/contract.pdf", "hash-abc")
FINGERPRINT = "seg-fp-001"


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def _chunk(start: int, end: int, *blocks: str) -> PlannedChunk:
    return PlannedChunk(
        span=TextSpan(start=start, end=end), block_ids=blocks or ("0",)
    )


def _plan(*chunks: PlannedChunk, **overrides: Any) -> ChunkPlan:
    fields: dict[str, Any] = {
        "document_id": DOCUMENT_ID,
        "strategy_name": "BlockAlignedChunker",
        "chunks": chunks,
    }
    fields.update(overrides)
    return ChunkPlan(**fields)


def _source() -> SourceDescriptor:
    return SourceDescriptor(
        source_uri="file:///corpus/contract.pdf",
        scheme="file",
        connector_name="LocalFilesystemConnector",
        extractor_name="PdfExtractor",
        acquired_at=NOW,
        size_bytes=4096,
        detected_media_type="application/pdf",
    )


def _node(start: int = 16, end: int = 50, **overrides: Any) -> KnowledgeNode:
    span = TextSpan(start=start, end=end)
    fields: dict[str, Any] = {
        "node_id": derive_node_id(DOCUMENT_ID, span, FINGERPRINT),
        "document_id": DOCUMENT_ID,
        "span": span,
        "text": DOCUMENT_TEXT[start:end],
        "node_type": NodeType.PROSE,
        "block_ids": ("1",),
        "ordinal": 1,
        "source": _source(),
        "segmentation_fingerprint": FINGERPRINT,
        "source_text_hash": "text-hash-abc",
        "source_schema_version": "1.0",
        "processed_at": NOW,
    }
    fields.update(overrides)
    return KnowledgeNode(**fields)


def _embedding(node: KnowledgeNode, **overrides: Any) -> NodeEmbedding:
    fields: dict[str, Any] = {
        "node_id": node.node_id,
        "document_id": node.document_id,
        "vector": (0.6, 0.8),
        "dimension": 2,
        "model_id": "test/model",
        "normalized": True,
        "composition_id": "heading-path-prefixed",
        "input_text_hash": "input-hash-1",
        "embedded_at": NOW,
    }
    fields.update(overrides)
    return NodeEmbedding(**fields)


def _relation(node: KnowledgeNode, **overrides: Any) -> NodeRelation:
    fields: dict[str, Any] = {
        "source_node_id": node.node_id,
        "target_node_id": "other-node",
        "relation_type": RelationType.PRECEDES,
        "document_id": node.document_id,
    }
    fields.update(overrides)
    return NodeRelation(**fields)


def _report(**overrides: Any) -> ProcessingReport:
    fields: dict[str, Any] = {
        "run_id": "run-1",
        "segmentation_fingerprint": FINGERPRINT,
        "started_at": NOW,
        "completed_at": NOW + timedelta(seconds=120),
    }
    fields.update(overrides)
    return ProcessingReport(**fields)


# ---------------------------------------------------------------------------
# A plan names spans, not nodes
# ---------------------------------------------------------------------------


def test_a_planned_chunk_carries_only_span_and_blocks() -> None:
    """
    A strategy that declared node type or heading path could mislabel a chunk,
    and nothing downstream would detect it. Assembly derives both from the
    document instead.
    """
    fields = set(PlannedChunk.model_fields)

    assert fields == {"span", "block_ids"}
    for forbidden in ("text", "node_type", "heading_path", "node_id"):
        assert forbidden not in fields


def test_a_chunk_must_name_the_blocks_it_covers() -> None:
    """Block references are the cross-check assembly validates the span
    against."""
    with pytest.raises(ValidationError, match="at least one block"):
        PlannedChunk(span=TextSpan(start=0, end=5), block_ids=())


def test_duplicate_block_references_are_rejected() -> None:
    with pytest.raises(ValidationError, match="duplicates"):
        PlannedChunk(span=TextSpan(start=0, end=5), block_ids=("0", "0"))


# ---------------------------------------------------------------------------
# Plan validation
# ---------------------------------------------------------------------------


def test_a_well_formed_plan_is_accepted() -> None:
    plan = _plan(_chunk(0, 14, "0"), _chunk(16, 50, "1"))

    assert plan.chunk_count == 2
    assert plan.covered_chars == 48
    assert plan.block_ids == ("0", "1")


def test_overlapping_chunks_are_rejected() -> None:
    """
    Overlap duplicates text across nodes, inflating the index and distorting
    the term frequency statistics retrieval depends on.
    """
    with pytest.raises(ValidationError, match="ascending and disjoint"):
        _plan(_chunk(0, 20, "0"), _chunk(10, 30, "1"))


def test_unordered_chunks_are_rejected() -> None:
    with pytest.raises(ValidationError, match="ascending and disjoint"):
        _plan(_chunk(20, 30, "1"), _chunk(0, 10, "0"))


def test_adjacent_chunks_are_permitted() -> None:
    """Zero gap is not overlap."""
    plan = _plan(_chunk(0, 10, "0"), _chunk(10, 20, "1"))

    assert plan.chunk_count == 2


def test_a_block_claimed_twice_is_rejected() -> None:
    with pytest.raises(ValidationError, match="claimed by both"):
        _plan(_chunk(0, 10, "0"), _chunk(10, 20, "0"))


def test_an_empty_plan_is_valid() -> None:
    """A document with no extractable content is a skip, not a failure."""
    plan = _plan()

    assert plan.is_empty is True
    assert plan.chunk_count == 0
    assert plan.covered_chars == 0


def test_a_plan_names_the_strategy_responsible() -> None:
    plan = _plan(_chunk(0, 5), strategy_name="TableAwareChunker")

    assert plan.strategy_name == "TableAwareChunker"


def test_a_plan_is_frozen() -> None:
    plan = _plan(_chunk(0, 5))

    with pytest.raises(ValidationError):
        plan.chunks = ()  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Unit ownership
# ---------------------------------------------------------------------------


def test_a_unit_carries_node_embedding_and_relations() -> None:
    node = _node()

    unit = KnowledgeUnit(
        node=node,
        embedding=_embedding(node),
        relations=(_relation(node),),
    )

    assert unit.node_id == node.node_id
    assert unit.document_id == node.document_id
    assert unit.is_embedded is True


def test_a_unit_without_an_embedding_is_valid() -> None:
    """Embeddings are a separate artifact; a corpus may be embedded later."""
    unit = KnowledgeUnit(node=_node())

    assert unit.embedding is None
    assert unit.is_embedded is False


def test_an_embedding_belonging_to_another_node_is_rejected() -> None:
    """
    A vector attached to the wrong node would return the wrong text for a
    query, with no downstream check able to detect it.
    """
    node = _node()
    foreign = _embedding(node, node_id="different-node")

    with pytest.raises(ValidationError, match="wrong text for a query"):
        KnowledgeUnit(node=node, embedding=foreign)


def test_an_embedding_naming_another_document_is_rejected() -> None:
    node = _node()
    foreign = _embedding(node, document_id="different-document")

    with pytest.raises(ValidationError, match="names document"):
        KnowledgeUnit(node=node, embedding=foreign)


def test_a_relation_not_originating_from_the_node_is_rejected() -> None:
    """A unit carries only its own outgoing relations, so each relation is
    written exactly once."""
    node = _node()
    incoming = _relation(node, source_node_id="other-node", target_node_id=node.node_id)

    with pytest.raises(ValidationError, match="exactly once"):
        KnowledgeUnit(node=node, relations=(incoming,))


def test_a_unit_is_frozen() -> None:
    unit = KnowledgeUnit(node=_node())

    with pytest.raises(ValidationError):
        unit.embedding = None  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Outcomes
# ---------------------------------------------------------------------------


def test_production_is_node_valued() -> None:
    """
    A node is the unit a sink consumes. Materializing every node of a document
    before yielding any would defeat the memory bound streaming provides.
    """
    outcome = NodeProduced(unit=KnowledgeUnit(node=_node()), duration_ms=3.2)

    assert outcome.kind == "NODE_PRODUCED"
    assert outcome.node_id
    assert outcome.document_id == DOCUMENT_ID


def test_a_skip_is_document_valued() -> None:
    """A document yielding nothing has no node to attach an outcome to."""
    outcome = DocumentSkipped(
        document_id=DOCUMENT_ID, reason=SkipReason.NO_EXTRACTABLE_CONTENT
    )

    assert outcome.kind == "DOCUMENT_SKIPPED"
    assert not hasattr(outcome, "node_id")


def test_an_empty_document_is_a_skip_not_a_failure() -> None:
    """Consistent with the removal of EmptyDocumentError from the hierarchy."""
    assert SkipReason.NO_EXTRACTABLE_CONTENT in set(SkipReason)
    assert not hasattr(models_module, "EmptyDocumentError")


def test_reprocessing_an_unchanged_document_is_a_skip() -> None:
    """Deterministic identity makes reprocessing idempotent, so skipping is
    an optimization rather than a loss."""
    outcome = DocumentSkipped(
        document_id=DOCUMENT_ID, reason=SkipReason.ALREADY_PROCESSED
    )

    assert outcome.reason is SkipReason.ALREADY_PROCESSED


def test_a_failure_captures_stage_and_retryability() -> None:
    try:
        raise NodeSpanIntegrityError(
            "text does not match span", document_id=DOCUMENT_ID, node_id="n-1"
        )
    except NodeSpanIntegrityError as error:
        failure = ProcessingFailure.from_exception(
            error, document_id=DOCUMENT_ID, failed_at=NOW
        )

    assert failure.stage is KnowledgeStage.ASSEMBLY
    assert failure.error_code == "NODE_SPAN_INTEGRITY"
    assert failure.node_id == "n-1"
    assert failure.retryable is False
    assert failure.attempts == 1


def test_a_retryable_failure_records_its_attempt_count() -> None:
    try:
        raise EmbeddingProviderError("unreachable", provider_name="P")
    except EmbeddingProviderError as error:
        failure = ProcessingFailure.from_exception(
            error, document_id=DOCUMENT_ID, failed_at=NOW, attempts=3
        )

    assert failure.retryable is True
    assert failure.attempts == 3
    assert failure.stage is KnowledgeStage.EMBEDDING


def test_identical_failures_share_a_traceback_digest() -> None:
    """Grouping is what makes triage across a large corpus tractable."""

    def _raise() -> None:
        raise CoverageError("gap")

    digests = []
    for _ in range(2):
        try:
            _raise()
        except CoverageError as error:
            digests.append(
                ProcessingFailure.from_exception(
                    error, document_id=DOCUMENT_ID, failed_at=NOW
                ).traceback_digest
            )

    assert digests[0] is not None
    assert digests[0] == digests[1]


def test_a_failure_prefers_the_identity_carried_by_the_exception() -> None:
    failure = ProcessingFailure.from_exception(
        CoverageError("gap", document_id="actual-doc"),
        document_id="fallback-doc",
        failed_at=NOW,
    )

    assert failure.document_id == "actual-doc"


def test_a_document_failure_exposes_its_document() -> None:
    failure = ProcessingFailure.from_exception(
        CoverageError("gap"), document_id=DOCUMENT_ID, failed_at=NOW
    )

    assert DocumentFailed(failure=failure).document_id == DOCUMENT_ID


# ---------------------------------------------------------------------------
# Outcome union
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("outcome", "kind"),
    [
        (
            NodeProduced(unit=KnowledgeUnit(node=_node()), duration_ms=1.0),
            "NODE_PRODUCED",
        ),
        (
            DocumentSkipped(
                document_id=DOCUMENT_ID, reason=SkipReason.EXCLUDED_BY_FILTER
            ),
            "DOCUMENT_SKIPPED",
        ),
        (
            DocumentFailed(
                failure=ProcessingFailure.from_exception(
                    CoverageError("gap"),
                    document_id=DOCUMENT_ID,
                    failed_at=NOW,
                )
            ),
            "DOCUMENT_FAILED",
        ),
    ],
)
def test_the_union_round_trips_by_discriminator(
    outcome: Any, kind: str
) -> None:
    adapter: TypeAdapter[ProcessingOutcome] = TypeAdapter(ProcessingOutcome)

    restored = adapter.validate_json(adapter.dump_json(outcome))

    assert restored.kind == kind
    assert type(restored) is type(outcome)
    assert restored == outcome


def test_an_unknown_discriminator_is_rejected() -> None:
    adapter: TypeAdapter[ProcessingOutcome] = TypeAdapter(ProcessingOutcome)

    with pytest.raises(ValidationError):
        adapter.validate_python({"kind": "UNKNOWN"})


def test_a_negative_duration_is_rejected() -> None:
    with pytest.raises(ValidationError):
        NodeProduced(unit=KnowledgeUnit(node=_node()), duration_ms=-1.0)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def test_a_report_records_the_segmentation_that_produced_the_corpus() -> None:
    """
    Node identity depends on it, so a report that did not name its own
    segmentation could not be compared against one that did.
    """
    report = _report()

    assert report.segmentation_fingerprint == FINGERPRINT


def test_a_report_aggregates_the_run() -> None:
    report = _report(
        documents_processed=80,
        documents_skipped=15,
        documents_failed=5,
        nodes_produced=1200,
        embeddings_produced=1200,
        relations_produced=2400,
        total_node_chars=960_000,
        nodes_by_type={NodeType.PROSE.value: 1100, NodeType.TABLE_REGION.value: 100},
        failures_by_stage={KnowledgeStage.ASSEMBLY.value: 5},
    )

    assert report.documents_seen == 100
    assert report.failure_ratio == pytest.approx(0.05)
    assert report.embedding_coverage == pytest.approx(1.0)
    assert report.mean_node_chars == pytest.approx(800.0)
    assert report.duration_seconds == pytest.approx(120.0)


def test_partial_embedding_is_visible_in_the_report() -> None:
    """An unembedded node is invisible to dense search, so retrieval must
    know the corpus is only partly embedded."""
    report = _report(nodes_produced=1000, embeddings_produced=400)

    assert report.embedding_coverage == pytest.approx(0.4)


def test_a_report_without_embedding_records_no_model() -> None:
    report = _report(nodes_produced=100)

    assert report.embedding_model_identity is None
    assert report.embedding_coverage == 0.0


def test_a_report_records_the_embedding_model_when_one_ran() -> None:
    report = _report(
        nodes_produced=10,
        embeddings_produced=10,
        embedding_model_identity="BAAI/bge-large-en-v1.5@a1b2c3",
    )

    assert report.embedding_model_identity == "BAAI/bge-large-en-v1.5@a1b2c3"


def test_an_empty_run_reports_zero_rather_than_dividing() -> None:
    report = _report()

    assert report.failure_ratio == 0.0
    assert report.embedding_coverage == 0.0
    assert report.mean_node_chars == 0.0


def test_an_inverted_interval_is_rejected() -> None:
    with pytest.raises(ValidationError, match="must not precede"):
        _report(completed_at=NOW - timedelta(seconds=1))


def test_node_type_counts_are_recoverable_as_enum_members() -> None:
    report = _report(
        nodes_by_type={NodeType.PROSE.value: 5, NodeType.CODE.value: 2}
    )

    recovered = dict(report.iter_node_types())

    assert recovered == {NodeType.PROSE: 5, NodeType.CODE: 2}


def test_unrecognized_node_type_keys_are_skipped() -> None:
    """A future schema version could introduce a type this build lacks."""
    report = _report(nodes_by_type={NodeType.PROSE.value: 5, "FUTURE_TYPE": 3})

    assert dict(report.iter_node_types()) == {NodeType.PROSE: 5}


def test_a_report_round_trips_through_json() -> None:
    report = _report(
        documents_processed=3,
        nodes_produced=12,
        skips_by_reason={SkipReason.NO_EXTRACTABLE_CONTENT.value: 1},
    )

    restored = ProcessingReport.model_validate_json(report.model_dump_json())

    assert restored == report


def test_an_aborted_run_records_its_reason() -> None:
    report = _report(
        documents_processed=10, documents_failed=90, aborted=True,
        abort_reason="failure ratio",
    )

    assert report.aborted is True
    assert report.failure_ratio == pytest.approx(0.9)


# ---------------------------------------------------------------------------
# Anti-pattern guards
# ---------------------------------------------------------------------------


def test_models_do_not_import_the_contract_layer() -> None:
    """
    Interfaces imports these models under TYPE_CHECKING; the dependency runs
    in one direction only.
    """
    source = inspect.getsource(models_module)

    assert "from ptfm.core.knowledge.interfaces" not in source


def test_models_import_no_concrete_implementation() -> None:
    source = inspect.getsource(models_module)

    for forbidden in (
        "from ptfm.core.knowledge.chunking",
        "from ptfm.core.knowledge.embedding",
        "from ptfm.core.knowledge.pipeline",
        "from ptfm.plugins",
        "import numpy",
    ):
        assert forbidden not in source


def test_no_model_exposes_a_confidence_or_similarity_field() -> None:
    forbidden = {"confidence", "score", "similarity", "relevance", "probability"}

    for name in models_module.__all__:
        candidate = getattr(models_module, name)
        fields = getattr(candidate, "model_fields", None)
        if fields is not None:
            assert not forbidden & set(fields), name


def test_the_module_declares_no_mutable_state() -> None:
    mutable = [
        name
        for name, value in vars(models_module).items()
        if not name.startswith("__") and isinstance(value, (list, dict, set))
    ]

    assert mutable == []


def test_the_module_contains_no_domain_vocabulary() -> None:
    source = inspect.getsource(models_module).lower()

    for term in ("jurisdiction", "statute", "ticker", "icd", "patient", "court"):
        assert term not in source


def test_the_module_declares_no_thresholds() -> None:
    """
    failure_ratio and embedding_coverage compute values and compare them to
    nothing; the engine holds the injected bound.
    """
    source = inspect.getsource(models_module)

    assert "> 0.8" not in source
    assert "0.5" not in source


def test_similarity_hashing_is_not_routed_through_identity() -> None:
    """Traceback grouping is diagnostic, explicitly not artifact identity."""
    source = inspect.getsource(models_module)

    assert "hashlib.blake2b" in source
    assert "HashProvider" not in source
