"""
Unit tests for the validation stage and the Core knowledge sinks.

File:
    tests/core/knowledge/test_validation_sinks.py
"""

from __future__ import annotations

import asyncio
import inspect
from datetime import UTC, datetime
from typing import Any, Sequence

import pytest

from ptfm.core.knowledge.config import ValidationConfig
from ptfm.core.knowledge.exceptions import (
    CoverageError,
    NodeSpanIntegrityError,
    ProvenanceViolationError,
    SinkUnavailableError,
    SinkWriteError,
)
from ptfm.core.knowledge.models import KnowledgeUnit
from ptfm.core.knowledge.pipeline import validation as validation_module
from ptfm.core.knowledge.pipeline.validation import ProvenanceValidator
from ptfm.core.knowledge import sinks as sinks_module
from ptfm.core.knowledge.sinks import (
    CompositeKnowledgeSink,
    InMemoryKnowledgeSink,
    NullKnowledgeSink,
)
from ptfm.core.schemas import (
    BlockAttributes,
    BlockType,
    DocumentBlock,
    DocumentMetadata,
    KnowledgeNode,
    NodeEmbedding,
    NodeRelation,
    NodeType,
    RawDocument,
    RelationType,
    SourceDescriptor,
    TextSpan,
    derive_document_id,
    derive_node_id,
)

NOW = datetime(2026, 8, 1, tzinfo=UTC)
FINGERPRINT = "1:seg-fp-001"
DOCUMENT_ID = derive_document_id("file:///corpus/a.pdf", "hash-abc")
TEXT = "Alpha body.\n\nBravo body.\n\nCharlie body."
TEXT_HASH = "text-hash-abc"

_SEGMENTS = ((0, 11), (13, 24), (26, 39))


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def _source() -> SourceDescriptor:
    return SourceDescriptor(
        source_uri="file:///corpus/a.pdf",
        scheme="file",
        connector_name="LocalFilesystemConnector",
        extractor_name="PdfExtractor",
        acquired_at=NOW,
        size_bytes=4096,
        detected_media_type="application/pdf",
    )


def _document() -> RawDocument:
    blocks = tuple(
        DocumentBlock(
            block_id=str(index),
            block_type=BlockType.PARAGRAPH,
            span=TextSpan(start=start, end=end),
            depth=0,
            ordinal=index,
            attributes=BlockAttributes(),
        )
        for index, (start, end) in enumerate(_SEGMENTS)
    )
    return RawDocument(
        document_id=DOCUMENT_ID,
        source=_source(),
        metadata=DocumentMetadata(char_count=len(TEXT), word_count=len(TEXT.split())),
        text=TEXT,
        blocks=blocks,
        content_hash="hash-abc",
        text_hash=TEXT_HASH,
        ingested_at=NOW,
    )


def _node(index: int, **overrides: Any) -> KnowledgeNode:
    start, end = _SEGMENTS[index]
    span = overrides.pop("span", TextSpan(start=start, end=end))
    fingerprint = overrides.pop("segmentation_fingerprint", FINGERPRINT)
    document_id = overrides.pop("document_id", DOCUMENT_ID)
    fields: dict[str, Any] = {
        "node_id": derive_node_id(document_id, span, fingerprint),
        "document_id": document_id,
        "span": span,
        "text": TEXT[span.start : span.end],
        "node_type": NodeType.PROSE,
        "block_ids": (str(index),),
        "ordinal": index,
        "source": _source(),
        "segmentation_fingerprint": fingerprint,
        "source_text_hash": TEXT_HASH,
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
        "model_revision": "r1",
        "normalized": True,
        "composition_id": "heading_path_prefixed",
        "input_text_hash": f"input-{node.ordinal}",
        "embedded_at": NOW,
    }
    fields.update(overrides)
    return NodeEmbedding(**fields)


def _units(count: int = 3, embed: bool = False) -> list[KnowledgeUnit]:
    units = []
    for index in range(count):
        node = _node(index)
        units.append(
            KnowledgeUnit(
                node=node, embedding=_embedding(node) if embed else None
            )
        )
    return units


def _validator(**config: Any) -> ProvenanceValidator:
    return ProvenanceValidator(ValidationConfig(**config))


# ===========================================================================
# VALIDATION
# ===========================================================================


def test_a_well_formed_unit_set_passes() -> None:
    _validator().validate(_document(), _units(embed=True))


def test_an_empty_unit_set_passes() -> None:
    """A document yielding nothing is a skip, not a violation."""
    _validator().validate(_document(), [])


# ---------------------------------------------------------------------------
# Cross-unit embedding comparability -- the check no other stage can make
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "override",
    [
        {"model_id": "other/model"},
        {"model_revision": "r2"},
        {"composition_id": "text_only"},
        {"dimension": 3, "vector": (0.0, 0.6, 0.8)},
        {"normalized": False, "vector": (3.0, 4.0)},
    ],
)
def test_mixed_embedding_identity_within_a_document_is_rejected(
    override: dict[str, Any],
) -> None:
    """
    Similarity across differing identity is arithmetically valid and
    semantically meaningless, and nothing downstream reports an error.
    """
    units = _units(embed=True)
    odd_node = units[1].node
    units[1] = KnowledgeUnit(
        node=odd_node, embedding=_embedding(odd_node, **override)
    )

    with pytest.raises(ProvenanceViolationError, match="not comparable"):
        _validator().validate(_document(), units)


def test_a_partly_embedded_document_is_permitted() -> None:
    """Embedding is a separate artifact; a corpus may be embedded later."""
    units = _units(embed=True)
    units[1] = KnowledgeUnit(node=units[1].node)

    _validator().validate(_document(), units)


def test_comparability_is_checked_even_without_provenance_verification() -> None:
    """A structural check that is linear in node count runs unconditionally."""
    units = _units(embed=True)
    odd = units[1].node
    units[1] = KnowledgeUnit(node=odd, embedding=_embedding(odd, model_id="other"))

    with pytest.raises(ProvenanceViolationError):
        _validator(verify_provenance=False, require_full_coverage=False).validate(
            _document(), units
        )


def test_no_other_stage_performs_the_comparability_check() -> None:
    """
    Unit construction verifies ownership; a batch sees only itself. Only here
    does the whole document's set exist at once.
    """
    from ptfm.core.knowledge.pipeline import embedding as embedding_module

    assert "is_comparable_to" in inspect.getsource(validation_module)
    assert "is_comparable_to" not in inspect.getsource(embedding_module)


# ---------------------------------------------------------------------------
# Uniform provenance
# ---------------------------------------------------------------------------


def test_a_node_naming_another_document_is_rejected() -> None:
    units = _units()
    units[1] = KnowledgeUnit(node=_node(1, document_id="a-different-document"))

    with pytest.raises(ProvenanceViolationError, match="another document"):
        _validator().validate(_document(), units)


def test_mixed_segmentation_fingerprints_are_rejected() -> None:
    """Identifiers derived under differing segmentation occupy disjoint
    spaces, so these nodes only appear to belong together."""
    units = _units()
    units[1] = KnowledgeUnit(node=_node(1, segmentation_fingerprint="1:other"))

    with pytest.raises(ProvenanceViolationError, match="disjoint spaces"):
        _validator().validate(_document(), units)


def test_a_stale_source_text_hash_is_rejected() -> None:
    """The node's span was computed against different text from the document
    it names."""
    units = _units()
    units[1] = KnowledgeUnit(node=_node(1, source_text_hash="a-different-hash"))

    with pytest.raises(ProvenanceViolationError, match="different text"):
        _validator().validate(_document(), units)


# ---------------------------------------------------------------------------
# Identity
# ---------------------------------------------------------------------------


def test_a_repeated_node_identifier_is_rejected() -> None:
    """A sink would treat the second as an update to the first, silently
    discarding content."""
    units = _units()
    units[2] = KnowledgeUnit(node=units[0].node.model_copy(update={"ordinal": 2}))

    with pytest.raises(ProvenanceViolationError, match="appears twice"):
        _validator().validate(_document(), units)


def test_an_unreproducible_identifier_is_rejected() -> None:
    """
    The node would be unreachable by anything recomputing its identity,
    including a later idempotent run.
    """
    units = _units()
    units[1] = KnowledgeUnit(node=_node(1).model_copy(update={"node_id": "fabricated"}))

    with pytest.raises(ProvenanceViolationError, match="does not reproduce"):
        _validator().validate(_document(), units)


def test_identity_checking_survives_a_span_alteration() -> None:
    node = _node(1)
    altered = node.model_copy(
        update={"span": TextSpan(start=13, end=20), "text": TEXT[13:20]}
    )
    units = [KnowledgeUnit(node=_node(0)), KnowledgeUnit(node=altered)]

    with pytest.raises(ProvenanceViolationError, match="does not reproduce"):
        _validator(require_full_coverage=False).validate(_document(), units)


# ---------------------------------------------------------------------------
# Ordering
# ---------------------------------------------------------------------------


def test_non_contiguous_ordinals_are_rejected() -> None:
    units = _units()
    units[1] = KnowledgeUnit(node=_node(1).model_copy(update={"ordinal": 7}))

    with pytest.raises(ProvenanceViolationError, match="ordinal"):
        _validator().validate(_document(), units)


def test_regressing_spans_are_rejected() -> None:
    """Overlapping nodes place the same text twice, distorting term frequency
    statistics."""
    units = _units()
    units.reverse()
    units = [
        KnowledgeUnit(node=unit.node.model_copy(update={"ordinal": index}))
        for index, unit in enumerate(units)
    ]

    with pytest.raises(ProvenanceViolationError, match="before its predecessor"):
        _validator(require_full_coverage=False).validate(_document(), units)


# ---------------------------------------------------------------------------
# Text and coverage
# ---------------------------------------------------------------------------


def test_altered_node_text_is_rejected() -> None:
    """Assembly slices text from the span and enrichment is additive, so a
    node was altered after construction."""
    units = _units()
    units[1] = KnowledgeUnit(node=_node(1).model_copy(update={"text": "tampered!!"}))

    with pytest.raises(NodeSpanIntegrityError, match="altered after construction"):
        _validator().validate(_document(), units)


def test_a_lost_node_is_detected_as_uncovered_content() -> None:
    """
    Planning verified coverage of the plan; this verifies coverage of what was
    emitted, which is a different statement.
    """
    units = _units()
    del units[1]
    units[1] = KnowledgeUnit(node=units[1].node.model_copy(update={"ordinal": 1}))

    with pytest.raises(CoverageError, match="lost between planning and emission"):
        _validator().validate(_document(), units)


def test_text_verification_can_be_disabled() -> None:
    units = _units()
    units[1] = KnowledgeUnit(node=_node(1).model_copy(update={"text": "tampered!!"}))

    _validator(verify_provenance=False, require_full_coverage=False).validate(
        _document(), units
    )


def test_coverage_verification_can_be_disabled() -> None:
    units = _units()
    del units[1]
    units[1] = KnowledgeUnit(node=units[1].node.model_copy(update={"ordinal": 1}))

    _validator(require_full_coverage=False).validate(_document(), units)


def test_both_document_walking_checks_default_on() -> None:
    config = ValidationConfig()

    assert config.verify_provenance is True
    assert config.require_full_coverage is True


# ===========================================================================
# SINKS
# ===========================================================================


async def test_the_null_sink_discards_and_counts() -> None:
    """A run can exercise the whole pipeline without persisting anything."""
    sink = NullKnowledgeSink()

    for unit in _units():
        await sink.write(unit)
    await sink.flush()
    await sink.close()

    assert sink.written_count == 3


async def test_the_null_sink_rejects_writes_after_close() -> None:
    """A run reporting success for units nothing received would be
    indistinguishable from one that persisted them."""
    sink = NullKnowledgeSink()
    await sink.close()

    with pytest.raises(SinkWriteError, match="closed sink"):
        await sink.write(_units()[0])


async def test_the_in_memory_sink_retains_and_reports() -> None:
    sink = InMemoryKnowledgeSink()

    for unit in _units(embed=True):
        await sink.write(unit)

    assert sink.node_count == 3
    assert sink.embedding_count == 3
    assert len(sink) == 3


async def test_the_in_memory_sink_reports_partial_embedding() -> None:
    """An unembedded node is invisible to dense retrieval."""
    sink = InMemoryKnowledgeSink()
    units = _units(embed=True)
    units[1] = KnowledgeUnit(node=units[1].node)

    for unit in units:
        await sink.write(unit)

    assert sink.node_count == 3
    assert sink.embedding_count == 2


async def test_rewriting_a_node_converges_rather_than_duplicating() -> None:
    """Identity is deterministic, so rewriting a node rewrites identical
    content."""
    sink = InMemoryKnowledgeSink()
    unit = _units()[0]

    await sink.write(unit)
    await sink.write(unit)

    assert sink.node_count == 1


async def test_units_are_retrievable_by_node_and_document() -> None:
    sink = InMemoryKnowledgeSink()
    units = _units()
    for unit in units:
        await sink.write(unit)

    assert sink.get(units[1].node_id) == units[1]
    assert sink.get("absent") is None
    assert len(sink.units_for(DOCUMENT_ID)) == 3
    assert sink.document_ids() == (DOCUMENT_ID,)


async def test_document_units_are_returned_in_reading_order() -> None:
    sink = InMemoryKnowledgeSink()
    for unit in reversed(_units()):
        await sink.write(unit)

    ordinals = [unit.node.ordinal for unit in sink.units_for(DOCUMENT_ID)]

    assert ordinals == [0, 1, 2]


async def test_relations_are_counted() -> None:
    sink = InMemoryKnowledgeSink()
    node = _node(0)
    other = _node(1)
    await sink.write(
        KnowledgeUnit(
            node=node,
            relations=(
                NodeRelation(
                    source_node_id=node.node_id,
                    target_node_id=other.node_id,
                    relation_type=RelationType.PRECEDES,
                    document_id=DOCUMENT_ID,
                ),
            ),
        )
    )

    assert sink.relation_count == 1


async def test_retained_units_survive_closing() -> None:
    """A test must be able to assert what a run wrote after the run tore
    down."""
    sink = InMemoryKnowledgeSink()
    await sink.write(_units()[0])
    await sink.close()

    assert sink.is_closed is True
    assert sink.node_count == 1


async def test_closing_is_idempotent() -> None:
    sink = InMemoryKnowledgeSink()

    await sink.close()
    await sink.close()

    assert sink.is_closed is True


async def test_concurrent_writes_are_all_retained() -> None:
    """The engine processes documents concurrently, so a sink is written to
    concurrently."""
    sink = InMemoryKnowledgeSink()
    units = _units()

    await asyncio.gather(*(sink.write(unit) for unit in units * 20))

    assert sink.node_count == 3


# ---------------------------------------------------------------------------
# Composite
# ---------------------------------------------------------------------------


async def test_a_composite_fans_out_to_every_destination() -> None:
    first = InMemoryKnowledgeSink("first")
    second = InMemoryKnowledgeSink("second")
    composite = CompositeKnowledgeSink([first, second])

    for unit in _units():
        await composite.write(unit)
    await composite.flush()
    await composite.close()

    assert first.node_count == 3
    assert second.node_count == 3
    assert first.flush_count == 1
    assert first.is_closed and second.is_closed


def test_an_empty_composite_is_rejected() -> None:
    """It would discard a run's output while reporting success."""
    with pytest.raises(SinkWriteError, match="at least one destination"):
        CompositeKnowledgeSink([])


def test_composed_sinks_must_have_distinct_names() -> None:
    """A failure could not be attributed to one of them."""
    with pytest.raises(SinkWriteError, match="report the name"):
        CompositeKnowledgeSink(
            [InMemoryKnowledgeSink("same"), InMemoryKnowledgeSink("same")]
        )


def test_a_composed_sink_must_report_a_name() -> None:
    with pytest.raises(SinkWriteError, match="does not report a name"):
        CompositeKnowledgeSink([object()])


async def test_a_write_failure_propagates_from_the_composite() -> None:
    class _Failing:
        name = "failing"

        async def write(self, unit: Any) -> None:
            raise SinkUnavailableError("unreachable", sink_name=self.name)

        async def flush(self) -> None:
            return None

        async def close(self) -> None:
            return None

    composite = CompositeKnowledgeSink([InMemoryKnowledgeSink("ok"), _Failing()])

    with pytest.raises(SinkUnavailableError):
        await composite.write(_units()[0])


async def test_a_partial_write_is_recoverable_by_re_running() -> None:
    """
    Determinism is what makes the composite's partial-write behaviour
    tolerable: a rewrite is idempotent.
    """
    good = InMemoryKnowledgeSink("good")

    class _Failing:
        name = "failing"
        calls = 0

        async def write(self, unit: Any) -> None:
            type(self).calls += 1
            if type(self).calls == 1:
                raise SinkUnavailableError("blip", sink_name=self.name)

        async def flush(self) -> None:
            return None

        async def close(self) -> None:
            return None

    composite = CompositeKnowledgeSink([good, _Failing()])
    unit = _units()[0]

    with pytest.raises(SinkUnavailableError):
        await composite.write(unit)
    await composite.write(unit)

    assert good.node_count == 1


async def test_every_destination_is_closed_even_when_one_fails() -> None:
    """A destination left open leaks whatever resources it holds."""
    closed: list[str] = []

    class _FailingClose:
        name = "failing"

        async def write(self, unit: Any) -> None:
            return None

        async def flush(self) -> None:
            return None

        async def close(self) -> None:
            closed.append(self.name)
            raise SinkWriteError("close failed", sink_name=self.name)

    later = InMemoryKnowledgeSink("later")
    composite = CompositeKnowledgeSink([_FailingClose(), later])

    with pytest.raises(SinkWriteError):
        await composite.close()

    assert closed == ["failing"]
    assert later.is_closed is True


def test_a_composite_names_its_destinations() -> None:
    composite = CompositeKnowledgeSink(
        [InMemoryKnowledgeSink("vector"), InMemoryKnowledgeSink("lexical")]
    )

    assert composite.name == "composite(vector, lexical)"
    assert len(composite) == 2


# ===========================================================================
# Anti-pattern guards
# ===========================================================================


def test_the_validator_is_synchronous() -> None:
    source = inspect.getsource(validation_module)

    assert "async def" not in source


def test_the_sinks_name_no_storage_technology() -> None:
    """Retrieval owns vector, lexical, metadata and graph storage."""
    source = inspect.getsource(sinks_module).lower()

    for term in (
        "faiss",
        "qdrant",
        "milvus",
        "chroma",
        "elasticsearch",
        "neo4j",
        "hnsw",
        "bm25",
        "postgres",
    ):
        assert term not in source


@pytest.mark.parametrize("module", [validation_module, sinks_module])
def test_no_module_imports_a_concrete_implementation(module: Any) -> None:
    source = inspect.getsource(module)

    for forbidden in (
        "from ptfm.core.knowledge.chunking",
        "from ptfm.core.knowledge.embedding",
        "from ptfm.core.knowledge.registry",
        "from ptfm.plugins",
        "import numpy",
    ):
        assert forbidden not in source


@pytest.mark.parametrize("module", [validation_module, sinks_module])
def test_no_module_declares_mutable_module_state(module: Any) -> None:
    mutable = [
        name
        for name, value in vars(module).items()
        if not name.startswith("_") and isinstance(value, (list, dict, set))
    ]

    assert mutable == []


@pytest.mark.parametrize("module", [validation_module, sinks_module])
def test_no_module_produces_a_confidence_value(module: Any) -> None:
    source = inspect.getsource(module).lower()

    assert "confidence" not in source


@pytest.mark.parametrize("module", [validation_module, sinks_module])
def test_no_module_contains_domain_vocabulary(module: Any) -> None:
    source = inspect.getsource(module).lower()

    for term in ("jurisdiction", "statute", "ticker", "icd", "patient", "court"):
        assert term not in source
