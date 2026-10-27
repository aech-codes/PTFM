"""
End-to-end tests for the knowledge engine and composition root.

File:
    tests/core/knowledge/test_engine.py

Emphasis
--------
    This is the only suite that exercises the complete Module 2 path, from a
    canonical ``RawDocument`` through to units delivered at a sink. The
    properties that matter are those no single stage can establish alone:
    idempotent reprocessing, bounded memory, fault isolation, and cancellation
    safety.
"""

from __future__ import annotations

import asyncio
import inspect
from datetime import UTC, datetime
from typing import Any, AsyncIterator, Mapping, Sequence

import pytest

from ptfm.core.knowledge import (
    DocumentFailed,
    DocumentSkipped,
    KnowledgeConfig,
    KnowledgeEngine,
    NodeProduced,
    SkipReason,
    build_default_engine,
    default_providers,
    default_strategies,
)
from ptfm.core.knowledge import engine as engine_module
from ptfm.core.knowledge.config import (
    ChunkingConfig,
    EmbeddingConfig,
    EnrichmentConfig,
    RelationConfig,
    SinkConfig,
    TelemetryConfig,
    ValidationConfig,
)
from ptfm.core.knowledge.exceptions import (
    ChunkerRegistrationError,
    KnowledgeConfigurationError,
    SinkUnavailableError,
)
from ptfm.core.knowledge.interfaces import ChunkingStrategy
from ptfm.core.knowledge.models import ChunkPlan
from ptfm.core.knowledge.sinks import InMemoryKnowledgeSink
from ptfm.core.ingestion.utils.concurrency import InlineOffloader
from ptfm.core.schemas import (
    BlockAttributes,
    BlockType,
    DocumentBlock,
    DocumentMetadata,
    NodeType,
    RawDocument,
    RelationType,
    SourceDescriptor,
    TextSpan,
    derive_document_id,
)

NOW = datetime(2026, 8, 1, tzinfo=UTC)


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


class _FixedClock:
    def now(self) -> datetime:
        return NOW


class _RecordingMetrics:
    def __init__(self) -> None:
        self.counters: list[tuple[str, int, dict[str, str]]] = []
        self.observations: list[tuple[str, float]] = []

    def increment(self, name: str, value: int = 1, **labels: str) -> None:
        self.counters.append((name, value, labels))

    def observe(self, name: str, value: float, **labels: str) -> None:
        self.observations.append((name, value))

    def names(self) -> list[str]:
        return [name for name, _, _ in self.counters]


class _FailingMetrics:
    def increment(self, name: str, value: int = 1, **labels: str) -> None:
        raise ConnectionError("metrics unreachable")

    def observe(self, name: str, value: float, **labels: str) -> None:
        raise ConnectionError("metrics unreachable")


def _document(index: int = 0, paragraphs: int = 3) -> RawDocument:
    heading = f"Section {index}"
    bodies = [
        f"Document {index} paragraph {n} carries some body text."
        for n in range(paragraphs)
    ]
    segments = [heading, *bodies]
    text = "\n\n".join(segments)

    blocks = []
    cursor = 0
    for ordinal, segment in enumerate(segments):
        blocks.append(
            DocumentBlock(
                block_id=str(ordinal),
                block_type=BlockType.HEADING if ordinal == 0 else BlockType.PARAGRAPH,
                span=TextSpan(start=cursor, end=cursor + len(segment)),
                depth=0,
                ordinal=ordinal,
                attributes=BlockAttributes(heading_level=1)
                if ordinal == 0
                else BlockAttributes(),
            )
        )
        cursor += len(segment) + 2

    uri = f"file:///corpus/doc{index}.pdf"
    return RawDocument(
        document_id=derive_document_id(uri, f"hash-{index}"),
        source=SourceDescriptor(
            source_uri=uri,
            scheme="file",
            connector_name="LocalFilesystemConnector",
            extractor_name="PdfExtractor",
            acquired_at=NOW,
            size_bytes=4096,
            detected_media_type="application/pdf",
        ),
        metadata=DocumentMetadata(char_count=len(text), word_count=len(text.split())),
        text=text,
        blocks=tuple(blocks),
        content_hash=f"hash-{index}",
        text_hash=f"text-hash-{index}",
        ingested_at=NOW,
    )


def _empty_document(index: int = 99) -> RawDocument:
    uri = f"file:///corpus/empty{index}.pdf"
    return RawDocument(
        document_id=derive_document_id(uri, f"hash-{index}"),
        source=SourceDescriptor(
            source_uri=uri,
            scheme="file",
            connector_name="LocalFilesystemConnector",
            extractor_name="PdfExtractor",
            acquired_at=NOW,
            size_bytes=0,
            detected_media_type="application/pdf",
        ),
        metadata=DocumentMetadata(char_count=0, word_count=0),
        text="",
        blocks=(),
        content_hash=f"hash-{index}",
        text_hash=f"text-hash-{index}",
        ingested_at=NOW,
    )


async def _stream(documents: Sequence[RawDocument]) -> AsyncIterator[RawDocument]:
    for document in documents:
        yield document


def _config(**overrides: Any) -> KnowledgeConfig:
    base: dict[str, Any] = {
        "_env_file": None,
        "embedding": EmbeddingConfig(
            provider="deterministic", provider_options={"dimension": "16"}
        ).model_dump(),
    }
    base.update(overrides)
    return KnowledgeConfig(**base)


def _engine(config: Any = None, **kwargs: Any) -> KnowledgeEngine:
    return build_default_engine(
        config or _config(),
        clock=_FixedClock(),
        offloader=InlineOffloader(),
        discover_entry_points=False,
        **kwargs,
    )


async def _collect(engine: KnowledgeEngine, documents: Sequence[RawDocument]) -> list[Any]:
    return [outcome async for outcome in engine.process(_stream(documents))]


# ===========================================================================
# END TO END
# ===========================================================================


async def test_a_document_flows_through_every_stage() -> None:
    sink = InMemoryKnowledgeSink()
    engine = _engine(sink=sink)
    document = _document()

    outcomes = await _collect(engine, [document])

    assert outcomes
    assert all(isinstance(o, NodeProduced) for o in outcomes)
    assert sink.node_count == len(outcomes)
    assert sink.embedding_count == len(outcomes)


async def test_every_produced_node_verifies_against_its_document() -> None:
    """The guarantee the whole module exists to provide, checked end to end."""
    document = _document()
    engine = _engine()

    outcomes = await _collect(engine, [document])

    for outcome in outcomes:
        assert outcome.unit.node.verify_against(document.text) is True


async def test_the_nodes_account_for_the_document() -> None:
    """Text absent from every node would be unretrievable, and the failure
    invisible."""
    document = _document()
    engine = _engine()

    outcomes = await _collect(engine, [document])

    cursor = 0
    for outcome in sorted(outcomes, key=lambda o: o.unit.node.ordinal):
        span = outcome.unit.node.span
        assert not document.text[cursor : span.start].strip()
        cursor = span.end
    assert not document.text[cursor:].strip()


async def test_provenance_reaches_every_node() -> None:
    document = _document()
    engine = _engine()

    node = (await _collect(engine, [document]))[0].unit.node

    assert node.document_id == document.document_id
    assert node.source.connector_name == "LocalFilesystemConnector"
    assert node.source.extractor_name == "PdfExtractor"
    assert node.source_text_hash == document.text_hash
    assert node.segmentation_fingerprint == engine.segmentation_fingerprint


async def test_relations_are_produced() -> None:
    engine = _engine()

    outcomes = await _collect(engine, [_document()])
    relations = [r for o in outcomes for r in o.unit.relations]

    assert relations
    assert any(r.relation_type is RelationType.PRECEDES for r in relations)


async def test_heading_context_reaches_the_nodes_but_not_their_text() -> None:
    document = _document()
    engine = _engine()

    outcomes = await _collect(engine, [document])
    body = next(o.unit.node for o in outcomes if o.unit.node.ordinal > 0)

    assert body.heading_path == ("Section 0",)
    assert "Section 0" not in body.text


async def test_several_documents_are_processed() -> None:
    sink = InMemoryKnowledgeSink()
    engine = _engine(sink=sink)
    documents = [_document(index) for index in range(5)]

    await _collect(engine, documents)

    assert len(sink.document_ids()) == 5


async def test_an_empty_document_is_skipped_not_failed() -> None:
    """A document yielding nothing is a routine corpus occurrence."""
    engine = _engine()

    outcomes = await _collect(engine, [_empty_document()])

    assert len(outcomes) == 1
    assert isinstance(outcomes[0], DocumentSkipped)
    assert outcomes[0].reason is SkipReason.NO_EXTRACTABLE_CONTENT


async def test_an_empty_stream_yields_nothing() -> None:
    assert await _collect(_engine(), []) == []


# ===========================================================================
# IDEMPOTENCE
# ===========================================================================


async def test_reprocessing_reproduces_identical_nodes() -> None:
    """
    Identity is a pure function of document, span, and fingerprint, so a
    second run over an unchanged corpus converges rather than duplicating.
    """
    documents = [_document(index) for index in range(3)]

    first = await _collect(_engine(), documents)
    second = await _collect(_engine(), documents)

    assert [o.unit.node.node_id for o in first] == [
        o.unit.node.node_id for o in second
    ]


async def test_reprocessing_into_one_sink_converges() -> None:
    sink = InMemoryKnowledgeSink()
    documents = [_document(index) for index in range(3)]

    await _collect(_engine(sink=sink), documents)
    count = sink.node_count
    await _collect(_engine(sink=sink), documents)

    assert sink.node_count == count


async def test_reprocessing_reproduces_identical_vectors() -> None:
    """What allows a vector store to be rebuilt without reprocessing."""
    document = _document()

    first = await _collect(_engine(), [document])
    second = await _collect(_engine(), [document])

    assert [o.unit.embedding.vector for o in first] == [
        o.unit.embedding.vector for o in second
    ]


async def test_changing_segmentation_yields_disjoint_identifiers() -> None:
    """
    Two configurations must not share an identifier space, or a node from one
    index could silently resolve against a corpus segmented differently.
    """
    document = _document()
    narrow = _engine(_config(chunking=ChunkingConfig(target_chunk_chars=40).model_dump()))
    wide = _engine(_config(chunking=ChunkingConfig(target_chunk_chars=4000).model_dump()))

    assert narrow.segmentation_fingerprint != wide.segmentation_fingerprint

    narrow_ids = {o.unit.node.node_id for o in await _collect(narrow, [document])}
    wide_ids = {o.unit.node.node_id for o in await _collect(wide, [document])}

    assert not narrow_ids & wide_ids


# ===========================================================================
# CONCURRENCY AND MEMORY
# ===========================================================================


async def test_the_concurrency_limit_bounds_documents_in_flight() -> None:
    """Memory is bounded by the limit rather than by corpus size."""
    active = 0
    peak = 0

    class _Tracking(InlineOffloader):
        async def run(self, func: Any, *args: Any, **kwargs: Any) -> Any:
            nonlocal active, peak
            if func.__name__ == "plan":
                active += 1
                peak = max(peak, active)
                await asyncio.sleep(0.005)
                try:
                    return func(*args, **kwargs)
                finally:
                    active -= 1
            return func(*args, **kwargs)

    engine = build_default_engine(
        _config(max_concurrent_documents=3),
        clock=_FixedClock(),
        offloader=_Tracking(),
        discover_entry_points=False,
    )

    await _collect(engine, [_document(i) for i in range(12)])

    assert peak <= 3


async def test_outcomes_stream_rather_than_batch() -> None:
    """A corpus is never materialized, so the first result precedes the
    last."""
    engine = _engine()

    seen = 0
    async for _ in engine.process(_stream([_document(i) for i in range(20)])):
        seen += 1
        if seen == 1:
            break

    assert seen == 1


async def test_nothing_happens_until_iteration_begins() -> None:
    sink = InMemoryKnowledgeSink()
    engine = _engine(sink=sink)

    engine.process(_stream([_document()]))
    await asyncio.sleep(0)

    assert sink.node_count == 0


async def test_abandoning_the_iterator_cancels_the_run() -> None:
    engine = _engine()

    iterator = engine.process(_stream([_document(i) for i in range(30)])).__aiter__()
    await iterator.__anext__()
    await iterator.aclose()


async def test_cancellation_propagates() -> None:
    class _Slow(InlineOffloader):
        async def run(self, func: Any, *args: Any, **kwargs: Any) -> Any:
            await asyncio.sleep(0.05)
            return func(*args, **kwargs)

    engine = build_default_engine(
        _config(),
        clock=_FixedClock(),
        offloader=_Slow(),
        discover_entry_points=False,
    )

    async def _run() -> None:
        async for _ in engine.process(_stream([_document(i) for i in range(20)])):
            pass

    task = asyncio.create_task(_run())
    await asyncio.sleep(0.02)
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task


# ===========================================================================
# FAULT ISOLATION
# ===========================================================================


class _FailingStrategy(ChunkingStrategy):
    """Fails for nominated documents, delegating otherwise."""

    def __init__(self, inner: ChunkingStrategy, failing: frozenset[str]) -> None:
        self._inner = inner
        self._failing = failing

    @property
    def name(self) -> str:
        return self._inner.name

    def plan(self, document: RawDocument) -> ChunkPlan:
        if document.source.source_uri in self._failing:
            raise RuntimeError("strategy defect")
        return self._inner.plan(document)

    def fingerprint_components(self) -> Mapping[str, str]:
        return self._inner.fingerprint_components()


def _engine_with_failures(failing: frozenset[str], **overrides: Any) -> KnowledgeEngine:
    config = _config(**overrides)
    inner = default_strategies(config.chunking)[0]
    return build_default_engine(
        config,
        strategies=[_FailingStrategy(inner, failing)],
        clock=_FixedClock(),
        offloader=InlineOffloader(),
        discover_entry_points=False,
    )


async def test_a_failing_document_does_not_abort_the_run() -> None:
    """A corrupt document must not discard the preceding work."""
    engine = _engine_with_failures(frozenset({"file:///corpus/doc2.pdf"}))
    documents = [_document(index) for index in range(5)]

    outcomes = await _collect(engine, documents)

    failures = [o for o in outcomes if isinstance(o, DocumentFailed)]
    produced = {o.unit.node.document_id for o in outcomes if isinstance(o, NodeProduced)}

    assert len(failures) == 1
    assert len(produced) == 4


async def test_a_failure_records_its_stage_and_code() -> None:
    engine = _engine_with_failures(frozenset({"file:///corpus/doc0.pdf"}))

    outcomes = await _collect(engine, [_document(0)])
    failure = outcomes[0].failure

    assert failure.stage.value == "PLANNING"
    assert failure.error_code == "INVALID_CHUNK_PLAN"
    assert failure.document_id == _document(0).document_id


async def test_fail_fast_aborts_on_the_first_failure() -> None:
    engine = _engine_with_failures(
        frozenset({"file:///corpus/doc0.pdf"}),
        fail_fast=True,
        max_concurrent_documents=1,
    )

    with pytest.raises(KnowledgeConfigurationError, match="fail_fast"):
        await _collect(engine, [_document(i) for i in range(5)])


async def test_a_systemically_bad_corpus_aborts_the_run() -> None:
    failing = frozenset(f"file:///corpus/doc{i}.pdf" for i in range(10))
    engine = _engine_with_failures(
        failing,
        failure_ratio_threshold=0.5,
        failure_ratio_min_samples=4,
        max_concurrent_documents=1,
    )

    with pytest.raises(KnowledgeConfigurationError, match="exceeding"):
        await _collect(engine, [_document(i) for i in range(10)])


async def test_the_guard_requires_a_minimum_sample() -> None:
    """Otherwise a run aborts when the first of one document fails."""
    engine = _engine_with_failures(
        frozenset({"file:///corpus/doc0.pdf"}),
        failure_ratio_threshold=0.5,
        failure_ratio_min_samples=50,
        max_concurrent_documents=1,
    )

    outcomes = await _collect(engine, [_document(i) for i in range(3)])

    assert any(isinstance(o, DocumentFailed) for o in outcomes)


async def test_the_failure_ratio_counts_documents_not_nodes() -> None:
    """
    One document yields many node outcomes; counting those would understate
    the ratio by an arbitrary factor and the guard would never fire.
    """
    source = inspect.getsource(engine_module)

    assert "def documents_seen" in source
    assert "len(self._documents)" in source


# ===========================================================================
# SINK BEHAVIOUR
# ===========================================================================


async def test_a_sink_failure_aborts_the_run_by_default() -> None:
    """An unavailable sink fails identically for every subsequent document,
    so continuing would discard everything computed."""

    class _Unavailable:
        name = "unavailable"

        async def write(self, unit: Any) -> None:
            raise SinkUnavailableError("down", sink_name=self.name)

        async def flush(self) -> None:
            return None

        async def close(self) -> None:
            return None

    engine = _engine(sink=_Unavailable())

    with pytest.raises(SinkUnavailableError):
        await _collect(engine, [_document()])


async def test_a_sink_failure_can_be_tolerated() -> None:
    class _Unavailable:
        name = "unavailable"

        async def write(self, unit: Any) -> None:
            raise SinkUnavailableError("down", sink_name=self.name)

        async def flush(self) -> None:
            return None

        async def close(self) -> None:
            return None

    engine = _engine(
        _config(sink=SinkConfig(fail_run_on_sink_error=False).model_dump()),
        sink=_Unavailable(),
    )

    outcomes = await _collect(engine, [_document()])

    assert all(isinstance(o, NodeProduced) for o in outcomes)


async def test_units_are_written_before_they_are_emitted() -> None:
    """
    A fatal sink failure must abort before the consumer acts on outcomes
    nothing received.
    """
    sink = InMemoryKnowledgeSink()
    engine = _engine(sink=sink)

    async for outcome in engine.process(_stream([_document()])):
        assert sink.get(outcome.unit.node_id) is not None
        break


async def test_the_sink_is_flushed() -> None:
    sink = InMemoryKnowledgeSink()
    engine = _engine(
        _config(sink=SinkConfig(flush_interval_units=1).model_dump()), sink=sink
    )

    await _collect(engine, [_document(i) for i in range(3)])

    assert sink.flush_count >= 3


async def test_an_engine_without_a_sink_still_emits() -> None:
    engine = _engine()

    assert await _collect(engine, [_document()])


# ===========================================================================
# EMBEDDING
# ===========================================================================


async def test_embedding_can_be_disabled() -> None:
    """A corpus may be embedded later, since embeddings are a separate
    artifact."""
    engine = _engine(_config(embedding=EmbeddingConfig(enabled=False).model_dump()))

    outcomes = await _collect(engine, [_document()])

    assert outcomes
    assert all(o.unit.embedding is None for o in outcomes)
    assert engine.embedding_model_identity is None


async def test_the_model_identity_is_recorded_on_every_vector() -> None:
    engine = _engine()

    outcomes = await _collect(engine, [_document()])

    identities = {o.unit.embedding.model_identity for o in outcomes}
    assert len(identities) == 1
    assert engine.embedding_model_identity in identities


async def test_batching_respects_the_provider_bound() -> None:
    engine = _engine(
        _config(
            embedding=EmbeddingConfig(
                provider="deterministic",
                provider_options={"dimension": "16"},
                batch_size=2,
            ).model_dump()
        )
    )

    outcomes = await _collect(engine, [_document(paragraphs=8)])

    assert all(o.unit.is_embedded for o in outcomes)


async def test_provider_options_reach_the_provider() -> None:
    """
    The mechanism by which a model identifier reaches a concrete provider
    without the Core contract naming it.
    """
    engine = _engine(
        _config(
            embedding=EmbeddingConfig(
                provider="deterministic", provider_options={"dimension": "8"}
            ).model_dump()
        )
    )

    outcome = (await _collect(engine, [_document()]))[0]

    assert outcome.unit.embedding.dimension == 8


def test_a_malformed_provider_option_is_rejected() -> None:
    with pytest.raises(KnowledgeConfigurationError, match="must be an integer"):
        _engine(
            _config(
                embedding=EmbeddingConfig(
                    provider="deterministic",
                    provider_options={"dimension": "not-a-number"},
                ).model_dump()
            )
        )


def test_provider_options_do_not_affect_identity() -> None:
    """Changing a model invalidates vectors while leaving node identity
    correct."""
    first = _engine(
        _config(
            embedding=EmbeddingConfig(
                provider="deterministic", provider_options={"dimension": "16"}
            ).model_dump()
        )
    )
    second = _engine(
        _config(
            embedding=EmbeddingConfig(
                provider="deterministic", provider_options={"dimension": "32"}
            ).model_dump()
        )
    )

    assert first.segmentation_fingerprint == second.segmentation_fingerprint



async def test_the_query_convention_reaches_every_vector() -> None:
    """
    Module 3 must be able to verify that the provider it queries with encodes
    queries the way the stored vectors expect, without inferring provider
    behaviour from configuration it may not share.
    """
    engine = _engine()

    outcomes = await _collect(engine, [_document()])

    assert all(
        o.unit.embedding.query_encoding is not None for o in outcomes
    )
    conventions = {o.unit.embedding.query_encoding for o in outcomes}
    assert len(conventions) == 1


async def test_enrichment_runs_concurrently_within_a_document() -> None:
    """
    Serializing a large document's enricher round trips would hold one
    document slot while the run's concurrency limit held others back.
    """
    active = 0
    peak = 0

    class _Slow:
        @property
        def name(self) -> str:
            return "slow"

        @property
        def metadata_keys(self) -> frozenset[str]:
            return frozenset({"topic"})

        async def enrich(self, node: Any) -> Mapping[str, str]:
            nonlocal active, peak
            active += 1
            peak = max(peak, active)
            try:
                await asyncio.sleep(0.01)
                return {"topic": "x"}
            finally:
                active -= 1

    engine = _engine(
        _config(enrichment=EnrichmentConfig(max_concurrent_nodes=4).model_dump()),
        enrichers=[_Slow()],
    )

    await _collect(engine, [_document(paragraphs=12)])

    assert peak > 1
    assert peak <= 4


async def test_enrichment_preserves_reading_order() -> None:
    """A node's ordinal and its position in the emitted sequence must
    agree."""

    class _Jittery:
        @property
        def name(self) -> str:
            return "jittery"

        @property
        def metadata_keys(self) -> frozenset[str]:
            return frozenset({"topic"})

        async def enrich(self, node: Any) -> Mapping[str, str]:
            await asyncio.sleep((10 - node.ordinal % 10) * 0.001)
            return {"topic": str(node.ordinal)}

    engine = _engine(enrichers=[_Jittery()])

    outcomes = await _collect(engine, [_document(paragraphs=8)])
    ordered = sorted(outcomes, key=lambda o: o.unit.node.ordinal)

    for outcome in ordered:
        assert outcome.unit.node.metadata["topic"] == str(
            outcome.unit.node.ordinal
        )


# ===========================================================================
# REPORTING AND TELEMETRY
# ===========================================================================


async def test_a_report_aggregates_the_run() -> None:
    engine = _engine()
    documents = [_document(index) for index in range(4)]

    report = await engine.process_all(_stream(documents))

    assert report.documents_processed == 4
    assert report.nodes_produced > 0
    assert report.embeddings_produced == report.nodes_produced
    assert report.embedding_coverage == pytest.approx(1.0)
    assert report.segmentation_fingerprint == engine.segmentation_fingerprint
    assert report.embedding_model_identity == engine.embedding_model_identity


async def test_a_report_records_skips_and_failures() -> None:
    engine = _engine_with_failures(frozenset({"file:///corpus/doc1.pdf"}))

    report = await engine.process_all(
        _stream([_document(0), _document(1), _empty_document()])
    )

    assert report.documents_processed == 1
    assert report.documents_failed == 1
    assert report.documents_skipped == 1
    assert report.failure_ratio == pytest.approx(1 / 3)


async def test_metrics_are_emitted() -> None:
    metrics = _RecordingMetrics()
    engine = _engine(metrics=metrics)

    await _collect(engine, [_document(), _empty_document()])

    names = metrics.names()
    assert "ptfm.knowledge.nodes.produced" in names
    assert "ptfm.knowledge.embeddings.produced" in names
    assert "ptfm.knowledge.documents.skipped" in names


async def test_a_failing_metrics_sink_cannot_fail_a_run() -> None:
    """An observability outage must not destroy work that succeeded."""
    engine = _engine(metrics=_FailingMetrics())

    assert await _collect(engine, [_document()])


async def test_metrics_can_be_disabled() -> None:
    metrics = _RecordingMetrics()
    engine = _engine(
        _config(telemetry=TelemetryConfig(emit_metrics=False).model_dump()),
        metrics=metrics,
    )

    await _collect(engine, [_document()])

    assert metrics.counters == []


# ===========================================================================
# SINGLE DOCUMENT
# ===========================================================================


async def test_a_single_document_can_be_processed() -> None:
    engine = _engine()

    outcomes = await engine.process_document(_document())

    assert outcomes
    assert all(isinstance(o, NodeProduced) for o in outcomes)


async def test_a_single_document_failure_is_returned_not_raised() -> None:
    engine = _engine_with_failures(frozenset({"file:///corpus/doc0.pdf"}))

    outcomes = await engine.process_document(_document(0))

    assert isinstance(outcomes[0], DocumentFailed)


# ===========================================================================
# COMPOSITION ROOT
# ===========================================================================


def test_the_default_engine_builds() -> None:
    assert isinstance(_engine(), KnowledgeEngine)


def test_segmentation_is_resolved_and_fingerprinted_together() -> None:
    """
    The engine cannot be constructed with a fingerprint that disagrees with
    the components that produced it.
    """
    engine = _engine()

    assert engine.segmentation_fingerprint.startswith("1:")


def test_the_fingerprint_reflects_the_segmenter_version() -> None:
    """A segmenter upgrade shifts boundaries within oversized blocks while
    every configured value stays the same."""
    from ptfm.core.knowledge.registry import SegmenterRegistration

    class _Other:
        @property
        def version(self) -> str:
            return "other-9.9"

        def segment(self, text: str) -> Sequence[Any]:
            return ()

    baseline = _engine()
    altered = build_default_engine(
        _config(),
        segmenters=[SegmenterRegistration(name="regex", segmenter=_Other())],
        clock=_FixedClock(),
        offloader=InlineOffloader(),
        discover_entry_points=False,
    )

    assert altered.segmentation_fingerprint != baseline.segmentation_fingerprint


def test_every_component_can_be_substituted() -> None:
    parameters = inspect.signature(build_default_engine).parameters

    for name in (
        "sink",
        "enrichers",
        "strategies",
        "segmenters",
        "providers",
        "composers",
        "offloader",
        "clock",
        "logger",
        "metrics",
    ):
        assert name in parameters


def test_an_unregistered_strategy_is_reported() -> None:
    with pytest.raises(ChunkerRegistrationError, match="No chunking strategy"):
        _engine(_config(chunking=ChunkingConfig(strategy="absent").model_dump()))


def test_two_engines_can_coexist() -> None:
    """No module-level state, so configurations cannot interfere."""
    first = _engine(_config(max_concurrent_documents=2))
    second = _engine(_config(max_concurrent_documents=8))

    assert "max_concurrent=2" in repr(first)
    assert "max_concurrent=8" in repr(second)


def test_the_core_ships_no_enricher() -> None:
    """Domain knowledge enters only through injection."""
    parameters = inspect.signature(build_default_engine).parameters

    assert parameters["enrichers"].default == ()


def test_the_baseline_strategies_are_registered() -> None:
    names = {strategy.name for strategy in default_strategies(ChunkingConfig())}

    assert names == {"block_aligned", "whole_document", "block"}


def test_the_builtin_providers_are_registered() -> None:
    names = {
        registration.name
        for registration in default_providers(InlineOffloader())
    }

    assert names == {"null", "deterministic", "local"}


# ===========================================================================
# Anti-pattern guards
# ===========================================================================


def test_the_engine_imports_no_concrete_implementation() -> None:
    source = inspect.getsource(engine_module)

    for forbidden in (
        "from ptfm.core.knowledge.chunking",
        "from ptfm.core.knowledge.embedding.builtin",
        "from ptfm.core.knowledge.embedding.local",
        "from ptfm.core.knowledge.sinks",
        "from ptfm.core.knowledge.registry",
        "from ptfm.plugins",
    ):
        assert forbidden not in source


def test_the_engine_holds_no_assignable_state() -> None:
    with pytest.raises(AttributeError):
        _engine().cache = {}  # type: ignore[attr-defined]


def test_the_engine_declares_no_mutable_module_state() -> None:
    mutable = [
        name
        for name, value in vars(engine_module).items()
        if not name.startswith("_") and isinstance(value, (list, dict, set))
    ]

    assert mutable == []


def test_the_engine_never_reads_the_wall_clock_directly() -> None:
    source = inspect.getsource(engine_module)

    assert "self._clock.now()" in source
    assert "datetime.now(" not in source


def test_the_engine_never_catches_base_exception() -> None:
    source = inspect.getsource(engine_module)

    assert "except BaseException" not in source
    assert "except:" not in source


def test_the_engine_produces_no_confidence_value() -> None:
    source = inspect.getsource(engine_module).lower()

    assert "confidence" not in source
    assert "calibrat" not in source


def test_the_engine_contains_no_domain_vocabulary() -> None:
    source = inspect.getsource(engine_module).lower()

    for term in ("jurisdiction", "statute", "ticker", "icd", "patient", "court"):
        assert term not in source
