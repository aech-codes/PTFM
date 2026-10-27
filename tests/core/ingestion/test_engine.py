"""
Unit and integration tests for the ingestion engine and composition root.

File:
    tests/core/ingestion/test_engine.py
"""

from __future__ import annotations

import asyncio
import inspect
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, AsyncIterator, Optional

import pytest

from ptfm.core.ingestion import (
    DocumentFailed,
    DocumentIngested,
    DocumentSkipped,
    IngestionConfig,
    IngestionEngine,
    InMemoryConnector,
    MemoryDocument,
    SkipReason,
    SourceSpec,
    build_default_engine,
    default_connectors,
    default_extractors,
)
from ptfm.core.ingestion import engine as engine_module
from ptfm.core.ingestion.config import (
    DeduplicationConfig,
    ExtractionConfig,
    TelemetryConfig,
)
from ptfm.core.ingestion.exceptions import (
    CorruptDocumentError,
    IngestionConfigurationError,
    IngestionStage,
    SourceError,
    SourceNotFoundError,
    UnsupportedSourceSchemeError,
)
from ptfm.core.ingestion.interfaces import SourceConnector
from ptfm.core.ingestion.models import SourceRecord
from ptfm.core.ingestion.pipeline.deduplication import InMemoryFingerprintStore
from ptfm.core.ingestion.registry import ConnectorRegistration
from ptfm.core.ingestion.utils.concurrency import InlineOffloader
from ptfm.core.schemas import RawDocument

NOW = datetime(2026, 7, 30, tzinfo=UTC)


# ---------------------------------------------------------------------------
# Doubles and builders
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
    """Metrics sink that raises on every call."""

    def increment(self, name: str, value: int = 1, **labels: str) -> None:
        raise ConnectionError("metrics backend unreachable")

    def observe(self, name: str, value: float, **labels: str) -> None:
        raise ConnectionError("metrics backend unreachable")


class _UnhealthyConnector(InMemoryConnector):
    async def health_check(self) -> bool:
        return False


class _RaisingHealthConnector(InMemoryConnector):
    async def health_check(self) -> bool:
        raise RuntimeError("credentials expired")


class _DiscoveryFailsConnector(InMemoryConnector):
    async def discover(self, spec: Any) -> AsyncIterator[SourceRecord]:
        async for record in super().discover(spec):
            yield record
        raise RuntimeError("listing interrupted")


class _FetchFailsConnector(InMemoryConnector):
    """Connector whose fetch fails for nominated identifiers."""

    def __init__(self, *args: Any, failing: frozenset[str] = frozenset(), **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._failing = failing

    async def fetch(self, record: SourceRecord) -> Any:
        if record.record_id in self._failing:
            raise SourceNotFoundError(
                "vanished", source_uri=record.source_uri
            )
        return await super().fetch(record)


class _SlowConnector(InMemoryConnector):
    """Connector recording concurrent fetches, to observe the limit."""

    def __init__(self, *args: Any, delay: float = 0.02, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._delay = delay
        self.active = 0
        self.peak = 0

    async def fetch(self, record: SourceRecord) -> Any:
        self.active += 1
        self.peak = max(self.peak, self.active)
        try:
            await asyncio.sleep(self._delay)
            return await super().fetch(record)
        finally:
            self.active -= 1


def _documents(count: int, *, prefix: str = "doc") -> list[MemoryDocument]:
    return [
        MemoryDocument(
            identifier=f"{prefix}{index}.txt",
            data=f"Document {index} body text.\n\nSecond paragraph.".encode(),
        )
        for index in range(count)
    ]


def _engine(
    documents: list[MemoryDocument],
    *,
    connector_class: type[InMemoryConnector] = InMemoryConnector,
    connector_kwargs: Optional[dict[str, Any]] = None,
    config: Optional[IngestionConfig] = None,
    **options: Any,
) -> tuple[IngestionEngine, list[InMemoryConnector]]:
    """Build an engine over an in-memory corpus, capturing constructed
    connectors so a test can inspect them."""
    built: list[InMemoryConnector] = []
    extra = connector_kwargs or {}

    def _factory(cfg: IngestionConfig) -> SourceConnector:
        connector = connector_class(
            cfg, documents, clock=_FixedClock(), **extra
        )
        built.append(connector)
        return connector

    engine = build_default_engine(
        config or IngestionConfig(_env_file=None),
        connectors=[
            ConnectorRegistration(scheme="memory", factory=_factory)
        ],
        extractors=[*default_extractors()[:1], *default_extractors()[-1:]],
        clock=_FixedClock(),
        offloader=InlineOffloader(),
        discover_entry_points=False,
        **options,
    )
    return engine, built


def _spec(**fields: Any) -> SourceSpec:
    return SourceSpec(root_uri="memory:///", **fields)


async def _collect(engine: IngestionEngine, spec: SourceSpec) -> list[Any]:
    return [outcome async for outcome in engine.ingest(spec)]


# ---------------------------------------------------------------------------
# End to end
# ---------------------------------------------------------------------------


async def test_documents_flow_through_every_stage() -> None:
    engine, _ = _engine(_documents(3))

    outcomes = await _collect(engine, _spec())

    assert len(outcomes) == 3
    assert all(isinstance(o, DocumentIngested) for o in outcomes)
    for outcome in outcomes:
        document = outcome.document
        assert isinstance(document, RawDocument)
        assert document.text
        assert document.blocks
        assert document.content_hash
        assert document.text_hash != document.content_hash


async def test_produced_documents_satisfy_the_canonical_contract() -> None:
    engine, _ = _engine(_documents(1))

    outcome = (await _collect(engine, _spec()))[0]
    document = outcome.document

    assert document.metadata.char_count == len(document.text)
    assert document.source.connector_name == "InMemoryConnector"
    assert document.ingested_at == NOW
    for block in document.leaf_blocks():
        assert document.block_text(block) is not None


async def test_an_empty_source_yields_nothing() -> None:
    engine, _ = _engine([])

    assert await _collect(engine, _spec()) == []


async def test_selection_filters_are_honoured() -> None:
    documents = _documents(2) + [MemoryDocument("skip.tmp", b"scratch")]
    engine, _ = _engine(documents)

    outcomes = await _collect(engine, _spec(exclude_patterns=("*.tmp",)))

    assert len(outcomes) == 2


async def test_the_document_cap_is_honoured() -> None:
    engine, _ = _engine(_documents(10))

    outcomes = await _collect(engine, _spec(max_documents=3))

    assert len(outcomes) == 3


# ---------------------------------------------------------------------------
# Streaming
# ---------------------------------------------------------------------------


async def test_outcomes_stream_rather_than_batch() -> None:
    """A corpus is never materialized, so the first result arrives before the
    last is processed."""
    engine, _ = _engine(_documents(20))

    seen = 0
    async for _ in engine.ingest(_spec()):
        seen += 1
        if seen == 1:
            break

    assert seen == 1


async def test_abandoning_the_iterator_cancels_the_run() -> None:
    engine, connectors = _engine(_documents(50))

    iterator = engine.ingest(_spec()).__aiter__()
    await iterator.__anext__()
    await iterator.aclose()

    assert connectors[0].document_count == 0


async def test_nothing_happens_until_iteration_begins() -> None:
    engine, connectors = _engine(_documents(3))

    engine.ingest(_spec())
    await asyncio.sleep(0)

    assert connectors == []


# ---------------------------------------------------------------------------
# Concurrency
# ---------------------------------------------------------------------------


async def test_the_concurrency_limit_is_respected() -> None:
    """Memory is bounded by the limit rather than by corpus size."""
    config = IngestionConfig(_env_file=None, max_concurrent_documents=3)
    engine, connectors = _engine(
        _documents(15), connector_class=_SlowConnector, config=config
    )

    await _collect(engine, _spec())

    assert connectors[0].peak <= 3
    assert connectors[0].peak > 1


async def test_a_limit_of_one_serializes_processing() -> None:
    config = IngestionConfig(_env_file=None, max_concurrent_documents=1)
    engine, connectors = _engine(
        _documents(5), connector_class=_SlowConnector, config=config
    )

    await _collect(engine, _spec())

    assert connectors[0].peak == 1


async def test_every_document_is_processed_under_concurrency() -> None:
    config = IngestionConfig(_env_file=None, max_concurrent_documents=4)
    engine, _ = _engine(_documents(23), config=config)

    outcomes = await _collect(engine, _spec())

    assert len(outcomes) == 23
    identifiers = {o.document.source.source_uri for o in outcomes}
    assert len(identifiers) == 23


# ---------------------------------------------------------------------------
# Fault isolation
# ---------------------------------------------------------------------------


async def test_a_failing_document_does_not_abort_the_run() -> None:
    """A corrupt file at position forty thousand must not discard the
    preceding work."""
    engine, _ = _engine(
        _documents(5),
        connector_class=_FetchFailsConnector,
        connector_kwargs={"failing": frozenset({"doc2.txt"})},
    )

    outcomes = await _collect(engine, _spec())

    assert len(outcomes) == 5
    assert sum(isinstance(o, DocumentIngested) for o in outcomes) == 4
    assert sum(isinstance(o, DocumentFailed) for o in outcomes) == 1


async def test_a_failure_is_recorded_with_its_stage_and_code() -> None:
    engine, _ = _engine(
        _documents(2),
        connector_class=_FetchFailsConnector,
        connector_kwargs={"failing": frozenset({"doc0.txt"})},
    )

    outcomes = await _collect(engine, _spec())
    failure = next(o for o in outcomes if isinstance(o, DocumentFailed)).failure

    assert failure.stage is IngestionStage.ACQUISITION
    assert failure.error_code == "SOURCE_NOT_FOUND"
    assert failure.source_uri.endswith("doc0.txt")
    assert failure.failed_at == NOW


async def test_fail_fast_aborts_on_the_first_failure() -> None:
    config = IngestionConfig(
        _env_file=None, fail_fast=True, max_concurrent_documents=1
    )
    engine, _ = _engine(
        _documents(5),
        connector_class=_FetchFailsConnector,
        connector_kwargs={"failing": frozenset({"doc0.txt"})},
        config=config,
    )

    with pytest.raises(IngestionConfigurationError, match="fail_fast"):
        await _collect(engine, _spec())


# ---------------------------------------------------------------------------
# Failure guard
# ---------------------------------------------------------------------------


async def test_a_systemically_bad_corpus_aborts_the_run() -> None:
    config = IngestionConfig(
        _env_file=None,
        failure_ratio_threshold=0.5,
        failure_ratio_min_samples=4,
        max_concurrent_documents=1,
    )
    failing = frozenset(f"doc{index}.txt" for index in range(10))
    engine, _ = _engine(
        _documents(10),
        connector_class=_FetchFailsConnector,
        connector_kwargs={"failing": failing},
        config=config,
    )

    with pytest.raises(IngestionConfigurationError) as excinfo:
        await _collect(engine, _spec())

    assert excinfo.value.context["failure_ratio"] > 0.5


async def test_the_guard_requires_a_minimum_sample() -> None:
    """Without it, a run would abort when the first of one document failed."""
    config = IngestionConfig(
        _env_file=None,
        failure_ratio_threshold=0.5,
        failure_ratio_min_samples=20,
        max_concurrent_documents=1,
    )
    engine, _ = _engine(
        _documents(3),
        connector_class=_FetchFailsConnector,
        connector_kwargs={"failing": frozenset({"doc0.txt", "doc1.txt", "doc2.txt"})},
        config=config,
    )

    outcomes = await _collect(engine, _spec())

    assert len(outcomes) == 3
    assert all(isinstance(o, DocumentFailed) for o in outcomes)


async def test_an_acceptable_failure_rate_does_not_abort() -> None:
    config = IngestionConfig(
        _env_file=None,
        failure_ratio_threshold=0.5,
        failure_ratio_min_samples=2,
        max_concurrent_documents=1,
    )
    engine, _ = _engine(
        _documents(10),
        connector_class=_FetchFailsConnector,
        connector_kwargs={"failing": frozenset({"doc0.txt"})},
        config=config,
    )

    outcomes = await _collect(engine, _spec())

    assert len(outcomes) == 10


# ---------------------------------------------------------------------------
# Run-level failures
# ---------------------------------------------------------------------------


async def test_an_unregistered_scheme_raises_immediately() -> None:
    engine, _ = _engine(_documents(1))

    with pytest.raises(UnsupportedSourceSchemeError):
        await _collect(engine, SourceSpec(root_uri="s3://bucket/prefix"))


async def test_an_unhealthy_connector_aborts_before_discovery() -> None:
    """A misconfiguration must fail immediately, not present as an empty
    corpus."""
    engine, _ = _engine(_documents(3), connector_class=_UnhealthyConnector)

    with pytest.raises(IngestionConfigurationError, match="not operational"):
        await _collect(engine, _spec())


async def test_a_raising_health_check_is_converted() -> None:
    engine, _ = _engine(_documents(1), connector_class=_RaisingHealthConnector)

    with pytest.raises(SourceError) as excinfo:
        await _collect(engine, _spec())

    assert isinstance(excinfo.value.__cause__, RuntimeError)


async def test_a_discovery_failure_propagates() -> None:
    engine, _ = _engine(_documents(2), connector_class=_DiscoveryFailsConnector)

    with pytest.raises(SourceError):
        await _collect(engine, _spec())


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------


async def test_identical_content_is_skipped_within_a_run() -> None:
    """Duplication is a skip, not a failure."""
    payload = b"Identical body text."
    documents = [
        MemoryDocument("a.txt", payload),
        MemoryDocument("b.txt", payload),
    ]
    config = IngestionConfig(_env_file=None, max_concurrent_documents=1)
    engine, _ = _engine(documents, config=config)

    outcomes = await _collect(engine, _spec())

    assert sum(isinstance(o, DocumentIngested) for o in outcomes) == 1
    skipped = [o for o in outcomes if isinstance(o, DocumentSkipped)]
    assert len(skipped) == 1
    assert skipped[0].reason is SkipReason.DUPLICATE_CONTENT


async def test_a_shared_store_skips_across_runs() -> None:
    store = InMemoryFingerprintStore()
    documents = _documents(3)
    engine, _ = _engine(documents, fingerprint_store=store)

    first = await _collect(engine, _spec())
    second = await _collect(engine, _spec())

    assert all(isinstance(o, DocumentIngested) for o in first)
    assert all(isinstance(o, DocumentSkipped) for o in second)


async def test_deduplication_can_be_disabled() -> None:
    payload = b"Identical body text."
    documents = [
        MemoryDocument("a.txt", payload),
        MemoryDocument("b.txt", payload),
    ]
    config = IngestionConfig(
        _env_file=None, deduplication=DeduplicationConfig(enabled=False)
    )
    engine, _ = _engine(documents, config=config)

    outcomes = await _collect(engine, _spec())

    assert all(isinstance(o, DocumentIngested) for o in outcomes)


async def test_identical_bytes_yield_identical_identifiers() -> None:
    """Idempotency: re-ingesting the same corpus must not duplicate it."""
    documents = _documents(2)
    first_engine, _ = _engine(documents)
    second_engine, _ = _engine(documents)

    first = await _collect(first_engine, _spec())
    second = await _collect(second_engine, _spec())

    assert [o.document.document_id for o in first] == [
        o.document.document_id for o in second
    ]


# ---------------------------------------------------------------------------
# Resource safety
# ---------------------------------------------------------------------------


async def test_payloads_are_released_on_success(tmp_path: Path) -> None:
    config = IngestionConfig(_env_file=None)
    config = config.with_overrides(
        acquisition={
            **config.acquisition.model_dump(),
            "spool_threshold_bytes": 64,
            "read_chunk_bytes": 32,
            "spool_directory": str(tmp_path),
        }
    )
    documents = [MemoryDocument("big.txt", b"x" * 4096)]
    engine, _ = _engine(documents, config=config)

    await _collect(engine, _spec())

    assert list(tmp_path.iterdir()) == []


async def test_payloads_are_released_on_failure(tmp_path: Path) -> None:
    """Leaking a temporary file per failure stays invisible until the disk
    fills."""
    config = IngestionConfig(_env_file=None)
    config = config.with_overrides(
        acquisition={
            **config.acquisition.model_dump(),
            "spool_threshold_bytes": 64,
            "read_chunk_bytes": 32,
            "spool_directory": str(tmp_path),
            "max_document_bytes": 1024,
        }
    )
    documents = [MemoryDocument("big.txt", b"x" * 4096)]
    engine, _ = _engine(documents, config=config)

    outcomes = await _collect(engine, _spec())

    assert all(isinstance(o, DocumentFailed) for o in outcomes)
    assert list(tmp_path.iterdir()) == []


async def test_the_connector_is_closed_on_every_path() -> None:
    engine, connectors = _engine(_documents(2))

    await _collect(engine, _spec())

    assert connectors[0].document_count == 0


async def test_the_connector_is_closed_when_the_run_aborts() -> None:
    engine, connectors = _engine(
        _documents(2), connector_class=_DiscoveryFailsConnector
    )

    with pytest.raises(SourceError):
        await _collect(engine, _spec())

    assert connectors[0].document_count == 0


async def test_cancellation_propagates_and_releases_resources() -> None:
    engine, connectors = _engine(
        _documents(50), connector_class=_SlowConnector
    )

    async def _run() -> None:
        async for _ in engine.ingest(_spec()):
            pass

    task = asyncio.create_task(_run())
    await asyncio.sleep(0.01)
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert connectors[0].document_count == 0


# ---------------------------------------------------------------------------
# Telemetry
# ---------------------------------------------------------------------------


async def test_metrics_are_emitted_for_every_outcome() -> None:
    metrics = _RecordingMetrics()
    engine, _ = _engine(
        _documents(3),
        connector_class=_FetchFailsConnector,
        connector_kwargs={"failing": frozenset({"doc0.txt"})},
        metrics=metrics,
    )

    await _collect(engine, _spec())

    names = metrics.names()
    assert names.count("ptfm.ingestion.documents.discovered") == 3
    assert names.count("ptfm.ingestion.documents.ingested") == 2
    assert names.count("ptfm.ingestion.documents.failed") == 1


async def test_a_failing_metrics_sink_cannot_fail_a_run() -> None:
    """An observability outage must not destroy work that succeeded."""
    engine, _ = _engine(_documents(3), metrics=_FailingMetrics())

    outcomes = await _collect(engine, _spec())

    assert len(outcomes) == 3
    assert all(isinstance(o, DocumentIngested) for o in outcomes)


async def test_metrics_can_be_disabled() -> None:
    metrics = _RecordingMetrics()
    config = IngestionConfig(
        _env_file=None, telemetry=TelemetryConfig(emit_metrics=False)
    )
    engine, _ = _engine(_documents(2), config=config, metrics=metrics)

    await _collect(engine, _spec())

    assert metrics.counters == []


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


async def test_a_report_aggregates_the_run() -> None:
    engine, _ = _engine(
        _documents(5),
        connector_class=_FetchFailsConnector,
        connector_kwargs={"failing": frozenset({"doc0.txt"})},
    )

    report = await engine.ingest_all(_spec())

    assert report.discovered_count == 5
    assert report.ingested_count == 4
    assert report.failed_count == 1
    assert report.failures_by_stage[IngestionStage.ACQUISITION.value] == 1
    assert report.total_bytes > 0
    assert report.failure_ratio == pytest.approx(0.2)


async def test_a_report_records_media_types() -> None:
    engine, _ = _engine(_documents(3))

    report = await engine.ingest_all(_spec())

    assert report.documents_by_media_type["text/plain"] == 3


# ---------------------------------------------------------------------------
# Single document
# ---------------------------------------------------------------------------


async def test_a_single_record_can_be_ingested() -> None:
    engine, _ = _engine([])
    connector = InMemoryConnector(
        IngestionConfig(_env_file=None),
        [MemoryDocument("a.txt", b"Body text.")],
        clock=_FixedClock(),
    )
    record = [r async for r in connector.discover(_spec())][0]

    outcome = await engine.ingest_one(record, connector)

    assert isinstance(outcome, DocumentIngested)
    assert outcome.document.text == "Body text."


async def test_a_single_record_failure_is_returned_not_raised() -> None:
    engine, _ = _engine([])
    connector = _FetchFailsConnector(
        IngestionConfig(_env_file=None),
        [MemoryDocument("a.txt", b"Body.")],
        clock=_FixedClock(),
        failing=frozenset({"a.txt"}),
    )
    record = [r async for r in connector.discover(_spec())][0]

    outcome = await engine.ingest_one(record, connector)

    assert isinstance(outcome, DocumentFailed)


# ---------------------------------------------------------------------------
# Composition root
# ---------------------------------------------------------------------------


def test_the_default_engine_builds() -> None:
    engine = build_default_engine(
        IngestionConfig(_env_file=None), discover_entry_points=False
    )

    assert isinstance(engine, IngestionEngine)


def test_the_default_connectors_are_registered() -> None:
    schemes = {registration.scheme for registration in default_connectors()}

    assert schemes == {"file", "memory"}


def test_the_default_connectors_are_independent_per_call() -> None:
    assert default_connectors() is not default_connectors()


def test_the_text_fallback_cannot_displace_a_dedicated_extractor() -> None:
    """Negative priority is what guarantees this."""
    fallbacks = [e for e in default_extractors() if e.priority < 0]

    assert len(fallbacks) == 1
    assert fallbacks[0].supported_media_types == frozenset({"text/*"})


def test_every_component_can_be_substituted() -> None:
    parameters = inspect.signature(build_default_engine).parameters

    for name in (
        "connectors",
        "extractors",
        "fingerprint_store",
        "language_detector",
        "hash_provider",
        "offloader",
        "clock",
        "logger",
        "metrics",
    ):
        assert name in parameters


def test_one_hash_provider_serves_acquisition_and_assembly() -> None:
    """Digests computed under differing algorithms are incomparable."""
    source = inspect.getsource(
        __import__("ptfm.core.ingestion", fromlist=["build_default_engine"])
    )

    assert source.count("resolved_hash") >= 3


def test_two_engines_can_coexist() -> None:
    """No module-level state, so configurations cannot interfere."""
    first = build_default_engine(
        IngestionConfig(_env_file=None, max_concurrent_documents=2),
        discover_entry_points=False,
    )
    second = build_default_engine(
        IngestionConfig(_env_file=None, max_concurrent_documents=8),
        discover_entry_points=False,
    )

    assert "max_concurrent=2" in repr(first)
    assert "max_concurrent=8" in repr(second)


def test_a_missing_optional_dependency_can_be_tolerated() -> None:
    config = IngestionConfig(
        _env_file=None, extraction=ExtractionConfig(strict_availability=False)
    )

    engine = build_default_engine(config, discover_entry_points=False)

    assert isinstance(engine, IngestionEngine)


# ---------------------------------------------------------------------------
# Filesystem integration
# ---------------------------------------------------------------------------


async def test_a_filesystem_corpus_ingests_end_to_end(tmp_path: Path) -> None:
    (tmp_path / "alpha.txt").write_text("Alpha body.\n\nSecond paragraph.")
    (tmp_path / "beta.md").write_text("# Beta\n\nBody text.")
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "gamma.csv").write_text("region,revenue\nNorth,120\n")

    engine = build_default_engine(
        IngestionConfig(_env_file=None),
        clock=_FixedClock(),
        offloader=InlineOffloader(),
        discover_entry_points=False,
    )

    outcomes = [
        outcome
        async for outcome in engine.ingest(
            SourceSpec(root_uri=tmp_path.as_uri())
        )
    ]

    assert len(outcomes) == 3
    assert all(isinstance(o, DocumentIngested) for o in outcomes)
    media_types = {o.document.source.detected_media_type for o in outcomes}
    assert "text/plain" in media_types


async def test_an_unchanged_filesystem_corpus_is_skipped(tmp_path: Path) -> None:
    """Version tags make repeated ingestion inexpensive."""
    (tmp_path / "alpha.txt").write_text("Alpha body text.")
    store = InMemoryFingerprintStore()
    engine = build_default_engine(
        IngestionConfig(_env_file=None),
        fingerprint_store=store,
        clock=_FixedClock(),
        offloader=InlineOffloader(),
        discover_entry_points=False,
    )
    spec = SourceSpec(root_uri=tmp_path.as_uri())

    first = [o async for o in engine.ingest(spec)]
    second = [o async for o in engine.ingest(spec)]

    assert all(isinstance(o, DocumentIngested) for o in first)
    assert all(isinstance(o, DocumentSkipped) for o in second)
    assert second[0].reason is SkipReason.UNCHANGED_VERSION


# ---------------------------------------------------------------------------
# Anti-pattern guards
# ---------------------------------------------------------------------------


def test_the_engine_imports_no_concrete_implementations() -> None:
    """The Core references abstractions only."""
    source = inspect.getsource(engine_module)

    for forbidden in (
        "from ptfm.core.ingestion.connectors",
        "from ptfm.core.ingestion.extractors",
        "from ptfm.plugins",
    ):
        assert forbidden not in source


def test_the_engine_holds_no_assignable_state() -> None:
    engine, _ = _engine([])

    with pytest.raises(AttributeError):
        engine.cache = {}  # type: ignore[attr-defined]


def test_the_engine_module_declares_no_mutable_state() -> None:
    mutable = [
        name
        for name, value in vars(engine_module).items()
        if not name.startswith("_") and isinstance(value, (list, dict, set))
    ]
    assert mutable == []


def test_the_engine_declares_no_hardcoded_limits() -> None:
    source = inspect.getsource(IngestionEngine)

    assert "self._config.max_concurrent_documents" in source
    assert "self._config.failure_ratio_threshold" in source
    assert "> 0.8" not in source


def test_the_engine_never_reads_the_wall_clock_directly() -> None:
    source = inspect.getsource(engine_module)

    assert "self._clock.now()" in source
    assert "datetime.now(" not in source
    assert "utcnow" not in source


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
