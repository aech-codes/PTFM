"""Unit tests for the acquisition stage."""

from __future__ import annotations

import asyncio
import inspect
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, AsyncIterator, Optional, Sequence

import pytest

from ptfm.core.ingestion.config import (
    KIB,
    AcquisitionConfig,
    IngestionConfig,
    RetryConfig,
)
from ptfm.core.ingestion.exceptions import (
    AcquisitionError,
    DocumentTooLargeError,
    EmptyDocumentError,
    IngestionStage,
    SourceAuthenticationError,
    SourceNotFoundError,
    SourceTimeoutError,
    SourceUnavailableError,
)
from ptfm.core.ingestion.interfaces import SourceConnector
from ptfm.core.ingestion.models import (
    AcquiredBlob,
    SkipReason,
    SourceRecord,
    SourceSpec,
)
from ptfm.core.ingestion.pipeline import acquisition as acquisition_module
from ptfm.core.ingestion.pipeline.acquisition import (
    AcquisitionStage,
    PayloadBuilder,
    matches_filters,
)
from ptfm.core.ingestion.utils.hashing import Blake2bHashProvider

NOW = datetime(2026, 7, 30, tzinfo=UTC)


# ---------------------------------------------------------------------------
# Doubles and builders
# ---------------------------------------------------------------------------


class _FixedClock:
    def now(self) -> datetime:
        return NOW


class _ImmediatePolicy:
    """Retry policy with no delays, so tests do not sleep."""

    def __init__(self, max_attempts: int = 3) -> None:
        self._max = max_attempts

    @property
    def max_attempts(self) -> int:
        return self._max

    def delay_for(self, attempt: int) -> float:
        return 0.0


class _Connector(SourceConnector):
    """Connector whose discovery and fetch behaviour is scripted per test."""

    def __init__(
        self,
        *,
        records: Sequence[SourceRecord] = (),
        fetch_results: Sequence[Any] = (),
        discover_error: Optional[BaseException] = None,
        discover_delay: float = 0.0,
        fetch_delay: float = 0.0,
    ) -> None:
        self._records = list(records)
        self._fetch_results = list(fetch_results)
        self._discover_error = discover_error
        self._discover_delay = discover_delay
        self._fetch_delay = fetch_delay
        self.fetch_calls = 0

    @property
    def name(self) -> str:
        return "TestConnector"

    @property
    def scheme(self) -> str:
        return "file"

    async def discover(self, spec: Any) -> AsyncIterator[SourceRecord]:
        for record in self._records:
            if self._discover_delay:
                await asyncio.sleep(self._discover_delay)
            yield record
        if self._discover_error is not None:
            raise self._discover_error

    async def fetch(self, record: SourceRecord) -> Any:
        self.fetch_calls += 1
        if self._fetch_delay:
            await asyncio.sleep(self._fetch_delay)
        index = min(self.fetch_calls - 1, len(self._fetch_results) - 1)
        result = self._fetch_results[index]
        if isinstance(result, BaseException):
            raise result
        return result

    async def health_check(self) -> bool:
        return True

    async def close(self) -> None:
        return None


def _record(
    uri: str = "file:///corpus/a.txt",
    size_bytes: int | None = None,
) -> SourceRecord:
    return SourceRecord(
        record_id="r-1",
        source_uri=uri,
        scheme="file",
        connector_name="TestConnector",
        discovered_at=NOW,
        filename=uri.rsplit("/", 1)[-1],
        size_bytes=size_bytes,
    )


def _blob(
    data: bytes = b"hello",
    *,
    record: SourceRecord | None = None,
    algorithm: str = "blake2b-256",
) -> AcquiredBlob:
    return AcquiredBlob(
        record=record or _record(),
        size_bytes=len(data),
        content_hash="hash-1",
        hash_algorithm=algorithm,
        acquired_at=NOW,
        data=data,
    )


def _stage(
    *,
    max_attempts: int = 3,
    **config: Any,
) -> AcquisitionStage:
    return AcquisitionStage(
        AcquisitionConfig(**config), _ImmediatePolicy(max_attempts)
    )


def _acq_config(**overrides: Any) -> AcquisitionConfig:
    return AcquisitionConfig(**overrides)


# ---------------------------------------------------------------------------
# Selection filtering
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("uri", "include", "exclude", "expected"),
    [
        ("file:///c/a.pdf", (), (), True),
        ("file:///c/a.pdf", ("*.pdf",), (), True),
        ("file:///c/a.txt", ("*.pdf",), (), False),
        ("file:///c/deep/a.pdf", ("*.pdf",), (), True),
        ("file:///c/a.pdf", (), ("*.pdf",), False),
        ("file:///c/a.pdf", ("*.pdf",), ("*.pdf",), False),
        ("file:///c/a.pdf", ("*.pdf", "*.txt"), (), True),
        ("file:///c/tmp/a.pdf", ("*.pdf",), ("*/tmp/*",), False),
    ],
)
def test_filter_semantics(
    uri: str, include: tuple[str, ...], exclude: tuple[str, ...], expected: bool
) -> None:
    spec = SourceSpec(
        root_uri="file:///c", include_patterns=include, exclude_patterns=exclude
    )

    assert matches_filters(uri, spec) is expected


def test_exclusion_always_wins_over_inclusion() -> None:
    spec = SourceSpec(
        root_uri="file:///c",
        include_patterns=("*",),
        exclude_patterns=("*.tmp",),
    )

    assert matches_filters("file:///c/a.txt", spec) is True
    assert matches_filters("file:///c/a.tmp", spec) is False


def test_excluded_record_produces_a_skip_outcome() -> None:
    spec = SourceSpec(root_uri="file:///c", exclude_patterns=("*.tmp",))
    stage = _stage()

    excluded = stage.excluded(_record("file:///c/a.tmp"), spec)
    included = stage.excluded(_record("file:///c/a.txt"), spec)

    assert excluded is not None
    assert excluded.reason is SkipReason.EXCLUDED_BY_FILTER
    assert included is None


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


async def test_discovery_yields_every_matching_record() -> None:
    records = [_record(f"file:///c/{n}.txt") for n in range(3)]
    connector = _Connector(records=records)
    stage = _stage()
    spec = SourceSpec(root_uri="file:///c")

    yielded = [r async for r in stage.discover(connector, spec)]

    assert [r.source_uri for r in yielded] == [r.source_uri for r in records]


async def test_discovery_applies_filters_authoritatively() -> None:
    """A connector that neglects to pre-filter cannot leak excluded documents."""
    connector = _Connector(
        records=[_record("file:///c/a.pdf"), _record("file:///c/b.tmp")]
    )
    stage = _stage()
    spec = SourceSpec(root_uri="file:///c", exclude_patterns=("*.tmp",))

    yielded = [r async for r in stage.discover(connector, spec)]

    assert [r.source_uri for r in yielded] == ["file:///c/a.pdf"]


async def test_document_cap_is_enforced() -> None:
    connector = _Connector(records=[_record(f"file:///c/{n}.txt") for n in range(10)])
    stage = _stage()
    spec = SourceSpec(root_uri="file:///c", max_documents=3)

    yielded = [r async for r in stage.discover(connector, spec)]

    assert len(yielded) == 3


async def test_cap_counts_only_records_that_passed_filters() -> None:
    connector = _Connector(
        records=[
            _record("file:///c/a.tmp"),
            _record("file:///c/b.txt"),
            _record("file:///c/c.txt"),
        ]
    )
    stage = _stage()
    spec = SourceSpec(
        root_uri="file:///c", exclude_patterns=("*.tmp",), max_documents=2
    )

    yielded = [r async for r in stage.discover(connector, spec)]

    assert [r.source_uri for r in yielded] == ["file:///c/b.txt", "file:///c/c.txt"]


async def test_discovery_failure_is_converted() -> None:
    connector = _Connector(
        records=[_record()], discover_error=RuntimeError("listing failed")
    )
    stage = _stage()

    with pytest.raises(acquisition_module.SourceError) as excinfo:
        async for _ in stage.discover(connector, SourceSpec(root_uri="file:///c")):
            pass

    error = excinfo.value
    assert isinstance(error.__cause__, RuntimeError)
    assert error.context["records_yielded"] == 1
    assert error.stage is IngestionStage.ACQUISITION


async def test_typed_discovery_error_passes_through_enriched() -> None:
    connector = _Connector(
        records=[], discover_error=SourceAuthenticationError("403 forbidden")
    )
    stage = _stage()

    with pytest.raises(SourceAuthenticationError) as excinfo:
        async for _ in stage.discover(connector, SourceSpec(root_uri="file:///c")):
            pass

    assert excinfo.value.context["records_yielded"] == 0


async def test_discovery_deadline_bounds_connector_time() -> None:
    connector = _Connector(
        records=[_record(f"file:///c/{n}.txt") for n in range(50)],
        discover_delay=0.05,
    )
    stage = _stage(discovery_timeout_seconds=0.12)

    with pytest.raises(SourceTimeoutError) as excinfo:
        async for _ in stage.discover(connector, SourceSpec(root_uri="file:///c")):
            pass

    assert excinfo.value.retryable is True
    assert excinfo.value.timeout_seconds == 0.12


async def test_slow_consumer_does_not_trigger_a_discovery_timeout() -> None:
    """
    The deadline measures connector time. Including consumer time would make
    a slow downstream engine cause a spurious discovery failure.
    """
    connector = _Connector(records=[_record(f"file:///c/{n}.txt") for n in range(5)])
    stage = _stage(discovery_timeout_seconds=0.2)

    yielded = []
    async for record in stage.discover(connector, SourceSpec(root_uri="file:///c")):
        await asyncio.sleep(0.06)
        yielded.append(record)

    assert len(yielded) == 5


# ---------------------------------------------------------------------------
# Pre-transfer size enforcement
# ---------------------------------------------------------------------------


async def test_oversized_declared_size_is_rejected_without_fetching() -> None:
    """The promise that an oversized document is never downloaded."""
    connector = _Connector(fetch_results=[_blob()])
    stage = _stage(max_document_bytes=1024, spool_threshold_bytes=512, read_chunk_bytes=256)

    with pytest.raises(DocumentTooLargeError) as excinfo:
        await stage.fetch(connector, _record(size_bytes=5000))

    assert connector.fetch_calls == 0
    assert excinfo.value.context["detected_during"] == "discovery"


async def test_record_with_no_declared_size_is_fetched() -> None:
    connector = _Connector(fetch_results=[_blob()])
    stage = _stage()

    assert await stage.fetch(connector, _record(size_bytes=None)) is not None


async def test_declared_size_at_the_limit_is_accepted() -> None:
    connector = _Connector(fetch_results=[_blob(b"x" * 100)])
    stage = _stage(max_document_bytes=100, spool_threshold_bytes=50, read_chunk_bytes=10)

    assert await stage.fetch(connector, _record(size_bytes=100)) is not None


# ---------------------------------------------------------------------------
# Retry
# ---------------------------------------------------------------------------


async def test_transient_failure_is_retried_and_succeeds() -> None:
    connector = _Connector(
        fetch_results=[SourceUnavailableError("connection reset"), _blob()]
    )
    stage = _stage(max_attempts=3)

    assert await stage.fetch(connector, _record()) is not None
    assert connector.fetch_calls == 2


async def test_retry_exhaustion_raises_with_the_attempt_count() -> None:
    connector = _Connector(fetch_results=[SourceUnavailableError("down")])
    stage = _stage(max_attempts=3)

    with pytest.raises(SourceUnavailableError) as excinfo:
        await stage.fetch(connector, _record())

    assert connector.fetch_calls == 3
    assert excinfo.value.context["attempts"] == 3
    assert excinfo.value.context["max_attempts"] == 3


@pytest.mark.parametrize(
    "error",
    [
        SourceAuthenticationError("403"),
        SourceNotFoundError("404"),
        DocumentTooLargeError(size_bytes=10, limit_bytes=5),
    ],
)
async def test_permanent_failures_are_not_retried(error: Exception) -> None:
    """Eligibility comes from the exception, never from a type check here."""
    connector = _Connector(fetch_results=[error])
    stage = _stage(max_attempts=5)

    with pytest.raises(type(error)):
        await stage.fetch(connector, _record())

    assert connector.fetch_calls == 1


async def test_unknown_exception_is_treated_as_permanent() -> None:
    """Assuming transience would spend the budget on a deterministic failure."""
    connector = _Connector(fetch_results=[RuntimeError("unexpected")])
    stage = _stage(max_attempts=5)

    with pytest.raises(acquisition_module.SourceError) as excinfo:
        await stage.fetch(connector, _record())

    assert connector.fetch_calls == 1
    assert excinfo.value.retryable is False
    assert isinstance(excinfo.value.__cause__, RuntimeError)


async def test_retrying_can_be_disabled() -> None:
    connector = _Connector(fetch_results=[SourceUnavailableError("down")])
    stage = _stage(max_attempts=1)

    with pytest.raises(SourceUnavailableError):
        await stage.fetch(connector, _record())

    assert connector.fetch_calls == 1


async def test_configured_schedule_is_consulted() -> None:
    delays: list[int] = []

    class _RecordingPolicy(_ImmediatePolicy):
        def delay_for(self, attempt: int) -> float:
            delays.append(attempt)
            return 0.0

    connector = _Connector(fetch_results=[SourceUnavailableError("down")])
    stage = AcquisitionStage(_acq_config(), _RecordingPolicy(3))

    with pytest.raises(SourceUnavailableError):
        await stage.fetch(connector, _record())

    assert delays == [1, 2, 3]


async def test_fetch_deadline_produces_a_retryable_timeout() -> None:
    connector = _Connector(fetch_results=[_blob()], fetch_delay=1.0)
    stage = _stage(max_attempts=2, fetch_timeout_seconds=0.02)

    with pytest.raises(SourceTimeoutError) as excinfo:
        await stage.fetch(connector, _record())

    assert connector.fetch_calls == 2
    assert excinfo.value.retryable is True


async def test_cancellation_propagates_untouched() -> None:
    connector = _Connector(fetch_results=[_blob()], fetch_delay=5.0)
    stage = _stage(fetch_timeout_seconds=30.0)
    task = asyncio.create_task(stage.fetch(connector, _record()))
    await asyncio.sleep(0)

    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task


# ---------------------------------------------------------------------------
# Post-transfer validation
# ---------------------------------------------------------------------------


async def test_empty_payload_is_rejected() -> None:
    connector = _Connector(fetch_results=[_blob(b"")])
    stage = _stage()

    with pytest.raises(EmptyDocumentError):
        await stage.fetch(connector, _record())


async def test_oversized_actual_payload_is_rejected() -> None:
    connector = _Connector(fetch_results=[_blob(b"x" * 5000)])
    stage = _stage(max_document_bytes=1024, spool_threshold_bytes=512, read_chunk_bytes=256)

    with pytest.raises(DocumentTooLargeError) as excinfo:
        await stage.fetch(connector, _record())

    assert excinfo.value.context["detected_during"] == "validation"


async def test_mismatched_record_is_rejected() -> None:
    connector = _Connector(
        fetch_results=[_blob(record=_record("file:///c/other.txt"))]
    )
    stage = _stage()

    with pytest.raises(AcquisitionError, match="Provenance"):
        await stage.fetch(connector, _record("file:///c/a.txt"))


async def test_unexpected_hash_algorithm_is_rejected() -> None:
    """Incomparable digests would silently disable duplicate detection."""
    connector = _Connector(fetch_results=[_blob(algorithm="sha256")])
    stage = _stage()

    with pytest.raises(AcquisitionError) as excinfo:
        await stage.fetch(connector, _record())

    assert excinfo.value.context["blob_algorithm"] == "sha256"
    assert excinfo.value.context["expected_algorithm"] == "blake2b-256"


async def test_rejected_payload_is_released(tmp_path: Path) -> None:
    """Leaking a temp file per rejection is invisible until the disk fills."""
    spool = tmp_path / "payload.bin"
    spool.write_bytes(b"x" * 5000)
    blob = AcquiredBlob(
        record=_record(),
        size_bytes=5000,
        content_hash="hash-1",
        hash_algorithm="blake2b-256",
        acquired_at=NOW,
        spool_path=spool,
    )
    connector = _Connector(fetch_results=[blob])
    stage = _stage(max_document_bytes=1024, spool_threshold_bytes=512, read_chunk_bytes=256)

    with pytest.raises(DocumentTooLargeError):
        await stage.fetch(connector, _record())

    assert not spool.exists()


async def test_accepted_payload_is_not_closed() -> None:
    connector = _Connector(fetch_results=[_blob(b"body")])
    stage = _stage()

    blob = await stage.fetch(connector, _record())

    assert blob.read() == b"body"


# ---------------------------------------------------------------------------
# PayloadBuilder: in-memory
# ---------------------------------------------------------------------------


def test_small_payload_is_retained_in_memory() -> None:
    config = _acq_config(spool_threshold_bytes=1 * KIB, read_chunk_bytes=256)

    with PayloadBuilder(_record(), config, clock=_FixedClock()) as builder:
        builder.write(b"hello ")
        builder.write(b"world")
        blob = builder.build()

    assert blob.is_spooled is False
    assert blob.read() == b"hello world"
    assert blob.size_bytes == 11
    assert blob.acquired_at == NOW


def test_digest_matches_whole_payload_hashing() -> None:
    payload = b"the quick brown fox"
    config = _acq_config()

    with PayloadBuilder(_record(), config) as builder:
        for start in range(0, len(payload), 4):
            builder.write(payload[start : start + 4])
        blob = builder.build()

    assert blob.content_hash == Blake2bHashProvider().hash_bytes(payload)
    assert blob.hash_algorithm == "blake2b-256"


def test_empty_chunks_are_ignored() -> None:
    with PayloadBuilder(_record(), _acq_config()) as builder:
        builder.write(b"")
        builder.write(b"data")
        builder.write(b"")
        blob = builder.build()

    assert blob.size_bytes == 4


def test_builder_repr_never_leaks_payload_bytes() -> None:
    builder = PayloadBuilder(_record(), _acq_config())
    builder.write(b"SENSITIVE")

    assert "SENSITIVE" not in repr(builder)
    assert "size_bytes=9" in repr(builder)


# ---------------------------------------------------------------------------
# PayloadBuilder: spooling
# ---------------------------------------------------------------------------


def test_payload_spills_beyond_the_threshold(tmp_path: Path) -> None:
    config = _acq_config(
        spool_threshold_bytes=64, read_chunk_bytes=16, spool_directory=str(tmp_path)
    )

    with PayloadBuilder(_record(), config) as builder:
        for _ in range(10):
            builder.write(b"x" * 16)
        blob = builder.build()

    assert blob.is_spooled is True
    assert blob.spool_path is not None
    assert blob.spool_path.parent == tmp_path
    assert blob.read() == b"x" * 160
    blob.close()


def test_data_buffered_before_the_spill_is_preserved(tmp_path: Path) -> None:
    config = _acq_config(
        spool_threshold_bytes=32, read_chunk_bytes=8, spool_directory=str(tmp_path)
    )

    with PayloadBuilder(_record(), config) as builder:
        builder.write(b"A" * 16)
        builder.write(b"B" * 32)
        blob = builder.build()

    assert blob.read() == b"A" * 16 + b"B" * 32
    blob.close()


def test_digest_is_identical_whether_spooled_or_not(tmp_path: Path) -> None:
    payload = b"y" * 200
    memory = _acq_config(spool_threshold_bytes=10 * KIB, read_chunk_bytes=128)
    spooled = _acq_config(
        spool_threshold_bytes=64, read_chunk_bytes=32, spool_directory=str(tmp_path)
    )

    digests = []
    for config in (memory, spooled):
        with PayloadBuilder(_record(), config) as builder:
            builder.write(payload)
            blob = builder.build()
            digests.append(blob.content_hash)
            blob.close()

    assert digests[0] == digests[1]


def test_successful_build_transfers_file_ownership(tmp_path: Path) -> None:
    config = _acq_config(
        spool_threshold_bytes=16, read_chunk_bytes=8, spool_directory=str(tmp_path)
    )

    with PayloadBuilder(_record(), config) as builder:
        builder.write(b"z" * 64)
        blob = builder.build()

    assert blob.spool_path is not None
    assert blob.spool_path.exists()

    blob.close()
    assert not blob.spool_path.exists()


# ---------------------------------------------------------------------------
# PayloadBuilder: streaming size enforcement
# ---------------------------------------------------------------------------


def test_limit_is_enforced_during_transfer() -> None:
    """A source declaring one kilobyte and streaming ten gigabytes is stopped."""
    config = _acq_config(
        max_document_bytes=100, spool_threshold_bytes=50, read_chunk_bytes=25
    )
    builder = PayloadBuilder(_record(), config)

    with pytest.raises(DocumentTooLargeError) as excinfo:
        with builder:
            for _ in range(10):
                builder.write(b"x" * 25)

    assert excinfo.value.limit_bytes == 100
    assert excinfo.value.context["detected_during"] == "transfer"


def test_writing_exactly_to_the_limit_is_permitted() -> None:
    config = _acq_config(
        max_document_bytes=100, spool_threshold_bytes=50, read_chunk_bytes=25
    )

    with PayloadBuilder(_record(), config) as builder:
        builder.write(b"x" * 100)
        blob = builder.build()

    assert blob.size_bytes == 100
    blob.close()


def test_limit_breach_removes_the_temporary_file(tmp_path: Path) -> None:
    config = _acq_config(
        max_document_bytes=100,
        spool_threshold_bytes=32,
        read_chunk_bytes=16,
        spool_directory=str(tmp_path),
    )
    builder = PayloadBuilder(_record(), config)

    with pytest.raises(DocumentTooLargeError):
        with builder:
            for _ in range(20):
                builder.write(b"x" * 16)

    assert list(tmp_path.iterdir()) == []


# ---------------------------------------------------------------------------
# PayloadBuilder: lifecycle
# ---------------------------------------------------------------------------


def test_abort_removes_the_temporary_file(tmp_path: Path) -> None:
    config = _acq_config(
        spool_threshold_bytes=16, read_chunk_bytes=8, spool_directory=str(tmp_path)
    )
    builder = PayloadBuilder(_record(), config)
    builder.write(b"x" * 64)

    builder.abort()
    builder.abort()

    assert list(tmp_path.iterdir()) == []


def test_context_exit_cleans_up_on_failure(tmp_path: Path) -> None:
    config = _acq_config(
        spool_threshold_bytes=16, read_chunk_bytes=8, spool_directory=str(tmp_path)
    )

    class _TransferFailed(Exception):
        pass

    builder = PayloadBuilder(_record(), config)
    with pytest.raises(_TransferFailed):
        with builder:
            builder.write(b"x" * 64)
            raise _TransferFailed

    assert list(tmp_path.iterdir()) == []


def test_write_after_build_is_rejected() -> None:
    builder = PayloadBuilder(_record(), _acq_config())
    builder.write(b"data")
    builder.build()

    with pytest.raises(AcquisitionError, match="already been finalized"):
        builder.write(b"more")


def test_double_build_is_rejected() -> None:
    builder = PayloadBuilder(_record(), _acq_config())
    builder.write(b"data")
    builder.build()

    with pytest.raises(AcquisitionError, match="already been finalized"):
        builder.build()


def test_write_after_abort_is_rejected() -> None:
    builder = PayloadBuilder(_record(), _acq_config())
    builder.abort()

    with pytest.raises(AcquisitionError):
        builder.write(b"data")


def test_unwritable_spool_directory_is_reported() -> None:
    config = _acq_config(
        spool_threshold_bytes=16,
        read_chunk_bytes=8,
        spool_directory="/nonexistent/ptfm/spool",
    )
    builder = PayloadBuilder(_record(), config)

    with pytest.raises(AcquisitionError) as excinfo:
        with builder:
            builder.write(b"x" * 64)

    assert isinstance(excinfo.value.__cause__, OSError)
    assert excinfo.value.context["spool_directory"] == "/nonexistent/ptfm/spool"


def test_builder_produces_a_zero_length_blob_for_no_writes() -> None:
    """Rejecting an empty payload is the stage's decision, not the builder's."""
    with PayloadBuilder(_record(), _acq_config()) as builder:
        blob = builder.build()

    assert blob.size_bytes == 0


# ---------------------------------------------------------------------------
# Composition
# ---------------------------------------------------------------------------


def test_stage_constructs_from_the_full_configuration() -> None:
    config = IngestionConfig(_env_file=None, retry=RetryConfig(max_attempts=7))

    stage = AcquisitionStage.from_config(config)

    assert stage._retry_policy.max_attempts == 7


async def test_builder_output_passes_stage_validation(tmp_path: Path) -> None:
    """End to end: the builder produces what the stage accepts."""
    config = _acq_config(
        spool_threshold_bytes=32, read_chunk_bytes=16, spool_directory=str(tmp_path)
    )
    record = _record()

    with PayloadBuilder(record, config, clock=_FixedClock()) as builder:
        builder.write(b"payload body " * 10)
        blob = builder.build()

    connector = _Connector(fetch_results=[blob])
    stage = AcquisitionStage(config, _ImmediatePolicy())

    validated = await stage.fetch(connector, record)

    assert validated is blob
    validated.close()


# ---------------------------------------------------------------------------
# Anti-pattern guards
# ---------------------------------------------------------------------------


def test_stage_declares_no_hardcoded_limits() -> None:
    source = inspect.getsource(AcquisitionStage)

    assert "self._config.max_document_bytes" in source
    assert "self._config.fetch_timeout_seconds" in source
    assert "self._config.discovery_timeout_seconds" in source
    assert "> 0.8" not in source


def test_builder_declares_no_hardcoded_limits() -> None:
    source = inspect.getsource(PayloadBuilder)

    assert "self._config.max_document_bytes" in source
    assert "self._config.spool_threshold_bytes" in source
    assert "self._config.spool_directory" in source


def test_retry_eligibility_is_never_a_type_check() -> None:
    source = inspect.getsource(AcquisitionStage)

    assert "error.retryable" in source
    assert "isinstance(error" not in source


def test_stage_never_catches_base_exception() -> None:
    source = inspect.getsource(acquisition_module)

    assert "except BaseException" not in source
    assert "except:" not in source


def test_stage_holds_no_assignable_state() -> None:
    with pytest.raises(AttributeError):
        _stage().cache = {}  # type: ignore[attr-defined]


def test_module_declares_no_mutable_state() -> None:
    mutable = [
        name
        for name, value in vars(acquisition_module).items()
        if not name.startswith("__") and isinstance(value, (list, dict, set))
    ]
    assert mutable == []


def test_stage_imports_no_concrete_connectors_or_plugins() -> None:
    source = inspect.getsource(acquisition_module)
    for forbidden in (
        "from ptfm.core.ingestion.connectors",
        "from ptfm.core.ingestion.extractors",
        "from ptfm.plugins",
        "import httpx",
        "import requests",
    ):
        assert forbidden not in source


def test_stage_contains_no_domain_vocabulary() -> None:
    source = inspect.getsource(acquisition_module).lower()
    for term in ("jurisdiction", "statute", "ticker", "icd", "patient", "court"):
        assert term not in source