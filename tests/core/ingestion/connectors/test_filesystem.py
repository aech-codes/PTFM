"""
Unit tests for the local filesystem source connector.

File:
    tests/core/ingestion/connectors/test_filesystem.py
"""

from __future__ import annotations

import inspect
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from ptfm.core.ingestion.config import (
    KIB,
    AcquisitionConfig,
    IngestionConfig,
)
from ptfm.core.ingestion.connectors import filesystem as filesystem_module
from ptfm.core.ingestion.connectors.filesystem import LocalFilesystemConnector
from ptfm.core.ingestion.exceptions import (
    DocumentTooLargeError,
    SourceAuthenticationError,
    SourceError,
    SourceNotFoundError,
)
from ptfm.core.ingestion.interfaces import SourceConnector
from ptfm.core.ingestion.models import SourceRecord, SourceSpec
from ptfm.core.ingestion.utils.hashing import Blake2bHashProvider

NOW = datetime(2026, 7, 30, tzinfo=UTC)

WINDOWS = sys.platform.startswith("win")
posix_only = pytest.mark.skipif(WINDOWS, reason="POSIX permission semantics")


# ---------------------------------------------------------------------------
# Doubles and builders
# ---------------------------------------------------------------------------


class _FixedClock:
    def now(self) -> datetime:
        return NOW


class _CountingOffloader:
    """Inline offloader that records every blocking call it received."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    @property
    def limit(self) -> int:
        return 1

    async def run(self, func: Any, *args: Any, **kwargs: Any) -> Any:
        self.calls.append(getattr(func, "__name__", type(func).__name__))
        return func(*args, **kwargs)


def _config(**acquisition: Any) -> IngestionConfig:
    if acquisition:
        return IngestionConfig(
            _env_file=None, acquisition=AcquisitionConfig(**acquisition)
        )
    return IngestionConfig(_env_file=None)


def _connector(
    config: IngestionConfig | None = None,
    *,
    offloader: Any = None,
    **options: Any,
) -> LocalFilesystemConnector:
    return LocalFilesystemConnector(
        config or _config(),
        clock=_FixedClock(),
        offloader=offloader,
        **options,
    )


def _spec(root: Path, **fields: Any) -> SourceSpec:
    return SourceSpec(root_uri=root.as_uri(), **fields)


async def _discover(
    connector: LocalFilesystemConnector, spec: SourceSpec
) -> list[SourceRecord]:
    return [record async for record in connector.discover(spec)]


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    """A small tree with nesting, mixed extensions, and an empty directory."""
    (tmp_path / "alpha.txt").write_bytes(b"alpha body")
    (tmp_path / "beta.pdf").write_bytes(b"%PDF-1.7 beta")
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "gamma.txt").write_bytes(b"gamma body")
    (nested / "delta.tmp").write_bytes(b"scratch")
    (tmp_path / "empty").mkdir()
    return tmp_path


# ---------------------------------------------------------------------------
# Contract conformance
# ---------------------------------------------------------------------------


def test_connector_satisfies_the_source_connector_contract() -> None:
    connector = _connector()

    assert isinstance(connector, SourceConnector)
    assert connector.scheme == "file"
    assert connector.name == "LocalFilesystemConnector"


def test_discover_is_an_async_generator_not_a_coroutine() -> None:
    assert inspect.isasyncgenfunction(LocalFilesystemConnector.discover)
    assert not inspect.iscoroutinefunction(LocalFilesystemConnector.discover)


@pytest.mark.parametrize("method", ["fetch", "health_check", "close"])
def test_lifecycle_methods_are_coroutines(method: str) -> None:
    assert inspect.iscoroutinefunction(getattr(LocalFilesystemConnector, method))


def test_connector_satisfies_the_factory_signature() -> None:
    assert isinstance(LocalFilesystemConnector(_config()), SourceConnector)


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


async def test_discovery_finds_every_file(corpus: Path) -> None:
    records = await _discover(_connector(), _spec(corpus))

    assert sorted(r.record_id for r in records) == [
        "alpha.txt",
        "beta.pdf",
        "nested/delta.tmp",
        "nested/gamma.txt",
    ]


async def test_discovery_order_is_deterministic(corpus: Path) -> None:
    connector = _connector()
    spec = _spec(corpus)

    first = [r.record_id for r in await _discover(connector, spec)]
    second = [r.record_id for r in await _discover(connector, spec)]

    assert first == second == sorted(first)


async def test_non_recursive_discovery_stays_at_the_root(corpus: Path) -> None:
    records = await _discover(_connector(), _spec(corpus, recursive=False))

    assert sorted(r.record_id for r in records) == ["alpha.txt", "beta.pdf"]


async def test_selection_filters_are_applied(corpus: Path) -> None:
    records = await _discover(
        _connector(), _spec(corpus, exclude_patterns=("*.tmp",))
    )

    assert "nested/delta.tmp" not in [r.record_id for r in records]


async def test_include_patterns_are_applied(corpus: Path) -> None:
    records = await _discover(
        _connector(), _spec(corpus, include_patterns=("*.txt",))
    )

    assert sorted(r.record_id for r in records) == [
        "alpha.txt",
        "nested/gamma.txt",
    ]


async def test_empty_directories_yield_nothing(tmp_path: Path) -> None:
    assert await _discover(_connector(), _spec(tmp_path)) == []


async def test_records_carry_complete_provenance(corpus: Path) -> None:
    records = await _discover(_connector(), _spec(corpus))
    record = next(r for r in records if r.record_id == "alpha.txt")

    assert record.source_uri == (corpus / "alpha.txt").as_uri()
    assert record.scheme == "file"
    assert record.connector_name == "LocalFilesystemConnector"
    assert record.filename == "alpha.txt"
    assert record.size_bytes == 10
    assert record.discovered_at == NOW


async def test_no_media_type_is_declared(corpus: Path) -> None:
    """A filesystem asserts nothing; guessing would duplicate detection."""
    records = await _discover(_connector(), _spec(corpus))

    assert all(r.declared_media_type is None for r in records)


# ---------------------------------------------------------------------------
# Version tags
# ---------------------------------------------------------------------------


async def test_version_tag_is_stable_for_an_unchanged_file(corpus: Path) -> None:
    connector = _connector()
    spec = _spec(corpus)

    first = {r.record_id: r.version_tag for r in await _discover(connector, spec)}
    second = {r.record_id: r.version_tag for r in await _discover(connector, spec)}

    assert first == second
    assert all(tag is not None for tag in first.values())


async def test_version_tag_changes_when_content_changes(corpus: Path) -> None:
    """The mechanism that lets an unchanged corpus be skipped before fetch."""
    connector = _connector()
    spec = _spec(corpus)
    before = {r.record_id: r.version_tag for r in await _discover(connector, spec)}

    target = corpus / "alpha.txt"
    target.write_bytes(b"substantially rewritten body")
    os.utime(target, ns=(0, 1))

    after = {r.record_id: r.version_tag for r in await _discover(connector, spec)}

    assert after["alpha.txt"] != before["alpha.txt"]
    assert after["beta.pdf"] == before["beta.pdf"]


async def test_version_tag_combines_time_and_size(corpus: Path) -> None:
    record = next(
        r
        for r in await _discover(_connector(), _spec(corpus))
        if r.record_id == "alpha.txt"
    )

    assert record.version_tag is not None
    assert record.version_tag.endswith("-10")


# ---------------------------------------------------------------------------
# Root validation
# ---------------------------------------------------------------------------


async def test_missing_root_is_reported(tmp_path: Path) -> None:
    absent = tmp_path / "absent"

    with pytest.raises(SourceNotFoundError):
        await _discover(_connector(), SourceSpec(root_uri=absent.as_uri()))


async def test_root_that_is_a_file_is_reported(tmp_path: Path) -> None:
    target = tmp_path / "a.txt"
    target.write_bytes(b"data")

    with pytest.raises(SourceError, match="not a directory"):
        await _discover(_connector(), SourceSpec(root_uri=target.as_uri()))


async def test_remote_host_uris_are_refused() -> None:
    with pytest.raises(SourceError, match="names host"):
        await _discover(
            _connector(), SourceSpec(root_uri="file://fileserver/share")
        )


# ---------------------------------------------------------------------------
# Partial failure
# ---------------------------------------------------------------------------


@posix_only
async def test_unreadable_subdirectory_is_skipped_not_fatal(
    corpus: Path,
) -> None:
    """One restrictive directory must not discard a whole run."""
    restricted = corpus / "restricted"
    restricted.mkdir()
    (restricted / "hidden.txt").write_bytes(b"unreachable")
    restricted.chmod(0o000)

    connector = _connector()
    try:
        records = await _discover(connector, _spec(corpus))
    finally:
        restricted.chmod(0o755)

    assert "alpha.txt" in [r.record_id for r in records]
    assert [Path(s.path).name for s in connector.skipped_paths] == ["restricted"]
    assert "permission" in connector.skipped_paths[0].reason


@posix_only
async def test_unreadable_root_is_fatal(tmp_path: Path) -> None:
    root = tmp_path / "locked"
    root.mkdir()
    root.chmod(0o000)

    try:
        with pytest.raises(SourceAuthenticationError):
            await _discover(_connector(), _spec(root))
    finally:
        root.chmod(0o755)


async def test_no_omissions_are_recorded_for_a_clean_tree(corpus: Path) -> None:
    connector = _connector()

    await _discover(connector, _spec(corpus))

    assert connector.skipped_paths == ()


# ---------------------------------------------------------------------------
# Symbolic links
# ---------------------------------------------------------------------------


@posix_only
async def test_symlinks_are_omitted_by_default(corpus: Path) -> None:
    (corpus / "link.txt").symlink_to(corpus / "alpha.txt")

    records = await _discover(_connector(), _spec(corpus))

    assert "link.txt" not in [r.record_id for r in records]


@posix_only
async def test_symlinks_can_be_followed_explicitly(corpus: Path) -> None:
    (corpus / "link.txt").symlink_to(corpus / "alpha.txt")

    records = await _discover(
        _connector(follow_symlinks=True), _spec(corpus)
    )

    assert "link.txt" in [r.record_id for r in records]


@posix_only
async def test_a_directory_symlink_cycle_does_not_hang(corpus: Path) -> None:
    """The default exists precisely to make this terminate."""
    (corpus / "nested" / "loop").symlink_to(corpus)

    records = await _discover(_connector(), _spec(corpus))

    assert len(records) == 4


# ---------------------------------------------------------------------------
# Fetching
# ---------------------------------------------------------------------------


async def test_fetch_reads_the_file(corpus: Path) -> None:
    connector = _connector()
    record = next(
        r for r in await _discover(connector, _spec(corpus)) if r.record_id == "alpha.txt"
    )

    blob = await connector.fetch(record)

    assert blob.read() == b"alpha body"
    assert blob.size_bytes == 10


async def test_digest_matches_the_shared_provider(tmp_path: Path) -> None:
    payload = b"the quick brown fox jumps over the lazy dog"
    (tmp_path / "a.txt").write_bytes(payload)
    connector = _connector()
    record = (await _discover(connector, _spec(tmp_path)))[0]

    blob = await connector.fetch(record)

    assert blob.content_hash == Blake2bHashProvider().hash_bytes(payload)
    assert blob.hash_algorithm == "blake2b-256"


async def test_large_file_is_read_in_chunks_and_spooled(tmp_path: Path) -> None:
    payload = b"z" * 4096
    (tmp_path / "big.bin").write_bytes(payload)
    config = _config(
        spool_threshold_bytes=256, read_chunk_bytes=64, max_document_bytes=1 * KIB * 64
    )
    connector = _connector(config)
    record = (await _discover(connector, _spec(tmp_path)))[0]

    blob = await connector.fetch(record)

    assert blob.is_spooled is True
    assert blob.read() == payload
    assert blob.content_hash == Blake2bHashProvider().hash_bytes(payload)
    blob.close()


async def test_oversized_file_is_rejected_during_transfer(tmp_path: Path) -> None:
    (tmp_path / "big.bin").write_bytes(b"x" * 4096)
    config = _config(
        max_document_bytes=512, spool_threshold_bytes=256, read_chunk_bytes=64
    )
    connector = _connector(config)
    record = (await _discover(connector, _spec(tmp_path)))[0]

    with pytest.raises(DocumentTooLargeError) as excinfo:
        await connector.fetch(record)

    assert excinfo.value.context["detected_during"] == "transfer"


async def test_deleted_file_is_reported(tmp_path: Path) -> None:
    target = tmp_path / "a.txt"
    target.write_bytes(b"data")
    connector = _connector()
    record = (await _discover(connector, _spec(tmp_path)))[0]

    target.unlink()

    with pytest.raises(SourceNotFoundError) as excinfo:
        await connector.fetch(record)

    assert excinfo.value.retryable is False


@posix_only
async def test_unreadable_file_is_an_authentication_failure(
    tmp_path: Path,
) -> None:
    target = tmp_path / "a.txt"
    target.write_bytes(b"data")
    connector = _connector()
    record = (await _discover(connector, _spec(tmp_path)))[0]
    target.chmod(0o000)

    try:
        with pytest.raises(SourceAuthenticationError) as excinfo:
            await connector.fetch(record)
    finally:
        target.chmod(0o644)

    assert excinfo.value.retryable is False


async def test_empty_file_produces_a_zero_length_payload(tmp_path: Path) -> None:
    """Rejecting an empty payload is the acquisition stage's decision."""
    (tmp_path / "empty.txt").write_bytes(b"")
    connector = _connector()
    record = (await _discover(connector, _spec(tmp_path)))[0]

    blob = await connector.fetch(record)

    assert blob.size_bytes == 0


# ---------------------------------------------------------------------------
# Async safety
# ---------------------------------------------------------------------------


async def test_every_blocking_operation_is_offloaded(corpus: Path) -> None:
    offloader = _CountingOffloader()
    connector = _connector(offloader=offloader)

    records = await _discover(connector, _spec(corpus))
    await connector.fetch(records[0])

    assert "_probe_directory" in offloader.calls
    assert offloader.calls.count("_scan_directory") >= 2
    assert "_read_file" in offloader.calls


async def test_directories_are_scanned_one_at_a_time(corpus: Path) -> None:
    """Walking the tree in a single call would block a thread throughout and
    would prevent discovery from streaming."""
    offloader = _CountingOffloader()

    await _discover(_connector(offloader=offloader), _spec(corpus))

    assert offloader.calls.count("_scan_directory") == 3


def test_connector_performs_no_unoffloaded_filesystem_access() -> None:
    source = inspect.getsource(LocalFilesystemConnector)

    assert ".open(" not in source.replace("path.open(\"rb\")", "")
    assert "os.scandir" not in source


# ---------------------------------------------------------------------------
# Containment
# ---------------------------------------------------------------------------


async def test_paths_outside_the_permitted_roots_are_refused(
    tmp_path: Path,
) -> None:
    permitted = tmp_path / "permitted"
    permitted.mkdir()
    (permitted / "a.txt").write_bytes(b"data")
    outside = tmp_path / "outside.txt"
    outside.write_bytes(b"secret")

    connector = _connector(allowed_roots=[permitted])
    record = (await _discover(connector, _spec(permitted)))[0]
    forged = record.model_copy(update={"source_uri": outside.as_uri()})

    with pytest.raises(SourceError, match="outside every permitted root"):
        await connector.fetch(forged)


async def test_paths_within_the_permitted_roots_are_read(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_bytes(b"data")
    connector = _connector(allowed_roots=[tmp_path])
    record = (await _discover(connector, _spec(tmp_path)))[0]

    assert (await connector.fetch(record)).read() == b"data"


async def test_containment_is_not_enforced_when_unconfigured(
    tmp_path: Path,
) -> None:
    (tmp_path / "a.txt").write_bytes(b"data")
    connector = _connector()
    record = (await _discover(connector, _spec(tmp_path)))[0]

    assert (await connector.fetch(record)).read() == b"data"


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


async def test_health_check_reports_operational_without_roots() -> None:
    assert await _connector().health_check() is True


async def test_health_check_verifies_configured_roots(tmp_path: Path) -> None:
    healthy = _connector(allowed_roots=[tmp_path])
    unhealthy = _connector(allowed_roots=[tmp_path / "absent"])

    assert await healthy.health_check() is True
    assert await unhealthy.health_check() is False


async def test_close_is_idempotent_and_marks_unhealthy(tmp_path: Path) -> None:
    connector = _connector()

    await connector.close()
    await connector.close()

    assert await connector.health_check() is False


async def test_use_after_close_is_a_permanent_failure(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_bytes(b"data")
    connector = _connector()
    record = (await _discover(connector, _spec(tmp_path)))[0]
    await connector.close()

    with pytest.raises(SourceError) as excinfo:
        await connector.fetch(record)

    assert excinfo.value.retryable is False


# ---------------------------------------------------------------------------
# URI handling
# ---------------------------------------------------------------------------


def test_paths_round_trip_through_uris(tmp_path: Path) -> None:
    target = tmp_path / "reports" / "q3 final.pdf"

    uri = filesystem_module._uri_from_path(target)

    assert filesystem_module._path_from_uri(uri) == target


def test_foreign_schemes_are_refused() -> None:
    with pytest.raises(SourceError, match="does not serve"):
        filesystem_module._path_from_uri("s3://bucket/key")


def test_bare_paths_are_accepted_for_convenience(tmp_path: Path) -> None:
    assert filesystem_module._path_from_uri(str(tmp_path)) == tmp_path


# ---------------------------------------------------------------------------
# Anti-pattern guards
# ---------------------------------------------------------------------------


def test_connector_declares_no_hardcoded_limits() -> None:
    source = inspect.getsource(LocalFilesystemConnector)

    assert "self._config.acquisition.read_chunk_bytes" in source
    assert "> 0.8" not in source


def test_connector_does_not_reimplement_shared_machinery() -> None:
    source = inspect.getsource(filesystem_module)

    assert "PayloadBuilder" in source
    assert "matches_filters" in source
    assert "hashlib" not in source
    assert "tempfile" not in source


def test_connector_never_catches_base_exception() -> None:
    source = inspect.getsource(filesystem_module)

    assert "except BaseException" not in source
    assert "except:" not in source


def test_module_declares_no_mutable_state() -> None:
    mutable = [
        name
        for name, value in vars(filesystem_module).items()
        if not name.startswith("__") and isinstance(value, (list, dict, set))
    ]
    assert mutable == []


def test_connector_holds_no_assignable_state() -> None:
    with pytest.raises(AttributeError):
        _connector().cache = {}  # type: ignore[attr-defined]


def test_connector_contains_no_domain_vocabulary() -> None:
    source = inspect.getsource(filesystem_module).lower()
    for term in ("jurisdiction", "statute", "ticker", "icd", "patient", "court"):
        assert term not in source
