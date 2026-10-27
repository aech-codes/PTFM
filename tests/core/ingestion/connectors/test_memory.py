"""
Unit tests for the in-memory source connector.

File:
    tests/core/ingestion/connectors/test_memory.py
"""

from __future__ import annotations

import inspect
from datetime import UTC, datetime
from typing import Any

import pytest

from ptfm.core.ingestion.config import (
    KIB,
    AcquisitionConfig,
    IngestionConfig,
)
from ptfm.core.ingestion.connectors import memory as memory_module
from ptfm.core.ingestion.connectors.memory import InMemoryConnector, MemoryDocument
from ptfm.core.ingestion.exceptions import (
    ConnectorRegistrationError,
    DocumentTooLargeError,
    SourceError,
    SourceNotFoundError,
)
from ptfm.core.ingestion.interfaces import SourceConnector
from ptfm.core.ingestion.models import SourceRecord, SourceSpec
from ptfm.core.ingestion.utils.hashing import Blake2bHashProvider

NOW = datetime(2026, 7, 30, tzinfo=UTC)


# ---------------------------------------------------------------------------
# Doubles and builders
# ---------------------------------------------------------------------------


class _FixedClock:
    def now(self) -> datetime:
        return NOW


class _CountingOffloader:
    """Inline offloader that records how often it was used."""

    def __init__(self) -> None:
        self.calls = 0

    @property
    def limit(self) -> int:
        return 1

    async def run(self, func: Any, *args: Any, **kwargs: Any) -> Any:
        self.calls += 1
        return func(*args, **kwargs)


def _config(**acquisition: Any) -> IngestionConfig:
    if acquisition:
        return IngestionConfig(
            _env_file=None, acquisition=AcquisitionConfig(**acquisition)
        )
    return IngestionConfig(_env_file=None)


def _connector(
    *documents: MemoryDocument,
    config: IngestionConfig | None = None,
    offloader: Any = None,
) -> InMemoryConnector:
    return InMemoryConnector(
        config or _config(),
        documents,
        clock=_FixedClock(),
        offloader=offloader,
    )


def _spec(root: str = "memory:///", **fields: Any) -> SourceSpec:
    return SourceSpec(root_uri=root, **fields)


async def _discover(
    connector: InMemoryConnector, spec: SourceSpec | None = None
) -> list[SourceRecord]:
    return [record async for record in connector.discover(spec or _spec())]


# ---------------------------------------------------------------------------
# Contract conformance
# ---------------------------------------------------------------------------


def test_connector_satisfies_the_source_connector_contract() -> None:
    connector = _connector()

    assert isinstance(connector, SourceConnector)
    assert connector.scheme == "memory"
    assert connector.name == "InMemoryConnector"


def test_discover_is_an_async_generator_not_a_coroutine() -> None:
    """The base class rejects the wrong declaration; confirm the right one."""
    assert inspect.isasyncgenfunction(InMemoryConnector.discover)
    assert not inspect.iscoroutinefunction(InMemoryConnector.discover)


@pytest.mark.parametrize("method", ["fetch", "health_check", "close"])
def test_lifecycle_methods_are_coroutines(method: str) -> None:
    assert inspect.iscoroutinefunction(getattr(InMemoryConnector, method))


def test_connector_satisfies_the_factory_signature() -> None:
    """The registry constructs connectors from configuration alone."""
    assert isinstance(InMemoryConnector(_config()), SourceConnector)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_duplicate_identifiers_are_rejected() -> None:
    with pytest.raises(ConnectorRegistrationError, match="Duplicate"):
        _connector(
            MemoryDocument("a.txt", b"first"),
            MemoryDocument("a.txt", b"second"),
        )


@pytest.mark.parametrize("identifier", ["", "   ", "/"])
def test_blank_identifiers_are_rejected(identifier: str) -> None:
    with pytest.raises(ConnectorRegistrationError, match="non-empty"):
        _connector(MemoryDocument(identifier, b"data"))


def test_leading_separators_are_normalized() -> None:
    connector = _connector(MemoryDocument("/a.txt", b"data"))

    assert connector.document_count == 1


def test_construction_from_a_mapping() -> None:
    connector = InMemoryConnector.from_mapping(
        _config(), {"a.txt": b"alpha", "b.txt": b"beta"}, clock=_FixedClock()
    )

    assert connector.document_count == 2


def test_filename_defaults_to_the_final_identifier_segment() -> None:
    assert MemoryDocument("reports/q3.pdf", b"x").resolved_filename == "q3.pdf"
    assert MemoryDocument("q3.pdf", b"x", filename="other.pdf").resolved_filename == (
        "other.pdf"
    )


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


async def test_discovery_yields_every_document_in_order() -> None:
    connector = _connector(
        MemoryDocument("a.txt", b"alpha"),
        MemoryDocument("b.txt", b"beta"),
        MemoryDocument("c.txt", b"gamma"),
    )

    records = await _discover(connector)

    assert [r.record_id for r in records] == ["a.txt", "b.txt", "c.txt"]


async def test_discovery_is_deterministic() -> None:
    connector = _connector(
        MemoryDocument("a.txt", b"alpha"), MemoryDocument("b.txt", b"beta")
    )

    assert await _discover(connector) == await _discover(connector)


async def test_records_carry_complete_provenance() -> None:
    connector = _connector(
        MemoryDocument(
            "reports/q3.pdf",
            b"payload",
            declared_media_type="application/pdf",
            version_tag="v7",
            attributes={"origin": "upload"},
        )
    )

    record = (await _discover(connector))[0]

    assert record.source_uri == "memory:///reports/q3.pdf"
    assert record.scheme == "memory"
    assert record.connector_name == "InMemoryConnector"
    assert record.filename == "q3.pdf"
    assert record.declared_media_type == "application/pdf"
    assert record.size_bytes == 7
    assert record.version_tag == "v7"
    assert record.attributes == {"origin": "upload"}
    assert record.discovered_at == NOW


async def test_root_prefix_scopes_discovery() -> None:
    connector = _connector(
        MemoryDocument("reports/q3.pdf", b"a"),
        MemoryDocument("reports/q4.pdf", b"b"),
        MemoryDocument("notes/memo.txt", b"c"),
    )

    records = await _discover(connector, _spec("memory:///reports"))

    assert [r.record_id for r in records] == ["reports/q3.pdf", "reports/q4.pdf"]


async def test_prefix_matching_respects_segment_boundaries() -> None:
    """A root of /reports must not select /reports-archive."""
    connector = _connector(
        MemoryDocument("reports/a.txt", b"a"),
        MemoryDocument("reports-archive/b.txt", b"b"),
    )

    records = await _discover(connector, _spec("memory:///reports"))

    assert [r.record_id for r in records] == ["reports/a.txt"]


async def test_empty_root_selects_everything() -> None:
    connector = _connector(
        MemoryDocument("a.txt", b"a"), MemoryDocument("deep/b.txt", b"b")
    )

    assert len(await _discover(connector, _spec("memory:///"))) == 2


async def test_selection_filters_are_applied() -> None:
    connector = _connector(
        MemoryDocument("a.pdf", b"a"),
        MemoryDocument("b.txt", b"b"),
        MemoryDocument("c.tmp", b"c"),
    )

    records = await _discover(
        connector, _spec(include_patterns=("*.pdf", "*.txt"))
    )

    assert [r.record_id for r in records] == ["a.pdf", "b.txt"]


async def test_exclusion_filters_are_applied() -> None:
    connector = _connector(
        MemoryDocument("a.pdf", b"a"), MemoryDocument("b.tmp", b"b")
    )

    records = await _discover(connector, _spec(exclude_patterns=("*.tmp",)))

    assert [r.record_id for r in records] == ["a.pdf"]


async def test_empty_connector_discovers_nothing() -> None:
    assert await _discover(_connector()) == []


async def test_identifiers_are_percent_encoded_in_uris() -> None:
    connector = _connector(MemoryDocument("reports/q3 final.pdf", b"x"))

    record = (await _discover(connector))[0]

    assert record.source_uri == "memory:///reports/q3%20final.pdf"


# ---------------------------------------------------------------------------
# Fetching
# ---------------------------------------------------------------------------


async def test_fetch_returns_the_payload() -> None:
    connector = _connector(MemoryDocument("a.txt", b"hello world"))
    record = (await _discover(connector))[0]

    blob = await connector.fetch(record)

    assert blob.read() == b"hello world"
    assert blob.size_bytes == 11
    assert blob.record.source_uri == record.source_uri


async def test_digest_matches_the_shared_provider() -> None:
    """A differing algorithm would be rejected by the acquisition stage."""
    payload = b"the quick brown fox"
    connector = _connector(MemoryDocument("a.txt", payload))
    record = (await _discover(connector))[0]

    blob = await connector.fetch(record)

    assert blob.content_hash == Blake2bHashProvider().hash_bytes(payload)
    assert blob.hash_algorithm == "blake2b-256"


async def test_acquisition_instant_comes_from_the_injected_clock() -> None:
    connector = _connector(MemoryDocument("a.txt", b"data"))
    record = (await _discover(connector))[0]

    assert (await connector.fetch(record)).acquired_at == NOW


async def test_fetching_an_unknown_record_is_reported() -> None:
    connector = _connector(MemoryDocument("a.txt", b"data"))
    record = (await _discover(connector))[0]
    unknown = record.model_copy(update={"source_uri": "memory:///absent.txt"})

    with pytest.raises(SourceNotFoundError):
        await connector.fetch(unknown)


async def test_oversized_payload_is_rejected() -> None:
    config = _config(
        max_document_bytes=64, spool_threshold_bytes=32, read_chunk_bytes=16
    )
    connector = _connector(MemoryDocument("big.bin", b"x" * 500), config=config)
    record = (await _discover(connector))[0]

    with pytest.raises(DocumentTooLargeError):
        await connector.fetch(record)


async def test_repeated_fetches_yield_identical_digests() -> None:
    connector = _connector(MemoryDocument("a.txt", b"stable"))
    record = (await _discover(connector))[0]

    first = await connector.fetch(record)
    second = await connector.fetch(record)

    assert first.content_hash == second.content_hash


# ---------------------------------------------------------------------------
# Offload decisions
# ---------------------------------------------------------------------------


async def test_small_payloads_are_built_without_offloading() -> None:
    """Below the threshold no filesystem access is possible, so offloading
    would add scheduling cost for nothing."""
    offloader = _CountingOffloader()
    config = _config(spool_threshold_bytes=1 * KIB, read_chunk_bytes=256)
    connector = _connector(
        MemoryDocument("a.txt", b"small"), config=config, offloader=offloader
    )
    record = (await _discover(connector))[0]

    await connector.fetch(record)

    assert offloader.calls == 0


async def test_large_payloads_are_offloaded() -> None:
    offloader = _CountingOffloader()
    config = _config(spool_threshold_bytes=64, read_chunk_bytes=32)
    connector = _connector(
        MemoryDocument("big.bin", b"x" * 512), config=config, offloader=offloader
    )
    record = (await _discover(connector))[0]

    blob = await connector.fetch(record)

    assert offloader.calls == 1
    assert blob.is_spooled is True
    blob.close()


async def test_spooled_and_inline_payloads_share_a_digest() -> None:
    payload = b"y" * 200
    inline = _connector(
        MemoryDocument("a.bin", payload),
        config=_config(spool_threshold_bytes=10 * KIB, read_chunk_bytes=512),
    )
    spooled = _connector(
        MemoryDocument("a.bin", payload),
        config=_config(spool_threshold_bytes=64, read_chunk_bytes=32),
    )

    first = await inline.fetch((await _discover(inline))[0])
    second = await spooled.fetch((await _discover(spooled))[0])

    assert first.content_hash == second.content_hash
    second.close()


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


async def test_health_check_reports_operational() -> None:
    assert await _connector().health_check() is True


async def test_close_releases_documents_and_is_idempotent() -> None:
    connector = _connector(MemoryDocument("a.txt", b"data"))

    await connector.close()
    await connector.close()

    assert connector.document_count == 0
    assert await connector.health_check() is False


async def test_use_after_close_is_a_permanent_failure() -> None:
    connector = _connector(MemoryDocument("a.txt", b"data"))
    record = (await _discover(connector))[0]
    await connector.close()

    with pytest.raises(SourceError) as excinfo:
        await connector.fetch(record)

    assert excinfo.value.retryable is False


async def test_discovery_after_close_is_rejected() -> None:
    connector = _connector(MemoryDocument("a.txt", b"data"))
    await connector.close()

    with pytest.raises(SourceError):
        await _discover(connector)


async def test_context_manager_closes_the_connector() -> None:
    connector = _connector(MemoryDocument("a.txt", b"data"))

    async with connector:
        assert await connector.health_check() is True

    assert await connector.health_check() is False


# ---------------------------------------------------------------------------
# Anti-pattern guards
# ---------------------------------------------------------------------------


def test_repr_never_leaks_payload_bytes() -> None:
    connector = _connector(MemoryDocument("a.txt", b"SENSITIVE"))

    assert "SENSITIVE" not in repr(connector)


def test_connector_holds_no_assignable_state() -> None:
    with pytest.raises(AttributeError):
        _connector().cache = {}  # type: ignore[attr-defined]


def test_module_declares_no_mutable_state() -> None:
    mutable = [
        name
        for name, value in vars(memory_module).items()
        if not name.startswith("__") and isinstance(value, (list, dict, set))
    ]
    assert mutable == []


def test_connector_declares_no_hardcoded_limits() -> None:
    source = inspect.getsource(InMemoryConnector)

    assert "self._config.acquisition" in source
    assert "> 0.8" not in source


def test_connector_does_not_reimplement_shared_machinery() -> None:
    """Digests, spooling, and filtering are written once, not per connector."""
    source = inspect.getsource(memory_module)

    assert "PayloadBuilder" in source
    assert "matches_filters" in source
    assert "hashlib" not in source
    assert "tempfile" not in source


def test_connector_contains_no_domain_vocabulary() -> None:
    source = inspect.getsource(memory_module).lower()
    for term in ("jurisdiction", "statute", "ticker", "icd", "patient", "court"):
        assert term not in source
