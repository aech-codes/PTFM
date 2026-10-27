"""Unit tests for the deduplication stage."""

from __future__ import annotations

import inspect
from datetime import UTC, datetime
from typing import Any, Optional

import pytest

from ptfm.core.ingestion.config import DeduplicationConfig
from ptfm.core.ingestion.exceptions import (
    DeduplicationError,
    FingerprintStoreError,
    IngestionStage,
)
from ptfm.core.ingestion.interfaces import (
    DocumentFingerprintStore,
    FingerprintRecord,
)
from ptfm.core.ingestion.models import (
    AcquiredBlob,
    DocumentSkipped,
    SkipReason,
    SourceRecord,
)
from ptfm.core.ingestion.pipeline import deduplication as dedup_module
from ptfm.core.ingestion.pipeline.deduplication import (
    Deduplicator,
    ExtendedFingerprintStore,
    InMemoryFingerprintStore,
    NearDuplicateMatch,
    VersionCursor,
    compute_simhash,
    hamming_distance,
)
from ptfm.core.schemas import (
    DocumentMetadata,
    ExtractionWarningCode,
    RawDocument,
    SourceDescriptor,
    derive_document_id,
)

NOW = datetime(2026, 7, 30, tzinfo=UTC)


# ---------------------------------------------------------------------------
# Doubles and builders
# ---------------------------------------------------------------------------


class _FixedClock:
    def now(self) -> datetime:
        return NOW


class _BaseOnlyStore:
    """Store satisfying only the base protocol, with no extended capability."""

    def __init__(self) -> None:
        self._records: dict[str, FingerprintRecord] = {}

    async def lookup(self, content_hash: str) -> Optional[FingerprintRecord]:
        return self._records.get(content_hash)

    async def record(self, record: FingerprintRecord) -> None:
        self._records.setdefault(record.content_hash, record)

    async def close(self) -> None:
        self._records.clear()


class _FailingStore(InMemoryFingerprintStore):
    """Store raising on a nominated operation."""

    def __init__(self, failing: str, error: Exception) -> None:
        super().__init__()
        self._failing = failing
        self._error = error

    async def lookup(self, content_hash: str) -> Optional[FingerprintRecord]:
        if self._failing == "lookup":
            raise self._error
        return await super().lookup(content_hash)

    async def record(self, record: FingerprintRecord) -> None:
        if self._failing == "record":
            raise self._error
        await super().record(record)

    async def get_version_cursor(self, source_uri: str) -> Optional[VersionCursor]:
        if self._failing == "get_version_cursor":
            raise self._error
        return await super().get_version_cursor(source_uri)

    async def nearest_simhash(
        self, fingerprint: int, max_distance: int
    ) -> Optional[NearDuplicateMatch]:
        if self._failing == "nearest_simhash":
            raise self._error
        return await super().nearest_simhash(fingerprint, max_distance)


def _record(
    uri: str = "file:///corpus/a.txt",
    version_tag: str | None = None,
) -> SourceRecord:
    return SourceRecord(
        record_id="r-1",
        source_uri=uri,
        scheme="file",
        connector_name="TestConnector",
        discovered_at=NOW,
        version_tag=version_tag,
    )


def _blob(
    content_hash: str = "hash-1",
    uri: str = "file:///corpus/a.txt",
) -> AcquiredBlob:
    return AcquiredBlob(
        record=_record(uri),
        size_bytes=5,
        content_hash=content_hash,
        hash_algorithm="blake2b-256",
        acquired_at=NOW,
        data=b"hello",
    )


def _document(
    *,
    content_hash: str = "hash-1",
    uri: str = "file:///corpus/a.txt",
    text: str = "the quick brown fox jumps over the lazy dog",
    version_tag: str | None = None,
) -> RawDocument:
    return RawDocument(
        document_id=derive_document_id(uri, content_hash),
        source=SourceDescriptor(
            source_uri=uri,
            scheme="file",
            connector_name="TestConnector",
            acquired_at=NOW,
            size_bytes=len(text),
            detected_media_type="text/plain",
            version_tag=version_tag,
        ),
        metadata=DocumentMetadata(
            char_count=len(text), word_count=len(text.split())
        ),
        text=text,
        content_hash=content_hash,
        text_hash="text-hash",
        ingested_at=NOW,
    )


def _dedup(
    store: Any = None,
    **config: Any,
) -> tuple[Deduplicator, Any]:
    resolved = store if store is not None else InMemoryFingerprintStore()
    return (
        Deduplicator(resolved, DeduplicationConfig(**config), _FixedClock()),
        resolved,
    )


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


def test_in_memory_store_satisfies_both_protocols() -> None:
    store = InMemoryFingerprintStore()

    assert isinstance(store, DocumentFingerprintStore)
    assert isinstance(store, ExtendedFingerprintStore)


def test_base_only_store_is_not_extended() -> None:
    store = _BaseOnlyStore()

    assert isinstance(store, DocumentFingerprintStore)
    assert not isinstance(store, ExtendedFingerprintStore)


def test_capability_is_resolved_once_at_construction() -> None:
    extended, _ = _dedup(detect_near_duplicates=True)
    limited, _ = _dedup(_BaseOnlyStore(), detect_near_duplicates=True)

    assert extended.supports_version_cursors is True
    assert extended.supports_near_duplicates is True
    assert limited.supports_version_cursors is False
    assert limited.supports_near_duplicates is False


# ---------------------------------------------------------------------------
# Exact duplicate detection
# ---------------------------------------------------------------------------


async def test_new_payload_is_not_a_duplicate() -> None:
    dedup, _ = _dedup()

    assert await dedup.check_duplicate(_blob()) is None


async def test_recorded_payload_is_recognized_as_a_duplicate() -> None:
    """Duplication is a skip, not a failure."""
    dedup, _ = _dedup()
    document = _document()
    await dedup.record(document)

    outcome = await dedup.check_duplicate(_blob())

    assert isinstance(outcome, DocumentSkipped)
    assert outcome.reason is SkipReason.DUPLICATE_CONTENT
    assert outcome.existing_document_id == document.document_id
    assert outcome.kind == "SKIPPED"


async def test_identical_content_at_a_different_uri_is_a_duplicate() -> None:
    dedup, _ = _dedup()
    await dedup.record(_document(uri="file:///corpus/original.txt"))

    outcome = await dedup.check_duplicate(
        _blob(uri="file:///corpus/copy.txt")
    )

    assert outcome is not None
    assert outcome.source_uri == "file:///corpus/copy.txt"


async def test_different_content_is_not_a_duplicate() -> None:
    dedup, _ = _dedup()
    await dedup.record(_document(content_hash="hash-1"))

    assert await dedup.check_duplicate(_blob(content_hash="hash-2")) is None


async def test_exact_detection_works_without_extended_capability() -> None:
    dedup, _ = _dedup(_BaseOnlyStore())
    await dedup.record(_document())

    assert await dedup.check_duplicate(_blob()) is not None


async def test_first_sighting_is_retained_on_re_record() -> None:
    """first_seen_at must remain truthful."""
    store = InMemoryFingerprintStore()
    dedup, _ = _dedup(store)

    await dedup.record(_document(uri="file:///first.txt"))
    await dedup.record(_document(uri="file:///second.txt"))

    recorded = await store.lookup("hash-1")
    assert recorded is not None
    assert recorded.source_uri == "file:///first.txt"


async def test_recorded_sighting_uses_the_injected_clock() -> None:
    store = InMemoryFingerprintStore()
    dedup, _ = _dedup(store)

    await dedup.record(_document())

    recorded = await store.lookup("hash-1")
    assert recorded is not None
    assert recorded.first_seen_at == NOW


# ---------------------------------------------------------------------------
# Disabled deduplication
# ---------------------------------------------------------------------------


async def test_disabled_deduplication_never_skips() -> None:
    dedup, _ = _dedup(enabled=False)
    await dedup.record(_document())

    assert await dedup.check_duplicate(_blob()) is None
    assert await dedup.check_unchanged(_record(version_tag="v1")) is None


async def test_disabled_deduplication_records_nothing() -> None:
    store = InMemoryFingerprintStore()
    dedup, _ = _dedup(store, enabled=False)

    await dedup.record(_document())

    assert await store.lookup("hash-1") is None


# ---------------------------------------------------------------------------
# Unchanged version skipping
# ---------------------------------------------------------------------------


async def test_unchanged_version_is_skipped_before_fetch() -> None:
    """The mechanism that makes repeated ingestion inexpensive."""
    dedup, _ = _dedup()
    document = _document(version_tag="etag-abc")
    await dedup.record(document)

    outcome = await dedup.check_unchanged(_record(version_tag="etag-abc"))

    assert isinstance(outcome, DocumentSkipped)
    assert outcome.reason is SkipReason.UNCHANGED_VERSION
    assert outcome.existing_document_id == document.document_id


async def test_changed_version_is_not_skipped() -> None:
    dedup, _ = _dedup()
    await dedup.record(_document(version_tag="etag-abc"))

    assert await dedup.check_unchanged(_record(version_tag="etag-xyz")) is None


async def test_record_without_a_version_tag_is_never_skipped() -> None:
    dedup, _ = _dedup()
    await dedup.record(_document(version_tag="etag-abc"))

    assert await dedup.check_unchanged(_record(version_tag=None)) is None


async def test_unseen_uri_is_not_skipped() -> None:
    dedup, _ = _dedup()

    assert await dedup.check_unchanged(_record(version_tag="etag-abc")) is None


async def test_version_cursor_is_updated_on_re_ingestion() -> None:
    store = InMemoryFingerprintStore()
    dedup, _ = _dedup(store)

    await dedup.record(_document(content_hash="h1", version_tag="v1"))
    await dedup.record(_document(content_hash="h2", version_tag="v2"))

    cursor = await store.get_version_cursor("file:///corpus/a.txt")
    assert cursor is not None
    assert cursor.version_tag == "v2"


async def test_version_skipping_can_be_disabled() -> None:
    dedup, _ = _dedup(skip_unchanged_versions=False)
    await dedup.record(_document(version_tag="etag-abc"))

    assert dedup.supports_version_cursors is False
    assert await dedup.check_unchanged(_record(version_tag="etag-abc")) is None


async def test_version_skipping_degrades_without_extended_capability() -> None:
    dedup, _ = _dedup(_BaseOnlyStore())
    await dedup.record(_document(version_tag="etag-abc"))

    assert await dedup.check_unchanged(_record(version_tag="etag-abc")) is None


# ---------------------------------------------------------------------------
# SimHash
# ---------------------------------------------------------------------------


def test_identical_text_yields_identical_fingerprints() -> None:
    text = "the quick brown fox jumps over the lazy dog"

    first = compute_simhash(text, bits=64)
    second = compute_simhash(text, bits=64)

    assert first == second
    assert first is not None
    assert hamming_distance(first, second) == 0


def test_a_small_edit_yields_a_small_distance() -> None:
    original = " ".join(f"word{n}" for n in range(200))
    edited = original.replace("word100", "changed")

    first = compute_simhash(original, bits=64)
    second = compute_simhash(edited, bits=64)

    assert first is not None and second is not None
    assert 0 < hamming_distance(first, second) <= 8


def test_unrelated_text_yields_a_large_distance() -> None:
    first = compute_simhash(
        " ".join(f"alpha{n}" for n in range(200)), bits=64
    )
    second = compute_simhash(
        " ".join(f"omega{n}" for n in range(200)), bits=64
    )

    assert first is not None and second is not None
    assert hamming_distance(first, second) > 8


def test_text_shorter_than_one_shingle_yields_no_fingerprint() -> None:
    assert compute_simhash("two words", bits=64, shingle_size=3) is None
    assert compute_simhash("", bits=64) is None


def test_exactly_one_shingle_yields_a_fingerprint() -> None:
    assert compute_simhash("one two three", bits=64, shingle_size=3) is not None


@pytest.mark.parametrize("bits", [32, 64, 128, 256])
def test_supported_fingerprint_widths(bits: int) -> None:
    fingerprint = compute_simhash("a b c d e f g h", bits=bits)

    assert fingerprint is not None
    assert 0 <= fingerprint < (1 << bits)


@pytest.mark.parametrize("bits", [0, -8, 7, 63])
def test_invalid_fingerprint_width_is_rejected(bits: int) -> None:
    with pytest.raises(DeduplicationError, match="multiple of"):
        compute_simhash("a b c d", bits=bits)


def test_invalid_shingle_size_is_rejected() -> None:
    with pytest.raises(DeduplicationError, match="at least 1"):
        compute_simhash("a b c", bits=64, shingle_size=0)


@pytest.mark.parametrize(
    ("first", "second", "expected"),
    [(0, 0, 0), (0b1010, 0b1010, 0), (0b0000, 0b1111, 4), (0b1100, 0b1010, 2)],
)
def test_hamming_distance(first: int, second: int, expected: int) -> None:
    assert hamming_distance(first, second) == expected


# ---------------------------------------------------------------------------
# Near-duplicate detection
# ---------------------------------------------------------------------------


async def test_near_duplicates_are_off_by_default() -> None:
    dedup, _ = _dedup()

    assert dedup.supports_near_duplicates is False
    assert await dedup.check_near_duplicate("a b c d e f") is None


async def test_near_duplicate_is_reported_as_a_warning_not_a_skip() -> None:
    """Discarding near duplicates is a corpus policy, not an ingestion one."""
    dedup, _ = _dedup(detect_near_duplicates=True, near_duplicate_max_distance=8)
    original = " ".join(f"word{n}" for n in range(200))
    document = _document(text=original)
    await dedup.record(document)

    warning = await dedup.check_near_duplicate(
        original.replace("word100", "changed")
    )

    assert warning is not None
    assert warning.code is ExtractionWarningCode.NEAR_DUPLICATE_DETECTED
    assert warning.detail["similar_document_id"] == document.document_id
    assert not isinstance(warning, DocumentSkipped)


async def test_unrelated_text_produces_no_warning() -> None:
    dedup, _ = _dedup(detect_near_duplicates=True, near_duplicate_max_distance=3)
    await dedup.record(
        _document(text=" ".join(f"alpha{n}" for n in range(200)))
    )

    warning = await dedup.check_near_duplicate(
        " ".join(f"omega{n}" for n in range(200))
    )

    assert warning is None


async def test_similarity_bound_governs_reporting() -> None:
    original = " ".join(f"word{n}" for n in range(200))
    edited = original.replace("word50", "x").replace("word150", "y")

    strict, _ = _dedup(detect_near_duplicates=True, near_duplicate_max_distance=0)
    lax, _ = _dedup(detect_near_duplicates=True, near_duplicate_max_distance=32)
    for dedup in (strict, lax):
        await dedup.record(_document(text=original))

    assert await strict.check_near_duplicate(edited) is None
    assert await lax.check_near_duplicate(edited) is not None


async def test_short_text_produces_no_near_duplicate_warning() -> None:
    dedup, _ = _dedup(detect_near_duplicates=True)
    await dedup.record(_document(text="a b c d e f g h"))

    assert await dedup.check_near_duplicate("one") is None


async def test_near_duplicate_detection_requires_extended_capability() -> None:
    dedup, _ = _dedup(_BaseOnlyStore(), detect_near_duplicates=True)

    assert await dedup.check_near_duplicate("a b c d e f g h") is None


async def test_no_fingerprint_is_recorded_when_detection_is_off() -> None:
    store = InMemoryFingerprintStore()
    dedup, _ = _dedup(store, detect_near_duplicates=False)

    await dedup.record(_document())

    assert await store.nearest_simhash(0, 256) is None


async def test_nearest_match_selection_is_deterministic() -> None:
    store = InMemoryFingerprintStore()
    await store.record_simhash("doc-far", 0b1111)
    await store.record_simhash("doc-near", 0b0001)

    match = await store.nearest_simhash(0b0000, max_distance=8)

    assert match == NearDuplicateMatch(document_id="doc-near", distance=1)


# ---------------------------------------------------------------------------
# Store failures
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "operation", ["lookup", "get_version_cursor", "nearest_simhash"]
)
async def test_store_read_failure_is_wrapped(operation: str) -> None:
    store = _FailingStore(operation, ConnectionError("store unreachable"))
    dedup, _ = _dedup(
        store, detect_near_duplicates=True
    )

    calls = {
        "lookup": lambda: dedup.check_duplicate(_blob()),
        "get_version_cursor": lambda: dedup.check_unchanged(
            _record(version_tag="v1")
        ),
        "nearest_simhash": lambda: dedup.check_near_duplicate(
            " ".join(f"w{n}" for n in range(20))
        ),
    }

    with pytest.raises(FingerprintStoreError) as excinfo:
        await calls[operation]()

    error = excinfo.value
    assert error.operation == operation
    assert isinstance(error.__cause__, ConnectionError)
    assert error.stage is IngestionStage.DEDUPLICATION


async def test_store_write_failure_is_wrapped() -> None:
    store = _FailingStore("record", OSError("disk full"))
    dedup, _ = _dedup(store)

    with pytest.raises(FingerprintStoreError) as excinfo:
        await dedup.record(_document())

    assert excinfo.value.operation == "record"


async def test_store_failures_are_not_retryable() -> None:
    """A degraded store must fail loudly, not be absorbed by a retry loop."""
    store = _FailingStore("lookup", ConnectionError("down"))
    dedup, _ = _dedup(store)

    with pytest.raises(FingerprintStoreError) as excinfo:
        await dedup.check_duplicate(_blob())

    assert excinfo.value.retryable is False


async def test_already_wrapped_store_error_is_not_double_wrapped() -> None:
    original = FingerprintStoreError("already typed", operation="lookup")
    store = _FailingStore("lookup", original)
    dedup, _ = _dedup(store)

    with pytest.raises(FingerprintStoreError) as excinfo:
        await dedup.check_duplicate(_blob())

    assert excinfo.value is original


# ---------------------------------------------------------------------------
# In-memory store behaviour
# ---------------------------------------------------------------------------


async def test_store_roundtrip() -> None:
    store = InMemoryFingerprintStore()
    record = FingerprintRecord(
        content_hash="h1",
        document_id="doc-1",
        source_uri="file:///a.txt",
        first_seen_at=NOW,
    )

    assert await store.lookup("h1") is None
    await store.record(record)
    assert await store.lookup("h1") == record


async def test_close_clears_all_state_and_is_idempotent() -> None:
    store = InMemoryFingerprintStore()
    await store.record(
        FingerprintRecord("h1", "doc-1", "file:///a.txt", NOW)
    )
    await store.set_version_cursor("file:///a.txt", VersionCursor("v1", "doc-1"))
    await store.record_simhash("doc-1", 42)

    await store.close()
    await store.close()

    assert await store.lookup("h1") is None
    assert await store.get_version_cursor("file:///a.txt") is None
    assert await store.nearest_simhash(42, 0) is None


async def test_in_memory_state_does_not_survive_a_new_store() -> None:
    """Documented limitation: no cross-run detection without persistence."""
    first = InMemoryFingerprintStore()
    await first.record(FingerprintRecord("h1", "doc-1", "file:///a.txt", NOW))

    second = InMemoryFingerprintStore()

    assert await second.lookup("h1") is None


async def test_concurrent_identical_payloads_may_both_pass_the_check() -> None:
    """
    Known and documented: lookup and record are separate awaits. Sequential
    and cross-run detection is exact; closing this window requires a
    uniqueness constraint at the storage layer.
    """
    import asyncio

    dedup, _ = _dedup()

    outcomes = await asyncio.gather(
        dedup.check_duplicate(_blob()), dedup.check_duplicate(_blob())
    )

    assert outcomes == [None, None]


# ---------------------------------------------------------------------------
# Anti-pattern guards
# ---------------------------------------------------------------------------


def test_duplicate_is_never_modelled_as_an_exception() -> None:
    assert not hasattr(dedup_module, "DuplicateDocumentError")

    source = inspect.getsource(Deduplicator)
    assert "SkipReason.DUPLICATE_CONTENT" in source
    assert "raise DuplicateDocument" not in source


def test_stage_declares_no_hardcoded_limits() -> None:
    source = inspect.getsource(Deduplicator)

    assert "self._config.simhash_bits" in source
    assert "self._config.near_duplicate_max_distance" in source
    assert "> 0.8" not in source
    assert "distance <= 3" not in source


def test_similarity_hashing_does_not_use_the_identity_provider() -> None:
    """Similarity fingerprinting is explicitly not identity."""
    source = inspect.getsource(dedup_module)

    assert "hashlib.blake2b" in source
    assert "HashProvider" not in source


def test_stage_never_catches_base_exception() -> None:
    source = inspect.getsource(dedup_module)

    assert "except BaseException" not in source
    assert "except:" not in source


def test_deduplicator_holds_no_assignable_state() -> None:
    dedup, _ = _dedup()

    with pytest.raises(AttributeError):
        dedup.cache = {}  # type: ignore[attr-defined]


def test_module_declares_no_mutable_state() -> None:
    mutable = [
        name
        for name, value in vars(dedup_module).items()
        if not name.startswith("__") and isinstance(value, (list, dict, set))
    ]
    assert mutable == []


def test_stage_never_reads_the_wall_clock_directly() -> None:
    source = inspect.getsource(dedup_module)

    assert "self._clock.now()" in source
    assert "datetime.now(" not in source
    assert "utcnow" not in source


def test_stage_imports_no_concrete_implementations() -> None:
    source = inspect.getsource(dedup_module)
    for forbidden in (
        "from ptfm.core.ingestion.extractors",
        "from ptfm.core.ingestion.connectors",
        "from ptfm.plugins",
    ):
        assert forbidden not in source


def test_stage_contains_no_domain_vocabulary() -> None:
    source = inspect.getsource(dedup_module).lower()
    for term in ("jurisdiction", "statute", "ticker", "icd", "patient", "court"):
        assert term not in source