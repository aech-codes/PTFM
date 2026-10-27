"""
PTFM Deduplication Stage

File:
    ptfm/core/ingestion/pipeline/deduplication.py

Description:
    Recognizes documents that have already been ingested, so that repeated
    runs over a corpus do not reprocess unchanged material.

Duplication is not a failure
----------------------------
    Encountering a duplicate is the expected steady state of incremental
    re-ingestion: a second run over an unchanged corpus produces a skip for
    nearly every record, which is precisely the intended behaviour. Skips
    are therefore reported as ``DocumentSkipped`` outcomes, never as
    exceptions. Only store faults raise.

Three checks at three points
----------------------------
    The checks cannot be combined, because each needs an input that is
    available at a different point in the pipeline:

        check_unchanged      BEFORE fetch, from the source version tag.
                             Skipping here avoids the transfer entirely,
                             which is what makes re-ingestion cheap.

        check_duplicate      AFTER fetch, from the payload digest. Detects
                             identical content that arrived under a
                             different URI or without a version tag.

        check_near_duplicate AFTER normalization, from the document text.
                             Reports a WARNING and still ingests: whether
                             to discard near duplicates is a corpus policy
                             decision, not an ingestion one.

Optional store capabilities
---------------------------
    ``DocumentFingerprintStore`` is keyed by content digest alone and cannot
    answer version-cursor or similarity queries. Stores that support those
    features additionally satisfy ``ExtendedFingerprintStore``, declared
    below. The capability is determined once at construction; when it is
    absent, exact duplicate detection operates normally and the two optional
    features disable themselves.

Near-duplicate scalability
--------------------------
    Similarity search is delegated to the store rather than performed here,
    so that the search strategy is an implementation concern. The bundled
    in-memory store scans linearly, which is O(n) per document and therefore
    O(n-squared) across a corpus. That is unsuitable for large corpora, which
    is why the feature defaults to off. A production store implements
    locality-sensitive hashing behind the same method with no change to this
    stage.

Concurrency
-----------
    Lookup and record are separate awaits, so two identical payloads
    processed concurrently within one run may both pass the exact-duplicate
    check. Sequential and cross-run detection is exact. Closing the
    concurrent window requires a uniqueness constraint at the storage layer,
    which a persistent store provides and an in-memory store cannot.

Dependency contract
-------------------
    Imports only the standard library, ``ptfm.core.schemas`` and the
    ingestion ``exceptions``, ``config``, ``models`` and ``interfaces``
    modules.
"""

from __future__ import annotations

import asyncio
import hashlib
from typing import Final, NamedTuple, Optional, Protocol, runtime_checkable

from ptfm.core.ingestion.config import DeduplicationConfig
from ptfm.core.ingestion.exceptions import (
    DeduplicationError,
    FingerprintStoreError,
)
from ptfm.core.ingestion.interfaces import (
    Clock,
    DocumentFingerprintStore,
    FingerprintRecord,
)
from ptfm.core.ingestion.models import (
    AcquiredBlob,
    DocumentSkipped,
    SkipReason,
    SourceRecord,
)
from ptfm.core.schemas import (
    ExtractionWarning,
    ExtractionWarningCode,
    RawDocument,
)

__all__ = [
    "NearDuplicateMatch",
    "VersionCursor",
    "ExtendedFingerprintStore",
    "compute_simhash",
    "hamming_distance",
    "InMemoryFingerprintStore",
    "Deduplicator",
]

#: Bits per byte, used to convert a fingerprint width into a digest size.
_BITS_PER_BYTE: Final[int] = 8

#: Default word-shingle width for similarity fingerprinting. Three words
#: captures local phrasing without being so long that a single edit
#: invalidates every shingle containing it.
#:
#: This value properly belongs in ``DeduplicationConfig``. It is exposed as
#: a constructor parameter so that it remains injectable without amending
#: approved configuration.
_DEFAULT_SHINGLE_SIZE: Final[int] = 3


# ============================================================================
# OPTIONAL STORE CAPABILITIES
# ============================================================================


class VersionCursor(NamedTuple):
    """
    The source-side version last recorded for a URI.

    Attributes
    ----------
    version_tag:
        Opaque source version marker, such as an ETag or modification
        timestamp, as reported when the document was last ingested.
    document_id:
        Canonical identifier assigned at that time.
    """

    version_tag: str
    document_id: str


class NearDuplicateMatch(NamedTuple):
    """
    A previously ingested document similar to the one being processed.

    Attributes
    ----------
    document_id:
        Canonical identifier of the similar document.
    distance:
        Hamming distance between the two similarity fingerprints. Lower
        values indicate greater similarity.
    """

    document_id: str
    distance: int


@runtime_checkable
class ExtendedFingerprintStore(DocumentFingerprintStore, Protocol):
    """
    A fingerprint store additionally supporting version cursors and
    similarity search.

    ``DocumentFingerprintStore`` is keyed by content digest alone, which is
    sufficient for exact duplicate detection but cannot answer either "what
    version did I last see at this URI" or "is anything similar to this
    already present". Stores providing those capabilities satisfy this
    protocol; the deduplicator detects the capability once at construction
    and disables the corresponding features when it is absent.

    Similarity search is declared here rather than implemented in the
    deduplicator so that the search strategy belongs to the store. A linear
    scan is adequate for modest corpora; a production store implements
    locality-sensitive hashing behind the same method.

    Note that ``runtime_checkable`` verifies member presence only. Signature
    conformance is established by ``mypy --strict``.
    """

    async def get_version_cursor(self, source_uri: str) -> Optional[VersionCursor]:
        """
        Retrieve the version last recorded for a URI.

        Parameters
        ----------
        source_uri:
            URI to look up.

        Returns
        -------
        Optional[VersionCursor]
            The recorded cursor, or ``None`` when the URI has not been
            ingested or carried no version tag.

        Raises
        ------
        FingerprintStoreError
            If the store cannot be read.
        """
        ...

    async def set_version_cursor(
        self,
        source_uri: str,
        cursor: VersionCursor,
    ) -> None:
        """
        Record the version observed for a URI.

        Parameters
        ----------
        source_uri:
            URI being recorded.
        cursor:
            The version tag and document identifier to persist.

        Returns
        -------
        None

        Raises
        ------
        FingerprintStoreError
            If the store cannot be written.
        """
        ...

    async def record_simhash(self, document_id: str, fingerprint: int) -> None:
        """
        Persist a similarity fingerprint.

        Parameters
        ----------
        document_id:
            Canonical identifier of the document.
        fingerprint:
            Similarity fingerprint as an unsigned integer.

        Returns
        -------
        None

        Raises
        ------
        FingerprintStoreError
            If the store cannot be written.
        """
        ...

    async def nearest_simhash(
        self,
        fingerprint: int,
        max_distance: int,
    ) -> Optional[NearDuplicateMatch]:
        """
        Find a recorded fingerprint within a Hamming distance.

        Parameters
        ----------
        fingerprint:
            Fingerprint to search for.
        max_distance:
            Largest distance considered a match.

        Returns
        -------
        Optional[NearDuplicateMatch]
            The closest match within the bound, or ``None`` if none exists.

        Raises
        ------
        FingerprintStoreError
            If the store cannot be searched.
        """
        ...


# ============================================================================
# SIMILARITY FINGERPRINTING
# ============================================================================


def compute_simhash(
    text: str,
    *,
    bits: int,
    shingle_size: int = _DEFAULT_SHINGLE_SIZE,
) -> Optional[int]:
    """
    Compute a similarity fingerprint over document text.

    Overlapping word shingles are hashed and their bits votes summed
    position by position; the resulting fingerprint has a bit set wherever
    the votes were net positive. Two documents differing in a small
    proportion of their content produce fingerprints differing in a small
    number of bits, which is what makes Hamming distance a usable similarity
    measure.

    Hashing uses ``hashlib`` directly rather than the injected
    ``HashProvider``. Similarity fingerprinting is explicitly not identity,
    and routing it through the identity abstraction would blur a boundary
    that matters.

    Parameters
    ----------
    text:
        Normalized document text.
    bits:
        Fingerprint width. Must be a positive multiple of eight.
    shingle_size:
        Number of consecutive words per shingle.

    Returns
    -------
    Optional[int]
        The fingerprint, or ``None`` when the text contains too few words to
        form a single shingle. A fingerprint over one or two words carries no
        usable similarity signal.

    Raises
    ------
    DeduplicationError
        If the fingerprint width is not a positive multiple of eight, or the
        shingle size is below one.
    """
    if bits <= 0 or bits % _BITS_PER_BYTE != 0:
        raise DeduplicationError(
            f"Similarity fingerprint width must be a positive multiple of "
            f"{_BITS_PER_BYTE}, got {bits}.",
            context={"bits": bits},
        )
    if shingle_size < 1:
        raise DeduplicationError(
            f"Shingle size must be at least 1, got {shingle_size}.",
            context={"shingle_size": shingle_size},
        )

    tokens = text.split()
    if len(tokens) < shingle_size:
        return None

    digest_size = bits // _BITS_PER_BYTE
    votes = [0] * bits

    for start in range(len(tokens) - shingle_size + 1):
        shingle = " ".join(tokens[start : start + shingle_size])
        digest = hashlib.blake2b(
            shingle.encode("utf-8"), digest_size=digest_size
        ).digest()
        value = int.from_bytes(digest, "big")
        for position in range(bits):
            if (value >> position) & 1:
                votes[position] += 1
            else:
                votes[position] -= 1

    fingerprint = 0
    for position in range(bits):
        if votes[position] > 0:
            fingerprint |= 1 << position
    return fingerprint


def hamming_distance(first: int, second: int) -> int:
    """
    Count the differing bits between two fingerprints.

    Parameters
    ----------
    first:
        First fingerprint.
    second:
        Second fingerprint.

    Returns
    -------
    int
        Number of bit positions at which the fingerprints differ.
    """
    return (first ^ second).bit_count()


# ============================================================================
# IN-MEMORY STORE
# ============================================================================


class InMemoryFingerprintStore:
    """
    Fingerprint store held entirely in process memory.

    Satisfies ``ExtendedFingerprintStore``, so all three deduplication
    checks are available. Suitable for single-run ingestion, embedded use,
    and testing.

    Not suitable for production incremental ingestion: state is lost when
    the process exits, so a subsequent run rediscovers every document as
    new. A persistent implementation behind the same protocol provides
    cross-run detection and, through a uniqueness constraint, closes the
    concurrent check-then-record window that an in-memory store cannot.

    Similarity search is a linear scan, which is O(n) per query and
    therefore O(n-squared) across a corpus.

    Internal state is guarded by a lock so that concurrent access cannot
    corrupt it. The lock does not make check-then-record atomic across
    separate calls; see the module documentation.
    """

    __slots__ = ("_by_hash", "_cursors", "_simhashes", "_lock")

    def __init__(self) -> None:
        self._by_hash: dict[str, FingerprintRecord] = {}
        self._cursors: dict[str, VersionCursor] = {}
        self._simhashes: dict[str, int] = {}
        self._lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # Base protocol
    # ------------------------------------------------------------------

    async def lookup(self, content_hash: str) -> Optional[FingerprintRecord]:
        """
        Retrieve a prior sighting of a payload digest.

        Parameters
        ----------
        content_hash:
            Digest of the raw payload bytes.

        Returns
        -------
        Optional[FingerprintRecord]
            The prior sighting, or ``None`` if the payload is new.
        """
        async with self._lock:
            return self._by_hash.get(content_hash)

    async def record(self, record: FingerprintRecord) -> None:
        """
        Persist a payload sighting.

        The first sighting of a digest is retained. A later sighting of the
        same content at a different URI does not overwrite it, so that
        ``first_seen_at`` remains truthful.

        Parameters
        ----------
        record:
            The sighting to persist.

        Returns
        -------
        None
        """
        async with self._lock:
            self._by_hash.setdefault(record.content_hash, record)

    async def close(self) -> None:
        """
        Release all retained state.

        Idempotent.

        Returns
        -------
        None
        """
        async with self._lock:
            self._by_hash.clear()
            self._cursors.clear()
            self._simhashes.clear()

    # ------------------------------------------------------------------
    # Extended protocol
    # ------------------------------------------------------------------

    async def get_version_cursor(self, source_uri: str) -> Optional[VersionCursor]:
        """
        Retrieve the version last recorded for a URI.

        Parameters
        ----------
        source_uri:
            URI to look up.

        Returns
        -------
        Optional[VersionCursor]
            The recorded cursor, or ``None`` when absent.
        """
        async with self._lock:
            return self._cursors.get(source_uri)

    async def set_version_cursor(
        self,
        source_uri: str,
        cursor: VersionCursor,
    ) -> None:
        """
        Record the version observed for a URI.

        Overwrites any previous cursor, since the most recent observation is
        the one a later run must compare against.

        Parameters
        ----------
        source_uri:
            URI being recorded.
        cursor:
            The version tag and document identifier to persist.

        Returns
        -------
        None
        """
        async with self._lock:
            self._cursors[source_uri] = cursor

    async def record_simhash(self, document_id: str, fingerprint: int) -> None:
        """
        Persist a similarity fingerprint.

        Parameters
        ----------
        document_id:
            Canonical identifier of the document.
        fingerprint:
            Similarity fingerprint as an unsigned integer.

        Returns
        -------
        None
        """
        async with self._lock:
            self._simhashes[document_id] = fingerprint

    async def nearest_simhash(
        self,
        fingerprint: int,
        max_distance: int,
    ) -> Optional[NearDuplicateMatch]:
        """
        Find the closest recorded fingerprint within a Hamming distance.

        Implemented as a linear scan. Ties are broken by document identifier
        so that the result is deterministic regardless of insertion order.

        Parameters
        ----------
        fingerprint:
            Fingerprint to search for.
        max_distance:
            Largest distance considered a match.

        Returns
        -------
        Optional[NearDuplicateMatch]
            The closest match within the bound, or ``None`` if none exists.
        """
        async with self._lock:
            candidates = list(self._simhashes.items())

        best: Optional[NearDuplicateMatch] = None
        for document_id, recorded in candidates:
            distance = hamming_distance(fingerprint, recorded)
            if distance > max_distance:
                continue
            if best is None or (distance, document_id) < (
                best.distance,
                best.document_id,
            ):
                best = NearDuplicateMatch(
                    document_id=document_id, distance=distance
                )
        return best


# ============================================================================
# DEDUPLICATOR
# ============================================================================


class Deduplicator:
    """
    Decides whether a discovered document has already been ingested.

    Parameters
    ----------
    store:
        Fingerprint store. When it additionally satisfies
        ``ExtendedFingerprintStore``, version-cursor and similarity features
        become available; otherwise only exact duplicate detection operates.
        The capability is determined once here rather than per document.
    config:
        Deduplication configuration: whether detection is enabled, whether
        near duplicates are sought, the fingerprint width, the similarity
        bound, and whether unchanged versions are skipped.
    clock:
        Supplies the instant recorded as a payload's first sighting.
        Injected so that fingerprint records are deterministic under test.
    shingle_size:
        Words per similarity shingle. Belongs in ``DeduplicationConfig``;
        exposed here so that it remains injectable without amending approved
        configuration.
    """

    __slots__ = ("_store", "_config", "_clock", "_shingle_size", "_extended")

    def __init__(
        self,
        store: DocumentFingerprintStore,
        config: DeduplicationConfig,
        clock: Clock,
        *,
        shingle_size: int = _DEFAULT_SHINGLE_SIZE,
    ) -> None:
        self._store = store
        self._config = config
        self._clock = clock
        self._shingle_size = shingle_size
        self._extended: Optional[ExtendedFingerprintStore] = (
            store if isinstance(store, ExtendedFingerprintStore) else None
        )

    # ------------------------------------------------------------------
    # Capability
    # ------------------------------------------------------------------

    @property
    def supports_version_cursors(self) -> bool:
        """
        Whether unchanged-version skipping is operative.

        Requires both that the feature is configured and that the injected
        store supports version cursors.

        Returns
        -------
        bool
            ``True`` when :meth:`check_unchanged` can skip a record.
        """
        return (
            self._config.enabled
            and self._config.skip_unchanged_versions
            and self._extended is not None
        )

    @property
    def supports_near_duplicates(self) -> bool:
        """
        Whether near-duplicate detection is operative.

        Returns
        -------
        bool
            ``True`` when :meth:`check_near_duplicate` can report a match.
        """
        return (
            self._config.enabled
            and self._config.detect_near_duplicates
            and self._extended is not None
        )

    # ------------------------------------------------------------------
    # Check one: before fetch
    # ------------------------------------------------------------------

    async def check_unchanged(
        self,
        record: SourceRecord,
    ) -> Optional[DocumentSkipped]:
        """
        Determine whether a record is unchanged since it was last ingested.

        Performed BEFORE the payload is transferred. A record whose source
        version tag matches the one recorded previously is skipped without
        being fetched at all, which is the mechanism that makes repeated
        ingestion of a large corpus inexpensive.

        Parameters
        ----------
        record:
            The discovered record, before transfer.

        Returns
        -------
        Optional[DocumentSkipped]
            A skip outcome when the record is unchanged, otherwise ``None``.

        Raises
        ------
        FingerprintStoreError
            If the store cannot be read.
        """
        if not self.supports_version_cursors or record.version_tag is None:
            return None

        assert self._extended is not None
        cursor = await self._read(
            self._extended.get_version_cursor(record.source_uri),
            operation="get_version_cursor",
            source_uri=record.source_uri,
        )

        if cursor is None or cursor.version_tag != record.version_tag:
            return None

        return DocumentSkipped(
            source_uri=record.source_uri,
            reason=SkipReason.UNCHANGED_VERSION,
            existing_document_id=cursor.document_id,
        )

    # ------------------------------------------------------------------
    # Check two: after fetch
    # ------------------------------------------------------------------

    async def check_duplicate(
        self,
        blob: AcquiredBlob,
    ) -> Optional[DocumentSkipped]:
        """
        Determine whether an acquired payload has been ingested before.

        Compares the payload digest, so identical content is recognized even
        when it arrives under a different URI or carries no version tag.

        Note that lookup and record are separate operations. Two identical
        payloads processed concurrently within one run may therefore both
        pass this check; sequential and cross-run detection is exact. See
        the module documentation.

        Parameters
        ----------
        blob:
            The acquired payload.

        Returns
        -------
        Optional[DocumentSkipped]
            A skip outcome when the payload is a duplicate, otherwise
            ``None``.

        Raises
        ------
        FingerprintStoreError
            If the store cannot be read.
        """
        if not self._config.enabled:
            return None

        existing = await self._read(
            self._store.lookup(blob.content_hash),
            operation="lookup",
            source_uri=blob.source_uri,
        )
        if existing is None:
            return None

        return DocumentSkipped(
            source_uri=blob.source_uri,
            reason=SkipReason.DUPLICATE_CONTENT,
            existing_document_id=existing.document_id,
        )

    # ------------------------------------------------------------------
    # Check three: after normalization
    # ------------------------------------------------------------------

    async def check_near_duplicate(
        self,
        text: str,
    ) -> Optional[ExtractionWarning]:
        """
        Determine whether normalized text closely resembles earlier content.

        Returns a WARNING rather than a skip. The document is still
        ingested: whether to discard near duplicates depends on the purpose
        of the corpus, which is a policy decision belonging to the operator
        rather than to ingestion.

        Parameters
        ----------
        text:
            The normalized document text.

        Returns
        -------
        Optional[ExtractionWarning]
            A warning describing the match, or ``None`` when detection is
            inactive, the text is too short to fingerprint, or nothing
            similar was found.

        Raises
        ------
        DeduplicationError
            If the configured fingerprint width is invalid.
        FingerprintStoreError
            If the store cannot be searched.
        """
        if not self.supports_near_duplicates:
            return None

        fingerprint = compute_simhash(
            text,
            bits=self._config.simhash_bits,
            shingle_size=self._shingle_size,
        )
        if fingerprint is None:
            return None

        assert self._extended is not None
        match = await self._read(
            self._extended.nearest_simhash(
                fingerprint, self._config.near_duplicate_max_distance
            ),
            operation="nearest_simhash",
            source_uri=None,
        )
        if match is None:
            return None

        return ExtractionWarning(
            code=ExtractionWarningCode.NEAR_DUPLICATE_DETECTED,
            message=(
                f"Document text closely resembles previously ingested "
                f"document {match.document_id!r}; it has been retained"
            ),
            detail={
                "similar_document_id": match.document_id,
                "hamming_distance": match.distance,
                "fingerprint_bits": self._config.simhash_bits,
            },
        )

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    async def record(self, document: RawDocument) -> None:
        """
        Record a successfully ingested document for future comparison.

        Persists the payload digest, and where the store supports them, the
        source version cursor and the similarity fingerprint. Recording is
        skipped entirely when deduplication is disabled, so that no state
        accumulates for a feature that is not in use.

        Parameters
        ----------
        document:
            The canonical document that was ingested.

        Returns
        -------
        None

        Raises
        ------
        FingerprintStoreError
            If the store cannot be written.
        """
        if not self._config.enabled:
            return

        source_uri = document.source.source_uri

        await self._write(
            self._store.record(
                FingerprintRecord(
                    content_hash=document.content_hash,
                    document_id=document.document_id,
                    source_uri=source_uri,
                    first_seen_at=self._clock.now(),
                )
            ),
            operation="record",
            source_uri=source_uri,
        )

        if self._extended is None:
            return

        version_tag = document.source.version_tag
        if self._config.skip_unchanged_versions and version_tag is not None:
            await self._write(
                self._extended.set_version_cursor(
                    source_uri,
                    VersionCursor(
                        version_tag=version_tag,
                        document_id=document.document_id,
                    ),
                ),
                operation="set_version_cursor",
                source_uri=source_uri,
            )

        if not self._config.detect_near_duplicates:
            return

        fingerprint = compute_simhash(
            document.text,
            bits=self._config.simhash_bits,
            shingle_size=self._shingle_size,
        )
        if fingerprint is None:
            return

        await self._write(
            self._extended.record_simhash(document.document_id, fingerprint),
            operation="record_simhash",
            source_uri=source_uri,
        )

    # ------------------------------------------------------------------
    # Store error handling
    # ------------------------------------------------------------------

    async def _read[T](
        self,
        awaitable: object,
        *,
        operation: str,
        source_uri: Optional[str],
    ) -> T:
        """
        Await a store read, converting any failure into a PTFM exception.

        Parameters
        ----------
        awaitable:
            The store coroutine to await.
        operation:
            Name of the store operation, for diagnostics.
        source_uri:
            URI being processed, when applicable.

        Returns
        -------
        T
            The store's result.

        Raises
        ------
        FingerprintStoreError
            If the store raised. Deliberately not retryable: a degraded
            store silently disables duplicate detection and corrupts corpus
            integrity, so the condition must surface rather than be absorbed
            by a retry loop.
        """
        try:
            return await awaitable  # type: ignore[misc, no-any-return]
        except FingerprintStoreError:
            raise
        except Exception as exc:
            raise FingerprintStoreError(
                f"Fingerprint store operation {operation!r} failed: "
                f"{type(exc).__name__}: {exc}",
                operation=operation,
                source_uri=source_uri,
            ) from exc

    async def _write(
        self,
        awaitable: object,
        *,
        operation: str,
        source_uri: Optional[str],
    ) -> None:
        """
        Await a store write, converting any failure into a PTFM exception.

        Parameters
        ----------
        awaitable:
            The store coroutine to await.
        operation:
            Name of the store operation, for diagnostics.
        source_uri:
            URI being processed.

        Returns
        -------
        None

        Raises
        ------
        FingerprintStoreError
            If the store raised.
        """
        await self._read(awaitable, operation=operation, source_uri=source_uri)