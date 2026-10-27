"""
PTFM Acquisition Stage

File:
    ptfm/core/ingestion/pipeline/acquisition.py

Description:
    Wraps source connector operations with the concerns every connector
    would otherwise reimplement: deadlines, bounded retry, selection
    filtering, size enforcement, threshold spooling, incremental hashing,
    and conversion of transport failures into PTFM exceptions.

Two components
--------------
    AcquisitionStage
        Drives a connector on behalf of the engine. Applies deadlines and
        retry, enforces selection filters and size limits, and validates
        what the connector returned.

    PayloadBuilder
        Shared machinery a connector uses to construct an ``AcquiredBlob``.
        Handles threshold spilling to temporary storage, incremental
        hashing, streaming size enforcement, and temp file cleanup.

    The builder exists because ``SourceConnector.fetch`` returns a fully
    formed ``AcquiredBlob``, which means the connector performs the
    construction. Without shared machinery, every connector would
    independently reimplement spilling, hashing, limit enforcement, and
    cleanup, giving each four separate opportunities to get it wrong.
    Construction remains the connector's, but the mechanism is written once.

Size enforcement at three points
--------------------------------
    Declared, before transfer
        A record whose declared size exceeds the limit is rejected without
        being fetched at all.

    Streaming, during transfer
        The builder raises the moment accumulated bytes exceed the limit.
        This is the enforcement that matters: a source declaring one
        kilobyte and streaming ten gigabytes defeats both the pre-transfer
        and post-transfer checks.

    Actual, after transfer
        A final check, in case a connector constructed a blob without using
        the builder.

Retry policy
------------
    Eligibility is determined solely by the ``retryable`` class attribute on
    the raised exception. No call site inspects exception types, so adding a
    transient failure mode requires no change here.

    An exception that is not a PTFM type is treated as PERMANENT. Assuming
    transience would spend the retry budget on a deterministic failure and
    delay quarantine. Connectors are contractually obliged to raise typed
    ``SourceError`` subclasses; this handling is a safety net, and the safe
    default for a net is not to retry.

Discovery deadline
------------------
    The discovery deadline measures time spent inside the connector, not
    elapsed wall time. Wrapping the whole iteration in a timeout would
    include time the CONSUMER spent processing yielded records, so a slow
    downstream engine would trigger a spurious discovery timeout. Only time
    spent awaiting the connector is accumulated against the budget.

Dependency contract
-------------------
    Imports only the standard library, ``ptfm.core.schemas`` and the
    ingestion ``exceptions``, ``config``, ``models``, ``interfaces`` and
    ``utils.hashing`` modules.
"""

from __future__ import annotations

import asyncio
import fnmatch
import os
import tempfile
import time
from pathlib import Path
from types import TracebackType
from typing import AsyncIterator, Final, Optional
from urllib.parse import urlparse

from ptfm.core.ingestion.config import AcquisitionConfig, IngestionConfig
from ptfm.core.ingestion.exceptions import (
    AcquisitionError,
    DocumentTooLargeError,
    EmptyDocumentError,
    PTFMIngestionError,
    SourceError,
    SourceTimeoutError,
)
from ptfm.core.ingestion.interfaces import (
    Clock,
    HashProvider,
    IncrementalHasher,
    RetryPolicy,
    SourceConnector,
)
from ptfm.core.ingestion.models import (
    AcquiredBlob,
    DocumentSkipped,
    SkipReason,
    SourceRecord,
    SourceSpec,
)
from ptfm.core.ingestion.utils.hashing import Blake2bHashProvider

__all__ = [
    "matches_filters",
    "PayloadBuilder",
    "AcquisitionStage",
]

#: Prefix applied to spooled temporary files so that orphans left by an
#: abnormal termination are identifiable and safe to reap.
_SPOOL_PREFIX: Final[str] = "ptfm-ingest-"


# ============================================================================
# SELECTION FILTERING
# ============================================================================


def matches_filters(source_uri: str, spec: SourceSpec) -> bool:
    """
    Report whether a URI satisfies a specification's selection filters.

    Include patterns are applied first; an empty include list accepts
    everything. Exclude patterns are applied second and always win, so a URI
    matching both is rejected.

    Patterns are matched against the URI path with shell glob semantics, in
    which ``*`` crosses path separators. A pattern of ``*.pdf`` therefore
    matches at any depth, and ``**/*.pdf`` behaves equivalently.

    Declared here as a module function, rather than as a method, so that a
    connector may reuse it to pre-filter cheaply during discovery without
    creating a second implementation of the semantics. The stage applies it
    authoritatively regardless, so a connector that neglects to pre-filter
    cannot leak excluded documents into the corpus.

    Parameters
    ----------
    source_uri:
        URI of the candidate document.
    spec:
        The source specification supplying the patterns.

    Returns
    -------
    bool
        ``True`` when the URI should be ingested.
    """
    path = urlparse(source_uri).path or source_uri

    if spec.include_patterns and not any(
        fnmatch.fnmatch(path, pattern) for pattern in spec.include_patterns
    ):
        return False

    return not any(
        fnmatch.fnmatch(path, pattern) for pattern in spec.exclude_patterns
    )


# ============================================================================
# PAYLOAD BUILDER
# ============================================================================


class PayloadBuilder:
    """
    Accumulates a payload into an ``AcquiredBlob``, spilling to disk beyond a
    threshold.

    Blocking by contract. Every write may perform file I/O, so a connector
    must drive this from a worker thread and never from the event loop. That
    is required in any case, since reading a source is itself blocking.

    Small payloads are retained in memory. Once accumulated bytes reach the
    configured threshold, everything buffered so far is written to a
    temporary file and subsequent writes append to it, which keeps peak
    memory flat across a corpus of arbitrary size.

    The configured maximum document size is enforced on every write, so a
    source that understates its size cannot exhaust the host.

    Intended for use as a context manager, which guarantees that a temporary
    file is removed if the transfer fails or is cancelled::

        with PayloadBuilder(record, config) as builder:
            for chunk in source_chunks:
                builder.write(chunk)
            return builder.build()

    Parameters
    ----------
    record:
        The source record being fetched. Recorded on the resulting blob as
        provenance.
    config:
        Acquisition configuration supplying the size limit, the spool
        threshold, and the spool directory.
    hash_provider:
        Fingerprint provider. Defaults to BLAKE2b-256.
    clock:
        Supplies the completion instant recorded on the blob. When ``None``,
        the monotonic-independent system clock is used via
        :func:`time.time`; injecting a clock makes the instant deterministic
        under test.
    """

    __slots__ = (
        "_record",
        "_config",
        "_hasher",
        "_algorithm",
        "_clock",
        "_buffer",
        "_size",
        "_spool_path",
        "_spool_handle",
        "_built",
        "_aborted",
    )

    def __init__(
        self,
        record: SourceRecord,
        config: AcquisitionConfig,
        *,
        hash_provider: Optional[HashProvider] = None,
        clock: Optional[Clock] = None,
    ) -> None:
        provider = hash_provider or Blake2bHashProvider()
        self._record = record
        self._config = config
        self._hasher: IncrementalHasher = provider.new_incremental()
        self._algorithm = provider.algorithm
        self._clock = clock
        self._buffer: list[bytes] = []
        self._size = 0
        self._spool_path: Optional[Path] = None
        self._spool_handle: Optional[object] = None
        self._built = False
        self._aborted = False

    # ------------------------------------------------------------------
    # Accumulation
    # ------------------------------------------------------------------

    def write(self, chunk: bytes) -> None:
        """
        Append the next chunk of the payload.

        Blocking once the payload has spilled to temporary storage. The
        digest is updated incrementally, so a spooled payload is never read
        back merely to be fingerprinted.

        Parameters
        ----------
        chunk:
            Next contiguous slice of the payload. An empty chunk is ignored.

        Returns
        -------
        None

        Raises
        ------
        DocumentTooLargeError
            If accumulated bytes exceed the configured maximum. Raised as
            soon as the limit is crossed, so a source understating its size
            cannot exhaust the host.
        AcquisitionError
            If the builder has already been finalized, or a temporary file
            could not be created or written.
        """
        if self._built or self._aborted:
            raise AcquisitionError(
                "PayloadBuilder has already been finalized and cannot accept "
                "further data.",
                source_uri=self._record.source_uri,
            )
        if not chunk:
            return

        limit = self._config.max_document_bytes
        prospective = self._size + len(chunk)
        if prospective > limit:
            raise DocumentTooLargeError(
                size_bytes=prospective,
                limit_bytes=limit,
                source_uri=self._record.source_uri,
                context={"detected_during": "transfer"},
            )

        self._hasher.update(chunk)
        self._size = prospective

        if self._spool_handle is not None:
            self._append_to_spool(chunk)
            return

        self._buffer.append(chunk)
        if self._size >= self._config.spool_threshold_bytes:
            self._spill()

    def build(self) -> AcquiredBlob:
        """
        Finalize accumulation and produce the acquired payload.

        Ownership of any temporary file transfers to the returned blob,
        whose ``close`` releases it. The builder no longer removes it, so
        exiting the context manager after a successful build is safe.

        Returns
        -------
        AcquiredBlob
            The payload with its digest and acquisition provenance.

        Raises
        ------
        AcquisitionError
            If the builder has already been finalized, or the temporary file
            could not be closed.
        """
        if self._built or self._aborted:
            raise AcquisitionError(
                "PayloadBuilder has already been finalized.",
                source_uri=self._record.source_uri,
            )

        if self._spool_handle is not None:
            try:
                self._spool_handle.close()  # type: ignore[attr-defined]
            except OSError as exc:
                self._remove_spool()
                raise AcquisitionError(
                    f"Spooled payload for {self._record.source_uri!r} could "
                    f"not be closed: {exc}",
                    source_uri=self._record.source_uri,
                ) from exc
            self._spool_handle = None

        self._built = True
        acquired_at = (
            self._clock.now()
            if self._clock is not None
            else _system_instant()
        )

        if self._spool_path is not None:
            return AcquiredBlob(
                record=self._record,
                size_bytes=self._size,
                content_hash=self._hasher.hexdigest(),
                hash_algorithm=self._algorithm,
                acquired_at=acquired_at,
                spool_path=self._spool_path,
            )

        return AcquiredBlob(
            record=self._record,
            size_bytes=self._size,
            content_hash=self._hasher.hexdigest(),
            hash_algorithm=self._algorithm,
            acquired_at=acquired_at,
            data=b"".join(self._buffer),
        )

    def abort(self) -> None:
        """
        Discard accumulated data and remove any temporary file.

        Idempotent, and a no-op once :meth:`build` has succeeded, since
        ownership of the temporary file has by then transferred to the blob.

        Returns
        -------
        None
        """
        if self._built:
            return
        self._aborted = True
        self._buffer.clear()
        if self._spool_handle is not None:
            try:
                self._spool_handle.close()  # type: ignore[attr-defined]
            except OSError:
                pass
            self._spool_handle = None
        self._remove_spool()

    # ------------------------------------------------------------------
    # Context management
    # ------------------------------------------------------------------

    def __enter__(self) -> PayloadBuilder:
        """
        Enter a context managing this builder's temporary storage.

        Returns
        -------
        PayloadBuilder
            This builder.
        """
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        """
        Discard accumulated data unless the payload was successfully built.

        Guarantees that a failed or cancelled transfer leaves no temporary
        file behind.

        Parameters
        ----------
        exc_type:
            Type of the propagating exception, if any.
        exc:
            The propagating exception, if any.
        tb:
            Traceback of the propagating exception, if any.

        Returns
        -------
        None
        """
        self.abort()

    # ------------------------------------------------------------------
    # Spooling
    # ------------------------------------------------------------------

    def _spill(self) -> None:
        """
        Move buffered data to a temporary file and switch to appending.

        Returns
        -------
        None

        Raises
        ------
        AcquisitionError
            If the temporary file could not be created or written.
        """
        directory = self._config.spool_directory
        try:
            descriptor, path = tempfile.mkstemp(
                prefix=_SPOOL_PREFIX, dir=directory
            )
            handle = os.fdopen(descriptor, "wb")
        except OSError as exc:
            raise AcquisitionError(
                f"Temporary storage for {self._record.source_uri!r} could not "
                f"be created in {directory or 'the system default location'}: "
                f"{exc}",
                source_uri=self._record.source_uri,
                context={"spool_directory": directory},
            ) from exc

        self._spool_path = Path(path)
        self._spool_handle = handle

        buffered = b"".join(self._buffer)
        self._buffer.clear()
        self._append_to_spool(buffered)

    def _append_to_spool(self, chunk: bytes) -> None:
        """
        Append a chunk to the temporary file.

        Parameters
        ----------
        chunk:
            Bytes to append.

        Returns
        -------
        None

        Raises
        ------
        AcquisitionError
            If the write failed. The temporary file is removed first, so a
            full disk does not leave an orphan behind.
        """
        assert self._spool_handle is not None
        try:
            self._spool_handle.write(chunk)  # type: ignore[attr-defined]
        except OSError as exc:
            self.abort()
            raise AcquisitionError(
                f"Spooled payload for {self._record.source_uri!r} could not be "
                f"written: {exc}",
                source_uri=self._record.source_uri,
            ) from exc

    def _remove_spool(self) -> None:
        """
        Delete the temporary file if one exists.

        Missing files are tolerated so that cleanup during a cancelled run
        cannot itself raise.

        Returns
        -------
        None
        """
        if self._spool_path is None:
            return
        self._spool_path.unlink(missing_ok=True)
        self._spool_path = None

    def __repr__(self) -> str:
        """
        Return a representation that never includes payload bytes.

        Returns
        -------
        str
            Safe representation for logs and error messages.
        """
        location = "spooled" if self._spool_path is not None else "memory"
        return (
            f"PayloadBuilder(source_uri={self._record.source_uri!r}, "
            f"size_bytes={self._size}, storage={location!r})"
        )


def _system_instant() -> "datetime":  # noqa: F821 - annotation resolved lazily
    """
    Return the current instant as a timezone-aware UTC datetime.

    Used only when no clock is injected into a builder. Defined as a
    function so that no module-level import of the wall clock is required at
    a point where an injected clock is the norm.

    Returns
    -------
    datetime
        The current instant in UTC.
    """
    from datetime import UTC, datetime

    return datetime.now(UTC)


# ============================================================================
# ACQUISITION STAGE
# ============================================================================


class AcquisitionStage:
    """
    Drives a source connector on behalf of the ingestion engine.

    Parameters
    ----------
    config:
        Acquisition configuration: size limits, deadlines, and spooling
        behaviour.
    retry_policy:
        Supplies the retry schedule. Whether a failure is eligible at all is
        decided by the exception, never by this policy.
    hash_provider:
        Fingerprint provider, used to verify that a connector fingerprinted
        its payload under the expected algorithm. Digests computed under
        different algorithms are incomparable, which would silently disable
        cross-connector duplicate detection.
    """

    __slots__ = ("_config", "_retry_policy", "_algorithm")

    def __init__(
        self,
        config: AcquisitionConfig,
        retry_policy: RetryPolicy,
        *,
        hash_provider: Optional[HashProvider] = None,
    ) -> None:
        self._config = config
        self._retry_policy = retry_policy
        self._algorithm = (hash_provider or Blake2bHashProvider()).algorithm

    @classmethod
    def from_config(
        cls,
        config: IngestionConfig,
        *,
        hash_provider: Optional[HashProvider] = None,
    ) -> AcquisitionStage:
        """
        Construct a stage from the full ingestion configuration.

        Convenience for the composition root, which holds the complete
        configuration while this stage requires only part of it.

        Parameters
        ----------
        config:
            The complete ingestion configuration.
        hash_provider:
            Fingerprint provider used for algorithm verification.

        Returns
        -------
        AcquisitionStage
            A stage bound to the acquisition configuration and the retry
            schedule it declares.
        """
        return cls(
            config.acquisition,
            config.build_retry_policy(),
            hash_provider=hash_provider,
        )

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------

    async def discover(
        self,
        connector: SourceConnector,
        spec: SourceSpec,
    ) -> AsyncIterator[SourceRecord]:
        """
        Enumerate candidate documents, applying filters and the document cap.

        The deadline bounds time spent INSIDE the connector, accumulated
        across successive records. Time the consumer spends processing a
        yielded record is excluded, so a slow downstream engine cannot cause
        a spurious discovery timeout.

        Selection filters are applied here authoritatively. A connector may
        additionally pre-filter for efficiency, but a connector that does not
        cannot thereby leak excluded documents.

        Parameters
        ----------
        connector:
            The connector to enumerate.
        spec:
            Source specification supplying filters and the optional cap.

        Yields
        ------
        SourceRecord
            Each candidate document that passes the filters, up to the cap.

        Raises
        ------
        SourceTimeoutError
            If accumulated connector time exceeds the discovery deadline.
        SourceError
            If enumeration failed for any other reason.
        """
        budget = self._config.discovery_timeout_seconds
        spent = 0.0
        yielded = 0
        iterator = connector.discover(spec).__aiter__()

        while True:
            remaining = budget - spent
            if remaining <= 0.0:
                raise SourceTimeoutError(
                    f"Discovery of {spec.root_uri!r} exceeded the {budget} "
                    f"second budget of connector time after yielding "
                    f"{yielded} records.",
                    timeout_seconds=budget,
                    source_uri=spec.root_uri,
                    context={"records_yielded": yielded},
                )

            started = time.monotonic()
            try:
                record = await asyncio.wait_for(
                    iterator.__anext__(), timeout=remaining
                )
            except StopAsyncIteration:
                return
            except TimeoutError as exc:
                raise SourceTimeoutError(
                    f"Discovery of {spec.root_uri!r} exceeded the {budget} "
                    f"second budget of connector time after yielding "
                    f"{yielded} records.",
                    timeout_seconds=budget,
                    source_uri=spec.root_uri,
                    context={"records_yielded": yielded},
                ) from exc
            except PTFMIngestionError as error:
                error.add_context(records_yielded=yielded)
                raise
            except Exception as exc:
                raise SourceError(
                    f"Discovery of {spec.root_uri!r} by "
                    f"{connector.name!r} failed with "
                    f"{type(exc).__name__}: {exc}",
                    source_uri=spec.root_uri,
                    context={
                        "connector_name": connector.name,
                        "records_yielded": yielded,
                        "original_exception": type(exc).__name__,
                    },
                ) from exc
            finally:
                spent += time.monotonic() - started

            if not matches_filters(record.source_uri, spec):
                continue

            yield record
            yielded += 1

            if spec.max_documents is not None and yielded >= spec.max_documents:
                return

    def excluded(self, record: SourceRecord, spec: SourceSpec) -> Optional[DocumentSkipped]:
        """
        Report a record rejected by the specification's selection filters.

        Provided so that a caller enumerating records outside
        :meth:`discover` can produce the same skip outcome. Exclusion is a
        normal result, not a fault.

        Parameters
        ----------
        record:
            The candidate record.
        spec:
            The source specification.

        Returns
        -------
        Optional[DocumentSkipped]
            A skip outcome when the record is excluded, otherwise ``None``.
        """
        if matches_filters(record.source_uri, spec):
            return None
        return DocumentSkipped(
            source_uri=record.source_uri,
            reason=SkipReason.EXCLUDED_BY_FILTER,
        )

    # ------------------------------------------------------------------
    # Fetching
    # ------------------------------------------------------------------

    async def fetch(
        self,
        connector: SourceConnector,
        record: SourceRecord,
    ) -> AcquiredBlob:
        """
        Transfer a record's payload, retrying transient failures.

        A record whose declared size exceeds the configured limit is
        rejected before any transfer is attempted. Each attempt is bounded
        by the fetch deadline, and retries are governed by the injected
        schedule with eligibility taken from the raised exception.

        Parameters
        ----------
        connector:
            The connector to fetch from.
        record:
            A record previously yielded by discovery.

        Returns
        -------
        AcquiredBlob
            The validated payload. The caller owns it and is responsible for
            closing it.

        Raises
        ------
        DocumentTooLargeError
            If the declared or actual size exceeds the configured limit.
        EmptyDocumentError
            If the payload contains no bytes.
        SourceTimeoutError
            If every attempt exceeded the fetch deadline.
        SourceError
            If the transfer failed for any other reason.
        AcquisitionError
            If the connector returned a payload that does not correspond to
            the requested record, or fingerprinted it under an unexpected
            algorithm.
        """
        self._enforce_declared_size(record)

        max_attempts = self._retry_policy.max_attempts
        attempt = 0

        while True:
            attempt += 1

            delay = self._retry_policy.delay_for(attempt)
            if delay > 0.0:
                await asyncio.sleep(delay)

            try:
                blob = await asyncio.wait_for(
                    connector.fetch(record),
                    timeout=self._config.fetch_timeout_seconds,
                )
            except TimeoutError as exc:
                error: PTFMIngestionError = SourceTimeoutError(
                    f"Transfer of {record.source_uri!r} exceeded the "
                    f"{self._config.fetch_timeout_seconds} second deadline.",
                    timeout_seconds=self._config.fetch_timeout_seconds,
                    source_uri=record.source_uri,
                )
                error.__cause__ = exc
            except PTFMIngestionError as raised:
                if raised.source_uri is None:
                    raised.source_uri = record.source_uri
                error = raised
            except Exception as exc:
                # Unrecognized failures are treated as permanent. Assuming
                # transience would spend the retry budget on a deterministic
                # failure and delay quarantine.
                error = SourceError(
                    f"Transfer of {record.source_uri!r} by "
                    f"{connector.name!r} failed with "
                    f"{type(exc).__name__}: {exc}",
                    source_uri=record.source_uri,
                    context={
                        "connector_name": connector.name,
                        "original_exception": type(exc).__name__,
                    },
                )
                error.__cause__ = exc
            else:
                return self._validate(blob, record, connector)

            if error.retryable and attempt < max_attempts:
                continue

            error.add_context(attempts=attempt, max_attempts=max_attempts)
            raise error

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _enforce_declared_size(self, record: SourceRecord) -> None:
        """
        Reject a record whose declared size exceeds the configured limit.

        Applied before any transfer, so that an oversized document is never
        downloaded. A record declaring no size is transferred and checked
        during accumulation instead.

        Parameters
        ----------
        record:
            The candidate record.

        Returns
        -------
        None

        Raises
        ------
        DocumentTooLargeError
            If the declared size exceeds the limit.
        """
        declared = record.size_bytes
        limit = self._config.max_document_bytes
        if declared is None or declared <= limit:
            return

        raise DocumentTooLargeError(
            size_bytes=declared,
            limit_bytes=limit,
            source_uri=record.source_uri,
            context={"detected_during": "discovery"},
        )

    def _validate(
        self,
        blob: AcquiredBlob,
        record: SourceRecord,
        connector: SourceConnector,
    ) -> AcquiredBlob:
        """
        Verify a transferred payload, releasing it if it is unacceptable.

        A rejected payload is closed before the exception propagates. Without
        this, every rejected document would leak a temporary file, which
        remains invisible until the disk fills.

        Parameters
        ----------
        blob:
            The payload returned by the connector.
        record:
            The record that was requested.
        connector:
            The connector that produced the payload.

        Returns
        -------
        AcquiredBlob
            The payload, unchanged, when acceptable.

        Raises
        ------
        EmptyDocumentError
            If the payload contains no bytes.
        DocumentTooLargeError
            If the payload exceeds the configured limit.
        AcquisitionError
            If the payload does not correspond to the requested record, or
            was fingerprinted under an unexpected algorithm.
        """
        try:
            if blob.record.source_uri != record.source_uri:
                raise AcquisitionError(
                    f"Connector {connector.name!r} was asked for "
                    f"{record.source_uri!r} but returned a payload for "
                    f"{blob.record.source_uri!r}. Provenance would be "
                    f"incorrect.",
                    source_uri=record.source_uri,
                    context={
                        "requested_uri": record.source_uri,
                        "returned_uri": blob.record.source_uri,
                        "connector_name": connector.name,
                    },
                )

            if blob.size_bytes == 0:
                raise EmptyDocumentError(
                    f"Payload for {record.source_uri!r} contains no bytes, "
                    f"which ordinarily indicates a truncated transfer or a "
                    f"permissions problem at the source.",
                    source_uri=record.source_uri,
                )

            limit = self._config.max_document_bytes
            if blob.size_bytes > limit:
                raise DocumentTooLargeError(
                    size_bytes=blob.size_bytes,
                    limit_bytes=limit,
                    source_uri=record.source_uri,
                    context={"detected_during": "validation"},
                )

            if blob.hash_algorithm != self._algorithm:
                raise AcquisitionError(
                    f"Connector {connector.name!r} fingerprinted "
                    f"{record.source_uri!r} using "
                    f"{blob.hash_algorithm!r}, but {self._algorithm!r} is "
                    f"configured. Digests computed under differing algorithms "
                    f"are incomparable, which would silently disable "
                    f"duplicate detection.",
                    source_uri=record.source_uri,
                    context={
                        "blob_algorithm": blob.hash_algorithm,
                        "expected_algorithm": self._algorithm,
                        "connector_name": connector.name,
                    },
                )
        except PTFMIngestionError:
            blob.close()
            raise

        return blob