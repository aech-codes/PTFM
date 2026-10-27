"""
PTFM Data Ingestion Contracts

File:
    ptfm/core/ingestion/interfaces.py

Description:
    Defines every abstraction on which the Data Ingestion Module depends.
    The ingestion engine and its pipeline stages reference ONLY the types
    declared here; they never import a concrete connector, extractor,
    detector, store, or logging implementation.

Two orthogonal abstractions
---------------------------
    SourceConnector   answers "where do the bytes come from?"  (transport)
    ContentExtractor  answers "how do bytes become text?"      (media type)

These are never merged. Collapsing them yields an O(sources x formats)
implementation matrix; keeping them orthogonal yields O(sources + formats),
so every new connector inherits every existing extractor at no cost.

Execution contract
------------------
    - Connector operations are asynchronous. They perform network and file
      I/O and must never block the event loop.
    - Extractor operations are SYNCHRONOUS. Third-party parsers are
      blocking and CPU-bound; the engine guarantees they execute in a
      worker thread. Declaring them async would conceal blocking work
      behind a coroutine.

Both halves of that contract are enforced at class-definition time by
``__init_subclass__`` rather than left to documentation.

Dependency contract
-------------------
    This module imports only from the standard library and from
    ``ptfm.core.ingestion.exceptions``. Model types are imported under
    ``TYPE_CHECKING`` exclusively, so ``interfaces`` carries no runtime
    dependency on ``models``.
"""

from __future__ import annotations

import inspect
from abc import ABC, abstractmethod
from datetime import datetime
from types import TracebackType
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncIterator,
    Callable,
    Final,
    NamedTuple,
    ParamSpec,
    Protocol,
    Sequence,
    TypeVar,
    runtime_checkable,
)

from ptfm.core.ingestion.exceptions import (
    ConnectorRegistrationError,
    ExtractorRegistrationError,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ptfm.core.ingestion.models import (
        AcquiredBlob,
        DetectedFormat,
        ExtractionInput,
        ExtractionOutput,
        SourceRecord,
        SourceSpec,
    )

__all__ = [
    "MEDIA_TYPE_WILDCARD",
    "normalize_media_type",
    "SourceConnector",
    "ContentExtractor",
    "FormatDetector",
    "LanguageDetection",
    "LanguageDetector",
    "FingerprintRecord",
    "DocumentFingerprintStore",
    "IncrementalHasher",
    "HashProvider",
    "Clock",
    "IdGenerator",
    "RetryPolicy",
    "ThreadOffloader",
    "MetricsSink",
    "StructuredLogger",
]

#: Media type matching every payload. Permitted only as a last-resort
#: fallback registration, never as a primary extractor declaration.
MEDIA_TYPE_WILDCARD: Final[str] = "*/*"

_P = ParamSpec("_P")
_T = TypeVar("_T")


def normalize_media_type(media_type: str) -> str:
    """
    Normalize a media type to its canonical comparison form.

    Parameters are discarded and the type is lowercased, so that
    ``"Text/Plain; charset=UTF-8"`` and ``"text/plain"`` compare equal.
    Every registry key, extractor declaration, and detection result passes
    through this function, which is why it is declared at the contract
    layer rather than inside any one consumer.

    Parameters
    ----------
    media_type:
        Raw media type string, optionally carrying parameters.

    Returns
    -------
    str
        Lowercased ``type/subtype`` with parameters and whitespace removed.

    Raises
    ------
    ValueError
        If the input is empty or contains no type component.
    """
    candidate = media_type.split(";", 1)[0].strip().lower()
    if not candidate:
        raise ValueError(f"Invalid media type: {media_type!r}")
    return candidate


# ============================================================================
# SOURCE CONNECTORS
# ============================================================================


class SourceConnector(ABC):
    """
    Acquires raw document payloads from a single class of transport.

    A connector knows nothing about document formats. It yields bytes and
    provenance; interpretation is the extractor's responsibility. This is
    what allows an IMAP connector to be added without writing an email
    parser, and an email parser to be added without writing a connector.

    Implementation contract
    -----------------------
    - ``discover`` is declared as a plain ``def`` returning an
      ``AsyncIterator``. Implement it as an async generator::

          async def discover(self, spec: SourceSpec) -> AsyncIterator[SourceRecord]:
              for entry in ...:
                  yield SourceRecord(...)

      Declaring it ``async def`` is a type error: calling an async
      generator returns the iterator directly and must not be awaited.
    - ``fetch``, ``health_check`` and ``close`` MUST be coroutine functions.
    - All I/O must be non-blocking. Synchronous filesystem or network calls
      inside these coroutines violate the async contract and are rejected
      by the module's architecture-compliance tests.
    - Transport failures must be raised as ``SourceError`` subclasses so
      that the retry layer can consult ``retryable``.

    Both halves of the async contract are verified at class-definition time.
    """

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """
        Verify the asynchronous contract of a connector implementation.

        Only members defined directly on the subclass are inspected, so
        intermediate abstract bases are not falsely rejected.

        Parameters
        ----------
        **kwargs:
            Forwarded to :meth:`ABC.__init_subclass__`.

        Returns
        -------
        None

        Raises
        ------
        ConnectorRegistrationError
            If ``discover`` is a coroutine function, or if ``fetch``,
            ``health_check`` or ``close`` is not a coroutine function.
        """
        super().__init_subclass__(**kwargs)

        discover = cls.__dict__.get("discover")
        if discover is not None and inspect.iscoroutinefunction(discover):
            raise ConnectorRegistrationError(
                f"{cls.__name__}.discover must be an async generator function "
                f"declared as 'def ... -> AsyncIterator[SourceRecord]', not "
                f"'async def'. Calling an async generator returns the iterator "
                f"directly and must not be awaited.",
            )

        for method_name in ("fetch", "health_check", "close"):
            member = cls.__dict__.get(method_name)
            if member is None or getattr(member, "__isabstractmethod__", False):
                continue
            if not inspect.iscoroutinefunction(member):
                raise ConnectorRegistrationError(
                    f"{cls.__name__}.{method_name} must be a coroutine function "
                    f"declared with 'async def'.",
                )

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Stable identifier of this connector implementation.

        Recorded verbatim as ``SourceDescriptor.connector_name`` so that a
        document's provenance names the component that acquired it.

        Returns
        -------
        str
            Implementation identifier, for example
            ``"LocalFilesystemConnector"``.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def scheme(self) -> str:
        """
        URI scheme this connector serves.

        Used as the connector registry key. Dispatch is purely by scheme,
        which is what keeps the engine free of transport branching.

        Returns
        -------
        str
            Lowercase scheme without a separator, for example ``"file"``.
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Two-phase acquisition
    # ------------------------------------------------------------------

    @abstractmethod
    def discover(self, spec: SourceSpec) -> AsyncIterator[SourceRecord]:
        """
        Enumerate candidate documents without transferring their payloads.

        Discovery is deliberately cheap and metadata-only. Yielding size
        and version information ahead of transfer is what makes incremental
        re-ingestion possible: unchanged records are skipped without ever
        being downloaded.

        Implement as an async generator.

        Parameters
        ----------
        spec:
            Source specification: root URI, selection filters, and an
            optional incremental cursor.

        Yields
        ------
        SourceRecord
            Lightweight descriptor of one candidate document.

        Raises
        ------
        SourceError
            If the source cannot be enumerated.
        """
        raise NotImplementedError

    @abstractmethod
    async def fetch(self, record: SourceRecord) -> AcquiredBlob:
        """
        Transfer the payload for a previously discovered record.

        Implementations should spool payloads exceeding the configured
        threshold to temporary storage rather than holding them in memory.

        Parameters
        ----------
        record:
            A record previously yielded by :meth:`discover`.

        Returns
        -------
        AcquiredBlob
            The payload together with its acquisition provenance.

        Raises
        ------
        SourceNotFoundError
            If the record no longer exists at the source.
        SourceUnavailableError
            If the source is transiently unreachable.
        SourceTimeoutError
            If the transfer exceeds its deadline.
        SourceAuthenticationError
            If credentials are rejected.
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @abstractmethod
    async def health_check(self) -> bool:
        """
        Verify that the source is reachable and credentials are valid.

        Invoked before a run begins so that a misconfiguration fails fast
        rather than after partial ingestion.

        Returns
        -------
        bool
            ``True`` when the source is operational.

        Raises
        ------
        SourceError
            If reachability cannot be determined.
        """
        raise NotImplementedError

    @abstractmethod
    async def close(self) -> None:
        """
        Release transport resources held by this connector.

        Must be idempotent: repeated invocation is not an error.

        Returns
        -------
        None
        """
        raise NotImplementedError

    async def __aenter__(self) -> SourceConnector:
        """
        Enter an asynchronous context managing this connector's lifetime.

        Returns
        -------
        SourceConnector
            This connector.
        """
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """
        Close the connector on context exit, including on failure.

        Guarantees resource release even when a run aborts or is cancelled,
        which is why the engine always acquires connectors through this
        protocol rather than by direct construction.

        Parameters
        ----------
        exc_type:
            Type of the propagating exception, if any.
        exc:
            The propagating exception, if any.
        traceback:
            Traceback of the propagating exception, if any.

        Returns
        -------
        None
        """
        await self.close()


# ============================================================================
# CONTENT EXTRACTORS
# ============================================================================


class ContentExtractor(ABC):
    """
    Converts a payload of a known media type into text and structure.

    An extractor knows nothing about transport. It receives bytes and a
    resolved media type, and returns a normalized extraction result.

    Why this interface is synchronous
    ---------------------------------
    Every viable parsing library is blocking and CPU-bound. Declaring
    :meth:`extract` as ``async`` would let implementers run blocking work
    on the event loop with a coroutine wrapper concealing it. A synchronous
    signature states the truth: the method blocks, and the extraction stage
    guarantees it executes in a worker thread via the injected
    :class:`ThreadOffloader`.

    Implementation contract
    -----------------------
    - :meth:`extract` MUST NOT be a coroutine function. This is enforced at
      class-definition time.
    - :meth:`extract` MUST NOT perform network I/O, spawn an event loop, or
      mutate shared state. It must be safe to call concurrently from
      multiple threads with distinct inputs.
    - Third-party parser exceptions MUST be wrapped in ``ExtractionError``
      subclasses with the original preserved via ``raise ... from exc``.
      This is the sole boundary at which library exceptions are converted.
    - Degraded fidelity is reported as ``ExtractionWarning`` entries on the
      result, never by silently discarding content.
    """

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """
        Verify the synchronous contract of an extractor implementation.

        Parameters
        ----------
        **kwargs:
            Forwarded to :meth:`ABC.__init_subclass__`.

        Returns
        -------
        None

        Raises
        ------
        ExtractorRegistrationError
            If ``extract`` is declared as a coroutine function or an async
            generator function.
        """
        super().__init_subclass__(**kwargs)

        extract = cls.__dict__.get("extract")
        if extract is None:
            return
        if inspect.iscoroutinefunction(extract) or inspect.isasyncgenfunction(extract):
            raise ExtractorRegistrationError(
                f"{cls.__name__}.extract must be synchronous. Parsing is "
                f"blocking and CPU-bound; the extraction stage executes it in "
                f"a worker thread. An async signature would conceal blocking "
                f"work on the event loop.",
            )

    # ------------------------------------------------------------------
    # Declaration
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Stable identifier of this extractor implementation.

        Recorded on failures as ``ExtractorInternalError.extractor_name``.

        Returns
        -------
        str
            Implementation identifier, for example ``"PdfExtractor"``.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def supported_media_types(self) -> frozenset[str]:
        """
        Media types this extractor handles.

        Entries must be normalized via :func:`normalize_media_type`. A
        subtype wildcard such as ``"text/*"`` and the universal wildcard
        :data:`MEDIA_TYPE_WILDCARD` are permitted, but only as deliberate
        fallback registrations.

        Returns
        -------
        frozenset[str]
            Normalized media type patterns.
        """
        raise NotImplementedError

    @property
    def priority(self) -> int:
        """
        Resolution precedence among extractors claiming the same media type.

        Higher wins. Built-in extractors use zero, so a site-specific or
        higher-fidelity implementation can override one without forking the
        module. The registry rejects equal-priority collisions rather than
        resolving them arbitrarily, keeping resolution deterministic.

        Returns
        -------
        int
            Precedence value; defaults to ``0``.
        """
        return 0

    @classmethod
    def check_availability(cls) -> None:
        """
        Verify that this extractor's optional dependencies are installed.

        Invoked at registration so that a missing packaging extra produces
        an actionable startup error rather than an ``ImportError`` forty
        thousand documents into a run. Extractors built on the standard
        library inherit the default no-op.

        Returns
        -------
        None

        Raises
        ------
        ExtractorUnavailableError
            If a required optional dependency is absent.
        """
        return None

    # ------------------------------------------------------------------
    # Resolution
    # ------------------------------------------------------------------

    def supports(self, media_type: str) -> bool:
        """
        Report whether this extractor claims a media type.

        Matching proceeds from most to least specific: exact type, subtype
        wildcard, then universal wildcard. Implemented concretely here so
        that every extractor and the registry share identical semantics.

        Parameters
        ----------
        media_type:
            Media type to test; normalized before comparison.

        Returns
        -------
        bool
            ``True`` when this extractor claims the media type.

        Raises
        ------
        ValueError
            If the media type is malformed.
        """
        normalized = normalize_media_type(media_type)
        patterns = self.supported_media_types
        if normalized in patterns:
            return True
        primary_type = normalized.split("/", 1)[0]
        return f"{primary_type}/*" in patterns or MEDIA_TYPE_WILDCARD in patterns

    # ------------------------------------------------------------------
    # Extraction
    # ------------------------------------------------------------------

    @abstractmethod
    def extract(self, payload: ExtractionInput) -> ExtractionOutput:
        """
        Convert a payload into text and structural blocks.

        Blocking and CPU-bound by contract. Never invoke directly from a
        coroutine; the extraction stage offloads this call to a worker
        thread.

        Parameters
        ----------
        payload:
            The raw bytes together with the resolved media type, encoding,
            and originating URI.

        Returns
        -------
        ExtractionOutput
            Extracted text, structural blocks, format-native metadata, and
            any non-fatal warnings.

        Raises
        ------
        CorruptDocumentError
            If the payload is malformed and cannot be parsed.
        EncryptedDocumentError
            If the payload is password protected.
        ExtractorInternalError
            If the underlying parser raised an unanticipated exception.
        """
        raise NotImplementedError


# ============================================================================
# FORMAT AND LANGUAGE DETECTION
# ============================================================================


@runtime_checkable
class FormatDetector(Protocol):
    """
    Resolves the media type and encoding of a payload.

    Deliberately a pure function over a header buffer rather than a
    consumer of the payload itself. The acquisition stage reads the leading
    bytes and passes them here, which makes detection deterministic,
    I/O-free, exhaustively testable, and structurally incapable of becoming
    an unexpected source of blocking I/O.

    Note that ``runtime_checkable`` verifies member presence only, not
    signatures. Signature conformance is established by ``mypy --strict``.
    """

    def detect(
        self,
        header: bytes,
        *,
        filename: str | None = None,
        declared_media_type: str | None = None,
    ) -> DetectedFormat:
        """
        Resolve the media type and encoding of a payload.

        Signals are weighted from strongest to weakest: magic bytes, then
        the type declared by the source, then the filename extension.
        Disagreement between signals is reported as a warning on the
        result rather than resolved silently.

        Parameters
        ----------
        header:
            Leading bytes of the payload, typically the first few kilobytes.
        filename:
            Originating filename, when available.
        declared_media_type:
            Media type asserted by the source. Advisory only.

        Returns
        -------
        DetectedFormat
            Resolved media type, encoding, contributing signals, and any
            disagreement warnings.

        Raises
        ------
        FormatDetectionError
            If no signal yields a usable media type.
        """
        ...


class LanguageDetection(NamedTuple):
    """
    Result of natural-language identification over document text.

    Attributes
    ----------
    language:
        BCP 47 language tag, for example ``"en"``.
    confidence:
        Detector self-reported certainty in ``[0, 1]``.

        PROHIBITION. This value is a signal about which human language a
        string is written in. It is NOT a truth, verification, or
        calibration confidence, and it must never be treated as one. Its
        sole permitted consumer is the metadata harvester, which compares
        it against a configuration-injected value to decide whether to
        attach a diagnostic ``ExtractionWarning``. It must never gate,
        filter, reorder, or alter document content, must never reach the
        verification or decision path, and must never appear as a field on
        ``RawDocument`` or ``DocumentMetadata``. Raw model scores are
        prohibited outside the conformal calibration layer.
    """

    language: str
    confidence: float


@runtime_checkable
class LanguageDetector(Protocol):
    """
    Identifies the natural language of document text.

    Language is a format-generic property, not a domain concept, so
    detection belongs to the Core rather than to a plugin.
    """

    def detect(self, text: str) -> LanguageDetection | None:
        """
        Identify the dominant natural language of a text.

        Parameters
        ----------
        text:
            Normalized document text.

        Returns
        -------
        LanguageDetection | None
            The detection result, or ``None`` when the text is too short or
            too ambiguous to support any determination.
        """
        ...


# ============================================================================
# DEDUPLICATION
# ============================================================================


class FingerprintRecord(NamedTuple):
    """
    Prior sighting of a document payload.

    Attributes
    ----------
    content_hash:
        Digest of the raw payload bytes.
    document_id:
        Canonical identifier assigned when the payload was first ingested.
    source_uri:
        URI at which the payload was first seen.
    first_seen_at:
        Timezone-aware instant of first ingestion.
    """

    content_hash: str
    document_id: str
    source_uri: str
    first_seen_at: datetime


@runtime_checkable
class DocumentFingerprintStore(Protocol):
    """
    Persists payload fingerprints so that duplicates can be recognized
    across ingestion runs.

    Asynchronous because production implementations are network-backed.
    Encountering a duplicate is NOT an error: it is the expected steady
    state of incremental re-ingestion and is reported as a skipped outcome.
    Only store faults raise.
    """

    async def lookup(self, content_hash: str) -> FingerprintRecord | None:
        """
        Retrieve a prior sighting of a payload digest.

        Parameters
        ----------
        content_hash:
            Digest of the raw payload bytes.

        Returns
        -------
        FingerprintRecord | None
            The prior sighting, or ``None`` if the payload is new.

        Raises
        ------
        FingerprintStoreError
            If the store cannot be read.
        """
        ...

    async def record(self, record: FingerprintRecord) -> None:
        """
        Persist a payload sighting.

        Parameters
        ----------
        record:
            The sighting to persist.

        Returns
        -------
        None

        Raises
        ------
        FingerprintStoreError
            If the store cannot be written.
        """
        ...

    async def close(self) -> None:
        """
        Release resources held by the store.

        Must be idempotent.

        Returns
        -------
        None
        """
        ...


# ============================================================================
# HASHING
# ============================================================================


@runtime_checkable
class IncrementalHasher(Protocol):
    """
    Accumulates a digest over a payload delivered in chunks.

    Required because payloads above the spool threshold are never fully
    materialized in memory; a whole-payload hashing interface would force
    exactly the materialization the streaming design exists to avoid.
    """

    def update(self, chunk: bytes) -> None:
        """
        Incorporate the next chunk of the payload.

        Parameters
        ----------
        chunk:
            Next contiguous slice of the payload.

        Returns
        -------
        None
        """
        ...

    def hexdigest(self) -> str:
        """
        Finalize and return the accumulated digest.

        Returns
        -------
        str
            Lowercase hexadecimal digest.
        """
        ...


@runtime_checkable
class HashProvider(Protocol):
    """
    Produces content fingerprints.

    Injected rather than called directly so that tests can supply a
    deterministic stub, which is what makes the document-identity
    determinism tests possible.
    """

    @property
    def algorithm(self) -> str:
        """
        Name of the digest algorithm in use.

        Recorded alongside fingerprints so that a future algorithm change
        is a detectable, migratable event rather than a silent corruption.

        Returns
        -------
        str
            Algorithm identifier, for example ``"blake2b-256"``.
        """
        ...

    def hash_bytes(self, data: bytes) -> str:
        """
        Compute the digest of an in-memory payload.

        Parameters
        ----------
        data:
            Payload bytes.

        Returns
        -------
        str
            Lowercase hexadecimal digest.
        """
        ...

    def hash_text(self, text: str) -> str:
        """
        Compute the digest of normalized text.

        Encoding is fixed by the implementation so that the same text
        always yields the same digest regardless of the source encoding.

        Parameters
        ----------
        text:
            Normalized document text.

        Returns
        -------
        str
            Lowercase hexadecimal digest.
        """
        ...

    def new_incremental(self) -> IncrementalHasher:
        """
        Create a hasher for a chunk-delivered payload.

        Returns
        -------
        IncrementalHasher
            A fresh, independent accumulator.
        """
        ...


# ============================================================================
# TIME AND IDENTIFIERS
# ============================================================================


@runtime_checkable
class Clock(Protocol):
    """
    Supplies the current instant.

    Injected so that timestamps are controllable in tests. Without this,
    the determinism and idempotency properties of ingestion could not be
    asserted.
    """

    def now(self) -> datetime:
        """
        Return the current instant.

        Returns
        -------
        datetime
            A timezone-aware instant. Naive datetimes are a contract
            violation: they cannot be compared across the pipeline.
        """
        ...


@runtime_checkable
class IdGenerator(Protocol):
    """
    Generates non-deterministic correlation identifiers.

    PROHIBITION. This must never be used to assign ``document_id``.
    Document identity is a deterministic derivation from source URI and
    content digest via ``ptfm.core.schemas.derive_document_id``, which is
    what makes ingestion idempotent and evidence citations stable across
    index rebuilds. This generator exists solely for run and trace
    identifiers, which carry no semantic meaning.
    """

    def new_id(self) -> str:
        """
        Generate a fresh correlation identifier.

        Returns
        -------
        str
            A unique identifier.
        """
        ...


# ============================================================================
# EXECUTION POLICY
# ============================================================================


@runtime_checkable
class RetryPolicy(Protocol):
    """
    Supplies the retry schedule for transient acquisition failures.

    The policy governs timing only. Whether a failure is eligible at all is
    decided by the ``retryable`` class attribute on the raised exception,
    so no call site inspects exception types.
    """

    @property
    def max_attempts(self) -> int:
        """
        Total attempts permitted, inclusive of the first.

        Returns
        -------
        int
            A value of at least one.
        """
        ...

    def delay_for(self, attempt: int) -> float:
        """
        Compute the delay preceding a given attempt.

        Implementations should apply jitter; synchronized retries across
        concurrent workers can amplify a transient source failure into a
        sustained one.

        Parameters
        ----------
        attempt:
            One-based index of the attempt about to be made.

        Returns
        -------
        float
            Delay in seconds; zero for the first attempt.
        """
        ...


@runtime_checkable
class ThreadOffloader(Protocol):
    """
    Executes a blocking callable off the event loop.

    Injected rather than calling ``anyio.to_thread.run_sync`` directly so
    that the concurrency library stays swappable, capacity limits are
    configurable, and tests can substitute an inline runner that removes
    thread scheduling from the failure surface.
    """

    async def run(
        self,
        func: Callable[_P, _T],
        *args: _P.args,
        **kwargs: _P.kwargs,
    ) -> _T:
        """
        Run a blocking callable in a worker thread and await its result.

        Implementations must bound concurrency; an unbounded thread pool
        converts a large corpus into thread exhaustion.

        Parameters
        ----------
        func:
            The blocking callable.
        *args:
            Positional arguments forwarded to the callable.
        **kwargs:
            Keyword arguments forwarded to the callable.

        Returns
        -------
        _T
            The callable's return value.

        Raises
        ------
        Exception
            Any exception raised by the callable propagates unchanged, so
            that stage-level wrapping remains the sole conversion point.
        """
        ...


# ============================================================================
# OBSERVABILITY
# ============================================================================


@runtime_checkable
class MetricsSink(Protocol):
    """
    Receives operational counters and observations.

    Implementations MUST NOT raise and MUST NOT block. A telemetry outage
    must never fail an ingestion run; the telemetry module guarantees this
    by wrapping any injected sink.
    """

    def increment(self, name: str, value: int = 1, **labels: str) -> None:
        """
        Increment a counter.

        Parameters
        ----------
        name:
            Metric name.
        value:
            Amount to add.
        **labels:
            Dimensional labels such as ``stage`` or ``media_type``.

        Returns
        -------
        None
        """
        ...

    def observe(self, name: str, value: float, **labels: str) -> None:
        """
        Record a distribution observation such as a latency or size.

        Parameters
        ----------
        name:
            Metric name.
        value:
            Observed value.
        **labels:
            Dimensional labels.

        Returns
        -------
        None
        """
        ...


@runtime_checkable
class StructuredLogger(Protocol):
    """
    Emits machine-queryable log events.

    Declared as a Protocol so that no logging library appears in the Core
    import graph. ``structlog`` bound loggers satisfy it structurally.
    Ingestion failures are per-document and must be queryable by
    ``document_id``, ``source_uri``, and ``stage``; unstructured string
    logs are unusable at corpus scale.
    """

    def bind(self, **values: Any) -> StructuredLogger:
        """
        Return a logger carrying additional persistent context.

        Parameters
        ----------
        **values:
            Context merged into every subsequent event.

        Returns
        -------
        StructuredLogger
            A logger bound to the merged context.
        """
        ...

    def debug(self, event: str, **values: Any) -> None:
        """
        Emit a debug event.

        Parameters
        ----------
        event:
            Stable event name.
        **values:
            Structured event fields.

        Returns
        -------
        None
        """
        ...

    def info(self, event: str, **values: Any) -> None:
        """
        Emit an informational event.

        Parameters
        ----------
        event:
            Stable event name.
        **values:
            Structured event fields.

        Returns
        -------
        None
        """
        ...

    def warning(self, event: str, **values: Any) -> None:
        """
        Emit a warning event.

        Parameters
        ----------
        event:
            Stable event name.
        **values:
            Structured event fields.

        Returns
        -------
        None
        """
        ...

    def error(self, event: str, **values: Any) -> None:
        """
        Emit an error event.

        Parameters
        ----------
        event:
            Stable event name.
        **values:
            Structured event fields.

        Returns
        -------
        None
        """
        ...


def _assert_protocol_exports(names: Sequence[str]) -> None:
    """
    Verify at import time that every exported name resolves.

    Guards against an entry in ``__all__`` drifting away from an actual
    definition during refactoring, which would otherwise surface as an
    ``AttributeError`` in an unrelated module.

    Parameters
    ----------
    names:
        Exported names to verify.

    Returns
    -------
    None

    Raises
    ------
    RuntimeError
        If an exported name is not defined in this module.
    """
    missing = [name for name in names if name not in globals()]
    if missing:
        raise RuntimeError(
            f"ptfm.core.ingestion.interfaces exports undefined names: {missing}"
        )


_assert_protocol_exports(__all__)