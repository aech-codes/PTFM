"""Unit tests for the Data Ingestion Module contract layer."""

from __future__ import annotations

import inspect
from datetime import UTC, datetime
from typing import Any, AsyncIterator

import pytest

from ptfm.core.ingestion.exceptions import (
    ConnectorRegistrationError,
    ExtractorRegistrationError,
    ExtractorUnavailableError,
)
from ptfm.core.ingestion.interfaces import (
    MEDIA_TYPE_WILDCARD,
    Clock,
    ContentExtractor,
    DocumentFingerprintStore,
    FingerprintRecord,
    FormatDetector,
    HashProvider,
    IdGenerator,
    IncrementalHasher,
    LanguageDetection,
    LanguageDetector,
    MetricsSink,
    RetryPolicy,
    SourceConnector,
    StructuredLogger,
    ThreadOffloader,
    normalize_media_type,
)

FIXED_TIME = datetime(2026, 1, 1, tzinfo=UTC)


# ---------------------------------------------------------------------------
# Conforming reference implementations
#
# Model types are referenced only in annotations, which are strings under
# 'from __future__ import annotations'. The contract layer therefore has no
# runtime dependency on models, and these fakes need no model instances.
# ---------------------------------------------------------------------------


class _ConformingConnector(SourceConnector):
    """Minimal connector satisfying the full contract."""

    def __init__(self) -> None:
        self.close_calls = 0

    @property
    def name(self) -> str:
        return "ConformingConnector"

    @property
    def scheme(self) -> str:
        return "memory"

    async def discover(self, spec: Any) -> AsyncIterator[Any]:
        yield "record-0"
        yield "record-1"

    async def fetch(self, record: Any) -> Any:
        return f"blob:{record}"

    async def health_check(self) -> bool:
        return True

    async def close(self) -> None:
        self.close_calls += 1


class _ConformingExtractor(ContentExtractor):
    """Minimal extractor satisfying the full contract."""

    @property
    def name(self) -> str:
        return "ConformingExtractor"

    @property
    def supported_media_types(self) -> frozenset[str]:
        return frozenset({"text/plain"})

    def extract(self, payload: Any) -> Any:
        return f"extracted:{payload}"


class _WildcardSubtypeExtractor(_ConformingExtractor):
    @property
    def supported_media_types(self) -> frozenset[str]:
        return frozenset({"text/*"})


class _UniversalExtractor(_ConformingExtractor):
    @property
    def supported_media_types(self) -> frozenset[str]:
        return frozenset({MEDIA_TYPE_WILDCARD})


class _PrioritizedExtractor(_ConformingExtractor):
    @property
    def priority(self) -> int:
        return 100


class _UnavailableExtractor(_ConformingExtractor):
    @classmethod
    def check_availability(cls) -> None:
        raise ExtractorUnavailableError(
            media_type="application/pdf",
            missing_dependency="pypdfium2",
            install_extra="ptfm[pdf]",
        )


# ---------------------------------------------------------------------------
# normalize_media_type
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("text/plain", "text/plain"),
        ("Text/Plain", "text/plain"),
        ("text/plain; charset=utf-8", "text/plain"),
        ("  application/PDF  ", "application/pdf"),
        ("text/html;charset=ISO-8859-1", "text/html"),
        ("*/*", "*/*"),
    ],
)
def test_media_type_normalization(raw: str, expected: str) -> None:
    assert normalize_media_type(raw) == expected


@pytest.mark.parametrize("raw", ["", "   ", "; charset=utf-8"])
def test_media_type_normalization_rejects_malformed_input(raw: str) -> None:
    with pytest.raises(ValueError):
        normalize_media_type(raw)


# ---------------------------------------------------------------------------
# ABC enforcement
# ---------------------------------------------------------------------------


def test_abstract_classes_cannot_be_instantiated() -> None:
    with pytest.raises(TypeError):
        SourceConnector()  # type: ignore[abstract]
    with pytest.raises(TypeError):
        ContentExtractor()  # type: ignore[abstract]


def test_partial_connector_implementation_is_rejected() -> None:
    class _Partial(SourceConnector):
        @property
        def name(self) -> str:
            return "Partial"

        @property
        def scheme(self) -> str:
            return "partial"

        async def fetch(self, record: Any) -> Any:
            return record

        async def health_check(self) -> bool:
            return True

        async def close(self) -> None:
            return None

    with pytest.raises(TypeError):
        _Partial()  # type: ignore[abstract]


def test_partial_extractor_implementation_is_rejected() -> None:
    class _Partial(ContentExtractor):
        @property
        def name(self) -> str:
            return "Partial"

        def extract(self, payload: Any) -> Any:
            return payload

    with pytest.raises(TypeError):
        _Partial()  # type: ignore[abstract]


# ---------------------------------------------------------------------------
# Machine-enforced async/sync contract
# ---------------------------------------------------------------------------


def test_async_extractor_is_rejected_at_class_definition() -> None:
    """The synchronous contract is enforced, not merely documented."""
    with pytest.raises(ExtractorRegistrationError, match="must be synchronous"):

        class _AsyncExtractor(ContentExtractor):
            @property
            def name(self) -> str:
                return "AsyncExtractor"

            @property
            def supported_media_types(self) -> frozenset[str]:
                return frozenset({"text/plain"})

            async def extract(self, payload: Any) -> Any:  # type: ignore[override]
                return payload


def test_async_generator_extractor_is_rejected() -> None:
    with pytest.raises(ExtractorRegistrationError):

        class _StreamingExtractor(ContentExtractor):
            @property
            def name(self) -> str:
                return "StreamingExtractor"

            @property
            def supported_media_types(self) -> frozenset[str]:
                return frozenset({"text/plain"})

            async def extract(self, payload: Any) -> Any:  # type: ignore[override]
                yield payload


def test_coroutine_discover_is_rejected() -> None:
    """'async def discover' is a type error: the iterator must not be awaited."""
    with pytest.raises(ConnectorRegistrationError, match="async generator"):

        class _BadDiscover(_ConformingConnector):
            async def discover(self, spec: Any) -> Any:  # type: ignore[override]
                return ["record"]


@pytest.mark.parametrize("method_name", ["fetch", "health_check", "close"])
def test_synchronous_connector_methods_are_rejected(method_name: str) -> None:
    namespace: dict[str, Any] = {method_name: lambda self, *a, **k: None}

    with pytest.raises(ConnectorRegistrationError, match="coroutine function"):
        type("_BlockingConnector", (_ConformingConnector,), namespace)


def test_conforming_subclasses_define_cleanly() -> None:
    assert _ConformingConnector().scheme == "memory"
    assert _ConformingExtractor().name == "ConformingExtractor"


def test_discover_is_declared_as_an_async_generator() -> None:
    assert inspect.isasyncgenfunction(_ConformingConnector.discover)
    assert not inspect.iscoroutinefunction(SourceConnector.discover)


def test_extract_is_declared_synchronously_on_the_base() -> None:
    assert not inspect.iscoroutinefunction(ContentExtractor.extract)
    assert not inspect.isasyncgenfunction(ContentExtractor.extract)


# ---------------------------------------------------------------------------
# Connector behaviour
# ---------------------------------------------------------------------------


async def test_discover_streams_records() -> None:
    connector = _ConformingConnector()

    records = [record async for record in connector.discover(spec=None)]

    assert records == ["record-0", "record-1"]


async def test_async_context_manager_closes_connector() -> None:
    connector = _ConformingConnector()

    async with connector as entered:
        assert entered is connector
        assert connector.close_calls == 0

    assert connector.close_calls == 1


async def test_connector_is_closed_when_body_raises() -> None:
    """Resource release must survive an aborted or cancelled run."""
    connector = _ConformingConnector()

    class _RunAborted(Exception):
        pass

    with pytest.raises(_RunAborted):
        async with connector:
            raise _RunAborted

    assert connector.close_calls == 1


# ---------------------------------------------------------------------------
# Extractor resolution
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("media_type", "supported"),
    [
        ("text/plain", True),
        ("Text/Plain; charset=utf-8", True),
        ("text/html", False),
        ("application/pdf", False),
    ],
)
def test_exact_media_type_matching(media_type: str, supported: bool) -> None:
    assert _ConformingExtractor().supports(media_type) is supported


@pytest.mark.parametrize(
    ("media_type", "supported"),
    [("text/plain", True), ("text/html", True), ("application/pdf", False)],
)
def test_subtype_wildcard_matching(media_type: str, supported: bool) -> None:
    assert _WildcardSubtypeExtractor().supports(media_type) is supported


@pytest.mark.parametrize("media_type", ["text/plain", "application/pdf", "image/png"])
def test_universal_wildcard_matches_everything(media_type: str) -> None:
    assert _UniversalExtractor().supports(media_type) is True


def test_supports_rejects_malformed_media_type() -> None:
    with pytest.raises(ValueError):
        _ConformingExtractor().supports("")


def test_priority_defaults_to_zero_and_is_overridable() -> None:
    assert _ConformingExtractor().priority == 0
    assert _PrioritizedExtractor().priority == 100


def test_availability_check_defaults_to_a_no_op() -> None:
    assert _ConformingExtractor.check_availability() is None


def test_availability_failure_names_the_install_extra() -> None:
    with pytest.raises(ExtractorUnavailableError) as excinfo:
        _UnavailableExtractor.check_availability()

    assert excinfo.value.install_extra == "ptfm[pdf]"


def test_availability_is_checkable_without_instantiation() -> None:
    """The registry validates dependencies before constructing an extractor."""
    assert inspect.ismethod(_UnavailableExtractor.check_availability)


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


class _FakeClock:
    def now(self) -> datetime:
        return FIXED_TIME


class _FakeIdGenerator:
    def __init__(self) -> None:
        self._counter = 0

    def new_id(self) -> str:
        self._counter += 1
        return f"id-{self._counter}"


class _FakeIncrementalHasher:
    def __init__(self) -> None:
        self._parts: list[bytes] = []

    def update(self, chunk: bytes) -> None:
        self._parts.append(chunk)

    def hexdigest(self) -> str:
        return b"".join(self._parts).hex()


class _FakeHashProvider:
    @property
    def algorithm(self) -> str:
        return "fake-hex"

    def hash_bytes(self, data: bytes) -> str:
        return data.hex()

    def hash_text(self, text: str) -> str:
        return text.encode("utf-8").hex()

    def new_incremental(self) -> _FakeIncrementalHasher:
        return _FakeIncrementalHasher()


class _FakeDetector:
    def detect(
        self,
        header: bytes,
        *,
        filename: str | None = None,
        declared_media_type: str | None = None,
    ) -> str:
        return "text/plain"


class _FakeLanguageDetector:
    def detect(self, text: str) -> LanguageDetection | None:
        return LanguageDetection(language="en", confidence=0.99) if text else None


class _FakeFingerprintStore:
    def __init__(self) -> None:
        self._records: dict[str, FingerprintRecord] = {}

    async def lookup(self, content_hash: str) -> FingerprintRecord | None:
        return self._records.get(content_hash)

    async def record(self, record: FingerprintRecord) -> None:
        self._records[record.content_hash] = record

    async def close(self) -> None:
        self._records.clear()


class _FakeRetryPolicy:
    @property
    def max_attempts(self) -> int:
        return 3

    def delay_for(self, attempt: int) -> float:
        return 0.0 if attempt <= 1 else float(attempt)


class _InlineOffloader:
    async def run(self, func: Any, *args: Any, **kwargs: Any) -> Any:
        return func(*args, **kwargs)


class _FakeMetricsSink:
    def __init__(self) -> None:
        self.counters: list[tuple[str, int, dict[str, str]]] = []
        self.observations: list[tuple[str, float, dict[str, str]]] = []

    def increment(self, name: str, value: int = 1, **labels: str) -> None:
        self.counters.append((name, value, labels))

    def observe(self, name: str, value: float, **labels: str) -> None:
        self.observations.append((name, value, labels))


class _FakeLogger:
    def __init__(self, **context: Any) -> None:
        self.context = context
        self.events: list[tuple[str, str, dict[str, Any]]] = []

    def bind(self, **values: Any) -> _FakeLogger:
        return _FakeLogger(**{**self.context, **values})

    def debug(self, event: str, **values: Any) -> None:
        self.events.append(("debug", event, values))

    def info(self, event: str, **values: Any) -> None:
        self.events.append(("info", event, values))

    def warning(self, event: str, **values: Any) -> None:
        self.events.append(("warning", event, values))

    def error(self, event: str, **values: Any) -> None:
        self.events.append(("error", event, values))


@pytest.mark.parametrize(
    ("instance", "protocol"),
    [
        (_FakeClock(), Clock),
        (_FakeIdGenerator(), IdGenerator),
        (_FakeHashProvider(), HashProvider),
        (_FakeIncrementalHasher(), IncrementalHasher),
        (_FakeDetector(), FormatDetector),
        (_FakeLanguageDetector(), LanguageDetector),
        (_FakeFingerprintStore(), DocumentFingerprintStore),
        (_FakeRetryPolicy(), RetryPolicy),
        (_InlineOffloader(), ThreadOffloader),
        (_FakeMetricsSink(), MetricsSink),
        (_FakeLogger(), StructuredLogger),
    ],
)
def test_reference_fakes_satisfy_their_protocols(
    instance: object, protocol: type
) -> None:
    assert isinstance(instance, protocol)


def test_nonconforming_object_fails_structural_check() -> None:
    assert not isinstance(object(), Clock)


# ---------------------------------------------------------------------------
# Protocol semantics
# ---------------------------------------------------------------------------


def test_clock_must_supply_timezone_aware_instants() -> None:
    now = _FakeClock().now()

    assert now.tzinfo is not None


def test_id_generator_produces_distinct_values() -> None:
    generator = _FakeIdGenerator()

    assert generator.new_id() != generator.new_id()


def test_incremental_hashing_matches_whole_payload_hashing() -> None:
    provider = _FakeHashProvider()
    payload = b"chunk-one" b"chunk-two"

    hasher = provider.new_incremental()
    hasher.update(b"chunk-one")
    hasher.update(b"chunk-two")

    assert hasher.hexdigest() == provider.hash_bytes(payload)


def test_incremental_hashers_are_independent() -> None:
    provider = _FakeHashProvider()
    first = provider.new_incremental()
    second = provider.new_incremental()

    first.update(b"abc")

    assert second.hexdigest() != first.hexdigest()


def test_retry_policy_does_not_delay_the_first_attempt() -> None:
    policy = _FakeRetryPolicy()

    assert policy.delay_for(1) == 0.0
    assert policy.delay_for(2) > 0.0


async def test_fingerprint_store_roundtrip() -> None:
    store = _FakeFingerprintStore()
    record = FingerprintRecord(
        content_hash="hash-1",
        document_id="doc-1",
        source_uri="file:///a.txt",
        first_seen_at=FIXED_TIME,
    )

    assert await store.lookup("hash-1") is None

    await store.record(record)

    assert await store.lookup("hash-1") == record


async def test_offloader_forwards_arguments_and_result() -> None:
    offloader = _InlineOffloader()

    result = await offloader.run(lambda a, b=0: a + b, 2, b=3)

    assert result == 5


async def test_offloader_propagates_exceptions_unchanged() -> None:
    """Stage-level wrapping must remain the sole exception conversion point."""

    class _ParserFailure(Exception):
        pass

    def _boom() -> None:
        raise _ParserFailure("xref damaged")

    with pytest.raises(_ParserFailure):
        await _InlineOffloader().run(_boom)


def test_logger_bind_returns_a_logger_with_merged_context() -> None:
    logger = _FakeLogger(run_id="r1")

    bound = logger.bind(document_id="d1")

    assert isinstance(bound, StructuredLogger)
    assert bound.context == {"run_id": "r1", "document_id": "d1"}
    assert logger.context == {"run_id": "r1"}


# ---------------------------------------------------------------------------
# Anti-pattern guards
# ---------------------------------------------------------------------------


def test_language_confidence_is_not_a_document_field() -> None:
    """Detection confidence is diagnostic only and must never reach RawDocument."""
    from ptfm.core.schemas import DocumentMetadata, RawDocument

    forbidden = {"confidence", "language_confidence", "detection_confidence"}
    assert not forbidden & set(DocumentMetadata.model_fields)
    assert not forbidden & set(RawDocument.model_fields)


def test_language_detection_confidence_is_bounded() -> None:
    detection = _FakeLanguageDetector().detect("hello world")

    assert detection is not None
    assert 0.0 <= detection.confidence <= 1.0


def test_interfaces_module_has_no_third_party_runtime_imports() -> None:
    """The contract layer must not bind the Core to any concrete library."""
    import ptfm.core.ingestion.interfaces as module

    source = inspect.getsource(module)
    for forbidden in ("import structlog", "import anyio", "import httpx", "import pydantic"):
        assert forbidden not in source


def test_module_declares_no_mutable_state() -> None:
    import ptfm.core.ingestion.interfaces as module

    mutable = [
        name
        for name, value in vars(module).items()
        if not name.startswith("__") and isinstance(value, (list, dict, set))
    ]
    assert mutable == []