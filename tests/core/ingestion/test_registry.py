"""Unit tests for the Data Ingestion Module registries."""

from __future__ import annotations

from typing import Any, AsyncIterator, Sequence

import pytest

from ptfm.core.ingestion import registry as registry_module
from ptfm.core.ingestion.config import ExtractionConfig, IngestionConfig
from ptfm.core.ingestion.exceptions import (
    ConnectorRegistrationError,
    ExtractorRegistrationError,
    ExtractorUnavailableError,
    UnsupportedMediaTypeError,
    UnsupportedSourceSchemeError,
)
from ptfm.core.ingestion.interfaces import (
    MEDIA_TYPE_WILDCARD,
    ContentExtractor,
    SourceConnector,
)
from ptfm.core.ingestion.registry import (
    CONNECTOR_ENTRY_POINT_GROUP,
    EXTRACTOR_ENTRY_POINT_GROUP,
    ConnectorRegistration,
    ConnectorRegistry,
    ExtractorRegistry,
)


def _config() -> IngestionConfig:
    return IngestionConfig(_env_file=None)


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class _Connector(SourceConnector):
    """Connector whose scheme is supplied at construction."""

    def __init__(self, scheme: str = "memory", name: str = "MemoryConnector") -> None:
        self._scheme = scheme
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    @property
    def scheme(self) -> str:
        return self._scheme

    async def discover(self, spec: Any) -> AsyncIterator[Any]:
        yield "record"

    async def fetch(self, record: Any) -> Any:
        return record

    async def health_check(self) -> bool:
        return True

    async def close(self) -> None:
        return None


class _Extractor(ContentExtractor):
    """Extractor with configurable name, patterns, and priority."""

    def __init__(
        self,
        name: str,
        patterns: Sequence[str],
        priority: int = 0,
    ) -> None:
        self._name = name
        self._patterns = frozenset(patterns)
        self._priority = priority

    @property
    def name(self) -> str:
        return self._name

    @property
    def supported_media_types(self) -> frozenset[str]:
        return self._patterns

    @property
    def priority(self) -> int:
        return self._priority

    def extract(self, payload: Any) -> Any:
        return payload


class _UnavailableExtractor(_Extractor):
    @classmethod
    def check_availability(cls) -> None:
        raise ExtractorUnavailableError(
            media_type="application/pdf",
            missing_dependency="pypdfium2",
            install_extra="ptfm[pdf]",
        )


def _registry(*extractors: ContentExtractor, **config: Any) -> ExtractorRegistry:
    return ExtractorRegistry(extractors, config=ExtractionConfig(**config))


# ---------------------------------------------------------------------------
# Connector registration
# ---------------------------------------------------------------------------


def test_registered_scheme_resolves_to_a_connector() -> None:
    registry = ConnectorRegistry(
        [ConnectorRegistration(scheme="memory", factory=lambda cfg: _Connector())]
    )

    connector = registry.create("memory", _config())

    assert isinstance(connector, SourceConnector)
    assert registry.supported_schemes == ("memory",)
    assert len(registry) == 1


def test_scheme_resolution_is_case_insensitive() -> None:
    registry = ConnectorRegistry(
        [ConnectorRegistration(scheme="FILE", factory=lambda cfg: _Connector("file"))]
    )

    assert registry.supports("file")
    assert registry.create("FiLe", _config()).scheme == "file"


@pytest.mark.parametrize("scheme", ["", "1file", "file://", "fi le", "http:"])
def test_malformed_schemes_are_rejected(scheme: str) -> None:
    with pytest.raises(ConnectorRegistrationError, match="Invalid URI scheme"):
        ConnectorRegistry(
            [ConnectorRegistration(scheme=scheme, factory=lambda cfg: _Connector())]
        )


def test_duplicate_scheme_registration_is_rejected() -> None:
    with pytest.raises(ConnectorRegistrationError, match="Duplicate connector"):
        ConnectorRegistry(
            [
                ConnectorRegistration(scheme="file", factory=lambda cfg: _Connector("file")),
                ConnectorRegistration(scheme="file", factory=lambda cfg: _Connector("file")),
            ]
        )


def test_non_callable_factory_is_rejected() -> None:
    with pytest.raises(ConnectorRegistrationError, match="not callable"):
        ConnectorRegistry(
            [ConnectorRegistration(scheme="file", factory="not-a-factory")]  # type: ignore[arg-type]
        )


# ---------------------------------------------------------------------------
# Connector resolution
# ---------------------------------------------------------------------------


def test_unregistered_scheme_lists_what_is_available() -> None:
    registry = ConnectorRegistry(
        [
            ConnectorRegistration(scheme="file", factory=lambda cfg: _Connector("file")),
            ConnectorRegistration(scheme="memory", factory=lambda cfg: _Connector()),
        ]
    )

    with pytest.raises(UnsupportedSourceSchemeError) as excinfo:
        registry.create("s3", _config())

    assert excinfo.value.supported_schemes == ("file", "memory")
    assert "file, memory" in excinfo.value.message


def test_empty_registry_reports_no_schemes() -> None:
    with pytest.raises(UnsupportedSourceSchemeError, match="none"):
        ConnectorRegistry([]).create("file", _config())


def test_factory_receives_the_injected_configuration() -> None:
    seen: list[IngestionConfig] = []

    def _factory(config: IngestionConfig) -> SourceConnector:
        seen.append(config)
        return _Connector()

    config = _config()
    ConnectorRegistry(
        [ConnectorRegistration(scheme="memory", factory=_factory)]
    ).create("memory", config)

    assert seen == [config]


def test_each_call_produces_a_fresh_connector() -> None:
    """Connectors own transport resources with a per-run lifecycle."""
    registry = ConnectorRegistry(
        [ConnectorRegistration(scheme="memory", factory=lambda cfg: _Connector())]
    )

    assert registry.create("memory", _config()) is not registry.create("memory", _config())


def test_failing_factory_is_wrapped_in_a_ptfm_exception() -> None:
    def _factory(config: IngestionConfig) -> SourceConnector:
        raise RuntimeError("credentials missing")

    registry = ConnectorRegistry(
        [ConnectorRegistration(scheme="s3", factory=_factory)]
    )

    with pytest.raises(ConnectorRegistrationError) as excinfo:
        registry.create("s3", _config())

    assert isinstance(excinfo.value.__cause__, RuntimeError)
    assert "credentials missing" in excinfo.value.message


def test_factory_returning_a_non_connector_is_rejected() -> None:
    registry = ConnectorRegistry(
        [ConnectorRegistration(scheme="memory", factory=lambda cfg: object())]  # type: ignore[return-value]
    )

    with pytest.raises(ConnectorRegistrationError, match="does not implement"):
        registry.create("memory", _config())


def test_scheme_drift_between_registration_and_connector_is_detected() -> None:
    """Registered scheme and declared scheme must agree, or dispatch lies."""
    registry = ConnectorRegistry(
        [ConnectorRegistration(scheme="file", factory=lambda cfg: _Connector("s3"))]
    )

    with pytest.raises(ConnectorRegistrationError, match="declares scheme"):
        registry.create("file", _config())


def test_describe_reports_registration_origin() -> None:
    registry = ConnectorRegistry(
        [
            ConnectorRegistration(
                scheme="memory", factory=lambda cfg: _Connector(), origin="explicit"
            )
        ]
    )

    assert registry.describe() == {"memory": "explicit"}


# ---------------------------------------------------------------------------
# Extractor resolution: specificity
# ---------------------------------------------------------------------------


def test_exact_media_type_resolves() -> None:
    pdf = _Extractor("PdfExtractor", ["application/pdf"])

    assert _registry(pdf).resolve("application/pdf") is pdf


def test_resolution_normalizes_parameters_and_casing() -> None:
    text = _Extractor("TextExtractor", ["text/plain"])
    registry = _registry(text)

    assert registry.resolve("Text/Plain; charset=UTF-8") is text


def test_subtype_wildcard_resolves_unclaimed_subtypes() -> None:
    fallback = _Extractor("TextFallback", ["text/*"])

    assert _registry(fallback).resolve("text/csv") is fallback


def test_universal_wildcard_resolves_anything() -> None:
    universal = _Extractor("Universal", [MEDIA_TYPE_WILDCARD])

    assert _registry(universal).resolve("image/png") is universal


def test_specificity_dominates_priority() -> None:
    """A high-priority fallback must never shadow a purpose-built extractor."""
    exact = _Extractor("TextExtractor", ["text/plain"], priority=0)
    universal = _Extractor("Universal", [MEDIA_TYPE_WILDCARD], priority=100)

    registry = _registry(exact, universal)

    assert registry.resolve("text/plain") is exact
    assert registry.resolve("image/png") is universal


def test_subtype_wildcard_outranks_universal_wildcard() -> None:
    subtype = _Extractor("TextFallback", ["text/*"], priority=0)
    universal = _Extractor("Universal", [MEDIA_TYPE_WILDCARD], priority=50)

    assert _registry(subtype, universal).resolve("text/csv") is subtype


def test_unresolvable_media_type_lists_registered_patterns() -> None:
    registry = _registry(
        _Extractor("PdfExtractor", ["application/pdf"]),
        _Extractor("TextFallback", ["text/*"]),
    )

    with pytest.raises(UnsupportedMediaTypeError) as excinfo:
        registry.resolve("image/png")

    assert excinfo.value.supported_media_types == ("application/pdf", "text/*")


def test_resolve_or_none_returns_none_rather_than_raising() -> None:
    registry = _registry(_Extractor("PdfExtractor", ["application/pdf"]))

    assert registry.resolve_or_none("image/png") is None
    assert registry.supports("application/pdf") is True
    assert registry.supports("image/png") is False


# ---------------------------------------------------------------------------
# Extractor resolution: priority
# ---------------------------------------------------------------------------


def test_higher_priority_wins_within_a_tier() -> None:
    builtin = _Extractor("BuiltinPdf", ["application/pdf"], priority=0)
    override = _Extractor("VendorPdf", ["application/pdf"], priority=10)

    assert _registry(builtin, override).resolve("application/pdf") is override
    assert _registry(override, builtin).resolve("application/pdf") is override


def test_equal_priority_collision_is_rejected() -> None:
    """Insertion-order resolution would make dispatch depend on import order."""
    first = _Extractor("FirstPdf", ["application/pdf"])
    second = _Extractor("SecondPdf", ["application/pdf"])

    with pytest.raises(ExtractorRegistrationError, match="both claim"):
        _registry(first, second)


def test_wildcard_collision_at_equal_priority_is_rejected() -> None:
    with pytest.raises(ExtractorRegistrationError, match="both claim"):
        _registry(
            _Extractor("FallbackA", ["text/*"]),
            _Extractor("FallbackB", ["text/*"]),
        )


def test_registration_order_does_not_affect_resolution() -> None:
    a = _Extractor("A", ["text/plain"], priority=1)
    b = _Extractor("B", ["text/*"], priority=9)
    c = _Extractor("C", [MEDIA_TYPE_WILDCARD], priority=99)

    forward = _registry(a, b, c)
    reverse = _registry(c, b, a)

    for media_type in ("text/plain", "text/html", "image/png"):
        assert forward.resolve(media_type).name == reverse.resolve(media_type).name


# ---------------------------------------------------------------------------
# Extractor declaration validation
# ---------------------------------------------------------------------------


def test_extractor_declaring_no_media_types_is_rejected() -> None:
    with pytest.raises(ExtractorRegistrationError, match="declares no supported"):
        _registry(_Extractor("Empty", []))


@pytest.mark.parametrize("pattern", ["*/plain", "text/pl*in", "plaintext"])
def test_unsupported_pattern_forms_are_rejected(pattern: str) -> None:
    with pytest.raises(ExtractorRegistrationError):
        _registry(_Extractor("Bad", [pattern]))


def test_duplicate_extractor_names_are_rejected() -> None:
    with pytest.raises(ExtractorRegistrationError, match="Duplicate extractor name"):
        _registry(
            _Extractor("Same", ["text/plain"]),
            _Extractor("Same", ["application/pdf"]),
        )


def test_malformed_media_type_at_resolution_is_rejected() -> None:
    with pytest.raises(ExtractorRegistrationError, match="malformed"):
        _registry(_Extractor("Text", ["text/plain"])).resolve("")


def test_one_extractor_may_claim_several_patterns() -> None:
    multi = _Extractor("Office", ["application/vnd.ms-excel", "text/csv"])
    registry = _registry(multi)

    assert registry.resolve("text/csv") is multi
    assert registry.resolve("application/vnd.ms-excel") is multi


# ---------------------------------------------------------------------------
# Availability and exclusion
# ---------------------------------------------------------------------------


def test_strict_availability_aborts_on_a_missing_dependency() -> None:
    with pytest.raises(ExtractorUnavailableError) as excinfo:
        _registry(
            _UnavailableExtractor("PdfExtractor", ["application/pdf"]),
            strict_availability=True,
        )

    assert excinfo.value.install_extra == "ptfm[pdf]"


def test_lax_availability_skips_and_records_the_reason() -> None:
    registry = _registry(
        _Extractor("TextExtractor", ["text/plain"]),
        _UnavailableExtractor("PdfExtractor", ["application/pdf"]),
        strict_availability=False,
    )

    assert [s.extractor_name for s in registry.skipped_extractors] == ["PdfExtractor"]
    assert "pypdfium2" in registry.skipped_extractors[0].reason
    assert registry.resolve_or_none("application/pdf") is None
    assert registry.resolve("text/plain").name == "TextExtractor"


def test_disabled_extractors_are_excluded() -> None:
    registry = _registry(
        _Extractor("TextExtractor", ["text/plain"]),
        _Extractor("PdfExtractor", ["application/pdf"]),
        disabled_extractors=frozenset({"PdfExtractor"}),
    )

    assert len(registry) == 1
    assert registry.resolve_or_none("application/pdf") is None
    assert [s.reason for s in registry.skipped_extractors] == ["disabled by configuration"]


def test_disabling_does_not_trigger_an_availability_check() -> None:
    """A disabled extractor's missing dependency must not abort startup."""
    registry = _registry(
        _UnavailableExtractor("PdfExtractor", ["application/pdf"]),
        disabled_extractors=frozenset({"PdfExtractor"}),
        strict_availability=True,
    )

    assert len(registry) == 0


def test_disabling_frees_a_pattern_for_a_lower_priority_extractor() -> None:
    registry = _registry(
        _Extractor("VendorPdf", ["application/pdf"], priority=10),
        _Extractor("BuiltinPdf", ["application/pdf"], priority=0),
        disabled_extractors=frozenset({"VendorPdf"}),
    )

    assert registry.resolve("application/pdf").name == "BuiltinPdf"


def test_describe_maps_patterns_to_winning_extractors() -> None:
    registry = _registry(
        _Extractor("PdfExtractor", ["application/pdf"]),
        _Extractor("TextFallback", ["text/*"]),
    )

    assert registry.describe() == {
        "application/pdf": "PdfExtractor",
        "text/*": "TextFallback",
    }


# ---------------------------------------------------------------------------
# Entry-point discovery
# ---------------------------------------------------------------------------


class _FakeEntryPoint:
    """Stand-in for importlib.metadata.EntryPoint."""

    def __init__(self, name: str, group: str, loader: Any) -> None:
        self.name = name
        self.group = group
        self._loader = loader

    def load(self) -> Any:
        return self._loader()


def _patch_entry_points(
    monkeypatch: pytest.MonkeyPatch, points: Sequence[_FakeEntryPoint]
) -> None:
    monkeypatch.setattr(
        registry_module,
        "entry_points",
        lambda group: [p for p in points if p.group == group],
    )


def test_connector_entry_points_are_discovered(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_entry_points(
        monkeypatch,
        [
            _FakeEntryPoint(
                "s3",
                CONNECTOR_ENTRY_POINT_GROUP,
                lambda: (lambda cfg: _Connector("s3", "S3Connector")),
            )
        ],
    )

    registry = ConnectorRegistry.from_entry_points()

    assert registry.supported_schemes == ("s3",)
    assert registry.describe()["s3"] == "entry_point:s3"


def test_explicit_registrations_combine_with_discovered_ones(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_entry_points(
        monkeypatch,
        [
            _FakeEntryPoint(
                "s3",
                CONNECTOR_ENTRY_POINT_GROUP,
                lambda: (lambda cfg: _Connector("s3")),
            )
        ],
    )

    registry = ConnectorRegistry.from_entry_points(
        additional=[
            ConnectorRegistration(scheme="file", factory=lambda cfg: _Connector("file"))
        ]
    )

    assert registry.supported_schemes == ("file", "s3")


def test_entry_point_colliding_with_an_explicit_scheme_is_an_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A third-party package must not silently displace a built-in connector."""
    _patch_entry_points(
        monkeypatch,
        [
            _FakeEntryPoint(
                "file",
                CONNECTOR_ENTRY_POINT_GROUP,
                lambda: (lambda cfg: _Connector("file")),
            )
        ],
    )

    with pytest.raises(ConnectorRegistrationError, match="Duplicate connector"):
        ConnectorRegistry.from_entry_points(
            additional=[
                ConnectorRegistration(
                    scheme="file", factory=lambda cfg: _Connector("file")
                )
            ]
        )


def test_broken_entry_point_names_itself(monkeypatch: pytest.MonkeyPatch) -> None:
    def _explode() -> Any:
        raise ImportError("missing transitive dependency")

    _patch_entry_points(
        monkeypatch, [_FakeEntryPoint("s3", CONNECTOR_ENTRY_POINT_GROUP, _explode)]
    )

    with pytest.raises(ConnectorRegistrationError) as excinfo:
        ConnectorRegistry.from_entry_points()

    assert "s3" in excinfo.value.message
    assert isinstance(excinfo.value.__cause__, ImportError)


def test_non_callable_entry_point_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_entry_points(
        monkeypatch,
        [_FakeEntryPoint("s3", CONNECTOR_ENTRY_POINT_GROUP, lambda: "not-callable")],
    )

    with pytest.raises(ConnectorRegistrationError, match="not a callable"):
        ConnectorRegistry.from_entry_points()


def test_extractor_entry_points_are_discovered(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_entry_points(
        monkeypatch,
        [
            _FakeEntryPoint(
                "epub",
                EXTRACTOR_ENTRY_POINT_GROUP,
                lambda: (lambda: _Extractor("EpubExtractor", ["application/epub+zip"])),
            )
        ],
    )

    registry = ExtractorRegistry.from_entry_points(config=ExtractionConfig())

    assert registry.resolve("application/epub+zip").name == "EpubExtractor"


def test_extractor_entry_point_producing_a_non_extractor_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_entry_points(
        monkeypatch,
        [
            _FakeEntryPoint(
                "bad", EXTRACTOR_ENTRY_POINT_GROUP, lambda: (lambda: object())
            )
        ],
    )

    with pytest.raises(ExtractorRegistrationError, match="does not implement"):
        ExtractorRegistry.from_entry_points(config=ExtractionConfig())


def test_third_party_extractor_overrides_by_priority(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Overriding a built-in requires a higher priority, never load order."""
    _patch_entry_points(
        monkeypatch,
        [
            _FakeEntryPoint(
                "pdf",
                EXTRACTOR_ENTRY_POINT_GROUP,
                lambda: (lambda: _Extractor("VendorPdf", ["application/pdf"], priority=10)),
            )
        ],
    )

    registry = ExtractorRegistry.from_entry_points(
        config=ExtractionConfig(),
        additional=[_Extractor("BuiltinPdf", ["application/pdf"], priority=0)],
    )

    assert registry.resolve("application/pdf").name == "VendorPdf"


# ---------------------------------------------------------------------------
# Immutability and anti-pattern guards
# ---------------------------------------------------------------------------


def test_registries_expose_no_registration_method() -> None:
    """Constructed complete; resolution cannot change mid-run."""
    for cls in (ConnectorRegistry, ExtractorRegistry):
        assert not hasattr(cls, "register")
        assert not hasattr(cls, "unregister")
        assert not hasattr(cls, "clear")


def test_registry_internals_are_not_assignable() -> None:
    registry = _registry(_Extractor("Text", ["text/plain"]))

    with pytest.raises(AttributeError):
        registry.unexpected_attribute = 1  # type: ignore[attr-defined]


def test_returned_mappings_are_read_only() -> None:
    describe = _registry(_Extractor("Text", ["text/plain"])).describe()

    with pytest.raises(TypeError):
        describe["text/plain"] = "Other"  # type: ignore[index]


def test_module_declares_no_mutable_state() -> None:
    mutable = [
        name
        for name, value in vars(registry_module).items()
        if not name.startswith("_") and isinstance(value, (list, dict, set))
    ]
    assert mutable == []


def test_registry_imports_no_concrete_implementations() -> None:
    """The Core must reference abstractions only."""
    import inspect

    source = inspect.getsource(registry_module)
    for forbidden in (
        "from ptfm.core.ingestion.connectors",
        "from ptfm.core.ingestion.extractors",
        "from ptfm.core.ingestion.pipeline",
        "from ptfm.core.ingestion.models",
        "from ptfm.plugins",
    ):
        assert forbidden not in source


def test_registry_declares_no_numeric_thresholds() -> None:
    """Priority comparisons are between extractor-declared values only."""
    import inspect

    source = inspect.getsource(registry_module)
    assert "priority >" not in source.replace("challenger.priority > incumbent.priority", "")
    assert "> 0.8" not in source