"""
Unit tests for the Knowledge Processing registries.

File:
    tests/core/knowledge/test_registry.py

Emphasis
--------
    Segmentation resolution. A strategy or segmenter substituted without a
    corresponding fingerprint change would leave a corpus holding node
    identifiers that describe a segmentation which never ran, and nothing
    anywhere would report an error.
"""

from __future__ import annotations

import inspect
from typing import Any, Mapping, Sequence

import pytest

from ptfm.core.knowledge import registry as registry_module
from ptfm.core.knowledge.config import (
    ChunkingConfig,
    EmbeddingConfig,
    KnowledgeConfig,
)
from ptfm.core.knowledge.exceptions import (
    ChunkerRegistrationError,
    EmbedderRegistrationError,
    ProviderUnavailableError,
)
from ptfm.core.knowledge.interfaces import (
    ChunkingStrategy,
    ComposedInput,
    EmbeddingProvider,
    ModelIdentity,
    Sentence,
)
from ptfm.core.knowledge.registry import (
    CHUNKER_ENTRY_POINT_GROUP,
    EMBEDDER_ENTRY_POINT_GROUP,
    EmbeddingRegistry,
    ProviderRegistration,
    SegmentationRegistry,
    SegmenterRegistration,
)


def _config(**overrides: Any) -> KnowledgeConfig:
    return KnowledgeConfig(_env_file=None, **overrides)


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class _Strategy(ChunkingStrategy):
    """Strategy whose name and declared parameters are supplied per test."""

    def __init__(
        self,
        name: str = "block_aligned",
        components: Mapping[str, str] | None = None,
    ) -> None:
        self._name = name
        self._components = (
            dict(components) if components is not None else {"window": "512"}
        )

    @property
    def name(self) -> str:
        return self._name

    def plan(self, document: Any) -> Any:
        return document

    def fingerprint_components(self) -> Mapping[str, str]:
        return dict(self._components)


class _Segmenter:
    def __init__(self, version: str = "regex-1.0") -> None:
        self._version = version

    @property
    def version(self) -> str:
        return self._version

    def segment(self, text: str) -> Sequence[Sentence]:
        return [Sentence(start=0, end=len(text))] if text else []


class _Provider(EmbeddingProvider):
    def __init__(self, config: Any = None) -> None:
        self.config = config

    @property
    def name(self) -> str:
        return "TestProvider"

    @property
    def max_batch_size(self) -> int:
        return 32

    def model_identity(self) -> ModelIdentity:
        return ModelIdentity(model_id="test/model", dimension=4, normalized=True)

    async def embed(self, texts: Sequence[str]) -> Sequence[tuple[float, ...]]:
        return [(1.0, 0.0, 0.0, 0.0) for _ in texts]

    async def health_check(self) -> bool:
        return True

    async def close(self) -> None:
        return None


class _Composer:
    def __init__(self, composition_id: str = "heading_path_prefixed") -> None:
        self._id = composition_id

    @property
    def composition_id(self) -> str:
        return self._id

    def compose(self, node: Any) -> ComposedInput:
        return ComposedInput(text="composed", composition_id=self._id)


def _segmentation(
    *,
    strategies: Any = None,
    segmenters: Any = None,
) -> SegmentationRegistry:
    return SegmentationRegistry(
        strategies if strategies is not None else [_Strategy()],
        segmenters
        if segmenters is not None
        else [SegmenterRegistration(name="regex", segmenter=_Segmenter())],
    )


def _embedding(
    *,
    providers: Any = None,
    composers: Any = None,
) -> EmbeddingRegistry:
    return EmbeddingRegistry(
        providers
        if providers is not None
        else [ProviderRegistration(name="null", factory=_Provider)],
        composers if composers is not None else [_Composer()],
    )


# ---------------------------------------------------------------------------
# Resolution is inseparable from fingerprinting
# ---------------------------------------------------------------------------


def test_resolution_returns_components_with_their_fingerprint() -> None:
    """
    Separating them would permit a fingerprint computed from an incomplete
    component set, leaving identifiers describing a segmentation that never
    ran.
    """
    resolved = _segmentation().resolve(_config())

    assert resolved.strategy.name == "block_aligned"
    assert resolved.fingerprint.startswith("1:")
    assert resolved.components


def test_there_is_no_way_to_resolve_without_a_fingerprint() -> None:
    public = [
        name
        for name in dir(SegmentationRegistry)
        if not name.startswith("_") and callable(getattr(SegmentationRegistry, name, None))
    ]

    assert "resolve" in public
    assert "resolve_strategy" not in public
    assert "get_strategy" not in public


def test_the_strategy_contributes_to_the_fingerprint() -> None:
    """Configuration cannot see a strategy's own parameters."""
    narrow = _segmentation(strategies=[_Strategy(components={"window": "512"})])
    wide = _segmentation(strategies=[_Strategy(components={"window": "1024"})])

    assert (
        narrow.resolve(_config()).fingerprint
        != wide.resolve(_config()).fingerprint
    )


def test_the_segmenter_version_contributes_to_the_fingerprint() -> None:
    """
    The contribution configuration cannot see: an upgrade shifts boundaries
    within oversized blocks while every configured value stays the same.
    """
    old = _segmentation(
        segmenters=[
            SegmenterRegistration(name="regex", segmenter=_Segmenter("regex-1.0"))
        ]
    )
    new = _segmentation(
        segmenters=[
            SegmenterRegistration(name="regex", segmenter=_Segmenter("regex-2.0"))
        ]
    )

    assert old.resolve(_config()).fingerprint != new.resolve(_config()).fingerprint


def test_configured_boundary_parameters_still_contribute() -> None:
    registry = _segmentation()
    baseline = registry.resolve(_config())

    altered = registry.resolve(
        _config(chunking=ChunkingConfig(max_chunk_chars=1_500).model_dump())
    )

    assert altered.fingerprint != baseline.fingerprint


def test_every_contribution_is_retained_for_diagnosis() -> None:
    """A mismatch must be diagnosable by comparing components rather than by
    observing that two opaque digests differ."""
    components = _segmentation().resolve(_config()).components

    assert any(key.startswith("chunking.") for key in components)
    assert any(key.startswith("strategy.") for key in components)
    assert "segmenter.version" in components


def test_runtime_contributions_are_namespaced() -> None:
    """A strategy and a segmenter contributing the same parameter name must
    not collide, and neither may collide with a configured parameter."""
    registry = _segmentation(
        strategies=[_Strategy(components={"strategy": "impostor"})]
    )

    components = registry.resolve(_config()).components

    assert components["chunking.strategy"] == "block_aligned"
    assert components["strategy.strategy"] == "impostor"


def test_resolution_is_deterministic() -> None:
    registry = _segmentation()
    config = _config()

    assert registry.resolve(config).fingerprint == registry.resolve(config).fingerprint


# ---------------------------------------------------------------------------
# A name collision is never an override
# ---------------------------------------------------------------------------


def test_a_duplicate_strategy_name_is_rejected() -> None:
    """
    A strategy replacing another would place different boundaries while the
    fingerprint still records the original name.
    """
    with pytest.raises(ChunkerRegistrationError, match="never an override"):
        _segmentation(
            strategies=[_Strategy("block_aligned"), _Strategy("block_aligned")]
        )


def test_a_duplicate_segmenter_name_is_rejected() -> None:
    with pytest.raises(ChunkerRegistrationError, match="Duplicate sentence"):
        _segmentation(
            segmenters=[
                SegmenterRegistration(name="regex", segmenter=_Segmenter()),
                SegmenterRegistration(name="regex", segmenter=_Segmenter()),
            ]
        )


def test_a_duplicate_provider_name_is_rejected() -> None:
    with pytest.raises(EmbedderRegistrationError, match="Duplicate embedding provider"):
        _embedding(
            providers=[
                ProviderRegistration(name="null", factory=_Provider),
                ProviderRegistration(name="null", factory=_Provider),
            ]
        )


def test_a_duplicate_composition_identifier_is_rejected() -> None:
    """
    The identifier is recorded on every vector, so two composers sharing one
    would make vectors from differing compositions silently comparable.
    """
    with pytest.raises(EmbedderRegistrationError, match="silently comparable"):
        _embedding(composers=[_Composer("shared"), _Composer("shared")])


def test_an_alternative_registers_under_its_own_name() -> None:
    registry = _segmentation(
        strategies=[_Strategy("block_aligned"), _Strategy("vendor_aligned")]
    )

    assert registry.strategy_names == ("block_aligned", "vendor_aligned")


# ---------------------------------------------------------------------------
# Registration validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name", ["", "Block_Aligned", "block-aligned", "1block", "block aligned"]
)
def test_malformed_component_names_are_rejected(name: str) -> None:
    """
    A name must render identically in configuration, in an environment
    variable, and in a fingerprint component.
    """
    with pytest.raises(ChunkerRegistrationError, match="Invalid chunking strategy name"):
        _segmentation(strategies=[_Strategy(name)])


def test_a_strategy_declaring_no_parameters_is_rejected() -> None:
    """
    A strategy with no declared parameters cannot be distinguished from a
    differently configured instance of itself.
    """
    with pytest.raises(ChunkerRegistrationError, match="no.*boundary-affecting"):
        _segmentation(strategies=[_Strategy(components={})])


def test_a_non_string_fingerprint_value_is_rejected() -> None:
    """A non-string value renders unstably, so identical configuration could
    produce differing digests."""
    with pytest.raises(ChunkerRegistrationError, match="non-string value"):
        _segmentation(strategies=[_Strategy(components={"window": 512})])  # type: ignore[dict-item]


def test_a_segmenter_without_a_version_is_rejected() -> None:
    """
    Without a version, an upgrade that shifts boundaries would change node
    identity while the fingerprint stayed the same.
    """
    with pytest.raises(ChunkerRegistrationError, match="reports no version"):
        _segmentation(
            segmenters=[
                SegmenterRegistration(name="regex", segmenter=_Segmenter(""))
            ]
        )


def test_a_non_callable_provider_factory_is_rejected() -> None:
    with pytest.raises(EmbedderRegistrationError, match="not callable"):
        _embedding(
            providers=[ProviderRegistration(name="null", factory="not-callable")]  # type: ignore[arg-type]
        )


# ---------------------------------------------------------------------------
# Resolution failure
# ---------------------------------------------------------------------------


def test_an_unregistered_strategy_lists_what_is_available() -> None:
    registry = _segmentation(
        strategies=[_Strategy("block_aligned"), _Strategy("table_aware")]
    )

    with pytest.raises(ChunkerRegistrationError) as excinfo:
        registry.resolve(
            _config(chunking=ChunkingConfig(strategy="absent").model_dump())
        )

    assert "block_aligned, table_aware" in excinfo.value.message


def test_an_unregistered_segmenter_lists_what_is_available() -> None:
    with pytest.raises(ChunkerRegistrationError, match="No sentence segmenter"):
        _segmentation().resolve(
            _config(
                chunking=ChunkingConfig(sentence_segmenter="pysbd").model_dump()
            )
        )


def test_an_empty_registry_reports_none_available() -> None:
    registry = SegmentationRegistry([], [])

    with pytest.raises(ChunkerRegistrationError, match="none"):
        registry.resolve(_config())


def test_an_unregistered_provider_lists_what_is_available() -> None:
    with pytest.raises(EmbedderRegistrationError, match="No embedding provider"):
        _embedding().resolve(
            _config(embedding=EmbeddingConfig(provider="absent").model_dump())
        )


def test_an_unregistered_composition_is_reported() -> None:
    with pytest.raises(EmbedderRegistrationError, match="No embedding composer"):
        _embedding().resolve(
            _config(embedding=EmbeddingConfig(composition="absent").model_dump())
        )


# ---------------------------------------------------------------------------
# Provider construction
# ---------------------------------------------------------------------------


def test_a_provider_is_constructed_per_resolution() -> None:
    """A provider owns resources with a per-run lifecycle."""
    registry = _embedding()
    config = _config()

    assert registry.resolve(config).provider is not registry.resolve(config).provider


def test_the_factory_receives_the_configuration() -> None:
    registry = _embedding()
    config = _config()

    resolved = registry.resolve(config)

    assert resolved.provider.config is config  # type: ignore[attr-defined]


def test_a_missing_optional_dependency_names_a_remedy() -> None:
    """
    A missing extra must produce an actionable message rather than an import
    error surfacing from an unrelated module.
    """

    def _factory(config: Any) -> Any:
        raise ImportError("no module named sentence_transformers", name="sentence_transformers")

    registry = _embedding(
        providers=[ProviderRegistration(name="null", factory=_factory)]
    )

    with pytest.raises(ProviderUnavailableError) as excinfo:
        registry.resolve(_config())

    assert excinfo.value.missing_dependency == "sentence_transformers"
    assert excinfo.value.install_extra == "ptfm[embeddings]"


def test_an_already_typed_unavailability_is_not_rewrapped() -> None:
    original = ProviderUnavailableError(
        component="c", missing_dependency="d", install_extra="e"
    )

    def _factory(config: Any) -> Any:
        raise original

    registry = _embedding(
        providers=[ProviderRegistration(name="null", factory=_factory)]
    )

    with pytest.raises(ProviderUnavailableError) as excinfo:
        registry.resolve(_config())

    assert excinfo.value is original


def test_a_failing_factory_is_wrapped() -> None:
    def _factory(config: Any) -> Any:
        raise RuntimeError("model file corrupt")

    registry = _embedding(
        providers=[ProviderRegistration(name="null", factory=_factory)]
    )

    with pytest.raises(EmbedderRegistrationError) as excinfo:
        registry.resolve(_config())

    assert isinstance(excinfo.value.__cause__, RuntimeError)


def test_a_factory_returning_the_wrong_type_is_rejected() -> None:
    registry = _embedding(
        providers=[ProviderRegistration(name="null", factory=lambda config: object())]
    )

    with pytest.raises(EmbedderRegistrationError, match="does not implement"):
        registry.resolve(_config())


def test_a_composer_is_shared_not_constructed() -> None:
    """Composers are stateless, so one instance serves every resolution."""
    registry = _embedding()
    config = _config()

    assert registry.resolve(config).composer is registry.resolve(config).composer


# ---------------------------------------------------------------------------
# Entry-point discovery
# ---------------------------------------------------------------------------


class _FakeEntryPoint:
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


def test_strategies_are_discovered_from_entry_points(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_entry_points(
        monkeypatch,
        [
            _FakeEntryPoint(
                "vendor_aligned",
                CHUNKER_ENTRY_POINT_GROUP,
                lambda: (lambda: _Strategy("vendor_aligned")),
            )
        ],
    )

    registry = SegmentationRegistry.from_entry_points()

    assert registry.strategy_names == ("vendor_aligned",)


def test_a_discovered_name_colliding_with_a_builtin_is_an_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A third-party package must not silently displace a built-in strategy."""
    _patch_entry_points(
        monkeypatch,
        [
            _FakeEntryPoint(
                "block_aligned",
                CHUNKER_ENTRY_POINT_GROUP,
                lambda: (lambda: _Strategy("block_aligned")),
            )
        ],
    )

    with pytest.raises(ChunkerRegistrationError, match="never an override"):
        SegmentationRegistry.from_entry_points(
            strategies=[_Strategy("block_aligned")]
        )


def test_a_broken_entry_point_names_itself(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _explode() -> Any:
        raise ImportError("missing transitive dependency")

    _patch_entry_points(
        monkeypatch, [_FakeEntryPoint("bad", CHUNKER_ENTRY_POINT_GROUP, _explode)]
    )

    with pytest.raises(ChunkerRegistrationError) as excinfo:
        SegmentationRegistry.from_entry_points()

    assert "bad" in excinfo.value.message
    assert isinstance(excinfo.value.__cause__, ImportError)


def test_an_entry_point_producing_the_wrong_type_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_entry_points(
        monkeypatch,
        [
            _FakeEntryPoint(
                "bad", CHUNKER_ENTRY_POINT_GROUP, lambda: (lambda: object())
            )
        ],
    )

    with pytest.raises(ChunkerRegistrationError, match="does not implement"):
        SegmentationRegistry.from_entry_points()


def test_providers_are_discovered_from_entry_points(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_entry_points(
        monkeypatch,
        [
            _FakeEntryPoint(
                "local", EMBEDDER_ENTRY_POINT_GROUP, lambda: _Provider
            )
        ],
    )

    registry = EmbeddingRegistry.from_entry_points(composers=[_Composer()])

    assert registry.provider_names == ("local",)


# ---------------------------------------------------------------------------
# Introspection and immutability
# ---------------------------------------------------------------------------


def test_registries_describe_their_registrations() -> None:
    described = _segmentation().describe()

    assert described["strategy:block_aligned"] == "_Strategy"
    assert described["segmenter:regex"].endswith("@regex-1.0")


def test_the_embedding_registry_describes_its_registrations() -> None:
    described = _embedding().describe()

    assert described["provider:null"] == "explicit"
    assert described["composer:heading_path_prefixed"] == "_Composer"


def test_registries_expose_no_registration_method() -> None:
    """Constructed complete; resolution cannot change mid-run."""
    for cls in (SegmentationRegistry, EmbeddingRegistry):
        assert not hasattr(cls, "register")
        assert not hasattr(cls, "unregister")
        assert not hasattr(cls, "clear")


def test_registry_internals_are_not_assignable() -> None:
    with pytest.raises(AttributeError):
        _segmentation().extra = 1  # type: ignore[attr-defined]


def test_returned_mappings_are_read_only() -> None:
    described = _segmentation().describe()

    with pytest.raises(TypeError):
        described["strategy:block_aligned"] = "Other"  # type: ignore[index]


def test_resolved_components_are_read_only() -> None:
    components = _segmentation().resolve(_config()).components

    with pytest.raises(TypeError):
        components["chunking.strategy"] = "other"  # type: ignore[index]


# ---------------------------------------------------------------------------
# Anti-pattern guards
# ---------------------------------------------------------------------------


def test_the_module_imports_no_concrete_implementation() -> None:
    source = inspect.getsource(registry_module)

    for forbidden in (
        "from ptfm.core.knowledge.chunking",
        "from ptfm.core.knowledge.embedding",
        "from ptfm.core.knowledge.pipeline",
        "from ptfm.core.knowledge.models",
        "from ptfm.plugins",
    ):
        assert forbidden not in source


def test_the_module_declares_no_mutable_state() -> None:
    mutable = [
        name
        for name, value in vars(registry_module).items()
        if not name.startswith("_") and isinstance(value, (list, dict, set))
    ]

    assert mutable == []


def test_the_module_declares_no_thresholds() -> None:
    source = inspect.getsource(registry_module)

    assert "> 0.8" not in source
    assert "confidence" not in source.lower()
    assert "similarity" not in source.lower()


def test_the_module_contains_no_domain_vocabulary() -> None:
    source = inspect.getsource(registry_module).lower()

    for term in ("jurisdiction", "statute", "ticker", "icd", "patient", "court"):
        assert term not in source
