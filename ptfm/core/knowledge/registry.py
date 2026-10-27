"""
PTFM Knowledge Processing Registries

File:
    ptfm/core/knowledge/registry.py

Description:
    Resolves configured component names to the implementations that serve
    them. Because resolution happens here, the engine contains no branching on
    strategy or provider, and a new implementation can be added without
    editing any existing file.

Two registries, two responsibilities
------------------------------------
    SegmentationRegistry
        Holds chunking strategies and sentence segmenters, and resolves them
        together with the segmentation fingerprint they imply.

    EmbeddingRegistry
        Holds embedding provider factories and text composers, and resolves
        them together.

    The split follows the module's two execution planes rather than the kinds
    of object involved. Segmentation components determine node identity;
    embedding components determine vectors, which have an independent
    lifecycle.

Resolution is inseparable from fingerprinting
---------------------------------------------
    ``SegmentationRegistry.resolve`` returns the strategy, the segmenter, AND
    the fingerprint together. There is no way to obtain the components without
    the fingerprint that describes them, and no way to compute the fingerprint
    without holding the components.

    This matters because the fingerprint must include contributions that
    configuration cannot see: a strategy's own parameters, and a segmenter's
    version. Were fingerprinting a separate step, a caller could omit a
    contribution and produce a fingerprint that fails to describe the
    segmentation it names. Nothing would report an error; node identifiers
    would simply cease to mean what they claim.

A name collision is always an error, never an override
------------------------------------------------------
    Module 1 permitted a third-party extractor to override a built-in by
    declaring a higher priority, because extractor choice affects only how a
    document is parsed.

    That is not acceptable here. A third-party strategy silently replacing
    ``block_aligned`` would place different boundaries while the fingerprint
    still records ``chunking.strategy=block_aligned``, so the corpus would
    hold nodes whose identifiers describe a segmentation that never ran.
    Registration therefore rejects a duplicate name outright, and an
    alternative implementation must be registered under its own name and
    selected explicitly.

Immutability
------------
    Both registries are constructed complete and are immutable thereafter.
    There is no registration method and no mutable phase, so resolution cannot
    change during a run.

Dependency contract
-------------------
    Imports only the standard library, ``ptfm.core.knowledge.exceptions``,
    ``ptfm.core.knowledge.config`` and ``ptfm.core.knowledge.interfaces``. It
    must not import ``models``, any stage, or any concrete implementation.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from importlib.metadata import EntryPoint, entry_points
from types import MappingProxyType
from typing import Callable, Final, Iterable, Mapping, NamedTuple, Self, TypeAlias

from ptfm.core.knowledge.config import KnowledgeConfig
from ptfm.core.knowledge.exceptions import (
    ChunkerRegistrationError,
    EmbedderRegistrationError,
    ProviderUnavailableError,
)
from ptfm.core.knowledge.interfaces import (
    ChunkingStrategy,
    EmbeddingProvider,
    EmbeddingTextComposer,
    SentenceSegmenter,
)

__all__ = [
    "CHUNKER_ENTRY_POINT_GROUP",
    "SEGMENTER_ENTRY_POINT_GROUP",
    "EMBEDDER_ENTRY_POINT_GROUP",
    "COMPOSER_ENTRY_POINT_GROUP",
    "ProviderFactory",
    "SegmenterRegistration",
    "ProviderRegistration",
    "ResolvedSegmentation",
    "ResolvedEmbedding",
    "SegmentationRegistry",
    "EmbeddingRegistry",
]

#: Entry-point groups through which a separately installed package may
#: contribute an implementation without any edit to PTFM.
CHUNKER_ENTRY_POINT_GROUP: Final[str] = "ptfm.knowledge.chunkers"
SEGMENTER_ENTRY_POINT_GROUP: Final[str] = "ptfm.knowledge.segmenters"
EMBEDDER_ENTRY_POINT_GROUP: Final[str] = "ptfm.knowledge.embedders"
COMPOSER_ENTRY_POINT_GROUP: Final[str] = "ptfm.knowledge.composers"

#: Constructs an embedding provider for one run. A provider holds a model in
#: memory or a connection to a service, so it has a lifecycle and is built per
#: run rather than shared as an instance.
ProviderFactory: TypeAlias = Callable[[KnowledgeConfig], EmbeddingProvider]

#: Permitted component name grammar. Restricted so that a name renders
#: identically in configuration, in an environment variable, in an entry-point
#: declaration, and in a fingerprint component.
_NAME_PATTERN: Final[re.Pattern[str]] = re.compile(r"^[a-z][a-z0-9_]*$")

#: Namespaces applied to runtime fingerprint contributions, so that a strategy
#: and a segmenter contributing the same parameter name do not collide, and
#: neither collides with a configured chunking parameter.
_STRATEGY_NAMESPACE: Final[str] = "strategy"
_SEGMENTER_NAMESPACE: Final[str] = "segmenter"


# ============================================================================
# REGISTRATION RECORDS
# ============================================================================


@dataclass(frozen=True, slots=True)
class SegmenterRegistration:
    """
    A sentence segmenter bound to the name that selects it.

    The name is supplied explicitly because ``SentenceSegmenter`` declares a
    version but not a name. That asymmetry is deliberate: a segmenter's
    version is identity-affecting and must come from the implementation, while
    its name is merely a selector and may reasonably be chosen by whoever
    registers it.

    Registering a different implementation under an existing name is
    nonetheless safe rather than silent, because the version participates in
    the fingerprint: the substitution changes the digest even though the
    configured name does not change.

    Attributes
    ----------
    name:
        Selector matching ``ChunkingConfig.sentence_segmenter``.
    segmenter:
        The implementation.
    origin:
        Where the registration came from, for diagnostics.
    """

    name: str
    segmenter: SentenceSegmenter
    origin: str = "explicit"


@dataclass(frozen=True, slots=True)
class ProviderRegistration:
    """
    An embedding provider factory bound to the name that selects it.

    A factory rather than an instance, because a provider owns resources: a
    local model occupies gigabytes and takes seconds to load, and a remote
    provider holds connections. One is constructed per run and closed when the
    run ends.

    Attributes
    ----------
    name:
        Selector matching ``EmbeddingConfig.provider``.
    factory:
        Callable producing a provider for one run.
    origin:
        Where the registration came from, for diagnostics.
    """

    name: str
    factory: ProviderFactory
    origin: str = "explicit"


# ============================================================================
# RESOLVED BUNDLES
# ============================================================================


class ResolvedSegmentation(NamedTuple):
    """
    The segmentation components for a run, with the fingerprint they imply.

    Returned as one value because the fingerprint is only meaningful alongside
    the components it describes. Separating them would permit a fingerprint
    computed from an incomplete component set, which would leave node
    identifiers claiming a segmentation that never ran.

    Attributes
    ----------
    strategy:
        The chunking strategy.
    segmenter:
        The sentence segmenter used where a block must be divided internally.
    fingerprint:
        The segmentation fingerprint, composed from configuration together
        with the runtime contributions of both components.
    components:
        Every contribution to the fingerprint, retained so that a mismatch can
        be diagnosed by comparing components rather than by observing that two
        opaque digests differ.
    """

    strategy: ChunkingStrategy
    segmenter: SentenceSegmenter
    fingerprint: str
    components: Mapping[str, str]


class ResolvedEmbedding(NamedTuple):
    """
    The embedding components for a run.

    Attributes
    ----------
    provider:
        The constructed provider. The caller owns its lifetime and is
        responsible for closing it, ordinarily through ``async with``.
    composer:
        The rule composing embedding input from a node's fields.
    """

    provider: EmbeddingProvider
    composer: EmbeddingTextComposer


# ============================================================================
# SEGMENTATION REGISTRY
# ============================================================================


class SegmentationRegistry:
    """
    Resolves the components that determine where node boundaries fall.

    Immutable once constructed. Both component kinds are held together because
    both contribute to the segmentation fingerprint and both are resolved at
    the same moment; separating them would permit a caller to obtain one
    without the other and compute an incomplete fingerprint.

    Parameters
    ----------
    strategies:
        Chunking strategies, keyed by their own declared names.
    segmenters:
        Sentence segmenter registrations.

    Raises
    ------
    ChunkerRegistrationError
        If a name is malformed, a name is registered twice, a strategy
        declares no boundary-affecting parameters, or a strategy's declared
        name does not match the key it would be registered under.
    """

    __slots__ = ("_strategies", "_segmenters")

    def __init__(
        self,
        strategies: Iterable[ChunkingStrategy],
        segmenters: Iterable[SegmenterRegistration],
    ) -> None:
        resolved_strategies: dict[str, ChunkingStrategy] = {}

        for strategy in strategies:
            name = _validated_name(
                strategy.name, kind="chunking strategy", error=ChunkerRegistrationError
            )

            if name in resolved_strategies:
                raise ChunkerRegistrationError(
                    f"Duplicate chunking strategy name {name!r}. A name "
                    f"collision is never an override here: a strategy "
                    f"replacing another would place different boundaries "
                    f"while the fingerprint still records the original name, "
                    f"so the corpus would hold identifiers describing a "
                    f"segmentation that never ran. Register the alternative "
                    f"under its own name.",
                    strategy_name=name,
                )

            components = strategy.fingerprint_components()
            if not components:
                raise ChunkerRegistrationError(
                    f"Chunking strategy {name!r} declares no "
                    f"boundary-affecting parameters. A strategy with no "
                    f"declared parameters cannot be distinguished from a "
                    f"differently configured instance of itself, so a "
                    f"parameter change would not alter node identity.",
                    strategy_name=name,
                )
            _validate_components(components, owner=name, error=ChunkerRegistrationError)

            resolved_strategies[name] = strategy

        resolved_segmenters: dict[str, SegmenterRegistration] = {}

        for registration in segmenters:
            name = _validated_name(
                registration.name,
                kind="sentence segmenter",
                error=ChunkerRegistrationError,
            )

            existing = resolved_segmenters.get(name)
            if existing is not None:
                raise ChunkerRegistrationError(
                    f"Duplicate sentence segmenter name {name!r}: "
                    f"{existing.origin} and {registration.origin} both claim "
                    f"it.",
                    strategy_name=name,
                )

            version = registration.segmenter.version
            if not version or not version.strip():
                raise ChunkerRegistrationError(
                    f"Sentence segmenter {name!r} reports no version. The "
                    f"version participates in the segmentation fingerprint, "
                    f"so without it an upgrade that shifts boundaries within "
                    f"oversized blocks would change node identity while the "
                    f"fingerprint stayed the same.",
                    strategy_name=name,
                )

            resolved_segmenters[name] = SegmenterRegistration(
                name=name,
                segmenter=registration.segmenter,
                origin=registration.origin,
            )

        self._strategies: Mapping[str, ChunkingStrategy] = MappingProxyType(
            resolved_strategies
        )
        self._segmenters: Mapping[str, SegmenterRegistration] = MappingProxyType(
            resolved_segmenters
        )

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_entry_points(
        cls,
        *,
        strategies: Iterable[ChunkingStrategy] = (),
        segmenters: Iterable[SegmenterRegistration] = (),
        chunker_group: str = CHUNKER_ENTRY_POINT_GROUP,
        segmenter_group: str = SEGMENTER_ENTRY_POINT_GROUP,
    ) -> Self:
        """
        Build a registry from installed entry points plus explicit entries.

        Allows a separately installed package to contribute a strategy or
        segmenter with no edit to PTFM. Each entry point must resolve to a
        zero-argument callable returning the implementation.

        A discovered name colliding with an explicit one is an error rather
        than an override, for the reason given in the module documentation.

        Parameters
        ----------
        strategies:
            Strategies supplied directly by the composition root.
        segmenters:
            Segmenter registrations supplied directly.
        chunker_group:
            Entry-point group for strategies.
        segmenter_group:
            Entry-point group for segmenters.

        Returns
        -------
        Self
            The constructed registry.

        Raises
        ------
        ChunkerRegistrationError
            If an entry point cannot be loaded, produces the wrong type, or
            collides with another registration.
        """
        discovered_strategies: list[ChunkingStrategy] = []
        for point in entry_points(group=chunker_group):
            instance = _load_and_build(
                point, expected="chunking strategy", error=ChunkerRegistrationError
            )
            if not isinstance(instance, ChunkingStrategy):
                raise ChunkerRegistrationError(
                    f"Chunking strategy entry point {point.name!r} produced "
                    f"{type(instance).__name__}, which does not implement "
                    f"ChunkingStrategy.",
                    strategy_name=point.name,
                )
            discovered_strategies.append(instance)

        discovered_segmenters: list[SegmenterRegistration] = []
        for point in entry_points(group=segmenter_group):
            instance = _load_and_build(
                point, expected="sentence segmenter", error=ChunkerRegistrationError
            )
            if not isinstance(instance, SentenceSegmenter):
                raise ChunkerRegistrationError(
                    f"Sentence segmenter entry point {point.name!r} produced "
                    f"{type(instance).__name__}, which does not implement "
                    f"SentenceSegmenter.",
                    strategy_name=point.name,
                )
            discovered_segmenters.append(
                SegmenterRegistration(
                    name=point.name,
                    segmenter=instance,
                    origin=f"entry_point:{point.name}",
                )
            )

        return cls(
            [*discovered_strategies, *strategies],
            [*discovered_segmenters, *segmenters],
        )

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def strategy_names(self) -> tuple[str, ...]:
        """
        Registered strategy names, in sorted order.

        Returns
        -------
        tuple[str, ...]
            Available strategy names.
        """
        return tuple(sorted(self._strategies))

    @property
    def segmenter_names(self) -> tuple[str, ...]:
        """
        Registered segmenter names, in sorted order.

        Returns
        -------
        tuple[str, ...]
            Available segmenter names.
        """
        return tuple(sorted(self._segmenters))

    def describe(self) -> Mapping[str, str]:
        """
        Summarize registrations for startup diagnostics.

        Returns
        -------
        Mapping[str, str]
            Namespaced component names mapped to a description of the
            implementation serving them.
        """
        described = {
            f"strategy:{name}": type(strategy).__name__
            for name, strategy in self._strategies.items()
        }
        described.update(
            {
                f"segmenter:{name}": (
                    f"{type(registration.segmenter).__name__}"
                    f"@{registration.segmenter.version}"
                )
                for name, registration in self._segmenters.items()
            }
        )
        return MappingProxyType(dict(sorted(described.items())))

    # ------------------------------------------------------------------
    # Resolution
    # ------------------------------------------------------------------

    def resolve(self, config: KnowledgeConfig) -> ResolvedSegmentation:
        """
        Resolve the segmentation components and the fingerprint they imply.

        The components and the fingerprint are returned together and cannot be
        obtained separately. Were they separable, a caller could compute a
        fingerprint from an incomplete component set, and the resulting node
        identifiers would describe a segmentation that never ran, with nothing
        reporting an error.

        Parameters
        ----------
        config:
            The configuration naming the components to resolve.

        Returns
        -------
        ResolvedSegmentation
            The strategy, the segmenter, the fingerprint, and every
            contribution to it.

        Raises
        ------
        ChunkerRegistrationError
            If a named component is not registered.
        KnowledgeConfigurationError
            If two components contribute the same fingerprint key with
            differing values.
        """
        strategy_name = config.chunking.strategy
        strategy = self._strategies.get(strategy_name)
        if strategy is None:
            raise ChunkerRegistrationError(
                f"No chunking strategy registered under {strategy_name!r}. "
                f"Registered strategies: "
                f"{', '.join(self.strategy_names) or 'none'}.",
                strategy_name=strategy_name,
            )

        segmenter_name = config.chunking.sentence_segmenter
        registration = self._segmenters.get(segmenter_name)
        if registration is None:
            raise ChunkerRegistrationError(
                f"No sentence segmenter registered under {segmenter_name!r}. "
                f"Registered segmenters: "
                f"{', '.join(self.segmenter_names) or 'none'}.",
                strategy_name=segmenter_name,
            )

        runtime_components: dict[str, str] = {
            f"{_STRATEGY_NAMESPACE}.{key}": value
            for key, value in strategy.fingerprint_components().items()
        }
        # The segmenter's version is the contribution configuration cannot
        # see: an upgrade shifts boundaries within oversized blocks while
        # every configured value stays the same.
        runtime_components[f"{_SEGMENTER_NAMESPACE}.version"] = (
            registration.segmenter.version
        )

        fingerprint = config.segmentation_fingerprint(runtime_components)

        components = dict(config.boundary_components())
        components.update(runtime_components)

        return ResolvedSegmentation(
            strategy=strategy,
            segmenter=registration.segmenter,
            fingerprint=fingerprint,
            components=MappingProxyType(components),
        )

    def __len__(self) -> int:
        """
        Number of registered strategies.

        Returns
        -------
        int
            Strategy count.
        """
        return len(self._strategies)

    def __repr__(self) -> str:
        """
        Return a representation listing registered component names.

        Returns
        -------
        str
            Diagnostic representation.
        """
        return (
            f"SegmentationRegistry(strategies={list(self.strategy_names)!r}, "
            f"segmenters={list(self.segmenter_names)!r})"
        )


# ============================================================================
# EMBEDDING REGISTRY
# ============================================================================


class EmbeddingRegistry:
    """
    Resolves the components that produce vectors.

    Immutable once constructed. Providers are held as factories rather than
    instances, because a provider owns resources with a per-run lifecycle;
    composers are held as instances, because they are stateless.

    Nothing here contributes to node identity. Replacing a provider or a
    composer invalidates vectors while leaving every node boundary, span, and
    identifier correct, so a corpus can be re-embedded without reprocessing.

    Parameters
    ----------
    providers:
        Embedding provider registrations.
    composers:
        Text composers, keyed by their own declared composition identifiers.

    Raises
    ------
    EmbedderRegistrationError
        If a name is malformed, a name is registered twice, or a factory is
        not callable.
    """

    __slots__ = ("_providers", "_composers")

    def __init__(
        self,
        providers: Iterable[ProviderRegistration],
        composers: Iterable[EmbeddingTextComposer],
    ) -> None:
        resolved_providers: dict[str, ProviderRegistration] = {}

        for registration in providers:
            name = _validated_name(
                registration.name,
                kind="embedding provider",
                error=EmbedderRegistrationError,
            )

            existing = resolved_providers.get(name)
            if existing is not None:
                raise EmbedderRegistrationError(
                    f"Duplicate embedding provider name {name!r}: "
                    f"{existing.origin} and {registration.origin} both claim "
                    f"it.",
                    provider_name=name,
                )

            if not callable(registration.factory):
                raise EmbedderRegistrationError(
                    f"Embedding provider factory for {name!r} is not callable.",
                    provider_name=name,
                )

            resolved_providers[name] = ProviderRegistration(
                name=name,
                factory=registration.factory,
                origin=registration.origin,
            )

        resolved_composers: dict[str, EmbeddingTextComposer] = {}

        for composer in composers:
            name = _validated_name(
                composer.composition_id,
                kind="embedding text composer",
                error=EmbedderRegistrationError,
            )

            if name in resolved_composers:
                raise EmbedderRegistrationError(
                    f"Duplicate embedding composition identifier {name!r}. "
                    f"The identifier is recorded on every vector produced, so "
                    f"two composers sharing one would make vectors from "
                    f"differing compositions indistinguishable and silently "
                    f"comparable.",
                    provider_name=name,
                )

            resolved_composers[name] = composer

        self._providers: Mapping[str, ProviderRegistration] = MappingProxyType(
            resolved_providers
        )
        self._composers: Mapping[str, EmbeddingTextComposer] = MappingProxyType(
            resolved_composers
        )

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_entry_points(
        cls,
        *,
        providers: Iterable[ProviderRegistration] = (),
        composers: Iterable[EmbeddingTextComposer] = (),
        embedder_group: str = EMBEDDER_ENTRY_POINT_GROUP,
        composer_group: str = COMPOSER_ENTRY_POINT_GROUP,
    ) -> Self:
        """
        Build a registry from installed entry points plus explicit entries.

        A provider entry point must resolve to a factory accepting a
        configuration; a composer entry point must resolve to a zero-argument
        callable returning a composer.

        Parameters
        ----------
        providers:
            Provider registrations supplied directly.
        composers:
            Composers supplied directly.
        embedder_group:
            Entry-point group for providers.
        composer_group:
            Entry-point group for composers.

        Returns
        -------
        Self
            The constructed registry.

        Raises
        ------
        EmbedderRegistrationError
            If an entry point cannot be loaded, produces the wrong type, or
            collides with another registration.
        """
        discovered_providers = [
            ProviderRegistration(
                name=point.name,
                factory=_load_entry_point(
                    point,
                    expected="embedding provider factory",
                    error=EmbedderRegistrationError,
                ),
                origin=f"entry_point:{point.name}",
            )
            for point in entry_points(group=embedder_group)
        ]

        discovered_composers: list[EmbeddingTextComposer] = []
        for point in entry_points(group=composer_group):
            instance = _load_and_build(
                point,
                expected="embedding text composer",
                error=EmbedderRegistrationError,
            )
            if not isinstance(instance, EmbeddingTextComposer):
                raise EmbedderRegistrationError(
                    f"Embedding composer entry point {point.name!r} produced "
                    f"{type(instance).__name__}, which does not implement "
                    f"EmbeddingTextComposer.",
                    provider_name=point.name,
                )
            discovered_composers.append(instance)

        return cls(
            [*discovered_providers, *providers],
            [*discovered_composers, *composers],
        )

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def provider_names(self) -> tuple[str, ...]:
        """
        Registered provider names, in sorted order.

        Returns
        -------
        tuple[str, ...]
            Available provider names.
        """
        return tuple(sorted(self._providers))

    @property
    def composition_ids(self) -> tuple[str, ...]:
        """
        Registered composition identifiers, in sorted order.

        Returns
        -------
        tuple[str, ...]
            Available composition identifiers.
        """
        return tuple(sorted(self._composers))

    def describe(self) -> Mapping[str, str]:
        """
        Summarize registrations for startup diagnostics.

        Returns
        -------
        Mapping[str, str]
            Namespaced component names mapped to registration origin or
            implementation.
        """
        described = {
            f"provider:{name}": registration.origin
            for name, registration in self._providers.items()
        }
        described.update(
            {
                f"composer:{name}": type(composer).__name__
                for name, composer in self._composers.items()
            }
        )
        return MappingProxyType(dict(sorted(described.items())))

    # ------------------------------------------------------------------
    # Resolution
    # ------------------------------------------------------------------

    def resolve(self, config: KnowledgeConfig) -> ResolvedEmbedding:
        """
        Construct the embedding components for a run.

        A provider is constructed fresh, because it owns resources with a
        per-run lifecycle. The caller is responsible for closing it,
        ordinarily through ``async with``.

        Parameters
        ----------
        config:
            The configuration naming the components to resolve.

        Returns
        -------
        ResolvedEmbedding
            The constructed provider and the composer.

        Raises
        ------
        EmbedderRegistrationError
            If a named component is not registered, or the factory failed or
            returned the wrong type.
        ProviderUnavailableError
            If the provider's optional dependency is not installed.
        """
        provider_name = config.embedding.provider
        registration = self._providers.get(provider_name)
        if registration is None:
            raise EmbedderRegistrationError(
                f"No embedding provider registered under {provider_name!r}. "
                f"Registered providers: "
                f"{', '.join(self.provider_names) or 'none'}.",
                provider_name=provider_name,
            )

        composition_id = config.embedding.composition
        composer = self._composers.get(composition_id)
        if composer is None:
            raise EmbedderRegistrationError(
                f"No embedding composer registered under {composition_id!r}. "
                f"Registered compositions: "
                f"{', '.join(self.composition_ids) or 'none'}.",
                provider_name=composition_id,
            )

        try:
            provider = registration.factory(config)
        except ProviderUnavailableError:
            # Already the actionable form, naming the missing dependency and
            # the extra that supplies it.
            raise
        except ImportError as exc:
            # A provider whose optional dependency is absent. Converted here
            # so that the failure names a remedy rather than surfacing as an
            # import error from an unrelated module.
            raise ProviderUnavailableError(
                component=f"embedding provider {provider_name!r}",
                missing_dependency=exc.name or "unknown",
                install_extra="ptfm[embeddings]",
            ) from exc
        except Exception as exc:
            raise EmbedderRegistrationError(
                f"Embedding provider factory for {provider_name!r} "
                f"({registration.origin}) raised {type(exc).__name__}: {exc}",
                provider_name=provider_name,
            ) from exc

        if not isinstance(provider, EmbeddingProvider):
            raise EmbedderRegistrationError(
                f"Embedding provider factory for {provider_name!r} returned "
                f"{type(provider).__name__}, which does not implement "
                f"EmbeddingProvider.",
                provider_name=provider_name,
            )

        return ResolvedEmbedding(provider=provider, composer=composer)

    def __len__(self) -> int:
        """
        Number of registered providers.

        Returns
        -------
        int
            Provider count.
        """
        return len(self._providers)

    def __repr__(self) -> str:
        """
        Return a representation listing registered component names.

        Returns
        -------
        str
            Diagnostic representation.
        """
        return (
            f"EmbeddingRegistry(providers={list(self.provider_names)!r}, "
            f"compositions={list(self.composition_ids)!r})"
        )


# ============================================================================
# HELPERS
# ============================================================================


def _validated_name(
    name: str,
    *,
    kind: str,
    error: type[Exception],
) -> str:
    """
    Verify that a component name is well formed.

    The grammar is restricted so that a name renders identically wherever it
    appears: in configuration, in an environment variable, in an entry-point
    declaration, and in a fingerprint component. A name differing by case or
    punctuation between those places would resolve in one and fail in another.

    Parameters
    ----------
    name:
        The declared name.
    kind:
        Component kind, for the error message.
    error:
        Exception type to raise.

    Returns
    -------
    str
        The validated name.

    Raises
    ------
    Exception
        Of the supplied type, if the name is malformed.
    """
    if not _NAME_PATTERN.match(name or ""):
        raise error(
            f"Invalid {kind} name {name!r}. A name must begin with a lowercase "
            f"letter and contain only lowercase letters, digits, and "
            f"underscores, so that it renders identically in configuration, "
            f"in an environment variable, and in a fingerprint component."
        )
    return name


def _validate_components(
    components: Mapping[str, str],
    *,
    owner: str,
    error: type[Exception],
) -> None:
    """
    Verify that declared fingerprint components are usable in a digest.

    Values must be strings. A non-string value would render through ``str``
    in a form that could differ between Python versions or between a value
    supplied in code and the same value supplied through configuration,
    making the digest unstable for identical configuration.

    Parameters
    ----------
    components:
        The declared components.
    owner:
        Name of the declaring component, for the error message.
    error:
        Exception type to raise.

    Returns
    -------
    None

    Raises
    ------
    Exception
        Of the supplied type, if a key or value is unusable.
    """
    for key, value in components.items():
        if not isinstance(key, str) or not key:
            raise error(
                f"Component {owner!r} declares a fingerprint key that is not a "
                f"non-empty string: {key!r}"
            )
        if not isinstance(value, str):
            raise error(
                f"Component {owner!r} declares a non-string value for "
                f"fingerprint key {key!r}: {value!r}. A non-string value would "
                f"render unstably, so identical configuration could produce "
                f"differing digests."
            )


def _load_entry_point(
    point: EntryPoint,
    *,
    expected: str,
    error: type[Exception],
) -> Callable[..., object]:
    """
    Load an entry point, converting any failure into a PTFM exception.

    A third-party package may raise anything at import. Wrapping here means a
    broken plugin produces an actionable startup message naming the entry
    point, rather than an arbitrary traceback from an unrelated module.

    Parameters
    ----------
    point:
        The entry point to load.
    expected:
        Description of the expected object, for the error message.
    error:
        Exception type to raise.

    Returns
    -------
    Callable[..., object]
        The loaded callable.

    Raises
    ------
    Exception
        Of the supplied type, if loading failed or produced a non-callable.
    """
    try:
        loaded = point.load()
    except Exception as exc:
        raise error(
            f"Entry point {point.name!r} in group {point.group!r} could not be "
            f"loaded: {type(exc).__name__}: {exc}"
        ) from exc

    if not callable(loaded):
        raise error(
            f"Entry point {point.name!r} in group {point.group!r} resolved to "
            f"{type(loaded).__name__}, which is not a callable {expected}."
        )

    return loaded


def _load_and_build(
    point: EntryPoint,
    *,
    expected: str,
    error: type[Exception],
) -> object:
    """
    Load an entry point and invoke it to produce an implementation.

    Parameters
    ----------
    point:
        The entry point to load.
    expected:
        Description of the expected object, for the error message.
    error:
        Exception type to raise.

    Returns
    -------
    object
        The constructed implementation, unvalidated.

    Raises
    ------
    Exception
        Of the supplied type, if loading or construction failed.
    """
    factory = _load_entry_point(point, expected=expected, error=error)
    try:
        return factory()
    except Exception as exc:
        raise error(
            f"Entry point {point.name!r} in group {point.group!r} raised "
            f"{type(exc).__name__} while constructing a {expected}: {exc}"
        ) from exc
