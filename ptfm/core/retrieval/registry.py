"""
PTFM Retrieval Registries

File:
    ptfm/core/retrieval/registry.py

Description:
    Resolves configured component names to the implementations that serve
    them. Because resolution happens here, the engine contains no branching on
    arm or fusion strategy, and a new retrieval method can be added without
    editing any existing file.

Two registries, two lifecycles
------------------------------
    ArmRegistry
        Holds retrieval arms as factories. An arm owns a connection to a
        store, so it is constructed per engine rather than shared as an
        instance.

    FusionRegistry
        Holds fusion strategies as instances. Fusion is pure and stateless,
        so one instance serves every query.

A name collision is an error, never an override
-----------------------------------------------
    Ingestion permitted a third-party extractor to displace a built-in by
    declaring a higher priority, because extractor choice affects only how a
    document is parsed.

    That is not acceptable here. An arm silently replacing the dense arm would
    search a different index while diagnostics still recorded ``dense``, so a
    degraded result would be attributed to a component that never ran.
    Registration therefore rejects a duplicate name outright, and an
    alternative implementation registers under its own name and is selected
    explicitly.

Arms are resolved as a set, not one at a time
---------------------------------------------
    ``ArmRegistry.resolve`` returns every enabled arm together, and rejects a
    configuration that would leave none. A query served by no arm returns an
    empty result, which is indistinguishable from an absence of evidence and
    would cause a refusal to answer a question the corpus supports.

    Resolving one arm at a time would make that condition unobservable until
    the query path had already committed to it.

Immutability
------------
    Both registries are constructed complete and are immutable thereafter.
    There is no registration method and no mutable phase, so resolution cannot
    change during a run.

Dependency contract
-------------------
    Imports only the standard library, ``ptfm.core.retrieval.exceptions``,
    ``ptfm.core.retrieval.config`` and ``ptfm.core.retrieval.interfaces``. It
    must not import ``models``, any stage, or any concrete backend.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from importlib.metadata import EntryPoint, entry_points
from types import MappingProxyType
from typing import Callable, Final, Iterable, Mapping, Self, TypeAlias

from ptfm.core.retrieval.config import ArmConfig, RetrievalConfig
from ptfm.core.retrieval.exceptions import (
    BackendUnavailableError,
    RetrievalConfigurationError,
    StrategyRegistrationError,
)
from ptfm.core.retrieval.interfaces import FusionStrategy, RetrievalStrategy

__all__ = [
    "ARM_ENTRY_POINT_GROUP",
    "FUSION_ENTRY_POINT_GROUP",
    "ArmFactory",
    "ArmRegistration",
    "ArmRegistry",
    "FusionRegistry",
]

#: Entry-point groups through which a separately installed package may
#: contribute an implementation with no edit to PTFM.
ARM_ENTRY_POINT_GROUP: Final[str] = "ptfm.retrieval.arms"
FUSION_ENTRY_POINT_GROUP: Final[str] = "ptfm.retrieval.fusion"

#: Constructs a retrieval arm for one engine. An arm owns a connection to a
#: store, so it has a lifecycle and is built per engine rather than shared.
ArmFactory: TypeAlias = Callable[[RetrievalConfig], RetrievalStrategy]

#: Permitted component name grammar. Restricted so that a name renders
#: identically wherever it appears: in configuration, in an environment
#: variable, in an entry-point declaration, and in a diagnostic.
_NAME_PATTERN: Final[re.Pattern[str]] = re.compile(r"^[a-z][a-z0-9_]*$")


# ============================================================================
# REGISTRATION RECORDS
# ============================================================================


@dataclass(frozen=True, slots=True)
class ArmRegistration:
    """
    A retrieval arm factory bound to the name that selects it.

    Attributes
    ----------
    name:
        Selector matching a configured arm name, and the name recorded in
        diagnostics when the arm degrades.
    factory:
        Callable producing an arm for one engine.
    origin:
        Where the registration came from, so that a collision names both
        claimants rather than reporting only that one occurred.
    """

    name: str
    factory: ArmFactory
    origin: str = "explicit"


# ============================================================================
# ARM REGISTRY
# ============================================================================


class ArmRegistry:
    """
    Resolves the retrieval arms that will serve a query.

    Immutable once constructed.

    Parameters
    ----------
    registrations:
        The arms available.

    Raises
    ------
    StrategyRegistrationError
        If a name is malformed, a name is registered twice, or a factory is
        not callable.
    """

    __slots__ = ("_registrations",)

    def __init__(self, registrations: Iterable[ArmRegistration]) -> None:
        resolved: dict[str, ArmRegistration] = {}

        for registration in registrations:
            name = _validated_name(registration.name, kind="retrieval arm")

            existing = resolved.get(name)
            if existing is not None:
                raise StrategyRegistrationError(
                    f"Duplicate retrieval arm name {name!r}: {existing.origin} "
                    f"and {registration.origin} both claim it. A name "
                    f"collision is never an override here, because an arm "
                    f"silently replacing another would search a different "
                    f"index while diagnostics still named the original, "
                    f"attributing a degraded result to a component that never "
                    f"ran. Register the alternative under its own name.",
                    strategy_name=name,
                )

            if not callable(registration.factory):
                raise StrategyRegistrationError(
                    f"Retrieval arm factory for {name!r} is not callable.",
                    strategy_name=name,
                )

            resolved[name] = ArmRegistration(
                name=name, factory=registration.factory, origin=registration.origin
            )

        self._registrations: Mapping[str, ArmRegistration] = MappingProxyType(
            resolved
        )

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_entry_points(
        cls,
        *,
        registrations: Iterable[ArmRegistration] = (),
        group: str = ARM_ENTRY_POINT_GROUP,
    ) -> Self:
        """
        Build a registry from installed entry points plus explicit entries.

        Allows a separately installed package to contribute a retrieval method
        with no edit to PTFM. Each entry point must resolve to a factory
        accepting a configuration.

        A discovered name colliding with an explicit one is an error rather
        than an override, for the reason given in the module documentation.

        Parameters
        ----------
        registrations:
            Arms supplied directly by the composition root.
        group:
            Entry-point group to discover.

        Returns
        -------
        Self
            The constructed registry.

        Raises
        ------
        StrategyRegistrationError
            If an entry point cannot be loaded, or collides with another
            registration.
        """
        discovered = [
            ArmRegistration(
                name=point.name,
                factory=_load_entry_point(point, expected="retrieval arm factory"),
                origin=f"entry_point:{point.name}",
            )
            for point in entry_points(group=group)
        ]
        return cls([*discovered, *registrations])

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def names(self) -> tuple[str, ...]:
        """
        Registered arm names, in sorted order.

        Returns
        -------
        tuple[str, ...]
            Available arm names.
        """
        return tuple(sorted(self._registrations))

    def describe(self) -> Mapping[str, str]:
        """
        Summarize registrations for startup diagnostics.

        Returns
        -------
        Mapping[str, str]
            Arm names mapped to registration origin.
        """
        return MappingProxyType(
            {
                name: registration.origin
                for name, registration in sorted(self._registrations.items())
            }
        )

    # ------------------------------------------------------------------
    # Resolution
    # ------------------------------------------------------------------

    def resolve(
        self,
        config: RetrievalConfig,
    ) -> tuple[tuple[RetrievalStrategy, ArmConfig], ...]:
        """
        Construct every enabled arm, with the configuration that governs it.

        Returned together rather than one at a time, because a configuration
        enabling no arm must be refused before the query path commits to it: a
        query served by no arm returns an empty result, which is
        indistinguishable from an absence of evidence and would cause a
        refusal to answer a question the corpus supports.

        Each arm is paired with its own configuration so that the engine reads
        a deadline and a fetch depth from one place rather than re-deriving
        them per arm.

        Parameters
        ----------
        config:
            The configuration naming the arms to resolve.

        Returns
        -------
        tuple[tuple[RetrievalStrategy, ArmConfig], ...]
            Constructed arms with their configuration, in configuration order.

        Raises
        ------
        StrategyRegistrationError
            If a configured arm is not registered, or a factory failed or
            returned the wrong type.
        RetrievalConfigurationError
            If no arm is enabled.
        BackendUnavailableError
            If an arm's optional dependency is not installed.
        """
        enabled = config.enabled_arms()

        if not enabled:
            raise RetrievalConfigurationError(
                "No retrieval arm is enabled. A query served by no arm returns "
                "an empty result, which cannot be distinguished from an "
                "absence of evidence and would cause a refusal to answer a "
                "question the corpus may well support.",
                context={"registered_arms": list(self.names)},
            )

        resolved: list[tuple[RetrievalStrategy, ArmConfig]] = []

        for name in enabled:
            registration = self._registrations.get(name)
            if registration is None:
                raise StrategyRegistrationError(
                    f"No retrieval arm registered under {name!r}. Registered "
                    f"arms: {', '.join(self.names) or 'none'}.",
                    strategy_name=name,
                )

            arm = _construct_arm(registration, config)
            _verify_declared_name(arm, name)
            resolved.append((arm, config.arm_config(name)))

        return tuple(resolved)

    def __len__(self) -> int:
        """
        Number of registered arms.

        Returns
        -------
        int
            Arm count.
        """
        return len(self._registrations)

    def __repr__(self) -> str:
        """
        Return a representation listing registered arm names.

        Returns
        -------
        str
            Diagnostic representation.
        """
        return f"ArmRegistry(arms={list(self.names)!r})"


# ============================================================================
# FUSION REGISTRY
# ============================================================================


class FusionRegistry:
    """
    Resolves the strategy that combines the arms' orderings.

    Immutable once constructed. Strategies are held as instances rather than
    factories, because fusion is pure and stateless: it performs no I/O, owns
    no connection, and has no lifecycle to manage.

    Parameters
    ----------
    strategies:
        The strategies available, keyed by their own declared names.

    Raises
    ------
    StrategyRegistrationError
        If a name is malformed, registered twice, or the strategy does not
        implement the fusion contract.
    """

    __slots__ = ("_strategies",)

    def __init__(self, strategies: Iterable[FusionStrategy]) -> None:
        resolved: dict[str, FusionStrategy] = {}

        for strategy in strategies:
            # Verified here as well as at entry-point discovery. Direct
            # registration is the composition root's path and the one used by
            # every deployment, so validating only discovered strategies would
            # apply the weaker check to the commoner case.
            if not isinstance(strategy, FusionStrategy):
                raise StrategyRegistrationError(
                    f"{type(strategy).__name__} does not implement "
                    f"FusionStrategy. A strategy that does not satisfy the "
                    f"contract would be resolved successfully and fail at the "
                    f"first query, after the arms had already run.",
                    strategy_name=getattr(strategy, "name", None),
                )

            name = _validated_name(strategy.name, kind="fusion strategy")

            if name in resolved:
                raise StrategyRegistrationError(
                    f"Duplicate fusion strategy name {name!r}. Two strategies "
                    f"sharing a name would make the ranking depend on "
                    f"registration order, so a result could not be reproduced "
                    f"from its configuration.",
                    strategy_name=name,
                )

            resolved[name] = strategy

        self._strategies: Mapping[str, FusionStrategy] = MappingProxyType(resolved)

    @classmethod
    def from_entry_points(
        cls,
        *,
        strategies: Iterable[FusionStrategy] = (),
        group: str = FUSION_ENTRY_POINT_GROUP,
    ) -> Self:
        """
        Build a registry from installed entry points plus explicit entries.

        Each entry point must resolve to a zero-argument callable returning a
        strategy.

        Parameters
        ----------
        strategies:
            Strategies supplied directly by the composition root.
        group:
            Entry-point group to discover.

        Returns
        -------
        Self
            The constructed registry.

        Raises
        ------
        StrategyRegistrationError
            If an entry point cannot be loaded, produces the wrong type, or
            collides with another registration.
        """
        discovered: list[FusionStrategy] = []

        for point in entry_points(group=group):
            factory = _load_entry_point(point, expected="fusion strategy factory")
            try:
                instance = factory()
            except Exception as exc:
                raise StrategyRegistrationError(
                    f"Fusion strategy entry point {point.name!r} raised "
                    f"{type(exc).__name__} while constructing: {exc}",
                    strategy_name=point.name,
                ) from exc

            if not isinstance(instance, FusionStrategy):
                raise StrategyRegistrationError(
                    f"Fusion strategy entry point {point.name!r} produced "
                    f"{type(instance).__name__}, which does not implement "
                    f"FusionStrategy.",
                    strategy_name=point.name,
                )
            discovered.append(instance)

        return cls([*discovered, *strategies])

    @property
    def names(self) -> tuple[str, ...]:
        """
        Registered strategy names, in sorted order.

        Returns
        -------
        tuple[str, ...]
            Available strategy names.
        """
        return tuple(sorted(self._strategies))

    def describe(self) -> Mapping[str, str]:
        """
        Summarize registrations for startup diagnostics.

        Returns
        -------
        Mapping[str, str]
            Strategy names mapped to implementation type.
        """
        return MappingProxyType(
            {
                name: type(strategy).__name__
                for name, strategy in sorted(self._strategies.items())
            }
        )

    def resolve(self, config: RetrievalConfig) -> FusionStrategy:
        """
        Resolve the configured fusion strategy.

        Parameters
        ----------
        config:
            The configuration naming the strategy.

        Returns
        -------
        FusionStrategy
            The strategy.

        Raises
        ------
        StrategyRegistrationError
            If the named strategy is not registered.
        """
        name = config.fusion.strategy
        strategy = self._strategies.get(name)

        if strategy is None:
            raise StrategyRegistrationError(
                f"No fusion strategy registered under {name!r}. Registered "
                f"strategies: {', '.join(self.names) or 'none'}.",
                strategy_name=name,
            )

        return strategy

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
        Return a representation listing registered strategy names.

        Returns
        -------
        str
            Diagnostic representation.
        """
        return f"FusionRegistry(strategies={list(self.names)!r})"


# ============================================================================
# HELPERS
# ============================================================================


def _validated_name(name: str, *, kind: str) -> str:
    """
    Verify that a component name is well formed.

    The grammar is restricted so that a name renders identically wherever it
    appears: in configuration, in an environment variable, in an entry-point
    declaration, and in a diagnostic. A name differing by case or punctuation
    between those places would resolve in one and fail in another.

    Parameters
    ----------
    name:
        The declared name.
    kind:
        Component kind, for the error message.

    Returns
    -------
    str
        The validated name.

    Raises
    ------
    StrategyRegistrationError
        If the name is malformed.
    """
    if not _NAME_PATTERN.match(name or ""):
        raise StrategyRegistrationError(
            f"Invalid {kind} name {name!r}. A name must begin with a lowercase "
            f"letter and contain only lowercase letters, digits, and "
            f"underscores, so that it renders identically in configuration, in "
            f"an environment variable, and in a diagnostic.",
            strategy_name=name or None,
        )
    return name


def _construct_arm(
    registration: ArmRegistration,
    config: RetrievalConfig,
) -> RetrievalStrategy:
    """
    Invoke an arm factory, converting any failure.

    Parameters
    ----------
    registration:
        The registration to construct from.
    config:
        Configuration passed to the factory.

    Returns
    -------
    RetrievalStrategy
        The constructed arm.

    Raises
    ------
    BackendUnavailableError
        If the arm's optional dependency is absent.
    StrategyRegistrationError
        If the factory failed or returned the wrong type.
    """
    try:
        arm = registration.factory(config)
    except BackendUnavailableError:
        # Already the actionable form, naming the missing dependency and the
        # extra that supplies it.
        raise
    except ImportError as exc:
        # A backend whose optional dependency is absent. Converted here so
        # that the failure names a remedy rather than surfacing as an import
        # error from an unrelated module.
        raise BackendUnavailableError(
            component=f"retrieval arm {registration.name!r}",
            missing_dependency=exc.name or "unknown",
            install_extra="ptfm[retrieval]",
        ) from exc
    except Exception as exc:
        raise StrategyRegistrationError(
            f"Retrieval arm factory for {registration.name!r} "
            f"({registration.origin}) raised {type(exc).__name__}: {exc}",
            strategy_name=registration.name,
        ) from exc

    if not isinstance(arm, RetrievalStrategy):
        raise StrategyRegistrationError(
            f"Retrieval arm factory for {registration.name!r} returned "
            f"{type(arm).__name__}, which does not implement "
            f"RetrievalStrategy.",
            strategy_name=registration.name,
        )

    return arm


def _verify_declared_name(arm: RetrievalStrategy, expected: str) -> None:
    """
    Ensure an arm's own name matches the key it was registered under.

    Drift between the two would make diagnostics unreliable in precisely the
    situation they exist for: an arm registered as ``dense`` but reporting
    ``sparse`` would attribute its own degradation to another component, and
    an operator would investigate the wrong thing.

    Parameters
    ----------
    arm:
        The constructed arm.
    expected:
        The name it was registered under.

    Returns
    -------
    None

    Raises
    ------
    StrategyRegistrationError
        If the names differ.
    """
    declared = arm.name

    if declared == expected:
        return

    raise StrategyRegistrationError(
        f"Retrieval arm registered as {expected!r} reports its own name as "
        f"{declared!r}. Diagnostics record the reported name, so the "
        f"discrepancy would attribute one arm's degradation to another.",
        strategy_name=expected,
        context={"declared_name": declared},
    )


def _load_entry_point(point: EntryPoint, *, expected: str) -> Callable[..., object]:
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

    Returns
    -------
    Callable[..., object]
        The loaded callable.

    Raises
    ------
    StrategyRegistrationError
        If loading failed or produced a non-callable.
    """
    try:
        loaded = point.load()
    except Exception as exc:
        raise StrategyRegistrationError(
            f"Entry point {point.name!r} in group {point.group!r} could not be "
            f"loaded: {type(exc).__name__}: {exc}",
            strategy_name=point.name,
        ) from exc

    if not callable(loaded):
        raise StrategyRegistrationError(
            f"Entry point {point.name!r} in group {point.group!r} resolved to "
            f"{type(loaded).__name__}, which is not a callable {expected}.",
            strategy_name=point.name,
        )

    return loaded
