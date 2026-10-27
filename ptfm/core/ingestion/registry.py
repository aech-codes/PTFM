"""
PTFM Data Ingestion Registries

File:
    ptfm/core/ingestion/registry.py

Description:
    Resolves URI schemes to source connectors and media types to content
    extractors. The registries are the sole dispatch mechanism in the
    Data Ingestion Module: because resolution happens here, the engine
    contains no transport branching and no format branching, and a new
    connector or extractor can be added without editing any existing file.

Resolution rules
----------------
    Connectors dispatch purely by URI scheme. One connector per scheme;
    a collision is a configuration error.

    Extractors dispatch by media type in strict specificity order:

        1. exact match          "application/pdf"
        2. subtype wildcard     "text/*"
        3. universal wildcard   "*/*"

    Priority breaks ties WITHIN a tier only. Specificity always dominates,
    so a high-priority universal fallback can never shadow a purpose-built
    extractor. An equal-priority collision within one tier is rejected at
    construction rather than resolved by insertion order, which would make
    dispatch depend on import sequence.

Immutability
------------
    Both registries are constructed complete and are immutable thereafter.
    There is no registration method and no mutable phase, so resolution
    cannot change during a run.

Dependency contract
-------------------
    Imports only the standard library, ``ptfm.core.ingestion.exceptions``,
    ``ptfm.core.ingestion.config`` and ``ptfm.core.ingestion.interfaces``.
    It must not import ``models``, any stage, or any concrete connector or
    extractor. The Core references abstractions only.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from importlib.metadata import EntryPoint, entry_points
from types import MappingProxyType
from typing import Callable, Final, Iterable, Mapping, Optional, Self, TypeAlias

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
    normalize_media_type,
)

__all__ = [
    "CONNECTOR_ENTRY_POINT_GROUP",
    "EXTRACTOR_ENTRY_POINT_GROUP",
    "ConnectorFactory",
    "ConnectorRegistration",
    "SkippedExtractor",
    "ConnectorRegistry",
    "ExtractorRegistry",
]

#: Entry-point groups through which a separately installed package may
#: contribute a connector or extractor without any edit to PTFM.
CONNECTOR_ENTRY_POINT_GROUP: Final[str] = "ptfm.ingestion.connectors"
EXTRACTOR_ENTRY_POINT_GROUP: Final[str] = "ptfm.ingestion.extractors"

#: Constructs a connector for one ingestion run. A connector class whose
#: ``__init__`` accepts an :class:`IngestionConfig` satisfies this directly.
ConnectorFactory: TypeAlias = Callable[[IngestionConfig], SourceConnector]

#: URI scheme grammar, per RFC 3986 section 3.1, lowercased.
_SCHEME_PATTERN: Final[re.Pattern[str]] = re.compile(r"^[a-z][a-z0-9+.\-]*$")

#: Matches a subtype wildcard declaration such as ``"text/*"``.
_SUBTYPE_WILDCARD_PATTERN: Final[re.Pattern[str]] = re.compile(r"^[a-z0-9!#$&^_.+\-]+/\*$")


# ============================================================================
# REGISTRATION RECORDS
# ============================================================================


@dataclass(frozen=True, slots=True)
class ConnectorRegistration:
    """
    One connector bound to the URI scheme it serves.

    The scheme is carried explicitly because ``SourceConnector.scheme`` is
    an instance property and cannot be read from a factory without
    constructing a connector. :meth:`ConnectorRegistry.create` verifies
    that the constructed connector agrees with the registered scheme, so
    drift between the two is caught deterministically at first use.

    Attributes
    ----------
    scheme:
        Lowercase URI scheme this connector serves.
    factory:
        Callable producing a connector for one run.
    origin:
        Where the registration came from, for diagnostics. ``"explicit"``
        for composition-root registration, or ``"entry_point:<name>"``.
    """

    scheme: str
    factory: ConnectorFactory
    origin: str = "explicit"


@dataclass(frozen=True, slots=True)
class SkippedExtractor:
    """
    An extractor excluded from a registry, with the reason.

    Exposed as data rather than logged so that the registry performs no
    I/O and so that exclusions are directly assertable in tests. The
    composition root is responsible for reporting these at startup.

    Attributes
    ----------
    extractor_name:
        Identifier of the excluded extractor.
    reason:
        Why it was excluded.
    """

    extractor_name: str
    reason: str


# ============================================================================
# CONNECTOR REGISTRY
# ============================================================================


class ConnectorRegistry:
    """
    Maps URI schemes to source connector factories.

    Immutable once constructed. Dispatch by scheme alone is what keeps the
    ingestion engine free of any transport-specific branching.

    Parameters
    ----------
    registrations:
        The complete set of connector registrations.

    Raises
    ------
    ConnectorRegistrationError
        If a scheme is malformed, a factory is not callable, or two
        registrations claim the same scheme.
    """

    __slots__ = ("_by_scheme",)

    def __init__(self, registrations: Iterable[ConnectorRegistration]) -> None:
        resolved: dict[str, ConnectorRegistration] = {}

        for registration in registrations:
            scheme = registration.scheme.strip().lower()

            if not _SCHEME_PATTERN.match(scheme):
                raise ConnectorRegistrationError(
                    f"Invalid URI scheme {registration.scheme!r}. A scheme must "
                    f"begin with a letter and contain only letters, digits, "
                    f"'+', '-' or '.', and must not include '://'.",
                    scheme=registration.scheme,
                )

            if not callable(registration.factory):
                raise ConnectorRegistrationError(
                    f"Connector factory for scheme {scheme!r} is not callable",
                    scheme=scheme,
                )

            existing = resolved.get(scheme)
            if existing is not None:
                raise ConnectorRegistrationError(
                    f"Duplicate connector registration for scheme {scheme!r}: "
                    f"{existing.origin} and {registration.origin} both claim it. "
                    f"Exactly one connector may serve a scheme.",
                    scheme=scheme,
                )

            resolved[scheme] = ConnectorRegistration(
                scheme=scheme,
                factory=registration.factory,
                origin=registration.origin,
            )

        self._by_scheme: Mapping[str, ConnectorRegistration] = MappingProxyType(resolved)

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_entry_points(
        cls,
        *,
        group: str = CONNECTOR_ENTRY_POINT_GROUP,
        additional: Iterable[ConnectorRegistration] = (),
    ) -> Self:
        """
        Build a registry from installed entry points plus explicit entries.

        Allows a separately installed package to contribute a connector
        with no edit to PTFM. The entry point must resolve to a callable
        accepting an :class:`IngestionConfig`; a connector class whose
        ``__init__`` takes one satisfies this directly. The entry-point
        name supplies the scheme.

        Parameters
        ----------
        group:
            Entry-point group to scan.
        additional:
            Registrations supplied directly by the composition root.
            Combined with discovered entries; a collision with a
            discovered scheme is an error, not an override.

        Returns
        -------
        Self
            The constructed registry.

        Raises
        ------
        ConnectorRegistrationError
            If an entry point cannot be loaded, or any registration is
            invalid or collides.
        """
        discovered = [
            ConnectorRegistration(
                scheme=point.name,
                factory=_load_entry_point(point, expected="connector factory"),
                origin=f"entry_point:{point.name}",
            )
            for point in entry_points(group=group)
        ]
        return cls([*discovered, *additional])

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def supported_schemes(self) -> tuple[str, ...]:
        """
        Schemes this registry can dispatch, in sorted order.

        Returns
        -------
        tuple[str, ...]
            Registered schemes.
        """
        return tuple(sorted(self._by_scheme))

    def supports(self, scheme: str) -> bool:
        """
        Report whether a scheme is registered.

        Parameters
        ----------
        scheme:
            URI scheme to test; compared case-insensitively.

        Returns
        -------
        bool
            ``True`` when a connector is registered for the scheme.
        """
        return scheme.strip().lower() in self._by_scheme

    def describe(self) -> Mapping[str, str]:
        """
        Summarize registrations for startup diagnostics.

        Returns
        -------
        Mapping[str, str]
            Scheme to registration origin.
        """
        return MappingProxyType(
            {scheme: reg.origin for scheme, reg in self._by_scheme.items()}
        )

    # ------------------------------------------------------------------
    # Resolution
    # ------------------------------------------------------------------

    def create(self, scheme: str, config: IngestionConfig) -> SourceConnector:
        """
        Construct the connector serving a scheme.

        A fresh connector is produced per call because connectors own
        transport resources with a per-run lifecycle. The caller is
        responsible for closing it, ordinarily via ``async with``.

        Parameters
        ----------
        scheme:
            URI scheme to resolve; compared case-insensitively.
        config:
            Configuration injected into the factory.

        Returns
        -------
        SourceConnector
            A connector serving the scheme.

        Raises
        ------
        UnsupportedSourceSchemeError
            If no connector is registered for the scheme.
        ConnectorRegistrationError
            If the factory fails, returns a non-connector, or returns a
            connector whose own scheme disagrees with the registered one.
        """
        normalized = scheme.strip().lower()
        registration = self._by_scheme.get(normalized)
        if registration is None:
            raise UnsupportedSourceSchemeError(
                scheme=normalized,
                supported_schemes=self.supported_schemes,
            )

        try:
            connector = registration.factory(config)
        except Exception as exc:
            raise ConnectorRegistrationError(
                f"Connector factory for scheme {normalized!r} "
                f"({registration.origin}) raised {type(exc).__name__}: {exc}",
                scheme=normalized,
            ) from exc

        if not isinstance(connector, SourceConnector):
            raise ConnectorRegistrationError(
                f"Connector factory for scheme {normalized!r} returned "
                f"{type(connector).__name__}, which does not implement "
                f"SourceConnector",
                scheme=normalized,
            )

        if connector.scheme.strip().lower() != normalized:
            raise ConnectorRegistrationError(
                f"Connector {connector.name!r} is registered for scheme "
                f"{normalized!r} but declares scheme {connector.scheme!r}. "
                f"The registered scheme and the connector's own scheme must "
                f"agree, or dispatch will not match the connector's behaviour.",
                scheme=normalized,
            )

        return connector

    def __len__(self) -> int:
        """
        Number of registered schemes.

        Returns
        -------
        int
            Registration count.
        """
        return len(self._by_scheme)

    def __repr__(self) -> str:
        """
        Return a representation listing registered schemes.

        Returns
        -------
        str
            Diagnostic representation.
        """
        return f"ConnectorRegistry(schemes={list(self.supported_schemes)!r})"


# ============================================================================
# EXTRACTOR REGISTRY
# ============================================================================


class ExtractorRegistry:
    """
    Maps media types to content extractors.

    Immutable once constructed. Resolution tables are precomputed, so
    dispatch is a constant-time lookup rather than a scan across
    extractors — which matters because it happens once per document.

    Resolution order is strictly by specificity: exact match, then subtype
    wildcard, then universal wildcard. Priority breaks ties only within a
    tier, so a high-priority universal fallback cannot shadow a
    purpose-built extractor.

    Parameters
    ----------
    extractors:
        Candidate extractors. Those named in
        ``config.disabled_extractors`` are excluded, and those whose
        optional dependencies are absent are either rejected or skipped
        according to ``config.strict_availability``.
    config:
        Extraction configuration. Deliberately narrower than the full
        ingestion configuration: only the deny-list and the availability
        policy are relevant here.

    Raises
    ------
    ExtractorRegistrationError
        If an extractor declares no media types, declares a malformed
        pattern, duplicates another extractor's name, or collides with
        another extractor at equal priority for the same pattern.
    ExtractorUnavailableError
        If an optional dependency is absent and strict availability is in
        force.
    """

    __slots__ = ("_exact", "_subtype", "_universal", "_extractors", "_skipped")

    def __init__(
        self,
        extractors: Iterable[ContentExtractor],
        *,
        config: ExtractionConfig,
    ) -> None:
        admitted: list[ContentExtractor] = []
        skipped: list[SkippedExtractor] = []
        seen_names: set[str] = set()

        for extractor in extractors:
            name = extractor.name

            if name in seen_names:
                raise ExtractorRegistrationError(
                    f"Duplicate extractor name {name!r}. Extractor names must "
                    f"be unique; they identify extractors in diagnostics and "
                    f"in the disabled-extractor deny-list."
                )
            seen_names.add(name)

            if name in config.disabled_extractors:
                skipped.append(
                    SkippedExtractor(extractor_name=name, reason="disabled by configuration")
                )
                continue

            try:
                type(extractor).check_availability()
            except ExtractorUnavailableError as exc:
                if config.strict_availability:
                    raise
                skipped.append(
                    SkippedExtractor(
                        extractor_name=name,
                        reason=f"unavailable: {exc.missing_dependency} "
                        f"(install extra {exc.install_extra})",
                    )
                )
                continue

            admitted.append(extractor)

        exact: dict[str, ContentExtractor] = {}
        subtype: dict[str, ContentExtractor] = {}
        universal: Optional[ContentExtractor] = None

        for extractor in admitted:
            patterns = extractor.supported_media_types
            if not patterns:
                raise ExtractorRegistrationError(
                    f"Extractor {extractor.name!r} declares no supported media "
                    f"types and could never be resolved"
                )

            for raw_pattern in patterns:
                pattern = _validate_pattern(raw_pattern, extractor_name=extractor.name)

                if pattern == MEDIA_TYPE_WILDCARD:
                    universal = _select_winner(universal, extractor, pattern)
                elif _SUBTYPE_WILDCARD_PATTERN.match(pattern):
                    primary = pattern.split("/", 1)[0]
                    subtype[primary] = _select_winner(
                        subtype.get(primary), extractor, pattern
                    )
                else:
                    exact[pattern] = _select_winner(
                        exact.get(pattern), extractor, pattern
                    )

        self._exact: Mapping[str, ContentExtractor] = MappingProxyType(exact)
        self._subtype: Mapping[str, ContentExtractor] = MappingProxyType(subtype)
        self._universal: Optional[ContentExtractor] = universal
        self._extractors: tuple[ContentExtractor, ...] = tuple(admitted)
        self._skipped: tuple[SkippedExtractor, ...] = tuple(skipped)

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_entry_points(
        cls,
        *,
        config: ExtractionConfig,
        group: str = EXTRACTOR_ENTRY_POINT_GROUP,
        additional: Iterable[ContentExtractor] = (),
    ) -> Self:
        """
        Build a registry from installed entry points plus explicit entries.

        Each entry point must resolve to a zero-argument callable
        returning a :class:`ContentExtractor`; an extractor class with a
        default constructor satisfies this directly.

        Parameters
        ----------
        config:
            Extraction configuration governing exclusions and availability.
        group:
            Entry-point group to scan.
        additional:
            Extractors supplied directly by the composition root. A
            third-party extractor overrides a built-in by declaring a
            higher priority, never by registration order.

        Returns
        -------
        Self
            The constructed registry.

        Raises
        ------
        ExtractorRegistrationError
            If an entry point cannot be loaded or produces a non-extractor.
        ExtractorUnavailableError
            If an optional dependency is absent under strict availability.
        """
        discovered: list[ContentExtractor] = []
        for point in entry_points(group=group):
            factory = _load_entry_point(point, expected="extractor factory")
            try:
                instance = factory()
            except Exception as exc:
                raise ExtractorRegistrationError(
                    f"Extractor entry point {point.name!r} raised "
                    f"{type(exc).__name__} during construction: {exc}"
                ) from exc
            if not isinstance(instance, ContentExtractor):
                raise ExtractorRegistrationError(
                    f"Extractor entry point {point.name!r} produced "
                    f"{type(instance).__name__}, which does not implement "
                    f"ContentExtractor"
                )
            discovered.append(instance)

        return cls([*discovered, *additional], config=config)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def extractors(self) -> tuple[ContentExtractor, ...]:
        """
        Extractors admitted to this registry.

        Returns
        -------
        tuple[ContentExtractor, ...]
            Admitted extractors, in registration order.
        """
        return self._extractors

    @property
    def skipped_extractors(self) -> tuple[SkippedExtractor, ...]:
        """
        Extractors excluded from this registry, with reasons.

        Exposed for the composition root to log at startup. The registry
        performs no I/O of its own.

        Returns
        -------
        tuple[SkippedExtractor, ...]
            Exclusion records.
        """
        return self._skipped

    @property
    def supported_media_types(self) -> tuple[str, ...]:
        """
        Media type patterns this registry can dispatch, in sorted order.

        Returns
        -------
        tuple[str, ...]
            Exact types and wildcard patterns.
        """
        patterns = set(self._exact)
        patterns.update(f"{primary}/*" for primary in self._subtype)
        if self._universal is not None:
            patterns.add(MEDIA_TYPE_WILDCARD)
        return tuple(sorted(patterns))

    def describe(self) -> Mapping[str, str]:
        """
        Summarize resolution for startup diagnostics.

        Returns
        -------
        Mapping[str, str]
            Media type pattern to the extractor name that wins it.
        """
        mapping = {pattern: ex.name for pattern, ex in self._exact.items()}
        mapping.update({f"{p}/*": ex.name for p, ex in self._subtype.items()})
        if self._universal is not None:
            mapping[MEDIA_TYPE_WILDCARD] = self._universal.name
        return MappingProxyType(dict(sorted(mapping.items())))

    # ------------------------------------------------------------------
    # Resolution
    # ------------------------------------------------------------------

    def resolve_or_none(self, media_type: str) -> Optional[ContentExtractor]:
        """
        Resolve the extractor for a media type, or ``None``.

        Matching proceeds strictly by specificity: exact type, then
        subtype wildcard, then universal wildcard. Specificity always
        dominates priority.

        Parameters
        ----------
        media_type:
            Media type to resolve. Normalized before lookup, so
            parameters and casing are immaterial.

        Returns
        -------
        Optional[ContentExtractor]
            The winning extractor, or ``None`` if nothing matches.

        Raises
        ------
        ExtractorRegistrationError
            If the media type is malformed.
        """
        try:
            normalized = normalize_media_type(media_type)
        except ValueError as exc:
            raise ExtractorRegistrationError(
                f"Cannot resolve malformed media type {media_type!r}",
                media_type=media_type,
            ) from exc

        exact = self._exact.get(normalized)
        if exact is not None:
            return exact

        primary = normalized.split("/", 1)[0]
        wildcard = self._subtype.get(primary)
        if wildcard is not None:
            return wildcard

        return self._universal

    def resolve(self, media_type: str) -> ContentExtractor:
        """
        Resolve the extractor for a media type.

        Parameters
        ----------
        media_type:
            Media type to resolve.

        Returns
        -------
        ContentExtractor
            The winning extractor.

        Raises
        ------
        UnsupportedMediaTypeError
            If no extractor claims the media type. The message lists every
            registered pattern, so the misconfiguration is self-diagnosing.
        ExtractorRegistrationError
            If the media type is malformed.
        """
        extractor = self.resolve_or_none(media_type)
        if extractor is None:
            raise UnsupportedMediaTypeError(
                media_type=normalize_media_type(media_type),
                supported_media_types=self.supported_media_types,
            )
        return extractor

    def supports(self, media_type: str) -> bool:
        """
        Report whether any extractor claims a media type.

        Parameters
        ----------
        media_type:
            Media type to test.

        Returns
        -------
        bool
            ``True`` when the media type resolves.

        Raises
        ------
        ExtractorRegistrationError
            If the media type is malformed.
        """
        return self.resolve_or_none(media_type) is not None

    def __len__(self) -> int:
        """
        Number of admitted extractors.

        Returns
        -------
        int
            Admitted extractor count.
        """
        return len(self._extractors)

    def __repr__(self) -> str:
        """
        Return a representation listing admitted extractors.

        Returns
        -------
        str
            Diagnostic representation.
        """
        names = [extractor.name for extractor in self._extractors]
        return f"ExtractorRegistry(extractors={names!r}, skipped={len(self._skipped)})"


# ============================================================================
# HELPERS
# ============================================================================


def _validate_pattern(pattern: str, *, extractor_name: str) -> str:
    """
    Normalize and validate a declared media type pattern.

    Parameters
    ----------
    pattern:
        Pattern as declared by an extractor.
    extractor_name:
        Declaring extractor, for the error message.

    Returns
    -------
    str
        The normalized pattern.

    Raises
    ------
    ExtractorRegistrationError
        If the pattern is malformed or uses an unsupported wildcard form.
    """
    try:
        normalized = normalize_media_type(pattern)
    except ValueError as exc:
        raise ExtractorRegistrationError(
            f"Extractor {extractor_name!r} declares malformed media type "
            f"{pattern!r}",
            media_type=pattern,
        ) from exc

    if normalized == MEDIA_TYPE_WILDCARD:
        return normalized
    if _SUBTYPE_WILDCARD_PATTERN.match(normalized):
        return normalized
    if "*" in normalized:
        raise ExtractorRegistrationError(
            f"Extractor {extractor_name!r} declares unsupported wildcard form "
            f"{pattern!r}. Only an exact type, a subtype wildcard such as "
            f"'text/*', or the universal wildcard '*/*' are permitted.",
            media_type=pattern,
        )
    if "/" not in normalized:
        raise ExtractorRegistrationError(
            f"Extractor {extractor_name!r} declares media type {pattern!r} "
            f"without a subtype",
            media_type=pattern,
        )
    return normalized


def _select_winner(
    incumbent: Optional[ContentExtractor],
    challenger: ContentExtractor,
    pattern: str,
) -> ContentExtractor:
    """
    Choose between two extractors claiming the same pattern.

    Higher priority wins. An equal-priority collision is rejected rather
    than resolved by insertion order, which would make dispatch depend on
    import sequence and produce different behaviour between environments.

    Parameters
    ----------
    incumbent:
        Extractor currently holding the pattern, if any.
    challenger:
        Extractor also claiming it.
    pattern:
        The contested pattern, for the error message.

    Returns
    -------
    ContentExtractor
        The winning extractor.

    Raises
    ------
    ExtractorRegistrationError
        If both claim the pattern at equal priority.
    """
    if incumbent is None:
        return challenger
    if challenger.priority > incumbent.priority:
        return challenger
    if challenger.priority < incumbent.priority:
        return incumbent
    raise ExtractorRegistrationError(
        f"Extractors {incumbent.name!r} and {challenger.name!r} both claim "
        f"media type {pattern!r} at priority {incumbent.priority}. Resolution "
        f"would depend on registration order; raise the priority of the "
        f"intended extractor to make it deterministic.",
        media_type=pattern,
    )


def _load_entry_point(point: EntryPoint, *, expected: str) -> Callable[..., object]:
    """
    Load an entry point, converting any failure into a PTFM exception.

    A third-party package may raise anything at import. Wrapping here
    means a broken plugin produces an actionable startup message naming
    the entry point, rather than an arbitrary traceback from an unrelated
    module.

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
    ConnectorRegistrationError
        If the entry point belongs to the connector group and fails.
    ExtractorRegistrationError
        If the entry point belongs to the extractor group and fails.
    """
    is_connector = point.group == CONNECTOR_ENTRY_POINT_GROUP
    try:
        loaded = point.load()
    except Exception as exc:
        message = (
            f"Entry point {point.name!r} in group {point.group!r} could not be "
            f"loaded: {type(exc).__name__}: {exc}"
        )
        if is_connector:
            raise ConnectorRegistrationError(message, scheme=point.name) from exc
        raise ExtractorRegistrationError(message) from exc

    if not callable(loaded):
        message = (
            f"Entry point {point.name!r} in group {point.group!r} resolved to "
            f"{type(loaded).__name__}, which is not a callable {expected}"
        )
        if is_connector:
            raise ConnectorRegistrationError(message, scheme=point.name)
        raise ExtractorRegistrationError(message)

    return loaded