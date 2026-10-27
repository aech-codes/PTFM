"""
PTFM Metadata Harvesting Stage

File:
    ptfm/core/ingestion/pipeline/metadata.py

Description:
    Merges the format-native metadata reported by a content extractor with
    natural language detection and counts computed from the normalized
    document text, producing the canonical ``DocumentMetadata``.

Scope
-----
    Strictly format-generic. Every field produced here exists independently
    of any knowledge domain: titles, authors, timestamps, page counts,
    language, character and word counts.

    Domain metadata such as jurisdiction, ticker symbol, diagnostic code, or
    study design is NOT produced here. It is the responsibility of Knowledge
    Processing and of domain plugins. This stage contains no domain
    vocabulary and no domain-specific parsing.

Confidence prohibition
----------------------
    Language detection reports a self-reported certainty. That value governs
    ONLY whether a diagnostic ``ExtractionWarning`` is attached.

    A low-confidence detection still populates ``detected_language``.
    Discarding content or metadata on the basis of a model score would make
    a raw neural confidence into a gate, which is prohibited outside the
    conformal calibration layer. The confidence never appears as a field on
    ``DocumentMetadata`` and never reaches the verification, calibration, or
    decision path.

Timestamp policy
----------------
    Parser timestamps are frequently naive: OOXML core properties are
    specified as UTC, and most PDF producers omit an offset. A naive instant
    cannot be compared against the timezone-aware timestamps used elsewhere
    in the pipeline, so naive values are interpreted as UTC and aware values
    are converted to UTC.

    This coercion is not warned about. Warning would emit a diagnostic for
    nearly every Office document ingested, which is noise rather than
    signal.

    A timestamp later than the current instant is definitionally wrong and
    is dropped with a warning. No minimum plausible date is enforced: doing
    so would require a hardcoded year and would risk discarding legitimate
    metadata from genuinely old digitized documents.

Purity contract
---------------
    Synchronous, free of I/O, and holding no mutated instance state, so a
    single instance serves all concurrent workers. Language detection is
    CPU-bound over a bounded sample and is executed by the caller inside the
    same worker thread as structure normalization.

Dependency contract
-------------------
    Imports only the standard library, ``ptfm.core.schemas`` and the
    ingestion ``config``, ``models`` and ``interfaces`` modules.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Final, Optional, Sequence

from ptfm.core.ingestion.config import MetadataConfig
from ptfm.core.ingestion.interfaces import Clock, LanguageDetector
from ptfm.core.ingestion.models import ExtractedMetadata
from ptfm.core.schemas import (
    DocumentMetadata,
    ExtractionWarning,
    ExtractionWarningCode,
)

__all__ = [
    "HarvestedMetadata",
    "MetadataHarvester",
]

#: A run of whitespace, collapsed within metadata strings. Titles extracted
#: from PDFs commonly contain embedded newlines.
_WHITESPACE_RUN: Final[re.Pattern[str]] = re.compile(r"\s+")


# ============================================================================
# RESULT
# ============================================================================


@dataclass(frozen=True, slots=True)
class HarvestedMetadata:
    """
    Canonical metadata paired with the warnings raised while producing it.

    Stage-local: this type carries results from metadata harvesting to
    canonical assembly and is not a cross-module contract. Consumers outside
    the ingestion package depend on ``DocumentMetadata`` instead.

    Attributes
    ----------
    metadata:
        The canonical metadata record.
    warnings:
        Non-fatal observations recorded during harvesting, such as an
        uncertain language determination or a discarded future timestamp.
    """

    metadata: DocumentMetadata
    warnings: tuple[ExtractionWarning, ...] = ()


# ============================================================================
# HARVESTER
# ============================================================================


class MetadataHarvester:
    """
    Produces canonical document metadata from extractor output and text.

    Parameters
    ----------
    config:
        Harvesting configuration: whether language detection runs, its
        sample and minimum input sizes, the certainty below which a warning
        is attached, and the caps on title length and author count.
    clock:
        Supplies the current instant, used to identify and discard
        timestamps dated in the future. Injected so that timestamp handling
        is deterministic under test.
    language_detector:
        Natural language identifier. When ``None``, no language detection is
        performed regardless of configuration, which allows a deployment to
        omit the dependency entirely.
    """

    __slots__ = ("_config", "_clock", "_detector")

    def __init__(
        self,
        config: MetadataConfig,
        clock: Clock,
        language_detector: Optional[LanguageDetector] = None,
    ) -> None:
        self._config = config
        self._clock = clock
        self._detector = language_detector

    # ------------------------------------------------------------------
    # Public contract
    # ------------------------------------------------------------------

    def harvest(
        self,
        extracted: ExtractedMetadata,
        text: str,
    ) -> HarvestedMetadata:
        """
        Merge format-native metadata with computed and detected properties.

        Character count is always computed from the supplied text rather
        than taken from the extractor. ``RawDocument`` requires
        ``char_count`` to equal the text length exactly, so deriving it here
        makes that invariant hold by construction instead of depending on
        every format reporting its own length correctly.

        Parameters
        ----------
        extracted:
            Format-native metadata as reported by the extractor.
        text:
            The normalized document text produced by structure
            normalization.

        Returns
        -------
        HarvestedMetadata
            The canonical metadata record and any non-fatal observations.
        """
        warnings: list[ExtractionWarning] = []

        title = self._resolve_title(extracted.title, warnings)
        authors = self._resolve_authors(extracted.authors, warnings)
        created_at = self._resolve_timestamp(
            extracted.created_at, field_name="created_at", warnings=warnings
        )
        modified_at = self._resolve_timestamp(
            extracted.modified_at, field_name="modified_at", warnings=warnings
        )
        _check_timestamp_ordering(created_at, modified_at, warnings)
        detected_language = self._detect_language(text, warnings)

        metadata = DocumentMetadata(
            title=title,
            authors=authors,
            created_at=created_at,
            modified_at=modified_at,
            declared_language=_clean(extracted.declared_language),
            detected_language=detected_language,
            page_count=extracted.page_count,
            char_count=len(text),
            word_count=_count_words(text),
            producer=_clean(extracted.producer),
            format_properties=dict(extracted.properties),
        )

        return HarvestedMetadata(metadata=metadata, warnings=tuple(warnings))

    # ------------------------------------------------------------------
    # Title and authors
    # ------------------------------------------------------------------

    def _resolve_title(
        self,
        title: Optional[str],
        warnings: list[ExtractionWarning],
    ) -> Optional[str]:
        """
        Clean and bound a declared document title.

        Some formats emit an entire first page as the title, so the length
        is capped. Truncation is reported because a title of that size
        indicates the format's title field is not being used as intended,
        which is useful to know when assessing corpus quality.

        Parameters
        ----------
        title:
            Title as declared by the format.
        warnings:
            Accumulator to which a truncation observation is appended.

        Returns
        -------
        Optional[str]
            The cleaned title, or ``None`` when absent or blank.
        """
        cleaned = _clean(title)
        if cleaned is None:
            return None

        limit = self._config.max_title_chars
        if len(cleaned) <= limit:
            return cleaned

        warnings.append(
            ExtractionWarning(
                code=ExtractionWarningCode.OTHER,
                message=(
                    f"Declared title exceeded {limit} characters and was "
                    f"truncated"
                ),
                detail={
                    "field": "title",
                    "original_length": len(cleaned),
                    "limit": limit,
                },
            )
        )
        return cleaned[:limit]

    def _resolve_authors(
        self,
        authors: Sequence[str],
        warnings: list[ExtractionWarning],
    ) -> tuple[str, ...]:
        """
        Clean, deduplicate, and bound the declared author list.

        Order is preserved because author order is frequently meaningful.
        Duplicates are removed because formats commonly repeat an author
        across several metadata fields.

        Parameters
        ----------
        authors:
            Authors as declared by the format.
        warnings:
            Accumulator to which a truncation observation is appended.

        Returns
        -------
        tuple[str, ...]
            Cleaned authors, in declared order, capped at the configured
            maximum.
        """
        resolved: list[str] = []
        seen: set[str] = set()

        for author in authors:
            cleaned = _clean(author)
            if cleaned is None or cleaned in seen:
                continue
            seen.add(cleaned)
            resolved.append(cleaned)

        limit = self._config.max_authors
        if len(resolved) <= limit:
            return tuple(resolved)

        warnings.append(
            ExtractionWarning(
                code=ExtractionWarningCode.OTHER,
                message=(
                    f"Declared author list contained {len(resolved)} entries "
                    f"and was truncated to {limit}"
                ),
                detail={
                    "field": "authors",
                    "original_count": len(resolved),
                    "limit": limit,
                },
            )
        )
        return tuple(resolved[:limit])

    # ------------------------------------------------------------------
    # Timestamps
    # ------------------------------------------------------------------

    def _resolve_timestamp(
        self,
        value: Optional[datetime],
        *,
        field_name: str,
        warnings: list[ExtractionWarning],
    ) -> Optional[datetime]:
        """
        Coerce a declared timestamp to UTC and reject impossible values.

        Naive instants are interpreted as UTC, which matches the OOXML
        specification and is the most defensible assumption for formats that
        omit an offset. Aware instants are converted to UTC so that all
        stored timestamps are uniformly represented and directly comparable.

        A timestamp later than the current instant is definitionally wrong
        and is discarded. No lower bound is enforced, since any minimum
        plausible date would be an arbitrary hardcoded constant and would
        risk discarding legitimate metadata from old digitized documents.

        Parameters
        ----------
        value:
            Timestamp as declared by the format.
        field_name:
            Field being resolved, for the warning detail.
        warnings:
            Accumulator to which a rejection observation is appended.

        Returns
        -------
        Optional[datetime]
            A timezone-aware UTC instant, or ``None`` when absent or
            impossible.
        """
        if value is None:
            return None

        coerced = (
            value.replace(tzinfo=UTC)
            if value.tzinfo is None
            else value.astimezone(UTC)
        )

        now = self._clock.now()
        if coerced <= now:
            return coerced

        warnings.append(
            ExtractionWarning(
                code=ExtractionWarningCode.METADATA_UNAVAILABLE,
                message=(
                    f"Declared {field_name} is dated in the future and was "
                    f"discarded"
                ),
                detail={
                    "field": field_name,
                    "declared_value": coerced.isoformat(),
                    "observed_at": now.isoformat(),
                },
            )
        )
        return None

    # ------------------------------------------------------------------
    # Language
    # ------------------------------------------------------------------

    def _detect_language(
        self,
        text: str,
        warnings: list[ExtractionWarning],
    ) -> Optional[str]:
        """
        Identify the dominant natural language of the document text.

        Detection runs over a bounded leading sample: examining an entire
        large document is wasteful and no more accurate.

        A low-certainty determination is STILL RECORDED. The configured
        minimum certainty governs only whether a diagnostic warning is
        attached. Discarding a detection on the basis of a model score would
        turn a raw neural confidence into a gate, which is prohibited
        outside the conformal calibration layer.

        Detector failure is non-fatal. Language is advisory metadata, and
        losing it must not quarantine a document whose text extracted
        correctly.

        Parameters
        ----------
        text:
            The normalized document text.
        warnings:
            Accumulator to which an uncertainty or failure observation is
            appended.

        Returns
        -------
        Optional[str]
            A BCP 47 language tag, or ``None`` when detection was skipped,
            yielded nothing, or failed.
        """
        if self._detector is None or not self._config.detect_language:
            return None

        if len(text) < self._config.language_detection_min_chars:
            # Short text is a normal property of many documents, not a
            # fidelity problem, so no warning is recorded.
            return None

        sample = text[: self._config.language_detection_sample_chars]

        try:
            detection = self._detector.detect(sample)
        except Exception as exc:  # noqa: BLE001 - advisory metadata only
            warnings.append(
                ExtractionWarning(
                    code=ExtractionWarningCode.LANGUAGE_DETECTION_UNCERTAIN,
                    message=(
                        f"Language detection failed with "
                        f"{type(exc).__name__} and was skipped"
                    ),
                    detail={"exception_type": type(exc).__name__},
                )
            )
            return None

        if detection is None:
            warnings.append(
                ExtractionWarning(
                    code=ExtractionWarningCode.LANGUAGE_DETECTION_UNCERTAIN,
                    message="Language could not be determined from the text",
                    detail={"sample_chars": len(sample)},
                )
            )
            return None

        language = _clean(detection.language)
        if language is None:
            return None

        if detection.confidence < self._config.language_detection_min_confidence:
            warnings.append(
                ExtractionWarning(
                    code=ExtractionWarningCode.LANGUAGE_DETECTION_UNCERTAIN,
                    message=(
                        f"Language determined as {language!r} with low "
                        f"detector certainty; the determination has been "
                        f"retained and is diagnostic only"
                    ),
                    detail={
                        "detected_language": language,
                        # Diagnostic only. Never consumed as a gate, never
                        # persisted as a field, never reaches verification.
                        "detector_confidence": detection.confidence,
                        "sample_chars": len(sample),
                    },
                )
            )

        return language.lower()


# ============================================================================
# HELPERS
# ============================================================================


def _clean(value: Optional[str]) -> Optional[str]:
    """
    Normalize a metadata string, treating a blank result as absent.

    Whitespace runs are collapsed because titles extracted from PDFs
    commonly contain embedded newlines, and leading or trailing whitespace
    is removed.

    Parameters
    ----------
    value:
        Raw metadata string, possibly ``None``.

    Returns
    -------
    Optional[str]
        The cleaned string, or ``None`` when absent or whitespace only.
    """
    if value is None:
        return None
    cleaned = _WHITESPACE_RUN.sub(" ", value).strip()
    return cleaned or None


def _count_words(text: str) -> int:
    """
    Count whitespace-delimited tokens in the document text.

    Advisory only. Tokenization is not standardized across scripts, and this
    measure is meaningless for unspaced writing systems such as Chinese and
    Japanese. It is provided for diagnostics and coarse corpus statistics,
    never as an input to any decision.

    Parameters
    ----------
    text:
        The normalized document text.

    Returns
    -------
    int
        Token count; zero for empty or whitespace-only text.
    """
    return len(text.split())


def _check_timestamp_ordering(
    created_at: Optional[datetime],
    modified_at: Optional[datetime],
    warnings: list[ExtractionWarning],
) -> None:
    """
    Report a modification timestamp that precedes creation.

    Both values are retained. The inconsistency lies in the source
    document's own metadata, and silently discarding one would lose
    information that may still be useful; recording the observation lets an
    operator judge.

    Parameters
    ----------
    created_at:
        Resolved creation timestamp.
    modified_at:
        Resolved modification timestamp.
    warnings:
        Accumulator to which the observation is appended.

    Returns
    -------
    None
    """
    if created_at is None or modified_at is None:
        return
    if modified_at >= created_at:
        return

    warnings.append(
        ExtractionWarning(
            code=ExtractionWarningCode.OTHER,
            message=(
                "Declared modification timestamp precedes the declared "
                "creation timestamp; both have been retained"
            ),
            detail={
                "created_at": created_at.isoformat(),
                "modified_at": modified_at.isoformat(),
            },
        )
    )