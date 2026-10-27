"""Unit tests for the metadata harvesting stage."""

from __future__ import annotations

import inspect
from datetime import UTC, datetime, timedelta, timezone
from typing import Optional

import pytest

from ptfm.core.ingestion.config import MetadataConfig
from ptfm.core.ingestion.interfaces import LanguageDetection
from ptfm.core.ingestion.models import ExtractedMetadata
from ptfm.core.ingestion.pipeline import metadata as metadata_module
from ptfm.core.ingestion.pipeline.metadata import (
    HarvestedMetadata,
    MetadataHarvester,
)
from ptfm.core.schemas import DocumentMetadata, ExtractionWarningCode

NOW = datetime(2026, 7, 30, 12, 0, tzinfo=UTC)


# ---------------------------------------------------------------------------
# Doubles
# ---------------------------------------------------------------------------


class _FixedClock:
    def now(self) -> datetime:
        return NOW


class _Detector:
    """Language detector returning a fixed answer, or raising."""

    def __init__(
        self,
        result: Optional[LanguageDetection] = None,
        error: Optional[Exception] = None,
    ) -> None:
        self.result = result
        self.error = error
        self.samples: list[str] = []

    def detect(self, text: str) -> Optional[LanguageDetection]:
        self.samples.append(text)
        if self.error is not None:
            raise self.error
        return self.result


def _harvester(
    detector: Optional[_Detector] = None,
    **config: object,
) -> MetadataHarvester:
    return MetadataHarvester(
        MetadataConfig(**config),  # type: ignore[arg-type]
        _FixedClock(),
        detector,
    )


def _codes(result: HarvestedMetadata) -> list[ExtractionWarningCode]:
    return [warning.code for warning in result.warnings]


# ---------------------------------------------------------------------------
# Canonical invariant
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text", ["", "hello", "a longer body of text\n\nwith paragraphs", "café"]
)
def test_char_count_always_matches_the_text(text: str) -> None:
    """RawDocument requires exact agreement; deriving it here guarantees it."""
    result = _harvester().harvest(ExtractedMetadata(), text)

    assert result.metadata.char_count == len(text)


def test_extractor_reported_length_cannot_corrupt_the_count() -> None:
    """The extractor has no field for this, and none is trusted if it had."""
    result = _harvester().harvest(
        ExtractedMetadata(properties={"reported_chars": "99999"}), "short"
    )

    assert result.metadata.char_count == 5


def test_harvested_metadata_constructs_a_valid_canonical_record() -> None:
    result = _harvester().harvest(ExtractedMetadata(title="T"), "body text")

    assert isinstance(result.metadata, DocumentMetadata)
    assert result.metadata.word_count == 2


# ---------------------------------------------------------------------------
# Passthrough
# ---------------------------------------------------------------------------


def test_format_native_fields_pass_through() -> None:
    extracted = ExtractedMetadata(
        title="Quarterly Report",
        authors=("Ada Lovelace", "Alan Turing"),
        declared_language="en-GB",
        page_count=12,
        producer="LibreOffice 7.6",
        properties={"pdf_version": "1.7"},
    )

    result = _harvester().harvest(extracted, "body")

    metadata = result.metadata
    assert metadata.title == "Quarterly Report"
    assert metadata.authors == ("Ada Lovelace", "Alan Turing")
    assert metadata.declared_language == "en-GB"
    assert metadata.page_count == 12
    assert metadata.producer == "LibreOffice 7.6"
    assert metadata.format_properties == {"pdf_version": "1.7"}
    assert result.warnings == ()


def test_absent_metadata_yields_a_minimal_record() -> None:
    result = _harvester().harvest(ExtractedMetadata(), "")

    metadata = result.metadata
    assert metadata.title is None
    assert metadata.authors == ()
    assert metadata.created_at is None
    assert metadata.page_count is None
    assert metadata.char_count == 0
    assert metadata.word_count == 0


def test_zero_page_count_is_preserved_rather_than_dropped() -> None:
    result = _harvester().harvest(ExtractedMetadata(page_count=0), "x")

    assert result.metadata.page_count == 0


# ---------------------------------------------------------------------------
# String cleaning
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("  Padded Title  ", "Padded Title"),
        ("Multi\nline\ntitle", "Multi line title"),
        ("Too    many     spaces", "Too many spaces"),
        ("Tabs\tand\tnewlines\n", "Tabs and newlines"),
    ],
)
def test_metadata_strings_are_cleaned(raw: str, expected: str) -> None:
    result = _harvester().harvest(ExtractedMetadata(title=raw), "body")

    assert result.metadata.title == expected


@pytest.mark.parametrize("raw", ["", "   ", "\n\t\n"])
def test_blank_strings_become_absent(raw: str) -> None:
    result = _harvester().harvest(
        ExtractedMetadata(title=raw, producer=raw, declared_language=raw), "body"
    )

    assert result.metadata.title is None
    assert result.metadata.producer is None
    assert result.metadata.declared_language is None


def test_overlong_title_is_truncated_and_reported() -> None:
    harvester = _harvester(max_title_chars=20)

    result = harvester.harvest(ExtractedMetadata(title="x" * 100), "body")

    assert result.metadata.title is not None
    assert len(result.metadata.title) == 20
    assert _codes(result) == [ExtractionWarningCode.OTHER]
    assert result.warnings[0].detail["original_length"] == 100


def test_title_at_the_limit_is_not_reported() -> None:
    result = _harvester(max_title_chars=10).harvest(
        ExtractedMetadata(title="x" * 10), "body"
    )

    assert result.warnings == ()


# ---------------------------------------------------------------------------
# Authors
# ---------------------------------------------------------------------------


def test_author_order_is_preserved() -> None:
    result = _harvester().harvest(
        ExtractedMetadata(authors=("Zoe", "Adam", "Mia")), "body"
    )

    assert result.metadata.authors == ("Zoe", "Adam", "Mia")


def test_duplicate_and_blank_authors_are_removed() -> None:
    result = _harvester().harvest(
        ExtractedMetadata(authors=("Ada", "  ", "Ada", "  Ada  ", "Alan")), "body"
    )

    assert result.metadata.authors == ("Ada", "Alan")


def test_author_list_is_capped_and_reported() -> None:
    harvester = _harvester(max_authors=2)

    result = harvester.harvest(
        ExtractedMetadata(authors=("A", "B", "C", "D")), "body"
    )

    assert result.metadata.authors == ("A", "B")
    assert result.warnings[0].detail["original_count"] == 4


# ---------------------------------------------------------------------------
# Timestamps
# ---------------------------------------------------------------------------


def test_naive_timestamp_is_interpreted_as_utc_without_warning() -> None:
    """Warning here would fire on nearly every Office document."""
    naive = datetime(2025, 3, 1, 9, 30)

    result = _harvester().harvest(ExtractedMetadata(created_at=naive), "body")

    created = result.metadata.created_at
    assert created is not None
    assert created.tzinfo is not None
    assert created == datetime(2025, 3, 1, 9, 30, tzinfo=UTC)
    assert result.warnings == ()


def test_aware_timestamp_is_converted_to_utc() -> None:
    offset = timezone(timedelta(hours=5, minutes=30))
    aware = datetime(2025, 3, 1, 15, 0, tzinfo=offset)

    result = _harvester().harvest(ExtractedMetadata(modified_at=aware), "body")

    modified = result.metadata.modified_at
    assert modified is not None
    assert modified.utcoffset() == timedelta(0)
    assert modified == datetime(2025, 3, 1, 9, 30, tzinfo=UTC)


def test_all_resolved_timestamps_are_timezone_aware() -> None:
    result = _harvester().harvest(
        ExtractedMetadata(
            created_at=datetime(2024, 1, 1), modified_at=datetime(2024, 6, 1)
        ),
        "body",
    )

    assert result.metadata.created_at is not None
    assert result.metadata.modified_at is not None
    assert result.metadata.created_at.tzinfo is not None
    assert result.metadata.modified_at.tzinfo is not None


def test_future_timestamp_is_discarded_and_reported() -> None:
    future = NOW + timedelta(days=365)

    result = _harvester().harvest(ExtractedMetadata(created_at=future), "body")

    assert result.metadata.created_at is None
    assert _codes(result) == [ExtractionWarningCode.METADATA_UNAVAILABLE]
    assert result.warnings[0].detail["field"] == "created_at"


def test_timestamp_at_the_current_instant_is_accepted() -> None:
    result = _harvester().harvest(ExtractedMetadata(created_at=NOW), "body")

    assert result.metadata.created_at == NOW


def test_old_timestamps_are_retained() -> None:
    """No minimum date is enforced; a hardcoded year could discard real data."""
    old = datetime(1887, 5, 3, tzinfo=UTC)

    result = _harvester().harvest(ExtractedMetadata(created_at=old), "body")

    assert result.metadata.created_at == old
    assert result.warnings == ()


def test_both_timestamps_are_checked_independently() -> None:
    result = _harvester().harvest(
        ExtractedMetadata(
            created_at=NOW + timedelta(days=1),
            modified_at=NOW + timedelta(days=2),
        ),
        "body",
    )

    assert result.metadata.created_at is None
    assert result.metadata.modified_at is None
    assert len(result.warnings) == 2


def test_inverted_timestamp_ordering_is_reported_and_both_retained() -> None:
    created = datetime(2025, 6, 1, tzinfo=UTC)
    modified = datetime(2025, 1, 1, tzinfo=UTC)

    result = _harvester().harvest(
        ExtractedMetadata(created_at=created, modified_at=modified), "body"
    )

    assert result.metadata.created_at == created
    assert result.metadata.modified_at == modified
    assert _codes(result) == [ExtractionWarningCode.OTHER]


def test_equal_timestamps_are_not_reported() -> None:
    instant = datetime(2025, 1, 1, tzinfo=UTC)

    result = _harvester().harvest(
        ExtractedMetadata(created_at=instant, modified_at=instant), "body"
    )

    assert result.warnings == ()


def test_ordering_check_requires_both_timestamps() -> None:
    result = _harvester().harvest(
        ExtractedMetadata(created_at=datetime(2025, 1, 1, tzinfo=UTC)), "body"
    )

    assert result.warnings == ()


# ---------------------------------------------------------------------------
# Language detection
# ---------------------------------------------------------------------------


def test_detected_language_is_recorded() -> None:
    detector = _Detector(LanguageDetection(language="en", confidence=0.98))

    result = _harvester(detector).harvest(ExtractedMetadata(), "x" * 200)

    assert result.metadata.detected_language == "en"
    assert result.warnings == ()


def test_language_tag_is_lowercased() -> None:
    detector = _Detector(LanguageDetection(language="EN", confidence=0.9))

    result = _harvester(detector).harvest(ExtractedMetadata(), "x" * 200)

    assert result.metadata.detected_language == "en"


def test_detection_uses_a_bounded_sample() -> None:
    detector = _Detector(LanguageDetection(language="en", confidence=0.9))
    harvester = _harvester(detector, language_detection_sample_chars=50)

    harvester.harvest(ExtractedMetadata(), "x" * 5000)

    assert len(detector.samples[0]) == 50


def test_short_text_skips_detection_without_warning() -> None:
    detector = _Detector(LanguageDetection(language="en", confidence=0.9))
    harvester = _harvester(detector, language_detection_min_chars=100)

    result = harvester.harvest(ExtractedMetadata(), "brief")

    assert result.metadata.detected_language is None
    assert detector.samples == []
    assert result.warnings == ()


def test_detection_can_be_disabled() -> None:
    detector = _Detector(LanguageDetection(language="en", confidence=0.9))
    harvester = _harvester(detector, detect_language=False)

    result = harvester.harvest(ExtractedMetadata(), "x" * 200)

    assert result.metadata.detected_language is None
    assert detector.samples == []


def test_absent_detector_skips_detection() -> None:
    result = _harvester(None).harvest(ExtractedMetadata(), "x" * 200)

    assert result.metadata.detected_language is None
    assert result.warnings == ()


def test_indeterminate_detection_is_reported() -> None:
    result = _harvester(_Detector(None)).harvest(ExtractedMetadata(), "x" * 200)

    assert result.metadata.detected_language is None
    assert _codes(result) == [ExtractionWarningCode.LANGUAGE_DETECTION_UNCERTAIN]


def test_detector_failure_degrades_without_failing_the_document() -> None:
    """Losing advisory metadata must not quarantine a well-extracted document."""
    detector = _Detector(error=RuntimeError("model unavailable"))

    result = _harvester(detector).harvest(ExtractedMetadata(title="T"), "x" * 200)

    assert result.metadata.detected_language is None
    assert result.metadata.title == "T"
    assert result.warnings[0].detail["exception_type"] == "RuntimeError"


def test_blank_language_tag_is_treated_as_absent() -> None:
    detector = _Detector(LanguageDetection(language="   ", confidence=0.9))

    result = _harvester(detector).harvest(ExtractedMetadata(), "x" * 200)

    assert result.metadata.detected_language is None


# ---------------------------------------------------------------------------
# Confidence prohibition
# ---------------------------------------------------------------------------


def test_low_confidence_language_is_still_recorded() -> None:
    """
    The threshold decides whether to warn, never whether to keep. Filtering
    on a model score would make it a content gate.
    """
    detector = _Detector(LanguageDetection(language="cy", confidence=0.11))
    harvester = _harvester(detector, language_detection_min_confidence=0.9)

    result = harvester.harvest(ExtractedMetadata(), "x" * 200)

    assert result.metadata.detected_language == "cy"
    assert _codes(result) == [ExtractionWarningCode.LANGUAGE_DETECTION_UNCERTAIN]


def test_confidence_appears_only_as_diagnostic_detail() -> None:
    detector = _Detector(LanguageDetection(language="en", confidence=0.2))
    harvester = _harvester(detector, language_detection_min_confidence=0.9)

    result = harvester.harvest(ExtractedMetadata(), "x" * 200)

    assert result.warnings[0].detail["detector_confidence"] == 0.2
    assert not hasattr(result.metadata, "detector_confidence")


def test_confidence_at_the_threshold_produces_no_warning() -> None:
    detector = _Detector(LanguageDetection(language="en", confidence=0.5))
    harvester = _harvester(detector, language_detection_min_confidence=0.5)

    result = harvester.harvest(ExtractedMetadata(), "x" * 200)

    assert result.metadata.detected_language == "en"
    assert result.warnings == ()


@pytest.mark.parametrize("confidence", [0.0, 0.01, 0.49, 0.5, 0.99, 1.0])
def test_language_survives_at_every_confidence(confidence: float) -> None:
    detector = _Detector(LanguageDetection(language="de", confidence=confidence))

    result = _harvester(detector).harvest(ExtractedMetadata(), "x" * 200)

    assert result.metadata.detected_language == "de"


def test_canonical_metadata_exposes_no_confidence_field() -> None:
    forbidden = {
        "confidence",
        "score",
        "probability",
        "detector_confidence",
        "language_confidence",
    }

    assert not forbidden & set(DocumentMetadata.model_fields)


# ---------------------------------------------------------------------------
# Word count
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("", 0),
        ("   ", 0),
        ("one", 1),
        ("one two three", 3),
        ("spread\nacross\n\nlines", 3),
        ("multiple    spaces", 2),
    ],
)
def test_word_count_is_whitespace_delimited(text: str, expected: int) -> None:
    assert _harvester().harvest(ExtractedMetadata(), text).metadata.word_count == expected


def test_word_count_is_documented_as_advisory_for_unspaced_scripts() -> None:
    result = _harvester().harvest(ExtractedMetadata(), "日本語のテキストです")

    assert result.metadata.word_count == 1
    assert "Advisory" in (metadata_module._count_words.__doc__ or "")


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_harvesting_is_deterministic() -> None:
    detector = _Detector(LanguageDetection(language="en", confidence=0.9))
    harvester = _harvester(detector)
    extracted = ExtractedMetadata(title="T", created_at=datetime(2025, 1, 1))

    first = harvester.harvest(extracted, "x" * 200)
    second = harvester.harvest(extracted, "x" * 200)

    assert first == second


def test_multiple_observations_are_accumulated_in_order() -> None:
    detector = _Detector(None)
    harvester = _harvester(
        detector, max_title_chars=5, max_authors=1
    )

    result = harvester.harvest(
        ExtractedMetadata(
            title="x" * 50,
            authors=("A", "B"),
            created_at=NOW + timedelta(days=1),
        ),
        "x" * 200,
    )

    assert _codes(result) == [
        ExtractionWarningCode.OTHER,
        ExtractionWarningCode.OTHER,
        ExtractionWarningCode.METADATA_UNAVAILABLE,
        ExtractionWarningCode.LANGUAGE_DETECTION_UNCERTAIN,
    ]


# ---------------------------------------------------------------------------
# Anti-pattern guards
# ---------------------------------------------------------------------------


def test_stage_contains_no_domain_vocabulary() -> None:
    source = inspect.getsource(metadata_module).lower()
    for term in (
        "jurisdiction",
        "statute",
        "ticker",
        "icd",
        "diagnosis",
        "court",
        "patient",
    ):
        assert term not in source


def test_stage_declares_no_hardcoded_limits() -> None:
    source = inspect.getsource(MetadataHarvester)

    assert "self._config.max_title_chars" in source
    assert "self._config.max_authors" in source
    assert "self._config.language_detection_min_confidence" in source
    assert "self._config.language_detection_sample_chars" in source
    assert "1900" not in source and "> 0.8" not in source


def test_stage_is_synchronous_and_performs_no_io() -> None:
    source = inspect.getsource(metadata_module)
    for forbidden in ("async def", "await ", "open(", "requests", "httpx"):
        assert forbidden not in source


def test_stage_never_reads_the_wall_clock_directly() -> None:
    """Timestamp handling must be deterministic under test."""
    source = inspect.getsource(metadata_module)

    assert "self._clock.now()" in source
    assert "datetime.now(" not in source
    assert "utcnow" not in source


def test_stage_never_catches_base_exception() -> None:
    source = inspect.getsource(metadata_module)

    assert "except BaseException" not in source
    assert "except:" not in source


def test_harvester_holds_no_assignable_state() -> None:
    with pytest.raises(AttributeError):
        _harvester().cache = {}  # type: ignore[attr-defined]


def test_module_declares_no_mutable_state() -> None:
    mutable = [
        name
        for name, value in vars(metadata_module).items()
        if not name.startswith("__") and isinstance(value, (list, dict, set))
    ]
    assert mutable == []


def test_stage_imports_no_concrete_implementations() -> None:
    source = inspect.getsource(metadata_module)
    for forbidden in (
        "from ptfm.core.ingestion.extractors",
        "from ptfm.core.ingestion.connectors",
        "from ptfm.plugins",
        "lingua",
        "langdetect",
    ):
        assert forbidden not in source