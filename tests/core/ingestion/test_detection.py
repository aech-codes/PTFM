"""Unit tests for the format detection stage."""

from __future__ import annotations

import codecs
from typing import Optional

import pytest

from ptfm.core.ingestion.config import DetectionConfig
from ptfm.core.ingestion.exceptions import (
    EncodingDetectionError,
    FormatDetectionError,
)
from ptfm.core.ingestion.interfaces import FormatDetector
from ptfm.core.ingestion.models import DetectedFormat, DetectionSignalSource
from ptfm.core.ingestion.pipeline import detection as detection_module
from ptfm.core.ingestion.pipeline.detection import (
    DefaultFormatDetector,
    MagicSignature,
    _is_text_media_type,
)
from ptfm.core.schemas import ExtractionWarningCode

DOCX = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
XLSX = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"


def _detector(**config: object) -> DefaultFormatDetector:
    return DefaultFormatDetector(DetectionConfig(**config))  # type: ignore[arg-type]


class _StubSniffer:
    """Charset sniffer returning a fixed answer."""

    def __init__(self, encoding: Optional[str]) -> None:
        self.encoding = encoding
        self.calls = 0

    def sniff(self, sample: bytes) -> Optional[str]:
        self.calls += 1
        return self.encoding


# ---------------------------------------------------------------------------
# Protocol conformance and purity
# ---------------------------------------------------------------------------


def test_detector_satisfies_the_format_detector_protocol() -> None:
    assert isinstance(_detector(), FormatDetector)


def test_detection_is_deterministic() -> None:
    detector = _detector()
    header = b"%PDF-1.7\nbody"

    first = detector.detect(header, filename="a.pdf")
    second = detector.detect(header, filename="a.pdf")

    assert first == second


def test_detector_holds_no_assignable_state() -> None:
    with pytest.raises(AttributeError):
        _detector().cache = {}  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Magic byte recognition
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("header", "expected"),
    [
        (b"%PDF-1.7\n", "application/pdf"),
        (b"{\\rtf1\\ansi", "application/rtf"),
        (b"PK\x03\x04\x14\x00", "application/zip"),
        (b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1", "application/x-ole-storage"),
        (b"\x1f\x8b\x08\x00", "application/gzip"),
        (b"BZh91AY", "application/x-bzip2"),
        (b"\x89PNG\r\n\x1a\n", "image/png"),
        (b"GIF89a\x00", "image/gif"),
        (b"\xff\xd8\xff\xe0", "image/jpeg"),
        (b"II*\x00", "image/tiff"),
    ],
)
def test_byte_signatures_are_recognized(header: bytes, expected: str) -> None:
    assert _detector().detect(header).media_type == expected


@pytest.mark.parametrize(
    "header",
    [
        b"<!DOCTYPE html><html>",
        b"<!doctype HTML>",
        b"\n\n  <html lang='en'>",
        b"<HTML>",
    ],
)
def test_html_is_recognized_case_insensitively_and_after_whitespace(
    header: bytes,
) -> None:
    assert _detector().detect(header).media_type == "text/html"


def test_xml_declaration_is_recognized() -> None:
    assert _detector().detect(b"<?xml version='1.0'?>").media_type == "application/xml"


def test_magic_bytes_produce_an_authoritative_signal() -> None:
    detected = _detector().detect(b"%PDF-1.7")

    signal = detected.signal_for(DetectionSignalSource.MAGIC_BYTES)

    assert signal is not None
    assert signal.media_type == "application/pdf"


def test_signature_specificity_ordering_is_stable() -> None:
    specificities = [signature.specificity for signature in detection_module._SIGNATURES]

    assert specificities == sorted(specificities, reverse=True)


def test_signature_matching_tolerates_a_short_buffer() -> None:
    signature = MagicSignature("application/pdf", b"%PDF-")

    assert signature.matches(b"%PD") is False
    assert signature.matches(b"%PDF-") is True


# ---------------------------------------------------------------------------
# Refinement
# ---------------------------------------------------------------------------


def test_zip_magic_refines_to_docx_by_extension() -> None:
    """A .docx IS a ZIP; the extension is more specific, not contradictory."""
    detected = _detector().detect(b"PK\x03\x04\x14\x00", filename="report.docx")

    assert detected.media_type == DOCX
    assert detected.warnings == ()


def test_zip_magic_refines_by_declared_type_when_no_extension() -> None:
    detected = _detector().detect(b"PK\x03\x04", declared_media_type=XLSX)

    assert detected.media_type == XLSX
    assert detected.warnings == ()


def test_ole_magic_refines_to_legacy_office() -> None:
    detected = _detector().detect(
        b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1", filename="old.doc"
    )

    assert detected.media_type == "application/msword"
    assert detected.warnings == ()


@pytest.mark.parametrize(
    ("filename", "expected"),
    [
        ("notes.md", "text/markdown"),
        ("data.csv", "text/csv"),
        ("data.tsv", "text/tab-separated-values"),
        ("payload.json", "application/json"),
        ("stream.jsonl", "application/x-ndjson"),
    ],
)
def test_text_sniffing_refines_to_signature_free_formats(
    filename: str, expected: str
) -> None:
    """Markdown and CSV carry no byte signature; only the extension names them."""
    detected = _detector().detect(b"col_a,col_b\n1,2\n", filename=filename)

    assert detected.media_type == expected
    assert detected.warnings == ()


def test_xml_declaration_refines_to_xhtml() -> None:
    detected = _detector().detect(
        b"<?xml version='1.0'?><html/>", filename="page.xhtml"
    )

    assert detected.media_type == "application/xhtml+xml"
    assert detected.warnings == ()


def test_refinement_prefers_the_extension_over_the_declared_type() -> None:
    detected = _detector().detect(
        b"PK\x03\x04", filename="book.epub", declared_media_type=DOCX
    )

    assert detected.media_type == "application/epub+zip"


def test_unrelated_extension_does_not_refine_a_container() -> None:
    detected = _detector().detect(b"PK\x03\x04", filename="archive.pdf")

    assert detected.media_type == "application/zip"
    assert len(detected.warnings) == 1


# ---------------------------------------------------------------------------
# Disagreement
# ---------------------------------------------------------------------------


def test_magic_bytes_win_over_a_conflicting_extension() -> None:
    detected = _detector().detect(b"%PDF-1.7", filename="document.txt")

    assert detected.media_type == "application/pdf"
    assert len(detected.warnings) == 1
    warning = detected.warnings[0]
    assert warning.code is ExtractionWarningCode.MEDIA_TYPE_MISMATCH
    assert warning.detail["signal_media_type"] == "text/plain"
    assert warning.detail["selected_media_type"] == "application/pdf"


def test_magic_bytes_win_over_a_conflicting_declared_type() -> None:
    detected = _detector().detect(
        b"%PDF-1.7", declared_media_type="text/html"
    )

    assert detected.media_type == "application/pdf"
    assert [w.detail["signal_source"] for w in detected.warnings] == ["DECLARED"]


def test_both_weaker_signals_may_disagree() -> None:
    detected = _detector().detect(
        b"\x89PNG\r\n\x1a\n", filename="a.pdf", declared_media_type="text/csv"
    )

    assert detected.media_type == "image/png"
    assert len(detected.warnings) == 2


def test_agreeing_signals_produce_no_warning() -> None:
    detected = _detector().detect(
        b"%PDF-1.7", filename="a.pdf", declared_media_type="application/pdf"
    )

    assert detected.warnings == ()
    assert len(detected.signals) == 3


# ---------------------------------------------------------------------------
# Fallback ordering
# ---------------------------------------------------------------------------


def test_declared_type_resolves_unrecognized_binary() -> None:
    detected = _detector().detect(
        b"\x00\x01\x02\x03", declared_media_type="application/octet-stream"
    )

    assert detected.media_type == "application/octet-stream"


def test_extension_resolves_when_nothing_else_can() -> None:
    detected = _detector().detect(b"\x00\xff\x00\xff", filename="mystery.pptx")

    assert detected.media_type.endswith("presentationml.presentation")


def test_declared_type_outranks_the_extension() -> None:
    detected = _detector().detect(
        b"\x00\x01", filename="a.csv", declared_media_type="application/vnd.ms-excel"
    )

    assert detected.media_type == "application/vnd.ms-excel"


def test_declared_type_is_ignored_when_untrusted() -> None:
    detector = _detector(trust_declared_media_type=False)

    detected = detector.detect(
        b"\x00\x01", filename="a.pdf", declared_media_type="text/html"
    )

    assert detected.media_type == "application/pdf"
    assert detected.warnings == ()


def test_untrusted_declaration_is_still_recorded_as_a_signal() -> None:
    """Discarded signals remain inspectable after the fact."""
    detector = _detector(trust_declared_media_type=False)

    detected = detector.detect(b"%PDF-", declared_media_type="text/html")

    assert detected.signal_for(DetectionSignalSource.DECLARED) is not None


def test_malformed_declared_type_is_treated_as_absent() -> None:
    detected = _detector().detect(b"%PDF-1.7", declared_media_type="   ")

    assert detected.media_type == "application/pdf"
    assert detected.signal_for(DetectionSignalSource.DECLARED) is None


def test_declared_type_parameters_are_normalized() -> None:
    detected = _detector().detect(
        b"\x00\x01", declared_media_type="Text/HTML; charset=utf-8"
    )

    assert detected.media_type == "text/html"


@pytest.mark.parametrize(
    "filename", ["archive.tar.gz", "report.final.docx", "path/to/notes.md"]
)
def test_compound_and_path_filenames_resolve_by_final_suffix(filename: str) -> None:
    assert _detector().detect(b"\x00\x01", filename=filename).media_type is not None


def test_windows_style_paths_are_handled() -> None:
    detected = _detector().detect(b"\x00\x01", filename="C:\\docs\\report.pdf")

    assert detected.media_type == "application/pdf"


@pytest.mark.parametrize("filename", ["README", "archive.unknownext", ""])
def test_unrecognized_filenames_contribute_no_signal(filename: str) -> None:
    detected = _detector().detect(b"plain text body", filename=filename)

    assert detected.signal_for(DetectionSignalSource.EXTENSION) is None


# ---------------------------------------------------------------------------
# Detection failure
# ---------------------------------------------------------------------------


def test_unidentifiable_payload_raises() -> None:
    with pytest.raises(FormatDetectionError) as excinfo:
        _detector().detect(b"\x00\xff\xfe\x01\x02")

    assert excinfo.value.context["header_bytes_examined"] == 5


def test_empty_header_without_metadata_raises() -> None:
    with pytest.raises(FormatDetectionError):
        _detector().detect(b"")


def test_empty_header_resolves_from_the_filename() -> None:
    """A zero-byte payload is acquisition's concern, not detection's."""
    detected = _detector().detect(b"", filename="empty.txt")

    assert detected.media_type == "text/plain"


# ---------------------------------------------------------------------------
# Text sniffing
# ---------------------------------------------------------------------------


def test_utf8_payload_without_signature_is_detected_as_text() -> None:
    detected = _detector().detect("Plain prose with accents: café".encode("utf-8"))

    assert detected.media_type == "text/plain"
    assert detected.encoding == "utf-8"


def test_nul_byte_marks_a_payload_as_binary() -> None:
    with pytest.raises(FormatDetectionError):
        _detector().detect(b"text-like\x00but-binary")


def test_truncated_multibyte_tail_does_not_defeat_text_sniffing() -> None:
    """A header buffer is an arbitrary byte cut; a severed character is normal."""
    payload = ("prose " * 20 + "café").encode("utf-8")
    truncated = payload[:-1]

    assert _detector().detect(truncated).media_type == "text/plain"


@pytest.mark.parametrize("cut", [1, 2, 3])
def test_text_classification_is_independent_of_buffer_size(cut: int) -> None:
    payload = ("x" * 40 + "日本語").encode("utf-8")

    assert _detector().detect(payload[:-cut]).media_type == "text/plain"


# ---------------------------------------------------------------------------
# Encoding resolution
# ---------------------------------------------------------------------------


def test_binary_formats_carry_no_encoding() -> None:
    assert _detector().detect(b"%PDF-1.7").encoding is None


@pytest.mark.parametrize(
    ("bom", "expected"),
    [
        (codecs.BOM_UTF8, "utf-8-sig"),
        (codecs.BOM_UTF16_LE, "utf-16"),
        (codecs.BOM_UTF16_BE, "utf-16"),
        (codecs.BOM_UTF32_LE, "utf-32"),
    ],
)
def test_byte_order_marks_resolve_to_consuming_codecs(bom: bytes, expected: str) -> None:
    """A surviving U+FEFF would displace every block span by one character."""
    detected = _detector().detect(bom + b"content", filename="a.txt")

    assert detected.encoding == expected


def test_utf8_bom_codec_actually_strips_the_mark() -> None:
    payload = codecs.BOM_UTF8 + "Title".encode("utf-8")
    detected = _detector().detect(payload, filename="a.txt")

    assert detected.encoding is not None
    assert payload.decode(detected.encoding) == "Title"


def test_utf32_mark_is_not_shadowed_by_the_utf16_prefix() -> None:
    detected = _detector().detect(codecs.BOM_UTF32_LE + b"body", filename="a.txt")

    assert detected.encoding == "utf-32"


def test_valid_utf8_is_preferred_without_consulting_a_sniffer() -> None:
    sniffer = _StubSniffer("iso-8859-1")
    detector = DefaultFormatDetector(DetectionConfig(), sniffer=sniffer)

    detected = detector.detect("café".encode("utf-8"), filename="a.txt")

    assert detected.encoding == "utf-8"
    assert sniffer.calls == 0


def test_sniffer_resolves_a_legacy_encoding() -> None:
    sniffer = _StubSniffer("iso-8859-1")
    detector = DefaultFormatDetector(DetectionConfig(), sniffer=sniffer)

    detected = detector.detect("café".encode("iso-8859-1"), filename="a.txt")

    assert detected.encoding == "iso-8859-1"
    assert sniffer.calls == 1


def test_configured_fallback_applies_when_no_sniffer_is_injected() -> None:
    detector = _detector(default_text_encoding="iso-8859-1", max_replacement_ratio=1.0)

    detected = detector.detect("café".encode("iso-8859-1"), filename="a.txt")

    assert detected.encoding == "iso-8859-1"


def test_unusable_sniffer_result_falls_through_to_the_default() -> None:
    sniffer = _StubSniffer("utf-8")
    detector = DefaultFormatDetector(
        DetectionConfig(default_text_encoding="iso-8859-1", max_replacement_ratio=1.0),
        sniffer=sniffer,
    )

    detected = detector.detect("café".encode("iso-8859-1"), filename="a.txt")

    assert detected.encoding == "iso-8859-1"


def test_undecodable_text_raises_with_attempted_encodings() -> None:
    detector = _detector(default_text_encoding="ascii", max_replacement_ratio=0.0)

    with pytest.raises(EncodingDetectionError) as excinfo:
        detector.detect("héllo wörld ünïcode".encode("utf-16-le"), filename="a.txt")

    assert "utf-8" in excinfo.value.attempted_encodings
    assert "ascii" in excinfo.value.attempted_encodings


def test_a_permissive_replacement_ratio_accepts_lossy_decoding() -> None:
    strict = _detector(default_text_encoding="ascii", max_replacement_ratio=0.0)
    permissive = _detector(default_text_encoding="ascii", max_replacement_ratio=1.0)
    payload = "héllo wörld ünïcode".encode("utf-16-le")

    with pytest.raises(EncodingDetectionError):
        strict.detect(payload, filename="a.txt")

    assert permissive.detect(payload, filename="a.txt").encoding == "ascii"


def test_unknown_sniffer_codec_is_rejected_rather_than_propagated() -> None:
    sniffer = _StubSniffer("not-a-real-codec")
    detector = DefaultFormatDetector(DetectionConfig(), sniffer=sniffer)

    detected = detector.detect("café".encode("iso-8859-1"), filename="a.txt")

    assert detected.encoding == DetectionConfig().default_text_encoding


def test_empty_text_payload_uses_the_configured_default() -> None:
    detected = _detector().detect(b"", filename="empty.txt")

    assert detected.encoding == "utf-8"


# ---------------------------------------------------------------------------
# Cross-module consistency
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "media_type",
    [
        "text/plain",
        "text/html",
        "text/csv",
        "text/markdown",
        "application/json",
        "application/xml",
        "application/xhtml+xml",
        "application/pdf",
        "application/zip",
        "image/png",
        DOCX,
    ],
)
def test_text_classification_agrees_with_the_canonical_model(media_type: str) -> None:
    """
    The helper duplicates DetectedFormat.is_text because encoding must be
    resolved before the frozen model exists. This test binds them together.
    """
    canonical = DetectedFormat(media_type=media_type).is_text

    assert _is_text_media_type(media_type) is canonical


def test_every_refinement_target_is_a_known_extension_type() -> None:
    """A refinement no extension can produce would be unreachable."""
    known = set(detection_module._EXTENSION_MEDIA_TYPES.values())
    for general, specifics in detection_module._REFINEMENTS.items():
        unreachable = {s for s in specifics if s not in known}
        assert unreachable <= {"application/rss+xml"}, general


# ---------------------------------------------------------------------------
# Anti-pattern guards
# ---------------------------------------------------------------------------


def test_detection_performs_no_io() -> None:
    import inspect

    source = inspect.getsource(detection_module)
    for forbidden in ("open(", "Path(", "requests", "httpx", "socket", "async def"):
        assert forbidden not in source


def test_module_declares_no_mutable_state() -> None:
    mutable = [
        name
        for name, value in vars(detection_module).items()
        if not name.startswith("__") and isinstance(value, (list, dict, set))
    ]
    assert mutable == []


def test_detection_declares_no_hardcoded_limits() -> None:
    """Every operational limit originates from injected configuration."""
    import inspect

    source = inspect.getsource(DefaultFormatDetector)
    assert "self._config.max_replacement_ratio" in source
    assert "self._config.encoding_sample_bytes" in source
    assert "0.5" not in source and "0.8" not in source


def test_detection_imports_no_third_party_parsers() -> None:
    import inspect

    source = inspect.getsource(detection_module)
    for forbidden in ("import magic", "import puremagic", "charset_normalizer", "chardet"):
        assert forbidden not in source