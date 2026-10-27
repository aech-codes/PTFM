"""
PTFM Format Detection Stage

File:
    ptfm/core/ingestion/pipeline/detection.py

Description:
    Resolves the media type and character encoding of a payload from its
    leading bytes, the filename, and any type declared by the source.

Purity contract
---------------
    Detection is a pure function over a header buffer. It performs no I/O,
    holds no state, and never touches the payload beyond the bytes handed
    to it. The acquisition stage reads the prefix; this stage interprets
    it. That separation makes detection deterministic, exhaustively
    testable, and structurally incapable of becoming an unexpected source
    of blocking I/O.

Signal authority
----------------
    Magic bytes are authoritative: they describe the payload itself.
    A source-declared type and a filename extension are assertions that
    may be absent, stale, or wrong.

Refinement versus disagreement
------------------------------
    Container formats make naive precedence wrong. A .docx file is a ZIP
    archive; its magic bytes can only say ``application/zip`` because the
    ZIP central directory naming the parts lives at the END of the file,
    which a header buffer does not contain.

    Detection therefore distinguishes two relationships between signals:

        Refinement
            The weaker signal names a subtype of what the payload
            inspection found, as ``.docx`` refines ``application/zip`` or
            ``.md`` refines ``text/plain``. The specific type wins and no
            warning is recorded. This is correct behaviour, not conflict.

        Disagreement
            The weaker signal names something unrelated, as a ``.txt``
            extension on a payload whose magic bytes say PDF. The
            authoritative signal wins and a MEDIA_TYPE_MISMATCH warning is
            recorded for operator inspection.

    Without this distinction every Office document ingested would emit a
    spurious mismatch warning.

Dependency contract
-------------------
    Imports only the standard library, ``ptfm.core.schemas`` and the
    ingestion ``exceptions``, ``config``, ``models`` and ``interfaces``
    modules. Charset detection beyond the standard library is injected
    rather than imported, so this stage binds the Core to no third-party
    parsing library.
"""

from __future__ import annotations

import codecs
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Final, Mapping, Optional, Protocol, Sequence, runtime_checkable

from ptfm.core.ingestion.config import DetectionConfig
from ptfm.core.ingestion.exceptions import (
    EncodingDetectionError,
    FormatDetectionError,
)
from ptfm.core.ingestion.interfaces import normalize_media_type
from ptfm.core.ingestion.models import (
    DetectedFormat,
    DetectionSignal,
    DetectionSignalSource,
)
from ptfm.core.schemas import ExtractionWarning, ExtractionWarningCode

__all__ = [
    "MagicSignature",
    "CharsetSniffer",
    "DefaultFormatDetector",
]

#: Replacement character produced by a lossy decode.
_REPLACEMENT_CHAR: Final[str] = "\ufffd"

#: Longest byte-order mark considered, bounding the BOM probe.
_MAX_BOM_LENGTH: Final[int] = 4

#: Longest UTF-8 sequence, bounding the truncated-tail trim.
_MAX_UTF8_SEQUENCE: Final[int] = 4

#: Leading whitespace tolerated before a markup signature.
_MARKUP_LEAD_BYTES: Final[bytes] = b" \t\r\n\f\v"


# ============================================================================
# SIGNATURE TABLE
# ============================================================================


@dataclass(frozen=True, slots=True)
class MagicSignature:
    """
    A byte pattern identifying a media type.

    Attributes
    ----------
    media_type:
        Media type indicated by a match.
    pattern:
        Byte sequence to match.
    offset:
        Position at which the pattern begins.
    case_insensitive:
        Whether matching ignores ASCII case. Required for markup
        signatures, which appear in any case in real documents.
    skip_leading_whitespace:
        Whether leading whitespace is skipped before matching. Required
        for markup, which is frequently preceded by blank lines.
    """

    media_type: str
    pattern: bytes
    offset: int = 0
    case_insensitive: bool = False
    skip_leading_whitespace: bool = False

    @property
    def specificity(self) -> int:
        """
        Ranking weight for resolving overlapping signatures.

        Longer patterns reaching further into the payload are more
        specific, so a four-byte container signature never shadows a
        longer, more precise one at the same offset.

        Returns
        -------
        int
            Offset plus pattern length.
        """
        return self.offset + len(self.pattern)

    def matches(self, header: bytes) -> bool:
        """
        Report whether this signature matches a header buffer.

        Parameters
        ----------
        header:
            Leading bytes of the payload.

        Returns
        -------
        bool
            ``True`` when the pattern is present at the expected position.
        """
        candidate = header
        if self.skip_leading_whitespace:
            candidate = candidate.lstrip(_MARKUP_LEAD_BYTES)

        start = self.offset
        end = start + len(self.pattern)
        if len(candidate) < end:
            return False

        window = candidate[start:end]
        if self.case_insensitive:
            return window.lower() == self.pattern.lower()
        return window == self.pattern


#: Byte signatures recognized by the detector, ordered most specific first
#: at module load so that resolution is deterministic and order-independent
#: at call time.
_SIGNATURES: Final[tuple[MagicSignature, ...]] = tuple(
    sorted(
        (
            # Documents
            MagicSignature("application/pdf", b"%PDF-"),
            MagicSignature("application/rtf", b"{\\rtf"),
            # Container formats. These are intentionally generic: the parts
            # that would identify the specific format are not present in a
            # header buffer. Refinement resolves them.
            MagicSignature("application/zip", b"PK\x03\x04"),
            MagicSignature("application/zip", b"PK\x05\x06"),
            MagicSignature("application/zip", b"PK\x07\x08"),
            MagicSignature(
                "application/x-ole-storage", b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"
            ),
            # Markup. Case-insensitive and whitespace-tolerant because real
            # documents are not written to a canonical form.
            MagicSignature(
                "text/html",
                b"<!doctype html",
                case_insensitive=True,
                skip_leading_whitespace=True,
            ),
            MagicSignature(
                "text/html",
                b"<html",
                case_insensitive=True,
                skip_leading_whitespace=True,
            ),
            MagicSignature(
                "application/xml", b"<?xml", skip_leading_whitespace=True
            ),
            # Compression
            MagicSignature("application/gzip", b"\x1f\x8b"),
            MagicSignature("application/x-bzip2", b"BZh"),
            # Images
            MagicSignature("image/png", b"\x89PNG\r\n\x1a\n"),
            MagicSignature("image/gif", b"GIF89a"),
            MagicSignature("image/gif", b"GIF87a"),
            MagicSignature("image/jpeg", b"\xff\xd8\xff"),
            MagicSignature("image/tiff", b"II*\x00"),
            MagicSignature("image/tiff", b"MM\x00*"),
        ),
        key=lambda signature: (-signature.specificity, signature.media_type),
    )
)

#: Filename extension to media type. The weakest signal, and the only one
#: capable of naming formats that carry no distinguishing byte signature.
_EXTENSION_MEDIA_TYPES: Final[Mapping[str, str]] = {
    # Plain and lightweight markup
    "txt": "text/plain",
    "text": "text/plain",
    "log": "text/plain",
    "md": "text/markdown",
    "markdown": "text/markdown",
    "rst": "text/x-rst",
    # Tabular
    "csv": "text/csv",
    "tsv": "text/tab-separated-values",
    # Structured
    "json": "application/json",
    "jsonl": "application/x-ndjson",
    "ndjson": "application/x-ndjson",
    "xml": "application/xml",
    "yaml": "application/yaml",
    "yml": "application/yaml",
    # Web
    "html": "text/html",
    "htm": "text/html",
    "xhtml": "application/xhtml+xml",
    # Office, OOXML
    "docx": (
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    ),
    "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "pptx": (
        "application/vnd.openxmlformats-officedocument.presentationml.presentation"
    ),
    "dotx": (
        "application/vnd.openxmlformats-officedocument.wordprocessingml.template"
    ),
    "potx": (
        "application/vnd.openxmlformats-officedocument.presentationml.template"
    ),
    # Office, legacy binary
    "doc": "application/msword",
    "xls": "application/vnd.ms-excel",
    "ppt": "application/vnd.ms-powerpoint",
    # OpenDocument
    "odt": "application/vnd.oasis.opendocument.text",
    "ods": "application/vnd.oasis.opendocument.spreadsheet",
    "odp": "application/vnd.oasis.opendocument.presentation",
    # Other documents
    "pdf": "application/pdf",
    "rtf": "application/rtf",
    "epub": "application/epub+zip",
    "eml": "message/rfc822",
    # Archives
    "zip": "application/zip",
    "gz": "application/gzip",
    "bz2": "application/x-bzip2",
}

#: Media types a payload-inspection result may be refined INTO by a weaker
#: signal. A refinement is a more specific description of the same payload,
#: not a contradiction, so it is applied silently.
_REFINEMENTS: Final[Mapping[str, frozenset[str]]] = {
    "application/zip": frozenset(
        {
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.template",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            "application/vnd.openxmlformats-officedocument.presentationml.presentation",
            "application/vnd.openxmlformats-officedocument.presentationml.template",
            "application/vnd.oasis.opendocument.text",
            "application/vnd.oasis.opendocument.spreadsheet",
            "application/vnd.oasis.opendocument.presentation",
            "application/epub+zip",
        }
    ),
    "application/x-ole-storage": frozenset(
        {
            "application/msword",
            "application/vnd.ms-excel",
            "application/vnd.ms-powerpoint",
        }
    ),
    # Content sniffing establishes only that a payload is text. Markdown,
    # CSV, TSV and NDJSON carry no distinguishing byte signature and can be
    # named only by extension or declaration.
    "text/plain": frozenset(
        {
            "text/markdown",
            "text/x-rst",
            "text/csv",
            "text/tab-separated-values",
            "application/json",
            "application/x-ndjson",
            "application/yaml",
            "message/rfc822",
        }
    ),
    # An XML declaration is compatible with any XML-based document type.
    "application/xml": frozenset(
        {"application/xhtml+xml", "text/html", "application/rss+xml"}
    ),
}

#: Byte-order marks mapped to the codec that consumes them. Resolving to a
#: BOM-stripping codec matters: a surviving U+FEFF at character zero would
#: shift every block span in the document by one.
_BYTE_ORDER_MARKS: Final[tuple[tuple[bytes, str], ...]] = (
    (codecs.BOM_UTF32_LE, "utf-32"),
    (codecs.BOM_UTF32_BE, "utf-32"),
    (codecs.BOM_UTF8, "utf-8-sig"),
    (codecs.BOM_UTF16_LE, "utf-16"),
    (codecs.BOM_UTF16_BE, "utf-16"),
)

#: Media types requiring a character encoding. Must agree with
#: ``DetectedFormat.is_text``; a test binds the two together.
_TEXT_MEDIA_TYPES: Final[frozenset[str]] = frozenset(
    {"application/json", "application/xml", "application/xhtml+xml"}
)


def _is_text_media_type(media_type: str) -> bool:
    """
    Report whether a media type requires character encoding resolution.

    Duplicates the logic of ``DetectedFormat.is_text`` because encoding
    must be resolved before that frozen model can be constructed. The two
    are bound together by a test that compares them across every relevant
    media type, so divergence fails immediately rather than silently.

    Parameters
    ----------
    media_type:
        Normalized media type.

    Returns
    -------
    bool
        ``True`` when the payload is textual.
    """
    return media_type.startswith("text/") or media_type in _TEXT_MEDIA_TYPES


# ============================================================================
# OPTIONAL CHARSET DETECTION
# ============================================================================


@runtime_checkable
class CharsetSniffer(Protocol):
    """
    Optional statistical character encoding detector.

    Byte-order marks and strict UTF-8 validation are exact and require no
    dependency, so they are handled by the detector itself. A sniffer adds
    value only for legacy non-UTF-8 encodings.

    Declared as an injected Protocol rather than an import so that this
    stage binds the Core to no third-party library and its tests exercise
    detector logic rather than library behaviour. A thin adapter over
    ``charset-normalizer`` satisfies it.
    """

    def sniff(self, sample: bytes) -> Optional[str]:
        """
        Infer the character encoding of a byte sample.

        Parameters
        ----------
        sample:
            Leading bytes of a text payload.

        Returns
        -------
        Optional[str]
            A Python codec name, or ``None`` when no determination is
            possible.
        """
        ...


# ============================================================================
# DETECTOR
# ============================================================================


class DefaultFormatDetector:
    """
    Resolves media type and encoding from payload bytes and metadata.

    Satisfies the ``FormatDetector`` protocol. Stateless and pure: the same
    inputs always produce the same result, and no instance state is
    mutated, so a single instance serves all concurrent workers.

    Parameters
    ----------
    config:
        Detection configuration governing declared-type trust, sample
        sizes, the acceptable proportion of replacement characters, and
        the fallback encoding.
    sniffer:
        Optional statistical charset detector consulted only after
        byte-order mark inspection and strict UTF-8 validation have both
        failed to determine an encoding.
    """

    __slots__ = ("_config", "_sniffer")

    def __init__(
        self,
        config: DetectionConfig,
        *,
        sniffer: Optional[CharsetSniffer] = None,
    ) -> None:
        self._config = config
        self._sniffer = sniffer

    # ------------------------------------------------------------------
    # Public contract
    # ------------------------------------------------------------------

    def detect(
        self,
        header: bytes,
        *,
        filename: str | None = None,
        declared_media_type: str | None = None,
    ) -> DetectedFormat:
        """
        Resolve the media type and encoding of a payload.

        Signals are gathered from payload inspection, the source-declared
        type, and the filename extension, then reconciled by authority
        with refinement applied where a weaker signal names a subtype of
        what inspection found.

        Parameters
        ----------
        header:
            Leading bytes of the payload.
        filename:
            Originating filename, when the transport exposes one.
        declared_media_type:
            Media type asserted by the source. Advisory; honoured only
            when ``trust_declared_media_type`` is enabled.

        Returns
        -------
        DetectedFormat
            Resolved media type, encoding, every contributing signal, and
            any disagreement warnings.

        Raises
        ------
        FormatDetectionError
            If no signal yields a usable media type.
        EncodingDetectionError
            If a text payload cannot be decoded within the configured
            replacement ratio.
        """
        signals = self._gather_signals(
            header, filename=filename, declared_media_type=declared_media_type
        )

        if not signals:
            raise FormatDetectionError(
                "Unable to determine media type: the payload matched no known "
                "signature, is not decodable as text, and neither a declared "
                "media type nor a recognized filename extension was supplied.",
                context={
                    "header_bytes_examined": len(header),
                    "filename": filename,
                    "declared_media_type": declared_media_type,
                },
            )

        media_type, warnings = self._reconcile(signals)
        encoding = self._resolve_encoding(header, media_type)

        return DetectedFormat(
            media_type=media_type,
            encoding=encoding,
            signals=signals,
            warnings=warnings,
        )

    # ------------------------------------------------------------------
    # Signal gathering
    # ------------------------------------------------------------------

    def _gather_signals(
        self,
        header: bytes,
        *,
        filename: str | None,
        declared_media_type: str | None,
    ) -> tuple[DetectionSignal, ...]:
        """
        Collect every available detection signal.

        Parameters
        ----------
        header:
            Leading bytes of the payload.
        filename:
            Originating filename, when available.
        declared_media_type:
            Media type asserted by the source.

        Returns
        -------
        tuple[DetectionSignal, ...]
            Signals in descending order of authority. Sources yielding no
            determination are omitted entirely.
        """
        signals: list[DetectionSignal] = []

        payload_type = self._inspect_payload(header)
        if payload_type is not None:
            signals.append(
                DetectionSignal(
                    source=DetectionSignalSource.MAGIC_BYTES,
                    media_type=payload_type,
                )
            )

        declared = _safe_normalize(declared_media_type)
        if declared is not None:
            signals.append(
                DetectionSignal(
                    source=DetectionSignalSource.DECLARED, media_type=declared
                )
            )

        extension_type = _media_type_for_filename(filename)
        if extension_type is not None:
            signals.append(
                DetectionSignal(
                    source=DetectionSignalSource.EXTENSION,
                    media_type=extension_type,
                )
            )

        return tuple(signals)

    def _inspect_payload(self, header: bytes) -> Optional[str]:
        """
        Determine a media type from the payload bytes alone.

        Byte signatures are tried most specific first. When none matches,
        the buffer is tested for decodability as text, which establishes
        ``text/plain`` and permits later refinement to Markdown, CSV, or
        another signature-free text format.

        Content sniffing is classified as a MAGIC_BYTES signal because it
        is payload inspection, in the same category as a byte signature.

        Parameters
        ----------
        header:
            Leading bytes of the payload.

        Returns
        -------
        Optional[str]
            The indicated media type, or ``None`` if the payload is
            unrecognized binary or empty.
        """
        if not header:
            return None

        for signature in _SIGNATURES:
            if signature.matches(header):
                return signature.media_type

        return "text/plain" if _looks_like_text(header) else None

    # ------------------------------------------------------------------
    # Reconciliation
    # ------------------------------------------------------------------

    def _reconcile(
        self, signals: Sequence[DetectionSignal]
    ) -> tuple[str, tuple[ExtractionWarning, ...]]:
        """
        Select a media type from the gathered signals.

        The strongest available signal establishes a baseline. A weaker
        signal naming a subtype of that baseline refines it silently; a
        weaker signal naming something unrelated is recorded as a
        disagreement and discarded.

        Parameters
        ----------
        signals:
            Signals in descending order of authority.

        Returns
        -------
        tuple[str, tuple[ExtractionWarning, ...]]
            The selected media type and any disagreement warnings.
        """
        by_source = {signal.source: signal.media_type for signal in signals}

        declared = by_source.get(DetectionSignalSource.DECLARED)
        if declared is not None and not self._config.trust_declared_media_type:
            declared = None

        extension = by_source.get(DetectionSignalSource.EXTENSION)
        baseline = by_source.get(DetectionSignalSource.MAGIC_BYTES)

        if baseline is None:
            baseline = declared if declared is not None else extension

        assert baseline is not None  # guaranteed by the caller's emptiness check

        selected = baseline
        for candidate in (extension, declared):
            if candidate is not None and _refines(selected, candidate):
                selected = candidate
                break

        warnings: list[ExtractionWarning] = []
        for source, candidate in (
            (DetectionSignalSource.DECLARED, declared),
            (DetectionSignalSource.EXTENSION, extension),
        ):
            if candidate is None or candidate == selected:
                continue
            if _refines(candidate, selected) or _refines(selected, candidate):
                continue
            warnings.append(
                ExtractionWarning(
                    code=ExtractionWarningCode.MEDIA_TYPE_MISMATCH,
                    message=(
                        f"{source.value} signal indicates {candidate!r} but "
                        f"{selected!r} was selected"
                    ),
                    detail={
                        "signal_source": source.value,
                        "signal_media_type": candidate,
                        "selected_media_type": selected,
                    },
                )
            )

        return selected, tuple(warnings)

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------

    def _resolve_encoding(self, header: bytes, media_type: str) -> Optional[str]:
        """
        Determine the character encoding of a text payload.

        Resolution order is exactness first: a byte-order mark is
        conclusive, strict UTF-8 validation is conclusive, an injected
        sniffer is consulted only afterwards, and the configured fallback
        is used last.

        Validation is performed against the header buffer only. Decoding
        the complete payload is the extractor's responsibility; this stage
        determines which codec it should use.

        Parameters
        ----------
        header:
            Leading bytes of the payload.
        media_type:
            The selected media type.

        Returns
        -------
        Optional[str]
            A Python codec name for text payloads, or ``None`` for binary
            formats.

        Raises
        ------
        EncodingDetectionError
            If no candidate decodes the sample within the configured
            replacement ratio.
        """
        if not _is_text_media_type(media_type):
            return None

        bom_encoding = _encoding_from_byte_order_mark(header)
        if bom_encoding is not None:
            return bom_encoding

        sample = header[: self._config.encoding_sample_bytes]
        if not sample:
            return self._config.default_text_encoding

        validatable = _trim_incomplete_utf8_tail(sample)
        if _decodes_strictly(validatable, "utf-8"):
            return "utf-8"

        attempted = ["utf-8"]

        if self._sniffer is not None:
            sniffed = self._sniffer.sniff(sample)
            if sniffed is not None:
                attempted.append(sniffed)
                if self._is_acceptable(validatable, sniffed):
                    return sniffed

        fallback = self._config.default_text_encoding
        if fallback not in attempted:
            attempted.append(fallback)
        if self._is_acceptable(validatable, fallback):
            return fallback

        raise EncodingDetectionError(
            f"Unable to decode a {media_type!r} payload within the configured "
            f"replacement ratio of {self._config.max_replacement_ratio}.",
            attempted_encodings=attempted,
            context={"media_type": media_type, "sample_bytes": len(sample)},
        )

    def _is_acceptable(self, sample: bytes, encoding: str) -> bool:
        """
        Report whether an encoding decodes a sample acceptably.

        A decode that succeeds only by substituting replacement characters
        for a large proportion of the input has not identified the
        encoding; it has merely produced output. The configured ratio is
        an operational data quality limit and carries no truth judgement.

        Parameters
        ----------
        sample:
            Bytes to decode.
        encoding:
            Candidate codec name.

        Returns
        -------
        bool
            ``True`` when the replacement proportion is within the
            configured limit and the codec is known.
        """
        try:
            decoded = sample.decode(encoding, errors="replace")
        except LookupError:
            return False

        if not decoded:
            return True

        ratio = decoded.count(_REPLACEMENT_CHAR) / len(decoded)
        return ratio <= self._config.max_replacement_ratio


# ============================================================================
# HELPERS
# ============================================================================


def _safe_normalize(media_type: str | None) -> Optional[str]:
    """
    Normalize a media type, treating malformed input as absent.

    A source declaring a malformed type must not abort detection: the
    other signals may well resolve the payload correctly.

    Parameters
    ----------
    media_type:
        Candidate media type, possibly ``None`` or malformed.

    Returns
    -------
    Optional[str]
        The normalized type, or ``None`` if absent or unusable.
    """
    if media_type is None:
        return None
    try:
        return normalize_media_type(media_type)
    except ValueError:
        return None


def _media_type_for_filename(filename: str | None) -> Optional[str]:
    """
    Resolve a media type from a filename extension.

    Compound extensions are handled by taking the final component, so
    ``report.final.docx`` resolves correctly.

    Parameters
    ----------
    filename:
        Filename or path, possibly ``None``.

    Returns
    -------
    Optional[str]
        The indicated media type, or ``None`` when absent or unrecognized.
    """
    if not filename:
        return None
    suffix = PurePosixPath(filename.replace("\\", "/")).suffix
    if not suffix:
        return None
    return _EXTENSION_MEDIA_TYPES.get(suffix.lstrip(".").lower())


def _refines(general: str, specific: str) -> bool:
    """
    Report whether one media type is a more specific form of another.

    Parameters
    ----------
    general:
        The broader media type, typically from payload inspection.
    specific:
        The candidate refinement.

    Returns
    -------
    bool
        ``True`` when ``specific`` names a subtype of ``general``.
    """
    return specific in _REFINEMENTS.get(general, frozenset())


def _encoding_from_byte_order_mark(header: bytes) -> Optional[str]:
    """
    Identify an encoding from a leading byte-order mark.

    Marks are tested longest first so that the four-byte UTF-32 marks are
    not shadowed by the two-byte UTF-16 marks that prefix them.

    The returned codec consumes the mark. Returning plain ``"utf-8"`` for a
    BOM-prefixed payload would leave U+FEFF at character zero of the
    document text, displacing every block span by one.

    Parameters
    ----------
    header:
        Leading bytes of the payload.

    Returns
    -------
    Optional[str]
        A BOM-consuming codec name, or ``None`` if no mark is present.
    """
    probe = header[:_MAX_BOM_LENGTH]
    for mark, encoding in _BYTE_ORDER_MARKS:
        if probe.startswith(mark):
            return encoding
    return None


def _trim_incomplete_utf8_tail(sample: bytes) -> bytes:
    """
    Remove a multi-byte sequence severed by the buffer boundary.

    A header buffer is an arbitrary byte cut, so a UTF-8 character is
    routinely truncated at the tail. Validating without trimming would
    reject valid UTF-8 documents and make classification depend on the
    configured buffer size, which would be non-deterministic with respect
    to configuration.

    Parameters
    ----------
    sample:
        Bytes to trim.

    Returns
    -------
    bytes
        The sample with at most three trailing bytes of an incomplete
        sequence removed.
    """
    for trailing in range(1, min(_MAX_UTF8_SEQUENCE, len(sample) + 1)):
        byte = sample[-trailing]
        if byte < 0x80:
            break
        if byte >= 0xC0:
            expected = _utf8_sequence_length(byte)
            if expected > trailing:
                return sample[:-trailing]
            break
    return sample


def _utf8_sequence_length(lead_byte: int) -> int:
    """
    Determine the length of a UTF-8 sequence from its lead byte.

    Parameters
    ----------
    lead_byte:
        The leading byte of a sequence.

    Returns
    -------
    int
        Total sequence length in bytes; ``1`` for a non-lead byte.
    """
    if lead_byte >= 0xF0:
        return 4
    if lead_byte >= 0xE0:
        return 3
    if lead_byte >= 0xC0:
        return 2
    return 1


def _decodes_strictly(sample: bytes, encoding: str) -> bool:
    """
    Report whether a sample decodes without error under an encoding.

    Parameters
    ----------
    sample:
        Bytes to decode.
    encoding:
        Candidate codec name.

    Returns
    -------
    bool
        ``True`` when decoding succeeds with no substitution.
    """
    try:
        sample.decode(encoding, errors="strict")
    except (UnicodeDecodeError, LookupError):
        return False
    return True


def _looks_like_text(header: bytes) -> bool:
    """
    Report whether a header buffer plausibly contains text.

    A NUL byte is treated as conclusive evidence of binary content: it
    does not occur in UTF-8 text and is common in binary formats. A
    payload free of NUL bytes that decodes as UTF-8 is treated as text.

    Parameters
    ----------
    header:
        Leading bytes of the payload.

    Returns
    -------
    bool
        ``True`` when the payload appears to be text.
    """
    if b"\x00" in header:
        return False
    return _decodes_strictly(_trim_incomplete_utf8_tail(header), "utf-8")