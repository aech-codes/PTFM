"""
PTFM Plain Text Extractor

File:
    ptfm/core/ingestion/extractors/plaintext.py

Description:
    Extracts blocks from plain text payloads.

Structure model
---------------
    Plain text carries no markup, so the only structure available is
    segmentation: runs of text separated by blank lines become paragraphs.
    Nothing further is inferred.

    Inferring more would be guessing. A leading line is not reliably a title,
    an indented run is not reliably a quotation, and an underlined line is not
    reliably a heading. Recording a guess as structure would place an
    unfounded claim into the evidence chain, where a later citation would
    inherit it. Segmentation is the most that can be asserted honestly.

Decoding
--------
    Decoding is attempted strictly first. If that fails, the payload is
    decoded with replacement and an ``ENCODING_REPLACEMENT`` warning is
    recorded, so that a handful of malformed bytes at the end of a large file
    degrades fidelity rather than discarding the document.

    No proportion of replacement characters is judged here. Format detection
    already applied the configured limit to a sample and concluded the payload
    was text; applying a second, independent limit would duplicate that
    judgement and let the two disagree.

Dependency contract
-------------------
    Imports only the standard library, ``ptfm.core.schemas`` and the ingestion
    ``exceptions``, ``interfaces`` and ``models`` modules. No third-party
    parser is required, so this extractor is always available and can serve as
    the fallback for text formats whose dedicated extractor is not installed.
"""

from __future__ import annotations

import re
from typing import Final, Iterable, Optional

from pydantic import ValidationError

from ptfm.core.ingestion.exceptions import ExtractorInternalError
from ptfm.core.ingestion.interfaces import ContentExtractor, normalize_media_type
from ptfm.core.ingestion.models import (
    ExtractedBlock,
    ExtractedMetadata,
    ExtractionInput,
    ExtractionOutput,
)
from ptfm.core.schemas import (
    BlockType,
    ExtractionWarning,
    ExtractionWarningCode,
)

__all__ = ["decode_payload", "PlainTextExtractor"]

#: Media types claimed unless the composition root declares otherwise.
_DEFAULT_MEDIA_TYPES: Final[frozenset[str]] = frozenset({"text/plain"})

#: Encoding assumed when detection resolved none. Detection supplies an
#: encoding for every text media type, so this applies only when a payload
#: reaches this extractor by an unusual route.
_FALLBACK_ENCODING: Final[str] = "utf-8"

#: Character produced in place of an undecodable byte.
_REPLACEMENT_CHAR: Final[str] = "\ufffd"

#: Byte order mark, as a decoded character. A codec resolved by detection
#: consumes it, but a payload reaching this extractor with a plain codec would
#: otherwise carry it as the first character of the document, displacing every
#: block span by one.
_BOM_CHARACTER: Final[str] = "\ufeff"

#: Any line ending, normalized to a line feed before segmentation. The
#: structure stage normalizes the canonical text as well; this is required
#: here because a blank line separator cannot otherwise be recognized in a
#: payload using carriage returns.
_LINE_ENDINGS: Final[re.Pattern[str]] = re.compile(r"\r\n|\r|\u2028|\u2029")

#: A blank line, optionally containing horizontal whitespace, separating
#: paragraphs.
_PARAGRAPH_BREAK: Final[re.Pattern[str]] = re.compile(r"\n[ \t]*\n")


def decode_payload(
    payload: ExtractionInput,
    warnings: list[ExtractionWarning],
) -> str:
    """
    Decode a text payload, degrading rather than failing on bad bytes.

    A strict decode is attempted first, so that a well-formed payload is
    reproduced exactly. On failure the payload is decoded with replacement and
    the loss is recorded, because discarding an entire document over a few
    malformed trailing bytes serves nobody.

    Any leading byte order mark is removed. A mark left in place would occupy
    character zero of the document text and displace every block span by one.

    Shared with other text extractors so that decoding behaves identically
    across formats rather than being reimplemented per extractor.

    Parameters
    ----------
    payload:
        The payload to decode.
    warnings:
        Accumulator to which an ``ENCODING_REPLACEMENT`` observation is
        appended when the decode was lossy.

    Returns
    -------
    str
        The decoded text.

    Raises
    ------
    ExtractorInternalError
        If the resolved encoding is not a codec this interpreter recognizes,
        which indicates a detection or configuration fault rather than a
        malformed document.
    """
    encoding = payload.encoding or _FALLBACK_ENCODING

    try:
        text = payload.data.decode(encoding, errors="strict")
    except LookupError as exc:
        raise ExtractorInternalError(
            f"Encoding {encoding!r} resolved for {payload.source_uri!r} is not "
            f"a recognized codec.",
            extractor_name="decode_payload",
            source_uri=payload.source_uri,
            context={"encoding": encoding},
        ) from exc
    except UnicodeDecodeError:
        text = payload.data.decode(encoding, errors="replace")
        warnings.append(
            ExtractionWarning(
                code=ExtractionWarningCode.ENCODING_REPLACEMENT,
                message=(
                    f"Payload could not be decoded strictly as {encoding!r}; "
                    f"undecodable bytes were replaced"
                ),
                detail={
                    "encoding": encoding,
                    "replacement_count": text.count(_REPLACEMENT_CHAR),
                    "character_count": len(text),
                },
            )
        )

    return text.removeprefix(_BOM_CHARACTER)


class PlainTextExtractor(ContentExtractor):
    """
    Segments plain text payloads into paragraph blocks.

    Stateless: no instance state is mutated during extraction, so one instance
    serves all concurrent workers.

    Parameters
    ----------
    media_types:
        Media types this extractor claims. Parameterized so that a composition
        root can additionally register it for ``text/*``, giving every text
        format a working fallback when its dedicated extractor is not
        installed. The default claims plain text alone.
    priority:
        Resolution precedence. A fallback registration should be given a
        negative value, so that it can never displace a dedicated extractor
        for a format it merely happens to be able to read.
    """

    __slots__ = ("_media_types", "_priority")

    def __init__(
        self,
        media_types: Optional[Iterable[str]] = None,
        *,
        priority: int = 0,
    ) -> None:
        self._media_types = (
            frozenset(normalize_media_type(item) for item in media_types)
            if media_types is not None
            else _DEFAULT_MEDIA_TYPES
        )
        self._priority = priority

    # ------------------------------------------------------------------
    # Declaration
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        """
        Stable identifier of this extractor implementation.

        Returns
        -------
        str
            The identifier ``"PlainTextExtractor"``.
        """
        return "PlainTextExtractor"

    @property
    def supported_media_types(self) -> frozenset[str]:
        """
        Media types this extractor claims.

        Returns
        -------
        frozenset[str]
            Normalized media type patterns.
        """
        return self._media_types

    @property
    def priority(self) -> int:
        """
        Resolution precedence among extractors claiming the same media type.

        Returns
        -------
        int
            The configured precedence.
        """
        return self._priority

    # ------------------------------------------------------------------
    # Extraction
    # ------------------------------------------------------------------

    def extract(self, payload: ExtractionInput) -> ExtractionOutput:
        """
        Segment a plain text payload into paragraph blocks.

        Blocking and CPU-bound by contract. The extraction stage executes this
        in a worker thread; it must never be invoked from a coroutine.

        Parameters
        ----------
        payload:
            The raw bytes together with the resolved media type and encoding.

        Returns
        -------
        ExtractionOutput
            One paragraph block per blank-line-separated run of text, with any
            decoding observations recorded.

        Raises
        ------
        ExtractorInternalError
            If the resolved encoding is unrecognized, or the assembled result
            failed its own validation, which would indicate a defect in this
            extractor.
        """
        warnings: list[ExtractionWarning] = []
        text = decode_payload(payload, warnings)

        blocks = tuple(
            ExtractedBlock(block_type=BlockType.PARAGRAPH, text=segment)
            for segment in _segment_paragraphs(text)
        )

        try:
            return ExtractionOutput(
                blocks=blocks,
                metadata=ExtractedMetadata(
                    properties={"line_count": str(_count_lines(text))}
                ),
                warnings=tuple(warnings),
                extractor_name=self.name,
            )
        except ValidationError as exc:
            raise ExtractorInternalError(
                f"{self.name} produced an invalid result for "
                f"{payload.source_uri!r}: {exc}",
                extractor_name=self.name,
                source_uri=payload.source_uri,
            ) from exc


# ============================================================================
# HELPERS
# ============================================================================


def _segment_paragraphs(text: str) -> list[str]:
    """
    Split text into paragraphs on blank lines.

    Line endings are normalized first, so that a payload using carriage
    returns segments identically to one using line feeds. Segments that are
    empty or contain only whitespace are dropped: a blank run carries no
    content, and emitting a block for it would produce structure that
    represents nothing.

    Parameters
    ----------
    text:
        The decoded document text.

    Returns
    -------
    list[str]
        Paragraph texts in document order. Empty when the payload contains no
        non-whitespace content.
    """
    normalized = _LINE_ENDINGS.sub("\n", text)
    return [
        segment
        for segment in _PARAGRAPH_BREAK.split(normalized)
        if segment.strip()
    ]


def _count_lines(text: str) -> int:
    """
    Count the lines in a decoded payload.

    Recorded as format-native metadata, since line count is a property a text
    file genuinely has, unlike a page or slide count.

    Parameters
    ----------
    text:
        The decoded document text.

    Returns
    -------
    int
        Number of lines; zero for an empty payload.
    """
    if not text:
        return 0
    return len(_LINE_ENDINGS.sub("\n", text).split("\n"))
