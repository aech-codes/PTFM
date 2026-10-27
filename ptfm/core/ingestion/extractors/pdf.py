"""
PTFM PDF Extractor

File:
    ptfm/core/ingestion/extractors/pdf.py

Description:
    Extracts a block tree from Portable Document Format payloads.

A PDF has no structure to read
------------------------------
    This is the defining constraint of this extractor and the reason it
    differs from every other one in the module. A DOCX states that a paragraph
    is a level two heading. A PDF states that a glyph appears at a coordinate
    in a font at a size. There is no heading, no paragraph, and no list in the
    file; there are only positioned characters that a human reader assembles
    into those things by looking at them.

    Everything above the character level must therefore be inferred, and every
    inference can be wrong.

What is claimed, and what is not
--------------------------------
    Claimed:
        Pages, which the format genuinely declares.
        Line grouping, from the vertical position the format genuinely
        records.
        Paragraph grouping, from vertical gaps between lines.

    Not claimed:
        Headings. A larger or bolder line is frequently a heading and
        frequently a pull quote, a header, a figure label, or a page number.
        No heading block is ever emitted, because a wrong heading enters the
        corpus as an assertion about document structure that the document
        never made, and a citation quoting it inherits that assertion.

        Tables. Aligned text is sometimes a table and sometimes columns of
        prose. Reconstructing a grid from coordinates would produce cell
        positions that are confidently wrong, and a value cited by row and
        column would then be cited incorrectly.

        Lists. A leading bullet character is a glyph like any other.

    Where the format supplies an outline, its entries are recorded as metadata
    rather than converted into heading blocks, since an outline names
    destinations rather than delimiting content.

Paragraph grouping is the one heuristic
---------------------------------------
    Lines are grouped into paragraphs where the vertical gap between them
    exceeds the document's own typical line spacing by a configured factor.
    The comparison is against the median gap measured within the same page, so
    the extractor adapts to each document's leading rather than applying a
    fixed measurement that would suit one typesetting and not another.

    The grouping is stated as an inference in a warning, once per document, so
    that a downstream reader knows the paragraph boundaries were derived
    rather than declared.

Scanned documents
-----------------
    A page of images yields no extractable text. That is reported as a
    warning and the page is retained as an empty structure, so that page
    numbering stays correct and an operator can see which pages need optical
    character recognition. No recognition is performed here; it is deferred to
    a later iteration.

Parser selection
----------------
    ``pypdfium2`` is used. It is permissively licensed, which matters: the
    faster alternative ``PyMuPDF`` is AGPL, a licensing hazard for a framework
    intended for enterprise deployment. It exposes character positions, which
    line and paragraph grouping requires, and it is fast enough for corpus
    scale.

    Alternative considered: ``pdfminer.six``, pure Python and permissively
    licensed, but substantially slower.

    The dependency is optional, supplied by the ``ptfm[pdf]`` extra and
    reported through ``check_availability`` at registration rather than as an
    ``ImportError`` partway through a corpus.

Dependency contract
-------------------
    Imports the standard library, ``pypdfium2``, ``ptfm.core.schemas`` and the
    ingestion ``exceptions``, ``interfaces`` and ``models`` modules.
"""

from __future__ import annotations

import re
import statistics
from dataclasses import dataclass
from typing import Any, Final, Iterator, Optional, Sequence

from pydantic import ValidationError

from ptfm.core.ingestion.exceptions import (
    CorruptDocumentError,
    EncryptedDocumentError,
    ExtractorInternalError,
    ExtractorUnavailableError,
)
from ptfm.core.ingestion.interfaces import ContentExtractor
from ptfm.core.ingestion.models import (
    ExtractedBlock,
    ExtractedMetadata,
    ExtractionInput,
    ExtractionOutput,
)
from ptfm.core.schemas import (
    BlockAttributes,
    BlockType,
    ExtractionWarning,
    ExtractionWarningCode,
)

__all__ = ["PdfExtractor"]

#: Media type claimed by this extractor.
_MEDIA_TYPES: Final[frozenset[str]] = frozenset({"application/pdf"})

#: Distribution supplying the parser, and the packaging extra providing it.
_DEPENDENCY: Final[str] = "pypdfium2"
_INSTALL_EXTRA: Final[str] = "ptfm[pdf]"

#: Multiple of a page's typical line spacing above which a vertical gap is
#: treated as a paragraph boundary rather than ordinary leading.
#:
#: Compared against the median gap measured within the same page, so the
#: extractor adapts to each document's typesetting instead of applying a fixed
#: measurement that would suit one layout and not another.
_PARAGRAPH_GAP_FACTOR: Final[float] = 1.6

#: Fallback multiple of text height used where a page has too few lines for a
#: median gap to be meaningful.
_FALLBACK_GAP_FACTOR: Final[float] = 1.8

#: Fewest lines on a page for its median gap to be measured. Below this the
#: fallback applies, since a median over one or two samples describes nothing.
_MIN_LINES_FOR_MEDIAN: Final[int] = 4

#: Largest vertical distance, as a fraction of text height, within which two
#: characters are treated as belonging to the same line. Accommodates
#: subscripts, superscripts, and minor baseline variation.
_LINE_TOLERANCE_FACTOR: Final[float] = 0.5

#: Horizontal gap, as a multiple of character width, above which a space is
#: inserted between adjacent characters. Many PDFs position words by
#: coordinate without emitting space characters at all.
_WORD_GAP_FACTOR: Final[float] = 0.3

#: A run of whitespace, collapsed within a line so that coordinate-derived
#: spacing does not become runs of document content.
_WHITESPACE_RUN: Final[re.Pattern[str]] = re.compile(r"[ \t]+")


@dataclass(frozen=True, slots=True)
class _Character:
    """
    One positioned glyph on a page.

    Attributes
    ----------
    text:
        The character.
    left:
        Horizontal position of the character's left edge.
    bottom:
        Vertical position of the character's baseline.
    width:
        Horizontal extent.
    height:
        Vertical extent.
    """

    text: str
    left: float
    bottom: float
    width: float
    height: float


@dataclass(frozen=True, slots=True)
class _Line:
    """
    A run of characters sharing a baseline.

    Attributes
    ----------
    text:
        The assembled line text.
    bottom:
        Baseline position, used to measure the gap to the following line.
    height:
        Typical character height on the line.
    """

    text: str
    bottom: float
    height: float


class PdfExtractor(ContentExtractor):
    """
    Extracts a page and paragraph block tree from PDF payloads.

    Stateless: all parsing state is local to a call, so one instance serves
    all concurrent workers.

    Parameters
    ----------
    group_paragraphs:
        Whether lines are grouped into paragraphs by vertical spacing. When
        disabled, each line becomes its own block, which is the more
        conservative choice for documents whose layout defeats grouping, such
        as multi-column academic papers.
    """

    __slots__ = ("_group_paragraphs",)

    def __init__(self, *, group_paragraphs: bool = True) -> None:
        self._group_paragraphs = group_paragraphs

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
            The identifier ``"PdfExtractor"``.
        """
        return "PdfExtractor"

    @property
    def supported_media_types(self) -> frozenset[str]:
        """
        Media types this extractor claims.

        Returns
        -------
        frozenset[str]
            The PDF media type.
        """
        return _MEDIA_TYPES

    @classmethod
    def check_availability(cls) -> None:
        """
        Verify that the PDF parser is installed.

        Invoked at registration, so that a missing extra produces an
        actionable startup error naming the remedy rather than an
        ``ImportError`` partway through a corpus.

        Returns
        -------
        None

        Raises
        ------
        ExtractorUnavailableError
            If the parser is not installed.
        """
        try:
            import pypdfium2  # noqa: F401
        except ImportError as exc:
            raise ExtractorUnavailableError(
                media_type="application/pdf",
                missing_dependency=_DEPENDENCY,
                install_extra=_INSTALL_EXTRA,
            ) from exc

    # ------------------------------------------------------------------
    # Extraction
    # ------------------------------------------------------------------

    def extract(self, payload: ExtractionInput) -> ExtractionOutput:
        """
        Extract page and paragraph blocks from a PDF payload.

        Blocking and CPU-bound by contract. The extraction stage executes this
        in a worker thread; it must never be invoked from a coroutine.

        Parameters
        ----------
        payload:
            The raw bytes together with the resolved media type.

        Returns
        -------
        ExtractionOutput
            One page container per page, holding paragraph blocks, together
            with document metadata and the observations recorded while
            extracting.

        Raises
        ------
        ExtractorUnavailableError
            If the parser is not installed.
        EncryptedDocumentError
            If the document is password protected.
        CorruptDocumentError
            If the document is malformed and cannot be opened.
        ExtractorInternalError
            If the parser raised an unanticipated exception, or the assembled
            result failed its own validation.
        """
        self.check_availability()
        import pypdfium2

        warnings: list[ExtractionWarning] = []
        document: Any = None

        try:
            document = pypdfium2.PdfDocument(payload.data)
            page_count = len(document)
            blocks = self._extract_pages(document, page_count, warnings)
            metadata = _harvest_metadata(document, page_count, warnings)
        except pypdfium2.PdfiumError as exc:
            raise _classify_parser_error(exc, payload.source_uri, self.name) from exc
        except ExtractorInternalError:
            raise
        except Exception as exc:
            raise ExtractorInternalError(
                f"PDF parser raised {type(exc).__name__} while reading "
                f"{payload.source_uri!r}: {exc}",
                extractor_name=self.name,
                source_uri=payload.source_uri,
            ) from exc
        finally:
            if document is not None:
                # The parser holds native resources that are not released by
                # garbage collection alone.
                document.close()

        try:
            return ExtractionOutput(
                blocks=blocks,
                metadata=metadata,
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

    # ------------------------------------------------------------------
    # Pages
    # ------------------------------------------------------------------

    def _extract_pages(
        self,
        document: Any,
        page_count: int,
        warnings: list[ExtractionWarning],
    ) -> tuple[ExtractedBlock, ...]:
        """
        Extract every page into a container holding its text blocks.

        A page yielding no text is retained as an empty structure rather than
        omitted, so that page numbering stays correct and an operator can see
        which pages require optical character recognition.

        Parameters
        ----------
        document:
            The opened document.
        page_count:
            Number of pages.
        warnings:
            Accumulator to which observations are appended.

        Returns
        -------
        tuple[ExtractedBlock, ...]
            Blocks in depth-first pre-order.

        Raises
        ------
        ExtractorInternalError
            If a page could not be read.
        """
        blocks: list[ExtractedBlock] = []
        empty_pages: list[int] = []
        grouping_applied = False

        for page_number in range(page_count):
            page_index = len(blocks)
            blocks.append(
                ExtractedBlock(
                    block_type=BlockType.SECTION,
                    text=None,
                    attributes=BlockAttributes(page_number=page_number + 1),
                )
            )

            lines = self._read_page_lines(document, page_number)
            segments = self._group_lines(lines)
            if len(segments) < len(lines):
                grouping_applied = True

            emitted = 0
            for segment in segments:
                if not segment.strip():
                    continue
                blocks.append(
                    ExtractedBlock(
                        block_type=BlockType.PARAGRAPH,
                        text=segment,
                        parent_index=page_index,
                        attributes=BlockAttributes(page_number=page_number + 1),
                    )
                )
                emitted += 1

            if emitted == 0:
                empty_pages.append(page_number + 1)
                # A page container with no children could not be assigned a
                # span, so an empty block preserves the page's position in the
                # document without asserting content.
                blocks.append(
                    ExtractedBlock(
                        block_type=BlockType.PARAGRAPH,
                        text="",
                        parent_index=page_index,
                        attributes=BlockAttributes(page_number=page_number + 1),
                    )
                )

        _report_empty_pages(empty_pages, page_count, warnings)
        if grouping_applied and self._group_paragraphs:
            _report_paragraph_inference(warnings)

        return tuple(blocks)

    def _read_page_lines(self, document: Any, page_number: int) -> list[_Line]:
        """
        Read one page and assemble its characters into lines.

        Parameters
        ----------
        document:
            The opened document.
        page_number:
            Zero-based page index.

        Returns
        -------
        list[_Line]
            Lines in reading order, top to bottom.

        Raises
        ------
        ExtractorInternalError
            If the page could not be read.
        """
        page: Any = None
        textpage: Any = None
        try:
            page = document[page_number]
            textpage = page.get_textpage()
            characters = list(_iter_characters(textpage))
        except Exception as exc:
            raise ExtractorInternalError(
                f"Page {page_number + 1} could not be read: "
                f"{type(exc).__name__}: {exc}",
                extractor_name="PdfExtractor",
                context={"page_number": page_number + 1},
            ) from exc
        finally:
            # Native resources again, released in reverse acquisition order.
            if textpage is not None:
                textpage.close()
            if page is not None:
                page.close()

        return _assemble_lines(characters)

    def _group_lines(self, lines: Sequence[_Line]) -> list[str]:
        """
        Group lines into paragraphs by vertical spacing.

        A gap exceeding the page's own typical line spacing by the configured
        factor is treated as a paragraph boundary. Comparing against the
        page's median gap rather than a fixed measurement means the extractor
        adapts to each document's leading.

        Parameters
        ----------
        lines:
            The page's lines in reading order.

        Returns
        -------
        list[str]
            Paragraph texts. Where grouping is disabled, each line is returned
            separately.
        """
        if not lines:
            return []
        if not self._group_paragraphs:
            return [line.text for line in lines]
        if len(lines) == 1:
            return [lines[0].text]

        gaps = [
            abs(lines[index].bottom - lines[index + 1].bottom)
            for index in range(len(lines) - 1)
        ]
        threshold = _paragraph_threshold(gaps, lines)

        paragraphs: list[str] = []
        current: list[str] = [lines[0].text]

        for index, gap in enumerate(gaps):
            if gap > threshold:
                paragraphs.append(" ".join(current))
                current = [lines[index + 1].text]
            else:
                current.append(lines[index + 1].text)

        paragraphs.append(" ".join(current))
        return paragraphs


# ============================================================================
# CHARACTER AND LINE ASSEMBLY
# ============================================================================


def _iter_characters(textpage: Any) -> Iterator[_Character]:
    """
    Iterate the positioned characters of a page.

    Parameters
    ----------
    textpage:
        The page's text object.

    Yields
    ------
    _Character
        Each character with its position and extent.
    """
    count = textpage.count_chars()

    for index in range(count):
        text = textpage.get_text_range(index, 1)
        if not text:
            continue
        left, bottom, right, top = textpage.get_charbox(index, loose=False)
        yield _Character(
            text=text,
            left=left,
            bottom=bottom,
            width=max(0.0, right - left),
            height=max(0.0, top - bottom),
        )


def _assemble_lines(characters: Sequence[_Character]) -> list[_Line]:
    """
    Assemble positioned characters into lines.

    Characters are grouped by baseline, within a tolerance that accommodates
    subscripts, superscripts, and minor baseline variation. Within a line,
    characters are ordered by horizontal position and a space is inserted
    where the gap between them exceeds a fraction of character width, because
    many PDFs position words by coordinate without emitting space characters
    at all.

    Parameters
    ----------
    characters:
        The page's characters, in the order the format stored them.

    Returns
    -------
    list[_Line]
        Lines in reading order, top to bottom.
    """
    if not characters:
        return []

    # The format stores characters in drawing order, which is not necessarily
    # reading order, so they are sorted by position: descending vertically,
    # since PDF coordinates increase upward, then ascending horizontally.
    ordered = sorted(characters, key=lambda item: (-item.bottom, item.left))

    lines: list[_Line] = []
    current: list[_Character] = [ordered[0]]

    for character in ordered[1:]:
        reference = current[-1]
        tolerance = max(reference.height, character.height) * _LINE_TOLERANCE_FACTOR
        if abs(character.bottom - reference.bottom) <= tolerance:
            current.append(character)
            continue
        line = _build_line(current)
        if line is not None:
            lines.append(line)
        current = [character]

    line = _build_line(current)
    if line is not None:
        lines.append(line)

    return lines


def _build_line(characters: Sequence[_Character]) -> Optional[_Line]:
    """
    Assemble one line's characters into text.

    Parameters
    ----------
    characters:
        Characters sharing a baseline.

    Returns
    -------
    Optional[_Line]
        The assembled line, or ``None`` when it holds no visible content.
    """
    if not characters:
        return None

    ordered = sorted(characters, key=lambda item: item.left)
    parts: list[str] = [ordered[0].text]

    for previous, character in zip(ordered, ordered[1:], strict=False):
        gap = character.left - (previous.left + previous.width)
        reference = max(previous.width, character.width)
        if reference > 0 and gap > reference * _WORD_GAP_FACTOR:
            parts.append(" ")
        parts.append(character.text)

    text = _WHITESPACE_RUN.sub(" ", "".join(parts)).strip()
    if not text:
        return None

    heights = [item.height for item in ordered if item.height > 0]
    return _Line(
        text=text,
        bottom=statistics.median([item.bottom for item in ordered]),
        height=statistics.median(heights) if heights else 0.0,
    )


def _paragraph_threshold(
    gaps: Sequence[float],
    lines: Sequence[_Line],
) -> float:
    """
    Determine the vertical gap above which a paragraph boundary is inferred.

    The page's median gap is used where enough lines are present for it to
    describe anything, so that the threshold reflects the document's own
    leading. Otherwise a multiple of text height is used, since a median over
    one or two samples is not a measurement.

    Parameters
    ----------
    gaps:
        Vertical gaps between consecutive lines.
    lines:
        The page's lines.

    Returns
    -------
    float
        The gap above which a paragraph boundary is inferred.
    """
    if len(lines) >= _MIN_LINES_FOR_MEDIAN:
        median_gap = statistics.median(gaps)
        if median_gap > 0:
            return median_gap * _PARAGRAPH_GAP_FACTOR

    heights = [line.height for line in lines if line.height > 0]
    if heights:
        return statistics.median(heights) * _FALLBACK_GAP_FACTOR

    # No usable measurement, so no boundary is inferred and the page becomes
    # a single paragraph. Claiming boundaries without evidence would be worse
    # than claiming none.
    return float("inf")


# ============================================================================
# METADATA AND DIAGNOSTICS
# ============================================================================


def _harvest_metadata(
    document: Any,
    page_count: int,
    warnings: list[ExtractionWarning],
) -> ExtractedMetadata:
    """
    Recover the document's information dictionary and outline.

    Outline entries are recorded as a property rather than converted into
    heading blocks. An outline names destinations within the document rather
    than delimiting content, so its entries cannot be placed in the block tree
    without asserting boundaries the document does not define.

    Timestamps are not recovered. PDF dates use a format that frequently omits
    or misstates its timezone offset, and a misinterpreted instant is worse
    than an absent one.

    Parameters
    ----------
    document:
        The opened document.
    page_count:
        Number of pages.
    warnings:
        Accumulator to which an observation is appended when metadata is
        unavailable.

    Returns
    -------
    ExtractedMetadata
        Format-native metadata.
    """
    properties: dict[str, str] = {}
    title: Optional[str] = None
    authors: tuple[str, ...] = ()
    producer: Optional[str] = None

    try:
        information = document.get_metadata_dict() or {}
    except Exception as exc:
        warnings.append(
            ExtractionWarning(
                code=ExtractionWarningCode.METADATA_UNAVAILABLE,
                message=(
                    f"Document information could not be read "
                    f"({type(exc).__name__}); content was extracted regardless"
                ),
                detail={"exception_type": type(exc).__name__},
            )
        )
        information = {}

    for key, value in information.items():
        cleaned = str(value).strip()
        if not cleaned:
            continue
        normalized = key.strip().lower()
        if normalized == "title":
            title = cleaned
        elif normalized == "author":
            authors = tuple(
                part.strip()
                for part in re.split(r"[,;]| and ", cleaned)
                if part.strip()
            )
        elif normalized == "producer":
            producer = cleaned
        elif normalized in ("creationdate", "moddate"):
            # Retained verbatim rather than parsed. PDF date strings are
            # frequently malformed or lack a timezone, and a wrong instant is
            # worse than none.
            properties[normalized] = cleaned
        else:
            properties[normalized] = cleaned

    outline = _read_outline(document)
    if outline:
        properties["outline_entry_count"] = str(len(outline))
        properties["outline"] = " | ".join(outline)

    return ExtractedMetadata(
        title=title,
        authors=authors,
        page_count=page_count,
        producer=producer,
        properties=properties,
    )


def _read_outline(document: Any) -> list[str]:
    """
    Read the document's outline entry titles.

    Parameters
    ----------
    document:
        The opened document.

    Returns
    -------
    list[str]
        Entry titles in document order, empty when no outline is present or
        it cannot be read.
    """
    try:
        entries = [
            (item.title or "").strip()
            for item in document.get_toc()
            if getattr(item, "title", None)
        ]
    except Exception:
        # An outline is an enhancement to metadata. Its absence or
        # unreadability must not affect a document whose content extracted
        # correctly.
        return []
    return [entry for entry in entries if entry]


def _report_empty_pages(
    empty_pages: Sequence[int],
    page_count: int,
    warnings: list[ExtractionWarning],
) -> None:
    """
    Record pages that yielded no extractable text.

    Reported once for the document rather than once per page, so that a wholly
    scanned document does not produce one warning per page. The page numbers
    are retained so an operator can see exactly which pages require optical
    character recognition.

    Parameters
    ----------
    empty_pages:
        One-based numbers of pages yielding no text.
    page_count:
        Total number of pages.
    warnings:
        Accumulator to which the observation is appended.

    Returns
    -------
    None
    """
    if not empty_pages:
        return

    scanned = len(empty_pages) == page_count
    message = (
        "No page yielded extractable text; the document is most likely "
        "scanned and requires optical character recognition"
        if scanned
        else f"{len(empty_pages)} of {page_count} pages yielded no "
        f"extractable text"
    )

    warnings.append(
        ExtractionWarning(
            code=ExtractionWarningCode.EMPTY_PAGE,
            message=message,
            detail={
                "empty_page_count": len(empty_pages),
                "page_count": page_count,
                "empty_pages": list(empty_pages[:50]),
                "wholly_empty": scanned,
            },
        )
    )


def _report_paragraph_inference(warnings: list[ExtractionWarning]) -> None:
    """
    Record that paragraph boundaries were inferred rather than declared.

    Stated explicitly so that a downstream reader knows the block boundaries
    in this document were derived from vertical spacing rather than read from
    the format, which declares no such structure.

    Parameters
    ----------
    warnings:
        Accumulator to which the observation is appended.

    Returns
    -------
    None
    """
    warnings.append(
        ExtractionWarning(
            code=ExtractionWarningCode.STRUCTURE_INFERRED,
            message=(
                "Paragraph boundaries were inferred from vertical line "
                "spacing. The format declares no paragraph, heading, or list "
                "structure, so no heading or table blocks were emitted."
            ),
            detail={"gap_factor": _PARAGRAPH_GAP_FACTOR},
        )
    )


def _classify_parser_error(
    exc: Exception,
    source_uri: str,
    extractor_name: str,
) -> Exception:
    """
    Convert a parser error into the appropriate PTFM exception.

    An encrypted document and a corrupt one are distinguished because their
    remedies differ entirely: one needs a password, the other needs replacing.

    Parameters
    ----------
    exc:
        The parser's exception.
    source_uri:
        URI of the document being extracted.
    extractor_name:
        Identifier of this extractor.

    Returns
    -------
    Exception
        The exception to raise in place of the parser's, which the caller
        chains as ``__cause__``.
    """
    message = str(exc).lower()

    if "password" in message or "encrypt" in message:
        return EncryptedDocumentError(
            f"Document {source_uri!r} is encrypted or password protected.",
            source_uri=source_uri,
            context={"extractor_name": extractor_name},
        )

    return CorruptDocumentError(
        f"Document {source_uri!r} could not be parsed: {exc}",
        source_uri=source_uri,
        context={"extractor_name": extractor_name},
    )
