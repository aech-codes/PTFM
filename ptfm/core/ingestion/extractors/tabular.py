"""
PTFM Tabular Extractor

File:
    ptfm/core/ingestion/extractors/tabular.py

Description:
    Extracts a table block tree from delimiter-separated payloads, covering
    comma separated, tab separated, and related formats.

Standard library only
---------------------
    The standard library ``csv`` module handles quoting, embedded delimiters,
    embedded newlines, and dialect sniffing. Adding pandas or pyarrow for what
    is fundamentally row iteration would introduce a large dependency, and a
    DataFrame's type inference is actively unwanted here: a value must reach
    the corpus as the source wrote it, not as the interpretation a numeric
    parser placed on it. An identifier such as ``007`` must not become ``7``,
    and a version such as ``1.10`` must not become ``1.1``.

Structure model
---------------
    A table container holds one row container per record, each holding one
    cell per field. Header cells are marked where a header row was
    identified, so that downstream chunking can carry column names alongside
    values rather than losing them.

    The row and cell tree is preserved rather than flattened to text because
    it is what makes a cell citable. Evidence pointing at a single value can
    name the row and column it came from, which a flattened rendering cannot
    support.

Header detection
----------------
    A first row is treated as a header only when every field is non-empty,
    unique, and not purely numeric. That combination is characteristic of
    column names and uncommon in data rows.

    Where the test does not pass, no header is claimed and every row is
    treated as data. A wrongly identified header would silently remove a data
    row from the corpus, which is worse than leaving column names
    unidentified.

Ragged rows
-----------
    Rows with more or fewer fields than the header are preserved exactly as
    written, and reported once. Padding or truncating would fabricate or
    discard data; recording the observation lets an operator judge whether
    the file is malformed or merely irregular.

Dependency contract
-------------------
    Imports only the standard library, ``ptfm.core.schemas`` and the ingestion
    ``exceptions``, ``interfaces``, ``models`` and ``extractors.plaintext``
    modules. Decoding is shared with the plain text extractor rather than
    reimplemented.
"""

from __future__ import annotations

import csv
import io
from typing import Final, Iterator, Optional, Sequence

from pydantic import ValidationError

from ptfm.core.ingestion.exceptions import (
    CorruptDocumentError,
    ExtractorInternalError,
)
from ptfm.core.ingestion.extractors.plaintext import decode_payload
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

__all__ = ["TabularExtractor"]

#: Media types claimed by this extractor.
_MEDIA_TYPES: Final[frozenset[str]] = frozenset(
    {"text/csv", "text/tab-separated-values"}
)

#: Delimiter implied by a media type, used when sniffing is inconclusive.
#: A tab separated file is defined by its delimiter, so it is never sniffed.
_MEDIA_TYPE_DELIMITERS: Final[dict[str, str]] = {
    "text/csv": ",",
    "text/tab-separated-values": "\t",
}

#: Delimiters offered to the dialect sniffer, in the order preferred when the
#: evidence is balanced.
_CANDIDATE_DELIMITERS: Final[str] = ",\t;|"

#: Leading characters sampled for dialect detection. Sufficient to establish a
#: delimiter without scanning a large file twice.
_SNIFF_SAMPLE_CHARS: Final[int] = 8_192

#: Largest field the parser will accept, in characters. The module default is
#: comparatively small and rejects legitimate files containing embedded prose,
#: while an unbounded limit would let a malformed quotation consume the whole
#: document as a single field.
_MAX_FIELD_CHARS: Final[int] = 4 * 1024 * 1024


class TabularExtractor(ContentExtractor):
    """
    Extracts a table block tree from delimiter-separated payloads.

    Stateless: all parsing state is local to a call, so one instance serves
    all concurrent workers.

    Parameters
    ----------
    detect_header:
        Whether the first row is examined for column names. Disable when
        ingesting files known to be headerless, so that a first data row
        cannot be mistaken for column names.
    """

    __slots__ = ("_detect_header",)

    def __init__(self, *, detect_header: bool = True) -> None:
        self._detect_header = detect_header

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
            The identifier ``"TabularExtractor"``.
        """
        return "TabularExtractor"

    @property
    def supported_media_types(self) -> frozenset[str]:
        """
        Media types this extractor claims.

        Returns
        -------
        frozenset[str]
            The comma and tab separated media types.
        """
        return _MEDIA_TYPES

    # ------------------------------------------------------------------
    # Extraction
    # ------------------------------------------------------------------

    def extract(self, payload: ExtractionInput) -> ExtractionOutput:
        """
        Parse a delimiter-separated payload into a table block tree.

        Blocking and CPU-bound by contract. The extraction stage executes this
        in a worker thread; it must never be invoked from a coroutine.

        Parameters
        ----------
        payload:
            The raw bytes together with the resolved media type and encoding.

        Returns
        -------
        ExtractionOutput
            A table container holding one row per record, each holding one
            cell per field, together with any observations recorded while
            parsing.

        Raises
        ------
        CorruptDocumentError
            If the payload is malformed beyond what the parser can recover
            from, such as an unterminated quotation.
        ExtractorInternalError
            If the resolved encoding is unrecognized, or the assembled result
            failed its own validation.
        """
        warnings: list[ExtractionWarning] = []
        text = decode_payload(payload, warnings)

        if not text.strip():
            return self._empty_result(warnings)

        delimiter = _resolve_delimiter(payload.media_type, text, warnings)
        rows = self._read_rows(text, delimiter, payload)

        if not rows:
            return self._empty_result(warnings)

        header = self._identify_header(rows)
        blocks = _build_blocks(rows, header_present=header is not None)
        _report_ragged_rows(rows, header, warnings)

        try:
            return ExtractionOutput(
                blocks=blocks,
                metadata=ExtractedMetadata(
                    properties={
                        "delimiter": _describe_delimiter(delimiter),
                        "row_count": str(len(rows)),
                        "column_count": str(len(rows[0])),
                        "has_header": str(header is not None).lower(),
                    }
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

    def _read_rows(
        self,
        text: str,
        delimiter: str,
        payload: ExtractionInput,
    ) -> list[list[str]]:
        """
        Parse the payload into rows of fields.

        Values are returned exactly as written. No type inference is applied,
        because an identifier such as ``007`` and a version such as ``1.10``
        must reach the corpus unchanged rather than as the interpretation a
        numeric parser would place on them.

        Entirely empty rows are dropped, since a blank line between records is
        formatting rather than data.

        Parameters
        ----------
        text:
            The decoded payload.
        delimiter:
            The resolved field delimiter.
        payload:
            The originating payload, for error provenance.

        Returns
        -------
        list[list[str]]
            Rows of fields, in document order.

        Raises
        ------
        CorruptDocumentError
            If the parser could not read the payload.
        """
        previous_limit = csv.field_size_limit()
        try:
            csv.field_size_limit(_MAX_FIELD_CHARS)
            reader = csv.reader(
                io.StringIO(text, newline=""), delimiter=delimiter
            )
            return [row for row in reader if any(field.strip() for field in row)]
        except csv.Error as exc:
            raise CorruptDocumentError(
                f"Delimiter-separated payload {payload.source_uri!r} could "
                f"not be parsed: {exc}",
                source_uri=payload.source_uri,
                context={
                    "extractor_name": self.name,
                    "delimiter": _describe_delimiter(delimiter),
                },
            ) from exc
        finally:
            # The limit is process-global, so it is restored rather than left
            # raised for every other consumer in the process.
            csv.field_size_limit(previous_limit)

    def _identify_header(self, rows: Sequence[Sequence[str]]) -> Optional[Sequence[str]]:
        """
        Determine whether the first row holds column names.

        A header is claimed only when every field is non-empty, unique, and
        not purely numeric, which is characteristic of column names and
        uncommon in data rows.

        Where the test does not pass, no header is claimed. A wrongly
        identified header would silently remove a data row from the corpus,
        which is worse than leaving column names unidentified.

        Parameters
        ----------
        rows:
            The parsed rows.

        Returns
        -------
        Optional[Sequence[str]]
            The header row, or ``None`` when none was identified.
        """
        if not self._detect_header or not rows:
            return None

        candidate = rows[0]
        if not candidate:
            return None

        values = [field.strip() for field in candidate]
        if not all(values):
            return None
        if len(set(values)) != len(values):
            return None
        if any(_is_numeric(value) for value in values):
            return None

        return candidate

    def _empty_result(
        self,
        warnings: list[ExtractionWarning],
    ) -> ExtractionOutput:
        """
        Produce the result for a payload containing no records.

        No table container is emitted. A container with no rows would hold no
        text and could not be assigned a span, and asserting an empty table
        would describe structure the document does not have.

        Parameters
        ----------
        warnings:
            Observations recorded while decoding.

        Returns
        -------
        ExtractionOutput
            An empty result carrying the decoding observations.
        """
        return ExtractionOutput(
            blocks=(),
            metadata=ExtractedMetadata(properties={"row_count": "0"}),
            warnings=tuple(warnings),
            extractor_name=self.name,
        )


# ============================================================================
# BLOCK CONSTRUCTION
# ============================================================================


def _build_blocks(
    rows: Sequence[Sequence[str]],
    *,
    header_present: bool,
) -> tuple[ExtractedBlock, ...]:
    """
    Convert parsed rows into a canonical table block tree.

    Blocks are emitted in depth-first pre-order: the table, then each row,
    then that row's cells. Every cell carries a text child, since a cell is a
    container and one holding no children could not be assigned a span.

    Parameters
    ----------
    rows:
        The parsed rows.
    header_present:
        Whether the first row holds column names.

    Returns
    -------
    tuple[ExtractedBlock, ...]
        Blocks in depth-first pre-order.
    """
    blocks: list[ExtractedBlock] = [
        ExtractedBlock(block_type=BlockType.TABLE, text=None)
    ]

    for row_index, row in enumerate(rows):
        is_header_row = header_present and row_index == 0

        row_block_index = len(blocks)
        blocks.append(
            ExtractedBlock(
                block_type=BlockType.TABLE_ROW,
                text=None,
                parent_index=0,
                attributes=BlockAttributes(
                    row_index=row_index, is_header=is_header_row
                ),
            )
        )

        for column_index, field in enumerate(row):
            cell_index = len(blocks)
            blocks.append(
                ExtractedBlock(
                    block_type=BlockType.TABLE_CELL,
                    text=None,
                    parent_index=row_block_index,
                    attributes=BlockAttributes(
                        row_index=row_index,
                        column_index=column_index,
                        is_header=is_header_row,
                    ),
                )
            )
            # A cell is a container, so its value lives in a leaf child.
            # Empty values are still emitted, because a blank cell is part of
            # the table's shape and omitting it would misalign every
            # subsequent column.
            blocks.append(
                ExtractedBlock(
                    block_type=BlockType.PARAGRAPH,
                    text=field,
                    parent_index=cell_index,
                )
            )

        if not row:
            # A row with no fields would be a childless container. Not
            # reachable through the reader, which never yields one, but
            # guarded so that the invariant holds by construction.
            blocks.append(
                ExtractedBlock(
                    block_type=BlockType.TABLE_CELL,
                    text=None,
                    parent_index=row_block_index,
                )
            )
            blocks.append(
                ExtractedBlock(
                    block_type=BlockType.PARAGRAPH,
                    text="",
                    parent_index=len(blocks) - 1,
                )
            )

    return tuple(blocks)


# ============================================================================
# DIALECT AND DIAGNOSTICS
# ============================================================================


def _resolve_delimiter(
    media_type: str,
    text: str,
    warnings: list[ExtractionWarning],
) -> str:
    """
    Determine the field delimiter for a payload.

    A tab separated media type is definitive and is never sniffed. Otherwise
    the leading sample is examined, and the media type's implied delimiter is
    used when sniffing is inconclusive.

    Parameters
    ----------
    media_type:
        The resolved media type.
    text:
        The decoded payload.
    warnings:
        Accumulator to which an observation is appended when the detected
        delimiter differs from the one the media type implies.

    Returns
    -------
    str
        The field delimiter.
    """
    implied = _MEDIA_TYPE_DELIMITERS.get(media_type, ",")

    if media_type == "text/tab-separated-values":
        # The format is defined by its delimiter; sniffing could only
        # contradict its own definition.
        return "\t"

    try:
        dialect = csv.Sniffer().sniff(
            text[:_SNIFF_SAMPLE_CHARS], delimiters=_CANDIDATE_DELIMITERS
        )
    except csv.Error:
        # Inconclusive, which a single-column file always is. The media type's
        # implied delimiter is the best available answer and is harmless when
        # there are no delimiters to find.
        return implied

    detected = dialect.delimiter
    if detected != implied:
        warnings.append(
            ExtractionWarning(
                code=ExtractionWarningCode.STRUCTURE_INFERRED,
                message=(
                    f"Detected field delimiter "
                    f"{_describe_delimiter(detected)} differs from the "
                    f"{_describe_delimiter(implied)} implied by the media "
                    f"type; the detected delimiter was used"
                ),
                detail={
                    "detected_delimiter": _describe_delimiter(detected),
                    "implied_delimiter": _describe_delimiter(implied),
                    "media_type": media_type,
                },
            )
        )
    return detected


def _report_ragged_rows(
    rows: Sequence[Sequence[str]],
    header: Optional[Sequence[str]],
    warnings: list[ExtractionWarning],
) -> None:
    """
    Record rows whose field count differs from the first row.

    Rows are preserved exactly as written. Padding or truncating them would
    fabricate or discard data; recording the observation lets an operator
    judge whether the file is malformed or merely irregular.

    Reported once for the whole document rather than once per row, so that a
    systematically irregular file does not flood the warning list.

    Parameters
    ----------
    rows:
        The parsed rows.
    header:
        The identified header row, when one was found.
    warnings:
        Accumulator to which the observation is appended.

    Returns
    -------
    None
    """
    expected = len(header) if header is not None else len(rows[0])
    ragged = [
        index for index, row in enumerate(rows) if len(row) != expected
    ]
    if not ragged:
        return

    warnings.append(
        ExtractionWarning(
            code=ExtractionWarningCode.TABLE_STRUCTURE_UNCERTAIN,
            message=(
                f"{len(ragged)} of {len(rows)} rows have a field count "
                f"differing from the expected {expected}; all rows were "
                f"preserved as written"
            ),
            detail={
                "expected_columns": expected,
                "ragged_row_count": len(ragged),
                "total_row_count": len(rows),
                "first_ragged_row": ragged[0],
            },
        )
    )


def _is_numeric(value: str) -> bool:
    """
    Report whether a value parses as a number.

    Used only to reject a first row as column names. A row of numbers is data,
    not a header.

    Parameters
    ----------
    value:
        The field value.

    Returns
    -------
    bool
        ``True`` when the value parses as a number.
    """
    try:
        float(value.replace(",", ""))
    except ValueError:
        return False
    return True


def _describe_delimiter(delimiter: str) -> str:
    """
    Render a delimiter in a form readable in diagnostics.

    Parameters
    ----------
    delimiter:
        The delimiter character.

    Returns
    -------
    str
        A readable description, naming whitespace characters rather than
        emitting them.
    """
    names = {"\t": "tab", ",": "comma", ";": "semicolon", "|": "pipe", " ": "space"}
    return names.get(delimiter, repr(delimiter))


def _iter_cells(
    blocks: Sequence[ExtractedBlock],
) -> Iterator[tuple[int, ExtractedBlock]]:
    """
    Iterate the cell blocks of a table tree with their positions.

    Provided for tests and for callers inspecting a result without
    reimplementing the tree walk.

    Parameters
    ----------
    blocks:
        Blocks in depth-first pre-order.

    Yields
    ------
    tuple[int, ExtractedBlock]
        Index and block for every table cell.
    """
    for index, block in enumerate(blocks):
        if block.block_type is BlockType.TABLE_CELL:
            yield index, block
