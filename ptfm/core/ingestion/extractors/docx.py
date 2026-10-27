"""
PTFM DOCX Extractor

File:
    ptfm/core/ingestion/extractors/docx.py

Description:
    Extracts a structured block tree from Word processing documents in the
    Office Open XML format.

Structure is declared, not inferred
-----------------------------------
    A DOCX file states its own structure. Paragraph styles name headings and
    their levels, list membership is recorded explicitly, and tables are
    real tables rather than aligned text. Nothing here needs to be guessed,
    which distinguishes this extractor sharply from the PDF one: every block
    type and heading level emitted below is something the author declared.

    Where a style name is unrecognized the paragraph becomes an ordinary
    paragraph rather than being assigned a level by appearance. A heading
    invented from font size would enter the corpus as an assertion about
    document structure that the document itself never made.

Document order
--------------
    ``python-docx`` exposes paragraphs and tables as separate collections,
    which loses their interleaving: a document of prose, table, prose would
    present as two paragraphs followed by one table. Reading order matters
    for evidence, so the body element is walked directly and each child
    dispatched by its XML tag, which preserves the sequence the author wrote.

Parser selection
----------------
    ``python-docx`` is used. It exposes the paragraph, style, and table model
    directly, which is exactly what the block tree needs, and it is mature
    and permissively licensed.

    Alternatives considered: ``mammoth``, which converts to HTML and loses
    fidelity in the process, and ``docx2python``, whose structural API is
    weaker.

    The dependency is optional, supplied by the ``ptfm[office]`` extra and
    reported through ``check_availability`` at registration rather than as an
    ``ImportError`` partway through a corpus.

Dependency contract
-------------------
    Imports the standard library, ``python-docx``, ``ptfm.core.schemas`` and
    the ingestion ``exceptions``, ``interfaces`` and ``models`` modules.
"""

from __future__ import annotations

import io
import re
import zipfile
from typing import TYPE_CHECKING, Any, Final, Optional

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

if TYPE_CHECKING:  # pragma: no cover - typing only
    from docx.document import Document
    from docx.table import Table
    from docx.text.paragraph import Paragraph

__all__ = ["DocxExtractor"]

#: Media types claimed by this extractor.
_MEDIA_TYPES: Final[frozenset[str]] = frozenset(
    {
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.template",
    }
)

#: Distribution supplying the parser, and the packaging extra providing it.
_DEPENDENCY: Final[str] = "python-docx"
_INSTALL_EXTRA: Final[str] = "ptfm[office]"

#: Namespace of the WordprocessingML body elements, used to dispatch body
#: children by tag while preserving document order.
_WORDPROCESSING_NS: Final[str] = (
    "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
)

#: Body child tags carrying content.
_PARAGRAPH_TAG: Final[str] = f"{_WORDPROCESSING_NS}p"
_TABLE_TAG: Final[str] = f"{_WORDPROCESSING_NS}tbl"

#: Style names, lowercased and stripped of digits, that denote a heading.
#: Localized documents use translated style names, so the common European
#: forms are included; an unrecognized name yields an ordinary paragraph
#: rather than a guessed level.
_HEADING_STYLE_PREFIXES: Final[tuple[str, ...]] = (
    "heading",
    "titre",
    "berschrift",
    "encabezado",
    "titolo",
    "kop",
)

#: Style names denoting a document title or subtitle. Mapped to the highest
#: outline levels, which is what they represent.
_TITLE_STYLES: Final[dict[str, int]] = {"title": 1, "subtitle": 2}

#: Style names denoting a block quotation.
_QUOTE_STYLES: Final[frozenset[str]] = frozenset(
    {"quote", "intense quote", "block text", "blockquote"}
)

#: Style names denoting preformatted or code content.
_CODE_STYLES: Final[frozenset[str]] = frozenset(
    {"code", "html preformatted", "plain text", "source code"}
)

#: Style names denoting a caption.
_CAPTION_STYLES: Final[frozenset[str]] = frozenset({"caption", "table caption"})

#: Trailing digits in a style name, which express the outline or list level.
_TRAILING_DIGITS: Final[re.Pattern[str]] = re.compile(r"(\d+)\s*$")

#: A run of whitespace, collapsed within paragraph text so that layout
#: spacing does not become document content.
_WHITESPACE_RUN: Final[re.Pattern[str]] = re.compile(r"[ \t]+")

#: Deepest outline level Word expresses.
_MAX_HEADING_LEVEL: Final[int] = 9


class DocxExtractor(ContentExtractor):
    """
    Extracts a structured block tree from Office Open XML word processing
    documents.

    Stateless: all parsing state is local to a call, so one instance serves
    all concurrent workers.
    """

    __slots__ = ()

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
            The identifier ``"DocxExtractor"``.
        """
        return "DocxExtractor"

    @property
    def supported_media_types(self) -> frozenset[str]:
        """
        Media types this extractor claims.

        Returns
        -------
        frozenset[str]
            The word processing document and template media types.
        """
        return _MEDIA_TYPES

    @classmethod
    def check_availability(cls) -> None:
        """
        Verify that the document parser is installed.

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
            import docx  # noqa: F401
        except ImportError as exc:
            raise ExtractorUnavailableError(
                media_type=(
                    "application/vnd.openxmlformats-officedocument."
                    "wordprocessingml.document"
                ),
                missing_dependency=_DEPENDENCY,
                install_extra=_INSTALL_EXTRA,
            ) from exc

    # ------------------------------------------------------------------
    # Extraction
    # ------------------------------------------------------------------

    def extract(self, payload: ExtractionInput) -> ExtractionOutput:
        """
        Parse a word processing document into a canonical block tree.

        Blocking and CPU-bound by contract. The extraction stage executes this
        in a worker thread; it must never be invoked from a coroutine.

        Parameters
        ----------
        payload:
            The raw bytes together with the resolved media type.

        Returns
        -------
        ExtractionOutput
            The block tree, core properties metadata, and any observations
            recorded while parsing.

        Raises
        ------
        ExtractorUnavailableError
            If the parser is not installed.
        EncryptedDocumentError
            If the document is password protected.
        CorruptDocumentError
            If the package is malformed and cannot be opened.
        ExtractorInternalError
            If the parser raised an unanticipated exception, or the assembled
            result failed its own validation.
        """
        self.check_availability()
        import docx

        warnings: list[ExtractionWarning] = []
        stream = io.BytesIO(payload.data)

        try:
            document = docx.Document(stream)
        except zipfile.BadZipFile as exc:
            # An encrypted OOXML file is an OLE compound document rather than
            # a ZIP package, so it presents as a malformed archive. The two
            # are distinguished by inspecting the payload signature, since the
            # remedy differs entirely: supply a password, or discard the file.
            if _is_ole_container(payload.data):
                raise EncryptedDocumentError(
                    f"Document {payload.source_uri!r} appears to be encrypted "
                    f"or password protected.",
                    source_uri=payload.source_uri,
                    context={"extractor_name": self.name},
                ) from exc
            raise CorruptDocumentError(
                f"Document {payload.source_uri!r} is not a readable package: "
                f"{exc}",
                source_uri=payload.source_uri,
                context={"extractor_name": self.name},
            ) from exc
        except KeyError as exc:
            # The parser raises KeyError when a required part is absent from
            # the package, which is a malformed document rather than a defect.
            raise CorruptDocumentError(
                f"Document {payload.source_uri!r} is missing a required "
                f"package part: {exc}",
                source_uri=payload.source_uri,
                context={"extractor_name": self.name},
            ) from exc
        except Exception as exc:
            raise ExtractorInternalError(
                f"Document parser raised {type(exc).__name__} while opening "
                f"{payload.source_uri!r}: {exc}",
                extractor_name=self.name,
                source_uri=payload.source_uri,
            ) from exc

        try:
            blocks = _BodyTranslator(document, warnings).translate()
            metadata = _harvest_metadata(document, warnings)
        except RecursionError as exc:
            raise CorruptDocumentError(
                f"Document {payload.source_uri!r} is nested too deeply to "
                f"traverse.",
                source_uri=payload.source_uri,
                context={"extractor_name": self.name},
            ) from exc
        except Exception as exc:
            raise ExtractorInternalError(
                f"Document parser raised {type(exc).__name__} while reading "
                f"{payload.source_uri!r}: {exc}",
                extractor_name=self.name,
                source_uri=payload.source_uri,
            ) from exc

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


# ============================================================================
# BODY TRANSLATION
# ============================================================================


class _BodyTranslator:
    """
    Converts a document body into a flat parent-indexed block list.

    The body element is walked directly rather than through the parser's
    paragraph and table collections, because those are exposed separately and
    lose the interleaving between them. A document of prose, table, prose
    would otherwise present as two paragraphs followed by one table, which
    misrepresents reading order and therefore misplaces evidence.

    One instance per document, since it accumulates per-document state.

    Parameters
    ----------
    document:
        The opened document.
    warnings:
        Accumulator to which observations are appended.
    """

    __slots__ = ("_document", "_warnings", "_blocks", "_list_stack", "_reported_styles")

    def __init__(self, document: Document, warnings: list[ExtractionWarning]) -> None:
        self._document = document
        self._warnings = warnings
        self._blocks: list[ExtractedBlock] = []
        # Open list containers, innermost last, each paired with the level it
        # represents. Word expresses nesting as a level on each paragraph
        # rather than as containment, so the tree is reconstructed from the
        # sequence of levels observed.
        self._list_stack: list[tuple[int, int]] = []
        self._reported_styles: set[str] = set()

    def translate(self) -> tuple[ExtractedBlock, ...]:
        """
        Convert the document body into canonical blocks.

        Returns
        -------
        tuple[ExtractedBlock, ...]
            Blocks in depth-first pre-order.
        """
        from docx.table import Table
        from docx.text.paragraph import Paragraph

        body = self._document.element.body

        for child in body.iterchildren():
            if child.tag == _PARAGRAPH_TAG:
                self._translate_paragraph(Paragraph(child, self._document))
            elif child.tag == _TABLE_TAG:
                self._close_lists(0)
                self._translate_table(Table(child, self._document))
            # Section properties and other body children carry layout rather
            # than content and contribute no block.

        self._close_lists(0)
        return tuple(self._blocks)

    # ------------------------------------------------------------------
    # Paragraphs
    # ------------------------------------------------------------------

    def _translate_paragraph(self, paragraph: Paragraph) -> None:
        """
        Translate one paragraph according to its declared style.

        Parameters
        ----------
        paragraph:
            The paragraph to translate.

        Returns
        -------
        None
        """
        text = _paragraph_text(paragraph)
        style = _style_name(paragraph)

        list_level = _list_level(paragraph)
        if list_level is not None:
            self._translate_list_item(text, list_level)
            return

        self._close_lists(0)

        if not text.strip():
            # A blank paragraph is Word's usual means of vertical spacing. It
            # carries no content, and emitting a block for it would add a zero
            # length span describing layout.
            return

        heading_level = _heading_level(style)
        if heading_level is not None:
            self._append_leaf(
                BlockType.HEADING,
                text,
                attributes=BlockAttributes(heading_level=heading_level),
            )
            return

        block_type = _styled_block_type(style)
        if block_type is BlockType.QUOTE:
            # QUOTE is a container in the canonical model, so its text lives
            # in a leaf child.
            index = self._append_container(BlockType.QUOTE, BlockAttributes())
            self._append_leaf(BlockType.PARAGRAPH, text, parent=index)
            return

        if block_type is not None:
            self._append_leaf(block_type, text)
            return

        if style:
            self._record_unrecognized_style(style)
        self._append_leaf(BlockType.PARAGRAPH, text)

    def _translate_list_item(self, text: str, level: int) -> None:
        """
        Translate a paragraph that belongs to a list.

        Word records a level on each list paragraph rather than nesting them,
        so containment is reconstructed here from the sequence of levels
        observed: a deeper level opens containers, a shallower one closes
        them.

        Parameters
        ----------
        text:
            The paragraph's text.
        level:
            The declared list level, zero for an outermost item.

        Returns
        -------
        None
        """
        self._close_lists(level + 1)

        while len(self._list_stack) <= level:
            depth = len(self._list_stack)
            parent = self._list_stack[-1][1] if self._list_stack else None
            index = self._append_container(
                BlockType.LIST,
                BlockAttributes(list_level=depth),
                parent=parent,
            )
            self._list_stack.append((depth, index))

        list_index = self._list_stack[level][1]
        item_index = self._append_container(
            BlockType.LIST_ITEM,
            BlockAttributes(list_level=level),
            parent=list_index,
        )
        # An empty list item is legitimate, and LIST_ITEM is a container, so a
        # leaf child is emitted regardless of whether text is present.
        self._append_leaf(BlockType.PARAGRAPH, text, parent=item_index, allow_empty=True)

    def _close_lists(self, depth: int) -> None:
        """
        Close open list containers deeper than a level.

        Parameters
        ----------
        depth:
            Number of list containers that should remain open.

        Returns
        -------
        None
        """
        while len(self._list_stack) > depth:
            self._list_stack.pop()

    # ------------------------------------------------------------------
    # Tables
    # ------------------------------------------------------------------

    def _translate_table(self, table: Table) -> None:
        """
        Translate one table into a row and cell tree.

        Cell positions are recorded so that a value remains citable by row and
        column. Merged cells are reported rather than reconstructed: the
        parser presents a merged cell once per grid position it occupies, and
        reconstructing the logical grid would require span information the
        model cannot yet carry, so a warning is recorded and duplicated
        content is not emitted twice.

        Parameters
        ----------
        table:
            The table to translate.

        Returns
        -------
        None
        """
        table_index = self._append_container(BlockType.TABLE, BlockAttributes())
        merged_observed = False

        for row_index, row in enumerate(table.rows):
            row_index_block = self._append_container(
                BlockType.TABLE_ROW,
                BlockAttributes(row_index=row_index),
                parent=table_index,
            )

            seen_cells: set[int] = set()
            for column_index, cell in enumerate(row.cells):
                identity = id(cell._tc)
                if identity in seen_cells:
                    # The same underlying cell repeated across grid positions,
                    # which is how the parser presents a horizontal merge.
                    merged_observed = True
                    continue
                seen_cells.add(identity)

                cell_index = self._append_container(
                    BlockType.TABLE_CELL,
                    BlockAttributes(
                        row_index=row_index, column_index=column_index
                    ),
                    parent=row_index_block,
                )
                self._append_leaf(
                    BlockType.PARAGRAPH,
                    _cell_text(cell),
                    parent=cell_index,
                    allow_empty=True,
                )

        if merged_observed:
            self._warnings.append(
                ExtractionWarning(
                    code=ExtractionWarningCode.TABLE_STRUCTURE_UNCERTAIN,
                    message=(
                        "Table contains merged cells; each was recorded once "
                        "at its first grid position, so column positions may "
                        "not align with the rendered layout"
                    ),
                    detail={"table_block_index": table_index},
                )
            )

    # ------------------------------------------------------------------
    # Block construction
    # ------------------------------------------------------------------

    def _append_leaf(
        self,
        block_type: BlockType,
        text: str,
        *,
        parent: Optional[int] = None,
        attributes: Optional[BlockAttributes] = None,
        allow_empty: bool = False,
    ) -> None:
        """
        Append a content-bearing block.

        Parameters
        ----------
        block_type:
            Canonical block type. Must be a leaf type.
        text:
            The block's text.
        parent:
            Index of the enclosing container block, or ``None`` at the top
            level.
        attributes:
            Optional structural attributes.
        allow_empty:
            Whether an empty block is emitted. Required where the block
            satisfies a container's need for a child, since a container
            holding no children could not be assigned a span.

        Returns
        -------
        None
        """
        if not allow_empty and not text.strip():
            return

        self._blocks.append(
            ExtractedBlock(
                block_type=block_type,
                text=text,
                parent_index=parent,
                attributes=attributes or BlockAttributes(),
            )
        )

    def _append_container(
        self,
        block_type: BlockType,
        attributes: BlockAttributes,
        *,
        parent: Optional[int] = None,
    ) -> int:
        """
        Append a container block.

        Parameters
        ----------
        block_type:
            Canonical container block type.
        attributes:
            Structural attributes.
        parent:
            Index of the enclosing container block.

        Returns
        -------
        int
            Index of the appended container.
        """
        index = len(self._blocks)
        self._blocks.append(
            ExtractedBlock(
                block_type=block_type,
                text=None,
                parent_index=parent,
                attributes=attributes,
            )
        )
        return index

    def _record_unrecognized_style(self, style: str) -> None:
        """
        Record that a style name yielded no specific block type.

        Recorded once per style name rather than once per paragraph, so that a
        document using one custom style throughout does not flood the warning
        list. The paragraph is still extracted as ordinary body text; nothing
        is lost, and no structure is claimed that the style did not declare.

        Parameters
        ----------
        style:
            The paragraph's style name.

        Returns
        -------
        None
        """
        normalized = style.strip().lower()
        if not normalized or normalized in {"normal", "body text", "default"}:
            return
        if normalized in self._reported_styles:
            return
        self._reported_styles.add(normalized)

        self._warnings.append(
            ExtractionWarning(
                code=ExtractionWarningCode.STRUCTURE_INFERRED,
                message=(
                    f"Paragraph style {style!r} has no canonical block type; "
                    f"affected paragraphs were extracted as body text"
                ),
                detail={"style_name": style},
            )
        )


# ============================================================================
# STYLE INTERPRETATION
# ============================================================================


def _style_name(paragraph: Paragraph) -> str:
    """
    Recover a paragraph's style name.

    Parameters
    ----------
    paragraph:
        The paragraph to inspect.

    Returns
    -------
    str
        The style name, or an empty string when no style is resolvable.
    """
    style = getattr(paragraph, "style", None)
    if style is None:
        return ""
    return (getattr(style, "name", "") or "").strip()


def _heading_level(style: str) -> Optional[int]:
    """
    Derive an outline level from a paragraph style name.

    Only names the document itself declares as headings yield a level. A
    level is never derived from font size or weight: a heading invented from
    appearance would enter the corpus as an assertion about structure that the
    author never made.

    Localized style names are recognized for the common European languages,
    since a document authored in French declares its headings as ``Titre 1``
    rather than ``Heading 1``.

    Parameters
    ----------
    style:
        The paragraph's style name.

    Returns
    -------
    Optional[int]
        The outline level, or ``None`` when the style is not a heading.
    """
    if not style:
        return None

    normalized = style.strip().lower()

    title_level = _TITLE_STYLES.get(normalized)
    if title_level is not None:
        return title_level

    stem = _TRAILING_DIGITS.sub("", normalized).strip()
    if not any(stem.startswith(prefix) for prefix in _HEADING_STYLE_PREFIXES):
        return None

    match = _TRAILING_DIGITS.search(normalized)
    if match is None:
        # A heading style carrying no level, which Word permits, is the
        # highest level the document expresses.
        return 1

    level = int(match.group(1))
    return level if 1 <= level <= _MAX_HEADING_LEVEL else None


def _styled_block_type(style: str) -> Optional[BlockType]:
    """
    Map a paragraph style name to a canonical block type.

    Parameters
    ----------
    style:
        The paragraph's style name.

    Returns
    -------
    Optional[BlockType]
        The block type the style declares, or ``None`` when the style implies
        ordinary body text or is unrecognized.
    """
    normalized = style.strip().lower()
    if not normalized:
        return None
    if normalized in _QUOTE_STYLES:
        return BlockType.QUOTE
    if normalized in _CODE_STYLES:
        return BlockType.CODE
    if normalized in _CAPTION_STYLES:
        return BlockType.CAPTION
    return None


def _list_level(paragraph: Paragraph) -> Optional[int]:
    """
    Recover a paragraph's list level, when it belongs to a list.

    Word records list membership as numbering properties on the paragraph
    rather than by containment, so membership and depth are read from those
    properties directly.

    Parameters
    ----------
    paragraph:
        The paragraph to inspect.

    Returns
    -------
    Optional[int]
        The list level, zero for an outermost item, or ``None`` when the
        paragraph is not part of a list.
    """
    properties = getattr(paragraph._p, "pPr", None)
    if properties is None:
        return None

    numbering = getattr(properties, "numPr", None)
    if numbering is None:
        return None

    level_element = getattr(numbering, "ilvl", None)
    if level_element is None:
        return 0

    value = getattr(level_element, "val", None)
    if value is None:
        return 0
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0


# ============================================================================
# TEXT AND METADATA
# ============================================================================


def _paragraph_text(paragraph: Paragraph) -> str:
    """
    Recover a paragraph's readable text.

    Horizontal whitespace runs are collapsed, so that layout spacing does not
    become document content. Line breaks within the paragraph are preserved,
    since they are content the author inserted deliberately.

    Parameters
    ----------
    paragraph:
        The paragraph whose text is required.

    Returns
    -------
    str
        The readable text.
    """
    return _WHITESPACE_RUN.sub(" ", paragraph.text or "").strip()


def _cell_text(cell: Any) -> str:
    """
    Recover a table cell's readable text.

    A cell holds paragraphs, whose texts are joined by line feeds so that a
    multi-paragraph cell remains legible as a single value.

    Parameters
    ----------
    cell:
        The cell whose text is required.

    Returns
    -------
    str
        The readable text.
    """
    parts = [
        _WHITESPACE_RUN.sub(" ", paragraph.text or "").strip()
        for paragraph in cell.paragraphs
    ]
    return "\n".join(part for part in parts if part)


def _harvest_metadata(
    document: Document,
    warnings: list[ExtractionWarning],
) -> ExtractedMetadata:
    """
    Recover the document's core properties.

    Office Open XML specifies core property timestamps as UTC, so they are
    passed through as the parser supplies them. The metadata stage coerces
    naive instants to UTC, which is the correct interpretation for this
    format.

    Parameters
    ----------
    document:
        The opened document.
    warnings:
        Accumulator to which an observation is appended when properties are
        unavailable.

    Returns
    -------
    ExtractedMetadata
        Format-native metadata.
    """
    try:
        properties = document.core_properties
    except Exception as exc:
        # A package lacking a core properties part is unusual but valid, and
        # its content is still fully extractable.
        warnings.append(
            ExtractionWarning(
                code=ExtractionWarningCode.METADATA_UNAVAILABLE,
                message=(
                    f"Core properties could not be read "
                    f"({type(exc).__name__}); document content was extracted "
                    f"regardless"
                ),
                detail={"exception_type": type(exc).__name__},
            )
        )
        return ExtractedMetadata()

    authors = tuple(
        part.strip()
        for part in (properties.author or "").split(";")
        if part.strip()
    )

    extra: dict[str, str] = {}
    for key, value in (
        ("subject", properties.subject),
        ("keywords", properties.keywords),
        ("category", properties.category),
        ("last_modified_by", properties.last_modified_by),
        ("revision", properties.revision),
    ):
        if value:
            extra[key] = str(value).strip()

    return ExtractedMetadata(
        title=(properties.title or "").strip() or None,
        authors=authors,
        created_at=properties.created,
        modified_at=properties.modified,
        declared_language=(properties.language or "").strip() or None,
        # Page count is deliberately absent. A DOCX has no fixed pagination:
        # the page count recorded in a package is whatever the last renderer
        # produced, and reporting it would state as fact something that
        # depends on the reader's fonts and page size.
        page_count=None,
        properties=extra,
    )


def _is_ole_container(data: bytes) -> bool:
    """
    Report whether a payload is an OLE compound document.

    An encrypted Office Open XML file is wrapped in an OLE container rather
    than a ZIP package, so it presents as a malformed archive. Distinguishing
    the two matters because the remedies differ entirely: an encrypted file
    needs a password, whereas a corrupt one needs replacing.

    Parameters
    ----------
    data:
        The payload bytes.

    Returns
    -------
    bool
        ``True`` when the payload carries the OLE compound document
        signature.
    """
    return data[:8] == b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"
