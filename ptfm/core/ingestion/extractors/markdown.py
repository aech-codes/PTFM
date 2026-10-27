"""
PTFM Markdown Extractor

File:
    ptfm/core/ingestion/extractors/markdown.py

Description:
    Extracts a structured block tree from Markdown payloads.

Why a parser rather than segmentation
-------------------------------------
    Markdown could be segmented like plain text, but the result would be poor
    in a way that matters downstream. A link written as ``[the ruling](url)``
    would enter the corpus with its target and brackets intact, so retrieval
    would index punctuation and URLs alongside prose, and a citation quoting
    the passage would quote markup. Headings, list nesting, and code fences
    would likewise be lost, discarding structure the author supplied
    explicitly.

    A real parse recovers the author's structure and yields the text a reader
    would actually see.

Parser selection
----------------
    ``markdown-it-py`` is used. It is CommonMark compliant, pure Python, MIT
    licensed, actively maintained, and exposes a flat token stream whose
    open and close markers map directly onto the parent-index block model,
    so no recursive traversal is required.

    Alternatives considered: ``mistune``, which is faster but less strict
    about CommonMark edge cases such as lazy continuation, and
    ``commonmark.py``, which is unmaintained.

    The dependency is optional, supplied by the ``ptfm[markdown]`` extra and
    reported through ``check_availability`` at registration rather than as an
    ``ImportError`` partway through a corpus. Where it is absent, registering
    ``PlainTextExtractor`` for ``text/*`` at a negative priority keeps
    Markdown documents ingestible as unstructured text.

Container blocks always receive a child
---------------------------------------
    The canonical model rejects a container block with no children, since such
    a block would occupy no text and could not be assigned a span. Markdown
    readily produces empty containers: a table cell left blank, a list item
    with no body. Every container closed without a child is therefore given an
    empty paragraph, preserving the table or list shape without asserting
    content that is not there.

Thread safety
-------------
    A parser is constructed per call. The extractor contract permits
    concurrent invocation from several worker threads, and constructing the
    parser afresh removes any question of shared parser state. The cost is
    negligible beside parsing the document itself.

Dependency contract
-------------------
    Imports the standard library, ``markdown-it-py``, ``ptfm.core.schemas``
    and the ingestion ``exceptions``, ``interfaces``, ``models`` and
    ``extractors.plaintext`` modules. Decoding is shared with the plain text
    extractor rather than reimplemented.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Final, Optional

from pydantic import ValidationError

from ptfm.core.ingestion.exceptions import (
    CorruptDocumentError,
    ExtractorInternalError,
    ExtractorUnavailableError,
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

if TYPE_CHECKING:  # pragma: no cover - typing only
    from markdown_it.token import Token

__all__ = ["MarkdownExtractor"]

#: Media types claimed by this extractor.
_MEDIA_TYPES: Final[frozenset[str]] = frozenset({"text/markdown"})

#: Distribution supplying the parser, and the packaging extra that provides
#: it. Reported when the dependency is absent so that the remedy is stated
#: rather than left to be inferred.
_DEPENDENCY: Final[str] = "markdown-it-py"
_INSTALL_EXTRA: Final[str] = "ptfm[markdown]"

#: Syntax extensions enabled beyond strict CommonMark. Tables are in
#: widespread use in documentation corpora and carry real structure;
#: strikethrough is enabled so that struck text is recovered as text rather
#: than as tildes.
_ENABLED_RULES: Final[tuple[str, ...]] = ("table", "strikethrough")

#: Delimiter opening and closing a YAML front matter block.
_FRONT_MATTER_DELIMITER: Final[str] = "---"

#: A simple scalar assignment within front matter. Deliberately not a YAML
#: grammar; see :func:`_split_front_matter`.
_FRONT_MATTER_SCALAR: Final[re.Pattern[str]] = re.compile(
    r"^(?P<key>[A-Za-z0-9_-]+)\s*:\s*(?P<value>.*?)\s*$"
)

#: Front matter keys promoted to typed metadata fields rather than retained
#: as opaque properties.
_TITLE_KEYS: Final[frozenset[str]] = frozenset({"title"})
_AUTHOR_KEYS: Final[frozenset[str]] = frozenset({"author", "authors"})

#: Separator by which a front matter author list is split. Only a flat inline
#: list is recognized; anything structured is left as an opaque property.
_AUTHOR_SEPARATOR: Final[str] = ","

#: Inline token types whose content is document text. Emphasis, link, and
#: strikethrough markers are absent because their text arrives as separate
#: text tokens at the same level, so skipping the markers yields the text a
#: reader would see without the syntax that produced it.
_TEXT_TOKEN_TYPES: Final[frozenset[str]] = frozenset({"text", "code_inline"})

#: Inline token types rendered as a line break within a block.
_BREAK_TOKEN_TYPES: Final[frozenset[str]] = frozenset({"softbreak", "hardbreak"})

#: Block token types that carry structure but no content of their own, and
#: are therefore not represented as blocks. Emitting them would add tree
#: depth without adding text.
_TRANSPARENT_TOKEN_TYPES: Final[frozenset[str]] = frozenset(
    {"thead_open", "thead_close", "tbody_open", "tbody_close", "hr"}
)

#: Block container token types mapped to canonical container blocks.
_CONTAINER_TOKENS: Final[dict[str, BlockType]] = {
    "bullet_list_open": BlockType.LIST,
    "ordered_list_open": BlockType.LIST,
    "list_item_open": BlockType.LIST_ITEM,
    "blockquote_open": BlockType.QUOTE,
    "table_open": BlockType.TABLE,
    "tr_open": BlockType.TABLE_ROW,
    "th_open": BlockType.TABLE_CELL,
    "td_open": BlockType.TABLE_CELL,
}

#: Closing token for each container opening token.
_CONTAINER_CLOSERS: Final[frozenset[str]] = frozenset(
    {
        "bullet_list_close",
        "ordered_list_close",
        "list_item_close",
        "blockquote_close",
        "table_close",
        "tr_close",
        "th_close",
        "td_close",
    }
)


class MarkdownExtractor(ContentExtractor):
    """
    Extracts a structured block tree from Markdown payloads.

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
            The identifier ``"MarkdownExtractor"``.
        """
        return "MarkdownExtractor"

    @property
    def supported_media_types(self) -> frozenset[str]:
        """
        Media types this extractor claims.

        Returns
        -------
        frozenset[str]
            The Markdown media type.
        """
        return _MEDIA_TYPES

    @classmethod
    def check_availability(cls) -> None:
        """
        Verify that the Markdown parser is installed.

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
            import markdown_it  # noqa: F401
        except ImportError as exc:
            raise ExtractorUnavailableError(
                media_type="text/markdown",
                missing_dependency=_DEPENDENCY,
                install_extra=_INSTALL_EXTRA,
            ) from exc

    # ------------------------------------------------------------------
    # Extraction
    # ------------------------------------------------------------------

    def extract(self, payload: ExtractionInput) -> ExtractionOutput:
        """
        Parse a Markdown payload into a canonical block tree.

        Blocking and CPU-bound by contract. The extraction stage executes this
        in a worker thread; it must never be invoked from a coroutine.

        Parameters
        ----------
        payload:
            The raw bytes together with the resolved media type and encoding.

        Returns
        -------
        ExtractionOutput
            The block tree, front matter metadata, and any observations
            recorded while parsing.

        Raises
        ------
        ExtractorUnavailableError
            If the parser is not installed.
        CorruptDocumentError
            If the parser rejected the payload.
        ExtractorInternalError
            If the resolved encoding is unrecognized, the parser raised an
            unanticipated exception, or the assembled result failed its own
            validation.
        """
        self.check_availability()
        from markdown_it import MarkdownIt

        warnings: list[ExtractionWarning] = []
        text = decode_payload(payload, warnings)
        body, front_matter = _split_front_matter(text, warnings)

        # Constructed per call. Concurrent invocation is permitted by the
        # extractor contract, and a fresh parser removes any question of
        # shared state at a cost negligible beside parsing.
        parser = MarkdownIt().enable(list(_ENABLED_RULES))

        try:
            tokens = parser.parse(body)
        except RecursionError as exc:
            # Deeply nested structures can exhaust the stack. That is a
            # property of the document, not a defect in the parser.
            raise CorruptDocumentError(
                f"Markdown payload {payload.source_uri!r} is nested too "
                f"deeply to parse.",
                source_uri=payload.source_uri,
                context={"extractor_name": self.name},
            ) from exc
        except Exception as exc:
            raise ExtractorInternalError(
                f"Markdown parser raised {type(exc).__name__} while parsing "
                f"{payload.source_uri!r}: {exc}",
                extractor_name=self.name,
                source_uri=payload.source_uri,
            ) from exc

        translator = _TokenTranslator(warnings)
        blocks = translator.translate(tokens)

        try:
            return ExtractionOutput(
                blocks=blocks,
                metadata=_build_metadata(front_matter, translator.first_heading),
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
# TOKEN TRANSLATION
# ============================================================================


class _TokenTranslator:
    """
    Converts a Markdown token stream into a flat parent-indexed block list.

    The token stream is already flat, with explicit open and close markers, so
    translation is a single pass over a stack of open containers. No recursion
    is involved, and the resulting depth-first pre-order is exactly what the
    canonical model requires.

    One instance per document, since it accumulates per-document state.

    Parameters
    ----------
    warnings:
        Accumulator to which observations about skipped content are appended.
    """

    __slots__ = ("_warnings", "_blocks", "_stack", "_child_counts", "_first_heading")

    def __init__(self, warnings: list[ExtractionWarning]) -> None:
        self._warnings = warnings
        self._blocks: list[ExtractedBlock] = []
        self._stack: list[int] = []
        self._child_counts: dict[int, int] = {}
        self._first_heading: Optional[str] = None

    @property
    def first_heading(self) -> Optional[str]:
        """
        Text of the document's first heading, when one was present.

        Returns
        -------
        Optional[str]
            The heading text, or ``None`` if the document has no headings.
        """
        return self._first_heading

    def translate(self, tokens: list[Token]) -> tuple[ExtractedBlock, ...]:
        """
        Convert a token stream into canonical blocks.

        Parameters
        ----------
        tokens:
            The parser's block-level token stream.

        Returns
        -------
        tuple[ExtractedBlock, ...]
            Blocks in depth-first pre-order.
        """
        index = 0
        while index < len(tokens):
            index = self._consume(tokens, index)

        # A malformed stream could leave containers open. Closing them here
        # guarantees every container receives its required child rather than
        # producing a result the canonical model would reject.
        while self._stack:
            self._close_container()

        return tuple(self._blocks)

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------

    def _consume(self, tokens: list[Token], index: int) -> int:
        """
        Translate the token at a position, and any tokens it absorbs.

        Parameters
        ----------
        tokens:
            The token stream.
        index:
            Position of the token to translate.

        Returns
        -------
        int
            Position of the next token to translate.
        """
        token = tokens[index]
        token_type = token.type

        if token_type in _TRANSPARENT_TOKEN_TYPES:
            return index + 1

        if token_type == "heading_open":
            return self._consume_heading(tokens, index)

        if token_type == "paragraph_open":
            return self._consume_paragraph(tokens, index)

        if token_type in ("fence", "code_block"):
            self._append_leaf(BlockType.CODE, token.content)
            return index + 1

        if token_type in _CONTAINER_TOKENS:
            self._open_container(_CONTAINER_TOKENS[token_type], token)
            return index + 1

        if token_type in _CONTAINER_CLOSERS:
            self._close_container()
            return index + 1

        if token_type == "html_block":
            # Raw HTML embedded in Markdown is not parsed here. Emitting it as
            # text would place markup into the corpus, where retrieval would
            # index tag names as prose.
            self._warn_skipped("html_block")
            return index + 1

        if token_type == "inline":
            # An inline token outside a heading or paragraph is unusual but
            # carries real text, so it is preserved rather than discarded.
            self._append_leaf(BlockType.PARAGRAPH, _inline_text(token))
            return index + 1

        self._warn_skipped(token_type)
        return index + 1

    def _consume_heading(self, tokens: list[Token], index: int) -> int:
        """
        Translate a heading and its inline content.

        Parameters
        ----------
        tokens:
            The token stream.
        index:
            Position of the ``heading_open`` token.

        Returns
        -------
        int
            Position following the ``heading_close`` token.
        """
        open_token = tokens[index]
        text, next_index = _absorb_inline(tokens, index + 1, "heading_close")

        if self._first_heading is None and text.strip():
            self._first_heading = text.strip()

        self._append_leaf(
            BlockType.HEADING,
            text,
            attributes=BlockAttributes(heading_level=_heading_level(open_token)),
        )
        return next_index

    def _consume_paragraph(self, tokens: list[Token], index: int) -> int:
        """
        Translate a paragraph and its inline content.

        Tight list items produce paragraph tokens marked hidden, meaning they
        would not be rendered as paragraph elements. They are still emitted
        here, because a list item is a container and requires a leaf child to
        hold its text.

        Parameters
        ----------
        tokens:
            The token stream.
        index:
            Position of the ``paragraph_open`` token.

        Returns
        -------
        int
            Position following the ``paragraph_close`` token.
        """
        text, next_index = _absorb_inline(tokens, index + 1, "paragraph_close")
        self._append_leaf(BlockType.PARAGRAPH, text)
        return next_index

    # ------------------------------------------------------------------
    # Block construction
    # ------------------------------------------------------------------

    def _append_leaf(
        self,
        block_type: BlockType,
        text: str,
        *,
        attributes: Optional[BlockAttributes] = None,
    ) -> None:
        """
        Append a content-bearing block beneath the innermost open container.

        Parameters
        ----------
        block_type:
            Canonical block type. Must be a leaf type.
        text:
            The block's text.
        attributes:
            Optional structural attributes.

        Returns
        -------
        None
        """
        parent = self._stack[-1] if self._stack else None
        if parent is not None:
            self._child_counts[parent] = self._child_counts.get(parent, 0) + 1

        self._blocks.append(
            ExtractedBlock(
                block_type=block_type,
                text=text,
                parent_index=parent,
                attributes=attributes or BlockAttributes(),
            )
        )

    def _open_container(self, block_type: BlockType, token: Token) -> None:
        """
        Open a container block beneath the innermost open container.

        Parameters
        ----------
        block_type:
            Canonical container block type.
        token:
            The opening token, consulted for structural attributes.

        Returns
        -------
        None
        """
        parent = self._stack[-1] if self._stack else None
        if parent is not None:
            self._child_counts[parent] = self._child_counts.get(parent, 0) + 1

        index = len(self._blocks)
        self._blocks.append(
            ExtractedBlock(
                block_type=block_type,
                text=None,
                parent_index=parent,
                attributes=self._container_attributes(block_type, token),
            )
        )
        self._stack.append(index)

    def _close_container(self) -> None:
        """
        Close the innermost open container, ensuring it holds a child.

        A container with no children cannot be assigned a span and is rejected
        by the canonical model. Markdown produces such containers readily, as
        with a blank table cell, so an empty paragraph is supplied. The
        table's shape is preserved without asserting content that is absent.

        Returns
        -------
        None
        """
        index = self._stack.pop()
        if self._child_counts.get(index, 0) > 0:
            return

        self._child_counts[index] = 1
        self._blocks.append(
            ExtractedBlock(
                block_type=BlockType.PARAGRAPH,
                text="",
                parent_index=index,
            )
        )

    def _container_attributes(
        self,
        block_type: BlockType,
        token: Token,
    ) -> BlockAttributes:
        """
        Derive structural attributes for a container block.

        Parameters
        ----------
        block_type:
            The container's canonical type.
        token:
            The opening token.

        Returns
        -------
        BlockAttributes
            Attributes appropriate to the container type.
        """
        if block_type is BlockType.LIST:
            return BlockAttributes(list_level=self._list_depth())
        if block_type is BlockType.LIST_ITEM:
            return BlockAttributes(list_level=max(0, self._list_depth() - 1))
        if block_type is BlockType.TABLE_ROW:
            return BlockAttributes(
                row_index=self._sibling_position(),
                is_header=token.type == "tr_open" and self._in_header_row(token),
            )
        if block_type is BlockType.TABLE_CELL:
            return BlockAttributes(
                column_index=self._sibling_position(),
                is_header=token.type == "th_open",
            )
        return BlockAttributes()

    def _list_depth(self) -> int:
        """
        Count the list containers currently open.

        Returns
        -------
        int
            Nesting depth, zero for an outermost list.
        """
        return sum(
            1
            for index in self._stack
            if self._blocks[index].block_type is BlockType.LIST
        )

    def _sibling_position(self) -> int:
        """
        Position the next child will occupy within its parent.

        Returns
        -------
        int
            Zero-based sibling index.
        """
        if not self._stack:
            return 0
        return self._child_counts.get(self._stack[-1], 0)

    def _in_header_row(self, token: Token) -> bool:
        """
        Report whether a row belongs to the table header.

        The parser marks header cells rather than header rows, so a row is
        treated as a header row only when the table has yet to produce any
        row, which is the position a Markdown table header always occupies.

        Parameters
        ----------
        token:
            The row's opening token.

        Returns
        -------
        bool
            ``True`` when this is the table's first row.
        """
        del token
        return self._sibling_position() == 0

    def _warn_skipped(self, token_type: str) -> None:
        """
        Record that a token contributed no block.

        Recorded once per token type rather than once per occurrence, so that
        a document containing many of the same element does not flood the
        warning list.

        Parameters
        ----------
        token_type:
            The parser's name for the skipped token.

        Returns
        -------
        None
        """
        already_recorded = any(
            warning.detail.get("token_type") == token_type
            for warning in self._warnings
            if warning.code is ExtractionWarningCode.UNSUPPORTED_ELEMENT_SKIPPED
        )
        if already_recorded:
            return

        self._warnings.append(
            ExtractionWarning(
                code=ExtractionWarningCode.UNSUPPORTED_ELEMENT_SKIPPED,
                message=(
                    f"Markdown element of type {token_type!r} contributed no "
                    f"block and was skipped"
                ),
                detail={"token_type": token_type},
            )
        )


# ============================================================================
# INLINE TEXT
# ============================================================================


def _absorb_inline(
    tokens: list[Token],
    index: int,
    closing_type: str,
) -> tuple[str, int]:
    """
    Collect inline text up to a closing token.

    Parameters
    ----------
    tokens:
        The token stream.
    index:
        Position at which to begin collecting.
    closing_type:
        Token type that terminates collection.

    Returns
    -------
    tuple[str, int]
        The collected text, and the position following the closing token. When
        the closing token is absent, which a well-formed stream never
        produces, collection stops at the end of the stream.
    """
    parts: list[str] = []

    while index < len(tokens):
        token = tokens[index]
        if token.type == closing_type:
            return "".join(parts), index + 1
        if token.type == "inline":
            parts.append(_inline_text(token))
        index += 1

    return "".join(parts), index


def _inline_text(token: Token) -> str:
    """
    Recover the readable text of an inline token.

    Emphasis, link, and strikethrough markers are dropped while their text is
    retained, so a link renders as its label rather than as its label followed
    by a URL in brackets. Image alternative text is retained, since it is the
    only textual description the document offers of the image.

    Parameters
    ----------
    token:
        An ``inline`` token.

    Returns
    -------
    str
        The readable text.
    """
    children = getattr(token, "children", None)
    if not children:
        return token.content

    parts: list[str] = []
    for child in children:
        if child.type in _TEXT_TOKEN_TYPES:
            parts.append(child.content)
        elif child.type in _BREAK_TOKEN_TYPES:
            parts.append("\n")
        elif child.type == "image":
            # ``content`` holds the alternative text, which is the only
            # description of the image the document provides.
            parts.append(child.content)

    return "".join(parts)


def _heading_level(token: Token) -> Optional[int]:
    """
    Derive an outline level from a heading token.

    Parameters
    ----------
    token:
        A ``heading_open`` token, whose tag names the level.

    Returns
    -------
    Optional[int]
        The level, or ``None`` when the tag is not a recognized heading tag.
    """
    tag = getattr(token, "tag", "")
    if len(tag) == 2 and tag[0] == "h" and tag[1].isdigit():
        return int(tag[1])
    return None


# ============================================================================
# FRONT MATTER
# ============================================================================


def _split_front_matter(
    text: str,
    warnings: list[ExtractionWarning],
) -> tuple[str, dict[str, str]]:
    """
    Separate a leading YAML front matter block from the document body.

    Front matter is ubiquitous in documentation corpora. Left in place it does
    not parse as Markdown at all: the opening delimiter becomes a thematic
    break and the keys become paragraph text, so every such document would
    begin with several blocks of metadata masquerading as prose.

    Only simple scalar assignments are recovered. This is deliberately NOT a
    YAML parser: nested mappings, block scalars, and anchors are ignored
    rather than approximated, because a wrong value recorded as metadata is
    worse than no value. Recovering the common flat case removes the noise
    that matters while promising nothing further.

    Parameters
    ----------
    text:
        The decoded document text.
    warnings:
        Accumulator to which a malformed-delimiter observation is appended.

    Returns
    -------
    tuple[str, dict[str, str]]
        The document body with front matter removed, and the scalar keys
        recovered from it.
    """
    if not text.startswith(f"{_FRONT_MATTER_DELIMITER}\n"):
        return text, {}

    lines = text.split("\n")
    closing: Optional[int] = None
    for position in range(1, len(lines)):
        if lines[position].rstrip() == _FRONT_MATTER_DELIMITER:
            closing = position
            break

    if closing is None:
        warnings.append(
            ExtractionWarning(
                code=ExtractionWarningCode.METADATA_UNAVAILABLE,
                message=(
                    "Document opens with a front matter delimiter that is "
                    "never closed; it has been treated as document content"
                ),
            )
        )
        return text, {}

    recovered: dict[str, str] = {}
    for line in lines[1:closing]:
        if not line.strip() or line.startswith("#"):
            continue
        match = _FRONT_MATTER_SCALAR.match(line)
        if match is None:
            continue
        value = match.group("value").strip("\"'")
        if not value:
            # An empty value introduces a nested block or a list, neither of
            # which is interpreted here.
            continue
        recovered[match.group("key").lower()] = value

    return "\n".join(lines[closing + 1 :]), recovered


def _build_metadata(
    front_matter: dict[str, str],
    first_heading: Optional[str],
) -> ExtractedMetadata:
    """
    Assemble format-native metadata from front matter and document structure.

    A declared title is preferred over an inferred one. Where none is
    declared, the document's first heading is used, which is the long-settled
    convention for Markdown documents and is what a reader would take the
    title to be.

    Parameters
    ----------
    front_matter:
        Scalar keys recovered from the front matter block.
    first_heading:
        Text of the document's first heading, when present.

    Returns
    -------
    ExtractedMetadata
        Format-native metadata. Keys promoted to typed fields are removed from
        the opaque properties, so that no value is recorded twice.
    """
    properties = dict(front_matter)

    title: Optional[str] = None
    for key in _TITLE_KEYS:
        if key in properties:
            title = properties.pop(key)
            break
    if title is None:
        title = first_heading

    authors: tuple[str, ...] = ()
    for key in _AUTHOR_KEYS:
        if key not in properties:
            continue
        raw = properties.pop(key)
        authors = tuple(
            part.strip().strip("[]")
            for part in raw.split(_AUTHOR_SEPARATOR)
            if part.strip().strip("[]")
        )
        break

    return ExtractedMetadata(
        title=title,
        authors=authors,
        # Timestamps are deliberately not recovered from front matter. Dates
        # appear there in many formats and with no timezone, and a
        # misinterpreted instant is worse than an absent one.
        declared_language=properties.pop("lang", None)
        or properties.pop("language", None),
        properties=properties,
    )
