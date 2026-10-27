"""
PTFM HTML Extractor

File:
    ptfm/core/ingestion/extractors/html.py

Description:
    Extracts a structured block tree from HTML and XHTML payloads.

Boilerplate is the central problem
----------------------------------
    A web page is mostly not its content. Navigation menus, cookie banners,
    sidebars, footers, and advertising commonly outweigh the article itself.
    Ingesting all of it has a specific and damaging effect on the pipeline:
    every page from a site contributes the same navigation text, so those
    phrases become the most frequent strings in the corpus, retrieval scores
    them as significant, and a claim can be "supported" by evidence that is
    a cookie notice.

    Two defences are applied. Non-content elements are removed outright:
    scripts, styles, navigation, headers, footers, and asides. Where the
    optional ``trafilatura`` dependency is present, main-content detection
    runs first and its result is parsed instead, which is substantially more
    accurate on news and article pages than element removal alone.

    Boilerplate removal is applied only when a document looks like an article.
    A short fragment or a table-driven page would be gutted by it, so removal
    is skipped where the document is small or where the detector returns
    materially less text than the document contains.

Parser selection
----------------
    ``selectolax`` is used, over the lexbor backend. It is an order of
    magnitude faster than BeautifulSoup on large documents, which matters
    because web corpora are the highest volume source, and it recovers from
    malformed markup rather than raising, which matters because real HTML is
    frequently invalid.

    Alternatives considered: ``lxml`` with BeautifulSoup, slower with a much
    larger API surface, and ``html5lib``, specification exact but too slow for
    corpus scale.

    Both dependencies are optional, supplied by the ``ptfm[web]`` extra and
    reported through ``check_availability`` at registration rather than as an
    ``ImportError`` partway through a corpus.

Structure model
---------------
    Semantic elements map onto canonical blocks: headings, paragraphs, lists,
    tables, quotations, code, and figures. Presentational elements do not.
    A ``div`` becomes a block only when it directly holds text, since HTML
    uses it for both structure and layout and treating every one as a section
    would produce a tree many levels deep that describes styling rather than
    meaning.

Dependency contract
-------------------
    Imports the standard library, ``selectolax``, ``ptfm.core.schemas`` and
    the ingestion ``exceptions``, ``interfaces``, ``models`` and
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
    from selectolax.lexbor import LexborHTMLParser, LexborNode

__all__ = ["HtmlExtractor"]

#: Media types claimed by this extractor.
_MEDIA_TYPES: Final[frozenset[str]] = frozenset(
    {"text/html", "application/xhtml+xml"}
)

#: Distributions supplying the parser and the optional content detector, and
#: the packaging extra providing them.
_PARSER_DEPENDENCY: Final[str] = "selectolax"
_INSTALL_EXTRA: Final[str] = "ptfm[web]"

#: Elements removed before extraction. Scripts and styles carry code rather
#: than prose; the remainder are the structural furniture of a page and repeat
#: identically across every document from a site.
_DISCARDED_TAGS: Final[tuple[str, ...]] = (
    "script",
    "style",
    "noscript",
    "template",
    "svg",
    "canvas",
    "iframe",
    "object",
    "embed",
    "nav",
    "header",
    "footer",
    "aside",
    "form",
    "button",
    "select",
    "textarea",
    "dialog",
)

#: Heading tags mapped to their outline level.
_HEADING_LEVELS: Final[dict[str, int]] = {
    "h1": 1,
    "h2": 2,
    "h3": 3,
    "h4": 4,
    "h5": 5,
    "h6": 6,
}

#: Leaf elements mapped to canonical block types. Their text is taken whole
#: rather than descended into, since inline markup within them is styling
#: rather than structure.
_LEAF_TAGS: Final[dict[str, BlockType]] = {
    "p": BlockType.PARAGRAPH,
    "pre": BlockType.CODE,
    "code": BlockType.CODE,
    "figcaption": BlockType.CAPTION,
    "caption": BlockType.CAPTION,
    "dt": BlockType.PARAGRAPH,
    "dd": BlockType.PARAGRAPH,
    "address": BlockType.PARAGRAPH,
}

#: Container elements mapped to canonical container block types.
_CONTAINER_TAGS: Final[dict[str, BlockType]] = {
    "ul": BlockType.LIST,
    "ol": BlockType.LIST,
    "dl": BlockType.LIST,
    "li": BlockType.LIST_ITEM,
    "blockquote": BlockType.QUOTE,
    "table": BlockType.TABLE,
    "tr": BlockType.TABLE_ROW,
    "th": BlockType.TABLE_CELL,
    "td": BlockType.TABLE_CELL,
    "figure": BlockType.FIGURE,
    "section": BlockType.SECTION,
    "article": BlockType.SECTION,
    "main": BlockType.SECTION,
}

#: Elements that carry structure but no content of their own. Descended into
#: without producing a block, so that the tree describes meaning rather than
#: markup depth.
_TRANSPARENT_TAGS: Final[frozenset[str]] = frozenset(
    {"html", "body", "thead", "tbody", "tfoot", "colgroup", "picture", "span"}
)

#: Elements whose text must not be collapsed, because their whitespace is
#: content rather than formatting.
_WHITESPACE_SIGNIFICANT_TAGS: Final[frozenset[str]] = frozenset({"pre"})

#: Meta tag names carrying document metadata, mapped to a canonical key.
#: Both plain names and OpenGraph properties are recognized, since news and
#: article pages commonly supply one and not the other.
_METADATA_KEYS: Final[dict[str, str]] = {
    "author": "author",
    "article:author": "author",
    "og:title": "title",
    "twitter:title": "title",
    "description": "description",
    "og:description": "description",
    "og:site_name": "site_name",
    "generator": "generator",
    "keywords": "keywords",
}

#: Separator by which a multi-author meta value is split.
_AUTHOR_SEPARATORS: Final[re.Pattern[str]] = re.compile(r"\s*[,;]\s*|\s+and\s+")

#: A run of whitespace, collapsed within a block's text. Applied here so that
#: markup indentation does not become document content; the structure stage
#: normalizes the assembled text separately.
_WHITESPACE_RUN: Final[re.Pattern[str]] = re.compile(r"\s+")

#: Shortest document, in characters of extracted text, on which boilerplate
#: removal is attempted. A short fragment has no boilerplate to remove and
#: would be gutted by a detector tuned for full article pages.
_MIN_BOILERPLATE_REMOVAL_CHARS: Final[int] = 2_000

#: Smallest share of a document's text that main-content detection must
#: return for its result to be used. A detector returning materially less has
#: most likely mistaken content for furniture, as happens on table-driven and
#: list-driven pages, so its result is discarded rather than trusted.
_MIN_CONTENT_RETENTION: Final[float] = 0.2


class HtmlExtractor(ContentExtractor):
    """
    Extracts a structured block tree from HTML and XHTML payloads.

    Stateless: all parsing state is local to a call, so one instance serves
    all concurrent workers.

    Parameters
    ----------
    remove_boilerplate:
        Whether main-content detection is attempted before parsing. Enabled by
        default, and silently inactive when the optional detector is not
        installed. Disable when ingesting HTML that is not article shaped,
        such as generated reports or exported documentation, where the whole
        document is content.
    """

    __slots__ = ("_remove_boilerplate",)

    def __init__(self, *, remove_boilerplate: bool = True) -> None:
        self._remove_boilerplate = remove_boilerplate

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
            The identifier ``"HtmlExtractor"``.
        """
        return "HtmlExtractor"

    @property
    def supported_media_types(self) -> frozenset[str]:
        """
        Media types this extractor claims.

        Returns
        -------
        frozenset[str]
            The HTML and XHTML media types.
        """
        return _MEDIA_TYPES

    @classmethod
    def check_availability(cls) -> None:
        """
        Verify that the HTML parser is installed.

        Only the parser is required. Main-content detection is genuinely
        optional: without it, element removal still discards the bulk of page
        furniture, so a deployment lacking the detector produces noisier but
        usable documents rather than none.

        Returns
        -------
        None

        Raises
        ------
        ExtractorUnavailableError
            If the parser is not installed.
        """
        try:
            import selectolax.lexbor  # noqa: F401
        except ImportError as exc:
            raise ExtractorUnavailableError(
                media_type="text/html",
                missing_dependency=_PARSER_DEPENDENCY,
                install_extra=_INSTALL_EXTRA,
            ) from exc

    # ------------------------------------------------------------------
    # Extraction
    # ------------------------------------------------------------------

    def extract(self, payload: ExtractionInput) -> ExtractionOutput:
        """
        Parse an HTML payload into a canonical block tree.

        Blocking and CPU-bound by contract. The extraction stage executes this
        in a worker thread; it must never be invoked from a coroutine.

        Parameters
        ----------
        payload:
            The raw bytes together with the resolved media type and encoding.

        Returns
        -------
        ExtractionOutput
            The block tree, document metadata, and any observations recorded
            while parsing.

        Raises
        ------
        ExtractorUnavailableError
            If the parser is not installed.
        CorruptDocumentError
            If the document is nested too deeply to parse.
        ExtractorInternalError
            If the resolved encoding is unrecognized, the parser raised an
            unanticipated exception, or the assembled result failed its own
            validation.
        """
        self.check_availability()
        from selectolax.lexbor import LexborHTMLParser

        warnings: list[ExtractionWarning] = []
        markup = decode_payload(payload, warnings)

        try:
            document = LexborHTMLParser(markup)
        except Exception as exc:
            raise ExtractorInternalError(
                f"HTML parser raised {type(exc).__name__} while parsing "
                f"{payload.source_uri!r}: {exc}",
                extractor_name=self.name,
                source_uri=payload.source_uri,
            ) from exc

        # Metadata is harvested before any pruning, because the elements that
        # carry it live in the document head, which content extraction
        # discards.
        metadata = _harvest_metadata(document)

        root = self._resolve_content_root(document, markup, warnings)

        try:
            blocks = _TreeTranslator(warnings).translate(root)
        except RecursionError as exc:
            raise CorruptDocumentError(
                f"HTML payload {payload.source_uri!r} is nested too deeply to "
                f"traverse.",
                source_uri=payload.source_uri,
                context={"extractor_name": self.name},
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

    # ------------------------------------------------------------------
    # Content isolation
    # ------------------------------------------------------------------

    def _resolve_content_root(
        self,
        document: LexborHTMLParser,
        markup: str,
        warnings: list[ExtractionWarning],
    ) -> LexborNode:
        """
        Determine the subtree from which content should be extracted.

        Main-content detection is attempted first where it is available and
        the document is large enough to plausibly contain boilerplate. Its
        result is accepted only when it retains a substantial share of the
        document's text, since a detector that discards most of a page has
        more likely misjudged a table-driven or list-driven layout than found
        a small article within a large page.

        Element removal is applied in either case, so that a document reaching
        this method without detection still loses its scripts, navigation, and
        footers.

        Parameters
        ----------
        document:
            The parsed document.
        markup:
            The decoded markup, required by the detector, which operates on
            source text rather than on a parsed tree.
        warnings:
            Accumulator to which an observation is appended when detection
            runs or is rejected.

        Returns
        -------
        LexborNode
            The root of the subtree to extract from.
        """
        extracted = self._detect_main_content(document, markup, warnings)
        if extracted is not None:
            return extracted

        _prune(document.root)
        return document.root

    def _detect_main_content(
        self,
        document: LexborHTMLParser,
        markup: str,
        warnings: list[ExtractionWarning],
    ) -> Optional[LexborNode]:
        """
        Attempt main-content detection, returning a parsed subtree.

        Parameters
        ----------
        document:
            The parsed document, consulted for its total text length.
        markup:
            The decoded markup.
        warnings:
            Accumulator to which an observation is appended when detection
            runs, is rejected, or fails.

        Returns
        -------
        Optional[LexborNode]
            The root of the detected content, or ``None`` when detection was
            skipped, unavailable, rejected, or unsuccessful.
        """
        if not self._remove_boilerplate:
            return None

        try:
            import trafilatura
        except ImportError:
            # Genuinely optional. Element removal alone still discards most
            # page furniture, so a deployment without the detector produces
            # noisier but usable documents.
            return None

        from selectolax.lexbor import LexborHTMLParser

        full_text = document.root.text(separator=" ", strip=True) if document.root else ""
        if len(full_text) < _MIN_BOILERPLATE_REMOVAL_CHARS:
            # Too small to contain boilerplate worth removing, and small
            # enough that a detector tuned for article pages would gut it.
            return None

        try:
            detected = trafilatura.extract(
                markup,
                output_format="html",
                include_tables=True,
                include_formatting=True,
                favor_recall=True,
            )
        except Exception as exc:
            # Detection is an enhancement. Its failure must not fail a
            # document that element removal can still handle.
            warnings.append(
                ExtractionWarning(
                    code=ExtractionWarningCode.PARTIAL_EXTRACTION,
                    message=(
                        f"Main content detection raised "
                        f"{type(exc).__name__} and was skipped; boilerplate "
                        f"may remain"
                    ),
                    detail={"exception_type": type(exc).__name__},
                )
            )
            return None

        if not detected:
            return None

        candidate = LexborHTMLParser(detected)
        if candidate.root is None:
            return None

        retained = candidate.root.text(separator=" ", strip=True)
        if len(retained) < len(full_text) * _MIN_CONTENT_RETENTION:
            warnings.append(
                ExtractionWarning(
                    code=ExtractionWarningCode.STRUCTURE_INFERRED,
                    message=(
                        "Main content detection retained too little of the "
                        "document and was rejected; the full document was "
                        "extracted instead"
                    ),
                    detail={
                        "document_chars": len(full_text),
                        "retained_chars": len(retained),
                    },
                )
            )
            return None

        warnings.append(
            ExtractionWarning(
                code=ExtractionWarningCode.STRUCTURE_INFERRED,
                message=(
                    "Page boilerplate was removed by main content detection"
                ),
                detail={
                    "document_chars": len(full_text),
                    "retained_chars": len(retained),
                },
            )
        )
        _prune(candidate.root)
        return candidate.root


# ============================================================================
# TREE TRANSLATION
# ============================================================================


class _TreeTranslator:
    """
    Converts a parsed HTML tree into a flat parent-indexed block list.

    Traversal is depth first, so blocks are emitted in pre-order, which is
    exactly what the canonical model requires. An explicit stack is used
    rather than recursion, because deeply nested markup is common enough on
    the web that recursion depth would be a real failure mode.

    One instance per document, since it accumulates per-document state.

    Parameters
    ----------
    warnings:
        Accumulator to which observations about skipped content are appended.
    """

    __slots__ = ("_warnings", "_blocks", "_child_counts", "_reported_tags")

    def __init__(self, warnings: list[ExtractionWarning]) -> None:
        self._warnings = warnings
        self._blocks: list[ExtractedBlock] = []
        self._child_counts: dict[int, int] = {}
        self._reported_tags: set[str] = set()

    def translate(self, root: Optional[LexborNode]) -> tuple[ExtractedBlock, ...]:
        """
        Convert a parsed tree into canonical blocks.

        Parameters
        ----------
        root:
            Root of the subtree to convert.

        Returns
        -------
        tuple[ExtractedBlock, ...]
            Blocks in depth-first pre-order.
        """
        if root is None:
            return ()

        self._visit(root, parent=None)
        self._close_empty_containers()
        return tuple(self._blocks)

    # ------------------------------------------------------------------
    # Traversal
    # ------------------------------------------------------------------

    def _visit(self, node: LexborNode, parent: Optional[int]) -> None:
        """
        Translate a node and its descendants.

        Parameters
        ----------
        node:
            The node to translate.
        parent:
            Index of the enclosing container block, or ``None`` at the root.

        Returns
        -------
        None
        """
        tag = (node.tag or "").lower()

        if tag in _DISCARDED_TAGS:
            return

        if tag in _HEADING_LEVELS:
            self._append_leaf(
                BlockType.HEADING,
                _node_text(node),
                parent,
                attributes=BlockAttributes(heading_level=_HEADING_LEVELS[tag]),
            )
            return

        if tag in _LEAF_TAGS:
            self._append_leaf(_LEAF_TAGS[tag], _node_text(node), parent)
            return

        if tag in _CONTAINER_TAGS:
            index = self._append_container(
                _CONTAINER_TAGS[tag], parent, self._container_attributes(tag, parent)
            )
            self._visit_children(node, index)
            return

        if tag in _TRANSPARENT_TAGS or tag in ("-text", "", "head", "title", "meta"):
            # Structural or already harvested. Descended into without
            # producing a block, so the tree reflects meaning rather than
            # markup depth.
            if tag not in ("head", "title", "meta"):
                self._visit_children(node, parent)
            return

        if tag == "div":
            self._visit_div(node, parent)
            return

        # An unrecognized element may still hold recognized descendants, so it
        # is descended into rather than discarded, and its own direct text is
        # preserved if it has any.
        direct = _direct_text(node)
        if direct:
            self._append_leaf(BlockType.PARAGRAPH, direct, parent)
        else:
            self._record_unrecognized(tag)
        self._visit_children(node, parent)

    def _visit_div(self, node: LexborNode, parent: Optional[int]) -> None:
        """
        Translate a generic container element.

        A ``div`` becomes a block only when it directly holds text. HTML uses
        it for both structure and layout, and treating every one as a section
        would produce a tree many levels deep describing styling rather than
        meaning.

        Parameters
        ----------
        node:
            The element to translate.
        parent:
            Index of the enclosing container block.

        Returns
        -------
        None
        """
        direct = _direct_text(node)
        if direct:
            self._append_leaf(BlockType.PARAGRAPH, direct, parent)
        self._visit_children(node, parent)

    def _visit_children(self, node: LexborNode, parent: Optional[int]) -> None:
        """
        Translate every child of a node.

        Parameters
        ----------
        node:
            The node whose children are to be translated.
        parent:
            Index of the enclosing container block.

        Returns
        -------
        None
        """
        child = node.child
        while child is not None:
            self._visit(child, parent)
            child = child.next

    # ------------------------------------------------------------------
    # Block construction
    # ------------------------------------------------------------------

    def _append_leaf(
        self,
        block_type: BlockType,
        text: str,
        parent: Optional[int],
        *,
        attributes: Optional[BlockAttributes] = None,
    ) -> None:
        """
        Append a content-bearing block, unless it holds no text.

        A leaf with no text is omitted rather than emitted empty, since an
        empty paragraph represents nothing and would add a zero length span
        to the document for no purpose.

        Parameters
        ----------
        block_type:
            Canonical block type. Must be a leaf type.
        text:
            The block's text.
        parent:
            Index of the enclosing container block.
        attributes:
            Optional structural attributes.

        Returns
        -------
        None
        """
        if not text.strip():
            return

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

    def _append_container(
        self,
        block_type: BlockType,
        parent: Optional[int],
        attributes: BlockAttributes,
    ) -> int:
        """
        Append a container block.

        Parameters
        ----------
        block_type:
            Canonical container block type.
        parent:
            Index of the enclosing container block.
        attributes:
            Structural attributes.

        Returns
        -------
        int
            Index of the appended container.
        """
        if parent is not None:
            self._child_counts[parent] = self._child_counts.get(parent, 0) + 1

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

    def _close_empty_containers(self) -> None:
        """
        Give every childless container an empty leaf child.

        A container with no children cannot be assigned a span and is rejected
        by the canonical model. HTML produces such containers readily, as with
        an empty table cell used for layout, so an empty paragraph is
        supplied. The table's shape is preserved without asserting content
        that is absent.

        Returns
        -------
        None
        """
        childless = [
            index
            for index, block in enumerate(self._blocks)
            if block.text is None and self._child_counts.get(index, 0) == 0
        ]
        for index in childless:
            self._child_counts[index] = 1
            self._blocks.append(
                ExtractedBlock(
                    block_type=BlockType.PARAGRAPH,
                    text="",
                    parent_index=index,
                )
            )

    # ------------------------------------------------------------------
    # Attributes and diagnostics
    # ------------------------------------------------------------------

    def _container_attributes(
        self,
        tag: str,
        parent: Optional[int],
    ) -> BlockAttributes:
        """
        Derive structural attributes for a container element.

        Parameters
        ----------
        tag:
            The element's lowercased tag name.
        parent:
            Index of the enclosing container block.

        Returns
        -------
        BlockAttributes
            Attributes appropriate to the element.
        """
        position = self._child_counts.get(parent, 0) if parent is not None else 0

        if tag == "tr":
            return BlockAttributes(row_index=position)
        if tag in ("th", "td"):
            return BlockAttributes(
                column_index=position, is_header=tag == "th"
            )
        if tag == "li":
            return BlockAttributes(list_level=self._list_depth(parent))
        return BlockAttributes()

    def _list_depth(self, parent: Optional[int]) -> int:
        """
        Count the list containers enclosing a position.

        Parameters
        ----------
        parent:
            Index of the enclosing container block.

        Returns
        -------
        int
            Nesting depth, zero for an outermost list item.
        """
        depth = 0
        index = parent
        while index is not None:
            block = self._blocks[index]
            if block.block_type is BlockType.LIST:
                depth += 1
            index = block.parent_index
        return max(0, depth - 1)

    def _record_unrecognized(self, tag: str) -> None:
        """
        Record that an element type contributed no block.

        Recorded once per tag name rather than once per occurrence, so that a
        document containing many of the same element does not flood the
        warning list.

        Parameters
        ----------
        tag:
            The element's lowercased tag name.

        Returns
        -------
        None
        """
        if not tag or tag in self._reported_tags:
            return
        self._reported_tags.add(tag)
        self._warnings.append(
            ExtractionWarning(
                code=ExtractionWarningCode.UNSUPPORTED_ELEMENT_SKIPPED,
                message=(
                    f"HTML element {tag!r} has no canonical block type; its "
                    f"descendants were still extracted"
                ),
                detail={"tag": tag},
            )
        )


# ============================================================================
# TEXT AND METADATA
# ============================================================================


def _node_text(node: LexborNode) -> str:
    """
    Recover the readable text of an element and its descendants.

    Whitespace is collapsed, so that markup indentation does not become
    document content, except within elements whose whitespace is significant.

    Parameters
    ----------
    node:
        The element whose text is required.

    Returns
    -------
    str
        The readable text.
    """
    tag = (node.tag or "").lower()

    if tag in _WHITESPACE_SIGNIFICANT_TAGS:
        return node.text(separator="", strip=False).strip("\n")

    return _WHITESPACE_RUN.sub(" ", node.text(separator=" ", strip=False)).strip()


def _direct_text(node: LexborNode) -> str:
    """
    Recover only the text held directly by an element.

    Text belonging to child elements is excluded, so that descending into a
    generic container does not record the same text twice: once for the
    container and again for each child that owns part of it.

    Parameters
    ----------
    node:
        The element whose direct text is required.

    Returns
    -------
    str
        The directly held text, collapsed and stripped.
    """
    parts: list[str] = []
    child = node.child
    while child is not None:
        if (child.tag or "").lower() == "-text":
            parts.append(child.text(deep=False, strip=False) or "")
        child = child.next
    return _WHITESPACE_RUN.sub(" ", "".join(parts)).strip()


def _harvest_metadata(document: LexborHTMLParser) -> ExtractedMetadata:
    """
    Recover document metadata from the head elements.

    Performed before content extraction, because the elements carrying
    metadata live in the document head, which extraction discards.

    Parameters
    ----------
    document:
        The parsed document.

    Returns
    -------
    ExtractedMetadata
        Format-native metadata. Keys promoted to typed fields are removed from
        the opaque properties, so that no value is recorded twice.
    """
    properties: dict[str, str] = {}

    for element in document.css("meta"):
        attributes = element.attributes
        key = attributes.get("name") or attributes.get("property")
        content = attributes.get("content")
        if not key or not content:
            continue
        canonical = _METADATA_KEYS.get(key.strip().lower())
        if canonical is None:
            continue
        properties.setdefault(canonical, content.strip())

    title = properties.pop("title", None)
    if title is None:
        element = document.css_first("title")
        if element is not None:
            title = _node_text(element) or None

    authors: tuple[str, ...] = ()
    raw_author = properties.pop("author", None)
    if raw_author:
        authors = tuple(
            part.strip()
            for part in _AUTHOR_SEPARATORS.split(raw_author)
            if part.strip()
        )

    language: Optional[str] = None
    root = document.css_first("html")
    if root is not None:
        language = (root.attributes.get("lang") or "").strip() or None

    return ExtractedMetadata(
        title=title,
        authors=authors,
        # Timestamps are deliberately not recovered. Page metadata expresses
        # dates in many formats, frequently without a timezone, and a
        # misinterpreted instant is worse than an absent one.
        declared_language=language,
        producer=properties.pop("generator", None),
        properties=properties,
    )


def _prune(root: Optional[LexborNode]) -> None:
    """
    Remove non-content elements from a parsed tree.

    Applied before traversal so that page furniture never reaches the block
    tree. Without it, every document from a site would contribute the same
    navigation text, which would then be among the most frequent strings in
    the corpus and would be scored as significant by retrieval.

    Parameters
    ----------
    root:
        Root of the tree to prune. Ignored when absent.

    Returns
    -------
    None
    """
    if root is None:
        return
    for tag in _DISCARDED_TAGS:
        for element in root.css(tag):
            element.decompose()
