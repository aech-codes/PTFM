"""
PTFM Embedding Composition Stage

File:
    ptfm/core/knowledge/pipeline/composition.py

Description:
    Builds the text submitted to an embedding model from a node's fields.

Why composition exists at all
-----------------------------
    A chunk reading "The applicable rate is 4.5 percent" is nearly useless to
    retrieval without "Interest Rates". The obvious remedy is to prepend the
    heading to the chunk, and it is the wrong one: the moment the stored text
    ceases to be an exact substring of the source document, the node's span no
    longer identifies the text the node contains, and a verifier citing that
    span resolves to text the node does not hold.

    Composition separates the two. What is STORED remains exact; what is
    EMBEDDED carries context. The node is never modified, and the rule applied
    is recorded on the resulting vector, so a node embedded with its heading
    path is distinguishable from the same node embedded without it. Comparing
    across compositions yields a number that resembles a similarity and means
    nothing.

Truncation prefers to lose context, never content
-------------------------------------------------
    Configuration guarantees that a node at the maximum size fits within the
    embedding input bound, so only the heading path can push an input over it.

    Where that happens, the path is shortened from the outermost heading
    inward. The outermost heading is the least specific and contributes least
    to distinguishing this node from its neighbours, while the node's own text
    is the content being retrieved and is never truncated. Losing "Part I"
    costs little; losing the end of a clause could change what the vector
    means.

Execution contract
------------------
    Synchronous and CPU-bound. The embedding stage invokes it while preparing
    a batch.

Dependency contract
-------------------
    Imports only the standard library, ``ptfm.core.schemas`` and the knowledge
    ``exceptions``, ``config`` and ``interfaces`` modules.
"""

from __future__ import annotations

from typing import Final

from ptfm.core.knowledge.config import EmbeddingConfig
from ptfm.core.knowledge.exceptions import InputTooLargeError
from ptfm.core.knowledge.interfaces import ComposedInput
from ptfm.core.schemas import KnowledgeNode

__all__ = [
    "HEADING_PATH_COMPOSITION",
    "TEXT_ONLY_COMPOSITION",
    "HeadingPathComposer",
    "TextOnlyComposer",
]

#: Identifiers of the compositions shipped with the module. Recorded on every
#: vector produced, so a change here is a re-embedding event: vectors carrying
#: differing identifiers are not comparable.
HEADING_PATH_COMPOSITION: Final[str] = "heading_path_prefixed"
TEXT_ONLY_COMPOSITION: Final[str] = "text_only"

#: Separator between heading path entries within composed input. Chosen to
#: read naturally to a language model as a hierarchy rather than as prose.
_PATH_SEPARATOR: Final[str] = " > "

#: Separator between the heading path and the node text.
_CONTEXT_SEPARATOR: Final[str] = "\n\n"


class HeadingPathComposer:
    """
    Composes embedding input as the node's heading path followed by its text.

    The default. Retrieval quality on structured documents depends heavily on
    a chunk carrying its section context, and the heading path is the cheapest
    accurate source of it: the author wrote those headings deliberately, so
    unlike an inferred summary the context cannot be wrong.

    Satisfies the ``EmbeddingTextComposer`` protocol. Stateless, so one
    instance serves all concurrent workers.

    Parameters
    ----------
    config:
        Embedding configuration supplying the input bound.
    """

    __slots__ = ("_config",)

    def __init__(self, config: EmbeddingConfig) -> None:
        self._config = config

    @property
    def composition_id(self) -> str:
        """
        Stable identifier of this composition rule.

        Returns
        -------
        str
            The identifier recorded on every vector produced with it.
        """
        return HEADING_PATH_COMPOSITION

    def compose(self, node: KnowledgeNode) -> ComposedInput:
        """
        Build the embedding input for a node.

        The node is never modified. Its stored text remains the exact
        substring of the source document at its span; only the text submitted
        to the model carries additional context.

        Parameters
        ----------
        node:
            The node to embed.

        Returns
        -------
        ComposedInput
            The text to embed and this composition's identifier.

        Raises
        ------
        InputTooLargeError
            If the node's own text exceeds the configured bound, which
            configuration is validated to prevent and therefore indicates a
            strategy producing nodes larger than its configuration permits.
        """
        limit = self._config.max_input_chars

        if len(node.text) > limit:
            # The heading path cannot be shortened enough to help: the
            # content itself is over the bound. Configuration validates that
            # a node at the maximum size fits, so reaching here means a
            # strategy produced a node larger than its configuration allows.
            raise InputTooLargeError(
                input_chars=len(node.text),
                max_chars=limit,
                document_id=node.document_id,
                node_id=node.node_id,
                context={"composition_id": self.composition_id},
            )

        headings = _fit_heading_path(node, limit)

        if not headings:
            return ComposedInput(text=node.text, composition_id=self.composition_id)

        prefix = _PATH_SEPARATOR.join(headings)
        return ComposedInput(
            text=f"{prefix}{_CONTEXT_SEPARATOR}{node.text}",
            composition_id=self.composition_id,
        )

    def __repr__(self) -> str:
        """
        Return a representation naming this composition.

        Returns
        -------
        str
            Diagnostic representation.
        """
        return f"HeadingPathComposer(composition_id={self.composition_id!r})"


class TextOnlyComposer:
    """
    Composes embedding input as the node's text alone.

    Appropriate where documents carry no meaningful heading structure, such as
    a corpus of short standalone records, and where the heading path would
    contribute noise rather than context.

    Also the honest choice when comparing retrieval quality: a run under this
    composition and a run under the heading-path composition produce vectors
    that are not comparable, and the differing composition identifier makes
    that visible rather than leaving two corpora silently mixed.

    Satisfies the ``EmbeddingTextComposer`` protocol. Stateless.

    Parameters
    ----------
    config:
        Embedding configuration supplying the input bound.
    """

    __slots__ = ("_config",)

    def __init__(self, config: EmbeddingConfig) -> None:
        self._config = config

    @property
    def composition_id(self) -> str:
        """
        Stable identifier of this composition rule.

        Returns
        -------
        str
            The identifier recorded on every vector produced with it.
        """
        return TEXT_ONLY_COMPOSITION

    def compose(self, node: KnowledgeNode) -> ComposedInput:
        """
        Build the embedding input for a node.

        Parameters
        ----------
        node:
            The node to embed.

        Returns
        -------
        ComposedInput
            The node's text and this composition's identifier.

        Raises
        ------
        InputTooLargeError
            If the node's text exceeds the configured bound.
        """
        limit = self._config.max_input_chars

        if len(node.text) > limit:
            raise InputTooLargeError(
                input_chars=len(node.text),
                max_chars=limit,
                document_id=node.document_id,
                node_id=node.node_id,
                context={"composition_id": self.composition_id},
            )

        return ComposedInput(text=node.text, composition_id=self.composition_id)

    def __repr__(self) -> str:
        """
        Return a representation naming this composition.

        Returns
        -------
        str
            Diagnostic representation.
        """
        return f"TextOnlyComposer(composition_id={self.composition_id!r})"


# ============================================================================
# HELPERS
# ============================================================================


def _fit_heading_path(node: KnowledgeNode, limit: int) -> tuple[str, ...]:
    """
    Shorten a heading path until the composed input fits.

    Headings are dropped from the outermost inward. The outermost is the least
    specific and contributes least to distinguishing this node from its
    neighbours, whereas the innermost names the section the node actually sits
    in. The node's own text is never shortened, since it is the content being
    retrieved.

    Parameters
    ----------
    node:
        The node being composed.
    limit:
        Longest permitted composed input.

    Returns
    -------
    tuple[str, ...]
        The headings that fit, outermost first. Empty when none does, in which
        case the node's text is embedded alone.
    """
    headings = node.heading_path
    available = limit - len(node.text) - len(_CONTEXT_SEPARATOR)

    while headings:
        rendered = len(_PATH_SEPARATOR.join(headings))
        if rendered <= available:
            return headings
        headings = headings[1:]

    return ()
