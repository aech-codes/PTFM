"""
PTFM Sentence Segmentation

File:
    ptfm/core/knowledge/chunking/segmenters.py

Description:
    Locates sentence boundaries within a text, by character offset.

Used only as a fallback
-----------------------
    A block boundary is a division the author made deliberately, and is always
    preferred. Sentence segmentation applies only where a single block exceeds
    the size budget and must be divided internally, which on most corpora
    means it is never invoked at all.

    That narrow role is why a rule-based segmenter is the right default. A
    model-based one would be more accurate on hard cases, at the cost of a
    dependency, a download, and non-determinism across versions, to improve
    handling of a situation that arises rarely and in which any reasonable
    boundary suffices.

Offsets, not strings
--------------------
    The contract returns offsets. A segmenter returning sentence strings would
    be unusable here: the caller must place boundaries as offsets into the
    document, and recovering an offset by searching for a returned string is
    fragile and wrong wherever the string occurs more than once.

Determinism
-----------
    Segmentation affects where node boundaries fall, so it affects node
    identity. This implementation is a pure function of its input with no
    model, no locale lookup, and no dependency, which is what allows the same
    corpus to reprocess to the same identifiers on any machine.

    Its version participates in the segmentation fingerprint. Changing the
    rules below is a corpus migration and MUST be accompanied by a version
    increment, or nodes produced before and after would share identifiers
    while covering different text.

Dependency contract
-------------------
    Standard library only, and the knowledge ``interfaces`` module.
"""

from __future__ import annotations

import re
from typing import Final, Sequence

from ptfm.core.knowledge.interfaces import Sentence

__all__ = ["REGEX_SEGMENTER_NAME", "RegexSentenceSegmenter"]

#: Selector matching ``ChunkingConfig.sentence_segmenter``.
REGEX_SEGMENTER_NAME: Final[str] = "regex"

#: Version of the segmentation rules. Participates in the segmentation
#: fingerprint, so ANY change to the patterns below must increment it.
_SEGMENTER_VERSION: Final[str] = "regex-1.0"

#: Characters that can terminate a sentence, including the full stops and
#: question marks used in CJK and Devanagari scripts. Restricting these to the
#: ASCII set would leave many documents with no boundary at all, so an
#: oversized block in them could not be divided.
_TERMINATORS: Final[str] = ".!?\u3002\uff01\uff1f\u0964\u06d4"

#: Closing marks permitted between a terminator and the following whitespace,
#: so that a sentence ending inside a quotation or parenthesis is not split
#: from its closing mark.
_CLOSERS: Final[str] = "\"'\u201d\u2019)\\]}\u300d\u300f\u00bb"

#: A boundary is a terminator, optional closing marks, then whitespace. The
#: pattern matches the whitespace so that the following sentence begins at its
#: first non-space character, leaving inter-sentence whitespace between spans
#: rather than inside them.
_BOUNDARY: Final[re.Pattern[str]] = re.compile(
    rf"[{re.escape(_TERMINATORS)}][{re.escape(_CLOSERS)}]*\s+"
)

#: A paragraph break is a boundary regardless of punctuation. A block holding
#: several paragraphs of text that never terminate a sentence, such as a list
#: of headings run together, would otherwise be indivisible.
_PARAGRAPH_BREAK: Final[re.Pattern[str]] = re.compile(r"\n\s*\n")

#: Tokens that end in a full stop without ending a sentence. Kept deliberately
#: short: every entry is a guess about a language, and a wrong entry produces a
#: silently missing boundary. The cost of omitting one is a slightly longer
#: node, which is far cheaper than a wrong one.
_NON_TERMINAL: Final[frozenset[str]] = frozenset(
    {
        "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st",
        "no", "vs", "etc", "eg", "ie", "cf", "al",
        "inc", "ltd", "co", "corp", "fig", "vol", "pp", "ed",
        "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "oct", "nov", "dec",
    }
)

#: Trailing word immediately before a candidate boundary, used to test the
#: abbreviation list.
_TRAILING_WORD: Final[re.Pattern[str]] = re.compile(r"([A-Za-z]+)\.$")


class RegexSentenceSegmenter:
    """
    Rule-based sentence segmenter returning character offsets.

    Satisfies the ``SentenceSegmenter`` protocol. Stateless and deterministic,
    so one instance serves all concurrent workers and the same text always
    segments identically.

    Accuracy is deliberately modest. The segmenter runs only when a block
    exceeds the size budget, where the requirement is a defensible boundary
    rather than a linguistically perfect one, and where a wrong boundary costs
    a slightly awkward node rather than a lost or duplicated one.
    """

    __slots__ = ()

    @property
    def version(self) -> str:
        """
        Version identifier of these segmentation rules.

        Contributes to the segmentation fingerprint, so an upgrade that shifts
        a boundary is detectable rather than silently changing node identity.

        Returns
        -------
        str
            Version identifier.
        """
        return _SEGMENTER_VERSION

    def segment(self, text: str) -> Sequence[Sentence]:
        """
        Locate the sentences within a text.

        Offsets are relative to the text supplied; the caller translates them
        into document coordinates.

        Sentences are contiguous and non-overlapping. Inter-sentence
        whitespace falls between spans rather than inside them, which keeps a
        node's text free of leading blank space while leaving the whitespace
        available to the coverage check as a permitted gap.

        Parameters
        ----------
        text:
            Text to segment.

        Returns
        -------
        Sequence[Sentence]
            Sentences in reading order. Empty for a text offering no boundary,
            which the caller treats as unsplittable.
        """
        if not text or not text.strip():
            return ()

        boundaries = sorted(
            {*self._punctuation_boundaries(text), *self._paragraph_boundaries(text)}
        )
        if not boundaries:
            return ()

        sentences: list[Sentence] = []
        cursor = 0

        for end in boundaries:
            if end <= cursor:
                continue
            start = self._advance_past_space(text, cursor)
            if start < end:
                sentences.append(Sentence(start=start, end=end))
            cursor = end

        start = self._advance_past_space(text, cursor)
        if start < len(text):
            sentences.append(Sentence(start=start, end=len(text)))

        # One sentence means no interior boundary was found, so the text is
        # not divisible and the caller must be told so rather than handed a
        # segmentation that changes nothing.
        return tuple(sentences) if len(sentences) > 1 else ()

    # ------------------------------------------------------------------
    # Boundary detection
    # ------------------------------------------------------------------

    def _punctuation_boundaries(self, text: str) -> list[int]:
        """
        Locate boundaries marked by sentence-terminating punctuation.

        Parameters
        ----------
        text:
            Text to scan.

        Returns
        -------
        list[int]
            Offsets at which a sentence ends, excluding trailing whitespace.
        """
        boundaries: list[int] = []

        for match in _BOUNDARY.finditer(text):
            end = match.start() + len(match.group().rstrip())
            if self._is_abbreviation(text, end):
                continue
            boundaries.append(end)

        return boundaries

    def _paragraph_boundaries(self, text: str) -> list[int]:
        """
        Locate boundaries at blank lines.

        A paragraph break divides text regardless of punctuation, so that a
        block whose lines never terminate a sentence remains divisible.

        Parameters
        ----------
        text:
            Text to scan.

        Returns
        -------
        list[int]
            Offsets at which a paragraph ends.
        """
        return [match.start() for match in _PARAGRAPH_BREAK.finditer(text)]

    def _is_abbreviation(self, text: str, end: int) -> bool:
        """
        Report whether a candidate boundary follows a known abbreviation.

        Parameters
        ----------
        text:
            The text being scanned.
        end:
            Offset just past the terminating punctuation.

        Returns
        -------
        bool
            ``True`` when the preceding token is a known abbreviation and the
            boundary should be suppressed.
        """
        match = _TRAILING_WORD.search(text[:end])
        if match is None:
            return False
        return match.group(1).lower() in _NON_TERMINAL

    @staticmethod
    def _advance_past_space(text: str, cursor: int) -> int:
        """
        Advance an offset past leading whitespace.

        Parameters
        ----------
        text:
            The text being scanned.
        cursor:
            Offset to advance from.

        Returns
        -------
        int
            Offset of the first non-space character at or after the cursor.
        """
        while cursor < len(text) and text[cursor].isspace():
            cursor += 1
        return cursor

    def __repr__(self) -> str:
        """
        Return a representation naming this segmenter's version.

        Returns
        -------
        str
            Diagnostic representation.
        """
        return f"RegexSentenceSegmenter(version={self.version!r})"
