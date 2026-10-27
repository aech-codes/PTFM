"""
PTFM Retrieval Tokenization

File:
    ptfm/core/retrieval/stores/tokenization.py

Description:
    The definition of what counts as a term, shared by every lexical backend.

Why this is shared rather than per-backend
------------------------------------------
    A lexical index has two separable behaviours: which nodes it MATCHES, and in
    what ORDER it returns them.

    Ranking may legitimately differ between backends. A score is declared
    uncalibrated, fusion is rank-only, and one lexical arm runs per query, so
    nothing downstream reads meaning into a magnitude or compares magnitudes
    across engines. A deployment that swaps backends should expect a different
    order.

    Matching may not. If one backend splits on word characters and another
    stems, applies a stopword list, or treats a hyphenated word as three terms,
    the two select different NODES for the same query. That is a difference in
    what the query means, not in how confidently it is answered, and it would
    make the in-memory stores useless as the reference the other backends are
    tested against.

    So the term rule lives here, in one place, and every ``LexicalIndex``
    implementation uses it. A backend whose engine cannot be made to honour it
    must say so in its own documentation rather than diverge quietly.

Why the rule is this simple
---------------------------
    Word characters, lowercased, nothing else. No stemming, no stopword list, no
    language detection.

    Each of those is a deployment's choice, and each makes a query match text
    that does not contain the terms the caller wrote. For a system whose output
    is evidence, matching more than was asked for is not a neutral convenience:
    it widens the candidate set with passages the caller cannot see the
    connection to, and the burden of that widening lands on the verification
    stages downstream.

    A deployment wanting linguistic analysis configures it in the backend it
    chooses, knowing that it has moved away from the reference.

Dependency contract
-------------------
    Standard library only. Imported by every lexical backend, so it must not
    pull in a driver, a model, or a configuration object.
"""

from __future__ import annotations

import re
from typing import Final

__all__ = [
    "TOKEN_PATTERN",
    "TOKENIZER_VERSION",
    "tokenize",
]

#: The term rule. ``\w`` under Unicode matches letters, digits, and the
#: underscore across every script, so a corpus in a non-Latin script tokenizes
#: without a separate rule for it.
TOKEN_PATTERN: Final[re.Pattern[str]] = re.compile(r"\w+", re.UNICODE)

#: Version of the term rule.
#:
#: Changing the rule changes which nodes a query matches, so an index built
#: under one rule and queried under another returns a set neither rule would
#: produce. This constant exists so that a backend maintaining derived term data
#: can record which rule produced it and refuse to mix two.
TOKENIZER_VERSION: Final[str] = "words-1.0"


def tokenize(text: str) -> tuple[str, ...]:
    """
    Split text into the terms a lexical search matches on.

    Parameters
    ----------
    text:
        The text to split. May be a query or a node's text; both must be split
        the same way, or a node would fail to match a term it contains.

    Returns
    -------
    tuple[str, ...]
        Lowercase terms in order of appearance, with repeats retained. Repeats
        matter to a frequency-based score, so removing them here would change
        ranking on behalf of a backend that had not asked.
    """
    return tuple(match.group().lower() for match in TOKEN_PATTERN.finditer(text))
