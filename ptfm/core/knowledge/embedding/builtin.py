"""
PTFM Built-in Embedding Providers

File:
    ptfm/core/knowledge/embedding/builtin.py

Description:
    Providers that require no model and no network, and therefore belong in
    the Core: one that declines to embed, and one that embeds deterministically
    from a digest.

Why a hash-based provider is not a toy
--------------------------------------
    ``DeterministicEmbeddingProvider`` derives a vector from a digest of the
    input text. The vectors carry no semantics whatever: two paraphrases of
    one sentence are as distant as two unrelated ones.

    It is nonetheless the single most useful provider during development, for
    a reason that has nothing to do with retrieval quality. Every property of
    this module that is worth testing — that identifiers are stable, that
    coverage holds, that units reach a sink, that a corpus reprocesses
    identically — is independent of what the vectors mean. Testing those
    properties against a real model would make the suite slow, require a
    multi-gigabyte download, and introduce a dependency whose version could
    change results.

    Because it is deterministic, it also makes the ONE embedding property
    that does matter architecturally testable: that re-embedding a corpus
    reproduces identical vectors, which is what allows a vector store to be
    rebuilt without reprocessing.

    It must never be used in production, and its docstring says so plainly
    rather than relying on the name to convey it.

Why no real model lives here
----------------------------
    A provider wrapping a neural model belongs behind an optional dependency,
    because requiring a multi-gigabyte install of every deployment — including
    one that only ingests and indexes lexically — would be wrong. Those
    providers live in sibling modules and are registered through entry points.

Dependency contract
-------------------
    Standard library only, plus the knowledge ``exceptions`` and
    ``interfaces`` modules. No model, no network, no optional dependency.
"""

from __future__ import annotations

import hashlib
import math
import struct
from typing import Final, Sequence

from ptfm.core.knowledge.config import EmbeddingConfig
from ptfm.core.knowledge.exceptions import EmbeddingProviderError
from ptfm.core.knowledge.interfaces import EmbeddingProvider, ModelIdentity

__all__ = [
    "NULL_PROVIDER_NAME",
    "DETERMINISTIC_PROVIDER_NAME",
    "NullEmbeddingProvider",
    "DeterministicEmbeddingProvider",
]

#: Selectors matching ``EmbeddingConfig.provider``.
NULL_PROVIDER_NAME: Final[str] = "null"
DETERMINISTIC_PROVIDER_NAME: Final[str] = "deterministic"

#: Model identifier recorded on every vector the deterministic provider
#: produces. Deliberately explicit about what it is, so that a corpus embedded
#: with it during development cannot later be mistaken for a real one.
_DETERMINISTIC_MODEL_ID: Final[str] = "ptfm/deterministic-hash-embedding"

#: Version of the derivation. Recorded as the model revision, so that a change
#: to how vectors are derived is visible as a model change rather than
#: silently altering vectors under an unchanged identity.
_DETERMINISTIC_REVISION: Final[str] = "1"

#: Bytes consumed per vector component. Four bytes yield a 32-bit unsigned
#: integer, which is ample resolution for a component in [-1, 1].
_BYTES_PER_COMPONENT: Final[int] = 4

#: Largest value a component's raw integer can take, used to map it into
#: [-1, 1].
_COMPONENT_SCALE: Final[float] = float(0xFFFFFFFF)

#: Batch bound for providers that impose none of their own. Neither provider
#: here does real work, so the bound exists only to keep a caller's batching
#: logic exercised rather than trivially satisfied.
_DEFAULT_MAX_BATCH: Final[int] = 512


class NullEmbeddingProvider(EmbeddingProvider):
    """
    A provider that declines to embed.

    Constructed when embedding is disabled, so that the composition root has
    no conditional path and every stage can be wired unconditionally.

    Calling :meth:`embed` is an error rather than a no-op returning empty
    vectors. A zero vector is a valid input to a similarity computation and
    would place every node at an identical distance from every query, which is
    a silent and total corruption of retrieval. Failing loudly means the
    misconfiguration is discovered at the first batch rather than after a
    corpus has been indexed.

    Parameters
    ----------
    config:
        Accepted for interface uniformity so that every provider is
        constructed identically. No field is consulted.
    """

    __slots__ = ("_config",)

    def __init__(self, config: EmbeddingConfig) -> None:
        self._config = config

    @property
    def name(self) -> str:
        """
        Stable identifier of this provider.

        Returns
        -------
        str
            Implementation identifier.
        """
        return NULL_PROVIDER_NAME

    @property
    def max_batch_size(self) -> int:
        """
        Largest batch this provider accepts.

        Returns
        -------
        int
            A nominal bound; no batch is ever embedded.
        """
        return _DEFAULT_MAX_BATCH

    def model_identity(self) -> ModelIdentity:
        """
        Report the model this provider embeds with.

        Returns
        -------
        ModelIdentity
            A one-dimensional identity. No vector is ever produced under it,
            so the value serves only to satisfy the contract.
        """
        return ModelIdentity(
            model_id="ptfm/null",
            dimension=1,
            normalized=False,
            revision=_DETERMINISTIC_REVISION,
        )

    async def embed(self, texts: Sequence[str]) -> Sequence[tuple[float, ...]]:
        """
        Decline to embed.

        Parameters
        ----------
        texts:
            The inputs that would have been embedded.

        Returns
        -------
        Sequence[tuple[float, ...]]
            Never returns.

        Raises
        ------
        EmbeddingProviderError
            Always. Reaching this method means embedding is enabled while the
            null provider is configured, which is a wiring fault: returning
            zero vectors instead would place every node at an identical
            distance from every query, corrupting retrieval silently.
        """
        raise EmbeddingProviderError(
            f"The null embedding provider cannot produce vectors, but was "
            f"asked to embed {len(texts)} inputs. Either disable embedding or "
            f"configure a provider that can embed; returning empty vectors "
            f"would make every node equidistant from every query.",
            provider_name=self.name,
            batch_size=len(texts),
        )

    async def embed_query(self, query: str) -> tuple[float, ...]:
        """
        Decline to embed a query.

        Parameters
        ----------
        query:
            The query that would have been embedded.

        Returns
        -------
        tuple[float, ...]
            Never returns.

        Raises
        ------
        EmbeddingProviderError
            Always, for the same reason :meth:`embed` does.
        """
        await self.embed([query])
        raise AssertionError("unreachable")  # pragma: no cover

    async def health_check(self) -> bool:
        """
        Report whether the provider is operational.

        Returns
        -------
        bool
            Always ``False``. The provider cannot embed, and reporting
            otherwise would let a run begin that must fail at its first batch.
        """
        return False

    async def close(self) -> None:
        """
        Release resources. None are held.

        Returns
        -------
        None
        """
        return None

    def __repr__(self) -> str:
        """
        Return a representation of this provider.

        Returns
        -------
        str
            Diagnostic representation.
        """
        return "NullEmbeddingProvider()"


class DeterministicEmbeddingProvider(EmbeddingProvider):
    """
    Derives a vector from a digest of the input text.

    NOT FOR PRODUCTION USE. The vectors carry no semantic content: two
    paraphrases of one sentence are as distant as two unrelated sentences, so
    retrieval over them is no better than random. A corpus embedded with this
    provider is not searchable in any meaningful sense.

    Its purpose is testing and development. Every architectural property of
    this module — identifier stability, coverage, unit delivery, idempotent
    reprocessing — is independent of what vectors mean, and verifying them
    against a real model would require a multi-gigabyte download and make the
    suite slow. Because the derivation is deterministic, it also makes
    testable the one embedding property that is architecturally significant:
    that re-embedding reproduces identical vectors, which is what allows a
    vector store to be rebuilt without reprocessing the corpus.

    Vectors are unit-normalized, so that the normalization claim recorded on
    every embedding is true and downstream comparability checks exercise the
    same path they would with a real model.

    Parameters
    ----------
    config:
        Supplies the dimension, so that a test can exercise the dimension a
        target model uses without loading it.

    Raises
    ------
    EmbeddingProviderError
        If the configured dimension is not usable.
    """

    __slots__ = ("_config", "_dimension")

    def __init__(self, config: EmbeddingConfig, *, dimension: int = 256) -> None:
        if dimension < 1:
            raise EmbeddingProviderError(
                f"Embedding dimension must be at least one, got {dimension}.",
                provider_name=DETERMINISTIC_PROVIDER_NAME,
            )
        self._config = config
        self._dimension = dimension

    @property
    def name(self) -> str:
        """
        Stable identifier of this provider.

        Returns
        -------
        str
            Implementation identifier.
        """
        return DETERMINISTIC_PROVIDER_NAME

    @property
    def max_batch_size(self) -> int:
        """
        Largest batch this provider accepts.

        Returns
        -------
        int
            A nominal bound. Derivation is cheap, so the bound exists to keep
            a caller's batching logic exercised rather than to protect the
            provider.
        """
        return _DEFAULT_MAX_BATCH

    def model_identity(self) -> ModelIdentity:
        """
        Report the model this provider embeds with.

        The identifier states plainly that the vectors are derived from a
        hash, so that a corpus embedded during development cannot later be
        mistaken for one embedded by a real model.

        Returns
        -------
        ModelIdentity
            Identifier, dimension, normalization, and revision.
        """
        return ModelIdentity(
            model_id=_DETERMINISTIC_MODEL_ID,
            dimension=self._dimension,
            normalized=True,
            revision=_DETERMINISTIC_REVISION,
            max_input_chars=self._config.max_input_chars,
        )

    async def embed(self, texts: Sequence[str]) -> Sequence[tuple[float, ...]]:
        """
        Derive a vector for each input.

        No awaiting occurs. The method is a coroutine because the contract
        requires it, not because the work is asynchronous, and the derivation
        is fast enough that offloading it would cost more than it saves.

        Parameters
        ----------
        texts:
            The inputs to embed.

        Returns
        -------
        Sequence[tuple[float, ...]]
            One unit-normalized vector per input, in input order.
        """
        return tuple(self._derive(text) for text in texts)

    async def embed_query(self, query: str) -> tuple[float, ...]:
        """
        Embed a query.

        Symmetric: the derivation applies no query-side transformation, which
        is why :attr:`ModelIdentity.query_encoding` is empty. A query and a
        passage of identical text therefore yield identical vectors, which is
        what makes the provider usable for testing a retrieval path end to
        end.

        Parameters
        ----------
        query:
            The query text.

        Returns
        -------
        tuple[float, ...]
            A vector comparable with those produced by :meth:`embed`.
        """
        return self._derive(query)

    async def health_check(self) -> bool:
        """
        Report whether the provider is operational.

        Returns
        -------
        bool
            Always ``True``. The provider has no dependency that could fail.
        """
        return True

    async def close(self) -> None:
        """
        Release resources. None are held.

        Returns
        -------
        None
        """
        return None

    # ------------------------------------------------------------------
    # Derivation
    # ------------------------------------------------------------------

    def _derive(self, text: str) -> tuple[float, ...]:
        """
        Derive one unit-normalized vector from a text.

        Digest output is extended by counter until enough bytes exist for the
        configured dimension, so that any dimension is supported without the
        derivation depending on the digest's natural width.

        Parameters
        ----------
        text:
            The text to embed.

        Returns
        -------
        tuple[float, ...]
            A unit-normalized vector of the configured dimension.
        """
        needed = self._dimension * _BYTES_PER_COMPONENT
        material = bytearray()
        counter = 0

        encoded = text.encode("utf-8")
        while len(material) < needed:
            digest = hashlib.blake2b(
                encoded + counter.to_bytes(4, "big"), digest_size=64
            ).digest()
            material.extend(digest)
            counter += 1

        components = [
            (value / _COMPONENT_SCALE) * 2.0 - 1.0
            for value in struct.unpack_from(
                f">{self._dimension}I", bytes(material), 0
            )
        ]

        return _normalize(components)

    def __repr__(self) -> str:
        """
        Return a representation naming the configured dimension.

        Returns
        -------
        str
            Diagnostic representation.
        """
        return f"DeterministicEmbeddingProvider(dimension={self._dimension})"


def _normalize(components: Sequence[float]) -> tuple[float, ...]:
    """
    Scale a vector to unit magnitude.

    A vector of zero magnitude is returned with its first component set, since
    a zero vector cannot be normalized and would fail the normalization claim
    recorded on every embedding. The case is vanishingly unlikely with digest
    material and is handled so that no input can produce an invalid vector.

    Parameters
    ----------
    components:
        The raw components.

    Returns
    -------
    tuple[float, ...]
        A vector of unit magnitude.
    """
    magnitude = math.sqrt(sum(value * value for value in components))

    if magnitude == 0.0:
        return (1.0,) + (0.0,) * (len(components) - 1)

    return tuple(value / magnitude for value in components)
