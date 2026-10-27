"""
PTFM Embedding Stage

File:
    ptfm/core/knowledge/pipeline/embedding.py

Description:
    Converts knowledge nodes into vectors under a named model.

Batch-oriented, with buffering left to the engine
-------------------------------------------------
    This stage embeds a batch it is given. It does not accumulate one.

    Buffering is a streaming concern: how long a partial batch waits, and how
    memory is bounded while it fills, are decisions about the shape of a run
    rather than about embedding. Placing them here would make the stage
    untestable without a clock and a scheduler, and would put two unrelated
    responsibilities behind one interface.

    The engine therefore accumulates nodes to the configured batch size or the
    flush deadline, whichever arrives first, and hands this stage a batch.

What is verified, and why each check exists
-------------------------------------------
    A provider is trusted for the vectors it returns and for nothing else.
    Three properties are checked, because each failure is silent:

        Count and order
            A provider returning results out of order would attach vectors to
            the wrong nodes. Queries would then return text unrelated to the
            match, and no downstream check could detect it.

        Dimension
            A vector of unexpected dimension means the provider is not the
            model it reports itself to be, or that two models have been mixed.
            Either corrupts every comparison in the store.

        Normalization
            A provider reporting normalized vectors while returning
            unnormalized ones makes dot product diverge from cosine
            similarity. Results are still ranked and acted upon, merely
            wrongly. The canonical schema verifies the claim at construction;
            this stage rejects the batch before it reaches that point, so the
            provider rather than the node is named.

Retry
-----
    Only genuinely transient failures are retried, decided by the ``retryable``
    attribute on the raised exception rather than by inspecting types here. A
    dimension mismatch is deterministic and is never retried.

Dependency contract
-------------------
    Imports only the standard library, ``ptfm.core.schemas``, the knowledge
    ``exceptions``, ``config`` and ``interfaces`` modules, and the retry
    contract from the ingestion layer, which is reused rather than redeclared.
"""

from __future__ import annotations

import asyncio
import math
from typing import Final, Optional, Sequence

from ptfm.core.ingestion.interfaces import RetryPolicy
from ptfm.core.knowledge.config import EmbeddingConfig
from ptfm.core.knowledge.exceptions import (
    DimensionMismatchError,
    EmbeddingProviderError,
    EmbeddingTimeoutError,
    ModelIdentityError,
    PTFMKnowledgeError,
)
from ptfm.core.knowledge.interfaces import (
    Clock,
    EmbeddingProvider,
    EmbeddingTextComposer,
    HashProvider,
    ModelIdentity,
)
from ptfm.core.schemas import KnowledgeNode, NodeEmbedding

__all__ = ["EmbeddingStage"]

#: Tolerance within which a vector claiming normalization must have unit
#: magnitude. Matches the canonical schema's tolerance, so a batch this stage
#: accepts cannot then be rejected at construction.
_UNIT_NORM_TOLERANCE: Final[float] = 1e-3


class EmbeddingStage:
    """
    Embeds a batch of knowledge nodes.

    Parameters
    ----------
    provider:
        The embedding provider. Its lifetime is owned by the caller, which
        closes it when the run ends.
    composer:
        The rule composing embedding input from a node's fields.
    hash_provider:
        Used to digest the composed input, so that a vector is reproducible
        and its staleness detectable.
    config:
        Embedding configuration supplying the deadline and retry count.
    clock:
        Supplies the instant recorded on each embedding. Injected so that
        embeddings are deterministic under test.
    retry_policy:
        Supplies the delay before each attempt. When absent, retries proceed
        without delay, which is acceptable for a local provider but risks
        compounding pressure on a rate-limited remote one.
    """

    __slots__ = (
        "_provider",
        "_composer",
        "_hash_provider",
        "_config",
        "_clock",
        "_retry_policy",
        "_identity",
    )

    def __init__(
        self,
        provider: EmbeddingProvider,
        composer: EmbeddingTextComposer,
        hash_provider: HashProvider,
        config: EmbeddingConfig,
        clock: Clock,
        *,
        retry_policy: Optional[RetryPolicy] = None,
    ) -> None:
        self._provider = provider
        self._composer = composer
        self._hash_provider = hash_provider
        self._config = config
        self._clock = clock
        self._retry_policy = retry_policy
        # Resolved once. Identity is recorded on every vector, so a provider
        # changing model mid-run would produce a corpus whose vectors are not
        # mutually comparable; reading it once makes that impossible.
        self._identity = _resolve_identity(provider)

    @property
    def model_identity(self) -> ModelIdentity:
        """
        The model this stage embeds with.

        Returns
        -------
        ModelIdentity
            Identifier, dimension, normalization, and optional revision.
        """
        return self._identity

    @property
    def max_batch_size(self) -> int:
        """
        Largest batch this stage accepts.

        Bounded by both configuration and the provider's own maximum, since a
        provider knows its limits and exceeding them fails every batch rather
        than merely being inefficient.

        Returns
        -------
        int
            The effective batch bound.
        """
        return min(self._config.batch_size, self._provider.max_batch_size)

    async def embed(
        self,
        nodes: Sequence[KnowledgeNode],
    ) -> tuple[NodeEmbedding, ...]:
        """
        Produce a vector for each node in a batch.

        Parameters
        ----------
        nodes:
            The nodes to embed. Must not exceed :attr:`max_batch_size`.

        Returns
        -------
        tuple[NodeEmbedding, ...]
            One embedding per node, in input order.

        Raises
        ------
        InputTooLargeError
            If a node's composed input exceeds the configured bound.
        EmbeddingProviderError
            If the provider failed after every permitted attempt.
        EmbeddingTimeoutError
            If every attempt exceeded its deadline.
        DimensionMismatchError
            If the provider returned a vector of unexpected dimension, or the
            wrong number of vectors.
        """
        if not nodes:
            return ()

        composed = [self._composer.compose(node) for node in nodes]
        vectors = await self._call_provider([item.text for item in composed])

        _verify_batch(vectors, nodes, self._identity, self._provider.name)

        embedded_at = self._clock.now()
        return tuple(
            NodeEmbedding(
                node_id=node.node_id,
                document_id=node.document_id,
                vector=tuple(vector),
                dimension=self._identity.dimension,
                model_id=self._identity.model_id,
                model_revision=self._identity.revision,
                normalized=self._identity.normalized,
                composition_id=item.composition_id,
                # Carried from the provider so that a consumer can verify the
                # provider it queries with encodes queries the same way,
                # rather than inferring that from configuration it may not
                # share.
                query_encoding=self._identity.query_encoding,
                # Digests what was actually submitted, so a vector is
                # reproducible and a stale one detectable: an input hash no
                # longer matching the node's current composition means the
                # vector is out of date.
                input_text_hash=self._hash_provider.hash_text(item.text),
                embedded_at=embedded_at,
            )
            for node, item, vector in zip(nodes, composed, vectors, strict=True)
        )

    # ------------------------------------------------------------------
    # Provider invocation
    # ------------------------------------------------------------------

    async def _call_provider(
        self,
        texts: Sequence[str],
    ) -> Sequence[Sequence[float]]:
        """
        Call the provider, retrying transient failures under a deadline.

        Eligibility for retry is decided by the ``retryable`` attribute on the
        raised exception rather than by inspecting types here, so adding a
        transient failure mode requires no change to this method.

        Parameters
        ----------
        texts:
            The composed inputs.

        Returns
        -------
        Sequence[Sequence[float]]
            The provider's vectors, unvalidated.

        Raises
        ------
        EmbeddingProviderError
            If the provider failed after every permitted attempt.
        EmbeddingTimeoutError
            If every attempt exceeded its deadline.
        PTFMKnowledgeError
            Re-raised where the provider raised a knowledge exception that is
            not retryable.
        """
        attempts = self._config.max_retries
        attempt = 0

        while True:
            attempt += 1
            await self._delay_before(attempt)

            try:
                return await asyncio.wait_for(
                    self._provider.embed(texts),
                    timeout=self._config.timeout_seconds,
                )
            except TimeoutError as exc:
                error: PTFMKnowledgeError = EmbeddingTimeoutError(
                    f"Provider {self._provider.name!r} exceeded the "
                    f"{self._config.timeout_seconds} second deadline while "
                    f"embedding {len(texts)} inputs.",
                    timeout_seconds=self._config.timeout_seconds,
                    batch_size=len(texts),
                )
                error.__cause__ = exc
            except PTFMKnowledgeError as raised:
                error = raised
            except Exception as exc:
                error = EmbeddingProviderError(
                    f"Provider {self._provider.name!r} raised "
                    f"{type(exc).__name__} while embedding "
                    f"{len(texts)} inputs: {exc}",
                    provider_name=self._provider.name,
                    batch_size=len(texts),
                )
                error.__cause__ = exc

            if error.retryable and attempt < attempts:
                continue

            error.add_context(attempts=attempt, max_attempts=attempts)
            raise error

    async def _delay_before(self, attempt: int) -> None:
        """
        Wait the interval an injected policy prescribes before an attempt.

        Without a policy, retries proceed immediately. That is acceptable for
        a local provider, where a failure is unlikely to be load related, and
        risky for a rate-limited remote one, where immediate retry compounds
        the pressure that caused the failure.

        Parameters
        ----------
        attempt:
            One-based index of the attempt about to be made.

        Returns
        -------
        None
        """
        if self._retry_policy is None or attempt <= 1:
            return

        delay = self._retry_policy.delay_for(attempt)
        if delay > 0.0:
            await asyncio.sleep(delay)

    def __repr__(self) -> str:
        """
        Return a representation naming the model and composition in use.

        Returns
        -------
        str
            Diagnostic representation.
        """
        return (
            f"EmbeddingStage(model={self._identity.identity!r}, "
            f"composition={self._composer.composition_id!r})"
        )


# ============================================================================
# VERIFICATION
# ============================================================================


def _resolve_identity(provider: EmbeddingProvider) -> ModelIdentity:
    """
    Read a provider's model identity once, converting any failure.

    Identity is recorded on every vector, so it must be available before the
    first batch. A provider unable to report it cannot be used: a corpus whose
    vectors carry no model attribution becomes an undifferentiated mixture the
    moment a second model is introduced.

    Parameters
    ----------
    provider:
        The provider to interrogate.

    Returns
    -------
    ModelIdentity
        The provider's identity.

    Raises
    ------
    ModelIdentityError
        If the identity is unavailable or malformed.
    """
    try:
        identity = provider.model_identity()
    except PTFMKnowledgeError:
        raise
    except Exception as exc:
        raise ModelIdentityError(
            f"Provider {provider.name!r} could not report its model identity: "
            f"{type(exc).__name__}: {exc}",
            provider_name=provider.name,
        ) from exc

    if not isinstance(identity, ModelIdentity):
        raise ModelIdentityError(
            f"Provider {provider.name!r} reported "
            f"{type(identity).__name__} rather than a ModelIdentity.",
            provider_name=provider.name,
        )

    if not identity.model_id or identity.dimension < 1:
        raise ModelIdentityError(
            f"Provider {provider.name!r} reported an unusable identity: "
            f"model {identity.model_id!r}, dimension {identity.dimension}.",
            provider_name=provider.name,
        )

    return identity


def _verify_batch(
    vectors: Sequence[Sequence[float]],
    nodes: Sequence[KnowledgeNode],
    identity: ModelIdentity,
    provider_name: str,
) -> None:
    """
    Verify a provider's response before any vector is retained.

    Each check guards a failure that would otherwise be silent, corrupting the
    store while every operation continued to report success.

    Parameters
    ----------
    vectors:
        The provider's vectors.
    nodes:
        The nodes submitted, in the same order.
    identity:
        The provider's declared model identity.
    provider_name:
        Identifier of the provider, for diagnostics.

    Returns
    -------
    None

    Raises
    ------
    DimensionMismatchError
        If the count or any dimension is wrong, or a vector claiming
        normalization does not have unit magnitude.
    """
    if len(vectors) != len(nodes):
        raise DimensionMismatchError(
            expected_dimension=len(nodes),
            actual_dimension=len(vectors),
            provider_name=provider_name,
            context={
                "failure": "vector_count",
                "detail": (
                    "The provider returned a different number of vectors than "
                    "inputs, so vectors could not be matched to nodes by "
                    "position."
                ),
            },
        )

    for index, vector in enumerate(vectors):
        if len(vector) != identity.dimension:
            raise DimensionMismatchError(
                expected_dimension=identity.dimension,
                actual_dimension=len(vector),
                provider_name=provider_name,
                node_id=nodes[index].node_id,
                document_id=nodes[index].document_id,
            )

        if not identity.normalized:
            continue

        magnitude = math.sqrt(sum(value * value for value in vector))
        if abs(magnitude - 1.0) > _UNIT_NORM_TOLERANCE:
            raise DimensionMismatchError(
                expected_dimension=identity.dimension,
                actual_dimension=identity.dimension,
                provider_name=provider_name,
                node_id=nodes[index].node_id,
                document_id=nodes[index].document_id,
                context={
                    "failure": "normalization",
                    "magnitude": magnitude,
                    "detail": (
                        "The provider declares normalized vectors but returned "
                        "one of non-unit magnitude, which makes dot product "
                        "diverge from cosine similarity without any visible "
                        "error."
                    ),
                },
            )
