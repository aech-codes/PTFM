"""
PTFM Local Embedding Provider

File:
    ptfm/core/knowledge/embedding/local.py

Description:
    Embeds using a locally executed transformer model, through
    ``sentence-transformers``.

    Requires the ``ptfm[embeddings]`` extra. A deployment that only ingests
    and indexes lexically must not be made to install several gigabytes of
    model tooling, so the dependency is optional and its absence produces an
    actionable message rather than an import error from an unrelated module.

Why sentence-transformers
-------------------------
    It exposes pooling and normalization explicitly. Both are recorded on
    every vector this module produces: ``normalized`` determines whether a dot
    product equals cosine similarity, and a wrong value corrupts every
    comparison silently. A library that normalized implicitly, or that changed
    its default between versions, could not satisfy that requirement.

    Alternatives considered: ``FlagEmbedding``, which is narrower in model
    support, and raw ``transformers``, which would require this module to
    implement pooling itself and so to own a correctness concern the library
    already solves.

The event loop is never blocked
-------------------------------
    Model inference is CPU or GPU bound and takes tens to hundreds of
    milliseconds per batch. Running it inline would stall every other document
    in flight, so it is offloaded to a worker thread.

    Offloading is bounded by a semaphore rather than left to the executor.
    A model holds its weights once, and issuing several concurrent batches
    against one instance contends for the same device without improving
    throughput, while multiplying peak memory by the number of batches in
    flight.

Model identity is resolved once, from the model itself
------------------------------------------------------
    Dimension and normalization are read from the loaded model rather than
    taken from configuration. A configured dimension that disagreed with the
    model would be discovered only when a vector store rejected a write, or
    worse, when two models were silently mixed in one index.

Dependency contract
-------------------
    Imports the knowledge ``exceptions``, ``config`` and ``interfaces``
    modules, and the ingestion ``ThreadOffloader`` contract. The optional
    dependency is imported inside the constructor, so that this module can be
    imported and its availability checked without it.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any, Final, Optional, Sequence

from ptfm.core.knowledge.config import EmbeddingConfig
from ptfm.core.knowledge.exceptions import (
    EmbeddingProviderError,
    ModelIdentityError,
    ProviderUnavailableError,
)
from ptfm.core.knowledge.interfaces import (
    EmbeddingProvider,
    ModelIdentity,
    ThreadOffloader,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from sentence_transformers import SentenceTransformer

__all__ = [
    "LOCAL_PROVIDER_NAME",
    "DEFAULT_MODEL_ID",
    "DEFAULT_MULTILINGUAL_MODEL_ID",
    "SentenceTransformerProvider",
]

#: Selector matching ``EmbeddingConfig.provider``.
LOCAL_PROVIDER_NAME: Final[str] = "local"

#: Default English model. Strong retrieval performance for its size, a
#: permissive licence, and well-understood conventions for query and passage
#: prefixes.
DEFAULT_MODEL_ID: Final[str] = "BAAI/bge-large-en-v1.5"

#: Default multilingual model, for corpora that are not predominantly
#: English. Named here rather than chosen automatically: language detection
#: over a corpus is a judgement an operator should make explicitly.
DEFAULT_MULTILINGUAL_MODEL_ID: Final[str] = "intfloat/multilingual-e5-large"

#: Distribution supplying the model runtime, and the extra that installs it.
_DEPENDENCY: Final[str] = "sentence-transformers"
_INSTALL_EXTRA: Final[str] = "ptfm[embeddings]"


class SentenceTransformerProvider(EmbeddingProvider):
    """
    Embeds using a locally executed transformer model.

    Parameters
    ----------
    config:
        Embedding configuration. Supplies the normalization requirement, which
        is checked against what the model actually does rather than assumed.
    offloader:
        Executes inference off the event loop.
    model_id:
        Model to load. Defaults to :data:`DEFAULT_MODEL_ID`.
    device:
        Device to run on, such as ``"cuda"`` or ``"cpu"``. ``None`` lets the
        library choose.
    batch_size:
        Largest batch submitted to the model. Defaults to the configured batch
        size, since the model imposes no hard bound of its own; the practical
        limit is device memory.
    max_concurrent_batches:
        Batches permitted in flight at once. One by default: a model holds its
        weights once, so concurrent batches contend for the same device
        without improving throughput while multiplying peak memory.
    query_prefix:
        Text prepended to a query before embedding, where the model expects
        one. Applied by :meth:`embed_query`, so that no consumer needs to know
        it: a consumer embeds a query by asking this provider, and verifies
        compatibility through the query encoding recorded on every vector.
    passage_prefix:
        Text prepended to a passage before embedding, where the model expects
        one. Some models are trained with asymmetric prefixes, and omitting
        them degrades retrieval measurably while producing no visible error.

    Raises
    ------
    ProviderUnavailableError
        If ``sentence-transformers`` is not installed.
    EmbeddingProviderError
        If the model could not be loaded.
    ModelIdentityError
        If the model does not report a usable dimension.
    """

    __slots__ = (
        "_config",
        "_offloader",
        "_model_id",
        "_model",
        "_identity",
        "_batch_size",
        "_semaphore",
        "_query_prefix",
        "_passage_prefix",
        "_closed",
    )

    def __init__(
        self,
        config: EmbeddingConfig,
        offloader: ThreadOffloader,
        *,
        model_id: str = DEFAULT_MODEL_ID,
        device: Optional[str] = None,
        batch_size: Optional[int] = None,
        max_concurrent_batches: int = 1,
        query_prefix: str = "",
        passage_prefix: str = "",
    ) -> None:
        model_class = _load_model_class()

        try:
            model = model_class(model_id, device=device)
        except Exception as exc:
            raise EmbeddingProviderError(
                f"Model {model_id!r} could not be loaded: "
                f"{type(exc).__name__}: {exc}",
                provider_name=LOCAL_PROVIDER_NAME,
            ) from exc

        self._config = config
        self._offloader = offloader
        self._model_id = model_id
        self._model = model
        self._batch_size = batch_size or config.batch_size
        self._semaphore = asyncio.Semaphore(max(1, max_concurrent_batches))
        self._query_prefix = query_prefix
        self._passage_prefix = passage_prefix
        self._closed = False
        # Read from the model, not from configuration. A configured dimension
        # disagreeing with the model would surface only when a store rejected
        # a write, or when two models were silently mixed in one index.
        self._identity = _resolve_identity(
            model, model_id, config, _query_encoding(query_prefix)
        )

    @property
    def name(self) -> str:
        """
        Stable identifier of this provider.

        Returns
        -------
        str
            Implementation identifier.
        """
        return LOCAL_PROVIDER_NAME

    @property
    def max_batch_size(self) -> int:
        """
        Largest batch this provider accepts.

        Returns
        -------
        int
            The configured batch bound.
        """
        return self._batch_size

    @property
    def query_prefix(self) -> str:
        """
        Prefix this model expects on a query.

        Applied automatically by :meth:`embed_query`. Exposed for diagnostics
        rather than for a consumer to apply itself, since a consumer applying
        it would be reproducing provider behaviour the provider already owns.

        Returns
        -------
        str
            The prefix, empty when the model expects none.
        """
        return self._query_prefix

    async def embed_query(self, query: str) -> tuple[float, ...]:
        """
        Embed a query, applying whatever transformation the model expects.

        The query prefix is applied here rather than by a consumer. A model
        trained with asymmetric prefixes retrieves measurably worse without
        them, and the degradation produces no error at all, so the knowledge
        of which prefix a model wants must sit with the component that loaded
        the model.

        Parameters
        ----------
        query:
            The query text.

        Returns
        -------
        tuple[float, ...]
            A vector comparable with those produced by :meth:`embed`.

        Raises
        ------
        EmbeddingProviderError
            If the provider is closed, or inference failed.
        """
        if self._closed:
            raise EmbeddingProviderError(
                "Provider is closed and can no longer embed.",
                provider_name=self.name,
            )

        async with self._semaphore:
            try:
                vectors = await self._offloader.run(
                    self._encode, [f"{self._query_prefix}{query}"]
                )
            except Exception as exc:
                raise EmbeddingProviderError(
                    f"Model {self._model_id!r} failed while embedding a "
                    f"query: {type(exc).__name__}: {exc}",
                    provider_name=self.name,
                    batch_size=1,
                ) from exc

        return vectors[0]

    def model_identity(self) -> ModelIdentity:
        """
        Report the model this provider embeds with.

        Resolved once at construction from the loaded model.

        Returns
        -------
        ModelIdentity
            Identifier, dimension, normalization, and input bound.
        """
        return self._identity

    async def embed(self, texts: Sequence[str]) -> Sequence[tuple[float, ...]]:
        """
        Embed a batch of texts.

        Inference is offloaded to a worker thread, so the event loop continues
        serving other documents while the model runs.

        Parameters
        ----------
        texts:
            The inputs to embed.

        Returns
        -------
        Sequence[tuple[float, ...]]
            One vector per input, in input order.

        Raises
        ------
        EmbeddingProviderError
            If the provider is closed, or inference failed.
        """
        if self._closed:
            raise EmbeddingProviderError(
                "Provider is closed and can no longer embed.",
                provider_name=self.name,
                batch_size=len(texts),
            )

        if not texts:
            return ()

        prepared = [f"{self._passage_prefix}{text}" for text in texts]

        async with self._semaphore:
            try:
                vectors = await self._offloader.run(self._encode, prepared)
            except Exception as exc:
                raise EmbeddingProviderError(
                    f"Model {self._model_id!r} failed while embedding "
                    f"{len(texts)} inputs: {type(exc).__name__}: {exc}",
                    provider_name=self.name,
                    batch_size=len(texts),
                ) from exc

        return vectors

    def _encode(self, texts: Sequence[str]) -> tuple[tuple[float, ...], ...]:
        """
        Run inference. Blocking; executed in a worker thread.

        Normalization is requested explicitly rather than left to the model's
        default, so that the ``normalized`` flag recorded on every vector is
        true by construction rather than by assumption.

        Parameters
        ----------
        texts:
            The prepared inputs.

        Returns
        -------
        tuple[tuple[float, ...], ...]
            One vector per input, in input order.
        """
        encoded = self._model.encode(
            list(texts),
            batch_size=self._batch_size,
            normalize_embeddings=self._identity.normalized,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return tuple(tuple(float(value) for value in row) for row in encoded)

    async def health_check(self) -> bool:
        """
        Verify that the model can embed.

        Runs one trivial inference rather than inspecting attributes, since a
        model can be loaded and still fail on its first batch, and discovering
        that after a corpus has been segmented is expensive.

        Returns
        -------
        bool
            ``True`` when the model produced a vector of the expected
            dimension.
        """
        if self._closed:
            return False

        try:
            vectors = await self.embed(["health check"])
        except Exception:  # noqa: BLE001 - a health check reports, never raises
            return False

        return bool(vectors) and len(vectors[0]) == self._identity.dimension

    async def close(self) -> None:
        """
        Release the model. Idempotent.

        A loaded model holds substantial memory, and on a device it holds
        allocations that no other process can use until released.

        Returns
        -------
        None
        """
        if self._closed:
            return

        self._closed = True
        self._model = None  # type: ignore[assignment]

    def __repr__(self) -> str:
        """
        Return a representation naming the model in use.

        Returns
        -------
        str
            Diagnostic representation.
        """
        return (
            f"SentenceTransformerProvider(model={self._model_id!r}, "
            f"dimension={self._identity.dimension})"
        )


# ============================================================================
# HELPERS
# ============================================================================


def _load_model_class() -> type[SentenceTransformer]:
    """
    Import the model class, converting absence into an actionable failure.

    Imported here rather than at module scope so that this module can be
    imported, and a registry can report the provider as unavailable, without
    the dependency installed.

    Returns
    -------
    type[SentenceTransformer]
        The model class.

    Raises
    ------
    ProviderUnavailableError
        If the dependency is not installed.
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise ProviderUnavailableError(
            component=f"embedding provider {LOCAL_PROVIDER_NAME!r}",
            missing_dependency=_DEPENDENCY,
            install_extra=_INSTALL_EXTRA,
        ) from exc

    return SentenceTransformer


def _resolve_identity(
    model: Any,
    model_id: str,
    config: EmbeddingConfig,
    query_encoding: str,
) -> ModelIdentity:
    """
    Read a model's identity from the model itself.

    Parameters
    ----------
    model:
        The loaded model.
    model_id:
        Identifier the model was loaded under.
    config:
        Embedding configuration, supplying the normalization requirement and
        the input bound.
    query_encoding:
        Identifier of this provider's query convention.

    Returns
    -------
    ModelIdentity
        The resolved identity.

    Raises
    ------
    ModelIdentityError
        If the model does not report a usable dimension.
    """
    try:
        dimension = model.get_sentence_embedding_dimension()
    except Exception as exc:
        raise ModelIdentityError(
            f"Model {model_id!r} did not report an embedding dimension: "
            f"{type(exc).__name__}: {exc}",
            provider_name=LOCAL_PROVIDER_NAME,
        ) from exc

    if not isinstance(dimension, int) or dimension < 1:
        raise ModelIdentityError(
            f"Model {model_id!r} reported an unusable embedding dimension: "
            f"{dimension!r}.",
            provider_name=LOCAL_PROVIDER_NAME,
        )

    return ModelIdentity(
        model_id=model_id,
        dimension=dimension,
        # Normalization is requested at inference, so the model normalizes
        # exactly when configuration requires it and the recorded flag is true
        # by construction.
        normalized=config.require_normalized,
        revision=_model_revision(model),
        max_input_chars=config.max_input_chars,
        query_encoding=query_encoding,
    )


def _query_encoding(query_prefix: str) -> str:
    """
    Name this provider's query convention.

    Derived from the prefix rather than configured separately, so the recorded
    convention cannot disagree with the transformation actually applied. A
    provider whose query side is indistinguishable from its passage side
    reports an empty convention, which is what a symmetric model is.

    Parameters
    ----------
    query_prefix:
        Text this provider prepends to a query.

    Returns
    -------
    str
        A stable identifier for the convention, empty when symmetric.
    """
    if not query_prefix:
        return ""
    return f"prefix:{query_prefix}"


def _model_revision(model: Any) -> Optional[str]:
    """
    Recover a model's revision, where the library exposes one.

    A model identifier alone is not always sufficient, since weights behind a
    name can change. Where no revision is available the field stays empty
    rather than being invented, since a fabricated revision would be worse
    than an absent one.

    Parameters
    ----------
    model:
        The loaded model.

    Returns
    -------
    Optional[str]
        The revision, or ``None``.
    """
    for attribute in ("model_card_data", "_model_config"):
        container = getattr(model, attribute, None)
        revision = getattr(container, "base_model_revision", None) or (
            container.get("revision") if isinstance(container, dict) else None
        )
        if isinstance(revision, str) and revision:
            return revision

    return None
