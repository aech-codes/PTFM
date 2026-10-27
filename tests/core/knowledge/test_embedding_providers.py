"""
Unit tests for the built-in and local embedding providers.

File:
    tests/core/knowledge/test_embedding_providers.py
"""

from __future__ import annotations

import inspect
import math
from typing import Any, Sequence

import pytest

from ptfm.core.knowledge.config import EmbeddingConfig
from ptfm.core.knowledge.embedding import builtin as builtin_module
from ptfm.core.knowledge.embedding import local as local_module
from ptfm.core.knowledge.embedding.builtin import (
    DeterministicEmbeddingProvider,
    NullEmbeddingProvider,
)
from ptfm.core.knowledge.embedding.local import (
    DEFAULT_MODEL_ID,
    SentenceTransformerProvider,
)
from ptfm.core.knowledge.exceptions import (
    EmbeddingProviderError,
    ModelIdentityError,
    ProviderUnavailableError,
)
from ptfm.core.knowledge.interfaces import EmbeddingProvider, ModelIdentity


def _config(**overrides: Any) -> EmbeddingConfig:
    return EmbeddingConfig(**overrides)


def _magnitude(vector: Sequence[float]) -> float:
    return math.sqrt(sum(value * value for value in vector))


# ===========================================================================
# CONTRACT CONFORMANCE
#
# Every provider must pass these. A provider defect is otherwise silent: a
# wrong dimension or a false normalization claim corrupts every comparison in
# the store while every operation continues to report success.
# ===========================================================================


def _conforming_providers() -> list[EmbeddingProvider]:
    return [DeterministicEmbeddingProvider(_config(), dimension=16)]


@pytest.mark.parametrize("provider", _conforming_providers())
async def test_a_provider_returns_one_vector_per_input(
    provider: EmbeddingProvider,
) -> None:
    vectors = await provider.embed(["alpha", "bravo", "charlie"])

    assert len(vectors) == 3


@pytest.mark.parametrize("provider", _conforming_providers())
async def test_a_provider_honours_its_declared_dimension(
    provider: EmbeddingProvider,
) -> None:
    identity = provider.model_identity()

    vectors = await provider.embed(["alpha", "bravo"])

    assert all(len(vector) == identity.dimension for vector in vectors)


@pytest.mark.parametrize("provider", _conforming_providers())
async def test_a_normalization_claim_is_truthful(
    provider: EmbeddingProvider,
) -> None:
    """A false claim makes dot product diverge from cosine similarity and
    corrupts every comparison invisibly."""
    identity = provider.model_identity()

    vectors = await provider.embed(["alpha", "bravo"])

    if identity.normalized:
        assert all(
            math.isclose(_magnitude(vector), 1.0, abs_tol=1e-6)
            for vector in vectors
        )


@pytest.mark.parametrize("provider", _conforming_providers())
async def test_batching_does_not_change_a_vector(
    provider: EmbeddingProvider,
) -> None:
    """
    A provider whose output depended on batch composition would produce a
    corpus whose vectors differed by how it happened to be chunked into
    batches.
    """
    together = await provider.embed(["alpha", "bravo"])
    alone = await provider.embed(["alpha"])

    assert together[0] == alone[0]


@pytest.mark.parametrize("provider", _conforming_providers())
async def test_an_empty_batch_returns_nothing(
    provider: EmbeddingProvider,
) -> None:
    assert await provider.embed([]) == ()


@pytest.mark.parametrize("provider", _conforming_providers())
async def test_a_provider_declares_a_usable_batch_bound(
    provider: EmbeddingProvider,
) -> None:
    assert provider.max_batch_size >= 1



@pytest.mark.parametrize("provider", _conforming_providers())
async def test_a_provider_embeds_a_query(provider: EmbeddingProvider) -> None:
    """
    The query side is the provider's own responsibility, so every provider
    must implement it rather than leaving a consumer to reconstruct it.
    """
    vector = await provider.embed_query("what is the rate")

    assert len(vector) == provider.model_identity().dimension


@pytest.mark.parametrize("provider", _conforming_providers())
async def test_a_symmetric_provider_encodes_both_sides_alike(
    provider: EmbeddingProvider,
) -> None:
    """A symmetric model declares an empty convention, and must behave
    accordingly."""
    if provider.model_identity().query_encoding:
        pytest.skip("provider is asymmetric")

    passage = (await provider.embed(["identical text"]))[0]
    query = await provider.embed_query("identical text")

    assert passage == query


@pytest.mark.parametrize("provider", _conforming_providers())
def test_a_provider_reports_a_stable_identity(
    provider: EmbeddingProvider,
) -> None:
    """Identity is recorded on every vector, so a provider changing model
    mid-run would produce a corpus whose vectors are not comparable."""
    assert provider.model_identity() == provider.model_identity()


@pytest.mark.parametrize("provider", _conforming_providers())
def test_a_provider_reports_a_usable_identity(
    provider: EmbeddingProvider,
) -> None:
    identity = provider.model_identity()

    assert isinstance(identity, ModelIdentity)
    assert identity.model_id
    assert identity.dimension >= 1


@pytest.mark.parametrize("provider", _conforming_providers())
async def test_a_provider_closes_idempotently(
    provider: EmbeddingProvider,
) -> None:
    await provider.close()
    await provider.close()


# ===========================================================================
# NULL PROVIDER
# ===========================================================================


async def test_the_null_provider_refuses_to_embed() -> None:
    """
    Returning zero vectors would place every node at an identical distance
    from every query, corrupting retrieval silently. Failing loudly means the
    misconfiguration surfaces at the first batch.
    """
    with pytest.raises(EmbeddingProviderError, match="equidistant"):
        await NullEmbeddingProvider(_config()).embed(["alpha"])


async def test_the_null_provider_reports_itself_unhealthy() -> None:
    """Reporting otherwise would let a run begin that must fail at its first
    batch."""
    assert await NullEmbeddingProvider(_config()).health_check() is False


def test_the_null_provider_satisfies_the_contract() -> None:
    """It exists so the composition root has no conditional path."""
    provider = NullEmbeddingProvider(_config())

    assert provider.name == "null"
    assert provider.model_identity().dimension >= 1


async def test_the_null_provider_closes_cleanly() -> None:
    provider = NullEmbeddingProvider(_config())

    await provider.close()
    await provider.close()


# ===========================================================================
# DETERMINISTIC PROVIDER
# ===========================================================================


async def test_identical_text_yields_identical_vectors() -> None:
    """
    The one embedding property that is architecturally significant: a vector
    store can be rebuilt without reprocessing the corpus.
    """
    first = DeterministicEmbeddingProvider(_config(), dimension=32)
    second = DeterministicEmbeddingProvider(_config(), dimension=32)

    assert await first.embed(["alpha"]) == await second.embed(["alpha"])


async def test_different_text_yields_different_vectors() -> None:
    provider = DeterministicEmbeddingProvider(_config(), dimension=32)

    vectors = await provider.embed(["alpha", "bravo"])

    assert vectors[0] != vectors[1]


async def test_vectors_are_unit_normalized() -> None:
    provider = DeterministicEmbeddingProvider(_config(), dimension=64)

    vectors = await provider.embed(["alpha", "bravo", "charlie"])

    for vector in vectors:
        assert math.isclose(_magnitude(vector), 1.0, abs_tol=1e-9)


@pytest.mark.parametrize("dimension", [1, 3, 16, 384, 1024])
async def test_any_dimension_is_supported(dimension: int) -> None:
    """Digest output is extended by counter, so the derivation does not depend
    on the digest's natural width."""
    provider = DeterministicEmbeddingProvider(_config(), dimension=dimension)

    vectors = await provider.embed(["alpha"])

    assert len(vectors[0]) == dimension
    assert math.isclose(_magnitude(vectors[0]), 1.0, abs_tol=1e-9)


async def test_components_fall_within_the_unit_range() -> None:
    provider = DeterministicEmbeddingProvider(_config(), dimension=128)

    vector = (await provider.embed(["alpha"]))[0]

    assert all(-1.0 <= value <= 1.0 for value in vector)


async def test_an_empty_string_embeds_successfully() -> None:
    provider = DeterministicEmbeddingProvider(_config(), dimension=16)

    vectors = await provider.embed([""])

    assert math.isclose(_magnitude(vectors[0]), 1.0, abs_tol=1e-9)


async def test_unicode_text_embeds_successfully() -> None:
    provider = DeterministicEmbeddingProvider(_config(), dimension=16)

    vectors = await provider.embed(["日本語のテキスト", "Ελληνικά"])

    assert len(vectors) == 2
    assert vectors[0] != vectors[1]


def test_an_unusable_dimension_is_rejected() -> None:
    with pytest.raises(EmbeddingProviderError, match="at least one"):
        DeterministicEmbeddingProvider(_config(), dimension=0)


def test_the_identity_states_that_vectors_are_derived() -> None:
    """
    A corpus embedded during development must not later be mistaken for one
    embedded by a real model.
    """
    identity = DeterministicEmbeddingProvider(_config()).model_identity()

    assert "deterministic" in identity.model_id
    assert identity.revision is not None


def test_the_production_warning_is_prominent() -> None:
    """The name alone does not convey that the vectors carry no semantics."""
    doc = DeterministicEmbeddingProvider.__doc__ or ""

    assert "NOT FOR PRODUCTION USE" in doc


async def test_the_deterministic_provider_reports_itself_healthy() -> None:
    assert await DeterministicEmbeddingProvider(_config()).health_check() is True


# ===========================================================================
# LOCAL PROVIDER
# ===========================================================================


class _FakeModel:
    """Stands in for a loaded transformer."""

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        device: Any = None,
        *,
        dimension: int = 8,
        fail: bool = False,
    ) -> None:
        self.model_id = model_id
        self.device = device
        self._dimension = dimension
        self._fail = fail
        self.calls: list[dict[str, Any]] = []

    def get_sentence_embedding_dimension(self) -> int:
        return self._dimension

    def encode(self, texts: list[str], **kwargs: Any) -> list[list[float]]:
        self.calls.append({"texts": list(texts), **kwargs})
        if self._fail:
            raise RuntimeError("CUDA out of memory")
        unit = 1.0 / math.sqrt(self._dimension)
        return [[unit] * self._dimension for _ in texts]


class _InlineOffloader:
    async def run(self, func: Any, *args: Any, **kwargs: Any) -> Any:
        return func(*args, **kwargs)


def _local(
    monkeypatch: pytest.MonkeyPatch,
    *,
    model: Any = None,
    **kwargs: Any,
) -> SentenceTransformerProvider:
    built = model or _FakeModel()
    monkeypatch.setattr(
        local_module, "_load_model_class", lambda: (lambda *a, **k: built)
    )
    return SentenceTransformerProvider(
        _config(), _InlineOffloader(), **kwargs
    )


def test_a_missing_dependency_names_a_remedy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    A deployment that only indexes lexically must not be made to install
    several gigabytes of model tooling.
    """
    def _absent() -> Any:
        raise ProviderUnavailableError(
            component="embedding provider 'local'",
            missing_dependency="sentence-transformers",
            install_extra="ptfm[embeddings]",
        )

    monkeypatch.setattr(local_module, "_load_model_class", _absent)

    with pytest.raises(ProviderUnavailableError) as excinfo:
        SentenceTransformerProvider(_config(), _InlineOffloader())

    assert excinfo.value.install_extra == "ptfm[embeddings]"


def test_the_dependency_is_imported_lazily() -> None:
    """The module must import, and a registry must report availability,
    without the dependency installed."""
    source = inspect.getsource(local_module)
    module_level = source.split("def _load_model_class")[0]

    assert "from sentence_transformers import" not in module_level.replace(
        "if TYPE_CHECKING", "#"
    ).split("__all__")[0].replace("    from sentence_transformers", "")


def test_dimension_is_read_from_the_model_not_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    A configured dimension disagreeing with the model would surface only when
    a store rejected a write, or when two models were silently mixed.
    """
    provider = _local(monkeypatch, model=_FakeModel(dimension=384))

    assert provider.model_identity().dimension == 384


def test_an_unusable_model_dimension_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Bad(_FakeModel):
        def get_sentence_embedding_dimension(self) -> Any:
            return None

    with pytest.raises(ModelIdentityError, match="unusable"):
        _local(monkeypatch, model=_Bad())


def test_a_failing_model_load_is_converted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _explode(*args: Any, **kwargs: Any) -> Any:
        raise OSError("model files not found")

    monkeypatch.setattr(local_module, "_load_model_class", lambda: _explode)

    with pytest.raises(EmbeddingProviderError, match="could not be loaded"):
        SentenceTransformerProvider(_config(), _InlineOffloader())


async def test_inference_is_offloaded(monkeypatch: pytest.MonkeyPatch) -> None:
    """Inference takes tens to hundreds of milliseconds; running it inline
    would stall every other document in flight."""
    calls: list[str] = []

    class _RecordingOffloader:
        async def run(self, func: Any, *args: Any, **kwargs: Any) -> Any:
            calls.append(func.__name__)
            return func(*args, **kwargs)

    model = _FakeModel()
    monkeypatch.setattr(
        local_module, "_load_model_class", lambda: (lambda *a, **k: model)
    )
    provider = SentenceTransformerProvider(_config(), _RecordingOffloader())

    await provider.embed(["alpha"])

    assert calls == ["_encode"]


async def test_normalization_is_requested_explicitly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The recorded flag must be true by construction, not by assumption."""
    model = _FakeModel()
    provider = _local(monkeypatch, model=model)

    await provider.embed(["alpha"])

    assert model.calls[0]["normalize_embeddings"] is True


async def test_normalization_can_be_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _FakeModel()
    monkeypatch.setattr(
        local_module, "_load_model_class", lambda: (lambda *a, **k: model)
    )
    provider = SentenceTransformerProvider(
        _config(require_normalized=False), _InlineOffloader()
    )

    await provider.embed(["alpha"])

    assert model.calls[0]["normalize_embeddings"] is False
    assert provider.model_identity().normalized is False


async def test_a_passage_prefix_is_applied(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    A model trained with asymmetric prefixes retrieves measurably worse
    without them, and the degradation is invisible.
    """
    model = _FakeModel()
    provider = _local(monkeypatch, model=model, passage_prefix="passage: ")

    await provider.embed(["alpha"])

    assert model.calls[0]["texts"] == ["passage: alpha"]


def test_the_query_prefix_is_exposed_for_retrieval(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Retrieval must obtain it from the same place the model was
    configured."""
    provider = _local(monkeypatch, query_prefix="query: ")

    assert provider.query_prefix == "query: "


async def test_inference_failure_is_converted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = _local(monkeypatch, model=_FakeModel(fail=True))

    with pytest.raises(EmbeddingProviderError) as excinfo:
        await provider.embed(["alpha"])

    assert isinstance(excinfo.value.__cause__, RuntimeError)
    assert excinfo.value.retryable is True


async def test_a_health_check_runs_real_inference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A model can load and still fail on its first batch, and discovering
    that after segmentation is expensive."""
    model = _FakeModel()
    provider = _local(monkeypatch, model=model)

    assert await provider.health_check() is True
    assert model.calls


async def test_a_health_check_reports_rather_than_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = _local(monkeypatch, model=_FakeModel(fail=True))

    assert await provider.health_check() is False


async def test_a_closed_provider_refuses_to_embed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = _local(monkeypatch)

    await provider.close()

    with pytest.raises(EmbeddingProviderError, match="closed"):
        await provider.embed(["alpha"])


async def test_closing_releases_the_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A loaded model holds allocations no other process can use until
    released."""
    provider = _local(monkeypatch)

    await provider.close()
    await provider.close()

    assert await provider.health_check() is False


async def test_an_empty_batch_skips_inference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _FakeModel()
    provider = _local(monkeypatch, model=model)

    assert await provider.embed([]) == ()
    assert model.calls == []


async def test_concurrent_batches_are_bounded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    A model holds its weights once, so concurrent batches contend for the same
    device without improving throughput while multiplying peak memory.
    """
    import asyncio

    active = 0
    peak = 0

    class _Tracking(_FakeModel):
        def encode(self, texts: list[str], **kwargs: Any) -> list[list[float]]:
            nonlocal active, peak
            active += 1
            peak = max(peak, active)
            try:
                return super().encode(texts, **kwargs)
            finally:
                active -= 1

    class _YieldingOffloader:
        async def run(self, func: Any, *args: Any, **kwargs: Any) -> Any:
            await asyncio.sleep(0)
            return func(*args, **kwargs)

    model = _Tracking()
    monkeypatch.setattr(
        local_module, "_load_model_class", lambda: (lambda *a, **k: model)
    )
    provider = SentenceTransformerProvider(
        _config(), _YieldingOffloader(), max_concurrent_batches=1
    )

    await asyncio.gather(*(provider.embed([f"text {n}"]) for n in range(8)))

    assert peak == 1



async def test_the_provider_applies_the_query_prefix_itself(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    The knowledge of which prefix a model wants must sit with the component
    that loaded the model, not with a consumer reconstructing it.
    """
    model = _FakeModel()
    provider = _local(monkeypatch, model=model, query_prefix="query: ")

    await provider.embed_query("what is the rate")

    assert model.calls[0]["texts"] == ["query: what is the rate"]


def test_the_query_convention_is_derived_from_the_prefix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Derived rather than configured separately, so the recorded convention
    cannot disagree with the transformation actually applied.
    """
    asymmetric = _local(monkeypatch, query_prefix="query: ")
    symmetric = _local(monkeypatch)

    assert asymmetric.model_identity().query_encoding == "prefix:query: "
    assert symmetric.model_identity().query_encoding == ""


async def test_a_closed_provider_refuses_to_embed_a_query(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = _local(monkeypatch)

    await provider.close()

    with pytest.raises(EmbeddingProviderError, match="closed"):
        await provider.embed_query("anything")


# ===========================================================================
# Anti-pattern guards
# ===========================================================================


@pytest.mark.parametrize("module", [builtin_module, local_module])
def test_no_provider_imports_a_concrete_pipeline_component(
    module: Any,
) -> None:
    source = inspect.getsource(module)

    for forbidden in (
        "from ptfm.core.knowledge.pipeline",
        "from ptfm.core.knowledge.registry",
        "from ptfm.plugins",
    ):
        assert forbidden not in source


@pytest.mark.parametrize("module", [builtin_module, local_module])
def test_no_provider_declares_mutable_module_state(module: Any) -> None:
    mutable = [
        name
        for name, value in vars(module).items()
        if not name.startswith("_") and isinstance(value, (list, dict, set))
    ]

    assert mutable == []


@pytest.mark.parametrize("module", [builtin_module, local_module])
def test_no_provider_produces_a_confidence_value(module: Any) -> None:
    """A provider produces vectors. Similarity is computed at query time and
    calibrated downstream."""
    source = inspect.getsource(module).lower()

    assert "confidence" not in source
    assert "def similarity" not in source


@pytest.mark.parametrize("module", [builtin_module, local_module])
def test_no_provider_contains_domain_vocabulary(module: Any) -> None:
    source = inspect.getsource(module).lower()

    for term in ("jurisdiction", "statute", "ticker", "icd", "patient", "court"):
        assert term not in source


def test_the_builtin_module_needs_no_optional_dependency() -> None:
    """It must work in a deployment that installs no model tooling."""
    source = inspect.getsource(builtin_module)

    for forbidden in ("sentence_transformers", "torch", "numpy", "transformers"):
        assert forbidden not in source
