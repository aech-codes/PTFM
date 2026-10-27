"""
Unit tests for the Knowledge Processing contract layer.

File:
    tests/core/knowledge/test_interfaces.py
"""

from __future__ import annotations

import inspect
from typing import Any, Mapping, Sequence

import pytest

from ptfm.core.knowledge import interfaces as interfaces_module
from ptfm.core.knowledge.exceptions import (
    ChunkerRegistrationError,
    EmbedderRegistrationError,
)
from ptfm.core.knowledge.interfaces import (
    ChunkingStrategy,
    Clock,
    ComposedInput,
    EmbeddingProvider,
    EmbeddingTextComposer,
    HashProvider,
    KnowledgeEnricher,
    KnowledgeSink,
    ModelIdentity,
    Sentence,
    SentenceSegmenter,
    ThreadOffloader,
)


# ---------------------------------------------------------------------------
# Conforming reference implementations
#
# Model and schema types are referenced only in annotations, which are strings
# under 'from __future__ import annotations'. The contract layer therefore has
# no runtime dependency on either, and these fakes need no model instances.
# ---------------------------------------------------------------------------


class _Strategy(ChunkingStrategy):
    """Minimal strategy satisfying the full contract."""

    @property
    def name(self) -> str:
        return "TestStrategy"

    def plan(self, document: Any) -> Any:
        return ("plan", document)

    def fingerprint_components(self) -> Mapping[str, str]:
        return {"strategy": self.name, "max_chunk_chars": "1000"}


class _Provider(EmbeddingProvider):
    """Minimal provider satisfying the full contract."""

    def __init__(self) -> None:
        self.closed = 0

    @property
    def name(self) -> str:
        return "TestProvider"

    @property
    def max_batch_size(self) -> int:
        return 32

    def model_identity(self) -> ModelIdentity:
        return ModelIdentity(
            model_id="test/model", dimension=4, normalized=True, revision="r1"
        )

    async def embed(self, texts: Sequence[str]) -> Sequence[tuple[float, ...]]:
        return [(0.5, 0.5, 0.5, 0.5) for _ in texts]

    async def embed_query(self, query: str) -> tuple[float, ...]:
        return (0.5, 0.5, 0.5, 0.5)

    async def health_check(self) -> bool:
        return True

    async def close(self) -> None:
        self.closed += 1


class _Segmenter:
    @property
    def version(self) -> str:
        return "test-segmenter-1.0"

    def segment(self, text: str) -> Sequence[Sentence]:
        return [Sentence(start=0, end=len(text))] if text else []


class _Composer:
    @property
    def composition_id(self) -> str:
        return "heading-path-prefixed"

    def compose(self, node: Any) -> ComposedInput:
        return ComposedInput(text="composed", composition_id=self.composition_id)


class _Enricher:
    @property
    def name(self) -> str:
        return "TestEnricher"

    @property
    def metadata_keys(self) -> frozenset[str]:
        return frozenset({"topic"})

    async def enrich(self, node: Any) -> Mapping[str, str]:
        return {"topic": "finance"}


class _Sink:
    def __init__(self) -> None:
        self.written: list[Any] = []
        self.flushed = 0
        self.closed = 0

    @property
    def name(self) -> str:
        return "TestSink"

    async def write(self, unit: Any) -> None:
        self.written.append(unit)

    async def flush(self) -> None:
        self.flushed += 1

    async def close(self) -> None:
        self.closed += 1


# ---------------------------------------------------------------------------
# Machine-enforced execution contract
# ---------------------------------------------------------------------------


def test_an_async_chunking_strategy_is_rejected_at_definition() -> None:
    """
    Planning is CPU-bound and offloaded. An async signature would conceal
    blocking work on the event loop.
    """
    with pytest.raises(ChunkerRegistrationError, match="must be synchronous"):

        class _AsyncStrategy(ChunkingStrategy):
            @property
            def name(self) -> str:
                return "AsyncStrategy"

            async def plan(self, document: Any) -> Any:  # type: ignore[override]
                return document


def test_an_async_generator_strategy_is_rejected() -> None:
    with pytest.raises(ChunkerRegistrationError):

        class _StreamingStrategy(ChunkingStrategy):
            @property
            def name(self) -> str:
                return "StreamingStrategy"

            async def plan(self, document: Any) -> Any:  # type: ignore[override]
                yield document


@pytest.mark.parametrize(
    "method", ["embed", "embed_query", "health_check", "close"]
)
def test_a_synchronous_provider_method_is_rejected(method: str) -> None:
    """Every provider crosses a process or network boundary."""
    namespace: dict[str, Any] = {method: lambda self, *a, **k: None}

    with pytest.raises(EmbedderRegistrationError, match="coroutine function"):
        type("_BlockingProvider", (_Provider,), namespace)


def test_conforming_implementations_define_cleanly() -> None:
    assert _Strategy().name == "TestStrategy"
    assert _Provider().name == "TestProvider"


def test_the_base_declarations_match_their_own_contract() -> None:
    assert not inspect.iscoroutinefunction(ChunkingStrategy.plan)
    assert inspect.iscoroutinefunction(EmbeddingProvider.embed)
    assert inspect.iscoroutinefunction(EmbeddingProvider.close)


# ---------------------------------------------------------------------------
# Provenance protection
# ---------------------------------------------------------------------------


def test_a_strategy_returns_a_plan_not_nodes() -> None:
    """
    A strategy names spans; assembly enforces the invariants. A strategy
    therefore has no means of producing a node whose text and span disagree.
    """
    signature = inspect.signature(ChunkingStrategy.plan)

    assert signature.return_annotation == "ChunkPlan"
    assert "KnowledgeNode" not in str(signature)


def test_an_enricher_returns_metadata_only() -> None:
    """
    Structurally incapable of altering provenance: it receives an immutable
    node and its return value cannot reach text, span, or identity.
    """
    signature = inspect.signature(KnowledgeEnricher.enrich)

    assert signature.return_annotation == "Mapping[str, str]"
    assert "KnowledgeNode" not in signature.return_annotation


def test_an_enricher_declares_the_keys_it_may_contribute() -> None:
    """
    Two enrichers claiming one key must be detected at configuration, not by
    one silently overwriting the other partway through a corpus.
    """
    assert _Enricher().metadata_keys == frozenset({"topic"})


async def test_an_enricher_produces_string_values() -> None:
    result = await _Enricher().enrich(node=None)

    assert all(isinstance(value, str) for value in result.values())


def test_composition_is_separate_from_the_stored_text() -> None:
    """
    Structural context reaches the vector without reaching the node's text,
    which must remain an exact substring of the source document.
    """
    composed = _Composer().compose(node=None)

    assert isinstance(composed, ComposedInput)
    assert composed.composition_id == "heading-path-prefixed"


# ---------------------------------------------------------------------------
# Segmentation fingerprint
# ---------------------------------------------------------------------------


def test_a_strategy_reports_its_boundary_affecting_parameters() -> None:
    components = _Strategy().fingerprint_components()

    assert "strategy" in components
    assert all(isinstance(value, str) for value in components.values())


def test_fingerprint_values_are_strings() -> None:
    """A digest must not depend on numeric formatting across processes."""
    signature = inspect.signature(ChunkingStrategy.fingerprint_components)

    assert signature.return_annotation == "Mapping[str, str]"


def test_the_inclusion_rule_is_documented_on_the_contract() -> None:
    """
    The rule governs decisions by contributors who did not participate in the
    design discussion, so it must be legible from the method itself.
    """
    doc = ChunkingStrategy.fingerprint_components.__doc__ or ""

    assert "IF AND ONLY IF" in doc
    assert "Embedding models, enrichers" in doc


def test_a_segmenter_reports_its_version() -> None:
    """An upgrade shifts boundaries within oversized blocks, so it changes
    node identity and must be detectable."""
    assert _Segmenter().version == "test-segmenter-1.0"


# ---------------------------------------------------------------------------
# Sentence offsets
# ---------------------------------------------------------------------------


def test_a_sentence_carries_offsets_not_only_text() -> None:
    """
    Recovering an offset by searching for a returned string is fragile and
    wrong wherever the string occurs more than once.
    """
    assert set(Sentence._fields) == {"start", "end"}
    assert "text" not in Sentence._fields


def test_sentence_length_is_derived() -> None:
    assert Sentence(start=10, end=25).length == 15


def test_a_segmenter_returns_located_sentences() -> None:
    sentences = _Segmenter().segment("A sentence.")

    assert sentences[0].start == 0
    assert sentences[0].end == 11


def test_an_unsplittable_text_yields_no_sentence() -> None:
    assert _Segmenter().segment("") == []


# ---------------------------------------------------------------------------
# Model identity
# ---------------------------------------------------------------------------


def test_model_identity_combines_identifier_and_revision() -> None:
    identity = ModelIdentity(
        model_id="BAAI/bge-large-en-v1.5",
        dimension=1024,
        normalized=True,
        revision="a1b2c3",
    )

    assert identity.identity == "BAAI/bge-large-en-v1.5@a1b2c3"


def test_model_identity_falls_back_to_the_bare_identifier() -> None:
    identity = ModelIdentity(model_id="m", dimension=8, normalized=False)

    assert identity.identity == "m"


def test_model_identity_records_normalization() -> None:
    """Determines whether dot product equals cosine similarity."""
    assert "normalized" in ModelIdentity._fields



def test_the_query_side_is_owned_by_the_provider() -> None:
    """
    A consumer reconstructing a model's query transformation from
    configuration would be guessing at provider-specific behaviour, and
    guessing wrong produces no error at all.
    """
    assert inspect.iscoroutinefunction(EmbeddingProvider.embed_query)


def test_model_identity_names_the_query_convention() -> None:
    """The identifier lets a consumer verify compatibility without knowing
    what the convention does."""
    assert "query_encoding" in ModelIdentity._fields

    symmetric = ModelIdentity(model_id="m", dimension=4, normalized=True)
    assert symmetric.query_encoding == ""


def test_input_limits_are_expressed_in_characters() -> None:
    """A token budget would bind the contract to one tokenizer."""
    assert "max_input_chars" in ModelIdentity._fields
    assert "max_input_tokens" not in ModelIdentity._fields


def test_a_provider_reports_a_stable_identity() -> None:
    provider = _Provider()

    assert provider.model_identity() == provider.model_identity()


# ---------------------------------------------------------------------------
# Provider behaviour
# ---------------------------------------------------------------------------


async def test_embedding_returns_one_vector_per_input_in_order() -> None:
    """
    Positional correspondence is the contract. A provider that reordered
    results would attach vectors to the wrong nodes undetectably.
    """
    vectors = await _Provider().embed(["a", "b", "c"])

    assert len(vectors) == 3
    assert all(len(vector) == 4 for vector in vectors)


async def test_a_provider_declares_its_batch_bound() -> None:
    """A provider unable to exploit batching reports one, rather than
    requiring the engine to know that."""
    assert _Provider().max_batch_size == 32


async def test_a_provider_closes_via_its_context_manager() -> None:
    provider = _Provider()

    async with provider as entered:
        assert entered is provider

    assert provider.closed == 1


async def test_a_provider_is_closed_when_the_body_raises() -> None:
    """A local model holds substantial memory; release must survive failure."""
    provider = _Provider()

    class _RunAborted(Exception):
        pass

    with pytest.raises(_RunAborted):
        async with provider:
            raise _RunAborted

    assert provider.closed == 1


# ---------------------------------------------------------------------------
# Sink
# ---------------------------------------------------------------------------


async def test_a_sink_accepts_units_and_flushes() -> None:
    sink = _Sink()

    await sink.write("unit-1")
    await sink.flush()
    await sink.close()

    assert sink.written == ["unit-1"]
    assert sink.flushed == 1
    assert sink.closed == 1


def test_a_sink_receives_a_unit_not_separate_artifacts() -> None:
    """
    Node, embedding, and relations arrive together so that a sink may write
    them atomically where its storage permits.
    """
    signature = inspect.signature(KnowledgeSink.write)

    assert signature.parameters["unit"].annotation == "KnowledgeUnit"


def test_the_sink_contract_names_no_storage_technology() -> None:
    """The module writes artifacts and remains unaware of index
    implementations."""
    source = inspect.getsource(KnowledgeSink).lower()

    for term in (
        "faiss",
        "qdrant",
        "milvus",
        "chroma",
        "elasticsearch",
        "neo4j",
        "hnsw",
        "bm25",
    ):
        assert term not in source


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("instance", "protocol"),
    [
        (_Segmenter(), SentenceSegmenter),
        (_Composer(), EmbeddingTextComposer),
        (_Enricher(), KnowledgeEnricher),
        (_Sink(), KnowledgeSink),
    ],
)
def test_reference_implementations_satisfy_their_protocols(
    instance: object, protocol: type
) -> None:
    assert isinstance(instance, protocol)


def test_a_nonconforming_object_fails_the_structural_check() -> None:
    assert not isinstance(object(), KnowledgeSink)


# ---------------------------------------------------------------------------
# Reuse
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name", ["Clock", "HashProvider", "ThreadOffloader", "IdGenerator",
             "MetricsSink", "StructuredLogger"]
)
def test_shared_concerns_are_reused_from_the_ingestion_layer(name: str) -> None:
    """
    A parallel declaration would let the two modules' vocabularies diverge.
    """
    from ptfm.core.ingestion import interfaces as ingestion_interfaces

    assert getattr(interfaces_module, name) is getattr(
        ingestion_interfaces, name
    )


def test_no_shared_concern_is_redeclared_locally() -> None:
    import ast

    tree = ast.parse(inspect.getsource(interfaces_module))
    declared = {
        node.name for node in tree.body if isinstance(node, ast.ClassDef)
    }

    for name in ("Clock", "HashProvider", "ThreadOffloader", "IdGenerator"):
        assert name not in declared, f"{name} is redeclared rather than reused"


# ---------------------------------------------------------------------------
# Module hygiene
# ---------------------------------------------------------------------------


def test_all_exports_resolve() -> None:
    for name in interfaces_module.__all__:
        assert hasattr(interfaces_module, name), name


def test_schema_and_model_imports_are_deferred() -> None:
    """
    Keeps the contract layer runtime-independent of models, so its tests run
    before models exists.
    """
    source = inspect.getsource(interfaces_module)

    assert "if TYPE_CHECKING:" in source
    deferred = source.split("if TYPE_CHECKING:")[1].split("__all__")[0]
    assert "from ptfm.core.knowledge.models import" in deferred
    assert "from ptfm.core.schemas import" in deferred


def test_no_concrete_implementation_is_imported() -> None:
    source = inspect.getsource(interfaces_module)

    for forbidden in (
        "from ptfm.core.knowledge.chunking",
        "from ptfm.core.knowledge.embedding",
        "from ptfm.plugins",
        "import numpy",
        "import sentence_transformers",
    ):
        assert forbidden not in source


def test_the_module_declares_no_mutable_state() -> None:
    mutable = [
        name
        for name, value in vars(interfaces_module).items()
        if not name.startswith("_") and isinstance(value, (list, dict, set))
    ]

    assert mutable == []


def test_the_module_contains_no_domain_vocabulary() -> None:
    source = inspect.getsource(interfaces_module).lower()

    for term in ("jurisdiction", "statute", "ticker", "icd", "patient", "court"):
        assert term not in source


def test_no_contract_produces_a_confidence_value() -> None:
    """
    Similarity is computed at query time and calibrated downstream. A score
    produced here would enter the pipeline outside calibration.
    """
    source = inspect.getsource(interfaces_module).lower()

    assert "def similarity" not in source
    assert "-> float" not in source
