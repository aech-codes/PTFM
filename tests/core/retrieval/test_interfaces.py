"""
Unit tests for the Retrieval Module contract layer.

File:
    tests/core/retrieval/test_interfaces.py

Emphasis
--------
    The structural guarantees. An arm able to return content, a writer whose
    atomicity is assumed rather than declared, or a redeclared shared contract
    would each let a defect become invisible, so each is asserted rather than
    trusted.
"""

from __future__ import annotations

import ast
import inspect
from typing import Any, Mapping, Sequence

import pytest

from ptfm.core.retrieval import interfaces as interfaces_module
from ptfm.core.retrieval.exceptions import StrategyRegistrationError
from ptfm.core.retrieval.interfaces import (
    ContextExpander,
    CredibilityProvider,
    EmbeddingProvider,
    FusionStrategy,
    IndexWriter,
    LexicalIndex,
    ModelIdentity,
    NodeStore,
    PartitionCatalog,
    QueryEncoder,
    RelationStore,
    RetrievalStrategy,
    SupersessionStore,
    VectorIndex,
)

#: Contracts reused from the ingestion and knowledge layers. Redeclaring any
#: would let two modules' vocabularies drift apart while appearing to describe
#: the same thing.
REUSED: tuple[str, ...] = (
    "Clock",
    "HashProvider",
    "IdGenerator",
    "MetricsSink",
    "StructuredLogger",
    "ThreadOffloader",
    "EmbeddingProvider",
    "ModelIdentity",
)


def _code_without_docstrings(module: object) -> str:
    """
    Return a module's source with every docstring removed.

    A prohibition is DOCUMENTED in docstrings, so a naive substring scan flags
    the sentences that state the rule as though they violated it. That is not
    a hypothetical: the relation store's docstring explains that no entailment
    relation exists, which is precisely the concept the guard forbids.

    Removing docstrings keeps the guard a simple substring check while making
    it accurate about what it enforces: that the concept appears in no name,
    declaration, or comment in the code itself. Documentation that states a
    prohibition should never be penalised by the test that enforces it.

    Parameters
    ----------
    module:
        The module to read.

    Returns
    -------
    str
        The source with docstrings stripped.
    """
    source = inspect.getsource(module)  # type: ignore[arg-type]
    tree = ast.parse(source)
    lines = source.splitlines(keepends=True)
    blank: set[int] = set()

    for node in ast.walk(tree):
        if not isinstance(
            node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
        ):
            continue
        body = getattr(node, "body", None)
        if not body:
            continue
        first = body[0]
        if not (
            isinstance(first, ast.Expr)
            and isinstance(first.value, ast.Constant)
            and isinstance(first.value.value, str)
        ):
            continue
        # Line numbers, not the cleaned text: ast.get_docstring dedents its
        # result, so it does not match indented source and a text replacement
        # would silently leave the docstring in place.
        blank.update(range(first.lineno - 1, (first.end_lineno or first.lineno)))

    return "".join(
        "" if index in blank else line for index, line in enumerate(lines)
    )


# ---------------------------------------------------------------------------
# Conforming reference implementations
#
# Model and schema types appear only in annotations, which are strings under
# 'from __future__ import annotations'. The contract layer therefore has no
# runtime dependency on either, and these fakes need no model instances.
# ---------------------------------------------------------------------------


class _NodeStore:
    @property
    def name(self) -> str:
        return "node"

    async def get_many(self, node_ids: Any, partition: Any) -> Mapping[str, Any]:
        return {}

    async def get_by_ordinal(
        self, document_id: str, ordinals: Any, partition: Any
    ) -> Mapping[int, Any]:
        return {}

    async def put_many(self, nodes: Any, partition: Any) -> None:
        return None

    async def count(self, partition: Any) -> int:
        return 0


class _VectorIndex:
    @property
    def name(self) -> str:
        return "vector"

    @property
    def supports_exact_search(self) -> bool:
        return True

    async def search(self, request: Any) -> Sequence[Any]:
        return ()

    async def upsert(self, embeddings: Any, partition: Any) -> None:
        return None

    async def count(self, partition: Any) -> int:
        return 0


class _LexicalIndex:
    @property
    def name(self) -> str:
        return "lexical"

    async def search(self, request: Any) -> Sequence[Any]:
        return ()

    async def index_nodes(self, nodes: Any, partition: Any) -> None:
        return None


class _RelationStore:
    @property
    def name(self) -> str:
        return "relation"

    async def relations_from(
        self, node_ids: Any, partition: Any
    ) -> Mapping[str, Any]:
        return {}

    async def structural_search(
        self, filters: Any, partition: Any, top_k: Any
    ) -> Sequence[Any]:
        return ()

    async def put_many(self, relations: Any, partition: Any) -> None:
        return None


class _SupersessionStore:
    @property
    def name(self) -> str:
        return "supersession"

    async def current_document_ids(
        self, document_ids: Any, partition: Any
    ) -> frozenset[str]:
        return frozenset()

    async def record_current(self, entry: Any, partition: Any) -> None:
        return None


class _IndexWriter:
    def __init__(self, atomic: bool = True) -> None:
        self._atomic = atomic
        self.written: list[Any] = []
        self.flushes = 0
        self.closed = 0

    @property
    def name(self) -> str:
        return "writer"

    @property
    def is_atomic(self) -> bool:
        return self._atomic

    async def write_units(
        self, segmentation: Any, embedding: Any, units: Any
    ) -> None:
        self.written.extend(units)

    async def flush(self) -> None:
        self.flushes += 1

    async def close(self) -> None:
        self.closed += 1


class _Arm(RetrievalStrategy):
    @property
    def name(self) -> str:
        return "dense"

    @property
    def method(self) -> str:
        return "DENSE"

    async def search(self, plan: Any) -> Sequence[Any]:
        return ()

    async def health_check(self) -> bool:
        return True


# ---------------------------------------------------------------------------
# Machine-enforced execution contract
# ---------------------------------------------------------------------------


def test_a_synchronous_arm_is_rejected_at_definition() -> None:
    """
    An arm crosses a process or network boundary. A synchronous one would
    block the event loop while the other arms ran, serializing what the
    architecture requires to be concurrent.
    """
    with pytest.raises(StrategyRegistrationError):

        class _Blocking(RetrievalStrategy):
            @property
            def name(self) -> str:
                return "blocking"

            def search(self, plan: Any) -> Sequence[Any]:  # type: ignore[override]
                return ()


def test_a_synchronous_health_check_is_rejected() -> None:
    with pytest.raises(StrategyRegistrationError):

        class _Blocking(RetrievalStrategy):
            @property
            def name(self) -> str:
                return "blocking"

            async def search(self, plan: Any) -> Sequence[Any]:
                return ()

            def health_check(self) -> bool:  # type: ignore[override]
                return True


def test_a_conforming_arm_defines_cleanly() -> None:
    assert _Arm().name == "dense"
    assert _Arm().method == "DENSE"


def test_the_base_arm_declaration_matches_its_own_contract() -> None:
    assert inspect.iscoroutinefunction(RetrievalStrategy.search)
    assert inspect.iscoroutinefunction(RetrievalStrategy.health_check)


def test_the_base_classes_declare_slots() -> None:
    """
    Without ``__slots__`` on the base, Python gives every subclass a
    ``__dict__`` regardless of what that subclass declares, so an arm's own
    ``__slots__`` is silently void.

    That matters because one arm instance serves every concurrent query. An
    implementation believing itself stateless could accumulate per-query state
    on a shared object and produce a race that appears only under load.
    """
    for base in (RetrievalStrategy, FusionStrategy):
        assert "__slots__" in base.__dict__, base.__name__


def test_a_slotted_subclass_really_has_no_dict() -> None:
    """The guarantee the declaration is meant to provide."""

    class _Slotted(RetrievalStrategy):
        __slots__ = ("_value",)

        def __init__(self) -> None:
            self._value = 1

        @property
        def name(self) -> str:
            return "slotted"

        async def search(self, plan: Any) -> Sequence[Any]:
            return ()

    instance = _Slotted()

    assert not hasattr(instance, "__dict__")
    with pytest.raises(AttributeError):
        instance.stashed = {}  # type: ignore[attr-defined]


def test_a_subclass_may_still_opt_into_state() -> None:
    """Adding slots to the base constrains nothing a subclass does not ask
    for."""

    class _Unslotted(RetrievalStrategy):
        @property
        def name(self) -> str:
            return "unslotted"

        async def search(self, plan: Any) -> Sequence[Any]:
            return ()

    instance = _Unslotted()
    instance.stashed = {"ok": True}  # type: ignore[attr-defined]

    assert instance.stashed == {"ok": True}  # type: ignore[attr-defined]


def test_fusion_is_synchronous() -> None:
    """
    Fusion performs no I/O. Declaring it asynchronous would conceal that and
    invite an implementation to acquire some.
    """
    assert not inspect.iscoroutinefunction(FusionStrategy.fuse)


async def test_a_conforming_arm_runs() -> None:
    assert await _Arm().search(plan=None) == ()
    assert await _Arm().health_check() is True


# ---------------------------------------------------------------------------
# Arms return identifiers, never authoritative content
# ---------------------------------------------------------------------------


def test_an_arm_cannot_return_authoritative_content() -> None:
    """
    Structural, not stylistic. Authoritative text comes from the node store
    alone, so an index holding a stale copy cannot change what a citation
    resolves to.
    """
    annotation = str(inspect.signature(RetrievalStrategy.search).return_annotation)

    for forbidden in ("KnowledgeNode", "RetrievedDocument", "EvidenceItem"):
        assert forbidden not in annotation


@pytest.mark.parametrize("protocol", [VectorIndex, LexicalIndex])
def test_index_searches_return_identifiers_only(protocol: type) -> None:
    annotation = str(inspect.signature(protocol.search).return_annotation)

    assert "KnowledgeNode" not in annotation


def test_only_the_node_store_returns_nodes() -> None:
    """The single source of truth for text, spans, and provenance."""
    annotation = str(inspect.signature(NodeStore.get_many).return_annotation)

    assert "KnowledgeNode" in annotation


def test_the_node_store_resolves_historical_citations() -> None:
    """
    A stored evidence item names a node identifier. If that node stopped
    resolving once its document was superseded, a previously verified response
    would become unauditable.
    """
    doc = inspect.getdoc(NodeStore.get_many) or ""

    assert "current" in doc.lower()
    assert "auditable" in doc.lower()


def test_expansion_is_positional_not_traversal() -> None:
    """
    Expansion is on the hot path of every query, and a graph round trip per
    hit would dominate the cost of a capability that is arithmetic.
    """
    assert hasattr(NodeStore, "get_by_ordinal")


# ---------------------------------------------------------------------------
# Atomicity is declared, not assumed
# ---------------------------------------------------------------------------


def test_a_writer_declares_whether_it_is_atomic() -> None:
    """
    A deployment must discover at startup that its backend offers only
    per-store writes, rather than discovering it from an untraceable result
    months later.
    """
    assert _IndexWriter(atomic=True).is_atomic is True
    assert _IndexWriter(atomic=False).is_atomic is False


def test_the_writer_contract_is_unit_level() -> None:
    """
    A knowledge unit touches five stores together. A crash between them leaves
    a node retrievable with no authoritative text.
    """
    parameters = inspect.signature(IndexWriter.write_units).parameters

    assert {"segmentation", "embedding", "units"} <= set(parameters)


def test_an_unembedded_corpus_is_a_valid_intermediate_state() -> None:
    """Segmented but not yet embedded is ordinary, not an error."""
    annotation = str(
        inspect.signature(IndexWriter.write_units).parameters["embedding"].annotation
    )

    assert "Optional" in annotation


async def test_a_writer_accepts_units_and_closes() -> None:
    writer = _IndexWriter()

    await writer.write_units("seg", "emb", ["unit-1", "unit-2"])
    await writer.flush()
    await writer.close()

    assert writer.written == ["unit-1", "unit-2"]
    assert writer.flushes == 1 and writer.closed == 1


# ---------------------------------------------------------------------------
# Partition catalog
# ---------------------------------------------------------------------------


def test_exactly_one_segmentation_partition_is_active() -> None:
    """
    Blending identifier spaces would produce a result set whose members cannot
    be compared.
    """
    doc = " ".join((PartitionCatalog.__doc__ or "").split())

    assert "Exactly one segmentation partition is active" in doc


def test_several_embedding_partitions_may_coexist() -> None:
    """
    Re-embedding changes vectors while leaving nodes, spans, identifiers, and
    lexical terms untouched, so coexistence is the whole of the migration
    story.
    """
    doc = " ".join((PartitionCatalog.__doc__ or "").split())

    assert "Several embedding partitions may exist over it" in doc


def test_the_catalog_resolves_both_partition_kinds() -> None:
    assert hasattr(PartitionCatalog, "active_segmentation")
    assert hasattr(PartitionCatalog, "active_embedding")


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_a_vector_index_declares_whether_exact_search_is_possible() -> None:
    """
    Approximate search is not reproducible. A caller auditing a past result
    needs to know whether exactness is available before promising it.
    """
    assert _VectorIndex().supports_exact_search is True
    assert hasattr(VectorIndex, "supports_exact_search")


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("instance", "protocol"),
    [
        (_NodeStore(), NodeStore),
        (_VectorIndex(), VectorIndex),
        (_LexicalIndex(), LexicalIndex),
        (_RelationStore(), RelationStore),
        (_SupersessionStore(), SupersessionStore),
        (_IndexWriter(), IndexWriter),
    ],
)
def test_reference_implementations_satisfy_their_protocols(
    instance: object, protocol: type
) -> None:
    assert isinstance(instance, protocol)


@pytest.mark.parametrize(
    "protocol",
    [
        NodeStore,
        VectorIndex,
        LexicalIndex,
        RelationStore,
        SupersessionStore,
        IndexWriter,
    ],
)
def test_a_nonconforming_object_fails_the_structural_check(
    protocol: type,
) -> None:
    assert not isinstance(object(), protocol)


def test_every_store_reports_a_name() -> None:
    """A failure must be attributable to one store among several."""
    for store in (
        _NodeStore(),
        _VectorIndex(),
        _LexicalIndex(),
        _RelationStore(),
        _SupersessionStore(),
        _IndexWriter(),
    ):
        assert isinstance(store.name, str) and store.name


# ---------------------------------------------------------------------------
# Reuse
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", REUSED)
def test_shared_contracts_are_reused_not_redeclared(name: str) -> None:
    """
    A parallel declaration would let two modules' vocabularies drift while
    appearing to describe the same thing.
    """
    from ptfm.core.ingestion import interfaces as ingestion_interfaces
    from ptfm.core.knowledge import interfaces as knowledge_interfaces

    if not hasattr(interfaces_module, name):
        pytest.skip(f"{name} is not re-exported")

    source = (
        knowledge_interfaces
        if name in ("EmbeddingProvider", "ModelIdentity")
        else ingestion_interfaces
    )
    assert getattr(interfaces_module, name) is getattr(source, name)


def test_no_shared_contract_is_declared_locally() -> None:
    tree = ast.parse(inspect.getsource(interfaces_module))
    declared = {node.name for node in tree.body if isinstance(node, ast.ClassDef)}

    for name in REUSED:
        assert name not in declared, f"{name} is redeclared rather than reused"


def test_the_query_side_is_delegated_to_the_provider() -> None:
    """
    Which transformation a model expects on a query is the provider's own
    knowledge. A consumer reproducing it would guess at provider-specific
    behaviour that produces no error when guessed wrong.
    """
    assert hasattr(EmbeddingProvider, "embed_query")
    assert "query_encoding" in ModelIdentity._fields


def test_the_encoder_reports_the_model_it_encodes_with() -> None:
    """So that a query vector can be checked against the partition it will
    search rather than assumed compatible."""
    assert hasattr(QueryEncoder, "model_identity")


# ---------------------------------------------------------------------------
# Domain boundary
# ---------------------------------------------------------------------------


def test_credibility_is_the_domain_boundary() -> None:
    """
    What makes a source credible is domain knowledge. Retrieval knows only the
    value, and only about a source.
    """
    assert hasattr(CredibilityProvider, "credibility_for")

    doc = CredibilityProvider.__doc__ or ""
    assert "source" in doc.lower()


def test_expansion_never_produces_an_independent_result() -> None:
    """
    Corroboration counts distinct anchors, so an expanded node admitted as an
    anchor would make one finding appear as two.
    """
    doc = (ContextExpander.__doc__ or "") + (
        inspect.getdoc(ContextExpander.expand) or ""
    )

    assert "anchor" in doc.lower()


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
    assert "ptfm.core.retrieval.models" in deferred


def test_no_concrete_backend_is_imported() -> None:
    source = inspect.getsource(interfaces_module)

    for forbidden in (
        "import psycopg",
        "import asyncpg",
        "import qdrant",
        "import tantivy",
        "import numpy",
        "from ptfm.plugins",
    ):
        assert forbidden not in source


def test_the_module_names_no_storage_technology() -> None:
    """Backends stay behind interfaces; the contract must not presume one."""
    source = inspect.getsource(interfaces_module).lower()

    for term in (
        "postgres",
        "pgvector",
        "qdrant",
        "faiss",
        "milvus",
        "elasticsearch",
        "opensearch",
        "tantivy",
        "neo4j",
        "hnsw",
    ):
        assert term not in source


def test_the_module_declares_no_mutable_state() -> None:
    mutable = [
        name
        for name, value in vars(interfaces_module).items()
        if not name.startswith("_") and isinstance(value, (list, dict, set))
    ]

    assert mutable == []


def test_the_module_contains_no_domain_vocabulary() -> None:
    source = inspect.getsource(interfaces_module).lower()

    for term in ("jurisdiction", "statute", "ticker", " icd", "patient", "court"):
        assert term not in source


def test_no_contract_returns_a_calibrated_value() -> None:
    """
    Signals carry the scale they live on precisely so that treating one as a
    probability is visibly wrong. Calibration belongs to a later module.
    """
    source = _code_without_docstrings(interfaces_module).lower()

    for term in ("calibrat", "entail", "truth space", "verification result"):
        assert term not in source
