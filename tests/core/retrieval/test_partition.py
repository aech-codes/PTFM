"""
Unit tests for compatibility partition derivation and resolution.

File:
    tests/core/retrieval/test_partition.py

Emphasis
--------
    Partitions are the mechanism that makes an incomparable comparison
    unexpressible rather than merely detectable. These tests assert that every
    way of producing one is refused, and that the refusals happen before a
    search runs rather than being inferred from its results.
"""

from __future__ import annotations

import ast
import inspect
from typing import Any, Optional, Sequence

import pytest

from ptfm.core.knowledge.interfaces import ModelIdentity
from ptfm.core.knowledge.models import KnowledgeUnit
from ptfm.core.retrieval import partition as partition_module
from ptfm.core.retrieval.config import PartitionConfig
from ptfm.core.retrieval.exceptions import (
    EmbeddingSpaceMismatchError,
    PartitionNotFoundError,
    QueryEncodingMismatchError,
    SegmentationMismatchError,
)
from ptfm.core.retrieval.models import EmbeddingPartition, SegmentationPartition
from ptfm.core.retrieval.partition import (
    PartitionResolver,
    embedding_for_vector,
    segmentation_for_node,
)
from ptfm.core.schemas import (
    KnowledgeNode,
    NodeEmbedding,
    SourceDescriptor,
    TextSpan,
    NodeType,
)

NOW = __import__("datetime").datetime(2026, 8, 1, tzinfo=__import__("datetime").UTC)


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def _source() -> SourceDescriptor:
    return SourceDescriptor(
        source_uri="file:///corpus/a.pdf",
        scheme="file",
        connector_name="LocalFilesystemConnector",
        extractor_name="PdfExtractor",
        acquired_at=NOW,
        size_bytes=1024,
        detected_media_type="application/pdf",
    )


def _node(
    node_id: str = "node-1",
    *,
    fingerprint: str = "1:abc",
    schema_version: str = "1.0",
) -> KnowledgeNode:
    return KnowledgeNode(
        node_id=node_id,
        document_id="doc-1",
        span=TextSpan(start=0, end=5),
        text="Alpha",
        node_type=NodeType.PROSE,
        block_ids=("0",),
        ordinal=0,
        source=_source(),
        segmentation_fingerprint=fingerprint,
        source_text_hash="text-hash",
        source_schema_version=schema_version,
        processed_at=NOW,
    )


def _embedding(
    node_id: str = "node-1",
    *,
    model_id: str = "BAAI/bge-large-en-v1.5",
    revision: Optional[str] = "r1",
    dimension: int = 2,
    normalized: bool = True,
    composition_id: str = "heading_path_prefixed",
    query_encoding: str = "prefix:query: ",
) -> NodeEmbedding:
    return NodeEmbedding(
        node_id=node_id,
        document_id="doc-1",
        vector=(0.6, 0.8),
        dimension=dimension,
        model_id=model_id,
        model_revision=revision,
        normalized=normalized,
        composition_id=composition_id,
        query_encoding=query_encoding,
        input_text_hash="input-hash",
        embedded_at=NOW,
    )


def _identity(**overrides: Any) -> ModelIdentity:
    fields: dict[str, Any] = {
        "model_id": "BAAI/bge-large-en-v1.5",
        "dimension": 2,
        "normalized": True,
        "revision": "r1",
        "query_encoding": "prefix:query: ",
    }
    fields.update(overrides)
    return ModelIdentity(**fields)


class _Catalog:
    """In-test catalog recording what was registered."""

    def __init__(
        self,
        segmentation: Optional[SegmentationPartition] = None,
        embedding: Optional[EmbeddingPartition] = None,
    ) -> None:
        self._segmentation = segmentation
        self._embedding = embedding
        self.registered: list[tuple[Any, Any, bool]] = []

    @property
    def name(self) -> str:
        return "test-catalog"

    async def active_segmentation(self) -> Optional[SegmentationPartition]:
        return self._segmentation

    async def active_embedding(
        self, segmentation: SegmentationPartition
    ) -> Optional[EmbeddingPartition]:
        return self._embedding

    async def register(
        self,
        segmentation: SegmentationPartition,
        embedding: Optional[EmbeddingPartition],
        *,
        activate: bool,
    ) -> None:
        self.registered.append((segmentation, embedding, activate))
        if activate:
            self._segmentation = segmentation
            self._embedding = embedding


def _live() -> tuple[SegmentationPartition, EmbeddingPartition]:
    segmentation = segmentation_for_node(_node())
    return segmentation, embedding_for_vector(_embedding(), segmentation)


def _resolver(
    catalog: Optional[_Catalog] = None, **config: Any
) -> PartitionResolver:
    if catalog is None:
        segmentation, embedding = _live()
        catalog = _Catalog(segmentation, embedding)
    return PartitionResolver(catalog, PartitionConfig(**config))


# ===========================================================================
# DERIVATION
# ===========================================================================


def test_a_segmentation_partition_derives_from_a_node() -> None:
    partition = segmentation_for_node(_node())

    assert partition.segmentation_fingerprint == "1:abc"
    assert partition.source_schema_version == "1.0"


def test_derivation_is_deterministic() -> None:
    assert segmentation_for_node(_node()).key == segmentation_for_node(_node()).key


def test_a_differing_fingerprint_derives_a_differing_partition() -> None:
    """Identifiers derived under differing segmentation occupy disjoint
    spaces."""
    assert (
        segmentation_for_node(_node(fingerprint="1:def")).key
        != segmentation_for_node(_node()).key
    )


def test_the_document_contract_version_participates() -> None:
    """
    A span is an offset into document text, so a contract altering
    normalization shifts every offset while the segmentation fingerprint stays
    identical.
    """
    assert (
        segmentation_for_node(_node(schema_version="2.0")).key
        != segmentation_for_node(_node()).key
    )


def test_an_embedding_partition_derives_from_a_vector() -> None:
    segmentation = segmentation_for_node(_node())

    partition = embedding_for_vector(_embedding(), segmentation)

    assert partition.dimension == 2
    assert partition.query_encoding == "prefix:query: "
    assert partition.segmentation.key == segmentation.key


@pytest.mark.parametrize(
    "override",
    [
        {"model_id": "intfloat/multilingual-e5-large"},
        {"revision": "r2"},
        {"dimension": 4},
        {"normalized": False},
        {"composition_id": "text_only"},
        {"query_encoding": ""},
    ],
)
def test_every_vector_attribute_separates_the_space(
    override: dict[str, Any],
) -> None:
    """Each is a reason two vectors cannot be compared."""
    segmentation = segmentation_for_node(_node())

    baseline = embedding_for_vector(_embedding(), segmentation)
    altered = embedding_for_vector(_embedding(**override), segmentation)

    assert not baseline.describes_same_space(altered)


# ===========================================================================
# QUERY RESOLUTION
# ===========================================================================


async def test_a_query_resolves_the_active_segmentation() -> None:
    segmentation, _ = _live()

    resolved = await _resolver().resolve_for_query()

    assert resolved.key == segmentation.key


async def test_a_query_resolves_the_active_embedding_partition() -> None:
    segmentation, embedding = _live()

    resolved = await _resolver().resolve_embedding_for_query(
        segmentation, _identity()
    )

    assert resolved is not None
    assert resolved.key == embedding.key


async def test_an_unindexed_corpus_is_reported() -> None:
    """A deployment state rather than a query fault, and distinguishable from
    one."""
    with pytest.raises(PartitionNotFoundError):
        await _resolver(_Catalog()).resolve_for_query()


async def test_a_mismatched_query_convention_is_refused() -> None:
    """
    The check exists because its absence is invisible: embedding a query as
    though it were a passage returns plausible results in a worse order, and
    nothing reports a fault.
    """
    segmentation, _ = _live()

    with pytest.raises(QueryEncodingMismatchError):
        await _resolver().resolve_embedding_for_query(
            segmentation, _identity(query_encoding="")
        )


def test_the_query_convention_check_cannot_be_disabled() -> None:
    """
    A switch could only ever be used to hide a failure the operator cannot
    otherwise detect, so no setting exists to turn it off.
    """
    assert "verify_query_encoding" not in PartitionConfig.model_fields


@pytest.mark.parametrize(
    "override",
    [
        {"model_id": "intfloat/multilingual-e5-large"},
        {"revision": "r2"},
        {"dimension": 4},
        {"normalized": False},
    ],
)
async def test_a_provider_from_another_space_is_refused(
    override: dict[str, Any],
) -> None:
    """
    A query vector from one space compared against another produces a ranking
    that is arithmetically valid and semantically meaningless.
    """
    segmentation, _ = _live()

    with pytest.raises(EmbeddingSpaceMismatchError):
        await _resolver().resolve_embedding_for_query(
            segmentation, _identity(**override)
        )


async def test_compatibility_is_established_before_a_search_runs() -> None:
    """
    Resolution happens on the query path ahead of any arm, so an incompatible
    pairing is refused rather than inferred from poor results.
    """
    source = inspect.getsource(PartitionResolver.resolve_embedding_for_query)

    assert "_verify_space" in source
    assert "_verify_query_encoding" in source


async def test_no_model_identity_yields_no_dense_retrieval() -> None:
    """
    A deployment may run without an embedding provider, and lexical and
    structural retrieval continue.
    """
    segmentation, _ = _live()

    assert await _resolver().resolve_embedding_for_query(segmentation, None) is None


async def test_a_missing_embedding_partition_can_be_required() -> None:
    segmentation, _ = _live()
    resolver = _resolver(
        _Catalog(segmentation, None), require_embedding_partition=True
    )

    with pytest.raises(PartitionNotFoundError):
        await resolver.resolve_embedding_for_query(segmentation, _identity())


async def test_a_missing_embedding_partition_can_be_tolerated() -> None:
    """A corpus segmented but not yet embedded is a valid intermediate
    state."""
    segmentation, _ = _live()
    resolver = _resolver(
        _Catalog(segmentation, None), require_embedding_partition=False
    )

    assert (
        await resolver.resolve_embedding_for_query(segmentation, _identity()) is None
    )


# ===========================================================================
# INGEST RESOLUTION
# ===========================================================================


async def test_units_resolve_to_a_partition_pair() -> None:
    segmentation, embedding = _live()

    resolved_segmentation, resolved_embedding = _resolver().resolve_for_units(
        [KnowledgeUnit(node=_node(), embedding=_embedding())]
    )

    assert resolved_segmentation.key == segmentation.key
    assert resolved_embedding is not None
    assert resolved_embedding.key == embedding.key


async def test_unembedded_units_resolve_without_an_embedding_partition() -> None:
    resolved_segmentation, resolved_embedding = _resolver().resolve_for_units(
        [KnowledgeUnit(node=_node())]
    )

    assert resolved_segmentation is not None
    assert resolved_embedding is None


async def test_a_batch_mixing_segmentations_is_refused() -> None:
    """
    Writing them together would place identifiers from disjoint spaces in one
    partition, where a lookup could resolve to unrelated content.
    """
    units = [
        KnowledgeUnit(node=_node("node-1")),
        KnowledgeUnit(node=_node("node-2", fingerprint="1:def")),
    ]

    with pytest.raises(SegmentationMismatchError):
        _resolver().resolve_for_units(units)


async def test_a_batch_mixing_embedding_spaces_is_refused() -> None:
    """
    Two models' vectors in one index produce comparisons that are silently
    meaningless.
    """
    units = [
        KnowledgeUnit(node=_node("node-1"), embedding=_embedding("node-1")),
        KnowledgeUnit(
            node=_node("node-2"),
            embedding=_embedding("node-2", model_id="other/model"),
        ),
    ]

    with pytest.raises(EmbeddingSpaceMismatchError):
        _resolver().resolve_for_units(units)


async def test_an_empty_batch_is_refused_or_yields_nothing() -> None:
    """A batch with no units names no partition, so it must not silently
    resolve to the active one."""
    try:
        segmentation, _ = _resolver().resolve_for_units([])
    except (SegmentationMismatchError, PartitionNotFoundError, ValueError):
        return
    assert segmentation is None


# ===========================================================================
# REGISTRATION
# ===========================================================================


async def test_registration_delegates_to_the_catalog() -> None:
    segmentation, embedding = _live()
    catalog = _Catalog()

    await PartitionResolver(catalog, PartitionConfig()).register(
        segmentation, embedding, activate=True
    )

    assert len(catalog.registered) == 1
    assert catalog.registered[0][2] is True


async def test_a_partition_can_be_registered_without_activation() -> None:
    """
    Building a new embedding partition alongside the active one is how a
    migration proceeds: dual write, then an atomic switch.
    """
    segmentation, embedding = _live()
    catalog = _Catalog(segmentation, embedding)

    await PartitionResolver(catalog, PartitionConfig()).register(
        segmentation, embedding, activate=False
    )

    assert catalog.registered[0][2] is False


# ===========================================================================
# Anti-pattern guards
# ===========================================================================


def _code_without_docstrings(module: object) -> str:
    """
    Return a module's source with every docstring removed.

    Line ranges are used rather than text replacement because
    ``ast.get_docstring`` dedents its result, which does not match indented
    source.

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
        blank.update(range(first.lineno - 1, (first.end_lineno or first.lineno)))

    return "".join("" if i in blank else line for i, line in enumerate(lines))


def test_the_resolver_holds_no_assignable_state() -> None:
    with pytest.raises(AttributeError):
        _resolver().cache = {}  # type: ignore[attr-defined]


def test_the_module_names_no_storage_technology() -> None:
    source = inspect.getsource(partition_module).lower()

    for term in (
        "postgres",
        "pgvector",
        "qdrant",
        "faiss",
        "milvus",
        "elasticsearch",
        "tantivy",
        "neo4j",
    ):
        assert term not in source


def test_the_module_contains_no_domain_vocabulary() -> None:
    source = inspect.getsource(partition_module).lower()

    for term in ("jurisdiction", "statute", "ticker", " icd", "patient", "court"):
        assert term not in source


def test_the_module_declares_no_calibration_concept() -> None:
    source = _code_without_docstrings(partition_module).lower()

    for term in ("calibrat", "entail", "truth space", "verification result"):
        assert term not in source


def test_the_module_declares_no_mutable_state() -> None:
    mutable = [
        name
        for name, value in vars(partition_module).items()
        if not name.startswith("_") and isinstance(value, (list, dict, set))
    ]

    assert mutable == []


def test_the_module_imports_no_backend() -> None:
    source = inspect.getsource(partition_module)

    for forbidden in (
        "import psycopg",
        "import asyncpg",
        "import qdrant",
        "import numpy",
        "from ptfm.plugins",
    ):
        assert forbidden not in source
