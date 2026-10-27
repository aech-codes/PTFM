"""
Unit tests for the relation, enrichment, composition and embedding stages.

File:
    tests/core/knowledge/test_relations_enrichment_embedding.py
"""

from __future__ import annotations

import asyncio
import inspect
import math
from datetime import UTC, datetime
from typing import Any, Mapping, Sequence

import pytest

from ptfm.core.knowledge.config import (
    EmbeddingConfig,
    EnrichmentConfig,
    EnrichmentFailurePolicy,
    RelationConfig,
)
from ptfm.core.knowledge.exceptions import (
    DimensionMismatchError,
    EmbeddingProviderError,
    EmbeddingTimeoutError,
    EnricherContractViolationError,
    EnricherFailedError,
    EnricherTimeoutError,
    EnrichmentError,
    InputTooLargeError,
    KnowledgeConfigurationError,
    ModelIdentityError,
    RelationError,
)
from ptfm.core.knowledge.interfaces import ComposedInput, ModelIdentity
from ptfm.core.knowledge.pipeline import composition as composition_module
from ptfm.core.knowledge.pipeline import embedding as embedding_module
from ptfm.core.knowledge.pipeline import enrichment as enrichment_module
from ptfm.core.knowledge.pipeline import relations as relations_module
from ptfm.core.knowledge.pipeline.composition import (
    HeadingPathComposer,
    TextOnlyComposer,
)
from ptfm.core.knowledge.pipeline.embedding import EmbeddingStage
from ptfm.core.knowledge.pipeline.enrichment import EnrichmentStage
from ptfm.core.knowledge.pipeline.relations import RelationDeriver
from ptfm.core.ingestion.utils.hashing import Blake2bHashProvider
from ptfm.core.schemas import (
    BlockAttributes,
    BlockType,
    DocumentBlock,
    DocumentMetadata,
    KnowledgeNode,
    NodeType,
    RawDocument,
    RelationType,
    SourceDescriptor,
    TextSpan,
    derive_document_id,
    derive_node_id,
)

NOW = datetime(2026, 8, 1, tzinfo=UTC)
FINGERPRINT = "1:seg-fp-001"
DOCUMENT_ID = derive_document_id("file:///corpus/a.pdf", "hash-abc")


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


class _FixedClock:
    def now(self) -> datetime:
        return NOW


class _Logger:
    def __init__(self) -> None:
        self.events: list[tuple[str, str, dict[str, Any]]] = []

    def bind(self, **values: Any) -> _Logger:
        return self

    def debug(self, event: str, **values: Any) -> None:
        self.events.append(("debug", event, values))

    def info(self, event: str, **values: Any) -> None:
        self.events.append(("info", event, values))

    def warning(self, event: str, **values: Any) -> None:
        self.events.append(("warning", event, values))

    def error(self, event: str, **values: Any) -> None:
        self.events.append(("error", event, values))


def _source() -> SourceDescriptor:
    return SourceDescriptor(
        source_uri="file:///corpus/a.pdf",
        scheme="file",
        connector_name="LocalFilesystemConnector",
        extractor_name="PdfExtractor",
        acquired_at=NOW,
        size_bytes=4096,
        detected_media_type="application/pdf",
    )


def _block(
    block_id: str,
    start: int,
    end: int,
    *,
    block_type: BlockType = BlockType.PARAGRAPH,
    parent_id: str | None = None,
    depth: int = 0,
    ordinal: int = 0,
    page: int | None = None,
) -> DocumentBlock:
    return DocumentBlock(
        block_id=block_id,
        block_type=block_type,
        span=TextSpan(start=start, end=end),
        parent_id=parent_id,
        depth=depth,
        ordinal=ordinal,
        attributes=BlockAttributes(page_number=page),
    )


def _document(text: str, blocks: Sequence[DocumentBlock]) -> RawDocument:
    return RawDocument(
        document_id=DOCUMENT_ID,
        source=_source(),
        metadata=DocumentMetadata(char_count=len(text), word_count=len(text.split())),
        text=text,
        blocks=tuple(blocks),
        content_hash="hash-abc",
        text_hash="text-hash-abc",
        ingested_at=NOW,
    )


def _node(
    start: int,
    end: int,
    text: str,
    *,
    block_ids: tuple[str, ...] = ("0",),
    node_type: NodeType = NodeType.PROSE,
    heading_path: tuple[str, ...] = (),
    ordinal: int = 0,
    pages: tuple[int, ...] = (),
    metadata: Mapping[str, str] | None = None,
) -> KnowledgeNode:
    span = TextSpan(start=start, end=end)
    return KnowledgeNode(
        node_id=derive_node_id(DOCUMENT_ID, span, FINGERPRINT),
        document_id=DOCUMENT_ID,
        span=span,
        text=text,
        node_type=node_type,
        block_ids=block_ids,
        heading_path=heading_path,
        ordinal=ordinal,
        page_numbers=pages,
        source=_source(),
        segmentation_fingerprint=FINGERPRINT,
        source_text_hash="text-hash-abc",
        source_schema_version="1.0",
        metadata=dict(metadata) if metadata else {},
        processed_at=NOW,
    )


# ===========================================================================
# RELATIONS
# ===========================================================================


def _deriver(**config: Any) -> RelationDeriver:
    return RelationDeriver(RelationConfig(**config))


def _simple_document() -> tuple[RawDocument, list[KnowledgeNode]]:
    text = "Part I\n\nBody one.\n\nBody two."
    blocks = [
        _block("0", 0, 6, block_type=BlockType.HEADING),
        _block("1", 8, 17, ordinal=1),
        _block("2", 19, 28, ordinal=2),
    ]
    nodes = [
        _node(0, 6, "Part I", block_ids=("0",), node_type=NodeType.HEADING, ordinal=0),
        _node(8, 17, "Body one.", block_ids=("1",), heading_path=("Part I",), ordinal=1),
        _node(19, 28, "Body two.", block_ids=("2",), heading_path=("Part I",), ordinal=2),
    ]
    return _document(text, blocks), nodes


def test_reading_order_relations_link_adjacent_nodes() -> None:
    """Reading order is what lets a consumer expand a retrieved node into the
    passage surrounding it."""
    document, nodes = _simple_document()

    relations = _deriver(derive_hierarchy=False, derive_co_membership=False).derive(
        document, nodes
    )

    assert len(relations) == 2
    assert all(r.relation_type is RelationType.PRECEDES for r in relations)
    assert relations[0].source_node_id == nodes[0].node_id
    assert relations[0].target_node_id == nodes[1].node_id


def test_hierarchy_relations_reach_the_owning_heading() -> None:
    document, nodes = _simple_document()

    relations = _deriver(derive_sequence=False, derive_co_membership=False).derive(
        document, nodes
    )

    assert len(relations) == 2
    assert all(r.relation_type is RelationType.PARENT_SECTION for r in relations)
    assert {r.target_node_id for r in relations} == {nodes[0].node_id}


def test_a_heading_is_not_its_own_parent_section() -> None:
    document, nodes = _simple_document()

    relations = _deriver(derive_sequence=False, derive_co_membership=False).derive(
        document, nodes
    )

    assert nodes[0].node_id not in {r.source_node_id for r in relations}


def test_sections_are_identified_by_full_path_not_title() -> None:
    """Two sections sharing a title under different parents remain distinct."""
    text = "Part I\n\nNotes\n\nBody A.\n\nPart II\n\nNotes\n\nBody B."
    blocks = [
        _block("0", 0, 6, block_type=BlockType.HEADING),
        _block("1", 8, 13, block_type=BlockType.HEADING, ordinal=1),
        _block("2", 15, 22, ordinal=2),
        _block("3", 24, 31, block_type=BlockType.HEADING, ordinal=3),
        _block("4", 33, 38, block_type=BlockType.HEADING, ordinal=4),
        _block("5", 40, 47, ordinal=5),
    ]
    nodes = [
        _node(0, 6, "Part I", block_ids=("0",), node_type=NodeType.HEADING, ordinal=0),
        _node(8, 13, "Notes", block_ids=("1",), node_type=NodeType.HEADING,
              heading_path=("Part I",), ordinal=1),
        _node(15, 22, "Body A.", block_ids=("2",),
              heading_path=("Part I", "Notes"), ordinal=2),
        _node(24, 31, "Part II", block_ids=("3",), node_type=NodeType.HEADING,
              ordinal=3),
        _node(33, 38, "Notes", block_ids=("4",), node_type=NodeType.HEADING,
              heading_path=("Part II",), ordinal=4),
        _node(40, 47, "Body B.", block_ids=("5",),
              heading_path=("Part II", "Notes"), ordinal=5),
    ]
    document = _document(text, blocks)

    relations = _deriver(derive_sequence=False, derive_co_membership=False).derive(
        document, nodes
    )
    by_source = {r.source_node_id: r.target_node_id for r in relations}

    assert by_source[nodes[2].node_id] == nodes[1].node_id
    assert by_source[nodes[5].node_id] == nodes[4].node_id


def test_co_membership_is_chained_not_cliqued() -> None:
    """
    A clique over a large table would make relation count grow quadratically
    and let one table dominate a corpus.
    """
    text = "\n\n".join(f"Row {n}" for n in range(6))
    blocks = [_block("t", 0, len(text), block_type=BlockType.TABLE)]
    cursor = 0
    nodes = []
    for index in range(6):
        row = f"Row {index}"
        blocks.append(
            _block(f"t.{index}", cursor, cursor + len(row),
                   block_type=BlockType.TABLE_CELL, parent_id="t", depth=1,
                   ordinal=index)
        )
        nodes.append(
            _node(cursor, cursor + len(row), row, block_ids=(f"t.{index}",),
                  node_type=NodeType.TABLE_REGION, ordinal=index)
        )
        cursor += len(row) + 2
    document = _document(text, blocks)

    relations = _deriver(derive_sequence=False, derive_hierarchy=False).derive(
        document, nodes
    )
    table = [r for r in relations if r.relation_type is RelationType.SAME_TABLE]

    # Linear in members, not quadratic.
    assert len(table) == 5
    assert len(table) < (len(nodes) * (len(nodes) - 1)) // 2


def test_co_membership_carries_the_grouping_key() -> None:
    """The full group must be recoverable without a relation for every pair."""
    text = "A\n\nB"
    blocks = [
        _block("t", 0, 4, block_type=BlockType.TABLE),
        _block("t.0", 0, 1, block_type=BlockType.TABLE_CELL, parent_id="t", depth=1),
        _block("t.1", 3, 4, block_type=BlockType.TABLE_CELL, parent_id="t", depth=1,
               ordinal=1),
    ]
    nodes = [
        _node(0, 1, "A", block_ids=("t.0",), node_type=NodeType.TABLE_REGION),
        _node(3, 4, "B", block_ids=("t.1",), node_type=NodeType.TABLE_REGION,
              ordinal=1),
    ]

    relations = _deriver(derive_sequence=False, derive_hierarchy=False).derive(
        _document(text, blocks), nodes
    )

    assert relations[0].attributes["container_block_id"] == "t"


def test_page_membership_is_derived() -> None:
    text = "A\n\nB"
    blocks = [
        _block("0", 0, 1, page=3),
        _block("1", 3, 4, ordinal=1, page=3),
    ]
    nodes = [
        _node(0, 1, "A", block_ids=("0",), pages=(3,)),
        _node(3, 4, "B", block_ids=("1",), ordinal=1, pages=(3,)),
    ]

    relations = _deriver(derive_sequence=False, derive_hierarchy=False).derive(
        _document(text, blocks), nodes
    )
    page = [r for r in relations if r.relation_type is RelationType.SAME_PAGE]

    assert len(page) == 1
    assert page[0].attributes["page_number"] == "3"


def test_inverse_relations_are_not_emitted_by_default() -> None:
    document, nodes = _simple_document()

    relations = _deriver(derive_hierarchy=False, derive_co_membership=False).derive(
        document, nodes
    )

    assert all(r.relation_type is RelationType.PRECEDES for r in relations)


def test_inverse_relations_can_be_enabled() -> None:
    document, nodes = _simple_document()

    relations = _deriver(
        derive_hierarchy=False,
        derive_co_membership=False,
        emit_inverse_relations=True,
    ).derive(document, nodes)

    assert len([r for r in relations if r.relation_type is RelationType.FOLLOWS]) == 2


def test_an_excessive_relation_count_is_rejected_not_truncated() -> None:
    """A silently incomplete graph cannot be distinguished from a complete
    one."""
    text = "\n\n".join(f"N{n}" for n in range(10))
    blocks = []
    nodes = []
    cursor = 0
    for index in range(10):
        chunk = f"N{index}"
        blocks.append(_block(str(index), cursor, cursor + len(chunk), ordinal=index))
        nodes.append(
            _node(cursor, cursor + len(chunk), chunk, block_ids=(str(index),),
                  ordinal=index, pages=(1,))
        )
        cursor += len(chunk) + 2

    with pytest.raises(RelationError, match="not truncated"):
        _deriver(max_relations_per_node=1).derive(_document(text, blocks), nodes)


def test_relations_are_disabled_by_configuration() -> None:
    document, nodes = _simple_document()

    assert _deriver(enabled=False).derive(document, nodes) == ()


def test_a_single_node_yields_no_relations() -> None:
    document, nodes = _simple_document()

    assert _deriver().derive(document, nodes[:1]) == ()


def test_no_semantic_relation_is_derivable() -> None:
    """A judgement recorded as structure would enter the knowledge base
    outside calibration."""
    source = inspect.getsource(relations_module)

    for term in ("SIMILAR_TO", "ENTAILS", "CONTRADICTS", "similarity"):
        assert term not in source


# ===========================================================================
# ENRICHMENT
# ===========================================================================


class _Enricher:
    def __init__(
        self,
        name: str = "test_enricher",
        keys: frozenset[str] = frozenset({"topic"}),
        result: Any = None,
        error: Exception | None = None,
        delay: float = 0.0,
    ) -> None:
        self._name = name
        self._keys = keys
        self._result = {"topic": "finance"} if result is None else result
        self._error = error
        self._delay = delay

    @property
    def name(self) -> str:
        return self._name

    @property
    def metadata_keys(self) -> frozenset[str]:
        return self._keys

    async def enrich(self, node: KnowledgeNode) -> Any:
        if self._delay:
            await asyncio.sleep(self._delay)
        if self._error is not None:
            raise self._error
        return self._result


def _enrichment(*enrichers: Any, **config: Any) -> EnrichmentStage:
    return EnrichmentStage(enrichers, EnrichmentConfig(**config), _Logger())


async def test_enrichment_attaches_metadata() -> None:
    node = _node(0, 5, "Body.")

    enriched = await _enrichment(_Enricher()).enrich(node)

    assert enriched.metadata == {"topic": "finance"}


async def test_enrichment_never_alters_provenance() -> None:
    """The guarantee that matters most is the one least worth taking on
    trust."""
    node = _node(0, 5, "Body.")

    enriched = await _enrichment(_Enricher()).enrich(node)

    assert enriched.node_id == node.node_id
    assert enriched.text == node.text
    assert enriched.span == node.span
    assert enriched.block_ids == node.block_ids
    assert enriched.segmentation_fingerprint == node.segmentation_fingerprint


async def test_the_original_node_is_not_mutated() -> None:
    node = _node(0, 5, "Body.")

    await _enrichment(_Enricher()).enrich(node)

    assert node.metadata == {}


async def test_an_enricher_contributing_nothing_returns_the_node_unchanged() -> None:
    node = _node(0, 5, "Body.")

    enriched = await _enrichment(_Enricher(result={})).enrich(node)

    assert enriched is node


async def test_enrichment_can_be_disabled() -> None:
    node = _node(0, 5, "Body.")

    enriched = await _enrichment(_Enricher(), enabled=False).enrich(node)

    assert enriched is node


async def test_a_named_selection_runs_only_those_enrichers() -> None:
    node = _node(0, 5, "Body.")
    stage = _enrichment(
        _Enricher("wanted", frozenset({"topic"})),
        _Enricher("skipped", frozenset({"other"}), result={"other": "x"}),
        enabled_enrichers=frozenset({"wanted"}),
    )

    enriched = await stage.enrich(node)

    assert enriched.metadata == {"topic": "finance"}


async def test_a_failure_degrades_by_default() -> None:
    """Discarding a correctly segmented node because a lookup failed would
    trade a small loss for a large one."""
    node = _node(0, 5, "Body.")
    stage = _enrichment(_Enricher(error=RuntimeError("lookup failed")))

    enriched = await stage.enrich(node)

    assert enriched is node


async def test_a_degraded_failure_is_logged() -> None:
    """A failure neither raised nor recorded is a silent loss no operator
    could discover."""
    logger = _Logger()
    stage = EnrichmentStage(
        (_Enricher(error=RuntimeError("boom")),), EnrichmentConfig(), logger
    )

    await stage.enrich(_node(0, 5, "Body."))

    assert any(event == "knowledge.enrichment.degraded" for _, event, _ in logger.events)


async def test_a_failure_can_be_made_fatal() -> None:
    node = _node(0, 5, "Body.")
    stage = _enrichment(
        _Enricher(error=RuntimeError("boom")),
        failure_policy=EnrichmentFailurePolicy.FAIL,
    )

    with pytest.raises(EnricherFailedError) as excinfo:
        await stage.enrich(node)

    assert isinstance(excinfo.value.__cause__, RuntimeError)


async def test_a_timeout_degrades_under_the_default_policy() -> None:
    node = _node(0, 5, "Body.")
    stage = _enrichment(_Enricher(delay=1.0), timeout_seconds=0.01)

    assert await stage.enrich(node) is node


async def test_a_timeout_is_fatal_under_the_strict_policy() -> None:
    stage = _enrichment(
        _Enricher(delay=1.0),
        timeout_seconds=0.01,
        failure_policy=EnrichmentFailurePolicy.FAIL,
    )

    with pytest.raises(EnricherTimeoutError):
        await stage.enrich(_node(0, 5, "Body."))


@pytest.mark.parametrize(
    "result",
    ["not a mapping", {"topic": {"nested": "object"}}, {"topic": 42}],
)
async def test_a_contract_violation_is_fatal_regardless_of_policy(
    result: Any,
) -> None:
    """
    A contract violation indicates a defective plugin rather than a transient
    condition, and admitting its output would contaminate the knowledge base.
    """
    stage = _enrichment(
        _Enricher(result=result),
        failure_policy=EnrichmentFailurePolicy.DEGRADE,
    )

    with pytest.raises(EnricherContractViolationError):
        await stage.enrich(_node(0, 5, "Body."))


async def test_an_undeclared_key_is_rejected() -> None:
    """Undeclared keys defeat the collision check that keeps metadata
    independent of execution order."""
    stage = _enrichment(
        _Enricher(keys=frozenset({"topic"}), result={"undeclared": "x"})
    )

    with pytest.raises(EnricherContractViolationError, match="did not declare"):
        await stage.enrich(_node(0, 5, "Body."))


def test_two_enrichers_claiming_one_key_are_rejected() -> None:
    """Resolving by execution order would make metadata depend on injection
    order."""
    with pytest.raises(KnowledgeConfigurationError, match="both declare"):
        _enrichment(
            _Enricher("first", frozenset({"topic"})),
            _Enricher("second", frozenset({"topic"})),
        )


def test_duplicate_enricher_names_are_rejected() -> None:
    with pytest.raises(KnowledgeConfigurationError, match="Duplicate enricher"):
        _enrichment(
            _Enricher("same", frozenset({"a"})),
            _Enricher("same", frozenset({"b"})),
        )


async def test_excessive_metadata_keys_are_rejected() -> None:
    keys = frozenset(f"k{n}" for n in range(10))
    stage = _enrichment(
        _Enricher(keys=keys, result={f"k{n}": "v" for n in range(10)}),
        max_metadata_keys=5,
    )

    with pytest.raises(EnrichmentError, match="metadata keys"):
        await stage.enrich(_node(0, 5, "Body."))


async def test_an_oversized_metadata_value_is_rejected() -> None:
    """Metadata carries attributes, not content; content belongs in the node
    text where it carries a span."""
    stage = _enrichment(
        _Enricher(result={"topic": "x" * 5_000}), max_metadata_value_chars=100
    )

    with pytest.raises(EnrichmentError, match="belongs in the node text"):
        await stage.enrich(_node(0, 5, "Body."))


# ===========================================================================
# COMPOSITION
# ===========================================================================


def _embedding_config(**overrides: Any) -> EmbeddingConfig:
    return EmbeddingConfig(**overrides)


def test_heading_context_reaches_the_vector_not_the_node() -> None:
    """
    What is stored stays exact; what is embedded carries context. The node is
    never modified.
    """
    node = _node(0, 34, "The applicable rate is 4.5 percent",
                 heading_path=("Part I", "Interest Rates"))

    composed = HeadingPathComposer(_embedding_config()).compose(node)

    assert composed.text.startswith("Part I > Interest Rates")
    assert node.text == "The applicable rate is 4.5 percent"
    assert "Interest Rates" not in node.text


def test_the_composition_identifier_is_recorded() -> None:
    node = _node(0, 5, "Body.")

    assert HeadingPathComposer(_embedding_config()).compose(node).composition_id == (
        "heading_path_prefixed"
    )
    assert TextOnlyComposer(_embedding_config()).compose(node).composition_id == (
        "text_only"
    )


def test_a_node_without_headings_embeds_its_text_alone() -> None:
    node = _node(0, 5, "Body.")

    assert HeadingPathComposer(_embedding_config()).compose(node).text == "Body."


def test_text_only_composition_omits_context() -> None:
    node = _node(0, 5, "Body.", heading_path=("Part I",))

    assert TextOnlyComposer(_embedding_config()).compose(node).text == "Body."


def test_truncation_drops_context_never_content() -> None:
    """
    Losing an outer heading costs little; losing the end of a clause could
    change what the vector means.
    """
    text = "x" * 90
    node = _node(0, 90, text, heading_path=("Outermost", "Middle", "Innermost"))

    composed = HeadingPathComposer(_embedding_config(max_input_chars=110)).compose(node)

    assert composed.text.endswith(text)
    assert "Innermost" in composed.text
    assert "Outermost" not in composed.text


def test_the_entire_path_is_dropped_when_nothing_fits() -> None:
    text = "x" * 100
    node = _node(0, 100, text, heading_path=("A very long heading indeed",))

    composed = HeadingPathComposer(_embedding_config(max_input_chars=101)).compose(node)

    assert composed.text == text


def test_content_exceeding_the_bound_is_reported() -> None:
    """Configuration validates that a node at the maximum size fits, so this
    indicates a strategy exceeding its configuration."""
    node = _node(0, 200, "x" * 200)

    with pytest.raises(InputTooLargeError):
        HeadingPathComposer(_embedding_config(max_input_chars=100)).compose(node)


def test_composition_never_returns_the_node_text_altered_in_place() -> None:
    node = _node(0, 5, "Body.", heading_path=("Part I",))
    before = node.text

    HeadingPathComposer(_embedding_config()).compose(node)

    assert node.text == before


# ===========================================================================
# EMBEDDING
# ===========================================================================


class _Provider:
    def __init__(
        self,
        *,
        dimension: int = 4,
        normalized: bool = True,
        vectors: Any = None,
        error: Exception | None = None,
        fail_times: int = 0,
        delay: float = 0.0,
        identity: Any = "default",
    ) -> None:
        self._dimension = dimension
        self._normalized = normalized
        self._vectors = vectors
        self._error = error
        self._fail_times = fail_times
        self._delay = delay
        self._identity = identity
        self.calls = 0

    @property
    def name(self) -> str:
        return "TestProvider"

    @property
    def max_batch_size(self) -> int:
        return 32

    def model_identity(self) -> Any:
        if self._identity == "default":
            return ModelIdentity(
                model_id="test/model",
                dimension=self._dimension,
                normalized=self._normalized,
                revision="r1",
            )
        if isinstance(self._identity, Exception):
            raise self._identity
        return self._identity

    async def embed(self, texts: Sequence[str]) -> Any:
        self.calls += 1
        if self._delay:
            await asyncio.sleep(self._delay)
        if self._error is not None and self.calls <= self._fail_times:
            raise self._error
        if self._vectors is not None:
            return self._vectors
        unit = 1.0 / math.sqrt(self._dimension)
        return [tuple(unit for _ in range(self._dimension)) for _ in texts]

    async def health_check(self) -> bool:
        return True

    async def close(self) -> None:
        return None


def _stage(provider: Any = None, **config: Any) -> EmbeddingStage:
    return EmbeddingStage(
        provider or _Provider(),
        HeadingPathComposer(EmbeddingConfig(**config)),
        Blake2bHashProvider(),
        EmbeddingConfig(**config),
        _FixedClock(),
    )


async def test_embedding_produces_one_vector_per_node() -> None:
    nodes = [_node(0, 5, "Alpha"), _node(6, 11, "Bravo", ordinal=1)]

    embeddings = await _stage().embed(nodes)

    assert len(embeddings) == 2
    assert [e.node_id for e in embeddings] == [n.node_id for n in nodes]


async def test_an_embedding_records_its_model_identity() -> None:
    embeddings = await _stage().embed([_node(0, 5, "Alpha")])

    assert embeddings[0].model_id == "test/model"
    assert embeddings[0].model_revision == "r1"
    assert embeddings[0].model_identity == "test/model@r1"


async def test_an_embedding_records_what_was_actually_submitted() -> None:
    """A vector whose input hash no longer matches its node's composition is
    out of date."""
    node = _node(0, 5, "Alpha", heading_path=("Part I",))
    composed = HeadingPathComposer(EmbeddingConfig()).compose(node)

    embeddings = await _stage().embed([node])

    assert embeddings[0].input_text_hash == Blake2bHashProvider().hash_text(
        composed.text
    )
    assert embeddings[0].composition_id == "heading_path_prefixed"


async def test_an_empty_batch_produces_no_embeddings() -> None:
    assert await _stage().embed([]) == ()


async def test_the_batch_bound_respects_the_provider() -> None:
    """A provider knows its limits; exceeding them fails every batch."""
    stage = _stage(_Provider(), batch_size=128)

    assert stage.max_batch_size == 32


async def test_a_wrong_vector_count_is_rejected() -> None:
    """Vectors could not be matched to nodes by position."""
    provider = _Provider(vectors=[(0.5, 0.5, 0.5, 0.5)])

    with pytest.raises(DimensionMismatchError, match="vector_count|different number"):
        await _stage(provider).embed([_node(0, 5, "A"), _node(6, 11, "B", ordinal=1)])


async def test_a_wrong_dimension_is_rejected() -> None:
    provider = _Provider(vectors=[(1.0, 0.0)])

    with pytest.raises(DimensionMismatchError):
        await _stage(provider).embed([_node(0, 5, "Alpha")])


async def test_a_false_normalization_claim_is_rejected() -> None:
    """
    Results would still be ranked and acted upon, merely wrongly. The stage
    names the provider rather than letting the node fail later.
    """
    provider = _Provider(normalized=True, vectors=[(3.0, 4.0, 0.0, 0.0)])

    with pytest.raises(DimensionMismatchError) as excinfo:
        await _stage(provider).embed([_node(0, 5, "Alpha")])

    assert excinfo.value.context["failure"] == "normalization"


async def test_an_unnormalized_provider_is_not_checked_for_magnitude() -> None:
    provider = _Provider(normalized=False, vectors=[(3.0, 4.0, 0.0, 0.0)])

    embeddings = await _stage(provider).embed([_node(0, 5, "Alpha")])

    assert embeddings[0].normalized is False


async def test_a_transient_failure_is_retried() -> None:
    provider = _Provider(
        error=EmbeddingProviderError("reset", provider_name="TestProvider"),
        fail_times=1,
    )

    embeddings = await _stage(provider, max_retries=3).embed([_node(0, 5, "Alpha")])

    assert len(embeddings) == 1
    assert provider.calls == 2


async def test_retry_exhaustion_reports_the_attempt_count() -> None:
    provider = _Provider(
        error=EmbeddingProviderError("down", provider_name="TestProvider"),
        fail_times=99,
    )

    with pytest.raises(EmbeddingProviderError) as excinfo:
        await _stage(provider, max_retries=3).embed([_node(0, 5, "Alpha")])

    assert excinfo.value.context["attempts"] == 3


async def test_a_deterministic_failure_is_not_retried() -> None:
    """Eligibility comes from the exception, never from a type check here."""
    provider = _Provider(
        error=DimensionMismatchError(
            expected_dimension=4, actual_dimension=2, provider_name="TestProvider"
        ),
        fail_times=99,
    )

    with pytest.raises(DimensionMismatchError):
        await _stage(provider, max_retries=5).embed([_node(0, 5, "Alpha")])

    assert provider.calls == 1


async def test_a_deadline_produces_a_retryable_timeout() -> None:
    provider = _Provider(delay=1.0)

    with pytest.raises(EmbeddingTimeoutError) as excinfo:
        await _stage(provider, timeout_seconds=0.01, max_retries=2).embed(
            [_node(0, 5, "Alpha")]
        )

    assert excinfo.value.retryable is True
    assert provider.calls == 2


async def test_model_identity_is_resolved_once() -> None:
    """A provider changing model mid-run would produce a corpus whose vectors
    are not mutually comparable."""
    provider = _Provider()
    stage = _stage(provider)

    assert stage.model_identity is stage.model_identity


def test_an_unusable_model_identity_is_rejected() -> None:
    provider = _Provider(identity=ModelIdentity(model_id="", dimension=0, normalized=False))

    with pytest.raises(ModelIdentityError, match="unusable identity"):
        _stage(provider)


def test_a_failing_identity_call_is_converted() -> None:
    provider = _Provider(identity=RuntimeError("model not loaded"))

    with pytest.raises(ModelIdentityError) as excinfo:
        _stage(provider)

    assert isinstance(excinfo.value.__cause__, RuntimeError)


def test_buffering_is_not_this_stage_s_responsibility() -> None:
    """
    How long a partial batch waits is a decision about the shape of a run, not
    about embedding.
    """
    source = inspect.getsource(embedding_module)

    assert "flush_deadline_seconds" not in source


# ===========================================================================
# Anti-pattern guards
# ===========================================================================


@pytest.mark.parametrize(
    "module",
    [relations_module, enrichment_module, composition_module, embedding_module],
)
def test_no_stage_imports_a_concrete_implementation(module: Any) -> None:
    source = inspect.getsource(module)

    for forbidden in (
        "from ptfm.core.knowledge.chunking",
        "from ptfm.core.knowledge.registry",
        "from ptfm.plugins",
        "import numpy",
        "import sentence_transformers",
    ):
        assert forbidden not in source


@pytest.mark.parametrize(
    "module",
    [relations_module, enrichment_module, composition_module, embedding_module],
)
def test_no_stage_declares_mutable_module_state(module: Any) -> None:
    mutable = [
        name
        for name, value in vars(module).items()
        if not name.startswith("_") and isinstance(value, (list, dict, set))
    ]

    assert mutable == []


@pytest.mark.parametrize(
    "module", [relations_module, composition_module]
)
def test_synchronous_stages_declare_no_coroutines(module: Any) -> None:
    source = inspect.getsource(module)

    assert "async def" not in source


def test_embedding_never_reads_the_wall_clock_directly() -> None:
    source = inspect.getsource(embedding_module)

    assert "self._clock.now()" in source
    assert "datetime.now(" not in source


@pytest.mark.parametrize(
    "module",
    [relations_module, enrichment_module, composition_module, embedding_module],
)
def test_no_stage_produces_a_confidence_value(module: Any) -> None:
    source = inspect.getsource(module).lower()

    assert "confidence" not in source


@pytest.mark.parametrize(
    "module",
    [relations_module, enrichment_module, composition_module, embedding_module],
)
def test_no_stage_contains_domain_vocabulary(module: Any) -> None:
    source = inspect.getsource(module).lower()

    for term in ("jurisdiction", "statute", "ticker", "icd", "patient", "court"):
        assert term not in source
