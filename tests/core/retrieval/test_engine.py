"""
Unit tests for the retrieval engine and the composition root.

File:
    tests/core/retrieval/test_engine.py

Emphasis
--------
    Three properties the pipeline exists to hold, each asserted by behaviour
    rather than by inspecting a declaration.

    Evidence is TRACEABLE. Every item's content is the authoritative node text,
    and that text is the raw source at the node's span. The fixtures below build
    nodes as literal slices of known source documents so the chain can be
    checked by comparison instead of trusted.

    A FAILED search is not an empty one. If every arm fails the query raises,
    because an empty result asserts that no evidence exists and a failed search
    establishes nothing of the kind.

    A REDUCED result says so. Anything that did not run appears in diagnostics,
    so a two-arm result cannot be mistaken for a three-arm one. The currency
    filter is the one stage outside this scheme: its failure would make an
    answer wrong rather than smaller, so it never degrades at all.

    The last group asserts that the engine is genuinely backend-neutral, by
    running the same engine over a store set whose vector index is a recording
    wrapper and checking that fetch depth reaches it.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime
from typing import Any, Mapping, Optional, Sequence

import pytest

from ptfm.core.knowledge.interfaces import ModelIdentity
from ptfm.core.knowledge.models import KnowledgeUnit
from ptfm.core.retrieval import (
    ArmStatus,
    DegradationPolicy,
    DenseConfig,
    EmbeddingPartition,
    ExpansionConfig,
    FilterField,
    FilterOperator,
    GraphConfig,
    IngestConfig,
    PartitionConfig,
    RetrievalConfig,
    RetrievalEngine,
    RetrievalFilter,
    RetrievalStores,
    SegmentationPartition,
    SparseConfig,
    build_default_engine,
    build_memory_stores,
)
from ptfm.core.retrieval.exceptions import (
    AllArmsFailedError,
    RetrievalConfigurationError,
    StoreUnavailableError,
    SupersessionError,
)
from ptfm.core.schemas import (
    DiscoveryMethod,
    KnowledgeNode,
    NodeEmbedding,
    NodeRelation,
    NodeType,
    RelationType,
    SourceDescriptor,
    TextSpan,
)

# ============================================================================
# FIXTURE DATA
# ============================================================================

_FINGERPRINT = "structural-1.0:test"
_SCHEMA_VERSION = "1.0"
_PROCESSED_AT = datetime(2026, 9, 24, tzinfo=UTC)

SEGMENTATION = SegmentationPartition(
    segmentation_fingerprint=_FINGERPRINT,
    source_schema_version=_SCHEMA_VERSION,
)
EMBEDDING = EmbeddingPartition(
    segmentation=SEGMENTATION,
    model_identity="test-model@1",
    dimension=3,
    normalized=True,
    composition_id="text-only",
)

#: Source documents. Every node is a literal slice of one of these, so the
#: Evidence-First chain is checkable by comparison rather than by assertion
#: about how it was built.
RAW_DOCUMENTS: Mapping[str, str] = {
    "doc1": (
        "The quick brown fox jumps over the lazy dog. "
        "Quick thinking wins races every single time. "
        "A third sentence about foxes and hounds here."
    ),
    "doc2": (
        "Financial statements for the fiscal year are attached. "
        "Revenue grew by a considerable margin this quarter."
    ),
}

SOURCE_URIS: Mapping[str, str] = {
    "doc1": "file:///doc1.txt",
    "doc2": "file:///doc2.txt",
}


def _unit_vector(components: Sequence[float]) -> tuple[float, ...]:
    """
    Scale a vector to unit length.

    Parameters
    ----------
    components:
        The raw components.

    Returns
    -------
    tuple[float, ...]
        The normalized vector.
    """
    magnitude = math.sqrt(sum(value * value for value in components)) or 1.0
    return tuple(value / magnitude for value in components)


#: node id, document, ordinal, span start, span end, vector
_NODE_SPECS: tuple[tuple[str, str, int, int, int, tuple[float, ...]], ...] = (
    ("d1n0", "doc1", 0, 0, 44, _unit_vector((1.0, 0.0, 0.0))),
    ("d1n1", "doc1", 1, 44, 87, _unit_vector((0.9, 0.1, 0.0))),
    ("d1n2", "doc1", 2, 87, 133, _unit_vector((0.2, 0.9, 0.0))),
    ("d2n0", "doc2", 0, 0, 53, _unit_vector((0.0, 0.0, 1.0))),
    ("d2n1", "doc2", 1, 53, 110, _unit_vector((0.0, 0.2, 0.9))),
)


def _node(node_id: str, document_id: str, ordinal: int, start: int, end: int) -> KnowledgeNode:
    """
    Build a node whose text is a literal slice of its source document.

    Parameters
    ----------
    node_id:
        The node identifier.
    document_id:
        The document the node belongs to.
    ordinal:
        Position within the document.
    start:
        Span start in the source text.
    end:
        Span end in the source text.

    Returns
    -------
    KnowledgeNode
        The node.
    """
    return KnowledgeNode(
        node_id=node_id,
        document_id=document_id,
        segmentation_fingerprint=_FINGERPRINT,
        source_schema_version=_SCHEMA_VERSION,
        ordinal=ordinal,
        text=RAW_DOCUMENTS[document_id][start:end],
        source_text_hash=f"sha256:{document_id}",
        span=TextSpan(start=start, end=end),
        node_type=NodeType.PROSE,
        block_ids=("b0",),
        source=SourceDescriptor(
            source_uri=SOURCE_URIS[document_id], version_tag="v1"
        ),
        heading_path=(),
        page_numbers=(1,),
        metadata={},
        processed_at=_PROCESSED_AT,
    )


def _units() -> tuple[KnowledgeUnit, ...]:
    """
    Build the corpus as knowledge units with embeddings and relations.

    Returns
    -------
    tuple[KnowledgeUnit, ...]
        The units.
    """
    built: list[KnowledgeUnit] = []
    for node_id, document_id, ordinal, start, end, vector in _NODE_SPECS:
        relations: tuple[NodeRelation, ...] = ()
        if ordinal > 0:
            previous = next(
                spec[0]
                for spec in _NODE_SPECS
                if spec[1] == document_id and spec[2] == ordinal - 1
            )
            relations = (
                NodeRelation(
                    source_node_id=previous,
                    target_node_id=node_id,
                    relation_type=RelationType.PRECEDES,
                ),
            )
        built.append(
            KnowledgeUnit(
                node=_node(node_id, document_id, ordinal, start, end),
                embedding=NodeEmbedding(
                    node_id=node_id,
                    document_id=document_id,
                    model_id="test-model",
                    model_revision="1",
                    dimension=3,
                    normalized=True,
                    composition_id="text-only",
                    vector=vector,
                ),
                relations=relations,
            )
        )
    return tuple(built)


class _Encoder:
    """
    Query encoder honouring the declared protocol.

    ``model_identity`` is a PROPERTY, as ``QueryEncoder`` declares it. That
    matters: an implementation exposing it as a method would have satisfied an
    earlier caller that invoked it as one, and this test exists partly to keep
    the protocol and its callers from drifting apart again.
    """

    __slots__ = ()

    @property
    def model_identity(self) -> ModelIdentity:
        """
        Report the model this encoder encodes with.

        Returns
        -------
        ModelIdentity
            The identity matching ``EMBEDDING``.
        """
        return ModelIdentity(
            model_id="test-model", dimension=3, normalized=True, revision="1"
        )

    async def encode(
        self,
        query: str,
        partition: EmbeddingPartition,
    ) -> tuple[float, ...]:
        """
        Encode a query deterministically.

        Parameters
        ----------
        query:
            The query text.
        partition:
            The embedding partition.

        Returns
        -------
        tuple[float, ...]
            A unit vector leaning toward whichever document the query mentions.
        """
        terms = set(query.lower().split())
        if terms & {"revenue", "financial", "fiscal"}:
            return _unit_vector((0.0, 0.1, 1.0))
        return _unit_vector((1.0, 0.05, 0.0))


class _Credibility:
    """
    Credibility provider that knows about some sources and not others.

    Attributes
    ----------
    _known:
        Credibility per source URI.
    """

    __slots__ = ("_known",)

    def __init__(self, known: Mapping[str, float]) -> None:
        """
        Store the known credibilities.

        Parameters
        ----------
        known:
            Credibility per source URI.
        """
        self._known = dict(known)

    @property
    def name(self) -> str:
        """
        Identify the provider.

        Returns
        -------
        str
            A stable identifier.
        """
        return "test-credibility"

    async def credibility_for(
        self,
        source_uris: Sequence[str],
    ) -> Mapping[str, float]:
        """
        Report credibility for the sources it knows.

        Omits the rest, so absence stays representable.

        Parameters
        ----------
        source_uris:
            Sources to assess.

        Returns
        -------
        Mapping[str, float]
            Credibility for known sources only.
        """
        return {uri: self._known[uri] for uri in source_uris if uri in self._known}


class _DepthRecorder:
    """
    Vector index wrapper recording the fetch depth it was asked for.

    Attributes
    ----------
    _inner:
        The wrapped index.
    requested:
        Depths requested, in order.
    """

    __slots__ = ("_inner", "requested")

    def __init__(self, inner: Any) -> None:
        """
        Wrap an index.

        Parameters
        ----------
        inner:
            The index to wrap.
        """
        self._inner = inner
        self.requested: list[int] = []

    @property
    def name(self) -> str:
        """
        Identify the wrapped index.

        Returns
        -------
        str
            The inner index's name.
        """
        return self._inner.name

    @property
    def supports_exact_search(self) -> bool:
        """
        Whether exact search is available.

        Returns
        -------
        bool
            Whatever the inner index reports.
        """
        return self._inner.supports_exact_search

    async def search(
        self,
        vector: Sequence[float],
        partition: EmbeddingPartition,
        top_k: int,
        *,
        filters: Sequence[RetrievalFilter] = (),
        exact: bool = False,
    ) -> Sequence[Any]:
        """
        Record the depth and delegate.

        Parameters
        ----------
        vector:
            The query vector.
        partition:
            The embedding partition.
        top_k:
            Fetch depth, which is what this wrapper exists to observe.
        filters:
            Structural constraints.
        exact:
            Whether approximate search is forbidden.

        Returns
        -------
        Sequence[Any]
            The inner index's candidates.
        """
        self.requested.append(top_k)
        return await self._inner.search(
            vector, partition, top_k, filters=filters, exact=exact
        )

    async def upsert(
        self,
        embeddings: Sequence[NodeEmbedding],
        partition: EmbeddingPartition,
    ) -> None:
        """
        Delegate a write.

        Parameters
        ----------
        embeddings:
            Embeddings to store.
        partition:
            The embedding partition.
        """
        await self._inner.upsert(embeddings, partition)

    async def count(self, partition: EmbeddingPartition) -> int:
        """
        Delegate a count.

        Parameters
        ----------
        partition:
            The embedding partition.

        Returns
        -------
        int
            The inner index's count.
        """
        return await self._inner.count(partition)


class _UnreachableVectorIndex:
    """
    Vector index whose reads fail and whose writes succeed.

    The realistic shape of an index that has lost its query path, and the shape
    that lets a test distinguish a degraded read from a failed ingest.
    """

    __slots__ = ("_inner",)

    def __init__(self, inner: Any) -> None:
        """
        Wrap an index.

        Parameters
        ----------
        inner:
            The index to write through to.
        """
        self._inner = inner

    @property
    def name(self) -> str:
        """
        Identify the index.

        Returns
        -------
        str
            A stable identifier.
        """
        return "unreachable-vector"

    @property
    def supports_exact_search(self) -> bool:
        """
        Whether exact search is available.

        Returns
        -------
        bool
            ``True``, which is irrelevant while reads fail.
        """
        return True

    async def search(self, *args: Any, **kwargs: Any) -> Sequence[Any]:
        """
        Fail, as an unreachable index does.

        Parameters
        ----------
        args:
            Ignored.
        kwargs:
            Ignored.

        Returns
        -------
        Sequence[Any]
            Never returns.

        Raises
        ------
        StoreUnavailableError
            Always.
        """
        raise StoreUnavailableError(
            "vector index is unreachable", store_name="unreachable-vector"
        )

    async def upsert(
        self,
        embeddings: Sequence[NodeEmbedding],
        partition: EmbeddingPartition,
    ) -> None:
        """
        Delegate a write, which still succeeds.

        Parameters
        ----------
        embeddings:
            Embeddings to store.
        partition:
            The embedding partition.
        """
        await self._inner.upsert(embeddings, partition)

    async def count(self, partition: EmbeddingPartition) -> int:
        """
        Fail, as an unreachable index does.

        Parameters
        ----------
        partition:
            The embedding partition.

        Returns
        -------
        int
            Never returns.

        Raises
        ------
        StoreUnavailableError
            Always.
        """
        raise StoreUnavailableError(
            "vector index is unreachable", store_name="unreachable-vector"
        )


class _UnreachableLexicalIndex:
    """Lexical index whose reads fail."""

    __slots__ = ()

    @property
    def name(self) -> str:
        """
        Identify the index.

        Returns
        -------
        str
            A stable identifier.
        """
        return "unreachable-lexical"

    async def search(self, *args: Any, **kwargs: Any) -> Sequence[Any]:
        """
        Fail, as an unreachable index does.

        Parameters
        ----------
        args:
            Ignored.
        kwargs:
            Ignored.

        Returns
        -------
        Sequence[Any]
            Never returns.

        Raises
        ------
        StoreUnavailableError
            Always.
        """
        raise StoreUnavailableError(
            "lexical index is unreachable", store_name="unreachable-lexical"
        )

    async def index_nodes(
        self,
        nodes: Sequence[KnowledgeNode],
        partition: SegmentationPartition,
    ) -> None:
        """
        Accept a write without doing anything.

        Parameters
        ----------
        nodes:
            Nodes to index.
        partition:
            The segmentation partition.
        """


class _UnreachableSupersessionStore:
    """Supersession store that cannot report currency."""

    __slots__ = ()

    @property
    def name(self) -> str:
        """
        Identify the store.

        Returns
        -------
        str
            A stable identifier.
        """
        return "unreachable-supersession"

    async def current_document_ids(
        self,
        source_uris: Sequence[str],
        partition: SegmentationPartition,
    ) -> Mapping[str, Optional[str]]:
        """
        Fail, as an unreachable store does.

        Parameters
        ----------
        source_uris:
            Sources to look up.
        partition:
            The segmentation partition.

        Returns
        -------
        Mapping[str, Optional[str]]
            Never returns.

        Raises
        ------
        StoreUnavailableError
            Always.
        """
        raise StoreUnavailableError(
            "currency store is unreachable", store_name="unreachable-supersession"
        )

    async def record_current(
        self,
        source_uri: str,
        document_id: Optional[str],
        version_tag: Optional[str],
        partition: SegmentationPartition,
    ) -> None:
        """
        Accept a write without doing anything.

        Parameters
        ----------
        source_uri:
            The source.
        document_id:
            The current document.
        version_tag:
            The source version.
        partition:
            The segmentation partition.
        """


def _replace(stores: RetrievalStores, **overrides: Any) -> RetrievalStores:
    """
    Build a store set with some members substituted.

    Parameters
    ----------
    stores:
        The base store set.
    overrides:
        Members to replace.

    Returns
    -------
    RetrievalStores
        The substituted set.
    """
    members = {
        "nodes": stores.nodes,
        "vectors": stores.vectors,
        "lexical": stores.lexical,
        "relations": stores.relations,
        "supersession": stores.supersession,
        "catalog": stores.catalog,
        "writer": stores.writer,
    }
    members.update(overrides)
    return RetrievalStores(**members)


async def _seeded(
    config: Optional[RetrievalConfig] = None,
    *,
    stores: Optional[RetrievalStores] = None,
    **kwargs: Any,
) -> tuple[RetrievalEngine, RetrievalStores]:
    """
    Build an engine over a registered, populated corpus.

    Parameters
    ----------
    config:
        Retrieval configuration.
    stores:
        The stores to use, defaulting to a fresh in-memory set.
    kwargs:
        Further arguments for the composition root.

    Returns
    -------
    tuple[RetrievalEngine, RetrievalStores]
        The engine and its stores.
    """
    resolved = stores if stores is not None else build_memory_stores()
    await resolved.catalog.register(SEGMENTATION, EMBEDDING, activate=True)
    engine = build_default_engine(
        config, stores=resolved, encoder=_Encoder(), **kwargs
    )
    await engine.ingest(SEGMENTATION, EMBEDDING, _units())
    return engine, resolved


# ============================================================================
# EVIDENCE-FIRST
# ============================================================================


@pytest.mark.asyncio
async def test_evidence_content_is_the_raw_source_at_its_span() -> None:
    """
    Every evidence item's text is the source text its span denotes.

    The whole chain in one assertion: index to identifier, identifier to node,
    node to span, span to source. A break anywhere makes a citation point at
    something the source does not say.
    """
    engine, stores = await _seeded()
    outcome = await engine.retrieve("quick brown fox", top_k=3, exact=True)

    assert outcome.evidence.evidence, "expected evidence for a matching query"
    for item in outcome.evidence.evidence:
        held = await stores.nodes.get_many([item.node_id], SEGMENTATION)
        node = held[item.node_id]
        assert item.content == node.text
        raw = RAW_DOCUMENTS[item.document_id]
        assert node.text == raw[node.span.start : node.span.end]


@pytest.mark.asyncio
async def test_context_is_attributed_to_an_anchor_and_never_corroborates() -> None:
    """
    Context accompanies an anchor rather than counting as a second finding.

    Corroboration is the quantity calibration consumes, so a passage admitted
    only to make an anchor readable must not raise it.
    """
    engine, _ = await _seeded()
    outcome = await engine.retrieve("quick brown fox", top_k=3, exact=True)

    anchors = {
        item.node_id
        for item in outcome.evidence.evidence
        if item.node_id == item.anchor_node_id
    }
    context = [
        item
        for item in outcome.evidence.evidence
        if item.node_id != item.anchor_node_id
    ]

    assert len(anchors) == outcome.evidence.anchor_count
    assert len(context) == outcome.evidence.context_count
    assert all(item.anchor_node_id in anchors for item in context)
    assert all(
        record.method is DiscoveryMethod.EXPANSION
        for item in context
        for record in item.discovery
    )


@pytest.mark.asyncio
async def test_credibility_is_recorded_only_when_provided() -> None:
    """
    An unassessable source carries no credibility and is reported as such.

    A substituted value would enter retrieval reliability, and through it
    hallucination risk, indistinguishably from a measured one.
    """
    engine, _ = await _seeded(credibility=_Credibility({"file:///doc1.txt": 0.9}))
    outcome = await engine.retrieve("quick brown fox", top_k=3, exact=True)

    scored = {
        document.source_uri: document.credibility_score
        for document in outcome.evidence.documents
    }
    assert scored.get("file:///doc1.txt") == 0.9
    assert all(
        value is None for uri, value in scored.items() if uri != "file:///doc1.txt"
    )

    bare_engine, _ = await _seeded()
    bare = await bare_engine.retrieve("quick brown fox", top_k=3, exact=True)
    assert all(item.credibility_score is None for item in bare.evidence.evidence)
    assert set(bare.diagnostics.credibility_unavailable) == {
        document.source_uri for document in bare.evidence.documents
    }


# ============================================================================
# FAILURE IS NOT ABSENCE
# ============================================================================


@pytest.mark.asyncio
async def test_all_arms_failing_raises_instead_of_returning_empty() -> None:
    """
    A total search failure is not evidence that the corpus holds nothing.

    The single most consequential behaviour here: an empty result is a claim
    about the corpus, and a failed search supports no such claim.
    """
    base = build_memory_stores()
    stores = _replace(
        base,
        vectors=_UnreachableVectorIndex(base.vectors),
        lexical=_UnreachableLexicalIndex(),
    )
    await base.catalog.register(SEGMENTATION, EMBEDDING, activate=True)
    engine = build_default_engine(stores=stores, encoder=_Encoder())
    await engine.ingest(SEGMENTATION, EMBEDDING, _units())

    with pytest.raises(AllArmsFailedError) as raised:
        await engine.retrieve("quick brown fox")

    assert raised.value.degradable is False


@pytest.mark.asyncio
async def test_one_arm_failing_degrades_and_says_so() -> None:
    """
    A reduced result is returned and marked reduced.

    Returning it unmarked would let a consumer treat a one-arm result as a
    hybrid one.
    """
    base = build_memory_stores()
    stores = _replace(base, vectors=_UnreachableVectorIndex(base.vectors))
    await base.catalog.register(SEGMENTATION, EMBEDDING, activate=True)
    engine = build_default_engine(stores=stores, encoder=_Encoder())
    await engine.ingest(SEGMENTATION, EMBEDDING, _units())

    outcome = await engine.retrieve("quick brown fox", top_k=3)

    assert outcome.evidence.documents
    assert outcome.degraded is True
    assert outcome.diagnostics.failed_arms == ("dense",)
    assert outcome.diagnostics.contributing_arms == ("sparse",)


@pytest.mark.asyncio
async def test_strict_policy_refuses_to_degrade() -> None:
    """
    Under STRICT, a degradable failure fails the query.

    The policy exists for callers who would rather have no answer than a
    quietly narrower one.
    """
    base = build_memory_stores()
    stores = _replace(base, vectors=_UnreachableVectorIndex(base.vectors))
    await base.catalog.register(SEGMENTATION, EMBEDDING, activate=True)
    engine = build_default_engine(
        RetrievalConfig(degradation_policy=DegradationPolicy.STRICT),
        stores=stores,
        encoder=_Encoder(),
    )
    await engine.ingest(SEGMENTATION, EMBEDDING, _units())

    with pytest.raises(StoreUnavailableError):
        await engine.retrieve("quick brown fox")


@pytest.mark.parametrize(
    "policy", (DegradationPolicy.DEGRADE, DegradationPolicy.STRICT)
)
@pytest.mark.asyncio
async def test_currency_failure_never_degrades(policy: DegradationPolicy) -> None:
    """
    An unusable currency store fails the query under either policy.

    The one stage outside the degradation scheme. A missing arm yields less
    evidence, which diagnostics describe; a superseded document served as
    current yields evidence that is wrong and carries no marker at all.

    Parameters
    ----------
    policy:
        The degradation policy under test.
    """
    base = build_memory_stores()
    stores = _replace(base, supersession=_UnreachableSupersessionStore())
    await base.catalog.register(SEGMENTATION, EMBEDDING, activate=True)
    engine = build_default_engine(
        RetrievalConfig(degradation_policy=policy),
        stores=stores,
        encoder=_Encoder(),
    )
    await engine.ingest(SEGMENTATION, EMBEDDING, _units())

    with pytest.raises(SupersessionError):
        await engine.retrieve("quick brown fox")


@pytest.mark.asyncio
async def test_unrecorded_currency_is_counted_not_hidden() -> None:
    """
    Results whose source has no currency record are kept and counted.

    Distinct from a withheld result and from a checked one. Reporting zero
    superseded while the filter was inoperative would read as confirmation that
    everything is current.
    """
    stores = build_memory_stores()
    await stores.catalog.register(SEGMENTATION, EMBEDDING, activate=True)

    # Written past the writer, so nothing records currency for these sources.
    nodes = [unit.node for unit in _units()]
    await stores.nodes.put_many(nodes, SEGMENTATION)
    await stores.lexical.index_nodes(nodes, SEGMENTATION)
    await stores.vectors.upsert(
        [unit.embedding for unit in _units()], EMBEDDING
    )

    engine = build_default_engine(stores=stores, encoder=_Encoder())
    outcome = await engine.retrieve("quick brown fox", top_k=3, exact=True)

    assert outcome.evidence.documents
    assert outcome.diagnostics.currency_unchecked > 0
    assert outcome.diagnostics.superseded_count == 0


@pytest.mark.asyncio
async def test_superseded_documents_are_withheld() -> None:
    """
    A document with a newer version does not reach the result.

    Parameters
    ----------
    None
    """
    engine, stores = await _seeded()
    await stores.supersession.record_current(
        "file:///doc1.txt", "doc1-revised", "v2", SEGMENTATION
    )

    outcome = await engine.retrieve("quick brown fox", top_k=5, exact=True)

    assert all(
        document.document_id != "doc1" for document in outcome.evidence.documents
    )
    assert outcome.diagnostics.superseded_count > 0
    assert outcome.diagnostics.currency_unchecked == 0


# ============================================================================
# SKIPPING IS NOT FAILING
# ============================================================================


@pytest.mark.asyncio
async def test_missing_encoder_skips_dense_rather_than_failing() -> None:
    """
    Lexical-only retrieval is a configuration, not a fault.

    The arm is recorded as skipped with a reason, so diagnostics distinguish
    "not asked" from "asked and broken".
    """
    stores = build_memory_stores()
    await stores.catalog.register(SEGMENTATION, EMBEDDING, activate=True)
    engine = build_default_engine(stores=stores)
    await engine.ingest(SEGMENTATION, EMBEDDING, _units())

    outcome = await engine.retrieve("quick brown fox", top_k=3)

    assert outcome.evidence.documents
    dense = next(
        arm for arm in outcome.diagnostics.arms if arm.arm_name == "dense"
    )
    assert dense.status is ArmStatus.SKIPPED
    assert "vector" in (dense.message or "").lower()


@pytest.mark.asyncio
async def test_under_covered_partition_skips_dense() -> None:
    """
    Dense retrieval is skipped when too few nodes carry vectors.

    Searching a partially embedded corpus returns neighbours drawn from the
    embedded subset while appearing to search all of it, and nothing in the
    result distinguishes that from an absence of near neighbours.
    """
    stores = build_memory_stores()
    await stores.catalog.register(SEGMENTATION, EMBEDDING, activate=True)
    config = RetrievalConfig(
        partitions=PartitionConfig(minimum_embedding_coverage=0.9)
    )
    engine = build_default_engine(config, stores=stores, encoder=_Encoder())

    units = _units()
    sparse_units = [
        KnowledgeUnit(node=unit.node, embedding=None, relations=unit.relations)
        for unit in units[:-1]
    ] + [units[-1]]
    await engine.ingest(SEGMENTATION, EMBEDDING, sparse_units)

    outcome = await engine.retrieve("quick brown fox")

    dense = next(
        arm for arm in outcome.diagnostics.arms if arm.arm_name == "dense"
    )
    assert dense.status is ArmStatus.SKIPPED
    assert "coverage" in (dense.message or "").lower()
    assert outcome.diagnostics.embedding_coverage is not None
    assert outcome.diagnostics.embedding_coverage < 0.9
    assert outcome.degraded is True


@pytest.mark.asyncio
async def test_coverage_threshold_requires_a_probe() -> None:
    """
    A coverage threshold with nothing measuring it is refused.

    A threshold nothing measures reads as enforced and is not, which is the
    failure mode this codebase treats as worse than either position.
    """
    engine = build_default_engine(
        RetrievalConfig(partitions=PartitionConfig(minimum_embedding_coverage=0.5)),
        encoder=_Encoder(),
    )
    assert engine is not None

    with pytest.raises(RetrievalConfigurationError):
        RetrievalEngine(
            preparer=None,  # type: ignore[arg-type]
            arms=(),
            fusion=None,  # type: ignore[arg-type]
            hydrator=None,  # type: ignore[arg-type]
            supersession=None,  # type: ignore[arg-type]
            expander=None,
            credibility=None,  # type: ignore[arg-type]
            assembler=None,  # type: ignore[arg-type]
            config=RetrievalConfig(),
            clock=None,  # type: ignore[arg-type]
            id_generator=None,  # type: ignore[arg-type]
            logger=None,  # type: ignore[arg-type]
            metrics=None,  # type: ignore[arg-type]
        )


# ============================================================================
# FETCH DEPTH
# ============================================================================


@pytest.mark.asyncio
async def test_fetch_depth_is_resolved_and_reaches_the_index() -> None:
    """
    Arms search at the resolved depth, not at the caller's requested count.

    Fetch depth exists so fusion can see where a node ranks in an arm's
    ordering even when it ranks lower there than elsewhere. If the resolution
    never reached the index, the setting governing it would read as effective
    while doing nothing.
    """
    base = build_memory_stores()
    recorder = _DepthRecorder(base.vectors)
    stores = _replace(base, vectors=recorder)
    await base.catalog.register(SEGMENTATION, EMBEDDING, activate=True)

    engine = build_default_engine(
        RetrievalConfig(
            dense=DenseConfig(oversample=3),
            expansion=ExpansionConfig(enabled=False),
        ),
        stores=stores,
        encoder=_Encoder(),
    )
    await engine.ingest(SEGMENTATION, EMBEDDING, _units())

    await engine.retrieve("fox", top_k=10)
    unfiltered = recorder.requested[-1]
    assert unfiltered > 10, "the arm floor must outrank a small requested count"

    document_filter = [
        RetrievalFilter(
            field=FilterField.DOCUMENT_ID,
            operator=FilterOperator.EQUALS,
            values=("doc1",),
        )
    ]
    await engine.retrieve("fox", top_k=10, filters=document_filter)
    filtered = recorder.requested[-1]
    assert filtered > unfiltered, "a filtered dense search must over-fetch"

    # Bounded, so a selective filter cannot turn a query into a scan.
    assert filtered <= 10 * 20


# ============================================================================
# COMPOSITION ROOT
# ============================================================================


@pytest.mark.asyncio
async def test_default_engine_is_complete_with_no_arguments() -> None:
    """
    An engine built with no arguments works.

    The measure of whether the defaults are genuinely usable rather than
    placeholders requiring a fixture to become real.
    """
    engine = build_default_engine()

    assert engine.arm_names
    assert engine.ingests is True
    assert "sparse" in engine.arm_names


def test_non_atomic_writer_is_refused_when_atomicity_is_required() -> None:
    """
    A writer that cannot guarantee atomicity is rejected at construction.

    Discovered after a partial write, the same fault leaves a node retrievable
    by one index with no authoritative text.
    """

    class _NonAtomicWriter:
        """Writer without atomicity."""

        __slots__ = ()

        @property
        def name(self) -> str:
            """
            Identify the writer.

            Returns
            -------
            str
                A stable identifier.
            """
            return "non-atomic"

        @property
        def is_atomic(self) -> bool:
            """
            Whether writes are atomic.

            Returns
            -------
            bool
                ``False``.
            """
            return False

        async def write_units(
            self,
            segmentation: SegmentationPartition,
            embedding: Optional[EmbeddingPartition],
            units: Sequence[KnowledgeUnit],
        ) -> None:
            """
            Accept units without writing.

            Parameters
            ----------
            segmentation:
                The segmentation partition.
            embedding:
                The embedding partition.
            units:
                Units to write.
            """

        async def flush(self) -> None:
            """Do nothing."""

        async def close(self) -> None:
            """Do nothing."""

    stores = _replace(build_memory_stores(), writer=_NonAtomicWriter())

    with pytest.raises(RetrievalConfigurationError):
        build_default_engine(stores=stores)

    relaxed = build_default_engine(
        RetrievalConfig(ingest=IngestConfig(require_atomic_writer=False)),
        stores=stores,
    )
    assert relaxed.ingests is True


@pytest.mark.asyncio
async def test_read_only_engine_refuses_to_ingest() -> None:
    """
    An engine with no writer refuses rather than silently doing nothing.

    A caller believing it had ingested a corpus would go on to query an empty
    one and read the emptiness as an absence of evidence.
    """
    stores = _replace(build_memory_stores(), writer=None)
    engine = build_default_engine(stores=stores)

    assert engine.ingests is False
    with pytest.raises(RetrievalConfigurationError):
        await engine.ingest(SEGMENTATION, EMBEDDING, _units())


def test_configuration_refuses_to_disable_every_arm() -> None:
    """
    A configuration leaving no arm enabled is invalid.

    Such an engine would answer every query with an empty result, which cannot
    be distinguished from a corpus holding nothing relevant.
    """
    with pytest.raises(Exception) as raised:
        RetrievalConfig(
            dense=DenseConfig(enabled=False),
            sparse=SparseConfig(enabled=False),
            graph=GraphConfig(enabled=False),
        )
    assert "arm" in str(raised.value).lower()


@pytest.mark.asyncio
async def test_health_check_distinguishes_reachable_from_unreachable() -> None:
    """
    Health reporting answers in both directions.

    A probe that always reports unhealthy would keep a deployment from ever
    becoming ready; one that always reports healthy would never detect an
    outage. Both are checks that cannot fail where it matters, so both
    directions are asserted here.
    """
    engine, _ = await _seeded()
    healthy = await engine.health_check()
    assert set(healthy) == set(engine.arm_names)
    assert all(healthy.values())

    base = build_memory_stores()
    stores = _replace(base, vectors=_UnreachableVectorIndex(base.vectors))
    await base.catalog.register(SEGMENTATION, EMBEDDING, activate=True)
    degraded_engine = build_default_engine(stores=stores, encoder=_Encoder())

    reported = await degraded_engine.health_check()
    assert reported["dense"] is False
    assert reported["sparse"] is True


@pytest.mark.asyncio
async def test_exact_search_is_reported_as_reproducible() -> None:
    """
    A result gathered without approximate search can be reproduced.

    A caller auditing a past answer needs that distinction recorded, because an
    approximate result cannot be re-derived.
    """
    engine, _ = await _seeded()

    exact = await engine.retrieve("quick brown fox", top_k=3, exact=True)
    assert exact.diagnostics.exact_search is True
    assert exact.reproducible is True


@pytest.mark.asyncio
async def test_outcome_representation_leaks_no_evidence_text() -> None:
    """
    A logged outcome does not disclose retrieved content.

    Diagnostics travel through logs, and retrieved text may be confidential
    even when the identifiers are not.
    """
    engine, _ = await _seeded()
    outcome = await engine.retrieve("quick brown fox", top_k=3, exact=True)

    rendered = repr(outcome)
    assert "fox" not in rendered
    assert "Revenue" not in rendered
    assert str(len(outcome.evidence.documents)) in rendered


@pytest.mark.asyncio
async def test_disabled_graph_arm_is_absent_and_enabling_it_contributes() -> None:
    """
    The structural arm appears only when enabled, and then it works.

    Off by default because a structural search with no constraint would return
    an arbitrary slice of the corpus.
    """
    engine, _ = await _seeded()
    assert "graph" not in engine.arm_names

    enabled, _ = await _seeded(RetrievalConfig(graph=GraphConfig(enabled=True)))
    assert "graph" in enabled.arm_names

    outcome = await enabled.retrieve(
        "quick brown fox",
        top_k=3,
        exact=True,
        filters=[
            RetrievalFilter(
                field=FilterField.DOCUMENT_ID,
                operator=FilterOperator.EQUALS,
                values=("doc1",),
            )
        ],
    )
    assert "graph" in outcome.diagnostics.contributing_arms


@pytest.mark.asyncio
async def test_structural_candidates_carry_contiguous_ranks() -> None:
    """
    Structural matches are ranked by position, not all tied at zero.

    Rank-based fusion reads these as an ordering. Giving every match rank zero
    would tie them all at the top and award each the maximum reciprocal rank,
    so a filter matching many nodes would outweigh every dense and lexical
    result at once.
    """
    stores = build_memory_stores()
    await stores.catalog.register(SEGMENTATION, EMBEDDING, activate=True)
    nodes = [unit.node for unit in _units()]
    await stores.nodes.put_many(nodes, SEGMENTATION)

    candidates = await stores.relations.structural_search(
        [
            RetrievalFilter(
                field=FilterField.DOCUMENT_ID,
                operator=FilterOperator.EQUALS,
                values=("doc1",),
            )
        ],
        SEGMENTATION,
        10,
    )

    assert len(candidates) > 1
    assert [candidate.rank for candidate in candidates] == list(
        range(len(candidates))
    )
