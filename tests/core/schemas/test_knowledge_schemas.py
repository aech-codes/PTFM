"""
Unit and property tests for the knowledge layer schemas.

File:
    tests/core/schemas/test_knowledge_schemas.py

Emphasis
--------
    The Evidence-First chain. A ``KnowledgeNode`` is the object every
    remaining module reasons over, so the tests that matter most are those
    proving a node alone is sufficient to recover its complete provenance, and
    that a node whose text and span disagree cannot be constructed.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime

import pytest
from hypothesis import given, settings, strategies as st
from pydantic import ValidationError

from ptfm.core.schemas import (
    KNOWLEDGE_NODE_SCHEMA_VERSION,
    PTFM_DOCUMENT_NAMESPACE,
    PTFM_NODE_NAMESPACE,
    KnowledgeNode,
    KnowledgeNodeBatch,
    NodeEmbedding,
    NodeRelation,
    NodeType,
    RelationType,
    SourceDescriptor,
    TextSpan,
    derive_document_id,
    derive_node_id,
)

NOW = datetime(2026, 8, 1, tzinfo=UTC)
DOCUMENT_TEXT = "Chapter One\n\nThe applicable rate is 4.5 percent.\n\nSee annex."
DOCUMENT_ID = derive_document_id("file:///corpus/contract.pdf", "hash-abc")
FINGERPRINT = "seg-fp-001"


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def _source(**overrides: object) -> SourceDescriptor:
    fields: dict[str, object] = {
        "source_uri": "file:///corpus/contract.pdf",
        "scheme": "file",
        "connector_name": "LocalFilesystemConnector",
        "extractor_name": "PdfExtractor",
        "acquired_at": NOW,
        "size_bytes": 4096,
        "detected_media_type": "application/pdf",
    }
    fields.update(overrides)
    return SourceDescriptor(**fields)  # type: ignore[arg-type]


def _node(
    *,
    start: int = 13,
    end: int = 47,
    text: str | None = None,
    **overrides: object,
) -> KnowledgeNode:
    span = TextSpan(start=start, end=end)
    resolved = DOCUMENT_TEXT[start:end] if text is None else text
    fields: dict[str, object] = {
        "node_id": derive_node_id(DOCUMENT_ID, span, FINGERPRINT),
        "document_id": DOCUMENT_ID,
        "span": span,
        "text": resolved,
        "node_type": NodeType.PROSE,
        "block_ids": ("1",),
        "heading_path": ("Chapter One",),
        "ordinal": 1,
        "source": _source(),
        "segmentation_fingerprint": FINGERPRINT,
        "source_text_hash": "text-hash-abc",
        "source_schema_version": "1.0",
        "processed_at": NOW,
    }
    fields.update(overrides)
    return KnowledgeNode(**fields)  # type: ignore[arg-type]


def _embedding(**overrides: object) -> NodeEmbedding:
    fields: dict[str, object] = {
        "node_id": "node-1",
        "document_id": DOCUMENT_ID,
        "vector": (0.6, 0.8),
        "dimension": 2,
        "model_id": "BAAI/bge-large-en-v1.5",
        "normalized": True,
        "composition_id": "heading-path-prefixed",
        "input_text_hash": "input-hash-1",
        "embedded_at": NOW,
    }
    fields.update(overrides)
    return NodeEmbedding(**fields)  # type: ignore[arg-type]


def _relation(**overrides: object) -> NodeRelation:
    fields: dict[str, object] = {
        "source_node_id": "node-1",
        "target_node_id": "node-2",
        "relation_type": RelationType.PRECEDES,
        "document_id": DOCUMENT_ID,
    }
    fields.update(overrides)
    return NodeRelation(**fields)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# The Evidence-First guarantee
# ---------------------------------------------------------------------------


def test_a_node_alone_recovers_the_entire_provenance_chain() -> None:
    """
    The guarantee the whole module exists to provide. With no other object in
    scope, every link must be recoverable without inference.
    """
    node = _node()

    assert node.document_id == DOCUMENT_ID
    assert node.block_ids == ("1",)
    assert (node.span.start, node.span.end) == (13, 47)
    assert node.text == DOCUMENT_TEXT[13:47]
    assert node.source.source_uri == "file:///corpus/contract.pdf"
    assert node.source.connector_name == "LocalFilesystemConnector"
    assert node.source.extractor_name == "PdfExtractor"
    assert node.segmentation_fingerprint == FINGERPRINT
    assert node.source_text_hash == "text-hash-abc"
    assert node.source_schema_version == "1.0"


def test_a_node_verifies_itself_against_its_source() -> None:
    """The definitive check, available to any consumer holding both."""
    node = _node()

    assert node.verify_against(DOCUMENT_TEXT) is True


def test_verification_fails_against_a_different_document() -> None:
    node = _node()

    assert node.verify_against("Entirely different content here.") is False


def test_verification_fails_when_the_source_text_changed_beneath_the_node() -> None:
    """A stale node must not silently verify against re-ingested text."""
    node = _node()
    shifted = "PREFIX " + DOCUMENT_TEXT

    assert node.verify_against(shifted) is False


def test_a_node_whose_text_disagrees_with_its_span_cannot_exist() -> None:
    """
    The strongest form of the invariant checkable without the source document.
    A node failing this could never satisfy the full invariant.
    """
    with pytest.raises(ValidationError, match="does not match span length"):
        _node(start=13, end=47, text="wrong length")


def test_the_span_type_is_shared_with_the_ingestion_layer() -> None:
    """
    Block spans and node spans index the same normalized text. Two structurally
    similar span types would silently diverge and sever the chain.
    """
    from ptfm.core.schemas.common import TextSpan as CommonSpan
    from ptfm.core.schemas import DocumentBlock

    assert KnowledgeNode.model_fields["span"].annotation is CommonSpan
    assert DocumentBlock.model_fields["span"].annotation is CommonSpan


def test_the_provenance_record_is_reused_not_redeclared() -> None:
    """One provenance vocabulary spans the whole pipeline."""
    from ptfm.core.schemas.ingestion import SourceDescriptor as IngestionSource

    assert KnowledgeNode.model_fields["source"].annotation is IngestionSource


def test_the_node_carries_the_extractor_identity() -> None:
    """
    Which extractor produced the structure is not inferable from the media
    type, and it determines what a span refers to.
    """
    node = _node()

    assert node.source.extractor_name == "PdfExtractor"


# ---------------------------------------------------------------------------
# Identity
# ---------------------------------------------------------------------------


def test_node_identity_is_deterministic() -> None:
    span = TextSpan(start=0, end=11)

    first = derive_node_id(DOCUMENT_ID, span, FINGERPRINT)
    second = derive_node_id(DOCUMENT_ID, span, FINGERPRINT)

    assert first == second


def test_segmentation_configuration_yields_disjoint_identifiers() -> None:
    """
    A boundary is a function of configuration, so two configurations must not
    share an identifier space. Otherwise a node retrieved from one index could
    silently resolve against a corpus segmented differently.
    """
    span = TextSpan(start=0, end=11)

    assert derive_node_id(DOCUMENT_ID, span, "fp-a") != derive_node_id(
        DOCUMENT_ID, span, "fp-b"
    )


@pytest.mark.parametrize(
    ("document_id", "start", "end"),
    [("other-doc", 0, 11), (DOCUMENT_ID, 1, 11), (DOCUMENT_ID, 0, 12)],
)
def test_identity_varies_with_every_input(
    document_id: str, start: int, end: int
) -> None:
    baseline = derive_node_id(DOCUMENT_ID, TextSpan(start=0, end=11), FINGERPRINT)

    altered = derive_node_id(
        document_id, TextSpan(start=start, end=end), FINGERPRINT
    )

    assert altered != baseline


def test_the_separator_prevents_field_boundary_ambiguity() -> None:
    """Differing inputs must not concatenate to the same key."""
    a = derive_node_id("ab", TextSpan(start=1, end=2), "c")
    b = derive_node_id("a", TextSpan(start=1, end=2), "bc")

    assert a != b


def test_node_and_document_namespaces_are_distinct() -> None:
    """A node identifier must never collide with a document identifier."""
    assert PTFM_NODE_NAMESPACE != PTFM_DOCUMENT_NAMESPACE
    assert PTFM_NODE_NAMESPACE.version == 5


@pytest.mark.parametrize(("doc", "fp"), [("", "fp"), ("doc", "")])
def test_identity_rejects_empty_components(doc: str, fp: str) -> None:
    with pytest.raises(ValueError):
        derive_node_id(doc, TextSpan(start=0, end=1), fp)


def test_a_node_exposes_its_identity_inputs() -> None:
    """An identifier mismatch must be diagnosable, not merely detectable."""
    node = _node()

    document_id, start, end, fingerprint = node.derives_identity_from()

    assert derive_node_id(
        document_id, TextSpan(start=start, end=end), fingerprint
    ) == node.node_id


def test_identity_is_independent_of_processing_time() -> None:
    early = _node(processed_at=datetime(2026, 1, 1, tzinfo=UTC))
    late = _node(processed_at=datetime(2026, 12, 1, tzinfo=UTC))

    assert early.node_id == late.node_id


# ---------------------------------------------------------------------------
# Node validation
# ---------------------------------------------------------------------------


def test_a_node_must_name_its_originating_blocks() -> None:
    with pytest.raises(ValidationError, match="at least one originating block"):
        _node(block_ids=())


def test_duplicate_block_references_are_rejected() -> None:
    with pytest.raises(ValidationError, match="duplicates"):
        _node(block_ids=("1", "1"))


def test_blank_heading_entries_are_rejected() -> None:
    with pytest.raises(ValidationError, match="blank entries"):
        _node(heading_path=("Chapter One", "   "))


@pytest.mark.parametrize("pages", [(2, 1), (1, 1), (0, 1), (-1,)])
def test_malformed_page_numbers_are_rejected(pages: tuple[int, ...]) -> None:
    with pytest.raises(ValidationError):
        _node(page_numbers=pages)


def test_well_formed_page_numbers_are_accepted() -> None:
    node = _node(page_numbers=(3, 4, 5))

    assert node.page_range == (3, 5)
    assert node.spans_multiple_pages is True


def test_an_unpaginated_node_reports_no_page_range() -> None:
    node = _node()

    assert node.page_range is None
    assert node.spans_multiple_pages is False


def test_a_zero_length_node_is_permitted() -> None:
    """A structural marker carries position without characters."""
    node = _node(start=11, end=11, text="")

    assert node.char_count == 0


def test_the_node_is_frozen() -> None:
    node = _node()

    with pytest.raises(ValidationError):
        node.text = "mutated"  # type: ignore[misc]


def test_unknown_fields_are_rejected() -> None:
    with pytest.raises(ValidationError):
        _node(unexpected_field="x")


def test_metadata_rejects_structured_domain_objects() -> None:
    """String-valued metadata is what keeps domain objects out of the core."""
    with pytest.raises(ValidationError):
        _node(metadata={"citation": {"court": "SCOTUS"}})


def test_the_node_exposes_no_confidence_field() -> None:
    forbidden = {"confidence", "score", "relevance", "similarity", "probability"}

    assert not forbidden & set(KnowledgeNode.model_fields)


def test_heading_context_renders_the_path() -> None:
    node = _node(heading_path=("Part I", "Chapter One", "Section 3"))

    assert node.heading_context == "Part I > Chapter One > Section 3"


def test_heading_context_is_not_stored_in_text() -> None:
    """Context reaches the embedding at compose time, never the stored text."""
    node = _node(heading_path=("Chapter One",))

    assert "Chapter One" not in node.text
    assert node.heading_context == "Chapter One"


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------


def test_an_embedding_records_its_model_identity() -> None:
    embedding = _embedding(model_revision="a1b2c3")

    assert embedding.model_identity == "BAAI/bge-large-en-v1.5@a1b2c3"


def test_model_identity_falls_back_to_the_bare_identifier() -> None:
    assert _embedding().model_identity == "BAAI/bge-large-en-v1.5"


def test_a_dimension_mismatch_is_rejected() -> None:
    with pytest.raises(ValidationError, match="cannot be compared"):
        _embedding(vector=(0.6, 0.8), dimension=3)


def test_an_empty_vector_is_rejected() -> None:
    with pytest.raises(ValidationError):
        _embedding(vector=(), dimension=1)


def test_a_false_normalization_claim_is_rejected() -> None:
    """
    A provider reporting normalized=True while returning unnormalized vectors
    corrupts every comparison, invisibly. Verifying converts that into a
    construction failure.
    """
    with pytest.raises(ValidationError, match="without any visible error"):
        _embedding(vector=(3.0, 4.0), dimension=2, normalized=True)


def test_a_genuinely_normalized_vector_is_accepted() -> None:
    embedding = _embedding(vector=(0.6, 0.8), dimension=2, normalized=True)

    assert math.isclose(
        math.sqrt(sum(v * v for v in embedding.vector)), 1.0, abs_tol=1e-6
    )


def test_an_unnormalized_vector_is_not_checked() -> None:
    embedding = _embedding(vector=(3.0, 4.0), dimension=2, normalized=False)

    assert embedding.normalized is False


@pytest.mark.parametrize(
    "overrides",
    [
        {"model_id": "other-model"},
        {"model_revision": "different"},
        {"composition_id": "text-only"},
        {"normalized": False, "vector": (3.0, 4.0)},
        {"query_encoding": "prefix:query: "},
    ],
)
def test_embeddings_differing_in_any_identity_are_incomparable(
    overrides: dict[str, object],
) -> None:
    """A similarity across differing identity looks like a number and means
    nothing."""
    baseline = _embedding()

    other = _embedding(**overrides)

    assert baseline.is_comparable_to(other) is False


def test_embeddings_of_identical_identity_are_comparable() -> None:
    assert _embedding().is_comparable_to(_embedding(node_id="node-2")) is True



def test_an_embedding_records_its_query_convention() -> None:
    """
    A passage and a query must be encoded compatibly for their similarity to
    mean anything, and several widely used models are trained asymmetrically.
    """
    embedding = _embedding(query_encoding="prefix:query: ")

    assert embedding.query_encoding == "prefix:query: "


def test_a_symmetric_model_records_an_empty_convention() -> None:
    assert _embedding().query_encoding == ""


def test_differing_query_conventions_are_incomparable() -> None:
    """Two corpora queried differently cannot share one query vector."""
    symmetric = _embedding()
    asymmetric = _embedding(query_encoding="prefix:query: ")

    assert symmetric.is_comparable_to(asymmetric) is False


def test_the_embedding_exposes_no_similarity_field() -> None:
    """Similarity is computed at query time and calibrated downstream."""
    forbidden = {"score", "similarity", "confidence", "relevance", "distance"}

    assert not forbidden & set(NodeEmbedding.model_fields)


def test_the_embedding_is_a_separate_artifact_from_the_node() -> None:
    """
    Coupling them would make a model swap force re-segmentation and change
    node identifiers, breaking stored citations.
    """
    assert "vector" not in KnowledgeNode.model_fields
    assert "embedding" not in KnowledgeNode.model_fields
    assert "model_id" not in KnowledgeNode.model_fields
    assert "node_id" in NodeEmbedding.model_fields


def test_the_embedding_carries_the_document_identifier() -> None:
    """Enables document-scoped deletion without joining to the node set."""
    assert _embedding().document_id == DOCUMENT_ID


# ---------------------------------------------------------------------------
# Relations
# ---------------------------------------------------------------------------


def test_a_relation_links_two_distinct_nodes() -> None:
    relation = _relation()

    assert relation.source_node_id == "node-1"
    assert relation.target_node_id == "node-2"
    assert relation.relation_type is RelationType.PRECEDES


def test_a_self_relation_is_rejected() -> None:
    with pytest.raises(ValidationError, match="must differ"):
        _relation(target_node_id="node-1")


@pytest.mark.parametrize(
    ("forward", "backward"),
    [
        (RelationType.PRECEDES, RelationType.FOLLOWS),
        (RelationType.FOLLOWS, RelationType.PRECEDES),
        (RelationType.PARENT_SECTION, RelationType.CHILD_SECTION),
        (RelationType.CHILD_SECTION, RelationType.PARENT_SECTION),
    ],
)
def test_directed_relations_invert(
    forward: RelationType, backward: RelationType
) -> None:
    """Traversal in either direction without storing both, which would double
    the relation count for no additional information."""
    inverse = _relation(relation_type=forward).inverse()

    assert inverse is not None
    assert inverse.relation_type is backward
    assert inverse.source_node_id == "node-2"
    assert inverse.target_node_id == "node-1"


@pytest.mark.parametrize(
    "relation_type",
    [RelationType.SAME_TABLE, RelationType.SAME_LIST, RelationType.SAME_PAGE],
)
def test_symmetric_relations_have_no_inverse(
    relation_type: RelationType,
) -> None:
    relation = _relation(relation_type=relation_type)

    assert relation.is_symmetric is True
    assert relation.inverse() is None


def test_relations_declare_only_structural_types() -> None:
    """
    A semantic judgement recorded as structure would enter the knowledge base
    outside the calibration layer that must govern it.
    """
    forbidden = {"SIMILAR_TO", "ENTAILS", "CONTRADICTS", "SUPPORTS", "REFUTES"}

    assert not forbidden & {member.value for member in RelationType}


def test_relations_are_intra_document() -> None:
    """Derived from one block tree, so both endpoints share a document."""
    assert "document_id" in NodeRelation.model_fields


def test_relation_attributes_reject_structured_values() -> None:
    with pytest.raises(ValidationError):
        _relation(attributes={"table": {"id": 3}})


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


def test_a_node_round_trips_through_json() -> None:
    node = _node(page_numbers=(2,), metadata={"topic": "interest"})

    restored = KnowledgeNode.model_validate_json(node.model_dump_json())

    assert restored == node
    assert restored.verify_against(DOCUMENT_TEXT) is True


def test_an_embedding_round_trips_through_json() -> None:
    embedding = _embedding(model_revision="rev-1")

    restored = NodeEmbedding.model_validate_json(embedding.model_dump_json())

    assert restored == embedding


def test_a_relation_round_trips_through_json() -> None:
    relation = _relation(attributes={"table_block_id": "3"})

    restored = NodeRelation.model_validate_json(relation.model_dump_json())

    assert restored == relation


def test_a_batch_round_trips_through_json() -> None:
    batch = KnowledgeNodeBatch(nodes=(_node(),), relations=(_relation(),))

    restored = KnowledgeNodeBatch.model_validate_json(batch.model_dump_json())

    assert restored == batch


def test_the_schema_version_is_recorded() -> None:
    assert _node().schema_version == KNOWLEDGE_NODE_SCHEMA_VERSION
    assert _embedding().schema_version == KNOWLEDGE_NODE_SCHEMA_VERSION


def test_sequence_fields_are_tuples() -> None:
    node = _node(page_numbers=(1,))

    assert isinstance(node.block_ids, tuple)
    assert isinstance(node.heading_path, tuple)
    assert isinstance(node.page_numbers, tuple)
    assert isinstance(_embedding().vector, tuple)


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------


@settings(max_examples=200)
@given(
    start=st.integers(min_value=0, max_value=len(DOCUMENT_TEXT)),
    length=st.integers(min_value=0, max_value=len(DOCUMENT_TEXT)),
)
def test_any_span_of_the_document_produces_a_verifiable_node(
    start: int, length: int
) -> None:
    """Property: for any span, a node built from that span verifies."""
    end = min(start + length, len(DOCUMENT_TEXT))

    node = _node(start=start, end=end, text=DOCUMENT_TEXT[start:end])

    assert node.verify_against(DOCUMENT_TEXT) is True


@settings(max_examples=200)
@given(
    start=st.integers(min_value=0, max_value=40),
    length=st.integers(min_value=1, max_value=20),
    corruption=st.text(min_size=1, max_size=5),
)
def test_any_text_corruption_is_caught_at_construction_or_verification(
    start: int, length: int, corruption: str
) -> None:
    """Property: a node cannot silently hold text its span does not identify."""
    end = min(start + length, len(DOCUMENT_TEXT))
    correct = DOCUMENT_TEXT[start:end]
    corrupted = corruption + correct[len(corruption) :]

    if corrupted == correct:
        return

    if len(corrupted) != end - start:
        with pytest.raises(ValidationError):
            _node(start=start, end=end, text=corrupted)
        return

    node = _node(start=start, end=end, text=corrupted)
    assert node.verify_against(DOCUMENT_TEXT) is False


@settings(max_examples=100)
@given(
    start=st.integers(min_value=0, max_value=50),
    length=st.integers(min_value=0, max_value=10),
    fingerprint=st.text(min_size=1, max_size=12),
)
def test_identity_derivation_is_always_deterministic(
    start: int, length: int, fingerprint: str
) -> None:
    span = TextSpan(start=start, end=start + length)

    assert derive_node_id(DOCUMENT_ID, span, fingerprint) == derive_node_id(
        DOCUMENT_ID, span, fingerprint
    )
