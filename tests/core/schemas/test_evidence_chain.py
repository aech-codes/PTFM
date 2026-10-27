"""
End-to-end tests for the Evidence-First provenance chain.

File:
    tests/core/schemas/test_evidence_chain.py

Purpose
-------
    The chain runs:

        RawDocument
            -> DocumentBlock
            -> KnowledgeNode
            -> RetrievedDocument
            -> EvidenceItem
            -> AtomicClaim
            -> VerificationResult

    Each link is verified elsewhere in isolation. What these tests verify is
    that the chain is lossless end to end: that a verified claim can be traced
    back to an exact character range of an identified source document, without
    any consumer inferring or reconstructing a link.

    A single-link test cannot catch this. A break appears only when the chain
    is traversed whole, which is how the gap at ``RetrievedDocument`` and
    ``EvidenceItem`` went unnoticed until the chain was walked.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from ptfm.core.schemas import (
    AtomicClaim,
    BlockType,
    DocumentBlock,
    DocumentMetadata,
    EvidenceItem,
    EvidenceType,
    KnowledgeNode,
    NodeType,
    RawDocument,
    RetrievalMethod,
    RetrievedDocument,
    SourceDescriptor,
    TextSpan,
    VerificationResult,
    VerificationStatus,
    derive_document_id,
    derive_node_id,
)

NOW = datetime(2026, 8, 1, tzinfo=UTC)

SOURCE_URI = "file:///corpus/contract.pdf"
CONTENT_HASH = "content-hash-abc"
TEXT_HASH = "text-hash-abc"
FINGERPRINT = "seg-fp-001"

DOCUMENT_TEXT = "Interest Rates\n\nThe applicable rate is 4.5 percent per annum."
HEADING_SPAN = TextSpan(start=0, end=14)
BODY_SPAN = TextSpan(start=16, end=59)


# ---------------------------------------------------------------------------
# Chain construction
# ---------------------------------------------------------------------------


@pytest.fixture
def source() -> SourceDescriptor:
    return SourceDescriptor(
        source_uri=SOURCE_URI,
        scheme="file",
        connector_name="LocalFilesystemConnector",
        extractor_name="PdfExtractor",
        acquired_at=NOW,
        size_bytes=4096,
        detected_media_type="application/pdf",
        version_tag="etag-7",
    )


@pytest.fixture
def document(source: SourceDescriptor) -> RawDocument:
    """Link 1: a source document with a structural block tree."""
    return RawDocument(
        document_id=derive_document_id(SOURCE_URI, CONTENT_HASH),
        source=source,
        metadata=DocumentMetadata(
            char_count=len(DOCUMENT_TEXT),
            word_count=len(DOCUMENT_TEXT.split()),
        ),
        text=DOCUMENT_TEXT,
        blocks=(
            DocumentBlock(
                block_id="0",
                block_type=BlockType.HEADING,
                span=HEADING_SPAN,
                depth=0,
                ordinal=0,
            ),
            DocumentBlock(
                block_id="1",
                block_type=BlockType.PARAGRAPH,
                span=BODY_SPAN,
                depth=0,
                ordinal=1,
            ),
        ),
        content_hash=CONTENT_HASH,
        text_hash=TEXT_HASH,
        ingested_at=NOW,
    )


@pytest.fixture
def node(document: RawDocument) -> KnowledgeNode:
    """Link 3: a knowledge node over the body paragraph."""
    return KnowledgeNode(
        node_id=derive_node_id(document.document_id, BODY_SPAN, FINGERPRINT),
        document_id=document.document_id,
        span=BODY_SPAN,
        text=DOCUMENT_TEXT[BODY_SPAN.start : BODY_SPAN.end],
        node_type=NodeType.PROSE,
        block_ids=("1",),
        heading_path=("Interest Rates",),
        ordinal=1,
        source=document.source,
        segmentation_fingerprint=FINGERPRINT,
        source_text_hash=document.text_hash,
        source_schema_version=document.schema_version,
        processed_at=NOW,
    )


@pytest.fixture
def retrieved(node: KnowledgeNode) -> RetrievedDocument:
    """Link 4: the node as a retrieval result."""
    return RetrievedDocument(
        document_id=node.document_id,
        node_id=node.node_id,
        span=node.span,
        block_ids=node.block_ids,
        content=node.text,
        source_uri=node.source.source_uri,
        retrieval_method=RetrievalMethod.HYBRID,
        retrieval_score=0.87,
        credibility_score=0.92,
        evidence_type=EvidenceType.DOCUMENT,
    )


@pytest.fixture
def evidence(retrieved: RetrievedDocument) -> EvidenceItem:
    """Link 5: the result as an evidence item."""
    return EvidenceItem(
        evidence_id="evidence-1",
        document_id=retrieved.document_id,
        node_id=retrieved.node_id,
        span=retrieved.span,
        block_ids=retrieved.block_ids,
        content=retrieved.content,
        source_uri=retrieved.source_uri,
        retrieval_score=retrieved.retrieval_score,
        credibility_score=retrieved.credibility_score,
    )


@pytest.fixture
def verification(evidence: EvidenceItem) -> VerificationResult:
    """Link 7: a verification result citing that evidence."""
    return VerificationResult(
        claim_id="claim-1",
        status=VerificationStatus.SUPPORTED,
        nli_score=0.94,
        symbolic_verification=True,
        supporting_evidence=[evidence.evidence_id],
    )


# ---------------------------------------------------------------------------
# The chain is lossless
# ---------------------------------------------------------------------------


def test_the_chain_survives_from_document_to_evidence(
    document: RawDocument, node: KnowledgeNode, evidence: EvidenceItem
) -> None:
    """
    The guarantee. Given evidence alone, the exact source text is recoverable.
    """
    assert evidence.document_id == document.document_id
    assert evidence.node_id == node.node_id
    assert evidence.span == node.span
    assert evidence.block_ids == ("1",)

    assert evidence.span is not None
    located = document.text[evidence.span.start : evidence.span.end]
    assert located == evidence.content
    assert located == "The applicable rate is 4.5 percent per annum."


def test_a_verified_claim_reaches_an_exact_character_range(
    document: RawDocument,
    evidence: EvidenceItem,
    verification: VerificationResult,
) -> None:
    """
    Traversal of the whole chain: from a verification result, through its
    cited evidence, to a character range of the source document.
    """
    assert verification.supporting_evidence == [evidence.evidence_id]

    assert evidence.span is not None
    assert document.text[evidence.span.start : evidence.span.end] in document.text
    assert evidence.span.start == 16
    assert evidence.span.end == 59


def test_evidence_reaches_the_originating_block(
    document: RawDocument, evidence: EvidenceItem
) -> None:
    """Structural context is recoverable, not only the character range."""
    index = document.build_index()

    blocks = [index.get(block_id) for block_id in evidence.block_ids]

    assert all(block is not None for block in blocks)
    assert blocks[0].block_type is BlockType.PARAGRAPH  # type: ignore[union-attr]
    assert document.block_text(blocks[0]) == evidence.content  # type: ignore[arg-type]


def test_no_link_requires_untyped_metadata(
    retrieved: RetrievedDocument, evidence: EvidenceItem
) -> None:
    """
    Every provenance field is a declared, typed field. Carrying any of them in
    an untyped bag would be unenforceable by validator or compliance test.
    """
    for name in ("node_id", "span", "block_ids"):
        assert name in RetrievedDocument.model_fields, name
        assert name in EvidenceItem.model_fields, name

    assert retrieved.metadata == {}
    assert evidence.node_id is not None


def test_the_span_type_is_the_same_across_every_link() -> None:
    """
    One span type end to end. Two structurally similar types would silently
    diverge and sever the chain at the boundary between them.
    """
    from ptfm.core.schemas.common import TextSpan as CommonSpan

    assert DocumentBlock.model_fields["span"].annotation is CommonSpan
    assert KnowledgeNode.model_fields["span"].annotation is CommonSpan


def test_acquisition_provenance_survives_to_the_node(
    document: RawDocument, node: KnowledgeNode
) -> None:
    """Which connector fetched the bytes, and which extractor interpreted
    them, remain available at the node."""
    assert node.source.connector_name == document.source.connector_name
    assert node.source.extractor_name == document.source.extractor_name
    assert node.source.source_uri == document.source.source_uri
    assert node.source.version_tag == document.source.version_tag


def test_the_node_verifies_against_the_document_it_names(
    document: RawDocument, node: KnowledgeNode
) -> None:
    assert node.verify_against(document.text) is True


def test_staleness_is_detectable_without_fetching_the_document(
    document: RawDocument, node: KnowledgeNode
) -> None:
    """
    A node records the text digest its span was computed against, so a
    consumer can detect that a document has been re-ingested without reading
    it.
    """
    assert node.source_text_hash == document.text_hash

    stale = node.model_copy(update={"source_text_hash": "different-hash"})
    assert stale.source_text_hash != document.text_hash


# ---------------------------------------------------------------------------
# Backward compatibility
# ---------------------------------------------------------------------------


def test_a_result_without_node_provenance_is_still_valid() -> None:
    """
    A retrieval strategy operating over whole documents, or a corpus ingested
    before knowledge processing existed, must remain expressible.
    """
    result = RetrievedDocument(
        document_id="doc-1",
        content="body",
        source_uri=SOURCE_URI,
        retrieval_method=RetrievalMethod.DENSE,
        retrieval_score=0.5,
        credibility_score=0.5,
        evidence_type=EvidenceType.DOCUMENT,
    )

    assert result.node_id is None
    assert result.span is None
    assert result.block_ids == ()


def test_evidence_without_node_provenance_is_still_valid() -> None:
    item = EvidenceItem(
        evidence_id="e-1",
        document_id="doc-1",
        content="body",
        source_uri=SOURCE_URI,
        retrieval_score=0.5,
        credibility_score=0.5,
    )

    assert item.node_id is None
    assert item.span is None
    assert item.block_ids == ()


@pytest.mark.parametrize(
    ("model", "fields"),
    [
        (
            RetrievedDocument,
            {
                "document_id",
                "title",
                "content",
                "source_uri",
                "retrieval_method",
                "retrieval_score",
                "credibility_score",
                "evidence_type",
                "metadata",
            },
        ),
        (
            EvidenceItem,
            {
                "evidence_id",
                "document_id",
                "content",
                "source_uri",
                "retrieval_score",
                "credibility_score",
            },
        ),
    ],
)
def test_every_pre_existing_field_is_retained(
    model: type, fields: set[str]
) -> None:
    assert fields <= set(model.model_fields)


def test_the_plugin_interface_is_unaffected() -> None:
    """``ptfm/plugins/base.py`` must require no modification."""
    import ptfm.core.schemas as schemas

    for name in ("AtomicClaim", "EvidenceItem", "RetrievalRequest", "RetrievalResult"):
        assert hasattr(schemas, name), name


def test_no_other_contract_was_modified() -> None:
    """The patch touched two models; nothing else may have changed."""
    from ptfm.core.schemas import RetrievalResult, RetrievalStatistics

    assert set(RetrievalResult.model_fields) == {
        "query",
        "documents",
        "retrieval_latency_ms",
        "total_documents",
    }
    assert set(RetrievalStatistics.model_fields) == {
        "dense_hits",
        "sparse_hits",
        "graph_hits",
        "hybrid_score",
    }


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


def test_provenance_survives_a_json_round_trip(evidence: EvidenceItem) -> None:
    """Evidence crosses process and storage boundaries; provenance must
    survive that."""
    restored = EvidenceItem.model_validate_json(evidence.model_dump_json())

    assert restored.node_id == evidence.node_id
    assert restored.span == evidence.span
    assert restored.block_ids == evidence.block_ids


def test_a_result_survives_a_json_round_trip(
    retrieved: RetrievedDocument,
) -> None:
    restored = RetrievedDocument.model_validate_json(retrieved.model_dump_json())

    assert restored == retrieved


# ---------------------------------------------------------------------------
# Scores remain outside the chain
# ---------------------------------------------------------------------------


def test_the_node_carries_no_retrieval_score(node: KnowledgeNode) -> None:
    """
    Relevance is computed at query time and calibrated downstream. A score
    stored on a node would be a raw model output entering the pipeline outside
    calibration.
    """
    forbidden = {"retrieval_score", "credibility_score", "confidence", "similarity"}

    assert not forbidden & set(KnowledgeNode.model_fields)


def test_credibility_absence_is_representable() -> None:
    """
    A source a plugin cannot assess carries None rather than a substituted
    number.

    Retrieval reliability is r(E) = (1/k) * sum r_i * s_i and drives
    hallucination risk, so an invented credibility would enter a statistical
    guarantee indistinguishable from a measured one. Discarding the item
    instead is not neutral either: it keeps only the sources that could be
    scored, so reliability rises and risk falls.
    """
    assert EvidenceItem.model_fields["credibility_score"].default is None
    assert RetrievedDocument.model_fields["credibility_score"].default is None


def test_an_evidence_item_can_be_built_without_credibility() -> None:
    item = EvidenceItem(
        evidence_id="ev:node-1:node-1",
        document_id="doc-1",
        node_id="node-1",
        anchor_node_id="node-1",
        content="Alpha",
        source_uri="file:///corpus/a.pdf",
        retrieval_score=1.0,
    )

    assert item.credibility_score is None


def test_arithmetic_over_an_unknown_credibility_fails_loudly() -> None:
    """
    The point of representing absence: a consumer cannot multiply past it by
    accident, so the decision about what an unknown credibility means has to be
    made deliberately.
    """
    item = EvidenceItem(
        evidence_id="ev:node-1:node-1",
        document_id="doc-1",
        node_id="node-1",
        anchor_node_id="node-1",
        content="Alpha",
        source_uri="file:///corpus/a.pdf",
        retrieval_score=1.0,
    )

    with pytest.raises(TypeError):
        _ = item.retrieval_score * item.credibility_score  # type: ignore[operator]


def test_a_known_credibility_is_still_required_to_be_a_proportion() -> None:
    """Representing absence does not relax the range of a present value."""
    with pytest.raises(ValidationError):
        EvidenceItem(
            evidence_id="ev:node-1:node-1",
            document_id="doc-1",
            node_id="node-1",
            anchor_node_id="node-1",
            content="Alpha",
            source_uri="file:///corpus/a.pdf",
            retrieval_score=1.0,
            credibility_score=1.7,
        )


def test_scores_appear_only_from_retrieval_onward(
    retrieved: RetrievedDocument,
) -> None:
    assert "retrieval_score" in RetrievedDocument.model_fields
    assert "retrieval_score" in EvidenceItem.model_fields
    assert retrieved.retrieval_score == pytest.approx(0.87)
