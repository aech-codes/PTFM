"""
PTFM Knowledge Schemas

File:
    ptfm/core/schemas/knowledge.py

Layer:
    Knowledge. May import from ``common`` and ``ingestion``.

Description:
    Declares the canonical output contract of the Knowledge Processing Module
    (Module 2): the knowledge node, its embedding, and the structural
    relations between nodes.

Evidence-First contract
-----------------------
    A ``KnowledgeNode`` is the object every remaining module reasons over, so
    it must carry the entire provenance chain rather than a reference to it.
    Given a node alone, and with no other object in scope, a consumer can
    recover:

        document_id                 the source document
        block_ids                   the originating structural blocks
        span                        the exact character range in that document
        text                        the exact substring at that range
        source                      connector, extractor, URI, acquisition time
        segmentation_fingerprint    the configuration that placed the boundary
        source_text_hash            the document text the span was computed on
        source_schema_version       the document contract the span assumes

    Embedding model identity is recorded on ``NodeEmbedding`` rather than on
    the node, and is recovered by joining on ``node_id``. This is deliberate
    and is explained under "Why embeddings are separate" below.

Two artifacts, two lifecycles
-----------------------------
    Segmentation is deterministic and cheap. Embedding is model-versioned and
    expensive. Their lifecycles diverge the moment an embedding model is
    replaced: every vector is invalidated while every boundary, span, and
    provenance record remains exactly correct.

    Were the vector a field on the node, replacing a model would force a
    re-segmentation of the corpus, recomputing text and spans that did not
    change, and would produce new node identifiers, breaking every stored
    citation.

    They are therefore separate artifacts joined by ``node_id``. A corpus can
    be re-embedded without reprocessing, and two models can coexist over one
    node set during a migration.

Text is never mutated
---------------------
    ``KnowledgeNode.text`` is the exact substring of ``RawDocument.text`` at
    the node's span. No cleaning, header stripping, or context injection is
    performed: the moment the stored text ceases to be a substring, the span
    no longer identifies the text the node contains, and a verifier citing
    that span resolves to text the node does not hold.

    Structural context reaches the embedding through ``heading_path``,
    composed at embed time. What is stored stays exact; what is embedded
    carries context.

Invariant enforcement
---------------------
    The complete invariant, ``text == raw.text[span]``, requires the source
    document and is verified by the assembly and validation stages.

    What is enforced here is every condition checkable without that document,
    including the necessary condition ``len(text) == span.length``. A node
    that fails this can never satisfy the full invariant, so catching it at
    construction fails the artifact before it can reach a knowledge base.

Stability
---------
    These are cross-module contracts, consumed by Retrieval, Verification,
    Calibration, and domain plugins. A change here is a change to the
    interface between modules.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime
from enum import StrEnum
from typing import Dict, Final, Optional
from uuid import NAMESPACE_URL, UUID, uuid5

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ptfm.core.schemas.common import TextSpan
from ptfm.core.schemas.ingestion import SourceDescriptor

__all__ = [
    "KNOWLEDGE_NODE_SCHEMA_VERSION",
    "PTFM_NODE_NAMESPACE",
    "derive_node_id",
    "NodeType",
    "RelationType",
    "KnowledgeNode",
    "NodeEmbedding",
    "NodeRelation",
    "KnowledgeNodeBatch",
]


KNOWLEDGE_NODE_SCHEMA_VERSION: Final[str] = "1.0"

#: Namespace for deterministic UUIDv5 node identifiers. Derived from a stable
#: URL so that the namespace itself is auditable and reproducible, and
#: distinct from the document namespace so that a node identifier can never
#: collide with a document identifier.
PTFM_NODE_NAMESPACE: Final[UUID] = uuid5(
    NAMESPACE_URL, "https://ptfm.dev/schemas/1.0/knowledge-node"
)

#: Tolerance within which a vector declaring itself normalized must have unit
#: magnitude. Accommodates accumulated floating point error across the
#: thousands of terms in a high-dimensional vector without admitting a vector
#: that was never normalized at all.
_UNIT_NORM_TOLERANCE: Final[float] = 1e-3


def derive_node_id(
    document_id: str,
    span: TextSpan,
    segmentation_fingerprint: str,
) -> str:
    """
    Derive the deterministic canonical identifier for a knowledge node.

    Identity is a pure function of the source document, the character range,
    and the segmentation configuration that placed that range. Reprocessing
    an unchanged document under unchanged configuration therefore reproduces
    the same identifiers, which is what makes processing idempotent and keeps
    citations stable across index rebuilds.

    The segmentation fingerprint participates deliberately. A boundary is a
    function of configuration, so two configurations must produce disjoint
    identifier spaces. Were the fingerprint omitted, a node retrieved from an
    index built under one configuration could silently resolve against a
    corpus processed under another, and the text at that identifier would not
    be the text that was indexed.

    Parameters
    ----------
    document_id:
        Canonical identifier of the source document.
    span:
        Character range into the source document text.
    segmentation_fingerprint:
        Digest of every configuration item able to change a span boundary.

    Returns
    -------
    str
        String form of a UUIDv5 in :data:`PTFM_NODE_NAMESPACE`.

    Raises
    ------
    ValueError
        If the document identifier or fingerprint is empty.
    """
    if not document_id:
        raise ValueError("document_id must be a non-empty string")
    if not segmentation_fingerprint:
        raise ValueError("segmentation_fingerprint must be a non-empty string")

    # The separator prevents field-boundary ambiguity: without it, differing
    # combinations could concatenate to the same string.
    key = (
        f"{document_id}\x00{span.start}\x00{span.end}"
        f"\x00{segmentation_fingerprint}"
    )
    return str(uuid5(PTFM_NODE_NAMESPACE, key))


# ----------------------------------------------------------------------------
# Taxonomy
# ----------------------------------------------------------------------------


class NodeType(StrEnum):
    """
    Structural character of a knowledge node's content.

    Derived from the block types the node covers, not inferred from its text.
    A node is what its source structure says it is.

    Retrieval and chunk rendering both benefit: a table region is presented
    and scored differently from prose, and a downstream consumer should not
    have to guess which it holds.
    """

    #: Continuous prose. The common case.
    PROSE = "PROSE"
    #: A heading with no accompanying body, which occurs where a section is
    #: large enough that its heading forms its own node.
    HEADING = "HEADING"
    #: One or more complete table rows.
    TABLE_REGION = "TABLE_REGION"
    #: One or more complete list items.
    LIST_REGION = "LIST_REGION"
    #: Source code or preformatted content, whose internal whitespace is
    #: significant.
    CODE = "CODE"
    #: A figure together with its caption.
    FIGURE_REGION = "FIGURE_REGION"
    #: A quotation.
    QUOTE = "QUOTE"
    #: Several structural kinds within one node, which occurs when a size
    #: budget spans a boundary between them.
    MIXED = "MIXED"


class RelationType(StrEnum):
    """
    Structural relation between two knowledge nodes.

    Deliberately structural only. There is no ``SIMILAR_TO``, ``ENTAILS``, or
    ``CONTRADICTS``: those are verification judgements, and a judgement
    recorded as though it were structure would enter the knowledge base
    outside the calibration layer that must govern it.

    Every relation here is recoverable from the source document's block tree,
    so a relation is a fact about the document rather than a claim about
    meaning. Both Retrieval and Verification consume them, as peers.
    """

    #: The source node immediately precedes the target in reading order.
    PRECEDES = "PRECEDES"
    #: Inverse of ``PRECEDES``.
    FOLLOWS = "FOLLOWS"
    #: The target node holds the section heading under which the source falls.
    PARENT_SECTION = "PARENT_SECTION"
    #: Inverse of ``PARENT_SECTION``.
    CHILD_SECTION = "CHILD_SECTION"
    #: Both nodes cover rows of one table.
    SAME_TABLE = "SAME_TABLE"
    #: Both nodes cover items of one list.
    SAME_LIST = "SAME_LIST"
    #: Both nodes originate on one page of a paginated document.
    SAME_PAGE = "SAME_PAGE"


# ----------------------------------------------------------------------------
# Knowledge node
# ----------------------------------------------------------------------------


class KnowledgeNode(BaseModel):
    """
    A provenance-preserving unit of retrievable content.

    The canonical output of Knowledge Processing and the object on which
    Retrieval, Verification, Calibration, and domain plugins operate.

    Self-containment
    ----------------
    A node carries its text rather than referencing it, because a node is
    transported independently of its document. A vector store returns a node
    with no ``RawDocument`` in scope, and requiring a document fetch to render
    a search result would place a read on the retrieval hot path.

    This differs from the ingestion module's blocks, which store spans alone
    because they are never transported without their document. The divergence
    risk that motivated that decision is neutralized here by the
    construction-time invariants below.

    Invariants enforced at construction
    -----------------------------------
    1. ``len(text)`` equals ``span.length``. Necessary for the full invariant
       ``text == raw.text[span]``, and checkable without the source document.
    2. ``block_ids`` is non-empty. A node with no originating block could not
       be traced to document structure.
    3. ``block_ids`` contains no duplicates.
    4. ``heading_path`` contains no empty entries.
    5. ``page_numbers`` is ascending and free of duplicates.
    6. ``segmentation_fingerprint`` is present.

    The complete text invariant requires the source document and is verified
    by the assembly and validation stages, which raise
    ``NodeSpanIntegrityError`` on violation.

    Attributes
    ----------
    node_id:
        Deterministic identifier from :func:`derive_node_id`. Typed as ``str``
        to join directly against ``NodeEmbedding`` and ``NodeRelation``, and
        against evidence records downstream.
    document_id:
        Identifier of the source ``RawDocument``. The first link of the
        provenance chain.
    span:
        Character range into ``RawDocument.text``. The same span type the
        ingestion layer uses, so block spans and node spans index the same
        text and remain directly comparable.
    text:
        The exact substring of ``RawDocument.text`` at :attr:`span`. Never
        cleaned, trimmed, or augmented.
    node_type:
        Structural character, derived from the covered blocks.
    block_ids:
        Identifiers of the covered blocks, in document order. Recovers the
        structure the flat text loses, and lets a consumer reach the original
        block tree without inferring which blocks were involved.
    heading_path:
        Ancestor headings, outermost first. Supplies the context that makes an
        isolated chunk interpretable, without that context contaminating
        :attr:`text`.
    ordinal:
        Zero-based position among the nodes of this document.
    page_numbers:
        One-based pages the node originates on, for paginated formats. A
        citable location for a human reader.
    source:
        Acquisition and interpretation provenance, reused verbatim from the
        ingestion layer rather than re-declared. One provenance vocabulary
        spans the whole pipeline.
    metadata:
        Enricher output. String-valued so that structured domain objects
        cannot enter a domain-agnostic schema.
    segmentation_fingerprint:
        Digest of every configuration item able to change a span boundary.
        Participates in :attr:`node_id`, so differing configurations yield
        disjoint identifier spaces.
    source_text_hash:
        Digest of the ``RawDocument.text`` the span was computed against.
        Lets a consumer detect that a node has become stale relative to a
        re-ingested document without fetching that document.
    source_schema_version:
        Version of the ``RawDocument`` contract the span assumes. A future
        contract altering text normalization would shift every offset, and
        recording the version makes that mismatch detectable rather than
        silently wrong.
    processed_at:
        Timezone-aware instant at which the node was produced.
    schema_version:
        Version of this contract, for corpus migration.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    node_id: str = Field(min_length=1)
    document_id: str = Field(min_length=1)
    span: TextSpan
    text: str
    node_type: NodeType
    block_ids: tuple[str, ...]
    heading_path: tuple[str, ...] = ()
    ordinal: int = Field(ge=0)
    page_numbers: tuple[int, ...] = ()
    source: SourceDescriptor
    metadata: Dict[str, str] = Field(default_factory=dict)
    segmentation_fingerprint: str = Field(min_length=1)
    source_text_hash: str = Field(min_length=1)
    source_schema_version: str = Field(min_length=1)
    processed_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    schema_version: str = KNOWLEDGE_NODE_SCHEMA_VERSION

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    @model_validator(mode="after")
    def _validate_text_matches_span_length(self) -> KnowledgeNode:
        """
        Ensure the stored text is the right length for its span.

        A necessary condition for ``text == raw.text[span]``, and the
        strongest form of that invariant checkable without the source
        document. A node failing it can never satisfy the full invariant, so
        rejecting it here fails the artifact before it can reach a knowledge
        base.

        Returns
        -------
        KnowledgeNode
            The validated instance.

        Raises
        ------
        ValueError
            If the text length disagrees with the span length.
        """
        if len(self.text) != self.span.length:
            raise ValueError(
                f"KnowledgeNode text length ({len(self.text)}) does not match "
                f"span length ({self.span.length}) for span "
                f"[{self.span.start}, {self.span.end}). The stored text must "
                f"be the exact substring of the source document at this span."
            )
        return self

    @model_validator(mode="after")
    def _validate_block_references(self) -> KnowledgeNode:
        """
        Ensure the node names the blocks it originates from.

        Returns
        -------
        KnowledgeNode
            The validated instance.

        Raises
        ------
        ValueError
            If no block is named, or a block is named twice.
        """
        if not self.block_ids:
            raise ValueError(
                "KnowledgeNode must reference at least one originating block; "
                "a node with no block reference cannot be traced to document "
                "structure"
            )
        if len(set(self.block_ids)) != len(self.block_ids):
            raise ValueError(
                f"KnowledgeNode block_ids contains duplicates: "
                f"{list(self.block_ids)}"
            )
        return self

    @model_validator(mode="after")
    def _validate_structural_fields(self) -> KnowledgeNode:
        """
        Ensure the heading path and page numbers are well formed.

        Returns
        -------
        KnowledgeNode
            The validated instance.

        Raises
        ------
        ValueError
            If a heading entry is blank, or page numbers are unordered,
            duplicated, or non-positive.
        """
        if any(not heading.strip() for heading in self.heading_path):
            raise ValueError(
                "KnowledgeNode heading_path must not contain blank entries"
            )

        pages = self.page_numbers
        if pages:
            if any(page < 1 for page in pages):
                raise ValueError(
                    f"KnowledgeNode page_numbers must be one-based, got "
                    f"{list(pages)}"
                )
            if list(pages) != sorted(set(pages)):
                raise ValueError(
                    f"KnowledgeNode page_numbers must be ascending and free "
                    f"of duplicates, got {list(pages)}"
                )
        return self

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def char_count(self) -> int:
        """
        Length of the node's text.

        Returns
        -------
        int
            Character count, equal to the span length by invariant.
        """
        return len(self.text)

    @property
    def heading_context(self) -> str:
        """
        The heading path rendered as a single readable trail.

        Supplies the structural context an isolated node otherwise lacks, for
        display and for embedding composition. Never stored on :attr:`text`.

        Returns
        -------
        str
            Headings joined outermost first, empty when the node has no
            ancestor headings.
        """
        return " > ".join(self.heading_path)

    @property
    def page_range(self) -> Optional[tuple[int, int]]:
        """
        First and last page the node originates on.

        Returns
        -------
        Optional[tuple[int, int]]
            The inclusive page range, or ``None`` for an unpaginated format.
        """
        if not self.page_numbers:
            return None
        return self.page_numbers[0], self.page_numbers[-1]

    @property
    def spans_multiple_pages(self) -> bool:
        """
        Whether the node crosses a page boundary.

        Returns
        -------
        bool
            ``True`` when the node originates on more than one page.
        """
        return len(self.page_numbers) > 1

    def verify_against(self, document_text: str) -> bool:
        """
        Verify the complete provenance invariant against a source text.

        The definitive Evidence-First check: the node's text must be exactly
        what its span identifies in the document it claims to come from.
        Available to any consumer holding both, so the guarantee is auditable
        after the fact rather than only at production time.

        Parameters
        ----------
        document_text:
            The ``RawDocument.text`` this node claims to originate from.

        Returns
        -------
        bool
            ``True`` when the span resolves to exactly this node's text.
        """
        if self.span.end > len(document_text):
            return False
        return document_text[self.span.start : self.span.end] == self.text

    def derives_identity_from(self) -> tuple[str, int, int, str]:
        """
        Return the inputs from which this node's identifier is derived.

        Lets a consumer recompute and confirm :attr:`node_id` without knowing
        the derivation's internal form, which makes an identifier mismatch
        diagnosable rather than merely detectable.

        Returns
        -------
        tuple[str, int, int, str]
            Document identifier, span start, span end, and segmentation
            fingerprint.
        """
        return (
            self.document_id,
            self.span.start,
            self.span.end,
            self.segmentation_fingerprint,
        )


# ----------------------------------------------------------------------------
# Embedding
# ----------------------------------------------------------------------------


class NodeEmbedding(BaseModel):
    """
    A vector representation of a knowledge node under a named model.

    Declared separately from :class:`KnowledgeNode` because the two have
    different lifecycles. Replacing an embedding model invalidates every
    vector while leaving every node boundary, span, and provenance record
    correct. Coupling them would make a model swap force a re-segmentation of
    the corpus and would change node identifiers, breaking stored citations.

    Attributes
    ----------
    node_id:
        Identifier of the embedded node. The join key.
    document_id:
        Identifier of the source document. Duplicated from the node so that a
        document can be deleted from a vector store without first joining to
        the node set.
    vector:
        The embedding.
    dimension:
        Declared dimension, validated against the vector.
    model_id:
        Identifier of the model that produced the vector, for example
        ``"BAAI/bge-large-en-v1.5"``. Without it, a corpus embedded across a
        model upgrade becomes an undifferentiated mixture whose similarity
        scores are meaningless.
    model_revision:
        Commit, tag, or version pin, when the provider reports one. A model
        identifier alone is not always sufficient: weights behind a name can
        change.
    normalized:
        Whether the vector has unit magnitude. Determines whether a dot
        product equals cosine similarity, so a wrong value silently corrupts
        every comparison. Verified at construction when ``True``.
    composition_id:
        Identifier of the composition that produced the embedded text, since
        a node embedded with its heading path differs from the same node
        embedded without it.
    query_encoding:
        Identifier of the transformation a query must undergo to be comparable
        with this vector.

        Recorded because a passage and a query must be encoded compatibly for
        their similarity to mean anything, and several widely used retrieval
        models are trained asymmetrically. A consumer holding this vector can
        verify that the provider it intends to query with applies the same
        convention, rather than inferring provider behaviour from
        configuration it may not share.

        The identifier names the convention; applying it is the provider's
        own responsibility, through ``EmbeddingProvider.embed_query``. Empty
        for a symmetric model.
    input_text_hash:
        Digest of the text actually submitted to the model. Makes the vector
        reproducible, and makes staleness detectable: a vector whose input
        hash no longer matches its node's current composition is out of date.
    embedded_at:
        Timezone-aware instant at which the vector was produced.
    schema_version:
        Version of this contract.

    Prohibition
    -----------
    No similarity, relevance, or confidence value is recorded here. Similarity
    is computed at query time by Retrieval and calibrated by the conformal
    layer. A score stored on this artifact would be a raw model output
    entering the pipeline outside calibration.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    node_id: str = Field(min_length=1)
    document_id: str = Field(min_length=1)
    vector: tuple[float, ...]
    dimension: int = Field(ge=1)
    model_id: str = Field(min_length=1)
    model_revision: Optional[str] = None
    normalized: bool = False
    composition_id: str = Field(min_length=1)
    query_encoding: str = ""
    input_text_hash: str = Field(min_length=1)
    embedded_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    schema_version: str = KNOWLEDGE_NODE_SCHEMA_VERSION

    @model_validator(mode="after")
    def _validate_dimension(self) -> NodeEmbedding:
        """
        Ensure the declared dimension matches the vector.

        A mismatch means the provider is not the model it reports itself to
        be, or that vectors from two models have been mixed. Either corrupts
        every downstream comparison, silently, so the condition must fail at
        construction.

        Returns
        -------
        NodeEmbedding
            The validated instance.

        Raises
        ------
        ValueError
            If the vector is empty or its length disagrees with
            ``dimension``.
        """
        if not self.vector:
            raise ValueError("NodeEmbedding vector must not be empty")
        if len(self.vector) != self.dimension:
            raise ValueError(
                f"NodeEmbedding declares dimension {self.dimension} but "
                f"carries a vector of length {len(self.vector)}. Vectors of "
                f"differing dimension cannot be compared."
            )
        return self

    @model_validator(mode="after")
    def _validate_normalization_claim(self) -> NodeEmbedding:
        """
        Ensure a vector claiming to be normalized actually is.

        A provider reporting ``normalized=True`` while returning unnormalized
        vectors corrupts every similarity computation downstream, and does so
        invisibly: results are returned, ranked, and acted upon, merely
        wrongly. Verifying the claim converts a silent corruption into a
        construction failure.

        Checked only when the claim is made, so an unnormalized vector costs
        nothing.

        Returns
        -------
        NodeEmbedding
            The validated instance.

        Raises
        ------
        ValueError
            If the vector claims normalization but does not have unit
            magnitude within tolerance.
        """
        if not self.normalized:
            return self

        magnitude = math.sqrt(sum(value * value for value in self.vector))
        if abs(magnitude - 1.0) > _UNIT_NORM_TOLERANCE:
            raise ValueError(
                f"NodeEmbedding declares normalized=True but the vector has "
                f"magnitude {magnitude:.6f}. A false normalization claim makes "
                f"dot product diverge from cosine similarity, corrupting every "
                f"comparison without any visible error."
            )
        return self

    @property
    def model_identity(self) -> str:
        """
        The model identifier together with its revision, where known.

        Returns
        -------
        str
            A single identity string suitable for grouping and comparison.
        """
        if self.model_revision is None:
            return self.model_id
        return f"{self.model_id}@{self.model_revision}"

    def is_comparable_to(self, other: NodeEmbedding) -> bool:
        """
        Report whether two embeddings may be meaningfully compared.

        Two vectors are comparable only when produced by the same model
        revision, at the same dimension, under the same composition, the same
        normalization, and the same query convention. Comparing across any of
        these yields a number that looks like a similarity but means nothing.

        The query convention participates because two corpora embedded under
        differing query conventions are queried differently, so a single query
        vector cannot be compared against both.

        Parameters
        ----------
        other:
            The embedding to compare against.

        Returns
        -------
        bool
            ``True`` when a similarity between the two would be meaningful.
        """
        return (
            self.model_identity == other.model_identity
            and self.dimension == other.dimension
            and self.composition_id == other.composition_id
            and self.normalized == other.normalized
            and self.query_encoding == other.query_encoding
        )


# ----------------------------------------------------------------------------
# Relations
# ----------------------------------------------------------------------------


class NodeRelation(BaseModel):
    """
    A structural relation between two knowledge nodes.

    Derived from the source document's block tree, so every relation is a
    fact about the document rather than a claim about meaning. Consumed by
    Retrieval, for structural expansion of a result set, and by Verification,
    for symbolic reasoning over document structure. The two are peers: neither
    reaches these relations through the other.

    Attributes
    ----------
    source_node_id:
        Identifier of the originating node.
    target_node_id:
        Identifier of the related node.
    relation_type:
        The structural relation that holds.
    document_id:
        Identifier of the document both nodes belong to. Relations are
        intra-document by construction, since they derive from one block tree.
    attributes:
        Additional structural detail, such as the identifier of the shared
        table. String-valued, consistent with node metadata.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    source_node_id: str = Field(min_length=1)
    target_node_id: str = Field(min_length=1)
    relation_type: RelationType
    document_id: str = Field(min_length=1)
    attributes: Dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_distinct_endpoints(self) -> NodeRelation:
        """
        Ensure a node is not related to itself.

        A self-relation carries no information and would create a cycle in
        every structural traversal.

        Returns
        -------
        NodeRelation
            The validated instance.

        Raises
        ------
        ValueError
            If both endpoints are the same node.
        """
        if self.source_node_id == self.target_node_id:
            raise ValueError(
                f"NodeRelation endpoints must differ; both are "
                f"{self.source_node_id!r}"
            )
        return self

    def inverse(self) -> Optional[NodeRelation]:
        """
        Return the relation holding in the opposite direction.

        Lets a consumer traverse structure in either direction without the
        producing stage storing both, which would double the relation count
        for no additional information.

        Returns
        -------
        Optional[NodeRelation]
            The inverse relation, or ``None`` for a symmetric type, where the
            relation already holds in both directions.
        """
        inverses = {
            RelationType.PRECEDES: RelationType.FOLLOWS,
            RelationType.FOLLOWS: RelationType.PRECEDES,
            RelationType.PARENT_SECTION: RelationType.CHILD_SECTION,
            RelationType.CHILD_SECTION: RelationType.PARENT_SECTION,
        }
        inverse_type = inverses.get(self.relation_type)
        if inverse_type is None:
            return None

        return NodeRelation(
            source_node_id=self.target_node_id,
            target_node_id=self.source_node_id,
            relation_type=inverse_type,
            document_id=self.document_id,
            attributes=dict(self.attributes),
        )

    @property
    def is_symmetric(self) -> bool:
        """
        Whether the relation holds equally in both directions.

        Returns
        -------
        bool
            ``True`` for co-membership relations such as sharing a table.
        """
        return self.relation_type in {
            RelationType.SAME_TABLE,
            RelationType.SAME_LIST,
            RelationType.SAME_PAGE,
        }


# ----------------------------------------------------------------------------
# Batch
# ----------------------------------------------------------------------------


class KnowledgeNodeBatch(BaseModel):
    """
    An ordered collection of knowledge nodes.

    Provided for batch hand-off and for fixture serialization in tests. The
    streaming processing path does not materialize batches; it yields nodes
    individually so that memory stays bounded by concurrency rather than by
    corpus size.

    Attributes
    ----------
    nodes:
        The nodes in the batch.
    relations:
        Structural relations among them.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    nodes: tuple[KnowledgeNode, ...] = ()
    relations: tuple[NodeRelation, ...] = ()
