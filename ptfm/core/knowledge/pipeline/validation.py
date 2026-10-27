"""
PTFM Validation Stage

File:
    ptfm/core/knowledge/pipeline/validation.py

Description:
    The final gate before knowledge artifacts leave the module.

What this stage sees that no earlier stage does
-----------------------------------------------
    Every earlier stage has a partial view. Planning sees a plan but no nodes.
    Assembly sees nodes before enrichment. Enrichment sees one node at a time.
    Embedding sees one batch.

    This stage is the only point at which the COMPLETE set of units for a
    document exists, after enrichment and after embedding. The checks below
    are therefore the ones that cannot be performed anywhere else, rather than
    a repetition of what the pipeline already established.

    Two of them matter most:

        Cross-unit comparability
            Every embedding within a document must come from one model, at one
            dimension, under one composition and one normalization. A document
            whose units mix two models produces similarity scores that are
            arithmetically valid and semantically meaningless, and no other
            stage can detect it: ownership is checked per unit, and a batch
            sees only itself.

        Survival
            The nodes emitted must still account for the document. A node lost
            between planning and emission would leave text that no node
            covers, which is the invisible failure the coverage invariant
            exists to prevent, arriving by a route the planning check cannot
            see.

Why a failure here means something different
--------------------------------------------
    A planning failure indicates a defective strategy. A failure HERE
    indicates a defect in this module: the pipeline was supposed to make the
    condition impossible. Messages say so, because the remedy differs entirely
    from that of a malformed document.

Cheap checks always run
-----------------------
    Structural checks are linear in node count and catch catastrophic
    corruption, so they run unconditionally. Text re-verification and coverage
    walk the document text and are governed by configuration, which is where
    an operator trading assurance for throughput should express that choice.

Execution contract
------------------
    Synchronous and CPU-bound. The engine executes it in a worker thread.

Dependency contract
-------------------
    Imports only the standard library, ``ptfm.core.schemas`` and the knowledge
    ``exceptions``, ``config`` and ``models`` modules.
"""

from __future__ import annotations

from typing import Final, Sequence

from ptfm.core.knowledge.config import ValidationConfig
from ptfm.core.knowledge.exceptions import (
    CoverageError,
    NodeSpanIntegrityError,
    ProvenanceViolationError,
)
from ptfm.core.knowledge.models import KnowledgeUnit
from ptfm.core.schemas import RawDocument, derive_node_id

__all__ = ["ProvenanceValidator"]

#: Longest text excerpt included in a diagnostic.
_EXCERPT_CHARS: Final[int] = 120


class ProvenanceValidator:
    """
    Verifies a document's complete unit set before it leaves the module.

    Stateless: all per-document state is local to a call, so one instance
    serves all concurrent workers.

    Parameters
    ----------
    config:
        Validation configuration governing whether the document-walking checks
        run.
    """

    __slots__ = ("_config",)

    def __init__(self, config: ValidationConfig) -> None:
        self._config = config

    def validate(
        self,
        document: RawDocument,
        units: Sequence[KnowledgeUnit],
    ) -> None:
        """
        Verify every invariant that requires the complete unit set.

        Blocking and CPU-bound by contract. The engine executes this in a
        worker thread; it must never be invoked from a coroutine.

        Parameters
        ----------
        document:
            The source document the units claim to come from.
        units:
            Every unit produced for that document, in document order.

        Returns
        -------
        None

        Raises
        ------
        ProvenanceViolationError
            If identity, ordering, uniformity, or embedding comparability is
            violated.
        NodeSpanIntegrityError
            If a node's text is not the substring of the document at its span.
        CoverageError
            If document content exists that no node covers.
        """
        if not units:
            return

        self._verify_uniform_provenance(document, units)
        self._verify_identity(document, units)
        self._verify_ordering(document, units)
        self._verify_embedding_comparability(document, units)

        if self._config.verify_provenance:
            self._verify_text(document, units)

        if self._config.require_full_coverage:
            self._verify_coverage(document, units)

    # ------------------------------------------------------------------
    # Structural checks, always run
    # ------------------------------------------------------------------

    def _verify_uniform_provenance(
        self,
        document: RawDocument,
        units: Sequence[KnowledgeUnit],
    ) -> None:
        """
        Ensure every unit belongs to this document under one segmentation.

        A unit naming another document would attach this document's content to
        another's provenance. A unit carrying a different segmentation
        fingerprint would sit in a disjoint identifier space from its
        neighbours while appearing to belong beside them.

        Parameters
        ----------
        document:
            The source document.
        units:
            The document's units.

        Returns
        -------
        None

        Raises
        ------
        ProvenanceViolationError
            If a unit names another document or another segmentation.
        """
        fingerprint = units[0].node.segmentation_fingerprint

        for unit in units:
            node = unit.node

            if node.document_id != document.document_id:
                raise ProvenanceViolationError(
                    f"Node {node.node_id!r} names document "
                    f"{node.document_id!r} while belonging to the unit set of "
                    f"{document.document_id!r}. Its content would be attached "
                    f"to another document's provenance.",
                    violated_invariant="node.document_id == document.document_id",
                    document_id=document.document_id,
                    node_id=node.node_id,
                )

            if node.segmentation_fingerprint != fingerprint:
                raise ProvenanceViolationError(
                    f"Node {node.node_id!r} carries segmentation fingerprint "
                    f"{node.segmentation_fingerprint!r} while its siblings "
                    f"carry {fingerprint!r}. Identifiers derived under "
                    f"differing segmentation occupy disjoint spaces, so these "
                    f"nodes only appear to belong together.",
                    violated_invariant="uniform segmentation_fingerprint",
                    document_id=document.document_id,
                    node_id=node.node_id,
                )

            if node.source_text_hash != document.text_hash:
                raise ProvenanceViolationError(
                    f"Node {node.node_id!r} records source text hash "
                    f"{node.source_text_hash!r} while the document's is "
                    f"{document.text_hash!r}. The node's span was computed "
                    f"against different text from the document it names.",
                    violated_invariant="node.source_text_hash == document.text_hash",
                    document_id=document.document_id,
                    node_id=node.node_id,
                )

    def _verify_identity(
        self,
        document: RawDocument,
        units: Sequence[KnowledgeUnit],
    ) -> None:
        """
        Ensure every identifier is unique and reproducible from its inputs.

        Identity is a pure function of document, span, and segmentation
        fingerprint. An identifier that does not reproduce means one of those
        three changed after derivation, so the node would be unreachable by
        anything recomputing its identity, including a later idempotent run.

        Parameters
        ----------
        document:
            The source document.
        units:
            The document's units.

        Returns
        -------
        None

        Raises
        ------
        ProvenanceViolationError
            If an identifier repeats or does not reproduce.
        """
        seen: set[str] = set()

        for unit in units:
            node = unit.node

            if node.node_id in seen:
                raise ProvenanceViolationError(
                    f"Node identifier {node.node_id!r} appears twice in one "
                    f"document. A sink would treat the second as an update to "
                    f"the first, silently discarding content.",
                    violated_invariant="unique node_id within a document",
                    document_id=document.document_id,
                    node_id=node.node_id,
                )
            seen.add(node.node_id)

            expected = derive_node_id(
                node.document_id, node.span, node.segmentation_fingerprint
            )
            if expected != node.node_id:
                raise ProvenanceViolationError(
                    f"Node identifier {node.node_id!r} does not reproduce from "
                    f"its own document, span, and segmentation fingerprint, "
                    f"which yield {expected!r}. The node would be unreachable "
                    f"by anything recomputing its identity, including a later "
                    f"idempotent run.",
                    violated_invariant=(
                        "node_id == derive_node_id(document_id, span, fingerprint)"
                    ),
                    document_id=document.document_id,
                    node_id=node.node_id,
                )

    def _verify_ordering(
        self,
        document: RawDocument,
        units: Sequence[KnowledgeUnit],
    ) -> None:
        """
        Ensure nodes are ordered, disjoint, and contiguously numbered.

        Ordinals are how a consumer reconstructs reading order without holding
        the document, and overlapping spans would place the same text in two
        nodes, distorting the term frequency statistics retrieval depends on.

        Parameters
        ----------
        document:
            The source document.
        units:
            The document's units.

        Returns
        -------
        None

        Raises
        ------
        ProvenanceViolationError
            If ordinals are not contiguous from zero, or spans overlap or
            regress.
        """
        previous_end = 0

        for expected_ordinal, unit in enumerate(units):
            node = unit.node

            if node.ordinal != expected_ordinal:
                raise ProvenanceViolationError(
                    f"Node {node.node_id!r} carries ordinal {node.ordinal} at "
                    f"position {expected_ordinal}. Ordinals are how reading "
                    f"order is reconstructed without the document.",
                    violated_invariant="ordinals contiguous from zero",
                    document_id=document.document_id,
                    node_id=node.node_id,
                )

            if node.span.start < previous_end:
                raise ProvenanceViolationError(
                    f"Node {node.node_id!r} begins at {node.span.start}, "
                    f"before its predecessor ends at {previous_end}. "
                    f"Overlapping nodes place the same text twice, distorting "
                    f"the term frequency statistics retrieval depends on.",
                    violated_invariant="node spans ordered and disjoint",
                    document_id=document.document_id,
                    node_id=node.node_id,
                )

            previous_end = node.span.end

    def _verify_embedding_comparability(
        self,
        document: RawDocument,
        units: Sequence[KnowledgeUnit],
    ) -> None:
        """
        Ensure every embedding in the document is mutually comparable.

        The check no other stage can perform. Unit construction verifies that
        an embedding belongs to its node; a batch sees only itself. Only here
        does the whole document's set exist at once.

        A document mixing two models, dimensions, compositions, or
        normalization states produces similarity scores that are
        arithmetically valid and semantically meaningless. Nothing downstream
        reports an error: results are returned and ranked, merely wrongly.

        Parameters
        ----------
        document:
            The source document.
        units:
            The document's units.

        Returns
        -------
        None

        Raises
        ------
        ProvenanceViolationError
            If two embeddings in the document are not comparable.
        """
        reference = next(
            (unit.embedding for unit in units if unit.embedding is not None), None
        )
        if reference is None:
            return

        for unit in units:
            embedding = unit.embedding
            if embedding is None or embedding.is_comparable_to(reference):
                continue

            raise ProvenanceViolationError(
                f"Embedding for node {embedding.node_id!r} is not comparable "
                f"with its siblings: {embedding.model_identity!r} at "
                f"dimension {embedding.dimension} under composition "
                f"{embedding.composition_id!r} against "
                f"{reference.model_identity!r} at dimension "
                f"{reference.dimension} under {reference.composition_id!r}. "
                f"Similarity across them would be arithmetically valid and "
                f"semantically meaningless.",
                violated_invariant="uniform embedding identity within a document",
                document_id=document.document_id,
                node_id=embedding.node_id,
            )

    # ------------------------------------------------------------------
    # Document-walking checks, governed by configuration
    # ------------------------------------------------------------------

    def _verify_text(
        self,
        document: RawDocument,
        units: Sequence[KnowledgeUnit],
    ) -> None:
        """
        Ensure every node's text is the substring at its span.

        Assembly produces text by slicing at the span, so a mismatch here
        indicates that something altered a node afterwards. Enrichment is
        contractually additive, so reaching this is a defect in the module
        rather than in any plugin.

        Parameters
        ----------
        document:
            The source document.
        units:
            The document's units.

        Returns
        -------
        None

        Raises
        ------
        NodeSpanIntegrityError
            If a node's text and span disagree.
        """
        for unit in units:
            node = unit.node
            if node.verify_against(document.text):
                continue

            raise NodeSpanIntegrityError(
                f"Node {node.node_id!r} carries text that is not the substring "
                f"of its document at [{node.span.start}, {node.span.end}). "
                f"Assembly slices text from the span and enrichment is "
                f"additive, so a node was altered after construction.",
                expected_text=document.text[node.span.start : node.span.end],
                actual_text=node.text,
                span_start=node.span.start,
                span_end=node.span.end,
                document_id=document.document_id,
                node_id=node.node_id,
            )

    def _verify_coverage(
        self,
        document: RawDocument,
        units: Sequence[KnowledgeUnit],
    ) -> None:
        """
        Ensure the emitted nodes still account for the document.

        Planning verified coverage of the plan. This verifies coverage of what
        was actually emitted, which is a different statement: a node lost
        between planning and emission leaves text no node covers, arriving by
        a route the planning check cannot see.

        Parameters
        ----------
        document:
            The source document.
        units:
            The document's units.

        Returns
        -------
        None

        Raises
        ------
        CoverageError
            If non-whitespace text falls outside every node.
        """
        text = document.text
        cursor = 0

        for unit in units:
            span = unit.node.span
            gap = text[cursor : span.start]
            if gap and not gap.isspace():
                raise self._coverage_error(document, cursor, span.start, gap)
            cursor = max(cursor, span.end)

        trailing = text[cursor:]
        if trailing and not trailing.isspace():
            raise self._coverage_error(document, cursor, len(text), trailing)

    def _coverage_error(
        self,
        document: RawDocument,
        start: int,
        end: int,
        excerpt: str,
    ) -> CoverageError:
        """
        Build a coverage failure naming the uncovered region.

        Parameters
        ----------
        document:
            The source document.
        start:
            Start offset of the uncovered region.
        end:
            End offset of the uncovered region.
        excerpt:
            The uncovered text.

        Returns
        -------
        CoverageError
            The failure to raise.
        """
        return CoverageError(
            f"Emitted nodes do not account for document content at "
            f"[{start}, {end}): {excerpt[:_EXCERPT_CHARS]!r}. The plan covered "
            f"this text, so a node was lost between planning and emission.",
            document_id=document.document_id,
            uncovered_start=start,
            uncovered_end=end,
            uncovered_chars=len(excerpt.strip()),
        )

    def __repr__(self) -> str:
        """
        Return a representation naming the configured checks.

        Returns
        -------
        str
            Diagnostic representation.
        """
        return (
            f"ProvenanceValidator(verify_provenance="
            f"{self._config.verify_provenance}, require_full_coverage="
            f"{self._config.require_full_coverage})"
        )
