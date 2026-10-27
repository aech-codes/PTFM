"""
PTFM Credibility Annotation and Evidence Assembly

File:
    ptfm/core/retrieval/pipeline/assembly.py

Description:
    Attaches source credibility, then builds the evidence items and retrieved
    documents that leave this module.

    The last stage of retrieval. Everything downstream reasons about what is
    constructed here, and can reason about nothing that is not.

Credibility is recorded, never inferred
---------------------------------------
    A source whose credibility a plugin cannot assess carries ``None``, and
    ``None`` travels all the way to the evidence item.

    Nothing is substituted, and the reason is arithmetic rather than
    fastidiousness. Retrieval reliability is

        r(E) = (1/k) * sum_i  r_i * s_i

    and hallucination risk is

        H = S(q) * (1 - r(E))

    so a credibility this module invented would enter a statistical guarantee
    in a form indistinguishable from one a plugin measured. On a three-item
    result set with one unscoreable source, the choice between substituting
    zero, one half, and one moves H across a range wider than half its own
    value.

    Discarding the item instead is not the neutral alternative it appears to
    be. It keeps only the sources that could be scored, so reliability rises
    and risk falls: ignorance is recorded as reliability, which is the wrong
    direction for a system whose purpose is calibrated doubt.

    What an unknown credibility means is therefore decided by the layer that
    owns the guarantee, and this module makes the unknown visible so that the
    decision can be made at all.

An anchor is a finding; context is not
--------------------------------------
    Every item records how it arrived.

    An anchor is a node retrieval found. It anchors itself, and its discovery
    records name each arm that found it, with that arm's rank, score, and
    scale.

    A context node accompanies the anchor it adjoins. It anchors to that node,
    never to itself, and its single discovery record names the expansion and
    the anchor it came from.

    Corroboration downstream counts items where ``node_id == anchor_node_id``.
    That rule is correct only if the distinction is exact here, so it is
    verified after assembly rather than assumed.

The relevance signal is ordinal
-------------------------------
    ``retrieval_score`` is required to lie in ``[0, 1]``, one module away from
    the layer that produces probabilities. It is derived from fused POSITION
    rather than from the fused value, as the reciprocal ``1 / (1 + rank)``.

    Position is the only thing about a fused result that survives every arm's
    scale being incomparable, so a rank-derived number cannot be mistaken for a
    measurement of anything. The fused value itself, and every arm's raw score
    with the scale it lives on, remain on the item's signals for any consumer
    that wants them.

Dependency contract
-------------------
    Imports only the standard library, the retrieval ``exceptions``,
    ``config``, ``models`` and ``interfaces`` modules, and the shared schemas.
"""

from __future__ import annotations

from typing import Final, Mapping, Optional, Sequence

from ptfm.core.retrieval.config import CredibilityConfig
from ptfm.core.retrieval.exceptions import (
    AnchorViolationError,
    CorroborationIntegrityError,
    CredibilityError,
    CredibilityUnavailableError,
    EvidenceAssemblyError,
)
from ptfm.core.retrieval.interfaces import CredibilityProvider, StructuredLogger
from ptfm.core.retrieval.pipeline.hydration import HydratedResult
from ptfm.core.schemas import (
    DiscoveryMethod,
    DiscoveryRecord,
    EvidenceItem,
    EvidenceType,
    KnowledgeNode,
    RetrievalMethod,
    RetrievedDocument,
)

__all__ = ["CredibilityAnnotator", "EvidenceAssembler", "AssembledEvidence"]

#: Separator between the parts of an evidence identifier.
_EVIDENCE_SEPARATOR: Final[str] = ":"

#: Prefix distinguishing an evidence identifier from the node identifier it is
#: built from, so that the two cannot be confused in a log or a citation.
_EVIDENCE_PREFIX: Final[str] = "ev"


class AssembledEvidence:
    """
    The evidence produced for one query, with the counts a caller needs.

    Returns documents and evidence items together because they describe the
    same results in two shapes the two consumers expect, and building them in
    one pass guarantees they cannot disagree about what was retrieved.

    Parameters
    ----------
    documents:
        Retrieved documents, in fused order with context following its anchor.
    evidence:
        Evidence items, in the same order.
    anchor_count:
        Items found independently. The unit of corroboration, reported so that
        a consumer need not recompute it and risk recomputing it differently.
    context_count:
        Items included as context. Never corroboration.
    credibility_unavailable:
        Sources whose credibility could not be obtained, in sorted order.
    """

    __slots__ = (
        "documents",
        "evidence",
        "anchor_count",
        "context_count",
        "credibility_unavailable",
    )

    def __init__(
        self,
        documents: tuple[RetrievedDocument, ...],
        evidence: tuple[EvidenceItem, ...],
        anchor_count: int,
        context_count: int,
        credibility_unavailable: tuple[str, ...],
    ) -> None:
        self.documents = documents
        self.evidence = evidence
        self.anchor_count = anchor_count
        self.context_count = context_count
        self.credibility_unavailable = credibility_unavailable

    def __len__(self) -> int:
        """
        Total number of evidence items.

        Returns
        -------
        int
            Anchors plus context.
        """
        return len(self.evidence)

    def __repr__(self) -> str:
        """
        Return a representation naming the anchor and context counts.

        Returns
        -------
        str
            Diagnostic representation.
        """
        return (
            f"AssembledEvidence(anchors={self.anchor_count}, "
            f"context={self.context_count}, "
            f"credibility_unavailable={len(self.credibility_unavailable)})"
        )


class CredibilityAnnotator:
    """
    Obtains source credibility from a domain plugin.

    The domain boundary of this module, alongside query preprocessing. What
    makes a source credible is domain knowledge: a regulator's filing, a
    peer-reviewed paper, an internal wiki page, and a preprint are judged by
    criteria retrieval knows nothing about.

    Concerns a SOURCE, not a claim. Nothing here judges whether anything is
    true.

    Parameters
    ----------
    provider:
        The credibility provider, ordinarily backed by a domain plugin.
        ``None`` where a deployment configures none, in which case every source
        carries an unknown credibility rather than an assumed one.
    config:
        Credibility configuration.
    logger:
        Structured logger.
    """

    __slots__ = ("_provider", "_config", "_logger")

    def __init__(
        self,
        provider: Optional[CredibilityProvider],
        config: CredibilityConfig,
        logger: StructuredLogger,
    ) -> None:
        self._provider = provider
        self._config = config
        self._logger = logger

    async def credibility_for(
        self,
        source_uris: Sequence[str],
    ) -> Mapping[str, Optional[float]]:
        """
        Report the credibility of each source.

        Parameters
        ----------
        source_uris:
            Distinct sources to assess.

        Returns
        -------
        Mapping[str, Optional[float]]
            Credibility by source URI, ``None`` where none was available.

            Every requested source appears in the result. Absence is
            represented by a ``None`` value rather than by a missing key, so a
            caller cannot silently skip a source it failed to look up.

        Notes
        -----
        A provider failing is degradation rather than failure: every source
        becomes unknown, the query still answers, and the unknowns are visible.
        Refusing to answer because a credibility service was unreachable would
        trade a recorded uncertainty for no answer at all.
        """
        unknown: dict[str, Optional[float]] = {uri: None for uri in source_uris}

        if self._provider is None or not self._config.enabled or not source_uris:
            return unknown

        try:
            reported = await self._provider.credibility_for(tuple(source_uris))
        except (CredibilityError, CredibilityUnavailableError) as error:
            self._logger.warning(
                "retrieval.credibility.degraded",
                provider=self._provider.name,
                error_code=error.error_code,
                sources=len(source_uris),
            )
            return unknown
        except Exception as exc:  # noqa: BLE001 - a plugin may raise anything
            self._logger.warning(
                "retrieval.credibility.degraded",
                provider=self._provider.name,
                error_type=type(exc).__name__,
                message=str(exc),
                sources=len(source_uris),
            )
            return unknown

        resolved: dict[str, Optional[float]] = {}
        for uri in source_uris:
            resolved[uri] = _validated(reported.get(uri), uri, self._logger)

        missing = [uri for uri, value in resolved.items() if value is None]
        if missing:
            # The provider answered but could not assess these. Recorded
            # because the absence changes what a consumer can compute, and
            # nothing else in the result would say so.
            self._logger.debug(
                "retrieval.credibility.unavailable",
                provider=self._provider.name,
                unavailable=len(missing),
                sources=len(source_uris),
            )

        return resolved

    def __repr__(self) -> str:
        """
        Return a representation naming the provider.

        Returns
        -------
        str
            Diagnostic representation.
        """
        name = self._provider.name if self._provider is not None else "none"
        return f"CredibilityAnnotator(provider={name!r})"


class EvidenceAssembler:
    """
    Builds the evidence items and documents that leave the module.

    Stateless and synchronous. Every input has already been resolved, so
    assembly is a pure transformation and declaring it asynchronous would
    suggest otherwise.

    Parameters
    ----------
    logger:
        Structured logger.
    """

    __slots__ = ("_logger",)

    def __init__(self, logger: StructuredLogger) -> None:
        self._logger = logger

    def assemble(
        self,
        anchors: Sequence[HydratedResult],
        context: Mapping[str, Sequence[KnowledgeNode]],
        credibility: Mapping[str, Optional[float]],
    ) -> AssembledEvidence:
        """
        Build evidence from anchors and their context.

        Emitted in fused order, each anchor immediately followed by its own
        context in reading order. A consumer reading top-down therefore meets a
        finding and then the passages that surround it, which is the order in
        which the evidence makes sense.

        Parameters
        ----------
        anchors:
            Hydrated results, in fused order. Each was found by retrieval and
            is an anchor.
        context:
            Context nodes by the anchor identifier they accompany, as expansion
            assigned them.
        credibility:
            Credibility by source URI, ``None`` where unavailable.

        Returns
        -------
        AssembledEvidence
            Documents, evidence items, and the counts describing them.

        Raises
        ------
        EvidenceAssemblyError
            If context names an anchor that was not retrieved.
        CorroborationIntegrityError
            If the assembled set would misrepresent how much corroboration
            exists.
        AnchorViolationError
            If a context item claims to be an anchor.
        """
        anchor_ids = {result.node_id for result in anchors}
        self._verify_context_anchors(context, anchor_ids)

        documents: list[RetrievedDocument] = []
        evidence: list[EvidenceItem] = []
        context_count = 0

        for result in anchors:
            node = result.node
            score = _ordinal_relevance(result.candidate.fused_rank)
            source_credibility = credibility.get(node.source.source_uri)

            documents.append(
                _document(
                    node=node,
                    anchor=result,
                    retrieval_score=score,
                    credibility=source_credibility,
                    is_anchor=True,
                )
            )
            evidence.append(
                _evidence(
                    node=node,
                    anchor_node_id=node.node_id,
                    discovery=_anchor_discovery(result),
                    retrieval_score=score,
                    credibility=source_credibility,
                )
            )

            for position, companion in enumerate(context.get(node.node_id, ())):
                # Context inherits its anchor's relevance position. It was not
                # ranked, so it has no position of its own, and inventing one
                # would imply a measurement nobody made.
                companion_credibility = credibility.get(
                    companion.source.source_uri
                )
                documents.append(
                    _document(
                        node=companion,
                        anchor=result,
                        retrieval_score=score,
                        credibility=companion_credibility,
                        is_anchor=False,
                    )
                )
                evidence.append(
                    _evidence(
                        node=companion,
                        anchor_node_id=node.node_id,
                        discovery=(
                            DiscoveryRecord(
                                method=DiscoveryMethod.EXPANSION,
                                rank=position,
                                raw_score=None,
                                scale=None,
                                via_node_id=node.node_id,
                            ),
                        ),
                        retrieval_score=score,
                        credibility=companion_credibility,
                    )
                )
                context_count += 1

        assembled = AssembledEvidence(
            documents=tuple(documents),
            evidence=tuple(evidence),
            anchor_count=len(anchors),
            context_count=context_count,
            credibility_unavailable=tuple(
                sorted(uri for uri, value in credibility.items() if value is None)
            ),
        )

        self._verify(assembled, anchor_ids)
        return assembled

    # ------------------------------------------------------------------
    # Verification
    # ------------------------------------------------------------------

    def _verify_context_anchors(
        self,
        context: Mapping[str, Sequence[KnowledgeNode]],
        anchor_ids: set[str],
    ) -> None:
        """
        Ensure every context group names an anchor that exists.

        Context keyed to an absent anchor would be dropped silently, losing the
        very passages expansion was run to obtain, and nothing would report the
        loss.

        Parameters
        ----------
        context:
            Context by anchor identifier.
        anchor_ids:
            Identifiers of the retrieved anchors.

        Returns
        -------
        None

        Raises
        ------
        EvidenceAssemblyError
            If context names an unknown anchor.
        """
        orphaned = sorted(set(context) - anchor_ids)
        if not orphaned:
            return

        raise EvidenceAssemblyError(
            f"Context was supplied for {len(orphaned)} anchors that are not in "
            f"the result set: {', '.join(orphaned)}. Those passages would be "
            f"discarded without anything reporting their loss.",
            context={"orphaned_anchors": orphaned},
        )

    def _verify(
        self,
        assembled: AssembledEvidence,
        anchor_ids: set[str],
    ) -> None:
        """
        Verify that the assembled set represents corroboration truthfully.

        Corroboration is counted downstream as the number of items whose node
        is its own anchor. Three ways that count could be wrong are checked
        here, and each would inflate it:

        An item appearing twice would be counted twice.

        A context item claiming to be its own anchor would turn an
        accompanying passage into an independent finding.

        An item anchored to something absent from the set would leave a dangling
        citation that no consumer could resolve.

        The checks should never fire, since assembly constructs the anchors and
        expansion excluded them from context. They exist because the failure
        they guard against produces confidence erring toward overconfidence,
        which is the worst outcome available to this system.

        Parameters
        ----------
        assembled:
            The assembled evidence.
        anchor_ids:
            Identifiers of the retrieved anchors.

        Returns
        -------
        None

        Raises
        ------
        CorroborationIntegrityError
            If an item repeats or its anchor is absent.
        AnchorViolationError
            If a context item claims to be an anchor.
        """
        seen: set[str] = set()

        for item in assembled.evidence:
            node_id = item.node_id
            if node_id is None:
                raise CorroborationIntegrityError(
                    f"Evidence item {item.evidence_id!r} names no node, so it "
                    f"cannot be traced to a passage.",
                    violated_invariant="every evidence item names a node",
                )

            if node_id in seen:
                raise CorroborationIntegrityError(
                    f"Node {node_id!r} appears in the evidence set more than "
                    f"once. One passage would be counted twice.",
                    violated_invariant="unique node_id within an evidence set",
                    node_id=node_id,
                )
            seen.add(node_id)

            if item.anchor_node_id not in anchor_ids:
                raise CorroborationIntegrityError(
                    f"Evidence item {item.evidence_id!r} is anchored to "
                    f"{item.anchor_node_id!r}, which is not in the result set. "
                    f"The citation could not be resolved.",
                    violated_invariant="every anchor is present in the set",
                    node_id=node_id,
                )

            if node_id != item.anchor_node_id and node_id in anchor_ids:
                raise AnchorViolationError(
                    f"Node {node_id!r} is an anchor but was assembled as "
                    f"context for {item.anchor_node_id!r}. Corroboration counts "
                    f"distinct anchors, so this would make one finding appear "
                    f"as two.",
                    node_id=node_id,
                    anchor_node_id=item.anchor_node_id,
                )

        counted = sum(1 for item in assembled.evidence if item.is_anchor)
        if counted != assembled.anchor_count:
            raise CorroborationIntegrityError(
                f"The evidence set contains {counted} anchors while "
                f"{assembled.anchor_count} were retrieved. The corroboration "
                f"count a consumer computes would differ from what retrieval "
                f"found.",
                violated_invariant="reported anchor count matches the set",
            )

    def __repr__(self) -> str:
        """
        Return a representation of this assembler.

        Returns
        -------
        str
            Diagnostic representation.
        """
        return "EvidenceAssembler()"


# ============================================================================
# HELPERS
# ============================================================================


def _ordinal_relevance(fused_rank: int) -> float:
    """
    Project a fused position onto the unit interval.

    The reciprocal ``1 / (1 + rank)``: first place scores one, second one half,
    third one third.

    Derived from POSITION rather than from the fused value. Position is the
    only property of a fused result that survives the arms' scales being
    mutually incomparable, so a number derived from it cannot be mistaken for a
    measurement of similarity, relevance, or probability. The fused value and
    every arm's raw score remain on the item's signals for a consumer that
    wants them.

    Parameters
    ----------
    fused_rank:
        Zero-based position in the fused ordering.

    Returns
    -------
    float
        A value in ``(0, 1]``, strictly decreasing in rank.
    """
    return 1.0 / (1.0 + fused_rank)


def _validated(
    value: Optional[float],
    source_uri: str,
    logger: StructuredLogger,
) -> Optional[float]:
    """
    Accept a credibility only if it is usable.

    A value outside the unit interval is treated as unavailable rather than
    clamped. Clamping would convert a plugin's error into a plausible number
    and feed it to the reliability calculation, which is the substitution this
    module exists to avoid.

    Parameters
    ----------
    value:
        The reported credibility, if any.
    source_uri:
        The source it describes, for diagnosis.
    logger:
        Structured logger.

    Returns
    -------
    Optional[float]
        The credibility, or ``None`` where it was absent or unusable.
    """
    if value is None:
        return None

    if not isinstance(value, (int, float)) or value != value:
        logger.warning(
            "retrieval.credibility.unusable",
            source_uri=source_uri,
            reason="not a number",
        )
        return None

    if not 0.0 <= float(value) <= 1.0:
        logger.warning(
            "retrieval.credibility.unusable",
            source_uri=source_uri,
            reason="outside the unit interval",
            reported=float(value),
        )
        return None

    return float(value)


def _anchor_discovery(result: HydratedResult) -> tuple[DiscoveryRecord, ...]:
    """
    Render every arm's observation of an anchor as discovery records.

    One record per arm that found the node, each carrying that arm's rank, raw
    score, and the scale the score lives on.

    Several records describe ONE finding observed several ways, not several
    findings. Corroboration counts anchors, never records, which is why an
    anchor found by three arms still contributes one.

    Parameters
    ----------
    result:
        The hydrated anchor.

    Returns
    -------
    tuple[DiscoveryRecord, ...]
        Discovery records in the order fusion fixed.
    """
    return tuple(
        DiscoveryRecord(
            method=signal.method,
            rank=signal.rank,
            raw_score=signal.raw_score,
            scale=signal.scale,
            via_node_id=None,
        )
        for signal in result.candidate.signals
    )


def _evidence_id(node_id: str, anchor_node_id: str) -> str:
    """
    Derive a stable identifier for an evidence item.

    Deterministic, so that repeating a query against an unchanged corpus
    produces the same identifiers and two runs can be compared.

    Includes the anchor as well as the node, because the same passage may
    legitimately appear as context for a different anchor in a different query,
    and the pair is what identifies its role here.

    Parameters
    ----------
    node_id:
        The node this item cites.
    anchor_node_id:
        The anchor it belongs to.

    Returns
    -------
    str
        The evidence identifier.
    """
    return _EVIDENCE_SEPARATOR.join((_EVIDENCE_PREFIX, anchor_node_id, node_id))


def _document(
    *,
    node: KnowledgeNode,
    anchor: HydratedResult,
    retrieval_score: float,
    credibility: Optional[float],
    is_anchor: bool,
) -> RetrievedDocument:
    """
    Build the document view of one result.

    Parameters
    ----------
    node:
        The node this document describes.
    anchor:
        The anchor it belongs to, whose signals it reports when it is itself
        the anchor.
    retrieval_score:
        The ordinal relevance signal.
    credibility:
        Source credibility, or ``None`` where unavailable.
    is_anchor:
        Whether this node was found independently.

    Returns
    -------
    RetrievedDocument
        The document, carrying the span and blocks that make it citable.
    """
    return RetrievedDocument(
        document_id=node.document_id,
        node_id=node.node_id,
        span=node.span,
        block_ids=node.block_ids,
        title=node.heading_path[-1] if node.heading_path else None,
        content=node.text,
        source_uri=node.source.source_uri,
        retrieval_method=_retrieval_method(anchor, is_anchor),
        signals=anchor.candidate.signals if is_anchor else (),
        retrieval_score=retrieval_score,
        credibility_score=credibility,
        evidence_type=EvidenceType.DOCUMENT,
        metadata=dict(node.metadata),
    )


def _evidence(
    *,
    node: KnowledgeNode,
    anchor_node_id: str,
    discovery: tuple[DiscoveryRecord, ...],
    retrieval_score: float,
    credibility: Optional[float],
) -> EvidenceItem:
    """
    Build the evidence view of one result.

    Parameters
    ----------
    node:
        The node this item cites.
    anchor_node_id:
        The anchor it belongs to. Equal to the node's own identifier when the
        node is an anchor.
    discovery:
        How this item arrived.
    retrieval_score:
        The ordinal relevance signal.
    credibility:
        Source credibility, or ``None`` where unavailable.

    Returns
    -------
    EvidenceItem
        The evidence item, traceable to an exact span of an identified source.
    """
    return EvidenceItem(
        evidence_id=_evidence_id(node.node_id, anchor_node_id),
        document_id=node.document_id,
        node_id=node.node_id,
        anchor_node_id=anchor_node_id,
        discovery=discovery,
        span=node.span,
        block_ids=node.block_ids,
        content=node.text,
        source_uri=node.source.source_uri,
        retrieval_score=retrieval_score,
        credibility_score=credibility,
    )


def _retrieval_method(anchor: HydratedResult, is_anchor: bool) -> RetrievalMethod:
    """
    Report which retrieval method a document is attributed to.

    A legacy single-valued field, retained because consumers depend on it. It
    reports HYBRID whenever more than one arm contributed, because naming one
    of several would misattribute the finding, and the complete per-arm truth
    is on the signals regardless.

    Parameters
    ----------
    anchor:
        The anchor whose arms are consulted.
    is_anchor:
        Whether the document is the anchor itself.

    Returns
    -------
    RetrievalMethod
        The attributed method.
    """
    if not is_anchor:
        return RetrievalMethod.HYBRID

    methods = {signal.method for signal in anchor.candidate.signals}
    if len(methods) != 1:
        return RetrievalMethod.HYBRID

    only = next(iter(methods))
    mapping = {
        DiscoveryMethod.DENSE: RetrievalMethod.DENSE,
        DiscoveryMethod.SPARSE: RetrievalMethod.SPARSE,
        DiscoveryMethod.GRAPH: RetrievalMethod.GRAPH,
    }
    return mapping.get(only, RetrievalMethod.HYBRID)
