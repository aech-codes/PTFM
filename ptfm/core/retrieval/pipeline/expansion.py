"""
PTFM Context Expansion

File:
    ptfm/core/retrieval/pipeline/expansion.py

Description:
    Adds the neighbouring nodes of a retrieved result as accompanying context,
    under bounds that prevent context from ever being mistaken for an
    independent finding.

Why this exists at all
----------------------
    Knowledge Processing produces nodes with disjoint spans and no overlap,
    because overlapping nodes would duplicate text and duplicated text enters
    verification as two evidence items rather than one, inflating apparent
    corroboration in the exact quantity this system exists to calibrate.

    Removing overlap left one genuine gap: a claim whose support straddles two
    adjacent nodes may be matched by neither alone. Expansion closes that gap by
    REFERENCE instead of duplication, and it is the designated remedy for it.

    The remedy must not reintroduce the disease. Everything below is in service
    of that.

An anchor is a finding; context is not
--------------------------------------
    An anchor is a node retrieval found. Context is a node included because it
    adjoins one.

    Corroboration is counted downstream by distinct anchors. So a context node
    admitted as an anchor would make one finding appear as two, which is the
    same harm overlapping nodes would have caused, arriving by a different
    route. Three invariants hold, and each closes one way that could happen:

    A context node is never an anchor. It is attached to the anchor it
    accompanies and carries no standing of its own.

    A node that IS an anchor is never also context. It was found
    independently; listing it as another anchor's companion would either
    duplicate it or demote a finding to a companion.

    A context node appears once in the whole result. A node adjacent to two
    anchors is assigned to exactly one of them, deterministically.

Neighbours are ordinal, not traversed
-------------------------------------
    A node's neighbour is the node at the adjacent ordinal within the same
    document. Knowledge Processing guarantees ordinals contiguous from zero and
    spans disjoint and ascending, so that is exact rather than heuristic.

    ``PRECEDES`` relations state the same adjacency and remain the canonical
    record of it for consumers that reason over document structure. Retrieval
    does not read them, because expansion runs on the hot path of every query
    and a graph round trip per result would be the dominant cost of a capability
    that is arithmetic.

Dependency contract
-------------------
    Imports only the standard library, the retrieval ``exceptions``, ``config``,
    ``models`` and ``interfaces`` modules, and the knowledge schemas.
"""

from __future__ import annotations

from typing import Final, Mapping, Sequence

from ptfm.core.retrieval.config import ExpansionConfig
from ptfm.core.retrieval.exceptions import AnchorViolationError, ExpansionError
from ptfm.core.retrieval.interfaces import NodeStore, StructuredLogger
from ptfm.core.retrieval.models import SegmentationPartition
from ptfm.core.schemas import KnowledgeNode

__all__ = ["ORDINAL_EXPANDER_NAME", "OrdinalContextExpander"]

#: Selector for this expander.
ORDINAL_EXPANDER_NAME: Final[str] = "ordinal"


class OrdinalContextExpander:
    """
    Adds a result's ordinal neighbours as accompanying context.

    Satisfies the ``ContextExpander`` protocol.

    Deterministic. The context returned for a given set of anchors is a
    function of those anchors and the corpus, so an audit repeating a query
    against an unchanged corpus sees the same evidence groups.

    Bounds, all from configuration
    ------------------------------
    ``neighbour_radius``
        How many ordinals either side of an anchor to include. One by default:
        enough to close the straddling-claim gap, and small enough that a result
        set does not grow into a re-creation of the index inflation that removing
        overlap prevented.

    ``same_document_only``
        Whether a neighbour must come from the anchor's own document. Enabled by
        default, because the node before a document's first is not adjacent to it
        in any meaningful sense; it belongs to a different document that happened
        to be indexed nearby.

    ``max_context_nodes``
        A ceiling on context across the whole result set, so that a query
        matching many anchors cannot multiply its own cost by the radius.

    Parameters
    ----------
    store:
        The node store, consulted by ordinal.
    config:
        Expansion configuration.
    logger:
        Structured logger.
    """

    __slots__ = ("_store", "_config", "_logger")

    def __init__(
        self,
        store: NodeStore,
        config: ExpansionConfig,
        logger: StructuredLogger,
    ) -> None:
        self._store = store
        self._config = config
        self._logger = logger

    @property
    def name(self) -> str:
        """
        Stable identifier of this expander.

        Returns
        -------
        str
            Implementation identifier.
        """
        return ORDINAL_EXPANDER_NAME

    async def expand(
        self,
        anchors: Sequence[KnowledgeNode],
        partition: SegmentationPartition,
    ) -> Mapping[str, tuple[KnowledgeNode, ...]]:
        """
        Find the context accompanying each anchor.

        Parameters
        ----------
        anchors:
            The independently retrieved nodes, in fused order. Order is
            significant: where two anchors share a neighbour, the earlier anchor
            takes it, so the higher-ranked result carries the context.
        partition:
            The segmentation partition to resolve within.

        Returns
        -------
        Mapping[str, tuple[KnowledgeNode, ...]]
            Context nodes keyed by the anchor they accompany, each anchor's
            context in reading order. Anchors with no context are absent rather
            than mapped to an empty tuple, so a caller iterating the mapping
            visits only anchors that gained something.

        Raises
        ------
        ExpansionError
            If context could not be resolved.
        AnchorViolationError
            If the resolved context would breach an anchor invariant, which
            indicates a defect in this module rather than in the corpus.
        """
        if not anchors or not self._config.enabled:
            return {}

        radius = self._config.neighbour_radius
        if radius < 1 or self._config.max_context_nodes < 1:
            return {}

        anchor_ids = {anchor.node_id for anchor in anchors}
        wanted = self._wanted_ordinals(anchors, radius)

        try:
            resolved = await self._resolve(wanted, partition)
        except ExpansionError:
            raise
        except Exception as exc:
            raise ExpansionError(
                f"Node store {self._store.name!r} could not resolve context "
                f"for {len(anchors)} anchors: {type(exc).__name__}: {exc}",
                context={"store_name": self._store.name},
            ) from exc

        context = self._assign(anchors, radius, resolved, anchor_ids)
        self._verify(context, anchor_ids)

        added = sum(len(nodes) for nodes in context.values())
        if added:
            self._logger.debug(
                "retrieval.expansion.applied",
                anchors=len(anchors),
                context_nodes=added,
                radius=radius,
            )

        return context

    # ------------------------------------------------------------------
    # Resolution
    # ------------------------------------------------------------------

    def _wanted_ordinals(
        self,
        anchors: Sequence[KnowledgeNode],
        radius: int,
    ) -> Mapping[str, tuple[int, ...]]:
        """
        Determine which ordinals to fetch, grouped by document.

        Grouped and deduplicated before any call, so that several anchors in one
        document cost one request rather than one each, and a neighbour two
        anchors share is fetched once.

        Parameters
        ----------
        anchors:
            The anchors to expand.
        radius:
            Ordinals either side of each anchor.

        Returns
        -------
        Mapping[str, tuple[int, ...]]
            Ascending ordinals by document identifier.
        """
        wanted: dict[str, set[int]] = {}

        for anchor in anchors:
            ordinals = wanted.setdefault(anchor.document_id, set())
            for offset in range(-radius, radius + 1):
                if offset == 0:
                    continue
                neighbour = anchor.ordinal + offset
                # A negative ordinal precedes a document's first node and
                # cannot exist; the store would simply omit it, but asking is
                # pointless.
                if neighbour >= 0:
                    ordinals.add(neighbour)

        return {
            document_id: tuple(sorted(ordinals))
            for document_id, ordinals in wanted.items()
        }

    async def _resolve(
        self,
        wanted: Mapping[str, tuple[int, ...]],
        partition: SegmentationPartition,
    ) -> Mapping[str, Mapping[int, KnowledgeNode]]:
        """
        Fetch the requested ordinals from the node store.

        Parameters
        ----------
        wanted:
            Ordinals by document identifier.
        partition:
            The segmentation partition to resolve within.

        Returns
        -------
        Mapping[str, Mapping[int, KnowledgeNode]]
            Resolved nodes by document, then by ordinal. An ordinal past a
            document's last node is absent, which is ordinary at a document's
            edges and not a fault.

        Raises
        ------
        ExpansionError
            If the store could not be reached.
        """
        resolved: dict[str, Mapping[int, KnowledgeNode]] = {}

        # Sequential rather than concurrent. Expansion is a bounded follow-up to
        # a query that has already run, and issuing a burst of requests here
        # would contend with the arms of queries still in flight for the
        # connection pool the whole engine shares.
        for document_id, ordinals in sorted(wanted.items()):
            resolved[document_id] = await self._store.get_by_ordinal(
                document_id, ordinals, partition
            )

        return resolved

    # ------------------------------------------------------------------
    # Assignment
    # ------------------------------------------------------------------

    def _assign(
        self,
        anchors: Sequence[KnowledgeNode],
        radius: int,
        resolved: Mapping[str, Mapping[int, KnowledgeNode]],
        anchor_ids: frozenset[str] | set[str],
    ) -> dict[str, tuple[KnowledgeNode, ...]]:
        """
        Attach each resolved neighbour to exactly one anchor.

        Anchors are visited in fused order and the first to claim a neighbour
        keeps it. That makes assignment deterministic and puts shared context on
        the higher-ranked result, where a consumer reading top-down encounters
        it soonest.

        Two exclusions are applied here, and they are the invariants this module
        exists to hold:

        A node that is itself an anchor is never taken as context. It was found
        independently, and listing it as a companion would either duplicate it
        or demote a finding.

        A node already claimed by an earlier anchor is not taken again. Emitting
        it twice would let one passage be counted twice.

        Parameters
        ----------
        anchors:
            The anchors, in fused order.
        radius:
            Ordinals either side of each anchor.
        resolved:
            Nodes fetched from the store.
        anchor_ids:
            Identifiers of every anchor.

        Returns
        -------
        dict[str, tuple[KnowledgeNode, ...]]
            Context by anchor identifier, each in reading order.
        """
        context: dict[str, tuple[KnowledgeNode, ...]] = {}
        claimed: set[str] = set()
        budget = self._config.max_context_nodes

        for anchor in anchors:
            if len(claimed) >= budget:
                break

            by_ordinal = resolved.get(anchor.document_id, {})
            taken: list[KnowledgeNode] = []

            for offset in range(-radius, radius + 1):
                if offset == 0:
                    continue
                if len(claimed) >= budget:
                    break

                neighbour = by_ordinal.get(anchor.ordinal + offset)
                if neighbour is None:
                    continue
                if neighbour.node_id in anchor_ids:
                    continue
                if neighbour.node_id in claimed:
                    continue
                if (
                    self._config.same_document_only
                    and neighbour.document_id != anchor.document_id
                ):
                    continue

                claimed.add(neighbour.node_id)
                taken.append(neighbour)

            if taken:
                context[anchor.node_id] = tuple(
                    sorted(taken, key=lambda node: node.ordinal)
                )

        return context

    def _verify(
        self,
        context: Mapping[str, Sequence[KnowledgeNode]],
        anchor_ids: frozenset[str] | set[str],
    ) -> None:
        """
        Verify the anchor invariants after assignment.

        Defence in depth. Assignment already excludes anchors and repeats, so
        this should never fire; it exists because the consequence of a breach is
        inflated corroboration, and inflated corroboration produces confidence
        that errs toward overconfidence, which is the worst outcome available to
        this system. A check that never fires is cheap; the failure it guards
        against is not.

        Parameters
        ----------
        context:
            The assigned context.
        anchor_ids:
            Identifiers of every anchor.

        Returns
        -------
        None

        Raises
        ------
        AnchorViolationError
            If a context node is an anchor, or appears more than once.
        """
        seen: set[str] = set()

        for anchor_id, nodes in context.items():
            for node in nodes:
                if node.node_id in anchor_ids:
                    raise AnchorViolationError(
                        f"Node {node.node_id!r} was assigned as context for "
                        f"anchor {anchor_id!r} while being an anchor itself. "
                        f"Corroboration counts distinct anchors, so this would "
                        f"make one finding appear as two.",
                        node_id=node.node_id,
                        anchor_node_id=anchor_id,
                    )
                if node.node_id in seen:
                    raise AnchorViolationError(
                        f"Node {node.node_id!r} was assigned as context more "
                        f"than once. One passage would be counted twice.",
                        node_id=node.node_id,
                        anchor_node_id=anchor_id,
                    )
                seen.add(node.node_id)

    def __repr__(self) -> str:
        """
        Return a representation naming the configured bounds.

        Returns
        -------
        str
            Diagnostic representation.
        """
        return (
            f"OrdinalContextExpander(radius="
            f"{self._config.neighbour_radius}, "
            f"max_context={self._config.max_context_nodes}, "
            f"enabled={self._config.enabled})"
        )
