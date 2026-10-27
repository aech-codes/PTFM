"""
PTFM Result Hydration and Supersession

File:
    ptfm/core/retrieval/pipeline/hydration.py

Description:
    Turns ranked identifiers into authoritative nodes, and removes those whose
    documents no longer represent their sources.

Where evidence becomes evidence
-------------------------------
    Up to this point a result is an identifier and a set of signals. Nothing
    retrieved so far carries text, a span, or a source.

    Hydration is where that changes, and it is the single point at which
    content enters a result. It reads from the node store and from nowhere
    else. An index may hold its own copy of a node's text for its own
    purposes, and that copy is never consulted: if it were, a stale index
    could change what a citation resolves to, and no downstream check would
    notice.

    Every guarantee the module makes about provenance is established here or
    not at all.

Supersession filters, and never deletes
---------------------------------------
    A document edited at its source produces a new document identifier, and
    the previous document's nodes remain in the store. They must: a stored
    evidence item cites a node identifier, and if that identifier stopped
    resolving, a previously verified response would become unauditable, which
    forfeits the guarantee this system exists to provide.

    So superseded nodes are not removed, they are withheld. They remain
    resolvable by identifier for audit, and stop being returned as current
    evidence.

    A missing supersession record is treated as ``not known to be
    superseded``, and the result is counted rather than silently accepted. The
    alternative, dropping nodes whose source has no record, would discard
    valid evidence on the strength of a record that does not exist, and the
    loss would be invisible.

Dependency contract
-------------------
    Imports only the standard library, the retrieval ``exceptions``,
    ``config``, ``models`` and ``interfaces`` modules, and the knowledge
    schemas. It constructs no store and names no backend.
"""

from __future__ import annotations

from typing import Final, Mapping, Optional, Sequence

from ptfm.core.retrieval.config import SupersessionConfig
from ptfm.core.retrieval.exceptions import (
    HydrationError,
    NodeNotFoundError,
    ProvenanceIncompleteError,
    SupersessionError,
)
from ptfm.core.retrieval.interfaces import NodeStore, StructuredLogger, SupersessionStore
from ptfm.core.retrieval.models import FusedCandidate, SegmentationPartition
from ptfm.core.schemas import KnowledgeNode

__all__ = ["HydratedResult", "NodeHydrator", "SupersessionFilter"]

#: Provenance a node must carry to become evidence. Each is a link in the chain
#: from a claim back to a region of an identified source document, and a node
#: missing any one of them cannot be cited.
_REQUIRED_PROVENANCE: Final[tuple[str, ...]] = (
    "node_id",
    "document_id",
    "span",
    "text",
    "source",
    "segmentation_fingerprint",
)


class HydratedResult:
    """
    One ranked result, with its authoritative node attached.

    Pairs the node with the fused candidate that produced it, so that a later
    stage can build evidence without re-joining two sequences by identifier and
    without either sequence's order being load-bearing.

    Deliberately not a Pydantic model. It is an internal carrier between two
    adjacent stages, never serialized and never crossing a module boundary, and
    validating an already-validated node again on every result would cost
    latency on the query path for no guarantee that is not already held.

    Parameters
    ----------
    candidate:
        The fused candidate, carrying every arm's observation of this node.
    node:
        The authoritative node, resolved from the node store.
    """

    __slots__ = ("candidate", "node")

    def __init__(self, candidate: FusedCandidate, node: KnowledgeNode) -> None:
        self.candidate = candidate
        self.node = node

    @property
    def node_id(self) -> str:
        """
        Identifier of the resolved node.

        Returns
        -------
        str
            The node identifier.
        """
        return self.node.node_id

    @property
    def document_id(self) -> str:
        """
        Identifier of the document this result came from.

        Returns
        -------
        str
            The document identifier.
        """
        return self.node.document_id

    @property
    def source_uri(self) -> str:
        """
        Source this result's document was acquired from.

        Returns
        -------
        str
            The source URI.
        """
        return self.node.source.source_uri

    def __repr__(self) -> str:
        """
        Return a representation naming the node and its fused rank.

        Returns
        -------
        str
            Diagnostic representation.
        """
        return (
            f"HydratedResult(node_id={self.node_id!r}, "
            f"fused_rank={self.candidate.fused_rank})"
        )


class NodeHydrator:
    """
    Resolves ranked identifiers to authoritative nodes.

    The only component that reads content, and therefore the only one that can
    establish provenance. Everything downstream inherits what it verifies here.

    Provenance is verified on every result, with no setting to relax it.
    Completeness of provenance is the guarantee the module exists to provide, so
    a switch could only ever be used to admit evidence that cannot be cited.
    The cost is a handful of attribute reads per result, which does not merit a
    knob.

    Parameters
    ----------
    store:
        The authoritative node store.
    logger:
        Structured logger.
    """

    __slots__ = ("_store", "_logger")

    def __init__(self, store: NodeStore, logger: StructuredLogger) -> None:
        self._store = store
        self._logger = logger

    async def hydrate(
        self,
        candidates: Sequence[FusedCandidate],
        partition: SegmentationPartition,
    ) -> tuple[HydratedResult, ...]:
        """
        Resolve candidates to nodes, preserving fused order.

        Resolved in one batch. A per-candidate round trip would multiply the
        store's latency by the result count on the hottest path in the module.

        Parameters
        ----------
        candidates:
            The fused candidates, in fused order.
        partition:
            The segmentation partition to resolve within.

        Returns
        -------
        tuple[HydratedResult, ...]
            One result per candidate, in fused order.

        Raises
        ------
        HydrationError
            If the store could not be reached.
        NodeNotFoundError
            If an index returned an identifier the node store does not hold.
        ProvenanceIncompleteError
            If a resolved node cannot be cited.
        """
        if not candidates:
            return ()

        node_ids = tuple(candidate.node_id for candidate in candidates)

        try:
            resolved = await self._store.get_many(node_ids, partition)
        except (NodeNotFoundError, ProvenanceIncompleteError):
            raise
        except HydrationError:
            raise
        except Exception as exc:
            raise HydrationError(
                f"Node store {self._store.name!r} could not resolve "
                f"{len(node_ids)} identifiers: {type(exc).__name__}: {exc}. "
                f"Without it a result carries no text, span, or source, so "
                f"there is nothing that could be called evidence.",
                context={"store_name": self._store.name},
            ) from exc

        self._verify_resolved(node_ids, resolved, partition)

        results = tuple(
            HydratedResult(candidate=candidate, node=resolved[candidate.node_id])
            for candidate in candidates
        )

        for result in results:
            self._verify_provenance(result.node)
            self._verify_partition_membership(result.node, partition)

        return results

    # ------------------------------------------------------------------
    # Verification
    # ------------------------------------------------------------------

    def _verify_resolved(
        self,
        node_ids: Sequence[str],
        resolved: Mapping[str, KnowledgeNode],
        partition: SegmentationPartition,
    ) -> None:
        """
        Ensure every ranked identifier resolved.

        An unresolved identifier means an index and the node store have
        diverged, ordinarily because a write reached one and not the other.

        Surfaced rather than skipped. Silently dropping the result would
        present an inconsistent store as a merely less relevant corpus, and the
        inconsistency would persist unnoticed while queries quietly returned
        less than the corpus holds.

        Parameters
        ----------
        node_ids:
            Identifiers that were requested.
        resolved:
            Identifiers the store returned.
        partition:
            The partition searched, recorded for diagnosis.

        Returns
        -------
        None

        Raises
        ------
        NodeNotFoundError
            If any identifier did not resolve.
        """
        missing = [node_id for node_id in node_ids if node_id not in resolved]
        if not missing:
            return

        raise NodeNotFoundError(
            f"{len(missing)} of {len(node_ids)} ranked identifiers do not "
            f"resolve in node store {self._store.name!r}. An index and the "
            f"node store have diverged, so the corpus is returning less than "
            f"it holds.",
            missing_node_ids=missing,
            context={
                "store_name": self._store.name,
                "partition_key": partition.key,
            },
        )

    def _verify_provenance(self, node: KnowledgeNode) -> None:
        """
        Ensure a node carries what a citation needs.

        The last guard on the Evidence-First chain inside retrieval. A node
        without a span, a document, or a source cannot be traced to a region of
        an identified document, and admitting one would let a verified claim
        cite something unlocatable.

        Parameters
        ----------
        node:
            The node to check.

        Returns
        -------
        None

        Raises
        ------
        ProvenanceIncompleteError
            If a required field is absent or empty.
        """
        missing = [
            field
            for field in _REQUIRED_PROVENANCE
            if getattr(node, field, None) in (None, "")
        ]

        if missing:
            raise ProvenanceIncompleteError(
                f"Node {node.node_id!r} lacks provenance required to become "
                f"evidence: {', '.join(missing)}. A claim citing it could not "
                f"be traced to a region of an identified document.",
                missing_fields=missing,
                node_id=node.node_id,
            )

        if not getattr(node.source, "source_uri", ""):
            raise ProvenanceIncompleteError(
                f"Node {node.node_id!r} carries a source descriptor with no "
                f"source URI, so the document it came from cannot be named.",
                missing_fields=["source.source_uri"],
                node_id=node.node_id,
            )

    def _verify_partition_membership(
        self,
        node: KnowledgeNode,
        partition: SegmentationPartition,
    ) -> None:
        """
        Ensure a resolved node belongs to the partition that was searched.

        A node from another segmentation would carry spans computed against a
        different division of its document, so its offsets would not mean what
        they say.

        Checked on every result rather than trusted. The partition is how the
        store is addressed, so a mismatch should be impossible and the check
        should never fire; that is exactly why it is cheap to keep and why its
        absence would be the thing that let a store which mixed partitions go
        unnoticed.

        Parameters
        ----------
        node:
            The node to check.
        partition:
            The partition that was searched.

        Returns
        -------
        None

        Raises
        ------
        ProvenanceIncompleteError
            If the node belongs to another partition.
        """
        if partition.accepts(
            node.segmentation_fingerprint, node.source_schema_version
        ):
            return

        raise ProvenanceIncompleteError(
            f"Node {node.node_id!r} was segmented under "
            f"{node.segmentation_fingerprint!r} against document contract "
            f"{node.source_schema_version!r}, which is not the partition "
            f"searched. Its spans were computed against a different division "
            f"of its document, so its offsets do not mean what they say here.",
            missing_fields=["segmentation_fingerprint"],
            node_id=node.node_id,
            context={"partition_key": partition.key},
        )

    def __repr__(self) -> str:
        """
        Return a representation naming the store.

        Returns
        -------
        str
            Diagnostic representation.
        """
        return f"NodeHydrator(store={self._store.name!r})"


class SupersessionFilter:
    """
    Withholds results whose documents no longer represent their sources.

    Withholds rather than deletes. A superseded node stays resolvable by
    identifier so that evidence cited before the change remains auditable; it
    simply stops being offered as current.

    Withdrawn and superseded are treated differently
    ------------------------------------------------
    ``SupersessionConfig.include_withdrawn`` admits material from a source that
    has been removed, and never an older version of a source that still exists.

    The asymmetry is deliberate. A superseded document has a successor, so
    including it would offer two versions of one source as though both were
    current, which is the failure the filter exists to prevent. No setting
    admits that.

    A withdrawn source has no successor, so the question is different: whether
    to return material a publisher retracted. That is a judgement a deployment
    must make for itself, and the default is to withhold it. An audit needing
    retracted material can still reach it by identifier, which is why enabling
    this is a convenience rather than the only route to it.

    Parameters
    ----------
    store:
        The supersession store.
    config:
        Supersession configuration.
    logger:
        Structured logger.
    """

    __slots__ = ("_store", "_config", "_logger")

    def __init__(
        self,
        store: SupersessionStore,
        config: SupersessionConfig,
        logger: StructuredLogger,
    ) -> None:
        self._store = store
        self._config = config
        self._logger = logger

    async def filter(
        self,
        results: Sequence[HydratedResult],
        partition: SegmentationPartition,
    ) -> tuple[tuple[HydratedResult, ...], int, int]:
        """
        Remove results whose documents have been superseded or withdrawn.

        Looked up by distinct source rather than per result, because a result
        set ordinarily spans far fewer sources than results and a per-result
        lookup would repeat the same question.

        Filtering is unconditional. ``SupersessionConfig`` rejects a
        configuration that disables it, so a branch on that setting here would
        be unreachable, and unreachable code that appears to offer a choice is
        worse than no code at all.

        Parameters
        ----------
        results:
            Hydrated results, in fused order.
        partition:
            The segmentation partition to consult.

        Returns
        -------
        tuple[tuple[HydratedResult, ...], int, int]
            The surviving results in their original order, the number withheld,
            and the number whose source had no record and therefore passed
            unchecked.

            The last count is returned rather than logged alone because it
            measures how much of the filter was inoperative for this query, and
            a consumer deciding how much to trust a result set should be able
            to see it.

        Raises
        ------
        SupersessionError
            If currency could not be determined. Never degraded past: serving
            superseded content as current is a truthfulness failure, and an
            invisible one.
        """
        if not results:
            return (), 0, 0

        source_uris = sorted({result.source_uri for result in results})

        try:
            current = await self._store.current_document_ids(source_uris, partition)
        except SupersessionError:
            raise
        except Exception as exc:
            raise SupersessionError(
                f"Supersession store {self._store.name!r} could not report "
                f"which documents are current: {type(exc).__name__}: {exc}. "
                f"Proceeding would serve superseded content as current "
                f"evidence, which no consumer could detect.",
                context={"store_name": self._store.name},
            ) from exc

        kept: list[HydratedResult] = []
        withheld = 0
        unchecked = 0

        for result in results:
            verdict = _currency_of(result, current)

            if verdict == _Currency.UNRECORDED:
                unchecked += 1
                kept.append(result)
            elif verdict == _Currency.CURRENT:
                kept.append(result)
            elif (
                verdict == _Currency.WITHDRAWN and self._config.include_withdrawn
            ):
                # A source with no successor. Admitting it answers what the
                # source said before removal, which a superseded document's
                # older version cannot answer without presenting two versions
                # of one source as current.
                kept.append(result)
            else:
                withheld += 1

        if withheld:
            self._logger.debug(
                "retrieval.supersession.withheld",
                withheld=withheld,
                sources=len(source_uris),
            )

        if unchecked:
            # An ingest path that wrote nodes without recording their source's
            # currency. Reported rather than passed over, because the filter was
            # inoperative for those results and nothing else would say so.
            self._logger.warning(
                "retrieval.supersession.unrecorded_sources",
                unchecked=unchecked,
                sources=len(source_uris),
            )

        return tuple(kept), withheld, unchecked

    def __repr__(self) -> str:
        """
        Return a representation naming the store and whether filtering is on.

        Returns
        -------
        str
            Diagnostic representation.
        """
        return (
            f"SupersessionFilter(store={self._store.name!r}, "
            f"enabled={self._config.filter_superseded})"
        )


# ============================================================================
# HELPERS
# ============================================================================


class _Currency:
    """
    Verdict of the supersession check for one result.

    A three-valued outcome rather than a boolean, because ``not current`` and
    ``nothing is known`` call for different handling and collapsing them would
    either discard valid evidence or hide an inoperative filter.
    """

    CURRENT: Final[str] = "CURRENT"
    SUPERSEDED: Final[str] = "SUPERSEDED"
    WITHDRAWN: Final[str] = "WITHDRAWN"
    UNRECORDED: Final[str] = "UNRECORDED"


def _currency_of(
    result: HydratedResult,
    current: Mapping[str, Optional[str]],
) -> str:
    """
    Decide whether one result's document is still current.

    Three cases, distinguished because they mean different things:

    A source ABSENT from the mapping has no record. The document is not known
    to be superseded, so it passes. Dropping it would discard valid evidence on
    the strength of a record that does not exist.

    A source present with no document identifier is a TOMBSTONE: the source was
    withdrawn. Every node from it is withheld, and this is distinct from having
    no record at all, which is why a tombstone is stored rather than the record
    being removed.

    A source present with an identifier names the document that currently
    represents it. Any other document from that source has been superseded.

    Parameters
    ----------
    result:
        The result to judge.
    current:
        Current document identifier by source URI, as the store reports it.

    Returns
    -------
    str
        One of the ``_Currency`` verdicts.
    """
    if result.source_uri not in current:
        return _Currency.UNRECORDED

    current_document = current[result.source_uri]

    if current_document is None:
        return _Currency.WITHDRAWN

    if current_document == result.document_id:
        return _Currency.CURRENT

    return _Currency.SUPERSEDED
