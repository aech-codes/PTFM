"""
PTFM Retrieval Arms

File:
    ptfm/core/retrieval/pipeline/arms.py

Description:
    The three ways this module finds candidate nodes: by vector similarity, by
    term match, and by structural constraint.

Arms find; they do not judge, and they do not carry content
-----------------------------------------------------------
    An arm returns node identifiers, ranks, and optionally a score with the
    scale it lives on. It never returns text, spans, or provenance, so an
    index holding a stale or corrupted copy cannot change what a citation
    resolves to. Authoritative content arrives later, from the node store.

    No arm converts its score into a probability, and none compares its score
    with another arm's. A cosine similarity, a lexical relevance value, and a
    structural membership are three incomparable quantities, which is why each
    travels with its scale and why fusion consumes ranks rather than scores.

Independence is the point
-------------------------
    The arms share nothing but the plan. Each holds its own store, applies its
    own deadline, and fails on its own, so one being unavailable reduces a
    result rather than failing a query.

    That independence is why an arm must not raise for an empty result.
    Finding nothing is an answer, and reporting it as a failure would make a
    corpus that simply lacks the answer indistinguishable from a store that
    could not be reached.

Honest signals
--------------
    The structural arm frequently produces membership rather than relevance:
    the rows of a table have no intrinsic order with respect to a query.
    Where that is so, it returns candidates carrying rank alone and no score,
    and the ``RANK_ONLY`` scale records that the absence is deliberate.

    Fabricating a relevance value to make the arm look comparable with the
    others would be worse than contributing none, because a fabricated signal
    is indistinguishable from a measured one once it reaches fusion.

Dependency contract
-------------------
    Imports only the standard library and the retrieval ``exceptions``,
    ``config``, ``models`` and ``interfaces`` modules. No backend driver, no
    plugin, no store implementation.
"""

from __future__ import annotations

import asyncio
import time
from typing import Final, Sequence

from ptfm.core.retrieval.config import DenseConfig, GraphConfig, SparseConfig
from ptfm.core.retrieval.exceptions import (
    ArmTimeoutError,
    ArmUnavailableError,
    PTFMRetrievalError,
)
from ptfm.core.retrieval.interfaces import (
    LexicalIndex,
    RelationStore,
    RetrievalStrategy,
    VectorIndex,
)
from ptfm.core.retrieval.models import (
    Candidate,
    EmbeddingPartition,
    FilterField,
    FilterOperator,
    QueryPlan,
    RetrievalFilter,
    SegmentationPartition,
)
from ptfm.core.schemas import DiscoveryMethod

__all__ = [
    "DENSE_ARM_NAME",
    "SPARSE_ARM_NAME",
    "GRAPH_ARM_NAME",
    "DenseArm",
    "SparseArm",
    "StructuralArm",
]

# ============================================================================
# HEALTH PROBES
# ============================================================================

#: Partitions and a filter used only to ask a store whether it is answering.
#:
#: A health check has to exercise the store's real query path, and every query
#: path takes a partition. Passing ``None`` would raise inside the store and
#: report every healthy backend as unhealthy; passing nothing at all lets a
#: store short-circuit on an empty argument and report an unreachable backend as
#: healthy. Both are checks that cannot fail in the direction that matters.
#:
#: These partitions are deliberately ones no corpus holds. A query against them
#: runs end to end and returns nothing, which is exactly what a readiness probe
#: needs: it establishes that the store answers, and asserts nothing about
#: content. A probe that expected results would fail on an empty corpus, which
#: is not an unhealthy one.
_PROBE_SEGMENTATION: Final[SegmentationPartition] = SegmentationPartition(
    segmentation_fingerprint="ptfm-health-probe",
    source_schema_version="ptfm-health-probe",
)

_PROBE_EMBEDDING: Final[EmbeddingPartition] = EmbeddingPartition(
    segmentation=_PROBE_SEGMENTATION,
    model_identity="ptfm-health-probe",
    dimension=1,
    normalized=False,
    composition_id="ptfm-health-probe",
)

#: A structural search returns nothing when given no filter, short-circuiting
#: before it reaches the store, so the probe supplies one.
_PROBE_FILTER: Final[RetrievalFilter] = RetrievalFilter(
    field=FilterField.DOCUMENT_ID,
    operator=FilterOperator.EQUALS,
    values=("ptfm-health-probe",),
)

#: A lexical search returns nothing when the query has no terms, short-circuiting
#: before it reaches the store, so the probe supplies a term.
_PROBE_TERM: Final[str] = "ptfmhealthprobe"

#: Selectors matching the configured arm names.
DENSE_ARM_NAME: Final[str] = "dense"
SPARSE_ARM_NAME: Final[str] = "sparse"
GRAPH_ARM_NAME: Final[str] = "graph"


class DenseArm(RetrievalStrategy):
    """
    Finds nodes whose vectors are nearest the query vector.

    The primary recall mechanism for prose. It finds passages that mean what
    the query means without sharing its words, which is precisely what a term
    index cannot do.

    Requires a query vector, so it does not run when encoding was unavailable
    or when a deployment has no embedding provider. That is reported as a
    skipped arm rather than a failed one: the arm was never asked, and calling
    a deliberate configuration a failure would misdescribe it.

    Comparability is guaranteed by the partition rather than checked here.
    Vectors from differing models are not in the same partition, so a query
    reaching this arm has already been established as comparable with what it
    will search.

    Parameters
    ----------
    index:
        The vector index to search.
    config:
        Arm configuration supplying depth, deadline, and exactness.
    
    Fetch depth
    -----------
    The count searched for is ``plan.top_k``, which the engine has already
    resolved through ``RetrievalConfig.resolve_fetch_depth``: at least the
    caller's requested count, at least this arm's configured floor, and deeper
    still for a filtered approximate search. The arm does not compute it, so a
    query cannot be served at a depth that disagrees with the one recorded in
    diagnostics.
    """

    __slots__ = ("_index", "_config")

    def __init__(self, index: VectorIndex, config: DenseConfig) -> None:
        self._index = index
        self._config = config

    @property
    def name(self) -> str:
        """
        Stable identifier of this arm.

        Returns
        -------
        str
            Implementation identifier.
        """
        return DENSE_ARM_NAME

    @property
    def method(self) -> DiscoveryMethod:
        """
        How this arm discovers nodes.

        Recorded on every candidate, so a result's discovery history names the
        method that found it.

        Returns
        -------
        DiscoveryMethod
            Dense discovery.
        """
        return DiscoveryMethod.DENSE

    @property
    def requires_vector(self) -> bool:
        """
        Whether this arm can run without a query vector.

        Returns
        -------
        bool
            ``True``; a vector space cannot be searched without a vector.
        """
        return True

    async def search(self, plan: QueryPlan) -> Sequence[Candidate]:
        """
        Find the nearest nodes to the query vector.

        Parameters
        ----------
        plan:
            The prepared query.

        Returns
        -------
        Sequence[Candidate]
            Candidates in descending similarity. Empty when the plan carries
            no vector, which is a skip rather than a failure.

        Raises
        ------
        ArmUnavailableError
            If the index could not be reached.
        ArmTimeoutError
            If the search exceeded its deadline.
        """
        if not plan.has_vector or plan.embedding is None:
            # The arm was never asked. Returning empty rather than raising
            # keeps a deliberate configuration from being reported as a fault.
            return ()

        return await _guarded(
            self._index.search(
                plan.vector,  # type: ignore[arg-type]
                plan.embedding,
                plan.top_k,
                filters=plan.filters,
                exact=plan.exact or self._config.exact,
            ),
            arm_name=self.name,
            timeout_seconds=self._config.timeout_seconds,
            query_id=plan.query_id,
        )

    async def health_check(self) -> bool:
        """
        Verify that this arm can search.

        Returns
        -------
        bool
            ``True`` when the index responded.
        """
        try:
            await self._index.count(_PROBE_EMBEDDING)
        except Exception:  # noqa: BLE001 - a health check reports, never raises
            return False
        return True

    def __repr__(self) -> str:
        """
        Return a representation naming the configured depth.

        Returns
        -------
        str
            Diagnostic representation.
        """
        return f"DenseArm(top_k={self._config.top_k}, exact={self._config.exact})"


class SparseArm(RetrievalStrategy):
    """
    Finds nodes matching the query's terms.

    Not a fallback for dense retrieval but a complement to it, and on this
    system's workload an essential one. Identifiers, quantities, dates, and
    rare proper nouns are exactly what verification checks, and exactly what
    an embedding blurs: a vector cannot reliably distinguish one account
    number from another, while a term index matches it exactly.

    Requires no vector, so it serves whenever dense retrieval cannot: with no
    embedding provider configured, or after an encoding failure has degraded
    the query.

    Parameters
    ----------
    index:
        The lexical index to search.
    config:
        Arm configuration supplying depth and deadline.
    
    Fetch depth
    -----------
    The count searched for is ``plan.top_k``, which the engine has already
    resolved through ``RetrievalConfig.resolve_fetch_depth``: at least the
    caller's requested count, at least this arm's configured floor, and deeper
    still for a filtered approximate search. The arm does not compute it, so a
    query cannot be served at a depth that disagrees with the one recorded in
    diagnostics.
    """

    __slots__ = ("_index", "_config")

    def __init__(self, index: LexicalIndex, config: SparseConfig) -> None:
        self._index = index
        self._config = config

    @property
    def name(self) -> str:
        """
        Stable identifier of this arm.

        Returns
        -------
        str
            Implementation identifier.
        """
        return SPARSE_ARM_NAME

    @property
    def method(self) -> DiscoveryMethod:
        """
        How this arm discovers nodes.

        Returns
        -------
        DiscoveryMethod
            Sparse discovery.
        """
        return DiscoveryMethod.SPARSE

    @property
    def requires_vector(self) -> bool:
        """
        Whether this arm can run without a query vector.

        Returns
        -------
        bool
            ``False``; term matching needs only text.
        """
        return False

    async def search(self, plan: QueryPlan) -> Sequence[Candidate]:
        """
        Find nodes matching the query's terms.

        Parameters
        ----------
        plan:
            The prepared query.

        Returns
        -------
        Sequence[Candidate]
            Candidates in descending relevance. Scores are corpus-relative and
            unbounded, which is why they are never compared against a
            similarity.

        Raises
        ------
        ArmUnavailableError
            If the index could not be reached.
        ArmTimeoutError
            If the search exceeded its deadline.
        """
        if not plan.text.strip():
            return ()

        return await _guarded(
            self._index.search(
                plan.text,
                plan.segmentation,
                plan.top_k,
                filters=plan.filters,
            ),
            arm_name=self.name,
            timeout_seconds=self._config.timeout_seconds,
            query_id=plan.query_id,
        )

    async def health_check(self) -> bool:
        """
        Verify that this arm can search.

        Returns
        -------
        bool
            ``True`` when the index responded.
        """
        try:
            await self._index.search(
                _PROBE_TERM, _PROBE_SEGMENTATION, 1
            )
        except Exception:  # noqa: BLE001 - a health check reports, never raises
            return False
        return True

    def __repr__(self) -> str:
        """
        Return a representation naming the configured depth.

        Returns
        -------
        str
            Diagnostic representation.
        """
        return f"SparseArm(top_k={self._config.top_k})"


class StructuralArm(RetrievalStrategy):
    """
    Finds nodes satisfying a structural constraint.

    Answers questions similarity cannot: which nodes fall under a heading,
    belong to a table, appear on a page, or carry a metadata value a domain
    plugin attached. That is a narrower job than the other two arms do, and
    the arm is honest about it.

    Membership, not relevance
    -------------------------
    A structural constraint frequently implies no ordering. The rows of a
    table have no intrinsic rank with respect to a query, and neither do the
    nodes under a heading.

    Where the store returns candidates without scores, this arm passes that
    through rather than manufacturing one. A fabricated relevance value is
    indistinguishable from a measured one by the time it reaches fusion, so
    contributing none is strictly better than contributing an invented one.

    A constraint is required
    ------------------------
    Without filters the arm has no question to answer, and it returns nothing
    rather than the corpus. Returning everything would present arbitrary nodes
    as responsive to a query they have no relation to, and would swamp the
    arms that did answer it.

    Parameters
    ----------
    store:
        The relation store to search.
    config:
        Arm configuration supplying depth and deadline.
    
    Fetch depth
    -----------
    The count searched for is ``plan.top_k``, which the engine has already
    resolved through ``RetrievalConfig.resolve_fetch_depth``: at least the
    caller's requested count, at least this arm's configured floor, and deeper
    still for a filtered approximate search. The arm does not compute it, so a
    query cannot be served at a depth that disagrees with the one recorded in
    diagnostics.
    """

    __slots__ = ("_store", "_config")

    def __init__(self, store: RelationStore, config: GraphConfig) -> None:
        self._store = store
        self._config = config

    @property
    def name(self) -> str:
        """
        Stable identifier of this arm.

        Returns
        -------
        str
            Implementation identifier.
        """
        return GRAPH_ARM_NAME

    @property
    def method(self) -> DiscoveryMethod:
        """
        How this arm discovers nodes.

        Returns
        -------
        DiscoveryMethod
            Structural discovery.
        """
        return DiscoveryMethod.GRAPH

    @property
    def requires_vector(self) -> bool:
        """
        Whether this arm can run without a query vector.

        Returns
        -------
        bool
            ``False``; structure is searched by constraint, not by similarity.
        """
        return False

    async def search(self, plan: QueryPlan) -> Sequence[Candidate]:
        """
        Find nodes satisfying the plan's structural constraint.

        Parameters
        ----------
        plan:
            The prepared query.

        Returns
        -------
        Sequence[Candidate]
            Matching nodes. Empty when the plan carries no constraint, which
            is a skip rather than a failure: an unconstrained structural
            search has no question to answer.

        Raises
        ------
        ArmUnavailableError
            If the store could not be reached.
        ArmTimeoutError
            If the search exceeded its deadline.
        """
        if not plan.filters:
            return ()

        return await _guarded(
            self._store.structural_search(
                plan.filters,
                plan.segmentation,
                plan.top_k,
            ),
            arm_name=self.name,
            timeout_seconds=self._config.timeout_seconds,
            query_id=plan.query_id,
        )

    async def health_check(self) -> bool:
        """
        Verify that this arm can search.

        Returns
        -------
        bool
            ``True`` when the store responded.
        """
        try:
            await self._store.structural_search(
                (_PROBE_FILTER,), _PROBE_SEGMENTATION, 1
            )
        except Exception:  # noqa: BLE001 - a health check reports, never raises
            return False
        return True

    def __repr__(self) -> str:
        """
        Return a representation naming the configured depth.

        Returns
        -------
        str
            Diagnostic representation.
        """
        return f"StructuralArm(top_k={self._config.top_k})"


# ============================================================================
# SHARED EXECUTION
# ============================================================================


async def _guarded(
    search: object,
    *,
    arm_name: str,
    timeout_seconds: float,
    query_id: str,
) -> Sequence[Candidate]:
    """
    Run a store search under a deadline, converting every failure.

    Shared by all three arms, because the deadline and the conversion are the
    same concern regardless of what is being searched, and three copies would
    drift.

    Every failure becomes a typed, degradable arm error. The engine catches
    those and records a diagnostic, so one store being unreachable reduces a
    result rather than failing a query.

    Cancellation is never converted. An abandoned query must terminate
    promptly, and turning cancellation into a degradable failure would let a
    cancelled arm appear to have merely failed.

    Parameters
    ----------
    search:
        The awaitable store search.
    arm_name:
        Identifier of the arm, recorded on any failure.
    timeout_seconds:
        The arm's deadline.
    query_id:
        Correlation identifier.

    Returns
    -------
    Sequence[Candidate]
        The store's candidates, unchanged.

    Raises
    ------
    ArmTimeoutError
        If the search exceeded its deadline.
    ArmUnavailableError
        If the search failed for any other reason.
    """
    started = time.monotonic()

    try:
        return await asyncio.wait_for(search, timeout=timeout_seconds)  # type: ignore[arg-type]
    except TimeoutError as exc:
        raise ArmTimeoutError(
            arm_name=arm_name,
            timeout_seconds=timeout_seconds,
            query_id=query_id,
        ) from exc
    except PTFMRetrievalError as error:
        # Already typed by the store. Enriched rather than replaced, so that a
        # store's own diagnosis survives.
        if error.query_id is None:
            error.query_id = query_id
        error.add_context(
            arm_name=arm_name,
            elapsed_ms=(time.monotonic() - started) * 1000.0,
        )
        raise
    except Exception as exc:
        raise ArmUnavailableError(
            f"Arm {arm_name!r} failed: {type(exc).__name__}: {exc}",
            arm_name=arm_name,
            query_id=query_id,
        ) from exc
