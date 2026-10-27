"""
PTFM Retrieval Engine

File:
    ptfm/core/retrieval/engine.py

Description:
    The orchestrator that runs one query through every retrieval stage, and the
    ingestion path that populates the stores those stages read.

The pipeline, and why the order is not negotiable
-------------------------------------------------
    prepare -> search -> fuse -> hydrate -> currency -> expand -> credibility
    -> assemble

    Each boundary exists because of a data dependency that cannot be reordered
    without weakening a guarantee:

    HYDRATION precedes everything that reads text. Arms return identifiers and
    uncalibrated signals, never content: an index holds a copy of a passage
    that may be stale, and a stale copy presented as evidence is a citation to
    something the source does not say. Authoritative text comes only from the
    node store, and provenance is verified there.

    CURRENCY follows hydration, because deciding whether a result is current
    needs the source URI that hydration resolves, not the identifier the index
    returned.

    EXPANSION follows currency, so that context is never gathered around an
    anchor that was about to be withheld.

    CREDIBILITY follows expansion but is asked only about ANCHOR sources.
    Context is there to make an anchor readable, and scoring it would put a
    passage nobody retrieved into a corroboration count.

    ASSEMBLY is last and invents nothing. It carries forward what the earlier
    stages established and refuses to fill a gap any of them left open.

What this engine will not do
----------------------------
    It computes no confidence, no calibration, and no verification. Retrieval
    signals leave here exactly as uncalibrated as they arrived, each named by
    its scale. The engine's contribution to truthfulness is that everything it
    returns is traceable and that everything it could not establish is
    reported.

    It also never returns an empty result to stand in for a failed search. If
    every arm fails, that is not evidence of absence, and the query raises.

Degradation
-----------
    A degradable stage failure reduces the result and is recorded; a
    non-degradable one fails the query. Which is which is a property of the
    exception rather than a decision made here, so two stages cannot disagree
    about the same failure. Under ``DegradationPolicy.STRICT`` nothing degrades
    at all.

    The distinction that matters most: a reduced result is always marked as
    reduced. ``RetrievalDiagnostics.degraded`` is true whenever anything did
    not run, so a consumer cannot mistake a two-arm result for a three-arm one.

    One stage is outside this scheme. The currency filter never degrades,
    because its failure would make an answer WRONG rather than smaller: a
    superseded document served as current is confidently incorrect and carries
    no marker a consumer could detect. Degradation is for missing evidence, not
    for unverified evidence.

Dependency contract
-------------------
    Every collaborator is injected, and every one is named by protocol. This
    module imports no store, no backend, and no driver, so the engine runs
    unchanged over the in-memory stores or over PostgreSQL. Construction lives
    in the composition root.
"""

from __future__ import annotations

import asyncio
import time
from typing import Mapping, Optional, Sequence

from ptfm.core.ingestion.interfaces import (
    Clock,
    IdGenerator,
    MetricsSink,
    StructuredLogger,
)
from ptfm.core.knowledge.models import KnowledgeUnit
from ptfm.core.retrieval.config import ArmConfig, RetrievalConfig
from ptfm.core.retrieval.exceptions import (
    AllArmsFailedError,
    PTFMRetrievalError,
    RetrievalConfigurationError,
    StoreUnavailableError,
)
from ptfm.core.retrieval.interfaces import (
    ContextExpander,
    FusionStrategy,
    IndexWriter,
    NodeStore,
    RetrievalStrategy,
    VectorIndex,
)
from ptfm.core.retrieval.models import (
    ArmDiagnostic,
    ArmResult,
    ArmStatus,
    EmbeddingPartition,
    QueryPlan,
    RetrievalDiagnostics,
    RetrievalFilter,
    SegmentationPartition,
)
from ptfm.core.retrieval.pipeline.assembly import (
    AssembledEvidence,
    CredibilityAnnotator,
    EvidenceAssembler,
)
from ptfm.core.retrieval.pipeline.encoding import QueryPreparer
from ptfm.core.retrieval.pipeline.hydration import (
    HydratedResult,
    NodeHydrator,
    SupersessionFilter,
)
from ptfm.core.schemas import DiscoveryMethod, KnowledgeNode, RetrievalResult

__all__ = [
    "RetrievalEngine",
    "RetrievalOutcome",
    "EmbeddingCoverageProbe",
]


class RetrievalOutcome:
    """
    Everything one query produced, including what it failed to produce.

    Evidence and diagnostics travel together and are never separable. A caller
    holding evidence without diagnostics cannot tell whether an arm was missing
    when it was gathered, and that is precisely the difference between a
    complete result and a degraded one presented as complete.

    Attributes
    ----------
    query:
        The query text as searched, after any preprocessing.
    evidence:
        The assembled documents and evidence items.
    diagnostics:
        What ran, what did not, and what could not be established.
    """

    __slots__ = ("query", "evidence", "diagnostics")

    def __init__(
        self,
        query: str,
        evidence: AssembledEvidence,
        diagnostics: RetrievalDiagnostics,
    ) -> None:
        """
        Bind evidence to the diagnostics describing how it was obtained.

        Parameters
        ----------
        query:
            The query text as searched.
        evidence:
            The assembled evidence.
        diagnostics:
            The query's diagnostics.
        """
        self.query = query
        self.evidence = evidence
        self.diagnostics = diagnostics

    @property
    def degraded(self) -> bool:
        """
        Whether any stage failed to run in full.

        Returns
        -------
        bool
            ``True`` when the result is reduced.
        """
        return self.diagnostics.degraded

    @property
    def reproducible(self) -> bool:
        """
        Whether this result can be reproduced exactly.

        Returns
        -------
        bool
            ``True`` when no approximate search contributed.
        """
        return self.diagnostics.reproducible

    def to_result(self) -> RetrievalResult:
        """
        Render the outcome in the cross-module retrieval schema.

        The narrower shape other modules consume. Diagnostics are not part of
        it, so a caller needing to know whether the result was degraded reads
        this object rather than its rendering.

        Returns
        -------
        RetrievalResult
            The documents, with the query and latency.
        """
        return RetrievalResult(
            query=self.query,
            documents=list(self.evidence.documents),
            retrieval_latency_ms=self.diagnostics.latency_ms,
            total_documents=len(self.evidence.documents),
        )

    def __repr__(self) -> str:
        """
        Describe the outcome without disclosing evidence text.

        Returns
        -------
        str
            A representation naming counts and degradation only, so a log line
            cannot leak retrieved content.
        """
        return (
            f"RetrievalOutcome(documents={len(self.evidence.documents)}, "
            f"anchors={self.evidence.anchor_count}, "
            f"context={self.evidence.context_count}, "
            f"degraded={self.degraded})"
        )


class EmbeddingCoverageProbe:
    """
    Measures what fraction of a partition's nodes carry vectors.

    Exists because dense retrieval over a partially embedded corpus returns a
    neighbour set drawn from the embedded subset while appearing to search the
    whole of it. Nothing in the result distinguishes "no near neighbours" from
    "the near neighbours have no vectors yet", so the fraction has to be
    measured rather than assumed.

    The measurement is cached and invalidated on write, not recomputed per
    query. Coverage changes only when something is ingested, and two counts per
    query would be two round trips for a number that had not moved.

    Attributes
    ----------
    _nodes:
        Store counting nodes in a segmentation partition.
    _vectors:
        Index counting vectors in an embedding partition.
    _cached:
        Last measurement, keyed by the pair of partitions measured.
    """

    __slots__ = ("_nodes", "_vectors", "_cached")

    def __init__(self, nodes: NodeStore, vectors: VectorIndex) -> None:
        """
        Bind the probe to the stores it counts.

        Parameters
        ----------
        nodes:
            The node store.
        vectors:
            The vector index.
        """
        self._nodes = nodes
        self._vectors = vectors
        self._cached: dict[tuple[str, str], Optional[float]] = {}

    def invalidate(self) -> None:
        """
        Discard the cached measurement.

        Called after a write, because coverage moves only when something is
        ingested.
        """
        self._cached.clear()

    async def coverage(
        self,
        segmentation: SegmentationPartition,
        embedding: EmbeddingPartition,
    ) -> Optional[float]:
        """
        Measure the fraction of nodes carrying a vector.

        Parameters
        ----------
        segmentation:
            The segmentation partition holding the nodes.
        embedding:
            The embedding partition holding the vectors.

        Returns
        -------
        Optional[float]
            The fraction in ``[0, 1]``, or ``None`` when it could not be
            measured. ``None`` rather than a guess: an unmeasured coverage must
            not be reported as full coverage, and reporting it as zero would
            disable dense retrieval on a store that was merely unreachable for
            a moment.

            An empty partition yields ``1.0``, since every one of its zero
            nodes is embedded. That is the vacuous answer rather than a
            failure, and it keeps an empty corpus from looking under-embedded.
        """
        key = (segmentation.key, embedding.key)
        if key in self._cached:
            return self._cached[key]

        try:
            node_count = await self._nodes.count(segmentation)
            vector_count = await self._vectors.count(embedding)
        except PTFMRetrievalError:
            return None

        measured = (
            1.0
            if node_count == 0
            else min(1.0, max(0.0, vector_count / node_count))
        )
        self._cached[key] = measured
        return measured


class RetrievalEngine:
    """
    Runs a query through every retrieval stage and returns traceable evidence.

    Holds no state that varies per query. One engine serves every concurrent
    query, bounded by a semaphore sized from configuration, so a burst of
    queries cannot open an unbounded number of store connections.

    Attributes
    ----------
    _preparer:
        Resolves partitions and encodes the query.
    _arms:
        Retrieval strategies with the configuration governing each.
    _fusion:
        Combines arm orderings.
    _hydrator:
        Resolves identifiers to authoritative nodes.
    _supersession:
        Withholds results whose source has a newer version.
    _expander:
        Gathers neighbouring context around anchors.
    _credibility:
        Annotates anchor sources, or records that it could not.
    _assembler:
        Builds evidence from anchors and context.
    _writer:
        Populates the stores, when this engine also ingests.
    _coverage:
        Measures embedding coverage, when a probe was supplied.
    """

    __slots__ = (
        "_preparer",
        "_arms",
        "_fusion",
        "_hydrator",
        "_supersession",
        "_expander",
        "_credibility",
        "_assembler",
        "_writer",
        "_coverage",
        "_config",
        "_clock",
        "_ids",
        "_logger",
        "_metrics",
        "_gate",
    )

    def __init__(
        self,
        *,
        preparer: QueryPreparer,
        arms: Sequence[tuple[RetrievalStrategy, ArmConfig]],
        fusion: FusionStrategy,
        hydrator: NodeHydrator,
        supersession: SupersessionFilter,
        expander: Optional[ContextExpander],
        credibility: CredibilityAnnotator,
        assembler: EvidenceAssembler,
        config: RetrievalConfig,
        clock: Clock,
        id_generator: IdGenerator,
        logger: StructuredLogger,
        metrics: MetricsSink,
        writer: Optional[IndexWriter] = None,
        coverage: Optional[EmbeddingCoverageProbe] = None,
    ) -> None:
        """
        Assemble the engine from its stages.

        Parameters
        ----------
        preparer:
            Query preparation stage.
        arms:
            Retrieval strategies paired with their configuration. The
            configuration travels with the strategy because fetch depth is
            resolved per arm and the engine must not guess which config belongs
            to which.
        fusion:
            Fusion strategy.
        hydrator:
            Node hydration stage.
        supersession:
            Currency filter.
        expander:
            Context expansion stage, or ``None`` when expansion is disabled.
        credibility:
            Credibility annotation stage.
        assembler:
            Evidence assembly stage.
        config:
            Retrieval configuration.
        clock:
            Time source, injected so that latency is measurable in a test.
        id_generator:
            Correlation identifier source.
        logger:
            Structured logger.
        metrics:
            Metrics sink.
        writer:
            Index writer, when this engine also ingests. ``None`` makes the
            engine read-only, and ``ingest`` then refuses rather than silently
            doing nothing.
        coverage:
            Embedding coverage probe, when coverage is to be enforced.

        Raises
        ------
        RetrievalConfigurationError
            If no arm was supplied, if a required atomic writer is not atomic,
            or if coverage is to be enforced with no probe to measure it. Each
            is refused at construction because each would otherwise surface as
            a query-time fault attributed to the wrong stage.
        """
        if not arms:
            raise RetrievalConfigurationError(
                "A retrieval engine needs at least one arm. An engine with no "
                "arm would return an empty result for every query, which is "
                "indistinguishable from a corpus containing no relevant "
                "evidence."
            )

        if (
            writer is not None
            and config.ingest.require_atomic_writer
            and not writer.is_atomic
        ):
            raise RetrievalConfigurationError(
                f"Configuration requires an atomic index writer but "
                f"{writer.name!r} does not provide atomicity. A non-atomic "
                f"writer can leave a node retrievable by one index with no "
                f"authoritative text, so the requirement is refused at "
                f"construction rather than discovered after a partial write."
            )

        if config.partitions.minimum_embedding_coverage > 0.0 and coverage is None:
            raise RetrievalConfigurationError(
                f"Configuration requires embedding coverage of at least "
                f"{config.partitions.minimum_embedding_coverage} but no "
                f"coverage probe was supplied. A threshold nothing measures "
                f"reads as enforced and is not."
            )

        self._preparer = preparer
        self._arms = tuple(arms)
        self._fusion = fusion
        self._hydrator = hydrator
        self._supersession = supersession
        self._expander = expander
        self._credibility = credibility
        self._assembler = assembler
        self._writer = writer
        self._coverage = coverage
        self._config = config
        self._clock = clock
        self._ids = id_generator
        self._logger = logger
        self._metrics = metrics
        self._gate = asyncio.Semaphore(config.max_concurrent_queries)

    @property
    def arm_names(self) -> tuple[str, ...]:
        """
        Names of the configured arms, in the order they are searched.

        Returns
        -------
        tuple[str, ...]
            The arm names.
        """
        return tuple(arm.name for arm, _ in self._arms)

    @property
    def ingests(self) -> bool:
        """
        Whether this engine can populate the stores.

        Returns
        -------
        bool
            ``True`` when an index writer was supplied.
        """
        return self._writer is not None

    async def retrieve(
        self,
        query: str,
        *,
        top_k: Optional[int] = None,
        filters: Sequence[RetrievalFilter] = (),
        exact: Optional[bool] = None,
    ) -> RetrievalOutcome:
        """
        Run one query through the pipeline.

        Parameters
        ----------
        query:
            The query text.
        top_k:
            Results requested. Defaults to the configured value, bounded by the
            configured maximum.
        filters:
            Structural constraints, applied during search rather than after it.
        exact:
            Whether to forbid approximate search. Defaults to configuration.
            A caller auditing a past answer passes ``True``, since an
            approximate result is not reproducible.

        Returns
        -------
        RetrievalOutcome
            Evidence with the diagnostics describing how it was obtained.

        Raises
        ------
        AllArmsFailedError
            If every arm failed. Never degraded to an empty result: an empty
            result asserts that no evidence exists, and a failed search
            establishes nothing of the kind.
        SupersessionError
            If document currency could not be determined. Never degraded past,
            whatever the policy: see ``_filter_currency``.
        PTFMRetrievalError
            If a non-degradable stage failed, or any stage failed under
            ``DegradationPolicy.STRICT``.
        """
        started = time.perf_counter()
        async with self._gate:
            try:
                async with asyncio.timeout(self._config.query_timeout_seconds):
                    return await self._run(query, top_k, filters, exact, started)
            except TimeoutError as error:
                raise StoreUnavailableError(
                    f"The query exceeded its "
                    f"{self._config.query_timeout_seconds}s budget. Reported "
                    f"as a failure rather than returning whatever had arrived, "
                    f"because a partial result gathered from an unknown subset "
                    f"of arms cannot be described in diagnostics.",
                    store_name="retrieval:engine",
                ) from error

    async def _run(
        self,
        query: str,
        top_k: Optional[int],
        filters: Sequence[RetrievalFilter],
        exact: Optional[bool],
        started: float,
    ) -> RetrievalOutcome:
        """
        Execute the pipeline stages in order.

        Parameters
        ----------
        query:
            The query text.
        top_k:
            Results requested.
        filters:
            Structural constraints.
        exact:
            Whether to forbid approximate search.
        started:
            Monotonic start time, for latency.

        Returns
        -------
        RetrievalOutcome
            The outcome.

        Raises
        ------
        PTFMRetrievalError
            If a non-degradable stage failed.
        """
        plan = await self._preparer.prepare(
            query, top_k=top_k, filters=filters, exact=exact
        )
        log = self._logger.bind(query_id=plan.query_id)

        coverage = await self._measure_coverage(plan)
        results, diagnostics = await self._search(plan, coverage, log)

        fused = self._fusion.fuse(results, plan.top_k)
        hydrated = await self._hydrator.hydrate(fused, plan.segmentation)

        anchors, superseded, unchecked = await self._filter_currency(
            hydrated, plan, log
        )
        context = await self._expand(anchors, plan, log)
        credibility = await self._credibility.credibility_for(
            tuple({result.source_uri for result in anchors})
        )
        assembled = self._assembler.assemble(anchors, context, credibility)

        latency = (time.perf_counter() - started) * 1000.0
        outcome = RetrievalOutcome(
            query=plan.text,
            evidence=assembled,
            diagnostics=RetrievalDiagnostics(
                query_id=plan.query_id,
                segmentation_key=plan.segmentation.key,
                embedding_key=(
                    plan.embedding.key if plan.embedding is not None else None
                ),
                arms=diagnostics,
                exact_search=plan.exact,
                embedding_coverage=coverage,
                fused_count=len(fused),
                expanded_count=assembled.context_count,
                superseded_count=superseded,
                currency_unchecked=unchecked,
                credibility_unavailable=assembled.credibility_unavailable,
                latency_ms=latency,
            ),
        )

        self._record(outcome)
        log.info(
            "retrieval.completed",
            documents=len(assembled.documents),
            anchors=assembled.anchor_count,
            context=assembled.context_count,
            degraded=outcome.degraded,
            latency_ms=round(latency, 3),
        )
        return outcome

    async def _measure_coverage(self, plan: QueryPlan) -> Optional[float]:
        """
        Measure embedding coverage for the plan's partitions.

        Parameters
        ----------
        plan:
            The query plan.

        Returns
        -------
        Optional[float]
            The coverage fraction, or ``None`` when there is nothing to measure
            or it could not be measured.
        """
        if self._coverage is None or plan.embedding is None:
            return None
        return await self._coverage.coverage(plan.segmentation, plan.embedding)

    async def _search(
        self,
        plan: QueryPlan,
        coverage: Optional[float],
        log: StructuredLogger,
    ) -> tuple[tuple[ArmResult, ...], tuple[ArmDiagnostic, ...]]:
        """
        Run every arm concurrently and collect what each produced.

        Concurrently because the arms are independent and a hybrid query's
        latency is otherwise their sum. Each arm's failure is isolated: one
        unavailable index reduces the result rather than failing the query,
        provided at least one arm succeeded.

        Parameters
        ----------
        plan:
            The query plan.
        coverage:
            Measured embedding coverage, used to decide whether dense retrieval
            may be trusted.
        log:
            Logger bound to this query.

        Returns
        -------
        tuple[tuple[ArmResult, ...], tuple[ArmDiagnostic, ...]]
            Results from the arms that produced candidates, and one diagnostic
            per configured arm including those that did not run.

        Raises
        ------
        AllArmsFailedError
            If every arm failed.
        PTFMRetrievalError
            If an arm failed non-degradably, or any failed under STRICT.
        """
        skipped = self._skip_reasons(plan, coverage)
        runnable = [
            (arm, arm_config)
            for arm, arm_config in self._arms
            if arm.name not in skipped
        ]

        gathered = await asyncio.gather(
            *(
                self._run_arm(arm, arm_config, plan)
                for arm, arm_config in runnable
            ),
            return_exceptions=True,
        )

        results: list[ArmResult] = []
        diagnostics: list[ArmDiagnostic] = [
            ArmDiagnostic(
                arm_name=name,
                method=method,
                status=ArmStatus.SKIPPED,
                message=reason,
            )
            for name, (method, reason) in skipped.items()
        ]
        failures: dict[str, str] = {}

        for (arm, _), produced in zip(runnable, gathered):
            if isinstance(produced, BaseException):
                diagnostic, fatal = self._diagnose(arm, produced)
                diagnostics.append(diagnostic)
                failures[arm.name] = diagnostic.message or diagnostic.status.value
                if fatal is not None:
                    raise fatal
                continue

            result, diagnostic = produced
            diagnostics.append(diagnostic)
            if result is not None:
                results.append(result)

        if failures and len(failures) == len(runnable):
            # Every arm that ran, failed. An empty result would assert that no
            # evidence exists, which a failed search does not establish.
            raise AllArmsFailedError(
                failures=failures,
                query_id=plan.query_id,
                context={"skipped": tuple(skipped)},
            )

        if not results:
            log.warning(
                "retrieval.no_candidates",
                arms_run=len(runnable),
                arms_skipped=len(skipped),
            )

        return tuple(results), tuple(diagnostics)

    def _skip_reasons(
        self,
        plan: QueryPlan,
        coverage: Optional[float],
    ) -> dict[str, tuple[DiscoveryMethod, str]]:
        """
        Determine which arms must not run, and why.

        An arm is skipped rather than allowed to fail when the reason is known
        in advance. A skip is recorded as a skip, never as a success with no
        candidates, because the two mean different things to a consumer
        deciding how much of the corpus was actually searched.

        Parameters
        ----------
        plan:
            The query plan.
        coverage:
            Measured embedding coverage.

        Returns
        -------
        dict[str, tuple[DiscoveryMethod, str]]
            Method and reason per skipped arm name.
        """
        minimum = self._config.partitions.minimum_embedding_coverage
        under_covered = (
            minimum > 0.0 and coverage is not None and coverage < minimum
        )

        skipped: dict[str, tuple[DiscoveryMethod, str]] = {}
        for arm, _ in self._arms:
            needs_vector = bool(getattr(arm, "requires_vector", False))
            if needs_vector and not plan.has_vector:
                skipped[arm.name] = (
                    arm.method,
                    "No query vector: the query was not encoded, so a dense "
                    "search has nothing to compare against.",
                )
            elif needs_vector and under_covered:
                skipped[arm.name] = (
                    arm.method,
                    f"Embedding coverage {coverage:.4f} is below the required "
                    f"{minimum}. Searching a partially embedded corpus would "
                    f"return neighbours drawn from the embedded subset while "
                    f"appearing to search all of it.",
                )
        return skipped

    async def _run_arm(
        self,
        arm: RetrievalStrategy,
        arm_config: ArmConfig,
        plan: QueryPlan,
    ) -> tuple[Optional[ArmResult], ArmDiagnostic]:
        """
        Run one arm at its resolved fetch depth.

        Fetch depth is resolved here rather than inside the arm, so that one
        place decides it and the depth a query ran at is the depth diagnostics
        describe. The arm receives a plan whose ``top_k`` is that depth; fusion
        trims back to the caller's requested count.

        Parameters
        ----------
        arm:
            The strategy to run.
        arm_config:
            That arm's configuration.
        plan:
            The query plan.

        Returns
        -------
        tuple[Optional[ArmResult], ArmDiagnostic]
            The result when the arm produced candidates, and its diagnostic.

        Raises
        ------
        PTFMRetrievalError
            If the arm failed. Handled by the caller, which decides whether the
            failure degrades the result or fails the query.
        """
        depth = self._config.resolve_fetch_depth(
            arm_config, plan.top_k, filtered=bool(plan.filters)
        )
        arm_plan = (
            plan if depth == plan.top_k else plan.model_copy(update={"top_k": depth})
        )

        started = time.perf_counter()
        candidates = await arm.search(arm_plan)
        latency = (time.perf_counter() - started) * 1000.0

        if not candidates:
            return None, ArmDiagnostic(
                arm_name=arm.name,
                method=arm.method,
                status=ArmStatus.EMPTY,
                latency_ms=latency,
            )

        return (
            ArmResult(
                arm_name=arm.name,
                method=arm.method,
                candidates=tuple(candidates),
                latency_ms=latency,
            ),
            ArmDiagnostic(
                arm_name=arm.name,
                method=arm.method,
                status=ArmStatus.COMPLETED,
                candidate_count=len(candidates),
                latency_ms=latency,
            ),
        )

    def _diagnose(
        self,
        arm: RetrievalStrategy,
        error: BaseException,
    ) -> tuple[ArmDiagnostic, Optional[BaseException]]:
        """
        Turn an arm's failure into a diagnostic, and decide whether it is fatal.

        Whether a failure degrades is read from the exception's own
        ``degradable`` flag rather than decided here, so two stages cannot
        classify the same failure differently. ``DegradationPolicy.STRICT``
        makes every failure fatal regardless.

        A failure that is not a retrieval error at all is always fatal: an
        unexpected exception type means the arm failed in a way nobody
        anticipated, and continuing would describe that in diagnostics as
        though it had been understood.

        Parameters
        ----------
        arm:
            The arm that failed.
        error:
            The failure.

        Returns
        -------
        tuple[ArmDiagnostic, Optional[BaseException]]
            The diagnostic, and the exception to raise when the failure is
            fatal.
        """
        if not isinstance(error, PTFMRetrievalError):
            return (
                ArmDiagnostic(
                    arm_name=arm.name,
                    method=arm.method,
                    status=ArmStatus.FAILED,
                    error_code=type(error).__name__,
                    message=str(error) or type(error).__name__,
                ),
                error,
            )

        status = (
            ArmStatus.TIMED_OUT
            if "Timeout" in type(error).__name__
            else ArmStatus.FAILED
        )
        diagnostic = ArmDiagnostic.from_exception(
            error, arm_name=arm.name, method=arm.method, status=status
        )
        fatal = (
            None
            if error.degradable and self._config.degrades()
            else error
        )
        return diagnostic, fatal

    async def _filter_currency(
        self,
        hydrated: Sequence[HydratedResult],
        plan: QueryPlan,
        log: StructuredLogger,
    ) -> tuple[tuple[HydratedResult, ...], int, int]:
        """
        Withhold results whose source has a newer version.

        Deliberately NOT wrapped in a degradation branch, unlike every other
        stage here. If currency cannot be determined, the query fails.

        The reason is that this is the one stage whose failure would produce a
        WRONG answer rather than a smaller one. A missing arm yields less
        evidence, which diagnostics can describe and a consumer can weigh.
        Serving a superseded document as current yields evidence that is
        confidently wrong and carries no marker a consumer could detect. The
        supersession filter therefore raises a non-degradable error, and
        configuration cannot disable the filter either.

        The ``unchecked`` count this returns is a different matter, and is not a
        failure: it counts results whose source has no currency record at all,
        which the filter resolves successfully and reports.

        Parameters
        ----------
        hydrated:
            Hydrated results.
        plan:
            The query plan.
        log:
            Logger bound to this query, used to surface an inoperative filter.

        Returns
        -------
        tuple[tuple[HydratedResult, ...], int, int]
            Results to keep, the count withheld as superseded, and the count
            whose source carried no currency record.

        Raises
        ------
        SupersessionError
            If currency could not be determined, whatever the degradation
            policy.
        """
        kept, withheld, unchecked = await self._supersession.filter(
            hydrated, plan.segmentation
        )
        if unchecked:
            log.warning(
                "retrieval.currency_unrecorded",
                unchecked=unchecked,
                kept=len(kept),
            )
        return kept, withheld, unchecked

    async def _expand(
        self,
        anchors: Sequence[HydratedResult],
        plan: QueryPlan,
        log: StructuredLogger,
    ) -> Mapping[str, Sequence[KnowledgeNode]]:
        """
        Gather neighbouring context around the anchors.

        Parameters
        ----------
        anchors:
            The anchors to expand around.
        plan:
            The query plan.
        log:
            Logger bound to this query.

        Returns
        -------
        Mapping[str, Sequence[KnowledgeNode]]
            Context nodes per anchor node identifier. Empty when expansion is
            disabled or failed degradably, which yields anchors without context
            rather than no result.

        Raises
        ------
        PTFMRetrievalError
            If expansion failed non-degradably, or under STRICT.
        """
        if self._expander is None or not anchors:
            return {}

        try:
            return await self._expander.expand(
                tuple(result.node for result in anchors), plan.segmentation
            )
        except PTFMRetrievalError as error:
            if not (error.degradable and self._config.degrades()):
                raise
            log.warning(
                "retrieval.expansion_failed",
                error_code=type(error).__name__,
                anchors=len(anchors),
            )
            return {}

    def _record(self, outcome: RetrievalOutcome) -> None:
        """
        Emit metrics for one completed query.

        Parameters
        ----------
        outcome:
            The completed outcome.
        """
        if not self._config.telemetry.emit_metrics:
            return

        diagnostics = outcome.diagnostics
        self._metrics.increment("ptfm_retrieval_queries_total")
        self._metrics.observe(
            "ptfm_retrieval_latency_ms", diagnostics.latency_ms
        )
        self._metrics.observe(
            "ptfm_retrieval_documents", float(len(outcome.evidence.documents))
        )
        if diagnostics.degraded:
            self._metrics.increment("ptfm_retrieval_degraded_total")
        for arm in diagnostics.arms:
            self._metrics.increment(
                "ptfm_retrieval_arm_total",
                arm=arm.arm_name,
                status=arm.status.value,
            )

    async def ingest(
        self,
        segmentation: SegmentationPartition,
        embedding: Optional[EmbeddingPartition],
        units: Sequence[KnowledgeUnit],
    ) -> int:
        """
        Write knowledge units into the retrieval stores.

        Written in configured batches, each flushed before the next is buffered,
        so that memory does not scale with corpus size. Every batch is a single
        atomic write when the writer provides atomicity, which configuration can
        require.

        Parameters
        ----------
        segmentation:
            The segmentation partition the nodes belong to.
        embedding:
            The embedding partition the vectors belong to, when the units carry
            embeddings.
        units:
            Units to write.

        Returns
        -------
        int
            The number of units written.

        Raises
        ------
        RetrievalConfigurationError
            If this engine has no writer. Refused rather than silently doing
            nothing, because a caller that believed it had ingested a corpus
            would query an empty one.
        IndexWriteError
            If a batch failed.
        """
        if self._writer is None:
            raise RetrievalConfigurationError(
                "This engine is read-only: no index writer was supplied, so it "
                "cannot ingest. Build it with a writer to populate the stores."
            )

        if not units:
            return 0

        batch_size = self._config.ingest.batch_size
        written = 0
        for start in range(0, len(units), batch_size):
            batch = units[start : start + batch_size]
            await self._writer.write_units(segmentation, embedding, batch)
            await self._writer.flush()
            written += len(batch)

        if self._coverage is not None:
            # Coverage moved, so the cached measurement is now wrong.
            self._coverage.invalidate()

        self._logger.info(
            "retrieval.ingested",
            units=written,
            segmentation_key=segmentation.key,
            embedding_key=embedding.key if embedding is not None else None,
            atomic=self._writer.is_atomic,
        )
        return written

    async def health_check(self) -> Mapping[str, bool]:
        """
        Report which arms can currently reach their stores.

        Every arm is checked even after one fails, because the useful answer is
        which arms are available rather than whether all of them are.

        Returns
        -------
        Mapping[str, bool]
            Availability per arm name. An arm whose check raised is reported as
            unavailable rather than propagating, since a health check that
            fails tells a caller less than one that answers.
        """
        checks = await asyncio.gather(
            *(arm.health_check() for arm, _ in self._arms),
            return_exceptions=True,
        )
        return {
            arm.name: (result is True)
            for (arm, _), result in zip(self._arms, checks)
        }

    async def close(self) -> None:
        """
        Release the writer, flushing anything it still holds.

        The stores themselves are not closed here: the engine did not open them
        and does not own their lifetime. Closing a backend the composition root
        owns would break a second engine sharing it.

        Raises
        ------
        IndexWriteError
            If a final flush failed.
        """
        if self._writer is not None:
            await self._writer.close()

    def __repr__(self) -> str:
        """
        Describe the engine's wiring.

        Returns
        -------
        str
            A representation naming the arms and the fusion strategy.
        """
        return (
            f"RetrievalEngine(arms={list(self.arm_names)}, "
            f"fusion={self._fusion.name!r}, ingests={self.ingests})"
        )
