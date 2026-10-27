"""
PTFM Retrieval Configuration

File:
    ptfm/core/retrieval/config.py

Description:
    Declares every operational parameter of the Retrieval Module.

    This module is the SOLE location in which retrieval numeric values are
    defined. No stage, arm, fusion strategy, or backend may contain a numeric
    literal compared against a limit; every such value is injected from here.

Nothing here is a threshold on truth
------------------------------------
    Every value in this module is operational: how many results to fetch, how
    long to wait, how far to expand, how large a batch may be. None of them
    gates what may be believed.

    That distinction is enforced rather than asserted. There is no minimum
    score, no relevance cut-off, and no similarity floor, because any such
    value would be a judgement about evidence made by a component that has no
    business making one. A result that ranks last is still evidence; whether
    it supports a claim is decided by verification, and how much confidence to
    place in it is decided by calibration.

    A test sweeps every field name for the vocabulary of such a threshold.

Nothing here is identity-affecting
----------------------------------
    Unlike Knowledge Processing, where chunking configuration determines node
    identity, no value in this module changes what any artifact IS. Retrieval
    reads a corpus it did not create. Changing a deadline or a fetch depth
    changes which nodes a query returns and never what those nodes contain, so
    no fingerprint is derived from anything here.

    The partition keys that do carry identity are computed from the corpus's
    own recorded properties, in ``models``, not from configuration.

Over-fetch is a correctness concern, not a performance one
----------------------------------------------------------
    Each arm fetches more than the caller asked for. That is not a
    speculative optimization: rank-based fusion needs to see a node's position
    in every arm's ordering, and an arm that stopped at the caller's ``top_k``
    would report a node as absent when it merely ranked lower there. The
    fused ordering would then depend on how deep each arm happened to look.

Environment layering
--------------------
    Every value may be supplied by environment variable using the prefix
    ``PTFM_RETRIEVAL_`` with ``__`` separating nesting levels::

        PTFM_RETRIEVAL_MAX_CONCURRENT_QUERIES=32
        PTFM_RETRIEVAL_DENSE__TOP_K=100
        PTFM_RETRIEVAL_EXPANSION__NEIGHBOUR_RADIUS=2

Dependency contract
-------------------
    Imports only the standard library, Pydantic, pydantic-settings and
    ``ptfm.core.retrieval.exceptions``. It must not import ``interfaces``,
    ``models``, or any stage, arm, or backend.
"""

from __future__ import annotations

import os
from enum import StrEnum
from typing import Final, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from ptfm.core.retrieval.exceptions import RetrievalConfigurationError
from ptfm.core.schemas import DiscoveryMethod

__all__ = [
    "DegradationPolicy",
    "ArmConfig",
    "DenseConfig",
    "SparseConfig",
    "GraphConfig",
    "FusionConfig",
    "ExpansionConfig",
    "SupersessionConfig",
    "CredibilityConfig",
    "PartitionConfig",
    "IngestConfig",
    "TelemetryConfig",
    "RetrievalConfig",
]

#: Fallback worker count when the host CPU count cannot be determined.
_FALLBACK_CPU_COUNT: Final[int] = 4

#: Multiplier applied to a caller's requested result count when deciding how
#: deep each arm should fetch. Declared as a bound rather than a free value
#: because over-fetching without limit turns every query into a scan.
_MAX_OVER_FETCH: Final[int] = 20


class DegradationPolicy(StrEnum):
    """
    How a query responds to an arm that could not contribute.

    The default tolerates a failed arm because the remaining arms still answer
    usefully, and refusing would turn a partial outage into a total one.

    Strictness is available for deployments where a partial answer is worse
    than none. That is a real position rather than a defensive one: a caller
    comparing results over time, or auditing a prior response, needs to know
    that the same arms ran, and a silently narrower result set would be worse
    for them than an explicit failure.
    """

    #: Serve what the remaining arms found, recording the loss in diagnostics.
    DEGRADE = "DEGRADE"
    #: Fail the query if any configured arm did not contribute.
    STRICT = "STRICT"


# ============================================================================
# ARMS
# ============================================================================


class ArmConfig(BaseModel):
    """
    Settings shared by every retrieval arm.

    Attributes
    ----------
    enabled:
        Whether the arm runs. A disabled arm is recorded as skipped rather
        than omitted from diagnostics, so that a narrower result set is
        attributable to configuration rather than mistaken for a failure.
    top_k:
        Candidates the arm fetches.

        Ordinarily larger than the caller's requested result count, so that
        fusion can see where a node ranks in this arm's ordering even when it
        ranks lower here than elsewhere. See the module documentation.
    timeout_seconds:
        Deadline for one search. Bounds the query's latency against the
        slowest arm rather than against the sum of them, since arms run
        concurrently.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", validate_default=True)

    enabled: bool = True
    top_k: int = Field(default=100, ge=1, le=10_000)
    timeout_seconds: float = Field(default=5.0, gt=0.0)


class DenseConfig(ArmConfig):
    """
    Vector search settings.

    Attributes
    ----------
    exact:
        Whether approximate search is forbidden by default.

        Disabled by default because approximate search is what makes vector
        retrieval tractable at corpus scale. Enabled where a result must be
        reproducible: approximate indexes vary with construction order, so an
        audit that must reproduce a prior result cannot rely on them.
    oversample:
        Multiplier applied to :attr:`top_k` when a structural filter is
        present.

        A filtered approximate search examines neighbours and discards those
        failing the filter, so a selective filter can exhaust the candidates
        before the requested count is met. Fetching more neighbours before
        filtering is how an approximate index avoids returning fewer results
        than it could.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", validate_default=True)

    exact: bool = False
    oversample: int = Field(default=4, ge=1, le=64)


class SparseConfig(ArmConfig):
    """
    Lexical search settings.

    Attributes
    ----------
    analyzer:
        Identifier of the text analysis applied to queries and documents.

        Named rather than described, because what an analyzer DOES is the
        backend's knowledge. It is nonetheless configuration rather than a
        constant, since stemming and stop-word choices are properties of a
        corpus's language and register, and a corpus of clinical abbreviations
        wants different handling from one of prose.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", validate_default=True)

    analyzer: str = Field(default="standard", min_length=1)


class GraphConfig(ArmConfig):
    """
    Structural search settings.

    Disabled by default, and that default is deliberate. The graph arm answers
    genuine structural questions, such as which nodes fall under a heading or
    belong to a table, but it contributes little to ordinary relevance
    ranking. Enabling it by default would spend a store round trip on most
    queries for a contribution most queries do not use.

    Attributes
    ----------
    max_traversal_depth:
        Longest relation chain followed.

        Bounded because traversal breadth compounds with depth: a table with
        many rows reached at depth three can return more nodes than the corpus
        has documents.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", validate_default=True)

    enabled: bool = False
    max_traversal_depth: int = Field(default=2, ge=1, le=10)


# ============================================================================
# FUSION
# ============================================================================


class FusionConfig(BaseModel):
    """
    How several arms' orderings become one.

    Attributes
    ----------
    strategy:
        Identifier of the fusion strategy.

        Rank-based by default. Ranks are the only quantity comparable across
        arms: a similarity, a lexical relevance score, and a structural
        ordering occupy different ranges with different meanings, and
        combining them by value would require calibrating them against one
        another, which is a later module's responsibility and not this one's.
    rank_constant:
        Smoothing constant of reciprocal rank fusion.

        Governs how sharply the top of an ordering is favoured. A small value
        makes a single arm's first result dominate; a large one flattens the
        contribution across positions.
    method_weights:
        Per-method multipliers applied to a method's contribution.

        Empty by default, which weights every arm equally. A weight is an
        assertion that one method is more trustworthy than another for a
        corpus, which is a deployment's judgement to make and never a default
        to assume.

        A PARTIAL map: a method absent from it carries a weight of one. That is
        what lets a deployment raise one arm without restating the others, and
        without the map needing revision whenever a new method is added.

        A weight of zero excludes a method from the ranking while leaving its
        observation on the result, which is how an arm can inform a consumer
        without influencing order. Zeroing every method is refused, since a
        successful search would then return an empty ordering.

        Weights scale the RANK contribution, never a score, so a weighted
        deployment remains scale-free.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", validate_default=True)

    strategy: str = Field(default="reciprocal_rank", min_length=1)
    rank_constant: float = Field(default=60.0, gt=0.0)
    method_weights: dict[str, float] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_weights(self) -> FusionConfig:
        """
        Ensure every declared weight is usable.

        Returns
        -------
        FusionConfig
            The validated instance.

        Raises
        ------
        ValueError
            If a weight is negative, or every weight is zero.
        """
        for method, weight in self.method_weights.items():
            if weight < 0.0:
                raise ValueError(
                    f"Fusion weight for {method!r} is negative ({weight}). A "
                    f"negative weight would rank a method's own best results "
                    f"last."
                )

        # A method absent from the map carries a weight of one, so an
        # all-zero map empties the ranking only when it names every method.
        # Treating a partial map as exhaustive would refuse the legitimate and
        # useful case of excluding one arm from the ranking while keeping its
        # observation.
        named = {
            method
            for method in DiscoveryMethod
            if method.value in {key.upper() for key in self.method_weights}
        }
        if (
            self.method_weights
            and named == set(DiscoveryMethod)
            and not any(self.method_weights.values())
        ):
            raise ValueError(
                "Every fusion weight is zero, which would discard every arm's "
                "contribution and return an empty ordering from a successful "
                "search."
            )
        return self


# ============================================================================
# EXPANSION
# ============================================================================


class ExpansionConfig(BaseModel):
    """
    How retrieved results gather surrounding context.

    Expansion serves the case where a claim's support straddles two adjacent
    nodes. It is the reason overlapping nodes are unnecessary, and it must not
    reintroduce their harm: expanded context accompanies the node it was
    expanded from and never becomes an independently retrieved result.

    Attributes
    ----------
    enabled:
        Whether context is gathered at all.
    neighbour_radius:
        Ordinal neighbours gathered on each side of a retrieved node.

        One by default. Expansion inflates the text a consumer must read
        without adding an independent source, so a wider radius costs
        proportionally more for a contribution that falls off quickly with
        distance.
    same_document_only:
        Whether expansion stops at a document boundary.

        Enabled, and effectively a constant: the node adjacent to the last
        node of one document is the first node of an unrelated one, and
        presenting it as context would be false. Exposed rather than hidden so
        that the guarantee is visible where the behaviour is configured.
    max_context_nodes:
        Largest number of context nodes gathered for one query.

        Bounds a query that retrieves many results in one dense region, where
        each result's neighbours are its siblings' neighbours too.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", validate_default=True)

    enabled: bool = True
    neighbour_radius: int = Field(default=1, ge=0, le=10)
    same_document_only: bool = True
    max_context_nodes: int = Field(default=100, ge=0, le=10_000)

    @model_validator(mode="after")
    def _validate_document_boundary(self) -> ExpansionConfig:
        """
        Ensure expansion cannot cross a document boundary.

        Returns
        -------
        ExpansionConfig
            The validated instance.

        Raises
        ------
        ValueError
            If expansion is permitted to cross documents.
        """
        if self.enabled and not self.same_document_only:
            raise ValueError(
                "Expansion must not cross a document boundary. The node "
                "adjacent to a document's last node belongs to an unrelated "
                "document, and presenting it as context would attach text to "
                "a source it did not come from."
            )
        return self


# ============================================================================
# SUPERSESSION AND CREDIBILITY
# ============================================================================


class SupersessionConfig(BaseModel):
    """
    How superseded documents are handled.

    Supersession is a retrieval-time filter, never a deletion. A superseded
    node stays resolvable so that evidence cited before the change remains
    auditable; it simply stops being returned as current evidence.

    Attributes
    ----------
    filter_superseded:
        Whether superseded documents are withheld from results.

        Enabled, and effectively a constant. Serving superseded content as
        current evidence is a truthfulness failure invisible to every
        consumer, so the setting exists to make the guarantee visible rather
        than to offer a choice. Disabling it is rejected.
    include_withdrawn:
        Whether documents whose source was removed are returned.

        Distinct from supersession: a withdrawn source has no replacement, and
        returning its content would present material the publisher retracted.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", validate_default=True)

    filter_superseded: bool = True
    include_withdrawn: bool = False

    @model_validator(mode="after")
    def _validate_filtering(self) -> SupersessionConfig:
        """
        Ensure superseded content cannot be served as current.

        Returns
        -------
        SupersessionConfig
            The validated instance.

        Raises
        ------
        ValueError
            If filtering is disabled.
        """
        if not self.filter_superseded:
            raise ValueError(
                "Superseded documents must not be served as current evidence. "
                "Doing so is a truthfulness failure invisible to every "
                "consumer, and historical resolution by identifier already "
                "keeps prior citations auditable without it."
            )
        return self


class CredibilityConfig(BaseModel):
    """
    How source credibility is obtained.

    Credibility concerns a SOURCE, not a claim. The value is an input to
    retrieval reliability; nothing in this module judges whether a claim is
    true.

    Attributes
    ----------
    enabled:
        Whether credibility is requested at all. Disabled where no plugin
        supplies it, in which case every result reports credibility as
        unavailable rather than as some default.
    timeout_seconds:
        Deadline for one credibility lookup.
    cache_size:
        Distinct sources whose credibility is retained within a process.

        A result set ordinarily spans few distinct sources and a corpus spans
        few relative to its documents, so caching removes most lookups. Zero
        disables caching, which a deployment whose credibility varies over
        time may require.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", validate_default=True)

    enabled: bool = True
    timeout_seconds: float = Field(default=2.0, gt=0.0)
    cache_size: int = Field(default=1_024, ge=0, le=1_000_000)


# ============================================================================
# PARTITIONS AND INGEST
# ============================================================================


class PartitionConfig(BaseModel):
    """
    How compatibility partitions are resolved.

    Attributes
    ----------
    require_active_partition:
        Whether a query fails when no partition is active.

        Enabled. An empty result from an unindexed corpus would be
        indistinguishable from an absence of matching evidence, and a
        verification engine told there is no evidence refuses to answer a
        question the corpus may well support.
    require_embedding_partition:
        Whether a query fails when no embedding partition is active.

        Disabled. A corpus segmented but not yet embedded is a valid
        intermediate state, and lexical and structural retrieval serve it
        perfectly well. Enabled where a deployment's answers depend on
        semantic matching and a lexical-only result would mislead.
    minimum_embedding_coverage:
        Proportion of a partition's nodes that must carry a vector before
        dense retrieval runs.

        NOT a relevance threshold. It describes how completely an index was
        built, not how well anything matched: an unembedded node is invisible
        to dense search however well it would have matched. Zero permits dense
        retrieval over a partially built partition, with the incompleteness
        recorded in diagnostics.
    
    Query encoding is not configurable
    ----------------------------------
        Compatibility between a query's encoding and the vectors it searches
        is always verified, and no setting disables it.

        A mismatch is invisible by nature: several widely used retrieval
        models are trained asymmetrically, and embedding a query as though it
        were a passage returns plausible results in a worse order while
        nothing anywhere reports a fault. A setting that could switch the
        check off would exist only to hide a failure the operator cannot
        otherwise see, so it is absent rather than merely defaulted on.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", validate_default=True)

    require_active_partition: bool = True
    require_embedding_partition: bool = False
    minimum_embedding_coverage: float = Field(default=0.0, ge=0.0, le=1.0)


class IngestConfig(BaseModel):
    """
    How knowledge units are written into the indexes.

    Attributes
    ----------
    batch_size:
        Units written in one operation.
    require_atomic_writer:
        Whether a writer that cannot write a unit atomically is rejected.

        Enabled. A unit touches several stores, and a partial write leaves a
        node retrievable with no authoritative text: a result that appears
        valid and cannot be traced. A deployment accepting the weaker
        guarantee must say so, rather than discovering it from an untraceable
        result later.
    write_timeout_seconds:
        Deadline for one write.
    max_retries:
        Attempts permitted for a transient write failure, inclusive of the
        first.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", validate_default=True)

    batch_size: int = Field(default=100, ge=1, le=10_000)
    require_atomic_writer: bool = True
    write_timeout_seconds: float = Field(default=60.0, gt=0.0)
    max_retries: int = Field(default=3, ge=1, le=10)


class TelemetryConfig(BaseModel):
    """
    Observability behaviour.

    Attributes
    ----------
    emit_metrics:
        Whether operational counters and observations are emitted.
    log_level:
        Minimum severity emitted by the module logger.
    log_query_text:
        Whether query text is logged.

        Disabled. A query is user input and may carry personal or
        confidential material, so recording it by default would make a
        logging system a copy of everything anyone asked.
    log_result_ids:
        Whether retrieved node identifiers are logged. Useful for diagnosing
        ranking, and voluminous on a busy service.

    Raises
    ------
    ValueError
        If the log level is not recognized.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", validate_default=True)

    emit_metrics: bool = True
    log_level: str = "INFO"
    log_query_text: bool = False
    log_result_ids: bool = False

    @model_validator(mode="after")
    def _validate_log_level(self) -> TelemetryConfig:
        """
        Ensure the log level is a recognized severity name.

        Returns
        -------
        TelemetryConfig
            The validated instance.

        Raises
        ------
        ValueError
            If the level is unknown.
        """
        permitted = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if self.log_level.upper() not in permitted:
            raise ValueError(
                f"log_level {self.log_level!r} is not one of {sorted(permitted)}"
            )
        return self


# ============================================================================
# ROOT CONFIGURATION
# ============================================================================


class RetrievalConfig(BaseSettings):
    """
    Complete operational configuration of the Retrieval Module.

    Constructed once at the composition root and injected into the engine and
    every stage. Nothing constructs its own configuration, and no component
    reads environment variables directly.

    Attributes
    ----------
    default_top_k:
        Results returned when a caller names no count.
    max_top_k:
        Largest result count a caller may request. Bounds the work one query
        can demand, and with it the memory a result set occupies.
    query_timeout_seconds:
        Deadline for a complete query, including every arm, fusion, hydration,
        and expansion.
    max_concurrent_queries:
        Queries served concurrently. The primary control over both throughput
        and peak memory.
    store_thread_limit:
        Worker threads available for blocking store operations. ``None``
        derives a value from the host CPU count.
    degradation_policy:
        How a query responds to an arm that could not contribute.
    dense:
        Vector search settings.
    sparse:
        Lexical search settings.
    graph:
        Structural search settings.
    fusion:
        How arms' orderings are combined.
    expansion:
        How results gather surrounding context.
    supersession:
        How superseded documents are handled.
    credibility:
        How source credibility is obtained.
    partitions:
        How compatibility partitions are resolved.
    ingest:
        How knowledge units are written.
    telemetry:
        Observability.

    Raises
    ------
    ValueError
        If the default result count exceeds the maximum, if no arm is enabled,
        or if an arm fetches fewer candidates than a caller may request.
    """

    model_config = SettingsConfigDict(
        env_prefix="PTFM_RETRIEVAL_",
        env_nested_delimiter="__",
        env_file=".env",
        env_file_encoding="utf-8",
        frozen=True,
        extra="forbid",
        validate_default=True,
    )

    default_top_k: int = Field(default=10, ge=1, le=1_000)
    max_top_k: int = Field(default=100, ge=1, le=1_000)
    query_timeout_seconds: float = Field(default=30.0, gt=0.0)
    max_concurrent_queries: int = Field(default=32, ge=1, le=4_096)
    store_thread_limit: Optional[int] = Field(default=None, ge=1, le=1_024)
    degradation_policy: DegradationPolicy = DegradationPolicy.DEGRADE

    dense: DenseConfig = Field(default_factory=DenseConfig)
    sparse: SparseConfig = Field(default_factory=SparseConfig)
    graph: GraphConfig = Field(default_factory=GraphConfig)
    fusion: FusionConfig = Field(default_factory=FusionConfig)
    expansion: ExpansionConfig = Field(default_factory=ExpansionConfig)
    supersession: SupersessionConfig = Field(default_factory=SupersessionConfig)
    credibility: CredibilityConfig = Field(default_factory=CredibilityConfig)
    partitions: PartitionConfig = Field(default_factory=PartitionConfig)
    ingest: IngestConfig = Field(default_factory=IngestConfig)
    telemetry: TelemetryConfig = Field(default_factory=TelemetryConfig)

    @model_validator(mode="after")
    def _validate_result_bounds(self) -> RetrievalConfig:
        """
        Ensure the default result count is requestable.

        Returns
        -------
        RetrievalConfig
            The validated instance.

        Raises
        ------
        ValueError
            If the default exceeds the maximum.
        """
        if self.default_top_k > self.max_top_k:
            raise ValueError(
                f"default_top_k ({self.default_top_k}) exceeds max_top_k "
                f"({self.max_top_k}), so the default request would be "
                f"rejected."
            )
        return self

    @model_validator(mode="after")
    def _validate_arms_enabled(self) -> RetrievalConfig:
        """
        Ensure at least one arm can run.

        Returns
        -------
        RetrievalConfig
            The validated instance.

        Raises
        ------
        ValueError
            If every arm is disabled.
        """
        if not any(
            (self.dense.enabled, self.sparse.enabled, self.graph.enabled)
        ):
            raise ValueError(
                "Every retrieval arm is disabled, so no query could return "
                "anything. An empty result from a configuration fault is "
                "indistinguishable from an absence of matching evidence."
            )
        return self

    @model_validator(mode="after")
    def _validate_fetch_depth(self) -> RetrievalConfig:
        """
        Ensure each enabled arm fetches at least what a caller may request.

        An arm fetching fewer candidates than the caller asked for cannot
        contribute to the tail of the result set, so the fused ordering would
        silently depend on that arm's own limit rather than on relevance.

        Returns
        -------
        RetrievalConfig
            The validated instance.

        Raises
        ------
        ValueError
            If an enabled arm fetches fewer candidates than ``max_top_k``.
        """
        for name, arm in (
            ("dense", self.dense),
            ("sparse", self.sparse),
            ("graph", self.graph),
        ):
            if not arm.enabled:
                continue
            if arm.top_k < self.max_top_k:
                raise ValueError(
                    f"Arm {name!r} fetches {arm.top_k} candidates while a "
                    f"caller may request {self.max_top_k}. An arm that stops "
                    f"short cannot contribute to the tail of a result set, so "
                    f"the fused ordering would depend on that arm's limit "
                    f"rather than on relevance."
                )
        return self

    # ------------------------------------------------------------------
    # Derived settings
    # ------------------------------------------------------------------

    def resolve_top_k(self, requested: Optional[int] = None) -> int:
        """
        Determine how many results to return for a request.

        Parameters
        ----------
        requested:
            The caller's requested count, or ``None`` for the default.

        Returns
        -------
        int
            The result count to serve.

        Raises
        ------
        RetrievalConfigurationError
            If the request exceeds the configured maximum. Rejected rather
            than silently reduced, because a caller receiving fewer results
            than it asked for would have no way to tell whether the corpus or
            the configuration was responsible.
        """
        if requested is None:
            return self.default_top_k

        if requested < 1:
            raise RetrievalConfigurationError(
                f"Requested result count must be at least one, got "
                f"{requested}.",
                context={"requested": requested},
            )

        if requested > self.max_top_k:
            raise RetrievalConfigurationError(
                f"Requested {requested} results, exceeding the configured "
                f"maximum of {self.max_top_k}.",
                context={"requested": requested, "maximum": self.max_top_k},
            )

        return requested

    def resolve_fetch_depth(
        self,
        arm: ArmConfig,
        top_k: int,
        *,
        filtered: bool = False,
    ) -> int:
        """
        Determine how many candidates an arm should fetch.

        At least the caller's requested count, and ordinarily more, so that
        fusion can see where a node ranks in this arm's ordering even when it
        ranks lower here than elsewhere.

        A filtered search on an approximate index fetches further still,
        because such a search examines neighbours and discards those failing
        the filter, so a selective filter can exhaust the candidates before
        the requested count is met.

        Parameters
        ----------
        arm:
            The arm's configuration.
        top_k:
            The caller's requested result count.
        filtered:
            Whether a structural filter applies.

        Returns
        -------
        int
            Candidates to fetch, bounded so that over-fetching cannot turn a
            query into a scan.
        """
        depth = max(arm.top_k, top_k)

        if filtered and isinstance(arm, DenseConfig):
            depth *= arm.oversample

        return min(depth, top_k * _MAX_OVER_FETCH)

    def degrades(self) -> bool:
        """
        Report whether a degradable failure should reduce a result rather than
        fail the query.

        Exposed as a predicate rather than left to each stage comparing the
        policy enum, so that the meaning of the policy is defined once. A
        stage writing its own comparison could interpret a future third policy
        differently from its neighbours.

        Returns
        -------
        bool
            ``True`` when degradable failures are tolerated.
        """
        return self.degradation_policy is DegradationPolicy.DEGRADE

    def resolve_thread_limit(self) -> int:
        """
        Determine the worker thread count for blocking store operations.

        Returns the configured value when set, and otherwise derives one from
        the host CPU count.

        Returns
        -------
        int
            Thread limit of at least one.
        """
        if self.store_thread_limit is not None:
            return self.store_thread_limit
        return max(1, os.cpu_count() or _FALLBACK_CPU_COUNT)

    def enabled_arms(self) -> tuple[str, ...]:
        """
        Report which arms will run, in a stable order.

        Returns
        -------
        tuple[str, ...]
            Names of the enabled arms.
        """
        return tuple(
            name
            for name, arm in (
                ("dense", self.dense),
                ("sparse", self.sparse),
                ("graph", self.graph),
            )
            if arm.enabled
        )

    def arm_config(self, name: str) -> ArmConfig:
        """
        Retrieve one arm's configuration by name.

        Parameters
        ----------
        name:
            The arm's name.

        Returns
        -------
        ArmConfig
            That arm's configuration.

        Raises
        ------
        RetrievalConfigurationError
            If no such arm is configured.
        """
        arms: dict[str, ArmConfig] = {
            "dense": self.dense,
            "sparse": self.sparse,
            "graph": self.graph,
        }
        arm = arms.get(name)
        if arm is None:
            raise RetrievalConfigurationError(
                f"No configuration for arm {name!r}. Configured arms: "
                f"{', '.join(sorted(arms))}.",
                context={"arm_name": name},
            )
        return arm

    def with_overrides(self, **overrides: object) -> RetrievalConfig:
        """
        Produce a revalidated copy carrying the supplied changes.

        Uses full reconstruction rather than a shallow copy so that
        cross-field validators run against the merged result. A copy that
        skipped validation could hold a combination this class explicitly
        rejects.

        Parameters
        ----------
        **overrides:
            Top-level field values to replace.

        Returns
        -------
        RetrievalConfig
            The revalidated configuration.

        Raises
        ------
        RetrievalConfigurationError
            If the merged configuration is invalid.
        """
        merged = {**self.model_dump(), **overrides}
        try:
            return RetrievalConfig(**merged)  # type: ignore[arg-type]
        except ValueError as exc:
            raise RetrievalConfigurationError(
                f"Invalid retrieval configuration override: {exc}"
            ) from exc
