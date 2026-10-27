"""
PTFM Retrieval Exception Hierarchy

File:
    ptfm/core/retrieval/exceptions.py

Description:
    Defines every exception raised by the Retrieval Module, rooted in
    ``PTFMError``. The hierarchy mirrors the query pipeline so that a failure
    can be attributed to a stage without string parsing.

Pipeline stage mapping
----------------------

    Configuration   -> RetrievalConfigurationError
    Partition       -> PartitionError
    Encoding        -> QueryEncodingError
    Search          -> SearchError
    Fusion          -> FusionError
    Supersession    -> SupersessionError
    Hydration       -> HydrationError
    Expansion       -> ExpansionError
    Credibility     -> CredibilityError
    Assembly        -> EvidenceAssemblyError
    Ingest          -> IndexWriteError

Degradation, not silence
------------------------
    Retrieval is the one PTFM module that is expected to produce a useful
    answer while a component is unavailable. A single failed arm is a
    degraded result, not a failed query.

    That makes the distinction between a degraded condition and a failed one
    the central concern of this hierarchy, and it is drawn deliberately:

        An arm failing is DEGRADATION. Other arms serve, and the result
        records that the arm was unavailable. The exception is caught by the
        engine and converted into a diagnostic.

        Every arm failing is FAILURE. Returning an empty result would be
        indistinguishable from "no matching evidence", and a verification
        engine told there is no evidence will refuse to answer a question it
        could have answered. A system outage must never be presentable as an
        absence of evidence.

        The node store failing is FAILURE, always. Authoritative text and
        provenance come from it alone, so without it there are no evidence
        items, only identifiers.

        Compatibility failing is FAILURE, always. Serving results from an
        incompatible partition would return vectors that cannot be compared
        or spans that do not mean what they say.

Nothing here judges truth
-------------------------
    No exception in this module concerns confidence, entailment, calibration,
    or credibility of a claim. ``CredibilityError`` reports that a plugin
    could not supply a SOURCE credibility value; it is a plumbing failure,
    and the value it failed to obtain is an input to retrieval reliability
    rather than a judgement about a claim.

Dependency contract
-------------------
    Imports only the standard library and ``ptfm.utils.exceptions``. No
    schema, model, or third-party type may leak into a caller's ``except``
    clause.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any, ClassVar, Mapping, Sequence

from ptfm.utils.exceptions import PTFMError

__all__ = [
    "RetrievalStage",
    "PTFMRetrievalError",
    # Configuration
    "RetrievalConfigurationError",
    "StrategyRegistrationError",
    "StoreRegistrationError",
    "BackendUnavailableError",
    # Partition and compatibility
    "PartitionError",
    "PartitionNotFoundError",
    "SegmentationMismatchError",
    "EmbeddingSpaceMismatchError",
    "QueryEncodingMismatchError",
    "AmbiguousPartitionError",
    # Query encoding
    "QueryEncodingError",
    "QueryTooLargeError",
    "EmptyQueryError",
    # Search
    "SearchError",
    "ArmUnavailableError",
    "ArmTimeoutError",
    "AllArmsFailedError",
    "StoreUnavailableError",
    # Fusion
    "FusionError",
    "IncomparableSignalsError",
    # Supersession
    "SupersessionError",
    # Hydration
    "HydrationError",
    "NodeNotFoundError",
    "ProvenanceIncompleteError",
    # Expansion
    "ExpansionError",
    "AnchorViolationError",
    # Credibility
    "CredibilityError",
    "CredibilityUnavailableError",
    # Assembly
    "EvidenceAssemblyError",
    "CorroborationIntegrityError",
    # Ingest
    "IndexWriteError",
    "PartialIndexWriteError",
]


class RetrievalStage(StrEnum):
    """
    Pipeline stage in which a retrieval failure originated.

    Used for structured logging, degradation diagnostics, and per-stage
    metrics. Declared here rather than in ``models`` because the exception
    module sits at the base of the module dependency graph.
    """

    UNSPECIFIED = "UNSPECIFIED"
    CONFIGURATION = "CONFIGURATION"
    PARTITION = "PARTITION"
    ENCODING = "ENCODING"
    SEARCH = "SEARCH"
    FUSION = "FUSION"
    SUPERSESSION = "SUPERSESSION"
    HYDRATION = "HYDRATION"
    EXPANSION = "EXPANSION"
    CREDIBILITY = "CREDIBILITY"
    ASSEMBLY = "ASSEMBLY"
    INGEST = "INGEST"


# ============================================================================
# BASE
# ============================================================================


class PTFMRetrievalError(PTFMError):
    """
    Base class for all Retrieval Module failures.

    Attributes
    ----------
    stage:
        Pipeline stage that produced the failure. Declared as a class constant
        by each subtree, so that stage attribution cannot be forged by a raise
        site.
    degradable:
        Whether the engine may continue with a reduced result rather than
        failing the query. Declared as a class constant for the same reason.

        Distinct from ``retryable``. A retryable failure may succeed if
        attempted again; a degradable one need not be attempted again at all,
        because the query can be answered without whatever it produced.

    Parameters
    ----------
    message:
        Human-readable description of the failure.
    query_id:
        Identifier of the query being served, when known.
    node_id:
        Identifier of the knowledge node involved, when the failure concerns
        one.
    context:
        Additional structured detail.
    """

    stage: ClassVar[RetrievalStage] = RetrievalStage.UNSPECIFIED
    degradable: ClassVar[bool] = False

    def __init__(
        self,
        message: str,
        *,
        query_id: str | None = None,
        node_id: str | None = None,
        context: Mapping[str, Any] | None = None,
    ) -> None:
        merged: dict[str, Any] = dict(context) if context else {}
        merged.setdefault("stage", str(self.stage))
        merged.setdefault("degradable", self.degradable)
        if query_id is not None:
            merged.setdefault("query_id", query_id)
        if node_id is not None:
            merged.setdefault("node_id", node_id)
        super().__init__(message, context=merged)
        self.query_id: str | None = query_id
        self.node_id: str | None = node_id

    def to_dict(self) -> dict[str, Any]:
        """
        Render the failure as a structured mapping including stage detail.

        Returns
        -------
        dict[str, Any]
            The base payload extended with ``stage``, ``degradable``,
            ``query_id`` and ``node_id``.
        """
        payload = super().to_dict()
        payload["stage"] = str(self.stage)
        payload["degradable"] = self.degradable
        payload["query_id"] = self.query_id
        payload["node_id"] = self.node_id
        return payload


# ============================================================================
# CONFIGURATION
# ============================================================================


class RetrievalConfigurationError(PTFMRetrievalError):
    """
    Invalid or inconsistent retrieval configuration.

    Run-level and never degradable: a misconfiguration would affect every
    query identically, so serving a reduced result would misrepresent a
    permanent condition as a transient one.
    """

    stage: ClassVar[RetrievalStage] = RetrievalStage.CONFIGURATION


class StrategyRegistrationError(RetrievalConfigurationError):
    """
    A retrieval strategy could not be registered or resolved.

    Parameters
    ----------
    message:
        Description of the failure.
    strategy_name:
        The strategy involved, when known.
    """

    def __init__(
        self,
        message: str,
        *,
        strategy_name: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(message, **kwargs)
        self.strategy_name: str | None = strategy_name
        if strategy_name is not None:
            self.add_context(strategy_name=strategy_name)


class StoreRegistrationError(RetrievalConfigurationError):
    """
    A store backend could not be registered or resolved.

    Parameters
    ----------
    message:
        Description of the failure.
    store_name:
        The store involved, when known.
    """

    def __init__(
        self,
        message: str,
        *,
        store_name: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(message, **kwargs)
        self.store_name: str | None = store_name
        if store_name is not None:
            self.add_context(store_name=store_name)


class BackendUnavailableError(RetrievalConfigurationError):
    """
    A backend is configured but its optional dependency is not installed.

    Raised at startup rather than on the first query, so that a missing
    packaging extra produces an actionable message instead of an
    ``ImportError`` surfacing from an unrelated module mid-service.

    Parameters
    ----------
    component:
        Identifier of the unavailable component.
    missing_dependency:
        Distribution name of the absent package.
    install_extra:
        Name of the packaging extra that provides it.
    """

    def __init__(
        self,
        *,
        component: str,
        missing_dependency: str,
        install_extra: str,
        **kwargs: Any,
    ) -> None:
        message = (
            f"Component {component!r} is unavailable: required dependency "
            f"{missing_dependency!r} is not installed. Install it with the "
            f"{install_extra!r} extra."
        )
        super().__init__(message, **kwargs)
        self.component: str = component
        self.missing_dependency: str = missing_dependency
        self.install_extra: str = install_extra
        self.add_context(
            component=component,
            missing_dependency=missing_dependency,
            install_extra=install_extra,
        )


# ============================================================================
# PARTITION AND COMPATIBILITY
# ============================================================================


class PartitionError(PTFMRetrievalError):
    """
    Failure resolving or validating a compatibility partition.

    Never degradable. A partition defines which artifacts may be compared
    with which, so proceeding despite a partition failure means comparing
    things that are not comparable, which produces results that are
    arithmetically valid and meaningless.
    """

    stage: ClassVar[RetrievalStage] = RetrievalStage.PARTITION


class PartitionNotFoundError(PartitionError):
    """
    No partition exists for the requested compatibility key.

    Ordinarily means a corpus has not been indexed under the current
    configuration, which is a deployment state rather than a query fault.

    Parameters
    ----------
    message:
        Description of the failure.
    partition_key:
        The key that could not be resolved.
    """

    def __init__(
        self,
        message: str,
        *,
        partition_key: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(message, **kwargs)
        self.partition_key: str | None = partition_key
        if partition_key is not None:
            self.add_context(partition_key=partition_key)


class SegmentationMismatchError(PartitionError):
    """
    A segmentation fingerprint does not match the partition it addresses.

    Node identifiers are derived in part from the segmentation fingerprint, so
    identifiers produced under differing segmentation occupy disjoint spaces.
    An identifier from one space either fails to resolve in another or, worse,
    resolves to unrelated content.

    Parameters
    ----------
    expected:
        Fingerprint the partition was built under.
    actual:
        Fingerprint presented.
    """

    def __init__(
        self,
        *,
        expected: str,
        actual: str,
        **kwargs: Any,
    ) -> None:
        message = (
            f"Segmentation fingerprint mismatch: the partition was built under "
            f"{expected!r} but {actual!r} was presented. Identifiers derived "
            f"under differing segmentation occupy disjoint spaces, so results "
            f"would be unresolvable or would resolve to unrelated content."
        )
        super().__init__(message, **kwargs)
        self.expected: str = expected
        self.actual: str = actual
        self.add_context(expected=expected, actual=actual)


class EmbeddingSpaceMismatchError(PartitionError):
    """
    Vectors from differing embedding spaces would be compared.

    A similarity computed across two models, dimensions, normalizations, or
    passage compositions is arithmetically valid and semantically
    meaningless. Nothing downstream reports an error: results are returned
    and ranked, merely wrongly.

    Parameters
    ----------
    message:
        Description of the mismatch.
    expected_identity:
        Embedding identity the partition holds.
    actual_identity:
        Embedding identity presented.
    """

    def __init__(
        self,
        message: str,
        *,
        expected_identity: str | None = None,
        actual_identity: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(message, **kwargs)
        self.expected_identity: str | None = expected_identity
        self.actual_identity: str | None = actual_identity
        detail: dict[str, Any] = {}
        if expected_identity is not None:
            detail["expected_identity"] = expected_identity
        if actual_identity is not None:
            detail["actual_identity"] = actual_identity
        if detail:
            self.add_context(**detail)


class QueryEncodingMismatchError(PartitionError):
    """
    A query would be encoded incompatibly with the passages it searches.

    Several widely used retrieval models are trained asymmetrically, expecting
    a distinct transformation on each side. Embedding a query as though it
    were a passage degrades retrieval measurably and produces no error at all,
    which is why the condition is detected here rather than left to surface as
    poor results.

    Parameters
    ----------
    expected:
        Query convention the partition's vectors were produced under.
    actual:
        Query convention the configured provider applies.
    """

    def __init__(
        self,
        *,
        expected: str,
        actual: str,
        **kwargs: Any,
    ) -> None:
        message = (
            f"Query encoding mismatch: the partition's vectors expect "
            f"{expected!r} but the configured provider applies {actual!r}. "
            f"Encoding a query incompatibly with its passages degrades "
            f"retrieval without producing any visible error."
        )
        super().__init__(message, **kwargs)
        self.expected: str = expected
        self.actual: str = actual
        self.add_context(expected=expected, actual=actual)


class AmbiguousPartitionError(PartitionError):
    """
    Several segmentation partitions could serve a query.

    Exactly one segmentation partition may serve queries at a time. Blending
    two would mix identifier spaces and produce a result set whose members
    cannot be compared, so the ambiguity is refused rather than resolved by
    an arbitrary choice.

    Several EMBEDDING partitions over one segmentation are not ambiguous and
    do not raise this: one is active and the others are being built or
    retired.

    Parameters
    ----------
    candidates:
        The partitions that could have served the query.
    """

    def __init__(
        self,
        *,
        candidates: Sequence[str],
        **kwargs: Any,
    ) -> None:
        message = (
            f"Several segmentation partitions could serve this query: "
            f"{', '.join(candidates)}. Exactly one may be active, because "
            f"blending identifier spaces produces results whose members "
            f"cannot be compared."
        )
        super().__init__(message, **kwargs)
        self.candidates: tuple[str, ...] = tuple(candidates)
        self.add_context(candidates=list(candidates))


# ============================================================================
# QUERY ENCODING
# ============================================================================


class QueryEncodingError(PTFMRetrievalError):
    """
    Failure converting a query into a vector.

    Degradable. Dense retrieval becomes unavailable while sparse and
    structural retrieval continue, which is a materially reduced but still
    useful result.
    """

    stage: ClassVar[RetrievalStage] = RetrievalStage.ENCODING
    degradable: ClassVar[bool] = True


class QueryTooLargeError(QueryEncodingError):
    """
    A query exceeds what the embedding provider accepts.

    Not degradable despite its parent: a query too long for the model is a
    property of the request rather than of the system, and it will fail
    identically on every attempt and every arm boundary. Reporting it lets a
    caller shorten the query, whereas degrading would silently answer a
    different question from the one asked.

    Parameters
    ----------
    query_chars:
        Length of the query.
    max_chars:
        Provider maximum.
    """

    degradable: ClassVar[bool] = False

    def __init__(
        self,
        *,
        query_chars: int,
        max_chars: int,
        **kwargs: Any,
    ) -> None:
        message = (
            f"Query of {query_chars} characters exceeds the provider maximum "
            f"of {max_chars}."
        )
        super().__init__(message, **kwargs)
        self.query_chars: int = query_chars
        self.max_chars: int = max_chars
        self.add_context(query_chars=query_chars, max_chars=max_chars)


class EmptyQueryError(QueryEncodingError):
    """
    A query carries no searchable content.

    Not degradable, for the same reason as an oversized query: it is a
    property of the request. Returning results for an empty query would
    return arbitrary content presented as relevant.
    """

    degradable: ClassVar[bool] = False


# ============================================================================
# SEARCH
# ============================================================================


class SearchError(PTFMRetrievalError):
    """
    Failure executing a retrieval arm.

    Degradable by default. The arms are independent, so one failing leaves
    the others able to serve, and the result records which arm was
    unavailable rather than presenting a reduced set as complete.
    """

    stage: ClassVar[RetrievalStage] = RetrievalStage.SEARCH
    degradable: ClassVar[bool] = True


class ArmUnavailableError(SearchError):
    """
    A retrieval arm could not execute.

    Parameters
    ----------
    message:
        Description of the failure.
    arm_name:
        Identifier of the arm that failed.
    """

    retryable: ClassVar[bool] = True

    def __init__(
        self,
        message: str,
        *,
        arm_name: str,
        **kwargs: Any,
    ) -> None:
        super().__init__(message, **kwargs)
        self.arm_name: str = arm_name
        self.add_context(arm_name=arm_name)


class ArmTimeoutError(SearchError):
    """
    A retrieval arm exceeded its deadline.

    Parameters
    ----------
    arm_name:
        Identifier of the arm that timed out.
    timeout_seconds:
        The deadline that was exceeded.
    """

    retryable: ClassVar[bool] = True

    def __init__(
        self,
        *,
        arm_name: str,
        timeout_seconds: float,
        **kwargs: Any,
    ) -> None:
        message = (
            f"Retrieval arm {arm_name!r} exceeded its {timeout_seconds} second "
            f"deadline."
        )
        super().__init__(message, **kwargs)
        self.arm_name: str = arm_name
        self.timeout_seconds: float = timeout_seconds
        self.add_context(arm_name=arm_name, timeout_seconds=timeout_seconds)


class AllArmsFailedError(SearchError):
    """
    Every retrieval arm failed.

    NOT degradable, and this is the most consequential distinction in the
    module. An empty result would be indistinguishable from "no matching
    evidence exists", and a verification engine told there is no evidence
    will refuse to answer a question it could have answered. A system outage
    must never be presentable as an absence of evidence.

    Parameters
    ----------
    failures:
        Description of each arm's failure, keyed by arm name, so that a
        diagnosis does not require correlating separate log entries.
    """

    degradable: ClassVar[bool] = False
    retryable: ClassVar[bool] = True

    def __init__(
        self,
        *,
        failures: Mapping[str, str],
        **kwargs: Any,
    ) -> None:
        rendered = "; ".join(f"{arm}: {reason}" for arm, reason in failures.items())
        message = (
            f"Every retrieval arm failed ({rendered}). An empty result is not "
            f"returned, because it would be indistinguishable from an absence "
            f"of evidence and would cause a refusal to answer a question the "
            f"corpus may well support."
        )
        super().__init__(message, **kwargs)
        self.failures: Mapping[str, str] = dict(failures)
        self.add_context(failed_arms=list(failures))


class StoreUnavailableError(SearchError):
    """
    A backing store could not be reached.

    Degradable only where the store serves a single arm. The node store is
    not such a store, and its unavailability surfaces as a hydration failure
    rather than as this.

    Parameters
    ----------
    message:
        Description of the failure.
    store_name:
        Identifier of the unreachable store.
    """

    retryable: ClassVar[bool] = True

    def __init__(
        self,
        message: str,
        *,
        store_name: str,
        **kwargs: Any,
    ) -> None:
        super().__init__(message, **kwargs)
        self.store_name: str = store_name
        self.add_context(store_name=store_name)


# ============================================================================
# FUSION
# ============================================================================


class FusionError(PTFMRetrievalError):
    """
    Failure combining the results of several arms.

    Not degradable: fusion is what turns several partial orderings into one,
    so a failure means there is no ranking to serve.
    """

    stage: ClassVar[RetrievalStage] = RetrievalStage.FUSION


class IncomparableSignalsError(FusionError):
    """
    A fusion strategy was asked to combine signals on incomparable scales.

    Raised by score-combining strategies. Rank-based fusion cannot encounter
    this, because it discards scores and uses ordinal position alone, which is
    the reason rank-based fusion is the default.

    Averaging a cosine similarity with a lexical relevance score yields a
    number that resembles a measurement and is not one.

    Parameters
    ----------
    message:
        Description of the failure.
    scales:
        The scales that would have been combined.
    """

    def __init__(
        self,
        message: str,
        *,
        scales: Sequence[str] = (),
        **kwargs: Any,
    ) -> None:
        super().__init__(message, **kwargs)
        self.scales: tuple[str, ...] = tuple(scales)
        if scales:
            self.add_context(scales=list(scales))


# ============================================================================
# SUPERSESSION
# ============================================================================


class SupersessionError(PTFMRetrievalError):
    """
    Failure determining which documents are current.

    Not degradable. Proceeding without the filter would serve superseded
    content as current evidence, which is a truthfulness failure rather than
    a completeness one, and is invisible to every consumer.
    """

    stage: ClassVar[RetrievalStage] = RetrievalStage.SUPERSESSION


# ============================================================================
# HYDRATION
# ============================================================================


class HydrationError(PTFMRetrievalError):
    """
    Failure obtaining authoritative content and provenance for a result.

    Never degradable. Retrieval arms return identifiers and signals, so
    without the node store there is no text, no span, and no provenance:
    there are candidate identifiers and nothing that could be called
    evidence.
    """

    stage: ClassVar[RetrievalStage] = RetrievalStage.HYDRATION


class NodeNotFoundError(HydrationError):
    """
    An index returned an identifier the node store does not hold.

    Indicates that an index and the node store have diverged, ordinarily
    because a write was applied to one and not the other. The condition is
    surfaced rather than skipped: silently dropping the result would present
    an inconsistent store as a merely less relevant corpus.

    Parameters
    ----------
    message:
        Description of the failure.
    missing_node_ids:
        Identifiers that could not be resolved.
    """

    def __init__(
        self,
        message: str,
        *,
        missing_node_ids: Sequence[str] = (),
        **kwargs: Any,
    ) -> None:
        super().__init__(message, **kwargs)
        self.missing_node_ids: tuple[str, ...] = tuple(missing_node_ids)
        if missing_node_ids:
            self.add_context(missing_node_ids=list(missing_node_ids))


class ProvenanceIncompleteError(HydrationError):
    """
    A hydrated node lacks provenance required to become evidence.

    The final guard on the Evidence-First chain within retrieval. An evidence
    item without a span, a document identifier, or a source cannot be traced
    to a region of an identified document, and admitting one would let a
    verified claim cite something unlocatable.

    Parameters
    ----------
    message:
        Description of the violation.
    missing_fields:
        Provenance fields that were absent.
    """

    def __init__(
        self,
        message: str,
        *,
        missing_fields: Sequence[str] = (),
        **kwargs: Any,
    ) -> None:
        super().__init__(message, **kwargs)
        self.missing_fields: tuple[str, ...] = tuple(missing_fields)
        if missing_fields:
            self.add_context(missing_fields=list(missing_fields))


# ============================================================================
# EXPANSION
# ============================================================================


class ExpansionError(PTFMRetrievalError):
    """
    Failure adding structural context to a retrieved result.

    Degradable. Expansion adds context to results that are already complete
    without it, so a failure yields narrower evidence rather than none.
    """

    stage: ClassVar[RetrievalStage] = RetrievalStage.EXPANSION
    degradable: ClassVar[bool] = True


class AnchorViolationError(ExpansionError):
    """
    An expanded node would be treated as an independently retrieved one.

    NOT degradable, despite its parent. Corroboration is counted by the number
    of distinct anchors, so an expanded node admitted as an anchor makes one
    finding appear as two. That is the precise harm that removing overlapping
    chunks from Knowledge Processing was intended to prevent, and permitting
    it here would reintroduce it by another route.

    A verification engine consuming inflated corroboration produces a
    confidence that is wrong in the direction of overconfidence, which is the
    worst available failure for this system.

    Parameters
    ----------
    message:
        Description of the violation.
    node_id:
        The node that would have been mis-anchored.
    anchor_node_id:
        The anchor it should have accompanied.
    """

    degradable: ClassVar[bool] = False

    def __init__(
        self,
        message: str,
        *,
        anchor_node_id: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(message, **kwargs)
        self.anchor_node_id: str | None = anchor_node_id
        if anchor_node_id is not None:
            self.add_context(anchor_node_id=anchor_node_id)


# ============================================================================
# CREDIBILITY
# ============================================================================


class CredibilityError(PTFMRetrievalError):
    """
    Failure obtaining a source credibility value from a domain plugin.

    Concerns the credibility of a SOURCE, which is an input to retrieval
    reliability. It is not a judgement about a claim, and nothing in this
    module produces one.
    """

    stage: ClassVar[RetrievalStage] = RetrievalStage.CREDIBILITY
    degradable: ClassVar[bool] = True


class CredibilityUnavailableError(CredibilityError):
    """
    A credibility value could not be obtained and must not be invented.

    Retrieval reliability is computed as the mean of relevance multiplied by
    credibility across evidence, and that value carries a statistical
    guarantee. Substituting a default for an unobtainable credibility would
    insert an invented number into that guarantee, where it would be
    indistinguishable from a measured one.

    The absence is therefore recorded as an absence. A downstream consumer may
    decide what to do about it; retrieval may not decide on its behalf by
    fabricating a value.

    Parameters
    ----------
    message:
        Description of the failure.
    source_uri:
        The source whose credibility was sought.
    plugin_name:
        Identifier of the plugin that failed, when known.
    """

    def __init__(
        self,
        message: str,
        *,
        source_uri: str,
        plugin_name: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(message, **kwargs)
        self.source_uri: str = source_uri
        self.plugin_name: str | None = plugin_name
        self.add_context(source_uri=source_uri)
        if plugin_name is not None:
            self.add_context(plugin_name=plugin_name)


# ============================================================================
# ASSEMBLY
# ============================================================================


class EvidenceAssemblyError(PTFMRetrievalError):
    """
    Failure building evidence items from retrieved results.

    Not degradable. Evidence is the module's output, so a failure here means
    there is nothing to return.
    """

    stage: ClassVar[RetrievalStage] = RetrievalStage.ASSEMBLY


class CorroborationIntegrityError(EvidenceAssemblyError):
    """
    An assembled evidence set would misrepresent how much corroboration exists.

    Raised when a node appears more than once, when an item's anchor is
    absent from the set, or when an expanded item claims to be an anchor.

    Each of those inflates the number of distinct anchors, and the number of
    distinct anchors is what a verification engine counts as independent
    support. Inflating it produces confidence that is wrong in the direction
    of overconfidence.

    Parameters
    ----------
    message:
        Description of the violation.
    violated_invariant:
        Name of the invariant that failed, so that a diagnosis identifies the
        guarantee that was broken rather than only that something was.
    """

    def __init__(
        self,
        message: str,
        *,
        violated_invariant: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(message, **kwargs)
        self.violated_invariant: str | None = violated_invariant
        if violated_invariant is not None:
            self.add_context(violated_invariant=violated_invariant)


# ============================================================================
# INGEST
# ============================================================================


class IndexWriteError(PTFMRetrievalError):
    """
    Failure writing a knowledge unit into the indexes.

    Raised on the ingest plane, where this module implements the knowledge
    sink contract. Not degradable: a unit that was not written is not
    retrievable, and reporting success would leave a corpus quietly
    incomplete.
    """

    stage: ClassVar[RetrievalStage] = RetrievalStage.INGEST


class PartialIndexWriteError(IndexWriteError):
    """
    A unit was written to some stores and not others.

    Leaves a node retrievable by one index while its authoritative text or
    provenance is absent, which is the worst failure available to an
    evidence system: a result that appears valid and cannot be traced.

    Recoverable in practice, because node identity is deterministic and
    rewriting the unit reproduces identical content. The condition is
    nonetheless raised rather than tolerated, so that recovery is performed
    deliberately rather than assumed.

    Parameters
    ----------
    message:
        Description of the failure.
    written_stores:
        Stores that accepted the unit.
    failed_stores:
        Stores that did not.
    """

    retryable: ClassVar[bool] = True

    def __init__(
        self,
        message: str,
        *,
        written_stores: Sequence[str] = (),
        failed_stores: Sequence[str] = (),
        **kwargs: Any,
    ) -> None:
        super().__init__(message, **kwargs)
        self.written_stores: tuple[str, ...] = tuple(written_stores)
        self.failed_stores: tuple[str, ...] = tuple(failed_stores)
        self.add_context(
            written_stores=list(written_stores),
            failed_stores=list(failed_stores),
        )
