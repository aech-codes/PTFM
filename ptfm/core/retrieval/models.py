"""
PTFM Retrieval Internal Models

File:
    ptfm/core/retrieval/models.py

Description:
    Defines the models exchanged between stages of the Retrieval Module. These
    are INTERNAL to retrieval. The cross-module contracts are
    ``RetrievedDocument``, ``EvidenceItem`` and ``RetrievalResult``, declared
    in ``ptfm.core.schemas.retrieval``.

Two partition kinds, deliberately separate
------------------------------------------
    A segmentation partition holds everything that depends on how a corpus was
    divided: nodes, lexical terms, relations, supersession. An embedding
    partition holds only vectors.

    Keeping them separate is what makes re-embedding cheap. A model change
    writes a new embedding partition and leaves nodes, spans, identifiers, and
    lexical terms untouched, so a migration does not duplicate the corpus on
    disk for stores that do not depend on the model at all.

    It is also what makes incompatibility inexpressible rather than merely
    detected: vectors from two models are not in the same partition, so they
    cannot be compared by any query that could be written.

Candidates carry signals, never content
---------------------------------------
    A ``Candidate`` is an identifier, a rank, an optional score, and the scale
    that score lives on. It carries no text, no span, and no provenance,
    because an index must not be able to determine what a citation resolves
    to. Authoritative content arrives later, from the node store.

Fusion produces one entry per node
----------------------------------
    A node found by three arms is ONE finding corroborated by method, not
    three findings. ``FusedCandidate`` therefore holds several signals rather
    than appearing several times, and the invariant is enforced at
    construction.

    This matters beyond tidiness. Corroboration is counted downstream by
    distinct anchors, and a node duplicated across arms would inflate that
    count in the quantity the system exists to calibrate.

Diagnostics record what did not happen
--------------------------------------
    A result assembled from two arms because a third was unavailable is not
    the same result as one assembled from three, and the difference must be
    visible. ``RetrievalDiagnostics`` records which arms ran, which degraded,
    and why, so that a reduced result is never presentable as a complete one.

Dependency contract
-------------------
    Imports only the standard library, Pydantic, ``ptfm.core.schemas`` and
    ``ptfm.core.retrieval.exceptions``. It must NOT import ``interfaces``,
    which imports these models under ``TYPE_CHECKING``; the dependency runs in
    one direction only.
"""

from __future__ import annotations

import hashlib
import traceback
from enum import StrEnum
from typing import Final, Iterator, Optional, Sequence

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ptfm.core.retrieval.exceptions import PTFMRetrievalError, RetrievalStage
from ptfm.core.schemas import DiscoveryMethod, RetrievalSignal, SignalScale

__all__ = [
    # Re-export for a single consumer import site
    "RetrievalStage",
    "DiscoveryMethod",
    "SignalScale",
    "RetrievalSignal",
    # Partitions
    "SegmentationPartition",
    "EmbeddingPartition",
    # Filters and plans
    "FilterOperator",
    "FilterField",
    "RetrievalFilter",
    "QueryPlan",
    # Candidates
    "Candidate",
    "ArmResult",
    "FusedCandidate",
    # Diagnostics
    "ArmStatus",
    "ArmDiagnostic",
    "RetrievalDiagnostics",
]

#: Separator between fingerprint components. A byte that cannot occur in a
#: component value, so differing component sets cannot serialize identically.
_KEY_SEPARATOR: Final[str] = "\x00"

#: Digest width for a partition key, in bytes.
_KEY_DIGEST_SIZE: Final[int] = 16


# ============================================================================
# PARTITIONS
# ============================================================================


class SegmentationPartition(BaseModel):
    """
    A corpus divided under one segmentation, in one document contract.

    Identifies everything whose meaning depends on how a corpus was divided:
    nodes, their spans, their identifiers, the lexical index over them, the
    relations between them, and which of their documents are current.

    Attributes
    ----------
    segmentation_fingerprint:
        Digest of every configuration item able to change a node boundary,
        carried on every node by Knowledge Processing. Node identifiers are
        derived in part from it, so partitions under differing fingerprints
        occupy disjoint identifier spaces.
    source_schema_version:
        Version of the document contract the spans were computed against.

        Participates because a node's span is an offset into the document
        text, and a future contract altering text normalization would shift
        every offset while the segmentation fingerprint stayed identical: the
        fingerprint covers chunking configuration and the segmenter, not the
        document contract. Without this component, spans from two contracts
        would share a partition and mean different things.

    Raises
    ------
    ValueError
        If either component is empty.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    segmentation_fingerprint: str = Field(min_length=1)
    source_schema_version: str = Field(min_length=1)

    @property
    def key(self) -> str:
        """
        Stable identifier of this partition.

        Derived rather than stored, so it cannot disagree with the components
        it names.

        Returns
        -------
        str
            A digest identifying the partition.
        """
        return _digest(
            (
                f"segmentation={self.segmentation_fingerprint}",
                f"schema={self.source_schema_version}",
            )
        )

    def accepts(self, node_fingerprint: str, node_schema_version: str) -> bool:
        """
        Report whether a node belongs in this partition.

        Parameters
        ----------
        node_fingerprint:
            The node's segmentation fingerprint.
        node_schema_version:
            The document contract version the node's span assumes.

        Returns
        -------
        bool
            ``True`` when the node's spans and identifier mean here what they
            meant where they were produced.
        """
        return (
            node_fingerprint == self.segmentation_fingerprint
            and node_schema_version == self.source_schema_version
        )

    def __str__(self) -> str:
        """
        Return a readable identifier for diagnostics.

        Returns
        -------
        str
            The partition key.
        """
        return self.key


class EmbeddingPartition(BaseModel):
    """
    Vectors produced by one model over one segmentation partition.

    Several may exist over a single segmentation partition: one serving
    queries, others being built or retired. That asymmetry is the whole of the
    migration story, and it holds because re-embedding changes vectors while
    leaving nodes, spans, identifiers, and lexical terms untouched.

    Every component below is a reason two vectors cannot be compared. A
    partition is therefore not merely a label: it is the mechanism that makes
    an incomparable comparison unexpressible rather than detectable.

    Attributes
    ----------
    segmentation:
        The segmentation partition these vectors describe. Vectors are
        meaningless without the nodes they belong to, so an embedding
        partition is always subordinate to a segmentation partition.
    model_identity:
        Identifier and revision of the model, as reported by the provider that
        produced the vectors.
    dimension:
        Vector width.

        Not redundant with the model identity. Several models expose
        truncated representations at more than one width from a single set of
        weights, so identity alone does not determine dimension.
    normalized:
        Whether the vectors have unit magnitude. Determines whether an inner
        product equals a cosine similarity, so mixing normalized and
        unnormalized vectors silently changes what a score means.
    composition_id:
        Identifier of the rule that composed passage text before embedding. A
        node embedded with its heading path is not comparable with the same
        node embedded without it.
    query_encoding:
        Identifier of the transformation a query must undergo to be comparable
        with these vectors. Empty for a symmetric model.

    Raises
    ------
    ValueError
        If a required component is empty or the dimension is not positive.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    segmentation: SegmentationPartition
    model_identity: str = Field(min_length=1)
    dimension: int = Field(ge=1)
    normalized: bool
    composition_id: str = Field(min_length=1)
    query_encoding: str = ""

    @property
    def key(self) -> str:
        """
        Stable identifier of this partition.

        Returns
        -------
        str
            A digest identifying the partition, incorporating its
            segmentation.
        """
        return _digest(
            (
                f"segmentation_key={self.segmentation.key}",
                f"model={self.model_identity}",
                f"dimension={self.dimension}",
                f"normalized={'true' if self.normalized else 'false'}",
                f"composition={self.composition_id}",
                f"query_encoding={self.query_encoding}",
            )
        )

    def accepts_query_encoding(self, encoding: str) -> bool:
        """
        Report whether a provider encodes queries compatibly with these
        vectors.

        Checked before a query runs rather than inferred from results,
        because an incompatible pairing degrades retrieval measurably and
        produces no error of its own.

        Parameters
        ----------
        encoding:
            The convention the configured provider applies.

        Returns
        -------
        bool
            ``True`` when the conventions agree.
        """
        return encoding == self.query_encoding

    def describes_same_space(self, other: EmbeddingPartition) -> bool:
        """
        Report whether two partitions hold comparable vectors.

        Parameters
        ----------
        other:
            The partition to compare against.

        Returns
        -------
        bool
            ``True`` when a similarity between their vectors would be
            meaningful.
        """
        return self.key == other.key

    def __str__(self) -> str:
        """
        Return a readable identifier for diagnostics.

        Returns
        -------
        str
            The partition key.
        """
        return self.key


def _digest(components: Sequence[str]) -> str:
    """
    Compose a stable digest from ordered components.

    The algorithm is fixed rather than injected. A partition key must be
    reproducible across every deployment and process indefinitely, so a
    configurable digest would let two deployments compute different keys for
    identical partitions and treat compatible data as incompatible.

    Parameters
    ----------
    components:
        Components in a fixed order, each already rendered as a string.

    Returns
    -------
    str
        A hexadecimal digest.
    """
    payload = _KEY_SEPARATOR.join(components)
    return hashlib.blake2b(
        payload.encode("utf-8"), digest_size=_KEY_DIGEST_SIZE
    ).hexdigest()


# ============================================================================
# FILTERS AND PLANS
# ============================================================================


class FilterField(StrEnum):
    """
    A node attribute a retrieval filter may constrain.

    Enumerated rather than free-form, and this is the decision that keeps
    filtering domain-neutral and safe.

    Domain-neutral, because every member names a structural property that
    Knowledge Processing records for any corpus: none of them presumes a
    legal, clinical, or financial document. Domain-specific filtering reaches
    the engine through metadata, which a plugin populates and which the engine
    treats as opaque keys and values.

    Safe, because a filter constructed from an enumeration cannot be built by
    concatenating query text into a backend's filter language. A free-form
    field name would put user input into a position where a backend parses it.
    """

    #: The document a node belongs to.
    DOCUMENT_ID = "DOCUMENT_ID"
    #: The source a document came from.
    SOURCE_URI = "SOURCE_URI"
    #: Structural character of the node, such as prose or a table region.
    NODE_TYPE = "NODE_TYPE"
    #: A heading in the node's ancestor path.
    HEADING_PATH = "HEADING_PATH"
    #: A page the node originates on.
    PAGE_NUMBER = "PAGE_NUMBER"
    #: Position of the node within its document.
    ORDINAL = "ORDINAL"
    #: An enricher-supplied metadata key. The domain-facing member.
    METADATA = "METADATA"


class FilterOperator(StrEnum):
    """
    How a filter compares a node attribute with a value.

    Deliberately small. Every operator here is expressible by every backend
    worth supporting, so a filter written against this vocabulary does not
    become unportable, and a backend need not emulate an operator it lacks.
    """

    EQUALS = "EQUALS"
    NOT_EQUALS = "NOT_EQUALS"
    IN = "IN"
    NOT_IN = "NOT_IN"
    #: Matches when the attribute's sequence begins with the given values,
    #: which is how a heading path selects a section and its descendants.
    STARTS_WITH = "STARTS_WITH"
    GREATER_THAN_OR_EQUAL = "GREATER_THAN_OR_EQUAL"
    LESS_THAN_OR_EQUAL = "LESS_THAN_OR_EQUAL"


class RetrievalFilter(BaseModel):
    """
    A structural constraint applied during search.

    Applied during search rather than after it, so that a backend does not
    spend effort ranking results a caller will discard, and so that a filtered
    query still returns the requested number of results.

    Attributes
    ----------
    field:
        The attribute constrained.
    operator:
        How the attribute is compared.
    values:
        Values compared against. Always a sequence, even for a single-valued
        operator, so that a backend adapter handles one shape rather than two.
    metadata_key:
        The metadata key constrained, required when the field is metadata and
        forbidden otherwise.

        The one place a caller supplies a name rather than choosing from an
        enumeration, and therefore the one place a backend adapter must bind
        a parameter rather than compose a string.

    Raises
    ------
    ValueError
        If the operator and value count disagree, or the metadata key is
        present without a metadata field or absent with one.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    field: FilterField
    operator: FilterOperator
    values: tuple[str, ...]
    metadata_key: Optional[str] = None

    @model_validator(mode="after")
    def _validate_metadata_key(self) -> RetrievalFilter:
        """
        Ensure a metadata key accompanies exactly a metadata field.

        Returns
        -------
        RetrievalFilter
            The validated instance.

        Raises
        ------
        ValueError
            If the key and field disagree.
        """
        if self.field is FilterField.METADATA and not self.metadata_key:
            raise ValueError(
                "A metadata filter must name the metadata key it constrains."
            )
        if self.field is not FilterField.METADATA and self.metadata_key:
            raise ValueError(
                f"metadata_key is meaningful only for a METADATA filter, not "
                f"for {self.field.value}."
            )
        return self

    @model_validator(mode="after")
    def _validate_value_count(self) -> RetrievalFilter:
        """
        Ensure the value count suits the operator.

        Returns
        -------
        RetrievalFilter
            The validated instance.

        Raises
        ------
        ValueError
            If no value is supplied, or a single-valued operator is given
            several.
        """
        if not self.values:
            raise ValueError(
                f"A {self.operator.value} filter requires at least one value."
            )

        single_valued = {
            FilterOperator.EQUALS,
            FilterOperator.NOT_EQUALS,
            FilterOperator.GREATER_THAN_OR_EQUAL,
            FilterOperator.LESS_THAN_OR_EQUAL,
        }
        if self.operator in single_valued and len(self.values) != 1:
            raise ValueError(
                f"A {self.operator.value} filter takes exactly one value, got "
                f"{len(self.values)}."
            )
        return self


class QueryPlan(BaseModel):
    """
    Everything an arm needs to search, and nothing it does not.

    Passed to every arm, so that adding an input requires no change to the
    arm contract. An arm reads what it needs and ignores the rest: a lexical
    arm never touches the vector, a dense arm never touches the text.

    Attributes
    ----------
    query_id:
        Correlation identifier for the query, carried into diagnostics.
    text:
        Query text after domain preprocessing. Used by lexical search.
    vector:
        Query vector, or ``None`` when dense retrieval is unavailable.

        Absence is ordinary rather than exceptional: a deployment may run
        without an embedding provider, or encoding may have degraded, and
        lexical and structural retrieval continue in either case.
    segmentation:
        The segmentation partition to search.
    embedding:
        The embedding partition to search, ``None`` when there is no vector.
    top_k:
        Largest number of candidates each arm should return.
    filters:
        Structural constraints applied during search.
    exact:
        Whether approximate search is forbidden.

        Requested when a result must be reproducible, as when auditing a past
        response. Approximate indexes trade exactness for speed and are not
        reproducible across index rebuilds, so a caller promising
        reproducibility must be able to ask for the slower guarantee.

    Raises
    ------
    ValueError
        If a vector is supplied without an embedding partition or the reverse.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    query_id: str = Field(min_length=1)
    text: str
    vector: Optional[tuple[float, ...]] = None
    segmentation: SegmentationPartition
    embedding: Optional[EmbeddingPartition] = None
    top_k: int = Field(ge=1)
    filters: tuple[RetrievalFilter, ...] = ()
    exact: bool = False

    @model_validator(mode="after")
    def _validate_vector_pairing(self) -> QueryPlan:
        """
        Ensure a vector and its partition accompany one another.

        A vector without a partition could not be checked for comparability
        before searching, and a partition without a vector describes a search
        that cannot run.

        Returns
        -------
        QueryPlan
            The validated instance.

        Raises
        ------
        ValueError
            If exactly one of the two is present.
        """
        if self.vector is not None and self.embedding is None:
            raise ValueError(
                "A query vector requires the embedding partition it belongs "
                "to, so that comparability can be established before the "
                "search runs rather than inferred from its results."
            )
        if self.embedding is not None and self.vector is None:
            raise ValueError(
                "An embedding partition without a query vector describes a "
                "search that cannot run."
            )
        return self

    @model_validator(mode="after")
    def _validate_partition_agreement(self) -> QueryPlan:
        """
        Ensure the embedding partition belongs to the segmentation partition.

        Returns
        -------
        QueryPlan
            The validated instance.

        Raises
        ------
        ValueError
            If the embedding partition describes a different segmentation.
        """
        if (
            self.embedding is not None
            and self.embedding.segmentation.key != self.segmentation.key
        ):
            raise ValueError(
                f"Embedding partition describes segmentation "
                f"{self.embedding.segmentation.key!r} while the plan searches "
                f"{self.segmentation.key!r}. Vectors would be compared against "
                f"nodes they were not produced from."
            )
        return self

    @property
    def has_vector(self) -> bool:
        """
        Whether dense retrieval is possible for this plan.

        Returns
        -------
        bool
            ``True`` when a vector and its partition are present.
        """
        return self.vector is not None


# ============================================================================
# CANDIDATES
# ============================================================================


class Candidate(BaseModel):
    """
    One arm's uncalibrated observation about one node.

    The complete output of a retrieval arm: an identifier, a position, and
    optionally a score with the scale it lives on. No text, no span, no
    provenance.

    That restriction is structural rather than stylistic. Authoritative
    content comes from the node store alone, so an index holding a stale or
    corrupted copy cannot change what a citation resolves to. An arm has no
    mechanism by which to corrupt evidence, only to fail to find it.

    Attributes
    ----------
    node_id:
        Identifier of the node found.
    method:
        How the node was discovered.
    rank:
        Zero-based position within this arm's own ordering. Always meaningful,
        whereas a score may be absent.
    score:
        The arm's own score, ``None`` where it produces an ordering without
        one, as a structural predicate does. Uncalibrated: it is not a
        probability, and converting it into one belongs to the calibration
        layer alone.
    scale:
        The scale the score lives on. Recorded so that comparing scores from
        different arms is visibly wrong rather than silently plausible.

    Raises
    ------
    ValueError
        If a score is supplied without a scale that admits one.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    node_id: str = Field(min_length=1)
    method: DiscoveryMethod
    rank: int = Field(ge=0)
    score: Optional[float] = None
    scale: SignalScale = SignalScale.RANK_ONLY

    @model_validator(mode="after")
    def _validate_score_scale(self) -> Candidate:
        """
        Ensure a score and its scale agree about whether a score exists.

        Returns
        -------
        Candidate
            The validated instance.

        Raises
        ------
        ValueError
            If a score accompanies a rank-only scale, or a scoring scale
            carries no score.
        """
        if self.score is not None and self.scale is SignalScale.RANK_ONLY:
            raise ValueError(
                "A candidate carrying a score must name the scale that score "
                "lives on; RANK_ONLY declares that no score exists."
            )
        if self.score is None and self.scale is not SignalScale.RANK_ONLY:
            raise ValueError(
                f"A candidate with scale {self.scale.value} declares a score "
                f"it does not carry. Use RANK_ONLY where an arm orders "
                f"results without scoring them."
            )
        return self

    def as_signal(self) -> RetrievalSignal:
        """
        Render this observation as a retrieval signal.

        Returns
        -------
        RetrievalSignal
            The signal recorded on the result this candidate contributes to.
        """
        return RetrievalSignal(
            method=self.method,
            rank=self.rank,
            raw_score=self.score,
            scale=self.scale,
        )


class ArmResult(BaseModel):
    """
    One arm's complete contribution to a query.

    Carries the arm's identity alongside its candidates, so that fusion knows
    which arm produced which ordering without the caller maintaining a
    parallel mapping.

    Attributes
    ----------
    arm_name:
        Identifier of the arm.
    method:
        How this arm discovers nodes.
    candidates:
        The arm's ordering. Empty when nothing matched, which is an answer
        rather than a failure.
    latency_ms:
        Wall-clock time the arm took.

    Raises
    ------
    ValueError
        If ranks are not contiguous from zero, a node appears twice, or a
        candidate's method disagrees with the arm's.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    arm_name: str = Field(min_length=1)
    method: DiscoveryMethod
    candidates: tuple[Candidate, ...] = ()
    latency_ms: float = Field(default=0.0, ge=0.0)

    @model_validator(mode="after")
    def _validate_ordering(self) -> ArmResult:
        """
        Ensure the arm produced a well-formed ordering.

        Rank-based fusion depends on ranks being an ordering rather than an
        arbitrary set of integers, so a gap or a repeat would distort every
        fused score derived from them.

        Returns
        -------
        ArmResult
            The validated instance.

        Raises
        ------
        ValueError
            If ranks are not contiguous from zero, or a node appears twice.
        """
        seen: set[str] = set()

        for position, candidate in enumerate(self.candidates):
            if candidate.rank != position:
                raise ValueError(
                    f"Arm {self.arm_name!r} returned rank {candidate.rank} at "
                    f"position {position}. Ranks must be contiguous from zero, "
                    f"because rank-based fusion treats them as an ordering."
                )
            if candidate.node_id in seen:
                raise ValueError(
                    f"Arm {self.arm_name!r} returned node "
                    f"{candidate.node_id!r} twice. One arm finding a node once "
                    f"is one observation; repeating it would weight that node "
                    f"twice in fusion."
                )
            seen.add(candidate.node_id)

            if candidate.method is not self.method:
                raise ValueError(
                    f"Arm {self.arm_name!r} declares method "
                    f"{self.method.value} but returned a candidate discovered "
                    f"by {candidate.method.value}."
                )

        return self

    @property
    def node_ids(self) -> tuple[str, ...]:
        """
        Identifiers this arm returned, in its own order.

        Returns
        -------
        tuple[str, ...]
            Node identifiers.
        """
        return tuple(candidate.node_id for candidate in self.candidates)

    def __len__(self) -> int:
        """
        Number of candidates this arm returned.

        Returns
        -------
        int
            Candidate count.
        """
        return len(self.candidates)


class FusedCandidate(BaseModel):
    """
    One node, with every arm's observation of it.

    A node found by three arms is ONE finding corroborated by method, not
    three findings. This model exists to make that structural: several signals
    on one entry, never several entries.

    The distinction is not presentational. Corroboration is counted downstream
    by distinct anchors, so a node duplicated across arms would inflate that
    count in exactly the quantity the system exists to calibrate.

    Attributes
    ----------
    node_id:
        Identifier of the node.
    signals:
        Every arm's observation, in a deterministic order.
    fused_rank:
        Zero-based position in the fused ordering.
    fused_score:
        The fusion strategy's own value.

        Uncalibrated and ordinal. A reciprocal rank fusion value is derived
        from positions alone and carries no probabilistic meaning whatever;
        treating it as a confidence would be a category error, which is why
        its scale travels with it.
    fused_scale:
        The scale the fused value lives on.

    Raises
    ------
    ValueError
        If no signal is present, or two signals come from one method.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    node_id: str = Field(min_length=1)
    signals: tuple[RetrievalSignal, ...]
    fused_rank: int = Field(ge=0)
    fused_score: Optional[float] = None
    fused_scale: SignalScale = SignalScale.RECIPROCAL_RANK

    @model_validator(mode="after")
    def _validate_signals(self) -> FusedCandidate:
        """
        Ensure the observations are present and distinct by method.

        Returns
        -------
        FusedCandidate
            The validated instance.

        Raises
        ------
        ValueError
            If no signal is present, or one method contributed twice.
        """
        if not self.signals:
            raise ValueError(
                "A fused candidate must carry at least one signal; a node with "
                "no observation was never found by anything."
            )

        methods = [signal.method for signal in self.signals]
        if len(set(methods)) != len(methods):
            raise ValueError(
                f"Node {self.node_id!r} carries two signals from one method. "
                f"An arm finding a node once is one observation, and counting "
                f"it twice would weight that node beyond what was observed."
            )
        return self

    @property
    def methods(self) -> frozenset[DiscoveryMethod]:
        """
        The methods that found this node.

        Returns
        -------
        frozenset[DiscoveryMethod]
            Distinct discovery methods.
        """
        return frozenset(signal.method for signal in self.signals)

    @property
    def arm_count(self) -> int:
        """
        Number of arms that found this node.

        A diagnostic about retrieval agreement, and explicitly NOT a measure
        of corroboration. Corroboration counts distinct source passages;
        several arms finding one passage is one passage.

        Returns
        -------
        int
            Distinct method count.
        """
        return len(self.methods)

    def signal_for(self, method: DiscoveryMethod) -> Optional[RetrievalSignal]:
        """
        Retrieve one method's observation of this node.

        Parameters
        ----------
        method:
            The method whose observation is sought.

        Returns
        -------
        Optional[RetrievalSignal]
            The signal, or ``None`` when that method did not find this node.
        """
        for signal in self.signals:
            if signal.method is method:
                return signal
        return None


# ============================================================================
# DIAGNOSTICS
# ============================================================================


class ArmStatus(StrEnum):
    """
    How an arm fared during a query.

    Recorded for every configured arm, including those that did not run, so
    that a result assembled from two arms is distinguishable from one
    assembled from three.
    """

    #: Ran and returned results.
    COMPLETED = "COMPLETED"
    #: Ran and found nothing, which is an answer rather than a failure.
    EMPTY = "EMPTY"
    #: Did not run because a required input was absent, as a dense arm with no
    #: query vector.
    SKIPPED = "SKIPPED"
    #: Failed. Its contribution is missing from the result.
    FAILED = "FAILED"
    #: Exceeded its deadline. Its contribution is missing from the result.
    TIMED_OUT = "TIMED_OUT"


class ArmDiagnostic(BaseModel):
    """
    What became of one arm during a query.

    Attributes
    ----------
    arm_name:
        Identifier of the arm.
    method:
        How this arm discovers nodes.
    status:
        How it fared.
    candidate_count:
        Candidates it contributed.
    latency_ms:
        Wall-clock time it took, including a failed attempt.
    error_code:
        Machine-readable code when the arm failed.
    message:
        Human-readable description when the arm failed.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    arm_name: str = Field(min_length=1)
    method: DiscoveryMethod
    status: ArmStatus
    candidate_count: int = Field(default=0, ge=0)
    latency_ms: float = Field(default=0.0, ge=0.0)
    error_code: Optional[str] = None
    message: Optional[str] = None
    traceback_digest: Optional[str] = None

    @classmethod
    def from_exception(
        cls,
        error: PTFMRetrievalError,
        *,
        arm_name: str,
        method: DiscoveryMethod,
        status: ArmStatus,
        latency_ms: float = 0.0,
    ) -> ArmDiagnostic:
        """
        Build a diagnostic from an arm's failure.

        Parameters
        ----------
        error:
            The failure.
        arm_name:
            Identifier of the arm.
        method:
            How the arm discovers nodes.
        status:
            Whether the arm failed or timed out.
        latency_ms:
            Time spent before failing.

        Returns
        -------
        ArmDiagnostic
            The populated diagnostic.
        """
        return cls(
            arm_name=arm_name,
            method=method,
            status=status,
            latency_ms=latency_ms,
            error_code=error.error_code,
            message=error.message,
            traceback_digest=_traceback_digest(error),
        )

    @property
    def contributed(self) -> bool:
        """
        Whether this arm contributed to the result.

        Returns
        -------
        bool
            ``True`` when the arm ran successfully, whether or not it found
            anything.
        """
        return self.status in {ArmStatus.COMPLETED, ArmStatus.EMPTY}


class RetrievalDiagnostics(BaseModel):
    """
    What happened during one query, including what did not.

    A result assembled from two arms because a third was unavailable is not
    the same result as one assembled from three, and the difference must be
    visible to whatever consumes it. Recording only successes would let a
    degraded result be presented as a complete one, which is the silent
    failure this module is most exposed to.

    Attributes
    ----------
    query_id:
        Correlation identifier for the query.
    segmentation_key:
        The segmentation partition served.
    embedding_key:
        The embedding partition served, ``None`` when dense retrieval did not
        run.
    arms:
        One entry per configured arm, including those that did not run.
    exact_search:
        Whether approximate search was forbidden. Recorded because a result is
        reproducible only when it was.
    embedding_coverage:
        Proportion of the segmentation partition's nodes that carry a vector,
        ``None`` when dense retrieval did not run.

        A value below one means dense recall is incomplete for reasons of
        indexing rather than relevance, and an unembedded node is invisible to
        dense search however well it matches.
    fused_count:
        Distinct nodes after fusion.
    expanded_count:
        Context nodes added. Never counted toward corroboration.
    superseded_count:
        Results withheld because their documents are no longer current.
    currency_unchecked:
        Results returned WITHOUT a currency check, because the supersession
        store could not be consulted.

        Reported separately from ``superseded_count`` because the two are
        different claims. A superseded result was checked and withheld; an
        unchecked one may or may not be current, and nobody knows which.
        Folding either count into the other, or omitting this one, would let a
        caller read "0 superseded" as "everything here is current" when the
        check never ran — the same substitution of ignorance for a measurement
        that made credibility absence worth representing.
    credibility_unavailable:
        Sources whose credibility could not be obtained, recorded rather than
        defaulted so that no invented value enters a computation carrying a
        statistical guarantee.
    latency_ms:
        Wall-clock time for the query.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    query_id: str = Field(min_length=1)
    segmentation_key: str = Field(min_length=1)
    embedding_key: Optional[str] = None
    arms: tuple[ArmDiagnostic, ...] = ()
    exact_search: bool = False
    embedding_coverage: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    fused_count: int = Field(default=0, ge=0)
    expanded_count: int = Field(default=0, ge=0)
    superseded_count: int = Field(default=0, ge=0)
    currency_unchecked: int = Field(default=0, ge=0)
    credibility_unavailable: tuple[str, ...] = ()
    latency_ms: float = Field(default=0.0, ge=0.0)

    @property
    def degraded(self) -> bool:
        """
        Whether any configured arm failed to contribute.

        The single question a consumer most needs answered about a result, and
        the reason diagnostics record arms that did not run.

        Returns
        -------
        bool
            ``True`` when an arm failed, timed out, or was skipped.
        """
        return any(not arm.contributed for arm in self.arms)

    @property
    def reproducible(self) -> bool:
        """
        Whether this result could be reproduced from an unchanged corpus.

        Requires exact search and a complete set of contributing arms.
        Approximate indexes are not reproducible across rebuilds, and a
        degraded result depends on which components happened to be reachable.

        Returns
        -------
        bool
            ``True`` when the result is reproducible.
        """
        return self.exact_search and not self.degraded

    @property
    def contributing_arms(self) -> tuple[str, ...]:
        """
        Names of the arms that contributed.

        Returns
        -------
        tuple[str, ...]
            Arm names, in the order recorded.
        """
        return tuple(arm.arm_name for arm in self.arms if arm.contributed)

    @property
    def failed_arms(self) -> tuple[str, ...]:
        """
        Names of the arms that did not contribute.

        Returns
        -------
        tuple[str, ...]
            Arm names, in the order recorded.
        """
        return tuple(arm.arm_name for arm in self.arms if not arm.contributed)

    def iter_failures(self) -> Iterator[ArmDiagnostic]:
        """
        Iterate the arms that failed or timed out.

        Skipped arms are excluded, because a skipped arm is a consequence of
        the query rather than a fault: a dense arm cannot run without a query
        vector, and reporting that as a failure would misdescribe it.

        Yields
        ------
        ArmDiagnostic
            Each failure.
        """
        for arm in self.arms:
            if arm.status in {ArmStatus.FAILED, ArmStatus.TIMED_OUT}:
                yield arm


def _traceback_digest(error: BaseException) -> Optional[str]:
    """
    Compute a short digest identifying a traceback shape.

    Uses ``hashlib`` directly rather than an injected hash provider: this is
    diagnostic grouping, explicitly not artifact identity, and routing it
    through the identity abstraction would blur a boundary that matters.

    Parameters
    ----------
    error:
        The exception whose traceback should be digested.

    Returns
    -------
    Optional[str]
        Sixteen hexadecimal characters, or ``None`` when the exception carries
        no traceback.
    """
    if error.__traceback__ is None:
        return None
    joined = "".join(traceback.format_tb(error.__traceback__)).encode("utf-8")
    return hashlib.blake2b(joined, digest_size=8).hexdigest()
