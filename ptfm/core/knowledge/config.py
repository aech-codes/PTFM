"""
PTFM Knowledge Processing Configuration

File:
    ptfm/core/knowledge/config.py

Description:
    Declares every operational parameter of the Knowledge Processing Module,
    and computes the segmentation fingerprint on which node identity depends.

    This module is the SOLE location in which knowledge processing numeric
    values are defined. No stage, strategy, or provider may contain a numeric
    literal compared against a limit; every such value is injected from here.

Two kinds of configuration
--------------------------
    Most parameters here are operational: batch sizes, deadlines, concurrency
    bounds, retry counts. Changing one alters throughput or resource use and
    nothing else.

    ``ChunkingConfig`` is different in kind. Its parameters determine where
    node boundaries fall, and node identity is derived in part from those
    boundaries. Changing any of them is therefore a CORPUS MIGRATION rather
    than a tuning adjustment: the resulting nodes occupy a disjoint identifier
    space, and every citation stored against the previous configuration ceases
    to resolve.

    The distinction is enforced structurally rather than by documentation.
    ``ChunkingConfig`` contains boundary-affecting parameters and nothing
    else, and the fingerprint is computed from the whole of it. "Does this
    parameter affect identity?" therefore reduces to "does it belong in
    ``ChunkingConfig``?", which is a placement decision made once, rather than
    an inclusion list that can silently drift out of step with the fields it
    describes.

The inclusion rule
------------------
    A parameter contributes to the segmentation fingerprint IF AND ONLY IF
    changing it can change some node's span.

    Embedding models, enrichers, relation derivation, validation strictness,
    concurrency, and telemetry never contribute. Including one would mean that
    adding an enricher invalidates every node identifier in the corpus,
    breaking every stored citation, for a change that alters no node's text or
    span.

    Omitting a parameter that does affect boundaries is the opposite failure,
    and the more dangerous one: identifiers change while the fingerprint does
    not, so an index silently fails to resolve against its corpus and no error
    is raised anywhere.

Runtime contributions
---------------------
    Configuration alone is not sufficient. A sentence segmenter's version
    affects where an oversized block is divided, and that version is a
    property of the injected implementation rather than of configuration.

    The fingerprint is therefore composed from configuration together with
    the runtime components that declare boundary-affecting identity: the
    strategy's own parameters and the segmenter's version. The composition
    root supplies them.

Environment layering
--------------------
    Every value may be supplied by environment variable using the prefix
    ``PTFM_KNOWLEDGE_`` with ``__`` separating nesting levels::

        PTFM_KNOWLEDGE_MAX_CONCURRENT_DOCUMENTS=16
        PTFM_KNOWLEDGE_CHUNKING__MAX_CHUNK_CHARS=1500
        PTFM_KNOWLEDGE_EMBEDDING__BATCH_SIZE=64

Dependency contract
-------------------
    Imports only the standard library, Pydantic, pydantic-settings and
    ``ptfm.core.knowledge.exceptions``. It must not import ``interfaces``,
    ``models``, or any stage, strategy, or provider.
"""

from __future__ import annotations

import hashlib
import os
from enum import StrEnum
from typing import Final, Mapping, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from ptfm.core.knowledge.exceptions import KnowledgeConfigurationError

__all__ = [
    "SEGMENTATION_FINGERPRINT_VERSION",
    "EnrichmentFailurePolicy",
    "ChunkingConfig",
    "EmbeddingConfig",
    "EnrichmentConfig",
    "RelationConfig",
    "ValidationConfig",
    "SinkConfig",
    "TelemetryConfig",
    "KnowledgeConfig",
    "compute_segmentation_fingerprint",
]

#: Version of the fingerprint composition scheme itself.
#:
#: Participates in every digest, so that a future change to how the
#: fingerprint is composed produces visibly different values rather than
#: silently colliding with fingerprints computed under the previous scheme.
#: Without it, a scheme change could map two different configurations onto one
#: fingerprint, which is the exact failure the fingerprint exists to prevent.
SEGMENTATION_FINGERPRINT_VERSION: Final[str] = "1"

#: Digest width for the fingerprint, in bytes. Fixed rather than configurable:
#: a per-deployment width would make fingerprints incomparable between
#: deployments with no visible indication.
_FINGERPRINT_DIGEST_SIZE: Final[int] = 16

#: Separator between fingerprint components. A byte that cannot occur in a
#: component name or value, so that differing component sets cannot serialize
#: to the same string.
_FINGERPRINT_SEPARATOR: Final[str] = "\x00"

#: Fallback worker count when the host CPU count cannot be determined.
_FALLBACK_CPU_COUNT: Final[int] = 4


class EnrichmentFailurePolicy(StrEnum):
    """
    How a run responds to an enricher failure.

    Enrichment is additive and produces metadata only, so a failure loses
    metadata rather than a node. Degrading is therefore the proportionate
    default, and failing is available where a deployment treats domain
    metadata as mandatory.
    """

    #: Record the failure, attach no metadata from that enricher, and continue.
    DEGRADE = "DEGRADE"
    #: Treat the failure as a document failure and quarantine the document.
    FAIL = "FAIL"


# ============================================================================
# SEGMENTATION IDENTITY
# ============================================================================


class ChunkingConfig(BaseModel):
    """
    Parameters determining where knowledge node boundaries fall.

    IDENTITY-AFFECTING. Every field in this model contributes to the
    segmentation fingerprint, and therefore to node identity. Changing any of
    them is a corpus migration: the resulting nodes occupy a disjoint
    identifier space, and every citation stored against the previous
    configuration ceases to resolve.

    Nothing that does not affect a boundary may be added here. A parameter
    governing validation strictness, telemetry, or performance belongs in
    another group, because placing it here would make an unrelated adjustment
    invalidate the entire corpus.

    Conversely, no boundary-affecting parameter may live elsewhere: it would
    change identity without changing the fingerprint, which is the silent
    failure the fingerprint exists to prevent.

    Overlap is deliberately absent
    ------------------------------
    Nodes never overlap, and no configuration can make them. Overlapping
    nodes would duplicate text across the corpus, and duplicated text enters
    verification as two evidence items rather than one, so a claim would
    appear better corroborated than the corpus warrants. In a system whose
    output is calibrated confidence, that is a correctness defect rather than
    an inefficiency.

    Overlap would also inflate the document frequency of any term in a
    duplicated region, lowering its inverse document frequency and making a
    discriminative term appear less discriminative to sparse retrieval, and it
    would leave no canonical answer to which node a cited character belongs
    to.

    The need overlap ordinarily serves, a claim whose support straddles two
    adjacent nodes, is met by reference rather than duplication: ``PRECEDES``
    relations link every adjacent pair, so retrieval can expand a hit into its
    neighbours at query time. That remedy is tunable per query, whereas
    overlap would be frozen into node identity at ingestion.

    Attributes
    ----------
    strategy:
        Identifier of the chunking strategy. Different strategies place
        different boundaries, so this is identity-affecting before any
        parameter is considered.
    sentence_segmenter:
        Identifier of the segmenter used to divide an oversized block. Its
        version also contributes, but at runtime rather than from
        configuration, since a version is a property of the installed
        implementation.
    target_chunk_chars:
        Preferred node size. A strategy accumulates blocks toward this size
        and closes a node at the first boundary beyond it.
    max_chunk_chars:
        Largest node permitted. A block sequence exceeding this is divided,
        and a single block exceeding it is split internally where the strategy
        supports it.
    min_chunk_chars:
        Smallest node ordinarily emitted. A trailing fragment below this size
        is merged into its predecessor rather than emitted alone, since a node
        of a few characters carries no retrievable meaning.
    split_oversized_blocks:
        Whether a single block exceeding the maximum is divided at sentence
        boundaries. When disabled, such a block yields one oversized node
        rather than failing, which some corpora prefer to fragmentation.
    preserve_table_rows:
        Whether a table is divided only at row boundaries. Splitting mid-row
        produces a node holding part of one record and part of the next, which
        is worse than an oversized node.
    respect_heading_boundaries:
        Whether a node is closed when a heading is encountered, so that a node
        does not span two sections.

    Raises
    ------
    ValueError
        If the size bounds are not ordered.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", validate_default=True)

    strategy: str = Field(default="block_aligned", min_length=1)
    sentence_segmenter: str = Field(default="regex", min_length=1)
    target_chunk_chars: int = Field(default=1_000, ge=1)
    max_chunk_chars: int = Field(default=2_000, ge=1)
    min_chunk_chars: int = Field(default=100, ge=0)
    split_oversized_blocks: bool = True
    preserve_table_rows: bool = True
    respect_heading_boundaries: bool = True

    @model_validator(mode="after")
    def _validate_size_ordering(self) -> ChunkingConfig:
        """
        Ensure the size bounds form a coherent ordering.

        Returns
        -------
        ChunkingConfig
            The validated instance.

        Raises
        ------
        ValueError
            If the minimum exceeds the target, or the target exceeds the
            maximum.
        """
        if self.min_chunk_chars > self.target_chunk_chars:
            raise ValueError(
                f"min_chunk_chars ({self.min_chunk_chars}) exceeds "
                f"target_chunk_chars ({self.target_chunk_chars}); no node "
                f"could satisfy both"
            )
        if self.target_chunk_chars > self.max_chunk_chars:
            raise ValueError(
                f"target_chunk_chars ({self.target_chunk_chars}) exceeds "
                f"max_chunk_chars ({self.max_chunk_chars}); every node would "
                f"be oversized on reaching its target"
            )
        return self

    def boundary_components(self) -> Mapping[str, str]:
        """
        Report every configured parameter able to change a span boundary.

        Derived from the whole model rather than from a maintained list, so
        that a field added here cannot be omitted by oversight. This is the
        reason the model contains boundary-affecting parameters exclusively:
        the question "does this parameter affect identity?" reduces to a
        placement decision rather than a separate enumeration that could drift.

        Values are rendered as strings so that the digest does not depend on
        numeric formatting or on how a value round-trips through an
        environment variable.

        Returns
        -------
        Mapping[str, str]
            Parameter names mapped to their values, namespaced by source.
        """
        return {
            f"chunking.{name}": _render(value)
            for name, value in sorted(self.model_dump().items())
        }


def compute_segmentation_fingerprint(
    *component_maps: Mapping[str, str],
) -> str:
    """
    Compose a fingerprint from every boundary-affecting component.

    Combines configuration with the runtime components that declare
    boundary-affecting identity of their own: the strategy's parameters and
    the sentence segmenter's version. Configuration alone would be
    insufficient, since a segmenter upgrade shifts boundaries within oversized
    blocks while no configured value changes.

    The digest algorithm is fixed rather than injected. A fingerprint must be
    reproducible across every deployment and every process, indefinitely: a
    deployment configured with a different hash provider would compute a
    different fingerprint for identical configuration, producing disjoint
    identifier spaces for no reason at all. Swappability would be a defect
    here rather than a feature.

    Parameters
    ----------
    *component_maps:
        Component mappings to combine. Keys are namespaced by their source, so
        two sources contributing the same parameter name do not collide.

    Returns
    -------
    str
        A hexadecimal digest prefixed by the fingerprint scheme version, in
        the form ``"1:<digest>"``. The prefix makes a future change to the
        composition scheme visible rather than silently colliding with values
        computed under the previous one.

    Raises
    ------
    KnowledgeConfigurationError
        If two sources contribute the same key with differing values, which
        would make the fingerprint depend on argument order.
    """
    merged: dict[str, str] = {}

    for components in component_maps:
        for key, value in components.items():
            existing = merged.get(key)
            if existing is not None and existing != value:
                raise KnowledgeConfigurationError(
                    f"Segmentation component {key!r} was contributed twice "
                    f"with differing values ({existing!r} and {value!r}). The "
                    f"fingerprint would depend on the order the components "
                    f"were supplied, which would make node identity "
                    f"non-deterministic.",
                    context={"component": key},
                )
            merged[key] = value

    # Sorted so that composition order cannot affect the digest.
    payload = _FINGERPRINT_SEPARATOR.join(
        f"{key}={merged[key]}" for key in sorted(merged)
    )
    digest = hashlib.blake2b(
        payload.encode("utf-8"), digest_size=_FINGERPRINT_DIGEST_SIZE
    ).hexdigest()

    return f"{SEGMENTATION_FINGERPRINT_VERSION}:{digest}"


def _render(value: object) -> str:
    """
    Render a configuration value in a form stable across processes.

    Booleans are rendered in lower case and collections are sorted, so that a
    value supplied through an environment variable digests identically to the
    same value supplied in code.

    Parameters
    ----------
    value:
        The configured value.

    Returns
    -------
    str
        A stable string rendering.
    """
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (frozenset, set)):
        return ",".join(sorted(str(item) for item in value))
    if isinstance(value, (list, tuple)):
        return ",".join(str(item) for item in value)
    return str(value)


# ============================================================================
# OPERATIONAL CONFIGURATION
# ============================================================================


class EmbeddingConfig(BaseModel):
    """
    Vector generation behaviour.

    NOT identity-affecting. Replacing an embedding model invalidates every
    vector while leaving every node boundary, span, and identifier correct, so
    nothing here contributes to the segmentation fingerprint. A corpus may be
    re-embedded without reprocessing.

    Attributes
    ----------
    enabled:
        Whether vectors are generated. When disabled the module produces nodes
        alone, which a deployment embedding separately or later may prefer.
    provider:
        Identifier of the embedding provider.
    composition:
        Identifier of the rule composing embedding input from a node's fields.
        Recorded on every vector, since a node embedded with its heading path
        is not comparable to the same node embedded without it.
    batch_size:
        Nodes submitted to the provider in one call. Bounded further by the
        provider's own declared maximum.
    flush_deadline_seconds:
        Longest a partial batch waits before being submitted. Without it, the
        final nodes of a run would wait indefinitely for a batch that will
        never fill.
    timeout_seconds:
        Deadline for one embedding call.
    max_retries:
        Attempts permitted for a transient provider failure, inclusive of the
        first.
    max_input_chars:
        Longest embedding input permitted. Advisory: a provider declaring a
        smaller maximum takes precedence, since it knows its own model.
    require_normalized:
        Whether a provider returning unnormalized vectors is rejected. Enabled
        by default, because a store mixing normalized and unnormalized vectors
        produces comparisons that are silently meaningless.
    provider_options:
        Settings passed verbatim to the configured provider's factory.

        Exists because the parameters that determine what a vector MEANS, such
        as which model is loaded and which prefixes it expects, are
        provider-specific and cannot be enumerated in a domain-neutral
        contract. Naming them here as first-class fields would require the
        Core to know which providers exist and what each accepts, so adding a
        provider would mean editing this file.

        Values are strings, so that a setting supplied in code and the same
        setting supplied through an environment variable are indistinguishable
        by the time a factory sees them. A factory is responsible for
        interpreting and validating its own options; an unrecognized key is
        that factory's error to raise, not this contract's.

        NOT identity-affecting. Changing a model invalidates vectors while
        leaving every node boundary, span, and identifier correct.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", validate_default=True)

    enabled: bool = True
    provider: str = Field(default="null", min_length=1)
    composition: str = Field(default="heading_path_prefixed", min_length=1)
    batch_size: int = Field(default=32, ge=1, le=1024)
    flush_deadline_seconds: float = Field(default=2.0, gt=0.0)
    timeout_seconds: float = Field(default=120.0, gt=0.0)
    max_retries: int = Field(default=3, ge=1, le=10)
    max_input_chars: int = Field(default=8_192, ge=1)
    require_normalized: bool = True
    provider_options: Mapping[str, str] = Field(default_factory=dict)


class EnrichmentConfig(BaseModel):
    """
    Domain metadata enrichment behaviour.

    NOT identity-affecting. Enrichment is additive and contributes metadata
    only; it cannot alter a node's text, span, or identifier, so nothing here
    reaches the segmentation fingerprint. Adding an enricher must never
    invalidate a corpus.

    Attributes
    ----------
    enabled:
        Whether enrichment runs at all.
    enabled_enrichers:
        Enricher names to run. An empty set runs every registered enricher,
        which is the ordinary case; naming a subset is for staged rollout.
    timeout_seconds:
        Deadline for one enricher on one node.
    failure_policy:
        How an enricher failure is treated.
    max_metadata_keys:
        Largest number of metadata keys permitted on a node. Bounds a
        defective enricher, which could otherwise attach unbounded metadata to
        every node in a corpus.
    max_metadata_value_chars:
        Longest metadata value permitted. Metadata is for attributes, not for
        content: a large value indicates an enricher attempting to store text,
        which belongs in the node.
    max_concurrent_nodes:
        Nodes enriched concurrently within one document.

        Enrichment is per-node independent and ordinarily crosses a network or
        database boundary, so running it one node at a time would serialize a
        large document's round trips inside a single document slot while the
        run's concurrency limit held other documents back. Bounded rather than
        unlimited, because an enricher's backing service has its own capacity
        and a thousand-node document should not present a thousand
        simultaneous requests to it.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", validate_default=True)

    enabled: bool = True
    enabled_enrichers: frozenset[str] = frozenset()
    timeout_seconds: float = Field(default=30.0, gt=0.0)
    failure_policy: EnrichmentFailurePolicy = EnrichmentFailurePolicy.DEGRADE
    max_metadata_keys: int = Field(default=64, ge=1)
    max_metadata_value_chars: int = Field(default=1_024, ge=1)
    max_concurrent_nodes: int = Field(default=8, ge=1, le=256)


class RelationConfig(BaseModel):
    """
    Structural relation derivation.

    NOT identity-affecting. Relations reference nodes; they do not determine
    where a node begins or ends.

    Attributes
    ----------
    enabled:
        Whether relations are derived.
    derive_sequence:
        Whether reading-order relations between adjacent nodes are derived.
    derive_hierarchy:
        Whether section containment relations are derived from the heading
        structure.
    derive_co_membership:
        Whether shared-table, shared-list, and shared-page relations are
        derived.
    emit_inverse_relations:
        Whether both directions of a directed relation are emitted.

        Disabled by default. Storing both doubles the relation count for no
        additional information, since a consumer can invert a relation
        directly. Enable where a sink cannot traverse an edge backwards.
    max_relations_per_node:
        Largest number of outgoing relations permitted for one node. Bounds a
        pathological document, such as one table with tens of thousands of
        rows, whose co-membership relations would otherwise grow
        quadratically.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", validate_default=True)

    enabled: bool = True
    derive_sequence: bool = True
    derive_hierarchy: bool = True
    derive_co_membership: bool = True
    emit_inverse_relations: bool = False
    max_relations_per_node: int = Field(default=256, ge=1)


class ValidationConfig(BaseModel):
    """
    Verification of produced nodes against their source documents.

    NOT identity-affecting. Validation observes what segmentation produced; it
    does not decide where a boundary falls.

    Attributes
    ----------
    verify_provenance:
        Whether every node's text is verified to be the exact substring of its
        source document at its span.

        Enabled by default and should remain so. This check is the runtime
        guarantee that a claim verified against a node is traceable to an
        exact region of an identified source document; disabling it makes the
        guarantee aspirational.
    require_full_coverage:
        Whether every non-whitespace character of a document must belong to
        exactly one node.

        Enabled by default. Text absent from every node cannot be retrieved,
        so a claim it would have supported becomes unverifiable, and the
        failure is invisible: the run reports success while the evidence is
        silently missing.
    verify_block_alignment:
        Whether each node's span is verified to be consistent with the blocks
        it claims to cover. Catches a strategy whose span and block references
        disagree.
    max_nodes_per_document:
        Largest number of nodes one document may produce. A document yielding
        millions of nodes is a fault, not a corpus.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", validate_default=True)

    verify_provenance: bool = True
    require_full_coverage: bool = True
    verify_block_alignment: bool = True
    max_nodes_per_document: int = Field(default=100_000, ge=1)


class SinkConfig(BaseModel):
    """
    Behaviour when writing artifacts to a destination.

    Attributes
    ----------
    flush_interval_units:
        Units written between flushes. A sink buffering internally is given a
        bounded opportunity to make its writes durable.
    write_timeout_seconds:
        Deadline for one write.
    max_retries:
        Attempts permitted for a transient sink failure, inclusive of the
        first.
    fail_run_on_sink_error:
        Whether a sink failure aborts the run rather than failing the
        document.

        Enabled by default. Unlike a malformed document, an unavailable sink
        will fail identically for every subsequent document, so continuing
        would produce a long run that discards everything it computes.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", validate_default=True)

    flush_interval_units: int = Field(default=100, ge=1)
    write_timeout_seconds: float = Field(default=60.0, gt=0.0)
    max_retries: int = Field(default=3, ge=1, le=10)
    fail_run_on_sink_error: bool = True


class TelemetryConfig(BaseModel):
    """
    Observability behaviour.

    Attributes
    ----------
    emit_metrics:
        Whether operational counters and observations are emitted.
    log_level:
        Minimum severity emitted by the module logger.
    log_node_events:
        Whether a per-node event is logged. Informative on a small corpus,
        overwhelming on a large one, since nodes outnumber documents by an
        order of magnitude.
    progress_interval_documents:
        Documents between progress events.

    Raises
    ------
    ValueError
        If the log level is not recognized.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", validate_default=True)

    emit_metrics: bool = True
    log_level: str = "INFO"
    log_node_events: bool = False
    progress_interval_documents: int = Field(default=100, ge=1)

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


class KnowledgeConfig(BaseSettings):
    """
    Complete operational configuration of the Knowledge Processing Module.

    Constructed once at the composition root and injected into the engine and
    every stage. Nothing constructs its own configuration, and no component
    reads environment variables directly.

    Environment layering uses the prefix ``PTFM_KNOWLEDGE_`` with ``__``
    separating nesting levels, for example
    ``PTFM_KNOWLEDGE_CHUNKING__MAX_CHUNK_CHARS``.

    Attributes
    ----------
    max_concurrent_documents:
        Documents processed concurrently. The primary control over both
        throughput and peak memory.
    processing_thread_limit:
        Worker threads available for blocking segmentation and assembly.
        ``None`` derives a value from the host CPU count.
    fail_fast:
        Whether the first document failure aborts the run. Disabled by
        default: a malformed document late in a corpus must not discard the
        preceding work.
    failure_ratio_threshold:
        Proportion of failed documents above which a run is aborted as
        systemically bad. An operational guard over run health, carrying no
        truth judgement and never influencing verification.
    failure_ratio_min_samples:
        Documents that must be processed before the ratio guard may fire.
        Without it, a run aborts when the first of one document fails.
    chunking:
        Boundary-affecting parameters. IDENTITY-AFFECTING; see
        :class:`ChunkingConfig`.
    embedding:
        Vector generation.
    enrichment:
        Domain metadata enrichment.
    relations:
        Structural relation derivation.
    validation:
        Provenance verification.
    sink:
        Artifact persistence.
    telemetry:
        Observability.

    Raises
    ------
    ValueError
        If the embedding input bound could not accommodate the largest node
        the chunking configuration may produce.
    """

    model_config = SettingsConfigDict(
        env_prefix="PTFM_KNOWLEDGE_",
        env_nested_delimiter="__",
        env_file=".env",
        env_file_encoding="utf-8",
        frozen=True,
        extra="forbid",
        validate_default=True,
    )

    max_concurrent_documents: int = Field(default=8, ge=1, le=1024)
    processing_thread_limit: Optional[int] = Field(default=None, ge=1, le=1024)
    fail_fast: bool = False
    failure_ratio_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    failure_ratio_min_samples: int = Field(default=20, ge=1)

    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    enrichment: EnrichmentConfig = Field(default_factory=EnrichmentConfig)
    relations: RelationConfig = Field(default_factory=RelationConfig)
    validation: ValidationConfig = Field(default_factory=ValidationConfig)
    sink: SinkConfig = Field(default_factory=SinkConfig)
    telemetry: TelemetryConfig = Field(default_factory=TelemetryConfig)

    @model_validator(mode="after")
    def _validate_embedding_capacity(self) -> KnowledgeConfig:
        """
        Ensure embedding input can accommodate the largest possible node.

        Composition may prepend a heading path, so the input is at least as
        long as the node text. A bound smaller than the largest node
        guarantees that some node fails composition, and doing so only once a
        sufficiently large node appears would make the failure look like a
        corpus problem rather than a configuration one.

        Returns
        -------
        KnowledgeConfig
            The validated instance.

        Raises
        ------
        ValueError
            If the embedding bound is smaller than the maximum node size.
        """
        if not self.embedding.enabled:
            return self

        if self.embedding.max_input_chars < self.chunking.max_chunk_chars:
            raise ValueError(
                f"embedding.max_input_chars "
                f"({self.embedding.max_input_chars}) is smaller than "
                f"chunking.max_chunk_chars ({self.chunking.max_chunk_chars}); "
                f"a node at the maximum size could never be embedded"
            )
        return self

    # ------------------------------------------------------------------
    # Segmentation identity
    # ------------------------------------------------------------------

    def segmentation_fingerprint(
        self,
        *runtime_components: Mapping[str, str],
    ) -> str:
        """
        Compute the fingerprint on which node identity depends.

        Composed from the configured boundary-affecting parameters together
        with any runtime components supplied. The composition root supplies
        the strategy's own parameters and the sentence segmenter's version,
        neither of which configuration can know: a segmenter upgrade shifts
        boundaries within oversized blocks while every configured value stays
        the same.

        Only :attr:`chunking` contributes from configuration. Embedding,
        enrichment, relations, validation, concurrency, and telemetry never
        do, because none of them can change where a node begins or ends.

        Parameters
        ----------
        *runtime_components:
            Additional boundary-affecting components declared by injected
            implementations.

        Returns
        -------
        str
            A versioned digest, disjoint across differing configurations.

        Raises
        ------
        KnowledgeConfigurationError
            If two sources contribute the same component with differing
            values.
        """
        return compute_segmentation_fingerprint(
            self.chunking.boundary_components(), *runtime_components
        )

    def boundary_components(self) -> Mapping[str, str]:
        """
        Report the configured components contributing to node identity.

        Exposed so that a fingerprint mismatch can be diagnosed by comparing
        components rather than by observing that two opaque digests differ.

        Returns
        -------
        Mapping[str, str]
            The chunking parameters, namespaced.
        """
        return self.chunking.boundary_components()

    # ------------------------------------------------------------------
    # Derived settings
    # ------------------------------------------------------------------

    def resolve_thread_limit(self) -> int:
        """
        Determine the worker thread count for blocking work.

        Returns the configured value when set. Otherwise derives one from the
        host CPU count: segmentation and assembly are CPU-bound with no I/O,
        so oversubscription would only add scheduling overhead.

        Returns
        -------
        int
            Thread limit of at least one.
        """
        if self.processing_thread_limit is not None:
            return self.processing_thread_limit
        return max(1, os.cpu_count() or _FALLBACK_CPU_COUNT)

    def resolve_batch_size(self, provider_maximum: int) -> int:
        """
        Determine the embedding batch size for a provider.

        The provider's own maximum takes precedence where it is smaller, since
        a provider knows its own limits and exceeding them would fail every
        batch rather than merely being inefficient.

        Parameters
        ----------
        provider_maximum:
            Largest batch the provider accepts.

        Returns
        -------
        int
            Batch size of at least one.

        Raises
        ------
        KnowledgeConfigurationError
            If the provider declares a non-positive maximum.
        """
        if provider_maximum < 1:
            raise KnowledgeConfigurationError(
                f"Embedding provider declares a maximum batch size of "
                f"{provider_maximum}, which is not usable.",
                context={"provider_maximum": provider_maximum},
            )
        return min(self.embedding.batch_size, provider_maximum)

    def enricher_is_enabled(self, name: str) -> bool:
        """
        Report whether a named enricher should run.

        An empty selection runs every registered enricher, which is the
        ordinary case; naming a subset is for staged rollout.

        Parameters
        ----------
        name:
            Enricher identifier.

        Returns
        -------
        bool
            ``True`` when the enricher should run.
        """
        if not self.enrichment.enabled:
            return False
        if not self.enrichment.enabled_enrichers:
            return True
        return name in self.enrichment.enabled_enrichers

    def with_overrides(self, **overrides: object) -> KnowledgeConfig:
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
        KnowledgeConfig
            The revalidated configuration.

        Raises
        ------
        KnowledgeConfigurationError
            If the merged configuration is invalid.
        """
        merged = {**self.model_dump(), **overrides}
        try:
            return KnowledgeConfig(**merged)  # type: ignore[arg-type]
        except ValueError as exc:
            raise KnowledgeConfigurationError(
                f"Invalid knowledge processing configuration override: {exc}"
            ) from exc
