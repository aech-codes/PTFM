"""
PTFM Data Ingestion Configuration

File:
    ptfm/core/ingestion/config.py

Description:
    Declares every operational parameter of the Data Ingestion Module.
    This module is the SOLE location in which ingestion numeric values are
    defined. No stage, connector, or extractor may contain a numeric
    literal compared against a limit; every such value is injected from
    here.

Scope prohibition
-----------------
    The values declared here are OPERATIONAL limits: byte sizes, timeouts,
    concurrency bounds, retry counts, buffer sizes. None of them is a
    truth, verification, or calibration threshold, and none may ever
    influence whether a claim is accepted, whether evidence is trusted, or
    what confidence a response carries.

    Verification thresholds originate exclusively from the conformal
    calibration layer. A confidence threshold introduced into this file
    would place a hardcoded truth judgement in the ingestion path, which
    is prohibited.

    The single field that observes a model score,
    ``MetadataConfig.language_detection_min_confidence``, governs ONLY
    whether a diagnostic warning is attached. It never gates, filters,
    reorders, or discards content.

Environment layering
--------------------
    Every value may be supplied by environment variable using the prefix
    ``PTFM_INGESTION_`` with ``__`` separating nesting levels::

        PTFM_INGESTION_MAX_CONCURRENT_DOCUMENTS=32
        PTFM_INGESTION_ACQUISITION__MAX_DOCUMENT_BYTES=209715200
        PTFM_INGESTION_RETRY__MAX_ATTEMPTS=5

Dependency contract
-------------------
    Imports only the standard library, Pydantic, pydantic-settings, and
    ``ptfm.core.ingestion.exceptions``. It must not import ``interfaces``,
    ``models``, or any stage, connector, or extractor.
"""

from __future__ import annotations

import os
import random
import unicodedata
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Callable, Final, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from ptfm.core.ingestion.exceptions import IngestionConfigurationError

__all__ = [
    "KIB",
    "MIB",
    "TextNormalizationForm",
    "RetryConfig",
    "ExponentialBackoffRetryPolicy",
    "AcquisitionConfig",
    "DetectionConfig",
    "ExtractionConfig",
    "StructureConfig",
    "MetadataConfig",
    "DeduplicationConfig",
    "TelemetryConfig",
    "IngestionConfig",
]

#: Byte-size units. Declared so that size fields read as quantities rather
#: than as unexplained seven-digit integers.
KIB: Final[int] = 1024
MIB: Final[int] = 1024 * KIB

#: Fallback worker count when the host CPU count cannot be determined.
_FALLBACK_CPU_COUNT: Final[int] = 4

#: Multiplier applied to CPU count when deriving the extraction thread
#: limit. Parsing is CPU-bound but interleaves file reads, so a modest
#: oversubscription keeps cores busy without inducing thrashing.
_THREAD_OVERSUBSCRIPTION: Final[int] = 2


class TextNormalizationForm(StrEnum):
    """
    Unicode normalization form applied to document text.

    NFC is the default: it is the form assumed by the span invariants of
    ``RawDocument`` and by every downstream tokenizer, and it is stable
    under concatenation, which matters because the structure stage builds
    document text by joining block texts.
    """

    NFC = "NFC"
    NFD = "NFD"
    NFKC = "NFKC"
    NFKD = "NFKD"


# ============================================================================
# RETRY
# ============================================================================


class RetryConfig(BaseModel):
    """
    Retry schedule for transient acquisition failures.

    Governs timing only. Whether a failure is eligible for retry at all is
    determined by the ``retryable`` class attribute on the raised
    exception, so no call site inspects exception types.

    Attributes
    ----------
    max_attempts:
        Total attempts permitted, inclusive of the first. A value of one
        disables retrying.
    base_delay_seconds:
        Delay preceding the second attempt.
    backoff_multiplier:
        Factor by which the delay grows per subsequent attempt.
    max_delay_seconds:
        Ceiling on any single delay.
    jitter_ratio:
        Proportion of each delay subject to random reduction, in ``[0, 1]``.
        Jitter is not optional in practice: synchronized retries across
        concurrent workers can amplify a transient source failure into a
        sustained one.

    Raises
    ------
    ValueError
        If the maximum delay is below the base delay.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", validate_default=True)

    max_attempts: int = Field(default=3, ge=1, le=10)
    base_delay_seconds: float = Field(default=0.5, gt=0.0)
    backoff_multiplier: float = Field(default=2.0, ge=1.0)
    max_delay_seconds: float = Field(default=30.0, gt=0.0)
    jitter_ratio: float = Field(default=0.25, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _validate_delay_bounds(self) -> RetryConfig:
        """
        Ensure the delay ceiling does not invert the backoff curve.

        Returns
        -------
        RetryConfig
            The validated instance.

        Raises
        ------
        ValueError
            If ``max_delay_seconds`` is below ``base_delay_seconds``.
        """
        if self.max_delay_seconds < self.base_delay_seconds:
            raise ValueError(
                f"max_delay_seconds ({self.max_delay_seconds}) must not be "
                f"below base_delay_seconds ({self.base_delay_seconds}); a "
                f"lower ceiling silently inverts the backoff curve"
            )
        return self


@dataclass(frozen=True, slots=True)
class ExponentialBackoffRetryPolicy:
    """
    Jittered exponential backoff satisfying the ``RetryPolicy`` protocol.

    Co-located with :class:`RetryConfig` because the policy is pure
    computation over those values with no I/O, so keeping parameters and
    their interpretation together avoids a second source of truth.

    The random source is injected rather than referenced globally. Jitter
    is required for correctness in production and would otherwise make
    every delay assertion in the test suite nondeterministic.

    Attributes
    ----------
    config:
        The schedule parameters.
    rng:
        Callable returning a value in ``[0, 1)``. Defaults to
        :func:`random.random`; tests supply a fixed value.
    """

    config: RetryConfig
    rng: Callable[[], float] = field(default=random.random)

    @property
    def max_attempts(self) -> int:
        """
        Total attempts permitted, inclusive of the first.

        Returns
        -------
        int
            The configured attempt limit.
        """
        return self.config.max_attempts

    def delay_for(self, attempt: int) -> float:
        """
        Compute the delay preceding a given attempt.

        The delay grows geometrically from the base delay, is clamped to
        the configured ceiling, and is then reduced by a random fraction
        of itself bounded by the jitter ratio.

        Parameters
        ----------
        attempt:
            One-based index of the attempt about to be made. The first
            attempt is never delayed.

        Returns
        -------
        float
            Non-negative delay in seconds.

        Raises
        ------
        ValueError
            If ``attempt`` is below one.
        """
        if attempt < 1:
            raise ValueError(f"attempt must be at least 1, got {attempt}")
        if attempt == 1:
            return 0.0

        growth = self.config.backoff_multiplier ** (attempt - 2)
        delay = min(
            self.config.base_delay_seconds * growth,
            self.config.max_delay_seconds,
        )
        reduction = delay * self.config.jitter_ratio * self.rng()
        return max(0.0, delay - reduction)


# ============================================================================
# STAGE CONFIGURATION
# ============================================================================


class AcquisitionConfig(BaseModel):
    """
    Payload transfer limits and buffering behaviour.

    Attributes
    ----------
    max_document_bytes:
        Largest payload accepted. Enforced before transfer completes
        wherever the source declares a size, so an oversized document is
        rejected without being downloaded.
    spool_threshold_bytes:
        Payloads at or above this size are spooled to temporary storage
        rather than held in memory. This is what keeps peak memory flat
        across a corpus of arbitrary size.
    read_chunk_bytes:
        Transfer chunk size. Also the granularity of incremental hashing,
        so spooled payloads are never materialized solely to be digested.
    fetch_timeout_seconds:
        Deadline for transferring one payload.
    discovery_timeout_seconds:
        Deadline for a complete discovery pass.
    spool_directory:
        Directory for spooled payloads. ``None`` uses the system default.

    Raises
    ------
    ValueError
        If the spool threshold exceeds the maximum document size, or the
        read chunk exceeds the spool threshold.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", validate_default=True)

    max_document_bytes: int = Field(default=100 * MIB, ge=1)
    spool_threshold_bytes: int = Field(default=8 * MIB, ge=1)
    read_chunk_bytes: int = Field(default=64 * KIB, ge=1)
    fetch_timeout_seconds: float = Field(default=120.0, gt=0.0)
    discovery_timeout_seconds: float = Field(default=600.0, gt=0.0)
    spool_directory: Optional[str] = None

    @model_validator(mode="after")
    def _validate_size_ordering(self) -> AcquisitionConfig:
        """
        Ensure the size limits form a coherent hierarchy.

        Returns
        -------
        AcquisitionConfig
            The validated instance.

        Raises
        ------
        ValueError
            If the thresholds are ordered such that spooling could never
            engage, or a single read would exceed the spool threshold.
        """
        if self.spool_threshold_bytes > self.max_document_bytes:
            raise ValueError(
                f"spool_threshold_bytes ({self.spool_threshold_bytes}) exceeds "
                f"max_document_bytes ({self.max_document_bytes}); spooling "
                f"could never engage"
            )
        if self.read_chunk_bytes > self.spool_threshold_bytes:
            raise ValueError(
                f"read_chunk_bytes ({self.read_chunk_bytes}) exceeds "
                f"spool_threshold_bytes ({self.spool_threshold_bytes}); a "
                f"single read would defeat the memory bound spooling provides"
            )
        return self


class DetectionConfig(BaseModel):
    """
    Media type and encoding resolution behaviour.

    Attributes
    ----------
    header_bytes:
        Number of leading payload bytes passed to the format detector.
        Sufficient for every magic-byte signature in use while remaining
        cheap to read from a spooled payload.
    trust_declared_media_type:
        Whether a source-declared type may be selected when magic bytes
        yield nothing. Disable when ingesting from sources whose declared
        types are known to be unreliable, such as misconfigured web hosts.
    encoding_sample_bytes:
        Number of bytes sampled for charset detection.
    max_replacement_ratio:
        Proportion of replacement characters above which decoding is
        treated as failed rather than merely lossy. An operational data
        quality limit, not a confidence threshold.
    default_text_encoding:
        Encoding assumed when detection yields no answer for a text
        payload.

    Raises
    ------
    ValueError
        If the default encoding is not a recognized codec.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", validate_default=True)

    header_bytes: int = Field(default=8 * KIB, ge=16)
    trust_declared_media_type: bool = True
    encoding_sample_bytes: int = Field(default=64 * KIB, ge=64)
    max_replacement_ratio: float = Field(default=0.05, ge=0.0, le=1.0)
    default_text_encoding: str = "utf-8"

    @model_validator(mode="after")
    def _validate_encoding(self) -> DetectionConfig:
        """
        Ensure the fallback encoding is a codec Python can resolve.

        Returns
        -------
        DetectionConfig
            The validated instance.

        Raises
        ------
        ValueError
            If the codec is unknown.
        """
        try:
            "probe".encode(self.default_text_encoding)
        except LookupError as exc:
            raise ValueError(
                f"default_text_encoding {self.default_text_encoding!r} is not a "
                f"recognized codec"
            ) from exc
        return self


class ExtractionConfig(BaseModel):
    """
    Content extraction limits.

    Attributes
    ----------
    timeout_seconds:
        Deadline for parsing one document. Bounds the blast radius of
        parser pathologies such as decompression bombs.
    max_blocks:
        Largest block count accepted from an extractor. A pathological
        document producing millions of blocks is a fault, not a corpus.
    max_text_chars:
        Largest normalized text length accepted.
    disabled_extractors:
        Extractor names to exclude from registration. Deliberately a
        deny-list: an allow-list leaves every newly added extractor
        silently inert until someone remembers to enable it, and that
        failure is invisible.
    strict_availability:
        Whether a missing optional dependency aborts startup. When
        disabled, the affected extractor is skipped and a warning is
        logged, which suits deployments that ingest only a subset of
        formats.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", validate_default=True)

    timeout_seconds: float = Field(default=300.0, gt=0.0)
    max_blocks: int = Field(default=1_000_000, ge=1)
    max_text_chars: int = Field(default=50_000_000, ge=1)
    disabled_extractors: frozenset[str] = frozenset()
    strict_availability: bool = True


class StructureConfig(BaseModel):
    """
    Canonical text and block tree construction.

    Attributes
    ----------
    normalization_form:
        Unicode normalization applied to document text.
    block_separator:
        String inserted between sibling block texts when assembling the
        document. Must be whitespace: the ``RawDocument`` span invariants
        permit only whitespace between leaf spans, so a non-whitespace
        separator would produce content outside the block tree and fail
        validation.
    collapse_whitespace:
        Whether runs of intra-block whitespace are collapsed. Performed
        only in an offset-preserving manner so that spans remain valid.
    strip_block_text:
        Whether leading and trailing whitespace is removed from block text
        before spans are computed.
    max_block_depth:
        Deepest nesting accepted. Guards against pathological markup.
    validate_span_integrity:
        Whether span invariants are verified after assembly. Enabled by
        default and should remain so: this check is the runtime guarantee
        that evidence remains traceable to an exact source location.

    Raises
    ------
    ValueError
        If the separator is not whitespace, or the normalization form is
        not supported by the platform.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", validate_default=True)

    normalization_form: TextNormalizationForm = TextNormalizationForm.NFC
    block_separator: str = "\n\n"
    collapse_whitespace: bool = True
    strip_block_text: bool = True
    max_block_depth: int = Field(default=64, ge=1)
    validate_span_integrity: bool = True

    @model_validator(mode="after")
    def _validate_separator(self) -> StructureConfig:
        """
        Ensure the block separator cannot violate the span invariants.

        Returns
        -------
        StructureConfig
            The validated instance.

        Raises
        ------
        ValueError
            If the separator is empty, contains non-whitespace, or contains
            a carriage return.
        """
        if not self.block_separator:
            raise ValueError("block_separator must not be empty")
        if not self.block_separator.isspace():
            raise ValueError(
                f"block_separator {self.block_separator!r} must contain only "
                f"whitespace; RawDocument permits only whitespace between leaf "
                f"spans, so any other separator would place content outside "
                f"the block tree"
            )
        if "\r" in self.block_separator:
            raise ValueError(
                "block_separator must not contain carriage returns; "
                "RawDocument.text requires normalized line endings"
            )
        return self

    @model_validator(mode="after")
    def _validate_normalization_support(self) -> StructureConfig:
        """
        Ensure the requested normalization form is available.

        Returns
        -------
        StructureConfig
            The validated instance.

        Raises
        ------
        ValueError
            If the platform cannot apply the form.
        """
        try:
            unicodedata.normalize(self.normalization_form.value, "probe")
        except ValueError as exc:
            raise ValueError(
                f"normalization_form {self.normalization_form.value!r} is not "
                f"supported on this platform"
            ) from exc
        return self


class MetadataConfig(BaseModel):
    """
    Generic metadata harvesting behaviour.

    Attributes
    ----------
    detect_language:
        Whether natural language identification runs.
    language_detection_sample_chars:
        Leading characters sampled for detection. Detection over an entire
        large document is wasteful and no more accurate.
    language_detection_min_chars:
        Shortest text on which detection is attempted. Below this length
        detection is unreliable enough to be misleading.
    language_detection_min_confidence:
        Detector self-reported certainty below which a diagnostic warning
        is attached.

        PROHIBITION. This governs ONLY whether an
        ``ExtractionWarning`` is recorded. It must never gate, filter,
        reorder, or discard content, must never reach the verification,
        calibration, or decision path, and must never be persisted as a
        field on ``RawDocument`` or ``DocumentMetadata``. It concerns which
        human language a string is written in, not whether anything is
        true.
    max_title_chars:
        Longest declared title retained. Some formats emit an entire first
        page as a title.
    max_authors:
        Largest number of declared authors retained.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", validate_default=True)

    detect_language: bool = True
    language_detection_sample_chars: int = Field(default=4 * KIB, ge=1)
    language_detection_min_chars: int = Field(default=32, ge=1)
    language_detection_min_confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    max_title_chars: int = Field(default=1024, ge=1)
    max_authors: int = Field(default=64, ge=1)

    @model_validator(mode="after")
    def _validate_sample_bounds(self) -> MetadataConfig:
        """
        Ensure the detection sample is not smaller than the minimum input.

        Returns
        -------
        MetadataConfig
            The validated instance.

        Raises
        ------
        ValueError
            If the sample size is below the minimum input length, which
            would make detection unreachable.
        """
        if self.language_detection_sample_chars < self.language_detection_min_chars:
            raise ValueError(
                f"language_detection_sample_chars "
                f"({self.language_detection_sample_chars}) is below "
                f"language_detection_min_chars "
                f"({self.language_detection_min_chars}); detection could never "
                f"run"
            )
        return self


class DeduplicationConfig(BaseModel):
    """
    Duplicate detection behaviour.

    Encountering a duplicate is a normal outcome of incremental
    re-ingestion, reported as a skip rather than a failure.

    Attributes
    ----------
    enabled:
        Whether exact duplicate detection runs.
    detect_near_duplicates:
        Whether near-duplicate detection runs. Near duplicates are
        reported as a warning and still ingested; discarding them is a
        corpus policy decision, not an ingestion one.
    simhash_bits:
        Fingerprint width for near-duplicate detection.
    near_duplicate_max_distance:
        Largest Hamming distance treated as a near duplicate. An
        operational similarity limit over text fingerprints; it carries no
        truth judgement.
    skip_unchanged_versions:
        Whether records whose source version tag is unchanged are skipped
        without transfer. The mechanism that makes re-ingestion cheap.

    Raises
    ------
    ValueError
        If the near-duplicate distance is not below the fingerprint width.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", validate_default=True)

    enabled: bool = True
    detect_near_duplicates: bool = False
    simhash_bits: int = Field(default=64, ge=32, le=256)
    near_duplicate_max_distance: int = Field(default=3, ge=0)
    skip_unchanged_versions: bool = True

    @model_validator(mode="after")
    def _validate_distance_bound(self) -> DeduplicationConfig:
        """
        Ensure the similarity bound is meaningful for the fingerprint width.

        Returns
        -------
        DeduplicationConfig
            The validated instance.

        Raises
        ------
        ValueError
            If the distance is not strictly below the fingerprint width,
            which would classify every document as a near duplicate.
        """
        if self.near_duplicate_max_distance >= self.simhash_bits:
            raise ValueError(
                f"near_duplicate_max_distance "
                f"({self.near_duplicate_max_distance}) must be below "
                f"simhash_bits ({self.simhash_bits}); otherwise every document "
                f"matches every other"
            )
        return self


class TelemetryConfig(BaseModel):
    """
    Observability behaviour.

    Attributes
    ----------
    emit_metrics:
        Whether operational counters and observations are emitted.
    log_level:
        Minimum severity emitted by the ingestion logger.
    log_document_events:
        Whether a per-document event is logged. Informative on a small
        corpus, overwhelming on a large one.
    progress_interval:
        Number of documents between progress events. Progress logging is
        how a long run remains observable without per-document noise.

    Raises
    ------
    ValueError
        If the log level is not recognized.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", validate_default=True)

    emit_metrics: bool = True
    log_level: str = "INFO"
    log_document_events: bool = False
    progress_interval: int = Field(default=100, ge=1)

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
                f"log_level {self.log_level!r} is not one of "
                f"{sorted(permitted)}"
            )
        return self


# ============================================================================
# ROOT CONFIGURATION
# ============================================================================


class IngestionConfig(BaseSettings):
    """
    Complete operational configuration of the Data Ingestion Module.

    Constructed once at the composition root and injected into the engine
    and every stage. Nothing constructs its own configuration, and no
    component reads environment variables directly.

    Environment layering uses the prefix ``PTFM_INGESTION_`` with ``__``
    separating nesting levels, for example
    ``PTFM_INGESTION_ACQUISITION__MAX_DOCUMENT_BYTES``.

    Attributes
    ----------
    max_concurrent_documents:
        Documents processed concurrently. The primary control over both
        throughput and peak memory.
    extraction_thread_limit:
        Worker threads available for blocking parser execution. ``None``
        derives a value from the host CPU count, which is the only correct
        default across a two-core container and a sixty-four-core host.
    fail_fast:
        Whether the first document failure aborts the run. Disabled by
        default: a corrupt document at position forty thousand must not
        discard the preceding work.
    failure_ratio_threshold:
        Proportion of failed documents above which a run is aborted as
        systemically bad. An operational guard over run health; it carries
        no truth judgement and never influences verification.
    failure_ratio_min_samples:
        Documents that must be processed before the ratio guard may fire.
        Without this, a run aborts when the first of one document fails.
    quarantine_failures:
        Whether failure records are retained for inspection.
    retry:
        Retry schedule for transient acquisition failures.
    acquisition:
        Payload transfer limits.
    detection:
        Media type and encoding resolution.
    extraction:
        Content extraction limits.
    structure:
        Text and block tree construction.
    metadata:
        Generic metadata harvesting.
    deduplication:
        Duplicate detection.
    telemetry:
        Observability.

    Raises
    ------
    ValueError
        If concurrency values are incoherent with one another.
    """

    model_config = SettingsConfigDict(
        env_prefix="PTFM_INGESTION_",
        env_nested_delimiter="__",
        env_file=".env",
        env_file_encoding="utf-8",
        frozen=True,
        extra="forbid",
        validate_default=True,
    )

    max_concurrent_documents: int = Field(default=8, ge=1, le=1024)
    extraction_thread_limit: Optional[int] = Field(default=None, ge=1, le=1024)
    fail_fast: bool = False
    failure_ratio_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    failure_ratio_min_samples: int = Field(default=20, ge=1)
    quarantine_failures: bool = True

    retry: RetryConfig = Field(default_factory=RetryConfig)
    acquisition: AcquisitionConfig = Field(default_factory=AcquisitionConfig)
    detection: DetectionConfig = Field(default_factory=DetectionConfig)
    extraction: ExtractionConfig = Field(default_factory=ExtractionConfig)
    structure: StructureConfig = Field(default_factory=StructureConfig)
    metadata: MetadataConfig = Field(default_factory=MetadataConfig)
    deduplication: DeduplicationConfig = Field(default_factory=DeduplicationConfig)
    telemetry: TelemetryConfig = Field(default_factory=TelemetryConfig)

    @model_validator(mode="after")
    def _validate_detection_budget(self) -> IngestionConfig:
        """
        Ensure format detection cannot defeat the spooling memory bound.

        Returns
        -------
        IngestionConfig
            The validated instance.

        Raises
        ------
        ValueError
            If the detection header exceeds the spool threshold, which
            would force every payload into memory purely to identify it.
        """
        header = self.detection.header_bytes
        spool = self.acquisition.spool_threshold_bytes
        if header > spool:
            raise ValueError(
                f"detection.header_bytes ({header}) exceeds "
                f"acquisition.spool_threshold_bytes ({spool}); detection would "
                f"materialize payloads that spooling exists to keep out of "
                f"memory"
            )
        return self

    def resolve_thread_limit(self) -> int:
        """
        Determine the worker thread count for blocking extraction.

        Returns the configured value when set. Otherwise derives one from
        the host CPU count, modestly oversubscribed because parsing
        interleaves file reads with computation.

        Implemented as a method rather than a default factory so that the
        derivation is inspectable and does not silently differ between
        construction sites.

        Returns
        -------
        int
            Thread limit of at least one.
        """
        if self.extraction_thread_limit is not None:
            return self.extraction_thread_limit
        cpus = os.cpu_count() or _FALLBACK_CPU_COUNT
        return max(1, cpus * _THREAD_OVERSUBSCRIPTION)

    def build_retry_policy(
        self,
        rng: Optional[Callable[[], float]] = None,
    ) -> ExponentialBackoffRetryPolicy:
        """
        Construct the default retry policy from this configuration.

        Parameters
        ----------
        rng:
            Random source returning a value in ``[0, 1)``. Defaults to
            :func:`random.random`; tests supply a fixed value to make
            delays deterministic.

        Returns
        -------
        ExponentialBackoffRetryPolicy
            A policy bound to the configured schedule.
        """
        if rng is None:
            return ExponentialBackoffRetryPolicy(config=self.retry)
        return ExponentialBackoffRetryPolicy(config=self.retry, rng=rng)

    def with_overrides(self, **overrides: Any) -> IngestionConfig:
        """
        Produce a revalidated copy carrying the supplied changes.

        Uses full reconstruction rather than ``model_copy`` so that
        cross-field validators run against the merged result. A copy that
        skipped validation could hold a combination this class explicitly
        rejects.

        Parameters
        ----------
        **overrides:
            Top-level field values to replace.

        Returns
        -------
        IngestionConfig
            The revalidated configuration.

        Raises
        ------
        IngestionConfigurationError
            If the merged configuration is invalid.
        """
        merged = {**self.model_dump(), **overrides}
        try:
            return IngestionConfig(**merged)
        except ValueError as exc:
            raise IngestionConfigurationError(
                f"Invalid ingestion configuration override: {exc}",
            ) from exc