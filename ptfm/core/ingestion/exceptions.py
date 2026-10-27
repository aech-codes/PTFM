"""
PTFM Data Ingestion Exception Hierarchy

File:
    ptfm/core/ingestion/exceptions.py

Description:
    Defines every exception raised by the Data Ingestion Module, rooted in
    ``PTFMError``. The hierarchy mirrors the ingestion pipeline stages so
    that a failure can be attributed to a stage without string parsing.

Pipeline stage mapping:

    Configuration   -> IngestionConfigurationError
    Acquisition     -> SourceError, AcquisitionError
    Detection       -> FormatError
    Extraction      -> ExtractionError
    Structure       -> StructureError
    Metadata        -> MetadataExtractionError
    Deduplication   -> DeduplicationError
    Normalization   -> NormalizationError

Design Contract:
    - ``stage`` and ``retryable`` are class-level constants. Raise sites
      never supply them, which makes stage attribution unforgeable.
    - Only transport-transient failures are retryable. Parse, format,
      authentication, and size failures are permanent and route directly
      to quarantine.
    - Duplicate documents are NOT an error condition. Deduplication is a
      normal outcome of incremental re-ingestion and is represented by a
      skipped-outcome variant in ``ptfm.core.ingestion.models``.
    - This module imports only from the standard library and from
      ``ptfm.utils.exceptions``. It must remain free of third-party
      dependencies so that no parser or validation library type leaks
      into caller ``except`` clauses.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any, ClassVar, Mapping, Sequence

from ptfm.utils.exceptions import PTFMError

__all__ = [
    "IngestionStage",
    "PTFMIngestionError",
    # Configuration
    "IngestionConfigurationError",
    "ConnectorRegistrationError",
    "ExtractorRegistrationError",
    "ExtractorUnavailableError",
    # Source
    "SourceError",
    "UnsupportedSourceSchemeError",
    "SourceUnavailableError",
    "SourceAuthenticationError",
    "SourceNotFoundError",
    "SourceTimeoutError",
    # Acquisition
    "AcquisitionError",
    "DocumentTooLargeError",
    "EmptyDocumentError",
    "ChecksumMismatchError",
    # Format
    "FormatError",
    "FormatDetectionError",
    "UnsupportedMediaTypeError",
    "EncodingDetectionError",
    # Extraction
    "ExtractionError",
    "CorruptDocumentError",
    "EncryptedDocumentError",
    "ExtractionTimeoutError",
    "ExtractorInternalError",
    # Structure
    "StructureError",
    "StructureParsingError",
    "SpanIntegrityError",
    # Metadata
    "MetadataExtractionError",
    # Normalization
    "NormalizationError",
    "SchemaValidationError",
    # Deduplication
    "DeduplicationError",
    "FingerprintStoreError",
]


class IngestionStage(StrEnum):
    """
    Pipeline stage in which an ingestion failure originated.

    Used for structured logging, quarantine routing, and per-stage failure
    metrics. Declared here rather than in ``models`` because the exception
    module sits at the base of the ingestion dependency graph.
    """

    UNSPECIFIED = "UNSPECIFIED"
    CONFIGURATION = "CONFIGURATION"
    ACQUISITION = "ACQUISITION"
    DETECTION = "DETECTION"
    EXTRACTION = "EXTRACTION"
    STRUCTURE = "STRUCTURE"
    METADATA = "METADATA"
    DEDUPLICATION = "DEDUPLICATION"
    NORMALIZATION = "NORMALIZATION"


# ============================================================================
# BASE
# ============================================================================


class PTFMIngestionError(PTFMError):
    """
    Base class for all Data Ingestion Module failures.

    Attributes
    ----------
    stage:
        Pipeline stage that produced the failure. Declared as a class
        constant by each subtree.

    Parameters
    ----------
    message:
        Human-readable description of the failure.
    source_uri:
        URI of the source record being processed, when known.
    document_id:
        Canonical document identifier, when the failure occurred after
        identity derivation.
    context:
        Additional structured detail.

    Raises
    ------
    None
    """

    stage: ClassVar[IngestionStage] = IngestionStage.UNSPECIFIED

    def __init__(
        self,
        message: str,
        *,
        source_uri: str | None = None,
        document_id: str | None = None,
        context: Mapping[str, Any] | None = None,
    ) -> None:
        merged: dict[str, Any] = dict(context) if context else {}
        merged.setdefault("stage", str(self.stage))
        if source_uri is not None:
            merged.setdefault("source_uri", source_uri)
        if document_id is not None:
            merged.setdefault("document_id", document_id)
        super().__init__(message, context=merged)
        self.source_uri: str | None = source_uri
        self.document_id: str | None = document_id

    def to_dict(self) -> dict[str, Any]:
        """
        Render the failure as a structured mapping including stage detail.

        Returns
        -------
        dict[str, Any]
            The base payload extended with ``stage``, ``source_uri`` and
            ``document_id``.
        """
        payload = super().to_dict()
        payload["stage"] = str(self.stage)
        payload["source_uri"] = self.source_uri
        payload["document_id"] = self.document_id
        return payload


# ============================================================================
# CONFIGURATION
# ============================================================================


class IngestionConfigurationError(PTFMIngestionError):
    """
    Invalid or inconsistent ingestion configuration.

    Configuration failures are run-level and abort the entire ingestion
    run rather than quarantining a single document.
    """

    stage: ClassVar[IngestionStage] = IngestionStage.CONFIGURATION


class ConnectorRegistrationError(IngestionConfigurationError):
    """
    A source connector could not be registered.

    Raised on duplicate scheme registration, an invalid scheme string, or
    a factory that does not satisfy the ``SourceConnector`` contract.

    Parameters
    ----------
    message:
        Description of the registration failure.
    scheme:
        The URI scheme involved, when known.
    """

    def __init__(self, message: str, *, scheme: str | None = None, **kwargs: Any) -> None:
        super().__init__(message, **kwargs)
        self.scheme: str | None = scheme
        if scheme is not None:
            self.add_context(scheme=scheme)


class ExtractorRegistrationError(IngestionConfigurationError):
    """
    A content extractor could not be registered.

    Raised on an ambiguous priority collision for a media type, or an
    implementation that does not satisfy the ``ContentExtractor`` contract.

    Parameters
    ----------
    message:
        Description of the registration failure.
    media_type:
        The media type involved, when known.
    """

    def __init__(
        self,
        message: str,
        *,
        media_type: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(message, **kwargs)
        self.media_type: str | None = media_type
        if media_type is not None:
            self.add_context(media_type=media_type)


class ExtractorUnavailableError(IngestionConfigurationError):
    """
    An extractor is registered but its optional dependency is not installed.

    Raised at startup rather than mid-corpus, so that a missing install
    extra produces an actionable message instead of an ``ImportError``
    forty thousand documents into a run.

    Parameters
    ----------
    media_type:
        Media type the unavailable extractor would have handled.
    missing_dependency:
        Distribution name of the absent package.
    install_extra:
        Name of the packaging extra that provides the dependency.
    """

    def __init__(
        self,
        *,
        media_type: str,
        missing_dependency: str,
        install_extra: str,
        **kwargs: Any,
    ) -> None:
        message = (
            f"No extractor available for media type {media_type!r}: "
            f"required dependency {missing_dependency!r} is not installed. "
            f"Install it with the {install_extra!r} extra."
        )
        super().__init__(message, **kwargs)
        self.media_type: str = media_type
        self.missing_dependency: str = missing_dependency
        self.install_extra: str = install_extra
        self.add_context(
            media_type=media_type,
            missing_dependency=missing_dependency,
            install_extra=install_extra,
        )


# ============================================================================
# SOURCE
# ============================================================================


class SourceError(PTFMIngestionError):
    """Failure originating in the source connector transport layer."""

    stage: ClassVar[IngestionStage] = IngestionStage.ACQUISITION


class UnsupportedSourceSchemeError(SourceError):
    """
    No connector is registered for the requested URI scheme.

    Parameters
    ----------
    scheme:
        The unresolvable URI scheme.
    supported_schemes:
        Schemes currently present in the connector registry, included in
        the message to make the misconfiguration self-diagnosing.
    """

    def __init__(
        self,
        *,
        scheme: str,
        supported_schemes: Sequence[str] = (),
        **kwargs: Any,
    ) -> None:
        known = ", ".join(sorted(supported_schemes)) or "none"
        message = (
            f"No source connector registered for scheme {scheme!r}. "
            f"Registered schemes: {known}."
        )
        super().__init__(message, **kwargs)
        self.scheme: str = scheme
        self.supported_schemes: tuple[str, ...] = tuple(supported_schemes)
        self.add_context(scheme=scheme, supported_schemes=list(supported_schemes))


class SourceUnavailableError(SourceError):
    """
    The source is temporarily unreachable.

    Transient. Eligible for bounded retry with jittered backoff.
    """

    retryable: ClassVar[bool] = True


class SourceTimeoutError(SourceError):
    """
    A source operation exceeded its configured deadline.

    Transient. Eligible for bounded retry with jittered backoff.

    Parameters
    ----------
    timeout_seconds:
        The deadline that was exceeded.
    """

    retryable: ClassVar[bool] = True

    def __init__(
        self,
        message: str,
        *,
        timeout_seconds: float | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(message, **kwargs)
        self.timeout_seconds: float | None = timeout_seconds
        if timeout_seconds is not None:
            self.add_context(timeout_seconds=timeout_seconds)


class SourceAuthenticationError(SourceError):
    """
    Credentials were rejected or are insufficient for the source.

    Permanent. Retrying with identical credentials cannot succeed and
    risks triggering account lockout.
    """


class SourceNotFoundError(SourceError):
    """
    The requested source record does not exist.

    Permanent. Distinguished from :class:`SourceUnavailableError` so that
    a deleted document is quarantined immediately rather than retried.
    """


# ============================================================================
# ACQUISITION
# ============================================================================


class AcquisitionError(PTFMIngestionError):
    """Failure while acquiring or validating a document payload."""

    stage: ClassVar[IngestionStage] = IngestionStage.ACQUISITION


class DocumentTooLargeError(AcquisitionError):
    """
    A payload exceeds the configured maximum document size.

    Parameters
    ----------
    size_bytes:
        Observed or declared payload size.
    limit_bytes:
        Configured maximum permitted size.
    """

    def __init__(
        self,
        *,
        size_bytes: int,
        limit_bytes: int,
        **kwargs: Any,
    ) -> None:
        message = (
            f"Document size {size_bytes} bytes exceeds the configured "
            f"limit of {limit_bytes} bytes."
        )
        super().__init__(message, **kwargs)
        self.size_bytes: int = size_bytes
        self.limit_bytes: int = limit_bytes
        self.add_context(size_bytes=size_bytes, limit_bytes=limit_bytes)


class EmptyDocumentError(AcquisitionError):
    """
    The acquired payload contains no bytes.

    Treated as a fault rather than an empty success, because a zero-byte
    payload almost always indicates a truncated transfer or a permissions
    problem at the source.
    """


class ChecksumMismatchError(AcquisitionError):
    """
    A payload failed integrity verification against its declared digest.

    Parameters
    ----------
    expected_digest:
        Digest declared by the source.
    actual_digest:
        Digest computed over the received payload.
    algorithm:
        Name of the digest algorithm used.
    """

    def __init__(
        self,
        *,
        expected_digest: str,
        actual_digest: str,
        algorithm: str,
        **kwargs: Any,
    ) -> None:
        message = (
            f"Payload integrity check failed using {algorithm}: expected "
            f"{expected_digest}, computed {actual_digest}."
        )
        super().__init__(message, **kwargs)
        self.expected_digest: str = expected_digest
        self.actual_digest: str = actual_digest
        self.algorithm: str = algorithm
        self.add_context(
            expected_digest=expected_digest,
            actual_digest=actual_digest,
            algorithm=algorithm,
        )


# ============================================================================
# FORMAT DETECTION
# ============================================================================


class FormatError(PTFMIngestionError):
    """Failure during media type or encoding resolution."""

    stage: ClassVar[IngestionStage] = IngestionStage.DETECTION


class FormatDetectionError(FormatError):
    """
    The media type of a payload could not be determined.

    Raised only when magic bytes, declared type, and filename extension
    all fail to yield a usable answer.
    """


class UnsupportedMediaTypeError(FormatError):
    """
    The media type was detected but no extractor handles it.

    Parameters
    ----------
    media_type:
        The detected media type.
    supported_media_types:
        Media types currently present in the extractor registry.
    """

    def __init__(
        self,
        *,
        media_type: str,
        supported_media_types: Sequence[str] = (),
        **kwargs: Any,
    ) -> None:
        known = ", ".join(sorted(supported_media_types)) or "none"
        message = (
            f"No extractor registered for media type {media_type!r}. "
            f"Registered media types: {known}."
        )
        super().__init__(message, **kwargs)
        self.media_type: str = media_type
        self.supported_media_types: tuple[str, ...] = tuple(supported_media_types)
        self.add_context(
            media_type=media_type,
            supported_media_types=list(supported_media_types),
        )


class EncodingDetectionError(FormatError):
    """
    The character encoding of a text payload could not be resolved, or
    decoding produced an unacceptable proportion of replacement characters.

    Parameters
    ----------
    message:
        Description of the decoding failure.
    attempted_encodings:
        Encodings tried before giving up.
    """

    def __init__(
        self,
        message: str,
        *,
        attempted_encodings: Sequence[str] = (),
        **kwargs: Any,
    ) -> None:
        super().__init__(message, **kwargs)
        self.attempted_encodings: tuple[str, ...] = tuple(attempted_encodings)
        if attempted_encodings:
            self.add_context(attempted_encodings=list(attempted_encodings))


# ============================================================================
# EXTRACTION
# ============================================================================


class ExtractionError(PTFMIngestionError):
    """Failure while converting a payload into text and structure."""

    stage: ClassVar[IngestionStage] = IngestionStage.EXTRACTION


class CorruptDocumentError(ExtractionError):
    """
    The payload is malformed and cannot be parsed.

    Permanent. Distinguished from :class:`ExtractorInternalError` so that
    corpus quality metrics separate bad input from defective extractors.
    """


class EncryptedDocumentError(ExtractionError):
    """
    The payload is encrypted or password protected.

    Permanent under the current configuration. Surfaced distinctly so an
    operator can supply credentials rather than assume data loss.
    """


class ExtractionTimeoutError(ExtractionError):
    """
    Extraction exceeded its per-document deadline.

    Not retryable: a deterministic parser that exceeded the deadline once
    will exceed it again. The document is quarantined for inspection,
    which also bounds the blast radius of parser pathologies such as
    decompression bombs.

    Parameters
    ----------
    timeout_seconds:
        The deadline that was exceeded.
    """

    def __init__(
        self,
        message: str,
        *,
        timeout_seconds: float,
        **kwargs: Any,
    ) -> None:
        super().__init__(message, **kwargs)
        self.timeout_seconds: float = timeout_seconds
        self.add_context(timeout_seconds=timeout_seconds)


class ExtractorInternalError(ExtractionError):
    """
    A third-party parser raised an unanticipated exception.

    This is the single boundary at which library-specific exceptions are
    converted into PTFM types. The original exception is preserved as
    ``__cause__``.

    Parameters
    ----------
    message:
        Description of the failure.
    extractor_name:
        Identifier of the extractor implementation that failed.
    """

    def __init__(
        self,
        message: str,
        *,
        extractor_name: str,
        **kwargs: Any,
    ) -> None:
        super().__init__(message, **kwargs)
        self.extractor_name: str = extractor_name
        self.add_context(extractor_name=extractor_name)


# ============================================================================
# STRUCTURE
# ============================================================================


class StructureError(PTFMIngestionError):
    """Failure while building the canonical document block tree."""

    stage: ClassVar[IngestionStage] = IngestionStage.STRUCTURE


class StructureParsingError(StructureError):
    """
    Extractor output could not be normalized into a valid block tree.

    Raised on unresolvable parent references, cyclic structures, or
    non-contiguous ordinals.
    """


class SpanIntegrityError(StructureError):
    """
    Block character spans do not faithfully reconstruct the document text.

    This is a correctness violation of the Evidence-First principle: if
    spans drift, an ``EvidenceItem`` can no longer be traced back to an
    exact location in the source document. Ingestion fails the document
    rather than emitting untraceable content.

    Parameters
    ----------
    message:
        Description of the integrity violation.
    block_id:
        Identifier of the offending block, when isolable.
    expected_length:
        Length of the normalized document text.
    actual_length:
        Total length reconstructed from leaf block spans.
    """

    def __init__(
        self,
        message: str,
        *,
        block_id: str | None = None,
        expected_length: int | None = None,
        actual_length: int | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(message, **kwargs)
        self.block_id: str | None = block_id
        self.expected_length: int | None = expected_length
        self.actual_length: int | None = actual_length
        detail: dict[str, Any] = {}
        if block_id is not None:
            detail["block_id"] = block_id
        if expected_length is not None:
            detail["expected_length"] = expected_length
        if actual_length is not None:
            detail["actual_length"] = actual_length
        if detail:
            self.add_context(**detail)


# ============================================================================
# METADATA
# ============================================================================


class MetadataExtractionError(PTFMIngestionError):
    """
    Failure while harvesting generic document metadata.

    Applies only to format-generic metadata. Domain metadata is out of
    scope for the ingestion module and is produced by Knowledge Processing.
    """

    stage: ClassVar[IngestionStage] = IngestionStage.METADATA


# ============================================================================
# NORMALIZATION
# ============================================================================


class NormalizationError(PTFMIngestionError):
    """Failure while assembling the canonical ``RawDocument``."""

    stage: ClassVar[IngestionStage] = IngestionStage.NORMALIZATION


class SchemaValidationError(NormalizationError):
    """
    The assembled document failed canonical schema validation.

    Validation errors are passed as plain mappings rather than as a
    third-party validation object, so that no validation library type
    escapes into caller ``except`` clauses.

    Parameters
    ----------
    message:
        Description of the validation failure.
    validation_errors:
        Normalized validation error records, typically obtained from
        ``pydantic.ValidationError.errors()``.
    """

    def __init__(
        self,
        message: str,
        *,
        validation_errors: Sequence[Mapping[str, Any]] = (),
        **kwargs: Any,
    ) -> None:
        super().__init__(message, **kwargs)
        self.validation_errors: tuple[Mapping[str, Any], ...] = tuple(validation_errors)
        if validation_errors:
            self.add_context(
                validation_errors=[dict(item) for item in validation_errors]
            )


# ============================================================================
# DEDUPLICATION
# ============================================================================


class DeduplicationError(PTFMIngestionError):
    """
    Failure during duplicate detection.

    Note that encountering a duplicate document is NOT an error. It is the
    expected outcome of incremental re-ingestion and is represented by a
    skipped-outcome variant, not by an exception.
    """

    stage: ClassVar[IngestionStage] = IngestionStage.DEDUPLICATION


class FingerprintStoreError(DeduplicationError):
    """
    The document fingerprint store could not be read or written.

    Deliberately not retryable. A degraded fingerprint store silently
    disables duplicate detection and corrupts corpus integrity, so the
    condition must surface loudly rather than be absorbed by a retry loop.

    Parameters
    ----------
    message:
        Description of the store failure.
    operation:
        The store operation attempted, such as ``"lookup"`` or ``"record"``.
    """

    def __init__(
        self,
        message: str,
        *,
        operation: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(message, **kwargs)
        self.operation: str | None = operation
        if operation is not None:
            self.add_context(operation=operation)