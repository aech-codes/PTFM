"""
PTFM Canonical Assembly Stage

File:
    ptfm/core/ingestion/pipeline/normalization.py

Description:
    Assembles the outputs of every preceding ingestion stage into the
    canonical ``RawDocument``, which is the sole cross-module contract of the
    Data Ingestion Module.

    This is the last stage of the pipeline. Everything upstream produces
    partial results; this stage is where they become a single validated
    document, and where the document's identity is derived.

Inputs and their origins
------------------------
    AcquiredBlob        acquisition   payload digest, size, transfer instant
    DetectedFormat      detection     media type, encoding, signal warnings
    ExtractionOutput    extraction    extractor identity, extractor warnings
    NormalizedStructure structure     document text, block tree, warnings
    HarvestedMetadata   metadata      canonical metadata, warnings
    ExtractionWarning   deduplication optional near-duplicate observation

Deterministic identity
----------------------
    ``document_id`` is derived from the source URI and the payload digest via
    ``ptfm.core.schemas.derive_document_id``. It is NEVER generated randomly.

    ``IdGenerator`` is deliberately not among this stage's dependencies. The
    dependency-injection table originally listed it here, but document
    identity became a deterministic derivation, and a random identifier would
    destroy two properties the pipeline depends on:

        idempotency
            Re-ingesting identical bytes from the same URI must produce the
            same document, so that a repeated run is a no-op rather than a
            duplication.

        citation stability
            An ``EvidenceItem`` referring to a document by identifier must
            keep referring to the same document across index rebuilds.

    Omitting the generator entirely, rather than injecting it and documenting
    that it must not be used, removes the opportunity for misuse.

Two digests, two purposes
-------------------------
    content_hash
        Digest of the raw payload bytes, computed during transfer. The basis
        of document identity and of exact duplicate detection.

    text_hash
        Digest of the normalized text, computed here because this is the only
        point at which the normalized text and the hash provider are both
        available. Detects substantive change when a payload is re-encoded,
        re-compressed, or re-saved without its content actually changing.

Final validation gate
---------------------
    Constructing ``RawDocument`` re-validates every structural and span
    invariant declared in ``ptfm.core.schemas``. Those checks have already
    been performed by the structure stage, so a failure here indicates a
    defect in the pipeline itself rather than a defect in the source
    document. It is therefore reported as ``SchemaValidationError`` with the
    individual validation failures attached, and the document is quarantined
    rather than emitted with invalid spans.

Purity contract
---------------
    Synchronous, free of I/O, and holding no mutated instance state, so a
    single instance serves all concurrent workers.

Dependency contract
-------------------
    Imports only the standard library, Pydantic, ``ptfm.core.schemas`` and
    the ingestion ``exceptions``, ``models``, ``interfaces``,
    ``pipeline.structure`` and ``pipeline.metadata`` modules.
"""

from __future__ import annotations

from datetime import datetime
from typing import Final, Optional, Sequence

from pydantic import ValidationError

from ptfm.core.ingestion.exceptions import (
    NormalizationError,
    SchemaValidationError,
)
from ptfm.core.ingestion.interfaces import Clock, HashProvider
from ptfm.core.ingestion.models import AcquiredBlob, DetectedFormat
from ptfm.core.ingestion.pipeline.metadata import HarvestedMetadata
from ptfm.core.ingestion.pipeline.structure import NormalizedStructure
from ptfm.core.schemas import (
    ExtractionWarning,
    ExtractionWarningCode,
    RawDocument,
    SourceDescriptor,
    derive_document_id,
)

__all__ = ["RawDocumentNormalizer"]

#: Largest number of warnings retained on a single document.
#:
#: Detection, structure, metadata, and deduplication each contribute a
#: bounded number of observations, but extractor warnings are unbounded in
#: principle: a pathological document can produce one per malformed element.
#: A document carrying hundreds of thousands of warnings is unusable for
#: diagnosis and expensive to persist, so the list is truncated and the
#: truncation itself recorded.
#:
#: This value properly belongs in ``ExtractionConfig``. It is exposed as a
#: constructor parameter so that it remains injectable without amending
#: approved configuration.
_DEFAULT_MAX_WARNINGS: Final[int] = 1_000


class RawDocumentNormalizer:
    """
    Assembles stage outputs into the canonical ``RawDocument``.

    Parameters
    ----------
    hash_provider:
        Fingerprint provider used to digest the normalized text. Must be the
        same provider that produced the payload digest during acquisition:
        two digests on one document computed under different algorithms would
        be mutually incomparable, and the disagreement is verified here.
    clock:
        Supplies the instant recorded as ``ingested_at``. Injected so that
        assembled documents are byte-identical under test, which is what
        makes the determinism and idempotency properties assertable.
    max_warnings:
        Largest number of warnings retained. Belongs in
        ``ExtractionConfig``; exposed here so that it remains injectable
        without amending approved configuration.
    """

    __slots__ = ("_hash_provider", "_clock", "_max_warnings")

    def __init__(
        self,
        hash_provider: HashProvider,
        clock: Clock,
        *,
        max_warnings: int = _DEFAULT_MAX_WARNINGS,
    ) -> None:
        if max_warnings < 1:
            raise NormalizationError(
                f"max_warnings must be at least 1, got {max_warnings}.",
                context={"max_warnings": max_warnings},
            )
        self._hash_provider = hash_provider
        self._clock = clock
        self._max_warnings = max_warnings

    # ------------------------------------------------------------------
    # Public contract
    # ------------------------------------------------------------------

    def assemble(
        self,
        *,
        blob: AcquiredBlob,
        detected: DetectedFormat,
        structure: NormalizedStructure,
        metadata: HarvestedMetadata,
        extractor_name: Optional[str] = None,
        extraction_warnings: Sequence[ExtractionWarning] = (),
        deduplication_warning: Optional[ExtractionWarning] = None,
    ) -> RawDocument:
        """
        Produce the canonical document for one ingested payload.

        Parameters
        ----------
        blob:
            The acquired payload, supplying the content digest, the actual
            transferred size, and the transfer instant. Not read here: only
            its metadata is consulted, so an already-closed blob is
            acceptable and no I/O is performed.
        detected:
            The resolved media type and encoding, together with any signal
            disagreement warnings.
        structure:
            The normalized document text and its canonical block tree.
        metadata:
            The canonical metadata record and its warnings.
        extractor_name:
            Identifier of the extractor that produced the block tree.
            Recorded on the provenance descriptor, completing the chain from
            source bytes to canonical document: the connector determines which
            bytes entered the pipeline, the extractor determines what
            structure was derived from them. ``None`` where no extractor ran.
        extraction_warnings:
            Warnings reported by the extractor, ordinarily
            ``ExtractionOutput.warnings``. Passed separately because nothing
            else on the extraction result is needed at this point.
        deduplication_warning:
            Optional near-duplicate observation. A near duplicate is still
            ingested; whether to discard one is a corpus policy decision
            belonging to the operator.

        Returns
        -------
        RawDocument
            The validated canonical document.

        Raises
        ------
        NormalizationError
            If the payload carries no digest, the supplied metadata does not
            correspond to the supplied text, the payload digest was computed
            under a different algorithm than the injected provider, or the
            injected clock returned a naive instant.
        SchemaValidationError
            If the assembled document fails canonical validation, which
            indicates a pipeline defect rather than a defect in the source
            document.
        """
        text = structure.text

        self._verify_digest(blob)
        self._verify_algorithm(blob)
        self._verify_metadata_agreement(metadata, text, blob)
        ingested_at = self._resolve_ingestion_instant(blob)

        document_id = self._derive_identity(blob)
        text_hash = self._hash_provider.hash_text(text)

        warnings = self._merge_warnings(
            detection=detected.warnings,
            extraction=extraction_warnings,
            structure=structure.warnings,
            metadata=metadata.warnings,
            deduplication=deduplication_warning,
        )

        try:
            return RawDocument(
                document_id=document_id,
                source=self._build_source_descriptor(
                    blob, detected, extractor_name
                ),
                metadata=metadata.metadata,
                text=text,
                blocks=structure.blocks,
                content_hash=blob.content_hash,
                text_hash=text_hash,
                warnings=warnings,
                ingested_at=ingested_at,
            )
        except ValidationError as exc:
            # Reaching here means an invariant the structure stage is
            # responsible for was not upheld. The document is quarantined
            # rather than emitted with spans that cannot be trusted to
            # locate evidence in the source.
            raise SchemaValidationError(
                f"Assembled document for {blob.source_uri!r} failed canonical "
                f"validation. This indicates a pipeline defect rather than a "
                f"malformed source document.",
                validation_errors=[dict(item) for item in exc.errors()],
                source_uri=blob.source_uri,
                document_id=document_id,
                context={
                    "media_type": detected.media_type,
                    "char_count": len(text),
                    "block_count": len(structure.blocks),
                },
            ) from exc

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------

    def _derive_identity(self, blob: AcquiredBlob) -> str:
        """
        Derive the deterministic canonical identifier for a document.

        Identity is a pure function of origin and payload, so re-ingesting
        the same bytes from the same URI always produces the same identifier.

        Parameters
        ----------
        blob:
            The acquired payload.

        Returns
        -------
        str
            String form of a deterministic UUIDv5.

        Raises
        ------
        NormalizationError
            If the identifier could not be derived.
        """
        try:
            return derive_document_id(blob.source_uri, blob.content_hash)
        except ValueError as exc:
            raise NormalizationError(
                f"Canonical identifier for {blob.source_uri!r} could not be "
                f"derived: {exc}",
                source_uri=blob.source_uri,
            ) from exc

    # ------------------------------------------------------------------
    # Provenance
    # ------------------------------------------------------------------

    def _build_source_descriptor(
        self,
        blob: AcquiredBlob,
        detected: DetectedFormat,
        extractor_name: Optional[str],
    ) -> SourceDescriptor:
        """
        Construct the provenance record for an ingested document.

        The size recorded is the number of bytes actually transferred, not
        the size the source declared during discovery. A declared size is an
        assertion that may be wrong, and provenance must be truthful.

        Likewise the acquisition instant is the moment transfer completed,
        taken from the payload, rather than the moment assembly occurred.
        ``RawDocument.ingested_at`` records the latter, so the two remain
        distinguishable.

        Parameters
        ----------
        blob:
            The acquired payload.
        detected:
            The resolved media type and encoding.
        extractor_name:
            Identifier of the extractor that produced the block tree.

        Returns
        -------
        SourceDescriptor
            The provenance record.
        """
        record = blob.record
        return SourceDescriptor(
            source_uri=record.source_uri,
            scheme=record.scheme,
            connector_name=record.connector_name,
            # Records which component derived the block tree. Not inferable
            # from the media type, since resolution depends on registry
            # priority, exclusions, and optional dependency availability.
            extractor_name=extractor_name,
            acquired_at=blob.acquired_at,
            size_bytes=blob.size_bytes,
            detected_media_type=detected.media_type,
            # Advisory. Retained even when it disagrees with the detected
            # type, so that a disagreement recorded as a warning can be
            # reconciled against what the source actually claimed.
            declared_media_type=record.declared_media_type,
            encoding=detected.encoding,
            version_tag=record.version_tag,
        )

    # ------------------------------------------------------------------
    # Warnings
    # ------------------------------------------------------------------

    def _merge_warnings(
        self,
        *,
        detection: Sequence[ExtractionWarning],
        extraction: Sequence[ExtractionWarning],
        structure: Sequence[ExtractionWarning],
        metadata: Sequence[ExtractionWarning],
        deduplication: Optional[ExtractionWarning],
    ) -> tuple[ExtractionWarning, ...]:
        """
        Combine the warnings raised by every stage into one ordered sequence.

        Warnings appear in pipeline order, so that reading them top to bottom
        follows the order in which the observations were made. Ordering is
        fully determined by stage, which keeps assembly deterministic.

        Identical warnings are NOT collapsed. A repeated observation is
        itself information, such as one empty-container warning per blank
        page of a scanned document, and collapsing them would understate the
        extent of a fidelity problem.

        Parameters
        ----------
        detection:
            Warnings from format detection.
        extraction:
            Warnings from the content extractor.
        structure:
            Warnings from structure normalization.
        metadata:
            Warnings from metadata harvesting.
        deduplication:
            Optional near-duplicate observation.

        Returns
        -------
        tuple[ExtractionWarning, ...]
            The merged warnings, truncated to the configured maximum with the
            truncation itself recorded as a final warning.
        """
        merged: list[ExtractionWarning] = [
            *detection,
            *extraction,
            *structure,
            *metadata,
        ]
        if deduplication is not None:
            merged.append(deduplication)

        if len(merged) <= self._max_warnings:
            return tuple(merged)

        # One slot is reserved for the truncation notice, so that a reader is
        # never silently shown a partial list.
        retained = merged[: self._max_warnings - 1]
        retained.append(
            ExtractionWarning(
                code=ExtractionWarningCode.OTHER,
                message=(
                    f"Document produced {len(merged)} warnings; the list has "
                    f"been truncated to {self._max_warnings} entries"
                ),
                detail={
                    "total_warnings": len(merged),
                    "retained_warnings": self._max_warnings,
                },
            )
        )
        return tuple(retained)

    # ------------------------------------------------------------------
    # Verification
    # ------------------------------------------------------------------

    def _verify_digest(self, blob: AcquiredBlob) -> None:
        """
        Verify that the payload carries a usable content digest.

        The digest is the basis of document identity, so an absent one cannot
        be substituted for or recomputed here: the payload may be spooled,
        and re-reading it would be blocking I/O in a synchronous stage.

        Parameters
        ----------
        blob:
            The acquired payload.

        Returns
        -------
        None

        Raises
        ------
        NormalizationError
            If the digest is absent or blank.
        """
        if blob.content_hash.strip():
            return
        raise NormalizationError(
            f"Payload for {blob.source_uri!r} carries no content digest, so a "
            f"deterministic identifier cannot be derived.",
            source_uri=blob.source_uri,
        )

    def _verify_algorithm(self, blob: AcquiredBlob) -> None:
        """
        Verify that both digests on the document use one algorithm.

        This is a check of internal coherence, distinct from the check the
        acquisition stage performs against configuration. Acquisition
        establishes that a connector fingerprinted its payload as configured;
        this establishes that the payload digest and the text digest carried
        by the SAME document were produced by the same algorithm. A document
        holding two digests under different algorithms is incoherent
        regardless of what configuration says.

        Parameters
        ----------
        blob:
            The acquired payload.

        Returns
        -------
        None

        Raises
        ------
        NormalizationError
            If the payload digest algorithm differs from the injected
            provider's.
        """
        expected = self._hash_provider.algorithm
        if blob.hash_algorithm == expected:
            return
        raise NormalizationError(
            f"Payload for {blob.source_uri!r} was fingerprinted using "
            f"{blob.hash_algorithm!r}, but the normalizer would fingerprint "
            f"its text using {expected!r}. A document carrying two digests "
            f"computed under differing algorithms is incoherent.",
            source_uri=blob.source_uri,
            context={
                "payload_algorithm": blob.hash_algorithm,
                "text_algorithm": expected,
            },
        )

    def _verify_metadata_agreement(
        self,
        metadata: HarvestedMetadata,
        text: str,
        blob: AcquiredBlob,
    ) -> None:
        """
        Verify that the metadata was harvested from the supplied text.

        ``RawDocument`` requires ``metadata.char_count`` to equal the text
        length exactly, and would reject a mismatch itself. Checking here
        first attributes the fault correctly: a disagreement means the
        harvester and this stage were handed different text, which is a wiring
        defect in the pipeline rather than a schema violation.

        Parameters
        ----------
        metadata:
            The harvested metadata.
        text:
            The normalized document text.
        blob:
            The acquired payload, for error provenance.

        Returns
        -------
        None

        Raises
        ------
        NormalizationError
            If the recorded character count does not match the text.
        """
        recorded = metadata.metadata.char_count
        if recorded == len(text):
            return
        raise NormalizationError(
            f"Metadata for {blob.source_uri!r} records {recorded} characters "
            f"but the supplied text contains {len(text)}. The metadata "
            f"harvester and the assembly stage were given different text.",
            source_uri=blob.source_uri,
            context={"metadata_char_count": recorded, "text_length": len(text)},
        )

    def _resolve_ingestion_instant(self, blob: AcquiredBlob) -> datetime:
        """
        Obtain the instant at which this document was assembled.

        The clock contract requires a timezone-aware instant. A naive value is
        rejected here rather than stored, because it would compare unequally
        against every other timestamp in the pipeline and would raise a
        ``TypeError`` at an unrelated point far from its cause.

        Parameters
        ----------
        blob:
            The acquired payload, for error provenance.

        Returns
        -------
        datetime
            A timezone-aware instant.

        Raises
        ------
        NormalizationError
            If the injected clock returned a naive instant.
        """
        instant = self._clock.now()
        if instant.tzinfo is not None:
            return instant
        raise NormalizationError(
            f"The injected clock returned a naive instant while assembling "
            f"{blob.source_uri!r}. Every PTFM timestamp must be timezone "
            f"aware, or comparisons across the pipeline will fail.",
            source_uri=blob.source_uri,
        )

    def __repr__(self) -> str:
        """
        Return a representation naming the configured hash algorithm.

        Returns
        -------
        str
            Diagnostic representation.
        """
        return (
            f"RawDocumentNormalizer(algorithm="
            f"{self._hash_provider.algorithm!r}, "
            f"max_warnings={self._max_warnings})"
        )
