"""
PTFM Extraction Stage

File:
    ptfm/core/ingestion/pipeline/extraction.py

Description:
    Resolves the content extractor for a detected media type, executes the
    blocking parse off the event loop under a deadline, and converts any
    non-PTFM exception into a PTFM type.

Async safety
------------
    Every viable parsing library is synchronous and CPU-bound. This stage
    is the boundary that keeps that fact from reaching the event loop: the
    extractor is invoked exclusively through the injected
    ``ThreadOffloader``, never directly.

    Reading a spooled payload is also blocking, so the read and the parse
    are performed inside a SINGLE offloaded callable. Splitting them would
    double the context switches and invite the read to be performed on the
    event loop.

Deadline semantics, stated honestly
-----------------------------------
    ``ExtractionConfig.timeout_seconds`` bounds how long the PIPELINE waits.
    It does NOT terminate the parser.

    Python cannot cancel a running thread. When the deadline expires, the
    awaiting coroutine is released and the document is quarantined, but the
    worker thread continues executing the parser until it returns of its own
    accord. A pathological document, such as a decompression bomb, therefore
    occupies one worker thread for the remainder of the run.

    What the deadline provides:
        - the document is quarantined promptly rather than stalling the run
        - the pipeline continues processing other documents
        - the run terminates rather than hanging indefinitely

    What it does not provide:
        - reclaimed thread capacity

    A true hard bound requires process-pool offloading, which is a
    ``ThreadOffloader`` implementation concern and is deferred. This
    limitation is documented rather than concealed, because a deadline that
    silently fails to bound resource consumption is more dangerous than a
    known one.

Exception conversion boundary
-----------------------------
    This stage is the SOLE location at which third-party parser exceptions
    become PTFM exceptions. Extractors are contractually obliged to raise
    PTFM types themselves, so this layer functions as a safety net rather
    than a translation table:

        PTFMIngestionError   re-raised, enriched with document context
        Exception            wrapped in ExtractorInternalError, cause kept
        BaseException        NEVER caught, so cancellation and interruption
                             propagate untouched

Dependency contract
-------------------
    Imports only the standard library, ``ptfm.core.schemas`` and the
    ingestion ``exceptions``, ``config``, ``models``, ``interfaces`` and
    ``registry`` modules. No parsing library is imported, so extractor
    exceptions are classified by name rather than by type.
"""

from __future__ import annotations

import asyncio
from typing import Final, Optional

from ptfm.core.ingestion.config import ExtractionConfig
from ptfm.core.ingestion.exceptions import (
    CorruptDocumentError,
    EncryptedDocumentError,
    ExtractionTimeoutError,
    ExtractorInternalError,
    PTFMIngestionError,
)
from ptfm.core.ingestion.interfaces import ContentExtractor, ThreadOffloader
from ptfm.core.ingestion.models import (
    AcquiredBlob,
    DetectedFormat,
    ExtractionInput,
    ExtractionOutput,
)
from ptfm.core.ingestion.registry import ExtractorRegistry
from ptfm.core.schemas import ExtractionWarning, ExtractionWarningCode

__all__ = ["ExtractionStage"]

#: Exception class names, across known parsing libraries, that indicate a
#: password protected or otherwise encrypted payload. Matched by name
#: because importing the libraries would bind the Core to them.
#:
#: This mapping is a FALLBACK for extractors that do not comply with the
#: contract obliging them to raise ``EncryptedDocumentError`` directly. It
#: is inherently fragile: a library renaming an exception silently degrades
#: the outcome to ``ExtractorInternalError``. That degradation is safe, as
#: the document is still quarantined with its cause preserved, but the
#: correct remedy is always to fix the extractor.
_ENCRYPTION_EXCEPTION_NAMES: Final[frozenset[str]] = frozenset(
    {
        "FileEncryptedError",
        "PdfReadError",
        "PasswordError",
        "DependencyError",
        "PDFPasswordIncorrect",
        "PDFEncryptionError",
    }
)

#: Lowercase tokens whose presence in an exception message indicates
#: encryption. Deliberately narrow: a broader list would misclassify
#: unrelated failures, and a misclassification here tells an operator to
#: hunt for credentials that do not exist.
_ENCRYPTION_MESSAGE_TOKENS: Final[tuple[str, ...]] = (
    "password",
    "encrypted",
    "encryption",
    "decrypt",
)

#: Exception class names indicating structurally malformed input rather
#: than a defective extractor. Subject to the same fragility caveat.
_CORRUPTION_EXCEPTION_NAMES: Final[frozenset[str]] = frozenset(
    {
        "PackageNotFoundError",
        "BadZipFile",
        "PdfiumError",
        "PDFSyntaxError",
        "XMLSyntaxError",
        "ParseError",
    }
)


class ExtractionStage:
    """
    Executes content extraction for one document.

    Parameters
    ----------
    registry:
        Extractor registry resolving media types to extractors. Resolution
        is a constant-time lookup, so dispatch cost is negligible per
        document.
    offloader:
        Thread offloader through which the blocking parse is executed. The
        extractor is never invoked directly, which is what keeps parsing
        off the event loop.
    config:
        Extraction configuration supplying the per-document deadline.
    """

    __slots__ = ("_registry", "_offloader", "_config")

    def __init__(
        self,
        registry: ExtractorRegistry,
        offloader: ThreadOffloader,
        config: ExtractionConfig,
    ) -> None:
        self._registry = registry
        self._offloader = offloader
        self._config = config

    # ------------------------------------------------------------------
    # Public contract
    # ------------------------------------------------------------------

    async def extract(
        self,
        blob: AcquiredBlob,
        detected: DetectedFormat,
    ) -> ExtractionOutput:
        """
        Extract text and structure from an acquired payload.

        Resolves the extractor for the detected media type, then reads the
        payload and parses it inside a single worker thread under the
        configured deadline.

        Parameters
        ----------
        blob:
            The acquired payload. Read inside the worker thread, so a
            spooled payload never blocks the event loop.
        detected:
            The resolved media type and encoding.

        Returns
        -------
        ExtractionOutput
            Extracted blocks, format-native metadata, and any warnings. A
            warning is appended when the payload yielded no text.

        Raises
        ------
        UnsupportedMediaTypeError
            If no extractor is registered for the detected media type.
        ExtractionTimeoutError
            If the deadline expires. Note that the parser thread continues
            running; see the module documentation.
        EncryptedDocumentError
            If the payload is password protected.
        CorruptDocumentError
            If the payload is structurally malformed.
        ExtractorInternalError
            If the extractor raised an unanticipated exception, returned a
            value of the wrong type, or the payload could not be read.
        """
        extractor = self._registry.resolve(detected.media_type)

        output = await self._run_extraction(extractor, blob, detected)

        if not output.is_empty:
            return output

        return output.model_copy(
            update={
                "warnings": (
                    *output.warnings,
                    ExtractionWarning(
                        code=ExtractionWarningCode.PARTIAL_EXTRACTION,
                        message=(
                            "Extraction produced no text. The payload may "
                            "contain only images, or may require optical "
                            "character recognition."
                        ),
                        detail={
                            "media_type": detected.media_type,
                            "extractor_name": extractor.name,
                            "size_bytes": str(blob.size_bytes),
                        },
                    ),
                )
            }
        )

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    async def _run_extraction(
        self,
        extractor: ContentExtractor,
        blob: AcquiredBlob,
        detected: DetectedFormat,
    ) -> ExtractionOutput:
        """
        Run the extractor in a worker thread under the deadline.

        Parameters
        ----------
        extractor:
            The resolved extractor.
        blob:
            The acquired payload.
        detected:
            The resolved media type and encoding.

        Returns
        -------
        ExtractionOutput
            The extractor's validated result.

        Raises
        ------
        ExtractionTimeoutError
            If the deadline expires before the parse completes.
        PTFMIngestionError
            Re-raised, enriched with document context, when the extractor
            raised a PTFM exception directly.
        ExtractorInternalError
            If the extractor raised any other exception or returned a value
            of the wrong type.
        """
        timeout = self._config.timeout_seconds
        source_uri = blob.source_uri

        try:
            result = await asyncio.wait_for(
                self._offloader.run(
                    _read_and_extract, extractor, blob, detected
                ),
                timeout=timeout,
            )
        except TimeoutError as exc:
            # The awaiting coroutine is released here, but the parser thread
            # continues until it returns. See the module documentation.
            raise ExtractionTimeoutError(
                f"Extraction of {source_uri!r} by {extractor.name!r} exceeded "
                f"the {timeout} second deadline. The document has been "
                f"quarantined; the parser thread may still be running.",
                timeout_seconds=timeout,
                source_uri=source_uri,
                context={
                    "extractor_name": extractor.name,
                    "media_type": detected.media_type,
                    "size_bytes": blob.size_bytes,
                    "thread_reclaimed": False,
                },
            ) from exc
        except PTFMIngestionError as error:
            _enrich(error, source_uri=source_uri, extractor_name=extractor.name)
            raise
        except Exception as exc:
            raise _convert(
                exc,
                source_uri=source_uri,
                extractor=extractor,
                detected=detected,
            ) from exc

        if not isinstance(result, ExtractionOutput):
            raise ExtractorInternalError(
                f"Extractor {extractor.name!r} returned "
                f"{type(result).__name__}, which is not an ExtractionOutput.",
                extractor_name=extractor.name,
                source_uri=source_uri,
                context={"returned_type": type(result).__name__},
            )

        return result


# ============================================================================
# WORKER-THREAD ENTRY POINT
# ============================================================================


def _read_and_extract(
    extractor: ContentExtractor,
    blob: AcquiredBlob,
    detected: DetectedFormat,
) -> ExtractionOutput:
    """
    Read a payload and parse it, entirely within a worker thread.

    Both operations block: reading a spooled payload performs disk I/O, and
    parsing is CPU-bound. Combining them into one offloaded callable halves
    the context switches and removes any path by which the read could be
    performed on the event loop.

    Parameters
    ----------
    extractor:
        The resolved extractor.
    blob:
        The acquired payload.
    detected:
        The resolved media type and encoding.

    Returns
    -------
    ExtractionOutput
        The extractor's result, returned unvalidated to the caller.

    Raises
    ------
    ExtractorInternalError
        If the payload could not be read from temporary storage.
    Exception
        Any exception raised by the extractor propagates unchanged, so that
        conversion remains the caller's single responsibility.
    """
    try:
        data = blob.read()
    except (OSError, ValueError) as exc:
        raise ExtractorInternalError(
            f"Payload for {blob.source_uri!r} could not be read prior to "
            f"extraction: {type(exc).__name__}: {exc}",
            extractor_name=extractor.name,
            source_uri=blob.source_uri,
            context={"spooled": blob.is_spooled},
        ) from exc

    return extractor.extract(
        ExtractionInput(
            source_uri=blob.source_uri,
            media_type=detected.media_type,
            data=data,
            encoding=detected.encoding,
            filename=blob.record.filename,
        )
    )


# ============================================================================
# EXCEPTION HANDLING
# ============================================================================


def _enrich(
    error: PTFMIngestionError,
    *,
    source_uri: str,
    extractor_name: str,
) -> None:
    """
    Attach document context to a PTFM exception in flight.

    An extractor knows the payload it was handed but not always the URI it
    originated from. Filling that in here means a quarantine record always
    identifies the document, without every extractor having to remember to
    supply it.

    Parameters
    ----------
    error:
        The propagating exception.
    source_uri:
        URI of the document being extracted.
    extractor_name:
        Identifier of the extractor that raised.

    Returns
    -------
    None
    """
    if error.source_uri is None:
        error.source_uri = source_uri
    error.add_context(source_uri=source_uri, extractor_name=extractor_name)


def _convert(
    exc: Exception,
    *,
    source_uri: str,
    extractor: ContentExtractor,
    detected: DetectedFormat,
) -> PTFMIngestionError:
    """
    Convert a non-PTFM exception into a PTFM exception.

    Classification is by exception class name and message token rather than
    by type, because importing the parsing libraries would bind the Core to
    them. An unrecognized exception becomes ``ExtractorInternalError``,
    which is always a safe outcome: the document is quarantined and the
    original cause is preserved for inspection.

    Parameters
    ----------
    exc:
        The exception raised by the extractor.
    source_uri:
        URI of the document being extracted.
    extractor:
        The extractor that raised.
    detected:
        The resolved media type and encoding.

    Returns
    -------
    PTFMIngestionError
        The exception to raise in place of ``exc``, which the caller chains
        as ``__cause__``.
    """
    name = type(exc).__name__
    message = str(exc)
    context = {
        "extractor_name": extractor.name,
        "media_type": detected.media_type,
        "original_exception": name,
    }

    if _indicates_encryption(name, message):
        return EncryptedDocumentError(
            f"Payload {source_uri!r} appears to be encrypted or password "
            f"protected: {name}: {message}",
            source_uri=source_uri,
            context=context,
        )

    if name in _CORRUPTION_EXCEPTION_NAMES:
        return CorruptDocumentError(
            f"Payload {source_uri!r} is structurally malformed and could not "
            f"be parsed by {extractor.name!r}: {name}: {message}",
            source_uri=source_uri,
            context=context,
        )

    return ExtractorInternalError(
        f"Extractor {extractor.name!r} raised an unanticipated "
        f"{name} while parsing {source_uri!r}: {message}",
        extractor_name=extractor.name,
        source_uri=source_uri,
        context=context,
    )


def _indicates_encryption(exception_name: str, message: str) -> bool:
    """
    Report whether an exception indicates an encrypted payload.

    Requires either a known exception class name or an explicit token in
    the message. The token list is kept narrow because a false positive
    directs an operator to search for credentials that do not exist.

    Parameters
    ----------
    exception_name:
        Class name of the raised exception.
    message:
        The exception message.

    Returns
    -------
    bool
        ``True`` when the payload is probably encrypted.
    """
    if exception_name in _ENCRYPTION_EXCEPTION_NAMES:
        return True
    lowered = message.lower()
    return any(token in lowered for token in _ENCRYPTION_MESSAGE_TOKENS)