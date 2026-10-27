"""
PTFM In-Memory Source Connector

File:
    ptfm/core/ingestion/connectors/memory.py

Description:
    Serves documents held directly in process memory.

Purpose
-------
    Three uses, all first-class rather than incidental:

        Testing
            Every pipeline test needs a source. A connector with no
            filesystem, no network, and no fixtures makes the pipeline
            testable without introducing either into the test surface.

        Embedded ingestion
            A caller that already holds document bytes, such as a service
            handling an upload, can ingest them without first writing them to
            disk purely to satisfy a connector.

        Reference implementation
            The smallest complete example of the ``SourceConnector``
            contract, which is what a new connector is written against.

Scoping
-------
    Documents are addressed as ``memory:///<identifier>``. A specification
    whose root path is a prefix of a document's path selects that document,
    so a single connector can serve several logical collections.

Blocking behaviour
------------------
    Payloads are already resident, so no transfer occurs and no I/O is
    performed. Construction is nonetheless routed through ``PayloadBuilder``,
    so that digests, size enforcement, and provenance are produced by exactly
    the same code that serves every other connector rather than reimplemented
    here.

    The builder spills to temporary storage beyond a configured threshold,
    which is the one path in this connector that can block. Payloads below
    the threshold are therefore built on the calling thread, and only those
    at or above it are offloaded. The distinction is decided from the payload
    size, which is known exactly, so the choice is never a guess.

Dependency contract
-------------------
    Imports only the standard library and the ingestion ``exceptions``,
    ``config``, ``models``, ``interfaces``, ``pipeline.acquisition`` and
    ``utils.concurrency`` modules.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import AsyncIterator, Final, Iterable, Mapping, Optional
from urllib.parse import quote, urlparse

from ptfm.core.ingestion.config import IngestionConfig
from ptfm.core.ingestion.exceptions import (
    ConnectorRegistrationError,
    SourceError,
    SourceNotFoundError,
)
from ptfm.core.ingestion.interfaces import (
    Clock,
    HashProvider,
    SourceConnector,
    ThreadOffloader,
)
from ptfm.core.ingestion.models import AcquiredBlob, SourceRecord, SourceSpec
from ptfm.core.ingestion.pipeline.acquisition import PayloadBuilder, matches_filters
from ptfm.core.ingestion.utils.concurrency import InlineOffloader

__all__ = ["MemoryDocument", "InMemoryConnector"]

#: URI scheme served by this connector.
_SCHEME: Final[str] = "memory"

#: Characters left unescaped when building a document URI. The path separator
#: is preserved so that hierarchical identifiers remain meaningful and can be
#: matched by a specification root prefix.
_URI_SAFE_CHARACTERS: Final[str] = "/"


@dataclass(frozen=True, slots=True)
class MemoryDocument:
    """
    A document held in memory, together with the metadata a source would
    ordinarily supply.

    Attributes
    ----------
    identifier:
        Path-like identifier, unique within one connector. Forms the path
        component of the document URI, so a hierarchical identifier such as
        ``"reports/q3.pdf"`` can be selected by a specification rooted at
        ``memory:///reports``.
    data:
        The complete payload.
    filename:
        Filename presented to format detection. Defaults to the final segment
        of the identifier, which is ordinarily what a caller intends.
    declared_media_type:
        Media type asserted for this document. Advisory: format detection
        treats it as a weaker signal than the payload's own bytes.
    version_tag:
        Opaque version marker. Supplying one enables unchanged-version
        skipping across runs; omitting it means the document is always
        transferred and compared by digest instead.
    attributes:
        Additional transport metadata. String-valued, so that structured
        domain objects cannot enter the pipeline through a generic field.
    """

    identifier: str
    data: bytes
    filename: Optional[str] = None
    declared_media_type: Optional[str] = None
    version_tag: Optional[str] = None
    attributes: Mapping[str, str] = field(default_factory=dict)

    @property
    def resolved_filename(self) -> str:
        """
        Filename presented to format detection.

        Returns
        -------
        str
            The declared filename, or the final segment of the identifier.
        """
        if self.filename is not None:
            return self.filename
        return self.identifier.rsplit("/", 1)[-1] or self.identifier


class InMemoryConnector(SourceConnector):
    """
    Serves documents held in process memory.

    Parameters
    ----------
    config:
        Ingestion configuration. Only the acquisition section is consulted,
        for size limits and the spool threshold, but the full configuration
        is accepted so that this class satisfies ``ConnectorFactory``
        directly.
    documents:
        The documents to serve. Ordering is preserved, so discovery is
        deterministic.
    clock:
        Supplies the discovery instant. Injected so that discovered records
        are byte-identical under test.
    offloader:
        Executes the payload build when it may spill to temporary storage.
        Defaults to inline execution, which is correct here because a payload
        below the spool threshold performs no I/O at all. A composition root
        ingesting large in-memory payloads should inject the shared bounded
        offloader.
    hash_provider:
        Fingerprint provider. Must match the provider configured elsewhere in
        the pipeline, since the acquisition stage rejects a payload
        fingerprinted under an unexpected algorithm.

    Raises
    ------
    ConnectorRegistrationError
        If two documents share an identifier, or an identifier is blank.
        Records must be uniquely addressable within one discovery pass.
    """

    __slots__ = (
        "_config",
        "_documents",
        "_clock",
        "_offloader",
        "_hash_provider",
        "_closed",
    )

    def __init__(
        self,
        config: IngestionConfig,
        documents: Iterable[MemoryDocument] = (),
        *,
        clock: Optional[Clock] = None,
        offloader: Optional[ThreadOffloader] = None,
        hash_provider: Optional[HashProvider] = None,
    ) -> None:
        resolved: dict[str, MemoryDocument] = {}

        for document in documents:
            identifier = document.identifier.strip().lstrip("/")
            if not identifier:
                raise ConnectorRegistrationError(
                    "MemoryDocument identifiers must be non-empty.",
                    scheme=_SCHEME,
                )
            uri = _uri_for(identifier)
            if uri in resolved:
                raise ConnectorRegistrationError(
                    f"Duplicate in-memory document identifier "
                    f"{identifier!r}. Records must be uniquely addressable "
                    f"within one discovery pass.",
                    scheme=_SCHEME,
                )
            resolved[uri] = document

        self._config = config
        self._documents: Mapping[str, MemoryDocument] = resolved
        self._clock = clock
        self._offloader: ThreadOffloader = offloader or InlineOffloader()
        self._hash_provider = hash_provider
        self._closed = False

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_mapping(
        cls,
        config: IngestionConfig,
        payloads: Mapping[str, bytes],
        **options: object,
    ) -> InMemoryConnector:
        """
        Construct a connector from identifiers mapped to payloads.

        Convenience for the common case in which no per-document metadata is
        required.

        Parameters
        ----------
        config:
            Ingestion configuration.
        payloads:
            Identifiers mapped to their complete payloads.
        **options:
            Forwarded to the constructor.

        Returns
        -------
        InMemoryConnector
            A connector serving the supplied payloads.
        """
        return cls(
            config,
            [
                MemoryDocument(identifier=identifier, data=data)
                for identifier, data in payloads.items()
            ],
            **options,  # type: ignore[arg-type]
        )

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        """
        Stable identifier of this connector implementation.

        Returns
        -------
        str
            The identifier ``"InMemoryConnector"``.
        """
        return "InMemoryConnector"

    @property
    def scheme(self) -> str:
        """
        URI scheme this connector serves.

        Returns
        -------
        str
            The scheme ``"memory"``.
        """
        return _SCHEME

    @property
    def document_count(self) -> int:
        """
        Number of documents held by this connector.

        Returns
        -------
        int
            Document count, which becomes zero once the connector is closed.
        """
        return len(self._documents)

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------

    async def discover(self, spec: SourceSpec) -> AsyncIterator[SourceRecord]:
        """
        Enumerate the documents selected by a specification.

        Documents are yielded in the order supplied at construction, so
        discovery is deterministic. Selection filters are applied here as an
        efficiency measure; the acquisition stage applies them again
        authoritatively, so a connector that omits them cannot leak excluded
        documents into a corpus.

        Parameters
        ----------
        spec:
            Source specification. Its root path selects documents by prefix,
            so a root of ``memory:///`` selects everything.

        Yields
        ------
        SourceRecord
            One record per selected document.

        Raises
        ------
        SourceError
            If the connector has been closed.
        """
        self._ensure_open()

        prefix = _normalize_prefix(spec.root_uri)
        discovered_at = self._clock.now() if self._clock is not None else _now()

        for uri, document in self._documents.items():
            if not _within_prefix(uri, prefix):
                continue
            if not matches_filters(uri, spec):
                continue

            yield SourceRecord(
                record_id=document.identifier,
                source_uri=uri,
                scheme=_SCHEME,
                connector_name=self.name,
                discovered_at=discovered_at,
                filename=document.resolved_filename,
                declared_media_type=document.declared_media_type,
                size_bytes=len(document.data),
                version_tag=document.version_tag,
                attributes=dict(document.attributes),
            )

    # ------------------------------------------------------------------
    # Fetching
    # ------------------------------------------------------------------

    async def fetch(self, record: SourceRecord) -> AcquiredBlob:
        """
        Produce the payload for a previously discovered record.

        No transfer occurs, since the payload is already resident. The blob
        is nonetheless constructed through ``PayloadBuilder`` so that
        digests, size enforcement, and provenance are identical to those
        produced by every other connector.

        A payload below the spool threshold cannot touch the filesystem and
        is therefore built on the calling thread. One at or above it may spill
        to temporary storage, so it is offloaded.

        Parameters
        ----------
        record:
            A record previously yielded by :meth:`discover`.

        Returns
        -------
        AcquiredBlob
            The payload with its digest and acquisition provenance.

        Raises
        ------
        SourceNotFoundError
            If no document is held under the record's URI.
        SourceError
            If the connector has been closed.
        DocumentTooLargeError
            If the payload exceeds the configured maximum document size.
        """
        self._ensure_open()

        document = self._documents.get(record.source_uri)
        if document is None:
            raise SourceNotFoundError(
                f"No in-memory document is held under "
                f"{record.source_uri!r}.",
                source_uri=record.source_uri,
                context={"held_documents": len(self._documents)},
            )

        if len(document.data) < self._config.acquisition.spool_threshold_bytes:
            # Below the threshold the builder retains everything in memory,
            # so no filesystem access is possible and offloading would only
            # add scheduling overhead.
            return self._build(record, document.data)

        return await self._offloader.run(self._build, record, document.data)

    def _build(self, record: SourceRecord, data: bytes) -> AcquiredBlob:
        """
        Construct the acquired payload for a document.

        Blocking once the payload exceeds the spool threshold, at which point
        the builder writes to temporary storage. The caller decides whether
        this runs on a worker thread.

        Parameters
        ----------
        record:
            The record being fetched.
        data:
            The complete payload.

        Returns
        -------
        AcquiredBlob
            The payload with its digest and acquisition provenance.

        Raises
        ------
        DocumentTooLargeError
            If the payload exceeds the configured maximum document size.
        AcquisitionError
            If temporary storage could not be created or written.
        """
        with PayloadBuilder(
            record,
            self._config.acquisition,
            hash_provider=self._hash_provider,
            clock=self._clock,
        ) as builder:
            # Written in a single call: the payload is already contiguous in
            # memory, so chunking it would achieve nothing beyond additional
            # copies.
            builder.write(data)
            return builder.build()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def health_check(self) -> bool:
        """
        Verify that this connector can serve documents.

        Returns
        -------
        bool
            ``True`` while the connector is open, ``False`` once it has been
            closed.
        """
        return not self._closed

    async def close(self) -> None:
        """
        Release the documents held by this connector.

        Idempotent. Payloads are dropped so that a long-lived process does
        not retain a corpus in memory after ingestion has finished.

        Returns
        -------
        None
        """
        if self._closed:
            return
        self._closed = True
        self._documents = {}

    def _ensure_open(self) -> None:
        """
        Guard against use after release.

        Returns
        -------
        None

        Raises
        ------
        SourceError
            If the connector has been closed. Raised as a permanent failure,
            since retrying against a released connector cannot succeed.
        """
        if not self._closed:
            return
        raise SourceError(
            f"{self.name} has been closed and no longer holds any documents."
        )

    def __repr__(self) -> str:
        """
        Return a representation that never includes payload bytes.

        Returns
        -------
        str
            Diagnostic representation.
        """
        return (
            f"InMemoryConnector(documents={len(self._documents)}, "
            f"closed={self._closed})"
        )


# ============================================================================
# HELPERS
# ============================================================================


def _now() -> "datetime":  # noqa: F821 - annotation resolved lazily
    """
    Return the current instant as a timezone-aware UTC datetime.

    Used only when no clock is injected. Defined as a function so that the
    wall clock is not imported at module level, where an injected clock is
    the expected source of time.

    Returns
    -------
    datetime
        The current instant in UTC.
    """
    from datetime import UTC, datetime

    return datetime.now(UTC)


def _uri_for(identifier: str) -> str:
    """
    Build the document URI for an identifier.

    Path separators are preserved so that hierarchical identifiers remain
    selectable by a specification root prefix; every other character is
    percent-encoded so that the result is a well-formed URI.

    Parameters
    ----------
    identifier:
        Path-like document identifier, without a leading separator.

    Returns
    -------
    str
        A URI of the form ``memory:///<identifier>``.
    """
    return f"{_SCHEME}:///{quote(identifier, safe=_URI_SAFE_CHARACTERS)}"


def _normalize_prefix(root_uri: str) -> str:
    """
    Reduce a specification root to a comparable path prefix.

    Parameters
    ----------
    root_uri:
        Root URI from the source specification.

    Returns
    -------
    str
        The path component with a single leading separator and no trailing
        separator. An empty result selects every document.
    """
    path = urlparse(root_uri).path
    return "/" + path.strip("/") if path.strip("/") else ""


def _within_prefix(uri: str, prefix: str) -> bool:
    """
    Report whether a document URI falls under a root path prefix.

    Matching is on whole path segments, so a root of ``/reports`` selects
    ``/reports/q3.pdf`` but not ``/reports-archive/q3.pdf``.

    Parameters
    ----------
    uri:
        The document URI.
    prefix:
        Normalized root path prefix. An empty prefix matches everything.

    Returns
    -------
    bool
        ``True`` when the document falls under the prefix.
    """
    if not prefix:
        return True
    path = urlparse(uri).path
    return path == prefix or path.startswith(f"{prefix}/")
