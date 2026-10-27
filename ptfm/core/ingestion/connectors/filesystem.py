"""
PTFM Local Filesystem Source Connector

File:
    ptfm/core/ingestion/connectors/filesystem.py

Description:
    Serves documents from a locally accessible filesystem, including network
    filesystems mounted into the local namespace.

Async safety
------------
    Every filesystem operation blocks: directory listing, metadata retrieval,
    and reading alike. All of them are routed through the injected
    ``ThreadOffloader``, and none is performed on the event loop.

    Directory listing is offloaded PER DIRECTORY rather than for the tree as
    a whole. Walking a large tree inside a single offloaded call would occupy
    a worker thread for the entire walk and would prevent discovery from
    streaming, so the first record would not be available until the last one
    had been found.

Incremental re-ingestion
------------------------
    Each record carries a version tag derived from the file's modification
    time and size. A file unchanged since a previous run therefore presents
    an unchanged tag and is skipped before any transfer occurs, which is what
    makes repeated ingestion of a large corpus inexpensive.

Symbolic links
--------------
    Not followed by default. A directory symlink can form a cycle, and a
    symlink pointing outside the root silently widens the corpus beyond what
    the specification selected. Following can be enabled explicitly, and the
    root containment check remains available independently.

Partial failure
---------------
    An unreadable subdirectory does not abort discovery. One directory with
    restrictive permissions must not discard a run over forty thousand
    documents. Such directories are skipped and recorded in
    :attr:`LocalFilesystemConnector.skipped_paths` for the composition root
    to report, rather than being silently ignored. A root that cannot be read
    is a configuration failure and does raise.

Dependency contract
-------------------
    Imports only the standard library and the ingestion ``exceptions``,
    ``config``, ``models``, ``interfaces``, ``pipeline.acquisition`` and
    ``utils.concurrency`` modules.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import AsyncIterator, Final, Optional, Sequence
from urllib.parse import urlparse
from urllib.request import url2pathname

from ptfm.core.ingestion.config import IngestionConfig
from ptfm.core.ingestion.exceptions import (
    PTFMIngestionError,
    SourceAuthenticationError,
    SourceError,
    SourceNotFoundError,
    SourceUnavailableError,
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

__all__ = ["SkippedPath", "LocalFilesystemConnector"]

#: URI scheme served by this connector.
_SCHEME: Final[str] = "file"


@dataclass(frozen=True, slots=True)
class SkippedPath:
    """
    A directory omitted from discovery, with the reason.

    Exposed as data rather than logged, so that the connector performs no
    output of its own and so that omissions are directly assertable. The
    composition root is responsible for reporting these.

    Attributes
    ----------
    path:
        The directory that could not be enumerated.
    reason:
        Why it was skipped.
    """

    path: str
    reason: str


@dataclass(frozen=True, slots=True)
class _FileEntry:
    """
    A file observed during a directory scan.

    Metadata is captured inside the offloaded scan, so that no further
    blocking call is required once the scan returns.

    Attributes
    ----------
    path:
        Absolute path to the file.
    size_bytes:
        Size reported by the filesystem.
    mtime_ns:
        Modification time in nanoseconds, forming part of the version tag.
    """

    path: Path
    size_bytes: int
    mtime_ns: int


class LocalFilesystemConnector(SourceConnector):
    """
    Serves documents from a local or locally mounted filesystem.

    Parameters
    ----------
    config:
        Ingestion configuration. The acquisition section supplies size
        limits, the read chunk size, and the spool threshold. The full
        configuration is accepted so that this class satisfies
        ``ConnectorFactory`` directly.
    clock:
        Supplies the discovery instant. Injected so that discovered records
        are byte-identical under test.
    offloader:
        Executes every blocking filesystem operation. Defaults to inline
        execution, which is correct only for tests and embedded single
        document use; a composition root MUST inject the shared bounded
        offloader, or filesystem reads will run on the event loop.
    hash_provider:
        Fingerprint provider. Must match the provider configured elsewhere in
        the pipeline, since the acquisition stage rejects a payload
        fingerprinted under an unexpected algorithm.
    follow_symlinks:
        Whether symbolic links are followed. Disabled by default: a directory
        symlink can form a cycle, and any symlink can point outside the
        selected root.
    allowed_roots:
        When supplied, a fetch is refused for any path outside these
        directories. Defence in depth against a record that did not originate
        from this connector's own discovery. When omitted, no containment
        check is applied, which is safe in the ordinary case because the
        engine only ever fetches records that discovery produced.
    """

    __slots__ = (
        "_config",
        "_clock",
        "_offloader",
        "_hash_provider",
        "_follow_symlinks",
        "_allowed_roots",
        "_skipped",
        "_closed",
    )

    def __init__(
        self,
        config: IngestionConfig,
        *,
        clock: Optional[Clock] = None,
        offloader: Optional[ThreadOffloader] = None,
        hash_provider: Optional[HashProvider] = None,
        follow_symlinks: bool = False,
        allowed_roots: Optional[Sequence[Path | str]] = None,
    ) -> None:
        self._config = config
        self._clock = clock
        self._offloader: ThreadOffloader = offloader or InlineOffloader()
        self._hash_provider = hash_provider
        self._follow_symlinks = follow_symlinks
        self._allowed_roots: Optional[tuple[Path, ...]] = (
            tuple(Path(root).resolve() for root in allowed_roots)
            if allowed_roots is not None
            else None
        )
        # Per-run, append-only record of directories that could not be
        # enumerated. The connector is constructed fresh for each run, so this
        # is run-scoped instance state rather than accumulating global state.
        self._skipped: list[SkippedPath] = []
        self._closed = False

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
            The identifier ``"LocalFilesystemConnector"``.
        """
        return "LocalFilesystemConnector"

    @property
    def scheme(self) -> str:
        """
        URI scheme this connector serves.

        Returns
        -------
        str
            The scheme ``"file"``.
        """
        return _SCHEME

    @property
    def skipped_paths(self) -> tuple[SkippedPath, ...]:
        """
        Directories omitted from discovery, with reasons.

        Returns
        -------
        tuple[SkippedPath, ...]
            Omission records, in the order encountered.
        """
        return tuple(self._skipped)

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------

    async def discover(self, spec: SourceSpec) -> AsyncIterator[SourceRecord]:
        """
        Enumerate the files beneath a specification's root.

        Traversal is depth first with entries sorted by name, so discovery
        order is deterministic across runs and across platforms.

        Each directory listing is a separate offloaded call, which keeps any
        single blocking operation short and allows records to stream as they
        are found rather than only after the entire tree has been walked.

        Selection filters are applied here so that excluded files are never
        stated. The acquisition stage applies them again authoritatively, so
        omitting them here would cost efficiency but could not admit an
        excluded document.

        Parameters
        ----------
        spec:
            Source specification supplying the root URI, the selection
            filters, and whether to descend into subdirectories.

        Yields
        ------
        SourceRecord
            One record per file selected by the specification.

        Raises
        ------
        SourceNotFoundError
            If the root does not exist.
        SourceError
            If the root is not a directory, the connector has been closed, or
            the root URI cannot be interpreted as a path.
        SourceAuthenticationError
            If the root itself cannot be read.
        """
        self._ensure_open()

        root = _path_from_uri(spec.root_uri)
        await self._verify_root(root)

        discovered_at = self._clock.now() if self._clock is not None else _now()
        stack: list[Path] = [root]

        while stack:
            directory = stack.pop()

            try:
                files, subdirectories = await self._offloader.run(
                    _scan_directory, directory, self._follow_symlinks
                )
            except PermissionError as exc:
                # One restrictive directory must not discard the whole run.
                self._skipped.append(
                    SkippedPath(
                        path=str(directory), reason=f"permission denied: {exc}"
                    )
                )
                continue
            except OSError as exc:
                self._skipped.append(
                    SkippedPath(
                        path=str(directory),
                        reason=f"{type(exc).__name__}: {exc}",
                    )
                )
                continue

            for entry in files:
                uri = _uri_from_path(entry.path)
                if not matches_filters(uri, spec):
                    continue

                yield SourceRecord(
                    record_id=_relative_identifier(entry.path, root),
                    source_uri=uri,
                    scheme=_SCHEME,
                    connector_name=self.name,
                    discovered_at=discovered_at,
                    filename=entry.path.name,
                    # No media type is declared. A filesystem asserts nothing
                    # about content, and guessing from the extension here
                    # would merely duplicate the weakest signal that format
                    # detection already derives for itself.
                    declared_media_type=None,
                    size_bytes=entry.size_bytes,
                    version_tag=_version_tag(entry),
                    attributes={},
                )

            if spec.recursive:
                # Reversed, because the stack pops from the end and the
                # entries arrive sorted; this yields ascending traversal.
                stack.extend(reversed(subdirectories))

    async def _verify_root(self, root: Path) -> None:
        """
        Verify that a discovery root exists and can be enumerated.

        Checked before traversal begins, so that a misconfigured root fails
        immediately rather than presenting as an empty corpus.

        Parameters
        ----------
        root:
            The resolved root directory.

        Returns
        -------
        None

        Raises
        ------
        SourceNotFoundError
            If the root does not exist.
        SourceError
            If the root exists but is not a directory.
        SourceAuthenticationError
            If the root cannot be read.
        """
        status = await self._offloader.run(_probe_directory, root)

        if status == "missing":
            raise SourceNotFoundError(
                f"Discovery root {str(root)!r} does not exist.",
                source_uri=_uri_from_path(root),
            )
        if status == "not_a_directory":
            raise SourceError(
                f"Discovery root {str(root)!r} is not a directory.",
                source_uri=_uri_from_path(root),
            )
        if status == "denied":
            raise SourceAuthenticationError(
                f"Discovery root {str(root)!r} cannot be read.",
                source_uri=_uri_from_path(root),
            )

    # ------------------------------------------------------------------
    # Fetching
    # ------------------------------------------------------------------

    async def fetch(self, record: SourceRecord) -> AcquiredBlob:
        """
        Read a file's contents into an acquired payload.

        The file is read in configured chunks and streamed into a
        ``PayloadBuilder``, so that the digest is accumulated during
        transfer, the size limit is enforced as bytes arrive, and a large
        file spills to temporary storage rather than being held in memory.

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
            If the file no longer exists.
        SourceAuthenticationError
            If the file cannot be read.
        SourceUnavailableError
            If the read failed for another reason, which on a network
            filesystem is frequently transient.
        SourceError
            If the path lies outside the permitted roots, refers to a
            directory, or the connector has been closed.
        DocumentTooLargeError
            If the file exceeds the configured maximum document size.
        """
        self._ensure_open()

        path = _path_from_uri(record.source_uri)
        self._enforce_containment(path, record)

        return await self._offloader.run(self._read_file, record, path)

    def _read_file(self, record: SourceRecord, path: Path) -> AcquiredBlob:
        """
        Read a file and build its payload.

        Blocking throughout. Invoked only inside a worker thread.

        Parameters
        ----------
        record:
            The record being fetched.
        path:
            Absolute path to the file.

        Returns
        -------
        AcquiredBlob
            The payload with its digest and acquisition provenance.

        Raises
        ------
        SourceNotFoundError
            If the file no longer exists.
        SourceAuthenticationError
            If the file cannot be read.
        SourceUnavailableError
            If the read failed for another reason.
        SourceError
            If the path refers to a directory.
        DocumentTooLargeError
            If the file exceeds the configured maximum document size.
        """
        chunk_size = self._config.acquisition.read_chunk_bytes

        try:
            with PayloadBuilder(
                record,
                self._config.acquisition,
                hash_provider=self._hash_provider,
                clock=self._clock,
            ) as builder:
                with path.open("rb") as handle:
                    while True:
                        chunk = handle.read(chunk_size)
                        if not chunk:
                            break
                        builder.write(chunk)
                return builder.build()
        except PTFMIngestionError:
            # Size violations and spool failures are already typed and carry
            # better context than anything that could be added here.
            raise
        except FileNotFoundError as exc:
            raise SourceNotFoundError(
                f"File {str(path)!r} no longer exists.",
                source_uri=record.source_uri,
            ) from exc
        except PermissionError as exc:
            raise SourceAuthenticationError(
                f"File {str(path)!r} cannot be read: {exc}",
                source_uri=record.source_uri,
            ) from exc
        except IsADirectoryError as exc:
            raise SourceError(
                f"Path {str(path)!r} refers to a directory, not a document.",
                source_uri=record.source_uri,
            ) from exc
        except OSError as exc:
            # Treated as transient. A `file://` root is frequently a network
            # filesystem, where a read failure is often momentary, and
            # retrying a genuinely permanent local error costs only the
            # remaining attempts.
            raise SourceUnavailableError(
                f"File {str(path)!r} could not be read: "
                f"{type(exc).__name__}: {exc}",
                source_uri=record.source_uri,
            ) from exc

    def _enforce_containment(self, path: Path, record: SourceRecord) -> None:
        """
        Verify that a path lies within the permitted roots.

        Applied only when roots were configured. The engine fetches solely
        records produced by this connector's own discovery, so containment is
        defence in depth rather than a routine check.

        Parameters
        ----------
        path:
            The path being fetched.
        record:
            The record being fetched, for error provenance.

        Returns
        -------
        None

        Raises
        ------
        SourceError
            If the path resolves outside every permitted root.
        """
        if self._allowed_roots is None:
            return

        resolved = path.resolve()
        if any(resolved.is_relative_to(root) for root in self._allowed_roots):
            return

        raise SourceError(
            f"Path {str(resolved)!r} lies outside every permitted root and "
            f"will not be read.",
            source_uri=record.source_uri,
            context={
                "resolved_path": str(resolved),
                "allowed_roots": [str(root) for root in self._allowed_roots],
            },
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def health_check(self) -> bool:
        """
        Verify that the configured roots are readable directories.

        A discovery root arrives with the specification rather than at
        construction, so when no permitted roots were configured there is
        nothing to verify ahead of a run and this reports operational.
        Discovery performs its own root check regardless, so a missing root
        still fails immediately rather than presenting as an empty corpus.

        Returns
        -------
        bool
            ``True`` when every configured root is a readable directory, and
            when the connector has not been closed.
        """
        if self._closed:
            return False
        if not self._allowed_roots:
            return True

        for root in self._allowed_roots:
            if await self._offloader.run(_probe_directory, root) != "ok":
                return False
        return True

    async def close(self) -> None:
        """
        Release resources held by this connector.

        Idempotent. No handles are retained between fetches, so this only
        marks the connector unusable.

        Returns
        -------
        None
        """
        self._closed = True

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
        raise SourceError(f"{self.name} has been closed.")

    def __repr__(self) -> str:
        """
        Return a diagnostic representation of this connector.

        Returns
        -------
        str
            Diagnostic representation.
        """
        roots = (
            "unrestricted"
            if self._allowed_roots is None
            else [str(root) for root in self._allowed_roots]
        )
        return (
            f"LocalFilesystemConnector(allowed_roots={roots!r}, "
            f"follow_symlinks={self._follow_symlinks}, "
            f"skipped={len(self._skipped)})"
        )


# ============================================================================
# BLOCKING HELPERS
# ============================================================================


def _scan_directory(
    directory: Path,
    follow_symlinks: bool,
) -> tuple[tuple[_FileEntry, ...], tuple[Path, ...]]:
    """
    List one directory, capturing file metadata as it goes.

    Blocking. Invoked only inside a worker thread.

    Metadata is captured here rather than by the caller, so that discovering
    a file costs one blocking call rather than one call per file.

    Entries are sorted by name so that traversal order is deterministic
    across runs and across filesystems, which differ in the order they
    return directory entries.

    Parameters
    ----------
    directory:
        The directory to list.
    follow_symlinks:
        Whether symbolic links are followed. When disabled they are omitted
        entirely, since a directory link can form a cycle and any link can
        point outside the selected root.

    Returns
    -------
    tuple[tuple[_FileEntry, ...], tuple[Path, ...]]
        The files found, and the subdirectories to descend into.

    Raises
    ------
    OSError
        If the directory could not be listed. The caller decides whether that
        is fatal or merely a skip.
    """
    files: list[_FileEntry] = []
    subdirectories: list[Path] = []

    with os.scandir(directory) as entries:
        for entry in sorted(entries, key=lambda item: item.name):
            if entry.is_symlink() and not follow_symlinks:
                continue

            try:
                if entry.is_dir(follow_symlinks=follow_symlinks):
                    subdirectories.append(Path(entry.path))
                    continue
                if not entry.is_file(follow_symlinks=follow_symlinks):
                    # Sockets, devices, and FIFOs are not documents.
                    continue
                status = entry.stat(follow_symlinks=follow_symlinks)
            except OSError:
                # A single unreadable entry, such as a broken link or a file
                # removed mid-scan, must not discard the rest of the
                # directory.
                continue

            files.append(
                _FileEntry(
                    path=Path(entry.path),
                    size_bytes=status.st_size,
                    mtime_ns=status.st_mtime_ns,
                )
            )

    return tuple(files), tuple(subdirectories)


def _probe_directory(path: Path) -> str:
    """
    Classify the readability of a directory.

    Blocking. Invoked only inside a worker thread.

    Returns a classification rather than raising, so that the caller can
    choose the exception appropriate to its context: a missing discovery root
    is fatal, whereas a missing permitted root merely reports unhealthy.

    Parameters
    ----------
    path:
        The directory to probe.

    Returns
    -------
    str
        One of ``"ok"``, ``"missing"``, ``"not_a_directory"`` or
        ``"denied"``.
    """
    try:
        if not path.exists():
            return "missing"
        if not path.is_dir():
            return "not_a_directory"
        # Listing a single entry establishes readability without the cost of
        # enumerating a large directory.
        with os.scandir(path) as entries:
            next(iter(entries), None)
    except PermissionError:
        return "denied"
    except OSError:
        return "missing"
    return "ok"


# ============================================================================
# PURE HELPERS
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


def _path_from_uri(uri: str) -> Path:
    """
    Interpret a ``file`` URI as a filesystem path.

    Percent-encoding is decoded, and platform-specific forms such as a
    Windows drive letter are handled, so that a URI produced by
    :func:`_uri_from_path` round-trips.

    Parameters
    ----------
    uri:
        A ``file`` URI, or a bare path for caller convenience.

    Returns
    -------
    Path
        The corresponding filesystem path.

    Raises
    ------
    SourceError
        If the URI carries a scheme other than ``file``, or names a remote
        host, which this connector cannot reach.
    """
    parsed = urlparse(uri)

    if parsed.scheme and parsed.scheme != _SCHEME:
        raise SourceError(
            f"URI {uri!r} carries scheme {parsed.scheme!r}, which this "
            f"connector does not serve.",
            source_uri=uri,
        )

    if parsed.netloc not in ("", "localhost"):
        raise SourceError(
            f"URI {uri!r} names host {parsed.netloc!r}. This connector reads "
            f"only paths reachable in the local filesystem namespace; mount "
            f"the remote location and reference it by its local path.",
            source_uri=uri,
        )

    if not parsed.scheme:
        return Path(uri)
    return Path(url2pathname(parsed.path))


def _uri_from_path(path: Path) -> str:
    """
    Render a filesystem path as a ``file`` URI.

    Parameters
    ----------
    path:
        The path to render. Made absolute if it is not already, since a URI
        cannot express a relative path.

    Returns
    -------
    str
        The corresponding ``file`` URI.
    """
    return (path if path.is_absolute() else path.resolve()).as_uri()


def _relative_identifier(path: Path, root: Path) -> str:
    """
    Derive a connector-scoped identifier for a discovered file.

    The path relative to the discovery root, in POSIX form, so that
    identifiers are stable across platforms.

    Parameters
    ----------
    path:
        Absolute path to the file.
    root:
        The discovery root.

    Returns
    -------
    str
        The relative identifier, falling back to the absolute path when the
        file does not lie beneath the root.
    """
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def _version_tag(entry: _FileEntry) -> str:
    """
    Derive an opaque version marker for a file.

    Combines modification time with size. Either alone is defeatable: a
    filesystem with coarse timestamp granularity can report an unchanged
    time across a rapid edit, and an edit that preserves length leaves the
    size unchanged. Together they detect ordinary modification reliably
    enough to make incremental re-ingestion worthwhile, and any change they
    miss is still caught by the payload digest after transfer.

    Parameters
    ----------
    entry:
        The file entry captured during the directory scan.

    Returns
    -------
    str
        An opaque version tag.
    """
    return f"{entry.mtime_ns}-{entry.size_bytes}"
