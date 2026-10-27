"""
PTFM Content Fingerprinting

File:
    ptfm/core/ingestion/utils/hashing.py

Description:
    Provides the default ``HashProvider`` implementation used to fingerprint
    document payloads and normalized text.

Algorithm choice
----------------
    BLAKE2b truncated to 256 bits. Cryptographically sound, faster than
    SHA-256 in CPython, and available from the standard library, so no
    dependency is introduced for a function performed once per document.

    The algorithm name encodes the digest width, so that a future change of
    either is a detectable, migratable event rather than a silent corruption
    of every previously recorded fingerprint.

Dependency contract
-------------------
    Imports only the standard library.
"""

from __future__ import annotations

import hashlib
from typing import Final

__all__ = ["Blake2bHashProvider"]

#: Digest width in bytes. Fixed rather than configurable: a per-deployment
#: width would make fingerprints incomparable between deployments without
#: any visible indication.
_DIGEST_SIZE: Final[int] = 32

#: Encoding used for text fingerprinting. Fixed so that identical text always
#: yields an identical digest regardless of the source payload encoding.
_TEXT_ENCODING: Final[str] = "utf-8"


class _IncrementalBlake2b:
    """
    Accumulates a BLAKE2b digest over a payload delivered in chunks.

    Satisfies the ``IncrementalHasher`` protocol. Required because payloads
    above the spool threshold are never fully materialized in memory, so a
    whole-payload interface would force exactly the materialization spooling
    exists to avoid.
    """

    __slots__ = ("_hasher",)

    def __init__(self) -> None:
        self._hasher = hashlib.blake2b(digest_size=_DIGEST_SIZE)

    def update(self, chunk: bytes) -> None:
        """
        Incorporate the next chunk of the payload.

        Parameters
        ----------
        chunk:
            Next contiguous slice of the payload.

        Returns
        -------
        None
        """
        self._hasher.update(chunk)

    def hexdigest(self) -> str:
        """
        Finalize and return the accumulated digest.

        Returns
        -------
        str
            Lowercase hexadecimal digest.
        """
        return self._hasher.hexdigest()


class Blake2bHashProvider:
    """
    Produces BLAKE2b-256 content fingerprints.

    Satisfies the ``HashProvider`` protocol. Stateless, so one instance
    serves all concurrent workers.
    """

    __slots__ = ()

    @property
    def algorithm(self) -> str:
        """
        Name of the digest algorithm in use.

        Recorded alongside every fingerprint so that an algorithm change is
        detectable rather than silent.

        Returns
        -------
        str
            The identifier ``"blake2b-256"``.
        """
        return "blake2b-256"

    def hash_bytes(self, data: bytes) -> str:
        """
        Compute the digest of an in-memory payload.

        Parameters
        ----------
        data:
            Payload bytes.

        Returns
        -------
        str
            Lowercase hexadecimal digest.
        """
        return hashlib.blake2b(data, digest_size=_DIGEST_SIZE).hexdigest()

    def hash_text(self, text: str) -> str:
        """
        Compute the digest of normalized text.

        Parameters
        ----------
        text:
            Normalized document text.

        Returns
        -------
        str
            Lowercase hexadecimal digest.
        """
        return self.hash_bytes(text.encode(_TEXT_ENCODING))

    def new_incremental(self) -> _IncrementalBlake2b:
        """
        Create a hasher for a chunk-delivered payload.

        Returns
        -------
        _IncrementalBlake2b
            A fresh, independent accumulator.
        """
        return _IncrementalBlake2b()