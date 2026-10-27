"""
PTFM Ingestion Concurrency Utilities

File:
    ptfm/core/ingestion/utils/concurrency.py

Description:
    Provides the ``ThreadOffloader`` implementations through which every
    blocking operation in the Data Ingestion Module reaches a worker thread.

Why this exists
---------------
    Parsing libraries and filesystem reads are synchronous and blocking.
    Calling them from a coroutine stalls the event loop and therefore every
    document being processed concurrently. Offloading is the mechanism that
    prevents it, and routing all offloading through one injected abstraction
    means the concurrency library remains replaceable and the capacity bound
    is enforced in exactly one place.

Cancellation semantics
----------------------
    Blocking work is submitted with cancellation ABANDONMENT enabled: when
    the awaiting coroutine is cancelled, it is released immediately and the
    worker thread continues to completion on its own.

    This is deliberate, and it is what makes the extraction deadline behave
    as documented. The alternative, waiting for the thread before honouring a
    cancellation, would mean a per-document deadline could not release the
    pipeline at all: a parser stuck for an hour would hold the deadline open
    for an hour.

    The consequence is stated plainly rather than concealed. A pathological
    document occupies one worker thread until its parser returns. Python
    cannot terminate a running thread, so a genuine hard bound requires
    process-pool offloading behind this same protocol.

Dependency contract
-------------------
    Imports only the standard library, ``anyio``, and
    ``ptfm.core.ingestion.exceptions``.

    ``anyio`` is already present transitively through Starlette and FastAPI,
    so no new dependency is introduced. It supplies a capacity limiter, which
    plain ``asyncio.to_thread`` does not: an unbounded pool converts a large
    corpus into thread exhaustion. Version 4 or later is required, since the
    cancellation parameter was renamed in that release.
"""

from __future__ import annotations

import functools
from typing import Callable, ParamSpec, TypeVar

import anyio
import anyio.to_thread

from ptfm.core.ingestion.exceptions import IngestionConfigurationError

__all__ = ["BoundedThreadOffloader", "InlineOffloader"]

_P = ParamSpec("_P")
_T = TypeVar("_T")


class BoundedThreadOffloader:
    """
    Executes blocking callables in a capacity-limited worker thread pool.

    Satisfies the ``ThreadOffloader`` protocol. A single instance should be
    constructed at the composition root and shared by every stage and
    connector, so that the configured limit bounds the process as a whole
    rather than each component independently.

    Parameters
    ----------
    limit:
        Maximum number of blocking calls executing concurrently. Ordinarily
        obtained from ``IngestionConfig.resolve_thread_limit``.

    Raises
    ------
    IngestionConfigurationError
        If the limit is below one.
    """

    __slots__ = ("_limiter",)

    def __init__(self, limit: int) -> None:
        if limit < 1:
            raise IngestionConfigurationError(
                f"Thread offloader limit must be at least 1, got {limit}.",
                context={"limit": limit},
            )
        self._limiter = anyio.CapacityLimiter(limit)

    @property
    def limit(self) -> int:
        """
        Maximum number of concurrent blocking calls.

        Returns
        -------
        int
            The configured limit.
        """
        return int(self._limiter.total_tokens)

    async def run(
        self,
        func: Callable[_P, _T],
        *args: _P.args,
        **kwargs: _P.kwargs,
    ) -> _T:
        """
        Run a blocking callable in a worker thread and await its result.

        Waits for a slot when the pool is saturated, so submitting work is
        naturally back-pressured rather than unbounded.

        Cancellation releases the caller immediately; the worker thread is
        abandoned and runs to completion. See the module documentation for
        why, and for what that costs.

        Parameters
        ----------
        func:
            The blocking callable.
        *args:
            Positional arguments forwarded to the callable.
        **kwargs:
            Keyword arguments forwarded to the callable.

        Returns
        -------
        _T
            The callable's return value.

        Raises
        ------
        Exception
            Any exception raised by the callable propagates unchanged, so
            that stage-level wrapping remains the single conversion point.
        """
        # anyio.to_thread.run_sync forwards positional arguments only, so the
        # call is bound up front rather than partially forwarded.
        call = functools.partial(func, *args, **kwargs)
        return await anyio.to_thread.run_sync(
            call,
            limiter=self._limiter,
            abandon_on_cancel=True,
        )

    def __repr__(self) -> str:
        """
        Return a representation naming the configured limit.

        Returns
        -------
        str
            Diagnostic representation.
        """
        return f"BoundedThreadOffloader(limit={self.limit})"


class InlineOffloader:
    """
    Executes blocking callables on the calling thread.

    Satisfies the ``ThreadOffloader`` protocol while performing no
    offloading at all.

    Intended for tests, where removing thread scheduling from the failure
    surface makes assertions deterministic, and for embedded single-document
    use where the caller has already established that no event loop is being
    starved.

    MUST NOT be used for corpus ingestion. Every parse would execute on the
    event loop, stalling every other document in flight.
    """

    __slots__ = ()

    @property
    def limit(self) -> int:
        """
        Nominal concurrency limit.

        Returns
        -------
        int
            Always one, since work executes on the calling thread.
        """
        return 1

    async def run(
        self,
        func: Callable[_P, _T],
        *args: _P.args,
        **kwargs: _P.kwargs,
    ) -> _T:
        """
        Run a callable immediately on the calling thread.

        Parameters
        ----------
        func:
            The callable.
        *args:
            Positional arguments forwarded to the callable.
        **kwargs:
            Keyword arguments forwarded to the callable.

        Returns
        -------
        _T
            The callable's return value.

        Raises
        ------
        Exception
            Any exception raised by the callable propagates unchanged.
        """
        return func(*args, **kwargs)

    def __repr__(self) -> str:
        """
        Return a representation of this offloader.

        Returns
        -------
        str
            Diagnostic representation.
        """
        return "InlineOffloader()"
