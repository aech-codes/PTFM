"""
PTFM Knowledge Sinks

File:
    ptfm/core/knowledge/sinks.py

Description:
    Implementations of the ``KnowledgeSink`` protocol that belong in the Core
    because they are independent of any storage technology.

    Real persistence is supplied by Retrieval (Module 3), which owns vector,
    lexical, metadata, and graph storage and their schemas. This module knows
    nothing of them.

Why these three, and no others
------------------------------
    NullKnowledgeSink
        Discards everything. Lets a run exercise the entire pipeline without
        persisting anything, which is how a corpus is validated before
        committing storage to it.

    InMemoryKnowledgeSink
        Retains everything and can be queried. The reference implementation
        against which the protocol's semantics are defined, and what makes
        engine tests possible without infrastructure.

    CompositeKnowledgeSink
        Writes to several sinks in turn. The engine takes one sink, but a
        deployment ordinarily has a vector store, a lexical index, and a graph
        store. Fanning out is a composition concern rather than a storage one,
        so it belongs here rather than being solved separately by each backend.

Placement
---------
    A top-level module rather than a pipeline stage. A sink is not a
    transformation: it is the boundary at which the module hands off its
    output, and the engine writes to it after the pipeline has finished with a
    document.

Concurrency
-----------
    The engine processes documents concurrently, so a sink is written to
    concurrently. The in-memory implementation guards its state with a lock;
    the composite delegates and holds none.

Dependency contract
-------------------
    Imports only the standard library, the knowledge ``exceptions`` module and
    ``models``. No storage client, and no reference to any index technology.
"""

from __future__ import annotations

import asyncio
from typing import Iterable, Iterator, Optional

from ptfm.core.knowledge.exceptions import SinkWriteError
from ptfm.core.knowledge.models import KnowledgeUnit

__all__ = [
    "NullKnowledgeSink",
    "InMemoryKnowledgeSink",
    "CompositeKnowledgeSink",
]


class NullKnowledgeSink:
    """
    A sink that discards every unit.

    Satisfies the ``KnowledgeSink`` protocol. Lets a run exercise the whole
    pipeline, including validation, without persisting anything, which is how
    a corpus is checked before storage is committed to it.

    Counts what it discarded, so a run against it still reports what it would
    have written.
    """

    __slots__ = ("_written", "_flushes", "_closed")

    def __init__(self) -> None:
        self._written = 0
        self._flushes = 0
        self._closed = False

    @property
    def name(self) -> str:
        """
        Stable identifier of this sink.

        Returns
        -------
        str
            Implementation identifier.
        """
        return "null"

    @property
    def written_count(self) -> int:
        """
        Number of units discarded.

        Returns
        -------
        int
            Unit count.
        """
        return self._written

    async def write(self, unit: KnowledgeUnit) -> None:
        """
        Discard a unit, counting it.

        Parameters
        ----------
        unit:
            The unit to discard.

        Returns
        -------
        None

        Raises
        ------
        SinkWriteError
            If the sink has been closed. Writing after close is a sequencing
            defect in the caller, and accepting it silently would let a run
            report success for units nothing received.
        """
        self._require_open()
        self._written += 1

    async def flush(self) -> None:
        """
        Record a flush. Nothing is buffered.

        Returns
        -------
        None
        """
        self._flushes += 1

    async def close(self) -> None:
        """
        Close the sink. Idempotent.

        Returns
        -------
        None
        """
        self._closed = True

    def _require_open(self) -> None:
        """
        Ensure the sink has not been closed.

        Returns
        -------
        None

        Raises
        ------
        SinkWriteError
            If the sink is closed.
        """
        if self._closed:
            raise SinkWriteError(
                "Unit written to a closed sink. A run reporting success for "
                "units nothing received would be indistinguishable from one "
                "that persisted them.",
                sink_name=self.name,
            )

    def __repr__(self) -> str:
        """
        Return a representation reporting the discarded count.

        Returns
        -------
        str
            Diagnostic representation.
        """
        return f"NullKnowledgeSink(written={self._written})"


class InMemoryKnowledgeSink:
    """
    A sink retaining every unit in memory.

    The reference implementation of the protocol, and what makes engine tests
    possible without infrastructure. Not intended for production: a corpus of
    any size will exhaust memory, and nothing survives the process.

    Writes are last-one-wins by node identifier. Because identity is
    deterministic, rewriting a node means rewriting identical content, so a
    re-run over an unchanged corpus converges rather than duplicating.

    Parameters
    ----------
    name:
        Identifier reported in diagnostics. Parameterized so that a composite
        holding several of these can distinguish them.
    """

    __slots__ = ("_name", "_units", "_lock", "_flushes", "_closed")

    def __init__(self, name: str = "in_memory") -> None:
        self._name = name
        self._units: dict[str, KnowledgeUnit] = {}
        self._lock = asyncio.Lock()
        self._flushes = 0
        self._closed = False

    @property
    def name(self) -> str:
        """
        Stable identifier of this sink.

        Returns
        -------
        str
            Implementation identifier.
        """
        return self._name

    # ------------------------------------------------------------------
    # Protocol
    # ------------------------------------------------------------------

    async def write(self, unit: KnowledgeUnit) -> None:
        """
        Retain a unit, replacing any earlier unit with the same identifier.

        Parameters
        ----------
        unit:
            The unit to retain.

        Returns
        -------
        None

        Raises
        ------
        SinkWriteError
            If the sink has been closed.
        """
        async with self._lock:
            self._require_open()
            self._units[unit.node_id] = unit

    async def flush(self) -> None:
        """
        Record a flush. Nothing is buffered.

        Returns
        -------
        None
        """
        async with self._lock:
            self._flushes += 1

    async def close(self) -> None:
        """
        Close the sink. Idempotent, and retained units remain readable.

        Units survive closing so that a test can assert what a run wrote after
        the run has torn down.

        Returns
        -------
        None
        """
        async with self._lock:
            self._closed = True

    # ------------------------------------------------------------------
    # Inspection
    # ------------------------------------------------------------------

    @property
    def units(self) -> tuple[KnowledgeUnit, ...]:
        """
        Every retained unit, in write order.

        Returns
        -------
        tuple[KnowledgeUnit, ...]
            The retained units.
        """
        return tuple(self._units.values())

    @property
    def node_count(self) -> int:
        """
        Number of distinct nodes retained.

        Returns
        -------
        int
            Node count.
        """
        return len(self._units)

    @property
    def embedding_count(self) -> int:
        """
        Number of retained units carrying a vector.

        A count below :attr:`node_count` means the corpus is partly embedded,
        and an unembedded node is invisible to dense retrieval.

        Returns
        -------
        int
            Embedding count.
        """
        return sum(1 for unit in self._units.values() if unit.is_embedded)

    @property
    def relation_count(self) -> int:
        """
        Total number of retained relations.

        Returns
        -------
        int
            Relation count.
        """
        return sum(len(unit.relations) for unit in self._units.values())

    @property
    def flush_count(self) -> int:
        """
        Number of flushes recorded.

        Returns
        -------
        int
            Flush count.
        """
        return self._flushes

    @property
    def is_closed(self) -> bool:
        """
        Whether the sink has been closed.

        Returns
        -------
        bool
            ``True`` once closed.
        """
        return self._closed

    def get(self, node_id: str) -> Optional[KnowledgeUnit]:
        """
        Retrieve a unit by node identifier.

        Parameters
        ----------
        node_id:
            The identifier to resolve.

        Returns
        -------
        Optional[KnowledgeUnit]
            The unit, or ``None`` if absent.
        """
        return self._units.get(node_id)

    def units_for(self, document_id: str) -> tuple[KnowledgeUnit, ...]:
        """
        Retrieve every unit belonging to a document, in reading order.

        Parameters
        ----------
        document_id:
            The document to select.

        Returns
        -------
        tuple[KnowledgeUnit, ...]
            The document's units, ordered by node ordinal.
        """
        matching = [
            unit for unit in self._units.values() if unit.document_id == document_id
        ]
        return tuple(sorted(matching, key=lambda unit: unit.node.ordinal))

    def document_ids(self) -> tuple[str, ...]:
        """
        Identifiers of every document with retained units, in sorted order.

        Returns
        -------
        tuple[str, ...]
            Document identifiers.
        """
        return tuple(sorted({unit.document_id for unit in self._units.values()}))

    def __len__(self) -> int:
        """
        Number of distinct nodes retained.

        Returns
        -------
        int
            Node count.
        """
        return len(self._units)

    def __iter__(self) -> Iterator[KnowledgeUnit]:
        """
        Iterate the retained units in write order.

        Yields
        ------
        KnowledgeUnit
            Each retained unit.
        """
        return iter(tuple(self._units.values()))

    def _require_open(self) -> None:
        """
        Ensure the sink has not been closed.

        Returns
        -------
        None

        Raises
        ------
        SinkWriteError
            If the sink is closed.
        """
        if self._closed:
            raise SinkWriteError(
                "Unit written to a closed sink.", sink_name=self._name
            )

    def __repr__(self) -> str:
        """
        Return a representation reporting retained counts.

        Returns
        -------
        str
            Diagnostic representation.
        """
        return (
            f"InMemoryKnowledgeSink(name={self._name!r}, "
            f"nodes={len(self._units)}, embeddings={self.embedding_count})"
        )


class CompositeKnowledgeSink:
    """
    Writes each unit to several sinks in turn.

    The engine writes to one sink, but a deployment ordinarily has a vector
    store, a lexical index, and a graph store. Fanning out is a composition
    concern rather than a storage one, so solving it once here is better than
    each backend solving it again.

    Partial writes
    --------------
    Sinks are written sequentially, and the first failure propagates. A sink
    earlier in the sequence may therefore have accepted a unit that a later
    one rejected, leaving the destinations inconsistent.

    This is tolerable rather than ignored, and specifically because node
    identity is deterministic: re-running the document produces the same
    identifiers and the same content, so a rewrite is idempotent and the
    inconsistency resolves. A sink that is not idempotent must not be composed
    this way.

    Writing to every sink before reporting the first failure was the
    alternative, and it is worse: it would continue writing into a
    destination already known to be inconsistent with its siblings.

    Parameters
    ----------
    sinks:
        The sinks to write to, in order.

    Raises
    ------
    SinkWriteError
        If no sink is supplied, or two sinks share a name. A composite with no
        destination would silently discard a run's output while reporting
        success.
    """

    __slots__ = ("_sinks",)

    def __init__(self, sinks: Iterable[object]) -> None:
        resolved = tuple(sinks)

        if not resolved:
            raise SinkWriteError(
                "A composite sink requires at least one destination; an empty "
                "composite would discard a run's output while reporting "
                "success.",
                sink_name="composite",
            )

        seen: set[str] = set()
        for sink in resolved:
            name = getattr(sink, "name", None)
            if not isinstance(name, str) or not name:
                raise SinkWriteError(
                    f"Composed sink {type(sink).__name__} does not report a "
                    f"name, so a failure could not identify which destination "
                    f"produced it.",
                    sink_name="composite",
                )
            if name in seen:
                raise SinkWriteError(
                    f"Two composed sinks report the name {name!r}; a failure "
                    f"could not be attributed to one of them.",
                    sink_name="composite",
                )
            seen.add(name)

        self._sinks = resolved

    @property
    def name(self) -> str:
        """
        Stable identifier of this sink, naming its destinations.

        Returns
        -------
        str
            Implementation identifier.
        """
        names = ", ".join(sink.name for sink in self._sinks)  # type: ignore[attr-defined]
        return f"composite({names})"

    @property
    def sinks(self) -> tuple[object, ...]:
        """
        The composed destinations, in write order.

        Returns
        -------
        tuple[object, ...]
            The sinks.
        """
        return self._sinks

    async def write(self, unit: KnowledgeUnit) -> None:
        """
        Write a unit to every destination in turn.

        Parameters
        ----------
        unit:
            The unit to write.

        Returns
        -------
        None

        Raises
        ------
        SinkWriteError
            If a destination rejected the write.
        SinkUnavailableError
            If a destination was unreachable.
        """
        for sink in self._sinks:
            await sink.write(unit)  # type: ignore[attr-defined]

    async def flush(self) -> None:
        """
        Flush every destination in turn.

        Returns
        -------
        None

        Raises
        ------
        SinkWriteError
            If a destination could not commit buffered writes.
        """
        for sink in self._sinks:
            await sink.flush()  # type: ignore[attr-defined]

    async def close(self) -> None:
        """
        Close every destination, and every one of them.

        Unlike writing, closing does not stop at the first failure. A
        destination left open leaks whatever resources it holds, so every one
        is closed and the first failure is raised afterwards.

        Returns
        -------
        None

        Raises
        ------
        BaseException
            The first failure raised while closing, after every destination
            has been closed.
        """
        first: Optional[BaseException] = None

        for sink in self._sinks:
            try:
                await sink.close()  # type: ignore[attr-defined]
            except Exception as exc:  # noqa: BLE001 - every sink must be closed
                if first is None:
                    first = exc

        if first is not None:
            raise first

    def __len__(self) -> int:
        """
        Number of composed destinations.

        Returns
        -------
        int
            Destination count.
        """
        return len(self._sinks)

    def __repr__(self) -> str:
        """
        Return a representation naming the destinations.

        Returns
        -------
        str
            Diagnostic representation.
        """
        names = [sink.name for sink in self._sinks]  # type: ignore[attr-defined]
        return f"CompositeKnowledgeSink(sinks={names!r})"
