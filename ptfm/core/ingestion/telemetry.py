"""
PTFM Ingestion Telemetry

File:
    ptfm/core/ingestion/telemetry.py

Description:
    Supplies the logging and metrics implementations used by the ingestion
    engine, together with the guarantee that neither can fail a run.

Telemetry must never fail ingestion
-----------------------------------
    A metrics endpoint being unreachable, or a logging handler raising, must
    not abort a corpus run. Both sinks are therefore wrapped so that any
    exception they raise is swallowed at the boundary.

    This is the one place in the module where an exception is deliberately
    discarded rather than converted. Everywhere else, swallowing an error
    would hide a fault in the pipeline; here, propagating one would let an
    observability outage destroy work that had otherwise succeeded.

Structured by default
---------------------
    Ingestion failures are per-document and must be queryable by
    ``document_id``, ``source_uri``, and ``stage``. Unstructured string logs
    are unusable at corpus scale, so the logger contract is structured and a
    standard library fallback renders key-value pairs rather than prose.

Dependency contract
-------------------
    Imports only the standard library and the ingestion ``interfaces`` module.
    ``structlog`` satisfies the ``StructuredLogger`` protocol structurally and
    is used when present, so no logging library appears in the Core import
    graph.
"""

from __future__ import annotations

import logging
from typing import Any, Final, Mapping, Optional

from ptfm.core.ingestion.interfaces import MetricsSink, StructuredLogger

__all__ = [
    "StdlibStructuredLogger",
    "NullMetricsSink",
    "SafeMetricsSink",
    "SafeLogger",
    "build_logger",
    "build_metrics_sink",
]

#: Logger name under which ingestion events are emitted.
_LOGGER_NAME: Final[str] = "ptfm.ingestion"


class StdlibStructuredLogger:
    """
    Structured logger backed by the standard library.

    Satisfies the ``StructuredLogger`` protocol without requiring a logging
    dependency. Bound context is rendered as key-value pairs appended to the
    event name, which is greppable and machine-parseable even without a JSON
    renderer configured.

    Used when ``structlog`` is absent. Where it is present, a bound
    ``structlog`` logger satisfies the same protocol and is preferred, since
    it renders genuine JSON.

    Parameters
    ----------
    logger:
        The underlying standard library logger.
    context:
        Context included in every event emitted by this instance.
    """

    __slots__ = ("_logger", "_context")

    def __init__(
        self,
        logger: logging.Logger,
        context: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self._logger = logger
        self._context: dict[str, Any] = dict(context) if context else {}

    def bind(self, **values: Any) -> StdlibStructuredLogger:
        """
        Return a logger carrying additional persistent context.

        A new instance is returned rather than the existing one mutated, so
        that a logger bound for one document cannot leak its context into
        another.

        Parameters
        ----------
        **values:
            Context merged into every subsequent event.

        Returns
        -------
        StdlibStructuredLogger
            A logger bound to the merged context.
        """
        return StdlibStructuredLogger(self._logger, {**self._context, **values})

    def debug(self, event: str, **values: Any) -> None:
        """
        Emit a debug event.

        Parameters
        ----------
        event:
            Stable event name.
        **values:
            Structured event fields.

        Returns
        -------
        None
        """
        self._emit(logging.DEBUG, event, values)

    def info(self, event: str, **values: Any) -> None:
        """
        Emit an informational event.

        Parameters
        ----------
        event:
            Stable event name.
        **values:
            Structured event fields.

        Returns
        -------
        None
        """
        self._emit(logging.INFO, event, values)

    def warning(self, event: str, **values: Any) -> None:
        """
        Emit a warning event.

        Parameters
        ----------
        event:
            Stable event name.
        **values:
            Structured event fields.

        Returns
        -------
        None
        """
        self._emit(logging.WARNING, event, values)

    def error(self, event: str, **values: Any) -> None:
        """
        Emit an error event.

        Parameters
        ----------
        event:
            Stable event name.
        **values:
            Structured event fields.

        Returns
        -------
        None
        """
        self._emit(logging.ERROR, event, values)

    def _emit(self, level: int, event: str, values: Mapping[str, Any]) -> None:
        """
        Render and emit one event.

        The level is checked before rendering, so that a debug event costs
        nothing when debug logging is disabled. At corpus scale this matters:
        rendering a per-document event that is then discarded would be
        performed once per document.

        Parameters
        ----------
        level:
            Standard library severity level.
        event:
            Stable event name.
        values:
            Structured event fields, merged over the bound context.

        Returns
        -------
        None
        """
        if not self._logger.isEnabledFor(level):
            return

        merged = {**self._context, **values}
        if not merged:
            self._logger.log(level, event)
            return

        rendered = " ".join(f"{key}={value!r}" for key, value in merged.items())
        self._logger.log(level, "%s %s", event, rendered)


class NullMetricsSink:
    """
    Metrics sink that discards every observation.

    Satisfies the ``MetricsSink`` protocol. Used when metrics emission is
    disabled, so that the engine has no conditional path around emission and
    every stage can record unconditionally.
    """

    __slots__ = ()

    def increment(self, name: str, value: int = 1, **labels: str) -> None:
        """
        Discard a counter increment.

        Parameters
        ----------
        name:
            Metric name.
        value:
            Amount that would have been added.
        **labels:
            Dimensional labels.

        Returns
        -------
        None
        """
        return None

    def observe(self, name: str, value: float, **labels: str) -> None:
        """
        Discard a distribution observation.

        Parameters
        ----------
        name:
            Metric name.
        value:
            Value that would have been recorded.
        **labels:
            Dimensional labels.

        Returns
        -------
        None
        """
        return None


class SafeMetricsSink:
    """
    Metrics sink wrapper that cannot raise.

    A metrics backend being unreachable must not abort a corpus run, so every
    exception the wrapped sink raises is discarded. This is deliberate
    swallowing, confined to telemetry, where propagating would let an
    observability outage destroy work that had otherwise succeeded.

    Cancellation and interruption still propagate, since ``BaseException`` is
    never caught.

    Parameters
    ----------
    sink:
        The sink to protect.
    """

    __slots__ = ("_sink",)

    def __init__(self, sink: MetricsSink) -> None:
        self._sink = sink

    def increment(self, name: str, value: int = 1, **labels: str) -> None:
        """
        Increment a counter, discarding any failure.

        Parameters
        ----------
        name:
            Metric name.
        value:
            Amount to add.
        **labels:
            Dimensional labels.

        Returns
        -------
        None
        """
        try:
            self._sink.increment(name, value, **labels)
        except Exception:  # noqa: BLE001 - telemetry must never fail a run
            return None

    def observe(self, name: str, value: float, **labels: str) -> None:
        """
        Record an observation, discarding any failure.

        Parameters
        ----------
        name:
            Metric name.
        value:
            Observed value.
        **labels:
            Dimensional labels.

        Returns
        -------
        None
        """
        try:
            self._sink.observe(name, value, **labels)
        except Exception:  # noqa: BLE001 - telemetry must never fail a run
            return None


class SafeLogger:
    """
    Structured logger wrapper that cannot raise.

    A logging handler raising must not abort a corpus run, for the same
    reason a metrics backend must not. Every exception the wrapped logger
    raises is discarded.

    Cancellation and interruption still propagate, since ``BaseException`` is
    never caught.

    Parameters
    ----------
    logger:
        The logger to protect.
    """

    __slots__ = ("_logger",)

    def __init__(self, logger: StructuredLogger) -> None:
        self._logger = logger

    def bind(self, **values: Any) -> SafeLogger:
        """
        Return a logger carrying additional persistent context.

        Parameters
        ----------
        **values:
            Context merged into every subsequent event.

        Returns
        -------
        SafeLogger
            A protected logger bound to the merged context. On failure the
            unbound logger is returned, so that binding cannot fail a run.
        """
        try:
            return SafeLogger(self._logger.bind(**values))
        except Exception:  # noqa: BLE001 - telemetry must never fail a run
            return self

    def debug(self, event: str, **values: Any) -> None:
        """
        Emit a debug event, discarding any failure.

        Parameters
        ----------
        event:
            Stable event name.
        **values:
            Structured event fields.

        Returns
        -------
        None
        """
        self._emit("debug", event, values)

    def info(self, event: str, **values: Any) -> None:
        """
        Emit an informational event, discarding any failure.

        Parameters
        ----------
        event:
            Stable event name.
        **values:
            Structured event fields.

        Returns
        -------
        None
        """
        self._emit("info", event, values)

    def warning(self, event: str, **values: Any) -> None:
        """
        Emit a warning event, discarding any failure.

        Parameters
        ----------
        event:
            Stable event name.
        **values:
            Structured event fields.

        Returns
        -------
        None
        """
        self._emit("warning", event, values)

    def error(self, event: str, **values: Any) -> None:
        """
        Emit an error event, discarding any failure.

        Parameters
        ----------
        event:
            Stable event name.
        **values:
            Structured event fields.

        Returns
        -------
        None
        """
        self._emit("error", event, values)

    def _emit(self, level: str, event: str, values: Mapping[str, Any]) -> None:
        """
        Dispatch one event to the wrapped logger.

        Parameters
        ----------
        level:
            Method name on the wrapped logger.
        event:
            Stable event name.
        values:
            Structured event fields.

        Returns
        -------
        None
        """
        try:
            getattr(self._logger, level)(event, **values)
        except Exception:  # noqa: BLE001 - telemetry must never fail a run
            return None


# ============================================================================
# CONSTRUCTION
# ============================================================================


def build_logger(
    log_level: str,
    *,
    logger: Optional[StructuredLogger] = None,
) -> StructuredLogger:
    """
    Construct the logger used by an ingestion run.

    A supplied logger is protected and returned unchanged otherwise. Where
    none is supplied, ``structlog`` is used when present and the standard
    library otherwise, so that a deployment gains real JSON rendering by
    installing a package rather than by changing code.

    Parameters
    ----------
    log_level:
        Minimum severity to emit.
    logger:
        Logger supplied by the composition root, if any.

    Returns
    -------
    StructuredLogger
        A logger that cannot fail a run.
    """
    if logger is not None:
        return SafeLogger(logger)

    try:
        import structlog

        return SafeLogger(structlog.get_logger(_LOGGER_NAME))
    except ImportError:
        stdlib = logging.getLogger(_LOGGER_NAME)
        stdlib.setLevel(getattr(logging, log_level.upper(), logging.INFO))
        return SafeLogger(StdlibStructuredLogger(stdlib))


def build_metrics_sink(
    enabled: bool,
    *,
    sink: Optional[MetricsSink] = None,
) -> MetricsSink:
    """
    Construct the metrics sink used by an ingestion run.

    A discarding sink is returned when emission is disabled, rather than
    ``None``, so that the engine has no conditional path around emission and
    every stage can record unconditionally.

    Parameters
    ----------
    enabled:
        Whether metrics are emitted.
    sink:
        Sink supplied by the composition root, if any.

    Returns
    -------
    MetricsSink
        A sink that cannot fail a run.
    """
    if not enabled or sink is None:
        return NullMetricsSink()
    return SafeMetricsSink(sink)
