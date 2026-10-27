"""
PTFM Knowledge Processing Telemetry

File:
    ptfm/core/knowledge/telemetry.py

Description:
    Supplies the logging and metrics implementations used by the knowledge
    engine, and the guarantee that neither can fail a run.

Telemetry must never fail a run
-------------------------------
    A metrics endpoint being unreachable, or a logging handler raising, must
    not abort a corpus run. Both sinks are wrapped so that any exception they
    raise is discarded at the boundary.

    This is the one place in the module where an exception is deliberately
    swallowed rather than converted. Everywhere else, swallowing would hide a
    pipeline fault; here, propagating would let an observability outage
    destroy work that had otherwise succeeded.

    Cancellation and interruption still propagate, since ``BaseException`` is
    never caught.

Reuse
-----
    The implementations are those of the ingestion module, re-exported under
    a knowledge-facing name. The concerns are identical, and a second
    implementation would drift from the first while claiming to do the same
    thing.

Dependency contract
-------------------
    Imports the ingestion telemetry module and the knowledge ``interfaces``
    module. No logging library appears in the Core import graph: ``structlog``
    satisfies the protocol structurally and is used when present.
"""

from __future__ import annotations

from typing import Optional

from ptfm.core.ingestion.telemetry import (
    NullMetricsSink,
    SafeLogger,
    SafeMetricsSink,
    StdlibStructuredLogger,
)
from ptfm.core.ingestion.telemetry import build_logger as _build_ingestion_logger
from ptfm.core.ingestion.telemetry import (
    build_metrics_sink as _build_ingestion_metrics,
)
from ptfm.core.knowledge.interfaces import MetricsSink, StructuredLogger

__all__ = [
    "StdlibStructuredLogger",
    "NullMetricsSink",
    "SafeLogger",
    "SafeMetricsSink",
    "build_logger",
    "build_metrics_sink",
]


def build_logger(
    log_level: str,
    *,
    logger: Optional[StructuredLogger] = None,
) -> StructuredLogger:
    """
    Construct the logger used by a knowledge processing run.

    A supplied logger is protected and returned; otherwise one is built from
    ``structlog`` when present and the standard library otherwise, so that a
    deployment gains real JSON rendering by installing a package rather than
    by changing code.

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
    return _build_ingestion_logger(log_level, logger=logger)


def build_metrics_sink(
    enabled: bool,
    *,
    sink: Optional[MetricsSink] = None,
) -> MetricsSink:
    """
    Construct the metrics sink used by a knowledge processing run.

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
    return _build_ingestion_metrics(enabled, sink=sink)
