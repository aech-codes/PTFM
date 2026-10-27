"""
PTFM Root Exception Hierarchy

File:
    ptfm/utils/exceptions.py

Description:
    Defines the single root exception type for the Probably Truthful
    Foundation Model. Every exception raised by PTFM Core modules,
    utilities, and domain plugins must derive from ``PTFMError``.

    This module is intentionally dependency-free. It imports only from
    the Python standard library so that it may be imported from any
    layer of the system, including the lowest-level utilities, without
    creating an import cycle.

Design Contract:
    - Generic exceptions (``Exception``, ``ValueError``, ``RuntimeError``)
      are never raised by PTFM code.
    - Third-party exceptions are wrapped at the boundary of the module
      that owns the dependency, using ``raise PTFMSubclass(...) from exc``
      so that the original traceback is preserved as ``__cause__``.
    - Every exception carries structured context suitable for machine
      -queryable logging via :meth:`PTFMError.to_dict`.
"""

from __future__ import annotations

import re
from types import MappingProxyType
from typing import Any, ClassVar, Mapping

__all__ = ["PTFMError"]


_CAMEL_BOUNDARY = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")


def _derive_error_code(class_name: str) -> str:
    """
    Derive a stable SCREAMING_SNAKE_CASE error code from a class name.

    The trailing ``Error`` suffix is removed and camel-case boundaries are
    converted to underscores, with correct handling of acronym runs.

    Parameters
    ----------
    class_name:
        The ``__name__`` of the exception class.

    Returns
    -------
    str
        The derived error code, e.g. ``"SourceUnavailableError"`` becomes
        ``"SOURCE_UNAVAILABLE"`` and ``"PTFMIngestionError"`` becomes
        ``"PTFM_INGESTION"``.
    """
    stem = class_name[: -len("Error")] if class_name.endswith("Error") else class_name
    if not stem:
        stem = class_name
    return _CAMEL_BOUNDARY.sub("_", stem).upper()


def _rebuild_ptfm_error(
    cls: type[PTFMError],
    args: tuple[Any, ...],
    state: dict[str, Any],
) -> PTFMError:
    """
    Reconstruct a :class:`PTFMError` during unpickling.

    Bypasses ``__init__`` so that subclasses with keyword-only or derived
    constructor signatures round-trip correctly across process boundaries.

    Parameters
    ----------
    cls:
        The concrete exception class to rebuild.
    args:
        The value of ``BaseException.args`` captured at pickle time.
    state:
        The instance ``__dict__`` captured at pickle time.

    Returns
    -------
    PTFMError
        The reconstructed exception instance.
    """
    error = cls.__new__(cls)
    BaseException.__init__(error, *args)
    error.__dict__.update(state)
    return error


class PTFMError(Exception):
    """
    Root exception for all PTFM failures.

    Attributes
    ----------
    error_code:
        Stable machine-readable identifier, derived automatically from the
        class name unless explicitly declared in the class body.
    retryable:
        Whether the operation that raised this error may reasonably be
        retried. Retry layers consult this flag rather than inspecting
        exception types at the call site. Defaults to ``False``; only
        genuinely transient conditions override it.

    Parameters
    ----------
    message:
        Human-readable description of the failure.
    context:
        Optional structured key/value detail included in
        :meth:`to_dict` output. Copied on construction and exposed as an
        immutable mapping.

    Raises
    ------
    None
    """

    error_code: ClassVar[str] = "PTFM"
    retryable: ClassVar[bool] = False

    def __init__(
        self,
        message: str,
        *,
        context: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.message: str = message
        self._context: dict[str, Any] = dict(context) if context else {}

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """
        Derive :attr:`error_code` for subclasses that do not declare one.

        Parameters
        ----------
        **kwargs:
            Forwarded to :meth:`Exception.__init_subclass__`.

        Returns
        -------
        None
        """
        super().__init_subclass__(**kwargs)
        if "error_code" not in cls.__dict__:
            cls.error_code = _derive_error_code(cls.__name__)

    @property
    def context(self) -> Mapping[str, Any]:
        """
        Structured detail associated with this failure.

        Returns
        -------
        Mapping[str, Any]
            A read-only view of the context mapping.
        """
        return MappingProxyType(self._context)

    def add_context(self, **values: Any) -> None:
        """
        Attach additional structured detail as the exception propagates.

        Intended for outer layers that know more about the operation than
        the raise site did, such as the identifier of the document being
        processed when the failure occurred.

        Parameters
        ----------
        **values:
            Key/value pairs merged into the context mapping.

        Returns
        -------
        None
        """
        self._context.update(values)

    def to_dict(self) -> dict[str, Any]:
        """
        Render the exception as a structured, log-serializable mapping.

        Returns
        -------
        dict[str, Any]
            Keys: ``error_type``, ``error_code``, ``message``,
            ``retryable``, ``context`` and, when a chained cause is
            present, ``cause_type`` and ``cause_message``.
        """
        payload: dict[str, Any] = {
            "error_type": type(self).__name__,
            "error_code": self.error_code,
            "message": self.message,
            "retryable": self.retryable,
            "context": dict(self._context),
        }
        cause = self.__cause__
        if cause is not None:
            payload["cause_type"] = type(cause).__name__
            payload["cause_message"] = str(cause)
        return payload

    def __reduce__(self) -> tuple[Any, ...]:
        """
        Support pickling across process boundaries.

        Returns
        -------
        tuple[Any, ...]
            A callable and its arguments sufficient to reconstruct the
            instance, preserving both ``args`` and instance state.
        """
        return (
            _rebuild_ptfm_error,
            (type(self), self.args, dict(self.__dict__)),
        )

    def __str__(self) -> str:
        """
        Return the human-readable message.

        Returns
        -------
        str
            The message passed at construction time.
        """
        return self.message

    def __repr__(self) -> str:
        """
        Return an unambiguous representation including the error code.

        Returns
        -------
        str
            Representation of the form ``ClassName(code=..., message=...)``.
        """
        return (
            f"{type(self).__name__}(code={self.error_code!r}, "
            f"message={self.message!r})"
        )