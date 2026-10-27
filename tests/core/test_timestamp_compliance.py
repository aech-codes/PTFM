# tests/core/test_timestamp_compliance.py
"""Repository-wide guard against naive timestamp defaults."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from ptfm.core.schemas import UserQuery

PACKAGE_ROOT = Path(__file__).resolve().parents[2] / "ptfm"


def test_user_query_timestamp_is_timezone_aware() -> None:
    assert UserQuery(query="q").timestamp.tzinfo is not None


def test_no_module_calls_deprecated_utcnow() -> None:
    offenders: list[str] = []
    for path in PACKAGE_ROOT.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Attribute)
                and node.attr in {"utcnow", "utcfromtimestamp"}
            ):
                offenders.append(f"{path.name}:{node.lineno}")
    assert offenders == [], f"Deprecated naive datetime APIs: {offenders}"


@pytest.mark.parametrize("model_name", ["UserQuery"])
def test_datetime_defaults_produce_aware_instants(model_name: str) -> None:
    from ptfm.core import schemas

    model = getattr(schemas, model_name)
    instance = model(query="q") if model_name == "UserQuery" else model()
    for field_name, field in model.model_fields.items():
        if field.default_factory is None:
            continue
        value = getattr(instance, field_name)
        if hasattr(value, "tzinfo"):
            assert value.tzinfo is not None, field_name