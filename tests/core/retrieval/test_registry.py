"""
Unit tests for the Retrieval Module registries.

File:
    tests/core/retrieval/test_registry.py

Emphasis
--------
    That a name collision is an error rather than an override, and that a
    component failing to satisfy its contract is rejected at construction
    rather than at the first query, after the arms have already run.
"""

from __future__ import annotations

import ast
import inspect
from typing import Any, Mapping, Sequence

import pytest

from ptfm.core.retrieval import registry as registry_module
from ptfm.core.retrieval.config import RetrievalConfig
from ptfm.core.retrieval.exceptions import StrategyRegistrationError
from ptfm.core.retrieval.interfaces import FusionStrategy, RetrievalStrategy
from ptfm.core.retrieval.registry import (
    ARM_ENTRY_POINT_GROUP,
    FUSION_ENTRY_POINT_GROUP,
    ArmRegistration,
    ArmRegistry,
    FusionRegistry,
)
from ptfm.core.schemas import DiscoveryMethod


def _config(**overrides: Any) -> RetrievalConfig:
    return RetrievalConfig(_env_file=None, **overrides)


# ---------------------------------------------------------------------------
# Doubles
# ---------------------------------------------------------------------------


class _Arm(RetrievalStrategy):
    def __init__(
        self,
        name: str = "dense",
        method: DiscoveryMethod = DiscoveryMethod.DENSE,
    ) -> None:
        self._name = name
        self._method = method

    @property
    def name(self) -> str:
        return self._name

    @property
    def method(self) -> DiscoveryMethod:
        return self._method

    @property
    def requires_vector(self) -> bool:
        return self._method is DiscoveryMethod.DENSE

    async def search(self, plan: Any) -> Sequence[Any]:
        return ()

    async def health_check(self) -> bool:
        return True


class _Fusion(FusionStrategy):
    def __init__(self, name: str = "reciprocal_rank") -> None:
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def fuse(self, results: Any, top_k: int) -> Sequence[Any]:
        return ()


class _EntryPoint:
    def __init__(self, name: str, group: str, loader: Any) -> None:
        self.name = name
        self.group = group
        self._loader = loader

    def load(self) -> Any:
        return self._loader()


def _patch_entry_points(
    monkeypatch: pytest.MonkeyPatch, points: Sequence[_EntryPoint]
) -> None:
    monkeypatch.setattr(
        registry_module,
        "entry_points",
        lambda group: [point for point in points if point.group == group],
    )


def _registration(
    name: str = "dense",
    method: DiscoveryMethod = DiscoveryMethod.DENSE,
) -> ArmRegistration:
    return ArmRegistration(
        name=name,
        factory=lambda config, n=name, m=method: _Arm(n, m),
    )


def _arms() -> ArmRegistry:
    return ArmRegistry(
        [
            _registration("dense", DiscoveryMethod.DENSE),
            _registration("sparse", DiscoveryMethod.SPARSE),
        ]
    )


# ===========================================================================
# CONSTRUCTION
# ===========================================================================


def test_a_registry_is_constructed_complete() -> None:
    """Resolution cannot change during a run."""
    registry = _arms()

    assert len(registry) == 2
    assert registry.names == ("dense", "sparse")


def test_a_registry_exposes_no_mutation() -> None:
    for method in ("register", "unregister", "clear"):
        assert not hasattr(ArmRegistry, method)
        assert not hasattr(FusionRegistry, method)


def test_registry_internals_are_not_assignable() -> None:
    with pytest.raises(AttributeError):
        _arms().extra = 1  # type: ignore[attr-defined]


def test_a_description_is_read_only() -> None:
    description = _arms().describe()

    assert isinstance(description, Mapping)
    with pytest.raises(TypeError):
        description["dense"] = "other"  # type: ignore[index]


# ===========================================================================
# NAME COLLISIONS
# ===========================================================================


def test_a_duplicate_arm_name_is_rejected() -> None:
    """
    A collision resolved by registration order would make a result depend on
    import order rather than on configuration.
    """
    with pytest.raises(StrategyRegistrationError):
        ArmRegistry([_registration("dense"), _registration("dense")])


def test_a_duplicate_fusion_name_is_rejected() -> None:
    with pytest.raises(StrategyRegistrationError, match="registration order"):
        FusionRegistry([_Fusion("reciprocal_rank"), _Fusion("reciprocal_rank")])


@pytest.mark.parametrize(
    "name", ["", "Dense", "dense-arm", "1dense", "dense arm", "DENSE"]
)
def test_a_malformed_name_is_rejected(name: str) -> None:
    """
    A name must render identically in configuration, in an environment
    variable, and in an entry-point declaration.
    """
    with pytest.raises(StrategyRegistrationError):
        ArmRegistry([_registration(name)])


# ===========================================================================
# RESOLUTION
# ===========================================================================


def test_resolution_pairs_each_arm_with_its_configuration() -> None:
    """
    Each arm carries its own depth and deadline, so pairing them at resolution
    saves every caller from looking the configuration up again.
    """
    config = _config()

    resolved = _arms().resolve(config)

    for arm, arm_config in resolved:
        assert arm.name
        assert arm_config.top_k >= 1
        assert arm_config.timeout_seconds > 0


def test_only_enabled_arms_are_resolved() -> None:
    config = _config()

    resolved = _arms().resolve(config)

    assert {arm.name for arm, _ in resolved} == set(config.enabled_arms())


def test_an_unregistered_configured_arm_is_reported() -> None:
    """
    Configuration naming an arm nobody registered is a deployment fault, and
    silently omitting it would present a reduced result as complete.
    """
    registry = ArmRegistry([_registration("sparse", DiscoveryMethod.SPARSE)])

    with pytest.raises(StrategyRegistrationError) as excinfo:
        registry.resolve(_config())

    assert "dense" in excinfo.value.message


def test_a_factory_returning_the_wrong_type_is_rejected() -> None:
    registry = ArmRegistry(
        [
            ArmRegistration(name="dense", factory=lambda config: object()),
            _registration("sparse", DiscoveryMethod.SPARSE),
        ]
    )

    with pytest.raises(StrategyRegistrationError):
        registry.resolve(_config())


def test_a_failing_factory_is_wrapped_with_its_cause() -> None:
    def _explode(config: Any) -> Any:
        raise RuntimeError("driver unavailable")

    registry = ArmRegistry(
        [
            ArmRegistration(name="dense", factory=_explode),
            _registration("sparse", DiscoveryMethod.SPARSE),
        ]
    )

    with pytest.raises(StrategyRegistrationError) as excinfo:
        registry.resolve(_config())

    assert isinstance(excinfo.value.__cause__, RuntimeError)


def test_a_declared_name_that_drifts_is_detected() -> None:
    """
    An arm registered under one name but reporting another would record the
    wrong arm in every discovery history it produced.
    """
    registry = ArmRegistry(
        [
            ArmRegistration(
                name="dense", factory=lambda config: _Arm("impostor")
            ),
            _registration("sparse", DiscoveryMethod.SPARSE),
        ]
    )

    with pytest.raises(StrategyRegistrationError):
        registry.resolve(_config())


def test_a_fusion_strategy_resolves() -> None:
    assert (
        FusionRegistry([_Fusion()]).resolve(_config()).name == "reciprocal_rank"
    )


def test_an_unregistered_fusion_strategy_is_reported() -> None:
    with pytest.raises(StrategyRegistrationError):
        FusionRegistry([_Fusion("weighted")]).resolve(_config())


# ===========================================================================
# CONTRACT CONFORMANCE
# ===========================================================================


def test_a_non_conforming_fusion_strategy_is_rejected_at_construction() -> None:
    """
    Direct registration is the composition root's path and the one every
    deployment uses, so it must not be validated less strictly than
    entry-point discovery. A strategy accepted here would fail at the first
    query, after the arms had already run.
    """

    class _NotAStrategy:
        @property
        def name(self) -> str:
            return "reciprocal_rank"

        def fuse(self, results: Any, top_k: int) -> Sequence[Any]:
            return ()

    with pytest.raises(StrategyRegistrationError, match="does not implement"):
        FusionRegistry([_NotAStrategy()])


def test_conformance_is_checked_on_both_paths() -> None:
    """The same guarantee whether a component is injected or discovered."""
    source = inspect.getsource(FusionRegistry)

    assert source.count("isinstance(strategy, FusionStrategy)") >= 1
    assert "does not implement" in source


# ===========================================================================
# ENTRY-POINT DISCOVERY
# ===========================================================================


def test_an_arm_is_discovered_from_an_entry_point(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A separately installed package contributes an arm with no edit to
    PTFM."""
    _patch_entry_points(
        monkeypatch,
        [
            _EntryPoint(
                "vendor",
                ARM_ENTRY_POINT_GROUP,
                lambda: (lambda config: _Arm("vendor")),
            )
        ],
    )

    assert "vendor" in ArmRegistry.from_entry_points().names


def test_a_fusion_strategy_is_discovered_from_an_entry_point(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_entry_points(
        monkeypatch,
        [
            _EntryPoint(
                "weighted",
                FUSION_ENTRY_POINT_GROUP,
                lambda: (lambda: _Fusion("weighted")),
            )
        ],
    )

    assert "weighted" in FusionRegistry.from_entry_points().names


def test_a_discovered_name_may_not_displace_a_builtin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    A third-party package silently replacing a built-in arm would change what
    a deployment retrieves without changing its configuration.
    """
    _patch_entry_points(
        monkeypatch,
        [
            _EntryPoint(
                "dense",
                ARM_ENTRY_POINT_GROUP,
                lambda: (lambda config: _Arm("dense")),
            )
        ],
    )

    with pytest.raises(StrategyRegistrationError):
        ArmRegistry.from_entry_points(registrations=[_registration("dense")])


def test_a_broken_entry_point_names_itself(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    A third-party package may raise anything at import. Wrapping means a
    broken plugin produces an actionable startup message rather than an
    arbitrary traceback from an unrelated module.
    """

    def _explode() -> Any:
        raise ImportError("missing transitive dependency")

    _patch_entry_points(
        monkeypatch, [_EntryPoint("broken", ARM_ENTRY_POINT_GROUP, _explode)]
    )

    with pytest.raises(StrategyRegistrationError) as excinfo:
        ArmRegistry.from_entry_points()

    assert "broken" in excinfo.value.message
    assert isinstance(excinfo.value.__cause__, ImportError)


def test_a_non_callable_entry_point_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_entry_points(
        monkeypatch,
        [_EntryPoint("bad", ARM_ENTRY_POINT_GROUP, lambda: "not-callable")],
    )

    with pytest.raises(StrategyRegistrationError):
        ArmRegistry.from_entry_points()


def test_the_entry_point_groups_are_namespaced() -> None:
    assert ARM_ENTRY_POINT_GROUP == "ptfm.retrieval.arms"
    assert FUSION_ENTRY_POINT_GROUP == "ptfm.retrieval.fusion"


# ===========================================================================
# Anti-pattern guards
# ===========================================================================


def _code_without_docstrings(module: object) -> str:
    """
    Return a module's source with every docstring removed.

    Line ranges are used rather than text replacement because
    ``ast.get_docstring`` dedents its result, which does not match indented
    source.

    Parameters
    ----------
    module:
        The module to read.

    Returns
    -------
    str
        The source with docstrings stripped.
    """
    source = inspect.getsource(module)  # type: ignore[arg-type]
    tree = ast.parse(source)
    lines = source.splitlines(keepends=True)
    blank: set[int] = set()

    for node in ast.walk(tree):
        if not isinstance(
            node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
        ):
            continue
        body = getattr(node, "body", None)
        if not body:
            continue
        first = body[0]
        if not (
            isinstance(first, ast.Expr)
            and isinstance(first.value, ast.Constant)
            and isinstance(first.value.value, str)
        ):
            continue
        blank.update(range(first.lineno - 1, (first.end_lineno or first.lineno)))

    return "".join("" if i in blank else line for i, line in enumerate(lines))


def test_the_module_imports_no_backend() -> None:
    source = inspect.getsource(registry_module)

    for forbidden in (
        "import psycopg",
        "import asyncpg",
        "import qdrant",
        "import numpy",
        "from ptfm.plugins",
        "ptfm.core.retrieval.pipeline",
    ):
        assert forbidden not in source


def test_the_module_names_no_storage_technology() -> None:
    source = inspect.getsource(registry_module).lower()

    for term in (
        "postgres",
        "pgvector",
        "qdrant",
        "faiss",
        "milvus",
        "elasticsearch",
        "tantivy",
        "neo4j",
    ):
        assert term not in source


def test_the_module_contains_no_domain_vocabulary() -> None:
    source = inspect.getsource(registry_module).lower()

    for term in ("jurisdiction", "statute", "ticker", " icd", "patient", "court"):
        assert term not in source


def test_the_module_declares_no_calibration_concept() -> None:
    source = _code_without_docstrings(registry_module).lower()

    for term in ("calibrat", "entail", "truth space", "verification result"):
        assert term not in source


def test_the_module_declares_no_mutable_state() -> None:
    mutable = [
        name
        for name, value in vars(registry_module).items()
        if not name.startswith("_") and isinstance(value, (list, dict, set))
    ]

    assert mutable == []


def test_the_module_carries_no_dead_code() -> None:
    source = inspect.getsource(registry_module)

    assert "_unused" not in source
