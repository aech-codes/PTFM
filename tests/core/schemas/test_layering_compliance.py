"""
Architecture-compliance tests for the schema package layering.

File:
    tests/core/schemas/test_layering_compliance.py

Purpose
-------
    The schema package is layered by pipeline position with a strictly acyclic
    dependency graph. That rule is invisible in the source: nothing prevents a
    contributor adding an import that inverts it, and a cycle would surface
    only as an obscure import error in an unrelated module.

    These tests state the rule as an executable constraint, so that a
    violation fails the build rather than accumulating.

Analysis method
---------------
    Imports are resolved from the abstract syntax tree rather than by
    importing the modules. Static analysis catches a violation even when the
    offending import is inside a ``TYPE_CHECKING`` guard or a function body,
    both of which would evade a runtime check while still expressing a
    dependency the layering forbids.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

import ptfm.core.schemas as schemas_package

#: Directory holding the layer modules.
PACKAGE_ROOT = Path(schemas_package.__file__).parent

#: Layers in dependency order, lowest first. A module may import from any
#: layer preceding it and none following it.
LAYER_ORDER: tuple[str, ...] = (
    "common",
    "ingestion",
    "knowledge",
    "retrieval",
    "verification",
)

#: Fully qualified package prefix, used to identify intra-package imports.
PACKAGE_PREFIX = "ptfm.core.schemas"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _layer_path(layer: str) -> Path:
    """
    Return the file backing a layer.

    Parameters
    ----------
    layer:
        Layer name.

    Returns
    -------
    Path
        Path to the layer module.
    """
    return PACKAGE_ROOT / f"{layer}.py"


def _intra_package_imports(path: Path) -> set[str]:
    """
    Collect the layers a module imports from within this package.

    Both absolute and relative import forms are resolved, and imports nested
    inside ``TYPE_CHECKING`` guards or function bodies are included, since
    each expresses a dependency the layering governs.

    Parameters
    ----------
    path:
        Module to analyse.

    Returns
    -------
    set[str]
        Layer names imported by the module.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imported: set[str] = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.level > 0 and node.module in LAYER_ORDER:
                # Relative form: ``from .ingestion import ...``
                imported.add(node.module)
            elif node.module and node.module.startswith(f"{PACKAGE_PREFIX}."):
                imported.add(node.module.removeprefix(f"{PACKAGE_PREFIX}."))
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith(f"{PACKAGE_PREFIX}."):
                    imported.add(alias.name.removeprefix(f"{PACKAGE_PREFIX}."))

    return {name for name in imported if name in LAYER_ORDER}


def _permitted_layers(layer: str) -> set[str]:
    """
    Return the layers a given layer may import from.

    Parameters
    ----------
    layer:
        Layer name.

    Returns
    -------
    set[str]
        Every layer strictly below the given one.
    """
    return set(LAYER_ORDER[: LAYER_ORDER.index(layer)])


def _declared_exports(path: Path) -> list[str]:
    """
    Read a module's ``__all__`` without importing it.

    Parameters
    ----------
    path:
        Module to analyse.

    Returns
    -------
    list[str]
        Declared export names, empty when ``__all__`` is absent or is not a
        literal list of strings.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

    for node in tree.body:
        targets = (
            node.targets
            if isinstance(node, ast.Assign)
            else [node.target]
            if isinstance(node, ast.AnnAssign)
            else []
        )
        for target in targets:
            if isinstance(target, ast.Name) and target.id == "__all__":
                value = node.value
                if isinstance(value, (ast.List, ast.Tuple)):
                    return [
                        element.value
                        for element in value.elts
                        if isinstance(element, ast.Constant)
                        and isinstance(element.value, str)
                    ]
    return []


# ---------------------------------------------------------------------------
# Layering
# ---------------------------------------------------------------------------


def test_every_layer_exists() -> None:
    for layer in LAYER_ORDER:
        assert _layer_path(layer).is_file(), layer


@pytest.mark.parametrize("layer", LAYER_ORDER)
def test_a_layer_imports_only_from_below(layer: str) -> None:
    """The rule the package exists to make enforceable."""
    imported = _intra_package_imports(_layer_path(layer))
    permitted = _permitted_layers(layer)

    violations = imported - permitted

    assert not violations, (
        f"{layer}.py imports from {sorted(violations)}, which is at or above "
        f"its own layer. Permitted: {sorted(permitted) or 'none'}."
    )


def test_the_base_layer_has_no_intra_package_imports() -> None:
    assert _intra_package_imports(_layer_path("common")) == set()


def test_the_dependency_graph_is_acyclic() -> None:
    """
    Verified independently of the ordering rule.

    A layering violation and a cycle are distinct failures. Checking both
    means a future revision to the permitted ordering cannot silently
    introduce a cycle.
    """
    graph = {layer: _intra_package_imports(_layer_path(layer)) for layer in LAYER_ORDER}
    visiting: set[str] = set()
    visited: set[str] = set()

    def _walk(layer: str, path: tuple[str, ...]) -> None:
        if layer in visiting:
            raise AssertionError(f"Import cycle: {' -> '.join((*path, layer))}")
        if layer in visited:
            return
        visiting.add(layer)
        for dependency in sorted(graph[layer]):
            _walk(dependency, (*path, layer))
        visiting.discard(layer)
        visited.add(layer)

    for layer in LAYER_ORDER:
        _walk(layer, ())


def test_no_layer_imports_the_package_facade() -> None:
    """Importing the facade from a layer would create a cycle through it."""
    for layer in LAYER_ORDER:
        source = _layer_path(layer).read_text(encoding="utf-8")
        assert "from ptfm.core.schemas import" not in source, layer


def test_the_known_dependencies_are_present() -> None:
    """
    Pins the intended shape, so that an accidental removal is visible.

    ``ingestion`` indexes into normalized text with the common span type, and
    ``verification`` composes a retrieval metric, so both dependencies are
    expected rather than incidental.
    """
    assert "common" in _intra_package_imports(_layer_path("ingestion"))
    assert "retrieval" in _intra_package_imports(_layer_path("verification"))


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------


def test_the_facade_exports_the_union_of_its_layers() -> None:
    declared: set[str] = set()
    for layer in LAYER_ORDER:
        declared.update(_declared_exports(_layer_path(layer)))

    exported = set(schemas_package.__all__)

    assert declared - exported == set(), (
        f"Declared in a layer but absent from the package interface: "
        f"{sorted(declared - exported)}"
    )
    assert exported - declared == set(), (
        f"Exported by the package but declared in no layer: "
        f"{sorted(exported - declared)}"
    )


def test_every_exported_name_resolves() -> None:
    for name in schemas_package.__all__:
        assert hasattr(schemas_package, name), name


def test_exports_are_unique() -> None:
    assert len(schemas_package.__all__) == len(set(schemas_package.__all__))


def test_no_name_is_declared_by_two_layers() -> None:
    """A duplicated name would make the resolved export order-dependent."""
    seen: dict[str, str] = {}
    for layer in LAYER_ORDER:
        for name in _declared_exports(_layer_path(layer)):
            assert name not in seen, (
                f"{name!r} is declared by both {seen[name]}.py and {layer}.py"
            )
            seen[name] = layer


def test_the_facade_uses_explicit_imports() -> None:
    """Wildcard re-export would make the public surface incidental."""
    source = (PACKAGE_ROOT / "__init__.py").read_text(encoding="utf-8")

    assert "import *" not in source


# ---------------------------------------------------------------------------
# Backward compatibility
# ---------------------------------------------------------------------------


MODULE_ONE_PUBLIC_API: tuple[str, ...] = (
    "RawDocument",
    "RawDocumentBatch",
    "DocumentBlock",
    "BlockAttributes",
    "BlockType",
    "BlockIndex",
    "TextSpan",
    "SourceDescriptor",
    "DocumentMetadata",
    "ExtractionWarning",
    "ExtractionWarningCode",
    "CONTAINER_BLOCK_TYPES",
    "PTFM_DOCUMENT_NAMESPACE",
    "RAW_DOCUMENT_SCHEMA_VERSION",
    "derive_document_id",
    "UserQuery",
)

PLUGIN_INTERFACE_API: tuple[str, ...] = (
    "AtomicClaim",
    "EvidenceItem",
    "RetrievalRequest",
    "RetrievalResult",
)


@pytest.mark.parametrize("name", MODULE_ONE_PUBLIC_API)
def test_module_one_imports_are_unchanged(name: str) -> None:
    """Every name Module 1 imports must resolve from the package as before."""
    assert hasattr(schemas_package, name), name


@pytest.mark.parametrize("name", PLUGIN_INTERFACE_API)
def test_the_plugin_interface_is_unchanged(name: str) -> None:
    """``ptfm/plugins/base.py`` must require no modification."""
    assert hasattr(schemas_package, name), name


def test_the_frozen_provenance_record_is_intact() -> None:
    """The extractor provenance field added at the close of Module 1."""
    fields = schemas_package.SourceDescriptor.model_fields

    for name in (
        "source_uri",
        "scheme",
        "connector_name",
        "extractor_name",
        "acquired_at",
        "size_bytes",
        "detected_media_type",
        "declared_media_type",
        "encoding",
        "version_tag",
    ):
        assert name in fields, name


def test_the_evidence_first_span_type_is_shared() -> None:
    """
    A single span type across layers is what keeps the evidence chain
    unbroken. Two structurally similar types would silently diverge.
    """
    from ptfm.core.schemas.common import TextSpan as CommonSpan

    assert schemas_package.TextSpan is CommonSpan
    assert (
        schemas_package.DocumentBlock.model_fields["span"].annotation is CommonSpan
    )


def test_the_knowledge_layer_reuses_the_layers_below_it() -> None:
    """
    Declaring a parallel span or provenance type would break the evidence
    chain into disconnected vocabularies. The knowledge layer must reuse
    both.
    """
    from ptfm.core.schemas import knowledge

    assert knowledge.__all__, "knowledge layer declares no exports"
    assert _intra_package_imports(_layer_path("knowledge")) == {
        "common",
        "ingestion",
    }


def test_the_node_reuses_the_shared_span_and_provenance_types() -> None:
    from ptfm.core.schemas.common import TextSpan
    from ptfm.core.schemas.ingestion import SourceDescriptor

    fields = schemas_package.KnowledgeNode.model_fields

    assert fields["span"].annotation is TextSpan
    assert fields["source"].annotation is SourceDescriptor


# ---------------------------------------------------------------------------
# Documentation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("layer", LAYER_ORDER)
def test_every_layer_documents_its_position(layer: str) -> None:
    """The rule must be legible from the file, not only from this test."""
    tree = ast.parse(_layer_path(layer).read_text(encoding="utf-8"))
    docstring = ast.get_docstring(tree) or ""

    assert docstring, f"{layer}.py has no module docstring"
    assert "Layer:" in docstring, f"{layer}.py does not state its layer"


def test_the_ingestion_layer_records_its_frozen_status() -> None:
    tree = ast.parse(_layer_path("ingestion").read_text(encoding="utf-8"))
    docstring = ast.get_docstring(tree) or ""

    assert "FROZEN" in docstring
