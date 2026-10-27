"""
Unit tests for result hydration and supersession filtering.

File:
    tests/core/retrieval/test_hydration.py

Emphasis
--------
    Hydration is the single point at which content enters a result, so every
    provenance guarantee the module makes is established here or not at all.
    Supersession is the point at which stale evidence is withheld without being
    made unauditable.
"""

from __future__ import annotations

import ast
import inspect
from datetime import UTC, datetime
from typing import Any, Mapping, Optional, Sequence

import pytest
from pydantic import ValidationError

from ptfm.core.retrieval.config import SupersessionConfig
from ptfm.core.retrieval.exceptions import (
    HydrationError,
    NodeNotFoundError,
    ProvenanceIncompleteError,
    SupersessionError,
)
from ptfm.core.retrieval.models import FusedCandidate, SegmentationPartition
from ptfm.core.retrieval.pipeline import hydration as hydration_module
from ptfm.core.retrieval.pipeline.hydration import (
    HydratedResult,
    NodeHydrator,
    SupersessionFilter,
)
from ptfm.core.schemas import (
    DiscoveryMethod,
    KnowledgeNode,
    NodeType,
    RetrievalSignal,
    SignalScale,
    SourceDescriptor,
    TextSpan,
)

NOW = datetime(2026, 9, 23, tzinfo=UTC)

SOURCE_A = "file:///corpus/a.pdf"
SOURCE_B = "file:///corpus/b.pdf"


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def _partition() -> SegmentationPartition:
    return SegmentationPartition(
        segmentation_fingerprint="1:abc", source_schema_version="1.0"
    )


def _source(uri: str = SOURCE_A) -> SourceDescriptor:
    return SourceDescriptor(
        source_uri=uri,
        scheme="file",
        connector_name="LocalFilesystemConnector",
        extractor_name="PdfExtractor",
        acquired_at=NOW,
        size_bytes=1024,
        detected_media_type="application/pdf",
    )


def _node(
    node_id: str = "node-1",
    *,
    document_id: str = "doc-1",
    ordinal: int = 0,
    source_uri: str = SOURCE_A,
    fingerprint: str = "1:abc",
    schema_version: str = "1.0",
    text: str = "Alpha",
) -> KnowledgeNode:
    return KnowledgeNode(
        node_id=node_id,
        document_id=document_id,
        span=TextSpan(start=0, end=len(text)),
        text=text,
        node_type=NodeType.PROSE,
        block_ids=("0",),
        ordinal=ordinal,
        source=_source(source_uri),
        segmentation_fingerprint=fingerprint,
        source_text_hash="text-hash",
        source_schema_version=schema_version,
        processed_at=NOW,
    )


def _candidate(node_id: str, rank: int = 0) -> FusedCandidate:
    return FusedCandidate(
        node_id=node_id,
        signals=(
            RetrievalSignal(
                method=DiscoveryMethod.DENSE,
                rank=rank,
                raw_score=0.9 - rank * 0.1,
                scale=SignalScale.COSINE,
            ),
        ),
        fused_rank=rank,
        fused_score=1.0 - rank * 0.1,
    )


def _result(
    node_id: str, document_id: str, source_uri: str, rank: int = 0
) -> HydratedResult:
    return HydratedResult(
        candidate=_candidate(node_id, rank),
        node=_node(node_id, document_id=document_id, source_uri=source_uri, ordinal=rank),
    )


class _Logger:
    def __init__(self) -> None:
        self.events: list[str] = []

    def bind(self, **values: Any) -> _Logger:
        return self

    def debug(self, event: str, **values: Any) -> None:
        self.events.append(event)

    def info(self, event: str, **values: Any) -> None:
        self.events.append(event)

    def warning(self, event: str, **values: Any) -> None:
        self.events.append(event)

    def error(self, event: str, **values: Any) -> None:
        self.events.append(event)


class _NodeStore:
    """Records how it was called, so batching is observable."""

    def __init__(
        self,
        nodes: Sequence[KnowledgeNode] = (),
        *,
        failure: Optional[Exception] = None,
    ) -> None:
        self._nodes = {node.node_id: node for node in nodes}
        self._failure = failure
        self.calls: list[tuple[str, ...]] = []

    @property
    def name(self) -> str:
        return "node"

    async def get_many(
        self, node_ids: Sequence[str], partition: Any
    ) -> Mapping[str, KnowledgeNode]:
        self.calls.append(tuple(node_ids))
        if self._failure is not None:
            raise self._failure
        return {
            node_id: self._nodes[node_id]
            for node_id in node_ids
            if node_id in self._nodes
        }

    async def get_by_ordinal(
        self, document_id: str, ordinals: Sequence[int], partition: Any
    ) -> Mapping[int, KnowledgeNode]:
        return {}

    async def put_many(self, nodes: Any, partition: Any) -> None:
        return None

    async def count(self, partition: Any) -> int:
        return len(self._nodes)


class _SupersessionStore:
    def __init__(
        self,
        mapping: Optional[Mapping[str, Optional[str]]] = None,
        *,
        failure: Optional[Exception] = None,
    ) -> None:
        self._mapping = dict(mapping or {})
        self._failure = failure
        self.calls: list[tuple[str, ...]] = []

    @property
    def name(self) -> str:
        return "supersession"

    async def current_document_ids(
        self, source_uris: Sequence[str], partition: Any
    ) -> Mapping[str, Optional[str]]:
        self.calls.append(tuple(source_uris))
        if self._failure is not None:
            raise self._failure
        return {
            uri: self._mapping[uri] for uri in source_uris if uri in self._mapping
        }

    async def record_current(
        self,
        source_uri: str,
        document_id: Optional[str],
        version_tag: Optional[str],
        partition: Any,
    ) -> None:
        return None


def _hydrator(store: Optional[_NodeStore] = None) -> NodeHydrator:
    return NodeHydrator(store or _NodeStore(), _Logger())


def _filter(
    mapping: Optional[Mapping[str, Optional[str]]] = None,
    *,
    failure: Optional[Exception] = None,
    logger: Optional[_Logger] = None,
    **config: Any,
) -> SupersessionFilter:
    return SupersessionFilter(
        _SupersessionStore(mapping, failure=failure),
        SupersessionConfig(**config),
        logger or _Logger(),
    )


# ===========================================================================
# HYDRATION
# ===========================================================================


async def test_hydration_resolves_candidates_to_authoritative_nodes() -> None:
    nodes = [_node("node-0", ordinal=0), _node("node-1", ordinal=1)]

    results = await _hydrator(_NodeStore(nodes)).hydrate(
        [_candidate("node-0", 0), _candidate("node-1", 1)], _partition()
    )

    assert [result.node_id for result in results] == ["node-0", "node-1"]
    assert all(isinstance(result, HydratedResult) for result in results)


async def test_hydration_preserves_fused_order() -> None:
    """Ranking was decided by fusion; hydration must not reorder it."""
    nodes = [_node("node-0"), _node("node-1"), _node("node-2")]
    candidates = [_candidate("node-2", 0), _candidate("node-0", 1), _candidate("node-1", 2)]

    results = await _hydrator(_NodeStore(nodes)).hydrate(candidates, _partition())

    assert [result.node_id for result in results] == ["node-2", "node-0", "node-1"]


async def test_content_comes_from_the_node_store() -> None:
    """
    An index may hold its own copy of a node's text. If that copy were
    consulted, a stale index could change what a citation resolves to and no
    downstream check would notice.
    """
    results = await _hydrator(_NodeStore([_node("node-0", text="Authoritative")])).hydrate(
        [_candidate("node-0")], _partition()
    )

    assert results[0].node.text == "Authoritative"


async def test_hydration_resolves_in_one_batch() -> None:
    """
    A per-candidate round trip would multiply the store's latency by the result
    count on the hottest path in the module.
    """
    store = _NodeStore([_node(f"node-{index}") for index in range(5)])

    await _hydrator(store).hydrate(
        [_candidate(f"node-{index}", index) for index in range(5)], _partition()
    )

    assert len(store.calls) == 1
    assert len(store.calls[0]) == 5


async def test_a_hydrated_result_exposes_its_provenance() -> None:
    results = await _hydrator(_NodeStore([_node("node-0")])).hydrate(
        [_candidate("node-0")], _partition()
    )

    assert results[0].document_id == "doc-1"
    assert results[0].source_uri == SOURCE_A


async def test_a_hydrated_result_retains_every_retrieval_signal() -> None:
    """Provenance of discovery survives hydration, not only provenance of
    content."""
    results = await _hydrator(_NodeStore([_node("node-0")])).hydrate(
        [_candidate("node-0")], _partition()
    )

    assert results[0].candidate.signals[0].method is DiscoveryMethod.DENSE


async def test_hydrating_nothing_yields_nothing() -> None:
    assert await _hydrator().hydrate([], _partition()) == ()


# ---------------------------------------------------------------------------
# Divergence and failure
# ---------------------------------------------------------------------------


async def test_an_unresolvable_identifier_is_surfaced() -> None:
    """
    An index and the node store have diverged. Silently dropping the result
    would present an inconsistent store as a merely less relevant corpus, and
    the inconsistency would persist unnoticed.
    """
    store = _NodeStore([_node("node-0")])

    with pytest.raises(NodeNotFoundError) as excinfo:
        await _hydrator(store).hydrate(
            [_candidate("node-0"), _candidate("ghost", 1)], _partition()
        )

    assert excinfo.value.missing_node_ids == ("ghost",)
    assert excinfo.value.degradable is False


async def test_a_store_failure_is_fatal_and_carries_its_cause() -> None:
    """
    Arms return identifiers, so without the node store there is no text, span,
    or source: candidates, not evidence.
    """
    store = _NodeStore(failure=RuntimeError("connection reset"))

    with pytest.raises(HydrationError) as excinfo:
        await _hydrator(store).hydrate([_candidate("node-0")], _partition())

    assert isinstance(excinfo.value.__cause__, RuntimeError)
    assert excinfo.value.degradable is False


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("field", ["span", "text", "source", "document_id"])
async def test_a_node_missing_provenance_cannot_become_evidence(field: str) -> None:
    """
    The last guard on the Evidence-First chain inside retrieval. A claim citing
    such a node could not be traced to a region of an identified document.
    """
    node = _node("node-0")
    object.__setattr__(node, field, None)

    with pytest.raises(ProvenanceIncompleteError) as excinfo:
        await _hydrator(_NodeStore([node])).hydrate([_candidate("node-0")], _partition())

    assert field in excinfo.value.missing_fields


async def test_a_node_whose_source_has_no_uri_is_refused() -> None:
    """The document it came from cannot be named."""
    node = _node("node-0")
    object.__setattr__(node, "source", _source(""))

    with pytest.raises(ProvenanceIncompleteError):
        await _hydrator(_NodeStore([node])).hydrate([_candidate("node-0")], _partition())


async def test_provenance_verification_has_no_off_switch() -> None:
    """
    Completeness of provenance is the guarantee the module exists to provide, so
    a switch could only ever be used to admit evidence that cannot be cited.
    """
    parameters = inspect.signature(NodeHydrator.__init__).parameters

    assert "validation" not in parameters
    assert "verify" not in parameters


@pytest.mark.parametrize(
    "override",
    [{"fingerprint": "1:other"}, {"schema_version": "2.0"}],
)
async def test_a_node_from_another_partition_is_refused(
    override: dict[str, Any],
) -> None:
    """
    Its spans were computed against a different division of its document, so its
    offsets do not mean what they say here.
    """
    node = _node("node-0", **override)

    with pytest.raises(ProvenanceIncompleteError, match="not the partition"):
        await _hydrator(_NodeStore([node])).hydrate([_candidate("node-0")], _partition())


# ===========================================================================
# SUPERSESSION
# ===========================================================================


async def test_a_superseded_document_is_withheld() -> None:
    """Its source is now represented by a different document."""
    results = [
        _result("node-1", "doc-old", SOURCE_A, 0),
        _result("node-2", "doc-new", SOURCE_A, 1),
    ]

    kept, withheld, unchecked = await _filter({SOURCE_A: "doc-new"}).filter(
        results, _partition()
    )

    assert [result.node_id for result in kept] == ["node-2"]
    assert withheld == 1
    assert unchecked == 0


async def test_surviving_results_keep_their_order() -> None:
    results = [
        _result("node-1", "doc-new", SOURCE_A, 0),
        _result("node-2", "doc-old", SOURCE_A, 1),
        _result("node-3", "doc-b", SOURCE_B, 2),
    ]

    kept, _, _ = await _filter(
        {SOURCE_A: "doc-new", SOURCE_B: "doc-b"}
    ).filter(results, _partition())

    assert [result.node_id for result in kept] == ["node-1", "node-3"]


async def test_currency_is_looked_up_once_per_distinct_source() -> None:
    """
    A result set ordinarily spans far fewer sources than results, and a
    per-result lookup would repeat the same question.
    """
    store = _SupersessionStore({SOURCE_A: "doc-1"})
    subject = SupersessionFilter(store, SupersessionConfig(), _Logger())

    await subject.filter(
        [_result(f"node-{index}", "doc-1", SOURCE_A, index) for index in range(4)],
        _partition(),
    )

    assert len(store.calls) == 1
    assert len(store.calls[0]) == 1


async def test_a_withdrawn_source_is_withheld_by_default() -> None:
    """A tombstone: the source was removed and has no successor."""
    kept, withheld, _ = await _filter({SOURCE_A: None}).filter(
        [_result("node-1", "doc-1", SOURCE_A)], _partition()
    )

    assert kept == ()
    assert withheld == 1


async def test_a_withdrawn_source_can_be_admitted() -> None:
    kept, withheld, _ = await _filter(
        {SOURCE_A: None}, include_withdrawn=True
    ).filter([_result("node-1", "doc-1", SOURCE_A)], _partition())

    assert len(kept) == 1
    assert withheld == 0


async def test_admitting_withdrawn_content_never_admits_a_superseded_version() -> None:
    """
    The asymmetry that matters: a superseded document has a successor, so
    including it would offer two versions of one source as though both were
    current. No setting admits that.
    """
    kept, withheld, _ = await _filter(
        {SOURCE_A: "doc-new"}, include_withdrawn=True
    ).filter([_result("node-1", "doc-old", SOURCE_A)], _partition())

    assert kept == ()
    assert withheld == 1


async def test_a_source_with_no_record_passes_and_is_counted() -> None:
    """
    Not known to be superseded. Dropping it would discard valid evidence on the
    strength of a record that does not exist, and the loss would be invisible.
    """
    logger = _Logger()

    kept, withheld, unchecked = await _filter({}, logger=logger).filter(
        [_result("node-1", "doc-1", SOURCE_A)], _partition()
    )

    assert len(kept) == 1
    assert withheld == 0
    assert unchecked == 1
    assert "retrieval.supersession.unrecorded_sources" in logger.events


async def test_a_tombstone_is_distinguishable_from_no_record() -> None:
    """
    Collapsing the two would make a deletion indistinguishable from an
    omission, which is why a tombstone is stored rather than the record being
    removed.
    """
    result = [_result("node-1", "doc-1", SOURCE_A)]

    _, withdrawn_withheld, _ = await _filter({SOURCE_A: None}).filter(
        result, _partition()
    )
    _, absent_withheld, absent_unchecked = await _filter({}).filter(
        result, _partition()
    )

    assert withdrawn_withheld == 1
    assert absent_withheld == 0
    assert absent_unchecked == 1


async def test_a_supersession_failure_is_fatal() -> None:
    """
    Proceeding would serve superseded content as current evidence, which no
    consumer could detect.
    """
    subject = _filter(failure=RuntimeError("store down"))

    with pytest.raises(SupersessionError) as excinfo:
        await subject.filter([_result("node-1", "doc-1", SOURCE_A)], _partition())

    assert isinstance(excinfo.value.__cause__, RuntimeError)
    assert excinfo.value.degradable is False


async def test_filtering_nothing_short_circuits() -> None:
    assert await _filter().filter([], _partition()) == ((), 0, 0)


def test_supersession_filtering_cannot_be_disabled() -> None:
    """
    Serving superseded content as current is a truthfulness failure invisible to
    every consumer, so the setting exists to make the guarantee visible rather
    than to offer a choice.
    """
    with pytest.raises(ValidationError):
        SupersessionConfig(filter_superseded=False)


def test_the_filter_does_not_branch_on_a_constant_setting() -> None:
    """
    Unreachable code that appears to offer a choice is worse than no code at
    all.
    """
    source = inspect.getsource(hydration_module)

    assert "not self._config.filter_superseded" not in source


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


def test_hydration_reads_content_from_no_index() -> None:
    source = inspect.getsource(hydration_module)

    for forbidden in ("VectorIndex", "LexicalIndex", "RelationStore"):
        assert forbidden not in source


def test_the_module_imports_no_backend() -> None:
    source = inspect.getsource(hydration_module)

    for forbidden in (
        "import psycopg",
        "import asyncpg",
        "import qdrant",
        "import numpy",
        "from ptfm.plugins",
    ):
        assert forbidden not in source


def test_the_module_names_no_storage_technology() -> None:
    source = inspect.getsource(hydration_module).lower()

    for term in ("postgres", "pgvector", "qdrant", "faiss", "elasticsearch", "neo4j"):
        assert term not in source


def test_the_module_contains_no_domain_vocabulary() -> None:
    source = inspect.getsource(hydration_module).lower()

    for term in ("jurisdiction", "statute", "ticker", " icd", "patient", "court"):
        assert term not in source


def test_the_module_declares_no_calibrated_concept() -> None:
    source = _code_without_docstrings(hydration_module).lower()

    for term in ("confidence", "calibrat", "entail", "truth space"):
        assert term not in source


def test_the_module_declares_no_mutable_state() -> None:
    mutable = [
        name
        for name, value in vars(hydration_module).items()
        if not name.startswith("_") and isinstance(value, (list, dict, set))
    ]

    assert mutable == []


def test_the_module_never_catches_base_exception() -> None:
    assert "except BaseException" not in inspect.getsource(hydration_module)


def test_the_stages_hold_no_assignable_state() -> None:
    """One instance serves all concurrent queries."""
    with pytest.raises(AttributeError):
        _hydrator().cache = {}  # type: ignore[attr-defined]

    with pytest.raises(AttributeError):
        _filter().cache = {}  # type: ignore[attr-defined]
