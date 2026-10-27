"""
Unit tests for credibility annotation and evidence assembly.

File:
    tests/core/retrieval/test_assembly.py

Emphasis
--------
    Two things this stage must never do: invent a credibility, and let an
    accompanying passage pass for an independent finding. The first would put a
    number nobody measured inside a statistical guarantee; the second would
    inflate corroboration, which is the harm removing overlapping nodes was
    meant to prevent.
"""

from __future__ import annotations

import ast
import inspect
import math
from datetime import UTC, datetime
from typing import Any, Mapping, Optional, Sequence

import pytest

from ptfm.core.retrieval.config import CredibilityConfig
from ptfm.core.retrieval.exceptions import (
    AnchorViolationError,
    CorroborationIntegrityError,
    CredibilityUnavailableError,
    EvidenceAssemblyError,
)
from ptfm.core.retrieval.models import FusedCandidate
from ptfm.core.retrieval.pipeline import assembly as assembly_module
from ptfm.core.retrieval.pipeline.assembly import (
    CredibilityAnnotator,
    EvidenceAssembler,
)
from ptfm.core.retrieval.pipeline.hydration import HydratedResult
from ptfm.core.schemas import (
    DiscoveryMethod,
    KnowledgeNode,
    NodeType,
    RetrievalMethod,
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


def _node(
    node_id: str,
    *,
    document_id: str = "doc-1",
    ordinal: int = 0,
    source_uri: str = SOURCE_A,
    text: str = "Alpha",
    heading_path: Sequence[str] = (),
) -> KnowledgeNode:
    return KnowledgeNode(
        node_id=node_id,
        document_id=document_id,
        span=TextSpan(start=0, end=len(text)),
        text=text,
        node_type=NodeType.PROSE,
        block_ids=(str(ordinal),),
        ordinal=ordinal,
        source=SourceDescriptor(
            source_uri=source_uri,
            scheme="file",
            connector_name="LocalFilesystemConnector",
            extractor_name="PdfExtractor",
            acquired_at=NOW,
            size_bytes=1024,
            detected_media_type="application/pdf",
        ),
        heading_path=tuple(heading_path),
        segmentation_fingerprint="1:abc",
        source_text_hash="text-hash",
        source_schema_version="1.0",
        processed_at=NOW,
    )


def _anchor(
    node_id: str,
    fused_rank: int = 0,
    *,
    arms: Sequence[tuple[DiscoveryMethod, int, Optional[float], Optional[SignalScale]]] = (
        (DiscoveryMethod.DENSE, 0, 0.9, SignalScale.COSINE),
    ),
    **node_kwargs: Any,
) -> HydratedResult:
    return HydratedResult(
        candidate=FusedCandidate(
            node_id=node_id,
            signals=tuple(
                RetrievalSignal(method=method, rank=rank, raw_score=score, scale=scale)
                for method, rank, score, scale in arms
            ),
            fused_rank=fused_rank,
            fused_score=1.0 - fused_rank * 0.01,
        ),
        node=_node(node_id, **node_kwargs),
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


class _Provider:
    def __init__(
        self,
        reported: Optional[Mapping[str, float]] = None,
        *,
        failure: Optional[Exception] = None,
    ) -> None:
        self._reported = dict(reported or {})
        self._failure = failure

    @property
    def name(self) -> str:
        return "TestPlugin"

    async def credibility_for(
        self, source_uris: Sequence[str]
    ) -> Mapping[str, float]:
        if self._failure is not None:
            raise self._failure
        return {
            uri: self._reported[uri] for uri in source_uris if uri in self._reported
        }


def _annotator(
    provider: Optional[_Provider] = None,
    *,
    logger: Optional[_Logger] = None,
    **config: Any,
) -> CredibilityAnnotator:
    return CredibilityAnnotator(
        provider, CredibilityConfig(**config), logger or _Logger()
    )


def _assembler() -> EvidenceAssembler:
    return EvidenceAssembler(_Logger())


# ===========================================================================
# CREDIBILITY IS RECORDED, NEVER INVENTED
# ===========================================================================


async def test_a_reported_credibility_is_recorded() -> None:
    result = await _annotator(_Provider({SOURCE_A: 0.9})).credibility_for([SOURCE_A])

    assert result[SOURCE_A] == 0.9


async def test_an_unreported_source_carries_none() -> None:
    """
    Absence is represented by a None value rather than a missing key, so a
    caller cannot silently skip a source it failed to look up.
    """
    result = await _annotator(_Provider({SOURCE_A: 0.9})).credibility_for(
        [SOURCE_A, SOURCE_B]
    )

    assert SOURCE_B in result
    assert result[SOURCE_B] is None


async def test_no_provider_means_unknown_not_assumed() -> None:
    """A deployment configuring no plugin does not thereby assert that every
    source is credible."""
    result = await _annotator(None).credibility_for([SOURCE_A])

    assert result[SOURCE_A] is None


async def test_a_provider_failure_degrades_to_all_unknown() -> None:
    """
    Refusing to answer because a credibility service was unreachable would
    trade a recorded uncertainty for no answer at all.
    """
    logger = _Logger()
    annotator = _annotator(
        _Provider(failure=CredibilityUnavailableError("down", source_uri=SOURCE_A)),
        logger=logger,
    )

    result = await annotator.credibility_for([SOURCE_A, SOURCE_B])

    assert result == {SOURCE_A: None, SOURCE_B: None}
    assert "retrieval.credibility.degraded" in logger.events


async def test_an_arbitrary_plugin_exception_also_degrades() -> None:
    """A plugin is third-party code and may raise anything."""
    annotator = _annotator(_Provider(failure=RuntimeError("plugin crashed")))

    assert await annotator.credibility_for([SOURCE_A]) == {SOURCE_A: None}


@pytest.mark.parametrize("reported", [1.7, -0.2, float("nan")])
async def test_an_unusable_value_becomes_unknown_rather_than_clamped(
    reported: float,
) -> None:
    """
    Clamping would convert a plugin's error into a plausible number and feed it
    to the reliability calculation, which is the substitution this module exists
    to avoid.
    """
    logger = _Logger()

    result = await _annotator(
        _Provider({SOURCE_A: reported}), logger=logger
    ).credibility_for([SOURCE_A])

    assert result[SOURCE_A] is None
    assert "retrieval.credibility.unusable" in logger.events


async def test_credibility_can_be_disabled() -> None:
    result = await _annotator(
        _Provider({SOURCE_A: 0.9}), enabled=False
    ).credibility_for([SOURCE_A])

    assert result[SOURCE_A] is None


def test_no_default_credibility_appears_in_the_source() -> None:
    """A neutral-looking constant is the failure mode being guarded against."""
    source = inspect.getsource(assembly_module)

    assert "credibility_score=0.5" not in source
    assert "credibility_score=1.0" not in source
    assert "credibility or " not in source


# ===========================================================================
# ANCHORS
# ===========================================================================


def test_an_anchor_anchors_itself() -> None:
    """The unit of corroboration: an item found by retrieval."""
    assembled = _assembler().assemble([_anchor("node-1")], {}, {SOURCE_A: 0.9})

    item = assembled.evidence[0]
    assert item.node_id == item.anchor_node_id == "node-1"
    assert item.is_anchor is True


def test_an_anchor_records_one_discovery_per_arm() -> None:
    """
    Several records describe ONE finding observed several ways, not several
    findings.
    """
    anchor = _anchor(
        "node-1",
        arms=(
            (DiscoveryMethod.DENSE, 0, 0.9, SignalScale.COSINE),
            (DiscoveryMethod.SPARSE, 2, 7.1, SignalScale.LEXICAL),
            (DiscoveryMethod.GRAPH, 0, None, None),
        ),
    )

    item = _assembler().assemble([anchor], {}, {SOURCE_A: 0.9}).evidence[0]

    assert len(item.discovery) == 3
    assert {record.method for record in item.discovery} == {
        DiscoveryMethod.DENSE,
        DiscoveryMethod.SPARSE,
        DiscoveryMethod.GRAPH,
    }


def test_a_discovery_record_carries_its_arms_rank_score_and_scale() -> None:
    anchor = _anchor(
        "node-1",
        arms=(
            (DiscoveryMethod.DENSE, 0, 0.9, SignalScale.COSINE),
            (DiscoveryMethod.SPARSE, 4, 7.1, SignalScale.LEXICAL),
        ),
    )

    item = _assembler().assemble([anchor], {}, {SOURCE_A: 0.9}).evidence[0]

    assert {
        (record.method, record.rank, record.raw_score, record.scale)
        for record in item.discovery
    } == {
        (DiscoveryMethod.DENSE, 0, 0.9, SignalScale.COSINE),
        (DiscoveryMethod.SPARSE, 4, 7.1, SignalScale.LEXICAL),
    }


def test_an_anchor_discovery_names_no_origin_node() -> None:
    """It was not expanded from anything."""
    item = _assembler().assemble([_anchor("node-1")], {}, {SOURCE_A: 0.9}).evidence[0]

    assert all(record.via_node_id is None for record in item.discovery)


def test_a_multi_arm_anchor_still_counts_once() -> None:
    """Corroboration counts anchors, never discovery records."""
    anchor = _anchor(
        "node-1",
        arms=(
            (DiscoveryMethod.DENSE, 0, 0.9, SignalScale.COSINE),
            (DiscoveryMethod.SPARSE, 1, 7.1, SignalScale.LEXICAL),
        ),
    )

    assembled = _assembler().assemble([anchor], {}, {SOURCE_A: 0.9})

    assert assembled.anchor_count == 1
    assert sum(1 for item in assembled.evidence if item.is_anchor) == 1


# ===========================================================================
# CONTEXT
# ===========================================================================


def _with_context() -> Any:
    anchor = _anchor("node-1", 0)
    context = {
        "node-1": [
            _node("ctx-before", ordinal=1, text="Before"),
            _node("ctx-after", ordinal=2, text="After"),
        ]
    }
    return _assembler().assemble([anchor], context, {SOURCE_A: 0.9})


def test_context_is_never_an_anchor() -> None:
    """
    Corroboration counts distinct anchors, so an accompanying passage admitted
    as an anchor would make one finding appear as two.
    """
    assembled = _with_context()

    companions = [item for item in assembled.evidence if item.node_id != "node-1"]
    assert companions
    assert all(item.is_anchor is False for item in companions)


def test_context_anchors_to_the_node_it_accompanies() -> None:
    assembled = _with_context()

    companions = [item for item in assembled.evidence if item.node_id != "node-1"]
    assert all(item.anchor_node_id == "node-1" for item in companions)


def test_context_records_the_expansion_and_its_origin() -> None:
    assembled = _with_context()

    companion = assembled.evidence[1]
    assert len(companion.discovery) == 1
    assert companion.discovery[0].method is DiscoveryMethod.EXPANSION
    assert companion.discovery[0].via_node_id == "node-1"


def test_context_discovery_carries_no_score() -> None:
    """It was not ranked, so there is nothing to report."""
    companion = _with_context().evidence[1]

    assert companion.discovery[0].raw_score is None


def test_context_follows_its_own_anchor() -> None:
    """
    A consumer reading top-down meets a finding and then the passages that
    surround it, which is the order in which the evidence makes sense.
    """
    anchor_one = _anchor("node-1", 0)
    anchor_two = _anchor("node-2", 1, document_id="doc-2", ordinal=9)
    context = {"node-1": [_node("ctx-a", ordinal=1)]}

    assembled = _assembler().assemble(
        [anchor_one, anchor_two], context, {SOURCE_A: 0.9}
    )

    assert [item.node_id for item in assembled.evidence] == [
        "node-1",
        "ctx-a",
        "node-2",
    ]


def test_anchors_and_context_are_counted_separately() -> None:
    assembled = _with_context()

    assert assembled.anchor_count == 1
    assert assembled.context_count == 2
    assert len(assembled) == 3


def test_the_corroboration_count_reflects_findings_not_items() -> None:
    """
    The whole point. Three items, one finding. A consumer counting anchors gets
    one; a consumer counting items would get three, which is why the count is
    reported rather than left to be recomputed.
    """
    assembled = _with_context()

    assert len(assembled.evidence) == 3
    assert sum(1 for item in assembled.evidence if item.is_anchor) == 1
    assert assembled.anchor_count == 1


# ===========================================================================
# CREDIBILITY PROPAGATION
# ===========================================================================


def test_a_known_credibility_reaches_the_evidence_item() -> None:
    item = _assembler().assemble([_anchor("node-1")], {}, {SOURCE_A: 0.75}).evidence[0]

    assert item.credibility_score == 0.75


def test_an_unknown_credibility_reaches_the_evidence_item_as_none() -> None:
    item = _assembler().assemble([_anchor("node-1")], {}, {SOURCE_A: None}).evidence[0]

    assert item.credibility_score is None


def test_credibility_is_resolved_per_source_not_per_item() -> None:
    """Two items from one source carry one source's credibility."""
    anchors = [_anchor("node-1", 0), _anchor("node-2", 1, source_uri=SOURCE_B)]

    assembled = _assembler().assemble(
        anchors, {}, {SOURCE_A: 0.9, SOURCE_B: None}
    )

    assert assembled.evidence[0].credibility_score == 0.9
    assert assembled.evidence[1].credibility_score is None


def test_unavailable_sources_are_reported() -> None:
    """
    The absence changes what a consumer can compute, and nothing else in the
    result would say so.
    """
    anchors = [_anchor("node-1", 0), _anchor("node-2", 1, source_uri=SOURCE_B)]

    assembled = _assembler().assemble(
        anchors, {}, {SOURCE_A: 0.9, SOURCE_B: None}
    )

    assert assembled.credibility_unavailable == (SOURCE_B,)


def test_retrieval_reliability_computes_when_credibility_is_complete() -> None:
    """
    Math spec section 3: r(E) = (1/k) * sum_i r_i * s_i
    """
    anchors = [_anchor("node-1", 0), _anchor("node-2", 1)]

    assembled = _assembler().assemble(anchors, {}, {SOURCE_A: 0.8})

    reliability = sum(
        item.retrieval_score * item.credibility_score for item in assembled.evidence
    ) / len(assembled.evidence)

    assert reliability == pytest.approx((1.0 * 0.8 + 0.5 * 0.8) / 2)


def test_reliability_arithmetic_fails_loudly_when_credibility_is_unknown() -> None:
    """
    The reason absence is represented rather than filled in. A consumer cannot
    multiply past an unknown by accident, so the decision about what it means
    has to be made deliberately by the layer that owns the guarantee.
    """
    assembled = _assembler().assemble([_anchor("node-1")], {}, {SOURCE_A: None})

    with pytest.raises(TypeError):
        sum(
            item.retrieval_score * item.credibility_score
            for item in assembled.evidence
        )


# ===========================================================================
# THE ORDINAL RELEVANCE SIGNAL
# ===========================================================================


def test_relevance_is_the_reciprocal_of_fused_position() -> None:
    """
    Position is the only property of a fused result that survives the arms'
    scales being mutually incomparable, so a number derived from it cannot be
    mistaken for a measurement.
    """
    anchors = [_anchor(f"node-{rank}", rank) for rank in range(4)]

    assembled = _assembler().assemble(anchors, {}, {SOURCE_A: 0.9})

    assert [item.retrieval_score for item in assembled.evidence] == [
        1.0,
        0.5,
        pytest.approx(1 / 3),
        0.25,
    ]


def test_relevance_ignores_the_fused_value() -> None:
    """
    Two results at the same position score the same however different their
    fused values, because the fused value lives on an incomparable scale.
    """
    low = HydratedResult(
        candidate=FusedCandidate(
            node_id="node-1",
            signals=(
                RetrievalSignal(
                    method=DiscoveryMethod.DENSE,
                    rank=0,
                    raw_score=0.1,
                    scale=SignalScale.COSINE,
                ),
            ),
            fused_rank=0,
            fused_score=0.001,
        ),
        node=_node("node-1"),
    )
    high = _anchor("node-2", 0)

    assert (
        _assembler().assemble([low], {}, {SOURCE_A: 0.9}).evidence[0].retrieval_score
        == _assembler()
        .assemble([high], {}, {SOURCE_A: 0.9})
        .evidence[0]
        .retrieval_score
    )


def test_relevance_stays_within_the_unit_interval() -> None:
    anchors = [_anchor(f"node-{rank}", rank) for rank in range(50)]

    assembled = _assembler().assemble(anchors, {}, {SOURCE_A: 0.9})

    assert all(0.0 < item.retrieval_score <= 1.0 for item in assembled.evidence)


def test_context_inherits_its_anchors_position() -> None:
    """It was not ranked, so inventing a position would imply a measurement
    nobody made."""
    assembled = _with_context()

    anchor_score = assembled.evidence[0].retrieval_score
    assert all(
        item.retrieval_score == anchor_score for item in assembled.evidence
    )


# ===========================================================================
# DOCUMENTS
# ===========================================================================


def test_documents_mirror_evidence_one_to_one() -> None:
    """
    Built in one pass so the two views cannot disagree about what was
    retrieved.
    """
    assembled = _with_context()

    assert [document.node_id for document in assembled.documents] == [
        item.node_id for item in assembled.evidence
    ]


def test_a_multi_arm_anchor_is_attributed_to_hybrid_retrieval() -> None:
    """Naming one of several arms would misattribute the finding."""
    anchor = _anchor(
        "node-1",
        arms=(
            (DiscoveryMethod.DENSE, 0, 0.9, SignalScale.COSINE),
            (DiscoveryMethod.SPARSE, 1, 7.1, SignalScale.LEXICAL),
        ),
    )

    assembled = _assembler().assemble([anchor], {}, {SOURCE_A: 0.9})

    assert assembled.documents[0].retrieval_method is RetrievalMethod.HYBRID


@pytest.mark.parametrize(
    ("method", "expected"),
    [
        (DiscoveryMethod.DENSE, RetrievalMethod.DENSE),
        (DiscoveryMethod.SPARSE, RetrievalMethod.SPARSE),
        (DiscoveryMethod.GRAPH, RetrievalMethod.GRAPH),
    ],
)
def test_a_single_arm_anchor_is_attributed_to_that_arm(
    method: DiscoveryMethod, expected: RetrievalMethod
) -> None:
    anchor = _anchor("node-1", arms=((method, 0, None, None),))

    assembled = _assembler().assemble([anchor], {}, {SOURCE_A: 0.9})

    assert assembled.documents[0].retrieval_method is expected


def test_an_anchor_document_carries_its_signals() -> None:
    """The complete per-arm truth, including each score and its scale."""
    assembled = _with_context()

    assert len(assembled.documents[0].signals) == 1


def test_a_context_document_carries_no_signals() -> None:
    """No arm observed it."""
    assembled = _with_context()

    assert assembled.documents[1].signals == ()


def test_a_document_titles_itself_from_its_heading_path() -> None:
    anchor = _anchor("node-1", heading_path=("Chapter 2", "Rates"))

    assembled = _assembler().assemble([anchor], {}, {SOURCE_A: 0.9})

    assert assembled.documents[0].title == "Rates"


# ===========================================================================
# INTEGRITY
# ===========================================================================


def test_context_naming_an_absent_anchor_is_refused() -> None:
    """
    Those passages would be discarded without anything reporting their loss.
    """
    with pytest.raises(EvidenceAssemblyError, match="not in the result set"):
        _assembler().assemble(
            [_anchor("node-1")], {"ghost": [_node("ctx-a")]}, {SOURCE_A: 0.9}
        )


def test_an_anchor_supplied_as_context_is_refused() -> None:
    """It was found independently; demoting it would lose a finding."""
    anchors = [_anchor("node-1", 0), _anchor("node-2", 1)]

    with pytest.raises((AnchorViolationError, CorroborationIntegrityError)):
        _assembler().assemble(anchors, {"node-1": [_node("node-2")]}, {SOURCE_A: 0.9})


def test_a_repeated_anchor_is_refused() -> None:
    """One passage would be counted twice."""
    with pytest.raises(CorroborationIntegrityError, match="more than once"):
        _assembler().assemble(
            [_anchor("node-1", 0), _anchor("node-1", 1)], {}, {SOURCE_A: 0.9}
        )


def test_repeated_context_is_refused() -> None:
    with pytest.raises(CorroborationIntegrityError):
        _assembler().assemble(
            [_anchor("node-1")],
            {"node-1": [_node("ctx-a"), _node("ctx-a")]},
            {SOURCE_A: 0.9},
        )


def test_assembling_nothing_yields_nothing() -> None:
    assembled = _assembler().assemble([], {}, {})

    assert len(assembled) == 0
    assert assembled.anchor_count == 0
    assert assembled.context_count == 0


# ===========================================================================
# EVIDENCE-FIRST CHAIN
# ===========================================================================


def test_evidence_content_is_the_node_text() -> None:
    """Nothing re-derives it, so the chain back to the document holds."""
    node = _node("node-1", text="The applicable rate is four percent.")
    anchor = HydratedResult(
        candidate=FusedCandidate(
            node_id="node-1",
            signals=(
                RetrievalSignal(
                    method=DiscoveryMethod.DENSE,
                    rank=0,
                    raw_score=0.9,
                    scale=SignalScale.COSINE,
                ),
            ),
            fused_rank=0,
            fused_score=1.0,
        ),
        node=node,
    )

    item = _assembler().assemble([anchor], {}, {SOURCE_A: 0.9}).evidence[0]

    assert item.content == node.text


def test_the_span_and_blocks_survive_assembly() -> None:
    assembled = _with_context()
    item = assembled.evidence[0]

    assert item.span is not None
    assert item.block_ids
    assert item.document_id == "doc-1"
    assert item.source_uri == SOURCE_A


def test_an_evidence_identifier_is_deterministic() -> None:
    """Repeating a query against an unchanged corpus produces comparable
    identifiers."""
    first = _assembler().assemble([_anchor("node-1")], {}, {SOURCE_A: 0.9})
    second = _assembler().assemble([_anchor("node-1")], {}, {SOURCE_A: 0.9})

    assert first.evidence[0].evidence_id == second.evidence[0].evidence_id


def test_an_evidence_identifier_records_the_role() -> None:
    """
    The same passage may appear as context for a different anchor in a different
    query, and the pair is what identifies its role here.
    """
    assembled = _with_context()

    anchor_id = assembled.evidence[0].evidence_id
    context_id = assembled.evidence[1].evidence_id

    assert anchor_id != context_id
    assert "node-1" in context_id


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


def test_assembly_is_synchronous() -> None:
    """Every input has already been resolved, so assembly is a pure
    transformation."""
    assert "async def assemble" not in inspect.getsource(assembly_module)


def test_the_module_touches_no_store() -> None:
    source = inspect.getsource(assembly_module)

    for forbidden in ("NodeStore", "VectorIndex", "LexicalIndex", "RelationStore"):
        assert forbidden not in source


def test_the_module_imports_no_backend_or_plugin() -> None:
    source = inspect.getsource(assembly_module)

    for forbidden in (
        "import psycopg",
        "import qdrant",
        "import numpy",
        "from ptfm.plugins",
    ):
        assert forbidden not in source


def test_the_module_names_no_storage_technology() -> None:
    source = inspect.getsource(assembly_module).lower()

    for term in ("postgres", "pgvector", "qdrant", "faiss", "elasticsearch", "neo4j"):
        assert term not in source


def test_the_module_contains_no_domain_vocabulary() -> None:
    source = inspect.getsource(assembly_module).lower()

    for term in ("jurisdiction", "statute", "ticker", " icd", "patient", "court"):
        assert term not in source


def test_the_module_declares_no_calibration_concept() -> None:
    """Retrieval produces uncalibrated signals; calibration belongs later."""
    source = _code_without_docstrings(assembly_module).lower()

    for term in ("calibrat", "entail", "truth space", "verification result"):
        assert term not in source


def test_the_module_declares_no_mutable_state() -> None:
    mutable = [
        name
        for name, value in vars(assembly_module).items()
        if not name.startswith("_") and isinstance(value, (list, dict, set))
    ]

    assert mutable == []


def test_the_module_never_catches_base_exception() -> None:
    assert "except BaseException" not in inspect.getsource(assembly_module)


def test_the_stages_hold_no_assignable_state() -> None:
    with pytest.raises(AttributeError):
        _assembler().cache = {}  # type: ignore[attr-defined]

    with pytest.raises(AttributeError):
        _annotator().cache = {}  # type: ignore[attr-defined]
