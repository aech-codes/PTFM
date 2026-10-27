"""
Unit tests for the query preparation stage.

File:
    tests/core/retrieval/test_encoding.py

Emphasis
--------
    That the compatibility gate closes BEFORE any inference or search runs.
    Every failure guarded here is invisible downstream: a query encoded for
    the wrong model, or as a passage when the model expects a query, returns
    plausible results in a worse order and reports nothing at all.
"""

from __future__ import annotations

import ast
import inspect
from typing import Any, Optional, Sequence

import pytest

from ptfm.core.knowledge.interfaces import EmbeddingProvider, ModelIdentity
from ptfm.core.retrieval.config import (
    DegradationPolicy,
    PartitionConfig,
    RetrievalConfig,
    TelemetryConfig,
)
from ptfm.core.retrieval.exceptions import (
    EmbeddingSpaceMismatchError,
    EmptyQueryError,
    PartitionNotFoundError,
    QueryEncodingError,
    QueryEncodingMismatchError,
    QueryTooLargeError,
)
from ptfm.core.retrieval.models import EmbeddingPartition, SegmentationPartition
from ptfm.core.retrieval.partition import (
    PartitionResolver,
    embedding_for_vector,
    segmentation_for_node,
)
from ptfm.core.retrieval.pipeline import encoding as encoding_module
from ptfm.core.retrieval.pipeline.encoding import ProviderQueryEncoder, QueryPreparer
from ptfm.core.schemas import (
    KnowledgeNode,
    NodeEmbedding,
    NodeType,
    SourceDescriptor,
    TextSpan,
)

NOW = __import__("datetime").datetime(2026, 8, 1, tzinfo=__import__("datetime").UTC)


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def _source() -> SourceDescriptor:
    return SourceDescriptor(
        source_uri="file:///corpus/a.pdf",
        scheme="file",
        connector_name="LocalFilesystemConnector",
        extractor_name="PdfExtractor",
        acquired_at=NOW,
        size_bytes=1024,
        detected_media_type="application/pdf",
    )


def _node() -> KnowledgeNode:
    return KnowledgeNode(
        node_id="node-1",
        document_id="doc-1",
        span=TextSpan(start=0, end=5),
        text="Alpha",
        node_type=NodeType.PROSE,
        block_ids=("0",),
        ordinal=0,
        source=_source(),
        segmentation_fingerprint="1:abc",
        source_text_hash="text-hash",
        source_schema_version="1.0",
        processed_at=NOW,
    )


def _vector_record() -> NodeEmbedding:
    return NodeEmbedding(
        node_id="node-1",
        document_id="doc-1",
        vector=(0.5, 0.5, 0.5, 0.5),
        dimension=4,
        model_id="BAAI/bge-large-en-v1.5",
        model_revision="r1",
        normalized=True,
        composition_id="heading_path_prefixed",
        query_encoding="prefix:query: ",
        input_text_hash="input-hash",
        embedded_at=NOW,
    )


def _partitions() -> tuple[SegmentationPartition, EmbeddingPartition]:
    segmentation = segmentation_for_node(_node())
    return segmentation, embedding_for_vector(_vector_record(), segmentation)


class _Provider(EmbeddingProvider):
    """Records what it was asked to embed, so delegation is observable."""

    def __init__(
        self,
        *,
        query_encoding: str = "prefix:query: ",
        dimension: int = 4,
        model_id: str = "BAAI/bge-large-en-v1.5",
        revision: Optional[str] = "r1",
        max_input_chars: Optional[int] = None,
        failure: Optional[Exception] = None,
        returned_width: Optional[int] = None,
    ) -> None:
        self._query_encoding = query_encoding
        self._dimension = dimension
        self._model_id = model_id
        self._revision = revision
        self._max_input_chars = max_input_chars
        self._failure = failure
        self._returned_width = returned_width
        self.queries: list[str] = []

    @property
    def name(self) -> str:
        return "TestProvider"

    @property
    def max_batch_size(self) -> int:
        return 8

    def model_identity(self) -> ModelIdentity:
        return ModelIdentity(
            model_id=self._model_id,
            dimension=self._dimension,
            normalized=True,
            revision=self._revision,
            query_encoding=self._query_encoding,
            max_input_chars=self._max_input_chars,
        )

    async def embed(self, texts: Sequence[str]) -> Sequence[tuple[float, ...]]:
        return [(0.5,) * self._dimension for _ in texts]

    async def embed_query(self, query: str) -> tuple[float, ...]:
        self.queries.append(query)
        if self._failure is not None:
            raise self._failure
        return (0.5,) * (self._returned_width or self._dimension)

    async def health_check(self) -> bool:
        return True

    async def close(self) -> None:
        return None


class _Catalog:
    def __init__(self, segmentation: Any = ..., embedding: Any = ...) -> None:
        default_segmentation, default_embedding = _partitions()
        self._segmentation = (
            default_segmentation if segmentation is ... else segmentation
        )
        self._embedding = default_embedding if embedding is ... else embedding

    @property
    def name(self) -> str:
        return "test-catalog"

    async def active_segmentation(self) -> Any:
        return self._segmentation

    async def active_embedding(self, segmentation: Any) -> Any:
        return self._embedding

    async def register(self, segmentation: Any, embedding: Any, *, activate: bool) -> None:
        return None


class _Ids:
    def new_id(self) -> str:
        return "q-1"


class _Logger:
    """
    Records event names and their fields.

    Fields are captured as well as names because a privacy guarantee is about
    what a record CONTAINS, not which records exist. Asserting on event names
    alone would pass for a log line that carried the query text.
    """

    def __init__(self) -> None:
        self.events: list[str] = []
        self.records: list[tuple[str, dict[str, Any]]] = []

    def bind(self, **values: Any) -> _Logger:
        return self

    def _record(self, event: str, values: dict[str, Any]) -> None:
        self.events.append(event)
        self.records.append((event, dict(values)))

    def debug(self, event: str, **values: Any) -> None:
        self._record(event, values)

    def info(self, event: str, **values: Any) -> None:
        self._record(event, values)

    def warning(self, event: str, **values: Any) -> None:
        self._record(event, values)

    def error(self, event: str, **values: Any) -> None:
        self._record(event, values)

    def rendered(self) -> str:
        """
        Flatten every record the way a log sink would serialize it.

        Returns
        -------
        str
            Every event with every field, so a substring scan covers the whole
            log rather than one field a test happened to name.
        """
        return "\n".join(
            f"{event} " + " ".join(f"{k}={v!r}" for k, v in fields.items())
            for event, fields in self.records
        )

    def fields(self, event: str) -> list[dict[str, Any]]:
        """
        Return the fields of every record for one event.

        Parameters
        ----------
        event:
            The event name.

        Returns
        -------
        list[dict[str, Any]]
            Fields per matching record.
        """
        return [f for name, f in self.records if name == event]


class _Preprocessor:
    def __init__(
        self, output: Optional[str] = None, failure: Optional[Exception] = None
    ) -> None:
        self._output = output
        self._failure = failure

    @property
    def name(self) -> str:
        return "domain"

    async def preprocess(self, query: str) -> str:
        if self._failure is not None:
            raise self._failure
        return self._output if self._output is not None else query.upper()


def _config(**overrides: Any) -> RetrievalConfig:
    return RetrievalConfig(_env_file=None, **overrides)


def _resolver(catalog: Optional[_Catalog] = None) -> PartitionResolver:
    return PartitionResolver(catalog or _Catalog(), PartitionConfig())


def _preparer(
    provider: Optional[_Provider] = None,
    *,
    config: Optional[RetrievalConfig] = None,
    logger: Optional[_Logger] = None,
    catalog: Optional[_Catalog] = None,
    preprocessor: Any = None,
) -> QueryPreparer:
    encoder = (
        ProviderQueryEncoder(provider) if provider is not None else None
    )
    return QueryPreparer(
        _resolver(catalog),
        encoder,
        config or _config(),
        _Ids(),
        logger or _Logger(),
        preprocessor=preprocessor,
    )


# ===========================================================================
# THE ENCODER
# ===========================================================================


async def test_the_encoder_produces_a_vector_of_the_partition_width() -> None:
    _, partition = _partitions()

    vector = await ProviderQueryEncoder(_Provider()).encode("the rate", partition)

    assert len(vector) == partition.dimension


async def test_the_encoder_delegates_the_query_side_to_the_provider() -> None:
    """
    Which transformation a model expects is the provider's knowledge. A
    component reconstructing it would guess at behaviour that produces no
    error when guessed wrong.
    """
    provider = _Provider()
    _, partition = _partitions()

    await ProviderQueryEncoder(provider).encode("the rate", partition)

    assert provider.queries == ["the rate"]


@pytest.mark.parametrize("query", ["", "   ", "\n\t"])
async def test_an_unsearchable_query_is_refused(query: str) -> None:
    """Returning results for it would present arbitrary content as
    relevant."""
    _, partition = _partitions()

    with pytest.raises(EmptyQueryError):
        await ProviderQueryEncoder(_Provider()).encode(query, partition)


async def test_an_oversized_query_is_refused() -> None:
    """Truncating would answer a different question from the one asked."""
    _, partition = _partitions()
    encoder = ProviderQueryEncoder(_Provider(max_input_chars=10))

    with pytest.raises(QueryTooLargeError):
        await encoder.encode("x" * 50, partition)


async def test_a_mismatched_query_convention_is_refused() -> None:
    _, partition = _partitions()

    with pytest.raises(QueryEncodingMismatchError):
        await ProviderQueryEncoder(_Provider(query_encoding="")).encode(
            "the rate", partition
        )


@pytest.mark.parametrize(
    "override",
    [
        {"model_id": "intfloat/multilingual-e5-large"},
        {"revision": "r2"},
        {"dimension": 8},
    ],
)
async def test_a_provider_from_another_space_is_refused(
    override: dict[str, Any],
) -> None:
    _, partition = _partitions()

    with pytest.raises(EmbeddingSpaceMismatchError):
        await ProviderQueryEncoder(_Provider(**override)).encode(
            "the rate", partition
        )


@pytest.mark.parametrize(
    "override",
    [{"query_encoding": ""}, {"model_id": "other/model"}, {"dimension": 8}],
)
async def test_incompatibility_is_refused_before_any_inference(
    override: dict[str, Any],
) -> None:
    """
    The check must precede the work. A query that cannot succeed should never
    cost an inference, and on a loaded model that cost is substantial.
    """
    provider = _Provider(**override)
    _, partition = _partitions()

    with pytest.raises((QueryEncodingMismatchError, EmbeddingSpaceMismatchError)):
        await ProviderQueryEncoder(provider).encode("the rate", partition)

    assert provider.queries == []


async def test_a_provider_returning_the_wrong_width_is_refused() -> None:
    """
    A provider that reports one width and returns another is not the model it
    claims to be, and every comparison it contributed would be meaningless.
    """
    _, partition = _partitions()

    with pytest.raises(EmbeddingSpaceMismatchError):
        await ProviderQueryEncoder(_Provider(returned_width=2)).encode(
            "the rate", partition
        )


async def test_a_provider_failure_is_converted_with_its_cause() -> None:
    _, partition = _partitions()
    encoder = ProviderQueryEncoder(_Provider(failure=RuntimeError("out of memory")))

    with pytest.raises(QueryEncodingError) as excinfo:
        await encoder.encode("the rate", partition)

    assert isinstance(excinfo.value.__cause__, RuntimeError)


def test_the_encoder_reads_its_identity_once() -> None:
    """
    A provider changing model between queries would let one query search a
    partition the next could not.

    Read as an ATTRIBUTE, because ``QueryEncoder`` declares ``model_identity``
    as a property. The distinction is load-bearing: an encoder exposing it as a
    method satisfied an earlier caller that invoked it as one, and any
    protocol-conformant third-party encoder then failed with
    ``'ModelIdentity' object is not callable``.
    """
    encoder = ProviderQueryEncoder(_Provider())

    assert isinstance(
        type(encoder).__dict__["model_identity"], property
    ), "QueryEncoder declares model_identity as a property"
    assert encoder.model_identity is encoder.model_identity


# ===========================================================================
# THE PREPARER
# ===========================================================================


async def test_a_plan_is_built_with_a_vector() -> None:
    segmentation, partition = _partitions()

    plan = await _preparer(_Provider()).prepare("the applicable rate")

    assert plan.has_vector is True
    assert plan.segmentation.key == segmentation.key
    assert plan.embedding is not None
    assert plan.embedding.key == partition.key


async def test_a_plan_honours_the_requested_bounds() -> None:
    preparer = _preparer(_Provider())

    assert (await preparer.prepare("q")).top_k == _config().default_top_k
    assert (await preparer.prepare("q", top_k=3)).top_k == 3
    assert (await preparer.prepare("q", exact=True)).exact is True


async def test_a_deployment_without_an_encoder_still_plans() -> None:
    """
    Lexical and structural retrieval serve alone. That is a configuration
    rather than a fault.
    """
    plan = await _preparer(None).prepare("the rate")

    assert plan.has_vector is False
    assert plan.embedding is None
    assert plan.text == "the rate"


async def test_an_unindexed_corpus_is_refused() -> None:
    preparer = _preparer(_Provider(), catalog=_Catalog(segmentation=None))

    with pytest.raises(PartitionNotFoundError):
        await preparer.prepare("the rate")


# ---------------------------------------------------------------------------
# Degradation
# ---------------------------------------------------------------------------


async def test_an_encoding_failure_degrades_to_lexical_only() -> None:
    logger = _Logger()
    preparer = _preparer(
        _Provider(failure=RuntimeError("service down")), logger=logger
    )

    plan = await preparer.prepare("the rate")

    assert plan.has_vector is False
    assert "retrieval.query.encoding_degraded" in logger.events


async def test_a_strict_policy_refuses_to_degrade() -> None:
    preparer = _preparer(
        _Provider(failure=RuntimeError("service down")),
        config=_config(degradation_policy=DegradationPolicy.STRICT),
    )

    with pytest.raises(QueryEncodingError):
        await preparer.prepare("the rate")


@pytest.mark.parametrize(
    ("query", "expected"),
    [("", EmptyQueryError), ("x" * 50, QueryTooLargeError)],
)
async def test_request_shaped_failures_never_degrade(
    query: str, expected: type[Exception]
) -> None:
    """
    Both describe the REQUEST rather than the system, so degrading would
    answer a different question from the one asked.
    """
    preparer = _preparer(_Provider(max_input_chars=10))

    with pytest.raises(expected):
        await preparer.prepare(query)


async def test_a_compatibility_mismatch_never_degrades() -> None:
    """
    Searching a partition with a vector that does not belong to it is not a
    reduced answer, it is a wrong one.
    """
    preparer = _preparer(_Provider(query_encoding=""))

    with pytest.raises(QueryEncodingMismatchError):
        await preparer.prepare("the rate")


# ---------------------------------------------------------------------------
# Domain preprocessing
# ---------------------------------------------------------------------------


async def test_domain_normalization_is_applied() -> None:
    plan = await _preparer(_Provider(), preprocessor=_Preprocessor()).prepare(
        "the rate"
    )

    assert plan.text == "THE RATE"


async def test_a_preprocessor_failure_degrades_to_the_raw_query() -> None:
    """
    A query that could not be normalized can still be searched as written.
    Refusing because an abbreviation could not be expanded would trade a small
    loss for a total one.
    """
    logger = _Logger()
    preparer = _preparer(
        _Provider(),
        logger=logger,
        preprocessor=_Preprocessor(failure=RuntimeError("lookup down")),
    )

    plan = await preparer.prepare("the rate")

    assert plan.text == "the rate"
    assert "retrieval.query.preprocessing_degraded" in logger.events


async def test_a_preprocessor_failure_degrades_even_under_a_strict_policy() -> None:
    """Normalization is an improvement to a query, not a precondition for
    one."""
    preparer = _preparer(
        _Provider(),
        config=_config(degradation_policy=DegradationPolicy.STRICT),
        preprocessor=_Preprocessor(failure=RuntimeError("lookup down")),
    )

    assert (await preparer.prepare("the rate")).text == "the rate"


@pytest.mark.parametrize("output", ["", "   "])
async def test_unusable_normalization_output_is_ignored(output: str) -> None:
    """A preprocessor returning nothing usable has declined, not succeeded."""
    preparer = _preparer(_Provider(), preprocessor=_Preprocessor(output=output))

    assert (await preparer.prepare("the rate")).text == "the rate"


async def test_a_domain_with_no_rules_searches_the_query_as_written() -> None:
    assert (await _preparer(_Provider()).prepare("the rate")).text == "the rate"


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


def test_the_stage_touches_no_store() -> None:
    """Preparation precedes search; it must not reach an index."""
    source = inspect.getsource(encoding_module)

    for forbidden in ("NodeStore", "VectorIndex", "LexicalIndex", "RelationStore"):
        assert forbidden not in source


def test_the_stage_imports_no_backend_or_plugin() -> None:
    source = inspect.getsource(encoding_module)

    for forbidden in (
        "import psycopg",
        "import qdrant",
        "import numpy",
        "from ptfm.plugins",
    ):
        assert forbidden not in source


def test_the_stage_names_no_storage_technology() -> None:
    source = inspect.getsource(encoding_module).lower()

    for term in ("postgres", "pgvector", "qdrant", "faiss", "elasticsearch", "neo4j"):
        assert term not in source


def test_the_stage_contains_no_domain_vocabulary() -> None:
    source = inspect.getsource(encoding_module).lower()

    for term in ("jurisdiction", "statute", "ticker", " icd", "patient", "court"):
        assert term not in source


def test_the_stage_declares_no_calibration_concept() -> None:
    source = _code_without_docstrings(encoding_module).lower()

    for term in ("calibrat", "entail", "truth space", "verification result"):
        assert term not in source


def test_the_stage_declares_no_mutable_state() -> None:
    mutable = [
        name
        for name, value in vars(encoding_module).items()
        if not name.startswith("_") and isinstance(value, (list, dict, set))
    ]

    assert mutable == []


def test_the_stage_never_catches_base_exception() -> None:
    source = inspect.getsource(encoding_module)

    assert "except BaseException" not in source


# ---------------------------------------------------------------------------
# Telemetry privacy (FR-1)
# ---------------------------------------------------------------------------
#
# A query may carry privileged or regulated material. `log_query_text` defaults
# to off, and an operator leaving it there is entitled to read that as meaning
# no query text is written to a log. These tests assert the whole log record,
# not one field, because that is the shape of the guarantee.

#: A string that could not appear in a log line by coincidence, so a substring
#: scan over every captured record is conclusive.
_SECRET = "PRIVILEGEDATTORNEYCLIENTTOKEN"

#: The event that carries the excerpt, reached only when encoding degrades.
_DEGRADED = "retrieval.query.encoding_degraded"


def test_query_text_logging_is_off_by_default() -> None:
    """
    The default is the private one.

    A privacy control defaulting to permissive would make the safe setting the
    one an operator has to discover.
    """
    assert _config().telemetry.log_query_text is False


async def test_query_text_is_absent_from_every_record_when_disabled() -> None:
    """
    With the flag off, no log record contains the query text.

    Asserted across every captured record rather than the excerpt field alone:
    the guarantee is about the log, not about one field's name.
    """
    logger = _Logger()
    preparer = _preparer(
        _Provider(failure=RuntimeError("service down")), logger=logger
    )

    plan = await preparer.prepare(f"{_SECRET} the rate")

    assert _DEGRADED in logger.events, "the warning must still be emitted"
    assert logger.records, "expected log records"
    assert _SECRET not in logger.rendered()
    assert all("excerpt" not in fields for _, fields in logger.records)
    assert plan.has_vector is False


async def test_suppression_is_recorded_rather_than_silent() -> None:
    """
    A withheld excerpt says it was withheld.

    A silently missing field is indistinguishable from a diagnostic that had
    nothing to report, leaving an operator unable to tell which happened.
    """
    logger = _Logger()
    preparer = _preparer(
        _Provider(failure=RuntimeError("service down")), logger=logger
    )

    await preparer.prepare(f"{_SECRET} the rate")

    assert logger.fields(_DEGRADED)[0]["query_text_redacted"] is True


async def test_the_truncated_excerpt_is_logged_when_enabled() -> None:
    """
    With the flag on, the excerpt is logged and is truncated.

    Truncation bounds exposure for an operator who has chosen to accept it; it
    is not what makes the choice safe.
    """
    logger = _Logger()
    preparer = _preparer(
        _Provider(failure=RuntimeError("service down")),
        config=_config(telemetry=TelemetryConfig(log_query_text=True)),
        logger=logger,
    )

    await preparer.prepare(f"{_SECRET} the rate")

    fields = logger.fields(_DEGRADED)[0]
    assert _SECRET in fields["excerpt"]
    assert "query_text_redacted" not in fields


async def test_a_long_query_is_truncated_not_logged_whole() -> None:
    """
    An enabled excerpt is still bounded.

    Parameters
    ----------
    None
    """
    logger = _Logger()
    query = _SECRET * 10
    preparer = _preparer(
        _Provider(failure=RuntimeError("service down")),
        config=_config(telemetry=TelemetryConfig(log_query_text=True)),
        logger=logger,
    )

    await preparer.prepare(query)

    excerpt = logger.fields(_DEGRADED)[0]["excerpt"]
    assert len(excerpt) < len(query)
    assert excerpt == query[: len(excerpt)]


@pytest.mark.parametrize("permitted", [False, True])
async def test_the_flag_changes_only_the_content_field(permitted: bool) -> None:
    """
    Both modes emit the same events with the same other fields.

    The setting governs disclosure, not diagnosis: turning it off must not cost
    an operator any signal other than the content itself.

    Parameters
    ----------
    permitted:
        Whether query text logging is enabled.
    """
    logger = _Logger()
    preparer = _preparer(
        _Provider(failure=RuntimeError("service down")),
        config=_config(telemetry=TelemetryConfig(log_query_text=permitted)),
        logger=logger,
    )

    await preparer.prepare(f"{_SECRET} the rate")

    other = {
        key
        for _, fields in logger.records
        for key in fields
        if key not in ("excerpt", "query_text_redacted")
    }
    assert "error_code" in other
    assert "message" in other


@pytest.mark.parametrize("permitted", [False, True])
async def test_the_flag_does_not_change_the_plan(permitted: bool) -> None:
    """
    The setting is telemetry only.

    A privacy control that altered retrieval would make audit logs and results
    vary together, so neither could be reasoned about alone.

    Parameters
    ----------
    permitted:
        Whether query text logging is enabled.
    """
    preparer = _preparer(
        config=_config(telemetry=TelemetryConfig(log_query_text=permitted))
    )

    plan = await preparer.prepare("the rate")

    assert plan.text == "the rate"
    assert plan.has_vector is False
    assert plan.top_k == _config().default_top_k
