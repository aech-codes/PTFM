"""
Tests for extractor provenance on the canonical document.

File:
    tests/core/ingestion/test_extractor_provenance.py

Scope
-----
    Covers ``SourceDescriptor.extractor_name`` and the path that populates it,
    from extraction through canonical assembly to an end-to-end run.

    Supplements the existing suites rather than replacing them. The field is
    optional and additive, so no previously delivered test requires
    modification.
"""

from __future__ import annotations

import inspect
from datetime import UTC, datetime
from typing import Any, Optional

import pytest

from ptfm.core.ingestion import (
    DocumentIngested,
    IngestionConfig,
    InMemoryConnector,
    MemoryDocument,
    SourceSpec,
    build_default_engine,
    default_extractors,
)
from ptfm.core.ingestion.models import (
    AcquiredBlob,
    DetectedFormat,
    SourceRecord,
)
from ptfm.core.ingestion.pipeline import normalization as normalization_module
from ptfm.core.ingestion.pipeline.metadata import HarvestedMetadata
from ptfm.core.ingestion.pipeline.normalization import RawDocumentNormalizer
from ptfm.core.ingestion.pipeline.structure import NormalizedStructure
from ptfm.core.ingestion.registry import ConnectorRegistration
from ptfm.core.ingestion.utils.concurrency import InlineOffloader
from ptfm.core.ingestion.utils.hashing import Blake2bHashProvider
from ptfm.core.schemas import (
    DocumentMetadata,
    RawDocument,
    SourceDescriptor,
)

NOW = datetime(2026, 7, 30, tzinfo=UTC)
TEXT = "Body paragraph."


# ---------------------------------------------------------------------------
# Doubles and builders
# ---------------------------------------------------------------------------


class _FixedClock:
    def now(self) -> datetime:
        return NOW


def _record() -> SourceRecord:
    return SourceRecord(
        record_id="r-1",
        source_uri="file:///corpus/a.txt",
        scheme="file",
        connector_name="LocalFilesystemConnector",
        discovered_at=NOW,
        filename="a.txt",
    )


def _blob() -> AcquiredBlob:
    return AcquiredBlob(
        record=_record(),
        size_bytes=16,
        content_hash="abc123",
        hash_algorithm="blake2b-256",
        acquired_at=NOW,
        data=b"x" * 16,
    )


def _assemble(extractor_name: Optional[str] = None) -> RawDocument:
    normalizer = RawDocumentNormalizer(Blake2bHashProvider(), _FixedClock())
    return normalizer.assemble(
        blob=_blob(),
        detected=DetectedFormat(media_type="text/plain", encoding="utf-8"),
        structure=NormalizedStructure(text=TEXT, blocks=()),
        metadata=HarvestedMetadata(
            metadata=DocumentMetadata(
                char_count=len(TEXT), word_count=len(TEXT.split())
            )
        ),
        extractor_name=extractor_name,
    )


def _spec() -> SourceSpec:
    return SourceSpec(root_uri="memory:///")


def _engine(documents: list[MemoryDocument]) -> Any:
    def _factory(config: IngestionConfig) -> Any:
        return InMemoryConnector(config, documents, clock=_FixedClock())

    return build_default_engine(
        IngestionConfig(_env_file=None),
        connectors=[ConnectorRegistration(scheme="memory", factory=_factory)],
        extractors=list(default_extractors()),
        clock=_FixedClock(),
        offloader=InlineOffloader(),
        discover_entry_points=False,
    )


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


def test_the_descriptor_declares_the_field() -> None:
    assert "extractor_name" in SourceDescriptor.model_fields


def test_the_field_is_optional() -> None:
    """Additive and non-breaking: previously serialized documents remain
    valid."""
    descriptor = SourceDescriptor(
        source_uri="file:///a.txt",
        scheme="file",
        connector_name="LocalFilesystemConnector",
        acquired_at=NOW,
        size_bytes=16,
        detected_media_type="text/plain",
    )

    assert descriptor.extractor_name is None


def test_the_descriptor_remains_frozen() -> None:
    from pydantic import ValidationError

    descriptor = SourceDescriptor(
        source_uri="file:///a.txt",
        scheme="file",
        connector_name="C",
        extractor_name="E",
        acquired_at=NOW,
        size_bytes=1,
        detected_media_type="text/plain",
    )

    with pytest.raises(ValidationError):
        descriptor.extractor_name = "Other"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------


def test_the_extractor_is_recorded_on_the_document() -> None:
    document = _assemble("PdfExtractor")

    assert document.source.extractor_name == "PdfExtractor"


def test_an_absent_extractor_is_recorded_as_none() -> None:
    document = _assemble(None)

    assert document.source.extractor_name is None


def test_the_provenance_chain_records_both_components() -> None:
    """The connector determines which bytes entered; the extractor determines
    what structure was derived from them."""
    document = _assemble("MarkdownExtractor")

    assert document.source.connector_name == "LocalFilesystemConnector"
    assert document.source.extractor_name == "MarkdownExtractor"


def test_the_field_survives_a_json_round_trip() -> None:
    document = _assemble("DocxExtractor")

    restored = RawDocument.model_validate_json(document.model_dump_json())

    assert restored.source.extractor_name == "DocxExtractor"
    assert restored == document


def test_the_extractor_does_not_affect_document_identity() -> None:
    """Identity derives from origin and payload alone, so re-ingestion under a
    different extractor must not create a second document."""
    first = _assemble("PdfExtractor")
    second = _assemble("PlainTextExtractor")

    assert first.document_id == second.document_id


def test_assembly_remains_deterministic() -> None:
    assert _assemble("PdfExtractor") == _assemble("PdfExtractor")


# ---------------------------------------------------------------------------
# End to end
# ---------------------------------------------------------------------------


async def test_the_engine_records_the_extractor_that_ran() -> None:
    engine = _engine([MemoryDocument("a.txt", b"Body text.\n\nSecond para.")])

    outcomes = [outcome async for outcome in engine.ingest(_spec())]

    assert len(outcomes) == 1
    assert isinstance(outcomes[0], DocumentIngested)
    assert outcomes[0].document.source.extractor_name == "PlainTextExtractor"


async def test_the_recorded_extractor_matches_the_result() -> None:
    documents = [
        MemoryDocument("a.txt", b"Plain body text."),
        MemoryDocument("b.csv", b"region,revenue\nNorth,120\n"),
    ]
    engine = _engine(documents)

    outcomes = [outcome async for outcome in engine.ingest(_spec())]
    by_uri = {
        outcome.document.source.source_uri: outcome.document
        for outcome in outcomes
    }

    assert by_uri["memory:///a.txt"].source.extractor_name == "PlainTextExtractor"
    assert by_uri["memory:///b.csv"].source.extractor_name == "TabularExtractor"


async def test_documents_are_identifiable_by_extractor() -> None:
    """Corpus remediation after an extractor defect becomes a field lookup
    rather than an inference from configuration."""
    documents = [
        MemoryDocument(f"doc{index}.csv", b"a,b\n1,2\n") for index in range(3)
    ] + [MemoryDocument("note.txt", b"Prose body.")]
    engine = _engine(documents)

    outcomes = [outcome async for outcome in engine.ingest(_spec())]
    affected = [
        outcome.document
        for outcome in outcomes
        if outcome.document.source.extractor_name == "TabularExtractor"
    ]

    assert len(affected) == 3


async def test_a_fallback_extractor_is_distinguishable_from_a_dedicated_one() -> None:
    """
    The case the field exists for. The same media type resolves to different
    extractors depending on deployment, producing materially different block
    structures that the media type alone cannot distinguish.
    """
    from ptfm.core.ingestion.extractors.plaintext import PlainTextExtractor

    def _factory(config: IngestionConfig) -> Any:
        return InMemoryConnector(
            config,
            [MemoryDocument("doc.md", b"# Heading\n\nBody text.")],
            clock=_FixedClock(),
        )

    fallback_only = build_default_engine(
        IngestionConfig(_env_file=None),
        connectors=[ConnectorRegistration(scheme="memory", factory=_factory)],
        extractors=[PlainTextExtractor(["text/*"], priority=-100)],
        clock=_FixedClock(),
        offloader=InlineOffloader(),
        discover_entry_points=False,
    )

    outcomes = [outcome async for outcome in fallback_only.ingest(_spec())]
    document = outcomes[0].document

    assert document.source.detected_media_type == "text/markdown"
    assert document.source.extractor_name == "PlainTextExtractor"


# ---------------------------------------------------------------------------
# Wiring
# ---------------------------------------------------------------------------


def test_assembly_accepts_the_extractor_identity() -> None:
    parameters = inspect.signature(RawDocumentNormalizer.assemble).parameters

    assert "extractor_name" in parameters
    assert parameters["extractor_name"].default is None


def test_the_identity_reaches_the_descriptor() -> None:
    source = inspect.getsource(normalization_module)

    assert "extractor_name=extractor_name" in source


def test_the_engine_forwards_the_extractor_identity() -> None:
    from ptfm.core.ingestion import engine as engine_module

    source = inspect.getsource(engine_module)

    assert "extraction.extractor_name" in source
