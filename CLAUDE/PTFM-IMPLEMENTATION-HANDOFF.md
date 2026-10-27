# PTFM — Authoritative Implementation Handoff

**Last verified:** 2026-09-25
**Status:** Modules 1, 2, 3 complete. Modules 2 and 3 frozen. Module 4 (Verification / Decision) not started.
**Source of truth:** the implemented code. Where this document and the original proposal disagree, this document wins.

---

## 0. How to use this document

This is the document to read before touching PTFM. It was assembled by extracting
structure from the code itself — module paths come from each file's own `File:`
docstring header, exports from each `__all__`, signatures from the AST — not from
recollection.

Companion documents in this project, all still current:

| Doc | What it holds |
|---|---|
| `claude/module-3-freeze-review.md` | Full Module 3 audit: layers, contract, checklist, Module 4 risks |
| `claude/retrieval-backend-parity.md` | What is guaranteed identical across retrieval backends, and what is not |
| `claude/contract-changes.md` | CC-1 (credibility absence), with the arithmetic that justified it |
| `claude/technical-debt.md` | TD-1 in full |
| `claude/inert-guarantees.md` | Register of 16 "reads as present but does nothing" defects, and what finds them |
| `PTFM MathSpec.md` | The equations. Module 4 implements §1, §2, §4–§11 |
| `CoreSchemas.py`, `PluginInterface.py`, `AntiPatterns.md` | Original contract references |

**One thing to internalise before writing code here.** The dominant defect class
in this codebase is not a wrong algorithm — it is a *guarantee expressed
somewhere that does not execute*: a config field nothing reads, a docstring
promising more than the type delivers, a check that cannot fail, a branch
predicated on something that cannot occur. Sixteen instances have been found,
**every one by execution, none by review**. The standing rule:

> Dead or ineffective protection is worse than either position, because it reads
> as present. Assert the **behaviour** a declaration is supposed to produce,
> never the presence of the declaration.

---

## 1. What PTFM is

A domain-agnostic trust layer between a Foundation Model and an end user. It
does not generate answers; it decides whether an answer is supportable, and
refuses when it is not.

The pipeline, fixed:

```
Evidence → Verification → Confidence Calibration → Decision
```

Four modules realise it:

| # | Module | Package | Status | Lines |
|---|---|---|---|---|
| 1 | Ingestion | `ptfm/core/ingestion/` | Complete | ~9,700 |
| 2 | Knowledge Processing | `ptfm/core/knowledge/` | **FROZEN** | ~12,500 |
| 3 | Retrieval | `ptfm/core/retrieval/` | **FROZEN** | 17,470 |
| 4 | Verification / Decision | `ptfm/core/verification/` (to create) | **NOT STARTED** | — |

Plus a shared contract layer, `ptfm/core/schemas/`, which every module depends
on and which depends on nothing.

**Target stack:** Python 3.11+, Pydantic v2, asyncio, FastAPI (not yet used),
PyTorch (Module 4 will be first to need it). PostgreSQL 16 + pgvector 0.6.0 +
psycopg 3.3 as the default retrieval backend, behind interfaces.

---

## 2. Repository structure

Verified against each file's own `File:` header.

```
ptfm/core/
├── schemas/                      # Cross-module contract. Depends on nothing.
│   ├── __init__.py               # 59 exports — the only import surface other modules use
│   ├── common.py                 # TextSpan, UserQuery, LLMRequest/Response, ExecutionTrace
│   ├── ingestion.py              # RawDocument, DocumentBlock, SourceDescriptor, BlockIndex
│   ├── knowledge.py              # KnowledgeNode, NodeEmbedding, NodeRelation
│   ├── retrieval.py              # RetrievedDocument, EvidenceItem, DiscoveryRecord, SignalScale
│   └── verification.py           # MODULE 4'S TARGET SHAPES — already defined, unused
│
├── ingestion/                    # MODULE 1
│   ├── __init__.py               # build_default_engine, 51 exports
│   ├── engine.py                 # IngestionEngine
│   ├── interfaces.py             # SourceConnector, ContentExtractor, Clock, HashProvider, …
│   ├── config.py                 # IngestionConfig + 8 groups
│   ├── telemetry.py              # StdlibStructuredLogger, NullMetricsSink, SafeLogger/SafeMetricsSink
│   ├── connectors/
│   │   ├── filesystem.py         # LocalFilesystemConnector
│   │   └── memory.py             # InMemoryConnector, InMemoryFingerprintStore
│   ├── extractors/
│   │   ├── pdf.py                # PdfExtractor
│   │   ├── docx.py               # DocxExtractor
│   │   ├── html.py               # HtmlExtractor
│   │   ├── markdown.py           # MarkdownExtractor
│   │   ├── plaintext.py          # PlainTextExtractor
│   │   └── tabular.py            # TabularExtractor
│   ├── pipeline/
│   │   └── normalization.py      # RawDocumentNormalizer
│   └── utils/
│       └── concurrency.py        # BoundedThreadOffloader, InlineOffloader
│
├── knowledge/                    # MODULE 2 — FROZEN
│   ├── __init__.py               # build_default_engine, 56 exports
│   ├── engine.py                 # KnowledgeEngine
│   ├── interfaces.py             # ChunkingStrategy, EmbeddingProvider, SentenceSegmenter, …
│   ├── models.py                 # KnowledgeUnit, ChunkPlan, PlannedChunk, outcomes
│   ├── config.py                 # KnowledgeConfig + 7 groups
│   ├── exceptions.py
│   ├── registry.py               # SegmentationRegistry, EmbeddingRegistry
│   ├── sinks.py                  # Null / InMemory / Composite KnowledgeSink
│   ├── telemetry.py
│   ├── chunking/
│   │   ├── structural.py         # StructuralChunker  (selector "block_aligned")
│   │   ├── baselines.py          # WholeDocumentChunker, BlockChunker
│   │   └── segmenters.py         # RegexSentenceSegmenter  (version "regex-1.0")
│   ├── embedding/
│   │   ├── builtin.py            # NullEmbeddingProvider, DeterministicEmbeddingProvider
│   │   └── local.py              # SentenceTransformerProvider
│   └── pipeline/
│       ├── planning.py           # chunk planning + block-alignment rule
│       ├── assembly.py           # plan → KnowledgeNode
│       ├── relations.py          # structural relation derivation
│       ├── enrichment.py         # plugin enrichers, bounded concurrency
│       ├── embedding.py          # batching, within-document
│       ├── composition.py        # HeadingPathComposer, TextOnlyComposer
│       └── validation.py         # provenance + coverage + block alignment
│
└── retrieval/                    # MODULE 3 — FROZEN
    ├── __init__.py               # build_default_engine, 101 exports, composition root
    ├── engine.py                 # RetrievalEngine, RetrievalOutcome, EmbeddingCoverageProbe
    ├── interfaces.py             # 13 protocols
    ├── models.py                 # partitions, filters, QueryPlan, Candidate, diagnostics
    ├── config.py                 # RetrievalConfig + 11 groups
    ├── exceptions.py             # 33 classes, 11 stages
    ├── partition.py              # PartitionResolver + derivation helpers
    ├── registry.py               # ArmRegistry, FusionRegistry
    ├── pipeline/
    │   ├── encoding.py           # ProviderQueryEncoder, QueryPreparer
    │   ├── arms.py               # DenseArm, SparseArm, StructuralArm
    │   ├── fusion.py             # ReciprocalRankFusion
    │   ├── hydration.py          # NodeHydrator, SupersessionFilter, HydratedResult
    │   ├── expansion.py          # OrdinalContextExpander
    │   └── assembly.py           # CredibilityAnnotator, EvidenceAssembler, AssembledEvidence
    └── stores/
        ├── tokenization.py       # the shared term rule — both lexical backends use it
        ├── memory.py             # 6 stores + writer. THE CORRECTNESS REFERENCE
        └── postgres/
            ├── schema.py         # DDL, driver-free (stdlib imports only)
            ├── filters.py        # RetrievalFilter → SQL. The injection boundary
            └── stores.py         # 6 stores + atomic writer + PostgresBackend

tests/core/{schemas,ingestion,knowledge,retrieval}/   # mirrors the package tree
```

**Naming convention for delivered files.** Files are delivered flat with prefixed
names (`retrieval_pg_stores.py`, `knowledge_structural.py`). Every file's
docstring opens with a `File:` block giving its real repo path. **Place files by
that header, not by the delivered filename.**

---

## 3. The shared contract layer — `ptfm/core/schemas/`

59 exports. Depends on nothing; every module depends on it. A change here is a
change to all four modules, so it needs approval.

### 3.1 What Module 3 hands Module 4

```python
class EvidenceItem(BaseModel):
    evidence_id: str
    document_id: str
    node_id: Optional[str] = None
    anchor_node_id: Optional[str] = None          # == node_id for an anchor
    discovery: tuple[DiscoveryRecord, ...] = ()
    span: Optional[TextSpan] = None
    block_ids: tuple[str, ...] = ()
    content: str
    source_uri: str
    retrieval_score: float = Field(ge=0.0, le=1.0)
    credibility_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)  # CC-1

class RetrievedDocument(BaseModel):
    document_id: str
    node_id: Optional[str] = None
    span: Optional[TextSpan] = None
    block_ids: tuple[str, ...] = ()
    title: Optional[str] = None
    content: str
    source_uri: str
    retrieval_method: RetrievalMethod
    signals: tuple[RetrievalSignal, ...] = ()
    retrieval_score: float = Field(ge=0.0, le=1.0)
    credibility_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)  # CC-1
    evidence_type: EvidenceType
    metadata: Dict[str, Any] = Field(default_factory=dict)
```

`DiscoveryMethod` = DENSE | SPARSE | GRAPH | EXPANSION | RERANK
`SignalScale` = COSINE | INNER_PRODUCT | DISTANCE | LEXICAL | RECIPROCAL_RANK | RERANK | RANK_ONLY

### 3.2 Module 4's target shapes — already defined in `schemas/verification.py`

**These exist and are unused.** Module 4 implements against them; it should not
invent parallel types.

| Class | Fields |
|---|---|
| `AtomicClaim` | `claim_id, claim_text, source_sentence, sentence_index, entities, relations` |
| `ClaimCollection` | `claims: List[AtomicClaim]` |
| `VerificationResult` | `claim_id, status, nli_score, symbolic_verification, supporting_evidence, contradicting_evidence, explanation` |
| `VerificationBatch` | `results` |
| `TruthSpaceClaim` | `claim_id, membership_probability, is_member, confidence, verification_status` |
| `TruthSpace` | `verified_claims: List[TruthSpaceClaim]` |
| `SemanticEntropy` | `entropy, num_generations` |
| `CalibrationResult` | `calibrated_confidence, nonconformity_score, conformal_threshold` |
| `ConfidenceMetrics` | `semantic_entropy, retrieval_reliability, calibrated_confidence, response_confidence, confidence_level` |
| `RetryPlan` | `retry_count, expanded_queries` |
| `DecisionOutcome` | `decision, confidence, reason, retry_plan` |
| `VerifiedResponse` | `response_text, decision, overall_confidence, verified_claims, supporting_sources, diagnostics, latency_ms` |

Enums: `DecisionEnum` = GENERATE | RETRY | REFUSE · `VerificationStatus` =
SUPPORTED | REFUTED | INSUFFICIENT_INFO · `ConfidenceLevel` = LOW | MEDIUM | HIGH

### 3.3 Knowledge schemas — the Evidence-First carriers

```python
class KnowledgeNode(BaseModel):
    node_id, document_id
    span: TextSpan                     # into RawDocument.text
    text: str                          # MUST equal raw.text[span.start:span.end]
    node_type: NodeType
    block_ids: tuple[str, ...]
    heading_path: tuple[str, ...] = ()
    ordinal: int                       # position in document; unique per (doc, partition)
    page_numbers: tuple[int, ...] = ()
    source: SourceDescriptor
    metadata: Dict[str, str]           # enricher-supplied; domain-facing
    segmentation_fingerprint: str      # partition identity
    source_text_hash: str
    source_schema_version: str
    processed_at: datetime
    schema_version: str

class NodeEmbedding(BaseModel):
    node_id, document_id
    vector: tuple[float, ...]
    dimension: int
    model_id: str
    model_revision: Optional[str]
    normalized: bool
    composition_id: str
    query_encoding: str = ''           # the query-side convention these vectors expect
    input_text_hash: str
    embedded_at: datetime
```

`NodeType` = PROSE | HEADING | TABLE_REGION | LIST_REGION | CODE | FIGURE_CAPTION (etc.)
`RelationType` = PRECEDES | FOLLOWS | PARENT_SECTION | CHILD_SECTION | SAME_TABLE | SAME_LIST | SAME_PAGE

**The Evidence-First invariant, enforced at every hop:**

```
node.text == raw_document.text[node.span.start : node.span.end]
```

---

## 4. Module 1 — Ingestion

Source → `RawDocument` with block structure, provenance and hashes.

```python
from ptfm.core.ingestion import build_default_engine

engine = build_default_engine(
    config=None, *, connectors=None, extractors=None,
    fingerprint_store=None, language_detector=None, hash_provider=None,
    offloader=None, clock=None, logger=None, metrics=None,
    discover_entry_points=True)

async for outcome in engine.ingest(spec): ...          # AsyncIterator[IngestionOutcome]
report = await engine.ingest_all(spec)                 # IngestionReport
outcome = await engine.ingest_one(record, connector)
```

Outcomes are a closed set: `DocumentIngested | DocumentSkipped | DocumentFailed`,
with `SkipReason` and `IngestionFailure` carrying why. A document never silently
disappears.

**Connectors:** `LocalFilesystemConnector`, `InMemoryConnector`.
**Extractors:** PDF, DOCX, HTML, Markdown, plaintext, tabular — one class each,
all producing `DocumentBlock` trees with spans into the normalised text.
**Config groups:** Acquisition, Detection, Extraction, Structure, Metadata,
Deduplication, Retry, Telemetry.
**Infrastructure protocols** (reused by Modules 2 and 3): `Clock`,
`HashProvider`, `ThreadOffloader`, `MetricsSink`, `StructuredLogger`.

`discover_entry_points=True` here — Module 1 predates the later convention
(§8) of defaulting discovery off. Left as built.

---

## 5. Module 2 — Knowledge Processing (FROZEN)

`RawDocument` → `KnowledgeUnit` (node + optional embedding + relations).

```python
from ptfm.core.knowledge import build_default_engine

engine = build_default_engine(
    config=None, *, sink=None, enrichers=(), strategies=None, segmenters=None,
    providers=None, composers=None, offloader=None, clock=None,
    logger=None, metrics=None, discover_entry_points=True)

engine.segmentation_fingerprint      # -> str   ties output to a partition
engine.embedding_model_identity      # -> Optional[str]
async for outcome in engine.process(documents): ...
report = await engine.process_all(documents)
outcomes = await engine.process_document(document)
```

**Protocols:** `ChunkingStrategy`, `SentenceSegmenter`, `EmbeddingProvider`,
`EmbeddingTextComposer`, `KnowledgeEnricher`, `KnowledgeSink`.
**Config groups:** Chunking, Embedding, Enrichment, Relations, Validation, Sink,
Telemetry.

### 5.1 Decisions that Module 3 and 4 depend on

**`overlap_chars` was removed, deliberately.** Overlapping chunks duplicate text,
and duplicated text enters verification as *two* evidence items — inflating
apparent corroboration, the exact quantity the system exists to calibrate. This
is the load-bearing constraint of the whole design: **reference, not
duplication**. If you ever reintroduce overlap, corroboration counting in
Module 4 becomes unsound.

**The block-alignment rule** is containment *plus* intersection: a node's span
must lie within its named blocks' extent, **and** the named blocks must be
exactly those leaves the span touches. Strictly stronger than containment alone
— it also catches spurious block naming.

**`segmentation_fingerprint` contributes a component iff that component can
change a span boundary.** This keeps partition identity minimal and meaningful.

**`query_encoding` is derived, not declared.** `SentenceTransformerProvider`
computes `_query_encoding(prefix)`, so the recorded convention cannot disagree
with the transformation actually applied. The provider *performs* encoding; the
vector *names* it.

**Embedding batching is within-document only**, and `_RunCounters.documents_seen`
counts distinct documents — counting node outcomes understated the failure ratio
(0.0164 vs 0.0500 measured).

**TD-1 (open, accepted).** `ChunkingStrategy` and `EmbeddingProvider` declare no
`__slots__`, so the six implementations that do declare it get a `__dict__`
anyway and their declaration is inert. No wrong output today; the guard against
per-document state on a shared strategy simply does nothing. Closes with two
lines and a regression test. Not fixed because Module 2 is frozen.

---

## 6. Module 3 — Retrieval (FROZEN)

Query → traceable evidence plus an honest account of what could not be
established. **No confidence, no calibration, no verification** — machine-checked.

### 6.1 Entry point

```python
from ptfm.core.retrieval import build_default_engine

engine = build_default_engine(
    config=None, *, stores=None, encoder=None, preprocessor=None,
    credibility=None, arm_registrations=(), fusion_strategies=(),
    clock=None, id_generator=None, logger=None, metrics=None,
    discover_plugins=False)

outcome = await engine.retrieve(query, *, top_k=None, filters=(), exact=None)
written = await engine.ingest(segmentation, embedding, units)
health  = await engine.health_check()      # -> Mapping[str, bool] per arm
await engine.close()
```

`build_default_engine()` with **no arguments** is complete and usable over the
in-memory stores. That is the test of whether defaults are real.

### 6.2 Execution flow — the order is not negotiable

```
1 PREPARE     resolve active segmentation → resolve embedding partition
              → verify model identity AND query-encoding convention → encode
2 SEARCH      arms concurrently, each at its own resolved fetch depth
              skip-with-reason: no vector | coverage below threshold
              ALL arms failed ⇒ AllArmsFailedError — never an empty result
3 FUSE        ReciprocalRankFusion (k=60, rank-only)
4 HYDRATE     NodeStore is the ONLY source of text and provenance
              provenance verified unconditionally; no off switch
5 CURRENCY    SupersessionFilter ⇒ (kept, superseded, unchecked)
              NEVER degrades
6 EXPAND      OrdinalContextExpander, around anchors only
7 CREDIBILITY anchor sources only; absence stays absent
8 ASSEMBLE    ⇒ AssembledEvidence + RetrievalDiagnostics
```

Hydration precedes everything that reads text because an index holds a copy that
may be stale, and a stale copy presented as evidence is a citation to something
the source does not say. Currency needs the source URI hydration resolves.
Expansion follows currency so context is never gathered around an anchor about
to be withheld. Credibility asks about anchors only, because scoring context
would put a passage nobody retrieved into a corroboration count.

### 6.3 What Module 4 receives

```python
class RetrievalOutcome:            # __slots__; evidence and diagnostics inseparable
    query: str
    evidence: AssembledEvidence    # .documents .evidence .anchor_count
                                   # .context_count .credibility_unavailable
    diagnostics: RetrievalDiagnostics
    degraded: bool                 # anything did not run
    reproducible: bool             # no approximate search contributed
    def to_result(self) -> RetrievalResult
```

`RetrievalDiagnostics` — every field Module 4 should read:

| Field | Meaning |
|---|---|
| `arms: tuple[ArmDiagnostic, ...]` | per-arm `ArmStatus`: COMPLETED / EMPTY / SKIPPED / FAILED / TIMED_OUT |
| `degraded`, `contributing_arms`, `failed_arms` | how much of the corpus was searched |
| `exact_search`, `reproducible` | whether this answer can be re-derived |
| `embedding_coverage: Optional[float]` | `None` = **unmeasured**, never "full" |
| `fused_count`, `expanded_count` | anchors vs accompanying context |
| `superseded_count` | checked and withheld |
| `currency_unchecked` | **no currency record existed** — a different claim (CC-2) |
| `credibility_unavailable: tuple[str, ...]` | sources with no credibility, listed not defaulted |
| `latency_ms`, `query_id`, `segmentation_key`, `embedding_key` | correlation |

### 6.4 The 13 protocols (all injectable)

`NodeStore`, `VectorIndex`, `LexicalIndex`, `RelationStore`, `SupersessionStore`,
`PartitionCatalog`, `IndexWriter`, `QueryEncoder`, `QueryPreprocessor`,
`RetrievalStrategy`, `FusionStrategy`, `ContextExpander`, `CredibilityProvider`.

Six store protocols are `@runtime_checkable`; `PartitionCatalog` is not — an
asymmetry, harmless, but know it before writing an `isinstance` check.

### 6.5 Partitions — why incomparability is unexpressible

Two compatibility keys, not one:

- `SegmentationPartition(segmentation_fingerprint, source_schema_version)` — keys
  nodes, lexical terms, relations, currency.
- `EmbeddingPartition(segmentation, model_identity, dimension, normalized,
  composition_id, query_encoding)` — keys vectors *only*.

That split **is** the migration story: re-embedding writes a new vector set and
leaves nodes, spans, identifiers and lexical terms untouched. Exactly one
segmentation and one embedding partition is active at a time, enforced by a
partial unique index in PostgreSQL and an invariant in memory — so no query can
observe two identifier spaces.

### 6.6 Degradation

| Situation | Behaviour |
|---|---|
| Degradable failure, `DEGRADE` | result reduced, recorded, `degraded = True` |
| Degradable failure, `STRICT` | query fails |
| Non-degradable failure | query fails, either policy |
| All arms failed | query fails, either policy |
| **Currency check failed** | **query fails, either policy** |

8 of 33 exceptions are degradable. Degradability is a property of the
*exception*, never of the stage, so two stages cannot classify one failure
differently.

The currency asymmetry is the rule worth memorising:

> Degradation is for **missing** evidence, never for **unverified** evidence. A
> missing arm yields less evidence, which diagnostics describe and a consumer can
> weigh. A superseded document served as current yields evidence that is
> confidently wrong and carries no marker a consumer could detect.

### 6.7 Backends

**In-memory (`stores/memory.py`) is the correctness reference.** Six stores plus
an atomic writer, one shared `matches_filters` implementation, three consumers.

**PostgreSQL (`stores/postgres/`) is the default for production**, and the reason
is atomicity, not speed. A knowledge unit's write spans five concerns — node,
vector, lexical terms, relations, currency. Across separate services that write
cannot be atomic, and a crash partway leaves a node one index returns and no
store can attest to. `is_atomic = True` is **proven by executing a rollback**,
not asserted.

Design notes worth knowing:

- One `ptfm_vector` table with an **unconstrained `vector` column** serves every
  embedding partition. HNSW is refused on such a column, so approximate indexes
  are **per-partition partial indexes over a dimension cast** — valid because a
  partition knows its dimension at registration.
- Approximate indexing is **off by default**: approximate results are not
  reproducible, and a caller auditing a past answer needs exact search. Exact
  search is always available.
- Lexical search lives as a **generated `tsvector` column on the node row**, so
  a node and its search terms cannot drift; the `'simple'` configuration applies
  no stemming and removes no stop words, because matching a term the caller did
  not write is not a property an evidence system should acquire by default.
- `ptfm_to_number(text)` is a plpgsql function returning NULL on bad input,
  because SQL has no total cast and the reference treats an ordered comparison
  against a non-numeric value as *no match*, not an error.
- **`stores/postgres/filters.py` is the injection boundary.** Every value is a
  bound parameter; only `FilterField`/`FilterOperator` enum members are composed
  structurally. `metadata_key` — the one name a caller supplies — is bound to a
  `jsonb` key operator, never interpolated as an identifier.

**Parity** (full detail in `claude/retrieval-backend-parity.md`):

*Identical*: filter semantics (382 cases), lexical **matching**, node provenance
field-by-field, absence-vs-emptiness, three-valued currency, partition
isolation, rank contiguity, scale naming.

*Deliberately different*: lexical **ranking** (`ts_rank_cd` vs Okapi BM25);
vector **precision** (pgvector stores float32, measured Δ 2.36e-08);
approximate search when enabled.

---

## 7. Design decisions register

The load-bearing ones, with the reason. Changing any of these changes the
system's guarantees.

| Decision | Reason |
|---|---|
| No chunk overlap | Duplicated text inflates corroboration, the quantity being calibrated |
| Evidence text comes only from the node store | An index's stale copy presented as evidence is a false citation |
| Provenance verification has no off switch | A check that can be disabled reads as present when off |
| Compatibility partitions, two keys | Makes incomparability *unexpressible*, not merely detected |
| RRF, k=60, rank-only | Scale-free. Verified: inflating one arm's scores 1.0 → 999.0 leaves fused values bit-identical |
| Signals uncalibrated, each with a named scale | A scale names the quantity; it does not license cross-scale comparison |
| `filter_superseded` cannot be `False` | Currency is a correctness guarantee, not a preference |
| Credibility absence representable (`Optional[float]`) | CC-1 — see §9 |
| `currency_unchecked` reported separately | CC-2 — "0 superseded" must not read as "all current" |
| All arms failed ⇒ raise | An empty result asserts the corpus holds nothing; a failed search asserts nothing |
| Structural search with no filter returns nothing | An unconstrained structural search returns an arbitrary slice of the corpus |
| Shared tokenizer in `stores/tokenization.py` | One term rule, two backends. Ranking may differ; matching may not |
| Query encoding verified **before** inference | A mismatch degrades retrieval measurably and produces no error of its own |
| Entry-point discovery off by default (Module 3) | Behaviour should not change because an unrelated package was installed |
| Query text logged only when `log_query_text` is on | A privacy control defaulting to permissive makes the safe setting the one an operator must discover |

---

## 8. Conventions actually followed

**Module structure.** Docstring (with `File:` path, a "why this exists" section,
and a *Dependency contract* section) → stdlib imports → third-party → internal →
custom exceptions → classes/functions. Internal imports alphabetical.

**Documentation.** Every public class and function documents purpose, parameters,
returns, raises, and the equation when applicable. Docstrings state *why* a
decision was made, not only what the code does — that is how the reasoning
survives a handoff. Private module-level helpers are documented too.

**Typing.** Complete hints on every public signature. API boundaries are
Pydantic v2 models. Non-serialising internal aggregates (`HydratedResult`,
`AssembledEvidence`, `RetrievalOutcome`, `RetrievalStores`) are plain
`__slots__` classes.

**`__slots__`.** On every non-Pydantic, non-Enum, non-Protocol, non-exception
class — *including base classes*, because Python gives a subclass a `__dict__`
unless every class in the MRO declares it (this is TD-1).

**Immutability.** `model_config = ConfigDict(frozen=True, extra="forbid")` on
models; `validate_default=True` on config.

**Dependency injection everywhere.** No stage constructs a collaborator. The
composition root (`__init__.py`) is the only place a concrete store is named.

**Exceptions.** Custom PTFM types, never generic. `stage` and `degradable` as
`ClassVar`s. `StoreUnavailableError` requires `store_name=` — a store must
identify itself.

**Config naming.** `PTFM_RETRIEVAL_` prefix, `__` nested delimiter.

**Async.** Native async throughout; CPU-bound work through `ThreadOffloader`. No
blocking I/O in an async function — machine-checked.

**Secrets.** Connection strings are injected and never appear in a `__repr__`, a
log, or an exception. Every store's `__repr__` names the store and nothing else.

**Testing.** pytest, `--asyncio-mode=auto`, hypothesis for property tests. Test
docstrings state what property is being protected and why — a test that only
says what it calls does not survive a refactor.

---

## 9. Contract changes

**CC-1 — credibility absence is representable** (approved, applied).
`EvidenceItem.credibility_score`, `RetrievedDocument.credibility_score`,
`BaseDomainPlugin.source_credibility` → `Optional[float]`.

Why it justified a cross-module change: credibility feeds `r(E) = (1/k)·Σ rᵢsᵢ`
and through it `H = S(q)·(1−r(E))`. On a three-item set with one unscoreable
source at `S(q)=0.8`, hallucination risk spans **0.260 to 0.440** purely on what
value is substituted. Discarding the item is not neutral either — it raises
reliability, recording ignorance as confidence.

**CC-2 — `RetrievalDiagnostics.currency_unchecked`** added, additive, default
`0`. The count was already computed and had nowhere to go.

---

## 10. Verification completed

Re-verified 2026-09-25 from a cold container start, against a live PostgreSQL
16.13 / pgvector 0.6.0 cluster.

| Suite | Proves | Result |
|---|---|---|
| `t_diff.py` | SQL filter translation vs the reference: 382 field/operator/value combinations incl. injection payloads | **0 mismatches** |
| `t_pg.py` | PostgreSQL stores, atomicity, partition isolation, mixed dimensions | **71/71** |
| `t_e2e.py` | Engine end to end over the reference | **62/62** |
| `t_e2e_pg.py` | **Same engine, unchanged, over the live database** | **24/24** |
| `t_fr1.py` | Telemetry privacy control, both modes | **16/16** |
| `tests/` (14 modules, Module 3) | Full unit suite | **675 passed** |
| `t_guards.py` | Compliance guards | **52 PASS** |
| `audit_freeze.py` | Architecture audit — see §11 | runs clean but for §11 items |

**1,282 executed assertions for Module 3 alone.** Modules 1 and 2 carry their
own suites (`tests/core/ingestion/`, `tests/core/knowledge/`) not re-run in this
session.

**Evidence-First is machine-checked, not trusted.** Fixtures build every node as
a literal slice of a known source document, so the chain is verified by
comparison on both backends:

```
item.content == node.text == RAW[document_id][node.span.start : node.span.end]
```

**Atomicity is proven by rollback.** A batch whose second group violates the
position-uniqueness index leaves the row count unchanged, leaves the valid node
from the first group unwritten, and leaves the buffer intact for retry.

**A note on the harness.** The sandbox's Pydantic stand-in originally enforced
neither `frozen` nor `extra="forbid"`, nor nested coercion or env loading,
leaving 20 tests of exactly those constraints unfalsifiable and silently
failing. It was hardened; that is what turned a reported 614 assertions into
1,282 genuinely executed ones. **If you rebuild a sandbox, verify it enforces
the constraints your tests assert** — a test that cannot exercise a constraint
verifies nothing about it.

---

## 11. Open items

### 11.1 Technical debt

**TD-1** — `__slots__` inert on Module 2's strategy base classes. Open by
decision (Module 2 frozen). §5.1.

**FR-2** — third-party exception messages are an unfiltered log channel.
`message=error.message` on `retrieval.query.encoding_degraded` and
`message=str(exc)` on `retrieval.query.preprocessing_degraded` relay strings
built by third-party providers/preprocessors. A component embedding the query in
its exception message reaches the log whatever `log_query_text` is set to. No
first-party code does this. If the guarantee is "no query text in logs under any
circumstance", these two need the same gate or a redaction pass.

### 11.2 Nine dead configuration fields — reported, undisposed by decision

None affects correctness; none is a privacy control. Listed with recommendation
in `claude/module-3-freeze-review.md` §7.2. The two worth knowing:

- `PartitionConfig.require_active_partition` — the **flag is inert but the
  guarantee is unconditionally enforced** (`resolve_for_query` always raises).
  Setting it `False` currently misleads. Should become a documented constant.
- `RetrievalConfig.store_thread_limit` — field *and* `resolve_thread_limit()`
  both uncalled. Recommend removal; both backends are natively async.

Others: `SparseConfig.analyzer`, `GraphConfig.max_traversal_depth` (reserved for
the deferred graph backend), `CredibilityConfig.cache_size`,
`IngestConfig.write_timeout_seconds`, `IngestConfig.max_retries`,
`TelemetryConfig.log_level`, `TelemetryConfig.log_result_ids`.

### 11.3 Three exceptions never raised

`IncomparableSignalsError` (reserved for a score-combining fusion strategy),
`AmbiguousPartitionError` (catalog uniqueness enforced structurally instead),
`CredibilityUnavailableError` (superseded by CC-1 — recommend removal).

### 11.4 Deferred extension points — defined, not implemented

| Point | Attaches at | Why deferred |
|---|---|---|
| **Reranking stage** | Between fusion and hydration | A reranker scoring index copies rather than authoritative text defeats the reason hydration comes first. **Must run on hydrated text.** |
| **Graph database backend** | `RelationStore` | Current structural arm is filter-based, not traversal-based |
| **Qdrant backend** | `VectorIndex` | Would forfeit single-transaction atomicity. Contracts already permit it |
| **Learned-sparse arm** | `RetrievalStrategy` + entry point | Needs a model; no contract change |
| **Cross-partition query** | Would relax partition isolation | Deliberately not permitted |

---

## 12. Module 4 — what to build next

`ptfm/core/verification/` — Verification, Calibration, Verifiable Truth Space,
and the Decision Engine. Target shapes already exist in
`ptfm/core/schemas/verification.py` (§3.2); implement against them.

### 12.1 Scope, from MathSpec

| MathSpec § | Component | Notes |
|---|---|---|
| §1 | Atomic claim extraction `f: R → C` | Atomic, independently verifiable, context-preserving, semantically complete. **Verify claims, never paragraphs.** |
| §4 | Neural NLI: `V_raw = p_e − p_c` | **Never** pass raw neural scores to the decision engine |
| §5 | Symbolic verification `P(c,E) ∈ {0,1}` | Plugin-supplied. When symbolic verification exists, do not rely on NLI alone |
| §6 | Semantic entropy `S(q) = −Σ p(Cₖ|q) log p(Cₖ|q)` | Over semantic clusters of N generations |
| §7 | Nonconformity `α(c) = 1 − Ĉ(c)` | |
| §8 | Conformal threshold `q̂ = Q(⌈(n+1)(1−α)⌉ / n)` | Coverage `P(c ∈ Γ(E) \| c true) ≥ 1−α` |
| §2 | VTS `Γ(E) = {c : V(c,E) ≥ τ ∧ P(c,E) = 1}` | **Only claims inside Γ may reach a response** |
| §9 | `C_response = (Π Ĉᵢ)^(1/m)` | Geometric mean — one weak claim cannot be masked |
| §3 | `r(E) = (1/k) Σ rᵢsᵢ` | **Read §12.3 before implementing this** |
| §10 | `H = S(q)(1 − r(E))` | |
| §11 | GENERATE if `C_response ≥ max(τ, q̂)`; RETRY if `τ_low ≤ C_response < τ` and `retry_count < N_max`; else REFUSE | Thresholds from conformal calibration, **never hardcoded** |

### 12.2 Suggested shape, following Modules 2–3

```
ptfm/core/verification/
├── __init__.py        # build_default_engine + composition root
├── engine.py          # VerificationEngine orchestrating the stages
├── interfaces.py      # ClaimExtractor, NLIModel, SymbolicVerifier,
│                      # EntropyEstimator, Calibrator, DecisionPolicy
├── models.py          # internal aggregates (plain __slots__ classes)
├── config.py          # VerificationConfig + groups, PTFM_VERIFICATION_ prefix
├── exceptions.py      # stage enum + degradable/retryable ClassVars
└── pipeline/
    ├── extraction.py  # §1
    ├── nli.py         # §4 — PyTorch, through ThreadOffloader
    ├── symbolic.py     # §5 — plugin-driven
    ├── entropy.py     # §6
    ├── calibration.py # §7, §8
    ├── truth_space.py # §2
    └── decision.py    # §9, §11
```

Reuse: `Clock`, `IdGenerator`, `MetricsSink`, `StructuredLogger`,
`ThreadOffloader` from `ptfm.core.ingestion.interfaces`.

### 12.3 Risks Module 4 must explicitly account for

These are the ones that will produce a *plausible wrong number* if missed.

**R-1 — `credibility_score` may be `None`, and that is not zero.** CC-1 made
absence representable precisely so `r(E)` cannot silently multiply past it.
Module 4 must decide explicitly what unknown credibility means. Substituting any
value swings `H` by up to 69% on a small evidence set; discarding unscoreable
items biases reliability *upward*, recording ignorance as confidence. Neither is
a default — it is a modelling decision Module 4 owns. `None` raises rather than
yielding a plausible number; that is the point.

**R-2 — `retrieval_score` is ordinal, not probabilistic.** It is `1/(1+rank)`.
Not a similarity, not a likelihood, not a confidence. Feeding it into `r(E)` as
`rᵢ` without a documented transform gives a statistical guarantee an input with
no distributional meaning.

**R-3 — corroboration counts anchors only.** `anchor_count` is the
corroboration-bearing quantity. Context nodes carry
`DiscoveryMethod.EXPANSION`. **Counting `len(evidence)` as corroboration inflates
it** — the precise harm removing `overlap_chars` was meant to prevent. Filter on
`item.node_id == item.anchor_node_id`.

**R-4 — degraded ≠ less reliable.** `degraded = True` means an arm did not run.
What is present is fully traceable; what is unknown is what a missing arm would
have found. Treat as reduced *recall*, not reduced reliability of what is there.

**R-5 — `currency_unchecked > 0` means the currency filter was inoperative for
those results.** Distinct from `superseded_count`. Reading "0 superseded" as
"everything is current" is wrong whenever `currency_unchecked > 0`. A currency
*outage* never reaches Module 4 — it fails the query.

**R-6 — `embedding_coverage = None` means unmeasured, not full.** Treating it as
full licenses trust in a dense arm that may have searched a fraction of the
corpus.

**R-7 — `reproducible = False` means the answer cannot be re-derived.** Any
audit trail or appeal path must record `exact_search`, and re-verification must
pass `exact=True`.

**R-8 — lexical ranking is backend-dependent.** Same matched set, different
order on PostgreSQL vs the reference. Never persist a rank as a property of the
corpus; compare sets, not sequences, in cross-backend baselines.

**R-9 — never catch `AllArmsFailedError` and proceed with zero evidence.**
Module 3 goes to some length to ensure a failed search never presents as an
empty corpus. Swallowing it reintroduces exactly the failure the design
prevents. **A refusal is the correct outcome, not an unsupported answer.**

**R-10 — `model_identity` is a property on `QueryEncoder` and a method on
Module 2's `EmbeddingProvider`.** A real naming hazard across the boundary; it
caused a fault that survived nine implementation steps because the only
implementation agreed with the only caller.

**R-11 — RETRY must feed back into retrieval, not into generation.** MathSpec
§11 specifies expanded search, query decomposition and claim-targeted retrieval.
Module 3 supports all three through `filters`, `top_k` and re-querying — no
contract change needed. Bound retries by `N_max` and record `retry_count` in
`RetryPlan`.

### 12.4 First things to do

1. Read `claude/module-3-freeze-review.md` §3 (the contract) and §8 (risks).
2. Decide R-1 explicitly and write the decision down — it gates `r(E)`, and
   therefore everything downstream.
3. Decide R-2's transform from ordinal rank to relevance `rᵢ`, and document why.
4. Write `ptfm/core/verification/exceptions.py` first, as in Modules 2–3: the
   stage taxonomy and degradability model shape every later signature.
5. Port `audit_freeze.py` to the new module as soon as it has a config class.
   It found the last two defects and it is cheap.

---

## 13. Development environment

Reproducing the verification setup:

```bash
# PostgreSQL 16 + pgvector, socket-only (listen_addresses='')
PGDATA=/var/lib/postgresql/ptfm        # NOT /tmp — permission denied for postgres user
pg_ctl -D $PGDATA -o "-k /tmp -p 55432" start
# database ptfm, extension: CREATE EXTENSION vector;
apt-get install -y postgresql-16-pgvector
pip install psycopg psycopg_pool pytest pytest-asyncio hypothesis
```

DSN used: `host=/tmp port=55432 dbname=ptfm user=postgres`

A `shim/` directory provides executable stand-ins for `pydantic` and
`pydantic_settings` so the real module code runs without those packages. The
shim **must** enforce `frozen`, `extra="forbid"`, nested-model coercion from
dicts, and `PTFM_*` env loading with `__` nesting — see §10's note on why.

Verification scripts (not part of the shipped package): `t_diff.py`, `t_pg.py`,
`t_e2e.py`, `t_e2e_pg.py`, `t_fr1.py`, `t_guards.py`, `audit_freeze.py`.
`audit_freeze.py` is worth keeping in CI: it exits non-zero on MUST FIX and it
is what found the last privacy defect.

---

## 14. Working agreement

The process that produced this, worth continuing:

- **Architecture is approved before implementation.** Once approved, routine
  implementation choices are delegated; a single word ("continue", "proceed")
  means advance to the next planned step without re-litigating.
- **Stop only** for a change to a frozen contract or a genuine architectural
  decision. Report defects found; do not silently weaken an invariant to make
  implementation easier.
- **Findings are classified** MUST FIX / SHOULD FIX / DEFER / ACCEPT, and
  deferred items are written down. A defect found and left unrecorded is
  indistinguishable from one never found.
- **Claims are verified by execution.** "It should work" is not a result. Every
  defect of consequence in this project was found by running something.
