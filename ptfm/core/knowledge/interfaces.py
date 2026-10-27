"""
PTFM Knowledge Processing Contracts

File:
    ptfm/core/knowledge/interfaces.py

Description:
    Defines every abstraction on which the Knowledge Processing Module
    depends. The engine and its pipeline stages reference ONLY the types
    declared here; they never import a concrete strategy, provider, enricher,
    or sink.

Three orthogonal abstractions
-----------------------------
    ChunkingStrategy    answers "where do unit boundaries fall?"
    EmbeddingProvider   answers "how does text become a vector?"
    KnowledgeEnricher   answers "what domain metadata attaches?"

    These are never merged. Collapsing any pair produces a multiplicative
    implementation matrix; kept orthogonal, a new embedding model requires no
    chunking change and a new domain plugin requires neither.

Execution contract
------------------
    Segmentation is synchronous. Planning is CPU-bound with no I/O, and
    declaring it asynchronous would invite blocking work onto the event loop
    behind a coroutine that conceals it. The engine offloads it to a worker
    thread.

    Embedding, enrichment, and sink writes are asynchronous. Each crosses a
    process, network, or storage boundary.

    Both halves are enforced at class-definition time rather than left to
    documentation.

Provenance protection
---------------------
    Two contracts are shaped specifically so that provenance cannot be
    violated by an implementation rather than merely should not be:

        ChunkingStrategy returns a PLAN, not nodes. A strategy names spans;
        the assembly stage constructs nodes and enforces the invariants. A
        strategy therefore has no means of producing a node whose text and
        span disagree.

        KnowledgeEnricher receives an immutable node and returns METADATA
        ONLY. It is never handed a mutable node and its return value cannot
        reach text, span, or identity, so a plugin cannot alter what a
        citation resolves to.

Reuse
-----
    ``Clock``, ``HashProvider``, ``ThreadOffloader``, ``IdGenerator``,
    ``MetricsSink`` and ``StructuredLogger`` are imported from the ingestion
    contract layer rather than redeclared. One vocabulary for these concerns
    spans both modules; a parallel declaration would allow the two to diverge.

Dependency contract
-------------------
    Imports only the standard library, ``ptfm.core.knowledge.exceptions`` and
    ``ptfm.core.ingestion.interfaces``. Schema and model types are imported
    under ``TYPE_CHECKING`` exclusively, so this module carries no runtime
    dependency on either and its tests run before ``models`` exists.
"""

from __future__ import annotations

import inspect
from types import TracebackType
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncIterator,
    Mapping,
    NamedTuple,
    Optional,
    Protocol,
    Sequence,
    runtime_checkable,
)

from ptfm.core.ingestion.interfaces import (
    Clock,
    HashProvider,
    IdGenerator,
    MetricsSink,
    StructuredLogger,
    ThreadOffloader,
)
from ptfm.core.knowledge.exceptions import (
    ChunkerRegistrationError,
    EmbedderRegistrationError,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ptfm.core.knowledge.models import ChunkPlan, KnowledgeUnit
    from ptfm.core.schemas import KnowledgeNode, RawDocument

__all__ = [
    # Re-exported from the ingestion contract layer
    "Clock",
    "HashProvider",
    "IdGenerator",
    "MetricsSink",
    "StructuredLogger",
    "ThreadOffloader",
    # Value types
    "ModelIdentity",
    "Sentence",
    "ComposedInput",
    # Contracts
    "ChunkingStrategy",
    "EmbeddingProvider",
    "KnowledgeEnricher",
    "KnowledgeSink",
    "SentenceSegmenter",
    "EmbeddingTextComposer",
]


# ============================================================================
# VALUE TYPES
# ============================================================================


class ModelIdentity(NamedTuple):
    """
    Identity of an embedding model, recorded on every vector it produces.

    Without this, a corpus embedded across a model upgrade becomes an
    undifferentiated mixture whose similarity scores are meaningless, and the
    condition is undetectable after the fact.

    Attributes
    ----------
    model_id:
        Stable model identifier, for example ``"BAAI/bge-large-en-v1.5"``.
    dimension:
        Vector dimension the model produces.
    normalized:
        Whether the model returns unit-magnitude vectors. Determines whether a
        dot product equals cosine similarity, so a wrong value silently
        corrupts every comparison. Verified against the vector at
        construction of ``NodeEmbedding``.
    revision:
        Commit, tag, or version pin, when the provider reports one. A model
        identifier alone is not always sufficient, since weights behind a name
        can change.
    max_input_chars:
        Longest input the model accepts, in characters. Expressed in
        characters rather than tokens so that the contract binds no particular
        tokenizer.
    query_encoding:
        Identifier of the transformation this provider applies to a query
        before embedding it.

        Recorded on every vector, because a passage and a query must be
        encoded compatibly for their similarity to mean anything. Several
        widely used retrieval models are trained asymmetrically, expecting a
        distinct instruction or prefix on each side; embedding a query as
        though it were a passage degrades retrieval measurably and produces no
        error at all.

        The identifier names the convention rather than describing it. What a
        convention DOES is the provider's own knowledge, applied through
        :meth:`EmbeddingProvider.embed_query`, so no consumer needs to
        reproduce provider-specific behaviour. The identifier exists so that a
        consumer can verify the provider it holds is the one that produced the
        vectors it is querying against.

        Empty for a symmetric model, which encodes both sides identically.
    """

    model_id: str
    dimension: int
    normalized: bool
    revision: Optional[str] = None
    max_input_chars: Optional[int] = None
    query_encoding: str = ""

    @property
    def identity(self) -> str:
        """
        The identifier together with its revision, where known.

        Returns
        -------
        str
            A single identity string suitable for grouping and comparison.
        """
        if self.revision is None:
            return self.model_id
        return f"{self.model_id}@{self.revision}"


class Sentence(NamedTuple):
    """
    A sentence located within a text by character offset.

    Offsets are mandatory, not incidental. A segmenter returning sentence
    strings alone would be unusable here: a strategy must place boundaries as
    offsets into the document text, and recovering an offset by searching for
    a returned string is both fragile and wrong wherever the string occurs
    more than once.

    Attributes
    ----------
    start:
        Offset of the sentence's first character, relative to the text
        segmented.
    end:
        Offset one past the sentence's last character.
    """

    start: int
    end: int

    @property
    def length(self) -> int:
        """
        Number of characters in the sentence.

        Returns
        -------
        int
            ``end - start``.
        """
        return self.end - self.start


class ComposedInput(NamedTuple):
    """
    Text submitted to an embedding model, together with what produced it.

    The composed text may differ from the node's stored text, since it may
    carry structural context such as the heading path. What is stored stays an
    exact substring of the source document; what is embedded carries context.

    Attributes
    ----------
    text:
        The text to embed.
    composition_id:
        Identifier of the composition rule applied. Recorded on the resulting
        embedding, so that a vector produced with heading context is
        distinguishable from one produced without it. Comparing across
        compositions yields a number that resembles a similarity and means
        nothing.
    """

    text: str
    composition_id: str


# ============================================================================
# SEGMENTATION
# ============================================================================


class ChunkingStrategy:
    """
    Decides where knowledge node boundaries fall within a document.

    Returns a PLAN rather than nodes. A strategy names spans; the assembly
    stage materializes them and enforces the provenance invariants. This
    separation means a strategy has no means of producing a node whose text
    and span disagree, so a defective strategy fails at plan validation rather
    than by corrupting a corpus.

    Implementation contract
    -----------------------
    - :meth:`plan` MUST be synchronous. Planning is CPU-bound with no I/O, and
      an asynchronous signature would conceal blocking work on the event loop.
      The engine offloads the call to a worker thread.

    - :meth:`plan` MUST be deterministic. The same document under the same
      configuration must yield the same boundaries, since node identity is
      derived from them. A strategy whose boundaries vary between runs would
      produce a new identifier for unchanged content, breaking every stored
      citation.

    - :meth:`plan` MUST NOT mutate the document or retain state between calls.
      A single instance serves all concurrent workers.

    - Every parameter able to change a boundary MUST be reported by
      :meth:`fingerprint_components`. A parameter omitted there would change
      node identity without changing the fingerprint, which is the precise
      condition that makes an index silently unresolvable against its corpus.

    The synchronous requirement is enforced at class-definition time.
    """

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """
        Verify the synchronous contract of a strategy implementation.

        Parameters
        ----------
        **kwargs:
            Forwarded to ``object.__init_subclass__``.

        Returns
        -------
        None

        Raises
        ------
        ChunkerRegistrationError
            If ``plan`` is declared as a coroutine or async generator
            function.
        """
        super().__init_subclass__(**kwargs)

        plan = cls.__dict__.get("plan")
        if plan is None:
            return
        if inspect.iscoroutinefunction(plan) or inspect.isasyncgenfunction(plan):
            raise ChunkerRegistrationError(
                f"{cls.__name__}.plan must be synchronous. Planning is "
                f"CPU-bound and is executed in a worker thread by the engine; "
                f"an async signature would conceal blocking work on the event "
                f"loop.",
                strategy_name=cls.__name__,
            )

    @property
    def name(self) -> str:
        """
        Stable identifier of this strategy.

        Participates in the segmentation fingerprint, so renaming a strategy
        is a corpus migration rather than a cosmetic change.

        Returns
        -------
        str
            Implementation identifier.
        """
        raise NotImplementedError

    def plan(self, document: RawDocument) -> ChunkPlan:
        """
        Determine the span boundaries for a document.

        Parameters
        ----------
        document:
            The source document. Its normalized text and block tree are the
            only inputs; a strategy must not consult anything external, or it
            would not be deterministic.

        Returns
        -------
        ChunkPlan
            Ordered, non-overlapping, block-aligned spans covering the
            document's content.

        Raises
        ------
        InvalidChunkPlanError
            If the strategy cannot produce a valid plan.
        UnsplittableBlockError
            If a block exceeds the size budget and offers no interior
            boundary.
        """
        raise NotImplementedError

    def fingerprint_components(self) -> Mapping[str, str]:
        """
        Report every parameter able to change a span boundary.

        The inclusion rule is exact: a parameter belongs here IF AND ONLY IF
        changing it can change any node's span.

        Embedding models, enrichers, and metadata configuration never belong
        here. Including one would mean that adding an enricher invalidates
        every node identifier in the corpus, breaking every stored citation,
        for a change that alters no node's text or span. Omitting a parameter
        that does affect boundaries is the opposite failure: identities change
        while the fingerprint does not, so an index silently fails to resolve
        against its corpus.

        Returns
        -------
        Mapping[str, str]
            Component names mapped to their values, in a form stable across
            processes. Values are strings so that the digest does not depend
            on numeric formatting.
        """
        raise NotImplementedError


@runtime_checkable
class SentenceSegmenter(Protocol):
    """
    Divides a text into sentences located by character offset.

    Used only where a single block exceeds the size budget and must be split
    internally. A block boundary is an author's own division and is always
    preferred; a sentence boundary is a fallback.

    A segmenter's version participates in the segmentation fingerprint,
    because upgrading one shifts boundaries within oversized blocks and
    therefore changes node identity.

    Note that ``runtime_checkable`` verifies member presence only. Signature
    conformance is established by ``mypy --strict``.
    """

    @property
    def version(self) -> str:
        """
        Version identifier of this segmenter and its underlying library.

        Contributes to the segmentation fingerprint. An upgrade that shifts a
        boundary must be detectable, or nodes produced before and after would
        share identifiers while covering different text.

        Returns
        -------
        str
            Version identifier.
        """
        ...

    def segment(self, text: str) -> Sequence[Sentence]:
        """
        Locate the sentences within a text.

        Offsets are relative to the text supplied, and the caller translates
        them into document coordinates.

        Parameters
        ----------
        text:
            Text to segment.

        Returns
        -------
        Sequence[Sentence]
            Sentences in reading order, ordered and non-overlapping. May be
            empty for a text carrying no sentence boundary, in which case the
            caller treats the text as unsplittable.
        """
        ...


# ============================================================================
# EMBEDDING
# ============================================================================


class EmbeddingProvider:
    """
    Converts text into vectors under a named model.

    Batch-oriented at the interface, so that a provider can exploit batching
    natively rather than the engine simulating it one input at a time. A model
    embedding a single text at a time wastes most of the throughput available
    to it.

    Implementation contract
    -----------------------
    - :meth:`embed` MUST be asynchronous. Every provider crosses a process or
      network boundary, whether to a local model in a worker thread or to a
      remote service.

    - :meth:`embed` MUST return one vector per input, in input order. A
      provider reordering or dropping results would associate vectors with the
      wrong nodes, which no downstream check could detect.

    - Every returned vector MUST match the declared dimension, and MUST
      genuinely be normalized when the identity claims normalization. A false
      claim makes dot product diverge from cosine similarity and corrupts
      every comparison, invisibly.

    - :meth:`model_identity` MUST be stable for the lifetime of the instance.
      Identity is recorded on every vector, so a provider changing model
      mid-run would produce a corpus whose vectors are not mutually
      comparable.

    The asynchronous requirement is enforced at class-definition time.
    """

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """
        Verify the asynchronous contract of a provider implementation.

        Parameters
        ----------
        **kwargs:
            Forwarded to ``object.__init_subclass__``.

        Returns
        -------
        None

        Raises
        ------
        EmbedderRegistrationError
            If ``embed``, ``health_check`` or ``close`` is defined and is not
            a coroutine function.
        """
        super().__init_subclass__(**kwargs)

        for method_name in ("embed", "embed_query", "health_check", "close"):
            member = cls.__dict__.get(method_name)
            if member is None:
                continue
            if not inspect.iscoroutinefunction(member):
                raise EmbedderRegistrationError(
                    f"{cls.__name__}.{method_name} must be a coroutine "
                    f"function declared with 'async def'. Embedding crosses a "
                    f"process or network boundary and must not block the "
                    f"event loop.",
                    provider_name=cls.__name__,
                )

    @property
    def name(self) -> str:
        """
        Stable identifier of this provider implementation.

        Distinct from the model identity: one provider implementation may
        serve several models.

        Returns
        -------
        str
            Implementation identifier.
        """
        raise NotImplementedError

    @property
    def max_batch_size(self) -> int:
        """
        Largest number of inputs accepted in one call.

        The engine batches up to this bound, so a provider unable to exploit
        batching reports one rather than requiring the engine to know that.

        Returns
        -------
        int
            Maximum batch size, at least one.
        """
        raise NotImplementedError

    def model_identity(self) -> ModelIdentity:
        """
        Report the model this provider embeds with.

        Recorded on every vector produced, which is what allows a corpus
        embedded across a model change to be distinguished rather than
        silently mixed.

        Returns
        -------
        ModelIdentity
            Identifier, dimension, normalization, and optional revision.

        Raises
        ------
        ModelIdentityError
            If the identity cannot be determined.
        """
        raise NotImplementedError

    async def embed(self, texts: Sequence[str]) -> Sequence[tuple[float, ...]]:
        """
        Convert a batch of texts into vectors.

        Parameters
        ----------
        texts:
            Texts to embed. Length must not exceed :attr:`max_batch_size`.

        Returns
        -------
        Sequence[tuple[float, ...]]
            One vector per input, in input order. Positional correspondence is
            the contract: a provider that reorders results would attach
            vectors to the wrong nodes undetectably.

        Raises
        ------
        EmbeddingProviderError
            If the provider failed. Transient and eligible for retry.
        EmbeddingTimeoutError
            If the request exceeded its deadline.
        DimensionMismatchError
            If a returned vector does not match the declared dimension.
        """
        raise NotImplementedError

    async def embed_query(self, query: str) -> tuple[float, ...]:
        """
        Embed a query so that it is comparable with embedded passages.

        Declared on the contract, rather than left to a consumer, because the
        transformation a model expects on the query side is the provider's own
        knowledge. A consumer reconstructing it from configuration would be
        guessing at provider-specific behaviour, and guessing wrong produces
        no error: results are returned and ranked, merely worse.

        A symmetric model applies no transformation and this is equivalent to
        embedding the query as a passage. An asymmetric one applies whatever
        its training requires, and names that convention through
        :attr:`ModelIdentity.query_encoding` so a consumer can verify
        compatibility with the vectors it holds.

        Parameters
        ----------
        query:
            The query text.

        Returns
        -------
        tuple[float, ...]
            A vector comparable with those produced by :meth:`embed`.

        Raises
        ------
        EmbeddingProviderError
            If the provider failed.
        EmbeddingTimeoutError
            If the request exceeded its deadline.
        """
        raise NotImplementedError

    async def health_check(self) -> bool:
        """
        Verify that the provider is operational.

        Invoked before a run begins so that a missing model or unreachable
        service fails immediately rather than after a corpus has been
        segmented.

        Returns
        -------
        bool
            ``True`` when the provider can embed.
        """
        raise NotImplementedError

    async def close(self) -> None:
        """
        Release resources held by this provider.

        Must be idempotent. A local model holds substantial memory, and a
        remote provider holds connections.

        Returns
        -------
        None
        """
        raise NotImplementedError

    async def __aenter__(self) -> EmbeddingProvider:
        """
        Enter an asynchronous context managing this provider's lifetime.

        Returns
        -------
        EmbeddingProvider
            This provider.
        """
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """
        Close the provider on context exit, including on failure.

        Parameters
        ----------
        exc_type:
            Type of the propagating exception, if any.
        exc:
            The propagating exception, if any.
        traceback:
            Traceback of the propagating exception, if any.

        Returns
        -------
        None
        """
        await self.close()


@runtime_checkable
class EmbeddingTextComposer(Protocol):
    """
    Builds the text submitted to an embedding model from a node's fields.

    Exists so that structural context can reach the vector without reaching
    the node's stored text. A chunk reading "The applicable rate is 4.5
    percent" is nearly useless to retrieval without "Interest Rates", but
    prepending that heading to the stored text would break the invariant that
    the text is an exact substring of the source document.

    The composition applied is recorded on the resulting embedding, since a
    node embedded with its heading path is not comparable to the same node
    embedded without it.
    """

    @property
    def composition_id(self) -> str:
        """
        Stable identifier of this composition rule.

        Recorded on every embedding produced with it, so that vectors from
        differing compositions are distinguishable rather than silently
        compared.

        Returns
        -------
        str
            Composition identifier.
        """
        ...

    def compose(self, node: KnowledgeNode) -> ComposedInput:
        """
        Build the embedding input for a node.

        Parameters
        ----------
        node:
            The node to embed. Immutable, and never modified by composition.

        Returns
        -------
        ComposedInput
            The text to embed and the identifier of the rule that produced it.
        """
        ...


# ============================================================================
# ENRICHMENT
# ============================================================================


@runtime_checkable
class KnowledgeEnricher(Protocol):
    """
    Attaches domain metadata to a knowledge node.

    The plugin boundary of this module, and the only point at which domain
    knowledge enters. The Core contains none.

    Additive by construction
    ------------------------
    An enricher receives an immutable node and returns METADATA ONLY. It is
    never handed a mutable node, and its return value cannot reach text, span,
    block references, or identity.

    This is a structural guarantee rather than a convention: an enricher has
    no mechanism by which to alter what a citation resolves to, so a defective
    or hostile plugin can add nothing worse than unhelpful metadata.

    Values are strings. A structured return type would let a domain object
    enter a domain-agnostic schema, and the contract violation would be
    invisible at the point it occurred.

    Declared as a protocol separate from ``BaseDomainPlugin`` so that adopting
    it breaks no existing plugin. A plugin implements it by choice.
    """

    @property
    def name(self) -> str:
        """
        Stable identifier of this enricher.

        Recorded on failures, and used to configure which enrichers run.

        Returns
        -------
        str
            Implementation identifier.
        """
        ...

    @property
    def metadata_keys(self) -> frozenset[str]:
        """
        Keys this enricher may contribute.

        Declared in advance so that two enrichers claiming the same key are
        detected at configuration rather than by one silently overwriting the
        other partway through a corpus.

        Returns
        -------
        frozenset[str]
            The keys this enricher produces.
        """
        ...

    async def enrich(self, node: KnowledgeNode) -> Mapping[str, str]:
        """
        Produce domain metadata for a node.

        Parameters
        ----------
        node:
            The node to enrich. Immutable.

        Returns
        -------
        Mapping[str, str]
            Metadata to attach. Keys must fall within
            :attr:`metadata_keys`; values must be strings. An empty mapping is
            valid and means the enricher had nothing to contribute.

        Raises
        ------
        EnricherFailedError
            If enrichment failed.
        EnricherContractViolationError
            If the enricher would return something the contract disallows.
        """
        ...


# ============================================================================
# SINK
# ============================================================================


@runtime_checkable
class KnowledgeSink(Protocol):
    """
    Receives knowledge artifacts for persistence.

    The boundary at which this module hands off its output. It is deliberately
    abstract: the module writes artifacts and remains unaware of index
    implementations entirely.

    Why the boundary sits here
    --------------------------
    Vector, lexical, metadata, and graph indexes have four different rebuild
    triggers, none of which is a change to segmentation. Placing their
    construction inside this module would force one lifecycle onto four, and
    would gate every future retrieval strategy on a change to the module that
    produces content rather than the module that queries it.

    Retrieval therefore owns index construction and supplies sinks. This
    module defines the protocol, ships an in-memory implementation for
    testing, and knows nothing further.

    Implementations must tolerate being written to concurrently, since the
    engine processes documents in parallel.
    """

    @property
    def name(self) -> str:
        """
        Stable identifier of this sink.

        Recorded on failures so that one failing sink among several is
        identifiable.

        Returns
        -------
        str
            Implementation identifier.
        """
        ...

    async def write(self, unit: KnowledgeUnit) -> None:
        """
        Persist one knowledge unit.

        A unit carries a node, its optional embedding, and its outgoing
        relations together, so that a sink may write them atomically where its
        storage permits.

        Parameters
        ----------
        unit:
            The artifacts to persist.

        Returns
        -------
        None

        Raises
        ------
        SinkWriteError
            If the sink rejected the write. Deterministic.
        SinkUnavailableError
            If the sink is temporarily unreachable. Transient.
        """
        ...

    async def flush(self) -> None:
        """
        Commit any buffered writes.

        Invoked periodically and before close. A sink buffering internally
        must make its writes durable here, since the engine offers no
        guarantee about when it will next be called.

        Returns
        -------
        None

        Raises
        ------
        SinkWriteError
            If buffered writes could not be committed.
        SinkUnavailableError
            If the sink is temporarily unreachable.
        """
        ...

    async def close(self) -> None:
        """
        Flush and release resources held by this sink.

        Must be idempotent.

        Returns
        -------
        None
        """
        ...


def _assert_protocol_exports(names: Sequence[str]) -> None:
    """
    Verify at import time that every exported name resolves.

    Guards against an entry in ``__all__`` drifting away from an actual
    definition during refactoring, which would otherwise surface as an
    ``AttributeError`` in an unrelated module.

    Parameters
    ----------
    names:
        Exported names to verify.

    Returns
    -------
    None

    Raises
    ------
    RuntimeError
        If an exported name is not defined in this module.
    """
    missing = [name for name in names if name not in globals()]
    if missing:
        raise RuntimeError(
            f"ptfm.core.knowledge.interfaces exports undefined names: {missing}"
        )


_assert_protocol_exports(__all__)
