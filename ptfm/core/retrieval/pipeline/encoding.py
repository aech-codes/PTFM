"""
PTFM Query Preparation Stage

File:
    ptfm/core/retrieval/pipeline/encoding.py

Description:
    Turns a user query into a plan the retrieval arms can execute: normalized
    for the domain, resolved to a compatibility partition, and encoded into a
    vector the partition's stored vectors can be compared against.

The compatibility gate
----------------------
    This stage is where an incompatible query is stopped, and it is stopped
    BEFORE any arm runs.

    That ordering is the whole point. Every failure guarded here is invisible
    downstream: a query encoded for the wrong model, or as a passage when the
    model expects a query, returns plausible results in a worse order and
    reports nothing at all. Detecting it after the fact is impossible, so it
    must be prevented in advance.

    Checks are also ordered cheapest first. An empty query, an oversized
    query, and a partition mismatch are each established before the model is
    invoked, so a query that cannot succeed never costs an inference.

Degradation is bounded
----------------------
    A failure to encode leaves the query without a vector. Dense retrieval is
    then skipped and lexical and structural retrieval continue, which is a
    materially reduced result rather than no result.

    Two conditions are exempt, because they describe the REQUEST rather than
    the system: an empty query and one too large for the model. Degrading
    those would answer a different question from the one asked, so they are
    refused instead.

    A compatibility mismatch is likewise never degraded past. Searching a
    partition with a vector that does not belong to it is not a reduced
    answer, it is a wrong one.

The query side belongs to the provider
--------------------------------------
    This stage never applies a prefix, an instruction, or any other
    transformation. It calls the provider's own query method, because which
    transformation a model expects is the provider's knowledge, and a
    component reconstructing it from configuration would be guessing at
    behaviour that produces no error when guessed wrong.

Dependency contract
-------------------
    Imports only the standard library and the retrieval ``exceptions``,
    ``config``, ``models``, ``interfaces`` and ``partition`` modules, plus the
    knowledge contract layer for the embedding provider. No backend, no
    plugin, no store.
"""

from __future__ import annotations

import time
from typing import Final, Mapping, Optional, Sequence

from ptfm.core.retrieval.config import RetrievalConfig
from ptfm.core.retrieval.exceptions import (
    EmbeddingSpaceMismatchError,
    EmptyQueryError,
    PTFMRetrievalError,
    QueryEncodingError,
    QueryEncodingMismatchError,
    QueryTooLargeError,
)
from ptfm.core.retrieval.interfaces import (
    EmbeddingProvider,
    IdGenerator,
    ModelIdentity,
    QueryEncoder,
    QueryPreprocessor,
    StructuredLogger,
)
from ptfm.core.retrieval.models import (
    EmbeddingPartition,
    QueryPlan,
    RetrievalFilter,
)
from ptfm.core.retrieval.partition import PartitionResolver

__all__ = ["ProviderQueryEncoder", "QueryPreparer"]

#: Longest query excerpt included in a diagnostic, when one is included at all.
#: A query may be long, and a log line holding all of it is both unwieldy and
#: more likely to retain content an operator did not intend to persist.
#:
#: Truncation is a bound on exposure, never a substitute for the decision about
#: whether to expose anything: 80 characters of privileged text is still
#: privileged text. Whether an excerpt is logged is governed by
#: ``TelemetryConfig.log_query_text``, which defaults to off.
_EXCERPT_CHARS: Final[int] = 80


class ProviderQueryEncoder:
    """
    Encodes queries using an embedding provider's own query method.

    Satisfies the ``QueryEncoder`` protocol.

    Compatibility is verified before the provider is invoked. A query encoded
    against the wrong partition is not a slower failure but a silent one, so
    the check must precede the work rather than follow it.

    Parameters
    ----------
    provider:
        The embedding provider. Its ``embed_query`` applies whatever
        transformation the model expects; this class applies none.

    Raises
    ------
    ModelIdentityError
        If the provider cannot report a usable identity.
    """

    __slots__ = ("_provider", "_identity")

    def __init__(self, provider: EmbeddingProvider) -> None:
        self._provider = provider
        # Read once. Identity determines which partitions this encoder may
        # serve, and a provider changing model between queries would let one
        # query search a partition the next could not.
        self._identity = provider.model_identity()

    @property
    def model_identity(self) -> ModelIdentity:
        """
        Report the model this encoder encodes with.

        Returns
        -------
        ModelIdentity
            Identifier, dimension, normalization, and query convention.
        """
        return self._identity

    async def encode(
        self,
        query: str,
        partition: EmbeddingPartition,
    ) -> tuple[float, ...]:
        """
        Encode a query for a given embedding partition.

        Parameters
        ----------
        query:
            The query text, after any domain preprocessing.
        partition:
            The partition the resulting vector will be compared against.

        Returns
        -------
        tuple[float, ...]
            A vector comparable with the partition's stored vectors.

        Raises
        ------
        EmptyQueryError
            If the query carries no searchable content.
        QueryTooLargeError
            If the query exceeds what the model accepts.
        QueryEncodingMismatchError
            If this encoder's query convention differs from the partition's.
        EmbeddingSpaceMismatchError
            If this encoder's model differs from the partition's.
        QueryEncodingError
            If the provider failed to encode.
        """
        self._verify_query(query)
        self._verify_partition(partition)

        try:
            vector = await self._provider.embed_query(query)
        except PTFMRetrievalError:
            raise
        except Exception as exc:
            raise QueryEncodingError(
                f"Provider {self._provider.name!r} failed to encode a query: "
                f"{type(exc).__name__}: {exc}",
                context={"provider_name": self._provider.name},
            ) from exc

        return self._verify_vector(vector, partition)

    # ------------------------------------------------------------------
    # Verification
    # ------------------------------------------------------------------

    def _verify_query(self, query: str) -> None:
        """
        Ensure the query is searchable and within the model's bound.

        Both conditions describe the request rather than the system, so both
        are refused rather than degraded: returning results for an empty query
        would present arbitrary content as relevant, and truncating an
        oversized one would answer a different question from the one asked.

        Parameters
        ----------
        query:
            The query text.

        Returns
        -------
        None

        Raises
        ------
        EmptyQueryError
            If the query carries no searchable content.
        QueryTooLargeError
            If the query exceeds what the model accepts.
        """
        if not query or not query.strip():
            raise EmptyQueryError(
                "The query carries no searchable content. Returning results "
                "for it would present arbitrary content as relevant."
            )

        limit = self._identity.max_input_chars
        if limit is not None and len(query) > limit:
            raise QueryTooLargeError(query_chars=len(query), max_chars=limit)

    def _verify_partition(self, partition: EmbeddingPartition) -> None:
        """
        Ensure this encoder may serve the partition.

        Performed before the provider is invoked, so that a query which cannot
        succeed never costs an inference.

        Parameters
        ----------
        partition:
            The partition the vector will search.

        Returns
        -------
        None

        Raises
        ------
        QueryEncodingMismatchError
            If the query conventions differ.
        EmbeddingSpaceMismatchError
            If the model, dimension, or normalization differs.
        """
        if not partition.accepts_query_encoding(self._identity.query_encoding):
            raise QueryEncodingMismatchError(
                expected=partition.query_encoding,
                actual=self._identity.query_encoding,
                context={"partition_key": partition.key},
            )

        if (
            self._identity.identity != partition.model_identity
            or self._identity.dimension != partition.dimension
            or self._identity.normalized != partition.normalized
        ):
            raise EmbeddingSpaceMismatchError(
                f"This encoder embeds with {self._identity.identity!r} at "
                f"dimension {self._identity.dimension}, while the partition "
                f"holds vectors from {partition.model_identity!r} at "
                f"dimension {partition.dimension}. A query vector from one "
                f"space compared against the other produces a ranking that is "
                f"arithmetically valid and semantically meaningless.",
                expected_identity=partition.model_identity,
                actual_identity=self._identity.identity,
                context={"partition_key": partition.key},
            )

    def _verify_vector(
        self,
        vector: Sequence[float],
        partition: EmbeddingPartition,
    ) -> tuple[float, ...]:
        """
        Ensure the provider returned a vector the partition can compare.

        The provider declared its dimension when its identity was read; this
        checks that it delivered one. A provider that reports one width and
        returns another is not the model it claims to be, and every comparison
        it contributed would be meaningless.

        Parameters
        ----------
        vector:
            The provider's vector.
        partition:
            The partition the vector will search.

        Returns
        -------
        tuple[float, ...]
            The verified vector.

        Raises
        ------
        EmbeddingSpaceMismatchError
            If the width disagrees with the partition.
        """
        if len(vector) != partition.dimension:
            raise EmbeddingSpaceMismatchError(
                f"Provider {self._provider.name!r} returned a query vector of "
                f"width {len(vector)} for a partition holding vectors of "
                f"width {partition.dimension}.",
                expected_identity=partition.model_identity,
                actual_identity=self._identity.identity,
                context={"partition_key": partition.key},
            )

        return tuple(float(value) for value in vector)

    def __repr__(self) -> str:
        """
        Return a representation naming the model in use.

        Returns
        -------
        str
            Diagnostic representation.
        """
        return f"ProviderQueryEncoder(model={self._identity.identity!r})"


class QueryPreparer:
    """
    Builds an executable plan from a user query.

    Sequences the four things that must happen before any arm runs: domain
    normalization, partition resolution, query encoding, and assembly into a
    plan the arms share.

    Stateless. All per-query state is local to a call, so one instance serves
    all concurrent queries.

    Parameters
    ----------
    resolver:
        Resolves the partitions a query will search.
    encoder:
        Produces the query vector. ``None`` where a deployment runs without an
        embedding provider, in which case dense retrieval never runs and
        lexical and structural retrieval serve alone.
    config:
        Supplies result bounds and the degradation policy.
    id_generator:
        Supplies the query correlation identifier. Never used for anything a
        result is identified by.
    logger:
        Structured logger.
    preprocessor:
        Applies domain normalization. ``None`` for a domain with no rules, in
        which case the query is searched as written.
    """

    __slots__ = (
        "_resolver",
        "_encoder",
        "_config",
        "_id_generator",
        "_logger",
        "_preprocessor",
    )

    def __init__(
        self,
        resolver: PartitionResolver,
        encoder: Optional[QueryEncoder],
        config: RetrievalConfig,
        id_generator: IdGenerator,
        logger: StructuredLogger,
        *,
        preprocessor: Optional[QueryPreprocessor] = None,
    ) -> None:
        self._resolver = resolver
        self._encoder = encoder
        self._config = config
        self._id_generator = id_generator
        self._logger = logger
        self._preprocessor = preprocessor

    async def prepare(
        self,
        query: str,
        *,
        top_k: Optional[int] = None,
        filters: Sequence[RetrievalFilter] = (),
        exact: Optional[bool] = None,
    ) -> QueryPlan:
        """
        Prepare a query for execution.

        Parameters
        ----------
        query:
            The query text as supplied by the caller.
        top_k:
            Results requested. The configured default applies when absent.
        filters:
            Structural constraints applied during search.
        exact:
            Whether approximate search is forbidden. The configured default
            applies when absent.

        Returns
        -------
        QueryPlan
            A plan every arm can execute. Its vector is absent when dense
            retrieval is unavailable, which is a reduced plan rather than an
            invalid one.

        Raises
        ------
        EmptyQueryError
            If the query carries no searchable content.
        QueryTooLargeError
            If the query exceeds what the model accepts.
        PartitionNotFoundError
            If no partition can serve the query.
        QueryEncodingMismatchError
            If the encoder's query convention differs from the partition's.
        EmbeddingSpaceMismatchError
            If the encoder's model differs from the partition's.
        QueryEncodingError
            If encoding failed and the configured policy is strict.
        """
        started = time.monotonic()
        query_id = self._id_generator.new_id()
        logger = self._logger.bind(query_id=query_id)

        # Checked before anything else. An empty query costs nothing to
        # reject and would otherwise reach a store as an unbounded scan.
        if not query or not query.strip():
            raise EmptyQueryError(
                "The query carries no searchable content.",
                query_id=query_id,
            )

        text = await self._normalize(query, query_id, logger)
        segmentation = await self._resolver.resolve_for_query()
        embedding, vector = await self._encode(text, segmentation, query_id, logger)

        plan = QueryPlan(
            query_id=query_id,
            text=text,
            vector=vector,
            segmentation=segmentation,
            embedding=embedding,
            top_k=self._config.resolve_top_k(top_k),
            filters=tuple(filters),
            exact=self._config.dense.exact if exact is None else exact,
        )

        logger.debug(
            "retrieval.query.prepared",
            segmentation_key=segmentation.key,
            embedding_key=embedding.key if embedding is not None else None,
            has_vector=plan.has_vector,
            top_k=plan.top_k,
            filters=len(plan.filters),
            exact=plan.exact,
            preparation_ms=(time.monotonic() - started) * 1000.0,
        )

        return plan

    # ------------------------------------------------------------------
    # Stages
    # ------------------------------------------------------------------

    async def _normalize(
        self,
        query: str,
        query_id: str,
        logger: StructuredLogger,
    ) -> str:
        """
        Apply domain normalization, tolerating its absence and its failure.

        A preprocessor failing degrades regardless of the configured policy. A
        query that could not be normalized can still be searched as written,
        which is a worse query rather than no query, and refusing to answer
        because an abbreviation could not be expanded would trade a small loss
        for a total one.

        A preprocessor returning nothing usable is treated as having declined,
        for the same reason.

        Parameters
        ----------
        query:
            The query text.
        query_id:
            Correlation identifier.
        logger:
            Logger bound to this query.

        Returns
        -------
        str
            The normalized query, or the original where normalization was
            absent or failed.
        """
        if self._preprocessor is None:
            return query

        try:
            normalized = await self._preprocessor.preprocess(query)
        except Exception as exc:  # noqa: BLE001 - degrading is proportionate
            logger.warning(
                "retrieval.query.preprocessing_degraded",
                preprocessor=self._preprocessor.name,
                error_type=type(exc).__name__,
                message=str(exc),
            )
            return query

        if not isinstance(normalized, str) or not normalized.strip():
            logger.warning(
                "retrieval.query.preprocessing_declined",
                preprocessor=self._preprocessor.name,
            )
            return query

        return normalized

    async def _encode(
        self,
        text: str,
        segmentation: object,
        query_id: str,
        logger: StructuredLogger,
    ) -> tuple[Optional[EmbeddingPartition], Optional[tuple[float, ...]]]:
        """
        Resolve the embedding partition and encode the query for it.

        Returns a partition and a vector together, or neither. A partition
        without a vector describes a search that cannot run, and a vector
        without its partition could not have been checked for comparability.

        Parameters
        ----------
        text:
            The normalized query.
        segmentation:
            The segmentation partition being searched.
        query_id:
            Correlation identifier.
        logger:
            Logger bound to this query.

        Returns
        -------
        tuple[Optional[EmbeddingPartition], Optional[tuple[float, ...]]]
            The partition and vector, or ``(None, None)`` when dense retrieval
            is unavailable.

        Raises
        ------
        EmptyQueryError
            If the query carries no searchable content.
        QueryTooLargeError
            If the query exceeds what the model accepts.
        QueryEncodingMismatchError
            If the conventions differ.
        EmbeddingSpaceMismatchError
            If the spaces differ.
        QueryEncodingError
            If encoding failed and the configured policy is strict.
        """
        if self._encoder is None:
            # A deployment without an embedding provider. Lexical and
            # structural retrieval serve alone, which is a configuration
            # rather than a fault.
            return None, None

        model = self._encoder.model_identity
        partition = await self._resolver.resolve_embedding_for_query(
            segmentation,  # type: ignore[arg-type]
            model,
        )
        if partition is None:
            logger.debug("retrieval.query.no_embedding_partition")
            return None, None

        try:
            vector = await self._encoder.encode(text, partition)
        except (EmptyQueryError, QueryTooLargeError):
            # Properties of the request. Degrading would answer a different
            # question from the one asked.
            raise
        except (QueryEncodingMismatchError, EmbeddingSpaceMismatchError):
            # Not a reduced answer but a wrong one. Never degraded past.
            raise
        except QueryEncodingError as error:
            if not self._config.degrades():
                raise
            logger.warning(
                "retrieval.query.encoding_degraded",
                error_code=error.error_code,
                message=error.message,
                **self._query_text_fields(text),
            )
            return None, None

        return partition, vector

    def _query_text_fields(self, text: str) -> Mapping[str, object]:
        """
        Build the log fields that would disclose query content, if permitted.

        The one place this module decides whether query text may be written to a
        log, so the decision is auditable by reading a single function rather
        than by checking every call site.

        Governed by ``TelemetryConfig.log_query_text``, which defaults to off.
        A query may carry privileged or regulated material, and an operator who
        leaves that setting at its default is entitled to read it as meaning no
        query text is persisted.

        When suppression applies, the warning still carries a marker saying so.
        A silently absent field would be indistinguishable from a query that had
        no text to show, leaving an operator unable to tell whether redaction
        happened or the diagnostic simply had nothing to report.

        Parameters
        ----------
        text:
            The query text, as prepared.

        Returns
        -------
        Mapping[str, object]
            Either a truncated excerpt, or a marker recording that the excerpt
            was withheld. Never both, and never neither.
        """
        if self._config.telemetry.log_query_text:
            return {"excerpt": text[:_EXCERPT_CHARS]}
        return {"query_text_redacted": True}

    def __repr__(self) -> str:
        """
        Return a representation naming the configured components.

        Returns
        -------
        str
            Diagnostic representation.
        """
        return (
            f"QueryPreparer(encoder="
            f"{'present' if self._encoder is not None else 'absent'}, "
            f"preprocessor="
            f"{self._preprocessor.name if self._preprocessor else 'none'})"
        )
