"""
PTFM PostgreSQL Retrieval Stores

File:
    ptfm/core/retrieval/stores/postgres/stores.py

Description:
    The PostgreSQL implementation of every retrieval store protocol, and the
    backend that owns their shared connection pool.

Why PostgreSQL is the default backend
-------------------------------------
    Not for scale, and not for vector search quality. A dedicated vector engine
    beats it on both.

    It is the default because a knowledge unit's write spans five concerns —
    node, vector, lexical terms, relations, currency — and PostgreSQL can commit
    all five in one transaction. Across separate services that write cannot be
    atomic, and a crash partway through leaves a node that one index will return
    and no store can attest to. An evidence system's worst available failure is
    a result that looks valid and cannot be traced, so the ability to make that
    state unreachable outranks throughput.

    ``PostgresIndexWriter.is_atomic`` is therefore ``True``, and the claim is
    backed by one real transaction rather than by a sequence of writes that
    usually succeed together.

Where this backend differs from the in-memory reference
------------------------------------------------------
    The in-memory stores define what the protocols MEAN, and this backend agrees
    with them on every question of semantics: which nodes a filter selects, what
    provenance a node carries, what currency states are distinguishable, what a
    partition boundary forbids. Those agreements are established by differential
    test rather than by inspection.

    Lexical RANKING is different, and deliberately so. The in-memory index
    implements Okapi BM25; PostgreSQL scores with ``ts_rank_cd``, which weighs
    term proximity and has no document-length normalization term. The two order
    the same matching nodes differently.

    This is permitted, and the reasons are worth stating rather than assuming.
    A lexical signal is declared uncalibrated and carries ``SignalScale.LEXICAL``
    precisely so that nothing downstream may read meaning into its magnitude.
    Fusion is rank-only, so no arithmetic ever combines a BM25 value with a
    ``ts_rank_cd`` value. And one lexical arm runs per query, so two lexical
    scales never meet in one result set. What a deployment must understand is
    that switching backends changes lexical result ORDER; it does not change
    what any score is taken to mean, because nothing takes it to mean anything.

    MATCHING is aligned on purpose. Both backends tokenize with the shared term
    rule in ``stores.tokenization``, and here the tokens are combined with
    tsquery OR, so a node matches when it shares any term. Using ``plainto_tsquery`` would
    have conjoined the terms and selected a strictly smaller set than the
    reference, which is a semantic difference rather than a ranking one and
    would not have been acceptable.

    Approximate vector search, when enabled, also returns a different neighbour
    set than exact search. That is what approximate means, and it is why exact
    search remains available unconditionally.

    Vector PRECISION differs too, because pgvector stores single-precision
    floats. ``PostgresVectorIndex`` documents what follows from that.

Dependency contract
-------------------
    Requires ``psycopg`` and ``psycopg_pool``, which is why this backend is an
    optional extra rather than a core dependency. Nothing in the retrieval
    pipeline imports this module; it is selected by the composition root, so a
    deployment that installs neither package still has a working engine over the
    in-memory stores.

    Credentials arrive as an injected connection string and are never recorded
    in a representation, a log message, or an exception. Every store's
    ``__repr__`` below names the backend and nothing else for that reason.
"""

from __future__ import annotations

import json
from types import TracebackType
from typing import Any, Final, Mapping, Optional, Sequence

from psycopg import errors, sql
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool

from ptfm.core.knowledge.models import KnowledgeUnit
from ptfm.core.retrieval.exceptions import (
    EmbeddingSpaceMismatchError,
    IndexWriteError,
    PartialIndexWriteError,
    PartitionError,
    RetrievalConfigurationError,
    SegmentationMismatchError,
    StoreUnavailableError,
)
from ptfm.core.retrieval.models import (
    Candidate,
    EmbeddingPartition,
    RetrievalFilter,
    SegmentationPartition,
)
from ptfm.core.retrieval.partition import embedding_for_vector
from ptfm.core.retrieval.stores.postgres.filters import FilterTranslator
from ptfm.core.retrieval.stores.postgres.schema import (
    CREATE_STATEMENTS,
    REQUIRED_EXTENSIONS,
    SCHEMA_VERSION,
    ann_index_statement,
)
from ptfm.core.retrieval.stores.tokenization import tokenize
from ptfm.core.schemas import (
    DiscoveryMethod,
    KnowledgeNode,
    NodeEmbedding,
    NodeRelation,
    NodeType,
    SignalScale,
    SourceDescriptor,
    TextSpan,
)

__all__ = [
    "PostgresBackend",
    "PostgresNodeStore",
    "PostgresVectorIndex",
    "PostgresLexicalIndex",
    "PostgresRelationStore",
    "PostgresSupersessionStore",
    "PostgresPartitionCatalog",
    "PostgresIndexWriter",
]


#: The concerns one flush transaction spans. Reported together on an
#: indeterminate failure, because a single transaction covers all of them and
#: so an unknown outcome is unknown for all of them equally.
_WRITTEN_STORES: Final[tuple[str, ...]] = (
    "ptfm_node",
    "ptfm_vector",
    "ptfm_relation",
    "ptfm_supersession",
)

#: Columns needed to reconstruct a node with its provenance intact. Named
#: explicitly rather than selected with a wildcard, so that adding a column to
#: the schema cannot silently change what a reconstruction reads.
_NODE_COLUMNS: Final[str] = (
    "node_id, document_id, ordinal, span_start, span_end, node_text, "
    "node_type, block_ids, heading_path, page_numbers, node_metadata, "
    "source, segmentation_fingerprint, source_text_hash, "
    "source_schema_version, processed_at"
)



class PostgresBackend:
    """
    Owner of the connection pool and the schema, and factory for the stores.

    One backend per database. The stores share its pool rather than opening
    connections of their own, which is what makes a single transaction across
    all of them possible.

    Attributes
    ----------
    pool:
        The connection pool. Opened by ``open`` and closed by ``close``.
    """

    __slots__ = ("_pool", "_conninfo", "_min_size", "_max_size", "_build_ann")

    def __init__(
        self,
        conninfo: str,
        *,
        min_size: int = 1,
        max_size: int = 8,
        build_ann_indexes: bool = False,
    ) -> None:
        """
        Configure a backend without connecting.

        Parameters
        ----------
        conninfo:
            A libpq connection string. Injected, and never recorded in a
            representation or an exception, because it carries credentials.
        min_size:
            Connections held open when idle.
        max_size:
            Maximum concurrent connections. Bounds the database's exposure to a
            burst of queries, which matters because a hybrid query opens several
            arms at once.
        build_ann_indexes:
            Whether registering an embedding partition also builds an
            approximate index for it.

            Off by default. An approximate index makes results depend on
            insertion order and index construction, which a caller auditing a
            past answer cannot reproduce. A deployment that wants the latency
            enables it knowingly, and exact search stays available either way.

        Raises
        ------
        RetrievalConfigurationError
            If the pool bounds are not a usable range.
        """
        if min_size < 0 or max_size < 1 or min_size > max_size:
            raise RetrievalConfigurationError(
                f"Pool bounds min_size={min_size} max_size={max_size} do not "
                f"describe a usable range."
            )

        self._conninfo = conninfo
        self._min_size = min_size
        self._max_size = max_size
        self._build_ann = build_ann_indexes
        self._pool: Optional[AsyncConnectionPool] = None

    @property
    def name(self) -> str:
        """
        Identify the backend in diagnostics.

        Present because ``StoreUnavailableError`` requires a store to name
        itself: a degraded arm is reported by name, and an unnamed one leaves a
        caller unable to tell which store failed.

        Returns
        -------
        str
            A stable identifier.
        """
        return "postgres:PostgresBackend"

    @property
    def pool(self) -> AsyncConnectionPool:
        """
        The open connection pool.

        Returns
        -------
        AsyncConnectionPool
            The pool.

        Raises
        ------
        StoreUnavailableError
            If the backend has not been opened. Raised rather than opening
            lazily, because a pool opened inside a query inherits that query's
            cancellation and would be torn down unpredictably.
        """
        if self._pool is None:
            raise StoreUnavailableError(
                "The PostgreSQL backend is not open. Call open() before using "
                "its stores, or use the backend as an async context manager.",
                store_name=self.name,
            )
        return self._pool

    @property
    def builds_ann_indexes(self) -> bool:
        """
        Whether registration builds approximate indexes.

        Returns
        -------
        bool
            ``True`` when approximate indexes are built.
        """
        return self._build_ann

    async def open(self) -> None:
        """
        Open the pool and verify the database can serve this backend.

        Verification happens here rather than at first query, so that a missing
        extension is a startup failure with a clear cause rather than an arm
        failure attributed to retrieval.

        Raises
        ------
        StoreUnavailableError
            If the database cannot be reached, or lacks a required extension.
        """
        pool = AsyncConnectionPool(
            self._conninfo,
            min_size=self._min_size,
            max_size=self._max_size,
            open=False,
        )
        try:
            await pool.open(wait=True, timeout=30.0)
        except Exception as error:
            await pool.close()
            raise StoreUnavailableError(
                f"Could not open a connection pool to the retrieval database: "
                f"{type(error).__name__}.",
                store_name=self.name,
            ) from error

        self._pool = pool
        try:
            await self._verify_extensions()
        except Exception:
            await self.close()
            raise

    async def close(self) -> None:
        """
        Close the pool.

        Idempotent, so a composition root may close a backend it is unsure
        about.
        """
        if self._pool is not None:
            await self._pool.close()
            self._pool = None

    async def __aenter__(self) -> PostgresBackend:
        """
        Open the backend.

        Returns
        -------
        PostgresBackend
            This backend, opened.
        """
        await self.open()
        return self

    async def __aexit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc: Optional[BaseException],
        traceback: Optional[TracebackType],
    ) -> None:
        """
        Close the backend.

        Parameters
        ----------
        exc_type:
            Exception type, if the block raised.
        exc:
            Exception instance, if the block raised.
        traceback:
            Traceback, if the block raised.
        """
        await self.close()

    async def _verify_extensions(self) -> None:
        """
        Confirm every required extension is installed.

        Raises
        ------
        StoreUnavailableError
            If an extension is missing.
        """
        async with self.pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "SELECT extname FROM pg_extension WHERE extname = ANY(%s)",
                    (list(REQUIRED_EXTENSIONS),),
                )
                installed = {row[0] for row in await cur.fetchall()}

        missing = tuple(
            name for name in REQUIRED_EXTENSIONS if name not in installed
        )
        if missing:
            raise StoreUnavailableError(
                f"The retrieval database is missing required extension(s): "
                f"{', '.join(missing)}. Install them with CREATE EXTENSION "
                f"before starting the engine; the engine does not create "
                f"extensions because doing so needs privileges it should not "
                f"hold at query time.",
                store_name=self.name,
            )

    async def apply_schema(self) -> None:
        """
        Create the schema if it is absent.

        Every statement is idempotent, so applying this to an existing database
        is a no-op. Separated from ``open`` because a deployment may run
        migrations under a privileged role and queries under an unprivileged
        one.

        Raises
        ------
        StoreUnavailableError
            If the schema could not be applied.
        """
        try:
            async with self.pool.connection() as conn:
                async with conn.transaction():
                    async with conn.cursor() as cur:
                        for statement in CREATE_STATEMENTS:
                            await cur.execute(statement)
        except Exception as error:
            raise StoreUnavailableError(
                f"Could not apply retrieval schema version {SCHEMA_VERSION}: "
                f"{type(error).__name__}.",
                store_name=self.name,
            ) from error

    async def ensure_ann_index(self, partition: EmbeddingPartition) -> None:
        """
        Build the approximate index for one embedding partition.

        Called when a partition is registered, and only when the backend was
        configured to build them. Index construction is not run inside a
        transaction with anything else: it is slow, and holding a write
        transaction open for its duration would block the writes it exists to
        accelerate.

        Parameters
        ----------
        partition:
            The embedding partition to index.

        Raises
        ------
        StoreUnavailableError
            If the index could not be created.
        """
        statement = ann_index_statement(partition.key, partition.dimension)
        try:
            async with self.pool.connection() as conn:
                await conn.set_autocommit(True)
                async with conn.cursor() as cur:
                    await cur.execute(statement)
        except Exception as error:
            raise StoreUnavailableError(
                f"Could not create an approximate index for embedding "
                f"partition {partition}: {type(error).__name__}.",
                store_name=self.name,
            ) from error

    def node_store(self) -> PostgresNodeStore:
        """
        Build the node store.

        Returns
        -------
        PostgresNodeStore
            A store over this backend's pool.
        """
        return PostgresNodeStore(self)

    def vector_index(self) -> PostgresVectorIndex:
        """
        Build the vector index.

        Returns
        -------
        PostgresVectorIndex
            An index over this backend's pool.
        """
        return PostgresVectorIndex(self)

    def lexical_index(self) -> PostgresLexicalIndex:
        """
        Build the lexical index.

        Returns
        -------
        PostgresLexicalIndex
            An index over this backend's pool.
        """
        return PostgresLexicalIndex(self)

    def relation_store(self) -> PostgresRelationStore:
        """
        Build the relation store.

        Returns
        -------
        PostgresRelationStore
            A store over this backend's pool.
        """
        return PostgresRelationStore(self)

    def supersession_store(self) -> PostgresSupersessionStore:
        """
        Build the supersession store.

        Returns
        -------
        PostgresSupersessionStore
            A store over this backend's pool.
        """
        return PostgresSupersessionStore(self)

    def partition_catalog(self) -> PostgresPartitionCatalog:
        """
        Build the partition catalog.

        Returns
        -------
        PostgresPartitionCatalog
            A catalog over this backend's pool.
        """
        return PostgresPartitionCatalog(self)

    def index_writer(self) -> PostgresIndexWriter:
        """
        Build an atomic index writer.

        Returns
        -------
        PostgresIndexWriter
            A writer over this backend's pool.
        """
        return PostgresIndexWriter(self)

    def __repr__(self) -> str:
        """
        Describe the backend without disclosing its connection string.

        Returns
        -------
        str
            A representation naming the pool state only.
        """
        state = "open" if self._pool is not None else "closed"
        return f"PostgresBackend({state}, max_size={self._max_size})"


class _PostgresStore:
    """
    Shared behaviour for stores over a single backend.

    Holds the backend rather than a connection, so that every operation takes a
    connection from the pool for its duration and returns it. A store holding a
    connection would serialize the arms that a hybrid query runs concurrently.

    Attributes
    ----------
    _backend:
        The owning backend.
    _translator:
        Filter translator, shared because it is stateless.
    """

    __slots__ = ("_backend", "_translator")

    def __init__(self, backend: PostgresBackend) -> None:
        """
        Bind the store to a backend.

        Parameters
        ----------
        backend:
            The backend whose pool this store uses.
        """
        self._backend = backend
        self._translator = FilterTranslator()

    @property
    def name(self) -> str:
        """
        Identify the store in diagnostics.

        Returns
        -------
        str
            A stable identifier.
        """
        return f"postgres:{type(self).__name__}"

    def __repr__(self) -> str:
        """
        Describe the store.

        Returns
        -------
        str
            A representation naming the store and nothing about its credentials.
        """
        return f"{type(self).__name__}()"


class PostgresNodeStore(_PostgresStore):
    """
    Authoritative node storage.

    The only source of node text and provenance in this backend. Retrieval arms
    return identifiers; this store returns what those identifiers denote, which
    is the separation that keeps an index's stale copy of a passage from
    becoming evidence.
    """

    __slots__ = ()

    async def get_many(
        self,
        node_ids: Sequence[str],
        partition: SegmentationPartition,
    ) -> Mapping[str, KnowledgeNode]:
        """
        Fetch nodes by identifier within one segmentation partition.

        Returns only the nodes that exist. A caller distinguishes a missing node
        by its absence from the mapping, which is how hydration turns a stale
        index entry into a reported gap rather than a silent omission.

        Parameters
        ----------
        node_ids:
            Identifiers to fetch. Duplicates are harmless.
        partition:
            The segmentation partition to read from.

        Returns
        -------
        Mapping[str, KnowledgeNode]
            Nodes by identifier.

        Raises
        ------
        StoreUnavailableError
            If the store could not be read.
        """
        if not node_ids:
            return {}

        rows = await self._fetch(
            sql.SQL(
                "SELECT " + _NODE_COLUMNS + " FROM ptfm_node "
                "WHERE partition_key = %s AND node_id = ANY(%s)"
            ),
            (partition.key, list(dict.fromkeys(node_ids))),
        )
        return {
            row["node_id"]: _node_from_row(row, self.name) for row in rows
        }

    async def get_by_ordinal(
        self,
        document_id: str,
        ordinals: Sequence[int],
        partition: SegmentationPartition,
    ) -> Mapping[int, KnowledgeNode]:
        """
        Fetch nodes by position within a document.

        How context expansion finds a node's neighbours. The unique index on
        position guarantees at most one node per ordinal, so the mapping cannot
        lose a node to a collision.

        Parameters
        ----------
        document_id:
            The document to read from.
        ordinals:
            Positions to fetch. Positions with no node are absent from the
            result.
        partition:
            The segmentation partition to read from.

        Returns
        -------
        Mapping[int, KnowledgeNode]
            Nodes by position.

        Raises
        ------
        StoreUnavailableError
            If the store could not be read.
        """
        wanted = [ordinal for ordinal in dict.fromkeys(ordinals) if ordinal >= 0]
        if not wanted:
            return {}

        rows = await self._fetch(
            sql.SQL(
                "SELECT " + _NODE_COLUMNS + " FROM ptfm_node "
                "WHERE partition_key = %s AND document_id = %s "
                "AND ordinal = ANY(%s)"
            ),
            (partition.key, document_id, wanted),
        )
        return {
            int(row["ordinal"]): _node_from_row(row, self.name) for row in rows
        }

    async def put_many(
        self,
        nodes: Sequence[KnowledgeNode],
        partition: SegmentationPartition,
    ) -> None:
        """
        Store nodes, replacing any already recorded under the same identifiers.

        Replacement rather than rejection, so reprocessing a document is
        idempotent. Nothing is deleted: a node absent from a reprocessed
        document keeps its row, because a citation recorded against it must
        still resolve.

        Parameters
        ----------
        nodes:
            Nodes to store.
        partition:
            The segmentation partition to write into.

        Raises
        ------
        SegmentationMismatchError
            If a node does not belong to the named partition. Refused rather
            than accepted, because a node stored under a partition whose
            segmentation differs from its own would be returned for spans the
            partition does not describe.
        IndexWriteError
            If the write failed.
        """
        if not nodes:
            return

        _require_partition_match(nodes, partition)
        try:
            async with self._backend.pool.connection() as conn:
                async with conn.transaction():
                    async with conn.cursor() as cur:
                        await _write_nodes(cur, nodes, partition)
        except PartitionError:
            raise
        except Exception as error:
            raise IndexWriteError(
                f"Could not store {len(nodes)} node(s): "
                f"{type(error).__name__}."
            ) from error

    async def count(self, partition: SegmentationPartition) -> int:
        """
        Count stored nodes in a partition.

        Parameters
        ----------
        partition:
            The segmentation partition to count.

        Returns
        -------
        int
            The node count.

        Raises
        ------
        StoreUnavailableError
            If the store could not be read.
        """
        rows = await self._fetch(
            sql.SQL("SELECT count(*) AS total FROM ptfm_node "
                    "WHERE partition_key = %s"),
            (partition.key,),
        )
        return int(rows[0]["total"])

    async def _fetch(
        self,
        query: sql.Composable,
        params: tuple[Any, ...],
    ) -> list[dict[str, Any]]:
        """
        Run a query returning dictionary rows.

        Parameters
        ----------
        query:
            The composed query.
        params:
            Bound parameters.

        Returns
        -------
        list[dict[str, Any]]
            The rows.

        Raises
        ------
        StoreUnavailableError
            If the query failed.
        """
        try:
            async with self._backend.pool.connection() as conn:
                async with conn.cursor(row_factory=dict_row) as cur:
                    await cur.execute(query, params)
                    return list(await cur.fetchall())
        except Exception as error:
            raise StoreUnavailableError(
                f"Node store query failed: {type(error).__name__}.",
                store_name=self.name,
            ) from error


class PostgresVectorIndex(_PostgresStore):
    """
    Dense vector search over pgvector.

    Keyed by embedding partition, so a query cannot reach vectors from a model
    it was not encoded by. The similarity returned is cosine similarity derived
    from pgvector's cosine distance, reported under ``SignalScale.COSINE`` and
    uncalibrated: it says which candidates are nearer, not how likely any of
    them is to be true.

    Stored precision
    ----------------
    pgvector's ``vector`` type holds single-precision floats. A vector written
    from Python doubles is therefore read back rounded to float32, and a
    similarity computed here differs from one computed over the unrounded
    vectors by roughly 1e-7 or less.

    This is inherent to the type rather than a choice made here: pgvector offers
    no double-precision vector, and ``halfvec`` would be coarser still.

    The consequence worth knowing is not the magnitude, which nothing
    downstream reads — scores are uncalibrated and fusion is rank-only. It is
    that two candidates whose true similarities differ by less than that margin
    may be ORDERED differently here than by a backend that keeps full
    precision. Ties break on identifier, so each backend remains internally
    reproducible; the orders they produce are not guaranteed identical.
    """

    __slots__ = ()

    @property
    def supports_exact_search(self) -> bool:
        """
        Whether exact search is available.

        Always ``True``. Exact search is a sequential scan with a distance
        computation, which needs no index and therefore cannot be unavailable.
        This matters because a caller auditing a past answer needs a
        reproducible search, and an approximate index does not provide one.

        Returns
        -------
        bool
            ``True``.
        """
        return True

    async def search(
        self,
        vector: Sequence[float],
        partition: EmbeddingPartition,
        top_k: int,
        *,
        filters: Sequence[RetrievalFilter] = (),
        exact: bool = False,
    ) -> Sequence[Candidate]:
        """
        Find the nearest vectors in an embedding partition.

        Filters constrain node attributes, so the vector table is joined to the
        node table. Filtering during the search rather than after it keeps a
        filtered query returning the number of results asked for.

        Parameters
        ----------
        vector:
            The query vector, encoded by the partition's model.
        partition:
            The embedding partition to search.
        top_k:
            Maximum candidates to return.
        filters:
            Structural constraints.
        exact:
            Whether to force an exhaustive scan. When ``True``, index scans are
            disabled for the statement so that an approximate index present on
            the partition cannot change the result.

        Returns
        -------
        Sequence[Candidate]
            Candidates ranked nearest first, with cosine similarity as the
            score. Ranks are contiguous from zero.

        Raises
        ------
        EmbeddingSpaceMismatchError
            If the query vector's width differs from the partition's. Refused
            rather than compared, because a distance between vectors of
            different width is not a distance.
        StoreUnavailableError
            If the search failed.
        """
        if top_k < 1:
            return ()

        if len(vector) != partition.dimension:
            raise EmbeddingSpaceMismatchError(
                f"Query vector has width {len(vector)} but partition "
                f"{partition.key!r} holds vectors of width "
                f"{partition.dimension}. A similarity between vectors of "
                f"different width is undefined, so the search is refused "
                f"rather than approximated.",
                expected_identity=partition.model_identity,
                context={"partition_key": partition.key},
            )

        condition = self._translator.translate_all(
            tuple(filters), column_prefix="nd"
        )
        cast = sql.SQL("::vector({})").format(
            sql.Literal(partition.dimension)
        )
        query = sql.SQL(
            "SELECT vc.node_id AS node_id, "
            "       1 - (vc.embedding{cast} <=> {placeholder}{cast}) AS sim "
            "FROM ptfm_vector AS vc "
            "JOIN ptfm_node AS nd "
            "  ON nd.partition_key = {seg_key} AND nd.node_id = vc.node_id "
            "WHERE vc.partition_key = {emb_key} "
            "  AND vc.dimension = {dimension} "
            "  AND ({condition}) "
            "ORDER BY vc.embedding{cast} <=> {placeholder}{cast}, vc.node_id "
            "LIMIT {limit}"
        ).format(
            cast=cast,
            placeholder=sql.Placeholder(),
            seg_key=sql.Placeholder(),
            emb_key=sql.Placeholder(),
            dimension=sql.Placeholder(),
            condition=condition.expression,
            limit=sql.Placeholder(),
        )

        values = list(vector)
        params = (
            values,
            partition.segmentation.key,
            partition.key,
            partition.dimension,
            *condition.params,
            values,
            top_k,
        )

        try:
            async with self._backend.pool.connection() as conn:
                async with conn.transaction():
                    async with conn.cursor(row_factory=dict_row) as cur:
                        if exact:
                            # LOCAL, so the setting reverts with the
                            # transaction and cannot leak into another query
                            # that shares the pooled connection.
                            await cur.execute(
                                "SET LOCAL enable_indexscan = off"
                            )
                            await cur.execute(
                                "SET LOCAL enable_indexonlyscan = off"
                            )
                        await cur.execute(query, params)
                        rows = list(await cur.fetchall())
        except Exception as error:
            raise StoreUnavailableError(
                f"Vector search failed in embedding partition {partition}: "
                f"{type(error).__name__}.",
                store_name=self.name,
            ) from error

        return tuple(
            Candidate(
                node_id=row["node_id"],
                method=DiscoveryMethod.DENSE,
                rank=rank,
                score=float(row["sim"]),
                scale=SignalScale.COSINE,
            )
            for rank, row in enumerate(rows)
        )

    async def upsert(
        self,
        embeddings: Sequence[NodeEmbedding],
        partition: EmbeddingPartition,
    ) -> None:
        """
        Store vectors, replacing any already recorded for the same nodes.

        Parameters
        ----------
        embeddings:
            Embeddings to store.
        partition:
            The embedding partition to write into.

        Raises
        ------
        EmbeddingSpaceMismatchError
            If an embedding does not describe the partition's space.
        IndexWriteError
            If the write failed.
        """
        if not embeddings:
            return

        _require_embedding_match(embeddings, partition)
        try:
            async with self._backend.pool.connection() as conn:
                async with conn.transaction():
                    async with conn.cursor() as cur:
                        await _write_vectors(cur, embeddings, partition)
        except PartitionError:
            raise
        except Exception as error:
            raise IndexWriteError(
                f"Could not store {len(embeddings)} embedding(s): "
                f"{type(error).__name__}."
            ) from error

    async def count(self, partition: EmbeddingPartition) -> int:
        """
        Count stored vectors in an embedding partition.

        Parameters
        ----------
        partition:
            The embedding partition to count.

        Returns
        -------
        int
            The vector count.

        Raises
        ------
        StoreUnavailableError
            If the count failed.
        """
        try:
            async with self._backend.pool.connection() as conn:
                async with conn.cursor() as cur:
                    await cur.execute(
                        "SELECT count(*) FROM ptfm_vector "
                        "WHERE partition_key = %s",
                        (partition.key,),
                    )
                    row = await cur.fetchone()
        except Exception as error:
            raise StoreUnavailableError(
                f"Vector count failed: {type(error).__name__}.",
                store_name=self.name,
            ) from error
        return int(row[0]) if row else 0


class PostgresLexicalIndex(_PostgresStore):
    """
    Term search over PostgreSQL full text search.

    Matching is aligned with the in-memory index: the query is split by the same
    tokenizer and the terms are combined disjunctively, so a node matches when
    it shares any term. Ranking is not aligned, and the module docstring explains
    why that is acceptable and what a deployment should expect from it.
    """

    __slots__ = ()

    async def search(
        self,
        query: str,
        partition: SegmentationPartition,
        top_k: int,
        *,
        filters: Sequence[RetrievalFilter] = (),
    ) -> Sequence[Candidate]:
        """
        Find nodes sharing terms with the query.

        The tsquery is composed from bound parameters, one per term, combined
        with tsquery OR. Composing it from the query text would put caller input
        where the database parses an expression; binding each term individually
        keeps the structure in this module and the values in parameters.

        ``plainto_tsquery`` is used per term rather than ``to_tsquery`` because
        it never raises on unexpected input: a term consisting only of
        punctuation yields an empty tsquery, which contributes nothing, rather
        than a syntax error that would fail the arm.

        Parameters
        ----------
        query:
            The query text.
        partition:
            The segmentation partition to search.
        top_k:
            Maximum candidates to return.
        filters:
            Structural constraints.

        Returns
        -------
        Sequence[Candidate]
            Candidates ranked most relevant first, scored under
            ``SignalScale.LEXICAL``. Empty when the query has no terms, which is
            a query that asks for nothing rather than an error.

        Raises
        ------
        StoreUnavailableError
            If the search failed.
        """
        if top_k < 1:
            return ()

        terms = tokenize(query)
        if not terms:
            return ()

        tsquery = sql.SQL(" || ").join(
            sql.SQL("plainto_tsquery('simple', {})").format(sql.Placeholder())
            for _ in terms
        )
        condition = self._translator.translate_all(tuple(filters))

        composed = sql.SQL(
            "WITH q AS (SELECT ({tsquery}) AS tsq) "
            "SELECT node_id, ts_rank_cd(search_vector, q.tsq) AS relevance "
            "FROM ptfm_node, q "
            "WHERE partition_key = {partition} "
            "  AND search_vector @@ q.tsq "
            "  AND ({condition}) "
            "ORDER BY relevance DESC, node_id "
            "LIMIT {limit}"
        ).format(
            tsquery=tsquery,
            partition=sql.Placeholder(),
            condition=condition.expression,
            limit=sql.Placeholder(),
        )

        params = (*terms, partition.key, *condition.params, top_k)

        try:
            async with self._backend.pool.connection() as conn:
                async with conn.cursor(row_factory=dict_row) as cur:
                    await cur.execute(composed, params)
                    rows = list(await cur.fetchall())
        except Exception as error:
            raise StoreUnavailableError(
                f"Lexical search failed in partition {partition}: "
                f"{type(error).__name__}.",
                store_name=self.name,
            ) from error

        return tuple(
            Candidate(
                node_id=row["node_id"],
                method=DiscoveryMethod.SPARSE,
                rank=rank,
                score=float(row["relevance"]),
                scale=SignalScale.LEXICAL,
            )
            for rank, row in enumerate(rows)
        )

    async def index_nodes(
        self,
        nodes: Sequence[KnowledgeNode],
        partition: SegmentationPartition,
    ) -> None:
        """
        Make nodes searchable by term.

        Writes the nodes, because the searchable form is a generated column on
        the node row rather than a separate structure. A node and its terms are
        therefore written by one statement and cannot disagree.

        Calling this after ``PostgresNodeStore.put_many`` for the same nodes is
        harmless: the write is idempotent.

        Parameters
        ----------
        nodes:
            Nodes to index.
        partition:
            The segmentation partition to write into.

        Raises
        ------
        SegmentationMismatchError
            If a node does not belong to the named partition.
        IndexWriteError
            If the write failed.
        """
        if not nodes:
            return

        _require_partition_match(nodes, partition)
        try:
            async with self._backend.pool.connection() as conn:
                async with conn.transaction():
                    async with conn.cursor() as cur:
                        await _write_nodes(cur, nodes, partition)
        except PartitionError:
            raise
        except Exception as error:
            raise IndexWriteError(
                f"Could not index {len(nodes)} node(s) for lexical search: "
                f"{type(error).__name__}."
            ) from error


class PostgresRelationStore(_PostgresStore):
    """
    Structural relations between nodes.

    Serves two purposes that must not be confused: traversal from known nodes,
    and structural selection with no query vector at all. Neither ranks by
    relevance, so both report ``SignalScale.RANK_ONLY``: a structural match is
    a match, not a degree of one.
    """

    __slots__ = ()

    async def relations_from(
        self,
        node_ids: Sequence[str],
        partition: SegmentationPartition,
        *,
        relation_types: Sequence[str] = (),
    ) -> Mapping[str, Sequence[NodeRelation]]:
        """
        Fetch relations originating at the given nodes.

        Parameters
        ----------
        node_ids:
            Source nodes.
        partition:
            The segmentation partition to read from.
        relation_types:
            Relation types to include. Empty includes all.

        Returns
        -------
        Mapping[str, Sequence[NodeRelation]]
            Relations by source node, deterministically ordered so that a
            traversal is reproducible. Nodes with no relations are absent.

        Raises
        ------
        StoreUnavailableError
            If the read failed.
        """
        if not node_ids:
            return {}

        clauses = [
            sql.SQL("partition_key = {}").format(sql.Placeholder()),
            sql.SQL("source_node_id = ANY({})").format(sql.Placeholder()),
        ]
        params: list[Any] = [
            partition.key,
            list(dict.fromkeys(node_ids)),
        ]
        if relation_types:
            clauses.append(
                sql.SQL("relation_type = ANY({}::text[])").format(
                    sql.Placeholder()
                )
            )
            params.append([str(kind) for kind in relation_types])

        query = sql.SQL(
            "SELECT source_node_id, target_node_id, relation_type "
            "FROM ptfm_relation WHERE {where} "
            "ORDER BY source_node_id, relation_type, target_node_id"
        ).format(where=sql.SQL(" AND ").join(clauses))

        try:
            async with self._backend.pool.connection() as conn:
                async with conn.cursor(row_factory=dict_row) as cur:
                    await cur.execute(query, tuple(params))
                    rows = list(await cur.fetchall())
        except Exception as error:
            raise StoreUnavailableError(
                f"Relation lookup failed: {type(error).__name__}.",
                store_name=self.name,
            ) from error

        grouped: dict[str, list[NodeRelation]] = {}
        for row in rows:
            grouped.setdefault(row["source_node_id"], []).append(
                NodeRelation(
                    source_node_id=row["source_node_id"],
                    target_node_id=row["target_node_id"],
                    relation_type=row["relation_type"],
                )
            )
        return {key: tuple(value) for key, value in grouped.items()}

    async def structural_search(
        self,
        filters: Sequence[RetrievalFilter],
        partition: SegmentationPartition,
        top_k: int,
    ) -> Sequence[Candidate]:
        """
        Select nodes by structure alone.

        Returns nothing when given no filters. An unconstrained structural
        search would return an arbitrary slice of the corpus, and arbitrary
        content presented as retrieved evidence is exactly what this system
        exists to prevent.

        Ordered by document and position, which is a reproducible order and
        carries no implication of relevance.

        Parameters
        ----------
        filters:
            Structural constraints. Empty yields no candidates.
        partition:
            The segmentation partition to search.
        top_k:
            Maximum candidates to return.

        Returns
        -------
        Sequence[Candidate]
            Candidates with no score, under ``SignalScale.RANK_ONLY``.

        Raises
        ------
        StoreUnavailableError
            If the search failed.
        """
        if top_k < 1 or not filters:
            return ()

        condition = self._translator.translate_all(tuple(filters))
        query = sql.SQL(
            "SELECT node_id FROM ptfm_node "
            "WHERE partition_key = {partition} AND ({condition}) "
            "ORDER BY document_id, ordinal, node_id "
            "LIMIT {limit}"
        ).format(
            partition=sql.Placeholder(),
            condition=condition.expression,
            limit=sql.Placeholder(),
        )

        try:
            async with self._backend.pool.connection() as conn:
                async with conn.cursor() as cur:
                    await cur.execute(
                        query,
                        (partition.key, *condition.params, top_k),
                    )
                    rows = list(await cur.fetchall())
        except Exception as error:
            raise StoreUnavailableError(
                f"Structural search failed: {type(error).__name__}.",
                store_name=self.name,
            ) from error

        return tuple(
            Candidate(
                node_id=row[0],
                method=DiscoveryMethod.GRAPH,
                rank=rank,
                score=None,
                scale=SignalScale.RANK_ONLY,
            )
            for rank, row in enumerate(rows)
        )

    async def put_many(
        self,
        relations: Sequence[NodeRelation],
        partition: SegmentationPartition,
    ) -> None:
        """
        Store relations, ignoring any already recorded.

        Parameters
        ----------
        relations:
            Relations to store.
        partition:
            The segmentation partition to write into.

        Raises
        ------
        IndexWriteError
            If the write failed.
        """
        if not relations:
            return

        try:
            async with self._backend.pool.connection() as conn:
                async with conn.transaction():
                    async with conn.cursor() as cur:
                        await _write_relations(cur, relations, partition)
        except Exception as error:
            raise IndexWriteError(
                f"Could not store {len(relations)} relation(s): "
                f"{type(error).__name__}."
            ) from error


class PostgresSupersessionStore(_PostgresStore):
    """
    Which document currently represents each source.

    Distinguishes three states, because collapsing any two of them loses
    information a caller needs: a source with a recorded current document, a
    source recorded as withdrawn, and a source with no record at all. The
    middle state is a row whose document is NULL; the last is the absence of a
    row. Reporting an unrecorded source as withdrawn would withhold evidence on
    the strength of a fact nobody asserted.
    """

    __slots__ = ()

    async def current_document_ids(
        self,
        source_uris: Sequence[str],
        partition: SegmentationPartition,
    ) -> Mapping[str, Optional[str]]:
        """
        Look up the current document for each source.

        Parameters
        ----------
        source_uris:
            Sources to look up.
        partition:
            The segmentation partition to read from.

        Returns
        -------
        Mapping[str, Optional[str]]
            The current document per source. A source mapped to ``None`` is
            recorded as withdrawn; a source absent from the mapping has no
            record, and the two are not interchangeable.

        Raises
        ------
        StoreUnavailableError
            If the read failed.
        """
        if not source_uris:
            return {}

        try:
            async with self._backend.pool.connection() as conn:
                async with conn.cursor(row_factory=dict_row) as cur:
                    await cur.execute(
                        "SELECT source_uri, current_document_id "
                        "FROM ptfm_supersession "
                        "WHERE partition_key = %s AND source_uri = ANY(%s)",
                        (partition.key, list(dict.fromkeys(source_uris))),
                    )
                    rows = list(await cur.fetchall())
        except Exception as error:
            raise StoreUnavailableError(
                f"Supersession lookup failed: {type(error).__name__}.",
                store_name=self.name,
            ) from error

        return {row["source_uri"]: row["current_document_id"] for row in rows}

    async def record_current(
        self,
        source_uri: str,
        document_id: Optional[str],
        version_tag: Optional[str],
        partition: SegmentationPartition,
    ) -> None:
        """
        Record which document currently represents a source.

        Parameters
        ----------
        source_uri:
            The source.
        document_id:
            The current document, or ``None`` to record the source as
            withdrawn.
        version_tag:
            The source's version, when known.
        partition:
            The segmentation partition to write into.

        Raises
        ------
        IndexWriteError
            If the write failed.
        """
        try:
            async with self._backend.pool.connection() as conn:
                async with conn.transaction():
                    async with conn.cursor() as cur:
                        await _write_supersession(
                            cur, source_uri, document_id, version_tag, partition
                        )
        except Exception as error:
            raise IndexWriteError(
                f"Could not record currency for a source: "
                f"{type(error).__name__}."
            ) from error


class PostgresPartitionCatalog(_PostgresStore):
    """
    Registry of partitions and of which are active.

    The catalog is what makes a migration safe: a new partition is registered
    and populated while the active one continues serving queries, and activation
    is a single row update. Uniqueness of the active partition is enforced by a
    partial index, so two processes racing to activate different partitions
    cannot both succeed and no query can observe two identifier spaces.
    """

    __slots__ = ()

    async def active_segmentation(self) -> Optional[SegmentationPartition]:
        """
        The segmentation partition currently serving queries.

        Returns
        -------
        Optional[SegmentationPartition]
            The active partition, or ``None`` when none is active.

        Raises
        ------
        StoreUnavailableError
            If the read failed.
        """
        try:
            async with self._backend.pool.connection() as conn:
                async with conn.cursor(row_factory=dict_row) as cur:
                    await cur.execute(
                        "SELECT segmentation_fingerprint, source_schema_version "
                        "FROM ptfm_partition "
                        "WHERE kind = 'segmentation' AND is_active"
                    )
                    row = await cur.fetchone()
        except Exception as error:
            raise StoreUnavailableError(
                f"Could not read the active segmentation partition: "
                f"{type(error).__name__}.",
                store_name=self.name,
            ) from error

        if row is None:
            return None
        return SegmentationPartition(
            segmentation_fingerprint=row["segmentation_fingerprint"],
            source_schema_version=row["source_schema_version"],
        )

    async def active_embedding(
        self,
        segmentation: SegmentationPartition,
    ) -> Optional[EmbeddingPartition]:
        """
        The embedding partition currently serving queries over a segmentation.

        Parameters
        ----------
        segmentation:
            The segmentation partition.

        Returns
        -------
        Optional[EmbeddingPartition]
            The active embedding partition, or ``None`` when none is active.

        Raises
        ------
        StoreUnavailableError
            If the read failed.
        """
        try:
            async with self._backend.pool.connection() as conn:
                async with conn.cursor(row_factory=dict_row) as cur:
                    await cur.execute(
                        "SELECT model_identity, dimension, normalized, "
                        "       composition_id, query_encoding "
                        "FROM ptfm_partition "
                        "WHERE kind = 'embedding' AND is_active "
                        "  AND segmentation_key = %s",
                        (segmentation.key,),
                    )
                    row = await cur.fetchone()
        except Exception as error:
            raise StoreUnavailableError(
                f"Could not read the active embedding partition: "
                f"{type(error).__name__}.",
                store_name=self.name,
            ) from error

        if row is None:
            return None
        return EmbeddingPartition(
            segmentation=segmentation,
            model_identity=row["model_identity"],
            dimension=int(row["dimension"]),
            normalized=bool(row["normalized"]),
            composition_id=row["composition_id"],
            query_encoding=row["query_encoding"] or "",
        )

    async def register(
        self,
        segmentation: SegmentationPartition,
        embedding: Optional[EmbeddingPartition],
        *,
        activate: bool,
    ) -> None:
        """
        Register partitions, optionally making them the active ones.

        Registration and activation happen in one transaction, so a partition
        can never be active without being registered.

        Parameters
        ----------
        segmentation:
            The segmentation partition to register.
        embedding:
            The embedding partition to register, when there is one.
        activate:
            Whether these become the partitions serving queries.

        Raises
        ------
        PartitionError
            If the embedding partition describes a different segmentation than
            the one given, or if activation would leave two partitions active.
        IndexWriteError
            If the write failed.
        """
        if embedding is not None and (
            embedding.segmentation.key != segmentation.key
        ):
            raise PartitionError(
                f"Embedding partition {embedding} describes segmentation "
                f"{embedding.segmentation} but was registered against "
                f"{segmentation}. Registering it would make the catalog claim "
                f"a vector set covers spans it was not built from."
            )

        try:
            async with self._backend.pool.connection() as conn:
                async with conn.transaction():
                    async with conn.cursor() as cur:
                        await self._register_segmentation(
                            cur, segmentation, activate
                        )
                        if embedding is not None:
                            await self._register_embedding(
                                cur, embedding, activate
                            )
        except errors.UniqueViolation as error:
            raise PartitionError(
                "Activating this partition would leave two partitions active "
                "at once, which the catalog forbids so that no query can "
                "observe two identifier spaces. Deactivate the current one "
                "first."
            ) from error
        except PartitionError:
            raise
        except Exception as error:
            raise IndexWriteError(
                f"Could not register partitions: {type(error).__name__}."
            ) from error

        if embedding is not None and self._backend.builds_ann_indexes:
            # Outside the transaction: index construction is slow, and holding
            # a write transaction open for it would block the writes it exists
            # to accelerate.
            await self._backend.ensure_ann_index(embedding)

    async def _register_segmentation(
        self,
        cursor: Any,
        segmentation: SegmentationPartition,
        activate: bool,
    ) -> None:
        """
        Write a segmentation partition row.

        Parameters
        ----------
        cursor:
            An open cursor inside the registration transaction.
        segmentation:
            The partition to register.
        activate:
            Whether it becomes active, deactivating any other.
        """
        if activate:
            await cursor.execute(
                "UPDATE ptfm_partition SET is_active = false "
                "WHERE kind = 'segmentation' AND is_active "
                "  AND partition_key <> %s",
                (segmentation.key,),
            )

        await cursor.execute(
            "INSERT INTO ptfm_partition ("
            "  partition_key, kind, segmentation_key, "
            "  segmentation_fingerprint, source_schema_version, is_active, "
            "  schema_version"
            ") VALUES (%s, 'segmentation', %s, %s, %s, %s, %s) "
            "ON CONFLICT (partition_key) DO UPDATE SET "
            "  is_active = ptfm_partition.is_active OR EXCLUDED.is_active",
            (
                segmentation.key,
                segmentation.key,
                segmentation.segmentation_fingerprint,
                segmentation.source_schema_version,
                activate,
                SCHEMA_VERSION,
            ),
        )

    async def _register_embedding(
        self,
        cursor: Any,
        embedding: EmbeddingPartition,
        activate: bool,
    ) -> None:
        """
        Write an embedding partition row.

        Parameters
        ----------
        cursor:
            An open cursor inside the registration transaction.
        embedding:
            The partition to register.
        activate:
            Whether it becomes active, deactivating any other over the same
            segmentation.
        """
        segmentation = embedding.segmentation
        if activate:
            await cursor.execute(
                "UPDATE ptfm_partition SET is_active = false "
                "WHERE kind = 'embedding' AND is_active "
                "  AND segmentation_key = %s AND partition_key <> %s",
                (segmentation.key, embedding.key),
            )

        await cursor.execute(
            "INSERT INTO ptfm_partition ("
            "  partition_key, kind, segmentation_key, "
            "  segmentation_fingerprint, source_schema_version, "
            "  model_identity, dimension, normalized, composition_id, "
            "  query_encoding, is_active, schema_version"
            ") VALUES (%s, 'embedding', %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) "
            "ON CONFLICT (partition_key) DO UPDATE SET "
            "  is_active = ptfm_partition.is_active OR EXCLUDED.is_active",
            (
                embedding.key,
                segmentation.key,
                segmentation.segmentation_fingerprint,
                segmentation.source_schema_version,
                embedding.model_identity,
                embedding.dimension,
                embedding.normalized,
                embedding.composition_id,
                embedding.query_encoding,
                activate,
                SCHEMA_VERSION,
            ),
        )


class PostgresIndexWriter(_PostgresStore):
    """
    Atomic writer for knowledge units.

    The reason this backend exists. Buffered units are written in ONE
    transaction spanning nodes, vectors, relations, and currency, so a failure
    anywhere leaves the database exactly as it was. A node retrievable by an
    index with no authoritative text is therefore unreachable, not merely
    unlikely.

    Buffering is deliberate. Writing each unit as it arrives would make
    atomicity a property of a single unit rather than of the batch, and a
    partially ingested document is precisely the state ingestion must not be
    able to leave behind.
    """

    __slots__ = ("_buffer",)

    def __init__(self, backend: PostgresBackend) -> None:
        """
        Bind the writer to a backend.

        Parameters
        ----------
        backend:
            The backend whose pool this writer uses.
        """
        super().__init__(backend)
        self._buffer: list[
            tuple[
                SegmentationPartition,
                Optional[EmbeddingPartition],
                tuple[KnowledgeUnit, ...],
            ]
        ] = []

    @property
    def is_atomic(self) -> bool:
        """
        Whether a flush either fully succeeds or changes nothing.

        ``True``, backed by a single database transaction rather than by an
        ordering of writes that usually succeed together.

        Returns
        -------
        bool
            ``True``.
        """
        return True

    @property
    def pending_units(self) -> int:
        """
        How many units are buffered but not yet committed.

        Returns
        -------
        int
            The buffered unit count.
        """
        return sum(len(units) for _, _, units in self._buffer)

    async def write_units(
        self,
        segmentation: SegmentationPartition,
        embedding: Optional[EmbeddingPartition],
        units: Sequence[KnowledgeUnit],
    ) -> None:
        """
        Buffer units for the next flush.

        Validates partition membership here rather than at flush, so a
        mismatched unit is rejected at the point the caller can still act on it
        instead of failing a batch assembled from many sources.

        Parameters
        ----------
        segmentation:
            The segmentation partition the nodes belong to.
        embedding:
            The embedding partition the vectors belong to, when the units carry
            embeddings.
        units:
            Units to write.

        Raises
        ------
        SegmentationMismatchError
            If a node does not belong to the named segmentation partition.
        EmbeddingSpaceMismatchError
            If an embedding does not describe the named embedding partition.
        PartitionError
            If units carry embeddings with no embedding partition given.
        """
        if not units:
            return

        nodes = [unit.node for unit in units]
        _require_partition_match(nodes, segmentation)

        embeddings = [
            unit.embedding for unit in units if unit.embedding is not None
        ]
        if embeddings:
            if embedding is None:
                raise PartitionError(
                    f"{len(embeddings)} unit(s) carry an embedding but no "
                    f"embedding partition was given. Storing a vector without "
                    f"naming the space it belongs to would make it "
                    f"incomparable with every other vector, including itself "
                    f"after a model change."
                )
            _require_embedding_match(embeddings, embedding)

        self._buffer.append((segmentation, embedding, tuple(units)))

    async def flush(self) -> None:
        """
        Commit every buffered unit in one transaction.

        The buffer is cleared only after the transaction commits. A failed flush
        therefore leaves the buffer intact and the database untouched, so a
        caller may retry without reconstructing what it had already handed over.

        Raises
        ------
        PartialIndexWriteError
            If the transaction failed but the database's state after the failure
            could not be confirmed as unchanged. Distinguished from an ordinary
            write failure because the two call for different responses: a
            rolled-back write may be retried, while an indeterminate one must be
            reconciled.
        IndexWriteError
            If the transaction failed and was rolled back, leaving the database
            unchanged.
        """
        if not self._buffer:
            return

        batch = tuple(self._buffer)
        try:
            async with self._backend.pool.connection() as conn:
                async with conn.transaction():
                    async with conn.cursor() as cur:
                        for segmentation, embedding, units in batch:
                            await self._write_batch(
                                cur, segmentation, embedding, units
                            )
        except errors.OperationalError as error:
            # The connection failed, so whether the transaction committed is
            # not knowable from here. Reported as indeterminate rather than
            # assumed rolled back.
            raise PartialIndexWriteError(
                f"A unit write failed with a connection-level error, so "
                f"whether it committed is unknown: {type(error).__name__}. "
                f"Verify the affected documents before retrying.",
                failed_stores=_WRITTEN_STORES,
            ) from error
        except Exception as error:
            raise IndexWriteError(
                f"A unit write failed and was rolled back, leaving the "
                f"database unchanged: {type(error).__name__}."
            ) from error

        self._buffer.clear()

    async def _write_batch(
        self,
        cursor: Any,
        segmentation: SegmentationPartition,
        embedding: Optional[EmbeddingPartition],
        units: Sequence[KnowledgeUnit],
    ) -> None:
        """
        Write one buffered group inside the flush transaction.

        Order within the transaction does not affect atomicity, since nothing
        commits until all of it does. Nodes go first regardless, so that a
        constraint violation names the node rather than a dependent row.

        Parameters
        ----------
        cursor:
            An open cursor inside the flush transaction.
        segmentation:
            The segmentation partition.
        embedding:
            The embedding partition, when the units carry embeddings.
        units:
            The units to write.
        """
        await _write_nodes(cursor, [unit.node for unit in units], segmentation)

        embeddings = [
            unit.embedding for unit in units if unit.embedding is not None
        ]
        if embeddings and embedding is not None:
            await _write_vectors(cursor, embeddings, embedding)

        relations = [
            relation for unit in units for relation in unit.relations
        ]
        if relations:
            await _write_relations(cursor, relations, segmentation)

        for source_uri, (document_id, version_tag) in _current_documents(
            units
        ).items():
            await _write_supersession(
                cursor, source_uri, document_id, version_tag, segmentation
            )

    async def close(self) -> None:
        """
        Flush anything buffered and release the writer.

        Flushes rather than discards, because a caller that closed without
        flushing more likely forgot than intended to lose the batch. A failing
        flush propagates, so the loss cannot pass unnoticed either way.

        Raises
        ------
        IndexWriteError
            If the final flush failed.
        PartialIndexWriteError
            If the final flush left an indeterminate state.
        """
        await self.flush()


def _current_documents(
    units: Sequence[KnowledgeUnit],
) -> dict[str, tuple[Optional[str], Optional[str]]]:
    """
    Determine the current document per source in a batch.

    Derived from the units rather than recorded separately, so that ingesting a
    document and marking it current cannot disagree.

    A batch naming two documents for one source is a contradiction the caller
    must resolve, not one this function may pick a winner for, so the later
    document wins only when the two agree; otherwise the source is left for the
    caller to record explicitly.

    Parameters
    ----------
    units:
        The units being written.

    Returns
    -------
    dict[str, tuple[Optional[str], Optional[str]]]
        Document identifier and version tag per source URI, omitting any source
        the batch describes inconsistently.
    """
    seen: dict[str, set[str]] = {}
    tags: dict[str, Optional[str]] = {}
    for unit in units:
        source_uri = unit.node.source.source_uri
        seen.setdefault(source_uri, set()).add(unit.node.document_id)
        tags[source_uri] = unit.node.source.version_tag

    return {
        source_uri: (next(iter(documents)), tags[source_uri])
        for source_uri, documents in seen.items()
        if len(documents) == 1
    }


def _require_partition_match(
    nodes: Sequence[KnowledgeNode],
    partition: SegmentationPartition,
) -> None:
    """
    Confirm every node belongs to a segmentation partition.

    Checked before any write, because a node stored under a partition whose
    segmentation differs from its own would be returned for spans that
    partition does not describe, and the span is what makes a node evidence.

    Parameters
    ----------
    nodes:
        Nodes to check.
    partition:
        The partition they are claimed to belong to.

    Raises
    ------
    SegmentationMismatchError
        If a node belongs to a different partition.
    """
    for node in nodes:
        if not partition.accepts(
            node.segmentation_fingerprint, node.source_schema_version
        ):
            raise SegmentationMismatchError(
                expected=partition.segmentation_fingerprint,
                actual=node.segmentation_fingerprint,
                context={
                    "node_id": node.node_id,
                    "partition_key": partition.key,
                    "node_schema_version": node.source_schema_version,
                    "partition_schema_version": partition.source_schema_version,
                },
            )


def _require_embedding_match(
    embeddings: Sequence[NodeEmbedding],
    partition: EmbeddingPartition,
) -> None:
    """
    Confirm every embedding describes a partition's vector space.

    Parameters
    ----------
    embeddings:
        Embeddings to check.
    partition:
        The partition they are claimed to belong to.

    Raises
    ------
    EmbeddingSpaceMismatchError
        If an embedding describes a different space.
    """
    for embedding in embeddings:
        actual = embedding_for_vector(embedding, partition.segmentation)
        if not partition.describes_same_space(actual):
            raise EmbeddingSpaceMismatchError(
                f"Embedding for node {embedding.node_id} belongs to space "
                f"{actual.key!r} but was offered to partition "
                f"{partition.key!r}. A distance between vectors from "
                f"different spaces is not a distance, so the write is refused "
                f"rather than stored where it would be compared.",
                expected_identity=partition.model_identity,
                actual_identity=actual.model_identity,
                context={"node_id": embedding.node_id},
            )


async def _write_nodes(
    cursor: Any,
    nodes: Sequence[KnowledgeNode],
    partition: SegmentationPartition,
) -> None:
    """
    Insert or replace node rows.

    One ``executemany`` rather than a statement per node, because a document's
    nodes number in the hundreds and a round trip each would dominate ingestion
    time.

    ``source_uri`` is written from the source descriptor rather than accepted
    separately, so the indexed copy cannot disagree with the descriptor it is
    derived from.

    Parameters
    ----------
    cursor:
        An open cursor.
    nodes:
        Nodes to write.
    partition:
        The segmentation partition to write into.
    """
    await cursor.executemany(
        "INSERT INTO ptfm_node ("
        "  partition_key, node_id, document_id, ordinal, span_start, "
        "  span_end, node_text, node_type, block_ids, heading_path, "
        "  page_numbers, node_metadata, source_uri, source, "
        "  segmentation_fingerprint, source_text_hash, "
        "  source_schema_version, processed_at"
        ") VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) "
        "ON CONFLICT (partition_key, node_id) DO UPDATE SET "
        "  document_id = EXCLUDED.document_id, "
        "  ordinal = EXCLUDED.ordinal, "
        "  span_start = EXCLUDED.span_start, "
        "  span_end = EXCLUDED.span_end, "
        "  node_text = EXCLUDED.node_text, "
        "  node_type = EXCLUDED.node_type, "
        "  block_ids = EXCLUDED.block_ids, "
        "  heading_path = EXCLUDED.heading_path, "
        "  page_numbers = EXCLUDED.page_numbers, "
        "  node_metadata = EXCLUDED.node_metadata, "
        "  source_uri = EXCLUDED.source_uri, "
        "  source = EXCLUDED.source, "
        "  source_text_hash = EXCLUDED.source_text_hash, "
        "  processed_at = EXCLUDED.processed_at",
        [_node_row(node, partition) for node in nodes],
    )


def _node_row(
    node: KnowledgeNode,
    partition: SegmentationPartition,
) -> tuple[Any, ...]:
    """
    Render a node as its row values.

    The source descriptor is stored whole as ``jsonb`` rather than decomposed
    into columns. Its shape belongs to Knowledge Processing, and enumerating its
    fields here would make a Module 2 addition a Module 3 schema change.

    Parameters
    ----------
    node:
        The node to render.
    partition:
        The segmentation partition it is written into.

    Returns
    -------
    tuple[Any, ...]
        Row values in the order the insert statement expects.
    """
    return (
        partition.key,
        node.node_id,
        node.document_id,
        node.ordinal,
        node.span.start,
        node.span.end,
        node.text,
        str(node.node_type.value),
        list(node.block_ids),
        list(node.heading_path),
        list(node.page_numbers),
        json.dumps(dict(node.metadata), sort_keys=True),
        node.source.source_uri,
        _dump_source(node.source),
        node.segmentation_fingerprint,
        node.source_text_hash,
        node.source_schema_version,
        node.processed_at,
    )


def _dump_source(source: SourceDescriptor) -> str:
    """
    Serialize a source descriptor for storage.

    Uses the model's own serialization when it has one, so that a field added to
    the descriptor round-trips without a change here.

    Parameters
    ----------
    source:
        The descriptor to serialize.

    Returns
    -------
    str
        A JSON object.
    """
    dump = getattr(source, "model_dump_json", None)
    if callable(dump):
        return str(dump())
    return json.dumps(
        {
            key: value
            for key, value in vars(source).items()
            if not key.startswith("_")
        },
        default=str,
        sort_keys=True,
    )


def _node_from_row(row: Mapping[str, Any], store_name: str) -> KnowledgeNode:
    """
    Reconstruct a node from its row.

    Every provenance field is restored, including the span and the source text
    hash, because hydration verifies that a node's text is the text its span
    denotes in the source it names. A reconstruction that dropped any of them
    would make that check unperformable, and an unperformable check reads as a
    passing one.

    Parameters
    ----------
    row:
        A row selected with ``_NODE_COLUMNS``.
    store_name:
        The store reporting the failure, so that a caller can tell which store
        holds the unreadable row.

    Returns
    -------
    KnowledgeNode
        The node, with provenance intact.

    Raises
    ------
    StoreUnavailableError
        If the row cannot be interpreted as a node, which indicates a schema
        written by an incompatible version rather than a transient fault.
    """
    try:
        return KnowledgeNode(
            node_id=row["node_id"],
            document_id=row["document_id"],
            segmentation_fingerprint=row["segmentation_fingerprint"],
            source_schema_version=row["source_schema_version"],
            ordinal=int(row["ordinal"]),
            text=row["node_text"],
            source_text_hash=row["source_text_hash"],
            span=TextSpan(start=int(row["span_start"]), end=int(row["span_end"])),
            node_type=NodeType(row["node_type"]),
            block_ids=tuple(row["block_ids"] or ()),
            source=_load_source(row["source"]),
            heading_path=tuple(row["heading_path"] or ()),
            page_numbers=tuple(int(page) for page in (row["page_numbers"] or ())),
            metadata=dict(row["node_metadata"] or {}),
            processed_at=row["processed_at"],
        )
    except Exception as error:
        raise StoreUnavailableError(
            f"A stored node row could not be interpreted as a node: "
            f"{type(error).__name__}. This indicates rows written under an "
            f"incompatible schema version rather than a transient fault.",
            store_name=store_name,
        ) from error


def _load_source(payload: Any) -> SourceDescriptor:
    """
    Reconstruct a source descriptor from its stored form.

    Parameters
    ----------
    payload:
        The stored object, already decoded from ``jsonb``.

    Returns
    -------
    SourceDescriptor
        The descriptor.
    """
    fields = payload if isinstance(payload, Mapping) else json.loads(payload)
    validate = getattr(SourceDescriptor, "model_validate", None)
    if callable(validate):
        return validate(dict(fields))
    return SourceDescriptor(**dict(fields))


async def _write_vectors(
    cursor: Any,
    embeddings: Sequence[NodeEmbedding],
    partition: EmbeddingPartition,
) -> None:
    """
    Insert or replace vector rows.

    The vector is passed as a string in pgvector's literal form and cast, which
    avoids requiring the driver's vector adapter to be registered. The value is
    still a bound parameter, so the composition remains structural.

    Parameters
    ----------
    cursor:
        An open cursor.
    embeddings:
        Embeddings to write.
    partition:
        The embedding partition to write into.
    """
    await cursor.executemany(
        "INSERT INTO ptfm_vector (partition_key, node_id, dimension, embedding) "
        "VALUES (%s, %s, %s, %s::vector) "
        "ON CONFLICT (partition_key, node_id) DO UPDATE SET "
        "  dimension = EXCLUDED.dimension, "
        "  embedding = EXCLUDED.embedding",
        [
            (
                partition.key,
                embedding.node_id,
                embedding.dimension,
                _vector_literal(embedding.vector),
            )
            for embedding in embeddings
        ],
    )


def _vector_literal(vector: Sequence[float]) -> str:
    """
    Render a vector in pgvector's literal form.

    ``repr`` is used per component rather than a shortened decimal form, so the
    literal itself loses nothing. The COLUMN does lose something: pgvector's
    ``vector`` type stores single-precision floats, so a component written from
    a Python float is read back rounded to float32. That is a property of the
    type and not of this rendering, which is why the rendering does not
    compound it.

    Parameters
    ----------
    vector:
        The components.

    Returns
    -------
    str
        A bracketed, comma-separated literal.
    """
    return "[" + ",".join(repr(float(value)) for value in vector) + "]"


async def _write_relations(
    cursor: Any,
    relations: Sequence[NodeRelation],
    partition: SegmentationPartition,
) -> None:
    """
    Insert relation rows, ignoring duplicates.

    ``DO NOTHING`` rather than an update, because a relation row carries nothing
    beyond its own identity: re-inserting it has no field to refresh.

    Parameters
    ----------
    cursor:
        An open cursor.
    relations:
        Relations to write.
    partition:
        The segmentation partition to write into.
    """
    await cursor.executemany(
        "INSERT INTO ptfm_relation ("
        "  partition_key, source_node_id, target_node_id, relation_type"
        ") VALUES (%s, %s, %s, %s) ON CONFLICT DO NOTHING",
        [
            (
                partition.key,
                relation.source_node_id,
                relation.target_node_id,
                str(
                    getattr(
                        relation.relation_type, "value", relation.relation_type
                    )
                ),
            )
            for relation in relations
        ],
    )


async def _write_supersession(
    cursor: Any,
    source_uri: str,
    document_id: Optional[str],
    version_tag: Optional[str],
    partition: SegmentationPartition,
) -> None:
    """
    Record which document currently represents a source.

    Parameters
    ----------
    cursor:
        An open cursor.
    source_uri:
        The source.
    document_id:
        The current document, or ``None`` to record withdrawal.
    version_tag:
        The source's version, when known.
    partition:
        The segmentation partition to write into.
    """
    await cursor.execute(
        "INSERT INTO ptfm_supersession ("
        "  partition_key, source_uri, current_document_id, version_tag"
        ") VALUES (%s, %s, %s, %s) "
        "ON CONFLICT (partition_key, source_uri) DO UPDATE SET "
        "  current_document_id = EXCLUDED.current_document_id, "
        "  version_tag = EXCLUDED.version_tag, "
        "  recorded_at = now()",
        (partition.key, source_uri, document_id, version_tag),
    )
