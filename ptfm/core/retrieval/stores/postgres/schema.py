"""
PTFM PostgreSQL Retrieval Schema

File:
    ptfm/core/retrieval/stores/postgres/schema.py

Description:
    The relational schema behind the PostgreSQL retrieval backend, and the
    statements that create it.

Why one database holds every store
----------------------------------
    A knowledge unit's write touches nodes, vectors, lexical terms, relations,
    and supersession together. Across separate services those writes cannot be
    atomic, and a crash between them leaves a node retrievable by one index
    with no authoritative text: a result that appears valid and cannot be
    traced, which is the worst failure available to an evidence system.

    Holding all of it in one database makes the unit write a single
    transaction. That is the reason this backend is the default, and the reason
    the stores below share a connection rather than owning one each.

Lexical search lives on the node row
------------------------------------
    The searchable form of a node's text is a column on the node table rather
    than a table of its own.

    A separate table would be a second thing to keep consistent with the first,
    and the consistency would have to be maintained by the application rather
    than by the schema. As a column it cannot drift: a node and its search
    terms are written by one statement and deleted by one cascade.

    The protocols stay separate regardless. ``NodeStore`` and ``LexicalIndex``
    remain distinct contracts that happen to be served by one table, which is
    what lets a deployment replace the lexical arm with a dedicated engine
    without the node store noticing.

Vectors are keyed by embedding partition
----------------------------------------
    Everything else is keyed by SEGMENTATION partition. Vectors are keyed by
    EMBEDDING partition, because that is the whole of the migration story:
    re-embedding a corpus writes a new set of vectors and leaves nodes, spans,
    identifiers, and lexical terms untouched.

    The column is an unconstrained ``vector``, so one table serves partitions of
    differing dimension. That is a deliberate trade, and its consequence is
    stated where the index helper is defined.

Nothing is deleted
------------------
    No statement here removes a node. A superseded document's nodes stay
    resolvable so that a citation recorded before the change still resolves,
    and supersession is expressed by a row in the currency table rather than by
    a deletion.

Dependency contract
-------------------
    Statements are declared as text constants and contain no interpolated
    value. This module imports nothing beyond the standard library, so the
    schema can be read, printed, and diffed without a database driver
    installed.
"""

from __future__ import annotations

from typing import Final

__all__ = [
    "SCHEMA_VERSION",
    "REQUIRED_EXTENSIONS",
    "CREATE_STATEMENTS",
    "ann_index_statement",
    "drop_statements",
]

#: Version of this schema. A change that alters a column's meaning must
#: increment it, because a deployment reading rows written under an earlier
#: version would misinterpret them rather than fail.
SCHEMA_VERSION: Final[str] = "1"

#: Extensions the backend requires. Vector similarity is the only one: full
#: text search is built in, so a deployment needs no extension for the lexical
#: arm.
REQUIRED_EXTENSIONS: Final[tuple[str, ...]] = ("vector",)


#: Interpret text as a number, yielding NULL when it is not one.
#:
#: Exists because the in-memory reference treats an ordered comparison against
#: a non-numeric value as no match rather than as an error, and SQL has no
#: total cast. A bare ``::double precision`` raises, and a regex guard combined
#: with ``AND`` or ``CASE`` does not reliably stop the planner evaluating the
#: cast first, so the suppression has to happen inside a function.
#:
#: NULL propagates through the comparison to NULL, which a ``WHERE`` clause
#: treats as not matching: the same answer the reference gives, reached by the
#: same reasoning rather than by coincidence.
_CREATE_TO_NUMBER: Final[str] = """
CREATE OR REPLACE FUNCTION ptfm_to_number(candidate text)
RETURNS double precision
LANGUAGE plpgsql
IMMUTABLE
STRICT
PARALLEL SAFE
AS $$
BEGIN
    RETURN candidate::double precision;
EXCEPTION WHEN invalid_text_representation OR numeric_value_out_of_range THEN
    RETURN NULL;
END;
$$
"""

#: Nodes, with their provenance and their searchable text.
#:
#: The generated ``search_vector`` column is what makes the lexical arm
#: consistent by construction: it cannot disagree with the text beside it
#: because the database derives it from that text. An application-maintained
#: column could drift from the text it indexes, and a node retrievable by terms
#: it no longer contains is an evidence item whose support is fictional.
#:
#: Only the node's own text is indexed. Heading text is deliberately excluded:
#: a section is selected precisely by a ``HEADING_PATH`` filter, and folding the
#: same headings into the free-text score would let a node rank on words that a
#: filter already expresses exactly, which is a second, weaker mechanism for
#: something the first one does correctly.
#:
#: The ``'simple'`` configuration is chosen over a language configuration
#: because it applies no stemming and removes no stop words. A language
#: configuration would make matching depend on a guess about the document's
#: language, and a wrong guess silently drops terms. Matching a term the caller
#: did not write is not a property an evidence system should acquire by
#: default.
_CREATE_NODE: Final[str] = """
CREATE TABLE IF NOT EXISTS ptfm_node (
    partition_key            text        NOT NULL,
    node_id                  text        NOT NULL,
    document_id              text        NOT NULL,
    ordinal                  integer     NOT NULL,
    span_start               integer     NOT NULL,
    span_end                 integer     NOT NULL,
    node_text                text        NOT NULL,
    node_type                text        NOT NULL,
    block_ids                text[]      NOT NULL DEFAULT '{}',
    heading_path             text[]      NOT NULL DEFAULT '{}',
    page_numbers             integer[]   NOT NULL DEFAULT '{}',
    node_metadata            jsonb       NOT NULL DEFAULT '{}'::jsonb,
    source_uri               text        NOT NULL,
    source                   jsonb       NOT NULL,
    segmentation_fingerprint text        NOT NULL,
    source_text_hash         text        NOT NULL,
    source_schema_version    text        NOT NULL,
    processed_at             timestamptz NOT NULL,
    search_vector            tsvector    GENERATED ALWAYS AS (
        to_tsvector('simple', node_text)
    ) STORED,
    CONSTRAINT ptfm_node_pkey PRIMARY KEY (partition_key, node_id),
    CONSTRAINT ptfm_node_span_ordered CHECK (span_end >= span_start),
    CONSTRAINT ptfm_node_ordinal_natural CHECK (ordinal >= 0)
)
"""

#: Position lookup, which is how context expansion finds a node's neighbours.
#: Unique because two nodes cannot occupy one position in a document, and
#: declaring that here makes a segmentation defect a write failure rather than
#: an ambiguous read.
_CREATE_NODE_ORDINAL_INDEX: Final[str] = """
CREATE UNIQUE INDEX IF NOT EXISTS ptfm_node_position
    ON ptfm_node (partition_key, document_id, ordinal)
"""

_CREATE_NODE_SEARCH_INDEX: Final[str] = """
CREATE INDEX IF NOT EXISTS ptfm_node_search
    ON ptfm_node USING GIN (search_vector)
"""

#: Supports the supersession filter, which resolves currency by source.
_CREATE_NODE_SOURCE_INDEX: Final[str] = """
CREATE INDEX IF NOT EXISTS ptfm_node_source
    ON ptfm_node (partition_key, source_uri)
"""

#: Supports metadata filters, which are how a domain plugin's attributes reach
#: a query without the Core naming any of them.
_CREATE_NODE_METADATA_INDEX: Final[str] = """
CREATE INDEX IF NOT EXISTS ptfm_node_metadata
    ON ptfm_node USING GIN (node_metadata jsonb_path_ops)
"""

#: Vectors, keyed by embedding partition.
#:
#: The column is an unconstrained ``vector`` so that one table serves every
#: partition whatever its dimension. ``dimension`` is stored beside it so a
#: query can restrict itself to rows it can compare against, which matters
#: because the column itself does not enforce a width.
_CREATE_VECTOR: Final[str] = """
CREATE TABLE IF NOT EXISTS ptfm_vector (
    partition_key text    NOT NULL,
    node_id       text    NOT NULL,
    dimension     integer NOT NULL,
    embedding     vector  NOT NULL,
    CONSTRAINT ptfm_vector_pkey PRIMARY KEY (partition_key, node_id),
    CONSTRAINT ptfm_vector_dimension_positive CHECK (dimension > 0)
)
"""

#: Structural relations. The primary key makes a rewrite idempotent, so
#: reprocessing a document cannot accumulate duplicate edges that would make a
#: traversal visit one target repeatedly.
_CREATE_RELATION: Final[str] = """
CREATE TABLE IF NOT EXISTS ptfm_relation (
    partition_key  text NOT NULL,
    source_node_id text NOT NULL,
    target_node_id text NOT NULL,
    relation_type  text NOT NULL,
    CONSTRAINT ptfm_relation_pkey
        PRIMARY KEY (partition_key, source_node_id, target_node_id, relation_type)
)
"""

_CREATE_RELATION_SOURCE_INDEX: Final[str] = """
CREATE INDEX IF NOT EXISTS ptfm_relation_source
    ON ptfm_relation (partition_key, source_node_id)
"""

#: Which document currently represents each source.
#:
#: ``current_document_id`` is nullable on purpose: a row with no document is a
#: tombstone marking a withdrawn source, which is distinct from having no row at
#: all. Collapsing the two would make a deletion indistinguishable from an
#: omission, so the distinction is carried by the schema rather than by a
#: convention.
_CREATE_SUPERSESSION: Final[str] = """
CREATE TABLE IF NOT EXISTS ptfm_supersession (
    partition_key       text        NOT NULL,
    source_uri          text        NOT NULL,
    current_document_id text,
    version_tag         text,
    recorded_at         timestamptz NOT NULL DEFAULT now(),
    CONSTRAINT ptfm_supersession_pkey PRIMARY KEY (partition_key, source_uri)
)
"""

#: The partition catalog.
#:
#: The partial unique index is what makes "exactly one active segmentation" a
#: property of the database rather than of the application. Two processes racing
#: to activate different partitions cannot both succeed, so a query can never
#: observe two identifier spaces.
_CREATE_PARTITION: Final[str] = """
CREATE TABLE IF NOT EXISTS ptfm_partition (
    partition_key            text        NOT NULL,
    kind                     text        NOT NULL,
    segmentation_key         text        NOT NULL,
    segmentation_fingerprint text        NOT NULL,
    source_schema_version    text        NOT NULL,
    model_identity           text,
    dimension                integer,
    normalized               boolean,
    composition_id           text,
    query_encoding           text,
    is_active                boolean     NOT NULL DEFAULT false,
    schema_version           text        NOT NULL,
    registered_at            timestamptz NOT NULL DEFAULT now(),
    CONSTRAINT ptfm_partition_pkey PRIMARY KEY (partition_key),
    CONSTRAINT ptfm_partition_kind
        CHECK (kind IN ('segmentation', 'embedding'))
)
"""

_CREATE_ONE_ACTIVE_SEGMENTATION: Final[str] = """
CREATE UNIQUE INDEX IF NOT EXISTS ptfm_partition_one_active_segmentation
    ON ptfm_partition ((kind))
    WHERE kind = 'segmentation' AND is_active
"""

#: At most one active embedding partition per segmentation. Several may exist,
#: which is how a new one is built beside the one serving queries.
_CREATE_ONE_ACTIVE_EMBEDDING: Final[str] = """
CREATE UNIQUE INDEX IF NOT EXISTS ptfm_partition_one_active_embedding
    ON ptfm_partition (segmentation_key)
    WHERE kind = 'embedding' AND is_active
"""

#: Every statement needed to create the schema, in dependency order. Each is
#: idempotent, so applying them to an existing database is a no-op rather than
#: an error.
CREATE_STATEMENTS: Final[tuple[str, ...]] = (
    _CREATE_TO_NUMBER,
    _CREATE_NODE,
    _CREATE_NODE_ORDINAL_INDEX,
    _CREATE_NODE_SEARCH_INDEX,
    _CREATE_NODE_SOURCE_INDEX,
    _CREATE_NODE_METADATA_INDEX,
    _CREATE_VECTOR,
    _CREATE_RELATION,
    _CREATE_RELATION_SOURCE_INDEX,
    _CREATE_SUPERSESSION,
    _CREATE_PARTITION,
    _CREATE_ONE_ACTIVE_SEGMENTATION,
    _CREATE_ONE_ACTIVE_EMBEDDING,
)


def ann_index_statement(
    partition_key: str,
    dimension: int,
    *,
    lists: int = 0,
) -> str:
    """
    Build the statement creating an approximate index for one partition.

    Why this is per-partition and why it is optional
    -----------------------------------------------
    An approximate index requires a vector column of known width, and the
    vector column here is deliberately unconstrained so that one table can
    serve partitions of differing dimension. The resolution is a partial index
    over a width cast, restricted to one partition: the partition knows its
    dimension, so the cast is valid for every row the index covers.

    Optional because approximate search is not reproducible. Results shift with
    insertion order and index construction, so a caller auditing a past answer
    must be able to search exactly. Without this index every search is exact
    and linear; with it, exact search remains available and approximate search
    becomes fast. A deployment chooses, and the choice is visible in the
    schema rather than buried in a query plan.

    Parameters
    ----------
    partition_key:
        The embedding partition to index. Interpolated as a literal, so callers
        must pass a value the catalog produced rather than user input; partition
        keys are digests, which contain no character that could terminate the
        literal.
    dimension:
        The partition's vector width.
    lists:
        Unused, accepted so that a caller need not distinguish index types.
        Retained for signature stability rather than ignored silently: HNSW
        takes no list count, and an IVFFlat variant would.

    Returns
    -------
    str
        A ``CREATE INDEX`` statement.

    Raises
    ------
    ValueError
        If the dimension is not positive, or the partition key is not a
        hexadecimal digest. The second check is what makes literal
        interpolation safe here.
    """
    if dimension < 1:
        raise ValueError(
            f"An approximate index requires a positive dimension, got "
            f"{dimension}."
        )

    if not partition_key or not all(
        character in "0123456789abcdef" for character in partition_key
    ):
        raise ValueError(
            f"Partition key {partition_key!r} is not a hexadecimal digest. "
            f"Index names and partial-index predicates cannot be parameterized, "
            f"so only a value of known shape may be interpolated."
        )

    return (
        f"CREATE INDEX IF NOT EXISTS ptfm_vector_ann_{partition_key} "
        f"ON ptfm_vector USING hnsw "
        f"((embedding::vector({dimension})) vector_cosine_ops) "
        f"WHERE partition_key = '{partition_key}'"
    )


def drop_statements() -> tuple[str, ...]:
    """
    Build the statements that remove the schema entirely.

    For tearing down a test database, and for nothing else. No part of normal
    operation removes a table, because a superseded document's nodes must stay
    resolvable for a citation recorded before the change to resolve.

    Returns
    -------
    tuple[str, ...]
        ``DROP TABLE`` statements, in an order that respects nothing because
        the tables carry no foreign keys between them.
    """
    tables = (
        "ptfm_node",
        "ptfm_vector",
        "ptfm_relation",
        "ptfm_supersession",
        "ptfm_partition",
    )
    return tuple(
        f"DROP TABLE IF EXISTS {table} CASCADE" for table in tables
    ) + ("DROP FUNCTION IF EXISTS ptfm_to_number(text)",)
