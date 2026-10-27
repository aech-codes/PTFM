"""
PTFM Partition Resolution

File:
    ptfm/core/retrieval/partition.py

Description:
    Resolves the partitions a query serves, and derives the partitions a
    knowledge unit belongs to.

    This is where compatibility stops being something the system checks and
    becomes something it cannot express.

Two resolutions, one asymmetry
------------------------------
    On INGEST, a partition is DERIVED from the artifacts themselves. A node
    states the segmentation it was produced under and the document contract
    its spans assume; an embedding states its model, dimension, normalization,
    composition, and query convention. Nothing is configured, because
    configuration could disagree with the data and the data is what will be
    queried.

    On QUERY, a partition is LOOKED UP from the catalog. Exactly one
    segmentation partition serves queries; several embedding partitions may
    exist over it, one active and the others being built or retired.

    The asymmetry is the point. Ingest cannot place an artifact in a partition
    it does not belong to, and query cannot reach two partitions at once, so
    an incomparable comparison is not merely detected but unwritable.

What resolution refuses to do
-----------------------------
    It does not repair. A unit whose nodes disagree about their segmentation
    is rejected rather than split across partitions, and a query whose
    provider encodes queries differently from the partition's vectors is
    refused rather than served with a warning.

    Both conditions are silent by nature: mixed segmentation yields
    identifiers that resolve to the wrong content, and a mismatched query
    convention degrades retrieval measurably while producing no error at all.
    A warning that a caller may ignore is not proportionate to a failure that
    a caller cannot see.

Determinism
-----------
    Partition keys are pure functions of their components, computed by the
    models themselves. This module composes and compares them; it invents
    nothing, so two deployments given identical artifacts resolve identical
    partitions.

Dependency contract
-------------------
    Imports only the standard library, ``ptfm.core.schemas``, and the
    retrieval ``exceptions``, ``config``, ``models`` and ``interfaces``
    modules. It reaches no store and no backend: resolution is a decision
    about identity, and the catalog it consults is injected.
"""

from __future__ import annotations

from typing import Optional, Sequence

from ptfm.core.knowledge.models import KnowledgeUnit
from ptfm.core.retrieval.config import PartitionConfig
from ptfm.core.retrieval.exceptions import (
    EmbeddingSpaceMismatchError,
    PartitionNotFoundError,
    QueryEncodingMismatchError,
    SegmentationMismatchError,
)
from ptfm.core.retrieval.interfaces import ModelIdentity, PartitionCatalog
from ptfm.core.retrieval.models import EmbeddingPartition, SegmentationPartition
from ptfm.core.schemas import KnowledgeNode, NodeEmbedding

__all__ = [
    "segmentation_for_node",
    "embedding_for_vector",
    "PartitionResolver",
]


# ============================================================================
# DERIVATION FROM ARTIFACTS
# ============================================================================


def segmentation_for_node(node: KnowledgeNode) -> SegmentationPartition:
    """
    Derive the segmentation partition a node belongs to.

    Read from the node rather than from configuration. A node carries the
    segmentation fingerprint it was produced under and the document contract
    version its span assumes, and those are the facts that determine what its
    identifier and offsets mean. Configuration could disagree with them, and
    the node is what will be queried.

    Parameters
    ----------
    node:
        The node whose partition is sought.

    Returns
    -------
    SegmentationPartition
        The partition this node belongs to.
    """
    return SegmentationPartition(
        segmentation_fingerprint=node.segmentation_fingerprint,
        source_schema_version=node.source_schema_version,
    )


def embedding_for_vector(
    embedding: NodeEmbedding,
    segmentation: SegmentationPartition,
) -> EmbeddingPartition:
    """
    Derive the embedding partition a vector belongs to.

    Every component is read from the embedding itself, for the same reason a
    node's segmentation is: each names a way in which two vectors can fail to
    be comparable, and the vector is the authority on its own provenance.

    Parameters
    ----------
    embedding:
        The vector whose partition is sought.
    segmentation:
        The segmentation partition of the node this vector describes. Vectors
        are meaningless without the nodes they belong to, so the embedding
        partition is always subordinate to one.

    Returns
    -------
    EmbeddingPartition
        The partition this vector belongs to.
    """
    return EmbeddingPartition(
        segmentation=segmentation,
        model_identity=embedding.model_identity,
        dimension=embedding.dimension,
        normalized=embedding.normalized,
        composition_id=embedding.composition_id,
        query_encoding=embedding.query_encoding,
    )


# ============================================================================
# RESOLVER
# ============================================================================


class PartitionResolver:
    """
    Resolves the partitions an ingest batch or a query belongs to.

    Stateless apart from injected collaborators, so one instance serves all
    concurrent work.

    Parameters
    ----------
    catalog:
        Records which partitions exist and which serve queries.
    config:
        Partition configuration governing which compatibility checks are
        enforced.
    """

    __slots__ = ("_catalog", "_config")

    def __init__(self, catalog: PartitionCatalog, config: PartitionConfig) -> None:
        self._catalog = catalog
        self._config = config

    # ------------------------------------------------------------------
    # Ingest
    # ------------------------------------------------------------------

    def resolve_for_units(
        self,
        units: Sequence[KnowledgeUnit],
    ) -> tuple[SegmentationPartition, Optional[EmbeddingPartition]]:
        """
        Derive the partitions a batch of knowledge units belongs to.

        Every unit in a batch must agree. A batch spanning two segmentations
        is rejected rather than split, because splitting would silently accept
        a corpus that mixes identifier spaces, and identifiers from one space
        either fail to resolve in another or resolve to unrelated content.

        An unembedded batch is ordinary rather than exceptional: a corpus may
        be segmented now and embedded later, and that intermediate state is
        valid.

        A partly embedded batch is also accepted. Some nodes carry vectors and
        others do not, which happens whenever embedding is resumed after an
        interruption. The vectors present must agree with one another.

        Parameters
        ----------
        units:
            The units to place.

        Returns
        -------
        tuple[SegmentationPartition, Optional[EmbeddingPartition]]
            The segmentation partition, and the embedding partition when any
            unit carries a vector.

        Raises
        ------
        SegmentationMismatchError
            If the units disagree about their segmentation or document
            contract.
        EmbeddingSpaceMismatchError
            If the units' vectors disagree about their embedding space.
        PartitionNotFoundError
            If no unit was supplied.
        """
        if not units:
            raise PartitionNotFoundError(
                "No units were supplied, so no partition can be derived. An "
                "empty batch has no segmentation to belong to."
            )

        segmentation = self._resolve_segmentation(units)
        embedding = self._resolve_embedding(units, segmentation)
        return segmentation, embedding

    def _resolve_segmentation(
        self,
        units: Sequence[KnowledgeUnit],
    ) -> SegmentationPartition:
        """
        Derive the one segmentation partition a batch belongs to.

        Parameters
        ----------
        units:
            The units to place.

        Returns
        -------
        SegmentationPartition
            The partition every unit belongs to.

        Raises
        ------
        SegmentationMismatchError
            If two units disagree.
        """
        first = segmentation_for_node(units[0].node)

        for unit in units[1:]:
            candidate = segmentation_for_node(unit.node)
            if candidate.key == first.key:
                continue

            raise SegmentationMismatchError(
                expected=first.segmentation_fingerprint,
                actual=candidate.segmentation_fingerprint,
                node_id=unit.node.node_id,
                context={
                    "expected_schema_version": first.source_schema_version,
                    "actual_schema_version": candidate.source_schema_version,
                    "detail": (
                        "A batch spanning two segmentations is rejected rather "
                        "than split, because a corpus mixing identifier spaces "
                        "produces identifiers that resolve to unrelated "
                        "content."
                    ),
                },
            )

        return first

    def _resolve_embedding(
        self,
        units: Sequence[KnowledgeUnit],
        segmentation: SegmentationPartition,
    ) -> Optional[EmbeddingPartition]:
        """
        Derive the embedding partition a batch's vectors belong to.

        Parameters
        ----------
        units:
            The units to place.
        segmentation:
            The segmentation partition already resolved for the batch.

        Returns
        -------
        Optional[EmbeddingPartition]
            The partition, or ``None`` when no unit carries a vector.

        Raises
        ------
        EmbeddingSpaceMismatchError
            If two vectors would not be comparable, or a vector describes a
            node from another segmentation.
        """
        first: Optional[EmbeddingPartition] = None

        for unit in units:
            if unit.embedding is None:
                continue

            candidate = embedding_for_vector(unit.embedding, segmentation)

            if first is None:
                first = candidate
                continue

            if candidate.describes_same_space(first):
                continue

            raise EmbeddingSpaceMismatchError(
                f"Vectors in one batch belong to different embedding spaces: "
                f"{first.model_identity!r} at dimension {first.dimension} "
                f"against {candidate.model_identity!r} at dimension "
                f"{candidate.dimension}. A similarity computed across them "
                f"would be arithmetically valid and semantically meaningless.",
                expected_identity=first.model_identity,
                actual_identity=candidate.model_identity,
                node_id=unit.node.node_id,
            )

        return first

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    async def resolve_for_query(self) -> SegmentationPartition:
        """
        Determine which segmentation partition serves queries.

        Exactly one does. The catalog enforces that, so ambiguity is resolved
        where partitions are recorded rather than negotiated here.

        Returns
        -------
        SegmentationPartition
            The active partition.

        Raises
        ------
        PartitionNotFoundError
            If no partition is active, which ordinarily means no corpus has
            been indexed under the current configuration.
        """
        active = await self._catalog.active_segmentation()

        if active is None:
            raise PartitionNotFoundError(
                "No segmentation partition is active, so there is nothing to "
                "query. Either no corpus has been indexed, or an indexed "
                "corpus has not been activated."
            )

        return active

    async def resolve_embedding_for_query(
        self,
        segmentation: SegmentationPartition,
        model: Optional[ModelIdentity],
    ) -> Optional[EmbeddingPartition]:
        """
        Determine which embedding partition serves dense retrieval.

        Verifies before searching that the configured provider produces
        vectors belonging to the active partition's space, and encodes queries
        the way that partition's vectors expect. Both are checked here rather
        than left to surface as results, because a mismatch degrades retrieval
        measurably while producing no error of its own.

        Parameters
        ----------
        segmentation:
            The active segmentation partition.
        model:
            Identity of the model the configured provider embeds with, or
            ``None`` when no provider is configured.

        Returns
        -------
        Optional[EmbeddingPartition]
            The partition dense retrieval should search, or ``None`` when
            dense retrieval is unavailable and the query proceeds without it.

        Raises
        ------
        PartitionNotFoundError
            If no embedding partition is active and configuration requires
            one.
        EmbeddingSpaceMismatchError
            If the provider's vectors would not be comparable with the
            partition's.
        QueryEncodingMismatchError
            If the provider encodes queries differently from the convention
            the partition's vectors were produced under.
        """
        active = await self._catalog.active_embedding(segmentation)

        if active is None:
            if self._config.require_embedding_partition:
                raise PartitionNotFoundError(
                    "No embedding partition is active for the segmentation "
                    "partition serving queries, so dense retrieval cannot "
                    "run. Either the corpus has not been embedded, or an "
                    "embedded partition has not been activated.",
                    partition_key=segmentation.key,
                )
            # Dense retrieval is unavailable, and the query proceeds without
            # it. Recorded by the caller as a skipped arm rather than a
            # failure, since lexical and structural retrieval still serve.
            return None

        if model is None:
            if self._config.require_embedding_partition:
                raise PartitionNotFoundError(
                    "An embedding partition is active but no embedding "
                    "provider is configured, so its vectors cannot be "
                    "searched.",
                    partition_key=active.key,
                )
            return None

        self._verify_space(active, model)
        self._verify_query_encoding(active, model)
        return active

    def _verify_space(
        self,
        partition: EmbeddingPartition,
        model: ModelIdentity,
    ) -> None:
        """
        Ensure the provider's vectors belong to the partition's space.

        Parameters
        ----------
        partition:
            The active embedding partition.
        model:
            Identity of the configured provider's model.

        Returns
        -------
        None

        Raises
        ------
        EmbeddingSpaceMismatchError
            If the model, dimension, or normalization differs.
        """
        if (
            model.identity == partition.model_identity
            and model.dimension == partition.dimension
            and model.normalized == partition.normalized
        ):
            return

        raise EmbeddingSpaceMismatchError(
            f"The configured provider embeds with {model.identity!r} at "
            f"dimension {model.dimension} "
            f"({'normalized' if model.normalized else 'unnormalized'}), while "
            f"the active partition holds vectors from "
            f"{partition.model_identity!r} at dimension {partition.dimension} "
            f"({'normalized' if partition.normalized else 'unnormalized'}). "
            f"A query vector from one space compared against the other "
            f"produces a ranking that is arithmetically valid and "
            f"semantically meaningless.",
            expected_identity=partition.model_identity,
            actual_identity=model.identity,
            context={
                "expected_dimension": partition.dimension,
                "actual_dimension": model.dimension,
                "expected_normalized": partition.normalized,
                "actual_normalized": model.normalized,
                "partition_key": partition.key,
            },
        )

    def _verify_query_encoding(
        self,
        partition: EmbeddingPartition,
        model: ModelIdentity,
    ) -> None:
        """
        Ensure the provider encodes queries as the partition's vectors expect.

        The check that exists because its absence is invisible. Several widely
        used retrieval models are trained asymmetrically, expecting a distinct
        transformation on the query side; embedding a query as though it were
        a passage returns plausible results in a worse order, and nothing
        anywhere reports a fault.

        Unconditional, with no setting to disable it. A switch could only ever
        be used to hide a failure the operator cannot otherwise detect.

        Parameters
        ----------
        partition:
            The active embedding partition.
        model:
            Identity of the configured provider's model.

        Returns
        -------
        None

        Raises
        ------
        QueryEncodingMismatchError
            If the conventions differ.
        """
        if partition.accepts_query_encoding(model.query_encoding):
            return

        raise QueryEncodingMismatchError(
            expected=partition.query_encoding,
            actual=model.query_encoding,
            context={"partition_key": partition.key},
        )

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    async def register(
        self,
        segmentation: SegmentationPartition,
        embedding: Optional[EmbeddingPartition],
        *,
        activate: bool,
    ) -> None:
        """
        Record that a partition exists, and optionally activate it.

        Activation is the atomic switch a migration turns. Until it is turned,
        a partition can be written to without serving anything, which is what
        lets a corpus be re-segmented or re-embedded while the previous one
        continues to answer queries.

        Parameters
        ----------
        segmentation:
            The segmentation partition to record.
        embedding:
            The embedding partition to record, when one exists.
        activate:
            Whether these partitions should serve queries.

        Returns
        -------
        None

        Raises
        ------
        PartitionError
            If the partitions could not be recorded.
        """
        await self._catalog.register(segmentation, embedding, activate=activate)

    def __repr__(self) -> str:
        """
        Return a representation naming the catalog in use.

        Returns
        -------
        str
            Diagnostic representation.
        """
        return f"PartitionResolver(catalog={self._catalog.name!r})"
