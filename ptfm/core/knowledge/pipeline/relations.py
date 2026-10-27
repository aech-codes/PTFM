"""
PTFM Relation Derivation Stage

File:
    ptfm/core/knowledge/pipeline/relations.py

Description:
    Derives structural relations between the knowledge nodes of one document.

Relations are facts, not judgements
-----------------------------------
    Every relation produced here is recoverable from the source document's
    block tree: reading order, section containment, and co-membership of a
    table, list, or page.

    Nothing semantic is derived. There is no similarity, entailment, or
    contradiction relation, because those are verification judgements and a
    judgement recorded as though it were structure would enter the knowledge
    base outside the calibration layer that must govern it.

    Both Retrieval and Verification consume these relations, as peers.
    Retrieval expands a result set along document structure; Verification
    traverses structure while reasoning symbolically. Neither reaches them
    through the other.

Co-membership is chained, not cliqued
-------------------------------------
    A table divided into five hundred nodes contains one hundred and
    twenty-five thousand pairs. Emitting a relation for each would make
    relation count grow quadratically with table size, so a single large table
    would dominate a corpus.

    Co-membership is therefore emitted between CONSECUTIVE members, with the
    identifier of the shared container carried as a relation attribute. The
    pairwise statement remains true, the full group is recoverable either by
    following the chain or by querying the attribute, and the count grows
    linearly.

Direction
---------
    Directed relations are emitted in one direction by default. A consumer
    inverts a relation directly, so storing both would double the count for no
    additional information. Where a sink cannot traverse an edge backwards,
    configuration enables the inverse.

Execution contract
------------------
    Synchronous and CPU-bound. The engine executes it in a worker thread; it
    must never be invoked from a coroutine.

Dependency contract
-------------------
    Imports only the standard library, ``ptfm.core.schemas`` and the knowledge
    ``exceptions``, ``config`` and ``models`` modules.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Final, Mapping, Optional, Sequence

from ptfm.core.knowledge.config import RelationConfig
from ptfm.core.knowledge.exceptions import RelationError
from ptfm.core.schemas import (
    BlockType,
    NodeType,
    DocumentBlock,
    KnowledgeNode,
    NodeRelation,
    RawDocument,
    RelationType,
)

__all__ = ["RelationDeriver"]

#: Container block types whose members are related by co-membership, mapped to
#: the relation that expresses it.
_CO_MEMBERSHIP_CONTAINERS: Final[Mapping[BlockType, RelationType]] = {
    BlockType.TABLE: RelationType.SAME_TABLE,
    BlockType.LIST: RelationType.SAME_LIST,
}

#: Attribute naming the container shared by two co-members. Carrying it makes
#: the full group recoverable without emitting a relation for every pair.
_CONTAINER_ATTRIBUTE: Final[str] = "container_block_id"

#: Attribute naming the page shared by two nodes.
_PAGE_ATTRIBUTE: Final[str] = "page_number"


class RelationDeriver:
    """
    Derives structural relations among the nodes of one document.

    Stateless: all per-document state is local to a call, so one instance
    serves all concurrent workers.

    Parameters
    ----------
    config:
        Relation configuration governing which relation families are derived,
        whether inverses are emitted, and how many outgoing relations one node
        may have.
    """

    __slots__ = ("_config",)

    def __init__(self, config: RelationConfig) -> None:
        self._config = config

    def derive(
        self,
        document: RawDocument,
        nodes: Sequence[KnowledgeNode],
    ) -> tuple[NodeRelation, ...]:
        """
        Derive every configured relation among a document's nodes.

        Blocking and CPU-bound by contract. The engine executes this in a
        worker thread; it must never be invoked from a coroutine.

        Parameters
        ----------
        document:
            The source document, consulted for its block tree.
        nodes:
            The document's nodes, in document order.

        Returns
        -------
        tuple[NodeRelation, ...]
            Relations in derivation order: sequence, then hierarchy, then
            co-membership.

        Raises
        ------
        RelationError
            If a node accumulates more outgoing relations than configured.
        """
        if not self._config.enabled or len(nodes) < 2:
            # A single node has nothing to relate to, and no relation is
            # meaningful across documents: relations derive from one block
            # tree.
            return ()

        relations: list[NodeRelation] = []

        if self._config.derive_sequence:
            relations.extend(_derive_sequence(document, nodes))

        if self._config.derive_hierarchy:
            relations.extend(_derive_hierarchy(document, nodes))

        if self._config.derive_co_membership:
            relations.extend(_derive_container_membership(document, nodes))
            relations.extend(_derive_page_membership(document, nodes))

        if self._config.emit_inverse_relations:
            relations.extend(_derive_inverses(relations))

        self._enforce_relation_bound(document, relations)
        return tuple(relations)

    def _enforce_relation_bound(
        self,
        document: RawDocument,
        relations: Sequence[NodeRelation],
    ) -> None:
        """
        Ensure no node accumulates more outgoing relations than configured.

        Chained co-membership keeps the ordinary count to a handful per node,
        so this bound fires only on genuine pathology. It raises rather than
        truncating, because silently discarding relations would leave a graph
        that is quietly incomplete and no consumer could tell.

        Parameters
        ----------
        document:
            The source document, for error provenance.
        relations:
            The derived relations.

        Returns
        -------
        None

        Raises
        ------
        RelationError
            If a node exceeds the configured bound.
        """
        counts: dict[str, int] = defaultdict(int)
        for relation in relations:
            counts[relation.source_node_id] += 1

        limit = self._config.max_relations_per_node
        for node_id, count in counts.items():
            if count <= limit:
                continue
            raise RelationError(
                f"Node {node_id!r} accumulated {count} outgoing relations, "
                f"exceeding the configured limit of {limit}. Relations are "
                f"not truncated, because a silently incomplete graph cannot "
                f"be distinguished from a complete one.",
                document_id=document.document_id,
                node_id=node_id,
                context={"relation_count": count, "limit": limit},
            )

    def __repr__(self) -> str:
        """
        Return a representation naming the configured relation families.

        Returns
        -------
        str
            Diagnostic representation.
        """
        families = [
            name
            for name, enabled in (
                ("sequence", self._config.derive_sequence),
                ("hierarchy", self._config.derive_hierarchy),
                ("co_membership", self._config.derive_co_membership),
            )
            if enabled
        ]
        return f"RelationDeriver(families={families!r})"


# ============================================================================
# DERIVATION
# ============================================================================


def _derive_sequence(
    document: RawDocument,
    nodes: Sequence[KnowledgeNode],
) -> list[NodeRelation]:
    """
    Relate each node to the one that follows it in reading order.

    Reading order is what lets a consumer expand a retrieved node into the
    passage surrounding it, which is frequently what makes an isolated chunk
    interpretable.

    Parameters
    ----------
    document:
        The source document.
    nodes:
        The document's nodes, in document order.

    Returns
    -------
    list[NodeRelation]
        One relation per adjacent pair.
    """
    return [
        NodeRelation(
            source_node_id=current.node_id,
            target_node_id=following.node_id,
            relation_type=RelationType.PRECEDES,
            document_id=document.document_id,
        )
        for current, following in zip(nodes, nodes[1:], strict=False)
    ]


def _derive_hierarchy(
    document: RawDocument,
    nodes: Sequence[KnowledgeNode],
) -> list[NodeRelation]:
    """
    Relate each node to the node holding the heading that introduces it.

    A node's heading path names its enclosing sections. Where a heading node
    exists for the innermost of them, the two are related, so that a consumer
    holding a retrieved node can reach the section it belongs to.

    Sections are identified by their full heading path rather than by heading
    text alone, so that two sections sharing a title under different parents
    remain distinct. Where a document repeats an identical path, the most
    recent heading owns the section, which matches how a reader would resolve
    it.

    Parameters
    ----------
    document:
        The source document.
    nodes:
        The document's nodes, in document order.

    Returns
    -------
    list[NodeRelation]
        One relation per node having a resolvable enclosing section.
    """
    # Path of a heading node, including its own text, mapped to that node.
    owners: dict[tuple[str, ...], str] = {}
    relations: list[NodeRelation] = []

    for node in nodes:
        if node.heading_path:
            owner = owners.get(node.heading_path)
            # A heading is not its own parent section.
            if owner is not None and owner != node.node_id:
                relations.append(
                    NodeRelation(
                        source_node_id=node.node_id,
                        target_node_id=owner,
                        relation_type=RelationType.PARENT_SECTION,
                        document_id=document.document_id,
                    )
                )

        if node.node_type is NodeType.HEADING:
            title = node.text.strip()
            if title:
                owners[(*node.heading_path, title)] = node.node_id

    return relations


def _derive_container_membership(
    document: RawDocument,
    nodes: Sequence[KnowledgeNode],
) -> list[NodeRelation]:
    """
    Relate consecutive nodes belonging to one table or list.

    Chained rather than cliqued. A table divided into five hundred nodes
    contains one hundred and twenty-five thousand pairs, so emitting a
    relation for each would make relation count grow quadratically with table
    size and let a single large table dominate a corpus.

    The shared container's identifier travels as a relation attribute, so the
    full group is recoverable either by following the chain or by querying the
    attribute, while the count grows linearly.

    Parameters
    ----------
    document:
        The source document.
    nodes:
        The document's nodes, in document order.

    Returns
    -------
    list[NodeRelation]
        One relation per consecutive pair sharing a container.
    """
    blocks_by_id = {block.block_id: block for block in document.blocks}
    relations: list[NodeRelation] = []

    # Most recent node seen for each container, so that only consecutive
    # members are linked.
    previous: dict[str, str] = {}

    for node in nodes:
        for container_id, relation_type in _containers_of(node, blocks_by_id):
            earlier = previous.get(container_id)
            if earlier is not None:
                relations.append(
                    NodeRelation(
                        source_node_id=earlier,
                        target_node_id=node.node_id,
                        relation_type=relation_type,
                        document_id=document.document_id,
                        attributes={_CONTAINER_ATTRIBUTE: container_id},
                    )
                )
            previous[container_id] = node.node_id

    return relations


def _containers_of(
    node: KnowledgeNode,
    blocks_by_id: Mapping[str, DocumentBlock],
) -> list[tuple[str, RelationType]]:
    """
    Find the co-membership containers enclosing a node.

    Walks the block ancestry of the node's first block. Only the nearest
    container of each kind is reported: a row nested in a table within a table
    belongs most immediately to the inner one, and relating it to both would
    assert a membership the reader would not recognize.

    Parameters
    ----------
    node:
        The node whose containers are sought.
    blocks_by_id:
        The document's blocks, keyed by identifier.

    Returns
    -------
    list[tuple[str, RelationType]]
        Container identifiers with the relation each implies.
    """
    if not node.block_ids:
        return []

    found: dict[RelationType, str] = {}
    block = blocks_by_id.get(node.block_ids[0])

    while block is not None:
        relation_type = _CO_MEMBERSHIP_CONTAINERS.get(block.block_type)
        if relation_type is not None and relation_type not in found:
            found[relation_type] = block.block_id
        block = (
            blocks_by_id.get(block.parent_id)
            if block.parent_id is not None
            else None
        )

    return [(container_id, kind) for kind, container_id in found.items()]


def _derive_page_membership(
    document: RawDocument,
    nodes: Sequence[KnowledgeNode],
) -> list[NodeRelation]:
    """
    Relate consecutive nodes originating on one page.

    Chained for the same reason as container membership: a dense page can hold
    many nodes, and a clique over them would grow quadratically.

    A node spanning several pages is a member of each, since it genuinely
    originates on all of them.

    Parameters
    ----------
    document:
        The source document.
    nodes:
        The document's nodes, in document order.

    Returns
    -------
    list[NodeRelation]
        One relation per consecutive pair sharing a page.
    """
    relations: list[NodeRelation] = []
    previous: dict[int, str] = {}

    for node in nodes:
        for page in node.page_numbers:
            earlier = previous.get(page)
            if earlier is not None:
                relations.append(
                    NodeRelation(
                        source_node_id=earlier,
                        target_node_id=node.node_id,
                        relation_type=RelationType.SAME_PAGE,
                        document_id=document.document_id,
                        attributes={_PAGE_ATTRIBUTE: str(page)},
                    )
                )
            previous[page] = node.node_id

    return relations


def _derive_inverses(
    relations: Sequence[NodeRelation],
) -> list[NodeRelation]:
    """
    Produce the reverse of every directed relation.

    Emitted only where configured. A consumer can invert a relation directly,
    so storing both directions doubles the count for no additional
    information; the option exists for sinks unable to traverse an edge
    backwards.

    Symmetric relations already hold in both directions and are skipped.

    Parameters
    ----------
    relations:
        The relations derived so far.

    Returns
    -------
    list[NodeRelation]
        The inverse of each directed relation.
    """
    inverses: list[NodeRelation] = []

    for relation in relations:
        inverse: Optional[NodeRelation] = relation.inverse()
        if inverse is not None:
            inverses.append(inverse)

    return inverses
