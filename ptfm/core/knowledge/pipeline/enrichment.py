"""
PTFM Enrichment Stage

File:
    ptfm/core/knowledge/pipeline/enrichment.py

Description:
    Applies domain metadata enrichment to knowledge nodes.

    This is the only point in the module at which domain knowledge enters. The
    Core contains none, and every enricher arrives through injection.

Additive by construction, and verified to be so
-----------------------------------------------
    An enricher receives an immutable node and returns metadata only. It is
    never handed a mutable node, and its return value cannot reach text, span,
    block references, or identity.

    That is a structural guarantee, but this stage does not rely on structure
    alone. After enrichment it verifies that identity, text, and span are
    unchanged, so that a defect in this stage itself cannot silently alter
    what a citation resolves to. The guarantee that matters most is the one
    least worth taking on trust.

Failure degrades rather than losing a node
------------------------------------------
    Enrichment produces metadata, so a failure loses metadata. Discarding a
    correctly segmented node because a domain lookup timed out would trade a
    small loss for a large one, and the node remains fully retrievable
    without the enricher's contribution.

    Configuration may make failure fatal where a deployment treats domain
    metadata as mandatory, but the default degrades.

Key ownership is declared in advance
------------------------------------
    An enricher declares the keys it may contribute. Two enrichers claiming
    one key are rejected at configuration rather than resolved by whichever
    happens to run last, which would make a node's metadata depend on
    registration order.

Execution contract
------------------
    Asynchronous. An enricher may consult a database, a service, or a model,
    and each node is enriched under a deadline.

Dependency contract
-------------------
    Imports only the standard library, ``ptfm.core.schemas`` and the knowledge
    ``exceptions``, ``config`` and ``interfaces`` modules.
"""

from __future__ import annotations

import asyncio
from typing import Final, Mapping, Sequence

from ptfm.core.knowledge.config import EnrichmentConfig, EnrichmentFailurePolicy
from ptfm.core.knowledge.exceptions import (
    EnricherContractViolationError,
    EnricherFailedError,
    EnricherTimeoutError,
    EnrichmentError,
    KnowledgeConfigurationError,
    PTFMKnowledgeError,
)
from ptfm.core.knowledge.interfaces import KnowledgeEnricher, StructuredLogger
from ptfm.core.schemas import KnowledgeNode

__all__ = ["EnrichmentStage"]

#: Fields an enricher must never be able to affect. Verified after every
#: enrichment, so that a defect in this stage cannot alter what a citation
#: resolves to.
_PROVENANCE_FIELDS: Final[tuple[str, ...]] = (
    "node_id",
    "document_id",
    "text",
    "span",
    "block_ids",
    "segmentation_fingerprint",
    "source_text_hash",
)


class EnrichmentStage:
    """
    Applies configured enrichers to a node's metadata.

    Parameters
    ----------
    enrichers:
        The enrichers available. Which of them run is decided per node by
        configuration, so a staged rollout needs no change to the injected
        set.
    config:
        Enrichment configuration governing deadlines, failure policy, and
        metadata bounds.
    logger:
        Structured logger. Enricher failures are recorded rather than raised
        under the degrading policy, so without a log they would be invisible.

    Raises
    ------
    KnowledgeConfigurationError
        If two enrichers declare the same metadata key, which would make a
        node's metadata depend on the order they happened to run in.
    """

    __slots__ = ("_enrichers", "_config", "_logger")

    def __init__(
        self,
        enrichers: Sequence[KnowledgeEnricher],
        config: EnrichmentConfig,
        logger: StructuredLogger,
    ) -> None:
        _verify_key_ownership(enrichers)
        self._enrichers = tuple(enrichers)
        self._config = config
        self._logger = logger

    async def enrich(self, node: KnowledgeNode) -> KnowledgeNode:
        """
        Produce a node carrying the metadata its enrichers contributed.

        Parameters
        ----------
        node:
            The node to enrich. Never modified; a new node carrying the merged
            metadata is returned.

        Returns
        -------
        KnowledgeNode
            The node with enricher metadata attached, or the node unchanged
            when enrichment is disabled or contributed nothing.

        Raises
        ------
        EnrichmentError
            If an enricher failed and the configured policy is fatal.
        EnricherContractViolationError
            If an enricher returned something its contract disallows, which is
            fatal regardless of policy.
        """
        if not self._config.enabled or not self._enrichers:
            return node

        metadata = dict(node.metadata)

        for enricher in self._enrichers:
            contributed = await self._invoke(enricher, node)
            if contributed:
                metadata.update(contributed)

        if metadata == node.metadata:
            return node

        _verify_metadata_bounds(metadata, node, self._config)
        enriched = node.model_copy(update={"metadata": metadata})
        _verify_provenance_unchanged(node, enriched)
        return enriched

    # ------------------------------------------------------------------
    # Invocation
    # ------------------------------------------------------------------

    async def _invoke(
        self,
        enricher: KnowledgeEnricher,
        node: KnowledgeNode,
    ) -> Mapping[str, str]:
        """
        Run one enricher under its deadline, applying the failure policy.

        Parameters
        ----------
        enricher:
            The enricher to run.
        node:
            The node being enriched.

        Returns
        -------
        Mapping[str, str]
            The enricher's contribution, empty where it failed under the
            degrading policy.

        Raises
        ------
        EnrichmentError
            If the enricher failed and the policy is fatal.
        EnricherContractViolationError
            If the enricher returned something its contract disallows. Fatal
            regardless of policy, since a contract violation indicates a
            defective plugin rather than a transient condition, and admitting
            its output would contaminate the knowledge base.
        """
        if not _enricher_selected(enricher, self._config):
            return {}

        try:
            contributed = await asyncio.wait_for(
                enricher.enrich(node), timeout=self._config.timeout_seconds
            )
        except TimeoutError as exc:
            return self._handle_failure(
                EnricherTimeoutError(
                    enricher_name=enricher.name,
                    timeout_seconds=self._config.timeout_seconds,
                    document_id=node.document_id,
                    node_id=node.node_id,
                ),
                cause=exc,
                node=node,
            )
        except PTFMKnowledgeError as error:
            if error.node_id is None:
                error.node_id = node.node_id
            if error.document_id is None:
                error.document_id = node.document_id
            return self._handle_failure(error, cause=None, node=node)
        except Exception as exc:
            return self._handle_failure(
                EnricherFailedError(
                    f"Enricher {enricher.name!r} raised "
                    f"{type(exc).__name__}: {exc}",
                    enricher_name=enricher.name,
                    document_id=node.document_id,
                    node_id=node.node_id,
                ),
                cause=exc,
                node=node,
            )

        _verify_contribution(contributed, enricher, node)
        return contributed

    def _handle_failure(
        self,
        error: PTFMKnowledgeError,
        *,
        cause: BaseException | None,
        node: KnowledgeNode,
    ) -> Mapping[str, str]:
        """
        Apply the configured policy to an enricher failure.

        Parameters
        ----------
        error:
            The typed failure.
        cause:
            The originating exception, chained when present.
        node:
            The node being enriched.

        Returns
        -------
        Mapping[str, str]
            An empty contribution, under the degrading policy.

        Raises
        ------
        PTFMKnowledgeError
            The supplied failure, under the fatal policy.
        """
        if self._config.failure_policy is EnrichmentFailurePolicy.FAIL:
            if cause is not None:
                raise error from cause
            raise error

        # Degrading. Logged rather than raised, because a failure that is
        # neither raised nor recorded is a silent loss of metadata that no
        # operator could discover.
        self._logger.warning(
            "knowledge.enrichment.degraded",
            document_id=node.document_id,
            node_id=node.node_id,
            error_code=error.error_code,
            error_type=type(error).__name__,
            message=error.message,
        )
        return {}

    def __repr__(self) -> str:
        """
        Return a representation naming the configured enrichers.

        Returns
        -------
        str
            Diagnostic representation.
        """
        names = [enricher.name for enricher in self._enrichers]
        return (
            f"EnrichmentStage(enrichers={names!r}, "
            f"policy={self._config.failure_policy.value})"
        )


# ============================================================================
# VERIFICATION
# ============================================================================


def _verify_key_ownership(enrichers: Sequence[KnowledgeEnricher]) -> None:
    """
    Ensure no two enrichers claim the same metadata key.

    A collision resolved by execution order would make a node's metadata
    depend on the order enrichers happened to be injected in, which is neither
    deterministic nor discoverable.

    Parameters
    ----------
    enrichers:
        The enrichers to verify.

    Returns
    -------
    None

    Raises
    ------
    KnowledgeConfigurationError
        If two enrichers declare the same key, or an enricher name repeats.
    """
    owners: dict[str, str] = {}
    seen_names: set[str] = set()

    for enricher in enrichers:
        if enricher.name in seen_names:
            raise KnowledgeConfigurationError(
                f"Duplicate enricher name {enricher.name!r}. Names identify "
                f"enrichers in diagnostics and in the enabled-enricher "
                f"selection, so they must be unique.",
                context={"enricher_name": enricher.name},
            )
        seen_names.add(enricher.name)

        for key in enricher.metadata_keys:
            owner = owners.get(key)
            if owner is not None:
                raise KnowledgeConfigurationError(
                    f"Enrichers {owner!r} and {enricher.name!r} both declare "
                    f"metadata key {key!r}. Resolving the collision by "
                    f"execution order would make a node's metadata depend on "
                    f"injection order.",
                    context={"metadata_key": key},
                )
            owners[key] = enricher.name


def _enricher_selected(
    enricher: KnowledgeEnricher,
    config: EnrichmentConfig,
) -> bool:
    """
    Report whether a configured selection includes an enricher.

    An empty selection runs every injected enricher, which is the ordinary
    case; naming a subset is for staged rollout.

    Parameters
    ----------
    enricher:
        The enricher to test.
    config:
        Enrichment configuration.

    Returns
    -------
    bool
        ``True`` when the enricher should run.
    """
    if not config.enabled_enrichers:
        return True
    return enricher.name in config.enabled_enrichers


def _verify_contribution(
    contributed: object,
    enricher: KnowledgeEnricher,
    node: KnowledgeNode,
) -> None:
    """
    Ensure an enricher's return value satisfies its contract.

    An enricher may contribute string-valued metadata within its declared
    keys, and nothing else. This is the boundary that keeps domain logic out
    of provenance: a return value able to carry structured objects, or to
    reach keys the enricher never declared, would let a plugin place arbitrary
    content into a domain-agnostic schema.

    Parameters
    ----------
    contributed:
        The enricher's return value.
    enricher:
        The enricher that produced it.
    node:
        The node being enriched.

    Returns
    -------
    None

    Raises
    ------
    EnricherContractViolationError
        If the value is not a mapping, carries a non-string key or value, or
        uses a key the enricher did not declare.
    """
    if not isinstance(contributed, Mapping):
        raise EnricherContractViolationError(
            f"Enricher {enricher.name!r} returned "
            f"{type(contributed).__name__} rather than a mapping of metadata.",
            enricher_name=enricher.name,
            document_id=node.document_id,
            node_id=node.node_id,
        )

    declared = enricher.metadata_keys

    for key, value in contributed.items():
        if not isinstance(key, str):
            raise EnricherContractViolationError(
                f"Enricher {enricher.name!r} returned a non-string metadata "
                f"key: {key!r}.",
                enricher_name=enricher.name,
                document_id=node.document_id,
                node_id=node.node_id,
            )
        if not isinstance(value, str):
            raise EnricherContractViolationError(
                f"Enricher {enricher.name!r} returned a non-string value for "
                f"key {key!r}: {type(value).__name__}. Structured values would "
                f"place a domain object into a domain-agnostic schema.",
                enricher_name=enricher.name,
                offending_key=key,
                document_id=node.document_id,
                node_id=node.node_id,
            )
        if key not in declared:
            raise EnricherContractViolationError(
                f"Enricher {enricher.name!r} contributed key {key!r}, which it "
                f"did not declare. Undeclared keys defeat the collision check "
                f"that keeps metadata independent of execution order.",
                enricher_name=enricher.name,
                offending_key=key,
                document_id=node.document_id,
                node_id=node.node_id,
            )


def _verify_metadata_bounds(
    metadata: Mapping[str, str],
    node: KnowledgeNode,
    config: EnrichmentConfig,
) -> None:
    """
    Ensure accumulated metadata stays within its configured bounds.

    Bounds a defective enricher, which could otherwise attach unbounded
    metadata to every node in a corpus. A long value in particular indicates
    an enricher attempting to store content, which belongs in the node's text
    where it carries a span.

    Parameters
    ----------
    metadata:
        The merged metadata.
    node:
        The node being enriched.
    config:
        Enrichment configuration.

    Returns
    -------
    None

    Raises
    ------
    EnrichmentError
        If the metadata exceeds a configured bound.
    """
    if len(metadata) > config.max_metadata_keys:
        raise EnrichmentError(
            f"Node accumulated {len(metadata)} metadata keys, exceeding the "
            f"configured limit of {config.max_metadata_keys}.",
            document_id=node.document_id,
            node_id=node.node_id,
            context={"key_count": len(metadata)},
        )

    for key, value in metadata.items():
        if len(value) <= config.max_metadata_value_chars:
            continue
        raise EnrichmentError(
            f"Metadata value for key {key!r} is {len(value)} characters, "
            f"exceeding the configured limit of "
            f"{config.max_metadata_value_chars}. Metadata carries attributes, "
            f"not content; content belongs in the node text, where it carries "
            f"a span.",
            document_id=node.document_id,
            node_id=node.node_id,
            context={"metadata_key": key, "value_chars": len(value)},
        )


def _verify_provenance_unchanged(
    original: KnowledgeNode,
    enriched: KnowledgeNode,
) -> None:
    """
    Ensure enrichment altered nothing but metadata.

    The enricher contract makes provenance alteration structurally impossible,
    so this verifies the stage itself rather than the plugin. The guarantee
    that matters most is the one least worth taking on trust: if enrichment
    could change a node's identity or text, every citation into that corpus
    would silently mean something else.

    Parameters
    ----------
    original:
        The node before enrichment.
    enriched:
        The node after enrichment.

    Returns
    -------
    None

    Raises
    ------
    EnrichmentError
        If any provenance field changed.
    """
    for field in _PROVENANCE_FIELDS:
        if getattr(original, field) == getattr(enriched, field):
            continue
        raise EnrichmentError(
            f"Enrichment altered provenance field {field!r}. Enrichment is "
            f"additive by contract, so this indicates a defect in the "
            f"enrichment stage rather than in any enricher.",
            document_id=original.document_id,
            node_id=original.node_id,
            context={"altered_field": field},
        )
