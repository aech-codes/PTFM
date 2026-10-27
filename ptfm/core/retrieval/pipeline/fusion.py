"""
PTFM Rank Fusion

File:
    ptfm/core/retrieval/pipeline/fusion.py

Description:
    Combines the orderings produced by several retrieval arms into one
    ranking, and collapses each node to a single entry carrying every arm's
    observation of it.

One node is one finding
----------------------
    A node found by the dense, sparse, and structural arms is ONE passage
    corroborated by three methods, not three passages. Fusion therefore emits
    one entry per distinct node with three signals on it, never three entries.

    This is the module's central responsibility and it is not presentational.
    Corroboration is counted downstream by distinct anchors, so a node
    duplicated across arms would inflate that count in precisely the quantity
    the system exists to calibrate. Deduplication here is what makes the count
    downstream correct.

Ranks, never scores
-------------------
    Reciprocal rank fusion uses each arm's ORDINAL POSITION and discards its
    score entirely.

    That is the reason it is the default, and the reason it is safe. A cosine
    similarity, a lexical relevance value, and a structural distance occupy
    different ranges with different meanings; combining them numerically
    produces a value that resembles a measurement and is not one. Rank fusion
    cannot make that mistake, because it never reads a score.

    The scores are not discarded from the RESULT. Each arm's raw score travels
    on its own signal, with the scale it lives on, so a consumer can inspect
    what each arm observed. Fusion simply does not arithmetically combine them.

    The fused value is likewise ordinal. It is not a probability, and its scale
    says so.

Determinism
-----------
    Identical inputs produce an identical ranking, and the order in which the
    arms happened to finish cannot affect it. Two nodes with equal fused value
    are ordered by identifier, which is stable across processes and index
    rebuilds.

    Without that guarantee, the same query against an unchanged corpus would
    rank differently according to network timing, and no result could be
    reproduced for audit.

Dependency contract
-------------------
    Imports only the standard library and the retrieval ``exceptions``,
    ``config`` and ``models`` modules, plus the shared schema enums. It touches
    no store, performs no I/O, and is synchronous.
"""

from __future__ import annotations

from typing import Final, Iterable, Mapping, Sequence

from ptfm.core.retrieval.config import FusionConfig
from ptfm.core.retrieval.exceptions import FusionError
from ptfm.core.retrieval.interfaces import FusionStrategy
from ptfm.core.retrieval.models import ArmResult, Candidate, FusedCandidate
from ptfm.core.schemas import DiscoveryMethod, RetrievalSignal, SignalScale

__all__ = ["RECIPROCAL_RANK_FUSION_NAME", "ReciprocalRankFusion"]

#: Selector matching ``FusionConfig.strategy``.
RECIPROCAL_RANK_FUSION_NAME: Final[str] = "reciprocal_rank"

#: Weight applied to an arm for which configuration names none. Unity, so that
#: an unweighted deployment is the plain reciprocal rank formula and adding a
#: weight for one arm does not implicitly rescale the others.
_DEFAULT_WEIGHT: Final[float] = 1.0


class ReciprocalRankFusion(FusionStrategy):
    """
    Combines arm orderings by reciprocal rank.

    For each node, the fused value is the sum over the arms that found it of

        weight(method) / (rank_constant + rank)

    where ``rank`` is that arm's zero-based position for the node.

    Properties that matter
    ----------------------
    Scale-free
        No score is read, so arms whose scores are incomparable combine
        correctly. This is the property that makes hybrid retrieval expressible
        without first calibrating every arm's scale against every other.

    Agreement-rewarding
        A node found by several arms accumulates a contribution from each, so
        agreement across methods raises a node without any arm's score being
        trusted in absolute terms.

    Bounded influence
        The rank constant damps the top of each arm's list. With a constant of
        sixty, first place contributes ``1/61`` and tenth ``1/70``: close
        enough that one arm's confident wrong answer cannot dominate, far
        enough apart that a genuine ordering still tells.

    Deterministic
        A pure function of the candidates. Arm completion order cannot affect
        the outcome, and ties break on identifier.

    Weighting
        ``FusionConfig.method_weights`` scales a method's contribution. It
        weights the RANK contribution, never a score, so a weighted deployment
        is still scale-free. A weight of zero excludes a method from the
        ranking while leaving its observation on the result, which is how an
        arm can inform a consumer without influencing order.

    Parameters
    ----------
    config:
        Fusion configuration supplying the rank constant and any per-method
        weights.

    Raises
    ------
    FusionError
        If a configured weight names no known discovery method, which would
        otherwise be a silently ineffective setting.
    """

    __slots__ = ("_rank_constant", "_weights")

    def __init__(self, config: FusionConfig) -> None:
        self._rank_constant = config.rank_constant
        self._weights = _validated_weights(config.method_weights)

    @property
    def name(self) -> str:
        """
        Stable identifier of this strategy.

        Returns
        -------
        str
            Implementation identifier.
        """
        return RECIPROCAL_RANK_FUSION_NAME

    def fuse(
        self,
        results: Sequence[ArmResult],
        top_k: int,
    ) -> Sequence[FusedCandidate]:
        """
        Combine several arms' orderings into one ranking.

        Parameters
        ----------
        results:
            One result per arm that executed. An arm that failed is absent, so
            fusion operates on what succeeded rather than on placeholders. An
            arm that ran and found nothing contributes nothing and lowers no
            node.
        top_k:
            Largest number of results to return.

        Returns
        -------
        Sequence[FusedCandidate]
            Distinct nodes in fused order, each carrying every signal that
            contributed to it. A node found by several arms appears ONCE.

        Raises
        ------
        FusionError
            If two arms report the same discovery method, which the signal
            vocabulary cannot represent, or if an arm's ordering is malformed.
        """
        if top_k < 1:
            raise FusionError(
                f"Cannot fuse to {top_k} results; at least one is required."
            )

        self._verify_distinct_methods(results)

        observations: dict[str, list[Candidate]] = {}
        totals: dict[str, float] = {}

        for result in results:
            weight = self._weights.get(result.method, _DEFAULT_WEIGHT)

            for candidate in result.candidates:
                observations.setdefault(candidate.node_id, []).append(candidate)
                totals[candidate.node_id] = totals.get(
                    candidate.node_id, 0.0
                ) + weight / (self._rank_constant + candidate.rank)

        # Descending fused value, then ascending identifier. The identifier
        # tie-break is what makes the ranking reproducible: without it, two
        # nodes of equal value would be ordered by whichever arm happened to
        # report first, which varies with network timing.
        ordered = sorted(
            totals,
            key=lambda node_id: (-totals[node_id], node_id),
        )

        return tuple(
            FusedCandidate(
                node_id=node_id,
                signals=_ordered_signals(observations[node_id]),
                fused_rank=position,
                fused_score=totals[node_id],
                fused_scale=SignalScale.RECIPROCAL_RANK,
            )
            for position, node_id in enumerate(ordered[:top_k])
        )

    def _verify_distinct_methods(self, results: Sequence[ArmResult]) -> None:
        """
        Ensure no two arms report the same discovery method.

        A signal records the METHOD that produced it, and a fused candidate
        admits at most one signal per method. Two arms sharing a method
        therefore cannot both be represented on a node they both found.

        Refused rather than resolved by discarding one observation. Dropping an
        arm's finding would leave the discovery history incomplete while
        reporting success, and the cause would be a configuration nobody was
        told about. The registry prevents two arms sharing a NAME but cannot
        prevent two sharing a method, so the condition is caught here.

        Parameters
        ----------
        results:
            The arm results to fuse.

        Returns
        -------
        None

        Raises
        ------
        FusionError
            If a method appears more than once.
        """
        seen: dict[DiscoveryMethod, str] = {}

        for result in results:
            existing = seen.get(result.method)
            if existing is not None:
                raise FusionError(
                    f"Arms {existing!r} and {result.arm_name!r} both report "
                    f"discovery method {result.method.value}. A fused result "
                    f"records one signal per method, so both arms' "
                    f"observations of a shared node could not be represented "
                    f"and one would be silently lost.",
                    context={
                        "method": result.method.value,
                        "arms": [existing, result.arm_name],
                    },
                )
            seen[result.method] = result.arm_name

    def __repr__(self) -> str:
        """
        Return a representation naming the configured constant.

        Returns
        -------
        str
            Diagnostic representation.
        """
        if self._weights:
            rendered = {
                method.value: weight
                for method, weight in sorted(
                    self._weights.items(), key=lambda item: item[0].value
                )
            }
            weights = f", weights={rendered}"
        else:
            weights = ""

        return (
            f"ReciprocalRankFusion(rank_constant={self._rank_constant}"
            f"{weights})"
        )


# ============================================================================
# HELPERS
# ============================================================================


def _ordered_signals(
    candidates: Iterable[Candidate],
) -> tuple[RetrievalSignal, ...]:
    """
    Render a node's observations as signals in a deterministic order.

    Sorted by method name rather than left in arrival order, so that two
    otherwise identical results compare equal regardless of which arm finished
    first. A consumer diffing two runs should see a difference only where the
    retrieval genuinely differed.

    Parameters
    ----------
    candidates:
        Every arm's observation of one node.

    Returns
    -------
    tuple[RetrievalSignal, ...]
        The observations as signals, ordered by method.
    """
    return tuple(
        candidate.as_signal()
        for candidate in sorted(candidates, key=lambda item: item.method.value)
    )


def _validated_weights(
    weights: Mapping[str, float],
) -> Mapping[DiscoveryMethod, float]:
    """
    Resolve configured weights to discovery methods.

    A weight naming no known method is refused rather than ignored. An operator
    who misspells a method name would otherwise believe a weighting was in
    force while the ranking remained unweighted, and nothing would report the
    discrepancy.

    Parameters
    ----------
    weights:
        Weights keyed by discovery method name, as configuration supplies them.

    Returns
    -------
    Mapping[DiscoveryMethod, float]
        Weights keyed by method.

    Raises
    ------
    FusionError
        If a key names no known method, or a weight is negative.
    """
    resolved: dict[DiscoveryMethod, float] = {}
    known = {method.value: method for method in DiscoveryMethod}

    for key, weight in weights.items():
        method = known.get(key.upper())
        if method is None:
            raise FusionError(
                f"Fusion weight names {key!r}, which is not a discovery "
                f"method. Known methods are "
                f"{', '.join(sorted(known))}. An unrecognised key would leave "
                f"the ranking unweighted while appearing to weight it.",
                context={"unknown_method": key},
            )
        if weight < 0.0:
            raise FusionError(
                f"Fusion weight for {method.value} is {weight}. A negative "
                f"weight would lower a node because an arm found it, which "
                f"inverts what a retrieval observation means.",
                context={"method": method.value, "weight": weight},
            )
        resolved[method] = float(weight)

    return resolved
