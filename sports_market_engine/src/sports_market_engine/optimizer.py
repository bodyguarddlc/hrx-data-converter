from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from math import prod

from .models import Leg, Slip


@dataclass(frozen=True)
class OptimizerConfig:
    min_fair_probability: float = 0.50
    min_edge: float = 0.0
    candidate_pool: int = 80
    one_leg_per_event: bool = True
    hit_weight: float = 1.35
    payout_weight: float = 0.65
    edge_weight: float = 0.20


def optimize(
    legs: list[Leg],
    leg_counts: tuple[int, ...] = (2, 3, 4),
    top_n: int = 20,
    config: OptimizerConfig = OptimizerConfig(),
) -> list[Slip]:
    candidates = [
        x for x in legs
        if x.fair_probability >= config.min_fair_probability and x.edge >= config.min_edge
    ]
    candidates.sort(key=lambda x: _leg_score(x, config), reverse=True)

    # candidate_pool=0 enables an exhaustive search over every filtered leg.
    if config.candidate_pool > 0:
        candidates = candidates[: config.candidate_pool]

    frontier: list[Slip] = []
    for count in sorted(set(leg_counts)):
        if count < 1:
            continue
        for combo in combinations(candidates, count):
            if config.one_leg_per_event and len({x.event_id for x in combo}) != count:
                continue

            hit_probability = prod(x.fair_probability for x in combo)
            payout = prod(x.decimal_odds for x in combo)
            expected_return = hit_probability * payout
            score = (
                (hit_probability ** config.hit_weight)
                * (payout ** config.payout_weight)
                * (prod(1.0 + max(x.edge, -0.99) for x in combo) ** config.edge_weight)
            )
            slip = Slip(
                legs=tuple(combo),
                hit_probability=hit_probability,
                payout_multiple=payout,
                expected_return=expected_return,
                score=score,
            )
            _insert_pareto(frontier, slip)

    frontier.sort(
        key=lambda s: (s.score, s.hit_probability, s.expected_return, s.payout_multiple),
        reverse=True,
    )
    return frontier[:top_n]


def _leg_score(leg: Leg, config: OptimizerConfig) -> float:
    return (
        (leg.fair_probability ** config.hit_weight)
        * (leg.decimal_odds ** config.payout_weight)
        * ((1.0 + max(leg.edge, -0.99)) ** config.edge_weight)
    )


def _insert_pareto(frontier: list[Slip], slip: Slip) -> None:
    """Maintain the exact hit-probability/payout Pareto frontier for visited slips."""
    for other in frontier:
        if _dominates(other, slip):
            return
    frontier[:] = [other for other in frontier if not _dominates(slip, other)]
    frontier.append(slip)


def _dominates(a: Slip, b: Slip) -> bool:
    return (
        a.hit_probability >= b.hit_probability
        and a.payout_multiple >= b.payout_multiple
        and (
            a.hit_probability > b.hit_probability
            or a.payout_multiple > b.payout_multiple
        )
    )
