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
    candidates.sort(
        key=lambda x: (
            (x.fair_probability ** config.hit_weight)
            * (x.decimal_odds ** config.payout_weight)
            * ((1.0 + max(x.edge, -0.99)) ** config.edge_weight)
        ),
        reverse=True,
    )
    candidates = candidates[: config.candidate_pool]

    slips: list[Slip] = []
    for count in sorted(set(leg_counts)):
        if count < 1:
            continue
        for combo in combinations(candidates, count):
            if config.one_leg_per_event and len({x.event_id for x in combo}) != count:
                continue

            hit_probability = prod(x.fair_probability for x in combo)
            payout = prod(x.decimal_odds for x in combo)
            expected_return = hit_probability * payout

            # Log-equivalent multiplicative objective:
            # prioritizes probability, then payout, then positive pricing edge.
            score = (
                (hit_probability ** config.hit_weight)
                * (payout ** config.payout_weight)
                * (prod(1.0 + max(x.edge, -0.99) for x in combo) ** config.edge_weight)
            )

            slips.append(
                Slip(
                    legs=tuple(combo),
                    hit_probability=hit_probability,
                    payout_multiple=payout,
                    expected_return=expected_return,
                    score=score,
                )
            )

    slips.sort(key=lambda s: (s.score, s.hit_probability, s.expected_return), reverse=True)
    return _pareto_filter(slips)[:top_n]


def _pareto_filter(slips: list[Slip]) -> list[Slip]:
    """Keep slips not strictly dominated on both hit probability and payout."""
    frontier: list[Slip] = []
    for slip in slips:
        dominated = any(
            other.hit_probability >= slip.hit_probability
            and other.payout_multiple >= slip.payout_multiple
            and (
                other.hit_probability > slip.hit_probability
                or other.payout_multiple > slip.payout_multiple
            )
            for other in frontier
        )
        if not dominated:
            frontier.append(slip)
    return frontier
