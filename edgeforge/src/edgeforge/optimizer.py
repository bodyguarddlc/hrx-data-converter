from __future__ import annotations

import math
from dataclasses import dataclass

from .models import Candidate, Ticket


@dataclass(frozen=True, slots=True)
class OptimizeConfig:
    min_legs: int = 2
    max_legs: int = 5
    min_joint_probability: float = 0.15
    min_edge: float = 0.01
    min_ev: float = 0.0
    hit_weight: float = 0.65
    payout_weight: float = 0.35
    beam_width: int = 2500
    max_candidates: int = 60
    distinct_event: bool = True


def _score(p: float, payout: float, hit_weight: float, payout_weight: float) -> float:
    return hit_weight * math.log(max(p, 1e-15)) + payout_weight * math.log(max(payout, 1.0))


def _valid_extension(legs: tuple[Candidate, ...], nxt: Candidate, distinct_event: bool) -> bool:
    if any(x.offer.selection_key == nxt.offer.selection_key for x in legs):
        return False
    if distinct_event and any(x.offer.event_id == nxt.offer.event_id for x in legs):
        return False
    if any(x.offer.market_instance_key == nxt.offer.market_instance_key for x in legs):
        return False
    return True


def _make_ticket(
    legs: tuple[Candidate, ...],
    hit_weight: float,
    payout_weight: float,
) -> Ticket:
    p = math.prod(x.fair_probability for x in legs)
    payout = math.prod(x.offer.decimal_odds for x in legs)
    return Ticket(
        legs=legs,
        joint_probability=p,
        decimal_payout=payout,
        expected_net_return=p * payout - 1.0,
        score=_score(p, payout, hit_weight, payout_weight),
    )


def pareto_frontier(tickets: list[Ticket]) -> list[Ticket]:
    ordered = sorted(
        tickets,
        key=lambda t: (t.joint_probability, t.decimal_payout),
        reverse=True,
    )
    frontier: list[Ticket] = []
    best_payout = -1.0
    for ticket in ordered:
        if ticket.decimal_payout > best_payout + 1e-12:
            frontier.append(ticket)
            best_payout = ticket.decimal_payout
    return sorted(frontier, key=lambda t: t.score, reverse=True)


def optimize(candidates: list[Candidate], config: OptimizeConfig) -> tuple[Ticket | None, list[Ticket]]:
    if not 1 <= config.min_legs <= config.max_legs:
        raise ValueError("require 1 <= min_legs <= max_legs")
    if not 0 < config.min_joint_probability <= 1:
        raise ValueError("min_joint_probability must be in (0,1]")
    if config.hit_weight < 0 or config.payout_weight < 0:
        raise ValueError("weights must be non-negative")
    if config.hit_weight + config.payout_weight == 0:
        raise ValueError("at least one objective weight must be positive")

    pool = [
        c for c in candidates
        if c.edge >= config.min_edge and c.expected_value >= config.min_ev
    ][: config.max_candidates]
    pool.sort(
        key=lambda c: (
            _score(
                c.fair_probability,
                c.offer.decimal_odds,
                config.hit_weight,
                config.payout_weight,
            ),
            c.expected_value,
        ),
        reverse=True,
    )

    beam: list[tuple[Candidate, ...]] = [tuple()]
    feasible: list[Ticket] = []

    for depth in range(1, config.max_legs + 1):
        expanded: list[Ticket] = []
        for legs in beam:
            start = 0
            if legs:
                last_key = legs[-1].offer.selection_key
                for i, c in enumerate(pool):
                    if c.offer.selection_key == last_key:
                        start = i + 1
                        break
            for nxt in pool[start:]:
                if not _valid_extension(legs, nxt, config.distinct_event):
                    continue
                ticket = _make_ticket(
                    legs + (nxt,),
                    config.hit_weight,
                    config.payout_weight,
                )
                if ticket.joint_probability + 1e-15 < config.min_joint_probability:
                    continue
                expanded.append(ticket)
                if depth >= config.min_legs:
                    feasible.append(ticket)

        expanded.sort(key=lambda t: (t.score, t.expected_net_return), reverse=True)
        beam = [t.legs for t in expanded[: config.beam_width]]
        if not beam:
            break

    if not feasible:
        return None, []

    frontier = pareto_frontier(feasible)
    best = max(frontier, key=lambda t: (t.score, t.expected_net_return))
    return best, frontier
