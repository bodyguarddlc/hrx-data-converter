from datetime import datetime, timezone

from sports_market_engine.models import Leg
from sports_market_engine.optimizer import OptimizerConfig, optimize


def leg(event_id: str, p: float, odds: float, edge: float = 0.03) -> Leg:
    return Leg(
        event_id=event_id,
        sport_key="test",
        event_name=event_id,
        commence_time=datetime.now(timezone.utc),
        market="h2h",
        selection=f"{event_id}-pick",
        point=None,
        bookmaker="book",
        decimal_odds=odds,
        fair_probability=p,
        edge=edge,
    )


def test_optimizer_respects_one_leg_per_event():
    legs = [
        leg("A", 0.70, 1.55),
        leg("A", 0.66, 1.70),
        leg("B", 0.68, 1.62),
        leg("C", 0.64, 1.75),
    ]
    slips = optimize(
        legs,
        leg_counts=(2,),
        top_n=10,
        config=OptimizerConfig(candidate_pool=20, one_leg_per_event=True),
    )
    assert slips
    assert all(len({x.event_id for x in s.legs}) == 2 for s in slips)


def test_optimizer_returns_pareto_candidates():
    legs = [
        leg("A", 0.78, 1.42),
        leg("B", 0.72, 1.55),
        leg("C", 0.61, 1.95),
        leg("D", 0.56, 2.15),
    ]
    slips = optimize(legs, leg_counts=(2,), top_n=10)
    assert slips
    assert all(s.hit_probability > 0 for s in slips)
    assert all(s.payout_multiple > 1 for s in slips)
