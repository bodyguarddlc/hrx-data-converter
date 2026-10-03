from datetime import datetime, timedelta, timezone

from edgeforge.models import Offer
from edgeforge.optimizer import OptimizeConfig, optimize
from edgeforge.pricing import build_candidates


NOW = datetime(2026, 10, 3, 18, 0, tzinfo=timezone.utc)


def offer(event, book, market, name, odds, point=None):
    return Offer(
        sport_key="test_sport",
        event_id=event,
        commence_time=NOW + timedelta(hours=2),
        home_team=f"{event}-home",
        away_team=f"{event}-away",
        bookmaker=book,
        market_key=market,
        outcome_name=name,
        decimal_odds=odds,
        point=point,
        last_update=NOW - timedelta(seconds=20),
    )


def test_devig_and_best_price():
    offers = [
        offer("e1", "bookA", "h2h", "A", 2.10),
        offer("e1", "bookA", "h2h", "B", 1.80),
        offer("e1", "bookB", "h2h", "A", 2.20),
        offer("e1", "bookB", "h2h", "B", 1.75),
    ]
    candidates = build_candidates(offers, min_books=2, now=NOW)
    a = next(c for c in candidates if c.offer.outcome_name == "A")
    assert a.offer.bookmaker == "bookB"
    assert a.offer.decimal_odds == 2.20
    assert 0 < a.fair_probability < 1
    assert a.books_observed == 2


def test_stale_lines_removed():
    stale = Offer(
        sport_key="test",
        event_id="e",
        commence_time=NOW + timedelta(hours=1),
        home_team="H",
        away_team="A",
        bookmaker="book",
        market_key="h2h",
        outcome_name="H",
        decimal_odds=2.0,
        last_update=NOW - timedelta(hours=1),
    )
    assert build_candidates([stale], min_books=1, max_age_seconds=60, now=NOW) == []


def test_optimizer_respects_probability_floor_and_distinct_events():
    offers = []
    for event, a_odds, b_odds in [
        ("e1", 2.25, 1.72),
        ("e2", 2.20, 1.75),
        ("e3", 2.15, 1.78),
    ]:
        for book, bump in [("bookA", 0.00), ("bookB", 0.08)]:
            offers.extend([
                offer(event, book, "h2h", f"{event}-home", a_odds + bump),
                offer(event, book, "h2h", f"{event}-away", b_odds),
            ])

    candidates = build_candidates(offers, min_books=2, now=NOW)
    best, frontier = optimize(
        candidates,
        OptimizeConfig(
            min_legs=2,
            max_legs=3,
            min_joint_probability=0.16,
            min_edge=-1.0,
            min_ev=-1.0,
            distinct_event=True,
            max_candidates=20,
        ),
    )
    assert best is not None
    assert best.joint_probability >= 0.16
    assert len(best.legs) >= 2
    assert len({x.offer.event_id for x in best.legs}) == len(best.legs)
    assert frontier
