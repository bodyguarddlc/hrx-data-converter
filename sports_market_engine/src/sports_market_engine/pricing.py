from __future__ import annotations

from collections import defaultdict
from statistics import median

from .models import Leg


def implied_probability(decimal_odds: float) -> float:
    if decimal_odds <= 1.0:
        raise ValueError("decimal odds must be > 1.0")
    return 1.0 / decimal_odds


def normalize_market(outcomes: list[dict]) -> dict[tuple[str, float | None], float]:
    """Remove bookmaker margin with proportional normalization."""
    raw: dict[tuple[str, float | None], float] = {}
    total = 0.0
    for outcome in outcomes:
        price = float(outcome["price"])
        key = (str(outcome["name"]), _point(outcome.get("point")))
        p = implied_probability(price)
        raw[key] = p
        total += p
    if total <= 0:
        return {}
    return {k: v / total for k, v in raw.items()}


def extract_legs(events: list[dict], min_books: int = 2) -> list[Leg]:
    """
    For each event/market/selection/point:
      1. de-vig each bookmaker's market,
      2. aggregate fair probability by median across books,
      3. retain the best available price.
    """
    legs: list[Leg] = []

    for event in events:
        fair_samples: dict[tuple[str, str, float | None], list[float]] = defaultdict(list)
        best_price: dict[tuple[str, str, float | None], tuple[float, str]] = {}

        for book in event.get("bookmakers", []):
            book_name = str(book.get("title") or book.get("key") or "unknown")
            for market in book.get("markets", []):
                market_key = str(market.get("key"))
                outcomes = market.get("outcomes", [])
                normalized = normalize_market(outcomes)
                for outcome in outcomes:
                    selection = str(outcome["name"])
                    point = _point(outcome.get("point"))
                    key = (market_key, selection, point)
                    if (selection, point) in normalized:
                        fair_samples[key].append(normalized[(selection, point)])
                    price = float(outcome["price"])
                    if price > best_price.get(key, (0.0, ""))[0]:
                        best_price[key] = (price, book_name)

        for key, samples in fair_samples.items():
            if len(samples) < min_books:
                continue
            market_key, selection, point = key
            price, book_name = best_price[key]
            fair_p = median(samples)
            edge = fair_p * price - 1.0
            legs.append(
                Leg(
                    event_id=str(event["id"]),
                    sport_key=str(event["sport_key"]),
                    event_name=f'{event.get("away_team", "?")} @ {event.get("home_team", "?")}',
                    commence_time=_parse_dt(str(event["commence_time"])),
                    market=market_key,
                    selection=selection,
                    point=point,
                    bookmaker=book_name,
                    decimal_odds=price,
                    fair_probability=fair_p,
                    edge=edge,
                )
            )

    return legs


def _point(value) -> float | None:
    return None if value is None else float(value)


def _parse_dt(value: str):
    from datetime import datetime
    return datetime.fromisoformat(value.replace("Z", "+00:00"))
