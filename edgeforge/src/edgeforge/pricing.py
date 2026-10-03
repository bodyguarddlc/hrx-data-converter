from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
from statistics import median
from typing import Iterable

from .models import Candidate, Offer


def implied_probability(decimal_odds: float) -> float:
    if decimal_odds <= 1.0:
        raise ValueError("decimal odds must be > 1.0")
    return 1.0 / decimal_odds


def _devig_group(offers: list[Offer]) -> dict[tuple, float]:
    raw = [implied_probability(o.decimal_odds) for o in offers]
    overround = sum(raw)
    if len(offers) < 2 or overround <= 0:
        return {}
    return {o.selection_key: p / overround for o, p in zip(offers, raw)}


def build_candidates(
    offers: Iterable[Offer],
    *,
    min_books: int = 2,
    max_age_seconds: int | None = 300,
    now: datetime | None = None,
) -> list[Candidate]:
    """Create best-price candidate legs from raw bookmaker offers.

    Fair probability is the median of per-bookmaker de-vig probabilities for
    the identical selection. This is a market consensus, not an independent
    predictive model.
    """
    now = now or datetime.now(timezone.utc)
    fresh: list[Offer] = []
    for offer in offers:
        if offer.decimal_odds <= 1.0:
            continue
        if max_age_seconds is not None and offer.last_update is not None:
            stamp = offer.last_update
            if stamp.tzinfo is None:
                stamp = stamp.replace(tzinfo=timezone.utc)
            age = (now - stamp.astimezone(timezone.utc)).total_seconds()
            if age > max_age_seconds:
                continue
        fresh.append(offer)

    by_book_market: dict[tuple, list[Offer]] = defaultdict(list)
    for offer in fresh:
        by_book_market[
            (offer.bookmaker, offer.event_id, offer.market_key, offer.line_bucket)
        ].append(offer)

    fair_samples: dict[tuple, list[float]] = defaultdict(list)
    books: dict[tuple, set[str]] = defaultdict(set)
    for group in by_book_market.values():
        fair = _devig_group(group)
        for offer in group:
            if offer.selection_key in fair:
                fair_samples[offer.selection_key].append(fair[offer.selection_key])
                books[offer.selection_key].add(offer.bookmaker)

    best_offer: dict[tuple, Offer] = {}
    for offer in fresh:
        key = offer.selection_key
        if key not in fair_samples:
            continue
        current = best_offer.get(key)
        if current is None or offer.decimal_odds > current.decimal_odds:
            best_offer[key] = offer

    out: list[Candidate] = []
    for key, offer in best_offer.items():
        book_count = len(books[key])
        if book_count < min_books:
            continue
        fair_p = float(median(fair_samples[key]))
        implied_p = implied_probability(offer.decimal_odds)
        ev = fair_p * offer.decimal_odds - 1.0
        out.append(
            Candidate(
                offer=offer,
                fair_probability=fair_p,
                implied_probability=implied_p,
                edge=fair_p - implied_p,
                expected_value=ev,
                books_observed=book_count,
            )
        )
    return sorted(out, key=lambda c: (c.expected_value, c.edge), reverse=True)
