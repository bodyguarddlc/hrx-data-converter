from __future__ import annotations

import csv
import json
import math
import os
import statistics
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass
from datetime import date, datetime, time as dt_time, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence
from zoneinfo import ZoneInfo


ALL_REGIONS = ("us", "us2", "uk", "eu", "au")
FEATURED_MARKETS = ("h2h", "spreads", "totals")


@dataclass(frozen=True)
class Candidate:
    event_id: str
    sport_key: str
    sport_title: str
    commence_time: str
    home_team: str
    away_team: str
    market: str
    outcome: str
    description: str
    point: float | None
    best_book: str
    best_decimal: float
    consensus_probability: float
    fair_probability: float
    dispersion: float
    edge: float
    books: int

    @property
    def label(self) -> str:
        extra = f" {self.point:g}" if self.point is not None else ""
        desc = f" ({self.description})" if self.description else ""
        return f"{self.sport_title}: {self.outcome}{extra}{desc} [{self.market}]"


@dataclass(frozen=True)
class Ticket:
    legs: tuple[Candidate, ...]
    joint_probability: float
    decimal_payout: float
    expected_gross: float
    expected_profit: float
    score: float


@dataclass
class ScanResult:
    events: list[dict[str, Any]]
    credits_used_estimate: int
    credits_remaining_header: int | None
    skipped: list[str]
    notes: list[str]


def implied_probability(decimal_odds: float) -> float:
    if decimal_odds <= 1.0:
        raise ValueError("decimal odds must be greater than 1")
    return 1.0 / decimal_odds


def devig_probabilities(decimal_prices: Sequence[float]) -> list[float]:
    raw = [implied_probability(p) for p in decimal_prices]
    total = sum(raw)
    if total <= 0:
        raise ValueError("invalid market")
    return [p / total for p in raw]


def _safe_point(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _group_point(market: str, point: float | None) -> float | None:
    if point is None:
        return None
    if "spread" in market:
        return abs(point)
    return point


def _selection_key(event_id: str, market: str, outcome: dict[str, Any]) -> tuple[Any, ...]:
    return (
        event_id,
        market,
        str(outcome.get("name", "")),
        str(outcome.get("description", "")),
        _safe_point(outcome.get("point")),
    )


def _market_group_key(market: str, outcome: dict[str, Any]) -> tuple[Any, ...]:
    description = str(outcome.get("description", ""))
    point = _group_point(market, _safe_point(outcome.get("point")))
    return market, description, point


def build_candidates(
    events: Sequence[dict[str, Any]],
    *,
    min_books: int = 2,
    confidence_z: float = 0.40,
) -> list[Candidate]:
    """Convert raw bookmaker events into conservative consensus selections.

    Each bookmaker market is de-vigged independently. Exact selections are then
    aggregated across books. Probability is shrunk toward the offered implied
    price when cross-book support is thin and reduced by disagreement.
    """
    agg: dict[tuple[Any, ...], dict[str, Any]] = {}

    for event in events:
        event_id = str(event.get("id", ""))
        if not event_id:
            continue
        for book in event.get("bookmakers", []) or []:
            book_key = str(book.get("key") or book.get("title") or "unknown")
            for market in book.get("markets", []) or []:
                market_key = str(market.get("key", ""))
                groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
                for outcome in market.get("outcomes", []) or []:
                    try:
                        price = float(outcome.get("price"))
                    except (TypeError, ValueError):
                        continue
                    if price <= 1.0:
                        continue
                    enriched = dict(outcome)
                    enriched["_price"] = price
                    groups.setdefault(_market_group_key(market_key, enriched), []).append(enriched)

                for outcomes in groups.values():
                    if len(outcomes) < 2:
                        continue
                    prices = [o["_price"] for o in outcomes]
                    fair = devig_probabilities(prices)
                    for outcome, p_fair in zip(outcomes, fair):
                        key = _selection_key(event_id, market_key, outcome)
                        row = agg.setdefault(
                            key,
                            {
                                "event": event,
                                "market": market_key,
                                "outcome": str(outcome.get("name", "")),
                                "description": str(outcome.get("description", "")),
                                "point": _safe_point(outcome.get("point")),
                                "prices": [],
                                "probs": [],
                                "books": set(),
                            },
                        )
                        row["prices"].append((outcome["_price"], book_key))
                        row["probs"].append(float(p_fair))
                        row["books"].add(book_key)

    candidates: list[Candidate] = []
    for row in agg.values():
        n_books = len(row["books"])
        if n_books < min_books:
            continue
        event = row["event"]
        best_decimal, best_book = max(row["prices"], key=lambda x: x[0])
        probs = row["probs"]
        consensus = statistics.median(probs)
        dispersion = statistics.pstdev(probs) if len(probs) > 1 else 0.0

        market_implied = implied_probability(best_decimal)
        support_weight = min(1.0, n_books / 6.0)
        shrunk = support_weight * consensus + (1.0 - support_weight) * market_implied
        conservative = max(0.001, min(0.999, shrunk - confidence_z * dispersion))
        edge = conservative * best_decimal - 1.0

        candidates.append(
            Candidate(
                event_id=str(event.get("id", "")),
                sport_key=str(event.get("sport_key", "")),
                sport_title=str(event.get("sport_title", event.get("sport_key", ""))),
                commence_time=str(event.get("commence_time", "")),
                home_team=str(event.get("home_team", "")),
                away_team=str(event.get("away_team", "")),
                market=row["market"],
                outcome=row["outcome"],
                description=row["description"],
                point=row["point"],
                best_book=best_book,
                best_decimal=float(best_decimal),
                consensus_probability=float(consensus),
                fair_probability=float(conservative),
                dispersion=float(dispersion),
                edge=float(edge),
                books=n_books,
            )
        )

    return sorted(
        candidates,
        key=lambda c: (c.edge, c.fair_probability, c.best_decimal),
        reverse=True,
    )


def _ticket_from_legs(
    legs: Sequence[Candidate],
    *,
    hit_weight: float,
    payout_weight: float,
) -> Ticket:
    joint_p = math.prod(x.fair_probability for x in legs)
    payout = math.prod(x.best_decimal for x in legs)
    expected_gross = joint_p * payout
    score = (joint_p ** hit_weight) * (payout ** payout_weight)
    return Ticket(
        legs=tuple(legs),
        joint_probability=joint_p,
        decimal_payout=payout,
        expected_gross=expected_gross,
        expected_profit=expected_gross - 1.0,
        score=score,
    )


def optimize_tickets(
    candidates: Sequence[Candidate],
    *,
    min_legs: int = 2,
    max_legs: int = 4,
    top_n: int = 25,
    min_edge: float = 0.01,
    min_leg_probability: float = 0.52,
    max_leg_decimal: float = 3.50,
    one_leg_per_event: bool = True,
    beam_width: int = 800,
    hit_weight: float = 1.0,
    payout_weight: float = 1.0,
) -> list[Ticket]:
    """Beam-search tickets using conservative hit probability x payout.

    The default objective is exactly P(ticket) * decimal payout. By default the
    optimizer will not put two legs from the same event in one ticket because
    naive multiplication would overstate hit rate for correlated outcomes.
    """
    pool = [
        c
        for c in candidates
        if c.edge >= min_edge
        and c.fair_probability >= min_leg_probability
        and c.best_decimal <= max_leg_decimal
    ]
    pool.sort(
        key=lambda c: (
            c.fair_probability * c.best_decimal,
            c.edge,
            c.fair_probability,
        ),
        reverse=True,
    )

    # Bound the candidate pool before beam expansion while keeping broad event coverage.
    by_event: dict[str, list[Candidate]] = {}
    for c in pool:
        by_event.setdefault(c.event_id, []).append(c)
    compact: list[Candidate] = []
    for event_candidates in by_event.values():
        compact.extend(event_candidates[:4])
    pool = sorted(
        compact,
        key=lambda c: (c.fair_probability * c.best_decimal, c.edge),
        reverse=True,
    )[:240]

    states: list[tuple[int, tuple[Candidate, ...], Ticket]] = []
    for idx, c in enumerate(pool):
        t = _ticket_from_legs((c,), hit_weight=hit_weight, payout_weight=payout_weight)
        states.append((idx, (c,), t))
    states.sort(key=lambda x: x[2].score, reverse=True)
    states = states[:beam_width]

    results: list[Ticket] = []
    seen: set[tuple[tuple[str, str, str, str], ...]] = set()

    for leg_count in range(2, max_legs + 1):
        next_states: list[tuple[int, tuple[Candidate, ...], Ticket]] = []
        for last_idx, legs, _ in states:
            event_ids = {x.event_id for x in legs}
            for idx in range(last_idx + 1, len(pool)):
                c = pool[idx]
                if one_leg_per_event and c.event_id in event_ids:
                    continue
                new_legs = legs + (c,)
                ticket = _ticket_from_legs(
                    new_legs,
                    hit_weight=hit_weight,
                    payout_weight=payout_weight,
                )
                next_states.append((idx, new_legs, ticket))

        next_states.sort(
            key=lambda x: (
                x[2].score,
                x[2].expected_gross,
                x[2].joint_probability,
            ),
            reverse=True,
        )
        states = next_states[:beam_width]

        if leg_count >= min_legs:
            for _, legs, ticket in states:
                sig = tuple(
                    sorted(
                        (
                            x.event_id,
                            x.market,
                            f"{x.outcome}|{x.description}",
                            "" if x.point is None else f"{x.point:g}",
                        )
                        for x in legs
                    )
                )
                if sig in seen:
                    continue
                seen.add(sig)
                results.append(ticket)

    results.sort(
        key=lambda t: (t.score, t.expected_gross, t.joint_probability),
        reverse=True,
    )
    return results[:top_n]


class TheOddsAPI:
    base_url = "https://api.the-odds-api.com"

    def __init__(self, api_key: str, *, timeout: int = 20, pause: float = 0.0):
        if not api_key:
            raise ValueError("THE_ODDS_API_KEY is required")
        self.api_key = api_key
        self.timeout = timeout
        self.pause = pause
        self.last_headers: dict[str, str] = {}

    def _get(self, path: str, params: dict[str, Any]) -> Any:
        query = dict(params)
        query["apiKey"] = self.api_key
        url = self.base_url + path + "?" + urllib.parse.urlencode(query)
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "elite-odds-engine/1.0"},
            method="GET",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                self.last_headers = {k.lower(): v for k, v in resp.headers.items()}
                data = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Odds API HTTP {e.code}: {detail}") from e
        if self.pause:
            time.sleep(self.pause)
        return data

    def sports(self) -> list[dict[str, Any]]:
        return self._get("/v4/sports/", {})

    def events(
        self,
        sport: str,
        commence_from: str,
        commence_to: str,
    ) -> list[dict[str, Any]]:
        return self._get(
            f"/v4/sports/{urllib.parse.quote(sport)}/events",
            {
                "dateFormat": "iso",
                "commenceTimeFrom": commence_from,
                "commenceTimeTo": commence_to,
            },
        )

    def odds(
        self,
        sport: str,
        *,
        regions: Sequence[str],
        markets: Sequence[str],
        commence_from: str,
        commence_to: str,
    ) -> list[dict[str, Any]]:
        return self._get(
            f"/v4/sports/{urllib.parse.quote(sport)}/odds",
            {
                "regions": ",".join(regions),
                "markets": ",".join(markets),
                "oddsFormat": "decimal",
                "dateFormat": "iso",
                "commenceTimeFrom": commence_from,
                "commenceTimeTo": commence_to,
            },
        )

    def event_markets(
        self,
        sport: str,
        event_id: str,
        *,
        regions: Sequence[str],
    ) -> dict[str, Any]:
        return self._get(
            f"/v4/sports/{urllib.parse.quote(sport)}/events/{urllib.parse.quote(event_id)}/markets",
            {
                "regions": ",".join(regions),
                "dateFormat": "iso",
            },
        )

    def event_odds(
        self,
        sport: str,
        event_id: str,
        *,
        regions: Sequence[str],
        markets: Sequence[str],
    ) -> dict[str, Any]:
        return self._get(
            f"/v4/sports/{urllib.parse.quote(sport)}/events/{urllib.parse.quote(event_id)}/odds",
            {
                "regions": ",".join(regions),
                "markets": ",".join(markets),
                "oddsFormat": "decimal",
                "dateFormat": "iso",
            },
        )

    @property
    def credits_remaining(self) -> int | None:
        raw = self.last_headers.get("x-requests-remaining")
        try:
            return int(raw) if raw is not None else None
        except ValueError:
            return None


def _day_bounds(day: date, tz_name: str) -> tuple[str, str]:
    tz = ZoneInfo(tz_name)
    start_local = datetime.combine(day, dt_time.min, tzinfo=tz)
    end_local = datetime.combine(day, dt_time.max, tzinfo=tz)
    start_utc = start_local.astimezone(timezone.utc).replace(microsecond=0)
    end_utc = end_local.astimezone(timezone.utc).replace(microsecond=0)
    return start_utc.isoformat().replace("+00:00", "Z"), end_utc.isoformat().replace("+00:00", "Z")


def _merge_events(target: dict[str, dict[str, Any]], incoming: Iterable[dict[str, Any]]) -> None:
    for event in incoming:
        event_id = str(event.get("id", ""))
        if not event_id:
            continue
        if event_id not in target:
            target[event_id] = dict(event)
            continue
        existing = target[event_id]
        by_book = {
            str(b.get("key") or b.get("title")): b
            for b in existing.get("bookmakers", []) or []
        }
        for book in event.get("bookmakers", []) or []:
            key = str(book.get("key") or book.get("title"))
            if key not in by_book:
                existing.setdefault("bookmakers", []).append(book)
                by_book[key] = book
                continue
            current = by_book[key]
            markets_by_key = {str(m.get("key")): m for m in current.get("markets", []) or []}
            for market in book.get("markets", []) or []:
                mk = str(market.get("key"))
                markets_by_key[mk] = market
            current["markets"] = list(markets_by_key.values())


def _market_coverage(event_market_payload: dict[str, Any]) -> list[str]:
    coverage: dict[str, int] = {}
    for book in event_market_payload.get("bookmakers", []) or []:
        for market in book.get("markets", []) or []:
            key = str(market.get("key", ""))
            if key:
                coverage[key] = coverage.get(key, 0) + 1
    return [
        key
        for key, _ in sorted(
            coverage.items(),
            key=lambda kv: (kv[1], kv[0]),
            reverse=True,
        )
    ]


def scan_day(
    provider: TheOddsAPI,
    *,
    day: date,
    timezone_name: str = "America/Los_Angeles",
    regions: Sequence[str] = ("us", "eu"),
    max_credits: int = 80,
    include_spreads_totals: bool = True,
    deep: bool = False,
    deep_market_cap: int = 12,
    max_deep_events: int = 8,
) -> ScanResult:
    """Scan all same-day active sports, spending quota in coverage-first order.

    Pass 1: discover all same-day events via the free events endpoint.
    Pass 2: get h2h for every sport possible.
    Pass 3: add spreads/totals if budget remains.
    Pass 4: optional event-level market discovery and props, most-covered first.
    """
    if not regions:
        raise ValueError("at least one region is required")
    if max_credits < 1:
        raise ValueError("max_credits must be positive")

    start, end = _day_bounds(day, timezone_name)
    active_sports = [s for s in provider.sports() if s.get("active", True)]
    sports_with_events: list[tuple[dict[str, Any], list[dict[str, Any]]]] = []
    notes: list[str] = []
    skipped: list[str] = []

    for sport in active_sports:
        key = str(sport.get("key", ""))
        if not key:
            continue
        todays = provider.events(key, start, end)
        if todays:
            sports_with_events.append((sport, todays))

    # Broad coverage first: price one region across every sport before
    # spending credits on cross-region line shopping or market depth.
    region_cost = len(regions)
    primary_region = (regions[0],)
    extra_regions = tuple(regions[1:])
    merged: dict[str, dict[str, Any]] = {}
    credits = 0

    for sport, _ in sports_with_events:
        key = str(sport["key"])
        if credits + 1 > max_credits:
            skipped.append(f"{key}: h2h primary-region (budget)")
            continue
        odds = provider.odds(
            key,
            regions=primary_region,
            markets=("h2h",),
            commence_from=start,
            commence_to=end,
        )
        _merge_events(merged, odds)
        credits += 1

    # Once every possible sport has baseline pricing, add the requested
    # foreign/domestic regions for line shopping.
    if extra_regions:
        extra_cost = len(extra_regions)
        for sport, todays in sorted(
            sports_with_events,
            key=lambda x: len(x[1]),
            reverse=True,
        ):
            key = str(sport["key"])
            if credits + extra_cost > max_credits:
                skipped.append(f"{key}: h2h extra-regions (budget)")
                continue
            odds = provider.odds(
                key,
                regions=extra_regions,
                markets=("h2h",),
                commence_from=start,
                commence_to=end,
            )
            _merge_events(merged, odds)
            credits += extra_cost

    if include_spreads_totals:
        # Add market depth in the same coverage-first order: primary region
        # across sports, then extra regions where budget remains.
        for sport, todays in sorted(
            sports_with_events,
            key=lambda x: len(x[1]),
            reverse=True,
        ):
            key = str(sport["key"])
            if credits + 2 > max_credits:
                skipped.append(f"{key}: spreads/totals primary-region (budget)")
                continue
            odds = provider.odds(
                key,
                regions=primary_region,
                markets=("spreads", "totals"),
                commence_from=start,
                commence_to=end,
            )
            _merge_events(merged, odds)
            credits += 2

        if extra_regions:
            extra_depth_cost = 2 * len(extra_regions)
            for sport, todays in sorted(
                sports_with_events,
                key=lambda x: len(x[1]),
                reverse=True,
            ):
                key = str(sport["key"])
                if credits + extra_depth_cost > max_credits:
                    skipped.append(f"{key}: spreads/totals extra-regions (budget)")
                    continue
                odds = provider.odds(
                    key,
                    regions=extra_regions,
                    markets=("spreads", "totals"),
                    commence_from=start,
                    commence_to=end,
                )
                _merge_events(merged, odds)
                credits += extra_depth_cost

    if deep and credits < max_credits:
        deep_events: list[dict[str, Any]] = []
        for _, todays in sports_with_events:
            deep_events.extend(todays)
        # Earlier events first, then stable id for deterministic runs.
        deep_events.sort(key=lambda e: (str(e.get("commence_time", "")), str(e.get("id", ""))))

        for event in deep_events[:max_deep_events]:
            if credits + 1 > max_credits:
                break
            sport_key = str(event.get("sport_key", ""))
            event_id = str(event.get("id", ""))
            discovered = provider.event_markets(sport_key, event_id, regions=regions)
            credits += 1
            keys = [
                k
                for k in _market_coverage(discovered)
                if k not in FEATURED_MARKETS
            ]
            if not keys:
                continue

            affordable = max(0, (max_credits - credits) // region_cost)
            selected = keys[: min(deep_market_cap, affordable)]
            if not selected:
                break
            deep_odds = provider.event_odds(
                sport_key,
                event_id,
                regions=regions,
                markets=selected,
            )
            _merge_events(merged, [deep_odds])
            credits += len(selected) * region_cost

    if sports_with_events and credits == 0:
        notes.append("No priced sports fit the configured credit budget.")
    if skipped:
        notes.append(
            "Coverage was budget-limited. Increase --max-credits, reduce regions, or disable depth markets."
        )
    if deep:
        notes.append(
            "Deep market mode prioritizes markets seen at the most bookmakers; event-market discovery itself costs one credit per event."
        )

    return ScanResult(
        events=list(merged.values()),
        credits_used_estimate=credits,
        credits_remaining_header=provider.credits_remaining,
        skipped=skipped,
        notes=notes,
    )


def write_outputs(
    out_dir: str | os.PathLike[str],
    *,
    scan: ScanResult,
    candidates: Sequence[Candidate],
    tickets: Sequence[Ticket],
) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    with (out / "scan.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "credits_used_estimate": scan.credits_used_estimate,
                "credits_remaining_header": scan.credits_remaining_header,
                "skipped": scan.skipped,
                "notes": scan.notes,
                "events": scan.events,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    with (out / "candidates.json").open("w", encoding="utf-8") as f:
        json.dump([asdict(c) for c in candidates], f, indent=2, ensure_ascii=False)

    candidate_fields = list(asdict(candidates[0]).keys()) if candidates else [
        "event_id", "sport_key", "sport_title", "commence_time", "home_team",
        "away_team", "market", "outcome", "description", "point", "best_book",
        "best_decimal", "consensus_probability", "fair_probability",
        "dispersion", "edge", "books",
    ]
    with (out / "candidates.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=candidate_fields)
        writer.writeheader()
        for c in candidates:
            writer.writerow(asdict(c))

    ticket_rows = []
    for rank, t in enumerate(tickets, 1):
        ticket_rows.append(
            {
                "rank": rank,
                "legs": len(t.legs),
                "joint_probability": t.joint_probability,
                "decimal_payout": t.decimal_payout,
                "expected_gross": t.expected_gross,
                "expected_profit": t.expected_profit,
                "score": t.score,
                "selections": " || ".join(
                    f"{x.label} @ {x.best_decimal:.3f} {x.best_book}" for x in t.legs
                ),
            }
        )
    with (out / "tickets.json").open("w", encoding="utf-8") as f:
        json.dump(
            [
                {
                    **{k: v for k, v in row.items() if k != "selections"},
                    "legs_detail": [asdict(x) for x in tickets[i].legs],
                }
                for i, row in enumerate(ticket_rows)
            ],
            f,
            indent=2,
            ensure_ascii=False,
        )
    with (out / "tickets.csv").open("w", encoding="utf-8", newline="") as f:
        fields = [
            "rank", "legs", "joint_probability", "decimal_payout",
            "expected_gross", "expected_profit", "score", "selections",
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(ticket_rows)
