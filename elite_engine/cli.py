from __future__ import annotations

import argparse
import os
from datetime import date

from .engine import (
    ALL_REGIONS,
    TheOddsAPI,
    build_candidates,
    optimize_tickets,
    scan_day,
    write_outputs,
)


def _resolve_day(value: str, timezone_name: str) -> date:
    if value.lower() == "today":
        return datetime.now(ZoneInfo(timezone_name)).date()
    return date.fromisoformat(value)


def _parse_regions(value: str) -> tuple[str, ...]:
    if value.strip().lower() == "all":
        return ALL_REGIONS
    regions = tuple(x.strip() for x in value.split(",") if x.strip())
    unknown = [x for x in regions if x not in ALL_REGIONS]
    if unknown:
        raise argparse.ArgumentTypeError(
            f"unknown regions: {', '.join(unknown)}; valid: {', '.join(ALL_REGIONS)}"
        )
    return regions


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="elite-engine",
        description="Quota-aware same-day sports odds scanner and ticket research optimizer.",
    )
    p.add_argument("--date", default="today")
    p.add_argument("--timezone", default="America/Los_Angeles")
    p.add_argument(
        "--regions",
        type=_parse_regions,
        default=("us", "eu"),
        help="Comma-separated regions or 'all'. Default: us,eu",
    )
    p.add_argument("--max-credits", type=int, default=80)
    p.add_argument("--deep", action="store_true", help="Discover and price non-featured event markets.")
    p.add_argument("--deep-market-cap", type=int, default=12)
    p.add_argument("--max-deep-events", type=int, default=8)
    p.add_argument("--moneyline-only", action="store_true")
    p.add_argument("--min-books", type=int, default=2)
    p.add_argument("--confidence-z", type=float, default=0.40)
    p.add_argument("--min-legs", type=int, default=2)
    p.add_argument("--max-legs", type=int, default=4)
    p.add_argument("--top", type=int, default=25)
    p.add_argument("--min-edge", type=float, default=0.01)
    p.add_argument("--min-leg-prob", type=float, default=0.52)
    p.add_argument("--max-leg-decimal", type=float, default=3.50)
    p.add_argument("--hit-weight", type=float, default=1.0)
    p.add_argument("--payout-weight", type=float, default=1.0)
    p.add_argument("--allow-same-event", action="store_true")
    p.add_argument("--out", default="elite_engine/output")
    return p


def main() -> int:
    args = build_parser().parse_args()
    api_key = os.environ.get("THE_ODDS_API_KEY", "")
    provider = TheOddsAPI(api_key)
    selected_day = _resolve_day(args.date, args.timezone)

    scan = scan_day(
        provider,
        day=selected_day,
        timezone_name=args.timezone,
        regions=args.regions,
        max_credits=args.max_credits,
        include_spreads_totals=not args.moneyline_only,
        deep=args.deep,
        deep_market_cap=args.deep_market_cap,
        max_deep_events=args.max_deep_events,
    )
    candidates = build_candidates(
        scan.events,
        min_books=args.min_books,
        confidence_z=args.confidence_z,
    )
    tickets = optimize_tickets(
        candidates,
        min_legs=args.min_legs,
        max_legs=args.max_legs,
        top_n=args.top,
        min_edge=args.min_edge,
        min_leg_probability=args.min_leg_prob,
        max_leg_decimal=args.max_leg_decimal,
        one_leg_per_event=not args.allow_same_event,
        hit_weight=args.hit_weight,
        payout_weight=args.payout_weight,
    )
    write_outputs(args.out, scan=scan, candidates=candidates, tickets=tickets)

    print(
        f"events={len(scan.events)} candidates={len(candidates)} "
        f"tickets={len(tickets)} est_credits={scan.credits_used_estimate} "
        f"remaining={scan.credits_remaining_header}"
    )
    for rank, ticket in enumerate(tickets[:10], 1):
        print(
            f"#{rank} P={ticket.joint_probability:.3%} "
            f"payout={ticket.decimal_payout:.2f}x "
            f"EV={ticket.expected_profit:+.2%} score={ticket.score:.4f}"
        )
        for leg in ticket.legs:
            print(
                f"  - {leg.label} @ {leg.best_decimal:.3f} {leg.best_book} "
                f"p={leg.fair_probability:.2%} edge={leg.edge:+.2%}"
            )
    if scan.skipped:
        print(f"budget_skips={len(scan.skipped)}; inspect {args.out}/scan.json")
    print(f"wrote {args.out}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
