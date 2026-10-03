from __future__ import annotations

import argparse
import asyncio
import json
import os
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from .optimizer import OptimizerConfig
from .scanner import scan_day


def _csv_tuple(value: str) -> tuple[str, ...]:
    return tuple(x.strip() for x in value.split(",") if x.strip())


def _int_tuple(value: str) -> tuple[int, ...]:
    return tuple(int(x.strip()) for x in value.split(",") if x.strip())


def _resolve_date(value: str, timezone_name: str) -> date:
    if value.lower() == "today":
        return datetime.now(ZoneInfo(timezone_name)).date()
    return date.fromisoformat(value)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Scan same-day multi-region sports markets and optimize hit-rate/payout slips."
    )
    p.add_argument("--date", default="today", help="YYYY-MM-DD or 'today'")
    p.add_argument("--timezone", default="America/Los_Angeles")
    p.add_argument("--regions", default="us,us2,uk,eu,au", type=_csv_tuple)
    p.add_argument("--markets", default="h2h,spreads,totals", type=_csv_tuple)
    p.add_argument("--legs", default="2,3,4", type=_int_tuple)
    p.add_argument("--top", default=20, type=int)
    p.add_argument("--min-books", default=2, type=int)
    p.add_argument("--min-prob", default=0.50, type=float)
    p.add_argument("--min-edge", default=0.0, type=float)
    p.add_argument(
        "--candidate-pool",
        default=80,
        type=int,
        help="Top single-leg candidates to search; 0 means exhaustive over all filtered legs.",
    )
    p.add_argument("--hit-weight", default=1.35, type=float)
    p.add_argument("--payout-weight", default=0.65, type=float)
    p.add_argument("--edge-weight", default=0.20, type=float)
    p.add_argument("--allow-same-event", action="store_true")
    p.add_argument("--output", type=Path)
    return p


async def _run(args: argparse.Namespace) -> int:
    api_key = os.getenv("ODDS_API_KEY")
    if not api_key:
        raise SystemExit("ODDS_API_KEY is required")

    target_date = _resolve_date(args.date, args.timezone)
    config = OptimizerConfig(
        min_fair_probability=args.min_prob,
        min_edge=args.min_edge,
        candidate_pool=args.candidate_pool,
        one_leg_per_event=not args.allow_same_event,
        hit_weight=args.hit_weight,
        payout_weight=args.payout_weight,
        edge_weight=args.edge_weight,
    )
    result = await scan_day(
        api_key=api_key,
        target_date=target_date,
        timezone_name=args.timezone,
        regions=args.regions,
        markets=args.markets,
        leg_counts=args.legs,
        top_n=args.top,
        min_books=args.min_books,
        config=config,
    )

    payload = {
        "date": target_date.isoformat(),
        "timezone": args.timezone,
        "events_scanned": result.events_scanned,
        "candidate_legs": result.candidate_legs,
        "slips": [x.to_dict() for x in result.slips],
    }

    if args.output:
        args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(json.dumps(payload, indent=2))
    return 0


def main() -> None:
    args = build_parser().parse_args()
    raise SystemExit(asyncio.run(_run(args)))


if __name__ == "__main__":
    main()
