from __future__ import annotations

import argparse
import asyncio
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from .optimizer import OptimizeConfig, optimize
from .pricing import build_candidates
from .providers import TheOddsAPIProvider


def _csv(value: str) -> tuple[str, ...]:
    return tuple(x.strip() for x in value.split(",") if x.strip())


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="edgeforge")
    sub = p.add_subparsers(dest="command", required=True)
    s = sub.add_parser("scan", help="scan a day's markets and build a ticket")
    s.add_argument("--date", help="YYYY-MM-DD; default is today in --timezone")
    s.add_argument("--timezone", default="America/Los_Angeles")
    s.add_argument("--regions", default="us,us2,uk,eu,au")
    s.add_argument("--markets", default="h2h,spreads,totals")
    s.add_argument("--sports", default="", help="comma-separated provider sport keys")
    s.add_argument("--deep-markets", action="store_true")
    s.add_argument("--deep-market-cap", type=int, default=30)
    s.add_argument("--max-deep-events", type=int, default=80)
    s.add_argument("--max-age-seconds", type=int, default=300)
    s.add_argument("--min-books", type=int, default=2)
    s.add_argument("--min-edge", type=float, default=0.01)
    s.add_argument("--min-ev", type=float, default=0.0)
    s.add_argument("--min-joint-prob", type=float, default=0.15)
    s.add_argument("--min-legs", type=int, default=2)
    s.add_argument("--max-legs", type=int, default=5)
    s.add_argument("--hit-weight", type=float, default=0.65)
    s.add_argument("--payout-weight", type=float, default=0.35)
    s.add_argument("--beam-width", type=int, default=2500)
    s.add_argument("--max-candidates", type=int, default=60)
    s.add_argument("--allow-same-event", action="store_true")
    s.add_argument("--output")
    return p


async def _scan(args: argparse.Namespace) -> dict:
    tz = ZoneInfo(args.timezone)
    day = (
        datetime.strptime(args.date, "%Y-%m-%d").date()
        if args.date
        else datetime.now(tz).date()
    )
    api_key = os.getenv("ODDS_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("ODDS_API_KEY is required")

    provider = TheOddsAPIProvider(api_key, regions=_csv(args.regions))
    offers = await provider.scan_day(
        day,
        timezone_name=args.timezone,
        markets=_csv(args.markets),
        sports=set(_csv(args.sports)) or None,
        deep_markets=args.deep_markets,
        deep_market_cap=max(1, args.deep_market_cap),
        max_deep_events=max(1, args.max_deep_events),
    )
    candidates = build_candidates(
        offers,
        min_books=max(1, args.min_books),
        max_age_seconds=None if args.max_age_seconds < 0 else args.max_age_seconds,
        now=datetime.now(timezone.utc),
    )
    cfg = OptimizeConfig(
        min_legs=args.min_legs,
        max_legs=args.max_legs,
        min_joint_probability=args.min_joint_prob,
        min_edge=args.min_edge,
        min_ev=args.min_ev,
        hit_weight=args.hit_weight,
        payout_weight=args.payout_weight,
        beam_width=args.beam_width,
        max_candidates=args.max_candidates,
        distinct_event=not args.allow_same_event,
    )
    best, frontier = optimize(candidates, cfg)

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "date": day.isoformat(),
        "timezone": args.timezone,
        "regions": list(_csv(args.regions)),
        "markets": list(_csv(args.markets)),
        "deep_markets": bool(args.deep_markets),
        "raw_offer_count": len(offers),
        "candidate_count": len(candidates),
        "candidate_preview": [c.to_dict() for c in candidates[:50]],
        "selected_ticket": best.to_dict() if best else None,
        "pareto_frontier": [t.to_dict() for t in frontier[:20]],
        "notes": [
            "Fair probabilities are cross-book de-vig consensus unless replaced by an external model.",
            "Same-event legs are blocked by default because multiplying marginal probabilities assumes independence.",
            "Provider coverage and quota determine how close the scan is to all available markets.",
        ],
    }


def main() -> None:
    args = _parser().parse_args()
    if args.command != "scan":
        raise SystemExit(2)
    payload = asyncio.run(_scan(args))
    text = json.dumps(payload, indent=2, sort_keys=False)
    if args.output:
        Path(args.output).write_text(text + "\n", encoding="utf-8")
    else:
        print(text)


if __name__ == "__main__":
    main()
