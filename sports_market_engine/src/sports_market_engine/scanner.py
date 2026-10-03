from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo

from .models import Leg, Slip
from .optimizer import OptimizerConfig, optimize
from .pricing import extract_legs
from .provider import TheOddsAPI


@dataclass(frozen=True)
class ScanResult:
    events_scanned: int
    candidate_legs: int
    slips: list[Slip]


async def scan_day(
    api_key: str,
    target_date: date,
    timezone_name: str,
    regions: tuple[str, ...],
    markets: tuple[str, ...],
    leg_counts: tuple[int, ...],
    top_n: int,
    min_books: int = 2,
    config: OptimizerConfig = OptimizerConfig(),
) -> ScanResult:
    tz = ZoneInfo(timezone_name)
    local_start = datetime.combine(target_date, time.min, tzinfo=tz)
    local_end = local_start + timedelta(days=1) - timedelta(microseconds=1)
    start_utc = local_start.astimezone(timezone.utc)
    end_utc = local_end.astimezone(timezone.utc)

    provider = TheOddsAPI(api_key=api_key)
    events = await provider.odds_for_day(
        start_utc=start_utc,
        end_utc=end_utc,
        regions=regions,
        markets=markets,
    )
    legs: list[Leg] = extract_legs(events, min_books=min_books)
    slips = optimize(legs, leg_counts=leg_counts, top_n=top_n, config=config)

    return ScanResult(
        events_scanned=len(events),
        candidate_legs=len(legs),
        slips=slips,
    )
