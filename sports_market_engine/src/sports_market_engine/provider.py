from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Iterable

import httpx


class TheOddsAPI:
    BASE_URL = "https://api.the-odds-api.com/v4"

    def __init__(self, api_key: str, timeout: float = 30.0, concurrency: int = 8):
        self.api_key = api_key
        self.timeout = timeout
        self.concurrency = concurrency

    async def active_sports(self) -> list[dict]:
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.get(
                f"{self.BASE_URL}/sports",
                params={"apiKey": self.api_key},
            )
            response.raise_for_status()
            sports = response.json()
        return [s for s in sports if s.get("active") and not s.get("has_outrights")]

    async def odds_for_day(
        self,
        start_utc: datetime,
        end_utc: datetime,
        regions: Iterable[str],
        markets: Iterable[str],
    ) -> list[dict]:
        sports = await self.active_sports()
        sem = asyncio.Semaphore(self.concurrency)
        regions_value = ",".join(dict.fromkeys(regions))
        markets_value = ",".join(dict.fromkeys(markets))

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            async def fetch(sport_key: str) -> list[dict]:
                params = {
                    "apiKey": self.api_key,
                    "regions": regions_value,
                    "markets": markets_value,
                    "oddsFormat": "decimal",
                    "dateFormat": "iso",
                    "commenceTimeFrom": _iso_z(start_utc),
                    "commenceTimeTo": _iso_z(end_utc),
                }
                async with sem:
                    for attempt in range(4):
                        response = await client.get(
                            f"{self.BASE_URL}/sports/{sport_key}/odds",
                            params=params,
                        )
                        if response.status_code != 429:
                            response.raise_for_status()
                            return response.json()
                        await asyncio.sleep(2 ** attempt)
                return []

            batches = await asyncio.gather(*(fetch(str(s["key"])) for s in sports))

        events: list[dict] = []
        for batch in batches:
            events.extend(batch)
        # Provider filtering should already enforce boundaries; retain a defensive check.
        return [
            e for e in events
            if start_utc <= _parse_dt(str(e["commence_time"])) <= end_utc
        ]


def _parse_dt(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _iso_z(value: datetime) -> str:
    return value.astimezone().isoformat().replace("+00:00", "Z")
