from __future__ import annotations

import asyncio
from datetime import date, datetime, time, timezone
from typing import Any
from zoneinfo import ZoneInfo

import httpx

from ..models import Offer


class TheOddsAPIProvider:
    BASE_URL = "https://api.the-odds-api.com/v4"

    def __init__(
        self,
        api_key: str,
        *,
        regions: tuple[str, ...] = ("us", "us2", "uk", "eu", "au"),
        timeout_seconds: float = 30.0,
        concurrency: int = 8,
    ) -> None:
        if not api_key:
            raise ValueError("api_key is required")
        self.api_key = api_key
        self.regions = regions
        self.timeout = timeout_seconds
        self._sem = asyncio.Semaphore(max(1, concurrency))

    async def _get(self, client: httpx.AsyncClient, path: str, **params: Any) -> Any:
        params["apiKey"] = self.api_key
        async with self._sem:
            r = await client.get(
                f"{self.BASE_URL}{path}",
                params=params,
                timeout=self.timeout,
            )
            r.raise_for_status()
            return r.json()

    async def active_sports(self, client: httpx.AsyncClient) -> list[dict[str, Any]]:
        data = await self._get(client, "/sports")
        return [x for x in data if x.get("active") and not x.get("has_outrights")]

    @staticmethod
    def _parse_dt(value: str | None) -> datetime | None:
        if not value:
            return None
        return datetime.fromisoformat(value.replace("Z", "+00:00"))

    @classmethod
    def _flatten_event(cls, event: dict[str, Any]) -> list[Offer]:
        commence = cls._parse_dt(event.get("commence_time"))
        if commence is None:
            return []
        out: list[Offer] = []
        for book in event.get("bookmakers", []):
            book_key = book.get("key") or book.get("title") or "unknown"
            book_update = cls._parse_dt(book.get("last_update"))
            for market in book.get("markets", []):
                market_key = market.get("key", "unknown")
                market_update = cls._parse_dt(market.get("last_update")) or book_update
                for outcome in market.get("outcomes", []):
                    price = outcome.get("price")
                    if not isinstance(price, (int, float)) or price <= 1.0:
                        continue
                    point = outcome.get("point")
                    out.append(
                        Offer(
                            sport_key=event.get("sport_key", "unknown"),
                            event_id=event.get("id", ""),
                            commence_time=commence,
                            home_team=event.get("home_team", ""),
                            away_team=event.get("away_team", ""),
                            bookmaker=book_key,
                            market_key=market_key,
                            outcome_name=str(outcome.get("name", "")),
                            decimal_odds=float(price),
                            point=float(point) if isinstance(point, (int, float)) else None,
                            last_update=market_update,
                        )
                    )
        return out

    async def _featured_for_sport(
        self,
        client: httpx.AsyncClient,
        sport_key: str,
        markets: tuple[str, ...],
    ) -> list[dict[str, Any]]:
        return await self._get(
            client,
            f"/sports/{sport_key}/odds",
            regions=",".join(self.regions),
            markets=",".join(markets),
            oddsFormat="decimal",
            dateFormat="iso",
        )

    async def _deep_for_event(
        self,
        client: httpx.AsyncClient,
        sport_key: str,
        event_id: str,
        deep_market_cap: int,
    ) -> list[dict[str, Any]]:
        market_payload = await self._get(
            client,
            f"/sports/{sport_key}/events/{event_id}/markets",
            regions=",".join(self.regions),
            dateFormat="iso",
        )
        keys: list[str] = []
        seen: set[str] = set()
        for book in market_payload.get("bookmakers", []):
            for market in book.get("markets", []):
                key = market.get("key")
                if key and key not in seen:
                    seen.add(key)
                    keys.append(key)
                if len(keys) >= deep_market_cap:
                    break
            if len(keys) >= deep_market_cap:
                break
        if not keys:
            return []
        event = await self._get(
            client,
            f"/sports/{sport_key}/events/{event_id}/odds",
            regions=",".join(self.regions),
            markets=",".join(keys),
            oddsFormat="decimal",
            dateFormat="iso",
        )
        return [event]

    async def scan_day(
        self,
        day: date,
        *,
        timezone_name: str,
        markets: tuple[str, ...] = ("h2h", "spreads", "totals"),
        sports: set[str] | None = None,
        deep_markets: bool = False,
        deep_market_cap: int = 30,
        max_deep_events: int = 80,
    ) -> list[Offer]:
        tz = ZoneInfo(timezone_name)
        start_local = datetime.combine(day, time.min, tzinfo=tz)
        end_local = datetime.combine(day, time.max, tzinfo=tz)
        start_utc = start_local.astimezone(timezone.utc)
        end_utc = end_local.astimezone(timezone.utc)

        async with httpx.AsyncClient(
            headers={"User-Agent": "edgeforge/0.1"}
        ) as client:
            active = await self.active_sports(client)
            sport_keys = [
                s["key"] for s in active
                if not sports or s["key"] in sports
            ]

            results = await asyncio.gather(
                *[
                    self._featured_for_sport(client, key, markets)
                    for key in sport_keys
                ],
                return_exceptions=True,
            )

            events_by_id: dict[str, dict[str, Any]] = {}
            for result in results:
                if isinstance(result, Exception):
                    continue
                for event in result:
                    stamp = self._parse_dt(event.get("commence_time"))
                    if stamp and start_utc <= stamp <= end_utc:
                        events_by_id[event["id"]] = event

            payloads = list(events_by_id.values())
            if deep_markets and events_by_id:
                deep_targets = list(events_by_id.values())[:max_deep_events]
                deep = await asyncio.gather(
                    *[
                        self._deep_for_event(
                            client,
                            e["sport_key"],
                            e["id"],
                            deep_market_cap,
                        )
                        for e in deep_targets
                    ],
                    return_exceptions=True,
                )
                for result in deep:
                    if isinstance(result, Exception):
                        continue
                    payloads.extend(result)

        offers: list[Offer] = []
        seen: set[tuple] = set()
        for event in payloads:
            for offer in self._flatten_event(event):
                key = (
                    offer.bookmaker,
                    offer.selection_key,
                    offer.decimal_odds,
                    offer.last_update,
                )
                if key not in seen:
                    seen.add(key)
                    offers.append(offer)
        return offers
