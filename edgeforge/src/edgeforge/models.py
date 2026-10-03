from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any


@dataclass(frozen=True, slots=True)
class Offer:
    sport_key: str
    event_id: str
    commence_time: datetime
    home_team: str
    away_team: str
    bookmaker: str
    market_key: str
    outcome_name: str
    decimal_odds: float
    point: float | None = None
    last_update: datetime | None = None

    @property
    def line_bucket(self) -> float | None:
        if self.point is None:
            return None
        if self.market_key in {"spreads", "totals"}:
            return abs(round(float(self.point), 4))
        return round(float(self.point), 4)

    @property
    def selection_key(self) -> tuple[Any, ...]:
        return (
            self.event_id,
            self.market_key,
            self.outcome_name,
            None if self.point is None else round(float(self.point), 4),
        )

    @property
    def market_instance_key(self) -> tuple[Any, ...]:
        return (self.event_id, self.market_key, self.line_bucket)


@dataclass(frozen=True, slots=True)
class Candidate:
    offer: Offer
    fair_probability: float
    implied_probability: float
    edge: float
    expected_value: float
    books_observed: int

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["offer"]["commence_time"] = self.offer.commence_time.isoformat()
        if self.offer.last_update:
            data["offer"]["last_update"] = self.offer.last_update.isoformat()
        return data


@dataclass(frozen=True, slots=True)
class Ticket:
    legs: tuple[Candidate, ...]
    joint_probability: float
    decimal_payout: float
    expected_net_return: float
    score: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "joint_probability": self.joint_probability,
            "decimal_payout": self.decimal_payout,
            "expected_net_return": self.expected_net_return,
            "score": self.score,
            "legs": [leg.to_dict() for leg in self.legs],
        }
