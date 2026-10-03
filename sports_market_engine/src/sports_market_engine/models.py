from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime


@dataclass(frozen=True)
class Leg:
    event_id: str
    sport_key: str
    event_name: str
    commence_time: datetime
    market: str
    selection: str
    point: float | None
    bookmaker: str
    decimal_odds: float
    fair_probability: float
    edge: float

    @property
    def key(self) -> tuple[str, str, str, float | None]:
        return (self.event_id, self.market, self.selection, self.point)

    def to_dict(self) -> dict:
        out = asdict(self)
        out["commence_time"] = self.commence_time.isoformat()
        return out


@dataclass(frozen=True)
class Slip:
    legs: tuple[Leg, ...]
    hit_probability: float
    payout_multiple: float
    expected_return: float
    score: float

    def to_dict(self) -> dict:
        return {
            "legs": [leg.to_dict() for leg in self.legs],
            "hit_probability": self.hit_probability,
            "payout_multiple": self.payout_multiple,
            "expected_return": self.expected_return,
            "score": self.score,
        }
