"""Quota-aware sports market scanner and parlay research optimizer."""

from .engine import (
    Candidate,
    Ticket,
    TheOddsAPI,
    build_candidates,
    optimize_tickets,
    scan_day,
)

__all__ = [
    "Candidate",
    "Ticket",
    "TheOddsAPI",
    "build_candidates",
    "optimize_tickets",
    "scan_day",
]
