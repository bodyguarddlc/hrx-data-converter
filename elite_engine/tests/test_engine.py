import unittest

from elite_engine.engine import (
    build_candidates,
    devig_probabilities,
    optimize_tickets,
)


def _event(event_id, home, away, prices_a, prices_b):
    return {
        "id": event_id,
        "sport_key": "test_sport",
        "sport_title": "Test",
        "commence_time": "2026-10-03T20:00:00Z",
        "home_team": home,
        "away_team": away,
        "bookmakers": [
            {
                "key": "book_a",
                "markets": [
                    {
                        "key": "h2h",
                        "outcomes": [
                            {"name": home, "price": prices_a[0]},
                            {"name": away, "price": prices_a[1]},
                        ],
                    }
                ],
            },
            {
                "key": "book_b",
                "markets": [
                    {
                        "key": "h2h",
                        "outcomes": [
                            {"name": home, "price": prices_b[0]},
                            {"name": away, "price": prices_b[1]},
                        ],
                    }
                ],
            },
        ],
    }


class EngineTests(unittest.TestCase):
    def test_devig_sums_to_one(self):
        probs = devig_probabilities([1.80, 2.10])
        self.assertAlmostEqual(sum(probs), 1.0, places=12)
        self.assertGreater(probs[0], probs[1])

    def test_candidate_uses_best_price(self):
        events = [_event("e1", "A", "B", (1.90, 2.00), (1.95, 1.98))]
        candidates = build_candidates(events, min_books=2, confidence_z=0.0)
        a = next(c for c in candidates if c.outcome == "A")
        self.assertAlmostEqual(a.best_decimal, 1.95)
        self.assertEqual(a.best_book, "book_b")
        self.assertEqual(a.books, 2)

    def test_optimizer_avoids_same_event(self):
        events = [
            _event("e1", "A", "B", (2.15, 1.80), (2.20, 1.78)),
            _event("e2", "C", "D", (2.10, 1.82), (2.18, 1.80)),
            _event("e3", "E", "F", (2.05, 1.85), (2.12, 1.82)),
        ]
        candidates = build_candidates(events, min_books=2, confidence_z=0.0)
        tickets = optimize_tickets(
            candidates,
            min_legs=2,
            max_legs=3,
            top_n=10,
            min_edge=-1.0,
            min_leg_probability=0.30,
            max_leg_decimal=4.0,
        )
        self.assertTrue(tickets)
        for ticket in tickets:
            ids = [leg.event_id for leg in ticket.legs]
            self.assertEqual(len(ids), len(set(ids)))


if __name__ == "__main__":
    unittest.main()
