# Sports Market Engine

Same-day multi-region market scanner that turns bookmaker prices into de-vigged consensus probabilities and searches for slips on the hit-rate/payout Pareto frontier.

## What it does

- Pulls all active, non-outright sports from The Odds API.
- Scans a specific local calendar day across US, UK, EU and Australian bookmaker regions.
- Supports featured moneyline/head-to-head, spread and total markets.
- De-vigs each bookmaker market, then uses the median fair probability across books.
- Keeps the best available decimal price for each matching outcome.
- Filters low-probability / negative-edge legs.
- Defaults to one leg per event to avoid pretending same-game correlations are independent.
- Ranks 2/3/4-leg slips using a tunable multiplicative objective:
  `hit_probability^hit_weight × payout^payout_weight × edge_factor^edge_weight`.
- Returns only non-dominated hit-rate/payout combinations.

This optimizes estimates, not certainty. No odds model can guarantee wins or profit.

## Install

```bash
cd sports_market_engine
python -m venv .venv
source .venv/bin/activate
pip install -e .
export ODDS_API_KEY=...
```

## Scan today

```bash
sports-scan \
  --date today \
  --timezone America/Los_Angeles \
  --regions us,us2,uk,eu,au \
  --markets h2h,spreads,totals \
  --legs 2,3,4 \
  --top 20 \
  --output picks.json
```

## Tuning the objective

Higher `--hit-weight` favors safer combinations. Higher `--payout-weight` favors larger multipliers. `--edge-weight` favors outcomes where the consensus fair probability beats the best listed price.

A useful starting point:

```bash
sports-scan --hit-weight 1.5 --payout-weight 0.6 --edge-weight 0.25
```

## Architecture

- `provider.py`: live odds ingestion.
- `pricing.py`: implied probability, de-vigging, consensus pricing.
- `optimizer.py`: constraints, Pareto filtering and objective scoring.
- `scanner.py`: day/timezone orchestration.
- `cli.py`: command-line interface.

## Next upgrades

1. Add exchange / sharp-book adapters and weight books by closing-line accuracy.
2. Add historical snapshots for calibration and walk-forward backtests.
3. Estimate same-game and cross-market correlations instead of forbidding them.
4. Add injury/news feature inputs and sport-specific models.
5. Track line movement and stale-book latency.
6. Replace independent-leg hit probability with a covariance/copula model.
