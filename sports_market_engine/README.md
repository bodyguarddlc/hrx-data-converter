# Sports Market Engine

Same-day multi-region market scanner that turns bookmaker prices into de-vigged consensus probabilities and searches for slips on the hit-rate/payout Pareto frontier.

## What it does

- Pulls all active, non-outright sports from The Odds API.
- Scans a specific local calendar day across US, UK, EU and Australian bookmaker regions.
- Supports featured moneyline/head-to-head, spread and total markets.
- De-vigs each bookmaker market, then uses the median fair probability across books.
- Keeps the best available decimal price for each matching outcome.
- Filters low-probability / negative-edge legs.
- Defaults to one leg per event so same-game legs are not incorrectly treated as independent.
- Maintains the exact hit-rate/payout Pareto frontier for every combination it evaluates.
- Ranks frontier slips with a tunable multiplicative objective:
  `hit_probability^hit_weight × payout^payout_weight × edge_factor^edge_weight`.

This optimizes model estimates, not certainty. No odds model can guarantee wins or profit.

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

The `today` value is resolved in the timezone supplied to `--timezone`, not the host machine's timezone.

## Exact vs bounded search

By default the optimizer searches the top 80 filtered single-leg candidates, then exactly maintains the hit-rate/payout Pareto frontier across all combinations in that pool.

For an exhaustive search over every filtered leg:

```bash
sports-scan --candidate-pool 0
```

That can become expensive when thousands of legs are available.

## Tuning the objective

Higher `--hit-weight` favors safer combinations. Higher `--payout-weight` favors larger multipliers. `--edge-weight` favors outcomes where the consensus fair probability beats the best listed price.

A probability-heavy starting point:

```bash
sports-scan --hit-weight 1.5 --payout-weight 0.6 --edge-weight 0.25
```

## Architecture

- `provider.py`: live multi-region odds ingestion.
- `pricing.py`: implied probability, de-vigging, consensus pricing.
- `optimizer.py`: constraints, exact Pareto maintenance, objective scoring.
- `scanner.py`: local-day/timezone orchestration.
- `cli.py`: command-line interface.

## Next upgrades

1. Add exchange / sharper-book adapters and weight books by historical closing-line accuracy.
2. Add historical snapshots for calibration and walk-forward backtests.
3. Estimate same-game and cross-market correlations instead of forbidding same-game legs.
4. Add injury/news feature inputs and sport-specific predictive models.
5. Track line movement, stale-book latency and best-price persistence.
6. Replace independent cross-event hit probability with a covariance/copula model where data supports it.
