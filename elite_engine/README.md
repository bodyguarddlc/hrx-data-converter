# Elite Odds Engine

A free, standard-library Python engine for same-day sports-market research. It discovers every active sport with an event on the requested local calendar day, prices available markets from multiple bookmaker regions, removes bookmaker margin, line-shops the best price, and ranks multi-event tickets using conservative probability x payout.

It does **not** guarantee profit or a hit rate. Sportsbooks change prices, provider coverage is incomplete, and betting outcomes remain uncertain. The engine is intentionally conservative about same-event correlation.

## Why this design

- **Coverage before depth.** The Odds API sports and events endpoints are used first so the engine can discover same-day events without spending odds credits.
- **Domestic + foreign.** Regions can be any combination of us, us2, uk, eu, au, or all.
- **Free-tier aware.** A hard per-run credit budget prevents accidental quota burn.
- **De-vigged consensus.** Each bookmaker's mutually exclusive market is normalized to remove overround, then consensus probability is taken across books.
- **Disagreement penalty.** A conservative fair probability is reduced when books disagree.
- **Best-price shopping.** The engine uses the highest decimal price for the exact same selection/line.
- **Correlation guard.** By default, no ticket contains more than one leg from the same event.
- **Hit-rate x payout.** The default ticket score is joint fair probability multiplied by decimal payout. Both exponents are configurable.

## Provider

The default adapter uses The Odds API v4. As of October 2026, its public free plan advertises 500 credits/month, all sports, most bookmakers and all betting markets. Featured odds cost one credit per requested market x region. Same-day event discovery does not count against the usage quota. Event market discovery costs one credit per event.

Create a free API key with the provider and expose it as an environment variable:

~~~bash
export THE_ODDS_API_KEY="your-key"
~~~

No Python packages are required.

## Quick start

From the repository root:

~~~bash
python -m elite_engine --date today --regions us,eu --max-credits 80
~~~

Worldwide bookmaker regions:

~~~bash
python -m elite_engine --date today --regions all --max-credits 250
~~~

Moneyline/1X2 only, which is much cheaper and maximizes broad coverage:

~~~bash
python -m elite_engine --date today --regions all --moneyline-only --max-credits 100
~~~

Enable expensive non-featured markets such as props/alternate lines when budget remains:

~~~bash
python -m elite_engine --date today --regions us,eu --deep --max-credits 150
~~~

Outputs are written to elite_engine/output:

- scan.json — raw merged market data plus quota/skipped notes
- candidates.json / candidates.csv — exact selections with consensus probability, conservative probability, dispersion, best book/price, and edge
- tickets.json / tickets.csv — ranked multi-event combinations

## Useful controls

~~~text
--min-books 2
--confidence-z 0.40
--min-edge 0.01
--min-leg-prob 0.52
--max-leg-decimal 3.50
--min-legs 2
--max-legs 4
--hit-weight 1.0
--payout-weight 1.0
--allow-same-event
~~~

Keep --allow-same-event off unless you replace the independence assumption with a sport-specific joint-probability model.

## What the score means

For independent cross-event legs:

~~~text
joint_probability = product(conservative fair probability of each leg)
decimal_payout    = product(best decimal price of each leg)
score             = joint_probability^hit_weight * decimal_payout^payout_weight
expected_profit   = joint_probability * decimal_payout - 1
~~~

With both weights at 1.0, score is expected gross return. This is a ranking statistic, not a promise that the probability estimates are correct.

## Free-tier reality

"All current markets worldwide" and "zero cost forever" conflict when the upstream data provider charges quota per region/market. The engine handles that by:

1. discovering every same-day sport/event first at zero odds-credit cost,
2. pricing h2h/1X2 broadly before spending on spreads/totals,
3. spending remaining quota on deeper markets,
4. recording every market skipped by the configured budget.

For sustainable daily usage, reduce regions, use --moneyline-only, or add another lawful free data adapter rather than scraping sportsbook sites against their terms.

## Tests

~~~bash
python -m unittest discover elite_engine/tests -v
~~~
