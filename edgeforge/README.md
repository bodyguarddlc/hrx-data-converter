# EdgeForge

EdgeForge is an open-source, provider-agnostic engine for scanning same-day sports betting markets, removing bookmaker margin, comparing prices across books, and constructing a ticket on the **Pareto frontier of modeled hit probability vs payout**.

It does **not** claim guaranteed wins. Coverage is limited by the configured data provider, its bookmakers/regions, market availability, API quota, and your probability model.

## What v0.1 does

- discovers active sports from The Odds API v4
- scans same-day featured markets across configurable regions (US, US2, UK, EU, AU)
- optionally discovers deeper event-level markets
- converts decimal prices to implied probabilities
- removes overround per bookmaker/market using multiplicative normalization
- forms a robust cross-book fair-probability consensus with the median de-vig probability
- keeps the best available price for an identical selection
- filters stale, thin, low-edge, or low-EV selections
- searches ticket combinations with a beam search
- enforces a minimum modeled ticket hit-rate and leg-count bounds
- blocks multiple legs from the same event by default to avoid pretending independence
- returns the Pareto frontier and a configurable balanced choice

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e "./edgeforge[dev]"
export ODDS_API_KEY="your-key"
```

## Scan today's slate

```bash
edgeforge scan \
  --timezone America/Los_Angeles \
  --regions us,us2,uk,eu,au \
  --markets h2h,spreads,totals \
  --min-edge 0.015 \
  --min-ev 0.01 \
  --min-joint-prob 0.16 \
  --min-legs 2 \
  --max-legs 5
```

For provider-supported deeper markets:

```bash
edgeforge scan --deep-markets --deep-market-cap 30
```

Deep scans can consume substantially more provider quota.

## Optimization

For a ticket with legs (i=1..n):

- modeled hit probability: (P = \prod_i p_i)
- decimal payout: (D = \prod_i d_i)
- expected gross return: (P \times D)
- expected net return: (P \times D - 1)

There is no single ticket that mathematically "maximizes hit rate and payout" in all cases because those objectives conflict. EdgeForge first computes non-dominated tickets (the Pareto frontier), then selects from that frontier using:

```
score = hit_weight * log(P) + payout_weight * log(D)
```

The default weights favor hit rate. Change them to match your objective. A ticket is never accepted below `--min-joint-prob`.

## Probability model

The default v0.1 probability estimate is a **cross-book de-vig consensus**, not a predictive sports model. It is deliberately conservative and useful for line shopping, but it is not independent of the market.

Domain models can replace consensus probabilities before optimization. The repository's existing baseball modeling work can be wired in as a model adapter in a later module without changing the optimizer.

## Correlation

Independence is a dangerous assumption for same-game legs. EdgeForge therefore permits at most one leg per event by default. Use `--allow-same-event` only if you supply a defensible joint-probability/correlation model.

## Output

The CLI emits JSON containing:

- scan metadata
- candidate legs with fair probability, offered price, edge and EV
- the selected balanced ticket
- up to 20 Pareto-frontier alternatives

Use `--output result.json` to save it.

## Development

```bash
pip install -e "./edgeforge[dev]"
pytest edgeforge/tests
```

## License

MIT. See [LICENSE](LICENSE).
