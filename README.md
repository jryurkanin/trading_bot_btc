# trading_bot_btc

Production-oriented Bitcoin strategy codebase for Coinbase Advanced Trade with:

- backtesting
- paper trading / dry-run
- live execution
- risk controls, regime switching, and stateful persistence

> Long-only by design; no shorting assumptions are used.

## Features

- **Data layer**: hourly and daily BTC candles with pagination, UTC normalization, and local sqlite cache.
- **Public inputs (optional)**: Fear & Greed + Blockchain.com charts/stats, behind feature flags.
- **Regime detection**:
  - daily macro regime (`RISK_ON` / `RISK_OFF`)
  - hourly micro regime (TREND / RANGE / HIGH_VOL / NEUTRAL)
  - optional HMM regime switcher (`hmm_regime_enabled`)
- **Strategies**:
  - regime-switching orchestrator
  - RANGE mean-reversion (Bollinger)
  - TREND breakout (Donchian / EMA cross)
- **Risk engine**:
  - drawdown breaker + stale data breaker
  - daily loss / consecutive-loss kill switches
  - manual kill switch + safe-mode flatten
  - vol-target sizing and caps
- **Execution**:
  - exchange constraint enforcement (min size/notional + increment rounding)
  - limit-first placement + fallback-to-market
  - order lifecycle persistence (partial fill states)
  - deterministic cancel/replace on timeout
  - idempotent client_order_id + paper mode simulator
- **Operational hardening**:
  - health status file + alert log output
  - cycle failure / order failure threshold alerts
- **Backtesting**:
  - fee-aware + slippage
  - equity curve, trade log, metrics
  - regime-attribution analytics
  - walk-forward testing scaffold

## Repository Layout

```text
.
├── src/bot
│   ├── coinbase_client.py
│   ├── config.py
│   ├── data/
│   ├── features/
│   ├── strategy/
│   ├── backtest/
│   ├── execution/
│   ├── live/
├── scripts/
│   ├── backtest.py
│   ├── optimize.py
│   ├── frontier_sweep.py
│   ├── frontier_sweep_v3.py
│   └── trade.py
├── tests/
└── README.md
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Create environment vars for trading:

```bash
export COINBASE_API_KEY=...
export COINBASE_API_SECRET=...
# optional
export COINBASE_API_PASSPHRASE=...
```

## Quickstart

### 1) Backtest

```bash
python scripts/backtest.py \
  --product BTC-USD \
  --start 2020-01-01T00:00:00Z \
  --end 2026-01-31T00:00:00Z \
  --initial-equity 10000
```

This writes:
- `reports/equity_curve.csv`
- `reports/trades.csv`
- `reports/decisions.csv`
- `reports/macro_bucket_attribution.csv`
- `reports/report.json`

`report.json` includes `macro_bucket_attribution` with per-bucket (`OFF`, `ON_HALF`, `ON_FULL`) time, exposure, PnL, fees, turnover, trade count, and warnings.

### Strategy variants (backward compatible)

- `regime_switching`: legacy binary macro gate behavior
- `regime_switching_v2`: score-based macro scaling + trend booster
- `regime_switching_v3`: stateful daily macro gate (`OFF` / `ON_HALF` / `ON_FULL`) with hysteresis + directional trend boost

Defaults preserve prior behavior unless you explicitly choose `regime_switching_v2` or `regime_switching_v3`.

Enable v2 behavior (score-based macro + trend boost):

```bash
python scripts/backtest.py \
  --product BTC-USD \
  --start 2021-01-01T00:00:00Z \
  --end 2026-01-31T00:00:00Z \
  --strategy regime_switching_v2 \
  --fill-model bid_ask \
  --macro-mode score \
  --macro-score-floor 0.25 \
  --macro-score-min-to-trade 0.25 \
  --trend-boost-enabled \
  --trend-boost-multiplier 1.25 \
  --trend-boost-adx-threshold 25
```

Enable v3 behavior (stateful macro gate + bucket multipliers + directional boost):

```bash
python scripts/backtest.py \
  --product BTC-USD \
  --start 2021-01-01T00:00:00Z \
  --end 2026-01-31T00:00:00Z \
  --strategy regime_switching_v3 \
  --fill-model bid_ask \
  --macro-mode stateful_gate \
  --macro-enter-threshold 0.75 \
  --macro-exit-threshold 0.25 \
  --macro-confirm-days 2 \
  --macro-min-on-days 2 \
  --macro-min-off-days 1 \
  --macro-half-multiplier 0.5 \
  --macro-full-multiplier 1.0 \
  --trend-boost-enabled \
  --trend-boost-multiplier 1.10 \
  --trend-boost-adx-threshold 25 \
  --trend-boost-confirm-days 2 \
  --trend-boost-min-on-days 2 \
  --trend-boost-min-off-days 1
```

### 2) Walk-forward sweep (legacy)

```bash
python scripts/optimize.py \
  --product BTC-USD \
  --start 2020-01-01T00:00:00Z \
  --end 2026-01-31T00:00:00Z \
  --grid adx_trend_threshold=20,25,30 \
  --grid range_tranche_size=0.2,0.25
```

### 3) Frontier sweep (walk-forward + cost stress)

General sweep (legacy/v2 grid):

```bash
python scripts/frontier_sweep.py \
  --product BTC-USD \
  --strategy regime_switching_v2 \
  --fill-model bid_ask \
  --start 2021-01-01T00:00:00Z \
  --end 2026-01-31T00:00:00Z
```

V3-specific sweep (stateful gate + directional boost parameter space):

```bash
python scripts/frontier_sweep_v3.py \
  --product BTC-USD \
  --fill-model bid_ask \
  --start 2021-01-01T00:00:00Z \
  --end 2026-01-31T00:00:00Z
```

Outputs:
- `artifacts/frontier*/summary.csv` — one row per (params, window, scenario)
- `artifacts/frontier*/frontier.csv` — ranked top configs
- `artifacts/frontier*/best_config.json` — recommended config + reproduce command

Interpretation:
- ranking favors robust validation performance under stress costs,
- then lower drawdown/turnover,
- then higher Sharpe.

### 4) Paper trade (2+ cycles)

```bash
python scripts/trade.py --paper --product BTC-USD --cycles 2
```

### 5) Live (sandbox)

```bash
python scripts/trade.py --live --product BTC-USD --sandbox
```

Live mode requires valid Coinbase credentials and careful monitoring.

## Notes

- Static sandbox endpoint is supported in `coinbase_client.py` (set `use_sandbox=true` in config or `COINBASE_USE_SANDBOX=true`).
- Public data sources are optional and can be toggled in config.
- Tests expect no secrets or network access and focus on determinism in pure components.

## Tests

```bash
pytest -q
```
