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
  - **macro_gate_benchmark**
  - **macro_only_v2**
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
│   ├── frontier_sweep_macro_only.py
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

Optional FRED key (required only when FRED overlay is enabled):

```bash
export FRED_API_KEY=your_fred_key
```

### Optional CUDA acceleration

Backtests can use CUDA for rolling-volatility feature pipelines when a compatible GPU
and CuPy are available.

```bash
# example for CUDA 12.x environments
pip install cupy-cuda12x
```

Then run with:

```bash
--acceleration-backend cuda
```

If CUDA is unavailable, the engine falls back to CPU and records the fallback reason in diagnostics.

### Optional FRED macro overlay

FRED integration is **off by default** for safety. When enabled, it:
- fetches public macro/financial condition series from the official FRED API,
- caches responses locally,
- applies conservative availability lags (daily/weekly/monthly),
- builds `fred_risk_off_score` and `fred_penalty_multiplier`,
- scales macro score as a headwind/tailwind overlay (no micro-regime switching changes).

Enable via config (`fred.enabled=true`) or CLI flags on backtests:

```bash
python scripts/backtest.py \
  --product BTC-USD \
  --start 2021-01-01T00:00:00Z \
  --end 2026-01-31T00:00:00Z \
  --fred-enabled \
  --fred-max-risk-off-penalty 0.5 \
  --fred-risk-off-score-ema-span 16 \
  --fred-lag-stress-multiplier 1.0
```

`report.json` includes a `fred` section with series provenance, lags, weights, warnings, and cache stats.

## Quickstart

### 1) Backtest

```bash
python scripts/backtest.py \
  --product BTC-USD \
  --start 2020-01-01T00:00:00Z \
  --end 2026-01-31T00:00:00Z \
  --initial-equity 10000 \
  --acceleration-backend auto
```

This writes:
- `reports/equity_curve.csv`
- `reports/trades.csv`
- `reports/decisions.csv`
- `reports/macro_bucket_attribution.csv`
- `reports/report.json`

`report.json` includes `macro_bucket_attribution` with per-bucket (`OFF`, `ON_HALF`, `ON_FULL`) time, exposure, PnL, fees, turnover, trade count, and warnings.

### Strategy variants

This bot now runs on benchmark-only strategy by default:

- `macro_gate_benchmark`: macro gate + vol-targeting without micro-regime scaling.

```bash
python scripts/backtest.py \
  --product BTC-USD \
  --start 2021-01-01T00:00:00Z \
  --end 2026-01-31T00:00:00Z \
  --strategy macro_gate_benchmark \
  --fill-model bid_ask
```

`macro_only_v2` is a new strategy that uses only daily macro signals and stateful sizing:
- signal modes: `sma200_band`, `mom_6_12`, `sma200_and_mom`, `sma200_or_mom`, `score4_legacy`
- two-state macro gate with confirmation/hysteresis
- optional realized-vol inverse targeting
- drawdown breaker with cooldown/re-entry confirm

Example:

```bash
python scripts/backtest.py \
  --strategy macro_only_v2 \
  --product BTC-USD \
  --start 2021-01-01T00:00:00Z \
  --end 2026-01-31T00:00:00Z \
  --macro2-signal-mode sma200_and_mom \
  --macro2-confirm-days 2 \
  --macro2-min-on-days 2 \
  --macro2-min-off-days 1 \
  --macro2-vol-mode inverse_vol \
  --macro2-target-ann-vol-half 0.30 \
  --macro2-target-ann-vol-full 0.60 \
  --macro2-dd-threshold 0.25
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

General sweep (benchmark parameter space):

```bash
python scripts/frontier_sweep.py \
  --product BTC-USD \
  --strategy macro_gate_benchmark \
  --fill-model bid_ask \
  --start 2021-01-01T00:00:00Z \
  --end 2026-01-31T00:00:00Z
```

Macro-only frontier sweep (walk-forward + cost scenarios baseline/stress_1/stress_2):

```bash
python scripts/frontier_sweep_macro_only.py \
  --product BTC-USD \
  --fill-model bid_ask \
  --start 2021-01-01T00:00:00Z \
  --test-end 2026-01-31T00:00:00Z \
  --acceleration-backend auto \
  --workers 4 \
  --small
```

Include FRED dimensions in the sweep grid (optional):

```bash
python scripts/frontier_sweep_macro_only.py \
  --product BTC-USD \
  --fill-model bid_ask \
  --start 2021-01-01T00:00:00Z \
  --test-end 2026-01-31T00:00:00Z \
  --include-fred-grid
```

Outputs:
- `artifacts/frontier_macro_only_v2/summary.csv` — one row per (params, window, scenario)
- `artifacts/frontier_macro_only_v2/frontier.csv` — ranked top configs
- `artifacts/frontier_macro_only_v2/best_config.json` — recommended config + benchmark summary

Interpretation:
- ranking favors robust validation performance under stress-1 drawdown,
- then Sharpe,
- then lower drawdown / lower fees / lower turnover.

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
