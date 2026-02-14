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
  - drawdown breaker
  - stale data breaker
  - vol-target sizing and caps
- **Execution**:
  - limit-first placement + fallback-to-market
  - idempotent client_order_id
  - paper mode simulator
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
- `reports/report.json`

### 2) Walk-forward sweep

```bash
python scripts/optimize.py \
  --product BTC-USD \
  --start 2020-01-01T00:00:00Z \
  --end 2026-01-31T00:00:00Z \
  --grid adx_trend_threshold=20,25,30 \
  --grid range_tranche_size=0.2,0.25
```

### 3) Paper trade (2+ cycles)

```bash
python scripts/trade.py --paper --product BTC-USD --cycles 2
```

### 4) Live (sandbox)

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
