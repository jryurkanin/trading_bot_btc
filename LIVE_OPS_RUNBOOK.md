# Live Ops Runbook (Phase 3)

This runbook is for deploying and operating `trading_bot_btc` safely in paper/live modes.

## 1) Preflight (before any live capital)

1. **Run tests**
   ```bash
   pytest -q
   ```

2. **Run a smoke backtest**
   ```bash
   python3 scripts/backtest.py \
     --product BTC-USD \
     --start 2025-06-01T00:00:00Z \
     --end 2025-06-03T00:00:00Z \
     --strategy macro_gate_benchmark \
     --output /tmp/tb_smoke
   ```

3. **Check critical execution config**
   - `execution.maker_first`
   - `execution.order_timeout_s`
   - `execution.cancel_replace_on_timeout`
   - `execution.replace_with_market_on_timeout`
   - `risk.manual_kill_switch`
   - `risk.safe_mode`

4. **Confirm credentials/environment**
   - `COINBASE_API_KEY`
   - `COINBASE_API_SECRET`
   - Optional: `COINBASE_API_PASSPHRASE`
   - Optional FRED overlay: `FRED_API_KEY` (required only if `fred.enabled=true`)

## 2) Paper verification (required)

Run at least a few cycles in paper mode before any live run:

```bash
python3 scripts/trade.py --paper --product BTC-USD --cycles 3
```

Validate:
- No exceptions in console logs
- Health file updates (`.trading_bot_cache/health_status.json`)
- No repeated order-failure alerts

## 3) Sandbox verification (recommended)

If using Coinbase sandbox:

```bash
python3 scripts/trade.py --live --product BTC-USD --sandbox --cycles 3
```

Validate order lifecycle behavior:
- Open orders persisted in state store
- Timeout cancel/replace behavior observed when expected

## 4) Go-live checklist

1. Start with low risk settings and short monitoring window.
2. Launch live runner:
   ```bash
   python3 scripts/trade.py --live --product BTC-USD
   ```
3. Confirm first hourly cycle completes and health is `ok`.

## 5) Runtime monitoring

### Health / alerts
- Health: `.trading_bot_cache/health_status.json`
- Alerts: `.trading_bot_cache/alerts.log`

Watch for:
- `consecutive_order_failures`
- `consecutive_cycle_failures`
- `stale_hourly_feed`
- FRED warnings (if enabled)

### State store quick checks

```bash
sqlite3 .trading_bot_cache/live_state.sqlite "select count(*) from open_orders;"
sqlite3 .trading_bot_cache/live_state.sqlite "select ts, payload from decisions order by ts desc limit 3;"
```

## 6) Emergency actions

### Immediate risk-off (preferred)
Set in config and restart process:
- `risk.manual_kill_switch = true`
- Keep `risk.safe_mode = true`

This forces target exposure to 0 through risk caps.

### Hard stop
Stop the live process supervisor/service immediately if behavior is abnormal.

## 7) Notes on hardened behavior (Phase 1 + 2)

- Reconcile polling now occurs during inter-hour waits (timeout logic is no longer effectively hourly-only).
- Live routing uses real-time BBO for order pricing inputs.
- Cancel failures do not blindly drop local order state.
- `maker_first` is now honored in live runner.
- Live daily features now support FRED overlay parity with backtests.

