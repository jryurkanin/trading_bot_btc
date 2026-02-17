#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone, timedelta
from itertools import product
from pathlib import Path
import sys

# Make local src discoverable when running directly from repository root
sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

from bot.config import BotConfig
from bot.coinbase_client import RESTClientWrapper
from bot.data.candles import CandleStore, CandleQuery
from bot.backtest.walkforward import walk_forward_test, choose_robust_parameter_set


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--product", default="BTC-USD")
    p.add_argument("--start", required=True)
    p.add_argument("--end", required=True)
    p.add_argument("--config", default=None)
    p.add_argument("--acceleration-backend", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument(
        "--strategy",
        default="macro_gate_benchmark",
        choices=[
            "macro_gate_benchmark",
            "macro_only_v2",
            "regime_switching_v3",
            "regime_switching_v4_core",
            "v5_adaptive",
        ],
    )
    p.add_argument("--out", default="reports")
    p.add_argument("--grid", action="append", help="KEY=VAL, can be repeated (e.g. bb_window=20)")
    p.add_argument("--metric", default="cagr")
    return p.parse_args()


def parse_ts(raw: str) -> datetime:
    ts = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    if ts.tzinfo is None:
        return ts.replace(tzinfo=timezone.utc)
    return ts.astimezone(timezone.utc)


def _prefetch_start(start: datetime, cfg: BotConfig) -> datetime:
    warmup_days = max(
        400,
        int(getattr(cfg.regime, "mom_12m_days", 365) or 365) + 30,
        int(getattr(cfg.regime, "vol_lookback_days", 365) or 365) + 30,
        int(getattr(cfg.fred, "daily_z_lookback", 252) or 252) + 30,
    )
    return start - timedelta(days=warmup_days)


def build_grid(items: list[str]):
    if not items:
        return []

    # simple parser for keys with comma-separated values: key=1,2,3
    key_vals = []
    for s in items:
        if "=" not in s:
            continue
        k, v = s.split("=", 1)
        vals = [x.strip() for x in v.split(",")]
        parsed = []
        for val in vals:
            vlow = val.lower()
            if vlow in {"true", "false", "1", "0", "yes", "no"}:
                parsed.append(vlow in {"true", "1", "yes"})
                continue
            try:
                if "." in vlow or "e" in vlow:
                    parsed.append(float(val))
                else:
                    parsed.append(int(val))
            except ValueError:
                try:
                    parsed.append(float(val))
                except ValueError:
                    parsed.append(val)
        key_vals.append((k.strip(), parsed))

    if not key_vals:
        return []

    out = []
    for combo in product(*[v for _, v in key_vals]):
        cfg = {}
        for (k, _), v in zip(key_vals, combo):
            cfg[k] = v
        out.append(cfg)
    return out


def main() -> int:
    args = parse_args()
    cfg = BotConfig.load(args.config)
    cfg.data.product = args.product
    cfg.backtest.strategy = args.strategy
    cfg.backtest.acceleration_backend = args.acceleration_backend
    # Macro benchmark optimization: keep settings explicit and deterministic.
    cfg.regime.trend_boost_enabled = False
    start = parse_ts(args.start)
    end = parse_ts(args.end)
    prefetch_start = _prefetch_start(start, cfg)

    client = RESTClientWrapper(cfg.coinbase, cfg.data)
    store = CandleStore(cfg.data)
    hourly = store.get_candles(client, CandleQuery(product=args.product, timeframe="1h", start=prefetch_start, end=end))
    daily = store.get_candles(client, CandleQuery(product=args.product, timeframe="1d", start=prefetch_start, end=end))

    grid = build_grid(args.grid or [])
    if not grid:
        grid = [
            {
                "adx_trend_threshold": 25,
                "adx_range_threshold": 20,
                "chop_trend_threshold": 38.2,
                "chop_range_threshold": 61.8,
            },
            {
                "adx_trend_threshold": 20,
                "adx_range_threshold": 18,
                "chop_trend_threshold": 40,
                "chop_range_threshold": 58,
            },
        ]

    results = walk_forward_test(hourly, daily, cfg, grid)
    best = choose_robust_parameter_set(results, metric_name=args.metric)

    def _serialize_result(r):
        return {
            "test_start": str(r.test_start),
            "test_end": str(r.test_end),
            "params": r.params,
            "metrics": r.metrics,
            "diagnostics": r.diagnostics,
        }

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    out_path = out / "optimization.json"
    out_path.write_text(json.dumps({"results": [_serialize_result(r) for r in results], "best": best}, indent=2), encoding="utf-8")
    print("Optimization done")
    print(json.dumps(best, indent=2))
    print(f"Saved: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
