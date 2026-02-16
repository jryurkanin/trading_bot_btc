#!/usr/bin/env python3
"""Frontier sweep for V5 adaptive strategy vs macro-gate benchmark.

Runs both ``v5_adaptive`` and ``macro_gate_benchmark`` for each parameter
set and compares them.  A config is only selected if V5 beats the benchmark
on validation.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import statistics
import sys

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

from bot.backtest.engine import BacktestEngine
from bot.backtest.reporting import dumps_strict_json, write_strict_json
from bot.backtest.macro_attribution import compute_macro_bucket_attribution
from bot.coinbase_client import RESTClientWrapper
from bot.config import BotConfig
from bot.data.candles import CandleQuery, CandleStore


DEFAULT_GRID_SPACE_V5: dict[str, list[Any]] = {
    # Asymmetric micro multipliers
    "v5_micro_trend_mult": [1.0, 1.1, 1.15, 1.25],
    "v5_micro_range_mult": [0.75, 0.85, 1.0],
    "v5_micro_neutral_mult": [0.9, 1.0],
    "v5_micro_high_vol_mult": [0.0],
    "v5_micro_max_mult": [1.5],
    # Adaptive macro gate
    "v5_adaptive_enter_base": [0.50, 0.75],
    "v5_adaptive_exit_base": [0.15, 0.25],
    "v5_adaptive_sensitivity": [0.0, 0.10, 0.15, 0.25],
    "v5_adaptive_confirm_days": [1, 2],
    "v5_adaptive_min_on_days": [1, 2],
    "v5_adaptive_min_off_days": [1],
    "v5_adaptive_half_multiplier": [0.40, 0.50, 0.60],
    "v5_adaptive_full_multiplier": [1.0],
    # Shared
    "target_ann_vol": [0.20, 0.30],
}

# Small grid for quick smoke tests
SMALL_GRID_SPACE_V5: dict[str, list[Any]] = {
    "v5_micro_trend_mult": [1.0, 1.15],
    "v5_micro_range_mult": [0.85, 1.0],
    "v5_micro_neutral_mult": [1.0],
    "v5_micro_high_vol_mult": [0.0],
    "v5_micro_max_mult": [1.5],
    "v5_adaptive_enter_base": [0.75],
    "v5_adaptive_exit_base": [0.25],
    "v5_adaptive_sensitivity": [0.0, 0.15],
    "v5_adaptive_confirm_days": [2],
    "v5_adaptive_min_on_days": [2],
    "v5_adaptive_min_off_days": [1],
    "v5_adaptive_half_multiplier": [0.50],
    "v5_adaptive_full_multiplier": [1.0],
    "target_ann_vol": [0.20],
}


@dataclass(frozen=True)
class CostScenario:
    name: str
    fee_bps_delta: float
    impact_bps_delta: float
    spread_bps_delta: float


SCENARIOS = [
    CostScenario("baseline", 0.0, 0.0, 0.0),
    CostScenario("stress_1", 5.0, 2.0, 2.0),
    CostScenario("stress_2", 10.0, 5.0, 5.0),
]


@dataclass(frozen=True)
class Window:
    name: str
    start: datetime
    end: datetime


def parse_ts(raw: str) -> datetime:
    ts = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    if ts.tzinfo is None:
        return ts.replace(tzinfo=timezone.utc)
    return ts.astimezone(timezone.utc)


def parse_grid_values(raw: str) -> list[Any]:
    vals: list[Any] = []
    for chunk in [x.strip() for x in raw.split(",") if x.strip()]:
        low = chunk.lower()
        if low in {"true", "false", "1", "0", "yes", "no"}:
            vals.append(low in {"true", "1", "yes"})
            continue
        try:
            if "." in low or "e" in low:
                vals.append(float(chunk))
            else:
                vals.append(int(chunk))
            continue
        except ValueError:
            pass
        vals.append(chunk)
    return vals


def parse_grid_flags(items: list[str]) -> dict[str, list[Any]]:
    out: dict[str, list[Any]] = {}
    for item in items:
        if "=" not in item:
            continue
        key, raw = item.split("=", 1)
        out[key.strip()] = parse_grid_values(raw)
    return out


def product_grid(space: dict[str, list[Any]]) -> list[dict[str, Any]]:
    keys = list(space.keys())
    vals = [space[k] for k in keys]
    combos = []
    for tup in itertools.product(*vals):
        combos.append({k: v for k, v in zip(keys, tup, strict=False)})
    return combos


def load_grid(path: str | None, grid_flags: dict[str, list[Any]], small: bool = False) -> list[dict[str, Any]]:
    if path:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if isinstance(payload, list):
            base = [dict(x) for x in payload if isinstance(x, dict)]
            if not grid_flags:
                return base
            ext = product_grid(grid_flags)
            out: list[dict[str, Any]] = []
            for b in base:
                for e in ext:
                    m = dict(b)
                    m.update(e)
                    out.append(m)
            return out
        if isinstance(payload, dict):
            space = {str(k): list(v) for k, v in payload.items() if isinstance(v, list)}
            space.update(grid_flags)
            return product_grid(space)
        raise ValueError("Unsupported --grid-config format (must be object or list)")

    space = dict(SMALL_GRID_SPACE_V5 if small else DEFAULT_GRID_SPACE_V5)
    space.update(grid_flags)
    return product_grid(space)


def clone_cfg(cfg: BotConfig) -> BotConfig:
    if hasattr(cfg, "model_dump") and hasattr(BotConfig, "model_validate"):
        return BotConfig.model_validate(cfg.model_dump())
    if hasattr(cfg, "dict"):
        return BotConfig.parse_obj(cfg.dict())
    return BotConfig.parse_obj(dict(cfg.__dict__))


def set_param(cfg: BotConfig, key: str, value: Any) -> None:
    if "." in key:
        obj: Any = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            obj = getattr(obj, p)
        setattr(obj, parts[-1], value)
        return

    for section in [cfg.regime, cfg.execution, cfg.backtest, cfg.risk]:
        if hasattr(section, key):
            setattr(section, key, value)
            return

    raise KeyError(f"Unknown parameter key: {key}")


def configure_strategy(cfg: BotConfig, strategy: str) -> None:
    cfg.backtest.strategy = strategy
    cfg.regime.trend_boost_enabled = False


def run_window(
    base_cfg: BotConfig,
    product: str,
    hourly: pd.DataFrame,
    daily: pd.DataFrame,
    window: Window,
    params: dict[str, Any],
    scenario: CostScenario,
    base_maker_rate: float,
    base_taker_rate: float,
    strategy: str,
) -> dict[str, Any]:
    cfg = clone_cfg(base_cfg)
    cfg.data.product = product
    configure_strategy(cfg, strategy)

    for k, v in params.items():
        set_param(cfg, k, v)

    # For benchmark, map v5 adaptive gate params → v4 equivalents so
    # the benchmark uses the same macro gate configuration.
    if strategy == "macro_gate_benchmark":
        v5_to_v4_map = {
            "v5_adaptive_enter_base": "v4_macro_enter_threshold",
            "v5_adaptive_exit_base": "v4_macro_exit_threshold",
            "v5_adaptive_half_base": "v4_macro_half_threshold",
            "v5_adaptive_full_base": "v4_macro_full_threshold",
            "v5_adaptive_confirm_days": "v4_macro_confirm_days",
            "v5_adaptive_min_on_days": "v4_macro_min_on_days",
            "v5_adaptive_min_off_days": "v4_macro_min_off_days",
            "v5_adaptive_half_multiplier": "v4_macro_half_multiplier",
            "v5_adaptive_full_multiplier": "v4_macro_full_multiplier",
        }
        for v5_key, v4_key in v5_to_v4_map.items():
            if v5_key in params:
                set_param(cfg, v4_key, params[v5_key])

    maker_rate = float(base_maker_rate + scenario.fee_bps_delta / 10_000.0)
    taker_rate = float(base_taker_rate + scenario.fee_bps_delta / 10_000.0)

    cfg.execution.impact_bps = float(cfg.execution.impact_bps + scenario.impact_bps_delta)
    cfg.execution.spread_bps = float(cfg.execution.spread_bps + scenario.spread_bps_delta)

    engine = BacktestEngine(
        product=product,
        hourly_candles=hourly,
        daily_candles=daily,
        start=window.start,
        end=window.end,
        config=cfg.backtest,
        fees=(maker_rate, taker_rate),
        slippage_bps=cfg.backtest.slippage_bps,
        use_spread_slippage=cfg.backtest.use_spread_slippage,
        regime_config=cfg.regime,
        risk_config=cfg.risk,
        execution_config=cfg.execution,
    )
    result = engine.run()
    eq = result.equity_curve["equity"]
    net_pnl = float(eq.iloc[-1] - eq.iloc[0]) if len(eq) else 0.0

    macro_bucket_report, _ = compute_macro_bucket_attribution(
        result.equity_curve,
        result.decisions,
        result.trades,
        initial_equity=cfg.backtest.initial_equity,
    )

    buckets = macro_bucket_report.get("buckets", {})
    off_time_share = float((buckets.get("OFF") or {}).get("time_share", 0.0) or 0.0)
    half_time_share = float((buckets.get("ON_HALF") or {}).get("time_share", 0.0) or 0.0)
    full_time_share = float((buckets.get("ON_FULL") or {}).get("time_share", 0.0) or 0.0)

    return {
        "strategy": strategy,
        "window": window.name,
        "scenario": scenario.name,
        "start": window.start.isoformat(),
        "end": window.end.isoformat(),
        "cagr": result.metrics.get("cagr"),
        "sharpe": result.metrics.get("sharpe"),
        "sortino": result.metrics.get("sortino"),
        "max_drawdown": result.metrics.get("max_drawdown"),
        "profit_factor": result.metrics.get("profit_factor"),
        "turnover": result.metrics.get("turnover"),
        "trade_count": result.diagnostics.get("trade_count"),
        "net_pnl": net_pnl,
        "maker_rate": maker_rate,
        "taker_rate": taker_rate,
        "impact_bps": cfg.execution.impact_bps,
        "spread_bps": cfg.execution.spread_bps,
        "macro_off_time_share": off_time_share,
        "macro_half_time_share": half_time_share,
        "macro_full_time_share": full_time_share,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run V5 adaptive frontier sweep with benchmark comparison."
    )
    p.add_argument("--product", default="BTC-USD")
    p.add_argument("--start", default="2021-01-01T00:00:00Z")
    p.add_argument(
        "--end",
        default=datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z"),
    )
    p.add_argument("--train-start", default="2021-01-01T00:00:00Z")
    p.add_argument("--train-end", default="2023-12-31T23:00:00Z")
    p.add_argument("--val-start", default="2024-01-01T00:00:00Z")
    p.add_argument("--val-end", default="2024-12-31T23:00:00Z")
    p.add_argument("--test-start", default="2025-01-01T00:00:00Z")
    p.add_argument("--test-end", default=None)
    p.add_argument(
        "--fill-model",
        default="bid_ask",
        choices=["next_open", "bid_ask", "worst_case_bar"],
    )
    p.add_argument("--config", default=None)
    p.add_argument("--grid-config", default=None)
    p.add_argument("--grid", action="append", default=[])
    p.add_argument("--small", action="store_true", help="Use small grid for quick tests")
    p.add_argument("--turnover-max", type=float, default=700.0)
    p.add_argument("--max-drawdown-max", type=float, default=0.30)
    p.add_argument("--min-full-time-share", type=float, default=0.05)
    p.add_argument("--top-n", type=int, default=10)
    p.add_argument("--output-dir", default="artifacts/frontier_v5")
    p.add_argument("--maker-bps", type=float, default=10.0)
    p.add_argument("--taker-bps", type=float, default=25.0)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cfg = BotConfig.load(args.config)
    cfg.data.product = args.product
    cfg.execution.fill_model = args.fill_model

    start = parse_ts(args.start)
    end = parse_ts(args.end)
    test_end = parse_ts(args.test_end) if args.test_end else end

    windows = [
        Window("train", parse_ts(args.train_start), parse_ts(args.train_end)),
        Window("val", parse_ts(args.val_start), parse_ts(args.val_end)),
        Window("test", parse_ts(args.test_start), test_end),
    ]

    client = RESTClientWrapper(cfg.coinbase, cfg.data)
    store = CandleStore(cfg.data)
    hourly = store.get_candles(
        client=client,
        query=CandleQuery(product=args.product, timeframe="1h", start=start, end=end),
    )
    daily = store.get_candles(
        client=client,
        query=CandleQuery(product=args.product, timeframe="1d", start=start, end=end),
    )

    base_maker_rate = args.maker_bps / 10_000.0
    base_taker_rate = args.taker_bps / 10_000.0
    try:
        tx = client.get_transaction_summary(args.product)
        m = float(tx.maker_fee_rate)
        t = float(tx.taker_fee_rate)
        if m > 0:
            base_maker_rate = m
        if t > 0:
            base_taker_rate = t
    except Exception:
        pass

    grid_flags = parse_grid_flags(args.grid)
    param_sets = load_grid(args.grid_config, grid_flags, small=args.small)
    total_runs = len(param_sets) * 2 * len(windows) * len(SCENARIOS)
    print(f"V5 frontier sweep: {len(param_sets)} param sets × 2 strategies × "
          f"{len(windows)} windows × {len(SCENARIOS)} scenarios = {total_runs} runs")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict[str, Any]] = []
    grouped: dict[str, dict[str, dict[str, dict[str, dict[str, Any]]]]] = {}

    strategies = ["v5_adaptive", "macro_gate_benchmark"]

    for i, params in enumerate(param_sets):
        param_id = f"p{i:04d}"
        grouped[param_id] = {}
        print(f"  [{i + 1}/{len(param_sets)}] {param_id}", flush=True)
        for strategy in strategies:
            grouped[param_id][strategy] = {}
            for window in windows:
                grouped[param_id][strategy][window.name] = {}
                for scenario in SCENARIOS:
                    try:
                        row = run_window(
                            base_cfg=cfg,
                            product=args.product,
                            hourly=hourly,
                            daily=daily,
                            window=window,
                            params=params,
                            scenario=scenario,
                            base_maker_rate=base_maker_rate,
                            base_taker_rate=base_taker_rate,
                            strategy=strategy,
                        )
                        row["param_id"] = param_id
                        row["params"] = json.dumps(params, sort_keys=True)
                        summary_rows.append(row)
                        grouped[param_id][strategy][window.name][scenario.name] = row
                    except Exception as exc:
                        summary_rows.append(
                            {
                                "param_id": param_id,
                                "strategy": strategy,
                                "params": json.dumps(params, sort_keys=True),
                                "window": window.name,
                                "scenario": scenario.name,
                                "start": window.start.isoformat(),
                                "end": window.end.isoformat(),
                                "error": str(exc),
                            }
                        )

    summary_path = out_dir / "summary.csv"
    if summary_rows:
        cols = sorted({k for row in summary_rows for k in row.keys()})
        with summary_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for row in summary_rows:
                w.writerow(row)

    # --- Ranking: V5 must beat benchmark ---
    ranked: list[dict[str, Any]] = []
    for i, params in enumerate(param_sets):
        param_id = f"p{i:04d}"

        v5_val = grouped.get(param_id, {}).get("v5_adaptive", {}).get("val", {})
        bench_val = grouped.get(param_id, {}).get("macro_gate_benchmark", {}).get("val", {})
        v5_test = grouped.get(param_id, {}).get("v5_adaptive", {}).get("test", {})

        v5_base = v5_val.get("baseline")
        v5_s1 = v5_val.get("stress_1")
        v5_s2 = v5_val.get("stress_2")
        bench_base = bench_val.get("baseline")

        if not v5_base or not v5_s1 or not v5_s2 or not bench_base:
            continue

        # V5 must beat benchmark on validation baseline net PnL
        v5_pnl = float(v5_base.get("net_pnl", 0.0) or 0.0)
        bench_pnl = float(bench_base.get("net_pnl", 0.0) or 0.0)
        if v5_pnl <= bench_pnl:
            continue

        # stress_1 must be profitable
        if float(v5_s1.get("net_pnl", 0.0) or 0.0) <= 0.0:
            continue

        # max drawdown check
        max_dd_ok = (
            abs(float(v5_base.get("max_drawdown", 0.0) or 0.0)) <= args.max_drawdown_max
            and abs(float(v5_s1.get("max_drawdown", 0.0) or 0.0)) <= args.max_drawdown_max
            and abs(float(v5_s2.get("max_drawdown", 0.0) or 0.0)) <= args.max_drawdown_max
        )
        if not max_dd_ok:
            continue

        worst_turnover = max(
            float(v5_base.get("turnover", 0.0) or 0.0),
            float(v5_s1.get("turnover", 0.0) or 0.0),
            float(v5_s2.get("turnover", 0.0) or 0.0),
        )
        if worst_turnover > args.turnover_max:
            continue

        full_share_min = min(
            float(v5_base.get("macro_full_time_share", 0.0) or 0.0),
            float(v5_s1.get("macro_full_time_share", 0.0) or 0.0),
            float(v5_s2.get("macro_full_time_share", 0.0) or 0.0),
        )
        if full_share_min < args.min_full_time_share:
            continue

        cagr_vals = [
            float(v5_base.get("cagr", 0.0) or 0.0),
            float(v5_s1.get("cagr", 0.0) or 0.0),
            float(v5_s2.get("cagr", 0.0) or 0.0),
        ]
        sharpe_vals = [
            float(v5_base.get("sharpe", 0.0) or 0.0),
            float(v5_s1.get("sharpe", 0.0) or 0.0),
            float(v5_s2.get("sharpe", 0.0) or 0.0),
        ]
        val_score = float(statistics.median(cagr_vals))
        val_sharpe_med = float(statistics.median(sharpe_vals))

        ranked.append(
            {
                "param_id": param_id,
                "params": params,
                "val_score": val_score,
                "val_sharpe_med": val_sharpe_med,
                "val_v5_pnl": v5_pnl,
                "val_bench_pnl": bench_pnl,
                "val_v5_alpha": v5_pnl - bench_pnl,
                "val_cagr_baseline": float(v5_base.get("cagr", 0.0) or 0.0),
                "val_cagr_stress_1": float(v5_s1.get("cagr", 0.0) or 0.0),
                "val_cagr_stress_2": float(v5_s2.get("cagr", 0.0) or 0.0),
                "val_turnover_worst": worst_turnover,
                "val_max_drawdown_worst": min(
                    float(v5_base.get("max_drawdown", 0.0) or 0.0),
                    float(v5_s1.get("max_drawdown", 0.0) or 0.0),
                    float(v5_s2.get("max_drawdown", 0.0) or 0.0),
                ),
                "val_full_time_share_min": full_share_min,
                "test_cagr_stress_1": float(
                    (v5_test.get("stress_1") or {}).get("cagr", 0.0) or 0.0
                ),
                "test_sharpe_stress_1": float(
                    (v5_test.get("stress_1") or {}).get("sharpe", 0.0) or 0.0
                ),
                "test_max_drawdown_stress_1": float(
                    (v5_test.get("stress_1") or {}).get("max_drawdown", 0.0) or 0.0
                ),
            }
        )

    ranked.sort(
        key=lambda r: (
            r["val_score"],
            r["val_sharpe_med"],
            r["val_v5_alpha"],
            r["val_max_drawdown_worst"],
            -r["val_turnover_worst"],
        ),
        reverse=True,
    )

    top = ranked[: max(1, int(args.top_n))]
    frontier_rows: list[dict[str, Any]] = []
    for r in top:
        flat = dict(r)
        flat["params"] = json.dumps(r["params"], sort_keys=True)
        frontier_rows.append(flat)

    frontier_path = out_dir / "frontier.csv"
    if frontier_rows:
        cols = sorted({k for row in frontier_rows for k in row.keys()})
        with frontier_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for row in frontier_rows:
                w.writerow(row)

    best = top[0] if top else None
    if best is not None:
        best_cfg_patch = {
            "regime": {
                "trend_boost_enabled": False,
                **best["params"],
            },
            "execution": {"fill_model": args.fill_model},
            "backtest": {"strategy": "v5_adaptive"},
        }
        best_cfg_path = write_strict_json(
            out_dir / "best_config.json", best_cfg_patch
        )

        repro_cmd = (
            f"python3 scripts/backtest.py --product {args.product} "
            f"--start {args.test_start} --end {args.test_end or args.end} "
            f"--strategy v5_adaptive --fill-model {args.fill_model} "
            f"--config {best_cfg_path} --output {out_dir / 'best_test_repro'}"
        )

        best_payload = {
            "best": best,
            "constraints": {
                "turnover_max": args.turnover_max,
                "max_drawdown_max": args.max_drawdown_max,
                "min_full_time_share": args.min_full_time_share,
                "v5_must_beat_benchmark": True,
            },
            "reproduce_test_command": repro_cmd,
            "paths": {
                "summary_csv": str(summary_path),
                "frontier_csv": str(frontier_path),
                "best_config": str(best_cfg_path),
            },
        }
        write_strict_json(
            out_dir / "best_config.json",
            {**best_cfg_patch, "frontier": best_payload},
        )

        print("V5 adaptive frontier sweep completed — winning configs found!")
        print(dumps_strict_json(best_payload, indent=2))
        print("Reproduce best test run:")
        print(repro_cmd)
    else:
        print(
            "V5 adaptive frontier sweep completed but no config satisfied constraints "
            "(including v5 > benchmark)."
        )
        print(f"Summary: {summary_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
