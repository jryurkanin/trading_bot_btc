#!/usr/bin/env python3
"""Frontier sweep for ``macro_only_v2`` with walk-forward + cost stress."""
from __future__ import annotations

import argparse
import csv
import itertools
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import statistics
import sys

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

from bot.backtest.engine import BacktestEngine
from bot.backtest.reporting import write_strict_json, dumps_strict_json
from bot.backtest.macro_attribution import compute_macro_bucket_attribution
from bot.config import BotConfig
from bot.coinbase_client import RESTClientWrapper
from bot.data.candles import CandleQuery, CandleStore


DEFAULT_GRID_MACRO_ONLY: dict[str, list[Any]] = {
    "macro2_signal_mode": ["sma200_and_mom", "sma200_or_mom", "mom_6_12"],
    "macro2_confirm_days": [1, 2],
    "macro2_min_on_days": [1, 2],
    "macro2_min_off_days": [1],
    "macro2_weight_half": [0.4, 0.5],
    "macro2_weight_full": [1.0],
    "macro2_vol_mode": ["inverse_vol", "none"],
    "macro2_vol_lookback_days": [60],
    "macro2_vol_floor": [0.05],
    "macro2_target_ann_vol_half": [0.25, 0.30],
    "macro2_target_ann_vol_full": [0.50, 0.60],
    "macro2_dd_threshold": [0.20, 0.25],
    "macro2_dd_cooldown_days": [8, 10],
    "macro2_dd_reentry_confirm_days": [2],
    "macro2_dd_safe_weight": [0.0],
}


SMALL_GRID_MACRO_ONLY: dict[str, list[Any]] = {
    "macro2_signal_mode": ["sma200_and_mom", "mom_6_12"],
    "macro2_confirm_days": [2],
    "macro2_min_on_days": [1, 2],
    "macro2_min_off_days": [1],
    "macro2_weight_half": [0.5],
    "macro2_weight_full": [1.0],
    "macro2_vol_mode": ["inverse_vol", "none"],
    "macro2_vol_lookback_days": [60],
    "macro2_vol_floor": [0.05],
    "macro2_target_ann_vol_half": [0.30],
    "macro2_target_ann_vol_full": [0.60],
    "macro2_dd_threshold": [0.25],
    "macro2_dd_cooldown_days": [10],
    "macro2_dd_reentry_confirm_days": [2],
    "macro2_dd_safe_weight": [0.0],
}


DEFAULT_FRED_SWEEP_SPACE: dict[str, list[Any]] = {
    "fred.enabled": [False, True],
    "fred.max_risk_off_penalty": [0.0, 0.25, 0.5, 0.75],
    "fred.risk_off_score_ema_span": [8, 16, 32],
    "fred.lag_stress_multiplier": [1.0, 1.5, 2.0],
    "fred_risk_weight_scale": [1.0],
}


@dataclass(frozen=True)
class CostScenario:
    name: str
    fee_bps_delta: float
    impact_bps_delta: float


SCENARIOS = [
    CostScenario("baseline", 0.0, 0.0),
    CostScenario("stress_1", 5.0, 2.0),
    CostScenario("stress_2", 10.0, 5.0),
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


_WORKER_CTX: dict[str, Any] = {}


def _worker_init(ctx: dict[str, Any]) -> None:
    global _WORKER_CTX
    _WORKER_CTX = ctx


def _cfg_from_payload(payload: dict[str, Any]) -> BotConfig:
    if hasattr(BotConfig, "model_validate"):
        return BotConfig.model_validate(payload)
    return BotConfig.parse_obj(payload)


def _run_param_worker(task: tuple[int, dict[str, Any]]) -> tuple[str, list[dict[str, Any]], dict[str, dict[str, dict[str, Any]]]]:
    idx, params = task
    param_id = f"p{idx:04d}"

    ctx = _WORKER_CTX
    base_cfg = _cfg_from_payload(ctx["base_cfg_payload"])
    product = str(ctx["product"])
    hourly = ctx["hourly"]
    daily = ctx["daily"]
    base_maker = float(ctx["base_maker"])
    base_taker = float(ctx["base_taker"])

    grouped_param: dict[str, dict[str, dict[str, Any]]] = {}
    rows: list[dict[str, Any]] = []

    for w_payload in ctx["windows"]:
        window = Window(
            str(w_payload["name"]),
            parse_ts(str(w_payload["start"])),
            parse_ts(str(w_payload["end"])),
        )
        grouped_param[window.name] = {}
        for scenario in SCENARIOS:
            try:
                row = run_window(
                    base_cfg=base_cfg,
                    product=product,
                    hourly=hourly,
                    daily=daily,
                    window=window,
                    params=params,
                    scenario=scenario,
                    base_maker=base_maker,
                    base_taker=base_taker,
                    strategy="macro_only_v2",
                )
                row["param_id"] = param_id
                row["params"] = json.dumps(params, sort_keys=True)
                grouped_param[window.name][scenario.name] = row
                rows.append(row)
            except Exception as exc:
                rows.append(
                    {
                        "param_id": param_id,
                        "params": json.dumps(params, sort_keys=True),
                        "strategy": "macro_only_v2",
                        "window": window.name,
                        "scenario": scenario.name,
                        "start": window.start.isoformat(),
                        "end": window.end.isoformat(),
                        "error": str(exc),
                    }
                )

    return param_id, rows, grouped_param


def parse_grid_values(raw: str) -> list[Any]:
    out: list[Any] = []
    for chunk in [x.strip() for x in raw.split(",") if x.strip()]:
        low = chunk.lower()
        if low in {"true", "false", "1", "0", "yes", "no"}:
            out.append(low in {"true", "1", "yes"})
            continue
        try:
            if "." in low or "e" in low:
                out.append(float(chunk))
            else:
                out.append(int(chunk))
            continue
        except ValueError:
            pass
        out.append(chunk)
    return out


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


def load_grid(
    path: str | None,
    grid_flags: dict[str, list[Any]],
    *,
    small: bool = False,
    include_fred_grid: bool = False,
) -> list[dict[str, Any]]:
    if path:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if isinstance(payload, list):
            base = [dict(x) for x in payload if isinstance(x, dict)]
            if not grid_flags:
                return base
            out: list[dict[str, Any]] = []
            for b in base:
                for e in product_grid(grid_flags):
                    m = dict(b)
                    m.update(e)
                    out.append(m)
            return out

        if isinstance(payload, dict):
            space = {str(k): list(v) for k, v in payload.items() if isinstance(v, list)}
            space.update(grid_flags)
            return product_grid(space)
        raise ValueError("Unsupported --grid-config format (must be object or list)")

    space = dict(SMALL_GRID_MACRO_ONLY if small else DEFAULT_GRID_MACRO_ONLY)
    if include_fred_grid:
        space.update(DEFAULT_FRED_SWEEP_SPACE)
    space.update(grid_flags)
    return product_grid(space)


def clone_cfg(cfg: BotConfig) -> BotConfig:
    if hasattr(cfg, "model_dump") and hasattr(BotConfig, "model_validate"):
        return BotConfig.model_validate(cfg.model_dump())
    if hasattr(cfg, "dict"):
        return BotConfig.parse_obj(cfg.dict())
    return BotConfig.parse_obj(dict(cfg.__dict__))


def set_param(cfg: BotConfig, key: str, value: Any) -> None:
    if key == "fred_risk_weight_scale":
        scale = float(value)
        cfg.fred.risk_off_weights = {
            str(k): float(v) * scale
            for k, v in cfg.fred.risk_off_weights.items()
        }
        return

    if "." in key:
        obj: Any = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            obj = getattr(obj, p)
        setattr(obj, parts[-1], value)
        return

    for section in [cfg.regime, cfg.execution, cfg.backtest, cfg.risk, cfg.fred]:
        if hasattr(section, key):
            setattr(section, key, value)
            return

    raise KeyError(f"Unknown parameter key: {key}")


def configure_strategy(cfg: BotConfig, strategy: str) -> None:
    cfg.backtest.strategy = strategy
    cfg.regime.trend_boost_enabled = False


def _bucket_fees(table: dict[str, Any] | None) -> tuple[float, float, float]:
    if not table:
        return 0.0, 0.0, 0.0
    off = float((table.get("OFF") or {}).get("fees", 0.0) or 0.0)
    half = float((table.get("ON_HALF") or {}).get("fees", 0.0) or 0.0)
    full = float((table.get("ON_FULL") or {}).get("fees", 0.0) or 0.0)
    return off + half + full, half, full


def _bucket_share(table: dict[str, Any] | None, key: str) -> float:
    if not table:
        return 0.0
    return float(((table.get(key) or {}).get("time_share", 0.0) or 0.0))


def run_window(
    base_cfg: BotConfig,
    product: str,
    hourly: pd.DataFrame,
    daily: pd.DataFrame,
    window: Window,
    params: dict[str, Any],
    scenario: CostScenario,
    base_maker: float,
    base_taker: float,
    strategy: str,
) -> dict[str, Any]:
    cfg = clone_cfg(base_cfg)
    cfg.data.product = product
    configure_strategy(cfg, strategy)

    for k, v in params.items():
        set_param(cfg, k, v)

    if strategy == "macro_only_v2" and getattr(cfg.regime, "macro2_signal_mode", None) is None:
        cfg.regime.macro2_signal_mode = "sma200_and_mom"

    maker_rate = float(base_maker + scenario.fee_bps_delta / 10_000.0)
    taker_rate = float(base_taker + scenario.fee_bps_delta / 10_000.0)

    cfg.execution.impact_bps = float(cfg.execution.impact_bps + scenario.impact_bps_delta)

    engine = BacktestEngine(
        product=product,
        hourly_candles=hourly,
        daily_candles=daily,
        start=window.start,
        end=window.end,
        config=cfg.backtest,
        fees=(maker_rate, taker_rate),
        slippage_bps=cfg.backtest.slippage_bps,
        use_spread_slippage=cfg.backtest.use_spread_slashing if hasattr(cfg.backtest, "use_spread_slashing") else cfg.backtest.use_spread_slippage,
        regime_config=cfg.regime,
        risk_config=cfg.risk,
        execution_config=cfg.execution,
        fred_config=cfg.fred,
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

    bucket_total_fees, half_fees, full_fees = _bucket_fees(macro_bucket_report.get("buckets", {}))
    row: dict[str, Any] = {
        "strategy": strategy,
        "window": window.name,
        "scenario": scenario.name,
        "start": window.start.isoformat(),
        "end": window.end.isoformat(),
        "cagr": result.metrics.get("cagr"),
        "sharpe": result.metrics.get("sharpe"),
        "max_drawdown": result.metrics.get("max_drawdown"),
        "profit_factor": result.metrics.get("profit_factor"),
        "turnover": result.metrics.get("turnover"),
        "trade_count": result.diagnostics.get("trade_count"),
        "net_pnl": net_pnl,
        "macro_off_time_share": _bucket_share(macro_bucket_report.get("buckets", {}), "OFF"),
        "macro_half_time_share": _bucket_share(macro_bucket_report.get("buckets", {}), "ON_HALF"),
        "macro_full_time_share": _bucket_share(macro_bucket_report.get("buckets", {}), "ON_FULL"),
        "macro_bucket_fees_total": bucket_total_fees,
        "macro_bucket_fees_half": half_fees,
        "macro_bucket_fees_full": full_fees,
        "maker_rate": maker_rate,
        "taker_rate": taker_rate,
        "impact_bps": cfg.execution.impact_bps,
        "spread_bps": cfg.execution.spread_bps,
        "acceleration_requested": getattr(cfg.backtest, "acceleration_backend", "auto"),
        "acceleration_backend": result.diagnostics.get("acceleration_backend", "cpu"),
        "acceleration_cuda_available": result.diagnostics.get("acceleration_cuda_available", 0),
        "acceleration_device": result.diagnostics.get("acceleration_device"),
        "fred_enabled": bool(cfg.fred.enabled),
        "fred_max_risk_off_penalty": float(cfg.fred.max_risk_off_penalty),
        "fred_risk_off_score_ema_span": int(cfg.fred.risk_off_score_ema_span),
        "fred_lag_stress_multiplier": float(cfg.fred.lag_stress_multiplier),
        "fred_cache_hit_rate": float((result.diagnostics.get("fred") or {}).get("cache_hit_rate", 0.0) or 0.0),
        "fred_series_used_count": int(len((result.diagnostics.get("fred") or {}).get("series_used", []))),
    }
    return row


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sweep macro_only_v2 with walk-forward and stress costs")
    p.add_argument("--product", default="BTC-USD")
    p.add_argument("--start", default="2021-01-01T00:00:00Z")
    p.add_argument("--end", default=None)
    p.add_argument("--train-start", default="2021-01-01T00:00:00Z")
    p.add_argument("--train-end", default="2023-12-31T23:00:00Z")
    p.add_argument("--val-start", default="2024-01-01T00:00:00Z")
    p.add_argument("--val-end", default="2024-12-31T23:00:00Z")
    p.add_argument("--test-start", default="2025-01-01T00:00:00Z")
    p.add_argument("--test-end", default=None)
    p.add_argument("--fill-model", default="bid_ask", choices=["next_open", "bid_ask", "worst_case_bar"])
    p.add_argument("--acceleration-backend", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--config", default=None)
    p.add_argument("--grid-config", default=None)
    p.add_argument("--grid", action="append", default=[])
    p.add_argument("--small", action="store_true", help="smaller walk-forward grid for quick checks")
    p.add_argument("--include-fred-grid", action="store_true", help="include FRED overlay parameters in sweep grid")
    p.add_argument("--workers", type=int, default=1, help="parallel worker processes for parameter sets")
    p.add_argument("--turnover-max", type=float, default=700.0)
    p.add_argument("--max-drawdown-max", type=float, default=0.30)
    p.add_argument("--top-n", type=int, default=5)
    p.add_argument("--output-dir", default="artifacts/frontier_macro_only_v2")
    p.add_argument("--maker-bps", type=float, default=10.0)
    p.add_argument("--taker-bps", type=float, default=25.0)
    return p.parse_args()


def main() -> int:
    args = parse_args()

    cfg = BotConfig.load(args.config)
    cfg.data.product = args.product
    cfg.execution.fill_model = args.fill_model
    cfg.backtest.acceleration_backend = args.acceleration_backend

    start = parse_ts(args.start)
    now = datetime.now(timezone.utc)
    end = parse_ts(args.end) if args.end else now
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
    param_sets = load_grid(
        args.grid_config,
        grid_flags,
        small=args.small,
        include_fred_grid=bool(args.include_fred_grid),
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict[str, Any]] = []

    # Baseline benchmark rows for comparison (same windows/scenarios).
    baseline_rows: list[dict[str, Any]] = []
    print("Running baseline macro_gate_benchmark comparator")
    for window in windows:
        for scenario in SCENARIOS:
            try:
                row = run_window(
                    base_cfg=cfg,
                    product=args.product,
                    hourly=hourly,
                    daily=daily,
                    window=window,
                    params={},
                    scenario=scenario,
                    base_maker=base_maker_rate,
                    base_taker=base_taker_rate,
                    strategy="macro_gate_benchmark",
                )
                row["param_id"] = "baseline"
                baseline_rows.append(row)
            except Exception as exc:
                baseline_rows.append(
                    {
                        "param_id": "baseline",
                        "strategy": "macro_gate_benchmark",
                        "window": window.name,
                        "scenario": scenario.name,
                        "start": window.start.isoformat(),
                        "end": window.end.isoformat(),
                        "error": str(exc),
                    }
                )

    grouped: dict[str, dict[str, dict[str, dict[str, Any]]]] = {}

    workers = max(1, int(args.workers or 1))
    if workers <= 1 or len(param_sets) <= 1:
        for i, params in enumerate(param_sets):
            param_id = f"p{i:04d}"
            grouped[param_id] = {}
            print(f"[{i + 1}/{len(param_sets)}] {param_id}", flush=True)

            for window in windows:
                grouped[param_id][window.name] = {}
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
                            base_maker=base_maker_rate,
                            base_taker=base_taker_rate,
                            strategy="macro_only_v2",
                        )
                        row["param_id"] = param_id
                        row["params"] = json.dumps(params, sort_keys=True)
                        grouped[param_id][window.name][scenario.name] = row
                        summary_rows.append(row)
                    except Exception as exc:
                        summary_rows.append(
                            {
                                "param_id": param_id,
                                "params": json.dumps(params, sort_keys=True),
                                "strategy": "macro_only_v2",
                                "window": window.name,
                                "scenario": scenario.name,
                                "start": window.start.isoformat(),
                                "end": window.end.isoformat(),
                                "error": str(exc),
                            }
                        )
    else:
        max_workers = min(workers, len(param_sets))
        print(f"Running macro_only_v2 parameter grid with {max_workers} workers", flush=True)

        if hasattr(cfg, "model_dump"):
            base_cfg_payload = cfg.model_dump()
        elif hasattr(cfg, "dict"):
            base_cfg_payload = cfg.dict()
        else:
            base_cfg_payload = dict(cfg.__dict__)

        worker_ctx = {
            "base_cfg_payload": base_cfg_payload,
            "product": args.product,
            "hourly": hourly,
            "daily": daily,
            "base_maker": float(base_maker_rate),
            "base_taker": float(base_taker_rate),
            "windows": [
                {
                    "name": w.name,
                    "start": w.start.isoformat(),
                    "end": w.end.isoformat(),
                }
                for w in windows
            ],
        }

        done = 0
        with ProcessPoolExecutor(
            max_workers=max_workers,
            initializer=_worker_init,
            initargs=(worker_ctx,),
        ) as executor:
            future_map = {
                executor.submit(_run_param_worker, (i, params)): (i, params)
                for i, params in enumerate(param_sets)
            }

            for future in as_completed(future_map):
                i, params = future_map[future]
                param_id = f"p{i:04d}"
                try:
                    pid, rows, grouped_param = future.result()
                    grouped[pid] = grouped_param
                    summary_rows.extend(rows)
                    done += 1
                    print(f"[{done}/{len(param_sets)}] {pid}", flush=True)
                except Exception as exc:
                    done += 1
                    grouped[param_id] = {}
                    print(f"[{done}/{len(param_sets)}] {param_id} failed", flush=True)
                    summary_rows.append(
                        {
                            "param_id": param_id,
                            "params": json.dumps(params, sort_keys=True),
                            "strategy": "macro_only_v2",
                            "error": str(exc),
                        }
                    )

    summary_rows.extend(baseline_rows)

    summary_path = out_dir / "summary.csv"
    if summary_rows:
        cols = sorted({k for row in summary_rows for k in row.keys()})
        with summary_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for row in summary_rows:
                w.writerow(row)

    ranked: list[dict[str, Any]] = []
    for i, params in enumerate(param_sets):
        param_id = f"p{i:04d}"

        val_baseline = grouped.get(param_id, {}).get("val", {})
        val = val_baseline.get("stress_1") if isinstance(val_baseline, dict) else None
        if val is None:
            continue

        val2 = val_baseline.get("stress_2")

        if not val or not val.get("cagr") or not val2 or not val2.get("cagr"):
            continue

        if float(val.get("cagr", 0.0) or 0.0) <= 0.0 or float(val2.get("cagr", 0.0) or 0.0) <= 0.0:
            continue

        if abs(float(val.get("max_drawdown", 0.0) or 0.0)) > args.max_drawdown_max:
            continue

        total_turnover = max(
            float(val_baseline.get("baseline", {}).get("turnover", 0.0) or 0.0),
            float(val.get("turnover", 0.0) or 0.0),
            float(val2.get("turnover", 0.0) or 0.0),
        )
        if total_turnover > args.turnover_max:
            continue

        ranked.append(
            {
                "param_id": param_id,
                "params": params,
                "val_stress1_cagr": float(val.get("cagr", 0.0) or 0.0),
                "val_stress1_sharpe": float(val.get("sharpe", 0.0) or 0.0),
                "val_stress1_max_drawdown": float(val.get("max_drawdown", 0.0) or 0.0),
                "val_stress1_fees": float(val.get("macro_bucket_fees_total", 0.0) or 0.0),
                "val_stress1_turnover": float(val.get("turnover", 0.0) or 0.0),
            }
        )

    ranked.sort(
        key=lambda r: (
            r["val_stress1_cagr"],
            r["val_stress1_sharpe"],
            -abs(r["val_stress1_max_drawdown"]),
            -r["val_stress1_fees"],
            -r["val_stress1_turnover"],
        ),
        reverse=True,
    )

    frontier_path = out_dir / "frontier.csv"
    frontier_rows: list[dict[str, Any]] = []
    for r in ranked[: max(1, int(args.top_n))]:
        item = dict(r)
        item["params"] = json.dumps(r["params"], sort_keys=True)
        frontier_rows.append(item)

    if frontier_rows:
        cols = sorted({k for row in frontier_rows for k in row.keys()})
        with frontier_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for row in frontier_rows:
                w.writerow(row)

    best = frontier_rows[0] if frontier_rows else None
    if best is not None:
        best_params = json.loads(best["params"])
        best_regime: dict[str, Any] = {"macro2_signal_mode": "sma200_and_mom"}
        best_fred: dict[str, Any] = {}

        for k, v in best_params.items():
            if str(k).startswith("fred."):
                best_fred[str(k).split(".", 1)[1]] = v
            elif str(k) == "fred_risk_weight_scale":
                scale = float(v)
                best_fred["risk_off_weights"] = {
                    str(name): float(weight) * scale
                    for name, weight in cfg.fred.risk_off_weights.items()
                }
            else:
                best_regime[str(k)] = v

        best_cfg = {
            "regime": best_regime,
            "backtest": {
                "strategy": "macro_only_v2",
            },
        }
        if best_fred:
            best_cfg["fred"] = best_fred
        best_cfg_path = write_strict_json(out_dir / "best_config.json", best_cfg)

        test_benchmark = grouped.get(best.get("param_id", ""), {}).get("test", {}).get("stress_1")
        test_repro = (
            f"python3 scripts/backtest.py --product {args.product} "
            f"--start {args.test_start} --end {args.test_end or end.isoformat().replace('+00:00', 'Z')} "
            f"--strategy macro_only_v2 --config {best_cfg_path} --output {out_dir / 'best_test_repro'}"
        )

        report = {
            "best": best,
            "constraints": {
                "max_drawdown_max": args.max_drawdown_max,
                "turnover_max": args.turnover_max,
            },
            "reproduce_test_command": test_repro,
            "best_config": best_cfg,
            "files": {
                "summary_csv": str(summary_path),
                "frontier_csv": str(frontier_path),
                "best_config_json": str(best_cfg_path),
            },
            "test_window_stress_1": {
                "cagr": float(test_benchmark.get("cagr", 0.0) if isinstance(test_benchmark, dict) else 0.0),
                "sharpe": float(test_benchmark.get("sharpe", 0.0) if isinstance(test_benchmark, dict) else 0.0),
                "max_drawdown": float(test_benchmark.get("max_drawdown", 0.0) if isinstance(test_benchmark, dict) else 0.0),
            },
            "benchmark_summary": baseline_rows,
        }
        write_strict_json(out_dir / "best_summary.json", report)

        print("macro_only_v2 frontier sweep completed — best config found")
        print(dumps_strict_json(report, indent=2))
        print(f"Summary: {summary_path}")
        print(f"Frontier: {frontier_path}")
        print(f"Reproduce best: {test_repro}")
    else:
        print("macro_only_v2 frontier sweep completed, but no configuration passed filters.")
        print(f"Summary: {summary_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
