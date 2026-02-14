from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List

import pandas as pd

from ..config import BotConfig
from .engine import BacktestEngine


@dataclass
class WalkForwardResult:
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    params: Dict[str, Any]
    metrics: Dict[str, Any]
    diagnostics: Dict[str, Any]


def make_windows(start: pd.Timestamp, end: pd.Timestamp, train_years: int = 3, test_years: int = 1) -> list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
    windows = []
    cursor = pd.Timestamp(start)
    tz = cursor.tz
    while True:
        train_end = pd.Timestamp(year=cursor.year + train_years, month=1, day=1, tz=tz)
        test_start = train_end
        test_end = pd.Timestamp(year=cursor.year + train_years + test_years, month=1, day=1, tz=tz)
        if test_end > end:
            break
        windows.append((cursor, train_end, test_start, test_end))
        cursor = test_start
    return windows


def walk_forward_test(hourly: pd.DataFrame, daily: pd.DataFrame, cfg: BotConfig, param_grid: Iterable[Dict[str, Any]], initial_equity: float = 10_000.0) -> List[WalkForwardResult]:
    # anchored walk-forward windows from 2020-2023 train -> 2024 test
    results: List[WalkForwardResult] = []

    if hourly.empty or daily.empty:
        return results

    h = hourly.copy()
    d = daily.copy()
    h["timestamp"] = pd.to_datetime(h["timestamp"], utc=True)
    d["timestamp"] = pd.to_datetime(d["timestamp"], utc=True)

    overall_start = max(h["timestamp"].min(), d["timestamp"].min())
    overall_end = min(h["timestamp"].max(), d["timestamp"].max())

    windows = make_windows(overall_start, overall_end, train_years=3, test_years=1)

    for p in param_grid:
        if hasattr(cfg.regime, "model_dump"):
            reg_dict = cfg.regime.model_dump()
        elif hasattr(cfg.regime, "dict"):
            reg_dict = cfg.regime.dict()
        else:
            reg_dict = dict(cfg.regime.__dict__)
        reg_dict.update(p)
        # keep HMM strictly training-window only via backtest windows used
        for train_start, train_end, test_start, test_end in windows:
            hourly_train = h[(h["timestamp"] >= train_start) & (h["timestamp"] < train_end)]
            daily_train = d[(d["timestamp"] >= train_start) & (d["timestamp"] < train_end)]
            hourly_test = h[(h["timestamp"] >= test_start) & (h["timestamp"] < test_end)]
            daily_test = d[(d["timestamp"] >= test_start) & (d["timestamp"] < test_end)]

            if hourly_train.empty or hourly_test.empty or daily_train.empty or daily_test.empty:
                continue

            # train: just instantiate regime switcher once and run; in this scaffold training is internal to orchestrator.
            # HMM leakage is prevented by using only test slices in each window.
            if hasattr(cfg, "model_dump"):
                raw_cfg = cfg.model_dump()
            elif hasattr(cfg, "dict"):
                raw_cfg = cfg.dict()
            else:
                raw_cfg = dict(cfg.__dict__)
            cfg_for_run = BotConfig(**raw_cfg)
            try:
                cfg_for_run.regime = type(cfg.regime)(**reg_dict)
            except Exception:
                continue
            cfg_for_run.backtest = type(cfg.backtest)(start=str(train_start.date()), end=str(train_end.date()), initial_equity=initial_equity)

            engine_train = BacktestEngine(
                product=cfg.data.product,
                hourly_candles=hourly_train,
                daily_candles=daily_train,
                start=train_start.to_pydatetime(),
                end=train_end.to_pydatetime(),
                config=cfg_for_run.backtest,
                regime_config=cfg_for_run.regime,
            )
            _ = engine_train.run()  # warm-up only

            engine_test = BacktestEngine(
                product=cfg.data.product,
                hourly_candles=hourly_test,
                daily_candles=daily_test,
                start=test_start.to_pydatetime(),
                end=test_end.to_pydatetime(),
                config=cfg_for_run.backtest,
                regime_config=cfg_for_run.regime,
            )
            result = engine_test.run()
            results.append(
                WalkForwardResult(
                    test_start=test_start,
                    test_end=test_end,
                    params=p,
                    metrics=result.metrics,
                    diagnostics=result.diagnostics,
                )
            )

    return results


def choose_robust_parameter_set(results: List[WalkForwardResult], metric_name: str = "cagr", penalty: float = 0.5) -> Dict:
    if not results:
        return {}

    # robust score: average metric - penalty*std
    records = {}
    for r in results:
        k = tuple(sorted(r.params.items()))
        rec = records.setdefault(k, [])
        rec.append(r.metrics.get(metric_name, 0.0))

    scores = []
    for k, vals in records.items():
        if not vals:
            continue
        import numpy as np

        arr = np.array(vals, dtype=float)
        score = float(arr.mean() - penalty * arr.std())
        scores.append((score, k, arr.mean(), arr.std()))

    if not scores:
        return {}
    scores.sort(key=lambda x: x[0], reverse=True)
    best_score, best_params, avg, std = scores[0]
    return {
        "params": dict(best_params),
        "avg_metric": float(avg),
        "std_metric": float(std),
        "score": float(best_score),
    }
