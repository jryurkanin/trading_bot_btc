from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Dict, Iterable, List

import pandas as pd

from ..config import BotConfig
from ..system_log import get_system_logger
from .engine import BacktestEngine

logger = get_system_logger("backtest.walkforward")


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


def _set_cfg_param(cfg: BotConfig, key: str, value: Any) -> None:
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
        for part in parts[:-1]:
            obj = getattr(obj, part)
        setattr(obj, parts[-1], value)
        return

    for section in [cfg.regime, cfg.execution, cfg.backtest, cfg.risk, cfg.fred]:
        if hasattr(section, key):
            setattr(section, key, value)
            return

    raise KeyError(f"Unknown walk-forward parameter: {key}")


def walk_forward_test(hourly: pd.DataFrame, daily: pd.DataFrame, cfg: BotConfig, param_grid: Iterable[Dict[str, Any]], initial_equity: float = 10_000.0) -> List[WalkForwardResult]:
    # anchored walk-forward windows from 2020-2023 train -> 2024 test
    results: List[WalkForwardResult] = []

    if hourly.empty or daily.empty:
        logger.warning("walkforward_skipped reason=empty_input hourly_empty=%s daily_empty=%s", hourly.empty, daily.empty)
        return results

    h = hourly.copy()
    d = daily.copy()
    h["timestamp"] = pd.to_datetime(h["timestamp"], utc=True)
    d["timestamp"] = pd.to_datetime(d["timestamp"], utc=True)

    overall_start = max(h["timestamp"].min(), d["timestamp"].min())
    overall_end = min(h["timestamp"].max(), d["timestamp"].max())

    windows = make_windows(overall_start, overall_end, train_years=3, test_years=1)
    param_sets = list(param_grid)

    logger.info(
        "walkforward_start product=%s params=%d windows=%d range_start=%s range_end=%s",
        cfg.data.product,
        len(param_sets),
        len(windows),
        overall_start,
        overall_end,
    )

    skipped_windows = 0

    for p_idx, p in enumerate(param_sets, start=1):
        if hasattr(cfg, "model_dump"):
            raw_cfg = cfg.model_dump()
        elif hasattr(cfg, "dict"):
            raw_cfg = cfg.dict()
        else:
            raw_cfg = dict(cfg.__dict__)

        try:
            cfg_for_param = BotConfig(**raw_cfg)
            for k, v in dict(p).items():
                _set_cfg_param(cfg_for_param, str(k), v)
        except Exception:
            logger.exception("walkforward_param_invalid index=%d params=%s", p_idx, p)
            continue

        logger.info("walkforward_param_start index=%d/%d params=%s", p_idx, len(param_sets), p)

        # Warmup: include pre-window history for feature warmup (SMA200,
        # momentum, FRED z-scores, etc.) — mirror the _prefetch_start logic
        # used by all script-level entrypoints.
        warmup_days = max(
            400,
            int(getattr(cfg_for_param.regime, "mom_12m_days", 365) or 365) + 30,
            int(getattr(cfg_for_param.regime, "vol_lookback_days", 365) or 365) + 30,
            int(getattr(cfg_for_param.fred, "daily_z_lookback", 252) or 252) + 30,
        )
        warmup_td = timedelta(days=warmup_days)

        for w_idx, (train_start, train_end, test_start, test_end) in enumerate(windows, start=1):
            # Include warmup history before each window start so the engine
            # can compute rolling features without defaulting to zero/OFF.
            train_prefetch = train_start - warmup_td
            test_prefetch = test_start - warmup_td

            hourly_train = h[(h["timestamp"] >= train_prefetch) & (h["timestamp"] < train_end)]
            daily_train = d[(d["timestamp"] >= train_prefetch) & (d["timestamp"] < train_end)]
            hourly_test = h[(h["timestamp"] >= test_prefetch) & (h["timestamp"] < test_end)]
            daily_test = d[(d["timestamp"] >= test_prefetch) & (d["timestamp"] < test_end)]

            if hourly_train.empty or hourly_test.empty or daily_train.empty or daily_test.empty:
                skipped_windows += 1
                logger.debug(
                    "walkforward_window_skipped param_index=%d window_index=%d train_start=%s test_end=%s",
                    p_idx,
                    w_idx,
                    train_start,
                    test_end,
                )
                continue

            # train: run on training slice first, then evaluate on out-of-sample test slice.
            # Feature pipelines only use values available at each bar timestamp.
            if hasattr(cfg_for_param, "model_dump"):
                raw_run_cfg = cfg_for_param.model_dump()
            elif hasattr(cfg_for_param, "dict"):
                raw_run_cfg = cfg_for_param.dict()
            else:
                raw_run_cfg = dict(cfg_for_param.__dict__)

            cfg_for_run = BotConfig(**raw_run_cfg)
            if hasattr(cfg_for_param.backtest, "model_dump"):
                bt_payload = cfg_for_param.backtest.model_dump()
            elif hasattr(cfg_for_param.backtest, "dict"):
                bt_payload = cfg_for_param.backtest.dict()
            else:
                bt_payload = dict(cfg_for_param.backtest.__dict__)
            bt_payload.update(
                {
                    "start": str(train_start.date()),
                    "end": str(train_end.date()),
                    "initial_equity": initial_equity,
                }
            )
            cfg_for_run.backtest = type(cfg_for_param.backtest)(**bt_payload)

            logger.info(
                "walkforward_window_start param_index=%d/%d window_index=%d/%d train=%s..%s test=%s..%s",
                p_idx,
                len(param_sets),
                w_idx,
                len(windows),
                train_start.date(),
                train_end.date(),
                test_start.date(),
                test_end.date(),
            )

            engine_train = BacktestEngine(
                product=cfg.data.product,
                hourly_candles=hourly_train,
                daily_candles=daily_train,
                start=train_start.to_pydatetime(),
                end=train_end.to_pydatetime(),
                config=cfg_for_run.backtest,
                regime_config=cfg_for_run.regime,
                risk_config=cfg_for_run.risk,
                execution_config=cfg_for_run.execution,
                fred_config=cfg_for_run.fred,
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
                risk_config=cfg_for_run.risk,
                execution_config=cfg_for_run.execution,
                fred_config=cfg_for_run.fred,
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
            logger.info(
                "walkforward_window_complete param_index=%d window_index=%d metric_cagr=%s metric_sharpe=%s",
                p_idx,
                w_idx,
                result.metrics.get("cagr"),
                result.metrics.get("sharpe"),
            )

    logger.info(
        "walkforward_complete results=%d skipped_windows=%d params=%d",
        len(results),
        skipped_windows,
        len(param_sets),
    )

    return results


def choose_robust_parameter_set(results: List[WalkForwardResult], metric_name: str = "cagr", penalty: float = 0.5) -> Dict:
    if not results:
        logger.warning("walkforward_choose_best_empty_results metric=%s", metric_name)
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
        logger.warning("walkforward_choose_best_no_scores metric=%s", metric_name)
        return {}
    scores.sort(key=lambda x: x[0], reverse=True)
    best_score, best_params, avg, std = scores[0]
    logger.info(
        "walkforward_choose_best metric=%s candidates=%d best_score=%.6f avg=%.6f std=%.6f",
        metric_name,
        len(scores),
        best_score,
        avg,
        std,
    )
    return {
        "params": dict(best_params),
        "avg_metric": float(avg),
        "std_metric": float(std),
        "score": float(best_score),
    }
