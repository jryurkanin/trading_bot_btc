from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from . import indicators


DEFAULT_COMPONENTS = [
    "close_gt_sma50",
    "close_gt_sma200",
    "ret_28d_pos",
    "ret_90d_pos",
]


@dataclass
class MacroScoreResult:
    score: float
    components: dict[str, float]
    multiplier: float
    enabled_components: list[str]


def _safe_component(condition: bool | float | int | None) -> float:
    if condition is None:
        return 0.0
    try:
        return 1.0 if bool(condition) else 0.0
    except Exception:
        return 0.0


def _last_close(daily_df: pd.DataFrame) -> float:
    if daily_df.empty:
        return 0.0
    try:
        return float(daily_df["close"].iloc[-1])
    except Exception:
        return 0.0


def _ret_pos(close: pd.Series, lookback_days: int) -> float:
    if close.empty:
        return 0.0
    base = close.shift(lookback_days).iloc[-1]
    if pd.isna(base) or float(base) == 0.0:
        return 0.0
    ret = float(close.iloc[-1] / float(base) - 1.0)
    return _safe_component(ret > 0)


def compute_macro_score(daily_df: pd.DataFrame, enabled_components: list[str] | None = None) -> tuple[float, dict[str, float], list[str]]:
    if daily_df is None or daily_df.empty:
        comps = {k: 0.0 for k in DEFAULT_COMPONENTS}
        return 0.0, comps, []

    close = daily_df["close"].astype(float)
    enabled = list(enabled_components) if enabled_components else list(DEFAULT_COMPONENTS)
    enabled_set = set(enabled)

    last = _last_close(daily_df)
    sma50 = indicators.sma(close, 50).iloc[-1] if len(close) else float("nan")
    sma200 = indicators.sma(close, 200).iloc[-1] if len(close) else float("nan")

    components = {
        "close_gt_sma50": _safe_component(pd.notna(sma50) and last > float(sma50)),
        "close_gt_sma200": _safe_component(pd.notna(sma200) and last > float(sma200)),
        "ret_28d_pos": _ret_pos(close, 28),
        "ret_90d_pos": _ret_pos(close, 90),
    }

    used = [k for k in components.keys() if k in enabled_set]
    if not used:
        used = list(DEFAULT_COMPONENTS)

    score = float(sum(components[k] for k in used) / max(1, len(used)))
    return score, components, used


def macro_multiplier_from_score(
    score: float,
    *,
    transform: str = "linear",
    floor: float = 0.0,
    min_to_trade: float = 0.25,
    piecewise_levels: list[float] | None = None,
) -> float:
    score = float(max(0.0, min(1.0, score)))
    floor = float(max(0.0, min(1.0, floor)))
    min_to_trade = float(max(0.0, min(1.0, min_to_trade)))

    if score < min_to_trade:
        return 0.0

    if transform == "piecewise":
        levels = [float(x) for x in (piecewise_levels or [0.0, 0.33, 0.66, 1.0])]
        levels = sorted(max(0.0, min(1.0, x)) for x in levels)
        if len(levels) == 1:
            return levels[0]

        # normalize score by floor, then map to equal-width bins.
        denom = max(1e-12, 1.0 - floor)
        normalized = max(0.0, min(1.0, (score - floor) / denom))
        bins = len(levels) - 1
        idx = min(int(normalized * bins + 1e-12), bins)
        return float(levels[idx])

    denom = max(1e-12, 1.0 - floor)
    linear = (score - floor) / denom
    return float(max(0.0, min(1.0, linear)))


def macro_result(daily_df: pd.DataFrame, cfg: Any) -> MacroScoreResult:
    score, components, used = compute_macro_score(daily_df, getattr(cfg, "macro_score_components", None))
    mult = macro_multiplier_from_score(
        score,
        transform=str(getattr(cfg, "macro_score_transform", "linear")),
        floor=float(getattr(cfg, "macro_score_floor", 0.0)),
        min_to_trade=float(getattr(cfg, "macro_score_min_to_trade", 0.25)),
        piecewise_levels=list(getattr(cfg, "macro_piecewise_levels", [0.0, 0.33, 0.66, 1.0])),
    )
    return MacroScoreResult(score=score, components=components, multiplier=mult, enabled_components=used)
