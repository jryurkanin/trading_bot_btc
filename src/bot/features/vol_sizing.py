from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from ..features.macro_score import MacroState


def realized_ann_vol_from_daily(daily_df: pd.DataFrame, lookback_days: int | None = None) -> float:
    """Compute annualized realized vol from daily log returns.

    Returns 0.0 when unavailable/insufficient.
    """
    if daily_df is None or daily_df.empty:
        return 0.0

    if "close" not in daily_df.columns:
        return 0.0

    close = pd.to_numeric(daily_df["close"], errors="coerce").astype(float)
    if close.empty:
        return 0.0

    log_ret = np.log(close / close.shift(1))
    if lookback_days is not None and lookback_days > 1:
        log_ret = log_ret.tail(int(lookback_days))

    log_ret = log_ret.dropna()
    if log_ret.empty:
        return 0.0

    vol = float(np.std(log_ret.to_numpy(), ddof=0) * math.sqrt(365))
    return vol if np.isfinite(vol) else 0.0


def state_weight_for_gate_state(state: MacroState, cfg: Any) -> float:
    if state == MacroState.ON_FULL:
        return float(getattr(cfg, "macro2_weight_full", 1.0))
    if state == MacroState.ON_HALF:
        return float(getattr(cfg, "macro2_weight_half", 0.5))
    return float(getattr(cfg, "macro2_weight_off", 0.0))


def state_target_vol(state: MacroState, cfg: Any) -> float:
    if state == MacroState.ON_FULL:
        return float(getattr(cfg, "macro2_target_ann_vol_full", 0.60))
    if state == MacroState.ON_HALF:
        return float(getattr(cfg, "macro2_target_ann_vol_half", 0.30))
    return 0.0


def sized_weight(
    *,
    state: MacroState,
    realized_vol: float,
    mode: str,
    cfg: Any,
) -> float:
    """Apply volatile sizing for macro2 states.

    - vol_mode == "none": return gate state weight directly.
    - vol_mode == "inverse_vol": weight = target_ann_vol / max(realized_vol, vol_floor).

    If vol_mode is unknown, fall back to current state's raw weight.
    """
    wt = state_weight_for_gate_state(state, cfg)
    mode = str(mode or "none").lower()

    if mode == "none":
        return max(0.0, min(1.0, wt))

    if mode == "inverse_vol":
        target_ann_vol = max(0.0, state_target_vol(state, cfg))
        vol_floor = max(1e-9, float(getattr(cfg, "macro2_vol_floor", 0.05)))
        base = target_ann_vol / max(float(realized_vol or 0.0), vol_floor)
        return max(0.0, min(1.0, base))

    # Unknown modes fail-safe to state weight.
    return max(0.0, min(1.0, wt))
