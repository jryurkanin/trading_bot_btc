from __future__ import annotations

from enum import Enum
from typing import Any

import pandas as pd

from ..features.indicators import sma
from ..features.macro_score import compute_macro_score
from ..config import RegimeConfig


class MacroStrength(str, Enum):
    OFF = "OFF"
    ON_HALF = "ON_HALF"
    ON_FULL = "ON_FULL"


def _to_timestamped_close(daily_df: pd.DataFrame) -> pd.Series:
    close = daily_df["close"].astype(float)
    return close


def _require_dated_series(daily_df: pd.DataFrame) -> pd.Series:
    if daily_df is None or daily_df.empty:
        raise ValueError("no_daily_data")
    return _to_timestamped_close(daily_df)


def _momentum_positive(close: pd.Series, lookback_days: int) -> float:
    if close.empty:
        return 0.0
    lb = max(1, int(lookback_days))
    if len(close) <= lb:
        return 0.0
    base = close.shift(lb).iloc[-1]
    last = close.iloc[-1]
    if pd.isna(base) or pd.isna(last) or float(base) <= 0.0:
        return 0.0
    return float(last / float(base) - 1.0)


def _strength_rank(strength: MacroStrength) -> int:
    return {
        MacroStrength.OFF: 0,
        MacroStrength.ON_HALF: 1,
        MacroStrength.ON_FULL: 2,
    }[strength]


def _rank_to_strength(rank: int) -> MacroStrength:
    if rank <= 0:
        return MacroStrength.OFF
    if rank == 1:
        return MacroStrength.ON_HALF
    return MacroStrength.ON_FULL


def _combine_strength_or(a: MacroStrength, b: MacroStrength) -> MacroStrength:
    return _rank_to_strength(max(_strength_rank(a), _strength_rank(b)))


def _combine_strength_and(a: MacroStrength, b: MacroStrength) -> MacroStrength:
    return _rank_to_strength(min(_strength_rank(a), _strength_rank(b)))


def macro_sma200_band_signal(daily_df: pd.DataFrame, cfg: RegimeConfig | Any) -> MacroStrength:
    """Signal from SMA200 with entry/exit banding.

    - ON_FULL: close >= SMA200 * (1 + entry_band)
    - ON_HALF: SMA200 * (1 + exit_band) <= close < SMA200 * (1 + entry_band)
    - OFF: otherwise
    """
    try:
        close = _require_dated_series(daily_df)
    except ValueError:
        return MacroStrength.OFF

    if len(close) < 200:
        return MacroStrength.OFF

    sma200 = sma(close, 200).iloc[-1]
    last = close.iloc[-1]
    if pd.isna(sma200) or pd.isna(last) or float(sma200) <= 0.0:
        return MacroStrength.OFF

    entry = float(getattr(cfg, "sma200_entry_band", 0.0) or 0.0)
    exit_band = float(getattr(cfg, "sma200_exit_band", 0.0) or 0.0)

    level_full = float(sma200) * (1.0 + entry)
    level_half = float(sma200) * (1.0 + exit_band)

    # if the user accidentally passes overlapping bands, keep deterministic ordering:
    level_full = max(level_full, level_half)
    level_half = min(level_full, level_half)

    if float(last) >= level_full:
        return MacroStrength.ON_FULL
    if float(last) >= level_half:
        return MacroStrength.ON_HALF
    return MacroStrength.OFF


def macro_mom_6_12_signal(daily_df: pd.DataFrame, cfg: RegimeConfig | Any) -> MacroStrength:
    """Signal from 6m and 12m momentum signs.

    - ON_FULL: both 6m and 12m momentum > 0
    - ON_HALF: exactly one positive
    - OFF: none
    """
    try:
        close = _require_dated_series(daily_df)
    except ValueError:
        return MacroStrength.OFF

    mom6 = _momentum_positive(close, int(getattr(cfg, "mom_6m_days", 180) or 180))
    mom12 = _momentum_positive(close, int(getattr(cfg, "mom_12m_days", 365) or 365))

    if mom6 > 0 and mom12 > 0:
        return MacroStrength.ON_FULL
    if mom6 > 0 or mom12 > 0:
        return MacroStrength.ON_HALF
    return MacroStrength.OFF


def macro_sma200_and_mom_signal(daily_df: pd.DataFrame, cfg: RegimeConfig | Any) -> MacroStrength:
    """AND combination: both SMA band and momentum must agree."""
    sma_sig = macro_sma200_band_signal(daily_df, cfg)
    mom_sig = macro_mom_6_12_signal(daily_df, cfg)
    return _combine_strength_and(sma_sig, mom_sig)


def macro_sma200_or_mom_signal(daily_df: pd.DataFrame, cfg: RegimeConfig | Any) -> MacroStrength:
    """OR combination: best of SMA band or momentum."""
    sma_sig = macro_sma200_band_signal(daily_df, cfg)
    mom_sig = macro_mom_6_12_signal(daily_df, cfg)
    return _combine_strength_or(sma_sig, mom_sig)


def macro_score4_legacy_signal(daily_df: pd.DataFrame, cfg: RegimeConfig | Any) -> MacroStrength:
    """Translate legacy 4-component macro score into OFF/HALF/FULL."""
    try:
        score, _, used = compute_macro_score(daily_df, getattr(cfg, "macro_score_components", None))
    except ValueError:
        return MacroStrength.OFF

    if not used:
        return MacroStrength.OFF

    score = float(score)
    if score >= 0.999999:
        return MacroStrength.ON_FULL
    if score >= 0.5:
        return MacroStrength.ON_HALF
    return MacroStrength.OFF


def macro_signal_strength(daily_df: pd.DataFrame, cfg: RegimeConfig | Any) -> MacroStrength:
    """Resolve active macro signal mode from configuration."""
    mode = str(getattr(cfg, "macro2_signal_mode", "sma200_and_mom"))
    if mode == "sma200_band":
        return macro_sma200_band_signal(daily_df, cfg)
    if mode == "mom_6_12":
        return macro_mom_6_12_signal(daily_df, cfg)
    if mode == "sma200_and_mom":
        return macro_sma200_and_mom_signal(daily_df, cfg)
    if mode == "sma200_or_mom":
        return macro_sma200_or_mom_signal(daily_df, cfg)
    if mode == "score4_legacy":
        return macro_score4_legacy_signal(daily_df, cfg)
    # fallback
    return macro_sma200_and_mom_signal(daily_df, cfg)


# convenience alias keeps imports tidy in callers
MacroStrengthDecision = MacroStrength
