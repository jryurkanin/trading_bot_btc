"""Adaptive Trend 6H Strategy — 6-hour momentum + Chandelier exit + monthly reoptimization."""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product as itertools_product
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from ..config import RegimeConfig
from ..features import indicators
from ..features.macro_score import MacroState
from ..features.regime import RegimeState
from ..system_log import get_system_logger
from .macro_gate import V4MacroGate
from .regime_switching_orchestrator import RegimeDecisionBundle

logger = get_system_logger("strategy.adaptive_trend_6h")

_PERIODS_PER_YEAR_6H = 1460  # 365 * 24 / 6


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

@dataclass
class Adaptive6HState:
    in_position: bool = False
    entry_price: Optional[float] = None
    trailing_stop: Optional[float] = None
    last_6h_bar_end: Optional[pd.Timestamp] = None

    active_params: Dict[str, Any] = field(default_factory=dict)
    last_reopt_key: Optional[str] = None


# ---------------------------------------------------------------------------
# 6H resampling (no lookahead)
# ---------------------------------------------------------------------------

def resample_6h(hourly_df: pd.DataFrame, hourly_idx: int, bar_hours: int = 6) -> pd.DataFrame:
    """Resample hourly OHLCV to *bar_hours*-hour bars without lookahead.

    Only data up to and including *hourly_idx* is used, so every 6H bar is
    computed from data already available at the current hour.  We use
    ``label="left"`` so the bar label equals the *start* of the period,
    which is always <= the last hourly timestamp that fed into the bar.

    .. note::

       For backtesting, prefer :func:`precompute_6h_bars` which resamples
       the full hourly series once and returns a mapping that allows O(1)
       slicing per bar.  This function remains as a convenience for live
       trading where the hourly DataFrame grows incrementally.
    """
    h = hourly_df.iloc[: hourly_idx + 1].copy()
    if h.empty:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume", "timestamp"])

    # Handle timestamp as column or index
    if "timestamp" in h.columns:
        ts = pd.to_datetime(h["timestamp"], utc=True)
        h = h.set_index(ts)
    elif isinstance(h.index, pd.DatetimeIndex):
        if h.index.tz is None:
            h.index = h.index.tz_localize("UTC")
    else:
        ts = pd.to_datetime(h.index, utc=True)
        h.index = ts

    rule = f"{bar_hours}h"
    o = h["open"].resample(rule, closed="right", label="left").first()
    hi = h["high"].resample(rule, closed="right", label="left").max()
    lo = h["low"].resample(rule, closed="right", label="left").min()
    c = h["close"].resample(rule, closed="right", label="left").last()
    v = h["volume"].resample(rule, closed="right", label="left").sum()

    out = pd.DataFrame({"open": o, "high": hi, "low": lo, "close": c, "volume": v}).dropna()
    out["timestamp"] = out.index
    return out


def precompute_6h_bars(
    hourly_df: pd.DataFrame,
    bar_hours: int = 6,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Resample the *full* hourly series to 6H bars once and build an index
    mapping so each backtest bar can obtain the correct no-lookahead slice
    in O(1) via ``np.searchsorted``.

    Returns
    -------
    h6_full : pd.DataFrame
        All 6H bars with the same columns as :func:`resample_6h`.
    max_hourly_idx : np.ndarray[int]
        ``max_hourly_idx[i]`` is the last (inclusive) hourly row-position
        that contributed to ``h6_full.iloc[i]``.  To obtain the valid 6H
        bars at a given ``hourly_idx``, use::

            n_valid = int(np.searchsorted(max_hourly_idx, hourly_idx, side="right"))
            h6_slice = h6_full.iloc[:n_valid]
    """
    h = hourly_df.copy()
    if h.empty:
        empty = pd.DataFrame(columns=["open", "high", "low", "close", "volume", "timestamp"])
        return empty, np.array([], dtype=int)

    if "timestamp" in h.columns:
        ts = pd.to_datetime(h["timestamp"], utc=True)
        h = h.set_index(ts)
    elif isinstance(h.index, pd.DatetimeIndex):
        if h.index.tz is None:
            h.index = h.index.tz_localize("UTC")
    else:
        ts = pd.to_datetime(h.index, utc=True)
        h.index = ts

    rule = f"{bar_hours}h"
    o = h["open"].resample(rule, closed="right", label="left").first()
    hi = h["high"].resample(rule, closed="right", label="left").max()
    lo = h["low"].resample(rule, closed="right", label="left").min()
    c = h["close"].resample(rule, closed="right", label="left").last()
    v = h["volume"].resample(rule, closed="right", label="left").sum()

    h6 = pd.DataFrame({"open": o, "high": hi, "low": lo, "close": c, "volume": v}).dropna()
    h6["timestamp"] = h6.index

    if h6.empty:
        return h6, np.array([], dtype=int)

    # Build the mapping: for each 6H bar, the last hourly index that fed
    # into it.  With closed="right" / label="left", a 6H bar labelled *t*
    # covers the interval (t, t + bar_hours].  The last hourly timestamp
    # contributing to it is therefore <= t + bar_hours.
    hourly_timestamps = h.index
    bar_end_offsets = h6.index + pd.Timedelta(hours=bar_hours)

    max_hourly_idx = np.empty(len(h6), dtype=int)
    for i, bar_end in enumerate(bar_end_offsets):
        pos = int(hourly_timestamps.searchsorted(bar_end, side="right")) - 1
        max_hourly_idx[i] = max(0, pos)

    return h6, max_hourly_idx


# ---------------------------------------------------------------------------
# Indicators on 6H bars
# ---------------------------------------------------------------------------

def _momentum(close: pd.Series, lookback: int) -> Optional[float]:
    if len(close) <= lookback:
        return None
    return float(close.iloc[-1] / close.iloc[-lookback - 1] - 1.0)


def _atr_6h(h6: pd.DataFrame, window: int) -> Optional[float]:
    if len(h6) < window:
        return None
    atr_series = indicators.atr(h6["high"], h6["low"], h6["close"], window=window)
    val = atr_series.iloc[-1]
    if pd.isna(val):
        return None
    return float(val)


def _realized_vol_6h(close: pd.Series, window: int = 20) -> Optional[float]:
    if len(close) < window + 1:
        return None
    returns = close.pct_change().dropna()
    rv = indicators.realized_vol(returns, window=window, periods_per_year=_PERIODS_PER_YEAR_6H)
    val = rv.iloc[-1]
    if pd.isna(val):
        return None
    return float(val)


# ---------------------------------------------------------------------------
# Internal fast simulator for reoptimization
# ---------------------------------------------------------------------------

def _simulate_window(
    h6: pd.DataFrame,
    L: int,
    theta: float,
    atr_window: int,
    atr_mult: float,
    cost_per_unit_turnover: float,
) -> Dict[str, float]:
    """Run a fast long/flat simulation on 6H bars and return scoring metrics."""
    close = h6["close"].values.astype(float)
    high = h6["high"].values.astype(float)
    low = h6["low"].values.astype(float)
    n = len(close)

    if n <= max(L, atr_window) + 1:
        return {"sharpe": 0.0, "calmar": 0.0, "turnover": 0.0, "n_trades": 0}

    # Compute ATR array
    atr_series = indicators.atr(
        pd.Series(high), pd.Series(low), pd.Series(close), window=atr_window
    ).values.astype(float)

    position = 0.0  # 0 or 1
    stop = 0.0
    returns = []
    turnover_sum = 0.0
    n_trades = 0

    for i in range(max(L, atr_window), n):
        mom = close[i] / close[i - L] - 1.0 if close[i - L] > 0 else 0.0
        cur_atr = atr_series[i]

        prev_pos = position

        if position == 0.0:
            # Entry check
            if mom > theta and not np.isnan(cur_atr) and cur_atr > 0:
                position = 1.0
                stop = close[i] - atr_mult * cur_atr
                n_trades += 1
        else:
            # Update trailing stop (ratchet up)
            if not np.isnan(cur_atr) and cur_atr > 0:
                candidate = close[i] - atr_mult * cur_atr
                stop = max(stop, candidate)
            # Exit check on 6H low
            if low[i] <= stop:
                position = 0.0
                stop = 0.0

        # Return for this bar
        bar_ret = (close[i] / close[i - 1] - 1.0) if close[i - 1] > 0 else 0.0
        gross_ret = prev_pos * bar_ret
        delta = abs(position - prev_pos)
        turnover_sum += delta
        cost = delta * cost_per_unit_turnover
        returns.append(gross_ret - cost)

    if len(returns) < 2:
        return {"sharpe": 0.0, "calmar": 0.0, "turnover": 0.0, "n_trades": 0}

    arr = np.array(returns, dtype=float)
    mean_r = float(np.mean(arr))
    std_r = float(np.std(arr, ddof=1))

    sharpe = float(np.sqrt(_PERIODS_PER_YEAR_6H) * mean_r / std_r) if std_r > 0 else 0.0

    # Max drawdown for calmar
    cum = np.cumprod(1.0 + arr)
    peak = np.maximum.accumulate(cum)
    dd = (cum - peak) / np.where(peak > 0, peak, 1.0)
    max_dd = float(np.min(dd))

    # CAGR
    n_bars = len(arr)
    years = n_bars / _PERIODS_PER_YEAR_6H
    cagr = float((cum[-1]) ** (1.0 / years) - 1.0) if years > 0 and cum[-1] > 0 else 0.0
    calmar = float(cagr / abs(max_dd)) if abs(max_dd) > 1e-10 else 0.0

    turnover_rate = turnover_sum / max(n_bars, 1)

    return {
        "sharpe": sharpe,
        "calmar": calmar,
        "turnover": turnover_rate,
        "n_trades": n_trades,
    }


# ---------------------------------------------------------------------------
# Strategy class
# ---------------------------------------------------------------------------

class AdaptiveTrend6HStrategy:
    """6H momentum + Chandelier exit with periodic parameter reoptimization."""

    def __init__(self, cfg: RegimeConfig) -> None:
        self.cfg = cfg
        self.state = Adaptive6HState()

        # Macro overlay
        self._gate: Optional[V4MacroGate] = None
        if cfg.adaptive6h_use_macro_gate:
            self._gate = V4MacroGate(cfg)

        # Daily bar caching (same pattern as MacroGateBenchmarkStrategy)
        self._daily_index_cache_sig: Optional[tuple] = None
        self._daily_index_values: Optional[np.ndarray] = None
        self._daily_ts_cache: Dict[int, pd.DataFrame] = {}
        self._daily_last_ts_cache: Dict[int, Optional[pd.Timestamp]] = {}

    def reset(self) -> None:
        self.state = Adaptive6HState()
        if self._gate is not None:
            self._gate.reset()
        self._daily_ts_cache.clear()
        self._daily_last_ts_cache.clear()
        self._daily_index_cache_sig = None
        self._daily_index_values = None

    # ------------------------------------------------------------------
    # Precomputed features — 6H bars built once for the full series
    # ------------------------------------------------------------------

    @staticmethod
    def get_precomputed_features(
        hourly_df: pd.DataFrame,
        cfg: RegimeConfig,
        *,
        backend: str = "cpu",
    ) -> Dict[str, Any]:
        bar_hours = int(getattr(cfg, "adaptive6h_bar_hours", 6) or 6)
        h6_full, h6_max_hourly_idx = precompute_6h_bars(hourly_df, bar_hours=bar_hours)
        return {
            "_h6_full": h6_full,
            "_h6_max_hourly_idx": h6_max_hourly_idx,
        }

    # ------------------------------------------------------------------
    # Default params
    # ------------------------------------------------------------------

    def _default_params(self) -> Dict[str, Any]:
        return {
            "L": self.cfg.adaptive6h_mom_lookbacks[0],
            "theta": self.cfg.adaptive6h_entry_thresholds[0],
            "atr_window": self.cfg.adaptive6h_atr_window_choices[0],
            "atr_mult": self.cfg.adaptive6h_atr_mult_choices[0],
        }

    # ------------------------------------------------------------------
    # Reoptimization
    # ------------------------------------------------------------------

    def _reopt_key(self, ts: pd.Timestamp) -> str:
        if self.cfg.adaptive6h_reopt_cadence == "weekly":
            iso = ts.isocalendar()
            return f"{iso[0]}-W{iso[1]:02d}"
        return ts.strftime("%Y-%m")

    def _maybe_reoptimize(self, h6: pd.DataFrame) -> bool:
        """Run grid search on historical 6H window when entering a new period."""
        if h6.empty:
            return False

        raw_ts = h6["timestamp"].iloc[-1]
        last6_end = pd.to_datetime(raw_ts, utc=True) if raw_ts is not None else None
        if last6_end is None:
            return False

        key = self._reopt_key(last6_end)
        if key == self.state.last_reopt_key:
            return False

        window_end = last6_end
        window_start = window_end - pd.Timedelta(days=self.cfg.adaptive6h_reopt_lookback_days)

        h6_ts = pd.to_datetime(h6["timestamp"], utc=True)
        h6_window = h6[(h6_ts > window_start) & (h6_ts <= window_end)]

        if len(h6_window) < 10:
            self.state.last_reopt_key = key
            return False

        # Cost proxy from config (bps per unit turnover)
        cost_per_unit = self.cfg.adaptive6h_reopt_cost_bps / 10_000.0

        best_score = -np.inf
        best_params: Optional[Dict[str, Any]] = None

        for L, theta, atr_w, atr_m in itertools_product(
            self.cfg.adaptive6h_mom_lookbacks,
            self.cfg.adaptive6h_entry_thresholds,
            self.cfg.adaptive6h_atr_window_choices,
            self.cfg.adaptive6h_atr_mult_choices,
        ):
            metrics = _simulate_window(h6_window, L, theta, atr_w, atr_m, cost_per_unit)

            if metrics["n_trades"] < self.cfg.adaptive6h_min_trades_in_window:
                continue

            if self.cfg.adaptive6h_objective == "sharpe":
                score = metrics["sharpe"]
            elif self.cfg.adaptive6h_objective == "calmar":
                score = metrics["calmar"]
            else:  # sharpe_minus_turnover
                score = metrics["sharpe"] - self.cfg.adaptive6h_turnover_penalty * metrics["turnover"]

            if score > best_score:
                best_score = score
                best_params = {"L": L, "theta": theta, "atr_window": atr_w, "atr_mult": atr_m}

        if best_params is not None:
            self.state.active_params = best_params
        elif not self.state.active_params:
            self.state.active_params = self._default_params()

        self.state.last_reopt_key = key
        logger.info(
            "adaptive6h_reopt key=%s params=%s score=%.4f",
            key,
            self.state.active_params,
            best_score if best_score > -np.inf else 0.0,
        )
        return True

    # ------------------------------------------------------------------
    # Position sizing
    # ------------------------------------------------------------------

    def _position_size(self, rv: Optional[float]) -> float:
        if not self.cfg.adaptive6h_use_vol_target:
            return self.cfg.adaptive6h_max_position_fraction
        if rv is None or rv <= 0:
            return 0.0
        raw = self.cfg.adaptive6h_target_ann_vol / rv
        return min(self.cfg.adaptive6h_max_position_fraction,
                   max(self.cfg.adaptive6h_min_position_fraction, raw))

    # ------------------------------------------------------------------
    # Trailing stop management
    # ------------------------------------------------------------------

    def _enter_position(self, entry_price: float, atr_val: float, atr_mult: float) -> None:
        self.state.in_position = True
        self.state.entry_price = entry_price
        self.state.trailing_stop = entry_price - atr_mult * atr_val

    def _update_trailing_stop(self, close: float, atr_val: float, atr_mult: float) -> None:
        candidate = close - atr_mult * atr_val
        if self.state.trailing_stop is not None:
            self.state.trailing_stop = max(self.state.trailing_stop, candidate)
        else:
            self.state.trailing_stop = candidate

    def _exit_position(self) -> None:
        self.state.in_position = False
        self.state.entry_price = None
        self.state.trailing_stop = None

    # ------------------------------------------------------------------
    # Daily bar helpers (for macro gate)
    # ------------------------------------------------------------------

    @staticmethod
    def _to_timestamp_col(df: pd.DataFrame) -> pd.Series:
        if "timestamp" in df.columns:
            return pd.to_datetime(df["timestamp"], utc=True)
        idx = pd.to_datetime(df.index, utc=True)
        return pd.Series(idx, index=df.index)

    @staticmethod
    def _day_cache_key(decision_ts: pd.Timestamp) -> int:
        ts = pd.Timestamp(decision_ts)
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        else:
            ts = ts.tz_convert("UTC")
        return int(ts.floor("D").value)

    def _ensure_daily_index_cache(self, daily_df: pd.DataFrame) -> None:
        if daily_df is None or daily_df.empty:
            self._daily_index_cache_sig = None
            self._daily_index_values = None
            return
        ts = self._to_timestamp_col(daily_df)
        if len(ts):
            last_val = int(pd.Timestamp(ts.iloc[-1]).value)
        else:
            last_val = 0
        sig = (int(len(daily_df)), last_val)
        if self._daily_index_cache_sig == sig and self._daily_index_values is not None:
            return
        self._daily_index_values = ts.to_numpy(dtype="datetime64[ns]")
        self._daily_index_cache_sig = sig
        self._daily_ts_cache.clear()
        self._daily_last_ts_cache.clear()

    def _closed_daily_cached(self, daily_df: pd.DataFrame, decision_ts: pd.Timestamp) -> pd.DataFrame:
        if daily_df is None or daily_df.empty:
            return pd.DataFrame(columns=daily_df.columns if daily_df is not None else [])
        self._ensure_daily_index_cache(daily_df)
        key = self._day_cache_key(decision_ts)
        if key in self._daily_ts_cache:
            return self._daily_ts_cache[key]
        cutoff = pd.Timestamp(decision_ts)
        if cutoff.tzinfo is None:
            cutoff = cutoff.tz_localize("UTC")
        else:
            cutoff = cutoff.tz_convert("UTC")
        closed_cutoff = cutoff.floor("D")
        idx_vals = self._daily_index_values
        if idx_vals is None:
            d = pd.DataFrame(columns=daily_df.columns)
        else:
            pos = int(np.searchsorted(idx_vals, closed_cutoff.to_numpy(), side="left"))
            d = daily_df.iloc[:pos]
        self._daily_ts_cache[key] = d
        return d

    def _latest_daily_ts_cached(self, daily_df: pd.DataFrame, decision_ts: pd.Timestamp) -> Optional[pd.Timestamp]:
        key = self._day_cache_key(decision_ts)
        if key in self._daily_last_ts_cache:
            return self._daily_last_ts_cache[key]
        closed = self._closed_daily_cached(daily_df, decision_ts)
        if closed.empty:
            self._daily_last_ts_cache[key] = None
            return None
        ts = self._to_timestamp_col(closed)
        last = pd.to_datetime(ts.iloc[-1], utc=True)
        self._daily_last_ts_cache[key] = last
        return last

    def _macro_info(self, daily_df: pd.DataFrame, decision_ts: pd.Timestamp) -> Dict[str, Any]:
        """Return macro multiplier, state, score, and components."""
        if self._gate is None:
            return {
                "mult": 1.0,
                "state": MacroState.ON_FULL,
                "score": 1.0,
                "score_raw": 1.0,
                "components": {},
            }
        daily_closed = self._closed_daily_cached(daily_df, decision_ts)
        daily_bar_ts = self._latest_daily_ts_cached(daily_df, decision_ts)
        state, mult, score, components = self._gate.update(daily_closed, daily_bar_ts)
        mult = max(float(mult), self.cfg.adaptive6h_macro_multiplier_floor)
        return {
            "mult": mult,
            "state": state if isinstance(state, MacroState) else MacroState.OFF,
            "score": float(score),
            "score_raw": float(score),
            "components": components if isinstance(components, dict) else {},
        }

    @staticmethod
    def _latest_daily_feature(daily_closed: pd.DataFrame, col: str, *, default: float = float("nan")) -> float:
        if daily_closed is None or daily_closed.empty or col not in daily_closed.columns:
            return default
        val = daily_closed[col].iloc[-1]
        if pd.isna(val):
            return default
        return float(val)

    # ------------------------------------------------------------------
    # State persistence
    # ------------------------------------------------------------------

    def runtime_state(self) -> dict:
        gate_payload: Dict[str, Any] = {}
        if self._gate is not None:
            gate_state = self._gate._cached_state
            gate_payload = {
                "cached_state": gate_state.value if isinstance(gate_state, MacroState) else str(gate_state),
                "cached_multiplier": float(self._gate._cached_multiplier),
                "cached_score": float(self._gate._cached_score),
                "cached_components": dict(self._gate._cached_components),
                "last_daily_ts": str(self._gate._last_daily_ts) if self._gate._last_daily_ts is not None else None,
                "state_machine": self._gate._gate.snapshot().__dict__,
            }
        return {
            "gate": gate_payload,
            "position": {
                "in_position": self.state.in_position,
                "entry_price": self.state.entry_price,
                "trailing_stop": self.state.trailing_stop,
                "last_6h_bar_end": str(self.state.last_6h_bar_end) if self.state.last_6h_bar_end is not None else None,
            },
            "reopt": {
                "active_params": dict(self.state.active_params) if self.state.active_params else {},
                "last_reopt_key": self.state.last_reopt_key,
            },
        }

    def load_runtime_state(self, payload: dict | None) -> None:
        if not isinstance(payload, dict):
            return

        # Restore position state
        pos = payload.get("position")
        if isinstance(pos, dict):
            if isinstance(pos.get("in_position"), bool):
                self.state.in_position = pos["in_position"]
            ep = pos.get("entry_price")
            if isinstance(ep, (int, float)):
                self.state.entry_price = float(ep)
            ts_val = pos.get("trailing_stop")
            if isinstance(ts_val, (int, float)):
                self.state.trailing_stop = float(ts_val)
            l6 = pos.get("last_6h_bar_end")
            if l6 is not None:
                try:
                    self.state.last_6h_bar_end = pd.Timestamp(l6)
                except Exception:
                    pass

        # Restore reopt state
        reopt = payload.get("reopt")
        if isinstance(reopt, dict):
            ap = reopt.get("active_params")
            if isinstance(ap, dict) and ap:
                self.state.active_params = dict(ap)
            lk = reopt.get("last_reopt_key")
            if isinstance(lk, str):
                self.state.last_reopt_key = lk

        # Restore gate state
        gate_state = payload.get("gate")
        if isinstance(gate_state, dict) and self._gate is not None:
            raw_state = gate_state.get("cached_state")
            if isinstance(raw_state, str):
                try:
                    self._gate._cached_state = MacroState(raw_state)
                except Exception:
                    pass
            mult = gate_state.get("cached_multiplier")
            if isinstance(mult, (int, float)):
                self._gate._cached_multiplier = float(mult)
            score = gate_state.get("cached_score")
            if isinstance(score, (int, float)):
                self._gate._cached_score = float(score)
            comp = gate_state.get("cached_components")
            if isinstance(comp, dict):
                self._gate._cached_components = comp
            lt = gate_state.get("last_daily_ts")
            if lt is not None:
                try:
                    self._gate._last_daily_ts = pd.Timestamp(lt)
                except Exception:
                    pass
            sm = gate_state.get("state_machine")
            if isinstance(sm, dict):
                self._gate._gate.restore(sm)

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def compute_target_position(
        self,
        timestamp: pd.Timestamp,
        hourly_df: pd.DataFrame,
        daily_df: pd.DataFrame,
        current_exposure: float,
        hourly_idx: int | None = None,
        micro_precomputed: Dict[str, Any] | None = None,
    ) -> RegimeDecisionBundle:
        if hourly_df.empty:
            return self._empty_bundle(timestamp, "no_hourly_data")

        if hourly_idx is None:
            hourly_idx = len(hourly_df) - 1
        hourly_idx = max(0, min(int(hourly_idx), len(hourly_df) - 1))

        ts = pd.Timestamp(timestamp)
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        else:
            ts = ts.tz_convert("UTC")

        # 1) Build 6H bars — use precomputed O(1) lookup when available,
        #    fall back to per-bar resample for live runner compatibility.
        precomputed = micro_precomputed or {}
        h6_full = precomputed.get("_h6_full")
        h6_max_idx = precomputed.get("_h6_max_hourly_idx")

        if h6_full is not None and h6_max_idx is not None and len(h6_max_idx) > 0:
            n_valid = int(np.searchsorted(h6_max_idx, hourly_idx, side="right"))
            h6 = h6_full.iloc[:n_valid]
        else:
            h6 = resample_6h(hourly_df, hourly_idx, self.cfg.adaptive6h_bar_hours)
        if len(h6) < 5:
            return self._empty_bundle(ts, "insufficient_6h_history")

        last6 = h6.iloc[-1]
        last6_end = pd.to_datetime(last6["timestamp"], utc=True)

        # 2) Detect new 6H bar
        is_new_6h_bar = (
            self.state.last_6h_bar_end is None
            or last6_end > self.state.last_6h_bar_end
        )

        # 3) Reopt on period boundaries (only when a new 6H bar is available)
        did_reopt = False
        if is_new_6h_bar:
            did_reopt = self._maybe_reoptimize(h6)

        # 4) Load active params or default
        params = self.state.active_params or self._default_params()
        L = params["L"]
        theta = params["theta"]
        atr_window = params["atr_window"]
        atr_mult_val = params["atr_mult"]

        # 5) Compute indicators on 6H bars
        mom = _momentum(h6["close"], L)
        atr_val = _atr_6h(h6, atr_window)
        rv = _realized_vol_6h(h6["close"], window=min(20, len(h6) - 1))

        # 6) Position sizing
        base = self._position_size(rv)

        # 7) Macro overlay
        macro_info = self._macro_info(daily_df, ts)
        macro_mult = macro_info["mult"]
        macro_state: MacroState = macro_info["state"]
        macro_score_raw = macro_info["score_raw"]
        macro_score = macro_info["score"]
        macro_components = macro_info["components"]

        target_when_long = max(0.0, min(
            self.cfg.adaptive6h_max_position_fraction,
            base * macro_mult,
        ))

        # 8) Update trailing stop on new 6H bar if in position
        if is_new_6h_bar and self.state.in_position and atr_val is not None and atr_val > 0:
            self._update_trailing_stop(float(last6["close"]), atr_val, atr_mult_val)

        # 9) Stop breach check using current hourly bar
        hour = hourly_df.iloc[hourly_idx]
        hourly_low = float(hour["low"])
        stop_breached = (
            self.state.in_position
            and self.state.trailing_stop is not None
            and hourly_low <= self.state.trailing_stop
        )

        # 10) Entry/exit logic
        if self.state.in_position:
            if stop_breached:
                self._exit_position()
                target = 0.0
            else:
                target = target_when_long
        else:
            if (
                mom is not None
                and mom > theta
                and atr_val is not None
                and atr_val > 0
            ):
                self._enter_position(
                    entry_price=float(last6["close"]),
                    atr_val=atr_val,
                    atr_mult=atr_mult_val,
                )
                target = target_when_long
            else:
                target = 0.0

        # 11) Save last6 timestamp
        if is_new_6h_bar:
            self.state.last_6h_bar_end = last6_end

        # 12) FRED features from daily (if available)
        daily_closed = self._closed_daily_cached(daily_df, ts) if daily_df is not None and not daily_df.empty else pd.DataFrame()
        fred_risk_off_score = float(macro_components.get("fred_risk_off_score", 0.0))
        fred_penalty_multiplier = float(macro_components.get("fred_penalty_multiplier", 1.0))

        # 13) Build metadata (consistent with other strategies)
        macro_on = macro_mult > 0
        macro_reason = f"adaptive_6h_{macro_state.value.lower()}"

        metadata: Dict[str, Any] = {
            "realized_vol": float(rv) if rv is not None else 0.0,
            "base_fraction": float(base),
            "macro_score": float(macro_score),
            "macro_score_raw": float(macro_score_raw),
            "macro_score_after_fred": float(macro_score),
            "fred_risk_off_score": fred_risk_off_score,
            "fred_penalty_multiplier": fred_penalty_multiplier,
            "fred_comp_vix_z": float(macro_components.get("fred_VIXCLS_z_level", np.nan)),
            "fred_comp_hy_oas_z": float(macro_components.get("fred_BAMLH0A0HYM2_z_level", np.nan)),
            "fred_comp_stlfsi_z": float(macro_components.get("fred_STLFSI4_z_level", np.nan)),
            "fred_comp_nfci_z": float(macro_components.get("fred_NFCI_z_level", np.nan)),
            "fred_vix_level": self._latest_daily_feature(daily_closed, "fred_VIXCLS_level"),
            "fred_hy_oas_level": self._latest_daily_feature(daily_closed, "fred_BAMLH0A0HYM2_level"),
            "fred_stlfsi_level": self._latest_daily_feature(daily_closed, "fred_STLFSI4_level"),
            "fred_nfci_level": self._latest_daily_feature(daily_closed, "fred_NFCI_level"),
            "macro_state": macro_state.value,
            "macro_multiplier": float(macro_mult),
            "macro_mult": float(macro_mult),
            "macro_reason": macro_reason,
            "micro_regime": RegimeState.NEUTRAL.value,
            "base_target": float(base),
            "current_position_fraction": float(current_exposure),
            "macro_mode": "adaptive_trend_6h",
            "macro_components": str(macro_components),
            # Strategy-specific diagnostics
            "did_reopt": int(did_reopt),
            "params_L": params.get("L"),
            "params_theta": params.get("theta"),
            "params_atr_window": params.get("atr_window"),
            "params_atr_mult": params.get("atr_mult"),
            "mom": float(mom) if mom is not None else 0.0,
            "atr": float(atr_val) if atr_val is not None else 0.0,
            "rv6h": float(rv) if rv is not None else 0.0,
            "trailing_stop": float(self.state.trailing_stop) if self.state.trailing_stop is not None else 0.0,
            "target_when_long": float(target_when_long),
            "in_position": int(self.state.in_position),
            "stop_breached": int(stop_breached),
            "is_new_6h_bar": int(is_new_6h_bar),
            "last6h_end": str(last6_end),
            # Compat fields for engine recording
            "trend_boost_active": 0,
            "boost_multiplier_applied": 1.0,
        }

        return RegimeDecisionBundle(
            macro_risk_on=macro_on,
            macro_reason=macro_reason,
            micro_regime=RegimeState.NEUTRAL,
            micro_reason="adaptive_6h_momentum",
            strategy_name="adaptive_trend_6h_v1",
            base_target=float(base),
            regime_multiplier=float(macro_mult),
            regime_target=float(target_when_long),
            final_target=float(target),
            metadata=metadata,
        )

    def _empty_bundle(self, ts: pd.Timestamp, reason: str) -> RegimeDecisionBundle:
        return RegimeDecisionBundle(
            macro_risk_on=False,
            macro_reason=reason,
            micro_regime=RegimeState.NEUTRAL,
            micro_reason=reason,
            strategy_name="adaptive_trend_6h_v1",
            base_target=0.0,
            regime_multiplier=0.0,
            regime_target=0.0,
            final_target=0.0,
            metadata={"timestamp": str(ts), "reason": reason},
        )
