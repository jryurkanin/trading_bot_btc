from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from . import indicators


class RegimeState(str, Enum):
    TREND = "TREND"
    RANGE = "RANGE"
    NEUTRAL = "NEUTRAL"
    HIGH_VOL = "HIGH_VOL"


def compute_adx_di(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> tuple[pd.Series, pd.Series, pd.Series]:
    prev_high = high.shift(1)
    prev_low = low.shift(1)

    up_move = high - prev_high
    down_move = prev_low - low

    plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=high.index)
    minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=high.index)

    tr = indicators.true_range(high, low, close)
    atr = tr.rolling(window=window, min_periods=window).mean()

    plus_di = 100 * (plus_dm.rolling(window=window, min_periods=window).sum() / atr.replace(0, np.nan))
    minus_di = 100 * (minus_dm.rolling(window=window, min_periods=window).sum() / atr.replace(0, np.nan))

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx = dx.rolling(window=window, min_periods=window).mean()
    return adx.fillna(0.0), plus_di.fillna(0.0), minus_di.fillna(0.0)


def compute_adx(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    adx, _, _ = compute_adx_di(high, low, close, window=window)
    return adx


def compute_chop(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    high_window = high.rolling(window=window, min_periods=window)
    low_window = low.rolling(window=window, min_periods=window)
    tr_sum = indicators.true_range(high, low, close).rolling(window=window, min_periods=window).sum()
    numerator = tr_sum
    denominator = (high_window.max() - low_window.min()).replace(0, pd.NA)
    with np.errstate(invalid="ignore", divide="ignore"):
        chop = 100 * np.log10(numerator / denominator) / np.log10(window)
    return chop.fillna(0.0)


def compute_volatility_state(realized_vol: pd.Series, rolling_window: int = 365 * 24, quantile: float = 0.90) -> pd.Series:
    threshold = realized_vol.rolling(window=rolling_window, min_periods=max(20, int(rolling_window * 0.5))).quantile(quantile)
    return pd.Series(np.where(realized_vol > threshold, 1, 0), index=realized_vol.index)


@dataclass
class RegimeDecision:
    micro: RegimeState
    rule_score: float
    adx: float
    chop: float
    vol: float


class RuleBasedRegimeSwitcher:
    def __init__(
        self,
        adx_trend: float = 25.0,
        adx_range: float = 20.0,
        chop_trend: float = 38.2,
        chop_range: float = 61.8,
        confirmation_bars: int = 3,
        min_duration_hours: int = 6,
    ) -> None:
        self.adx_trend = adx_trend
        self.adx_range = adx_range
        self.chop_trend = chop_trend
        self.chop_range = chop_range
        self.confirmation_bars = confirmation_bars
        self.min_duration_hours = min_duration_hours
        self._candidate_regime = RegimeState.NEUTRAL
        self._confirmed_regime = RegimeState.NEUTRAL
        self._candidate_count = 0
        self._regime_age = 0

    def reset(self):
        self._candidate_regime = RegimeState.NEUTRAL
        self._confirmed_regime = RegimeState.NEUTRAL
        self._candidate_count = 0
        self._regime_age = 0

    def _vote(self, adx: float, chop: float, is_high_vol: bool) -> RegimeState:
        if is_high_vol:
            return RegimeState.HIGH_VOL
        if adx >= self.adx_trend or chop <= self.chop_trend:
            return RegimeState.TREND
        if adx <= self.adx_range or chop >= self.chop_range:
            return RegimeState.RANGE
        return RegimeState.NEUTRAL

    def step(self, adx: float, chop: float, is_high_vol: bool) -> RegimeState:
        vote = self._vote(adx, chop, is_high_vol)
        if vote == self._candidate_regime:
            self._candidate_count += 1
        else:
            self._candidate_regime = vote
            self._candidate_count = 1

        if self._candidate_count >= self.confirmation_bars:
            if vote != self._confirmed_regime and self._regime_age >= self.min_duration_hours:
                self._confirmed_regime = vote
                self._regime_age = 0
            elif vote == self._confirmed_regime:
                self._regime_age = 0

        self._regime_age += 1
        return self._confirmed_regime

    def compute_series(self, df: pd.DataFrame, vol_threshold: float) -> pd.Series:
        req = {"high", "low", "close", "realized_vol"}
        if not req.issubset(df.columns):
            raise ValueError(f"df must include columns {req}")

        adx = compute_adx(df["high"], df["low"], df["close"], 14)
        chop = compute_chop(df["high"], df["low"], df["close"], 14)
        out = []
        self.reset()
        for i in range(len(df)):
            is_high_vol = float(df["realized_vol"].iloc[i]) > float(vol_threshold)
            regime = self.step(float(adx.iloc[i]), float(chop.iloc[i]), bool(is_high_vol))
            out.append(regime.value)
        return pd.Series(out, index=df.index)


class HMMRegimeSwitcher:
    def __init__(self, enabled: bool = False, n_states: int = 3, window: int = 1000):
        self.enabled = enabled
        self.n_states = n_states
        self.window = window
        self.model = None
        self._state_map: dict[int, RegimeState] = {}

    def fit(self, X: np.ndarray) -> None:
        if not self.enabled:
            return
        try:
            from hmmlearn.hmm import GaussianHMM
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("hmmlearn is required when hmm_regime_enabled=True") from exc

        self.model = GaussianHMM(
            n_components=self.n_states,
            covariance_type="full",
            n_iter=100,
            random_state=42,
        )
        self.model.fit(X)
        states = self.model.predict(X)
        # map state by average volatility level
        means = self.model.means_.reshape(self.n_states, -1)
        order = np.argsort(means[:, 0])
        mapping = {}
        if self.n_states >= 3:
            mapping[int(order[-1])] = RegimeState.HIGH_VOL
            mapping[int(order[0])] = RegimeState.RANGE
            mapping[int(order[1])] = RegimeState.TREND if self.n_states == 3 else RegimeState.NEUTRAL
        else:
            for o, state in zip(order, [RegimeState.RANGE, RegimeState.TREND, RegimeState.HIGH_VOL]):
                mapping[int(o)] = state
        self._state_map = mapping

    def predict_one(self, features: np.ndarray) -> RegimeState:
        if not self.enabled or self.model is None:
            return RegimeState.NEUTRAL
        state = int(self.model.predict(features.reshape(1, -1))[0])
        return self._state_map.get(state, RegimeState.NEUTRAL)
