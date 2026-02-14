from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import pandas as pd

from ..config import RegimeConfig
from ..features import indicators
from ..features.regime import HMMRegimeSwitcher, RegimeState, RuleBasedRegimeSwitcher, compute_adx, compute_chop
from .sub_strategies.mean_reversion_bb import MeanReversionBBStrategy, RangeStrategyConfig
from .sub_strategies.trend_following_breakout import TrendFollowingBreakoutStrategy, TrendStrategyConfig
from .base import StrategyDecision


@dataclass
class RegimeDecisionBundle:
    macro_risk_on: bool
    macro_reason: str
    micro_regime: RegimeState
    micro_reason: str
    strategy_name: str
    base_target: float
    regime_multiplier: float
    regime_target: float
    final_target: float
    metadata: Dict[str, float | str | int]


class RegimeSwitchingOrchestrator:
    """Main strategy orchestrator."""

    REGIME_MULTIPLIERS = {
        RegimeState.TREND: 1.0,
        RegimeState.RANGE: 0.75,
        RegimeState.NEUTRAL: 0.5,
        RegimeState.HIGH_VOL: 0.0,
    }

    def __init__(self, cfg: RegimeConfig):
        self.cfg = cfg
        self.rule_switcher = RuleBasedRegimeSwitcher(
            adx_trend=cfg.adx_trend_threshold,
            adx_range=cfg.adx_range_threshold,
            chop_trend=cfg.chop_trend_threshold,
            chop_range=cfg.chop_range_threshold,
            confirmation_bars=cfg.regime_confirmation_bars,
            min_duration_hours=cfg.min_regime_duration_hours,
        )
        self.hmm_switcher = HMMRegimeSwitcher(
            enabled=cfg.hmm_regime_enabled,
            n_states=cfg.hmm_n_states,
            window=cfg.hmm_window_hours,
        )
        self.range_strategy = MeanReversionBBStrategy(
            RangeStrategyConfig(
                bb_window=cfg.bb_window,
                bb_stdev=cfg.bb_stdev,
                tranche_size=cfg.range_tranche_size,
                max_exposure=cfg.range_max_exposure,
                min_time_between_trades_hours=cfg.range_min_time_between_trades_hours,
                max_trades_per_day=cfg.range_max_trades_per_day,
            )
        )
        self.trend_strategy = TrendFollowingBreakoutStrategy(
            TrendStrategyConfig(
                mode=cfg.trend_mode,
                donchian_window=cfg.donchian_window,
                ema_fast=cfg.ema_fast,
                ema_slow=cfg.ema_slow,
                atr_window=cfg.atr_window,
                atr_mult=cfg.atr_mult,
                trend_exposure_cap=cfg.trend_exposure_cap,
                vol_target_multiplier=cfg.vol_target_multiplier,
            )
        )

    def reset(self):
        self.rule_switcher.reset()
        self.range_strategy.reset()
        self.trend_strategy.reset()

    def _macro_risk(self, daily_df: pd.DataFrame) -> Tuple[bool, str]:
        if len(daily_df) < max(self.cfg.daily_trend_window, self.cfg.daily_momentum_window + 1):
            return False, "insufficient_daily_history"

        close = daily_df["close"]
        sma = indicators.sma(close, self.cfg.daily_trend_window).iloc[-1]

        base = close.shift(self.cfg.daily_momentum_window).iloc[-1]
        ret = (close.iloc[-1] / base) - 1.0 if pd.notna(base) and float(base) != 0 else 0.0
        mom_quant = indicators.percentile(close.pct_change(), self.cfg.daily_momentum_window, self.cfg.daily_momentum_quantile).iloc[-1]
        mom_quant = float(mom_quant) if pd.notna(mom_quant) else 0.0

        if pd.isna(sma) or close.iloc[-1] <= sma:
            return False, "daily_trend_below_sma"
        if not (ret > 0 or ret > mom_quant):
            return False, "daily_momentum_not_ok"
        return True, "risk_on"

    def _realized_vol(self, hourly: pd.DataFrame) -> pd.Series:
        return indicators.realized_vol(hourly["close"].pct_change(), self.cfg.realized_vol_window)

    def _micro_regime(self, hourly: pd.DataFrame, idx: int) -> RegimeState:
        if idx < 2 or hourly.empty:
            return RegimeState.NEUTRAL

        adx = compute_adx(hourly["high"], hourly["low"], hourly["close"], self.cfg.adx_window)
        chop = compute_chop(hourly["high"], hourly["low"], hourly["close"], self.cfg.chop_window)
        rv = self._realized_vol(hourly)

        if self.cfg.hmm_regime_enabled and len(hourly) >= self.cfg.hmm_window_hours:
            start = max(0, idx - self.cfg.hmm_window_hours + 1)
            feats = pd.concat(
                [
                    hourly[["close", "high", "low", "volume"]].pct_change().fillna(0).iloc[start : idx + 1],
                    rv.iloc[start : idx + 1],
                ],
                axis=1,
            ).fillna(0)
            if len(feats) >= 20:
                self.hmm_switcher.fit(feats.tail(self.cfg.hmm_window_hours).values)
                return self.hmm_switcher.predict_one(feats.iloc[-1].values)

        lookback = max(24, self.cfg.vol_lookback_days * 24)
        vol_thr_series = rv.rolling(lookback, min_periods=max(30, lookback // 4)).quantile(self.cfg.vol_high_threshold_quantile)
        vol_thr_val = vol_thr_series.iloc[idx] if idx < len(vol_thr_series) else None
        vol_thr = float(vol_thr_val) if vol_thr_val is not None and pd.notna(vol_thr_val) else float("inf")
        rv_val = float(rv.iloc[idx]) if idx < len(rv) and pd.notna(rv.iloc[idx]) else 0.0
        return self.rule_switcher.step(float(adx.iloc[idx]), float(chop.iloc[idx]), bool(rv_val > vol_thr))

    def compute_target_position(
        self,
        timestamp: pd.Timestamp,
        hourly_df: pd.DataFrame,
        daily_df: pd.DataFrame,
        current_exposure: float,
    ) -> RegimeDecisionBundle:
        if hourly_df.empty:
            return RegimeDecisionBundle(
                macro_risk_on=False,
                macro_reason="no_hourly_data",
                micro_regime=RegimeState.NEUTRAL,
                micro_reason="no_data",
                strategy_name="none",
                base_target=0.0,
                regime_multiplier=0.0,
                regime_target=0.0,
                final_target=0.0,
                metadata={"timestamp": str(timestamp)},
            )

        macro_on, macro_reason = self._macro_risk(daily_df)
        micro_regime = self._micro_regime(hourly_df, len(hourly_df) - 1)

        rv_last = self._realized_vol(hourly_df).iloc[-1]
        realized_vol = float(rv_last) if rv_last is not None and pd.notna(rv_last) else 0.0
        if realized_vol <= 0 or pd.isna(realized_vol):
            realized_vol = 0.0
            base_target = 0.0
        else:
            base_target = min(self.cfg.target_ann_vol / realized_vol, self.cfg.max_position_fraction)

        if not macro_on:
            return RegimeDecisionBundle(
                macro_risk_on=False,
                macro_reason=macro_reason,
                micro_regime=micro_regime,
                micro_reason="macro_off",
                strategy_name="flat",
                base_target=base_target,
                regime_multiplier=0.0,
                regime_target=0.0,
                final_target=0.0,
                metadata={"realized_vol": realized_vol},
            )

        strategy_ratio = 0.0
        strategy_name = ""
        metadata: Dict[str, float | str | int] = {"realized_vol": realized_vol, "base_target": base_target}

        if micro_regime == RegimeState.TREND:
            strategy_name = self.trend_strategy.name
            strategy_ratio = self.trend_strategy.compute_target(hourly_df, current_exposure, timestamp)
            metadata.update(self.trend_strategy.signal_reason(hourly_df, current_exposure, timestamp))
        elif micro_regime == RegimeState.RANGE:
            strategy_name = self.range_strategy.name
            prev_row = hourly_df.iloc[-2] if len(hourly_df) >= 2 else None
            strategy_ratio = self.range_strategy.compute_target(
                hourly_df.iloc[-1], prev_row, current_exposure, timestamp, hourly_df["close"]
            )
            metadata.update(self.range_strategy.signal_reason(hourly_df.iloc[-1], prev_row, current_exposure, timestamp, hourly_df["close"]))
        elif micro_regime == RegimeState.HIGH_VOL:
            strategy_name = "high_vol_guard"
            if self.cfg.high_vol_cap <= 0:
                strategy_ratio = 0.0
            else:
                strategy_ratio = 1.0
        else:
            strategy_name = "neutral"
            strategy_ratio = min(1.0, current_exposure / base_target) if base_target > 0 else 0.0

        regime_mult = self.REGIME_MULTIPLIERS.get(micro_regime, 0.5)
        if micro_regime == RegimeState.HIGH_VOL and self.cfg.high_vol_cap > 0:
            regime_mult = self.cfg.high_vol_cap

        regime_target = base_target * regime_mult
        final_target = regime_target * strategy_ratio
        final_target = max(0.0, min(self.cfg.max_position_fraction, float(final_target)))

        return RegimeDecisionBundle(
            macro_risk_on=True,
            macro_reason="risk_on",
            micro_regime=micro_regime,
            micro_reason="computed",
            strategy_name=strategy_name,
            base_target=base_target,
            regime_multiplier=regime_mult,
            regime_target=regime_target,
            final_target=final_target,
            metadata=metadata,
        )

    def to_decision(self, bundle: RegimeDecisionBundle, timestamp: pd.Timestamp, current_exposure: float) -> StrategyDecision:
        return StrategyDecision(
            timestamp=timestamp,
            target_fraction=float(bundle.final_target),
            regime=bundle.micro_regime.value,
            sub_regime=bundle.strategy_name,
            metadata={
                **bundle.metadata,
                "macro_risk_on": bundle.macro_risk_on,
                "macro_reason": bundle.macro_reason,
                "micro_reason": bundle.micro_reason,
                "base_target": bundle.base_target,
                "regime_multiplier": bundle.regime_multiplier,
                "regime_target": bundle.regime_target,
                "prev_exposure": current_exposure,
            },
        )
