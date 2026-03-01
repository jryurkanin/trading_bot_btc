"""Enhanced mean reversion strategies (8.7).

Planned enhancements over mean_reversion_bb.py:
- Multi-timeframe Bollinger: combine hourly + 4h + daily BB for confluence
- RSI divergence confirmation (price makes new low but RSI doesn't)
- Volume profile integration: trade only at high-volume price nodes
- Dynamic BB width: adjust stdev multiplier based on regime volatility
- Limit order placement at BB bands instead of market orders
- Fade threshold: only enter when price has mean-reverted X% from band touch
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd

from ...features.indicators import bollinger_bands


@dataclass
class EnhancedMeanReversionConfig:
    enabled: bool = False
    bb_window: int = 20
    bb_stdev: float = 2.0
    use_rsi_divergence: bool = True
    rsi_window: int = 14
    rsi_oversold: float = 30.0
    rsi_overbought: float = 70.0
    multi_timeframe: bool = False
    fade_pct: float = 0.3  # enter after price fades 30% from band touch
    tranche_size: float = 0.25
    max_exposure: float = 0.75


class EnhancedMeanReversionStrategy:
    """Enhanced mean reversion with RSI divergence and multi-timeframe confluence.

    Currently a placeholder wrapping the base Bollinger Band logic.
    """

    name = "mean_reversion_enhanced"

    def __init__(self, cfg: Optional[EnhancedMeanReversionConfig] = None):
        self.cfg = cfg or EnhancedMeanReversionConfig()

    def compute_signal(
        self,
        hourly_df: pd.DataFrame,
        idx: int,
    ) -> float:
        """Return target exposure in [-max_exposure, max_exposure].

        Positive = long mean-reversion, negative = short mean-reversion.
        Currently returns 0.0 (no signal).
        """
        if not self.cfg.enabled:
            return 0.0
        # TODO: implement RSI divergence, multi-TF BB confluence, fade logic
        return 0.0
