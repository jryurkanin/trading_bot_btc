"""Volatility surface features (8.5).

Planned features:
- Implied vol term structure slope (front month vs back month)
- Put-call skew (25-delta risk reversal)
- ATM implied vol vs realized vol spread (vol risk premium)
- IV percentile rank (current IV vs historical range)
- Variance risk premium signal for position sizing

Data sources: Deribit options API, or aggregated crypto options data.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class VolSurfaceConfig:
    enabled: bool = False
    provider: str = "deribit"
    iv_percentile_window: int = 90  # days for IV percentile rank
    skew_delta: float = 0.25  # delta for risk reversal calculation


def build_vol_surface_features(
    daily_df: pd.DataFrame,
    config: VolSurfaceConfig,
) -> pd.DataFrame:
    """Compute volatility surface features.

    Returns daily_df with additional columns prefixed ``volsurf_``.
    Currently a no-op placeholder.
    """
    if not config.enabled:
        return daily_df
    # TODO: fetch options data, compute IV term structure, skew, VRP
    return daily_df
