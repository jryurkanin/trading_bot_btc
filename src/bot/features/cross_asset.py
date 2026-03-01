"""Cross-asset signal features (8.4).

Planned features:
- BTC vs ETH relative strength / beta decomposition
- BTC vs Gold correlation regime (risk-on vs risk-off indicator)
- DXY (Dollar Index) momentum as headwind/tailwind signal
- BTC vs equity indices (SPX, NDX) rolling correlation
- Crypto-specific: BTC dominance rate momentum

Data sources: FRED for macro (DXY, Gold), exchange APIs for crypto pairs.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class CrossAssetConfig:
    enabled: bool = False
    pairs: list[str] | None = None  # e.g. ["ETH-USD", "GLD", "DXY"]
    correlation_window: int = 30  # days for rolling correlation


def build_cross_asset_features(
    daily_df: pd.DataFrame,
    config: CrossAssetConfig,
) -> pd.DataFrame:
    """Compute cross-asset correlation and relative strength features.

    Returns daily_df with additional columns prefixed ``xasset_``.
    Currently a no-op placeholder.
    """
    if not config.enabled:
        return daily_df
    # TODO: fetch cross-asset data, compute rolling correlations, relative strength
    return daily_df
