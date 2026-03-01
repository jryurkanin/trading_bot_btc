"""On-chain analytics feature pipeline (8.1).

Planned features:
- Exchange flow metrics (net inflow/outflow, exchange reserve changes)
- Active address momentum (30d/90d growth rates)
- MVRV Z-score (market value to realized value ratio)
- Spent Output Profit Ratio (SOPR) for profit-taking detection
- Miner revenue / hash rate momentum for supply-side pressure

Data sources: Glassnode, CryptoQuant, or similar on-chain data providers.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass
class OnchainConfig:
    enabled: bool = False
    provider: str = "glassnode"
    api_key: str = ""
    metrics: list[str] | None = None  # e.g. ["exchange_netflow", "mvrv_z", "sopr"]
    lookback_days: int = 365


def build_onchain_features(
    daily_df: pd.DataFrame,
    config: OnchainConfig,
) -> pd.DataFrame:
    """Compute on-chain features and merge into daily_df.

    Returns daily_df with additional columns prefixed ``onchain_``.
    Currently a no-op placeholder.
    """
    if not config.enabled:
        return daily_df
    # TODO: implement provider client, fetch metrics, align to daily timestamps
    return daily_df
