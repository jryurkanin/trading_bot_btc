"""Orderbook microstructure features (8.2).

Planned features:
- Bid/ask imbalance ratio at configurable depth levels (e.g. top 5, 10, 20 levels)
- Volume-weighted mid price vs simple mid price divergence
- Order flow toxicity (VPIN - Volume-synchronized Probability of Informed Trading)
- Large order detection / iceberg detection heuristics
- Spread dynamics (rolling bid-ask spread percentile)

Data sources: Coinbase L2/L3 orderbook websocket, or exchange REST snapshots.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class OrderbookConfig:
    enabled: bool = False
    depth_levels: int = 10
    snapshot_interval_seconds: int = 60
    imbalance_windows: list[int] | None = None  # rolling windows for imbalance metrics


def compute_orderbook_features(
    orderbook_snapshots: pd.DataFrame,
    config: OrderbookConfig,
) -> pd.DataFrame:
    """Compute orderbook microstructure features from L2 snapshots.

    Expected columns in orderbook_snapshots:
    - timestamp, bid_prices, bid_sizes, ask_prices, ask_sizes

    Returns DataFrame with columns prefixed ``ob_``.
    Currently a no-op placeholder.
    """
    if not config.enabled:
        return pd.DataFrame()
    # TODO: implement imbalance ratio, VPIN, spread dynamics
    return pd.DataFrame()
