"""ML-based regime detection (8.3).

Planned approaches:
- Hidden Markov Model (HMM) with Gaussian emissions on returns + vol
- Unsupervised clustering (GMM) on feature vectors to detect regime shifts
- Online changepoint detection (Bayesian online changepoint / PELT)
- Feature importance from gradient-boosted trees for regime classification
- Ensemble: combine ADX/CHOP rule-based regime with ML-detected regimes

Integration: Outputs a regime label compatible with MicroRegime enum,
can be used as an alternative or complement to the rule-based regime.py.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import pandas as pd


class MLRegime(str, Enum):
    BULL_TREND = "bull_trend"
    BEAR_TREND = "bear_trend"
    HIGH_VOL = "high_vol"
    RANGE = "range"
    TRANSITION = "transition"


@dataclass
class MLRegimeConfig:
    enabled: bool = False
    model_type: str = "hmm"  # "hmm", "gmm", "changepoint"
    n_regimes: int = 4
    retrain_interval_days: int = 30
    min_train_samples: int = 500


def detect_ml_regime(
    hourly_df: pd.DataFrame,
    config: MLRegimeConfig,
) -> pd.Series:
    """Detect market regime using ML model.

    Returns Series of MLRegime values aligned to hourly_df index.
    Currently a no-op placeholder returning RANGE for all rows.
    """
    if not config.enabled:
        return pd.Series(MLRegime.RANGE, index=hourly_df.index)
    # TODO: implement HMM/GMM fitting and prediction
    return pd.Series(MLRegime.RANGE, index=hourly_df.index)
