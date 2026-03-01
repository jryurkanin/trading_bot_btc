from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bot.backtest.metrics import bootstrap_sharpe_confidence, compute_sharpe


def _make_returns(mean: float, std: float, n: int, seed: int = 0) -> pd.Series:
    rng = np.random.default_rng(seed)
    return pd.Series(rng.normal(mean, std, n))


class TestSharpeBootstrapPositiveMean:
    """Positive-mean returns should yield a significantly positive Sharpe."""

    def setup_method(self):
        self.returns = _make_returns(mean=0.001, std=0.01, n=5000, seed=7)
        self.result = bootstrap_sharpe_confidence(
            self.returns, n_bootstrap=5000, seed=42,
        )

    def test_ci_lower_bound_positive(self):
        lo, hi = self.result.confidence_intervals["95%"]
        assert lo > 0, f"95% CI lower bound should be > 0, got {lo}"

    def test_p_value_significant(self):
        assert self.result.p_value_sharpe_leq_0 < 0.05

    def test_point_estimate_positive(self):
        assert self.result.point_estimate > 0


class TestSharpeBootstrapZeroMean:
    """Zero-mean symmetric returns → p-value near 0.5, CI straddles 0."""

    def setup_method(self):
        self.returns = _make_returns(mean=0.0, std=0.01, n=5000, seed=12)
        self.result = bootstrap_sharpe_confidence(
            self.returns, n_bootstrap=5000, seed=42,
        )

    def test_p_value_near_half(self):
        assert 0.2 < self.result.p_value_sharpe_leq_0 < 0.8

    def test_ci_straddles_zero(self):
        lo, hi = self.result.confidence_intervals["95%"]
        assert lo < 0 < hi


class TestSharpeBootstrapReproducibility:
    """Deterministic seed should give identical results."""

    def test_same_seed_same_result(self):
        ret = _make_returns(mean=0.0005, std=0.01, n=2000, seed=5)
        r1 = bootstrap_sharpe_confidence(ret, n_bootstrap=1000, seed=99)
        r2 = bootstrap_sharpe_confidence(ret, n_bootstrap=1000, seed=99)
        assert r1.bootstrap_mean == r2.bootstrap_mean
        assert r1.p_value_sharpe_leq_0 == r2.p_value_sharpe_leq_0
        assert r1.confidence_intervals == r2.confidence_intervals


class TestSharpeBootstrapPointEstimate:
    """Point estimate should match compute_sharpe exactly."""

    def test_matches_compute_sharpe(self):
        ret = _make_returns(mean=0.0003, std=0.01, n=3000, seed=3)
        result = bootstrap_sharpe_confidence(ret, n_bootstrap=100, seed=1)
        expected = compute_sharpe(ret)
        assert result.point_estimate == pytest.approx(expected, rel=1e-10)


class TestSharpeBootstrapCIOrdering:
    """90% CI should be inside 95% which should be inside 99%."""

    def test_nested_intervals(self):
        ret = _make_returns(mean=0.0005, std=0.01, n=3000, seed=8)
        result = bootstrap_sharpe_confidence(
            ret, n_bootstrap=5000, seed=42,
            confidence_levels=(0.90, 0.95, 0.99),
        )
        lo90, hi90 = result.confidence_intervals["90%"]
        lo95, hi95 = result.confidence_intervals["95%"]
        lo99, hi99 = result.confidence_intervals["99%"]

        assert lo99 <= lo95 <= lo90
        assert hi90 <= hi95 <= hi99

    def test_hac_ci_ordering(self):
        ret = _make_returns(mean=0.0005, std=0.01, n=3000, seed=8)
        result = bootstrap_sharpe_confidence(
            ret, n_bootstrap=100, seed=42,
            confidence_levels=(0.90, 0.95, 0.99),
        )
        lo90, hi90 = result.hac_confidence_intervals["90%"]
        lo95, hi95 = result.hac_confidence_intervals["95%"]
        lo99, hi99 = result.hac_confidence_intervals["99%"]

        assert lo99 <= lo95 <= lo90
        assert hi90 <= hi95 <= hi99


def test_hac_se_is_positive():
    ret = _make_returns(mean=0.0005, std=0.01, n=2000, seed=1)
    result = bootstrap_sharpe_confidence(ret, n_bootstrap=100, seed=1)
    assert result.hac_std_error > 0


def test_n_simulations_stored():
    ret = _make_returns(mean=0.0, std=0.01, n=500, seed=0)
    result = bootstrap_sharpe_confidence(ret, n_bootstrap=200, seed=1)
    assert result.n_simulations == 200
