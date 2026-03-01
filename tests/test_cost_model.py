"""Tests for CostModel including new funding rate (Section 7)."""
from __future__ import annotations

import pytest

from bot.backtest.cost_model import CostModel


def test_fee_rate_maker():
    cm = CostModel(maker_fee_rate=0.001, taker_fee_rate=0.0025)
    assert cm.fee_rate(is_maker=True) == 0.001


def test_fee_rate_taker():
    cm = CostModel(maker_fee_rate=0.001, taker_fee_rate=0.0025)
    assert cm.fee_rate(is_maker=False) == 0.0025


def test_fee_calculation():
    cm = CostModel(maker_fee_rate=0.001, taker_fee_rate=0.0025)
    assert cm.fee(10000.0, is_maker=True) == pytest.approx(10.0)
    assert cm.fee(10000.0, is_maker=False) == pytest.approx(25.0)


def test_fee_negative_notional():
    cm = CostModel(maker_fee_rate=0.001, taker_fee_rate=0.0025)
    assert cm.fee(-500.0, is_maker=True) == 0.0


def test_slippage_buy():
    s = CostModel.slippage_cost("BUY", 101.0, 100.0, 1.0)
    assert s == pytest.approx(1.0)


def test_slippage_sell():
    s = CostModel.slippage_cost("SELL", 99.0, 100.0, 1.0)
    assert s == pytest.approx(1.0)


def test_slippage_bps_buy():
    bps = CostModel.slippage_bps("BUY", 100.10, 100.0)
    assert bps == pytest.approx(10.0, rel=0.01)


def test_funding_cost_zero_default():
    cm = CostModel(maker_fee_rate=0.001, taker_fee_rate=0.0025)
    assert cm.funding_cost_per_bar(10000.0) == 0.0


def test_funding_cost_annual_5_pct():
    cm = CostModel(maker_fee_rate=0.001, taker_fee_rate=0.0025, funding_rate_annual=0.05)
    cost = cm.funding_cost_per_bar(100000.0, bars_per_year=8760)
    expected = 100000.0 * 0.05 / 8760
    assert cost == pytest.approx(expected)


def test_funding_cost_negative_position():
    """Funding cost is charged on absolute position value."""
    cm = CostModel(maker_fee_rate=0.001, taker_fee_rate=0.0025, funding_rate_annual=0.10)
    cost = cm.funding_cost_per_bar(-50000.0, bars_per_year=8760)
    assert cost > 0
