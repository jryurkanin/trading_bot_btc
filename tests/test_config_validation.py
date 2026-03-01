"""Tests for config field validators (Section 7)."""
from __future__ import annotations

import pytest

from bot.config import RegimeConfig


def test_adx_threshold_ordering():
    """adx_range_threshold must be < adx_trend_threshold."""
    with pytest.raises(ValueError, match="adx_range_threshold"):
        RegimeConfig(adx_range_threshold=30.0, adx_trend_threshold=25.0)


def test_adx_threshold_equal_rejected():
    """Equal adx thresholds should also be rejected."""
    with pytest.raises(ValueError, match="adx_range_threshold"):
        RegimeConfig(adx_range_threshold=25.0, adx_trend_threshold=25.0)


def test_adx_threshold_valid():
    cfg = RegimeConfig(adx_range_threshold=15.0, adx_trend_threshold=25.0)
    assert cfg.adx_range_threshold == 15.0
    assert cfg.adx_trend_threshold == 25.0


def test_macro_threshold_ordering():
    """macro_exit_threshold must be < macro_enter_threshold."""
    with pytest.raises(ValueError, match="macro_exit_threshold"):
        RegimeConfig(macro_exit_threshold=0.80, macro_enter_threshold=0.75)


def test_target_ann_vol_positive():
    """target_ann_vol must be > 0."""
    with pytest.raises(ValueError, match="target_ann_vol"):
        RegimeConfig(target_ann_vol=0.0)


def test_target_ann_vol_negative_rejected():
    with pytest.raises(ValueError, match="target_ann_vol"):
        RegimeConfig(target_ann_vol=-0.1)


def test_valid_config_defaults():
    """Default values should pass validation."""
    cfg = RegimeConfig()
    assert cfg.target_ann_vol > 0
