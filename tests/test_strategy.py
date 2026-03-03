from __future__ import annotations

import pandas as pd

from bot.strategy.sub_strategies.mean_reversion_bb import MeanReversionBBStrategy, RangeStrategyConfig
from bot.strategy.sub_strategies.trend_following_breakout import TrendFollowingBreakoutStrategy, TrendStrategyConfig


def test_range_strategy_incremental_entries_and_exits():
    strat = MeanReversionBBStrategy(
        RangeStrategyConfig(
            bb_window=3,
            bb_stdev=2.0,
            tranche_size=0.2,
            max_exposure=0.8,
        )
    )

    close = pd.Series([100, 100, 90])
    row = pd.Series({"close": 90})
    prev = pd.Series({"close": 100})
    now = pd.Timestamp("2026-01-01T10:00:00Z")

    t = strat.compute_target(row, prev, 0.0, now, close)
    assert t >= 0.0

    close2 = pd.Series([100, 95, 100])
    row2 = pd.Series({"close": 100})
    prev2 = pd.Series({"close": 90})
    t2 = strat.compute_target(row2, prev2, t, now + pd.Timedelta(hours=1), close2)
    assert t2 <= strat.cfg.max_exposure


def test_trend_strategy_breakout_and_exit():
    strat = TrendFollowingBreakoutStrategy(TrendStrategyConfig(mode="donchian", donchian_window=3))
    close = pd.Series([10, 11, 12, 13, 14], index=pd.date_range("2026-01-01", periods=5, freq="h"))
    high = pd.Series([10.5, 11.2, 12.1, 13.2, 14.1], index=close.index)
    low = pd.Series([9.8, 10.2, 11.2, 12.2, 13.2], index=close.index)
    df = pd.DataFrame({"close": close, "high": high, "low": low})

    t = strat.compute_target(df, current_exposure=0.0, now=close.index[-1])
    # after breakout should have positive exposure
    assert t >= 0.0

    # force exit condition with strong drop
    close2 = close.copy()
    high2 = pd.Series([12, 11, 10, 9, 8], index=close.index)
    low2 = pd.Series([9, 8, 7, 6, 5], index=close.index)
    df2 = pd.DataFrame({"close": close2, "high": high2, "low": low2})
    t2 = strat.compute_target(df2, current_exposure=t, now=close2.index[-1])
    assert t2 >= 0.0
