from __future__ import annotations

import pandas as pd

from bot.features.indicators import sma, ema, bollinger_bands, rsi, atr, realized_vol


def test_sma_ema_shapes():
    s = pd.Series([1, 2, 3, 4, 5], dtype=float)
    out = sma(s, 3)
    assert len(out) == 5
    assert out.iloc[0] == 1

    ema_out = ema(s, 3)
    assert len(ema_out) == 5
    assert ema_out.iloc[-1] > ema_out.iloc[0]


def test_bollinger_and_rsi():
    s = pd.Series([1, 2, 3, 2, 1, 2, 3, 4, 5], dtype=float)
    mid, up, lo = bollinger_bands(s, window=3, stdev=2.0)
    assert len(mid) == len(s)
    assert (up >= mid).all()
    assert (mid >= lo).all()

    r = rsi(s, window=3)
    assert len(r) == len(s)
    assert ((r >= 0) & (r <= 100)).all()


def test_atr_and_realized_vol():
    high = pd.Series([10, 11, 12, 13, 14], dtype=float)
    low = pd.Series([9, 10, 11, 12, 13], dtype=float)
    close = pd.Series([9.5, 10.5, 11, 12.5, 13.5], dtype=float)

    a = atr(high, low, close, window=2)
    assert len(a) == 5
    assert not a.isna().all()

    rv = realized_vol(close.pct_change(), window=2)
    assert len(rv) == 5
    assert not rv.isna().all()
