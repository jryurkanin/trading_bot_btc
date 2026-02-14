from __future__ import annotations

import pandas as pd

from bot.features.regime import RuleBasedRegimeSwitcher, RegimeState


def test_regime_anti_chatter():
    # synthetic values where trend appears after confirmation window
    adx = [19, 19, 21, 22, 24, 26, 26, 26]
    chop = [70, 70, 65, 64, 63, 62, 60, 55]
    vol = [1, 1, 1, 1, 1, 1, 1, 1]
    sw = RuleBasedRegimeSwitcher(adx_trend=25, adx_range=20, chop_trend=38.2, chop_range=61.8, confirmation_bars=3, min_duration_hours=1)

    out = []
    for i in range(len(adx)):
        out.append(sw.step(adx[i], chop[i], False))

    # start in neutral
    assert out[0] == RegimeState.NEUTRAL
    # confirm not immediate transition
    assert RegimeState.TREND not in out[:3]
    # after enough evidence can transition
    assert RegimeState.TREND in out
