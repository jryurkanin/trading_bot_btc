from __future__ import annotations

import pandas as pd

from bot.data.fred_client import FredClient
from bot.features.fred_features import align_fred_series_to_target


def test_align_fred_series_respects_availability_lag_no_lookahead():
    target_ts = pd.Series(
        pd.to_datetime(
            [
                "2024-01-01T12:00:00Z",
                "2024-01-02T00:00:00Z",
                "2024-01-02T12:00:00Z",
                "2024-01-03T00:00:00Z",
            ],
            utc=True,
        )
    )

    observations = pd.DataFrame(
        {
            "observation_date": pd.to_datetime(["2024-01-01", "2024-01-02"], utc=True),
            "value": [1.0, 2.0],
        }
    )

    aligned = align_fred_series_to_target(target_ts, observations, lag_hours=24)

    # Jan-01 observation becomes available at Jan-02 00:00 UTC.
    assert pd.isna(aligned.iloc[0])
    assert float(aligned.iloc[1]) == 1.0
    assert float(aligned.iloc[2]) == 1.0

    # Jan-02 observation becomes available at Jan-03 00:00 UTC.
    assert float(aligned.iloc[3]) == 2.0


def test_fred_cache_key_stable_and_roundtrip_load(tmp_path):
    params_a = {
        "api_key": "test",
        "series_id": "VIXCLS",
        "observation_start": "2020-01-01",
        "observation_end": "2020-01-31",
        "file_type": "json",
        "output_type": 1,
        "units": "lin",
        "aggregation_method": "avg",
    }
    params_b = {
        "series_id": "VIXCLS",
        "observation_end": "2020-01-31",
        "units": "lin",
        "observation_start": "2020-01-01",
        "output_type": 1,
        "api_key": "test",
        "aggregation_method": "avg",
        "file_type": "json",
    }

    assert FredClient._cache_key(params_a) == FredClient._cache_key(params_b)

    client = FredClient(
        api_key="test",
        cache_dir=tmp_path,
        use_stale_cache_for_backtest=True,
    )
    try:
        df = pd.DataFrame(
            {
                "observation_date": pd.to_datetime(["2020-01-01", "2020-01-02"], utc=True),
                "value": [20.0, 21.5],
                "realtime_start": pd.to_datetime(["2020-01-01", "2020-01-02"], utc=True),
                "realtime_end": pd.to_datetime(["2020-01-01", "2020-01-02"], utc=True),
            }
        )

        client._save_cache("VIXCLS", params_a, df)
        loaded = client._load_cache("VIXCLS", params_b)

        assert loaded is not None
        assert list(loaded["value"].astype(float)) == [20.0, 21.5]
        assert "UTC" in str(loaded["observation_date"].dtype)
    finally:
        client.close()
