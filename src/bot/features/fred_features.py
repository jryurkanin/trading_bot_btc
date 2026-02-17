from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from typing import Any
import logging
import math

import numpy as np
import pandas as pd

from ..config import FredConfig
from ..data.fred_client import FredClient

logger = logging.getLogger("trading_bot.features.fred")


@dataclass
class FredFeatureBuildResult:
    daily_features: pd.DataFrame
    report: dict[str, Any]


def _to_utc_ts(series_or_index: pd.Series | pd.Index) -> pd.Series:
    return pd.to_datetime(series_or_index, utc=True, errors="coerce")


def _rolling_z(series: pd.Series, window: int, clip: float) -> pd.Series:
    w = max(10, int(window))
    min_periods = max(5, w // 4)
    mu = series.rolling(w, min_periods=min_periods).mean()
    sigma = series.rolling(w, min_periods=min_periods).std(ddof=0)
    z = (series - mu) / sigma.replace(0.0, np.nan)
    z = z.replace([np.inf, -np.inf], np.nan)
    c = float(max(0.5, clip))
    return z.clip(lower=-c, upper=c)


def align_fred_series_to_target(
    target_ts: pd.Series,
    observations: pd.DataFrame,
    *,
    lag_hours: float,
) -> pd.Series:
    """Align a FRED series to target timestamps with conservative availability lag.

    Uses ``merge_asof`` with ``available_ts <= target_ts`` to enforce no lookahead.
    Merge keys are int64 epoch-ns to avoid datetime unit-mismatch errors
    (e.g. datetime64[s] vs datetime64[us]).
    """
    target_index = pd.Index(_to_utc_ts(target_ts))
    if observations is None or observations.empty:
        return pd.Series(index=target_index, dtype=float)

    t = pd.DataFrame({"target_ts": target_index})
    t = t.dropna(subset=["target_ts"]).sort_values("target_ts").reset_index(drop=True)

    src = observations.copy()
    src["observation_date"] = _to_utc_ts(src["observation_date"])
    src["value"] = pd.to_numeric(src.get("value"), errors="coerce")
    src = src.dropna(subset=["observation_date", "value"]).sort_values("observation_date")
    if src.empty or t.empty:
        return pd.Series(index=target_index, dtype=float)

    src["available_ts"] = src["observation_date"] + pd.to_timedelta(float(lag_hours), unit="h")

    t["merge_key_ns"] = t["target_ts"].astype("int64")
    src["merge_key_ns"] = src["available_ts"].astype("int64")

    aligned = pd.merge_asof(
        t.sort_values("merge_key_ns"),
        src[["merge_key_ns", "value"]].sort_values("merge_key_ns"),
        on="merge_key_ns",
        direction="backward",
    )

    out = aligned.set_index("target_ts")["value"]
    out = out.reindex(target_index)
    out.index = target_index
    return out


def _default_lag_hours_for_frequency(cfg: FredConfig, freq: str) -> float:
    f = str(freq or "daily").lower()
    if f == "weekly":
        return float(cfg.default_availability_lag_hours_weekly)
    if f == "monthly":
        return float(cfg.default_availability_lag_hours_monthly)
    return float(cfg.default_availability_lag_hours_daily)


def _ffill_limit_by_frequency(freq: str) -> int:
    f = str(freq or "daily").lower()
    if f == "weekly":
        return 35
    if f == "monthly":
        return 95
    return 10


def _lookback_by_frequency(cfg: FredConfig, freq: str) -> int:
    f = str(freq or "daily").lower()
    if f == "weekly":
        return int(cfg.weekly_z_lookback)
    if f == "monthly":
        return int(cfg.monthly_z_lookback)
    return int(cfg.daily_z_lookback)


def _component_to_unit_interval(series: pd.Series, clip: float) -> pd.Series:
    c = float(max(0.5, clip))
    out = (series + c) / (2.0 * c)
    return out.clip(lower=0.0, upper=1.0)


def _compose_fred_risk_off_score(df: pd.DataFrame, cfg: FredConfig) -> tuple[pd.Series, dict[str, float]]:
    # Components are oriented so positive values imply higher risk-off pressure.
    components: dict[str, pd.Series] = {}

    if "fred_VIXCLS_z_level" in df.columns:
        components["VIXCLS"] = df["fred_VIXCLS_z_level"]
    if "fred_BAMLH0A0HYM2_z_level" in df.columns:
        components["BAMLH0A0HYM2"] = df["fred_BAMLH0A0HYM2_z_level"]
    if "fred_BAA10Y_z_level" in df.columns:
        components["BAA10Y"] = df["fred_BAA10Y_z_level"]
    if "fred_STLFSI4_z_level" in df.columns:
        components["STLFSI4"] = df["fred_STLFSI4_z_level"]
    if "fred_NFCI_z_level" in df.columns:
        components["NFCI"] = df["fred_NFCI_z_level"]
    if "fred_DTWEXBGS_z_delta_20" in df.columns:
        components["DTWEXBGS"] = df["fred_DTWEXBGS_z_delta_20"]

    # Yield curve inversion proxy (negative slope => risk-off)
    slope = None
    if "fred_T10Y3M_level" in df.columns:
        slope = pd.to_numeric(df["fred_T10Y3M_level"], errors="coerce")
    elif "fred_T10Y2Y_level" in df.columns:
        slope = pd.to_numeric(df["fred_T10Y2Y_level"], errors="coerce")
    if slope is not None:
        inversion = (-slope).clip(lower=0.0)
        components["curve_inversion"] = _rolling_z(inversion, max(60, int(cfg.daily_z_lookback // 2)), cfg.zscore_clip)

    if "fred_WALCL_z_yoy" in df.columns:
        components["WALCL"] = -pd.to_numeric(df["fred_WALCL_z_yoy"], errors="coerce")
    if "fred_M2SL_z_yoy" in df.columns:
        components["M2SL"] = -pd.to_numeric(df["fred_M2SL_z_yoy"], errors="coerce")

    if not components:
        return pd.Series(0.0, index=df.index), {}

    weighted_sum = pd.Series(0.0, index=df.index, dtype=float)
    weight_total = pd.Series(0.0, index=df.index, dtype=float)

    applied_weights: dict[str, float] = {}
    for key, oriented in components.items():
        w = float(cfg.risk_off_weights.get(key, 0.0) or 0.0)
        if abs(w) < 1e-12:
            continue
        oriented = pd.to_numeric(oriented, errors="coerce")
        unit = _component_to_unit_interval(oriented, cfg.zscore_clip)
        valid = oriented.notna().astype(float)
        weighted_sum = weighted_sum + (unit.fillna(0.0) * w)
        weight_total = weight_total + (abs(w) * valid)
        applied_weights[key] = w

    out = weighted_sum / weight_total.replace(0.0, np.nan)
    out = out.fillna(0.0).clip(lower=0.0, upper=1.0)
    return out, applied_weights


def _series_summary(series: pd.Series) -> dict[str, float | int | None]:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return {
            "count": 0,
            "mean": None,
            "std": None,
            "min": None,
            "max": None,
            "last": None,
        }
    return {
        "count": int(len(s)),
        "mean": float(s.mean()),
        "std": float(s.std(ddof=0)) if len(s) > 1 else 0.0,
        "min": float(s.min()),
        "max": float(s.max()),
        "last": float(s.iloc[-1]),
    }


def build_fred_daily_overlay_features(
    daily_df: pd.DataFrame,
    cfg: FredConfig,
) -> FredFeatureBuildResult:
    """Fetch FRED, enforce availability lags, and build macro overlay features."""
    if daily_df is None or daily_df.empty:
        return FredFeatureBuildResult(
            daily_features=daily_df if daily_df is not None else pd.DataFrame(),
            report={
                "enabled": bool(cfg.enabled),
                "series_used": [],
                "series_lags_hours": {},
                "warnings": ["daily_df_empty"],
            },
        )

    if not cfg.enabled:
        return FredFeatureBuildResult(
            daily_features=daily_df.copy(),
            report={
                "enabled": False,
                "series_used": [],
                "series_lags_hours": {},
                "warnings": [],
            },
        )

    ts = _to_utc_ts(daily_df["timestamp"] if "timestamp" in daily_df.columns else daily_df.index)
    ts = pd.Series(ts).dropna().sort_values().reset_index(drop=True)
    if ts.empty:
        return FredFeatureBuildResult(
            daily_features=daily_df.copy(),
            report={
                "enabled": True,
                "series_used": [],
                "series_lags_hours": {},
                "warnings": ["daily_timestamps_empty"],
            },
        )

    api_key = str(cfg.api_key or "").strip()
    if not api_key:
        logger.warning("FRED enabled but api_key missing; skipping FRED overlay")
        return FredFeatureBuildResult(
            daily_features=daily_df.copy(),
            report={
                "enabled": True,
                "series_used": [],
                "series_lags_hours": {},
                "warnings": ["fred_api_key_missing"],
            },
        )

    lookback_days = max(365 * 5, int(cfg.daily_z_lookback) * 3)
    obs_start = (ts.min() - pd.Timedelta(days=lookback_days)).date().isoformat()
    obs_end = (ts.max() + pd.Timedelta(days=3)).date().isoformat()

    client = FredClient(
        api_key=api_key,
        base_url=cfg.api_base_url,
        cache_dir=cfg.cache_dir,
        timeout_seconds=cfg.http_timeout_seconds,
        max_retries=cfg.max_retries,
        backoff_seconds=cfg.backoff_seconds,
        cache_ttl_hours=cfg.cache_ttl_hours,
        use_stale_cache_for_backtest=cfg.use_stale_cache_for_backtest,
    )

    out = daily_df.copy()
    out["timestamp"] = _to_utc_ts(out["timestamp"] if "timestamp" in out.columns else out.index)
    out = out.sort_values("timestamp").reset_index(drop=True)

    warnings: list[str] = []
    series_used: list[str] = []
    failed_series: list[str] = []
    series_lags_hours: dict[str, float] = {}
    per_series_summary: dict[str, dict[str, Any]] = {}
    generated_cols: dict[str, pd.Series] = {}

    try:
        for entry in list(cfg.series):
            if not isinstance(entry, dict):
                continue
            series_id = str(entry.get("series_id") or "").strip()
            if not series_id:
                continue

            freq_hint = str(entry.get("native_frequency_hint") or "daily").lower()
            lag_hours = float(entry.get("availability_lag_hours") or _default_lag_hours_for_frequency(cfg, freq_hint))
            lag_hours = lag_hours * float(max(0.1, cfg.lag_stress_multiplier))
            series_lags_hours[series_id] = lag_hours

            try:
                realtime_start = obs_start if cfg.realtime_mode == "vintage_dates" else None
                realtime_end = obs_end if cfg.realtime_mode == "vintage_dates" else None
                obs = client.get_series_observations(
                    series_id,
                    observation_start=obs_start,
                    observation_end=obs_end,
                    realtime_start=realtime_start,
                    realtime_end=realtime_end,
                    output_type=1,
                )
            except Exception as exc:
                warnings.append(f"series_fetch_failed:{series_id}:{exc.__class__.__name__}")
                failed_series.append(series_id)
                client.stats.series_errors += 1
                logger.warning("FRED fetch failed for %s (%s)", series_id, exc.__class__.__name__)
                continue

            if obs.empty:
                warnings.append(f"series_empty:{series_id}")
                failed_series.append(series_id)
                continue

            aligned = align_fred_series_to_target(out["timestamp"], obs, lag_hours=lag_hours)
            aligned = pd.to_numeric(aligned, errors="coerce")
            ffill_limit = _ffill_limit_by_frequency(freq_hint)
            level = aligned.ffill(limit=ffill_limit).reset_index(drop=True)

            prefix = f"fred_{series_id}"
            delta_5 = level.diff(5)
            delta_20 = level.diff(20)
            delta_60 = level.diff(60)

            lookback = _lookback_by_frequency(cfg, freq_hint)
            z_level = _rolling_z(level, lookback, cfg.zscore_clip)
            z_delta_5 = _rolling_z(delta_5, lookback, cfg.zscore_clip)
            z_delta_20 = _rolling_z(delta_20, lookback, cfg.zscore_clip)
            z_delta_60 = _rolling_z(delta_60, lookback, cfg.zscore_clip)

            generated_cols[f"{prefix}_level"] = level
            generated_cols[f"{prefix}_delta_5"] = delta_5
            generated_cols[f"{prefix}_delta_20"] = delta_20
            generated_cols[f"{prefix}_delta_60"] = delta_60
            generated_cols[f"{prefix}_z_level"] = z_level
            generated_cols[f"{prefix}_z_delta_5"] = z_delta_5
            generated_cols[f"{prefix}_z_delta_20"] = z_delta_20
            generated_cols[f"{prefix}_z_delta_60"] = z_delta_60

            if bool(entry.get("index_like", False)):
                pct_5 = level.pct_change(5)
                pct_20 = level.pct_change(20)
                pct_60 = level.pct_change(60)
                z_pct_20 = _rolling_z(pct_20, lookback, cfg.zscore_clip)
                generated_cols[f"{prefix}_pct_5"] = pct_5
                generated_cols[f"{prefix}_pct_20"] = pct_20
                generated_cols[f"{prefix}_pct_60"] = pct_60
                generated_cols[f"{prefix}_z_pct_20"] = z_pct_20

            if series_id in {"WALCL", "M2SL"}:
                yoy = level.pct_change(252)
                generated_cols[f"{prefix}_yoy"] = yoy
                generated_cols[f"{prefix}_z_yoy"] = _rolling_z(yoy, lookback, cfg.zscore_clip)

            series_used.append(series_id)
            per_series_summary[series_id] = {
                "friendly_name": str(entry.get("friendly_name") or series_id),
                "frequency": freq_hint,
                "lag_hours": lag_hours,
                "stats": _series_summary(level),
            }

        if generated_cols:
            feature_df = pd.DataFrame({k: v.values for k, v in generated_cols.items()}, index=out.index)
            out = pd.concat([out, feature_df], axis=1)

        risk_off, applied_weights = _compose_fred_risk_off_score(out, cfg)
        out["fred_risk_off_score"] = risk_off.clip(lower=0.0, upper=1.0)
        span = max(2, int(cfg.risk_off_score_ema_span))
        out["fred_risk_off_score_smooth"] = out["fred_risk_off_score"].ewm(span=span, adjust=False).mean().clip(lower=0.0, upper=1.0)

        max_penalty = float(max(0.0, min(1.0, cfg.max_risk_off_penalty)))
        out["fred_penalty_multiplier"] = (1.0 - out["fred_risk_off_score_smooth"] * max_penalty).clip(lower=0.0, upper=1.0)

        report = {
            "enabled": True,
            "realtime_mode": cfg.realtime_mode,
            "series_used": sorted(series_used),
            "series_failed": sorted(set(failed_series)),
            "series_lags_hours": series_lags_hours,
            "risk_off_weights": {k: float(v) for k, v in cfg.risk_off_weights.items()},
            "applied_risk_off_weights": applied_weights,
            "max_risk_off_penalty": max_penalty,
            "feature_windows": {
                "delta": [5, 20, 60],
                "daily_z_lookback": int(cfg.daily_z_lookback),
                "weekly_z_lookback": int(cfg.weekly_z_lookback),
                "monthly_z_lookback": int(cfg.monthly_z_lookback),
                "risk_off_score_ema_span": int(cfg.risk_off_score_ema_span),
                "lag_stress_multiplier": float(cfg.lag_stress_multiplier),
            },
            "series_summary": per_series_summary,
            "cache_hit_rate": float(client.stats.cache_hit_rate),
            "cache_hits": int(client.stats.cache_hits),
            "cache_misses": int(client.stats.cache_misses),
            "requests": int(client.stats.requests),
            "warnings": sorted(set(warnings)),
        }
        return FredFeatureBuildResult(daily_features=out, report=report)
    finally:
        client.close()
