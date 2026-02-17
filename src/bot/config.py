from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Dict, Literal, Optional, Set
import json
import os
from datetime import datetime
import tomllib

try:
    from pydantic import BaseModel, Field, field_validator
except Exception:  # pragma: no cover - pydantic v1 compatibility
    from pydantic import BaseModel, Field, validator

    def field_validator(*fields, mode="after", **kwargs):  # type: ignore
        pre = mode == "before"

        def _wrap(fn):
            return validator(*fields, pre=pre, always=kwargs.get("always", False))(fn)

        return _wrap


TIMEFRAME = Literal["1h", "1d"]


def _load_default_fred_series_registry() -> list[dict[str, Any]]:
    registry_path = Path(__file__).resolve().parent / "data" / "fred_series_registry.json"
    if not registry_path.exists():
        return []
    try:
        payload = json.loads(registry_path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            return [dict(x) for x in payload if isinstance(x, dict)]
    except Exception:
        pass
    return []


class CoinbaseConfig(BaseModel):
    api_key: Optional[str] = Field(default=None, alias="COINBASE_API_KEY")
    api_secret: Optional[str] = Field(default=None, alias="COINBASE_API_SECRET")
    api_passphrase: Optional[str] = Field(default=None, alias="COINBASE_API_PASSPHRASE")
    use_sandbox: bool = Field(default=False, alias="COINBASE_USE_SANDBOX")
    request_timeout_s: float = 15.0
    request_burst: int = 5
    requests_per_minute: int = 120
    sandbox_error_variant: Optional[str] = Field(default=None, alias="COINBASE_SANDBOX_ERROR_VARIANT")
    max_clock_skew_s: float = 300.0

    @field_validator("api_key", "api_secret", mode="before")
    def _trim_str(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        value = str(value).strip()
        return value or None


class DataConfig(BaseModel):
    product: str = "BTC-USD"
    cache_dir: Path = Field(default=Path(".trading_bot_cache"))
    use_parquet_cache: bool = False
    stale_after_minutes: int = 15
    cache_ttl_hours: int = 6
    hourly_limit: int = 350
    daily_limit: int = 350
    force_refresh: bool = False

    @field_validator("cache_dir", mode="before")
    def _cache_dir(cls, value: Any) -> Path:
        if isinstance(value, Path):
            return value
        return Path(value)


class PublicSourcesConfig(BaseModel):
    enabled: bool = False
    fear_greed_enabled: bool = False
    blockchain_enabled: bool = False
    cache_ttl_minutes: int = 60


class FredConfig(BaseModel):
    enabled: bool = False
    api_key: Optional[str] = Field(default=None, alias="FRED_API_KEY")
    api_base_url: str = "https://api.stlouisfed.org/fred"
    cache_dir: Path = Field(default=Path(".trading_bot_cache/fred"))
    http_timeout_seconds: float = 15.0
    max_retries: int = 5
    backoff_seconds: float = 0.5
    cache_ttl_hours: float = 24.0
    use_stale_cache_for_backtest: bool = True

    # Data treatment
    realtime_mode: Literal["lagged_latest", "vintage_dates"] = "lagged_latest"
    default_availability_lag_hours: dict[str, float] = Field(
        default_factory=lambda: {
            "daily": 24.0,
            "weekly": float(7 * 24),
            "monthly": float(35 * 24),
        }
    )
    # backward-compatible flat keys
    default_availability_lag_hours_daily: int = 24
    default_availability_lag_hours_weekly: int = 7 * 24
    default_availability_lag_hours_monthly: int = 35 * 24
    lag_stress_multiplier: float = 1.0

    # Feature engineering
    daily_z_lookback: int = 252
    weekly_z_lookback: int = 104
    monthly_z_lookback: int = 60
    zscore_clip: float = 4.0

    # Macro overlay controls
    max_risk_off_penalty: float = 0.5
    risk_off_score_ema_span: int = 16
    risk_off_weights: dict[str, float] = Field(
        default_factory=lambda: {
            "VIXCLS": 1.0,
            "BAMLH0A0HYM2": 1.0,
            "BAA10Y": 0.8,
            "STLFSI4": 0.7,
            "NFCI": 0.7,
            "DTWEXBGS": 0.4,
            "curve_inversion": 0.8,
            "WALCL": 0.2,
            "M2SL": 0.2,
        }
    )

    # Series registry loaded from repo-local defaults and overrideable in config.
    series: list[dict[str, Any]] = Field(default_factory=_load_default_fred_series_registry)

    @field_validator("api_key", mode="before")
    def _trim_api_key(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        out = str(value).strip()
        return out or None

    @field_validator("default_availability_lag_hours", mode="before")
    def _lag_map(cls, value: Any) -> dict[str, float]:
        base = {
            "daily": 24.0,
            "weekly": float(7 * 24),
            "monthly": float(35 * 24),
        }
        if not isinstance(value, dict):
            return base
        out = dict(base)
        for key, raw in value.items():
            k = str(key).strip().lower()
            if k not in out:
                continue
            try:
                out[k] = float(raw)
            except Exception:
                continue
        return out

    @field_validator("cache_dir", mode="before")
    def _cache_dir(cls, value: Any) -> Path:
        if isinstance(value, Path):
            return value
        return Path(value)


class RegimeConfig(BaseModel):
    # Macro
    daily_trend_window: int = 50
    daily_momentum_window: int = 28
    daily_momentum_quantile: float = 0.66

    # Macro scoring mode (backward-compatible default keeps legacy binary gating)
    macro_mode: Literal["binary", "score", "stateful_gate"] = "binary"
    macro_score_components: list[str] = Field(
        default_factory=lambda: [
            "close_gt_sma50",
            "close_gt_sma200",
            "ret_28d_pos",
            "ret_90d_pos",
        ]
    )
    macro_score_transform: Literal["linear", "piecewise"] = "linear"
    macro_score_floor: float = 0.0
    macro_score_min_to_trade: float = 0.25
    macro_piecewise_levels: list[float] = Field(default_factory=lambda: [0.0, 0.33, 0.66, 1.0])

    # Stateful macro gate (v3)
    macro_enter_threshold: float = 0.75
    macro_exit_threshold: float = 0.25
    macro_full_threshold: float = 1.0
    macro_half_threshold: float = 0.75
    macro_confirm_days: int = 2
    macro_min_on_days: int = 2
    macro_min_off_days: int = 1
    macro_half_multiplier: float = 0.5
    macro_full_multiplier: float = 1.0

    # Trend playbook selection
    trend_playbook: Literal["core_momentum_daily", "breakout"] = "core_momentum_daily"

    # Micro
    adx_window: int = 14
    chop_window: int = 14
    realized_vol_window: int = 168
    vol_lookback_days: int = 365
    vol_high_threshold_quantile: float = 0.90
    adx_trend_threshold: float = 25.0
    adx_range_threshold: float = 20.0
    chop_trend_threshold: float = 38.2
    chop_range_threshold: float = 61.8
    regime_confirmation_bars: int = 3
    min_regime_duration_hours: int = 6

    # Range
    bb_window: int = 20
    bb_stdev: float = 2.0
    range_tranche_size: float = 0.25
    range_max_exposure: float = 0.75
    range_min_time_between_trades_hours: float = 2.0
    range_max_trades_per_day: int = 4

    # Hourly overlay controls (small add/reduce around core baseline)
    enable_hourly_overlay: bool = True
    overlay_max_adjustment: float = 0.25

    # Trend
    trend_mode: Literal["donchian", "ema_cross"] = "donchian"
    donchian_window: int = 55
    ema_fast: int = 20
    ema_slow: int = 50
    trend_exposure_cap: float = 1.0
    vol_target_multiplier: float = 1.0
    atr_window: int = 14
    atr_mult: float = 3.0

    # Trend-strength booster (disabled by default for backward compatibility)
    trend_boost_enabled: bool = False
    trend_boost_multiplier: float = 1.25
    trend_boost_adx_threshold: float = 25.0
    trend_boost_macro_score_threshold: float = 0.75
    trend_boost_confirm_days: int = 2
    trend_boost_min_on_days: int = 2
    trend_boost_min_off_days: int = 1
    trend_boost_regime_gate: Literal["micro_trend", "daily_sma50"] = "micro_trend"
    trend_boost_require_micro_trend: bool = True
    trend_boost_require_above_sma200: bool = True
    trend_boost_sma50_slope_lookback_days: int = 10

    # High vol
    high_vol_cap: float = 0.2

    # Vol targeting
    target_ann_vol: float = 0.25
    realized_vol_window_hours: int = 168
    max_position_fraction: float = 1.0

    # HMM
    hmm_regime_enabled: bool = False
    hmm_window_hours: int = 1000
    hmm_n_states: int = 3

    # Back-compat switch for old sub-strategy switching behavior
    legacy_substrategy_switching: bool = False

    # V4 core strategy config
    v4_macro_enter_threshold: float = 0.75
    v4_macro_exit_threshold: float = 0.25
    v4_macro_half_threshold: float = 0.75
    v4_macro_full_threshold: float = 1.0
    v4_macro_confirm_days: int = 2
    v4_macro_min_on_days: int = 2
    v4_macro_min_off_days: int = 1
    v4_macro_half_multiplier: float = 0.5
    v4_macro_full_multiplier: float = 1.0
    v4_micro_mult_trend: float = 1.0
    v4_micro_mult_range: float = 1.0
    v4_micro_mult_neutral: float = 1.0
    v4_micro_mult_high_vol: float = 0.0
    v4_core_risk_refresh: str = "daily"

    # V5 adaptive strategy config
    # -- Asymmetric micro regime multipliers (allows boost > 1.0 in trends) --
    v5_micro_trend_mult: float = 1.15
    v5_micro_range_mult: float = 0.85
    v5_micro_neutral_mult: float = 1.0
    v5_micro_high_vol_mult: float = 0.0
    v5_micro_max_mult: float = 1.5  # hard cap on any micro multiplier

    # -- Adaptive macro gate --
    v5_adaptive_gate_enabled: bool = True
    v5_adaptive_vol_window_days: int = 60
    v5_adaptive_enter_base: float = 0.75
    v5_adaptive_exit_base: float = 0.25
    v5_adaptive_half_base: float = 0.75
    v5_adaptive_full_base: float = 1.0
    v5_adaptive_sensitivity: float = 0.15  # max threshold shift per unit vol-z
    v5_adaptive_confirm_days: int = 2
    v5_adaptive_min_on_days: int = 2
    v5_adaptive_min_off_days: int = 1
    v5_adaptive_half_multiplier: float = 0.5
    v5_adaptive_full_multiplier: float = 1.0

    # macro_only_v2 controls
    macro2_signal_mode: Literal[
        "sma200_band",
        "mom_6_12",
        "sma200_and_mom",
        "sma200_or_mom",
        "score4_legacy",
    ] = "sma200_and_mom"
    macro2_confirm_days: int = 2
    macro2_min_on_days: int = 2
    macro2_min_off_days: int = 1
    macro2_weight_off: float = 0.0
    macro2_weight_half: float = 0.50
    macro2_weight_full: float = 1.0
    macro2_vol_mode: Literal["none", "inverse_vol"] = "inverse_vol"
    macro2_vol_lookback_days: int = 60
    macro2_vol_floor: float = 0.05
    macro2_target_ann_vol_half: float = 0.30
    macro2_target_ann_vol_full: float = 0.60
    macro2_dd_enabled: bool = True
    macro2_dd_threshold: float = 0.25
    macro2_dd_cooldown_days: int = 10
    macro2_dd_reentry_confirm_days: int = 2
    macro2_dd_safe_weight: float = 0.0
    sma200_entry_band: float = 0.0
    sma200_exit_band: float = 0.0
    mom_6m_days: int = 180
    mom_12m_days: int = 365


class RiskConfig(BaseModel):
    max_drawdown_cut_pct: float = 0.25
    max_exposure: float = 1.0
    max_additional_exposure_on_drawdown: float = 0.5
    stale_bar_max_multiplier: int = 2
    cutoff_no_new_entries: bool = True

    # kill-switches
    daily_loss_limit_pct: Optional[float] = 0.08
    max_consecutive_losses: Optional[int] = 6
    manual_kill_switch: bool = False
    safe_mode: bool = True


class ExecutionConfig(BaseModel):
    limit_price_offset_bps: float = 3.0
    order_timeout_s: int = 60
    fallback_to_market: bool = True
    post_only: bool = True
    max_slippage_bps: float = 35.0
    spread_bps: float = 15.0
    impact_bps: float = 2.0
    maker_bps: float = 10.0
    taker_bps: float = 25.0
    paper_fill_delay_s: float = 0.5

    # backtest execution realism
    fill_model: Literal["next_open", "bid_ask", "worst_case_bar"] = "bid_ask"
    rebalance_policy: Literal["signal_change_only", "band", "always"] = "signal_change_only"

    # anti-churn controls
    min_trade_notional_usd: float = 50.0
    min_exposure_delta: float = 0.05
    target_quantization_step: float = 0.25
    min_time_between_trades_hours: float = 1.0
    max_trades_per_day: int = 8
    max_allowed_slippage_bps: float = 50.0

    # exchange constraints
    enforce_product_constraints: bool = True
    min_notional_buffer_quote: float = 0.0

    # maker-first routing
    maker_first: bool = True
    maker_timeout_seconds: int = 60
    maker_retries: int = 3
    allow_taker_fallback: bool = False
    taker_fallback_only_if_edge_exceeds_cost: bool = True

    # cancel/replace lifecycle
    cancel_replace_on_timeout: bool = True
    replace_with_market_on_timeout: bool = True


class FeesConfig(BaseModel):
    maker_bps: float = 10.0
    taker_bps: float = 25.0


class LoggingConfig(BaseModel):
    json_logs: bool = True
    console_level: str = "INFO"
    file_path: Optional[str] = "logs/trading_bot.log"


class BacktestConfig(BaseModel):
    start: Optional[str] = None
    end: Optional[str] = None
    strategy: str = "macro_gate_benchmark"
    initial_equity: float = 10000.0
    output_dir: str = "reports"
    maker_bps: float = 10.0
    taker_bps: float = 25.0
    slippage_bps: float = 5.0
    use_spread_slippage: bool = True
    max_trades_per_year: Optional[int] = None
    ci_mode: bool = False

    # compute acceleration backend for backtests (falls back to CPU if CUDA unavailable)
    acceleration_backend: Literal["auto", "cpu", "cuda"] = "auto"
    acceleration_min_bars: int = 2048

    VALID_STRATEGIES: ClassVar[Set[str]] = {
        "macro_gate_benchmark",
        "macro_only_v2",
        # Reactivated legacy strategies
        "regime_switching_v3",
        "regime_switching_orchestrator",
        "regime_switching",
        "regime_switching_v4_core",
        "v4_core",
        "v5_adaptive",
    }

    @field_validator("strategy", mode="after")
    def _validate_strategy(cls, value: str) -> str:
        if value not in cls.VALID_STRATEGIES:
            raise ValueError(f"Unknown strategy '{value}'. Valid: {sorted(cls.VALID_STRATEGIES)}")
        return value


class RuntimeConfig(BaseModel):
    mode: Literal["paper", "live", "backtest"] = "backtest"
    cycles: Optional[int] = None
    tick_seconds: int = 60

    # operational hardening
    healthcheck_file: str = ".trading_bot_cache/health_status.json"
    alert_file: str = ".trading_bot_cache/alerts.log"
    max_consecutive_cycle_failures: int = 3
    max_consecutive_order_failures: int = 3
    stale_feed_alert_minutes: int = 90


class BotConfig(BaseModel):
    coinbase: CoinbaseConfig = Field(default_factory=CoinbaseConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    public_sources: PublicSourcesConfig = Field(default_factory=PublicSourcesConfig)
    fred: FredConfig = Field(default_factory=FredConfig)
    regime: RegimeConfig = Field(default_factory=RegimeConfig)
    risk: RiskConfig = Field(default_factory=RiskConfig)
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)
    fees: FeesConfig = Field(default_factory=FeesConfig)
    backtest: BacktestConfig = Field(default_factory=BacktestConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)

    @field_validator("coinbase", "data", "public_sources", "fred", "regime", "risk", "execution", "fees", "backtest", "logging", "runtime", mode="before")
    def _convert_nested(cls, value: Any) -> Dict[str, Any]:
        if value is None:
            return {}
        if isinstance(value, dict):
            return value
        return value

    @classmethod
    def load(cls, path: Optional[str] = None) -> "BotConfig":
        # merge config file (json/toml/yaml-like plain key=val) with env vars
        raw: Dict[str, Any] = {}
        if path:
            file_path = Path(path)
            if not file_path.exists():
                raise FileNotFoundError(f"Config file not found: {path}")
            text = file_path.read_text(encoding="utf-8")
            if file_path.suffix.lower() in {".yml", ".yaml"}:
                try:
                    import yaml  # type: ignore
                except Exception as err:
                    raise RuntimeError("PyYAML required for YAML config files") from err
                raw = yaml.safe_load(text) or {}
            elif file_path.suffix.lower() == ".toml":
                raw = tomllib.loads(text)
            else:
                raw = json.loads(text)

        def env(name: str, current: Any, cast=str):
            env_name = f"TRADING_BOT_{name}"
            if env_name not in os.environ:
                return current
            raw_value = os.environ[env_name]
            if cast in {int, float, bool}:
                if cast is bool:
                    return str(raw_value).lower() in {"1", "true", "yes", "on"}
                return cast(raw_value)
            return cast(raw_value)

        coinbase_raw = raw.get("coinbase", {}) if isinstance(raw.get("coinbase", {}), dict) else {}
        fred_raw = raw.get("fred", {}) if isinstance(raw.get("fred", {}), dict) else {}

        env_overrides = {
            # Use alias keys so pydantic accepts them consistently across v1/v2.
            "coinbase.COINBASE_API_KEY": env("COINBASE_API_KEY", coinbase_raw.get("COINBASE_API_KEY", coinbase_raw.get("api_key"))),
            "coinbase.COINBASE_API_SECRET": env("COINBASE_API_SECRET", coinbase_raw.get("COINBASE_API_SECRET", coinbase_raw.get("api_secret"))),
            "coinbase.COINBASE_API_PASSPHRASE": env("COINBASE_API_PASSPHRASE", coinbase_raw.get("COINBASE_API_PASSPHRASE", coinbase_raw.get("api_passphrase"))),
            "coinbase.COINBASE_USE_SANDBOX": env("COINBASE_USE_SANDBOX", coinbase_raw.get("COINBASE_USE_SANDBOX", coinbase_raw.get("use_sandbox", False)), bool),
            "fred.FRED_API_KEY": os.getenv(
                "FRED_API_KEY",
                env("FRED_API_KEY", fred_raw.get("FRED_API_KEY", fred_raw.get("api_key"))),
            ),
        }

        def _set_path(d: dict[str, Any], dotted: str, value: Any) -> None:
            if value is None:
                return
            parts = dotted.split(".")
            cur = d
            for p in parts[:-1]:
                if p not in cur or not isinstance(cur[p], dict):
                    cur[p] = {}
                cur = cur[p]
            cur[parts[-1]] = value

        merged = dict(raw)

        # Backward compatibility: map non-aliased keys in config files to aliases.
        coinbase_merged = merged.setdefault("coinbase", {}) if isinstance(merged.get("coinbase", {}), dict) else {}
        if isinstance(coinbase_merged, dict):
            if "api_key" in coinbase_merged and "COINBASE_API_KEY" not in coinbase_merged:
                coinbase_merged["COINBASE_API_KEY"] = coinbase_merged.get("api_key")
            if "api_secret" in coinbase_merged and "COINBASE_API_SECRET" not in coinbase_merged:
                coinbase_merged["COINBASE_API_SECRET"] = coinbase_merged.get("api_secret")
            if "api_passphrase" in coinbase_merged and "COINBASE_API_PASSPHRASE" not in coinbase_merged:
                coinbase_merged["COINBASE_API_PASSPHRASE"] = coinbase_merged.get("api_passphrase")
            if "use_sandbox" in coinbase_merged and "COINBASE_USE_SANDBOX" not in coinbase_merged:
                coinbase_merged["COINBASE_USE_SANDBOX"] = coinbase_merged.get("use_sandbox")

        fred_merged = merged.setdefault("fred", {}) if isinstance(merged.get("fred", {}), dict) else {}
        if isinstance(fred_merged, dict):
            if "api_key" in fred_merged and "FRED_API_KEY" not in fred_merged:
                fred_merged["FRED_API_KEY"] = fred_merged.get("api_key")

        for k, v in env_overrides.items():
            _set_path(merged, k, v)

        # Also expose direct env vars for convenience.
        if isinstance(merged.get("coinbase"), dict):
            c = merged["coinbase"]
            c.setdefault("COINBASE_API_KEY", os.getenv("COINBASE_API_KEY", c.get("COINBASE_API_KEY")))
            c.setdefault("COINBASE_API_SECRET", os.getenv("COINBASE_API_SECRET", c.get("COINBASE_API_SECRET")))
            c.setdefault("COINBASE_API_PASSPHRASE", os.getenv("COINBASE_API_PASSPHRASE", c.get("COINBASE_API_PASSPHRASE")))
        if isinstance(merged.get("fred"), dict):
            f = merged["fred"]
            f.setdefault("FRED_API_KEY", os.getenv("FRED_API_KEY", f.get("FRED_API_KEY")))

        if hasattr(cls, "model_validate"):
            cfg = cls.model_validate(merged)
        else:
            cfg = cls.parse_obj(merged)
        return cfg


@dataclass(frozen=True)
class IntervalSpec:
    timeframe: TIMEFRAME
    seconds: int


def timeframe_to_seconds(tf: TIMEFRAME) -> int:
    if tf == "1h":
        return 60 * 60
    if tf == "1d":
        return 24 * 60 * 60
    raise ValueError(f"Unsupported timeframe: {tf}")


def now_utc() -> datetime:
    return datetime.utcnow().replace(tzinfo=None)
