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
    strategy: str = "regime_switching"
    initial_equity: float = 10000.0
    output_dir: str = "reports"
    maker_bps: float = 10.0
    taker_bps: float = 25.0
    slippage_bps: float = 5.0
    use_spread_slippage: bool = True
    max_trades_per_year: Optional[int] = None
    ci_mode: bool = False

    VALID_STRATEGIES: ClassVar[Set[str]] = {
        "regime_switching",
        "regime_switching_v2",
        "regime_switching_v3",
        "regime_switching_v4_core",
        "macro_gate_benchmark",
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
    regime: RegimeConfig = Field(default_factory=RegimeConfig)
    risk: RiskConfig = Field(default_factory=RiskConfig)
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)
    fees: FeesConfig = Field(default_factory=FeesConfig)
    backtest: BacktestConfig = Field(default_factory=BacktestConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)

    @field_validator("coinbase", "data", "public_sources", "regime", "risk", "execution", "fees", "backtest", "logging", "runtime", mode="before")
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

        env_overrides = {
            "coinbase.api_key": env("COINBASE_API_KEY", raw.get("coinbase", {}).get("api_key") if isinstance(raw.get("coinbase", {}), dict) else None),
            "coinbase.api_secret": env("COINBASE_API_SECRET", raw.get("coinbase", {}).get("api_secret") if isinstance(raw.get("coinbase", {}), dict) else None),
            "coinbase.use_sandbox": env("COINBASE_USE_SANDBOX", raw.get("coinbase", {}).get("use_sandbox") if isinstance(raw.get("coinbase", {}), dict) else False, bool),
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
        for k, v in env_overrides.items():
            _set_path(merged, k, v)

        # Also expose direct env vars for convenience
        merged.setdefault("coinbase", {}).setdefault("api_key", os.getenv("COINBASE_API_KEY", merged.get("coinbase", {}).get("api_key")))
        merged.setdefault("coinbase", {}).setdefault("api_secret", os.getenv("COINBASE_API_SECRET", merged.get("coinbase", {}).get("api_secret")))
        merged.setdefault("coinbase", {}).setdefault("api_passphrase", os.getenv("COINBASE_API_PASSPHRASE", merged.get("coinbase", {}).get("api_passphrase")))

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
