from __future__ import annotations

import importlib.util
import sys
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)  # type: ignore[assignment]
    return module


def _base_args(tmp_path: Path, *, run_id: str, resume: bool) -> SimpleNamespace:
    return SimpleNamespace(
        product="BTC-USD",
        start="2025-01-01T00:00:00Z",
        end="2025-12-31T23:00:00Z",
        train_start="2025-01-01T00:00:00Z",
        train_end="2025-06-30T23:00:00Z",
        val_start="2025-07-01T00:00:00Z",
        val_end="2025-09-30T23:00:00Z",
        test_start="2025-10-01T00:00:00Z",
        test_end="2025-12-31T23:00:00Z",
        strategy="macro_gate_benchmark",
        fill_model="bid_ask",
        acceleration_backend="cpu",
        config=None,
        grid_config=None,
        grid=[],
        include_fred_grid=False,
        small=False,
        turnover_max=700.0,
        max_drawdown_max=0.30,
        top_n=2,
        output_dir=str(tmp_path / "frontier_out"),
        run_id=run_id,
        resume=resume,
        checkpoint_every=1,
        maker_bps=10.0,
        taker_bps=25.0,
    )


def _synthetic_candles() -> tuple[pd.DataFrame, pd.DataFrame]:
    hourly_ts = pd.date_range("2025-01-01", periods=24 * 10, freq="h", tz="UTC")
    hourly = pd.DataFrame(
        {
            "timestamp": hourly_ts,
            "open": 50_000.0,
            "high": 50_010.0,
            "low": 49_990.0,
            "close": 50_000.0,
            "volume": 100.0,
        }
    )

    daily_ts = pd.date_range("2025-01-01", periods=365, freq="D", tz="UTC")
    daily = pd.DataFrame(
        {
            "timestamp": daily_ts,
            "open": 50_000.0,
            "high": 50_100.0,
            "low": 49_900.0,
            "close": 50_000.0,
            "volume": 1_000.0,
        }
    )
    return hourly, daily


def test_frontier_sweep_contracts_and_resume_fingerprint(monkeypatch, tmp_path):
    module = _load_module(
        "frontier_sweep_contract_test",
        REPO_ROOT / "scripts" / "frontier_sweep.py",
    )

    args_holder = {"value": _base_args(tmp_path, run_id="contract1", resume=False)}
    grid_holder = {
        "value": [
            {"target_ann_vol": 0.30},
            {"target_ann_vol": 0.40},
        ]
    }

    hourly, daily = _synthetic_candles()

    class _Tx:
        maker_fee_rate = 0.0
        taker_fee_rate = 0.0

    class _FakeRest:
        def __init__(self, *_args, **_kwargs):
            pass

        def get_transaction_summary(self, _product: str):
            return _Tx()

    class _FakeStore:
        def __init__(self, *_args, **_kwargs):
            pass

        def get_candles(self, *, client, query):
            _ = client
            return hourly if query.timeframe == "1h" else daily

    def _fake_run_window(*, window, params, scenario, **_kwargs):
        vol_bonus = 0.01 if float(params.get("target_ann_vol", 0.0)) >= 0.40 else 0.0
        scenario_penalty = {
            "baseline": 0.00,
            "stress_1": -0.01,
            "stress_2": -0.02,
        }[scenario.name]
        base_cagr = {
            "train": 0.08,
            "val": 0.10,
            "test": 0.07,
        }[window.name]
        cagr = base_cagr + vol_bonus + scenario_penalty
        sharpe = cagr * 10.0
        return {
            "window": window.name,
            "scenario": scenario.name,
            "start": window.start.isoformat(),
            "end": window.end.isoformat(),
            "cagr": cagr,
            "sharpe": sharpe,
            "sortino": sharpe,
            "max_drawdown": -0.12,
            "profit_factor": 1.2,
            "turnover": 120.0,
            "trade_count": 15,
            "net_pnl": 100.0,
            "maker_rate": 0.0,
            "taker_rate": 0.0,
            "impact_bps": 1.0,
            "spread_bps": 1.0,
            "fred_enabled": False,
            "fred_max_risk_off_penalty": 0.0,
            "fred_risk_off_score_ema_span": 8,
            "fred_lag_stress_multiplier": 1.0,
            "fred_cache_hit_rate": 1.0,
            "fred_series_used_count": 0,
        }

    monkeypatch.setattr(module, "parse_args", lambda: args_holder["value"])
    monkeypatch.setattr(module, "setup_system_logger", lambda: str(tmp_path / "system.log"))
    monkeypatch.setattr(module, "_validate_acceleration_backend", lambda _x: True)
    monkeypatch.setattr(module.BotConfig, "load", staticmethod(lambda _cfg=None: module.BotConfig()))
    monkeypatch.setattr(module, "RESTClientWrapper", _FakeRest)
    monkeypatch.setattr(module, "CandleStore", _FakeStore)
    monkeypatch.setattr(module, "run_window", _fake_run_window)
    monkeypatch.setattr(module, "load_grid", lambda *_a, **_k: [dict(x) for x in grid_holder["value"]])

    rc_first = module.main()
    assert rc_first == 0

    run_dir = Path(args_holder["value"].output_dir) / "run_contract1"
    best_summary_path = run_dir / "best_summary.json"
    filter_path = run_dir / "filter_rejections.json"
    checkpoint_path = run_dir / "checkpoint.json"

    assert best_summary_path.exists()
    best_summary = json.loads(best_summary_path.read_text(encoding="utf-8"))
    assert {
        "strategy",
        "run_id",
        "best",
        "constraints",
        "reproduce_test_command",
        "best_config",
        "files",
        "paths",
        "test_window_stress_1",
    }.issubset(best_summary.keys())

    assert filter_path.exists()
    filter_payload = json.loads(filter_path.read_text(encoding="utf-8"))
    assert {
        "run_id",
        "strategy",
        "total_param_sets",
        "accepted_param_sets",
        "rejections",
    }.issubset(filter_payload.keys())

    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    assert checkpoint.get("grid_hash")
    assert checkpoint.get("window_hash")

    # Safe resume with unchanged grid should succeed.
    args_holder["value"] = _base_args(tmp_path, run_id="contract1", resume=True)
    rc_resume_ok = module.main()
    assert rc_resume_ok == 0

    # Unsafe resume with changed grid should be rejected.
    grid_holder["value"] = [
        {"target_ann_vol": 0.30},
        {"target_ann_vol": 0.40},
        {"target_ann_vol": 0.50},
    ]
    args_holder["value"] = _base_args(tmp_path, run_id="contract1", resume=True)
    rc_resume_bad = module.main()
    assert rc_resume_bad == 2


# ---------------------------------------------------------------------------
# Strategy-specific grid selection tests
# ---------------------------------------------------------------------------

class TestStrategyGridSelection:
    """Verify that load_grid selects the correct grid based on strategy."""

    @staticmethod
    def _load_module():
        return _load_module(
            "frontier_sweep_grid_test",
            REPO_ROOT / "scripts" / "frontier_sweep.py",
        )

    def test_adaptive_trend_6h_uses_strategy_grid(self):
        mod = self._load_module()
        param_sets = mod.load_grid(None, {}, strategy="adaptive_trend_6h_v1")
        keys = set(param_sets[0].keys())
        assert "adaptive6h_use_macro_gate" in keys
        assert "adaptive6h_target_ann_vol" in keys
        # Must NOT contain default grid keys
        assert "macro_mode" not in keys
        assert "trend_boost_multiplier" not in keys

    def test_default_strategy_uses_default_grid(self):
        mod = self._load_module()
        param_sets = mod.load_grid(None, {}, strategy="macro_gate_benchmark")
        keys = set(param_sets[0].keys())
        assert "macro_mode" in keys
        assert "trend_boost_multiplier" in keys
        assert "adaptive6h_use_macro_gate" not in keys

    def test_unknown_strategy_falls_back_to_default(self):
        mod = self._load_module()
        param_sets = mod.load_grid(None, {}, strategy="unknown_strategy_xyz")
        keys = set(param_sets[0].keys())
        assert "macro_mode" in keys

    def test_adaptive_grid_includes_macro_gate_toggle(self):
        """The grid MUST include both True and False for the macro gate."""
        mod = self._load_module()
        param_sets = mod.load_grid(None, {}, strategy="adaptive_trend_6h_v1")
        gate_values = {p["adaptive6h_use_macro_gate"] for p in param_sets}
        assert gate_values == {True, False}

    def test_small_adaptive_grid(self):
        mod = self._load_module()
        param_sets = mod.load_grid(None, {}, strategy="adaptive_trend_6h_v1", small=True)
        assert len(param_sets) >= 2
        keys = set(param_sets[0].keys())
        assert "adaptive6h_use_macro_gate" in keys

    def test_grid_flags_override_strategy_grid(self):
        mod = self._load_module()
        param_sets = mod.load_grid(
            None,
            {"adaptive6h_target_ann_vol": [0.50]},
            strategy="adaptive_trend_6h_v1",
        )
        vol_values = {p["adaptive6h_target_ann_vol"] for p in param_sets}
        assert vol_values == {0.50}

    def test_all_adaptive_grid_keys_are_valid_config_params(self):
        """Every key in the strategy grid must resolve via set_param."""
        mod = self._load_module()
        param_sets = mod.load_grid(None, {}, strategy="adaptive_trend_6h_v1")
        invalid = mod.validate_grid_keys(mod.BotConfig(), param_sets)
        assert invalid == [], f"Invalid grid keys: {invalid}"
