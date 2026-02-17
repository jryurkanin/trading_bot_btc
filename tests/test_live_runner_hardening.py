from __future__ import annotations

import time
from datetime import datetime, timezone
from types import SimpleNamespace

import pandas as pd

from bot.coinbase_client import Account, BestBidAsk
from bot.config import BotConfig
from bot.execution.state_store import BotStateStore
from bot.features.regime import RegimeState
from bot.live.runner import LiveRunner
from bot.strategy.regime_switching_orchestrator import RegimeDecisionBundle


class DummyOrchestrator:
    def compute_target_position(self, timestamp, hourly_df, daily_df, current_exposure):
        return RegimeDecisionBundle(
            macro_risk_on=True,
            macro_reason="test",
            micro_regime=RegimeState.NEUTRAL,
            micro_reason="test",
            strategy_name="test_strategy",
            base_target=0.5,
            regime_multiplier=1.0,
            regime_target=0.5,
            final_target=0.5,
            metadata={},
        )


class RunnerUnderTest(LiveRunner):
    def __init__(self, *args, hourly_df: pd.DataFrame, daily_df: pd.DataFrame, **kwargs):
        self._hourly_df = hourly_df
        self._daily_df = daily_df
        super().__init__(*args, **kwargs)

    def _get_candles(self, product: str, timeframe: str, lookback_hours: int) -> pd.DataFrame:  # type: ignore[override]
        if timeframe == "1h":
            return self._hourly_df.copy()
        return self._daily_df.copy()


class ClientWithBBO:
    def get_accounts(self):
        return [
            Account(uuid="btc", currency="BTC", balance=0.1, available_balance=0.1),
            Account(uuid="usd", currency="USD", balance=1000.0, available_balance=1000.0),
        ]

    def get_best_bid_ask(self, product_id: str):
        return BestBidAsk(bid=100.0, ask=101.0, time=datetime.now(tz=timezone.utc))


class CapturingRouter:
    def __init__(self):
        self.last_target_kwargs = None

    def target_to_order(self, **kwargs):
        self.last_target_kwargs = kwargs
        return []


class RoutingPathCaptureRouter:
    def __init__(self, maker_mode: str = "maker_unfilled"):
        self.maker_mode = maker_mode
        self.maker_first_calls = 0
        self.limit_calls = 0

    def target_to_order(self, **kwargs):
        return [
            SimpleNamespace(
                client_order_id="ord-1",
                product=kwargs["product"],
                side="BUY",
                size=0.01,
                order_type="limit",
                price=100.0,
            )
        ]

    def place_maker_first(self, product: str, side: str, size: float, now):
        self.maker_first_calls += 1
        return {
            "mode": self.maker_mode,
            "order_id": "ord-1" if self.maker_mode != "maker_unfilled" else None,
            "submitted_size": size,
            "response": {"status": "FILLED"} if self.maker_mode == "maker_limit" else None,
        }

    def place_limit_with_fallback(
        self,
        product: str,
        side: str,
        size: float,
        bid: float,
        ask: float,
        now,
        fallback_to_market: bool,
        client_order_id: str,
    ):
        self.limit_calls += 1
        return {
            "mode": "limit",
            "order_id": client_order_id,
            "submitted_size": size,
            "response": {"status": "OPEN"},
        }


class RecordingDailyOrchestrator(DummyOrchestrator):
    def __init__(self):
        self.saw_fred_penalty = False

    def compute_target_position(self, timestamp, hourly_df, daily_df, current_exposure):
        self.saw_fred_penalty = "fred_penalty_multiplier" in daily_df.columns
        return super().compute_target_position(timestamp, hourly_df, daily_df, current_exposure)


class ClientWithOpenOrder:
    def list_orders(self, product_id=None):
        return [{"client_order_id": "order-1", "status": "OPEN", "filled_size": "0"}]


class CancelFailRouter:
    def cancel_order(self, client_order_id: str) -> bool:
        return False

    def make_order_id(self, product: str, side: str, size: float, now):
        return "replacement-id"


def _hourly_daily_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    now = pd.Timestamp.now(tz="UTC").floor("h")
    hourly_ts = [now - pd.Timedelta(hours=2), now - pd.Timedelta(hours=1)]
    hourly = pd.DataFrame(
        {
            "timestamp": hourly_ts,
            "open": [100.0, 100.0],
            "high": [101.0, 101.0],
            "low": [99.0, 99.0],
            "close": [100.0, 100.0],
            "volume": [1.0, 1.0],
        }
    )

    day0 = now.floor("D")
    daily_ts = [day0 - pd.Timedelta(days=3), day0 - pd.Timedelta(days=2), day0 - pd.Timedelta(days=1)]
    daily = pd.DataFrame(
        {
            "timestamp": daily_ts,
            "open": [100.0, 100.0, 100.0],
            "high": [101.0, 101.0, 101.0],
            "low": [99.0, 99.0, 99.0],
            "close": [100.0, 100.0, 100.0],
            "volume": [1.0, 1.0, 1.0],
        }
    )
    return hourly, daily


def test_live_runner_uses_best_bid_ask_for_order_routing(tmp_path):
    hourly, daily = _hourly_daily_frames()

    cfg = BotConfig()
    cfg.data.product = "BTC-USD"

    state = BotStateStore(tmp_path / "state.sqlite")
    runner = RunnerUnderTest(
        cfg,
        client=ClientWithBBO(),
        state_store=state,
        paper=False,
        cycles=1,
        hourly_df=hourly,
        daily_df=daily,
    )
    runner.orchestrator = DummyOrchestrator()
    router = CapturingRouter()
    runner.order_router = router

    _ = runner.step_once(0)

    assert router.last_target_kwargs is not None
    assert router.last_target_kwargs["latest_bid"] == 100.0
    assert router.last_target_kwargs["latest_ask"] == 101.0


def test_reconcile_keeps_order_when_cancel_fails(tmp_path):
    hourly, daily = _hourly_daily_frames()

    cfg = BotConfig()
    cfg.data.product = "BTC-USD"
    cfg.execution.order_timeout_s = 1

    state = BotStateStore(tmp_path / "state.sqlite")
    created_at = int(time.time()) - 120
    state.put_open_order(
        order_id="order-1",
        product="BTC-USD",
        side="BUY",
        size=0.05,
        order_type="limit",
        price=100.0,
        created_at_ts=created_at,
        status="submitted",
        filled_size=0.0,
        metadata={},
    )

    runner = RunnerUnderTest(
        cfg,
        client=ClientWithOpenOrder(),
        state_store=state,
        paper=True,
        cycles=1,
        hourly_df=hourly,
        daily_df=daily,
    )
    runner.order_router = CancelFailRouter()

    runner._reconcile()

    assert state.get_open_order("order-1") is not None
    assert runner.consecutive_order_failures == 1


def test_live_runner_uses_maker_first_path_when_enabled(tmp_path):
    hourly, daily = _hourly_daily_frames()

    cfg = BotConfig()
    cfg.data.product = "BTC-USD"
    cfg.execution.maker_first = True

    state = BotStateStore(tmp_path / "state.sqlite")
    runner = RunnerUnderTest(
        cfg,
        client=ClientWithBBO(),
        state_store=state,
        paper=False,
        cycles=1,
        hourly_df=hourly,
        daily_df=daily,
    )
    runner.orchestrator = DummyOrchestrator()
    router = RoutingPathCaptureRouter(maker_mode="maker_unfilled")
    runner.order_router = router

    decision = runner.step_once(0)

    assert router.maker_first_calls == 1
    assert router.limit_calls == 0
    assert decision.filled is False
    assert state.list_open_orders_dict() == []


def test_live_runner_uses_limit_path_when_maker_first_disabled(tmp_path):
    hourly, daily = _hourly_daily_frames()

    cfg = BotConfig()
    cfg.data.product = "BTC-USD"
    cfg.execution.maker_first = False

    state = BotStateStore(tmp_path / "state.sqlite")
    runner = RunnerUnderTest(
        cfg,
        client=ClientWithBBO(),
        state_store=state,
        paper=False,
        cycles=1,
        hourly_df=hourly,
        daily_df=daily,
    )
    runner.orchestrator = DummyOrchestrator()
    router = RoutingPathCaptureRouter(maker_mode="maker_unfilled")
    runner.order_router = router

    decision = runner.step_once(0)

    assert router.maker_first_calls == 0
    assert router.limit_calls == 1
    assert decision.filled is True
    assert len(state.list_open_orders_dict()) == 1


def test_live_runner_applies_fred_overlay_when_enabled(tmp_path, monkeypatch):
    hourly, daily = _hourly_daily_frames()

    cfg = BotConfig()
    cfg.data.product = "BTC-USD"
    cfg.fred.enabled = True
    cfg.fred.api_key = "dummy"

    def _fake_fred_build(daily_df, fred_cfg):
        out = daily_df.copy()
        out["fred_penalty_multiplier"] = 0.9
        out["fred_risk_off_score"] = 0.2
        out["fred_risk_off_score_smooth"] = 0.2
        return SimpleNamespace(
            daily_features=out,
            report={"enabled": True, "warnings": [], "series_used": ["TEST"]},
        )

    monkeypatch.setattr("bot.live.runner.build_fred_daily_overlay_features", _fake_fred_build)

    state = BotStateStore(tmp_path / "state.sqlite")
    runner = RunnerUnderTest(
        cfg,
        client=ClientWithBBO(),
        state_store=state,
        paper=True,
        cycles=1,
        hourly_df=hourly,
        daily_df=daily,
    )
    rec = RecordingDailyOrchestrator()
    runner.orchestrator = rec

    _ = runner.step_once(0)

    assert rec.saw_fred_penalty is True
    assert runner._last_fred_report.get("enabled") is True
