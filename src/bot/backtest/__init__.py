"""Backtesting package for trading_bot_btc."""

from .engine import BacktestEngine, BacktestResult  # noqa: F401
from .metrics import compute_metrics  # noqa: F401
from .metrics import bootstrap_sharpe_confidence, SharpeBootstrapResult  # noqa: F401
from .monte_carlo import run_monte_carlo, MonteCarloConfig, MonteCarloResult  # noqa: F401
from .cpcv import run_cpcv, CPCVConfig, CPCVResult  # noqa: F401
from .cost_sensitivity import run_cost_sensitivity, CostSensitivityConfig, CostSensitivityResult  # noqa: F401
from .walkforward import walk_forward_test, choose_robust_parameter_set  # noqa: F401
from .regime_reports import performance_by_regime, time_in_regime, regime_switch_count  # noqa: F401
