#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import statistics
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Make local src discoverable when running directly from the repository root
sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

from bot.config import BacktestConfig
from bot.acceleration.cuda_backend import resolve_acceleration_backend
from bot.system_log import setup_system_logger, get_system_logger

ROOT_DIR = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = ROOT_DIR / "scripts"
REPORTS_BASE_DIR = ROOT_DIR / "reports" / "frontier_sweep_all_strategies"

StrategyRunner = tuple[str, str]

ACTIVE_STRATEGIES: list[str] = sorted(BacktestConfig.VALID_STRATEGIES)

RUNNERS: dict[str, StrategyRunner] = {
    "macro_gate_benchmark": ("frontier_sweep.py", "macro_gate_benchmark"),
    "macro_only_v2": ("frontier_sweep_macro_only.py", "macro_only_v2"),
    "v5_adaptive": ("frontier_sweep_v5.py", "v5_adaptive"),
    "regime_switching_v4_core": ("frontier_sweep_core.py", "regime_switching_v4_core"),
    "regime_switching_orchestrator": ("frontier_sweep_v3.py", "regime_switching_orchestrator"),
    "macro_gate_state": ("frontier_sweep_v3.py", "macro_gate_state"),
}

# Strategy-level caps that keep full multi-strategy sweeps within a practical 24h budget
# on a single machine while preserving broad parameter-space coverage.
DEFAULT_FAST_24H_MAX_PARAM_SETS: dict[str, int] = {
    "macro_gate_benchmark": 600,
    "macro_gate_state": 1200,
    "regime_switching_orchestrator": 1200,
    "regime_switching_v4_core": 900,
    "v5_adaptive": 900,
    "macro_only_v2": 800,
}

logger = get_system_logger("scripts.frontier_sweep_all_strategies")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run frontier sweeps for all active strategies over the same date windows."
    )
    p.add_argument("--product", default="BTC-USD")
    p.add_argument("--start", default="2024-01-01T00:00:00Z")
    p.add_argument("--end", default="2026-02-16T23:00:00Z")
    p.add_argument("--train-start", default="2024-01-01T00:00:00Z")
    p.add_argument("--train-end", default="2024-12-31T23:00:00Z")
    p.add_argument("--val-start", default="2025-01-01T00:00:00Z")
    p.add_argument("--val-end", default="2025-12-31T23:00:00Z")
    p.add_argument("--test-start", default="2026-01-01T00:00:00Z")
    p.add_argument("--test-end", default="2026-02-16T23:00:00Z")
    p.add_argument("--fill-model", default="bid_ask", choices=["next_open", "bid_ask", "worst_case_bar"])
    p.add_argument("--acceleration-backend", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--config", default=None)
    p.add_argument("--grid-config", default=None, help="JSON file: either {param:[...]} or [{...}, ...]")
    p.add_argument("--grid", action="append", default=[], help="Repeatable KEY=v1,v2 override")
    p.add_argument("--include-fred-grid", action="store_true", help="include FRED overlay dimensions in sweep grid")
    p.add_argument("--small", action="store_true", help="Use reduced sweep grids for quick checks (where supported)")
    p.add_argument("--workers", type=int, default=20, help="Workers for macro-only frontier sweep")
    p.add_argument("--checkpoint-every", type=int, default=10, help="Checkpoint interval for strategy sweeps")
    p.add_argument("--run-id", default=None, help="Optional shared run id for per-strategy output directories")
    p.add_argument("--resume", action="store_true", help="Resume an existing run-id across all strategies")
    p.add_argument(
        "--strategy-workers",
        type=int,
        default=0,
        help="How many strategies to run concurrently (0=auto, default)",
    )
    p.add_argument("--top-n", type=int, default=5)
    p.add_argument("--turnover-max", type=float, default=700.0)
    p.add_argument("--max-drawdown-max", type=float, default=0.30)
    p.add_argument("--strategies", default=",".join(ACTIVE_STRATEGIES), help="Comma-separated strategies to run")
    p.add_argument(
        "--include-macro-only-baseline",
        action="store_true",
        help="Include benchmark baseline rows inside macro_only_v2 sweep (disabled by default when orchestrated)",
    )
    p.add_argument("--maker-bps", type=float, default=10.0)
    p.add_argument("--taker-bps", type=float, default=25.0)
    p.add_argument("--output-dir", default="artifacts/frontier_all_strategies", help="Output root directory")
    p.add_argument(
        "--timeout-seconds",
        type=int,
        default=21600,
        help="Timeout for each strategy run in seconds (default: 6 hours)",
    )
    p.add_argument(
        "--max-error-rate",
        type=float,
        default=0.05,
        help="Mark strategy run as failed if summary.csv error-rate exceeds this threshold",
    )
    p.add_argument(
        "--ranking-mode",
        choices=["absolute", "vs_benchmark"],
        default="vs_benchmark",
        help="How to choose best strategy across runs",
    )
    p.add_argument(
        "--fast-24h",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable 24h-oriented sweep defaults (reduced windows/scenarios + strategy caps)",
    )
    p.add_argument(
        "--max-param-sets",
        type=int,
        default=0,
        help="Global max param sets per strategy (0=disabled; strategy overrides still apply)",
    )
    p.add_argument(
        "--strategy-max-param-sets",
        default="",
        help="Comma-separated per-strategy cap overrides, e.g. macro_gate_state=1500,v5_adaptive=1200",
    )
    p.add_argument(
        "--search-seed",
        type=int,
        default=42,
        help="Deterministic seed for capped-grid sampling",
    )
    p.add_argument(
        "--optimize-windows",
        default=None,
        help="Comma-separated windows passed to sweeps (default under --fast-24h: val,test)",
    )
    p.add_argument(
        "--optimize-scenarios",
        default=None,
        help="Comma-separated scenarios passed to sweeps (default under --fast-24h: baseline,stress_1)",
    )
    return p.parse_args()


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _run_reports_dir(run_id: str | None = None) -> Path:
    REPORTS_BASE_DIR.mkdir(parents=True, exist_ok=True)
    raw = (run_id or "").strip()
    if raw:
        token = raw if raw.startswith("run_") else f"run_{raw}"
        return REPORTS_BASE_DIR / token
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    return REPORTS_BASE_DIR / f"run_{timestamp}"


def _write_progress(progress_file: Path, event: dict[str, Any]) -> None:
    payload = {"timestamp": _utc_timestamp(), **event}
    progress_file.parent.mkdir(parents=True, exist_ok=True)
    with progress_file.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload) + "\n")
    logger.debug("progress_event path=%s payload=%s", progress_file, payload)


def _run_command(cmd: list[str], timeout_seconds: int) -> subprocess.CompletedProcess:
    logger.info("subprocess_start timeout=%ss cmd=%s", timeout_seconds, " ".join(cmd))
    try:
        completed = subprocess.run(
            cmd,
            cwd=str(ROOT_DIR),
            text=True,
            check=False,
            timeout=timeout_seconds,
        )
        logger.info(
            "subprocess_done rc=%s cmd=%s",
            completed.returncode,
            " ".join(cmd),
        )
        if completed.stdout:
            logger.debug("subprocess_stdout cmd=%s stdout=%s", " ".join(cmd), completed.stdout[-4000:])
        if completed.stderr:
            logger.debug("subprocess_stderr cmd=%s stderr=%s", " ".join(cmd), completed.stderr[-4000:])
        return completed
    except subprocess.TimeoutExpired as exc:
        logger.exception("subprocess_timeout timeout=%ss cmd=%s", timeout_seconds, " ".join(cmd))
        return subprocess.CompletedProcess(
            args=cmd,
            returncode=124,
            stdout=(exc.stdout or "") if exc.stdout else "",
            stderr=(exc.stderr or "") + f"\nCommand timed out after {timeout_seconds}s\n",
        )


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value if value is not None else default)
    except (TypeError, ValueError):
        return default


def _parse_strategy_cap_overrides(raw: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for chunk in [c.strip() for c in str(raw or "").split(",") if c.strip()]:
        if "=" in chunk:
            key, val = chunk.split("=", 1)
        elif ":" in chunk:
            key, val = chunk.split(":", 1)
        else:
            continue
        key = key.strip()
        if key not in RUNNERS:
            continue
        try:
            parsed = int(val.strip())
        except ValueError:
            continue
        if parsed > 0:
            out[key] = parsed
    return out


def _resolve_effective_windows(args: argparse.Namespace) -> str:
    if args.optimize_windows:
        return str(args.optimize_windows)
    return "val,test" if bool(args.fast_24h) else "train,val,test"


def _resolve_effective_scenarios(args: argparse.Namespace) -> str:
    if args.optimize_scenarios:
        return str(args.optimize_scenarios)
    return "baseline,stress_1" if bool(args.fast_24h) else "baseline,stress_1,stress_2"


def _resolve_strategy_max_param_sets(
    args: argparse.Namespace,
    strategy: str,
    overrides: dict[str, int],
) -> int:
    if strategy in overrides:
        return int(overrides[strategy])
    if int(args.max_param_sets or 0) > 0:
        return int(args.max_param_sets)
    if bool(args.fast_24h):
        return int(DEFAULT_FAST_24H_MAX_PARAM_SETS.get(strategy, 0) or 0)
    return 0


def _build_base_args(args: argparse.Namespace) -> list[str]:
    base_args = [
        "--product",
        args.product,
        "--start",
        args.start,
        "--train-start",
        args.train_start,
        "--train-end",
        args.train_end,
        "--val-start",
        args.val_start,
        "--val-end",
        args.val_end,
        "--test-start",
        args.test_start,
        "--fill-model",
        args.fill_model,
        "--acceleration-backend",
        args.acceleration_backend,
        "--top-n",
        str(args.top_n),
        "--turnover-max",
        str(args.turnover_max),
        "--max-drawdown-max",
        str(args.max_drawdown_max),
        "--maker-bps",
        str(args.maker_bps),
        "--taker-bps",
        str(args.taker_bps),
    ]

    if args.config:
        base_args.extend(["--config", args.config])
    if args.grid_config:
        base_args.extend(["--grid-config", args.grid_config])
    if args.grid:
        for raw in args.grid:
            base_args.extend(["--grid", raw])

    effective_windows = _resolve_effective_windows(args)
    effective_scenarios = _resolve_effective_scenarios(args)
    if effective_windows:
        base_args.extend(["--optimize-windows", effective_windows])
    if effective_scenarios:
        base_args.extend(["--optimize-scenarios", effective_scenarios])
    base_args.extend(["--search-seed", str(int(args.search_seed))])

    return base_args


def _validate_acceleration_backend(requested: str) -> bool:
    ctx = resolve_acceleration_backend(requested)
    if requested == "cuda" and ctx.backend != "cuda":
        print(
            f"ERROR: --acceleration-backend=cuda requested, but CUDA is unavailable ({ctx.reason or 'unknown reason'}).",
            file=sys.stderr,
        )
        return False
    if requested in {"auto", "cuda"}:
        detail = ctx.device_name if ctx.device_name else (ctx.reason or "")
        print(f"Acceleration backend resolved: {ctx.backend}{f' ({detail})' if detail else ''}")
    return True


def _resolve_strategy_workers(args: argparse.Namespace, selected_count: int) -> int:
    selected_count = max(1, int(selected_count))
    requested = int(args.strategy_workers)

    if requested <= 0:
        cpu = max(1, int(os.cpu_count() or 1))
        # Keep strategy fan-out conservative by default; each strategy may also parallelize internally.
        auto_workers = max(1, min(selected_count, max(1, cpu // 2)))
        workers = auto_workers
    else:
        workers = min(max(1, requested), selected_count)

    ctx = resolve_acceleration_backend(args.acceleration_backend)
    if ctx.backend == "cuda" and workers > 1:
        print("CUDA backend detected; forcing --strategy-workers=1 to avoid GPU contention.")
        return 1
    return workers


def _build_strategy_command(
    args: argparse.Namespace,
    strategy: str,
    baseline_args: list[str],
    strategy_dir: Path,
    run_id: str,
    max_param_sets: int,
) -> tuple[list[str], str]:
    script_name, _default_tag = RUNNERS[strategy]
    args_for_strategy = baseline_args.copy()
    args_for_strategy.extend([
        "--output-dir",
        str(strategy_dir),
        "--run-id",
        run_id,
        "--checkpoint-every",
        str(args.checkpoint_every),
    ])
    if int(max_param_sets) > 0:
        args_for_strategy.extend(["--max-param-sets", str(int(max_param_sets))])
    if args.resume:
        args_for_strategy.append("--resume")

    cmd = [sys.executable, str(SCRIPTS_DIR / script_name)]

    if script_name == "frontier_sweep.py":
        cmd.extend(["--strategy", strategy])
        cmd.extend(["--end", args.end])
        cmd.extend(["--test-end", args.test_end])
        if args.small:
            cmd.append("--small")
        if args.include_fred_grid:
            cmd.append("--include-fred-grid")
        cmd.extend(args_for_strategy)
        return cmd, script_name

    if script_name == "frontier_sweep_macro_only.py":
        cmd.extend(["--test-end", args.test_end])
        cmd.extend(["--workers", str(args.workers)])
        if args.small:
            cmd.append("--small")
        if args.include_fred_grid:
            cmd.append("--include-fred-grid")
        if not args.include_macro_only_baseline:
            cmd.append("--skip-benchmark-baseline")
        cmd.extend(args_for_strategy)
        return cmd, script_name

    if script_name in {"frontier_sweep_v3.py", "frontier_sweep_core.py", "frontier_sweep_v5.py"}:
        cmd.extend(["--end", args.end])
        cmd.extend(["--test-end", args.test_end])
        if script_name == "frontier_sweep_v3.py":
            cmd.extend(["--strategy", strategy])
        if args.small:
            cmd.append("--small")
        cmd.extend(args_for_strategy)
        return cmd, script_name

    # Fallback for future runners.
    cmd.extend(args_for_strategy)
    return cmd, script_name


def _load_best_summary(summary_path: Path, strategy_dir: Path, strategy: str) -> dict[str, Any] | None:
    if summary_path.exists():
        with summary_path.open("r", encoding="utf-8") as f:
            return json.load(f)

    frontier_path = strategy_dir / "frontier.csv"
    if not frontier_path.exists():
        return None

    with frontier_path.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None

    top = rows[0]

    cagr_source = top.get(
        "test_cagr_stress_1",
        top.get("val_stress1_cagr", top.get("val_score", top.get("cagr", 0.0))),
    )
    sharpe_source = top.get(
        "test_sharpe_stress_1",
        top.get("val_sharpe_stress_1", top.get("val_stress1_sharpe", top.get("sharpe", 0.0))),
    )
    max_drawdown_source = top.get(
        "test_max_drawdown_stress_1",
        top.get("val_max_drawdown_worst", top.get("val_stress1_max_drawdown", top.get("max_drawdown", 0.0))),
    )
    turnover_source = top.get(
        "test_turnover_stress_1",
        top.get("val_turnover_worst", top.get("val_stress1_turnover", top.get("turnover", 0.0))),
    )

    cagr = _to_float(cagr_source)
    sharpe = _to_float(sharpe_source)
    max_drawdown = _to_float(max_drawdown_source)
    turnover = _to_float(turnover_source)

    # Normalize to best_summary-like structure for downstream comparisons.
    return {
        "strategy": strategy,
        "best": top,
        "test_window_stress_1": {
            "cagr": cagr,
            "sharpe": sharpe,
            "max_drawdown": max_drawdown,
            "trade_count": top.get("trade_count") or top.get("val_trade_count", 0),
            "turnover": turnover,
        },
        "best_config": {},
    }

def _extract_score(summary: dict[str, Any]) -> tuple[float, float, float, float]:
    test = summary.get("test_window_stress_1") if isinstance(summary, dict) else None
    if isinstance(test, dict):
        cagr = float(test.get("cagr", 0.0) or 0.0)
        sharpe = float(test.get("sharpe", 0.0) or 0.0)
        drawdown = float(test.get("max_drawdown", 0.0) or 0.0)
        turnover = float(test.get("turnover", 0.0) or 0.0)
        return (cagr, sharpe, drawdown, turnover)

    best = summary.get("best", {}) if isinstance(summary, dict) else {}
    cagr = float(
        best.get("test_cagr_stress_1", best.get("val_stress1_cagr", best.get("val_score", 0.0))) or 0.0
    )
    sharpe = float(
        best.get(
            "test_sharpe_stress_1",
            best.get("val_sharpe_stress_1", best.get("val_stress1_sharpe", best.get("sharpe", 0.0))),
        )
        or 0.0
    )
    drawdown = float(
        best.get(
            "test_max_drawdown_stress_1",
            best.get("val_max_drawdown_worst", best.get("val_stress1_max_drawdown", best.get("max_drawdown", 0.0))),
        )
        or 0.0
    )
    turnover = float(
        best.get(
            "test_turnover_stress_1",
            best.get("val_turnover_worst", best.get("val_stress1_turnover", best.get("val_turnover", best.get("turnover", 0.0)))),
        )
        or 0.0
    )
    return (cagr, sharpe, drawdown, turnover)


def _extract_validation_score(summary: dict[str, Any]) -> tuple[float, float, float, float]:
    best = summary.get("best", {}) if isinstance(summary, dict) else {}
    cagr = float(best.get("val_cagr_stress_1", best.get("val_stress1_cagr", best.get("val_score", 0.0))) or 0.0)
    sharpe = float(
        best.get(
            "val_sharpe_stress_1",
            best.get("val_stress1_sharpe", best.get("val_sharpe_med", best.get("sharpe", 0.0))),
        )
        or 0.0
    )
    drawdown = float(
        best.get(
            "val_max_drawdown_worst",
            best.get("val_stress1_max_drawdown", best.get("max_drawdown", 0.0)),
        )
        or 0.0
    )
    turnover = float(
        best.get(
            "val_turnover_worst",
            best.get("val_stress1_turnover", best.get("val_turnover", best.get("turnover", 0.0))),
        )
        or 0.0
    )
    return (cagr, sharpe, drawdown, turnover)


def _absolute_sort_key(score: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    return (score[0], score[1], score[2], -score[3])


def _multifold_robust_score(strategy_dir: Path, summary: dict[str, Any]) -> dict[str, Any] | None:
    best = summary.get("best") if isinstance(summary, dict) else None
    if not isinstance(best, dict):
        return None

    param_id = str(best.get("param_id", "") or "").strip()
    if not param_id:
        return None

    summary_csv = strategy_dir / "summary.csv"
    if not summary_csv.exists():
        return None

    try:
        with summary_csv.open("r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
    except Exception:
        return None

    filtered = []
    for row in rows:
        if str(row.get("param_id", "") or "").strip() != param_id:
            continue
        if str(row.get("error", "") or "").strip():
            continue
        filtered.append(row)

    if not filtered:
        return None

    by_window: dict[str, list[dict[str, Any]]] = {}
    for row in filtered:
        window = str(row.get("window", "") or "").strip() or "unknown"
        by_window.setdefault(window, []).append(row)

    fold_cagr: list[float] = []
    fold_sharpe: list[float] = []
    scenario_spreads: list[float] = []
    all_drawdown: list[float] = []
    all_turnover: list[float] = []

    for _window, entries in sorted(by_window.items()):
        cagr_vals = [_to_float(r.get("cagr", 0.0)) for r in entries]
        sharpe_vals = [_to_float(r.get("sharpe", 0.0)) for r in entries]
        draw_vals = [_to_float(r.get("max_drawdown", 0.0)) for r in entries]
        turnover_vals = [_to_float(r.get("turnover", 0.0)) for r in entries]

        if not cagr_vals:
            continue

        fold_cagr.append(float(statistics.mean(cagr_vals)))
        fold_sharpe.append(float(statistics.mean(sharpe_vals)))
        scenario_spreads.append(float(statistics.pstdev(cagr_vals)) if len(cagr_vals) > 1 else 0.0)

        all_drawdown.extend(draw_vals)
        all_turnover.extend(turnover_vals)

    if not fold_cagr:
        return None

    fold_mean_cagr = float(statistics.mean(fold_cagr))
    fold_std_cagr = float(statistics.pstdev(fold_cagr)) if len(fold_cagr) > 1 else 0.0
    fold_mean_sharpe = float(statistics.mean(fold_sharpe))
    fold_std_sharpe = float(statistics.pstdev(fold_sharpe)) if len(fold_sharpe) > 1 else 0.0
    scenario_penalty = float(statistics.mean(scenario_spreads)) if scenario_spreads else 0.0

    robust_cagr = fold_mean_cagr - fold_std_cagr - scenario_penalty
    robust_sharpe = fold_mean_sharpe - fold_std_sharpe
    worst_drawdown = float(min(all_drawdown)) if all_drawdown else 0.0
    avg_turnover = float(statistics.mean(all_turnover)) if all_turnover else 0.0

    return {
        "score": (robust_cagr, robust_sharpe, worst_drawdown, avg_turnover),
        "fold_count": len(fold_cagr),
        "scenario_count": len(filtered),
        "fold_cagr_mean": fold_mean_cagr,
        "fold_cagr_std": fold_std_cagr,
        "scenario_cagr_penalty": scenario_penalty,
    }


def _score_payload(score: tuple[float, float, float, float]) -> dict[str, float]:
    return {
        "cagr": float(score[0]),
        "sharpe": float(score[1]),
        "max_drawdown": float(score[2]),
        "turnover": float(score[3]),
    }


def _write_latest_successful_pointer(
    *,
    run_reports_dir: Path,
    summary_path: Path,
    final_report_path: Path,
    report: dict[str, Any],
) -> None:
    REPORTS_BASE_DIR.mkdir(parents=True, exist_ok=True)
    pointer_payload = {
        "run_id": run_reports_dir.name,
        "updated_at": _utc_timestamp(),
        "report_path": str(final_report_path),
        "summary_path": str(summary_path),
        "validation_winner": (report.get("validation_winner") or {}).get("strategy"),
        "oos_winner": (report.get("oos_winner") or {}).get("strategy"),
    }
    (REPORTS_BASE_DIR / "latest_successful_run.json").write_text(
        json.dumps(pointer_payload, indent=2),
        encoding="utf-8",
    )
    (REPORTS_BASE_DIR / "latest_successful_run.txt").write_text(
        str(final_report_path) + "\n",
        encoding="utf-8",
    )


def _summary_error_stats(strategy_dir: Path) -> tuple[int, int, float]:
    summary_path = strategy_dir / "summary.csv"
    if not summary_path.exists():
        return (0, 0, 0.0)

    try:
        with summary_path.open("r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
    except Exception:
        return (0, 0, 0.0)

    total = len(rows)
    if total <= 0:
        return (0, 0, 0.0)

    errors = 0
    for row in rows:
        err = row.get("error") if isinstance(row, dict) else None
        if err is not None and str(err).strip() != "":
            errors += 1

    rate = float(errors) / float(total)
    return (errors, total, rate)


def _execute_strategy_job(
    strategy: str,
    script_name: str,
    strategy_dir: Path,
    cmd: list[str],
    timeout_seconds: int,
    max_error_rate: float,
) -> dict[str, Any]:
    started_at = datetime.now(timezone.utc)
    logger.info(
        "strategy_job_start strategy=%s runner=%s timeout=%s output_dir=%s",
        strategy,
        script_name,
        timeout_seconds,
        strategy_dir,
    )
    completed = _run_command(cmd, timeout_seconds=timeout_seconds)
    finished_at = datetime.now(timezone.utc)
    elapsed_sec = (finished_at - started_at).total_seconds()

    summary = None
    effective_return_code = int(completed.returncode)

    if completed.returncode == 0:
        err_rows, total_rows, error_rate = _summary_error_stats(strategy_dir)
        if total_rows > 0 and error_rate > float(max_error_rate):
            effective_return_code = 65
            print(
                f"Strategy {strategy} exceeded max error rate: "
                f"{err_rows}/{total_rows} ({error_rate:.1%}) > {max_error_rate:.1%}"
            )

        summary = _load_best_summary(
            strategy_dir / "best_summary.json",
            strategy_dir=strategy_dir,
            strategy=strategy,
        )
        if summary is None:
            print(f"Warning: no best_summary.json found for {strategy}. Parsing frontier.csv fallback")
    else:
        err_rows, total_rows, error_rate = _summary_error_stats(strategy_dir)
        print(f"Strategy {strategy} failed with exit code {completed.returncode}")

    logger.info(
        "strategy_job_done strategy=%s runner=%s effective_rc=%s raw_rc=%s duration=%.2fs error_rows=%s total_rows=%s error_rate=%.3f summary=%s",
        strategy,
        script_name,
        effective_return_code,
        int(completed.returncode),
        elapsed_sec,
        err_rows,
        total_rows,
        error_rate,
        summary is not None,
    )

    return {
        "strategy": strategy,
        "script": script_name,
        "return_code": effective_return_code,
        "raw_return_code": int(completed.returncode),
        "output_dir": str(strategy_dir),
        "summary": summary,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "duration_seconds": elapsed_sec,
        "start_time": started_at.isoformat(),
        "end_time": finished_at.isoformat(),
        "error_rows": int(err_rows),
        "total_rows": int(total_rows),
        "error_rate": float(error_rate),
    }



def _run_all_strategies(args: argparse.Namespace) -> tuple[list[dict[str, Any]], Path, Path]:
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    run_reports_dir = _run_reports_dir(args.run_id)
    run_reports_dir.mkdir(parents=True, exist_ok=True)
    progress_path = run_reports_dir / "progress.jsonl"
    run_id = run_reports_dir.name

    selected = [s.strip() for s in args.strategies.split(",") if s.strip()]
    unknown = [s for s in selected if s not in RUNNERS]
    if unknown:
        raise ValueError(
            f"Unknown strategy requested: {', '.join(unknown)}. Active options: {', '.join(ACTIVE_STRATEGIES)}"
        )

    equivalence_key = {
        "macro_gate_state": "regime_switching_orchestrator",
        "regime_switching_orchestrator": "regime_switching_orchestrator",
    }
    deduped: list[str] = []
    seen_keys: set[str] = set()
    for strategy in selected:
        key = equivalence_key.get(strategy, strategy)
        if key in seen_keys:
            print(f"Skipping duplicate-equivalent strategy: {strategy} (equivalent to {key})")
            continue
        seen_keys.add(key)
        deduped.append(strategy)
    selected = deduped

    strategy_workers = _resolve_strategy_workers(args, len(selected))
    strategy_cap_overrides = _parse_strategy_cap_overrides(args.strategy_max_param_sets)

    _write_progress(
        progress_path,
        {
            "event": "run_started",
            "run_id": run_id,
            "output_root": str(output_root),
            "strategies": selected,
            "args": vars(args),
            "strategy_workers": strategy_workers,
            "effective_optimize_windows": _resolve_effective_windows(args),
            "effective_optimize_scenarios": _resolve_effective_scenarios(args),
            "strategy_cap_overrides": strategy_cap_overrides,
            "fast_24h": bool(args.fast_24h),
        },
    )

    baseline_args = _build_base_args(args)

    jobs: list[dict[str, Any]] = []
    for index, strategy in enumerate(selected, start=1):
        _script, default_tag = RUNNERS[strategy]
        strategy_base_dir = output_root / default_tag
        strategy_base_dir.mkdir(parents=True, exist_ok=True)
        strategy_run_dir = strategy_base_dir / run_id

        strategy_max_param_sets = _resolve_strategy_max_param_sets(
            args=args,
            strategy=strategy,
            overrides=strategy_cap_overrides,
        )

        cmd, script_name = _build_strategy_command(
            args=args,
            strategy=strategy,
            baseline_args=baseline_args,
            strategy_dir=strategy_base_dir,
            run_id=run_id,
            max_param_sets=strategy_max_param_sets,
        )
        command = " ".join(cmd)

        jobs.append(
            {
                "index": index,
                "strategy": strategy,
                "script_name": script_name,
                "strategy_dir": strategy_run_dir,
                "strategy_base_dir": strategy_base_dir,
                "cmd": cmd,
                "command": command,
                "max_param_sets": int(strategy_max_param_sets),
            }
        )

        _write_progress(
            progress_path,
            {
                "event": "strategy_started",
                "run_id": run_id,
                "strategy": strategy,
                "strategy_index": index,
                "command": command,
                "output_dir": str(strategy_run_dir),
                "max_param_sets": int(strategy_max_param_sets),
            },
        )

    results: list[dict[str, Any]] = []

    if strategy_workers <= 1:
        for job in jobs:
            print(f"\n=== Running frontier sweep: {job['strategy']} ===")
            print("Command:", job["command"])
            result = _execute_strategy_job(
                strategy=job["strategy"],
                script_name=job["script_name"],
                strategy_dir=job["strategy_dir"],
                cmd=job["cmd"],
                timeout_seconds=args.timeout_seconds,
                max_error_rate=args.max_error_rate,
            )
            result["strategy_index"] = job["index"]
            results.append(result)

            _write_progress(
                progress_path,
                {
                    "event": "strategy_completed",
                    "run_id": run_id,
                    "strategy": job["strategy"],
                    "strategy_index": job["index"],
                    "return_code": result["return_code"],
                    "raw_return_code": result.get("raw_return_code"),
                    "output_dir": str(job["strategy_dir"]),
                    "duration_seconds": result["duration_seconds"],
                    "error_rows": result.get("error_rows", 0),
                    "total_rows": result.get("total_rows", 0),
                    "error_rate": result.get("error_rate", 0.0),
                    "summary": {"available": result["summary"] is not None},
                    "max_param_sets": int(job.get("max_param_sets", 0) or 0),
                },
            )
    else:
        with ThreadPoolExecutor(max_workers=strategy_workers) as pool:
            fut_to_job = {
                pool.submit(
                    _execute_strategy_job,
                    strategy=job["strategy"],
                    script_name=job["script_name"],
                    strategy_dir=job["strategy_dir"],
                    cmd=job["cmd"],
                    timeout_seconds=args.timeout_seconds,
                    max_error_rate=args.max_error_rate,
                ): job
                for job in jobs
            }

            for fut in as_completed(fut_to_job):
                job = fut_to_job[fut]
                result = fut.result()
                result["strategy_index"] = job["index"]
                results.append(result)

                _write_progress(
                    progress_path,
                    {
                        "event": "strategy_completed",
                        "run_id": run_id,
                        "strategy": job["strategy"],
                        "strategy_index": job["index"],
                        "return_code": result["return_code"],
                        "raw_return_code": result.get("raw_return_code"),
                        "output_dir": str(job["strategy_dir"]),
                        "duration_seconds": result["duration_seconds"],
                        "error_rows": result.get("error_rows", 0),
                        "total_rows": result.get("total_rows", 0),
                        "error_rate": result.get("error_rate", 0.0),
                        "summary": {"available": result["summary"] is not None},
                    "max_param_sets": int(job.get("max_param_sets", 0) or 0),
                    },
                )

    results.sort(key=lambda r: int(r.get("strategy_index", 0)))

    run_complete_payload = {
        "event": "run_completed",
        "run_id": run_id,
        "strategies_completed": len(results),
        "output_root": str(output_root),
        "reports_dir": str(run_reports_dir),
    }
    _write_progress(progress_path, run_complete_payload)

    return results, run_reports_dir, progress_path


def _summarize(
    results: list[dict[str, Any]],
    output_root: Path,
    run_reports_dir: Path,
    progress_file: Path,
    ranking_mode: str,
) -> dict[str, Any] | None:
    print("\n=== Frontier sweep summary (all active strategies) ===")

    oos_scores: dict[str, tuple[float, float, float, float]] = {}
    validation_scores: dict[str, tuple[float, float, float, float]] = {}
    ranking_scores: dict[str, tuple[float, float, float, float]] = {}
    ranking_basis: dict[str, str] = {}
    multifold_details: dict[str, dict[str, Any]] = {}
    valid_results: list[dict[str, Any]] = []

    for result in results:
        strategy = result["strategy"]
        summary = result["summary"]
        rc = result["return_code"]
        err_rows = int(result.get("error_rows", 0) or 0)
        total_rows = int(result.get("total_rows", 0) or 0)
        err_rate = float(result.get("error_rate", 0.0) or 0.0)
        print(f"\n- {strategy}: exit={rc}, output={result['output_dir']}")

        if total_rows > 0:
            print(f"  Error rows: {err_rows}/{total_rows} ({err_rate:.1%})")

        if rc != 0:
            print("  Status: failed")
            continue
        if not summary:
            print("  Status: completed (no best summary detected)")
            continue

        best_row = summary.get("best") if isinstance(summary, dict) else None
        if best_row in (None, {}, []):
            print("  Status: completed (no winning config)")
            continue

        oos_score = _extract_score(summary)
        val_score = _extract_validation_score(summary)
        robust = _multifold_robust_score(Path(result["output_dir"]), summary)

        rank_score = oos_score
        basis = "oos_stress1"
        if robust and isinstance(robust.get("score"), tuple):
            rank_score = robust["score"]
            basis = "multifold_robust"
            multifold_details[strategy] = robust

        oos_scores[strategy] = oos_score
        validation_scores[strategy] = val_score
        ranking_scores[strategy] = rank_score
        ranking_basis[strategy] = basis
        valid_results.append(result)

        print("  Status: success")
        print(
            f"  Validation score => cagr: {val_score[0]:.6f}, sharpe: {val_score[1]:.6f}, "
            f"drawdown: {val_score[2]:.6f}, turnover: {val_score[3]:.6f}"
        )
        print(
            f"  OOS score => cagr: {oos_score[0]:.6f}, sharpe: {oos_score[1]:.6f}, "
            f"drawdown: {oos_score[2]:.6f}, turnover: {oos_score[3]:.6f}"
        )
        if robust:
            print(
                f"  Multifold robust score => cagr: {rank_score[0]:.6f}, sharpe: {rank_score[1]:.6f}, "
                f"drawdown: {rank_score[2]:.6f}, turnover: {rank_score[3]:.6f} "
                f"(folds={robust.get('fold_count', 0)}, scenario_rows={robust.get('scenario_count', 0)})"
            )

    report: dict[str, Any] = {
        "results": results,
        "best": None,
        "validation_winner": None,
        "oos_winner": None,
        "output_root": str(output_root),
        "reports_dir": str(run_reports_dir),
        "ranking_mode": ranking_mode,
        "ranking_basis": ranking_basis,
    }

    if not valid_results:
        print("\nNo successful strategy run produced a best_summary result.")
    else:
        # Validation winner (absolute, independent of benchmark-relative ranking).
        validation_entry = max(
            valid_results,
            key=lambda r: _absolute_sort_key(validation_scores.get(r["strategy"], (0.0, 0.0, 0.0, 0.0))),
        )
        validation_strategy = validation_entry["strategy"]
        validation_summary = validation_entry["summary"] if isinstance(validation_entry.get("summary"), dict) else {}
        validation_report = {
            "strategy": validation_strategy,
            "score": _score_payload(validation_scores[validation_strategy]),
            "config": validation_summary.get("best_config") or validation_summary.get("best_cfg") or {},
            "output_dir": validation_entry["output_dir"],
        }
        report["validation_winner"] = validation_report

        print("\n=== Validation winner ===")
        print(json.dumps(validation_report, indent=2))

        benchmark_score = ranking_scores.get("macro_gate_benchmark")
        best_entry: dict[str, Any] | None = None
        best_sort_key: tuple[float, float, float, float] = (-1e18, -1e18, -1e18, -1e18)

        candidate_results = list(valid_results)
        if ranking_mode == "vs_benchmark" and benchmark_score is not None:
            candidate_results = [
                r for r in valid_results if str(r.get("strategy")) != "macro_gate_benchmark"
            ]
            if not candidate_results:
                print("\nNo non-benchmark strategy produced a comparable best summary.")

        for result in candidate_results:
            strategy = result["strategy"]
            score = ranking_scores.get(strategy)
            if score is None:
                continue

            if ranking_mode == "vs_benchmark" and benchmark_score is not None:
                sort_key = (
                    score[0] - benchmark_score[0],
                    score[1] - benchmark_score[1],
                    abs(benchmark_score[2]) - abs(score[2]),
                    benchmark_score[3] - score[3],
                )
                print(
                    f"  Δ vs benchmark [{strategy}]: "
                    f"cagr={sort_key[0]:+.6f}, sharpe={sort_key[1]:+.6f}, "
                    f"drawdown={sort_key[2]:+.6f}, turnover={sort_key[3]:+.6f}"
                )
                if sort_key[0] <= 0.0:
                    print(
                        f"  Skipping [{strategy}] due to non-positive Δcagr vs benchmark ({sort_key[0]:+.6f})"
                    )
                    continue
            else:
                sort_key = _absolute_sort_key(score)

            if sort_key > best_sort_key:
                best_sort_key = sort_key
                best_entry = result

        if best_entry is None:
            print("\nCould not determine best strategy from successful results.")
        else:
            best_summary = best_entry["summary"]
            if not isinstance(best_summary, dict):
                print("\nCould not parse best strategy details for final report.")
            else:
                strategy = best_entry["strategy"]
                selected_score = ranking_scores.get(strategy, (0.0, 0.0, 0.0, 0.0))
                oos_score = oos_scores.get(strategy, (0.0, 0.0, 0.0, 0.0))
                oos_report: dict[str, Any] = {
                    "strategy": strategy,
                    "selection_score": _score_payload(selected_score),
                    "selection_basis": ranking_basis.get(strategy, "oos_stress1"),
                    "oos_score": _score_payload(oos_score),
                    "score_mode": ranking_mode,
                    "config": best_summary.get("best_config") or best_summary.get("best_cfg") or {},
                    "performance": best_summary.get("test_window_stress_1")
                    or _score_payload(oos_score),
                    "output_dir": best_entry["output_dir"],
                }

                if strategy in multifold_details:
                    oos_report["multifold"] = {
                        k: v
                        for k, v in multifold_details[strategy].items()
                        if k != "score"
                    }

                if ranking_mode == "vs_benchmark" and benchmark_score is not None:
                    oos_report["delta_vs_benchmark"] = {
                        "cagr": selected_score[0] - benchmark_score[0],
                        "sharpe": selected_score[1] - benchmark_score[1],
                        "max_drawdown": abs(benchmark_score[2]) - abs(selected_score[2]),
                        "turnover": benchmark_score[3] - selected_score[3],
                    }

                print("\n=== OOS winner ===")
                print(json.dumps(oos_report, indent=2))
                print("Output:", best_entry["output_dir"])

                report["oos_winner"] = oos_report
                report["best"] = oos_report

    summary_path = output_root / "all_strategies_summary.json"
    summary_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    final_report_path = run_reports_dir / "final_summary.json"
    final_report_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    # Latest successful pointer for operator visibility.
    if any(int(r.get("return_code", 1)) == 0 for r in results):
        _write_latest_successful_pointer(
            run_reports_dir=run_reports_dir,
            summary_path=summary_path,
            final_report_path=final_report_path,
            report=report,
        )

    _write_progress(
        progress_file,
        {
            "event": "summary_written",
            "run_id": run_reports_dir.name,
            "summary_path": str(summary_path),
            "report_path": str(final_report_path),
            "has_best": report["best"] is not None,
            "ranking_mode": ranking_mode,
        },
    )

    return report


def main() -> int:
    args = parse_args()
    try:
        log_path = setup_system_logger(level=logging.INFO)
    except TypeError:
        log_path = setup_system_logger()
    logger.info("frontier_all_start log_path=%s args=%s", log_path, vars(args))

    if not ACTIVE_STRATEGIES:
        print("No active strategies discovered in BacktestConfig.VALID_STRATEGIES")
        return 1

    if args.max_error_rate < 0.0 or args.max_error_rate > 1.0:
        print("ERROR: --max-error-rate must be between 0.0 and 1.0", file=sys.stderr)
        return 2

    if args.resume and not args.run_id:
        print("ERROR: --resume requires --run-id so the existing run can be located", file=sys.stderr)
        return 2

    if not _validate_acceleration_backend(args.acceleration_backend):
        return 2

    results, run_reports_dir, progress_path = _run_all_strategies(args)
    summary_report = _summarize(
        results,
        output_root=Path(args.output_dir),
        run_reports_dir=run_reports_dir,
        progress_file=progress_path,
        ranking_mode=args.ranking_mode,
    )

    logger.info("frontier_all_complete summary=%s", summary_report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
