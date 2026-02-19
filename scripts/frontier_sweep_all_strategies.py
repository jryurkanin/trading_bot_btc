#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

# Make local src discoverable when running directly from the repository root
sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

from bot.config import BacktestConfig

ROOT_DIR = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = ROOT_DIR / "scripts"

StrategyRunner = tuple[str, str]

ACTIVE_STRATEGIES: list[str] = sorted(BacktestConfig.VALID_STRATEGIES)

RUNNERS: dict[str, StrategyRunner] = {
    "macro_gate_benchmark": ("frontier_sweep.py", "macro_gate_benchmark"),
    "macro_only_v2": ("frontier_sweep_macro_only.py", "macro_only_v2"),
}


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
    p.add_argument("--small", action="store_true", help="Use reduced macro-only sweep grid for quick checks")
    p.add_argument("--workers", type=int, default=20, help="Workers for macro-only frontier sweep")
    p.add_argument("--top-n", type=int, default=5)
    p.add_argument("--turnover-max", type=float, default=700.0)
    p.add_argument("--max-drawdown-max", type=float, default=0.30)
    p.add_argument("--strategies", default=",".join(ACTIVE_STRATEGIES), help="Comma-separated strategies to run")
    p.add_argument("--maker-bps", type=float, default=10.0)
    p.add_argument("--taker-bps", type=float, default=25.0)
    p.add_argument("--output-dir", default="artifacts/frontier_all_strategies", help="Output root directory")
    p.add_argument(
        "--timeout-seconds",
        type=int,
        default=21600,
        help="Timeout for each strategy run in seconds (default: 6 hours)",
    )
    return p.parse_args()


def _run_command(cmd: list[str], timeout_seconds: int) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(
            cmd,
            cwd=str(ROOT_DIR),
            text=True,
            check=False,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        return subprocess.CompletedProcess(
            args=cmd,
            returncode=124,
            stdout=(exc.stdout or "") if exc.stdout else "",
            stderr=(exc.stderr or "") + f"\nCommand timed out after {timeout_seconds}s\n",
        )


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
    if args.include_fred_grid:
        base_args.append("--include-fred-grid")
    return base_args


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

    cagr = float(top.get("test_cagr_stress_1", top.get("val_stress1_cagr", 0.0))) or 0.0
    sharpe = float(top.get("test_sharpe_stress_1", top.get("val_sharpe_stress_1", top.get("sharpe", 0.0)))) or 0.0
    max_drawdown = float(top.get("test_max_drawdown_stress_1", top.get("val_max_drawdown_worst", top.get("max_drawdown", 0.0)))) or 0.0

    # Normalize to best_summary-like structure for downstream comparisons.
    return {
        "strategy": strategy,
        "best": top,
        "test_window_stress_1": {
            "cagr": cagr,
            "sharpe": sharpe,
            "max_drawdown": max_drawdown,
            "trade_count": top.get("trade_count") or top.get("val_trade_count", 0),
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
        best.get("test_cagr_stress_1", best.get("val_stress1_cagr", 0.0)) or 0.0
    )
    sharpe = float(best.get("val_sharpe_stress_1", 0.0) or best.get("sharpe", 0.0) or 0.0)
    drawdown = float(best.get("val_max_drawdown_worst", 0.0) or 0.0)
    turnover = float(best.get("val_turnover", 0.0) or best.get("turnover", 0.0) or 0.0)
    return (cagr, sharpe, drawdown, turnover)


def _run_all_strategies(args: argparse.Namespace) -> list[dict[str, Any]]:
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    selected = [s.strip() for s in args.strategies.split(",") if s.strip()]
    unknown = [s for s in selected if s not in RUNNERS]
    if unknown:
        raise ValueError(f"Unknown strategy requested: {', '.join(unknown)}. Active options: {', '.join(ACTIVE_STRATEGIES)}")

    baseline_args = _build_base_args(args)
    results: list[dict[str, Any]] = []

    for strategy in selected:
        script_name, default_tag = RUNNERS[strategy]
        strategy_dir = output_root / default_tag
        strategy_dir.mkdir(parents=True, exist_ok=True)

        args_for_strategy = baseline_args.copy()
        args_for_strategy.extend(["--output-dir", str(strategy_dir)])

        cmd = [sys.executable, str(SCRIPTS_DIR / script_name)]

        if strategy == "macro_gate_benchmark":
            cmd.extend(["--strategy", strategy])
            cmd.extend(["--end", args.end])
            cmd.extend(["--test-end", args.test_end])
            cmd.extend(args_for_strategy)
        elif strategy == "macro_only_v2":
            cmd.extend(["--test-end", args.test_end])
            cmd.extend(["--workers", str(args.workers)])
            if args.small:
                cmd.append("--small")
            cmd.extend(args_for_strategy)
        else:
            # Fallback if future strategy runners are added.
            cmd.extend(args_for_strategy)

        print(f"\n=== Running frontier sweep: {strategy} ===")
        print("Command:", " ".join(cmd))
        completed = _run_command(cmd, timeout_seconds=args.timeout_seconds)

        summary = None
        if completed.returncode == 0:
            summary = _load_best_summary(strategy_dir / "best_summary.json", strategy_dir=strategy_dir, strategy=strategy)
            if summary is None:
                # If a future workflow doesn't emit best_summary, try to use frontier.csv as fallback.
                print(f"Warning: no best_summary.json found for {strategy}. Parsing frontier.csv fallback")
        else:
            print(f"Strategy {strategy} failed with exit code {completed.returncode}")

        result: dict[str, Any] = {
            "strategy": strategy,
            "script": script_name,
            "return_code": completed.returncode,
            "output_dir": str(strategy_dir),
            "summary": summary,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        }
        results.append(result)

    return results


def _summarize(results: list[dict[str, Any]], output_root: Path) -> dict[str, Any] | None:
    print("\n=== Frontier sweep summary (all active strategies) ===")

    best_entry: dict[str, Any] | None = None
    best_score: tuple[float, float, float, float] = (-1.0, -1.0, 0.0, 0.0)

    for result in results:
        strategy = result["strategy"]
        summary = result["summary"]
        rc = result["return_code"]
        print(f"\n- {strategy}: exit={rc}, output={result['output_dir']}")

        if rc != 0:
            print("  Status: failed")
            continue
        if not summary:
            print("  Status: completed (no best summary detected)")
            continue

        score = _extract_score(summary)
        print(f"  Status: success")
        print(f"  Best cagr: {score[0]:.6f}, sharpe: {score[1]:.6f}, drawdown: {score[2]:.6f}, turnover: {score[3]:.6f}")
        if score > best_score:
            best_entry = result
            best_score = score

    if best_entry is None:
        print("\nNo successful strategy run produced a best_summary result.")
        report: dict[str, Any] = {
            "results": results,
            "best": None,
            "output_root": str(output_root),
        }
        (output_root / "all_strategies_summary.json").write_text(
            json.dumps(report, indent=2),
            encoding="utf-8",
        )
        return report

    best_summary = best_entry["summary"]
    if not isinstance(best_summary, dict):
        print("\nCould not parse best strategy details for final report.")
        report = {
            "results": results,
            "best": None,
            "output_root": str(output_root),
        }
        (output_root / "all_strategies_summary.json").write_text(
            json.dumps(report, indent=2),
            encoding="utf-8",
        )
        return report

    best_report = {
        "strategy": best_entry["strategy"],
        "score": {
            "cagr": best_score[0],
            "sharpe": best_score[1],
            "max_drawdown": best_score[2],
            "turnover": best_score[3],
        },
        "config": best_summary.get("best_config") or best_summary.get("best_cfg") or {},
        "performance": best_summary.get("test_window_stress_1")
        or {
            "cagr": best_score[0],
            "sharpe": best_score[1],
            "max_drawdown": best_score[2],
            "turnover": best_score[3],
        },
        "output_dir": best_entry["output_dir"],
    }

    print("\n=== Best strategy ===")
    print(json.dumps(best_report, indent=2))
    print("Output:", best_entry["output_dir"])

    summary_report = {
        "output_root": str(output_root),
        "results": results,
        "best": best_report,
    }
    (output_root / "all_strategies_summary.json").write_text(
        json.dumps(summary_report, indent=2),
        encoding="utf-8",
    )
    return summary_report


def main() -> int:
    args = parse_args()
    if not ACTIVE_STRATEGIES:
        print("No active strategies discovered in BacktestConfig.VALID_STRATEGIES")
        return 1

    results = _run_all_strategies(args)
    _summarize(results, output_root=Path(args.output_dir))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
