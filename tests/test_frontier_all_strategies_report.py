from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path

MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "frontier_sweep_all_strategies.py"
spec = importlib.util.spec_from_file_location("frontier_sweep_all_strategies", MODULE_PATH)
assert spec and spec.loader
fas = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fas)  # type: ignore[assignment]


def _write_summary_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "param_id",
                "window",
                "scenario",
                "cagr",
                "sharpe",
                "max_drawdown",
                "turnover",
                "trade_count",
                "net_pnl",
                "error",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _strategy_result(
    strategy_dir: Path,
    *,
    strategy: str,
    param_id: str,
    cagr_base: float,
    sharpe_base: float,
    dd: float,
) -> dict[str, object]:
    rows = []
    for window, base_cagr in (("train", cagr_base + 0.02), ("val", cagr_base + 0.04), ("test", cagr_base + 0.01)):
        for scenario in ("baseline", "stress_1", "stress_2"):
            penalty = {
                "baseline": 0.00,
                "stress_1": -0.01,
                "stress_2": -0.02,
            }[scenario]
            rows.append(
                {
                    "param_id": param_id,
                    "window": window,
                    "scenario": scenario,
                    "cagr": base_cagr + penalty,
                    "sharpe": (base_cagr + penalty) * 8,
                    "max_drawdown": dd,
                    "turnover": 150.0,
                    "trade_count": 20,
                    "net_pnl": 120.0,
                    "error": "",
                }
            )
    csv_path = strategy_dir / "summary.csv"
    _write_summary_csv(csv_path, rows)

    best_summary = {
        "best": {
            "param_id": param_id,
            "val_stress1_cagr": cagr_base + 0.04 - 0.01,
            "val_cagr_stress_1": cagr_base + 0.04 - 0.01,
            "val_sharpe_stress_1": (cagr_base + 0.04 - 0.01) * 8,
            "val_stress1_sharpe": (cagr_base + 0.04 - 0.01) * 8,
            "val_max_drawdown_worst": dd,
            "val_stress1_max_drawdown": dd,
            "val_turnover": 150.0,
            "val_stress1_turnover": 150.0,
            "test_cagr_stress_1": cagr_base + 0.01 - 0.01,
            "test_sharpe_stress_1": (cagr_base + 0.01 - 0.01) * 8,
            "test_max_drawdown_stress_1": dd,
            "test_turnover_stress_1": 150.0,
        },
        "test_window_stress_1": {
            "cagr": cagr_base,
            "sharpe": sharpe_base,
            "max_drawdown": dd,
            "turnover": 150.0,
            "trade_count": 20,
        },
        "best_config": {"param_id": param_id},
    }
    best_path = strategy_dir / "best_summary.json"
    best_path.write_text(json.dumps(best_summary), encoding="utf-8")

    return {
        "strategy": strategy,
        "output_dir": str(strategy_dir),
        "summary": best_summary,
        "return_code": 0,
        "error_rows": 0,
        "total_rows": 1,
        "error_rate": 0.0,
    }


def test_all_strategies_report_labels_and_latest_pointer(tmp_path: Path) -> None:
    run_reports_dir = fas._run_reports_dir("test_report_labels")
    run_reports_dir.mkdir(parents=True, exist_ok=True)
    progress_file = run_reports_dir / "progress.jsonl"
    output_root = Path(tmp_path / "artifacts")

    bench_dir = output_root / "macro_gate_benchmark" / "run_test_report_labels"
    macro_only_dir = output_root / "macro_only_v2" / "run_test_report_labels"

    results = [
        _strategy_result(bench_dir, strategy="macro_gate_benchmark", param_id="p0", cagr_base=0.022, sharpe_base=0.12, dd=-0.08),
        _strategy_result(macro_only_dir, strategy="macro_only_v2", param_id="p0", cagr_base=0.030, sharpe_base=0.24, dd=-0.06),
    ]

    report = fas._summarize(
        results=results,
        output_root=output_root,
        run_reports_dir=run_reports_dir,
        progress_file=progress_file,
        ranking_mode="vs_benchmark",
    )

    assert report is not None
    assert report["validation_winner"]["strategy"] == "macro_only_v2"
    assert report["oos_winner"]["strategy"] == "macro_only_v2"

    latest_json = fas.REPORTS_BASE_DIR / "latest_successful_run.json"
    latest_txt = fas.REPORTS_BASE_DIR / "latest_successful_run.txt"
    assert latest_json.exists()
    assert latest_txt.exists()
    payload = json.loads(latest_json.read_text(encoding="utf-8"))
    assert payload["validation_winner"] == "macro_only_v2"
    assert payload["oos_winner"] == "macro_only_v2"

    final_path = run_reports_dir / "final_summary.json"
    final_payload = json.loads(final_path.read_text(encoding="utf-8"))
    assert final_payload["validation_winner"]["strategy"] == "macro_only_v2"
    assert final_payload["oos_winner"]["strategy"] == "macro_only_v2"


def test_multifold_robust_scoring_prefers_stable_windows(tmp_path: Path) -> None:
    report_dir = Path(tmp_path / "stable_report")
    rows_unstable = [
        {"param_id": "p0", "window": "train", "scenario": "baseline", "cagr": 0.05, "sharpe": 1.0, "max_drawdown": -0.08, "turnover": 120.0, "trade_count": 10, "net_pnl": 10.0, "error": ""},
        {"param_id": "p0", "window": "train", "scenario": "stress_1", "cagr": -0.20, "sharpe": -1.0, "max_drawdown": -0.09, "turnover": 180.0, "trade_count": 10, "net_pnl": -4.0, "error": ""},
        {"param_id": "p0", "window": "train", "scenario": "stress_2", "cagr": 0.06, "sharpe": 1.1, "max_drawdown": -0.07, "turnover": 150.0, "trade_count": 12, "net_pnl": 11.0, "error": ""},
        {"param_id": "p0", "window": "val", "scenario": "baseline", "cagr": 0.04, "sharpe": 0.8, "max_drawdown": -0.10, "turnover": 130.0, "trade_count": 8, "net_pnl": 8.0, "error": ""},
        {"param_id": "p0", "window": "val", "scenario": "stress_1", "cagr": 0.03, "sharpe": 0.7, "max_drawdown": -0.11, "turnover": 150.0, "trade_count": 9, "net_pnl": 9.0, "error": ""},
        {"param_id": "p0", "window": "val", "scenario": "stress_2", "cagr": 0.02, "sharpe": 0.5, "max_drawdown": -0.12, "turnover": 140.0, "trade_count": 7, "net_pnl": 7.0, "error": ""},
        {"param_id": "p0", "window": "test", "scenario": "baseline", "cagr": 0.01, "sharpe": 0.6, "max_drawdown": -0.14, "turnover": 160.0, "trade_count": 10, "net_pnl": 6.0, "error": ""},
        {"param_id": "p0", "window": "test", "scenario": "stress_1", "cagr": -0.02, "sharpe": -0.6, "max_drawdown": -0.15, "turnover": 220.0, "trade_count": 11, "net_pnl": -2.0, "error": ""},
        {"param_id": "p0", "window": "test", "scenario": "stress_2", "cagr": 0.00, "sharpe": 0.4, "max_drawdown": -0.16, "turnover": 170.0, "trade_count": 10, "net_pnl": 2.0, "error": ""},
    ]
    _write_summary_csv(report_dir / "summary.csv", rows_unstable)

    best_summary = {
        "best": {
            "param_id": "p0",
            "val_cagr_stress_1": 0.03,
            "val_sharpe_stress_1": 0.7,
            "val_max_drawdown_worst": -0.11,
            "val_turnover": 130.0,
            "test_cagr_stress_1": -0.02,
            "test_sharpe_stress_1": -0.6,
            "test_max_drawdown_stress_1": -0.15,
            "test_turnover_stress_1": 220.0,
        },
        "best_config": {"param_id": "p0"},
    }

    score = fas._multifold_robust_score(report_dir, best_summary)
    assert score is not None
    assert score["fold_cagr_std"] > 0.0
    assert score["scenario_cagr_penalty"] > 0.0
    assert score["score"][0] < 0.03
