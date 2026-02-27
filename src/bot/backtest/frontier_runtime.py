from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def resolve_run_dir(output_dir: str | Path, run_id: str | None, resume: bool) -> tuple[Path, str]:
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    resolved_run_id = (run_id or "").strip()
    if resume and not resolved_run_id:
        latest_path = output_root / "latest_run.txt"
        if latest_path.exists():
            try:
                resolved_run_id = latest_path.read_text(encoding="utf-8").strip()
            except Exception:
                resolved_run_id = ""

    if not resolved_run_id:
        resolved_run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")

    run_token = resolved_run_id if resolved_run_id.startswith("run_") else f"run_{resolved_run_id}"
    run_dir = output_root / run_token
    run_dir.mkdir(parents=True, exist_ok=True)

    try:
        (output_root / "latest_run.txt").write_text(run_token + "\n", encoding="utf-8")
    except Exception:
        pass

    return run_dir, run_token


def load_summary_rows(summary_path: Path) -> list[dict[str, Any]]:
    if not summary_path.exists():
        return []
    try:
        with summary_path.open("r", encoding="utf-8") as f:
            return list(csv.DictReader(f))
    except Exception:
        return []


def write_summary_rows(summary_path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        if summary_path.exists():
            try:
                summary_path.unlink()
            except Exception:
                pass
        return

    cols = sorted({k for row in rows for k in row.keys()})
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def load_checkpoint(checkpoint_path: Path) -> dict[str, Any]:
    if not checkpoint_path.exists():
        return {}
    try:
        return json.loads(checkpoint_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_checkpoint(checkpoint_path: Path, payload: dict[str, Any]) -> None:
    checkpoint_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def derive_processed_param_ids(summary_rows: list[dict[str, Any]], expected_rows_per_param: int) -> set[str]:
    expected = max(1, int(expected_rows_per_param))
    counts: dict[str, int] = {}
    for row in summary_rows:
        pid_raw = row.get("param_id")
        if pid_raw is None:
            continue
        pid = str(pid_raw).strip()
        if not pid or pid == "baseline":
            continue
        counts[pid] = counts.get(pid, 0) + 1
    return {pid for pid, count in counts.items() if count >= expected}
