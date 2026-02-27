from __future__ import annotations

import csv
import hashlib
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


def _json_ready(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    return value


def stable_hash(payload: Any) -> str:
    blob = json.dumps(_json_ready(payload), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _normalize_windows(windows: list[Any]) -> list[dict[str, str]]:
    normalized: list[dict[str, str]] = []
    for item in windows:
        if isinstance(item, dict):
            name = str(item.get("name", ""))
            start_raw = item.get("start")
            end_raw = item.get("end")
        else:
            name = str(getattr(item, "name", ""))
            start_raw = getattr(item, "start", "")
            end_raw = getattr(item, "end", "")

        if isinstance(start_raw, datetime):
            start = start_raw.isoformat()
        else:
            start = str(start_raw)

        if isinstance(end_raw, datetime):
            end = end_raw.isoformat()
        else:
            end = str(end_raw)

        normalized.append({"name": name, "start": start, "end": end})
    return normalized


def build_checkpoint_fingerprint(strategy: str, param_sets: list[dict[str, Any]], windows: list[Any]) -> dict[str, str]:
    return {
        "strategy": str(strategy),
        "grid_hash": stable_hash(param_sets),
        "window_hash": stable_hash(_normalize_windows(windows)),
    }


def checkpoint_fingerprint_mismatches(
    checkpoint_payload: dict[str, Any],
    expected_fingerprint: dict[str, str],
) -> dict[str, dict[str, Any]]:
    mismatches: dict[str, dict[str, Any]] = {}
    for key in ("strategy", "grid_hash", "window_hash"):
        expected = expected_fingerprint.get(key)
        observed = checkpoint_payload.get(key)
        if observed is None:
            mismatches[key] = {
                "reason": "missing",
                "expected": expected,
                "observed": None,
            }
            continue
        if str(observed) != str(expected):
            mismatches[key] = {
                "reason": "mismatch",
                "expected": expected,
                "observed": observed,
            }
    return mismatches


def build_filter_rejections_payload(
    *,
    run_id: str,
    strategy: str,
    total_param_sets: int,
    rejection_counts: dict[str, int],
) -> dict[str, Any]:
    accepted = int(rejection_counts.get("accepted", 0) or 0)
    rejections: dict[str, int] = {}
    for key, value in sorted(rejection_counts.items()):
        if key == "accepted":
            continue
        rejections[str(key)] = int(value or 0)

    return {
        "run_id": run_id,
        "strategy": strategy,
        "total_param_sets": int(total_param_sets),
        "accepted_param_sets": accepted,
        "rejections": rejections,
    }
