#!/usr/bin/env python3
"""Compute adaptive_trend_6h_v1 frontier-sweep progress for the active run."""

from __future__ import annotations

import csv
import json
import re
import subprocess
from datetime import datetime, timezone
from statistics import median
from pathlib import Path

RUN_ID = "run_20260301T211500Z_auto10w_bench_adapt6h"
ROOT = Path('/mnt/c/Users/josep/trading_bot_btc')
ART = ROOT / 'artifacts/frontier_all_strategies'
SYSTEM_LOG = ROOT / 'system_log.log'
PROGRESS = ROOT / 'reports/frontier_sweep_all_strategies' / RUN_ID / 'progress.jsonl'
STRATEGY = 'adaptive_trend_6h_v1'


def dtparse(ts: str) -> datetime:
    return datetime.fromisoformat(ts.replace('Z', '+00:00'))


def find_latest_progress(event: str):
    if not PROGRESS.exists():
        return None
    latest = None
    for line in PROGRESS.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get('event') == event and row.get('run_id') == RUN_ID and row.get('strategy') == STRATEGY:
            t = dtparse(row['timestamp'])
            if latest is None or t > latest[0]:
                latest = (t, row)
    return latest[1] if latest else None


def is_running() -> bool:
    try:
        # shell pattern to include strategy and strategy-specific run-id
        p = subprocess.run(
            ["bash", "-lc", f"pgrep -af 'frontier_sweep.py --strategy {STRATEGY}.*run-id {RUN_ID}'"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if p.returncode != 0:
            return False
        lines = [ln for ln in p.stdout.splitlines() if ln.strip()]
        return bool(lines)
    except Exception:
        return False


def current_phase() -> tuple[str, str]:
    """Return (phase_window, scenario_phase) based on latest engine run_start for this strategy."""
    if not SYSTEM_LOG.exists():
        return ('unknown', 'unknown')

    p = None
    pid = None
    # locate current pid from any matching process line in checkpoint (if running)
    try:
        pproc = subprocess.run(
            ["bash", "-lc", f"pgrep -f 'frontier_sweep.py --strategy {STRATEGY} --product BTC-USD.*{RUN_ID}' | head -n 1"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        out = pproc.stdout.strip().splitlines()
        if out:
            pid = out[0].split(':')[0].strip()
    except Exception:
        pid = None

    lines = SYSTEM_LOG.read_text().splitlines()
    candidates = []
    for ln in lines:
        if 'engine_run_start' not in ln and 'engine_run_complete' not in ln:
            continue
        if 'adaptive_trend_6h_v1' not in ln:
            continue
        if pid and pid not in ln and ('engine_run_complete' in ln or 'engine_run_start' in ln):
            # when pid known, restrict to avoid other strategy logs from different workers
            continue
        # parse timestamp and window
        m = re.match(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),", ln)
        m2 = re.search(r"start=(\d{4}-\d{2}-\d{2})", ln)
        if not m or not m2:
            continue
        ts = datetime.strptime(m.group(1), '%Y-%m-%d %H:%M:%S')
        candidates.append((ts, ln, m2.group(1)))

    if not candidates:
        return ('unknown', 'unknown')

    candidates.sort(key=lambda x: x[0])
    last_start_ts, last_line, start_date = candidates[-1]
    phase = 'test' if start_date >= '2026-01-01' else 'val'

    # detect whether last start had a completion already
    param_phase = 'baseline'
    # if an engine_run_complete for same strategy occurs after latest start, we're in second scenario or next window
    after_start = False
    for ln in lines:
        if ln == last_line:
            after_start = True
            continue
        if not after_start:
            continue
        if 'engine_run_complete strategy=' + STRATEGY in ln and ('adaptive_trend_6h_v1' in ln):
            if pid is None or pid in ln:
                param_phase = 'stress_1'
    return (phase, param_phase)


def checkpoint_info():
    cp = ART / STRATEGY / RUN_ID / 'checkpoint.json'
    if not cp.exists():
        return None
    return json.loads(cp.read_text())


def best_candidate_from_summary() -> dict | None:
    best_file = ART / STRATEGY / RUN_ID / 'best_summary.json'
    if best_file.exists():
        b = json.loads(best_file.read_text())
        if b.get('best'):
            return {
                'source': 'best_summary',
                'param_id': b['best'].get('param_id'),
                'cagr': b.get('test_window_stress_1', {}).get('cagr', 0.0),
                'sharpe': b.get('test_window_stress_1', {}).get('sharpe', 0.0),
                'max_drawdown': b.get('test_window_stress_1', {}).get('max_drawdown', 0.0),
            }

    # derive rough best from summary rows
    summary_path = ART / STRATEGY / RUN_ID / 'summary.csv'
    if not summary_path.exists():
        return None

    rows = list(csv.DictReader(summary_path.open()))
    rows = [r for r in rows if not str(r.get('error', '')).strip()]
    if not rows:
        return None

    from collections import defaultdict

    grouped = defaultdict(lambda: defaultdict(dict))
    for r in rows:
        grouped[r['param_id']][r['window']][r['scenario']] = r

    candidates = []
    for pid, byw in grouped.items():
        val = byw.get('val', {})
        test = byw.get('test', {})
        base = val.get('baseline') or val.get('stress_1')
        stress1 = val.get('stress_1') or val.get('baseline')
        if not base or not stress1:
            continue
        if float(stress1.get('net_pnl', 0.0) or 0.0) <= 0.0:
            continue
        stress2 = val.get('stress_2') or stress1
        if abs(float(base.get('max_drawdown', 0.0) or 0.0)) > 0.3 or abs(float(stress1.get('max_drawdown', 0.0) or 0.0)) > 0.3:
            continue
        test_stress = test.get('stress_1') or test.get('baseline')
        if not test_stress:
            continue
        turnover = max(float(base.get('turnover', 0.0) or 0.0), float(stress1.get('turnover', 0.0) or 0.0), float((stress2).get('turnover', 0.0) or 0.0))
        if turnover > 700.0:
            continue
        cand = {
            'param_id': pid,
            'val_score': median([
                float(base.get('cagr', 0.0) or 0.0),
                float(stress1.get('cagr', 0.0) or 0.0),
                float(stress2.get('cagr', 0.0) or 0.0),
            ]),
            'cagr': float(test_stress.get('cagr', 0.0) or 0.0),
            'sharpe': float(test_stress.get('sharpe', 0.0) or 0.0),
            'max_drawdown': float(test_stress.get('max_drawdown', 0.0) or 0.0),
        }
        candidates.append(cand)

    if not candidates:
        return None

    top = max(candidates, key=lambda x: (x['val_score'], x['cagr'], x['sharpe'], -abs(x['max_drawdown'])))
    top['source'] = 'derived'
    return top


def fmt_eta(seconds: float | None) -> str:
    if not seconds or seconds <= 0 or seconds != seconds:
        return 'n/a'
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h}h {m}m {s}s"


cp = checkpoint_info()
if not cp:
    print(json.dumps({'error': 'checkpoint missing'}))
    raise SystemExit(1)

processed = cp.get('processed_count', 0)
total = cp.get('total_param_sets', 0)
remaining = max(0, total - processed)
run_started = find_latest_progress('strategy_started')
start_ts = dtparse(run_started['timestamp']) if run_started else None
updated_at = cp.get('updated_at')
completed = bool(cp.get('completed'))
phase, scenario = current_phase()

eta_phase = None
if processed and start_ts:
    elapsed = (datetime.now(timezone.utc) - start_ts).total_seconds()
    if elapsed > 0:
        eta_phase = (total - processed) * (elapsed / processed)
eta_full = eta_phase
confidence = (processed / total * 100.0) if total else 0.0

best = best_candidate_from_summary()

payload = {
    'run_id': RUN_ID,
    'run_running': is_running(),
    'strategy': STRATEGY,
    'processed': processed,
    'total': total,
    'remaining': remaining,
    'updated_at': updated_at,
    'completed': completed,
    'current_phase': phase,
    'current_param_phase': scenario,
    'eta_current_phase': fmt_eta(eta_phase),
    'eta_full_run': fmt_eta(eta_full),
    'confidence_percent': round(confidence, 2),
    'best': best,
}
print(json.dumps(payload, indent=2))
