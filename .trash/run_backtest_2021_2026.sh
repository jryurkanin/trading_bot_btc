#!/usr/bin/env bash
set -euo pipefail

# Run a wider-range backtest from 2021-01-01 through 2026-02-15.
# Usage:
#   ./scripts/run_backtest_2021_2026.sh [strategy]
# Example:
#   ./scripts/run_backtest_2021_2026.sh macro_only_v2

STRATEGY="${1:-macro_only_v2}"
START_TS="2021-01-01T00:00:00Z"
END_TS="2026-02-15T00:00:00Z"
OUT_DIR="reports/backtest_2021_01_01_2026_02_15_${STRATEGY}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${REPO_ROOT}"

if [[ -f "../.venv/bin/activate" ]]; then
  # Workspace-level venv (current setup)
  # shellcheck disable=SC1091
  source "../.venv/bin/activate"
elif [[ -f ".venv/bin/activate" ]]; then
  # Repo-local venv fallback
  # shellcheck disable=SC1091
  source ".venv/bin/activate"
fi

python3 scripts/backtest.py \
  --product BTC-USD \
  --strategy "${STRATEGY}" \
  --fill-model bid_ask \
  --start "${START_TS}" \
  --end "${END_TS}" \
  --acceleration-backend auto \
  --output "${OUT_DIR}"

echo "Backtest finished: ${OUT_DIR}/report.json"
