#!/usr/bin/env bash
# Duration × eta study: sweep eta across multiple LISA observation durations.
#
# For each duration the eta grid is derived from η = min(1, c/Nb) for a set of
# c values, but the script only exposes durations and seeds — c is an internal
# implementation detail, not a user-facing parameter.
#
# Usage:
#   ./run_duration_eta_study.sh
#   DURATIONS="7 30 91 182 365" SEEDS="0" ./run_duration_eta_study.sh

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PY="${ROOT}/.venv/bin/python"
SCRIPT="${ROOT}/docs/studies/lisa/eta_sweep.py"

DURATIONS="${DURATIONS:-7 30 91 182 365}"
SEEDS="${SEEDS:-0}"
BLOCK_DAYS="${BLOCK_DAYS:-7}"
K="${K:-100}"
K_THETA_IM="${K_THETA_IM:-4}"

if [[ ! -x "${PY}" ]]; then
  echo "ERROR: venv python not found at ${PY}" >&2
  exit 1
fi

# Delegate all per-duration eta/label computation to Python (bash 3 compatible).
PLAN=$("${PY}" - <<'PYEOF'
import sys, os
sys.path.insert(0, os.environ.get("ROOT", ".") + "/src")

durations = [int(d) for d in os.environ.get("DURATIONS", "7 30 91 182 365").split()]
block_days = int(os.environ.get("BLOCK_DAYS", "7"))
# Internal c-space grid; not exposed as a user parameter.
c_values = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0]

def idx_to_label(idx):
    chars = []
    while idx > 0:
        idx -= 1
        chars.append(chr(ord("A") + idx % 26))
        idx //= 26
    return "".join(reversed(chars))

label_idx = 53  # BA — after AW-AZ used by eta_validation runs

lines = []
for d in durations:
    nb = max(1, d // block_days)
    seen = {}
    etas, labels = [], []
    for c in c_values:
        eta = round(min(1.0, c / nb), 6)
        key = f"{eta:.5g}"
        if key not in seen:
            seen[key] = True
            etas.append(eta)
            labels.append(idx_to_label(label_idx))
            label_idx += 1
    lines.append(f"{d}|{nb}|{' '.join(str(e) for e in etas)}|{' '.join(labels)}")

print("\n".join(lines))
PYEOF
)

# Parse the plan and run each duration.
while IFS='|' read -r DURATION NB ETAS LABELS; do
  echo "Duration=${DURATION}d  Nb=${NB}"
  echo "  etas:   ${ETAS}"
  echo "  labels: ${LABELS}"

  # shellcheck disable=SC2206
  SEED_ARGS=( ${SEEDS} )
  # shellcheck disable=SC2206
  ETA_ARGS=( ${ETAS} )
  # shellcheck disable=SC2206
  LABEL_ARGS=( ${LABELS} )

  "${PY}" "${SCRIPT}" \
    --seeds "${SEED_ARGS[@]}" \
    --etas "${ETA_ARGS[@]}" \
    --labels "${LABEL_ARGS[@]}" \
    --duration-days "${DURATION}" \
    --block-days "${BLOCK_DAYS}" \
    --K "${K}" \
    --K-theta-im "${K_THETA_IM}" \
    --keep-nc \
    "$@"

done <<< "${PLAN}"

echo "Duration-eta study complete."
