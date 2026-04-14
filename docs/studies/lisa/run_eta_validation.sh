#!/usr/bin/env bash
# Run a focused set of eta values for LISA 365-day analysis with full metrics.
#
# Unlike run_eta_sweep.sh this script:
#   - Keeps inference_data.nc (--keep-nc) so quantile-based metrics are available
#   - Uses --force to re-run even if summaries exist
#   - Targets only the candidate eta range [0.04, 0.125] plus eta=1 as baseline
#
# Usage:
#   ./run_eta_validation.sh               # default: seeds 0, etas 1.0 0.125 0.08 0.04
#   SEEDS="0 1 2" ./run_eta_validation.sh
#   ETAS="0.08 0.125" ./run_eta_validation.sh

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PY="${ROOT}/.venv/bin/python"
SCRIPT="${ROOT}/docs/studies/lisa/eta_sweep.py"

SEEDS="${SEEDS:-0}"
ETAS="${ETAS:-1.0 0.125 0.08 0.04}"
LABELS="${LABELS:-AW AX AY AZ}"
K="${K:-100}"
K_DELTA="${K_DELTA:-}"
K_THETA_RE="${K_THETA_RE:-}"
K_THETA_IM="${K_THETA_IM:-4}"
DURATION="${DURATION:-365}"

if [[ ! -x "${PY}" ]]; then
  echo "ERROR: venv python not found at ${PY}" >&2
  exit 1
fi

# shellcheck disable=SC2206
SEED_ARGS=( ${SEEDS} )
# shellcheck disable=SC2206
ETA_ARGS=( ${ETAS} )
# shellcheck disable=SC2206
LABEL_ARGS=( ${LABELS} )

CMD=(
  "${PY}" "${SCRIPT}"
  --seeds "${SEED_ARGS[@]}"
  --etas "${ETA_ARGS[@]}"
  --labels "${LABEL_ARGS[@]}"
  --K "${K}"
  --K-theta-im "${K_THETA_IM}"
  --duration-days "${DURATION}"
  --keep-nc
  --force
)

if [[ -n "${K_DELTA}" ]]; then
  CMD+=( --K-delta "${K_DELTA}" )
fi
if [[ -n "${K_THETA_RE}" ]]; then
  CMD+=( --K-theta-re "${K_THETA_RE}" )
fi

echo "Running: ${CMD[*]} $*"
"${CMD[@]}" "$@"
