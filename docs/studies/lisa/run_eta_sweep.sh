#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PY="${ROOT}/.venv/bin/python"
SCRIPT="${ROOT}/docs/studies/lisa/eta_sweep.py"

SEEDS="${SEEDS:-0}"
ETAS="${ETAS:-1.0 0.25 0.125 0.08 0.06 0.04}"
LABELS="${LABELS:-AQ AR AS AT AU AV}"
K="${K:-100}"
K_DELTA="${K_DELTA:-}"
K_THETA_RE="${K_THETA_RE:-}"
K_THETA_IM="${K_THETA_IM:-4}"

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
)

if [[ -n "${K_DELTA}" ]]; then
  CMD+=( --K-delta "${K_DELTA}" )
fi
if [[ -n "${K_THETA_RE}" ]]; then
  CMD+=( --K-theta-re "${K_THETA_RE}" )
fi

"${CMD[@]}" "$@"
