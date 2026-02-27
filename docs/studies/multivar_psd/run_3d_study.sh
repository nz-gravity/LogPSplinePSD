#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${HERE}/../../.." && pwd)"
PY="${ROOT}/.venv/bin/python"
STUDY="${HERE}/3d_study.py"

if [[ ! -x "${PY}" ]]; then
  echo "Missing venv python at ${PY}"
  exit 1
fi

if [[ ! -f "${STUDY}" ]]; then
  echo "Could not find ${STUDY}"
  exit 1
fi

RUNS="${RUNS:-100}"
SEED_START="${SEED_START:-0}"
LOG_DIR="${LOG_DIR:-${HERE}/logs_var3_runs}"
mkdir -p "${LOG_DIR}"

run_batch () {
  local label="$1"
  local mode="$2"

  for ((i=0; i<RUNS; i++)); do
    local seed=$((SEED_START + i))
    local run_idx=$((i + 1))
    local logfile="${LOG_DIR}/${label}_run${run_idx}_seed${seed}.log"

    echo "[${label}] run ${run_idx}/${RUNS} | mode=${mode} | seed=${seed}"
    "${PY}" "${STUDY}" \
      "${seed}" \
      "${mode}" \
      2>&1 | tee "${logfile}"
  done
}

# Setting 2: large -> N=16384, Nb=4, Nh=4
run_batch "large_n16384_nb4_nh4" "large"

echo "Completed all runs. Logs in ${LOG_DIR}"
