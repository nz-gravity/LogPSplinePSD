#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${HERE}/../../.." && pwd)"
PY="${ROOT}/.venv/bin/python"
STUDY="${HERE}/var3_study.py"

if [[ ! -x "${PY}" ]]; then
  echo "Missing venv python at ${PY}"
  exit 1
fi

if [[ ! -f "${STUDY}" ]]; then
  echo "Could not find ${STUDY}"
  exit 1
fi

RUNS="${RUNS:-100}"
SEED_START="${SEED_START:-1}"
COARSE_NH_LONG="${COARSE_NH_LONG:-5}"
LOG_DIR="${LOG_DIR:-${HERE}/logs_var3_runs}"
mkdir -p "${LOG_DIR}"

run_batch () {
  local label="$1"
  local n="$2"
  local coarse_nh="$3"

  for ((i=0; i<RUNS; i++)); do
    local seed=$((SEED_START + i))
    local run_idx=$((i + 1))
    local logfile="${LOG_DIR}/${label}_run${run_idx}_seed${seed}.log"

    echo "[${label}] run ${run_idx}/${RUNS} | N=${n} | seed=${seed} | coarse_Nh=${coarse_nh}"
    "${PY}" "${STUDY}" \
      --N "${n}" \
      --seed "${seed}" \
      --coarse-Nh "${coarse_nh}" \
      2>&1 | tee "${logfile}"
  done
}

# 1) N = 2^11, no coarse graining (Wishart averaging still enabled in var3_study.py)
run_batch "n2048_avg_only" 2048 0

# 2) N = 2^16, coarse graining + averaging
run_batch "n65536_coarse_avg" 65536 "${COARSE_NH_LONG}"

echo "Completed all runs. Logs in ${LOG_DIR}"
