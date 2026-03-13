#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${HERE}/../../.." && pwd)"
CLUSTER_PYTHON_DEFAULT="/fred/oz303/avajpeyi/codes/LogPSplinePSD/.venv/bin/python"
LOCAL_PYTHON_DEFAULT="${ROOT}/.venv/bin/python"

if [[ -x "${CLUSTER_PYTHON_DEFAULT}" ]]; then
  PYTHON_DEFAULT="${CLUSTER_PYTHON_DEFAULT}"
else
  PYTHON_DEFAULT="${LOCAL_PYTHON_DEFAULT}"
fi

PYTHON_BIN="${PYTHON_BIN:-${PYTHON_DEFAULT}}"
MODULE_SETUP="${MODULE_SETUP:-module load gcc/13.3.0 python/3.12.3}"
STUDY_SCRIPT="${HERE}/3d_study.py"

ARRAY_ID="${SLURM_ARRAY_TASK_ID:-${ARRAY_ID:-0}}"
SEEDS_PER_TASK="${SEEDS_PER_TASK:-10}"
MODE="${MODE:-short_nb4}"
KNOTS="${KNOTS:-20}"
WINDOW="${WINDOW:-rect}"
LOG_DIR="${LOG_DIR:-${HERE}/logs_var3_runs}"

if [[ -n "${MODULE_SETUP}" ]]; then
  if type module >/dev/null 2>&1; then
    eval "${MODULE_SETUP}"
  else
    echo "Skipping module setup because 'module' is unavailable in this shell."
    echo "Requested setup: ${MODULE_SETUP}"
  fi
fi

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Could not find executable Python at ${PYTHON_BIN}"
  echo "Set PYTHON_BIN=/path/to/python if needed."
  exit 1
fi

if [[ ! -f "${STUDY_SCRIPT}" ]]; then
  echo "Could not find study script at ${STUDY_SCRIPT}"
  exit 1
fi

if ! [[ "${ARRAY_ID}" =~ ^[0-9]+$ ]]; then
  echo "ARRAY_ID must be a non-negative integer, got: ${ARRAY_ID}"
  exit 1
fi

if ! [[ "${SEEDS_PER_TASK}" =~ ^[1-9][0-9]*$ ]]; then
  echo "SEEDS_PER_TASK must be a positive integer, got: ${SEEDS_PER_TASK}"
  exit 1
fi

mkdir -p "${LOG_DIR}"

START_SEED=$((ARRAY_ID * SEEDS_PER_TASK))
END_SEED=$((START_SEED + SEEDS_PER_TASK - 1))
RUN_LOG="${LOG_DIR}/var3_rect_local_array${ARRAY_ID}.log"

echo "Running local var3 batch"
echo "  array_id=${ARRAY_ID}"
echo "  seeds=${START_SEED}..${END_SEED}"
echo "  mode=${MODE}, K=${KNOTS}, window=${WINDOW}"
echo "  module_setup=${MODULE_SETUP}"
echo "  python=${PYTHON_BIN}"
echo "  log=${RUN_LOG}"

for seed in $(seq "${START_SEED}" "${END_SEED}"); do
  echo "Running seed=${seed}, window=${WINDOW}, K=${KNOTS}, mode=${MODE}" | tee -a "${RUN_LOG}"
  "${PYTHON_BIN}" "${STUDY_SCRIPT}" "${seed}" "${MODE}" --K "${KNOTS}" --window "${WINDOW}" 2>&1 | tee -a "${RUN_LOG}"
done

echo "Completed. Log written to ${RUN_LOG}"
