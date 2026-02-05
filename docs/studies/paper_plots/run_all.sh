#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${HERE}/../../.." && pwd)"

PY="${ROOT}/.venv/bin/python"
if [[ ! -x "${PY}" ]]; then
  echo "Missing venv python at ${PY}"
  exit 1
fi

OUT_BASE="${ROOT}/docs/studies/paper_plots/results"

SHORT_N="${SHORT_N:-10000}"
LONG_N="${LONG_N:-60000}"
BLOCK_SIZE="${BLOCK_SIZE:-5000}"
CG_BINS="${CG_BINS:-512}"

LISA_DATA="${LISA_DATA:-${ROOT}/data/tdi.h5}"
LISA_DOWNSAMPLE="${LISA_DOWNSAMPLE:-1}"

FMIN="${FMIN:-1e-4}"
FMAX_RESTRICTED="${FMAX_RESTRICTED:-1e-2}"
FMAX_FULL="${FMAX_FULL:-1e-1}"

PAPER_MODE="${PAPER_MODE:-draft}"     # draft|paper
PAPER_OVERWRITE="${PAPER_OVERWRITE:-0}" # 0|1

DO_OVERWRITE=0
if [[ "${PAPER_OVERWRITE}" == "1" ]]; then
  DO_OVERWRITE=1
fi

VAR3_ARGS=(--block-size "${BLOCK_SIZE}")
LISA_ARGS=(--data "${LISA_DATA}" --downsample "${LISA_DOWNSAMPLE}" --block-size "${BLOCK_SIZE}")

case "${PAPER_MODE}" in
  draft)
    VAR3_ARGS+=(--samples 400 --warmup 400 --chains 2 --vi-steps 5000)
    LISA_ARGS+=(--samples 800 --warmup 800 --chains 2 --vi-steps 20000)
    ;;
  paper)
    VAR3_ARGS+=(--samples 1000 --warmup 1000 --chains 4 --vi-steps 20000)
    LISA_ARGS+=(--samples 4000 --warmup 4000 --chains 4 --vi-steps 200000)
    ;;
  *)
    echo "Unknown PAPER_MODE='${PAPER_MODE}' (expected draft|paper)"
    exit 1
    ;;
esac

echo "Paper jobs:"
echo "  mode=${PAPER_MODE} overwrite=${PAPER_OVERWRITE}"
echo "  VAR3: short=${SHORT_N} long=${LONG_N} block_size=${BLOCK_SIZE} coarse_bins=${CG_BINS}"
echo "  LISA: data=${LISA_DATA} downsample=${LISA_DOWNSAMPLE} fmin=${FMIN} restricted_fmax=${FMAX_RESTRICTED} full_fmax=${FMAX_FULL}"
echo "  out=${OUT_BASE}"
echo

run_var3 () {
  local name="$1"
  local n_time="$2"
  local coarse_bins="$3"
  local outdir="${OUT_BASE}/${name}"
  echo "==> VAR3 ${name}"
  local -a cmd
  cmd=(
    "${PY}" "${ROOT}/docs/studies/paper_plots/var3_paper_job.py"
    --outdir "${outdir}"
    --n-time "${n_time}"
    --coarse-bins "${coarse_bins}"
    "${VAR3_ARGS[@]}"
  )
  if [[ "${DO_OVERWRITE}" == "1" ]]; then
    cmd+=(--overwrite)
  fi
  "${cmd[@]}"
}

run_lisa () {
  local name="$1"
  local n_time="$2"
  local coarse_bins="$3"
  local fmax="$4"
  local outdir="${OUT_BASE}/${name}"
  echo "==> LISA ${name}"
  local -a cmd
  cmd=(
    "${PY}" "${ROOT}/docs/studies/paper_plots/lisa_paper_job.py"
    --outdir "${outdir}"
    --n-time "${n_time}"
    --coarse-bins "${coarse_bins}"
    --fmin "${FMIN}"
    --fmax "${fmax}"
    "${LISA_ARGS[@]}"
  )
  if [[ "${DO_OVERWRITE}" == "1" ]]; then
    cmd+=(--overwrite)
  fi
  "${cmd[@]}"
}

# 1) VAR(3) study
run_var3 "var3_short_raw" "${SHORT_N}" 0
run_var3 "var3_long_raw" "${LONG_N}" 0
run_var3 "var3_long_cg${CG_BINS}" "${LONG_N}" "${CG_BINS}"

# 2) LISA restricted band: [1e-4, 1e-2]
run_lisa "lisa_short_restricted_raw" "${SHORT_N}" 0 "${FMAX_RESTRICTED}"
run_lisa "lisa_long_restricted_raw" "${LONG_N}" 0 "${FMAX_RESTRICTED}"
run_lisa "lisa_long_restricted_cg${CG_BINS}" "${LONG_N}" "${CG_BINS}" "${FMAX_RESTRICTED}"

# 3) LISA full band (defaults to [1e-4, 1e-1])
run_lisa "lisa_short_full_raw" "${SHORT_N}" 0 "${FMAX_FULL}"
run_lisa "lisa_long_full_raw" "${LONG_N}" 0 "${FMAX_FULL}"
run_lisa "lisa_long_full_cg${CG_BINS}" "${LONG_N}" "${CG_BINS}" "${FMAX_FULL}"

echo
echo "Done. Outputs are under ${OUT_BASE}/"
