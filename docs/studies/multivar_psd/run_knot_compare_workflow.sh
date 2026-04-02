#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${HERE}/../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${ROOT}/.venv/bin/python}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python executable not found: ${PYTHON_BIN}"
  echo "Set PYTHON_BIN=/path/to/python"
  exit 1
fi

SEED="${SEED:-0}"
N="${N:-16384}"          # 2^14 by default
NB="${NB:-4}"
K="${K:-20}"
WINDOW="${WINDOW:-rect}"
WITH_POSTERIOR="${WITH_POSTERIOR:-1}"  # 1=yes, 0=no
POSTERIOR_SAMPLES="${POSTERIOR_SAMPLES:-200}"
POSTERIOR_WARMUP="${POSTERIOR_WARMUP:-200}"
POSTERIOR_CHAINS="${POSTERIOR_CHAINS:-1}"
POSTERIOR_VI_STEPS="${POSTERIOR_VI_STEPS:-5000}"
OUTDIR="${OUTDIR:-${HERE}/out_knot_diagnostics/var2_3d_seed${SEED}_N${N}_Nb${NB}_K${K}}"

echo "Running knot-comparison workflow"
echo "  seed=${SEED} N=${N} Nb=${NB} K=${K} window=${WINDOW}"
echo "  with_posterior=${WITH_POSTERIOR}"
echo "  outdir=${OUTDIR}"

PYTHONPATH="${ROOT}/src" \
"${PYTHON_BIN}" "${HERE}/plot_knot_scoring_diagnostics.py" \
  --seed "${SEED}" \
  --N "${N}" \
  --Nb "${NB}" \
  --K "${K}" \
  --window "${WINDOW}" \
  --outdir "${OUTDIR}" \
  $(
    if [[ "${WITH_POSTERIOR}" == "1" ]]; then
      printf -- "--with-posterior --posterior-samples %s --posterior-warmup %s --posterior-chains %s --posterior-vi-steps %s" \
        "${POSTERIOR_SAMPLES}" "${POSTERIOR_WARMUP}" "${POSTERIOR_CHAINS}" "${POSTERIOR_VI_STEPS}"
    fi
  )

PYTHONPATH="${ROOT}/src" \
"${PYTHON_BIN}" "${HERE}/analyse_knot_scoring_compare.py" --outdir "${OUTDIR}"

if [[ "${WITH_POSTERIOR}" == "1" ]]; then
  S_IDATA="${OUTDIR}/posterior_spectral/inference_data.nc"
  C_IDATA="${OUTDIR}/posterior_cholesky/inference_data.nc"
  if [[ -f "${S_IDATA}" && -f "${C_IDATA}" ]]; then
    PYTHONPATH="${ROOT}/src" \
    "${PYTHON_BIN}" "${HERE}/overplot_posteriors_3x3.py" \
      --spectral-idata "${S_IDATA}" \
      --cholesky-idata "${C_IDATA}" \
      --out "${OUTDIR}/posterior_overplot_3x3.png"
  else
    echo "Skipping posterior overplot; missing inference_data.nc files."
  fi
fi

echo "Done."
echo "Key files:"
echo "  ${OUTDIR}/s_matrix_knots_3x3.png"
echo "  ${OUTDIR}/analysis_summary.md"
if [[ "${WITH_POSTERIOR}" == "1" ]]; then
  echo "  ${OUTDIR}/posterior_comparison_3x3.png"
  echo "  ${OUTDIR}/posterior_overplot_3x3.png"
fi
