#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${HERE}"

PYTHON_BIN="${PYTHON_BIN:-python}"
OUTDIR="${OUTDIR:-out_var3_headnode}"
SEEDS="${SEEDS:-0 1 2}"
KNOTS="${KNOTS:-20}"
N_TIME="${N_TIME:-16384}"
WINDOW="${WINDOW:-rect}"
GRID="${GRID:-default}"
QUICK="${QUICK:-1}"
COLLECT="${COLLECT:-1}"
LABEL_PREFIX="${LABEL_PREFIX:-head}"
N_SAMPLES="${N_SAMPLES:-}"
N_WARMUP="${N_WARMUP:-}"
NUM_CHAINS="${NUM_CHAINS:-}"
VI_STEPS="${VI_STEPS:-}"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "Could not find python executable: ${PYTHON_BIN}"
  exit 1
fi

COMMON_ARGS=(
  --K "${KNOTS}"
  --window "${WINDOW}"
  --outdir "${OUTDIR}"
  --n-time "${N_TIME}"
)

if [[ "${QUICK}" == "1" ]]; then
  COMMON_ARGS+=(--quick)
fi
if [[ -n "${N_SAMPLES}" ]]; then
  COMMON_ARGS+=(--n-samples "${N_SAMPLES}")
fi
if [[ -n "${N_WARMUP}" ]]; then
  COMMON_ARGS+=(--n-warmup "${N_WARMUP}")
fi
if [[ -n "${NUM_CHAINS}" ]]; then
  COMMON_ARGS+=(--num-chains "${NUM_CHAINS}")
fi
if [[ -n "${VI_STEPS}" ]]; then
  COMMON_ARGS+=(--vi-steps "${VI_STEPS}")
fi

LABELS=()
NBS=()
NHS=()

case "${GRID}" in
  default)
    LABELS=(
      "${LABEL_PREFIX}_nb4_nh1"
      "${LABEL_PREFIX}_nb4_nh4"
      "${LABEL_PREFIX}_nb8_nh1"
    )
    NBS=(4 4 8)
    NHS=(0 4 0)
    ;;
  rankdef)
    LABELS=(
      "${LABEL_PREFIX}_nb4_nh1"
      "${LABEL_PREFIX}_nb4_nh4"
      "${LABEL_PREFIX}_nb8_nh1"
      "${LABEL_PREFIX}_nb2_nh1"
    )
    NBS=(4 4 8 2)
    NHS=(0 4 0 0)
    ;;
  full)
    LABELS=(
      "${LABEL_PREFIX}_nb4_nh1"
      "${LABEL_PREFIX}_nb4_nh2"
      "${LABEL_PREFIX}_nb4_nh4"
      "${LABEL_PREFIX}_nb8_nh1"
      "${LABEL_PREFIX}_nb8_nh2"
      "${LABEL_PREFIX}_nb16_nh1"
      "${LABEL_PREFIX}_nb2_nh1"
    )
    NBS=(4 4 4 8 8 16 2)
    NHS=(0 2 4 0 2 0 0)
    ;;
  *)
    echo "Unknown GRID='${GRID}'. Use one of: default, rankdef, full."
    exit 1
    ;;
esac

echo "Running headnode sweep"
echo "  dir:    ${HERE}"
echo "  python: ${PYTHON_BIN}"
echo "  outdir: ${OUTDIR}"
echo "  seeds:  ${SEEDS}"
echo "  grid:   ${GRID}"
echo "  quick:  ${QUICK}"
if [[ -n "${N_SAMPLES}${N_WARMUP}${NUM_CHAINS}${VI_STEPS}" ]]; then
  echo "  overrides: n_samples=${N_SAMPLES:-default} n_warmup=${N_WARMUP:-default} num_chains=${NUM_CHAINS:-default} vi_steps=${VI_STEPS:-default}"
fi

for seed in ${SEEDS}; do
  for idx in "${!LABELS[@]}"; do
    label="${LABELS[$idx]}"
    nb="${NBS[$idx]}"
    nh="${NHS[$idx]}"

    echo
    echo "=== seed=${seed} label=${label} Nb=${nb} Nh=${nh:-0} ==="
    PYTHONUNBUFFERED=1 "${PYTHON_BIN}" 3d_study.py \
      "${seed}" \
      large \
      "${COMMON_ARGS[@]}" \
      --nb-override "${nb}" \
      --coarse-nh "${nh}" \
      --label "${label}"
  done
done

if [[ "${COLLECT}" == "1" ]]; then
  echo
  echo "Collecting summaries from ${OUTDIR}"
  "${PYTHON_BIN}" collect_results.py \
    --results-dir "${OUTDIR}" \
    --glob "*${LABEL_PREFIX}_*" \
    --out "results_${LABEL_PREFIX}_${GRID}"
fi
