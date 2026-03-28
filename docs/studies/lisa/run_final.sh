#!/usr/bin/env bash
set -euo pipefail

ROOT="/Users/avi/Documents/projects/LogPSplinePSD"
PY="${ROOT}/.venv/bin/python"
STUDY="${ROOT}/docs/studies/lisa"

if [[ ! -x "${PY}" ]]; then
  echo "Missing venv python at ${PY}" >&2
  exit 1
fi

cd "${STUDY}"

SHARED_ARGS=(
  --K 48
  --diff-order 2
  --knot-method uniform
  --wishart-window tukey
  --wishart-tukey-alpha 0.1
  --welch-window hann
  --coarse-Nc 8192
  --block-days 7
  --target-accept 0.8
  --max-tree-depth 10
  --n-warmup 2000
  --n-samples 1500
  --no-vi
  --keep-nc
)

# The run slug is deterministic from the shared args above.
RUN_X_SLUG="k48_d2_kmuniform_wwtukey0p1_ewhann_nc8192_bd7d_ta0.8_td10_viOff_tauOff"
RUN_X_YEAR_DIR="runs/run_x_d2_k48_uniform_no_excision"
RUN_X_90D_DIR="runs/run_x_90d_d2_k48_uniform_no_excision"
RUN_X_30D_DIR="runs/run_x_30d_d2_k48_uniform_no_excision"
RUN_X_YEAR_SEED="${RUN_X_YEAR_DIR}/${RUN_X_SLUG}/seed_0"

run_xyz_analysis() {
  local duration_days="$1"
  local outdir="$2"
  local label="$3"

  echo "=== ${label}: run_x — XYZ analysis (${duration_days} days) ==="
  "${PY}" main.py 0 \
    --outdir "${outdir}" \
    --duration-days "${duration_days}" \
    "${SHARED_ARGS[@]}"
  echo ""
}

run_xyz_analysis 30 "${RUN_X_30D_DIR}" "Step 1/4"
run_xyz_analysis 90 "${RUN_X_90D_DIR}" "Step 2/4"
run_xyz_analysis 365 "${RUN_X_YEAR_DIR}" "Step 3/4"

echo ""
echo "=== Step 4/4: paper figures ==="
echo "  Figure 1: XYZ posterior PSD"
echo "  Figure 2: XYZ posterior → AET (rotation, no resampling)"
echo "  Data overlay: raw periodogram from inference_data.nc"

"${PY}" paper_final_plots.py \
  --run-x "${RUN_X_YEAR_SEED}" \
  --freq-units \
  --data-overlay raw \
  --outdir paper_figs

echo ""
echo "Run directories:"
echo "  30d:  ${STUDY}/${RUN_X_30D_DIR}/${RUN_X_SLUG}/seed_0"
echo "  90d:  ${STUDY}/${RUN_X_90D_DIR}/${RUN_X_SLUG}/seed_0"
echo "  365d: ${STUDY}/${RUN_X_YEAR_DIR}/${RUN_X_SLUG}/seed_0"
echo ""
echo "=== Done. Figures in ${STUDY}/paper_figs/ ==="
