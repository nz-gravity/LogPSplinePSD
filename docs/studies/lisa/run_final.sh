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
  --duration-days 365
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
RUN_X_DIR="runs/run_x_d2_k48_uniform_no_excision"
RUN_X_SEED="${RUN_X_DIR}/${RUN_X_SLUG}/seed_0"
echo "=== Step 1/2: run_x — XYZ analysis ==="
"${PY}" main.py 0 \
  --outdir "${RUN_X_DIR}" \
  "${SHARED_ARGS[@]}"

echo ""
echo "=== Step 2/2: paper figures ==="
echo "  Figure 1: XYZ posterior PSD"
echo "  Figure 2: XYZ posterior → AET (rotation, no resampling)"
echo "  Data overlay: raw periodogram from inference_data.nc"

"${PY}" paper_final_plots.py \
  --run-x "${RUN_X_SEED}" \
  --freq-units \
  --data-overlay raw \
  --outdir paper_figs

echo ""
echo "=== Done. Figures in ${STUDY}/paper_figs/ ==="
