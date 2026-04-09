#!/usr/bin/env bash
# Sanity sweep for LISA density-based knot placement with analytical PSD
# guidance enabled via run_lisa_mcmc(use_analytical_psd_for_knots=True).
#
# This targets the specific question:
#   do 30d / 90d / 180d / 365d runs behave sensibly when density knots are
#   guided by the analytical PSD while remaining empirically initialized?
#
# Notes:
# - No extra CLI flag is needed: the analytical PSD path is on by default in
#   docs/studies/lisa/utils/inference.py.
# - This only matters for --knot-method density. Uniform/log do not use the
#   residual density allocator in the same way.
#
# Usage:
#   bash run_density_analytical_sanity.sh
#   bash run_density_analytical_sanity.sh run_AO
#   SEED=1 bash run_density_analytical_sanity.sh
#   K_THETA_IM=4 bash run_density_analytical_sanity.sh
#   NULL_HALF_WIDTH_HZ=0.00075 bash run_density_analytical_sanity.sh
set -euo pipefail

ROOT="/Users/avi/Documents/projects/LogPSplinePSD"
PY="${ROOT}/.venv/bin/python"
STUDY="${ROOT}/docs/studies/lisa"

if [[ ! -x "${PY}" ]]; then
  echo "Missing venv python at ${PY}" >&2
  exit 1
fi

cd "${STUDY}"

SEED="${SEED:-0}"
TARGET="${1:-all}"

K=100
K_DELTA="${K_DELTA:-${K}}"
K_THETA_RE="${K_THETA_RE:-${K}}"
# LISA XYZ has analytically zero imaginary cross-spectrum, so the default here
# is the minimum knot count allowed by the current degree-2 spline setup.
K_THETA_IM="${K_THETA_IM:-2}"
DIFF_ORDER=2
KNOT_METHOD=density
NC=8192
BLOCK_DAYS=7
ALPHA=3.0
BETA=3.0
TARGET_ACCEPT=0.8
MAX_TREE_DEPTH=10
N_WARMUP=1500
N_SAMPLES=1000

NULL_HALF_WIDTH_HZ="${NULL_HALF_WIDTH_HZ:-0.0005}"
NULL_TAG="${NULL_HALF_WIDTH_HZ/./p}"
NULL_ARGS=(
  --null-excision
  "0.030:${NULL_HALF_WIDTH_HZ}"
  "0.060:${NULL_HALF_WIDTH_HZ}"
  "0.090:${NULL_HALF_WIDTH_HZ}"
)

BASE_ARGS=(
  --block-days "${BLOCK_DAYS}"
  --K "${K}"
  --K-delta "${K_DELTA}"
  --K-theta-re "${K_THETA_RE}"
  --K-theta-im "${K_THETA_IM}"
  --diff-order "${DIFF_ORDER}"
  --knot-method "${KNOT_METHOD}"
  --coarse-Nc "${NC}"
  --alpha-delta "${ALPHA}"
  --beta-delta "${BETA}"
  --wishart-window tukey
  --wishart-tukey-alpha 0.1
  --welch-window hann
  --target-accept "${TARGET_ACCEPT}"
  --max-tree-depth "${MAX_TREE_DEPTH}"
  --n-warmup "${N_WARMUP}"
  --n-samples "${N_SAMPLES}"
  --no-vi
  --keep-nc
  "${NULL_ARGS[@]}"
)

LABELS=(
  run_AM
  run_AN
  run_AO
  run_AP
)

DURATIONS=(30 90 180 365)

run_one() {
  local label="$1"
  local duration="$2"
  local outdir="runs/${label}_${duration}d_density_analytical_hw${NULL_TAG}_kdelta${K_DELTA}_ktre${K_THETA_RE}_ktim${K_THETA_IM}"
  local sentinel_glob="${outdir}/*/seed_${SEED}/compact_run_summary.json"

  if compgen -G "${sentinel_glob}" > /dev/null 2>&1; then
    echo "[SKIP] ${label} seed=${SEED} (result exists)"
    return 0
  fi

  echo ""
  echo "==================================================================="
  echo " ${label} | duration=${duration}d | seed=${SEED}"
  echo " K={delta=${K_DELTA}, theta_re=${K_THETA_RE}, theta_im=${K_THETA_IM}} | d=${DIFF_ORDER}"
  echo " knot_method=${KNOT_METHOD} | Nc=${NC}"
  echo " null excision=±${NULL_HALF_WIDTH_HZ} Hz | analytical PSD knots=ON"
  echo " outdir: ${outdir}"
  echo "==================================================================="

  "${PY}" main.py "${SEED}" \
    --outdir "${outdir}" \
    --duration-days "${duration}" \
    "${BASE_ARGS[@]}"

  local summary
  summary=$(find "${outdir}" -path "*/seed_${SEED}/compact_run_summary.json" | head -1 || true)
  if [[ -n "${summary}" ]]; then
    echo ""
    echo "  >>> Summary for ${label}:"
    "${PY}" - "${summary}" <<'PYEOF'
import json
import math
import sys
from pathlib import Path

summary_path = Path(sys.argv[1])
data = json.loads(summary_path.read_text())

def fmt(key: str) -> str:
    value = data.get(key, None)
    if value is None:
        return "n/a"
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        return f"{value:.4f}"
    return str(value)

print(f"  seed_dir:        {summary_path.parent}")
print(f"  coverage:        {fmt('coverage')}")
print(f"  riae_matrix:     {fmt('riae_matrix')}")
print(f"  coherence_riae:  {fmt('coherence_riae')}")
print(f"  rhat_max:        {fmt('rhat_max')}")
print(f"  ess_median:      {fmt('ess_median')}")
print(f"  divergences:     {fmt('n_divergences')}")
print(f"  runtime:         {fmt('runtime')}")
PYEOF
  fi
}

n=${#LABELS[@]}
for (( i=0; i<n; i++ )); do
  label="${LABELS[$i]}"
  if [[ "${TARGET}" == "all" || "${TARGET}" == "${label}" ]]; then
    run_one "${label}" "${DURATIONS[$i]}"
  fi
done

echo ""
echo "==================================================================="
echo " Density analytical-knot sanity sweep complete."
echo " Run directories are under: ${STUDY}/runs/"
echo "==================================================================="
