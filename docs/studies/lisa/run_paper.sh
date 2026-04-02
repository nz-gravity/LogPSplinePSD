#!/usr/bin/env bash
# Paper production runs: 30d, 90d, 365d.
#
# Configuration chosen from CI-width sweep (run_AL):
#   K=100, d2, uniform knots, Nc=8192, α/β=3.0, null excision ±1mHz
#
# Results from 30d test (run_AL):
#   Coverage: 86.5%  |  CI width: 5.8%  |  R-hat max: 1.23
#   No divergences, 0% max tree depth — best sampler behaviour in sweep.
#
# Usage:
#   bash run_paper.sh               # all three durations sequentially
#   bash run_paper.sh run_paper_30d # single duration
set -euo pipefail

ROOT="/Users/avi/Documents/projects/LogPSplinePSD"
PY="${ROOT}/.venv/bin/python"
STUDY="${ROOT}/docs/studies/lisa"

if [[ ! -x "${PY}" ]]; then
  echo "Missing venv python at ${PY}" >&2; exit 1
fi
cd "${STUDY}"

# ---------------------------------------------------------------------------
# Shared settings (run_AL configuration)
# ---------------------------------------------------------------------------
K=100
DIFF_ORDER=2
KNOT_METHOD=uniform
NC=8192
ALPHA=3.0
BETA=3.0
BLOCK_DAYS=7
TARGET_ACCEPT=0.8
MAX_TREE_DEPTH=10
N_WARMUP=1500   # more warmup than sweep (was 1000)
N_SAMPLES=1000  # more samples than sweep (was 500)
NULL_ARGS=(--null-excision 0.030:0.001 0.060:0.001 0.090:0.001)

BASE_ARGS=(
  --diff-order    ${DIFF_ORDER}
  --knot-method   ${KNOT_METHOD}
  --coarse-Nc     ${NC}
  --K             ${K}
  --alpha-delta   ${ALPHA}
  --beta-delta    ${BETA}
  --block-days    ${BLOCK_DAYS}
  --wishart-window tukey
  --wishart-tukey-alpha 0.1
  --welch-window  hann
  --target-accept ${TARGET_ACCEPT}
  --max-tree-depth ${MAX_TREE_DEPTH}
  --n-warmup      ${N_WARMUP}
  --n-samples     ${N_SAMPLES}
  --no-vi
  --keep-nc
  "${NULL_ARGS[@]}"
)

SLUG="k${K}_d${DIFF_ORDER}_km${KNOT_METHOD}_nc${NC}_null_excision"

# ---------------------------------------------------------------------------
# Duration configs
# ---------------------------------------------------------------------------
LABELS=(   run_paper_30d   run_paper_90d   run_paper_365d )
DURATIONS=( 30              90              365            )

TARGET="${1:-all}"

run_one() {
  local label="$1"
  local duration="$2"
  local outdir="runs/${label}_${SLUG}"

  echo ""
  echo "==================================================================="
  echo " ${label} | ${duration} days | K=${K} | Nc=${NC} | α/β=${ALPHA}/${BETA} | null excision"
  echo " outdir: ${outdir}"
  echo "==================================================================="

  "${PY}" main.py 0 \
    --outdir    "${outdir}" \
    --duration-days "${duration}" \
    "${BASE_ARGS[@]}"

  # CI-width report
  local npz
  npz=$(find "${outdir}" -name "compact_ci_curves.npz" 2>/dev/null | head -1 || true)
  if [[ -n "${npz}" ]]; then
    local seed_dir; seed_dir=$(dirname "${npz}")
    echo ""
    echo "  >>> Results for ${label}:"
    "${PY}" - "${seed_dir}" <<'PYEOF'
import sys, json
from pathlib import Path
import numpy as np

sd = Path(sys.argv[1])
d  = np.load(sd / "compact_ci_curves.npz")
q05 = np.array([d["psd_real_q05"][:, i, i] for i in range(3)]).T
q95 = np.array([d["psd_real_q95"][:, i, i] for i in range(3)]).T
q50 = np.array([d["psd_real_q50"][:, i, i] for i in range(3)]).T
rel  = (q95 - q05) / (q50 + 1e-300)
print(f"  CI width (median): {np.nanmedian(rel)*100:.2f}%")

sj = sd / "compact_run_summary.json"
if sj.exists():
    s = json.load(open(sj))
    print(f"  Coverage:          {s.get('coverage', float('nan')):.4f}")
    print(f"  RIAE:              {s.get('riae', 'n/a')}")
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
echo " Paper runs complete. Directories under: ${STUDY}/runs/"
echo " Make plots: python paper_final_plots.py --run-dir runs/run_paper_365d_${SLUG}/..."
echo "==================================================================="
