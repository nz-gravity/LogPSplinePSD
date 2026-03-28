#!/usr/bin/env bash
# CI-width sweep: vary K, smoothing prior (alpha_delta / beta_delta), and coarse-graining.
# All runs use 30-day duration (faster iteration).
# Run labels continue from run_y: run_z, run_AA … run_AH
#
# 30-day raw bins in [fmin,fmax]: 60420  →  default CG Nc=8192 → 10× merge
#
# Hypotheses to test:
#   (a) Increase K          → more local flexibility → wider CIs
#   (b) Looser prior        → smaller alpha/beta_delta → heavier tail → wider CIs
#   (c) Less coarse-graining → fewer bins merged → weaker likelihood per bin → wider CIs
#       run_AF: Nc=12000  (5× merge, half the default merging)
#       run_AG: Nc=30000  (2× merge)
#       run_AH: Nc=60420  (no CG — all raw bins)
#
# Usage:
#   bash run_ci_sweep.sh              # run all
#   bash run_ci_sweep.sh run_AF       # run a single label
set -euo pipefail

ROOT="/Users/avi/Documents/projects/LogPSplinePSD"
PY="${ROOT}/.venv/bin/python"
STUDY="${ROOT}/docs/studies/lisa"

if [[ ! -x "${PY}" ]]; then
  echo "Missing venv python at ${PY}" >&2
  exit 1
fi
cd "${STUDY}"

DURATION=30

# Shared args — coarse-Nc is set per-run below
BASE_ARGS=(
  --duration-days ${DURATION}
  --diff-order 2
  --knot-method uniform
  --wishart-window tukey
  --wishart-tukey-alpha 0.1
  --welch-window hann
  --block-days 7
  --target-accept 0.8
  --max-tree-depth 10
  --n-warmup 1000
  --n-samples 500
  --no-vi
  --keep-nc
)

# ---------------------------------------------------------------------------
# Run table  (30-day raw bins in band: 60420; default CG: Nc=8192 → 10× merge)
# Columns: LABEL  K   Nc     ALPHA  BETA  NULL_EXCISION  NOTE
# ---------------------------------------------------------------------------
#  run_z   48   8192   3.0    3.0   none      BASELINE (run_x 30d settings, fewer samples)
#  run_AA  100  8192   3.0    3.0   none      2× more knots
#  run_AB  200  8192   3.0    3.0   none      4× more knots
#  run_AC  48   8192   1.0    1.0   none      Tighter prior (sanity check → expect narrower)
#  run_AD  48   8192   0.1    0.1   none      Very loose prior (heavy tail)
#  run_AE  100  8192   0.1    0.1   none      Large K + very loose prior
#  run_AF  48   12000  3.0    3.0   none      Half the merging (5×)
#  run_AG  48   30000  3.0    3.0   none      Minimal merging (2×)
#  run_AH  48   60420  3.0    3.0   none      No coarse-graining (1×, all raw bins)
#  run_AI  48   8192   3.0    3.0   standard  Null excision (±1mHz at 0.030/0.060/0.090 Hz)
#  run_AJ  100  8192   0.1    0.1   standard  Best model combo + null excision
#  run_AK  200  8192   3.0    3.0   standard  K=200 + null excision (target ≥90% coverage)
#  run_AL  100  8192   3.0    3.0   standard  K=100 + null excision (good prior geometry)
# ---------------------------------------------------------------------------
# NOTE: run_AJ (K=100, α/β=0.1, null excision) showed pathological sampler behaviour:
# channels 1&2 hit max tree depth 99.8% of the time (step size 0.002 vs 0.38 for
# channel 0). Root cause: loose prior α/β=0.1 allows δ→0, making the off-diagonal
# posterior near-degenerate. Standard prior (3.0/3.0) avoids this — see run_AL/AK.

LABELS=( run_z  run_AA run_AB run_AC run_AD run_AE run_AF  run_AG  run_AH  run_AI  run_AJ  run_AK  run_AL )
KVALS=(  48     100    200    48     48     100    48      48      48      48      100     200     100    )
NC_VALS=( 8192  8192   8192   8192   8192   8192   12000   30000   60420   8192    8192    8192    8192   )
ALPHAS=( 3.0    3.0    3.0    1.0    0.1    0.1    3.0     3.0     3.0     3.0     0.1     3.0     3.0    )
BETAS=(  3.0    3.0    3.0    1.0    0.1    0.1    3.0     3.0     3.0     3.0     0.1     3.0     3.0    )
# "none" = no null excision; "standard" = LISA TDI-2 nulls 0.030/0.060/0.090 Hz ±1mHz
NULL_EXCISION=( none   none   none   none   none   none   none    none    none    standard  standard  standard  standard )
NOTES=(
  "baseline"
  "k100"
  "k200"
  "tight_prior"
  "loose_prior"
  "k100_loose"
  "cg5x"
  "cg2x"
  "nocg"
  "null_excision"
  "k100_loose_nullexc"
  "k200_nullexc"
  "k100_nullexc"
)

TARGET="${1:-all}"

run_one() {
  local label="$1"
  local k="$2"
  local nc="$3"
  local alpha="$4"
  local beta="$5"
  local null_exc="$6"
  local note="$7"
  local outdir="runs/${label}_30d_d2_k${k}_nc${nc}_a${alpha}_b${beta}"

  echo ""
  echo "==================================================================="
  echo " ${label} | K=${k} | Nc=${nc} | alpha=${alpha} | beta=${beta} | null=${null_exc} | ${note}"
  echo " outdir: ${outdir}"
  echo "==================================================================="

  local null_args=()
  if [[ "${null_exc}" == "standard" ]]; then
    null_args=( --null-excision 0.030:0.001 0.060:0.001 0.090:0.001 )
  fi

  "${PY}" main.py 0 \
    --outdir "${outdir}" \
    --K "${k}" \
    --alpha-delta "${alpha}" \
    --beta-delta "${beta}" \
    --coarse-Nc "${nc}" \
    "${null_args[@]+"${null_args[@]}"}" \
    "${BASE_ARGS[@]}"

  # Quick CI-width report
  local npz
  npz=$(find "${outdir}" -name "compact_ci_curves.npz" 2>/dev/null | head -1 || true)
  if [[ -n "${npz}" ]]; then
    local seed_dir
    seed_dir=$(dirname "${npz}")
    echo ""
    echo "  >>> CI-width summary for ${label}:"
    "${PY}" - "${seed_dir}" <<'PYEOF'
import sys, numpy as np
d = np.load(f"{sys.argv[1]}/compact_ci_curves.npz")
# psd_real_q* shape: (n_freq, p, p) — diagonal elements are the auto-PSDs
q05 = np.array([d["psd_real_q05"][:, i, i] for i in range(3)]).T  # (n_freq, 3)
q95 = np.array([d["psd_real_q95"][:, i, i] for i in range(3)]).T
q50 = np.array([d["psd_real_q50"][:, i, i] for i in range(3)]).T
rel = (q95 - q05) / (q50 + 1e-300)
per_ch = np.nanmedian(rel, axis=0)
print(f"  Per-channel median 90% rel CI width: {per_ch}")
print(f"  Overall median: {np.nanmedian(rel):.4f}  ({np.nanmedian(rel)*100:.2f}%)")
PYEOF
  fi
}

n=${#LABELS[@]}
for (( i=0; i<n; i++ )); do
  label="${LABELS[$i]}"
  if [[ "${TARGET}" == "all" || "${TARGET}" == "${label}" ]]; then
    run_one "${label}" "${KVALS[$i]}" "${NC_VALS[$i]}" "${ALPHAS[$i]}" "${BETAS[$i]}" "${NULL_EXCISION[$i]}" "${NOTES[$i]}"
  fi
done

echo ""
echo "==================================================================="
echo " Sweep complete. Run directories under: ${STUDY}/runs/"
echo " Re-run a single config: bash run_ci_sweep.sh run_AF"
echo "==================================================================="
