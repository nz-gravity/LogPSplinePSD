#!/usr/bin/env bash
# η-tempering validation test: Nb=16, Nh=2 with eta=1.0 vs eta='auto'
#
# This script validates that eta-tempering is responsible for the high
# coverage observed in the cluster results (91%+).
#
# Runs the same configuration (Nb=16, Nh=2) with two different eta values:
#   eta=1.0    (no tempering) → expect low coverage (~30-50%)
#   eta='auto' (adaptive)      → expect high coverage (~91%)
#
# Each eta setting is tested with 20 seeds for statistical stability.
#
# Usage: bash run_local_check.sh

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
export PYTHON_BIN="${ROOT}/.venv/bin/python"
export PROJECT_ROOT="${ROOT}"
export STUDY_DIR="${ROOT}/docs/studies/multivar_psd"

cd "${STUDY_DIR}"

export XLA_FLAGS="--xla_force_host_platform_device_count=4"

# Small MCMC budget for a quick check (~5-10 min per run locally)
export N_SAMPLES=500
export N_WARMUP=1000
export NUM_CHAINS=4
export VI_STEPS=50000

export OUTDIR="out_var3_eta_validation"
export WINDOW="rect"
export KNOTS="30"
export MODE="large"
export N_TIME="16384"
export NB="16"
export COARSE_NH="2"

mkdir -p "${OUTDIR}"

run_condition() {
    local SEED="$1"
    local ETA="$2"
    local ETA_LABEL="$3"

    local LB=$(( N_TIME / NB ))
    local RAW_NELL=$(( LB / 2 ))
    local NH_LABEL="Nh${COARSE_NH}"
    local NB_SUFFIX="_Nb${NB}"

    local RUN_DIR="${OUTDIR}/seed_${SEED}_${MODE}_N${N_TIME}_K${KNOTS}_${WINDOW}_${NH_LABEL}${NB_SUFFIX}_eta${ETA_LABEL}"
    if [[ -f "${RUN_DIR}/compact_run_summary.json" ]]; then
        echo "Already done: ${RUN_DIR}"
        return 0
    fi

    echo ""
    echo "=== seed=${SEED}  Nb=${NB}  Nh=${COARSE_NH}  eta=${ETA} ==="
    "${PYTHON_BIN}" 3d_study.py "${SEED}" "${MODE}" \
        --K "${KNOTS}" \
        --window "${WINDOW}" \
        --outdir "${OUTDIR}" \
        --n-time "${N_TIME}" \
        --nb-override "${NB}" \
        --coarse-nh "${COARSE_NH}" \
        --label "eta${ETA_LABEL}" \
        --n-samples "${N_SAMPLES}" \
        --n-warmup "${N_WARMUP}" \
        --num-chains "${NUM_CHAINS}" \
        --vi-steps "${VI_STEPS}" \
        --eta "${ETA}"
}

# Run 20 seeds with eta=1.0 (no tempering)
echo "========== Running eta=1.0 (no tempering) =========="
for SEED in {0..19}; do
    run_condition "${SEED}" "1.0" "1p0"
done

# Run 20 seeds with eta='auto' (adaptive tempering)
echo ""
echo "========== Running eta='auto' (adaptive tempering) =========="
for SEED in {0..19}; do
    run_condition "${SEED}" "auto" "auto"
done

echo ""
echo "=== Results Summary ==="
"${PYTHON_BIN}" - <<'PY'
import glob, json, numpy as np
from pathlib import Path

results = []
for p in sorted(glob.glob("out_var3_eta_validation/**/compact_run_summary.json", recursive=True)):
    d = json.load(open(p))
    results.append(d)

if not results:
    print("No results found.")
    exit()

results.sort(key=lambda r: (r.get("sampling_eta_channel_0", r.get("eta", 0)), r.get("seed", 0)))

# Group by eta value and compute statistics
eta_groups = {}
for r in results:
    eta = r.get("sampling_eta_channel_0", r.get("eta", "?"))
    if eta not in eta_groups:
        eta_groups[eta] = []
    eta_groups[eta].append(r)

print(f"{'η':>8} {'seed':>5} {'coverage':>10} {'riae':>8} {'rhat':>7}")
print("-" * 45)
for eta in sorted(eta_groups.keys(), key=lambda x: (isinstance(x, str), x)):
    for r in eta_groups[eta]:
        seed = r.get("seed", "?")
        cov  = r.get("coverage", float("nan"))
        riae = r.get("riae", r.get("riae_matrix", float("nan")))
        rhat = r.get("rhat_max", float("nan"))
        print(f"{str(eta):>8} {str(seed):>5} {cov:>10.4f} {riae:>8.4f} {rhat:>7.4f}")

print("\n=== Summary by η ===")
print(f"{'η':>8} {'N':>3} {'coverage':>12} {'riae':>10} {'rhat':>8}")
print("-" * 45)
for eta in sorted(eta_groups.keys(), key=lambda x: (isinstance(x, str), x)):
    group = eta_groups[eta]
    covs  = [r.get("coverage", float("nan")) for r in group]
    riaes = [r.get("riae", r.get("riae_matrix", float("nan"))) for r in group]
    rhats = [r.get("rhat_max", float("nan")) for r in group]

    cov_mean = np.nanmean(covs)
    cov_std  = np.nanstd(covs)
    riae_mean = np.nanmean(riaes)
    riae_std  = np.nanstd(riaes)
    rhat_max  = np.nanmax(rhats)

    print(f"{str(eta):>8} {len(group):>3} {cov_mean:>7.4f}±{cov_std:.4f} {riae_mean:>7.4f}±{riae_std:.4f} {rhat_max:>8.4f}")
PY
