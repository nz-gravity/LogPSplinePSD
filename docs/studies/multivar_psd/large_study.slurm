#!/bin/bash
#SBATCH --job-name=sim_large
#SBATCH --output=logs_sim_large/job_%A_%a.out
#SBATCH --error=logs_sim_large/job_%A_%a.err
#SBATCH --array=0-4
#SBATCH --time=0:35:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=4

module --force purge
module load gcc/13.3.0 python/3.12.3

set -uo pipefail

export XLA_FLAGS="--xla_force_host_platform_device_count=4"

PYTHON_BIN=/fred/oz303/avajpeyi/codes/LogPSplinePSD/.venv/bin/python
SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID:-0}   # fallback for local testing

cd /fred/oz303/avajpeyi/codes/LogPSplinePSD/docs/studies/multivar_psd
mkdir -p logs_sim_large

# This study is intended to cover seeds 0..99:
#   5 array tasks x 20 seeds/task = 100 seeds total.
SEEDS_PER_TASK=${SEEDS_PER_TASK:-20}
SEED_OFFSET=${SEED_OFFSET:-0}
OUTDIR=${OUTDIR:-out_sim_large}

# mode: large (N=16384, Nb=4, coarse_Nh=4), VI-init NUTS, K=20, rect window
START_SEED=$((SEED_OFFSET + SLURM_ARRAY_TASK_ID * SEEDS_PER_TASK))
END_SEED=$((START_SEED + SEEDS_PER_TASK - 1))
FAILED_SEEDS=()

for SEED in $(seq ${START_SEED} ${END_SEED}); do
    SEED_OUTDIR="${OUTDIR}/seed_${SEED}_large_N16384_K20_rect_Nh4"
    if [[ -f "${SEED_OUTDIR}/compact_run_summary.json" ]]; then
        echo "Skipping seed=${SEED}; found ${SEED_OUTDIR}/compact_run_summary.json"
        continue
    fi
    echo "Running seed=${SEED}, mode=large, N=16384, Nb=4, Nh=4, K=20, window=rect"
    if ! "${PYTHON_BIN}" 3d_study.py "${SEED}" large --K 20 --window rect --outdir "${OUTDIR}"; then
        echo "Seed ${SEED} failed; continuing with the remaining seeds in this array task."
        FAILED_SEEDS+=("${SEED}")
    fi
done

if (( ${#FAILED_SEEDS[@]} > 0 )); then
    echo "Array task ${SLURM_ARRAY_TASK_ID} completed with failed seeds: ${FAILED_SEEDS[*]}"
    exit 1
fi
