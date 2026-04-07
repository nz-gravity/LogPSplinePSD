#!/bin/bash
#SBATCH --job-name=eta_temper
#SBATCH --output=logs_eta/job_%A_%a.out
#SBATCH --error=logs_eta/job_%A_%a.err
#SBATCH --array=0-6
#SBATCH --time=1:00:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=4

# Phase 1: Static η grid sweep
# 7 array tasks (one per η value), each runs 20 seeds sequentially.
# η values: 0.01 0.05 0.1 0.3 0.5 0.75 1.0

module --force purge
module load gcc/13.3.0 python/3.12.3

export XLA_FLAGS="--xla_force_host_platform_device_count=4"

PYTHON_BIN=/fred/oz303/avajpeyi/codes/LogPSplinePSD/.venv/bin/python

cd /fred/oz303/avajpeyi/codes/LogPSplinePSD/docs/studies/multivar_psd
mkdir -p logs_eta

SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID:-0}

# Map array task to η value
ETA_VALUES=(0.01 0.05 0.1 0.3 0.5 0.75 1.0)
ETA=${ETA_VALUES[$SLURM_ARRAY_TASK_ID]}

echo "=== Array task ${SLURM_ARRAY_TASK_ID}: eta=${ETA}, seeds 0-19 ==="

for SEED in $(seq 0 19); do
    echo "Running eta=${ETA}, seed=${SEED}"
    "${PYTHON_BIN}" eta_tempering_study.py phase1 --seeds "${SEED}" --outdir out_eta
done
