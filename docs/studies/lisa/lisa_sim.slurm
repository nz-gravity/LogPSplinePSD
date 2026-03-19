#!/bin/bash
#SBATCH --job-name=lisa_sim
#SBATCH --output=logs_lisa_sim/lisa_%A_%a.out
#SBATCH --error=logs_lisa_sim/lisa_%A_%a.err
#SBATCH --array=0-99
#SBATCH --time=07:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=4

module --force purge
module load gcc/13.3.0 python/3.12.3

export XLA_FLAGS="--xla_force_host_platform_device_count=4"

PYTHON_BIN=/fred/oz303/avajpeyi/codes/LogPSplinePSD/.venv/bin/python

SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID:-0}   # fallback for local testing
SEED=${SLURM_ARRAY_TASK_ID}

echo "Running LISA sim study seed=${SEED}"
"${PYTHON_BIN}" main.py ${SEED} \
    --outdir out_lisa_sim \
    --duration-days 365 \
    --K 20 \
    --block-days 7 \
    --coarse-Nc 2048
