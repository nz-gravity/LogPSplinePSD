#!/bin/bash
#SBATCH --job-name=var3_gps_basis
#SBATCH --output=logs_var3_runs/var3_gps_basis_%A_%a.out
#SBATCH --error=logs_var3_runs/var3_gps_basis_%A_%a.err
#SBATCH --array=0-9
#SBATCH --time=1:00:00
#SBATCH --mem=6000M
#SBATCH --cpus-per-task=4

module --force purge
module load gcc/13.3.0 python/3.12.3
export XLA_FLAGS="--xla_force_host_platform_device_count=4"
PYTHON_BIN=/fred/oz303/avajpeyi/codes/LogPSplinePSD/.venv/bin/python

SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID:-0}   # fallback for local testing

START_SEED=$((SLURM_ARRAY_TASK_ID * 10))
END_SEED=$((START_SEED + 9))

for SEED in $(seq ${START_SEED} ${END_SEED}); do
    echo "Running seed=${SEED}, window=hann, K=20, mode=short_nb4, basis=GPS penalty"
    "${PYTHON_BIN}" 3d_study.py ${SEED} short_nb4 --K 20 --window hann --outdir out_var3_gps_basis
done
