#!/bin/bash
#SBATCH --job-name=sim_short
#SBATCH --output=logs_sim_short/job_%A_%a.out
#SBATCH --error=logs_sim_short/job_%A_%a.err
#SBATCH --array=0-9
#SBATCH --time=1:00:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=4

module --force purge
module load gcc/13.3.0 python/3.12.3

export XLA_FLAGS="--xla_force_host_platform_device_count=4"

PYTHON_BIN=/fred/oz303/avajpeyi/codes/LogPSplinePSD/.venv/bin/python

cd /fred/oz303/avajpeyi/codes/LogPSplinePSD/docs/studies/multivar_psd
mkdir -p logs_sim_short

SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID:-0}   # fallback for local testing

# 10 array tasks x 10 seeds = 100 seeds total
# mode: short_nb4 (N=2048, Nb=4), VI-init NUTS, K=20, rect window
START_SEED=$((SLURM_ARRAY_TASK_ID * 10))
END_SEED=$((START_SEED + 9))

for SEED in $(seq ${START_SEED} ${END_SEED}); do
    echo "Running seed=${SEED}, mode=short_nb4, N=2048, Nb=4, K=20, window=rect"
    "${PYTHON_BIN}" 3d_study.py ${SEED} short_nb4 --K 20 --window rect --outdir out_sim_short
done
