#!/bin/bash
#SBATCH --job-name=var3_rect
#SBATCH --output=logs_var3_runs/var3_rect_%A_%a.out
#SBATCH --error=logs_var3_runs/var3_rect_%A_%a.err
#SBATCH --array=0-9
#SBATCH --time=1:00:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=4

module --force purge && ml gcc/12.3.0 python/3.11.3
source /fred/oz303/avajpeyi/venvs/jax_cpu_venv/bin/activate

SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID:-0}   # fallback for local testing

START_SEED=$((SLURM_ARRAY_TASK_ID * 10))
END_SEED=$((START_SEED + 9))

for SEED in $(seq ${START_SEED} ${END_SEED}); do
    echo "Running seed=${SEED}, window=rect, K=20, mode=short_nb4"
    python 3d_study.py ${SEED} short_nb4 --K 20 --window rect
done
