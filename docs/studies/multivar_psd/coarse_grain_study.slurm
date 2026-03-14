#!/bin/bash
#SBATCH --job-name=cg_study
#SBATCH --output=logs_cg_study/cg_%A_%a.out
#SBATCH --error=logs_cg_study/cg_%A_%a.err
#SBATCH --array=0-9
#SBATCH --time=1:00:00
#SBATCH --mem=2.5G
#SBATCH --cpus-per-task=4
module --force purge
module load gcc/13.3.0 python/3.12.3
PYTHON_BIN=/fred/oz303/avajpeyi/codes/LogPSplinePSD/.venv/bin/python
SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID:-0}   # fallback for local testing

cd /fred/oz303/avajpeyi/codes/LogPSplinePSD/docs/studies/multivar_psd
mkdir -p logs_cg_study

# 10 array tasks x 5 seeds = 50 seeds total
# Each task runs 5 seeds through all 4 coarse-graining conditions
# mode: short_nb4  (N=2048, Nb=4, K=20, rect)
# Nh=0 → no coarse-graining  (baseline)
# Nh=2 → average 2 adjacent bins
# Nh=4 → average 4 adjacent bins
# Nh=8 → average 8 adjacent bins

START_SEED=$((SLURM_ARRAY_TASK_ID * 5))
END_SEED=$((START_SEED + 4))

for SEED in $(seq ${START_SEED} ${END_SEED}); do
    for NH in 0 2 4 8; do
        echo "Running seed=${SEED}, coarse_nh=${NH}, K=20, mode=short_nb4, window=rect"
        "${PYTHON_BIN}" 3d_study.py ${SEED} short_nb4 --K 20 --window rect --coarse-nh ${NH}
    done
done
