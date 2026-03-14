#!/bin/bash
#SBATCH --job-name=large_study
#SBATCH --output=logs_large_study/large_%A_%a.out
#SBATCH --error=logs_large_study/large_%A_%a.err
#SBATCH --array=0-9
#SBATCH --time=2:30:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=4
module --force purge
module load gcc/13.3.0 python/3.12.3
PYTHON_BIN=/fred/oz303/avajpeyi/codes/LogPSplinePSD/.venv/bin/python
SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID:-0}   # fallback for local testing

cd /fred/oz303/avajpeyi/codes/LogPSplinePSD/docs/studies/multivar_psd
mkdir -p logs_large_study

# 10 array tasks x 5 seeds = 50 seeds total
# large mode: N=16384, Nb=4, coarse_Nh=4 (built into mode config), K=20, rect
# outdir slug: seed_<s>_large_N16384_K20_rect_Nh4

START_SEED=$((SLURM_ARRAY_TASK_ID * 5))
END_SEED=$((START_SEED + 4))

for SEED in $(seq ${START_SEED} ${END_SEED}); do
    echo "Running seed=${SEED}, mode=large, N=16384, Nb=4, Nh=4, K=20, window=rect"
    "${PYTHON_BIN}" 3d_study.py ${SEED} large --K 20 --window rect
done
