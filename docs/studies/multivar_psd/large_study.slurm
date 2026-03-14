#!/bin/bash
#SBATCH --job-name=large_study
#SBATCH --output=logs_large_study/large_%A_%a.out
#SBATCH --error=logs_large_study/large_%A_%a.err
#SBATCH --array=0-49
#SBATCH --time=02:30:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=4

module --force purge && ml gcc/12.3.0 python/3.11.3
source /fred/oz303/avajpeyi/venvs/jax_cpu_venv/bin/activate

SEED=${SLURM_ARRAY_TASK_ID:-0}

cd /fred/oz303/avajpeyi/LogPSplinePSD/docs/studies/multivar_psd
mkdir -p logs_large_study

# large mode: N=16384, Nb=4, coarse_Nh=4  (built into the mode config)
# K=20, rect window
# outdir slug: seed_<s>_large_N16384_K20_rect_Nh4

echo "--- seed=${SEED}  mode=large  N=16384  Nb=4  Nh=4  K=20  window=rect ---"
python 3d_study.py ${SEED} large --K 20 --window rect
