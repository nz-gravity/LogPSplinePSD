#!/bin/bash
#SBATCH --job-name=cg_study
#SBATCH --output=logs_cg_study/cg_%A_%a.out
#SBATCH --error=logs_cg_study/cg_%A_%a.err
#SBATCH --array=0-49
#SBATCH --time=02:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=4

module --force purge && ml gcc/12.3.0 python/3.11.3
source /fred/oz303/avajpeyi/venvs/jax_cpu_venv/bin/activate

SEED=${SLURM_ARRAY_TASK_ID:-0}

cd /fred/oz303/avajpeyi/LogPSplinePSD/docs/studies/multivar_psd
mkdir -p logs_cg_study

# Run all four coarse-graining conditions for this seed:
#   noNh  = no coarse-graining         (baseline, matches short_nb4 run)
#   Nh2   = average 2 adjacent bins
#   Nh4   = average 4 adjacent bins
#   Nh8   = average 8 adjacent bins
#
# mode: short_nb4  (N=2048, Nb=4)
# K=20, rect window
# outdir slug: seed_<s>_short_nb4_N2048_K20_rect_<nh_label>

for NH in 0 2 4 8; do
    echo "--- seed=${SEED}  coarse_nh=${NH} ---"
    python 3d_study.py ${SEED} short_nb4 --K 20 --window rect --coarse-nh ${NH}
done
