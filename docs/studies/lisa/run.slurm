#!/bin/bash
#SBATCH --job-name=lisa_multivar
#SBATCH --output=logs/lisa.out
#SBATCH --error=logs/lisa.err
#SBATCH --time=05:00:00
#SBATCH --mem=6G
#SBATCH --cpus-per-task=1

module --force purge && ml gcc/12.3.0 python/3.11.3 && source /fred/oz303/avajpeyi/venvs/jax_cpu_venv/bin/activate

python lisa_multivar.py
