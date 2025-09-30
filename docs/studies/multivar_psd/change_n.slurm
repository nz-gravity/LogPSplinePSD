#!/bin/bash
#SBATCH --job-name=sim_study
#SBATCH --output=sim_study_%A_%a.out
#SBATCH --error=sim_study_%A_%a.err
#SBATCH --array=1-10
#SBATCH --time=00:20:00
#SBATCH --mem=1G
#SBATCH --cpus-per-task=1


module --force purge && ml gcc/12.3.0 python/3.11.3 && source /fred/oz303/avajpeyi/venvs/jax_cpu_venv/bin/activate

# Set N values
N_VALUES=(2048 1024 512 256)
K=15
NREP=10 # Number of replicates per array task

OUTDIR="out_changing_n"

# Default SLURM_ARRAY_TASK_ID to 0 for local testing if not set
SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID:-0}

# Each array task runs 10 replicates
if [ "$SLURM_ARRAY_TASK_ID" -eq 0 ]; then
    START_REP=1
    END_REP=$NREP
else
    START_REP=$(( ($SLURM_ARRAY_TASK_ID - 1) * $NREP + 1 ))
    END_REP=$(( $SLURM_ARRAY_TASK_ID * $NREP ))
fi

for N in "${N_VALUES[@]}"; do
    for REPLICATE in $(seq $START_REP $END_REP); do
        SEED=$REPLICATE
        echo "python simulation_study.py --outdir $OUTDIR --N $N --K $K --seed $SEED"
         python simulation_study.py --outdir "$OUTDIR" --N $N --K $K --seed $SEED
    done
done
