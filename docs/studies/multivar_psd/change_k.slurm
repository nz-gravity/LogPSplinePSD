#!/bin/bash
#SBATCH --job-name=sim_study
#SBATCH --output=sim_study_%A_%a.out
#SBATCH --error=sim_study_%A_%a.err
#SBATCH --array=1-10
#SBATCH --time=02:00:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=1

module load python/3.9

# Default SLURM_ARRAY_TASK_ID to 0 for local testing if not set
SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID:-0}

# Set K values
K_VALUES=(5 10 15 20)
N=512
NREP=10
OUTDIR="out_changing_k"

# Each array task runs NREP replicates
if [ "$SLURM_ARRAY_TASK_ID" -eq 0 ]; then
    START_REP=1
    END_REP=$NREP
else
    START_REP=$(( ($SLURM_ARRAY_TASK_ID - 1) * $NREP + 1 ))
    END_REP=$(( $SLURM_ARRAY_TASK_ID * $NREP ))
fi

for K in "${K_VALUES[@]}"; do
    for ((i=0; i<$NREP; i++)); do
        REPLICATE=$((START_REP + i))
        SEED=$REPLICATE
        echo "python simulation_study.py --outdir $OUTDIR --N $N --K $K --seed $SEED"
        python simulation_study.py --outdir "$OUTDIR" --N $N --K $K --seed $SEED
    done
done
