#!/bin/bash
# SLURM array job: one task per LISA observation duration.
# Each task runs the c-space eta sweep for its assigned duration.
#
# Array index → duration mapping (0-indexed):
#   0 →   7d  (Nb= 1,  3 unique etas)
#   1 →  30d  (Nb= 4,  5 unique etas)
#   2 →  91d  (Nb=13,  7 unique etas)
#   3 → 182d  (Nb=26,  7 unique etas)
#   4 → 365d  (Nb=52,  7 unique etas)
#
# Submit:  sbatch run_duration_eta_study.slurm
# Monitor: squeue -u $USER

#SBATCH --job-name=lisa_dur_eta
#SBATCH --output=logs/dur_eta_%A_%a.out
#SBATCH --error=logs/dur_eta_%A_%a.err
#SBATCH --array=0-4
#SBATCH --time=2:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=4
#SBATCH --account=oz303

module --force purge
module load gcc/13.3.0 python/3.12.3

PYTHON_BIN=/fred/oz303/avajpeyi/codes/LogPSplinePSD/.venv/bin/python
SCRIPT=/fred/oz303/avajpeyi/codes/LogPSplinePSD/docs/studies/lisa/eta_sweep.py
OUTDIR=/fred/oz303/avajpeyi/codes/LogPSplinePSD/docs/studies/lisa/runs

export XLA_FLAGS="--xla_force_host_platform_device_count=4"

BLOCK_DAYS=7
K=100
K_THETA_IM=4
SEEDS="0"
# c-space grid: eta = min(1, c/Nb)
ETAS_C=(0.25 0.5 1.0 2.0 4.0 8.0 16.0)

# Duration and label-start-index per array task.
# Label index uses base-26 encoding; 53=BA (after AZ=52).
DURATIONS=(7 30 91 182 365)
LABEL_STARTS=(53 56 61 68 75)  # BA, BD, BI, BP, BW

DURATION="${DURATIONS[$SLURM_ARRAY_TASK_ID]}"
LABEL_IDX="${LABEL_STARTS[$SLURM_ARRAY_TASK_ID]}"
NB=$(( DURATION / BLOCK_DAYS ))
[[ "${NB}" -lt 1 ]] && NB=1

echo "Task ${SLURM_ARRAY_TASK_ID}: duration=${DURATION}d Nb=${NB}"

# Build unique eta and label lists from the c-space grid.
ETA_LIST=()
LABEL_LIST=()

_label() {
  "${PYTHON_BIN}" -c "
idx = $1
chars = []
while idx > 0:
    idx -= 1
    chars.append(chr(ord('A') + idx % 26))
    idx //= 26
print(''.join(reversed(chars)))
"
}

declare -A SEEN=()
for C in "${ETAS_C[@]}"; do
  ETA=$("${PYTHON_BIN}" -c "print(f'{min(1.0, ${C} / ${NB}):.6f}')")
  KEY=$("${PYTHON_BIN}" -c "print(f'{float(\"${ETA}\"):.5g}')")
  if [[ -z "${SEEN[$KEY]+x}" ]]; then
    SEEN[$KEY]=1
    ETA_LIST+=( "${ETA}" )
    LABEL_LIST+=( "$(_label ${LABEL_IDX})" )
    LABEL_IDX=$(( LABEL_IDX + 1 ))
  fi
done

echo "  etas:   ${ETA_LIST[*]}"
echo "  labels: ${LABEL_LIST[*]}"

mkdir -p "${OUTDIR}" logs

"${PYTHON_BIN}" "${SCRIPT}" \
  --outdir "${OUTDIR}" \
  --seeds ${SEEDS} \
  --etas "${ETA_LIST[@]}" \
  --labels "${LABEL_LIST[@]}" \
  --duration-days "${DURATION}" \
  --block-days "${BLOCK_DAYS}" \
  --K "${K}" \
  --K-theta-im "${K_THETA_IM}" \
  --keep-nc

echo "Task ${SLURM_ARRAY_TASK_ID} complete."
