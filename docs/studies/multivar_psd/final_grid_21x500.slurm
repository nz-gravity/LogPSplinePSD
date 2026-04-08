#!/bin/bash
# Final sweep for N=16384 over the 21-condition (Nb, Nc) grid.
#
# Strategy:
#   - Global work items are indexed over (seed, condition) pairs.
#   - Each array task processes TASKS_PER_ARRAY_ITEM work items serially.
#   - Completed runs are skipped automatically if compact_run_summary.json exists.
#   - Equivalent runs already present in OUTDIR are also skipped.
#
# Default sweep:
#   seeds: 0..499 (500 total)
#   conditions: 21
#   total work items: 500 * 21 = 10500
#
# Default submission:
#   sbatch docs/studies/multivar_psd/final_grid_21x500.slurm
#
# Notes:
#   - With TASKS_PER_ARRAY_ITEM=10, each array task handles 10 runs serially.
#   - You can safely resubmit later; completed runs should be skipped.

#SBATCH --job-name=var3_fg21
#SBATCH --output=logs_fg21/job_%A_%a.out
#SBATCH --error=logs_fg21/job_%A_%a.err
#SBATCH --array=0-1049%10
#SBATCH --time=6:00:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=4

module --force purge
module load gcc/13.3.0 python/3.12.3

set -euo pipefail

export XLA_FLAGS="--xla_force_host_platform_device_count=4"

PROJECT_ROOT=${PROJECT_ROOT:-/fred/oz303/avajpeyi/codes/LogPSplinePSD}
PYTHON_BIN=${PYTHON_BIN:-${PROJECT_ROOT}/.venv/bin/python}
STUDY_DIR=${STUDY_DIR:-${PROJECT_ROOT}/docs/studies/multivar_psd}

MODE=${MODE:-large}
N_TIME=${N_TIME:-16384}
KNOTS=${KNOTS:-30}
WINDOW=${WINDOW:-rect}
OUTDIR=${OUTDIR:-out_var3_final_fg21}
LABEL_PREFIX=${LABEL_PREFIX:-fg21}

# Inference controls (empty -> script defaults from 3d_study.py)
# Quick mode is intentionally disabled for this final sweep script.
N_SAMPLES=${N_SAMPLES:-}
N_WARMUP=${N_WARMUP:-}
NUM_CHAINS=${NUM_CHAINS:-}
VI_STEPS=${VI_STEPS:-}

# Seed controls
SEED_START=${SEED_START:-0}
SEEDS_TOTAL=${SEEDS_TOTAL:-500}

# Chunk controls
# TASKS_PER_ARRAY_ITEM:
#   how many global work items each array task executes serially
# TASK_ID_OFFSET:
#   global-item offset to support manual partial resubmissions if needed
TASKS_PER_ARRAY_ITEM=${TASKS_PER_ARRAY_ITEM:-10}
TASK_ID_OFFSET=${TASK_ID_OFFSET:-0}

SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
SLURM_JOB_ID=${SLURM_JOB_ID:-manual}

if [[ "${QUICK:-0}" != "0" ]]; then
    echo "QUICK mode is disabled in final_grid_21x500.slurm. Unset QUICK or use a non-final study script."
    exit 1
fi

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
    echo "Could not find python executable: ${PYTHON_BIN}"
    exit 1
fi

cd "${STUDY_DIR}"
mkdir -p logs_fg21

STATUS_DIR="${OUTDIR}/status"
mkdir -p "${STATUS_DIR}"

DONE_LOG="${STATUS_DIR}/done_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}.tsv"
SKIP_LOG="${STATUS_DIR}/skipped_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}.tsv"
FAIL_LOG="${STATUS_DIR}/failed_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}.tsv"

# 21-condition grid (Nb, Nc) for N=16384 with minimum Nc=256.
LABELS=(
    ${LABEL_PREFIX}_nb1_nc8192
    ${LABEL_PREFIX}_nb2_nc4096
    ${LABEL_PREFIX}_nb4_nc2048
    ${LABEL_PREFIX}_nb8_nc1024
    ${LABEL_PREFIX}_nb16_nc512
    ${LABEL_PREFIX}_nb32_nc256
    ${LABEL_PREFIX}_nb1_nc4096
    ${LABEL_PREFIX}_nb1_nc2048
    ${LABEL_PREFIX}_nb1_nc1024
    ${LABEL_PREFIX}_nb1_nc512
    ${LABEL_PREFIX}_nb1_nc256
    ${LABEL_PREFIX}_nb2_nc2048
    ${LABEL_PREFIX}_nb2_nc1024
    ${LABEL_PREFIX}_nb2_nc512
    ${LABEL_PREFIX}_nb2_nc256
    ${LABEL_PREFIX}_nb4_nc1024
    ${LABEL_PREFIX}_nb4_nc512
    ${LABEL_PREFIX}_nb4_nc256
    ${LABEL_PREFIX}_nb8_nc512
    ${LABEL_PREFIX}_nb8_nc256
    ${LABEL_PREFIX}_nb16_nc256
)

NBS=(
    1 2 4 8 16 32
    1 1 1 1 1
    2 2 2 2
    4 4 4
    8 8
    16
)

NCS=(
    8192 4096 2048 1024 512 256
    4096 2048 1024 512 256
    2048 1024 512 256
    1024 512 256
    512 256
    256
)

N_CONDITIONS=${#LABELS[@]}
TOTAL_WORK_ITEMS=$((SEEDS_TOTAL * N_CONDITIONS))
ACTIVE_FIRST=${TASK_ID_OFFSET}
ACTIVE_LAST=$((TASK_ID_OFFSET + TOTAL_WORK_ITEMS - 1))

if (( TASKS_PER_ARRAY_ITEM <= 0 )); then
    echo "TASKS_PER_ARRAY_ITEM must be > 0 (got ${TASKS_PER_ARRAY_ITEM})"
    exit 1
fi
if (( TASK_ID_OFFSET < 0 )); then
    echo "TASK_ID_OFFSET must be >= 0 (got ${TASK_ID_OFFSET})"
    exit 1
fi

GLOBAL_START=$((TASK_ID_OFFSET + SLURM_ARRAY_TASK_ID * TASKS_PER_ARRAY_ITEM))
GLOBAL_END=$((GLOBAL_START + TASKS_PER_ARRAY_ITEM - 1))

if (( GLOBAL_START > ACTIVE_LAST )); then
    echo "No active work for array task ${SLURM_ARRAY_TASK_ID} (start=${GLOBAL_START}, active_last=${ACTIVE_LAST})."
    exit 0
fi

if (( GLOBAL_END > ACTIVE_LAST )); then
    GLOBAL_END=${ACTIVE_LAST}
fi

echo "Array task ${SLURM_ARRAY_TASK_ID}: processing global items ${GLOBAL_START}..${GLOBAL_END} of active range ${ACTIVE_FIRST}..${ACTIVE_LAST}"

already_done_equivalent() {
    local outdir="$1"
    local seed="$2"
    local n_time="$3"
    local nb="$4"
    local coarse_nh="$5"

    "${PYTHON_BIN}" - <<'PY' "${outdir}" "${seed}" "${n_time}" "${nb}" "${coarse_nh}"
import glob
import json
import os
import sys

outdir = sys.argv[1]
seed = int(sys.argv[2])
n_time = int(sys.argv[3])
nb = int(sys.argv[4])
coarse_nh = int(sys.argv[5])  # 0 => OFF
target_nh = "OFF" if coarse_nh == 0 else coarse_nh

pattern = os.path.join(outdir, f"seed_{seed}_*", "compact_run_summary.json")
for path in glob.glob(pattern):
    try:
        with open(path) as f:
            row = json.load(f)
    except Exception:
        continue
    try:
        row_seed = int(row.get("seed", -1))
        row_n = int(row.get("N", -1))
        row_nb = int(row.get("Nb", -1))
    except Exception:
        continue
    row_nh_raw = row.get("Nh", "OFF")
    if isinstance(row_nh_raw, str) and row_nh_raw.upper() == "OFF":
        row_nh = "OFF"
    else:
        try:
            row_nh = int(row_nh_raw)
        except Exception:
            continue
    if (
        row_seed == seed
        and row_n == n_time
        and row_nb == nb
        and row_nh == target_nh
    ):
        print(path)
        sys.exit(0)
sys.exit(1)
PY
}

FAILED_ITEMS=()

for GLOBAL_ID in $(seq "${GLOBAL_START}" "${GLOBAL_END}"); do
    LOCAL_ID=$((GLOBAL_ID - TASK_ID_OFFSET))
    SEED_INDEX=$((LOCAL_ID / N_CONDITIONS))
    COND_INDEX=$((LOCAL_ID % N_CONDITIONS))
    SEED=$((SEED_START + SEED_INDEX))

    LABEL=${LABELS[$COND_INDEX]}
    NB=${NBS[$COND_INDEX]}
    NC=${NCS[$COND_INDEX]}

    if (( N_TIME % NB != 0 )); then
        echo "Invalid geometry: N=${N_TIME} is not divisible by Nb=${NB}"
        echo -e "${GLOBAL_ID}\t${SEED}\t${LABEL}\tbad_Nb" >> "${FAIL_LOG}"
        FAILED_ITEMS+=("global=${GLOBAL_ID},seed=${SEED},label=${LABEL},reason=bad_Nb")
        continue
    fi

    LB=$((N_TIME / NB))
    RAW_NELL=$((LB / 2))

    if (( RAW_NELL % NC != 0 )); then
        echo "Invalid geometry: N_ell=${RAW_NELL} is not divisible by Nc=${NC}"
        echo -e "${GLOBAL_ID}\t${SEED}\t${LABEL}\tbad_Nc" >> "${FAIL_LOG}"
        FAILED_ITEMS+=("global=${GLOBAL_ID},seed=${SEED},label=${LABEL},reason=bad_Nc")
        continue
    fi

    if (( NC == RAW_NELL )); then
        COARSE_NH=0
        NH_DESC=OFF
        NH_LABEL=noNh
    else
        COARSE_NH=$((RAW_NELL / NC))
        NH_DESC=${COARSE_NH}
        NH_LABEL=Nh${COARSE_NH}
    fi

    # 3d_study.py naming convention:
    # outdir/seed_<seed>_<mode>_N<N>_K<K>_<window>_<nh_label><_NbX if Nb!=4>_<label>
    NB_SUFFIX=""
    if (( NB != 4 )); then
        NB_SUFFIX=_Nb${NB}
    fi

    RUN_DIR="${OUTDIR}/seed_${SEED}_${MODE}_N${N_TIME}_K${KNOTS}_${WINDOW}_${NH_LABEL}${NB_SUFFIX}_${LABEL}"
    RUN_SUMMARY="${RUN_DIR}/compact_run_summary.json"

    if [[ -f "${RUN_SUMMARY}" ]]; then
        echo "Skipping completed target run: ${RUN_SUMMARY}"
        echo -e "${GLOBAL_ID}\t${SEED}\t${LABEL}\talready_has_summary" >> "${SKIP_LOG}"
        continue
    fi

    if already_done_equivalent "${OUTDIR}" "${SEED}" "${N_TIME}" "${NB}" "${COARSE_NH}"; then
        echo "Skipping equivalent existing run: seed=${SEED}, Nb=${NB}, Nh=${NH_DESC}"
        echo -e "${GLOBAL_ID}\t${SEED}\t${LABEL}\tequivalent_exists" >> "${SKIP_LOG}"
        continue
    fi

    echo "Running global=${GLOBAL_ID}: seed=${SEED}, label=${LABEL}, N=${N_TIME}, Nb=${NB}, Lb=${LB}, N_ell=${RAW_NELL}, Nc=${NC}, Nh=${NH_DESC}, window=${WINDOW}"

    CMD=(
        "${PYTHON_BIN}" 3d_study.py
        "${SEED}" "${MODE}"
        --K "${KNOTS}"
        --window "${WINDOW}"
        --outdir "${OUTDIR}"
        --n-time "${N_TIME}"
        --nb-override "${NB}"
        --coarse-nh "${COARSE_NH}"
        --label "${LABEL}"
    )

    if [[ -n "${N_SAMPLES}" ]]; then
        CMD+=(--n-samples "${N_SAMPLES}")
    fi
    if [[ -n "${N_WARMUP}" ]]; then
        CMD+=(--n-warmup "${N_WARMUP}")
    fi
    if [[ -n "${NUM_CHAINS}" ]]; then
        CMD+=(--num-chains "${NUM_CHAINS}")
    fi
    if [[ -n "${VI_STEPS}" ]]; then
        CMD+=(--vi-steps "${VI_STEPS}")
    fi

    if "${CMD[@]}"; then
        echo -e "${GLOBAL_ID}\t${SEED}\t${LABEL}\tsuccess" >> "${DONE_LOG}"
    else
        echo -e "${GLOBAL_ID}\t${SEED}\t${LABEL}\trun_failed" >> "${FAIL_LOG}"
        FAILED_ITEMS+=("global=${GLOBAL_ID},seed=${SEED},label=${LABEL}")
    fi
done

if (( ${#FAILED_ITEMS[@]} > 0 )); then
    echo "Array task ${SLURM_ARRAY_TASK_ID} completed with failures:"
    printf '  %s\n' "${FAILED_ITEMS[@]}"
    exit 1
fi

echo "Array task ${SLURM_ARRAY_TASK_ID} finished successfully."
