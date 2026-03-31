#!/bin/bash
# Explore coverage at fixed n=16384 while separating:
#   1) segment size / block count (Nb),
#   2) coarse-graining within a fixed block structure (Nh).
#
# Notation used below:
#   n   = number of time samples
#   Nb  = number of blocks
#   Lb  = samples per block = n / Nb
#   N_ell = retained positive-frequency bins before coarse-graining = floor(Lb / 2)
#   Nh  = number of retained frequencies merged per coarse bin
#   Nc  = number of coarse bins = N_ell / Nh   (or Nc = N_ell when Nh=1 / no coarse)
#
# Default: 20-seed pilot over 7 conditions.
# Full rerun example:
#   sbatch --array=0-4 --export=SEEDS_PER_TASK=20,GRID=full \
#     docs/studies/multivar_psd/segment_vs_coarse_study.slurm
#
# Pilot rerun example:
#   sbatch --array=0-3 --export=SEEDS_PER_TASK=5,GRID=pilot \
#     docs/studies/multivar_psd/segment_vs_coarse_study.slurm
#
# Condition sets:
#   GRID=pilot
#     - seg_nb4_nh1   : Nb=4,  Nh=1  (no coarse)
#     - seg_nb4_nh2   : Nb=4,  Nh=2
#     - seg_nb4_nh4   : Nb=4,  Nh=4
#     - seg_nb8_nh1   : Nb=8,  Nh=1  (no coarse)
#     - seg_nb8_nh2   : Nb=8,  Nh=2
#     - seg_nb16_nh1  : Nb=16, Nh=1  (no coarse)
#     - rankdef_nb2_nh1 : Nb=2, Nh=1 (rank-deficient p=3 stress test)
#
#   GRID=full
#     - seg_nb4_nh1
#     - seg_nb4_nh2
#     - seg_nb8_nh1
#     - seg_nb16_nh1
#     - rankdef_nb2_nh1
#
#SBATCH --job-name=var3_nbnh
#SBATCH --output=logs_nbnh/job_%A_%a.out
#SBATCH --error=logs_nbnh/job_%A_%a.err
#SBATCH --array=0-3
#SBATCH --time=3:00:00
#SBATCH --mem=5G
#SBATCH --cpus-per-task=4

module --force purge
module load gcc/13.3.0 python/3.12.3

set -uo pipefail

export XLA_FLAGS="--xla_force_host_platform_device_count=4"

PYTHON_BIN=/fred/oz303/avajpeyi/codes/LogPSplinePSD/.venv/bin/python
STUDY_DIR=/fred/oz303/avajpeyi/codes/LogPSplinePSD/docs/studies/multivar_psd

cd "${STUDY_DIR}"
mkdir -p logs_nbnh

SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
SEEDS_PER_TASK=${SEEDS_PER_TASK:-5}
SEED_OFFSET=${SEED_OFFSET:-0}
GRID=${GRID:-pilot}
OUTDIR=${OUTDIR:-out_var3}
N_TIME=${N_TIME:-16384}
KNOTS=${KNOTS:-20}
WINDOW=${WINDOW:-rect}

START_SEED=$((SEED_OFFSET + SLURM_ARRAY_TASK_ID * SEEDS_PER_TASK))
END_SEED=$((START_SEED + SEEDS_PER_TASK - 1))

if [[ "${GRID}" == "pilot" ]]; then
    LABELS=(
        seg_nb4_nh1
        seg_nb4_nh2
        seg_nb4_nh4
        seg_nb8_nh1
        seg_nb8_nh2
        seg_nb16_nh1
        rankdef_nb2_nh1
    )
    NBS=(4 4 4 8 8 16 2)
    NHS=(0 2 4 0 2 0 0)
elif [[ "${GRID}" == "full" ]]; then
    LABELS=(
        seg_nb4_nh1
        seg_nb4_nh2
        seg_nb8_nh1
        seg_nb16_nh1
        rankdef_nb2_nh1
    )
    NBS=(4 4 8 16 2)
    NHS=(0 2 0 0 0)
else
    echo "Unknown GRID='${GRID}'. Use GRID=pilot or GRID=full."
    exit 1
fi

FAILED_RUNS=()

for SEED in $(seq "${START_SEED}" "${END_SEED}"); do
    for IDX in "${!LABELS[@]}"; do
        LABEL=${LABELS[$IDX]}
        NB=${NBS[$IDX]}
        NH=${NHS[$IDX]}
        LB=$((N_TIME / NB))
        RAW_NELL=$((LB / 2))

        if (( NH == 0 )); then
            NH_DESC=1
            NC=${RAW_NELL}
            NH_LABEL=noNh
            COARSE_ARG=0
        else
            NH_DESC=${NH}
            NC=$((RAW_NELL / NH))
            NH_LABEL=Nh${NH}
            COARSE_ARG=${NH}
        fi

        NB_SUFFIX=""
        if (( NB != 4 )); then
            NB_SUFFIX=_Nb${NB}
        fi

        RUN_DIR="${OUTDIR}/seed_${SEED}_large_N${N_TIME}_K${KNOTS}_${WINDOW}_${NH_LABEL}${NB_SUFFIX}_${LABEL}"
        if [[ -f "${RUN_DIR}/compact_run_summary.json" ]]; then
            echo "Skipping seed=${SEED}, label=${LABEL}; found ${RUN_DIR}/compact_run_summary.json"
            continue
        fi

        echo "Running seed=${SEED}, label=${LABEL}, n=${N_TIME}, Nb=${NB}, Lb=${LB}, N_ell=${RAW_NELL}, Nh=${NH_DESC}, Nc=${NC}, window=${WINDOW}"
        if ! "${PYTHON_BIN}" 3d_study.py \
            "${SEED}" \
            large \
            --K "${KNOTS}" \
            --window "${WINDOW}" \
            --outdir "${OUTDIR}" \
            --n-time "${N_TIME}" \
            --nb-override "${NB}" \
            --coarse-nh "${COARSE_ARG}" \
            --label "${LABEL}"; then
            FAILED_RUNS+=("seed=${SEED},label=${LABEL}")
        fi
    done
done

if (( ${#FAILED_RUNS[@]} > 0 )); then
    echo "Array task ${SLURM_ARRAY_TASK_ID} completed with failures:"
    printf '  %s\n' "${FAILED_RUNS[@]}"
    exit 1
fi
