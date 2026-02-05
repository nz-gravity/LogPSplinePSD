#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${HERE}/../../.." && pwd)"

PY="${ROOT}/.venv/bin/python"
if [[ ! -x "${PY}" ]]; then
  echo "Missing venv python at ${PY}"
  exit 1
fi

OUT_BASE="${ROOT}/docs/studies/paper_plots/results"
DATA_DIR="${ROOT}/docs/studies/paper_plots/data"

SHORT_N="${SHORT_N:-10000}"
LONG_N="${LONG_N:-60000}"
BLOCK_SIZE="${BLOCK_SIZE:-5000}"
CG_BINS="${CG_BINS:-512}"
LISA_COARSE_Nh="${LISA_COARSE_Nh:-5}"

FMIN="${FMIN:-1e-4}"
FMAX_RESTRICTED="${FMAX_RESTRICTED:-1e-2}"
FMAX_FULL="${FMAX_FULL:-1e-1}"

LISA_MODEL="${LISA_MODEL:-scirdv1}"
LISA_SEED="${LISA_SEED:-123}"
LISA_REGEN="${LISA_REGEN:-0}" # 0|1
# Pick a cadence that supports the requested full band up to 1e-1 Hz:
# Nyquist = 1/(2*dt) >= 1e-1 => dt <= 5 seconds.
LISA_DELTA_T="${LISA_DELTA_T:-5}" # seconds (<= 50s, and satisfies Nyquist for fmax=1e-1)
LISA_SHORT_WEEKS="${LISA_SHORT_WEEKS:-4}"
LISA_LONG_WEEKS="${LISA_LONG_WEEKS:-12}"
LISA_BLOCK_SIZE="${LISA_BLOCK_SIZE:-5000}" # generation chunk length + Wishart block length

if [[ ! "${LISA_DELTA_T}" =~ ^[0-9]+$ ]] || [[ "${LISA_DELTA_T}" -le 0 ]]; then
  echo "LISA_DELTA_T must be a positive integer seconds value, got '${LISA_DELTA_T}'."
  exit 1
fi

if [[ ! "${LISA_BLOCK_SIZE}" =~ ^[0-9]+$ ]] || [[ "${LISA_BLOCK_SIZE}" -le 0 ]]; then
  echo "LISA_BLOCK_SIZE must be a positive integer, got '${LISA_BLOCK_SIZE}'."
  exit 1
fi

WEEK_SECONDS=604800
LISA_SHORT_SECONDS=$(( LISA_SHORT_WEEKS * WEEK_SECONDS ))
LISA_LONG_SECONDS=$(( LISA_LONG_WEEKS * WEEK_SECONDS ))
LISA_SHORT_N_RAW=$(( LISA_SHORT_SECONDS / LISA_DELTA_T ))
LISA_LONG_N_RAW=$(( LISA_LONG_SECONDS / LISA_DELTA_T ))

trim_to_block () {
  local n="$1"
  local b="$2"
  local r=$(( n % b ))
  if [[ "${r}" -eq 0 ]]; then
    echo "${n}"
  else
    echo $(( n - r ))
  fi
}

LISA_SHORT_N="$(trim_to_block "${LISA_SHORT_N_RAW}" "${LISA_BLOCK_SIZE}")"
LISA_LONG_N="$(trim_to_block "${LISA_LONG_N_RAW}" "${LISA_BLOCK_SIZE}")"

PAPER_MODE="${PAPER_MODE:-draft}"       # draft|paper
PAPER_OVERWRITE="${PAPER_OVERWRITE:-0}" # 0|1

DO_OVERWRITE=0
if [[ "${PAPER_OVERWRITE}" == "1" ]]; then
  DO_OVERWRITE=1
fi

mkdir -p "${OUT_BASE}"
mkdir -p "${DATA_DIR}"

SHORT_NPZ="${SHORT_NPZ:-${DATA_DIR}/lisa_synth_short_${LISA_SHORT_WEEKS}w_dt${LISA_DELTA_T}_B${LISA_BLOCK_SIZE}.npz}"
LONG_NPZ="${LONG_NPZ:-${DATA_DIR}/lisa_synth_long_${LISA_LONG_WEEKS}w_dt${LISA_DELTA_T}_B${LISA_BLOCK_SIZE}.npz}"

maybe_regen () {
  local out="$1"
  local n_time="$2"
	  if [[ "${LISA_REGEN}" == "1" ]]; then
	    "${PY}" "${ROOT}/docs/studies/paper_plots/generate_lisa_paper_data.py" \
	      --out "${out}" \
	      --n-time "${n_time}" \
	      --block-size "${LISA_BLOCK_SIZE}" \
	      --delta-t "${LISA_DELTA_T}" \
	      --fmin "${FMIN}" \
	      --fmax "${FMAX_FULL}" \
	      --seed "${LISA_SEED}" \
	      --model "${LISA_MODEL}" \
	      --overwrite
	    return
	  fi
	  if [[ ! -f "${out}" ]]; then
	    "${PY}" "${ROOT}/docs/studies/paper_plots/generate_lisa_paper_data.py" \
	      --out "${out}" \
	      --n-time "${n_time}" \
	      --block-size "${LISA_BLOCK_SIZE}" \
	      --delta-t "${LISA_DELTA_T}" \
	      --fmin "${FMIN}" \
	      --fmax "${FMAX_FULL}" \
	      --seed "${LISA_SEED}" \
	      --model "${LISA_MODEL}"
	  fi
	}

maybe_regen "${SHORT_NPZ}" "${LISA_SHORT_N}"
maybe_regen "${LONG_NPZ}" "${LISA_LONG_N}"

VAR3_ARGS=(--block-size "${BLOCK_SIZE}")

case "${PAPER_MODE}" in
  draft)
    VAR3_ARGS+=(--samples 400 --warmup 400 --chains 2 --vi-steps 5000)
    LISA_ARGS=(--samples 800 --warmup 800 --chains 2 --vi-steps 20000)
    ;;
  paper)
    VAR3_ARGS+=(--samples 1000 --warmup 1000 --chains 4 --vi-steps 20000)
    LISA_ARGS=(--samples 4000 --warmup 4000 --chains 4 --vi-steps 200000)
    ;;
  *)
    echo "Unknown PAPER_MODE='${PAPER_MODE}' (expected draft|paper)"
    exit 1
    ;;
esac

echo "Paper jobs:"
echo "  mode=${PAPER_MODE} overwrite=${PAPER_OVERWRITE}"
echo "  VAR3: short=${SHORT_N} long=${LONG_N} block_size=${BLOCK_SIZE} coarse_bins=${CG_BINS}"
echo "  LISA: short=${LISA_SHORT_WEEKS}w long=${LISA_LONG_WEEKS}w dt=${LISA_DELTA_T}s block_size=${LISA_BLOCK_SIZE} model=${LISA_MODEL} seed=${LISA_SEED}"
echo "        short_npz=${SHORT_NPZ}"
echo "        long_npz=${LONG_NPZ}"
echo "  fmin=${FMIN} restricted_fmax=${FMAX_RESTRICTED} full_fmax=${FMAX_FULL}"
echo "  out=${OUT_BASE}"
echo

run_var3 () {
  local name="$1"
  local n_time="$2"
  local coarse_bins="$3"
  local outdir="${OUT_BASE}/${name}"
  echo "==> VAR3 ${name}"
  local -a cmd
  cmd=(
    "${PY}" "${ROOT}/docs/studies/paper_plots/var3_paper_job.py"
    --outdir "${outdir}"
    --n-time "${n_time}"
    --coarse-bins "${coarse_bins}"
    "${VAR3_ARGS[@]}"
  )
  if [[ "${DO_OVERWRITE}" == "1" ]]; then
    cmd+=(--overwrite)
  fi
  "${cmd[@]}"
}

run_lisa () {
  local name="$1"
  local n_time="$2"
  local coarse_flag="$3"
  local fmax="$4"
  local synth_npz="$5"
  local outdir="${OUT_BASE}/${name}"
  echo "==> LISA ${name}"
  local -a cmd
		cmd=(
		    "${PY}" "${ROOT}/docs/studies/paper_plots/lisa_paper_job.py"
		    --outdir "${outdir}"
		    --synth-npz "${synth_npz}"
		    --n-time "${n_time}"
	    --block-size "${LISA_BLOCK_SIZE}"
		    ${coarse_flag}
		    --fmin "${FMIN}"
		    --fmax "${fmax}"
		    --seed "${LISA_SEED}"
		    "${LISA_ARGS[@]}"
		  )
  if [[ "${DO_OVERWRITE}" == "1" ]]; then
    cmd+=(--overwrite)
  fi
  "${cmd[@]}"
}

# 1) VAR(3) study
run_var3 "var3_short_raw" "${SHORT_N}" 0
run_var3 "var3_long_raw" "${LONG_N}" 0
run_var3 "var3_long_cg${CG_BINS}" "${LONG_N}" "${CG_BINS}"

# 2) LISA restricted band (requested: [1e-4, 1e-2]; will clamp to Nyquist if needed)
run_lisa "lisa_short_restricted_raw" "${LISA_SHORT_N}" "--coarse-n-freqs-per-bin 0" "${FMAX_RESTRICTED}" "${SHORT_NPZ}"
run_lisa "lisa_long_restricted_raw" "${LISA_LONG_N}" "--coarse-n-freqs-per-bin 0" "${FMAX_RESTRICTED}" "${LONG_NPZ}"
run_lisa "lisa_long_restricted_cg${LISA_COARSE_Nh}" "${LISA_LONG_N}" "--coarse-n-freqs-per-bin ${LISA_COARSE_Nh}" "${FMAX_RESTRICTED}" "${LONG_NPZ}"

# 3) LISA full band (requested; will clamp to Nyquist if needed)
run_lisa "lisa_short_full_raw" "${LISA_SHORT_N}" "--coarse-n-freqs-per-bin 0" "${FMAX_FULL}" "${SHORT_NPZ}"
run_lisa "lisa_long_full_raw" "${LISA_LONG_N}" "--coarse-n-freqs-per-bin 0" "${FMAX_FULL}" "${LONG_NPZ}"
run_lisa "lisa_long_full_cg${LISA_COARSE_Nh}" "${LISA_LONG_N}" "--coarse-n-freqs-per-bin ${LISA_COARSE_Nh}" "${FMAX_FULL}" "${LONG_NPZ}"

echo
echo "Done. Outputs are under ${OUT_BASE}/"
