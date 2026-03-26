#!/usr/bin/env bash
# LISA study run script.
#
# Each run writes to runs/run_<label>/. If the sentinel file
# (compact_run_summary.json for seed 0) already exists the run is skipped.
#
# Usage:
#   bash run_study.sh              # all runs, all seeds
#   bash run_study.sh run_h        # single run by label
#   SEEDS="0 1" bash run_study.sh  # override seed list
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PY="${ROOT}/.venv/bin/python"
SCRIPT="${ROOT}/docs/studies/lisa/main.py"
RUNS_DIR="${ROOT}/docs/studies/lisa/runs"

SEEDS="${SEEDS:-0 1 2 3 4}"

if [[ ! -x "${PY}" ]]; then
  echo "ERROR: venv python not found at ${PY}" >&2
  exit 1
fi

# ---------------------------------------------------------------------------
# Helper: run one seed, skip if sentinel already exists.
# Usage: run_seed <outdir> <seed> [extra args...]
# ---------------------------------------------------------------------------
run_seed() {
  local outdir="$1"
  local seed="$2"
  shift 2
  local sentinel="${outdir}/seed_${seed}/compact_run_summary.json"

  # The slug sub-directory is created by main.py; we check one level deeper.
  # Use a glob to handle the slug name without hard-coding it.
  if compgen -G "${outdir}/*/seed_${seed}/compact_run_summary.json" > /dev/null 2>&1; then
    echo "[SKIP] ${outdir} seed=${seed} (result exists)"
    return 0
  fi

  echo ""
  echo "========================================================"
  echo " RUNNING: $(basename "${outdir}")  seed=${seed}"
  echo "========================================================"
  "${PY}" "${SCRIPT}" "${seed}" --outdir "${outdir}" "$@"
}

# ---------------------------------------------------------------------------
# run_h — floor regularisation + null excision
#   Goal: eliminate transfer-null oscillations at 0.030/0.060/0.090 Hz.
#   Compare riae and coverage against run E (results_datagen_match).
# ---------------------------------------------------------------------------
run_h() {
  local outdir="${RUNS_DIR}/run_h_floor_excise"
  local common=(
    --duration-days 365 --block-days 7
    --wishart-window tukey --wishart-tukey-alpha 0.1
    --wishart-detrend constant
    --wishart-floor-fraction 1e-6
    --exclude-transfer-nulls --exclude-bins-per-side 3
    --coarse-Nc 8192
    --K 20 --knot-method density --diff-order 2
    --vi
  )
  for seed in ${SEEDS}; do
    run_seed "${outdir}" "${seed}" "${common[@]}"
  done
}

# ---------------------------------------------------------------------------
# run_i — more knots (K=32) to investigate low coverage
#   Goal: check if K=20 is under-flexible and causing overconfident posteriors.
#   Builds on run_h config (floor + excision included).
# ---------------------------------------------------------------------------
run_i() {
  local outdir="${RUNS_DIR}/run_i_k32"
  local common=(
    --duration-days 365 --block-days 7
    --wishart-window tukey --wishart-tukey-alpha 0.1
    --wishart-detrend constant
    --wishart-floor-fraction 1e-6
    --exclude-transfer-nulls --exclude-bins-per-side 3
    --coarse-Nc 8192
    --K 32 --knot-method density --diff-order 2
    --vi
  )
  for seed in ${SEEDS}; do
    run_seed "${outdir}" "${seed}" "${common[@]}"
  done
}

# ---------------------------------------------------------------------------
# run_j — looser smoothing prior (alpha=1, beta=1) to investigate low coverage
#   Goal: check if the default prior (alpha=3, beta=3) over-shrinks the
#   posterior and causes the ~0.30 coverage (expected 0.90).
# ---------------------------------------------------------------------------
run_j() {
  local outdir="${RUNS_DIR}/run_j_loose_prior"
  local common=(
    --duration-days 365 --block-days 7
    --wishart-window tukey --wishart-tukey-alpha 0.1
    --wishart-detrend constant
    --wishart-floor-fraction 1e-6
    --exclude-transfer-nulls --exclude-bins-per-side 3
    --coarse-Nc 8192
    --K 20 --knot-method density --diff-order 2
    --alpha-delta 1.0 --beta-delta 1.0
    --vi
  )
  for seed in ${SEEDS}; do
    run_seed "${outdir}" "${seed}" "${common[@]}"
  done
}

# ---------------------------------------------------------------------------
# run_l — null-region coverage isolation (fmax=0.025 Hz, avoids all nulls)
#   Hypothesis: the ~30% coverage is almost entirely due to near-null bins
#   where the true PSD → 0 but the spline is forced to interpolate a nonzero
#   value across the excised gap.  Cutting fmax below the first null at 0.030
#   Hz removes this contamination entirely.  If coverage jumps to ~0.90 here
#   we confirm the nulls are the root cause and excision strategy is the fix.
#   Uses K=32 (best RIAE so far), no VI (VI is broken: var_ratio=0 everywhere).
# ---------------------------------------------------------------------------
run_l() {
  local outdir="${RUNS_DIR}/run_l_nonull_band"
  local common=(
    --duration-days 365 --block-days 7
    --wishart-window tukey --wishart-tukey-alpha 0.1
    --wishart-detrend constant
    --wishart-floor-fraction 1e-6
    --fmin 1e-4 --fmax 0.025
    --coarse-Nc 8192
    --K 32 --knot-method density --diff-order 2
    --no-vi
  )
  for seed in ${SEEDS}; do
    run_seed "${outdir}" "${seed}" "${common[@]}"
  done
}

# ---------------------------------------------------------------------------
# run_m — wider null excision (bins_per_side=15), no VI, K=32
#   If run_l confirms the null region is the problem, we try wider excision
#   so that the spline never sees the near-zero bins near each null.
#   bins_per_side=15 at coarse resolution (~1.2e-5 Hz/bin) → ±1.8e-4 Hz band
#   around each null, covering the most affected region.
# ---------------------------------------------------------------------------
run_m() {
  local outdir="${RUNS_DIR}/run_m_wide_excise"
  local common=(
    --duration-days 365 --block-days 7
    --wishart-window tukey --wishart-tukey-alpha 0.1
    --wishart-detrend constant
    --wishart-floor-fraction 1e-6
    --exclude-transfer-nulls --exclude-bins-per-side 15
    --coarse-Nc 8192
    --K 32 --knot-method density --diff-order 2
    --no-vi
  )
  for seed in ${SEEDS}; do
    run_seed "${outdir}" "${seed}" "${common[@]}"
  done
}

# ---------------------------------------------------------------------------
# run_n — K=48, no VI, wide excision
#   K=20→RIAE 0.27, K=32→RIAE 0.12.  Continuing the trend to K=48 to check
#   if RIAE keeps halving and whether coverage also starts recovering.
# ---------------------------------------------------------------------------
run_n() {
  local outdir="${RUNS_DIR}/run_n_k48"
  local common=(
    --duration-days 365 --block-days 7
    --wishart-window tukey --wishart-tukey-alpha 0.1
    --wishart-detrend constant
    --wishart-floor-fraction 1e-6
    --exclude-transfer-nulls --exclude-bins-per-side 15
    --coarse-Nc 8192
    --K 48 --knot-method density --diff-order 2
    --no-vi
  )
  for seed in ${SEEDS}; do
    run_seed "${outdir}" "${seed}" "${common[@]}"
  done
}

# ---------------------------------------------------------------------------
# run_p — diff_order=1 to fix low-frequency bias
#   The old "good" run used d1 (first-difference penalty), which only
#   penalises roughness and freely tracks a power-law slope in log-log space.
#   d2 (current default) penalises curvature and fights the concave low-freq
#   rise of the LISA PSD, causing the upward bias below ~1e-3 Hz.
#   Replicates the old run's config as closely as possible in the new pipeline:
#   K=32, d1, density knots, tukey window, wide excision, no VI.
# ---------------------------------------------------------------------------
run_p() {
  local outdir="${RUNS_DIR}/run_p_d1"
  local common=(
    --duration-days 365 --block-days 7
    --wishart-window tukey --wishart-tukey-alpha 0.1
    --wishart-detrend constant
    --wishart-floor-fraction 1e-6
    --exclude-transfer-nulls --exclude-bins-per-side 15
    --coarse-Nc 8192
    --K 32 --knot-method density --diff-order 1
    --no-vi
  )
  for seed in ${SEEDS}; do
    run_seed "${outdir}" "${seed}" "${common[@]}"
  done
}

# ---------------------------------------------------------------------------
# run_o — multi-seed validation of best config found in L/M/N
#   Placeholder: update common args once the winning config is known.
# ---------------------------------------------------------------------------
run_o() {
  local outdir="${RUNS_DIR}/run_o_validation"
  echo "[INFO] run_o is a placeholder — update common args before running."
  local common=(
    --duration-days 365 --block-days 7
    --wishart-window tukey --wishart-tukey-alpha 0.1
    --wishart-detrend constant
    --wishart-floor-fraction 1e-6
    --exclude-transfer-nulls --exclude-bins-per-side 15
    --coarse-Nc 8192
    --K 48 --knot-method density --diff-order 2
    --no-vi
  )
  for seed in ${SEEDS}; do
    run_seed "${outdir}" "${seed}" "${common[@]}"
  done
}

# ---------------------------------------------------------------------------
# run_q — log knots to fix low-frequency knot starvation
#   "density" placement clusters knots at the PSD peak (~0.03 Hz), leaving
#   ~1 knot for the entire 1e-4–1e-3 Hz decade.  "log" gives ~10 knots/decade
#   uniformly, properly resolving the steep low-freq slope.
#   Uses d1 (best from run_p) + wide excision + K=32.
# ---------------------------------------------------------------------------
run_q() {
  local outdir="${RUNS_DIR}/run_q_log_knots"
  local common=(
    --duration-days 365 --block-days 7
    --wishart-window tukey --wishart-tukey-alpha 0.1
    --wishart-detrend constant
    --wishart-floor-fraction 1e-6
    --exclude-transfer-nulls --exclude-bins-per-side 15
    --coarse-Nc 8192
    --K 32 --knot-method log --diff-order 1
    --no-vi
  )
  for seed in ${SEEDS}; do
    run_seed "${outdir}" "${seed}" "${common[@]}"
  done
}

# ---------------------------------------------------------------------------
# run_r — density knots + d1 + K=48 (more knots to compensate starvation)
#   If density is preferred for the high-freq region, adding more knots
#   ensures at least a few end up in the low-freq decade.
# ---------------------------------------------------------------------------
run_r() {
  local outdir="${RUNS_DIR}/run_r_density_k48_d1"
  local common=(
    --duration-days 365 --block-days 7
    --wishart-window tukey --wishart-tukey-alpha 0.1
    --wishart-detrend constant
    --wishart-floor-fraction 1e-6
    --exclude-transfer-nulls --exclude-bins-per-side 15
    --coarse-Nc 8192
    --K 48 --knot-method density --diff-order 1
    --no-vi
  )
  for seed in ${SEEDS}; do
    run_seed "${outdir}" "${seed}" "${common[@]}"
  done
}

# ---------------------------------------------------------------------------
# run_s — run_r config re-run after per-frequency Wishart floor fix
#   The global-median floor (used in runs H–R) clipped eigenvalues at
#   low-power frequencies to a threshold derived from the high-freq peak,
#   destroying the off-diagonal structure and collapsing coherence to ~0
#   below 1e-3 Hz.  The fix (per-frequency floor) scales the threshold by
#   each bin's own trace, preserving coherence while still regularising
#   the transfer nulls.  Same config as run_r otherwise.
# ---------------------------------------------------------------------------
run_s() {
  local outdir="${RUNS_DIR}/run_s_perfreq_floor"
  local common=(
    --duration-days 365 --block-days 7
    --wishart-window tukey --wishart-tukey-alpha 0.1
    --wishart-detrend constant
    --wishart-floor-fraction 1e-6
    --exclude-transfer-nulls --exclude-bins-per-side 15
    --coarse-Nc 8192
    --K 48 --knot-method density --diff-order 1
    --no-vi
  )
  for seed in ${SEEDS}; do
    run_seed "${outdir}" "${seed}" "${common[@]}"
  done
}

# run_t: K=48, d2, density, perfreq floor, wider excision (bins_per_side=20)
# d2 penalty penalises curvature → smoother posterior, gentler null bridging.
run_t() {
  local outdir="${RUNS_DIR}/run_t_d2_k48_wide_excision"
  local common=(
    --duration-days 365 --block-days 7
    --wishart-window tukey --wishart-tukey-alpha 0.1
    --wishart-detrend constant
    --wishart-floor-fraction 1e-6
    --exclude-transfer-nulls --exclude-bins-per-side 20
    --coarse-Nc 8192
    --K 48 --knot-method density --diff-order 2
    --no-vi
  )
  for seed in ${SEEDS}; do
    run_seed "${outdir}" "${seed}" "${common[@]}"
  done
}

# run_u: Same as run_t but with much wider null exclusion (±3 mHz in Hz).
# bins_per_side=20 was only ±0.23 mHz — far too narrow.
run_u() {
  local outdir="${RUNS_DIR}/run_u_d2_k48_hw3mHz"
  local common=(
    --duration-days 365 --block-days 7
    --wishart-window tukey --wishart-tukey-alpha 0.1
    --wishart-detrend constant
    --wishart-floor-fraction 1e-6
    --exclude-transfer-nulls --exclude-half-width 0.003
    --coarse-Nc 8192
    --K 48 --knot-method density --diff-order 2
    --no-vi
  )
  for seed in ${SEEDS}; do
    run_seed "${outdir}" "${seed}" "${common[@]}"
  done
}

# run_v: K=48, d2, density, perfreq floor, NO excision.
# Let the smooth d2 spline handle nulls naturally — floor prevents Cholesky
# failures, spline can't track 10-OOM dips so it'll show a modest dip.
run_v() {
  local outdir="${RUNS_DIR}/run_v_d2_k48_no_excision"
  local common=(
    --duration-days 365 --block-days 7
    --wishart-window tukey --wishart-tukey-alpha 0.1
    --wishart-detrend constant
    --wishart-floor-fraction 1e-6
    --coarse-Nc 8192
    --K 48 --knot-method density --diff-order 2
    --no-vi
  )
  for seed in ${SEEDS}; do
    run_seed "${outdir}" "${seed}" "${common[@]}"
  done
}

# run_w: K=32, d2, density, perfreq floor, no excision.
# Fewer knots → less flexibility at nulls → can't chase noisy coherence.
run_w() {
  local outdir="${RUNS_DIR}/run_w_d2_k32_no_excision"
  local common=(
    --duration-days 365 --block-days 7
    --wishart-window tukey --wishart-tukey-alpha 0.1
    --wishart-detrend constant
    --wishart-floor-fraction 1e-6
    --coarse-Nc 8192
    --K 32 --knot-method density --diff-order 2
    --no-vi
  )
  for seed in ${SEEDS}; do
    run_seed "${outdir}" "${seed}" "${common[@]}"
  done
}

# run_x: K=48, d2, uniform knots, perfreq floor, no excision.
# Uniform knots spread evenly instead of clustering near the spectral peak
# (which is adjacent to the 0.03 Hz null), giving fewer DOFs near nulls.
run_x() {
  local outdir="${RUNS_DIR}/run_x_d2_k48_uniform_no_excision"
  local common=(
    --duration-days 365 --block-days 7
    --wishart-window tukey --wishart-tukey-alpha 0.1
    --wishart-detrend constant
    --wishart-floor-fraction 1e-6
    --coarse-Nc 8192
    --K 48 --knot-method uniform --diff-order 2
    --no-vi
  )
  for seed in ${SEEDS}; do
    run_seed "${outdir}" "${seed}" "${common[@]}"
  done
}

# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------
ALL_RUNS=(run_h run_i run_j run_l run_m run_n run_p run_q run_r run_s run_o run_t run_u run_v run_w run_x)

if [[ $# -eq 0 ]]; then
  for run in "${ALL_RUNS[@]}"; do
    "${run}"
  done
else
  for label in "$@"; do
    if declare -f "${label}" > /dev/null; then
      "${label}"
    else
      echo "ERROR: unknown run label '${label}'. Valid: ${ALL_RUNS[*]}" >&2
      exit 1
    fi
  done
fi

echo ""
echo "All requested runs complete."
