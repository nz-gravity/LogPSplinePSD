#!/usr/bin/env bash
set -euo pipefail

ROOT="/Users/avi/Documents/projects/LogPSplinePSD"
PY="${ROOT}/.venv/bin/python"
SCRIPT="${ROOT}/docs/studies/lisa/lisa_multivar.py"

if [[ ! -x "${PY}" ]]; then
  echo "Missing venv python at ${PY}" >&2
  exit 1
fi

# Baseline shared settings for quicker iteration.
BASE_ARGS=(
  --no-init-from-vi
  --no-compute-lnz
  --tau-off
  --target-accept 0.85
  --n-warmup 2000
  --n-samples 1000
)

# Sweep: windowing diagnostics at fixed fmin/fmax.
JOBS=(
  "--target-coarse-bins 8192 --n-knots 32 --diff-order 1 --knot-method density --fmin 1e-4 --fmax 1e-1 --wishart-window none --welch-window none"
  "--target-coarse-bins 8192 --n-knots 32 --diff-order 1 --knot-method density --fmin 1e-4 --fmax 1e-1 --wishart-window hann --welch-window hann"
  "--target-coarse-bins 8192 --n-knots 32 --diff-order 1 --knot-method density --fmin 1e-4 --fmax 1e-1 --wishart-window tukey --wishart-tukey-alpha 0.1 --welch-window tukey --welch-tukey-alpha 0.1"
  "--target-coarse-bins 8192 --n-knots 32 --diff-order 1 --knot-method density --fmin 1e-4 --fmax 1e-1 --wishart-window tukey --wishart-tukey-alpha 0.2 --welch-window tukey --welch-tukey-alpha 0.2"
)

for job in "${JOBS[@]}"; do
  echo "Running: ${job}"
  # shellcheck disable=SC2206
  EXTRA_ARGS=( ${job} )
  "${PY}" "${SCRIPT}" "${BASE_ARGS[@]}" "${EXTRA_ARGS[@]}"
done
