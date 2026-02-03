#!/usr/bin/env bash
set -euo pipefail

ROOT="/Users/avi/Documents/projects/LogPSplinePSD"
PYTHON_BIN="${PYTHON_BIN:-$ROOT/.venv/bin/python}"
SCRIPT="$ROOT/docs/studies/lisa/lisa_multivar.py"

run_case() {
  local tag="$1"
  local mode="$2"
  local scale="$3"
  local constant="$4"
  local blocks="$5"

  echo "=== Running $tag ==="
  LISA_RUN_TAG="$tag" \
  LISA_MAX_MONTHS="${LISA_MAX_MONTHS:-3}" \
  LISA_USE_NOISE_FLOOR=1 \
  LISA_NOISE_FLOOR_MODE="$mode" \
  LISA_NOISE_FLOOR_SCALE="$scale" \
  LISA_NOISE_FLOOR_CONSTANT="$constant" \
  LISA_NOISE_FLOOR_BLOCKS="$blocks" \
  "$PYTHON_BIN" "$SCRIPT"
}

# Theory-scaled floor (block 3 -> index 2)
run_case "theory_1e-4" "theory_scaled" "1e-4" "1e-6" "2"
run_case "theory_1e-2" "theory_scaled" "1e-2" "1e-6" "2"
run_case "theory_1e-1" "theory_scaled" "1e-1" "1e-6" "2"

# Constant floor (standardized units)
run_case "const_1e-6" "constant" "1e-4" "1e-6" "2"
run_case "const_1e-5" "constant" "1e-4" "1e-5" "2"

echo "=== Aggregate summary ==="
"$PYTHON_BIN" "$ROOT/docs/studies/lisa/aggregate_noise_floor_runs.py"
