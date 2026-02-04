#!/usr/bin/env bash
set -euo pipefail

ROOT="/Users/avi/Documents/projects/LogPSplinePSD"
PYTHON_BIN="${PYTHON_BIN:-$ROOT/.venv/bin/python}"
SCRIPT="$ROOT/docs/studies/lisa/lisa_multivar.py"

# Keep runs comparable + reasonably fast
MAX_DAYS="${LISA_MAX_DAYS:-90}"
NOISE_BLOCKS="${LISA_NOISE_FLOOR_BLOCKS:-2}" # block index 2 (0-based)
REUSE_EXISTING="${LISA_REUSE_EXISTING:-1}"

run_case() {
  local tag="$1"
  local constant="$2"
  local scale="$3"
  local tau="$4"
  local normw="$5"

  echo "=== Running $tag ==="
  PYTHONPATH="$ROOT/src" \
  LISA_RUN_TAG="$tag" \
  LISA_MAX_DAYS="$MAX_DAYS" \
  LISA_REUSE_EXISTING="$REUSE_EXISTING" \
  LISA_USE_NOISE_FLOOR=1 \
  LISA_NOISE_FLOOR_MODE="hybrid" \
  LISA_NOISE_FLOOR_CONSTANT="$constant" \
  LISA_NOISE_FLOOR_SCALE="$scale" \
  LISA_NOISE_FLOOR_TAU="$tau" \
  LISA_NOISE_FLOOR_BLOCKS="$NOISE_BLOCKS" \
  LISA_NORMALIZE_FREQ_WEIGHTS="$normw" \
  "$PYTHON_BIN" "$SCRIPT"
}

# Reference (stable): c=1e-5, s=1e-1, tau=1e-10
run_case "targeted_ref_c1e-5_s1e-1_t1e-10_normw0" "1e-5" "1e-1" "1e-10" "0"
run_case "targeted_ref_c1e-5_s1e-1_t1e-10_normw1" "1e-5" "1e-1" "1e-10" "1"

# Promising targeted (geometry win observed): c=3e-6, s=1e-2, tau=1e-8
run_case "targeted_prom_c3e-6_s1e-2_t1e-8_normw0" "3e-6" "1e-2" "1e-8" "0"
run_case "targeted_prom_c3e-6_s1e-2_t1e-8_normw1" "3e-6" "1e-2" "1e-8" "1"

# Slightly more theory-active variant to probe "stiffness only where needed":
# c=3e-6, s=1e-1, tau=1e-10 (tends to make scale*theory exceed constant more often).
run_case "targeted_theoryactive_c3e-6_s1e-1_t1e-10_normw0" "3e-6" "1e-1" "1e-10" "0"
run_case "targeted_theoryactive_c3e-6_s1e-1_t1e-10_normw1" "3e-6" "1e-1" "1e-10" "1"

echo "=== Aggregate summary (targeted runs) ==="
PYTHONPATH="$ROOT/src" "$PYTHON_BIN" "$ROOT/docs/studies/lisa/aggregate_noise_floor_runs.py" --name-contains "targeted_"
