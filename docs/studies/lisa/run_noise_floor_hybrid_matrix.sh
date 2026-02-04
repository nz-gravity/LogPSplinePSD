#!/usr/bin/env bash
set -euo pipefail

ROOT="/Users/avi/Documents/projects/LogPSplinePSD"
PYTHON_BIN="${PYTHON_BIN:-$ROOT/.venv/bin/python}"
SCRIPT="$ROOT/docs/studies/lisa/lisa_multivar.py"

# Keep these runs fast + comparable.
MAX_DAYS="${LISA_MAX_DAYS:-90}"
NOISE_BLOCKS="${LISA_NOISE_FLOOR_BLOCKS:-2}" # block index 2 (0-based)
REUSE_EXISTING="${LISA_REUSE_EXISTING:-1}"

CONSTANTS=("1e-6" "3e-6" "1e-5")
SCALES=("1e-3" "1e-2" "1e-1")
TAUS=("1e-12" "1e-10" "1e-8")

run_case() {
  local constant="$1"
  local scale="$2"
  local tau="$3"

  local tag="hybrid_c${constant}_s${scale}_t${tau}"
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
  "$PYTHON_BIN" "$SCRIPT"
}

for c in "${CONSTANTS[@]}"; do
  for s in "${SCALES[@]}"; do
    for t in "${TAUS[@]}"; do
      run_case "$c" "$s" "$t"
    done
  done
done

echo "=== Aggregate summary ==="
PYTHONPATH="$ROOT/src" "$PYTHON_BIN" "$ROOT/docs/studies/lisa/aggregate_noise_floor_runs.py" --name-contains "hybrid_"
