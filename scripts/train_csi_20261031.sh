#!/usr/bin/env bash
# All four methods: bash scripts/train_csi_20261031.sh
# One method: bash scripts/train_csi_20261031.sh --method baf_roi_heatmap_alpha
# Shared data -> selected models -> one matched evaluation protocol.
set -euo pipefail

# All machine-specific settings live here; no exported variables are required.
PYTHON=/home/tonic/anaconda3/envs/bafroi/bin/python
RUN_ROOT=/home/Projects/NSTC/EXTRA-DISC/Eason/tonic_nstc_csi_20261031
DEVICE=auto
THREADS=2
CONFIG=configs/recorded_20261031.json

cd "$(dirname "${BASH_SOURCE[0]}")/.."
if [[ ! -x "$PYTHON" ]]; then
  printf 'Python executable not found: %s\n' "$PYTHON" >&2
  exit 1
fi
mkdir -p "$RUN_ROOT/logs" "$RUN_ROOT/tmp"
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS="$THREADS" MKL_NUM_THREADS="$THREADS" OPENBLAS_NUM_THREADS="$THREADS"
export XDG_CACHE_HOME="$RUN_ROOT/.cache" MPLCONFIGDIR="$RUN_ROOT/.matplotlib"
export TMPDIR="$RUN_ROOT/tmp"
RUN_DIR="$RUN_ROOT/run_$(date -u +%Y%m%dT%H%M%SZ)"
mkdir "$RUN_DIR"
# Keep the log outside RUN_DIR: prepare requires an empty experiment directory.
LOG_FILE="$RUN_ROOT/logs/$(basename "$RUN_DIR").log"
ln -sfn "$LOG_FILE" "$RUN_ROOT/latest_training.log"
nohup "$PYTHON" -u scripts/run_experiments.py \
  --config "$CONFIG" --output "$RUN_DIR" --device "$DEVICE" --threads "$THREADS" \
  "$@" \
  > "$LOG_FILE" 2>&1 < /dev/null &
TRAINING_PID=$!
printf '%s\n' "$TRAINING_PID" > "$RUN_ROOT/latest_training.pid"
printf 'Started training: PID %s\nOutput: %s\nLog: %s\n' "$TRAINING_PID" "$RUN_DIR" "$LOG_FILE"
