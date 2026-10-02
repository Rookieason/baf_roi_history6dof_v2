#!/usr/bin/env bash
set -euo pipefail

DATASET_ROOT="${DATASET_ROOT:-}"
if [[ $# -gt 0 && "$1" != -* ]]; then
  DATASET_ROOT="$1"
  shift
fi

ENV_NAME="${ENV_NAME:-ROI_baseline}"
CONFIG="${CONFIG:-experiments/roi_heatmap_ablation/configs/default.json}"
LOG_DIR="${LOG_DIR:-outputs/roi_heatmap_ablation/logs}"
REQUIREMENTS="${REQUIREMENTS:-requirements.txt}"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda not found; please install conda or run from a shell where conda is available" >&2
  exit 1
fi

CONDA_BASE="$(conda info --base)"
source "$CONDA_BASE/etc/profile.d/conda.sh"

if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  conda create -y -n "$ENV_NAME" python=3.10
fi

conda activate "$ENV_NAME"
python -m pip install -r "$REQUIREMENTS"

mkdir -p "$LOG_DIR"

TS="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$LOG_DIR/run_${TS}.log"
PID_FILE="$LOG_DIR/run_${TS}.pid"

CMD=(python -m experiments.roi_heatmap_ablation.run_experiment --config "$CONFIG")
if [[ -n "$DATASET_ROOT" ]]; then
  CMD+=(--dataset-root "$DATASET_ROOT")
fi
CMD+=("$@")

nohup "${CMD[@]}" > "$LOG_FILE" 2>&1 &

PID="$!"
echo "$PID" > "$PID_FILE"

echo "started pid=$PID"
echo "log=$LOG_FILE"
echo "pid_file=$PID_FILE"
echo "tail -f $LOG_FILE"
