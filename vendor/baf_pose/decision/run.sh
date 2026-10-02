#!/usr/bin/env bash

set -euo pipefail

cd "$(dirname "$0")"

DATA_DIR="../../data/mm23_movement_csv_resampled_30Hz_with_conf_supervision"
CONFIDENCE_CHECKPOINT="../confidence/results/saved_model/confidence_best.pt"
FOCUSED_CHECKPOINT="../prediction/focused_branch/results/saved_model/focused_pretrain_finetune_best.pt"
EXPLORER_CHECKPOINT="../prediction/explorer_branch/results/saved_model/explorer_residual_rot_best.pt"

RESULTS_DIR="./results"
BAF_POSE_DIR="$RESULTS_DIR/baf_pose"
WITHOUT_BEHAVIOR_FUSION_DIR="$RESULTS_DIR/without_behavior_fusion"

mkdir -p "$BAF_POSE_DIR" "$WITHOUT_BEHAVIOR_FUSION_DIR"

require_file() {
  local file_path="$1"
  if [[ ! -f "$file_path" ]]; then
    echo "Required file not found: $file_path" >&2
    exit 1
  fi
}

require_file "$CONFIDENCE_CHECKPOINT"
require_file "$FOCUSED_CHECKPOINT"
require_file "$EXPLORER_CHECKPOINT"

if ! compgen -G "$DATA_DIR/test/*.csv" > /dev/null; then
  echo "No test CSV files found in: $DATA_DIR/test" >&2
  exit 1
fi

COMMON_ARGS=(
  --input_glob "$DATA_DIR/test/*.csv"
  --confidence_checkpoint "$CONFIDENCE_CHECKPOINT"
  --focused_checkpoint "$FOCUSED_CHECKPOINT"
  --explorer_checkpoint "$EXPLORER_CHECKPOINT"
  --seq_len 30
  --batch_size 256
)

echo "Running BAF-Pose inference."
python decision_fusion.py \
  "${COMMON_ARGS[@]}" \
  --method_name "baf_pose" \
  --output_csv "$BAF_POSE_DIR/predictions.csv" \
  --summary_csv "$BAF_POSE_DIR/summary_by_behavior.csv" \
  --alpha_mode temporal_optimized \
  --temporal_mu 0.6 \
  --temporal_init_mode alpha \
  --fusion_mode full_slerp \
  2>&1 | tee "$BAF_POSE_DIR/run.log"

echo "Running inference without behavior fusion."
python decision_fusion.py \
  "${COMMON_ARGS[@]}" \
  --method_name "without_behavior_fusion" \
  --output_csv "$WITHOUT_BEHAVIOR_FUSION_DIR/predictions.csv" \
  --summary_csv "$WITHOUT_BEHAVIOR_FUSION_DIR/summary_by_behavior.csv" \
  --alpha_mode fixed \
  --fixed_alpha 0.5 \
  --fusion_mode fixed_average \
  2>&1 | tee "$WITHOUT_BEHAVIOR_FUSION_DIR/run.log"

echo "Decision inference completed."
