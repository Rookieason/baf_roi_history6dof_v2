#!/usr/bin/env bash

set -euo pipefail

cd "$(dirname "$0")"

DATA_DIR="../../data/mm23_movement_csv_resampled_30Hz_with_conf_supervision"
RESULTS_DIR="./results"
MODEL_DIR="${RESULTS_DIR}/saved_model"
FIGURE_DIR="${RESULTS_DIR}/figures"

mkdir -p "$MODEL_DIR" "$FIGURE_DIR"

python train.py \
  --train_glob "${DATA_DIR}/train/*.csv" \
  --val_glob "${DATA_DIR}/val/*.csv" \
  --test_glob "${DATA_DIR}/test/*.csv" \
  --seq_len 30 \
  --batch_size 64 \
  --epochs 10 \
  --lr 1e-3 \
  --save_dir "$MODEL_DIR" \
  --json_dir "$RESULTS_DIR" \
  --focused_weight 3.0 \
  --explorer_weight 1.0 \
  --thresholds "0.3,0.4,0.5,0.6,0.7" \
  --threshold_criterion macro_f1

python inference.py \
  --input_glob "${DATA_DIR}/test/*.csv" \
  --checkpoint "${MODEL_DIR}/confidence_best.pt" \
  --output_csv "${RESULTS_DIR}/confidence_inference_test.csv" \
  --seq_len 30 \
  --batch_size 256 \
  --threshold 0.5

python plot.py \
  --json_path "${RESULTS_DIR}/confidence_training_results.json" \
  --output_dir "$FIGURE_DIR" \
  --split val \
  --selected_threshold 0.5

echo "Confidence training, inference, and plotting completed."
