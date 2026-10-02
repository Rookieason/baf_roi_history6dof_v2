#!/usr/bin/env bash

set -e

cd "$(dirname "$0")"

DATA_DIR="../../../data/mm23_movement_csv_resampled_30Hz_with_conf_supervision"
RESULTS_DIR="./results"
SAVED_MODEL_DIR="$RESULTS_DIR/saved_model"

mkdir -p "$SAVED_MODEL_DIR"

python train.py \
  --train_glob "$DATA_DIR/train/*.csv" \
  --val_glob "$DATA_DIR/val/*.csv" \
  --seq_len 30 \
  --batch_size 64 \
  --epochs 20 \
  --lr 1e-3 \
  --gamma_rot 0.3 \
  --eta_tail 0.1 \
  --grad_clip 1.0 \
  --weight_decay 1e-4 \
  --patience 5 \
  --save_dir "$SAVED_MODEL_DIR" \
  --json_dir "$RESULTS_DIR"

python inference.py \
  --input_glob "$DATA_DIR/test/*.csv" \
  --checkpoint "$SAVED_MODEL_DIR/explorer_residual_rot_best.pt" \
  --output_csv "$RESULTS_DIR/explorer_residual_rot_inference_test.csv" \
  --seq_len 30 \
  --batch_size 256

echo "Explorer branch training and inference completed."
