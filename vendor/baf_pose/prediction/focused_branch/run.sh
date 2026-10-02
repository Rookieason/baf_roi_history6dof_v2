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
  --epochs 8 \
  --lr 1e-3 \
  --lambda_smooth 0.1 \
  --lambda_tr 1.0 \
  --lambda_rot 0.3 \
  --grad_clip 1.0 \
  --weight_decay 1e-4 \
  --patience 4 \
  --sample_mode all \
  --save_dir "$SAVED_MODEL_DIR" \
  --json_dir "$RESULTS_DIR" \
  --checkpoint_name "focused_pretrain_all_best.pt" \
  --result_name "focused_pretrain_all_training_results.json"

python train.py \
  --train_glob "$DATA_DIR/train/*.csv" \
  --val_glob "$DATA_DIR/val/*.csv" \
  --seq_len 30 \
  --batch_size 64 \
  --epochs 12 \
  --lr 1e-4 \
  --lambda_smooth 0.1 \
  --lambda_tr 1.0 \
  --lambda_rot 0.3 \
  --grad_clip 1.0 \
  --weight_decay 1e-4 \
  --patience 5 \
  --sample_mode focused_clean \
  --init_checkpoint "$SAVED_MODEL_DIR/focused_pretrain_all_best.pt" \
  --save_dir "$SAVED_MODEL_DIR" \
  --json_dir "$RESULTS_DIR" \
  --checkpoint_name "focused_pretrain_finetune_best.pt" \
  --result_name "focused_pretrain_finetune_training_results.json"

python inference.py \
  --input_glob "$DATA_DIR/test/*.csv" \
  --checkpoint "$SAVED_MODEL_DIR/focused_pretrain_finetune_best.pt" \
  --output_csv "$RESULTS_DIR/focused_pretrain_finetune_inference_test.csv" \
  --seq_len 30 \
  --batch_size 256

echo "Focused branch training and inference completed."