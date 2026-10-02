#!/usr/bin/env bash

set -e

cd "$(dirname "$0")"

python txt_to_csv.py \
    --input-dir "../data/User Movement" \
    --output-dir "../data/mm23_movement_csv"

python resample_trajectory.py \
    --input-dir "../data/mm23_movement_csv" \
    --output-dir "../data/mm23_movement_csv_resampled_30Hz" \
    --target-hz 30

python csv_label_generator.py \
    --input-dir "../data/mm23_movement_csv_resampled_30Hz" \
    --output-dir "../data/mm23_movement_csv_resampled_30Hz_with_label"

python split_data.py \
    --input-dir "../data/mm23_movement_csv_resampled_30Hz_with_label" \
    --output-dir "../data/mm23_movement_csv_resampled_30Hz_with_label" \
    --train-ratio 0.70 \
    --val-ratio 0.15 \
    --test-ratio 0.15 \
    --seed 57

python supervision.py \
    --input-dir "../data/mm23_movement_csv_resampled_30Hz" \
    --output-dir "../data/mm23_movement_csv_resampled_30Hz_with_conf_supervision" \
    --split_after_supervision \
    --split_seed 57 \
    --train_ratio 0.70 \
    --val_ratio 0.15 \
    --test_ratio 0.15

echo "Data preparation completed."
