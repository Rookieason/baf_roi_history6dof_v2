#!/usr/bin/env bash

set -e

cd "$(dirname "$0")"

python feature_distribution_analyzer.py \
    --input-dir "../../data/mm23_movement_csv_resampled_30Hz" \
    --output-dir "results/feature_distribution"

python behavior_distribution_analyzer.py \
    --input-dir "../../data/mm23_movement_csv_resampled_30Hz_with_label" \
    --output-dir "results/behavior_distribution"