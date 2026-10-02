BAF-Pose Chapter 3 Figure Generation

This directory contains the programs used to generate the behavior and feature
distribution figures presented in Chapter 3.

Before running these programs, complete the data preparation process.

Execution

Run the following commands from:

Code/data_preparation/data_analyzer/

Step 1: Generate the feature distribution figures.

python feature_distribution_analyzer.py \
    --input-dir "../../data/mm23_movement_csv_resampled_30Hz" \
    --output-dir "results/feature_distribution"

This program generates:

gaze_entropy_dist.pdf
fixation_ratio_dist.pdf
head_yaw_entropy_dist.pdf
head_rate_yaw_q90_dist.pdf
head_rate_yaw_hi_prop_dist.pdf
eye_saccade_rate_dist.pdf
yaw_zero_cross_per_s_dist.pdf
head_speed_mean_dist.pdf

Step 2: Generate the behavior distribution figure.

python behavior_distribution_analyzer.py \
    --input-dir "../../data/mm23_movement_csv_resampled_30Hz_with_label" \
    --output-dir "results/behavior_distribution"

This program generates:

behavior_distribution_micro.pdf

All figures are saved in:

results/

The output directory is created automatically if it does not exist.