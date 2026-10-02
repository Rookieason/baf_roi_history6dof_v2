BAF-Pose Decision Fusion

This directory performs final decision fusion using the trained confidence,
Focused branch, and Explorer branch models.

Prerequisites

Complete the confidence, Focused branch, and Explorer branch training processes
before running this directory. The following model files are required:

Code/baf_pose/confidence/results/saved_model/confidence_best.pt
Code/baf_pose/prediction/focused_branch/results/saved_model/focused_pretrain_finetune_best.pt
Code/baf_pose/prediction/explorer_branch/results/saved_model/explorer_residual_rot_best.pt

The test data are read from:

Code/data/mm23_movement_csv_resampled_30Hz_with_conf_supervision/test/

Execution

Run the following commands from the project root:

cd Code/baf_pose/decision
bash run.sh

The script generates results for two methods.

1. baf_pose

This method uses the estimated behavioral confidence, temporal smoothing with
mu = 0.6, linear interpolation for translation, and spherical linear
interpolation for rotation.

2. without_behavior_fusion

This method removes behavior-dependent weighting and uses a fixed fusion weight
of 0.5. Translation and rotation use the same geometric interpolation as
BAF-Pose.

Outputs

results/
|-- baf_pose/
|   |-- predictions.csv
|   |-- summary_by_behavior.csv
|   `-- run.log
`-- without_behavior_fusion/
    |-- predictions.csv
    |-- summary_by_behavior.csv
    `-- run.log

All output directories are created automatically if they do not exist.
