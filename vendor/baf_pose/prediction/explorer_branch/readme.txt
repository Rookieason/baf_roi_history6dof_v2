BAF-Pose Explorer Branch

This directory contains the programs used to train and run inference with the
Explorer prediction branch.

Programs

train.py
Train the Explorer branch and save the best model and training results.

inference.py
Run the trained Explorer branch on the test set and save the predictions and
errors.

model.py
Define the TCN-based Explorer branch and its training losses.

utility.py
Load the trajectory data and construct Explorer training samples.

run.sh
Run training and inference in sequence.

Dataset

Complete the data preparation process before running this directory. The
Explorer branch uses:

Code/data/mm23_movement_csv_resampled_30Hz_with_conf_supervision/train/
Code/data/mm23_movement_csv_resampled_30Hz_with_conf_supervision/val/
Code/data/mm23_movement_csv_resampled_30Hz_with_conf_supervision/test/

Execution

Run the following command from:

Code/baf_pose/prediction/explorer_branch/

bash run.sh

The script performs the following steps:

Step 1: python train.py
Train the Explorer branch using the training and validation sets.

Step 2: python inference.py
Run inference on the test set using the best saved model.

Results

All outputs are stored in:

results/

The generated files are:

results/explorer_residual_rot_training_results.json
results/explorer_residual_rot_inference_test.csv
results/saved_model/explorer_residual_rot_best.pt

The output directories are created automatically if they do not exist.
