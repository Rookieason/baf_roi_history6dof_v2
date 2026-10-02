BAF-Pose Focused Branch

This directory contains the programs used to train and run inference with the
Focused prediction branch.

Programs

train.py
Train or fine-tune the Focused branch and save the best model and training
results.

inference.py
Run the trained Focused branch on the test set and save the predictions and
errors.

model.py
Define the LSTM-based Focused branch and its training losses, including
smoothness regularization.

utility.py
Load the trajectory data and construct Focused training samples.

run.sh
Run pre-training, Focused fine-tuning, and inference in sequence.

Dataset

Complete the data preparation process before running this directory. The
Focused branch uses:

Code/data/mm23_movement_csv_resampled_30Hz_with_conf_supervision/train/
Code/data/mm23_movement_csv_resampled_30Hz_with_conf_supervision/val/
Code/data/mm23_movement_csv_resampled_30Hz_with_conf_supervision/test/

Execution

Run the following command from:

Code/baf_pose/prediction/focused_branch/

bash run.sh

The script performs the following steps:

Step 1: python train.py
Pre-train the Focused branch using all valid training and validation samples.

Step 2: python train.py
Fine-tune the pre-trained model using Focused-to-Focused samples.

Step 3: python inference.py
Run inference on the test set using the best fine-tuned model.

Results

All outputs are stored in:

results/

The generated files are:

results/focused_pretrain_all_training_results.json
results/focused_pretrain_finetune_training_results.json
results/focused_pretrain_finetune_inference_test.csv
results/saved_model/focused_pretrain_all_best.pt
results/saved_model/focused_pretrain_finetune_best.pt

The output directories are created automatically if they do not exist.
