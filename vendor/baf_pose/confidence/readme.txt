BAF-Pose Behavioral Confidence Estimator

This directory contains the behavioral confidence estimator used by BAF-Pose.
The estimator predicts the Explorer confidence from a 30-frame 6DoF pose
history.

Required Data

Complete the data preparation process before running this module. The confidence
module uses the following dataset:

Code/data/mm23_movement_csv_resampled_30Hz_with_conf_supervision/

This dataset must contain the train, val, and test directories.

Execution

Run the following command from:

Code/baf_pose/confidence/

bash run.sh

The script performs the following steps:

Step 1: python train.py
Train the confidence estimator, evaluate it on the validation and test sets,
save the best model, and generate the training results JSON file.

Step 2: python inference.py
Apply the trained model to the test trajectories and generate the confidence
inference CSV file.

Step 3: python plot.py
Generate the validation confusion matrix and recall trade-off figures. These
figures are used only to inspect the confidence estimator and are not included
in the thesis.

Outputs

All generated files are stored under:

Code/baf_pose/confidence/results/

The main outputs are:

results/confidence_training_results.json
results/confidence_inference_test.csv
results/saved_model/confidence_best.pt
results/figures/val_confusion_matrix_ratio.pdf
results/figures/val_recall_tradeoff_by_threshold.pdf

The output directories are created automatically if they do not exist.
