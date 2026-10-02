BAF-Pose Data Preparation

This directory contains the programs used to prepare the trajectory dataset for BAF-Pose.

Dataset Setup

Download the MM'23 User Behavior Dataset from:

https://drive.google.com/drive/folders/10IneAJ6uMoI_BI93tuKj5_Oxj9qbkM29

From the downloaded dataset, copy the "User Movement" directory to:

Code/data/User Movement/

#############################################################################################################
IMPORTANT DATA CLEANING NOTICE!

The 6DoF data in "WangYan_chatting" are missing from Frame 3258 onward.
Remove Frame 3258 and all subsequent frames from this trajectory before running the data preparation process.
#############################################################################################################

Execution

Run the following command from the Code directory:

bash data_preparation/run.sh

The script performs the following steps:

Step 1: python txt_to_csv.py
Convert the original TXT trajectory files to CSV files.

Step 2: python resample_trajectory.py
Resample the trajectories to 30 Hz.

Step 3: python csv_label_generator.py
Generate motion features and behavioral labels.

Step 4: python split_data.py
Split the labeled files into training, validation, and test sets.

Step 5: python supervision.py
Generate confidence supervision data and split the files into training, validation, and test sets.

Generated Data

The intermediate data are stored in:

Code/data/mm23_movement_csv/
Code/data/mm23_movement_csv_resampled_30Hz/

The labeled dataset is stored in:

Code/data/mm23_movement_csv_resampled_30Hz_with_label/train/
Code/data/mm23_movement_csv_resampled_30Hz_with_label/val/
Code/data/mm23_movement_csv_resampled_30Hz_with_label/test/

The dataset with confidence supervision is stored in:

Code/data/mm23_movement_csv_resampled_30Hz_with_conf_supervision/train/
Code/data/mm23_movement_csv_resampled_30Hz_with_conf_supervision/val/
Code/data/mm23_movement_csv_resampled_30Hz_with_conf_supervision/test/

Dataset Usage

The Motivation LSTM may use either of the two datasets above. It reads the pose and Label columns and ignores the additional confidence supervision columns.

BAF-Pose and all SOTA baselines must use the dataset with confidence supervision. BAF-Pose uses the additional supervision columns, while the SOTA baselines use the same files and data split for a consistent comparison.

Only the train, val, and test directories contain the final data used for model training and evaluation.
