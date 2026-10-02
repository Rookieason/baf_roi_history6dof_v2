# ROI Prediction Ablation: 6DoF-Only vs Heatmap + 6DoF

## Goal

The goal of this experiment is to verify whether RF heatmap information provides additional useful signals for ROI selection compared with using only 6DoF motion history.

We compare two lightweight models:

1. **6DoF-only baseline**
2. **Heatmap + 6DoF model**

Both models predict the future ROI region for point-cloud / volumetric streaming.

---

## Dataset Location

The experiment will run on the machine where the full dataset is stored:

```text
/home/tonic/Projects/NSTC/sensor-agent-publish-subscribe/
```

The dataset contains many recorded sessions. Each session has corresponding data under:

```text
/home/tonic/Projects/NSTC/sensor-agent-publish-subscribe/artifacts/{var}
```

and

```text
/home/tonic/Projects/NSTC/sensor-agent-publish-subscribe/db/{var}
```

For each `{var}`, the corresponding `artifacts/{var}` and `db/{var}` folders belong to the same recorded session.

Inside each `db/{var}` folder, the 6DoF file name may vary, but it always starts with:

```text
agent6
```

Therefore, the loader should search by prefix instead of relying on a fixed filename.

also

```text
/home/tonic/Projects/NSTC/sensor-agent-publish-subscribe/heatmaps/heatmap_result/figures/{var}/ToF-Doppler/
```

and

```text
/home/tonic/Projects/NSTC/sensor-agent-publish-subscribe/heatmaps/heatmap_result/mat/{var}/ToF-Doppler/smoothed_CSI_avg.mat
```

---

## Important Implementation Constraint

The full dataset may contain thousands of sessions, so the code must avoid loading everything into memory at once.

The implementation should:

* scan dataset metadata first;
* process one session or one mini-batch at a time;
* stream samples during training;
* cache only lightweight metadata when necessary;
* avoid storing all heatmaps in RAM;
* use memory-mapped loading if needed.
* carefully handle OOM problem.

---

## Experiment Pipeline

### 1. Dataset Discovery

The code should scan:

```text
/home/tonic/Projects/NSTC/sensor-agent-publish-subscribe/artifacts/
```

and find all session folders.

For each session id `{var}`, the code should verify that the matching folder exists:

Each valid session should be registered as one dataset unit.

---

### 2. Train / Validation / Test Split

The dataset should be split by session, not by individual frame.

This prevents frames from the same recording sequence from appearing in both training and testing data.

Recommended split:

```text
train: 70%
valid: 15%
test: 15%
```

The split should be saved to a JSON file so future experiments use the same split.

Example:

```text
outputs/splits/dataset_split.json
```

---

### 3. Timestamp Alignment

For each session, the loader should align:

* 6DoF pose records;
* heatmap records;
* ROI ground-truth target.

Because different sensors may not be recorded at exactly the same timestamp, alignment should be done by nearest timestamp within a maximum tolerance.

Example:

```text
aligned_sample = {
  "timestamp": ...,
  "pose_history": ...,
  "heatmap": ...,
  "target_roi": ...
}
```

If no valid match is found within the tolerance, that sample should be skipped.

---

### 4. ROI Ground Truth Generation

The 3D world is divided into a regular grid:

```text
p x p x p
```

Each grid cell is treated as one ROI candidate.

The ground-truth ROI is generated from the future 6DoF position.

For example, given the current timestamp `t`, the target can be the ROI cell that contains the user position at:

```text
t + prediction_horizon
```

Possible prediction horizons:

```text
50 ms, 100 ms, 150 ms, 200 ms
```

The exact horizon should be configurable.

This is reasonable because the recorded future 6DoF trajectory provides the ground truth.

---

### 5. Input Features

#### 5.1 6DoF-Only Baseline

The 6DoF-only model uses recent pose history as input.

Possible features:

* position: `x, y, z`
* rotation: yaw, pitch, roll or quaternion
* velocity
* angular velocity
* acceleration if needed

The baseline should be lightweight.

A simple baseline can be:

* linear regression;
* logistic regression over ROI cells;
* small MLP.

---

#### 5.2 Heatmap + 6DoF Model

The heatmap + 6DoF model uses both:

* the same 6DoF features as the baseline;
* compact heatmap features.

The heatmap should not be fed as a huge raw tensor unless memory and speed are acceptable.

Recommended lightweight heatmap features:

* top-k peak locations;
* top-k peak values;
* heatmap center of mass;
* total heatmap energy;
* local energy around candidate ROI directions;
* low-resolution pooled heatmap.

This keeps the model suitable for realtime ROI selection.

---

### 6. Lightweight Model Training

The model should be intentionally small.

Recommended model choices:

1. **6DoF-only baseline**

   * linear regression or small MLP

2. **Heatmap + 6DoF**

   * small MLP
   * gradient boosting model
   * logistic regression with engineered heatmap features

The model output can be either:

* ROI class index;
* top-k ROI candidates;
* future 3D position, then converted to ROI.

For ROI selection, direct ROI classification is simpler.

---

## Evaluation Metrics

The comparison should include multiple metrics instead of only one score.

Recommended metrics:

### 1. Top-1 ROI Accuracy

Whether the predicted ROI cell matches the ground-truth ROI cell.

### 2. Top-k ROI Accuracy

Whether the ground-truth ROI appears in the top-k predicted ROI candidates.

Recommended values:

```text
k = 3, 5, 10
```

### 3. ROI Distance Error

Distance between the predicted ROI cell center and the ground-truth ROI cell center.

### 4. Future Position Error

If the model predicts future position before converting to ROI, evaluate the position error in meters.

### 5. Latency

Measure model inference time per sample.

This is important because ROI selection must run in realtime.

### 6. Memory Usage

Track peak memory usage during training and inference.

This is important because the dataset is large and heatmaps may be expensive.

### 7. Per-Horizon Performance

Evaluate separately for each prediction horizon:

```text
50 ms
100 ms
150 ms
200 ms
```

Heatmap information may become more useful at some horizons than others.

---

## Expected Output Structure

All new experiment code should be placed in a new folder, for example:

```text
experiments/roi_heatmap_ablation/
```

Suggested structure:

```text
experiments/roi_heatmap_ablation/
├── configs/
│   └── default.json
├── data/
│   ├── discover_sessions.py
│   ├── dataset_split.py
│   ├── timestamp_alignment.py
│   └── roi_label.py
├── features/
│   ├── pose_features.py
│   └── heatmap_features.py
├── models/
│   ├── baseline_6dof.py
│   └── heatmap_6dof.py
├── train.py
├── evaluate.py
├── run_experiment.py
└── README.md
```

---

## Notes and Corrections

The overall design is reasonable.

However, one detail should be clarified:

Using 6DoF history to generate ground truth is valid only if the target is the future ROI derived from future 6DoF position. The model input must only use past/current 6DoF data, not future 6DoF data.

Correct formulation:

```text
Input:
  6DoF history up to time t
  heatmap at or before time t

Target:
  ROI cell containing the user position at time t + horizon
```

Incorrect formulation:

```text
Input includes future 6DoF used to generate the target
```

That would cause data leakage.

Also, the dataset split should be session-based. Random frame-level splitting is not recommended because nearby frames are highly correlated and would make the test result overly optimistic.

---

## Final Comparison

The final report should compare:

```text
6DoF-only
vs
Heatmap + 6DoF
```

using the same train / valid / test split, the same prediction horizons, and the same ROI grid setting.

The main question is:

```text
Does adding heatmap-derived spatial information improve ROI prediction accuracy without making inference too slow for realtime use?
```
