# ROI Baseline Revision Specification

## Background

The current ROI baseline generates the wrong ground truth.

Currently, the pipeline assumes:

```
future HMD position
        ↓
position_to_roi(...)
        ↓
single ROI class
```

This is **not** how ROI is defined in the adaptive volumetric streaming system.

---

## Correct Definition

The ROI should be generated from

```
future HMD pose
        +
virtual object positions
        +
object partition grids
        +
view frustum
```

The label should answer:

> **At t + Δ, which partitions of every virtual object are inside the user's FoV?**

Therefore the prediction target is **NOT** a class index.

It is

```
[num_object_partitions]

0 = outside FoV
1 = inside FoV
```

For example,

```
6 objects
8 × 8 × 8 partitions/object

Output dimension

6 × 8 × 8 × 8
= 3072 binary labels
```

---

# Overall Principle

Only modify **roi_baseline-main**.

Do NOT modify

- adaptive-volumetric-stream-server-main
- sensor-agent-publish-subscribe-main

Those projects are only references for deriving the correct ROI generation logic.

Delete obsolete code instead of keeping compatibility.

Keep functions small and clearly separated.

Avoid introducing unnecessary abstractions.

---

# Files to Revise

---

# 1.

experiments/roi_heatmap_ablation/data/timestamp_alignment.py

## Current problem

AlignedSample only stores

- future position

ROI generation also needs

- future rotation

## Modify

Extend

```python
AlignedSample
```

to include

```python
target_rotation_deg
```

The iterator should output

```
history pose
history rotation

future position
future rotation
```

instead of only future position.

Nothing else needs to change in this file.

---

# 2.

experiments/roi_heatmap_ablation/data/roi_label.py

## Current problem

Current implementation

```
fit_roi_bounds()

position_to_roi()

roi_center()
```

assumes ROI is

```
future position
→ voxel index
```

Delete these functions completely.

They are fundamentally incorrect.

---

## Replace with

This file should become responsible only for

### A.

Loading scene object transforms

Read

```
adaptive-volumetric-stream-server

config_example_front_cache_transform_lod.json
```

Extract every virtual object's

```
obj_transform_pos_x_cm
obj_transform_pos_y_cm
obj_transform_pos_z_cm
```

Merge default config with stream config exactly like the streaming server.

---

### B.

Generate partition centers

Given

```
object center
object size
grid_dim
```

Generate

```
p × p × p
```

cell centers.

Return

```
(N,3)
```

for one object.

---

### C.

Build scene partition centers

Loop through every stream.

Concatenate

```
object1 partitions
object2 partitions
...
```

Return

```
(num_total_partitions,3)
```

---

### D.

Generate ROI mask

Implement

```
pose_to_roi_mask(...)
```

Input

```
future position
future rotation
partition centers
FoV
```

Output

```
[num_total_partitions]

0/1
```

Logic

1.

Build camera forward vector

using

```
yaw
pitch
roll
```

2.

For every partition

Compute

```
direction =
partition_center
-
camera_position
```

3.

Compute angle

using

```
dot(direction,forward)
```

4.

Inside FoV

↓

1

Outside FoV

↓

0

Optional

distance threshold

can also be applied.

This file should **not** know anything about machine learning.

It only generates labels.

---

# 3.

experiments/roi_heatmap_ablation/train.py

Current implementation computes

```
ROI class
```

Delete that logic.

---

## Remove

```
fit_roi_bounds()

position_to_roi()

num_classes

grid_size³
```

---

## Replace

Before training,

load

```
scene partition centers
```

once.

During dataset generation

replace

```
position_to_roi(...)
```

with

```
pose_to_roi_mask(...)
```

The training label becomes

```
binary vector
```

instead of integer class.

---

The batching function should now return

```
X

Y

(batch,
 num_total_partitions)
```

instead of

```
(batch,)
```

---

# 4.

experiments/roi_heatmap_ablation/models/baseline_6dof.py

Current model

```
SoftmaxClassifier
```

assumes

single-class classification.

Delete it.

---

Replace with

```
Linear

↓

Sigmoid

↓

Binary output
```

Training loss

```
Binary Cross Entropy
```

Prediction

```
probability

↓

threshold

↓

0/1
```

Keep

```
StreamingStandardizer
```

unchanged.

---

# 5.

experiments/roi_heatmap_ablation/evaluate.py

Current metrics

```
Top-1

Top-k

ROI distance
```

are no longer valid.

Delete all of them.

---

Generate ground truth

using

```
pose_to_roi_mask(...)
```

Predict

```
binary ROI mask
```

Compute

```
Exact Match Accuracy

Per-cell Accuracy

Precision

Recall

F1

IoU
```

Latency measurement remains unchanged.

---

# 6.

experiments/roi_heatmap_ablation/run_experiment.py

Current implementation computes

```
scene bounds
```

Delete all related code.

Instead

load

```
scene partition centers
```

once

before training.

Pass

```
grid_centers
```

into

```
train()

evaluate()
```

instead of

```
bounds
```

Metadata should store

```
adaptive config path

grid dimension

FoV

ROI output dimension
```

instead of ROI bounds.

---

# 7.

configs/default.json

Replace

```
roi
```

configuration

with

```
scene
```

Example

```json
{
    "scene": {
        "adaptive_config_path": "...",
        "grid_dim": 8,
        "fov_deg": 110,
        "max_distance_cm": null
    }
}
```

Evaluation

should use

```
threshold
```

instead of Top-K.

---

# Things That Should NOT Change

Keep

- timestamp alignment
- feature extraction
- heatmap feature extraction
- 6DoF feature extraction
- dataset split
- experiment runner
- logging
- latency measurement
- result CSV format (unless a metric no longer exists)

Only revise the ROI label generation pipeline.

---

# Final Pipeline

Current (incorrect)

```
Future Position
        │
        ▼
position_to_roi()
        │
        ▼
Single ROI Class
```

Correct

```
Future Position
        │
Future Rotation
        │
        ▼
Scene Object Positions
        │
        ▼
Partition Every Object
        │
        ▼
View Frustum Test
        │
        ▼
Binary ROI Mask
        │
        ▼
Linear + Sigmoid Baseline
        │
        ▼
Multi-label Prediction
```

---

# Expected Result

The baseline should predict

> **Which partitions of every virtual object will be visible after the prediction horizon.**

instead of

> **Which voxel contains the future HMD position.**

This aligns the baseline with the actual ROI definition used by the adaptive volumetric streaming system and makes the comparison between

- 6DoF only
- Heatmap + 6DoF

scientifically meaningful.