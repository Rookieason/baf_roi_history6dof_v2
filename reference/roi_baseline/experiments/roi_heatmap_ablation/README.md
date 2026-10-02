# ROI Heatmap Ablation

This experiment compares two lightweight ROI predictors:

- `6dof`: pose history only
- `heatmap_6dof`: the same pose history plus compact RF heatmap features

The pipeline discovers valid sessions by matching `artifacts/{session}`, `db/{session}`, an `agent6*` pose CSV, heatmap figures, and `smoothed_CSI_avg.mat`. Splits are saved by session to avoid frame-level leakage.

Run from the repository root:

```bash
python -m experiments.roi_heatmap_ablation.run_experiment
```

For a local copy of the dataset:

```bash
python -m experiments.roi_heatmap_ablation.run_experiment --dataset-root .
```

Outputs are written under `outputs/roi_heatmap_ablation/`, with the reusable split at `outputs/splits/dataset_split.json`.

The loader streams one session at a time. It keeps only lightweight session metadata globally, loads pose records for the active session, extracts compact heatmap features from that session, and never stores all heatmaps or all samples in memory.
