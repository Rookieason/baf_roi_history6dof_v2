from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid", context="paper")
plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
})


def load_json(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def get_result_block(data: Dict, split: str) -> Tuple[Dict, Dict, List[Dict], str]:
    if split == "test":
        result_block = data["final_test_result"]
        metrics_default = result_block["test_metrics_default_threshold_0_5"]
        sweep = result_block["test_threshold_sweep"]
        prefix = "test"
    else:
        result_block = data["best_validation_result"]
        metrics_default = result_block["val_default_threshold_metrics"]
        sweep = result_block["val_threshold_sweep"]
        prefix = "val"

    return result_block, metrics_default, sweep, prefix


def build_confusion_matrix(metrics_block: Dict) -> Tuple[np.ndarray, np.ndarray]:
    cm = metrics_block["confusion_matrix"]

    tn = cm["tn_focused"]
    fp = cm["fp_focused_as_explorer"]
    fn = cm["fn_explorer_as_focused"]
    tp = cm["tp_explorer"]

    mat_counts = np.array([
        [tn, fp],
        [fn, tp],
    ], dtype=np.int64)

    row_sums = mat_counts.sum(axis=1, keepdims=True)
    mat_ratio = mat_counts / np.maximum(row_sums, 1)

    return mat_counts, mat_ratio


def plot_confusion_matrix_with_ratio(
    metrics_block: Dict,
    split_name: str,
    save_path: str,
) -> None:
    mat_counts, mat_ratio = build_confusion_matrix(metrics_block)

    annot = np.empty_like(mat_counts, dtype=object)
    for i in range(mat_counts.shape[0]):
        for j in range(mat_counts.shape[1]):
            annot[i, j] = f"{mat_counts[i, j]}\n({mat_ratio[i, j]*100:.1f}%)"

    plt.figure(figsize=(6, 5))
    ax = sns.heatmap(
        mat_ratio,
        annot=annot,
        fmt="",
        cmap="Blues",
        cbar=True,
        xticklabels=["Focused", "Explorer"],
        yticklabels=["Focused", "Explorer"],
        vmin=0.0,
        vmax=1.0,
    )

    threshold = metrics_block["threshold"]
    plt.title(f"Confusion Matrix (Threshold = {threshold:.1f})")
    plt.ylabel("Actual Behavior")
    plt.xlabel("Predicted Behavior")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, format="pdf")
    plt.close()


def extract_recall_curves(sweep: List[Dict]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    thresholds = []
    explorer_recall = []
    focused_recall = []

    for item in sweep:
        thresholds.append(float(item["threshold"]))
        explorer_recall.append(float(item["per_class"]["Explorer"]["recall"]) * 100.0)
        focused_recall.append(float(item["per_class"]["Focused"]["recall"]) * 100.0)

    return (
        np.array(thresholds, dtype=np.float32),
        np.array(explorer_recall, dtype=np.float32),
        np.array(focused_recall, dtype=np.float32),
    )


def plot_recall_tradeoff_by_threshold(
    sweep: List[Dict],
    selected_threshold: float,
    split_name: str,
    save_path: str,
) -> None:
    thresholds, explorer_recall, focused_recall = extract_recall_curves(sweep)

    plt.figure(figsize=(7, 5))
    plt.plot(
        thresholds,
        explorer_recall,
        marker="o",
        linewidth=2,
        label="Explorer Recall",
        color="#e74c3c",
    )
    plt.plot(
        thresholds,
        focused_recall,
        marker="s",
        linewidth=2,
        label="Focused Recall",
        color="#3498db",
    )

    plt.axvline(x=selected_threshold, color="gray", linestyle="--", alpha=0.7)
    plt.text(
        selected_threshold + 0.01,
        min(explorer_recall.min(), focused_recall.min()) + 5,
        f"Selected\nThreshold = {selected_threshold:.1f}",
        color="gray",
    )

    plt.title(f"{split_name.upper()} Recall Trade-off across Thresholds")
    plt.xlabel("Classification Threshold")
    plt.ylabel("Recall (%)")
    plt.legend(loc="lower left")
    plt.ylim(40, 100)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, format="pdf")
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./results/figures")
    parser.add_argument("--split", type=str, default="test", choices=["test", "val"])
    parser.add_argument(
        "--selected_threshold",
        type=float,
        default=0.5,
        help="Threshold to highlight on recall trade-off plot",
    )
    args = parser.parse_args()

    ensure_dir(args.output_dir)

    data = load_json(args.json_path)
    _, metrics_default, sweep, prefix = get_result_block(data, args.split)

    cm_path = os.path.join(args.output_dir, f"{prefix}_confusion_matrix_ratio.pdf")
    recall_path = os.path.join(args.output_dir, f"{prefix}_recall_tradeoff_by_threshold.pdf")

    plot_confusion_matrix_with_ratio(
        metrics_block=metrics_default,
        split_name=prefix,
        save_path=cm_path,
    )

    plot_recall_tradeoff_by_threshold(
        sweep=sweep,
        selected_threshold=args.selected_threshold,
        split_name=prefix,
        save_path=recall_path,
    )

    print("[INFO] Saved figures:")
    print(f"  - {cm_path}")
    print(f"  - {recall_path}")


if __name__ == "__main__":
    main()
