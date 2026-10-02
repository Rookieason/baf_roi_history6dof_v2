from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


MODEL_LABELS = {
    "6dof": "Pose only",
    "heatmap_6dof": "Heatmap + pose",
}

MODEL_COLORS = {
    "6dof": "#4C78A8",
    "heatmap_6dof": "#F58518",
}

QUALITY_METRICS = [
    ("iou", "IoU"),
    ("f1", "F1"),
    ("recall", "Recall"),
    ("precision", "Precision"),
    ("per_cell_accuracy", "Per-cell acc."),
    ("exact_match_accuracy", "Exact match"),
]


def newest_log(default_root: Path) -> Path:
    logs_dir = default_root / "outputs" / "roi_heatmap_ablation" / "logs"
    logs = sorted(logs_dir.glob("*.log"), key=lambda path: path.stat().st_mtime, reverse=True)
    if not logs:
        raise FileNotFoundError(f"No .log files found under {logs_dir}")
    return logs[0]


def load_report(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < start:
        raise ValueError(f"{path} does not contain a JSON object")
    return json.loads(text[start : end + 1])


def extract_rows(report: dict) -> pd.DataFrame:
    rows: list[dict] = []
    for horizon_key, by_model in sorted(report["results"].items(), key=lambda item: int(item[0])):
        horizon_ms = int(horizon_key)
        for model_name, payload in sorted(by_model.items()):
            test = payload.get("test", {})
            train = payload.get("train", {})
            losses = train.get("losses", [])
            row = {
                "horizon_ms": horizon_ms,
                "model": model_name,
                "model_label": MODEL_LABELS.get(model_name, model_name),
                "train_samples": train.get("samples"),
                "train_loss_initial": losses[0] if losses else np.nan,
                "train_loss_final": losses[-1] if losses else np.nan,
                "train_loss_drop": (losses[0] - losses[-1]) if losses else np.nan,
            }
            row.update(test)
            rows.append(row)
    return pd.DataFrame(rows).sort_values(["horizon_ms", "model"]).reset_index(drop=True)


def compute_deltas(metrics: pd.DataFrame) -> pd.DataFrame:
    baseline = metrics[metrics["model"] == "6dof"].set_index("horizon_ms")
    heatmap = metrics[metrics["model"] == "heatmap_6dof"].set_index("horizon_ms")
    horizons = sorted(set(baseline.index).intersection(heatmap.index))

    numeric_cols = [
        "exact_match_accuracy",
        "per_cell_accuracy",
        "precision",
        "recall",
        "f1",
        "iou",
        "latency_ms_mean",
        "latency_ms_p95",
        "peak_memory_mb",
        "samples",
    ]
    rows: list[dict] = []
    for horizon in horizons:
        for metric in numeric_cols:
            base_value = float(baseline.loc[horizon, metric])
            heat_value = float(heatmap.loc[horizon, metric])
            rel_pct = ((heat_value - base_value) / base_value * 100.0) if base_value else np.nan
            rows.append(
                {
                    "horizon_ms": horizon,
                    "metric": metric,
                    "pose_only": base_value,
                    "heatmap_pose": heat_value,
                    "delta_abs": heat_value - base_value,
                    "delta_rel_pct": rel_pct,
                }
            )
    return pd.DataFrame(rows)


def plot_quality(metrics: pd.DataFrame, output_dir: Path) -> None:
    horizons = sorted(metrics["horizon_ms"].unique())
    fig, axes = plt.subplots(2, 3, figsize=(14, 7), sharex=True)
    axes = axes.ravel()
    for ax, (metric, title) in zip(axes, QUALITY_METRICS):
        for model_name, group in metrics.groupby("model", sort=False):
            group = group.sort_values("horizon_ms")
            ax.plot(
                group["horizon_ms"],
                group[metric],
                marker="o",
                linewidth=2.2,
                color=MODEL_COLORS.get(model_name),
                label=MODEL_LABELS.get(model_name, model_name),
            )
        ax.set_title(title)
        ax.set_xlabel("Prediction horizon (ms)")
        ax.set_xticks(horizons)
        ax.grid(True, alpha=0.25)
        ymin = max(0.0, float(metrics[metric].min()) - 0.05)
        ymax = min(1.0, float(metrics[metric].max()) + 0.05)
        ax.set_ylim(ymin, ymax)
    axes[0].legend(loc="best", frameon=False)
    fig.suptitle("ROI prediction quality by horizon", fontsize=15, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_dir / "quality_metrics_by_horizon.png", dpi=180)
    plt.close(fig)


def plot_efficiency(metrics: pd.DataFrame, output_dir: Path) -> None:
    horizons = sorted(metrics["horizon_ms"].unique())
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    ax_latency, ax_memory, ax_tradeoff, ax_samples = axes.ravel()

    for model_name, group in metrics.groupby("model", sort=False):
        group = group.sort_values("horizon_ms")
        color = MODEL_COLORS.get(model_name)
        label = MODEL_LABELS.get(model_name, model_name)
        ax_latency.plot(group["horizon_ms"], group["latency_ms_mean"], marker="o", linewidth=2.2, color=color, label=label)
        ax_memory.plot(group["horizon_ms"], group["peak_memory_mb"], marker="o", linewidth=2.2, color=color, label=label)
        ax_tradeoff.scatter(group["latency_ms_mean"], group["iou"], s=70, color=color, label=label)
        for _, row in group.iterrows():
            horizon = int(row["horizon_ms"])
            offset = {
                ("6dof", 50): (6, 5),
                ("6dof", 100): (6, -14),
                ("6dof", 150): (6, 8),
                ("6dof", 200): (6, 5),
                ("heatmap_6dof", 50): (-42, 5),
                ("heatmap_6dof", 100): (7, -16),
                ("heatmap_6dof", 150): (8, 9),
                ("heatmap_6dof", 200): (8, -4),
            }.get((model_name, horizon), (5, 4))
            ax_tradeoff.annotate(
                f"{horizon} ms",
                (row["latency_ms_mean"], row["iou"]),
                textcoords="offset points",
                xytext=offset,
                ha="right" if offset[0] < 0 else "left",
                fontsize=8,
            )

    ax_latency.set_title("Mean prediction latency")
    ax_latency.set_xlabel("Prediction horizon (ms)")
    ax_latency.set_ylabel("ms/sample")
    ax_latency.set_xticks(horizons)
    ax_latency.grid(True, alpha=0.25)
    ax_latency.legend(frameon=False)

    ax_memory.set_title("Peak evaluation memory")
    ax_memory.set_xlabel("Prediction horizon (ms)")
    ax_memory.set_ylabel("MB")
    ax_memory.set_xticks(horizons)
    ax_memory.set_yscale("log")
    ax_memory.grid(True, alpha=0.25, which="both")

    ax_tradeoff.set_title("Quality/latency tradeoff")
    ax_tradeoff.set_xlabel("Mean latency (ms/sample)")
    ax_tradeoff.set_ylabel("IoU")
    x_min = float(metrics["latency_ms_mean"].min())
    x_max = float(metrics["latency_ms_mean"].max())
    y_min = float(metrics["iou"].min())
    y_max = float(metrics["iou"].max())
    ax_tradeoff.set_xlim(x_min - (x_max - x_min) * 0.08, x_max + (x_max - x_min) * 0.08)
    ax_tradeoff.set_ylim(y_min - (y_max - y_min) * 0.12, y_max + (y_max - y_min) * 0.12)
    ax_tradeoff.grid(True, alpha=0.25)
    ax_tradeoff.legend(frameon=False)

    pivot = metrics.pivot(index="horizon_ms", columns="model_label", values="samples")
    pivot = pivot[[label for label in MODEL_LABELS.values() if label in pivot.columns]]
    pivot.plot(kind="bar", ax=ax_samples, color=[MODEL_COLORS["6dof"], MODEL_COLORS["heatmap_6dof"]])
    ax_samples.set_title("Evaluated test samples")
    ax_samples.set_xlabel("Prediction horizon (ms)")
    ax_samples.set_ylabel("samples")
    ax_samples.tick_params(axis="x", rotation=0)
    ax_samples.grid(True, axis="y", alpha=0.25)
    ax_samples.legend(frameon=False)

    fig.suptitle("Efficiency and evaluation coverage", fontsize=15, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_dir / "efficiency_tradeoffs.png", dpi=180)
    plt.close(fig)


def plot_training_loss(report: dict, output_dir: Path) -> None:
    horizons = sorted(report["results"], key=lambda value: int(value))
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    axes = axes.ravel()
    for ax, horizon_key in zip(axes, horizons):
        for model_name, payload in sorted(report["results"][horizon_key].items()):
            losses = payload.get("train", {}).get("losses", [])
            if not losses:
                continue
            epochs = np.arange(1, len(losses) + 1)
            ax.plot(
                epochs,
                losses,
                marker="o",
                linewidth=2.0,
                color=MODEL_COLORS.get(model_name),
                label=MODEL_LABELS.get(model_name, model_name),
            )
        ax.set_title(f"{horizon_key} ms")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.grid(True, alpha=0.25)
    axes[0].legend(frameon=False)
    fig.suptitle("Training loss curves", fontsize=15, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_dir / "training_loss_curves.png", dpi=180)
    plt.close(fig)


def plot_quality_delta_heatmap(deltas: pd.DataFrame, output_dir: Path) -> None:
    labels = {metric: title for metric, title in QUALITY_METRICS}
    quality = deltas[deltas["metric"].isin(labels)].copy()
    quality["metric_label"] = quality["metric"].map(labels)
    matrix = quality.pivot(index="metric_label", columns="horizon_ms", values="delta_rel_pct")
    matrix = matrix.loc[[labels[metric] for metric, _ in QUALITY_METRICS]]

    fig, ax = plt.subplots(figsize=(9, 5))
    max_abs = float(np.nanmax(np.abs(matrix.to_numpy())))
    image = ax.imshow(matrix, cmap="RdBu_r", vmin=-max_abs, vmax=max_abs, aspect="auto")
    ax.set_xticks(np.arange(matrix.shape[1]), labels=[str(col) for col in matrix.columns])
    ax.set_yticks(np.arange(matrix.shape[0]), labels=matrix.index)
    ax.set_xlabel("Prediction horizon (ms)")
    ax.set_title("Heatmap + pose relative change vs pose only")
    for row_index in range(matrix.shape[0]):
        for col_index in range(matrix.shape[1]):
            value = matrix.iloc[row_index, col_index]
            ax.text(col_index, row_index, f"{value:+.1f}%", ha="center", va="center", fontsize=9)
    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label("% change")
    fig.tight_layout()
    fig.savefig(output_dir / "quality_relative_change_heatmap.png", dpi=180)
    plt.close(fig)


def markdown_table(frame: pd.DataFrame | pd.Series) -> str:
    if isinstance(frame, pd.Series):
        table = frame.reset_index()
        table.columns = [frame.index.name or "metric", frame.name or "value"]
    else:
        table = frame.reset_index()

    formatted = table.copy()
    for column in formatted.columns:
        formatted[column] = formatted[column].map(lambda value: f"{value:.4f}" if isinstance(value, float) else str(value))

    headers = list(formatted.columns)
    rows = formatted.values.tolist()
    widths = [
        max(len(header), *(len(str(row[index])) for row in rows))
        for index, header in enumerate(headers)
    ]
    header_line = "| " + " | ".join(header.ljust(widths[index]) for index, header in enumerate(headers)) + " |"
    rule_line = "| " + " | ".join("-" * widths[index] for index in range(len(headers))) + " |"
    row_lines = [
        "| " + " | ".join(str(value).ljust(widths[index]) for index, value in enumerate(row)) + " |"
        for row in rows
    ]
    return "\n".join([header_line, rule_line, *row_lines])


def write_summary(report: dict, metrics: pd.DataFrame, deltas: pd.DataFrame, output_dir: Path) -> None:
    mean_by_model = metrics.groupby(["model", "model_label"])[
        ["iou", "f1", "precision", "recall", "per_cell_accuracy", "exact_match_accuracy", "latency_ms_mean", "peak_memory_mb"]
    ].mean()
    best_heatmap = metrics[metrics["model"] == "heatmap_6dof"].sort_values(["iou", "f1"], ascending=False).iloc[0]

    quality_delta_mean = (
        deltas[deltas["metric"].isin(["iou", "f1", "recall", "precision", "per_cell_accuracy", "exact_match_accuracy"])]
        .groupby("metric")["delta_abs"]
        .mean()
        .sort_index()
    )
    cost_delta_mean = deltas[deltas["metric"].isin(["latency_ms_mean", "peak_memory_mb"])].groupby("metric")["delta_abs"].mean()

    lines = [
        "# ROI heatmap ablation summary",
        "",
        f"Source log: `{report.get('_source_log', '')}`",
        "",
        "## Takeaway",
        "",
        "Heatmap + pose is better on overlap-oriented ROI metrics, especially IoU, F1, and recall. Exact full-mask match is lower, and evaluation uses fewer heatmap samples because samples without heatmaps are excluded.",
        "",
        "## Mean test metrics",
        "",
        markdown_table(mean_by_model.round(4)),
        "",
        "## Mean absolute change: heatmap + pose minus pose only",
        "",
        markdown_table(quality_delta_mean.round(4)),
        "",
        "## Mean added cost",
        "",
        markdown_table(cost_delta_mean.round(4)),
        "",
        "## Best heatmap horizon by IoU",
        "",
        f"- Horizon: {int(best_heatmap['horizon_ms'])} ms",
        f"- IoU: {best_heatmap['iou']:.4f}",
        f"- F1: {best_heatmap['f1']:.4f}",
        f"- Recall: {best_heatmap['recall']:.4f}",
        f"- Precision: {best_heatmap['precision']:.4f}",
        "",
        "## Caveat",
        "",
        "The two models are not evaluated on exactly the same number of samples: the heatmap model requires heatmap availability, while pose only does not. For a strict apples-to-apples comparison, rerun pose-only evaluation on the heatmap-available subset too.",
        "",
    ]
    (output_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Visualize ROI heatmap ablation results.")
    parser.add_argument("--log", type=Path, default=None, help="Path to run_*.log JSON output.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Directory for figures and tables.")
    args = parser.parse_args()

    log_path = args.log or newest_log(root)
    output_dir = args.output_dir or log_path.parent.parent / "visualizations"
    output_dir.mkdir(parents=True, exist_ok=True)

    report = load_report(log_path)
    report["_source_log"] = str(log_path)
    metrics = extract_rows(report)
    deltas = compute_deltas(metrics)

    metrics.to_csv(output_dir / "metrics_summary.csv", index=False)
    deltas.to_csv(output_dir / "heatmap_vs_pose_deltas.csv", index=False)
    plot_quality(metrics, output_dir)
    plot_efficiency(metrics, output_dir)
    plot_training_loss(report, output_dir)
    plot_quality_delta_heatmap(deltas, output_dir)
    write_summary(report, metrics, deltas, output_dir)

    print(f"Wrote visualizations to {output_dir}")


if __name__ == "__main__":
    main()
