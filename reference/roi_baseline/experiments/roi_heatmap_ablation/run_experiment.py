from __future__ import annotations

import argparse
import json
from pathlib import Path

from .data.dataset_split import load_or_create_split
from .data.discover_sessions import discover_sessions, sessions_from_ids
from .data.roi_label import build_scene_partition_centers
from .evaluate import evaluate_one, save_report
from .train import save_metadata, train_one


def load_config(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def resolve_path(dataset_root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else dataset_root / path


def resolve_existing_path(config_path: Path, dataset_root: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    for base in (config_path.parent, Path.cwd(), dataset_root):
        candidate = (base / path).resolve()
        if candidate.exists():
            return candidate
    return (Path.cwd() / path).resolve()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ROI heatmap ablation.")
    parser.add_argument("--config", default=str(Path(__file__).parent / "configs" / "default.json"))
    parser.add_argument("--dataset-root", default=None)
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    config = load_config(config_path)
    dataset_root = Path(args.dataset_root or config["dataset_root"]).resolve()
    outputs_dir = resolve_path(dataset_root, config["outputs_dir"])
    split_path = resolve_path(dataset_root, config["split_path"])

    sessions = discover_sessions(dataset_root)
    if not sessions:
        raise RuntimeError(f"No valid sessions found under {dataset_root}")

    split_cfg = config["split"]
    split = load_or_create_split(
        split_path,
        sessions,
        train_ratio=float(split_cfg["train"]),
        valid_ratio=float(split_cfg["valid"]),
        seed=int(config.get("seed", 7)),
    )
    train_sessions = sessions_from_ids(sessions, split["train"])
    test_sessions = sessions_from_ids(sessions, split["test"] or split["valid"])

    scene_cfg = config["scene"]
    adaptive_config_path = resolve_existing_path(config_path, dataset_root, scene_cfg["adaptive_config_path"])
    grid_dim = int(scene_cfg["grid_dim"])
    partition_centers = build_scene_partition_centers(
        adaptive_config_path,
        grid_dim,
        scene_cfg.get("object_size_cm", 100.0),
    )

    metadata = {
        "dataset_root": str(dataset_root),
        "split_path": str(split_path),
        "adaptive_config_path": str(adaptive_config_path),
        "grid_dim": grid_dim,
        "object_size_cm": scene_cfg.get("object_size_cm", 100.0),
        "fov_deg": float(scene_cfg["fov_deg"]),
        "max_distance_cm": scene_cfg.get("max_distance_cm"),
        "roi_output_dim": int(partition_centers.shape[0]),
        "horizons_ms": config["horizons_ms"],
    }
    save_metadata(outputs_dir / "metadata.json", metadata)

    report = {"metadata": metadata, "results": {}}
    for horizon_ms in config["horizons_ms"]:
        report["results"][str(horizon_ms)] = {}
        for model_kind in ("6dof", "heatmap_6dof"):
            model_path = outputs_dir / "models" / f"{model_kind}_h{horizon_ms}.npz"
            if not args.skip_train:
                train_info = train_one(train_sessions, int(horizon_ms), model_kind, config, partition_centers, model_path)
                report["results"][str(horizon_ms)][model_kind] = {"train": train_info}
            if not args.skip_eval:
                metrics = evaluate_one(test_sessions, int(horizon_ms), model_kind, config, model_path, partition_centers)
                report["results"][str(horizon_ms)].setdefault(model_kind, {})["test"] = metrics

    save_report(outputs_dir / "comparison_report.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

