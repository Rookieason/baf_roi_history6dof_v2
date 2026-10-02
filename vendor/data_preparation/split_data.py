"""Split labeled trajectory files into train, validation, and test subsets."""

import argparse
import random
import shutil
from pathlib import Path


def split_csv_files(
    input_dir: Path,
    output_dir: Path,
    train_ratio=0.70,
    val_ratio=0.15,
    test_ratio=0.15,
    seed=57,
):
    if abs(train_ratio + val_ratio + test_ratio - 1.0) >= 1e-6:
        raise ValueError("The train, validation, and test ratios must sum to 1")
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    files = sorted(input_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files found in: {input_dir}")

    random.Random(seed).shuffle(files)
    train_count = int(len(files) * train_ratio)
    val_count = int(len(files) * val_ratio)
    subsets = {
        "train": files[:train_count],
        "val": files[train_count:train_count + val_count],
        "test": files[train_count + val_count:],
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    for subset_name, subset_files in subsets.items():
        subset_dir = output_dir / subset_name
        subset_dir.mkdir(parents=True, exist_ok=True)
        for old_file in subset_dir.glob("*.csv"):
            old_file.unlink()
        for source_path in subset_files:
            shutil.copy2(source_path, subset_dir / source_path.name)

    manifest_path = output_dir / "split_manifest.txt"
    with manifest_path.open("w", encoding="utf-8") as manifest:
        manifest.write(f"seed={seed}\n")
        manifest.write(
            f"ratios=train:{train_ratio},val:{val_ratio},test:{test_ratio}\n"
        )
        for subset_name, subset_files in subsets.items():
            for source_path in subset_files:
                manifest.write(f"{subset_name}\t{source_path.name}\n")

    print(f"[DONE] Total files: {len(files)}")
    for subset_name, subset_files in subsets.items():
        print(f"[INFO] {subset_name}: {len(subset_files)} files")
    print(f"[INFO] Split manifest: {manifest_path}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=57)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    split_csv_files(
        arguments.input_dir.expanduser(),
        arguments.output_dir.expanduser(),
        arguments.train_ratio,
        arguments.val_ratio,
        arguments.test_ratio,
        arguments.seed,
    )
