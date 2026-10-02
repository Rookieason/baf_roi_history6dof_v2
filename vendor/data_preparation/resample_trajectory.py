"""Resample trajectory CSV files at a fixed rate with circular angle handling."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


ROTATION_COLUMNS = {
    "HeadRX", "HeadRY", "HeadRZ",
    "LHandRX", "LHandRY", "LHandRZ",
    "RHandRX", "RHandRY", "RHandRZ",
    "LEyeRX", "LEyeRY", "LEyeRZ",
    "REyeRX", "REyeRY", "REyeRZ",
}


def wrap_to_pi(radians):
    return (radians + np.pi) % (2 * np.pi) - np.pi


def interpolate_angle_degrees(new_time, timer, angle_degrees):
    valid = np.isfinite(angle_degrees) & np.isfinite(timer)
    if valid.sum() < 2:
        return np.full_like(new_time, np.nan, dtype=float)

    valid_time = timer[valid].astype(float)
    valid_angles = angle_degrees[valid].astype(float)
    unwrapped = np.unwrap(np.deg2rad(valid_angles))
    interpolated = np.interp(new_time, valid_time, unwrapped)
    return np.rad2deg(wrap_to_pi(interpolated))


def resample_fixed_rate(dataframe, target_hz=30.0):
    if target_hz <= 0:
        raise ValueError("target_hz must be greater than zero")
    if "Timer" not in dataframe.columns or "Frame" not in dataframe.columns:
        raise ValueError("The input CSV must contain Frame and Timer columns")

    dataframe = dataframe.sort_values("Timer").reset_index(drop=True)
    timer = pd.to_numeric(dataframe["Timer"], errors="coerce").to_numpy(dtype=float)
    if len(timer) < 2 or not np.all(np.isfinite(timer)):
        raise ValueError("Timer must contain at least two finite values")
    if timer[-1] <= timer[0]:
        raise ValueError("Timer must span a positive duration")

    new_time = np.arange(timer[0], timer[-1], 1.0 / target_hz, dtype=float)
    resampled = {"Timer": new_time}

    for column in dataframe.columns:
        if column in {"Frame", "Timer"}:
            continue

        values = pd.to_numeric(dataframe[column], errors="coerce").to_numpy(dtype=float)
        if column in ROTATION_COLUMNS:
            resampled[column] = interpolate_angle_degrees(new_time, timer, values)
            continue

        valid = np.isfinite(values) & np.isfinite(timer)
        if valid.sum() < 2:
            resampled[column] = np.full_like(new_time, np.nan, dtype=float)
        else:
            resampled[column] = np.interp(new_time, timer[valid], values[valid])

    frame_values = pd.to_numeric(dataframe["Frame"], errors="coerce").to_numpy(dtype=float)
    if not np.all(np.isfinite(frame_values)):
        raise ValueError("Frame contains non-numeric values")
    resampled["Frame"] = np.rint(np.interp(new_time, timer, frame_values)).astype(int)

    result = pd.DataFrame(resampled)
    ordered_columns = ["Frame"] + [column for column in result.columns if column != "Frame"]
    return result[ordered_columns]


def resample_folder(input_dir: Path, output_dir: Path, target_hz=30.0):
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    input_files = sorted(input_dir.glob("*.csv"))
    if not input_files:
        raise FileNotFoundError(f"No CSV files found in: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    for input_path in input_files:
        dataframe = pd.read_csv(input_path)
        result = resample_fixed_rate(dataframe, target_hz)
        output_path = output_dir / input_path.name
        result.to_csv(output_path, index=False)
        print(f"[OK] {input_path.name} -> {output_path.name} ({len(result)} rows)")

    print(f"[DONE] Resampled {len(input_files)} files at {target_hz:g} Hz")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--target-hz", type=float, default=30.0)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    resample_folder(
        arguments.input_dir.expanduser(),
        arguments.output_dir.expanduser(),
        arguments.target_hz,
    )
