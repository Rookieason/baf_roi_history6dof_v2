"""Generate window-level behavior features and labels for trajectory CSV files."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


WINDOW_SEC = 1.0
ANGLE_UNIT = "deg"
MIN_SAMPLES = 5

HI_HEAD_RATE_THRESH_DEG_S = 60.0
EXPLORER_YAW_Q90_DEG_S = 180.0
EXPLORER_ZERO_CROSS_S = 3.0
T_SPEED_NORM = 0.2
EXPLORER_SCORE_UPPER = 0.55
FOCUSED_SCORE_LOWER = 0.45

OUTPUT_COLUMNS = [
    "Label",
    "head_rate_yaw_q90",
    "yaw_zero_cross_per_s",
    "head_rate_yaw_hi_prop",
    "speed_mean",
    "explorer_score",
    "delta_rx_deg_s",
    "delta_ry_deg_s",
    "delta_rz_deg_s",
]
REQUIRED_COLUMNS = [
    "Timer", "HeadX", "HeadY", "HeadZ", "HeadRX", "HeadRY", "HeadRZ"
]


def to_radians(values):
    return np.deg2rad(values) if ANGLE_UNIT == "deg" else values


def circular_diff(first, second):
    difference = first - second
    return (difference + np.pi) % (2 * np.pi) - np.pi


def weighted_mean(values, weights):
    mask = (~np.isnan(values)) & (~np.isnan(weights))
    if np.sum(mask) == 0 or np.sum(weights[mask]) == 0:
        return np.nan
    return np.sum(values[mask] * weights[mask]) / np.sum(weights[mask])


def zero_crossing_count(values):
    values = np.asarray(values)
    if len(values) < 2:
        return np.nan
    signs = np.sign(values)
    signs[signs == 0] = 1
    return np.sum(signs[1:] * signs[:-1] < 0)


def label_dataframe(dataframe):
    missing = [column for column in REQUIRED_COLUMNS if column not in dataframe.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")

    dataframe = dataframe.copy()
    for column in OUTPUT_COLUMNS:
        if column not in dataframe.columns:
            dataframe[column] = "Unlabeled" if column == "Label" else np.nan

    timer = dataframe["Timer"].to_numpy(dtype=float)
    position = dataframe[["HeadX", "HeadY", "HeadZ"]].to_numpy(dtype=float)
    rotation = dataframe[["HeadRX", "HeadRY", "HeadRZ"]].to_numpy(dtype=float)
    rotation_radians = to_radians(rotation)

    high_rate_threshold = np.deg2rad(HI_HEAD_RATE_THRESH_DEG_S)
    explorer_yaw_q90 = np.deg2rad(EXPLORER_YAW_Q90_DEG_S)

    for index in range(len(dataframe)):
        current_time = timer[index]
        window_mask = (
            (timer >= current_time - WINDOW_SEC / 2)
            & (timer <= current_time + WINDOW_SEC / 2)
        )
        indices = np.where(window_mask)[0]
        if len(indices) < MIN_SAMPLES:
            continue

        time_segment = timer[indices]
        position_segment = position[indices]
        rotation_segment = rotation_radians[indices]
        time_differences = np.diff(time_segment)
        time_differences[time_differences <= 0] = np.nan
        if np.any(np.isnan(time_differences)) or len(time_differences) == 0:
            continue

        duration = time_segment[-1] - time_segment[0]
        if duration <= 0:
            continue

        displacement = np.linalg.norm(np.diff(position_segment, axis=0), axis=1)
        speed_mean = weighted_mean(displacement / time_differences, time_differences)

        yaw_difference = circular_diff(
            rotation_segment[1:, 1], rotation_segment[:-1, 1]
        )
        yaw_rate = np.abs(yaw_difference) / time_differences
        signed_yaw_rate = yaw_difference / time_differences
        head_rate_yaw_q90 = float(np.nanpercentile(yaw_rate, 90))
        head_rate_yaw_hi_prop = float(np.mean(yaw_rate > high_rate_threshold))
        switches = zero_crossing_count(signed_yaw_rate)
        yaw_zero_cross_per_s = switches / duration

        rotation_difference = np.diff(rotation_segment, axis=0)
        angular_velocity = np.abs(rotation_difference) / time_differences[:, None]
        delta_rx, delta_ry, delta_rz = np.mean(angular_velocity, axis=0)

        rate_component = np.clip(head_rate_yaw_q90 / explorer_yaw_q90, 0, 1)
        crossing_component = np.clip(
            yaw_zero_cross_per_s / EXPLORER_ZERO_CROSS_S, 0, 1
        )
        high_rate_component = np.clip(head_rate_yaw_hi_prop / 0.5, 0, 1)
        rotation_score = (
            0.5 * rate_component
            + 0.3 * crossing_component
            + 0.2 * high_rate_component
        )
        translation_score = np.clip(speed_mean / T_SPEED_NORM, 0, 1)
        combined_score = 0.6 * rotation_score + 0.4 * translation_score
        explorer_score = 1.0 / (1.0 + np.exp(-6 * (combined_score - 0.5)))

        if explorer_score >= EXPLORER_SCORE_UPPER:
            label = "Explorer"
        elif explorer_score <= FOCUSED_SCORE_LOWER:
            label = "Focused"
        else:
            label = "Unlabeled"

        dataframe.at[index, "speed_mean"] = speed_mean
        dataframe.at[index, "head_rate_yaw_q90"] = head_rate_yaw_q90
        dataframe.at[index, "yaw_zero_cross_per_s"] = yaw_zero_cross_per_s
        dataframe.at[index, "head_rate_yaw_hi_prop"] = head_rate_yaw_hi_prop
        dataframe.at[index, "delta_rx_deg_s"] = np.rad2deg(delta_rx)
        dataframe.at[index, "delta_ry_deg_s"] = np.rad2deg(delta_ry)
        dataframe.at[index, "delta_rz_deg_s"] = np.rad2deg(delta_rz)
        dataframe.at[index, "explorer_score"] = explorer_score
        dataframe.at[index, "Label"] = label

    return dataframe


def label_folder(input_dir: Path, output_dir: Path):
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    input_files = sorted(input_dir.glob("*.csv"))
    if not input_files:
        raise FileNotFoundError(f"No CSV files found in: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    failures = []
    for input_path in input_files:
        try:
            dataframe = pd.read_csv(input_path)
            labeled = label_dataframe(dataframe)
            output_path = output_dir / f"{input_path.stem}_labeled.csv"
            labeled.to_csv(output_path, index=False, float_format="%.6f")
            counts = labeled["Label"].value_counts().to_dict()
            print(f"[OK] {input_path.name} -> {output_path.name}: {counts}")
        except Exception as error:
            failures.append((input_path.name, str(error)))
            print(f"[FAIL] {input_path.name}: {error}")

    if failures:
        details = "; ".join(f"{name}: {reason}" for name, reason in failures)
        raise RuntimeError(f"Label generation failed for {len(failures)} files: {details}")
    print(f"[DONE] Labeled {len(input_files)} files")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    label_folder(arguments.input_dir.expanduser(), arguments.output_dir.expanduser())
