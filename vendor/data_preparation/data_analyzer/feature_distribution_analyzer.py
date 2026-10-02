# feature_distribution_analyzer_v3_thesis.py
# Strategy A: internal computation in rad / rad/s, export plots & CSV in deg / deg/s.
# Add stride + thesis-grade figures (PDF) + consistent styling and axis units.

import argparse
import os
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import entropy

def parse_args():
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Generate the Chapter 3 feature distribution figures."
    )
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=script_dir / "results" / "feature_distribution",
    )
    return parser.parse_args()


arguments = parse_args()
INPUT_FOLDER = arguments.input_dir.expanduser().resolve()
OUTPUT_FOLDER = arguments.output_dir.expanduser().resolve()

if not INPUT_FOLDER.is_dir():
    raise FileNotFoundError(f"Input directory does not exist: {INPUT_FOLDER}")

WINDOW_SEC = 1.0
STRIDE_SEC = 0.5              # new: stride for window centers (seconds)
ANGLE_UNIT = "deg"
MIN_SAMPLES = 5

FIX_RATE_THRESH_DEG_S = 20.0
SACCADE_THRESH_DEG_S = 60.0
HI_HEAD_RATE_THRESH_DEG_S = 60.0
BINS_ENTROPY = 18
MAX_COARSE_LAG_STEPS = 1

# Plot controls (thesis-grade)
FIG_W_IN = 3.4                # single-column width (inches)
FIG_H_IN = 2.6
HIST_BINS = 60
Q_LOW = 0.5                   # quantile clipping for histogram range
Q_HIGH = 99.5

OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
FIG_PDF_DIR = OUTPUT_FOLDER
FIG_PNG_DIR = OUTPUT_FOLDER

RAD2DEG = 180.0 / np.pi

# Use consistent, thesis-friendly matplotlib styles
plt.rcParams.update({
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "axes.linewidth": 0.8,
    "grid.linewidth": 0.6,
    "pdf.fonttype": 42,   # embed TrueType fonts
    "ps.fonttype": 42,
    "savefig.bbox": "tight",
})

# ======= Utilities =======
def to_radians(x):
    return np.deg2rad(x) if ANGLE_UNIT == "deg" else x

def circular_diff(a, b):
    d = a - b
    return (d + np.pi) % (2 * np.pi) - np.pi

def weighted_mean(values, weights):
    mask = (~np.isnan(values)) & (~np.isnan(weights))
    if np.sum(mask) == 0 or np.sum(weights[mask]) == 0:
        return np.nan
    return np.sum(values[mask] * weights[mask]) / np.sum(weights[mask])

def discretized_entropy_weighted(x_rad, w=None, bins=BINS_ENTROPY):
    # entropy() default log base = e -> nats
    x_rad = x_rad[~np.isnan(x_rad)]
    if len(x_rad) < 2 or np.allclose(x_rad, x_rad[0]):
        return 0.0
    hist, _ = np.histogram(x_rad, bins=bins, range=(-np.pi, np.pi), weights=w)
    total = np.sum(hist)
    if total <= 0:
        return 0.0
    p = np.clip(hist / total, 1e-12, 1.0)
    return float(entropy(p))

def safe_corr(a, b):
    if len(a) != len(b) or len(a) < 3:
        return np.nan
    a = np.asarray(a); b = np.asarray(b)
    if np.all(np.isnan(a)) or np.all(np.isnan(b)):
        return np.nan
    a = a - np.nanmean(a); b = b - np.nanmean(b)
    sa = np.nanstd(a); sb = np.nanstd(b)
    if sa < 1e-8 or sb < 1e-8:
        return np.nan
    return float(np.nanmean((a/sa) * (b/sb)))

def zero_crossing_count(x):
    x = np.asarray(x)
    if len(x) < 2:
        return np.nan
    s = np.sign(x)
    s[s == 0] = 1
    return int(np.sum(s[1:] * s[:-1] < 0))

# ======= Feature extraction =======
feature_records = []

for fname in os.listdir(INPUT_FOLDER):
    if not fname.endswith(".csv"):
        continue
    fpath = os.path.join(INPUT_FOLDER, fname)
    try:
        df = pd.read_csv(fpath)
        if "Timer" not in df.columns:
            print(f"[SKIP] {fname}: missing Timer")
            continue

        timer = df["Timer"].to_numpy()
        if len(timer) < MIN_SAMPLES:
            continue

        rot = df[["HeadRX", "HeadRY", "HeadRZ"]].to_numpy()
        rot_rad = to_radians(rot)

        eye_L = df["LEyeRY"].to_numpy() if "LEyeRY" in df.columns else np.full_like(timer, np.nan)
        eye_R = df["REyeRY"].to_numpy() if "REyeRY" in df.columns else np.full_like(timer, np.nan)
        conf_L = df["LEyeConf"].to_numpy() if "LEyeConf" in df.columns else np.zeros_like(timer)
        conf_R = df["REyeConf"].to_numpy() if "REyeConf" in df.columns else np.zeros_like(timer)

        conf_all = (conf_L + conf_R) / 2.0
        is_all_zero_file = np.nanmax(conf_all) == 0.0
        is_constant_file = (np.nanstd(conf_all) < 1e-6) and (np.nanmax(conf_all) > 0.0)

        t_min = float(np.nanmin(timer))
        t_max = float(np.nanmax(timer))
        t_start = t_min + WINDOW_SEC / 2.0
        t_end = t_max - WINDOW_SEC / 2.0
        if t_end <= t_start:
            continue

        # new: stride-based window centers
        centers = np.arange(t_start, t_end + 1e-9, STRIDE_SEC)

        for t_c in centers:
            mask = (timer >= t_c - WINDOW_SEC/2) & (timer <= t_c + WINDOW_SEC/2)
            idx = np.where(mask)[0]
            if len(idx) < MIN_SAMPLES:
                continue

            t_seg = timer[idx]
            rot_seg = rot_rad[idx]

            dt = np.diff(t_seg)
            dt[dt <= 0] = np.nan
            if np.any(np.isnan(dt)) or len(dt) == 0:
                continue

            duration = t_seg[-1] - t_seg[0]
            if duration <= 0:
                continue

            has_pos = all(c in df.columns for c in ["HeadX", "HeadY", "HeadZ"])
            if has_pos:
                pos = df[["HeadX","HeadY","HeadZ"]].to_numpy(dtype=float)
                pos_seg = pos[idx]
                dpos = np.diff(pos_seg, axis=0)
                head_speed_vec = np.linalg.norm(dpos, axis=1) / dt   # m/s
                head_speed_mean = weighted_mean(head_speed_vec, dt)  # m/s
            else:
                head_speed_mean = np.nan


            conf_seg = (conf_L[idx] + conf_R[idx]) / 2.0
            is_all_zero = np.nanmax(conf_seg) == 0.0 or is_all_zero_file
            is_constant = (np.nanstd(conf_seg) < 1e-6 and np.nanmax(conf_seg) > 0.0) or is_constant_file

            drot_vec = circular_diff(rot_seg[1:], rot_seg[:-1])
            head_rate_vec = np.linalg.norm(drot_vec, axis=1) / dt
            head_rate_yaw = np.abs(circular_diff(rot_seg[1:, 1], rot_seg[:-1, 1])) / dt
            head_rate_yaw_signed = circular_diff(rot_seg[1:, 1], rot_seg[:-1, 1]) / dt

            head_yaw = rot_seg[:, 1]
            w_time = np.append(dt, dt[-1])
            head_yaw_entropy = discretized_entropy_weighted(head_yaw, w=w_time, bins=BINS_ENTROPY)
            head_ang_speed_mean = weighted_mean(head_rate_vec, dt)
            ang_speed_mean = head_ang_speed_mean

            if not is_all_zero:
                denom = conf_L[idx] + conf_R[idx]
                numer = eye_L[idx] * conf_L[idx] + eye_R[idx] * conf_R[idx]
                eye_fused_deg_seg = np.divide(
                    numer.astype(float), denom.astype(float),
                    out=np.full(numer.shape, np.nan, dtype=float),
                    where=(denom > 0)
                )
                eye_yaw = to_radians(eye_fused_deg_seg)
                w_combined = w_time if is_constant else w_time * conf_seg

                gaze_entropy = discretized_entropy_weighted(eye_yaw, w=w_combined, bins=BINS_ENTROPY)
                eye_rate = np.abs(circular_diff(eye_yaw[1:], eye_yaw[:-1])) / dt

                dt_weighted = dt if is_constant else dt * conf_seg[:-1]
                numer_fix = np.sum(dt_weighted[eye_rate < np.deg2rad(FIX_RATE_THRESH_DEG_S)])
                denom_fix = np.sum(dt_weighted)
                fixation_ratio = np.nan if denom_fix <= 0 or np.isnan(denom_fix) else (numer_fix / denom_fix)
            else:
                gaze_entropy = np.nan
                fixation_ratio = np.nan
                eye_yaw = None
                eye_rate = None

            def q(x, p):
                return float(np.nanpercentile(x, p)) if x is not None and len(x) > 0 else np.nan

            head_rate_yaw_q50 = q(head_rate_yaw, 50)
            head_rate_yaw_q90 = q(head_rate_yaw, 90)

            head_rate_yaw_hi_prop = np.nan
            if head_rate_yaw is not None and len(head_rate_yaw) > 0:
                head_rate_yaw_hi_prop = float(np.mean(head_rate_yaw > np.deg2rad(HI_HEAD_RATE_THRESH_DEG_S)))

            eye_rate_q50 = q(eye_rate, 50)
            eye_rate_q90 = q(eye_rate, 90)

            eye_saccade_rate = np.nan
            eye_hi_prop = np.nan
            if eye_rate is not None and len(eye_rate) > 0:
                eye_hi = eye_rate > np.deg2rad(SACCADE_THRESH_DEG_S)
                if np.any(eye_hi):
                    runs = 1
                    for k in range(1, len(eye_hi)):
                        if eye_hi[k] and not eye_hi[k-1]:
                            runs += 1
                    eye_saccade_rate = runs / duration
                else:
                    eye_saccade_rate = 0.0
                eye_hi_prop = float(np.mean(eye_hi))

            yaw_zero_cross_per_s = np.nan
            if head_rate_yaw_signed is not None and len(head_rate_yaw_signed) > 1:
                switches = zero_crossing_count(head_rate_yaw_signed)
                yaw_zero_cross_per_s = switches / duration

            head_eye_corr = np.nan
            head_eye_lag_s = np.nan
            if eye_rate is not None and len(eye_rate) == len(head_rate_yaw):
                cands = []
                for lag in range(-MAX_COARSE_LAG_STEPS, MAX_COARSE_LAG_STEPS + 1):
                    if lag == 0:
                        h = head_rate_yaw
                        e = eye_rate
                    elif lag < 0:
                        h = head_rate_yaw[:lag]
                        e = eye_rate[-lag:]
                    else:
                        h = head_rate_yaw[lag:]
                        e = eye_rate[:-lag]
                    cands.append((safe_corr(h, e), lag))
                best = max(cands, key=lambda x: (np.nan_to_num(x[0], nan=-np.inf)))
                head_eye_corr = best[0]
                mean_dt = float(np.nanmean(dt)) if np.all(~np.isnan(dt)) else (duration / max(len(dt), 1))
                head_eye_lag_s = best[1] * mean_dt

            head_eye_gap_mean = np.nan
            head_eye_gap_std = np.nan
            if eye_yaw is not None and len(eye_yaw) == len(head_yaw):
                gap = np.abs(circular_diff(head_yaw, eye_yaw))
                head_eye_gap_mean = float(np.nanmean(gap))
                head_eye_gap_std = float(np.nanstd(gap))

            feature_records.append({
                "File": fname,
                "t_center": float(t_c),

                "gaze_entropy": gaze_entropy,                 # nats
                "fixation_ratio": fixation_ratio,             # ratio
                "ang_speed_mean": ang_speed_mean,             # rad/s
                "head_yaw_entropy": head_yaw_entropy,         # nats
                "head_ang_speed_mean": head_ang_speed_mean,   # rad/s

                "head_rate_yaw_q50": head_rate_yaw_q50,       # rad/s
                "head_rate_yaw_q90": head_rate_yaw_q90,       # rad/s
                "head_rate_yaw_hi_prop": head_rate_yaw_hi_prop,  # ratio

                "eye_rate_q50": eye_rate_q50,                 # rad/s
                "eye_rate_q90": eye_rate_q90,                 # rad/s
                "eye_saccade_rate": eye_saccade_rate,         # 1/s
                "eye_hi_prop": eye_hi_prop,                   # ratio

                "yaw_zero_cross_per_s": yaw_zero_cross_per_s, # 1/s
                "head_eye_corr": head_eye_corr,               # corr
                "head_eye_lag_s": head_eye_lag_s,             # s
                "head_eye_gap_mean": head_eye_gap_mean,       # rad
                "head_eye_gap_std": head_eye_gap_std,         # rad

                "head_speed_mean": head_speed_mean,           # m/s
            })

    except Exception as e:
        print(f"[FAIL] {fname}: {e}")

df_raw = pd.DataFrame(feature_records)
df_raw["has_eye_data"] = ~df_raw["gaze_entropy"].isna()

raw_path = os.path.join(OUTPUT_FOLDER, "feature_values_raw_rad.csv")
df_raw.to_csv(raw_path, index=False)

# Strategy A export (deg / deg/s)
FEATURE_EXPORT = {
    "gaze_entropy":          ("nat", 1.0),
    "head_yaw_entropy":      ("nat", 1.0),

    "fixation_ratio":        ("ratio", 1.0),
    "head_rate_yaw_hi_prop": ("ratio", 1.0),
    "eye_hi_prop":           ("ratio", 1.0),

    "ang_speed_mean":        ("deg/s", RAD2DEG),
    "head_ang_speed_mean":   ("deg/s", RAD2DEG),
    "head_rate_yaw_q50":     ("deg/s", RAD2DEG),
    "head_rate_yaw_q90":     ("deg/s", RAD2DEG),
    "eye_rate_q50":          ("deg/s", RAD2DEG),
    "eye_rate_q90":          ("deg/s", RAD2DEG),

    "head_eye_gap_mean":     ("deg", RAD2DEG),
    "head_eye_gap_std":      ("deg", RAD2DEG),

    "eye_saccade_rate":      ("1/s", 1.0),
    "yaw_zero_cross_per_s":  ("1/s", 1.0),

    "head_eye_lag_s":        ("s", 1.0),
    "head_eye_corr":         ("corr", 1.0),

    "head_speed_mean":       ("m/s", 1.0),
}

df_out = df_raw.copy()
for col, (_, scale) in FEATURE_EXPORT.items():
    if col in df_out.columns and scale != 1.0:
        df_out[col] = df_out[col] * scale

out_path = os.path.join(OUTPUT_FOLDER, "feature_values.csv")
df_out.to_csv(out_path, index=False)

PLOT_COLS = [
    "gaze_entropy", 
    "fixation_ratio", 
    "head_yaw_entropy", 
    "head_rate_yaw_q90", 
    "head_rate_yaw_hi_prop",
    "eye_saccade_rate", 
    "yaw_zero_cross_per_s", 
    "head_speed_mean",
    # "ang_speed_mean",
    # "head_ang_speed_mean",
    # "head_rate_yaw_q50", 
    # "eye_rate_q50", 
    # "eye_rate_q90", 
    # "eye_hi_prop",
    # "head_eye_corr", 
    # "head_eye_lag_s",
    # "head_eye_gap_mean", 
    # "head_eye_gap_std",
]

AXIS_LABELS = {
    "gaze_entropy": "Gaze entropy (nat)",
    "head_yaw_entropy": "Yaw entropy (nat)",
    "fixation_ratio": "Fixation stability (ratio)",
    "head_rate_yaw_q90": "Yaw-rate intensity (deg/s)",
    "head_rate_yaw_hi_prop": "High-rate yaw proportion (ratio)",
    "eye_saccade_rate": "Saccade rate (1/s)",
    "yaw_zero_cross_per_s": "Yaw direction reversal rate (1/s)",
    "head_speed_mean": "Translational speed (m/s)",
    "ang_speed_mean": "Angular speed (deg/s)",
    "head_ang_speed_mean": "Head angular speed (deg/s)",
    "head_rate_yaw_q50": "Median yaw rate (deg/s)",
    "eye_rate_q50": "Median gaze rate (deg/s)",
    "eye_rate_q90": "Gaze-rate intensity (deg/s)",
    "eye_hi_prop": "High-rate gaze proportion (ratio)",
    "head_eye_corr": "Head-gaze correlation",
    "head_eye_lag_s": "Head-gaze lag (s)",
    "head_eye_gap_mean": "Mean head-gaze gap (deg)",
    "head_eye_gap_std": "Head-gaze gap standard deviation (deg)",
}

EVENT_RATE_COLS = {
    "eye_saccade_rate",
    "yaw_zero_cross_per_s",
}


def axis_label(col: str) -> str:
    """Return a readable thesis label instead of the internal column name."""
    if col in AXIS_LABELS:
        return AXIS_LABELS[col]
    if col in FEATURE_EXPORT:
        unit = FEATURE_EXPORT[col][0]
        return f"{col.replace('_', ' ').title()} ({unit})"
    return col.replace("_", " ").title()


def event_rate_histogram_bins(values: np.ndarray) -> np.ndarray:
    """Create unit-width bins centered on integer event rates.

    Event rates are divided by each window's actual sampled duration, so values
    generated by the same event count may differ slightly around an integer.
    Centering one bin on each integer merges these sampling-induced neighbors
    without rounding or modifying the underlying feature values.
    """
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return np.array([-0.5, 0.5])

    max_center = max(0, int(np.ceil(float(finite.max()))))
    return np.arange(-0.5, max_center + 1.5, 1.0)

def fixed_xlim(col: str):
    # Ensure consistent, publication-friendly axes for bounded metrics
    if col in ["fixation_ratio", "head_rate_yaw_hi_prop", "eye_hi_prop"]:
        return (0.0, 1.0)
    if col == "head_eye_corr":
        return (-1.0, 1.0)
    return None

summary = {
    "config": {
        "WINDOW_SEC": WINDOW_SEC,
        "STRIDE_SEC": STRIDE_SEC,
        "HIST_BINS": HIST_BINS,
        "Q_LOW": Q_LOW,
        "Q_HIGH": Q_HIGH,
        "ANGLE_UNIT_RAW": ANGLE_UNIT,
        "export_strategy": "A (deg/deg/s export)",
        "entropy_unit": "nat (scipy.stats.entropy default)",
    },
    "features": {}
}

for col in PLOT_COLS:
    if col not in df_out.columns:
        continue

    valid = df_out[col].dropna().to_numpy()
    if len(valid) < 50:
        continue

    # Determine histogram range (bounded metrics use fixed limits; others use quantiles)
    xlim = fixed_xlim(col)
    clipped_low = 0
    clipped_high = 0
    if xlim is None:
        lo = np.nanpercentile(valid, Q_LOW)
        hi = np.nanpercentile(valid, Q_HIGH)
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            lo = float(np.nanmin(valid))
            hi = float(np.nanmax(valid))
        clipped_low = int(np.sum(valid < lo))
        clipped_high = int(np.sum(valid > hi))
        plot_vals = valid[(valid >= lo) & (valid <= hi)]
        hist_range = (lo, hi)
    else:
        lo, hi = xlim
        clipped_low = int(np.sum(valid < lo))
        clipped_high = int(np.sum(valid > hi))
        plot_vals = valid[(valid >= lo) & (valid <= hi)]
        hist_range = (lo, hi)

    if len(plot_vals) < 30:
        continue

    # Save summary stats (on full valid values)
    summary["features"][col] = {
        "unit": FEATURE_EXPORT.get(col, ("", 1.0))[0],
        "count": int(len(valid)),
        "mean": float(np.nanmean(valid)),
        "std": float(np.nanstd(valid)),
        "median": float(np.nanmedian(valid)),
        "q90": float(np.nanpercentile(valid, 90)),
        "q99": float(np.nanpercentile(valid, 99)),
        "hist_range": [float(hist_range[0]), float(hist_range[1])],
        "clipped_low_count": clipped_low,
        "clipped_high_count": clipped_high,
        "clipped_ratio": float((clipped_low + clipped_high) / max(len(valid), 1)),
    }

    # Plot
    plt.figure(figsize=(FIG_W_IN, FIG_H_IN))

    if col in EVENT_RATE_COLS:
        # Merge nearby rates produced by small variations in the actual sampled
        # window duration. The feature values remain unchanged.
        rate_bins = event_rate_histogram_bins(plot_vals)
        plt.hist(
            plot_vals,
            bins=rate_bins,
            density=True,
            alpha=0.85,
            rwidth=0.2,
        )
        plt.xlim(rate_bins[0], rate_bins[-1])
    else:
        plt.hist(
            plot_vals,
            bins=HIST_BINS,
            range=hist_range,
            density=True,
            alpha=0.85,
        )

    plt.xlabel(axis_label(col))
    plt.ylabel("Probability Density")
    plt.grid(alpha=0.3)
    plt.tight_layout()

    pdf_path = os.path.join(FIG_PDF_DIR, f"{col}_dist.pdf")
    png_path = os.path.join(FIG_PNG_DIR, f"{col}_dist.png")
    plt.savefig(pdf_path, format="pdf")
    plt.savefig(png_path, dpi=300)
    plt.close()

# Write summary
summary_path = os.path.join(OUTPUT_FOLDER, "summary.json")
with open(summary_path, "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

print("Done.")
print(f"CSV (raw rad): {raw_path}")
print(f"CSV (export deg): {out_path}")
print(f"Summary: {summary_path}")
print(f"Figures (PDF): {FIG_PDF_DIR}")
print(f"Figures (PNG): {FIG_PNG_DIR}")
