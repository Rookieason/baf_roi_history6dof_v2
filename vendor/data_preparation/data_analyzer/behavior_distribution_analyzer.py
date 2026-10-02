"""
Count the distribution of behavior types in the dataset.

Fixes vs original:
1) Only scan *_labeled.csv by default (matches the message and typical workflow).
2) Report BOTH:
   - Macro average: average of per-file percentages (your original "整體平均比例" meaning).
   - Micro average: global percentage weighted by total frames across all files (usually what you want).
3) Robustness:
   - Handle empty files (total=0)
   - Normalize label strings (strip, title-case) and treat NaN as "Unlabeled"
   - Warn if unknown labels appear
4) Export includes both per-file stats and a global summary row (optional).
"""

import argparse
import os
from pathlib import Path
import pandas as pd

def parse_args():
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Generate the Chapter 3 behavior distribution figure."
    )
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=script_dir / "results" / "behavior_distribution",
    )
    return parser.parse_args()


arguments = parse_args()
INPUT_FOLDER = arguments.input_dir.expanduser().resolve()
OUTPUT_FOLDER = arguments.output_dir.expanduser().resolve()

if not INPUT_FOLDER.is_dir():
    raise FileNotFoundError(f"Input directory does not exist: {INPUT_FOLDER}")
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
OUTPUT_FILE = OUTPUT_FOLDER / "behavior_stat_summary.csv"

# === 你要統計的 label set（若未來有 Balanced，可直接加進去）===
KNOWN_LABELS = ["Explorer", "Focused", "Unlabeled"]

# === 主程式 ===
records = []
unknown_label_counts = {}

# 只讀取 *_labeled.csv（可避免把中間產物或其他 csv 混進來）
csv_files = sorted([f for f in os.listdir(INPUT_FOLDER) if f.endswith("_labeled.csv")])

if not csv_files:
    print(f"[WARN] 找不到 *_labeled.csv 檔案於 {INPUT_FOLDER}")
    raise SystemExit(0)

global_total = 0
global_counts = {k: 0 for k in KNOWN_LABELS}

for fname in csv_files:
    fpath = os.path.join(INPUT_FOLDER, fname)
    try:
        df = pd.read_csv(fpath)

        if "Label" not in df.columns:
            print(f"[SKIP] {fname}: 沒有 Label 欄位")
            continue

        # Normalize labels:
        # - NaN -> "Unlabeled"
        # - strip spaces
        # - standardize case: "explorer" -> "Explorer"
        labels = (
            df["Label"]
            .fillna("Unlabeled")
            .astype(str)
            .str.strip()
            .str.lower()
            .str.capitalize()
        )

        total = len(labels)
        if total == 0:
            print(f"[SKIP] {fname}: 空檔案 (total=0)")
            continue

        counts = labels.value_counts().to_dict()

        # Track unknown labels (future-proof)
        for k, v in counts.items():
            if k not in KNOWN_LABELS:
                unknown_label_counts[k] = unknown_label_counts.get(k, 0) + int(v)

        explorer = int(counts.get("Explorer", 0))
        focused = int(counts.get("Focused", 0))
        unlabeled = int(counts.get("Unlabeled", 0))

        rec = {
            "File": fname,
            "Total": total,
            "Explorer": explorer,
            "Focused": focused,
            "Unlabeled": unlabeled,
            "Explorer_%": explorer / total * 100.0,
            "Focused_%": focused / total * 100.0,
            "Unlabeled_%": unlabeled / total * 100.0,
        }
        records.append(rec)

        # Global accumulate (micro average)
        global_total += total
        global_counts["Explorer"] += explorer
        global_counts["Focused"] += focused
        global_counts["Unlabeled"] += unlabeled

    except Exception as e:
        print(f"[FAIL] {fname}: {e}")

if not records:
    print("[WARN] 沒有任何可用的統計記錄（可能所有檔案都缺 Label 或為空）")
    raise SystemExit(0)

# === 匯出 per-file 統計結果 ===
summary_df = pd.DataFrame(records)

# 依 Total 由大到小排序，方便看哪些檔案影響最大
summary_df = summary_df.sort_values(by="Total", ascending=False).reset_index(drop=True)

# 加上一列 global summary（micro average）
global_row = {
    "File": "__GLOBAL_MICRO__",
    "Total": global_total,
    "Explorer": global_counts["Explorer"],
    "Focused": global_counts["Focused"],
    "Unlabeled": global_counts["Unlabeled"],
    "Explorer_%": (global_counts["Explorer"] / global_total * 100.0) if global_total else 0.0,
    "Focused_%": (global_counts["Focused"] / global_total * 100.0) if global_total else 0.0,
    "Unlabeled_%": (global_counts["Unlabeled"] / global_total * 100.0) if global_total else 0.0,
}
summary_with_global = pd.concat([summary_df, pd.DataFrame([global_row])], ignore_index=True)

summary_with_global.to_csv(OUTPUT_FILE, index=False, float_format="%.6f")

print("\n=== 統計完成（per-file）===")
print(summary_df[["File", "Explorer_%", "Focused_%", "Unlabeled_%"]])

# === Macro average：每個檔案比例的平均（你原本的寫法）===
macro_mean = summary_df[["Explorer_%", "Focused_%", "Unlabeled_%"]].mean()

print("\n=== 整體平均比例（Macro average：每檔案比例平均）===")
print(macro_mean)

# === Micro average：所有 frame 加總後的比例（通常更合理）===
print("\n=== 整體比例（Micro average：全資料加權）===")
print(pd.Series({
    "Explorer_%": global_row["Explorer_%"],
    "Focused_%": global_row["Focused_%"],
    "Unlabeled_%": global_row["Unlabeled_%"],
}))

if unknown_label_counts:
    print("\n[WARN] 發現未知 Label（未納入 KNOWN_LABELS），已統計如下：")
    for k in sorted(unknown_label_counts.keys()):
        print(f"  - {k}: {unknown_label_counts[k]}")

print(f"\n[OK] CSV 已輸出：{OUTPUT_FILE}")

import matplotlib.pyplot as plt

# === Plot: global micro distribution (PDF) ===
PLOT_PDF = OUTPUT_FOLDER / "behavior_distribution_micro.pdf"
PLOT_PNG = OUTPUT_FOLDER / "behavior_distribution_micro.png"  # optional

# Use micro average (global_row) as the primary distribution
labels_order = ["Explorer", "Focused", "Unlabeled"]
values_pct = [
    global_row["Explorer_%"],
    global_row["Focused_%"],
    global_row["Unlabeled_%"],
]

# plt.figure(figsize=(5.2, 3.2))
# plt.bar(labels_order, values_pct)
# plt.ylabel("Proportion (%)")
# plt.title("Window-level Regime Distribution")
# plt.ylim(0, max(values_pct) * 1.2)

# # annotate bars
# for i, v in enumerate(values_pct):
#     plt.text(i, v, f"{v:.2f}%", ha="center", va="bottom", fontsize=9)

# plt.tight_layout()
# plt.savefig(PLOT_PDF)              # vector figure for LaTeX
# plt.savefig(PLOT_PNG, dpi=200)     # optional quick preview
# plt.close()

fig, ax = plt.subplots(figsize=(5.2, 3.2))

# Explorer、Focused 使用較醒目的顏色，Unlabeled 使用灰色
bar_colors = [
    "tab:orange",  # Explorer
    "tab:blue",    # Focused
    "0.65",        # Unlabeled
]

bars = ax.bar(
    labels_order,
    values_pct,
    color=bar_colors,
    width=0.72,
    zorder=3,
)

ax.set_ylabel("Proportion (%)")
ax.set_ylim(0, max(values_pct) * 1.2)

# 完整四邊外框
for spine in ax.spines.values():
    spine.set_visible(True)
    spine.set_linewidth(0.8)

# 水平與垂直細網格線，置於柱體後方
ax.set_axisbelow(True)
ax.grid(
    True,
    which="major",
    axis="both",
    linestyle="-",
    linewidth=0.4,
    alpha=0.35,
)

# 柱頂百分比
for bar, value in zip(bars, values_pct):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        value,
        f"{value:.2f}%",
        ha="center",
        va="bottom",
        fontsize=9,
    )

fig.subplots_adjust(
    left=0.15,
    right=0.97,
    bottom=0.17,
    top=0.96,
)

fig.savefig(PLOT_PDF)
fig.savefig(PLOT_PNG, dpi=200)
plt.close(fig)

print(f"[OK] Plot saved: {PLOT_PDF}")
