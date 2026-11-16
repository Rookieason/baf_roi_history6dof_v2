# CSI-assisted ROI Selection：四種方法、共同實驗入口

研究問題：在相同 HMD history、資料切分、future horizons 與 ROI 幾何下，當下已可取得的 CSI heatmap，能否提供超出 HMD history 的資訊，讓未來 partition selection 更準確？

本分支為 `feature/csi-heatmap-fusion`。原 baseline 文件保存在 [README_BASELINE.md](README_BASELINE.md)。BAF 與 Motion Transformer 的模型仍在各自套件；此次統一了實驗入口與評估，並修正 BAF 不規則 decision gaps 的 observed α reset。原作者 `vendor/` 與 extracted model/fusion operators 不變。

## 四種方法

| `--method` | 程式位置 | HMD／heatmap 的角色 |
|---|---|---|
| `baf_roi` | `baf_roi/` | 原 BAF-Pose：Confidence + Focused + Explorer |
| `motion_transformer_roi` | `motion_transformer_roi/` | HMD history → Motion Transformer → future pose |
| `baf_roi_heatmap` | `baf_roi_heatmap/` | 預設 residual：HMD／heatmap encoder 修正凍結 BAF 的 future pose；保留 `expert_gate` 比較組 |
| `baf_roi_heatmap_alpha` | `baf_roi_heatmap_alpha/` | 8 個 heatmap 統計量 → 161 參數小 gate；只修正 BAF 每個 autoregressive step 的 α |

`roi_experiments/` 負責 method registry、執行順序、共同 evaluator、摘要及圖表，不放第五種模型。兩個 heatmap 方法共用原 BAF experts、因果 RF 資料介面及 ROI 幾何。新的 alpha 方法不是原 `expert_gate` 的改名：它確實改變每一步 fusion 和之後的預測 history。

## 一個入口：不指定 method 就全部跑

```bash
python -m pip install -r requirements-heatmap.txt

python scripts/run_experiments.py \
  --config configs/recorded_20261031.json \
  --output runs/all_01
```

只跑一種或指定多種：

```bash
python scripts/run_experiments.py \
  --config configs/recorded_20261031.json \
  --method baf_roi_heatmap_alpha \
  --output runs/alpha_01

python scripts/run_experiments.py \
  --config configs/recorded_20261031.json \
  --method baf_roi motion_transformer_roi \
  --output runs/baselines_01
```

預設依設定跑 50–1000 ms、每 50 ms 一個 horizon；四種方法共用同一份 cache／split、同一份 frozen BAF，以及同一組所有 horizons 都有 GT 的 origins。BAF 是兩個 heatmap 方法的依賴，即使只選 heatmap method，也需要訓練或重用 BAF。只選 HMD baselines 時不要求 RF 檔案；MT-only 不會額外訓練 BAF。

機器路徑已寫在既有 shell script 開頭時，也可以直接：

```bash
bash scripts/train_csi_20261031.sh
bash scripts/train_csi_20261031.sh --method baf_roi_heatmap_alpha
```

這兩個命令是替代用法，會各啟動一個背景實驗。腳本保留原 Python／`/mnt/codes` 路徑與 log 管理；在不同機器上先調整該檔開頭的設定。

## 重用實驗與結果

```bash
# 重用已準備的 cache、RF 及 frozen BAF，只訓練新的 alpha method。
python scripts/run_experiments.py \
  --prepared runs/all_01/prepared \
  --method baf_roi_heatmap_alpha \
  --output runs/alpha_reuse_01

# 重評同一 run 的 checkpoints，不重新訓練。
python scripts/run_experiments.py evaluate \
  --run runs/all_01 --output runs/eval_02

# 查閱將執行的 methods、資料來源與依賴，不訓練、不寫檔。
python scripts/run_experiments.py \
  --config configs/recorded_20261031.json \
  --output runs/inspect_01 --dry-run
```

主要輸出：`experiment.json`、`summary.csv`、`summary.json`、`models/<method>/`、`evaluations/test/<method>/per_origin.csv` 與 `report.json`。詳細 checkpoint 重用、通用資料來源與繪圖命令見 [共同入口使用說明](docs/README_EXPERIMENTS.md)。

## Heatmap 是否預先產生？

**目前 `recorded_20261031` 的訓練流程會讀取已產生的數值 heatmaps，不在每個 batch 重算 MUSIC。** 來源是設定中 dataset root 下的：

```text
intermediates/<session>/heatmaps/<session>_heatmaps.npz
```

`recorded.prepare_recorded()` 核對既有 map 與時間依賴，再轉成訓練用 NPZ／RF manifest。輸入是 AoA–ToF dB 數值，並非 PNG。現有原始 maps 90×130；因果視窗處理後為 `8×3×24×24`，分別保留 level、背景差分、時間差分。缺測或過期時回到原 BAF。

若資料來源只有 raw CSI，也保留獨立的生成入口：

```bash
python -m baf_roi_heatmap generate \
  --raw data/radio_record.npz \
  --config configs/heatmap_generator.example.json \
  --output data/maps/record_01.npz
```

raw CSI 必須有實測 timestamps、availability 與正確 subcarrier frequencies。生成、資料對齊、模型訓練是分開的階段；參見 [資料契約](docs/README_HEATMAP_ARCHITECTURE.md) 與 [真實 recording profile](docs/TRAIN_CSI_20261031.md)。

## 共同 evaluation

四個 methods 都經 `roi_experiments.evaluation`。舊 `baf_roi_heatmap.evaluation` 保留 compatibility imports。無 RF 的 standalone BAF／MT 也共用同一個 metric／report engine。

- 主表延續先前 PPT：**逐 origin Mean IoU**、**pooled Micro F1**，另列 precision、coverage/recall、FP/FN、selected count 與 pose error。
- 保留 all origins 及 RF-available subset；同一 RF 子集的 baseline 也重新評估。
- 同時輸出 pooled、original-recording macro、scenario macro。原始錄製檔因 gaps 被切成多段時，不會得到多個 recording 權重。
- 用完整 protocol、origin IDs、targets、scene／partition order 與 RF cohort hashes 檢查公平性，不只比較 sample count。
- 預設 prediction-side margins `0 5 10 20`，GT FoV 不變；margin 0 是主要比較，另外提供 coverage–selected-count 曲線。
- **SSIM 為 `not_evaluated`**：需要同一真實 future pose／content／renderer 的選取畫面與全內容 reference。Coverage 不能取代 SSIM，selected count 也不是實測 bytes。

## 文件

| 文件 | 內容 |
|---|---|
| [共同入口與操作](docs/README_EXPERIMENTS.md) | `--method`、相依模型、cache/checkpoint 重用、輸出、繪圖 |
| [原 heatmap 演算法](docs/README_HEATMAP_ALGORITHM.md) | AoA–ToF、因果對齊、CNN／GRU、residual、terminal expert gate |
| [新的 alpha-only 演算法](docs/README_HEATMAP_ALPHA.md) | 簡單 RF features、161 參數 gate、真正的逐步 α fusion、訓練與 fallback |
| [資料與模組架構](docs/README_HEATMAP_ARCHITECTURE.md) | raw／map NPZ、manifest、時間與幾何契約 |
| [評估定義與研究解讀](docs/README_HEATMAP_EVALUATION.md) | PPT 目標、matched comparison、aggregation、SSIM 邊界 |
| [驗證紀錄](docs/HEATMAP_VALIDATION.md) | 單元／回歸測試、合成整合、尚未實驗證明的項目 |
| [原始來源對照](docs/HEATMAP_SOURCES.md) | 力行論文、投影片、原 MUSIC 程式來源 |

## 軟體驗證

```bash
python -m baf_roi audit
python -m pytest -q
python scripts/smoke_heatmap.py --output runs/smoke_heatmap_01
```

Smoke 使用 synthetic HMD／CSI。軟體通過、真實 held-out ROI 提升、維持 rendered visual quality、完整 pipeline 30 Hz 是不同的結論；目前沒有由此次修改產生新的真實資料訓練結果。
