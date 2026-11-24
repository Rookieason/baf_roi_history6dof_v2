# CSI-assisted BAF：內部特徵 Residual

研究問題：在相同 HMD history、資料切分、future horizons 與 ROI 幾何下，當下已可取得的 CSI heatmap，能否提供超出 HMD history 的資訊，讓未來 partition selection 更準確？

本分支為 **`feature/csi-internal-residual`**，從 `feature/csi-heatmap-fusion` 的 `e34a836` 建立。主要方法為 **`baf_roi_heatmap_residual`**：CSI heatmap 在 Focused／Explorer 產生下一步姿態之前，補充兩個 expert 的內部特徵。Residual 的定義為 **BAF latent + CSI-conditioned latent residual**。原本的 post-forecast adapter 保留作歷史比較，不代表本分支的新方法。

每一步都先計算 HMD latent → 加入 heatmap residual → 使用原 BAF translation／rotation heads → 原 α fusion → 將預測姿態放入下一步 history。有效 CSI 的線上路徑只執行一次這樣的 rollout；不先完成另一條 BAF forecast，也沒有在輸出姿態上加 `Δp`／`Δrotation`。

完整研究動機、公式、訓練方式、控制實驗與限制見 **[內部 residual 設計與操作說明](docs/README_INTERNAL_RESIDUAL.md)**。原 baseline 文件仍在 [README_BASELINE.md](README_BASELINE.md)。

## 先跑本分支的主要比較

```bash
bash scripts/train_csi_internal_residual.sh
```

這會依照既有 `recorded_20261031` 設定，在背景執行 **BAF、Motion Transformer、內部 CSI residual**，共用同一份資料與 BAF checkpoint。機器路徑沿用 `scripts/train_csi_20261031.sh`。也可直接使用共同 Python 入口：

```bash
python scripts/run_experiments.py \
  --config configs/recorded_20261031.json \
  --method baf_roi motion_transformer_roi baf_roi_heatmap_residual \
  --output runs/internal_residual_01
```

已經有相容的 `prepared/` 時，可直接重用已訓練 BAF 與資料，只訓練新的內部 residual：

```bash
python scripts/run_experiments.py \
  --prepared /absolute/path/to/previous_run/prepared \
  --method baf_roi baf_roi_heatmap_residual \
  --output runs/internal_residual_reuse_01
```

新 checkpoint 是 `models/baf_roi_heatmap_residual/residual.pt`。舊的 `adapter.pt`、`alpha.pt` 不能當成這個模型的權重；既有 BAF checkpoint 可以重用。軟體驗證使用合成資料，尚未取得本方法在真實 held-out recordings 上的訓練結果。

## 五種方法與比較組

| `--method` | 程式位置 | HMD／heatmap 的角色 |
|---|---|---|
| `baf_roi` | `baf_roi/` | 原 BAF-Pose：Confidence + Focused + Explorer |
| `motion_transformer_roi` | `motion_transformer_roi/` | HMD history → Motion Transformer → future pose |
| `baf_roi_heatmap` | `baf_roi_heatmap/` | 歷史比較：HMD／heatmap encoder 修正完成的 BAF future pose；保留 terminal `expert_gate` |
| `baf_roi_heatmap_alpha` | `baf_roi_heatmap_alpha/` | 8 個 heatmap 統計量 → 161 參數小 gate；只修正 BAF 每個 autoregressive step 的 α |
| **`baf_roi_heatmap_residual`** | **`baf_roi_heatmap_residual/`** | **本分支主方法：有空間結構與時間順序的 RF encoder → 兩個 expert 的 latent residual → 原輸出 heads 與逐步 fusion** |

`roi_experiments/` 負責 method registry、執行順序、共同 evaluator、摘要及圖表。模型各在自己的套件，三個 heatmap 方法共用原 BAF checkpoint、因果 RF 資料介面及 ROI 幾何。`alpha` 改變 expert 的混合比例；新的內部 residual 改變 expert 在輸出姿態之前的特徵，兩者有不同的作用點。

## 一個入口：不指定 method 就全部跑（包含歷史比較）

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

預設依設定跑 50–1000 ms、每 50 ms 一個 horizon；五種方法共用同一份 cache／split，以及同一組所有 horizons 都有 GT 的 origins。三個 heatmap 方法共用 frozen BAF；即使只選 heatmap method，也需要訓練或重用 BAF。只選 HMD baselines 時不要求 RF 檔案；MT-only 不會額外訓練 BAF。

機器路徑已寫在既有 shell script 開頭時，也可以直接：

```bash
bash scripts/train_csi_20261031.sh
bash scripts/train_csi_20261031.sh --method baf_roi_heatmap_alpha
```

這兩個命令是替代用法，會各啟動一個背景實驗。腳本沿用原 Python 環境，訓練結果、checkpoint、log、cache 與暫存檔集中寫入 launcher 的 `RUN_ROOT`（目前為 `/home/tonic/Projects/NSTC/EXTRA-DISC/Eason/tonic_nstc_csi_20261031`）；在不同機器上先調整該檔開頭的設定。

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

## 一個參數產生 evaluation 圖表

```bash
python scripts/plot_evaluation.py --path baf_roi_eval_1123_backtoorigin
```

`--path` 指向實驗輸出目錄即可；也支援 `evaluations/`、`evaluations/test/` 或單一 method 的評估目錄。每個 split 分開比較，輸出到指定目錄下的 `plots/<split>/`；開啟 `index.html` 可瀏覽所有圖表。重跑會更新相同名稱的圖表與報表。

自動產生 PNG、PDF 及彙整 CSV，包含 IoU／F1／precision／coverage、相對 BAF 的差異、pooled／recording-macro／scenario-macro 比較、pose mean／P95 error、FP／FN／selected partitions、margin sweep 的 coverage–cost 曲線、scenario／recording 差異、IoU 分布與 origins 組成，以及各階段的 train／validation 曲線。各方法曲線若完全相同會重疊，可看 delta 圖確認。

主要比較使用 margin 0；若沒有共同的 margin 0，使用所有 horizons／methods 共有的最小 margin，並標示在標題與 manifest。跨 horizons 的摘要採等權平均。程式會驗證 per-origin CSV 的 SHA-256、protocol、targets、RF cohort 與實際 origin IDs，拒絕將不匹配的資料混畫。訓練曲線標記的是 history 中 validation 最佳的 epoch；不讀取模型 checkpoint，也不需要 CUDA 或 PyTorch，僅使用既有依賴 numpy、pandas、matplotlib。

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

五個 methods 都經 `roi_experiments.evaluation`。舊 `baf_roi_heatmap.evaluation` 保留 compatibility imports。無 RF 的 standalone BAF／MT 也共用同一個 metric／report engine。

- 主表延續先前 PPT：**逐 origin Mean IoU**、**pooled Micro F1**，另列 precision、coverage/recall、FP/FN、selected count 與 pose error。
- 保留 all origins 及 RF-available subset；同一 RF 子集的 baseline 也重新評估。
- 同時輸出 pooled、original-recording macro、scenario macro。原始錄製檔因 gaps 被切成多段時，不會得到多個 recording 權重。
- 用完整 protocol、origin IDs、targets、scene／partition order 與 RF cohort hashes 檢查公平性，不只比較 sample count。
- 預設 prediction-side margins `0 5 10 20`，GT FoV 不變；margin 0 是主要比較，另外提供 coverage–selected-count 曲線。
- **SSIM 為 `not_evaluated`**：需要同一真實 future pose／content／renderer 的選取畫面與全內容 reference。Coverage 不能取代 SSIM，selected count 也不是實測 bytes。

## 文件

| 文件 | 內容 |
|---|---|
| [內部 residual：本分支主方法](docs/README_INTERNAL_RESIDUAL.md) | 作用點、公式、單次 rollout、訓練、對照組與操作指令 |
| [本分支驗證紀錄](docs/INTERNAL_RESIDUAL_VALIDATION.md) | 實際軟體驗證結果及尚未量測的項目 |
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
