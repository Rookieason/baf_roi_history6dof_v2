# 五種 ROI 方法的共同執行與評估

本分支以 `scripts/run_experiments.py` 作為共同入口。`--method` 指定要執行的方法；省略時，會依序處理五種方法（含歷史比較），並把結果放進同一份比較表。本分支主方法為 `baf_roi_heatmap_residual`，設計與操作詳見 [內部 residual](README_INTERNAL_RESIDUAL.md)；可用 `bash scripts/train_csi_internal_residual.sh` 跑 BAF／MT／內部 residual 三組。

以下指令皆假設目前位於 repository 根目錄，且使用已安裝 `requirements-heatmap.txt` 的 Python 環境。`python -m roi_experiments` 與 `python scripts/run_experiments.py` 等價。

## 1. 五種方法與程式責任

| `--method` 名稱 | 演算法程式碼 | 輸入與作用 | 是否需要 BAF checkpoint |
|---|---|---|---|
| `baf_roi` | `baf_roi/` | 原始 HMD history → Confidence、Focused、Explorer → 逐步融合姿態 → ROI | 本身訓練或載入 BAF |
| `motion_transformer_roi` | `motion_transformer_roi/` | 相同 HMD history → Motion Transformer → 未來姿態 → ROI | 不需要 |
| `baf_roi_heatmap` | `baf_roi_heatmap/` | 歷史比較：凍結 BAF 完成預測後，執行 HMD／heatmap adapter | 需要，共用相同的凍結 BAF |
| `baf_roi_heatmap_alpha` | `baf_roi_heatmap_alpha/` | Heatmap 只修正 BAF rollout 每一步的 α，維持原本兩個姿態專家 | 需要，共用相同的凍結 BAF |
| `baf_roi_heatmap_residual` | `baf_roi_heatmap_residual/` | 每一步在兩個 expert 的 latent 加入 CSI residual，再經原輸出 heads 產生姿態 | 需要，共用相同的凍結 BAF |

`roi_experiments/` 放共同的 method registry、工作流程、評估、彙整與繪圖。各方法的模型與訓練演算法仍由上表對應的套件負責。Heatmap 方法可以重用 `baf_roi/` 的凍結專家，以及既有 heatmap 資料載入器。

`baf_roi_heatmap` 原有的 `expert_gate` 是對已經預測好的 horizon 終點做融合；新增 `baf_roi_heatmap_alpha` 則在 BAF 自回歸預測過程中修改 α。這兩個名稱代表不同方法，不能把結果混成同一組。

## 2. 最直接的執行方式

### 使用目前的 20261031 recording 設定，完整跑五個方法

```bash
python scripts/run_experiments.py \
  --config configs/recorded_20261031.json \
  --output runs/compare_20261031_01
```

`run` 是預設命令，也可以明寫成 `python scripts/run_experiments.py run ...`。省略 `--config`、`--cache`、`--prepared` 三者時，預設採用 `configs/recorded_20261031.json`。

此流程只準備一次共同 pose cache、只訓練一次 BAF，再讓三個 heatmap 方法共用這份 BAF。Motion Transformer 使用同一個 pose cache。五個方法最後使用同一份 prepared origins、ground truth、scene 與 horizons 評估。

### 只跑一種方法

```bash
python scripts/run_experiments.py \
  --config configs/recorded_20261031.json \
  --method baf_roi_heatmap_alpha \
  --output runs/alpha_20261031_01
```

只選擇 heatmap 方法時，若沒有提供 BAF checkpoint，仍會先訓練必要的 BAF；這是模型依賴。比較表只列出指定的方法。

### 一次選擇多種方法

```bash
python scripts/run_experiments.py \
  --config configs/recorded_20261031.json \
  --method baf_roi motion_transformer_roi \
  --output runs/hmd_baselines_01
```

只選這兩個 HMD-only 方法時，不需要 RF manifest、heatmap 檔或 CSI 檔。只選 `motion_transformer_roi` 時，也不會額外訓練 BAF。

### 沿用原本的背景執行 launcher

```bash
bash scripts/train_csi_20261031.sh
bash scripts/train_csi_20261031.sh --method baf_roi_heatmap_alpha
```

Launcher 內的 Python、run root 與 device 仍是機器設定；第一次換機器時，先調整這些設定。未指定 method 的第一個指令會跑五個方法。Launcher 會印出 PID、輸出目錄與 log 路徑；直接使用 Python 入口則在目前終端執行。

## 3. 選擇資料入口：config、cache、prepared

三個參數互斥，分別對應不同的工作進度。

| 入口 | 已經準備好的內容 | 入口會處理的工作 |
|---|---|---|
| `--config CONFIG.json` | 原始 HMD recordings 與資料設定 | 建立 pose cache。若是現有 `recorded_heatmaps` profile 且選了 heatmap 方法，也會匯入既有數值 heatmaps、建立 RF manifest |
| `--cache CACHE` | 已有 pose cache 與固定 split／normalization | 重用 cache；選 heatmap 方法時另外提供 `--rf-manifest` |
| `--prepared PREPARED` | 已有配對資料、因果 RF 契約、凍結 BAF 預測、共同 origins／targets／scene／horizons | 重用整份實驗資料，並自動載入該 prepared 所記錄的 BAF checkpoint |

### 既有 pose cache，加上已核對的 RF manifest

```bash
python scripts/run_experiments.py \
  --cache cache/experiment01 \
  --rf-manifest data/rf_manifest.json \
  --baseline-checkpoint runs/baf_seed42/baf_roi.pt \
  --output runs/compare_cached_01
```

RF manifest 必須明確對應這一個 cache，包含每個 recording 的 heatmap 檔、時鐘關係與 grouping。共同入口不會自行猜測兩個 recording 是否同步。若選了 heatmap 方法但缺少可用的資料入口，會報錯，不會默默略過該方法。

### 重用 prepared，只訓練新的 alpha 方法

```bash
python scripts/run_experiments.py \
  --prepared runs/compare_20261031_01/prepared \
  --method baf_roi_heatmap_alpha \
  --alpha-epochs 20 \
  --output runs/alpha_from_same_baf_01
```

這個指令不會重新訓練 BAF。它使用 prepared 內記錄的 baseline 路徑與 SHA-256，確保新增方法與原實驗共用相同權重。

Prepared 已固定 RF input config、scene 與 horizons，因此不能用新的 `--input-config`、`--scene-config` 或不同的 `--horizons-ms` 偷換既有實驗。要改這些條件，請從相同 pose cache 重新 prepare 成另一個實驗。

## 4. 資料路徑與 heatmap 前提

設定檔裡的 `dataset_root`、split manifest 與 scene 檔相對路徑，是**相對於設定檔所在目錄**解析；CLI 的 `--output` 等路徑則相對於執行指令的位置。

`configs/recorded_20261031.json` 目前指向相鄰的 `sensor-agent-publish-subscribe` 與 `adaptive-volumetric-stream-server` 專案。這些資料並沒有隨 ROI repository 一起提供。

現有 recorded importer 使用預先產生的 AoA–ToF **數值** heatmaps。它會讀取以下資料來建立時間與來源契約：

| 路徑，以 dataset root 為起點 | 用途 |
|---|---|
| `db/<session>/sensor.full.50010/` | HMD pose 與接收時間 |
| `intermediates/<session>/heatmaps/<session>_heatmaps.npz` | 已產生的數值 heatmaps |
| `intermediates/<session>/matched_csi/<session>_matched.csv` | Heatmap index 與各 NIC 原始資料的對應 |
| `db/<session>/csi.rx.1/` 至 `csi.rx.4/` | 實際接收時間，用於推算每張圖何時已可取得 |
| `intermediates/<session>/run_params.json` | 核對歷史生成設定 |
| `intermediates/<session>/merged_csi/<session>_merged.npz` | 來源雜湊與單張生成結果的數值核對 |

此 importer 還會核對 profile 指定的 `valder` 原始碼雜湊。因此，只有一個 heatmap NPZ、缺少原始對應資訊時，不能直接假裝已滿足 recorded profile。

若手上已經有符合本專案格式的 heatmap NPZ 與已核對的 RF manifest，可走 `--cache` 加 `--rf-manifest`；這條入口不要求重新執行 MUSIC。格式與時間欄位詳見 [架構文件](README_HEATMAP_ARCHITECTURE.md)。PNG 圖片不是訓練輸入。

## 5. 重用各方法的 checkpoint

| 參數 | 重用內容 |
|---|---|
| `--baseline-checkpoint` | BAF Confidence、Focused、Explorer |
| `--motion-transformer-checkpoint` | Motion Transformer |
| `--heatmap-checkpoint` | 原有 residual／expert_gate adapter |
| `--alpha-checkpoint` | 新增的 alpha-only adapter |

例如，所有模型都已訓練好，希望由共同入口建立一份結果：

```bash
python scripts/run_experiments.py \
  --prepared runs/compare_20261031_01/prepared \
  --motion-transformer-checkpoint runs/compare_20261031_01/models/motion_transformer_roi/motion_transformer_roi.pt \
  --heatmap-checkpoint runs/compare_20261031_01/models/baf_roi_heatmap/adapter.pt \
  --alpha-checkpoint runs/compare_20261031_01/models/baf_roi_heatmap_alpha/alpha.pt \
  --output runs/reused_models_01
```

BAF 與 MT checkpoint 必須使用完全相同的 cache ID 與 normalization；兩個 adapter 必須對應相同的 prepared experiment。僅僅「模型 shape 相同」不代表可交換 checkpoint。

若原本就是共同入口產生的 run，下一節的 `evaluate --run` 更直接，不需要重新列出這些路徑。

## 6. 只重新評估

```bash
python scripts/run_experiments.py evaluate \
  --run runs/compare_20261031_01 \
  --output runs/compare_20261031_eval02 \
  --split test \
  --margins 0 5 10 20
```

這裡完全不訓練模型。`--run` 讀取 `experiment.json`，使用它記錄的 checkpoint、cache 與 prepared 路徑，也會檢查 checkpoint 雜湊。

`evaluate` 省略 `--method` 時，評估來源 run 中原本選擇的全部方法；也可用 `--method baf_roi_heatmap_alpha` 只重評其中一種。若來源 run 沒有某方法的 checkpoint，會直接報錯。

要看 validation 的結果，改成 `--split valid`。所有輸出必須使用新的或空的目錄，避免覆寫既有實驗。

## 7. 常用設定

| 參數 | 行為 |
|---|---|
| `--device auto` | 有可用 CUDA 時使用 CUDA，否則 CPU；也可明寫 `cpu` 或 `cuda:0` |
| `--threads 2` | PyTorch CPU intra-op threads |
| `--horizons-ms 50 250 1000` | 新實驗共同使用這些 horizons；支援 50–1000 ms，不能重複 |
| `--mt-epochs 30` | MT 訓練上限；保留其既有 validation early stopping |
| `--heatmap-epochs 20` | 原有 heatmap adapter epochs；未指定時採 recorded profile 設定，否則 20 |
| `--alpha-epochs 20` | Alpha adapter epochs；未指定時與 heatmap epochs 相同 |
| `--lr 0.001` | Heatmap 模組的 learning rate |
| `--residual-epochs 20` | 新內部 residual 的訓練 epochs |
| `--residual-checkpoint <path>` | 重用相同 prepared experiment 的內部 residual checkpoint |
| `--residual-modality hmd` | 內部 residual 的 HMD-only residual-head 控制 |
| `--residual-ablation metadata_only` | 內部 residual 的 RF 內容消融；保留 RF timing／mask |
| `--heatmap-adapter residual` | `baf_roi_heatmap` 使用 residual；可改成 `expert_gate` |
| `--modality hmd` | 原有 heatmap adapter 的 HMD-only 容量對照；不會把 alpha method 改成其他模型 |
| `--margins 0 5 10 20` | 共同的預測 FoV margin sweep，預設就是這四個值 |
| `--delivery-delay-ms 100` | 在 prepared RF 資料上增加接收延遲，以相同 origin 集合檢查缺測 fallback |

`--seed` 與 `--batch-size` 在從 `--config` 建立新 cache 時，會先寫入解析後的設定。重用既有 cache 並重新訓練 BAF 時，必須遵守 cache 內的 BAF seed／batch size；若要不同設定，建立新 cache，或明確重用 BAF checkpoint。BAF 的四階段 epochs 仍由 config 的 `training.epochs` 控制。

帶有 `recorded_heatmaps` 的 profile 會保留原 launcher 的資源監控，重用此 profile 的 cache／prepared 也適用：開始時要求輸出磁碟至少 8 GiB 可用；執行期間每 5 秒檢查，磁碟不足 2 GiB、系統可用 RAM 不足 2 GiB，或程序峰值 RSS 超過 6 GiB 時停止程序，exit code 為 75。初始檢查也包含同樣的 RAM／RSS 限制。正常完成或訓練拋出例外時會停止監控執行緒。一般不含此 profile 的資料與 `--dry-run` 不啟用這個監控。

只驗證流程時可加 `--smoke-epochs 1`，會把這次需訓練的各模型／階段都設成一個 epoch，並在輸出標記 smoke test。這個結果用於軟體驗證，不能當完整訓練的研究結論。

執行前可以查看計畫：

```bash
python scripts/run_experiments.py \
  --config configs/recorded_20261031.json \
  --method baf_roi_heatmap_alpha \
  --output runs/alpha_20261031_01 \
  --dry-run
```

Dry-run 列出方法、BAF 依賴、資料入口、horizons 與 margins，不會訓練或寫入結果；它不是實際資料內容與同步正確性的完整檢驗。

## 8. 公平比較如何被固定

共同入口對同一次比較固定以下條件：

- 相同 pose cache、train／valid／test split、HMD normalization 與 30-frame／30-Hz history。
- 相同 scene、partition IDs 及其順序、ground-truth FoV 與 exact prediction horizons。
- 只採用在**所有指定 horizons** 都有 ground truth 的共同 origins；不讓某方法只評估比較容易的短 horizon samples。
- 三個 heatmap 方法共用相同的已訓練 BAF；CSI 模組訓練不修改原有 BAF 專家權重。內部 residual 的梯度仍穿過原 heads 與完整 rollout。
- Heatmap normalization 只由 training recordings 擬合；每個 origin 只讀取在當時已可取得、且符合時間窗限制的 RF 資料。
- 全部方法都由同一個 metric engine 算 ROI；沒有可用 RF 的 origin 仍保留在全體評估，並另外提供 RF-available 子集合。

輸出不是只比較 sample count。彙整時還要求 evaluation protocol、origin ID、target bytes、RF 可用樣本集合、額外接收延遲、scene、split 與 horizons 的識別資訊一致。不一致時會停止比較。

同一模型單獨執行與放入多模型比較時，使用同一組 ROI 指標定義。若另一次獨立訓練改了 seed、cache 或 checkpoint，那就是另一個實驗；不應把權重差異解讀成 CSI 的作用。

## 9. 評估指標與先前 PPT 的對應

主表與預設圖使用 `pooled`：Mean IoU 先逐 origin 計算再平均，Micro F1 先將所有 origins 的 TP／FP／FN 加總再計算。這對應先前 PPT 的 **Mean IoU + Micro F1 vs prediction horizon**，而非把 recording-macro 與 window mean 混為同一個數字。

對固定 horizon，令第 $i$ 個 origin 的預測 ROI 為 $P_i$、真實未來 ROI 為 $G_i$：

$$
\operatorname{MeanIoU}=\frac{1}{N}\sum_{i=1}^{N}\frac{|P_i\cap G_i|}{|P_i\cup G_i|},\qquad
\operatorname{MicroF1}=\frac{2\sum_i TP_i}{2\sum_i TP_i+\sum_i FP_i+\sum_i FN_i}.
$$

| 輸出欄位 | 定義與用途 |
|---|---|
| `iou_mean` | 每個 origin 的 IoU 平均，是主 Mean IoU 指標 |
| `f1_micro` | 所有 origins 合併計數後的 F1，是主 F1 指標 |
| `coverage_micro`／`recall_micro` | $\sum TP/(\sum TP+\sum FN)$，需要的 partitions 有多少被選到 |
| `selected_mean` | 每個 origin 平均選取 $TP+FP$ 個 partitions |
| `fp_mean`／`fn_mean` | 平均多選／漏選的 partition 數 |
| `delta_iou_mean_vs_baf` | 同 horizon、同 margin 下，相對 `baf_roi` 的 Mean IoU 差值 |
| `recording_macro` | 先按原始 recording 計算，再讓各 recording 等權；gap 切段不會變成多個獨立 recording 權重 |
| `scenario_macro` | 有 scenario metadata 時，進一步讓 scenario 等權；資料不足時明確為空 |

空集合慣例固定為：預測與 ground truth 都空時 IoU 為 1；precision／recall／coverage／F1 的分母為零時回傳 0，另有空集合計數及 `coverage_defined` 供辨識。

### Margin sweep 是什麼

Margin 只擴張**預測端**的 FoV，ground-truth FoV 不變。以目前 110° 的 cone 為例，`margin=10` 使預測 cone 全角變成 130°。這讓我們觀察多選一些 partitions 後，coverage 與漏選數如何變化。

這是一種幾何選取政策 sweep，不是模型的 confidence threshold sweep。若未來改用 learned per-partition scores，必須另外定義與實作其 threshold policy。

### SSIM 的狀態

先前研究目標是提高 Mean IoU，同時滿足實際渲染品質門檻。這一版仍沒有在共同 evaluator 裡提供 matched rendered frames 或 renderer，所以輸出中的 SSIM 是 `not_evaluated`，也沒有自行指定 $q_{\min}$。

要量 SSIM，必須在相同真實未來 pose、相同場景內容與 renderer 下，比較「只傳送／渲染選定 partitions」和完整參考畫面。Coverage 是幾何 proxy，不能代替這個實驗；selected count 也不是實際傳輸 bytes。

## 10. 輸出檔案與共同繪圖

| 輸出位置 | 內容 |
|---|---|
| `plan.json` | 此次方法、資料入口、依賴與 horizons／margins |
| `resolved_input_config.json` | 從 config 建立新 cache 時採用的解析後設定 |
| `data/pose_cache/` | 本次新建的共同 pose cache；若重用既有 cache，記錄其原路徑 |
| `data/maps/`、`data/rf_manifest.json` | 使用 recorded importer 時轉入的 heatmap 資料與 RF mapping |
| `models/baf_roi/baf_roi.pt` | 本次訓練的 BAF；重用 checkpoint 時不複製成另一份 |
| `models/motion_transformer_roi/motion_transformer_roi.pt` | 本次訓練的 MT |
| `models/baf_roi_heatmap/adapter.pt` | 本次訓練的原有 heatmap adapter |
| `models/baf_roi_heatmap_alpha/alpha.pt` | 本次訓練的 alpha adapter |
| `prepared/` | 共用的因果 RF 輸入設定、凍結 BAF 預測與未來 targets |
| `experiment.json` | Checkpoint 路徑／雜湊、cache／prepared、方法與是否完成 |
| `evaluations/test/<method>/report.json` | 每個 method 的 pooled、recording／scenario summary 與 protocol |
| `evaluations/test/<method>/per_origin.csv` | 逐 origin、horizon、margin 的計數與誤差，可追查比較內容 |
| `summary.csv`、`summary.json` | 所選方法共同比較表；validation 時對應路徑改為 `evaluations/valid/` |

使用同一次配對評估的結果繪圖：

```bash
python -m roi_experiments.plot \
  --evaluations \
    runs/compare_20261031_01/evaluations/test/baf_roi \
    runs/compare_20261031_01/evaluations/test/motion_transformer_roi \
    runs/compare_20261031_01/evaluations/test/baf_roi_heatmap \
    runs/compare_20261031_01/evaluations/test/baf_roi_heatmap_alpha \
  --output runs/compare_20261031_01/plots \
  --aggregation pooled \
  --margin 0
```

輸出包含 Mean IoU／Micro F1／coverage／FP／FN 等隨 horizon 的圖，以及各 horizon 的 coverage vs selected partitions 曲線；每張同時存成 PNG 與 PDF。`--margin 0` 固定 horizon 圖採用未擴張的預測，coverage 曲線仍會使用已評估的完整 margin sweep。

若明確要看 recording 等權的結果，可以改成 `--aggregation recording_macro`。繪圖會檢查 protocol、origins、targets、RF subset 與來源檔雜湊，不會將不相符的評估結果直接畫成同一組公平比較。

模型原理見 [Heatmap 演算法文件](README_HEATMAP_ALGORITHM.md)，較完整的實驗設計見 [Evaluation 文件](README_HEATMAP_EVALUATION.md)。
