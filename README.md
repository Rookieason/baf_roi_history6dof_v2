# BAF-Pose → ROI：僅使用 history 6DoF（v2）

## 延長 Horizon 與評估圖表

兩個模型的 prediction / evaluation 支援 **50–1000 ms**，1000 ms 使用 30 Hz 下的
30-step recursive rollout。可直接使用既有 v2 checkpoint，無須重訓。
`evaluate --horizons-ms` 僅覆寫此次評估；不指定時仍使用 cache manifest 的 horizons。
請保留訓練時的 cache 與 checkpoint，勿只為延長 horizon 重建 cache；cache ID 驗證仍會執行。
Preparation config 保持原本 50–250 ms 範圍，長 horizon 請透過 evaluation override 指定。

```bash
python -m baf_roi evaluate --cache cache/experiment01 --checkpoint runs/experiment01/baf_roi.pt --output runs/experiment01/test_1000ms --horizons-ms 200 400 600 800 1000 --device auto
python -m motion_transformer_roi evaluate --cache cache/experiment01 --checkpoint runs/mt_experiment01/motion_transformer_roi.pt --output runs/mt_experiment01/test_1000ms --horizons-ms 200 400 600 800 1000 --device auto
python -m baf_roi.plot_metrics --metrics runs/experiment01/test_1000ms/metrics.csv runs/mt_experiment01/test_1000ms/metrics.csv --labels BAF-ROI "Motion Transformer" --output runs/plots_1000ms --include-coverage
```

圖表工具也接受單一模型，輸出 PNG 與 PDF：

- `mean_iou_vs_horizon`：使用各 origin IoU 平均值 `iou_mean`，不是 `iou_micro`。
- `missed_redundant_vs_horizon`：使用 `fn / n_origins` 與 `fp / n_origins`。
- `coverage_vs_selected`（`--include-coverage`）：使用 `(tp + fp) / n_origins` 與
  `tp / (tp + fn)`，分母為零時 coverage 為 0，與 evaluator 的 recall 規則相同。
  各點標註 horizon；這些是 operating points，不是固定 horizon 的 selection trade-off curve。

同一次評估的所有 horizons 共用有效 origins；任一 horizon 缺少未來真實 pose 時，該 origin
會整體排除，因此 1000 ms 評估的 `n_origins` 可能減少。報告保留每條 trajectory 的
`rejected_missing_future`。模型比較須使用相同 cache、split、horizons、scene geometry 與
evaluation rules，並確認 `per_origin_metrics.csv` 的 origin/target 欄位相同；僅相同 origin
數量不足以證明對齊。圖表工具檢查 horizons 與 origin 數量，已有圖檔時拒絕覆寫。
長時間 rollout 可能累積誤差；這些 HMD-only 結果不能作為 CSI improvement 的證據。
完整 selection curve 需另做 prediction-side selection sweep，並保持 ground-truth ROI geometry 固定。

新增學長的 **Motion Transformer ROI 對照組**，位於 `motion_transformer_roi/`；保留作者模型/loss，共用 BAF cache、split、normalization 和 ROI 評估。操作及比較條件見 [docs/MOTION_TRANSFORMER.md](docs/MOTION_TRANSFORMER.md)。

```bash
python -m motion_transformer_roi train --cache cache/experiment01 --output runs/mt_experiment01 --device auto
python -m motion_transformer_roi evaluate --cache cache/experiment01 --checkpoint runs/mt_experiment01/motion_transformer_roi.pt --output runs/mt_experiment01/test --device auto
```

本專案把你提供的 BAF-Pose 程式移植到 `roi_baseline-main` 所描述的 agent6 記錄格式。它是 **pose-mediated ROI prediction**，不是另訓練一個直接輸出 ROI 的分類器。

```text
過去已觀測的 HMD 6DoF
  → 30 Hz、30 poses 的 causal history
  → 原版 Confidence CNN + Focused LSTM + Explorer TCN
  → temporal EMA + translation LERP / quaternion SLERP
  → 原論文的 recursive rollout
  → 50 / 100 / 150 / 200 / 250 ms 的 predicted pose
  → 固定場景幾何的可見性判定
  → 每個 horizon 的 multi-label ROI mask
```

**模型不讀取 CSI、heatmap、圖片、gaze、物件語意、場景座標或未來真實 pose。** 場景 partition 座標與相機 FoV 只供最後的幾何轉換；它們不是神經網路的輸入。沒有場景定義，就只能得到 predicted pose，不能有意義地命名哪些物件 partitions 是 ROI。

## v2：已使用補上的作者 data_preparation 原始碼

這版修正 v1 的 supervision 重建：**標籤採作者的 centered `[t-0.5,t+0.5]` 視窗，模型 input 仍為 past-only `[t-29/30,...,t]`。** 推論不執行標註規則、不读未來真實 pose 或 oracle label。

`vendor/data_preparation/` 的 20 個原始檔完整保存，`score_one_index` 直接載入；zero-rate handling、threshold、units 與邊界條件沿用原碼。詳見 **[本次逐項對照](docs/PREPARATION_REVIEW.md)**。

**v1 cache/checkpoint 不相容；請重新 prepare/train 到新目錄。** Neural architecture、loss、fusion、50–250 ms rollout 與 ROI mapping 沒有更換。

## 先釐清論文的 horizon

你提供的《Behavior-Aware 6DoF Head Pose Prediction for Volumetric Video Streaming》主要訓練 next-step predictor，但第 6.2.4 節、Figure 23（印刷頁 65／PDF 第 73 頁）**已經使用 recursive rollout，預測到 30 steps／1 秒**。本專案沿用此方法，不改模型為 multi-head，也不加入 multi-horizon training loss。

保留原版 30 Hz：一個 step 是 33.333… ms。100、200 ms 是第 3、6 step；50、150、250 ms 是第 1.5、4.5、7.5 step。因此最多預測 8 steps，再在相鄰「預測結果」之間做位置 LERP 和 quaternion SLERP，得到指定時間。沒有把 50 ms 偷換成 66.7 ms。

## 已保留與已修改的範圍

`vendor/baf_pose/` 的 23 個檔案保持原始位元組內容；其中三個 `model.py` 由新程式直接載入。`baf_roi/upstream_ops.py` 的四個函式逐字取自原版，包括 SLERP、EMA、fusion 與 weighted BCE。`python -m baf_roi audit` 會核對它們及新增 preparation 原始檔、23 個舊框架參考檔與 v1 supervision 封存檔。

原版推論使用整份測試檔的 position mean/std，包含未來 trajectory。本專案必須改成 **只由 training sessions 計算、保存到 checkpoint 的固定統計量**。這是實質改動，不能宣稱與原版 preprocessing 完全相同。

完整修改紀錄見 [docs/LOGIC_CHANGES.md](docs/LOGIC_CHANGES.md)，原始資料與幾何合約見 [docs/DATA_CONTRACT.md](docs/DATA_CONTRACT.md)。論文／程式差異與閱讀紀錄見 [docs/SOURCE_AUDIT.md](docs/SOURCE_AUDIT.md)。

## 安裝

在這個專案根目錄執行。可以使用已安裝、符合你硬體環境的 PyTorch；其餘依賴由 requirements 安裝。

```bash
python -m pip install -r requirements.txt
python -m baf_roi audit
python -m pytest -q
```

已驗證環境：Python 3.13.5、PyTorch 2.10.0+cpu、NumPy 2.3.5、SciPy 1.17.0、pandas 2.2.3。CUDA 路徑有 device 支援，但這次沒有 GPU 可供實測。`num_workers=0`，適用於 Windows，不需要 Linux 的 fork。

## 真實資料開始前的設定

編輯 `configs/default.json`。相對路徑以「設定檔所在資料夾」解析，不是任意工作目錄。Windows 路徑可寫成 `C:/Users/...`。

資料根目錄包含：

```text
sensor-agent-publish-subscribe/
└── db/
    ├── session_A/
    │   └── agent6_head/
    │       └── head.csv
    └── session_B/
        └── agent6_head.csv
```

不需要 `artifacts/`、MAT 或 PNG。每個 CSV 的 `payload_json` 內含 `pos[3]`、`rot_deg[3]`（或 `rot[3]`），時間來自 `payload.ts_utc`、CSV `ts_utc` 或 `recv_unix`。

必須先核對 `data.position_unit` 與 `data.rotation_convention`，然後才將 `coordinate_contract_confirmed` 設為 `true`。目前實作的是舊 ROI code 的語義：**yaw/pitch/roll，+X forward，+Z up，正 pitch 向上**。這是從舊程式推得的合約，不代表已由實際 recorder 原始碼或真實 CSV 證實。`position_unit` 的 `cm` 同樣只是待確認的起始設定；記錄若為公尺就改成 `m`。HMD 位置也必須與 scene centers 位於相同 world frame。

Scene 設定有兩條路：使用真實 world-space `partition_centers_file`；或沿用外部 adaptive server JSON 的 object centers 加上 uniform grids。後者必須確認 `object_size_cm`，並明確開啟 `allow_uniform_grid_approximation`。確認 FoV 與幾何後，才設 `geometry_confirmed=true`。模型訓練不需要 scene；ROI 評估才需要。

> 上傳的 `roi_baseline-main` 並沒有外部的 `config_example_front_cache_transform_lod.json`、真正的 partition bounds、相機投影矩陣或原始 session CSV。本專案不會替你虛構這些資料。

## 執行流程

每次使用新的輸出資料夾，避免不小心覆寫既有實驗。

```bash
python -m baf_roi prepare --config configs/default.json --output cache/experiment01
python -m baf_roi train --cache cache/experiment01 --output runs/experiment01 --device auto
python -m baf_roi evaluate --cache cache/experiment01 --checkpoint runs/experiment01/baf_roi.pt --output runs/experiment01/test --device auto
```

預設 training protocol：

| 階段 | 訓練資料 | Epoch 上限 | LR |
|---|---|---:|---:|
| Confidence | causal history 對應有效 centered Focused／Explorer target | 10 | 0.001 |
| Focused pretraining | 所有有效 next-step transitions | 8 | 0.001 |
| Focused fine-tuning | 當前與下一步皆為 Focused | 12 | 0.0001 |
| Explorer | 當前與下一步皆為 Explorer | 20 | 0.001 |

Batch size 64；branch 的 AdamW、weight decay、gradient clipping、ReduceLROnPlateau 與 early stopping 延續原版。Confidence 為 Adam 與原 run.sh 的 class-weighted BCE（Focused 3、Explorer 1）。只用 validation loss 選 checkpoint。

`prepare` 會列出每個 split 的兩種 regime、Unlabeled 與 clean transitions 數量。若任一訓練／驗證分支缺少需要的樣本，訓練會報錯；不會偷偷移動 test data、改標籤、放寬門檻或以另一分支頂替。

預設 split 是 session-level 70/15/15、seed 57。**若要與舊 baseline 公平比較，應指定同一份 split manifest**，不要讓兩個方法各自重新分割。`split.manifest` 支援 `train/valid/test`，也接受 `val` 作為 `valid` 的名稱；每個 session 必須恰好出現一次。

## 評估輸出與使用

`metrics.json`、`metrics.csv` 包含每個 horizon 的 multi-label precision／recall／F1／IoU、exact match、per-cell accuracy，以及 pose translation／rotation error。所有 horizons 使用同一批 origins，不按 regime 篩掉難樣本。另有逐 trajectory 結果與 `per_origin_metrics.csv`。

`predictions/*.npz` 儲存每筆 origin 的五個預測 pose、GT pose、alpha 與壓縮 ROI masks。`scene_snapshot.json` 提供 mask 欄位到 partition ID 的對照。還原 mask：

```python
import numpy as np
x = np.load("runs/experiment01/test/predictions/trajectory_00000_0000000.npz")
roi = np.unpackbits(x["roi_pred_packed"], axis=-1,
                    count=int(x["num_partitions"]), bitorder="little").astype(bool)
# roi.shape = [batch, 5, num_partitions]
```

NPZ 名稱依實際 test trajectory 而異，上例路徑需換成輸出檔。大量資料只要 summary 時，evaluate 加 `--no-save-masks`。

矩形 FoV 使用 `configs/scene_rectangular.example.json` 作為設定範例；數值不是你的 HMD 校正結果。可在評估時單獨指定 scene，而不用重訓 pose model：

```bash
python -m baf_roi evaluate --cache cache/experiment01 --checkpoint runs/experiment01/baf_roi.pt --output runs/experiment01/test_rectangular --scene-config configs/scene_rectangular.example.json
```

比較方法時必須一致使用同一種 ROI 定義，不能把圓錐版 baseline 與矩形版 BAF 的數字混在一起。

## 不需要未來資料的 prediction API

```python
from baf_roi.runtime import Predictor

predictor = Predictor.load("runs/experiment01/baf_roi.pt", device="cpu")
result = predictor.predict_observations(
    timestamps_s=observed_timestamps,       # 僅截至目前，嚴格遞增，單位秒
    positions=observed_positions,           # checkpoint 指定的記錄單位 cm 或 m
    rotations_deg=observed_yaw_pitch_roll,  # [N,3]
    horizons_ms=[50, 100, 150, 200, 250],
    stream_id="session_A",                 # 約 30 Hz 呼叫；新 session 要 reset
)
# result["positions_m"]: [1,5,3]，一律輸出公尺
# result["quaternions_wxyz"]: [1,5,4]
# result["rotations_ypr_deg"]: [1,5,3]
```

呼叫端至少提供約一秒可覆蓋 history 的觀測資料。`stream_id` 用來保留「已觀測時刻」的 confidence EMA；預測到未来的 EMA 不會寫回真實 stream state。不同使用者／session 不可共用同一個 stream ID。

也可用 JSON history：

```bash
python -m baf_roi predict --checkpoint runs/experiment01/baf_roi.pt --history history.json --output forecast.json
```

`history.json` 欄位就是 `timestamps_s`、`positions`、`rotations_deg`。加 `--scene-config` 就另外輸出 `roi_mask`。這個 JSON 不需要 target、Label、heatmap 或 scene feature。

## 測試與已知限制

```bash
python scripts/smoke_test.py --output smoke_output/new_run
```

此指令生成「明確標示為合成」的 agent6 CSV，四階段各訓練一個 epoch，測試完整流程。它不是你的研究 accuracy，不能當成真實實驗。交付不包含任何在真實資料上訓練的 checkpoint；合成測試權重也未打包，以免與正式實驗模型混淆。

本次執行驗證詳見 [docs/TEST_REPORT.md](docs/TEST_REPORT.md)。最重要的限制是：原版 normalization 已做必要的 causal 修正；行為標註已改用補上的作者原碼，但我們的 recorder/time-grid adapter 與作者 flat CSV 不完全相同；ROI 是 partition-center 的幾何近似，不含 occlusion、實際 mesh intersection 或雙眼不對稱投影；recursive rollout 可能累積誤差，必須在你的 held-out sessions 上量測。
### 跨 baseline 比較的額外提醒

這份 BAF adaptation 保留約一秒／30 poses 的 history，而舊 ROI 設定是 250 ms／8 samples。若要把差異歸因於 CSI 是否提供額外資訊，其他 baseline 與 proposal 的 history budget、sampling、split、target alignment 與 ROI geometry 也應統一。不能只統一 future horizon，卻讓各方法使用不同的過去資訊量。本版固定保留原 BAF 設定；要縮短它的 history，應另立實驗並記錄變更。

### 查核 v2 supervision

`cache/.../trajectory_*/supervision_metadata.npz` 保存 current_ / next_ 的作者數值欄位，與同目錄 anchors.npy 對齊；只供查核，不進模型。`head_rate_yaw_q90` 保留原碼的 rad/s 单位。對照前後版本標籤可執行：

```bash
python scripts/compare_supervision.py --cache cache/experiment02 --output runs/experiment02/supervision_comparison.json
```

它比較相同 input origins 上的 v1/v2 label，**不是模型 accuracy**。
