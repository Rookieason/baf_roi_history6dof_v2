# data_preparation.zip 對照與修正紀錄（v2）

## 結論

這份附件就是上一版缺少的 BAF-Pose 資料前處理程式。新增來源確認了兩個不能只憑論文摘要猜測的實作細節：

1. **原作者的 supervision label 使用 centered window `[t - 0.5, t + 0.5]`，不是 past-only model history。**
2. **yaw-rate 的 exact zero 在數方向變化時被視為正號，不是先刪除零。**

因此上一版依論文重建的 supervision 不能继续當成與原碼等價。本版改為直接載入原作者 `supervision.py::score_one_index`，不再自行重寫 scoring rule。Estimator、Focused、Explorer、loss、fusion 和 rollout 的數學沒有在這次更新中更換。

所有原始檔案與以下行號來自使用者上傳的附件，不是網路上其他版本。

## 1. 原始前處理實際做什麼

`vendor/data_preparation/run.sh` 的執行順序：

| 階段 | 檔案 | 原始用途 | 新專案的處理 |
|---|---|---|---|
| 1 | `txt_to_csv.py` | 把 MM'23 的 28 欄 TXT 轉成 flat CSV | 不拿它讀我們的 `payload_json`；保留 agent6 reader |
| 2 | `resample_trajectory.py` | 對整條 trajectory 建 30 Hz grid；位置線性插值、角度 unwrap/interpolate/wrap | 保留 v1 的 causal history adapter；supervision 另建離線 centered grid，見第 5 節 |
| 3 | `csv_label_generator.py` | motion statistics、score、`Label` | 作為獨立交叉核對來源，不取代第 5 步 |
| 4 | `split_data.py` | 對 `_with_label` CSV 做 file split | 保留我們的 session split，避免同 session 的 CSV 被拆開 |
| 5 | `supervision.py` | 重新從 resampled CSV 算 labels，增加 confidence metadata，再切 train/val/test | **直接使用此檔的原始 scoring 函式** |

第 5 步的 input 是 `_resampled_30Hz`，不是第 3 步的 `_with_label`。原 README 的 Dataset Usage 也指定 BAF-Pose 應使用 `_with_conf_supervision`。

`data_analyzer/` 是第 3 章 feature/behavior distribution 圖表程式。其 gaze、eye 等分析欄位沒有加入本專案的模型或 supervision input。附帶的 9 份 PDF 是原作者的既有圖表，不是我們的資料或新的測試結果。

原 README 對特定 MM'23 trajectory 的截斷警告只屬於那個 dataset；沒有把特定姓名／frame number 的清理規則套到我們的 sessions。

## 2. 最重要的修正：分清楚 input window 與 supervision window

### 原始碼證據

`supervision.py` 第 40–45 行明確说明保留 centered scoring window，供 causal dataset builder 使用。第 323–328 行：

```python
mask = (timer >= t_i - WINDOW_SEC / 2) & (timer <= t_i + WINDOW_SEC / 2)
idx = np.where(mask)[0]

if len(idx) < MIN_SAMPLES:
    return None
```

其中 `WINDOW_SEC = 1.0`、`MIN_SAMPLES = 5`。第 400–407 行同時記錄：

```text
confidence_window_end_time = t
confidence_window_start_time = t - 1.0
supervision_window_center_time = t
supervision_is_centered_window = 1
```

**`confidence_window_end_time=t` 不表示 supervision 是 past-only。** 這些欄位是在區分模型歷史視窗與離線標籤視窗。

### v1 與 v2

| 項目 | v1 | v2 |
|---|---|---|
| 神經網路的 history | `[t-29/30, ..., t]`，30 poses | 不變 |
| 當前 supervision label | 對上述 past-only history 算分數 | 原作者 `[t-0.5, t+0.5]` centered score |
| 下一步 supervision label | 對平移一步的 past-only history 算分數 | 原作者以 `t+1/30` 為中心的 score |
| confidence target | F=0、E=1，Unlabeled 排除 | 不變，但標籤內容改回原作者算法 |
| branch selection | 當前與下一步 Label 皆相同 | 不變，但使用修正後的 labels |

在完整、精確包含兩端點的 30 Hz grid 上，1 秒 centered interval 可以包含 31 samples；不是硬切 30 rows。實際原碼是時間條件，受邊界與浮點表示影響，以該 mask 為準。分母用實際取到的 `t_seg[-1]-t_seg[0]`。檔案尾端只有半個視窗時，原作者仍允許在至少 5 samples 下產生標籤；v2 保留此規則，不補造未來 pose。

下一步 clean label 的理論上界是 `t+1/30+0.5`，約 `t+533.3 ms`，而不是把 supervision 限制到本次 ROI 的 250 ms horizon。這仍然只是離線訓練資料準備。

### 為什麼推論仍符合 history-only

訓練範例是「past-only 6DoF input，對應 offline supervision target」。建立 target 使用同一訓練 session 的未來部分，不等於將那段未來當成模型 input。

推論時只有 trained estimator 從已觀測 history 產生 `alpha_raw`，再經 EMA 融合兩個分支。**不呼叫 heuristic supervisor，不讀 `Label`、`binary_target`、`explorer_score` 或未來真實 pose，也不拿 test label 當 oracle alpha。** Test origins 不按 label 篩選。

Session 先分配到 train/valid/test，再各自在所屬 session 裡建立 supervision；不同 split 或不同 gap segment 的資料不會混在一個標籤視窗裡。Normalization 仍只 fit training sessions。

## 3. Zero crossing、units 與邊界也必須對齊

### Zero crossing

原始 `supervision.py` 第 100–106 行：

```python
s = np.sign(x)
s[s == 0] = 1
return float(np.sum(s[1:] * s[:-1] < 0))
```

例如 signed yaw rates `[-1, 0, -1]`，原作者定義會計為兩次，而 v1 刪除零後計為零。v2 不評斷或另「修正」這個定義，直接沿用原碼。這是可影響 score／label 的實質差異。

### Units

原始第 349–361 行先把旋轉轉成 radians，因此 `head_rate_yaw_q90` 的儲存單位是 **rad/s**；60、180 deg/s 的 thresholds 也轉成 radians 後才比較。v2 的 audit metadata 保留原始單位，不把這欄重新命名卻默默換成 deg/s。

原始 debug 欄位 `delta_rx/ry/rz_deg_s` 使用普通 `np.diff`，不是 circular difference。跨 ±180° 時可能很大；v2 保留這個原碼行為。這些 debug 數值不會成為模型 input，Explorer 仍為原 run.sh 的 `aux_dim=0`。

### Invalid 與 Unlabeled

原作者 Unlabeled 對應 `binary_target=NaN`、`valid_for_confidence=0`；`score_one_index` 無法算分時回傳 None。v2 保留這個語義。內部 compact label codes 為 F=0、E=1、Unlabeled/invalid=-1，另以 next_label=-2 表示沒有合法下一步 pose target。這個 -2 不代表第四種行為。

## 4. 不可以把 HeadRY 的 array index 直接套到我們的資料

原始第 349–351 行以 `rot_seg_rad[:, 1]` 當 yaw，即原 dataset 的 `HeadRY`。

我們目前的 recorder 合約是 `rot_deg=[yaw,pitch,roll]`，因此 scoring-only adapter 把：

```text
recorder [yaw, pitch, roll] → supervisor [pitch, yaw, roll]
```

讓原函式第二個角度欄位取得「語義上的 yaw」。這個排列 **只供 supervision scoring**；它不是幾何 xyz rotation，也不拿來建立 quaternion。

模型既有幾何 adapter 仍使用 `[roll,-pitch,yaw]`，對應舊 ROI 的 `Rz(yaw) Ry(-pitch) Rx(roll)`。本次沒有改神經網路的 input order 或 quaternion 轉換。實際 recorder 是否符合這份 convention，仍須由真實資料／recorder code 確認。

## 5. 哪些資料協定仍然沒有逐位照搬原作者？

v2 保留原 scoring 函式的位元組與運算，但不宣稱整條不同 dataset 的前處理完全等價。

**時間 grid。** 原資料是先把整條 trajectory resample 成單一固定相位 30 Hz grid；v1 為避免 irregular recording 的當前 grid pose 插值偷偷使用晚到 raw sample，使用「實際已觀測 row 的 timestamp」當 decision origin。v2 不為修標籤而換掉此協定。每個 supervision 的 30 Hz grid 重新以同一個實際 origin t 對齊，再呼叫原函式。輸入 history 仍單獨以 cutoff=t 建立。

因此原 scoring 在「給定同一批 resampled times/poses」下完全一致，不表示不同的 origin/grid phase、raw sampling、數值量化或資料內容會產生同一串 labels。此項是資料時間 adapter 差異，仍記錄於 L05。

**Float serialization。** 原 supervision CLI 輸出 `%.6f` CSV；新專案使用既有 NPY float cache，不引入 CSV 六位小數 round-trip。物理公式未改，但位於 threshold 的數值可能受不同量化影響。這是既有資料介面差異，不宣稱 bitwise dataset reproduction。

**檔案與 split。** 不套 MM'23 TXT converter；保留 agent6 JSON reader、同 session grouping、gap segmentation、明確單位與 recorder convention。原 split 的 70/15/15、seed 57 原則不變，但 session split 當然不等於作者 flat-file split。

**Normalization。** 原全檔 mean/std 會包含 test 的未來位置；v1 的 training-only statistics 修正仍保留，見 L01。

**Unlabeled history。** 這次有了新 generator，可以確認：原 `confidence/utility.py` 第 53–58 行會先 `dropna(binary_target)` 再建視窗，而 generator 對 Unlabeled 設 NaN，因此 30 rows 可能跨過被刪掉的時刻。v2 保留 v1 的修正：只排除 Unlabeled 的「監督 target」，不刪去它對應的歷史 pose，維持 inference 一致的完整時間視窗。這不是本次新增 model logic，而是原先移植中必須明確補記的 dataset-builder 差異。

## 6. 實際改了哪些檔案

| 檔案 | 變更 |
|---|---|
| `vendor/data_preparation/` | 新附件全部 20 檔原樣保存、加入 SHA audit |
| `baf_roi/supervision.py` | 移除自行重建的 scoring；直接 import 原 `score_one_index`；新增 semantic yaw / centered-grid adapter |
| `baf_roi/data.py` | 當前與下一步 labels 使用 centered supervisor；保存 audit-only metadata；cache protocol v2 |
| `baf_roi/train.py` | 只有 checkpoint format/provenance 更新；optimizer、loss、stage math 不变 |
| `baf_roi/runtime.py` | 只有拒絕 v1 checkpoint 的版本 guard；推論數學不變 |
| `baf_roi/evaluate.py` | 只增加 supervision protocol 名稱到 report；評估演算法不變 |
| `requirements.txt` | 直接載入作者原模組需要 pandas |
| `tests/test_author_preparation.py` | 原碼欄位 parity、centered/causal 區別、zero handling、units、邊界、yaw axis 等回歸 |
| `scripts/verify_smoke_regressions.py` | 可重跑 batch-size 與 test-label 隔離驗證，不改正式資料 |
| `scripts/compare_supervision.py` | 可在同一組 origins 上比較 v1/v2 labels，不涉及 accuracy |
| `reference/v1_supervision.py` | 封存先前版本，只供對照，不在訓練或推論主路徑 |
| docs / provenance | 更新已解決的缺檔敘述、差異、測試與來源 |

三個原始 model.py、`models.py`、`pose.py`、`upstream_ops.py`、`roi.py` 均與 v1 相同。Focused LSTM、Explorer TCN、Confidence CNN、loss、EMA、SLERP、50–250 ms rollout/interpolation 均未變更。

## 7. 驗證與使用

目前 65 項 pytest 全部通過。另執行四個階段各 1 epoch 的完整 synthetic smoke test，並做 test-label corruption 與 batch-size regression。完整紀錄見 `TEST_REPORT.md`。

在同一個合成 fixture 上，633 個 input origins 中有 96 個 label 與 v1 不同。這只是證明前處理差異會影響標籤，不是你的資料比例，也不是準確率改進的證據。

**請重建 cache、重新跑訓練，不要沿用 v1 模型。** Confidence target 和兩個 specialized branches 的樣本集合已變；Focused pretraining 的 pose-only objective 本身未變，但整個正式流程會重新跑四階段，避免混用實驗來源。

```bash
python -m pip install -r requirements.txt
python -m baf_roi audit
python -m pytest -q
python -m baf_roi prepare --config configs/default.json --output cache/experiment02
python -m baf_roi train --cache cache/experiment02 --output runs/experiment02 --device auto
python -m baf_roi evaluate --cache cache/experiment02 --checkpoint runs/experiment02/baf_roi.pt --output runs/experiment02/test --device auto
```

本次依然沒有使用者的真實 session CSV／完整場景定義，因此沒有報告真實 ROI 準確率。新附件補齊的是原作者 supervision code，不是我們的 dataset 或已訓練模型。
