# Motion Transformer ROI 對照組

實作位於 `motion_transformer_roi/`；模型與 loss 作者原碼逐位元組保存在 `vendor/motion_transformer/`。執行不需要 `Code/`。來源路徑與 SHA256 見 `vendor/motion_transformer/provenance.json`。

## 方法與適配

保留學長 `Code/sota_baseline/motion_transformer/model.py` 的模型：7 維 normalized translation + wxyz quaternion、30 幀、d_model=128、4 heads、2 encoder layers、feedforward=256、latent=128、dropout=0.1、GELU、sinusoidal positional encoding 和 last-token pooling。輸出 translation residual 與相對 quaternion，依作者 `q_delta ⊗ q_current` 重建下一步旋轉。

loss 直接呼叫作者 `common_utility.py`：translation L1 + 0.3 × sign-invariant quaternion cosine loss。訓練依原 `run.sh`：30 epochs 上限、batch=64、AdamW lr=5e-4、weight_decay=1e-4、gradient clip=1、ReduceLROnPlateau factor=0.5/patience=2、early stopping patience=6；只依 validation loss 選 checkpoint。

資料適配沿用 `baf_roi`：

- 讀取 `db/<session>/agent6_head/head.csv` 或 `agent6_head.csv` 的 `payload_json`（`pos`、`rot_deg` 或 `rot`，以及既有 timestamp 欄位），單位、gap、座標定義見 [DATA_CONTRACT.md](DATA_CONTRACT.md)。
- **直接共用 BAF 已建立的 cache**，確保 session split、train-only mean/std、causal history 與 test origins 完全相同。`prepare` 是既有 `baf_roi.data.prepare` 的另一入口。
- past-only 30 Hz / 30 poses，position 轉公尺後 normalize，現有 yaw/pitch/roll 轉 wxyz quaternion。
- 用 `OriginDataset(..., mode='all')` 訓練全部有效 next-step transitions，包含 unlabeled 和行為轉換。不依 Focused/Explorer 篩選，模型和 loss 不讀取行為特徵。train/valid 只需有有效 next-step targets，無須同時出現兩種行為。
- 原 CSV pipeline 的逐檔 normalization 改為 BAF 的 train-only 固定 normalization；seed 預設沿用 cache 的 training seed（通常 42）。模型和 loss 保留原碼。
- next-step=1/30 秒；recursive rollout 只把預測 pose 放入下一個 history，最多 8 steps。位置 LERP、quaternion SLERP 對齊 50/100/150/200/250 ms。
- 共用 `baf_roi.evaluate`，確保相同 origins、target 時間、ROI scene/FoV、partition ordering、mask packing 和 metrics。baseline 沒有 confidence/EMA/fusion：CSV `alpha_t` 空白；NPZ 不含 `alpha_t` / `rollout_alpha`。

## 執行

使用已安裝 `requirements.txt` 的 Python（需 PyTorch，不新增依賴）。在專案根目錄執行：

```bash
# 首次建立 cache 才需要；已有 BAF cache 就直接沿用。
python -m motion_transformer_roi prepare --config configs/default.json --output cache/experiment01

python -m motion_transformer_roi train --cache cache/experiment01 --output runs/mt_experiment01 --device auto
python -m motion_transformer_roi evaluate --cache cache/experiment01 --checkpoint runs/mt_experiment01/motion_transformer_roi.pt --output runs/mt_experiment01/test --device auto
```

`configs/default.json` 是範本：首次 prepare 前需填真實 dataset 路徑、確認 position unit 和 rotation convention。ROI evaluate 須有真實 scene geometry；若 BAF 指定 `--scene-config`，baseline 也指定同一檔案。

訓練可用 `--epochs 1` 做 smoke test，正式預設 30。`--batch-size` 預設 64；若 BAF 使用其他 batch size，可明確指定相同值。評估支援 `--split valid`、`--batch-size`、`--scene-config`、`--no-save-masks`。輸出目錄須為新的或空目錄。

訓練輸出 `motion_transformer_roi.pt`、`training_history.json`、`eligible_counts.json`、`data_manifest.json`。評估輸出 `metrics.json/csv`、`per_origin_metrics.csv`、`scene_snapshot.json`、`split_manifest.json`、`predictions/*.npz`，格式與 BAF 共用（confidence 欄位差異如上）。checkpoint 保存 cache ID、normalization、split、模型設定、training protocol 和 source hashes，評估拒絕不同 cache。

學長原始資料集的 checkpoint 不符合這份 normalization/data contract；請在新資料 cache 重訓。

## 線上推論

```python
from motion_transformer_roi.runtime import Predictor

predictor = Predictor.load('runs/mt_experiment01/motion_transformer_roi.pt', device='cpu')
result = predictor.predict_observations(
    timestamps_s=observed_timestamps,
    positions=observed_positions,       # checkpoint 記錄的 cm / m 單位
    rotations_deg=observed_yaw_pitch_roll,
    horizons_ms=[50, 100, 150, 200, 250],
)
# positions_m: [1,5,3]; quaternions_wxyz: [1,5,4]
```

history 需覆蓋至少 29/30 秒且只含當下已觀測的 poses；拒絕跨 recording gap 插值。`forecast_batch` 直接輸入為 `[B,30,6]`、position **m**、yaw/pitch/roll degrees。也支援既有 JSON history 格式：

```bash
python -m motion_transformer_roi predict --checkpoint runs/mt_experiment01/motion_transformer_roi.pt --history history.json --output forecast.json
```

指定 `--scene-config` 可額外輸出 ROI masks 和 partition IDs。

## 驗證

```bash
python -m pytest -q
```

新增測試檢查來源 SHA256、原 loss、quaternion sign 等價、recursive rollout、fractional horizons、cm/m 單位轉換、batch 一致性、推論不呼叫 supervision，以及合成 CSV → cache → 一個 epoch 訓練 → checkpoint reload → ROI 評估。整合測試逐列對照 BAF origin/target 時間，逐批核對 ground-truth ROI mask，另測 valid split、mask 開關和 cache mismatch 拒絕。

合成測試只驗證流程與比較條件；正式準確率需用真實實驗資料重訓和評估。

本次驗證：Python 3.13 / PyTorch 2.9.0+cpu，完整測試 **70 passed**。合成 test session 的五個 horizons 各有 204 個相同 origins，與 BAF 逐列時間及 ground-truth ROI masks 完全一致；命令列 `predict --scene-config` 和 `evaluate --no-save-masks` 也執行成功。系統預設 Python 3.14 未裝 PyTorch，執行時請選用已安裝本專案依賴的 Python 環境。GPU 和真實資料 accuracy 尚未測試。
