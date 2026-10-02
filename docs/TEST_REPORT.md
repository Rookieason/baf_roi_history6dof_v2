# v2 執行驗證報告

## 範圍

這是原碼對照、程式回歸與合成資料整合測試，不是使用者真實 ROI dataset 的準確率評估。附件沒有 raw session CSV 或完整 scene geometry。本次沒有聲稱重現論文數字、真實 ROI accuracy 或跨使用者泛化。

## 環境

Python 3.13.5；PyTorch 2.10.0+cpu；NumPy 2.3.5；SciPy 1.17.0；pandas 2.2.3；CPU。沒有 GPU 實測。

## 自動測試

`python -m pytest -q`：**65 passed**。

原有 46 項的覆蓋保留，其中舊 supervision 測試已改成作者函式介面；另外加入 19 個測試案例：

- 三組合成 trajectories，共 453 個 timestamp，逐欄位核對原 `score_one_index` 與 wrapper 結果；在同樣輸入下 exact equality（含 NaN 等價）。
- 另一支原 `csv_label_generator.py` 與原 supervision 的 labels 一致，數值欄位以 1e-12 浮點容差核對。
- 相同 causal history，但更換 centered window 的未來部分，可以改變 offline label。
- Irregular native timestamps 與前／後邊界的 centered adapter 對照。
- Exact-zero yaw rates = 正號、MIN_SAMPLES=5、time-weighted speed、rad/s q90、原始 debug 角度差分行為。
- Semantic yaw 欄位轉換；避免誤把 pitch 當 yaw，且模型 physical quaternion/input conversion 不變。
- Unlabeled／invalid target 排除、current/next metadata 對齊、pose target 與 v1 的一致性。
- Inference 將 supervisor 函式替換成「一被呼叫就報錯」，仍能正常輸出五個 horizons。
- 實際重現原 confidence reader 先丟 NaN labels 造成的 history gap；新版不丟歷史 pose。
- v1 cache 與 checkpoint 被明確拒絕。

## 原始碼稽核

`python -m baf_roi audit` 通過，核對 **67 個來源檔案**：

- 原 BAF-Pose：23 檔。
- 新 data_preparation：20 檔，包含 9 份原作者圖表 PDF；保存不代表拿它們當新結果。
- 舊 ROI framework 參考：23 檔。
- v1 supervision 封存：1 檔。

另核對 4 個逐字保留的 weighted BCE／EMA／fusion／SLERP 函式。三個 model.py 的位元組均未修改。

## 完整 synthetic smoke test

三個 synthetic sessions，每個 720 筆 90 Hz recorder-format poses；明確分成一個 train、一個 valid、一個 test。兩個合成物件、每物件 4³ partitions，不是使用者的真實場景。

執行 prepare → Confidence 1 epoch → Focused pretrain 1 epoch → Focused fine-tune 1 epoch → Explorer 1 epoch → checkpoint reload → recursive rollout → ROI evaluation → history-only prediction。

每份資料 211 個 causal input windows；208 個有效 confidence labels；210 個 one-step targets；97 個 Focused clean transitions；108 個 Explorer clean transitions。Test 有 **204 個共用 origins × 5 horizons**。這些是測試樣本數，不是準確率。

## 額外驗證：test labels 不影響 prediction

複製 disposable cache，將 test 的 labels/next_labels 全部改成 -1、scores 全部改成 NaN，並刪除 test 的 supervision_metadata.npz；相同 checkpoint 重新評估，**所有 metrics 與原結果完全相同**。

這個故意破壞 cache 的操作只用於回歸測試，不是合法新實驗；正常使用必須保持 cache 不可變。結果支持模型與 test-origin selection 沒有使用 oracle supervision。

## 額外驗證：batch size

相同 checkpoint，evaluation batch size 16 與 7：五個 horizons 的 n_origins、TP、FP、FN、TN、exact match 完全相同。位置 mean error 最大絕對差小於 1.6e-8 m；旋轉 mean error 最大絕對差小於 3.6e-7 deg。此差異屬 batch 浮點運算；EMA 狀態沒有被 batch 邊界重設。

## v1/v2 labels 對照

對同一組合成 input origins，633 個標籤中有 96 個不同；每條 211 origins 有 32 個不同。v1 每條 F/E/U 計數 100/109/2，v2 為 99/109/3。總類別比例接近，不代表每個時間點的標籤一致。

這只說明 centered supervision 的修正會影響訓練樣本，不可外推為真實資料的標籤差異比例或預測改善。

## 證據檔案

`validation/pytest.txt`、`audit.json`、`smoke_log.txt`、`smoke_result.json`、`extra_validation.json`、`supervision_comparison.json`。v1 舊證據移到 `validation/v1/`，沒有當成 v2 重跑證據。

測試用模型權重不隨 ZIP 交付，以免混淆成真實資料模型。仍需使用者提供並確認 raw data、單位／座標、scene geometry，重新跑正式的 10/8/12/20 epoch 上限流程。

## 重跑額外驗證

先執行 smoke test，再指定新的回歸輸出目錄：

```bash
python scripts/smoke_test.py --output smoke_output/v2
python scripts/verify_smoke_regressions.py --smoke-dir smoke_output/v2 --output smoke_output/v2_regressions
```

回歸程式只接受有 SYNTHETIC 標記的本專案 smoke 產物；故意破壞標籤時先複製 cache，完成後刪除 disposable copy，不修改原 cache。交付另附 `V1_TO_V2_CODE.diff` 供逐行檢查執行程式變更；完整更新請使用 ZIP，而非只套該差異檔。
