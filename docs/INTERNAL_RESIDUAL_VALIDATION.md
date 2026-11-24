# Internal CSI residual：軟體驗證紀錄

日期：2026-11-24。分支：`feature/csi-internal-residual`。基底：`e34a83687390771b25438874b78f164db1a99b47`。

## 實際執行結果

| 驗證 | 結果 |
|---|---|
| 完整 `python -m pytest -q` | **244 passed**，35.26 秒；8 個既有環境／Transformer warnings，沒有失敗 |
| 新方法的專門測試 | **23 passed**，包含在上述 244 項內 |
| `python -m baf_roi audit` | **PASS**：67 個保留來源檔、4 個 verbatim functions |
| 全方法 fresh synthetic smoke | **PASS**：由合成 HMD／CSI 開始，BAF 各階段一個 epoch，包含五種方法的共同評估 |
| 內部 residual synthetic smoke | **PASS**：重用前項的合成 prepared，訓練一個 epoch，checkpoint 重載、重評與線上單次 rollout |
| Shell launchers | `bash -n` 通過 |
| 20261031 主方法 dry-run | 正確選取 BAF／MT／internal residual，50–1000 ms horizons |
| `git diff --check` | 通過 |

環境為 Python 3.12.14、PyTorch 2.15.0+cpu、NumPy 2.3.5，CPU，主要 thread limits 設為 2。沒有 GPU，因此未實測 CUDA／cuDNN。

機器可讀的執行結果保存在 [internal_residual_validation.json](internal_residual_validation.json)。這些結果是 **SYNTHETIC SOFTWARE INTEGRATION ONLY**，不代表真實研究資料的 ROI 改善。

## 測試具體驗證了什麼

`tests/test_heatmap_residual.py` 使用原作者的實際 expert modules，以及少量可追蹤的 fixture。測試並非只確認 tensor shape：

- 在原 translation head 的入口觀察 latent，確認 residual 在解碼之前加入；第一步 HMD input 一樣，下一步 history 已反映改變後的姿態。
- 確認新模型沒有 completed BAF endpoint 輸入；資料 batch 白名單不傳 targets 或 `base_p`／`base_q`。
- 對實際原 BAF 模型驗證 50／250／1000 ms 的零 residual 數值一致性；因 Torch／NumPy 幾何運算的浮點差異，內部計算使用位置 `1e-5 m`、旋轉 `<0.01°` 的檢查界限。正式 zero-initialized／missing-RF fallback 使用原 BAF 輸出，另以 exact equality 檢查。
- 在 frozen BAF heads 之後取得訓練梯度，並只監督最後 horizon，確認第一步的 residual 仍能收到梯度。更新後原 BAF state_dict 保持完全一致。
- RF 內容改變時，原輸出 head 收到的 latent 與預測會改變；每個 forecast 只呼叫 RF encoder 一次。
- 各步 feature residual 的相對幅度不超過設定值；shared horizons 與 horizon subset 的預測一致。
- 未來 RF maps、masked maps／timing 的 NaN 不會影響當前輸出；mixed batch 的缺 RF row 不受其他 row 影響。
- 有效 RF 的線上路徑不呼叫原 BAF 完整 forecast；缺 RF／零初始化則使用原 BAF 的替代路徑。
- Observed EMA 與原 BAF 相同，跨 gap 重設，失敗不提交狀態，錯誤 checkpoint identity 被拒絕。
- HMD residual-head control 不使用 RF 內容／timing，RF 缺測時仍能執行其訓練過的 residual。
- 非零模型對特定 RF context 恰好產生零 residual 時，線上仍只跑一次；離線使用相同 Torch 輸出。另一個 RF context 能啟用同一組參數，證明此測試不是零初始化的重複。

## 合成整合測試的範圍

`scripts/smoke_heatmap.py` 從合成 HMD 與已知 steering 的 synthetic CSI 開始，訓練 baseline、建立因果 RF／prepared 資料，再執行舊比較組、alpha、新內部 residual 的共用流程。此次使用 **181 個 matched test origins**、**50／250／1000 ms**，並驗證共同評估身分。執行時加上 `--skip-plots`，因此此次全流程 smoke 的 plot count 為 0；圖表相關單元測試已包含在完整 pytest。

`scripts/smoke_internal_residual.py` 重用該合成 prepared，使用 hidden=8、bottleneck=8 的小模型跑一個 epoch；不論主要 checkpoint 是否保留 epoch 0，都重載 `best_trained.pt` 以實際走過 trained residual 路徑。此次選到 epoch 1，但這只表示合成 fixture 上的選模結果。

內部 smoke 實際確認 checkpoint 重載、共同 evaluator、原資料的重評、online/direct-model 對齊、延遲 2 秒 RF 時的 exact baseline 回退，以及錯誤 RF grid 不污染 observed state。

## 可重現命令

```bash
python -m pip install -r requirements-heatmap.txt
python -m baf_roi audit
python -m pytest -q

python scripts/smoke_heatmap.py \
  --output runs/internal_validation_all --skip-plots

python scripts/smoke_internal_residual.py \
  --prepared runs/internal_validation_all/prepared \
  --output runs/internal_validation_online
```

`--output` 需使用新的空目錄。也可以不傳 `--prepared`，讓 internal smoke 自行生成一份合成資料。GitHub workflow 已加入新 branch 與 internal smoke；上述結果是本次本機實際執行的結果，不能當成尚未回報的遠端 CI 結果。

## 尚未取得的研究證據

尚未執行 20261031 真實 recording 的完整訓練，沒有新增真實 test ROI／F1／coverage 改善數字，也沒有 rendered SSIM、端到端 30 Hz 或 CUDA 的實測結果。模型的 feature residual 幅度限制不保證最終姿態誤差或視覺品質。這些需依 [設計文件的對照實驗](README_INTERNAL_RESIDUAL.md) 在真實資料與目標硬體上取得。
