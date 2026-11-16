# Heatmap extension：實際驗證紀錄

## 0. 2026-11-16：共同 runner 與 per-step alpha 的驗證

本次修改以檢查前分支 head `20ad7510c8dd937668d35bcc66f818c5d132c462` 為基礎。以下是本次實際在 CPU 執行的驗證，不代表已完成真實資料訓練；GitHub CI 狀態應另以該次 commit 的 Actions 結果為準。

| 驗證 | 結果 |
|---|---|
| 完整 unit／regression suite | **202 passed，8 warnings**；warnings 來自環境 pytest plugin 及既有 Transformer nested-tensor 設定 |
| 原始來源 audit | 67 個 source files、4 個 verbatim functions，PASS |
| Runtime | Python 3.12、PyTorch 2.8.0+cpu、NumPy 2.3.5 |
| 四方法共同 runner | BAF、Motion Transformer、heatmap residual、heatmap per-step alpha 全部完成；沒有默默略過 methods |
| Synthetic 訓練 | BAF 四階段各1 epoch、MT 1 epoch、原四種 adapters 各1 epoch、新 alpha gate 1 epoch |
| 共同資料 | 每 horizon 181 origins，50／250／1000 ms，prediction margins 0／5／10／20°；每 method 2,172 rows |
| 公平性 identities | 四方法的 protocol、origin、target、scene、RF cohort 一致 |
| Standalone vs paired baselines | BAF／MT 的完整 origin／target identities 及所有 pooled ROI 指標一致；BAF不同 batch sizes 有約1.4e-8m／3.3e-6°的 pose metric 浮點差，不影響此 fixture 的 ROI |
| Alpha-only targeted tests | 17 cases：原 BAF 數值對照、可微 frozen rollout、真正逐步 fusion、future-RF invariance、state／gap／fallback／batch composition 與數值邊界 |
| Exact fallback | Residual 與 alpha 在額外2秒 RF delay 下均精確回到原 BAF 的 float64 poses 與 ROI metrics |
| Shared reporting | 空集合、nonempty union／GT、原 recording macro、RF 去重、policy delay／matched cohort 檢查有回歸測試 |
| 輸出圖表 | 8 張 horizon 圖 + 3 張 coverage–selected-count 曲線，共11張；各存PNG／PDF，預設 pooled，已檢視版面 |
| Recorded resource guard | 保留原5秒disk／RAM／RSS監控及GPU reserve，config／cache／prepared路徑與正常／例外停止皆有測試 |

新 alpha 零 correction 的 Torch rollout 在實際原 BAF models、50／250／1000 ms 下，套用精確 fallback **之前**，與原 runtime 的最大差異約 `3.58e-7 m`／`0.0004891°`。無 correction／RF缺測時直接恢復原始 float64 output，這些小數差不會變成零修正的 ROI gain。

Smoke 的 checkpoint 允許選 epoch 0；程式不強制選訓練後的非零修正，也不以 synthetic accuracy 宣稱 CSI 有效。CUDA 訓練／推論、真實20261031資料的完整新實驗，以及 rendered SSIM 尚未在此次環境驗證。

共同流程已納入原 `scripts/smoke_heatmap.py`，既有 GitHub workflow 會自動執行；不需要新的私人測資。

## 1. 版本與證據

原 baseline commit：`2a75637da2e814205a56484a1e195c2922d9f9c0`。

第一次完整 GitHub CI 驗證的 code commit：`7960cb955f15dc2614d6583b85a3371f6c2094f2`。

已完成且成功的 CI：
https://github.com/Rookieason/baf_roi_history6dof_v2/actions/runs/41449589473

該次 CI 的 audit、unit/regression tests、synthetic integration 與 artifact upload 都是 success。以下數據來自實際 logs / JSON，不是預估。

## 2. 第一輪 CI 結果

| 驗證 | 結果 |
|---|---|
| Original baseline provenance | 67 個 source files、4 個 verbatim functions，PASS |
| Unit / regression tests | 137 passed，8 warnings；warnings 來自原模型既有 PyTorch 行為 |
| Python / PyTorch / NumPy | GitHub Ubuntu runner，Python3.11、PyTorch2.10.0+cpu、NumPy2.4.6 |
| BAF 原四階段 | synthetic fixture 上每階段1epoch，完成 |
| 新 adapters | residual_hmd、residual_rf、expert_gate_hmd、expert_gate_rf，每個1epoch，完成 |
| Frozen baselines | BAF、Focused、Explorer、fixed0.5，完成 |
| Horizons | 50、250、1000ms |
| Matched origins | 每個 horizon181個，同一批 origins |
| 2秒額外 RF delivery delay | RF全部不可用；pose與ROI metrics精確回到原BAF |
| 單一 test cluster 的 CI | 正確輸出null，不製造confidence interval |
| 其他流程 | validation margin policy、hindsight oracle、batch-one benchmark、8張plots，完成 |

這些 synthetic recordings 不是人體量測，也不是同步 CSI 的研究資料。CI 中 map calibration=true 只因 simulator 使用已知 steering，不是現場硬體校正成功。

## 3. 後續 protocol 擴充

新增同一 prepared origins 的 Motion Transformer 評估，並加入「同一 raw CSI 經不同處理後不得跨 split」的檢查。擴充後本機 unit/regression tests 已實跑 **139 passed，8 warnings**；本機PyTorch2.10.0+cpu / NumPy2.3.5。完整 expanded integration 由相同 GitHub workflow 執行；以分支對應 commit 的 Actions 結果為準，不把尚在執行的工作當作通過。

可直接查看分支的 workflow runs：
https://github.com/Rookieason/baf_roi_history6dof_v2/actions/workflows/heatmap-tests.yml

## 4. 原 MUSIC.py 數值對照：實際執行

對照來源 SHA256：`5387d0df1f7dd04f30bd6e36d3a2e7b3f8835f267c7b196224168d5b7afb9f78`。

使用 `scripts/check_heatmap_reference.py` 對 `sin` 與原 `cos`（本版命名legacy_cos）各測一次，同一raw support：

| 項目 | 最大誤差／條件 |
|---|---|
| Smoothed covariance | 1.3322676295501878e-15 |
| Steering matrix | allclose atol1e-12，PASS |
| MUSIC pseudo-spectrum | 4.005684672847565e-13 |

這是 numerical core 的相容性驗證。原本 abs-based preprocessing、whole-record DWT、centered MA / windows 與 plotting 沒有被當成已驗證的 causal predictor pipeline。

## 5. 測試涵蓋哪些錯誤

- 未來 raw CSI 被任意改動，已結束 maps 不變。
- 未來 maps 被任意放大，當前模型輸入不變。
- raw-support超出history、尚未交付、stale、時計不確定性過大時排除。
- linear/dB契約、負dB、非有限值、複數CSI與實際frequency grid檢查。
- 原MUSIC的covariance排列、single-pathpeak、兩種projection。
- HMDcontrol不依賴RFpixels；last-frame不偷帶差分歷史。
- quaternion sign invariance、零rotationvector的有限gradient、修正向量限幅。
- 零residual／缺測時精確baseline；online exception rollback observed state。
- train-only normalization；subject/group、重複RF、raw-source跨split拒絕。
- scene預測margin不改GT；policy拒絕test tuning；comparison拒絕unmatched experiments。
- 原baseline與MT既有測試仍執行，不只測新增套件。

## 6. 如何重跑

```bash
python -m pip install -r requirements-heatmap.txt
python -m baf_roi audit
python -m pytest -q
python scripts/smoke_heatmap.py --output runs/smoke_verify_01
python scripts/check_heatmap_reference.py --reference-file /path/to/original/MUSIC.py --output runs/reference_verify.json
```

最後一個命令需要使用者原 `MUSIC.py`；不需要把全部私人附件重新公開到repository。其他測試不需要真實資料或未提供的checkpoint。

輸出目錄必須是新的。CI會把source snapshot、pytest/smoke logs和summary JSON上傳成artifact；repository本身不包含synthetictrainedweights、實際個人量測或巨大的資料cache。

## 7. 不能由這些測試宣稱的事

尚未有真實 paired HMD–CSI test accuracy、跨subject/room泛化、可靠人體部位辨識、SSIM、Mbps節省或完整30Hz系統證據。單一synthetic benchmark也不能推論headset上的速度。程式提供測量工具及邊界；論文的效益需要真實資料來回答。
