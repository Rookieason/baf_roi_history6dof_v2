# 所有邏輯改動與原版差異紀錄

## 紀錄原則

原始來源是使用者提供的 `baf_pose.zip`、後續補上的 `data_preparation.zip` 與《力行論文(4).pdf》。主路徑以 **各模組 run.sh 的實際參數**為準，不把 factory function 的 optional default 誤當成實驗設定。論文與程式不同的地方直接列出，不默默把程式改成另一種方法。

`vendor/baf_pose/` 為原始 23 檔的完整副本；三個 model.py 直接載入。`provenance.json` 是逐檔 SHA-256 與逐字函式的驗證資料；`python -m baf_roi audit` 可檢查。`reference/roi_baseline/` 保留舊框架 23 個非 outputs 檔，不是執行中的預測器。

**「模型檔未改」不等於「整條流程完全沒改」。** 下列記錄涵蓋 preprocessing、訓練資料選取、推論、幾何、evaluation 與介面的差異。

## L01：全檔 normalization → 固定 training-only normalization【必要的實質改動】

原始 `prediction/focused_branch/utility.py`、`prediction/explorer_branch/utility.py` 與 `decision/utility.py` 對每個 CSV 的完整 positions 計算 mean/std；decision 推論也使用整份測試 trajectory。對線上時刻 t 而言，這依賴 t 之後的位置分佈，與只有 history 的要求不符。

新版本：`baf_roi/data.py::Moments` 只讀 train sessions 中保留片段的有效 raw positions，計算 global mean 與 population std，加上原版的 `1e-6`；保存於 cache 與 checkpoint。Focused、Explorer 共用它，validation/test/online 都不重新 fit。Confidence 仍使用 raw position 與 Euler 六個數值，沒有新增 normalization。

這會改變 branch 的 normalized translation residual 尺度、L1 loss 的隱含軸向權重與輸入分佈；不能宣稱 accuracy 或 pretrained weights 可直接相容。沒有將任何原 dataset checkpoint 冒用到新資料。選擇固定 train statistics，是為了避免再引入 window-dependent normalization/state，並明確排除 future leakage。

## L02：行為 supervision 改為直接使用作者原碼【v2 修正，取代 v1 的重建】

v1 在只有論文、沒有 generator 時，採 past-only 30-pose scoring，並移除 exact-zero yaw rates。新附件證實這兩點與原碼不同；v2 已撤換該實作，封存於 `reference/v1_supervision.py`，不在主路徑。

v2 原樣保存並直接 import `vendor/data_preparation/supervision.py::score_one_index`，使用原作者 `[t-0.5,t+0.5]` centered window、`MIN_SAMPLES=5`、zero sign → +1、time-weighted speed、actual duration、rad/s q90、原門檻及 Unlabeled semantics。所有 score/helper 的原始位元組保留；沒有依新資料重新調 threshold。

Confidence 的 input 仍只有 `[t-29/30,...,t]`。Centered label 的未來部分只屬於離線訓練／驗證 target，不進入 inference，不作 oracle alpha。下一步 clean label 以 `t+1/30` 為中心，Focused/Explorer 仍採 `Label[t]` 與 `Label[t+1]` 同 regime 的篩選。所有有效 pose-only transitions 仍供 Focused pretraining。

原 supervisor 的第二角度欄位是 semantic yaw。新 scoring-only adapter 把 recorder `[yaw,pitch,roll]` 排成 `[pitch,yaw,roll]`；模型的 physical xyz/quaternion adapter 不變。v1 的 actual-observed-origin 時間協定仍保留，scoring grid 重新以該 origin 對齊；不宣稱整條前處理与原 flat CSV 逐位相同。

完整來源行號、邊界、時間差異、units 及測試見 **[PREPARATION_REVIEW.md](PREPARATION_REVIEW.md)**。現在不再有「原 generator 未提供，無法核對算法」的限制；仍有不同 dataset 的資料／座標／時間 adapter 差異。

## L03：新增 recursive rollout 執行器【沿用論文；非新的預測模型】

原 ZIP 的主要 inference 是 next-step；沒有附第 6.2.4 節的 multi-step script。`runtime.py::forecast_batch` 按 Figure 23 重建該程序：每次預測後移除最舊 pose、append predicted pose，長度維持 30，再重新計算 Confidence 與兩分支。

所有 horizon 共用同一條 rollout prefix。沒有 teacher forcing，沒有在第 2 step 讀取真實 t+1 pose，沒有重複呼叫「真實未來 history」冒充多步。回填時把 quaternion 轉回共同 Euler pose 介面，下一輪沿同一套資料 adapter 產生 branch quaternion；沒有額外開啟 quaternion sequence sign continuity。

訓練依舊 one-step。沒有 direct horizon head、joint multi-horizon loss、scheduled sampling、long-horizon fine-tuning、線性放大一次 residual，或新增 encoder。誤差可能隨 rollout 累積。

## L04：指定毫秒的 fractional-step interpolation【必要新增】

保留 30 Hz、不改成 20／100 Hz，避免順便改動 model dynamics、history 長度與 supervision thresholds。所需 horizon 並非全部落在整數 frame。

| Horizon | 使用的 predicted steps | 權重 |
|---:|---|---:|
| 50 ms | 1、2 | 0.5 |
| 100 ms | 3 | 0 |
| 150 ms | 4、5 | 0.5 |
| 200 ms | 6 | 0 |
| 250 ms | 7、8 | 0.5 |

`pose.py::interpolate_predictions` 對位置用 LERP，對旋轉呼叫原版 SLERP。只在預測結果間插值，從未用真正 future pose 補齊輸出。不把請求 horizon 四捨五入；也支援 [50,250] 內其他不重複毫秒值。

## L05：history resampling 與 decision origin 的 causal 定義【必要協定改動】

論文使用 30 Hz、30 poses；舊 ROI 使用 250 ms 內的 8 個索引樣本。新版本保留 BAF 的 30 Hz／30 poses，而不是延用旧框架 8 samples。

直接把整份 irregular recording 先線性 resample，再拿某個 grid time 當 t，可能讓該點的插值使用 t 之後才量到的 raw pose。新 `Series` 的 origin 是實際 observed row：先建立約 30 Hz 的決策 ticks，再選 tick 之後第一筆已記錄的 raw row，以該 row 的實際 timestamp 定義 t。history grid 往回取 `t-29/30 ... t`，所有 interpolation support 必须不晚於這個 row。

位置仍線性 interpolation；角度仍 unwrap → interpolation → wrap，沒有改成新的 smoothing filter。`np.unwrap` 是 prefix-causal；测试證明改變 t 後的資料不影響 input。原始資料不規則時，決策間隔會有 native sample jitter，EMA 仍是原版「每個決策一次」的係數，而不是另改 time-aware filter。報表使用實際 origin time；不可稱 origins 完全等間隔。

30 samples 的時間跨度是 29/30 秒，名義 history budget 約一秒。history 不足不能 padding 或复制當前 pose 充數。

## L06：時間標籤改成精確 t+h，不再 nearest-within-40ms【必要 evaluation 改動】

舊 `timestamp_alignment.py` 找距 t+h 最近、誤差不超過 40 ms 的 row。短 horizon 下，此規則可能混入不同實際預測時間。新 target 是 raw trajectory 在精確 `t+h` 的線性／unwrapped-Euler interpolation，超出可量測範圍或跨過大 gap 就剔除，不做 extrapolation。

target interpolation 可使用 t 之後的 raw observations，因為它只是在建立離線 ground truth；這些 observations 永遠不進入 history。這不是保證感測器真的在該微秒有量測，而是明確定義的 interpolated ground truth。

五個 horizon 使用同一組具完整 future coverage 的 origins；不讓 250 ms 只拿到較容易的子集，再與其他 horizon 的不同樣本比較。

## L07：session-level split 與防洩漏檢查【dataset protocol 差異】

原 BAF 為 file-level 70/15/15、paper seed 57；舊 ROI 為 session-level、seed 7。新預設 session-level 70/15/15、seed 57，先 split 再 fit statistics。同 session 的所有 CSV／gap segments 必須在同 split，避免 shard leakage。

可以指定舊實驗的 `train/valid/test` manifest，這是跨 baseline 公平比較的建議方式。缺少、重複、互相重疊、空 split 都報錯。沒有舊 runner 的 `test or valid` fallback。Session split 不等於 subject split；不能因此聲稱測過 unseen-user generalization。

## L08：缺值、重複、gap 與 CSV 邊界【資料品質協定；會改變可用樣本集】

舊 reader 可忽略 malformed rows，甚至把 missing rotation 補成零；舊 discovery 只拿第一個 agent6 directory／CSV。新 discovery 讀到的每個 agent6 CSV 都是獨立 trajectory，沿用 session grouping，不任意拼接可能來自不同串流的檔案。短 shards 沒有足夠 history 時會列入 excluded segments。

新 reader 對 malformed payload、缺角度、非有限值與衝突的相同 timestamp 直接報錯。完全相同的 timestamp/pose 可以 deduplicate，並記錄數量。gap 超過 `max_gap_ms`（預設 100 ms）切段，window、target、EMA 都不跨 segment。沒有填造缺失 pose。

這些檢查會讓某些舊版曾納入的樣本不再可用；必須讓其他比較方法用同一協定。Recording 的 units 不自動猜測，沒有 timezone 的 ISO timestamp 不默認成機器所在時區。

## L09：EMA 真實狀態與 forecast 狀態隔離【新 multi-step/online 實作細節】

保留 `alpha_t = 0.6 * raw_alpha_t + 0.4 * previous_alpha`，第一筆以自己的 raw alpha 初始化。批次 evaluation 直接使用原版 `compute_temporal_optimized_alpha`，依真实 origins 時間順序更新。

原版 state key 是 basename，對不同 session 同名 `head.csv` 可能混淆。新版本使用唯一 trajectory key，且 gap／file 重置，online 使用明確 stream_id。rollout 分支只複製當前狀態向未來推進，絕不把第 8 step 的 alpha 寫回下一個實際 observed origin。否則下一次真實預測會被上一次虛擬未來污染。

`forecast_batch` 的 initial_alpha／previous_alpha 是內部介面，只能來自已觀測 history；不是允許塞 oracle labels 的額外感測 input。保留 EMA 也代表狀態可間接摘要早於最近 30 frames 的歷史，但其來源仍全部是過去 6DoF。沒有更改原 EMA 公式或做硬 routing。

## L10：pose → ROI 是新增下游模組，不是 BAF 原本的學習輸出【必要新增】

論文第 6.3.4 節明確說未把 pose predictions 轉成 viewport／tile selection。新 `roi.py` 才是這個研究任務的下游 mapping。網路只接受 history；幾何只在 predicted／ground-truth pose 後套同一函式。

預設 `legacy_center_cone` 保留舊 `roi_label.py` 的語義：對每個 partition center 做圓錐視角判定。已測過位置 cm→m 後與舊版等價。它不應被稱為完整矩形 frustum 或「該 partition 所有點實際可見」。

選配 `rectangular_center_frustum` 新增相機 local coordinates 中的水平／垂直 FoV、near/far test，會使用 roll。這是明確標記的替代 ROI 定義，必須知道相機校正數值並讓所有 baseline 同時切換；不是替 BAF 偷換更有利的 GT。

兩種模式都只測中心，不含 occlusion、AABB／mesh intersection、occupied points、實際點雲 partition bounds、左右眼非對稱投影。以 object center + uniform cube 生成 partitions 需要 explicit acknowledgement；非零 object rotation／非單位 scale 不默默忽略，請改用真實 world-space centers。

## L11：座標與單位轉換【資料 adapter，但影響物理意義】

旧 ROI 的 camera-forward 數學為 `Rz(yaw) Ry(-pitch) Rx(roll) @ [1,0,0]`，不是把 [yaw,pitch,roll] 原樣交給 SciPy `from_euler('xyz')`。新 adapter 把 Euler 換成 `[roll,-pitch,yaw]` 後交給相同的 xyz quaternion 表示；Confidence 的六通道也採一致的 xyz Euler 介面。

positions 依明確設定的 cm／m 轉成公尺，因為 paper 的 speed threshold 是 m/s，pose metrics 也以 m 表示。Branch input 的 7 scalars 是 xyz + unit quaternion，不是第七個自由度或新 modality。

這不是證明 recorder 一定用了舊 ROI 的 convention；實際 recorder／世界座標校正未附，所以要使用者確認。沒有擅自估計 alignment transform、以未來 trajectory 做 spatial recentering 或加入 train-time scene embeddings。

## L12：重寫資料框架與執行入口【工程改動；保留訓練數學】

原版 utilities 會把所有窗口展開到記憶體，並強依賴已處理 flat CSV 的欄位；旧 ROI discovery 又綁定 heatmap。新版本用每條 trajectory 的 NPY/mmap、compact integer indices 與 bounded cache 供樣，不讀 MAT／PNG。

Training 仍 shuffle 全部 eligible windows，沒有改為 trajectory-balanced sampling 或重新加權長短 sessions；batch size 預設 64、drop_last=False、workers=0。三個模型／原 loss 函式未更動。每階段重設原 train.py 的 seed 42。

保留 Confidence Adam LR 1e-3、無 scheduler／clip；保留 branches AdamW decay=1e-4、clip=1、plateau factor=.5/patience=2。Focused pretrain 8 epochs/early stop 4、fine-tune 12/5，Explorer 20/5，Confidence 10。Focused fine-tune 從最好的 pretrain checkpoint 起跑。

只用 validation loss 選 best；不移植與本任務無關的「掃描 test-set threshold 後報最佳 threshold」流程，fusion 仍用 continuous confidence 而不是閾值分類。載入 pretrained stage 改為 strict=True，避免漏載參數卻無聲繼續。輸出資料夾已存在時不覆寫。`--smoke-epochs` 是明示的測試覆寫，必須在結果標註，不能混稱原版正式 protocol。

## L13：evaluation 與輸出格式【新增；不是新的訓練 objective】

舊 ROI 的 single-class Top-k 不適合目前 multi-label task。報告新增 exact match、per-cell accuracy、micro precision/recall/F1/IoU、mean window IoU、非空 union 的 mean IoU，以及 empty-mask 計數；保留各 horizon 的 pose Euclidean error 與 SO(3) geodesic angular error。

為降低小角度計算的數值誤差，evaluation geodesic 用 `4 atan2(||q1-q2||, ||q1+q2||)`（先處理 q／-q），與標準 `2 acos(abs(dot))` 數學等價。**沒有改動原模型 TRAIN loss 中的 acos clamp 或其非零 numerical floor。**

兩 mask 都空時 IoU 設 1，同時另外報空集合數與排除 empty-union 的平均；precision/recall/F1 的零分母設 0。不能只拿 per-cell accuracy 說模型好，因為大量 negative cells 會讓它偏高。

Latency 是含 confidence／EMA／完整 rollout 的 batched amortized wall time，不含 I/O、ROI mapping；有 GPU synchronization。它不是原論文 batch-size-one、one-step neural latency，不能直接對比。輸出附 exact origin time、horizon、scene snapshot、partition ID order、split、原 CSV SHA-256 與 cache_id。NPZ 使用 packbits，還原時必須用記錄的 num_partitions 去除 padding bits。

## 特別容易誤判、但本版本沒有改動的地方

- Explorer factory 的 aux_dim default 是 4，但原 `run.sh` 不傳 `--use_aux`，train.py 的 store_true default 為 False。因此原執行路徑就是 **aux_dim=0**；本版本沒有擅自刪除一個已啟用的分支 input。
- 原 Confidence run.sh 明確用 Focused weight=3／Explorer weight=1，而論文公式寫一般 BCE。這裡保留 run.sh，不偷偷換成 unweighted BCE。
- 原 decision_fusion.py 的 CLI default 與 run.sh 不同；本版本保留 run.sh 的 temporal_optimized、mu=.6、init=alpha、full_slerp，不套 alpha calibration 或 fixed averaging。
- Focused／Explorer translation residual、左乘 quaternion residual、normalization、sign alignment、geodesic loss、Focused smoothness、Explorer tail weighting 都直接使用原 model.py。
- 沒有額外 modality、沒有新 attention module、沒有 ROI BCE 訓練 head，也沒有為追求測試分數而調整原權重／loss／architecture。

## 必須保留的研究表述

適合稱為「BAF-Pose 的 history-only、causally normalized、recursive-rollout adaptation，接上固定的 ROI geometry」。不適合稱為「所有 preprocessing 與作者完全相同的原版數字重現」，也不能用本次合成測試宣稱已在使用者真實 dataset 上提升 ROI accuracy。
### 跨 baseline 比較的額外提醒

這份 BAF adaptation 保留約一秒／30 poses 的 history，而舊 ROI 設定是 250 ms／8 samples。若要把差異歸因於 CSI 是否提供額外資訊，其他 baseline 與 proposal 的 history budget、sampling、split、target alignment 與 ROI geometry 也應統一。不能只統一 future horizon，卻讓各方法使用不同的過去資訊量。本版固定保留原 BAF 設定；要縮短它的 history，應另立實驗並記錄變更。

## L14：保留 Unlabeled 的歷史 pose【v1 已有，v2 補記原碼差異】

新 generator 對 Unlabeled 寫 binary_target=NaN。原 confidence reader 先 dropna 再建視窗，會把這些 pose rows 也刪去；其 30 rows 不一定仍是連續 30 Hz。新專案只排除不明確的 supervision target，保留完整歷史 pose。這使 training 與 history-only inference 使用一致的時間窗口。專門回歸測試已重現原 reader 的缺口。沒有把 future-derived Label 送入模型來決定哪些歷史 pose 可看。

## L15：v2 cache 與 checkpoint provenance【工程相容性防護】

Cache/統一 checkpoint format_version 改為 2，保存 author-centered protocol 及 source hash；舊格式明確拒絕並要求重建／重訓。每條 trajectory 另存 `supervision_metadata.npz`，含 current_ / next_ 前綴的原作者數值欄位，僅供稽核，不是模型 feature。訓練 optimizer、loss、early stopping、fusion 及 horizon math 均未因版本更新改動。

新增 pandas 依賴，只因直接 import 原作者檔案。Vendor CLI 的資料搬移／清空 split 操作不會被新 pipeline 執行。原 `%.6f` CSV round-trip 不套用到 NPY cache；這個數值序列化差異已列在 PREPARATION_REVIEW。

## v1 → v2 的範圍

本次修的是 supervision generation 與來源／格式防護，不是重設 BAF-Pose 架構。原有 L01、L03–L13 的理由仍適用。測試与合成資料結果以目前 TEST_REPORT 為準；原 v1 測試輸出封存在 `validation/v1/`，不冒充 v2 結果。
