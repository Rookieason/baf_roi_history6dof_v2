# 原始來源閱讀與對照紀錄

## 來源與可信範圍

v2 的方法來源是使用者提供的四份附件；前次交付 ZIP 另作變更稽核來源。SHA-256 與檔案大小記於 `source_archives.json`。沒有把其他網路版本或後來的 repository silently 取代附件。

| 附件 | 本次使用方式 | 缺少的必要外部項目 |
|---|---|---|
| roi_baseline-main (2).zip | 辨識 recorder schema、session 組織、現行 ROI target 與幾何 | raw session CSV、external adaptive scene config、真實 partition geometry |
| baf_pose.zip | 直接保留並載入模型與 loss；對照 run.sh、training、decision pipeline | 原 dataset、trained checkpoint、multi-step experiment script（generator 已由新附件補齊） |
| data_preparation.zip | 原樣保存 20 檔，直接呼叫 supervision.py；查核 centered labels / units / boundaries | 沒有我們的真實 session CSV 或已訓練模型 |
| 力行論文(4).pdf | 對照公式、training protocol、proxy supervision、recursive rollout | 不是新 dataset accuracy 的證據 |

## 新補齊的 preparation source

**本次最重要差異及原始行號見 [PREPARATION_REVIEW.md](PREPARATION_REVIEW.md)。** 原 source 用 centered 1 秒 supervision；exact zeros 視為正號；`HeadRY` 是 semantic yaw；q90 儲存 rad/s；Unlabeled target 是 NaN。不能再把 v1 的 history-only label reconstruction 宣稱為原碼。

## 舊 ROI framework 的實際狀態

`baseline.md` 描述過 single-class、future HMD position → world voxel 的早期 target；但 `REVISE.md` 與目前 `data/roi_label.py` 已改成「虛擬物件的哪些 partitions 在 future FoV 內」。不能把早期文件當成目前 ground truth。

`data/discover_sessions.py` 仍綁 artifacts、heatmap MAT／figures，且選第一個 agent6 directory／CSV。`timestamp_alignment.py` 能從 payload_json 讀取 pose，但會把 missing rotation 補零；history 取索引 samples，future target 使用 nearest-within-tolerance。

`data/roi_label.py` 合併 default 與 stream object transforms，使用 global object_size_cm 與 grid_dim 建立 uniform centers。它的 FoV 判定是以 forward vector 為軸的 cone，不是完整 rectangular frustum。測試會將新相容模式和這支原始函式做 golden comparison。

`models/baseline_6dof.py` 與 `train.py` 是直接 ROI multi-label baseline，不是 BAF-Pose 的 next-pose loss。沒有為了沿用框架而把 BAF 改成 ROI classifier。新 runner 也不沿用舊 `test or valid` fallback。

舊 outputs 中的圖、metrics 或 log 不作為本次 BAF 結果，也沒有把它們改名交付為新實驗。

## BAF 三個模組的實際架構

| Component | 原始來源 | 有效設定 |
|---|---|---|
| Confidence | confidence/model.py + train.py + run.sh | 6-channel history，CNN 32/64/128，kernels 5/5/3，BatchNorm、ReLU、Dropout、global average pooling、FC hidden 64、sigmoid |
| Focused | prediction/focused_branch/model.py | 7-channel position/quaternion，2-layer LSTM hidden 128，LayerNorm，translation／quaternion residual heads |
| Explorer | prediction/explorer_branch/model.py + train.py + run.sh | 7-channel pose，3 TCN blocks，dilations 1/2/4，每 block 兩個 causal convolutions，hidden 128、kernel 3、residual heads |

採原始 run.sh 設定的參數數量：Confidence 44,545；Focused 236,423；Explorer 330,631；合計 611,599。這是由交付原始程式實際建立模型後計數，並非從論文小數四捨五入反推。

Focused 和 Explorer 都輸出 normalized translation residual 與相對 quaternion；左乘當前 orientation 後正規化、與 current quaternion 對齊 sign。兩分支的 train loss 函式直接使用原檔，不另寫近似版本。

## 三個值得注意的 paper/code 或 code/default 差異

1. **aux_dim**：Explorer model factory default 是 4，但 train.py `--use_aux` 預設 False，run.sh 沒有傳這個 flag。所以交付使用 aux_dim=0；沒有新增 motion feature 輸入，也沒有從已啟用路徑刪掉 features。原 utility 雖會讀四個 auxiliary columns，但不代表它們一定進模型。
2. **confidence BCE**：論文第 5.1.4 節寫一般 BCE；實際 run.sh 設 Focused weight 3、Explorer weight 1。這次採後者，並逐字保留 weighted_bce_loss，不聲稱兩份公式完全相同。
3. **fusion mode**：decision_fusion.py CLI 的預設 calibration 不等於 run.sh 的實驗設定。這次採 run.sh 的 temporal_optimized、mu 0.6、init=alpha、full_slerp。

## 下一步與 multi-step 的對照

論文第 4 章、5 章與主訓練目標是 next-step。第 6.2.4 節、Figure 23（印刷頁 65／PDF 第 73 頁）另做 recursive rollout：drop oldest、append prediction，持續更新 confidence/fusion，30 steps 對應一秒。因此使用者「只有下一 frame」的理解對主要 learning target 是對的，但不涵蓋這版論文後面的 multi-step 實驗。

這次保留 one-step learning，加上論文描述但 ZIP 未附的 rollout 執行器。50／150／250 ms 的 fractional extraction 是這次新增的時間介面，不假裝原 code 已支援這五個整數毫秒。

## 論文主要對照位置

- 第 3.1.2 節：30 Hz resampling、linear scalar／unwrapped angle interpolation。
- 第 3.2.1–3.2.2 節：behavior score、threshold、normalization references。
- 第 5.1.1–5.1.4 節：proxy supervision、CNN confidence、EMA、BCE。
- 第 5.2.1 節：Focused recurrent encoder、residual reconstruction、smoothness。
- 第 5.2.2 節：Explorer TCN、rotation-tail weighting。
- 第 5.3 節：translation LERP、quaternion SLERP。
- 第 6.1.1、Table 4：history length、split seed、四階段 training settings。
- 第 6.2.4、Figure 23–24：recursive rollout、隨 horizon 增加的誤差。
- 第 6.3.4 節：原研究沒有做 viewport／tile mapping，也限制了 generalization 與 proxy label 的解釋。

## 新檔案分工

| 檔案 | 職責 |
|---|---|
| config.py | 設定讀取、合約檢查與 JSON metadata |
| data.py | 原記錄 reader、causal interpolation、session split、training-only stats、cache 與 mini-batch dataset |
| pose.py | 正確的 ypr/xyz/quaternion 介面、horizon interpolation、rotation metric |
| supervision.py | 直接呼叫原作者 centered supervisor，與 causal model input 隔離 |
| models.py | 載入三個原始 model.py，不包裝成另一種架構 |
| upstream_ops.py | 原版 EMA／SLERP／fusion／weighted BCE 的逐字函式 |
| train.py | 四階段 optimizer、validation、early stopping、checkpoint |
| runtime.py | 無 target 的 online prediction、recursive rollout、state isolation |
| roi.py | 固定場景 partition centers 與幾何 membership；不含 ML |
| evaluate.py | 所有 horizon 共用 origins，metrics、NPZ、scene snapshot |
| audit.py | 原始位元組與逐字函式驗證 |
| __main__.py | prepare/train/evaluate/predict/audit CLI |

## 原始碼權利聲明

附件內沒有獨立 LICENSE 檔。本專案保留原始碼與來源做使用者指定的研究移植／稽核，不替原作者宣告任何開源授權，也不保證可任意公開重新散布；公開 repository 前應確認原作者授權與引用方式。