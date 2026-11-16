# Heatmap 只輔助 BAF α：`baf_roi_heatmap_alpha`

這個方法用一個很小的 Heatmap 模組，調整 **BAF 每一步預測中真正使用的 Focused／Explorer 融合權重 α**。原本的 Confidence、Focused LSTM、Explorer TCN 全部凍結；位置與旋轉仍由這兩個原始專家產生。

研究問題是：在既有兩個專家不變的條件下，當下可取得的 CSI Heatmap，是否能幫助系統決定「這一步應該多相信哪個專家」，進而改善 future ROI？

主要程式位於 [model.py](../baf_roi_heatmap_alpha/model.py)、[train.py](../baf_roi_heatmap_alpha/train.py) 與 [runtime.py](../baf_roi_heatmap_alpha/runtime.py)。所有方法的正式比較由共同的 [evaluation.py](../roi_experiments/evaluation.py) 處理。

## 1. 與既有兩種 Heatmap adapter 的差別

| 方法 | Heatmap 改變的位置 | 是否改變 BAF rollout 中的中間預測 | 可以直接增加位置／旋轉修正量嗎？ |
|---|---|---|---|
| `baf_roi_heatmap` 的 `residual` | 原 BAF 完成 future pose 後，增加有界位置／旋轉修正 | 否 | 可以 |
| `baf_roi_heatmap` 的 `expert_gate` | 混合 Focused-only 與 Explorer-only 各自完成的 future predictions | 不改原 BAF 的逐步融合 | 不可以，僅混合兩條已完成的預測 |
| **`baf_roi_heatmap_alpha`** | BAF autoregressive loop 裡每一步實際的 α | **會；修正後 pose 進入下一步 history** | 不可以，只有一個 α 修正輸出 |

因此，這個新方法不是把舊 `expert_gate` 換名字。它改變專家在同一個預測步驟的融合，後續專家也會看到由這次融合產生的不同 history。模型檔案與訓練程式獨立放在 `baf_roi_heatmap_alpha/`；資料契約及 evaluation 則與其他方法共用。

## 2. 輸入：沿用已生成、已對齊的 numeric Heatmap

此套件不重新做 MUSIC，也不讀 PNG。它重用 `baf_roi_heatmap` 已有的 numeric AoA–ToF maps、因果對齊與 normalization，並從同一個 `prepared` experiment 取得 frozen BAF、HMD history、split、origins、horizons 及 scene。

預設每個 origin 使用過去 0.4 秒的 8 個查詢點；每張圖轉成 3 個 channels，resize 成 24×24：

| Channel | 內容 |
|---|---|
| Level，記為 $Z_j$ | 以 training-only median／robust scale 正規化並 clipping 的 spectrum |
| Background residual，記為 $B_j$ | 當前 level 減去這個 window 內較早、不同有效 maps 的平均 |
| Temporal difference，記為 $D_j$ | 當前 level 與前一張不同、仍符合時間限制的有效 map 的差 |

一張圖必須在該查詢時間之前已完成量測並可交付，raw support 也必須落在允許的歷史範圍內。預設 max age 為 0.15 秒、clock uncertainty 上限為 0.01 秒；至少一半查詢點有效，且最後一點有效，才啟用 RF correction。完整條件見 [既有演算法文件](README_HEATMAP_ALGORITHM.md) 的因果對齊章節。

上述 normalization 只由 training records 擬合。Validation／test 不會重新估計參數，未來 Heatmap 也不會進入現在的 features。

## 3. 只取 8 個簡單統計量

令 $V$ 為有效查詢點集合，$\langle X\rangle$ 表示單張圖的空間平均；first、last 是當前 window 裡第一個與最後一個有效查詢點。`heatmap_statistics()` 計算：

| 順序 | Feature | 精確含義 | 提供的線索 |
|---|---|---|---|
| 1 | `level_mean` | 有效 $Z_j$ 空間平均值的平均 | 目前 spectrum 相對 training scale 的程度 |
| 2 | `level_spatial_std` | 有效 $Z_j$ 空間標準差的平均，採 population std | 圖中分布的起伏程度 |
| 3 | `background_abs_mean` | 有效 $\lvert B_j\rvert$ 空間平均值的平均 | 相對近期背景的變化量 |
| 4 | `temporal_abs_mean` | 有效 $\lvert D_j\rvert$ 空間平均值的平均 | 近期圖樣變動的強度 |
| 5 | `level_change` | $\langle Z_{last}\rangle-\langle Z_{first}\rangle$ | window 內整體 level 的上升或下降 |
| 6 | `valid_fraction` | 有效查詢點數／全部查詢點數 | RF window 的完整程度 |
| 7 | `latest_age` | 最後有效查詢點的 map age／`max_age_s` | 最新 RF 資訊的陳舊程度 |
| 8 | `level_peak_minus_mean` | 有效 $\max Z_j-\langle Z_j\rangle$ 的平均 | 峰值相對整張圖的突出程度 |

無效查詢點不參與圖像與 timing 統計。這些數值是 spectrum 的統計量，**不是校準過的身體速度、人體部位位置或動作類別**。此版本也刻意沒有保留每個 peak 的具體角度／delay 位置，因此能力比完整 CNN／GRU Heatmap encoder 更有限。

Feature vector $f_t\in\mathbb R^8$ 經過小型 MLP：

$$
u_t=W_2\tanh(W_1 f_t+b_1)+b_2,
\qquad \delta_t=d_{max}\tanh(u_t).
$$

預設 hidden width 為 16，總參數數量為：

$$
(8\times16+16)+(16\times1+1)=161.
$$

只有這 **161 個參數**會訓練。輸出層 $W_2,b_2$ 初始化為零，所以初始 $\delta_t=0$。預設 $d_{max}=1$，程式接受的範圍是 $0<d_{max}\leq10$，以避免 float32 odds multiplier overflow。

## 4. α 的方向與每一步到底如何改變

### 4.1 先區分原始 confidence 與平滑後權重

原 BAF Confidence estimator 輸出 Explorer 傾向 $c_t\in[0,1]$。原始 observed temporal EMA 為：

$$
a_t^{obs}=\mu c_t+(1-\mu)a_{t-1}^{obs},\qquad \mu=0.6.
$$

第一個有效 origin、或 observed decision gap 超過設定上限時，以當下 $c_t$ 初始化；不同 stream 分別保存 state。`prepared` 內的 `alpha` 就是按照共同 observed-origin 規則取得的原 BAF EMA。

融合權重方向始終是：

$$
\alpha=0\Rightarrow\text{Focused},\qquad
\alpha=1\Rightarrow\text{Explorer}.
$$

### 4.2 在原 BAF 的 forecast loop 中修正權重

以 $k=0$ 表示 origin $t$ 後的第一個 30 Hz 預測步驟。這一步的 prior 是原 observed EMA；後續步驟沿用 BAF 的平滑形式：

$$
a_k^{prior}=
\begin{cases}
a_t^{obs}, & k=0,\\
\mu c_k+(1-\mu)\widetilde a_{k-1}, & k>0.
\end{cases}
$$

這裡的 $c_k$ 是凍結 Confidence estimator 對「真實歷史加上先前預測 poses」的輸出。然後只調整 log-odds：

$$
\widetilde a_k=\sigma\!\left(\operatorname{logit}(a_k^{prior})+\delta_t\right).
$$

程式使用數值等價、可保留 0／1 端點的形式：

$$
\widetilde a_k=
\frac{a_k^{prior}e^{\delta_t}}
{1-a_k^{prior}+a_k^{prior}e^{\delta_t}}.
$$

$\delta_t>0$ 時增加 Explorer 比重；$\delta_t<0$ 時增加 Focused 比重。修正後仍在 $[0,1]$，沒有增加一個額外 pose prediction。當 prior 恰為 0 或 1 時，此 odds 修正保留端點。

位置與旋轉繼續按照原 BAF 融合：

$$
\widehat p_{k+1}=(1-\widetilde a_k)p_{F,k}+\widetilde a_k p_{E,k},
$$

$$
\widehat q_{k+1}=\operatorname{SLERP}(q_{F,k},q_{E,k},\widetilde a_k).
$$

接著移除 history 最舊的一個 pose、加入 $(\widehat p_{k+1},\widehat q_{k+1})$，再跑下一步。Position 單位、Euler／quaternion 座標慣例與原 BAF 相同；不同 horizon 的輸出也沿用 30 Hz rollout 的位置插值與 quaternion SLERP。

**同一 origin 的 $\delta_t$ 在整段 future rollout 中固定。** 預測 1000 ms 不代表讀取之後一秒的 CSI；預測過程只重用決策當時已知的 Heatmap summary。不同 origin 到來時，才用新的觀測重新計算 features 與 $\delta_t$。

## 5. Observed state 與 forecast state 分開

| 狀態 | 來源 | 是否保存到下一次真實決策？ |
|---|---|---|
| `alpha_t`／$a_t^{obs}$ | 真實 HMD history 的原 BAF confidence EMA | 是，依 stream 保存 |
| `rollout_alpha`／$\widetilde a_k$ | 當前預測內的 α correction | 否 |
| 預測 poses | 當前 autoregressive rollout | 否，只在這次 forecast 的 history 內使用 |
| $\delta_t$ | 當下 causal RF window | 否，下個 origin 重新計算 |

因此，先前使用過 Heatmap，不會讓下一次 RF 缺失時仍帶著 RF 修改過的 observed α。Online 呼叫必須依時間順序；換新 trajectory 可呼叫 `reset(stream_id)`。若某次呼叫失敗，runtime 會回復呼叫前的 observed state。

## 6. 訓練：凍結專家，但梯度能穿過預測過程

訓練使用 PyTorch 寫成可微分的同一個 BAF rollout。Confidence、Focused、Explorer 的參數皆設為 `requires_grad=False`，並持續使用 eval 模式，避免 dropout 或 BatchNorm 統計改變。Optimizer 只接收 alpha gate 的 parameters。

「凍結網路」不等於切斷它對輸入的梯度。修正 α 會改變預測 pose，預測 pose 又成為下一步的輸入，因此後面 horizon 的 loss 可以經過 frozen networks，回傳到較早的 α correction。沒有用 future ground-truth pose 替代中間的預測 history。

Loss 共用 [baf_roi_heatmap/losses.py](../baf_roi_heatmap/losses.py) 的 `task_loss`：

$$
\mathcal L=\mathcal L_{position}+\mathcal L_{rotation}
+0.25\mathcal L_{soft\ ROI}
+0.001\,\operatorname{mean}\left[(\delta_t/d_{max})^2\right].
$$

- Position：對以 0.1 m 縮放的位置做 Smooth L1。
- Rotation：對正規化 quaternions 使用 sign-invariant 的 $4(1-\langle\widehat q,q\rangle^2)/(10^\circ\text{ in radians})^2$ surrogate。
- Soft ROI：使用可微分 visibility 與 Tversky loss，FP／FN 係數分別為 0.3／0.7；空 ground-truth ROI 則懲罰平均預測 visibility。
- Regularization：限制不必要的 logit correction，不代表 α 的人工正確答案。

Training labels 只有 loss 會讀取，不會傳入 `model.forward()`。選 checkpoint 時採 validation original-recording macro IoU，並對已訓練 horizons 等權平均；**epoch 0 的原 BAF fallback 也可被選中**。如果訓練後沒有改善 validation score，就不強制採用非零 correction。

這個 checkpoint-selection score 與報告中的主要 aggregation 必須分清楚。正式比較仍使用共同 evaluator：同 origins、targets、split、scene、horizons 與 RF cohort，主表保留 per-origin mean IoU 與 pooled Micro F1，另列 recording／scenario macro 指標。

## 7. Fallback 與診斷欄位

RF 缺失、太舊、支援時間不合規或有效查詢點不足時，`rf_available=False`，gate 強制輸出零 correction。即使 RF 可用，若實際 $\delta_t=0$，也標示為沒有套用 correction。

Evaluation 對這些樣本直接回填 `prepared` 內原 BAF 的 float64 p／q；online runtime 則保留原 BAF 的原始輸出。這項處理確保零修正與缺測 fallback 的 ROI 幾何不會被 Torch／NumPy 的極小數值差異改變。

| 欄位 | 含義 |
|---|---|
| `rf_available` | 這次 RF window 是否滿足既有因果與品質門檻 |
| `adapted`／`applied` | 是否實際套用非零 correction |
| `alpha_t` | 原 observed BAF EMA，沒有被 RF 修改 |
| `logit_delta` | 這次 forecast 使用的 $\delta_t$，online 非零修正時輸出 |
| `rollout_alpha` | 每個 30 Hz forecast step 實際使用的融合權重 |
| Evaluation `correction_applied`／`gate_kind` | 是否實際套用修正，以及 gate 是 observed-origin α、rollout upper-step α 或其他方法的終點 mixture |
| Evaluation `gate` | 有修正時是該 horizon 的 upper rollout step α；fallback 時是原 observed `alpha_t`，與 BAF report 一致 |

Fallback `gate` 在單獨評估或與有效 RF 樣本混合成 batch 時，語意與數值都保持一致。

## 8. 執行方式

### 建議：使用共同 runner

從 repo 根目錄執行；`--prepared` 指向已完成配對資料準備的 experiment：

```bash
python scripts/run_experiments.py \
  --prepared runs/heatmap_prepared \
  --method baf_roi_heatmap_alpha \
  --output runs/alpha_experiment \
  --device auto
```

此命令沿用 `prepared` 裡的 frozen BAF、資料切分、horizons 與 scene，訓練 alpha method 並用共同 evaluator 評估。Checkpoint 位於 `runs/alpha_experiment/models/baf_roi_heatmap_alpha/alpha.pt`；逐 origin 結果位於 `evaluations/test/baf_roi_heatmap_alpha/`，根目錄會產生 `summary.csv` 與 experiment metadata。

若要調整此方法的訓練 epochs，可以加 `--alpha-epochs 20`；`--batch-size`、`--lr`、`--seed` 也可明示。已存在的 alpha checkpoint 可用 `--alpha-checkpoint` 重用，但必須屬於同一個 prepared experiment。

要比較全部四種方法，省略 `--method`：

```bash
python scripts/run_experiments.py \
  --prepared runs/heatmap_prepared \
  --output runs/all_methods \
  --device auto
```

輸出目錄必須是新的或空的。已有 `prepared` 就不能藉由改 horizon 或 scene 參數重解釋同一份 targets；需要先建立新的 experiment。

### 單獨使用 package 的 train／evaluate

```bash
python -m baf_roi_heatmap_alpha --threads 2 train \
  --prepared runs/heatmap_prepared \
  --output runs/alpha_standalone \
  --epochs 20 --batch-size 64 --seed 42 \
  --hidden 16 --max-logit-delta 1.0 --device auto

python -m baf_roi_heatmap_alpha --threads 2 evaluate \
  --prepared runs/heatmap_prepared \
  --checkpoint runs/alpha_standalone/alpha.pt \
  --output runs/alpha_standalone_test \
  --split test --margins 0 5 10 20 --device auto
```

Package 的 evaluate 仍呼叫相同共同 evaluator；沒有另一套 alpha 專用 IoU 或 F1 定義。

### Online API

真實 signature 為：

```python
HeatmapAlphaPredictor(baseline_checkpoint, alpha_checkpoint, device="cpu")

predict_observations(
    timestamps_s, positions, rotations_deg,
    rf_series=None, horizons_ms=None, stream_id=None, delivery_delay_s=0.0,
)

reset(stream_id=None)
```

使用例：

```python
from baf_roi_heatmap_alpha.runtime import HeatmapAlphaPredictor

predictor = HeatmapAlphaPredictor(
    baseline_checkpoint="runs/base/baf_roi.pt",
    alpha_checkpoint="runs/alpha_standalone/alpha.pt",
    device="cpu",
)

result = predictor.predict_observations(
    timestamps_s=timestamps_s,
    positions=positions,
    rotations_deg=rotations_yaw_pitch_roll_deg,
    rf_series=rf_series,
    horizons_ms=None,       # 使用 checkpoint 記錄的 horizons
    stream_id="viewer_01",
    delivery_delay_s=0.0,
)

predictor.reset("viewer_01")
```

這裡的 `timestamps_s`、`positions`、`rotations_yaw_pitch_roll_deg` 是目前已收到的 HMD observations；`positions` 使用 baseline checkpoint 記錄的原始單位（cm 或 m）。`rf_series` 是既有 `baf_roi_heatmap.data.HeatmapSeries`，需使用核對過的 clock offset／uncertainty，並映射到與 HMD 相同的時鐘；傳 `None` 就會走 BAF fallback。輸出位置是 m，quaternion 順序是 wxyz。

Online horizon 必須屬於 checkpoint 記錄的已訓練 horizons。不要把尚未到達的 HMD observations 放入這次呼叫。

## 9. 能證明什麼，以及目前的限制

此方法適合檢驗「CSI 是否幫助選擇既有專家」。單一 step 的位置仍在兩個專家輸出之間，旋轉仍是兩者的 SLERP；如果兩個專家都缺少必要的方向線索，調整 α 不一定足夠。由於 recursive history 會改變，整段預測並不受限於兩條獨立完成軌跡之間的一條固定線段。

161 個 gate 參數不代表訓練成本只有小型 MLP。訓練仍需執行 frozen BAF 的可微 rollout，最長 1000 ms 對應 30 個 steps，並保存 backward 所需狀態；記憶體需求與 batch size、最大 horizon 都有關。八個統計特徵也可能丟失原圖中具有預測價值的空間資訊。

CPU 已完成 17 項 targeted tests，以及四方法共同 synthetic 訓練／評估整合測試。測試包括真正原始 BAF networks 的數值對照與 finite gradients、零修正及 RF outage 的 exact fallback、future-RF invariance、stream state、batch-independent fallback diagnostics，以及 quaternion／gimbal-lock 邊界。

CUDA 程式路徑尚未在本次環境實機驗證。訓練時在 frozen eval-mode LSTM forward 停用 cuDNN RNN 路徑，改用支援 input backward 的 native PyTorch 路徑；這可能影響 GPU 效能，不能用 CPU smoke test 宣稱 GPU 吞吐量或整個系統已達 30 Hz。

上述測試只證明程式與資料流程可執行，**不是實測 CSI 帶來 accuracy gain 的證據**。仍須在真實 held-out paired recordings 上進行共同 evaluation。Coverage 是幾何 proxy，selected partition count 不是網路 bytes；目前沒有 matched rendered frames，因此沒有 SSIM／視覺品質保證。詳見 [共同評估文件](README_HEATMAP_EVALUATION.md)。
