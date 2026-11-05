# Heatmap 演算法、研究理由與 Pseudocode

## 1. 問題定義：要改善的是未來 ROI，不是 α 分類率

在真實決策時間 t，已知 HMD 的 30 個 poses，取樣率 30 Hz：

$$
X_t=[x_{t-29/30},\ldots,x_t],\qquad x=(p_x,p_y,p_z,yaw,pitch,roll).
$$

30 個 samples 的首尾跨度是 29/30 秒，不是完整 1 秒。保留原本 HMD 觀測 EMA 的狀態；因此此數值限制的是顯式 pose/RF window，不能宣稱 predictor 的全部記憶只有一秒。

對每個預測 horizon h，輸出 future pose，再由固定場景 S 的幾何函數 V 轉成集合：

$$
\hat G_{t,h}=V(\hat p_{t,h},\hat q_{t,h};S),\qquad
G_{t,h}=V(p_{t+h},q_{t+h};S).
$$

研究比較的是同一 predictor family 的 `HMD-only` 與 `HMD+RF`。未來 pose / ROI 只能當 training target 或 evaluation reference，不能出現在模型 inputs。訓練 loss 的 scene 是已知靜態場景，不是 future observation，但會讓 objective 具有場景依賴性。

### 既有資料如何支持這個問題

20261016 投影片 slides 2–3、13、19–20 與 10/09 前研究先檢查 non-head body history 的增量資訊，再提出 CSI 實驗。其 motion-capture 結果不是 CSI sensing 成功的證據，也沒有證明固定的腳→骨盆→頭順序。因此本版不強制 skeleton reconstruction、不替 heatmap peaks 命名 body parts，而讓時序模型學習可預測的變化。來源細節見 [HEATMAP_SOURCES.md](HEATMAP_SOURCES.md)。

## 2. 為什麼不是只把 Heatmap 丟進 α？

原 BAF 用 causal HMD history 估計 Explorer 傾向，α 越大越靠近 Explorer。兩個分支是 Focused LSTM 和 Explorer TCN，最後以位置 LERP / quaternion SLERP 融合。原論文式 (5.7) 使用 μ=0.6 的 temporal smoothing。

單步融合的位置滿足：

$$
p=(1-\alpha)p_F+\alpha p_E,\quad 0\leq\alpha\leq1.
$$

所以它受限於兩個分支預測位置之間的線段；旋轉受限於相應 shortest-arc interpolation。這是融合形式的幾何限制，不表示所有 recursive α 序列都只在同一條固定線段上。

當兩個專家都沒預測到一個轉向，僅調比例不一定能利用 CSI 的新方向線索。相反地，當問題只是「該信任哪個專家」，gate 可能已經足夠。因此同時實作兩種方法，讓實驗回答，沒有預設 residual 一定較好。

| 選擇 | 優點 | 限制／本版處理 |
|---|---|---|
| 把 RF 特徵直接接進兩個專家並一起重訓 | 最大模型自由度 | 同時改 experts 和融合，很難區分增益來源；不作第一階段主方案 |
| RF-assisted expert gate | 機制簡單，可檢驗專家選擇假設 | 受候選 predictions 限制；保留為正式比較組 |
| RF-assisted bounded pose residual | 可修正共同偏誤，零初始化回到原 BAF | 容量增加、可能過擬合；用 HMD-only residual 對照、限幅及 validation 選擇 |

**原論文 Table 11 的 Explorer-only 平均 pose error 已優於完整 BAF。** 因此「比 BAF 好」不自動代表新增感測有用，必須同時比較 Explorer-only、固定融合與同架構 HMD control。這裡引用的是原論文結果，不是本分支的真實實驗結果。

## 3. 沿用原 Heatmap 程式的哪些部分？修正哪些部分？

參考使用者 2026-10-27 提供的 `MUSIC.py`、`pre_processing.py`、`signal_processing.py`、`main(20261027-120045).py`。本版不是把其 PNG 當 feature，也不是宣稱完全重現原前處理。

### 3.1 保留的核心

`music.py` 保留 AoA–ToF 的 spatial/frequency smoothing、Kronecker steering 排列與 noise-subspace spectrum：

$$
a_{r,k}(\theta,\tau)=
\exp[-j2\pi f_c d\,r\,g(\theta)/c]\,
\exp[-j2\pi f_k\tau].
$$

Rx-major、frequency-minor 的順序與原程式一致。`sin` 定義 g(θ)=sin θ；`legacy_cos` 明確對應原程式的 **g(θ)=1−cos θ**，不是一般的 cos θ。這是座標 convention，不可在沒有 array calibration 的情況下換名稱就當成物理角度。

對每個 trailing window 中的 packet、Tx、Rx/subcarrier sliding aperture 產生 snapshots x，累積：

$$
R=\frac1N\sum_n x_nx_n^H,\qquad
P(\theta,\tau)=\frac1{\|a(\theta,\tau)^H E_n\|_2^2+\epsilon}.
$$

`signal_dim` 決定 signal/noise 子空間分界，是需 validation 的 model-order 假設，不是已辨識的人數或身體部位數。零動態 covariance 會標記 invalid，不產生任意亮峰。

### 3.2 必須修正的部分及理由

原 `self_sanitize(x)=x*conj(x)/abs(x)` 對非零 x 等於 abs(x)，會消掉 complex phase。原幅度 DWT 也不是保留 coherent phase 的校正。本版預設保留 complex CSI，對每個已結束的 trailing raw window 減去其 complex mean；這抑制局部靜態成分，但可能削弱很慢的動作，可用 `preprocess=none` 作消融。

這個改動**不是** hardware phase calibration。Phase offset、packet timing / STO、array geometry、carrier 與 subcarrier mapping 仍須由量測系統確認。`legacy_amplitude` 只供探索對照，輸出永遠不標為已驗證物理 AoA–ToF。

原 `np.convolve(..., mode='same')` 及部分 MUSIC window 是置中處理，whole-record DWT 也可能使用後續 samples。本版只使用 `[end-window_packets+1, end]`，不讓 future samples 進入 map；packet gap 過大則 invalid。

原設定 nominal fs / BW 不足以推導真實子載波頻率。新輸入必須提供 `frequencies_hz[K]` 與實際 `timestamps_s[N]`。頻率 smoothing 需要均勻、已排序的 selected grid；不符合就拒絕，不能默默套 BW/K。

### 3.3 支援範圍

本版實作 **AoA–ToF sequence**，不是已完成 AoA–ToF–Doppler 3D 模型。原來源的其他 MUSIC views 是後續候選。Temporal difference 的正負只表示某 bin 的 spectrum 增減；**不等於 signed human velocity、Doppler 或某個 body part 的方向**。前研究的 signed XYZ velocity 僅支持保留方向資訊的動機，不能拿來替 RF representation 背書。

## 4. 因果對齊：先確認「那時候真的拿得到」

每張 map k 儲存三個時間：raw support start s_k、end e_k、計算完成可交付時間 a_k。它們經明示 clock offset 映射到 HMD clock。令 clock uncertainty 為 u，另加測試 delivery delay d；在 origin t 的查詢點 q_j，map 可用的條件為：

$$
s_k-u\geq t-W_{raw},\quad
 e_k+u\leq q_j,\quad
 a_k+d+u\leq q_j,\quad
 q_j-(e_k-u)\leq A_{max}.
$$

還必須 valid=true。從符合條件者選 measurement end 最新的一張，而不是「時間最近」的一張，亦不跨 future map 做線性插值。

預設 q_j 為 `[t−0.4,t]` 的 8 個查詢點；W_raw=29/30 秒，A_max=0.15 秒，允許 clock uncertainty≤0.01 秒。有效點比例至少 50%，且最後一點有效，才啟用 RF correction。這些門檻是可調設計，不是論文證實的最佳值。

例如 50 個 packets@實際 100 Hz 的 raw span 約 0.49 秒，最後 0.4 秒的 map history 加上 delivery/staleness，仍可能放進 29/30 秒預算。若延遲太大或 sample rate 太低，程式會減少有效點或退回 baseline，不能把長 support 偷藏在「一張圖」之內。

`available_at_s` 預設採目前 CPU 的 serial replay 計算時間並計入 backlog；也可輸入明示 delay model。前者不是量測時的歷史真實可用時間，後者更不是零延遲保證。正式研究應盡量由實際 pipeline 記錄 delivery timestamps。

## 5. Heatmap representation 的精確定義

輸入非負 linear pseudo-spectrum 時用 10log10(max(P,1e−12))；已是 dB 時不重複取 log，不把負 dB 截成零。

只從 training records 估計 median m 與 robust scale s：

$$
s=\max(1.4826\,\operatorname{median}|L-m|,10^{-3}),\quad
 Z=\operatorname{clip}((L-m)/s,-6,6).
$$

每個 training record 最多均勻抽取 4096 個有效 spectrum values 擬合；validation/test 永不 refit。保留相對尺度但沒有宣稱 MUSIC spectrum 是已校準接收功率。

每個查詢時間的三個 channels：

1. **Level** Z_j：現在的角度–delay 分布。
2. **Within-window background residual** Z_j−mean(previous distinct valid levels)：相對最近背景的變化。
3. **Signed temporal difference** Z_j−Z_prev：下一個不同 raw map 的增減。

第一個有效點沒有 past reference，後兩者為零。連續命中同一 map 不產生假的差分；missing query 會重置差分/background reference；超過 max-age 的兩張圖不做連續差分。本版 background reference 只在當前 window 內建立，不是跨整個 recording 的持續 EMA，因此 raw history budget 可檢查。

原物理 grid 必須一致且等間距，再 bilinear resize 成 24×24。這是計算量折衷；可能損失窄峰，需要比較更高解析度。不同 angle/delay 範圍不得只靠 resize 混為同一 grid。

## 6. Encoder 與輸出

HMD 使用原 `model_inputs`：training mean/std 正規化位置，旋轉轉成 wxyz quaternion，因此每個 timestep 是 7 維。HMD encoder 為 GRU，hidden=32。

RF 每張圖使用 Conv2d(3→8,k3,stride2)、GELU、Conv2d(8→16,k3,stride2)、GELU、adaptive pooling 2×2，得到 64 維。再加 map age 與相對查詢時間兩維，送入 masked GRUCell，hidden=32。無效 query 不更新 state，不讓 zero-filled 圖被當成一張真實低功率圖。

Concat HMD/RF representations 後，把 horizon seconds 作為額外輸入。MLP 為 `(2×hidden+1)→64→output`；horizon conditioning 用來分開近未來與較長未來需要的修正。沒有使用未來 CSI，所有 horizons 共用 t 時刻的 observed RF window。

### 6.1 主方法：bounded residual

令 v_p、v_r 是 MLP 的兩個三維輸出，定義

$$
B_b(v)=b\tanh(\|v\|_2)\frac{v}{\max(\|v\|_2,\epsilon)}.
$$

$$
\Delta p=B_{0.2m}(v_p),\qquad
\Delta r=B_{20\pi/180}(v_r),
$$

$$
\hat p=\hat p_{BAF}+\Delta p,\quad
\hat q=\operatorname{normalize}(Q(\Delta r)\otimes\hat q_{BAF}).
$$

Q 是 rotation-vector exponential quaternion，採左乘，所以修正是 world-frame rotation。不是直接相加 Euler angles，也不是加四維 quaternion。兩個 bound 限制的是向量 L2 norm，不是每軸各 0.2m / 20°。

輸出層權重與 bias 零初始化，初始 dp/dr=0。BAF models 凍結；修正不寫回 BAF rollout，也不寫回 observed alpha state。下一個 real origin 仍以真實 HMD observations 更新 baseline。

### 6.2 比較方法：horizon-conditioned expert gate

先獨立 recursive rollout Focused-only 與 Explorer-only，得到各 h 的 endpoints。再用原觀測 origin 的 α 作 prior：

$$
w_{t,h}=\sigma(\operatorname{logit}(\operatorname{clip}(\alpha_t,10^{-5},1-10^{-5}))+2\tanh(z_{t,h})).
$$

位置/旋轉分別為兩個 endpoints 的 LERP/SLERP。

**此方法不同於原 BAF 每個 recursive step 都重新算 α 的路徑。** Head=0 時是以 α_t 混合兩條純專家 rollout endpoints，不保證等於原 BAF；它應被稱為 `expert_gate` adaptation，不能稱為忠實的原 α replacement。若日後研究要求逐步調 α，需另立方法和 train/inference contract，不能把本比較組結果混稱。

HMD-only gate 與 RF gate 使用相同 task loss、相同 frozen experts；另外有原 BAF、每步固定0.5融合與純專家，避免把不同 rollout 方式帶來的改變誤認成 CSI gain。

### 6.3 缺測與數值一致性

RF 缺測／過期／有效點不足時，兩種 RF 方法都直接回到原 BAF。Residual=0 也回到原 BAF。`numpy_prediction` 在這些情況還原原始 float64 pose，而不是把 float32 adapter tensor 轉回 float64；否則在 FoV 邊界可能產生不是方法造成的 partition flip。

原 BAF 的 observed EMA 沒有受到 RF 修改，所以不需要用上一個 RF-fused EMA 假裝 fallback。Online failure 時 rollback observed state，避免例外造成下一次 state 不一致。

## 7. 訓練目標：直接對應 task，而非只模仿行為標籤

原 BAF 仍保留原四階段訓練與 behavioral proxy。新增 adapters **不再只用 Explorer/Focused label**，而使用真實 future pose 與 ROI supervision：

$$
\mathcal L=\mathcal L_p+\mathcal L_q+0.25\mathcal L_{ROI}+0.001\mathcal L_{reg}.
$$

- L_p：對 p/0.1m 與 target/0.1m 使用 SmoothL1，beta=1。
- L_q：`4(1−<q,q*>²)/(10° in radians)²` 的平均。它對 q 與 −q 等價，是 small-angle surrogate，不是最終報告的 degree error。
- L_ROI：由同一 scene 的 soft visibility 產生 probability，非空 GT 使用 `1−(TP+eps)/(TP+0.3FP+0.7FN+eps)`；空 GT 用 mean predicted probability。
- L_reg：normalized correction norm²；gate 為 bounded logit delta²。

漏掉必要 partition 的懲罰在 surrogate 中較重，目的在保留 coverage；0.3/0.7 不是對實際頻寬/畫質的校準係數。訓練可用 `--roi-weight 0` 比較 pose-only loss，檢驗 ROI supervision 是否必要。

Cone 的 soft visibility 為 sigmoid((cosine−cos(FoV/2))/0.02)；frustum 對各邊界用 0.03m sigmoid temperature。這是可微近似；最終所有數據都用原 `Scene.mask` 硬判定，不能拿 soft training score 代替 test IoU。

AdamW，lr=0.001、weight_decay=0.0001、gradient clip=5、預設20epochs/batch64。Checkpoint 由 validation **recording-macro hard IoU**、各 trained horizon 等權平均選擇；epoch0 也可被選中。沒有提高就應允許維持 baseline，不把訓練完成當成有效增益。

## 8. Pseudocode

### A. Causal map generation

```text
validate complex CSI, packet timestamps, availability, selected frequencies
validate explicit array/angle convention and signal subspace dimension
for each packet endpoint e at configured stride:
    raw_window = CSI[e-W+1 : e+1]
    if a raw timestamp gap exceeds max_raw_gap:
        mark map invalid
    else:
        optionally subtract complex mean of THIS completed window
        accumulate Rx/frequency-smoothed covariance
        eigendecompose; select declared noise subspace
        evaluate AoA-ToF pseudo-spectrum on declared axes
    save spectrum + raw start/end + delivery time + validity
```

### B. Prepare frozen task targets

```text
verify BAF checkpoint, pose cache, grouping, clocks and scene
fit RF normalization on TRAIN records only
for each record in chronological order:
    reset original observed-alpha state
    for each observed origin t:
        X = causal 30-pose history
        update observed BAF alpha from X; never commit forecast states
        if every requested future horizon has a GT pose:
            cache BAF recursive predictions and pure F/E/fixed rollouts
            cache GT future poses and hard ROI masks separately
save experiment identity, source hashes and exact common origin IDs
```

### C. Train adapter

```text
freeze original BAF parameters
initialize adapter output head to zero
for epoch in 0..E:
    if epoch > 0:
        for TRAIN origin batch:
            RF = past-only maps satisfying raw-support and delivery constraints
            encode HMD and masked RF sequence
            apply residual or terminal expert gate at each trained horizon
            compute pose + soft ROI + regularization loss from separate targets
            update adapter only
    evaluate hard ROI IoU on VALID records
    save checkpoint only if recording-macro validation score improves
never inspect TEST to choose architecture, bounds, epoch or margins
```

### D. Inference at a real origin

```text
receive only HMD observations available through t
run unchanged BAF to requested trained horizons
build RF window from maps available at their causal query times
if RF mode and window invalid:
    return original float64 BAF prediction
else:
    run adapter once, conditioning output on each horizon
    apply bounded residual OR terminal expert gate
convert corrected future pose to hard ROI with fixed scene
commit only original observed BAF alpha state
```

## 9. Hyperparameters、複雜度與可反駁性

所有預設值是第一版可重現方案，不是已找到最優值。優先在 validation 比較 map window、query count、map size、raw span、clock tolerance、correction bounds、ROI weight；不要在 test 上大規模搜尋最佳曲線。

預設 hidden32 的 adapter parameters：residual **19,542**、expert_gate **19,217**。HMD-only control 保留相同 module 結構，但 RF 專用參數不活躍，所以不是「活躍容量完全相等」的證明；需要 `metadata_only`、representation ablations 與多 seeds 補強。

令 map dimension A×D、smoothing aperture n=Rx_window×frequency_points、snapshots N、搜尋點 G。MUSIC dense covariance 約 O(Nn²)、eigendecomposition O(n³)、spectrum 約 O(Gn(n−Sdim))；記憶體含 covariance O(n²) 與 steering O(Gn)。生成器 snapshot 分塊1024，不代表完全 constant memory。CNN/GRU 增量固定且小；主要 BAF rollout 仍隨 ceil(30h_max) 增長，gate 還有純 F/E rollouts。Batch throughput 不能當 single-request latency。

可反駁結論：RF residual 若不優於 HMD residual，不能宣稱額外 sensing 有用；若只優於 BAF 而不優於 Explorer-only，不能忽略這個反例；若 effect 只在未校正 axes 或不合理 timing 下出現，應先視為資料問題。真正 novel 的研究結果需要真實同步資料與這些控制實驗支持，而不是網路多接一個 CNN 就自動成立。
