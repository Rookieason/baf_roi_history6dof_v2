# History 6DoF → 預測輸出的完整推論複雜度分析

工具針對 **BAF-ROI v2**，沿用 `Predictor.load`、`forecast_batch`、`predict_observations` 與 `Scene.mask`。
不修改權重、訓練或原作者模型。Motion Transformer 是另一獨立方法，不納入本工具的 BAF 總量。

## 在已訓練模型所在主機執行

同步本次程式更新到模型所在主機，在專案根目錄使用原本已裝好 PyTorch 的環境執行。
只需要可信任的 v2 BAF checkpoint，不需要訓練資料、cache 或重新訓練。

```bash
python -m baf_roi.complexity --checkpoint runs/experiment01/baf_roi.pt --output runs/complexity_baf --device auto --threads 4 --warmup 30 --iterations 300 --horizons-ms 50 100 150 200 250 --horizons-ms 200 400 600 800 1000
```

每個 `--horizons-ms` 是一次同時要求多個 horizon 的完整預測，這些 horizon **共用 rollout**。
`--device` 支援 `auto`、`cpu`、`cuda:0` 等。單 horizon 與 batch sweep 範例：

```bash
python -m baf_roi.complexity --checkpoint runs/experiment01/baf_roi.pt --output runs/complexity_sweep --device cuda:0 --batch-sizes 1 8 32 --horizons-ms 50 --horizons-ms 250 --horizons-ms 500 --horizons-ms 1000 --warmup 30 --iterations 300
```

如要涵蓋實際 observations 的重採樣、單位轉換及 ROI，提供既有 predict 使用的 JSON history 和 scene：

```bash
python -m baf_roi.complexity --checkpoint runs/experiment01/baf_roi.pt --history history.json --scene-config configs/my_confirmed_scene.json --output runs/complexity_full --device auto --profile --warmup 30 --iterations 300
```

history 欄位為 `timestamps_s[N]`、`positions[N,3]`、`rotations_deg[N,3]`。
位置單位依 checkpoint 的 `data.position_unit`；旋轉為 yaw/pitch/roll degrees。
觀測至少覆蓋 29/30 秒且符合 checkpoint 的 max-gap 限制。scene 沿用原本 geometry confirmation 與相對路徑解析。
沒有 scene 時終點為 pose，不應稱為完整 ROI pipeline。
沒有 history 時採用合成平滑輸入；提供 history 時每個 batch entry 都複製同一段，來源與 hash 寫入報告。

尚未取得 checkpoint 時可先驗證架構與工具：

```bash
python -m baf_roi.complexity --architecture-only --output smoke_output/complexity_check --device cpu --threads 2 --warmup 2 --iterations 5
```

此模式明確標記 `UNTRAINED_ARCHITECTURE_ONLY`，使用單位 normalization；其延遲不能代表另一台主機的已訓練模型。
在目前固定架構下，結構檢查結果為 **611,599 parameters**、**2,446,396 parameter bytes**，另有 1,816 buffer bytes。
250 ms 的 dense MACs 為 139,240,960／history；1000 ms 為 522,153,600／history。
這些是固定架構的邏輯計數，無法據此推算目標 GPU 的實際延遲。

## 分析範圍

```text
observations → 時間格點重採樣／單位轉換 → [B,30,6] history
  → 每個 rollout step：
      座標／normalization／quaternion → Confidence CNN + Focused LSTM + Explorer TCN
      → CPU 輸出／denormalization → confidence EMA → translation LERP + quaternion SLERP
      → 預測 pose 接回 history
  → exact-horizon 插值／輸出旋轉格式轉換 → poses
  → 可選：partition visibility → ROI masks
```

三分支每一步都執行，confidence 不會跳過分支。每步重新計算完整 30-frame history，
不跨步保留 LSTM hidden state 或 TCN feature cache。
K=ceil(30×max(horizons_ms)/1000)，所以 250 ms=8 steps、1000 ms=30 steps。
一次多 horizon 的成本不能以每個 horizon 獨立預測的成本相加。

| 面向 | 報告內容 |
|---|---|
| 模型大小 | 分支／整體 parameters、trainable parameters、parameter bytes、buffer bytes、dtype、config、架構 |
| 部署檔大小 | checkpoint file bytes，包含 metadata，不等於純權重大小 |
| 運算量 | 實際 forward hooks 計算每層／分支／整次 rollout dense MACs 與 FLOPs=2×MACs |
| 完整延遲 | forecast、observations API、可選 ROI；mean/std/min/p50/p90/p95/p99/max 與全部 raw samples |
| 吞吐與即時性 | histories/s、amortized ms/history、deadline miss fraction；deadline 預設 1000/30 ms，可用 `--deadline-ms` 更改 |
| 階段診斷 | 前處理、分支 forward、融合、插值、旋轉轉換及 remainder；額外插樁 pass |
| 記憶體 | 模型 tensor bytes、回傳 NumPy bytes、Python traced peak、程序 RSS 採樣、CUDA allocator baseline/peak/increment |
| 可重現性 | checkpoint／history／程式／模型 SHA256、git revision/dirty、seed、硬體、版本、threads、CUDA/cuDNN/TF32 設定 |
| Operator 診斷 | 可選 Chrome trace、CPU／CUDA operator 時間表 |

### MACs 與漸近複雜度

本工具計算指定算子的邏輯 dense MACs，**不是整個 pipeline 的完整 FLOP 指令數**：

- Conv1d：output.numel × (in_channels/groups) × kernel_size。使用 chomp 前的實際 output 長度，包含 TCN 計算後丟棄的 padded 尾端。
- Linear：output.numel × in_features。
- LSTM：逐層計入 B×T×directions×4h×(input_width+recurrent_width)；所有 gate 的 input/recurrent matrix products 都計入，helper 另支援 projection matrix products。
- 一 MAC 計為乘法與加法，dense FLOPs=2×dense MACs。

不計 bias、activation、normalization、pooling、LSTM gate elementwise、quaternion、NumPy/SciPy、EMA、fusion、插值、ROI、資料搬移的 FLOPs。
範圍外的 parameterized modules（例如 BatchNorm／LayerNorm）會列出，其參數仍納入 parameters。
發表時應寫「dense matrix/convolution FLOPs」並說明範圍，不能宣稱是「完整 pipeline FLOPs」。
backend 可能採不同算法，邏輯 MACs 不是硬體執行指令數；上述其他成本均納入端到端 wall time。

令 B 為 batch、T=30、H 為輸出 horizon 數、P 為 partition 數、W 為權重數，
N 為單個 history 經三分支的 dense 計算量（不含 B）：

```text
history → pose 時間：O(B K (N + T) + B H)
pose → ROI 時間：O(B H P)
observation adapter：另含 O(M) 的觀測驗證／unwrap，與 T 個格點查詢
空間：O(W) 常駐模型 + 單步 activation/workspace + O(BK + BH) rollout/輸出
ROI 中間陣列與 masks：O(BHP)
```

Conv 的主項為 T_out×C_out×C_in×kernel/groups，LSTM 為逐層 T×4h×(d+h)。
目前 runtime 還回傳 YPR、rollout 等多份陣列，實際 bytes 以報告為準。

### 量測口徑與限制

正式計時使用 eval + inference_mode，不啟用 autocast 或 compile；使用現有模型 FP32 架構。
以 perf_counter_ns 量測 host wall time，CUDA 在每次呼叫邊界 synchronize，納入 CPU/GPU 往返與 CPU SciPy 運算。
checkpoint 載入、scene 載入、磁碟／網路／JSON 序列化不在穩態 latency 內。
`model_setup_ms` 另記錄模型建置／載入／裝置搬移與 checkpoint 雜湊，並非 cold-start latency。

isolated device-resident 分支使用預先搬到裝置的 tensors；不能直接相加其 median 當作完整延遲。
batch amortized time 是吞吐量指標，不等於單一 request response latency。
batch=1 才量測現有 observations API，使用 stream_id=None；每次是獨立決策，不累積已觀測 EMA state，rollout 內仍正常 EMA。
本分析不主張量測完整串流 state 管理負載。

`stage_diagnostic` 額外執行一次同步插樁呼叫，具名區段不重疊。
remainder 含 tensor 建立／搬移、denormalization、驗證、EMA、history 更新、插樁開銷；不可全歸因為 transfer。
插樁與 trace 會增加成本，其總時間不能取代未插樁 benchmark。

CPU RSS 使用 Windows／Linux 程序約 1 ms 採樣，包含 interpreter、native allocator、模型與前面 case 的 cache，可能漏掉短暫尖峰。
`cpu_native_peak_bytes=null` 表示未量測精確 native peak，不代表零。Python traced peak 也不等於全部 native memory。
CUDA peak allocated/reserved 是 PyTorch allocator 峰值，不是整張 GPU 使用量；baseline 含持久模型 tensor，
increment 是當次相對 baseline 的 allocator 增量，不能當作完整 activation footprint。
memory 與 profiler 都在正式 latency 之外的 pass 執行。

比較不同方法時固定硬體、軟體、dtype、threads、batch、horizons、history 長度、scene 及背景負載，
使用獨立程序及新 output 目錄。建議至少 300 次，關注 p99／deadline miss 時增加次數並獨立重複實驗。
少量樣本的尾端百分位只有描述意義，沒有信賴區間保證。GPU 路徑須在目標 GPU 實測。
本工具不計算準確率，應搭配既有 held-out evaluation。

## 輸出

- `report.json`：所有 case、raw timings、逐層 counts、記憶體範圍、環境與來源資訊。
- `summary.md`：模型總數及各 case 的 dense GMAC、forecast p50/p95 與吞吐量。
- `measurements.csv`：每個 case／endpoint 的統計值，可直接匯入表格或繪圖；dense MACs 欄位固定代表完整 forecast，不能當作每個 endpoint 的運算量。
- `profile_NNN/trace.json`：可用 Chrome tracing／Perfetto 開啟，含 observation 重採樣、三分支與 ROI ranges。
- `profile_NNN/operators.txt`：CPU 摘要；CUDA 另有 `operators_cuda.txt`。

拒絕非空 output 目錄以保留實驗；每完成一個 case 儲存一次 JSON。
執行 `python -m pytest tests/test_complexity.py -q` 可檢查計數公式、shared rollout、batch scaling、
原預測不變／hook 清理，以及 checkpoint + observations + scene 報告流程。
