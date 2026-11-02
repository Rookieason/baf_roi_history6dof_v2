# Evaluation：每個指標與對照到底在回答什麼？

## 1. 研究主張與證據分層

最終 task 是 future ROI selection。前研究的 motion-capture 結果支持探索非頭部動態，但不是 CSI 已改善預測的證據；目前 smoke test 只證明軟體流程可執行。

本版檢驗：

**H1：** 在同一 adapter、相同 HMD history / frozen experts / targets / 訓練規則下，加入可因果取得的 Heatmap 是否改善 held-out ROI？

**H2：** 若有增益，是 RF 的空間／時間變化資訊，還是額外容量、輸入可用性 metadata、較長 history、時鐘錯誤等因素？

**H3：** 改善是否能轉成更好的 coverage–selected-partition trade-off，而非透過漏掉更多必要 partitions 讓 selected count 看起來更少？

**H4：** 在可測得的 delivery latency、missing RF 與 deployment 計算成本下，方法是否仍可使用？

IoU/F1、pose error、latency 分別是不同層的證據。沒有 rendered images 就不報 SSIM；沒有實際 partition bytes 就不把 selected count 叫 bitrate。較高 coverage 也不自動保證主觀畫質。

## 2. 必要比較組與各自目的

| 方法 | 實作 | 為什麼需要 |
|---|---|---|
| BAF-ROI | `evaluate --baseline baf` | 相對目前 HMD-only baseline 的整體變化 |
| Focused-only | `--baseline focused` | 原 LSTM 專家本身能做到什麼 |
| Explorer-only | `--baseline explorer` | 原論文點對點 pose error 已可能優於 BAF，不能省略 |
| Fixed α=0.5 | `--baseline fixed` | 每個 recursive step 固定融合，區分有無動態 gate |
| Motion Transformer | `--motion-transformer CHECKPOINT` | 延續既有 PPT 的另一種 HMD-only predictor family |
| HMD residual | `train --method residual --modality hmd` | 控制額外 correction head、task loss 與重新訓練的價值 |
| RF residual | `train --method residual --modality rf` | 主要研究候選，和 HMD residual 比才是增量 sensing 檢驗 |
| HMD expert gate | `train --method expert_gate --modality hmd` | 控制 terminal-expert-fusion adaptation 本身造成的改變 |
| RF expert gate | `train --method expert_gate --modality rf` | 檢驗教授所舉的專家權重輔助方向是否足夠 |

原 BAF 的 behavioral-confidence supervision 和新 adapters 的 pose/ROI task loss 不相同。因此 RF residual 勝過 BAF 時，不能直接把全部差距歸因於 RF。

`hmd` control 與 `rf` 使用相同 config、head、loss、seed、epochs、validation 規則，但 HMD control 的 RF modules 不活躍；它不是活躍參數容量完美相等的證明。需加 metadata-only control、多 seeds、representation ablations；更完整的 capacity-matched HMD encoder 可以作後續研究，但本版不聲稱已做。

### Representation ablations（須重新訓練）

| `--ablation` | 實際操作 | 要回答的問題 |
|---|---|---|
| `metadata_only` | maps 全零，但保留真實 mask/age/time | 增益是否來自 radio availability 或時間 metadata，而不是 spectrum |
| `spatial_mean` | 每 channel 對兩軸平均，再 broadcast | 完整空間布局是否比單一平均特徵有用；這不是校準功率 |
| `last_frame` | 只留最後 level，清除 earlier masks 及兩個 temporal channels | 是否真正需要動態序列，不讓 difference channels 偷帶歷史 |
| `angle_only` | 對 delay 軸平均 | delay 維度是否有額外價值 |
| `delay_only` | 對 angle 軸平均 | angle 維度是否有額外價值 |
| `abs_change` | background residual / temporal difference 取絕對值 | spectrum 增減符號是否有用；不是 Doppler 方向驗證 |

使用同一 checkpoint 在 test 直接抹掉資料是 robustness stress，不是重新訓練消融的替代品。兩種結果必須分開命名。

## 3. 公平資料與時間條件

所有方法必須共用 prepare 後的 `experiment_id`。它鎖定 HMD cache、split、scene、partition ordering、horizons、normalization 與 RF manifest。評估器使用同一份所有 horizons 皆有 GT 的 origins。

新增長 horizon 可能使 recording 尾段被排除；因此只做 50–250ms 的舊結果與做 50–1000ms 的新結果，不可直接當相同樣本比較。需要在同一 prepare 上重跑兩邊。程式不會默默只取兩份 CSV 的 intersection。

Train/valid/test 以完整 recording / subject group 分開。重疊 windows 不可隨機打散跨 splits。同 subject 有多個 sessions 時應用同 group_id；只以 session 切分，不能宣稱 unseen-subject 泛化。外部切出的片段若改寫 bytes 會繞過相同 SHA 檢查，因此真實 group/recording metadata 仍是必要的資料治理。

HMD 與 CSI 必須是同一次真實同步量測。不能因為 Nymeria 有 mocap、另一個資料夾有 RF，就把它們按 index 配對。前研究的 224 recordings 不代表自動具有 paired CSI。

RF raw-support 最早時間、latest measurement、delivery time、clock uncertainty 皆依 ALGORITHM 的條件判定。已在 t 後才算好的 map，不得用它的中心 timestamp 假裝 t 時可用。全錄影 normalized/filter/PCA 也要另行審核，不能只檢查神經網路的輸入 tensor。

## 4. Hard ROI metric 定義與用途

對 origin t、horizon h：P 是預測選取集合，G 是真實 future pose 對應集合。固定同一場景、同一 partition universe：

\[
TP=|P\cap G|,\quad FP=|P\setminus G|,\quad FN=|G\setminus P|.
\]

\[
IoU=\frac{TP}{TP+FP+FN},\quad
Precision=\frac{TP}{TP+FP},\quad
Coverage=Recall=\frac{TP}{TP+FN},\quad
F1=\frac{2TP}{2TP+FP+FN}.
\]

| 指標 | 物理／任務意義 | 不能推論什麼 |
|---|---|---|
| Mean IoU | 同時懲罰漏選和多選；適合衡量 required set 與 selected set 的一致性 | 不是畫面 SSIM |
| Micro F1 | precision/recall 的綜合表現，延續原 PPT | 與 IoU 不是獨立證據 |
| Coverage / recall | 真正所需 partitions 中有多少被保留 | 不知道每個 missed cell 對 rendering 的影響 |
| FN mean | 每次平均漏幾個必要 partitions | 不同大小 cell 的視覺重要性未加權 |
| FP mean | 每次平均多選幾個非必要 partitions | 不等於多傳的 bytes |
| Selected mean | 每次要處理/傳送的 partition 數量 proxy | cell payload 大小不同時不能換算 bandwidth |
| Exact match | 是否整個 ROI set 完全正確 | 粒度敏感，不當唯一主指標 |

同一 confusion counts 的 F1=2IoU/(1+IoU)，但先逐 origin average 與先 pooled counts 再計算不等價；不要混用兩種 aggregation。稀疏 ROI 的 TN 可能很多，因此 per-cell accuracy 容易看起來漂亮，不作主要判準。

### Empty-set convention

- P、G 都空：IoU=1，符合原 repo 的 set agreement convention。
- F1 或 precision 分母為0：定義0，與原報告口徑一致。
- 某個 aggregation 中 required count=0：coverage=null，不填1以假裝完整涵蓋。
- 同時報告 empty_ground_truth_fraction 與 mean IoU on nonempty GT，避免大量空視野主導結果。

### Pose diagnostic

Translation error=Euclidean distance，單位公尺；rotation 使用 sign-invariant quaternion geodesic angle，報 mean/p95 degrees。它們用來判斷 ROI gain 來自 translation 或 rotation，不取代 ROI 指標：同樣 pose error 在不同場景幾何可能造成不同 ROI error。

原 `Scene.mask` 是 **partition-center membership**，沒有 mesh/box exact intersection 或 occlusion。均勻 cube approximation 更不是實際 server partition 邊界；正式實驗應提供真實 world-space partition centers，且保持 coordinates、FoV、near/far 定義一致。

## 5. Aggregation 與信賴區間

報告有三層：

1. `pooled`：全部 origin counts 合併；長 recording 權重較大。
2. `recording_macro`：每個 trajectory/recording 先計 metrics，再對 recordings 等權平均。**本版 checkpoint selection 與配對主比較採這層。** 若一個 recording 因 gaps 分成多個 cache trajectories，這些 segment 會各有 macro 權重；須在論文說明，或外部依真實 recording 重新 aggregate。
3. `scenario_macro`：每個 scenario 內先對 recordings 平均，再對 scenarios 平均；scenario 都有有效 metadata 才輸出。

Paired comparison 在同一 origin 計 `right IoU − left IoU`，再取 recording means。Bootstrap 以 `group_id` 為 cluster 整組抽樣，不獨立抽高度重疊的 windows。預設2000 resamples、seed42；輸出 percentile 95% CI。抽到某個 group 時，該 group 的全部 recording means 一起加入，所以 estimand 是 recording-macro，不是 subject-macro。

只有一個 group 時 CI=null，因為無法從單一 cluster 合理估計群體間不確定性。很少的 groups 仍會讓 CI 不穩，不能因為算得出來就宣稱統計充分。

這些是**每 horizon、未作多重比較校正的描述性 CI**，不是所有 horizons 都通過顯著性檢定。正式論文應在 test 前指定 primary comparison/horizon，或另行做事先定義的 multiplicity control；本工具沒有偽裝成已提供 Holm-corrected p-values。

## 6. Horizons 與 seeds 的實驗安排

保留先前主軸 50/100/150/200/250ms；400/600ms 用來檢查前研究較明顯的 translation 訊號，800/1000ms 作延伸。原論文與原 repo 已有 recursive rollout，延長 horizon 本身不是這次的新貢獻。

預設 joint adapter 同時訓練所有指定 horizons，validation checkpoint 對 horizons 等權。若研究只在短 horizon 主張增益，應事先建立短 horizon 的實驗設定，不可看完 test 後才改 validation 權重挑版本。

建議預先登記一個 short-horizon primary endpoint（例如250ms）與 600ms translation 的 secondary diagnostic；這是實驗規劃建議，不是教授已確認的條件。所有原有 horizons 仍完整報告，不只展示改善最大的一點。

每個新增方法至少跑 seeds42/43/44，逐 seed 使用相同 data protocol。`compare` 的 bootstrap 是 data uncertainty，不涵蓋 retraining seed variance；seed 的 mean/SD 應另行彙整，不把 seed×window 全部當成獨立樣本。此版命令可逐 seed 執行，沒有假造跨 seeds 的自動顯著性結論。

## 7. 不降低 coverage，才談 fewer / more precise partitions

Primary geometric comparison 用相同固定 FoV，先看 IoU、FP、FN、coverage。Selected count 降低但 coverage 同時明顯下降，不能稱為成功。

為了回答 deployment policy 的問題，另外定義 prediction-side margin m：cone FoV 或 frustum 的水平/垂直 FoV 增加 2m；**GT FoV 始終不變**。

在 validation 每個 horizon 選：

\[
m_h^*=\arg\min_{m\in\mathcal M}\overline{|P_{h,m}|}
\quad\text{s.t.}\quad Coverage^{valid}_{micro}(h,m)\geq\rho.
\]

預設 ρ=0.95，只是可指定的 coverage 需求，不是畫質定律。各方法在 validation 選自己的 margin 後鎖定到 test；輸出實際 achieved test coverage 與 selected/FP/FN。若 validation 無候選可行，就報 infeasible。

**這不是數學上保證 test quality 的方法。** Test 未達95%就如實報告。即使兩者 coverage 差距看起來小，若要正式宣稱「non-inferior coverage」還需要事先設定 tolerance 並另外作 non-inferiority uncertainty 分析；本工具的 IoU bootstrap 不是該檢定。

## 8. Missing RF、latency 與 causal stress

每份 report 同時輸出 all origins 與 RF-available subset。主實驗不得只報 RF 可用的容易樣本而不交代 missing rate；secondary subset 是用來區分資料缺測和模型能力。

`--delivery-delay-ms` 只延後 available time，不改 GT、origin 或 horizon。可事先指定0/50/100/200ms等 stress levels。延遲導致 stale window 時會精確 fallback 到 BAF；不把 future map 平移到過去補足。

目前測試涵蓋：future raw CSI mutation 不影響先前 maps；future map mutation 不影響當前 inputs；超出 raw history 的 maps 排除；時計不確定性；資料重複跨 splits；missing fallback 的 pose/ROI 一致性。

任意 time-shuffle / reversed-sequence 可以作診斷，但本版未把它當成已完成對照命令；不能在結果表寫成有跑過。Direction ablation 使用 abs_change，不把它誤稱為 Doppler sign ablation。

## 9. 時間穩定性與 gate 的額外分析

每個 horizon 在相鄰 observed origins 上，計算 predicted changed-partition set 與 GT changed-partition set 的不一致比例：

\[
E_{trans}=\frac1K\sum_k \mathbf1[(P_t\triangle P_{t-1})_k\ne(G_t\triangle G_{t-1})_k].
\]

這比只量 prediction 自己變動多少更合理，因為使用者真的轉頭時，完全不變的預測不是穩定而是落後。跨 gaps 不計 transition。這仍是診斷，不是「畫面 jitter 已改善」的直接測量；大量 unchanged cells 也會稀釋其值。

Expert gate 的 per-horizon weight 可從 CSV 畫曲線。Residual 的 `gate` 欄是原 BAF α，不代表 residual 如何修正。原 α 是 weighted behavioral proxy 的輸出，不是校準的「Explorer 勝過 Focused」機率。

`oracle` 只比較有限 grid 的 terminal F/E mixtures，並把原 BAF 當候選保底。它以真實 ROI 選 candidate，是不可部署的 hindsight diagnostic。若有限 grid headroom 很小，表示這組候選不值得過度期待；不能斷言所有 continuous α 或逐步 α 方法的理論上界都小。

## 10. 預期圖表與可支持的結論

`plot` 產生八張獨立圖：IoU、F1、coverage、FP、FN、selected、translation error、rotation error，全部清楚標 recording-macro。Curve 是描述性 mean，不自帶 CI；CI 來自 paired comparison JSON，不能把線粗細當不確定性。

正式報告至少提供：共同 sample / recording / group 數；horizon curves；HMD vs RF paired ΔIoU 及 CI；固定 coverage policy 表；representation/metadata ablations；delay stress；硬體和 single-request latency 範圍。圖表未完成的項目不要寫成已觀察到。

結論判斷例：

- 只勝過原 BAF、未勝過 HMD residual：可說 adapter/task objective 改善，不能歸因 CSI。
- 勝過 HMD residual、但 metadata-only 也一樣好：尚不能說 spectrum dynamics 提供關鍵資訊。
- 只有長 horizon translation 改善：限縮到該 task/range，不宣稱所有短期 rotation 皆改善。
- coverage相近且FP降低：支持幾何 ROI 選取更精確，但仍不能聲稱相同 SSIM 下省下多少 Mbps。
- 未發現增益：這是對 representation / data quality / task alignment 的研究結果，不應把 negative result 藏起來。

## 11. 尚待真實研究完成

真實 paired measurements、radio calibration、時鐘精度、跨 subject/活動/房間驗證、多人干擾、實際 rendered quality、bitrate、完整 pipeline 30Hz deadlines，目前沒有被本分支的 synthetic CI 證明。此版本提供可執行且有控制條件的實驗框架，不提供假造論文結果。
