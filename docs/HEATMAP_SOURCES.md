# 來源、版本與設計對照

## 原始 repository

基準 commit：`2a75637da2e814205a56484a1e195c2922d9f9c0`，commit message `add complexity evaluate`。

原 `baf_roi/`、`motion_transformer_roi/`、`vendor/` 沒有被本延伸改寫。原 README blob `518fc7a3653a626276b99d906e9dee1128564a78` 逐字保存為根目錄 `README_BASELINE.md`。新增 package 是獨立的 `baf_roi_heatmap/`。

## 使用者提供的研究資料（未重新公開 PDF / 投影片）

| 來源 | 本次參考重點 | 對應設計 |
|---|---|---|
| `力行論文(5).pdf`，Ch.5，PDF45–47 / 印刷37–39 | causal confidence、Explorer=1/Focused=0、μ=0.6 EMA | 保留 original BAF，不把 α 解釋為校準的專家勝率 |
| 同論文，PDF76–78 / 印刷68–70，Tables11–13 | Explorer accuracy 與 Focused stability 的 trade-off | Explorer-only、fixed fusion 必須列為控制，不能假定 α 最佳 |
| `DOC-20261009-WA0000(1).pdf`，Finding3/4、pp.10–13 | signed motion 的動機、沒有 universal body–gaze chain、mocap 不等於 RF | 不強制 skeleton/body-part 估測；保留時序與符號但不冒稱 Doppler |
| `20261016.pptx`，research goal / baseline / evaluation / future-work 頁面 | 相同歷史與 origins 的 HMD-only comparisons，最終 ROI 而非只 pose | matched dataset、F1/IoU、FP/FN/coverage，新增 RF 的證據另做 |
| 2026-10-14/15 的研究討論 | more precise partitions；目前無 SSIM | 幾何 coverage proxy 與 rendering-quality claim 分開 |
| 2026-10-27/29 的 Heatmap 討論與程式 | CSI 頻率來源、MUSIC、map 物理意義 | 明示 carrier/subcarrier/timestamps/phase 與因果契約 |

前研究及先前 HMD-only baseline 的數字不能搬成 RF 方法的實驗成績。使用者先前 50–250ms 的主範圍保留；400/600ms 與至1000ms 的延伸是額外評估，不宣稱原 BAF 不支援長 rollout。

## 原 Heatmap 程式 SHA256

| 原檔案 | SHA256 |
|---|---|
| `MUSIC.py` | `5387d0df1f7dd04f30bd6e36d3a2e7b3f8835f267c7b196224168d5b7afb9f78` |
| `pre_processing.py` | `7db62ee0e46bf7e204d1c7d51728ddf10e55802b0644bd99de65f4e6c724a968` |
| `signal_processing.py` | `c429cad0feb4e1b0d74e4673262a66fecb8d8116f804dfdf8d76ddb9095439bf` |
| `main(20261027-120045).py` | `5f2c36417097a1b5d88fe77f4da3d7740d98485a14a0a34b237cfeecdbc00a9e` |

本版保留 numerical AoA-ToF core 的相同順序／運算，但不重新公開使用者的全部原始附件。`scripts/check_heatmap_reference.py` 接收原 `MUSIC.py` 路徑，可重跑同 raw window 的 covariance、steering、spectrum 比對。

```bash
python scripts/check_heatmap_reference.py --reference-file /path/to/original/MUSIC.py --output runs/reference_check.json
```

實際已執行 sin / cos 兩種 convention；最大 covariance error `1.3322676295501878e-15`、最大 spectrum error `4.005684672847565e-13`。此驗證不包含原全錄影 DWT、置中 moving average、PNG plotting，也不代表 real-world phase calibration 成功。

## 外部方法學來源

Kotaru et al., **SpotFi: Decimeter Level Localization Using WiFi**, ACM SIGCOMM 2015，作者提供 PDF：
https://web.stanford.edu/~mkotaru/sigcomm15-spotfi.pdf

使用這個來源僅支持 coherent CSI 的 antenna/subcarrier phase、joint AoA/ToF 與 timing/phase sanitization 的重要性。本版不宣稱重現 SpotFi localization，不借其 localization accuracy 替人體 sensing 或 ROI gain 背書。

## 有意不宣稱的內容

本專案沒有提出 universal body-part-to-heatmap mapping；沒有證明 subcarrier grid / carrier 參數符合使用者當時的硬體；沒有把軟體測試當真實 CSI 增益；也沒有把 bounded residual 網路本身宣稱為已獲確認的學術新穎性。碩論貢獻應由明確問題、可信量測、可反駁的比較與結果共同構成。
