# CSI-assisted ROI Selection：BAF-ROI 的 Heatmap 延伸

本分支基於 `2a75637da2e814205a56484a1e195c2922d9f9c0`。原始 `baf_roi/`、`motion_transformer_roi/`、`vendor/` 保持不變；原本 README 逐字保存在 [README_BASELINE.md](README_BASELINE.md)。

**研究問題：在相同 HMD history、相同切分、相同 future horizons 與 ROI 幾何下，當下已可取得的 CSI Heatmap，是否能提供超出 HMD history 的資訊，改善未來所需 partitions 的選取？**

教授提出的「Heatmap 輔助 α」是一個候選方案，不是先驗限制。本版實作兩條可比較路徑：

| 方法 | Heatmap 的用途 | 為什麼需要它 |
|---|---|---|
| `residual`，主要研究候選 | 依 horizon 修正凍結 BAF 的 future position / rotation | 不把 CSI 的作用限制成在兩個既有專家之間選比例；零初始化可回到原 BAF |
| `expert_gate`，機制比較組 | 依 horizon 調整 Focused / Explorer 的終點融合權重 | 檢驗「額外資訊主要幫助專家選擇」的假設；不是原 BAF 每一步 α 的原樣替換 |

兩者都有 `hmd` 與 `rf` 版本。真正檢驗 CSI 增量價值的比較是 **同一 adapter 的 HMD-only vs HMD+RF**，並同時比較原 BAF、Focused-only、Explorer-only、固定融合與 Motion Transformer。

> 已完成的是可執行程式、測試、合成資料整合驗證與評估工具。尚未使用真實同步 HMD–CSI 測資證明 accuracy gain。合成圖與 smoke test 不可當論文結果。

## 文件入口

| 文件 | 回答的問題 |
|---|---|
| [演算法與 pseudocode](docs/README_HEATMAP_ALGORITHM.md) | Heatmap 怎麼處理、為什麼；兩種方法、公式、loss、訓練與推論細節 |
| [完整架構與執行方式](docs/README_HEATMAP_ARCHITECTURE.md) | 模組/function 分工、資料格式、命令、線上 API 與部署邊界 |
| [Evaluation 與實驗設計](docs/README_HEATMAP_EVALUATION.md) | 每個對照與指標為什麼必要、matched origins、統計、coverage policy |
| [驗證紀錄](docs/HEATMAP_VALIDATION.md) | 實際執行的測試、CI、reference equivalence 與仍未證明的項目 |
| [來源與既有程式對照](docs/HEATMAP_SOURCES.md) | 力行論文／投影片／原 Heatmap 程式的具體對照與雜湊 |
| [原始 baseline 文件](README_BASELINE.md) | 原本 prepare、BAF 四階段訓練、MT、ROI 與複雜度分析 |

## 安裝與先跑通軟體

建議獨立 Python 3.11 環境。依硬體選擇 PyTorch；CI 使用 CPU PyTorch 2.10.0。

```bash
python -m pip install -r requirements-heatmap.txt
python -m baf_roi audit
python -m pytest -q
python scripts/smoke_heatmap.py --output runs/smoke_heatmap_01
python -m baf_roi_heatmap --help
```

`smoke_heatmap.py` 自行產生 **synthetic** HMD 與已知 steering CSI，執行原 BAF、MT、四種 adapters、共同 origins 評估、缺測 fallback、coverage policy、oracle、benchmark 與繪圖。輸出目錄必須是新的，避免覆寫實驗。

## 真實實驗的最短路徑

先沿用原專案產生 HMD cache 與 `baf_roi.pt`；raw CSI 必須含實測 timestamps、availability 與實際 subcarrier frequencies。不可從 PNG 或假定的 100 Hz 反推物理資料。

```bash
python -m baf_roi_heatmap generate --raw data/radio_record.npz --config configs/heatmap_generator.example.json --output data/maps/record_01.npz
python -m baf_roi_heatmap manifest-template --cache cache/experiment01 --maps-directory data/maps --output data/rf_manifest.json
```

第二個命令產生的是 **需要人工核對的模板**：校對每段 HMD 對應的 RF 檔、clock offset / uncertainty、participant group 與 calibration 證據後才可訓練。模板刻意設為 `synchronization_confirmed=false`，不是要求把旗標盲目改成 true。

完成資料契約後：

```bash
python -m baf_roi_heatmap prepare --cache cache/experiment01 --baseline runs/experiment01/baf_roi.pt --rf-manifest data/rf_manifest.json --output runs/heatmap_prepared --horizons 50 100 150 200 250 400 600 800 1000
python -m baf_roi_heatmap train --prepared runs/heatmap_prepared --method residual --modality hmd --seed 42 --output runs/residual_hmd_42
python -m baf_roi_heatmap train --prepared runs/heatmap_prepared --method residual --modality rf --seed 42 --output runs/residual_rf_42
python -m baf_roi_heatmap evaluate --prepared runs/heatmap_prepared --checkpoint runs/residual_hmd_42/adapter.pt --output runs/eval_hmd_42
python -m baf_roi_heatmap evaluate --prepared runs/heatmap_prepared --checkpoint runs/residual_rf_42/adapter.pt --output runs/eval_rf_42
python -m baf_roi_heatmap compare --left runs/eval_hmd_42 --right runs/eval_rf_42 --output runs/paired_42.json
```

完整對照、資料格式、線上呼叫與所有注意事項見架構及評估文件。

## 目前範圍與誠實界線

- 已實作 **numeric AoA–ToF** 生成與時序模型。尚未實作 Doppler / 3D cube 融合；Heatmap 正負差分不等於人體速度或 Doppler。
- 原 MUSIC 的 covariance / steering / spectrum 核心有數值對照；前處理與時間窗則刻意修正成可用於 causal prediction 的版本。
- 不自行產生硬體校正、時鐘同步或人體部位標籤。Heatmap 一個亮點不必然是一個 body part。
- ROI 使用現有 partition-center 幾何。Coverage 不是 SSIM，selected count 不是實際網路 bytes。
- 實驗固定場景參與 soft-ROI training loss；本版沒有宣稱跨新虛擬場景泛化。
- Live RF acquisition / hardware driver / network / renderer 不在此套件。Benchmark 不等於完整系統達成 30 Hz。
