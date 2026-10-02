# 資料、座標、時間與輸出合約

## 1. 原始來源的差異

| 面向 | BAF-Pose 原始 code | 使用者舊 ROI code／本次 adapter |
|---|---|---|
| 檔案 | 已 resample、已附 supervision 的 flat CSV | `db/<session>/agent6*/...csv` 的 recorder 記錄 |
| 时间 | `Timer`、30 Hz | payload／CSV timestamp；重新建立 causal 30 Hz history |
| Pose | `HeadX/Y/Z`、`HeadRX/RY/RZ` | `payload_json.pos`、`rot_deg` 或 `rot` |
| 行為標註 | 預先存在的 `Label` 等欄位 | 原作者 centered supervision 離線生成；不是模型 input |
| ROI | 沒有 pose→ROI 的 downstream evaluation | 虛擬物件 partitions 的 multi-label mask |
| Dataset split | 原作者 file split | 新資料按 session 分組，允許沿用舊 split manifest |

舊 `baseline.md` 的單一 HMD 所在 voxel 定義已被 `REVISE.md` 與目前 `roi_label.py` 的 multi-label definition 取代。本次不採用那個 single-class target。

## 2. Recorder CSV

最少需要 `payload_json` 這一欄，每個 cell 是 JSON object，例如：

```json
{
  "ts_utc": "2026-09-29T12:00:00.123Z",
  "pos": [150.0, 20.0, 165.0],
  "rot_deg": [30.0, 5.0, 0.0]
}
```

此例的座標數字只展示格式，不代表真實使用者資料。`pos` 與 rotation 必須是三個有限數值的 JSON arrays。沒有 `rot_deg` 時接受 `rot`，但仍解讀為度數，不自動猜 radians。

時間優先序：`payload.ts_utc` → CSV `ts_utc` → CSV `recv_unix`。ISO 必須有 timezone；接受 `T12:00:00` 與舊記錄中的 `T12-00-00` 形式。numeric timestamp 的單位由 `timestamp_numeric_unit` 指定；`recv_unix` 固定是 Unix seconds。

同一 timestamp 同一 pose 可以去重；同 timestamp 不同 pose 直接報錯。不同 CSV 不任意拼接。多個 CSV 仍共享 session split，避免資料洩漏。missing rotation 不以零代替；invalid rows 不無聲消失。

## 3. 6DoF convention 與單位

目前 adapter 對應舊 `roi_label.py` 的 convention：

```text
position: world xyz
rotation array: [yaw, pitch, roll] in degrees
camera forward: local +X
world/camera up at zero rotation: +Z
positive yaw: +X toward +Y
positive pitch: look upward
R_camera_to_world = Rz(yaw) @ Ry(-pitch) @ Rx(roll)
```

因此 SciPy extrinsic xyz Euler 介面需要 `[roll,-pitch,yaw]`，不能直接使用原 array 順序。Quaternion 一律儲存 `w,x,y,z`。Confidence 看 `[x_m,y_m,z_m,rx_deg,ry_deg,rz_deg]`，branches 看 `[normalized_xyz,qw,qx,qy,qz]`。

**這是從舊程式的計算推得，不是對 recorder 的獨立驗證。** 必須確認 recorder 的實際輸出就是這個 convention；還要確認它與 object coordinates 共用原點、朝向、尺度。若你的 HMD 實際是其他座標系，應明確實作／驗證新的 adapter，不只把 confirm flag 設成 true。

Raw position 可選 `cm` 或 `m`；BAF 內部、scene centers、輸出位置、distance planes 一律是公尺。原 adaptive JSON 的 `_cm` 欄位永遠按公分解讀。

## 4. Causal sample 與 horizon

對每個實際 observed origin `t`，模型窗口是：

```text
[t - 29/30 s, ..., t - 1/30 s, t]
```

窗口的插值只能使用 timestamp ≤ t 的 raw observations。資料最初的約一秒沒有足夠 history，不生成預測。資料 gap 超過設定值會切段。給 public API 的 timestamp 必須严格遞增，不會替你排序把未觀測的 future rows 混入 history。

Training target 為 `t + 1/30 s`；評估 target 為 `t + h/1000 s`。兩者由觀測 trajectory 在該時刻 interpolation 得到，不代表 native sensor 一定在這些時刻有原始 sample。GT 可以使用未來資料來定義答案，但 input 不可以。

測試 horizons 是 [50,100,150,200,250] ms，共用有完整 future coverage 的 origins。30 Hz rollout 產生 8 個 future poses，最遠到 266.667 ms，最後對指定 horizons 取值／插值。

### v2 的 supervision window 是另外一個概念

模型仍是上述 30-pose past-only input。當前 behavior target 使用作者 centered `[t-0.5,t+0.5]`；下一步 target 的行為標籤以 `t+1/30` 為中心。這些都只用於離線 supervision / branch selection；online prediction 不需要未來資料。原 MIN_SAMPLES=5 與片段邊界處的 partial centered windows 保留。

Scoring-only 角度 adapter 把 recorder `[yaw,pitch,roll]` 排成 `[pitch,yaw,roll]`，以符合原碼 `HeadRY`=yaw。這不是 physical xyz conversion；不能拿它替代 `pose.py` 的 `[roll,-pitch,yaw]`。

Supervision grid 以每個實際 decision origin 重新對齊 30 Hz；input grid 保持 causal cutoff。詳細差異見 PREPARATION_REVIEW。Cache 及 checkpoint 均為 version 2；v1 必須重建／重訓。

## 5. 場景幾何來源

### 優先：提供真實 world-space partition centers

`partition_centers_file` 是 JSON：

```json
{
  "coordinate_convention": "legacy_ypr_xforward",
  "unit": "m",
  "partition_ids": ["actorA:part_0", "actorA:part_1"],
  "centers_m": [[4.5, -0.5, 1.2], [4.5, -0.4, 1.2]]
}
```

順序就是輸出 mask 的順序。IDs 必須 unique，centers 數量必須相同。上例只有兩點，僅說明格式。你應從真正的 volumetric partition definitions 匯出中心，不用猜測 bounding boxes。

### 舊相容：adaptive config + uniform grids

先取 `defaults.config`，再以各 `streams[].config` 覆蓋。若 stream 沒有 config dict，採 stream 層的 `obj_transform_*`；這與舊 reader 一致。positions 來自 `obj_transform_pos_x_cm/y_cm/z_cm`。按原 streams 順序，對每個中心產生 uniform `grid_dim³` cells；同物件內 x-fastest，之後 y，最後 z。

當 grid_dim=8，輸出維度是 `objects_count × 512`；沒有 hard-code 六個物件或 3072 維。`object_size_cm` 可為 scalar 或 [x,y,z]，但這個簡化 adapter 使用所有物件共通的 size。不同物件真實大小、rotation、scale、occupied partition layout 應改用真實 centers 檔，不要假裝 uniform cube 精確代表它們。

此模式要將 `allow_uniform_grid_approximation` 設 true，並確認 size／geometry。nonzero rotation 或 non-unit scale 會報錯，以免無聲丟掉 transformation。沒有模擬未提供的 streaming-server 程式碼。

## 6. ROI membership 的兩個明確定義

`legacy_center_cone`：中心與 camera forward 的夾角 ≤ fov/2；max distance 可選。它是舊程式的圓錐近似，roll 不改變圓錐。

`rectangular_center_frustum`：把中心轉到 camera local frame，以 x 為 depth，比較 y/x 與水平半 FoV、z/x 與垂直半 FoV，並套 near/far。這裡是單相機、對稱矩形 frustum；不是直接重現真實 HMD 雙眼不對稱投影矩陣。

**兩者都只是 center membership。** 不含遮蔽、不檢查整個 bounding box 是否相交、不確認中心所在 partition 是否有 occupied points。這是 label definition 的限制，而不是 BAF-Pose 已驗證的 renderer visibility。

同一實驗對 predicted pose 和真實 pose 使用同一份 scene。所有 baseline 比較必須共用同一 scene、FoV、near/far、center order 與 target interpolation 協定。若不同 session 使用不同場景，本版的單一 scene evaluation 不可直接混用；需按場景分開評估或另擴充明確的 per-session scene mapping。

## 7. 輸出

`Predictor.forecast_batch` 的低階 input 是 `[B,30,6]`，位置已是公尺；`predict_observations` 則接受原記錄單位並自行處理。不要在已轉成公尺後又讓上層按 cm 轉一次。

主要輸出：positions_m `[B,H,3]`、quaternions_wxyz `[B,H,4]`、rotations_ypr_deg `[B,H,3]`，geometry mapping 產生 boolean mask `[B,H,P]`。H 的順序跟請求 horizons 一致，P 的順序由 `scene_snapshot.json` 的 partition_ids 指定。

NPZ 的 masks 用 little-bit-order packbits 壓縮；`num_partitions` 可還原有效 cells。metadata 包含 actual origin epoch seconds、精確 requested horizons、真實/預測 pose、當前 alpha 與 rollout alpha。CLI 單次預測不強迫提供 future ground truth。
### Cache 的使用限制

NPY cache 是 prepare 已驗證後的不可變實驗產物，請勿手動編輯。為避免 shuffled training 每次換 trajectory 都重新掃描整份檔案，mmap cache 路徑不重做 full-array validation；每個送入模型的 history 仍做 finite check。要改 raw data、split 或 preprocessing，請重建到新的 cache 目錄；不要只改 cache 的個別 arrays 或 manifest。
