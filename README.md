# Kaggle Orbit Wars — 銀牌（第 58／4,729 名，前 1.23%）

比賽：[Kaggle Orbit Wars](https://www.kaggle.com/competitions/orbit-wars)（2026-07-08 頒發）

本 repo 只收錄**核心程式**：能說明並（在你自備資料下）重建最終提交的配方——資料建構、特徵、模型、訓練、純 NumPy 推論與提交打包、對局評估，以及 JAX 自我對戰 PPO 主程式。

## 方法概要

- **以行為克隆（BC）為主**：從官方天梯對局 replay 中，依公開排行榜分數挑選強玩家當示範者，模仿其每步出手；贏局加權（×2）、輸局保留；對「沒有出手」的大量狀態做再平衡，避免模型塌成永遠不動。
- **模型**：entity attention（以行星為 token 的 transformer）＋分解式動作頭（每顆行星各自決定：是否出兵、目標行星、派出比例 20 檔）。每顆行星 25 維特徵（4 人局再加「是否 4 人局」與「我方駐軍排名」共 27 維）。
- **模型放大到與資料量相稱**：參數量由約 1.4M 一路放大到約 30M，實戰強度大幅躍升（監督準確率幾乎不動，必須看實戰）；加容量往**寬度**走，深度固定 3 層（加深反而表徵趨同）。
- **根因修復比調參有效**：例如 replay 反推目標時行星沒有跟著轉動、把強玩家的正確出手標成送頭的標籤錯誤；4 人局誤用 2 人局設定；發射點偏移造成的提前量誤差；地圖庫出生位偏差。
- **強化學習**：JAX 重寫、與官方引擎逐步對齊的遊戲環境上做自我對戰 PPO（雙 trunk critic、以 BC 為 KL 錨），4 人局用「只有第一名得分」的獎勵加上以產能為勢能的 reward shaping 破解龜縮。**在本次實驗的規模與設定下**，PPO 未明顯超越 BC，最終提交為 BC。
- **提交為純 NumPy 推論**（JAX 初始化太慢、超過單步時限），單一檔案內嵌權重。

## 最終部署

2P／4P **dispatch** 單一提交：開局時偵測玩家數，2 人局用 2P 模型、4 人局用 4P 模型。

| 槽 | 2P 模型 | 4P 模型 |
|---|---|---|
| 主（本 repo 主要說明） | h640／3 層，只用 2 人局資料、示範者分數 ≥1500，取第 2 epoch（`train_bc_2p_h640.sh`） | 「L3-pack」：h768／3 層＋4 人局兩欄特徵，2P＋4P 混合資料、示範者分數 ≥1520，取第 1 epoch（`train_bc_4p_l3pack.sh`） |
| 附註 | h768／3 層、示範者分數 ≥1520 的混合資料模型（本 repo 未附獨立腳本；要近似這個配置，需在 `train_bc_2p_h640.sh` 改寬度為 768、門檻為 1520，並拿掉 `--players 2` 改用 2P＋4P 混合資料，仍只是近似） | 同上 L3-pack |

兩槽只差 2P 模型，天梯分數落在彼此的漂移範圍內。出兵角度用提前量瞄準＋太陽碰撞防護（`segment` 模式，預設）。

## 內容

| 檔案 | 用途 |
|---|---|
| `gen_bc_demos_replay.py`、`gen_resumable.sh`、`obs_state.py` | replay → BC 資料 chunks（可續跑、按日期分片） |
| `jax_min_env_spike.py` | JAX 遊戲環境（與官方引擎對齊） |
| `jax_lean_selfplay.py` | 特徵、`LeanEntityPolicy` 模型、PPO 自我對戰主程式 |
| `jax_train_runtime.py` | JAX 訓練執行環境設定 |
| `bc_train_clone.py`、`train_bc_2p_h640.sh`、`train_bc_4p_l3pack.sh` | BC 訓練與最終兩個模型的配方 |
| `extend_ckpt.py` | 2P 模型輸入由 25 維擴成 27 維（新增欄零權重，行為不變），以便與 4P 模型共存於 dispatch |
| `submission/lean_submission_numpy.py` | 純 NumPy 推論範本（權重欄位為空） |
| `submission/_build_dispatch_submission.py`、`submission/_build_lean_submission.py` | 把 checkpoint 嵌入範本，產生 dispatch／單模型提交檔 |
| `eval_2p_full.py`、`eval_4p_full.py`、`measure_sun_deaths.py`、`verify_dispatch_route.py`、`eval_lean_verdict.sh` | 官方 env 對局評估、4 人局量尺、dispatch 路由驗證 |
| `run_ppo_dual_4p_melee.sh`、`build_map_bank.py` | 4 人局 PPO 微調與其所需的地圖庫 |

## 怎麼執行

```bash
# 安裝（CUDA 12 GPU；訓練需約 16GB VRAM）
pip install -r requirements.txt

# 0. 自備資料
#  - 官方 Orbit Wars 天梯對局 replay JSON，依日期放：replays/YYYYMMDD/**/*.json
#  - 公開排行榜 CSV（欄位 Rank,TeamName,Score），用來依示範者分數篩選

# 1. 產生 BC chunks（可續跑；中斷後重跑同一指令）
LB=leaderboard.csv RD=replays ./gen_resumable.sh b15 0 4 200000   # → bc_chunks/b15_*_c*.npz

# 2. 訓練
./train_bc_2p_h640.sh       # → ckpt/bc_2p_h640.ep2（取第 2 epoch）
./train_bc_4p_l3pack.sh     # → ckpt/bc_4p_l3pack.ep1（取第 1 epoch）

# 3. 2P 模型擴成 27 維，並標記 4 人局特徵（dispatch 只讀 2P 模型的 config）
python extend_ckpt.py ckpt/bc_2p_h640.ep2 ckpt/bc_2p_h640_x27 2
python -c "import pickle;d=pickle.load(open('ckpt/bc_2p_h640_x27','rb'));d.setdefault('config',{}).update(is_4p_feature=True,rank_feature=True);pickle.dump(d,open('ckpt/bc_2p_h640_b27','wb'))"

# 4. 打包提交（Kaggle 單檔上限 100 MiB；超過就把 main.py 放在 zip 頂層）
python submission/_build_dispatch_submission.py ckpt/bc_2p_h640_b27 ckpt/bc_4p_l3pack.ep1 submission/sub_dispatch.py
mkdir -p d && cp submission/sub_dispatch.py d/main.py && (cd d && zip -9 ../submission.zip main.py)

# 5. 驗證與評估
python submission/_build_lean_submission.py ckpt/bc_2p_h640_b27 submission/sub_std2p.py
python submission/_build_lean_submission.py ckpt/bc_4p_l3pack.ep1 submission/sub_std4p.py
OW_DISP=submission/sub_dispatch.py OW_STD2P=submission/sub_std2p.py python verify_dispatch_route.py 2   # 預期 0 mismatch
OW_DISP=submission/sub_dispatch.py OW_STD4P=submission/sub_std4p.py python verify_dispatch_route.py 4
OW_AGENT=submission/sub_dispatch.py OW_OPP=<你的對手.py> OW_WORKERS=8 OMP_NUM_THREADS=1 python eval_2p_full.py 120
OW_AGENT=submission/sub_dispatch.py OW_OPP=<你的對手.py> OW_NP=4 OMP_NUM_THREADS=1 python measure_sun_deaths.py 120

# （可選）4 人局 PPO 微調：預設純自我對戰
python build_map_bank.py --num-agents 4 --out kaggle_map_bank_4p.npz
BC=ckpt/bc_4p_l3pack.ep1 ./run_ppo_dual_4p_melee.sh 10
# 若要混入啟發式對手：OPP_PROD_FRAC=0.5 PRODUCER_MODULE=<你的啟發式.py>（介面見 jax_lean_selfplay.py 開頭）
```

## 不包含的東西與限制

- **不含模型權重**：所有提交檔都由上面的步驟自行訓練、打包產生。
- **不含 replay、排行榜 CSV**：請自行從 Kaggle 下載。
- **不含任何對手程式**：評估與 PPO 對手混合需要的啟發式 agent 一律由使用者自備（`OW_OPP`、`--producer-module`）。
- **只能重建配方，不能逐位元重現**：GPU 非決定性、資料下載時間點不同，都會使結果與原始提交有差異；同設定重訓的對戰勝率可相差約 10 個百分點。
