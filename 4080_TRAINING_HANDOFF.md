# 4080 训练交接说明

本文件用于在 RTX 4080 笔记本或新的 Codex 会话中快速恢复当前风预测工作。

## 先读文件

新电脑 clone 仓库后，按顺序先读：

1. `README.md`
2. `CODEX_HANDOFF.md`
3. `风预测规划/任务日志.md`
4. 本文件

## 当前主线

当前主线不是旧控制仿真收尾，而是：

```text
历史 10 分钟风速 / 风向
-> 未来 60 分钟风况预览
-> 分段风变风险输出
-> 后续主动压载提前调节输入
```

当前已具备两套可训练数据集：

```text
data/processed/wind_ml_10min/ballast_decision_dwd_helgoland_segmented_v1/
data/processed/wind_ml_10min/ballast_decision_fino1_segmented_v1/
```

当前主比较结果：

- segmented GRU:
  `outputs/wind_prediction/gru_segmented_head_v1/`
- segmented LightGBM:
  `outputs/wind_prediction/lightgbm_segmented_residual_e300/`
- residual LSTM:
  `outputs/wind_prediction/lstm2_h128_b2048_e25_dir002/`

当前判断：

- GRU 在压载关注事件和三段风险上最好，但对 LSTM 的优势不大；
- 下一步不应继续盲目堆结构；
- 更有价值的是：
  1. 先用现成的 FINO1 数据复训；
  2. 或把当前 GRU 接进压载控制预览接口，验证控制收益。

## 必须搬的数据

至少搬整个目录：

```text
data/processed/wind_ml_10min/ballast_decision_dwd_helgoland_segmented_v1/
data/processed/wind_ml_10min/ballast_decision_fino1_segmented_v1/
```

该目录约 `2.2 GB`，包含当前训练和评估最核心的数据：

- `X_train.npy`
- `X_validation.npy`
- `X_test.npy`
- `y_uv_*.npy`
- `y_uv_raw_*.npy`
- `y_speed_dir_raw_*.npy`
- `y_event_*.npy`
- `y_summary_raw_*.npy`
- `sample_index.csv.gz`
- `metadata.json`
- `scaler_train.json`
- `event_thresholds.json`

## 建议一起搬的数据

如果要继续做特征工程或重建数据集，建议同时搬：

```text
data/processed/wind_ml_10min/supervised_learning_table_10min.csv.gz
data/processed/wind_ml_10min/canonical_observations_10min.csv.gz
data/processed/wind_ml_10min/fino1_platform_10min/
data/reference/
```

## 建议一起搬的结果目录

如果新电脑要直接复查现有结果，建议搬：

```text
outputs/wind_prediction/gru_segmented_head_v1/
outputs/wind_prediction/lightgbm_segmented_residual_e300/
outputs/wind_prediction/lstm2_h128_b2048_e25_dir002/
outputs/wind_prediction/ppt_figures/
```

## 暂时不需要搬的内容

当前不是主线训练必需：

```text
data/processed/wind_ml_10min/sequences/
data/processed/wind_ml_10min/sequences_dwd_helgoland/
data/raw/wind_10min/DTU_wind_farm/
data/raw/wind_10min/NDBC_continuous_winds/
archive/
无用/
```

注意：`archive/` 在“做风预测训练”时不是必需，但如果要做控制接入实验，则需要搬。

## 4080 环境建议

建议使用虚拟环境：

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

确认 CUDA：

```bash
python - <<'PY'
import torch
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no cuda")
PY
```

训练时优先加：

```text
--device cuda
```

## 当前最值得复现实验

复跑 segmented GRU：

```bash
python scripts/modeling/train_ballast_lstm.py \
  --dataset-dir data/processed/wind_ml_10min/ballast_decision_dwd_helgoland_segmented_v1 \
  --output-dir outputs/wind_prediction/gru_segmented_head_v1 \
  --epochs 30 \
  --batch-size 2048 \
  --hidden-size 128 \
  --num-layers 1 \
  --dropout 0.10 \
  --event-loss-weight 0.02 \
  --direction-loss-weight 0.02 \
  --patience 6 \
  --lr-patience 2 \
  --lr-factor 0.5 \
  --threads 8 \
  --residual-regression \
  --model-type gru \
  --device cuda
```

复跑 segmented LightGBM：

```bash
python scripts/modeling/train_lightgbm_wind_baseline.py \
  --dataset-dir data/processed/wind_ml_10min/ballast_decision_dwd_helgoland_segmented_v1 \
  --output-dir outputs/wind_prediction/lightgbm_segmented_residual_e300 \
  --n-estimators 300 \
  --learning-rate 0.04 \
  --threads 8 \
  --threshold-mode best_f1
```

复跑 focal-loss GRU 仅作对照：

```bash
python scripts/modeling/train_ballast_lstm.py \
  --dataset-dir data/processed/wind_ml_10min/ballast_decision_dwd_helgoland_segmented_v1 \
  --output-dir outputs/wind_prediction/gru_segmented_focal_v1 \
  --epochs 30 \
  --batch-size 2048 \
  --hidden-size 128 \
  --num-layers 1 \
  --dropout 0.10 \
  --event-loss-weight 0.02 \
  --direction-loss-weight 0.02 \
  --patience 6 \
  --lr-patience 2 \
  --lr-factor 0.5 \
  --threads 8 \
  --residual-regression \
  --model-type gru \
  --event-loss-type focal \
  --focal-gamma 2.0 \
  --device cuda
```

注意：当前结果显示 focal-loss GRU 没有超过原 segmented GRU，只是保留为负结果证据。

## 下一步训练方向

当前不要直接上大 Transformer。优先级是：

1. 先用 `ballast_decision_fino1_segmented_v1` 在 4080 上复训当前 GRU / LightGBM；
2. 然后把当前 `gru_segmented_head_v1` 或 FINO1 复训版本接到压载控制预览接口，做控制收益验证；
3. 如果还要做模型训练，优先做少量稳定性复验和多随机种子，而不是继续堆新的结构。
