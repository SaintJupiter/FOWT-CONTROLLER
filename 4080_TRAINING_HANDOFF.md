# 4080 训练交接说明

本文件用于在 RTX 4080 笔记本或新的 Codex 会话中快速恢复当前风预测工作。

## 先读文件

新电脑 clone 仓库后，按顺序先读：

1. `README.md`
2. `CODEX_HANDOFF.md`
3. `风预测规划/任务日志.md`
4. 本文件

这几个文件已经记录当前数据、模型、结果、风险和下一步计划。

## 当前主线

当前项目主线不是旧的压载控制仿真，而是：

```text
历史 10 分钟风速/风向序列 -> 未来 60 分钟风变预测 -> 压载提前预判参考
```

当前主数据集：

```text
data/processed/wind_ml_10min/ballast_decision_dwd_helgoland/
```

当前最佳主模型：

```text
outputs/wind_prediction/gru_h128_b2048_e30_dir002/
```

当前强表格基线：

```text
outputs/wind_prediction/lightgbm_tabular_residual_e300/
```

当前分工况分析：

```text
outputs/wind_prediction/regime_threshold_analysis/
```

## GitHub 和大文件策略

仓库已经用 Git LFS 追踪：

```text
*.npy
*.pt
*.nc
*.zip
*.gz
*.joblib
```

如果 4080 电脑网络和 Git LFS 正常，理论上可以直接：

```bash
git clone git@github.com:SaintJupiter/FOWT-CONTROLLER.git
cd FOWT-CONTROLLER
git lfs install
git lfs pull
```

如果 Git LFS 下载慢、失败，或者大文件没有拉全，就用网盘手动搬下面这些文件。

## 必须手动搬的数据

如果只想继续训练当前 GRU / CNN-GRU / TCN 主线，至少搬整个目录：

```text
data/processed/wind_ml_10min/ballast_decision_dwd_helgoland/
```

大小约 `2.2 GB`。这个目录包含：

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

这是当前训练和评估最核心的数据。

## 建议一起搬的数据

如果还要重新做特征工程、重建数据集、或复现 persistence 分工况评估，建议同时搬：

```text
data/processed/wind_ml_10min/supervised_learning_table_10min.csv.gz
data/processed/wind_ml_10min/canonical_observations_10min.csv.gz
```

大小约：

- `supervised_learning_table_10min.csv.gz`：`346 MB`
- `canonical_observations_10min.csv.gz`：`39 MB`

注意：不要搬未压缩的：

```text
data/processed/wind_ml_10min/supervised_learning_table_10min.csv
```

它约 `878 MB`，只是 `.csv.gz` 的重复展开版本。

## 可选搬的模型和结果

如果希望新电脑直接复查当前结果，不重新训练即可评估，搬：

```text
outputs/wind_prediction/gru_h128_b2048_e30_dir002/
outputs/wind_prediction/lightgbm_tabular_residual_e300/
outputs/wind_prediction/regime_threshold_analysis/
outputs/wind_prediction/model_comparison_with_lightgbm.csv
outputs/wind_prediction/recurrent_model_comparison.csv
```

大小约：

- GRU 输出：`352 KB`
- LightGBM 输出：`24 MB`
- 分工况分析：`24 KB`

这些不是训练新模型的硬性必需项，但对对比和复现很有用。

## 暂时不需要搬的内容

下面这些当前不是主线训练必需：

```text
data/processed/wind_ml_10min/sequences/
data/processed/wind_ml_10min/sequences_dwd_helgoland/
data/raw/wind_10min/DTU_wind_farm/
data/raw/wind_10min/NDBC_continuous_winds/
archive/
无用/
```

原因：

- `sequences/` 和 `sequences_dwd_helgoland/` 是早期派生序列集，不是当前主动压载预判数据集；
- DTU/NDBC 原始数据后续可用于泛化验证，但当前先专注 DWD Helgoland 单站点；
- `archive/` 和 `无用/` 是历史材料或低价值残留，不参与当前训练。

## 4080 环境建议

建议使用 Python 虚拟环境：

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

如果 CUDA 可用，训练时加：

```text
--device cuda
```

## 当前可复现命令

复跑当前最佳 GRU 思路：

```bash
python scripts/modeling/train_ballast_lstm.py \
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
  --device cuda \
  --output-dir outputs/wind_prediction/gru_h128_b2048_e30_dir002
```

复跑 LightGBM 强基线：

```bash
python scripts/modeling/train_lightgbm_wind_baseline.py \
  --n-estimators 300 \
  --learning-rate 0.05 \
  --num-leaves 63 \
  --min-child-samples 100 \
  --threads 8 \
  --output-dir outputs/wind_prediction/lightgbm_tabular_residual_e300
```

复跑分工况评估：

```bash
python scripts/modeling/evaluate_regime_and_thresholds.py \
  --split test \
  --device cuda \
  --batch-size 4096
```

## 下一步训练方向

下一步不要直接上大 Transformer。当前结果显示提升点主要在：

- 风速突增；
- 风向突变；
- 矢量突变；
- 压载关注事件的漏报/误报权衡。

优先做：

1. 新增短时波动特征；
2. 训练 CNN-GRU 或 TCN；
3. 做分工况对比；
4. 输出不同压载预警阈值工作点。

当前论文表达应强调“风变风险预判能力”，不要只围绕平均风速 MAE。
