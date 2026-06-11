# FOWT Wind Prediction Workspace

本项目当前主线已调整为：基于真实高时间分辨率风速、风向数据，构建短时风场变化预测能力，并为后续压载系统提前预判提供输入。

旧的主动压载控制仿真、论文验证、参考模型和结果材料没有简单判定为“无用”，而是按生命周期重新分类：当前主线放在 `data/`、`src/`、`scripts/`；历史工程放在 `archive/`；论文和参考资料放在 `docs/`、`references/`；真正低价值的临时调试输出才放在 `无用/`。

## 先读这里

给另一台电脑或新的 Codex 会话接手时，先读：

```text
CODEX_HANDOFF.md
```

该文件记录了当前目标、已经完成的风预测数据集、最新 segmented 风预测结果、控制接入状态、下一步任务和注意事项。

当前最新状态：

- 已生成统一 10 分钟风预测 ML 表。
- 已生成 DWD Helgoland 单站点 `segmented_v1` 主动压载预判数据集。
- 已生成 FINO1 单站点海上高空观测底座 `fino1_platform_10min`。
- 已生成 FINO1 模型兼容数据集 `ballast_decision_fino1_segmented_v1`。
- 已建立 persistence 基线。
- 已完成 LSTM / segmented GRU / segmented LightGBM / CNN-GRU / TCN 对比。
- 当前主结果目录为 `outputs/wind_prediction/gru_segmented_head_v1`。
- 已完成控制链中的 `preview_trim_provider` / `preview_trim_bias` 预览接入点。
- 当前下一步更偏向 FINO1 复训或风预测接入压载控制验证，而不是继续小幅堆模型。

## 目录结构

```text
.
├── data/
│   ├── raw/wind_10min/        # 已下载的真实10分钟级风速/风向原始数据
│   └── processed/             # 后续清洗后的训练表、特征表
├── src/wind_prediction/       # 后续正式预测代码包
├── scripts/
│   ├── data_preparation/      # 数据读取、清洗、合并、特征构造脚本
│   └── modeling/              # 训练、验证、评估脚本
├── notebooks/                 # 探索性分析笔记
├── models/                    # 训练后的模型文件
├── outputs/                   # 图表、评估结果、临时实验输出
├── docs/                      # 当前项目说明和保留下来的历史技术文档
├── references/                # 论文、公开参考模型、外部资料
├── archive/                   # 有保留价值但非当前主线的旧工程和旧论文产物
└── 无用/                      # 真正低价值的旧调试输出、临时残留
```

## 当前最重要的数据

原始风数据在：

```text
data/raw/wind_10min/
```

其中优先级最高的是：

- `DTU_wind_farm/norre_m2_all.nc`
  - 42台风机 + 2个测风桅杆，10分钟统计。
  - 含风速、风向、nacelle wind speed、yaw position 等信息。

- `DTU_wind_farm/hornsrev_all.nc`
  - 真实 offshore met mast，10分钟风速和风向统计。

- `NDBC_continuous_winds/`
  - NOAA/NDBC 海上浮标 continuous winds。

- `DWD_10min_wind/`
  - DWD Helgoland 长时间10分钟风速/风向观测。

更详细的数据说明见：

```text
data/raw/wind_10min/README.md
```

当前主要训练数据集在：

```text
data/processed/wind_ml_10min/ballast_decision_dwd_helgoland/
```

以及新的 FINO1 底座与训练集：

```text
data/processed/wind_ml_10min/fino1_platform_10min/
data/processed/wind_ml_10min/ballast_decision_fino1_segmented_v1/
```

这个数据集使用 DWD Helgoland 10 分钟风速/风向，输入过去 120 分钟，输出未来 60 分钟风矢量序列和风变事件标签。当前样本数为：

- train：986,913
- validation：216,153
- test：215,929

注意：当前交接范围会通过 Git LFS 上传主动训练所需的大文件。另一台电脑 clone 后如果看到 `.npy`、`.nc`、`.zip`、`.gz` 或 `.pt` 文件很小，先执行：

```bash
git lfs pull
```

本次上传不包含重复的未压缩 `supervised_learning_table_10min.csv`，也不包含旧的基础 sequence 派生集和 PDF 文献。

## 当前模型结果

当前主结果是：

```text
outputs/wind_prediction/lstm_ballast_decision_residual_w002/
```

该模型是双输出 residual LSTM：

- 回归输出：未来 6 步 `u/v` 风矢量；
- 分类输出：5 类风变事件概率；
- 输入：12 个 10 分钟历史步，每步 31 个特征；
- 最优 epoch：8；
- 与 persistence 相比，普通风速和风矢量误差小幅改善，综合压载关注事件识别明显改善。

关键测试集对比：

| 指标 | persistence | residual LSTM |
| --- | ---: | ---: |
| 全未来步风速 MAE | 0.6037 m/s | 0.5862 m/s |
| 全未来步风矢量 MAE | 0.6674 m/s | 0.6496 m/s |
| 全未来步风向 MAE | 7.7437 deg | 8.0555 deg |
| 压载关注事件 recall | 0.0578 | 0.5277 |
| 压载关注事件 F1 | 0.1090 | 0.5099 |

当前结论：它更适合写成“风变风险预判模块”的基础结果，而不是直接宣称普通风向预测精度已经显著优于 persistence。

## 常用命令

生成统一 10 分钟 ML 表：

```bash
python scripts/data_preparation/prepare_wind_ml_dataset.py
```

生成主动压载预判数据集：

```bash
python scripts/data_preparation/prepare_ballast_wind_decision_dataset.py
```

计算 persistence 基线：

```bash
python scripts/modeling/evaluate_wind_decision_baselines.py
```

训练当前 residual LSTM 设置：

```bash
python scripts/modeling/train_ballast_lstm.py \
  --output-dir outputs/wind_prediction/lstm_ballast_decision_residual_w002 \
  --epochs 8 \
  --event-loss-weight 0.02 \
  --residual-regression
```

训练 LightGBM 表格强基线：

```bash
python scripts/modeling/train_lightgbm_wind_baseline.py \
  --n-estimators 300 \
  --learning-rate 0.05 \
  --num-leaves 63 \
  --min-child-samples 100 \
  --threads 8 \
  --output-dir outputs/wind_prediction/lightgbm_tabular_residual_e300
```

在 RTX 4080 电脑上训练时，如果 PyTorch CUDA 可用，可以加：

```text
--device cuda
```

## 当前开发建议

1. 先保留 persistence 作为所有模型的最低基线。
2. 补 LightGBM/CatBoost 表格强基线，判断短时预测中树模型是否强于小型 LSTM。
3. 再补 GRU、TCN、CNN-LSTM 时序基线。
4. 按普通风速误差、风矢量误差、风向误差和事件识别指标分别报告。
5. 外部泛化必须用 DTU、NOAA/NDBC 或其他 DWD 站点做 held-out 测试，不能只凭 Helgoland 单站点下结论。

正式可复用代码放入 `src/wind_prediction/`，命令行流程放入 `scripts/`。
