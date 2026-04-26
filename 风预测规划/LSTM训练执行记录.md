# LSTM训练执行记录

交接入口：`../CODEX_HANDOFF.md`

## 执行目标

本轮训练的目标是验证当前单站点 10 分钟风速/风向数据是否能够支撑一个面向主动压载提前判断的基础 LSTM 模型。模型不直接输出压载指令，而是输出未来 60 分钟的风矢量序列和风变事件概率。

## 文献与流程依据

- Fuentes-Barrios et al. 2022 使用 10 分钟风速观测，构造 12 个输入步预测 12 个未来步的 LSTM nowcasting 任务，支持当前“10 分钟数据、多步输出”的设置。来源：https://www.mdpi.com/2673-4931/19/1/30
- Sari et al. 2021 将风速与风向转成二维风矢量形式，用于风速/风向联合预测，支持当前避免直接回归 0-360 度角度的处理。来源：https://www.tandfonline.com/doi/abs/10.1080/18824889.2021.1894878
- Mode et al. 2025 采用回归和分类双输出结构处理极端风预测，并强调事件阈值与样本不平衡问题，支持当前“风矢量回归 + 风变事件分类”的双头模型。来源：https://www.sciencedirect.com/science/article/pii/S0167610525000315
- Blazakis et al. 2025 对风速和风向做多时域预测，并使用 sine/cosine 处理风向，说明不同预测时域下模型表现会有差异，支持当前按 t+10 到 t+60 分别评价。来源：https://www.mdpi.com/2079-9292/14/24/4856
- Pang & Dong 2026 指出风速预测中全样本预处理会造成未来数据泄漏，支持当前只用训练集归一化、按时间顺序划分、窗口不跨 split 的处理。来源：https://www.sciencedirect.com/science/article/abs/pii/S0307904X25004500

## 当前数据

- 数据目录：`data/processed/wind_ml_10min/ballast_decision_dwd_helgoland`
- 数据来源：单一站点 `DWD::02115_Helgoland::10min_wind`
- 输入：过去 120 分钟，即 12 个 10 分钟时间步
- 输出：未来 60 分钟，即 6 个 10 分钟时间步
- 输入特征：31 个，包括风速、风向 sin/cos、风矢量 u/v、10 分钟变化量、30/60/120 分钟滚动统计和周期时间特征
- 训练集：986,913 条
- 验证集：216,153 条
- 测试集：215,929 条

## 已跑模型

训练脚本：

`scripts/modeling/train_ballast_lstm.py`

已跑三组：

1. 直接预测未来绝对 `u/v`，事件损失权重 0.10。
2. 直接预测未来绝对 `u/v`，事件损失权重 0.02。
3. residual 回归版本，预测未来相对当前 `u/v` 的变化量，事件损失权重 0.02。

第三组效果最好，作为当前主结果。

主结果目录：

`outputs/wind_prediction/lstm_ballast_decision_residual_w002`

核心设置：

- LSTM hidden size：64
- LSTM layer：1
- batch size：4096
- epoch：8
- optimizer：AdamW
- 回归损失：MSE
- 分类损失：BCEWithLogitsLoss，使用训练集事件比例计算 `pos_weight`
- 事件阈值：在验证集上按 F1 选择，再固定到测试集
- 最优 epoch：8

## 测试集结果

与 persistence 基线相比：

| 指标 | persistence | residual LSTM | 变化 |
| --- | ---: | ---: | ---: |
| 全未来步风速 MAE | 0.6037 m/s | 0.5862 m/s | -0.0175 m/s |
| 全未来步风矢量 MAE | 0.6674 m/s | 0.6496 m/s | -0.0177 m/s |
| 全未来步风向 MAE | 7.7437 deg | 8.0555 deg | +0.3118 deg |
| t+60 风速 MAE | 0.7587 m/s | 0.7375 m/s | -0.0212 m/s |
| t+60 风矢量 MAE | 0.8570 m/s | 0.8275 m/s | -0.0295 m/s |
| t+60 风向 MAE | 10.5484 deg | 10.7330 deg | +0.1845 deg |
| 压载关注事件 recall | 0.0578 | 0.5277 | +0.4700 |
| 压载关注事件 F1 | 0.1090 | 0.5099 | +0.4009 |
| 压载关注事件 false alarm rate | 0.00024 | 0.05718 | +0.05694 |

## 判断

当前 LSTM 已经能在风速和风矢量误差上略优于 persistence，同时明显提高未来风变事件的识别能力。方向角 MAE 略差，说明当前模型对角度细节还不够稳，后续应继续强化风向误差和风向突变事件。

从主动压载提前判断角度看，当前结果比单纯 persistence 更有价值，因为 persistence 基本无法识别未来风速突增、风向突变和矢量突变，而 residual LSTM 对综合压载关注事件的召回率已经从 5.78% 提高到 52.77%。代价是误报率上升到约 5.72%，这一点需要在压载决策层继续设置保守阈值或确认机制。

## 输出文件

- 模型权重：`outputs/wind_prediction/lstm_ballast_decision_residual_w002/lstm_best.pt`
- 训练过程：`outputs/wind_prediction/lstm_ballast_decision_residual_w002/lstm_training_history.csv`
- 回归指标：`outputs/wind_prediction/lstm_ballast_decision_residual_w002/lstm_regression_metrics.csv`
- 事件指标：`outputs/wind_prediction/lstm_ballast_decision_residual_w002/lstm_event_metrics.csv`
- 事件阈值：`outputs/wind_prediction/lstm_ballast_decision_residual_w002/lstm_event_thresholds.json`
- 与基线对比：`outputs/wind_prediction/lstm_ballast_decision_residual_w002/lstm_vs_persistence_key_metrics.csv`

## 下一步

1. 先把 residual LSTM 作为当前基线模型。
2. 对风向误差增加约束，例如对预测 `u/v` 的方向角误差加入辅助损失，或单独增加方向变化分类头。
3. 加入 LightGBM/CatBoost 表格强基线，检查短时预测中树模型是否更强。
4. 再尝试 CNN-LSTM 或 TCN，用局部卷积捕捉 10-60 分钟内的短时波动。
5. 最后再上 Transformer/TFT，避免过早引入复杂模型而缺少可解释的基线。
