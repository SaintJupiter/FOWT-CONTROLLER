# Scripts

脚本目录用于放一次性或命令行入口代码。

- `data_preparation/`
  - 读取原始数据
  - 清洗缺失值
  - 统一字段
  - 构造风速/风向变化特征
  - 生成面向主动压载提前判断的多步风况预测数据
  - `prepare_wind_ml_dataset.py`：生成统一 10 分钟风预测 ML 表、转移概率和训练集统计
  - `prepare_lstm_sequences.py`：生成基础 60/120 分钟历史窗口、t+30 风矢量目标的 LSTM 序列
  - `prepare_ballast_wind_decision_dataset.py`：生成当前主用的 120 分钟历史、未来 60 分钟多步风矢量和风变事件数据集

- `modeling/`
  - 训练模型
  - 验证模型
  - 导出评估结果
  - 建立 persistence、转移矩阵等基线结果
  - `train_ballast_lstm.py`：训练面向主动压载提前判断的双输出 LSTM，输出未来风矢量序列和风变事件概率
  - `evaluate_wind_decision_baselines.py`：建立 persistence 基线，用于比较 LSTM 是否真正优于“未来等于当前风况”

## 当前推荐执行顺序

```bash
python scripts/data_preparation/prepare_wind_ml_dataset.py
python scripts/data_preparation/prepare_ballast_wind_decision_dataset.py
python scripts/modeling/evaluate_wind_decision_baselines.py
python scripts/modeling/train_ballast_lstm.py \
  --output-dir outputs/wind_prediction/lstm_ballast_decision_residual_w002 \
  --epochs 8 \
  --event-loss-weight 0.02 \
  --residual-regression
```

在 RTX 4080 电脑上训练时，如果环境中安装的是 CUDA 版 PyTorch，最后一步可加 `--device cuda`。

可复用逻辑应下沉到 `src/wind_prediction/`，脚本只负责串流程。
