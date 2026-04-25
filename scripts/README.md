# Scripts

脚本目录用于放一次性或命令行入口代码。

- `data_preparation/`
  - 读取原始数据
  - 清洗缺失值
  - 统一字段
  - 构造风速/风向变化特征
  - 生成面向主动压载提前判断的多步风况预测数据

- `modeling/`
  - 训练模型
  - 验证模型
  - 导出评估结果
  - 建立 persistence、转移矩阵等基线结果
  - `train_ballast_lstm.py`：训练面向主动压载提前判断的双输出 LSTM，输出未来风矢量序列和风变事件概率
  - `evaluate_wind_decision_baselines.py`：建立 persistence 基线，用于比较 LSTM 是否真正优于“未来等于当前风况”

可复用逻辑应下沉到 `src/wind_prediction/`，脚本只负责串流程。
