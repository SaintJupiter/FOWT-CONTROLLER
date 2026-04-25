# Data Directory

数据目录按机器学习项目常见结构拆分：

- `raw/`
  - 原始下载数据，不直接修改。
  - 当前核心是 `raw/wind_10min/`。

- `processed/`
  - 清洗、合并、特征构造后的训练表。
  - 后续建议输出统一 CSV/Parquet，例如：
    - `wind_observations_10min.parquet`
    - `wind_transition_features.parquet`

处理规则：原始数据保持只读，所有派生结果写入 `processed/`。
