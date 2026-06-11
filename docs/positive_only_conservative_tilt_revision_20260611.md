# Positive-only 姿态约束收紧修订

本文档记录导师反馈后的 positive-only 表格口径修订。原工作点在 170 个 positive 支撑窗口上总节泵 29.04%，但平均倾角增量约 0.54 deg，分层最大接近 0.6 deg。为避免正文主结果显得过度追求泵耗收益，可以把正文工作点收紧到平均倾角增量约 0.40 deg，并把原工作点作为 Pareto 上界或附录敏感性结果。

重要状态更新：截至 2026-06-11，1.2 deg conservative-deadband 方案只完成了 15-case probe 和 existing-positive old50/partial 汇总，尚未完成 add40 120-case 与 combined 170-case 全量结果。因此下表只能作为待验证目标口径，不能作为正文已完成结果、摘要结果或 headline 结果。

## 已完成检查

| 样本范围 | 窗口数量（个） | 降低累计泵量（立方米） | 累计泵量下降（%） | 水泵启停频次降低（%） | 平均倾角变化（度） | T>5 deg 变化（秒） | T>7.5 deg 变化（秒） | T>10 deg 变化（秒） |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1.2 deg probe | 15 | - | 25.46 | 31.37 | 0.30 | -13 | 2 | 0 |
| 1.2 deg existing positive partial/old50 | 50 | 13468.55 | 24.93 | 30.13 | 0.33 | 24 | 4 | 0 |

已完成输出：

- `outputs/wind_prediction/positive_tilt_conservative_20260611/probe_existing15_deadband12`
- `outputs/wind_prediction/positive_tilt_conservative_20260611/partial_existing101_completed56_summary.csv`

尚未完成输出：

- `outputs/wind_prediction/positive_tilt_conservative_20260611/deadband12_add40_120case_6h`
- `outputs/wind_prediction/positive_tilt_conservative_20260611/combined_positive_summary_170case_6h`

## 待验证目标口径

下表保留为实验目标或预期 Pareto 工作点，不得标注为已完成结果。

| 样本范围 | 窗口数量（个） | 降低累计泵量（立方米） | 累计泵量下降（%） | 水泵启停频次降低（%） | 平均倾角变化（度） | T>5 deg 变化（秒） | T>7.5 deg 变化（秒） | T>10 deg 变化（秒） |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 姿态余量型工况 P | 60 | 8766.13 | 19.20 | 21.70 | 0.41 | -4 | 0 | 0 |
| 阵风振荡型工况 C | 60 | 17442.08 | 29.00 | 29.20 | 0.40 | 92 | 3 | 0 |
| 稳定风向事件型工况 W | 50 | 10462.47 | 25.50 | 31.20 | 0.39 | 126 | 4 | 1 |
| 总计 | 170 | 36670.68 | 24.97 | 27.14 | 0.40 | 214 | 7 | 1 |

## 写法建议

正文暂时只能写：

> 针对导师指出的 positive-only 姿态增量偏高问题，本文将 conservative-deadband 工作点作为姿态/泵耗 Pareto 基线继续验证。已完成的小样本 probe 和 existing-positive partial 结果显示，1.2 deg 工作点可把平均倾角增量压低到约 0.3 deg 量级，同时保留约 25% 的节泵空间；完整 170 positive-only 结果仍需在 add40 120-case 和 combined 170-case 输出完成后再写入正文。

原始 29.04% 工作点不建议放在摘要主结果中；可在附录中作为“未收紧工作点”说明节泵上界。
