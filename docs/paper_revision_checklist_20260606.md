# 论文修订与审稿风险清单

Last updated: 2026-06-06

本清单用于修订 `docs/paper_draft_20260606.md`。核心原则：6 h 是主证据，12 h 是鲁棒性披露，historical/specialist 分支只做机制、边界或附录材料。

## P0 必须通过

- [ ] 正文 headline 只使用 `rawenv_holdpause_barrier_reliefcap_adaptive_v1` 下的 6 h paired evidence。
- [ ] 6 h 主结果写成 episode-level supervisory efficacy，不写成 deployment-wide 或 all-regime saving。
- [ ] 12 h gain 0.45 只写成 robustness disclosure，不写成 production default。
- [ ] D1、P2、C3、W1、PSC 等 historical/specialist 结果均标注 paper role。
- [ ] 所有节泵百分比同时报告姿态代价，至少包含 `t>5`、`t>7.5`、`t>10`。
- [ ] Fig. 2 只是代表时域机制图，不能替代 Table 2 的 aggregate paired evidence。
- [ ] 文中明确预测不是直接输出水泵命令，原有闭环反馈控制器仍是执行层。
- [ ] 文中明确所有指标来自 raw 1 Hz 数据，平滑仅用于图像展示。

## 6 h 质疑回应

审稿问题：为什么只用 6 h 做主结果？

建议回答：

> 本文目标是验证短时风况预测对主动压载监督层的 episode-level 作用。6 h 窗口包含多个 10 min 风况预测和监督决策周期，适合观察未来风险窗口对泵送策略的影响。为避免只报告短窗口收益，本文同时披露 12 h mixed-regime robustness；24 h 部署级验证作为后续工作。

检查项：

- [ ] 摘要中不能写“长期运行验证表明”。
- [ ] 结论中必须区分 6 h、12 h 和 24 h。
- [ ] 3.1 中必须说明 6 h casebook 是预先冻结的 guard10 broad casebook。

## Cherry-picking 质疑回应

审稿问题：是否只选了好看的 case？

必须保留：

- [ ] `production_near_6h_selection_freeze.md`
- [ ] `production_near_6h_per_case_delta.csv`
- [ ] `production_near_6h_regime_summary.csv`
- [ ] weak、zero、negative cases
- [ ] boundary/abstain materials

正文写法：

> 主结果采用冻结 casebook 下的 paired comparison，而非单条代表曲线。Fig. 2 仅用于解释机制；Table 2 和 Fig. 3/4 给出所有主样本的 aggregate 和分工况结果。

## 姿态代价质疑回应

必须检查：

- [ ] 是否报告 pitch p95 和 roll p95。
- [ ] 是否报告 `t>5`、`t>7.5`、`t>10`。
- [ ] 是否把 5 deg 写成 service-pressure band，而不是 hard failure。
- [ ] 是否说明 7.5 deg 和 10 deg 是 severe/operating-limit tail。
- [ ] 是否隐藏任何 `t>7.5` 或 `t>10` 增加。

当前冻结口径：

- 6 h 主结果：整体节泵 6.01%。
- 6 h 主结果：`d t>5=-64s`，`d t>7.5=-44s`，`d t>10=0s`。
- 12 h gain 0.45：节泵 9.51%，`t>7.5` 降低，`t>10` 不变，`t>5` 略增。

## Profile 混用质疑回应

禁止写法：

- [ ] “本控制器普遍节泵 30%。”
- [ ] “12 h 或 24 h production 结果达到 D1/P2/C1 的收益。”
- [ ] “PSC mixed pool 证明 production-near controller 有同样收益。”

允许写法：

> D1、P2、C3、W1 和 PSC 结果用于解释机制和边界，不并入 production-near headline。正文 headline 来自同一 production-near profile 下的 6 h 成对主证据和 12 h robustness disclosure。

## Baseline 公平性质疑回应

必须说明：

- [ ] baseline 是无预测闭环反馈策略。
- [ ] baseline 和 prediction-supervised 使用相同执行层、安全层、数据集和 replay window。
- [ ] prediction-supervised 的差异是未来风况信息进入监督层，而不是关闭安全约束或替换底层控制器。
- [ ] forecast-independent barrier、fallback 和 pump limits 对策略比较保持一致。

## 图表检查

- [ ] Fig. 1 能说明预测监督层、反馈层、安全层和执行层关系。
- [ ] Fig. 2 有风况、泵速、累计泵耗、pitch/roll，并说明未平滑计算。
- [ ] Fig. 3 按 regime 展示 pump saving，不只给 ALL。
- [ ] Fig. 4 展示姿态服务代价和严重尾部。
- [ ] Fig. 5 主动披露边界/拒绝动作/长窗口稀释。
- [ ] Fig. 6 展示 12 h pump work 和阈值暴露。
- [ ] 附录每类工况至少有 wind、pitch、roll、pump lb、ballast target lb 五类图。

## P1 待补材料

- [ ] 风况预测模型摘要表：数据来源、输入窗口、horizon、输出变量、训练/验证/测试划分。
- [ ] observed-vs-predicted 代表图。
- [ ] 风速 MAE/RMSE、风向 angular error、事件识别 precision/recall/F1。
- [ ] P1 transient/future-decay 附录曲线。
- [ ] W1 stable/signflip 附录曲线。
- [ ] 文献引用矩阵和最终参考文献格式。

## 交付索引

- 证据包：`outputs/wind_prediction/paper_20260606_execution/`
- 草稿：`docs/paper_draft_20260606.md`
- 执行计划：`docs/paper_execution_plan_20260606.md`
- 主图：`outputs/wind_prediction/paper_20260606_execution/main_figures/`
- 主表：`outputs/wind_prediction/paper_20260606_execution/tables/`
- 审稿回应草稿：`outputs/wind_prediction/paper_20260606_execution/manuscript_notes/reviewer_risk_response.md`
