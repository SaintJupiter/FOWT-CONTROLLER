# Docs

当前项目文档目录。原则：

- 当前仍作为论文、验证、控制链或交接依据的文档直接放在本目录。
- 旧程序架构、旧阶段台账、旧周报、旧草稿和旧章节改稿放在 `docs/archived/`，仅用于追溯历史判断。
- 当前风预测主线的交接入口仍在仓库根目录 `CODEX_HANDOFF.md`。

## Current Paper Track

论文框架、写作口径和文献策略：

```text
docs/ocean_engineering_submission_requirements.md
docs/123.md
docs/current_paper_framework.md
docs/current_literature_strategy.md
docs/paper_style_profile_xu_group.md
docs/paper_formula_reference.md
```

当前投稿目标为《海洋工程》中文版。涉及正文、图表、公式、参考文献或
Word排版时，必须先读`docs/ocean_engineering_submission_requirements.md`；
原《舰船科学技术》模板规则已废止。

论文验证边界和 claim 约束：

```text
docs/validation_protocol_v2.md
docs/frozen_h_full_validation_audit_20260710.md
```

`validation_protocol_v2.md` 是当前控制器开发与新实验的唯一规模约束：每个
工况固定为6小时，单次实验最多30组。1组和3组只用于排查程序问题，算法效果按10、20和30组逐级验证。
`frozen_h_full_validation_audit_20260710.md` 仅保存已投稿版本的150组历史证据，
其中19.85%的累计泵量降幅不作为V2的回归目标或验收门槛。旧验证方案仅用于
追溯，不再指导新实验。

## Current Control And Validation Track

控制链整改、指标语义、baseline 定位和验证记录：

```text
docs/controller_v2_framework.md
docs/controller_architecture_audit_20260811.md
docs/control_chain_rectification_plan_20260605.md
docs/attitude_metric_semantics.md
docs/closed_baseline_v1_positioning.md
docs/prediction_primary_10case_validation_notes.md
docs/target_lifecycle_design_note.md
```

`controller_v2_framework.md` 是当前控制程序的结构入口。新开发只经过
`wind_prediction.controller`，旧 Provider 和 casebook 代码仅用于复现历史结果，
不再承接新的控制分支。现阶段验收以配置可追溯、三周期冒烟运行和结构测试通过为准，
不以泵量降幅或当前模拟姿态作为调参依据。

## Wind Prediction Planning

当前风预测技术路线和训练记录主要放在：

```text
风预测规划/面向主动压载提前调节的风预测技术路线.md
风预测规划/风预测实现参考要点.md
风预测规划/LSTM训练执行记录.md
风预测规划/任务日志.md
```

## Archive

历史材料入口：

```text
docs/archived/README.md
```

除非需要追溯旧实验、旧论文口径或旧程序结构，否则不要把 archived 中的文件作为当前主线依据。
