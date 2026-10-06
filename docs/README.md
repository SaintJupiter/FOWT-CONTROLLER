# Docs

入口更新：2026-10-06。当前研究主线是预测驱动主动压载的Preview链路；旧V2和小论文材料保留作历史依据，不作为新开发的唯一入口。

当前项目文档目录。原则：

- 当前仍作为论文、验证、控制链或交接依据的文档直接放在本目录。
- 旧程序架构、旧阶段台账、旧周报、旧草稿和旧章节改稿放在 `docs/archived/`，仅用于追溯历史判断。
- 当前研究交接入口在仓库根目录[CODEX_HANDOFF.md](../CODEX_HANDOFF.md)，阅读顺序以其顶部当前入口为准。

## Current Thesis And Preview Track

当前课题、实施顺序与已有进度：

- [10.6课题规划](10.6课题规划.md)：研究定位、方法分组与贡献边界。
- [有限情景主动压载执行规划](10.6_有限情景主动压载_执行规划.md)：接口、阶段状态与后续实施顺序。
- [大论文执行记录](thesis_execution_progress_20261005.md)：已有运行结果及其限制。
- [预测到决策机制诊断](thesis_prediction_decision_mechanism_diagnosis_20261005.md)：预测、目标、实际执行及后果之间的关系。
- [近年预览控制全文评估](recent_preview_control_fulltext_review_20261006.md)及[文献索引](../references/multistage_control/README.md)：方法来源和迁移条件。

已经实现的基础为确定性预览优化、同口径物理载荷转换、实际泵送及滚动反馈。有限情景联合规划与低频区域分层目标仍属设计/实施中的研究版本，不据此宣布新性能结论。

当前代码入口为`src/wind_prediction/preview_mpc.py`、`preview_mpc_application.py`、`preview_mpc_control_cycle.py`及`preview_mpc_runtime.py`。共同实验入口为`scripts/validation/preview_mpc_continuous_experiment.py`。旧`wind_prediction.controller`不是当前Preview开发的唯一入口。

实验范围按10.6规划及具体任务确定：正式连续评价沿用固定6小时工况、单批最多30组的边界，但不自动按10、20、30组逐级运行。实现检查只覆盖本阶段确实需要的性质，不重复开展同类审查或为文档更新新增控制实验。历史150组与19.85%节泵结果不作为新版本的验收目标。

## Historical Small-Paper Track

小论文框架、写作口径和文献策略：

```text
docs/ocean_engineering_submission_requirements.md
docs/current_paper_framework.md
docs/current_literature_strategy.md
docs/paper_style_profile_xu_group.md
docs/paper_formula_reference.md
```

小论文投稿目标为《海洋工程》中文版。涉及该稿正文、图表、公式、参考文献或
Word排版时，先读本地`docs/ocean_engineering_submission_requirements.md`；
原《舰船科学技术》模板规则已废止。该投稿规范及部分训练、归档材料未包含在公开仓库中，不能将这些本地路径当作克隆后必然可用的入口。

论文验证边界和 claim 约束：

```text
docs/validation_protocol_v2.md
docs/frozen_h_full_validation_audit_20260710.md
```

`validation_protocol_v2.md`保存早期验证协议；涉及当前Preview实验规模与顺序时，以顶部10.6规划为准，不自动执行其旧的逐级验证流程。
`frozen_h_full_validation_audit_20260710.md` 仅保存已投稿版本的150组历史证据，
其中19.85%的累计泵量降幅不作为V2的回归目标或验收门槛。旧验证方案仅用于
追溯，不再指导新实验。

## Historical V2 Control And Validation Track

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

`controller_v2_framework.md`描述早期V2及其`wind_prediction.controller`公共接口，不再作为当前Preview主线的唯一结构入口。旧Provider和casebook仅用于历史复现或明确对照，不承接新的Preview控制分支。这里的三周期冒烟与结构验收属于当时阶段，不能代替当前研究版本的状态说明或性能证据。

## Local Wind Prediction Records

本地风预测技术路线和训练记录主要放在以下位置；它们不是本轮控制重构的自动训练任务，也未随公开仓库完整发布：

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
