# 当前小论文框架记忆

Last updated: 2026-06-11

Purpose: this file records the currently locked paper structure for the FOWT
active-ballast short paper. Detailed revision policy is frozen in
`docs/paper_revision_master_plan_20260611.md`.

## Writing Positioning

The paper should be written as a compact engineering short paper. It is not a
wind-prediction model competition, a strict optimization paper, or a
deployment-scale controller certification. The paper object is a short-horizon
wind-prediction-assisted upper-layer coordinated regulation algorithm for active
ballast control.

Core positioning:

> This paper proposes a short-horizon wind-condition-prediction-assisted
> upper-layer coordinated regulation method for active ballast systems of
> floating wind turbines. The prediction module provides future wind trends,
> segmented risk windows, event-warning signals, and pressure-proxy signals; it
> does not directly output ballast pump commands. The original closed-loop
> feedback controller remains the execution layer, while pump limits, fallback,
> and safety protection remain forecast-source independent. The upper
> decision layer performs regime identification, target lifecycle management,
> action hold/release/veto/cap, and safety/economy trade-off decisions.

## Hard Constraints

- Use simplified Chinese body text.
- Use a compact engineering-journal style.
- Do not expand beyond the locked 0-4 body structure.
- Do not use third-level headings unless absolutely necessary.
- Start the body from Chapter 0.
- After the introduction, Chapter 1 should directly enter the author's own
  control method, not a broad background or literature chapter.
- Avoid repeatedly using internal implementation names such as `closed-only`;
  write "原有闭环反馈控制器" or "无预测闭环反馈策略" in the paper body.
- Do not write "预测驱动水泵" or imply that the prediction model directly outputs
  pump commands.
- Do not invent a strict online optimizer, MPC, QP, or explicit constrained
  planning formulation if the code does not implement it.
- Keep production-near evidence separate from historical/specialist evidence.
- Treat 6 h as short-horizon episode-level evidence.
- Treat 12 h gain 0.45 as robustness disclosure only.
- Treat 24 h deployment-scale validation as future work unless a stable 24 h
  campaign is actually run.

## Latest 6.11 Planning Alignment

The newest planning note is accepted as a writing-execution aid, with these
guardrails:

- Keep the paper as a short-horizon prediction-assisted upper-layer coordinated
  regulation paper, not a wind-prediction contest or a strict optimal-control
  paper.
- Merge the former Chapter 1 and Chapter 2 into the algorithm chapter. The former
  prediction/risk chapter is no longer a standalone body chapter.
- Put prediction input/output, segmented risk windows, event flags, and pressure
  proxy definitions in Chapter 1; put prediction/risk evidence and representative
  window interpretation in Chapter 3.
- Use Chapter 2 only for validation setup: platform, replay protocol, comparison
  strategy, metrics, and evidence roles.
- Write the method as an engineering interface: input, output, gating decision,
  target update, and execution constraint.
- Use related control papers only as expression templates for module boundaries,
  state variables, and claim limits. Do not import their control laws or proof
  obligations into this paper.
- Remove citation placeholders and unverified external-reference claims before
  drafting.
- Avoid any wording that reintroduces the advisor-banned term through English
  roots or translations.

## Locked Chapter Framework

```text
摘要
关键词

0 引言

1 短时风况预测辅助的主动压载上层协调调节算法

2 仿真验证方案

3 仿真结果与分析

4 结论
```

## Section Responsibilities

### 摘要

Follow the group-paper abstract pattern:

1. Application scenario and need.
2. "针对..." problem statement.
3. Proposed method.
4. Method components.
5. Frozen validation evidence and evidence role.
6. "结果表明..." with only the current headline result.
7. Claim-boundary sentence.

The abstract must use the frozen production-near headline unless the evidence
map is explicitly updated. Do not use historical, selector-only, or
positive-only numbers as the headline.

### 0 引言

Keep short and task-driven. The introduction should cover:

- Floating wind turbines are affected by wind, wave, current, and mooring loads,
  producing attitude response.
- Active ballast can support attitude regulation and reduce unnecessary ballast
  adjustment.
- Existing active ballast regulation mainly relies on current-state feedback or
  low-frequency load compensation and lacks future wind-load preview.
- Short-term wind prediction can provide wind trend, segmented risk-window,
  event-warning, and pressure-proxy information.
- The paper studies how prediction information enters an upper-layer decision
  interface, not how prediction directly controls pumps.
- Explain that this upper-layer decision means mode selection and coordinated
  target management, not direct pump-command generation.

No standalone literature-review chapter.

### 1 短时风况预测辅助的主动压载上层协调调节算法

This chapter presents the paper's own algorithm and absorbs the former standalone
prediction/risk chapter where those contents define algorithm inputs.

Recommended second-level sections:

```text
1.1 算法总体链路与上层决策接口
1.2 短时风况预测、风险识别与压力代理
1.3 决策状态、风险门控与有限候选动作评价
1.4 目标生命周期管理与水泵执行约束
```

Present the complete information flow:

```text
历史风况窗口
-> 短时风况预测
-> 分段风险窗口 / 风况事件 / 压力代理
-> 上层决策接口
-> 目标保持/释放/拒绝/限幅/回退
-> 原有闭环反馈控制器
-> 水泵执行约束
-> 平台响应反馈
```

Emphasize:

- Prediction is the upper-layer decision information source, not the direct actuator
  command generator.
- The prediction model, segmented risk windows, event flags, and pressure proxy
  are algorithm inputs and interface variables, so they belong in Chapter 1.
- The upper layer has finite decision states such as hold, release, veto,
  cap, safe, and fallback.
- The method should be written as risk gating, target lifecycle management, and
  execution-limited control, not as a fictitious strict optimizer.
- The original feedback controller, safety floor, fallback, pump limits, and
  recovery logic remain unchanged.

Critical algorithm-writing tasks:

- Close the data flow from `X_t` to `m_t^*` and constrained pump command.
- Define prediction output, three risk windows, pressure proxy, posture feedback,
  lifecycle state, and fused state before using them.
- Give the posture/forecast fusion formula:
  `z_k=q_k+\lambda^{k-1}r_t`, with `r_t` derived from pitch/roll deadband scaling.
- Present strategy discrimination as gating plus finite candidate-action scoring,
  not as loose prose or a fictitious continuous optimizer.
- Explain parameter bases by category: physical/execution constraints, time-scale
  choices, posture/safety scales, and frozen-profile scoring/action parameters.
- Keep the result chapter as validation and explanation; do not let it carry
  definitions that should appear in the algorithm chapter.

### 2 仿真验证方案

This chapter only defines validation setup and comparison protocol. It should not
repeat the algorithm details already placed in Chapter 1.

Recommended second-level sections:

```text
2.1 仿真平台、数据窗口与回放协议
2.2 对比策略与执行一致性
2.3 评价指标与阈值语义
2.4 casebook 冻结与证据角色
```

Include:

- Same platform model, execution layer, safety layer, data window, and replay
  protocol for baseline and candidate.
- Raw 1 Hz metrics; smoothing is display-only.
- Pump work, pump starts/stops, pitch/roll p95, threshold exposure time, fallback,
  and target/latch behavior.
- Threshold semantics: 5 deg as service-pressure band; 7.5 deg and 10 deg as
  larger/severe tail exposure.
- Evidence roles: 6 h headline, 12 h disclosure, historical/specialist support,
  and 24 h future work.

### 3 仿真结果与分析

This chapter reports paired results and uses prediction/risk evidence only to
explain action differences and regime-dependent result changes.

Recommended second-level sections:

```text
3.1 6 h production-near paired 主结果
3.2 预测与风险识别对动作差异的解释
3.3 预测可行动、边界与低机会工况分析
3.4 12 h mixed-regime 鲁棒性分析
```

Include:

- Use the frozen production-near 6 h guard10 paired casebook as the current
  headline evidence.
- Use representative prediction/risk windows to explain hold/release/veto/cap
  decisions, not to replace paired aggregate evidence.
- Use 12 h gain 0.45 mixed-regime results only as robustness disclosure.
- State that 24 h deployment validation is outside the current paper scope.
- Use figure/table anchored result writing.
- Do not overclaim all-condition improvement.

Preferred wording:

> 在冻结的 6 h episode-level paired casebook 中，预测辅助上层协调调节策略
> 能够减少部分不必要泵送，并保持严重姿态暴露指标有界。

If reporting the gain 0.45 12 h robustness result, describe it as:

> 混合工况 12 h 鲁棒性检验中，候选参数相对 current-only 总泵耗降低
> 9.51%，严重姿态暴露未增加，但服务压力阈值附近存在轻微暴露代价。

Do not describe it as a universal 30% long-duration saving.

### 4 结论

Conclusion must remain tied to the frozen evidence map.

Future conclusion opening:

```text
本文针对浮式风机主动压载系统缺少未来风载预判与危险工况提前预警的问题，
提出了短时风况预测辅助的主动压载上层协调调节算法，并通过冻结工况窗口
下的成对仿真对其节泵效果和姿态边界进行了验证，主要结论如下：
（1）……
（2）……
（3）……
```

Conclusion claims must distinguish:

- 6 h main evidence: production-near short-horizon episode-level effectiveness.
- Positive-only or historical/specialist evidence: mechanism/support/appendix
  only.
- 12 h robustness evidence: mixed-regime trade-off and claim boundary.
- 24 h: future production/deployment-scale validation unless actually run.

## Approximate Word Allocation

```text
摘要：300-400 字
0 引言：650-800 字
1 上层协调调节算法：1700-2100 字
2 仿真验证方案：700-900 字
3 仿真结果与分析：1500-1800 字
4 结论：300-450 字
```

Expected total: about 4700-5600 Chinese characters excluding references,
figures, and tables.
