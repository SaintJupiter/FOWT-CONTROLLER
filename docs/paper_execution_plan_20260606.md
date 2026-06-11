# 论文后续执行规划

Last updated: 2026-06-06

本文档用于把 `6.6gpt.md`、当前论文框架、验证策略和最新图表展示意见合并成后续可执行清单。它是论文推进主控文件，不改变控制器代码和实验数据。

## 0. 总体判断

当前论文应按“短时风况预测辅助主动压载监督调节”的工程短论文来推进，而不是按“长期部署级生产控制器认证”来推进。

论文主张应限定为：

> 短时风况预测可以为浮式风机主动压载提供未来趋势、风险窗口和危险工况预警信息，使监督层在预测可行动工况中减少不必要泵送，并在已测试样本中保持严重姿态暴露指标有界。

论文不应声称：

- 所有工况下均能长期稳定节泵。
- 30% 左右节泵是 12 h 或 24 h 长时段 production 结果。
- P2、C3、PSC、deadband 或历史分支的收益可以直接合并为当前 production profile 的统一 headline。

本论文的证据结构采用三层：

| 证据层 | 作用 | 主时长 | 论文位置 |
|---|---|---:|---|
| 主证据 | 证明短时预测监督在 forecast-actionable episodes 中有用 | 6 h | 正文主结果 |
| 边界证据 | 证明边界工况、拒绝动作、收益伴随代价等现象 | 6 h 或 12 h | 正文一图一表，附录展开 |
| 鲁棒性披露 | 防止被质疑只挑短窗口，并说明长时段混合收益会稀释 | 12 h | 正文或附录重点表 |

24 h 不作为当前短论文硬门槛。只有当论文改成 production/default controller 或 deployment-scale benefit 时，才把 24 h 作为必要证据。

## 1. 执行原则

后续所有实验、图表和写作必须遵守以下原则。

1. 所有主结果必须是 paired comparison，即同一 case/window 下比较无预测闭环反馈策略与预测监督策略。
2. 所有节泵百分比必须同时报告姿态代价，不能只报 pump saving。
3. 6 h 主结果必须来自预先定义的 casebook 或清楚说明来源的固定样本池，不能只挑好看的代表曲线。
4. 12 h guard10 结果必须作为 robustness/boundary disclosure 出现，不能隐藏。
5. 24 h 只写作 future work 或 production-freeze requirement，不作为当前稿件必跑任务。
6. 不同工况分开作图，不把 P2、C3、W1、PSC 等机制不同的时域曲线混在一张图里。
7. pitch 和 roll 应分开展示，至少在代表曲线和附录曲线中分开。
8. 水泵图必须展示泵的磅数相关信息，优先使用 `pump lb/min` 与 `cumulative pumped lb`。
9. 压载状态图必须展示 `ballast mass lb` 与 `target mass lb`。
10. 平滑曲线只用于图像展示，所有 pump work、fallback、latch、姿态暴露指标必须来自 raw 1 Hz 数据。

## 2. 论文路线冻结

### 2.1 默认路线

默认采用“短时预测监督调节工程短论文”路线。

正文使用已冻结章节框架：

```text
摘要
关键词

0 引言

1 短时风况预测驱动的主动压载监督调节算法
1.1 预测监督调节算法框架
1.2 主动压载监督调节流程

2 短时风况预测与风险识别方法
2.1 风况数据与特征构造
2.2 风况预测模型与风险信号输出

3 主动压载监督调节仿真与性能分析
3.1 仿真工况与评价指标
3.2 仿真结果分析

4 结论
```

### 2.2 两条证据线必须分开

后续结果整理时必须把下列两条线分开。

| 证据线 | 可用于什么 | 不能怎么用 |
|---|---|---|
| production-near evidence | 当前 production-near profile 的 paper-facing 结果、12 h guard robustness | 不能和 deadband/P2/C3 历史分支混成统一 production headline |
| regime-specialist evidence | 解释不同预测可行动工况中的机制收益，如 P2、P1、P4/C1、W1、C3、PSC | 不能写成“当前 production controller 普遍节泵 30%” |

如果后续主文需要引用 P2、D1、P1、P4/C1、C3、W1、PSC 等历史或分工况结果，必须在表格中标注：

- 工况名称；
- 样本数量；
- horizon；
- 使用的 control profile 或实验分支；
- 论文角色：primary / mechanism / boundary / robustness / appendix only；
- 是否可用于 headline claim。

### 2.3 当前可直接使用的 12 h 鲁棒性事实

12 h guard10 mixed-regime robustness 采用以下固定事实：

| variant | pump work | saving vs current-only | `t>5 deg` | `t>7.5 deg` | `t>10 deg` | 论文解释 |
|---|---:|---:|---:|---:|---:|---|
| current-only | 12800.43 m3 | 0.00% | 11.0310% | 0.0785% | 0.0023% | 无预测闭环反馈基线 |
| production, gain 0.40 | 12406.56 m3 | 3.08% | 11.3125% | 0.0785% | 0.0023% | 当前 production profile 参考 |
| production-near, gain 0.45 | 11582.94 m3 | 9.51% | 11.4741% | 0.0683% | 0.0023% | 候选参数，作为 robustness trade-off |

写作口径：

> 混合工况 12 h 鲁棒性检验中，production-near gain 0.45 相对 current-only 总泵耗降低 9.51%，严重姿态暴露未增加，但服务压力阈值附近存在轻微暴露代价。该结果用于界定长时段混合工况下的收益边界，不作为全工况长期通用节泵声明。

## 3. 分阶段执行总览

| 阶段 | 目标 | 主要输出 | 通过标准 |
|---|---|---|---|
| A | 冻结主张和证据归属 | claim-evidence map | 每个结果都能对应 profile、工况、样本数和论文角色 |
| B | 冻结 6 h 主 casebook | casebook selection record | 样本选择规则先于最终结果解释，避免 cherry-picking |
| C | 补齐 paired metrics | 结果 CSV/Markdown 汇总 | pump、姿态暴露、fallback、latch 全部齐全 |
| D | 生成主文图表 | 6 张主图、3 张主表 | 每张图承担独立信息功能，不重复凑图 |
| E | 生成附录图表 | 每类工况至少 5 类图 | 风况、pitch、roll、pump、ballast 分开 |
| F | 起草正文 | 4700-5600 字中文稿 | 章节完整，结果和图表相互引用 |
| G | 审稿风险自检 | reviewer-risk checklist | 能回答 6 h、样本选择、姿态代价、profile 混用问题 |
| H | 定稿打包 | manuscript package | 正文、图表、附录、参考文献、结果索引齐全 |

## 4. 阶段 A：冻结主张与证据归属

### A1. 建立结果归属表

输入：

- `6.6gpt.md`
- `docs/paper_validation_strategy_20260605.md`
- `docs/control_validation_protocol.md`
- 历史结果目录和 casebook report

动作：

- 汇总所有当前要写进论文或附录的结果。
- 为每个结果标注：工况、样本数、时长、profile、baseline、主要收益、主要代价。
- 判定每个结果的论文角色。

建议输出：

```text
outputs/wind_prediction/paper_20260606_execution/claim_evidence_map.csv
outputs/wind_prediction/paper_20260606_execution/claim_evidence_map.md
```

字段建议：

```text
result_id
regime
sample_count
horizon
baseline
candidate_profile
pump_saving_pct
fallback_delta
pitch_p95_delta
roll_p95_delta
time_over_5_delta
time_over_7p5_delta
time_over_10_delta
paper_role
headline_allowed
notes
source_path
```

通过标准：

- 每个百分比都能追溯到原始输出目录。
- `headline_allowed` 不能默认为 yes，必须手动判定。
- deadband、P2、C3、PSC 等历史或机制结果不得直接标为 production headline。

### A2. 冻结论文主句

必须形成一条中文主句和一条英文主句。

中文建议：

> 本文提出一种基于短时风况预测的浮式风机主动压载监督调节方法，将未来风况趋势、风险窗口和危险工况预警转化为监督层策略选择信息，在预测可行动工况中降低不必要泵送，并在已测试样本内保持严重姿态暴露指标有界。

英文建议：

> This paper proposes a short-term wind-condition-prediction-supervised active ballasting strategy for floating wind turbines. The method uses future wind trends, risk windows, and hazardous-condition warnings as supervisory information to reduce unnecessary ballast pumping in forecast-actionable episodes while keeping severe attitude-exposure metrics bounded in the tested cases.

通过标准：

- 不出现 universal、deployment-wide、all regimes、long-duration general saving 等过强表达。
- 不把预测写成直接输出泵命令。

## 5. 阶段 B：冻结 6 h 主 casebook

### B1. 选择主结果样本池

默认策略：

- 6 h 作为主结果 horizon。
- 样本池必须是 broad 或 stratified。
- 每个 case 同时运行 current-only 和 prediction-supervised candidate。
- 机制展示 case 与主统计 casebook 分开。

推荐样本角色：

| 样本角色 | 预期表现 | 论文用途 |
|---|---|---|
| positive opportunity | 节泵或减少 latch，且不增加 7.5/10 deg 尾部 | 主结果 |
| medium positive | 有节泵但存在轻微 service-band 代价 | 主结果或讨论 |
| boundary/lookalike | abstain 或接近 current-only | 边界证明 |
| ordinary background | 不误触发、不显著恶化 | 泛化检查 |
| known failure reproducer | 修复 fallback 或 hold debt，不新增尾部风险 | 机制/安全证明 |

通过标准：

- 样本选择规则写清楚。
- 不能只选“最好看的 5 条”。
- 如果使用已有样本池，必须说明该池的来源和是否在最终结果前已固定。

### B1a. 样本量与数据时长支撑

样本量技术支撑已固化在：

```text
docs/validation_sample_size_support_20260606.md
```

当前可用于论文的方法学口径：

| 层级 | 数量/时长 | 作用 |
|---|---:|---|
| FINO1 原始实测风况 | 978,280 个 10 min 记录，约 18.6 年有效数据 | 说明数据来源具有长期真实风况基础 |
| 预测训练集 | 638,631 个 H240/F120 序列样本 | 训练风况预测与风险识别模型 |
| 预测验证集 | 133,949 个序列样本 | 模型选择与阈值检查 |
| 预测测试集 | 140,173 个序列样本 | held-out 预测评估与闭环 case 来源 |
| 闭环 mixed casebook | 101 个 6 h 窗口 | 主混合工况验证，含 positive、warning/boundary、background |
| positive 扩展 casebook | 120 个 6 h 窗口 | 扩展 P2/C3/W1 正向节水工况证据 |
| 当前闭环验证总量 | 221 个 6 h 窗口，1326 h | 支撑短时段、分工况、episode-level 控制结论 |

写作注意：

- `101-case` 是主 mixed validation；`170 positive` 是由其中 50 个
  positive 加上 120 个扩展 positive 构成的节水工况支撑证据，不能与
  101 简单相加成 271。
- 221 个 6 h 窗口约占 test split 的 5.7%（按 10 min 等效时长计），
  对闭环控制验证已经较充分。
- 由于存在相邻窗口，正文应写“预声明 6 h 评价窗口”，不要写成“221 个
  完全独立天气事件”。
- 该样本量支撑当前短论文主张，但不能外推为 20 年部署级或认证级载荷
  /疲劳验证。

### B2. 当前结果的使用建议

当前已有结果按下列方式进入论文：

| 工况/结果 | 当前样本信息 | 建议角色 | 写作注意 |
|---|---:|---|---|
| P2 clean neutral/headroom | 80 样本，约 41.16% 节泵，fallback 0 | primary 或 mechanism strong evidence | 若 profile 属历史分支，不能写成 production headline |
| D1 all80 6 h | 80 样本，约 23.93% 节泵，80/80 wins，fallback 0 | primary 6 h evidence | 需确认 profile 与论文主策略一致性 |
| P1 transient peak/future decay | 约 24 cases，约 18.13% | mechanism figure | 适合解释 future decay causal value |
| P4/C1 soft-decay | 28 cases，约 34%，26/28 wins，fallback 0 | primary/supporting regime | 需和 P2/D1 分开 |
| W1 direction-stable | 20 × 12 h，约 39.60%，fallback 0 | longer positive regime support | 不能和 12 h guard10 mixed result混成一个结论 |
| W1 boundary signflip | 11 cases，prediction-gated 0% saving | boundary/abstain evidence | 重点不是节泵，而是避免错误触发 |
| C3 gusty | 12-case runtime-release，约 19.78%，fallback 0 | boundary/cost-aware evidence | 需展示 latch 和姿态代价 |
| PSC mixed24 refresh-on | 24 cases，约 37.87%，但 t>5 增加较多 | appendix/cautionary evidence | 不作为 headline |
| 12 h guard10 | 10+ mixed-regime check，gain 0.45 节泵 9.51% | robustness disclosure | 正文或附录必须出现 |

执行判断：

- 若要把 P2/D1/P4/C1 作为正文主线，论文表述必须是“regime-stratified forecast-actionable evidence”，不是“当前 production profile 全局结果”。
- 若导师要求论文只写当前 production-near profile，则必须新增或整理一组当前 profile 的 broad 6 h casebook，把 P2/D1/P4/C1 降为机制或附录。

## 6. 阶段 C：补齐指标与统计摘要

### C1. 必须统计的指标

每个 case 至少统计：

| 指标组 | 指标 |
|---|---|
| 泵耗 | closed pump work、primary pump work、pump saving m3、pump saving pct、pump lb/min、cumulative pumped lb |
| 姿态典型值 | pitch p95、roll p95、max-axis p95、pitch RMS、roll RMS、pitch max、roll max |
| 姿态暴露 | time_over_3/4/5/7.5/10 deg |
| 持续风险 | max_continuous_over_3/4/5/7.5/10 deg |
| 暴露面积 | area_over_3/4/5/7.5/10 deg |
| 安全与执行 | fallback ratio、fallback seconds、latch switches、target refresh/reuse/release、pump start count |
| 预测相关 | forecast horizon、risk-window flag、wind speed/direction prediction error 或 risk classification quality |

### C2. 聚合统计

每个 regime 至少给出：

- N；
- mean pump saving；
- median pump saving；
- IQR；
- win-rate；
- fallback count；
- worst-case `d_time_over_7.5deg`；
- worst-case `d_time_over_10deg`；
- max posture p95 increase；
- 是否存在 pump 和姿态同时变差的 dominated case。

推荐输出：

```text
outputs/wind_prediction/paper_20260606_execution/regime_summary.csv
outputs/wind_prediction/paper_20260606_execution/regime_summary.md
outputs/wind_prediction/paper_20260606_execution/per_case_paired_delta.csv
```

通过标准：

- 表中不能只有均值。
- 所有 weak/negative cases 必须保留。
- 任何 `t>10 deg` 增加都必须在 notes 中解释。

## 7. 阶段 D：正文图表制作

正文推荐 6 张图、3 张表。若篇幅允许，可扩展到 8 张图。

### D1. 正文主图

| 编号 | 标题 | 内容 | 目的 |
|---|---|---|---|
| Fig. 1 | 预测监督主动压载控制框架 | 历史风况、短时预测、风险窗口、监督层、原有闭环反馈、主动压载执行 | 说明预测是监督信息，不直接下泵命令 |
| Fig. 2 | 代表性预测可行动工况时域响应 | 风速/风向/risk、supervisory state、pump、cumulative pump、pitch、roll | 展示为什么能省泵且姿态受控 |
| Fig. 3 | 分工况 paired pump-work reduction | P2、D1、P1、P4/C1 等分面 scatter/box/connected dots | 防止只报总均值 |
| Fig. 4 | 姿态服务代价与尾部暴露 | pitch p95、roll p95、time>5、time>7.5、time>10 分工况 | 回答节泵是否牺牲姿态 |
| Fig. 5 | 边界与拒绝动作工况 | C3、W1 signflip、PSC mixed24 分面展示收益和代价 | 主动披露边界，降低 cherry-picking 风险 |
| Fig. 6 | 12 h mixed-regime robustness | current-only、gain 0.40、gain 0.45 的 pump work 与阈值暴露 | 说明长时段混合工况收益边界 |

可选扩展：

| 编号 | 标题 | 内容 |
|---|---|---|
| Fig. 7 | pump saving vs attitude exposure Pareto | x 轴节泵百分比，y 轴姿态暴露代价，按 regime 标色 |
| Fig. 8 | forecast/risk quality summary | 多 horizon 风速/风向误差或 risk classification 指标 |

### D2. 正文主表

| 编号 | 标题 | 内容 |
|---|---|---|
| Table 1 | 工况、样本与证据层级 | regime、N、horizon、profile、baseline、论文角色 |
| Table 2 | 6 h 主结果分工况汇总 | pump saving、win-rate、fallback、pitch/roll p95 delta、t>5/7.5/10 delta |
| Table 3 | 边界与 12 h 鲁棒性汇总 | C3、W1 signflip、PSC、guard10 12 h 的收益、代价和结论标签 |

### D3. 图表风格要求

- 中文正文图标题简洁，图注解释完整。
- 图内不要塞长段说明文字。
- 分工况图优先使用 small multiples，不把不同机制混在同一条时间轴。
- 所有阈值线统一：5 deg、7.5 deg、10 deg。
- 泵相关图必须明确单位：`lb/min`、`lb`、或 `m3`，不能混用且不说明。
- 若使用平滑线，图注写明“smoothing is for visualization only”。

## 8. 阶段 E：附录图表制作

用户要求每种工况至少 5 个图。附录采用统一模板，每个 regime 或代表 case 输出以下五类图：

| 图类型 | 文件命名建议 | 内容 |
|---|---|---|
| 风况总览 | `{regime}_{case}_01_wind_risk.png` | wind speed、wind direction、risk signal、forecast window |
| pitch 单独图 | `{regime}_{case}_02_pitch.png` | closed vs primary pitch，含 5/7.5/10 deg 阈值线 |
| roll 单独图 | `{regime}_{case}_03_roll.png` | closed vs primary roll，含 5/7.5/10 deg 阈值线 |
| 水泵图 | `{regime}_{case}_04_pump_lb.png` | pump lb/min、cumulative pumped lb |
| 压载目标图 | `{regime}_{case}_05_ballast_target_lb.png` | ballast mass lb、target mass lb、target refresh/reuse 标记 |

附录推荐覆盖：

- P2 代表 case；
- D1 代表 case；
- P1 relief-decay 代表 case；
- P4/C1 soft-decay 代表 case；
- W1 direction-stable 代表 case；
- W1 signflip boundary 代表 case；
- C3 gusty 代表 case；
- PSC mixed24 代表 case；
- 12 h guard10 代表 case或 aggregate per-case delta。

通过标准：

- 每类工况的五张图必须信息不同。
- pitch 和 roll 不合并。
- 不同工况不混画。
- 附录中必须保留至少一个弱收益或边界 case，不能只放正例。

## 9. 阶段 F：正文写作顺序

### F1. 先写第 3 章

优先写 `3.1 仿真工况与评价指标` 和 `3.2 仿真结果分析`，因为结果边界决定摘要和结论。

`3.1` 必须说明：

- baseline 是无预测闭环反馈策略；
- prediction-supervised strategy 使用相同执行层和安全层；
- 6 h 是主 episode-level horizon；
- 12 h 是 robustness disclosure；
- 24 h 不作为当前短论文主张；
- 指标来自 raw 1 Hz 数据；
- 姿态阈值 3/4/5/7.5/10 deg 的语义。

`3.2` 写作顺序：

1. 先讲主结果总体趋势；
2. 再讲代表工况机制；
3. 再讲分工况 paired statistics；
4. 再讲姿态代价与 severe tail；
5. 再讲边界/拒绝动作工况；
6. 最后讲 12 h robustness boundary。

### F2. 再写第 1 章

第 1 章围绕“预测监督层如何接入主动压载”展开。

必须强调：

- 预测不是直接输出泵命令；
- 原有闭环反馈控制器仍是执行层；
- 预测输出被转化为趋势、风险窗口和危险预警；
- 监督层做 regime identification、strategy selection、constraint/release；
- barrier 是 forecast-source-independent planner safeguard，不是 learned-specific trick。

建议使用公式：

- 六自由度运动方程；
- 风载荷公式；
- 原有闭环反馈控制公式；
- 水泵质量变化约束；
- 候选策略评价函数。

### F3. 再写第 2 章

第 2 章要够用但不能变成风预测模型竞赛。

必须包含：

- 10 min 风况数据；
- 风速/风向或风矢量表示；
- 历史窗口和预测 horizon；
- 0-20 / 20-40 / 40-60 min 等风险窗口；
- 风速爬升、风向变化、持续高风、瞬态变化等风险信号；
- 风况预测误差或风险识别质量的简要结果。

建议使用公式：

- 风矢量与风速风向转换；
- 短时风况预测模型；
- 风况风险窗口识别；
- 预测压力代理信号。

### F4. 再写引言

引言采用任务驱动结构：

1. 浮式风机受风浪流耦合影响，姿态响应和压载调节重要。
2. 主动压载可以改善姿态和减少不必要调节，但现有方法多依赖当前姿态、平均风载或预设工况。
3. 短时风况预测可提供未来趋势和风险窗口。
4. 现有风预测研究多关注误差，未直接回答预测如何改善主动压载调节。
5. 本文提出短时风况预测驱动的主动压载监督调节方法，并用分工况 paired casebook 验证。

引用映射：

- 浮式平台背景：Roddier、Salic、Gaertner、OpenFAST；
- 主动压载：Meng、Mahfouz、Stansby；
- preview control：Wakui、Shah 或其他 FOWT MPC/feedforward；
- 风预测：Liu & Chen、Xie、Fuentes-Barrios、Sari、Blazakis、Modé、Alves；
- 运维意义：McMorland。

### F5. 最后写摘要和结论

摘要必须等主结果数值冻结后再填。

摘要结果句建议模板：

```text
结果表明，在[主工况/样本池]中，预测监督调节策略相较于无预测闭环反馈策略使累计泵耗降低[X]%，同时[7.5/10 deg 严重姿态暴露结论]；在 12 h 混合工况鲁棒性检验中，候选策略总泵耗降低 9.51%，但 5 deg 服务压力阈值附近存在轻微暴露代价。
```

结论建议分三条：

1. 方法贡献：预测趋势、风险窗口和预警信号可作为主动压载监督层输入。
2. 主结果：6 h forecast-actionable 工况中节泵，并保持 severe attitude exposure 有界。
3. 边界：混合长时段收益被稀释，12 h 结果作为边界披露，24 h deployment validation 作为后续工作。

## 10. 阶段 G：审稿风险自检

定稿前逐项检查。

### G1. 6 h 质疑

审稿人可能问：

> Why only 6 h?

论文中必须回答：

- 控制器基于 10 min rolling wind bucket；
- 6 h 包含多个监督决策周期；
- 本文目标是 episode-level supervisory efficacy；
- 12 h mixed-regime robustness 已披露；
- 24 h 属 deployment-scale future work。

### G2. cherry-picking 质疑

必须有：

- casebook selection rule；
- 每工况 N；
- per-case paired delta；
- weak/negative cases；
- boundary/abstain cases。

不能只有：

- 一个最好看的时域图；
- 一个总平均节泵百分比；
- 只展示 P2 或 W1 正例。

### G3. 姿态代价质疑

必须报告：

- pitch p95；
- roll p95；
- max-axis p95；
- time_over_5/7.5/10；
- max_continuous_over_5/7.5/10；
- area_over_5/7.5/10；
- fallback；
- worst case。

任何 `t>5 deg` 增加必须解释为 service-band trade-off，不能隐藏。

任何 `t>7.5 deg` 或 `t>10 deg` 增加必须进行 case-level explanation。

### G4. profile 混用质疑

必须检查：

- 正文 headline 是否只来自同一 profile 或清晰分工况 profile；
- deadband、P2、C3、PSC 是否被错误合并；
- 12 h gain 0.45 是否被写成 production default；
- historical diagnostic result 是否被写成 final controller result。

### G5. baseline 公平性质疑

必须写明：

- baseline 是普通无预测闭环反馈控制；
- baseline 包含基本工程约束；
- no-preview smoothing / reactive suppression 检查显示 baseline 不是故意做弱；
- prediction-supervised 与 current-only 的主要差异是未来风况信息进入监督层。

## 11. 阶段 H：最终交付包

最终建议形成以下文件结构：

```text
outputs/wind_prediction/paper_20260606_execution/
  claim_evidence_map.csv
  claim_evidence_map.md
  per_case_paired_delta.csv
  regime_summary.csv
  regime_summary.md
  main_figures/
    fig1_control_architecture.png
    fig2_representative_forecast_actionable_episode.png
    fig3_regime_paired_pump_reduction.png
    fig4_attitude_service_tradeoff.png
    fig5_boundary_abstain_regimes.png
    fig6_12h_guard_robustness.png
  appendix_figures/
    ...
  tables/
    table1_regime_evidence_hierarchy.csv
    table2_primary_6h_results.csv
    table3_boundary_robustness_summary.csv
  manuscript_notes/
    result_paragraphs.md
    reviewer_risk_response.md
```

论文写作文件建议：

```text
docs/paper_draft_20260606.md
docs/paper_revision_checklist_20260606.md
```

## 12. 推荐执行顺序

按优先级执行，不要同时发散。

### 第 1 步：结果归属冻结

- [ ] 建立 claim-evidence map。
- [ ] 标注每个结果是否可用于 headline。
- [ ] 把 P2/C3/PSC/W1/12h guard 的论文角色分开。

完成标志：

- 能清楚回答“当前论文到底主张哪个 profile、哪些结果只是机制或边界证据”。

### 第 2 步：6 h casebook 冻结或补跑

- [ ] 确认当前是否已有同一 profile 下的 broad 6 h paired casebook。
- [ ] 如果没有，则冻结 casebook selection rule 后补跑。
- [ ] 如果使用历史分工况结果，则写清楚它们是 regime-stratified evidence，不是 production-universal evidence。

完成标志：

- Table 1 可填完整。

### 第 3 步：统计表生成

- [ ] 生成 per-case paired delta。
- [ ] 生成 regime summary。
- [ ] 生成 12 h guard robustness table。
- [ ] 检查 raw 1 Hz 指标口径。

完成标志：

- Table 2 和 Table 3 可填完整。

### 第 4 步：主文图生成

- [ ] Fig. 1 框架图。
- [ ] Fig. 2 代表工况时域。
- [ ] Fig. 3 分工况节泵 paired 图。
- [ ] Fig. 4 姿态代价图。
- [ ] Fig. 5 边界/拒绝动作图。
- [ ] Fig. 6 12 h robustness 图。

完成标志：

- 每张图都有明确图注和对应正文段落。

### 第 5 步：附录图生成

- [ ] 每个选中 regime 输出 wind/risk、pitch、roll、pump lb、ballast target lb 五类图。
- [ ] 检查不同工况没有混画。
- [ ] 检查 pitch/roll 没有合成一张难读图。

完成标志：

- 附录可以支撑“不是只挑一张好看曲线”。

### 第 6 步：写第 3 章

- [ ] 写 `3.1 仿真工况与评价指标`。
- [ ] 写 `3.2 仿真结果分析`。
- [ ] 插入 Table 1-3 与 Fig. 2-6。

完成标志：

- 结果章节能独立说明主张、收益、代价和边界。

### 第 7 步：写第 1、2 章

- [ ] 写算法框架与流程。
- [ ] 写风况预测与风险识别。
- [ ] 插入 Fig. 1 和关键公式。
- [ ] 保证预测不是直接泵命令。

完成标志：

- 方法章节能解释第 3 章中监督层动作的来源。

### 第 8 步：写引言、摘要、结论

- [ ] 引言按“背景、已有方法、缺口、本文方法、验证”展开。
- [ ] 摘要填最终数字。
- [ ] 结论明确区分 6 h、12 h、24 h。

完成标志：

- 全文没有过强 claim。

### 第 9 步：审稿风险自检

- [ ] 逐项检查 6 h、cherry-picking、姿态代价、profile 混用、baseline 公平性。
- [ ] 准备一段 reviewer response。
- [ ] 检查所有图表是否在正文中被引用。

完成标志：

- 能用 1 页文字回答最严格审稿人的主要质疑。

### 第 10 步：最终排版与打包

- [ ] 统一术语。
- [ ] 统一单位。
- [ ] 统一图注格式。
- [ ] 检查引用。
- [ ] 整理附录和结果索引。

完成标志：

- 正文、图表、附录、结果索引、参考文献全部可交给导师或投稿系统。

## 13. 当前最小可执行版本

如果时间非常紧，最低限度完成以下内容即可形成可讲清楚的论文版本。

1. Table 1：工况与样本数。
2. Table 2：主 6 h 分工况结果。
3. Table 3：边界与 12 h robustness。
4. Fig. 1：方法框架。
5. Fig. 2：一个正例时域机制图。
6. Fig. 3：分工况 paired pump saving。
7. Fig. 4：姿态暴露代价。
8. Fig. 5：W1 signflip 或 C3 边界图。
9. Fig. 6：12 h guard robustness。
10. 附录每类工况五张图，至少覆盖 P2、P1、W1 signflip、C3、12 h guard。

最小版本仍必须保留 12 h robustness disclosure。不要为了突出 6 h 高收益而删掉 12 h。

## 14. 禁止事项

- [ ] 禁止把所有工况混成一个总节泵 headline。
- [ ] 禁止把 30% 左右收益写成 production long-duration saving。
- [ ] 禁止只展示正例，不展示边界或弱例。
- [ ] 禁止 pitch 和 roll 在关键图中混成难以判断的单图。
- [ ] 禁止用平滑后的图线计算指标。
- [ ] 禁止把 prediction model 写成直接输出 ballast command。
- [ ] 禁止把 12 h gain 0.45 写成已冻结 production default。
- [ ] 禁止把 24 h 写成已经验证，除非确实完成稳定 24 h campaign。

## 15. 下一次开工建议

下一次实际执行时，建议从下列任务开始：

1. 生成 `claim_evidence_map.csv/md`。
2. 确认是否已有同一 profile 的 broad 6 h paired casebook。
3. 若已有，直接生成 Table 1-3。
4. 若没有，先冻结 casebook selection rule，再补跑 current-only 与 prediction-supervised paired results。
5. 先做 Fig. 3、Fig. 4、Fig. 6，因为这三张最能决定论文能不能站住。

这一步完成后，再进入正文写作。

## 16. 当前论文材料缺口与补齐清单

本节用于回答“现在到底还缺什么论文材料”。结论是：当前已经具备很多实验、图像和策略文档，但还缺少一个能直接支撑论文写作的统一证据包。后续不要盲目继续扩跑，应优先把已有材料规整为可引用、可复核、可写进正文的形式。

### 16.1 已经具备的材料

1. 论文路线与主张边界已经基本明确。
   - `docs/current_paper_framework.md`
   - `docs/paper_validation_strategy_20260605.md`
   - `docs/control_validation_protocol.md`
   - `docs/paper_execution_plan_20260606.md`

2. 姿态指标、公式和写法基础已经有初稿。
   - `docs/attitude_metric_semantics.md`
   - `docs/paper_formula_reference.md`
   - `docs/paper_style_profile_xu_group.md`

3. 12 h guard10 mixed-regime robustness 数据已经足够作为长时段披露材料，不建议重复跑同类 12 h。
   - `outputs/wind_prediction/guard10_12h_current_only_baseline_20260605`
   - `outputs/wind_prediction/guard10_12h_production_learned_candidate_20260605`
   - `outputs/wind_prediction/guard10_12h_production_learned_gain045_20260605`
   - 可固定使用的结论：gain 0.40 相对 current-only 节泵 3.08%；gain 0.45 相对 current-only 节泵 9.51%；严重姿态暴露 `t>10 deg` 基本不变。

4. 已经有一批按用户要求生成的分工况曲线图，且 pitch、roll、pump pounds、ballast target pounds 已分开。
   - `outputs/wind_prediction/multi_regime_curve_figures_20260606_py312`
   - 当前包含 5 类工况：`c3_gusty_oscillation`、`d1_neutral_headroom`、`guard10_mixed_12h`、`p2_long_boundary`、`psc_selector_mixed_pool`
   - 每类已有 5 张图：wind overview、pitch、roll、pump pounds、ballast mass targets。

5. 历史 regime-specialist 证据很丰富，可以用于解释机制、边界和附录。
   - D1 neutral/headroom：有 80 case 级别的分工况统计，但属于 deadband/历史分支。
   - C1/P4 soft-decay：有稳定方向软衰减分支结果，可用于机制或附录。
   - C3 gusty/oscillation：有阵风振荡材料，可用于说明预测监督在振荡场景中的边界。
   - W1 stable/signflip：有预测必要性、方向翻转、拒绝动作等材料，可用于边界图和审稿解释。

### 16.2 必须补齐的 P0 材料

这些材料不补，论文主结果容易被质疑为“挑图”“混分支”或“无法复核”。

1. 统一 claim-evidence map。
   - 需要生成 `claim_evidence_map.csv` 和 `claim_evidence_map.md`。
   - 每一条论文主张必须对应：样本数、工况、horizon、control profile、指标、图表编号、源输出目录。
   - 明确标注每个结果是 primary、robustness、mechanism、boundary 还是 appendix only。

2. 同一 profile 下的 broad 6 h paired casebook。
   - 这是当前最大的硬缺口。
   - 需要确认是否已经存在“current-only vs prediction-supervised”且使用同一 production-near profile 的广覆盖 6 h 成对样本。
   - 如果不存在，必须先冻结 case selection rule，再补跑 6 h paired casebook。
   - 不能把 D1/P2/C3/PSC 历史分支收益直接合并成当前 production headline。

3. 统一 per-case paired delta 表。
   - 需要至少包含：case_id、regime、duration_h、profile、baseline_pump、learned_pump、saving_pct、pitch/roll/RMS、`t>5`、`t>7.5`、`t>10`、max attitude、fallback/latch/safety event。
   - 这是 Table 2、Table 3、Fig. 3、Fig. 4 的共同数据源。

4. 主论文图表包。
   - Fig. 1：算法框架图。
   - Fig. 2：代表正例时域机制图。
   - Fig. 3：分工况 paired pump saving 图。
   - Fig. 4：姿态暴露代价图。
   - Fig. 5：边界或失败模式图，例如 W1 signflip、C3 oscillation、P2 long-boundary。
   - Fig. 6：12 h guard robustness 图。
   - Table 1：工况与样本数。
   - Table 2：6 h 主结果。
   - Table 3：12 h robustness 与边界披露。

5. 单位定义与换算说明。
   - 现有结果中 pump work 常用 `m3`，用户展示需要 `lb`。
   - 论文正文建议主单位优先用 SI，即 `m3` 或 `kg`，图中可加 `lb` 辅助轴或附录说明。
   - 必须写清楚水密度假设、`pump lb/min` 与 `cumulative pumped lb` 的定义，避免 `m3`、`kg`、`lb` 混用。

6. 样本独立性与无泄漏说明。
   - 需要写明训练/验证/测试或 casebook 的时间窗口是否重叠。
   - 需要说明预测模型没有使用未来信息。
   - 如果历史工况样本存在重叠，必须在附录中披露并限制其论文角色。

### 16.3 应尽快补齐的 P1 材料

这些材料不一定阻止成稿，但会显著影响论文可信度。

1. 风况预测模型的 paper-ready 摘要。
   - 数据来源、采样间隔、训练/验证/测试划分。
   - 输入历史窗口、预测 horizon、输出变量。
   - 与 persistence baseline 的对比。
   - 风速 MAE/RMSE、风向 angular error、事件识别 precision/recall/F1。
   - 一张 observed-vs-predicted 代表图。
   - 只需证明预测信号足以服务监督层，不必写成模型竞赛论文。

2. 仿真模型与参数表。
   - 平台/风机模型来源。
   - 自由度或简化假设。
   - 水泵容量、限幅、响应时间。
   - 压载舱布局与质量范围。
   - 仿真步长、控制周期、case duration。

3. 统计稳健性材料。
   - 每类工况至少给 mean、median、IQR、win rate。
   - 建议补 bootstrap confidence interval 或 paired nonparametric test。
   - 如果时间有限，至少给 per-case paired scatter 和箱线图，避免只报均值。

4. 附录图像索引。
   - 当前已有 5 类工况 × 5 图。
   - 还建议补 P1 transient/future-decay、P4/C1 soft-decay、W1 stable、W1 signflip/abstain 等工况。
   - 每类仍保持至少 5 张图，不把不同机制混在一个土/图里。

5. 审稿风险问答。
   - 为什么主证据是 6 h 而不是 24 h。
   - 为什么 12 h 收益低于部分 6 h 工况。
   - 为什么不把 historical specialist 结果写成 production headline。
   - 如何保证 baseline 公平。
   - 姿态代价是否可接受。

### 16.4 可以后补的 P2 材料

这些材料用于提升完整度，不应阻塞当前论文主线。

1. 24 h production-scale validation。
   - 当前不作为短论文必要条件。
   - 只有当论文改写为长期部署级 production controller 时才必须补。

2. 更完整的消融实验。
   - 无预测、预测风险窗口、预测姿态门控、预测泵送抑制、完整监督层。
   - 当前可以先用已有边界和机制结果替代。

3. 更完整的参考文献矩阵。
   - 最终需要 BibTeX、DOI、引用位置矩阵。
   - 目前先保证引言和方法相关文献足够支撑，不必立刻做系统综述级别材料。

4. 更精细的排版材料。
   - 摘要、关键词、图注、nomenclature、data/code availability、funding/conflict statement。
   - 等主结果数字冻结后再写更稳。

### 16.5 不建议重复补的材料

1. 不建议继续无目标地扩跑 12 h mixed-regime。
   - 现有 12 h 已经足以说明长窗口混合工况下收益会稀释。
   - 继续扩跑的边际收益小于先整理 6 h 主证据。

2. 不建议继续只生成单 case 好看曲线。
   - 当前最缺的是 aggregate paired evidence，不是更多代表图。
   - 代表图只用于解释机制，不能替代表格和统计图。

3. 不建议把历史分支结果重新包装成当前 controller 结果。
   - 可以用，但必须标注为 regime-specialist/mechanism/boundary evidence。

### 16.6 立即补齐顺序

下一步按以下顺序执行。

1. 建立 `outputs/wind_prediction/paper_20260606_execution/` 统一结果包。
2. 生成 `claim_evidence_map.csv/md`，先把所有可用材料和论文角色登记清楚。
3. 核查是否已有 production-near broad 6 h paired casebook。
4. 如果已有，直接汇总 Table 1-3；如果没有，冻结 selection rule 后补跑 6 h paired casebook。
5. 从统一 per-case delta 表生成 Fig. 3、Fig. 4、Fig. 6。
6. 再补 Fig. 1 方法框架图、Fig. 2 机制正例图、Fig. 5 边界图。
7. 最后写第 3 章结果段落，并把 12 h guard robustness 作为诚实披露写入正文或附录。
