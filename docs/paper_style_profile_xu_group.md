# 师门小论文写作风格档案

Source samples:

- `/Users/saintyoung/Desktop/小论文/用于动力定位船舶运动预测的混合模型算法_无作者.docx`
- `/Users/saintyoung/Desktop/小论文/刘俊宏小论文.docx`

Purpose: calibrate future paper drafting toward the observed small-paper style used in the group. This is a soft style guide, not a content template.

## Overall Style

The style is a compact engineering journal style. It privileges clear task motivation, method decomposition, validation setup, and quantified comparison over broad theoretical discussion. The prose is formal but not ornate. Most paragraphs move directly from engineering demand to technical limitation to proposed method.

Typical paper logic:

1. Application scenario and engineering importance.
2. Existing methods or current research state.
3. Specific insufficiency under the target scenario.
4. Proposed method/strategy.
5. Simulation or experimental verification.
6. Quantified result and practical support statement.

Recommended voice for our paper:

- Use "本文/本研究" as the main subject.
- Prefer "针对...需求/难点，提出/构建/建立..." as the main problem-solution frame.
- Keep claims tied to figures, tables, experiments, or simulation results.
- State practical value as "为...提供理论支撑/数据支持/新的解决方案/优化指导".

## Abstract Pattern

Observed abstract structure is highly regular:

1. One sentence on application scenario and importance.
2. One sentence beginning with "针对..." that names the technical difficulty or demand.
3. One sentence stating the proposed method.
4. One or two sentences describing how the method is built.
5. One sentence describing experiment/simulation data.
6. One result sentence beginning with "结果表明".
7. One closing sentence beginning with "研究成果..." or "综上所述...".

Reusable abstract skeleton:

```text
针对[应用场景]中[关键需求/难点]，本文/本研究提出一种[方法名称]。该方法[核心机制1]，并通过[核心机制2]实现[目标]。在此基础上，基于[数据/仿真/试验]对[方法性能/策略精度]进行分析。结果表明：[主要定量结果]。[研究成果/综上所述]，该方法可为[工程系统]提供[理论支撑/预警信息/优化指导]。
```

## Introduction Pattern

The introduction is usually short and task-driven. It does not spend many paragraphs on large-field background. The preferred expansion is:

1. Start from the real engineering system and why the target quantity matters.
2. Give 1-2 concrete examples of operations or subsystems affected by the problem.
3. Summarize existing method families.
4. Explain why those methods are insufficient in the specific scenario.
5. End with the paper's method and validation object.

Common move types:

- "在...时，系统会受到...影响，从而产生..."
- "...对于...至关重要。"
- "近年来，针对...已取得...，但对于...仍..."
- "与...相比，虽然...，但...，这使得...更加困难。"
- "上述研究中，...，而...，因此现有方法并不适用。"
- "因此，本文基于...特点，提出...，并建立...以验证..."

For our FOWT paper, intro should avoid overexplaining every control detail. Use a larger framing:

- Existing active ballast systems maintain heel/trim/tower verticality mainly through current-state or low-frequency compensation.
- Floating wind loading is time-varying and delayed control action has pump and posture consequences.
- Existing prediction metrics do not directly answer whether prediction improves ballast regulation.
- Therefore this study introduces a forecast-informed active ballast framework and validates the overall control benefit against non-predictive baselines.

## Section Architecture

Observed top-level structures:

- Title
- Chinese abstract
- Keywords
- English title/abstract/keywords
- Introduction without an explicit "0 引言" heading in some samples
- Numbered method/system sections
- Simulation/experiment/result section
- Conclusion
- Acknowledgement if needed
- References

Typical numbered body:

```text
1 [System or method design]
2 [Core algorithm/model]
3 [Strategy or model construction]
4 [Simulation/experiment method]
5 [Result/error/performance analysis]
6 结论
```

For our paper, a matching structure could be:

```text
1 浮式风机主动压载预测控制问题描述
2 短时风况预测与风险分段方法
3 基于风况预览的主动压载控制策略
4 仿真模型与评价指标
5 预测控制效果分析
6 结论
```

## Paragraph Rhythm

Paragraphs are medium-long and information dense. They often contain 3-5 logical clauses in one paragraph. The rhythm is steady, not punchy.

Observed tendencies:

- Methods paragraphs often begin with purpose or condition, then give technical components.
- Results paragraphs begin with the analysis goal, then describe figure/table content, then interpret the trend.
- Conclusions use short numbered items.

Reusable paragraph pattern:

```text
为[验证/分析/探究]...，本研究设置了...。在保持...一致的前提下，分别对...进行对比。结果如图/表所示，随着...，...呈现...趋势。在相同...条件下，本文方法...，说明...
```

## Preferred Phrases

High-frequency, safe phrases:

- 针对...需求/难点
- 提出一种...方法/策略/算法
- 基于...建立/构建...
- 通过...实现/验证/计算...
- 为了验证...效果
- 为深入研究...性能表现
- 在保持...一致的前提下
- 结果如图...和表...所示
- 结果表明...
- 由结果可知...
- 可以观察到...
- 进一步分析...
- 有助于...
- 显著降低/提高...
- 为...提供理论支撑/数据支持/优化指导

Hedging and causality style:

- Use "可能", "有望", "可以", "有助于" for implications.
- Use "说明", "表明", "推断", "导致", "进而" for causal explanation.
- Avoid overly philosophical language.

## Method Writing Habits

Method sections use decomposition and procedural clarity.

Observed moves:

- Define the system or module first.
- State inputs and outputs.
- Explain why the module is needed.
- Describe implementation choices.
- Refer to equations and figures rather than long conceptual exposition.

Common structures:

```text
[模块]的构造如图...所示，包括...。其输入特征包括...，输出特征为...。
```

```text
考虑到...，本文采用...。在...过程中，首先...，随后...，最后...
```

For our paper, use module framing:

- 风预测模块: history window, forecast horizon, segmented risk.
- 控制策略模块: forecast-informed mode/ballast adjustment generation.
- 执行层: baseline feedback controller and pump constraints.
- 评价模块: posture, pump work, fallback, prediction-vs-current comparison.

## Result Writing Habits

Results are figure/table anchored and trend-first.

Preferred flow:

1. State what is being evaluated.
2. State comparison baseline.
3. Point to figure/table.
4. Describe trend.
5. Give key number.
6. Explain mechanism briefly.

Reusable result skeleton:

```text
为了验证[方法]的[效果/性能]，本研究选取[baseline]作为对比基准。结果如图...和表...所示，在[条件]下，本文方法相较于[baseline]的[指标]降低/提高了...。由结果可知，...[机制解释]，因此...[工程意义]。
```

For our paper, do not overuse negative-result language in the main result prose. If prediction's contribution is global framework improvement, write as:

- "预测控制策略整体上改善了主动压载系统的调节效果。"
- "相较于仅基于当前姿态反馈的控制方式，基于风况预览的方法能够提前识别未来风载变化并调整压载策略。"
- "在保证姿态安全约束的前提下，系统泵耗/姿态误差/超限时间得到改善。"

## Conclusion Style

Conclusions are direct and numbered. They restate the proposed method and list 3-4 findings. A final limitation/future-work paragraph is acceptable.

Reusable conclusion opening:

```text
本文针对[场景]下的[需求/问题]，提出了[方法/策略]，并通过[试验/仿真/数据]对其性能进行了验证，主要结论如下：
```

Conclusion item style:

- "（1）..."
- "（2）..."
- "（3）..."

Future work style:

```text
本文研究仅限于[限定条件]。为了更广泛的应用，后续需要[扩展方向]。
```

## Citation and Literature Style

The samples use compact citation summaries. They usually name method families and representative works, not extensive debate.

Common pattern:

```text
当前，...技术发展迅速。A等[1]提出/开发了...，B等[2]提出了...，C等[3]采用...。
```

Then transition:

```text
上述研究中，...，而...。对于[本文场景]，现有方法...，因此...
```

For our literature review, keep each paragraph focused:

- Existing active ballast/control systems.
- Floating wind preview/predictive control.
- Wind prediction models and event/risk prediction.
- Gap: limited work on using short-term wind prediction as an active ballast decision input.

## Style Guardrails For Our Paper

Do:

- Lead with engineering demand and validation.
- Use "针对/基于/通过/结果表明" style.
- Present the method as a system-level algorithmic improvement.
- Keep detailed mechanism caveats in Discussion or validation design, not in the opening claim.
- Make tables and figures carry the quantitative proof.

Do not:

- Say "市面上都是被动系统" too bluntly. Prefer "公开资料中，多数主动压载系统仍以当前姿态或低频风载补偿为主要控制依据".
- Overemphasize that only one small mechanism improves. The writing style permits system-level statements if backed by overall comparisons.
- Use long philosophical framing around causality; keep the falsifier/oracle logic as a methodology contribution.
- Make unsupported broad market claims without citation.

## Drafting Directive

When drafting future sections for this project, imitate the samples by default:

- Chinese body text in simplified Chinese.
- Engineering journal tone.
- Short title-like section headings.
- "需求-不足-方法-验证-结果" progression.
- Quantified claims tied to "图/表/结果表明".
- Final practical value sentence: "可为浮式风机主动压载控制系统提供..."
