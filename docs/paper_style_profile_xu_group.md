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

## Method Prose From Strong Prediction-Control Papers

This section records writing lessons from a live review of prediction-assisted control and wind-control papers, including LSTM-MPC irrigation scheduling, LIDAR-assisted wind turbine regulation, FOWT actuated-platform control, MPC with preview, disturbance-forecast MPC, scenario-based NMPC, SODA-MPC, wind-speed LSTM forecasting, wind-ramp event prediction, and risk-aware BLSTM-MPC vehicle planning.

Core observation: good method prose does not open with a generic module statement. It starts from the controlled system's specific need, then introduces the prediction quantity, then states how that quantity changes the controller. The prediction model is usually framed as an information source or process model, not as a tutorial topic.

### What To Learn

- Start from the actuator or control bottleneck. Good papers write from the system constraint: slow irrigation decisions, wind preview for load rejection, dynamic pedestrians for vehicle planning, disturbance forecasts for MPC, or platform tilt control. For this paper, start from ballast transfer delay and target-water updating, not from "LSTM module".
- Attach prediction to a downstream control consequence in the same paragraph. The sentence should answer: after prediction is obtained, what changes in the controller? For this paper: future wind vectors and wind-condition event probabilities enter equivalent attitude demand, candidate-action screening, and target ballast-water generation.
- Use concrete verbs instead of explanatory scaffolding. Prefer "得到、估计、整理、映射、参与、筛选、生成" over "用于为...提供..." repeated across sentences.
- Keep neural-network scope short. If LSTM is only an upstream predictor, state its input, output, and training-determined readout parameters. Do not expand LSTM gates unless the prediction model itself is a contribution.
- Avoid defensive contrast sentences. Do not write "本文不对 LSTM 单元结构本身作改进，预测模块的作用是..." in the main prose. If scope must be clarified, put it in one compact sentence after the formula: "LSTM 在本文中承担风况预瞄量生成作用，后续控制律由预测输出与姿态反馈共同确定。"
- Let formulas carry definitions, not prose. Avoid prose that merely repeats symbols. After a formula, explain why the output matters physically: future wind vector gives direction and magnitude; wind-condition event probability gates whether the prediction should participate in active ballast target generation.
- Use parameterized time language in the method section. Write "历史风况序列" and "预测时域" in methods; move "120 min、60 min、6 个预测步" to validation settings.
- Event recognition belongs to tables unless it changes a control variable directly. Event names, thresholds, and probability gates should be introduced as criteria; the formula section should show the control consequence, such as \(g_k\mathbf q_k\).

### Sentences To Avoid

Avoid this style:

```text
短时风况预测模块用于为主动压载目标水量生成提供未来风况输入。由于压载调节过程受水泵流量和舱内水量变化速度限制，其执行效果往往滞后于外部风况变化。因此，在目标压载水量更新前引入未来短时风况信息，可为后续姿态需求融合和候选动作判别提供预瞄依据。
```

Problems: "用于为...提供" is empty; "由于...因此..." reads like a template; "可为...提供预瞄依据" stops before saying how the control law changes.

Avoid this defensive style:

```text
本文不对 LSTM 单元结构本身作改进，预测模块的作用是...
```

Problems: it leads with what the paper does not do; it sounds like a response to a reviewer rather than a method description.

### Preferred 1.2 Opening Pattern

Use this style when drafting the prediction subsection:

```text
压载水转移具有执行滞后，目标水量更新需要在当前姿态反馈之外获得短时风况预瞄量。本文以历史风况序列为输入，由 LSTM 提取风速、风向变化中的时序特征，并输出预测时域内的水平风矢量和风况事件概率。前者给出后续等效姿态需求的方向和强度来源，后者用于判断预测信息是否参与当前目标水量生成。
```

This paragraph is better because it moves in one line from physical bottleneck to prediction output to control consequence. It avoids a tutorial tone and does not apologize for using a standard LSTM.

After the prediction-output formula, use:

```text
式中，\(\mathbf h_t\) 为历史风况序列的时序特征，\(\widehat{\mathbf y}_{t+j}\) 为第 \(j\) 个预测步的水平风矢量，\(\widehat p_{k,t}\) 为第 \(k\) 个预测时段的风况事件概率。读出矩阵和偏置由训练数据确定，不作为主动压载控制参数。预测风矢量随后用于构造等效姿态调节需求，风况事件概率用于确定该预测信息是否进入目标水量生成。
```

### Reference Papers Checked For This Rule

- [Agyeman et al., LSTM-based model predictive control with discrete inputs for irrigation scheduling](https://arxiv.org/abs/2112.06352): prediction model introduced through the scheduler objective and discrete actuator problem.
- [Stockhouse et al., Control of a Floating Wind Turbine on a Novel Actuated Platform](https://arxiv.org/abs/2110.14169): actuator types are explained through platform tilt/heave control consequences.
- [Mahdizadeh et al., LIDAR-Assisted Exact Output Regulation for Load Mitigation in Wind Turbines](https://arxiv.org/abs/1906.07550): preview wind information is tied immediately to the feedforward gain and disturbance rejection.
- [Fang and Chen, Model Predictive Control with Preview](https://arxiv.org/abs/2202.12585): preview disturbance is introduced by how it modifies the prediction horizon and cost.
- [McCloy et al., Contraction-constrained MPC using Disturbance Forecasts](https://arxiv.org/abs/2205.04033): disturbance forecasts are justified by closed-loop performance and stabilisation under nonlinear processes.
- [Pippia et al., Scenario-based NMPC for Building Heating Systems](https://arxiv.org/abs/2012.02011): forecasts/scenarios are motivated by disturbance uncertainty and control robustness.
- [Contreras et al., SODA-MPC](https://arxiv.org/abs/2406.02436): learned prediction is paired with a runtime reliability monitor and fallback action.
- [Liang et al., Multi-variable stacked LSTM wind speed forecasting](https://arxiv.org/abs/1811.09735): LSTM details are appropriate when prediction itself is the main contribution.
- [Gupta et al., Wind ramp event prediction](https://arxiv.org/abs/1610.05009): event prediction is introduced through threshold-defined ramp classes and reliability needs.
- [Huang and Jafari, Risk-aware BLSTM-MPC vehicle motion planning](https://arxiv.org/abs/2301.06201): trajectory prediction is described through conflict-risk estimation and MPC planning.

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

## Figure And Result Plotting Requirements

This section records the plotting rules learned from engineering-control, floating-wind, wind-forecasting, and prediction-assisted-control papers. Use it as a hard writing requirement for later figure design.

Core principle:

- A paper figure must answer a specific evidence question. Tables carry exact values; figures carry trends, mechanisms, distributions, and boundary behavior.
- Representative time-series figures explain how the strategy works, but aggregate tables or distribution/sorted plots prove whether the strategy works across samples.
- Any figure showing 累计泵量降低 must be paired with attitude-threshold exposure evidence. Do not present actuator benefit without posture-side guardrails.
- Do not hide cost or side effects. High-level control papers usually report main benefit, actuator effort, motion/load response, and constraint satisfaction together.

For this paper:

- Use the current case name: 预测可作用工况.
- Use the current algorithm names: 姿态反馈算法 and 预测辅助算法.
- Do not use internal or old expressions in figures or captions: 窗口、闭环、节泵、节水、补泵、监督策略、gated、casebook、profile、gain.
- Statistical values must come from the raw simulation outputs. Smoothed curves may be used only for visual clarity and must not define metrics.
- The figure caption must state the sorting basis when cases are sorted by cumulative pump volume, pump-volume reduction, or attitude exposure.

Recommended figure logic for Chapter 3:

- Table 11 should show the 170-case overall comparison: cumulative pump volume, start-stop count, and attitude-threshold exposure.
- The main cumulative-pump figure should focus on the 170 预测可作用工况. Do not reintroduce old low-disturbance or boundary groups as the main evidence chain.
- For per-case pump-volume figures, use two vertically aligned panels when needed: the upper panel shows per-case difference or paired cumulative-pump-volume curves; the lower panel shows the cumulative difference. Positive values should consistently mean lower cumulative pump volume under the prediction-assisted strategy.
- Attitude results should use threshold-exposure time and percentage change rather than average attitude change as the main presentation. The cell format can combine both values, such as `+26 s (+0.0024%)`.
- Representative time-series figures should place wind condition, target water amount or pump action, pitch/roll response, and cumulative pump volume on a shared time axis.

Visual style:

- Prefer restrained line plots, paired bars, sorted contribution plots, small multiples, and heatmaps only when the matrix itself carries meaning.
- Use grey for the baseline and blue for the proposed strategy. Use light fill only to show paired differences; avoid decorative gradients and heavy shadows.
- Draw threshold/reference lines directly in the plot when discussing `T>2°`, `T>3°`, `T>4°`, `T>5°`, `T>7.5°`, or `T>10°`.
- Avoid three-dimensional bars, rainbow palettes, dense gridlines, long text inside plots, and titles that look like internal experiment logs.

## Drafting Directive

When drafting future sections for this project, imitate the samples by default:

- Chinese body text in simplified Chinese.
- Engineering journal tone.
- Short title-like section headings.
- "需求-不足-方法-验证-结果" progression.
- Quantified claims tied to "图/表/结果表明".
- Final practical value sentence: "可为浮式风机主动压载控制系统提供..."

## Context-First Drafting Rule

Before drafting any new subsection, first review the preceding narrative and identify what the new subsection must inherit. Do not write a section as a standalone module description.

For this paper, every method subsection must explicitly connect to:

- The introduction's problem statement: traditional active ballast relies on current attitude feedback and responds slowly to short-term wind-condition changes.
- The previous subsection's flow: the method proceeds through wind prediction, risk identification, attitude-demand fusion, target-water generation, constraint screening, and pump execution.
- The next subsection's role: each subsection should hand over a concrete variable or decision quantity to the following subsection.

For `1.2`, this means the section should not begin with a generic sentence such as "短时风况预测模块用于...". It should inherit the introduction and `1.1`: the original posture-feedback process lacks future wind information, so `1.2` defines the two prediction outputs that enter the later target-water calculation, namely future wind vectors and wind-condition event probabilities.

## 50+ Related Papers: Writing Lessons To Reuse

This section records a writing-oriented reading pass over related high-level papers and technical reports. The purpose is not to import their claims, formulas, or figures. The purpose is to extract how strong papers structure methods, validate prediction-control links, and write results without sounding like code or a lab report.

Use these lessons as a checklist before drafting any future section.

| # | Paper or source | Area | Writing lesson for this paper |
|---:|---|---|---|
| 1 | Agyeman et al., LSTM-based MPC with discrete inputs for irrigation scheduling | Prediction-assisted MPC | Introduce the LSTM through the downstream actuator scheduling problem; do not make network internals the narrative center. |
| 2 | Stockhouse et al., Control of a Floating Wind Turbine on a Novel Actuated Platform | FOWT active platform/ballast control | Present actuators by their effect on platform tilt and heave, then compare control outcomes across operating conditions. |
| 3 | Mahdizadeh et al., LIDAR-assisted exact output regulation for wind turbines | Wind preview control | Treat preview wind as a feedforward disturbance source; write the control consequence immediately after the preview quantity. |
| 4 | Fang and Chen, MPC with Preview | Preview control theory | Explain preview information by how it changes the current control decision, not by a standalone prediction block. |
| 5 | McCloy et al., disturbance-forecast MPC for nonlinear processes | Disturbance forecast control | Acknowledge forecast uncertainty through weighting or conservative use instead of claiming future information is always reliable. |
| 6 | Pippia et al., scenario-based NMPC for building heating | Scenario forecast MPC | Use scenarios or forecast groups to explain robustness, but keep the result claim tied to cost and constraint metrics. |
| 7 | Contreras et al., SODA-MPC | Prediction reliability and MPC | If prediction information is conditionally used, write the acceptance mechanism through the action it enables or disables. |
| 8 | Liang et al., multivariable stacked LSTM wind speed forecasting | Wind forecasting | Gate equations are appropriate only when the prediction network is the paper's main contribution. |
| 9 | Gupta et al., wind ramp event prediction with gradient boosted trees | Wind event classification | Use F1/precision/recall for rare wind-event judgment; do not rely on accuracy alone. |
| 10 | Morales-Hernandez et al., direct classification for wind ramp forecasting under imbalance | Ramp event classification | When events are imbalanced, state the event frequency or imbalance problem before presenting F1. |
| 11 | Sharp et al., wind ramp event prediction | Ramp event prediction | Define event thresholds in words or tables; avoid turning every threshold into a numbered formula. |
| 12 | Huang and Jafari, risk-aware BLSTM-MPC vehicle planning | Prediction-risk-assisted planning | Prediction output should become a planning quantity, such as conflict risk or admission result, before it enters a controller. |
| 13 | Zheng et al., lane-change MPC with LSTM trajectory prediction | Trajectory prediction and MPC | The prediction module can be described by input-output variables when the controller is the paper's real contribution. |
| 14 | Satir et al., NMPC with LSTM target prediction | Prediction-assisted guidance | Put the emphasis on how predicted trajectory changes the control objective, not on model architecture. |
| 15 | Vo et al., ANN-based adaptive NMPC | Learning-assisted NMPC | Learned models in control papers are often justified by the variables they provide to the controller. |
| 16 | Bahwal et al., forecast and MPC of DER aggregators | Forecast-assisted scheduling | Forecast baselines belong in validation; the method section should stay focused on the scheduling/control mapping. |
| 17 | Hannula et al., Bayesian LSTM for heating MPC | Uncertainty-aware prediction control | If uncertainty or probability appears, write how the controller becomes more cautious, not just the probability formula. |
| 18 | Saviolo et al., PI-TCN for quadrotor MPC | Physics-informed prediction control | Physical consistency matters only when it changes the control model; otherwise keep it out of main formulas. |
| 19 | Wang et al., PI-WAN for wind-adaptive quadrotor prediction | Wind-adaptive prediction | Wind-vector variables should be tied to the actual force or motion channel they influence. |
| 20 | Jiang and Dong, ModNN vs LSTM for building control | Control-oriented prediction | Prediction accuracy alone is insufficient; control-oriented papers must show downstream cost or constraint effects. |
| 21 | Jonkman, Dynamics Modeling and Loads Analysis of an Offshore Floating Wind Turbine | FOWT dynamics | Start model sections from degrees of freedom and load sources, then expand only the terms central to the paper. |
| 22 | Jonkman, Dynamics of Offshore Floating Wind Turbines | FOWT model verification | Separate engineering model verification from algorithm comparison; do not overclaim absolute platform fidelity. |
| 23 | Jonkman, OC3-Hywind floating system definition | FOWT benchmark definition | Put platform parameters and geometry in tables; reserve equations for relationships that explain motion or control. |
| 24 | Robertson et al., OC4 semisubmersible floating system definition | Semisubmersible FOWT | A benchmark paper can be parameter-heavy; an algorithm paper should only quote parameters needed for reproducibility. |
| 25 | OpenFAST HydroDyn theory manual | Hydrodynamic modeling | Complex hydrodynamic submodels are best summarized by load categories unless they are the paper's contribution. |
| 26 | Duarte et al., wave-radiation force realization within FAST | Radiation-force modeling | Use module-level descriptions for complex physics; do not translate every internal computation into equations. |
| 27 | Lemmer et al., semisubmersible hull shape design | FOWT low-order modeling | Clarify the role of a simplified model before showing equations, especially when results rely on relative comparison. |
| 28 | Sarker et al., hydrodynamic modeling improvements for FOWTs | Hydrodynamic validation | A formula should appear where a modeling improvement occurs; otherwise a concise load term is enough. |
| 29 | Sakif et al., Morison equation with frequency-dependent coefficients | Hydrodynamic coefficients | Do not introduce coefficient complexity without parameter source and validation data. |
| 30 | Cordle and Jonkman, state of the art in FOWT design tools | FOWT simulation tools | Use tool capability and model scope to frame simulation credibility; avoid claiming high-fidelity behavior from a reduced model. |
| 31 | Skaare et al., Hywind Demo measurements and simulations | Full-scale FOWT measurements | Measured-vs-simulated comparisons should mention uncertainty and exceptions, not only agreement. |
| 32 | Thiagarajan and Dagher, review of floating platform concepts | Floating platform review | Background should group platform/control ideas by function, not list every prototype chronologically. |
| 33 | Capaldo and Mella, damping analysis of FOWT | FOWT platform damping control | Mechanism sections work well when they connect equations to platform pitch motion and fatigue metrics. |
| 34 | Liu et al., fault-tolerant individual pitch control of FOWTs | FOWT predictive repetitive control | Control papers report both control input and structural/load response; actuator burden cannot be hidden. |
| 35 | Liu et al., periodic load rejection for FOWTs | Constrained predictive repetitive control | Constraint satisfaction belongs beside performance improvement in result sections. |
| 36 | Liu et al., fast adaptive fault accommodation for FOWTs | Fault diagnosis and control | Representative cases explain process timing; statistical or multi-case evidence carries the conclusion. |
| 37 | Kheirabadi and Nagamune, dynamic parametric wind farm model | Time-varying wind and platform motion | Simulator papers validate components; algorithm papers should use simulators to support controlled comparisons. |
| 38 | Kleine et al., stability of wakes of floating wind turbines | FOWT wakes and motion | If coupled motion affects downstream quantities, use mechanism diagrams or grouped panels rather than isolated curves. |
| 39 | Stadtmann et al., FOWT digital twin | Digital twin and prediction | Prediction capability should be framed by its operational use: diagnosis, prediction, or prescriptive action. |
| 40 | Ribeiro et al., FLOATBench | FOWT benchmark and surrogate evaluation | Split evaluation by regime when generalization is claimed; otherwise keep the scope bounded to selected cases. |
| 41 | Didier et al., control strategies for floating wind turbines | FOWT control review | Results should report platform response, power/load implications, and control effort as a connected set. |
| 42 | Zhang et al., turbulent wind thrust control for floating platforms | Wind-thrust control | Show physical effect channels, such as pitch response under turbulent wind, before claiming control benefit. |
| 43 | Sundarrajan et al., open-loop control co-design of semisubmersible FOWTs | Control co-design | Co-design papers make tradeoffs explicit; adopt this habit for pump saving versus posture margin. |
| 44 | Bayat et al., nested control co-design of spar-buoy FOWT | Control co-design | Separate design variables, control variables, and evaluation outputs; do not let them blur in prose. |
| 45 | Mulders et al., quasi-LPV MPC for tower frequency excitation | Wind turbine MPC | Constraints and resonance avoidance need their own evidence, not a sentence buried after a performance table. |
| 46 | Frederik et al., wake mixing with dynamic individual pitch control | Wind farm control | Extra actuator activity must be reported alongside output improvement, otherwise the result looks incomplete. |
| 47 | Mark and Liu, distributionally robust MPC for wind farms | Wind farm robust MPC | Uncertainty-aware control papers state assumptions before results and keep robustness claims bounded. |
| 48 | Yekun et al., hybrid LSSVM-SVMD-LSTM wind speed forecasting | Hybrid wind forecasting | Forecasting papers often compare many baselines; in this paper, such comparisons should stay short and serve the control narrative. |
| 49 | Huang, attention-gated recurrent network with error correction | Short-term wind speed forecasting | Error-correction models show prediction gains by horizon/case; avoid borrowing that depth unless prediction becomes a contribution. |
| 50 | Ehsan et al., wind speed prediction and visualization using LSTM | LSTM forecasting visualization | A single predicted-vs-observed curve is intuitive but weak as primary evidence; pair it with aggregate metrics. |
| 51 | Perumpalot et al., cross-location wind speed forecasting | Cross-location prediction | Generalization claims need split logic; if this paper uses one data source, keep claims local to the test set. |
| 52 | FLOATBench dataset and benchmark for FOWT fatigue | FOWT benchmark evaluation | Benchmark papers use clear protocol levels; for this paper, state the selected-case protocol and do not imply all-regime performance. |
| 53 | Ampleman and Gayme, multi-region FOWF control | Wind farm control | Cross-condition control papers separate operating regions; this paper can separate only when data support that split. |
| 54 | Fernandez Bravo et al., surrogate-based co-design coupling analysis | FOWT coupling analysis | Coupling matrices and heatmaps should answer a coupling question; do not add them as decoration. |
| 55 | The local group papers on ship motion prediction and tow-body tracking | Chinese engineering writing | Chinese method sections value compact equations, clear variable definitions, and figure/table-anchored result paragraphs. |

## Condensed Writing Rules From The 50+ Paper Pass

### 1. Section openings should state the claim, not announce the topic

Weak:

```text
本节对短时风况预测结果进行分析。
```

Better:

```text
短时风况预测结果为后续目标水量更新提供可量化输入。
```

The second sentence tells a reader why the section exists. Use this style for Chapter 3 and the conclusion.

### 2. Prediction modules must be judged by downstream use

For this paper, do not write the prediction result section as if it were a wind-forecasting paper. The right sequence is:

1. Report wind-vector and wind-direction errors.
2. Report wind-condition event judgment metrics.
3. Explain that these outputs support prediction-information admission and target-water updating.
4. Move quickly to active-ballast results.

Do not add long LSTM architecture explanations in Chapter 3.

### 3. Result chapters should move from total evidence to distribution to mechanism

Use this order:

1. Overall table: total cumulative pump volume, start-stop count, attitude-threshold exposure.
2. Distribution figure: per-case pump reduction, start-stop change, posture statistic change.
3. Tradeoff figure: pump reduction versus posture-side change.
4. Representative case: explain how the process happens.

Never start a result chapter with a single attractive time series.

### 4. Tables carry exact numbers; figures carry patterns

Put exact values such as `146843.96 m³`, `104193.99 m³`, `29.04%`, and `32.17%` in a table. Use figures for distributions, tradeoffs, and time processes. Avoid simple bar charts when a distribution or tradeoff is available.

### 5. Every benefit claim needs a paired cost or boundary metric

If a paragraph says the prediction-assisted algorithm reduces cumulative pump volume, the same paragraph or the next one must mention posture-threshold exposure, p95 posture change, start-stop count, or water-pump action burden. This avoids the appearance that savings are obtained by ignoring posture.

### 6. Use bounded conclusion language

Allowed:

```text
在170组预测可作用工况中，预测辅助算法降低了压载执行代价。
```

Not allowed:

```text
预测辅助算法在所有海况下均能降低泵量。
```

The current evidence supports selected real-wind cases with prediction-actionable characteristics, not all possible offshore conditions.

### 7. Avoid defensive prose

Do not write:

```text
本文不作为完整六自由度水动力模型的工程级验证。
```

Write the positive scope instead:

```text
平台运动仿真模型用于在相同外部风况和执行约束下比较两种目标水量生成方法。
```

If a limitation must be stated, put it in the final discussion or conclusion.

### 8. Replace code-like actions with physical/control actions

Use:

- 整理、映射、判别、更新、筛选、执行、反馈
- 姿态调节需求、目标水量、实际舱内水量、累计泵量、水泵启停次数

Avoid:

- reshape、wrap、clip、profile、casebook、gated、pipeline object、strategy version

### 9. Formula writing must pass the “does it move the method?” test

Keep equations that move the reader from one physical/control quantity to another:

- predicted wind vector to equivalent posture demand;
- posture feedback to feedback demand;
- combined demand to candidate action;
- ballast mass to restoring moment;
- pump flow to water-volume update;
- pump flow to cumulative pump volume.

Do not number equations that only define a vector, recover a scalar, or list a threshold.

### 10. Figure captions should state the conclusion

Weak:

```text
图3 累计泵量降低率分布。
```

Better:

```text
图3 170组预测可作用工况中的压载执行代价分布。累计泵量降低率整体位于正值区间，说明预测辅助算法的执行代价降低并非由少数工况拉动。
```

Use the second style unless the journal format forces very short captions.

## Chapter-Specific Drafting Rules For The Current Manuscript

### Chapter 1: method

- Begin with the ballast-control bottleneck, not the neural network.
- Keep LSTM formulas limited to the outputs needed by the controller.
- Define prediction-information admission in prose and tables unless its result enters a control formula.
- Keep target-water generation continuous from feedback demand to candidate action to pump execution.

### Chapter 2: validation setup

- State the baseline and proposed algorithm once, then use their short names.
- Explain FINO1 and 10 min wind data as the source of real wind-condition sequences.
- Define 预测可作用工况 positively: future wind tendency can participate in current target-water generation and the process is not dominated by posture-boundary protection.
- Mention the 33.77% proportion and 170 cases as validation scope, not as all-regime representativeness.

### Chapter 3: results

- 3.1 should first validate short-term wind prediction outputs with a compact table and one figure.
- 3.2 should present overall active-ballast results and per-case distributions.
- 3.3 should explain process mechanisms through target water, actual water, pump action, and posture response.
- Do not reintroduce old 100/101-case evidence unless the advisor explicitly asks for supplementary analysis.

### Conclusion

- Use 3 numbered conclusions.
- First conclusion: cumulative pump volume and start-stop reduction.
- Second conclusion: posture-side changes and bounded attitude exposure.
- Third conclusion: prediction information changes target-water generation and pump execution.
- Final paragraph: mention wider validation under more complete hydrodynamic, wave, and mooring conditions as future work.
