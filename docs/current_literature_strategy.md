# 当前小论文文献策略记忆

Last updated: 2026-06-01

Purpose: this file records the current literature-use strategy for the FOWT active-ballast short paper. Future drafting should use this source hierarchy unless the user updates it.

## Core Judgment

The current literature chain is valid and not artificially assembled:

```text
浮式风机背景
-> 主动压载已有研究
-> 短时风速/风向预测
-> 预测/风险信号进入控制
-> 仿真评价与运维优化
```

The paper should not claim that all existing active-ballast systems are purely passive. A safer claim is:

> 公开文献和行业资料中的主动压载系统多以当前姿态、平均风载或预设工况为主要调节依据，短时风况预测与危险工况预警尚未作为压载监督调节的核心信息源得到充分展开。

## Important Correction

Replace the previous item:

```text
Valdivia-Bautista et al., 2023, The Potential of Machine Learning for Wind Speed and Direction Short-Term Forecasting: A Systematic Review
```

with:

```text
Alves et al., 2023, The Potential of Machine Learning for Wind Speed and Direction Short-Term Forecasting: A Systematic Review
```

Reason: the systematic review on wind-speed and wind-direction short-term forecasting / nowcasting is by Alves et al. Valdivia-Bautista et al. have a different review titled "Artificial Intelligence in Wind Speed Forecasting: A Review", which is useful but mainly wind-speed focused.

## Core Sources

Use these as the main academic support in the body text.

1. Roddier et al. (2010), "WindFloat: A floating foundation for offshore wind turbines"
   Use: introduction; floating semi-submersible foundation / WindFloat background.

2. Salic et al. (2019), "Control Strategies for Floating Offshore Wind Turbine: Challenges and Trends"
   Use: introduction; coupled wind-wave-load control challenges for FOWTs.

3. Meng et al. (2026), "Coupled analysis and performance evaluation of a semi-submersible floating wind turbine with active ballasting system"
   Use: Chapter 1 and Chapter 3; active ballast, PID posture feedback, pitch/roll, pump flow constraints, coupled simulation.

4. Mahfouz et al. (2021), "Response of the IEA Wind 15 MW WindCrete and Activefloat floating wind turbines to wind and second-order waves"
   Use: Chapter 1; Activefloat active ballast, average thrust / wind-speed operating-condition adjustment.

5. Stansby (2021), "Reduction of wave-induced pitch motion of a semi-sub wind platform by balancing heave excitation with pumping between floats"
   Use: Chapter 1; ballast water transfer can reduce platform motion response.

6. Wakui et al. (2021), "Stabilization of power output and platform motion of a floating offshore wind turbine-generator system using model predictive control based on previewed disturbances"
   Use: introduction and Chapter 1; previewed wind/wave disturbances entering FOWT control. This is a better bridge than general MPC-only sources for the paper's "prediction information enters control" logic.

7. Liu and Chen (2019), "Data processing strategies in wind energy forecasting models and applications: A comprehensive review"
   Use: Chapter 2.1; wind forecasting data preprocessing and feature construction.

8. Xie et al. (2021), "A Short-Term Wind Speed Forecasting Model Based on a Multi-Variable LSTM Network"
   Use: Chapter 2.2; multivariable LSTM for short-term wind-speed forecasting.

9. Fuentes-Barrios et al. (2022), "LSTM Model for Wind Speed and Power Generation Nowcasting"
   Use: Chapter 2.2; 10 min data and nowcasting, but note it is proceedings and should not carry the main theoretical weight.

10. Sari et al. (2021), "Deep convolutional long short-term memory network for forecasting wind speed and direction"
    Use: Chapter 2.1 and 2.2; wind-speed and wind-direction prediction.

11. Blazakis et al. (2025), "ML-Based Multi-Horizon Wind Speed and Wind Direction Forecasting for Aviation and Energy Applications in Coastal Crete"
    Use: Chapter 2.2; multi-horizon wind-speed/direction prediction and wind-direction sin/cos handling. Not a floating-offshore scenario, so use as method support.

12. Modé et al. (2025), "Short-term extreme wind speed forecasting using dual-output LSTM-based regression and classification model"
    Use: Chapter 2.2; regression + classification risk output, supporting hazardous wind-condition warning and risk-window design.

13. Alves et al. (2023), "The Potential of Machine Learning for Wind Speed and Direction Short-Term Forecasting: A Systematic Review"
    Use: Chapter 2; wind-speed/direction nowcasting review and gap statement that wind-direction or joint wind-speed/wind-direction nowcasting is underrepresented.

14. Gaertner et al. (2020), "Definition of the IEA Wind 15-Megawatt Offshore Reference Wind Turbine"
    Use: Chapter 3; reference wind turbine / simulation platform basis.

15. OpenFAST documentation
    Use: Chapter 3; coupled aero-hydro-servo-elastic simulation tool basis.

16. McMorland et al. (2022), "Operation and maintenance for floating wind turbines: A review"
    Use: introduction and Chapter 3; O&M relevance of reducing unnecessary actuation and improving warning/operation strategy.

## Secondary / Background Sources

Use these sparingly as background, not as the core academic evidence chain.

- Carbon Trust (2026), "Ballast Systems for Stability Control of Floating Platforms"
  Good for active-ballast system composition: pumps, valves, sensors, control system, real-time ballast adjustment. Treat as an industry report.

- Principle Power, WindFloat / Smart Hull Trim System
  Good for showing that commercial smart ballast / trim systems exist. Do not use as core theoretical evidence.

- Seaplace CROWN / LR Approval in Principle
  Good for industry background on automatic/semi-automatic ballast systems. Do not use as core theoretical evidence.

- Stockhouse et al. (2021), "Control of a Floating Wind Turbine on a Novel Actuated Platform"
  Relevant to variable ballast / actuated platform, but currently an arXiv preprint. Use only as related exploration if needed.

- Shah et al. (2021), "Platform motion minimization using model predictive control of a floating offshore wind turbine"
  Useful for general FOWT MPC and motion suppression, but it is not active ballast and not the direct method basis.

- Valdivia-Bautista et al. (2023), "Artificial Intelligence in Wind Speed Forecasting: A Review"
  Useful as a general wind-speed forecasting review, but not as the main source for wind-speed/wind-direction joint nowcasting.

## Section Mapping

```text
0 引言:
Roddier 2010; Salic 2019; Meng 2026; Carbon Trust 2026; McMorland 2022; Wakui 2021.

1 短时风况预测驱动的主动压载监督调节算法:
Meng 2026; Mahfouz 2021; Stansby 2021; Wakui 2021; Carbon Trust 2026.

2 短时风况预测与风险识别方法:
Liu & Chen 2019; Xie 2021; Fuentes-Barrios 2022; Sari 2021; Blazakis 2025; Modé 2025; Alves 2023.

3 主动压载监督调节仿真与性能分析:
Gaertner 2020; OpenFAST; Meng 2026; Wakui 2021; McMorland 2022.

4 结论:
Do not introduce new literature. Summarize the paper's own validated findings only.
```

## Writing Guardrails

- Do not call industry systems "passive". Write "当前状态反馈型" or "基于当前姿态/平均风载调节".
- Do not state that prediction directly outputs ballast commands. Write "预测结果被转化为趋势、风险窗口和危险预警信号，供监督层进行策略选择和约束".
- Do not overstate global pump-saving results before final numbers stabilize. Write "在预测识别的特定风况区间内".
- Use Wakui (2021) to support the general control idea of previewed disturbances, but still distinguish this paper's contribution: previewed wind-condition signals are connected to active-ballast supervisory regulation, not to a generic FOWT MPC controller.
