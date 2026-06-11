# Validation Sample Size Support

Last updated: 2026-06-06

Purpose: record the paper-facing sample-size and validation-duration argument
for the short-term wind-prediction-supervised active-ballast study. This note
supports the manuscript methods and reviewer-response language. It does not
change model training, controller behavior, or existing experiment outputs.

## 1. Evidence Levels

The paper uses three different denominators. They should not be mixed.

| level | role | current source | interpretation |
|---|---|---|---|
| raw real wind record | meteorological data basis | FINO1 fused 10 min wind record | long historical wind environment used to build the prediction dataset |
| ML train/validation/test samples | forecast-model evaluation | H240/F120 sequence dataset | overlapping 10 min sequence samples for wind prediction and risk classification |
| closed-loop control windows | ballast-control validation | fixed 6 h casebooks in held-out period | expensive controller replays used for pump and attitude metrics |

The control result should therefore be written as a closed-loop validation over
predeclared 6 h windows sampled from a held-out test-period wind record, not as
a random evaluation over all 20 calendar years.

## 2. Raw FINO1 Data Basis

The FINO1 processed record contains:

| item | value |
|---|---:|
| source | `BSH_FINO1::FINO1_Platform::fused_102m_speed_91m_dir_v1` |
| time range | 2005-01-01 to 2025-01-01 23:50 |
| rows | 978,280 |
| resolution | 10 min |
| valid-time equivalent | 163,046.7 h, about 18.60 years |

Source: `data/processed/wind_ml_10min/fino1_platform_10min/dataset_summary.csv`.

This supports the statement that the prediction and case-mining pipeline is
grounded in an approximately 20-calendar-year offshore-platform wind record,
with about 18.6 effective years of valid 10 min observations.

## 3. Prediction Dataset Split

The paper-facing forecasting dataset is:

```text
history_240m_future_120m_decision
history = 24 x 10 min = 240 min
future = 12 x 10 min = 120 min
```

The chronological split is:

| split | sequence samples | 10 min equivalent h | year equivalent | time span |
|---|---:|---:|---:|---|
| train | 638,631 | 106,438.5 | 12.14 | 2005-01-01 to 2018-11-01 |
| validation | 133,949 | 22,324.8 | 2.55 | 2018-11-01 to 2021-10-04 |
| test | 140,173 | 23,362.2 | 2.67 | 2021-10-04 to 2025-01-01 |
| validation + test | 274,122 | 45,687.0 | 5.21 | 2018-11-01 to 2025-01-01 |

Sources:

- `data/processed/wind_ml_10min/ballast_decision_fino1_meteo_aux_h240_f120_v1/dataset_summary.csv`
- `data/processed/wind_ml_10min/ballast_decision_fino1_meteo_aux_h240_f120_v1/sample_index.csv.gz`

Important wording: these are overlapping sequence samples, so the 10 min
equivalent hours are a scale reference rather than a count of independent
meteorological events.

## 4. Closed-Loop Validation Pool

The current broad closed-loop evidence combines two fixed 6 h pools:

| pool | role | cases |
|---|---|---:|
| `selector_mixed_6h_limit20_v1` | mixed validation: positive, warning/boundary, background | 101 |
| `selector_positive_add40_6h_v1` | positive-regime expansion | 120 |
| combined unique windows | all current 6 h closed-loop windows | 221 |

The `101`-case mixed pool contains:

| role | cases | purpose |
|---|---:|---|
| positive allow | 50 | test whether saving is available in forecast-actionable regimes |
| negative abstain | 31 | test warning/boundary cases where saving should be disabled |
| background abstain | 20 | test low-opportunity false activation |

The `120`-case expansion is positive-only and adds 40 cases each for P2, C3,
and W1-like positive regimes. It does not replace the 101-case mixed pool. It
extends the positive evidence from 50 to 170 cases:

| stratum | old positive cases | added positive cases | combined positive cases |
|---|---:|---:|---:|
| P2 neutral/headroom | 20 | 40 | 60 |
| C3 gusty/oscillatory | 20 | 40 | 60 |
| W1 stable-direction event | 10 | 40 | 50 |
| total | 50 | 120 | 170 |

Total closed-loop duration:

| count | horizon per case | total h |
|---:|---:|---:|
| 220 cases | 6 h | 1320 h |
| 221 cases | 6 h | 1326 h |

The repository currently contains 221 unique timestamps across the two pools.
They cover 176 unique dates, 36 unique months, and years 2021-2025. There are
28 adjacent pairs with start gaps below 6 h, so the paper should call them
"221 predeclared 6 h evaluation windows" rather than "221 fully independent
weather events".

Sources:

- `outputs/wind_prediction/selector_mixed_6h_limit20_v1/selector_gated_summary_101case_6h/selector_gated_summary.md`
- `outputs/wind_prediction/selector_mixed_6h_limit20_v1/selector_gated_summary_101case_6h/selector_gated_extended_readout.md`
- `outputs/wind_prediction/selector_positive_add40_6h_v1/combined_positive_summary_170case_6h/positive_combined_readout.md`
- `outputs/wind_prediction/selector_mixed_6h_limit20_v1/casebooks/selector_mixed_limit20_6h_cases.csv`
- `outputs/wind_prediction/selector_positive_add40_6h_v1/casebooks/positive_add40_6h_cases.csv`

## 5. Ratio Against ML Validation and Test Sets

Using 1320-1326 h as the closed-loop control-validation duration:

| denominator | denominator h | 1320 h ratio | 1326 h ratio |
|---|---:|---:|---:|
| raw FINO1 valid record | 163,046.7 | 0.81% | 0.81% |
| train split | 106,438.5 | 1.24% | 1.25% |
| validation split | 22,324.8 | 5.91% | 5.94% |
| test split | 23,362.2 | 5.65% | 5.68% |
| validation + test | 45,687.0 | 2.89% | 2.90% |
| all sequence samples | 152,125.5 | 0.87% | 0.87% |

Interpretation:

- The closed-loop replay is much smaller than the full 20-year raw record,
  because full closed-loop active-ballast simulation is the expensive evaluation
  layer.
- It is still a substantial subset of the held-out prediction period: roughly
  5.7% of the test split by 10 min time-equivalent scale.
- Since the windows are selected for regime coverage and boundary behavior,
  representativeness should be argued through stratification and predeclaration,
  not through simple random-sample language.

## 6. Literature Positioning

The current 1320-1326 h closed-loop validation is large for a control-oriented
FOWT paper, while still below certification or deployment-scale validation.

Relevant comparison points:

| reference | validation practice | relevance to this paper |
|---|---|---|
| Fontanella et al., 2021, *Wind Energy Science*, wave-feedforward FOWT control | For each listed condition, six independent 10 min wind-wave realizations were used, with an initial 1000 s pre-simulation removed from the results. | Shows that FOWT control papers often use condition-wise 10 min realizations when the claim is controller behavior rather than full deployment certification. |
| Guo and Schlipf, 2023, *Wind Energy Science*, lidar-assisted feedforward and MVFB control | Evaluates lidar-assisted and multivariable feedback controls for the IEA 15 MW reference FOWT using OpenFAST and IEC DLC-style assessment. | Shows that modern preview/feedforward FOWT studies emphasize structured load-case assessment and controller comparison, not a single long continuous deployment trace. |
| Haid et al. / NREL, 2013, simulation-length requirements for FOWT load analysis | Examines FOWT simulation length because 10 min may be too short for floating wind-wave response; reports that simulation length itself was not the dominant factor for the studied spar loads statistics. | Supports using a broad set of 6 h windows for episode-level control evidence, while warning against overclaiming certification-level fatigue or global load conclusions. |
| ABS, 2023, *Guidance Notes on Global Performance and Integrated Load Analysis for Offshore Wind Turbines* | Lists simulation settings, stochastic seeds, duration, transient time, wind/wave/current modeling, and control-system settings as important for integrated load analysis. | Clarifies that certification/global-performance validation is a stronger task than this paper's short-horizon controller evidence. |

Public sources:

- Fontanella et al. (2021), WES: https://wes.copernicus.org/articles/6/885/2021/
- Guo and Schlipf (2023), WES: https://wes.copernicus.org/articles/8/1299/2023/
- Stewart et al. / NREL conference paper: https://docs.nrel.gov/docs/fy13osti/58518.pdf
- ABS Guidance Notes (2023): https://ww2.eagle.org/content/dam/eagle/rules-and-guides/current/design_and_analysis/206-guidance-notes-on-global-performance-and-integrated-load-analysis-for-offshore-wind-turbines/206-fowt-gpa-gn-aug23.pdf

## 7. Paper-Safe Writing

Safe manuscript language:

```text
基于约 20 年 FINO1 平台 10 min 实测风速/风向数据，本文构建了
240 min 历史输入、120 min 未来输出的短时风况预测数据集。预测模型采用
时间顺序划分，训练集、验证集和测试集分别覆盖 2005-2018、2018-2021
和 2021-2025 年时段。闭环控制验证进一步从测试期构建 221 个预声明
6 h 评价窗口，总计 1326 h，用于比较无预测闭环反馈策略与预测监督调节
策略在泵耗、姿态暴露和安全回退方面的差异。
```

Safe interpretation:

```text
该验证规模显著超过仅用于机制展示的 10-case 测试，能够支撑短时段、
工况分层、episode-level 的预测监督控制结论。但该 1326 h 评价窗口仍不
等价于 20 年全生命周期部署验证，也不构成认证级载荷或疲劳结论。
```

Avoid:

```text
本文已经完成 20 年闭环部署验证。
```

Avoid:

```text
221 个样本全部是相互独立的极端海况事件。
```

Avoid:

```text
1326 h 验证证明所有工况下长期稳定节泵。
```

## 8. Recommended Reporting Table

For the final manuscript or appendix, report the sample-size context as:

| item | value |
|---|---:|
| FINO1 raw record | 978,280 ten-minute rows, about 18.6 effective years |
| prediction train samples | 638,631 |
| prediction validation samples | 133,949 |
| prediction test samples | 140,173 |
| closed-loop mixed 6 h windows | 101 |
| positive-regime expansion windows | 120 |
| total closed-loop windows | 221 |
| total closed-loop validation duration | 1326 h |
| closed-loop duration / test split | about 5.7% by 10 min equivalent time |

One concise final-reading sentence:

> 当前样本量对于短论文的短时预测监督控制结论是足够的；写作重点应放在
> 时间顺序 split、预声明 6 h 窗口、P2/C3/W1/预警/背景工况分层，以及
> 不把 1326 h 闭环验证外推成 20 年部署级结论。
