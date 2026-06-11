# Prediction-primary 10-case validation notes

Date: 2026-05-05

## Purpose

This run is not a broad blind scan. It is a focused validation set for the small-paper story:

1. freeze a fair non-preview closed-loop baseline;
2. show the pump-saving potential of prediction-primary planning;
3. expose where the current planner is not yet robust;
4. avoid hiding planner limits behind extra guard tuning.

## Controller roles

`closed_baseline_v1` is the fair non-preview baseline. It uses reactive PI/MIMO feedback and normal pump actuator constraints, but it has no future wind, no preview planner, and no prediction-based guard. This is the controller to compare against when claiming the value of preview/prediction-based ballast planning.

`Naive primary` is the high-saving prediction-planning case. It shows the maximum current benefit of using future wind blocks to reduce unnecessary ballast pumping, but it should not be described as a final robust industrial controller.

`Hold-risk gate` is a conservative supervisor. It asks whether a simple transparent safety gate can rescue obvious high-attitude cases without changing the planner. It is useful as an engineering check, but the 10-case result shows that it is not a universal final solution.

## 10-case outputs

Simulation outputs:

- `outputs/wind_prediction/prediction_primary_baseline_v1_10case_2h_naive`
- `outputs/wind_prediction/prediction_primary_baseline_v1_10case_2h_hold_risk_gate`
- `outputs/wind_prediction/reactive_closed_pump_saving_space_v1`
- `outputs/wind_prediction/closed_economy_pareto_v1`

Figure pack:

- `outputs/wind_prediction/paper_figures_prediction_primary_10case_v1/fig_10case_tradeoff_map.png`
- `outputs/wind_prediction/paper_figures_prediction_primary_10case_v1/fig_10case_metric_heatmap.png`
- `outputs/wind_prediction/paper_figures_prediction_primary_10case_v1/fig_case09_supervisor_mechanism.png`
- `outputs/wind_prediction/paper_figures_prediction_primary_10case_v1/fig_case04_boundary_mechanism.png`
- `outputs/wind_prediction/paper_figures_prediction_primary_10case_v1/prediction_primary_10case_summary.csv`
- `outputs/wind_prediction/paper_figures_prediction_primary_10case_v1/prediction_primary_10case_grouped_metrics.csv`

## Aggregate result

| run | mean pump saving | median pump saving | mean attitude penalty | mean fallback |
|---|---:|---:|---:|---:|
| Naive primary | 69.98% | 76.55% | 1.46 deg | 0.78% |
| Hold-risk gate | 27.45% | 20.77% | 1.00 deg | 14.98% |

Additional no-preview baseline checks:

| check | result | interpretation |
|---|---:|---|
| reactive-side low-risk pump-start proxy | 5.43% of closed pump work | many starts occur in low-risk periods, but they contribute little total pump work |
| strict actuator-only Pareto scan | best non-baseline saving 0.12% on calibration, selected candidate = baseline | ordinary no-preview pump retuning did not find meaningful extra saving without attitude changes |

Interpretation:

- Naive primary demonstrates a strong prediction-planning energy advantage, but it carries attitude penalties in several hard wind-transition windows.
- Hold-risk gate improves the average attitude penalty and strongly improves case 09 roll behavior, but it erases savings or even increases pump work in fast sign-flip/decay cases.
- The no-preview economy checks reduce the concern that the 70% result is merely caused by an artificially pump-heavy closed baseline. Under the strict actuator-only search used here, the frozen closed baseline is already near the practical no-preview pump-economy limit.
- Therefore the main bottleneck is not the frozen closed baseline. The bottleneck is the planner's prior risk envelope: the planner must reject actions that look cheap in pump cost but lead to large attitude penalties under the real plant response.

## Case-level reading

Strong support cases:

- 01 onset: about 83% pump saving with moderate attitude penalty.
- 07 sustained sign-flip: about 76% pump saving with almost no attitude penalty.
- 10 residual-high: about 59% pump saving with small attitude penalty.
- 08 quiet: no false benefit and no unnecessary actuation.

Boundary cases:

- 04 decay: naive still saves 61%, but pitch p95 penalty is about 3.65 deg. Hold-risk reduces the penalty to about 2.12 deg but costs more pump than closed baseline. This means a posterior guard is not enough.
- 06 sign-flip: hold-risk reduces attitude penalty slightly but removes almost all pump saving. Again, this points to planner-level risk screening.
- 09 high-pressure event: hold-risk is useful here. Pump saving drops from about 91% to about 38%, but worst p95 attitude penalty drops from about 2.45 deg to about 0.43 deg.

## Figure style note

Recent FOWT control papers commonly combine controller architecture, multi-case metrics, and time-domain response rather than relying on a single improvement percentage. The generated figure pack follows that style:

- trade-off map: pump saving vs attitude penalty;
- heatmap matrix: case-by-case energy, attitude, and fallback burden;
- mechanism figures: wind, pitch, roll, pump rate, planner action, and fallback strip.

Useful public references for positioning and figure style:

- Abbas et al. (2022), ROSCO reference open-source controller, Wind Energy Science: https://wes.copernicus.org/articles/7/53/2022/
- Capaldo and Mella (2023), FOWT damping control strategy, Wind Energy Science: https://wes.copernicus.org/articles/8/1319/2023/
- Mahfouz et al. (2021), IEA 15 MW WindCrete and Activefloat response using re-tuned ROSCO controllers, Wind Energy Science: https://wes.copernicus.org/articles/6/867/2021/
- Recent Ocean Engineering active-ballast work also reports coupled FOWT/active-ballast simulations with platform motion, pump/ballast characteristics, load-case tables, and power/mooring statistics rather than only a single improvement percentage: https://www.sciencedirect.com/science/article/pii/S0029801825028252

## Paper-safe claim

Safe wording:

> The proposed preview-based ballast planner substantially reduces ballast pump activity in representative wind-transition windows when compared with a frozen non-preview feedback baseline. A transparent supervisor reduces some high-attitude boundary cases, but the remaining decay and direction-change cases indicate that robust deployment requires planner-level risk-envelope constraints rather than further tuning of the reactive baseline.

Stronger wording now supported by the no-preview economy check:

> A strictly actuator-level retuning of the non-preview baseline did not produce a meaningful pump-work reduction under the same attitude constraints, which suggests that the observed pump-work reduction is not simply an artifact of an overactive closed-loop baseline.

Avoid wording:

> The final controller is uniformly better than closed-loop control.

Avoid wording:

> The whole 70% pump saving is directly caused by the learned wind model.

At this stage, the 10-case result supports the prediction-planning framework and its pump-saving mechanism. It does not yet support a fully robust final industrial controller claim.
