# Closed Baseline v1 Positioning

Date: 2026-05-05

## Purpose

`closed_baseline_v1` is frozen as the ordinary non-preview active-ballast baseline for comparison with the prediction-primary controller. Its role is not to be a vendor-grade optimized controller, and not to be intentionally weak. It is a transparent reactive baseline:

- PI/MIMO feedback from measured platform attitude/state.
- Pump stop/restart hysteresis.
- Minimum pump on/off dwell.
- Near-target stop confirmation dwell.
- Finite pump-rate ramp.
- No future wind, planner, pressure-block preview, bucket guard, or hold-current prediction target.

The frozen config is [configs/closed_baseline_v1.json](/Users/saintyoung/Desktop/FOWT-CONTROLLER-main/configs/closed_baseline_v1.json).

## Freeze Evidence

The validation script is [scripts/analysis/run_closed_baseline_freeze.py](/Users/saintyoung/Desktop/FOWT-CONTROLLER-main/scripts/analysis/run_closed_baseline_freeze.py).

The evidence report is [outputs/wind_prediction/closed_baseline_freeze_v2/closed_baseline_freeze_report.md](/Users/saintyoung/Desktop/FOWT-CONTROLLER-main/outputs/wind_prediction/closed_baseline_freeze_v2/closed_baseline_freeze_report.md).

Five two-hour mechanism windows were used:

- onset ramp
- decay ramp-down
- sustained high wind
- normal low-risk wind
- high-pressure direction change

Result:

- `closed_engineered_minimal` vs `closed_raw`: pump work changed by at most 0.18%, pitch/roll p95 changed by at most about 0.003/0.002 deg, while pump latch switches were reduced by 51%-74% in the active cases.
- `closed_smooth_candidate` vs `closed_engineered_minimal`: failed because pump work increased by up to 41.67% and latch reduction was not consistent.

Decision:

```text
Freeze closed_engineered_minimal as closed_baseline_v1.
Do not use the rejected smooth candidate as the main baseline.
```

## Literature Position

Commercial wind-turbine and floating-system controllers are usually not available as transparent algorithms with reusable parameters. In validation work, researchers commonly use reference controllers or self-defined PI/PID/gain-scheduled controllers that are documented and reproducible.

Useful anchors:

- ROSCO is explicitly presented as a reference open-source controller for fixed and floating offshore wind turbines. The paper states that reference controllers are widely used as baselines for advanced control studies, and that ROSCO represents standard industry practices while remaining open and tunable: https://wes.copernicus.org/articles/7/53/2022/
- FOWT MPC work commonly compares advanced controllers against gain-scheduled PI/reference feedback baselines rather than proprietary commercial controllers. Example: a 2025 SMPC study compares against gain-scheduled PI, LQR, and deterministic MPC: https://www.sciencedirect.com/science/article/pii/S0029801825021158
- A 2024 FOWT NMPC study evaluates against feedback-based control strategies rather than using a vendor black box: https://www.sciencedirect.com/science/article/pii/S0029801824030920
- A 2026 active-ballast FOWT paper uses a Python PID controller with pump-flow constraints for pitch/roll active ballast, which is very close in spirit to this project's frozen reactive ballast baseline: https://www.sciencedirect.com/science/article/pii/S0029801825028252
- Public comparison platforms exist for model validation and reproducibility, especially the IEA 15 MW reference turbine and UMaine VolturnUS-S semisubmersible platform: https://research-hub.nrel.gov/en/publications/iea-wind-tcp-task-37-definition-of-the-iea-15-megawatt-offshore-r and https://research-hub.nrel.gov/en/publications/definition-of-the-umaine-volturnus-s-reference-platform-developed/
- One-to-one field validation papers note that proprietary turbine controllers can be complex and are often approximated by tuned open-source controllers when the proprietary DLL/controller is not permitted: https://wes.copernicus.org/articles/9/1791/2024/

## Paper Wording Boundary

Safe wording:

```text
The proposed preview/prediction-primary ballast planner is compared against a frozen non-preview reactive active-ballast baseline. The baseline is a transparent PI/MIMO feedback controller with ordinary pump actuator constraints. It does not use forecast information or planner-level preview logic.
```

Avoid wording:

```text
The baseline reproduces a commercial OEM controller.
```

Also avoid:

```text
The prediction model alone provides all pump-work savings.
```

The current evidence supports this contribution boundary:

```text
The main contribution is the predictive ballast-control framework and its ability to use future wind/load information before large attitude errors fully develop. Forecast-source ablations remain auxiliary evidence, not the main claim.
```
