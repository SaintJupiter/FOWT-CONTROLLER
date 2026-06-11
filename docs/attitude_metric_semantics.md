# Attitude metric semantics

Last updated: 2026-06-05

This note freezes the shared vocabulary and acceptance reading used by
attitude-exposure summaries. It is a metric contract only; it does not change
controller logic.

## Exposure Signal

All standard exposure metrics use the max-axis attitude:

```text
attitude_exposure_deg = max(abs(pitch_deg), abs(roll_deg))
```

Pitch and roll are filtered together. Rows are kept only when both axes are
finite, so the two axes never drift out of alignment during metric calculation.

If a timeseries has `t_s`, the metric uses it to infer the sample interval from
the median positive adjacent time difference. If `t_s` is missing or invalid,
the fallback interval is 1 s.

## Windows

The shared windows are:

| window prefix | meaning |
|---|---|
| empty prefix | whole case |
| `startup_0_300s_` | first 300 s from the first valid sample |
| `steady_after_300s_` | samples at or after 300 s |

## Thresholds

| threshold | label | meaning | control use |
|---:|---|---|---|
| `3 deg` | `comfort_outside` | comfort/economy band occupancy | low-posture tracking cost accounting |
| `4 deg` | `near_service_pressure` | service-margin occupancy | economy caution and Pareto cost accounting |
| `5 deg` | `service_pressure` | service-pressure occupancy, not a failure line | duration/area budget and release-veto audit |
| `7.5 deg` | `severe_pressure` | strong posture-pressure occupancy | strong warning if sustained or worsening |
| `10 deg` | `operating_limit` | operating-limit occupancy | hard-protection review if sustained or increased |

## Acceptance Reading

The thresholds above are reporting bands, not single-point pass/fail limits.
Floating offshore structures can tolerate finite pitch and roll response, and
active ballast control is a trade-off between posture regulation, pump work, and
actuator wear. Therefore, `time_over_5deg_s` or a small positive
`d_time_over_5deg_s` must not be interpreted as a failure by itself.

Use the bands as a layered reading:

| layer | primary evidence | interpretation |
|---|---|---|
| hard failure | crash/NaN, large fallback increase, sustained or increased `>10 deg`, or pump work and posture both worse than baseline | reject as broken or dominated |
| strong warning | increased `>7.5 deg` time/area/continuous run, especially with fallback or worsening p95/max posture | not a clean headline; require case-level explanation or a guard |
| service-pressure cost | increased `>5 deg` time/area/continuous run or p95 posture cost | allowable as a Pareto cost only when pump/latch benefit is material |
| comfort/economy cost | increased `>3 deg` or `>4 deg` occupancy | expected for tolerance-band pump-saving modes; report but do not reject automatically |

For pump-saving claims, compare against the matched closed baseline and report
at least: pump work, p95 pitch/roll or max-axis p95, `time/max_continuous/area`
for `5/7.5/10 deg`, safety fallback, and latch switches. A controller is
dominated when it saves no pump, adds posture cost, and increases fallback or
target/latch debt. A controller can remain acceptable with more `3/4/5 deg`
occupancy when it materially reduces pump work or latch switching without
growing the `7.5/10 deg` tail or fallback.

## Metric Names

For each window and threshold, the shared module emits:

| metric | unit | definition |
|---|---:|---|
| `time_over_{threshold}_s` | s | total sampled time where exposure is strictly greater than the threshold |
| `max_continuous_over_{threshold}_s` | s | longest continuous run above the threshold |
| `area_over_{threshold}_deg_s` | deg s | sum of `(exposure - threshold)` while above the threshold |

Example casebook columns:

```text
primary_time_over_5deg_s
closed_time_over_5deg_s
d_time_over_5deg_s
primary_steady_after_300s_area_over_5deg_deg_s
```

Use these columns for new analysis instead of ad hoc names such as
`time_over_5`. If a legacy script still has those names, treat it as local
backward compatibility and migrate it to `src/wind_prediction/attitude_metrics.py`
when the script is next touched.

## Control Guidance

The exposure metrics are not a new controller layer. They are the reporting
contract for comparing controller behavior.

Use the existing `primary_safety_fallback` parameters for hard protection:

| purpose | preferred mechanism |
|---|---|
| service-pressure accounting around 5 deg | metrics, Pareto tables, and report columns |
| severe exposure around 7.5 deg | metrics plus `primary_safety_profile` tuning when sustained or worsening |
| operating-limit exposure around 10 deg | metrics plus emergency fallback tuning when sustained or increased |
| active target stuck near the fallback band | opt-in active-target release guard |

Do not add another hard-fallback layer just to track isolated 5 deg or 7.5 deg
events. If 7.5 or 10 deg tails are sustained or increasing, tune the existing
safety fallback profile first. Keep active-target release guards opt-in unless
a case shows that angle thresholds alone cannot represent the problem.
