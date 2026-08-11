# Control Validation Protocol

Last updated: 2026-06-11

This note freezes the validation horizon rules for prediction-primary ballast
controller experiments. It is a screening contract, not a controller change.

For paper-facing validation and claim wording, also use
`docs/paper_validation_strategy_20260605.md`. The paper track and the
production-freeze track are deliberately different: the current short paper may
use broad 6 h evidence plus a transparent 12 h robustness check, while a
production-default change still requires longer lifecycle-debt evidence.

## Horizon Rules

1. Runs shorter than 6 h are debug probes only.
2. A single 6 h run is not evidence that a 12 h or 24 h continuous run is safe.
3. A broad or stratified 6 h set is the minimum paper-facing candidate screen.
4. A transparent 12 h mixed-regime check is the current robustness requirement
   for the short paper.
5. Continuous 24 h or broader lifecycle-debt runs are required before freezing
   a profile as a production-default candidate.

The reason is target lifecycle debt. A short run can show early pump saving
while hiding late catch-up pumping, stale target recovery, or latch debt. Splitting
a 12 h or 24 h day into independent 6 h runs is also not equivalent to one
continuous run, because controller state is reset at the segment boundary.

The current paper does not claim deployment-scale 24 h pump saving. Its main
claim is short-horizon, episode-level forecast-supervised ballast regulation.
Therefore, 24 h is not a hard paper gate unless the manuscript claim is upgraded
to production or deployment scale.

## Recommended Gate: Experiment Development

Use this order for P2/C3 economy or hold-current profiles:

| stage | purpose | minimum evidence |
|---|---|---|
| debug probe | inspect a mechanism quickly | any duration is allowed, but never used as acceptance evidence |
| single 6 h reproducer | confirm the symptom or a known fix on one case | one known-good or known-bad case |
| stratified 6 h screen | decide whether the candidate deserves 24 h time | 3-4 targeted cases covering positive, boundary, and background samples |
| limited 24 h confirmation | detect daily lifecycle debt | one hard/boundary 24 h case plus one expected-positive 24 h case |
| broader 24 h casebook | production/deployment evidence | only after the limited 24 h confirmation is stable |

## Required Gate: Protocol Identity And Smoothing Provenance

Before any result is plotted, summarized, or interpreted, the actual
`run_protocol.json` must be checked against the intended algorithm identity. A
run is invalid for a claim if the protocol does not match the expected
`identity.primary_control_profile`, `configuration.primary_pump_profile`,
forecast source, horizon, and feature flags.

This gate is mandatory for the smoothed/smooth180 route. If the claim or user
request says the smoothed controller, smooth180 robustness route, actuator
smoothing, ramp/dwell smoothing, or the current smoothed branch is being
analyzed, then a non-smooth `posture_debt` run must not be substituted. Missing
time-series output in the smoothed run is a blocker for line plots; it is not a
reason to use an older non-smooth run.

Use the standard checker before plotting:

```bash
./.venv312/bin/python scripts/analysis/verify_casebook_protocol.py \
  path/to/run_protocol.json \
  --expect identity.primary_control_profile=<expected_profile> \
  --expect configuration.primary_pump_profile=<expected_pump_profile> \
  --require-timeseries
```

If the protocol check fails, label the run as historical/diagnostic only and do
not use it to answer questions about the current algorithm.

## Required Gate: Prediction-Decision Changes

Prediction-facing controller work must not proceed by stacking one more
runtime patch or threshold until a decision-level audit has shown why the
forecast signal is needed. Fixed deadband or other low-level safety-threshold
changes are allowed only as baselines or guardrails; they are not evidence of
forecast decision value.

Use this sequence for any change whose claim is that prediction improves pump
start/stop timing, hold/release timing, or economy-window selection:

| stage | action | promotion criterion |
|---|---|---|
| diagnosis | split the already-run positive windows into bad, neutral, and good cases by paired attitude and pump deltas | the bad cases show an interpretable decision failure, not just a tunable low-level threshold |
| forecast separability audit | compare learned forecast features against current-only, persistence, and shuffled/no-future views on those cases | learned forecast must separate at least one actionable failure mode from benign saving windows |
| shadow policy | log what the proposed prediction decision would have done without changing target, pump, or safety behavior | shadow decisions must be sparse, explainable, and aligned with the audited failure mode |
| canary replay | run only a predeclared 12-20 case set covering positive, bad-tail, boundary, and background windows | reduce the audited failure metric without new `>10 deg` tail, broad pump growth, or background false activation |
| ablation replay | rerun the same canary with learned, current-only, persistence, and shuffled/no-future inputs | learned must outperform or be more selective than non-forecast inputs under the same frozen controller parameters |
| final casebook | only after the canary and ablation pass, run the predeclared positive 170 or broader paper casebook | report every case, with learned-vs-ablation attribution and no post-hoc case removal |

Hard constraints:

1. Freeze low-level control parameters before the shadow-policy stage. Do not
   change PID deadbands, target-release deadbands, safety floors, pump-rate
   limits, or fallback thresholds while claiming forecast-decision value.
2. Do not add a new gate directly into the production path until it has passed
   as a shadow policy on the same windows.
3. Do not promote a result if the explanation is "the threshold was tuned until
   the aggregate improved." The explanation must name the forecast condition,
   the affected decision timing, and the cases where the decision changed.
4. Do not use a full 170-case or 221-case run as the first test of an unknown
   decision rule. Full runs are confirmation, not exploration.
5. Keep fixed conservative-deadband runs as baselines. They may show the
   pump/attitude trade-off floor, but they must not be written as the algorithmic
   contribution.
6. Do not write a paper-facing numeric result before the corresponding output
   directory, summary table, and run protocol exist. Partial runs must be labeled
   as partial/probe evidence in the draft, claim map, and any revision note.

The intended claim shape is:

```text
Under the same frozen low-level safety constraints, the forecast-supervised
decision policy changes pump hold/release timing in forecast-actionable windows,
preserving pump saving while reducing the audited attitude-exposure failure mode
relative to current-only, persistence, or shuffled/no-future controls.
```

## Recommended Gate: Current Short Paper

Use this evidence ladder for the current 5000-word engineering short paper:

| stage | purpose | minimum evidence |
|---|---|---|
| broad 6 h main casebook | headline episode-level evidence | predeclared cases/windows, current-only comparison, all cases reported |
| 12 h mixed-regime guard check | robustness and claim-boundary disclosure | guard10 or comparable mixed set with pump and attitude exposure table |
| 24 h check | optional only if the paper claims deployment-scale benefit | not required for the current short-horizon claim |

Do not replace the broad 6 h main evidence with a few hand-picked positive
examples. If 6 h examples are used for mechanism figures, separate them from the
main aggregate table.

## Acceptance Reading

For the stratified 6 h screen, do not require every case to save pump. Require
that the mechanism behaves in the right places:

| sample role | expected behavior |
|---|---|
| positive opportunity | save pump or reduce latch without growing the 7.5/10 deg tail |
| boundary/lookalike | abstain or stay close to the closed baseline |
| ordinary background | stay close to the closed baseline; avoid false activation |
| known failure reproducer | materially reduce the failure symptom without new fallback |

Do not promote a candidate to 24 h if a 6 h boundary/background case shows large
pump growth, new fallback, sustained `>7.5 deg`, or any increased `>10 deg` tail.

## Current Sample Policy

For P2, use the existing targeted 24 h pool and select 3-4 representative 6 h
windows per iteration:

- positive long/core opportunity
- medium positive opportunity
- ramp/event boundary lookalike
- ordinary background control

For C3, freeze `gusty_hold_current_isolated_v1` only as the current improved
candidate. It still needs an applicability gate before any production-level 24 h
claim, because the 6 h representative set shows safe but economically unstable
behavior outside the repeated-peak relief case.

## Reporting

Every candidate readout should include:

- closed and primary pump work
- pump change percentage
- latch switches
- fallback ratio
- `time/max_continuous/area` over 5, 7.5, and 10 deg
- planner target owners, especially target refresh, reuse, resume, and PI release

Use `docs/attitude_metric_semantics.md` for the attitude threshold semantics.

## Current Production-Near 12 h Reference

The current 12 h mixed-regime guard reference is documented in
`docs/paper_validation_strategy_20260605.md`.

Short reading:

- current-only: 12800.43 m3;
- production gain 0.40: 12406.56 m3, 3.08% saving;
- production-near gain 0.45: 11582.94 m3, 9.51% saving;
- gain 0.45 slightly increases `t>5 deg`, reduces `t>7.5 deg`, and does not
  increase `t>10 deg` in the guard10 12 h run.

This supports a paper robustness trade-off claim. It does not freeze gain 0.45
as the production default.
