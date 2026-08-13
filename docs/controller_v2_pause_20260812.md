# Controller V2 pause record - 2026-08-12

## Phase boundary

This phase only restructures the control-program framework. It does not tune
controller performance, claim pump reduction, or treat the current simulated
posture as physical validation evidence.

## Confirmed before the pause

- The new public entry is `wind_prediction.controller`.
- The active controller path no longer imports the historical Provider or
  `ballast_planner` stack.
- The active V2 controller modules contain about 3,000 lines in total; no
  active module exceeds the 1,200-line architecture guard.
- The historical 11,286-line `ballast_planner_provider.py` has been reduced to
  a small compatibility entry. Historical experiment code remains available
  for submitted-paper reproduction but is not the V2 development path.
- Candidate generation, ballast allocation, target semantics, pump execution
  preview, runtime state, plant adaptation and replay forecast adaptation have
  separate modules with one candidate evaluator and one final commit point.
- The replay adapter requires an explicit forecast source and does not fall
  back to measured future wind.
- `configs/controller_core_v2.json` and the strict configuration loader were
  added. The smoke runner records the normalized configuration digest.
- The compact controller completed three consecutive deterministic cycles.
- Before the final validation edits below, the complete repository test suite
  passed: 337 tests, 0 failures.
- Package-level historical exports were changed to lazy imports. Old import
  names still worked in a compatibility check, while importing the V2 public
  controller no longer eagerly loaded the old analysis stack.

## Written immediately before the pause and not yet tested

The following validation additions are present in the worktree but have not
been run through tests yet:

- constructor checks for pump duration, rates, tank capacity, hysteresis and
  pump-rate schedule in `execution_rollout.py`;
- constructor checks for forecast-policy booleans, thresholds and event keys
  in `forecast_action_policy.py`;
- cross-checks for deadband/envelope, prediction-stage count and candidate-cost
  weights in `controller_core.py`;
- `ForecastAssistedBallastController.from_config_file(...)` and the stored
  configuration digest in `controller_runtime.py`.

These changes must be treated as unverified until the focused and full tests
are rerun.

## First actions for the next session

1. Run focused tests for execution rollout, forecast action policy, controller
   configuration, core, runtime and architecture.
2. Fix any compatibility failures caused by the new constructor validation.
3. Add strict loader tests for missing sections, wrong section types, string
   booleans, invalid pump schedules and invalid forecast thresholds.
4. Change the smoke runner to use
   `ForecastAssistedBallastController.from_config_file(...)` and confirm the
   reported digest matches `configs/controller_core_v2.json`.
5. Rerun the complete test suite and update this record with the result.

## Deferred by design

- migrating the formal casebook runner from V1 Provider to V2;
- rebuilding or recalibrating the six-degree-of-freedom simulator;
- auditing total ballast mass, centre of gravity, inertia and restoring-force
  updates for three independent sea-connected tanks;
- 10-, 20- or 30-case six-hour performance experiments;
- controller parameter tuning and pump-reduction targets;
- full experiment evidence manifests for formal validation.

## Resume result later on 2026-08-12

The previously unverified configuration checks were completed and passed.
The compact controller was then connected directly to the existing pump and
six-degree-of-freedom plant through `controller_chain_runner.py`, without the
historical `ClosedLoopPolicy` changing the committed target.

Four fixed six-hour cases completed for feedback-only and prediction-assisted
variants. The final run passed finite-state, tank-capacity, exact-command,
forecast-connectivity and gross-regression checks. A minimum effective-demand
gate was added after the first run showed that very small forecast increments
could create unnecessary ordinary actions. No performance tuning was carried
out after the gate was added.

The connected-chain artifacts are in
`outputs/wind_prediction/controller_v2_chain_smoke_thresholded/`. They are
connectivity evidence only. The next phase should improve one module at a time,
starting with forecast admission and action magnitude, before any broader
performance experiment.

After the connected-chain changes, the complete repository suite passed:
345 tests, 0 failures.

## Framework verification on 2026-08-13

The research plant now separates the coupled rigid-body mass matrix from fixed
low-order added mass. Signed tank increments update total mass, centre of mass
and inertia, and the open-loop audit distinguishes independent tank-to-sea
exchange from internal transfer. The `research_incremental_v1` profile is
explicitly restricted to framework use.

The latest strengthening-case smoke run completed both six-hour variants and
passed all chain checks. A same-state future-content probe additionally shows
that changing only the model's future wind vectors changes candidate ranking
and the executable target. Evidence is stored in
`outputs/controller_v2_chain_smoke_reference_incremental_v9_20260813/` and
`outputs/platform_open_loop_audit_reference_incremental_v6_20260813/`.

The complete repository suite now passes 389 tests. Remaining work concerns
physical fidelity and calibration, not basic information flow: exact gravity
loads, variable-mass momentum, tank fill/free-surface effects, hydrodynamic and
mooring calibration, and comparison with a public reference platform.
