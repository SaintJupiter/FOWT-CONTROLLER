# Compact controller V2 framework

## Scope

This framework repairs the control-program structure before the platform
simulator is recalibrated. It does not claim that the present simulated pitch
or roll is physically representative, and it does not use pump reduction as a
current acceptance target.

The active controller path is:

```text
measured posture, wind, tank and pump state
  -> validated forecast evidence
  -> posture and forecast demand
  -> candidate action sequences
  -> exact ballast targets
  -> three-pump execution preview
  -> one evaluated and committed action
  -> real plant pump execution and six-DOF propagation
  -> measured tank, pump and posture state for the next cycle
```

Platform dynamics are outside this path. A later simulator can replace the
posture source and consume the same target-water output without changing the
controller interface.

## Public entry

New code imports `wind_prediction.controller` and calls
`ForecastAssistedBallastController.step(...)`.

The recommended configuration entry is
`load_controller_config("configs/controller_core_v2.json")`. The file uses a
versioned schema, rejects unknown keys and produces a normalized SHA-256
digest so a run can identify the exact controller configuration it used.

One call accepts:

- measured pitch and roll and their rates;
- current wind vector;
- optional future forecast evidence;
- actual masses and pump state of exactly three ballast tanks;
- the currently active target carried by `ControllerRuntimeState`.

It returns:

- the selected action and target operation;
- the exact three-tank target evaluated by the controller;
- the built-in actuator preview for the current decision interval;
- the next controller state;
- a compact trace containing forecast use, candidate cost, target, pump result,
  remaining target error and constraint-relevant state.

## Active modules

| Module | Responsibility |
|---|---|
| `controller.py` | Stable public import surface |
| `controller_configuration.py` | Strict versioned configuration and digest |
| `controller_runtime.py` | One-cycle orchestration and persistent state |
| `controller_plant_adapter.py` | Existing platform-state and target interface adapter |
| `controller_chain_runner.py` | Direct controller-to-plant loop and connectivity evidence |
| `controller_replay_adapter.py` | Replay history to explicit model forecast evidence |
| `controller_core.py` | Demand formation, candidate evaluation and final commit |
| `forecast_evidence.py` | Validated future wind evidence |
| `forecast_action_policy.py` | Permission for forecast-specific actions |
| `ballast_allocation.py` | Two-axis demand to three-tank mass mapping |
| `execution_rollout.py` | Three-pump target execution preview |
| `action_plan.py` | Evaluated-action identity and atomic commit rule |

The active decision path has one candidate evaluator and one commit point.
An action or target changed after evaluation cannot be committed without a new
evaluation.

The replay adapter requires an explicit forecast adapter. A missing model never
falls back to measured future wind. No-future comparisons use an explicitly
named current-only forecast adapter.

## Action and target semantics

- `strengthen`, `normal`, and `reduced` set a new exact target.
- `continue_target` preserves the active target.
- `release_target` moves the target to the measured tank masses.
- `reverse` sets a new target in the opposite compensation direction.

These names describe target changes, not fixed pump-speed gears. Pump flow is
computed separately from target error, capacity and pump-state limits.

## Compatibility boundary

`ballast_planner_provider.py`, `provider_*.py`, `ballast_planner.py` and
`provider_factory.py` form the V1 compatibility stack. They remain in the
repository to reproduce the submitted-paper result and existing evidence.
They are not part of the V2 controller interface and should receive no new
decision branches.

Once the new simulator and its adapters are ready, this compatibility stack
can be moved to the archive without changing the V2 controller.

## Current acceptance checks

The present acceptance level is intentionally structural:

1. exact input dimensions and finite values are checked;
2. forecast, decision and execution intervals must match;
3. constant future wind produces no forecast increment;
4. missing future prediction leaves feedback actions available but disables
   forecast-specific high-impact actions;
5. continue and release retain different target semantics;
6. candidate evaluation, committed target and executed target are identical;
7. consecutive cycles advance one controller state and retain unfinished
   target error;
8. the public V2 path has no dependency on Provider modules.
9. active V2 modules remain below the architecture size guard.
10. active actions are blocked when the remaining demand is below the configured
    minimum effective-demand ratio;
11. the exact evaluated target reaches the real plant without a second policy
    changing it;
12. real plant feedback, rather than preview state, starts the next control cycle.
13. every plant run exposes a complete resolved platform identity and an
    explicit use purpose;
14. forecast payload, measured decision state and candidate ranking receive
    separate content hashes;
15. a same-state probe changes only future wind-vector content and verifies
    that candidate ranking and the executable target respond.

Run the focused checks with:

```bash
PYTHONPATH=src .venv312/bin/python3.12 -m unittest \
  tests.test_action_plan \
  tests.test_ballast_allocation \
  tests.test_execution_rollout \
  tests.test_execution_rollout_target_release \
  tests.test_forecast_evidence \
  tests.test_forecast_action_policy \
  tests.test_forecast_action_policy_enforcement \
  tests.test_controller_core \
  tests.test_controller_runtime \
  tests.test_controller_configuration \
  tests.test_controller_architecture \
  tests.test_controller_plant_adapter \
  tests.test_controller_replay_adapter \
  tests.test_controller_smoke_script
```

Run the deterministic end-to-end smoke path with:

```bash
PYTHONPATH=src .venv312/bin/python3.12 \
  scripts/validation/run_controller_core_smoke.py
```

The smoke output records the configuration path and normalized digest,
forecast source and model version, completed cycle count and per-cycle traces.
It is a connectivity check, not a controller-performance experiment.

Run a selected six-hour controller-to-plant framework check with:

```bash
PYTHONPATH=src .venv312/bin/python3.12 \
  scripts/validation/run_controller_v2_chain_smoke.py \
  --output-dir outputs/controller_v2_chain_smoke_reference_incremental_v9_20260813
```

The configuration contains four fixed test-set cases spanning low disturbance,
future relief or reversal, oscillation and strengthening. A framework run may
select one to four of them. Each selected case runs feedback-only and
prediction-assisted variants through the pump model and six-degree-of-freedom
plant. Its manifest deliberately labels the output as framework evidence rather
than performance evidence.

## Connected-chain result on 2026-08-13

The current formal framework run covers all four configured cases. Each of its
eight variant trajectories completed 216,000 plant steps. Every committed
target reached the plant unchanged, all states remained finite, tank masses
stayed within capacity and platform mass balance closed to numerical
precision. Prediction evidence reached all 72 prediction-assisted decisions
and none of the 72 feedback-only decisions.

Across the strengthening and oscillation cases, the closed-loop traces contain
action and target differences; the relief/reversal case contains smaller target
differences, and the low-disturbance case remains inactive in both variants. A
separate same-state probe holds platform state, current wind, tank and pump state, event
probabilities and controller configuration fixed. Scaling only the actual
model's future wind vectors changes all three candidate-ranking hashes and
changes the executable target by as much as 49,760.60 kg. This is mechanism
evidence that future payload content reaches candidate evaluation. It does not
establish forecast accuracy, controller benefit or engineering performance.

Artifacts are stored under
`outputs/controller_v2_chain_smoke_reference_incremental_v9_20260813/` and
include a manifest, source and data hashes, platform identity, summary table,
sanity checks, forecast-content probes, sampled time series and decision
traces. The manifest records `formal_suite_complete=true`. The complete
repository suite contains 389 passing tests at this
stage.

Connected plant runs fail before stepping when controller execution constants
do not match the platform identity. Target slew remains available to isolated
execution-rollout studies but is deliberately rejected by the connected plant
adapter until one limiter owns both the evaluated and executed command path.
The framework run also shows that prediction-assisted pump volume is not lower
in every case, so these artifacts must not be used as performance evidence.

## Deferred work

The first simulator repair stage is connected through the
`research_incremental_v1` platform profile. It treats the initial working
ballast as the incremental-load reference, removes the zero-displacement
offset from the mooring curve, updates total mass, centre of mass and
point-mass ballast inertia after pump execution, and combines the coupled
rigid-body matrix with a fixed low-order added-mass matrix. The profile is
marked `framework_only_not_for_performance_validation`. The submitted-paper
`default` profile and its aliases remain unchanged.

The current state update uses generalized position and attitude rates directly
as the six velocity channels. It therefore has only a small-angle linear
interpretation near the reference attitude; it is not a body-coordinate
nonlinear six-degree-of-freedom formulation. The incremental reference also
subtracts the baseline load rather than solving an absolute gravity,
buoyancy, and mooring equilibrium. Both choices are temporary framework
conventions and must be replaced by an explicit mathematical contract and an
independent active plant module before physical calibration.

All four configured six-hour cases completed with the new profile. Both
controller variants retained finite states, exact target transfer and
tank-capacity compliance. Three cases exercised pump motion; their numerical
comparisons are connectivity evidence, not performance results.

The current open-loop evidence is stored under
`outputs/platform_open_loop_audit_reference_incremental_v6_20260813/`. Its
twelve directional, mass-property and signed-exchange checks pass, and its
summary records input, source and output hashes. The following physical-model
work remains deferred:

- validate the assembled mass, centre of gravity, inertia, hydrostatics and
  tank positions against one complete public reference-platform definition;
- add exact attitude-dependent gravity generalized forces, variable-mass
  momentum-flux terms and tank fill/free-surface geometry where required;
- replace the empirical wave-force path and copied horizontal mooring curve
  with literature-backed low-order hydrodynamic and three-line mooring models;
- calibrate free-decay periods, damping and wind-load gains against public
  OpenFAST or reference-platform responses;
- recalibrate action size, forecast admission and candidate cost only after
  individual modules have clear physical semantics;
- then perform staged 10-, 20- and 30-case six-hour evaluations.

The public-reference audit now distinguishes evidence from runtime use. The
official IEAWindSystems v1.1.16 values are recorded in
`configs/reference_platforms/volturnus_s_openfast_v1_1_16.json`, but that
manifest is deliberately marked as evidence-only. The historical
`reference_mapped_volturnus_s` profile is a period-matched mixed proxy and is
not eligible for controller validation.

The research plant now stores complete reference mass properties and applies
only signed tank-mass increments around the working ballast. This does not
claim that the current local tank geometry is a VolturnUS-S tank design; it
prepares the plant interface so a future complete reference mass matrix can be
inserted without recounting its baseline ballast.

Until then, simulated posture is a connectivity signal rather than physical
evidence for tuning or performance claims.
