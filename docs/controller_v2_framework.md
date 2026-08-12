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
  -> actuator state for the next cycle
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

## Deferred work

The following work begins only after the controller framework is frozen:

- recalibrate the six-degree-of-freedom simulator;
- audit three-tank sea exchange, total mass, centre of gravity, inertia and
  restoring-moment updates;
- connect simulator feedback through a thin adapter;
- then perform the staged 10-, 20- and 30-case six-hour evaluations.

Until then, simulated posture is a connectivity signal rather than physical
evidence for tuning or performance claims.
