# FOWT Predictive Ballast Control

Research code for predictive active-ballast control of a three-column floating
offshore wind turbine. The project uses short-horizon wind forecasts together
with measured platform posture and ballast-pump state to generate the next
three-tank target-water command.

## Current Status

The repository is being reorganized around a compact V2 controller. Its aim is
to make every control decision traceable before the platform simulator is
recalibrated and broader operating-condition studies begin.

- **V2 controller**: the active development path. It has a compact public
  interface, explicit forecast evidence, execution-aware candidate evaluation,
  one final target commit, and a versioned configuration file.
- **V1 compatibility code**: retained only to reproduce the submitted-paper
  workflow and historical records. It is not the entry point for new control
  development.
- **Physical validation**: still pending. Current simulated pitch and roll are
  connectivity signals for controller integration, not engineering-grade
  evidence for tuning or performance claims.

The submitted-paper 150-case result is historical evidence for that frozen
workflow. It is not a V2 regression target or a claim that the current V2
controller has completed wide-condition validation.

## V2 Control Path

```text
measured pitch/roll, wind, tank masses and pump state
  -> validated forecast evidence
  -> feedback demand + forecast demand
  -> candidate action sequences
  -> exact three-tank target water masses
  -> pump-execution preview
  -> one evaluated action and target commit
  -> actuator state for the next control cycle
```

The six-degree-of-freedom platform dynamics sit outside this controller path.
That separation allows a recalibrated simulator to provide measured state and
consume target-water commands without changing the controller decision logic.

## Repository Layout

```text
.
├── src/wind_prediction/
│   ├── controller.py                 # Public V2 import surface
│   ├── controller_core.py            # Demand, candidates and final decision
│   ├── controller_runtime.py         # One-cycle state progression
│   ├── controller_plant_adapter.py   # Existing plant-interface adapter
│   ├── controller_replay_adapter.py  # Explicit replay forecast evidence
│   ├── ballast_allocation.py          # Two-axis to three-tank allocation
│   ├── execution_rollout.py           # Pump execution preview
│   └── ballast_planner_provider.py    # V1 compatibility entry only
├── configs/controller_core_v2.json   # Versioned V2 configuration
├── scripts/validation/
│   └── run_controller_core_smoke.py   # Deterministic V2 smoke path
├── tests/                             # Unit, architecture and adapter checks
├── docs/controller_v2_framework.md    # Detailed V2 design note
└── archive/                           # Historical implementation material
```

## Controller Semantics

The V2 controller evaluates target updates rather than selecting fixed pump
speed gears. Its action families are:

- `strengthen`, `normal`, and `reduced`: create a new target with different
  update magnitudes;
- `continue_target`: retain the active target;
- `release_target`: return the target to the currently measured tank masses;
- `reverse`: create a target in the opposite compensation direction.

Pump flow is then computed from target error, available tank capacity and pump
state. Forecast-specific high-impact actions require explicit forecast
evidence; the replay path never substitutes measured future wind when a model
forecast is unavailable.

## Quick Start

The local development environment uses Python 3.12 and `.venv312`.

```bash
PYTHONPATH=src .venv312/bin/python3.12 \
  scripts/validation/run_controller_core_smoke.py
```

The smoke output records the controller configuration digest, forecast source,
model label, three consecutive control cycles, selected actions, target masses
and pump-preview state. It checks integration only; it is not a performance
experiment.

Run the compact-controller checks with:

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

For the full repository test suite:

```bash
PYTHONPATH=src .venv312/bin/python3.12 -m unittest discover -s tests -p 'test_*.py'
```

## Configuration and Traceability

`configs/controller_core_v2.json` is the recommended V2 configuration entry.
The loader rejects unknown keys, verifies key cross-constraints and generates a
normalized SHA-256 digest. This lets a run identify the effective control
configuration instead of relying on scattered script parameters.

Each V2 decision trace includes the selected action, target operation,
candidate score, forecast-use status, pump-preview result and remaining target
error. This is the basis for later decision analysis and formal experiment
manifests.

## What Is Deliberately Deferred

The following work is outside the current controller-framework milestone:

1. recalibration and cross-checking of the six-degree-of-freedom platform
   simulator;
2. treatment of total mass, centre of gravity, inertia and restoring moments
   for three independently sea-connected ballast tanks;
3. migration of the historical casebook runner from the V1 Provider stack;
4. staged 10-, 20- and 30-case six-hour experiments under a frozen run
   protocol;
5. parameter tuning and claims about pump-volume reduction.

See [the V2 framework note](docs/controller_v2_framework.md) for the module
interfaces, current structural checks and the deferred validation boundary.

## Data and Generated Artifacts

Raw wind records, processed datasets, trained model weights, generated figures
and paper artifacts are intentionally kept outside the versioned program
framework. Reproducible experiments should record their input manifests and
configuration digests, rather than committing transient output directories to
the source repository.
