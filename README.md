# FOWT Predictive Ballast Control

This repository contains the program framework for a floating offshore wind
turbine (FOWT) predictive ballast-control study. The current main branch is a
clean code-oriented workspace: it keeps the reusable controller, forecasting,
replay, validation, and experiment scripts, while excluding raw data, generated
outputs, trained model weights, and paper artifacts.

The project focus is not a single fixed parameter setting. It is a control
chain for using short-horizon wind prediction and replay evidence to support
ballast decisions under safety constraints.

## What This Code Does

The program is organized around five responsibilities:

1. Forecast interface
   - Converts wind-prediction sources into a common control-facing contract.
   - Supports learned forecasts, persistence/current-only baselines, and
     replay-compatible adapters.

2. Control decision layer
   - Builds prediction-aware ballast target updates.
   - Separates forecast signals, target lifecycle management, safety guards,
     and pump-action planning.

3. Safety and telemetry
   - Tracks attitude, target age, pump activity, control state, and guard
     decisions.
   - Keeps validation metadata close to each run through protocol records.

4. Replay and casebook validation
   - Replays selected wind windows through comparable control policies.
   - Supports controlled comparisons between learned, current-only,
     persistence, oracle, and ablation-style policies.

5. Experiment support
   - Provides scripts for data preparation, model training, offline audits,
     selector diagnostics, plotting, and paper-facing validation checks.

## Repository Layout

```text
.
├── src/wind_prediction/        # Reusable control and forecast framework
├── scripts/
│   ├── analysis/               # Replay, validation, diagnostics, summaries
│   ├── data_preparation/       # Dataset construction utilities
│   ├── experiments/            # Higher-level experiment entry points
│   ├── modeling/               # Forecast and decision-model training scripts
│   └── paper_figures/          # Figure generation helpers
├── configs/                    # Controller profiles, gates, and registries
├── docs/                       # Method notes, validation protocol, paper plan
├── requirements.txt            # Python dependencies
└── README.md
```

## Core Modules

`src/wind_prediction/forecast_contract.py` and
`src/wind_prediction/forecast_adapter.py` define how forecast signals enter the
controller.

`src/wind_prediction/ballast_planner.py` and
`src/wind_prediction/ballast_planner_provider.py` contain the main ballast
planning logic and runtime integration points.

`src/wind_prediction/target_lifecycle.py` manages target creation, refresh,
holding, release, and stale-target behavior.

`src/wind_prediction/safety_supervisor.py` provides safety supervision around
attitude and control-state boundaries.

`src/wind_prediction/replay_dataset.py` and
`src/wind_prediction/run_protocol.py` support replayable validation and
traceable run records.

`src/wind_prediction/attitude_metrics.py`,
`src/wind_prediction/control_contracts.py`, and
`src/wind_prediction/control_telemetry_defaults.py` define shared metrics,
contracts, and telemetry defaults.

## Main Script Families

`scripts/analysis/run_prediction_primary_casebook.py` is the main replay and
casebook runner for prediction-aware control experiments.

`scripts/analysis/run_positive_tilt_conservative_20260611.py` is a controlled
entry point for conservative attitude-sensitive follow-up runs.

`scripts/analysis/check_*` scripts are lightweight validation checks for
contracts, manifests, and smoke gates.

`scripts/modeling/` contains model-training and calibration utilities for wind
forecasting and decision-signal experiments.

`scripts/data_preparation/` contains dataset construction utilities. The data
itself is intentionally not stored in this repository.

## Data And Generated Artifacts

The clean main branch does not include:

- raw wind data;
- processed training arrays;
- generated replay outputs;
- trained model weights;
- large figures, PDFs, Office documents, or archived experiment packages.

Those files should be kept in external storage or regenerated locally. This
keeps the repository focused on the program framework and avoids mixing code
with transient experiment products.

## Environment

The active local environment for this project is Python 3.12. In the current
workspace this is represented by `.venv312`.

Typical local checks:

```bash
PYTHONPATH=src .venv312/bin/python -m py_compile $(find src scripts -name '*.py')
PYTHONPATH=src .venv312/bin/python -c "import wind_prediction"
PYTHONPATH=src .venv312/bin/python scripts/analysis/run_prediction_primary_casebook.py --help
```

## Validation Discipline

The repository includes protocol notes under `docs/` to keep experiment claims
separate from exploratory runs. A result should not be treated as paper-facing
evidence unless the corresponding run directory, summary table, and protocol
record all exist.

Recommended workflow:

1. inspect representative good and bad cases;
2. test decision logic in shadow mode where possible;
3. run small canary batches before full replay;
4. compare learned policies against current-only, persistence, shuffled, or
   other negative-control baselines;
5. only then promote stable runs into paper-facing summaries.

## Current Branch Purpose

`main` is now intended to be a clean program branch. It is suitable for reading
the controller architecture, running validation scripts, and extending the
forecast-aware ballast-control framework without carrying historical data or
large generated artifacts.
