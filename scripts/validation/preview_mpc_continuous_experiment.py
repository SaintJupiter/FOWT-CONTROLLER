#!/usr/bin/env python3
"""Compare preview content and horizon under identical recorded plant forcing.

Each fixed case is six hours long.  The plant disturbance in every controller
variant is reconstructed between adjacent recorded ENU wind vectors at the
physical substep rate, then converted through the shared rotor-load model.
Only the controller-visible forecast changes:

* ``current_observation_1block`` uses the current observed load for one control block;
* ``lstm_6block`` uses all six learned future load points;
* ``lstm_1block`` uses only the first learned future load point;
* ``persistence_6block`` repeats the current observed load for six blocks;
* ``recorded_oracle_6block`` uses the next six recorded loads noncausally.

The oracle is a diagnostic reference for measuring how much additional value
perfect future information can provide inside the otherwise frozen system.  It
cannot by itself distinguish controller, actuator, model or case-set limits.
It is not deployable, not a formal upper bound and not a population claim.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import replace
from datetime import datetime, timedelta
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SRC_DIRECTORY = ROOT / "src"
VALIDATION_DIRECTORY = Path(__file__).resolve().parent
for directory in (SRC_DIRECTORY, VALIDATION_DIRECTORY):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from fowt_platform import (  # noqa: E402
    IncrementalState,
    ThreeTankDifferentialModes,
    assemble_ballast_model_snapshot,
    solve_incremental_static_offset,
)
from real_lstm_preview_fixture import (  # noqa: E402
    DATASET_DIRECTORY,
    MODEL_DIRECTORY,
    REFERENCE_MANIFEST,
    REFERENCE_TANK_MASSES_KG,
    RealLstmPreviewResources,
    RealLstmWindRecord,
    TANK_CAPACITY_KG,
    TANK_COORDINATES_M,
    assemble_planner_rotor_preview_from_record,
    assemble_recorded_interval_rotor_load_substeps,
    assemble_real_lstm_rotor_preview_from_record,
    infer_real_lstm_wind_record,
    inspect_real_lstm_wind_record_operating_domain,
    prepare_real_lstm_preview_resources,
)
from preview_mpc_experiment_runtime import (  # noqa: E402
    average_rotor_loads,
    execution_config,
    initial_execution_state,
    research_runtime_assembly,
    run_preview_mpc_cycle,
)
from wind_prediction.execution_rollout import (  # noqa: E402
    ExecutionRolloutRequest,
    simulate_execution_step,
)
from wind_prediction.preview_mpc_control_cycle import (  # noqa: E402
    PreviewMPCNoSafeCandidateError,
)
from wind_prediction.preview_mpc_design import (  # noqa: E402
    research_preview_mpc_design_v4,
)
from wind_prediction.posture_feedback_baseline import (  # noqa: E402
    decide_posture_feedback_target,
    research_posture_feedback_baseline_v2,
)
from wind_prediction.physical_execution_platform_path import (  # noqa: E402
    advance_physical_execution_platform_path,
)
from wind_prediction.replay_dataset import TIMESTAMP_FMT  # noqa: E402


FIXED_CASE_ORIGINS = (
    "2022-12-09 20:40:00",
    "2023-02-15 14:10:00",
    "2024-10-23 18:10:00",
)
CYCLE_COUNT = 36
# Each window starts from the local static offset under its first recorded wind
# load. Optional warmup blocks remain available for diagnostics, but the fixed
# semantic cases do not require an unsupported historical prefix.
WARMUP_BLOCK_COUNT = 0
VARIANTS = (
    ("current_observation_1block", "current_observation", 1),
    ("lstm_6block", "lstm", 6),
    ("lstm_1block", "lstm", 1),
    ("persistence_6block", "persistence", 6),
    ("recorded_oracle_6block", "recorded_oracle", 6),
)
POSTURE_FEEDBACK_BASELINE = "posture_feedback_baseline"
ALL_VARIANT_NAMES = (POSTURE_FEEDBACK_BASELINE,) + tuple(
    name for name, _, _ in VARIANTS
)
CONTRASTS = (
    (POSTURE_FEEDBACK_BASELINE, "lstm_6block"),
    (POSTURE_FEEDBACK_BASELINE, "persistence_6block"),
    ("current_observation_1block", "lstm_6block"),
    ("current_observation_1block", "persistence_6block"),
    ("persistence_6block", "lstm_6block"),
    ("lstm_1block", "lstm_6block"),
    ("lstm_6block", "recorded_oracle_6block"),
    ("persistence_6block", "recorded_oracle_6block"),
)
CONTRAST_INTERPRETATIONS = {
    (POSTURE_FEEDBACK_BASELINE, "lstm_6block"): (
        "complete_prediction_mpc_relative_to_current_posture_feedback_on_the_"
        "shared_plant_and_pump_path"
    ),
    (POSTURE_FEEDBACK_BASELINE, "persistence_6block"): (
        "prediction_free_preview_structure_relative_to_current_posture_"
        "feedback_on_the_shared_plant_and_pump_path"
    ),
    ("current_observation_1block", "lstm_6block"): (
        "complete_prediction_mpc_relative_to_current_observation_single_block_"
        "reference_combines_forecast_content_and_preview_horizon"
    ),
    ("current_observation_1block", "persistence_6block"): (
        "preview_horizon_effect_under_current_observation_only_not_future_"
        "prediction_content"
    ),
    ("persistence_6block", "lstm_6block"): (
        "future_prediction_content_effect_with_identical_six_block_preview_"
        "structure"
    ),
    ("lstm_1block", "lstm_6block"): (
        "learned_future_preview_horizon_effect_with_identical_forecast_source"
    ),
    ("lstm_6block", "recorded_oracle_6block"): (
        "forecast_error_diagnostic_within_frozen_six_block_controller_not_an_"
        "engineering_upper_bound"
    ),
    ("persistence_6block", "recorded_oracle_6block"): (
        "perfect_future_information_diagnostic_relative_to_current_observation_"
        "persistence_not_a_deployable_comparison"
    ),
}
POSTURE_THRESHOLDS_DEG = (1.5, 2.0, 3.0, 5.0)
OBJECTIVE_COMPONENTS = (
    "running_posture",
    "terminal_posture",
    "tank_movement",
    "tank_throughput",
    "movement_change",
    "posture_slack",
    "numerical_regularization",
)


def _load_digest(values: list[np.ndarray]) -> str:
    array = np.ascontiguousarray(np.asarray(values, dtype=np.float64))
    return hashlib.sha256(array.tobytes()).hexdigest()


def _source_record_sequence_digest(
    records: list[RealLstmWindRecord],
) -> str:
    payload = json.dumps(
        [record.source_record_sha256 for record in records],
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _controller_platform_source_tree_sha256() -> str:
    """Fingerprint all local source modules that can affect the active chain.

    The screening runner imports modules through package initializers as well as
    direct imports.  Hashing both active source packages prevents a registry
    from remaining valid after an unlisted platform or controller helper is
    changed.  This is an execution identity, not a physical-model validation.
    """

    entries: list[tuple[str, str]] = []
    for package in ("fowt_platform", "wind_prediction"):
        package_root = SRC_DIRECTORY / package
        for path in sorted(package_root.rglob("*.py")):
            entries.append((path.relative_to(SRC_DIRECTORY).as_posix(), _file_sha256(path)))
    if not entries:
        raise RuntimeError("controller and platform source trees contain no Python modules")
    payload = json.dumps(entries, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


@lru_cache(maxsize=1)
def _runtime_source_artifact_identity() -> dict[str, str]:
    """Fingerprint the concrete model, replay and platform inputs used here.

    The digest is intentionally limited to files read by the real-LSTM replay
    path.  It is not a claim that these inputs establish physical validity.
    """

    canonical_observations = (
        DATASET_DIRECTORY.parent
        / "fino1_platform_10min"
        / "canonical_observations_10min.csv.gz"
    )
    files = {
        "forecast_adapter": SRC_DIRECTORY / "wind_prediction/forecast_adapter.py",
        "replay_dataset": SRC_DIRECTORY / "wind_prediction/replay_dataset.py",
        "dataset_manifest": SRC_DIRECTORY / "wind_prediction/dataset_manifest.py",
        "lstm_weights": MODEL_DIRECTORY / "lstm_best.pt",
        "lstm_config": MODEL_DIRECTORY / "lstm_config.json",
        "lstm_event_thresholds": MODEL_DIRECTORY / "lstm_event_thresholds.json",
        "lstm_regression_metrics": MODEL_DIRECTORY / "lstm_regression_metrics.csv",
        "dataset_metadata": DATASET_DIRECTORY / "metadata.json",
        "dataset_scaler": DATASET_DIRECTORY / "scaler_train.json",
        "dataset_sample_index": DATASET_DIRECTORY / "sample_index.csv.gz",
        "dataset_test_features": DATASET_DIRECTORY / "X_test.npy",
        "dataset_test_future_uv": DATASET_DIRECTORY / "y_uv_raw_test.npy",
        "dataset_test_events": DATASET_DIRECTORY / "y_event_test.npy",
        "canonical_test_observations": canonical_observations,
        "reference_platform_manifest": REFERENCE_MANIFEST,
        "rotor_load_input_audit": (
            VALIDATION_DIRECTORY / "run_rotor_normal_load_input_audit.py"
        ),
        "rotor_nominal_reference_state": (
            ROOT
            / "configs/reference_platforms"
            / "volturnus_s_openfast_nominal_reference_state_v1.json"
        ),
    }
    return {name: _file_sha256(path) for name, path in files.items()}


def runtime_source_artifact_sha256() -> str:
    payload = json.dumps(
        _runtime_source_artifact_identity(),
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def implementation_identity() -> dict[str, str]:
    files = {
        "preview_mpc": SRC_DIRECTORY / "wind_prediction/preview_mpc.py",
        "preview_mpc_design": (
            SRC_DIRECTORY / "wind_prediction/preview_mpc_design.py"
        ),
        "preview_mpc_runtime": (
            SRC_DIRECTORY / "wind_prediction/preview_mpc_runtime.py"
        ),
        "execution_rollout": (
            SRC_DIRECTORY / "wind_prediction/execution_rollout.py"
        ),
        "ballast_modes": SRC_DIRECTORY / "fowt_platform/ballast_modes.py",
        "ballast_snapshot": SRC_DIRECTORY / "fowt_platform/ballast_snapshot.py",
        "incremental_platform": SRC_DIRECTORY / "fowt_platform/incremental.py",
        "source_consistent_reference": (
            SRC_DIRECTORY / "fowt_platform/source_consistent_reference.py"
        ),
        "rotor_generalized_load": SRC_DIRECTORY / "fowt_platform/rotor_input.py",
        "generalized_load_forecast": (
            SRC_DIRECTORY / "fowt_platform/generalized_load_forecast.py"
        ),
        "forecast_replay_dataset": (
            SRC_DIRECTORY / "wind_prediction/replay_dataset.py"
        ),
        "real_lstm_preview_fixture": (
            VALIDATION_DIRECTORY / "real_lstm_preview_fixture.py"
        ),
        "rolling_controller_script": (
            VALIDATION_DIRECTORY
            / "preview_mpc_experiment_runtime.py"
        ),
        "validation_script": Path(__file__).resolve(),
    }
    identity = {name: _file_sha256(path) for name, path in files.items()}
    identity["controller_platform_source_tree"] = _controller_platform_source_tree_sha256()
    identity["runtime_source_artifacts"] = runtime_source_artifact_sha256()
    return identity


def prepare_source_records(
    *,
    origin: datetime,
    device: str,
    cycle_count: int,
    lookahead_count: int,
    resources: RealLstmPreviewResources,
    warmup_block_count: int = 0,
) -> list[RealLstmWindRecord]:
    if int(warmup_block_count) != warmup_block_count or warmup_block_count < 0:
        raise ValueError("warmup_block_count must be a non-negative integer")
    warmup_count = int(warmup_block_count)
    first_origin = origin - timedelta(minutes=10 * warmup_count)
    if str(device) != resources.device:
        raise ValueError("device must match the prepared real-LSTM resource bundle")
    records: list[RealLstmWindRecord] = []
    for offset in range(warmup_count + cycle_count + lookahead_count):
        records.append(
            infer_real_lstm_wind_record(
                resources=resources,
                origin=first_origin + timedelta(minutes=10 * offset),
            )
        )
    return records


def preflight_source_operating_domain(
    *,
    source_records: list[RealLstmWindRecord],
    resources: RealLstmPreviewResources,
    origin: datetime,
    cycle_count: int,
    warmup_block_count: int = 0,
) -> dict[str, Any]:
    """Check the exact source records that a rolling window will consume.

    Source inference and physical-input assembly happen once in
    ``prepare_source_records``.  This function only reads the stored scope and
    domain report, so preflight cannot inspect a different forecast realization
    from the one later supplied to the controller.
    """

    if int(warmup_block_count) != warmup_block_count or warmup_block_count < 0:
        raise ValueError("warmup_block_count must be a non-negative integer")
    warmup_count = int(warmup_block_count)
    checks: list[dict[str, Any]] = []
    for offset, source_record in enumerate(source_records):
        is_active = warmup_count <= offset < warmup_count + int(cycle_count)
        domain = inspect_real_lstm_wind_record_operating_domain(
            resources=resources,
            source_record=source_record,
            platform_state=IncrementalState.zeros(),
            include_future=is_active,
        )
        checks.append(
            {
                "record_offset": offset,
                "origin_time": str(source_record.forecast.origin_time),
                "source_record_scope": domain["checked_input_scope"],
                "source_record_sha256": source_record.source_record_sha256,
                "supported": bool(domain["supported"]),
                "unsupported_record_count": int(domain["unsupported_record_count"]),
                "records": domain["records"],
                "derived_nominal_inflow_range_mps": domain[
                    "derived_nominal_inflow_range_mps"
                ],
            }
        )
    unsupported = [check for check in checks if not check["supported"]]
    return {
        "purpose": "source_bound_nominal_rotor_operating_domain_preflight",
        "window_origin_time": origin.strftime(TIMESTAMP_FMT),
        "warmup_block_count": warmup_count,
        "checked_source_record_count": len(checks),
        "supported": not unsupported,
        "unsupported_source_record_count": len(unsupported),
        "checks": checks,
        "unsupported_checks": unsupported,
        "scope": (
            "exact_prepared_source_records_with_current_only_warmup_and_"
            "followup_records_and_current_plus_forecast_active_records"
        ),
    }


def _precondition_window_initial_state(
    *,
    source_records: list[RealLstmWindRecord],
    resources: RealLstmPreviewResources,
    warmup_block_count: int,
    runtime: Any,
) -> tuple[IncrementalState, Any, dict[str, Any]]:
    """Advance a candidate-independent, no-pump prefix before statistics.

    The prefix uses only recorded rotor endpoint loads and holds each tank at
    its actual mass.  It therefore establishes a common dynamic state without
    allowing a candidate's objective weights or future forecast to influence
    the window start.  Pump activity and all preconditioning time are excluded
    from six-hour aggregate metrics.
    """

    if int(warmup_block_count) != warmup_block_count or warmup_block_count < 0:
        raise ValueError("warmup_block_count must be a non-negative integer")
    warmup_count = int(warmup_block_count)
    if len(source_records) < warmup_count + 1:
        raise ValueError("source_records do not cover the requested warmup prefix")

    execution_state = initial_execution_state()
    capacities = np.full(3, TANK_CAPACITY_KG)
    initial_snapshot = assemble_ballast_model_snapshot(
        runtime_assembly=runtime,
        actual_tank_masses_kg=execution_state.actual_masses_kg,
        reference_tank_masses_kg=REFERENCE_TANK_MASSES_KG,
        tank_capacities_kg=capacities,
        tank_coordinates_m=TANK_COORDINATES_M,
    )
    initial_preview = assemble_real_lstm_rotor_preview_from_record(
        resources=resources,
        source_record=source_records[0],
        platform_state=IncrementalState.zeros(),
    )
    initial_load = (
        initial_preview.load_assembly.current_rotor_load.generalized_load_platform
        + initial_snapshot.incremental_ballast_load
    )
    initial_offset = solve_incremental_static_offset(
        initial_snapshot.matrices,
        initial_load,
    )
    platform_state = IncrementalState(
        position=initial_offset.position,
        velocity=np.zeros(6),
    )
    load_trace: list[np.ndarray] = []
    total_duration_s = 0.0
    for index in range(warmup_count):
        duration_s = float(source_records[index].forecast.sample_period_s)
        config = execution_config(block_duration_s=duration_s)
        substep_count = int(np.ceil(duration_s / config.internal_step_s))
        realised_loads = assemble_recorded_interval_rotor_load_substeps(
            resources=resources,
            current_record=source_records[index],
            following_record=source_records[index + 1],
            platform_state=platform_state,
            substep_count=substep_count,
        )
        rotor_load = average_rotor_loads(realised_loads)
        path = advance_physical_execution_platform_path(
            platform_state=platform_state,
            execution_state=execution_state,
            execution_request=ExecutionRolloutRequest.track(
                execution_state.actual_masses_kg
            ),
            execution_config=config,
            runtime_assembly=runtime,
            reference_tank_masses_kg=REFERENCE_TANK_MASSES_KG,
            tank_capacities_kg=capacities,
            tank_coordinates_m=TANK_COORDINATES_M,
            rotor_load=rotor_load,
            rotor_load_substeps=realised_loads,
            wave_load=np.zeros(6),
            other_load=np.zeros(6),
            duration_s=duration_s,
        )
        platform_state = path.final_platform_state
        execution_state = path.final_execution_state
        if not np.allclose(
            execution_state.actual_masses_kg,
            REFERENCE_TANK_MASSES_KG,
            rtol=0.0,
            atol=1.0e-9,
        ):
            raise RuntimeError("no-pump preconditioning changed tank masses")
        load_trace.extend(
            np.asarray(load.generalized_load_platform, dtype=float)
            for load in realised_loads
        )
        total_duration_s += duration_s
    return platform_state, execution_state, {
        "kind": (
            "candidate_independent_local_static_initialization_then_"
            "recorded_load_no_pump_preconditioning"
        ),
        "static_initialization_wind_source": (
            "first_available_recorded_observation"
        ),
        "static_initialization_residual_load_norm_n": float(
            np.linalg.norm(initial_offset.residual_load)
        ),
        "static_initial_roll_pitch_deg": np.rad2deg(
            initial_offset.position[[3, 4]]
        ).tolist(),
        "warmup_block_count": warmup_count,
        "duration_s": total_duration_s,
        "statistics_exclude_preconditioning": True,
        "external_rotor_load_sha256": _load_digest(load_trace),
        "final_roll_pitch_deg": np.rad2deg(platform_state.position[[3, 4]]).tolist(),
        "final_roll_pitch_rate_deg_s": np.rad2deg(
            platform_state.velocity[[3, 4]]
        ).tolist(),
        "final_tank_masses_kg": execution_state.actual_masses_kg.tolist(),
        "pump_activity": "held_off_by_tracking_current_tank_masses",
    }


def _compact_cycle(cycle: dict[str, Any]) -> dict[str, Any]:
    execution = cycle["execution"]
    design = cycle["controller_configuration"]["preview_mpc_design"]
    selected_candidate = next(
        item
        for item in execution["physical_precheck_candidates"]
        if item["source"] == execution["selected_request_source"]
    )
    return {
        "cycle_index": int(cycle["cycle_index"]),
        "origin_time": cycle["source_binding"]["origin_time"],
        "planner_oracle_future_endpoint_times": cycle["source_binding"][
            "planner_oracle_future_endpoint_times"
        ],
        "plan_success": bool(cycle["plan"]["success"]),
        "solver_converged": bool(cycle["plan"]["solver_converged"]),
        "plan_status": cycle["plan"]["status"],
        "preview_mpc_design_identity": design["identity"],
        "preview_mpc_design_status": design["status"],
        "preview_mpc_design_parameter_sha256": design["parameter_sha256"],
        "normalized_objective_priorities": cycle["controller_configuration"][
            "normalized_objective_priorities"
        ],
        "objective_priority_multipliers": cycle["controller_configuration"][
            "objective_priority_multipliers"
        ],
        "planner_objective_breakdown": cycle["plan"]["objective_breakdown"],
        "planned_absolute_tank_throughput_kg_by_block": cycle["plan"][
            "planned_absolute_tank_throughput_kg_by_block"
        ],
        "selected_request_source": execution["selected_request_source"],
        "solver_iterations": int(execution["solver"]["iterations"]),
        "solver_primal_residual": float(execution["solver"]["primal_residual"]),
        "solver_dual_residual": float(execution["solver"]["dual_residual"]),
        "maximum_scaled_constraint_violation": float(
            cycle["plan"]["maximum_scaled_constraint_violation"]
        ),
        "maximum_posture_constraint_violation_deg": float(
            cycle["plan"]["maximum_posture_constraint_violation_deg"]
        ),
        "maximum_tank_constraint_violation_kg": float(
            cycle["plan"]["maximum_tank_constraint_violation_kg"]
        ),
        "maximum_block_tank_movement_violation_kg": float(
            cycle["plan"]["maximum_block_tank_movement_violation_kg"]
        ),
        "maximum_first_block_actuator_violation_kg": float(
            cycle["plan"]["maximum_first_block_actuator_violation_kg"]
        ),
        "maximum_tank_throughput_epigraph_violation_kg": float(
            cycle["plan"]["maximum_tank_throughput_epigraph_violation_kg"]
        ),
        "used_fallback": bool(execution["used_fallback"]),
        "degraded_selection": bool(execution["degraded_selection"]),
        "emergency_selection": bool(execution["emergency_selection"]),
        "selected_from_planner_load_precheck_safe_options": bool(
            execution["selected_from_planner_load_precheck_safe_options"]
        ),
        "selected_horizon_within_working_model_scope": bool(
            execution["selected_horizon_within_working_model_scope"]
        ),
        "selected_first_block_within_working_model_scope": bool(
            execution["selected_first_block_within_working_model_scope"]
        ),
        "actual_execution_within_working_model_scope": bool(
            execution["realised_execution_within_working_model_scope"]
        ),
        "total_mass_within_scope": bool(
            selected_candidate["total_mass_within_scope"]
        ),
        "maximum_abs_total_mass_deviation_from_reference_kg": float(
            selected_candidate[
                "maximum_abs_total_mass_deviation_from_reference_kg"
            ]
        ),
        "selected_posture_precheck_passed": bool(
            execution["selected_posture_precheck_passed"]
        ),
        "no_safe_precheck_candidate": not any(
            item["posture_precheck_passed"]
            for item in execution["physical_precheck_candidates"]
        ),
        "actual_sampled_posture_limit_satisfied": bool(
            execution["sampled_posture_limit_satisfied"]
        ),
        "first_target_tank_mass_delta_kg": cycle["plan"][
            "first_target_tank_mass_delta_kg"
        ],
        "committed_target_masses_kg": execution["committed_target_masses_kg"],
        "actual_first_block_tank_delta_kg": execution[
            "actual_first_block_tank_delta_kg"
        ],
        "transferred_volume_m3": float(execution["transferred_volume_m3"]),
        "aggregate_pump_active_time_s": float(
            execution["aggregate_pump_active_time_s"]
        ),
        "pump_start_count": int(execution["pump_start_count"]),
        "pump_stop_count": int(execution["pump_stop_count"]),
        "pump_direction_switch_count": int(
            execution["pump_direction_switch_count"]
        ),
        "peak_abs_roll_pitch_deg": execution["actual_peak_abs_roll_pitch_deg"],
        "end_roll_pitch_deg": execution["actual_end_roll_pitch_deg"],
        "dominant_tilt_squared_integral_deg2_s": float(
            execution["dominant_tilt_squared_integral_deg2_s"]
        ),
        "dominant_tilt_time_above_deg_s": execution[
            "dominant_tilt_time_above_deg_s"
        ],
        "plant_generalized_rotor_load_platform": execution[
            "execution_generalized_rotor_load_platform"
        ],
    }


def _aggregate_objective_breakdown(
    cycles: list[dict[str, Any]],
) -> dict[str, Any]:
    rows = [item["planner_objective_breakdown"] for item in cycles]
    count = len(rows)
    component_sums = {
        name: float(sum(float(row[name]) for row in rows))
        for name in OBJECTIVE_COMPONENTS
    }
    absolute_total = float(sum(float(row["absolute_total"]) for row in rows))
    per_block_names = (
        "running_posture_by_block",
        "tank_movement_by_block",
        "tank_throughput_by_block",
        "movement_change_by_block",
    )
    per_block_sums = {
        name: np.sum(
            np.asarray([row[name] for row in rows], dtype=float), axis=0
        ).tolist()
        for name in per_block_names
    }
    first_block_sum = float(
        sum(float(row[name][0]) for row in rows for name in per_block_names)
    )
    return {
        "cycle_count": count,
        "component_sum": component_sums,
        "component_mean_per_cycle": {
            name: value / count for name, value in component_sums.items()
        },
        "component_fraction_of_absolute_total": {
            name: (value / absolute_total if absolute_total > 0.0 else 0.0)
            for name, value in component_sums.items()
        },
        "per_block_sum": per_block_sums,
        "first_block_running_and_movement_cost_sum": first_block_sum,
        "absolute_total_sum": absolute_total,
        "constant_offset_sum": float(
            sum(float(row["constant_offset"]) for row in rows)
        ),
        "reduced_total_sum": float(
            sum(float(row["reduced_total"]) for row in rows)
        ),
    }


def _combine_objective_breakdowns(
    summaries: list[dict[str, Any]],
) -> dict[str, Any]:
    cycle_count = int(sum(int(item["cycle_count"]) for item in summaries))
    component_sums = {
        name: float(sum(item["component_sum"][name] for item in summaries))
        for name in OBJECTIVE_COMPONENTS
    }
    absolute_total = float(sum(item["absolute_total_sum"] for item in summaries))
    per_block_sums = {
        name: np.sum(
            np.asarray([item["per_block_sum"][name] for item in summaries], dtype=float),
            axis=0,
        ).tolist()
        for name in (
            "running_posture_by_block",
            "tank_movement_by_block",
            "tank_throughput_by_block",
            "movement_change_by_block",
        )
    }
    return {
        "cycle_count": cycle_count,
        "component_sum": component_sums,
        "component_mean_per_cycle": {
            name: value / cycle_count for name, value in component_sums.items()
        },
        "component_fraction_of_absolute_total": {
            name: (value / absolute_total if absolute_total > 0.0 else 0.0)
            for name, value in component_sums.items()
        },
        "per_block_sum": per_block_sums,
        "first_block_running_and_movement_cost_sum": float(
            sum(item["first_block_running_and_movement_cost_sum"] for item in summaries)
        ),
        "absolute_total_sum": absolute_total,
        "constant_offset_sum": float(
            sum(item["constant_offset_sum"] for item in summaries)
        ),
        "reduced_total_sum": float(
            sum(item["reduced_total_sum"] for item in summaries)
        ),
    }


def _aggregate(cycles: list[dict[str, Any]]) -> dict[str, Any]:
    duration_s = float(len(cycles) * 600.0)
    squared_integral = float(
        sum(item["dominant_tilt_squared_integral_deg2_s"] for item in cycles)
    )
    peak = np.max(
        np.asarray([item["peak_abs_roll_pitch_deg"] for item in cycles], dtype=float),
        axis=0,
    )
    threshold_times = {
        str(threshold): float(
            sum(
                item["dominant_tilt_time_above_deg_s"][str(threshold)]
                for item in cycles
            )
        )
        for threshold in POSTURE_THRESHOLDS_DEG
    }
    solver_iterations = np.asarray(
        [item["solver_iterations"] for item in cycles], dtype=float
    )
    return {
        "completed_cycle_count": len(cycles),
        "continuous_duration_h": duration_s / 3600.0,
        "transferred_volume_m3": float(
            sum(item["transferred_volume_m3"] for item in cycles)
        ),
        "aggregate_pump_active_time_s": float(
            sum(item["aggregate_pump_active_time_s"] for item in cycles)
        ),
        "pump_start_count": int(sum(item["pump_start_count"] for item in cycles)),
        "pump_stop_count": int(sum(item["pump_stop_count"] for item in cycles)),
        "pump_direction_switch_count": int(
            sum(item["pump_direction_switch_count"] for item in cycles)
        ),
        "maximum_abs_roll_pitch_deg": peak.tolist(),
        "dominant_tilt_rms_deg": float(np.sqrt(squared_integral / duration_s)),
        "dominant_tilt_time_above_deg_s": threshold_times,
        "dominant_tilt_time_above_deg_percent": {
            str(threshold): 100.0 * threshold_times[str(threshold)] / duration_s
            for threshold in POSTURE_THRESHOLDS_DEG
        },
        "solver_nonconvergence_count": int(
            sum(not item["solver_converged"] for item in cycles)
        ),
        "reconstructed_plan_rejection_count": int(
            sum(
                item["solver_converged"] and not item["plan_success"]
                for item in cycles
            )
        ),
        "degraded_selection_count": int(
            sum(item["degraded_selection"] for item in cycles)
        ),
        # Compatibility aliases for the existing screening utilities. Their
        # precise meanings are recorded above and must not be conflated.
        "plan_failure_count": int(sum(not item["plan_success"] for item in cycles)),
        "fallback_count": int(sum(item["used_fallback"] for item in cycles)),
        "selected_precheck_failure_count": int(
            sum(not item["selected_posture_precheck_passed"] for item in cycles)
        ),
        "no_safe_precheck_candidate_count": int(
            sum(item["no_safe_precheck_candidate"] for item in cycles)
        ),
        "actual_posture_limit_failure_count": int(
            sum(
                not item["actual_sampled_posture_limit_satisfied"]
                for item in cycles
            )
        ),
        "working_model_scope_failure_count": int(
            sum(
                not item["actual_execution_within_working_model_scope"]
                for item in cycles
            )
        ),
        "total_mass_scope_failure_count": int(
            sum(not item["total_mass_within_scope"] for item in cycles)
        ),
        "maximum_abs_total_mass_deviation_from_reference_kg": float(
            max(
                item["maximum_abs_total_mass_deviation_from_reference_kg"]
                for item in cycles
            )
        ),
        "solver_iterations_mean": float(np.mean(solver_iterations)),
        "solver_iterations_p95": float(np.percentile(solver_iterations, 95.0)),
        "solver_iterations_maximum": int(np.max(solver_iterations)),
        "maximum_scaled_constraint_violation": float(
            max(item["maximum_scaled_constraint_violation"] for item in cycles)
        ),
        "selected_request_source_counts": dict(
            Counter(item["selected_request_source"] for item in cycles)
        ),
        "planner_objective_breakdown": _aggregate_objective_breakdown(cycles),
        "planned_first_block_absolute_tank_throughput_kg": float(
            sum(
                item["planned_absolute_tank_throughput_kg_by_block"][0]
                for item in cycles
            )
        ),
        "planned_horizon_absolute_tank_throughput_kg": float(
            sum(
                sum(item["planned_absolute_tank_throughput_kg_by_block"])
                for item in cycles
            )
        ),
    }


def _baseline_cycle_record(
    *,
    path: Any,
    decision: Any,
    plant_rotor_load: Any,
    reference_tank_masses_kg: np.ndarray,
    total_mass_scope_limit_kg: float,
) -> dict[str, Any]:
    posture = np.vstack(
        (
            path.substeps[0].start_platform_state.position[[3, 4]],
            *(
                step.next_platform_state.position[[3, 4]]
                for step in path.substeps
            ),
        )
    )
    durations = np.asarray([step.duration_s for step in path.substeps], dtype=float)
    dominant_tilt_deg = np.rad2deg(
        np.max(np.abs(posture[:-1]), axis=1)
    )
    model_scope_deg = np.rad2deg(
        research_preview_mpc_design_v4().maximum_abs_working_model_scope_roll_pitch_rad
    )
    tank_mass_states = np.vstack(
        [path.start_actual_tank_masses_kg]
        + [step.next_execution_state.actual_masses_kg for step in path.substeps]
    )
    maximum_abs_total_mass_deviation_kg = float(
        np.max(
            np.abs(
                np.sum(
                    tank_mass_states
                    - np.asarray(reference_tank_masses_kg, dtype=float),
                    axis=1,
                )
            )
        )
    )
    return {
        "controller_kind": "current_posture_feedback",
        "duration_s": float(path.duration_s),
        "committed_target_masses_kg": (
            path.substeps[0].execution_step.requested_target_kg.tolist()
        ),
        "actual_first_block_tank_delta_kg": (
            path.actual_tank_mass_delta_kg.tolist()
        ),
        "transferred_volume_m3": path.transferred_volume_m3,
        "aggregate_pump_active_time_s": path.aggregate_pump_active_time_s,
        "pump_start_count": path.pump_start_count,
        "pump_stop_count": path.pump_stop_count,
        "pump_direction_switch_count": path.pump_direction_switch_count,
        "peak_abs_roll_pitch_deg": np.rad2deg(
            np.max(np.abs(posture), axis=0)
        ).tolist(),
        "end_roll_pitch_deg": np.rad2deg(posture[-1]).tolist(),
        "dominant_tilt_squared_integral_deg2_s": float(
            np.sum(dominant_tilt_deg**2 * durations)
        ),
        "dominant_tilt_time_above_deg_s": {
            str(threshold): float(
                np.sum(durations[dominant_tilt_deg > threshold])
            )
            for threshold in POSTURE_THRESHOLDS_DEG
        },
        "plant_generalized_rotor_load_platform": (
            plant_rotor_load.generalized_load_platform.tolist()
        ),
        "actual_execution_within_working_model_scope": bool(
            np.all(np.max(np.abs(np.rad2deg(posture)), axis=0) <= model_scope_deg)
        ),
        "total_mass_scope_limit_kg": float(total_mass_scope_limit_kg),
        "maximum_abs_total_mass_deviation_from_reference_kg": (
            maximum_abs_total_mass_deviation_kg
        ),
        "total_mass_within_scope": bool(
            maximum_abs_total_mass_deviation_kg
            <= float(total_mass_scope_limit_kg) + 1.0e-9
        ),
        "feedback_active_axes": list(decision.active_axes),
        "requested_ballast_pitch_roll_moment_nm": (
            decision.requested_ballast_pitch_roll_moment_nm.tolist()
        ),
        "target_operation": decision.execution_request.operation.value,
    }


def _aggregate_baseline(cycles: list[dict[str, Any]]) -> dict[str, Any]:
    duration_s = float(sum(item["duration_s"] for item in cycles))
    squared_integral = float(
        sum(item["dominant_tilt_squared_integral_deg2_s"] for item in cycles)
    )
    threshold_times = {
        str(threshold): float(
            sum(
                item["dominant_tilt_time_above_deg_s"][str(threshold)]
                for item in cycles
            )
        )
        for threshold in POSTURE_THRESHOLDS_DEG
    }
    return {
        "completed_cycle_count": len(cycles),
        "continuous_duration_h": duration_s / 3600.0,
        "transferred_volume_m3": float(
            sum(item["transferred_volume_m3"] for item in cycles)
        ),
        "aggregate_pump_active_time_s": float(
            sum(item["aggregate_pump_active_time_s"] for item in cycles)
        ),
        "pump_start_count": int(sum(item["pump_start_count"] for item in cycles)),
        "pump_stop_count": int(sum(item["pump_stop_count"] for item in cycles)),
        "pump_direction_switch_count": int(
            sum(item["pump_direction_switch_count"] for item in cycles)
        ),
        "maximum_abs_roll_pitch_deg": np.max(
            np.asarray([item["peak_abs_roll_pitch_deg"] for item in cycles]),
            axis=0,
        ).tolist(),
        "dominant_tilt_rms_deg": float(np.sqrt(squared_integral / duration_s)),
        "dominant_tilt_time_above_deg_s": threshold_times,
        "dominant_tilt_time_above_deg_percent": {
            str(threshold): 100.0 * threshold_times[str(threshold)] / duration_s
            for threshold in POSTURE_THRESHOLDS_DEG
        },
        "working_model_scope_failure_count": int(
            sum(
                not item["actual_execution_within_working_model_scope"]
                for item in cycles
            )
        ),
        "total_mass_scope_failure_count": int(
            sum(not item["total_mass_within_scope"] for item in cycles)
        ),
        "maximum_abs_total_mass_deviation_from_reference_kg": float(
            max(
                item["maximum_abs_total_mass_deviation_from_reference_kg"]
                for item in cycles
            )
        ),
        "feedback_active_cycle_count": int(
            sum(any(item["feedback_active_axes"]) for item in cycles)
        ),
        "target_operation_counts": dict(
            Counter(item["target_operation"] for item in cycles)
        ),
    }


def _limit_feedback_decision_to_total_mass_scope(
    *,
    decision: Any,
    execution_state: Any,
    execution_config_value: Any,
    reference_tank_masses_kg: np.ndarray,
    total_mass_scope_limit_kg: float,
) -> tuple[Any, dict[str, Any]]:
    """Back off only when the shared pump preview exceeds total-mass scope."""

    request = decision.execution_request
    if request.target_masses_kg is None:
        return decision, {
            "applied": False,
            "reason": "release_request_has_no_new_target",
            "scale": 1.0,
        }
    initial_offset = float(
        np.sum(execution_state.actual_masses_kg)
        - np.sum(reference_tank_masses_kg)
    )

    def preview(candidate_request: ExecutionRolloutRequest) -> tuple[Any, float]:
        result = simulate_execution_step(
            execution_state,
            candidate_request,
            execution_config_value,
        )
        maximum_abs_offset = max(
            abs(initial_offset + result.minimum_total_mass_delta_from_initial_kg),
            abs(initial_offset + result.maximum_total_mass_delta_from_initial_kg),
        )
        return result, float(maximum_abs_offset)

    _, unscaled_peak = preview(request)
    if unscaled_peak <= total_mass_scope_limit_kg + 1.0e-9:
        return decision, {
            "applied": False,
            "reason": "unscaled_request_within_scope",
            "scale": 1.0,
            "unscaled_peak_abs_total_mass_offset_kg": unscaled_peak,
            "scaled_peak_abs_total_mass_offset_kg": unscaled_peak,
        }
    actual_masses = np.asarray(execution_state.actual_masses_kg, dtype=float)
    target = np.asarray(request.target_masses_kg, dtype=float)
    target_delta = target - actual_masses

    def request_at_scale(scale_value: float) -> ExecutionRolloutRequest:
        return ExecutionRolloutRequest.track(
            actual_masses + float(scale_value) * target_delta
        )

    _, zero_scale_peak = preview(request_at_scale(0.0))
    if zero_scale_peak > total_mass_scope_limit_kg + 1.0e-9:
        return decision, {
            "applied": False,
            "reason": "existing_pump_state_cannot_be_recovered_by_target_backoff",
            "scale": 1.0,
            "unscaled_peak_abs_total_mass_offset_kg": unscaled_peak,
            "zero_scale_peak_abs_total_mass_offset_kg": zero_scale_peak,
            "scope_limit_kg": float(total_mass_scope_limit_kg),
        }
    lower = 0.0
    upper = 1.0
    scaled_peak = zero_scale_peak
    for _ in range(12):
        midpoint = 0.5 * (lower + upper)
        _, midpoint_peak = preview(request_at_scale(midpoint))
        if midpoint_peak <= total_mass_scope_limit_kg + 1.0e-9:
            lower = midpoint
            scaled_peak = midpoint_peak
        else:
            upper = midpoint
    scale = lower
    scaled_request = request_at_scale(scale)
    scaled_target = np.asarray(scaled_request.target_masses_kg, dtype=float)
    scaled_decision = replace(
        decision,
        block_reachability_scale=decision.block_reachability_scale * scale,
        reachable_target_tank_masses_kg=scaled_target,
        execution_request=scaled_request,
    )
    return scaled_decision, {
        "applied": True,
        "reason": "shared_pump_preview_exceeded_total_mass_scope",
        "scale": scale,
        "unscaled_peak_abs_total_mass_offset_kg": unscaled_peak,
        "scaled_peak_abs_total_mass_offset_kg": scaled_peak,
        "scope_limit_kg": float(total_mass_scope_limit_kg),
    }


def _rotor_operating_domain_termination(
    *,
    cycle_index: int,
    stage: str,
    platform_state: IncrementalState,
    error: ValueError,
) -> dict[str, Any]:
    """Describe an out-of-domain rotor conversion without aborting other variants."""

    return {
        "cycle_index": int(cycle_index),
        "reason": "rotor_operating_model_out_of_domain",
        "stage": str(stage),
        "message": str(error),
        "platform_position": np.asarray(
            platform_state.position,
            dtype=float,
        ).tolist(),
        "platform_velocity": np.asarray(
            platform_state.velocity,
            dtype=float,
        ).tolist(),
        "platform_roll_pitch_deg": np.rad2deg(
            np.asarray(platform_state.position, dtype=float)[[3, 4]]
        ).tolist(),
        "platform_surge_sway_velocity_m_s": np.asarray(
            platform_state.velocity,
            dtype=float,
        )[[0, 1]].tolist(),
    }


def run_posture_feedback_variant(
    *,
    source_records: list[RealLstmWindRecord],
    resources: RealLstmPreviewResources,
    runtime: Any,
    warmup_block_count: int = 0,
    cycle_count: int = CYCLE_COUNT,
) -> dict[str, Any]:
    """Run the current-posture baseline through the shared physical plant."""

    resolved_cycle_count = int(cycle_count)
    if resolved_cycle_count <= 0 or resolved_cycle_count != cycle_count:
        raise ValueError("cycle_count must be a positive integer")
    if len(source_records) < int(warmup_block_count) + resolved_cycle_count + 1:
        raise ValueError("source_records do not cover the baseline window")
    platform_state, execution_state, preconditioning = _precondition_window_initial_state(
        source_records=source_records,
        resources=resources,
        warmup_block_count=warmup_block_count,
        runtime=runtime,
    )
    design = research_posture_feedback_baseline_v2()
    capacities = np.full(3, TANK_CAPACITY_KG)
    cycles: list[dict[str, Any]] = []
    plant_loads: list[np.ndarray] = []
    termination: dict[str, Any] | None = None
    for cycle_index in range(resolved_cycle_count):
        source_index = int(warmup_block_count) + cycle_index
        current_record = source_records[source_index]
        duration_s = float(current_record.forecast.sample_period_s)
        config = execution_config(block_duration_s=duration_s)
        substep_count = int(np.ceil(duration_s / config.internal_step_s))
        try:
            realised_loads = assemble_recorded_interval_rotor_load_substeps(
                resources=resources,
                current_record=current_record,
                following_record=source_records[source_index + 1],
                platform_state=platform_state,
                substep_count=substep_count,
            )
        except ValueError as exc:
            if "forecast leaves the nominal below-rated rotor operating domain" not in str(exc):
                raise
            termination = _rotor_operating_domain_termination(
                cycle_index=cycle_index,
                stage="recorded_wind_plant_load",
                platform_state=platform_state,
                error=exc,
            )
            break
        average_load = average_rotor_loads(realised_loads)
        snapshot = assemble_ballast_model_snapshot(
            runtime_assembly=runtime,
            actual_tank_masses_kg=execution_state.actual_masses_kg,
            reference_tank_masses_kg=REFERENCE_TANK_MASSES_KG,
            tank_capacities_kg=capacities,
            tank_coordinates_m=TANK_COORDINATES_M,
        )
        mpc_design = research_preview_mpc_design_v4()
        total_mass_scope_limit_kg = float(
            snapshot.matrices.restoring_stiffness[2, 2]
            * mpc_design.maximum_abs_total_tank_mass_equivalent_heave_m
            / runtime.gravity_m_s2
        )
        decision = decide_posture_feedback_target(
            platform_snapshot=snapshot,
            platform_state=platform_state,
            design=design,
            maximum_abs_tank_mass_change_per_block_kg=np.full(
                3,
                max(rate for _, rate in config.pump_rate_schedule_m3_min)
                * duration_s
                / 60.0
                * config.water_density_kg_m3,
            ),
        )
        decision, total_mass_precheck = _limit_feedback_decision_to_total_mass_scope(
            decision=decision,
            execution_state=execution_state,
            execution_config_value=config,
            reference_tank_masses_kg=REFERENCE_TANK_MASSES_KG,
            total_mass_scope_limit_kg=total_mass_scope_limit_kg,
        )
        path = advance_physical_execution_platform_path(
            platform_state=platform_state,
            execution_state=execution_state,
            execution_request=decision.execution_request,
            execution_config=config,
            runtime_assembly=runtime,
            reference_tank_masses_kg=REFERENCE_TANK_MASSES_KG,
            tank_capacities_kg=capacities,
            tank_coordinates_m=TANK_COORDINATES_M,
            rotor_load=average_load,
            rotor_load_substeps=realised_loads,
            wave_load=np.zeros(6),
            other_load=np.zeros(6),
            duration_s=duration_s,
        )
        cycle = _baseline_cycle_record(
            path=path,
            decision=decision,
            plant_rotor_load=average_load,
            reference_tank_masses_kg=REFERENCE_TANK_MASSES_KG,
            total_mass_scope_limit_kg=total_mass_scope_limit_kg,
        )
        cycle["cycle_index"] = cycle_index
        cycle["total_mass_precheck"] = total_mass_precheck
        if (
            not cycle["actual_execution_within_working_model_scope"]
            or not cycle["total_mass_within_scope"]
        ):
            termination = {
                "cycle_index": cycle_index,
                "reason": (
                    "realised_execution_exceeded_working_model_scope"
                    if not cycle["actual_execution_within_working_model_scope"]
                    else "realised_execution_exceeded_total_mass_scope"
                ),
                "cycle_diagnostic": cycle,
            }
            break
        cycles.append(cycle)
        plant_loads.extend(
            np.asarray(load.generalized_load_platform, dtype=float)
            for load in realised_loads
        )
        platform_state = path.final_platform_state
        execution_state = path.final_execution_state

    completed = len(cycles) == resolved_cycle_count
    return {
        "variant": POSTURE_FEEDBACK_BASELINE,
        "controller_kind": "current_posture_feedback",
        "is_noncausal_diagnostic": False,
        "plant_load_sha256": _load_digest(plant_loads),
        "source_record_sequence_sha256": _source_record_sequence_digest(
            source_records
        ),
        "preconditioning": preconditioning,
        "completed_requested_window": completed,
        "aggregate_scope": (
            "complete_requested_window"
            if completed
            else "diagnostic_prefix_only"
            if cycles
            else "no_executed_cycles"
        ),
        "aggregate_eligible_for_economic_comparison": completed,
        "requested_cycle_count": resolved_cycle_count,
        "termination": termination,
        "posture_feedback_design": design.as_dict(),
        "aggregate": _aggregate_baseline(cycles) if cycles else None,
        "cycles": cycles,
    }
def run_variant(
    *,
    name: str,
    planner_forecast_mode: str,
    planner_horizon_blocks: int,
    source_records: list[RealLstmWindRecord],
    resources: RealLstmPreviewResources,
    runtime: Any,
    modes: ThreeTankDifferentialModes,
    objective_priority_multipliers: dict[str, float] | None = None,
    warmup_block_count: int = 0,
    cycle_count: int = CYCLE_COUNT,
) -> dict[str, Any]:
    if int(cycle_count) != cycle_count or cycle_count <= 0:
        raise ValueError("cycle_count must be a positive integer")
    resolved_cycle_count = int(cycle_count)
    required_source_records = (
        int(warmup_block_count)
        + resolved_cycle_count
        + (planner_horizon_blocks if planner_forecast_mode == "recorded_oracle" else 1)
    )
    if len(source_records) < required_source_records:
        raise ValueError(
            "source_records do not cover the requested control window and lookahead"
        )
    platform_state, execution_state, preconditioning = _precondition_window_initial_state(
        source_records=source_records,
        resources=resources,
        warmup_block_count=warmup_block_count,
        runtime=runtime,
    )
    design_record = research_preview_mpc_design_v4().with_priority_multipliers(
        objective_priority_multipliers
    ).as_dict()
    cycles: list[dict[str, Any]] = []
    plant_loads: list[np.ndarray] = []
    termination: dict[str, Any] | None = None
    for cycle_index in range(resolved_cycle_count):
        source_index = int(warmup_block_count) + cycle_index
        current_record = source_records[source_index]
        duration_s = float(current_record.forecast.sample_period_s)
        config = execution_config(block_duration_s=duration_s)
        substep_count = int(np.ceil(duration_s / config.internal_step_s))
        try:
            realised_loads = assemble_recorded_interval_rotor_load_substeps(
                resources=resources,
                current_record=current_record,
                following_record=source_records[source_index + 1],
                platform_state=platform_state,
                substep_count=substep_count,
            )
        except ValueError as exc:
            if "forecast leaves the nominal below-rated rotor operating domain" not in str(exc):
                raise
            termination = _rotor_operating_domain_termination(
                cycle_index=cycle_index,
                stage="recorded_wind_plant_load",
                platform_state=platform_state,
                error=exc,
            )
            break
        plant_rotor_load = average_rotor_loads(realised_loads)
        oracle_source_records = None
        oracle_records = None
        if planner_forecast_mode == "recorded_oracle":
            oracle_source_records = tuple(
                source_records[index]
                for index in range(
                    source_index + 1,
                    source_index + 1 + planner_horizon_blocks,
                )
            )
            oracle_records = tuple(
                {
                    "recorded_observation_time": str(record.forecast.origin_time),
                    "enu_downwind_wind_mps": (
                        record.current_enu_downwind_wind_mps.tolist()
                    ),
                }
                for record in oracle_source_records
            )
        try:
            planner_preview = assemble_planner_rotor_preview_from_record(
                resources=resources,
                source_record=current_record,
                platform_state=platform_state,
                planner_forecast_mode=planner_forecast_mode,  # type: ignore[arg-type]
                horizon_blocks=planner_horizon_blocks,
                oracle_records=oracle_source_records,
            )
        except ValueError as exc:
            if "forecast leaves the nominal below-rated rotor operating domain" not in str(exc):
                raise
            termination = _rotor_operating_domain_termination(
                cycle_index=cycle_index,
                stage="planner_preview",
                platform_state=platform_state,
                error=exc,
            )
            break
        try:
            cycle, platform_state, execution_state = run_preview_mpc_cycle(
                source_preview=planner_preview,
                runtime=runtime,
                modes=modes,
                platform_state=platform_state,
                execution_state=execution_state,
                objective_priority_multipliers=objective_priority_multipliers,
                planner_forecast_mode=planner_forecast_mode,  # type: ignore[arg-type]
                planner_horizon_blocks=planner_horizon_blocks,
                planner_oracle_future_wind_records=oracle_records,
                plant_first_block_rotor_load=plant_rotor_load,
                plant_first_block_rotor_load_substeps=realised_loads,
                plant_load_source=(
                    "recorded_replay_linear_enu_substep_reconstruction_"
                    "converted_under_each_variant_cycle_initial_platform_state"
                ),
            )
        except PreviewMPCNoSafeCandidateError as exc:
            termination = {
                "cycle_index": cycle_index,
                "reason": "no_planner_load_safe_model_supported_candidate",
                "message": str(exc),
                "candidates": [
                    {
                        "lifecycle": item.lifecycle.value,
                        "posture_precheck_passed": (
                            item.physical_precheck.posture_precheck_passed
                        ),
                        "maximum_normalized_posture": (
                            item.physical_precheck.maximum_normalized_posture
                        ),
                        "first_block_within_working_model_scope": (
                            item.first_block_within_working_model_scope
                        ),
                        "first_block_maximum_normalized_working_model_scope": (
                            item.first_block_maximum_normalized_working_model_scope
                        ),
                        "horizon_within_working_model_scope": (
                            item.horizon_within_working_model_scope
                        ),
                        "maximum_normalized_working_model_scope": (
                            item.maximum_normalized_working_model_scope
                        ),
                        "rejection_reason": item.rejection_reason,
                    }
                    for item in exc.candidates
                ],
            }
            break
        cycle["cycle_index"] = cycle_index
        compact = _compact_cycle(cycle)
        if not compact["actual_execution_within_working_model_scope"]:
            termination = {
                "cycle_index": cycle_index,
                "reason": "realised_execution_exceeded_working_model_scope",
                "message": (
                    "the recorded-wind plant rollout left the declared low-order "
                    "working-model scope; this cycle is retained only as a "
                    "diagnostic and is excluded from performance aggregates"
                ),
                "cycle_diagnostic": compact,
            }
            break
        cycles.append(compact)
        if (
            not compact["selected_posture_precheck_passed"]
            or not compact["actual_sampled_posture_limit_satisfied"]
        ):
            raise RuntimeError(
                "six-hour variant failed the planner-load precheck or realised "
                f"posture limit at cycle {cycle_index}"
            )
        plant_loads.extend(
            np.asarray(load.generalized_load_platform, dtype=float)
            for load in realised_loads
        )
    design_identities = {
        (
            item["preview_mpc_design_identity"],
            item["preview_mpc_design_status"],
            item["preview_mpc_design_parameter_sha256"],
        )
        for item in cycles
    }
    if cycles and len(design_identities) != 1:
        raise RuntimeError("MPC design identity changed within one rolling variant")
    priority_configurations = {
        json.dumps(
            item["normalized_objective_priorities"],
            sort_keys=True,
            separators=(",", ":"),
        )
        for item in cycles
    }
    if cycles and len(priority_configurations) != 1:
        raise RuntimeError("objective priorities changed within one rolling variant")
    multiplier_configurations = {
        json.dumps(
            item["objective_priority_multipliers"],
            sort_keys=True,
            separators=(",", ":"),
        )
        for item in cycles
    }
    if cycles and len(multiplier_configurations) != 1:
        raise RuntimeError("objective multipliers changed within one rolling variant")
    completed_requested_window = len(cycles) == resolved_cycle_count
    return {
        "variant": name,
        "planner_forecast_mode": planner_forecast_mode,
        "planner_horizon_blocks": planner_horizon_blocks,
        "is_noncausal_diagnostic": planner_forecast_mode == "recorded_oracle",
        "plant_load_sha256": _load_digest(plant_loads),
        "plant_load_hash_scope": (
            "state_dependent_rotor_generalized_load_sequence_under_shared_"
            "recorded_wind"
        ),
        "source_record_sequence_sha256": _source_record_sequence_digest(
            source_records
        ),
        "preconditioning": preconditioning,
        "completed_requested_window": completed_requested_window,
        "aggregate_scope": (
            "complete_requested_window"
            if completed_requested_window
            else "diagnostic_prefix_only"
            if cycles
            else "no_executed_cycles"
        ),
        "aggregate_eligible_for_economic_comparison": (
            completed_requested_window
        ),
        "requested_cycle_count": resolved_cycle_count,
        "termination": termination,
        "preview_mpc_design_identity": design_record["identity"],
        "preview_mpc_design_status": design_record["status"],
        "preview_mpc_design_parameter_sha256": design_record[
            "parameter_sha256"
        ],
        "normalized_objective_priorities": design_record[
            "normalized_objective_priorities"
        ],
        "objective_priority_multipliers": dict(
            objective_priority_multipliers or {}
        ),
        "aggregate": _aggregate(cycles) if cycles else None,
        "cycles": cycles,
    }


def _relative_change(reference: float, alternative: float) -> float | None:
    if abs(reference) <= 1.0e-12:
        return None
    return 100.0 * (alternative - reference) / reference


def _contrast(reference: dict[str, Any], alternative: dict[str, Any]) -> dict[str, Any]:
    if not (
        reference.get("aggregate_eligible_for_economic_comparison", False)
        and alternative.get("aggregate_eligible_for_economic_comparison", False)
    ):
        raise ValueError(
            "economic contrast requires two complete requested-window aggregates"
        )
    ref = reference["aggregate"]
    alt = alternative["aggregate"]
    reference_targets = np.asarray(
        [item["committed_target_masses_kg"] for item in reference["cycles"]],
        dtype=float,
    )
    alternative_targets = np.asarray(
        [item["committed_target_masses_kg"] for item in alternative["cycles"]],
        dtype=float,
    )
    target_difference = np.linalg.norm(alternative_targets - reference_targets, axis=1)
    return {
        "reference_variant": reference["variant"],
        "alternative_variant": alternative["variant"],
        "interpretation": CONTRAST_INTERPRETATIONS[
            (reference["variant"], alternative["variant"])
        ],
        "difference_convention": "alternative_minus_reference",
        "transferred_volume_difference_m3": float(
            alt["transferred_volume_m3"] - ref["transferred_volume_m3"]
        ),
        "transferred_volume_relative_change_percent": _relative_change(
            ref["transferred_volume_m3"], alt["transferred_volume_m3"]
        ),
        "pump_active_time_difference_s": float(
            alt["aggregate_pump_active_time_s"]
            - ref["aggregate_pump_active_time_s"]
        ),
        "pump_start_count_difference": int(
            alt["pump_start_count"] - ref["pump_start_count"]
        ),
        "dominant_tilt_rms_difference_deg": float(
            alt["dominant_tilt_rms_deg"] - ref["dominant_tilt_rms_deg"]
        ),
        "dominant_tilt_time_above_deg_difference_s": {
            str(threshold): float(
                alt["dominant_tilt_time_above_deg_s"][str(threshold)]
                - ref["dominant_tilt_time_above_deg_s"][str(threshold)]
            )
            for threshold in POSTURE_THRESHOLDS_DEG
        },
        "cycles_with_different_committed_target_over_1kg": int(
            np.count_nonzero(target_difference > 1.0)
        ),
        "maximum_committed_target_difference_kg": float(
            np.max(target_difference, initial=0.0)
        ),
    }


def _completed_contrast_pairs(
    by_name: dict[str, dict[str, Any]],
) -> tuple[tuple[str, str], ...]:
    """Return contrasts whose two variants completed the requested window."""

    return tuple(
        (reference, alternative)
        for reference, alternative in CONTRASTS
        if reference in by_name
        and alternative in by_name
        and by_name[reference]["completed_requested_window"]
        and by_name[alternative]["completed_requested_window"]
    )


def _summarize_cases(
    cases: list[dict[str, Any]],
    *,
    variant_names: tuple[str, ...] | None = None,
    contrasts: tuple[tuple[str, str], ...] | None = None,
) -> dict[str, Any]:
    selected_variant_names = (
        ALL_VARIANT_NAMES
        if variant_names is None
        else tuple(variant_names)
    )
    selected_contrasts = CONTRASTS if contrasts is None else tuple(contrasts)
    if not cases:
        return {
            "status": "no_case_completed_the_requested_window",
            "by_variant": {},
            "contrasts": [],
        }
    by_variant: dict[str, dict[str, Any]] = {}
    for name in selected_variant_names:
        aggregates = [
            next(
                item["aggregate"]
                for item in case["variants"]
                if item["variant"] == name
            )
            for case in cases
        ]
        duration_s = float(
            sum(item["continuous_duration_h"] * 3600.0 for item in aggregates)
        )
        squared_integral = float(
            sum(
                item["dominant_tilt_rms_deg"] ** 2
                * item["continuous_duration_h"]
                * 3600.0
                for item in aggregates
            )
        )
        summary = {
            "case_count": len(aggregates),
            "continuous_duration_h": duration_s / 3600.0,
            "transferred_volume_m3": float(
                sum(item["transferred_volume_m3"] for item in aggregates)
            ),
            "aggregate_pump_active_time_s": float(
                sum(item["aggregate_pump_active_time_s"] for item in aggregates)
            ),
            "pump_start_count": int(
                sum(item["pump_start_count"] for item in aggregates)
            ),
            "pump_stop_count": int(
                sum(item["pump_stop_count"] for item in aggregates)
            ),
            "maximum_abs_roll_pitch_deg": np.max(
                np.asarray(
                    [item["maximum_abs_roll_pitch_deg"] for item in aggregates],
                    dtype=float,
                ),
                axis=0,
            ).tolist(),
            "dominant_tilt_rms_deg": float(
                np.sqrt(squared_integral / duration_s)
            ),
            "dominant_tilt_time_above_deg_s": {
                str(threshold): float(
                    sum(
                        item["dominant_tilt_time_above_deg_s"][str(threshold)]
                        for item in aggregates
                    )
                )
                for threshold in POSTURE_THRESHOLDS_DEG
            },
        }
        if all("plan_failure_count" in item for item in aggregates):
            summary.update({
            "solver_nonconvergence_count": int(
                sum(item["solver_nonconvergence_count"] for item in aggregates)
            ),
            "reconstructed_plan_rejection_count": int(
                sum(
                    item["reconstructed_plan_rejection_count"]
                    for item in aggregates
                )
            ),
            "degraded_selection_count": int(
                sum(item["degraded_selection_count"] for item in aggregates)
            ),
            "plan_failure_count": int(
                sum(item["plan_failure_count"] for item in aggregates)
            ),
            "fallback_count": int(sum(item["fallback_count"] for item in aggregates)),
            "selected_precheck_failure_count": int(
                sum(item["selected_precheck_failure_count"] for item in aggregates)
            ),
            "no_safe_precheck_candidate_count": int(
                sum(item["no_safe_precheck_candidate_count"] for item in aggregates)
            ),
            "actual_posture_limit_failure_count": int(
                sum(item["actual_posture_limit_failure_count"] for item in aggregates)
            ),
            "planner_objective_breakdown": _combine_objective_breakdowns(
                [item["planner_objective_breakdown"] for item in aggregates]
            ),
            "planned_first_block_absolute_tank_throughput_kg": float(
                sum(
                    item["planned_first_block_absolute_tank_throughput_kg"]
                    for item in aggregates
                )
            ),
            "planned_horizon_absolute_tank_throughput_kg": float(
                sum(
                    item["planned_horizon_absolute_tank_throughput_kg"]
                    for item in aggregates
                )
            ),
            })
        else:
            summary.update({
                "feedback_active_cycle_count": int(
                    sum(item["feedback_active_cycle_count"] for item in aggregates)
                ),
                "working_model_scope_failure_count": int(
                    sum(item["working_model_scope_failure_count"] for item in aggregates)
                ),
            })
        by_variant[name] = summary

    def aggregate_contrast(reference_name: str, alternative_name: str) -> dict[str, Any]:
        reference = by_variant[reference_name]
        alternative = by_variant[alternative_name]
        return {
            "reference_variant": reference_name,
            "alternative_variant": alternative_name,
            "interpretation": CONTRAST_INTERPRETATIONS[
                (reference_name, alternative_name)
            ],
            "difference_convention": "alternative_minus_reference",
            "transferred_volume_difference_m3": float(
                alternative["transferred_volume_m3"]
                - reference["transferred_volume_m3"]
            ),
            "transferred_volume_relative_change_percent": _relative_change(
                reference["transferred_volume_m3"],
                alternative["transferred_volume_m3"],
            ),
            "pump_active_time_difference_s": float(
                alternative["aggregate_pump_active_time_s"]
                - reference["aggregate_pump_active_time_s"]
            ),
            "pump_start_count_difference": int(
                alternative["pump_start_count"] - reference["pump_start_count"]
            ),
            "dominant_tilt_rms_difference_deg": float(
                alternative["dominant_tilt_rms_deg"]
                - reference["dominant_tilt_rms_deg"]
            ),
            "dominant_tilt_time_above_deg_difference_s": {
                str(threshold): float(
                    alternative["dominant_tilt_time_above_deg_s"][str(threshold)]
                    - reference["dominant_tilt_time_above_deg_s"][str(threshold)]
                )
                for threshold in POSTURE_THRESHOLDS_DEG
            },
            "cases_with_lower_alternative_volume": int(
                sum(
                    next(
                        item["aggregate"]["transferred_volume_m3"]
                        for item in case["variants"]
                        if item["variant"] == alternative_name
                    )
                    < next(
                        item["aggregate"]["transferred_volume_m3"]
                        for item in case["variants"]
                        if item["variant"] == reference_name
                    )
                    for case in cases
                )
            ),
        }

    return {
        "by_variant": by_variant,
        "contrasts": [
            aggregate_contrast(reference, alternative)
            for reference, alternative in selected_contrasts
        ],
    }


def _complete_cases_for_contrast(
    cases: list[dict[str, Any]],
    reference: str,
    alternative: str,
) -> list[dict[str, Any]]:
    """Keep cases completed by the two variants being compared."""

    return [
        case
        for case in cases
        if case["completed_requested_window_by_variant"][reference]
        and case["completed_requested_window_by_variant"][alternative]
    ]


def run_comparison(
    *,
    origins: tuple[str, ...] = FIXED_CASE_ORIGINS,
    device: str = "cpu",
    cycle_count: int = CYCLE_COUNT,
    warmup_block_count: int = WARMUP_BLOCK_COUNT,
) -> dict[str, Any]:
    if int(cycle_count) != cycle_count or cycle_count <= 0:
        raise ValueError("cycle_count must be a positive integer")
    resolved_cycle_count = int(cycle_count)
    if (
        int(warmup_block_count) != warmup_block_count
        or warmup_block_count < 0
    ):
        raise ValueError("warmup_block_count must be a non-negative integer")
    resolved_warmup_count = int(warmup_block_count)
    uses_predeclared_cases = (
        tuple(origins) == FIXED_CASE_ORIGINS
        and resolved_cycle_count == CYCLE_COUNT
        and resolved_warmup_count == WARMUP_BLOCK_COUNT
    )
    runtime = research_runtime_assembly()
    modes = ThreeTankDifferentialModes(
        tank_coordinates_m=TANK_COORDINATES_M,
        gravity_m_s2=runtime.gravity_m_s2,
    )
    resources = prepare_real_lstm_preview_resources(device=device)
    cases: list[dict[str, Any]] = []
    for origin_text in origins:
        origin = datetime.strptime(origin_text, TIMESTAMP_FMT)
        source_records = prepare_source_records(
            origin=origin,
            device=device,
            cycle_count=resolved_cycle_count,
            lookahead_count=max(horizon for _, _, horizon in VARIANTS),
            resources=resources,
            warmup_block_count=resolved_warmup_count,
        )
        domain_preflight = preflight_source_operating_domain(
            source_records=source_records,
            resources=resources,
            origin=origin,
            cycle_count=resolved_cycle_count,
            warmup_block_count=resolved_warmup_count,
        )
        if not domain_preflight["supported"]:
            raise RuntimeError(
                "fixed comparison case is outside the declared source-bound "
                f"rotor operating domain: {origin_text}"
            )
        variants = [
            run_posture_feedback_variant(
                source_records=source_records,
                resources=resources,
                runtime=runtime,
                warmup_block_count=resolved_warmup_count,
                cycle_count=resolved_cycle_count,
            ),
            *[
            run_variant(
                name=name,
                planner_forecast_mode=mode,
                planner_horizon_blocks=horizon,
                source_records=source_records,
                resources=resources,
                runtime=runtime,
                modes=modes,
                warmup_block_count=resolved_warmup_count,
                cycle_count=resolved_cycle_count,
            )
            for name, mode, horizon in VARIANTS
            ],
        ]
        source_digests = {
            variant["source_record_sequence_sha256"] for variant in variants
        }
        if len(source_digests) != 1:
            raise RuntimeError(
                "controller variants did not receive the same recorded-wind sequence"
            )
        by_name = {variant["variant"]: variant for variant in variants}
        completed_contrast_pairs = _completed_contrast_pairs(by_name)
        completed_requested_window = all(
            variant["completed_requested_window"] for variant in variants
        )
        cases.append(
            {
                "origin": origin_text,
                "duration_h": resolved_cycle_count * 600.0 / 3600.0,
                "shared_source_record_sequence_sha256": next(
                    iter(source_digests)
                ),
                "plant_load_sha256_by_variant": {
                    variant["variant"]: variant["plant_load_sha256"]
                    for variant in variants
                },
                "source_operating_domain_preflight": domain_preflight,
                "completed_requested_window": completed_requested_window,
                "completed_requested_window_by_variant": {
                    variant["variant"]: bool(
                        variant["completed_requested_window"]
                    )
                    for variant in variants
                },
                "variants": variants,
                "contrasts": [
                    _contrast(by_name[reference], by_name[alternative])
                    for reference, alternative in completed_contrast_pairs
                ],
            }
        )
    completed_cases = [
        case for case in cases if case["completed_requested_window"]
    ]
    pairwise_cross_case_summaries = []
    for reference, alternative in CONTRASTS:
        eligible_cases = _complete_cases_for_contrast(
            cases,
            reference,
            alternative,
        )
        pair_summary = _summarize_cases(
            eligible_cases,
            variant_names=(reference, alternative),
            contrasts=((reference, alternative),),
        )
        pairwise_cross_case_summaries.append(
            {
                "reference_variant": reference,
                "alternative_variant": alternative,
                "eligible_complete_case_count": len(eligible_cases),
                "excluded_case_count": len(cases) - len(eligible_cases),
                "summary": pair_summary,
            }
        )
    completed_case_count_by_variant = {
        name: sum(
            case["completed_requested_window_by_variant"][name]
            for case in cases
        )
        for name in ALL_VARIANT_NAMES
    }
    return {
        "purpose": (
            "fixed_case_six_hour_forecast_content_and_horizon_comparison_"
            "not_parameter_tuning"
            if uses_predeclared_cases
            else "caller_selected_debug_case_comparison_not_fixed_case_evidence"
        ),
        "boundaries": {
            "is_population_performance_claim": False,
            "is_dynamic_model_validated_for_performance_claim": False,
            "dynamic_model_evidence_status": (
                "source_bound_local_28p5s_wamit_radiation_damping_and_"
                "recorded_wind_substep_reconstruction_without_full_"
                "radiation_memory_or_viscous_damping_validation"
            ),
            "uses_fixed_six_hour_cases": uses_predeclared_cases,
            "control_cycle_count_per_case": resolved_cycle_count,
            "warmup_block_count": resolved_warmup_count,
            "controller_variants_share_identical_recorded_wind_sequence": True,
            "rotor_load_uses_each_variant_current_platform_state": True,
            "plant_load_definition": (
                "linear_enu_wind_reconstruction_at_physical_substep_midpoints_"
                "then_shared_nonlinear_rotor_load_conversion_under_each_"
                "variant_cycle_initial_platform_state"
            ),
            "future_prediction_used_only_by_planner": True,
            "recorded_oracle_is_noncausal_diagnostic_only": True,
            "recorded_oracle_is_not_a_formal_performance_upper_bound": True,
            "recorded_oracle_interpretation": (
                "perfect_future_information_value_within_the_frozen_controller_"
                "actuator_platform_model_and_case_set"
            ),
            "recorded_interval_load_used_only_after_request_selection": True,
            "candidate_precheck_load_source": (
                "controller_visible_first_interval_forecast_or_current_"
                "observation_persistence"
            ),
            "wave_and_other_loads": "explicit_zero",
            "window_preconditioning": (
                "local_static_initialization_then_candidate_independent_"
                "recorded_load_prefix_with_tanks_held"
                if resolved_warmup_count > 0
                else "candidate_independent_local_static_initialization"
            ),
            "preconditioning_excluded_from_six_hour_metrics": True,
            "future_platform_kinematics": "frozen_zero",
            "objective_weights_are_identical_between_variants": True,
            "economic_contrasts_require_both_variants_to_complete_the_window": True,
            "incomplete_prefix_metrics_are_diagnostic_only": True,
            "case_selection_basis": (
                "predeclared_from_recorded_wind_only: six-hour forward rotor-normal "
                "component remains within 5.3-8.0 m/s and varies over the window; "
                "no controller outcome used"
                if uses_predeclared_cases
                else "caller_supplied_debug_origins_not_predeclared_evidence"
            ),
        },
        "case_origins": list(origins),
        "case_count": len(origins),
        "completed_case_count": len(completed_cases),
        "completed_case_count_by_variant": completed_case_count_by_variant,
        "implementation_sha256": implementation_identity(),
        "variant_definitions": [
            {
                "name": POSTURE_FEEDBACK_BASELINE,
                "controller_kind": "current_posture_feedback",
                "decision_information": "current_posture_and_posture_rate_only",
            },
            *[
            {
                "name": name,
                "planner_forecast_mode": mode,
                "horizon_blocks": horizon,
                "decision_information": (
                    "current_observation_and_current_state_only"
                    if name == "current_observation_1block"
                    else "controller_visible_future_forecast"
                    if mode == "lstm"
                    else "current_observation_held_over_preview_horizon"
                    if mode == "persistence"
                    else "future_recorded_load_noncausal_diagnostic"
                ),
            }
            for name, mode, horizon in VARIANTS
            ],
        ],
        "all_variants_complete_cross_case_summary": _summarize_cases(
            completed_cases
        ),
        "pairwise_cross_case_summaries": pairwise_cross_case_summaries,
        "cases": cases,
    }
