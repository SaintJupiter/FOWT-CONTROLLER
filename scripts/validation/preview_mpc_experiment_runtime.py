#!/usr/bin/env python3
"""Run one source-bound real-LSTM forecast through preview planning and execution.

This is a semantic integration scenario, not a performance experiment.  Each
cycle binds one real-LSTM forecast directly to the current platform state,
converts it to a typed rotor-load preview, and submits only the selected first
tank target to the target-tracking pump model.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
import sys
from typing import Any, Callable, Literal

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SRC_DIRECTORY = ROOT / "src"
VALIDATION_DIRECTORY = Path(__file__).resolve().parent
for directory in (SRC_DIRECTORY, VALIDATION_DIRECTORY):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from fowt_platform import (
    BallastRuntimeAssembly,
    GeneralizedLoadForecast,
    IncrementalState,
    PlatformMatrices,
    RotorGeneralizedLoad,
    RotorNormalLoad,
    ThreeTankDifferentialModes,
    assemble_volturnus_static_restoring_aligned_runtime_assembly,
    load_local_radiation_damping,
)
from real_lstm_preview_fixture import (
    REFERENCE_MANIFEST,
    REFERENCE_TANK_MASSES_KG,
    TANK_CAPACITY_KG,
    TANK_COORDINATES_M,
    assemble_planner_rotor_preview_from_record,
    infer_real_lstm_wind_record,
    prepare_real_lstm_preview_resources,
)
from wind_prediction.execution_rollout import (
    ExecutionRolloutConfig,
    ExecutionRolloutState,
)
from wind_prediction.preview_mpc_application import (
    PreviewMPCApplication,
    PreviewMPCSourceIdentity,
)
from wind_prediction.preview_mpc_design import research_preview_mpc_design_v4
from wind_prediction.replay_dataset import TIMESTAMP_FMT
from wind_prediction.run_identity import sha256_json
from wind_prediction.source_bound_rotor_preview import (
    SourceBoundRotorPreview,
    endpoint_average_rotor_load,
)


LOCAL_RADIATION_DAMPING_PERIOD_S = 28.5
LOCAL_ROTATIONAL_EQUIVALENT_DAMPING_RATIO = 0.004


def research_runtime_assembly():
    """Build the bounded plant used by the continuous semantic experiment.

    WAMIT supplies the local radiation contribution at the pitch period.  The
    pitch free-decay record supplies a 0.4% apparent modal damping ratio.  That
    ratio is also applied to roll as an explicit three-column-hull symmetry
    approximation; it is not a six-DOF damping identification.
    """

    radiation = load_local_radiation_damping(
        REFERENCE_MANIFEST,
        requested_period_s=LOCAL_RADIATION_DAMPING_PERIOD_S,
    )
    radiation_only = assemble_volturnus_static_restoring_aligned_runtime_assembly(
        REFERENCE_MANIFEST,
        radiation.damping_matrix,
    )
    matrices = radiation_only.base_matrices
    damping = np.array(matrices.damping, dtype=float, copy=True)
    restoring = matrices.restoring_stiffness
    for index in (3, 4):
        target = 2.0 * LOCAL_ROTATIONAL_EQUIVALENT_DAMPING_RATIO * np.sqrt(
            matrices.mass[index, index] * restoring[index, index]
        )
        damping[index, index] = max(damping[index, index], float(target))
    return BallastRuntimeAssembly(
        base_matrices=PlatformMatrices(
            mass=matrices.mass,
            damping=damping,
            hydrostatic_stiffness=matrices.hydrostatic_stiffness,
            mooring_stiffness=matrices.mooring_stiffness,
            weight_stiffness=matrices.weight_stiffness,
        ),
        reference_mass_properties=radiation_only.reference_mass_properties,
        gravity_m_s2=radiation_only.gravity_m_s2,
        provenance=(
            "static_restoring_aligned_with_local_wamit_radiation_and_"
            "openfast_pitch_decay_equivalent_roll_pitch_damping"
        ),
    )


def execution_config(*, block_duration_s: float) -> ExecutionRolloutConfig:
    return ExecutionRolloutConfig(
        block_duration_s=block_duration_s,
        internal_step_s=5.0,
        water_density_kg_m3=1025.0,
        max_pump_rate_m3_min=1.0,
        tank_capacity_kg=TANK_CAPACITY_KG,
        stop_error_kg=300.0,
        restart_error_kg=500.0,
        min_on_s=20.0,
        min_off_s=12.0,
        near_target_hold_s=10.0,
        ramp_up_m3_min_per_s=2.0,
        ramp_down_m3_min_per_s=3.0,
        pump_rate_schedule_m3_min=((0.0, 1.0), (600.0, 1.0)),
    )


def initial_execution_state() -> ExecutionRolloutState:
    return ExecutionRolloutState(
        masses_kg=REFERENCE_TANK_MASSES_KG,
        target_masses_kg=REFERENCE_TANK_MASSES_KG,
        primary_target_kg=REFERENCE_TANK_MASSES_KG,
        pump_rates_m3_min=np.zeros(3),
        pump_latched=np.zeros(3, dtype=bool),
        pump_off_elapsed_s=np.full(3, 60.0),
    )


def rotor_load_from_record(record: dict[str, Any]) -> RotorGeneralizedLoad:
    return RotorGeneralizedLoad(
        relative_air_velocity_platform_mps=record[
            "relative_air_velocity_platform_mps"
        ],
        normal_load=RotorNormalLoad(
            thrust_n=float(record["thrust_n"]),
            force_platform_n=np.asarray(record["force_platform_n"], dtype=float),
        ),
        generalized_load_platform=record["generalized_load_platform"],
    )


def rotor_load_record(load: RotorGeneralizedLoad) -> dict[str, Any]:
    if not isinstance(load, RotorGeneralizedLoad):
        raise TypeError("load must be RotorGeneralizedLoad")
    return {
        "relative_air_velocity_platform_mps": (
            load.relative_air_velocity_platform_mps.tolist()
        ),
        "thrust_n": float(load.normal_load.thrust_n),
        "force_platform_n": load.normal_load.force_platform_n.tolist(),
        "generalized_load_platform": load.generalized_load_platform.tolist(),
    }


def interval_average_rotor_load(
    current_record: dict[str, Any],
    endpoint_record: dict[str, Any],
) -> RotorGeneralizedLoad:
    """Average adjacent source-bound rotor endpoint records for one block."""

    current = rotor_load_from_record(current_record)
    endpoint = rotor_load_from_record(endpoint_record)
    force = 0.5 * (
        current.normal_load.force_platform_n
        + endpoint.normal_load.force_platform_n
    )
    return RotorGeneralizedLoad(
        relative_air_velocity_platform_mps=0.5
        * (
            current.relative_air_velocity_platform_mps
            + endpoint.relative_air_velocity_platform_mps
        ),
        normal_load=RotorNormalLoad(
            thrust_n=0.5
            * (current.normal_load.thrust_n + endpoint.normal_load.thrust_n),
            force_platform_n=force,
        ),
        generalized_load_platform=0.5
        * (
            current.generalized_load_platform
            + endpoint.generalized_load_platform
        ),
    )


def average_rotor_loads(
    loads: tuple[RotorGeneralizedLoad, ...],
) -> RotorGeneralizedLoad:
    """Return the equal-duration average of realised rotor substep loads."""

    values = tuple(loads)
    if not values or not all(isinstance(load, RotorGeneralizedLoad) for load in values):
        raise ValueError("loads must contain RotorGeneralizedLoad values")
    return RotorGeneralizedLoad(
        relative_air_velocity_platform_mps=np.mean(
            np.vstack([load.relative_air_velocity_platform_mps for load in values]),
            axis=0,
        ),
        normal_load=RotorNormalLoad(
            thrust_n=float(np.mean([load.normal_load.thrust_n for load in values])),
            force_platform_n=np.mean(
                np.vstack([load.normal_load.force_platform_n for load in values]),
                axis=0,
            ),
        ),
        generalized_load_platform=np.mean(
            np.vstack([load.generalized_load_platform for load in values]),
            axis=0,
        ),
    )
def _posture_trace(path: Any) -> np.ndarray:
    return np.vstack(
        [
            np.asarray(step.start_platform_state.position[[3, 4]], dtype=float)
            for step in path.substeps
        ]
        + [np.asarray(path.final_platform_state.position[[3, 4]], dtype=float)]
    )


def _tank_trace(path: Any) -> np.ndarray:
    return np.vstack(
        [
            np.asarray(step.start_execution_state.actual_masses_kg, dtype=float)
            for step in path.substeps
        ]
        + [np.asarray(path.final_execution_state.actual_masses_kg, dtype=float)]
    )


PlannerForecastMode = Literal[
    "current_observation",
    "lstm",
    "persistence",
    "recorded_oracle",
]


def assemble_planner_load_forecast(
    *,
    source_preview: SourceBoundRotorPreview,
    planner_forecast_mode: PlannerForecastMode,
    planner_horizon_blocks: int | None,
) -> GeneralizedLoadForecast:
    """Build only the controller-visible load forecast.

    This helper deliberately does not choose the load used by the plant.  The
    caller may therefore compare forecast configurations under one recorded
    disturbance sequence instead of silently changing both controller input
    and plant forcing.
    """

    source_loads = source_preview.load_assembly.load_forecast
    forecast_metadata = getattr(source_preview.forecast, "metadata", {})
    declared_mode = forecast_metadata.get("planner_forecast_mode")
    if declared_mode is None:
        raise ValueError(
            "source_preview must declare planner_forecast_mode before load "
            "conversion"
        )
    if declared_mode != planner_forecast_mode:
        raise ValueError(
            "source_preview forecast mode does not match planner_forecast_mode"
        )
    current = source_loads.current_generalized_load
    future_loads = source_loads.future_generalized_loads
    lead_times = source_loads.lead_times_s
    available_blocks = int(future_loads.shape[0])
    horizon = available_blocks if planner_horizon_blocks is None else int(
        planner_horizon_blocks
    )
    if horizon <= 0 or horizon > available_blocks:
        raise ValueError(
            "planner_horizon_blocks must be within the available forecast horizon"
        )
    if planner_forecast_mode == "current_observation":
        if horizon != 1:
            raise ValueError(
                "current_observation mode requires exactly one control block"
            )
    elif planner_forecast_mode not in {"lstm", "persistence", "recorded_oracle"}:
        raise ValueError(f"unsupported planner_forecast_mode={planner_forecast_mode!r}")
    return GeneralizedLoadForecast(
        current_generalized_load=current,
        future_generalized_loads=future_loads[:horizon],
        lead_times_s=lead_times[:horizon],
    )


def validate_oracle_endpoint_times(
    *,
    origin_time: str,
    oracle_records: tuple[dict[str, Any], ...],
    lead_times_s: np.ndarray,
) -> list[str]:
    if len(oracle_records) != len(lead_times_s):
        raise ValueError("oracle records must match the requested forecast horizon")
    origin = datetime.strptime(origin_time, TIMESTAMP_FMT)
    endpoint_times: list[str] = []
    for index, (record, lead_time_s) in enumerate(
        zip(oracle_records, lead_times_s, strict=True)
    ):
        endpoint_time = str(record.get("recorded_observation_time", "")).strip()
        if not endpoint_time:
            raise ValueError(
                "oracle wind records must retain recorded_observation_time"
            )
        expected = origin + timedelta(seconds=float(lead_time_s))
        if datetime.strptime(endpoint_time, TIMESTAMP_FMT) != expected:
            raise ValueError(
                f"oracle endpoint {index} is not aligned with its forecast lead time"
            )
        endpoint_times.append(endpoint_time)
    return endpoint_times


def bind_source_identity(
    *,
    source_preview: SourceBoundRotorPreview,
    planner_forecast_mode: PlannerForecastMode,
    lead_times_s: np.ndarray,
    oracle_records: tuple[dict[str, Any], ...] | None,
) -> PreviewMPCSourceIdentity:
    """Bind the source records consumed before physical loads are assembled."""

    source = source_preview.forecast
    horizon = int(np.asarray(lead_times_s).size)
    consumed: dict[str, Any] = {
        "forecast_mode": planner_forecast_mode,
        "origin_time": source.origin_time,
        "lead_times_s": np.asarray(lead_times_s, dtype=float).tolist(),
        "current_observation": source_preview.source_record_payload()[
            "current_observation"
        ],
    }
    if planner_forecast_mode == "lstm":
        consumed["forecast"] = source_preview.source_record_payload()["forecast"]
        source_name = "source_bound_real_lstm_forecast"
        model_version = str(source.model_version)
        uses_future_information = True
    elif planner_forecast_mode == "recorded_oracle":
        if oracle_records is None or len(oracle_records) < horizon:
            raise ValueError(
                "recorded_oracle mode requires one source record per lead"
            )
        consumed["future_recorded_wind_records"] = list(
            oracle_records[:horizon]
        )
        source_name = "recorded_future_wind_oracle"
        model_version = "noncausal_diagnostic_oracle"
        uses_future_information = True
    else:
        source_name = "current_observation_only"
        model_version = "no_forecast_model"
        uses_future_information = False

    return PreviewMPCSourceIdentity(
        source=source_name,
        model_version=model_version,
        origin_time=str(source.origin_time),
        forecast_mode=planner_forecast_mode,
        source_record_sha256=sha256_json(consumed),
        sample_period_s=float(source.sample_period_s),
        lead_times_s=lead_times_s,
        uses_future_information=uses_future_information,
    )


def run_preview_mpc_cycle(
    *,
    source_preview: SourceBoundRotorPreview,
    runtime: Any,
    modes: ThreeTankDifferentialModes,
    platform_state: IncrementalState,
    execution_state: ExecutionRolloutState,
    objective_priority_multipliers: dict[str, float] | None = None,
    planner_forecast_mode: PlannerForecastMode = "lstm",
    planner_horizon_blocks: int | None = None,
    planner_oracle_future_wind_records: tuple[dict[str, Any], ...] | None = None,
    plant_first_block_rotor_load: RotorGeneralizedLoad | None = None,
    plant_first_block_rotor_load_substeps: (
        tuple[RotorGeneralizedLoad, ...] | None
    ) = None,
    plant_load_source: str | None = None,
    cycle_result_observer: Callable[[Any], None] | None = None,
) -> tuple[dict[str, Any], IncrementalState, ExecutionRolloutState]:
    if not isinstance(source_preview, SourceBoundRotorPreview):
        raise TypeError("source_preview must be SourceBoundRotorPreview")
    block_duration_s = float(source_preview.forecast.sample_period_s)

    load_forecast = assemble_planner_load_forecast(
        source_preview=source_preview,
        planner_forecast_mode=planner_forecast_mode,
        planner_horizon_blocks=planner_horizon_blocks,
    )
    oracle_endpoint_times: list[str] = []
    if planner_oracle_future_wind_records is not None:
        oracle_endpoint_times = validate_oracle_endpoint_times(
            origin_time=str(source_preview.forecast.origin_time),
            oracle_records=planner_oracle_future_wind_records[
                : load_forecast.horizon_steps
            ],
            lead_times_s=load_forecast.lead_times_s,
        )
    capacities = np.full(3, TANK_CAPACITY_KG)
    start_tank_masses = np.asarray(execution_state.actual_masses_kg, dtype=float)
    rollout_config = execution_config(block_duration_s=block_duration_s)
    multipliers = (
        {} if objective_priority_multipliers is None else objective_priority_multipliers
    )
    design = research_preview_mpc_design_v4().with_priority_multipliers(
        multipliers
    )
    if plant_first_block_rotor_load is None:
        if planner_forecast_mode != "lstm":
            raise ValueError(
                "non-LSTM planner modes require an explicit actual first-block "
                "rotor load"
            )
        plant_rotor_load = source_preview.first_interval_rotor_load
        resolved_plant_load_source = (
            "planner_first_interval_piecewise_linear_endpoint_average"
        )
    else:
        if not isinstance(plant_first_block_rotor_load, RotorGeneralizedLoad):
            raise TypeError("plant_first_block_rotor_load must be RotorGeneralizedLoad")
        plant_rotor_load = plant_first_block_rotor_load
        resolved_plant_load_source = str(plant_load_source or "external_plant_load")
        if not resolved_plant_load_source.strip():
            raise ValueError("plant_load_source must be non-empty")

    planner_first_block_rotor_load = source_preview.first_interval_rotor_load
    source_identity = bind_source_identity(
        source_preview=source_preview,
        planner_forecast_mode=planner_forecast_mode,
        lead_times_s=load_forecast.lead_times_s,
        oracle_records=planner_oracle_future_wind_records,
    )
    application = PreviewMPCApplication(
        runtime_assembly=runtime,
        reference_tank_masses_kg=REFERENCE_TANK_MASSES_KG,
        tank_capacities_kg=capacities,
        tank_coordinates_m=TANK_COORDINATES_M,
        ballast_modes=modes,
        design=design,
    )
    assembled, cycle_result = application.run_cycle(
        source_identity=source_identity,
        rotor_load_forecast=load_forecast,
        platform_state=platform_state,
        execution_state=execution_state,
        execution_config=rollout_config,
        planner_first_block_rotor_load=planner_first_block_rotor_load,
        execution_first_block_rotor_load=plant_rotor_load,
        execution_first_block_rotor_load_substeps=(
            plant_first_block_rotor_load_substeps
        ),
        wave_interval_loads=np.zeros((load_forecast.horizon_steps, 6)),
        other_interval_loads=np.zeros((load_forecast.horizon_steps, 6)),
    )
    snapshot = assembled.snapshot
    block_model = assembled.block_model
    disturbance = assembled.disturbance_blocks
    design = assembled.design
    design_record = design.as_dict()
    maximum_block_tank_change_kg = float(
        assembled.maximum_abs_tank_mass_change_per_block_kg[0]
    )
    if plant_first_block_rotor_load is None and not np.allclose(
        plant_rotor_load.generalized_load_platform
        + snapshot.incremental_ballast_load,
        disturbance.generalized_loads[0],
        rtol=0.0,
        atol=1.0e-7,
    ):
        raise ValueError(
            "default first-block execution load must equal the planner interval load"
        )
    plan = cycle_result.free_plan
    actuator_envelope = cycle_result.actuator_envelope
    planned_tank_movements = np.vstack(
        [
            modes.tank_mass_deltas_from_modes(modal_increment)
            for modal_increment in plan.modal_increments_kg
        ]
    )
    source_kinematics = source_preview.relative_inflow.kinematics

    payload: dict[str, Any] = {
        "source_binding": {
            **cycle_result.planning_source.to_dict(),
            "load_endpoint_mapping": disturbance.endpoint_mapping,
            "source_to_first_interval_mapping": (
                source_preview.interval_load_mapping
            ),
            "future_platform_kinematics_mode": (
                source_kinematics.future_platform_kinematics_mode.value
            ),
            "future_platform_kinematics_source": (
                source_kinematics.future_platform_kinematics_source
            ),
            "future_rotor_orientation_mode": (
                source_kinematics.future_rotor_orientation_mode.value
            ),
            "future_rotor_orientation_source": (
                source_kinematics.future_rotor_orientation_source
            ),
            "forecast_reliability_role": (
                "learned_future_load_input_to_mpc"
                if planner_forecast_mode == "lstm"
                else "recorded_future_load_input_to_noncausal_diagnostic_mpc"
                if planner_forecast_mode == "recorded_oracle"
                else "future_prediction_not_used_by_controller"
            ),
            "planner_forecast_mode": planner_forecast_mode,
            "planner_forecast_horizon_blocks": load_forecast.horizon_steps,
            "planner_oracle_future_endpoint_times": oracle_endpoint_times,
        },
        "controller_configuration": {
            "preview_mpc_design": design_record,
            "planned_differential_modes": 2,
            "block_count": load_forecast.horizon_steps,
            "block_duration_s": block_duration_s,
            "posture_sample_interval_s": float(
                block_duration_s * block_model.sample_fractions[0]
            ),
            "nominal_maximum_abs_roll_pitch_deg": np.rad2deg(
                design.maximum_abs_roll_pitch_rad
            ).tolist(),
            "absolute_maximum_abs_roll_pitch_deg": np.rad2deg(
                design.absolute_maximum_abs_roll_pitch_rad
            ).tolist(),
            "maximum_abs_tank_mass_change_per_block_kg": (
                maximum_block_tank_change_kg
            ),
            "weight_design": design_record["weight_design"],
            "normalized_objective_priorities": (
                design.normalized_objective_priorities
            ),
            "objective_priority_multipliers": {
                name: float(value) for name, value in multipliers.items()
            },
            "terminal_posture_cost_enabled": (
                design.terminal_posture_priority > 0.0
            ),
            "formal_recursive_feasibility_claimed": False,
            "first_block_actuator_envelope_source": actuator_envelope.source,
        },
        "plan": {
            "success": plan.success,
            "solver_converged": plan.status == 1,
            "status": plan.status,
            "message": plan.message,
            "objective_value": plan.objective_value,
            "objective_breakdown": {
                "running_posture_by_block": (
                    plan.objective_breakdown.running_posture_by_block.tolist()
                ),
                "running_posture": plan.objective_breakdown.running_posture,
                "terminal_posture": plan.objective_breakdown.terminal_posture,
                "tank_movement_by_block": (
                    plan.objective_breakdown.tank_movement_by_block.tolist()
                ),
                "tank_movement": plan.objective_breakdown.tank_movement,
                "tank_throughput_by_block": (
                    plan.objective_breakdown.tank_throughput_by_block.tolist()
                ),
                "tank_throughput": plan.objective_breakdown.tank_throughput,
                "movement_change_by_block": (
                    plan.objective_breakdown.movement_change_by_block.tolist()
                ),
                "movement_change": plan.objective_breakdown.movement_change,
                "posture_slack": plan.objective_breakdown.posture_slack,
                "numerical_regularization": (
                    plan.objective_breakdown.numerical_regularization
                ),
                "constant_offset": plan.objective_breakdown.constant_offset,
                "absolute_total": plan.objective_breakdown.absolute_total,
                "reduced_total": plan.objective_breakdown.reduced_total,
            },
            "modal_increments_kg": plan.modal_increments_kg.tolist(),
            "planned_tank_movement_kg_by_block": planned_tank_movements.tolist(),
            "planned_absolute_tank_throughput_kg_by_block": np.sum(
                np.abs(planned_tank_movements), axis=1
            ).tolist(),
            "planned_target_tank_masses_kg": (
                plan.planned_target_tank_masses_kg.tolist()
            ),
            "first_target_tank_mass_delta_kg": (
                plan.current_target_tank_masses_kg - start_tank_masses
            ).tolist(),
            "predicted_maximum_dominant_tilt_deg": float(
                np.rad2deg(plan.predicted_trajectory.maximum_dominant_tilt_rad)
            ),
            "maximum_posture_constraint_violation_deg": float(
                np.rad2deg(plan.maximum_posture_constraint_violation_rad)
            ),
            "maximum_nominal_posture_exceedance_deg": float(
                np.rad2deg(plan.maximum_nominal_posture_exceedance_rad)
            ),
            "posture_slack_deg": np.rad2deg(plan.posture_slack_rad).tolist(),
            "maximum_tank_constraint_violation_kg": (
                plan.maximum_tank_constraint_violation_kg
            ),
            "maximum_block_tank_movement_violation_kg": (
                plan.maximum_block_tank_movement_violation_kg
            ),
            "maximum_first_block_actuator_violation_kg": (
                plan.maximum_first_block_actuator_violation_kg
            ),
            "maximum_tank_throughput_epigraph_violation_kg": (
                plan.maximum_tank_throughput_epigraph_violation_kg
            ),
            "maximum_scaled_constraint_violation": (
                plan.maximum_scaled_constraint_violation
            ),
            "first_block_actuator_lower_tank_delta_kg": (
                actuator_envelope.first_block_lower_tank_delta_kg.tolist()
            ),
            "first_block_actuator_upper_tank_delta_kg": (
                actuator_envelope.first_block_upper_tank_delta_kg.tolist()
            ),
            "outstanding_differential_target_delta_kg": (
                actuator_envelope.outstanding_target_tank_delta_kg.tolist()
            ),
        },
        "execution": {"attempted": False},
    }

    path = cycle_result.execution_path
    actual_delta = path.actual_tank_mass_delta_kg
    committed_target = np.asarray(
        path.substeps[0].execution_step.requested_target_kg, dtype=float
    )
    target_delta = committed_target - start_tank_masses
    tank_trace = _tank_trace(path)
    net_tank_delta_trace = np.sum(
        tank_trace - start_tank_masses[None, :], axis=1
    )
    posture = _posture_trace(path)
    posture_interval = posture[:-1]
    dominant_tilt_deg = np.rad2deg(np.max(np.abs(posture_interval), axis=1))
    posture_time_above_deg_s = {
        str(threshold): float(
            np.count_nonzero(dominant_tilt_deg > threshold)
            * rollout_config.internal_step_s
        )
        for threshold in (1.5, 2.0, 3.0, 5.0)
    }
    projection = modes.project_tank_mass_deltas(actual_delta)
    absolute_projection = modes.project_tank_mass_deltas(
        path.final_execution_state.actual_masses_kg - REFERENCE_TANK_MASSES_KG
    )
    selected_candidate = cycle_result.selected_candidate
    realised_within_absolute_posture_limit = bool(
        np.all(
            np.max(np.abs(posture), axis=0)
            <= design.absolute_maximum_abs_roll_pitch_rad + 1.0e-12
        )
    )
    payload["execution"] = {
        "attempted": True,
        "selected_target_lifecycle": cycle_result.selected_lifecycle.value,
        "selection_reason": cycle_result.selection_reason.value,
        "selected_from_horizon_feasible_options": (
            cycle_result.selected_from_horizon_feasible_options
        ),
        "selected_from_planner_load_precheck_safe_options": (
            cycle_result.selected_from_planner_load_precheck_safe_options
        ),
        "selected_horizon_within_working_model_scope": (
            cycle_result.selected_horizon_within_working_model_scope
        ),
        "selected_first_block_within_working_model_scope": (
            cycle_result.selected_first_block_within_working_model_scope
        ),
        "realised_execution_within_working_model_scope": (
            cycle_result.realised_execution_within_working_model_scope
        ),
        "degraded_selection": (
            not cycle_result.selected_from_horizon_feasible_options
        ),
        "emergency_selection": (
            not cycle_result.selected_from_planner_load_precheck_safe_options
        ),
        # Compatibility aliases retained for downstream stage summaries.
        "used_fallback": (
            not cycle_result.selected_from_horizon_feasible_options
        ),
        "fallback_reason": (
            None
            if cycle_result.selected_from_horizon_feasible_options
            else cycle_result.selection_reason.value
        ),
        "selected_request_source": cycle_result.selected_lifecycle.value,
        "sampled_posture_limit_satisfied": (
            realised_within_absolute_posture_limit
        ),
        "selected_posture_precheck_passed": (
            selected_candidate.physical_precheck.posture_precheck_passed
        ),
        "selected_posture_precheck_reason": (
            selected_candidate.physical_precheck.posture_precheck_reason
        ),
        "physical_precheck_candidates": [
            {
                "source": item.lifecycle.value,
                "eligible": item.eligible,
                "rejection_reason": item.rejection_reason,
                "combined_physical_prefix_tail_objective": (
                    item.comparison_objective
                ),
                "tail_horizon_objective": item.tail_horizon_objective,
                "objective_breakdown": {
                    "running_posture_by_block": (
                        item.objective.running_posture_by_block.tolist()
                    ),
                    "tank_movement_by_block": (
                        item.objective.tank_movement_by_block.tolist()
                    ),
                    "tank_throughput_by_block": (
                        item.objective.tank_throughput_by_block.tolist()
                    ),
                    "movement_change_by_block": (
                        item.objective.movement_change_by_block.tolist()
                    ),
                    "terminal_posture": item.objective.terminal_posture,
                    "posture_slack": item.objective.posture_slack,
                    "posture_slack_rad": (
                        item.objective.posture_slack_rad.tolist()
                    ),
                },
                "actual_first_block_modal_increment_kg": (
                    item.reached_first_block_projection.modal_masses_kg.tolist()
                ),
                "tail_initial_tank_masses_kg": (
                    item.tail_initial_tank_masses_kg.tolist()
                ),
                "unrepresented_common_mass_change_kg": (
                    item.unrepresented_common_mass_change_kg
                ),
                "total_mass_scope_limit_kg": (
                    item.total_mass_scope_limit_kg
                ),
                "maximum_abs_total_mass_deviation_from_reference_kg": (
                    item.maximum_abs_total_mass_deviation_from_reference_kg
                ),
                "total_mass_within_scope": (
                    item.total_mass_within_scope
                ),
                "projection_residual_norm_kg": (
                    item.projection_residual_norm_kg
                ),
                "projection_residual_tolerance_kg": (
                    item.projection_residual_tolerance_kg
                ),
                "projection_residual_within_tolerance": (
                    item.projection_residual_within_tolerance
                ),
                "maximum_normalized_working_model_scope": (
                    item.maximum_normalized_working_model_scope
                ),
                "horizon_within_working_model_scope": (
                    item.horizon_within_working_model_scope
                ),
                "first_block_maximum_normalized_working_model_scope": (
                    item.first_block_maximum_normalized_working_model_scope
                ),
                "first_block_within_working_model_scope": (
                    item.first_block_within_working_model_scope
                ),
                "free_plan_vs_physical_endpoint_posture_error_deg": np.rad2deg(
                    item.first_block_endpoint_posture_error_rad
                ).tolist(),
                "free_plan_vs_physical_peak_posture_error_deg": np.rad2deg(
                    item.first_block_peak_posture_error_rad
                ).tolist(),
                "peak_abs_roll_pitch_deg": np.rad2deg(
                    item.physical_precheck.peak_abs_roll_pitch_rad
                ).tolist(),
                "terminal_abs_roll_pitch_deg": np.rad2deg(
                    item.physical_precheck.terminal_abs_roll_pitch_rad
                ).tolist(),
                "initial_normalized_posture": (
                    item.physical_precheck.initial_normalized_posture
                ),
                "maximum_normalized_posture": (
                    item.physical_precheck.maximum_normalized_posture
                ),
                "terminal_normalized_posture": (
                    item.physical_precheck.terminal_normalized_posture
                ),
                "within_sampled_posture_limit": (
                    item.physical_precheck.within_sampled_posture_limit
                ),
                "posture_precheck_passed": (
                    item.physical_precheck.posture_precheck_passed
                ),
                "posture_precheck_reason": (
                    item.physical_precheck.posture_precheck_reason
                ),
                "transferred_volume_m3": (
                    item.physical_precheck.execution_path.transferred_volume_m3
                ),
                "aggregate_pump_active_time_s": (
                    item.physical_precheck.execution_path.aggregate_pump_active_time_s
                ),
                "pump_start_count": (
                    item.physical_precheck.execution_path.pump_start_count
                ),
            }
            for item in cycle_result.candidates
        ],
        "committed_target_masses_kg": (
            committed_target.tolist()
        ),
        "planned_first_block_tank_delta_kg": target_delta.tolist(),
        "actual_first_block_tank_delta_kg": actual_delta.tolist(),
        "remaining_mass_to_target_kg": (
            committed_target
            - path.final_execution_state.actual_masses_kg
        ).tolist(),
        "actual_differential_modal_masses_kg": (
            projection.modal_masses_kg.tolist()
        ),
        "actual_common_tank_mass_delta_per_tank_kg": (
            projection.common_tank_mass_delta_kg
        ),
        "target_total_tank_mass_deviation_from_reference_kg": float(
            np.sum(committed_target - REFERENCE_TANK_MASSES_KG)
        ),
        "actual_total_tank_mass_deviation_from_reference_kg": float(
            np.sum(
                path.final_execution_state.actual_masses_kg
                - REFERENCE_TANK_MASSES_KG
            )
        ),
        "actual_common_mass_deviation_per_tank_from_reference_kg": (
            absolute_projection.common_tank_mass_delta_kg
        ),
        "actual_modal_masses_from_reference_kg": (
            absolute_projection.modal_masses_kg.tolist()
        ),
        "maximum_abs_transient_net_tank_mass_change_kg": float(
            np.max(np.abs(net_tank_delta_trace))
        ),
        "final_net_tank_mass_change_kg": float(np.sum(actual_delta)),
        "transferred_volume_m3": path.transferred_volume_m3,
        "aggregate_pump_active_time_s": path.aggregate_pump_active_time_s,
        "pump_start_count": path.pump_start_count,
        "pump_stop_count": path.pump_stop_count,
        "pump_direction_switch_count": path.pump_direction_switch_count,
        "actual_end_roll_pitch_deg": np.rad2deg(posture[-1]).tolist(),
        "actual_peak_abs_roll_pitch_deg": np.rad2deg(
            np.max(np.abs(posture), axis=0)
        ).tolist(),
        "dominant_tilt_rms_deg": float(
            np.sqrt(np.mean(np.square(dominant_tilt_deg)))
        ),
        "dominant_tilt_squared_integral_deg2_s": float(
            np.sum(np.square(dominant_tilt_deg))
            * rollout_config.internal_step_s
        ),
        "dominant_tilt_time_above_deg_s": posture_time_above_deg_s,
        "planned_first_block_end_roll_pitch_deg": np.rad2deg(
            [
                plan.predicted_trajectory.blocks[0].endpoint.platform.position[3],
                plan.predicted_trajectory.blocks[0].endpoint.platform.position[4],
            ]
        ).tolist(),
        "planned_first_block_peak_abs_roll_pitch_deg": np.rad2deg(
            [
                np.max(np.abs(plan.predicted_trajectory.blocks[0].roll_rad)),
                np.max(np.abs(plan.predicted_trajectory.blocks[0].pitch_rad)),
            ]
        ).tolist(),
        "execution_rotor_load_mapping": resolved_plant_load_source,
        "precheck_rotor_load_mapping": (
            "planner_visible_lstm_first_interval_endpoint_average"
            if planner_forecast_mode == "lstm"
            else "planner_visible_recorded_oracle_first_interval"
            if planner_forecast_mode == "recorded_oracle"
            else "planner_visible_current_observation_held_over_block"
        ),
        "precheck_generalized_rotor_load_platform": (
            planner_first_block_rotor_load.generalized_load_platform.tolist()
        ),
        "execution_generalized_rotor_load_platform": (
            plant_rotor_load.generalized_load_platform.tolist()
        ),
        "solver": {
            "name": plan.solver_name,
            "status": plan.message,
            "iterations": plan.solver_iterations,
            "primal_residual": plan.solver_primal_residual,
            "dual_residual": plan.solver_dual_residual,
        },
    }
    if cycle_result_observer is not None:
        cycle_result_observer(cycle_result)
    return payload, cycle_result.next_platform_state, cycle_result.next_execution_state


def run_rolling_semantic_check(
    *,
    origin: datetime | None = None,
    device: str = "cpu",
    cycle_count: int = 1,
) -> dict[str, Any]:
    if (
        not isinstance(cycle_count, int)
        or isinstance(cycle_count, bool)
        or not 1 <= cycle_count <= 36
    ):
        raise ValueError("cycle_count must be an integer in [1, 36]")

    runtime = research_runtime_assembly()
    modes = ThreeTankDifferentialModes(
        tank_coordinates_m=TANK_COORDINATES_M,
        gravity_m_s2=runtime.gravity_m_s2,
    )
    platform_state = IncrementalState.zeros()
    execution_state = initial_execution_state()
    resources = prepare_real_lstm_preview_resources(device=device)
    current_origin = (
        resources.replay.first_sample_timestamp() if origin is None else origin
    )
    cycle_records: list[dict[str, Any]] = []

    for cycle_index in range(cycle_count):
        source_record = infer_real_lstm_wind_record(
            resources=resources,
            origin=current_origin,
        )
        source_preview = assemble_planner_rotor_preview_from_record(
            resources=resources,
            source_record=source_record,
            platform_state=platform_state,
            planner_forecast_mode="lstm",
            horizon_blocks=source_record.forecast.horizon_steps,
        )
        source_origin = datetime.strptime(
            str(source_preview.forecast.origin_time), TIMESTAMP_FMT
        )
        cycle, next_platform_state, next_execution_state = run_preview_mpc_cycle(
            source_preview=source_preview,
            runtime=runtime,
            modes=modes,
            platform_state=platform_state,
            execution_state=execution_state,
        )
        cycle["cycle_index"] = cycle_index
        if not cycle["execution"]["realised_execution_within_working_model_scope"]:
            cycle_records.append(
                {
                    "cycle_index": cycle_index,
                    "diagnostic_only": True,
                    "exclusion_reason": (
                        "realised_execution_exceeded_working_model_scope"
                    ),
                    "cycle": cycle,
                }
            )
            break
        cycle_records.append(cycle)
        platform_state = next_platform_state
        execution_state = next_execution_state
        current_origin = source_origin + timedelta(
            seconds=float(cycle["controller_configuration"]["block_duration_s"])
        )
        if (
            not cycle["execution"]["selected_posture_precheck_passed"]
            or not cycle["execution"]["sampled_posture_limit_satisfied"]
        ):
            break

    executed_cycles = [
        cycle
        for cycle in cycle_records
        if not cycle.get("diagnostic_only", False)
        and cycle["execution"]["attempted"]
    ]
    maximum_actual_posture = [0.0, 0.0]
    if executed_cycles:
        maximum_actual_posture = np.max(
            np.asarray(
                [
                    cycle["execution"]["actual_peak_abs_roll_pitch_deg"]
                    for cycle in executed_cycles
                ],
                dtype=float,
            ),
            axis=0,
        ).tolist()
    return {
        "purpose": "real_lstm_preview_mpc_isolated_branch_rolling_semantic_check",
        "boundaries": {
            "is_performance_result": False,
            "is_parameter_tuning": False,
            "is_openfast_validation": False,
            "uses_real_lstm_forecast": True,
            "uses_only_first_planned_target_for_execution": True,
            "future_targets_are_diagnostic_only": True,
            "uses_current_target_tracking_pump_model": True,
            "later_cycles_start_from_prior_modeled_state_not_measurements": True,
            "has_parallel_baseline_branch": False,
        },
        "requested_cycle_count": cycle_count,
        "completed_cycle_count": len(executed_cycles),
        "continuous_duration_s": float(
            sum(
                cycle["controller_configuration"]["block_duration_s"]
                for cycle in executed_cycles
            )
        ),
        "aggregate": {
            "transferred_volume_m3": float(
                sum(cycle["execution"]["transferred_volume_m3"] for cycle in executed_cycles)
            ),
            "aggregate_pump_active_time_s": float(
                sum(
                    cycle["execution"]["aggregate_pump_active_time_s"]
                    for cycle in executed_cycles
                )
            ),
            "pump_start_count": int(
                sum(cycle["execution"]["pump_start_count"] for cycle in executed_cycles)
            ),
            "pump_stop_count": int(
                sum(cycle["execution"]["pump_stop_count"] for cycle in executed_cycles)
            ),
            "maximum_actual_abs_roll_pitch_deg": maximum_actual_posture,
            "maximum_abs_transient_net_tank_mass_change_kg": float(
                max(
                    (
                        cycle["execution"][
                            "maximum_abs_transient_net_tank_mass_change_kg"
                        ]
                        for cycle in executed_cycles
                    ),
                    default=0.0,
                )
            ),
            "maximum_abs_remaining_mass_to_target_kg": float(
                max(
                    (
                        np.max(np.abs(cycle["execution"]["remaining_mass_to_target_kg"]))
                        for cycle in executed_cycles
                    ),
                    default=0.0,
                )
            ),
            "maximum_abs_total_tank_mass_deviation_from_reference_kg": float(
                max(
                    (
                        abs(
                            cycle["execution"][
                                "actual_total_tank_mass_deviation_from_reference_kg"
                            ]
                        )
                        for cycle in executed_cycles
                    ),
                    default=0.0,
                )
            ),
        },
        "final_platform_position": platform_state.position.tolist(),
        "final_actual_tank_masses_kg": execution_state.actual_masses_kg.tolist(),
        "cycles": cycle_records,
    }
