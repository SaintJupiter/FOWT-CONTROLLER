#!/usr/bin/env python3
"""Run one real LSTM record through the first-interval physical fact chain.

This is an integration smoke check, not a controller experiment.  It binds a
single replay-origin observation and model forecast to the new physical path
under deliberately explicit, frozen platform assumptions.  A source-bound
below-rated nominal rotor schedule resolves the relative inflow at each lead
to a tabulated ``C_T``; it is not a dynamic ROSCO or controller simulation.
The report makes those boundaries visible rather than treating the resulting
numbers as an aerodynamic, platform-response, safety, or control result.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timedelta
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

from fowt_platform import (
    IncrementalState,
    NOMINAL_BELOW_RATED_GENERATING,
    assemble_ballast_model_snapshot,
    assemble_nominal_below_rated_operating_forecast,
    assemble_volturnus_static_restoring_aligned_runtime_assembly,
    inspect_nominal_below_rated_operating_inputs,
    load_rosco_nominal_below_rated_schedule_from_zip,
    load_rosco_rotor_performance_table_from_zip,
)
from fowt_platform.coordinates import downwind_normal_relative_wind_component
from run_rotor_normal_load_input_audit import (
    ARCHIVE_ROOT,
    MODEL_ARCHIVE_SHA256,
    MODEL_ZIP,
    ROTOR_TABLE_MEMBER,
    SEMI_DIRECTORY,
    run_audit as run_rotor_input_audit,
)
from wind_prediction.controller_replay_adapter import ReplayForecastEvidenceSource
from wind_prediction.execution_rollout import ExecutionRolloutConfig, ExecutionRolloutState
from wind_prediction.forecast_adapter import ForecastModelAdapter
from wind_prediction.forecast_evidence import (
    ForecastEvidence,
    lead_reliability_from_metrics,
    validate_forecast_evidence,
)
from wind_prediction.forecast_physical_load import (
    ForecastRotorLoadParameters,
)
from wind_prediction.forecast_rotor_inflow import (
    ForecastRotorKinematics,
    ForecastRotorRelativeInflow,
    assemble_forecast_rotor_relative_inflow,
)
from wind_prediction.forecast_physical_load import (
    rotor_geometry_at_declared_nacelle_yaw,
)
from wind_prediction.forecast_platform_trajectory import ForecastTrajectoryEnvironment
from wind_prediction.replay_dataset import Fino1ReplayDataset, TIMESTAMP_FMT
from wind_prediction.physical_forecast_cycle_facts import (
    assemble_first_interval_physical_cycle_input,
)
from wind_prediction.physical_forecast_admission import (
    bind_physical_forecast_admission,
)
from wind_prediction.physical_lifecycle_comparison import (
    assemble_physical_lifecycle_comparison_for_current_block,
    assemble_physical_lifecycle_horizon_comparison,
)
from wind_prediction.physical_current_cycle_decision import (
    PhysicalCurrentCycleDecisionPolicy,
)
from wind_prediction.physical_lifecycle_selection import PhysicalPostureLimits
from wind_prediction.physical_rolling_control_chain import (
    PhysicalRollingChainSetup,
    advance_physical_rolling_control_cycle,
)
from wind_prediction.replay_physical_cycle_adapter import (
    assemble_resolved_physical_rolling_cycle_input,
)
from wind_prediction.forecast_endpoint_horizon_response import (
    assemble_forecast_endpoint_held_responses,
)


DATASET_DIRECTORY = ROOT / "data/processed/wind_ml_10min/ballast_decision_fino1_meteo_aux_v1"
MODEL_DIRECTORY = ROOT / "outputs/wind_prediction/lstm_segmented_fino1_meteo_aux_v1"
REFERENCE_MANIFEST = ROOT / "configs/reference_platforms/volturnus_s_openfast_v1_1_16.json"
REFERENCE_TANK_MASSES_KG = np.array([1_108_000.0, 1_362_000.0, 1_362_000.0])
TANK_CAPACITY_KG = 1_896_250.0
TANK_COORDINATES_M = np.array(
    [
        [46.2, 0.0, -10.0],
        [-23.1, 46.2 * 0.866, -10.0],
        [-23.1, -46.2 * 0.866, -10.0],
    ],
    dtype=float,
)
_SOURCE_GEOMETRY_AUDIT_WIND_SPEED_MPS = 10.0
_FROZEN_HEADING_RAD = 0.0
_TEST_SHEAR_EXPONENT = 0.0
ROSCO_TUNING_MEMBER = (
    f"{ARCHIVE_ROOT}/{SEMI_DIRECTORY}/{SEMI_DIRECTORY}_ROSCO.yaml"
)
SOURCE_RECORD_CURRENT_ONLY = "current_only"
SOURCE_RECORD_CURRENT_AND_FORECAST = "current_and_forecast"
SOURCE_RECORD_SCOPES = frozenset(
    {SOURCE_RECORD_CURRENT_ONLY, SOURCE_RECORD_CURRENT_AND_FORECAST}
)


@dataclass(frozen=True)
class RealLstmSmokeResources:
    """Source/model resources shared for read-only use across replay origins."""

    device: str
    replay: Fino1ReplayDataset
    forecast_adapter: ForecastModelAdapter
    expected_lead_minutes: tuple[int, ...]
    lead_reliability: np.ndarray
    rotor_audit: dict[str, Any]
    performance_table: Any
    nominal_schedule: Any


def prepare_smoke_resources(*, device: str = "cpu") -> RealLstmSmokeResources:
    """Load replay, learned model metadata and rotor tables once per batch."""

    replay = Fino1ReplayDataset(DATASET_DIRECTORY, split="test")
    adapter = ForecastModelAdapter(MODEL_DIRECTORY, device=device)
    expected_lead_minutes = tuple(
        int(round(step * replay.update_interval_s / 60.0))
        for step in range(1, int(replay.future_steps) + 1)
    )
    if any(
        not np.isclose(
            lead_minutes,
            step * replay.update_interval_s / 60.0,
            rtol=0.0,
            atol=1e-9,
        )
        for step, lead_minutes in enumerate(expected_lead_minutes, start=1)
    ):
        raise RuntimeError(
            "replay forecast leads must resolve to whole-minute metric labels"
        )
    lead_reliability = lead_reliability_from_metrics(
        MODEL_DIRECTORY / "lstm_regression_metrics.csv",
        expected_lead_minutes=expected_lead_minutes,
    )
    lead_reliability.setflags(write=False)
    return RealLstmSmokeResources(
        device=str(device),
        replay=replay,
        forecast_adapter=adapter,
        expected_lead_minutes=expected_lead_minutes,
        lead_reliability=lead_reliability,
        rotor_audit=run_rotor_input_audit(
            wind_speed_mps=_SOURCE_GEOMETRY_AUDIT_WIND_SPEED_MPS,
        ),
        performance_table=load_rosco_rotor_performance_table_from_zip(
            MODEL_ZIP,
            ROTOR_TABLE_MEMBER,
            expected_archive_sha256=MODEL_ARCHIVE_SHA256,
        ),
        nominal_schedule=load_rosco_nominal_below_rated_schedule_from_zip(
            MODEL_ZIP,
            ROSCO_TUNING_MEMBER,
            expected_archive_sha256=MODEL_ARCHIVE_SHA256,
        ),
    )


def _lifecycle_summary(response: Any) -> dict[str, Any]:
    trace = response.lifecycle_trace
    end_state = response.execution_coupled_end_state
    actual_mass_delta = trace.actual_final_masses_kg - trace.execution_start_state.actual_masses_kg
    return {
        "lifecycle": response.lifecycle.value,
        "target_reached": bool(response.target_reached),
        "requested_target_masses_kg": [
            float(value) for value in trace.requested_target_masses_kg
        ],
        "actual_final_tank_masses_kg": [
            float(value) for value in trace.actual_final_masses_kg
        ],
        "remaining_mass_to_target_kg": [
            float(value) for value in trace.remaining_mass_to_target_kg
        ],
        "transferred_volume_m3": float(response.transferred_volume_m3),
        "aggregate_pump_active_time_s": float(response.aggregate_pump_active_time_s),
        "pump_start_count": int(response.pump_start_count),
        "actual_tank_mass_delta_kg": [
            float(value) for value in actual_mass_delta
        ],
        "net_actual_tank_mass_delta_kg": float(np.sum(actual_mass_delta)),
        "end_pitch_roll_rad": [
            float(end_state.position[4]),
            float(end_state.position[3]),
        ],
    }


def _execution_block_lifecycle_outcome_summary(comparison: Any) -> list[dict[str, Any]]:
    """Return common factual outcomes without assigning a preferred lifecycle."""

    return [
        {
            "lifecycle": outcome.lifecycle.value,
            "duration_s": float(outcome.rollout.duration_s),
            "transferred_volume_m3": float(outcome.transferred_volume_m3),
            "aggregate_pump_active_time_s": float(
                outcome.aggregate_pump_active_time_s
            ),
            "pump_start_count": int(outcome.pump_start_count),
            "pump_stop_count": int(outcome.pump_stop_count),
            "pump_direction_switch_count": int(outcome.pump_direction_switch_count),
            "end_pitch_roll_rad": [
                float(value) for value in outcome.end_pitch_roll_rad
            ],
            "peak_abs_pitch_roll_rad": [
                float(value) for value in outcome.peak_abs_pitch_roll_rad
            ],
            "end_pitch_roll_rate_rad_s": [
                float(value) for value in outcome.end_pitch_roll_rate_rad_s
            ],
            "peak_abs_pitch_roll_rate_rad_s": [
                float(value) for value in outcome.peak_abs_pitch_roll_rate_rad_s
            ],
            "reached_final_tank_masses_kg": [
                float(value) for value in outcome.reached_final_tank_masses_kg
            ],
            "remaining_mass_to_target_kg": [
                float(value) for value in outcome.remaining_mass_to_target_kg
            ],
        }
        for outcome in comparison.outcomes
    ]


def _execution_block_change_from_continuation_summary(
    comparison: Any,
) -> list[dict[str, Any]]:
    """Report alternative-minus-continuation facts without ranking operations."""

    return [
        {
            "lifecycle": lifecycle.value,
            "transferred_volume_delta_m3": float(
                change.transferred_volume_delta_m3
            ),
            "aggregate_pump_active_time_delta_s": float(
                change.aggregate_pump_active_time_delta_s
            ),
            "pump_start_count_delta": int(change.pump_start_count_delta),
            "pump_stop_count_delta": int(change.pump_stop_count_delta),
            "pump_direction_switch_count_delta": int(
                change.pump_direction_switch_count_delta
            ),
            "reached_final_tank_mass_delta_kg": [
                float(value) for value in change.reached_final_tank_mass_delta_kg
            ],
            "end_pitch_roll_delta_rad": [
                float(value) for value in change.end_pitch_roll_delta_rad
            ],
            "peak_abs_pitch_roll_delta_rad": [
                float(value) for value in change.peak_abs_pitch_roll_delta_rad
            ],
            "end_pitch_roll_rate_delta_rad_s": [
                float(value) for value in change.end_pitch_roll_rate_delta_rad_s
            ],
            "peak_abs_pitch_roll_rate_delta_rad_s": [
                float(value)
                for value in change.peak_abs_pitch_roll_rate_delta_rad_s
            ],
        }
        for lifecycle in comparison.available_lifecycles
        if lifecycle.value != "continue_existing"
        for change in (comparison.change_from_continuation(lifecycle),)
    ]


def _rotor_load_record(load: Any) -> dict[str, Any]:
    """Serialize one source-bound rotor load without changing its channel identity."""

    return {
        "relative_air_velocity_platform_mps": [
            float(value) for value in load.relative_air_velocity_platform_mps
        ],
        "thrust_n": float(load.normal_load.thrust_n),
        "force_platform_n": [
            float(value) for value in load.normal_load.force_platform_n
        ],
        "generalized_load_platform": [
            float(value) for value in load.generalized_load_platform
        ],
    }


def _held_lifecycle_horizon_outcome_summary(comparison: Any) -> list[dict[str, Any]]:
    """Report held-request horizon facts without assigning an operation value."""

    return [
        {
            "lifecycle": outcome.lifecycle.value,
            "duration_s": float(outcome.rollout.duration_s),
            "transferred_volume_m3": float(outcome.transferred_volume_m3),
            "aggregate_pump_active_time_s": float(
                outcome.aggregate_pump_active_time_s
            ),
            "pump_start_count": int(outcome.pump_start_count),
            "pump_stop_count": int(outcome.pump_stop_count),
            "pump_direction_switch_count": int(outcome.pump_direction_switch_count),
            "end_pitch_roll_rad": [
                float(value) for value in outcome.end_pitch_roll_rad
            ],
            "peak_abs_pitch_roll_rad": [
                float(value) for value in outcome.peak_abs_pitch_roll_rad
            ],
            "end_pitch_roll_rate_rad_s": [
                float(value) for value in outcome.end_pitch_roll_rate_rad_s
            ],
            "peak_abs_pitch_roll_rate_rad_s": [
                float(value)
                for value in outcome.peak_abs_pitch_roll_rate_rad_s
            ],
            "reached_final_tank_masses_kg": [
                float(value) for value in outcome.reached_final_tank_masses_kg
            ],
        }
        for outcome in comparison.outcomes
    ]


def _horizon_endpoint_preview_summary(facts: Any) -> dict[str, Any]:
    """Expose independent current-block previews without creating a target plan."""

    previews = facts.horizon_endpoint_previews
    return {
        "interpretation": (
            "independent_current_block_endpoint_previews_not_a_target_sequence"
        ),
        "execution_preview_duration_s": float(
            previews.execution_preview_duration_s
        ),
        "outcomes": [
            {
                "lead_index": int(preview.lead_index),
                "lead_time_s": float(preview.lead_time_s),
                "requested_target_masses_kg": [
                    float(value) for value in preview.target_tank_masses_kg
                ],
                "actual_final_tank_masses_kg": [
                    float(value)
                    for value in preview.reachability.execution_step.state.actual_masses_kg
                ],
                "remaining_mass_to_target_kg": [
                    float(value)
                    for value in preview.reachability.remaining_mass_to_endpoint_kg
                ],
                "transferred_volume_m3": float(
                    preview.reachability.execution_step.transferred_volume_m3
                ),
                "aggregate_pump_active_time_s": float(
                    preview.reachability.execution_step.active_time_s
                ),
                "pump_start_count": int(preview.reachability.execution_step.starts),
            }
            for preview in previews.previews
        ],
    }


def _endpoint_held_response_summary(response_set: Any) -> dict[str, Any]:
    """Expose parallel held endpoint facts without assigning a preferred one."""

    return {
        "interpretation": (
            "independent_endpoint_targets_projected_to_current_block_then_held_not_a_target_sequence"
        ),
        "outcomes": [
            {
                "lead_index": int(response.lead_index),
                "lead_time_s": float(response.lead_time_s),
                "requested_target_masses_kg": [
                    float(value) for value in response.target_tank_masses_kg
                ],
                "current_block_end_pitch_roll_rad": [
                    float(value) for value in response.current_block_end_pitch_roll_rad
                ],
                "horizon_end_pitch_roll_rad": [
                    float(value) for value in response.horizon_end_pitch_roll_rad
                ],
                "horizon_peak_abs_pitch_roll_rad": [
                    float(value)
                    for value in response.horizon_peak_abs_pitch_roll_rad
                ],
                "horizon_transferred_volume_m3": float(
                    response.horizon_transferred_volume_m3
                ),
            }
            for response in response_set.responses
        ],
    }


def _endpoint_held_response_change_summary(
    response_set: Any,
    horizon_comparison: Any,
) -> dict[str, Any]:
    """Expose endpoint-minus-continuation facts without a preferred endpoint."""

    return {
        "difference_convention": "endpoint_response_minus_continuation",
        "outcomes": [
            {
                "lead_index": int(change.response.lead_index),
                "lead_time_s": float(change.response.lead_time_s),
                "current_block_transferred_volume_delta_m3": float(
                    change.current_block_transferred_volume_delta_m3
                ),
                "horizon_transferred_volume_delta_m3": float(
                    change.horizon_transferred_volume_delta_m3
                ),
                "current_block_pump_active_time_delta_s": float(
                    change.current_block_pump_active_time_delta_s
                ),
                "horizon_pump_active_time_delta_s": float(
                    change.horizon_pump_active_time_delta_s
                ),
                "current_block_end_pitch_roll_delta_rad": [
                    float(value)
                    for value in change.current_block_end_pitch_roll_delta_rad
                ],
                "horizon_end_pitch_roll_delta_rad": [
                    float(value) for value in change.horizon_end_pitch_roll_delta_rad
                ],
                "horizon_peak_abs_pitch_roll_delta_rad": [
                    float(value)
                    for value in change.horizon_peak_abs_pitch_roll_delta_rad
                ],
            }
            for change in response_set.changes_from_continuation(
                horizon_comparison=horizon_comparison
            )
        ],
    }


def _operating_point_summary(point: Any) -> dict[str, float]:
    """Return the resolved, non-dynamic table node in report-friendly units."""

    return {
        "normal_relative_inflow_mps": float(point.normal_inflow_speed_mps),
        "pitch_deg": float(point.pitch_deg),
        "rotor_speed_rpm": float(point.rotor_speed_rpm),
        "tip_speed_ratio": float(point.tip_speed_ratio),
        "thrust_coefficient": float(point.thrust_coefficient),
    }


def _assert_relative_inflow_matches_final_load(
    *,
    facts: Any,
    relative_inflow: ForecastRotorRelativeInflow,
    parameters: ForecastRotorLoadParameters,
    kinematics: ForecastRotorKinematics,
) -> None:
    """Require the schedule input and final load path to share one ``U_n``."""

    _, normal = rotor_geometry_at_declared_nacelle_yaw(
        parameters,
        kinematics.current_nacelle_yaw_relative_platform_rad,
    )
    load_assembly = facts.trajectory.load_assembly
    current = downwind_normal_relative_wind_component(
        load_assembly.current_rotor_load.relative_air_velocity_platform_mps,
        normal,
    )
    future = np.asarray(
        [
            downwind_normal_relative_wind_component(
                load.relative_air_velocity_platform_mps,
                normal,
            )
            for load in load_assembly.future_rotor_loads
        ],
        dtype=float,
    )
    if not np.isclose(
        current,
        relative_inflow.current_normal_relative_inflow_mps,
        rtol=0.0,
        atol=1e-12,
    ) or not np.allclose(
        future,
        relative_inflow.future_normal_relative_inflows_mps,
        rtol=0.0,
        atol=1e-12,
    ):
        raise RuntimeError(
            "nominal rotor schedule normal inflow must match the final load path"
        )


def _current_observation_persistence_counterfactual(
    *,
    forecast: ForecastEvidence,
    current_enu_downwind_air_velocity_mps: Any,
) -> ForecastEvidence:
    """Repeat one origin observation across the existing future lead grid.

    This is a validation-only counterfactual.  ``provides_future_preview`` is
    true because the record contains an explicit vector for each future lead,
    not because it is a learned forecast or an admissible live-control input.
    The helper is intentionally local to this script so it cannot become an
    implicit fallback in the controller path.
    """

    validate_forecast_evidence(forecast)
    if not forecast.provides_future_preview:
        raise ValueError("persistence contrast requires a source future forecast")
    current = np.asarray(current_enu_downwind_air_velocity_mps, dtype=float).reshape(-1)
    if current.shape != (2,) or not np.all(np.isfinite(current)):
        raise ValueError("current ENU wind must contain two finite components")
    return ForecastEvidence(
        source="validation_current_observation_persistence",
        model_version="current_observation_repeat_not_model",
        origin_time=forecast.origin_time,
        sample_period_s=forecast.sample_period_s,
        uv_ms=np.repeat(current.reshape(1, 2), forecast.horizon_steps, axis=0),
        event_probs={},
        lead_reliability=np.zeros(forecast.horizon_steps, dtype=float),
        provides_future_preview=True,
        metadata={
            "validation_only_counterfactual": "current_observation_persistence",
            "source_forecast_source": forecast.source,
            "source_forecast_model_version": forecast.model_version,
        },
    )


def _physical_contrast_summary(
    *,
    learned_facts: Any,
    persistence_facts: Any,
    learned_relative_inflow: ForecastRotorRelativeInflow,
    persistence_relative_inflow: ForecastRotorRelativeInflow,
    learned_operating_forecast: Any,
    persistence_operating_forecast: Any,
) -> dict[str, Any]:
    """Report factual first-interval differences without ranking either path."""

    learned_new_track = learned_facts.new_track
    persistence_new_track = persistence_facts.new_track
    if learned_new_track is None or persistence_new_track is None:
        raise RuntimeError("smoke fixture requires new_track on both physical paths")
    learned_rhs = learned_facts.new_track_state_conditioned_rhs
    persistence_rhs = persistence_facts.new_track_state_conditioned_rhs
    if learned_rhs is None or persistence_rhs is None:
        raise RuntimeError("smoke fixture requires state-conditioned new_track rhs facts")
    return {
        "purpose": "future_wind_input_semantic_contrast_only",
        "boundaries": {
            "is_controller_experiment": False,
            "is_candidate_selection": False,
            "is_performance_result": False,
            "is_forecast_accuracy_assessment": False,
        },
        "same_entering_state": {
            "forecast_origin_time": learned_facts.trajectory.initial_state_time,
            "initial_tank_masses_kg": [
                float(value)
                for value in learned_facts.trajectory.platform_snapshot.actual_tank_masses_kg
            ],
            "first_interval_duration_s": float(
                learned_facts.trajectory.steps[0].duration_s
            ),
        },
        "source_bound_future_rotor_input_changes": {
            "learned_first_future_enu_downwind_wind_mps": [
                float(value)
                for value in learned_facts.trajectory.load_assembly.future_rotor_plane_enu_downwind_mps[0]
            ],
            "persistence_first_future_enu_downwind_wind_mps": [
                float(value)
                for value in persistence_facts.trajectory.load_assembly.future_rotor_plane_enu_downwind_mps[0]
            ],
            "learned_first_future_normal_relative_inflow_mps": float(
                learned_relative_inflow.future_normal_relative_inflows_mps[0]
            ),
            "persistence_first_future_normal_relative_inflow_mps": float(
                persistence_relative_inflow.future_normal_relative_inflows_mps[0]
            ),
            "learned_first_future_thrust_coefficient": float(
                learned_operating_forecast.future[0].thrust_coefficient
            ),
            "persistence_first_future_thrust_coefficient": float(
                persistence_operating_forecast.future[0].thrust_coefficient
            ),
            "future_generalized_load_difference_norm_by_lead": [
                float(np.linalg.norm(learned - persistence))
                for learned, persistence in zip(
                    learned_facts.trajectory.load_assembly.load_forecast.future_generalized_loads,
                    persistence_facts.trajectory.load_assembly.load_forecast.future_generalized_loads,
                )
            ],
        },
        "first_interval_new_track_difference": {
            "requested_target_mass_difference_kg": [
                float(value)
                for value in (
                    learned_new_track.requested_target_masses_kg
                    - persistence_new_track.requested_target_masses_kg
                )
            ],
            "reached_final_tank_mass_difference_kg": [
                float(value)
                for value in (
                    learned_new_track.reached_final_tank_masses_kg
                    - persistence_new_track.reached_final_tank_masses_kg
                )
            ],
        },
        "new_track_rhs_difference_at_each_scenario_frozen_endpoint": {
            "uses_execution_coupled_end_state": False,
            "uses_same_platform_state_across_scenarios": False,
            "state_conditioned_rhs_pitch_roll_difference_Nm": [
                float(
                    learned_rhs.reachable_dynamic_rhs_generalized_load[index]
                    - persistence_rhs.reachable_dynamic_rhs_generalized_load[index]
                )
                for index in (4, 3)
            ],
        },
    }


def run_smoke(
    *,
    origin: datetime | None = None,
    device: str = "cpu",
    resources: RealLstmSmokeResources | None = None,
    include_persistence_contrast: bool = False,
    include_current_cycle_decision: bool = False,
    rolling_cycle_count: int = 1,
    operating_domain_only: bool = False,
    source_record_scope: str = SOURCE_RECORD_CURRENT_AND_FORECAST,
    return_unsupported_source_record: bool = False,
) -> dict[str, Any]:
    """Return one inspectable real-model integration record.

    The source-bound below-rated schedule resolves each predicted relative
    inflow to a nominal table node.  It does not represent dynamic ROSCO,
    yaw control, above-rated operation, or an engineering performance result.
    """

    if (
        not isinstance(rolling_cycle_count, int)
        or isinstance(rolling_cycle_count, bool)
        or not 1 <= rolling_cycle_count <= 3
    ):
        raise ValueError("rolling_cycle_count must be an integer in [1, 3]")
    scope = str(source_record_scope)
    if scope not in SOURCE_RECORD_SCOPES:
        raise ValueError(
            "source_record_scope must be 'current_only' or "
            "'current_and_forecast'"
        )
    if scope == SOURCE_RECORD_CURRENT_ONLY and (
        include_persistence_contrast
        or include_current_cycle_decision
        or rolling_cycle_count != 1
    ):
        raise ValueError(
            "current_only source records cannot run forecast contrasts or "
            "controller-decision diagnostics"
        )

    if resources is None:
        resources = prepare_smoke_resources(device=device)
    elif resources.device != str(device):
        raise ValueError(
            "shared smoke resources were prepared for a different device"
        )

    replay = resources.replay
    selected_origin = replay.first_sample_timestamp() if origin is None else origin
    expected_lead_minutes = resources.expected_lead_minutes
    source = ReplayForecastEvidenceSource(
        replay_dataset=replay,
        start_timestamp=selected_origin,
        forecast_adapter=resources.forecast_adapter,
        lead_reliability=resources.lead_reliability,
    )

    rotor_audit = resources.rotor_audit
    source_parameters = rotor_audit["source_parameters"]
    reference_state = rotor_audit["source"]["candidate_reference_state"]["state"]
    first_time = selected_origin.strftime(TIMESTAMP_FMT)

    runtime = assemble_volturnus_static_restoring_aligned_runtime_assembly(
        REFERENCE_MANIFEST,
        np.zeros((6, 6)),
    )
    snapshot = assemble_ballast_model_snapshot(
        runtime_assembly=runtime,
        actual_tank_masses_kg=REFERENCE_TANK_MASSES_KG,
        reference_tank_masses_kg=REFERENCE_TANK_MASSES_KG,
        tank_capacities_kg=np.full(3, TANK_CAPACITY_KG),
        tank_coordinates_m=TANK_COORDINATES_M,
    )
    execution = ExecutionRolloutState(
        masses_kg=REFERENCE_TANK_MASSES_KG,
        target_masses_kg=REFERENCE_TANK_MASSES_KG,
        primary_target_kg=REFERENCE_TANK_MASSES_KG,
        pump_rates_m3_min=np.zeros(3),
        pump_latched=np.zeros(3, dtype=bool),
        pump_off_elapsed_s=np.full(3, 60.0),
    )
    config = ExecutionRolloutConfig(
        block_duration_s=600.0,
        internal_step_s=30.0,
        water_density_kg_m3=1025.0,
        max_pump_rate_m3_min=1.0,
        tank_capacity_kg=TANK_CAPACITY_KG,
        stop_error_kg=0.0,
        restart_error_kg=0.0,
        min_on_s=0.0,
        min_off_s=0.0,
        near_target_hold_s=0.0,
        ramp_up_m3_min_per_s=np.inf,
        ramp_down_m3_min_per_s=np.inf,
        pump_rate_schedule_m3_min=((0.0, 1.0), (600.0, 1.0)),
    )

    # The dataset declares the fixed model horizon.  The source is retrieved
    # once below and each emitted forecast lead receives its own table lookup.
    horizon = int(replay.future_steps)
    kinematics = ForecastRotorKinematics(
        current_platform_reference_velocity_platform_mps=np.zeros(3),
        current_platform_angular_velocity_platform_radps=np.zeros(3),
        future_platform_reference_velocities_platform_mps=np.zeros((horizon, 3)),
        future_platform_angular_velocities_platform_radps=np.zeros((horizon, 3)),
        future_platform_kinematics_mode="frozen_zero",
        future_platform_kinematics_source="explicit_real_lstm_smoke_fixture",
        current_nacelle_yaw_relative_platform_rad=0.0,
        future_rotor_orientation_mode="frozen_current",
        future_rotor_orientation_source=(
            "explicit_fixed_reference_nacelle_orientation_smoke_fixture"
        ),
    )
    parameters = ForecastRotorLoadParameters(
        air_density_kg_m3=float(source_parameters["air_density_kg_m3"]),
        rotor_radius_m=float(source_parameters["rotor_radius_m"]),
        wind_reference_height_m=102.0,
        rotor_plane_height_m=float(
            source_parameters["hub_from_platform_reference_frozen_equilibrium_m"][2]
        ),
        power_law_shear_exponent=_TEST_SHEAR_EXPONENT,
        frozen_equilibrium_heading_rad=_FROZEN_HEADING_RAD,
        rotor_force_application_point_from_reference_m=(
            source_parameters["hub_from_platform_reference_frozen_equilibrium_m"]
        ),
        downwind_rotor_normal_platform=(
            source_parameters["downwind_rotor_normal_frozen_equilibrium"]
        ),
    )
    environment = ForecastTrajectoryEnvironment(
        current_wave_load=np.zeros(6),
        current_other_load=np.zeros(6),
        wave_interval_loads=np.zeros((horizon, 6)),
        other_interval_loads=np.zeros((horizon, 6)),
        forecast_origin_time=first_time,
        lead_times_s=np.arange(1, horizon + 1, dtype=float) * replay.update_interval_s,
        provenance="explicit_zero_nonrotor_real_lstm_smoke_fixture",
    )
    source_bound = source.source_bound_forecast_and_current_enu_wind(0.0)
    if source_bound is None:
        raise RuntimeError("source-bound replay adapter unexpectedly returned no fact record")
    inferred_forecast, current_enu_downwind_air_velocity_mps = source_bound
    forecast = (
        inferred_forecast
        if scope == SOURCE_RECORD_CURRENT_AND_FORECAST
        else _current_observation_persistence_counterfactual(
            forecast=inferred_forecast,
            current_enu_downwind_air_velocity_mps=(
                current_enu_downwind_air_velocity_mps
            ),
        )
    )
    performance_table = resources.performance_table
    nominal_schedule = resources.nominal_schedule

    def assemble_relative_inflow(cycle_forecast: ForecastEvidence):
        return assemble_forecast_rotor_relative_inflow(
            forecast=cycle_forecast,
            current_enu_downwind_air_velocity_mps=current_enu_downwind_air_velocity_mps,
            current_wind_source="replay_origin_observation",
            current_wind_observation_time=first_time,
            parameters=parameters,
            kinematics=kinematics,
        )

    def operating_domain_report(cycle_forecast: ForecastEvidence) -> dict[str, Any]:
        relative_inflow = assemble_relative_inflow(cycle_forecast)
        return inspect_nominal_below_rated_operating_inputs(
            schedule=nominal_schedule,
            performance_table=performance_table,
            rotor_radius_m=parameters.rotor_radius_m,
            current_normal_inflow_speed_mps=(
                relative_inflow.current_normal_relative_inflow_mps
            ),
            future_normal_inflow_speeds_mps=(
                relative_inflow.future_normal_relative_inflows_mps
            ),
        )

    domain = operating_domain_report(forecast)
    if scope == SOURCE_RECORD_CURRENT_ONLY:
        domain = {
            **domain,
            "records": domain["records"][:1],
            "supported": bool(domain["records"][0]["supported"]),
            "unsupported_record_count": int(
                not bool(domain["records"][0]["supported"])
            ),
            "checked_input_scope": SOURCE_RECORD_CURRENT_ONLY,
        }
    else:
        domain = {
            **domain,
            "checked_input_scope": SOURCE_RECORD_CURRENT_AND_FORECAST,
        }
    if operating_domain_only or (
        return_unsupported_source_record and not domain["supported"]
    ):
        return {
            "purpose": "source_bound_nominal_rotor_operating_domain_preflight_only",
            "boundaries": {
                "is_controller_experiment": False,
                "is_performance_result": False,
                "converts_thrust_coefficient_to_load": False,
                "introduces_operating_state_fallback": False,
            },
            "source_record_scope": scope,
            "source_binding": {
                "origin_time": str(forecast.origin_time),
                "forecast_model_version": str(forecast.model_version),
                "forecast_horizon_steps": int(forecast.horizon_steps),
                "forecast_sample_period_s": float(forecast.sample_period_s),
                "inferred_forecast_model_version": str(
                    inferred_forecast.model_version
                ),
                "future_forecast_consumed": (
                    scope == SOURCE_RECORD_CURRENT_AND_FORECAST
                ),
            },
            "operating_domain": domain,
        }

    def assemble_cycle_facts(cycle_forecast: ForecastEvidence):
        relative_inflow = assemble_relative_inflow(cycle_forecast)
        operating_forecast = assemble_nominal_below_rated_operating_forecast(
            schedule=nominal_schedule,
            performance_table=performance_table,
            rotor_radius_m=parameters.rotor_radius_m,
            current_normal_inflow_speed_mps=(
                relative_inflow.current_normal_relative_inflow_mps
            ),
            future_normal_inflow_speeds_mps=(
                relative_inflow.future_normal_relative_inflows_mps
            ),
            operating_mode=NOMINAL_BELOW_RATED_GENERATING,
            operating_mode_source=(
                "explicit_real_lstm_smoke_fixture_declared_nominal_generating"
            ),
        )
        operating_state = kinematics.to_operating_state(
            current_thrust_coefficient=operating_forecast.current_thrust_coefficient,
            future_thrust_coefficients=(
                operating_forecast.future_thrust_coefficients
            ),
        )
        facts = assemble_first_interval_physical_cycle_input(
            forecast=cycle_forecast,
            current_enu_downwind_air_velocity_mps=current_enu_downwind_air_velocity_mps,
            current_wind_source="replay_origin_observation",
            current_wind_observation_time=first_time,
            rotor_load_parameters=parameters,
            rotor_operating_state=operating_state,
            platform_snapshot=snapshot,
            initial_platform_state=IncrementalState.zeros(),
            initial_platform_state_time=first_time,
            environment=environment,
            runtime_assembly=runtime,
            execution_state=execution,
            execution_config=config,
            execution_state_time=first_time,
        )
        _assert_relative_inflow_matches_final_load(
            facts=facts,
            relative_inflow=relative_inflow,
            parameters=parameters,
            kinematics=kinematics,
        )
        return facts, relative_inflow, operating_forecast

    facts, relative_inflow, operating_forecast = assemble_cycle_facts(forecast)
    first_interval_duration_s = float(facts.trajectory.steps[0].duration_s)
    if not np.isclose(config.block_duration_s, first_interval_duration_s):
        raise RuntimeError(
            "execution block duration must equal the first forecast interval "
            "for this first-interval smoke fixture"
        )
    admission = bind_physical_forecast_admission(
        decision_input=facts,
        forecast=forecast,
        admitted=True,
        basis=(
            "real_lstm_smoke_explicit_source_binding_only_not_a_control_admission_rule"
        ),
    )
    lifecycle_comparison = assemble_physical_lifecycle_comparison_for_current_block(
        decision_input=facts,
        forecast_admission=admission,
    )
    horizon_comparison = assemble_physical_lifecycle_horizon_comparison(
        current_block_comparison=lifecycle_comparison,
    )
    endpoint_held_responses = assemble_forecast_endpoint_held_responses(
        current_block_comparison=lifecycle_comparison,
    )

    load_assembly = facts.trajectory.load_assembly
    identity = load_assembly.input_identity
    if identity.forecast_horizon_steps != horizon:
        raise RuntimeError("dataset horizon does not match the source-bound forecast")
    result = {
        "purpose": (
            "real_lstm_to_rolling_current_cycle_execution_smoke_only"
            if include_current_cycle_decision and rolling_cycle_count > 1
            else "real_lstm_to_current_cycle_execution_smoke_only"
            if include_current_cycle_decision
            else "real_lstm_to_first_interval_physical_chain_smoke_only"
        ),
        "boundaries": {
            "is_controller_experiment": False,
            "is_candidate_selection": bool(include_current_cycle_decision),
            "is_performance_result": False,
            "is_platform_response_validation": False,
            "is_real_turbine_operating_state": False,
            "is_admission_rule_validation": False,
            "is_lifecycle_ranking": False,
            "is_held_request_horizon_control": False,
            "uses_source_bound_nominal_below_rated_schedule": True,
            "is_all_wind_direction_rotor_model": False,
        },
        "source_record_scope": scope,
        "source_binding": {
            "origin_time": identity.forecast_origin_time,
            "forecast_model_version": identity.forecast_model_version,
            "forecast_horizon_steps": identity.forecast_horizon_steps,
            "forecast_sample_period_s": identity.forecast_sample_period_s,
            "inferred_forecast_model_version": str(
                inferred_forecast.model_version
            ),
            "future_forecast_consumed": (
                scope == SOURCE_RECORD_CURRENT_AND_FORECAST
            ),
            "held_out_lead_labels_minutes": list(expected_lead_minutes),
            "held_out_lead_reliability": [
                float(value) for value in forecast.lead_reliability
            ],
            "held_out_lead_reliability_source": {
                "metrics_file": "lstm_regression_metrics.csv",
                "split": "validation",
                "metric": "vector_mae_ms",
                "role": "diagnostic_only_not_a_physical_load_weight",
            },
            "current_wind_source": load_assembly.input_identity.current_wind_source,
            "current_physical_input_enu_downwind_wind_mps": [
                float(value) for value in load_assembly.current_rotor_plane_enu_downwind_mps
            ],
            "first_forecast_physical_input_enu_downwind_wind_mps": [
                float(value)
                for value in load_assembly.future_rotor_plane_enu_downwind_mps[0]
            ],
            "last_forecast_physical_input_enu_downwind_wind_mps": [
                float(value)
                for value in load_assembly.future_rotor_plane_enu_downwind_mps[-1]
            ],
        },
        "operating_domain": domain,
        "admission_binding": {
            "admitted": admission.admitted,
            "basis": admission.basis,
            "forecast_origin_time": admission.forecast_identity.forecast_origin_time,
            "forecast_source": admission.forecast_identity.forecast_source,
            "forecast_model_version": (
                admission.forecast_identity.forecast_model_version
            ),
            "retains_raw_forecast_evidence": hasattr(admission, "forecast"),
        },
        "explicit_fixture_assumptions": {
            "nominal_below_rated_schedule": {
                "source_name": nominal_schedule.source_name,
                "source_sha256": nominal_schedule.source_sha256,
                "source_archive_sha256": nominal_schedule.source_archive_sha256,
                "normal_inflow_schedule_range_mps": [
                    nominal_schedule.cut_in_wind_speed_mps,
                    nominal_schedule.rated_wind_speed_mps,
                ],
                "operational_tip_speed_ratio": nominal_schedule.operational_tip_speed_ratio,
                "minimum_pitch_deg": nominal_schedule.minimum_pitch_deg,
                "dynamic_rosco_or_yaw_controller": False,
                "operating_mode": operating_forecast.operating_mode,
                "operating_mode_source": operating_forecast.operating_mode_source,
            },
            "performance_table": {
                "source_name": performance_table.source_name,
                "source_sha256": performance_table.source_sha256,
                "source_archive_sha256": performance_table.source_archive_sha256,
            },
            "resolved_rotor_operating_points": {
                "current": _operating_point_summary(operating_forecast.current),
                "first_future": _operating_point_summary(operating_forecast.future[0]),
                "last_future": _operating_point_summary(operating_forecast.future[-1]),
            },
            "frozen_platform_heading_rad": _FROZEN_HEADING_RAD,
            "wind_reference_height_m": parameters.wind_reference_height_m,
            "rotor_plane_height_m": parameters.rotor_plane_height_m,
            "power_law_shear_exponent": _TEST_SHEAR_EXPONENT,
            "future_platform_kinematics": "frozen_zero",
            "rotor_orientation": (
                "fixed_platform_heading_and_fixed_nacelle_yaw_positive_thrust_branch"
            ),
            "rotor_orientation_source": (
                kinematics.future_rotor_orientation_source
            ),
            "wave_and_other_loads": "explicit_zero",
            "platform_damping": "explicit_zero_matrix",
            "reference_platform_pitch_deg_for_axis_conversion": reference_state["pitch_deg"],
            "initial_tank_masses_kg": [
                float(value) for value in REFERENCE_TANK_MASSES_KG
            ],
            "tank_capacity_kg": TANK_CAPACITY_KG,
            "tank_coordinates_m": [
                [float(component) for component in coordinate]
                for coordinate in TANK_COORDINATES_M
            ],
            "initial_primary_target_masses_kg": [
                float(value) for value in execution.primary_target_masses_kg
            ],
            "water_density_kg_m3": config.water_density_kg_m3,
            "maximum_pump_rate_m3_min": config.max_pump_rate_m3_min,
        },
        "physical_chain": {
            "current_rotor_load_record": _rotor_load_record(
                load_assembly.current_rotor_load
            ),
            "future_rotor_load_records": [
                _rotor_load_record(load) for load in load_assembly.future_rotor_loads
            ],
            "current_generalized_rotor_load_platform": [
                float(value)
                for value in load_assembly.current_rotor_load.generalized_load_platform
            ],
            "first_future_generalized_rotor_load_platform": [
                float(value)
                for value in load_assembly.load_forecast.future_generalized_loads[0]
            ],
            "future_generalized_rotor_loads_platform": [
                [float(component) for component in load]
                for load in load_assembly.load_forecast.future_generalized_loads
            ],
            "forecast_load_lead_times_s": [
                float(value) for value in load_assembly.load_forecast.lead_times_s
            ],
            "first_interval_duration_s": first_interval_duration_s,
            "horizon_endpoint_previews": _horizon_endpoint_preview_summary(facts),
            "forecast_endpoint_held_responses": _endpoint_held_response_summary(
                endpoint_held_responses
            ),
            "forecast_endpoint_changes_from_continuation": (
                _endpoint_held_response_change_summary(
                    endpoint_held_responses,
                    horizon_comparison,
                )
            ),
            "lifecycle_responses": [
                _lifecycle_summary(facts.continue_existing),
                _lifecycle_summary(facts.release_to_current),
                _lifecycle_summary(facts.new_track),
            ],
            "execution_block_lifecycle_outcomes": (
                _execution_block_lifecycle_outcome_summary(lifecycle_comparison)
            ),
            "execution_block_changes_from_continuation": {
                "difference_convention": "alternative_minus_continuation",
                "outcomes": _execution_block_change_from_continuation_summary(
                    lifecycle_comparison
                ),
            },
            "held_request_forecast_horizon_outcomes": (
                _held_lifecycle_horizon_outcome_summary(horizon_comparison)
            ),
        },
    }
    if include_persistence_contrast:
        persistence_forecast = _current_observation_persistence_counterfactual(
            forecast=forecast,
            current_enu_downwind_air_velocity_mps=current_enu_downwind_air_velocity_mps,
        )
        (
            persistence_facts,
            persistence_relative_inflow,
            persistence_operating_forecast,
        ) = assemble_cycle_facts(persistence_forecast)
        result["persistence_counterfactual"] = _physical_contrast_summary(
            learned_facts=facts,
            persistence_facts=persistence_facts,
            learned_relative_inflow=relative_inflow,
            persistence_relative_inflow=persistence_relative_inflow,
            learned_operating_forecast=operating_forecast,
            persistence_operating_forecast=persistence_operating_forecast,
        )
    if include_current_cycle_decision:
        rolling_setup = PhysicalRollingChainSetup(
            runtime_assembly=runtime,
            reference_tank_masses_kg=REFERENCE_TANK_MASSES_KG,
            tank_capacities_kg=np.full(3, TANK_CAPACITY_KG),
            tank_coordinates_m=TANK_COORDINATES_M,
            rotor_load_parameters=parameters,
            execution_config=config,
            posture_limits=PhysicalPostureLimits(
                max_abs_pitch_roll_rad=np.deg2rad([15.0, 15.0]),
                max_abs_pitch_roll_rate_rad_s=[1.0, 1.0],
                source="explicit single-cycle semantic smoke envelope",
            ),
            decision_policy=PhysicalCurrentCycleDecisionPolicy(
                posture_equivalence_margin=0.05,
            ),
            minimum_lead_reliability=0.0,
            reliability_scope_basis=(
                "single-cycle semantic smoke retains each source-bound lead"
            ),
            future_platform_kinematics_source=(
                "current block uses actual entering platform velocity; future "
                "forecast leads use explicit frozen-zero kinematics"
            ),
        )

        platform_state = IncrementalState.zeros()
        execution_state = execution
        rolling_records = []
        rolling_origins: list[str] = []

        for cycle_index in range(rolling_cycle_count):
            if cycle_index == 0:
                cycle_forecast = forecast
                cycle_current_wind = current_enu_downwind_air_velocity_mps
            else:
                source_bound = source.source_bound_forecast_and_current_enu_wind(
                    cycle_index * config.block_duration_s
                )
                if source_bound is None:
                    raise RuntimeError(
                        "real replay source returned no same-origin observation and forecast "
                        f"for rolling cycle {cycle_index}"
                    )
                cycle_forecast, cycle_current_wind = source_bound

            cycle_origin = datetime.strptime(
                str(cycle_forecast.origin_time), TIMESTAMP_FMT
            )
            expected_origin = selected_origin + timedelta(
                seconds=cycle_index * config.block_duration_s
            )
            if cycle_origin != expected_origin:
                raise RuntimeError(
                    "replay forecast origin did not advance with the committed control "
                    "cadence"
                )
            if rolling_origins and cycle_forecast.origin_time == rolling_origins[-1]:
                raise RuntimeError("real replay rolling check reused a forecast origin")
            rolling_origins.append(str(cycle_forecast.origin_time))

            cycle_horizon = int(cycle_forecast.horizon_steps)
            cycle_kinematics = ForecastRotorKinematics(
                current_platform_reference_velocity_platform_mps=(
                    platform_state.velocity[:3]
                ),
                current_platform_angular_velocity_platform_radps=(
                    platform_state.velocity[3:]
                ),
                future_platform_reference_velocities_platform_mps=np.zeros(
                    (cycle_horizon, 3)
                ),
                future_platform_angular_velocities_platform_radps=np.zeros(
                    (cycle_horizon, 3)
                ),
                future_platform_kinematics_mode="frozen_zero",
                future_platform_kinematics_source=(
                    "explicit rolling semantic smoke frozen future kinematics"
                ),
                current_nacelle_yaw_relative_platform_rad=0.0,
                future_rotor_orientation_mode="frozen_current",
                future_rotor_orientation_source=(
                    "explicit fixed reference nacelle orientation rolling smoke"
                ),
            )
            cycle_relative_inflow = assemble_forecast_rotor_relative_inflow(
                forecast=cycle_forecast,
                current_enu_downwind_air_velocity_mps=cycle_current_wind,
                current_wind_source="replay_origin_observation",
                current_wind_observation_time=str(cycle_forecast.origin_time),
                parameters=parameters,
                kinematics=cycle_kinematics,
            )
            cycle_operating_forecast = assemble_nominal_below_rated_operating_forecast(
                schedule=nominal_schedule,
                performance_table=performance_table,
                rotor_radius_m=parameters.rotor_radius_m,
                current_normal_inflow_speed_mps=(
                    cycle_relative_inflow.current_normal_relative_inflow_mps
                ),
                future_normal_inflow_speeds_mps=(
                    cycle_relative_inflow.future_normal_relative_inflows_mps
                ),
                operating_mode=NOMINAL_BELOW_RATED_GENERATING,
                operating_mode_source=(
                    "explicit rolling semantic smoke nominal generating schedule"
                ),
            )
            zero_future_loads = np.zeros((cycle_horizon, 6))
            rolling_cycle = assemble_resolved_physical_rolling_cycle_input(
                forecast=cycle_forecast,
                current_enu_downwind_air_velocity_mps=cycle_current_wind,
                current_wind_source="replay_origin_observation",
                current_wind_observation_time=str(cycle_forecast.origin_time),
                start_time_s=cycle_index * config.block_duration_s,
                current_thrust_coefficient=(
                    cycle_operating_forecast.current_thrust_coefficient
                ),
                future_thrust_coefficients=(
                    cycle_operating_forecast.future_thrust_coefficients
                ),
                thrust_coefficient_source=(
                    "source_bound_nominal_below_rated_schedule_with_entering_state"
                ),
                current_nacelle_yaw_relative_platform_rad=(
                    cycle_kinematics.current_nacelle_yaw_relative_platform_rad
                ),
                current_nacelle_yaw_source=(
                    cycle_kinematics.future_rotor_orientation_source
                ),
                forecast_admitted=True,
                forecast_admission_basis=(
                    "real_lstm_rolling_semantic_smoke_explicit_source_binding"
                ),
                current_wave_load=np.zeros(6),
                current_other_load=np.zeros(6),
                future_wave_interval_loads=zero_future_loads,
                future_other_interval_loads=zero_future_loads,
                nonrotor_load_provenance=(
                    "explicit_zero_nonrotor_real_lstm_rolling_semantic_smoke"
                ),
            )
            record = advance_physical_rolling_control_cycle(
                setup=rolling_setup,
                cycle=rolling_cycle,
                platform_state=platform_state,
                execution_state=execution_state,
            )
            if record.status != "completed":
                raise RuntimeError(
                    "current-cycle selector abstained; rolling semantic smoke does not "
                    "insert a fallback command"
                )
            cycle_facts = (
                record.decision.response_set.current_block_comparison.decision_input
            )
            _assert_relative_inflow_matches_final_load(
                facts=cycle_facts,
                relative_inflow=cycle_relative_inflow,
                parameters=parameters,
                kinematics=cycle_kinematics,
            )
            if not np.allclose(
                record.actual_rotor_load.generalized_load_platform,
                cycle_facts.trajectory.load_assembly.current_rotor_load.generalized_load_platform,
                rtol=0.0,
                atol=1.0e-9,
            ):
                raise RuntimeError(
                    "rolling current load diverged from the source-bound current-cycle fact"
                )
            rolling_records.append(record)
            assert record.end_platform_state is not None
            assert record.end_execution_state is not None
            platform_state = record.end_platform_state
            execution_state = record.end_execution_state

        rolling_trace = {
            "purpose": "real_lstm_current_cycle_selection_to_modeled_state_handoff_only",
            "status": "completed",
            "planned_cycle_count": rolling_cycle_count,
            "completed_cycle_count": len(rolling_records),
            "records": [record.as_trace() for record in rolling_records],
            "final_platform_position": [
                float(value) for value in platform_state.position
            ],
            "final_actual_tank_masses_kg": [
                float(value) for value in execution_state.actual_masses_kg
            ],
        }
        result["current_cycle_decision_smoke"] = {
            "purpose": (
                "real-LSTM source-bound current-cycle selection and modeled state "
                "handoff only, not a control-performance result"
            ),
            "requested_cycle_count": rolling_cycle_count,
            "resolved_cycle_count": len(rolling_records),
            "state_handoff_basis": (
                "each later cycle starts from the preceding low-order model output, "
                "not a new platform or tank measurement"
            ),
            "forecast_origins": rolling_origins,
            "origin_cadence_matches_committed_control_blocks": True,
            "each_current_load_matches_its_source_bound_physical_fact": True,
            "future_kinematics": (
                "current entering platform velocity is used only for the current "
                "load; future forecast leads use explicit frozen-zero kinematics"
            ),
            "trace": rolling_trace,
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--origin",
        help="test-split replay history end in YYYY-MM-DD HH:MM:SS; default is first test sample",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--include-persistence-contrast",
        action="store_true",
        help="report a validation-only current-observation persistence contrast",
    )
    parser.add_argument(
        "--include-current-cycle-decision",
        action="store_true",
        help="run a source-bound rolling current-cycle selection and execution smoke check",
    )
    parser.add_argument(
        "--rolling-cycle-count",
        type=int,
        default=1,
        help="number of real replay cycles for the semantic smoke check (1 to 3)",
    )
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()
    origin = None if args.origin is None else datetime.strptime(args.origin, TIMESTAMP_FMT)
    payload = json.dumps(
        run_smoke(
            origin=origin,
            device=args.device,
            include_persistence_contrast=args.include_persistence_contrast,
            include_current_cycle_decision=args.include_current_cycle_decision,
            rolling_cycle_count=args.rolling_cycle_count,
        ),
        ensure_ascii=False,
        indent=2,
    )
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
