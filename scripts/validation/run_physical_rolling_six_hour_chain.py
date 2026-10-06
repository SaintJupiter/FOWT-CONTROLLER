"""Run one fixed six-hour physical decision-to-execution chain check.

This fixture verifies only that a source-bound forecast can form one physical
current-cycle decision, that exactly that request is committed, and that the
resulting actual pump and platform state seeds the following control cycle.
It deliberately uses an explicit zero-thrust rotor fixture and zero non-rotor
loads.  It is therefore not a controller-performance or platform-response
validation experiment.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
import math
from pathlib import Path

import numpy as np

from fowt_platform import IncrementalState, assemble_volturnus_static_restoring_aligned_runtime_assembly
from wind_prediction.execution_rollout import ExecutionRolloutConfig, ExecutionRolloutState
from wind_prediction.forecast_evidence import ForecastEvidence
from wind_prediction.forecast_physical_load import ForecastRotorLoadParameters
from wind_prediction.forecast_platform_trajectory import ForecastTrajectoryEnvironment
from wind_prediction.physical_current_cycle_decision import PhysicalCurrentCycleDecisionPolicy
from wind_prediction.physical_lifecycle_selection import PhysicalPostureLimits
from wind_prediction.physical_rolling_control_chain import (
    PhysicalRollingChainSetup,
    PhysicalRollingCycleInput,
    run_physical_rolling_control_chain,
)


ROOT = Path(__file__).resolve().parents[2]
REFERENCE_MANIFEST = ROOT / "configs/reference_platforms/volturnus_s_openfast_v1_1_16.json"
REFERENCE_TANK_MASSES_KG = np.array([1_108_000.0, 1_362_000.0, 1_362_000.0])
TANK_CAPACITY_KG = 1_896_250.0
TANK_COORDINATES_M = np.array(
    [
        [46.2, 0.0, -10.0],
        [-23.1, 46.2 * 0.866, -10.0],
        [-23.1, -46.2 * 0.866, -10.0],
    ]
)
_ORIGIN = datetime(2026, 8, 25, tzinfo=timezone.utc)
_BLOCK_DURATION_S = 1_200.0
_CYCLE_COUNT = 18


def _setup() -> PhysicalRollingChainSetup:
    execution = ExecutionRolloutConfig(
        block_duration_s=_BLOCK_DURATION_S,
        # This is a state-handoff fixture, not a time-step convergence study.
        # A 30 s substep keeps the six-hour semantic check short enough that it
        # does not turn into an accidental platform-integration benchmark.
        internal_step_s=30.0,
        water_density_kg_m3=1025.0,
        max_pump_rate_m3_min=1.0,
        target_slew_enabled=False,
        stop_error_kg=0.0,
        restart_error_kg=0.0,
        min_on_s=0.0,
        min_off_s=0.0,
        near_target_hold_s=0.0,
        ramp_up_m3_min_per_s=np.inf,
        ramp_down_m3_min_per_s=np.inf,
        pump_rate_schedule_m3_min=((0.0, 1.0), (20_000.0, 1.0)),
        tank_capacity_kg=TANK_CAPACITY_KG,
    )
    return PhysicalRollingChainSetup(
        runtime_assembly=assemble_volturnus_static_restoring_aligned_runtime_assembly(
            REFERENCE_MANIFEST,
            np.zeros((6, 6)),
        ),
        reference_tank_masses_kg=REFERENCE_TANK_MASSES_KG,
        tank_capacities_kg=np.full(3, TANK_CAPACITY_KG),
        tank_coordinates_m=TANK_COORDINATES_M,
        rotor_load_parameters=ForecastRotorLoadParameters(
            air_density_kg_m3=1.225,
            rotor_radius_m=100.0,
            wind_reference_height_m=100.0,
            rotor_plane_height_m=100.0,
            power_law_shear_exponent=0.0,
            frozen_equilibrium_heading_rad=np.pi,
            rotor_force_application_point_from_reference_m=[0.0, 0.0, 150.0],
            downwind_rotor_normal_platform=[1.0, 0.0, 0.0],
        ),
        execution_config=execution,
        posture_limits=PhysicalPostureLimits(
            max_abs_pitch_roll_rad=[math.radians(15.0), math.radians(15.0)],
            max_abs_pitch_roll_rate_rad_s=[1.0, 1.0],
            source="wide structural chain check envelope",
        ),
        decision_policy=PhysicalCurrentCycleDecisionPolicy(
            posture_equivalence_margin=0.05,
        ),
        minimum_lead_reliability=0.8,
        reliability_scope_basis="fixed structural fixture scope",
        future_platform_kinematics_source="explicit frozen-zero structural fixture",
    )


def _cycle(index: int) -> PhysicalRollingCycleInput:
    timestamp = _ORIGIN + timedelta(seconds=_BLOCK_DURATION_S * index)
    origin = timestamp.isoformat().replace("+00:00", "Z")
    future = np.array(
        [
            [0.0, -6.0],
            [0.0, -9.0],
            [0.0, -6.0],
            [0.0, -6.0],
            [0.0, -6.0],
            [0.0, -6.0],
        ],
        dtype=float,
    )
    forecast = ForecastEvidence(
        source="fixed_structural_forecast_fixture",
        model_version="physical-rolling-chain-v1",
        origin_time=origin,
        sample_period_s=_BLOCK_DURATION_S,
        uv_ms=future,
        lead_reliability=np.full(future.shape[0], 0.9),
        event_probs={"structural_fixture": 0.0},
        provides_future_preview=True,
    )
    return PhysicalRollingCycleInput(
        start_time_s=_BLOCK_DURATION_S * index,
        forecast=forecast,
        current_enu_downwind_air_velocity_mps=[0.0, -5.0],
        current_wind_source="fixed_structural_current_observation",
        current_wind_observation_time=origin,
        current_thrust_coefficient=0.0,
        current_nacelle_yaw_relative_platform_rad=0.0,
        current_nacelle_yaw_source="explicit fixed structural zero-yaw fixture",
        future_thrust_coefficients=np.zeros(future.shape[0]),
        thrust_coefficient_source="explicit_zero_thrust_structural_fixture",
        forecast_admitted=True,
        forecast_admission_basis="structural source-binding fixture",
        environment=ForecastTrajectoryEnvironment(
            current_wave_load=np.zeros(6),
            current_other_load=np.zeros(6),
            wave_interval_loads=np.zeros((future.shape[0], 6)),
            other_interval_loads=np.zeros((future.shape[0], 6)),
            forecast_origin_time=origin,
            lead_times_s=np.arange(1, future.shape[0] + 1, dtype=float)
            * _BLOCK_DURATION_S,
            provenance="explicit zero nonrotor structural fixture",
        ),
    )


def run_fixed_six_hour_physical_rolling_chain() -> dict[str, object]:
    """Run the structural six-hour fixture and return its decision trace."""

    setup = _setup()
    initial_execution = ExecutionRolloutState(
        masses_kg=REFERENCE_TANK_MASSES_KG,
        target_masses_kg=REFERENCE_TANK_MASSES_KG,
        primary_target_kg=REFERENCE_TANK_MASSES_KG,
        pump_rates_m3_min=np.zeros(3),
        pump_latched=np.zeros(3, dtype=bool),
        pump_off_elapsed_s=np.full(3, 60.0),
    )
    result = run_physical_rolling_control_chain(
        setup=setup,
        cycles=tuple(_cycle(index) for index in range(_CYCLE_COUNT)),
        initial_platform_state=IncrementalState.zeros(),
        initial_execution_state=initial_execution,
    )
    trace = result.as_trace()
    trace["assumption_ledger"] = {
        "control_cycle_s": _BLOCK_DURATION_S,
        "cycle_count": _CYCLE_COUNT,
        "forecast_points_per_cycle": 6,
        "rotor_operating_state": "explicit_zero_thrust_structural_fixture",
        "actual_platform_rotor_load_basis": "same-origin current load assembled for decision",
        "wave_load": "explicit_zero",
        "other_load": "explicit_zero",
        "result_role": "state_handoff_semantic_check_only_not_a_performance_experiment",
    }
    return trace


def main() -> None:
    result = run_fixed_six_hour_physical_rolling_chain()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["completed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
