"""Run a few fixed six-hour controller-chain structural checks.

This is deliberately not a performance experiment.  Each case uses synthetic
forecast evidence and an explicit zero-thrust rotor operating state, so it can
exercise the controller-to-pump-to-platform state handoff without pretending
to validate the current low-order aerodynamic response.  The output is a
compact JSON trace that retains the decision and committed-execution facts for
every 20-minute control cycle.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import json
import math
from pathlib import Path

import numpy as np

from fowt_platform import (
    IncrementalState,
    assemble_volturnus_static_restoring_aligned_runtime_assembly,
)
from wind_prediction.controller_core import ControlCoreConfig
from wind_prediction.execution_rollout import ExecutionRolloutConfig, ExecutionRolloutState
from wind_prediction.fixed_case_chain_check import (
    FixedChainCheckSetup,
    FixedChainCycleInput,
    FrozenCurrentRotorLoadAssumptions,
    run_fixed_case_chain_check,
)
from wind_prediction.forecast_action_policy import ForecastActionPolicyConfig
from wind_prediction.forecast_evidence import ForecastEvidence
from wind_prediction.forecast_physical_load import ForecastRotorLoadParameters


ROOT = Path(__file__).resolve().parents[2]
REFERENCE_MANIFEST = (
    ROOT
    / "configs"
    / "reference_platforms"
    / "volturnus_s_openfast_v1_1_16.json"
)
REFERENCE_TANK_MASSES_KG = np.array([1_108_000.0, 1_362_000.0, 1_362_000.0])
TANK_CAPACITY_KG = 1_896_250.0
TANK_COORDINATES_M = np.array(
    [
        [46.2, 0.0, -10.0],
        [-23.1, 46.2 * 0.866, -10.0],
        [-23.1, -46.2 * 0.866, -10.0],
    ]
)
_ORIGIN = datetime(2026, 8, 22, tzinfo=timezone.utc)
_CYCLE_COUNT = 18


@dataclass(frozen=True)
class StructuralCase:
    """One fixed six-hour sequence used only to check state handoff."""

    name: str
    initial_pitch_deg: float
    initial_roll_deg: float
    current_wind_uv_ms: tuple[float, float]
    future_wind_uv_ms: tuple[tuple[float, float], ...]


def _controller_config() -> ControlCoreConfig:
    execution = ExecutionRolloutConfig(
        block_duration_s=1200.0,
        internal_step_s=1.0,
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
    return ControlCoreConfig(
        stage_duration_s=1200.0,
        stage_count=3,
        execution=execution,
        posture_priority_envelope_deg=(3.0, 3.0),
        w_residual=80.0,
        w_terminal_residual=80.0,
        w_pump_volume=0.2,
        forecast_policy=ForecastActionPolicyConfig(
            enabled=True,
            stage_duration_s=1200.0,
            high_impact_reliability_min=0.65,
            high_impact_event_probability_min=0.60,
        ),
    )


def _setup() -> FixedChainCheckSetup:
    config = _controller_config()
    runtime = assemble_volturnus_static_restoring_aligned_runtime_assembly(
        REFERENCE_MANIFEST,
        np.zeros((6, 6)),
    )
    parameters = ForecastRotorLoadParameters(
        air_density_kg_m3=1.225,
        rotor_radius_m=100.0,
        wind_reference_height_m=100.0,
        rotor_plane_height_m=100.0,
        power_law_shear_exponent=0.0,
        frozen_equilibrium_heading_rad=np.pi,
        rotor_force_application_point_from_reference_m=[0.0, 0.0, 150.0],
        downwind_rotor_normal_platform=[1.0, 0.0, 0.0],
    )
    return FixedChainCheckSetup(
        controller_config=config,
        runtime_assembly=runtime,
        reference_tank_masses_kg=REFERENCE_TANK_MASSES_KG,
        tank_capacities_kg=np.full(3, TANK_CAPACITY_KG),
        tank_coordinates_m=TANK_COORDINATES_M,
        current_rotor_load=FrozenCurrentRotorLoadAssumptions(
            parameters=parameters,
            current_thrust_coefficient=0.0,
            thrust_coefficient_source="explicit_zero_thrust_structural_fixture",
        ),
    )


def _forecast(*, origin_time: str, vectors: tuple[tuple[float, float], ...]) -> ForecastEvidence:
    return ForecastEvidence(
        source="fixed_synthetic_forecast_fixture",
        model_version="structural-chain-v1",
        origin_time=origin_time,
        sample_period_s=600.0,
        uv_ms=np.asarray(vectors, dtype=float),
        lead_reliability=np.full(len(vectors), 0.9),
        event_probs={
            "attention_event_0_20m": 0.9,
            "attention_event_20_40m": 0.9,
            "attention_event_40_60m": 0.9,
        },
        provides_future_preview=True,
    )


def _cycle_inputs(case: StructuralCase) -> tuple[FixedChainCycleInput, ...]:
    cycles: list[FixedChainCycleInput] = []
    for index in range(_CYCLE_COUNT):
        timestamp = _ORIGIN + timedelta(seconds=1200 * index)
        origin_time = timestamp.isoformat().replace("+00:00", "Z")
        cycles.append(
            FixedChainCycleInput(
                start_time_s=1200.0 * index,
                current_wind_uv_ms=case.current_wind_uv_ms,
                current_wind_source="fixed_synthetic_current_observation",
                current_wind_observation_time=origin_time,
                forecast=_forecast(
                    origin_time=origin_time,
                    vectors=case.future_wind_uv_ms,
                ),
            )
        )
    return tuple(cycles)


def _initial_execution_state() -> ExecutionRolloutState:
    return ExecutionRolloutState(
        masses_kg=REFERENCE_TANK_MASSES_KG,
        target_masses_kg=REFERENCE_TANK_MASSES_KG,
        primary_target_kg=REFERENCE_TANK_MASSES_KG,
        pump_rates_m3_min=np.zeros(3),
        pump_latched=np.zeros(3, dtype=bool),
    )


def _initial_platform_state(case: StructuralCase) -> IncrementalState:
    return IncrementalState(
        position=np.array(
            [
                0.0,
                0.0,
                0.0,
                np.radians(case.initial_roll_deg),
                np.radians(case.initial_pitch_deg),
                0.0,
            ],
            dtype=float,
        ),
        velocity=np.zeros(6),
    )


def structural_cases() -> tuple[StructuralCase, ...]:
    """Return three distinct decision-evidence sequences, not tuned cases."""

    return (
        StructuralCase(
            name="steady_future_feedback_correction",
            initial_pitch_deg=4.5,
            initial_roll_deg=0.0,
            current_wind_uv_ms=(0.0, -5.0),
            future_wind_uv_ms=((0.0, -5.0),) * 6,
        ),
        StructuralCase(
            name="sustained_future_strengthening",
            initial_pitch_deg=4.5,
            initial_roll_deg=0.0,
            current_wind_uv_ms=(0.0, -5.0),
            future_wind_uv_ms=(
                (0.0, -8.0),
                (0.0, -10.0),
                (0.0, -12.0),
                (0.0, -12.0),
                (0.0, -12.0),
                (0.0, -12.0),
            ),
        ),
        StructuralCase(
            name="gradual_directional_turn",
            initial_pitch_deg=4.0,
            initial_roll_deg=1.5,
            current_wind_uv_ms=(0.0, -5.0),
            future_wind_uv_ms=(
                (0.0, -8.0),
                (1.0, -8.0),
                (2.0, -7.0),
                (3.0, -6.0),
                (2.0, -5.0),
                (1.0, -4.0),
            ),
        ),
    )


def _case_summary(result) -> dict[str, object]:
    actions: dict[str, int] = {}
    max_abs_pitch_deg = 0.0
    max_abs_roll_deg = 0.0
    transferred_volume_m3 = 0.0
    for record in result.records:
        actions[record.selected_action] = actions.get(record.selected_action, 0) + 1
        transferred_volume_m3 += record.committed_transferred_volume_m3
        for position in (record.start_platform_position, record.end_platform_position):
            max_abs_roll_deg = max(max_abs_roll_deg, abs(math.degrees(position[3])))
            max_abs_pitch_deg = max(max_abs_pitch_deg, abs(math.degrees(position[4])))
    return {
        "case_name": result.case_name,
        "status": result.status,
        "planned_cycle_count": result.planned_cycle_count,
        "attempted_cycle_count": result.attempted_cycle_count,
        "completed_cycle_count": result.completed_cycle_count,
        "selected_action_counts": actions,
        "committed_transferred_volume_m3": transferred_volume_m3,
        "max_abs_pitch_deg_for_chain_sanity_only": max_abs_pitch_deg,
        "max_abs_roll_deg_for_chain_sanity_only": max_abs_roll_deg,
    }


def run_structural_casebook() -> dict[str, object]:
    """Run the fixed casebook and return audit traces without aggregation."""

    setup = _setup()
    traces = []
    summaries = []
    for case in structural_cases():
        result = run_fixed_case_chain_check(
            case_name=case.name,
            setup=setup,
            cycles=_cycle_inputs(case),
            initial_platform_state=_initial_platform_state(case),
            initial_execution_state=_initial_execution_state(),
        )
        traces.append(result.as_trace())
        summaries.append(_case_summary(result))
    return {
        "purpose": "structural_state_handoff_only_not_a_performance_experiment",
        "assumption_ledger": {
            "control_cycle_s": 1200.0,
            "cycle_count_per_case": _CYCLE_COUNT,
            "forecast_points_per_cycle": 6,
            "forecast_sample_period_s": 600.0,
            "rotor_operating_state": "explicit_zero_thrust_structural_fixture",
            "current_thrust_coefficient": 0.0,
            "current_thrust_coefficient_source": (
                "explicit_zero_thrust_structural_fixture"
            ),
            "platform_damping": "explicit_zero_matrix_structural_fixture",
            "wave_load": "explicit_zero",
            "other_load": "explicit_zero",
            "future_forecast_load_applied_to_platform": False,
            "platform_runtime_provenance": setup.runtime_assembly.provenance,
        },
        "case_count": len(traces),
        "all_cases_completed": all(trace["completed"] for trace in traces),
        "summaries": summaries,
        "cases": traces,
    }


def main() -> None:
    report = run_structural_casebook()
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["all_cases_completed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
