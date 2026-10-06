from pathlib import Path
import unittest

import numpy as np

from fowt_platform import (
    assemble_volturnus_static_restoring_aligned_runtime_assembly,
    IncrementalState,
)
from wind_prediction.controller_core import ControlCoreConfig
from wind_prediction.execution_rollout import (
    ExecutionRolloutConfig,
    ExecutionRolloutState,
    ExecutionTargetOperation,
)
from wind_prediction.fixed_case_chain_check import (
    FixedChainCheckSetup,
    FixedChainCycleInput,
    FrozenCurrentRotorLoadAssumptions,
    run_fixed_case_chain_check,
)
from wind_prediction.forecast_action_policy import ForecastActionPolicyConfig
from wind_prediction.forecast_evidence import ForecastEvidence
from wind_prediction.forecast_physical_load import ForecastRotorLoadParameters


ROOT = Path(__file__).resolve().parents[1]
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


def _config(*, pump_weight: float = 0.2) -> ControlCoreConfig:
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
        w_pump_volume=pump_weight,
        forecast_policy=ForecastActionPolicyConfig(
            enabled=True,
            stage_duration_s=1200.0,
            high_impact_reliability_min=0.65,
            high_impact_event_probability_min=0.60,
        ),
    )


def _forecast(
    *,
    origin: str,
    speed_ms: float = 8.0,
    points: int = 6,
    future_vectors=None,
) -> ForecastEvidence:
    vectors = (
        np.array([(0.0, -speed_ms)] * points, dtype=float)
        if future_vectors is None
        else np.asarray(future_vectors, dtype=float)
    )
    return ForecastEvidence(
        source="fixed_check_test_forecast",
        model_version="test-v1",
        origin_time=origin,
        sample_period_s=600.0,
        uv_ms=vectors,
        lead_reliability=np.full(vectors.shape[0], 0.9),
        event_probs={
            "attention_event_0_20m": 0.9,
            "attention_event_20_40m": 0.9,
            "attention_event_40_60m": 0.9,
        },
        provides_future_preview=True,
    )


def _setup(config: ControlCoreConfig) -> FixedChainCheckSetup:
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
            current_thrust_coefficient=0.8,
            thrust_coefficient_source="explicit_test_operating_point",
        ),
    )


def _initial_execution(*, target_offset=(0.0, 0.0, 0.0)) -> ExecutionRolloutState:
    target = REFERENCE_TANK_MASSES_KG + np.asarray(target_offset, dtype=float)
    return ExecutionRolloutState(
        masses_kg=REFERENCE_TANK_MASSES_KG,
        target_masses_kg=target,
        primary_target_kg=target,
        pump_rates_m3_min=np.zeros(3),
        pump_latched=np.zeros(3, dtype=bool),
    )


class FixedCaseChainCheckTests(unittest.TestCase):
    def test_two_track_cycles_use_actual_end_state_as_the_next_cycle_input(self):
        config = _config()
        initial_platform = IncrementalState(
            position=np.array([0.0, 0.0, 0.0, 0.0, np.radians(4.5), 0.0]),
            velocity=np.zeros(6),
        )
        cycles = (
            FixedChainCycleInput(
                start_time_s=0.0,
                current_wind_uv_ms=(0.0, -5.0),
                current_wind_source="fixed_test_observation_0",
                current_wind_observation_time="2026-08-22T00:00:00",
                forecast=_forecast(origin="2026-08-22T00:00:00"),
            ),
            FixedChainCycleInput(
                start_time_s=1200.0,
                current_wind_uv_ms=(0.0, -5.0),
                current_wind_source="fixed_test_observation_1",
                current_wind_observation_time="2026-08-22T00:20:00",
                forecast=_forecast(origin="2026-08-22T00:20:00"),
            ),
        )

        result = run_fixed_case_chain_check(
            case_name="two_track_cycles",
            setup=_setup(config),
            cycles=cycles,
            initial_platform_state=initial_platform,
            initial_execution_state=_initial_execution(),
        )

        self.assertTrue(result.completed)
        self.assertEqual(len(result.records), 2)
        first, second = result.records
        self.assertEqual(first.status, "completed")
        self.assertEqual(second.status, "completed")
        self.assertEqual(first.execution_operation, ExecutionTargetOperation.TRACK.value)
        self.assertEqual(second.execution_operation, ExecutionTargetOperation.TRACK.value)
        self.assertEqual(
            second.start_actual_tank_masses_kg,
            first.end_actual_tank_masses_kg,
        )
        self.assertEqual(
            second.start_platform_position,
            first.end_platform_position,
        )
        self.assertEqual(
            first.current_wind_load_rule,
            "current_observation_and_platform_motion_at_cycle_start_frozen_over_control_block",
        )
        self.assertFalse(first.future_forecast_load_applied_to_platform)
        self.assertEqual(first.wave_load_source, "explicit_zero")
        self.assertEqual(first.other_load_source, "explicit_zero")
        self.assertGreater(np.linalg.norm(first.wind_generalized_load), 0.0)

    def test_release_advances_the_lifecycle_without_reinterpreting_it_as_track(self):
        config = _config(pump_weight=100.0)
        initial_platform = IncrementalState.zeros()
        initial_execution = _initial_execution(target_offset=(10_000.0, -5_000.0, -5_000.0))
        cycle = FixedChainCycleInput(
            start_time_s=0.0,
            current_wind_uv_ms=(0.0, -12.0),
            current_wind_source="fixed_test_observation_release",
            current_wind_observation_time="2026-08-22T00:00:00",
            forecast=_forecast(
                origin="2026-08-22T00:00:00",
                speed_ms=10.0,
            ),
        )

        result = run_fixed_case_chain_check(
            case_name="release_lifecycle_path",
            setup=_setup(config),
            cycles=(cycle,),
            initial_platform_state=initial_platform,
            initial_execution_state=initial_execution,
        )

        self.assertTrue(result.completed)
        self.assertEqual(result.status, "completed")
        self.assertEqual(result.planned_cycle_count, 1)
        self.assertEqual(result.attempted_cycle_count, 1)
        self.assertEqual(result.completed_cycle_count, 1)
        self.assertEqual(len(result.records), 1)
        record = result.records[0]
        self.assertEqual(record.status, "completed")
        self.assertEqual(
            record.execution_operation,
            ExecutionTargetOperation.RELEASE_TO_CURRENT.value,
        )
        self.assertEqual(record.start_actual_tank_masses_kg, record.end_actual_tank_masses_kg)
        np.testing.assert_allclose(
            result.final_execution_state.actual_masses_kg,
            initial_execution.actual_masses_kg,
        )
        trace = result.as_trace()
        self.assertEqual(trace["planned_cycle_count"], 1)
        self.assertEqual(trace["attempted_cycle_count"], 1)
        self.assertEqual(trace["completed_cycle_count"], 1)
        committed = trace["records"][0]["committed_execution"]
        self.assertEqual(committed["role"], "actual_p4_path_aggregate")
        self.assertTrue(committed["advanced"])
        self.assertEqual(committed["active_time_definition"], "aggregate_pump_seconds")
        self.assertGreater(
            np.linalg.norm(result.final_platform_state.position - initial_platform.position),
            0.0,
        )

    def test_incomplete_forecast_tail_is_rejected_before_any_platform_step(self):
        config = _config()
        cycle = FixedChainCycleInput(
            start_time_s=0.0,
            current_wind_uv_ms=(0.0, -5.0),
            current_wind_source="fixed_test_observation_short_forecast",
            current_wind_observation_time="2026-08-22T00:00:00",
            forecast=_forecast(
                origin="2026-08-22T00:00:00",
                points=5,
            ),
        )

        with self.assertRaisesRegex(ValueError, "need 6 points, got 5"):
            run_fixed_case_chain_check(
                case_name="reject_short_tail",
                setup=_setup(config),
                cycles=(cycle,),
                initial_platform_state=IncrementalState.zeros(),
                initial_execution_state=_initial_execution(),
            )

    def test_future_forecast_change_does_not_change_current_cycle_wind_load(self):
        config = _config()
        initial_platform = IncrementalState(
            position=np.array([0.0, 0.0, 0.0, 0.0, np.radians(4.5), 0.0]),
            velocity=np.zeros(6),
        )
        shared = {
            "start_time_s": 0.0,
            "current_wind_uv_ms": (0.0, -5.0),
            "current_wind_source": "fixed_test_same_current_observation",
            "current_wind_observation_time": "2026-08-22T00:00:00",
        }
        calm_future = FixedChainCycleInput(
            **shared,
            forecast=_forecast(
                origin="2026-08-22T00:00:00",
                future_vectors=[(0.0, -1.0)] * 6,
            ),
        )
        changing_future = FixedChainCycleInput(
            **shared,
            forecast=_forecast(
                origin="2026-08-22T00:00:00",
                future_vectors=[
                    (8.0, -18.0),
                    (10.0, -20.0),
                    (-9.0, -15.0),
                    (-12.0, 4.0),
                    (11.0, 2.0),
                    (7.0, -19.0),
                ],
            ),
        )

        calm = run_fixed_case_chain_check(
            case_name="same_current_calm_future",
            setup=_setup(config),
            cycles=(calm_future,),
            initial_platform_state=initial_platform,
            initial_execution_state=_initial_execution(),
        )
        changing = run_fixed_case_chain_check(
            case_name="same_current_changing_future",
            setup=_setup(config),
            cycles=(changing_future,),
            initial_platform_state=initial_platform,
            initial_execution_state=_initial_execution(),
        )

        self.assertEqual(calm.records[0].wind_generalized_load, changing.records[0].wind_generalized_load)
        self.assertFalse(calm.records[0].future_forecast_load_applied_to_platform)
        self.assertFalse(changing.records[0].future_forecast_load_applied_to_platform)

    def test_cycle_trace_keeps_decision_evidence_separate_from_committed_execution(self):
        config = _config()
        cycle = FixedChainCycleInput(
            start_time_s=0.0,
            current_wind_uv_ms=(0.0, -5.0),
            current_wind_source="fixed_test_trace_observation",
            current_wind_observation_time="2026-08-22T00:00:00",
            forecast=_forecast(
                origin="2026-08-22T00:00:00",
                future_vectors=[
                    (0.0, -8.0),
                    (0.0, -10.0),
                    (0.0, -12.0),
                    (0.0, -12.0),
                    (0.0, -12.0),
                    (0.0, -12.0),
                ],
            ),
        )
        result = run_fixed_case_chain_check(
            case_name="decision_trace",
            setup=_setup(config),
            cycles=(cycle,),
            initial_platform_state=IncrementalState(
                position=np.array([0.0, 0.0, 0.0, 0.0, np.radians(4.5), 0.0]),
                velocity=np.zeros(6),
            ),
            initial_execution_state=_initial_execution(),
        )

        trace = result.as_trace()
        record = trace["records"][0]
        self.assertEqual(record["status"], "completed")
        self.assertFalse(record["rotor_load"]["future_forecast_applied_to_platform"])
        selected_sequence = record["decision"]["selection"]["selected_sequence"]
        self.assertEqual(len(selected_sequence), 3)
        self.assertEqual(selected_sequence[0], "strengthen")
        self.assertTrue(
            record["decision"]["selection"]["selected_action_authorization"]
            ["allowed"]
        )
        self.assertEqual(len(record["decision"]["selected_stage_rollout"]), 3)
        self.assertGreaterEqual(
            record["committed_execution"]["transferred_volume_m3"],
            0.0,
        )


if __name__ == "__main__":
    unittest.main()
