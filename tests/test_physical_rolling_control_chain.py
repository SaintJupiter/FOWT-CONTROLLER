from dataclasses import replace
from datetime import datetime, timedelta
import unittest

import numpy as np

from fowt_platform import (
    BallastMassProperties,
    BallastRuntimeAssembly,
    IncrementalState,
    PlatformMatrices,
    weight_stiffness_about_reference,
)
from fowt_platform.rotor_input import quasi_steady_rotor_generalized_load_from_enu_wind
from wind_prediction.execution_rollout import ExecutionRolloutConfig, ExecutionRolloutState
from wind_prediction.forecast_evidence import ForecastEvidence
from wind_prediction.forecast_physical_load import (
    ForecastRotorLoadParameters,
    rotor_geometry_at_declared_nacelle_yaw,
)
from wind_prediction.forecast_platform_trajectory import ForecastTrajectoryEnvironment
from wind_prediction.physical_current_cycle_decision import (
    PhysicalCurrentCycleDecisionPolicy,
)
from wind_prediction.physical_lifecycle_selection import PhysicalPostureLimits
from wind_prediction.physical_rolling_control_chain import (
    PhysicalRollingChainSetup,
    PhysicalRollingCycleInput,
    advance_physical_rolling_control_cycle,
    run_physical_rolling_control_chain,
)


_REFERENCE_MASSES = np.array([1_000.0, 1_000.0, 1_000.0])
_CAPACITIES = np.array([3_000.0, 3_000.0, 3_000.0])
_COORDINATES = np.array(
    [[-10.0, 0.0, -1.0], [5.0, 8.0, -1.0], [5.0, -8.0, -1.0]]
)
_ORIGIN = datetime(2026, 8, 25, 0, 0, 0)


def _runtime() -> BallastRuntimeAssembly:
    properties = BallastMassProperties(
        total_mass_kg=100_000.0,
        center_of_mass_m=np.array([0.0, 0.0, -5.0]),
        inertia_about_reference_kg_m2=np.diag([1.0e7, 1.1e7, 1.2e7]),
    )
    return BallastRuntimeAssembly(
        base_matrices=PlatformMatrices(
            mass=np.diag([100_000.0, 100_000.0, 100_000.0, 1.0e7, 1.1e7, 1.2e7]),
            damping=np.diag([100.0, 100.0, 100.0, 1_000.0, 1_000.0, 1_000.0]),
            hydrostatic_stiffness=np.diag(
                [1_000.0, 1_000.0, 1_000.0, 100_000.0, 100_000.0, 100_000.0]
            ),
            mooring_stiffness=np.zeros((6, 6)),
            weight_stiffness=weight_stiffness_about_reference(
                total_mass_kg=properties.total_mass_kg,
                center_of_mass_m=properties.center_of_mass_m,
                gravity_m_s2=9.81,
            ),
        ),
        reference_mass_properties=properties,
        gravity_m_s2=9.81,
        provenance="physical_rolling_chain_test",
    )


def _execution_config() -> ExecutionRolloutConfig:
    return ExecutionRolloutConfig(
        block_duration_s=600.0,
        internal_step_s=30.0,
        water_density_kg_m3=1_000.0,
        max_pump_rate_m3_min=0.1,
        tank_capacity_kg=3_000.0,
        stop_error_kg=0.0,
        restart_error_kg=0.0,
        min_on_s=0.0,
        min_off_s=0.0,
        near_target_hold_s=0.0,
        ramp_up_m3_min_per_s=np.inf,
        ramp_down_m3_min_per_s=np.inf,
        pump_rate_schedule_m3_min=((0.0, 0.1), (10_000.0, 0.1)),
    )


def _setup(*, posture_limit: float = 10.0) -> PhysicalRollingChainSetup:
    return PhysicalRollingChainSetup(
        runtime_assembly=_runtime(),
        reference_tank_masses_kg=_REFERENCE_MASSES,
        tank_capacities_kg=_CAPACITIES,
        tank_coordinates_m=_COORDINATES,
        rotor_load_parameters=ForecastRotorLoadParameters(
            air_density_kg_m3=1.225,
            rotor_radius_m=2.0,
            wind_reference_height_m=100.0,
            rotor_plane_height_m=100.0,
            power_law_shear_exponent=0.0,
            frozen_equilibrium_heading_rad=np.pi,
            rotor_force_application_point_from_reference_m=[0.0, 0.0, 10.0],
            downwind_rotor_normal_platform=[1.0, 0.0, 0.0],
            nacelle_yaw_axis_point_from_reference_m=[0.0, 0.0, 10.0],
        ),
        execution_config=_execution_config(),
        posture_limits=PhysicalPostureLimits(
            max_abs_pitch_roll_rad=[posture_limit, posture_limit],
            max_abs_pitch_roll_rate_rad_s=[posture_limit, posture_limit],
            source="wide structural chain limit fixture",
        ),
        decision_policy=PhysicalCurrentCycleDecisionPolicy(
            posture_equivalence_margin=0.05
        ),
        minimum_lead_reliability=0.5,
        reliability_scope_basis="fixed fixture reliability scope",
        future_platform_kinematics_source="fixed fixture frozen zero kinematics",
    )


def _cycle(
    index: int,
    *,
    future_uv_ms=((0.0, -8.0), (0.0, -9.0), (0.0, -8.0)),
    current_uv_ms=(0.0, -7.0),
    forecast_admitted=True,
    current_nacelle_yaw_relative_platform_rad=0.0,
    future_wave_loads=None,
    future_other_loads=None,
) -> PhysicalRollingCycleInput:
    origin = _ORIGIN + timedelta(seconds=600.0 * index)
    origin_text = origin.isoformat()
    future = np.asarray(future_uv_ms, dtype=float)
    forecast = ForecastEvidence(
        source="physical_rolling_chain_fixture",
        model_version="fixture-v1",
        origin_time=origin_text,
        sample_period_s=600.0,
        uv_ms=future,
        lead_reliability=np.full(future.shape[0], 0.9),
        event_probs={"fixture": 0.0},
        provides_future_preview=True,
    )
    if future_wave_loads is None:
        future_wave_loads = np.zeros((future.shape[0], 6))
    if future_other_loads is None:
        future_other_loads = np.zeros((future.shape[0], 6))
    return PhysicalRollingCycleInput(
        start_time_s=600.0 * index,
        forecast=forecast,
        current_enu_downwind_air_velocity_mps=current_uv_ms,
        current_wind_source="same_origin_fixture_observation",
        current_wind_observation_time=origin_text,
        current_thrust_coefficient=0.8,
        current_nacelle_yaw_relative_platform_rad=(
            current_nacelle_yaw_relative_platform_rad
        ),
        current_nacelle_yaw_source="fixed fixture nacelle-yaw state",
        future_thrust_coefficients=np.full(future.shape[0], 0.8),
        thrust_coefficient_source="explicit fixture operating state",
        forecast_admitted=forecast_admitted,
        forecast_admission_basis="fixed fixture admission",
        environment=ForecastTrajectoryEnvironment(
            current_wave_load=np.zeros(6),
            current_other_load=np.zeros(6),
            wave_interval_loads=future_wave_loads,
            other_interval_loads=future_other_loads,
            forecast_origin_time=origin_text,
            lead_times_s=np.arange(1, future.shape[0] + 1, dtype=float) * 600.0,
            provenance="fixed fixture zero nonrotor loads",
        ),
    )


def _execution_state() -> ExecutionRolloutState:
    return ExecutionRolloutState(
        masses_kg=_REFERENCE_MASSES,
        target_masses_kg=_REFERENCE_MASSES,
        primary_target_kg=_REFERENCE_MASSES,
        pump_rates_m3_min=np.zeros(3),
        pump_latched=np.zeros(3, dtype=bool),
        pump_off_elapsed_s=np.full(3, 60.0),
    )


class PhysicalRollingControlChainTests(unittest.TestCase):
    def test_single_cycle_entry_hands_actual_state_to_a_fresh_next_cycle(self):
        setup = _setup()
        first = advance_physical_rolling_control_cycle(
            setup=setup,
            cycle=_cycle(0),
            platform_state=IncrementalState.zeros(),
            execution_state=_execution_state(),
        )

        self.assertEqual(first.status, "completed")
        self.assertIsNotNone(first.end_platform_state)
        self.assertIsNotNone(first.end_execution_state)
        second = advance_physical_rolling_control_cycle(
            setup=setup,
            cycle=_cycle(1),
            platform_state=first.end_platform_state,
            execution_state=first.end_execution_state,
        )

        self.assertEqual(second.status, "completed")
        np.testing.assert_allclose(
            second.start_platform_state.position,
            first.end_platform_state.position,
            rtol=0.0,
            atol=1.0e-12,
        )
        np.testing.assert_array_equal(
            second.start_execution_state.actual_masses_kg,
            first.end_execution_state.actual_masses_kg,
        )

    def test_single_cycle_entry_rejects_a_forecast_interval_that_mismatches_execution(self):
        cycle = _cycle(0)
        mismatched_cycle = replace(
            cycle,
            forecast=replace(cycle.forecast, sample_period_s=300.0),
        )

        with self.assertRaisesRegex(
            ValueError,
            "forecast sample period must equal the current execution block duration",
        ):
            advance_physical_rolling_control_cycle(
                setup=_setup(),
                cycle=mismatched_cycle,
                platform_state=IncrementalState.zeros(),
                execution_state=_execution_state(),
            )

    def test_each_cycle_commits_one_selected_request_and_hands_actual_state_forward(self):
        result = run_physical_rolling_control_chain(
            setup=_setup(),
            cycles=(_cycle(0), _cycle(1), _cycle(2)),
            initial_platform_state=IncrementalState.zeros(),
            initial_execution_state=_execution_state(),
        )

        self.assertTrue(result.completed)
        self.assertEqual(len(result.records), 3)
        for record in result.records:
            self.assertEqual(record.status, "completed")
            self.assertIsNotNone(record.actual_path)
            self.assertIsNotNone(record.end_platform_state)
            self.assertIsNotNone(record.end_execution_state)
            self.assertIs(
                record.decision.require_current_execution_request().operation,
                record.actual_path.substeps[0].execution_step.target_operation,
            )
        for earlier, later in zip(result.records, result.records[1:]):
            np.testing.assert_allclose(
                later.start_execution_state.actual_masses_kg,
                earlier.end_execution_state.actual_masses_kg,
                rtol=0.0,
                atol=1.0e-12,
            )
            for attribute in (
                "rate_limited_target_kg",
                "primary_target_masses_kg",
                "signed_flow_m3_min",
                "pump_latched",
                "pump_on_elapsed_s",
                "pump_off_elapsed_s",
                "pump_near_target_s",
                "pump_command_rates_m3_min",
                "last_flow_directions",
            ):
                np.testing.assert_array_equal(
                    getattr(later.start_execution_state, attribute),
                    getattr(earlier.end_execution_state, attribute),
                )
            np.testing.assert_allclose(
                later.start_platform_state.position,
                earlier.end_platform_state.position,
                rtol=0.0,
                atol=1.0e-12,
            )
            np.testing.assert_allclose(
                later.start_platform_state.velocity,
                earlier.end_platform_state.velocity,
                rtol=0.0,
                atol=1.0e-12,
            )

    def test_trace_keeps_the_complete_candidate_comparison_for_each_committed_request(self):
        result = run_physical_rolling_control_chain(
            setup=_setup(),
            cycles=(_cycle(0),),
            initial_platform_state=IncrementalState.zeros(),
            initial_execution_state=_execution_state(),
        )

        record = result.records[0]
        trace = record.as_trace()
        evaluations = trace["candidate_evaluations"]
        self.assertEqual(len(evaluations), len(record.decision.candidates))
        self.assertEqual(
            [item["route"] for item in evaluations],
            [candidate.route.value for candidate in record.decision.candidates],
        )
        self.assertEqual(
            [item["target_operation"] for item in evaluations],
            [
                candidate.execution_request.operation.value
                for candidate in record.decision.candidates
            ],
        )
        self.assertTrue(
            any(
                item["route"] == trace["selected_route"]
                and item["forecast_lead_index"]
                == trace["selected_forecast_lead_index"]
                for item in evaluations
            )
        )

    def test_actual_current_rotor_load_is_not_changed_by_future_forecast_values(self):
        calm_future = run_physical_rolling_control_chain(
            setup=_setup(),
            cycles=(_cycle(0, future_uv_ms=((0.0, -4.0),) * 3),),
            initial_platform_state=IncrementalState.zeros(),
            initial_execution_state=_execution_state(),
        )
        strong_future = run_physical_rolling_control_chain(
            setup=_setup(),
            cycles=(_cycle(0, future_uv_ms=((0.0, -16.0),) * 3),),
            initial_platform_state=IncrementalState.zeros(),
            initial_execution_state=_execution_state(),
        )

        np.testing.assert_allclose(
            calm_future.records[0].actual_rotor_load.generalized_load_platform,
            strong_future.records[0].actual_rotor_load.generalized_load_platform,
            rtol=0.0,
            atol=1.0e-12,
        )

    def test_schedule_rejects_reused_forecast_origin_for_a_later_state(self):
        with self.assertRaisesRegex(ValueError, "forecast origins must advance"):
            run_physical_rolling_control_chain(
                setup=_setup(),
                cycles=(_cycle(0), replace(_cycle(0), start_time_s=600.0)),
                initial_platform_state=IncrementalState.zeros(),
                initial_execution_state=_execution_state(),
            )

    def test_actual_current_load_preserves_explicit_nacelle_yaw(self):
        yaw_rad = np.pi / 4.0
        cycle = _cycle(0, current_nacelle_yaw_relative_platform_rad=yaw_rad)
        result = run_physical_rolling_control_chain(
            setup=_setup(),
            cycles=(cycle,),
            initial_platform_state=IncrementalState.zeros(),
            initial_execution_state=_execution_state(),
        )
        parameters = _setup().rotor_load_parameters
        point, normal = rotor_geometry_at_declared_nacelle_yaw(parameters, yaw_rad)
        expected = quasi_steady_rotor_generalized_load_from_enu_wind(
            air_density_kg_m3=parameters.air_density_kg_m3,
            rotor_radius_m=parameters.rotor_radius_m,
            thrust_coefficient=cycle.current_thrust_coefficient,
            enu_downwind_air_velocity_mps=np.asarray(cycle.current_enu_downwind_air_velocity_mps),
            frozen_equilibrium_heading_rad=parameters.frozen_equilibrium_heading_rad,
            platform_reference_velocity_platform_mps=np.zeros(3),
            platform_angular_velocity_platform_radps=np.zeros(3),
            rotor_force_application_point_from_reference_m=point,
            downwind_rotor_normal_platform=normal,
        )
        np.testing.assert_allclose(
            result.records[0].actual_rotor_load.generalized_load_platform,
            expected.generalized_load_platform,
            rtol=0.0,
            atol=1.0e-12,
        )

    def test_actual_path_uses_current_nonrotor_load_not_future_tail(self):
        cycle = _cycle(
            0,
            forecast_admitted=False,
            future_wave_loads=np.full((3, 6), 1.0e8),
            future_other_loads=np.full((3, 6), -1.0e8),
        )
        result = run_physical_rolling_control_chain(
            setup=_setup(),
            cycles=(cycle,),
            initial_platform_state=IncrementalState.zeros(),
            initial_execution_state=_execution_state(),
        )
        for substep in result.records[0].actual_path.substeps:
            np.testing.assert_allclose(substep.platform_step.loads.wave, np.zeros(6))
            np.testing.assert_allclose(substep.platform_step.loads.other, np.zeros(6))

    def test_release_request_becomes_a_real_hold_when_no_pump_is_running(self):
        initial_execution = ExecutionRolloutState(
            masses_kg=_REFERENCE_MASSES,
            target_masses_kg=[1_500.0, 800.0, 700.0],
            primary_target_kg=[1_500.0, 800.0, 700.0],
            pump_rates_m3_min=np.zeros(3),
            pump_latched=np.zeros(3, dtype=bool),
            pump_off_elapsed_s=np.full(3, 60.0),
        )
        result = run_physical_rolling_control_chain(
            setup=_setup(),
            cycles=(_cycle(0, forecast_admitted=False),),
            initial_platform_state=IncrementalState.zeros(),
            initial_execution_state=initial_execution,
        )

        record = result.records[0]
        self.assertEqual(record.decision.selected_route.value, "release_to_current")
        self.assertEqual(
            record.actual_path.substeps[0].execution_step.target_operation.value,
            "release_to_current",
        )
        self.assertEqual(record.actual_path.aggregate_pump_active_time_s, 0.0)
        np.testing.assert_allclose(
            record.end_execution_state.actual_masses_kg,
            initial_execution.actual_masses_kg,
            rtol=0.0,
            atol=1.0e-12,
        )

    def test_release_stops_an_active_pump_when_no_lifecycle_constraint_blocks_it(self):
        initial_execution = ExecutionRolloutState(
            masses_kg=_REFERENCE_MASSES,
            target_masses_kg=[1_500.0, 800.0, 700.0],
            primary_target_kg=[1_500.0, 800.0, 700.0],
            pump_rates_m3_min=[0.1, -0.1, 0.0],
            pump_latched=[True, True, False],
            pump_on_elapsed_s=[60.0, 60.0, 0.0],
            pump_off_elapsed_s=[0.0, 0.0, 60.0],
            pump_command_rates_m3_min=[0.1, 0.1, 0.0],
        )
        result = run_physical_rolling_control_chain(
            setup=_setup(),
            cycles=(_cycle(0, forecast_admitted=False),),
            initial_platform_state=IncrementalState.zeros(),
            initial_execution_state=initial_execution,
        )

        record = result.records[0]
        self.assertEqual(record.decision.selected_route.value, "release_to_current")
        self.assertEqual(record.actual_path.aggregate_pump_active_time_s, 0.0)
        self.assertEqual(record.actual_path.pump_stop_count, 2)
        selected = next(
            candidate
            for candidate in record.as_trace()["candidate_evaluations"]
            if candidate["route"] == record.decision.selected_route.value
            and candidate["forecast_lead_index"]
            == record.decision.selected_forecast_lead_index
        )
        self.assertEqual(selected["current_block_pump_stop_count"], 2)
        np.testing.assert_array_equal(
            record.end_execution_state.pump_latched,
            np.array([False, False, False]),
        )

    def test_abstention_does_not_apply_an_implicit_fallback_or_advance_state(self):
        initial_platform = IncrementalState.zeros()
        initial_execution = _execution_state()
        result = run_physical_rolling_control_chain(
            setup=_setup(posture_limit=1.0e-12),
            cycles=(_cycle(0), _cycle(1)),
            initial_platform_state=initial_platform,
            initial_execution_state=initial_execution,
        )

        self.assertEqual(result.status, "abstained")
        self.assertEqual(len(result.records), 1)
        self.assertEqual(result.records[0].status, "abstained")
        np.testing.assert_allclose(
            result.final_platform_state.position,
            initial_platform.position,
            rtol=0.0,
            atol=1.0e-12,
        )
        np.testing.assert_allclose(
            result.final_execution_state.actual_masses_kg,
            initial_execution.actual_masses_kg,
            rtol=0.0,
            atol=1.0e-12,
        )


if __name__ == "__main__":
    unittest.main()
