import unittest

import numpy as np

from fowt_platform import (
    BallastMassProperties,
    BallastRuntimeAssembly,
    IncrementalState,
    PlatformMatrices,
    RotorGeneralizedLoad,
    RotorNormalLoad,
    ThreeTankDifferentialModes,
    quasi_steady_rotor_generalized_load_from_enu_wind,
    weight_stiffness_about_reference,
)
from wind_prediction.execution_rollout import (
    ExecutionRolloutConfig,
    ExecutionRolloutRequest,
    ExecutionRolloutState,
)
from wind_prediction.preview_mpc import (
    PreviewBlockModel,
    PreviewMPCController,
    PreviewMPCConstraints,
    PreviewMPCWeights,
)
from wind_prediction.preview_mpc_runtime import (
    _evaluate_sampled_posture_path,
    assemble_preview_actuator_envelope,
    assemble_preview_mpc_fallback_requests,
    execute_preview_mpc_first_block,
)


REFERENCE_MASSES = np.full(3, 1_000.0)
CAPACITIES = np.full(3, 3_000.0)
COORDINATES = np.array(
    [[-10.0, 0.0, -1.0], [5.0, 8.0, -1.0], [5.0, -8.0, -1.0]]
)


def runtime() -> BallastRuntimeAssembly:
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
        provenance="preview_mpc_runtime_test",
    )


def successful_preview_case():
    assembly = runtime()
    modes = ThreeTankDifferentialModes(
        tank_coordinates_m=COORDINATES,
        gravity_m_s2=9.81,
    )
    model = PreviewBlockModel(
        matrices=assembly.base_matrices,
        generalized_load_per_mode_kg=modes.generalized_load_per_mode_kg,
        block_duration_s=60.0,
        sample_fractions=(0.25, 0.5, 0.75, 1.0),
    )
    controller = PreviewMPCController(
        block_model=model,
        ballast_modes=modes,
        weights=PreviewMPCWeights(
            roll=1.0,
            pitch=1.0,
            tank_movement=1.0e-9,
            movement_change=1.0e-10,
        ),
        constraints=PreviewMPCConstraints(
            maximum_abs_roll_rad=0.5,
            maximum_abs_pitch_rad=0.5,
            maximum_abs_tank_mass_change_per_block_kg=np.full(3, 500.0),
        ),
    )
    loads = np.zeros((3, 6))
    loads[:, 4] = np.array([2_000.0, 5_000.0, -2_000.0])
    plan = controller.solve(
        initial_platform_state=IncrementalState.zeros(),
        actual_tank_masses_kg=REFERENCE_MASSES,
        tank_capacities_kg=CAPACITIES,
        generalized_disturbance_loads=loads,
    )
    config = ExecutionRolloutConfig(
        block_duration_s=60.0,
        internal_step_s=10.0,
        water_density_kg_m3=1_000.0,
        max_pump_rate_m3_min=0.2,
        tank_capacity_kg=3_000.0,
        stop_error_kg=0.0,
        restart_error_kg=0.0,
        min_on_s=0.0,
        min_off_s=0.0,
        near_target_hold_s=0.0,
        ramp_up_m3_min_per_s=np.inf,
        ramp_down_m3_min_per_s=np.inf,
        pump_rate_schedule_m3_min=((0.0, 0.2), (10_000.0, 0.2)),
    )
    state = ExecutionRolloutState(
        masses_kg=REFERENCE_MASSES,
        target_masses_kg=REFERENCE_MASSES,
        primary_target_kg=REFERENCE_MASSES,
        pump_rates_m3_min=np.zeros(3),
        pump_latched=np.zeros(3, dtype=bool),
        pump_off_elapsed_s=np.full(3, 60.0),
    )
    zero_rotor = RotorGeneralizedLoad(
        relative_air_velocity_platform_mps=np.zeros(3),
        normal_load=RotorNormalLoad(
            thrust_n=0.0,
            force_platform_n=np.zeros(3),
        ),
        generalized_load_platform=np.zeros(6),
    )
    return assembly, modes, plan, config, state, zero_rotor


class PreviewMPCRuntimeTests(unittest.TestCase):
    def test_named_fallbacks_continue_primary_target_then_release(self):
        state = ExecutionRolloutState(
            masses_kg=np.array([100.0, 200.0, 300.0]),
            target_masses_kg=np.array([110.0, 210.0, 310.0]),
            primary_target_kg=np.array([120.0, 220.0, 320.0]),
            pump_rates_m3_min=np.zeros(3),
            pump_latched=np.zeros(3, dtype=bool),
        )

        fallbacks = assemble_preview_mpc_fallback_requests(state)

        self.assertEqual(
            [source for source, _ in fallbacks],
            [
                "fallback_continue_primary_target",
                "fallback_release_to_current",
            ],
        )
        np.testing.assert_allclose(
            fallbacks[0][1].target_masses_kg,
            state.primary_target_masses_kg,
        )
        self.assertEqual(
            fallbacks[1][1].operation.value,
            "release_to_current",
        )

    def test_realised_load_does_not_change_candidate_selection(self):
        assembly, _, plan, config, state, visible_load = successful_preview_case()
        realised_load = RotorGeneralizedLoad(
            relative_air_velocity_platform_mps=np.zeros(3),
            normal_load=RotorNormalLoad(
                thrust_n=0.0,
                force_platform_n=np.zeros(3),
            ),
            generalized_load_platform=np.array([0, 0, 0, 0, 2.0e5, 0]),
        )
        common = dict(
            plan=plan,
            platform_state=IncrementalState.zeros(),
            execution_state=state,
            execution_config=config,
            runtime_assembly=assembly,
            reference_tank_masses_kg=REFERENCE_MASSES,
            tank_capacities_kg=CAPACITIES,
            tank_coordinates_m=COORDINATES,
            current_rotor_load=visible_load,
            current_wave_load=np.zeros(6),
            current_other_load=np.zeros(6),
        )

        visible_execution = execute_preview_mpc_first_block(**common)
        realised_execution = execute_preview_mpc_first_block(
            **common,
            execution_rotor_load=realised_load,
        )

        self.assertEqual(
            realised_execution.selected_request_source,
            visible_execution.selected_request_source,
        )
        np.testing.assert_allclose(
            realised_execution.execution_request.target_masses_kg,
            visible_execution.execution_request.target_masses_kg,
        )
        np.testing.assert_allclose(
            realised_execution.selected_physical_precheck.execution_path.final_platform_state.position,
            visible_execution.execution_path.final_platform_state.position,
        )
        self.assertFalse(
            np.allclose(
                realised_execution.execution_path.final_platform_state.position,
                visible_execution.execution_path.final_platform_state.position,
            )
        )
        np.testing.assert_allclose(
            realised_execution.next_execution_state.actual_masses_kg,
            visible_execution.next_execution_state.actual_masses_kg,
        )

    def test_actuator_envelope_retains_current_pump_dwell_state(self):
        modes = ThreeTankDifferentialModes(
            tank_coordinates_m=COORDINATES,
            gravity_m_s2=9.81,
        )
        config = ExecutionRolloutConfig(
            block_duration_s=60.0,
            internal_step_s=5.0,
            water_density_kg_m3=1_000.0,
            max_pump_rate_m3_min=0.2,
            tank_capacity_kg=3_000.0,
            stop_error_kg=10.0,
            restart_error_kg=20.0,
            min_on_s=20.0,
            min_off_s=120.0,
            near_target_hold_s=10.0,
            ramp_up_m3_min_per_s=0.01,
            ramp_down_m3_min_per_s=0.02,
            pump_rate_schedule_m3_min=((0.0, 0.1), (100.0, 0.2)),
        )
        state = ExecutionRolloutState(
            masses_kg=REFERENCE_MASSES,
            target_masses_kg=REFERENCE_MASSES,
            primary_target_kg=np.array([1_200.0, 900.0, 900.0]),
            pump_rates_m3_min=np.zeros(3),
            pump_latched=np.zeros(3, dtype=bool),
            pump_off_elapsed_s=np.zeros(3),
        )

        envelope = assemble_preview_actuator_envelope(
            execution_state=state,
            execution_config=config,
            tank_capacities_kg=CAPACITIES,
            ballast_modes=modes,
        )

        np.testing.assert_allclose(
            envelope.first_block_lower_tank_delta_kg,
            0.0,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            envelope.first_block_upper_tank_delta_kg,
            0.0,
            atol=1e-12,
        )
        self.assertAlmostEqual(
            float(np.sum(envelope.outstanding_target_tank_delta_kg)),
            0.0,
            places=10,
        )

    def test_only_first_plan_target_is_committed_to_existing_executor(self):
        assembly = runtime()
        modes = ThreeTankDifferentialModes(
            tank_coordinates_m=COORDINATES,
            gravity_m_s2=9.81,
        )
        model = PreviewBlockModel(
            matrices=assembly.base_matrices,
            generalized_load_per_mode_kg=modes.generalized_load_per_mode_kg,
            block_duration_s=60.0,
            sample_fractions=(0.25, 0.5, 0.75, 1.0),
        )
        controller = PreviewMPCController(
            block_model=model,
            ballast_modes=modes,
            weights=PreviewMPCWeights(
                roll=1.0,
                pitch=1.0,
                tank_movement=1.0e-9,
                movement_change=1.0e-10,
            ),
            constraints=PreviewMPCConstraints(
                maximum_abs_roll_rad=0.5,
                maximum_abs_pitch_rad=0.5,
                maximum_abs_tank_mass_change_per_block_kg=np.full(3, 500.0),
            ),
        )
        loads = np.zeros((3, 6))
        loads[:, 4] = np.array([2_000.0, 5_000.0, -2_000.0])
        plan = controller.solve(
            initial_platform_state=IncrementalState.zeros(),
            actual_tank_masses_kg=REFERENCE_MASSES,
            tank_capacities_kg=CAPACITIES,
            generalized_disturbance_loads=loads,
        )
        self.assertTrue(plan.success, plan.message)

        execution_config = ExecutionRolloutConfig(
            block_duration_s=60.0,
            internal_step_s=10.0,
            water_density_kg_m3=1_000.0,
            max_pump_rate_m3_min=0.2,
            tank_capacity_kg=3_000.0,
            stop_error_kg=0.0,
            restart_error_kg=0.0,
            min_on_s=0.0,
            min_off_s=0.0,
            near_target_hold_s=0.0,
            ramp_up_m3_min_per_s=np.inf,
            ramp_down_m3_min_per_s=np.inf,
            pump_rate_schedule_m3_min=((0.0, 0.2), (10_000.0, 0.2)),
        )
        execution_state = ExecutionRolloutState(
            masses_kg=REFERENCE_MASSES,
            target_masses_kg=REFERENCE_MASSES,
            primary_target_kg=REFERENCE_MASSES,
            pump_rates_m3_min=np.zeros(3),
            pump_latched=np.zeros(3, dtype=bool),
            pump_off_elapsed_s=np.full(3, 60.0),
        )
        rotor_load = quasi_steady_rotor_generalized_load_from_enu_wind(
            air_density_kg_m3=1.225,
            rotor_radius_m=2.0,
            thrust_coefficient=0.8,
            enu_downwind_air_velocity_mps=[0.0, 0.0],
            frozen_equilibrium_heading_rad=0.0,
            platform_reference_velocity_platform_mps=np.zeros(3),
            platform_angular_velocity_platform_radps=np.zeros(3),
            rotor_force_application_point_from_reference_m=[0.0, 0.0, 10.0],
            downwind_rotor_normal_platform=[1.0, 0.0, 0.0],
        )
        executed = execute_preview_mpc_first_block(
            plan=plan,
            platform_state=IncrementalState.zeros(),
            execution_state=execution_state,
            execution_config=execution_config,
            runtime_assembly=assembly,
            reference_tank_masses_kg=REFERENCE_MASSES,
            tank_capacities_kg=CAPACITIES,
            tank_coordinates_m=COORDINATES,
            current_rotor_load=rotor_load,
            current_wave_load=np.zeros(6),
            current_other_load=np.zeros(6),
        )

        np.testing.assert_allclose(
            executed.execution_request.target_masses_kg,
            plan.planned_target_tank_masses_kg[0],
        )
        self.assertGreater(
            float(
                np.linalg.norm(
                    executed.next_execution_state.actual_masses_kg - REFERENCE_MASSES
                )
            ),
            0.0,
        )
        self.assertFalse(
            np.allclose(
                executed.next_execution_state.actual_masses_kg,
                plan.planned_target_tank_masses_kg[-1],
            )
        )

    def test_unsuccessful_plan_requires_and_records_explicit_fallback(self):
        assembly = runtime()
        modes = ThreeTankDifferentialModes(
            tank_coordinates_m=COORDINATES,
            gravity_m_s2=9.81,
        )
        model = PreviewBlockModel(
            matrices=assembly.base_matrices,
            generalized_load_per_mode_kg=modes.generalized_load_per_mode_kg,
            block_duration_s=60.0,
            sample_fractions=(0.5, 1.0),
        )
        controller = PreviewMPCController(
            block_model=model,
            ballast_modes=modes,
            weights=PreviewMPCWeights(
                roll=1.0,
                pitch=1.0,
                tank_movement=1e-9,
            ),
            constraints=PreviewMPCConstraints(
                maximum_abs_roll_rad=1e-5,
                maximum_abs_pitch_rad=1e-5,
                maximum_abs_tank_mass_change_per_block_kg=np.full(3, 1.0),
            ),
        )
        loads = np.zeros((2, 6))
        loads[:, 4] = 1e7
        plan = controller.solve(
            initial_platform_state=IncrementalState.zeros(),
            actual_tank_masses_kg=REFERENCE_MASSES,
            tank_capacities_kg=CAPACITIES,
            generalized_disturbance_loads=loads,
        )
        self.assertFalse(plan.success)

        config = ExecutionRolloutConfig(
            block_duration_s=60.0,
            internal_step_s=10.0,
            water_density_kg_m3=1_000.0,
            max_pump_rate_m3_min=0.2,
            tank_capacity_kg=3_000.0,
            stop_error_kg=0.0,
            restart_error_kg=0.0,
            min_on_s=0.0,
            min_off_s=0.0,
            near_target_hold_s=0.0,
            ramp_up_m3_min_per_s=np.inf,
            ramp_down_m3_min_per_s=np.inf,
            pump_rate_schedule_m3_min=((0.0, 0.2), (10_000.0, 0.2)),
        )
        state = ExecutionRolloutState(
            masses_kg=REFERENCE_MASSES,
            target_masses_kg=REFERENCE_MASSES,
            primary_target_kg=REFERENCE_MASSES,
            pump_rates_m3_min=np.zeros(3),
            pump_latched=np.zeros(3, dtype=bool),
            pump_off_elapsed_s=np.full(3, 60.0),
        )
        rotor_load = quasi_steady_rotor_generalized_load_from_enu_wind(
            air_density_kg_m3=1.225,
            rotor_radius_m=2.0,
            thrust_coefficient=0.8,
            enu_downwind_air_velocity_mps=[0.0, 0.0],
            frozen_equilibrium_heading_rad=0.0,
            platform_reference_velocity_platform_mps=np.zeros(3),
            platform_angular_velocity_platform_radps=np.zeros(3),
            rotor_force_application_point_from_reference_m=[0.0, 0.0, 10.0],
            downwind_rotor_normal_platform=[1.0, 0.0, 0.0],
        )
        executed = execute_preview_mpc_first_block(
            plan=plan,
            platform_state=IncrementalState.zeros(),
            execution_state=state,
            execution_config=config,
            runtime_assembly=assembly,
            reference_tank_masses_kg=REFERENCE_MASSES,
            tank_capacities_kg=CAPACITIES,
            tank_coordinates_m=COORDINATES,
            current_rotor_load=rotor_load,
            current_wave_load=np.zeros(6),
            current_other_load=np.zeros(6),
            fallback_request=ExecutionRolloutRequest.track(
                state.primary_target_masses_kg
            ),
        )

        self.assertTrue(executed.used_fallback)
        self.assertIn("preview_mpc_unsuccessful", executed.fallback_reason)
        np.testing.assert_allclose(
            executed.execution_request.target_masses_kg,
            state.primary_target_masses_kg,
        )

    def test_unsafe_mpc_request_selects_the_only_safe_fallback(self):
        assembly, _, plan, config, state, rotor = successful_preview_case()
        executed = execute_preview_mpc_first_block(
            plan=plan,
            platform_state=IncrementalState.zeros(),
            execution_state=state,
            execution_config=config,
            runtime_assembly=assembly,
            reference_tank_masses_kg=REFERENCE_MASSES,
            tank_capacities_kg=CAPACITIES,
            tank_coordinates_m=COORDINATES,
            current_rotor_load=rotor,
            current_wave_load=np.zeros(6),
            current_other_load=np.zeros(6),
            fallback_requests=((
                "release_to_current",
                ExecutionRolloutRequest.release_to_current(),
            ),),
            maximum_abs_roll_pitch_rad=np.full(2, 1.0e-4),
        )

        self.assertTrue(executed.used_fallback)
        self.assertEqual(executed.selected_request_source, "release_to_current")
        self.assertTrue(executed.sampled_posture_limit_satisfied)
        self.assertFalse(executed.physical_prechecks[0].posture_precheck_passed)
        self.assertTrue(executed.physical_prechecks[1].posture_precheck_passed)

    def test_all_unsafe_fallbacks_are_reported_without_claiming_safety(self):
        assembly, _, plan, config, state, _ = successful_preview_case()
        strong_pitch_load = RotorGeneralizedLoad(
            relative_air_velocity_platform_mps=np.zeros(3),
            normal_load=RotorNormalLoad(
                thrust_n=0.0,
                force_platform_n=np.zeros(3),
            ),
            generalized_load_platform=np.array([0, 0, 0, 0, 1.0e6, 0]),
        )
        executed = execute_preview_mpc_first_block(
            plan=plan,
            platform_state=IncrementalState.zeros(),
            execution_state=state,
            execution_config=config,
            runtime_assembly=assembly,
            reference_tank_masses_kg=REFERENCE_MASSES,
            tank_capacities_kg=CAPACITIES,
            tank_coordinates_m=COORDINATES,
            current_rotor_load=strong_pitch_load,
            current_wave_load=np.zeros(6),
            current_other_load=np.zeros(6),
            fallback_requests=(
                (
                    "continue_primary",
                    ExecutionRolloutRequest.track(state.primary_target_masses_kg),
                ),
                ("release_to_current", ExecutionRolloutRequest.release_to_current()),
            ),
            maximum_abs_roll_pitch_rad=np.full(2, 1.0e-5),
        )

        self.assertTrue(executed.used_fallback)
        self.assertFalse(executed.sampled_posture_limit_satisfied)
        self.assertIn("no fallback passed", executed.fallback_reason)

    def test_initially_outside_limit_accepts_a_nonworsening_recovery_path(self):
        assembly, _, plan, config, state, rotor = successful_preview_case()
        initial = IncrementalState(
            position=np.array([0.0, 0.0, 0.0, 0.0, 0.1, 0.0]),
            velocity=np.zeros(6),
        )
        executed = execute_preview_mpc_first_block(
            plan=plan,
            platform_state=initial,
            execution_state=state,
            execution_config=config,
            runtime_assembly=assembly,
            reference_tank_masses_kg=REFERENCE_MASSES,
            tank_capacities_kg=CAPACITIES,
            tank_coordinates_m=COORDINATES,
            current_rotor_load=rotor,
            current_wave_load=np.zeros(6),
            current_other_load=np.zeros(6),
            fallback_requests=((
                "release_to_current",
                ExecutionRolloutRequest.release_to_current(),
            ),),
            maximum_abs_roll_pitch_rad=np.full(2, 0.05),
        )

        selected = next(
            item
            for item in executed.physical_prechecks
            if item.source == executed.selected_request_source
        )
        self.assertFalse(selected.within_sampled_posture_limit)
        self.assertTrue(selected.posture_precheck_passed)
        self.assertLess(
            selected.terminal_normalized_posture,
            selected.initial_normalized_posture,
        )
        self.assertTrue(executed.selected_posture_precheck_passed)
        self.assertIn("axiswise", executed.selected_posture_precheck_reason)

    def test_recovery_cannot_transfer_an_exceedance_between_posture_axes(self):
        posture = np.array(
            [
                [0.060, 0.000],
                [0.055, 0.055],
                [0.050, 0.057],
            ]
        )

        passed, reason, within, initial, maximum, terminal = (
            _evaluate_sampled_posture_path(posture, np.full(2, 0.05))
        )

        self.assertFalse(passed)
        self.assertFalse(within)
        self.assertIn("without_axiswise", reason)
        self.assertAlmostEqual(initial, 1.2)
        self.assertAlmostEqual(maximum, 1.2)
        self.assertAlmostEqual(terminal, 1.14)


if __name__ == "__main__":
    unittest.main()
