import unittest

import numpy as np

from fowt_platform import IncrementalState, PlatformMatrices, ThreeTankDifferentialModes
from wind_prediction.preview_mpc import (
    MAX_SCALED_RECONSTRUCTION_VIOLATION,
    PreviewActuatorEnvelope,
    PreviewBlockModel,
    PreviewMPCController,
    PreviewMPCConstraints,
    PreviewMPCWeights,
)


TANK_COORDINATES_M = np.array(
    [
        [2.0, 0.0, -1.0],
        [-1.0, 2.0, -1.0],
        [-1.0, -2.0, -1.0],
    ]
)


class PreviewMPCControllerTests(unittest.TestCase):
    def setUp(self):
        matrices = PlatformMatrices(
            mass=np.diag([10.0, 12.0, 14.0, 20.0, 22.0, 24.0]),
            damping=np.diag([1.0, 1.0, 2.0, 3.0, 3.0, 1.0]),
            hydrostatic_stiffness=np.diag([0.0, 0.0, 30.0, 40.0, 50.0, 0.0]),
            mooring_stiffness=np.diag([4.0, 5.0, 0.0, 0.0, 0.0, 6.0]),
        )
        self.modes = ThreeTankDifferentialModes(
            tank_coordinates_m=TANK_COORDINATES_M,
            gravity_m_s2=10.0,
        )
        self.block_model = PreviewBlockModel(
            matrices=matrices,
            generalized_load_per_mode_kg=self.modes.generalized_load_per_mode_kg,
            block_duration_s=10.0,
            sample_fractions=(0.25, 0.5, 0.75, 1.0),
        )
        self.weights = PreviewMPCWeights(
            roll=1.0,
            pitch=1.0,
            tank_movement=1.0e-5,
            movement_change=1.0e-6,
        )
        self.controller = PreviewMPCController(
            block_model=self.block_model,
            ballast_modes=self.modes,
            weights=self.weights,
            constraints=PreviewMPCConstraints(
                maximum_abs_roll_rad=0.5,
                maximum_abs_pitch_rad=0.5,
                maximum_abs_tank_mass_change_per_block_kg=np.full(3, 100.0),
            ),
        )
        self.actual = np.full(3, 500.0)
        self.capacity = np.full(3, 1_000.0)

    def test_zero_disturbance_selects_hold(self):
        result = self.controller.solve(
            initial_platform_state=IncrementalState.zeros(),
            actual_tank_masses_kg=self.actual,
            tank_capacities_kg=self.capacity,
            generalized_disturbance_loads=np.zeros((3, 6)),
        )

        self.assertTrue(result.success, result.message)
        np.testing.assert_allclose(result.modal_increments_kg, 0.0, atol=1e-8)
        np.testing.assert_allclose(result.current_target_tank_masses_kg, self.actual)
        self.assertEqual(result.solver_name, "OSQP")
        self.assertLess(result.solver_primal_residual, 1.0e-7)
        self.assertLess(result.solver_dual_residual, 1.0e-7)
        self.assertGreater(result.solver_iterations, 0)
        self.assertGreaterEqual(
            float(np.min(np.linalg.eigvalsh(result.qp_problem.hessian))),
            -1.0e-10,
        )
        self.assertEqual(result.qp_problem.hessian.shape, (8, 8))
        self.assertEqual(result.qp_problem.constraint_matrix.shape[1], 8)
        np.testing.assert_allclose(result.posture_slack_rad, 0.0, atol=1e-10)

    def test_first_modal_increment_can_be_fixed_for_lifecycle_comparison(self):
        fixed = np.array([0.1, -0.1])
        result = self.controller.solve(
            initial_platform_state=IncrementalState.zeros(),
            actual_tank_masses_kg=self.actual,
            tank_capacities_kg=self.capacity,
            generalized_disturbance_loads=np.zeros((3, 6)),
            fixed_first_modal_increment_kg=fixed,
        )

        self.assertTrue(result.success, result.message)
        np.testing.assert_allclose(
            result.fixed_first_modal_increment_kg,
            fixed,
            rtol=0.0,
            atol=1.0e-7,
        )
        np.testing.assert_allclose(
            result.modal_increments_kg[0],
            fixed,
            rtol=0.0,
            atol=1.0e-7,
        )
        np.testing.assert_allclose(
            result.current_target_tank_masses_kg,
            self.actual + self.modes.tank_mass_deltas_from_modes(fixed),
            rtol=0.0,
            atol=1.0e-7,
        )

    def test_previous_physical_movement_is_used_by_tail_change_cost(self):
        previous_movement = self.modes.tank_mass_deltas_from_modes(
            np.array([10.0, 0.0])
        )
        without_prefix = self.controller.solve(
            initial_platform_state=IncrementalState.zeros(),
            actual_tank_masses_kg=self.actual,
            tank_capacities_kg=self.capacity,
            generalized_disturbance_loads=np.zeros((1, 6)),
        )
        after_prefix = self.controller.solve(
            initial_platform_state=IncrementalState.zeros(),
            actual_tank_masses_kg=self.actual,
            tank_capacities_kg=self.capacity,
            generalized_disturbance_loads=np.zeros((1, 6)),
            previous_tank_movement_kg=previous_movement,
        )

        self.assertTrue(without_prefix.success, without_prefix.message)
        self.assertTrue(after_prefix.success, after_prefix.message)
        np.testing.assert_allclose(
            without_prefix.modal_increments_kg,
            0.0,
            atol=1.0e-8,
        )
        after_movement = self.modes.tank_mass_deltas_from_modes(
            after_prefix.modal_increments_kg[0]
        )
        self.assertGreater(float(after_movement @ previous_movement), 0.0)
        self.assertGreater(float(np.linalg.norm(after_movement)), 1.0e-6)

    def test_actuator_reachability_and_previous_movement_are_independent_inputs(self):
        previous_movement = self.modes.tank_mass_deltas_from_modes(
            np.array([10.0, 0.0])
        )
        envelope = PreviewActuatorEnvelope(
            first_block_lower_tank_delta_kg=np.full(3, -100.0),
            first_block_upper_tank_delta_kg=np.full(3, 100.0),
            outstanding_target_tank_delta_kg=-previous_movement,
            source="unit_test_reachability_not_action_history",
        )

        result = self.controller.solve(
            initial_platform_state=IncrementalState.zeros(),
            actual_tank_masses_kg=self.actual,
            tank_capacities_kg=self.capacity,
            generalized_disturbance_loads=np.zeros((1, 6)),
            actuator_envelope=envelope,
            previous_tank_movement_kg=previous_movement,
        )

        self.assertTrue(result.success, result.message)
        first_movement = self.modes.tank_mass_deltas_from_modes(
            result.modal_increments_kg[0]
        )
        self.assertGreater(float(first_movement @ previous_movement), 0.0)

    def test_tail_running_posture_uses_original_horizon_duration(self):
        normalization_duration_s = 30.0
        initial = IncrementalState(
            position=np.array([0.0, 0.0, 0.0, 0.1, -0.2, 0.0]),
            velocity=np.zeros(6),
        )
        result = self.controller.solve(
            initial_platform_state=initial,
            actual_tank_masses_kg=self.actual,
            tank_capacities_kg=self.capacity,
            generalized_disturbance_loads=np.zeros((2, 6)),
            running_posture_normalization_duration_s=normalization_duration_s,
        )

        self.assertTrue(result.success, result.message)
        times = [0.0]
        posture = [initial.position[[3, 4]]]
        offset = 0.0
        for block in result.predicted_trajectory.blocks:
            times.extend(offset + block.elapsed_time_s)
            posture.extend(np.column_stack((block.roll_rad, block.pitch_rad)))
            offset += float(block.elapsed_time_s[-1])
        posture_array = np.asarray(posture)
        density = 0.5 * (
            self.weights.roll * posture_array[:, 0] ** 2
            + self.weights.pitch * posture_array[:, 1] ** 2
        )
        expected = np.trapezoid(density, np.asarray(times)) / normalization_duration_s
        self.assertAlmostEqual(
            result.objective_breakdown.running_posture,
            expected,
            places=10,
        )

    def test_tail_slack_inherits_physical_prefix_exceedance(self):
        controller = PreviewMPCController(
            block_model=self.block_model,
            ballast_modes=self.modes,
            weights=PreviewMPCWeights(
                roll=1.0,
                pitch=1.0,
                tank_movement=1.0e-5,
                posture_slack=100.0,
            ),
            constraints=PreviewMPCConstraints(
                maximum_abs_roll_rad=0.5,
                maximum_abs_pitch_rad=0.5,
                maximum_abs_tank_mass_change_per_block_kg=np.full(3, 100.0),
                maximum_posture_slack_rad=np.array([0.2, 0.2]),
            ),
        )
        inherited = np.array([0.08, 0.03])
        result = controller.solve(
            initial_platform_state=IncrementalState.zeros(),
            actual_tank_masses_kg=self.actual,
            tank_capacities_kg=self.capacity,
            generalized_disturbance_loads=np.zeros((2, 6)),
            minimum_posture_slack_rad=inherited,
        )

        self.assertTrue(result.success, result.message)
        np.testing.assert_allclose(
            result.posture_slack_rad,
            inherited,
            rtol=0.0,
            atol=1.0e-8,
        )

    def test_infeasible_fixed_first_increment_returns_a_failed_result(self):
        fixed = np.array([1.0e9, -1.0e9])

        result = self.controller.solve(
            initial_platform_state=IncrementalState.zeros(),
            actual_tank_masses_kg=self.actual,
            tank_capacities_kg=self.capacity,
            generalized_disturbance_loads=np.zeros((3, 6)),
            fixed_first_modal_increment_kg=fixed,
        )

        self.assertFalse(result.success)
        np.testing.assert_allclose(result.fixed_first_modal_increment_kg, fixed)

    def test_objective_breakdown_reconstructs_the_qp_objective(self):
        loads = np.zeros((3, 6))
        loads[1:, 4] = 10.0

        result = self.controller.solve(
            initial_platform_state=IncrementalState.zeros(),
            actual_tank_masses_kg=self.actual,
            tank_capacities_kg=self.capacity,
            generalized_disturbance_loads=loads,
        )

        self.assertTrue(result.success, result.message)
        breakdown = result.objective_breakdown
        self.assertEqual(breakdown.running_posture_by_block.shape, (3,))
        self.assertEqual(breakdown.tank_movement_by_block.shape, (3,))
        self.assertEqual(breakdown.movement_change_by_block.shape, (3,))
        for value in (
            breakdown.running_posture,
            breakdown.terminal_posture,
            breakdown.tank_movement,
            breakdown.tank_throughput,
            breakdown.movement_change,
            breakdown.posture_slack,
            breakdown.numerical_regularization,
            breakdown.absolute_total,
        ):
            self.assertGreaterEqual(value, 0.0)
        self.assertAlmostEqual(
            breakdown.absolute_total - breakdown.constant_offset,
            result.objective_value,
            places=9,
        )

    def test_l1_tank_throughput_epigraph_matches_physical_tank_movements(self):
        weights = PreviewMPCWeights.from_normalized_scales(
            roll_scale_rad=0.5,
            pitch_scale_rad=0.5,
            tank_movement_scale_kg=100.0,
            posture_slack_scale_rad=0.5,
            running_posture_priority=1.0,
            terminal_posture_priority=1.0,
            tank_movement_priority=1.0e-3,
            tank_throughput_priority=1.0e-2,
            movement_change_priority=1.0e-3,
            posture_slack_priority=1.0,
        )
        controller = PreviewMPCController(
            block_model=self.block_model,
            ballast_modes=self.modes,
            weights=weights,
            constraints=PreviewMPCConstraints(
                maximum_abs_roll_rad=10.0,
                maximum_abs_pitch_rad=10.0,
                maximum_abs_tank_mass_change_per_block_kg=np.full(3, 100.0),
            ),
        )
        loads = np.zeros((3, 6))
        loads[1:, 4] = 10.0

        result = controller.solve(
            initial_platform_state=IncrementalState.zeros(),
            actual_tank_masses_kg=self.actual,
            tank_capacities_kg=self.capacity,
            generalized_disturbance_loads=loads,
        )

        self.assertTrue(result.success, result.message)
        tank_movements = np.vstack(
            [
                self.modes.tank_mass_deltas_from_modes(value)
                for value in result.modal_increments_kg
            ]
        )
        np.testing.assert_allclose(
            result.objective_breakdown.tank_throughput_by_block,
            weights.tank_throughput * np.sum(np.abs(tank_movements), axis=1),
            rtol=0.0,
            atol=1.0e-8,
        )
        self.assertLess(
            result.maximum_tank_throughput_epigraph_violation_kg,
            1.0e-7,
        )
        self.assertLessEqual(
            result.maximum_scaled_constraint_violation,
            MAX_SCALED_RECONSTRUCTION_VIOLATION,
        )
        self.assertEqual(result.qp_problem.hessian.shape, (17, 17))
        self.assertAlmostEqual(
            result.objective_breakdown.reduced_total,
            result.objective_value,
            places=9,
        )

    def test_l1_throughput_does_not_reward_splitting_one_total_movement(self):
        weights = PreviewMPCWeights.from_normalized_scales(
            roll_scale_rad=0.5,
            pitch_scale_rad=0.5,
            tank_movement_scale_kg=100.0,
            posture_slack_scale_rad=0.5,
            tank_throughput_priority=1.0e-2,
        )
        single = np.array([[60.0, -40.0, -20.0]])
        split = np.repeat(single / 6.0, 6, axis=0)

        single_l1 = weights.tank_throughput * np.sum(np.abs(single))
        split_l1 = weights.tank_throughput * np.sum(np.abs(split))
        single_l2 = weights.tank_movement * np.sum(single**2)
        split_l2 = weights.tank_movement * np.sum(split**2)

        self.assertAlmostEqual(single_l1, split_l1)
        self.assertLess(split_l2, single_l2 / 5.9)

    def test_normalized_weight_design_preserves_physical_scales(self):
        weights = PreviewMPCWeights.from_normalized_scales(
            roll_scale_rad=0.2,
            pitch_scale_rad=0.25,
            tank_movement_scale_kg=100.0,
            posture_slack_scale_rad=0.05,
            running_posture_priority=1.0,
            terminal_posture_priority=2.0,
            tank_movement_priority=0.01,
            movement_change_priority=0.001,
            posture_slack_priority=100.0,
        )

        self.assertAlmostEqual(weights.roll, 25.0)
        self.assertAlmostEqual(weights.pitch, 16.0)
        self.assertAlmostEqual(weights.tank_movement, 1.0e-6)
        self.assertAlmostEqual(weights.movement_change, 1.0e-7)
        self.assertAlmostEqual(weights.posture_slack, 40_000.0)
        self.assertAlmostEqual(weights.terminal_roll, 50.0)
        self.assertAlmostEqual(weights.terminal_pitch, 32.0)

    def test_terminal_posture_cost_reduces_end_of_horizon_pitch(self):
        loads = np.zeros((3, 6))
        loads[-1, 4] = 50.0
        constraints = PreviewMPCConstraints(
            maximum_abs_roll_rad=10.0,
            maximum_abs_pitch_rad=10.0,
            maximum_abs_tank_mass_change_per_block_kg=np.full(3, 100.0),
        )

        def solve(terminal_priority: float):
            weights = PreviewMPCWeights.from_normalized_scales(
                roll_scale_rad=0.5,
                pitch_scale_rad=0.5,
                tank_movement_scale_kg=100.0,
                posture_slack_scale_rad=0.5,
                running_posture_priority=0.1,
                terminal_posture_priority=terminal_priority,
                tank_movement_priority=0.1,
                movement_change_priority=0.01,
                posture_slack_priority=1.0,
            )
            return PreviewMPCController(
                block_model=self.block_model,
                ballast_modes=self.modes,
                weights=weights,
                constraints=constraints,
            ).solve(
                initial_platform_state=IncrementalState.zeros(),
                actual_tank_masses_kg=self.actual,
                tank_capacities_kg=self.capacity,
                generalized_disturbance_loads=loads,
            )

        without_terminal = solve(0.0)
        with_terminal = solve(10.0)

        self.assertTrue(without_terminal.success, without_terminal.message)
        self.assertTrue(with_terminal.success, with_terminal.message)
        self.assertLess(
            abs(with_terminal.predicted_trajectory.blocks[-1].pitch_rad[-1]),
            abs(without_terminal.predicted_trajectory.blocks[-1].pitch_rad[-1]),
        )

    def test_future_pitch_load_changes_current_target(self):
        loads = np.zeros((3, 6))
        loads[1:, 4] = 10.0

        result = self.controller.solve(
            initial_platform_state=IncrementalState.zeros(),
            actual_tank_masses_kg=self.actual,
            tank_capacities_kg=self.capacity,
            generalized_disturbance_loads=loads,
        )

        self.assertTrue(result.success, result.message)
        self.assertGreater(float(np.linalg.norm(result.modal_increments_kg[0])), 1e-5)
        self.assertFalse(
            np.allclose(result.current_target_tank_masses_kg, self.actual, atol=1e-5)
        )

    def test_only_first_target_is_exposed_for_current_execution(self):
        loads = np.zeros((3, 6))
        loads[:, 3] = np.array([2.0, -5.0, 3.0])
        result = self.controller.solve(
            initial_platform_state=IncrementalState.zeros(),
            actual_tank_masses_kg=self.actual,
            tank_capacities_kg=self.capacity,
            generalized_disturbance_loads=loads,
        )

        self.assertTrue(result.success, result.message)
        np.testing.assert_allclose(
            result.current_target_tank_masses_kg,
            result.planned_target_tank_masses_kg[0],
        )

    def test_tank_targets_respect_capacity_and_per_block_movement(self):
        loads = np.zeros((4, 6))
        loads[:, 4] = 200.0
        wide_posture_controller = PreviewMPCController(
            block_model=self.block_model,
            ballast_modes=self.modes,
            weights=self.weights,
            constraints=PreviewMPCConstraints(
                maximum_abs_roll_rad=100.0,
                maximum_abs_pitch_rad=100.0,
                maximum_abs_tank_mass_change_per_block_kg=np.full(3, 100.0),
            ),
        )
        result = wide_posture_controller.solve(
            initial_platform_state=IncrementalState.zeros(),
            actual_tank_masses_kg=np.array([990.0, 500.0, 500.0]),
            tank_capacities_kg=self.capacity,
            generalized_disturbance_loads=loads,
        )

        self.assertTrue(result.success, result.message)
        self.assertTrue(np.all(result.planned_target_tank_masses_kg >= -1e-7))
        self.assertTrue(np.all(result.planned_target_tank_masses_kg <= 1_000.0 + 1e-7))
        tank_movements = np.vstack(
            [self.modes.tank_mass_deltas_from_modes(row) for row in result.modal_increments_kg]
        )
        self.assertTrue(np.all(np.abs(tank_movements) <= 100.0 + 1e-6))
        self.assertLessEqual(
            result.maximum_block_tank_movement_violation_kg,
            1.0e-7,
        )

    def test_initial_posture_outside_limit_returns_an_explicit_plan_status(self):
        initial = IncrementalState(
            position=np.array([0.0, 0.0, 0.0, 0.6, 0.0, 0.0]),
            velocity=np.zeros(6),
        )

        result = self.controller.solve(
            initial_platform_state=initial,
            actual_tank_masses_kg=self.actual,
            tank_capacities_kg=self.capacity,
            generalized_disturbance_loads=np.zeros((3, 6)),
        )

        self.assertIsInstance(result.success, bool)
        self.assertTrue(result.message)

    def test_first_control_move_respects_physical_actuator_envelope(self):
        loads = np.zeros((3, 6))
        loads[:, 4] = 1.0
        envelope = PreviewActuatorEnvelope(
            first_block_lower_tank_delta_kg=np.array([-4.0, -8.0, -7.0]),
            first_block_upper_tank_delta_kg=np.array([5.0, 6.0, 9.0]),
            outstanding_target_tank_delta_kg=np.zeros(3),
            source="unit_test_physical_executor_envelope",
        )

        result = self.controller.solve(
            initial_platform_state=IncrementalState.zeros(),
            actual_tank_masses_kg=self.actual,
            tank_capacities_kg=self.capacity,
            generalized_disturbance_loads=loads,
            actuator_envelope=envelope,
        )

        self.assertTrue(result.success, result.message)
        first_delta = (
            result.current_target_tank_masses_kg - self.actual
        )
        self.assertTrue(
            np.all(first_delta >= envelope.first_block_lower_tank_delta_kg - 1e-7)
        )
        self.assertTrue(
            np.all(first_delta <= envelope.first_block_upper_tank_delta_kg + 1e-7)
        )
        self.assertLessEqual(
            result.maximum_first_block_actuator_violation_kg,
            1e-7,
        )

    def test_bounded_posture_slack_recovers_a_nominally_infeasible_plan(self):
        controller = PreviewMPCController(
            block_model=self.block_model,
            ballast_modes=self.modes,
            weights=PreviewMPCWeights(
                roll=1.0,
                pitch=1.0,
                tank_movement=1.0e-5,
                movement_change=1.0e-6,
                posture_slack=100.0,
            ),
            constraints=PreviewMPCConstraints(
                maximum_abs_roll_rad=0.01,
                maximum_abs_pitch_rad=0.01,
                maximum_abs_tank_mass_change_per_block_kg=np.full(3, 1.0),
                maximum_posture_slack_rad=np.array([0.5, 0.5]),
            ),
        )
        loads = np.zeros((3, 6))
        loads[:, 4] = 10.0

        result = controller.solve(
            initial_platform_state=IncrementalState.zeros(),
            actual_tank_masses_kg=self.actual,
            tank_capacities_kg=self.capacity,
            generalized_disturbance_loads=loads,
        )

        self.assertTrue(result.success, result.message)
        self.assertGreater(float(result.posture_slack_rad[1]), 0.0)
        self.assertGreater(result.maximum_nominal_posture_exceedance_rad, 0.0)
        self.assertLessEqual(result.maximum_posture_constraint_violation_rad, 1e-7)


if __name__ == "__main__":
    unittest.main()
