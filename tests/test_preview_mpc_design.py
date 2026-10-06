from dataclasses import replace
import unittest

import numpy as np

from fowt_platform import IncrementalState, PlatformMatrices, ThreeTankDifferentialModes
from wind_prediction.preview_mpc import (
    PreviewBlockModel,
    PreviewMPCController,
    PreviewState,
)
from wind_prediction.preview_mpc_design import (
    PreviewMPCDesign,
    research_preview_mpc_design_v1,
    research_preview_mpc_design_v2,
    research_preview_mpc_design_v3,
    research_preview_mpc_design_v4,
)


TANK_COORDINATES_M = np.array(
    [
        [2.0, 0.0, -1.0],
        [-1.0, 2.0, -1.0],
        [-1.0, -2.0, -1.0],
    ]
)


class PreviewMPCDesignTests(unittest.TestCase):
    def setUp(self):
        self.design = research_preview_mpc_design_v1()
        matrices = PlatformMatrices(
            mass=np.diag([10.0, 12.0, 14.0, 20.0, 22.0, 24.0]),
            damping=np.diag([1.0, 1.0, 2.0, 3.0, 3.0, 1.0]),
            hydrostatic_stiffness=np.diag(
                [0.0, 0.0, 30.0, 40.0, 50.0, 0.0]
            ),
            mooring_stiffness=np.diag([4.0, 5.0, 0.0, 0.0, 0.0, 6.0]),
        )
        self.modes = ThreeTankDifferentialModes(
            tank_coordinates_m=TANK_COORDINATES_M,
            gravity_m_s2=10.0,
        )
        self.block_model = PreviewBlockModel(
            matrices=matrices,
            generalized_load_per_mode_kg=(
                self.modes.generalized_load_per_mode_kg
            ),
            block_duration_s=10.0,
            sample_fractions=(0.5, 1.0),
        )

    def test_named_working_point_records_status_and_priorities(self):
        self.assertEqual(self.design.identity, "research_preview_mpc_design_v1")
        self.assertEqual(
            self.design.status,
            "stage_working_point_not_calibrated_or_optimal",
        )
        self.assertEqual(
            self.design.normalized_objective_priorities,
            {
                "running_posture": 1.0,
                "terminal_posture": 1.0,
                "tank_movement": 1.0e-2,
                "tank_throughput": 5.0e-4,
                "movement_change": 1.0e-3,
                "posture_slack": 1.0e2,
            },
        )
        np.testing.assert_allclose(
            np.rad2deg(self.design.absolute_maximum_abs_roll_pitch_rad),
            [15.0, 15.0],
        )
        np.testing.assert_allclose(
            np.rad2deg(
                self.design.maximum_abs_working_model_scope_roll_pitch_rad
            ),
            [5.0, 5.0],
        )

    def test_priority_multiplier_returns_isolated_diagnostic_variant(self):
        variant = self.design.with_priority_multipliers(
            {"tank_throughput": 2.0}
        )

        self.assertIsInstance(variant, PreviewMPCDesign)
        self.assertEqual(self.design.tank_throughput_priority, 5.0e-4)
        self.assertEqual(variant.tank_throughput_priority, 1.0e-3)
        self.assertEqual(
            variant.tank_movement_priority,
            self.design.tank_movement_priority,
        )

        with self.assertRaises(ValueError):
            self.design.with_priority_multipliers({"unknown": 2.0})
        with self.assertRaises(ValueError):
            self.design.with_priority_multipliers({"tank_movement": 0.0})

    def test_assembly_returns_controller_that_solves_zero_load_hold(self):
        controller = self.design.assemble_controller(
            block_model=self.block_model,
            ballast_modes=self.modes,
            maximum_abs_tank_mass_change_per_block_kg=np.full(3, 100.0),
        )

        self.assertIsInstance(controller, PreviewMPCController)
        result = controller.solve(
            initial_platform_state=IncrementalState.zeros(),
            actual_tank_masses_kg=np.full(3, 500.0),
            tank_capacities_kg=np.full(3, 1_000.0),
            generalized_disturbance_loads=np.zeros((3, 6)),
        )
        self.assertTrue(result.success, result.message)
        np.testing.assert_allclose(result.modal_increments_kg, 0.0, atol=1.0e-8)

    def test_serialized_identity_exposes_design_not_runtime_case(self):
        record = self.design.as_dict()
        variant = self.design.with_priority_multipliers(
            {"tank_throughput": 2.0}
        ).as_dict()

        self.assertEqual(record["identity"], self.design.identity)
        self.assertEqual(record["status"], self.design.status)
        np.testing.assert_allclose(
            record["maximum_abs_roll_pitch_deg"],
            [12.0, 12.0],
        )
        np.testing.assert_allclose(
            record["absolute_maximum_abs_roll_pitch_deg"],
            [15.0, 15.0],
        )
        np.testing.assert_allclose(
            record["maximum_abs_working_model_scope_roll_pitch_deg"],
            [5.0, 5.0],
        )
        self.assertNotIn("maximum_abs_tank_mass_change_per_block_kg", record)
        self.assertEqual(
            record["objective_time_contract"]["running_posture"],
            "trapezoidal_time_average_over_the_declared_horizon",
        )
        self.assertIn(
            "not_yet_approved",
            record["objective_time_contract_status"],
        )
        self.assertEqual(len(record["parameter_sha256"]), 64)
        self.assertNotEqual(
            record["parameter_sha256"],
            variant["parameter_sha256"],
        )

    def test_v2_solver_domain_matches_the_working_model_scope(self):
        design = research_preview_mpc_design_v2()

        np.testing.assert_allclose(
            design.maximum_abs_roll_pitch_rad
            + design.maximum_posture_slack_rad,
            design.maximum_abs_working_model_scope_roll_pitch_rad,
        )
        np.testing.assert_allclose(
            design.absolute_maximum_abs_roll_pitch_rad,
            design.maximum_abs_working_model_scope_roll_pitch_rad,
        )
        np.testing.assert_allclose(design.maximum_posture_slack_rad, 0.0)
        np.testing.assert_allclose(
            np.rad2deg(design.posture_objective_scale_rad),
            [12.0, 12.0],
        )
        self.assertEqual(design.tank_movement_objective_scale_kg, 10_250.0)

    def test_v2_objective_scales_do_not_follow_runtime_actuator_limit(self):
        design = research_preview_mpc_design_v2()
        narrow = design.assemble_controller(
            block_model=self.block_model,
            ballast_modes=self.modes,
            maximum_abs_tank_mass_change_per_block_kg=np.full(3, 50.0),
        )
        wide = design.assemble_controller(
            block_model=self.block_model,
            ballast_modes=self.modes,
            maximum_abs_tank_mass_change_per_block_kg=np.full(3, 500.0),
        )

        self.assertEqual(narrow.weights, wide.weights)
        np.testing.assert_allclose(
            narrow.constraints.maximum_abs_tank_mass_change_per_block_kg,
            [50.0, 50.0, 50.0],
        )
        np.testing.assert_allclose(
            wide.constraints.maximum_abs_tank_mass_change_per_block_kg,
            [500.0, 500.0, 500.0],
        )

    def test_v2_solves_within_scope_when_the_same_wide_problem_would_leave_it(self):
        narrow_design = research_preview_mpc_design_v2()
        wide_design = replace(
            narrow_design,
            identity="test_wide_posture_domain",
            maximum_abs_roll_pitch_rad=np.deg2rad([12.0, 12.0]),
            absolute_maximum_abs_roll_pitch_rad=np.deg2rad([12.0, 12.0]),
            maximum_abs_working_model_scope_roll_pitch_rad=np.deg2rad(
                [12.0, 12.0]
            ),
        )
        disturbance = np.zeros((3, 6))
        disturbance[:, 4] = 60.0

        def solve(design: PreviewMPCDesign):
            controller = design.assemble_controller(
                block_model=self.block_model,
                ballast_modes=self.modes,
                maximum_abs_tank_mass_change_per_block_kg=np.full(3, 500.0),
            )
            return controller.solve(
                initial_platform_state=IncrementalState.zeros(),
                actual_tank_masses_kg=np.full(3, 500.0),
                tank_capacities_kg=np.full(3, 1_000.0),
                generalized_disturbance_loads=disturbance,
            )

        wide_result = solve(wide_design)
        narrow_result = solve(narrow_design)

        self.assertTrue(wide_result.success, wide_result.message)
        self.assertTrue(narrow_result.success, narrow_result.message)
        self.assertGreater(
            np.rad2deg(wide_result.predicted_trajectory.maximum_dominant_tilt_rad),
            5.0,
        )
        self.assertLessEqual(
            np.rad2deg(narrow_result.predicted_trajectory.maximum_dominant_tilt_rad),
            5.0 + 1.0e-6,
        )

    def test_v2_rejects_solver_success_when_physical_posture_is_infeasible(self):
        design = research_preview_mpc_design_v2()
        controller = design.assemble_controller(
            block_model=self.block_model,
            ballast_modes=self.modes,
            maximum_abs_tank_mass_change_per_block_kg=np.full(3, 500.0),
        )
        disturbance = np.zeros((3, 6))
        disturbance[:, 4] = 65.061065571156

        result = controller.solve(
            initial_platform_state=IncrementalState.zeros(),
            actual_tank_masses_kg=np.full(3, 500.0),
            tank_capacities_kg=np.full(3, 1_000.0),
            generalized_disturbance_loads=disturbance,
        )

        self.assertFalse(result.success)
        self.assertGreater(
            result.maximum_posture_constraint_violation_rad,
            1.0e-6,
        )
        self.assertIn("violates constraints", result.message)

    def test_v3_freezes_objective_time_and_total_mass_scope(self):
        design = research_preview_mpc_design_v3()
        record = design.as_dict()

        self.assertEqual(design.running_posture_reference_duration_s, 3600.0)
        self.assertEqual(
            design.maximum_abs_total_tank_mass_equivalent_heave_m,
            0.01,
        )
        self.assertEqual(design.tank_movement_priority, 0.0)
        self.assertGreater(design.tank_throughput_priority, 0.0)
        self.assertEqual(
            record["objective_time_contract"]["running_posture"],
            "trapezoidal_integral_divided_by_fixed_reference_duration",
        )
        self.assertEqual(
            record["running_posture_reference_duration_s"],
            3600.0,
        )

    def test_v3_controller_carries_fixed_running_posture_reference(self):
        design = research_preview_mpc_design_v3()
        controller = design.assemble_controller(
            block_model=self.block_model,
            ballast_modes=self.modes,
            maximum_abs_tank_mass_change_per_block_kg=np.full(3, 500.0),
        )

        self.assertEqual(controller.running_posture_reference_duration_s, 3600.0)

    def test_v3_running_posture_weights_accumulate_on_one_fixed_time_basis(self):
        design = research_preview_mpc_design_v3()
        controller = design.assemble_controller(
            block_model=self.block_model,
            ballast_modes=self.modes,
            maximum_abs_tank_mass_change_per_block_kg=np.full(3, 500.0),
        )
        initial = PreviewState.from_current_platform(IncrementalState.zeros())

        totals = []
        for block_count in (1, 2):
            trajectory = self.block_model.rollout(
                initial_state=initial,
                modal_increments_kg=np.zeros((block_count, 2)),
                generalized_disturbance_loads=np.zeros((block_count, 6)),
            )
            initial_weight, sample_weights = controller._posture_time_average_weights(
                trajectory,
                normalization_duration_s=(
                    controller.running_posture_reference_duration_s
                ),
            )
            totals.append(initial_weight + float(np.sum(sample_weights)))

        self.assertAlmostEqual(totals[0], 10.0 / 3_600.0)
        self.assertAlmostEqual(totals[1], 20.0 / 3_600.0)

    def test_v4_separates_nominal_posture_from_model_scope(self):
        design = research_preview_mpc_design_v4()

        np.testing.assert_allclose(
            np.rad2deg(design.maximum_abs_roll_pitch_rad),
            [5.0, 5.0],
        )
        np.testing.assert_allclose(
            np.rad2deg(
                design.maximum_abs_roll_pitch_rad
                + design.maximum_posture_slack_rad
            ),
            [5.0, 10.0],
        )
        np.testing.assert_allclose(
            np.rad2deg(design.absolute_maximum_abs_roll_pitch_rad),
            [5.0, 10.0],
        )
        np.testing.assert_allclose(
            np.rad2deg(
                design.maximum_abs_working_model_scope_roll_pitch_rad
            ),
            [5.0, 10.0],
        )


if __name__ == "__main__":
    unittest.main()
