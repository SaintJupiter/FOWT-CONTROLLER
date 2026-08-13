import unittest

import numpy as np

from fowt_platform import (
    ballast_gravity_load_about_reference,
    compute_ballast_mass_properties,
    compute_incremental_ballast_mass_properties,
    rigid_body_mass_matrix_about_reference,
    weight_stiffness_about_reference,
)


class BallastMassPropertiesTests(unittest.TestCase):
    def test_weight_stiffness_uses_vertical_center_of_mass_offset(self):
        stiffness = weight_stiffness_about_reference(
            total_mass_kg=10.0,
            center_of_mass_m=[1.0, -2.0, -3.0],
            gravity_m_s2=10.0,
        )

        expected = np.zeros((6, 6))
        expected[3, 3] = 300.0
        expected[4, 4] = 300.0
        np.testing.assert_array_equal(stiffness, expected)

    def test_ballast_gravity_load_follows_right_hand_moment_signs(self):
        load = ballast_gravity_load_about_reference(
            tank_mass_deltas_kg=[10.0, -4.0, 2.0],
            tank_coordinates_m=[
                [2.0, 3.0, -5.0],
                [-1.0, 4.0, -5.0],
                [0.0, -2.0, -5.0],
            ],
            gravity_m_s2=10.0,
        )

        # Fz = -g sum(dm), K = sum(y Fz), M = sum(-x Fz).
        np.testing.assert_allclose(
            load,
            [0.0, 0.0, -80.0, -100.0, 240.0, 0.0],
        )

    def test_balanced_three_tank_change_has_no_heave_force(self):
        load = ballast_gravity_load_about_reference(
            tank_mass_deltas_kg=[100.0, -50.0, -50.0],
            tank_coordinates_m=[
                [46.2, 0.0, -10.0],
                [-23.1, 40.0, -10.0],
                [-23.1, -40.0, -10.0],
            ],
        )

        self.assertEqual(load[2], 0.0)
        self.assertEqual(load[3], 0.0)
        self.assertGreater(load[4], 0.0)

    def test_rigid_body_mass_matrix_uses_reference_point_coupling_signs(self):
        mass = 10.0
        center = np.array([1.0, 2.0, -3.0])
        inertia_at_center = np.diag([100.0, 110.0, 120.0])
        inertia_about_reference = inertia_at_center + mass * (
            np.dot(center, center) * np.eye(3) - np.outer(center, center)
        )
        matrix = rigid_body_mass_matrix_about_reference(
            total_mass_kg=mass,
            center_of_mass_m=center,
            inertia_about_reference_kg_m2=inertia_about_reference,
        )

        expected_coupling = np.array(
            [
                [0.0, -30.0, -20.0],
                [30.0, 0.0, 10.0],
                [20.0, -10.0, 0.0],
            ]
        )
        np.testing.assert_allclose(matrix[:3, :3], mass * np.eye(3))
        np.testing.assert_allclose(matrix[:3, 3:], expected_coupling)
        np.testing.assert_allclose(matrix[3:, :3], expected_coupling.T)
        np.testing.assert_allclose(matrix[3:, 3:], inertia_about_reference)
        np.testing.assert_allclose(matrix, matrix.T)

    def test_zero_increment_reproduces_complete_reference_properties(self):
        reference_inertia = np.array(
            [[20.0, 1.0, 0.0], [1.0, 30.0, 2.0], [0.0, 2.0, 40.0]]
        )
        result = compute_incremental_ballast_mass_properties(
            reference_mass_kg=100.0,
            reference_center_of_mass_m=[1.0, -2.0, 3.0],
            reference_inertia_about_reference_kg_m2=reference_inertia,
            tank_mass_deltas_kg=[0.0, 0.0, 0.0],
            tank_coordinates_m=np.eye(3),
        )

        self.assertEqual(result.total_mass_kg, 100.0)
        np.testing.assert_array_equal(
            result.center_of_mass_m,
            [1.0, -2.0, 3.0],
        )
        np.testing.assert_array_equal(
            result.inertia_about_reference_kg_m2,
            reference_inertia,
        )

    def test_signed_increment_updates_reference_without_recounting_ballast(self):
        result = compute_incremental_ballast_mass_properties(
            reference_mass_kg=1000.0,
            reference_center_of_mass_m=[0.0, 0.0, 0.0],
            reference_inertia_about_reference_kg_m2=np.diag(
                [1000.0, 1000.0, 1000.0]
            ),
            tank_mass_deltas_kg=[10.0, -4.0, 0.0],
            tank_coordinates_m=[
                [2.0, 0.0, 0.0],
                [-1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ],
        )

        self.assertEqual(result.total_mass_kg, 1006.0)
        np.testing.assert_allclose(
            result.center_of_mass_m,
            [24.0 / 1006.0, 0.0, 0.0],
        )
        np.testing.assert_allclose(
            result.inertia_about_reference_kg_m2,
            np.diag([1000.0, 1036.0, 1036.0]),
        )

    def test_increment_rejects_nonphysical_final_properties(self):
        with self.assertRaisesRegex(ValueError, "total mass must remain positive"):
            compute_incremental_ballast_mass_properties(
                reference_mass_kg=10.0,
                reference_center_of_mass_m=[0.0, 0.0, 0.0],
                reference_inertia_about_reference_kg_m2=np.eye(3),
                tank_mass_deltas_kg=[-11.0, 0.0, 0.0],
                tank_coordinates_m=np.zeros((3, 3)),
            )

    def test_symmetric_three_tanks_preserve_center_and_add_known_inertia(self):
        sqrt_three = np.sqrt(3.0)
        result = compute_ballast_mass_properties(
            dry_mass_kg=1000.0,
            dry_center_of_mass_m=[0.0, 0.0, 0.0],
            dry_inertia_about_reference_kg_m2=np.diag([100.0, 200.0, 300.0]),
            tank_masses_kg=[10.0, 10.0, 10.0],
            tank_coordinates_m=[
                [1.0, 0.0, 0.0],
                [-0.5, sqrt_three / 2.0, 0.0],
                [-0.5, -sqrt_three / 2.0, 0.0],
            ],
        )

        self.assertEqual(result.total_mass_kg, 1030.0)
        np.testing.assert_allclose(
            result.center_of_mass_m,
            np.zeros(3),
            atol=1e-14,
        )
        np.testing.assert_allclose(
            result.inertia_about_reference_kg_m2,
            np.diag([115.0, 215.0, 330.0]),
            atol=1e-12,
        )

    def test_single_tank_increment_matches_parallel_axis_theorem(self):
        dry_inertia = np.diag([11.0, 13.0, 17.0])
        tank_mass = 5.0
        tank_coordinate = np.array([2.0, 3.0, 4.0])
        expected_increment = tank_mass * np.array(
            [
                [25.0, -6.0, -8.0],
                [-6.0, 20.0, -12.0],
                [-8.0, -12.0, 13.0],
            ]
        )

        result = compute_ballast_mass_properties(
            dry_mass_kg=20.0,
            dry_center_of_mass_m=[0.0, 0.0, 0.0],
            dry_inertia_about_reference_kg_m2=dry_inertia,
            tank_masses_kg=[tank_mass, 0.0, 0.0],
            tank_coordinates_m=[
                tank_coordinate,
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ],
        )

        np.testing.assert_allclose(
            result.inertia_about_reference_kg_m2,
            dry_inertia + expected_increment,
        )

    def test_total_mass_and_center_of_mass_follow_first_moments(self):
        result = compute_ballast_mass_properties(
            dry_mass_kg=10.0,
            dry_center_of_mass_m=[1.0, 0.0, 0.0],
            dry_inertia_about_reference_kg_m2=np.eye(3),
            tank_masses_kg=[2.0, 3.0, 5.0],
            tank_coordinates_m=[
                [0.0, 0.0, 0.0],
                [0.0, 2.0, 0.0],
                [0.0, 0.0, 4.0],
            ],
        )

        self.assertEqual(result.total_mass_kg, 20.0)
        np.testing.assert_allclose(result.center_of_mass_m, [0.5, 0.3, 1.0])

    def test_combined_inertia_is_symmetric(self):
        result = compute_ballast_mass_properties(
            dry_mass_kg=50.0,
            dry_center_of_mass_m=[0.2, -0.3, 0.4],
            dry_inertia_about_reference_kg_m2=[
                [20.0, 1.0, -2.0],
                [1.0, 30.0, 3.0],
                [-2.0, 3.0, 40.0],
            ],
            tank_masses_kg=[4.0, 5.0, 6.0],
            tank_coordinates_m=[
                [1.0, 2.0, 3.0],
                [-2.0, 0.5, 1.0],
                [0.0, -1.0, 2.0],
            ],
        )

        np.testing.assert_array_equal(
            result.inertia_about_reference_kg_m2,
            result.inertia_about_reference_kg_m2.T,
        )

    def test_rejects_invalid_dimensions_nonfinite_values_and_negative_mass(self):
        valid = {
            "dry_mass_kg": 10.0,
            "dry_center_of_mass_m": [0.0, 0.0, 0.0],
            "dry_inertia_about_reference_kg_m2": np.eye(3),
            "tank_masses_kg": [1.0, 1.0, 1.0],
            "tank_coordinates_m": np.zeros((3, 3)),
        }
        invalid_overrides = (
            {"dry_mass_kg": -1.0},
            {"dry_mass_kg": np.inf},
            {"dry_center_of_mass_m": [0.0, 0.0]},
            {"dry_center_of_mass_m": [0.0, np.nan, 0.0]},
            {"dry_inertia_about_reference_kg_m2": np.eye(2)},
            {
                "dry_inertia_about_reference_kg_m2": [
                    [1.0, 2.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                ]
            },
            {"tank_masses_kg": [1.0, -1.0, 1.0]},
            {"tank_masses_kg": [1.0, 1.0]},
            {"tank_coordinates_m": np.zeros((3, 2))},
            {
                "tank_coordinates_m": (
                    [[0.0, 0.0, 0.0]] * 2
                    + [[0.0, 0.0, np.inf]]
                )
            },
        )

        for overrides in invalid_overrides:
            with self.subTest(overrides=overrides):
                arguments = dict(valid)
                arguments.update(overrides)
                with self.assertRaises(ValueError):
                    compute_ballast_mass_properties(**arguments)

    def test_rejects_zero_total_mass_because_center_is_undefined(self):
        with self.assertRaisesRegex(ValueError, "total mass must be positive"):
            compute_ballast_mass_properties(
                dry_mass_kg=0.0,
                dry_center_of_mass_m=[0.0, 0.0, 0.0],
                dry_inertia_about_reference_kg_m2=np.zeros((3, 3)),
                tank_masses_kg=[0.0, 0.0, 0.0],
                tank_coordinates_m=np.zeros((3, 3)),
            )

    def test_rejects_nonpositive_combined_inertia(self):
        with self.assertRaisesRegex(ValueError, "positive definite"):
            compute_ballast_mass_properties(
                dry_mass_kg=10.0,
                dry_center_of_mass_m=[0.0, 0.0, 0.0],
                dry_inertia_about_reference_kg_m2=np.diag([1.0, 1.0, -1.0]),
                tank_masses_kg=[0.0, 0.0, 0.0],
                tank_coordinates_m=np.zeros((3, 3)),
            )


if __name__ == "__main__":
    unittest.main()
