import unittest

import numpy as np

from fowt_platform.ballast_modes import ThreeTankDifferentialModes


TANK_COORDINATES_M = np.array(
    [
        [2.0, 0.0, -1.0],
        [-1.0, 2.0, -1.0],
        [-1.0, -2.0, -1.0],
    ]
)


class ThreeTankDifferentialModesTests(unittest.TestCase):
    def setUp(self):
        self.modes = ThreeTankDifferentialModes(
            tank_coordinates_m=TANK_COORDINATES_M,
            gravity_m_s2=10.0,
        )

    def test_geometry_derives_zero_net_unit_pitch_and_roll_modes(self):
        basis = self.modes.tank_mass_basis

        np.testing.assert_allclose(np.sum(basis, axis=0), np.zeros(2), atol=1e-14)
        np.testing.assert_allclose(np.linalg.norm(basis, axis=0), np.ones(2))
        pitch_roll = self.modes.pitch_roll_moment_per_mode_kg
        self.assertGreater(pitch_roll[0, 0], 0.0)
        self.assertGreater(pitch_roll[1, 1], 0.0)
        np.testing.assert_allclose(pitch_roll[0, 1], 0.0, atol=1e-12)
        np.testing.assert_allclose(pitch_roll[1, 0], 0.0, atol=1e-12)

    def test_modal_round_trip_separates_common_mass_change(self):
        expected_modal = np.array([80.0, -25.0])
        common = 7.5
        tank_deltas = self.modes.tank_mass_deltas_from_modes(expected_modal) + common

        projection = self.modes.project_tank_mass_deltas(tank_deltas)

        np.testing.assert_allclose(projection.modal_masses_kg, expected_modal)
        self.assertAlmostEqual(projection.common_tank_mass_delta_kg, common)
        np.testing.assert_allclose(
            projection.residual_tank_mass_deltas_kg,
            0.0,
            atol=1.0e-12,
        )

    def test_generalized_load_matches_modal_tank_mass_change(self):
        modal = np.array([35.0, -12.0])
        direct = self.modes.generalized_load_from_modes(modal)
        mapped = self.modes.generalized_load_per_mode_kg @ modal

        np.testing.assert_allclose(direct, mapped)
        self.assertAlmostEqual(direct[2], 0.0, places=12)

    def test_capacity_constraints_match_explicit_tank_targets(self):
        reference = np.array([500.0, 450.0, 550.0])
        capacity = np.full(3, 1_000.0)
        matrix, bound = self.modes.modal_constraint_inequalities(
            reference_tank_masses_kg=reference,
            tank_capacities_kg=capacity,
        )
        feasible_modal = np.array([100.0, -50.0])
        target = self.modes.target_tank_masses(
            reference_tank_masses_kg=reference,
            modal_masses_kg=feasible_modal,
        )

        self.assertTrue(np.all(matrix @ feasible_modal <= bound + 1e-12))
        self.assertTrue(np.all(target >= 0.0))
        self.assertTrue(np.all(target <= capacity))

    def test_collinear_geometry_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "cannot form independent"):
            ThreeTankDifferentialModes(
                tank_coordinates_m=[
                    [0.0, 0.0, -1.0],
                    [1.0, 0.0, -1.0],
                    [2.0, 0.0, -1.0],
                ]
            )


if __name__ == "__main__":
    unittest.main()
