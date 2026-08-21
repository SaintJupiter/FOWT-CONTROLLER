import unittest

import numpy as np

from fowt_platform import (
    allocate_pitch_roll_moment_to_tanks,
    pitch_roll_moment_from_tank_mass_deltas,
)
from fowt_platform.ballast import ballast_gravity_load_about_reference


TANK_COORDINATES_M = np.array(
    [
        [2.0, 0.0, -1.0],
        [-1.0, 2.0, -1.0],
        [-1.0, -2.0, -1.0],
    ]
)
CAPACITIES_KG = np.full(3, 1_000.0)
ACTUAL_MASSES_KG = np.full(3, 500.0)


class BallastMomentAllocationTests(unittest.TestCase):
    def allocate(self, requested, *, actual=ACTUAL_MASSES_KG):
        return allocate_pitch_roll_moment_to_tanks(
            requested_pitch_roll_moment_nm=requested,
            actual_tank_masses_kg=actual,
            tank_capacities_kg=CAPACITIES_KG,
            tank_coordinates_m=TANK_COORDINATES_M,
            gravity_m_s2=10.0,
        )

    def test_zero_request_preserves_actual_masses_and_total_ballast(self):
        result = self.allocate([0.0, 0.0])

        np.testing.assert_array_equal(result.tank_mass_deltas_kg, np.zeros(3))
        np.testing.assert_array_equal(result.target_tank_masses_kg, ACTUAL_MASSES_KG)
        np.testing.assert_array_equal(result.achieved_pitch_roll_moment_nm, [0.0, 0.0])
        self.assertEqual(result.feasible_scale, 1.0)
        self.assertFalse(result.capacity_limited)

    def test_known_zero_net_delta_is_recovered_for_coupled_pitch_roll_request(self):
        expected_deltas = np.array([80.0, -25.0, -55.0])
        requested = pitch_roll_moment_from_tank_mass_deltas(
            tank_mass_deltas_kg=expected_deltas,
            tank_coordinates_m=TANK_COORDINATES_M,
            gravity_m_s2=10.0,
        )

        result = self.allocate(requested)

        np.testing.assert_allclose(result.tank_mass_deltas_kg, expected_deltas)
        np.testing.assert_allclose(result.achieved_pitch_roll_moment_nm, requested)
        np.testing.assert_allclose(result.residual_pitch_roll_moment_nm, [0.0, 0.0])
        self.assertAlmostEqual(float(np.sum(result.tank_mass_deltas_kg)), 0.0)

    def test_pitch_roll_request_order_matches_explicit_gravity_load_components(self):
        x_tank_load = ballast_gravity_load_about_reference(
            tank_mass_deltas_kg=[1.0, 0.0, 0.0],
            tank_coordinates_m=TANK_COORDINATES_M,
            gravity_m_s2=10.0,
        )
        y_tank_load = ballast_gravity_load_about_reference(
            tank_mass_deltas_kg=[0.0, 1.0, 0.0],
            tank_coordinates_m=TANK_COORDINATES_M,
            gravity_m_s2=10.0,
        )

        self.assertAlmostEqual(x_tank_load[4], 20.0)
        self.assertAlmostEqual(x_tank_load[3], 0.0)
        self.assertAlmostEqual(y_tank_load[4], -10.0)
        self.assertAlmostEqual(y_tank_load[3], -20.0)
        np.testing.assert_allclose(
            pitch_roll_moment_from_tank_mass_deltas(
                tank_mass_deltas_kg=[1.0, 0.0, 0.0],
                tank_coordinates_m=TANK_COORDINATES_M,
                gravity_m_s2=10.0,
            ),
            [20.0, 0.0],
        )
        np.testing.assert_allclose(
            pitch_roll_moment_from_tank_mass_deltas(
                tank_mass_deltas_kg=[0.0, 1.0, 0.0],
                tank_coordinates_m=TANK_COORDINATES_M,
                gravity_m_s2=10.0,
            ),
            [-10.0, -20.0],
        )

    def test_pure_pitch_and_pure_roll_requests_follow_ballast_gravity_signs(self):
        pitch_deltas = np.array([60.0, -30.0, -30.0])
        roll_deltas = np.array([0.0, 40.0, -40.0])
        for name, expected_deltas in (("pitch", pitch_deltas), ("roll", roll_deltas)):
            with self.subTest(name=name):
                requested = pitch_roll_moment_from_tank_mass_deltas(
                    tank_mass_deltas_kg=expected_deltas,
                    tank_coordinates_m=TANK_COORDINATES_M,
                    gravity_m_s2=10.0,
                )
                result = self.allocate(requested)
                np.testing.assert_allclose(result.tank_mass_deltas_kg, expected_deltas)
                np.testing.assert_allclose(result.achieved_pitch_roll_moment_nm, requested)

    def test_capacity_limit_scales_request_without_changing_moment_direction(self):
        expected_deltas = np.array([-100.0, 50.0, 50.0])
        requested = pitch_roll_moment_from_tank_mass_deltas(
            tank_mass_deltas_kg=expected_deltas,
            tank_coordinates_m=TANK_COORDINATES_M,
            gravity_m_s2=10.0,
        )
        result = self.allocate(requested, actual=np.array([20.0, 500.0, 500.0]))

        self.assertTrue(result.capacity_limited)
        self.assertAlmostEqual(result.feasible_scale, 0.2)
        np.testing.assert_allclose(result.tank_mass_deltas_kg, 0.2 * expected_deltas)
        np.testing.assert_allclose(result.achieved_pitch_roll_moment_nm, 0.2 * requested)
        np.testing.assert_allclose(result.target_tank_masses_kg, [0.0, 510.0, 510.0])

    def test_capacity_limit_for_inflow_preserves_bounds_and_zero_net_mass(self):
        expected_deltas = np.array([100.0, -50.0, -50.0])
        requested = pitch_roll_moment_from_tank_mass_deltas(
            tank_mass_deltas_kg=expected_deltas,
            tank_coordinates_m=TANK_COORDINATES_M,
            gravity_m_s2=10.0,
        )
        result = self.allocate(requested, actual=np.array([950.0, 500.0, 500.0]))

        self.assertTrue(result.capacity_limited)
        self.assertAlmostEqual(result.feasible_scale, 0.5)
        np.testing.assert_allclose(result.tank_mass_deltas_kg, 0.5 * expected_deltas)
        np.testing.assert_allclose(result.achieved_pitch_roll_moment_nm, 0.5 * requested)
        np.testing.assert_allclose(result.target_tank_masses_kg, [1_000.0, 475.0, 475.0])
        self.assertTrue(np.all(result.target_tank_masses_kg >= 0.0))
        self.assertTrue(np.all(result.target_tank_masses_kg <= CAPACITIES_KG))
        self.assertAlmostEqual(float(np.sum(result.tank_mass_deltas_kg)), 0.0)

    def test_collinear_tank_geometry_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "cannot span independent"):
            allocate_pitch_roll_moment_to_tanks(
                requested_pitch_roll_moment_nm=[100.0, 200.0],
                actual_tank_masses_kg=ACTUAL_MASSES_KG,
                tank_capacities_kg=CAPACITIES_KG,
                tank_coordinates_m=[
                    [0.0, 0.0, -1.0],
                    [1.0, 0.0, -1.0],
                    [2.0, 0.0, -1.0],
                ],
            )

    def test_out_of_bound_actual_mass_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "within tank capacities"):
            self.allocate([100.0, 0.0], actual=np.array([-1.0, 500.0, 500.0]))


if __name__ == "__main__":
    unittest.main()
