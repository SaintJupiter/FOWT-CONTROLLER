import sys
import unittest
from contextlib import redirect_stdout
import io
from pathlib import Path

import numpy as np


LEGACY_DIR = Path(__file__).resolve().parents[1] / "archive" / "legacy_fowt_control"
sys.path.insert(0, str(LEGACY_DIR))

from core_model import FloatingPlatform  # noqa: E402


class PlatformGeneralizedLoadsTests(unittest.TestCase):
    def setUp(self):
        with redirect_stdout(io.StringIO()):
            self.plant = FloatingPlatform(
                None,
                allow_linear_mooring_fallback=True,
            )
        self.plant.wave_components = []

    def test_components_sum_to_total_without_mutating_state(self):
        state = np.array(
            [2.0, -3.0, 0.4, 0.02, -0.03, 0.01, 0.5, -0.4, 0.2, 0.01, -0.02, 0.03],
            dtype=float,
        )
        state_before = state.copy()
        plant_state_before = self.plant.state.copy()

        loads = self.plant.generalized_load_components(5.0, state, 1.2e6, 30.0)

        expected_names = {
            "wind",
            "wave",
            "quadratic_drag",
            "mooring",
            "hydrostatic",
            "yaw_restoring",
            "dry_gravity",
            "ballast_gravity",
            "linear_damping",
            "total",
        }
        self.assertEqual(set(loads), expected_names)
        for values in loads.values():
            self.assertEqual(values.shape, (6,))
            self.assertTrue(np.all(np.isfinite(values)))
        component_sum = sum(
            values for name, values in loads.items() if name != "total"
        )
        np.testing.assert_allclose(loads["total"], component_sum)
        np.testing.assert_array_equal(state, state_before)
        np.testing.assert_array_equal(self.plant.state, plant_state_before)

    def test_cardinal_wind_directions_use_documented_force_and_moment_signs(self):
        state = np.zeros(12, dtype=float)
        thrust = 100.0
        arm = self.plant.arm_aero
        expected = {
            0.0: [100.0, 0.0, 0.0, 0.0, 100.0 * arm, 0.0],
            90.0: [0.0, 100.0, 0.0, -100.0 * arm, 0.0, 0.0],
            180.0: [-100.0, 0.0, 0.0, 0.0, -100.0 * arm, 0.0],
            270.0: [0.0, -100.0, 0.0, 100.0 * arm, 0.0, 0.0],
        }
        for direction, expected_wind in expected.items():
            with self.subTest(direction=direction):
                wind = self.plant.generalized_load_components(
                    0.0,
                    state,
                    thrust,
                    direction,
                )["wind"]
                np.testing.assert_allclose(wind, expected_wind, atol=1e-10)

    def test_mooring_and_hydrostatic_terms_oppose_displacement(self):
        state = np.zeros(12, dtype=float)
        state[:6] = [8.0, -7.0, 1.5, 0.04, -0.03, 0.02]
        loads = self.plant.generalized_load_components(0.0, state, 0.0, 0.0)

        self.assertLess(loads["mooring"][0] * state[0], 0.0)
        self.assertLess(loads["mooring"][1] * state[1], 0.0)
        for index in (2, 3, 4):
            self.assertLess(loads["hydrostatic"][index] * state[index], 0.0)
        self.assertLess(loads["yaw_restoring"][5] * state[5], 0.0)

    def test_quadratic_and_linear_damping_remove_kinetic_motion(self):
        state = np.zeros(12, dtype=float)
        state[6:12] = [1.5, -1.0, 0.7, 0.02, -0.03, 0.01]
        loads = self.plant.generalized_load_components(0.0, state, 0.0, 0.0)

        self.assertLess(np.dot(loads["quadratic_drag"][:3], state[6:9]), 0.0)
        self.assertLess(np.dot(loads["linear_damping"], state[6:12]), 0.0)

    def test_dynamics_uses_the_reported_total_load(self):
        state = np.array(
            [1.0, 2.0, -0.5, 0.01, -0.02, 0.03, 0.2, -0.1, 0.05, 0.0, 0.01, -0.02],
            dtype=float,
        )
        loads = self.plant.generalized_load_components(3.0, state, 9.0e5, 42.0)
        derivative = self.plant._dynamics(3.0, state, 9.0e5, 42.0)

        np.testing.assert_allclose(derivative[:6], state[6:12])
        np.testing.assert_allclose(derivative[6:12], self.plant.M_inv @ loads["total"])

    def test_invalid_state_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "12 finite"):
            self.plant.generalized_load_components(0.0, np.zeros(11), 0.0, 0.0)

    def test_seeded_wave_is_reproducible_and_heading_rotates_horizontal_load(self):
        with redirect_stdout(io.StringIO()):
            self.plant.set_irregular_wave(
                Hs=2.0,
                Tp=8.0,
                direction_deg=0.0,
                n_freqs=24,
                seed=17,
            )
            rotated = FloatingPlatform(None, allow_linear_mooring_fallback=True)
            rotated.set_irregular_wave(
                Hs=2.0,
                Tp=8.0,
                direction_deg=90.0,
                n_freqs=24,
                seed=17,
            )

        reference_force = self.plant._get_wave_forces(3.5)
        rotated_force = rotated._get_wave_forces(3.5)
        self.assertAlmostEqual(rotated_force[0], 0.0, delta=1e-8)
        self.assertAlmostEqual(rotated_force[1], reference_force[0], delta=1e-8)
        self.assertAlmostEqual(rotated_force[2], reference_force[2], delta=1e-8)
        self.assertAlmostEqual(rotated_force[3], -reference_force[4], delta=1e-8)
        self.assertAlmostEqual(rotated_force[4], 0.0, delta=1e-8)

    def test_incremental_profile_has_zero_load_at_reference_state(self):
        with redirect_stdout(io.StringIO()):
            plant = FloatingPlatform(
                None,
                platform_profile="research_incremental_v1",
                allow_linear_mooring_fallback=True,
            )
        plant.wave_components = []

        loads = plant.generalized_load_components(
            0.0,
            np.zeros(12, dtype=float),
            0.0,
            0.0,
        )

        np.testing.assert_allclose(loads["dry_gravity"], 0.0, atol=1e-12)
        np.testing.assert_allclose(loads["ballast_gravity"], 0.0, atol=1e-12)
        np.testing.assert_allclose(loads["total"], 0.0, atol=1e-12)

    def test_incremental_profile_single_tank_change_has_expected_load(self):
        with redirect_stdout(io.StringIO()):
            plant = FloatingPlatform(
                None,
                platform_profile="research_incremental_v1",
                allow_linear_mooring_fallback=True,
            )
        plant.wave_components = []
        increment_kg = 1000.0
        changed = plant.reference_ballast_mass.copy()
        changed[0] += increment_kg
        plant.force_ballast_mass(*changed)

        ballast = plant.generalized_load_components(
            0.0,
            np.zeros(12, dtype=float),
            0.0,
            0.0,
        )["ballast_gravity"]
        expected = np.zeros(6, dtype=float)
        expected[2] = -increment_kg * plant.g
        expected[3:6] = np.cross(
            plant.tank_pos[0],
            [0.0, 0.0, -increment_kg * plant.g],
        )

        np.testing.assert_allclose(ballast, expected, atol=1e-9)


if __name__ == "__main__":
    unittest.main()
