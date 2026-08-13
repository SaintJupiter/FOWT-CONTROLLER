import unittest

import numpy as np

from fowt_platform import analyze_undamped_modes


class PlatformModalAnalysisTests(unittest.TestCase):
    def test_diagonal_system_returns_known_frequencies_and_periods(self):
        mass = np.diag([1.0, 2.0, 4.0, 5.0, 8.0, 10.0])
        expected_angular_frequency = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        stiffness = np.diag(np.diag(mass) * expected_angular_frequency**2)

        modes = analyze_undamped_modes(mass=mass, stiffness=stiffness)

        np.testing.assert_allclose(
            modes.angular_frequency_rad_s,
            expected_angular_frequency,
        )
        np.testing.assert_allclose(
            modes.frequency_hz,
            expected_angular_frequency / (2.0 * np.pi),
        )
        np.testing.assert_allclose(
            modes.period_s,
            2.0 * np.pi / expected_angular_frequency,
        )

    def test_mode_shapes_are_mass_normalized(self):
        mass = np.diag([2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
        stiffness = np.diag([4.0, 12.0, 36.0, 80.0, 150.0, 252.0])

        modes = analyze_undamped_modes(mass=mass, stiffness=stiffness)

        np.testing.assert_allclose(
            modes.mode_shapes.T @ mass @ modes.mode_shapes,
            np.eye(6),
            atol=1e-12,
        )

    def test_negative_stiffness_mode_is_rejected(self):
        stiffness = np.eye(6)
        stiffness[0, 0] = -1.0

        with self.assertRaisesRegex(ValueError, "stiffness must be positive semidefinite"):
            analyze_undamped_modes(mass=np.eye(6), stiffness=stiffness)

    def test_low_frequency_mode_is_not_removed_by_large_stiffness_in_other_dof(self):
        mass = np.diag([1.0e4, 1.0, 1.0, 1.0, 1.0, 1.0])
        stiffness = np.diag([1.0, 4.0, 9.0, 16.0, 25.0, 1.0e12])

        modes = analyze_undamped_modes(mass=mass, stiffness=stiffness)

        self.assertAlmostEqual(modes.angular_frequency_rad_s[0], 0.01)
        self.assertTrue(np.all(modes.angular_frequency_rad_s > 0.0))


if __name__ == "__main__":
    unittest.main()
