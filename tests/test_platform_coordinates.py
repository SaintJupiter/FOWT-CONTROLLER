import unittest

import numpy as np

from fowt_platform import (
    enu_wind_to_platform,
    meteorological_wind_to_enu,
    meteorological_wind_to_platform,
    true_heading_from_yaw,
)


class PlatformCoordinateTests(unittest.TestCase):
    def test_cardinal_meteorological_directions_produce_downwind_enu_vectors(self):
        speed = 12.0
        cases = (
            (0.0, [0.0, -speed]),
            (0.5 * np.pi, [-speed, 0.0]),
            (np.pi, [0.0, speed]),
            (1.5 * np.pi, [speed, 0.0]),
        )

        for direction_from, expected in cases:
            with self.subTest(direction_from=direction_from):
                np.testing.assert_allclose(
                    meteorological_wind_to_enu(speed, direction_from),
                    expected,
                    atol=1e-12,
                )

    def test_enu_vector_is_resolved_in_forward_and_port_axes(self):
        eastward_wind = np.array([8.0, 0.0])

        np.testing.assert_allclose(
            enu_wind_to_platform(eastward_wind, heading_rad=0.0),
            [0.0, -8.0],
            atol=1e-12,
        )
        np.testing.assert_allclose(
            enu_wind_to_platform(eastward_wind, heading_rad=0.5 * np.pi),
            [8.0, 0.0],
            atol=1e-12,
        )

    def test_coordinate_rotation_preserves_wind_speed(self):
        platform_wind = meteorological_wind_to_platform(
            speed=17.5,
            direction_from_rad=0.37,
            heading_rad=1.24,
        )

        self.assertAlmostEqual(float(np.linalg.norm(platform_wind)), 17.5)

    def test_combined_direction_cases_follow_platform_axis_signs(self):
        speed = 10.0
        cases = (
            (0.0, 0.0, [-speed, 0.0]),
            (0.5 * np.pi, 0.0, [0.0, speed]),
            (0.0, 0.5 * np.pi, [0.0, -speed]),
        )

        for direction_from, heading, expected in cases:
            with self.subTest(direction_from=direction_from, heading=heading):
                np.testing.assert_allclose(
                    meteorological_wind_to_platform(
                        speed=speed,
                        direction_from_rad=direction_from,
                        heading_rad=heading,
                    ),
                    expected,
                    atol=1e-12,
                )

    def test_positive_right_hand_yaw_reduces_clockwise_true_heading(self):
        heading = true_heading_from_yaw(
            nominal_heading_rad=0.25,
            yaw_rad=0.10,
        )

        self.assertAlmostEqual(heading, 0.15)

    def test_true_heading_is_continuous_and_not_wrapped(self):
        heading = true_heading_from_yaw(
            nominal_heading_rad=0.05,
            yaw_rad=0.10,
        )

        self.assertAlmostEqual(heading, -0.05)

    def test_negative_wind_speed_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "speed must be non-negative"):
            meteorological_wind_to_enu(-1.0, 0.0)


if __name__ == "__main__":
    unittest.main()
