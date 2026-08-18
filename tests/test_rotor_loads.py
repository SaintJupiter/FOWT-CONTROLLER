import math
import unittest

import numpy as np

from fowt_platform.rotor_loads import quasi_steady_rotor_normal_load


class RotorLoadTests(unittest.TestCase):
    def test_load_uses_explicit_quasi_steady_ct_relation(self):
        load = quasi_steady_rotor_normal_load(
            air_density_kg_m3=1.225,
            rotor_radius_m=120.97,
            thrust_coefficient=0.73730209,
            relative_normal_wind_speed_mps=10.74,
            downwind_rotor_normal_platform=[
                math.cos(math.radians(-6.0)),
                0.0,
                math.sin(math.radians(-6.0)),
            ],
        )

        expected_thrust = (
            0.5
            * 1.225
            * math.pi
            * 120.97**2
            * 0.73730209
            * 10.74**2
        )
        self.assertAlmostEqual(load.thrust_n, expected_thrust)
        np.testing.assert_allclose(
            load.force_platform_n,
            expected_thrust
            * np.array(
                [
                    math.cos(math.radians(-6.0)),
                    0.0,
                    math.sin(math.radians(-6.0)),
                ]
            ),
        )
        self.assertFalse(load.force_platform_n.flags.writeable)

    def test_zero_relative_wind_returns_zero_force(self):
        load = quasi_steady_rotor_normal_load(
            air_density_kg_m3=1.225,
            rotor_radius_m=120.97,
            thrust_coefficient=0.75,
            relative_normal_wind_speed_mps=0.0,
            downwind_rotor_normal_platform=[1.0, 0.0, 0.0],
        )

        self.assertEqual(load.thrust_n, 0.0)
        np.testing.assert_allclose(load.force_platform_n, [0.0, 0.0, 0.0])

    def test_rejects_ambiguous_or_nonphysical_inputs(self):
        common = dict(
            air_density_kg_m3=1.225,
            rotor_radius_m=120.97,
            thrust_coefficient=0.75,
            relative_normal_wind_speed_mps=10.0,
            downwind_rotor_normal_platform=[1.0, 0.0, 0.0],
        )
        with self.assertRaisesRegex(ValueError, "thrust_coefficient"):
            quasi_steady_rotor_normal_load(
                **{**common, "thrust_coefficient": -0.1}
            )
        with self.assertRaisesRegex(ValueError, "relative_normal_wind_speed_mps"):
            quasi_steady_rotor_normal_load(
                **{**common, "relative_normal_wind_speed_mps": -0.1}
            )
        with self.assertRaisesRegex(ValueError, "unit vector"):
            quasi_steady_rotor_normal_load(
                **{**common, "downwind_rotor_normal_platform": [2.0, 0.0, 0.0]}
            )


if __name__ == "__main__":
    unittest.main()
