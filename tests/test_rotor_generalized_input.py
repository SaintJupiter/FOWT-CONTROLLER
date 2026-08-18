import unittest

import numpy as np

from fowt_platform import (
    IncrementalLoads,
    IncrementalPlatformModel,
    IncrementalState,
    PlatformMatrices,
    generalized_load_from_point_force,
    quasi_steady_rotor_generalized_load_from_platform_motion,
)


class RotorGeneralizedInputTests(unittest.TestCase):
    def test_stationary_platform_matches_point_force_mapping(self):
        application_point = np.array([4.0, 0.0, 120.0])
        assembled = quasi_steady_rotor_generalized_load_from_platform_motion(
            air_density_kg_m3=1.225,
            rotor_radius_m=100.0,
            thrust_coefficient=0.8,
            ambient_air_velocity_platform_mps=[10.0, 0.0, 0.0],
            platform_reference_velocity_platform_mps=[0.0, 0.0, 0.0],
            platform_angular_velocity_platform_radps=[0.0, 0.0, 0.0],
            rotor_force_application_point_from_reference_m=application_point,
            downwind_rotor_normal_platform=[1.0, 0.0, 0.0],
        )

        expected_thrust = 0.5 * 1.225 * np.pi * 100.0**2 * 0.8 * 10.0**2
        self.assertAlmostEqual(assembled.normal_load.thrust_n, expected_thrust)
        np.testing.assert_allclose(
            assembled.relative_air_velocity_platform_mps,
            [10.0, 0.0, 0.0],
        )
        np.testing.assert_allclose(
            assembled.generalized_load_platform,
            generalized_load_from_point_force(
                assembled.normal_load.force_platform_n,
                application_point,
            ),
        )

    def test_platform_motion_changes_relative_wind_and_rotor_load(self):
        common = dict(
            air_density_kg_m3=1.225,
            rotor_radius_m=100.0,
            thrust_coefficient=0.8,
            ambient_air_velocity_platform_mps=[10.0, 0.0, 0.0],
            platform_angular_velocity_platform_radps=[0.0, 0.0, 0.0],
            rotor_force_application_point_from_reference_m=[0.0, 0.0, 120.0],
            downwind_rotor_normal_platform=[1.0, 0.0, 0.0],
        )
        stationary = quasi_steady_rotor_generalized_load_from_platform_motion(
            **common,
            platform_reference_velocity_platform_mps=[0.0, 0.0, 0.0],
        )
        moving_downwind = quasi_steady_rotor_generalized_load_from_platform_motion(
            **common,
            platform_reference_velocity_platform_mps=[2.0, 0.0, 0.0],
        )

        np.testing.assert_allclose(
            moving_downwind.relative_air_velocity_platform_mps,
            [8.0, 0.0, 0.0],
        )
        self.assertAlmostEqual(
            moving_downwind.normal_load.thrust_n / stationary.normal_load.thrust_n,
            (8.0 / 10.0) ** 2,
        )

    def test_generalized_rotor_input_advances_small_angle_model(self):
        rotor_input = quasi_steady_rotor_generalized_load_from_platform_motion(
            air_density_kg_m3=1.225,
            rotor_radius_m=10.0,
            thrust_coefficient=1.0,
            ambient_air_velocity_platform_mps=[2.0, 0.0, 0.0],
            platform_reference_velocity_platform_mps=[0.0, 0.0, 0.0],
            platform_angular_velocity_platform_radps=[0.0, 0.0, 0.0],
            rotor_force_application_point_from_reference_m=[0.0, 0.0, 5.0],
            downwind_rotor_normal_platform=[1.0, 0.0, 0.0],
        )
        model = IncrementalPlatformModel(
            PlatformMatrices(
                mass=np.diag([10.0, 10.0, 10.0, 20.0, 20.0, 20.0]),
                damping=np.zeros((6, 6)),
                hydrostatic_stiffness=np.zeros((6, 6)),
                mooring_stiffness=np.zeros((6, 6)),
            )
        )
        loads = IncrementalLoads(
            wind=rotor_input.generalized_load_platform,
            wave=np.zeros(6),
            ballast=np.zeros(6),
            other=np.zeros(6),
        )

        advanced = model.advance_frozen_step(
            IncrementalState.zeros(),
            loads,
            duration_s=1.0,
        )

        self.assertGreater(advanced.position[0], 0.0)
        self.assertGreater(advanced.position[4], 0.0)


if __name__ == "__main__":
    unittest.main()
