import unittest

import numpy as np

from fowt_platform import (
    generalized_load_from_openfast_hub_wrench,
    openfast_hub_to_reference_rotation,
)


class OpenFastHubLoadTests(unittest.TestCase):
    def test_zero_tilt_and_openfast_zero_azimuth_align_hub_and_reference_axes(self):
        rotation = openfast_hub_to_reference_rotation(
            shaft_tilt_deg=0.0,
            azimuth_deg=0.0,
        )

        np.testing.assert_allclose(rotation, np.eye(3), atol=1e-12)

    def test_rotor_azimuth_rotates_transverse_hub_axes_before_statistics(self):
        rotation = openfast_hub_to_reference_rotation(
            shaft_tilt_deg=0.0,
            azimuth_deg=90.0,
        )

        np.testing.assert_allclose(
            rotation,
            np.array(
                [
                    [1.0, 0.0, 0.0],
                    [0.0, 0.0, -1.0],
                    [0.0, 1.0, 0.0],
                ]
            ),
            atol=1e-12,
        )

    def test_azimuth_blade1_up_offset_is_applied_before_transverse_rotation(self):
        rotation = openfast_hub_to_reference_rotation(
            shaft_tilt_deg=0.0,
            azimuth_deg=37.0,
            azimuth_blade1_up_deg=37.0,
        )

        np.testing.assert_allclose(rotation, np.eye(3), atol=1e-12)

    def test_shaft_tilt_rotates_hub_axial_force_in_the_frozen_reference_axes(self):
        load = generalized_load_from_openfast_hub_wrench(
            force_hub_n=[10.0, 0.0, 0.0],
            moment_hub_nm=[0.0, 0.0, 0.0],
            hub_from_platform_reference_m=[0.0, 0.0, 0.0],
            shaft_tilt_deg=-6.0,
            azimuth_deg=0.0,
        )

        np.testing.assert_allclose(
            load[:3],
            [10.0 * np.cos(np.deg2rad(6.0)), 0.0, -10.0 * np.sin(np.deg2rad(6.0))],
            atol=1e-12,
        )

    def test_equilibrium_pitch_is_removed_from_the_frozen_reference_frame(self):
        rotation = openfast_hub_to_reference_rotation(
            shaft_tilt_deg=0.0,
            azimuth_deg=0.0,
            platform_pitch_deg=-1.452,
            equilibrium_platform_pitch_deg=-1.452,
        )

        np.testing.assert_allclose(rotation, np.eye(3), atol=1e-12)

    def test_hub_wrench_is_rotated_and_shifted_to_the_platform_reference(self):
        load = generalized_load_from_openfast_hub_wrench(
            force_hub_n=[10.0, 0.0, 0.0],
            moment_hub_nm=[1.0, 2.0, 3.0],
            hub_from_platform_reference_m=[0.0, 0.0, 150.0],
            shaft_tilt_deg=0.0,
            azimuth_deg=0.0,
        )

        np.testing.assert_allclose(load[:3], [10.0, 0.0, 0.0])
        np.testing.assert_allclose(load[3:], [1.0, 1502.0, 3.0])

    def test_platform_yaw_changes_the_reference_load_direction(self):
        load = generalized_load_from_openfast_hub_wrench(
            force_hub_n=[10.0, 0.0, 0.0],
            moment_hub_nm=[0.0, 0.0, 0.0],
            hub_from_platform_reference_m=[0.0, 0.0, 0.0],
            shaft_tilt_deg=0.0,
            azimuth_deg=0.0,
            platform_yaw_deg=90.0,
        )

        np.testing.assert_allclose(
            load,
            [0.0, 10.0, 0.0, 0.0, 0.0, 0.0],
            atol=1e-12,
        )

    def test_nonzero_nacelle_yaw_is_rejected_for_the_narrow_audit(self):
        with self.assertRaisesRegex(ValueError, "nacelle_yaw_deg must be zero"):
            openfast_hub_to_reference_rotation(
                shaft_tilt_deg=0.0,
                azimuth_deg=0.0,
                nacelle_yaw_deg=1.0,
            )

    def test_nonfinite_hub_load_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "force_hub_n must contain"):
            generalized_load_from_openfast_hub_wrench(
                force_hub_n=[1.0, np.nan, 0.0],
                moment_hub_nm=[0.0, 0.0, 0.0],
                hub_from_platform_reference_m=[0.0, 0.0, 0.0],
                shaft_tilt_deg=0.0,
                azimuth_deg=0.0,
            )


if __name__ == "__main__":
    unittest.main()
