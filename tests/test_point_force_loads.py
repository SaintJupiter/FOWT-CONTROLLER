import unittest

import numpy as np

from fowt_platform import generalized_load_from_point_force


class PointForceLoadTests(unittest.TestCase):
    def test_force_and_moment_follow_the_cross_product_about_reference(self):
        generalized = generalized_load_from_point_force(
            force_platform_n=[12.0, -5.0, 3.0],
            application_point_from_reference_m=[4.0, -2.0, 10.0],
        )

        np.testing.assert_allclose(generalized[:3], [12.0, -5.0, 3.0])
        np.testing.assert_allclose(
            generalized[3:],
            np.cross([4.0, -2.0, 10.0], [12.0, -5.0, 3.0]),
        )

    def test_cardinal_horizontal_forces_have_expected_pitch_and_roll_signs(self):
        height_m = 150.0
        force_n = 1.0e6

        forward = generalized_load_from_point_force(
            [force_n, 0.0, 0.0], [0.0, 0.0, height_m]
        )
        port = generalized_load_from_point_force(
            [0.0, force_n, 0.0], [0.0, 0.0, height_m]
        )

        self.assertGreater(forward[4], 0.0)
        self.assertAlmostEqual(forward[3], 0.0)
        self.assertLess(port[3], 0.0)
        self.assertAlmostEqual(port[4], 0.0)

    def test_force_applied_at_reference_has_no_moment(self):
        generalized = generalized_load_from_point_force(
            [7.0, -11.0, 13.0], [0.0, 0.0, 0.0]
        )

        np.testing.assert_allclose(generalized[:3], [7.0, -11.0, 13.0])
        np.testing.assert_array_equal(generalized[3:], np.zeros(3))

    def test_wrong_shape_or_nonfinite_input_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "force_platform_n must have shape"):
            generalized_load_from_point_force([1.0, 2.0], [0.0, 0.0, 1.0])
        with self.assertRaisesRegex(ValueError, "application_point_from_reference_m"):
            generalized_load_from_point_force([1.0, 2.0, 3.0], [0.0, np.nan, 1.0])


if __name__ == "__main__":
    unittest.main()
