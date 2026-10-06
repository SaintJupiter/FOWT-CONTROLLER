import unittest

import numpy as np

from wind_prediction.physical_target_relation import (
    compare_tank_target_to_physical_endpoint,
)


_ACTUAL = np.array([500.0, 500.0, 500.0])
_CAPACITIES = np.full(3, 1_000.0)
_COORDINATES = np.array(
    [
        [-20.0, 0.0, -1.0],
        [10.0, 17.0, -1.0],
        [10.0, -17.0, -1.0],
    ]
)


class PhysicalTargetRelationTests(unittest.TestCase):
    def compare(self, *, target, endpoint):
        return compare_tank_target_to_physical_endpoint(
            actual_tank_masses_kg=_ACTUAL,
            proposed_target_tank_masses_kg=target,
            physical_endpoint_tank_masses_kg=endpoint,
            tank_capacities_kg=_CAPACITIES,
            tank_coordinates_m=_COORDINATES,
            gravity_m_s2=9.81,
        )

    def test_same_redistribution_direction_has_positive_cosine(self):
        relation = self.compare(
            target=[560.0, 470.0, 470.0],
            endpoint=[620.0, 440.0, 440.0],
        )

        self.assertAlmostEqual(relation.proposed_target_net_mass_delta_kg, 0.0)
        self.assertAlmostEqual(relation.physical_endpoint_net_mass_delta_kg, 0.0)
        self.assertGreater(relation.moment_direction_cosine, 0.99)
        self.assertTrue(relation.same_moment_half_plane)
        self.assertFalse(relation.opposite_moment_half_plane)

    def test_opposite_redistribution_direction_is_explicit(self):
        relation = self.compare(
            target=[560.0, 470.0, 470.0],
            endpoint=[440.0, 530.0, 530.0],
        )

        self.assertLess(relation.moment_direction_cosine, -0.99)
        self.assertFalse(relation.same_moment_half_plane)
        self.assertTrue(relation.opposite_moment_half_plane)

    def test_common_mode_target_is_separated_before_moment_comparison(self):
        relation = self.compare(
            target=[660.0, 570.0, 570.0],
            endpoint=[620.0, 440.0, 440.0],
        )

        self.assertAlmostEqual(relation.proposed_target_net_mass_delta_kg, 300.0)
        np.testing.assert_allclose(
            relation.proposed_target_redistribution_delta_kg,
            [60.0, -30.0, -30.0],
        )
        self.assertGreater(relation.moment_direction_cosine, 0.99)

    def test_zero_or_nonredistributing_endpoint_has_indeterminate_direction(self):
        relation = self.compare(
            target=[560.0, 470.0, 470.0],
            endpoint=_ACTUAL,
        )

        self.assertIsNone(relation.moment_direction_cosine)
        self.assertIsNone(relation.same_moment_half_plane)
        self.assertIsNone(relation.opposite_moment_half_plane)

    def test_nonzero_net_physical_endpoint_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "preserve total ballast mass"):
            self.compare(
                target=[560.0, 470.0, 470.0],
                endpoint=[620.0, 450.0, 450.0],
            )

    def test_off_center_tank_geometry_rejects_redistribution_only_comparison(self):
        with self.assertRaisesRegex(ValueError, "common-mode water exchange"):
            compare_tank_target_to_physical_endpoint(
                actual_tank_masses_kg=_ACTUAL,
                proposed_target_tank_masses_kg=[560.0, 470.0, 470.0],
                physical_endpoint_tank_masses_kg=[620.0, 440.0, 440.0],
                tank_capacities_kg=_CAPACITIES,
                tank_coordinates_m=_COORDINATES + [1.0, 0.0, 0.0],
                gravity_m_s2=9.81,
            )


if __name__ == "__main__":
    unittest.main()
