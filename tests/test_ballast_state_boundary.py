import unittest

import numpy as np

from fowt_platform import tank_mass_deltas_from_actual_masses


class BallastStateBoundaryTests(unittest.TestCase):
    def test_actual_masses_are_converted_relative_to_the_reference_state(self):
        deltas = tank_mass_deltas_from_actual_masses(
            actual_tank_masses_kg=[110.0, 80.0, 120.0],
            reference_tank_masses_kg=[100.0, 100.0, 100.0],
            tank_capacities_kg=[200.0, 200.0, 200.0],
        )

        np.testing.assert_array_equal(deltas, [10.0, -20.0, 20.0])

    def test_reference_state_has_zero_mass_delta(self):
        reference = np.array([100.0, 120.0, 140.0])
        deltas = tank_mass_deltas_from_actual_masses(
            actual_tank_masses_kg=reference,
            reference_tank_masses_kg=reference,
            tank_capacities_kg=[200.0, 200.0, 200.0],
        )

        np.testing.assert_array_equal(deltas, np.zeros(3))

    def test_actual_and_reference_masses_must_respect_each_tank_capacity(self):
        invalid_pairs = (
            ([-1.0, 100.0, 100.0], [100.0, 100.0, 100.0]),
            ([201.0, 100.0, 100.0], [100.0, 100.0, 100.0]),
            ([100.0, 100.0, 100.0], [201.0, 100.0, 100.0]),
        )
        for actual, reference in invalid_pairs:
            with self.subTest(actual=actual, reference=reference):
                with self.assertRaisesRegex(ValueError, "tank capacity"):
                    tank_mass_deltas_from_actual_masses(
                        actual_tank_masses_kg=actual,
                        reference_tank_masses_kg=reference,
                        tank_capacities_kg=[200.0, 200.0, 200.0],
                    )


if __name__ == "__main__":
    unittest.main()
