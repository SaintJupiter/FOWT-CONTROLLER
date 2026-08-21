import unittest

import numpy as np

from fowt_platform import sample_ballast_endpoint_path


CAPACITIES_KG = np.full(3, 1_000.0)


class BallastEndpointPathTests(unittest.TestCase):
    def test_zero_and_full_fractions_recover_actual_and_diagnostic_endpoint(self):
        actual = np.array([500.0, 500.0, 500.0])
        endpoint = np.array([400.0, 550.0, 550.0])

        samples = sample_ballast_endpoint_path(
            actual_tank_masses_kg=actual,
            diagnostic_target_tank_masses_kg=endpoint,
            tank_capacities_kg=CAPACITIES_KG,
            fractions=(0.0, 0.25, 0.5, 1.0),
        )

        self.assertEqual(tuple(sample.fraction for sample in samples), (0.0, 0.25, 0.5, 1.0))
        np.testing.assert_allclose(samples[0].hypothetical_tank_masses_kg, actual)
        np.testing.assert_allclose(samples[1].mass_delta_from_actual_kg, [-25.0, 12.5, 12.5])
        np.testing.assert_allclose(samples[2].hypothetical_tank_masses_kg, [450.0, 525.0, 525.0])
        np.testing.assert_allclose(samples[-1].hypothetical_tank_masses_kg, endpoint)

    def test_every_sample_preserves_mass_relative_to_the_current_actual_state(self):
        # The current total need not equal any external reference ballast total.
        actual = np.array([100.0, 300.0, 700.0])
        endpoint = np.array([80.0, 340.0, 680.0])

        samples = sample_ballast_endpoint_path(
            actual_tank_masses_kg=actual,
            diagnostic_target_tank_masses_kg=endpoint,
            tank_capacities_kg=CAPACITIES_KG,
            fractions=(0.1, 0.6, 1.0),
        )

        for sample in samples:
            with self.subTest(fraction=sample.fraction):
                self.assertAlmostEqual(sample.net_mass_delta_from_actual_kg, 0.0)
                self.assertAlmostEqual(float(np.sum(sample.mass_delta_from_actual_kg)), 0.0)
                self.assertAlmostEqual(
                    float(np.sum(sample.hypothetical_tank_masses_kg)),
                    float(np.sum(actual)),
                )
                self.assertTrue(sample.within_capacity)

    def test_invalid_endpoint_or_fraction_is_rejected_without_clipping(self):
        with self.assertRaisesRegex(ValueError, "preserve total ballast mass"):
            sample_ballast_endpoint_path(
                actual_tank_masses_kg=[500.0, 500.0, 500.0],
                diagnostic_target_tank_masses_kg=[450.0, 550.0, 550.1],
                tank_capacities_kg=CAPACITIES_KG,
                fractions=(0.5,),
            )
        with self.assertRaisesRegex(ValueError, "within tank capacities"):
            sample_ballast_endpoint_path(
                actual_tank_masses_kg=[500.0, 500.0, 500.0],
                diagnostic_target_tank_masses_kg=[-1.0, 750.0, 751.0],
                tank_capacities_kg=CAPACITIES_KG,
                fractions=(0.5,),
            )
        with self.assertRaisesRegex(ValueError, "values in \\[0, 1\\]"):
            sample_ballast_endpoint_path(
                actual_tank_masses_kg=[500.0, 500.0, 500.0],
                diagnostic_target_tank_masses_kg=[400.0, 550.0, 550.0],
                tank_capacities_kg=CAPACITIES_KG,
                fractions=(-0.1,),
            )


if __name__ == "__main__":
    unittest.main()
