import unittest

import numpy as np

from wind_prediction.ballast_allocation import (
    DEFAULT_COMPENSATION_ALLOCATION,
    compensation_to_mass_delta_kg,
    mass_delta_to_compensation_vec,
)
from wind_prediction.ballast_planner import (
    COMPENSATION_ATTITUDE_ALLOC,
    PlannerConfig,
    compensation_vec_from_mass_delta,
    tank_signal,
)


class BallastAllocationTests(unittest.TestCase):
    def setUp(self):
        self.deadband = np.array([1.0, 0.8], dtype=float)
        self.mass_quantum = 180000.0

    def forward(self, compensation):
        return compensation_to_mass_delta_kg(
            compensation,
            deadband_deg=self.deadband,
            action_mass_quantum_kg=self.mass_quantum,
        )

    def inverse(self, mass_delta):
        return mass_delta_to_compensation_vec(
            mass_delta,
            deadband_deg=self.deadband,
            action_mass_quantum_kg=self.mass_quantum,
        )

    def test_default_mapping_matches_existing_planner_functions(self):
        compensation = np.array([0.35, -0.16], dtype=float)
        cfg = PlannerConfig(
            deadband_pitch_deg=self.deadband[0],
            deadband_roll_deg=self.deadband[1],
            action_mass_quantum_kg=self.mass_quantum,
        )

        np.testing.assert_array_equal(
            DEFAULT_COMPENSATION_ALLOCATION,
            COMPENSATION_ATTITUDE_ALLOC,
        )
        mass_delta = self.forward(compensation)
        np.testing.assert_allclose(
            mass_delta,
            tank_signal(compensation, cfg) * self.mass_quantum,
            rtol=0.0,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            self.inverse(mass_delta),
            compensation_vec_from_mass_delta(mass_delta, cfg),
            rtol=0.0,
            atol=1e-12,
        )

    def test_compensation_round_trip(self):
        for compensation in (
            np.array([0.0, 0.0]),
            np.array([0.7, -0.3]),
            np.array([-1.2, 0.9]),
        ):
            with self.subTest(compensation=compensation):
                np.testing.assert_allclose(
                    self.inverse(self.forward(compensation)),
                    compensation,
                    rtol=0.0,
                    atol=1e-12,
                )

    def test_common_mode_mass_delta_has_no_attitude_compensation(self):
        compensation = self.inverse(np.array([1250.0, 1250.0, 1250.0]))

        np.testing.assert_allclose(compensation, np.zeros(2), atol=1e-12)

    def test_non_equal_deadband_scales_are_applied_per_axis(self):
        deadband = np.array([1.6, 0.45], dtype=float)
        compensation = np.array([0.8, -0.18], dtype=float)
        expected_signal = DEFAULT_COMPENSATION_ALLOCATION @ np.array([0.5, -0.4])

        mass_delta = compensation_to_mass_delta_kg(
            compensation,
            deadband_deg=deadband,
            action_mass_quantum_kg=50000.0,
        )

        np.testing.assert_allclose(mass_delta, expected_signal * 50000.0)
        np.testing.assert_allclose(
            mass_delta_to_compensation_vec(
                mass_delta,
                deadband_deg=deadband,
                action_mass_quantum_kg=50000.0,
            ),
            compensation,
            atol=1e-12,
        )

    def test_invalid_vector_inputs_are_rejected(self):
        invalid_compensations = (
            [1.0],
            [1.0, 2.0, 3.0],
            [[1.0], [2.0]],
            [np.nan, 0.0],
            [0.0, np.inf],
        )
        for compensation in invalid_compensations:
            with self.subTest(compensation=compensation):
                with self.assertRaises(ValueError):
                    self.forward(compensation)

        invalid_mass_deltas = (
            [1.0, 2.0],
            [1.0, 2.0, 3.0, 4.0],
            [[1.0], [2.0], [3.0]],
            [0.0, np.nan, 0.0],
            [0.0, 0.0, -np.inf],
        )
        for mass_delta in invalid_mass_deltas:
            with self.subTest(mass_delta=mass_delta):
                with self.assertRaises(ValueError):
                    self.inverse(mass_delta)

    def test_invalid_scales_are_rejected(self):
        invalid_deadbands = (
            [1.0],
            [1.0, 0.8, 0.6],
            [[1.0, 0.8]],
            [0.0, 0.8],
            [1.0, -0.8],
            [1.0, np.inf],
        )
        for deadband in invalid_deadbands:
            with self.subTest(deadband=deadband):
                with self.assertRaises(ValueError):
                    compensation_to_mass_delta_kg(
                        [0.2, -0.1],
                        deadband_deg=deadband,
                        action_mass_quantum_kg=self.mass_quantum,
                    )

        for mass_quantum in (0.0, -1.0, np.nan, np.inf, [180000.0]):
            with self.subTest(mass_quantum=mass_quantum):
                with self.assertRaises(ValueError):
                    compensation_to_mass_delta_kg(
                        [0.2, -0.1],
                        deadband_deg=self.deadband,
                        action_mass_quantum_kg=mass_quantum,
                    )

    def test_invalid_allocation_matrices_are_rejected(self):
        invalid_matrices = (
            np.ones((2, 3)),
            np.array([[-1.0, 0.0], [0.5, np.nan], [0.5, -1.0]]),
            np.array([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]]),
        )
        for matrix in invalid_matrices:
            with self.subTest(matrix=matrix):
                with self.assertRaises(ValueError):
                    compensation_to_mass_delta_kg(
                        [0.2, -0.1],
                        deadband_deg=self.deadband,
                        action_mass_quantum_kg=self.mass_quantum,
                        allocation_matrix=matrix,
                    )


if __name__ == "__main__":
    unittest.main()
