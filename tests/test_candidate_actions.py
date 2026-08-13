import unittest

import numpy as np

from wind_prediction.ballast_planner import (
    ACTIONS,
    STOP_EXECUTION_ACTION,
    PlannerConfig,
    action_vec,
    tank_signal,
)


class CandidateActionTests(unittest.TestCase):
    def setUp(self):
        self.cfg = PlannerConfig()
        self.pressure = np.array([2.0, 1.6], dtype=float)

    def test_canonical_action_set_is_stable(self):
        self.assertEqual(
            ACTIONS,
            (
                "hold",
                "pump_saving",
                "active_small",
                "active_medium",
                "active_reverse_small",
            ),
        )

    def test_forward_actions_preserve_direction_and_strength_order(self):
        saving = action_vec("pump_saving", self.pressure, self.cfg)
        small = action_vec("active_small", self.pressure, self.cfg)
        medium = action_vec("active_medium", self.pressure, self.cfg)

        self.assertGreater(float(np.dot(saving, self.pressure)), 0.0)
        self.assertGreater(float(np.dot(small, self.pressure)), 0.0)
        self.assertGreater(float(np.dot(medium, self.pressure)), 0.0)
        self.assertLess(np.linalg.norm(saving), np.linalg.norm(small))
        self.assertLess(np.linalg.norm(small), np.linalg.norm(medium))

    def test_hold_is_zero_and_reverse_opposes_forward_direction(self):
        hold = action_vec("hold", self.pressure, self.cfg)
        stop = action_vec(STOP_EXECUTION_ACTION, self.pressure, self.cfg)
        small = action_vec("active_small", self.pressure, self.cfg)
        reverse = action_vec("active_reverse_small", self.pressure, self.cfg)

        np.testing.assert_allclose(hold, np.zeros(2))
        np.testing.assert_allclose(stop, np.zeros(2))
        self.assertLess(float(np.dot(reverse, small)), 0.0)

    def test_three_tank_signal_has_zero_net_mass_request(self):
        for name in ACTIONS:
            avec = action_vec(name, self.pressure, self.cfg)
            signal = tank_signal(avec, self.cfg)
            self.assertAlmostEqual(float(np.sum(signal)), 0.0, places=12)


if __name__ == "__main__":
    unittest.main()
