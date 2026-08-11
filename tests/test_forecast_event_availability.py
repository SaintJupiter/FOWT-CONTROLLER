import unittest

from src.wind_prediction.ballast_planner_provider import (
    complete_event_probabilities_available,
)


class ForecastEventAvailabilityTests(unittest.TestCase):
    def test_requires_every_expected_event_probability(self):
        expected = ("near", "mid", "far")
        self.assertFalse(
            complete_event_probabilities_available(
                {"near": 0.1, "mid": 0.2},
                expected,
            )
        )

    def test_accepts_complete_finite_probabilities(self):
        expected = ("near", "mid", "far")
        self.assertTrue(
            complete_event_probabilities_available(
                {"near": 0.1, "mid": 0.2, "far": 0.3},
                expected,
            )
        )

    def test_rejects_non_finite_or_out_of_range_values(self):
        expected = ("near", "mid", "far")
        self.assertFalse(
            complete_event_probabilities_available(
                {"near": 0.1, "mid": float("nan"), "far": 0.3},
                expected,
            )
        )
        self.assertFalse(
            complete_event_probabilities_available(
                {"near": 0.1, "mid": 0.2, "far": 1.1},
                expected,
            )
        )


if __name__ == "__main__":
    unittest.main()
