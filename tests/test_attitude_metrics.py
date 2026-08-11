import unittest

import numpy as np

from src.wind_prediction.attitude_metrics import (
    ATTITUDE_EXPOSURE_METRIC_KEYS,
    compute_attitude_exposure_metrics,
)


class AttitudeMetricThresholdTests(unittest.TestCase):
    def test_fine_and_moderate_exposure_are_reported(self):
        pitch = np.array([0.0, 1.6, 2.1, 2.1])
        roll = np.zeros_like(pitch)

        metrics = compute_attitude_exposure_metrics(
            pitch,
            roll,
            time_s=np.arange(pitch.size, dtype=float),
        )

        self.assertIn("time_over_1p5deg_s", ATTITUDE_EXPOSURE_METRIC_KEYS)
        self.assertIn("time_over_2deg_s", ATTITUDE_EXPOSURE_METRIC_KEYS)
        self.assertEqual(metrics["time_over_1p5deg_s"], 3.0)
        self.assertEqual(metrics["time_over_2deg_s"], 2.0)
        self.assertAlmostEqual(metrics["area_over_1p5deg_deg_s"], 1.3)
        self.assertAlmostEqual(metrics["area_over_2deg_deg_s"], 0.2)


if __name__ == "__main__":
    unittest.main()
