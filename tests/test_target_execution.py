import sys
import unittest
from pathlib import Path

import numpy as np


LEGACY_DIR = Path(__file__).resolve().parents[1] / "archive" / "legacy_fowt_control"
sys.path.insert(0, str(LEGACY_DIR))

from target_execution import TargetSlewLimiter, resolve_target_slew_config


class TargetExecutionTests(unittest.TestCase):
    def test_target_slew_limiter_shapes_target_without_plant_state(self):
        limiter = TargetSlewLimiter(
            dt=1.0,
            rho=1000.0,
            max_capacity=10000.0,
            target_slew_rate_m3_min=6.0,
        )
        limiter.reset([1000.0, 1000.0, 1000.0])

        target, gap = limiter.update([2000.0, 0.0, 1000.0])

        np.testing.assert_allclose(target, [1100.0, 900.0, 1000.0])
        self.assertGreater(gap, 0.0)
        self.assertFalse(hasattr(limiter, "pump_active"))
        self.assertFalse(hasattr(limiter, "tank_masses"))

    def test_legacy_target_slew_keys_are_normalized(self):
        resolved = resolve_target_slew_config(
            {
                "actuator_authority": "plant",
                "enable_rate_limit": True,
                "rate_limit_m3_min": 7.5,
            }
        )

        self.assertEqual(resolved["actuator_authority"], "plant")
        self.assertTrue(resolved["enabled"])
        self.assertEqual(resolved["rate_m3_min"], 7.5)

    def test_canonical_target_slew_keys_take_precedence(self):
        resolved = resolve_target_slew_config(
            {
                "enable_target_slew_limit": False,
                "target_slew_rate_m3_min": 4.0,
                "enable_rate_limit": True,
                "rate_limit_m3_min": 9.0,
            }
        )

        self.assertFalse(resolved["enabled"])
        self.assertEqual(resolved["rate_m3_min"], 4.0)

    def test_non_plant_actuator_authority_fails_fast(self):
        with self.assertRaisesRegex(ValueError, "plant-owned pump execution"):
            resolve_target_slew_config({"actuator_authority": "policy"})


if __name__ == "__main__":
    unittest.main()
