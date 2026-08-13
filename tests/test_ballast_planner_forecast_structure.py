import unittest

import numpy as np

from src.wind_prediction.ballast_planner import (
    PlannerConfig,
    compute_pressure_blocks,
    pressure_proxy_vec,
)


class ForecastStructurePlannerTests(unittest.TestCase):
    def setUp(self):
        self.uv = np.asarray(
            [
                [4.0, 0.0],
                [16.0, 0.0],
                [8.0, 0.0],
                [-8.0, 0.0],
                [10.0, 0.0],
                [10.0, 0.0],
            ],
            dtype=float,
        )
        self.discounts = [1.0, 0.85, 0.7]

    def test_legacy_mode_preserves_vector_first_aggregation(self):
        cfg = PlannerConfig()

        blocks = compute_pressure_blocks(self.uv, self.discounts, cfg)

        expected = pressure_proxy_vec(self.uv[:2], cfg)
        np.testing.assert_allclose(blocks[0]["pressure_vec_raw"], expected)
        self.assertEqual(blocks[0]["lead_reliability_scale"], 1.0)

    def test_mean_step_force_retains_peak_before_aggregation(self):
        cfg = PlannerConfig(pressure_aggregation_mode="mean_step_force")

        blocks = compute_pressure_blocks(self.uv, self.discounts, cfg)

        legacy = pressure_proxy_vec(self.uv[:2], PlannerConfig())
        self.assertGreater(
            np.linalg.norm(blocks[0]["pressure_vec_raw"]),
            np.linalg.norm(legacy),
        )
        self.assertGreater(blocks[0]["peak_pressure_norm"], blocks[0]["pressure_norm"])

    def test_reliability_attenuates_only_when_enabled(self):
        reliability = np.asarray([1.0, 0.8, 0.7, 0.6, 0.5, 0.4], dtype=float)
        disabled = compute_pressure_blocks(
            self.uv,
            self.discounts,
            PlannerConfig(pressure_aggregation_mode="mean_step_force"),
            lead_reliability=reliability,
        )
        enabled = compute_pressure_blocks(
            self.uv,
            self.discounts,
            PlannerConfig(
                pressure_aggregation_mode="mean_step_force",
                lead_reliability_enabled=True,
            ),
            lead_reliability=reliability,
        )

        self.assertLess(enabled[2]["pressure_norm"], disabled[2]["pressure_norm"])
        self.assertAlmostEqual(enabled[2]["lead_reliability_scale"], 0.45)

    def test_within_block_reversal_and_event_probability_are_retained(self):
        blocks = compute_pressure_blocks(
            self.uv,
            self.discounts,
            PlannerConfig(pressure_aggregation_mode="mean_step_force"),
            event_probs={"attention_event_20_40m": 0.73},
        )

        self.assertEqual(blocks[1]["within_block_reversal"], 1)
        self.assertLess(blocks[1]["direction_consistency"], 0.1)
        self.assertAlmostEqual(blocks[1]["event_probability"], 0.73)

    def test_invalid_reliability_shape_is_rejected(self):
        with self.assertRaises(ValueError):
            compute_pressure_blocks(
                self.uv,
                self.discounts,
                PlannerConfig(),
                lead_reliability=np.ones(5),
            )


if __name__ == "__main__":
    unittest.main()
