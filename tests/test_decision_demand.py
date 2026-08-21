import unittest

import numpy as np

from wind_prediction.decision_demand import (
    LegacyWindDemandProxyConfig,
    form_stage_demand,
    legacy_wind_proxy_demand_deg,
    posture_feedback_demand_deg,
)


class DecisionDemandTests(unittest.TestCase):
    def setUp(self):
        self.proxy = LegacyWindDemandProxyConfig(
            demand_axis_scale_deg=(1.0, 0.8),
            reference_speed_ms=10.0,
            effect_cap=2.0,
            sign_multiplier=-1.0,
        )

    def test_feedback_deadband_is_separate_from_forecast_proxy_function(self):
        feedback = posture_feedback_demand_deg(
            (2.0, -1.3),
            deadband_deg=(1.0, 0.8),
        )

        np.testing.assert_allclose(feedback, (1.0, -0.5))

    def test_legacy_proxy_preserves_wind_difference_relation(self):
        current = legacy_wind_proxy_demand_deg(
            [(0.0, -10.0)],
            config=self.proxy,
        )
        future = legacy_wind_proxy_demand_deg(
            [(0.0, -20.0)],
            config=self.proxy,
        )

        np.testing.assert_allclose(current, (1.0, 0.0), atol=1e-12)
        np.testing.assert_allclose(future, (2.0, 0.0), atol=1e-12)

    def test_stage_records_direction_information_without_gating_forecast(self):
        stage = form_stage_demand(
            feedback_demand_deg=(0.0, 0.0),
            current_wind_uv_ms=(0.0, -10.0),
            future_wind_uv_ms=[(0.0, -12.0), (0.0, 12.0)],
            forecast_available=True,
            stage_discount=1.0,
            proxy_config=self.proxy,
            forecast_reason="forecast_direction_inconsistent_recorded",
            direction_consistency=0.1,
            direction_consistent=False,
        )

        self.assertFalse(stage.direction_consistent)
        self.assertEqual(
            stage.forecast_reason,
            "forecast_direction_inconsistent_recorded",
        )
        self.assertGreater(abs(stage.forecast_increment_deg[0]), 0.0)

    def test_forecast_gain_above_one_remains_accepted_by_demand_formation(self):
        stage = form_stage_demand(
            feedback_demand_deg=(0.0, 0.0),
            current_wind_uv_ms=(0.0, -10.0),
            future_wind_uv_ms=[(0.0, -12.0)],
            forecast_available=True,
            stage_discount=1.2,
            proxy_config=self.proxy,
            forecast_reason="forecast_direction_consistent",
            direction_consistency=1.0,
            direction_consistent=True,
        )

        self.assertGreater(abs(stage.forecast_increment_deg[0]), 0.0)


if __name__ == "__main__":
    unittest.main()
