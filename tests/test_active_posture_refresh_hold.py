import unittest

import numpy as np

from src.wind_prediction.ballast_planner_provider import BallastPlannerPreviewProvider


class ActivePostureRefreshHoldTests(unittest.TestCase):
    @staticmethod
    def _provider(include_hold):
        provider = BallastPlannerPreviewProvider.__new__(BallastPlannerPreviewProvider)
        provider.prediction_primary_enabled = True
        provider._target_action = "hold"
        provider.economy_pump_budget_allocator_mode = "budget"
        provider._primary_target_reused = False
        provider.active_posture_refresh_include_hold = include_hold
        provider.active_posture_refresh_update_interval_s = 300.0
        provider._active_posture_refresh_last_update_s = None
        provider._active_posture_refresh_count = 0
        provider._primary_refresh_owner_pending = ""
        provider._active_posture_refresh_correction = lambda *_: (
            True,
            np.array([1.0, 0.0]),
        )
        provider._fallback_risk_active_release_needed = lambda *_: (False, "free")
        provider._economy_budget_observe = lambda *_args, **_kwargs: None
        provider._economy_budget_allows_refresh = lambda **_kwargs: True
        provider.updated = False
        provider._update_primary_target = lambda *_args, **_kwargs: setattr(
            provider,
            "updated",
            True,
        )
        return provider

    def test_hold_candidate_is_unchanged_by_default(self):
        provider = self._provider(include_hold=False)

        provider._maybe_update_primary_target_between_buckets({}, 600.0)

        self.assertFalse(provider.updated)

    def test_hold_candidate_can_receive_bounded_posture_refresh(self):
        provider = self._provider(include_hold=True)

        provider._maybe_update_primary_target_between_buckets({}, 600.0)

        self.assertTrue(provider.updated)
        self.assertEqual(
            provider._primary_refresh_owner_pending,
            "between_bucket_active_posture",
        )


if __name__ == "__main__":
    unittest.main()
