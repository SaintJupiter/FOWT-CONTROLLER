import unittest
from unittest.mock import patch

import numpy as np

from wind_prediction.ballast_planner import PlannerConfig
from wind_prediction.forecast_action_policy import (
    ForecastActionPolicyConfig,
    evaluate_forecast_action_policy,
)
from wind_prediction.forecast_evidence import ForecastEvidence
from wind_prediction.provider_planning import (
    ProviderPlanningMixin,
    _current_wind_uv_ms,
    enforce_forecast_policy_on_first_actions,
)


class _ReplayDataset:
    event_columns = ()


class CurrentWindVectorTests(unittest.TestCase):
    def test_observation_uses_forecast_uv_convention(self):
        np.testing.assert_allclose(
            _current_wind_uv_ms({"ws": 10.0, "wd_deg": 90.0}),
            [-10.0, 0.0],
            atol=1e-12,
        )
        self.assertIsNone(_current_wind_uv_ms(None))
        self.assertIsNone(_current_wind_uv_ms({"ws": -1.0, "wd_deg": 0.0}))


class _PlanningProbe(ProviderPlanningMixin):
    def __init__(self, mode: str) -> None:
        self.forecast_action_policy_mode = mode
        self.forecast_action_policy_config = ForecastActionPolicyConfig(
            enabled=mode != "off",
        )
        self.replay_dataset = _ReplayDataset()
        self.cfg = PlannerConfig(
            candidate_action_mode="explicit_target_lifecycle",
        )
        self.block_discounts = [1.0, 1.0, 1.0]
        self.no_preview_myopic_horizon_enabled = False
        self.objective_mode = "economic"
        self.prediction_primary_scale = 1.0
        self.primary_hold_target_mode = "current"
        self._primary_target_kg = np.zeros(3, dtype=float)

    def _forecast_uv(self, sample):
        del sample
        return ForecastEvidence(
            source="unit_test",
            model_version="unit_test",
            origin_time=None,
            sample_period_s=600.0,
            uv_ms=np.asarray(
                [
                    [6.0, 0.0],
                    [7.0, 0.0],
                    [8.0, 0.0],
                    [9.0, 0.0],
                    [10.0, 0.0],
                    [11.0, 0.0],
                ],
                dtype=float,
            ),
            event_probs={
                "attention_event_0_20m": 0.9,
                "attention_event_20_40m": 0.9,
                "attention_event_40_60m": 0.9,
            },
            lead_reliability=np.ones(6, dtype=float),
        )

    def _observe_current_wind_stability(self, sample):
        del sample

    def _observe_forecast_pressure_current_support(self, wind_obs):
        del wind_obs

    def _observe_forecast_speed_shape(self, uv):
        del uv

    def _observe_psc_v2_short_shape(self, uv):
        del uv

    def _observe_far_horizon(self, uv):
        del uv

    def _reset_forecast_speed_shape_state(self, reason):
        del reason

    def _reset_psc_v2_short_state(self, reason):
        del reason

    def _reset_far_horizon_state(self, reason):
        del reason

    def _reset_h120_oracle_probe_bucket_state(self, reason):
        del reason

    def _hold_relief_debt_observe_before_planner(self, plant_info, norms):
        del norms
        return plant_info

    def _update_relief_decay_auto_v2_latch(self, plant_info):
        del plant_info

    def _trusted_event_gate(self, event_probs):
        return dict(event_probs), {}

    def _event_risk_scales(self, event_probs):
        del event_probs
        return [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]

    def _apply_forecast_pressure_trust_gate(self, blocks):
        return blocks

    def _select_objective_mode_cfg(self, planner_info, planner_blocks):
        del planner_info, planner_blocks
        return self.cfg, {
            "objective_mode_active": "economic",
            "objective_mode_persistent_high_buckets": 0,
            "objective_mode_future_rising": 0,
        }


def _pressure_blocks(*args, **kwargs):
    del args, kwargs
    return [
        {
            "pressure_vec": np.zeros(2, dtype=float),
            "pressure_vec_raw": np.zeros(2, dtype=float),
            "pressure_norm": 0.0,
        }
        for _ in range(3)
    ]


def _evaluate(sequence, blocks, plant_info, cfg):
    del blocks, plant_info, cfg
    costs = {"maintain_target": 0.0, "active_medium": 1.0}
    return {
        "sequence": sequence,
        "hard_reject_reason": "",
        "toy_cost": costs.get(sequence[0], 2.0),
    }


def _select_best(rows, cfg):
    del cfg
    ranked = sorted(rows, key=lambda row: row["toy_cost"])
    return ranked, [row["toy_cost"] for row in ranked]


class ForecastActionPolicyEnforcementTests(unittest.TestCase):
    def _run(self, mode: str, sequences):
        provider = _PlanningProbe(mode)
        with patch.multiple(
            "wind_prediction.provider_planning",
            compute_pressure_blocks=_pressure_blocks,
            apply_posture_state_to_blocks=lambda blocks, info, cfg: (blocks, {}),
            candidate_sequences=lambda cfg: list(sequences),
            evaluate_sequence=_evaluate,
            select_best_economic=_select_best,
            planner_action_vec=lambda *args, **kwargs: (
                np.asarray([1.0, 0.0], dtype=float),
                False,
            ),
        ):
            best, action_vec, _, _ = provider._run_planner(
                sample=object(),
                plant_info={},
            )
        return provider, best, action_vec

    def test_enforce_filters_first_actions_and_changes_final_action(self):
        sequences = [
            ("maintain_target",) * 3,
            ("active_medium",) * 3,
        ]

        shadow_provider, shadow_best, _ = self._run("shadow", sequences)
        enforce_provider, enforce_best, _ = self._run("enforce", sequences)

        self.assertEqual(shadow_best["sequence"][0], "maintain_target")
        self.assertEqual(enforce_best["sequence"][0], "active_medium")
        self.assertEqual(
            shadow_provider._forecast_action_policy_enforcement.candidate_count_after,
            2,
        )
        enforcement = enforce_provider._forecast_action_policy_enforcement
        self.assertTrue(enforcement.applied)
        self.assertEqual(enforcement.candidate_count_before, 2)
        self.assertEqual(enforcement.candidate_count_after, 1)
        self.assertEqual(enforcement.blocked_first_actions, ("maintain_target",))

    def test_shadow_records_policy_without_intervening(self):
        sequences = [
            ("maintain_target",) * 3,
            ("active_medium",) * 3,
        ]

        provider, best, _ = self._run("shadow", sequences)

        self.assertEqual(best["sequence"][0], "maintain_target")
        self.assertIsNotNone(provider._forecast_action_policy_snapshot)
        self.assertEqual(provider._forecast_action_policy_snapshot["mode"], "shadow")
        self.assertFalse(provider._forecast_action_policy_enforcement.applied)
        self.assertEqual(
            provider._forecast_action_policy_enforcement.reason,
            "mode_shadow",
        )

    def test_off_keeps_the_original_candidate_set(self):
        sequences = [("maintain_target",) * 3, ("active_medium",) * 3]

        provider, best, _ = self._run("off", sequences)

        self.assertEqual(best["sequence"][0], "maintain_target")
        self.assertIsNone(provider._forecast_action_policy_snapshot)
        self.assertFalse(provider._forecast_action_policy_enforcement.applied)
        self.assertEqual(provider._forecast_action_policy_enforcement.reason, "mode_off")

    def test_empty_filter_rolls_back_to_original_candidates(self):
        provider, best, action_vec = self._run(
            "enforce",
            [("maintain_target",) * 3],
        )

        self.assertEqual(best["sequence"], ("maintain_target",) * 3)
        np.testing.assert_allclose(action_vec, np.asarray([1.0, 0.0]))
        enforcement = provider._forecast_action_policy_enforcement
        self.assertFalse(enforcement.applied)
        self.assertEqual(enforcement.reason, "empty_filter_rollback")
        self.assertEqual(enforcement.candidate_count_before, 1)
        self.assertEqual(enforcement.candidate_count_after, 1)
        self.assertEqual(enforcement.blocked_first_actions, ("maintain_target",))

    def test_direct_empty_filter_returns_the_original_candidate_set(self):
        provider = _PlanningProbe("enforce")
        policy = evaluate_forecast_action_policy(
            provider._forecast_uv(None),
            config=provider.forecast_action_policy_config,
        )
        original = [("maintain_target",) * 3]

        filtered, enforcement = enforce_forecast_policy_on_first_actions(
            original,
            mode="enforce",
            result=policy,
        )

        self.assertEqual(filtered, original)
        self.assertFalse(enforcement.applied)
        self.assertEqual(enforcement.reason, "empty_filter_rollback")
        self.assertEqual(enforcement.candidate_count_before, 1)
        self.assertEqual(enforcement.candidate_count_after, 1)
        self.assertEqual(enforcement.blocked_first_actions, ("maintain_target",))

    def test_posture_demand_keeps_strengthening_candidate_available(self):
        provider = _PlanningProbe("enforce")
        policy = evaluate_forecast_action_policy(
            uv_ms=np.asarray(
                [[9.0, 0.0], [7.0, 0.0]] * 3,
                dtype=float,
            ),
            lead_reliability=np.ones(6, dtype=float),
            event_probs={},
            sample_period_s=600.0,
            config=provider.forecast_action_policy_config,
        )
        original = [
            ("maintain_target",) * 3,
            ("active_medium",) * 3,
        ]

        filtered, enforcement = enforce_forecast_policy_on_first_actions(
            original,
            mode="enforce",
            result=policy,
            posture_strengthening_required=True,
        )

        self.assertEqual(filtered, original)
        self.assertTrue(enforcement.posture_strengthen_restored)
        self.assertNotIn("active_medium", enforcement.blocked_first_actions)

    def test_direct_filter_preserves_unmanaged_ordinary_actions(self):
        provider = _PlanningProbe("enforce")
        result = provider._forecast_uv(None)
        policy = evaluate_forecast_action_policy(
            result,
            config=provider.forecast_action_policy_config,
        )
        sequences = [
            ("pump_saving",) * 3,
            ("active_small",) * 3,
            ("active_reverse_small",) * 3,
        ]

        filtered, enforcement = enforce_forecast_policy_on_first_actions(
            sequences,
            mode="enforce",
            result=policy,
        )

        self.assertEqual(
            [sequence[0] for sequence in filtered],
            ["pump_saving", "active_small"],
        )
        self.assertEqual(
            enforcement.blocked_first_actions,
            ("active_reverse_small",),
        )


if __name__ == "__main__":
    unittest.main()
