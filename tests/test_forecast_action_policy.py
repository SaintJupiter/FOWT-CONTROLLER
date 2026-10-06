import unittest

import numpy as np

from wind_prediction.forecast_action_policy import (
    ForecastActionPolicyConfig,
    build_forecast_action_policy_snapshot,
    evaluate_forecast_action_policy,
)
from wind_prediction.forecast_evidence import ForecastEvidence
from wind_prediction.forecast_stage_grid import build_forecast_stage_grid


class ForecastActionPolicyTests(unittest.TestCase):
    def _config(self, **overrides):
        values = {
            "enabled": True,
            "stage_duration_s": 1200.0,
            "relative_speed_change_threshold": 0.08,
            "minimum_speed_change_ms": 0.5,
            "direction_consistency_min": 0.8,
            "reversal_angle_deg": 120.0,
            "continuous_reversal_points": 2,
            "high_impact_reliability_min": 0.65,
            "high_impact_event_probability_min": 0.60,
        }
        values.update(overrides)
        return ForecastActionPolicyConfig(**values)

    def _evaluate(
        self,
        uv,
        *,
        current_uv=None,
        reliability=None,
        events=None,
        config=None,
    ):
        return evaluate_forecast_action_policy(
            uv_ms=np.asarray(uv, dtype=float),
            current_uv_ms=(
                None
                if current_uv is None
                else np.asarray(current_uv, dtype=float)
            ),
            lead_reliability=(
                np.ones(len(uv), dtype=float)
                if reliability is None
                else np.asarray(reliability, dtype=float)
            ),
            event_probs=events or {},
            sample_period_s=600.0,
            config=config or self._config(),
        )

    def test_current_observation_extends_first_stage_strengthening_trend(self):
        uv = [
            [7.0, 0.0],
            [7.0, 0.0],
            [7.0, 0.0],
            [7.0, 0.0],
            [7.0, 0.0],
            [7.0, 0.0],
        ]
        result = self._evaluate(
            uv,
            current_uv=[5.0, 0.0],
            events={"attention_event_0_20m": 0.9},
        )
        unsupported = self._evaluate(
            uv,
            current_uv=[5.0, 0.0],
            events={"attention_event_0_20m": 0.1},
        )

        first = result.stages[0]
        unsupported_first = unsupported.stages[0]
        self.assertAlmostEqual(first.speed_change_ms, 2.0)
        self.assertTrue(first.strengthening)
        self.assertTrue(first.strengthen.allowed)
        self.assertEqual(first.speed_change_ms, unsupported_first.speed_change_ms)
        self.assertEqual(first.mean_speed_ms, unsupported_first.mean_speed_ms)
        self.assertTrue(unsupported_first.strengthening)
        self.assertFalse(unsupported_first.strengthen.allowed)

    def test_current_observation_extends_first_stage_declining_trend(self):
        result = self._evaluate(
            [
                [7.0, 0.0],
                [7.0, 0.0],
                [7.0, 0.0],
                [7.0, 0.0],
                [7.0, 0.0],
                [7.0, 0.0],
            ],
            current_uv=[9.0, 0.0],
        )

        first = result.stages[0]
        self.assertAlmostEqual(first.speed_change_ms, -2.0)
        self.assertTrue(first.declining)
        self.assertTrue(first.hold.allowed)
        self.assertTrue(first.release.allowed)

    def test_current_direction_detects_first_stage_continuous_reversal(self):
        result = self._evaluate(
            [
                [-8.0, 0.0],
                [-8.0, 0.0],
                [-8.0, 0.0],
                [-8.0, 0.0],
                [-8.0, 0.0],
                [-8.0, 0.0],
            ],
            current_uv=[8.0, 0.0],
            events={"attention_event_0_20m": 0.9},
        )

        first = result.stages[0]
        self.assertTrue(first.continuous_reversal)
        self.assertEqual(first.reversal_run_length, 2)
        self.assertTrue(first.reverse.allowed)

    def test_omitting_current_observation_preserves_first_stage_behavior(self):
        uv = [
            [6.0, 0.0],
            [7.0, 0.0],
            [8.0, 0.0],
            [9.0, 0.0],
            [10.0, 0.0],
            [11.0, 0.0],
        ]

        inputs = {
            "uv_ms": np.asarray(uv, dtype=float),
            "lead_reliability": np.ones(len(uv), dtype=float),
            "event_probs": {},
            "sample_period_s": 600.0,
            "config": self._config(),
        }
        omitted = evaluate_forecast_action_policy(**inputs)
        explicit_none = evaluate_forecast_action_policy(
            **inputs,
            current_uv_ms=None,
        )

        self.assertEqual(omitted, explicit_none)
        self.assertAlmostEqual(omitted.stages[0].speed_change_ms, 1.0)

    def test_default_policy_is_disabled_and_does_not_evaluate_inputs(self):
        result = evaluate_forecast_action_policy()
        self.assertFalse(result.enabled)
        self.assertEqual(result.reason, "policy_disabled")
        self.assertEqual(result.stages, ())

    def test_shadow_snapshot_names_stage_actions_and_threshold_sources(self):
        result = self._evaluate(
            [
                [6.0, 0.0],
                [7.0, 0.0],
                [8.0, 0.0],
                [9.0, 0.0],
                [10.0, 0.0],
                [11.0, 0.0],
            ],
            events={
                "attention_event_0_20m": 0.8,
                "attention_event_20_40m": 0.8,
                "attention_event_40_60m": 0.8,
            },
        )

        snapshot = build_forecast_action_policy_snapshot(result, mode="shadow")

        self.assertEqual(snapshot["stage_labels"], [
            "0-20 min",
            "20-40 min",
            "40-60 min",
        ])
        self.assertIn("strengthen", snapshot["authorized_actions"]["0-20 min"])
        self.assertEqual(
            snapshot["threshold_source"]["0-20 min"],
            "config_fallback",
        )

    def test_six_points_form_three_stages_and_authorize_strengthening(self):
        result = self._evaluate(
            [
                [6.0, 0.0],
                [7.0, 0.0],
                [8.0, 0.0],
                [9.0, 0.0],
                [10.0, 0.0],
                [11.0, 0.0],
            ],
            events={
                "attention_event_0_20m": 0.8,
                "attention_event_20_40m": 0.8,
                "attention_event_40_60m": 0.8,
            },
        )
        self.assertEqual(len(result.stages), 3)
        self.assertTrue(all(stage.strengthening for stage in result.stages))
        self.assertTrue(all(stage.strengthen.allowed for stage in result.stages))
        self.assertTrue(
            all(stage.high_impact_event_supported for stage in result.stages)
        )

    def test_first_policy_stage_uses_plus_10_and_plus_20_min_records(self):
        result = self._evaluate(
            [
                [6.0, 0.0],
                [7.0, 0.0],
                [8.0, 0.0],
                [9.0, 0.0],
                [10.0, 0.0],
                [11.0, 0.0],
            ]
        )

        first = result.stages[0]
        self.assertEqual(first.point_indices, (0, 1))
        self.assertEqual(first.lead_start_s, 600.0)
        self.assertEqual(first.lead_end_s, 1200.0)

    def test_explicit_controller_grid_owns_policy_point_assignment(self):
        grid = build_forecast_stage_grid(
            sample_period_s=600.0,
            stage_duration_s=1200.0,
            stage_count=3,
            available_point_count=6,
        )
        result = evaluate_forecast_action_policy(
            uv_ms=np.asarray(
                [[6.0, 0.0], [7.0, 0.0], [8.0, 0.0],
                 [9.0, 0.0], [10.0, 0.0], [11.0, 0.0]],
                dtype=float,
            ),
            current_uv_ms=np.asarray([5.0, 0.0]),
            lead_reliability=np.ones(6, dtype=float),
            event_probs={
                "attention_event_0_20m": 0.9,
                "attention_event_20_40m": 0.9,
                "attention_event_40_60m": 0.9,
            },
            sample_period_s=600.0,
            stage_grid=grid,
            config=self._config(),
        )

        self.assertEqual(
            [(stage.point_indices, stage.lead_start_s, stage.lead_end_s) for stage in result.stages],
            [
                ((0, 1), 600.0, 1200.0),
                ((2, 3), 1800.0, 2400.0),
                ((4, 5), 3000.0, 3600.0),
            ],
        )

    def test_declining_forecast_authorizes_hold_and_release_without_event_support(self):
        result = self._evaluate(
            [
                [12.0, 0.0],
                [11.0, 0.0],
                [9.0, 0.0],
                [8.0, 0.0],
                [6.0, 0.0],
                [5.0, 0.0],
            ],
            reliability=[0.2] * 6,
        )
        self.assertTrue(all(stage.declining for stage in result.stages))
        self.assertTrue(all(stage.hold.allowed for stage in result.stages))
        self.assertTrue(all(stage.release.allowed for stage in result.stages))

    def test_single_point_direction_noise_does_not_authorize_reverse(self):
        result = self._evaluate(
            [
                [8.0, 0.0],
                [8.0, 0.0],
                [-8.0, 0.0],
                [8.0, 0.0],
                [8.0, 0.0],
                [8.0, 0.0],
            ],
            events={"attention_event_20_40m": 0.9},
        )
        self.assertFalse(any(stage.continuous_reversal for stage in result.stages))
        self.assertFalse(any(stage.reverse.allowed for stage in result.stages))

    def test_two_consecutive_reversed_points_authorize_reverse(self):
        result = self._evaluate(
            [
                [8.0, 0.0],
                [8.0, 0.0],
                [8.0, 0.0],
                [-8.0, 0.0],
                [-8.0, 0.0],
                [-8.0, 0.0],
            ],
            events={
                "attention_event_20_40m": 0.9,
                "attention_event_40_60m": 0.9,
            },
        )
        self.assertFalse(result.stages[1].continuous_reversal)
        self.assertTrue(result.stages[2].continuous_reversal)
        self.assertTrue(result.stages[2].reverse.allowed)

    def test_low_reliability_blocks_only_high_impact_authorization(self):
        uv = [
            [6.0, 0.0],
            [7.0, 0.0],
            [8.0, 0.0],
            [9.0, 0.0],
            [10.0, 0.0],
            [11.0, 0.0],
        ]
        events = {
            "attention_event_0_20m": 0.9,
            "attention_event_20_40m": 0.9,
            "attention_event_40_60m": 0.9,
        }
        result = self._evaluate(
            uv,
            reliability=[0.4] * 6,
            events=events,
        )
        trusted = self._evaluate(
            uv,
            reliability=[1.0] * 6,
            events=events,
        )
        self.assertTrue(all(stage.strengthening for stage in result.stages))
        np.testing.assert_allclose(
            [stage.mean_speed_ms for stage in result.stages],
            [stage.mean_speed_ms for stage in trusted.stages],
        )
        np.testing.assert_allclose(
            [stage.speed_change_ms for stage in result.stages],
            [stage.speed_change_ms for stage in trusted.stages],
        )
        self.assertTrue(all(not stage.strengthen.allowed for stage in result.stages))
        self.assertTrue(all(stage.strengthen.allowed for stage in trusted.stages))
        self.assertTrue(
            all(
                stage.strengthen.reason == "low_forecast_reliability"
                for stage in result.stages
            )
        )

    def test_accepts_forecast_evidence_directly(self):
        evidence = ForecastEvidence(
            source="test",
            model_version="test",
            origin_time=None,
            sample_period_s=600.0,
            uv_ms=np.asarray([[7.0, 0.0]] * 6, dtype=float),
            event_probs={},
            lead_reliability=np.ones(6, dtype=float),
        )
        result = evaluate_forecast_action_policy(evidence, config=self._config())
        self.assertEqual(len(result.stages), 3)
        self.assertTrue(all(stage.sustained for stage in result.stages))
        self.assertTrue(all(stage.hold.allowed for stage in result.stages))

    def test_model_stage_thresholds_override_generic_fallback(self):
        evidence = ForecastEvidence(
            source="test",
            model_version="test",
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
                "attention_event_0_20m": 0.70,
                "attention_event_20_40m": 0.70,
                "attention_event_40_60m": 0.70,
            },
            lead_reliability=np.ones(6, dtype=float),
            metadata={
                "event_thresholds": {
                    "thresholds": {
                        "attention_event_0_20m": 0.80,
                        "attention_event_20_40m": 0.65,
                    },
                    "threshold_mode": "best_f1",
                }
            },
        )
        result = evaluate_forecast_action_policy(evidence, config=self._config())

        first, second, third = result.stages
        self.assertFalse(first.strengthen.allowed)
        self.assertEqual(first.event_threshold, 0.80)
        self.assertEqual(first.event_threshold_source, "model_metadata")
        self.assertTrue(second.strengthen.allowed)
        self.assertEqual(second.event_threshold, 0.65)
        self.assertEqual(second.event_threshold_source, "model_metadata")
        self.assertTrue(third.strengthen.allowed)
        self.assertEqual(third.event_threshold, 0.60)
        self.assertEqual(third.event_threshold_source, "config_fallback")

    def test_direct_model_threshold_mapping_is_supported(self):
        evidence = ForecastEvidence(
            source="test",
            model_version="test",
            origin_time=None,
            sample_period_s=600.0,
            uv_ms=np.asarray([[6.0, 0.0], [7.0, 0.0]] * 3, dtype=float),
            event_probs={"attention_event_0_20m": 0.55},
            lead_reliability=np.ones(6, dtype=float),
            metadata={
                "event_thresholds": {"attention_event_0_20m": 0.50}
            },
        )
        result = evaluate_forecast_action_policy(evidence, config=self._config())
        self.assertTrue(result.stages[0].strengthen.allowed)
        self.assertEqual(result.stages[0].event_threshold, 0.50)

    def test_complete_model_threshold_blob_in_metadata_is_supported(self):
        evidence = ForecastEvidence(
            source="test",
            model_version="test",
            origin_time=None,
            sample_period_s=600.0,
            uv_ms=np.asarray([[6.0, 0.0], [7.0, 0.0]] * 3, dtype=float),
            event_probs={"attention_event_0_20m": 0.70},
            lead_reliability=np.ones(6, dtype=float),
            metadata={
                "threshold_mode": "best_f1",
                "thresholds": {"attention_event_0_20m": 0.80},
                "selected_on": "validation",
            },
        )
        result = evaluate_forecast_action_policy(evidence, config=self._config())
        self.assertFalse(result.stages[0].strengthen.allowed)
        self.assertEqual(result.stages[0].event_threshold, 0.80)
        self.assertEqual(
            result.stages[0].event_threshold_source,
            "model_metadata",
        )

    def test_invalid_model_threshold_fails_fast(self):
        evidence = ForecastEvidence(
            source="test",
            model_version="test",
            origin_time=None,
            sample_period_s=600.0,
            uv_ms=np.asarray([[7.0, 0.0]] * 6, dtype=float),
            event_probs={},
            lead_reliability=np.ones(6, dtype=float),
            metadata={
                "event_thresholds": {"attention_event_0_20m": 1.2}
            },
        )
        with self.assertRaisesRegex(ValueError, "must lie in"):
            evaluate_forecast_action_policy(evidence, config=self._config())


if __name__ == "__main__":
    unittest.main()
