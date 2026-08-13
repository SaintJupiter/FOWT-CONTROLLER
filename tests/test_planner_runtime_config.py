import json
import tempfile
import unittest
from pathlib import Path

from src.wind_prediction.planner_runtime_config import (
    load_planner_runtime_config,
    parse_planner_runtime_config,
    validate_execution_target_refresh_contract,
)


class PlannerRuntimeConfigTests(unittest.TestCase):
    def test_loads_versioned_config(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "planner.json"
            path.write_text(
                json.dumps(
                    {
                        "schema_version": "planner_runtime.v1",
                        "pressure_sign_multiplier": -1.0,
                        "default_discount_blocks": [1.0, 0.85, 0.7],
                    }
                ),
                encoding="utf-8",
            )

            config = load_planner_runtime_config(path)

            self.assertEqual(config.pressure_sign_multiplier, -1.0)
            self.assertEqual(config.default_discount_blocks, (1.0, 0.85, 0.7))
            self.assertFalse(config.execution_rollout.enabled)

    def test_rejects_increasing_remote_weight(self):
        with self.assertRaisesRegex(ValueError, "non-increasing"):
            parse_planner_runtime_config(
                {
                    "schema_version": "planner_runtime.v1",
                    "pressure_sign_multiplier": -1.0,
                    "default_discount_blocks": [1.0, 0.7, 0.8],
                }
            )

    def test_rejects_unknown_schema(self):
        with self.assertRaisesRegex(ValueError, "Unsupported"):
            parse_planner_runtime_config(
                {
                    "schema_version": "planner_runtime.v3",
                    "pressure_sign_multiplier": -1.0,
                    "default_discount_blocks": [1.0],
                }
            )

    def test_v2_enables_structured_forecast_evidence(self):
        config = parse_planner_runtime_config(
            {
                "schema_version": "planner_runtime.v2",
                "pressure_sign_multiplier": -1.0,
                "default_discount_blocks": [1.0, 0.85, 0.7],
                "forecast_evidence": {
                    "pressure_aggregation_mode": "mean_step_force",
                    "lead_reliability_enabled": True,
                },
            }
        )

        self.assertEqual(config.pressure_aggregation_mode, "mean_step_force")
        self.assertTrue(config.lead_reliability_enabled)
        self.assertFalse(config.intrastage_stepwise_enabled)
        self.assertFalse(config.execution_rollout.enabled)

    def test_v2_accepts_intrastage_stepwise_forecast_evaluation(self):
        config = parse_planner_runtime_config(
            {
                "schema_version": "planner_runtime.v2",
                "pressure_sign_multiplier": -1.0,
                "default_discount_blocks": [1.0, 0.85, 0.7],
                "forecast_evidence": {
                    "pressure_aggregation_mode": "mean_step_force",
                    "lead_reliability_enabled": True,
                    "intrastage_stepwise_enabled": True,
                },
            }
        )

        self.assertEqual(
            config.pressure_aggregation_mode,
            "intrastage_stepwise",
        )
        self.assertTrue(config.lead_reliability_enabled)
        self.assertTrue(config.intrastage_stepwise_enabled)

    def test_v2_parses_shadow_forecast_action_policy(self):
        config = parse_planner_runtime_config(
            {
                "schema_version": "planner_runtime.v2",
                "pressure_sign_multiplier": -1.0,
                "default_discount_blocks": [1.0, 0.85, 0.7],
                "forecast_action_policy": {
                    "mode": "shadow",
                    "high_impact_reliability_min": 0.7,
                    "stage_event_keys": [
                        "attention_event_0_20m",
                        "attention_event_20_40m",
                        "attention_event_40_60m",
                    ],
                },
            }
        )

        self.assertEqual(config.forecast_action_policy_mode, "shadow")
        self.assertTrue(config.forecast_action_policy.enabled)
        self.assertEqual(
            config.forecast_action_policy.high_impact_reliability_min,
            0.7,
        )

    def test_v1_rejects_forecast_action_policy(self):
        with self.assertRaisesRegex(ValueError, "cannot enable"):
            parse_planner_runtime_config(
                {
                    "schema_version": "planner_runtime.v1",
                    "pressure_sign_multiplier": -1.0,
                    "default_discount_blocks": [1.0, 0.85, 0.7],
                    "forecast_action_policy": {"mode": "shadow"},
                }
            )

    def test_rejects_unknown_forecast_action_policy_field(self):
        with self.assertRaisesRegex(ValueError, "unknown fields"):
            parse_planner_runtime_config(
                {
                    "schema_version": "planner_runtime.v2",
                    "pressure_sign_multiplier": -1.0,
                    "default_discount_blocks": [1.0, 0.85, 0.7],
                    "forecast_action_policy": {"unexpected": True},
                }
            )

    def test_v1_rejects_intrastage_stepwise_forecast_evaluation(self):
        with self.assertRaisesRegex(ValueError, "cannot enable"):
            parse_planner_runtime_config(
                {
                    "schema_version": "planner_runtime.v1",
                    "pressure_sign_multiplier": -1.0,
                    "default_discount_blocks": [1.0, 0.85, 0.7],
                    "forecast_evidence": {
                        "intrastage_stepwise_enabled": True,
                    },
                }
            )

    def test_v1_cannot_silently_enable_v2_features(self):
        with self.assertRaisesRegex(ValueError, "cannot enable"):
            parse_planner_runtime_config(
                {
                    "schema_version": "planner_runtime.v1",
                    "pressure_sign_multiplier": -1.0,
                    "default_discount_blocks": [1.0, 0.85, 0.7],
                    "forecast_evidence": {
                        "pressure_aggregation_mode": "mean_step_force",
                    },
                }
            )

    def test_v1_cannot_enable_execution_rollout(self):
        with self.assertRaisesRegex(ValueError, "cannot enable"):
            parse_planner_runtime_config(
                {
                    "schema_version": "planner_runtime.v1",
                    "pressure_sign_multiplier": -1.0,
                    "default_discount_blocks": [1.0, 0.85, 0.7],
                    "execution_rollout": {"enabled": True},
                }
            )

    def test_v1_allows_only_explicit_disabled_marker(self):
        config = parse_planner_runtime_config(
            {
                "schema_version": "planner_runtime.v1",
                "pressure_sign_multiplier": -1.0,
                "default_discount_blocks": [1.0, 0.85, 0.7],
                "execution_rollout": {"enabled": False},
            }
        )

        self.assertFalse(config.execution_rollout.enabled)
        with self.assertRaisesRegex(ValueError, "cannot configure"):
            parse_planner_runtime_config(
                {
                    "schema_version": "planner_runtime.v1",
                    "pressure_sign_multiplier": -1.0,
                    "default_discount_blocks": [1.0, 0.85, 0.7],
                    "execution_rollout": {
                        "enabled": False,
                        "block_duration_s": 600.0,
                    },
                }
            )

    def test_v2_parses_execution_rollout(self):
        config = parse_planner_runtime_config(
            {
                "schema_version": "planner_runtime.v2",
                "pressure_sign_multiplier": -1.0,
                "default_discount_blocks": [1.0, 0.85, 0.7],
                "execution_rollout": {
                    "enabled": True,
                    "block_duration_s": 1200.0,
                    "internal_step_s": 2.0,
                    "water_density_kg_m3": 1025.0,
                    "target_slew_enabled": True,
                    "target_slew_rate_m3_min": 8.0,
                    "stop_error_kg": 250.0,
                    "restart_error_kg": 450.0,
                    "min_on_s": 15.0,
                    "min_off_s": 10.0,
                    "near_target_hold_s": 8.0,
                    "ramp_up_m3_min_per_s": 2.0,
                    "ramp_down_m3_min_per_s": 3.0,
                    "max_pump_rate_m3_min": 12.0,
                    "tank_capacity_kg": 1800000.0,
                    "pump_rate_schedule_m3_min": [
                        {"error_kg": 0.0, "rate_m3_min": 0.0},
                        {"error_kg": 500.0, "rate_m3_min": 8.0},
                        {"error_kg": 1000.0, "rate_m3_min": 12.0},
                    ],
                },
            }
        )

        rollout = config.execution_rollout
        self.assertTrue(rollout.enabled)
        self.assertEqual(rollout.block_duration_s, 1200.0)
        self.assertEqual(rollout.internal_step_s, 2.0)
        self.assertTrue(rollout.target_slew_enabled)
        self.assertEqual(
            rollout.pump_rate_schedule_m3_min,
            ((0.0, 0.0), (500.0, 8.0), (1000.0, 12.0)),
        )

    def test_planner_authoritative_lifecycle_requires_execution_rollout(self):
        with self.assertRaisesRegex(ValueError, "requires execution_rollout"):
            parse_planner_runtime_config(
                {
                    "schema_version": "planner_runtime.v2",
                    "target_lifecycle_mode": "planner_authoritative",
                    "pressure_sign_multiplier": -1.0,
                    "default_discount_blocks": [1.0, 0.85, 0.7],
                }
            )

        config = parse_planner_runtime_config(
            {
                "schema_version": "planner_runtime.v2",
                "target_lifecycle_mode": "planner_authoritative",
                "pressure_sign_multiplier": -1.0,
                "default_discount_blocks": [1.0, 0.85, 0.7],
                "execution_rollout": {"enabled": True},
            }
        )
        self.assertEqual(config.target_lifecycle_mode, "planner_authoritative")

    def test_explicit_target_actions_require_authoritative_execution(self):
        with self.assertRaisesRegex(ValueError, "planner_authoritative"):
            parse_planner_runtime_config(
                {
                    "schema_version": "planner_runtime.v2",
                    "candidate_action_mode": "explicit_target_lifecycle",
                    "pressure_sign_multiplier": -1.0,
                    "default_discount_blocks": [1.0, 0.85, 0.7],
                    "execution_rollout": {"enabled": True},
                }
            )

        config = parse_planner_runtime_config(
            {
                "schema_version": "planner_runtime.v2",
                "target_lifecycle_mode": "planner_authoritative",
                "candidate_action_mode": "explicit_target_lifecycle",
                "pressure_sign_multiplier": -1.0,
                "default_discount_blocks": [1.0, 0.85, 0.7],
                "execution_rollout": {"enabled": True},
            }
        )
        self.assertEqual(
            config.candidate_action_mode,
            "explicit_target_lifecycle",
        )

    def test_v2_parses_target_replan_policy(self):
        config = parse_planner_runtime_config(
            {
                "schema_version": "planner_runtime.v2",
                "target_lifecycle_mode": "planner_authoritative",
                "candidate_action_mode": "explicit_target_lifecycle",
                "pressure_sign_multiplier": -1.0,
                "default_discount_blocks": [1.0, 0.85, 0.7],
                "target_replan_policy": {
                    "mode": "shadow",
                    "rate_noise_tolerance_deg_s": 0.001,
                },
                "execution_rollout": {"enabled": True},
            }
        )

        self.assertEqual(config.target_replan_policy.mode, "shadow")
        self.assertEqual(
            config.target_replan_policy.rate_noise_tolerance_deg_s,
            0.001,
        )

    def test_target_replan_policy_requires_v2_explicit_execution(self):
        with self.assertRaisesRegex(ValueError, "cannot enable"):
            parse_planner_runtime_config(
                {
                    "schema_version": "planner_runtime.v1",
                    "pressure_sign_multiplier": -1.0,
                    "default_discount_blocks": [1.0, 0.85, 0.7],
                    "target_replan_policy": {"mode": "enforce"},
                }
            )

        with self.assertRaisesRegex(ValueError, "explicit target-lifecycle"):
            parse_planner_runtime_config(
                {
                    "schema_version": "planner_runtime.v2",
                    "pressure_sign_multiplier": -1.0,
                    "default_discount_blocks": [1.0, 0.85, 0.7],
                    "target_replan_policy": {"mode": "enforce"},
                    "execution_rollout": {"enabled": True},
                }
            )

    def test_execution_rollout_requires_active_bucket_target_refresh(self):
        config = parse_planner_runtime_config(
            {
                "schema_version": "planner_runtime.v2",
                "pressure_sign_multiplier": -1.0,
                "default_discount_blocks": [1.0, 0.85, 0.7],
                "execution_rollout": {"enabled": True},
            }
        )

        with self.assertRaisesRegex(ValueError, "active_bucket"):
            validate_execution_target_refresh_contract(
                config,
                event_reset_mode="action",
            )
        validate_execution_target_refresh_contract(
            config,
            event_reset_mode="active_bucket",
            bias_shape="event_decay",
        )

        with self.assertRaisesRegex(ValueError, "bias_shape='event_decay'"):
            validate_execution_target_refresh_contract(
                config,
                event_reset_mode="active_bucket",
                bias_shape="hold",
            )

    def test_legacy_planner_does_not_require_active_bucket_refresh(self):
        config = parse_planner_runtime_config(
            {
                "schema_version": "planner_runtime.v1",
                "pressure_sign_multiplier": -1.0,
                "default_discount_blocks": [1.0, 0.85, 0.7],
            }
        )

        validate_execution_target_refresh_contract(
            config,
            event_reset_mode="action",
        )

    def test_v2_json_declares_disabled_execution_rollout(self):
        config_path = (
            Path(__file__).resolve().parents[1]
            / "configs"
            / "planner_runtime_v2.json"
        )

        config = load_planner_runtime_config(config_path)

        self.assertFalse(config.execution_rollout.enabled)
        self.assertEqual(config.execution_rollout.block_duration_s, 600.0)
        self.assertTrue(config.execution_rollout.target_slew_enabled)
        self.assertEqual(len(config.execution_rollout.pump_rate_schedule_m3_min), 8)
        self.assertEqual(config.pressure_aggregation_mode, "mean_step_force")
        self.assertFalse(config.intrastage_stepwise_enabled)

    def test_enabled_execution_rollout_must_match_forecast_stage_duration(self):
        with self.assertRaisesRegex(
            ValueError,
            "execution_rollout.block_duration_s must match",
        ):
            parse_planner_runtime_config(
                {
                    "schema_version": "planner_runtime.v2",
                    "pressure_sign_multiplier": -1.0,
                    "default_discount_blocks": [1.0, 0.85, 0.7],
                    "forecast_action_policy": {
                        "mode": "shadow",
                        "stage_duration_s": 1200.0,
                    },
                    "execution_rollout": {
                        "enabled": True,
                        "block_duration_s": 600.0,
                    },
                }
            )

    def test_rejects_non_boolean_flags(self):
        with self.assertRaisesRegex(ValueError, "must be a boolean"):
            parse_planner_runtime_config(
                {
                    "schema_version": "planner_runtime.v2",
                    "pressure_sign_multiplier": -1.0,
                    "default_discount_blocks": [1.0, 0.85, 0.7],
                    "execution_rollout": {"enabled": 1},
                }
            )
        with self.assertRaisesRegex(ValueError, "must be a boolean"):
            parse_planner_runtime_config(
                {
                    "schema_version": "planner_runtime.v2",
                    "pressure_sign_multiplier": -1.0,
                    "default_discount_blocks": [1.0, 0.85, 0.7],
                    "forecast_evidence": {"lead_reliability_enabled": "false"},
                }
            )

    def test_rejects_non_numeric_execution_value(self):
        with self.assertRaisesRegex(ValueError, "must be a number"):
            parse_planner_runtime_config(
                {
                    "schema_version": "planner_runtime.v2",
                    "pressure_sign_multiplier": -1.0,
                    "default_discount_blocks": [1.0],
                    "execution_rollout": {"block_duration_s": "1200"},
                }
            )

    def test_rejects_invalid_error_hysteresis(self):
        with self.assertRaisesRegex(ValueError, "cannot be smaller"):
            parse_planner_runtime_config(
                {
                    "schema_version": "planner_runtime.v2",
                    "pressure_sign_multiplier": -1.0,
                    "default_discount_blocks": [1.0],
                    "execution_rollout": {
                        "stop_error_kg": 500.0,
                        "restart_error_kg": 300.0,
                    },
                }
            )

    def test_rejects_unsorted_pump_schedule(self):
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            parse_planner_runtime_config(
                {
                    "schema_version": "planner_runtime.v2",
                    "pressure_sign_multiplier": -1.0,
                    "default_discount_blocks": [1.0],
                    "execution_rollout": {
                        "pump_rate_schedule_m3_min": [
                            {"error_kg": 0.0, "rate_m3_min": 0.0},
                            {"error_kg": 500.0, "rate_m3_min": 8.0},
                            {"error_kg": 400.0, "rate_m3_min": 9.0},
                        ]
                    },
                }
            )

    def test_rejects_decreasing_pump_schedule_rate(self):
        with self.assertRaisesRegex(ValueError, "non-decreasing"):
            parse_planner_runtime_config(
                {
                    "schema_version": "planner_runtime.v2",
                    "pressure_sign_multiplier": -1.0,
                    "default_discount_blocks": [1.0],
                    "execution_rollout": {
                        "pump_rate_schedule_m3_min": [
                            {"error_kg": 0.0, "rate_m3_min": 0.0},
                            {"error_kg": 500.0, "rate_m3_min": 8.0},
                            {"error_kg": 1000.0, "rate_m3_min": 7.0},
                        ]
                    },
                }
            )

    def test_rejects_schedule_rate_above_maximum(self):
        with self.assertRaisesRegex(ValueError, "cannot exceed"):
            parse_planner_runtime_config(
                {
                    "schema_version": "planner_runtime.v2",
                    "pressure_sign_multiplier": -1.0,
                    "default_discount_blocks": [1.0],
                    "execution_rollout": {
                        "max_pump_rate_m3_min": 5.0,
                        "pump_rate_schedule_m3_min": [
                            {"error_kg": 0.0, "rate_m3_min": 0.0},
                            {"error_kg": 500.0, "rate_m3_min": 6.0},
                        ],
                    },
                }
            )


if __name__ == "__main__":
    unittest.main()
