import sys
import unittest
from pathlib import Path

import numpy as np


LEGACY_DIR = Path(__file__).resolve().parents[1] / "archive" / "legacy_fowt_control"
sys.path.insert(0, str(LEGACY_DIR))

from controllers_extras import ClosedLoopPolicy, evaluate_forecast_safe_deadband  # noqa: E402


def preview(**overrides):
    values = {
        "preview_forecast_has_future": 1,
        "preview_forecast_control_trust_ok": 1,
        "preview_forecast_control_trust_event_prob_max": 0.30,
        "preview_pressure_block0_norm": 0.90,
        "preview_pressure_block1_norm": 0.85,
        "preview_pressure_block2_norm": 0.80,
        "preview_pressure_block02_dot": 0.60,
        "preview_primary_action": "hold",
    }
    values.update(overrides)
    return values


def state(pitch_deg=0.5, roll_deg=0.4, pitch_rate_deg_s=0.0, roll_rate_deg_s=0.0):
    values = np.zeros(12, dtype=float)
    values[3] = np.radians(roll_deg)
    values[4] = np.radians(pitch_deg)
    values[9] = np.radians(roll_rate_deg_s)
    values[10] = np.radians(pitch_rate_deg_s)
    return values


class ForecastSafeDeadbandDecisionTests(unittest.TestCase):
    def evaluate(self, forecast=None, platform_state=None):
        return evaluate_forecast_safe_deadband(
            preview(**forecast) if isinstance(forecast, dict) else (forecast or preview()),
            state() if platform_state is None else platform_state,
            enabled=True,
            event_probability_max=0.75,
            future_rise_max=0.05,
            minimum_direction_dot=0.0,
            posture_gate_deg=1.2,
            allowed_actions=("hold",),
        )

    def test_safe_lstm_hold_enables_deadband(self):
        result = self.evaluate()
        self.assertTrue(result["active"])
        self.assertEqual(result["reason"], "forecast_safe_deadband")

    def test_current_only_cannot_enable_deadband(self):
        result = self.evaluate({"preview_forecast_has_future": 0})
        self.assertFalse(result["active"])
        self.assertEqual(result["reason"], "no_future_forecast")

    def test_untrusted_forecast_cannot_enable_deadband(self):
        result = self.evaluate({"preview_forecast_control_trust_ok": 0})
        self.assertFalse(result["active"])
        self.assertEqual(result["reason"], "forecast_untrusted")

    def test_high_event_probability_blocks_deadband(self):
        result = self.evaluate(
            {"preview_forecast_control_trust_event_prob_max": 0.90}
        )
        self.assertFalse(result["active"])
        self.assertEqual(result["reason"], "event_risk_high")

    def test_required_event_probability_blocks_missing_event_head(self):
        result = evaluate_forecast_safe_deadband(
            preview(
                preview_forecast_control_trust_event_prob_max=0.0,
                preview_forecast_event_probs_available=0,
            ),
            state(),
            enabled=True,
            event_probability_max=0.75,
            future_rise_max=0.05,
            minimum_direction_dot=0.0,
            posture_gate_deg=1.2,
            allowed_actions=("hold",),
            require_event_probability=True,
        )
        self.assertFalse(result["active"])
        self.assertEqual(result["reason"], "event_probability_unavailable")

    def test_required_event_probability_accepts_available_safe_event_head(self):
        result = evaluate_forecast_safe_deadband(
            preview(preview_forecast_event_probs_available=1),
            state(),
            enabled=True,
            event_probability_max=0.75,
            future_rise_max=0.05,
            minimum_direction_dot=0.0,
            posture_gate_deg=1.2,
            allowed_actions=("hold",),
            require_event_probability=True,
        )
        self.assertTrue(result["active"])
        self.assertEqual(result["event_probability_available"], 1)

    def test_future_rise_blocks_deadband(self):
        result = self.evaluate(
            {
                "preview_pressure_block0_norm": 0.80,
                "preview_pressure_block1_norm": 0.95,
                "preview_pressure_block2_norm": 1.00,
            }
        )
        self.assertFalse(result["active"])
        self.assertEqual(result["reason"], "future_pressure_rising")

    def test_direction_reversal_blocks_deadband(self):
        result = self.evaluate({"preview_pressure_block02_dot": -0.10})
        self.assertFalse(result["active"])
        self.assertEqual(result["reason"], "direction_reversal")

    def test_posture_outside_gate_blocks_deadband(self):
        result = self.evaluate(platform_state=state(pitch_deg=1.3, roll_deg=0.4))
        self.assertFalse(result["active"])
        self.assertEqual(result["reason"], "posture_gate")

    def test_medium_action_does_not_use_deadband(self):
        result = self.evaluate({"preview_primary_action": "active_medium"})
        self.assertFalse(result["active"])
        self.assertEqual(result["reason"], "action_not_eligible")

    def test_small_action_can_use_deadband_when_configured(self):
        result = evaluate_forecast_safe_deadband(
            preview(preview_primary_action="active_small"),
            state(),
            enabled=True,
            event_probability_max=0.75,
            future_rise_max=0.05,
            minimum_direction_dot=0.0,
            posture_gate_deg=1.2,
            allowed_actions=("hold", "active_small"),
        )
        self.assertTrue(result["active"])

    def test_forecast_veto_mode_admits_any_candidate_when_future_is_not_dangerous(self):
        result = evaluate_forecast_safe_deadband(
            preview(
                preview_primary_action="active_medium",
                preview_pressure_block0_norm=0.70,
                preview_pressure_block1_norm=1.20,
                preview_pressure_block2_norm=1.30,
                preview_pressure_block02_dot=-0.40,
            ),
            state(),
            enabled=True,
            event_probability_max=0.95,
            future_rise_max=0.05,
            minimum_direction_dot=0.0,
            posture_gate_deg=1.5,
            allowed_actions=("hold",),
            gate_mode="forecast_veto",
        )
        self.assertTrue(result["active"])
        self.assertEqual(result["reason"], "forecast_safe_deadband")

    def test_forecast_veto_mode_blocks_predicted_danger(self):
        result = evaluate_forecast_safe_deadband(
            preview(preview_forecast_control_trust_event_prob_max=0.97),
            state(),
            enabled=True,
            event_probability_max=0.95,
            future_rise_max=0.05,
            minimum_direction_dot=0.0,
            posture_gate_deg=1.5,
            allowed_actions=("hold",),
            gate_mode="forecast_veto",
        )
        self.assertFalse(result["active"])
        self.assertEqual(result["reason"], "event_risk_high")

    def test_forecast_veto_mode_keeps_current_posture_guard(self):
        result = evaluate_forecast_safe_deadband(
            preview(preview_primary_action="active_medium"),
            state(pitch_deg=1.6, roll_deg=0.4),
            enabled=True,
            event_probability_max=0.95,
            future_rise_max=0.05,
            minimum_direction_dot=0.0,
            posture_gate_deg=1.5,
            allowed_actions=("hold",),
            gate_mode="forecast_veto",
        )
        self.assertFalse(result["active"])
        self.assertEqual(result["reason"], "posture_gate")


class ForecastSafeDeadbandPolicyTests(unittest.TestCase):
    class Controller:
        def __init__(self):
            self.deadband_enter = {"pitch": 0.5, "roll": 0.4, "heave": 0.05}
            self.deadband_exit = {"pitch": 0.25, "roll": 0.2, "heave": 0.025}
            self.max_mass = 2_500_000.0

        def reset(self):
            return None

    class Provider:
        def __init__(self):
            self.sync_calls = 0
            self.exit_calls = 0

        def synchronize_forecast_deadband_hold(self, plant_info, current_time, **kwargs):
            self.sync_calls += 1
            return np.asarray(plant_info["tank_masses"], dtype=float)

        def exit_forecast_deadband_hold(self, plant_info, current_time):
            self.exit_calls += 1

    def policy(self):
        return ClosedLoopPolicy(
            controller=self.Controller(),
            primary_safety_cfg={
                "forecast_safe_deadband_enabled": True,
                "forecast_safe_deadband_pitch_enter_deg": 1.5,
                "forecast_safe_deadband_roll_enter_deg": 1.5,
                "forecast_safe_deadband_pitch_exit_deg": 1.2,
                "forecast_safe_deadband_roll_exit_deg": 1.2,
                "forecast_safe_deadband_posture_gate_deg": 1.2,
            },
        )

    def test_policy_opens_and_restores_deadband(self):
        policy = self.policy()
        active = policy._apply_forecast_safe_deadband(state(), preview())
        self.assertTrue(active["active"])
        self.assertEqual(policy.controller.deadband_enter["pitch"], 1.5)
        self.assertEqual(policy.controller.deadband_exit["roll"], 1.2)

        inactive = policy._apply_forecast_safe_deadband(
            state(), preview(preview_forecast_has_future=0)
        )
        self.assertFalse(inactive["active"])
        self.assertEqual(policy.controller.deadband_enter["pitch"], 0.5)
        self.assertEqual(policy.controller.deadband_exit["roll"], 0.2)

    def test_policy_reset_clears_forecast_admission_debug_state(self):
        policy = self.policy()
        active = policy._apply_forecast_safe_deadband(
            state(),
            preview(preview_forecast_event_probs_available=1),
        )
        self.assertTrue(active["active"])

        policy.reset(np.zeros(3, dtype=float))

        self.assertFalse(policy._forecast_safe_deadband_dbg["active"])
        self.assertEqual(policy._forecast_safe_deadband_dbg["has_future"], 0)
        self.assertEqual(policy._forecast_safe_deadband_dbg["trust_ok"], 0)
        self.assertEqual(
            policy._forecast_safe_deadband_dbg["event_probability_available"],
            0,
        )

    def test_command_release_requires_forecast_admission_without_rewriting_provider(self):
        provider = self.Provider()
        policy = ClosedLoopPolicy(
            controller=self.Controller(),
            preview_trim_provider=provider,
            primary_safety_cfg={
                "forecast_safe_deadband_enabled": True,
                "forecast_safe_deadband_gate_mode": "forecast_veto",
                "forecast_safe_deadband_sync_provider_target": False,
                "forecast_safe_deadband_apply_pid_deadband": False,
                "forecast_safe_deadband_release_primary_target": False,
                "forecast_safe_deadband_posture_gate_deg": 1.5,
                "forecast_safe_deadband_event_probability_max": 0.99,
                "deadband_target_release_enabled": True,
                "deadband_target_release_require_forecast_safe": True,
                "deadband_target_release_pitch_deg": 1.5,
                "deadband_target_release_roll_deg": 1.5,
                "deadband_target_release_exit_pitch_deg": 1.5,
                "deadband_target_release_exit_roll_deg": 1.5,
                "deadband_target_release_blend": 1.0,
                "deadband_target_release_require_both_axes": True,
            },
        )
        masses = np.array([1000.0, 1000.0, 1000.0])
        command = np.array([1200.0, 800.0, 1000.0])
        plant = {"tank_masses": masses}
        ctrl_dbg = {
            "pid_details": {
                "pitch": {"in_deadband": 1, "deadband_exit": 1.5},
                "roll": {"in_deadband": 1, "deadband_exit": 1.5},
            }
        }

        blocked, blocked_dbg = policy._deadband_target_release_update(
            state(), plant, ctrl_dbg, command, command
        )
        np.testing.assert_allclose(blocked, command)
        self.assertEqual(blocked_dbg["reason"], "forecast_not_safe")

        admitted = policy._apply_forecast_safe_deadband(
            state(), preview(), plant_info_prev=plant, current_time=10.0
        )
        self.assertTrue(admitted["active"])
        self.assertEqual(policy.controller.deadband_enter["pitch"], 0.5)
        self.assertEqual(policy.controller.deadband_exit["roll"], 0.2)
        self.assertFalse(policy.forecast_safe_deadband_release_primary_target)
        released, released_dbg = policy._deadband_target_release_update(
            state(), plant, ctrl_dbg, command, command
        )
        np.testing.assert_allclose(released, masses)
        self.assertEqual(released_dbg["active"], 1)
        self.assertEqual(provider.sync_calls, 0)
        self.assertEqual(provider.exit_calls, 0)

    def test_forecast_release_can_use_its_own_posture_limit_outside_pid_deadband(self):
        policy = ClosedLoopPolicy(
            controller=self.Controller(),
            primary_safety_cfg={
                "forecast_safe_deadband_enabled": True,
                "forecast_safe_deadband_gate_mode": "forecast_veto",
                "forecast_safe_deadband_apply_pid_deadband": False,
                "forecast_safe_deadband_release_primary_target": False,
                "forecast_safe_deadband_posture_gate_deg": 1.5,
                "forecast_safe_deadband_event_probability_max": 0.99,
                "deadband_target_release_enabled": True,
                "deadband_target_release_require_forecast_safe": True,
                "deadband_target_release_use_pid_deadband": False,
                "deadband_target_release_pitch_deg": 1.0,
                "deadband_target_release_roll_deg": 1.0,
                "deadband_target_release_exit_pitch_deg": 1.5,
                "deadband_target_release_exit_roll_deg": 1.5,
            },
        )
        masses = np.array([1000.0, 1000.0, 1000.0])
        command = np.array([1200.0, 800.0, 1000.0])
        ctrl_dbg = {
            "pid_details": {
                "pitch": {"in_deadband": 0, "deadband_exit": 0.25},
                "roll": {"in_deadband": 0, "deadband_exit": 0.20},
            }
        }
        policy._apply_forecast_safe_deadband(state(), preview())
        released, release_dbg = policy._deadband_target_release_update(
            state(pitch_deg=0.9, roll_deg=0.8, pitch_rate_deg_s=-0.01, roll_rate_deg_s=-0.01),
            {"tank_masses": masses},
            ctrl_dbg,
            command,
            command,
        )
        np.testing.assert_allclose(released, masses)
        self.assertEqual(release_dbg["active"], 1)

    def test_policy_synchronizes_actionable_target_and_replans_on_exit(self):
        provider = self.Provider()
        policy = ClosedLoopPolicy(
            controller=self.Controller(),
            preview_trim_provider=provider,
            primary_safety_cfg={
                "forecast_safe_deadband_enabled": True,
                "forecast_safe_deadband_gate_mode": "forecast_veto",
                "forecast_safe_deadband_sync_provider_target": True,
                "forecast_safe_deadband_sync_require_recovering": True,
                "forecast_safe_deadband_sync_require_pump_demand": True,
                "forecast_safe_deadband_sync_future_rise_max": 0.10,
                "forecast_safe_deadband_pitch_enter_deg": 1.5,
                "forecast_safe_deadband_roll_enter_deg": 1.5,
                "forecast_safe_deadband_pitch_exit_deg": 1.2,
                "forecast_safe_deadband_roll_exit_deg": 1.2,
                "forecast_safe_deadband_posture_gate_deg": 1.5,
                "forecast_safe_deadband_event_probability_max": 0.95,
                "forecast_safe_deadband_allowed_actions": (
                    "hold",
                    "pump_saving",
                    "active_small",
                    "active_medium",
                    "active_reverse_small",
                ),
            },
        )
        preview_values = preview(
            preview_primary_action="active_medium",
            preview_primary_target_kg=[1200.0, 800.0, 1000.0],
            preview_primary_delta_kg=[200.0, -200.0, 0.0],
            preview_primary_enabled=1,
        )
        plant = {
            "tank_masses": np.array([1000.0, 1000.0, 1000.0]),
            "pump_rate_cmd_m3_min": np.array([0.1, 0.0, 0.0]),
        }

        active = policy._apply_forecast_safe_deadband(
            state(), preview_values, plant_info_prev=plant, current_time=10.0
        )
        self.assertTrue(active["active"])
        self.assertEqual(active["provider_target_synced"], 1)
        self.assertEqual(provider.sync_calls, 1)
        self.assertEqual(preview_values["preview_primary_action"], "hold")
        np.testing.assert_allclose(
            preview_values["preview_primary_target_kg"], plant["tank_masses"]
        )

        preview_values["preview_forecast_control_trust_event_prob_max"] = 0.99
        inactive = policy._apply_forecast_safe_deadband(
            state(), preview_values, plant_info_prev=plant, current_time=11.0
        )
        self.assertFalse(inactive["active"])
        self.assertEqual(inactive["provider_replan_requested"], 1)
        self.assertEqual(provider.exit_calls, 1)
        self.assertEqual(preview_values["preview_primary_enabled"], 0)

    def test_policy_does_not_synchronize_target_while_posture_moves_away(self):
        provider = self.Provider()
        policy = ClosedLoopPolicy(
            controller=self.Controller(),
            preview_trim_provider=provider,
            primary_safety_cfg={
                "forecast_safe_deadband_enabled": True,
                "forecast_safe_deadband_gate_mode": "forecast_veto",
                "forecast_safe_deadband_sync_provider_target": True,
                "forecast_safe_deadband_sync_require_recovering": True,
                "forecast_safe_deadband_sync_require_pump_demand": True,
                "forecast_safe_deadband_sync_future_rise_max": 0.10,
                "forecast_safe_deadband_posture_gate_deg": 1.5,
                "forecast_safe_deadband_event_probability_max": 0.95,
            },
        )
        preview_values = preview(
            preview_primary_action="active_medium",
            preview_primary_target_kg=[1200.0, 800.0, 1000.0],
            preview_primary_enabled=1,
        )
        plant = {
            "tank_masses": np.array([1000.0, 1000.0, 1000.0]),
            "pump_rate_cmd_m3_min": np.array([0.1, 0.0, 0.0]),
        }

        result = policy._apply_forecast_safe_deadband(
            state(pitch_deg=0.5, roll_deg=0.4, pitch_rate_deg_s=0.2),
            preview_values,
            plant_info_prev=plant,
            current_time=10.0,
        )

        self.assertTrue(result["active"])
        self.assertEqual(result["posture_recovering"], 0)
        self.assertEqual(result["provider_target_synced"], 0)
        self.assertEqual(result["provider_sync_reason"], "posture_not_recovering")
        self.assertEqual(provider.sync_calls, 0)

    def test_policy_replans_when_future_pressure_rises_during_synced_hold(self):
        provider = self.Provider()
        policy = ClosedLoopPolicy(
            controller=self.Controller(),
            preview_trim_provider=provider,
            primary_safety_cfg={
                "forecast_safe_deadband_enabled": True,
                "forecast_safe_deadband_gate_mode": "forecast_veto",
                "forecast_safe_deadband_sync_provider_target": True,
                "forecast_safe_deadband_sync_require_recovering": True,
                "forecast_safe_deadband_sync_require_pump_demand": True,
                "forecast_safe_deadband_sync_future_rise_max": 0.10,
                "forecast_safe_deadband_posture_gate_deg": 1.5,
                "forecast_safe_deadband_event_probability_max": 0.95,
            },
        )
        plant = {
            "tank_masses": np.array([1000.0, 1000.0, 1000.0]),
            "pump_rate_cmd_m3_min": np.array([0.1, 0.0, 0.0]),
        }
        safe_preview = preview(
            preview_primary_target_kg=[1200.0, 800.0, 1000.0],
            preview_primary_enabled=1,
        )
        policy._apply_forecast_safe_deadband(
            state(), safe_preview, plant_info_prev=plant, current_time=10.0
        )
        rising_preview = preview(
            preview_pressure_block0_norm=0.70,
            preview_pressure_block2_norm=0.90,
            preview_primary_target_kg=[1200.0, 800.0, 1000.0],
            preview_primary_enabled=1,
        )

        result = policy._apply_forecast_safe_deadband(
            state(), rising_preview, plant_info_prev=plant, current_time=11.0
        )

        self.assertTrue(result["active"])
        self.assertEqual(result["sync_future_rise_ok"], 0)
        self.assertEqual(result["provider_replan_requested"], 1)
        self.assertEqual(result["provider_sync_reason"], "future_pressure_rising")
        self.assertEqual(provider.exit_calls, 1)

    def test_policy_keeps_synced_hold_when_posture_temporarily_moves_away(self):
        provider = self.Provider()
        policy = ClosedLoopPolicy(
            controller=self.Controller(),
            preview_trim_provider=provider,
            primary_safety_cfg={
                "forecast_safe_deadband_enabled": True,
                "forecast_safe_deadband_gate_mode": "forecast_veto",
                "forecast_safe_deadband_sync_provider_target": True,
                "forecast_safe_deadband_sync_require_recovering": True,
                "forecast_safe_deadband_sync_require_pump_demand": True,
                "forecast_safe_deadband_sync_future_rise_max": 0.10,
                "forecast_safe_deadband_posture_gate_deg": 1.5,
                "forecast_safe_deadband_event_probability_max": 0.95,
            },
        )
        plant = {
            "tank_masses": np.array([1000.0, 1000.0, 1000.0]),
            "pump_rate_cmd_m3_min": np.array([0.1, 0.0, 0.0]),
        }
        first_preview = preview(
            preview_primary_target_kg=[1200.0, 800.0, 1000.0],
            preview_primary_enabled=1,
        )
        policy._apply_forecast_safe_deadband(
            state(), first_preview, plant_info_prev=plant, current_time=10.0
        )
        continuing_preview = preview(
            preview_primary_target_kg=[1000.0, 1000.0, 1000.0],
            preview_primary_enabled=1,
        )

        result = policy._apply_forecast_safe_deadband(
            state(pitch_rate_deg_s=0.2),
            continuing_preview,
            plant_info_prev=plant,
            current_time=11.0,
        )

        self.assertTrue(result["active"])
        self.assertEqual(result["posture_recovering"], 0)
        self.assertEqual(result["provider_target_synced"], 1)
        self.assertEqual(result["provider_replan_requested"], 0)
        self.assertEqual(provider.exit_calls, 0)

    def test_policy_does_not_sync_large_target_error_while_pump_is_idle(self):
        provider = self.Provider()
        policy = ClosedLoopPolicy(
            controller=self.Controller(),
            preview_trim_provider=provider,
            primary_safety_cfg={
                "forecast_safe_deadband_enabled": True,
                "forecast_safe_deadband_gate_mode": "forecast_veto",
                "forecast_safe_deadband_sync_provider_target": True,
                "forecast_safe_deadband_sync_require_recovering": True,
                "forecast_safe_deadband_sync_require_pump_demand": True,
                "forecast_safe_deadband_sync_future_rise_max": 0.10,
                "forecast_safe_deadband_posture_gate_deg": 1.5,
                "forecast_safe_deadband_event_probability_max": 0.95,
            },
        )
        preview_values = preview(
            preview_primary_target_kg=[42000.0, 42000.0, 42000.0],
            preview_primary_enabled=1,
        )
        plant = {
            "tank_masses": np.array([0.0, 0.0, 0.0]),
            "pump_rate_cmd_m3_min": np.zeros(3),
        }

        result = policy._apply_forecast_safe_deadband(
            state(), preview_values, plant_info_prev=plant, current_time=0.0
        )

        self.assertTrue(result["active"])
        self.assertEqual(result["pump_demand_active"], 0)
        self.assertEqual(result["actionable_pump_demand"], 0)
        self.assertEqual(result["provider_target_synced"], 0)
        self.assertEqual(result["provider_sync_reason"], "no_pump_demand")
        self.assertEqual(provider.sync_calls, 0)


if __name__ == "__main__":
    unittest.main()
