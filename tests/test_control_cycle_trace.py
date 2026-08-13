import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np


LEGACY_DIR = Path(__file__).resolve().parents[1] / "archive" / "legacy_fowt_control"
sys.path.insert(0, str(LEGACY_DIR))

from controllers_extras import ClosedLoopPolicy
from target_execution import describe_target_path


class _Limiter:
    def update(self, target):
        return np.asarray(target, dtype=float) - 5.0, 5.0


class ControlCycleTraceTests(unittest.TestCase):
    def test_prediction_primary_without_adjustment(self):
        trace = describe_target_path(
            primary_applied=True,
            primary_safety_fallback=False,
        )
        self.assertEqual(trace["target_intent_source"], "prediction_primary")
        self.assertEqual(trace["target_adjustment_path"], "none")
        self.assertEqual(trace["target_final_adjustment_stage"], "none")

    def test_safety_fallback_owns_intent(self):
        trace = describe_target_path(
            primary_applied=False,
            primary_safety_fallback=True,
        )
        self.assertEqual(
            trace["target_intent_source"], "posture_feedback_safety_fallback"
        )

    def test_adjustments_are_reported_in_execution_order(self):
        trace = describe_target_path(
            primary_applied=True,
            primary_safety_fallback=False,
            feedforward_delta_kg=2.0,
            preview_suppression_delta_kg=3.0,
            reactive_suppression_delta_kg=4.0,
            target_slew_delta_kg=5.0,
            deadband_release_active=True,
        )
        self.assertEqual(
            trace["target_adjustment_path"],
            "prediction_feedforward|preview_suppression|reactive_suppression|"
            "target_slew_limiter|deadband_release",
        )
        self.assertEqual(
            trace["target_final_adjustment_stage"], "deadband_release"
        )

    def test_target_transaction_applies_stages_in_order(self):
        policy = object.__new__(ClosedLoopPolicy)
        policy.controller = SimpleNamespace(max_mass=1000.0)
        policy.target_slew_limiter = _Limiter()
        policy._reactive_pump_suppression_update = lambda **kwargs: (
            np.asarray(kwargs["m_cmd_reference"], dtype=float) - 2.0,
            {"delta_mean_kg": 2.0},
        )
        policy._deadband_target_release_update = lambda **kwargs: (
            np.asarray(kwargs["m_cmd_applied"], dtype=float) - 1.0,
            {"active": 1, "delta_mean_kg": 1.0},
        )

        result = policy._commit_target_transaction(
            state=np.zeros(12),
            plant_info_prev=None,
            ctrl_dbg={},
            preview_trim_bias={"preview_mass_ff_kg": [10.0, 20.0, 30.0]},
            primary_safety_dbg={"fallback": 0},
            primary_applied=True,
            primary_target=np.array([100.0, 200.0, 300.0]),
        )

        np.testing.assert_allclose(result["target_pre_execution"], [108, 218, 328])
        np.testing.assert_allclose(
            result["target_after_slew_limiter"], [103, 213, 323]
        )
        np.testing.assert_allclose(result["final_target"], [102, 212, 322])
        self.assertEqual(
            result["trace"]["target_adjustment_path"],
            "prediction_feedforward|reactive_suppression|target_slew_limiter|"
            "deadband_release",
        )

    def test_safety_fallback_removes_prediction_feedforward(self):
        policy = object.__new__(ClosedLoopPolicy)
        policy.controller = SimpleNamespace(max_mass=1000.0)
        policy.target_slew_limiter = None
        policy._reactive_pump_suppression_update = lambda **kwargs: (
            np.asarray(kwargs["m_cmd_reference"], dtype=float),
            {"delta_mean_kg": 0.0},
        )
        policy._deadband_target_release_update = lambda **kwargs: (
            np.asarray(kwargs["m_cmd_applied"], dtype=float),
            {"active": 0, "delta_mean_kg": 0.0},
        )

        result = policy._commit_target_transaction(
            state=np.zeros(12),
            plant_info_prev=None,
            ctrl_dbg={},
            preview_trim_bias={"preview_mass_ff_kg": [10.0, 20.0, 30.0]},
            primary_safety_dbg={"fallback": 1},
            primary_applied=False,
            primary_target=np.array([100.0, 200.0, 300.0]),
        )

        np.testing.assert_allclose(result["final_target"], [100, 200, 300])
        self.assertEqual(
            result["trace"]["target_intent_source"],
            "posture_feedback_safety_fallback",
        )
        self.assertEqual(result["trace"]["target_adjustment_path"], "none")

    def test_zero_change_prediction_target_still_owns_hold(self):
        policy = object.__new__(ClosedLoopPolicy)
        policy.controller = SimpleNamespace(max_mass=1000.0)
        policy.forecast_safe_deadband_enabled = False
        policy.forecast_safe_deadband_sync_provider_target = False
        policy.forecast_safe_deadband_release_primary_target = False
        policy.forecast_safe_deadband_allowed_actions = ("hold",)
        policy.forecast_safe_deadband_reject_unsafe_hold = False
        policy._primary_safety_update = lambda **kwargs: {"fallback": 0}

        result = policy._select_primary_target(
            state=np.zeros(12),
            plant_info_prev={"tank_masses": np.array([100.0, 200.0, 300.0])},
            current_time=0.0,
            preview_trim_bias={
                "preview_primary_enabled": 1,
                "preview_primary_active": 0,
                "preview_primary_target_kg": [100.0, 200.0, 300.0],
                "preview_primary_action": "hold",
            },
            forecast_safe_deadband_dbg={"active": 0},
            feedback_target=np.array([400.0, 500.0, 600.0]),
        )

        self.assertEqual(result["active"], 0)
        self.assertEqual(result["applied"], 1)
        np.testing.assert_allclose(result["selected_target"], [100.0, 200.0, 300.0])


if __name__ == "__main__":
    unittest.main()
