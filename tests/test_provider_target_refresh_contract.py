import unittest
from types import SimpleNamespace

import numpy as np

from wind_prediction.action_plan import TargetOperation
from wind_prediction.provider_runtime import ProviderRuntimeMixin


class _Provider(ProviderRuntimeMixin):
    def __init__(self):
        self._forced_prefix_active = False
        self._forced_prefix_mode_active = "off"
        self._forced_prefix_target_update_active = False
        self._h120_scheduler_remote_risk_active = False
        self._h120_scheduler_remote_risk_trigger_count = 0
        self.gusty_hold_current_isolated_primary_enabled = False
        self.prediction_primary_enabled = True
        self.target_lifecycle_mode = "legacy"
        self._primary_target_initialized = True
        self.refresh_calls = 0
        self.resume_calls = 0

    def _relief_envelope_reset_bucket_log(self):
        pass

    def _apply_forced_prefix_action(self, _stage, _bucket, action, avec, *_args):
        return action, avec

    def _h120_scheduler_refresh(self, _plant_info):
        pass

    def _h120_pareto_mode_selector_refresh(self, _plant_info):
        pass

    def _reactive_floor_needed(self, *_args):
        return False, np.zeros(2), "disabled"

    def _oracle_preemptive_needed(self, *_args):
        return False, np.zeros(2), "disabled"

    def _h120_oracle_probe_preemptive_needed(self, *_args):
        return False, np.zeros(2), "disabled"

    def _h120_scheduler_prefloor_probe_needed(self, *_args):
        return False, np.zeros(2), "disabled"

    def _h120_axis_micro_prepare_needed(self, *_args):
        return False, np.zeros(2), "disabled"

    def _economy_budget_hold_release_needed(self, _plant_info):
        return False, "disabled"

    def _sustained_active_recompute_needed(self, _action):
        return False, "disabled"

    def _resume_paused_primary_target(self, *_args):
        self.resume_calls += 1
        return True

    def _apply_event_target_refresh(self, **_kwargs):
        self.refresh_calls += 1

    def _relief_envelope_mark_not_applicable(self):
        pass

    def _hold_relief_debt_observe_after_action(self, *_args):
        pass


class ProviderTargetRefreshContractTests(unittest.TestCase):
    def test_explicit_target_actions_map_to_distinct_lifecycle_operations(self):
        self.assertEqual(
            ProviderRuntimeMixin._target_operation_for_action("maintain_target"),
            TargetOperation.CONTINUE,
        )
        self.assertEqual(
            ProviderRuntimeMixin._target_operation_for_action("stop_execution"),
            TargetOperation.RELEASE,
        )
        self.assertEqual(
            ProviderRuntimeMixin._target_operation_for_action("hold"),
            TargetOperation.HOLD_CURRENT,
        )

    @staticmethod
    def _commit(provider, *, event_reset):
        provider._commit_planned_target(
            bucket=1,
            current_time=600.0,
            sample=None,
            state=np.zeros(6),
            plant_info={},
            blocks=[],
            previous_action="hold",
            previous_avec=np.zeros(2),
            new_action="active_small",
            first_avec=np.asarray([0.1, 0.0]),
            event_reset=event_reset,
            direction_reversal_entry_veto=False,
            direction_reversal_warning_active=False,
            planned_reactive_handoff=False,
        )

    def test_active_bucket_refresh_wins_over_paused_target_resume(self):
        provider = _Provider()

        self._commit(provider, event_reset=True)

        self.assertEqual(provider.refresh_calls, 1)
        self.assertEqual(provider.resume_calls, 0)

    def test_legacy_reuse_path_can_resume_a_compatible_paused_target(self):
        provider = _Provider()

        self._commit(provider, event_reset=False)

        self.assertEqual(provider.refresh_calls, 0)
        self.assertEqual(provider.resume_calls, 1)

    def test_planner_authoritative_mode_commits_evaluated_target_directly(self):
        provider = _Provider()
        provider.target_lifecycle_mode = "planner_authoritative"
        provider.cfg = SimpleNamespace(tank_capacity_kg=1000.0)
        provider._primary_anchor_masses_kg = np.asarray([100.0, 200.0, 300.0])
        provider._primary_target_kg = provider._primary_anchor_masses_kg.copy()
        provider._primary_delta_kg = np.zeros(3)
        provider._paused_primary_valid = True
        provider._action_plan_snapshot = {
            "target_operation": "set_delta",
            "final_target_masses_kg": [160.0, 170.0, 270.0],
        }

        provider._commit_planned_target(
            bucket=1,
            current_time=600.0,
            sample=None,
            state=np.zeros(6),
            plant_info={"tank_masses": np.asarray([110.0, 190.0, 300.0])},
            blocks=[],
            previous_action="hold",
            previous_avec=np.zeros(2),
            new_action="active_small",
            first_avec=np.asarray([0.1, 0.0]),
            event_reset=True,
            direction_reversal_entry_veto=False,
            direction_reversal_warning_active=False,
            planned_reactive_handoff=False,
        )

        np.testing.assert_allclose(
            provider._primary_target_kg,
            [160.0, 170.0, 270.0],
        )
        self.assertFalse(provider._paused_primary_valid)
        self.assertEqual(provider.refresh_calls, 0)

    def test_planner_authoritative_continue_preserves_existing_target_epoch(self):
        provider = _Provider()
        provider.target_lifecycle_mode = "planner_authoritative"
        provider.cfg = SimpleNamespace(tank_capacity_kg=1000.0)
        provider._primary_anchor_masses_kg = np.asarray([100.0, 200.0, 300.0])
        provider._primary_target_kg = np.asarray([160.0, 170.0, 270.0])
        provider._primary_delta_kg = np.asarray([60.0, -30.0, -30.0])
        provider._primary_target_last_reset_s = 300.0
        provider._action_plan_snapshot = {
            "target_operation": "continue",
            "final_target_masses_kg": [160.0, 170.0, 270.0],
        }

        provider._commit_planned_target(
            bucket=1,
            current_time=600.0,
            sample=None,
            state=np.zeros(6),
            plant_info={"tank_masses": np.asarray([120.0, 190.0, 290.0])},
            blocks=[],
            previous_action="active_small",
            previous_avec=np.asarray([0.1, 0.0]),
            new_action="maintain_target",
            first_avec=np.zeros(2),
            event_reset=False,
            direction_reversal_entry_veto=False,
            direction_reversal_warning_active=False,
            planned_reactive_handoff=False,
        )

        np.testing.assert_allclose(
            provider._primary_target_kg,
            [160.0, 170.0, 270.0],
        )
        self.assertEqual(provider._primary_target_last_reset_s, 300.0)
        self.assertTrue(provider._primary_target_reused)
        self.assertFalse(provider._primary_target_refreshed)

    def test_planner_authoritative_stop_execution_releases_to_current_masses(self):
        provider = _Provider()
        provider.target_lifecycle_mode = "planner_authoritative"
        provider.cfg = SimpleNamespace(tank_capacity_kg=1000.0)
        provider._primary_anchor_masses_kg = np.asarray([100.0, 200.0, 300.0])
        provider._primary_target_kg = np.asarray([160.0, 170.0, 270.0])
        provider._primary_delta_kg = np.asarray([60.0, -30.0, -30.0])
        provider._paused_primary_valid = True
        provider._action_plan_snapshot = {
            "target_operation": "release",
            "final_target_masses_kg": [120.0, 190.0, 290.0],
        }

        provider._commit_planned_target(
            bucket=1,
            current_time=600.0,
            sample=None,
            state=np.zeros(6),
            plant_info={"tank_masses": np.asarray([120.0, 190.0, 290.0])},
            blocks=[],
            previous_action="active_small",
            previous_avec=np.asarray([0.1, 0.0]),
            new_action="stop_execution",
            first_avec=np.zeros(2),
            event_reset=False,
            direction_reversal_entry_veto=False,
            direction_reversal_warning_active=False,
            planned_reactive_handoff=False,
        )

        np.testing.assert_allclose(provider._primary_target_kg, [120.0, 190.0, 290.0])
        np.testing.assert_allclose(provider._primary_delta_kg, np.zeros(3))
        self.assertFalse(provider._paused_primary_valid)
        self.assertTrue(provider._primary_target_refreshed)


if __name__ == "__main__":
    unittest.main()
