from __future__ import annotations

import json
import unittest
from datetime import datetime

import numpy as np

from wind_prediction.provider_runtime import ProviderRuntimeMixin
from wind_prediction.provider_telemetry import ProviderTelemetryMixin


class _TraceProvider(ProviderRuntimeMixin, ProviderTelemetryMixin):
    pass


class _ReplayDataset:
    @staticmethod
    def sample_for_history_end(_timestamp):
        return object()


class _RuntimeChainProbe(ProviderRuntimeMixin, ProviderTelemetryMixin):
    def __init__(self) -> None:
        self.replay_dataset = _ReplayDataset()
        self.forecast_adapter = None
        self._target_action = "active_small"
        self._target_pitch_planner_frame = 0.1
        self._target_roll_planner_frame = 0.2
        self.setpoint_bias_sign = 1.0
        self.forecast_advised_economy_hold_current_on_candidate = False
        self._forecast_advised_economy_candidate = False
        self.objective_mode = "economic"
        self.captured_trace = ()

    def _reset_forced_prefix_state(self) -> None:
        self._forced_prefix_active = False
        self._forced_prefix_mode_active = ""
        self._forced_source_label = ""

    @staticmethod
    def _bucket_timestamp(_current_time):
        return datetime(2026, 1, 1)

    @staticmethod
    def _run_planner(_sample, _plant_info, *, forecast_has_future):
        del forecast_has_future
        return (
            {
                "sequence": ("active_small", "hold", "hold"),
                "best_scalar_cost": 1.0,
                "selection_reason": "test",
            },
            np.array([0.1, 0.2]),
            "learned",
            (object(), object(), object()),
        )

    def _apply_forced_prefix_action(
        self, mode, _bucket, _action, _avec, _blocks, _plant_info
    ):
        self._forced_prefix_active = True
        self._forced_prefix_mode_active = mode
        self._forced_source_label = f"test_{mode}"
        if mode == "raw_action":
            return "active_medium", np.array([0.3, 0.2])
        return "active_medium", np.array([0.25, 0.2])

    def _relief_medium_cap_action(self, _action, _avec, _blocks, _state):
        self._relief_medium_cap_reason = "medium_to_small"
        return "active_small", np.array([0.15, 0.2])

    def _far_horizon_relief_gate_action(self, action, _avec, _blocks, _state):
        self._far_horizon_relief_gate_reason = "vector_only_cap"
        return action, np.array([0.12, 0.2])

    def _observe_forecast_advised_economy_decision(self, *args, **kwargs):
        del args, kwargs
        self._forecast_advised_economy_reason = "not_candidate"

    @staticmethod
    def _should_reset_event(*_args, **_kwargs):
        return True

    @staticmethod
    def _stall_refresh_action_vec(_avec, _plant_info, _current_time):
        return np.array([0.3, 0.2]), True, "stall_axis_refresh"

    @staticmethod
    def _gusty_oscillation_soft_cap_action(_avec, _plant_info):
        return np.array([0.2, 0.2]), True, "gust_cap"

    @staticmethod
    def _direction_reversal_layered_entry_veto_active(_plant_info, _blocks):
        return True, "direction_reversal"

    @staticmethod
    def _finalize_supervised_action(*, plan, best, action, avec):
        del plan, best
        return action, np.asarray(avec, dtype=float), None, False

    @staticmethod
    def _observe_active_effectiveness_bucket(*_args, **_kwargs):
        return None

    @staticmethod
    def _direction_reversal_layered_warning_active(_plant_info, _blocks):
        return False, "not_active"

    @staticmethod
    def _commit_planned_target(**_kwargs):
        return None

    def _append_replan_record(self, **_kwargs):
        self.captured_trace = tuple(self._decision_trace_steps)


class ProviderDecisionTraceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.provider = _TraceProvider()
        self.provider._decision_trace_steps = ()

    def test_same_action_with_changed_vector_is_a_modification(self) -> None:
        step = self.provider._record_decision_trace_step(
            name="stall_vector_rewrite",
            input_action="small",
            input_vector=np.array([0.2, -0.1]),
            output_action="small",
            output_vector=np.array([0.3, -0.1]),
            reason="axis_refresh",
        )

        self.assertTrue(step.changed)
        self.assertEqual(step.input_action, step.output_action)

    def test_all_post_ranking_steps_serialize_in_order(self) -> None:
        names = (
            "raw_forced",
            "medium_cap",
            "far_cap",
            "final_forced",
            "economy_hold",
            "stall_vector_rewrite",
            "gust_cap",
            "direction_reversal_veto",
        )
        action = "small"
        vector = np.array([0.1, 0.2])
        for index, name in enumerate(names):
            output_vector = vector.copy()
            if name in {"far_cap", "stall_vector_rewrite", "gust_cap"}:
                output_vector[0] += 0.1
            self.provider._record_decision_trace_step(
                name=name,
                input_action=action,
                input_vector=vector,
                output_action=action,
                output_vector=output_vector,
                reason=f"reason_{index}",
            )
            vector = output_vector

        fields = self.provider._decision_trace_log_fields()
        payload = json.loads(fields["decision_trace_json"])

        self.assertEqual([entry["name"] for entry in payload], list(names))
        self.assertEqual(fields["decision_trace_modification_count"], 3)
        self.assertNotIn(": ", fields["decision_trace_json"])
        self.assertNotIn(", ", fields["decision_trace_json"])
        self.assertEqual(payload[2]["metadata"]["input_vector"], [0.1, 0.2])
        self.assertEqual(payload[2]["metadata"]["output_vector"], [0.2, 0.2])

    def test_unchanged_step_does_not_increment_modification_count(self) -> None:
        self.provider._record_decision_trace_step(
            name="economy_hold",
            input_action="hold",
            input_vector=[0.0, 0.0],
            output_action="hold",
            output_vector=[0.0, 0.0],
            reason="not_applied",
        )

        fields = self.provider._decision_trace_log_fields(prefix="preview_")

        self.assertEqual(fields["preview_decision_trace_modification_count"], 0)
        self.assertEqual(
            json.loads(fields["preview_decision_trace_json"])[0]["changed"],
            0,
        )

    def test_runtime_chain_records_all_eight_post_ranking_steps(self) -> None:
        provider = _RuntimeChainProbe()

        provider._update_planner_bucket(
            bucket=1,
            current_time=600.0,
            state=object(),
            plant_info_now={},
            plant_info_prev={},
            replanned=True,
        )

        self.assertEqual(
            [step.name for step in provider.captured_trace],
            [
                "raw_forced",
                "medium_cap",
                "far_cap",
                "final_forced",
                "economy_hold",
                "stall_vector_rewrite",
                "gust_cap",
                "direction_reversal_veto",
            ],
        )
        fields = provider._decision_trace_log_fields()
        self.assertEqual(fields["decision_trace_modification_count"], 7)
        self.assertTrue(provider.captured_trace[2].changed)
        self.assertEqual(
            provider.captured_trace[2].input_action,
            provider.captured_trace[2].output_action,
        )


if __name__ == "__main__":
    unittest.main()
