import unittest
from dataclasses import replace

import numpy as np

from wind_prediction.action_plan import TargetOperation
from wind_prediction.controller_core import (
    ControlAction,
    ControlCoreConfig,
    ControlObservation,
    build_control_context,
    decide_control_cycle,
    execution_request_for_action,
)
from wind_prediction.execution_rollout import (
    ExecutionRolloutConfig,
    ExecutionRolloutState,
    ExecutionTargetOperation,
)
from wind_prediction.forecast_action_policy import ForecastActionPolicyConfig
from wind_prediction.forecast_evidence import ForecastEvidence


def _execution_state(*, target_offset_kg=(0.0, 0.0, 0.0)):
    masses = np.array([900_000.0, 900_000.0, 900_000.0])
    return ExecutionRolloutState(
        masses_kg=masses,
        target_masses_kg=masses + np.asarray(target_offset_kg, dtype=float),
        primary_target_kg=masses + np.asarray(target_offset_kg, dtype=float),
        pump_rates_m3_min=np.zeros(3),
        pump_latched=np.zeros(3, dtype=bool),
    )


def _observation(
    *,
    posture_deg=(0.0, 0.0),
    posture_rate_deg_s=(0.0, 0.0),
    current_wind_uv_ms=(0.0, -10.0),
    target_offset_kg=(0.0, 0.0, 0.0),
):
    return ControlObservation(
        time_s=0.0,
        posture_deg=posture_deg,
        posture_rate_deg_s=posture_rate_deg_s,
        current_wind_uv_ms=current_wind_uv_ms,
        execution_state=_execution_state(target_offset_kg=target_offset_kg),
    )


def _evidence(
    uv_ms,
    *,
    reliability=1.0,
    provides_future_preview=True,
    event_probability=1.0,
):
    vectors = np.asarray(uv_ms, dtype=float)
    return ForecastEvidence(
        source="unit_test",
        model_version="test",
        origin_time="2026-08-12T00:00:00",
        sample_period_s=600.0,
        uv_ms=vectors,
        lead_reliability=np.full(vectors.shape[0], reliability),
        event_probs={
            "attention_event_0_20m": event_probability,
            "attention_event_20_40m": event_probability,
            "attention_event_40_60m": event_probability,
        },
        provides_future_preview=provides_future_preview,
    )


def _config(**changes):
    base = ControlCoreConfig(
        stage_duration_s=1200.0,
        stage_count=3,
        execution=ExecutionRolloutConfig(
            block_duration_s=1200.0,
            internal_step_s=60.0,
            target_slew_enabled=False,
            tank_capacity_kg=1_896_250.0,
        ),
        forecast_policy=ForecastActionPolicyConfig(
            enabled=True,
            stage_duration_s=1200.0,
            high_impact_reliability_min=0.65,
            high_impact_event_probability_min=0.60,
        ),
    )
    return replace(base, **changes)


class ControllerCoreTests(unittest.TestCase):
    def test_constant_future_wind_adds_no_forecast_increment(self):
        context = build_control_context(
            _observation(),
            _evidence([(0.0, -10.0)] * 6),
            _config(),
        )

        for stage in context.stages:
            np.testing.assert_allclose(stage.forecast_increment_deg, np.zeros(2))
            np.testing.assert_allclose(stage.combined_demand_deg, np.zeros(2))

    def test_measured_posture_uses_physical_deadzone_before_decay(self):
        context = build_control_context(
            _observation(posture_deg=(2.0, -1.3)),
            _evidence([(0.0, -10.0)] * 6),
            _config(posture_decay=0.5),
        )

        np.testing.assert_allclose(context.feedback_demand_deg, [1.0, -0.5])
        np.testing.assert_allclose(context.stages[0].combined_demand_deg, [1.0, -0.5])
        np.testing.assert_allclose(context.stages[1].combined_demand_deg, [0.5, -0.25])

    def test_prediction_unavailable_keeps_feedback_actions_but_blocks_high_impact(self):
        evidence = _evidence(
            [(0.0, -10.0)] * 6,
            provides_future_preview=False,
        )
        context = build_control_context(
            _observation(posture_deg=(2.0, 0.0)),
            evidence,
            _config(),
        )
        near = context.stages[0].authorizations

        self.assertTrue(near[ControlAction.NORMAL].allowed)
        self.assertTrue(near[ControlAction.REDUCED].allowed)
        self.assertTrue(near[ControlAction.CONTINUE_TARGET].allowed)
        self.assertFalse(near[ControlAction.STRENGTHEN].allowed)
        self.assertFalse(near[ControlAction.RELEASE_TARGET].allowed)
        self.assertFalse(near[ControlAction.REVERSE].allowed)

    def test_continue_and_release_have_distinct_execution_semantics(self):
        state = _execution_state(target_offset_kg=(10_000.0, -5_000.0, -5_000.0))

        continuing = execution_request_for_action(
            ControlAction.CONTINUE_TARGET,
            state=state,
            action_vector_deg=np.zeros(2),
            config=_config(),
        )
        releasing = execution_request_for_action(
            ControlAction.RELEASE_TARGET,
            state=state,
            action_vector_deg=np.zeros(2),
            config=_config(),
        )

        self.assertEqual(continuing.operation, ExecutionTargetOperation.TRACK)
        np.testing.assert_allclose(continuing.target_masses_kg, state.primary_target_kg)
        self.assertEqual(
            releasing.operation,
            ExecutionTargetOperation.RELEASE_TO_CURRENT,
        )
        self.assertIsNone(releasing.target_masses_kg)

    def test_high_posture_priority_selects_a_corrective_action(self):
        config = _config(
            posture_priority_envelope_deg=(3.0, 3.0),
            w_residual=80.0,
            w_terminal_residual=80.0,
            w_pump_volume=0.2,
        )
        decision = decide_control_cycle(
            _observation(posture_deg=(4.5, 0.0)),
            _evidence([(0.0, -8.0)] * 6),
            config,
        )

        self.assertIn(
            decision.action,
            {ControlAction.NORMAL, ControlAction.STRENGTHEN},
        )
        self.assertGreater(decision.action_vector_deg[0], 0.0)
        self.assertEqual(decision.target_operation, TargetOperation.SET_DELTA)

    def test_decision_trace_records_all_ranked_candidates_and_executable_target(self):
        decision = decide_control_cycle(
            _observation(posture_deg=(2.2, 0.0)),
            _evidence([(0.0, -10.0)] * 6),
            _config(),
        )

        self.assertGreater(len(decision.ranked_candidates), 1)
        self.assertEqual(
            decision.ranked_candidates[0].sequence[0],
            decision.action,
        )
        self.assertEqual(len(decision.target_masses_kg), 3)
        self.assertTrue(np.all(np.asarray(decision.target_masses_kg) >= 0.0))
        self.assertTrue(
            np.all(
                np.asarray(decision.target_masses_kg)
                <= decision.config.execution.tank_capacity_kg
            )
        )
        trace = decision.as_trace()
        self.assertEqual(trace["selected_action"], decision.action.value)
        self.assertEqual(trace["candidate_count"], len(decision.ranked_candidates))
        self.assertIn("forecast", trace)
        self.assertIn("first_execution", trace)
        self.assertIn("execution_request", trace)
        self.assertEqual(
            trace["execution_request"]["operation"],
            decision.execution_request.operation.value,
        )
        self.assertEqual(len(trace["stages"]), 3)

    def test_small_forecast_only_demand_does_not_create_a_pump_target(self):
        decision = decide_control_cycle(
            _observation(posture_deg=(0.0, 0.0)),
            _evidence([(0.0, -10.5)] * 6),
            _config(minimum_action_demand_ratio=0.20),
        )

        self.assertEqual(decision.action, ControlAction.CONTINUE_TARGET)
        self.assertEqual(decision.target_operation, TargetOperation.CONTINUE)
        self.assertEqual(
            decision.execution_request.operation,
            ExecutionTargetOperation.TRACK,
        )
        np.testing.assert_allclose(
            decision.target_masses_kg,
            _observation().execution_state.actual_masses_kg,
        )
        np.testing.assert_allclose(
            decision.execution_request.target_masses_kg,
            _observation().execution_state.actual_masses_kg,
        )


if __name__ == "__main__":
    unittest.main()
