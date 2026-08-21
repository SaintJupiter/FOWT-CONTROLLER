import unittest
from dataclasses import replace

import numpy as np

from wind_prediction.action_plan import TargetOperation
from wind_prediction.controller_core import (
    ActionVectorSource,
    ControlAction,
    ControlCoreConfig,
    ControlObservation,
    _action_vector,
    _candidate_score_terms,
    action_semantics_for,
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
    def test_score_terms_preserve_the_existing_normalized_ranking_formula(self):
        config = _config()
        terms = _candidate_score_terms(
            residual_cost=1.5,
            terminal_residual_cost=0.25,
            transferred_volume_m3=360.0,
            active_time_s=1800.0,
            starts=3,
            direction_switches=2,
            config=config,
        )

        reference_volume = (
            config.action_mass_quantum_kg
            / config.execution.water_density_kg_m3
        )
        self.assertAlmostEqual(terms.residual, config.w_residual * 1.5)
        self.assertAlmostEqual(
            terms.terminal_residual,
            config.w_terminal_residual * 0.25,
        )
        self.assertAlmostEqual(
            terms.pump_volume,
            config.w_pump_volume * 360.0 / reference_volume,
        )
        self.assertAlmostEqual(
            terms.pump_runtime,
            config.w_pump_runtime * 1800.0 / (3.0 * config.stage_count * config.stage_duration_s),
        )
        self.assertAlmostEqual(
            terms.starts,
            config.w_starts * 3.0 / (3.0 * config.stage_count),
        )
        self.assertAlmostEqual(
            terms.direction_switches,
            config.w_direction_switches * 2.0 / (3.0 * config.stage_count),
        )

    def test_action_semantics_make_target_lifecycle_and_vector_source_explicit(self):
        expected = {
            ControlAction.CONTINUE_TARGET: (
                TargetOperation.CONTINUE,
                ActionVectorSource.NONE,
                False,
            ),
            ControlAction.RELEASE_TARGET: (
                TargetOperation.RELEASE,
                ActionVectorSource.NONE,
                False,
            ),
            ControlAction.REDUCED: (
                TargetOperation.SET_DELTA,
                ActionVectorSource.RESIDUAL_DEMAND,
                True,
            ),
            ControlAction.NORMAL: (
                TargetOperation.SET_DELTA,
                ActionVectorSource.RESIDUAL_DEMAND,
                True,
            ),
            ControlAction.STRENGTHEN: (
                TargetOperation.SET_DELTA,
                ActionVectorSource.RESIDUAL_DEMAND,
                True,
            ),
            ControlAction.REVERSE: (
                TargetOperation.SET_DELTA,
                ActionVectorSource.PREVIOUS_ACTION_VECTOR,
                True,
            ),
        }

        for action, (operation, vector_source, requires_demand) in expected.items():
            semantics = action_semantics_for(action)
            self.assertEqual(semantics.target_operation, operation)
            self.assertEqual(semantics.vector_source, vector_source)
            self.assertEqual(semantics.requires_nonzero_demand, requires_demand)

    def test_constant_future_wind_adds_no_forecast_increment(self):
        context = build_control_context(
            _observation(),
            _evidence([(0.0, -10.0)] * 6),
            _config(),
        )

        for stage in context.stages:
            np.testing.assert_allclose(stage.forecast_increment_deg, np.zeros(2))
            np.testing.assert_allclose(stage.combined_demand_deg, np.zeros(2))

    def test_full_reliability_preserves_legacy_forecast_increment_mapping(self):
        context = build_control_context(
            _observation(current_wind_uv_ms=(0.0, -10.0)),
            _evidence([(0.0, -12.0)] * 6, reliability=1.0),
            _config(wind_reference_ms=10.0, wind_effect_cap=2.0),
        )

        np.testing.assert_allclose(
            context.stages[0].forecast_increment_deg,
            (0.44, 0.0),
            atol=1e-12,
        )

    def test_implicit_legacy_demand_scale_preserves_existing_controller_result(self):
        implicit = _config(deadband_deg=(1.2, 0.9))
        explicit = _config(
            deadband_deg=(1.2, 0.9),
            legacy_demand_axis_scale_deg=(1.2, 0.9),
        )
        observation = _observation(posture_deg=(2.5, -1.8))
        evidence = _evidence([(0.0, -12.0)] * 6)

        implicit_context = build_control_context(observation, evidence, implicit)
        explicit_context = build_control_context(observation, evidence, explicit)
        implicit_decision = decide_control_cycle(observation, evidence, implicit)
        explicit_decision = decide_control_cycle(observation, evidence, explicit)

        self.assertIsNone(implicit.legacy_demand_axis_scale_deg)
        self.assertEqual(
            implicit.resolved_legacy_demand_axis_scale_deg,
            explicit.resolved_legacy_demand_axis_scale_deg,
        )
        for implicit_stage, explicit_stage in zip(
            implicit_context.stages,
            explicit_context.stages,
        ):
            np.testing.assert_allclose(
                implicit_stage.combined_demand_deg,
                explicit_stage.combined_demand_deg,
            )
        self.assertEqual(implicit_decision.action, explicit_decision.action)
        self.assertEqual(
            implicit_decision.target_masses_kg,
            explicit_decision.target_masses_kg,
        )
        self.assertAlmostEqual(implicit_decision.score, explicit_decision.score)
        self.assertEqual(
            [candidate.sequence for candidate in implicit_decision.ranked_candidates],
            [candidate.sequence for candidate in explicit_decision.ranked_candidates],
        )

    def test_deadband_change_leaves_legacy_proxy_scale_fixed_when_explicit(self):
        observation = _observation(posture_deg=(2.5, -1.8))
        evidence = _evidence([(0.0, -12.0)] * 6)
        lower_deadband = build_control_context(
            observation,
            evidence,
            _config(
                deadband_deg=(1.0, 0.8),
                legacy_demand_axis_scale_deg=(1.0, 0.8),
            ),
        )
        higher_deadband = build_control_context(
            observation,
            evidence,
            _config(
                deadband_deg=(1.8, 1.4),
                legacy_demand_axis_scale_deg=(1.0, 0.8),
            ),
        )

        self.assertNotEqual(
            lower_deadband.feedback_demand_deg,
            higher_deadband.feedback_demand_deg,
        )
        for lower_stage, higher_stage in zip(
            lower_deadband.stages,
            higher_deadband.stages,
        ):
            np.testing.assert_allclose(
                lower_stage.forecast_increment_deg,
                higher_stage.forecast_increment_deg,
            )

    def test_legacy_demand_scale_change_leaves_feedback_deadband_fixed(self):
        observation = _observation(posture_deg=(2.5, -1.8))
        evidence = _evidence([(0.0, -12.0)] * 6)
        smaller_scale = build_control_context(
            observation,
            evidence,
            _config(
                deadband_deg=(1.0, 0.8),
                legacy_demand_axis_scale_deg=(0.5, 0.4),
            ),
        )
        larger_scale = build_control_context(
            observation,
            evidence,
            _config(
                deadband_deg=(1.0, 0.8),
                legacy_demand_axis_scale_deg=(1.5, 1.2),
            ),
        )

        self.assertEqual(
            smaller_scale.feedback_demand_deg,
            larger_scale.feedback_demand_deg,
        )
        self.assertGreater(
            np.linalg.norm(
                np.asarray(larger_scale.stages[0].forecast_increment_deg)
                - np.asarray(smaller_scale.stages[0].forecast_increment_deg)
            ),
            0.0,
        )

    def test_legacy_demand_scale_controls_target_mass_mapping(self):
        state = _execution_state()
        action_vector = np.array([0.5, 0.0])
        smaller_scale = _config(
            deadband_deg=(1.0, 0.8),
            legacy_demand_axis_scale_deg=(0.5, 0.8),
        )
        larger_scale = _config(
            deadband_deg=(1.0, 0.8),
            legacy_demand_axis_scale_deg=(1.0, 0.8),
        )

        smaller_request = execution_request_for_action(
            ControlAction.NORMAL,
            state=state,
            action_vector_deg=action_vector,
            config=smaller_scale,
        )
        larger_request = execution_request_for_action(
            ControlAction.NORMAL,
            state=state,
            action_vector_deg=action_vector,
            config=larger_scale,
        )

        smaller_delta = (
            np.asarray(smaller_request.target_masses_kg) - state.actual_masses_kg
        )
        larger_delta = (
            np.asarray(larger_request.target_masses_kg) - state.actual_masses_kg
        )
        self.assertGreater(np.linalg.norm(smaller_delta), np.linalg.norm(larger_delta))

    def test_reliability_gates_high_impact_actions_without_scaling_forecast_demand(self):
        high_reliability = build_control_context(
            _observation(),
            _evidence([(0.0, -12.0)] * 6, reliability=0.9),
            _config(),
        )
        low_reliability = build_control_context(
            _observation(),
            _evidence([(0.0, -12.0)] * 6, reliability=0.5),
            _config(),
        )

        np.testing.assert_allclose(
            high_reliability.stages[0].forecast_increment_deg,
            low_reliability.stages[0].forecast_increment_deg,
        )
        self.assertTrue(
            high_reliability.stages[0].authorizations[
                ControlAction.STRENGTHEN
            ].allowed
        )
        self.assertFalse(
            low_reliability.stages[0].authorizations[
                ControlAction.STRENGTHEN
            ].allowed
        )

    def test_direction_inconsistency_is_recorded_without_new_demand_gate(self):
        context = build_control_context(
            _observation(),
            _evidence(
                [
                    (0.0, -12.0),
                    (0.0, 12.0),
                    (0.0, -12.0),
                    (0.0, 12.0),
                    (0.0, -12.0),
                    (0.0, 12.0),
                ]
            ),
            _config(),
        )
        first = context.stages[0]

        self.assertFalse(first.forecast_direction_consistent)
        self.assertLess(first.forecast_direction_consistency, 0.8)
        self.assertEqual(
            first.forecast_demand_reason,
            "forecast_direction_inconsistent_recorded",
        )
        self.assertGreater(abs(first.forecast_increment_deg[0]), 0.0)

    def test_policy_disabled_is_recorded_without_disabling_future_demand(self):
        context = build_control_context(
            _observation(),
            _evidence([(0.0, -12.0)] * 6),
            _config(
                forecast_policy=ForecastActionPolicyConfig(
                    enabled=False,
                    stage_duration_s=1200.0,
                )
            ),
        )

        self.assertEqual(
            context.stages[0].forecast_demand_reason,
            "forecast_policy_disabled",
        )
        self.assertGreater(abs(context.stages[0].forecast_increment_deg[0]), 0.0)

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

    def test_reverse_uses_the_previous_action_vector_for_a_new_target(self):
        config = _config(reverse_ratio=0.12)
        state = _execution_state()
        vector = _action_vector(
            ControlAction.REVERSE,
            demand_deg=np.array((4.0, 0.0)),
            previous_action_vector_deg=np.array((0.0, -0.4)),
            config=config,
        )
        request = execution_request_for_action(
            ControlAction.REVERSE,
            state=state,
            action_vector_deg=vector,
            config=config,
        )

        self.assertGreater(vector[1], 0.0)
        self.assertAlmostEqual(vector[0], 0.0, places=12)
        self.assertEqual(request.operation, ExecutionTargetOperation.TRACK)
        self.assertFalse(
            np.allclose(request.target_masses_kg, state.actual_masses_kg)
        )

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
        self.assertIn("selection", trace)
        self.assertEqual(
            trace["selection"]["selected_sequence"][0],
            decision.action.value,
        )
        self.assertTrue(
            trace["selection"]["selected_action_authorization"]["allowed"]
        )
        self.assertEqual(
            trace["selection"]["selected_action_semantics"]["target_operation"],
            decision.target_operation.value,
        )
        score_terms = trace["selection"]["score_terms"]
        self.assertAlmostEqual(score_terms["total"], decision.score)
        self.assertAlmostEqual(
            sum(
                score_terms[name]
                for name in (
                    "residual",
                    "terminal_residual",
                    "pump_volume",
                    "pump_runtime",
                    "starts",
                    "direction_switches",
                )
            ),
            decision.score,
        )
        self.assertEqual(
            trace["selection"]["score_weights"]["pump_volume"],
            decision.config.w_pump_volume,
        )
        ranking = trace["candidate_ranking_top"]
        self.assertGreaterEqual(len(ranking), 2)
        self.assertAlmostEqual(ranking[0]["score_terms"]["total"], ranking[0]["score"])
        self.assertAlmostEqual(
            ranking[0]["score_terms"]["total"],
            trace["selection"]["score_terms"]["total"],
        )
        self.assertAlmostEqual(ranking[1]["score_terms"]["total"], ranking[1]["score"])
        self.assertEqual(
            trace["execution_request"]["operation"],
            decision.execution_request.operation.value,
        )
        self.assertEqual(len(trace["stages"]), 3)
        self.assertIn("action_authorizations", trace["stages"][0])
        self.assertEqual(
            trace["stages"][0]["action_authorizations"]["normal"]["reason"],
            "feedback_action_available",
        )
        forecast_evidence = trace["stages"][0]["forecast_policy_evidence"]
        self.assertTrue(forecast_evidence["available"])
        self.assertTrue(forecast_evidence["policy_evaluated"])
        self.assertAlmostEqual(forecast_evidence["mean_speed_ms"], 10.0)
        self.assertAlmostEqual(forecast_evidence["event_probability"], 1.0)
        self.assertTrue(forecast_evidence["direction_consistent"])
        self.assertTrue(forecast_evidence["trend"]["sustained"])

    def test_trace_marks_stage_forecast_policy_evidence_unavailable_without_preview(self):
        decision = decide_control_cycle(
            _observation(posture_deg=(2.2, 0.0)),
            _evidence(
                [(0.0, -10.0)] * 6,
                provides_future_preview=False,
            ),
            _config(),
        )

        self.assertIsNone(
            decision.as_trace()["stages"][0]["forecast_policy_evidence"]
        )

    def test_forecast_stage_bounds_follow_timestamped_forecast_points(self):
        context = build_control_context(
            _observation(),
            _evidence([(0.0, -10.0)] * 6),
            _config(),
        )

        self.assertEqual(
            [(stage.lead_start_s, stage.lead_end_s) for stage in context.stages],
            [(600.0, 1200.0), (1800.0, 2400.0), (3000.0, 3600.0)],
        )

    def test_forecast_stage_bounds_remain_timestamped_when_policy_is_disabled(self):
        context = build_control_context(
            _observation(),
            _evidence([(0.0, -10.0)] * 6),
            _config(
                forecast_policy=ForecastActionPolicyConfig(
                    enabled=False,
                    stage_duration_s=1200.0,
                )
            ),
        )

        self.assertEqual(
            [(stage.lead_start_s, stage.lead_end_s) for stage in context.stages],
            [(600.0, 1200.0), (1800.0, 2400.0), (3000.0, 3600.0)],
        )

    def test_no_preview_stage_bounds_remain_planning_intervals(self):
        context = build_control_context(
            _observation(),
            _evidence(
                [(0.0, -10.0)] * 6,
                provides_future_preview=False,
            ),
            _config(),
        )

        self.assertEqual(
            [(stage.lead_start_s, stage.lead_end_s) for stage in context.stages],
            [(0.0, 1200.0), (1200.0, 2400.0), (2400.0, 3600.0)],
        )

    def test_trace_stage_bounds_agree_with_evaluated_forecast_policy_evidence(self):
        decision = decide_control_cycle(
            _observation(posture_deg=(2.2, 0.0)),
            _evidence([(0.0, -10.0)] * 6),
            _config(),
        )

        for stage in decision.as_trace()["stages"]:
            policy_evidence = stage["forecast_policy_evidence"]
            self.assertTrue(policy_evidence["policy_evaluated"])
            self.assertEqual(stage["lead_start_s"], policy_evidence["lead_start_s"])
            self.assertEqual(stage["lead_end_s"], policy_evidence["lead_end_s"])

    def test_trace_distinguishes_disabled_policy_from_unavailable_forecast(self):
        decision = decide_control_cycle(
            _observation(),
            _evidence([(0.0, -12.0)] * 6),
            _config(
                forecast_policy=ForecastActionPolicyConfig(
                    enabled=False,
                    stage_duration_s=1200.0,
                )
            ),
        )

        stage = decision.as_trace()["stages"][0]
        self.assertTrue(decision.as_trace()["forecast"]["available"])
        self.assertGreater(abs(stage["forecast_increment_deg"][0]), 0.0)
        self.assertEqual(
            stage["forecast_demand_reason"],
            "forecast_policy_disabled",
        )
        self.assertEqual(
            stage["forecast_policy_evidence"],
            {
                "available": True,
                "policy_evaluated": False,
                "reason": "policy_disabled",
            },
        )

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
