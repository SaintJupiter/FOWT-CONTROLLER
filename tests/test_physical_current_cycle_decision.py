from dataclasses import replace
import unittest
from unittest.mock import Mock

from tests.test_forecast_endpoint_posture_authorization import (
    _assembled,
    _forecast,
    _limits,
)
from tests.test_physical_forecast_cycle_facts import (
    _execution_config,
    _physical_facts,
)
from wind_prediction.execution_rollout import ExecutionRolloutRequest
from wind_prediction.forecast_execution_block_lifecycle_rollout import (
    ForecastExecutionBlockLifecycleRollout,
)
from wind_prediction.physical_current_cycle_decision import (
    PhysicalCurrentCycleCandidate,
    PhysicalCurrentCycleDecisionPolicy,
    PhysicalCurrentCycleDecisionReason,
    PhysicalCurrentCycleRoute,
    PhysicalCurrentCycleSelectionBasis,
    _select_current_cycle_candidate,
    _select_current_cycle_candidate_with_basis,
    _select_current_cycle_candidate_with_forecast_trend,
    assemble_physical_current_cycle_decision,
    assemble_physical_current_cycle_decision_from_input,
)
from wind_prediction.physical_forecast_admission import PhysicalForecastAdmission


def _candidate(
    route,
    *,
    posture,
    volume,
    active_time=0.0,
    starts=0,
    switches=0,
    lead=None,
):
    request = (
        ExecutionRolloutRequest.track((1.0, 0.0, 0.0))
        if route is not PhysicalCurrentCycleRoute.RELEASE_TO_CURRENT
        else ExecutionRolloutRequest.release_to_current()
    )
    rollout = Mock(spec=ForecastExecutionBlockLifecycleRollout)
    rollout.lifecycle_trace = Mock()
    rollout.lifecycle_trace.execution_request = request
    return PhysicalCurrentCycleCandidate(
        route=route,
        execution_request=request,
        current_block_rollout=rollout,
        max_normalized_posture=posture,
        current_block_transferred_volume_m3=volume,
        current_block_pump_active_time_s=active_time,
        current_block_pump_start_count=starts,
        current_block_pump_direction_switch_count=switches,
        forecast_lead_index=lead,
    )


class PhysicalCurrentCycleDecisionTests(unittest.TestCase):
    def test_materially_lower_posture_demand_precedes_execution_burden(self):
        safer = _candidate(
            PhysicalCurrentCycleRoute.FORECAST_ENDPOINT,
            posture=0.60,
            volume=12.0,
            lead=1,
        )
        cheaper = _candidate(
            PhysicalCurrentCycleRoute.RELEASE_TO_CURRENT,
            posture=0.72,
            volume=0.0,
        )

        selected = _select_current_cycle_candidate(
            candidates=(safer, cheaper),
            policy=PhysicalCurrentCycleDecisionPolicy(
                posture_equivalence_margin=0.05
            ),
        )

        self.assertIs(selected, safer)

        selected_with_basis, basis = _select_current_cycle_candidate_with_basis(
            candidates=(safer, cheaper),
            policy=PhysicalCurrentCycleDecisionPolicy(
                posture_equivalence_margin=0.05
            ),
        )
        self.assertIs(selected_with_basis, safer)
        self.assertIs(
            basis,
            PhysicalCurrentCycleSelectionBasis.LOWER_POSTURE_DEMAND,
        )

    def test_execution_burden_breaks_a_close_posture_tie(self):
        lower_pump = _candidate(
            PhysicalCurrentCycleRoute.RELEASE_TO_CURRENT,
            posture=0.64,
            volume=1.0,
        )
        lower_posture = _candidate(
            PhysicalCurrentCycleRoute.FORECAST_ENDPOINT,
            posture=0.60,
            volume=10.0,
            lead=0,
        )

        selected = _select_current_cycle_candidate(
            candidates=(lower_pump, lower_posture),
            policy=PhysicalCurrentCycleDecisionPolicy(
                posture_equivalence_margin=0.05
            ),
        )

        self.assertIs(selected, lower_pump)

        selected_with_basis, basis = _select_current_cycle_candidate_with_basis(
            candidates=(lower_pump, lower_posture),
            policy=PhysicalCurrentCycleDecisionPolicy(
                posture_equivalence_margin=0.05
            ),
        )
        self.assertIs(selected_with_basis, lower_pump)
        self.assertIs(
            basis,
            PhysicalCurrentCycleSelectionBasis.LOWER_EXECUTION_BURDEN,
        )

    def test_transient_or_reversing_request_can_prefer_a_posture_equivalent_release(self):
        forecast_endpoint = _candidate(
            PhysicalCurrentCycleRoute.FORECAST_ENDPOINT,
            posture=0.60,
            volume=0.1,
            lead=1,
        )
        continuation = _candidate(
            PhysicalCurrentCycleRoute.CONTINUE_EXISTING,
            posture=0.63,
            volume=2.0,
        )
        release = _candidate(
            PhysicalCurrentCycleRoute.RELEASE_TO_CURRENT,
            posture=0.64,
            volume=3.0,
        )
        selected, basis, applied = _select_current_cycle_candidate_with_forecast_trend(
            candidates=(forecast_endpoint, continuation, release),
            policy=PhysicalCurrentCycleDecisionPolicy(
                posture_equivalence_margin=0.05
            ),
            release_preference_enabled=True,
        )

        self.assertIs(selected, release)
        self.assertIs(
            basis,
            PhysicalCurrentCycleSelectionBasis.FORECAST_TREND_ROUTE_PREFERENCE,
        )
        self.assertTrue(applied)

    def test_trend_does_not_override_a_material_posture_advantage(self):
        forecast_endpoint = _candidate(
            PhysicalCurrentCycleRoute.FORECAST_ENDPOINT,
            posture=0.60,
            volume=10.0,
            lead=1,
        )
        continuation = _candidate(
            PhysicalCurrentCycleRoute.CONTINUE_EXISTING,
            posture=0.70,
            volume=0.0,
        )
        selected, basis, applied = _select_current_cycle_candidate_with_forecast_trend(
            candidates=(forecast_endpoint, continuation),
            policy=PhysicalCurrentCycleDecisionPolicy(
                posture_equivalence_margin=0.05
            ),
            release_preference_enabled=True,
        )

        self.assertIs(selected, forecast_endpoint)
        self.assertIs(
            basis,
            PhysicalCurrentCycleSelectionBasis.LOWER_POSTURE_DEMAND,
        )
        self.assertFalse(applied)

    def test_trend_has_no_effect_when_no_forecast_endpoint_is_available(self):
        continuation = _candidate(
            PhysicalCurrentCycleRoute.CONTINUE_EXISTING,
            posture=0.60,
            volume=2.0,
        )
        release = _candidate(
            PhysicalCurrentCycleRoute.RELEASE_TO_CURRENT,
            posture=0.62,
            volume=0.0,
        )
        selected, basis, applied = _select_current_cycle_candidate_with_forecast_trend(
            candidates=(continuation, release),
            policy=PhysicalCurrentCycleDecisionPolicy(
                posture_equivalence_margin=0.05
            ),
            release_preference_enabled=True,
        )

        self.assertIs(selected, release)
        self.assertIs(
            basis,
            PhysicalCurrentCycleSelectionBasis.LOWER_EXECUTION_BURDEN,
        )
        self.assertFalse(applied)

    def test_trend_is_not_reported_as_applied_when_release_already_wins(self):
        forecast_endpoint = _candidate(
            PhysicalCurrentCycleRoute.FORECAST_ENDPOINT,
            posture=0.60,
            volume=2.0,
            lead=0,
        )
        release = _candidate(
            PhysicalCurrentCycleRoute.RELEASE_TO_CURRENT,
            posture=0.62,
            volume=0.0,
        )

        selected, basis, applied = _select_current_cycle_candidate_with_forecast_trend(
            candidates=(forecast_endpoint, release),
            policy=PhysicalCurrentCycleDecisionPolicy(
                posture_equivalence_margin=0.05
            ),
            release_preference_enabled=True,
        )

        self.assertIs(selected, release)
        self.assertIs(
            basis,
            PhysicalCurrentCycleSelectionBasis.LOWER_EXECUTION_BURDEN,
        )
        self.assertFalse(applied)

    def test_no_request_is_selected_when_every_response_exceeds_limits(self):
        selected = _select_current_cycle_candidate(
            candidates=(
                _candidate(
                    PhysicalCurrentCycleRoute.CONTINUE_EXISTING,
                    posture=1.01,
                    volume=0.0,
                ),
                _candidate(
                    PhysicalCurrentCycleRoute.RELEASE_TO_CURRENT,
                    posture=1.20,
                    volume=0.0,
                ),
            ),
            policy=PhysicalCurrentCycleDecisionPolicy(
                posture_equivalence_margin=0.05
            ),
        )

        self.assertIsNone(selected)

        selected_with_basis, basis = _select_current_cycle_candidate_with_basis(
            candidates=(
                _candidate(
                    PhysicalCurrentCycleRoute.CONTINUE_EXISTING,
                    posture=1.01,
                    volume=0.0,
                ),
            ),
            policy=PhysicalCurrentCycleDecisionPolicy(
                posture_equivalence_margin=0.05
            ),
        )
        self.assertIsNone(selected_with_basis)
        self.assertIs(
            basis,
            PhysicalCurrentCycleSelectionBasis.NO_FEASIBLE_REQUEST,
        )

    def test_real_responses_expose_one_selected_request_and_its_candidates(self):
        response_set, horizon = _assembled(
            primary_target_masses_kg=(3_000.0, 0.0, 0.0),
        )

        decision = assemble_physical_current_cycle_decision(
            response_set=response_set,
            horizon_comparison=horizon,
            posture_limits=_limits(angle=0.04, rate=0.0116),
            policy=PhysicalCurrentCycleDecisionPolicy(
                posture_equivalence_margin=0.05
            ),
        )

        self.assertEqual(
            len(decision.candidates),
            2 + len(response_set.responses),
        )
        self.assertIsNotNone(decision.selected_candidate)
        self.assertIn(decision.selected_candidate, decision.candidates)
        self.assertIs(
            decision.current_execution_request,
            decision.selected_candidate.execution_request,
        )
        self.assertIs(
            decision.selected_current_block_rollout,
            decision.selected_candidate.current_block_rollout,
        )
        self.assertIs(
            decision.selected_current_block_rollout.lifecycle_trace.execution_request,
            decision.current_execution_request,
        )
        self.assertNotEqual(
            decision.reason,
            PhysicalCurrentCycleDecisionReason.NO_REQUEST_WITHIN_LIMITS,
        )

    def test_public_input_assembly_returns_the_selected_physical_rollout(self):
        decision_input = _physical_facts(
            future_uv_ms=[[0.0, -10.0], [2.0, -12.0], [-3.0, -8.0]],
            lead_reliability=(1.0, 1.0, 1.0),
            execution_config=replace(_execution_config(), block_duration_s=600.0),
        )
        forecast = _forecast(decision_input)
        decision = assemble_physical_current_cycle_decision_from_input(
            decision_input=decision_input,
            forecast_admission=PhysicalForecastAdmission(
                decision_input=decision_input,
                forecast=forecast,
                admitted=True,
                basis="fixture source-bound admission",
            ),
            posture_limits=_limits(angle=0.04, rate=0.0116),
            policy=PhysicalCurrentCycleDecisionPolicy(
                posture_equivalence_margin=0.05
            ),
        )

        self.assertIsNotNone(decision.current_execution_request)
        self.assertIsNotNone(decision.selected_current_block_rollout)
        self.assertIs(
            decision.selected_current_block_rollout.lifecycle_trace.execution_request,
            decision.current_execution_request,
        )


if __name__ == "__main__":
    unittest.main()
