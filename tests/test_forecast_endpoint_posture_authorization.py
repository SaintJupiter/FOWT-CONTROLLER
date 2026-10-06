from dataclasses import replace
import ast
from pathlib import Path
import unittest

import numpy as np

from tests.test_physical_forecast_cycle_facts import (
    _execution_config,
    _physical_facts,
    _REFERENCE_MASSES,
)
from wind_prediction.execution_rollout import ExecutionRolloutState
from wind_prediction.forecast_endpoint_horizon_response import (
    assemble_forecast_endpoint_held_responses,
)
from wind_prediction.forecast_endpoint_posture_authorization import (
    ForecastEndpointPostureAuthorizationReason,
    _derive_authorization,
    authorize_forecast_endpoint_from_posture_response,
)
from wind_prediction.forecast_evidence import ForecastEvidence
from wind_prediction.physical_forecast_admission import (
    bind_physical_forecast_admission,
    bind_physical_forecast_lead_reliability_scope,
)
from wind_prediction.physical_lifecycle_comparison import (
    assemble_physical_lifecycle_comparison_for_current_block,
    assemble_physical_lifecycle_horizon_comparison,
)
from wind_prediction.physical_lifecycle_selection import PhysicalPostureLimits


def _admission(facts, *, admitted=True):
    assembly = facts.trajectory.load_assembly
    identity = assembly.input_identity
    forecast = ForecastEvidence(
        source=identity.forecast_source,
        model_version=identity.forecast_model_version,
        origin_time=identity.forecast_origin_time,
        sample_period_s=identity.forecast_sample_period_s,
        uv_ms=(
            np.asarray(assembly.future_rotor_plane_enu_downwind_mps, dtype=float)
            / assembly.parameters.rotor_plane_speed_scale
        ),
        lead_reliability=np.ones(identity.forecast_horizon_steps),
        event_probs={"fixture_event": 0.5},
    )
    return bind_physical_forecast_admission(
        decision_input=facts,
        forecast=forecast,
        admitted=admitted,
        basis="fixture source-bound admission",
    )


def _forecast(facts, *, lead_reliability=None):
    assembly = facts.trajectory.load_assembly
    identity = assembly.input_identity
    if lead_reliability is None:
        lead_reliability = np.ones(identity.forecast_horizon_steps)
    return ForecastEvidence(
        source=identity.forecast_source,
        model_version=identity.forecast_model_version,
        origin_time=identity.forecast_origin_time,
        sample_period_s=identity.forecast_sample_period_s,
        uv_ms=(
            np.asarray(assembly.future_rotor_plane_enu_downwind_mps, dtype=float)
            / assembly.parameters.rotor_plane_speed_scale
        ),
        lead_reliability=np.asarray(lead_reliability, dtype=float),
        event_probs={"fixture_event": 0.5},
    )


def _assembled(*, admitted=True, primary_target_masses_kg=None):
    execution_state = None
    if primary_target_masses_kg is not None:
        primary_target = np.asarray(primary_target_masses_kg, dtype=float)
        execution_state = ExecutionRolloutState(
            masses_kg=_REFERENCE_MASSES,
            target_masses_kg=primary_target,
            primary_target_kg=primary_target,
            pump_rates_m3_min=np.zeros(3),
            pump_latched=np.zeros(3, dtype=bool),
            pump_off_elapsed_s=np.full(3, 60.0),
        )
    facts = _physical_facts(
        future_uv_ms=[[0.0, -10.0], [2.0, -12.0], [-3.0, -8.0]],
        lead_reliability=(1.0, 1.0, 1.0),
        execution_config=replace(_execution_config(), block_duration_s=600.0),
        execution_state=execution_state,
    )
    comparison = assemble_physical_lifecycle_comparison_for_current_block(
        decision_input=facts,
        forecast_admission=_admission(facts, admitted=admitted),
    )
    return (
        assemble_forecast_endpoint_held_responses(
            current_block_comparison=comparison,
        ),
        assemble_physical_lifecycle_horizon_comparison(
            current_block_comparison=comparison,
        ),
    )


def _limits(*, angle=1.0, rate=1.0):
    return PhysicalPostureLimits(
        max_abs_pitch_roll_rad=(angle, angle),
        max_abs_pitch_roll_rate_rad_s=(rate, rate),
        source="test caller-owned posture bounds",
    )


class ForecastEndpointPostureAuthorizationTests(unittest.TestCase):
    def test_continuation_within_limits_does_not_authorize_an_endpoint(self):
        response_set, horizon = _assembled()

        authorization = authorize_forecast_endpoint_from_posture_response(
            response_set=response_set,
            horizon_comparison=horizon,
            posture_limits=_limits(),
        )

        self.assertEqual(
            authorization.reason,
            ForecastEndpointPostureAuthorizationReason.CONTINUATION_WITHIN_LIMITS,
        )
        self.assertIsNone(authorization.authorized_lead_index)
        self.assertIsNone(authorization.authorized_response)
        self.assertIsNone(authorization.authorized_current_execution_request)

    def test_nonadmitted_forecast_never_authorizes_an_endpoint(self):
        response_set, horizon = _assembled(admitted=False)

        authorization = authorize_forecast_endpoint_from_posture_response(
            response_set=response_set,
            horizon_comparison=horizon,
            posture_limits=_limits(),
        )

        self.assertEqual(
            authorization.eligible_lead_indices,
            ()
        )
        self.assertEqual(
            authorization.reason,
            ForecastEndpointPostureAuthorizationReason.FORECAST_NOT_ADMITTED,
        )
        self.assertIsNone(authorization.authorized_lead_index)

    def test_authorization_rule_only_accepts_one_unambiguous_endpoint(self):
        self.assertEqual(
            _derive_authorization(
                forecast_admitted=True,
                origin_within_limits=False,
                continuation_within_limits=False,
                available_lead_indices=(2,),
                eligible_lead_indices=(2,),
            ),
            (
                None,
                ForecastEndpointPostureAuthorizationReason.INITIAL_STATE_OUTSIDE_SCOPE,
            ),
        )

    def test_real_response_authorizes_the_earliest_in_bounds_endpoint(self):
        response_set, horizon = _assembled(
            primary_target_masses_kg=(3_000.0, 0.0, 0.0),
        )

        authorization = authorize_forecast_endpoint_from_posture_response(
            response_set=response_set,
            horizon_comparison=horizon,
            posture_limits=_limits(angle=0.025, rate=0.0114),
        )

        self.assertEqual(authorization.eligible_lead_indices, (2,))
        self.assertEqual(authorization.authorized_lead_index, 2)
        self.assertEqual(
            authorization.reason,
            ForecastEndpointPostureAuthorizationReason.EARLIEST_ENDPOINT_WITHIN_LIMITS,
        )
        self.assertIs(
            authorization.authorized_response,
            response_set.response_for_lead(2),
        )
        self.assertIs(
            authorization.authorized_current_execution_request,
            response_set.response_for_lead(2).lifecycle_trace.execution_request,
        )

    def test_real_response_selects_the_earliest_when_multiple_endpoints_are_in_bounds(self):
        response_set, horizon = _assembled(
            primary_target_masses_kg=(3_000.0, 0.0, 0.0),
        )

        authorization = authorize_forecast_endpoint_from_posture_response(
            response_set=response_set,
            horizon_comparison=horizon,
            posture_limits=_limits(angle=0.04, rate=0.0116),
        )

        self.assertEqual(authorization.eligible_lead_indices, (0, 1, 2))
        self.assertEqual(
            authorization.reason,
            ForecastEndpointPostureAuthorizationReason.EARLIEST_ENDPOINT_WITHIN_LIMITS,
        )
        self.assertEqual(authorization.authorized_lead_index, 0)
        self.assertIs(
            authorization.authorized_current_execution_request,
            response_set.response_for_lead(0).lifecycle_trace.execution_request,
        )
        self.assertEqual(
            _derive_authorization(
                forecast_admitted=True,
                origin_within_limits=True,
                continuation_within_limits=False,
                available_lead_indices=(0, 1, 2),
                eligible_lead_indices=(2,),
            ),
            (
                2,
                ForecastEndpointPostureAuthorizationReason.EARLIEST_ENDPOINT_WITHIN_LIMITS,
            ),
        )
        self.assertEqual(
            _derive_authorization(
                forecast_admitted=True,
                origin_within_limits=True,
                continuation_within_limits=False,
                available_lead_indices=(0, 1, 2),
                eligible_lead_indices=(1, 2),
            ),
            (
                1,
                ForecastEndpointPostureAuthorizationReason.EARLIEST_ENDPOINT_WITHIN_LIMITS,
            ),
        )
        self.assertEqual(
            _derive_authorization(
                forecast_admitted=True,
                origin_within_limits=True,
                continuation_within_limits=False,
                available_lead_indices=(0, 1, 2),
                eligible_lead_indices=(),
            ),
            (
                None,
                ForecastEndpointPostureAuthorizationReason.NO_ENDPOINT_WITHIN_LIMITS,
            ),
        )

    def test_reliability_scope_can_abstain_before_an_unreliable_endpoint_is_used(self):
        response_set, horizon = _assembled(
            primary_target_masses_kg=(3_000.0, 0.0, 0.0),
        )
        facts = response_set.current_block_comparison.decision_input
        scope = bind_physical_forecast_lead_reliability_scope(
            decision_input=facts,
            forecast=_forecast(facts, lead_reliability=(0.8, 0.7, 0.6)),
            minimum_lead_reliability=0.9,
            basis="fixture held-out lead diagnostic cutoff",
        )

        authorization = authorize_forecast_endpoint_from_posture_response(
            response_set=response_set,
            horizon_comparison=horizon,
            posture_limits=_limits(angle=0.025, rate=0.0114),
            reliability_scope=scope,
        )

        self.assertEqual(authorization.available_lead_indices, ())
        self.assertEqual(authorization.eligible_lead_indices, ())
        self.assertEqual(
            authorization.reason,
            ForecastEndpointPostureAuthorizationReason.NO_RELIABILITY_ELIGIBLE_ENDPOINT,
        )
        self.assertIsNone(authorization.authorized_current_execution_request)

    def test_authorization_rejects_a_horizon_from_other_current_block(self):
        response_set, _ = _assembled()
        _, other_horizon = _assembled()

        with self.assertRaisesRegex(ValueError, "must retain the response set"):
            authorize_forecast_endpoint_from_posture_response(
                response_set=response_set,
                horizon_comparison=other_horizon,
                posture_limits=_limits(),
            )

    def test_module_avoids_legacy_policy_and_candidate_dependencies(self):
        path = (
            Path(__file__).resolve().parents[1]
            / "src"
            / "wind_prediction"
            / "forecast_endpoint_posture_authorization.py"
        )
        tree = ast.parse(path.read_text(encoding="utf-8"))
        imported_modules = {
            node.module
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module is not None
        }
        forbidden_fragments = (
            "candidate",
            "planner",
            "controller",
            "forecast_action_policy",
            "event",
        )
        self.assertFalse(
            any(
                any(fragment in module for fragment in forbidden_fragments)
                for module in imported_modules
            )
        )


if __name__ == "__main__":
    unittest.main()
