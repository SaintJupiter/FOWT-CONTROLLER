import ast
from pathlib import Path
import unittest

import numpy as np

from tests.test_physical_forecast_cycle_facts import _ORIGIN, _physical_facts
from wind_prediction.forecast_evidence import ForecastEvidence
from wind_prediction.physical_forecast_admission import (
    PhysicalForecastAdmission,
    bind_physical_forecast_lead_reliability_scope,
)
from wind_prediction.physical_forecast_rhs_ballast_trend import (
    PhysicalForecastRhsBallastAxis,
    PhysicalForecastRhsBallastAxisTrend,
    PhysicalForecastRhsBallastAxisTrendKind,
    PhysicalForecastRhsBallastTrendKind,
    PhysicalForecastRhsBallastTrendPolicy,
    _classify_axis_request,
    _combine_axis_trends,
    diagnose_physical_forecast_rhs_ballast_trend,
)
from wind_prediction.physical_current_cycle_decision import (
    PhysicalCurrentCycleDecisionPolicy,
    PhysicalCurrentCycleForecastTrendRoutePolicy,
    assemble_physical_current_cycle_decision_from_input,
)
from wind_prediction.physical_lifecycle_selection import PhysicalPostureLimits


_LEAD_INDICES = (0, 1, 2)
_LEAD_TIMES_S = np.array([600.0, 1_200.0, 1_800.0])


def _forecast_for(future_uv_ms, lead_reliability):
    return ForecastEvidence(
        source="replay_lstm_fixture",
        model_version="fixture-v1",
        origin_time=_ORIGIN,
        sample_period_s=600.0,
        uv_ms=np.asarray(future_uv_ms, dtype=float),
        lead_reliability=np.asarray(lead_reliability, dtype=float),
        event_probs={"wind_change": 0.6},
    )


class PhysicalForecastRhsBallastTrendTests(unittest.TestCase):
    def _classify(self, values_nm):
        return _classify_axis_request(
            axis=PhysicalForecastRhsBallastAxis.PITCH,
            values_nm=np.asarray(values_nm, dtype=float),
            lead_indices=_LEAD_INDICES,
            lead_times_s=_LEAD_TIMES_S,
            resolution_nm=10.0,
            minimum_evidence_duration_s=1_200.0,
        )

    def test_component_classification_uses_one_resolution_for_peak_reversal_and_growth(self):
        transient = self._classify((0.0, 50.0, 10.0))
        reversal = self._classify((20.0, 30.0, -20.0))
        growth = self._classify((0.0, 20.0, 40.0))

        self.assertIs(
            transient.kind,
            PhysicalForecastRhsBallastAxisTrendKind.TRANSIENT_PEAK,
        )
        self.assertEqual(transient.peak_lead_index, 1)
        self.assertIs(
            reversal.kind,
            PhysicalForecastRhsBallastAxisTrendKind.DIRECTION_REVERSAL,
        )
        self.assertIs(
            growth.kind,
            PhysicalForecastRhsBallastAxisTrendKind.PERSISTENT_GROWTH,
        )

    def test_monotonic_decay_is_not_mistaken_for_an_internal_transient_peak(self):
        decline = self._classify((50.0, 30.0, 10.0))

        self.assertIs(
            decline.kind,
            PhysicalForecastRhsBallastAxisTrendKind.MIXED_OR_STEADY,
        )

    def test_conflicting_axis_trends_remain_mixed(self):
        combined = _combine_axis_trends(
            (
                PhysicalForecastRhsBallastAxisTrend(
                    axis=PhysicalForecastRhsBallastAxis.PITCH,
                    kind=PhysicalForecastRhsBallastAxisTrendKind.DIRECTION_REVERSAL,
                    peak_lead_index=1,
                    initial_sign=1,
                    terminal_sign=-1,
                ),
                PhysicalForecastRhsBallastAxisTrend(
                    axis=PhysicalForecastRhsBallastAxis.ROLL,
                    kind=PhysicalForecastRhsBallastAxisTrendKind.PERSISTENT_GROWTH,
                    peak_lead_index=2,
                    initial_sign=1,
                    terminal_sign=1,
                ),
            )
        )

        self.assertIs(combined, PhysicalForecastRhsBallastTrendKind.MIXED)

    def test_short_or_negligible_series_is_not_interpreted_as_a_strategy_signal(self):
        short = _classify_axis_request(
            axis=PhysicalForecastRhsBallastAxis.ROLL,
            values_nm=np.array((0.0, 100.0)),
            lead_indices=(0, 1),
            lead_times_s=np.array((600.0, 1_200.0)),
            resolution_nm=10.0,
            minimum_evidence_duration_s=600.0,
        )
        negligible = self._classify((2.0, -3.0, 5.0))

        self.assertIs(
            short.kind,
            PhysicalForecastRhsBallastAxisTrendKind.INSUFFICIENT_EVIDENCE,
        )
        self.assertIs(
            negligible.kind,
            PhysicalForecastRhsBallastAxisTrendKind.INSUFFICIENT_EVIDENCE,
        )

    def test_trend_retains_exact_input_and_reliability_prefix(self):
        decision_input = _physical_facts(
            future_uv_ms=np.array([[0.0, -10.0], [1.0, -11.0], [2.0, -12.0]]),
            lead_reliability=(1.0, 1.0, 0.2),
        )
        forecast = _forecast_for(
            [[0.0, -10.0], [1.0, -11.0], [2.0, -12.0]],
            (1.0, 1.0, 0.2),
        )
        scope = bind_physical_forecast_lead_reliability_scope(
            decision_input=decision_input,
            forecast=forecast,
            minimum_lead_reliability=0.9,
            basis="fixture prefix",
        )

        trend = diagnose_physical_forecast_rhs_ballast_trend(
            decision_input=decision_input,
            reliability_scope=scope,
            policy=PhysicalForecastRhsBallastTrendPolicy(
                axis_resolution_nm=(1.0, 1.0),
                minimum_evidence_duration_s=600.0,
            ),
        )

        self.assertIs(trend.decision_input, decision_input)
        self.assertIs(trend.reliability_scope, scope)
        self.assertEqual(trend.lead_indices, (0, 1))
        self.assertEqual(trend.requested_pitch_roll_loads_nm.shape, (2, 2))
        np.testing.assert_allclose(
            trend.requested_pitch_roll_loads_nm,
            np.array(
                [
                    diagnostic.requested_ballast_pitch_roll_load_nm
                    for diagnostic in decision_input.horizon_rhs_ballast_diagnostics[:2]
                ]
            ),
        )
        self.assertIs(
            trend.kind,
            PhysicalForecastRhsBallastTrendKind.INSUFFICIENT_EVIDENCE,
        )

    def test_empty_reliable_prefix_remains_an_empty_nonactionable_fact(self):
        decision_input = _physical_facts(
            future_uv_ms=np.array([[0.0, -10.0], [1.0, -11.0], [2.0, -12.0]]),
            lead_reliability=(0.2, 1.0, 1.0),
        )
        forecast = _forecast_for(
            [[0.0, -10.0], [1.0, -11.0], [2.0, -12.0]],
            (0.2, 1.0, 1.0),
        )
        scope = bind_physical_forecast_lead_reliability_scope(
            decision_input=decision_input,
            forecast=forecast,
            minimum_lead_reliability=0.9,
            basis="fixture empty prefix",
        )

        trend = diagnose_physical_forecast_rhs_ballast_trend(
            decision_input=decision_input,
            reliability_scope=scope,
            policy=PhysicalForecastRhsBallastTrendPolicy(
                axis_resolution_nm=(1.0, 1.0),
                minimum_evidence_duration_s=600.0,
            ),
        )

        self.assertEqual(trend.lead_indices, ())
        self.assertEqual(trend.requested_pitch_roll_loads_nm.shape, (0, 2))
        self.assertIs(
            trend.kind,
            PhysicalForecastRhsBallastTrendKind.INSUFFICIENT_EVIDENCE,
        )

    def test_source_bound_reversal_can_explicitly_prefer_a_safe_release(self):
        future_uv = [
            [0.0, -7.0],
            [0.0, -12.0],
            [0.0, -7.0],
            [0.0, -7.0],
            [0.0, -7.0],
        ]
        decision_input = _physical_facts(
            future_uv_ms=np.array(future_uv),
            lead_reliability=(1.0, 1.0, 1.0, 1.0, 1.0),
        )
        forecast = _forecast_for(future_uv, (1.0, 1.0, 1.0, 1.0, 1.0))
        scope = bind_physical_forecast_lead_reliability_scope(
            decision_input=decision_input,
            forecast=forecast,
            minimum_lead_reliability=0.9,
            basis="fixture full prefix",
        )
        trend = diagnose_physical_forecast_rhs_ballast_trend(
            decision_input=decision_input,
            reliability_scope=scope,
            policy=PhysicalForecastRhsBallastTrendPolicy(
                axis_resolution_nm=(1.0, 1.0),
                minimum_evidence_duration_s=1_200.0,
            ),
        )
        common_arguments = dict(
            decision_input=decision_input,
            forecast_admission=PhysicalForecastAdmission(
                decision_input=decision_input,
                forecast=forecast,
                admitted=True,
                basis="fixture admission",
            ),
            posture_limits=PhysicalPostureLimits(
                max_abs_pitch_roll_rad=(0.04, 0.04),
                max_abs_pitch_roll_rate_rad_s=(0.0116, 0.0116),
                source="fixture posture limits",
            ),
            policy=PhysicalCurrentCycleDecisionPolicy(
                posture_equivalence_margin=0.05
            ),
            reliability_scope=scope,
            forecast_rhs_ballast_trend=trend,
        )
        baseline = assemble_physical_current_cycle_decision_from_input(
            **common_arguments
        )
        decision = assemble_physical_current_cycle_decision_from_input(
            **common_arguments,
            forecast_trend_route_policy=PhysicalCurrentCycleForecastTrendRoutePolicy(
                release_preferred_trend_kinds=(
                    PhysicalForecastRhsBallastTrendKind.DIRECTION_REVERSAL,
                )
            ),
        )

        self.assertIs(
            trend.kind,
            PhysicalForecastRhsBallastTrendKind.DIRECTION_REVERSAL,
        )
        reversing_axes = tuple(
            axis_trend
            for axis_trend in trend.axis_trends
            if axis_trend.kind
            is PhysicalForecastRhsBallastAxisTrendKind.DIRECTION_REVERSAL
        )
        self.assertTrue(reversing_axes)
        self.assertTrue(
            all(
                axis_trend.initial_sign != axis_trend.terminal_sign
                for axis_trend in reversing_axes
            )
        )
        self.assertFalse(baseline.forecast_trend_route_preference_applied)
        self.assertEqual(baseline.selected_route.value, "continue_existing")
        self.assertIs(decision.forecast_rhs_ballast_trend, trend)
        self.assertTrue(decision.forecast_trend_route_preference_applied)
        self.assertEqual(decision.selected_route.value, "release_to_current")
        release_candidate = next(
            candidate
            for candidate in decision.candidates
            if candidate.route.value == "release_to_current"
        )
        self.assertIs(
            decision.selected_current_block_rollout,
            release_candidate.current_block_rollout,
        )
        self.assertIsNotNone(decision.current_execution_request)

    def test_trend_module_does_not_import_raw_forecast_or_control_execution_layers(self):
        path = (
            Path(__file__).resolve().parents[1]
            / "src"
            / "wind_prediction"
            / "physical_forecast_rhs_ballast_trend.py"
        )
        tree = ast.parse(path.read_text(encoding="utf-8"))
        imported_modules = {
            node.module
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module is not None
        }
        forbidden_fragments = (
            "forecast_evidence",
            "controller",
            "execution",
            "pump",
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
