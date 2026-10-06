from dataclasses import replace
import unittest

import numpy as np

from tests.test_physical_forecast_cycle_facts import _physical_facts
from wind_prediction.forecast_evidence import ForecastEvidence
from wind_prediction.physical_forecast_admission import (
    PhysicalForecastAdmission,
    bind_physical_forecast_lead_reliability_scope,
)


def _matching_forecast(
    decision_input,
    *,
    source=None,
    model_version=None,
    origin_time=None,
    sample_period_s=None,
    uv_ms=None,
    lead_reliability=None,
    event_probs=None,
    provides_future_preview=True,
) -> ForecastEvidence:
    assembly = decision_input.trajectory.load_assembly
    identity = assembly.input_identity
    raw_uv = (
        np.asarray(assembly.future_rotor_plane_enu_downwind_mps, dtype=float)
        / assembly.parameters.rotor_plane_speed_scale
        if uv_ms is None
        else np.asarray(uv_ms, dtype=float)
    )
    if lead_reliability is None:
        lead_reliability = np.ones(raw_uv.shape[0])
    if event_probs is None:
        event_probs = {"fixture_event": 0.5}
    return ForecastEvidence(
        source=identity.forecast_source if source is None else source,
        model_version=(
            identity.forecast_model_version if model_version is None else model_version
        ),
        origin_time=(identity.forecast_origin_time if origin_time is None else origin_time),
        sample_period_s=(
            identity.forecast_sample_period_s
            if sample_period_s is None
            else sample_period_s
        ),
        uv_ms=raw_uv,
        lead_reliability=lead_reliability,
        event_probs=event_probs,
        provides_future_preview=provides_future_preview,
    )


class PhysicalForecastAdmissionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.facts = _physical_facts(
            future_uv_ms=np.array([[0.0, -10.0], [2.0, -12.0]]),
        )

    def test_matching_forecast_binds_a_caller_owned_admission(self):
        forecast = _matching_forecast(self.facts)

        admission = PhysicalForecastAdmission(
            decision_input=self.facts,
            forecast=forecast,
            admitted=True,
            basis="fixture supervisory admission",
        )

        self.assertIs(admission.decision_input, self.facts)
        self.assertIs(
            admission.forecast_identity,
            self.facts.trajectory.load_assembly.input_identity,
        )
        self.assertFalse(hasattr(admission, "forecast"))
        self.assertTrue(admission.admitted)

    def test_post_bind_mutation_of_evidence_cannot_change_admission_fact(self):
        forecast = _matching_forecast(self.facts)
        admission = PhysicalForecastAdmission(
            decision_input=self.facts,
            forecast=forecast,
            admitted=True,
            basis="fixture supervisory admission",
        )
        physical_wind_before = np.array(
            self.facts.trajectory.load_assembly.future_rotor_plane_enu_downwind_mps,
            copy=True,
        )

        forecast.uv_ms[0, 0] += 5.0
        forecast.lead_reliability[0] = 0.0
        forecast.event_probs["fixture_event"] = 1.0

        np.testing.assert_allclose(
            self.facts.trajectory.load_assembly.future_rotor_plane_enu_downwind_mps,
            physical_wind_before,
            rtol=0.0,
            atol=0.0,
        )
        self.assertIs(
            admission.forecast_identity,
            self.facts.trajectory.load_assembly.input_identity,
        )
        self.assertTrue(admission.admitted)

    def test_changed_future_vector_is_rejected_before_selection(self):
        forecast = _matching_forecast(self.facts)
        changed_uv = np.array(forecast.uv_ms, copy=True)
        changed_uv[1, 0] += 0.01

        with self.assertRaisesRegex(ValueError, "wind vectors"):
            PhysicalForecastAdmission(
                decision_input=self.facts,
                forecast=replace(forecast, uv_ms=changed_uv),
                admitted=True,
                basis="fixture supervisory admission",
            )

    def test_identity_mismatch_is_rejected(self):
        cases = (
            ({"source": "foreign_forecast"}, "source"),
            ({"model_version": "foreign-v2"}, "model_version"),
            ({"origin_time": "2026-08-22T00:10:00"}, "origin_time"),
            ({"sample_period_s": 300.0}, "sample_period_s"),
        )
        for replacements, expected_message in cases:
            with self.subTest(replacements=replacements):
                with self.assertRaisesRegex(ValueError, expected_message):
                    PhysicalForecastAdmission(
                        decision_input=self.facts,
                        forecast=_matching_forecast(self.facts, **replacements),
                        admitted=True,
                        basis="fixture supervisory admission",
                    )

    def test_horizon_mismatch_is_rejected(self):
        forecast = _matching_forecast(self.facts)
        extended_uv = np.vstack((forecast.uv_ms, forecast.uv_ms[-1]))

        with self.assertRaisesRegex(ValueError, "horizon"):
            PhysicalForecastAdmission(
                decision_input=self.facts,
                forecast=_matching_forecast(
                    self.facts,
                    uv_ms=extended_uv,
                    lead_reliability=np.ones(extended_uv.shape[0]),
                ),
                admitted=True,
                basis="fixture supervisory admission",
            )

    def test_event_and_reliability_metadata_do_not_rescale_physical_binding(self):
        forecast = _matching_forecast(
            self.facts,
            lead_reliability=np.array([0.2, 0.9]),
            event_probs={"wind_change": 0.01, "gust": 0.99},
        )

        admission = PhysicalForecastAdmission(
            decision_input=self.facts,
            forecast=forecast,
            admitted=False,
            basis="fixture supervisory rejection",
        )

        self.assertFalse(admission.admitted)

    def test_future_preview_is_required(self):
        with self.assertRaisesRegex(ValueError, "future preview"):
            PhysicalForecastAdmission(
                decision_input=self.facts,
                forecast=_matching_forecast(
                    self.facts,
                    provides_future_preview=False,
                ),
                admitted=True,
                basis="fixture supervisory admission",
            )

    def test_reliability_scope_keeps_only_the_contiguous_available_prefix(self):
        scope = bind_physical_forecast_lead_reliability_scope(
            decision_input=self.facts,
            forecast=_matching_forecast(
                self.facts,
                lead_reliability=np.array([1.0, 0.82]),
            ),
            minimum_lead_reliability=0.9,
            basis="fixture held-out lead diagnostic cutoff",
        )

        self.assertEqual(scope.lead_reliability, (1.0, 0.82))
        self.assertEqual(scope.available_lead_indices, (0,))
        self.assertIs(
            scope.forecast_identity,
            self.facts.trajectory.load_assembly.input_identity,
        )

    def test_reliability_scope_can_exclude_every_lead_without_rejecting_source_binding(self):
        scope = bind_physical_forecast_lead_reliability_scope(
            decision_input=self.facts,
            forecast=_matching_forecast(
                self.facts,
                lead_reliability=np.array([0.89, 0.95]),
            ),
            minimum_lead_reliability=0.9,
            basis="fixture held-out lead diagnostic cutoff",
        )

        self.assertEqual(scope.available_lead_indices, ())

    def test_reliability_scope_rejects_an_out_of_range_cutoff(self):
        with self.assertRaisesRegex(ValueError, "minimum_lead_reliability"):
            bind_physical_forecast_lead_reliability_scope(
                decision_input=self.facts,
                forecast=_matching_forecast(self.facts),
                minimum_lead_reliability=1.01,
                basis="fixture held-out lead diagnostic cutoff",
            )


if __name__ == "__main__":
    unittest.main()
