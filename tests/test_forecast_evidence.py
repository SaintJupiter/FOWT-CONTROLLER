import tempfile
import unittest
from pathlib import Path

import numpy as np

from wind_prediction.forecast_adapter import ForecastResult
from wind_prediction.forecast_evidence import (
    ForecastEvidence,
    evidence_from_result,
    lead_reliability_from_metrics,
    validate_forecast_evidence,
)


class ForecastEvidenceTests(unittest.TestCase):
    @staticmethod
    def _valid_evidence(**overrides):
        values = {
            "source": "test_source",
            "model_version": "test_model",
            "origin_time": "2026-01-01 00:00:00",
            "sample_period_s": 600.0,
            "uv_ms": np.array([[3.0, 4.0], [4.0, 3.0]], dtype=float),
            "event_probs": {"attention_event_0_20m": 0.7},
            "lead_reliability": np.array([1.0, 0.8]),
            "provides_future_preview": True,
            "metadata": {},
        }
        values.update(overrides)
        return values

    def test_result_conversion_preserves_full_trajectory_and_events(self):
        uv = np.array([[3.0, 4.0], [4.0, 3.0]], dtype=float)
        result = ForecastResult(
            wind_uv_raw=uv,
            wind_speed=np.linalg.norm(uv, axis=1),
            wind_dir_deg=np.array([210.0, 220.0]),
            event_probs={"attention_event_0_20m": 0.7},
            risk_levels={"attention_event_0_20m": True},
            model_version="test_model",
            timestamp="2026-01-01 00:00:00",
        )
        evidence = evidence_from_result(
            result,
            source="learned",
            sample_period_s=600.0,
            provides_future_preview=True,
            lead_reliability=np.array([1.0, 0.8]),
        )

        np.testing.assert_allclose(evidence.uv_ms, uv)
        np.testing.assert_allclose(evidence.lead_reliability, [1.0, 0.8])
        self.assertEqual(evidence.event_probs["attention_event_0_20m"], 0.7)
        self.assertEqual(evidence.horizon_minutes, 20.0)

    def test_evidence_arrays_and_event_mapping_are_immutable(self):
        evidence = ForecastEvidence(**self._valid_evidence())

        with self.assertRaises(ValueError):
            evidence.uv_ms[0, 0] = 99.0
        with self.assertRaises(ValueError):
            evidence.lead_reliability[0] = 0.0
        with self.assertRaises(TypeError):
            evidence.event_probs["new_event"] = 0.5

    def test_validation_rejects_missing_lead_reliability(self):
        result = ForecastResult(
            wind_uv_raw=np.ones((2, 2)),
            wind_speed=np.ones(2),
            wind_dir_deg=np.zeros(2),
            event_probs={},
            risk_levels={},
            model_version="test",
        )
        with self.assertRaisesRegex(ValueError, "one value per lead"):
            evidence = evidence_from_result(
                result,
                source="learned",
                sample_period_s=600.0,
                provides_future_preview=True,
                lead_reliability=np.ones(1),
            )
            validate_forecast_evidence(evidence)

    def test_validation_rejects_ambiguous_identity_and_metadata(self):
        invalid_cases = (
            ({"source": " "}, "source"),
            ({"model_version": ""}, "model_version"),
            ({"origin_time": " "}, "origin_time"),
            ({"provides_future_preview": 1}, "provides_future_preview"),
            ({"event_probs": {"": 0.7}}, "event names"),
            ({"metadata": []}, "metadata"),
        )
        for overrides, message in invalid_cases:
            with self.subTest(overrides=overrides), self.assertRaisesRegex(
                ValueError,
                message,
            ):
                validate_forecast_evidence(
                    ForecastEvidence(**self._valid_evidence(**overrides))
                )

    def test_reliability_is_derived_from_validation_lead_error(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metrics.csv"
            path.write_text(
                "model,split,scope,vector_mae_ms\n"
                "m,validation,tplus_10m,0.5\n"
                "m,validation,tplus_20m,1.0\n"
                "m,validation,tplus_30m,2.0\n",
                encoding="utf-8",
            )
            reliability = lead_reliability_from_metrics(
                path,
                minimum=0.3,
                expected_lead_minutes=(10, 20, 30),
            )
            np.testing.assert_allclose(reliability, [1.0, 0.5, 0.3])

    def test_reliability_rejects_a_same_length_but_shifted_metric_grid(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metrics.csv"
            path.write_text(
                "model,split,scope,vector_mae_ms\n"
                "m,validation,tplus_20m,0.5\n"
                "m,validation,tplus_30m,1.0\n"
                "m,validation,tplus_40m,2.0\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "exactly match"):
                lead_reliability_from_metrics(
                    path,
                    expected_lead_minutes=(10, 20, 30),
                )


if __name__ == "__main__":
    unittest.main()
