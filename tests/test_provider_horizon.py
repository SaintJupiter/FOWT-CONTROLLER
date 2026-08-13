import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from wind_prediction.provider_horizon import ProviderHorizonMixin


def _write_metrics(path: Path, values: list[float]) -> None:
    rows = ["model,split,scope,vector_mae_ms"]
    rows.extend(
        f"m,validation,tplus_{10 * index}m,{value}"
        for index, value in enumerate(values, start=1)
    )
    path.mkdir(parents=True, exist_ok=True)
    (path / "lstm_regression_metrics.csv").write_text(
        "\n".join(rows) + "\n",
        encoding="utf-8",
    )


class _Provider(ProviderHorizonMixin):
    def __init__(self, model_dir: Path | None, *, enabled: bool) -> None:
        self.forecast_adapter = SimpleNamespace(model_dir=model_dir)
        self.cfg = SimpleNamespace(lead_reliability_enabled=enabled)


class ProviderHorizonReliabilityTests(unittest.TestCase):
    def test_event_threshold_metadata_keeps_only_valid_probabilities(self):
        provider = _Provider(None, enabled=False)
        provider.forecast_adapter.thresholds = {
            "attention_event_0_20m": 0.85,
            "attention_event_20_40m": "0.80",
            "invalid_high": 1.2,
            "invalid_text": "not-a-number",
        }

        self.assertEqual(
            provider._forecast_event_thresholds_metadata(),
            {
                "attention_event_0_20m": 0.85,
                "attention_event_20_40m": 0.80,
            },
        )

    def test_loaded_status_and_model_identity_invalidate_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "model_a"
            second = root / "model_b"
            _write_metrics(first, [0.5, 1.0])
            _write_metrics(second, [1.0, 1.0])
            provider = _Provider(first, enabled=True)

            np.testing.assert_allclose(
                provider._forecast_lead_reliability(2),
                [1.0, 0.5],
            )
            first_key = provider._forecast_lead_reliability_cache_key
            self.assertEqual(provider._forecast_lead_reliability_status, "loaded")

            provider.forecast_adapter.model_dir = second
            np.testing.assert_allclose(
                provider._forecast_lead_reliability(2),
                [1.0, 1.0],
            )
            self.assertNotEqual(first_key, provider._forecast_lead_reliability_cache_key)
            self.assertEqual(
                provider._forecast_lead_reliability_source,
                str(second.resolve() / "lstm_regression_metrics.csv"),
            )

    def test_content_identity_invalidates_cache_within_same_model_dir(self):
        with tempfile.TemporaryDirectory() as directory:
            model_dir = Path(directory) / "model"
            _write_metrics(model_dir, [0.5, 1.0])
            provider = _Provider(model_dir, enabled=True)
            np.testing.assert_allclose(
                provider._forecast_lead_reliability(2),
                [1.0, 0.5],
            )
            first_key = provider._forecast_lead_reliability_cache_key

            _write_metrics(model_dir, [1.0, 1.0])
            np.testing.assert_allclose(
                provider._forecast_lead_reliability(2),
                [1.0, 1.0],
            )
            self.assertNotEqual(first_key, provider._forecast_lead_reliability_cache_key)

    def test_enabled_reliability_rejects_missing_source(self):
        with tempfile.TemporaryDirectory() as directory:
            provider = _Provider(Path(directory) / "missing_model", enabled=True)

            with self.assertRaisesRegex(ValueError, "source is missing"):
                provider._forecast_lead_reliability(2)

            self.assertEqual(provider._forecast_lead_reliability_status, "missing")

    def test_enabled_reliability_rejects_shape_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            model_dir = Path(directory) / "model"
            _write_metrics(model_dir, [0.5, 1.0])
            provider = _Provider(model_dir, enabled=True)

            with self.assertRaisesRegex(ValueError, "source is shape_mismatch"):
                provider._forecast_lead_reliability(3)

            self.assertEqual(
                provider._forecast_lead_reliability_status,
                "shape_mismatch",
            )

    def test_disabled_v1_behavior_keeps_ones_for_missing_source(self):
        with tempfile.TemporaryDirectory() as directory:
            provider = _Provider(Path(directory) / "missing_model", enabled=False)

            np.testing.assert_allclose(
                provider._forecast_lead_reliability(3),
                np.ones(3),
            )
            self.assertEqual(provider._forecast_lead_reliability_status, "missing")
            self.assertEqual(
                provider._forecast_lead_reliability_metadata()["status"],
                "missing",
            )

    def test_disabled_v1_behavior_still_reads_valid_legacy_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            model_dir = Path(directory) / "model"
            _write_metrics(model_dir, [0.5, 1.0])
            provider = _Provider(model_dir, enabled=False)

            np.testing.assert_allclose(
                provider._forecast_lead_reliability(2),
                [1.0, 0.5],
            )
            self.assertEqual(provider._forecast_lead_reliability_status, "loaded")


if __name__ == "__main__":
    unittest.main()
