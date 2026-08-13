import unittest
from datetime import datetime, timedelta
from types import SimpleNamespace

import numpy as np

from wind_prediction.controller_replay_adapter import ReplayForecastEvidenceSource
from wind_prediction.forecast_adapter import ForecastResult


class _Dataset:
    future_steps = 6
    event_columns = ("attention_event",)
    update_interval_s = 600.0

    def __init__(self, sample):
        self.sample = sample

    @staticmethod
    def simulation_timestamp(start_timestamp, current_time_s):
        return start_timestamp + timedelta(seconds=float(current_time_s))

    def sample_for_history_end(self, timestamp):
        del timestamp
        return self.sample


class _Adapter:
    model_version = "fake_lstm"
    provides_future_preview = True
    thresholds = {"attention_event": 0.6}

    @staticmethod
    def predict_window(x_window, timestamp=None):
        del x_window
        uv = np.array([(0.0, -10.0)] * 6, dtype=np.float32)
        return ForecastResult(
            wind_uv_raw=uv,
            wind_speed=np.full(6, 10.0, dtype=np.float32),
            wind_dir_deg=np.zeros(6, dtype=np.float32),
            event_probs={"attention_event": 0.8},
            risk_levels={"attention_event": True},
            model_version="fake_lstm",
            timestamp=timestamp,
        )


class ReplayForecastEvidenceSourceTests(unittest.TestCase):
    def test_source_uses_model_output_and_never_oracle_future(self):
        origin = datetime(2026, 8, 12, 0, 0, 0)
        sample = SimpleNamespace(
            series_id="synthetic-series",
            x_window=np.zeros((12, 4), dtype=np.float32),
            history_start=origin - timedelta(minutes=110),
            history_end=origin,
            y_uv_raw=np.full((6, 2), 999.0, dtype=np.float32),
        )
        source = ReplayForecastEvidenceSource(
            replay_dataset=_Dataset(sample),
            start_timestamp=origin,
            forecast_adapter=_Adapter(),
            lead_reliability=np.linspace(1.0, 0.7, 6),
        )

        evidence = source(0.0, {"ws": 10.0, "wd_deg": 0.0})

        self.assertIsNotNone(evidence)
        self.assertEqual(evidence.source, "fake_lstm")
        self.assertTrue(evidence.provides_future_preview)
        np.testing.assert_allclose(evidence.uv_ms[:, 1], -10.0)
        self.assertNotEqual(float(evidence.uv_ms[0, 0]), 999.0)
        self.assertEqual(evidence.metadata["event_thresholds"]["attention_event"], 0.6)
        self.assertEqual(evidence.metadata["series_id"], "synthetic-series")
        self.assertEqual(len(evidence.metadata["input_window_sha256"]), 64)

    def test_missing_replay_sample_returns_no_evidence(self):
        origin = datetime(2026, 8, 12, 0, 0, 0)
        source = ReplayForecastEvidenceSource(
            replay_dataset=_Dataset(None),
            start_timestamp=origin,
            forecast_adapter=_Adapter(),
        )
        self.assertIsNone(source(0.0, {"ws": 10.0, "wd_deg": 0.0}))

    def test_source_requires_an_explicit_forecast_adapter(self):
        with self.assertRaisesRegex(ValueError, "forecast_adapter is required"):
            ReplayForecastEvidenceSource(
                replay_dataset=_Dataset(None),
                start_timestamp=datetime(2026, 8, 12),
                forecast_adapter=None,
            )


if __name__ == "__main__":
    unittest.main()
