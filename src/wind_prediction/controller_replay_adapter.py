"""Replay-data forecast source for the compact controller."""

from __future__ import annotations

import hashlib
from datetime import datetime
from typing import Any, Mapping

import numpy as np

from .forecast_adapter import ForecastModelAdapter
from .forecast_contract import ForecastContract, validate_forecast_result
from .forecast_evidence import ForecastEvidence, evidence_from_result
from .replay_dataset import Fino1ReplayDataset


def _array_sha256(values: Any) -> str:
    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(repr(tuple(array.shape)).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


class ReplayForecastEvidenceSource:
    """Build model evidence at a simulation time without oracle fallback."""

    def __init__(
        self,
        *,
        replay_dataset: Fino1ReplayDataset,
        start_timestamp: datetime,
        forecast_adapter: ForecastModelAdapter,
        lead_reliability: np.ndarray | None = None,
    ) -> None:
        if forecast_adapter is None:
            raise ValueError(
                "forecast_adapter is required; use an explicit current-only adapter "
                "for the no-future baseline"
            )
        self.replay_dataset = replay_dataset
        self.start_timestamp = start_timestamp
        self.forecast_adapter = forecast_adapter
        self.lead_reliability = (
            None
            if lead_reliability is None
            else np.asarray(lead_reliability, dtype=float).reshape(-1).copy()
        )

    def __call__(
        self,
        current_time_s: float,
        wind_obs: Mapping[str, Any],
    ) -> ForecastEvidence | None:
        del wind_obs
        timestamp = self.replay_dataset.simulation_timestamp(
            self.start_timestamp,
            float(current_time_s),
        )
        sample = self.replay_dataset.sample_for_history_end(timestamp)
        if sample is None:
            return None
        forecast = self.forecast_adapter.predict_window(
            sample.x_window,
            timestamp=sample.history_end.strftime("%Y-%m-%d %H:%M:%S"),
        )
        contract = ForecastContract(
            future_steps=int(self.replay_dataset.future_steps),
            event_columns=tuple(self.replay_dataset.event_columns),
            require_event_probs=False,
        )
        forecast = validate_forecast_result(forecast, contract)
        reliability = self.lead_reliability
        if reliability is not None and reliability.size != forecast.wind_uv_raw.shape[0]:
            raise ValueError("lead_reliability length does not match forecast horizon")
        return evidence_from_result(
            forecast,
            source=str(forecast.model_version),
            sample_period_s=float(self.replay_dataset.update_interval_s),
            provides_future_preview=bool(
                getattr(self.forecast_adapter, "provides_future_preview", True)
            ),
            lead_reliability=reliability,
            metadata={
                "series_id": sample.series_id,
                "history_start": sample.history_start.strftime("%Y-%m-%d %H:%M:%S"),
                "history_end": sample.history_end.strftime("%Y-%m-%d %H:%M:%S"),
                "input_window_sha256": _array_sha256(sample.x_window),
                "event_thresholds": dict(
                    getattr(self.forecast_adapter, "thresholds", {}) or {}
                )
            },
        )


__all__ = ["ReplayForecastEvidenceSource"]
