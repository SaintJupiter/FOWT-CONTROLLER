"""Replay-data forecast source for the compact controller."""

from __future__ import annotations

import hashlib
from datetime import datetime
from typing import Any, Mapping

import numpy as np

from .forecast_adapter import ForecastModelAdapter
from .forecast_contract import ForecastContract, validate_forecast_result
from .forecast_evidence import ForecastEvidence, evidence_from_result
from .replay_dataset import Fino1ReplayDataset, ReplaySample
from .wind_conventions import wind_observation_to_enu_downwind_ms


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

    def _sample_at(self, current_time_s: float) -> ReplaySample | None:
        """Return the replay sample whose history end is this forecast origin."""

        timestamp = self.replay_dataset.simulation_timestamp(
            self.start_timestamp,
            float(current_time_s),
        )
        sample = self.replay_dataset.sample_for_history_end(timestamp)
        if sample is None:
            return None
        if sample.history_end != timestamp:
            raise ValueError(
                "replay sample history_end must match the requested forecast origin"
            )
        return sample

    def _evidence_from_sample(self, sample: ReplaySample) -> ForecastEvidence:
        """Run the configured model once on one replay sample."""

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
                ),
            },
        )

    def source_bound_forecast_and_current_observation(
        self,
        current_time_s: float,
    ) -> tuple[ForecastEvidence, Mapping[str, Any]] | None:
        """Return one model forecast and the current wind from the same sample.

        This is intended for the new physical path. It keeps the current
        observation and the historical model input bound to one replay origin;
        it does not use the sample's future labels and does not apply any
        controller policy.
        """

        sample = self._sample_at(current_time_s)
        if sample is None:
            return None
        evidence = self._evidence_from_sample(sample)
        if evidence.origin_time != sample.history_end.strftime("%Y-%m-%d %H:%M:%S"):
            raise ValueError("forecast evidence origin must match replay sample history_end")
        return evidence, dict(sample.wind_obs)

    def source_bound_forecast_and_current_enu_wind(
        self,
        current_time_s: float,
    ) -> tuple[ForecastEvidence, np.ndarray] | None:
        """Return model evidence and the same-sample current ENU wind vector.

        The returned vector is still a recorded observation at the forecast
        origin.  It is not a rotor-plane wind or a platform-axis velocity.
        """

        source_bound = self.source_bound_forecast_and_current_observation(
            current_time_s
        )
        if source_bound is None:
            return None
        evidence, wind_observation = source_bound
        enu_downwind = np.asarray(
            wind_observation_to_enu_downwind_ms(wind_observation),
            dtype=float,
        )
        enu_downwind.setflags(write=False)
        return evidence, enu_downwind

    def __call__(
        self,
        current_time_s: float,
        wind_obs: Mapping[str, Any],
    ) -> ForecastEvidence | None:
        del wind_obs
        source_bound = self.source_bound_forecast_and_current_observation(current_time_s)
        if source_bound is None:
            return None
        return source_bound[0]


__all__ = ["ReplayForecastEvidenceSource"]
