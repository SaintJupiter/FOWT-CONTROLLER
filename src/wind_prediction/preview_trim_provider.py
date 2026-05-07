from __future__ import annotations

from datetime import datetime

import numpy as np

from .candidate_plan_evaluator import CandidatePlanEvaluator
from .forecast_adapter import ForecastModelAdapter, ForecastResult
from .replay_dataset import Fino1ReplayDataset, ReplaySample


class _BaseReplayPreviewProvider:
    """Passive preview provider.

    This class keeps the control-chain interface wired, but intentionally does
    not convert forecast output into control actions. It only records what the
    forecast front-end would have supplied at each 10-minute update.
    """

    def __init__(
        self,
        replay_dataset: Fino1ReplayDataset,
        start_timestamp: datetime,
    ) -> None:
        self.replay_dataset = replay_dataset
        self.start_timestamp = start_timestamp
        self.update_interval_s = int(replay_dataset.update_interval_s)
        self._last_bucket = -1
        self.records: list[dict] = []
        self._cached_bias = self._zero_bias("preview_uninitialized")

    @staticmethod
    def _zero_bias(source: str) -> dict:
        return {
            "pitch_bias_deg": 0.0,
            "roll_bias_deg": 0.0,
            "source": str(source),
            "preview_gain": 0.0,
            "risk_level": "passive",
            "preview_mode": "passive_feed_only",
        }

    def reset(self) -> None:
        self._last_bucket = -1
        self.records = []
        self._cached_bias = self._zero_bias("preview_reset")

    def _bucket_timestamp(self, current_time: float) -> datetime:
        return self.replay_dataset.simulation_timestamp(self.start_timestamp, current_time)

    def _sample_for_time(self, current_time: float) -> ReplaySample | None:
        return self.replay_dataset.sample_for_history_end(self._bucket_timestamp(current_time))

    def _build_forecast(self, sample: ReplaySample) -> ForecastResult:
        raise NotImplementedError

    @staticmethod
    def _record_from_forecast(sample: ReplaySample, forecast: ForecastResult) -> dict:
        rec = {
            "history_end": sample.history_end.strftime("%Y-%m-%d %H:%M:%S"),
            "future_start": sample.future_start.strftime("%Y-%m-%d %H:%M:%S"),
            "future_end": sample.future_end.strftime("%Y-%m-%d %H:%M:%S"),
            "current_ws": float(sample.wind_obs["ws"]),
            "current_wd_deg": float(sample.wind_obs["wd_deg"]),
            "forecast_source": str(forecast.model_version),
        }
        for idx, (speed, direction) in enumerate(zip(forecast.wind_speed, forecast.wind_dir_deg), start=1):
            rec[f"pred_ws_tplus_{idx*10}m"] = float(speed)
            rec[f"pred_wd_tplus_{idx*10}m"] = float(direction)
        for name, prob in forecast.event_probs.items():
            rec[f"prob_{name}"] = float(prob)
        for name, flag in forecast.risk_levels.items():
            rec[f"flag_{name}"] = int(bool(flag))
        return rec

    def compute(self, state, wind_obs, plant_info_prev, current_time):
        bucket = int(np.floor(max(float(current_time), 0.0) / self.update_interval_s + 1e-9))
        if bucket == self._last_bucket:
            return dict(self._cached_bias)
        self._last_bucket = bucket

        sample = self._sample_for_time(current_time)
        if sample is None:
            self._cached_bias = self._zero_bias("preview_no_sample")
            return dict(self._cached_bias)

        forecast = self._build_forecast(sample)
        self.records.append(self._record_from_forecast(sample, forecast))
        self._cached_bias = self._zero_bias(f"{forecast.model_version}_passive")
        self._cached_bias["replay_timestamp"] = sample.history_end.strftime("%Y-%m-%d %H:%M:%S")
        self._cached_bias["ballast_attention_event_prob"] = float(
            forecast.event_probs.get("ballast_attention_event", 0.0)
        )
        return dict(self._cached_bias)


class ModelPreviewProvider(_BaseReplayPreviewProvider):
    def __init__(
        self,
        replay_dataset: Fino1ReplayDataset,
        forecast_adapter: ForecastModelAdapter,
        start_timestamp: datetime,
    ) -> None:
        super().__init__(replay_dataset=replay_dataset, start_timestamp=start_timestamp)
        self.forecast_adapter = forecast_adapter

    def _build_forecast(self, sample: ReplaySample) -> ForecastResult:
        return self.forecast_adapter.predict_window(
            sample.x_window,
            timestamp=sample.history_end.strftime("%Y-%m-%d %H:%M:%S"),
        )


class OraclePreviewProvider(_BaseReplayPreviewProvider):
    def _build_forecast(self, sample: ReplaySample) -> ForecastResult:
        speed = np.sqrt(sample.y_uv_raw[:, 0] ** 2 + sample.y_uv_raw[:, 1] ** 2).astype(np.float32)
        direction = (
            np.rad2deg(np.arctan2(-sample.y_uv_raw[:, 0], -sample.y_uv_raw[:, 1])) + 360.0
        ) % 360.0
        event_probs = {
            name: float(sample.y_event[idx])
            for idx, name in enumerate(self.replay_dataset.event_columns)
        }
        return ForecastResult(
            wind_uv_raw=np.array(sample.y_uv_raw, dtype=np.float32, copy=True),
            wind_speed=speed.astype(np.float32),
            wind_dir_deg=direction.astype(np.float32),
            event_probs=event_probs,
            risk_levels={name: bool(value >= 0.5) for name, value in event_probs.items()},
            model_version="oracle_feed",
            timestamp=sample.history_end.strftime("%Y-%m-%d %H:%M:%S"),
        )


class _BaseReplayDecisionProvider:
    """Active slow-layer decision provider for the closed-only baseline."""

    def __init__(
        self,
        replay_dataset: Fino1ReplayDataset,
        start_timestamp: datetime,
        evaluator: CandidatePlanEvaluator | None = None,
    ) -> None:
        self.replay_dataset = replay_dataset
        self.start_timestamp = start_timestamp
        self.update_interval_s = int(replay_dataset.update_interval_s)
        self.evaluator = evaluator if evaluator is not None else CandidatePlanEvaluator()
        self._last_bucket = -1
        self._cached_output = self._zero_output("decision_uninitialized")
        self.records: list[dict] = []

    @staticmethod
    def _zero_output(source: str) -> dict:
        return {
            "pitch_bias_deg": 0.0,
            "roll_bias_deg": 0.0,
            "pitch_sp_deg": 0.0,
            "roll_sp_deg": 0.0,
            "source": str(source),
            "preview_mode": "candidate_plan_evaluation",
            "risk_level": "idle",
            "selected_plan": "hold",
            "decision_activation": 0.0,
            "cost_total": 0.0,
        }

    def reset(self) -> None:
        self._last_bucket = -1
        self._cached_output = self._zero_output("decision_reset")
        self.records = []
        if hasattr(self.evaluator, "reset"):
            self.evaluator.reset()

    def _bucket_timestamp(self, current_time: float) -> datetime:
        return self.replay_dataset.simulation_timestamp(self.start_timestamp, current_time)

    def _sample_for_time(self, current_time: float) -> ReplaySample | None:
        return self.replay_dataset.sample_for_history_end(self._bucket_timestamp(current_time))

    def _build_forecast(self, sample: ReplaySample) -> ForecastResult:
        raise NotImplementedError

    def compute(self, state, wind_obs, plant_info_prev, current_time):
        bucket = int(np.floor(max(float(current_time), 0.0) / self.update_interval_s + 1e-9))
        if bucket == self._last_bucket:
            return dict(self._cached_output)
        self._last_bucket = bucket

        sample = self._sample_for_time(current_time)
        if sample is None:
            self._cached_output = self._zero_output("decision_no_sample")
            return dict(self._cached_output)

        forecast = self._build_forecast(sample)
        decision = self.evaluator.evaluate(
            state=np.asarray(state, dtype=float),
            wind_obs=wind_obs,
            plant_info_prev=plant_info_prev,
            forecast=forecast,
        )
        decision["replay_timestamp"] = sample.history_end.strftime("%Y-%m-%d %H:%M:%S")
        decision["forecast_source"] = str(forecast.model_version)
        self._cached_output = dict(decision)

        record = {
            "history_end": sample.history_end.strftime("%Y-%m-%d %H:%M:%S"),
            "current_ws": float(sample.wind_obs["ws"]),
            "current_wd_deg": float(sample.wind_obs["wd_deg"]),
            "forecast_source": str(forecast.model_version),
            "selected_plan": str(decision.get("selected_plan", "hold")),
            "selection_mode": str(decision.get("selection_mode", "")),
            "pitch_sp_deg": float(decision.get("pitch_sp_deg", 0.0)),
            "roll_sp_deg": float(decision.get("roll_sp_deg", 0.0)),
            "decision_activation": float(decision.get("decision_activation", 0.0)),
            "cost_total": float(decision.get("cost_total", 0.0)),
            "risk_0_20m": float(decision.get("segment_risk_0_20m", 0.0)),
            "risk_20_40m": float(decision.get("segment_risk_20_40m", 0.0)),
            "risk_40_60m": float(decision.get("segment_risk_40_60m", 0.0)),
            "ballast_attention_event_prob": float(
                decision.get("ballast_attention_event_prob", 0.0)
            ),
            "candidate_score_snapshot": str(decision.get("candidate_score_snapshot", "")),
            "screened_snapshot": str(decision.get("screened_snapshot", "")),
        }
        for idx, (speed, direction) in enumerate(zip(forecast.wind_speed, forecast.wind_dir_deg), start=1):
            record[f"pred_ws_tplus_{idx*10}m"] = float(speed)
            record[f"pred_wd_tplus_{idx*10}m"] = float(direction)
        self.records.append(record)
        return dict(self._cached_output)


class ModelDecisionProvider(_BaseReplayDecisionProvider):
    def __init__(
        self,
        replay_dataset: Fino1ReplayDataset,
        forecast_adapter: ForecastModelAdapter,
        start_timestamp: datetime,
        evaluator: CandidatePlanEvaluator | None = None,
    ) -> None:
        super().__init__(
            replay_dataset=replay_dataset,
            start_timestamp=start_timestamp,
            evaluator=evaluator,
        )
        self.forecast_adapter = forecast_adapter

    def _build_forecast(self, sample: ReplaySample) -> ForecastResult:
        return self.forecast_adapter.predict_window(
            sample.x_window,
            timestamp=sample.history_end.strftime("%Y-%m-%d %H:%M:%S"),
        )


class OracleDecisionProvider(_BaseReplayDecisionProvider):
    def _build_forecast(self, sample: ReplaySample) -> ForecastResult:
        speed = np.sqrt(sample.y_uv_raw[:, 0] ** 2 + sample.y_uv_raw[:, 1] ** 2).astype(np.float32)
        direction = (
            np.rad2deg(np.arctan2(-sample.y_uv_raw[:, 0], -sample.y_uv_raw[:, 1])) + 360.0
        ) % 360.0
        event_probs = {
            name: float(sample.y_event[idx])
            for idx, name in enumerate(self.replay_dataset.event_columns)
        }
        return ForecastResult(
            wind_uv_raw=np.array(sample.y_uv_raw, dtype=np.float32, copy=True),
            wind_speed=speed.astype(np.float32),
            wind_dir_deg=direction.astype(np.float32),
            event_probs=event_probs,
            risk_levels={name: bool(value >= 0.5) for name, value in event_probs.items()},
            model_version="oracle_decision",
            timestamp=sample.history_end.strftime("%Y-%m-%d %H:%M:%S"),
        )
