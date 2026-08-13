"""Thin adapter between the compact controller and the existing plant runner."""

from __future__ import annotations

import json
import hashlib
import math
from collections.abc import Callable, Mapping
from typing import Any

import numpy as np

from .controller_core import ControlAction, ControlCoreConfig
from .controller_runtime import (
    ControllerMeasurements,
    ControllerRuntimeState,
    ForecastAssistedBallastController,
)
from .execution_rollout import ExecutionRolloutState
from .forecast_evidence import ForecastEvidence


ForecastEvidenceSource = Callable[[float, Mapping[str, Any]], ForecastEvidence | None]


def wind_observation_to_uv_ms(wind_obs: Mapping[str, Any]) -> tuple[float, float]:
    """Convert speed and meteorological direction to the model UV convention."""

    try:
        speed = float(wind_obs["ws"])
        direction_deg = float(wind_obs["wd_deg"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("wind_obs must contain numeric ws and wd_deg") from exc
    if not math.isfinite(speed) or speed < 0.0 or not math.isfinite(direction_deg):
        raise ValueError("wind speed and direction must be finite and speed non-negative")
    direction_rad = math.radians(direction_deg)
    return (
        -speed * math.sin(direction_rad),
        -speed * math.cos(direction_rad),
    )


def _strict_three(name: str, values: Any) -> np.ndarray:
    array = np.asarray(values, dtype=float).reshape(-1)
    if array.size != 3 or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain exactly three finite values")
    return array.copy()


def _json_payload_sha256(document: Mapping[str, Any]) -> str:
    payload = json.dumps(
        dict(document),
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _json_numeric_vector(values: Any) -> list[float | str]:
    result: list[float | str] = []
    for value in np.asarray(values, dtype=float).reshape(-1):
        number = float(value)
        if number == math.inf:
            result.append("positive_infinity")
        elif number == -math.inf:
            result.append("negative_infinity")
        elif math.isnan(number):
            raise ValueError("decision state contains NaN")
        else:
            result.append(number)
    return result


def forecast_payload_sha256(forecast: ForecastEvidence | None) -> str | None:
    if forecast is None:
        return None
    document = {
        "source": forecast.source,
        "model_version": forecast.model_version,
        "origin_time": forecast.origin_time,
        "sample_period_s": float(forecast.sample_period_s),
        "uv_ms": np.asarray(forecast.uv_ms, dtype=float).tolist(),
        "event_probs": {
            str(key): float(value)
            for key, value in sorted(forecast.event_probs.items())
        },
        "lead_reliability": np.asarray(
            forecast.lead_reliability,
            dtype=float,
        ).tolist(),
        "provides_future_preview": bool(forecast.provides_future_preview),
        "metadata": dict(forecast.metadata),
    }
    return _json_payload_sha256(document)


def decision_state_sha256(
    *,
    state: Any,
    wind_observation: Mapping[str, Any],
    runtime_state: ControllerRuntimeState,
) -> str:
    execution = runtime_state.execution
    document = {
        "platform_state": _json_numeric_vector(state),
        "wind_observation": {
            "speed_ms": float(wind_observation["ws"]),
            "direction_deg": float(wind_observation["wd_deg"]),
        },
        "controller_runtime": {
            "cycle_index": int(runtime_state.cycle_index),
            "target_revision": int(runtime_state.target_revision),
            "last_action": runtime_state.last_action,
            "last_decision_time_s": runtime_state.last_decision_time_s,
        },
        "execution": {
            "masses_kg": _json_numeric_vector(execution.masses_kg),
            "target_masses_kg": _json_numeric_vector(execution.target_masses_kg),
            "primary_target_kg": _json_numeric_vector(execution.primary_target_kg),
            "pump_rates_m3_min": _json_numeric_vector(
                execution.pump_rates_m3_min
            ),
            "pump_latched": np.asarray(
                execution.pump_latched, dtype=bool
            ).astype(int).tolist(),
            "pump_on_elapsed_s": _json_numeric_vector(
                execution.pump_on_elapsed_s
            ),
            "pump_off_elapsed_s": _json_numeric_vector(
                execution.pump_off_elapsed_s
            ),
            "pump_near_target_s": _json_numeric_vector(
                execution.pump_near_target_s
            ),
            "pump_command_rates_m3_min": _json_numeric_vector(
                execution.pump_command_rates_m3_min
            ),
            "last_flow_directions": _json_numeric_vector(
                execution.last_flow_directions
            ),
        },
    }
    return _json_payload_sha256(document)


def candidate_ranking_sha256(decision) -> str:
    ranking = [
        {
            "sequence": [action.value for action in candidate.sequence],
            "score": candidate.score,
            "residual_cost": candidate.residual_cost,
            "terminal_residual_cost": candidate.terminal_residual_cost,
            "pump_volume_m3": candidate.transferred_volume_m3,
            "active_time_s": candidate.active_time_s,
            "starts": candidate.starts,
            "direction_switches": candidate.direction_switches,
            "first_action_vector_deg": list(candidate.first_action_vector_deg),
            "first_target_masses_kg": list(candidate.first_target_masses_kg),
        }
        for candidate in decision.ranked_candidates
    ]
    return _json_payload_sha256({"ranked_candidates": ranking})


class CompactControllerPlantAdapter:
    """Expose the V2 controller through the legacy plant preview interface.

    Candidate evaluation uses the actuator model in the compact controller.
    Persistent tank and pump state, however, is rebuilt from plant feedback at
    every decision instant so simulated execution never replaces observation.
    """

    def __init__(
        self,
        *,
        initial_tank_masses_kg: Any,
        forecast_source: ForecastEvidenceSource | None,
        config: ControlCoreConfig | None = None,
    ) -> None:
        resolved_config = ControlCoreConfig() if config is None else config
        if resolved_config.execution.target_slew_enabled:
            raise ValueError(
                "CompactControllerPlantAdapter does not support target slew; "
                "disable it until the connected command path owns the same "
                "rate limiter used by candidate execution rollout"
            )
        self.controller = ForecastAssistedBallastController(resolved_config)
        self.forecast_source = forecast_source
        self._initial_masses_kg = _strict_three(
            "initial_tank_masses_kg",
            initial_tank_masses_kg,
        )
        self.records: list[dict[str, Any]] = []
        self.reset()

    @property
    def update_interval_s(self) -> float:
        return float(self.controller.config.stage_duration_s)

    @property
    def target_revision(self) -> int:
        return int(self._runtime_state.target_revision)

    def reset(self) -> None:
        self._last_bucket: int | None = None
        self._active_target_kg = self._initial_masses_kg.copy()
        self._runtime_state = ControllerRuntimeState.initialize(
            self._initial_masses_kg
        )
        self._last_output: dict[str, Any] | None = None
        self.records.clear()

    def compute(
        self,
        state: Any,
        wind_obs: Mapping[str, Any],
        plant_info_prev: Mapping[str, Any] | None,
        current_time: float,
    ) -> dict[str, Any]:
        """Return the exact target-water command expected by the plant policy."""

        time_s = float(current_time)
        if not math.isfinite(time_s) or time_s < 0.0:
            raise ValueError("current_time must be finite and non-negative")
        bucket = int(math.floor(time_s / self.update_interval_s + 1e-9))
        plant_info = dict(plant_info_prev or {})
        masses = _strict_three(
            "plant tank_masses",
            plant_info.get("tank_masses", self._initial_masses_kg),
        )
        if bucket == self._last_bucket and self._last_output is not None:
            return self._output_for_current_masses(self._last_output, masses)

        posture, posture_rate = self._posture_from_state(state)
        current_uv = wind_observation_to_uv_ms(wind_obs)
        execution = self._execution_state_from_feedback(plant_info, masses)
        runtime_state = ControllerRuntimeState(
            execution=execution,
            cycle_index=self._runtime_state.cycle_index,
            target_revision=self._runtime_state.target_revision,
            last_action=self._runtime_state.last_action,
            last_decision_time_s=self._runtime_state.last_decision_time_s,
        )
        forecast = (
            None
            if self.forecast_source is None
            else self.forecast_source(time_s, wind_obs)
        )
        measurements = ControllerMeasurements(
            time_s=time_s,
            posture_deg=posture,
            posture_rate_deg_s=posture_rate,
            current_wind_uv_ms=current_uv,
        )
        input_state_sha256 = decision_state_sha256(
            state=state,
            wind_observation=wind_obs,
            runtime_state=runtime_state,
        )
        decision = self.controller.decide(measurements, runtime_state, forecast)
        target = np.asarray(decision.target_masses_kg, dtype=float)
        changed = not np.array_equal(target, self._active_target_kg)
        self._active_target_kg = target.copy()
        self._runtime_state = ControllerRuntimeState(
            execution=execution,
            cycle_index=runtime_state.cycle_index + 1,
            target_revision=runtime_state.target_revision + int(changed),
            last_action=decision.action,
            last_decision_time_s=time_s,
        )
        output = self._build_output(decision, masses, forecast)
        self._last_output = output
        self._last_bucket = bucket
        self.records.append(
            {
                "time_s": time_s,
                "bucket": bucket,
                "selected_action": decision.action.value,
                "target_operation": decision.target_operation.value,
                "forecast_available": int(decision.context.forecast_available),
                "forecast_source": None if forecast is None else forecast.source,
                "forecast_model_version": (
                    None if forecast is None else forecast.model_version
                ),
                "forecast_origin_time": (
                    None if forecast is None else forecast.origin_time
                ),
                "forecast_payload_sha256": forecast_payload_sha256(forecast),
                "forecast_input_window_sha256": (
                    None
                    if forecast is None
                    else forecast.metadata.get("input_window_sha256")
                ),
                "decision_state_sha256": input_state_sha256,
                "candidate_ranking_sha256": candidate_ranking_sha256(decision),
                "target_revision": self._runtime_state.target_revision,
                "target_masses_kg": target.tolist(),
                "trace": decision.as_trace(),
            }
        )
        return dict(output)

    def _execution_state_from_feedback(
        self,
        plant_info: dict[str, Any],
        masses: np.ndarray,
    ) -> ExecutionRolloutState:
        info = dict(plant_info)
        info["tank_masses"] = masses
        info.setdefault("target_ballast_mass", self._active_target_kg)
        info["primary_target_kg"] = self._active_target_kg
        return ExecutionRolloutState.from_plant_info(
            info,
            self.controller.config.execution,
        )

    @staticmethod
    def _posture_from_state(state: Any) -> tuple[tuple[float, float], tuple[float, float]]:
        values = np.asarray(state, dtype=float).reshape(-1)
        if values.size != 12 or not np.all(np.isfinite(values)):
            raise ValueError("platform state must contain exactly 12 finite values")
        posture = (math.degrees(values[4]), math.degrees(values[3]))
        posture_rate = (math.degrees(values[10]), math.degrees(values[9]))
        return posture, posture_rate

    def _build_output(
        self,
        decision,
        masses: np.ndarray,
        forecast: ForecastEvidence | None,
    ) -> dict[str, Any]:
        target = np.asarray(decision.target_masses_kg, dtype=float)
        delta = target - masses
        active = int(np.any(np.abs(delta) > self.controller.config.execution.stop_error_kg))
        return {
            "source": "compact_controller_v2",
            "pitch_bias_deg": 0.0,
            "roll_bias_deg": 0.0,
            "preview_pitch_bias_deg": 0.0,
            "preview_roll_bias_deg": 0.0,
            "preview_primary_enabled": 1,
            "preview_primary_active": active,
            "preview_primary_target_kg": target.tolist(),
            "preview_primary_delta_kg": delta.tolist(),
            "preview_primary_action": decision.action.value,
            "preview_target_operation": decision.target_operation.value,
            "preview_forecast_has_future": int(
                forecast is not None and forecast.provides_future_preview
            ),
            "preview_controller_v2_trace_json": json.dumps(
                decision.as_trace(),
                ensure_ascii=True,
                separators=(",", ":"),
                allow_nan=False,
            ),
        }

    def _output_for_current_masses(
        self,
        cached: dict[str, Any],
        masses: np.ndarray,
    ) -> dict[str, Any]:
        output = dict(cached)
        delta = self._active_target_kg - masses
        output["preview_primary_delta_kg"] = delta.tolist()
        output["preview_primary_active"] = int(
            np.any(np.abs(delta) > self.controller.config.execution.stop_error_kg)
        )
        return output


__all__ = [
    "CompactControllerPlantAdapter",
    "ForecastEvidenceSource",
    "candidate_ranking_sha256",
    "decision_state_sha256",
    "forecast_payload_sha256",
    "wind_observation_to_uv_ms",
]
