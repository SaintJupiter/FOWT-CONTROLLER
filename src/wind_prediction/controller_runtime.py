"""Single-cycle runtime for the compact ballast controller.

This module owns controller state progression only. Platform dynamics remain
outside the interface and can later be supplied by a calibrated simulator or
measured feedback without changing candidate selection.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np

from .controller_core import (
    ControlAction,
    ControlCoreConfig,
    ControlDecision,
    ControlObservation,
    decide_control_cycle,
)
from .execution_rollout import ExecutionRolloutState
from .forecast_evidence import ForecastEvidence


def _finite_vector(name: str, values: Any, size: int) -> tuple[float, ...]:
    try:
        normalized = tuple(float(value) for value in values)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain numeric values") from exc
    if len(normalized) != size or not all(math.isfinite(value) for value in normalized):
        raise ValueError(f"{name} must contain exactly {size} finite values")
    return normalized


@dataclass(frozen=True)
class ControllerMeasurements:
    """Measured inputs for one decision instant."""

    time_s: float
    posture_deg: tuple[float, float]
    posture_rate_deg_s: tuple[float, float]
    current_wind_uv_ms: tuple[float, float]

    def __post_init__(self) -> None:
        if not math.isfinite(float(self.time_s)):
            raise ValueError("time_s must be finite")
        object.__setattr__(
            self,
            "posture_deg",
            _finite_vector("posture_deg", self.posture_deg, 2),
        )
        object.__setattr__(
            self,
            "posture_rate_deg_s",
            _finite_vector("posture_rate_deg_s", self.posture_rate_deg_s, 2),
        )
        object.__setattr__(
            self,
            "current_wind_uv_ms",
            _finite_vector("current_wind_uv_ms", self.current_wind_uv_ms, 2),
        )


@dataclass(frozen=True)
class ControllerRuntimeState:
    """Minimal persistent state shared by consecutive control cycles."""

    execution: ExecutionRolloutState
    cycle_index: int = 0
    target_revision: int = 0
    last_action: ControlAction | None = None
    last_decision_time_s: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.execution, ExecutionRolloutState):
            raise TypeError("execution must be an ExecutionRolloutState")
        if not isinstance(self.cycle_index, int) or self.cycle_index < 0:
            raise ValueError("cycle_index must be a non-negative integer")
        if not isinstance(self.target_revision, int) or self.target_revision < 0:
            raise ValueError("target_revision must be a non-negative integer")
        if self.last_action is not None:
            object.__setattr__(self, "last_action", ControlAction(self.last_action))
        if self.last_decision_time_s is not None and not math.isfinite(
            float(self.last_decision_time_s)
        ):
            raise ValueError("last_decision_time_s must be finite when provided")

    @classmethod
    def initialize(
        cls,
        tank_masses_kg: Any,
        *,
        pump_rates_m3_min: Any = (0.0, 0.0, 0.0),
        pump_latched: Any = (False, False, False),
    ) -> "ControllerRuntimeState":
        masses = np.asarray(
            _finite_vector("tank_masses_kg", tank_masses_kg, 3),
            dtype=float,
        )
        rates = np.asarray(
            _finite_vector("pump_rates_m3_min", pump_rates_m3_min, 3),
            dtype=float,
        )
        latched = tuple(bool(value) for value in pump_latched)
        if len(latched) != 3:
            raise ValueError("pump_latched must contain exactly three values")
        execution = ExecutionRolloutState(
            masses_kg=masses,
            target_masses_kg=masses,
            primary_target_kg=masses,
            pump_rates_m3_min=rates,
            pump_latched=np.asarray(latched, dtype=bool),
        )
        return cls(execution=execution)


@dataclass(frozen=True)
class ControllerCycleResult:
    """One committed decision plus the actuator state after one interval."""

    decision: ControlDecision
    state: ControllerRuntimeState
    trace: Mapping[str, Any] = field(compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "trace", MappingProxyType(dict(self.trace)))


class ForecastAssistedBallastController:
    """Public V2 controller entry with one deterministic cycle method."""

    def __init__(
        self,
        config: ControlCoreConfig | None = None,
        *,
        config_sha256: str | None = None,
    ) -> None:
        self.config = ControlCoreConfig() if config is None else config
        self.config_sha256 = str(config_sha256 or "in-memory-default")

    @classmethod
    def from_config_file(
        cls,
        path: str | Path,
    ) -> "ForecastAssistedBallastController":
        from .controller_configuration import load_controller_config

        loaded = load_controller_config(path)
        return cls(loaded.config, config_sha256=loaded.sha256)

    def decide(
        self,
        measurements: ControllerMeasurements,
        state: ControllerRuntimeState,
        forecast: ForecastEvidence | None,
    ) -> ControlDecision:
        self._validate_time(measurements, state)
        observation = ControlObservation(
            time_s=measurements.time_s,
            posture_deg=measurements.posture_deg,
            posture_rate_deg_s=measurements.posture_rate_deg_s,
            current_wind_uv_ms=measurements.current_wind_uv_ms,
            execution_state=state.execution,
        )
        return decide_control_cycle(observation, forecast, self.config)

    def step(
        self,
        measurements: ControllerMeasurements,
        state: ControllerRuntimeState,
        forecast: ForecastEvidence | None,
    ) -> ControllerCycleResult:
        """Plan once and advance the built-in three-pump execution model once."""

        decision = self.decide(measurements, state, forecast)
        next_execution = decision.first_execution.state
        old_target = np.asarray(state.execution.primary_target_masses_kg, dtype=float)
        new_target = np.asarray(next_execution.primary_target_masses_kg, dtype=float)
        target_changed = not np.array_equal(old_target, new_target)
        next_state = ControllerRuntimeState(
            execution=next_execution,
            cycle_index=state.cycle_index + 1,
            target_revision=state.target_revision + int(target_changed),
            last_action=decision.action,
            last_decision_time_s=float(measurements.time_s),
        )
        trace = decision.as_trace()
        trace["cycle"] = {
            "index": next_state.cycle_index,
            "time_s": float(measurements.time_s),
            "target_revision": next_state.target_revision,
        }
        trace["state_after_execution"] = {
            "tank_masses_kg": next_execution.actual_masses_kg.tolist(),
            "active_target_kg": next_execution.primary_target_masses_kg.tolist(),
            "shaped_target_kg": next_execution.rate_limited_target_kg.tolist(),
            "remaining_target_error_kg": (
                next_execution.primary_target_masses_kg
                - next_execution.actual_masses_kg
            ).tolist(),
            "pump_rates_m3_min": next_execution.signed_flow_m3_min.tolist(),
            "pump_latched": next_execution.pump_latched.astype(bool).tolist(),
        }
        return ControllerCycleResult(
            decision=decision,
            state=next_state,
            trace=trace,
        )

    @staticmethod
    def _validate_time(
        measurements: ControllerMeasurements,
        state: ControllerRuntimeState,
    ) -> None:
        previous = state.last_decision_time_s
        if previous is not None and float(measurements.time_s) <= float(previous):
            raise ValueError("measurement time must advance between control cycles")


__all__ = [
    "ControllerCycleResult",
    "ControllerMeasurements",
    "ControllerRuntimeState",
    "ForecastAssistedBallastController",
]
