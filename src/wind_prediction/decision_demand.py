"""Explicit formation of the V2 controller's historical demand proxy.

This module deliberately preserves the existing empirical wind-to-demand
relation while keeping that calculation outside candidate selection. It is
not a platform-response model, nor does it define a physics-informed demand
interface. Before a physical mapping can replace this proxy, its output
quantity and units must be fixed and its relation to action-to-mass mapping
and residual scoring must be reviewed explicitly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np


_EPS = 1e-12


def _finite_pair(name: str, value: Any) -> np.ndarray:
    try:
        pair = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain two finite numeric values") from exc
    if pair.shape != (2,) or not np.all(np.isfinite(pair)):
        raise ValueError(f"{name} must contain two finite numeric values")
    return pair


def _finite_positive(name: str, value: float) -> float:
    try:
        scalar = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive finite scalar") from exc
    if not math.isfinite(scalar) or scalar <= 0.0:
        raise ValueError(f"{name} must be a positive finite scalar")
    return scalar


@dataclass(frozen=True)
class LegacyWindDemandProxyConfig:
    """Parameters of the historical empirical wind-demand proxy.

    ``demand_axis_scale_deg`` is the retained numerical scale of the legacy
    two-axis demand representation.  It is not a posture deadband, a platform
    parameter, or a physically calibrated ballast mapping.
    """

    demand_axis_scale_deg: tuple[float, float]
    reference_speed_ms: float
    effect_cap: float
    sign_multiplier: float

    def __post_init__(self) -> None:
        demand_scale = _finite_pair(
            "demand_axis_scale_deg",
            self.demand_axis_scale_deg,
        )
        if np.any(demand_scale <= 0.0):
            raise ValueError("demand_axis_scale_deg values must be positive")
        object.__setattr__(
            self,
            "demand_axis_scale_deg",
            tuple(float(value) for value in demand_scale),
        )
        object.__setattr__(
            self,
            "reference_speed_ms",
            _finite_positive("reference_speed_ms", self.reference_speed_ms),
        )
        cap = float(self.effect_cap)
        if not math.isfinite(cap) or cap < 0.0:
            raise ValueError("effect_cap must be finite and non-negative")
        object.__setattr__(self, "effect_cap", cap)
        sign = float(self.sign_multiplier)
        if sign not in {-1.0, 1.0}:
            raise ValueError("sign_multiplier must be -1 or 1")
        object.__setattr__(self, "sign_multiplier", sign)


@dataclass(frozen=True)
class StageDemand:
    """Demand components available to one candidate-planning stage."""

    feedback_demand_deg: tuple[float, float]
    forecast_increment_deg: tuple[float, float]
    combined_demand_deg: tuple[float, float]
    forecast_available: bool
    forecast_reason: str
    direction_consistency: float | None
    direction_consistent: bool | None


def posture_feedback_demand_deg(
    posture_deg: Any,
    *,
    deadband_deg: Any,
) -> np.ndarray:
    """Return the signed feedback demand outside the measured posture deadband."""

    posture = _finite_pair("posture_deg", posture_deg)
    deadband = _finite_pair("deadband_deg", deadband_deg)
    if np.any(deadband <= 0.0):
        raise ValueError("deadband_deg values must be positive")
    return np.sign(posture) * np.maximum(np.abs(posture) - deadband, 0.0)


def legacy_wind_proxy_demand_deg(
    uv_ms: Any,
    *,
    config: LegacyWindDemandProxyConfig,
) -> np.ndarray:
    """Map ENU downwind wind vectors to the retained empirical demand proxy."""

    vectors = np.asarray(uv_ms, dtype=float)
    if vectors.ndim != 2 or vectors.shape[1] != 2 or vectors.shape[0] == 0:
        raise ValueError("wind vectors must have shape (N, 2)")
    if not np.all(np.isfinite(vectors)):
        raise ValueError("wind vectors must contain only finite values")
    speeds = np.linalg.norm(vectors, axis=1)
    mean_speed = float(np.mean(speeds))
    mean_vector = np.mean(vectors, axis=0)
    direction_speed = float(np.linalg.norm(mean_vector))
    if mean_speed <= _EPS or direction_speed <= _EPS:
        return np.zeros(2, dtype=float)
    magnitude = min(
        (mean_speed / config.reference_speed_ms) ** 2,
        config.effect_cap,
    )
    u, v = (float(mean_vector[0]), float(mean_vector[1]))
    wind_direction = math.atan2(-u, -v)
    demand_scale = np.asarray(config.demand_axis_scale_deg, dtype=float)
    raw = np.array(
        [
            -demand_scale[0] * magnitude * math.cos(wind_direction),
            demand_scale[1] * magnitude * math.sin(wind_direction),
        ],
        dtype=float,
    )
    return config.sign_multiplier * raw


def form_stage_demand(
    *,
    feedback_demand_deg: Any,
    current_wind_uv_ms: Any,
    future_wind_uv_ms: Any | None,
    forecast_available: bool,
    stage_discount: float,
    proxy_config: LegacyWindDemandProxyConfig,
    forecast_reason: str,
    direction_consistency: float | None,
    direction_consistent: bool | None,
) -> StageDemand:
    """Combine feedback with one stage's empirical forecast increment.

    Direction coherence is carried from ``ForecastActionPolicy`` for audit. It
    does not yet gate ordinary forecast demand, because that policy choice must
    be justified together with the future physical demand mapping.
    """

    feedback = _finite_pair("feedback_demand_deg", feedback_demand_deg)
    discount = float(stage_discount)
    if not math.isfinite(discount) or discount < 0.0:
        raise ValueError("stage_discount must be non-negative and finite")
    if direction_consistency is not None:
        consistency = float(direction_consistency)
        if not math.isfinite(consistency) or not 0.0 <= consistency <= 1.0:
            raise ValueError("direction_consistency must lie in [0, 1]")
    else:
        consistency = None
    if direction_consistent is not None and type(direction_consistent) is not bool:
        raise ValueError("direction_consistent must be a boolean when provided")

    if forecast_available:
        if future_wind_uv_ms is None:
            raise ValueError("future_wind_uv_ms is required when forecast is available")
        current_wind = _finite_pair("current_wind_uv_ms", current_wind_uv_ms)
        current_proxy = legacy_wind_proxy_demand_deg(
            current_wind.reshape(1, 2),
            config=proxy_config,
        )
        future_proxy = legacy_wind_proxy_demand_deg(
            future_wind_uv_ms,
            config=proxy_config,
        )
        forecast_increment = (future_proxy - current_proxy) * discount
    else:
        forecast_increment = np.zeros(2, dtype=float)
    combined = feedback + forecast_increment
    return StageDemand(
        feedback_demand_deg=tuple(float(value) for value in feedback),
        forecast_increment_deg=tuple(float(value) for value in forecast_increment),
        combined_demand_deg=tuple(float(value) for value in combined),
        forecast_available=bool(forecast_available),
        forecast_reason=str(forecast_reason),
        direction_consistency=consistency,
        direction_consistent=direction_consistent,
    )


__all__ = [
    "LegacyWindDemandProxyConfig",
    "StageDemand",
    "form_stage_demand",
    "legacy_wind_proxy_demand_deg",
    "posture_feedback_demand_deg",
]
