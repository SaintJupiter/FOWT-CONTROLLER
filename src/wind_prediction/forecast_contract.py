from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .forecast_adapter import ForecastResult


@dataclass(frozen=True)
class ForecastContract:
    future_steps: int
    event_columns: tuple[str, ...] = ()
    require_event_probs: bool = False


def speed_direction_from_uv(uv: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(uv, dtype=np.float32)
    speed = np.sqrt(arr[:, 0] ** 2 + arr[:, 1] ** 2)
    direction = (np.rad2deg(np.arctan2(-arr[:, 0], -arr[:, 1])) + 360.0) % 360.0
    return speed.astype(np.float32), direction.astype(np.float32)


def forecast_from_oracle_sample(
    *,
    y_uv_raw: np.ndarray,
    y_event: np.ndarray,
    event_columns: Iterable[str],
    model_version: str,
    timestamp: str | None = None,
) -> ForecastResult:
    uv = np.asarray(y_uv_raw, dtype=np.float32)
    event_values = np.asarray(y_event, dtype=np.float32).reshape(-1)
    columns = list(event_columns)
    event_probs = {
        str(name): float(event_values[idx]) if idx < event_values.size else 0.0
        for idx, name in enumerate(columns)
    }
    speed, direction = speed_direction_from_uv(uv)
    result = ForecastResult(
        wind_uv_raw=np.array(uv, dtype=np.float32, copy=True),
        wind_speed=speed,
        wind_dir_deg=direction,
        event_probs=event_probs,
        risk_levels={name: bool(value >= 0.5) for name, value in event_probs.items()},
        model_version=str(model_version),
        timestamp=timestamp,
    )
    validate_forecast_result(result, ForecastContract(len(uv), tuple(columns), True))
    return result


def validate_forecast_result(
    forecast: ForecastResult,
    contract: ForecastContract,
    *,
    atol: float = 1e-3,
) -> ForecastResult:
    uv = np.asarray(forecast.wind_uv_raw, dtype=float)
    speed = np.asarray(forecast.wind_speed, dtype=float)
    direction = np.asarray(forecast.wind_dir_deg, dtype=float)
    expected_uv_shape = (int(contract.future_steps), 2)
    if uv.shape != expected_uv_shape:
        raise ValueError(f"forecast {forecast.model_version} uv shape {uv.shape}, expected {expected_uv_shape}")
    if speed.shape != (int(contract.future_steps),):
        raise ValueError(
            f"forecast {forecast.model_version} speed shape {speed.shape}, expected {(int(contract.future_steps),)}"
        )
    if direction.shape != (int(contract.future_steps),):
        raise ValueError(
            f"forecast {forecast.model_version} direction shape {direction.shape}, expected {(int(contract.future_steps),)}"
        )
    if not np.all(np.isfinite(uv)):
        raise ValueError(f"forecast {forecast.model_version} has non-finite uv values")
    if not np.all(np.isfinite(speed)):
        raise ValueError(f"forecast {forecast.model_version} has non-finite speed values")
    if not np.all(np.isfinite(direction)):
        raise ValueError(f"forecast {forecast.model_version} has non-finite direction values")
    if np.any(speed < -atol):
        raise ValueError(f"forecast {forecast.model_version} has negative wind speed")
    expected_speed, expected_direction = speed_direction_from_uv(uv.astype(np.float32))
    if not np.allclose(speed, expected_speed, atol=atol, rtol=0.0):
        raise ValueError(f"forecast {forecast.model_version} speed is inconsistent with uv")
    # Circular direction equality: compare shortest wrapped angular difference.
    diff = ((direction - expected_direction.astype(float) + 180.0) % 360.0) - 180.0
    if not np.all(np.abs(diff) <= max(atol, 1e-2)):
        raise ValueError(f"forecast {forecast.model_version} direction is inconsistent with uv")

    if contract.require_event_probs:
        missing = [name for name in contract.event_columns if name not in forecast.event_probs]
        if missing:
            raise ValueError(
                f"forecast {forecast.model_version} missing event probabilities: {', '.join(missing)}"
            )
    for name, value in forecast.event_probs.items():
        prob = float(value)
        if not np.isfinite(prob) or prob < -atol or prob > 1.0 + atol:
            raise ValueError(
                f"forecast {forecast.model_version} event probability {name}={prob} outside [0, 1]"
            )
    for name in forecast.risk_levels:
        if name not in forecast.event_probs:
            raise ValueError(
                f"forecast {forecast.model_version} risk flag {name} has no event probability"
            )
    return forecast
