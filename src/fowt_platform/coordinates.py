"""Coordinate conversions for the fixed-equilibrium platform model."""

from __future__ import annotations

from typing import Any

import numpy as np


def _finite_scalar(name: str, value: Any) -> float:
    scalar = float(value)
    if not np.isfinite(scalar):
        raise ValueError(f"{name} must be finite")
    return scalar


def _horizontal_vector(name: str, value: Any) -> np.ndarray:
    vector = np.asarray(value, dtype=float)
    if vector.shape != (2,):
        raise ValueError(f"{name} must have shape (2,), got {vector.shape}")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain only finite values")
    return np.array(vector, dtype=float, copy=True)


def meteorological_wind_to_enu(
    speed: Any,
    direction_from_rad: Any,
) -> np.ndarray:
    """Return the downwind vector in ENU axes from a meteorological direction.

    The direction is the wind source direction, measured clockwise from true
    north. The returned components are eastward and northward velocity.
    """

    wind_speed = _finite_scalar("speed", speed)
    if wind_speed < 0.0:
        raise ValueError("speed must be non-negative")
    direction = _finite_scalar("direction_from_rad", direction_from_rad)
    vector = np.array(
        [-wind_speed * np.sin(direction), -wind_speed * np.cos(direction)],
        dtype=float,
    )
    vector.setflags(write=False)
    return vector


def enu_wind_to_platform(enu_wind: Any, heading_rad: Any) -> np.ndarray:
    """Resolve an ENU wind vector into platform forward and port axes.

    Heading is the platform-forward direction measured clockwise from true
    north. The platform horizontal axes are forward and port.
    """

    vector = _horizontal_vector("enu_wind", enu_wind)
    heading = _finite_scalar("heading_rad", heading_rad)
    rotation = np.array(
        [
            [np.sin(heading), np.cos(heading)],
            [-np.cos(heading), np.sin(heading)],
        ],
        dtype=float,
    )
    result = rotation @ vector
    result.setflags(write=False)
    return result


def meteorological_wind_to_platform(
    speed: Any,
    direction_from_rad: Any,
    heading_rad: Any,
) -> np.ndarray:
    """Convert meteorological wind directly to platform horizontal axes."""

    return enu_wind_to_platform(
        meteorological_wind_to_enu(speed, direction_from_rad),
        heading_rad,
    )


def true_heading_from_yaw(nominal_heading_rad: Any, yaw_rad: Any) -> float:
    """Return continuous, unwrapped true heading from right-hand positive yaw."""

    nominal_heading = _finite_scalar("nominal_heading_rad", nominal_heading_rad)
    yaw = _finite_scalar("yaw_rad", yaw_rad)
    return nominal_heading - yaw
