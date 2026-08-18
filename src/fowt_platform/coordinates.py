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


def _three_vector(name: str, value: Any) -> np.ndarray:
    vector = np.asarray(value, dtype=float)
    if vector.shape != (3,):
        raise ValueError(f"{name} must have shape (3,), got {vector.shape}")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain only finite values")
    return np.array(vector, dtype=float, copy=True)


def _unit_three_vector(name: str, value: Any) -> np.ndarray:
    vector = _three_vector(name, value)
    if not np.isclose(np.linalg.norm(vector), 1.0, rtol=0.0, atol=1e-12):
        raise ValueError(f"{name} must be a unit vector")
    return vector


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


def relative_air_velocity_at_platform_point(
    ambient_air_velocity_platform_mps: Any,
    platform_reference_velocity_platform_mps: Any,
    platform_angular_velocity_platform_radps: Any,
    point_from_platform_reference_m: Any,
) -> np.ndarray:
    """Return air velocity relative to a moving point on the platform.

    All inputs use the same frozen equilibrium platform axes. The point
    velocity is the reference-point velocity plus ``omega cross r``. In the
    current small-angle model, ``omega`` is taken from incremental-state
    ``velocity[3:6]``; it is not a general large-attitude body-rate mapping.
    This function only establishes a kinematic relative-air vector. It neither
    chooses a rotor-normal direction nor includes induction, yaw, shear, or
    aerodynamic coefficients.
    """

    ambient = _three_vector(
        "ambient_air_velocity_platform_mps", ambient_air_velocity_platform_mps
    )
    reference_velocity = _three_vector(
        "platform_reference_velocity_platform_mps",
        platform_reference_velocity_platform_mps,
    )
    angular_velocity = _three_vector(
        "platform_angular_velocity_platform_radps",
        platform_angular_velocity_platform_radps,
    )
    point = _three_vector("point_from_platform_reference_m", point_from_platform_reference_m)
    point_velocity = reference_velocity + np.cross(angular_velocity, point)
    result = np.array(ambient - point_velocity, dtype=float, copy=True)
    result.setflags(write=False)
    return result


def downwind_normal_relative_wind_component(
    relative_air_velocity_platform_mps: Any,
    downwind_rotor_normal_platform: Any,
) -> float:
    """Project relative air velocity onto a supplied downwind rotor normal.

    The returned component is signed. A negative value is intentionally not
    clipped or converted to a magnitude: the caller must decide whether the
    rotor orientation and operating state permit a positive-thrust load model.
    Both vectors use the same frozen equilibrium platform axes.
    """

    relative_air_velocity = _three_vector(
        "relative_air_velocity_platform_mps", relative_air_velocity_platform_mps
    )
    normal = _unit_three_vector(
        "downwind_rotor_normal_platform", downwind_rotor_normal_platform
    )
    return float(relative_air_velocity @ normal)


def true_heading_from_yaw(nominal_heading_rad: Any, yaw_rad: Any) -> float:
    """Return continuous, unwrapped true heading from right-hand positive yaw."""

    nominal_heading = _finite_scalar("nominal_heading_rad", nominal_heading_rad)
    yaw = _finite_scalar("yaw_rad", yaw_rad)
    return nominal_heading - yaw
