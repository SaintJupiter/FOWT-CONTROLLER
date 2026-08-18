"""Scoped quasi-steady rotor-normal load relations.

This module deliberately accepts the *relative normal wind speed*, thrust
coefficient, and downwind rotor-normal direction from its caller.  It does
not infer yaw alignment, platform-relative wind, rotor control state, or a
coefficient law from a wind time series.  Those choices belong to a later
assembly layer with an explicit operating-range statement.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class RotorNormalLoad:
    """Quasi-steady rotor thrust and its platform-coordinate force vector."""

    thrust_n: float
    force_platform_n: np.ndarray


def quasi_steady_rotor_normal_load(
    *,
    air_density_kg_m3: Any,
    rotor_radius_m: Any,
    thrust_coefficient: Any,
    relative_normal_wind_speed_mps: Any,
    downwind_rotor_normal_platform: Any,
) -> RotorNormalLoad:
    """Return a rotor-normal load from an explicit ``C_T`` input.

    The scalar magnitude follows
    ``T = 0.5 * rho * pi * R**2 * C_T * U_n**2``.  ``U_n`` is the positive
    magnitude of the wind velocity relative to the rotor plane in its normal
    direction.  The supplied direction must already be a unit vector in the
    fixed platform axes and point downwind. ``C_T`` must be non-negative: this
    relation is limited to the positive-thrust branch and explicitly rejects
    negative entries that may exist in a wider rotor-performance table. The
    direction is intentionally not
    normalised here, so a caller cannot silently change a force magnitude by
    passing a non-unit direction.

    This is a load relation only.  It does not choose ``C_T`` from a rotor
    table and does not convert the force to a generalized load about a
    platform reference point.
    """

    density = _positive_scalar("air_density_kg_m3", air_density_kg_m3)
    radius = _positive_scalar("rotor_radius_m", rotor_radius_m)
    coefficient = _nonnegative_scalar("thrust_coefficient", thrust_coefficient)
    normal_speed = _nonnegative_scalar(
        "relative_normal_wind_speed_mps",
        relative_normal_wind_speed_mps,
    )
    direction = _unit_vector(
        "downwind_rotor_normal_platform",
        downwind_rotor_normal_platform,
    )

    thrust = 0.5 * density * np.pi * radius**2 * coefficient * normal_speed**2
    force = np.array(thrust * direction, dtype=float, copy=True)
    force.setflags(write=False)
    return RotorNormalLoad(thrust_n=float(thrust), force_platform_n=force)


def _positive_scalar(name: str, value: Any) -> float:
    scalar = _finite_scalar(name, value)
    if scalar <= 0.0:
        raise ValueError(f"{name} must be positive")
    return scalar


def _nonnegative_scalar(name: str, value: Any) -> float:
    scalar = _finite_scalar(name, value)
    if scalar < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return scalar


def _finite_scalar(name: str, value: Any) -> float:
    try:
        scalar = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not np.isfinite(scalar):
        raise ValueError(f"{name} must be finite")
    return scalar


def _unit_vector(name: str, value: Any) -> np.ndarray:
    vector = np.asarray(value, dtype=float)
    if vector.shape != (3,):
        raise ValueError(f"{name} must have shape (3,), got {vector.shape}")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain only finite values")
    norm = float(np.linalg.norm(vector))
    if not np.isclose(norm, 1.0, rtol=0.0, atol=1e-12):
        raise ValueError(f"{name} must be a unit vector")
    result = np.array(vector, dtype=float, copy=True)
    result.setflags(write=False)
    return result
