"""Explicit rotor-load input assembly for the frozen-equilibrium model.

This module joins the already separate relations for point kinematics,
quasi-steady rotor-normal loading and six-degree-of-freedom force mapping.
It deliberately leaves operating-state selection outside the model: callers
must supply the rotor-normal direction and a preselected thrust coefficient.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .coordinates import enu_wind_to_platform, relative_air_velocity_at_platform_point
from .incremental import generalized_load_from_point_force
from .rotor_loads import (
    RotorNormalLoad,
    quasi_steady_rotor_normal_load_from_relative_air,
)


def _three_vector(name: str, value: Any) -> np.ndarray:
    vector = np.asarray(value, dtype=float)
    if vector.shape != (3,):
        raise ValueError(f"{name} must have shape (3,), got {vector.shape}")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain only finite values")
    result = np.array(vector, dtype=float, copy=True)
    result.setflags(write=False)
    return result


def _six_vector(name: str, value: Any) -> np.ndarray:
    vector = np.asarray(value, dtype=float)
    if vector.shape != (6,):
        raise ValueError(f"{name} must have shape (6,), got {vector.shape}")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain only finite values")
    result = np.array(vector, dtype=float, copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class RotorGeneralizedLoad:
    """Rotor normal load and its generalized input about one reference point.

    All vectors use the frozen equilibrium platform axes.  The generalized
    load follows ``[X, Y, Z, K, M, N]`` and contains the supplied point force
    and its ``r cross F`` moment about the platform reference point.  It does
    not include an independent hub aerodynamic moment.
    """

    relative_air_velocity_platform_mps: Any
    normal_load: RotorNormalLoad
    generalized_load_platform: Any

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "relative_air_velocity_platform_mps",
            _three_vector(
                "relative_air_velocity_platform_mps",
                self.relative_air_velocity_platform_mps,
            ),
        )
        if not isinstance(self.normal_load, RotorNormalLoad):
            raise TypeError("normal_load must be a RotorNormalLoad")
        object.__setattr__(
            self,
            "generalized_load_platform",
            _six_vector("generalized_load_platform", self.generalized_load_platform),
        )


def quasi_steady_rotor_generalized_load_from_platform_motion(
    *,
    air_density_kg_m3: Any,
    rotor_radius_m: Any,
    thrust_coefficient: Any,
    ambient_air_velocity_platform_mps: Any,
    platform_reference_velocity_platform_mps: Any,
    platform_angular_velocity_platform_radps: Any,
    rotor_force_application_point_from_reference_m: Any,
    downwind_rotor_normal_platform: Any,
) -> RotorGeneralizedLoad:
    """Build an explicit quasi-steady rotor input for the low-order model.

    The relative air velocity is evaluated at the supplied rotor-force
    application point using the frozen-equilibrium platform kinematics.  Its
    positive component along the supplied downwind normal then produces a
    quasi-steady normal force, which is mapped to a generalized force about
    the platform reference point.

    This function does not select ``C_T``, infer a rotor operating state,
    model yaw alignment or induction, or add a local rotor moment.  Those
    inputs must be assembled explicitly by a later operating-condition layer.
    """

    application_point = _three_vector(
        "rotor_force_application_point_from_reference_m",
        rotor_force_application_point_from_reference_m,
    )
    relative_air = relative_air_velocity_at_platform_point(
        ambient_air_velocity_platform_mps,
        platform_reference_velocity_platform_mps,
        platform_angular_velocity_platform_radps,
        application_point,
    )
    normal_load = quasi_steady_rotor_normal_load_from_relative_air(
        air_density_kg_m3=air_density_kg_m3,
        rotor_radius_m=rotor_radius_m,
        thrust_coefficient=thrust_coefficient,
        relative_air_velocity_platform_mps=relative_air,
        downwind_rotor_normal_platform=downwind_rotor_normal_platform,
    )
    generalized_load = generalized_load_from_point_force(
        normal_load.force_platform_n,
        application_point,
    )
    return RotorGeneralizedLoad(
        relative_air_velocity_platform_mps=relative_air,
        normal_load=normal_load,
        generalized_load_platform=generalized_load,
    )


def quasi_steady_rotor_generalized_load_from_enu_wind(
    *,
    air_density_kg_m3: Any,
    rotor_radius_m: Any,
    thrust_coefficient: Any,
    enu_downwind_air_velocity_mps: Any,
    frozen_equilibrium_heading_rad: Any,
    platform_reference_velocity_platform_mps: Any,
    platform_angular_velocity_platform_radps: Any,
    rotor_force_application_point_from_reference_m: Any,
    downwind_rotor_normal_platform: Any,
) -> RotorGeneralizedLoad:
    """Build a rotor load from an explicit ENU horizontal wind vector.

    ``enu_downwind_air_velocity_mps`` has the fixed forecast-data convention
    ``[east, north]`` in m/s and points in the direction the air moves.  It is
    resolved once into the frozen-equilibrium platform axes using
    ``frozen_equilibrium_heading_rad`` and then lifted to ``[forward, port,
    0]``.  The zero vertical component is a low-order ambient-air assumption,
    not a height extrapolation or a shear model.

    The remaining arguments stay explicit: this helper does not infer the
    rotor normal, thrust coefficient, operating state, yaw alignment, or a
    time-varying platform heading.  It is therefore a coordinate composition
    around :func:`quasi_steady_rotor_generalized_load_from_platform_motion`,
    not a controller or an aerodynamic operating-condition model.
    """

    platform_horizontal = enu_wind_to_platform(
        enu_downwind_air_velocity_mps,
        heading_rad=frozen_equilibrium_heading_rad,
    )
    ambient_air_velocity_platform_mps = np.array(
        [platform_horizontal[0], platform_horizontal[1], 0.0],
        dtype=float,
    )
    return quasi_steady_rotor_generalized_load_from_platform_motion(
        air_density_kg_m3=air_density_kg_m3,
        rotor_radius_m=rotor_radius_m,
        thrust_coefficient=thrust_coefficient,
        ambient_air_velocity_platform_mps=ambient_air_velocity_platform_mps,
        platform_reference_velocity_platform_mps=platform_reference_velocity_platform_mps,
        platform_angular_velocity_platform_radps=platform_angular_velocity_platform_radps,
        rotor_force_application_point_from_reference_m=(
            rotor_force_application_point_from_reference_m
        ),
        downwind_rotor_normal_platform=downwind_rotor_normal_platform,
    )
