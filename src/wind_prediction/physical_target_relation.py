"""Describe, without selecting, the relation of one target to a physical endpoint.

The historical controller and the new physical diagnostic can both produce a
three-tank target, but their numerical target deltas must not be compared as
if every component had the same physical meaning.  In particular, a target
that changes all three tank masses together carries a common-mode component,
whereas the current P5 endpoint is a zero-net-mass redistribution.

This module separates those components and expresses the remaining
redistribution in the common ``(pitch, roll) = (M, K)`` ballast-moment
convention.  It does not authorize, rank, name or execute an action.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from fowt_platform.ballast_moment_allocation import (
    pitch_roll_moment_from_tank_mass_deltas,
)


_MASS_TOLERANCE_KG = 1.0e-8
_MOMENT_ZERO_TOLERANCE_NM = 1.0e-6


def _three(name: str, value: Any) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain three finite values") from exc
    if array.shape != (3,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain three finite values")
    result = np.array(array, dtype=float, copy=True)
    result.setflags(write=False)
    return result


def _within_capacity(name: str, masses: np.ndarray, capacities: np.ndarray) -> None:
    if np.any(masses < 0.0) or np.any(masses > capacities):
        raise ValueError(f"{name} must remain within tank capacities")


def _redistribution_component(mass_delta_kg: np.ndarray) -> np.ndarray:
    """Remove equal per-tank intake or discharge from a three-tank delta."""

    result = np.array(
        mass_delta_kg - float(np.mean(mass_delta_kg)),
        dtype=float,
        copy=True,
    )
    result.setflags(write=False)
    return result


def _require_common_mode_moment_free(
    *,
    tank_coordinates_m: Any,
    gravity_m_s2: Any,
) -> None:
    """Require the reference geometry to isolate common-mode water exchange.

    Removing an equal per-tank intake/discharge is valid for a pitch-roll
    direction comparison only when that equal change itself creates no
    pitch-roll moment about the frozen platform reference point.
    """

    common_mode_moment = pitch_roll_moment_from_tank_mass_deltas(
        tank_mass_deltas_kg=np.ones(3),
        tank_coordinates_m=tank_coordinates_m,
        gravity_m_s2=gravity_m_s2,
    )
    if float(np.linalg.norm(common_mode_moment)) > _MOMENT_ZERO_TOLERANCE_NM:
        raise ValueError(
            "tank geometry gives common-mode water exchange a pitch-roll moment; "
            "a redistribution-only target comparison is not defined"
        )


@dataclass(frozen=True)
class TankTargetEndpointRelation:
    """Factual relation between one proposed target and a physical endpoint.

    Moment quantities describe only the differential redistribution component.
    Net mass changes remain explicit because current pumps exchange water with
    the sea independently.  ``moment_direction_cosine`` is intentionally
    descriptive rather than a policy threshold: later control logic must
    decide what degree of alignment is sufficient for a lifecycle decision.
    """

    actual_tank_masses_kg: Any
    proposed_target_tank_masses_kg: Any
    physical_endpoint_tank_masses_kg: Any
    proposed_target_net_mass_delta_kg: float
    physical_endpoint_net_mass_delta_kg: float
    proposed_target_redistribution_delta_kg: Any
    physical_endpoint_redistribution_delta_kg: Any
    proposed_target_pitch_roll_moment_nm: Any
    physical_endpoint_pitch_roll_moment_nm: Any
    moment_direction_cosine: float | None
    same_moment_half_plane: bool | None
    opposite_moment_half_plane: bool | None

    def __post_init__(self) -> None:
        for name in (
            "actual_tank_masses_kg",
            "proposed_target_tank_masses_kg",
            "physical_endpoint_tank_masses_kg",
            "proposed_target_redistribution_delta_kg",
            "physical_endpoint_redistribution_delta_kg",
        ):
            object.__setattr__(self, name, _three(name, getattr(self, name)))
        for name in (
            "proposed_target_pitch_roll_moment_nm",
            "physical_endpoint_pitch_roll_moment_nm",
        ):
            vector = np.asarray(getattr(self, name), dtype=float)
            if vector.shape != (2,) or not np.all(np.isfinite(vector)):
                raise ValueError(f"{name} must contain two finite values")
            vector = np.array(vector, dtype=float, copy=True)
            vector.setflags(write=False)
            object.__setattr__(self, name, vector)
        for name in (
            "proposed_target_net_mass_delta_kg",
            "physical_endpoint_net_mass_delta_kg",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value):
                raise ValueError(f"{name} must be finite")
            object.__setattr__(self, name, value)
        cosine = self.moment_direction_cosine
        if cosine is not None:
            cosine = float(cosine)
            if not np.isfinite(cosine) or not -1.0 <= cosine <= 1.0:
                raise ValueError("moment_direction_cosine must lie in [-1, 1]")
            object.__setattr__(self, "moment_direction_cosine", cosine)
        for name in ("same_moment_half_plane", "opposite_moment_half_plane"):
            value = getattr(self, name)
            if value is not None and type(value) is not bool:
                raise ValueError(f"{name} must be a boolean or None")
        if cosine is None:
            if self.same_moment_half_plane is not None or self.opposite_moment_half_plane is not None:
                raise ValueError(
                    "moment half-plane flags must be None when direction is indeterminate"
                )
        elif bool(self.same_moment_half_plane) and bool(self.opposite_moment_half_plane):
            raise ValueError(
                "moment half-plane flags cannot both be true"
            )


def compare_tank_target_to_physical_endpoint(
    *,
    actual_tank_masses_kg: Any,
    proposed_target_tank_masses_kg: Any,
    physical_endpoint_tank_masses_kg: Any,
    tank_capacities_kg: Any,
    tank_coordinates_m: Any,
    gravity_m_s2: Any,
) -> TankTargetEndpointRelation:
    """Compare a target with a zero-net physical endpoint in moment space.

    The proposed target may contain a nonzero total mass change.  It is
    retained as an explicit fact but removed before pitch-roll direction is
    calculated, so a future controller cannot mistake common intake/discharge
    for a redistribution-driven restoring action.  The physical endpoint must
    preserve total ballast mass relative to the supplied actual state.
    """

    actual = _three("actual_tank_masses_kg", actual_tank_masses_kg)
    target = _three("proposed_target_tank_masses_kg", proposed_target_tank_masses_kg)
    endpoint = _three(
        "physical_endpoint_tank_masses_kg",
        physical_endpoint_tank_masses_kg,
    )
    capacities = _three("tank_capacities_kg", tank_capacities_kg)
    if np.any(capacities <= 0.0):
        raise ValueError("tank_capacities_kg values must be positive")
    _within_capacity("actual_tank_masses_kg", actual, capacities)
    _within_capacity("proposed_target_tank_masses_kg", target, capacities)
    _within_capacity("physical_endpoint_tank_masses_kg", endpoint, capacities)
    _require_common_mode_moment_free(
        tank_coordinates_m=tank_coordinates_m,
        gravity_m_s2=gravity_m_s2,
    )

    target_delta = target - actual
    endpoint_delta = endpoint - actual
    endpoint_net = float(np.sum(endpoint_delta))
    if not np.isclose(endpoint_net, 0.0, rtol=0.0, atol=_MASS_TOLERANCE_KG):
        raise ValueError(
            "physical_endpoint_tank_masses_kg must preserve total ballast mass "
            "relative to actual_tank_masses_kg"
        )

    target_redistribution = _redistribution_component(target_delta)
    endpoint_redistribution = _redistribution_component(endpoint_delta)
    target_moment = pitch_roll_moment_from_tank_mass_deltas(
        tank_mass_deltas_kg=target_redistribution,
        tank_coordinates_m=tank_coordinates_m,
        gravity_m_s2=gravity_m_s2,
    )
    endpoint_moment = pitch_roll_moment_from_tank_mass_deltas(
        tank_mass_deltas_kg=endpoint_redistribution,
        tank_coordinates_m=tank_coordinates_m,
        gravity_m_s2=gravity_m_s2,
    )
    target_norm = float(np.linalg.norm(target_moment))
    endpoint_norm = float(np.linalg.norm(endpoint_moment))
    if target_norm <= _MOMENT_ZERO_TOLERANCE_NM or endpoint_norm <= _MOMENT_ZERO_TOLERANCE_NM:
        cosine: float | None = None
        same_half_plane: bool | None = None
        opposite_half_plane: bool | None = None
    else:
        cosine = float(
            np.clip(
                np.dot(target_moment, endpoint_moment) / (target_norm * endpoint_norm),
                -1.0,
                1.0,
            )
        )
        same_half_plane = bool(cosine > 0.0)
        opposite_half_plane = bool(cosine < 0.0)

    return TankTargetEndpointRelation(
        actual_tank_masses_kg=actual,
        proposed_target_tank_masses_kg=target,
        physical_endpoint_tank_masses_kg=endpoint,
        proposed_target_net_mass_delta_kg=float(np.sum(target_delta)),
        physical_endpoint_net_mass_delta_kg=endpoint_net,
        proposed_target_redistribution_delta_kg=target_redistribution,
        physical_endpoint_redistribution_delta_kg=endpoint_redistribution,
        proposed_target_pitch_roll_moment_nm=target_moment,
        physical_endpoint_pitch_roll_moment_nm=endpoint_moment,
        moment_direction_cosine=cosine,
        same_moment_half_plane=same_half_plane,
        opposite_moment_half_plane=opposite_half_plane,
    )


__all__ = [
    "TankTargetEndpointRelation",
    "compare_tank_target_to_physical_endpoint",
]
