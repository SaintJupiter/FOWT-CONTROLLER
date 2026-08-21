"""Sample bounded intermediate states between actual tanks and a P5 endpoint.

This helper stays on the diagnostic side of the control boundary.  Given the
current actual three-tank masses and the zero-net-mass endpoint returned by
the physical redistribution diagnostic, it samples convex intermediate
redistributions.  It does not name control actions, select a sample, create
an execution request or model the pumps.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


_MASS_TOLERANCE_KG = 1e-8


def _finite_three(name: str, value: Any) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{name} must contain exactly three finite numeric values"
        ) from exc
    if array.shape != (3,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain exactly three finite numeric values")
    result = np.array(array, dtype=float, copy=True)
    result.setflags(write=False)
    return result


def _fractions(value: Any) -> tuple[float, ...]:
    try:
        fractions = tuple(float(item) for item in value)
    except (TypeError, ValueError) as exc:
        raise ValueError("fractions must be a non-empty sequence of finite values") from exc
    if not fractions or not all(np.isfinite(item) and 0.0 <= item <= 1.0 for item in fractions):
        raise ValueError("fractions must be a non-empty sequence with values in [0, 1]")
    return fractions


@dataclass(frozen=True)
class BallastEndpointPathSample:
    """One hypothetical redistribution on the path to a diagnostic endpoint.

    ``mass_delta_from_actual_kg`` is relative to the actual masses supplied to
    :func:`sample_ballast_endpoint_path`, not to a fixed platform reference
    state.  The path only accepts a zero-net-mass endpoint, so each sample
    preserves total ballast mass relative to that current state.
    """

    fraction: float
    hypothetical_tank_masses_kg: Any
    mass_delta_from_actual_kg: Any
    net_mass_delta_from_actual_kg: float
    within_capacity: bool

    def __post_init__(self) -> None:
        fraction = float(self.fraction)
        if not np.isfinite(fraction) or not 0.0 <= fraction <= 1.0:
            raise ValueError("fraction must lie in [0, 1]")
        object.__setattr__(self, "fraction", fraction)
        object.__setattr__(
            self,
            "hypothetical_tank_masses_kg",
            _finite_three(
                "hypothetical_tank_masses_kg", self.hypothetical_tank_masses_kg
            ),
        )
        deltas = _finite_three(
            "mass_delta_from_actual_kg", self.mass_delta_from_actual_kg
        )
        object.__setattr__(self, "mass_delta_from_actual_kg", deltas)
        net = float(self.net_mass_delta_from_actual_kg)
        if not np.isfinite(net) or not np.isclose(
            net,
            float(np.sum(deltas)),
            rtol=0.0,
            atol=_MASS_TOLERANCE_KG,
        ):
            raise ValueError(
                "net_mass_delta_from_actual_kg must equal the sum of mass_delta_from_actual_kg"
            )
        object.__setattr__(self, "net_mass_delta_from_actual_kg", net)
        if type(self.within_capacity) is not bool:
            raise ValueError("within_capacity must be a boolean")


def sample_ballast_endpoint_path(
    *,
    actual_tank_masses_kg: Any,
    diagnostic_target_tank_masses_kg: Any,
    tank_capacities_kg: Any,
    fractions: Any,
) -> tuple[BallastEndpointPathSample, ...]:
    """Return bounded samples from current tanks to a physical diagnostic endpoint.

    The supplied endpoint must already be a zero-net-mass redistribution from
    the supplied actual tank state, normally the allocation embedded in
    :class:`ForecastBallastDemandDiagnostic`.  Inputs outside capacity are
    rejected rather than clipped, because feasibility belongs to the physical
    allocation that produced the endpoint.  Intermediate points are convex
    combinations of two feasible endpoints and therefore remain feasible.

    This function has no controller, candidate-ranking or execution semantics.
    It is only a way to expose physically bounded endpoint fractions for later,
    separately defined evaluation.
    """

    actual = _finite_three("actual_tank_masses_kg", actual_tank_masses_kg)
    endpoint = _finite_three(
        "diagnostic_target_tank_masses_kg", diagnostic_target_tank_masses_kg
    )
    capacities = _finite_three("tank_capacities_kg", tank_capacities_kg)
    path_fractions = _fractions(fractions)

    if np.any(capacities <= 0.0):
        raise ValueError("tank_capacities_kg values must be positive")
    for name, masses in (
        ("actual_tank_masses_kg", actual),
        ("diagnostic_target_tank_masses_kg", endpoint),
    ):
        if np.any(masses < 0.0) or np.any(masses > capacities):
            raise ValueError(f"{name} must remain within tank capacities")

    endpoint_delta = endpoint - actual
    if not np.isclose(
        float(np.sum(endpoint_delta)),
        0.0,
        rtol=0.0,
        atol=_MASS_TOLERANCE_KG,
    ):
        raise ValueError(
            "diagnostic_target_tank_masses_kg must preserve total ballast mass "
            "relative to actual_tank_masses_kg"
        )

    samples = []
    for fraction in path_fractions:
        delta = fraction * endpoint_delta
        masses = actual + delta
        within_capacity = bool(
            np.all(masses >= -_MASS_TOLERANCE_KG)
            and np.all(masses <= capacities + _MASS_TOLERANCE_KG)
        )
        if not within_capacity:
            raise AssertionError("convex endpoint sample unexpectedly exceeds capacity")
        samples.append(
            BallastEndpointPathSample(
                fraction=fraction,
                hypothetical_tank_masses_kg=masses,
                mass_delta_from_actual_kg=delta,
                net_mass_delta_from_actual_kg=float(np.sum(delta)),
                within_capacity=True,
            )
        )
    return tuple(samples)


__all__ = [
    "BallastEndpointPathSample",
    "sample_ballast_endpoint_path",
]
