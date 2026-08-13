"""Small-angle six-degree-of-freedom platform model about a fixed equilibrium."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


DOF_COUNT = 6


def _vector(name: str, value: Any) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.shape != (DOF_COUNT,):
        raise ValueError(f"{name} must have shape (6,), got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    result = np.array(array, dtype=float, copy=True)
    result.setflags(write=False)
    return result


def _symmetric_matrix(name: str, value: Any) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.shape != (DOF_COUNT, DOF_COUNT):
        raise ValueError(f"{name} must have shape (6, 6), got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    if not np.allclose(array, array.T, rtol=1e-12, atol=1e-12):
        raise ValueError(f"{name} must be symmetric")
    result = np.array(array, dtype=float, copy=True)
    result.setflags(write=False)
    return result


def _require_positive_semidefinite(name: str, matrix: np.ndarray) -> None:
    eigenvalues = np.linalg.eigvalsh(matrix)
    tolerance = 1e-12 * max(1.0, float(np.max(np.abs(eigenvalues))))
    if float(np.min(eigenvalues)) < -tolerance:
        raise ValueError(f"{name} must be positive semidefinite")


@dataclass(frozen=True)
class PlatformMatrices:
    """Matrices expressed at one reference point in equilibrium platform axes."""

    mass: Any
    damping: Any
    hydrostatic_stiffness: Any
    mooring_stiffness: Any
    weight_stiffness: Any = None

    def __post_init__(self) -> None:
        mass = _symmetric_matrix("mass", self.mass)
        if np.min(np.linalg.eigvalsh(mass)) <= 0.0:
            raise ValueError("mass must be positive definite")
        damping = _symmetric_matrix("damping", self.damping)
        _require_positive_semidefinite("damping", damping)
        hydrostatic_stiffness = _symmetric_matrix(
            "hydrostatic_stiffness",
            self.hydrostatic_stiffness,
        )
        mooring_stiffness = _symmetric_matrix(
            "mooring_stiffness",
            self.mooring_stiffness,
        )
        weight_stiffness = _symmetric_matrix(
            "weight_stiffness",
            np.zeros((DOF_COUNT, DOF_COUNT))
            if self.weight_stiffness is None
            else self.weight_stiffness,
        )
        _require_positive_semidefinite(
            "total restoring stiffness",
            hydrostatic_stiffness + mooring_stiffness + weight_stiffness,
        )
        object.__setattr__(self, "mass", mass)
        object.__setattr__(self, "damping", damping)
        object.__setattr__(
            self,
            "hydrostatic_stiffness",
            hydrostatic_stiffness,
        )
        object.__setattr__(
            self,
            "mooring_stiffness",
            mooring_stiffness,
        )
        object.__setattr__(self, "weight_stiffness", weight_stiffness)

    @property
    def restoring_stiffness(self) -> np.ndarray:
        result = (
            self.hydrostatic_stiffness
            + self.mooring_stiffness
            + self.weight_stiffness
        )
        result.setflags(write=False)
        return result


@dataclass(frozen=True)
class IncrementalState:
    """Generalized displacement and its derivative about a fixed equilibrium."""

    position: Any
    velocity: Any

    def __post_init__(self) -> None:
        object.__setattr__(self, "position", _vector("position", self.position))
        object.__setattr__(self, "velocity", _vector("velocity", self.velocity))

    @classmethod
    def zeros(cls) -> "IncrementalState":
        return cls(position=np.zeros(DOF_COUNT), velocity=np.zeros(DOF_COUNT))


@dataclass(frozen=True)
class IncrementalLoads:
    """Incremental external loads, excluding left-hand-side restoring terms."""

    wind: Any
    wave: Any
    ballast: Any
    other: Any

    def __post_init__(self) -> None:
        for name in ("wind", "wave", "ballast", "other"):
            object.__setattr__(self, name, _vector(name, getattr(self, name)))

    @classmethod
    def zeros(cls) -> "IncrementalLoads":
        zero = np.zeros(DOF_COUNT)
        return cls(wind=zero, wave=zero, ballast=zero, other=zero)

    @property
    def total(self) -> np.ndarray:
        result = self.wind + self.wave + self.ballast + self.other
        result.setflags(write=False)
        return result


@dataclass(frozen=True)
class IncrementalStateDerivative:
    """Time derivative of an incremental platform state."""

    position_rate: Any
    velocity_rate: Any

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "position_rate",
            _vector("position_rate", self.position_rate),
        )
        object.__setattr__(
            self,
            "velocity_rate",
            _vector("velocity_rate", self.velocity_rate),
        )


@dataclass(frozen=True)
class IncrementalStaticOffset:
    """Static offset from a separately established reference equilibrium."""

    position: Any
    residual_load: Any

    def __post_init__(self) -> None:
        object.__setattr__(self, "position", _vector("position", self.position))
        object.__setattr__(
            self,
            "residual_load",
            _vector("residual_load", self.residual_load),
        )


def solve_incremental_static_offset(
    matrices: PlatformMatrices,
    incremental_static_load: Any,
) -> IncrementalStaticOffset:
    """Balance an incremental static load about a fixed reference equilibrium."""

    if not isinstance(matrices, PlatformMatrices):
        raise TypeError("matrices must be PlatformMatrices")
    load = _vector("incremental_static_load", incremental_static_load)
    stiffness = matrices.restoring_stiffness
    try:
        position = np.linalg.solve(stiffness, load)
    except np.linalg.LinAlgError as error:
        raise ValueError(
            "restoring stiffness is singular and cannot balance the static load"
        ) from error
    residual = load - stiffness @ position
    return IncrementalStaticOffset(position=position, residual_load=residual)


class IncrementalPlatformModel:
    """Evaluate the low-order dynamics without time integration or control logic."""

    def __init__(self, matrices: PlatformMatrices) -> None:
        if not isinstance(matrices, PlatformMatrices):
            raise TypeError("matrices must be PlatformMatrices")
        self._matrices = matrices

    @property
    def matrices(self) -> PlatformMatrices:
        return self._matrices

    def derivative(
        self,
        state: IncrementalState,
        loads: IncrementalLoads,
    ) -> IncrementalStateDerivative:
        if not isinstance(state, IncrementalState):
            raise TypeError("state must be IncrementalState")
        if not isinstance(loads, IncrementalLoads):
            raise TypeError("loads must be IncrementalLoads")
        net_load = (
            loads.total
            - self._matrices.damping @ state.velocity
            - self._matrices.restoring_stiffness @ state.position
        )
        acceleration = np.linalg.solve(self._matrices.mass, net_load)
        return IncrementalStateDerivative(
            position_rate=state.velocity,
            velocity_rate=acceleration,
        )
