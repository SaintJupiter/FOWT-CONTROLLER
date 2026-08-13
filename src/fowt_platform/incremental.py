"""Small-angle six-degree-of-freedom platform model about a fixed equilibrium."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.linalg import expm


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


def _three_vector(name: str, value: Any) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.shape != (3,):
        raise ValueError(f"{name} must have shape (3,), got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return np.array(array, dtype=float, copy=True)


def _finite_scalar(name: str, value: Any) -> float:
    try:
        scalar = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not np.isfinite(scalar):
        raise ValueError(f"{name} must be finite")
    return scalar


def generalized_load_from_point_force(
    force_platform_n: Any,
    application_point_from_reference_m: Any,
) -> np.ndarray:
    """Map a platform-coordinate point force to a six-DOF generalized load.

    Both arguments use the fixed equilibrium platform axes.  The returned
    vector follows ``[X, Y, Z, K, M, N]`` and its moment is taken about the
    platform reference point.
    """

    force = _three_vector("force_platform_n", force_platform_n)
    point = _three_vector(
        "application_point_from_reference_m",
        application_point_from_reference_m,
    )
    result = np.concatenate((force, np.cross(point, force)))
    result.setflags(write=False)
    return result


def _openfast_platform_to_inertial_rotation(
    *,
    platform_roll_deg: Any,
    platform_pitch_deg: Any,
    platform_yaw_deg: Any,
) -> np.ndarray:
    """Return OpenFAST platform axes expressed in the OpenFAST inertial axes.

    This reproduces the rigid platform yaw-pitch-roll construction used by
    ElastoDyn for the narrow, undeformed-tower audit configuration.  Its
    columns are the instantaneous platform forward, port and up axes in the
    equilibrium reference axes.
    """

    roll, pitch, yaw = np.deg2rad(
        [
            _finite_scalar("platform_roll_deg", platform_roll_deg),
            _finite_scalar("platform_pitch_deg", platform_pitch_deg),
            _finite_scalar("platform_yaw_deg", platform_yaw_deg),
        ]
    )
    forward_after_yaw = np.array([np.cos(yaw), np.sin(yaw), 0.0])
    port_after_yaw = np.array([-np.sin(yaw), np.cos(yaw), 0.0])
    up_reference = np.array([0.0, 0.0, 1.0])

    forward_after_pitch = (
        np.cos(pitch) * forward_after_yaw - np.sin(pitch) * up_reference
    )
    up_after_pitch = (
        np.sin(pitch) * forward_after_yaw + np.cos(pitch) * up_reference
    )
    port_after_roll = (
        np.cos(roll) * port_after_yaw + np.sin(roll) * up_after_pitch
    )
    up_after_roll = (
        -np.sin(roll) * port_after_yaw + np.cos(roll) * up_after_pitch
    )
    return np.column_stack(
        (forward_after_pitch, port_after_roll, up_after_roll)
    )


def openfast_hub_to_reference_rotation(
    *,
    shaft_tilt_deg: Any,
    azimuth_deg: Any,
    platform_roll_deg: Any = 0.0,
    platform_pitch_deg: Any = 0.0,
    platform_yaw_deg: Any = 0.0,
    nacelle_yaw_deg: Any = 0.0,
    azimuth_blade1_up_deg: Any = 0.0,
    equilibrium_platform_roll_deg: Any = 0.0,
    equilibrium_platform_pitch_deg: Any = 0.0,
    equilibrium_platform_yaw_deg: Any = 0.0,
) -> np.ndarray:
    """Return the AeroDyn ``h`` axes expressed in frozen platform axes.

    The result follows ElastoDyn's rigid coordinate construction for the
    prescribed-rotor OpenFAST node: its columns are ``x_h``, ``y_h`` and
    ``z_h`` in the frozen equilibrium platform axes.  It is intentionally
    limited to the audit configuration, in which tower deformation, rotor
    furl, shaft skew, Delta-3 effects and nacelle yaw are disabled or zero.
    """

    tilt = np.deg2rad(_finite_scalar("shaft_tilt_deg", shaft_tilt_deg))
    nacelle_yaw = np.deg2rad(
        _finite_scalar("nacelle_yaw_deg", nacelle_yaw_deg)
    )
    if not np.isclose(nacelle_yaw, 0.0, atol=1e-12):
        raise ValueError(
            "nacelle_yaw_deg must be zero in the prescribed-rotor load audit"
        )
    azimuth = np.deg2rad(_finite_scalar("azimuth_deg", azimuth_deg))
    blade1_up = np.deg2rad(
        _finite_scalar("azimuth_blade1_up_deg", azimuth_blade1_up_deg)
    )
    platform_rotation = _openfast_platform_to_inertial_rotation(
        platform_roll_deg=platform_roll_deg,
        platform_pitch_deg=platform_pitch_deg,
        platform_yaw_deg=platform_yaw_deg,
    )
    platform_forward = platform_rotation[:, 0]
    platform_port = platform_rotation[:, 1]
    platform_up = platform_rotation[:, 2]

    nacelle_forward = platform_forward
    nacelle_port = platform_port
    shaft_axis = (
        np.cos(tilt) * nacelle_forward + np.sin(tilt) * platform_up
    )
    shaft_vertical = (
        -np.sin(tilt) * nacelle_forward + np.cos(tilt) * platform_up
    )
    rotor_starboard = -nacelle_port
    rotor_azimuth = azimuth - blade1_up - np.pi / 2.0
    hub_port = (
        np.cos(rotor_azimuth) * shaft_vertical
        + np.sin(rotor_azimuth) * rotor_starboard
    )
    hub_up = (
        -np.sin(rotor_azimuth) * shaft_vertical
        + np.cos(rotor_azimuth) * rotor_starboard
    )
    equilibrium_rotation = _openfast_platform_to_inertial_rotation(
        platform_roll_deg=equilibrium_platform_roll_deg,
        platform_pitch_deg=equilibrium_platform_pitch_deg,
        platform_yaw_deg=equilibrium_platform_yaw_deg,
    )
    rotation = equilibrium_rotation.T @ np.column_stack(
        (shaft_axis, hub_port, hub_up)
    )
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-12):
        raise RuntimeError("OpenFAST hub-coordinate rotation is not orthogonal")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-12):
        raise RuntimeError("OpenFAST hub-coordinate rotation is not right handed")
    rotation.setflags(write=False)
    return rotation


def generalized_load_from_openfast_hub_wrench(
    force_hub_n: Any,
    moment_hub_nm: Any,
    hub_from_platform_reference_m: Any,
    *,
    shaft_tilt_deg: Any,
    azimuth_deg: Any,
    platform_roll_deg: Any = 0.0,
    platform_pitch_deg: Any = 0.0,
    platform_yaw_deg: Any = 0.0,
    nacelle_yaw_deg: Any = 0.0,
    azimuth_blade1_up_deg: Any = 0.0,
    equilibrium_platform_roll_deg: Any = 0.0,
    equilibrium_platform_pitch_deg: Any = 0.0,
    equilibrium_platform_yaw_deg: Any = 0.0,
) -> np.ndarray:
    """Map an AeroDyn ``h`` wrench to a load about the platform reference.

    ``force_hub_n`` and ``moment_hub_nm`` are AeroDyn hub-coordinate results,
    with the moment taken about the hub.  ``hub_from_platform_reference_m`` is
    the hub location in the undeformed platform axes.  The returned load is
    ordered as ``[X, Y, Z, K, M, N]`` in the frozen equilibrium axes.
    """

    force_hub = _three_vector("force_hub_n", force_hub_n)
    moment_hub = _three_vector("moment_hub_nm", moment_hub_nm)
    hub_from_reference = _three_vector(
        "hub_from_platform_reference_m",
        hub_from_platform_reference_m,
    )
    hub_rotation = openfast_hub_to_reference_rotation(
        shaft_tilt_deg=shaft_tilt_deg,
        azimuth_deg=azimuth_deg,
        platform_roll_deg=platform_roll_deg,
        platform_pitch_deg=platform_pitch_deg,
        platform_yaw_deg=platform_yaw_deg,
        nacelle_yaw_deg=nacelle_yaw_deg,
        azimuth_blade1_up_deg=azimuth_blade1_up_deg,
        equilibrium_platform_roll_deg=equilibrium_platform_roll_deg,
        equilibrium_platform_pitch_deg=equilibrium_platform_pitch_deg,
        equilibrium_platform_yaw_deg=equilibrium_platform_yaw_deg,
    )
    platform_rotation = _openfast_platform_to_inertial_rotation(
        platform_roll_deg=platform_roll_deg,
        platform_pitch_deg=platform_pitch_deg,
        platform_yaw_deg=platform_yaw_deg,
    )
    equilibrium_rotation = _openfast_platform_to_inertial_rotation(
        platform_roll_deg=equilibrium_platform_roll_deg,
        platform_pitch_deg=equilibrium_platform_pitch_deg,
        platform_yaw_deg=equilibrium_platform_yaw_deg,
    )
    relative_platform_rotation = equilibrium_rotation.T @ platform_rotation
    force_reference = hub_rotation @ force_hub
    moment_reference = (
        hub_rotation @ moment_hub
        + np.cross(
            relative_platform_rotation @ hub_from_reference,
            force_reference,
        )
    )
    result = np.concatenate((force_reference, moment_reference))
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
    """Evaluate or advance frozen low-order dynamics without control logic."""

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

    def advance_frozen_step(
        self,
        state: IncrementalState,
        loads: IncrementalLoads,
        duration_s: float,
    ) -> IncrementalState:
        """Advance one step while matrices and external loads stay fixed."""

        if not isinstance(state, IncrementalState):
            raise TypeError("state must be IncrementalState")
        if not isinstance(loads, IncrementalLoads):
            raise TypeError("loads must be IncrementalLoads")
        try:
            duration = float(duration_s)
        except (TypeError, ValueError) as exc:
            raise ValueError("duration_s must be finite and non-negative") from exc
        if not np.isfinite(duration) or duration < 0.0:
            raise ValueError("duration_s must be finite and non-negative")
        if duration == 0.0:
            return state

        mass_inverse_stiffness = np.linalg.solve(
            self._matrices.mass,
            self._matrices.restoring_stiffness,
        )
        mass_inverse_damping = np.linalg.solve(
            self._matrices.mass,
            self._matrices.damping,
        )
        mass_inverse_load = np.linalg.solve(self._matrices.mass, loads.total)
        affine_system = np.zeros((13, 13), dtype=float)
        affine_system[:6, 6:12] = np.eye(6)
        affine_system[6:12, :6] = -mass_inverse_stiffness
        affine_system[6:12, 6:12] = -mass_inverse_damping
        affine_system[6:12, 12] = mass_inverse_load
        initial = np.concatenate((state.position, state.velocity, [1.0]))
        advanced = expm(affine_system * duration) @ initial
        return IncrementalState(
            position=advanced[:6],
            velocity=advanced[6:12],
        )
