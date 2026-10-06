"""Resolve forecast wind records to explicit rotor-relative normal inflows.

The module sits immediately upstream of aerodynamic operating-point selection.
It applies the same geometry, height-transfer and frozen platform-kinematics
assumptions used by ``forecast_physical_load`` but deliberately stops before
choosing a thrust coefficient.  A caller can therefore bind a table-based
rotor schedule to the same relative normal wind that later forms the load.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from fowt_platform.coordinates import (
    downwind_normal_relative_wind_component,
    enu_wind_to_platform,
    relative_air_velocity_at_platform_point,
)

from .forecast_evidence import ForecastEvidence, validate_forecast_evidence
from .forecast_physical_load import (
    ForecastRotorLoadParameters,
    ForecastRotorOperatingState,
    FuturePlatformKinematicsMode,
    FutureRotorOrientationMode,
    rotor_geometry_at_declared_nacelle_yaw,
)


def _nonempty_text(name: str, value: Any) -> str:
    result = str(value).strip()
    if not result:
        raise ValueError(f"{name} must be a non-empty string")
    return result


def _finite_scalar(name: str, value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _vector(name: str, value: Any) -> np.ndarray:
    try:
        result = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must have shape (3,) and finite values") from exc
    if result.shape != (3,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must have shape (3,) and finite values")
    copied = np.array(result, dtype=float, copy=True)
    copied.setflags(write=False)
    return copied


def _future_vectors(name: str, value: Any) -> np.ndarray:
    try:
        result = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must have shape (H, 3) and finite values") from exc
    if (
        result.ndim != 2
        or result.shape[0] == 0
        or result.shape[1] != 3
        or not np.all(np.isfinite(result))
    ):
        raise ValueError(f"{name} must have shape (H, 3) and finite values")
    copied = np.array(result, dtype=float, copy=True)
    copied.setflags(write=False)
    return copied


def _horizontal_vector(name: str, value: Any) -> np.ndarray:
    try:
        result = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must have shape (2,) and finite values") from exc
    if result.shape != (2,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must have shape (2,) and finite values")
    return np.array(result, dtype=float, copy=True)


@dataclass(frozen=True)
class ForecastRotorKinematics:
    """Caller-owned platform and nacelle states used across forecast leads.

    This is not a platform-motion predictor.  It merely carries the explicit
    kinematic and frozen-orientation inputs required to convert wind records
    to rotor-relative inflow and then to create a compatible
    ``ForecastRotorOperatingState``.
    """

    current_platform_reference_velocity_platform_mps: Any
    current_platform_angular_velocity_platform_radps: Any
    future_platform_reference_velocities_platform_mps: Any
    future_platform_angular_velocities_platform_radps: Any
    future_platform_kinematics_mode: FuturePlatformKinematicsMode | str
    future_platform_kinematics_source: str
    current_nacelle_yaw_relative_platform_rad: float = 0.0
    future_rotor_orientation_mode: FutureRotorOrientationMode | str = (
        FutureRotorOrientationMode.FROZEN_CURRENT
    )
    future_rotor_orientation_source: str = "explicit_frozen_reference_orientation"

    def __post_init__(self) -> None:
        current_velocity = _vector(
            "current_platform_reference_velocity_platform_mps",
            self.current_platform_reference_velocity_platform_mps,
        )
        current_angular_velocity = _vector(
            "current_platform_angular_velocity_platform_radps",
            self.current_platform_angular_velocity_platform_radps,
        )
        future_velocities = _future_vectors(
            "future_platform_reference_velocities_platform_mps",
            self.future_platform_reference_velocities_platform_mps,
        )
        future_angular_velocities = _future_vectors(
            "future_platform_angular_velocities_platform_radps",
            self.future_platform_angular_velocities_platform_radps,
        )
        if future_velocities.shape != future_angular_velocities.shape:
            raise ValueError("future platform velocity arrays must have the same shape")
        try:
            kinematics_mode = FuturePlatformKinematicsMode(
                self.future_platform_kinematics_mode
            )
        except ValueError as exc:
            raise ValueError(
                "future_platform_kinematics_mode must be 'frozen_zero' or "
                "'explicit_per_lead'"
            ) from exc
        if kinematics_mode is FuturePlatformKinematicsMode.FROZEN_ZERO and (
            np.any(future_velocities != 0.0)
            or np.any(future_angular_velocities != 0.0)
        ):
            raise ValueError(
                "frozen_zero future platform kinematics require zero velocities"
            )
        try:
            orientation_mode = FutureRotorOrientationMode(
                self.future_rotor_orientation_mode
            )
        except ValueError as exc:
            raise ValueError("future_rotor_orientation_mode must be 'frozen_current'") from exc
        yaw = _finite_scalar(
            "current_nacelle_yaw_relative_platform_rad",
            self.current_nacelle_yaw_relative_platform_rad,
        )
        orientation_source = _nonempty_text(
            "future_rotor_orientation_source", self.future_rotor_orientation_source
        )
        if yaw != 0.0 and orientation_source == "explicit_frozen_reference_orientation":
            raise ValueError(
                "nonzero nacelle yaw requires an explicit future_rotor_orientation_source"
            )
        object.__setattr__(
            self,
            "current_platform_reference_velocity_platform_mps",
            current_velocity,
        )
        object.__setattr__(
            self,
            "current_platform_angular_velocity_platform_radps",
            current_angular_velocity,
        )
        object.__setattr__(
            self,
            "future_platform_reference_velocities_platform_mps",
            future_velocities,
        )
        object.__setattr__(
            self,
            "future_platform_angular_velocities_platform_radps",
            future_angular_velocities,
        )
        object.__setattr__(self, "future_platform_kinematics_mode", kinematics_mode)
        object.__setattr__(
            self,
            "future_platform_kinematics_source",
            _nonempty_text(
                "future_platform_kinematics_source",
                self.future_platform_kinematics_source,
            ),
        )
        object.__setattr__(self, "current_nacelle_yaw_relative_platform_rad", yaw)
        object.__setattr__(self, "future_rotor_orientation_mode", orientation_mode)
        object.__setattr__(
            self,
            "future_rotor_orientation_source",
            orientation_source,
        )

    @property
    def horizon_steps(self) -> int:
        return int(self.future_platform_reference_velocities_platform_mps.shape[0])

    def to_operating_state(
        self,
        *,
        current_thrust_coefficient: Any,
        future_thrust_coefficients: Any,
    ) -> ForecastRotorOperatingState:
        """Bind table-resolved coefficients to these unchanged kinematics."""

        try:
            future_coefficients = np.asarray(
                future_thrust_coefficients, dtype=float
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "future_thrust_coefficients must contain one finite non-negative "
                "value per future platform lead"
            ) from exc
        if (
            future_coefficients.shape != (self.horizon_steps,)
            or not np.all(np.isfinite(future_coefficients))
            or np.any(future_coefficients < 0.0)
        ):
            raise ValueError(
                "future_thrust_coefficients must contain one finite non-negative "
                "value per future platform lead"
            )

        return ForecastRotorOperatingState(
            current_thrust_coefficient=current_thrust_coefficient,
            future_thrust_coefficients=future_coefficients,
            current_platform_reference_velocity_platform_mps=(
                self.current_platform_reference_velocity_platform_mps
            ),
            current_platform_angular_velocity_platform_radps=(
                self.current_platform_angular_velocity_platform_radps
            ),
            future_platform_reference_velocities_platform_mps=(
                self.future_platform_reference_velocities_platform_mps
            ),
            future_platform_angular_velocities_platform_radps=(
                self.future_platform_angular_velocities_platform_radps
            ),
            future_platform_kinematics_mode=self.future_platform_kinematics_mode,
            future_platform_kinematics_source=self.future_platform_kinematics_source,
            current_nacelle_yaw_relative_platform_rad=(
                self.current_nacelle_yaw_relative_platform_rad
            ),
            future_rotor_orientation_mode=self.future_rotor_orientation_mode,
            future_rotor_orientation_source=self.future_rotor_orientation_source,
        )


@dataclass(frozen=True)
class ForecastRotorRelativeInflow:
    """Signed current and future normal inflows in one declared geometry."""

    forecast_source: str
    forecast_model_version: str
    forecast_origin_time: str | None
    current_wind_source: str
    current_wind_observation_time: str | None
    current_normal_relative_inflow_mps: float
    future_normal_relative_inflows_mps: Any
    parameters: ForecastRotorLoadParameters
    kinematics: ForecastRotorKinematics

    def __post_init__(self) -> None:
        for field in (
            "forecast_source",
            "forecast_model_version",
            "current_wind_source",
        ):
            object.__setattr__(self, field, _nonempty_text(field, getattr(self, field)))
        for field in ("forecast_origin_time", "current_wind_observation_time"):
            value = getattr(self, field)
            if value is not None:
                object.__setattr__(self, field, _nonempty_text(field, value))
        current = _finite_scalar(
            "current_normal_relative_inflow_mps",
            self.current_normal_relative_inflow_mps,
        )
        try:
            future = np.asarray(self.future_normal_relative_inflows_mps, dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "future_normal_relative_inflows_mps must be a finite one-dimensional sequence"
            ) from exc
        if future.ndim != 1 or future.shape[0] == 0 or not np.all(np.isfinite(future)):
            raise ValueError(
                "future_normal_relative_inflows_mps must be a finite one-dimensional sequence"
            )
        if not isinstance(self.parameters, ForecastRotorLoadParameters):
            raise TypeError("parameters must be a ForecastRotorLoadParameters")
        if not isinstance(self.kinematics, ForecastRotorKinematics):
            raise TypeError("kinematics must be a ForecastRotorKinematics")
        if future.shape[0] != self.kinematics.horizon_steps:
            raise ValueError("future inflow count must match kinematics horizon")
        copied = np.array(future, dtype=float, copy=True)
        copied.setflags(write=False)
        object.__setattr__(self, "current_normal_relative_inflow_mps", current)
        object.__setattr__(self, "future_normal_relative_inflows_mps", copied)


def assemble_forecast_rotor_relative_inflow(
    *,
    forecast: ForecastEvidence,
    current_enu_downwind_air_velocity_mps: Any,
    current_wind_source: str,
    current_wind_observation_time: str | None,
    parameters: ForecastRotorLoadParameters,
    kinematics: ForecastRotorKinematics,
) -> ForecastRotorRelativeInflow:
    """Resolve the forecast to signed relative inflow before choosing ``C_T``.

    The result retains negative values.  A positive-thrust operating schedule
    must explicitly reject or otherwise model such a direction mismatch;
    this low-level conversion never flips a sign or converts it to a magnitude.
    """

    if not isinstance(forecast, ForecastEvidence):
        raise TypeError("forecast must be a ForecastEvidence")
    validate_forecast_evidence(forecast)
    if not forecast.provides_future_preview:
        raise ValueError("forecast must provide a future preview")
    if not isinstance(parameters, ForecastRotorLoadParameters):
        raise TypeError("parameters must be a ForecastRotorLoadParameters")
    if not isinstance(kinematics, ForecastRotorKinematics):
        raise TypeError("kinematics must be a ForecastRotorKinematics")
    if kinematics.horizon_steps != forecast.horizon_steps:
        raise ValueError("kinematics horizon must match forecast horizon")
    current_wind = _horizontal_vector(
        "current_enu_downwind_air_velocity_mps",
        current_enu_downwind_air_velocity_mps,
    )
    source = _nonempty_text("current_wind_source", current_wind_source)
    observation_time = (
        None
        if current_wind_observation_time is None
        else _nonempty_text("current_wind_observation_time", current_wind_observation_time)
    )
    if forecast.origin_time is not None and observation_time != forecast.origin_time:
        raise ValueError("current_wind_observation_time must match forecast.origin_time")

    rotor_geometry = rotor_geometry_at_declared_nacelle_yaw(
        parameters,
        kinematics.current_nacelle_yaw_relative_platform_rad,
    )
    application_point, normal = rotor_geometry
    speed_scale = parameters.rotor_plane_speed_scale

    def normal_inflow(
        enu_wind: np.ndarray,
        reference_velocity: np.ndarray,
        angular_velocity: np.ndarray,
    ) -> float:
        platform_horizontal = enu_wind_to_platform(
            enu_wind * speed_scale,
            parameters.frozen_equilibrium_heading_rad,
        )
        ambient = np.array(
            [platform_horizontal[0], platform_horizontal[1], 0.0], dtype=float
        )
        relative = relative_air_velocity_at_platform_point(
            ambient,
            reference_velocity,
            angular_velocity,
            application_point,
        )
        return downwind_normal_relative_wind_component(relative, normal)

    current_inflow = normal_inflow(
        current_wind,
        kinematics.current_platform_reference_velocity_platform_mps,
        kinematics.current_platform_angular_velocity_platform_radps,
    )
    future_winds = np.asarray(forecast.uv_ms, dtype=float)
    future_inflows = np.asarray(
        [
            normal_inflow(
                future_winds[index],
                kinematics.future_platform_reference_velocities_platform_mps[index],
                kinematics.future_platform_angular_velocities_platform_radps[index],
            )
            for index in range(forecast.horizon_steps)
        ],
        dtype=float,
    )
    return ForecastRotorRelativeInflow(
        forecast_source=forecast.source,
        forecast_model_version=forecast.model_version,
        forecast_origin_time=forecast.origin_time,
        current_wind_source=source,
        current_wind_observation_time=observation_time,
        current_normal_relative_inflow_mps=current_inflow,
        future_normal_relative_inflows_mps=future_inflows,
        parameters=parameters,
        kinematics=kinematics,
    )


__all__ = [
    "ForecastRotorKinematics",
    "ForecastRotorRelativeInflow",
    "assemble_forecast_rotor_relative_inflow",
]
