"""Explicit conversion from discrete wind forecasts to physical rotor loads.

``ForecastEvidence`` contains ENU downwind wind-vector records at discrete
future timestamps, whereas ``GeneralizedLoadForecast`` contains six-component
loads in frozen platform axes.  This module joins the two only when every
aerodynamic and kinematic assumption is supplied by the caller.

It does not select a turbine operating state, infer a height correction,
interpolate between prediction points, schedule pumps, or score actions.  Its
output is therefore a physical-input contract for later diagnostics, not an
execution command.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np

from fowt_platform.generalized_load_forecast import GeneralizedLoadForecast
from fowt_platform.rotor_input import (
    RotorGeneralizedLoad,
    quasi_steady_rotor_generalized_load_from_enu_wind,
)

from .forecast_evidence import ForecastEvidence, validate_forecast_evidence


def _finite_scalar(name: str, value: Any) -> float:
    try:
        scalar = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not np.isfinite(scalar):
        raise ValueError(f"{name} must be finite")
    return scalar


def _finite_positive(name: str, value: Any) -> float:
    scalar = _finite_scalar(name, value)
    if scalar <= 0.0:
        raise ValueError(f"{name} must be positive")
    return scalar


def _finite_nonnegative(name: str, value: Any) -> float:
    scalar = _finite_scalar(name, value)
    if scalar < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return scalar


def _vector(name: str, value: Any, *, length: int) -> np.ndarray:
    try:
        vector = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must have shape ({length},) and finite values") from exc
    if vector.shape != (length,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must have shape ({length},) and finite values")
    result = np.array(vector, dtype=float, copy=True)
    result.setflags(write=False)
    return result


def _future_vectors(name: str, value: Any, *, horizon_steps: int) -> np.ndarray:
    try:
        vectors = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{name} must have shape ({horizon_steps}, 3) and finite values"
        ) from exc
    if vectors.shape != (horizon_steps, 3) or not np.all(np.isfinite(vectors)):
        raise ValueError(
            f"{name} must have shape ({horizon_steps}, 3) and finite values"
        )
    result = np.array(vectors, dtype=float, copy=True)
    result.setflags(write=False)
    return result


def _future_coefficients(value: Any, *, horizon_steps: int) -> np.ndarray:
    try:
        coefficients = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "future_thrust_coefficients must contain one finite non-negative "
            "value per forecast lead"
        ) from exc
    if coefficients.shape != (horizon_steps,) or not np.all(np.isfinite(coefficients)):
        raise ValueError(
            "future_thrust_coefficients must contain one finite non-negative "
            "value per forecast lead"
        )
    if np.any(coefficients < 0.0):
        raise ValueError("future_thrust_coefficients must be non-negative")
    result = np.array(coefficients, dtype=float, copy=True)
    result.setflags(write=False)
    return result


def _readonly_future_coefficients(value: Any) -> np.ndarray:
    """Copy a non-empty future coefficient sequence before it crosses a boundary."""

    try:
        coefficients = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "future_thrust_coefficients must be a non-empty finite sequence"
        ) from exc
    if (
        coefficients.ndim != 1
        or coefficients.shape[0] == 0
        or not np.all(np.isfinite(coefficients))
        or np.any(coefficients < 0.0)
    ):
        raise ValueError(
            "future_thrust_coefficients must be a non-empty finite non-negative sequence"
        )
    result = np.array(coefficients, dtype=float, copy=True)
    result.setflags(write=False)
    return result


def _readonly_future_vectors(name: str, value: Any) -> np.ndarray:
    """Copy non-empty H-by-3 future kinematics before they enter an assembly."""

    try:
        vectors = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{name} must have shape (H, 3) with H > 0 and finite values"
        ) from exc
    if (
        vectors.ndim != 2
        or vectors.shape[0] == 0
        or vectors.shape[1] != 3
        or not np.all(np.isfinite(vectors))
    ):
        raise ValueError(
            f"{name} must have shape (H, 3) with H > 0 and finite values"
        )
    result = np.array(vectors, dtype=float, copy=True)
    result.setflags(write=False)
    return result


def _rotate_platform_vector_about_up(vector: Any, yaw_rad: Any) -> np.ndarray:
    """Rotate a frozen-platform vector about positive platform ``z``.

    Platform axes use ``[forward, port, up]``.  A positive relative nacelle
    yaw therefore takes a forward-pointing vector toward port, which is the
    same right-hand convention used by ``true_heading_from_yaw``.  The helper
    rotates a reference geometry only.  It never changes the platform heading
    used to resolve ENU wind into platform coordinates.
    """

    source = _vector("rotor reference geometry", vector, length=3)
    yaw = _finite_scalar("nacelle_yaw_relative_platform_rad", yaw_rad)
    cosine = float(np.cos(yaw))
    sine = float(np.sin(yaw))
    result = np.array(
        [
            cosine * source[0] - sine * source[1],
            sine * source[0] + cosine * source[1],
            source[2],
        ],
        dtype=float,
    )
    result.setflags(write=False)
    return result


def rotor_geometry_at_declared_nacelle_yaw(
    parameters: "ForecastRotorLoadParameters",
    yaw_rad: Any,
) -> tuple[np.ndarray, np.ndarray]:
    """Return one internally consistent normal and hub point at declared yaw.

    The normal rotates about the platform ``+z`` axis.  The hub application
    point rotates only about the declared nacelle yaw-axis point.  Rotating
    the whole hub vector about the platform reference would incorrectly move
    fixed tower geometry whenever that reference differs from the yaw axis.
    """

    yaw = _finite_scalar("nacelle_yaw_relative_platform_rad", yaw_rad)
    reference_application_point = parameters.rotor_force_application_point_from_reference_m
    normal = _rotate_platform_vector_about_up(
        parameters.downwind_rotor_normal_platform,
        yaw,
    )
    if yaw == 0.0:
        application_point = np.array(reference_application_point, dtype=float, copy=True)
    else:
        yaw_axis = parameters.nacelle_yaw_axis_point_from_reference_m
        if yaw_axis is None:
            raise ValueError(
                "nonzero nacelle yaw requires nacelle_yaw_axis_point_from_reference_m"
            )
        application_point = np.array(
            yaw_axis
            + _rotate_platform_vector_about_up(
                reference_application_point - yaw_axis,
                yaw,
            ),
            dtype=float,
            copy=True,
        )
    application_point.setflags(write=False)
    # Rotation preserves the norm.  Keep the invariant explicit to prevent a
    # future geometry implementation from silently changing normal magnitude.
    if not np.isclose(np.linalg.norm(normal), 1.0, rtol=0.0, atol=1e-12):
        raise AssertionError("rotated downwind rotor normal must remain unit length")
    return application_point, normal


def _assemble_rotor_load_at_declared_orientation(
    *,
    context: str,
    common: dict[str, Any],
    rotor_geometry: tuple[np.ndarray, np.ndarray],
    thrust_coefficient: Any,
    enu_downwind_air_velocity_mps: Any,
    platform_reference_velocity_platform_mps: Any,
    platform_angular_velocity_platform_radps: Any,
) -> RotorGeneralizedLoad:
    """Assemble one load while preserving an invalid-orientation rejection."""

    application_point, normal = rotor_geometry
    try:
        return quasi_steady_rotor_generalized_load_from_enu_wind(
            **common,
            thrust_coefficient=thrust_coefficient,
            enu_downwind_air_velocity_mps=enu_downwind_air_velocity_mps,
            platform_reference_velocity_platform_mps=(
                platform_reference_velocity_platform_mps
            ),
            platform_angular_velocity_platform_radps=(
                platform_angular_velocity_platform_radps
            ),
            rotor_force_application_point_from_reference_m=application_point,
            downwind_rotor_normal_platform=normal,
        )
    except ValueError as exc:
        if "negative component along downwind_rotor_normal_platform" not in str(exc):
            raise
        raise ValueError(
            f"{context} lies outside the declared positive-thrust rotor orientation "
            "branch; supply a compatible externally declared orientation or an "
            "explicit no-thrust operating state"
        ) from exc


@dataclass(frozen=True)
class ForecastRotorLoadParameters:
    """Explicit geometry, atmosphere and wind-height assumptions.

    The wind vector is treated as an effective horizontal ENU vector at
    ``wind_reference_height_m``.  The compact power-law transfer scales speed
    to ``rotor_plane_height_m`` without direction veer, vertical wind or a
    turbulence-profile model.  The stored rotor force application point and
    downwind normal are the reference geometry at zero nacelle yaw relative
    to the frozen platform axes.  A caller-selected yaw state rotates the
    normal and the hub offset about the declared nacelle yaw-axis point before
    a rotor load is assembled.
    """

    air_density_kg_m3: float
    rotor_radius_m: float
    wind_reference_height_m: float
    rotor_plane_height_m: float
    power_law_shear_exponent: float
    frozen_equilibrium_heading_rad: float
    rotor_force_application_point_from_reference_m: Any
    downwind_rotor_normal_platform: Any
    nacelle_yaw_axis_point_from_reference_m: Any | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "air_density_kg_m3",
            _finite_positive("air_density_kg_m3", self.air_density_kg_m3),
        )
        object.__setattr__(
            self,
            "rotor_radius_m",
            _finite_positive("rotor_radius_m", self.rotor_radius_m),
        )
        object.__setattr__(
            self,
            "wind_reference_height_m",
            _finite_positive("wind_reference_height_m", self.wind_reference_height_m),
        )
        object.__setattr__(
            self,
            "rotor_plane_height_m",
            _finite_positive("rotor_plane_height_m", self.rotor_plane_height_m),
        )
        object.__setattr__(
            self,
            "power_law_shear_exponent",
            _finite_scalar("power_law_shear_exponent", self.power_law_shear_exponent),
        )
        object.__setattr__(
            self,
            "frozen_equilibrium_heading_rad",
            _finite_scalar(
                "frozen_equilibrium_heading_rad",
                self.frozen_equilibrium_heading_rad,
            ),
        )
        object.__setattr__(
            self,
            "rotor_force_application_point_from_reference_m",
            _vector(
                "rotor_force_application_point_from_reference_m",
                self.rotor_force_application_point_from_reference_m,
                length=3,
            ),
        )
        normal = _vector(
            "downwind_rotor_normal_platform",
            self.downwind_rotor_normal_platform,
            length=3,
        )
        if not np.isclose(np.linalg.norm(normal), 1.0, rtol=0.0, atol=1e-12):
            raise ValueError("downwind_rotor_normal_platform must be a unit vector")
        object.__setattr__(self, "downwind_rotor_normal_platform", normal)
        yaw_axis = self.nacelle_yaw_axis_point_from_reference_m
        if yaw_axis is not None:
            yaw_axis = _vector(
                "nacelle_yaw_axis_point_from_reference_m",
                yaw_axis,
                length=3,
            )
        object.__setattr__(self, "nacelle_yaw_axis_point_from_reference_m", yaw_axis)

    @property
    def rotor_plane_speed_scale(self) -> float:
        """Return the declared scalar speed transfer to the rotor plane."""

        return float(
            (self.rotor_plane_height_m / self.wind_reference_height_m)
            ** self.power_law_shear_exponent
        )


class FuturePlatformKinematicsMode(str, Enum):
    """How platform motion is supplied at the discrete forecast leads."""

    FROZEN_ZERO = "frozen_zero"
    EXPLICIT_PER_LEAD = "explicit_per_lead"


class FutureRotorOrientationMode(str, Enum):
    """How nacelle yaw is supplied at the discrete forecast leads.

    The first physical-input version carries only the current caller-owned
    nacelle yaw and freezes it throughout the forecast horizon.  It does not
    introduce a yaw controller, infer alignment from wind, or change the
    platform heading used for ENU-to-platform conversion.
    """

    FROZEN_CURRENT = "frozen_current"


@dataclass(frozen=True)
class ForecastRotorOperatingState:
    """Caller-selected aerodynamic coefficients and platform-motion inputs.

    Future platform velocities have one row for every prediction point and
    must use the same ordering as the predicted wind leads.  The caller
    declares whether those rows are a deliberately frozen-zero approximation
    or an externally supplied per-lead trajectory.  This type does not
    predict future platform motion.
    """

    current_thrust_coefficient: float
    future_thrust_coefficients: Any
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
    future_rotor_orientation_source: str = "legacy_frozen_reference_orientation"

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "current_thrust_coefficient",
            _finite_nonnegative(
                "current_thrust_coefficient",
                self.current_thrust_coefficient,
            ),
        )
        object.__setattr__(
            self,
            "current_platform_reference_velocity_platform_mps",
            _vector(
                "current_platform_reference_velocity_platform_mps",
                self.current_platform_reference_velocity_platform_mps,
                length=3,
            ),
        )
        object.__setattr__(
            self,
            "current_platform_angular_velocity_platform_radps",
            _vector(
                "current_platform_angular_velocity_platform_radps",
                self.current_platform_angular_velocity_platform_radps,
                length=3,
            ),
        )
        future_reference_velocities = _readonly_future_vectors(
            "future_platform_reference_velocities_platform_mps",
            self.future_platform_reference_velocities_platform_mps,
        )
        future_angular_velocities = _readonly_future_vectors(
            "future_platform_angular_velocities_platform_radps",
            self.future_platform_angular_velocities_platform_radps,
        )
        try:
            kinematics_mode = FuturePlatformKinematicsMode(
                self.future_platform_kinematics_mode
            )
        except ValueError as exc:
            raise ValueError(
                "future_platform_kinematics_mode must be 'frozen_zero' or "
                "'explicit_per_lead'"
            ) from exc
        kinematics_source = str(self.future_platform_kinematics_source).strip()
        if not kinematics_source:
            raise ValueError("future_platform_kinematics_source must be non-empty")
        if kinematics_mode is FuturePlatformKinematicsMode.FROZEN_ZERO and (
            np.any(future_reference_velocities != 0.0)
            or np.any(future_angular_velocities != 0.0)
        ):
            raise ValueError(
                "frozen_zero future platform kinematics require zero velocities"
            )

        current_yaw = _finite_scalar(
            "current_nacelle_yaw_relative_platform_rad",
            self.current_nacelle_yaw_relative_platform_rad,
        )
        try:
            orientation_mode = FutureRotorOrientationMode(
                self.future_rotor_orientation_mode
            )
        except ValueError as exc:
            raise ValueError("future_rotor_orientation_mode must be 'frozen_current'") from exc
        orientation_source = str(self.future_rotor_orientation_source).strip()
        if not orientation_source:
            raise ValueError("future_rotor_orientation_source must be non-empty")
        if (
            current_yaw != 0.0
            and orientation_source == "legacy_frozen_reference_orientation"
        ):
            raise ValueError(
                "nonzero nacelle yaw requires an explicit "
                "future_rotor_orientation_source"
            )

        object.__setattr__(
            self,
            "future_thrust_coefficients",
            _readonly_future_coefficients(self.future_thrust_coefficients),
        )
        object.__setattr__(
            self,
            "future_platform_reference_velocities_platform_mps",
            future_reference_velocities,
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
            kinematics_source,
        )
        object.__setattr__(
            self,
            "current_nacelle_yaw_relative_platform_rad",
            current_yaw,
        )
        object.__setattr__(
            self,
            "future_rotor_orientation_mode",
            orientation_mode,
        )
        object.__setattr__(
            self,
            "future_rotor_orientation_source",
            orientation_source,
        )


@dataclass(frozen=True)
class ForecastPhysicalInputIdentity:
    """Immutable provenance of one forecast-to-load conversion.

    This intentionally retains identifiers and timing only.  The numeric
    forecast remains in the source ``ForecastEvidence`` object and is already
    represented by the generated physical load points held by the assembly.
    """

    forecast_source: str
    forecast_model_version: str
    forecast_origin_time: str | None
    forecast_sample_period_s: float
    forecast_horizon_steps: int
    current_wind_source: str
    current_wind_observation_time: str | None
    future_platform_kinematics_mode: FuturePlatformKinematicsMode | str
    future_platform_kinematics_source: str
    future_rotor_orientation_mode: FutureRotorOrientationMode | str
    future_rotor_orientation_source: str

    def __post_init__(self) -> None:
        for name in (
            "forecast_source",
            "forecast_model_version",
            "current_wind_source",
            "future_platform_kinematics_source",
            "future_rotor_orientation_source",
        ):
            value = str(getattr(self, name)).strip()
            if not value:
                raise ValueError(f"{name} must be a non-empty string")
            object.__setattr__(self, name, value)
        for name in ("forecast_origin_time", "current_wind_observation_time"):
            value = getattr(self, name)
            if value is not None:
                normalized = str(value).strip()
                if not normalized:
                    raise ValueError(f"{name} must be a non-empty string when provided")
                object.__setattr__(self, name, normalized)
        if (
            self.forecast_origin_time is not None
            and self.current_wind_observation_time is not None
            and self.forecast_origin_time != self.current_wind_observation_time
        ):
            raise ValueError(
                "current_wind_observation_time must match forecast_origin_time "
                "when both are provided"
            )
        object.__setattr__(
            self,
            "forecast_sample_period_s",
            _finite_positive(
                "forecast_sample_period_s", self.forecast_sample_period_s
            ),
        )
        steps = int(self.forecast_horizon_steps)
        if steps <= 0:
            raise ValueError("forecast_horizon_steps must be positive")
        object.__setattr__(self, "forecast_horizon_steps", steps)
        try:
            kinematics_mode = FuturePlatformKinematicsMode(
                self.future_platform_kinematics_mode
            )
        except ValueError as exc:
            raise ValueError(
                "future_platform_kinematics_mode must be 'frozen_zero' or "
                "'explicit_per_lead'"
            ) from exc
        object.__setattr__(self, "future_platform_kinematics_mode", kinematics_mode)
        try:
            orientation_mode = FutureRotorOrientationMode(
                self.future_rotor_orientation_mode
            )
        except ValueError as exc:
            raise ValueError("future_rotor_orientation_mode must be 'frozen_current'") from exc
        object.__setattr__(self, "future_rotor_orientation_mode", orientation_mode)


@dataclass(frozen=True)
class ForecastGeneralizedLoadAssembly:
    """Current and future rotor loads together with their exact assumptions."""

    load_forecast: GeneralizedLoadForecast
    current_rotor_load: RotorGeneralizedLoad
    future_rotor_loads: tuple[RotorGeneralizedLoad, ...]
    parameters: ForecastRotorLoadParameters
    operating_state: ForecastRotorOperatingState
    input_identity: ForecastPhysicalInputIdentity
    current_rotor_plane_enu_downwind_mps: Any
    future_rotor_plane_enu_downwind_mps: Any

    def __post_init__(self) -> None:
        if not isinstance(self.load_forecast, GeneralizedLoadForecast):
            raise TypeError("load_forecast must be GeneralizedLoadForecast")
        if not isinstance(self.current_rotor_load, RotorGeneralizedLoad):
            raise TypeError("current_rotor_load must be RotorGeneralizedLoad")
        if not isinstance(self.parameters, ForecastRotorLoadParameters):
            raise TypeError("parameters must be ForecastRotorLoadParameters")
        if not isinstance(self.operating_state, ForecastRotorOperatingState):
            raise TypeError("operating_state must be ForecastRotorOperatingState")
        if not isinstance(self.input_identity, ForecastPhysicalInputIdentity):
            raise TypeError("input_identity must be ForecastPhysicalInputIdentity")
        if (
            self.input_identity.forecast_horizon_steps
            != self.load_forecast.horizon_steps
        ):
            raise ValueError(
                "input_identity horizon must match load_forecast horizon"
            )
        expected_lead_times_s = (
            np.arange(1, self.load_forecast.horizon_steps + 1, dtype=float)
            * self.input_identity.forecast_sample_period_s
        )
        if not np.allclose(
            self.load_forecast.lead_times_s,
            expected_lead_times_s,
            rtol=0.0,
            atol=1.0e-9,
        ):
            raise ValueError(
                "load_forecast lead times must match input_identity forecast cadence"
            )
        if (
            self.input_identity.future_platform_kinematics_mode
            is not self.operating_state.future_platform_kinematics_mode
        ):
            raise ValueError(
                "input_identity future platform kinematics mode must match "
                "operating_state"
            )
        if (
            self.input_identity.future_platform_kinematics_source
            != self.operating_state.future_platform_kinematics_source
        ):
            raise ValueError(
                "input_identity future platform kinematics source must match "
                "operating_state"
            )
        if (
            self.input_identity.future_rotor_orientation_mode
            is not self.operating_state.future_rotor_orientation_mode
        ):
            raise ValueError(
                "input_identity future rotor orientation mode must match operating_state"
            )
        if (
            self.input_identity.future_rotor_orientation_source
            != self.operating_state.future_rotor_orientation_source
        ):
            raise ValueError(
                "input_identity future rotor orientation source must match operating_state"
            )
        horizon = self.load_forecast.horizon_steps
        if self.operating_state.future_thrust_coefficients.shape != (horizon,):
            raise ValueError(
                "operating_state future thrust coefficients must match load_forecast horizon"
            )
        if (
            self.operating_state.future_platform_reference_velocities_platform_mps.shape
            != (horizon, 3)
            or self.operating_state.future_platform_angular_velocities_platform_radps.shape
            != (horizon, 3)
        ):
            raise ValueError(
                "operating_state future platform kinematics must match load_forecast horizon"
            )
        future_loads = tuple(self.future_rotor_loads)
        if len(future_loads) != self.load_forecast.horizon_steps or not all(
            isinstance(load, RotorGeneralizedLoad) for load in future_loads
        ):
            raise ValueError(
                "future_rotor_loads must contain one RotorGeneralizedLoad per forecast lead"
            )
        if not np.allclose(
            self.load_forecast.current_generalized_load,
            self.current_rotor_load.generalized_load_platform,
            rtol=0.0,
            atol=1.0e-8,
        ):
            raise ValueError(
                "load_forecast current generalized load must match current_rotor_load"
            )
        expected_future_loads = np.asarray(
            [load.generalized_load_platform for load in future_loads],
            dtype=float,
        )
        if not np.allclose(
            self.load_forecast.future_generalized_loads,
            expected_future_loads,
            rtol=0.0,
            atol=1.0e-8,
        ):
            raise ValueError(
                "load_forecast future generalized loads must match future_rotor_loads"
            )
        object.__setattr__(self, "future_rotor_loads", future_loads)
        object.__setattr__(
            self,
            "current_rotor_plane_enu_downwind_mps",
            _vector(
                "current_rotor_plane_enu_downwind_mps",
                self.current_rotor_plane_enu_downwind_mps,
                length=2,
            ),
        )
        future_wind = np.asarray(self.future_rotor_plane_enu_downwind_mps, dtype=float)
        if (
            future_wind.shape != (self.load_forecast.horizon_steps, 2)
            or not np.all(np.isfinite(future_wind))
        ):
            raise ValueError(
                "future_rotor_plane_enu_downwind_mps must contain one finite ENU vector per forecast lead"
            )
        stored_wind = np.array(future_wind, dtype=float, copy=True)
        stored_wind.setflags(write=False)
        object.__setattr__(self, "future_rotor_plane_enu_downwind_mps", stored_wind)


def assemble_forecast_generalized_rotor_loads(
    *,
    forecast: ForecastEvidence,
    current_enu_downwind_air_velocity_mps: Any,
    current_wind_source: str,
    current_wind_observation_time: str | None,
    parameters: ForecastRotorLoadParameters,
    operating_state: ForecastRotorOperatingState,
) -> ForecastGeneralizedLoadAssembly:
    """Build a current load and discrete future load points from one forecast.

    The current wind is deliberately a separate input.  Future loads preserve
    the forecasting convention: the first one is at ``+sample_period_s`` and
    no future load is applied at the forecast origin by this assembly.
    """

    if not isinstance(forecast, ForecastEvidence):
        raise TypeError("forecast must be ForecastEvidence")
    validate_forecast_evidence(forecast)
    if not forecast.provides_future_preview:
        raise ValueError("forecast must provide a future preview")
    if not isinstance(parameters, ForecastRotorLoadParameters):
        raise TypeError("parameters must be ForecastRotorLoadParameters")
    if not isinstance(operating_state, ForecastRotorOperatingState):
        raise TypeError("operating_state must be ForecastRotorOperatingState")

    horizon_steps = forecast.horizon_steps
    input_identity = ForecastPhysicalInputIdentity(
        forecast_source=forecast.source,
        forecast_model_version=forecast.model_version,
        forecast_origin_time=forecast.origin_time,
        forecast_sample_period_s=forecast.sample_period_s,
        forecast_horizon_steps=horizon_steps,
        current_wind_source=current_wind_source,
        current_wind_observation_time=current_wind_observation_time,
        future_platform_kinematics_mode=(
            operating_state.future_platform_kinematics_mode
        ),
        future_platform_kinematics_source=(
            operating_state.future_platform_kinematics_source
        ),
        future_rotor_orientation_mode=(
            operating_state.future_rotor_orientation_mode
        ),
        future_rotor_orientation_source=(
            operating_state.future_rotor_orientation_source
        ),
    )
    current_wind = _vector(
        "current_enu_downwind_air_velocity_mps",
        current_enu_downwind_air_velocity_mps,
        length=2,
    )
    future_coefficients = _future_coefficients(
        operating_state.future_thrust_coefficients,
        horizon_steps=horizon_steps,
    )
    future_reference_velocities = _future_vectors(
        "future_platform_reference_velocities_platform_mps",
        operating_state.future_platform_reference_velocities_platform_mps,
        horizon_steps=horizon_steps,
    )
    future_angular_velocities = _future_vectors(
        "future_platform_angular_velocities_platform_radps",
        operating_state.future_platform_angular_velocities_platform_radps,
        horizon_steps=horizon_steps,
    )

    speed_scale = parameters.rotor_plane_speed_scale
    current_rotor_plane_wind = current_wind * speed_scale
    future_rotor_plane_winds = np.asarray(forecast.uv_ms, dtype=float) * speed_scale
    common = {
        "air_density_kg_m3": parameters.air_density_kg_m3,
        "rotor_radius_m": parameters.rotor_radius_m,
        "frozen_equilibrium_heading_rad": parameters.frozen_equilibrium_heading_rad,
    }
    current_geometry = rotor_geometry_at_declared_nacelle_yaw(
        parameters,
        operating_state.current_nacelle_yaw_relative_platform_rad,
    )
    current_rotor_load = _assemble_rotor_load_at_declared_orientation(
        context="current wind",
        common=common,
        rotor_geometry=current_geometry,
        thrust_coefficient=operating_state.current_thrust_coefficient,
        enu_downwind_air_velocity_mps=current_rotor_plane_wind,
        platform_reference_velocity_platform_mps=(
            operating_state.current_platform_reference_velocity_platform_mps
        ),
        platform_angular_velocity_platform_radps=(
            operating_state.current_platform_angular_velocity_platform_radps
        ),
    )
    # The current external yaw is the only orientation state in this version.
    # Reuse the same geometry at each lead rather than rebuilding an apparent
    # per-lead orientation trajectory.
    future_rotor_loads = tuple(
        _assemble_rotor_load_at_declared_orientation(
            context=f"future forecast lead {index}",
            common=common,
            rotor_geometry=current_geometry,
            thrust_coefficient=future_coefficients[index],
            enu_downwind_air_velocity_mps=future_rotor_plane_winds[index],
            platform_reference_velocity_platform_mps=future_reference_velocities[index],
            platform_angular_velocity_platform_radps=future_angular_velocities[index],
        )
        for index in range(horizon_steps)
    )
    load_forecast = GeneralizedLoadForecast(
        current_generalized_load=current_rotor_load.generalized_load_platform,
        future_generalized_loads=np.asarray(
            [load.generalized_load_platform for load in future_rotor_loads],
            dtype=float,
        ),
        lead_times_s=(np.arange(horizon_steps, dtype=float) + 1.0)
        * float(forecast.sample_period_s),
    )
    return ForecastGeneralizedLoadAssembly(
        load_forecast=load_forecast,
        current_rotor_load=current_rotor_load,
        future_rotor_loads=future_rotor_loads,
        parameters=parameters,
        operating_state=operating_state,
        input_identity=input_identity,
        current_rotor_plane_enu_downwind_mps=current_rotor_plane_wind,
        future_rotor_plane_enu_downwind_mps=future_rotor_plane_winds,
    )


__all__ = [
    "ForecastPhysicalInputIdentity",
    "ForecastGeneralizedLoadAssembly",
    "FuturePlatformKinematicsMode",
    "FutureRotorOrientationMode",
    "ForecastRotorLoadParameters",
    "ForecastRotorOperatingState",
    "rotor_geometry_at_declared_nacelle_yaw",
    "assemble_forecast_generalized_rotor_loads",
]
