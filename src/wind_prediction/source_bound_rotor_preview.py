"""Source-bound nominal rotor-load preview for one control-cycle origin.

This module composes the existing wind, relative-inflow, nominal rotor-state,
and generalized-load relations.  It does not run a controller, rank ballast
actions, admit forecast events, or advance the platform.  Future platform
kinematics remain a caller-declared approximation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from fowt_platform import (
    IncrementalState,
    NOMINAL_BELOW_RATED_GENERATING,
    NominalBelowRatedRotorOperatingForecast,
    NominalBelowRatedRotorSchedule,
    RotorGeneralizedLoad,
    RotorNormalLoad,
    RotorPerformanceTable,
    assemble_nominal_below_rated_operating_forecast,
    inspect_nominal_below_rated_operating_inputs,
)

from .forecast_evidence import ForecastEvidence, validate_forecast_evidence
from .forecast_physical_load import (
    ForecastGeneralizedLoadAssembly,
    ForecastRotorLoadParameters,
    assemble_forecast_generalized_rotor_loads,
)
from .forecast_rotor_inflow import (
    ForecastRotorKinematics,
    ForecastRotorRelativeInflow,
    assemble_forecast_rotor_relative_inflow,
)
from .run_identity import sha256_json


class RotorPreviewOperatingDomainError(ValueError):
    """Declared operating-domain failure, separate from invalid state/input."""

    def __init__(self, operating_domain: dict[str, Any]):
        self.operating_domain = operating_domain
        self.unsupported_labels = tuple(record["label"] for record in
            operating_domain["records"] if not record["supported"])
        super().__init__("forecast leaves the nominal below-rated rotor operating domain "
                         f"at {self.unsupported_labels}")


@dataclass(frozen=True)
class SourceBoundRotorPreview:
    """One forecast record resolved to nominal rotor loads without policy."""

    forecast: ForecastEvidence
    current_enu_downwind_wind_mps: Any
    relative_inflow: ForecastRotorRelativeInflow
    operating_forecast: NominalBelowRatedRotorOperatingForecast
    load_assembly: ForecastGeneralizedLoadAssembly
    first_interval_rotor_load: RotorGeneralizedLoad
    interval_load_mapping: str

    def __post_init__(self) -> None:
        validate_forecast_evidence(self.forecast)
        current_wind = np.asarray(
            self.current_enu_downwind_wind_mps,
            dtype=float,
        )
        if current_wind.shape != (2,) or not np.all(np.isfinite(current_wind)):
            raise ValueError(
                "current_enu_downwind_wind_mps must have shape (2,) and be finite"
            )
        if not isinstance(self.relative_inflow, ForecastRotorRelativeInflow):
            raise TypeError("relative_inflow must be ForecastRotorRelativeInflow")
        if not isinstance(
            self.operating_forecast,
            NominalBelowRatedRotorOperatingForecast,
        ):
            raise TypeError(
                "operating_forecast must be "
                "NominalBelowRatedRotorOperatingForecast"
            )
        if not isinstance(self.load_assembly, ForecastGeneralizedLoadAssembly):
            raise TypeError("load_assembly must be ForecastGeneralizedLoadAssembly")
        if not isinstance(self.first_interval_rotor_load, RotorGeneralizedLoad):
            raise TypeError(
                "first_interval_rotor_load must be RotorGeneralizedLoad"
            )
        mapping = str(self.interval_load_mapping).strip()
        if mapping != "piecewise_linear_endpoint_average":
            raise ValueError(
                "interval_load_mapping must be "
                "'piecewise_linear_endpoint_average'"
            )
        expected = 0.5 * (
            self.load_assembly.current_rotor_load.generalized_load_platform
            + self.load_assembly.future_rotor_loads[0].generalized_load_platform
        )
        if not np.allclose(
            self.first_interval_rotor_load.generalized_load_platform,
            expected,
            rtol=0.0,
            atol=1.0e-8,
        ):
            raise ValueError(
                "first_interval_rotor_load must equal the first endpoint average"
            )
        stored_wind = np.array(current_wind, dtype=float, copy=True)
        stored_wind.setflags(write=False)
        object.__setattr__(
            self,
            "current_enu_downwind_wind_mps",
            stored_wind,
        )
        object.__setattr__(self, "interval_load_mapping", mapping)

    def source_record_payload(self) -> dict[str, Any]:
        """Return the raw forecast and current observation before load conversion."""

        forecast = self.forecast
        return {
            "forecast": {
                "source": forecast.source,
                "model_version": forecast.model_version,
                "origin_time": forecast.origin_time,
                "sample_period_s": float(forecast.sample_period_s),
                "uv_ms": np.asarray(forecast.uv_ms, dtype=float).tolist(),
                "event_probs": {
                    str(key): float(value)
                    for key, value in sorted(forecast.event_probs.items())
                },
                "lead_reliability": np.asarray(
                    forecast.lead_reliability,
                    dtype=float,
                ).tolist(),
                "provides_future_preview": bool(
                    forecast.provides_future_preview
                ),
                "metadata": dict(forecast.metadata),
            },
            "current_observation": {
                "enu_downwind_wind_mps": np.asarray(
                    self.current_enu_downwind_wind_mps,
                    dtype=float,
                ).tolist(),
                "source": self.relative_inflow.current_wind_source,
                "observation_time": (
                    self.relative_inflow.current_wind_observation_time
                ),
            },
        }

    @property
    def source_record_sha256(self) -> str:
        """Hash source facts without including platform-dependent loads."""

        return sha256_json(self.source_record_payload())


def endpoint_average_rotor_load(
    current: RotorGeneralizedLoad,
    future: RotorGeneralizedLoad,
) -> RotorGeneralizedLoad:
    return RotorGeneralizedLoad(
        relative_air_velocity_platform_mps=0.5
        * (
            current.relative_air_velocity_platform_mps
            + future.relative_air_velocity_platform_mps
        ),
        normal_load=RotorNormalLoad(
            thrust_n=0.5 * (current.normal_load.thrust_n + future.normal_load.thrust_n),
            force_platform_n=0.5
            * (
                current.normal_load.force_platform_n
                + future.normal_load.force_platform_n
            ),
        ),
        generalized_load_platform=0.5
        * (
            current.generalized_load_platform
            + future.generalized_load_platform
        ),
    )


def assemble_source_bound_nominal_rotor_preview(
    *,
    forecast: ForecastEvidence,
    current_enu_downwind_wind_mps: Any,
    current_wind_source: str,
    current_wind_observation_time: str,
    platform_state: IncrementalState,
    kinematics: ForecastRotorKinematics,
    parameters: ForecastRotorLoadParameters,
    schedule: NominalBelowRatedRotorSchedule,
    performance_table: RotorPerformanceTable,
    operating_mode_source: str,
) -> SourceBoundRotorPreview:
    """Resolve one source-bound forecast to a nominal below-rated load preview.

    The current platform velocity must be the same state supplied to the
    controller at this origin.  Future velocities and rotor orientation are
    carried by ``kinematics`` and therefore remain explicit approximations.
    """

    if not isinstance(platform_state, IncrementalState):
        raise TypeError("platform_state must be IncrementalState")
    if not isinstance(kinematics, ForecastRotorKinematics):
        raise TypeError("kinematics must be ForecastRotorKinematics")
    if not np.allclose(
        kinematics.current_platform_reference_velocity_platform_mps,
        platform_state.velocity[:3],
        rtol=0.0,
        atol=1.0e-12,
    ) or not np.allclose(
        kinematics.current_platform_angular_velocity_platform_radps,
        platform_state.velocity[3:],
        rtol=0.0,
        atol=1.0e-12,
    ):
        raise ValueError(
            "kinematics current velocities must match platform_state.velocity"
        )

    relative_inflow = assemble_forecast_rotor_relative_inflow(
        forecast=forecast,
        current_enu_downwind_air_velocity_mps=current_enu_downwind_wind_mps,
        current_wind_source=current_wind_source,
        current_wind_observation_time=current_wind_observation_time,
        parameters=parameters,
        kinematics=kinematics,
    )
    operating_domain = inspect_nominal_below_rated_operating_inputs(
        schedule=schedule,
        performance_table=performance_table,
        rotor_radius_m=parameters.rotor_radius_m,
        current_normal_inflow_speed_mps=(
            relative_inflow.current_normal_relative_inflow_mps
        ),
        future_normal_inflow_speeds_mps=(
            relative_inflow.future_normal_relative_inflows_mps
        ),
    )
    if not operating_domain["supported"]:
        raise RotorPreviewOperatingDomainError(operating_domain)
    operating_forecast = assemble_nominal_below_rated_operating_forecast(
        schedule=schedule,
        performance_table=performance_table,
        rotor_radius_m=parameters.rotor_radius_m,
        current_normal_inflow_speed_mps=(
            relative_inflow.current_normal_relative_inflow_mps
        ),
        future_normal_inflow_speeds_mps=(
            relative_inflow.future_normal_relative_inflows_mps
        ),
        operating_mode=NOMINAL_BELOW_RATED_GENERATING,
        operating_mode_source=operating_mode_source,
    )
    operating_state = kinematics.to_operating_state(
        current_thrust_coefficient=operating_forecast.current_thrust_coefficient,
        future_thrust_coefficients=operating_forecast.future_thrust_coefficients,
    )
    load_assembly = assemble_forecast_generalized_rotor_loads(
        forecast=forecast,
        current_enu_downwind_air_velocity_mps=current_enu_downwind_wind_mps,
        current_wind_source=current_wind_source,
        current_wind_observation_time=current_wind_observation_time,
        parameters=parameters,
        operating_state=operating_state,
    )
    first_interval_load = endpoint_average_rotor_load(
        load_assembly.current_rotor_load,
        load_assembly.future_rotor_loads[0],
    )
    return SourceBoundRotorPreview(
        forecast=forecast,
        current_enu_downwind_wind_mps=current_enu_downwind_wind_mps,
        relative_inflow=relative_inflow,
        operating_forecast=operating_forecast,
        load_assembly=load_assembly,
        first_interval_rotor_load=first_interval_load,
        interval_load_mapping="piecewise_linear_endpoint_average",
    )


__all__ = [
    "RotorPreviewOperatingDomainError",
    "SourceBoundRotorPreview",
    "assemble_source_bound_nominal_rotor_preview",
    "endpoint_average_rotor_load",
]
