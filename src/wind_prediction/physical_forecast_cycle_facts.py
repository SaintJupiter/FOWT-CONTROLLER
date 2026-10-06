"""Assemble one source-bound physical first-interval record.

This is a narrow integration boundary between a real forecast record and the
new physical shadow chain.  It deliberately stops before a control choice:
the returned lifecycle responses are factual replays from one shared actuator
state, not candidates with a score or a selected target.

The caller must still make every aerodynamic, platform and environmental
assumption explicit.  In particular, this helper does not infer a rotor
operating point, interpolate wind records, choose a lead, generate a safety
limit, or write a result back into the legacy controller.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from fowt_platform.ballast_snapshot import BallastModelSnapshot, BallastRuntimeAssembly
from fowt_platform.incremental import IncrementalState

from .execution_rollout import ExecutionRolloutConfig, ExecutionRolloutState
from .first_interval_physical_decision_input import (
    FirstIntervalPhysicalDecisionInput,
    assemble_first_interval_physical_decision_input,
)
from .forecast_evidence import ForecastEvidence, validate_forecast_evidence
from .forecast_physical_load import (
    ForecastRotorLoadParameters,
    ForecastRotorOperatingState,
    assemble_forecast_generalized_rotor_loads,
)
from .forecast_platform_trajectory import (
    ForecastTrajectoryEnvironment,
    rollout_frozen_forecast_platform_trajectory,
)
from .forecast_rhs_ballast_diagnostic import (
    diagnose_forecast_horizon_rhs_ballast_redistributions,
)
from .forecast_rhs_ballast_endpoint_preview import preview_forecast_rhs_ballast_horizon
from .physical_target_lifecycle import trace_physical_target_lifecycles


_STATE_TOLERANCE = 1.0e-10
_MASS_TOLERANCE_KG = 1.0e-8


def _time_bound_to_origin(name: str, value: Any, origin_time: str) -> str:
    """Require one caller-owned state timestamp to equal the forecast origin."""

    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty timestamp label")
    normalized = value.strip()
    if normalized != origin_time:
        raise ValueError(f"{name} must match forecast.origin_time")
    return normalized


def assemble_first_interval_physical_cycle_input(
    *,
    forecast: ForecastEvidence,
    current_enu_downwind_air_velocity_mps: Any,
    current_wind_source: str,
    current_wind_observation_time: str,
    rotor_load_parameters: ForecastRotorLoadParameters,
    rotor_operating_state: ForecastRotorOperatingState,
    platform_snapshot: BallastModelSnapshot,
    initial_platform_state: IncrementalState,
    initial_platform_state_time: str,
    environment: ForecastTrajectoryEnvironment,
    runtime_assembly: BallastRuntimeAssembly,
    execution_state: ExecutionRolloutState,
    execution_config: ExecutionRolloutConfig,
    execution_state_time: str,
) -> FirstIntervalPhysicalDecisionInput:
    """Build one unranked physical record without choosing a control action.

    The forecast origin is the unique time identity for current wind,
    platform initial state and pump execution.  The first forecast lead fixes
    both the RHS endpoint and actuator preview duration.  The function
    intentionally exposes no lead index, endpoint fraction, event probability,
    reliability, score or policy flag.
    """

    if not isinstance(forecast, ForecastEvidence):
        raise TypeError("forecast must be ForecastEvidence")
    validate_forecast_evidence(forecast)
    origin_time = forecast.origin_time
    if origin_time is None:
        raise ValueError("forecast.origin_time is required for physical cycle facts")
    current_wind_time = _time_bound_to_origin(
        "current_wind_observation_time",
        current_wind_observation_time,
        origin_time,
    )
    initial_state_time = _time_bound_to_origin(
        "initial_platform_state_time",
        initial_platform_state_time,
        origin_time,
    )
    execution_time = _time_bound_to_origin(
        "execution_state_time",
        execution_state_time,
        origin_time,
    )

    if not isinstance(platform_snapshot, BallastModelSnapshot):
        raise TypeError("platform_snapshot must be BallastModelSnapshot")
    if not isinstance(initial_platform_state, IncrementalState):
        raise TypeError("initial_platform_state must be IncrementalState")
    if not isinstance(rotor_load_parameters, ForecastRotorLoadParameters):
        raise TypeError("rotor_load_parameters must be ForecastRotorLoadParameters")
    if not isinstance(rotor_operating_state, ForecastRotorOperatingState):
        raise TypeError("rotor_operating_state must be ForecastRotorOperatingState")
    if not isinstance(environment, ForecastTrajectoryEnvironment):
        raise TypeError("environment must be ForecastTrajectoryEnvironment")
    if environment.forecast_origin_time is None or environment.lead_times_s is None:
        raise ValueError(
            "environment must carry the forecast origin and lead-time grid"
        )
    if environment.forecast_origin_time != origin_time:
        raise ValueError("environment forecast_origin_time must match forecast.origin_time")
    expected_leads = np.arange(1, forecast.horizon_steps + 1, dtype=float) * float(
        forecast.sample_period_s
    )
    if not np.allclose(
        environment.lead_times_s,
        expected_leads,
        rtol=0.0,
        atol=1.0e-9,
    ):
        raise ValueError("environment lead_times_s must match forecast lead times")
    if not isinstance(runtime_assembly, BallastRuntimeAssembly):
        raise TypeError("runtime_assembly must be BallastRuntimeAssembly")
    if not isinstance(execution_state, ExecutionRolloutState):
        raise TypeError("execution_state must be ExecutionRolloutState")
    if not isinstance(execution_config, ExecutionRolloutConfig):
        raise TypeError("execution_config must be ExecutionRolloutConfig")
    if not np.allclose(
        rotor_operating_state.current_platform_reference_velocity_platform_mps,
        initial_platform_state.velocity[:3],
        rtol=0.0,
        atol=_STATE_TOLERANCE,
    ) or not np.allclose(
        rotor_operating_state.current_platform_angular_velocity_platform_radps,
        initial_platform_state.velocity[3:],
        rtol=0.0,
        atol=_STATE_TOLERANCE,
    ):
        raise ValueError(
            "current rotor operating-state kinematics must match the initial platform state"
        )
    if not np.allclose(
        execution_state.actual_masses_kg,
        platform_snapshot.actual_tank_masses_kg,
        rtol=0.0,
        atol=_MASS_TOLERANCE_KG,
    ):
        raise ValueError(
            "execution-state actual tank masses must match the platform snapshot"
        )
    if not np.allclose(
        platform_snapshot.tank_capacities_kg,
        float(execution_config.tank_capacity_kg),
        rtol=0.0,
        atol=_MASS_TOLERANCE_KG,
    ):
        raise ValueError(
            "execution configuration capacity must match the platform snapshot"
        )

    load_assembly = assemble_forecast_generalized_rotor_loads(
        forecast=forecast,
        current_enu_downwind_air_velocity_mps=current_enu_downwind_air_velocity_mps,
        current_wind_source=current_wind_source,
        current_wind_observation_time=current_wind_time,
        parameters=rotor_load_parameters,
        operating_state=rotor_operating_state,
    )
    trajectory = rollout_frozen_forecast_platform_trajectory(
        load_assembly=load_assembly,
        platform_snapshot=platform_snapshot,
        initial_state=initial_platform_state,
        initial_state_time=initial_state_time,
        environment=environment,
    )
    horizon_rhs_ballast_diagnostics = (
        diagnose_forecast_horizon_rhs_ballast_redistributions(trajectory=trajectory)
    )
    horizon_endpoint_previews = preview_forecast_rhs_ballast_horizon(
        rhs_diagnostics=horizon_rhs_ballast_diagnostics,
        execution_state=execution_state,
        execution_config=execution_config,
        execution_start_time=execution_time,
    )
    new_track_preview = horizon_endpoint_previews.first_preview
    first_rhs_point = new_track_preview.rhs_diagnostic.rhs_point
    lifecycle_facts = trace_physical_target_lifecycles(
        execution_state=execution_state,
        execution_config=execution_config,
        execution_duration_s=first_rhs_point.lead_time_s,
        execution_start_time=execution_time,
        new_track_preview=new_track_preview,
    )
    return assemble_first_interval_physical_decision_input(
        trajectory=trajectory,
        lifecycle_facts=lifecycle_facts,
        runtime_assembly=runtime_assembly,
        first_rhs_point=first_rhs_point,
        horizon_rhs_ballast_diagnostics=horizon_rhs_ballast_diagnostics,
        horizon_endpoint_previews=horizon_endpoint_previews,
    )


__all__ = [
    "assemble_first_interval_physical_cycle_input",
]
