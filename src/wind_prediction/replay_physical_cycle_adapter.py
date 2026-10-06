"""Bind one replay-origin forecast record to the physical first-interval path.

This adapter only joins two existing boundaries: the replay source that owns a
model forecast and its same-origin observed wind, and the source-bound
physical first-interval assembler.  It deliberately creates no control
candidate, policy flag, score, inferred operating point, or hidden fallback.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .controller_replay_adapter import ReplayForecastEvidenceSource
from .execution_rollout import ExecutionRolloutConfig, ExecutionRolloutState
from .first_interval_physical_decision_input import FirstIntervalPhysicalDecisionInput
from .forecast_evidence import ForecastEvidence, validate_forecast_evidence
from .forecast_physical_load import (
    ForecastRotorLoadParameters,
    ForecastRotorOperatingState,
)
from .forecast_platform_trajectory import ForecastTrajectoryEnvironment
from .physical_forecast_cycle_facts import assemble_first_interval_physical_cycle_input
from .physical_rolling_control_chain import PhysicalRollingCycleInput

from fowt_platform.ballast_snapshot import BallastModelSnapshot, BallastRuntimeAssembly
from fowt_platform.incremental import IncrementalState


_REPLAY_ORIGIN_CURRENT_WIND_SOURCE = "replay_origin_observation"


@dataclass(frozen=True)
class ReplayPhysicalCycleEvidence:
    """One replay forecast and the physical input assembled from that forecast.

    The record preserves the single source call that supplied both objects so
    the caller can bind forecast admission without running prediction again.
    It does not itself admit a forecast or make a control decision.
    """

    forecast: ForecastEvidence
    decision_input: FirstIntervalPhysicalDecisionInput

    def __post_init__(self) -> None:
        if not isinstance(self.forecast, ForecastEvidence):
            raise TypeError("forecast must be ForecastEvidence")
        validate_forecast_evidence(self.forecast)
        if not isinstance(self.decision_input, FirstIntervalPhysicalDecisionInput):
            raise TypeError("decision_input must be FirstIntervalPhysicalDecisionInput")
        identity = self.decision_input.trajectory.load_assembly.input_identity
        if self.forecast.origin_time != identity.forecast_origin_time:
            raise ValueError("forecast origin must match the physical input origin")
        if self.forecast.source != identity.forecast_source:
            raise ValueError("forecast source must match the physical input source")
        if self.forecast.model_version != identity.forecast_model_version:
            raise ValueError("forecast model version must match the physical input")


def assemble_resolved_physical_rolling_cycle_input(
    *,
    forecast: ForecastEvidence,
    current_enu_downwind_air_velocity_mps: Any,
    current_wind_source: str,
    current_wind_observation_time: str,
    start_time_s: float,
    current_thrust_coefficient: float,
    future_thrust_coefficients: Any,
    thrust_coefficient_source: str,
    current_nacelle_yaw_relative_platform_rad: float,
    current_nacelle_yaw_source: str,
    forecast_admitted: bool,
    forecast_admission_basis: str,
    current_wave_load: Any,
    current_other_load: Any,
    future_wave_interval_loads: Any,
    future_other_interval_loads: Any,
    nonrotor_load_provenance: str,
) -> PhysicalRollingCycleInput:
    """Assemble one rolling input from caller-resolved physical inputs.

    The caller owns the relationship between the forecast, current wind and
    resolved rotor coefficients.  This helper deliberately does not claim to
    establish or validate that relationship.  It only builds the rolling
    input after a source adapter or other upstream code has done so.
    """

    if not isinstance(forecast, ForecastEvidence):
        raise TypeError("forecast must be ForecastEvidence")
    validate_forecast_evidence(forecast)
    origin_time = forecast.origin_time
    if origin_time is None:
        raise ValueError("source-bound forecast must carry an origin_time")
    horizon = int(forecast.horizon_steps)
    return PhysicalRollingCycleInput(
        start_time_s=start_time_s,
        forecast=forecast,
        current_enu_downwind_air_velocity_mps=current_enu_downwind_air_velocity_mps,
        current_wind_source=current_wind_source,
        current_wind_observation_time=current_wind_observation_time,
        current_thrust_coefficient=current_thrust_coefficient,
        current_nacelle_yaw_relative_platform_rad=(
            current_nacelle_yaw_relative_platform_rad
        ),
        current_nacelle_yaw_source=current_nacelle_yaw_source,
        future_thrust_coefficients=future_thrust_coefficients,
        thrust_coefficient_source=thrust_coefficient_source,
        forecast_admitted=forecast_admitted,
        forecast_admission_basis=forecast_admission_basis,
        environment=ForecastTrajectoryEnvironment(
            current_wave_load=current_wave_load,
            current_other_load=current_other_load,
            wave_interval_loads=future_wave_interval_loads,
            other_interval_loads=future_other_interval_loads,
            forecast_origin_time=origin_time,
            lead_times_s=(
                np.arange(1, horizon + 1, dtype=float)
                * float(forecast.sample_period_s)
            ),
            provenance=nonrotor_load_provenance,
        ),
    )


def assemble_replay_physical_rolling_cycle_input(
    *,
    forecast_source: ReplayForecastEvidenceSource,
    current_time_s: float,
    start_time_s: float,
    current_thrust_coefficient: float,
    future_thrust_coefficients: Any,
    thrust_coefficient_source: str,
    current_nacelle_yaw_relative_platform_rad: float,
    current_nacelle_yaw_source: str,
    forecast_admitted: bool,
    forecast_admission_basis: str,
    current_wave_load: Any,
    current_other_load: Any,
    future_wave_interval_loads: Any,
    future_other_interval_loads: Any,
    nonrotor_load_provenance: str,
) -> PhysicalRollingCycleInput | None:
    """Build one rolling-chain input from a single replay-origin source call.

    The replay source owns both the model forecast and the observed wind at
    its history end.  This adapter binds them into one rolling input and
    derives the environmental lead grid from the returned forecast.  Missing
    replay evidence remains missing; no persistence or current-only fallback
    is introduced here.

    Non-rotor loads and rotor operating-state assumptions are deliberately
    supplied by the caller.  The adapter only fixes their forecast-time
    identity and does not interpret reliability, event probabilities or a
    control action.
    """

    if not isinstance(forecast_source, ReplayForecastEvidenceSource):
        raise TypeError("forecast_source must be ReplayForecastEvidenceSource")
    source_bound = forecast_source.source_bound_forecast_and_current_enu_wind(
        current_time_s
    )
    if source_bound is None:
        return None
    forecast, current_enu_downwind_air_velocity_mps = source_bound
    origin_time = forecast.origin_time
    if origin_time is None:
        raise ValueError("source-bound forecast must carry an origin_time")
    return assemble_resolved_physical_rolling_cycle_input(
        start_time_s=start_time_s,
        forecast=forecast,
        current_enu_downwind_air_velocity_mps=current_enu_downwind_air_velocity_mps,
        current_wind_source=_REPLAY_ORIGIN_CURRENT_WIND_SOURCE,
        current_wind_observation_time=origin_time,
        current_thrust_coefficient=current_thrust_coefficient,
        current_nacelle_yaw_relative_platform_rad=(
            current_nacelle_yaw_relative_platform_rad
        ),
        current_nacelle_yaw_source=current_nacelle_yaw_source,
        future_thrust_coefficients=future_thrust_coefficients,
        thrust_coefficient_source=thrust_coefficient_source,
        forecast_admitted=forecast_admitted,
        forecast_admission_basis=forecast_admission_basis,
        current_wave_load=current_wave_load,
        current_other_load=current_other_load,
        future_wave_interval_loads=future_wave_interval_loads,
        future_other_interval_loads=future_other_interval_loads,
        nonrotor_load_provenance=nonrotor_load_provenance,
    )


def assemble_replay_physical_cycle_evidence(
    *,
    forecast_source: ReplayForecastEvidenceSource,
    current_time_s: float,
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
) -> ReplayPhysicalCycleEvidence | None:
    """Bind one replay forecast to the physical input assembled from it.

    A missing replay sample remains missing.  In particular, this function
    never uses future replay labels, a caller-supplied substitute wind, or a
    persistence fallback.  Platform and actuator timestamps remain explicit
    because the source cannot establish their time identities on its own.
    """

    if not isinstance(forecast_source, ReplayForecastEvidenceSource):
        raise TypeError("forecast_source must be ReplayForecastEvidenceSource")

    source_bound = forecast_source.source_bound_forecast_and_current_enu_wind(
        current_time_s
    )
    if source_bound is None:
        return None
    forecast, current_enu_downwind_air_velocity_mps = source_bound
    origin_time = forecast.origin_time
    if origin_time is None:
        raise ValueError("source-bound forecast must carry an origin_time")

    decision_input = assemble_first_interval_physical_cycle_input(
        forecast=forecast,
        current_enu_downwind_air_velocity_mps=current_enu_downwind_air_velocity_mps,
        current_wind_source=_REPLAY_ORIGIN_CURRENT_WIND_SOURCE,
        current_wind_observation_time=origin_time,
        rotor_load_parameters=rotor_load_parameters,
        rotor_operating_state=rotor_operating_state,
        platform_snapshot=platform_snapshot,
        initial_platform_state=initial_platform_state,
        initial_platform_state_time=initial_platform_state_time,
        environment=environment,
        runtime_assembly=runtime_assembly,
        execution_state=execution_state,
        execution_config=execution_config,
        execution_state_time=execution_state_time,
    )
    return ReplayPhysicalCycleEvidence(
        forecast=forecast,
        decision_input=decision_input,
    )


def assemble_replay_first_interval_physical_cycle_input(
    *,
    forecast_source: ReplayForecastEvidenceSource,
    current_time_s: float,
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
) -> FirstIntervalPhysicalDecisionInput | None:
    """Return only the physical input for compatibility with older callers."""

    evidence = assemble_replay_physical_cycle_evidence(
        forecast_source=forecast_source,
        current_time_s=current_time_s,
        rotor_load_parameters=rotor_load_parameters,
        rotor_operating_state=rotor_operating_state,
        platform_snapshot=platform_snapshot,
        initial_platform_state=initial_platform_state,
        initial_platform_state_time=initial_platform_state_time,
        environment=environment,
        runtime_assembly=runtime_assembly,
        execution_state=execution_state,
        execution_config=execution_config,
        execution_state_time=execution_state_time,
    )
    return None if evidence is None else evidence.decision_input


__all__ = [
    "ReplayPhysicalCycleEvidence",
    "assemble_resolved_physical_rolling_cycle_input",
    "assemble_replay_physical_rolling_cycle_input",
    "assemble_replay_first_interval_physical_cycle_input",
    "assemble_replay_physical_cycle_evidence",
]
