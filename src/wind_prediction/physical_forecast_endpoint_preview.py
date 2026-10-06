"""Trace one discrete physical forecast endpoint through pump execution.

This module joins already-computed, single-lead records from the physical
forecast, local ballast diagnostic and actuator-preview layers.  It is an
audit object, not a decision maker: callers explicitly supply the lead and
endpoint fraction, and this module neither ranks alternatives nor changes a
controller target.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from fowt_platform.ballast_endpoint_path import BallastEndpointPathSample
from fowt_platform.ballast_endpoint_path import sample_ballast_endpoint_path
from fowt_platform.forecast_ballast_profile_diagnostic import (
    TimedForecastBallastDemandDiagnostic,
)

from .forecast_physical_load import (
    ForecastGeneralizedLoadAssembly,
    ForecastPhysicalInputIdentity,
)
from .execution_rollout import ExecutionTargetOperation
from .forecast_platform_trajectory import ForecastPlatformTrajectory
from .physical_endpoint_reachability import PhysicalEndpointReachability
from .physical_endpoint_reachability import evaluate_physical_endpoint_reachability


_MASS_TOLERANCE_KG = 1e-8
_LOAD_TOLERANCE_NM = 1e-6


@dataclass(frozen=True)
class PhysicalForecastEndpointPreview:
    """One traceable forecast lead, ballast endpoint and actuator preview.

    The selected lead and endpoint fraction are caller-owned inputs.  The
    record makes their physical consistency inspectable before a later,
    separate controller is allowed to compare them with legacy candidates.
    """

    load_assembly: ForecastGeneralizedLoadAssembly
    lead_index: int
    timed_diagnostic: TimedForecastBallastDemandDiagnostic
    endpoint_sample: BallastEndpointPathSample
    reachability: PhysicalEndpointReachability
    execution_start_time: str | None
    trajectory: ForecastPlatformTrajectory | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.load_assembly, ForecastGeneralizedLoadAssembly):
            raise TypeError("load_assembly must be ForecastGeneralizedLoadAssembly")
        if not isinstance(self.lead_index, int) or isinstance(self.lead_index, bool):
            raise TypeError("lead_index must be an integer")
        if not 0 <= self.lead_index < self.load_assembly.load_forecast.horizon_steps:
            raise IndexError("lead_index is outside the physical forecast horizon")
        if not isinstance(self.timed_diagnostic, TimedForecastBallastDemandDiagnostic):
            raise TypeError("timed_diagnostic must be TimedForecastBallastDemandDiagnostic")
        if not isinstance(self.endpoint_sample, BallastEndpointPathSample):
            raise TypeError("endpoint_sample must be BallastEndpointPathSample")
        if not isinstance(self.reachability, PhysicalEndpointReachability):
            raise TypeError("reachability must be PhysicalEndpointReachability")

        origin_time = self.load_assembly.input_identity.forecast_origin_time
        if origin_time is None:
            if self.execution_start_time is not None:
                raise ValueError(
                    "execution_start_time must be None when forecast origin time is absent"
                )
        else:
            start_time = str(self.execution_start_time).strip()
            if not start_time:
                raise ValueError(
                    "execution_start_time is required when forecast origin time is known"
                )
            if start_time != origin_time:
                raise ValueError(
                    "execution_start_time must match the forecast origin time"
                )
            object.__setattr__(self, "execution_start_time", start_time)

        expected_lead = self.load_assembly.load_forecast.lead_time_at(self.lead_index)
        if not math.isclose(
            self.timed_diagnostic.lead_time_s,
            expected_lead,
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise ValueError("timed_diagnostic lead does not match the selected forecast lead")
        if not math.isclose(
            self.reachability.lead_time_s,
            expected_lead,
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise ValueError("reachability lead does not match the selected forecast lead")
        if not np.allclose(
            self.reachability.endpoint_sample.hypothetical_tank_masses_kg,
            self.endpoint_sample.hypothetical_tank_masses_kg,
            rtol=0.0,
            atol=_MASS_TOLERANCE_KG,
        ):
            raise ValueError("reachability must preview the supplied endpoint_sample")
        if (
            self.reachability.execution_step.target_operation
            is not ExecutionTargetOperation.TRACK
        ):
            raise ValueError("physical endpoint previews must use a track operation")
        if not np.allclose(
            self.reachability.execution_step.requested_target_kg,
            self.endpoint_sample.hypothetical_tank_masses_kg,
            rtol=0.0,
            atol=_MASS_TOLERANCE_KG,
        ):
            raise ValueError(
                "execution request must track the supplied endpoint_sample"
            )

        current_load = self.load_assembly.load_forecast.current_generalized_load
        future_load = self.load_assembly.load_forecast.future_load_at(self.lead_index)
        expected_pitch_roll = (future_load - current_load)[[4, 3]]
        if not np.allclose(
            self.timed_diagnostic.diagnostic.restoring_diagnostic.relative_pitch_roll_load_nm,
            expected_pitch_roll,
            rtol=0.0,
            atol=_LOAD_TOLERANCE_NM,
        ):
            raise ValueError(
                "timed_diagnostic does not represent the selected physical forecast load"
            )

        if not self.endpoint_sample.within_capacity or not np.isclose(
            self.endpoint_sample.net_mass_delta_from_actual_kg,
            0.0,
            rtol=0.0,
            atol=_MASS_TOLERANCE_KG,
        ):
            raise ValueError(
                "endpoint_sample must be a capacity-bounded zero-net redistribution"
            )
        inferred_actual = (
            self.endpoint_sample.hypothetical_tank_masses_kg
            - self.endpoint_sample.mass_delta_from_actual_kg
        )
        if not np.allclose(
            self.timed_diagnostic.actual_tank_masses_kg,
            inferred_actual,
            rtol=0.0,
            atol=_MASS_TOLERANCE_KG,
        ):
            raise ValueError(
                "timed_diagnostic and endpoint_sample must use the same actual tank state"
            )
        if not np.allclose(
            self.reachability.execution_start_state.actual_masses_kg,
            inferred_actual,
            rtol=0.0,
            atol=_MASS_TOLERANCE_KG,
        ):
            raise ValueError(
                "reachability must retain the physical endpoint preview start state"
            )
        if np.any(
            self.endpoint_sample.hypothetical_tank_masses_kg < 0.0
        ) or np.any(
            self.endpoint_sample.hypothetical_tank_masses_kg
            > self.timed_diagnostic.tank_capacities_kg
        ):
            raise ValueError(
                "endpoint_sample must remain within the physical diagnostic tank capacities"
            )
        diagnostic_target = self.timed_diagnostic.diagnostic.allocation.target_tank_masses_kg
        expected_endpoint = self.timed_diagnostic.actual_tank_masses_kg + self.endpoint_sample.fraction * (
            diagnostic_target - self.timed_diagnostic.actual_tank_masses_kg
        )
        if not np.allclose(
            self.endpoint_sample.hypothetical_tank_masses_kg,
            expected_endpoint,
            rtol=0.0,
            atol=_MASS_TOLERANCE_KG,
        ):
            raise ValueError(
                "endpoint_sample is not a path fraction of the selected diagnostic endpoint"
            )
        if self.trajectory is not None:
            if not isinstance(self.trajectory, ForecastPlatformTrajectory):
                raise TypeError("trajectory must be ForecastPlatformTrajectory when provided")
            if self.trajectory.load_assembly is not self.load_assembly:
                raise ValueError(
                    "trajectory must retain the preview load_assembly identity"
                )
            if self.execution_start_time != self.trajectory.initial_state_time:
                raise ValueError(
                    "trajectory origin must match the physical preview execution start time"
                )
            if not math.isclose(
                self.trajectory.steps[self.lead_index].end_time_s,
                expected_lead,
                rel_tol=0.0,
                abs_tol=1e-9,
            ):
                raise ValueError("trajectory lead must match the selected forecast lead")
            if not np.allclose(
                self.trajectory.platform_snapshot.actual_tank_masses_kg,
                inferred_actual,
                rtol=0.0,
                atol=_MASS_TOLERANCE_KG,
            ):
                raise ValueError(
                    "trajectory snapshot must retain the physical preview tank state"
                )
            trajectory_snapshot = self.trajectory.platform_snapshot
            if not np.allclose(
                trajectory_snapshot.tank_capacities_kg,
                self.timed_diagnostic.tank_capacities_kg,
                rtol=0.0,
                atol=_MASS_TOLERANCE_KG,
            ):
                raise ValueError(
                    "trajectory snapshot tank capacities must match the physical diagnostic"
                )
            if not np.allclose(
                trajectory_snapshot.tank_coordinates_m,
                self.timed_diagnostic.tank_coordinates_m,
                rtol=0.0,
                atol=1.0e-12,
            ):
                raise ValueError(
                    "trajectory snapshot tank coordinates must match the physical diagnostic"
                )
            if not math.isclose(
                trajectory_snapshot.gravity_m_s2,
                self.timed_diagnostic.gravity_m_s2,
                rel_tol=0.0,
                abs_tol=1.0e-12,
            ):
                raise ValueError(
                    "trajectory snapshot gravity must match the physical diagnostic"
                )
            if not np.allclose(
                trajectory_snapshot.tank_capacities_kg,
                float(self.reachability.execution_config.tank_capacity_kg),
                rtol=0.0,
                atol=_MASS_TOLERANCE_KG,
            ):
                raise ValueError(
                    "trajectory snapshot tank capacities must match the shared execution capacity"
                )

    @property
    def input_identity(self) -> ForecastPhysicalInputIdentity:
        """Return immutable forecast and current-observation provenance."""

        return self.load_assembly.input_identity

    @property
    def lead_time_s(self) -> float:
        """Return the one discrete future lead represented by this preview."""

        return float(self.timed_diagnostic.lead_time_s)

    @property
    def target_tank_masses_kg(self) -> np.ndarray:
        """Return this explicit endpoint target for common lifecycle handling."""

        return np.array(self.endpoint_sample.hypothetical_tank_masses_kg, copy=True)


def trace_physical_forecast_endpoint(
    *,
    load_assembly: ForecastGeneralizedLoadAssembly,
    lead_index: int,
    timed_diagnostic: TimedForecastBallastDemandDiagnostic,
    endpoint_sample: BallastEndpointPathSample,
    reachability: PhysicalEndpointReachability,
    execution_start_time: str | None,
    trajectory: ForecastPlatformTrajectory | None = None,
) -> PhysicalForecastEndpointPreview:
    """Create one validated trace without choosing or committing an action."""

    return PhysicalForecastEndpointPreview(
        load_assembly=load_assembly,
        lead_index=lead_index,
        timed_diagnostic=timed_diagnostic,
        endpoint_sample=endpoint_sample,
        reachability=reachability,
        execution_start_time=execution_start_time,
        trajectory=trajectory,
    )


def preview_physical_endpoint_fractions(
    *,
    load_assembly: ForecastGeneralizedLoadAssembly,
    lead_index: int,
    timed_diagnostic: TimedForecastBallastDemandDiagnostic,
    actual_tank_masses_kg: object,
    tank_capacities_kg: object,
    fractions: object,
    execution_state: object,
    execution_config: object,
    execution_duration_s: object,
    execution_start_time: str | None,
    trajectory: ForecastPlatformTrajectory | None = None,
) -> tuple[PhysicalForecastEndpointPreview, ...]:
    """Preview explicit fractions of one physical endpoint without selecting one.

    Each result is a *new-target* option for the same discrete forecast lead.
    It intentionally does not represent continuing a pre-existing target or
    releasing one to current water levels; those are separate target-lifecycle
    choices that cannot be silently conflated with a zero endpoint fraction.

    The current execution model has one shared tank-capacity parameter.  The
    physical diagnostic capacities must therefore be equal to that parameter
    before its endpoints can enter this actuator preview.
    """

    from .execution_rollout import ExecutionRolloutConfig, ExecutionRolloutState

    if not isinstance(load_assembly, ForecastGeneralizedLoadAssembly):
        raise TypeError("load_assembly must be ForecastGeneralizedLoadAssembly")
    if not isinstance(timed_diagnostic, TimedForecastBallastDemandDiagnostic):
        raise TypeError("timed_diagnostic must be TimedForecastBallastDemandDiagnostic")
    if not isinstance(execution_state, ExecutionRolloutState):
        raise TypeError("execution_state must be ExecutionRolloutState")
    if not isinstance(execution_config, ExecutionRolloutConfig):
        raise TypeError("execution_config must be ExecutionRolloutConfig")
    origin_time = load_assembly.input_identity.forecast_origin_time
    if origin_time is None:
        if execution_start_time is not None:
            raise ValueError(
                "execution_start_time must be None when forecast origin time is absent"
            )
    else:
        start_time = str(execution_start_time).strip()
        if not start_time:
            raise ValueError(
                "execution_start_time is required when forecast origin time is known"
            )
        if start_time != origin_time:
            raise ValueError(
                "execution_start_time must match the forecast origin time"
            )
    actual = np.asarray(actual_tank_masses_kg, dtype=float)
    capacities = np.asarray(tank_capacities_kg, dtype=float)
    if actual.shape != (3,) or not np.all(np.isfinite(actual)):
        raise ValueError("actual_tank_masses_kg must contain three finite values")
    if capacities.shape != (3,) or not np.all(np.isfinite(capacities)):
        raise ValueError("tank_capacities_kg must contain three finite values")
    if not np.allclose(
        actual,
        execution_state.actual_masses_kg,
        rtol=0.0,
        atol=_MASS_TOLERANCE_KG,
    ):
        raise ValueError(
            "actual_tank_masses_kg must match the actuator preview start state"
        )
    if not np.allclose(
        actual,
        timed_diagnostic.actual_tank_masses_kg,
        rtol=0.0,
        atol=_MASS_TOLERANCE_KG,
    ):
        raise ValueError(
            "actual_tank_masses_kg must match the physical diagnostic start state"
        )
    if not np.allclose(
        capacities,
        timed_diagnostic.tank_capacities_kg,
        rtol=0.0,
        atol=_MASS_TOLERANCE_KG,
    ):
        raise ValueError(
            "tank_capacities_kg must match the physical diagnostic capacities"
        )
    if not np.allclose(
        capacities,
        float(execution_config.tank_capacity_kg),
        rtol=0.0,
        atol=_MASS_TOLERANCE_KG,
    ):
        raise ValueError(
            "the current actuator preview requires one shared tank capacity"
        )

    samples = sample_ballast_endpoint_path(
        actual_tank_masses_kg=actual,
        diagnostic_target_tank_masses_kg=(
            timed_diagnostic.diagnostic.allocation.target_tank_masses_kg
        ),
        tank_capacities_kg=capacities,
        fractions=fractions,
    )
    return tuple(
        trace_physical_forecast_endpoint(
            load_assembly=load_assembly,
            lead_index=lead_index,
            timed_diagnostic=timed_diagnostic,
            endpoint_sample=sample,
            reachability=evaluate_physical_endpoint_reachability(
                lead_time_s=timed_diagnostic.lead_time_s,
                endpoint_sample=sample,
                execution_state=execution_state,
                execution_config=execution_config,
                execution_duration_s=execution_duration_s,
            ),
            execution_start_time=execution_start_time,
            trajectory=trajectory,
        )
        for sample in samples
    )


__all__ = [
    "PhysicalForecastEndpointPreview",
    "preview_physical_endpoint_fractions",
    "trace_physical_forecast_endpoint",
]
