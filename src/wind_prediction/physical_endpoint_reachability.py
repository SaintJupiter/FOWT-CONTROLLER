"""Execution reachability for one already-physical ballast endpoint.

P5 diagnostics return bounded three-tank endpoints at independent prediction
leads.  This module asks a narrower question for one such endpoint: starting
from the actual pump state, how much of that requested redistribution can be
executed within an explicitly supplied duration not exceeding either that lead
time or the current control block?

It neither chooses a lead, selects an endpoint fraction, ranks actions nor
updates a controller target.  It only returns the actual actuator outcome and
the remaining distance to the supplied physical endpoint.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Any

import numpy as np

from fowt_platform.ballast_endpoint_path import BallastEndpointPathSample

from .execution_rollout import (
    ExecutionRolloutConfig,
    ExecutionRolloutRequest,
    ExecutionRolloutState,
    ExecutionRolloutStep,
    ExecutionTargetOperation,
    simulate_execution_step,
)


_MASS_TOLERANCE_KG = 1e-8


def _positive_scalar(name: str, value: Any) -> float:
    try:
        scalar = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive finite scalar") from exc
    if not math.isfinite(scalar) or scalar <= 0.0:
        raise ValueError(f"{name} must be a positive finite scalar")
    return scalar


def _three(name: str, value: Any) -> np.ndarray:
    try:
        masses = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain three finite values") from exc
    if masses.shape != (3,) or not np.all(np.isfinite(masses)):
        raise ValueError(f"{name} must contain three finite values")
    result = np.array(masses, dtype=float, copy=True)
    result.setflags(write=False)
    return result


def _execution_states_match(
    left: ExecutionRolloutState,
    right: ExecutionRolloutState,
) -> bool:
    """Compare the complete persisted actuator state, not just tank masses."""

    for attribute in (
        "actual_masses_kg",
        "rate_limited_target_kg",
        "primary_target_masses_kg",
        "signed_flow_m3_min",
        "pump_latched",
        "pump_on_elapsed_s",
        "pump_off_elapsed_s",
        "pump_near_target_s",
        "pump_command_rates_m3_min",
        "last_flow_directions",
    ):
        if not np.array_equal(getattr(left, attribute), getattr(right, attribute)):
            return False
    return True


def _execution_steps_match(
    left: ExecutionRolloutStep,
    right: ExecutionRolloutStep,
) -> bool:
    """Require the stored endpoint preview to be reproducible exactly."""

    if (
        left.target_operation is not right.target_operation
        or left.target_slew_reset is not right.target_slew_reset
        or left.target_reached is not right.target_reached
        or left.starts != right.starts
        or left.stops != right.stops
        or left.direction_switches != right.direction_switches
        or not _execution_states_match(left.state, right.state)
    ):
        return False
    for attribute in (
        "requested_target_kg",
        "shaped_target_kg",
        "mass_delta_kg",
        "pump_volume_m3",
        "pump_runtime_s",
        "start_counts",
        "stop_counts",
        "direction_switch_counts",
    ):
        if not np.array_equal(getattr(left, attribute), getattr(right, attribute)):
            return False
    return bool(
        np.isclose(
            left.transferred_volume_m3,
            right.transferred_volume_m3,
            rtol=0.0,
            atol=1.0e-12,
        )
        and np.isclose(
            left.active_time_s,
            right.active_time_s,
            rtol=0.0,
            atol=1.0e-12,
        )
    )


@dataclass(frozen=True)
class PhysicalEndpointReachability:
    """Actual pump result for one independent physical endpoint sample.

    A zero-net-mass endpoint does not force zero instantaneous total ballast
    change.  ``executed_net_mass_delta_kg`` reports that transient change from
    the start state because the three tanks exchange water with the sea
    independently in the present execution model.  The complete pump start
    state is retained because identical tank masses can still evolve
    differently when latch, dwell, ramp, or target-shaping states differ.
    """

    lead_time_s: float
    execution_duration_s: float
    execution_start_state: ExecutionRolloutState
    execution_config: ExecutionRolloutConfig
    endpoint_sample: BallastEndpointPathSample
    execution_step: ExecutionRolloutStep
    remaining_mass_to_endpoint_kg: Any
    executed_net_mass_delta_kg: float

    def __post_init__(self) -> None:
        lead = _positive_scalar("lead_time_s", self.lead_time_s)
        duration = _positive_scalar("execution_duration_s", self.execution_duration_s)
        if duration > lead:
            raise ValueError("execution_duration_s must not exceed lead_time_s")
        if not isinstance(self.execution_start_state, ExecutionRolloutState):
            raise TypeError("execution_start_state must be ExecutionRolloutState")
        if not isinstance(self.execution_config, ExecutionRolloutConfig):
            raise TypeError("execution_config must be ExecutionRolloutConfig")
        if duration > float(self.execution_config.block_duration_s) + 1e-12:
            raise ValueError(
                "execution_duration_s must not exceed "
                "execution_config.block_duration_s"
            )
        if not isinstance(self.endpoint_sample, BallastEndpointPathSample):
            raise TypeError("endpoint_sample must be BallastEndpointPathSample")
        if not isinstance(self.execution_step, ExecutionRolloutStep):
            raise TypeError("execution_step must be ExecutionRolloutStep")
        start = _three(
            "execution_start_state.actual_masses_kg",
            self.execution_start_state.actual_masses_kg,
        )
        endpoint = self.endpoint_sample.hypothetical_tank_masses_kg
        if not np.allclose(
            endpoint - start,
            self.endpoint_sample.mass_delta_from_actual_kg,
            rtol=0.0,
            atol=_MASS_TOLERANCE_KG,
        ):
            raise ValueError(
                "endpoint_sample must be formed from execution_start_state actual masses"
            )
        if self.execution_step.target_operation is not ExecutionTargetOperation.TRACK:
            raise ValueError("execution_step must track the endpoint sample")
        if not np.allclose(
            self.execution_step.requested_target_kg,
            endpoint,
            rtol=0.0,
            atol=_MASS_TOLERANCE_KG,
        ):
            raise ValueError("execution_step must request the endpoint sample")
        remaining = _three(
            "remaining_mass_to_endpoint_kg",
            self.remaining_mass_to_endpoint_kg,
        )
        actual = self.execution_step.state.actual_masses_kg
        if not np.allclose(
            actual - start,
            self.execution_step.mass_delta_kg,
            rtol=0.0,
            atol=_MASS_TOLERANCE_KG,
        ):
            raise ValueError(
                "execution_step mass_delta_kg must start from execution_start_state"
            )
        expected_remaining = self.endpoint_sample.hypothetical_tank_masses_kg - actual
        if not np.allclose(
            remaining,
            expected_remaining,
            rtol=0.0,
            atol=_MASS_TOLERANCE_KG,
        ):
            raise ValueError(
                "remaining_mass_to_endpoint_kg must equal endpoint minus actual execution masses"
            )
        net = float(self.executed_net_mass_delta_kg)
        if not math.isfinite(net) or not math.isclose(
            net,
            float(np.sum(self.execution_step.mass_delta_kg)),
            rel_tol=0.0,
            abs_tol=_MASS_TOLERANCE_KG,
        ):
            raise ValueError(
                "executed_net_mass_delta_kg must equal the summed pump mass change"
            )
        recomputed = simulate_execution_step(
            self.execution_start_state,
            ExecutionRolloutRequest.track(endpoint),
            replace(self.execution_config, block_duration_s=duration),
        )
        if not _execution_steps_match(self.execution_step, recomputed):
            raise ValueError(
                "execution_step must be reproducible from the endpoint, pump state and configuration"
            )
        object.__setattr__(self, "lead_time_s", lead)
        object.__setattr__(self, "execution_duration_s", duration)
        object.__setattr__(self, "remaining_mass_to_endpoint_kg", remaining)
        object.__setattr__(self, "executed_net_mass_delta_kg", net)

    @property
    def endpoint_reached(self) -> bool:
        """Whether the actual pump result reached the supplied endpoint."""

        return bool(self.execution_step.target_reached)


def evaluate_physical_endpoint_reachability(
    *,
    lead_time_s: Any,
    endpoint_sample: BallastEndpointPathSample,
    execution_state: ExecutionRolloutState,
    execution_config: ExecutionRolloutConfig,
    execution_duration_s: Any,
) -> PhysicalEndpointReachability:
    """Run one explicit actuator preview toward a single P5 endpoint sample.

    ``execution_duration_s`` is a caller-selected preview duration and must
    not exceed either the independent endpoint's lead time or the current
    control block.  The existing execution model currently has one shared
    tank-capacity parameter; the endpoint is checked against that same
    capacity before previewing.
    """

    lead = _positive_scalar("lead_time_s", lead_time_s)
    duration = _positive_scalar("execution_duration_s", execution_duration_s)
    if duration > lead:
        raise ValueError("execution_duration_s must not exceed lead_time_s")
    if not isinstance(endpoint_sample, BallastEndpointPathSample):
        raise TypeError("endpoint_sample must be BallastEndpointPathSample")
    if not isinstance(execution_state, ExecutionRolloutState):
        raise TypeError("execution_state must be ExecutionRolloutState")
    if not isinstance(execution_config, ExecutionRolloutConfig):
        raise TypeError("execution_config must be ExecutionRolloutConfig")
    if duration > float(execution_config.block_duration_s) + 1e-12:
        raise ValueError(
            "execution_duration_s must not exceed "
            "execution_config.block_duration_s"
        )

    actual = _three("execution_state.actual_masses_kg", execution_state.actual_masses_kg)
    endpoint = _three(
        "endpoint_sample.hypothetical_tank_masses_kg",
        endpoint_sample.hypothetical_tank_masses_kg,
    )
    if not np.allclose(
        endpoint - actual,
        endpoint_sample.mass_delta_from_actual_kg,
        rtol=0.0,
        atol=_MASS_TOLERANCE_KG,
    ):
        raise ValueError(
            "endpoint_sample must be formed from execution_state.actual_masses_kg"
        )
    capacity = float(execution_config.tank_capacity_kg)
    if np.any(endpoint < 0.0) or np.any(endpoint > capacity):
        raise ValueError(
            "endpoint sample must remain within the shared execution tank capacity"
        )

    execution_step = simulate_execution_step(
        execution_state,
        ExecutionRolloutRequest.track(endpoint),
        replace(execution_config, block_duration_s=duration),
    )
    return PhysicalEndpointReachability(
        lead_time_s=lead,
        execution_duration_s=duration,
        execution_start_state=execution_state,
        execution_config=execution_config,
        endpoint_sample=endpoint_sample,
        execution_step=execution_step,
        remaining_mass_to_endpoint_kg=endpoint - execution_step.state.actual_masses_kg,
        executed_net_mass_delta_kg=float(np.sum(execution_step.mass_delta_kg)),
    )


__all__ = [
    "PhysicalEndpointReachability",
    "evaluate_physical_endpoint_reachability",
]
