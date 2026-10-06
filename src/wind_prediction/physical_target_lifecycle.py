"""Factual target-lifecycle traces for the physical control path.

This module gives three existing target meanings a common, inspectable form:
tracking one explicitly supplied physical endpoint, continuing the currently
stored primary target, and releasing the target to the current tank masses.
It does not choose between them, score their outcomes, infer safety, or write
anything back to the controller.  In particular, a zero physical-endpoint
fraction remains a new ``TRACK`` target and is never reinterpreted as either
continuation or release.
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import replace
from enum import Enum
import math
from typing import Any

import numpy as np

from .execution_rollout import (
    ExecutionRolloutConfig,
    ExecutionRolloutRequest,
    ExecutionRolloutState,
    ExecutionRolloutStep,
    ExecutionTargetOperation,
    simulate_execution_step,
)
from .physical_endpoint_preview_binding import (
    PhysicalEndpointPreview,
    is_physical_endpoint_preview,
)


_MASS_TOLERANCE_KG = 1e-8


class PhysicalTargetLifecycle(str, Enum):
    """Decision-layer target meanings before any evaluation is introduced."""

    NEW_TRACK = "new_track"
    CONTINUE_EXISTING = "continue_existing"
    RELEASE_TO_CURRENT = "release_to_current"


def _positive_duration(value: Any) -> float:
    try:
        duration = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("execution_duration_s must be a positive finite scalar") from exc
    if not math.isfinite(duration) or duration <= 0.0:
        raise ValueError("execution_duration_s must be a positive finite scalar")
    return duration


def _execution_start_time(value: Any) -> str:
    if not isinstance(value, str):
        raise ValueError("execution_start_time must be a non-empty string")
    start_time = value.strip()
    if not start_time:
        raise ValueError("execution_start_time must be a non-empty string")
    return start_time


def _same_execution_state(
    left: ExecutionRolloutState,
    right: ExecutionRolloutState,
) -> bool:
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


@dataclass(frozen=True)
class PhysicalTargetLifecycleTrace:
    """One unranked target operation and its actuator outcome.

    ``source_preview`` is present only for a new physical endpoint.  Existing
    target continuation and release deliberately do not manufacture forecast
    endpoint records, because they have their own lifecycle meanings.
    """

    lifecycle: PhysicalTargetLifecycle
    execution_start_state: ExecutionRolloutState
    execution_config: ExecutionRolloutConfig
    execution_duration_s: float
    execution_start_time: str
    execution_request: ExecutionRolloutRequest
    execution_step: ExecutionRolloutStep
    source_preview: PhysicalEndpointPreview | None = None

    def __post_init__(self) -> None:
        lifecycle = PhysicalTargetLifecycle(self.lifecycle)
        object.__setattr__(self, "lifecycle", lifecycle)
        duration = _positive_duration(self.execution_duration_s)
        object.__setattr__(self, "execution_duration_s", duration)
        execution_start_time = _execution_start_time(self.execution_start_time)
        object.__setattr__(self, "execution_start_time", execution_start_time)
        if not isinstance(self.execution_start_state, ExecutionRolloutState):
            raise TypeError("execution_start_state must be ExecutionRolloutState")
        if not isinstance(self.execution_config, ExecutionRolloutConfig):
            raise TypeError("execution_config must be ExecutionRolloutConfig")
        if duration > float(self.execution_config.block_duration_s) + 1e-12:
            raise ValueError(
                "execution_duration_s must not exceed "
                "execution_config.block_duration_s"
            )
        if not isinstance(self.execution_request, ExecutionRolloutRequest):
            raise TypeError("execution_request must be ExecutionRolloutRequest")
        if not isinstance(self.execution_step, ExecutionRolloutStep):
            raise TypeError("execution_step must be ExecutionRolloutStep")
        if self.execution_request.operation is not self.execution_step.target_operation:
            raise ValueError("execution request and actuator result must use one operation")
        if not np.allclose(
            self.execution_step.state.actual_masses_kg
            - self.execution_start_state.actual_masses_kg,
            self.execution_step.mass_delta_kg,
            rtol=0.0,
            atol=_MASS_TOLERANCE_KG,
        ):
            raise ValueError("execution_step must start from execution_start_state")
        if self.execution_request.operation is ExecutionTargetOperation.TRACK:
            if not np.allclose(
                self.execution_step.requested_target_kg,
                self.execution_request.target_masses_kg,
                rtol=0.0,
                atol=_MASS_TOLERANCE_KG,
            ):
                raise ValueError("track actuator outcome must retain the requested target")

        if lifecycle is PhysicalTargetLifecycle.NEW_TRACK:
            preview = self.source_preview
            if not is_physical_endpoint_preview(preview):
                raise ValueError("new_track requires one physical endpoint preview")
            if self.execution_request.operation is not ExecutionTargetOperation.TRACK:
                raise ValueError("new_track must use a track request")
            if not _same_execution_state(
                self.execution_start_state,
                preview.reachability.execution_start_state,
            ):
                raise ValueError("new_track must retain the preview pump start state")
            if self.execution_config != preview.reachability.execution_config:
                raise ValueError("new_track must retain the preview execution configuration")
            if execution_start_time != preview.execution_start_time:
                raise ValueError("new_track must retain the preview execution start time")
            if not math.isclose(
                duration,
                preview.reachability.execution_duration_s,
                rel_tol=0.0,
                abs_tol=1e-9,
            ):
                raise ValueError("new_track must retain the preview execution duration")
            if self.execution_step is not preview.reachability.execution_step:
                raise ValueError("new_track must reuse the preview actuator outcome")
            endpoint = preview.target_tank_masses_kg
            if not np.allclose(
                self.execution_request.target_masses_kg,
                endpoint,
                rtol=0.0,
                atol=_MASS_TOLERANCE_KG,
            ):
                raise ValueError("new_track request must match the physical endpoint")
            if not np.allclose(
                self.execution_step.requested_target_kg,
                endpoint,
                rtol=0.0,
                atol=_MASS_TOLERANCE_KG,
            ):
                raise ValueError("new_track actuator outcome must match the physical endpoint")
            return

        if self.source_preview is not None:
            raise ValueError("only new_track may carry a physical endpoint preview")

        if lifecycle is PhysicalTargetLifecycle.CONTINUE_EXISTING:
            if self.execution_request.operation is not ExecutionTargetOperation.TRACK:
                raise ValueError("continue_existing must use a track request")
            if not np.allclose(
                self.execution_request.target_masses_kg,
                self.execution_start_state.primary_target_masses_kg,
                rtol=0.0,
                atol=_MASS_TOLERANCE_KG,
            ):
                raise ValueError(
                    "continue_existing must track the stored primary target"
                )
            return

        if lifecycle is PhysicalTargetLifecycle.RELEASE_TO_CURRENT:
            if (
                self.execution_request.operation
                is not ExecutionTargetOperation.RELEASE_TO_CURRENT
            ):
                raise ValueError("release_to_current must use a release request")
            if not np.allclose(
                self.execution_step.requested_target_kg,
                self.execution_start_state.actual_masses_kg,
                rtol=0.0,
                atol=_MASS_TOLERANCE_KG,
            ):
                raise ValueError(
                    "release_to_current must derive its target from current masses"
                )
            return

        raise AssertionError(f"unhandled target lifecycle {lifecycle!r}")

    @property
    def requested_target_masses_kg(self) -> np.ndarray:
        """Return the target given to the actuator for this one trace."""

        return np.array(self.execution_step.requested_target_kg, copy=True)

    @property
    def actual_final_masses_kg(self) -> np.ndarray:
        """Return the actual tank masses after this one actuator preview."""

        return np.array(self.execution_step.state.actual_masses_kg, copy=True)

    @property
    def remaining_mass_to_target_kg(self) -> np.ndarray:
        """Return the actuator's remaining mass error after the preview."""

        return self.requested_target_masses_kg - self.actual_final_masses_kg


@dataclass(frozen=True)
class PhysicalTargetLifecycleFacts:
    """Unranked lifecycle facts from one shared pump state and time interval."""

    continue_existing: PhysicalTargetLifecycleTrace
    release_to_current: PhysicalTargetLifecycleTrace
    execution_start_time: str
    new_track: PhysicalTargetLifecycleTrace | None = None

    def __post_init__(self) -> None:
        continuation = self.continue_existing
        release = self.release_to_current
        if not isinstance(continuation, PhysicalTargetLifecycleTrace):
            raise TypeError("continue_existing must be PhysicalTargetLifecycleTrace")
        if not isinstance(release, PhysicalTargetLifecycleTrace):
            raise TypeError("release_to_current must be PhysicalTargetLifecycleTrace")
        if self.new_track is not None and not isinstance(
            self.new_track,
            PhysicalTargetLifecycleTrace,
        ):
            raise TypeError("new_track must be PhysicalTargetLifecycleTrace when provided")
        execution_start_time = _execution_start_time(self.execution_start_time)
        if (
            continuation.lifecycle is not PhysicalTargetLifecycle.CONTINUE_EXISTING
            or release.lifecycle is not PhysicalTargetLifecycle.RELEASE_TO_CURRENT
        ):
            raise ValueError("lifecycle facts require continuation and release traces")
        if (
            self.new_track is not None
            and self.new_track.lifecycle is not PhysicalTargetLifecycle.NEW_TRACK
        ):
            raise ValueError("new_track fact must use the new_track lifecycle")
        if continuation.execution_start_time != execution_start_time:
            raise ValueError("all lifecycle facts must share one execution start time")
        for trace in (release, self.new_track):
            if trace is None:
                continue
            if not _same_execution_state(
                continuation.execution_start_state,
                trace.execution_start_state,
            ):
                raise ValueError("all lifecycle facts must share one pump start state")
            if continuation.execution_config != trace.execution_config:
                raise ValueError("all lifecycle facts must share one execution configuration")
            if not math.isclose(
                continuation.execution_duration_s,
                trace.execution_duration_s,
                rel_tol=0.0,
                abs_tol=1e-9,
            ):
                raise ValueError("all lifecycle facts must share one execution duration")
            if trace.execution_start_time != execution_start_time:
                raise ValueError("all lifecycle facts must share one execution start time")
        object.__setattr__(self, "execution_start_time", execution_start_time)


def _trace_new_track_from_preview(
    *,
    execution_state: ExecutionRolloutState,
    execution_config: ExecutionRolloutConfig,
    execution_duration_s: float,
    execution_start_time: str,
    endpoint_preview: PhysicalEndpointPreview,
) -> PhysicalTargetLifecycleTrace:
    """Build one source-bound new-target trace from an already checked preview."""

    if not is_physical_endpoint_preview(endpoint_preview):
        raise TypeError("endpoint_preview must be a supported physical endpoint preview")
    request = ExecutionRolloutRequest.track(endpoint_preview.target_tank_masses_kg)
    return PhysicalTargetLifecycleTrace(
        lifecycle=PhysicalTargetLifecycle.NEW_TRACK,
        execution_start_state=execution_state,
        execution_config=execution_config,
        execution_duration_s=execution_duration_s,
        execution_start_time=execution_start_time,
        execution_request=request,
        execution_step=endpoint_preview.reachability.execution_step,
        source_preview=endpoint_preview,
    )


def trace_new_physical_target_lifecycle(
    *,
    execution_state: ExecutionRolloutState,
    execution_config: ExecutionRolloutConfig,
    execution_duration_s: Any,
    execution_start_time: Any,
    endpoint_preview: PhysicalEndpointPreview,
) -> PhysicalTargetLifecycleTrace:
    """Trace one supplied physical endpoint without creating other lifecycles.

    This is the narrow counterpart of :func:`trace_physical_target_lifecycles`
    for a caller that already has a concrete source-bound endpoint.  It retains
    the preview's pump result rather than simulating a second, approximate
    execution.  The function does not compare endpoints or decide whether this
    target should be used.
    """

    if not isinstance(execution_state, ExecutionRolloutState):
        raise TypeError("execution_state must be ExecutionRolloutState")
    if not isinstance(execution_config, ExecutionRolloutConfig):
        raise TypeError("execution_config must be ExecutionRolloutConfig")
    duration = _positive_duration(execution_duration_s)
    if duration > float(execution_config.block_duration_s) + 1e-12:
        raise ValueError(
            "execution_duration_s must not exceed "
            "execution_config.block_duration_s"
        )
    start_time = _execution_start_time(execution_start_time)
    return _trace_new_track_from_preview(
        execution_state=execution_state,
        execution_config=execution_config,
        execution_duration_s=duration,
        execution_start_time=start_time,
        endpoint_preview=endpoint_preview,
    )


def trace_physical_target_lifecycles(
    *,
    execution_state: ExecutionRolloutState,
    execution_config: ExecutionRolloutConfig,
    execution_duration_s: Any,
    execution_start_time: Any,
    new_track_preview: PhysicalEndpointPreview | None = None,
) -> PhysicalTargetLifecycleFacts:
    """Trace lifecycle operations from one state without choosing among them.

    ``new_track_preview`` is optional because physical endpoint construction is
    not required to inspect the pre-existing target lifecycle.  When supplied,
    it must have been generated from exactly the same full pump state,
    execution configuration, duration and execution start time.  The function
    reuses its already computed actuator outcome rather than simulating the
    endpoint again.
    """

    if not isinstance(execution_state, ExecutionRolloutState):
        raise TypeError("execution_state must be ExecutionRolloutState")
    if not isinstance(execution_config, ExecutionRolloutConfig):
        raise TypeError("execution_config must be ExecutionRolloutConfig")
    duration = _positive_duration(execution_duration_s)
    if duration > float(execution_config.block_duration_s) + 1e-12:
        raise ValueError(
            "execution_duration_s must not exceed "
            "execution_config.block_duration_s"
        )
    start_time = _execution_start_time(execution_start_time)
    local_config = replace(execution_config, block_duration_s=duration)

    continuation_request = ExecutionRolloutRequest.track(
        execution_state.primary_target_masses_kg
    )
    continuation = PhysicalTargetLifecycleTrace(
        lifecycle=PhysicalTargetLifecycle.CONTINUE_EXISTING,
        execution_start_state=execution_state,
        execution_config=execution_config,
        execution_duration_s=duration,
        execution_start_time=start_time,
        execution_request=continuation_request,
        execution_step=simulate_execution_step(
            execution_state,
            continuation_request,
            local_config,
        ),
    )

    release_request = ExecutionRolloutRequest.release_to_current()
    release = PhysicalTargetLifecycleTrace(
        lifecycle=PhysicalTargetLifecycle.RELEASE_TO_CURRENT,
        execution_start_state=execution_state,
        execution_config=execution_config,
        execution_duration_s=duration,
        execution_start_time=start_time,
        execution_request=release_request,
        execution_step=simulate_execution_step(
            execution_state,
            release_request,
            local_config,
        ),
    )

    new_track: PhysicalTargetLifecycleTrace | None = None
    if new_track_preview is not None:
        new_track = _trace_new_track_from_preview(
            execution_state=execution_state,
            execution_config=execution_config,
            execution_duration_s=duration,
            execution_start_time=start_time,
            endpoint_preview=new_track_preview,
        )

    return PhysicalTargetLifecycleFacts(
        continue_existing=continuation,
        release_to_current=release,
        execution_start_time=start_time,
        new_track=new_track,
    )


__all__ = [
    "PhysicalTargetLifecycle",
    "PhysicalTargetLifecycleFacts",
    "PhysicalTargetLifecycleTrace",
    "trace_new_physical_target_lifecycle",
    "trace_physical_target_lifecycles",
]
