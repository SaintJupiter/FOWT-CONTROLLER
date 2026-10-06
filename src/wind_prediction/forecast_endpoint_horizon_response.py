"""Replay every admitted forecast endpoint as an independent held response.

Each forecast lead already has a source-bound RHS ballast endpoint and a pump
preview from the same measured actuator state.  This module turns those
parallel endpoint facts into parallel physical responses: one endpoint is
declared at the current control origin, then that one request is held over the
available forecast loads.  The records are intentionally independent.  They
are not a target sequence, a rolling plan, an objective, or a selected action.

The first-lead response reuses the established current-block replay.  Later
leads add only the missing same-origin response facts, so a later policy can
eventually state which physical difference it relies on without silently
replaying a different pump state or inventing an intermediate target.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .forecast_execution_block_lifecycle_rollout import (
    ForecastExecutionBlockLifecycleRollout,
    HeldLifecycleForecastHorizonRollout,
    rollout_current_execution_block_lifecycle,
    rollout_held_lifecycle_to_forecast_horizon,
)
from .forecast_rhs_ballast_endpoint_preview import ForecastRhsBallastEndpointPreview
from .physical_lifecycle_comparison import (
    PhysicalLifecycleComparison,
    PhysicalLifecycleHorizonComparison,
)
from .physical_target_lifecycle import (
    PhysicalTargetLifecycle,
    PhysicalTargetLifecycleTrace,
    trace_new_physical_target_lifecycle,
)


_STATE_TOLERANCE = 1.0e-10


def _readonly_vector(value: Any, *, shape: tuple[int, ...], name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.shape != shape or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must have shape {shape} and contain finite values")
    result = np.array(array, dtype=float, copy=True)
    result.setflags(write=False)
    return result


def _readonly_time_vector(value: Any, *, name: str) -> np.ndarray:
    """Copy one strictly increasing post-origin sample-time vector."""

    array = np.asarray(value, dtype=float)
    if (
        array.ndim != 1
        or array.size == 0
        or not np.all(np.isfinite(array))
        or np.any(array <= 0.0)
        or np.any(np.diff(array) <= 0.0)
    ):
        raise ValueError(
            f"{name} must contain strictly increasing positive finite sample times"
        )
    result = np.array(array, dtype=float, copy=True)
    result.setflags(write=False)
    return result


def _readonly_pitch_roll_history(value: Any, *, name: str) -> np.ndarray:
    """Copy one finite post-origin ``(sample, pitch/roll)`` history."""

    array = np.asarray(value, dtype=float)
    if (
        array.ndim != 2
        or array.shape[0] == 0
        or array.shape[1] != 2
        or not np.all(np.isfinite(array))
    ):
        raise ValueError(
            f"{name} must have shape (sample_count, 2) and contain finite values"
        )
    result = np.array(array, dtype=float, copy=True)
    result.setflags(write=False)
    return result


def _horizon_pitch_roll_history(
    rollout: HeldLifecycleForecastHorizonRollout,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return post-origin sample times plus pitch/roll angle and rate histories."""

    times: list[float] = []
    positions: list[np.ndarray] = []
    velocities: list[np.ndarray] = []
    for interval in rollout.intervals:
        sample_time_s = float(interval.start_time_s)
        for substep in interval.physical_path.substeps:
            sample_time_s += float(substep.duration_s)
            times.append(sample_time_s)
            positions.append(
                np.asarray(
                    substep.next_platform_state.position[[4, 3]],
                    dtype=float,
                )
            )
            velocities.append(
                np.asarray(
                    substep.next_platform_state.velocity[[4, 3]],
                    dtype=float,
                )
            )
        if not np.isclose(sample_time_s, interval.end_time_s, rtol=0.0, atol=1.0e-9):
            raise ValueError("physical substeps must end at their forecast interval boundary")
    return (
        _readonly_time_vector(times, name="horizon_sample_times_s"),
        _readonly_pitch_roll_history(
            positions,
            name="horizon_pitch_roll_rad",
        ),
        _readonly_pitch_roll_history(
            velocities,
            name="horizon_pitch_roll_rate_rad_s",
        ),
    )


def _aligned_horizon_pitch_roll_histories(
    *,
    response: HeldLifecycleForecastHorizonRollout,
    continuation: HeldLifecycleForecastHorizonRollout,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return same-time endpoint and continuation histories in fixed axis order."""

    response_times, response_angles, response_rates = _horizon_pitch_roll_history(response)
    continuation_times, continuation_angles, continuation_rates = (
        _horizon_pitch_roll_history(continuation)
    )
    if (
        response_times.shape != continuation_times.shape
        or not np.allclose(
            response_times,
            continuation_times,
            rtol=0.0,
            atol=1.0e-9,
        )
    ):
        raise ValueError(
            "endpoint response and continuation must retain one aligned physical time grid"
        )
    return (
        response_times,
        response_angles,
        response_rates,
        continuation_angles,
        continuation_rates,
    )


def _states_match(left: object, right: object) -> bool:
    return bool(
        np.allclose(
            left.position,
            right.position,
            rtol=0.0,
            atol=_STATE_TOLERANCE,
        )
        and np.allclose(
            left.velocity,
            right.velocity,
            rtol=0.0,
            atol=_STATE_TOLERANCE,
        )
    )


@dataclass(frozen=True)
class ForecastEndpointHeldResponse:
    """One endpoint-derived request replayed from the current origin.

    ``endpoint_preview`` identifies the forecast lead that supplied the full
    tank target.  ``current_block_rollout`` applies that target only over the
    configured current block, while ``horizon_rollout`` holds the same request
    only to expose a conditional forecast consequence.  Neither trajectory is
    an actual later-cycle state handoff.
    """

    endpoint_preview: ForecastRhsBallastEndpointPreview
    lifecycle_trace: PhysicalTargetLifecycleTrace
    current_block_rollout: ForecastExecutionBlockLifecycleRollout
    horizon_rollout: HeldLifecycleForecastHorizonRollout

    def __post_init__(self) -> None:
        if not isinstance(self.endpoint_preview, ForecastRhsBallastEndpointPreview):
            raise TypeError("endpoint_preview must be ForecastRhsBallastEndpointPreview")
        if not isinstance(self.lifecycle_trace, PhysicalTargetLifecycleTrace):
            raise TypeError("lifecycle_trace must be PhysicalTargetLifecycleTrace")
        if not isinstance(
            self.current_block_rollout,
            ForecastExecutionBlockLifecycleRollout,
        ):
            raise TypeError(
                "current_block_rollout must be ForecastExecutionBlockLifecycleRollout"
            )
        if not isinstance(self.horizon_rollout, HeldLifecycleForecastHorizonRollout):
            raise TypeError("horizon_rollout must be HeldLifecycleForecastHorizonRollout")
        if self.lifecycle_trace.lifecycle is not PhysicalTargetLifecycle.NEW_TRACK:
            raise ValueError("endpoint response must retain a new_track lifecycle")
        if self.lifecycle_trace.source_preview is not self.endpoint_preview:
            raise ValueError("endpoint response must retain its exact source preview")
        if self.current_block_rollout.lifecycle_trace is not self.lifecycle_trace:
            raise ValueError("current-block rollout must retain the endpoint lifecycle")
        trajectory = self.endpoint_preview.rhs_diagnostic.rhs_point.trajectory
        if self.current_block_rollout.trajectory is not trajectory:
            raise ValueError("current-block rollout must use the endpoint trajectory")
        if self.horizon_rollout.prefix is not self.current_block_rollout:
            raise ValueError("horizon rollout must retain the endpoint current-block prefix")
        if self.horizon_rollout.prefix.trajectory is not trajectory:
            raise ValueError("horizon rollout must use the endpoint trajectory")

    @property
    def lead_index(self) -> int:
        """Return the forecast lead that supplied this independent endpoint."""

        return self.endpoint_preview.lead_index

    @property
    def lead_time_s(self) -> float:
        """Return the corresponding source-bound forecast time."""

        return self.endpoint_preview.lead_time_s

    @property
    def target_tank_masses_kg(self) -> np.ndarray:
        """Return the full diagnostic target held by this counterfactual."""

        return _readonly_vector(
            self.lifecycle_trace.requested_target_masses_kg,
            shape=(3,),
            name="target_tank_masses_kg",
        )

    @property
    def current_block_end_pitch_roll_rad(self) -> np.ndarray:
        """Return the current-block pitch and roll resulting from this target."""

        return _readonly_vector(
            self.current_block_rollout.final_platform_state.position[[4, 3]],
            shape=(2,),
            name="current_block_end_pitch_roll_rad",
        )

    @property
    def horizon_end_pitch_roll_rad(self) -> np.ndarray:
        """Return the conditional held-response pitch and roll at horizon end."""

        return _readonly_vector(
            self.horizon_rollout.final_platform_state.position[[4, 3]],
            shape=(2,),
            name="horizon_end_pitch_roll_rad",
        )

    @property
    def horizon_peak_abs_pitch_roll_rad(self) -> np.ndarray:
        """Return angle peaks on the held response, without interpreting them."""

        return _readonly_vector(
            self.horizon_rollout.post_origin_peak_abs_pitch_roll_rad,
            shape=(2,),
            name="horizon_peak_abs_pitch_roll_rad",
        )

    @property
    def horizon_peak_abs_pitch_roll_rate_rad_s(self) -> np.ndarray:
        """Return angular-rate peaks on the held response without a verdict."""

        return _readonly_vector(
            self.horizon_rollout.post_origin_peak_abs_pitch_roll_rate_rad_s,
            shape=(2,),
            name="horizon_peak_abs_pitch_roll_rate_rad_s",
        )

    @property
    def horizon_transferred_volume_m3(self) -> float:
        """Return pump transfer accumulated on this held counterfactual."""

        return float(
            sum(
                interval.physical_path.transferred_volume_m3
                for interval in self.horizon_rollout.intervals
            )
        )


@dataclass(frozen=True)
class ForecastEndpointHeldResponseSet:
    """Same-origin held responses for every admitted forecast endpoint.

    The set makes no response preferable.  Its only ordering is forecast lead
    order, which retains traceability back to the source-bound endpoint facts.
    A non-admitted forecast deliberately yields an empty set rather than an
    implicit fallback target.
    """

    current_block_comparison: PhysicalLifecycleComparison
    responses: tuple[ForecastEndpointHeldResponse, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.current_block_comparison, PhysicalLifecycleComparison):
            raise TypeError("current_block_comparison must be PhysicalLifecycleComparison")
        responses = tuple(self.responses)
        if not all(isinstance(item, ForecastEndpointHeldResponse) for item in responses):
            raise TypeError("responses must contain ForecastEndpointHeldResponse values")

        comparison = self.current_block_comparison
        if not comparison.forecast_admission.admitted:
            if responses:
                raise ValueError("non-admitted forecasts must not expose endpoint responses")
            object.__setattr__(self, "responses", responses)
            return

        first_outcome = comparison.new_track
        if first_outcome is None:
            raise ValueError("admitted forecast comparison must retain its first endpoint")
        previews = comparison.decision_input.horizon_endpoint_previews.previews
        if len(responses) != len(previews):
            raise ValueError("responses must contain one record per forecast endpoint")
        for index, (response, preview) in enumerate(zip(responses, previews)):
            if response.endpoint_preview is not preview:
                raise ValueError("endpoint responses must retain forecast-lead order")
            if response.lead_index != index:
                raise ValueError("endpoint responses must retain contiguous forecast leads")
            if response.current_block_rollout.trajectory is not comparison.decision_input.trajectory:
                raise ValueError("endpoint response must use the comparison trajectory")
        first = responses[0]
        if first.current_block_rollout is not first_outcome.rollout:
            raise ValueError("first endpoint response must reuse the established current-block rollout")
        object.__setattr__(self, "responses", responses)

    def response_for_lead(self, lead_index: int) -> ForecastEndpointHeldResponse:
        """Return one lead-bound response without comparing or selecting it."""

        if not isinstance(lead_index, int) or isinstance(lead_index, bool):
            raise TypeError("lead_index must be an integer")
        try:
            return self.responses[lead_index]
        except IndexError as exc:
            raise IndexError("lead_index is outside the admitted forecast horizon") from exc

    def change_from_continuation(
        self,
        *,
        horizon_comparison: PhysicalLifecycleHorizonComparison,
        lead_index: int,
    ) -> "ForecastEndpointHeldResponseChangeFromContinuation":
        """Expose one endpoint response relative to held continuation.

        The result is only an alternative-minus-continuation accounting record.
        It does not make either response preferable or authorize its endpoint
        as the current target.
        """

        return ForecastEndpointHeldResponseChangeFromContinuation(
            response_set=self,
            horizon_comparison=horizon_comparison,
            lead_index=lead_index,
        )

    def changes_from_continuation(
        self,
        *,
        horizon_comparison: PhysicalLifecycleHorizonComparison,
    ) -> tuple["ForecastEndpointHeldResponseChangeFromContinuation", ...]:
        """Return lead-ordered response deltas without ranking them."""

        return tuple(
            self.change_from_continuation(
                horizon_comparison=horizon_comparison,
                lead_index=lead_index,
            )
            for lead_index in range(len(self.responses))
        )


@dataclass(frozen=True)
class ForecastEndpointHeldResponseChangeFromContinuation:
    """One endpoint-held response minus the same-origin continuation response.

    All signs use ``endpoint response - continuation``.  The two paths share
    the same current actuator origin and forecast trajectory.  The record only
    preserves modeled differences in execution and platform response.  It is
    not an objective, a posture verdict, or a selected action.
    """

    response_set: ForecastEndpointHeldResponseSet
    horizon_comparison: PhysicalLifecycleHorizonComparison
    lead_index: int

    def __post_init__(self) -> None:
        if not isinstance(self.response_set, ForecastEndpointHeldResponseSet):
            raise TypeError("response_set must be ForecastEndpointHeldResponseSet")
        if not isinstance(self.horizon_comparison, PhysicalLifecycleHorizonComparison):
            raise TypeError("horizon_comparison must be PhysicalLifecycleHorizonComparison")
        if (
            self.horizon_comparison.current_block_comparison
            is not self.response_set.current_block_comparison
        ):
            raise ValueError(
                "horizon comparison must retain the response set current-block comparison"
            )
        if not isinstance(self.lead_index, int) or isinstance(self.lead_index, bool):
            raise TypeError("lead_index must be an integer")
        response = self.response_set.response_for_lead(self.lead_index)
        continuation = self.horizon_comparison.continue_existing
        if (
            response.current_block_rollout.trajectory
            is not continuation.current_block_outcome.rollout.trajectory
            or response.horizon_rollout.prefix.trajectory
            is not continuation.rollout.prefix.trajectory
        ):
            raise ValueError("endpoint response and continuation must retain one trajectory")
        _aligned_horizon_pitch_roll_histories(
            response=response.horizon_rollout,
            continuation=continuation.rollout,
        )

    @property
    def response(self) -> ForecastEndpointHeldResponse:
        """Return the endpoint-held response without interpreting it."""

        return self.response_set.response_for_lead(self.lead_index)

    @property
    def continuation(self):
        """Return the shared held continuation response used as reference."""

        return self.horizon_comparison.continue_existing

    @property
    def current_block_transferred_volume_delta_m3(self) -> float:
        return float(
            sum(
                interval.physical_path.transferred_volume_m3
                for interval in self.response.current_block_rollout.intervals
            )
            - self.continuation.current_block_outcome.transferred_volume_m3
        )

    @property
    def horizon_transferred_volume_delta_m3(self) -> float:
        return float(
            self.response.horizon_transferred_volume_m3
            - self.continuation.transferred_volume_m3
        )

    @property
    def current_block_pump_active_time_delta_s(self) -> float:
        return float(
            sum(
                interval.physical_path.aggregate_pump_active_time_s
                for interval in self.response.current_block_rollout.intervals
            )
            - self.continuation.current_block_outcome.aggregate_pump_active_time_s
        )

    @property
    def horizon_pump_active_time_delta_s(self) -> float:
        return float(
            sum(
                interval.physical_path.aggregate_pump_active_time_s
                for interval in self.response.horizon_rollout.intervals
            )
            - self.continuation.aggregate_pump_active_time_s
        )

    @property
    def current_block_pump_start_count_delta(self) -> int:
        return int(
            sum(
                interval.physical_path.pump_start_count
                for interval in self.response.current_block_rollout.intervals
            )
            - self.continuation.current_block_outcome.pump_start_count
        )

    @property
    def horizon_pump_start_count_delta(self) -> int:
        return int(
            sum(
                interval.physical_path.pump_start_count
                for interval in self.response.horizon_rollout.intervals
            )
            - self.continuation.pump_start_count
        )

    @property
    def current_block_reached_final_tank_mass_delta_kg(self) -> np.ndarray:
        return _readonly_vector(
            self.response.current_block_rollout.reached_final_tank_masses_kg
            - self.continuation.current_block_outcome.reached_final_tank_masses_kg,
            shape=(3,),
            name="current_block_reached_final_tank_mass_delta_kg",
        )

    @property
    def current_block_end_pitch_roll_delta_rad(self) -> np.ndarray:
        return _readonly_vector(
            self.response.current_block_end_pitch_roll_rad
            - self.continuation.current_block_outcome.end_pitch_roll_rad,
            shape=(2,),
            name="current_block_end_pitch_roll_delta_rad",
        )

    @property
    def horizon_end_pitch_roll_delta_rad(self) -> np.ndarray:
        return _readonly_vector(
            self.response.horizon_end_pitch_roll_rad
            - self.continuation.end_pitch_roll_rad,
            shape=(2,),
            name="horizon_end_pitch_roll_delta_rad",
        )

    @property
    def horizon_peak_abs_pitch_roll_delta_rad(self) -> np.ndarray:
        return _readonly_vector(
            self.response.horizon_peak_abs_pitch_roll_rad
            - self.continuation.peak_abs_pitch_roll_rad,
            shape=(2,),
            name="horizon_peak_abs_pitch_roll_delta_rad",
        )

    @property
    def horizon_sample_times_s(self) -> np.ndarray:
        """Return the common post-origin physical sample times for both paths."""

        times, _, _, _, _ = _aligned_horizon_pitch_roll_histories(
            response=self.response.horizon_rollout,
            continuation=self.continuation.rollout,
        )
        return times

    @property
    def horizon_pitch_roll_delta_rad(self) -> np.ndarray:
        """Return same-time signed angle changes, endpoint response minus continuation."""

        _, response_angles, _, continuation_angles, _ = (
            _aligned_horizon_pitch_roll_histories(
                response=self.response.horizon_rollout,
                continuation=self.continuation.rollout,
            )
        )
        return _readonly_pitch_roll_history(
            response_angles - continuation_angles,
            name="horizon_pitch_roll_delta_rad",
        )

    @property
    def horizon_pitch_roll_rate_delta_rad_s(self) -> np.ndarray:
        """Return same-time signed angular-rate changes, endpoint response minus continuation."""

        _, _, response_rates, _, continuation_rates = (
            _aligned_horizon_pitch_roll_histories(
                response=self.response.horizon_rollout,
                continuation=self.continuation.rollout,
            )
        )
        return _readonly_pitch_roll_history(
            response_rates - continuation_rates,
            name="horizon_pitch_roll_rate_delta_rad_s",
        )

    @property
    def horizon_abs_pitch_roll_delta_rad(self) -> np.ndarray:
        """Return same-time magnitude changes, ``abs(endpoint) - abs(continuation)``."""

        _, response_angles, _, continuation_angles, _ = (
            _aligned_horizon_pitch_roll_histories(
                response=self.response.horizon_rollout,
                continuation=self.continuation.rollout,
            )
        )
        return _readonly_pitch_roll_history(
            np.abs(response_angles) - np.abs(continuation_angles),
            name="horizon_abs_pitch_roll_delta_rad",
        )


def assemble_forecast_endpoint_held_responses(
    *,
    current_block_comparison: PhysicalLifecycleComparison,
) -> ForecastEndpointHeldResponseSet:
    """Replay each admitted forecast endpoint from one current physical state.

    For every later forecast lead, the full endpoint is projected back to the
    current decision origin and held as one fixed request.  This produces
    comparable physical facts, not a sequence of future commands.  The caller
    remains responsible for any later decision rule.
    """

    if not isinstance(current_block_comparison, PhysicalLifecycleComparison):
        raise TypeError("current_block_comparison must be PhysicalLifecycleComparison")
    if not current_block_comparison.forecast_admission.admitted:
        return ForecastEndpointHeldResponseSet(
            current_block_comparison=current_block_comparison,
            responses=(),
        )

    first_outcome = current_block_comparison.new_track
    if first_outcome is None:
        raise ValueError("admitted forecast comparison must retain its first endpoint")
    decision_input = current_block_comparison.decision_input
    previews = decision_input.horizon_endpoint_previews.previews
    first_trace = first_outcome.rollout.lifecycle_trace
    responses: list[ForecastEndpointHeldResponse] = []
    for index, preview in enumerate(previews):
        if index == 0:
            trace = first_trace
            current_rollout = first_outcome.rollout
        else:
            trace = trace_new_physical_target_lifecycle(
                execution_state=preview.reachability.execution_start_state,
                execution_config=preview.reachability.execution_config,
                execution_duration_s=preview.reachability.execution_duration_s,
                execution_start_time=preview.execution_start_time,
                endpoint_preview=preview,
            )
            current_rollout = rollout_current_execution_block_lifecycle(
                trajectory=decision_input.trajectory,
                lifecycle_trace=trace,
                runtime_assembly=decision_input.runtime_assembly,
            )
        responses.append(
            ForecastEndpointHeldResponse(
                endpoint_preview=preview,
                lifecycle_trace=trace,
                current_block_rollout=current_rollout,
                horizon_rollout=rollout_held_lifecycle_to_forecast_horizon(
                    prefix=current_rollout
                ),
            )
        )
    return ForecastEndpointHeldResponseSet(
        current_block_comparison=current_block_comparison,
        responses=tuple(responses),
    )


__all__ = [
    "ForecastEndpointHeldResponseChangeFromContinuation",
    "ForecastEndpointHeldResponse",
    "ForecastEndpointHeldResponseSet",
    "assemble_forecast_endpoint_held_responses",
]
