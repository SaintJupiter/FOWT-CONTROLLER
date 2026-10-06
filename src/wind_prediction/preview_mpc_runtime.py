"""Adapter from one preview-MPC solution to the existing physical executor.

This module owns only the seam between planning and execution.  It converts
the first tank target of a successful preview solution into one tracking
request, executes that request for the current control block, and returns the
actual pump and platform outcomes.  Future planned targets are retained as
diagnostic predictions and are never submitted as an actuator schedule.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from fowt_platform import BallastRuntimeAssembly, IncrementalState, RotorGeneralizedLoad
from fowt_platform import ThreeTankDifferentialModes

from .execution_rollout import (
    ExecutionRolloutConfig,
    ExecutionRolloutRequest,
    ExecutionRolloutState,
    simulate_execution_step,
)
from .physical_execution_platform_path import (
    PhysicalExecutionPlatformPath,
    advance_physical_execution_platform_path,
)
from .preview_mpc import PreviewActuatorEnvelope, PreviewMPCResult


def assemble_preview_mpc_fallback_requests(
    execution_state: ExecutionRolloutState,
) -> tuple[tuple[str, ExecutionRolloutRequest], ...]:
    """Return the named conservative requests used when an MPC target is unsafe."""

    if not isinstance(execution_state, ExecutionRolloutState):
        raise TypeError("execution_state must be ExecutionRolloutState")
    return (
        (
            "fallback_continue_primary_target",
            ExecutionRolloutRequest.track(
                execution_state.primary_target_masses_kg
            ),
        ),
        (
            "fallback_release_to_current",
            ExecutionRolloutRequest.release_to_current(),
        ),
    )


def assemble_preview_actuator_envelope(
    *,
    execution_state: ExecutionRolloutState,
    execution_config: ExecutionRolloutConfig,
    tank_capacities_kg: Any,
    ballast_modes: ThreeTankDifferentialModes,
) -> PreviewActuatorEnvelope:
    """Outer-bound current-block tank motion using the physical pump state.

    Two physical rollouts request the lower and upper tank bounds while
    retaining the measured latch, dwell, ramp and unfinished-target state.
    Their end masses define an axis-aligned outer envelope for the first MPC
    block.  Hybrid deadbands mean that an arbitrary interior target is not
    guaranteed to be attained.  The selected first target is therefore
    checked again with the physical execution path before commitment.  The
    The unfinished primary target is retained as diagnostic reachability state;
    movement continuity is defined separately by the previously realised tank
    movement.
    """

    if not isinstance(execution_state, ExecutionRolloutState):
        raise TypeError("execution_state must be ExecutionRolloutState")
    if not isinstance(execution_config, ExecutionRolloutConfig):
        raise TypeError("execution_config must be ExecutionRolloutConfig")
    if not isinstance(ballast_modes, ThreeTankDifferentialModes):
        raise TypeError("ballast_modes must be ThreeTankDifferentialModes")
    capacities = np.asarray(tank_capacities_kg, dtype=float)
    if capacities.shape != (3,) or not np.all(np.isfinite(capacities)):
        raise ValueError("tank_capacities_kg must contain three finite values")
    if np.any(capacities <= 0.0):
        raise ValueError("tank_capacities_kg values must be positive")
    if not np.allclose(
        capacities,
        float(execution_config.tank_capacity_kg),
        rtol=0.0,
        atol=1.0e-9,
    ):
        raise ValueError(
            "actuator-envelope capacities must match the physical executor"
        )

    current = np.asarray(execution_state.actual_masses_kg, dtype=float)
    lower_step = simulate_execution_step(
        execution_state,
        ExecutionRolloutRequest.track(np.zeros(3)),
        execution_config,
    )
    upper_step = simulate_execution_step(
        execution_state,
        ExecutionRolloutRequest.track(capacities),
        execution_config,
    )
    extreme_deltas = np.vstack(
        (
            lower_step.state.actual_masses_kg - current,
            upper_step.state.actual_masses_kg - current,
        )
    )
    lower = np.min(extreme_deltas, axis=0)
    upper = np.max(extreme_deltas, axis=0)

    outstanding = (
        np.clip(execution_state.primary_target_masses_kg, 0.0, capacities)
        - current
    )
    differential_outstanding = ballast_modes.project_tank_mass_deltas(
        outstanding
    ).differential_tank_mass_deltas_kg
    return PreviewActuatorEnvelope(
        first_block_lower_tank_delta_kg=lower,
        first_block_upper_tank_delta_kg=upper,
        outstanding_target_tank_delta_kg=differential_outstanding,
        source=(
            "physical_executor_extreme_target_rollouts_with_current_"
            "latch_dwell_ramp_and_unfinished_target_state_outer_envelope"
        ),
    )


@dataclass(frozen=True)
class PreviewMPCPhysicalPrecheck:
    """Physical first-block outcome for one request considered for commitment."""

    source: str
    execution_request: ExecutionRolloutRequest
    execution_path: PhysicalExecutionPlatformPath
    peak_abs_roll_pitch_rad: Any
    terminal_abs_roll_pitch_rad: Any
    initial_normalized_posture: float
    maximum_normalized_posture: float
    terminal_normalized_posture: float
    within_sampled_posture_limit: bool
    posture_precheck_passed: bool
    posture_precheck_reason: str

    def __post_init__(self) -> None:
        source = str(self.source).strip()
        if not source:
            raise ValueError("source must be non-empty")
        if not isinstance(self.execution_request, ExecutionRolloutRequest):
            raise TypeError("execution_request must be ExecutionRolloutRequest")
        if not isinstance(self.execution_path, PhysicalExecutionPlatformPath):
            raise TypeError("execution_path must be PhysicalExecutionPlatformPath")
        peak = np.asarray(self.peak_abs_roll_pitch_rad, dtype=float)
        if peak.shape != (2,) or not np.all(np.isfinite(peak)) or np.any(peak < 0.0):
            raise ValueError("peak_abs_roll_pitch_rad must contain two non-negative values")
        terminal = np.asarray(self.terminal_abs_roll_pitch_rad, dtype=float)
        if terminal.shape != (2,) or not np.all(np.isfinite(terminal)) or np.any(terminal < 0.0):
            raise ValueError("terminal_abs_roll_pitch_rad must contain two non-negative values")
        for name in (
            "initial_normalized_posture",
            "maximum_normalized_posture",
            "terminal_normalized_posture",
        ):
            normalized = float(getattr(self, name))
            if not np.isfinite(normalized) or normalized < 0.0:
                raise ValueError(f"{name} must be non-negative and finite")
            object.__setattr__(self, name, normalized)
        for name in ("within_sampled_posture_limit", "posture_precheck_passed"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be a boolean")
        reason = str(self.posture_precheck_reason).strip()
        if not reason:
            raise ValueError("posture_precheck_reason must be non-empty")
        peak = np.array(peak, copy=True)
        peak.setflags(write=False)
        terminal = np.array(terminal, copy=True)
        terminal.setflags(write=False)
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "peak_abs_roll_pitch_rad", peak)
        object.__setattr__(self, "terminal_abs_roll_pitch_rad", terminal)
        object.__setattr__(self, "posture_precheck_reason", reason)


def _evaluate_sampled_posture_path(
    posture_rad: Any,
    maximum_abs_roll_pitch_rad: Any,
) -> tuple[bool, str, bool, float, float, float]:
    """Evaluate sampled roll and pitch without allowing risk transfer by axis."""

    posture = np.asarray(posture_rad, dtype=float)
    limits = np.asarray(maximum_abs_roll_pitch_rad, dtype=float)
    if posture.ndim != 2 or posture.shape[1] != 2 or not np.all(np.isfinite(posture)):
        raise ValueError("posture_rad must have shape (sample_count, 2)")
    if limits.shape != (2,) or np.any(limits <= 0.0) or np.any(np.isnan(limits)):
        raise ValueError("maximum_abs_roll_pitch_rad must contain two positive values")

    normalized_path = np.abs(posture) / limits
    initial_by_axis = normalized_path[0]
    maximum_by_axis = np.max(normalized_path, axis=0)
    terminal_by_axis = normalized_path[-1]
    initial_normalized = float(np.max(initial_by_axis))
    maximum_normalized = float(np.max(maximum_by_axis))
    terminal_normalized = float(np.max(terminal_by_axis))
    tolerance = 1.0e-9
    recovery_tolerance = 1.0e-6
    within_limit = bool(np.all(maximum_by_axis <= 1.0 + tolerance))

    if np.all(initial_by_axis <= 1.0 + tolerance):
        passed = within_limit
        reason = "sampled_path_within_limit" if passed else "sampled_path_exceeds_limit"
    else:
        axis_passes: list[bool] = []
        for initial, maximum, terminal in zip(
            initial_by_axis,
            maximum_by_axis,
            terminal_by_axis,
            strict=True,
        ):
            if initial <= 1.0 + tolerance:
                axis_passes.append(bool(maximum <= 1.0 + tolerance))
            else:
                axis_passes.append(
                    bool(
                        maximum <= initial + recovery_tolerance
                        and terminal < initial - recovery_tolerance
                    )
                )
        passed = bool(all(axis_passes))
        reason = (
            "initially_outside_limit_with_axiswise_nonworsening_recovery"
            if passed
            else "initially_outside_limit_without_axiswise_verified_recovery"
        )
    return (
        passed,
        reason,
        within_limit,
        initial_normalized,
        maximum_normalized,
        terminal_normalized,
    )


def precheck_preview_mpc_request(
    *,
    source: str,
    request: ExecutionRolloutRequest,
    maximum_abs_roll_pitch_rad: np.ndarray,
    platform_state: IncrementalState,
    execution_state: ExecutionRolloutState,
    execution_config: ExecutionRolloutConfig,
    runtime_assembly: BallastRuntimeAssembly,
    reference_tank_masses_kg: Any,
    tank_capacities_kg: Any,
    tank_coordinates_m: Any,
    current_rotor_load: RotorGeneralizedLoad,
    current_wave_load: Any,
    current_other_load: Any,
) -> PreviewMPCPhysicalPrecheck:
    path = advance_physical_execution_platform_path(
        platform_state=platform_state,
        execution_state=execution_state,
        execution_request=request,
        execution_config=execution_config,
        runtime_assembly=runtime_assembly,
        reference_tank_masses_kg=reference_tank_masses_kg,
        tank_capacities_kg=tank_capacities_kg,
        tank_coordinates_m=tank_coordinates_m,
        rotor_load=current_rotor_load,
        wave_load=current_wave_load,
        other_load=current_other_load,
        duration_s=execution_config.block_duration_s,
    )
    posture = np.vstack(
        [
            step.start_platform_state.position[[3, 4]]
            for step in path.substeps
        ]
        + [path.final_platform_state.position[[3, 4]]]
    )
    peak = np.max(np.abs(posture), axis=0)
    terminal = np.abs(posture[-1])
    (
        precheck_passed,
        reason,
        within_limit,
        initial_normalized,
        normalized,
        terminal_normalized,
    ) = _evaluate_sampled_posture_path(posture, maximum_abs_roll_pitch_rad)
    return PreviewMPCPhysicalPrecheck(
        source=source,
        execution_request=request,
        execution_path=path,
        peak_abs_roll_pitch_rad=peak,
        terminal_abs_roll_pitch_rad=terminal,
        initial_normalized_posture=initial_normalized,
        maximum_normalized_posture=normalized,
        terminal_normalized_posture=terminal_normalized,
        within_sampled_posture_limit=within_limit,
        posture_precheck_passed=precheck_passed,
        posture_precheck_reason=reason,
    )


@dataclass(frozen=True)
class PreviewMPCExecutedCycle:
    """One plan attempt and the measured outcome of the committed request."""

    plan: PreviewMPCResult
    execution_request: ExecutionRolloutRequest
    execution_path: PhysicalExecutionPlatformPath
    selected_request_source: str = "mpc_first_target"
    physical_prechecks: tuple[PreviewMPCPhysicalPrecheck, ...] = ()
    sampled_posture_limit_satisfied: bool = True
    used_fallback: bool = False
    fallback_reason: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.plan, PreviewMPCResult):
            raise TypeError("plan must be PreviewMPCResult")
        if type(self.used_fallback) is not bool:
            raise ValueError("used_fallback must be a boolean")
        reason = None if self.fallback_reason is None else str(self.fallback_reason).strip()
        if not self.plan.success and not self.used_fallback:
            raise ValueError("an unsuccessful plan requires explicit fallback execution")
        if self.used_fallback and not reason:
            raise ValueError("fallback_reason must identify why fallback was used")
        if not self.used_fallback and reason:
            raise ValueError("fallback_reason is only valid for fallback execution")
        object.__setattr__(self, "fallback_reason", reason)
        selected_source = str(self.selected_request_source).strip()
        if not selected_source:
            raise ValueError("selected_request_source must be non-empty")
        prechecks = tuple(self.physical_prechecks)
        if not prechecks or not all(
            isinstance(item, PreviewMPCPhysicalPrecheck) for item in prechecks
        ):
            raise ValueError("physical_prechecks must contain evaluated candidates")
        if selected_source not in {item.source for item in prechecks}:
            raise ValueError("selected_request_source must identify one physical precheck")
        if type(self.sampled_posture_limit_satisfied) is not bool:
            raise ValueError("sampled_posture_limit_satisfied must be a boolean")
        object.__setattr__(self, "selected_request_source", selected_source)
        object.__setattr__(self, "physical_prechecks", prechecks)
        if not isinstance(self.execution_request, ExecutionRolloutRequest):
            raise TypeError("execution_request must be ExecutionRolloutRequest")
        if not isinstance(self.execution_path, PhysicalExecutionPlatformPath):
            raise TypeError("execution_path must be PhysicalExecutionPlatformPath")
        first_executed = self.execution_path.substeps[0].execution_step
        requested = np.asarray(first_executed.requested_target_kg, dtype=float)
        if not self.used_fallback:
            if not np.allclose(
                requested,
                self.plan.current_target_tank_masses_kg,
                rtol=0.0,
                atol=1.0e-9,
            ):
                raise ValueError("execution request must equal the first MPC tank target")
        if not np.allclose(
            first_executed.requested_target_kg,
            requested,
            rtol=0.0,
            atol=1.0e-9,
        ):
            raise ValueError("physical execution path must retain the committed target")

    @property
    def next_platform_state(self) -> IncrementalState:
        return self.execution_path.final_platform_state

    @property
    def next_execution_state(self) -> ExecutionRolloutState:
        return self.execution_path.final_execution_state

    @property
    def selected_physical_precheck(self) -> PreviewMPCPhysicalPrecheck:
        return next(
            item
            for item in self.physical_prechecks
            if item.source == self.selected_request_source
        )

    @property
    def selected_posture_precheck_passed(self) -> bool:
        return self.selected_physical_precheck.posture_precheck_passed

    @property
    def selected_posture_precheck_reason(self) -> str:
        return self.selected_physical_precheck.posture_precheck_reason


def execute_preview_mpc_first_block(
    *,
    plan: PreviewMPCResult,
    platform_state: IncrementalState,
    execution_state: ExecutionRolloutState,
    execution_config: ExecutionRolloutConfig,
    runtime_assembly: BallastRuntimeAssembly,
    reference_tank_masses_kg: Any,
    tank_capacities_kg: Any,
    tank_coordinates_m: Any,
    current_rotor_load: RotorGeneralizedLoad,
    execution_rotor_load: RotorGeneralizedLoad | None = None,
    current_wave_load: Any,
    current_other_load: Any,
    fallback_request: ExecutionRolloutRequest | None = None,
    fallback_requests: tuple[tuple[str, ExecutionRolloutRequest], ...] = (),
    maximum_abs_roll_pitch_rad: Any | None = None,
) -> PreviewMPCExecutedCycle:
    """Precheck candidates with visible loads and commit one request.

    ``current_rotor_load`` is the first-block load available when the
    controller selects a request.  ``execution_rotor_load`` may provide the
    load subsequently realised by the plant.  Separating them prevents
    recorded future wind from influencing candidate selection.
    """

    if not isinstance(plan, PreviewMPCResult):
        raise TypeError("plan must be PreviewMPCResult")
    if maximum_abs_roll_pitch_rad is None:
        limits = np.full(2, np.inf)
    else:
        limits = np.asarray(maximum_abs_roll_pitch_rad, dtype=float)
        if limits.shape != (2,) or not np.all(np.isfinite(limits)) or np.any(limits <= 0.0):
            raise ValueError("maximum_abs_roll_pitch_rad must contain two positive values")
    if not isinstance(current_rotor_load, RotorGeneralizedLoad):
        raise TypeError("current_rotor_load must be RotorGeneralizedLoad")
    realised_rotor_load = (
        current_rotor_load
        if execution_rotor_load is None
        else execution_rotor_load
    )
    if not isinstance(realised_rotor_load, RotorGeneralizedLoad):
        raise TypeError("execution_rotor_load must be RotorGeneralizedLoad")

    candidates: list[tuple[str, ExecutionRolloutRequest]] = []
    if plan.success:
        candidates.append(
            (
                "mpc_first_target",
                ExecutionRolloutRequest.track(plan.current_target_tank_masses_kg),
            )
        )
    if fallback_request is not None:
        if not isinstance(fallback_request, ExecutionRolloutRequest):
            raise TypeError("fallback_request must be ExecutionRolloutRequest")
        candidates.append(("fallback_request", fallback_request))
    for source, request in fallback_requests:
        if not isinstance(request, ExecutionRolloutRequest):
            raise TypeError("fallback requests must be ExecutionRolloutRequest values")
        candidates.append((str(source), request))
    if not candidates:
        raise ValueError("at least one MPC or fallback request must be available")

    prechecks = tuple(
        precheck_preview_mpc_request(
            source=source,
            request=request,
            maximum_abs_roll_pitch_rad=limits,
            platform_state=platform_state,
            execution_state=execution_state,
            execution_config=execution_config,
            runtime_assembly=runtime_assembly,
            reference_tank_masses_kg=reference_tank_masses_kg,
            tank_capacities_kg=tank_capacities_kg,
            tank_coordinates_m=tank_coordinates_m,
            current_rotor_load=current_rotor_load,
            current_wave_load=current_wave_load,
            current_other_load=current_other_load,
        )
        for source, request in candidates
    )
    mpc_precheck = next(
        (item for item in prechecks if item.source == "mpc_first_target"), None
    )
    if mpc_precheck is not None and mpc_precheck.posture_precheck_passed:
        selected = mpc_precheck
        used_fallback = False
        fallback_reason = None
    else:
        fallback_prechecks = tuple(
            item for item in prechecks if item.source != "mpc_first_target"
        )
        if not fallback_prechecks:
            reason = (
                f"preview_mpc_unsuccessful: {plan.message}"
                if not plan.success
                else "mpc_first_target_failed_physical_posture_precheck"
            )
            raise ValueError(f"{reason}; no fallback requests were supplied")
        safe = tuple(
            item for item in fallback_prechecks if item.posture_precheck_passed
        )
        pool = safe if safe else fallback_prechecks
        selected = min(
            pool,
            key=lambda item: (
                item.maximum_normalized_posture,
                item.execution_path.transferred_volume_m3,
                item.execution_path.aggregate_pump_active_time_s,
                item.execution_path.pump_start_count,
            ),
        )
        used_fallback = True
        fallback_reason = (
            f"preview_mpc_unsuccessful: {plan.message}"
            if not plan.success
            else "mpc_first_target_failed_physical_posture_precheck"
        )
        if not safe:
            fallback_reason += "; no fallback passed the physical posture precheck"
    request = selected.execution_request
    if execution_rotor_load is None:
        path = selected.execution_path
    else:
        path = advance_physical_execution_platform_path(
            platform_state=platform_state,
            execution_state=execution_state,
            execution_request=request,
            execution_config=execution_config,
            runtime_assembly=runtime_assembly,
            reference_tank_masses_kg=reference_tank_masses_kg,
            tank_capacities_kg=tank_capacities_kg,
            tank_coordinates_m=tank_coordinates_m,
            rotor_load=realised_rotor_load,
            wave_load=current_wave_load,
            other_load=current_other_load,
            duration_s=execution_config.block_duration_s,
        )
    realised_posture = np.vstack(
        [
            step.start_platform_state.position[[3, 4]]
            for step in path.substeps
        ]
        + [path.final_platform_state.position[[3, 4]]]
    )
    _, _, realised_within_limit, _, _, _ = _evaluate_sampled_posture_path(
        realised_posture,
        limits,
    )
    return PreviewMPCExecutedCycle(
        plan=plan,
        execution_request=request,
        execution_path=path,
        selected_request_source=selected.source,
        physical_prechecks=prechecks,
        sampled_posture_limit_satisfied=realised_within_limit,
        used_fallback=used_fallback,
        fallback_reason=fallback_reason,
    )


__all__ = [
    "PreviewMPCExecutedCycle",
    "PreviewMPCPhysicalPrecheck",
    "assemble_preview_actuator_envelope",
    "assemble_preview_mpc_fallback_requests",
    "execute_preview_mpc_first_block",
    "precheck_preview_mpc_request",
]
