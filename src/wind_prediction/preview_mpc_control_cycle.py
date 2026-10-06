"""One receding-horizon decision from forecast loads to physical execution.

The convex MPC remains responsible for continuous differential-ballast
planning.  This module owns the separate target-lifecycle decision that the
optimizer cannot represent without discrete variables: continue the existing
target, release it to the measured tank state, or track the newly planned MPC
target.  Every option starts from the same measured state and planner-visible
first-block load.  Only the selected first-block request is sent to the
physical executor.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from typing import Any

import numpy as np

from fowt_platform import (
    BallastRuntimeAssembly,
    DifferentialModeProjection,
    IncrementalState,
    RotorGeneralizedLoad,
    ThreeTankDifferentialModes,
    assemble_ballast_model_snapshot,
)

from .execution_rollout import (
    ExecutionRolloutConfig,
    ExecutionRolloutRequest,
    ExecutionRolloutState,
)
from .physical_execution_platform_path import (
    PhysicalExecutionPlatformPath,
    advance_physical_execution_platform_path,
)
from .preview_mpc import (
    PreviewActuatorEnvelope,
    PreviewBlockModel,
    PreviewMPCController,
    PreviewMPCResult,
)
from .preview_mpc_runtime import (
    PreviewMPCPhysicalPrecheck,
    assemble_preview_actuator_envelope,
    precheck_preview_mpc_request,
)
from .run_identity import sha256_json


class PreviewMPCTargetLifecycle(str, Enum):
    """Target operations compared at one control-cycle origin."""

    CONTINUE_EXISTING_TARGET = "continue_existing_target"
    RELEASE_TO_ACTUAL = "release_to_actual"
    TRACK_NEW_MPC_TARGET = "track_new_mpc_target"


class PreviewMPCSelectionReason(str, Enum):
    """How the current-cycle target operation was selected."""

    LOWEST_COMMON_HORIZON_OBJECTIVE = "lowest_common_horizon_objective"
    TIE_BROKEN_BY_PHYSICAL_EXECUTION = "tie_broken_by_physical_execution"
    DEGRADED_FIRST_BLOCK_SAFE_NO_FULLY_ELIGIBLE_OPTION = (
        "degraded_first_block_safe_no_fully_eligible_option"
    )


_LIFECYCLE_TIE_ORDER = {
    PreviewMPCTargetLifecycle.CONTINUE_EXISTING_TARGET: 0,
    PreviewMPCTargetLifecycle.RELEASE_TO_ACTUAL: 1,
    PreviewMPCTargetLifecycle.TRACK_NEW_MPC_TARGET: 2,
}

# The MPC objective is dimensionless after physical-scale normalization.
# Differences below this numerical/operational equivalence band are resolved
# using the physically replayed first-block burden instead of a solver-level
# distinction that has no practical control meaning.  This is a stage working
# tolerance, not a calibrated plant parameter.
_OBJECTIVE_EQUIVALENCE_ATOL = 1.0e-9
_OBJECTIVE_EQUIVALENCE_RTOL = 1.0e-6


def _readonly_finite_array(
    name: str,
    value: Any,
    shape: tuple[int, ...],
) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.shape != shape or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must have shape {shape} and contain finite values")
    result = np.array(array, copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class PreviewMPCPlanningSource:
    """Identity and consumed horizon of the information used for one decision."""

    source: str
    model_version: str
    origin_time: str
    forecast_mode: str
    source_record_sha256: str
    controller_visible_loads_sha256: str
    sample_period_s: float
    lead_times_s: Any
    uses_future_information: bool

    def __post_init__(self) -> None:
        for name in (
            "source",
            "model_version",
            "origin_time",
            "forecast_mode",
        ):
            value = str(getattr(self, name)).strip()
            if not value:
                raise ValueError(f"{name} must be non-empty")
            object.__setattr__(self, name, value)
        for name in (
            "source_record_sha256",
            "controller_visible_loads_sha256",
        ):
            digest = str(getattr(self, name)).strip().lower()
            if len(digest) != 64 or any(
                character not in "0123456789abcdef" for character in digest
            ):
                raise ValueError(f"{name} must be a 64-character SHA-256 digest")
            object.__setattr__(self, name, digest)
        sample_period = float(self.sample_period_s)
        if not np.isfinite(sample_period) or sample_period <= 0.0:
            raise ValueError("sample_period_s must be finite and positive")
        object.__setattr__(self, "sample_period_s", sample_period)
        lead_times = np.asarray(self.lead_times_s, dtype=float).reshape(-1)
        if (
            lead_times.size == 0
            or not np.all(np.isfinite(lead_times))
            or np.any(lead_times <= 0.0)
            or np.any(np.diff(lead_times) <= 0.0)
        ):
            raise ValueError("lead_times_s must be finite, positive, and increasing")
        lead_times = np.array(lead_times, copy=True)
        lead_times.setflags(write=False)
        object.__setattr__(self, "lead_times_s", lead_times)
        if type(self.uses_future_information) is not bool:
            raise ValueError("uses_future_information must be a boolean")

    @property
    def horizon_steps(self) -> int:
        return int(self.lead_times_s.size)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "model_version": self.model_version,
            "origin_time": self.origin_time,
            "forecast_mode": self.forecast_mode,
            "source_record_sha256": self.source_record_sha256,
            "controller_visible_loads_sha256": (
                self.controller_visible_loads_sha256
            ),
            "sample_period_s": self.sample_period_s,
            "lead_times_s": self.lead_times_s.tolist(),
            "uses_future_information": self.uses_future_information,
        }


@dataclass(frozen=True)
class PreviewMPCControlCycleInput:
    """Complete, source-bound input for one receding-horizon control cycle."""

    planning_source: PreviewMPCPlanningSource
    controller: PreviewMPCController
    ballast_modes: ThreeTankDifferentialModes
    platform_state: IncrementalState
    execution_state: ExecutionRolloutState
    execution_config: ExecutionRolloutConfig
    runtime_assembly: BallastRuntimeAssembly
    reference_tank_masses_kg: Any
    tank_capacities_kg: Any
    tank_coordinates_m: Any
    generalized_disturbance_loads: Any
    planner_first_block_rotor_load: RotorGeneralizedLoad
    current_wave_load: Any
    current_other_load: Any
    maximum_abs_roll_pitch_rad: Any
    execution_first_block_rotor_load: RotorGeneralizedLoad
    maximum_abs_working_model_scope_roll_pitch_rad: Any
    maximum_abs_total_tank_mass_deviation_from_reference_kg: float
    execution_first_block_rotor_load_substeps: (
        tuple[RotorGeneralizedLoad, ...] | None
    ) = None

    def __post_init__(self) -> None:
        for name, expected_type in (
            ("planning_source", PreviewMPCPlanningSource),
            ("controller", PreviewMPCController),
            ("ballast_modes", ThreeTankDifferentialModes),
            ("platform_state", IncrementalState),
            ("execution_state", ExecutionRolloutState),
            ("execution_config", ExecutionRolloutConfig),
            ("runtime_assembly", BallastRuntimeAssembly),
            ("planner_first_block_rotor_load", RotorGeneralizedLoad),
            ("execution_first_block_rotor_load", RotorGeneralizedLoad),
        ):
            if not isinstance(getattr(self, name), expected_type):
                raise TypeError(f"{name} must be {expected_type.__name__}")
        execution_substeps = self.execution_first_block_rotor_load_substeps
        if execution_substeps is not None:
            execution_substeps = tuple(execution_substeps)
            if not execution_substeps:
                raise ValueError(
                    "execution_first_block_rotor_load_substeps must not be empty"
                )
            if not all(
                isinstance(item, RotorGeneralizedLoad)
                for item in execution_substeps
            ):
                raise TypeError(
                    "execution_first_block_rotor_load_substeps must contain "
                    "RotorGeneralizedLoad values"
                )
            expected_count = int(
                math.ceil(
                    self.execution_config.block_duration_s
                    / self.execution_config.internal_step_s
                )
            )
            if len(execution_substeps) != expected_count:
                raise ValueError(
                    "execution_first_block_rotor_load_substeps must contain "
                    "one load per physical execution substep"
                )
            mean_generalized_load = np.mean(
                np.vstack(
                    [item.generalized_load_platform for item in execution_substeps]
                ),
                axis=0,
            )
            if not np.allclose(
                mean_generalized_load,
                self.execution_first_block_rotor_load.generalized_load_platform,
                rtol=1.0e-10,
                atol=1.0e-7,
            ):
                raise ValueError(
                    "execution_first_block_rotor_load must summarize the mean "
                    "realised substep load"
                )
            object.__setattr__(
                self,
                "execution_first_block_rotor_load_substeps",
                execution_substeps,
            )
        reference = _readonly_finite_array(
            "reference_tank_masses_kg",
            self.reference_tank_masses_kg,
            (3,),
        )
        capacities = _readonly_finite_array(
            "tank_capacities_kg",
            self.tank_capacities_kg,
            (3,),
        )
        coordinates = _readonly_finite_array(
            "tank_coordinates_m",
            self.tank_coordinates_m,
            (3, 3),
        )
        loads = _readonly_finite_array(
            "generalized_disturbance_loads",
            self.generalized_disturbance_loads,
            (self.planning_source.horizon_steps, 6),
        )
        if sha256_json(loads.tolist()) != (
            self.planning_source.controller_visible_loads_sha256
        ):
            raise ValueError(
                "controller-visible load digest must match the supplied MPC horizon"
            )
        wave = _readonly_finite_array("current_wave_load", self.current_wave_load, (6,))
        other = _readonly_finite_array("current_other_load", self.current_other_load, (6,))
        posture_limit = _readonly_finite_array(
            "maximum_abs_roll_pitch_rad",
            self.maximum_abs_roll_pitch_rad,
            (2,),
        )
        model_scope_limit = _readonly_finite_array(
            "maximum_abs_working_model_scope_roll_pitch_rad",
            self.maximum_abs_working_model_scope_roll_pitch_rad,
            (2,),
        )
        if np.any(capacities <= 0.0) or np.any(reference < 0.0) or np.any(reference > capacities):
            raise ValueError("reference tank masses must remain within positive capacities")
        if np.any(posture_limit <= 0.0):
            raise ValueError("maximum_abs_roll_pitch_rad values must be positive")
        if np.any(model_scope_limit <= 0.0):
            raise ValueError(
                "maximum_abs_working_model_scope_roll_pitch_rad values must be positive"
            )
        if np.any(model_scope_limit > posture_limit + 1.0e-15):
            raise ValueError(
                "working model scope must not exceed the absolute posture limit"
            )
        total_mass_scope = float(
            self.maximum_abs_total_tank_mass_deviation_from_reference_kg
        )
        if not np.isfinite(total_mass_scope) or total_mass_scope <= 0.0:
            raise ValueError(
                "maximum_abs_total_tank_mass_deviation_from_reference_kg "
                "must be positive and finite"
            )
        if self.controller.ballast_modes is not self.ballast_modes and not np.allclose(
            self.controller.ballast_modes.tank_mass_basis,
            self.ballast_modes.tank_mass_basis,
            rtol=0.0,
            atol=1.0e-12,
        ):
            raise ValueError("controller and control cycle must use the same ballast modes")
        block_duration = float(self.execution_config.block_duration_s)
        if not np.isclose(
            self.controller.block_model.block_duration_s,
            block_duration,
            rtol=0.0,
            atol=1.0e-12,
        ):
            raise ValueError("MPC and physical execution block durations must match")
        expected_leads = block_duration * np.arange(
            1,
            self.planning_source.horizon_steps + 1,
            dtype=float,
        )
        if not np.allclose(
            self.planning_source.lead_times_s,
            expected_leads,
            rtol=0.0,
            atol=1.0e-9,
        ):
            raise ValueError("planning lead times must align with MPC block endpoints")
        snapshot = assemble_ballast_model_snapshot(
            runtime_assembly=self.runtime_assembly,
            actual_tank_masses_kg=self.execution_state.actual_masses_kg,
            reference_tank_masses_kg=reference,
            tank_capacities_kg=capacities,
            tank_coordinates_m=coordinates,
        )
        for name in ("mass", "damping", "restoring_stiffness"):
            if not np.allclose(
                getattr(self.controller.block_model.matrices, name),
                getattr(snapshot.matrices, name),
                rtol=1.0e-12,
                atol=1.0e-9,
            ):
                raise ValueError(
                    "MPC block model must use the current actual ballast snapshot"
                )
        expected_first_load = (
            self.planner_first_block_rotor_load.generalized_load_platform
            + snapshot.incremental_ballast_load
            + wave
            + other
        )
        if not np.allclose(loads[0], expected_first_load, rtol=0.0, atol=1.0e-7):
            raise ValueError(
                "planner first-block physical load must match the MPC disturbance"
            )
        for name, value in (
            ("reference_tank_masses_kg", reference),
            ("tank_capacities_kg", capacities),
            ("tank_coordinates_m", coordinates),
            ("generalized_disturbance_loads", loads),
            ("current_wave_load", wave),
            ("current_other_load", other),
            ("maximum_abs_roll_pitch_rad", posture_limit),
            (
                "maximum_abs_working_model_scope_roll_pitch_rad",
                model_scope_limit,
            ),
        ):
            object.__setattr__(self, name, value)
        object.__setattr__(
            self,
            "maximum_abs_total_tank_mass_deviation_from_reference_kg",
            total_mass_scope,
        )


def _execution_states_match(
    left: ExecutionRolloutState,
    right: ExecutionRolloutState,
) -> bool:
    array_fields = (
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
    )
    return all(
        np.allclose(
            np.asarray(getattr(left, name), dtype=float),
            np.asarray(getattr(right, name), dtype=float),
            rtol=0.0,
            atol=1.0e-12,
        )
        for name in array_fields
    )


@dataclass(frozen=True)
class PreviewMPCCandidateObjective:
    """Unified full-horizon cost after one physical first-block replay."""

    running_posture_by_block: Any
    tank_movement_by_block: Any
    tank_throughput_by_block: Any
    movement_change_by_block: Any
    terminal_posture: float
    posture_slack: float
    posture_slack_rad: Any

    def __post_init__(self) -> None:
        arrays: dict[str, np.ndarray] = {}
        block_count: int | None = None
        for name in (
            "running_posture_by_block",
            "tank_movement_by_block",
            "tank_throughput_by_block",
            "movement_change_by_block",
        ):
            value = np.asarray(getattr(self, name), dtype=float)
            if value.ndim != 1 or value.size == 0 or not np.all(np.isfinite(value)):
                raise ValueError(f"{name} must be a non-empty finite vector")
            if np.any(value < -1.0e-12):
                raise ValueError(f"{name} must be non-negative")
            if block_count is None:
                block_count = int(value.size)
            elif value.size != block_count:
                raise ValueError("candidate objective arrays must have equal length")
            arrays[name] = np.maximum(value, 0.0)
        for name, value in arrays.items():
            object.__setattr__(self, name, _readonly_finite_array(name, value, value.shape))
        for name in ("terminal_posture", "posture_slack"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
            object.__setattr__(self, name, value)
        object.__setattr__(
            self,
            "posture_slack_rad",
            _readonly_finite_array(
                "posture_slack_rad",
                self.posture_slack_rad,
                (2,),
            ),
        )
        if np.any(self.posture_slack_rad < 0.0):
            raise ValueError("posture_slack_rad must be non-negative")

    @property
    def total(self) -> float:
        return float(
            np.sum(self.running_posture_by_block)
            + np.sum(self.tank_movement_by_block)
            + np.sum(self.tank_throughput_by_block)
            + np.sum(self.movement_change_by_block)
            + self.terminal_posture
            + self.posture_slack
        )


@dataclass(frozen=True)
class PreviewMPCLifecycleCandidate:
    """One target operation and its common-origin physical and MPC evidence."""

    lifecycle: PreviewMPCTargetLifecycle
    execution_request: ExecutionRolloutRequest
    physical_precheck: PreviewMPCPhysicalPrecheck
    reached_first_block_projection: DifferentialModeProjection
    tail_evaluation: PreviewMPCResult | None
    tail_initial_tank_masses_kg: Any
    objective: PreviewMPCCandidateObjective
    total_mass_scope_limit_kg: float
    maximum_abs_total_mass_deviation_from_reference_kg: float
    total_mass_within_scope: bool
    projection_residual_tolerance_kg: float
    projection_residual_within_tolerance: bool
    first_block_endpoint_posture_error_rad: Any
    first_block_peak_posture_error_rad: Any
    maximum_normalized_working_model_scope: float
    horizon_within_working_model_scope: bool
    first_block_maximum_normalized_working_model_scope: float
    first_block_within_working_model_scope: bool
    eligible: bool
    rejection_reason: str | None = None

    def __post_init__(self) -> None:
        lifecycle = PreviewMPCTargetLifecycle(self.lifecycle)
        object.__setattr__(self, "lifecycle", lifecycle)
        if not isinstance(self.execution_request, ExecutionRolloutRequest):
            raise TypeError("execution_request must be ExecutionRolloutRequest")
        if not isinstance(self.physical_precheck, PreviewMPCPhysicalPrecheck):
            raise TypeError("physical_precheck must be PreviewMPCPhysicalPrecheck")
        if self.physical_precheck.execution_request != self.execution_request:
            raise ValueError("physical precheck must evaluate the candidate request")
        if not isinstance(
            self.reached_first_block_projection,
            DifferentialModeProjection,
        ):
            raise TypeError(
                "reached_first_block_projection must be DifferentialModeProjection"
            )
        if self.tail_evaluation is not None and not isinstance(
            self.tail_evaluation, PreviewMPCResult
        ):
            raise TypeError("tail_evaluation must be PreviewMPCResult or None")
        tail_initial_masses = _readonly_finite_array(
            "tail_initial_tank_masses_kg",
            self.tail_initial_tank_masses_kg,
            (3,),
        )
        if not isinstance(self.objective, PreviewMPCCandidateObjective):
            raise TypeError("objective must be PreviewMPCCandidateObjective")
        for name in (
            "total_mass_scope_limit_kg",
            "maximum_abs_total_mass_deviation_from_reference_kg",
            "projection_residual_tolerance_kg",
            "maximum_normalized_working_model_scope",
            "first_block_maximum_normalized_working_model_scope",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
            object.__setattr__(self, name, value)
        for name in (
            "total_mass_within_scope",
            "projection_residual_within_tolerance",
            "horizon_within_working_model_scope",
            "first_block_within_working_model_scope",
        ):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be a boolean")
        for name in (
            "first_block_endpoint_posture_error_rad",
            "first_block_peak_posture_error_rad",
        ):
            value = np.asarray(getattr(self, name), dtype=float)
            if value.shape != (2,) or not np.all(np.isfinite(value)):
                raise ValueError(f"{name} must contain two finite values")
            value = np.array(value, copy=True)
            value.setflags(write=False)
            object.__setattr__(self, name, value)
        if type(self.eligible) is not bool:
            raise ValueError("eligible must be a boolean")
        reason = (
            None
            if self.rejection_reason is None
            else str(self.rejection_reason).strip()
        )
        if self.eligible and reason:
            raise ValueError("eligible candidates cannot have a rejection reason")
        if not self.eligible and not reason:
            raise ValueError("ineligible candidates require a rejection reason")
        object.__setattr__(self, "rejection_reason", reason)
        object.__setattr__(
            self,
            "tail_initial_tank_masses_kg",
            tail_initial_masses,
        )

    @property
    def tail_horizon_objective(self) -> float | None:
        if self.tail_evaluation is None:
            return None
        return float(self.tail_evaluation.objective_breakdown.absolute_total)

    @property
    def comparison_objective(self) -> float:
        return self.objective.total

    @property
    def previewed_first_block_tank_delta_kg(self) -> np.ndarray:
        return self.physical_precheck.execution_path.actual_tank_mass_delta_kg

    @property
    def unrepresented_common_mass_change_kg(self) -> float:
        return float(
            3.0
            * self.reached_first_block_projection.common_tank_mass_delta_kg
        )

    @property
    def common_mass_scope_limit_kg(self) -> float:
        """Compatibility alias for reports written before the explicit scope."""

        return self.total_mass_scope_limit_kg

    @property
    def common_mass_within_projection_scope(self) -> bool:
        """Compatibility alias for the explicit total-mass scope result."""

        return self.total_mass_within_scope

    @property
    def projection_residual_norm_kg(self) -> float:
        return float(
            np.linalg.norm(
                self.reached_first_block_projection.residual_tank_mass_deltas_kg
            )
        )


@dataclass(frozen=True)
class PreviewMPCControlCycleResult:
    """Selected target operation and the realised first-block state handoff."""

    planning_source: PreviewMPCPlanningSource
    free_plan: PreviewMPCResult
    actuator_envelope: PreviewActuatorEnvelope
    candidates: tuple[PreviewMPCLifecycleCandidate, ...]
    selected_lifecycle: PreviewMPCTargetLifecycle
    selection_reason: PreviewMPCSelectionReason
    execution_request: ExecutionRolloutRequest
    execution_path: PhysicalExecutionPlatformPath
    selected_from_horizon_feasible_options: bool
    selected_from_planner_load_precheck_safe_options: bool
    selected_horizon_within_working_model_scope: bool
    selected_first_block_within_working_model_scope: bool
    realised_execution_within_working_model_scope: bool

    def __post_init__(self) -> None:
        if not isinstance(self.planning_source, PreviewMPCPlanningSource):
            raise TypeError("planning_source must be PreviewMPCPlanningSource")
        if not isinstance(self.free_plan, PreviewMPCResult):
            raise TypeError("free_plan must be PreviewMPCResult")
        if not isinstance(self.actuator_envelope, PreviewActuatorEnvelope):
            raise TypeError("actuator_envelope must be PreviewActuatorEnvelope")
        candidates = tuple(self.candidates)
        if len(candidates) < 2 or not all(
            isinstance(item, PreviewMPCLifecycleCandidate) for item in candidates
        ):
            raise ValueError("candidates must contain evaluated lifecycle options")
        lifecycles = tuple(item.lifecycle for item in candidates)
        if len(set(lifecycles)) != len(lifecycles):
            raise ValueError("each lifecycle may appear at most once")
        selected = PreviewMPCTargetLifecycle(self.selected_lifecycle)
        if selected not in lifecycles:
            raise ValueError("selected_lifecycle must identify one candidate")
        reason = PreviewMPCSelectionReason(self.selection_reason)
        if not isinstance(self.execution_request, ExecutionRolloutRequest):
            raise TypeError("execution_request must be ExecutionRolloutRequest")
        if not isinstance(self.execution_path, PhysicalExecutionPlatformPath):
            raise TypeError("execution_path must be PhysicalExecutionPlatformPath")
        if type(self.selected_from_horizon_feasible_options) is not bool:
            raise ValueError(
                "selected_from_horizon_feasible_options must be a boolean"
            )
        if type(self.selected_from_planner_load_precheck_safe_options) is not bool:
            raise ValueError(
                "selected_from_planner_load_precheck_safe_options must be a boolean"
            )
        for name in (
            "selected_horizon_within_working_model_scope",
            "selected_first_block_within_working_model_scope",
            "realised_execution_within_working_model_scope",
        ):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be a boolean")
        candidate = next(item for item in candidates if item.lifecycle is selected)
        if candidate.execution_request != self.execution_request:
            raise ValueError("execution request must match the selected candidate")
        if (
            self.selected_from_horizon_feasible_options
            and not candidate.eligible
        ):
            raise ValueError("a horizon-feasible selection must be eligible")
        if (
            self.selected_from_planner_load_precheck_safe_options
            != candidate.physical_precheck.posture_precheck_passed
        ):
            raise ValueError(
                "planner-load precheck status must match the selected candidate"
            )
        if (
            self.selected_horizon_within_working_model_scope
            != candidate.horizon_within_working_model_scope
        ):
            raise ValueError(
                "selected horizon model-scope status must match the selected candidate"
            )
        if (
            self.selected_first_block_within_working_model_scope
            != candidate.first_block_within_working_model_scope
        ):
            raise ValueError(
                "selected first-block model-scope status must match the selected candidate"
            )
        actual_start = self.execution_path.substeps[0]
        candidate_start = candidate.physical_precheck.execution_path.substeps[0]
        if not np.allclose(
            actual_start.start_platform_state.position,
            candidate_start.start_platform_state.position,
            rtol=0.0,
            atol=1.0e-12,
        ) or not np.allclose(
            actual_start.start_platform_state.velocity,
            candidate_start.start_platform_state.velocity,
            rtol=0.0,
            atol=1.0e-12,
        ):
            raise ValueError(
                "actual execution and candidate precheck must share a platform origin"
            )
        if not _execution_states_match(
            actual_start.start_execution_state,
            candidate_start.start_execution_state,
        ):
            raise ValueError(
                "actual execution and candidate precheck must share an actuator origin"
            )
        if not np.allclose(
            actual_start.execution_step.requested_target_kg,
            candidate_start.execution_step.requested_target_kg,
            rtol=0.0,
            atol=1.0e-9,
        ):
            raise ValueError(
                "actual execution path must retain the selected candidate request"
            )
        object.__setattr__(self, "candidates", candidates)
        object.__setattr__(self, "selected_lifecycle", selected)
        object.__setattr__(self, "selection_reason", reason)

    @property
    def selected_candidate(self) -> PreviewMPCLifecycleCandidate:
        return next(
            item
            for item in self.candidates
            if item.lifecycle is self.selected_lifecycle
        )

    @property
    def next_platform_state(self) -> IncrementalState:
        return self.execution_path.final_platform_state

    @property
    def next_execution_state(self) -> ExecutionRolloutState:
        return self.execution_path.final_execution_state


class PreviewMPCNoSafeCandidateError(RuntimeError):
    """Raised when no planner-load-safe, model-supported action can be committed."""

    def __init__(
        self,
        candidates: tuple[PreviewMPCLifecycleCandidate, ...],
    ) -> None:
        self.candidates = tuple(candidates)
        minimum_violation = min(
            (
                item.physical_precheck.maximum_normalized_posture
                for item in self.candidates
            ),
            default=float("inf"),
        )
        super().__init__(
            "no lifecycle candidate passed both the planner-load posture "
            "precheck and working-model-scope check; "
            "the research controller did not commit a pump request "
            f"(minimum normalized posture={minimum_violation:.6g})"
        )


def _candidate_request(
    *,
    lifecycle: PreviewMPCTargetLifecycle,
    execution_state: ExecutionRolloutState,
    free_plan: PreviewMPCResult,
) -> ExecutionRolloutRequest | None:
    if lifecycle is PreviewMPCTargetLifecycle.CONTINUE_EXISTING_TARGET:
        return ExecutionRolloutRequest.track(
            execution_state.primary_target_masses_kg
        )
    if lifecycle is PreviewMPCTargetLifecycle.RELEASE_TO_ACTUAL:
        return ExecutionRolloutRequest.release_to_current()
    if free_plan.success:
        return ExecutionRolloutRequest.track(
            free_plan.current_target_tank_masses_kg
        )
    return None


def _physical_path_posture_samples(
    path: PhysicalExecutionPlatformPath,
) -> tuple[np.ndarray, np.ndarray]:
    elapsed = [0.0]
    posture = [path.substeps[0].start_platform_state.position[[3, 4]]]
    cumulative = 0.0
    for step in path.substeps:
        cumulative += float(step.duration_s)
        elapsed.append(cumulative)
        posture.append(step.next_platform_state.position[[3, 4]])
    return np.asarray(elapsed, dtype=float), np.asarray(posture, dtype=float)


def _running_posture_cost(
    *,
    elapsed_time_s: np.ndarray,
    posture_rad: np.ndarray,
    controller: PreviewMPCController,
    full_horizon_duration_s: float,
) -> float:
    weights = controller.weights
    density = 0.5 * (
        weights.roll * posture_rad[:, 0] ** 2
        + weights.pitch * posture_rad[:, 1] ** 2
    )
    return float(np.trapezoid(density, elapsed_time_s) / full_horizon_duration_s)


def _candidate_objective(
    *,
    precheck: PreviewMPCPhysicalPrecheck,
    tail_evaluation: PreviewMPCResult | None,
    controller: PreviewMPCController,
    reached_delta_kg: np.ndarray,
    previous_tank_movement_kg: np.ndarray,
    total_block_count: int,
    first_block_throughput_kg: float,
) -> PreviewMPCCandidateObjective:
    full_duration_s = float(
        controller.running_posture_reference_duration_s
        or total_block_count * controller.block_model.block_duration_s
    )
    first_time, first_posture = _physical_path_posture_samples(
        precheck.execution_path
    )
    running_posture = [
        _running_posture_cost(
            elapsed_time_s=first_time,
            posture_rad=first_posture,
            controller=controller,
            full_horizon_duration_s=full_duration_s,
        )
    ]
    all_posture = [first_posture]
    terminal_posture_rad = first_posture[-1]
    if tail_evaluation is not None:
        tail_trajectory = tail_evaluation.predicted_trajectory
        tail_start = tail_trajectory.initial_state.platform.position[[3, 4]]
        previous_posture = np.asarray(tail_start, dtype=float)
        all_posture.append(previous_posture[None, :])
        for block in tail_trajectory.blocks:
            block_posture = np.column_stack((block.roll_rad, block.pitch_rad))
            block_time = np.concatenate(([0.0], block.elapsed_time_s))
            block_path = np.vstack((previous_posture, block_posture))
            running_posture.append(
                _running_posture_cost(
                    elapsed_time_s=block_time,
                    posture_rad=block_path,
                    controller=controller,
                    full_horizon_duration_s=full_duration_s,
                )
            )
            all_posture.append(block_posture)
            previous_posture = block_posture[-1]
        terminal_posture_rad = previous_posture

    weights = controller.weights
    constraints = controller.constraints
    sampled_posture = np.vstack(all_posture)
    nominal_limits = np.array(
        [
            constraints.maximum_abs_roll_rad,
            constraints.maximum_abs_pitch_rad,
        ],
        dtype=float,
    )
    posture_slack_rad = np.max(
        np.maximum(np.abs(sampled_posture) - nominal_limits, 0.0),
        axis=0,
    )
    first_movement = 0.5 * weights.tank_movement * float(
        reached_delta_kg @ reached_delta_kg
    )
    first_throughput = weights.tank_throughput * float(first_block_throughput_kg)
    first_change = 0.5 * weights.movement_change * float(
        (reached_delta_kg - previous_tank_movement_kg)
        @ (reached_delta_kg - previous_tank_movement_kg)
    )
    if tail_evaluation is None:
        tail_movement = np.empty(0, dtype=float)
        tail_throughput = np.empty(0, dtype=float)
        tail_change = np.empty(0, dtype=float)
    else:
        tail_breakdown = tail_evaluation.objective_breakdown
        tail_movement = tail_breakdown.tank_movement_by_block
        tail_throughput = tail_breakdown.tank_throughput_by_block
        tail_change = tail_breakdown.movement_change_by_block
    terminal_posture = 0.5 * float(
        weights.terminal_roll * terminal_posture_rad[0] ** 2
        + weights.terminal_pitch * terminal_posture_rad[1] ** 2
    )
    posture_slack = 0.5 * weights.posture_slack * float(
        posture_slack_rad @ posture_slack_rad
    )
    return PreviewMPCCandidateObjective(
        running_posture_by_block=np.asarray(running_posture, dtype=float),
        tank_movement_by_block=np.concatenate(([first_movement], tail_movement)),
        tank_throughput_by_block=np.concatenate(
            ([first_throughput], tail_throughput)
        ),
        movement_change_by_block=np.concatenate(([first_change], tail_change)),
        terminal_posture=terminal_posture,
        posture_slack=posture_slack,
        posture_slack_rad=posture_slack_rad,
    )


def _candidate_sampled_posture(
    *,
    precheck: PreviewMPCPhysicalPrecheck,
    tail_evaluation: PreviewMPCResult | None,
) -> np.ndarray:
    """Return physical-prefix and predicted-tail roll/pitch samples."""

    _, prefix_posture = _physical_path_posture_samples(precheck.execution_path)
    samples = [prefix_posture]
    if tail_evaluation is not None:
        for block in tail_evaluation.predicted_trajectory.blocks:
            samples.append(np.column_stack((block.roll_rad, block.pitch_rad)))
    return np.vstack(samples)


def _evaluate_candidate(
    *,
    lifecycle: PreviewMPCTargetLifecycle,
    request: ExecutionRolloutRequest,
    free_plan: PreviewMPCResult,
    controller: PreviewMPCController,
    ballast_modes: ThreeTankDifferentialModes,
    platform_state: IncrementalState,
    execution_state: ExecutionRolloutState,
    execution_config: ExecutionRolloutConfig,
    runtime_assembly: BallastRuntimeAssembly,
    reference_tank_masses_kg: Any,
    tank_capacities_kg: Any,
    tank_coordinates_m: Any,
    generalized_disturbance_loads: Any,
    actuator_envelope: Any,
    planner_first_block_rotor_load: RotorGeneralizedLoad,
    current_wave_load: Any,
    current_other_load: Any,
    maximum_abs_roll_pitch_rad: Any,
    maximum_abs_working_model_scope_roll_pitch_rad: Any,
    maximum_abs_total_tank_mass_deviation_from_reference_kg: float,
) -> PreviewMPCLifecycleCandidate:
    precheck = precheck_preview_mpc_request(
        source=lifecycle.value,
        request=request,
        maximum_abs_roll_pitch_rad=np.asarray(
            maximum_abs_roll_pitch_rad, dtype=float
        ),
        platform_state=platform_state,
        execution_state=execution_state,
        execution_config=execution_config,
        runtime_assembly=runtime_assembly,
        reference_tank_masses_kg=reference_tank_masses_kg,
        tank_capacities_kg=tank_capacities_kg,
        tank_coordinates_m=tank_coordinates_m,
        current_rotor_load=planner_first_block_rotor_load,
        current_wave_load=current_wave_load,
        current_other_load=current_other_load,
    )
    reached_delta = (
        precheck.execution_path.final_execution_state.actual_masses_kg
        - execution_state.actual_masses_kg
    )
    projection = ballast_modes.project_tank_mass_deltas(reached_delta)
    initial_snapshot = assemble_ballast_model_snapshot(
        runtime_assembly=runtime_assembly,
        actual_tank_masses_kg=execution_state.actual_masses_kg,
        reference_tank_masses_kg=reference_tank_masses_kg,
        tank_capacities_kg=tank_capacities_kg,
        tank_coordinates_m=tank_coordinates_m,
    )
    tail_initial_masses = np.asarray(
        precheck.execution_path.final_execution_state.actual_masses_kg,
        dtype=float,
    )
    _, prefix_posture = _physical_path_posture_samples(
        precheck.execution_path
    )
    nominal_posture_limits = np.array(
        [
            controller.constraints.maximum_abs_roll_rad,
            controller.constraints.maximum_abs_pitch_rad,
        ],
        dtype=float,
    )
    prefix_posture_slack_rad = np.max(
        np.maximum(np.abs(prefix_posture) - nominal_posture_limits, 0.0),
        axis=0,
    )
    tail_evaluation: PreviewMPCResult | None = None
    total_block_count = np.asarray(generalized_disturbance_loads).shape[0]
    if total_block_count > 1:
        tail_snapshot = assemble_ballast_model_snapshot(
            runtime_assembly=runtime_assembly,
            actual_tank_masses_kg=tail_initial_masses,
            reference_tank_masses_kg=reference_tank_masses_kg,
            tank_capacities_kg=tank_capacities_kg,
            tank_coordinates_m=tank_coordinates_m,
        )
        external_tail_loads = (
            np.asarray(generalized_disturbance_loads, dtype=float)[1:]
            - initial_snapshot.incremental_ballast_load[None, :]
        )
        tail_loads = (
            external_tail_loads
            + tail_snapshot.incremental_ballast_load[None, :]
        )
        tail_block_model = PreviewBlockModel(
            matrices=tail_snapshot.matrices,
            generalized_load_per_mode_kg=(
                ballast_modes.generalized_load_per_mode_kg
            ),
            block_duration_s=controller.block_model.block_duration_s,
            sample_fractions=controller.block_model.sample_fractions,
        )
        full_horizon_reference_duration_s = float(
            controller.running_posture_reference_duration_s
            or total_block_count * controller.block_model.block_duration_s
        )
        tail_controller = PreviewMPCController(
            block_model=tail_block_model,
            ballast_modes=ballast_modes,
            weights=controller.weights,
            constraints=controller.constraints,
            running_posture_reference_duration_s=(
                full_horizon_reference_duration_s
            ),
        )
        tail_actuator_envelope = assemble_preview_actuator_envelope(
            execution_state=(
                precheck.execution_path.final_execution_state
            ),
            execution_config=execution_config,
            tank_capacities_kg=tank_capacities_kg,
            ballast_modes=ballast_modes,
        )
        tail_evaluation = tail_controller.solve(
            initial_platform_state=(
                precheck.execution_path.final_platform_state
            ),
            actual_tank_masses_kg=tail_initial_masses,
            tank_capacities_kg=tank_capacities_kg,
            generalized_disturbance_loads=tail_loads,
            actuator_envelope=tail_actuator_envelope,
            previous_tank_movement_kg=reached_delta,
            minimum_posture_slack_rad=np.minimum(
                prefix_posture_slack_rad,
                controller.constraints.maximum_posture_slack_rad,
            ),
        )

    objective = _candidate_objective(
        precheck=precheck,
        tail_evaluation=tail_evaluation,
        controller=controller,
        reached_delta_kg=reached_delta,
        previous_tank_movement_kg=np.asarray(
            execution_state.previous_block_tank_mass_delta_kg,
            dtype=float,
        ),
        total_block_count=total_block_count,
        first_block_throughput_kg=(
            precheck.execution_path.transferred_volume_m3
            * execution_config.water_density_kg_m3
        ),
    )
    total_mass_scope_limit_kg = float(
        maximum_abs_total_tank_mass_deviation_from_reference_kg
    )
    unrepresented_common_mass_change_kg = float(
        3.0 * projection.common_tank_mass_delta_kg
    )
    tank_mass_states = np.vstack(
        [precheck.execution_path.start_actual_tank_masses_kg]
        + [
            step.next_execution_state.actual_masses_kg
            for step in precheck.execution_path.substeps
        ]
    )
    total_mass_deviation = np.sum(
        tank_mass_states - np.asarray(reference_tank_masses_kg, dtype=float),
        axis=1,
    )
    maximum_abs_total_mass_deviation = float(
        np.max(np.abs(total_mass_deviation))
    )
    total_mass_within_scope = bool(
        maximum_abs_total_mass_deviation <= total_mass_scope_limit_kg + 1.0e-9
    )
    projection_residual_norm_kg = float(
        np.linalg.norm(projection.residual_tank_mass_deltas_kg)
    )
    projection_residual_tolerance_kg = float(
        1.0e-9 * max(1.0, float(np.linalg.norm(reached_delta)))
    )
    projection_residual_within_tolerance = bool(
        projection_residual_norm_kg <= projection_residual_tolerance_kg
    )
    model_scope_limits = np.asarray(
        maximum_abs_working_model_scope_roll_pitch_rad,
        dtype=float,
    )
    sampled_candidate_posture = _candidate_sampled_posture(
        precheck=precheck,
        tail_evaluation=tail_evaluation,
    )
    maximum_normalized_working_model_scope = float(
        np.max(np.abs(sampled_candidate_posture) / model_scope_limits[None, :])
    )
    horizon_within_working_model_scope = bool(
        maximum_normalized_working_model_scope <= 1.0 + 1.0e-9
    )
    first_block_maximum_normalized_working_model_scope = float(
        np.max(
            precheck.peak_abs_roll_pitch_rad
            / model_scope_limits
        )
    )
    first_block_within_working_model_scope = bool(
        first_block_maximum_normalized_working_model_scope <= 1.0 + 1.0e-9
    )
    physical_peak = precheck.peak_abs_roll_pitch_rad
    physical_endpoint = precheck.execution_path.final_platform_state.position[[3, 4]]
    free_block = free_plan.predicted_trajectory.blocks[0]
    planned_peak = np.array(
        [
            np.max(np.abs(free_block.roll_rad)),
            np.max(np.abs(free_block.pitch_rad)),
        ]
    )
    planned_endpoint = free_block.endpoint.platform.position[[3, 4]]
    rejection_reasons: list[str] = []
    if not precheck.posture_precheck_passed:
        rejection_reasons.append(precheck.posture_precheck_reason)
    if not horizon_within_working_model_scope:
        rejection_reasons.append("candidate_posture_exceeds_working_model_scope")
    if tail_evaluation is not None and not tail_evaluation.success:
        rejection_reasons.append(
            f"physical_prefix_tail_horizon_infeasible: {tail_evaluation.message}"
        )
    if np.any(
        objective.posture_slack_rad
        > controller.constraints.maximum_posture_slack_rad + 1.0e-10
    ):
        rejection_reasons.append(
            "combined_horizon_posture_slack_exceeds_limit"
        )
    if not total_mass_within_scope:
        rejection_reasons.append(
            "total_tank_mass_deviation_exceeds_scope"
        )
    if not projection_residual_within_tolerance:
        rejection_reasons.append(
            "first_block_tank_delta_projection_residual_exceeds_tolerance"
        )
    return PreviewMPCLifecycleCandidate(
        lifecycle=lifecycle,
        execution_request=request,
        physical_precheck=precheck,
        reached_first_block_projection=projection,
        tail_evaluation=tail_evaluation,
        tail_initial_tank_masses_kg=tail_initial_masses,
        objective=objective,
        total_mass_scope_limit_kg=total_mass_scope_limit_kg,
        maximum_abs_total_mass_deviation_from_reference_kg=(
            maximum_abs_total_mass_deviation
        ),
        total_mass_within_scope=total_mass_within_scope,
        projection_residual_tolerance_kg=projection_residual_tolerance_kg,
        projection_residual_within_tolerance=(
            projection_residual_within_tolerance
        ),
        first_block_endpoint_posture_error_rad=(
            planned_endpoint - physical_endpoint
        ),
        first_block_peak_posture_error_rad=(
            planned_peak - physical_peak
        ),
        maximum_normalized_working_model_scope=(
            maximum_normalized_working_model_scope
        ),
        horizon_within_working_model_scope=(
            horizon_within_working_model_scope
        ),
        first_block_maximum_normalized_working_model_scope=(
            first_block_maximum_normalized_working_model_scope
        ),
        first_block_within_working_model_scope=(
            first_block_within_working_model_scope
        ),
        eligible=not rejection_reasons,
        rejection_reason=("; ".join(rejection_reasons) or None),
    )


def _physical_burden_key(
    candidate: PreviewMPCLifecycleCandidate,
) -> tuple[float, float, int, int]:
    path = candidate.physical_precheck.execution_path
    return (
        float(path.transferred_volume_m3),
        float(path.aggregate_pump_active_time_s),
        int(path.pump_start_count),
        _LIFECYCLE_TIE_ORDER[candidate.lifecycle],
    )


def _select_candidate(
    candidates: tuple[PreviewMPCLifecycleCandidate, ...],
) -> tuple[
    PreviewMPCLifecycleCandidate,
    PreviewMPCSelectionReason,
    bool,
    bool,
]:
    eligible = tuple(item for item in candidates if item.eligible)
    if eligible:
        minimum_objective = min(item.comparison_objective for item in eligible)
        objective_tolerance = max(
            _OBJECTIVE_EQUIVALENCE_ATOL,
            abs(minimum_objective) * _OBJECTIVE_EQUIVALENCE_RTOL,
        )
        near_best = tuple(
            item
            for item in eligible
            if item.comparison_objective <= minimum_objective + objective_tolerance
        )
        selected = min(near_best, key=_physical_burden_key)
        reason = (
            PreviewMPCSelectionReason.LOWEST_COMMON_HORIZON_OBJECTIVE
            if len(near_best) == 1
            else PreviewMPCSelectionReason.TIE_BROKEN_BY_PHYSICAL_EXECUTION
        )
        return selected, reason, True, True

    # Receding-horizon execution commits only the physically replayed first
    # block. A tail-scope violation therefore permits an explicitly degraded
    # first-block-safe choice, but never normal horizon ranking.
    physically_safe = tuple(
        item
        for item in candidates
        if (
            item.physical_precheck.posture_precheck_passed
            and item.first_block_within_working_model_scope
            and item.total_mass_within_scope
            and item.projection_residual_within_tolerance
        )
    )
    if physically_safe:
        selected = min(
            physically_safe,
            key=lambda item: (
                item.physical_precheck.maximum_normalized_posture,
                item.physical_precheck.terminal_normalized_posture,
                *_physical_burden_key(item),
            ),
        )
        return (
            selected,
            PreviewMPCSelectionReason.DEGRADED_FIRST_BLOCK_SAFE_NO_FULLY_ELIGIBLE_OPTION,
            False,
            True,
        )

    raise PreviewMPCNoSafeCandidateError(candidates)


def run_preview_mpc_control_cycle(
    cycle_input: PreviewMPCControlCycleInput,
) -> PreviewMPCControlCycleResult:
    """Plan, compare target lifecycles, and execute one physical block.

    The realised plant load is never used during candidate construction or
    selection.  It is applied only after the request has been selected.
    """

    if not isinstance(cycle_input, PreviewMPCControlCycleInput):
        raise TypeError("cycle_input must be PreviewMPCControlCycleInput")
    controller = cycle_input.controller
    ballast_modes = cycle_input.ballast_modes
    platform_state = cycle_input.platform_state
    execution_state = cycle_input.execution_state
    execution_config = cycle_input.execution_config
    runtime_assembly = cycle_input.runtime_assembly
    reference_tank_masses_kg = cycle_input.reference_tank_masses_kg
    tank_capacities_kg = cycle_input.tank_capacities_kg
    tank_coordinates_m = cycle_input.tank_coordinates_m
    generalized_disturbance_loads = cycle_input.generalized_disturbance_loads
    planner_first_block_rotor_load = cycle_input.planner_first_block_rotor_load
    execution_first_block_rotor_load = cycle_input.execution_first_block_rotor_load
    execution_first_block_rotor_load_substeps = (
        cycle_input.execution_first_block_rotor_load_substeps
    )
    current_wave_load = cycle_input.current_wave_load
    current_other_load = cycle_input.current_other_load
    maximum_abs_roll_pitch_rad = cycle_input.maximum_abs_roll_pitch_rad
    maximum_abs_working_model_scope_roll_pitch_rad = (
        cycle_input.maximum_abs_working_model_scope_roll_pitch_rad
    )
    maximum_abs_total_tank_mass_deviation_from_reference_kg = (
        cycle_input.maximum_abs_total_tank_mass_deviation_from_reference_kg
    )
    realised_rotor_load = execution_first_block_rotor_load

    actuator_envelope = assemble_preview_actuator_envelope(
        execution_state=execution_state,
        execution_config=execution_config,
        tank_capacities_kg=tank_capacities_kg,
        ballast_modes=ballast_modes,
    )
    free_plan = controller.solve(
        initial_platform_state=platform_state,
        actual_tank_masses_kg=execution_state.actual_masses_kg,
        tank_capacities_kg=tank_capacities_kg,
        generalized_disturbance_loads=generalized_disturbance_loads,
        actuator_envelope=actuator_envelope,
        previous_tank_movement_kg=(
            execution_state.previous_block_tank_mass_delta_kg
        ),
    )

    candidates: list[PreviewMPCLifecycleCandidate] = []
    for lifecycle in PreviewMPCTargetLifecycle:
        request = _candidate_request(
            lifecycle=lifecycle,
            execution_state=execution_state,
            free_plan=free_plan,
        )
        if request is None:
            continue
        candidates.append(
            _evaluate_candidate(
                lifecycle=lifecycle,
                request=request,
                free_plan=free_plan,
                controller=controller,
                ballast_modes=ballast_modes,
                platform_state=platform_state,
                execution_state=execution_state,
                execution_config=execution_config,
                runtime_assembly=runtime_assembly,
                reference_tank_masses_kg=reference_tank_masses_kg,
                tank_capacities_kg=tank_capacities_kg,
                tank_coordinates_m=tank_coordinates_m,
                generalized_disturbance_loads=generalized_disturbance_loads,
                actuator_envelope=actuator_envelope,
                planner_first_block_rotor_load=planner_first_block_rotor_load,
                current_wave_load=current_wave_load,
                current_other_load=current_other_load,
                maximum_abs_roll_pitch_rad=maximum_abs_roll_pitch_rad,
                maximum_abs_working_model_scope_roll_pitch_rad=(
                    maximum_abs_working_model_scope_roll_pitch_rad
                ),
                maximum_abs_total_tank_mass_deviation_from_reference_kg=(
                    maximum_abs_total_tank_mass_deviation_from_reference_kg
                ),
            )
        )
    candidate_tuple = tuple(candidates)
    selected, reason, horizon_feasible, physically_safe = _select_candidate(
        candidate_tuple
    )

    execution_path = advance_physical_execution_platform_path(
        platform_state=platform_state,
        execution_state=execution_state,
        execution_request=selected.execution_request,
        execution_config=execution_config,
        runtime_assembly=runtime_assembly,
        reference_tank_masses_kg=reference_tank_masses_kg,
        tank_capacities_kg=tank_capacities_kg,
        tank_coordinates_m=tank_coordinates_m,
        rotor_load=realised_rotor_load,
        rotor_load_substeps=execution_first_block_rotor_load_substeps,
        wave_load=current_wave_load,
        other_load=current_other_load,
        duration_s=execution_config.block_duration_s,
    )
    _, realised_posture = _physical_path_posture_samples(execution_path)
    realised_execution_within_working_model_scope = bool(
        np.all(
            np.abs(realised_posture)
            <= maximum_abs_working_model_scope_roll_pitch_rad[None, :] + 1.0e-9
        )
    )

    return PreviewMPCControlCycleResult(
        planning_source=cycle_input.planning_source,
        free_plan=free_plan,
        actuator_envelope=actuator_envelope,
        candidates=candidate_tuple,
        selected_lifecycle=selected.lifecycle,
        selection_reason=reason,
        execution_request=selected.execution_request,
        execution_path=execution_path,
        selected_from_horizon_feasible_options=horizon_feasible,
        selected_from_planner_load_precheck_safe_options=physically_safe,
        selected_horizon_within_working_model_scope=(
            selected.horizon_within_working_model_scope
        ),
        selected_first_block_within_working_model_scope=(
            selected.first_block_within_working_model_scope
        ),
        realised_execution_within_working_model_scope=(
            realised_execution_within_working_model_scope
        ),
    )


__all__ = [
    "PreviewMPCCandidateObjective",
    "PreviewMPCControlCycleInput",
    "PreviewMPCControlCycleResult",
    "PreviewMPCLifecycleCandidate",
    "PreviewMPCNoSafeCandidateError",
    "PreviewMPCPlanningSource",
    "PreviewMPCSelectionReason",
    "PreviewMPCTargetLifecycle",
    "run_preview_mpc_control_cycle",
]
