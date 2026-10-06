"""Compare reachable ballast states at one fixed forecast state.

This module intentionally sits between factual actuator preview and a future
decision layer.  ``PhysicalOptionRhsComparison`` is limited to the earlier
relative-load endpoint route carried by ``PhysicalExecutionCompensationDiagnostic``:
the caller has already supplied one lead and one endpoint fraction.  The
state-conditioned full-RHS endpoint follows the separate
``PhysicalTargetLifecycleRhsDiagnostic`` route after its lifecycle operation
has been defined.

Both records are local counterfactuals only.  They are not future closed-loop
trajectories, static equilibria, safety results, ballast demands, or candidate
rankings.  In particular, neither record selects an endpoint fraction or names
it as a strengthen, release, or reversal action.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from fowt_platform.ballast_snapshot import (
    BallastModelSnapshot,
    BallastRuntimeAssembly,
    assemble_ballast_model_snapshot,
)
from fowt_platform.incremental import (
    IncrementalLoads,
    IncrementalPlatformModel,
    IncrementalStateDerivative,
)

from .forecast_platform_rhs_diagnostic import ForecastPlatformRhsPointDiagnostic
from .execution_rollout import (
    ExecutionRolloutState,
    ExecutionRolloutStep,
    simulate_execution_step,
)
from .physical_execution_compensation import PhysicalExecutionCompensationDiagnostic
from .physical_endpoint_preview_binding import validate_endpoint_preview_at_trajectory_lead
from .physical_target_lifecycle import PhysicalTargetLifecycleTrace


_LOAD_TOLERANCE = 1.0e-7
_STATE_TOLERANCE = 1.0e-10
_MASS_TOLERANCE = 1.0e-8


def _readonly_six(name: str, value: Any) -> np.ndarray:
    try:
        vector = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must have shape (6,) and finite values") from exc
    if vector.shape != (6,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must have shape (6,) and finite values")
    result = np.array(vector, dtype=float, copy=True)
    result.setflags(write=False)
    return result


def _snapshots_have_same_dynamics(
    left: BallastModelSnapshot,
    right: BallastModelSnapshot,
) -> bool:
    if not np.allclose(
        left.actual_tank_masses_kg,
        right.actual_tank_masses_kg,
        rtol=0.0,
        atol=_MASS_TOLERANCE,
    ) or not np.allclose(
        left.reference_tank_masses_kg,
        right.reference_tank_masses_kg,
        rtol=0.0,
        atol=_MASS_TOLERANCE,
    ) or not np.allclose(
        left.tank_capacities_kg,
        right.tank_capacities_kg,
        rtol=0.0,
        atol=_MASS_TOLERANCE,
    ) or not np.allclose(
        left.tank_coordinates_m,
        right.tank_coordinates_m,
        rtol=0.0,
        atol=1.0e-12,
    ) or not np.isclose(
        left.gravity_m_s2,
        right.gravity_m_s2,
        rtol=0.0,
        atol=1.0e-12,
    ) or left.runtime_provenance != right.runtime_provenance:
        return False
    if not np.allclose(
        left.incremental_ballast_load,
        right.incremental_ballast_load,
        rtol=0.0,
        atol=_LOAD_TOLERANCE,
    ):
        return False
    for name in (
        "mass",
        "damping",
        "hydrostatic_stiffness",
        "mooring_stiffness",
        "weight_stiffness",
    ):
        if not np.allclose(
            getattr(left.matrices, name),
            getattr(right.matrices, name),
            rtol=0.0,
            atol=_LOAD_TOLERANCE,
        ):
            return False
    return True


def _execution_states_match(
    left: ExecutionRolloutState,
    right: ExecutionRolloutState,
) -> bool:
    """Compare the complete pump state retained across one physical preview."""

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
    """Keep lifecycle facts tied to the actuator semantics that produced them."""

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


def _validate_lifecycle_trace_at_rhs_point(
    *,
    rhs_point: ForecastPlatformRhsPointDiagnostic,
    lifecycle_trace: PhysicalTargetLifecycleTrace,
) -> None:
    """Bind one executed lifecycle trace to the current first forecast lead."""

    if not isinstance(rhs_point, ForecastPlatformRhsPointDiagnostic):
        raise TypeError("rhs_point must be ForecastPlatformRhsPointDiagnostic")
    if not isinstance(lifecycle_trace, PhysicalTargetLifecycleTrace):
        raise TypeError("lifecycle_trace must be PhysicalTargetLifecycleTrace")
    if rhs_point.lead_index != 0:
        raise ValueError(
            "lifecycle rhs diagnostic is currently limited to the first forecast lead"
        )
    trajectory = rhs_point.trajectory
    if lifecycle_trace.execution_start_time != trajectory.initial_state_time:
        raise ValueError(
            "lifecycle execution start time must match the forecast trajectory origin"
        )
    if not np.isclose(
        lifecycle_trace.execution_duration_s,
        rhs_point.lead_time_s,
        rtol=0.0,
        atol=1.0e-9,
    ):
        raise ValueError(
            "lifecycle execution duration must match the first rhs forecast lead"
        )
    if not np.allclose(
        lifecycle_trace.execution_start_state.actual_masses_kg,
        trajectory.platform_snapshot.actual_tank_masses_kg,
        rtol=0.0,
        atol=_MASS_TOLERANCE,
    ):
        raise ValueError(
            "lifecycle execution start masses must match the rhs trajectory snapshot"
        )
    if not np.allclose(
        trajectory.platform_snapshot.tank_capacities_kg,
        float(lifecycle_trace.execution_config.tank_capacity_kg),
        rtol=0.0,
        atol=_MASS_TOLERANCE,
    ):
        raise ValueError(
            "lifecycle execution tank capacity must match the rhs trajectory snapshot"
        )
    preview = lifecycle_trace.source_preview
    if preview is not None:
        validate_endpoint_preview_at_trajectory_lead(
            preview=preview,
            trajectory=trajectory,
            lead_index=rhs_point.lead_index,
            lead_time_s=rhs_point.lead_time_s,
            first_lead_only=True,
        )

    local_config = replace(
        lifecycle_trace.execution_config,
        block_duration_s=lifecycle_trace.execution_duration_s,
    )
    recomputed = simulate_execution_step(
        lifecycle_trace.execution_start_state,
        lifecycle_trace.execution_request,
        local_config,
    )
    if not _execution_steps_match(lifecycle_trace.execution_step, recomputed):
        raise ValueError(
            "lifecycle execution step must be reproducible from its pump state and request"
        )


def _rebuild_reachable_rhs(
    *,
    rhs_point: ForecastPlatformRhsPointDiagnostic,
    actual_final_tank_masses_kg: Any,
    runtime_assembly: BallastRuntimeAssembly,
) -> tuple[
    BallastModelSnapshot,
    IncrementalLoads,
    np.ndarray,
    np.ndarray,
    IncrementalStateDerivative,
]:
    """Rebuild one same-time RHS after an already executed pump trace."""

    if not isinstance(rhs_point, ForecastPlatformRhsPointDiagnostic):
        raise TypeError("rhs_point must be ForecastPlatformRhsPointDiagnostic")
    if not isinstance(runtime_assembly, BallastRuntimeAssembly):
        raise TypeError("runtime_assembly must be BallastRuntimeAssembly")

    baseline_snapshot = rhs_point.trajectory.platform_snapshot
    if runtime_assembly.provenance != baseline_snapshot.runtime_provenance:
        raise ValueError(
            "runtime_assembly provenance must match the rhs trajectory platform snapshot"
        )
    reconstructed_baseline = assemble_ballast_model_snapshot(
        runtime_assembly=runtime_assembly,
        actual_tank_masses_kg=baseline_snapshot.actual_tank_masses_kg,
        reference_tank_masses_kg=baseline_snapshot.reference_tank_masses_kg,
        tank_capacities_kg=baseline_snapshot.tank_capacities_kg,
        tank_coordinates_m=baseline_snapshot.tank_coordinates_m,
    )
    if not _snapshots_have_same_dynamics(baseline_snapshot, reconstructed_baseline):
        raise ValueError(
            "runtime_assembly must reconstruct the rhs trajectory platform snapshot"
        )
    reachable_snapshot = assemble_ballast_model_snapshot(
        runtime_assembly=runtime_assembly,
        actual_tank_masses_kg=actual_final_tank_masses_kg,
        reference_tank_masses_kg=baseline_snapshot.reference_tank_masses_kg,
        tank_capacities_kg=baseline_snapshot.tank_capacities_kg,
        tank_coordinates_m=baseline_snapshot.tank_coordinates_m,
    )
    reachable_loads = IncrementalLoads(
        wind=rhs_point.loads.wind,
        wave=rhs_point.loads.wave,
        ballast=reachable_snapshot.incremental_ballast_load,
        other=rhs_point.loads.other,
    )
    state = rhs_point.platform_state
    derivative = IncrementalPlatformModel(reachable_snapshot.matrices).derivative(
        state,
        reachable_loads,
    )
    passive = (
        -reachable_snapshot.matrices.damping @ state.velocity
        - reachable_snapshot.matrices.restoring_stiffness @ state.position
    )
    dynamic_rhs = reachable_loads.total + passive
    return reachable_snapshot, reachable_loads, passive, dynamic_rhs, derivative


@dataclass(frozen=True)
class PhysicalOptionRhsComparison:
    """One reachable physical endpoint compared at its matching forecast state.

    ``rhs_point`` is the frozen-tank baseline at one forecast endpoint.
    ``compensation`` carries the selected endpoint fraction and the actual tank
    masses reached by the pump before the same lead time.  The reachable
    snapshot and derivative hold the future platform state and non-ballast
    loads fixed, changing only the tank-dependent model terms.
    """

    rhs_point: ForecastPlatformRhsPointDiagnostic
    compensation: PhysicalExecutionCompensationDiagnostic
    runtime_assembly: BallastRuntimeAssembly
    reachable_snapshot: BallastModelSnapshot
    reachable_loads: IncrementalLoads
    reachable_passive_generalized_load: Any
    reachable_dynamic_rhs_generalized_load: Any
    reachable_state_derivative: IncrementalStateDerivative
    delta_dynamic_rhs_from_frozen_baseline: Any
    delta_acceleration_from_frozen_baseline: Any

    def __post_init__(self) -> None:
        if not isinstance(self.rhs_point, ForecastPlatformRhsPointDiagnostic):
            raise TypeError("rhs_point must be ForecastPlatformRhsPointDiagnostic")
        if not isinstance(
            self.compensation, PhysicalExecutionCompensationDiagnostic
        ):
            raise TypeError(
                "compensation must be PhysicalExecutionCompensationDiagnostic"
            )
        if not isinstance(self.runtime_assembly, BallastRuntimeAssembly):
            raise TypeError("runtime_assembly must be BallastRuntimeAssembly")
        if not isinstance(self.reachable_snapshot, BallastModelSnapshot):
            raise TypeError("reachable_snapshot must be BallastModelSnapshot")
        if not isinstance(self.reachable_loads, IncrementalLoads):
            raise TypeError("reachable_loads must be IncrementalLoads")
        if not isinstance(self.reachable_state_derivative, IncrementalStateDerivative):
            raise TypeError(
                "reachable_state_derivative must be IncrementalStateDerivative"
            )

        preview = self.compensation.preview
        baseline_snapshot = self.rhs_point.trajectory.platform_snapshot
        if preview.load_assembly is not self.rhs_point.trajectory.load_assembly:
            raise ValueError(
                "physical endpoint preview must use the rhs trajectory load assembly"
            )
        if preview.lead_index != self.rhs_point.lead_index or not np.isclose(
            preview.lead_time_s,
            self.rhs_point.lead_time_s,
            rtol=0.0,
            atol=1.0e-9,
        ):
            raise ValueError(
                "physical endpoint preview must use the matching rhs forecast lead"
            )
        if self.compensation.platform_snapshot is not baseline_snapshot:
            raise ValueError(
                "physical compensation must use the rhs trajectory platform snapshot"
            )
        if not np.isclose(
            preview.reachability.execution_duration_s,
            self.rhs_point.lead_time_s,
            rtol=0.0,
            atol=1.0e-9,
        ):
            raise ValueError(
                "physical option comparison requires pump preview through the matching lead time"
            )
        (
            expected_reachable_snapshot,
            expected_reachable_loads,
            expected_passive,
            expected_dynamic_rhs,
            expected_derivative,
        ) = _rebuild_reachable_rhs(
            rhs_point=self.rhs_point,
            actual_final_tank_masses_kg=self.compensation.execution_final_tank_masses_kg,
            runtime_assembly=self.runtime_assembly,
        )
        if not _snapshots_have_same_dynamics(
            self.reachable_snapshot,
            expected_reachable_snapshot,
        ):
            raise ValueError(
                "reachable_snapshot must be reconstructed from the rhs trajectory runtime assembly"
            )
        if not np.allclose(
            self.reachable_snapshot.actual_tank_masses_kg,
            self.compensation.execution_final_tank_masses_kg,
            rtol=0.0,
            atol=_MASS_TOLERANCE,
        ):
            raise ValueError(
                "reachable_snapshot must use the pump preview final tank masses"
            )
        if not np.allclose(
            self.reachable_loads.wind,
            expected_reachable_loads.wind,
            rtol=0.0,
            atol=_LOAD_TOLERANCE,
        ) or not np.allclose(
            self.reachable_loads.wave,
            expected_reachable_loads.wave,
            rtol=0.0,
            atol=_LOAD_TOLERANCE,
        ) or not np.allclose(
            self.reachable_loads.other,
            expected_reachable_loads.other,
            rtol=0.0,
            atol=_LOAD_TOLERANCE,
        ):
            raise ValueError(
                "reachable comparison must preserve the rhs point non-ballast loads"
            )
        if not np.allclose(
            self.reachable_loads.ballast,
            expected_reachable_loads.ballast,
            rtol=0.0,
            atol=_LOAD_TOLERANCE,
        ):
            raise ValueError(
                "reachable loads must use the reachable snapshot ballast load"
            )

        passive = _readonly_six(
            "reachable_passive_generalized_load",
            self.reachable_passive_generalized_load,
        )
        dynamic_rhs = _readonly_six(
            "reachable_dynamic_rhs_generalized_load",
            self.reachable_dynamic_rhs_generalized_load,
        )
        delta_rhs = _readonly_six(
            "delta_dynamic_rhs_from_frozen_baseline",
            self.delta_dynamic_rhs_from_frozen_baseline,
        )
        delta_acceleration = _readonly_six(
            "delta_acceleration_from_frozen_baseline",
            self.delta_acceleration_from_frozen_baseline,
        )
        if not np.allclose(passive, expected_passive, rtol=0.0, atol=_LOAD_TOLERANCE):
            raise ValueError(
                "reachable_passive_generalized_load must match the reachable snapshot"
            )
        if not np.allclose(
            dynamic_rhs,
            expected_dynamic_rhs,
            rtol=0.0,
            atol=_LOAD_TOLERANCE,
        ):
            raise ValueError(
                "reachable_dynamic_rhs_generalized_load must equal external plus passive load"
            )
        if not np.allclose(
            self.reachable_state_derivative.position_rate,
            expected_derivative.position_rate,
            rtol=0.0,
            atol=_STATE_TOLERANCE,
        ) or not np.allclose(
            self.reachable_state_derivative.velocity_rate,
            expected_derivative.velocity_rate,
            rtol=0.0,
            atol=_STATE_TOLERANCE,
        ):
            raise ValueError(
                "reachable_state_derivative must match the reachable snapshot model"
            )
        if not np.allclose(
            delta_rhs,
            dynamic_rhs - self.rhs_point.dynamic_rhs_generalized_load,
            rtol=0.0,
            atol=_LOAD_TOLERANCE,
        ):
            raise ValueError(
                "delta_dynamic_rhs_from_frozen_baseline must use the matching rhs point"
            )
        if not np.allclose(
            delta_acceleration,
            (
                self.reachable_state_derivative.velocity_rate
                - self.rhs_point.state_derivative.velocity_rate
            ),
            rtol=0.0,
            atol=_STATE_TOLERANCE,
        ):
            raise ValueError(
                "delta_acceleration_from_frozen_baseline must use matching state derivatives"
            )

        object.__setattr__(self, "reachable_passive_generalized_load", passive)
        object.__setattr__(self, "reachable_dynamic_rhs_generalized_load", dynamic_rhs)
        object.__setattr__(self, "delta_dynamic_rhs_from_frozen_baseline", delta_rhs)
        object.__setattr__(self, "delta_acceleration_from_frozen_baseline", delta_acceleration)


@dataclass(frozen=True)
class PhysicalTargetLifecycleRhsDiagnostic:
    """Same-time six-DOF response facts for one target-lifecycle operation.

    The lifecycle trace already records how one existing target operation would
    be executed from the current pump state.  This record preserves that
    operation and evaluates its reached tank state at the identical first
    forecast endpoint used by the frozen no-action trajectory.  It is not an
    option score, an action selector, a safety result, or a closed-loop rollout.
    """

    rhs_point: ForecastPlatformRhsPointDiagnostic
    lifecycle_trace: PhysicalTargetLifecycleTrace
    runtime_assembly: BallastRuntimeAssembly
    reachable_snapshot: BallastModelSnapshot
    reachable_loads: IncrementalLoads
    reachable_passive_generalized_load: Any
    reachable_dynamic_rhs_generalized_load: Any
    reachable_state_derivative: IncrementalStateDerivative
    delta_dynamic_rhs_from_frozen_baseline: Any
    delta_acceleration_from_frozen_baseline: Any

    def __post_init__(self) -> None:
        _validate_lifecycle_trace_at_rhs_point(
            rhs_point=self.rhs_point,
            lifecycle_trace=self.lifecycle_trace,
        )
        if not isinstance(self.runtime_assembly, BallastRuntimeAssembly):
            raise TypeError("runtime_assembly must be BallastRuntimeAssembly")
        if not isinstance(self.reachable_snapshot, BallastModelSnapshot):
            raise TypeError("reachable_snapshot must be BallastModelSnapshot")
        if not isinstance(self.reachable_loads, IncrementalLoads):
            raise TypeError("reachable_loads must be IncrementalLoads")
        if not isinstance(self.reachable_state_derivative, IncrementalStateDerivative):
            raise TypeError(
                "reachable_state_derivative must be IncrementalStateDerivative"
            )

        expected_reachable_snapshot, _, _, _, _ = _rebuild_reachable_rhs(
            rhs_point=self.rhs_point,
            actual_final_tank_masses_kg=self.lifecycle_trace.actual_final_masses_kg,
            runtime_assembly=self.runtime_assembly,
        )
        if not _snapshots_have_same_dynamics(
            self.reachable_snapshot,
            expected_reachable_snapshot,
        ):
            raise ValueError(
                "reachable_snapshot must be reconstructed from the rhs trajectory runtime assembly"
            )

        if not np.allclose(
            self.reachable_snapshot.actual_tank_masses_kg,
            self.lifecycle_trace.actual_final_masses_kg,
            rtol=0.0,
            atol=_MASS_TOLERANCE,
        ):
            raise ValueError(
                "reachable_snapshot must use the lifecycle trace final tank masses"
            )
        if not np.allclose(
            self.reachable_loads.wind,
            self.rhs_point.loads.wind,
            rtol=0.0,
            atol=_LOAD_TOLERANCE,
        ) or not np.allclose(
            self.reachable_loads.wave,
            self.rhs_point.loads.wave,
            rtol=0.0,
            atol=_LOAD_TOLERANCE,
        ) or not np.allclose(
            self.reachable_loads.other,
            self.rhs_point.loads.other,
            rtol=0.0,
            atol=_LOAD_TOLERANCE,
        ):
            raise ValueError(
                "lifecycle rhs diagnostic must preserve the rhs point non-ballast loads"
            )
        if not np.allclose(
            self.reachable_loads.ballast,
            self.reachable_snapshot.incremental_ballast_load,
            rtol=0.0,
            atol=_LOAD_TOLERANCE,
        ):
            raise ValueError(
                "lifecycle rhs diagnostic must use the reached snapshot ballast load"
            )

        passive = _readonly_six(
            "reachable_passive_generalized_load",
            self.reachable_passive_generalized_load,
        )
        dynamic_rhs = _readonly_six(
            "reachable_dynamic_rhs_generalized_load",
            self.reachable_dynamic_rhs_generalized_load,
        )
        delta_rhs = _readonly_six(
            "delta_dynamic_rhs_from_frozen_baseline",
            self.delta_dynamic_rhs_from_frozen_baseline,
        )
        delta_acceleration = _readonly_six(
            "delta_acceleration_from_frozen_baseline",
            self.delta_acceleration_from_frozen_baseline,
        )
        state = self.rhs_point.platform_state
        matrices = self.reachable_snapshot.matrices
        expected_passive = (
            -matrices.damping @ state.velocity
            - matrices.restoring_stiffness @ state.position
        )
        expected_dynamic_rhs = self.reachable_loads.total + expected_passive
        expected_derivative = IncrementalPlatformModel(matrices).derivative(
            state,
            self.reachable_loads,
        )
        if not np.allclose(passive, expected_passive, rtol=0.0, atol=_LOAD_TOLERANCE):
            raise ValueError(
                "reachable_passive_generalized_load must match the reached snapshot"
            )
        if not np.allclose(
            dynamic_rhs,
            expected_dynamic_rhs,
            rtol=0.0,
            atol=_LOAD_TOLERANCE,
        ):
            raise ValueError(
                "reachable_dynamic_rhs_generalized_load must equal external plus passive load"
            )
        if not np.allclose(
            self.reachable_state_derivative.position_rate,
            expected_derivative.position_rate,
            rtol=0.0,
            atol=_STATE_TOLERANCE,
        ) or not np.allclose(
            self.reachable_state_derivative.velocity_rate,
            expected_derivative.velocity_rate,
            rtol=0.0,
            atol=_STATE_TOLERANCE,
        ):
            raise ValueError(
                "reachable_state_derivative must match the reached snapshot model"
            )
        if not np.allclose(
            matrices.mass @ self.reachable_state_derivative.velocity_rate,
            dynamic_rhs,
            rtol=0.0,
            atol=_LOAD_TOLERANCE,
        ):
            raise ValueError(
                "reachable acceleration must satisfy the reached state equation"
            )
        if not np.allclose(
            delta_rhs,
            dynamic_rhs - self.rhs_point.dynamic_rhs_generalized_load,
            rtol=0.0,
            atol=_LOAD_TOLERANCE,
        ):
            raise ValueError(
                "delta_dynamic_rhs_from_frozen_baseline must use the matching rhs point"
            )
        if not np.allclose(
            delta_acceleration,
            (
                self.reachable_state_derivative.velocity_rate
                - self.rhs_point.state_derivative.velocity_rate
            ),
            rtol=0.0,
            atol=_STATE_TOLERANCE,
        ):
            raise ValueError(
                "delta_acceleration_from_frozen_baseline must use matching state derivatives"
            )

        object.__setattr__(self, "reachable_passive_generalized_load", passive)
        object.__setattr__(self, "reachable_dynamic_rhs_generalized_load", dynamic_rhs)
        object.__setattr__(self, "delta_dynamic_rhs_from_frozen_baseline", delta_rhs)
        object.__setattr__(self, "delta_acceleration_from_frozen_baseline", delta_acceleration)


def diagnose_physical_target_lifecycle_rhs(
    *,
    rhs_point: ForecastPlatformRhsPointDiagnostic,
    lifecycle_trace: PhysicalTargetLifecycleTrace,
    runtime_assembly: BallastRuntimeAssembly,
) -> PhysicalTargetLifecycleRhsDiagnostic:
    """Evaluate one executed lifecycle trace at the matching first forecast lead.

    The function does not select between traces.  It only carries a concrete
    target operation through its already defined pump result and substitutes
    that reached tank state into the same local state equation.
    """

    if not isinstance(rhs_point, ForecastPlatformRhsPointDiagnostic):
        raise TypeError("rhs_point must be ForecastPlatformRhsPointDiagnostic")
    if not isinstance(lifecycle_trace, PhysicalTargetLifecycleTrace):
        raise TypeError("lifecycle_trace must be PhysicalTargetLifecycleTrace")
    if not isinstance(runtime_assembly, BallastRuntimeAssembly):
        raise TypeError("runtime_assembly must be BallastRuntimeAssembly")

    (
        reachable_snapshot,
        reachable_loads,
        passive,
        dynamic_rhs,
        derivative,
    ) = _rebuild_reachable_rhs(
        rhs_point=rhs_point,
        actual_final_tank_masses_kg=lifecycle_trace.actual_final_masses_kg,
        runtime_assembly=runtime_assembly,
    )
    return PhysicalTargetLifecycleRhsDiagnostic(
        rhs_point=rhs_point,
        lifecycle_trace=lifecycle_trace,
        runtime_assembly=runtime_assembly,
        reachable_snapshot=reachable_snapshot,
        reachable_loads=reachable_loads,
        reachable_passive_generalized_load=passive,
        reachable_dynamic_rhs_generalized_load=dynamic_rhs,
        reachable_state_derivative=derivative,
        delta_dynamic_rhs_from_frozen_baseline=(
            dynamic_rhs - rhs_point.dynamic_rhs_generalized_load
        ),
        delta_acceleration_from_frozen_baseline=(
            derivative.velocity_rate - rhs_point.state_derivative.velocity_rate
        ),
    )


def compare_reachable_physical_option_rhs(
    *,
    rhs_point: ForecastPlatformRhsPointDiagnostic,
    compensation: PhysicalExecutionCompensationDiagnostic,
    runtime_assembly: BallastRuntimeAssembly,
) -> PhysicalOptionRhsComparison:
    """Evaluate a pump-reachable endpoint at a matching frozen forecast point.

    The actuator preview must span exactly the selected forecast lead.  The
    function rebuilds both the baseline and reached tank snapshots from one
    runtime assembly, then checks that the supplied frozen trajectory snapshot
    is the same dynamic model before substituting the reached ballast state.
    No state is advanced and no option is ranked.
    """

    if not isinstance(rhs_point, ForecastPlatformRhsPointDiagnostic):
        raise TypeError("rhs_point must be ForecastPlatformRhsPointDiagnostic")
    if not isinstance(compensation, PhysicalExecutionCompensationDiagnostic):
        raise TypeError("compensation must be PhysicalExecutionCompensationDiagnostic")
    if not isinstance(runtime_assembly, BallastRuntimeAssembly):
        raise TypeError("runtime_assembly must be BallastRuntimeAssembly")

    (
        reachable_snapshot,
        reachable_loads,
        passive,
        dynamic_rhs,
        derivative,
    ) = _rebuild_reachable_rhs(
        rhs_point=rhs_point,
        actual_final_tank_masses_kg=compensation.execution_final_tank_masses_kg,
        runtime_assembly=runtime_assembly,
    )
    return PhysicalOptionRhsComparison(
        rhs_point=rhs_point,
        compensation=compensation,
        runtime_assembly=runtime_assembly,
        reachable_snapshot=reachable_snapshot,
        reachable_loads=reachable_loads,
        reachable_passive_generalized_load=passive,
        reachable_dynamic_rhs_generalized_load=dynamic_rhs,
        reachable_state_derivative=derivative,
        delta_dynamic_rhs_from_frozen_baseline=(
            dynamic_rhs - rhs_point.dynamic_rhs_generalized_load
        ),
        delta_acceleration_from_frozen_baseline=(
            derivative.velocity_rate - rhs_point.state_derivative.velocity_rate
        ),
    )


__all__ = [
    "PhysicalOptionRhsComparison",
    "PhysicalTargetLifecycleRhsDiagnostic",
    "compare_reachable_physical_option_rhs",
    "diagnose_physical_target_lifecycle_rhs",
]
