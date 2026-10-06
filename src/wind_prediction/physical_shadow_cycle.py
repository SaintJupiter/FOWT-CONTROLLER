"""Carry one selected physical shadow path into fresh next-cycle facts.

This module deliberately stays below a real closed-loop controller boundary.
The caller owns every new-cycle observation, forecast, environmental load and
rotor operating state.  Once those fresh facts have been assembled, this
module checks that their origin state follows the previously selected
current-block *shadow* rollout, then applies the existing abstaining lifecycle
selection rule.  It never creates a forecast, a target, an admission result or
a fallback action.

Consequently, a sequence built with this module is a rolling counterfactual
shadow chain.  It is useful for verifying time and state handoff semantics,
but is not evidence of a deployed closed-loop controller or of performance.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
import math

import numpy as np

from fowt_platform.incremental import IncrementalState
from fowt_platform.ballast_snapshot import BallastModelSnapshot

from .execution_rollout import ExecutionRolloutRequest, ExecutionRolloutState
from .first_interval_physical_decision_input import FirstIntervalPhysicalDecisionInput
from .forecast_execution_block_lifecycle_rollout import (
    ForecastExecutionBlockLifecycleRollout,
)
from .physical_forecast_admission import PhysicalForecastAdmission
from .physical_lifecycle_selection import (
    PhysicalLifecycleSelectionFact,
    PhysicalPostureLimits,
    assemble_physical_lifecycle_selection_for_current_block,
)


_STATE_TOLERANCE = 1.0e-10
_MASS_TOLERANCE_KG = 1.0e-8
_TIME_TOLERANCE_S = 1.0e-9


def _state_matches(left: IncrementalState, right: IncrementalState) -> bool:
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


def _execution_states_match(
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


def _same_static_ballast_definition(
    left: BallastModelSnapshot,
    right: BallastModelSnapshot,
) -> bool:
    """Compare ballast quantities that must not change inside one shadow chain.

    Actual tank masses intentionally change between cycles.  The reference
    masses, capacities and tank geometry instead define the frozen incremental
    model, so a change in any of them starts a different physical study.
    """

    for attribute in (
        "reference_tank_masses_kg",
        "tank_capacities_kg",
        "tank_coordinates_m",
    ):
        if not np.allclose(
            getattr(left, attribute),
            getattr(right, attribute),
            rtol=0.0,
            atol=_MASS_TOLERANCE_KG,
        ):
            return False
    return bool(
        math.isclose(
            left.gravity_m_s2,
            right.gravity_m_s2,
            rel_tol=0.0,
            abs_tol=_STATE_TOLERANCE,
        )
        and left.runtime_provenance == right.runtime_provenance
    )


def _parse_origin_time(value: str) -> datetime:
    try:
        return datetime.fromisoformat(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "shadow-cycle origin times must be ISO-8601 timestamps"
        ) from exc


@dataclass(frozen=True)
class PhysicalShadowCycleRequest:
    """Caller-owned fresh facts and declared limits for one current block.

    ``decision_input`` must be newly assembled for this block.  It may use a
    prior selected shadow endpoint only as its initial platform and pump state;
    it must not reuse the prior trajectory or lifecycle facts.
    """

    decision_input: FirstIntervalPhysicalDecisionInput
    posture_limits: PhysicalPostureLimits
    forecast_admission: PhysicalForecastAdmission

    def __post_init__(self) -> None:
        if not isinstance(self.decision_input, FirstIntervalPhysicalDecisionInput):
            raise TypeError(
                "decision_input must be FirstIntervalPhysicalDecisionInput"
            )
        if not isinstance(self.posture_limits, PhysicalPostureLimits):
            raise TypeError("posture_limits must be PhysicalPostureLimits")
        if not isinstance(self.forecast_admission, PhysicalForecastAdmission):
            raise TypeError("forecast_admission must be PhysicalForecastAdmission")
        if self.forecast_admission.decision_input is not self.decision_input:
            raise ValueError("forecast_admission must retain the decision_input")


@dataclass(frozen=True)
class PhysicalShadowCycleOutcome:
    """Selected or abstained result for one current-block shadow calculation."""

    request: PhysicalShadowCycleRequest
    selection: PhysicalLifecycleSelectionFact
    selected_rollout: ForecastExecutionBlockLifecycleRollout | None

    def __post_init__(self) -> None:
        if not isinstance(self.request, PhysicalShadowCycleRequest):
            raise TypeError("request must be PhysicalShadowCycleRequest")
        if not isinstance(self.selection, PhysicalLifecycleSelectionFact):
            raise TypeError("selection must be PhysicalLifecycleSelectionFact")
        if self.selection.decision_input is not self.request.decision_input:
            raise ValueError("selection must retain the request decision_input")
        if self.selection.posture_limits is not self.request.posture_limits:
            raise ValueError("selection must retain the request posture_limits")
        if self.selection.forecast_admission is not self.request.forecast_admission:
            raise ValueError("selection must retain the request forecast_admission")

        if self.selection.selected_lifecycle is None:
            if self.selected_rollout is not None:
                raise ValueError(
                    "an abstained shadow cycle must not expose a selected rollout"
                )
            return

        if not isinstance(
            self.selected_rollout,
            ForecastExecutionBlockLifecycleRollout,
        ):
            raise ValueError("a selected shadow cycle requires its selected rollout")
        if self.selected_rollout is not self.selection.require_selected_rollout():
            raise ValueError("selected_rollout must be the selection-owned shadow rollout")

    @property
    def is_selected(self) -> bool:
        """Return whether this block produced a path that may seed a shadow handoff."""

        return self.selected_rollout is not None

    def require_selected_rollout(self) -> ForecastExecutionBlockLifecycleRollout:
        """Return the selected shadow endpoint or require explicit abstention handling."""

        if self.selected_rollout is None:
            raise ValueError(
                "shadow cycle abstained; no platform or pump state may be handed to "
                "a later shadow cycle"
            )
        return self.selected_rollout

    def require_selected_execution_request(self) -> ExecutionRolloutRequest:
        """Return the selected current-block command without handing off shadow state."""

        return self.selection.require_selected_execution_request()


def _validate_fresh_handoff(
    *,
    previous: PhysicalShadowCycleOutcome,
    request: PhysicalShadowCycleRequest,
) -> None:
    """Check that fresh caller facts start from the selected prior shadow state."""

    selected = previous.require_selected_rollout()
    prior_input = previous.request.decision_input
    current_input = request.decision_input

    if current_input.trajectory is prior_input.trajectory:
        raise ValueError("next shadow cycle must use a fresh forecast trajectory")
    if current_input.lifecycle_facts is prior_input.lifecycle_facts:
        raise ValueError("next shadow cycle must use fresh target-lifecycle facts")
    if current_input.runtime_assembly is not selected.runtime_assembly:
        raise ValueError("next shadow cycle must retain the same runtime assembly")
    if not _same_static_ballast_definition(
        selected.trajectory.platform_snapshot,
        current_input.trajectory.platform_snapshot,
    ):
        raise ValueError(
            "next shadow cycle must retain the same reference tank definition"
        )
    if (
        current_input.lifecycle_facts.continue_existing.execution_config
        != selected.lifecycle_trace.execution_config
    ):
        raise ValueError(
            "next shadow cycle must retain the same pump execution configuration"
        )

    prior_origin = _parse_origin_time(prior_input.trajectory.initial_state_time)
    current_origin = _parse_origin_time(current_input.trajectory.initial_state_time)
    if (prior_origin.tzinfo is None) != (current_origin.tzinfo is None):
        raise ValueError("shadow-cycle origin times must use one timezone convention")
    expected_origin = prior_origin + timedelta(seconds=selected.duration_s)
    if abs((current_origin - expected_origin).total_seconds()) > _TIME_TOLERANCE_S:
        raise ValueError("next shadow-cycle origin time must follow the selected block")

    if current_input.lifecycle_facts.execution_start_time != (
        current_input.trajectory.initial_state_time
    ):
        raise ValueError(
            "next shadow-cycle lifecycle facts must start at the trajectory origin time"
        )
    if not _state_matches(
        current_input.trajectory.initial_state,
        selected.final_platform_state,
    ):
        raise ValueError(
            "next shadow-cycle initial platform state must match the selected prior state"
        )
    next_execution_state = (
        current_input.lifecycle_facts.continue_existing.execution_start_state
    )
    if not _execution_states_match(next_execution_state, selected.final_execution_state):
        raise ValueError(
            "next shadow-cycle pump state must match the selected prior pump state"
        )
    if not np.allclose(
        current_input.trajectory.platform_snapshot.actual_tank_masses_kg,
        selected.final_execution_state.actual_masses_kg,
        rtol=0.0,
        atol=_MASS_TOLERANCE_KG,
    ):
        raise ValueError(
            "next shadow-cycle tank snapshot must use the selected prior actual masses"
        )


def assemble_physical_shadow_cycle(
    *,
    request: PhysicalShadowCycleRequest,
    previous: PhysicalShadowCycleOutcome | None = None,
) -> PhysicalShadowCycleOutcome:
    """Select one current-block shadow path from fresh caller-owned facts.

    When ``previous`` is supplied, the next input must be a newly assembled
    cycle whose origin time, platform state and full pump state follow the
    prior selected shadow rollout.  If the current selection abstains, the
    result deliberately exposes no state for another handoff.
    """

    if not isinstance(request, PhysicalShadowCycleRequest):
        raise TypeError("request must be PhysicalShadowCycleRequest")
    if previous is not None:
        if not isinstance(previous, PhysicalShadowCycleOutcome):
            raise TypeError(
                "previous must be PhysicalShadowCycleOutcome when provided"
            )
        _validate_fresh_handoff(previous=previous, request=request)

    selection = assemble_physical_lifecycle_selection_for_current_block(
        decision_input=request.decision_input,
        posture_limits=request.posture_limits,
        forecast_admission=request.forecast_admission,
    )
    selected_rollout = (
        selection.require_selected_rollout()
        if selection.selected_lifecycle is not None
        else None
    )
    return PhysicalShadowCycleOutcome(
        request=request,
        selection=selection,
        selected_rollout=selected_rollout,
    )


__all__ = [
    "PhysicalShadowCycleOutcome",
    "PhysicalShadowCycleRequest",
    "assemble_physical_shadow_cycle",
]
