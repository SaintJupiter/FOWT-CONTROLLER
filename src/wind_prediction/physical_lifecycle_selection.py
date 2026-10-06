"""Constrained current-block selection from held physical lifecycle paths.

The physical chain produces three factual operations: continue the stored
target, release it to the measured tank masses, or track one source-bound
forecast endpoint.  Each operation is held through the available forecast
horizon before this module checks caller-supplied posture limits.  The result
may select only the already-replayed *current-block* request.  Later intervals
remain conditional counterfactuals and never become a multi-step plan.

The rule deliberately uses no forecast probability weight, pump-performance
score, or hidden preference between feasible paths.  It preserves an existing
request when that same request stays within the declared horizon envelope.  If
continuation does not, exactly one feasible alternative may be selected;
otherwise the record abstains for its caller to handle.

The result is an abstaining decision record.  If the physical facts do not
produce a unique order, the record carries no selected lifecycle and leaves the
fallback decision to its caller.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

from .first_interval_physical_decision_input import FirstIntervalPhysicalDecisionInput
from .execution_rollout import ExecutionRolloutRequest
from .forecast_execution_block_lifecycle_rollout import (
    ForecastExecutionBlockLifecycleRollout,
)
from .physical_forecast_admission import PhysicalForecastAdmission
from .physical_lifecycle_comparison import (
    PhysicalLifecycleComparison,
    PhysicalLifecycleHorizonComparison,
    PhysicalLifecycleHorizonOutcome,
    assemble_physical_lifecycle_comparison_for_current_block,
    assemble_physical_lifecycle_horizon_comparison,
)
from .physical_target_lifecycle import PhysicalTargetLifecycle


_PITCH_ROLL_INDICES = (4, 3)


def _readonly_positive_pair(name: str, value: Any) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain two positive finite values") from exc
    if array.shape != (2,) or not np.all(np.isfinite(array)) or np.any(array <= 0.0):
        raise ValueError(f"{name} must contain two positive finite values")
    result = np.array(array, dtype=float, copy=True)
    result.setflags(write=False)
    return result


def _nonempty_text(name: str, value: Any) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a non-empty string")
    result = value.strip()
    if not result:
        raise ValueError(f"{name} must be a non-empty string")
    return result


def _abs_pitch_roll(state: Any) -> tuple[np.ndarray, np.ndarray]:
    """Return absolute pitch and roll angle/rate in fixed ``(pitch, roll)`` order."""

    position = np.asarray(state.position[list(_PITCH_ROLL_INDICES)], dtype=float)
    velocity = np.asarray(state.velocity[list(_PITCH_ROLL_INDICES)], dtype=float)
    return np.abs(position), np.abs(velocity)


class PhysicalLifecycleSelectionReason(str, Enum):
    """Explicit reasons for a constrained physical lifecycle outcome."""

    FORECAST_NOT_ADMITTED = "forecast_not_admitted"
    INITIAL_STATE_OUTSIDE_SELECTION_SCOPE = "initial_state_outside_selection_scope"
    PRESERVE_EXISTING = "preserve_existing"
    UNIQUE_ELIGIBLE_ALTERNATIVE = "unique_eligible_alternative"
    NO_ELIGIBLE_LIFECYCLE = "no_eligible_lifecycle"
    MULTIPLE_ELIGIBLE_ALTERNATIVES = "multiple_eligible_alternatives"


@dataclass(frozen=True)
class PhysicalPostureLimits:
    """Caller-supplied path limits in the frozen low-order coordinate system.

    Fields use the fixed order ``(pitch, roll)``.  The record describes the
    limits adopted by a calling study or supervisory layer.  It does not claim
    they are certified safety limits.
    """

    max_abs_pitch_roll_rad: Any
    max_abs_pitch_roll_rate_rad_s: Any
    source: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "max_abs_pitch_roll_rad",
            _readonly_positive_pair(
                "max_abs_pitch_roll_rad",
                self.max_abs_pitch_roll_rad,
            ),
        )
        object.__setattr__(
            self,
            "max_abs_pitch_roll_rate_rad_s",
            _readonly_positive_pair(
                "max_abs_pitch_roll_rate_rad_s",
                self.max_abs_pitch_roll_rate_rad_s,
            ),
        )
        object.__setattr__(self, "source", _nonempty_text("source", self.source))


def _within_limits(
    *,
    outcome: PhysicalLifecycleHorizonOutcome,
    limits: PhysicalPostureLimits,
) -> bool:
    return bool(
        np.all(
            outcome.peak_abs_pitch_roll_rad
            <= limits.max_abs_pitch_roll_rad
        )
        and np.all(
            outcome.peak_abs_pitch_roll_rate_rad_s
            <= limits.max_abs_pitch_roll_rate_rad_s
        )
    )


def _origin_within_selection_scope(
    *,
    outcome: PhysicalLifecycleHorizonOutcome,
    limits: PhysicalPostureLimits,
) -> bool:
    """Check the shared origin without treating it as an alternative outcome."""

    position, velocity = _abs_pitch_roll(
        outcome.current_block_outcome.rollout.start_platform_state
    )
    return bool(
        np.all(position <= limits.max_abs_pitch_roll_rad)
        and np.all(velocity <= limits.max_abs_pitch_roll_rate_rad_s)
    )


def _derive_selection(
    *,
    limits: PhysicalPostureLimits,
    forecast_admitted: bool,
    continue_existing_outcome: PhysicalLifecycleHorizonOutcome,
    release_to_current_outcome: PhysicalLifecycleHorizonOutcome,
    new_track_outcome: PhysicalLifecycleHorizonOutcome | None,
) -> tuple[
    tuple[PhysicalTargetLifecycle, ...],
    PhysicalTargetLifecycle | None,
    PhysicalLifecycleSelectionReason,
]:
    if not forecast_admitted:
        return (), None, PhysicalLifecycleSelectionReason.FORECAST_NOT_ADMITTED
    if not _origin_within_selection_scope(
        outcome=continue_existing_outcome,
        limits=limits,
    ):
        return (), None, PhysicalLifecycleSelectionReason.INITIAL_STATE_OUTSIDE_SELECTION_SCOPE

    permitted: list[tuple[PhysicalTargetLifecycle, PhysicalLifecycleHorizonOutcome]] = []
    for lifecycle, outcome in (
        (PhysicalTargetLifecycle.CONTINUE_EXISTING, continue_existing_outcome),
        (PhysicalTargetLifecycle.RELEASE_TO_CURRENT, release_to_current_outcome),
        (PhysicalTargetLifecycle.NEW_TRACK, new_track_outcome),
    ):
        if outcome is not None and _within_limits(outcome=outcome, limits=limits):
            permitted.append((lifecycle, outcome))

    admissible = tuple(lifecycle for lifecycle, _ in permitted)
    continuation = next(
        (outcome for lifecycle, outcome in permitted if lifecycle is PhysicalTargetLifecycle.CONTINUE_EXISTING),
        None,
    )
    if continuation is not None:
        return (
            admissible,
            PhysicalTargetLifecycle.CONTINUE_EXISTING,
            PhysicalLifecycleSelectionReason.PRESERVE_EXISTING,
        )

    alternatives = [
        (lifecycle, outcome)
        for lifecycle, outcome in permitted
        if lifecycle is not PhysicalTargetLifecycle.CONTINUE_EXISTING
    ]
    if not alternatives:
        return admissible, None, PhysicalLifecycleSelectionReason.NO_ELIGIBLE_LIFECYCLE
    if len(alternatives) == 1:
        return (
            admissible,
            alternatives[0][0],
            PhysicalLifecycleSelectionReason.UNIQUE_ELIGIBLE_ALTERNATIVE,
        )
    return (
        admissible,
        None,
        PhysicalLifecycleSelectionReason.MULTIPLE_ELIGIBLE_ALTERNATIVES,
    )


@dataclass(frozen=True)
class PhysicalLifecycleSelectionFact:
    """One abstaining current-block selection from a held horizon comparison.

    Every horizon outcome starts from the same origin and retains the same
    current-block rollout.  ``forecast_admission`` records a caller-owned
    supervisory result and binds it to the forecast that generated these
    physical facts.  It authorizes the forecast-assisted branch but is not a
    continuous physical input or a hidden reliability weight.
    """

    decision_input: FirstIntervalPhysicalDecisionInput
    posture_limits: PhysicalPostureLimits
    forecast_admission: PhysicalForecastAdmission
    comparison: PhysicalLifecycleComparison
    horizon_comparison: PhysicalLifecycleHorizonComparison
    selection_eligible_lifecycles: tuple[PhysicalTargetLifecycle, ...] = field(init=False)
    selected_lifecycle: PhysicalTargetLifecycle | None = field(init=False)
    selection_reason: PhysicalLifecycleSelectionReason = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.decision_input, FirstIntervalPhysicalDecisionInput):
            raise TypeError("decision_input must be FirstIntervalPhysicalDecisionInput")
        if not isinstance(self.posture_limits, PhysicalPostureLimits):
            raise TypeError("posture_limits must be PhysicalPostureLimits")
        if not isinstance(self.forecast_admission, PhysicalForecastAdmission):
            raise TypeError("forecast_admission must be PhysicalForecastAdmission")
        if self.forecast_admission.decision_input is not self.decision_input:
            raise ValueError("forecast_admission must retain the decision_input")
        if not isinstance(self.comparison, PhysicalLifecycleComparison):
            raise TypeError("comparison must be PhysicalLifecycleComparison")
        if self.comparison.decision_input is not self.decision_input:
            raise ValueError("comparison must retain the decision_input")
        if self.comparison.forecast_admission is not self.forecast_admission:
            raise ValueError("comparison must retain the forecast_admission")
        if not isinstance(self.horizon_comparison, PhysicalLifecycleHorizonComparison):
            raise TypeError("horizon_comparison must be PhysicalLifecycleHorizonComparison")
        if self.horizon_comparison.current_block_comparison is not self.comparison:
            raise ValueError("horizon_comparison must retain the current-block comparison")
        eligible, selected, reason = _derive_selection(
            limits=self.posture_limits,
            forecast_admitted=self.forecast_admission.admitted,
            continue_existing_outcome=self.horizon_comparison.continue_existing,
            release_to_current_outcome=self.horizon_comparison.release_to_current,
            new_track_outcome=self.horizon_comparison.new_track,
        )
        object.__setattr__(self, "selection_eligible_lifecycles", eligible)
        object.__setattr__(self, "selected_lifecycle", selected)
        object.__setattr__(self, "selection_reason", reason)

    @property
    def forecast_admitted(self) -> bool:
        """Return the bound supervisory admission result for reporting."""

        return self.forecast_admission.admitted

    @property
    def forecast_admission_reason(self) -> str:
        """Return the caller-declared basis without recreating a free string."""

        return self.forecast_admission.basis

    @property
    def continue_existing_rollout(self) -> ForecastExecutionBlockLifecycleRollout:
        """Return the comparison-owned continuation rollout."""

        return self.comparison.continue_existing.rollout

    @property
    def release_to_current_rollout(self) -> ForecastExecutionBlockLifecycleRollout:
        """Return the comparison-owned release rollout."""

        return self.comparison.release_to_current.rollout

    @property
    def new_track_rollout(self) -> ForecastExecutionBlockLifecycleRollout | None:
        """Return the admitted comparison-owned forecast rollout, when present."""

        if self.comparison.new_track is None:
            return None
        return self.comparison.new_track.rollout

    def rollout_for(
        self,
        lifecycle: PhysicalTargetLifecycle | str,
    ) -> ForecastExecutionBlockLifecycleRollout:
        """Return one source-bound rollout without re-evaluating or re-planning it."""

        selected = PhysicalTargetLifecycle(lifecycle)
        if selected is PhysicalTargetLifecycle.CONTINUE_EXISTING:
            return self.continue_existing_rollout
        if selected is PhysicalTargetLifecycle.RELEASE_TO_CURRENT:
            return self.release_to_current_rollout
        if self.new_track_rollout is None:
            raise ValueError("new_track was not available for this control block")
        return self.new_track_rollout

    def require_selected_rollout(self) -> ForecastExecutionBlockLifecycleRollout:
        """Return the selected current-block shadow path or require caller handling.

        Horizon consequences determine the selection, but this method returns
        only the first current-block path.  The held later intervals are not a
        control handoff.  A missing selection is an abstention, not an implicit
        continuation or release.
        """

        if self.selected_lifecycle is None:
            raise ValueError(
                "selection abstained; an explicit caller policy is required "
                "before a cross-cycle state handoff"
            )
        return self.rollout_for(self.selected_lifecycle)

    def require_selected_execution_request(self) -> ExecutionRolloutRequest:
        """Return the one current-block request selected for external execution.

        This is the only command-facing result of the selection layer.  The
        attached rollout remains an auditable counterfactual and its final
        platform or pump state must not replace the next measured state.
        """

        return self.require_selected_rollout().lifecycle_trace.execution_request

    def peak_abs_pitch_roll_rad_for(
        self,
        lifecycle: PhysicalTargetLifecycle | str,
    ) -> np.ndarray:
        """Return the held-horizon peak attitude used by the selection rule."""

        return np.array(
            self.horizon_comparison.outcome_for(lifecycle).peak_abs_pitch_roll_rad,
            copy=True,
        )

    def peak_abs_pitch_roll_rate_rad_s_for(
        self,
        lifecycle: PhysicalTargetLifecycle | str,
    ) -> np.ndarray:
        """Return held-horizon peak angular rates used by the selection rule."""

        return np.array(
            self.horizon_comparison.outcome_for(lifecycle).peak_abs_pitch_roll_rate_rad_s,
            copy=True,
        )


def select_physical_lifecycle_for_current_block(
    *,
    decision_input: FirstIntervalPhysicalDecisionInput,
    posture_limits: PhysicalPostureLimits,
    forecast_admission: PhysicalForecastAdmission,
    comparison: PhysicalLifecycleComparison,
    horizon_comparison: PhysicalLifecycleHorizonComparison,
) -> PhysicalLifecycleSelectionFact:
    """Apply the minimal constrained rule to held lifecycle outcomes.

    The function does not create lifecycle targets, recompute a forecast, or
    write any chosen request back to the controller.  It can select only the
    first current-block request already contained in ``comparison``.  Its
    ``None`` selection means this layer abstains and requires an explicit
    caller fallback.
    """

    return PhysicalLifecycleSelectionFact(
        decision_input=decision_input,
        posture_limits=posture_limits,
        forecast_admission=forecast_admission,
        comparison=comparison,
        horizon_comparison=horizon_comparison,
    )


def assemble_physical_lifecycle_selection_for_current_block(
    *,
    decision_input: FirstIntervalPhysicalDecisionInput,
    posture_limits: PhysicalPostureLimits,
    forecast_admission: PhysicalForecastAdmission,
) -> PhysicalLifecycleSelectionFact:
    """Replay the available current-block lifecycles and return one selection fact.

    This is a narrow assembly boundary, not a second planner.  It derives every
    replay solely from the trajectory, runtime assembly and lifecycle traces
    already bound to ``decision_input``.  A rejected forecast skips the
    forecast-specific ``new_track`` replay, while the continuation and release
    replays remain available as auditable same-origin facts.  In that case the
    returned selector record still abstains and never evaluates a path to choose
    an operation.
    """

    if not isinstance(decision_input, FirstIntervalPhysicalDecisionInput):
        raise TypeError("decision_input must be FirstIntervalPhysicalDecisionInput")
    if not isinstance(posture_limits, PhysicalPostureLimits):
        raise TypeError("posture_limits must be PhysicalPostureLimits")
    if not isinstance(forecast_admission, PhysicalForecastAdmission):
        raise TypeError("forecast_admission must be PhysicalForecastAdmission")
    if forecast_admission.decision_input is not decision_input:
        raise ValueError("forecast_admission must retain the decision_input")

    comparison = assemble_physical_lifecycle_comparison_for_current_block(
        decision_input=decision_input,
        forecast_admission=forecast_admission,
    )
    horizon_comparison = assemble_physical_lifecycle_horizon_comparison(
        current_block_comparison=comparison,
    )

    return select_physical_lifecycle_for_current_block(
        decision_input=decision_input,
        posture_limits=posture_limits,
        forecast_admission=forecast_admission,
        comparison=comparison,
        horizon_comparison=horizon_comparison,
    )


__all__ = [
    "PhysicalLifecycleSelectionFact",
    "PhysicalLifecycleSelectionReason",
    "PhysicalPostureLimits",
    "assemble_physical_lifecycle_selection_for_current_block",
    "select_physical_lifecycle_for_current_block",
]
