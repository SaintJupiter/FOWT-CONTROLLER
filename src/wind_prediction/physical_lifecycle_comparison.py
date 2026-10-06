"""Collect same-origin physical outcomes without assigning a control preference.

The current V2 chain can replay three already declared target lifecycles over
one control block: continue an existing target, release it to the measured tank
state, or track a source-bound forecast endpoint.  This module puts their
modeled reachable pump and platform outcomes behind one shared interface.  It does not
introduce an objective, posture threshold, action weight, or selected target.

Keeping the factual comparison separate is deliberate: a later control policy
must compare the same physical paths rather than replaying each option with
slightly different inputs or silently reusing a legacy score.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .first_interval_physical_decision_input import FirstIntervalPhysicalDecisionInput
from .forecast_execution_block_lifecycle_rollout import (
    ForecastExecutionBlockLifecycleRollout,
    HeldLifecycleForecastHorizonRollout,
    rollout_current_execution_block_lifecycle,
    rollout_held_lifecycle_to_forecast_horizon,
)
from .physical_forecast_admission import PhysicalForecastAdmission
from .physical_target_lifecycle import PhysicalTargetLifecycle, PhysicalTargetLifecycleTrace


_PITCH_ROLL_INDICES = (4, 3)


def _readonly_vector(value: Any, *, shape: tuple[int, ...], name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.shape != shape or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must have shape {shape} and contain finite values")
    result = np.array(array, dtype=float, copy=True)
    result.setflags(write=False)
    return result


def _expected_trace(
    decision_input: FirstIntervalPhysicalDecisionInput,
    lifecycle: PhysicalTargetLifecycle,
) -> PhysicalTargetLifecycleTrace | None:
    facts = decision_input.lifecycle_facts
    return {
        PhysicalTargetLifecycle.CONTINUE_EXISTING: facts.continue_existing,
        PhysicalTargetLifecycle.RELEASE_TO_CURRENT: facts.release_to_current,
        PhysicalTargetLifecycle.NEW_TRACK: facts.new_track,
    }[lifecycle]


@dataclass(frozen=True)
class PhysicalLifecycleOutcome:
    """Observable result of one precomputed lifecycle path.

    The stored rollout owns the detailed, causally ordered substeps.  The
    properties below expose only common facts in fixed ``(pitch, roll)`` order
    so later policy code does not need to inspect pump or platform internals.
    They are modeled counterfactual outcomes under the frozen inputs of this
    block, not measured or closed-loop validation results.  No one of these
    values is an objective or a safety result by itself.
    """

    lifecycle: PhysicalTargetLifecycle
    rollout: ForecastExecutionBlockLifecycleRollout

    def __post_init__(self) -> None:
        if not isinstance(self.lifecycle, PhysicalTargetLifecycle):
            raise TypeError("lifecycle must be PhysicalTargetLifecycle")
        if not isinstance(self.rollout, ForecastExecutionBlockLifecycleRollout):
            raise TypeError("rollout must be ForecastExecutionBlockLifecycleRollout")
        if self.rollout.lifecycle is not self.lifecycle:
            raise ValueError("lifecycle must match rollout.lifecycle")

    @property
    def end_pitch_roll_rad(self) -> np.ndarray:
        """Return the block-end platform angle in fixed ``(pitch, roll)`` order."""

        return _readonly_vector(
            self.rollout.final_platform_state.position[list(_PITCH_ROLL_INDICES)],
            shape=(2,),
            name="end_pitch_roll_rad",
        )

    @property
    def end_pitch_roll_rate_rad_s(self) -> np.ndarray:
        """Return the block-end platform angular rate in ``(pitch, roll)`` order."""

        return _readonly_vector(
            self.rollout.final_platform_state.velocity[list(_PITCH_ROLL_INDICES)],
            shape=(2,),
            name="end_pitch_roll_rate_rad_s",
        )

    @property
    def peak_abs_pitch_roll_rad(self) -> np.ndarray:
        """Return the post-origin angle peak observed on this held path."""

        return _readonly_vector(
            self.rollout.post_origin_peak_abs_pitch_roll_rad,
            shape=(2,),
            name="peak_abs_pitch_roll_rad",
        )

    @property
    def peak_abs_pitch_roll_rate_rad_s(self) -> np.ndarray:
        """Return the post-origin angular-rate peak observed on this path."""

        return _readonly_vector(
            self.rollout.post_origin_peak_abs_pitch_roll_rate_rad_s,
            shape=(2,),
            name="peak_abs_pitch_roll_rate_rad_s",
        )

    @property
    def transferred_volume_m3(self) -> float:
        """Return total absolute water volume moved by all pumps in the block."""

        return float(
            sum(
                interval.physical_path.transferred_volume_m3
                for interval in self.rollout.intervals
            )
        )

    @property
    def aggregate_pump_active_time_s(self) -> float:
        """Return cumulative active seconds across all pumps, not wall time."""

        return float(
            sum(
                interval.physical_path.aggregate_pump_active_time_s
                for interval in self.rollout.intervals
            )
        )

    @property
    def pump_start_count(self) -> int:
        """Return starts observed while executing this one lifecycle request."""

        return int(
            sum(
                interval.physical_path.pump_start_count
                for interval in self.rollout.intervals
            )
        )

    @property
    def pump_stop_count(self) -> int:
        """Return stops observed while executing this one lifecycle request."""

        return int(
            sum(
                interval.physical_path.pump_stop_count
                for interval in self.rollout.intervals
            )
        )

    @property
    def pump_direction_switch_count(self) -> int:
        """Return direction changes observed while executing this lifecycle."""

        return int(
            sum(
                interval.physical_path.pump_direction_switch_count
                for interval in self.rollout.intervals
            )
        )

    @property
    def reached_final_tank_masses_kg(self) -> np.ndarray:
        """Return actual block-end tank masses, not the nominal target."""

        return _readonly_vector(
            self.rollout.reached_final_tank_masses_kg,
            shape=(3,),
            name="reached_final_tank_masses_kg",
        )

    @property
    def remaining_mass_to_target_kg(self) -> np.ndarray:
        """Return remaining distance to this lifecycle's block-end request."""

        return _readonly_vector(
            self.rollout.block_end_remaining_mass_to_target_kg,
            shape=(3,),
            name="remaining_mass_to_target_kg",
        )


@dataclass(frozen=True)
class PhysicalLifecycleComparison:
    """Same-origin factual outcomes available to a later decision policy.

    Continuing and releasing the current target are always present.  Tracking a
    forecast endpoint is present only when a caller-owned admission fact both
    accepts the forecast and binds it to this exact physical decision input.
    This record intentionally makes no preferred option available.
    """

    decision_input: FirstIntervalPhysicalDecisionInput
    forecast_admission: PhysicalForecastAdmission
    continue_existing: PhysicalLifecycleOutcome
    release_to_current: PhysicalLifecycleOutcome
    new_track: PhysicalLifecycleOutcome | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.decision_input, FirstIntervalPhysicalDecisionInput):
            raise TypeError("decision_input must be FirstIntervalPhysicalDecisionInput")
        if not isinstance(self.forecast_admission, PhysicalForecastAdmission):
            raise TypeError("forecast_admission must be PhysicalForecastAdmission")
        if self.forecast_admission.decision_input is not self.decision_input:
            raise ValueError("forecast_admission must retain the decision_input")
        self._validate_outcome(
            name="continue_existing",
            outcome=self.continue_existing,
            lifecycle=PhysicalTargetLifecycle.CONTINUE_EXISTING,
        )
        self._validate_outcome(
            name="release_to_current",
            outcome=self.release_to_current,
            lifecycle=PhysicalTargetLifecycle.RELEASE_TO_CURRENT,
        )
        expected_new_track = _expected_trace(
            self.decision_input,
            PhysicalTargetLifecycle.NEW_TRACK,
        )
        if self.new_track is None:
            if self.forecast_admission.admitted and expected_new_track is not None:
                raise ValueError(
                    "new_track is required when admitted physical facts contain a forecast endpoint"
                )
        else:
            if not self.forecast_admission.admitted:
                raise ValueError("new_track is unavailable when forecast is not admitted")
            self._validate_outcome(
                name="new_track",
                outcome=self.new_track,
                lifecycle=PhysicalTargetLifecycle.NEW_TRACK,
            )

    def _validate_outcome(
        self,
        *,
        name: str,
        outcome: PhysicalLifecycleOutcome,
        lifecycle: PhysicalTargetLifecycle,
    ) -> None:
        if not isinstance(outcome, PhysicalLifecycleOutcome):
            raise TypeError(f"{name} must be PhysicalLifecycleOutcome")
        if outcome.lifecycle is not lifecycle:
            raise ValueError(f"{name} must retain the {lifecycle.value} lifecycle")
        rollout = outcome.rollout
        if rollout.trajectory is not self.decision_input.trajectory:
            raise ValueError(f"{name} must retain the shared forecast trajectory")
        if rollout.runtime_assembly is not self.decision_input.runtime_assembly:
            raise ValueError(f"{name} must retain the shared runtime assembly")
        expected_trace = _expected_trace(self.decision_input, lifecycle)
        if expected_trace is None or rollout.lifecycle_trace is not expected_trace:
            raise ValueError(f"{name} must retain the matching lifecycle trace")

    @property
    def available_lifecycles(self) -> tuple[PhysicalTargetLifecycle, ...]:
        """Return lifecycle operations with one fully replayed physical outcome."""

        lifecycles = [
            PhysicalTargetLifecycle.CONTINUE_EXISTING,
            PhysicalTargetLifecycle.RELEASE_TO_CURRENT,
        ]
        if self.new_track is not None:
            lifecycles.append(PhysicalTargetLifecycle.NEW_TRACK)
        return tuple(lifecycles)

    @property
    def outcomes(self) -> tuple[PhysicalLifecycleOutcome, ...]:
        """Return outcomes in the stable lifecycle order used by this module."""

        result = [self.continue_existing, self.release_to_current]
        if self.new_track is not None:
            result.append(self.new_track)
        return tuple(result)

    def outcome_for(
        self,
        lifecycle: PhysicalTargetLifecycle | str,
    ) -> PhysicalLifecycleOutcome:
        """Return one factual outcome without selecting it."""

        selected = PhysicalTargetLifecycle(lifecycle)
        if selected is PhysicalTargetLifecycle.CONTINUE_EXISTING:
            return self.continue_existing
        if selected is PhysicalTargetLifecycle.RELEASE_TO_CURRENT:
            return self.release_to_current
        if self.new_track is None:
            raise ValueError("new_track is not available for this control block")
        return self.new_track

    def change_from_continuation(
        self,
        lifecycle: PhysicalTargetLifecycle | str,
    ) -> "PhysicalLifecycleChangeFromContinuation":
        """Expose one alternative's factual change from continuing the target.

        The returned signs always mean ``alternative - continuation``.  This is
        a shared accounting convention for later policy work, not a statement
        that a negative or positive value is preferable.
        """

        selected = PhysicalTargetLifecycle(lifecycle)
        if selected is PhysicalTargetLifecycle.CONTINUE_EXISTING:
            raise ValueError("continuation is the reference, not an alternative")
        self.outcome_for(selected)
        return PhysicalLifecycleChangeFromContinuation(
            comparison=self,
            lifecycle=selected,
        )


@dataclass(frozen=True)
class PhysicalLifecycleChangeFromContinuation:
    """Factual alternative-minus-continuation outcomes for one control block.

    This record intentionally leaves out distance to each operation's requested
    target because those targets have different meanings.  Every retained field
    is either a directly modeled execution result or a platform state quantity
    evaluated over the same frozen block.
    """

    comparison: PhysicalLifecycleComparison
    lifecycle: PhysicalTargetLifecycle

    def __post_init__(self) -> None:
        if not isinstance(self.comparison, PhysicalLifecycleComparison):
            raise TypeError("comparison must be PhysicalLifecycleComparison")
        lifecycle = PhysicalTargetLifecycle(self.lifecycle)
        if lifecycle is PhysicalTargetLifecycle.CONTINUE_EXISTING:
            raise ValueError("continuation is the reference, not an alternative")
        self.comparison.outcome_for(lifecycle)
        object.__setattr__(self, "lifecycle", lifecycle)

    @property
    def alternative(self) -> PhysicalLifecycleOutcome:
        """Return the alternative outcome represented by this factual delta."""

        return self.comparison.outcome_for(self.lifecycle)

    @property
    def continuation(self) -> PhysicalLifecycleOutcome:
        """Return the shared continuation outcome used as the reference."""

        return self.comparison.continue_existing

    @property
    def transferred_volume_delta_m3(self) -> float:
        return float(
            self.alternative.transferred_volume_m3
            - self.continuation.transferred_volume_m3
        )

    @property
    def aggregate_pump_active_time_delta_s(self) -> float:
        return float(
            self.alternative.aggregate_pump_active_time_s
            - self.continuation.aggregate_pump_active_time_s
        )

    @property
    def pump_start_count_delta(self) -> int:
        return int(self.alternative.pump_start_count - self.continuation.pump_start_count)

    @property
    def pump_stop_count_delta(self) -> int:
        return int(self.alternative.pump_stop_count - self.continuation.pump_stop_count)

    @property
    def pump_direction_switch_count_delta(self) -> int:
        return int(
            self.alternative.pump_direction_switch_count
            - self.continuation.pump_direction_switch_count
        )

    @property
    def reached_final_tank_mass_delta_kg(self) -> np.ndarray:
        return _readonly_vector(
            self.alternative.reached_final_tank_masses_kg
            - self.continuation.reached_final_tank_masses_kg,
            shape=(3,),
            name="reached_final_tank_mass_delta_kg",
        )

    @property
    def end_pitch_roll_delta_rad(self) -> np.ndarray:
        return _readonly_vector(
            self.alternative.end_pitch_roll_rad
            - self.continuation.end_pitch_roll_rad,
            shape=(2,),
            name="end_pitch_roll_delta_rad",
        )

    @property
    def peak_abs_pitch_roll_delta_rad(self) -> np.ndarray:
        return _readonly_vector(
            self.alternative.peak_abs_pitch_roll_rad
            - self.continuation.peak_abs_pitch_roll_rad,
            shape=(2,),
            name="peak_abs_pitch_roll_delta_rad",
        )

    @property
    def end_pitch_roll_rate_delta_rad_s(self) -> np.ndarray:
        return _readonly_vector(
            self.alternative.end_pitch_roll_rate_rad_s
            - self.continuation.end_pitch_roll_rate_rad_s,
            shape=(2,),
            name="end_pitch_roll_rate_delta_rad_s",
        )

    @property
    def peak_abs_pitch_roll_rate_delta_rad_s(self) -> np.ndarray:
        return _readonly_vector(
            self.alternative.peak_abs_pitch_roll_rate_rad_s
            - self.continuation.peak_abs_pitch_roll_rate_rad_s,
            shape=(2,),
            name="peak_abs_pitch_roll_rate_delta_rad_s",
        )


@dataclass(frozen=True)
class PhysicalLifecycleHorizonOutcome:
    """One declared lifecycle held through the available forecast horizon.

    The outcome starts from one already replayed current-block result and then
    holds that same request through the remaining source-bound forecast loads.
    It is a conditional physical counterfactual, not a later target update,
    rolling execution state, or preferred action.
    """

    lifecycle: PhysicalTargetLifecycle
    current_block_outcome: PhysicalLifecycleOutcome
    rollout: HeldLifecycleForecastHorizonRollout

    def __post_init__(self) -> None:
        if not isinstance(self.lifecycle, PhysicalTargetLifecycle):
            raise TypeError("lifecycle must be PhysicalTargetLifecycle")
        if not isinstance(self.current_block_outcome, PhysicalLifecycleOutcome):
            raise TypeError("current_block_outcome must be PhysicalLifecycleOutcome")
        if not isinstance(self.rollout, HeldLifecycleForecastHorizonRollout):
            raise TypeError("rollout must be HeldLifecycleForecastHorizonRollout")
        if (
            self.current_block_outcome.lifecycle is not self.lifecycle
            or self.rollout.lifecycle is not self.lifecycle
        ):
            raise ValueError("lifecycle must match the current-block and held outcomes")
        if self.rollout.prefix is not self.current_block_outcome.rollout:
            raise ValueError("held rollout must retain the supplied current-block outcome")

    @property
    def end_pitch_roll_rad(self) -> np.ndarray:
        """Return the modeled horizon-end angle in fixed ``(pitch, roll)`` order."""

        return _readonly_vector(
            self.rollout.final_platform_state.position[list(_PITCH_ROLL_INDICES)],
            shape=(2,),
            name="end_pitch_roll_rad",
        )

    @property
    def end_pitch_roll_rate_rad_s(self) -> np.ndarray:
        """Return the modeled horizon-end angular rate in ``(pitch, roll)`` order."""

        return _readonly_vector(
            self.rollout.final_platform_state.velocity[list(_PITCH_ROLL_INDICES)],
            shape=(2,),
            name="end_pitch_roll_rate_rad_s",
        )

    @property
    def peak_abs_pitch_roll_rad(self) -> np.ndarray:
        """Return modeled angle peaks over the held forecast horizon."""

        return _readonly_vector(
            self.rollout.post_origin_peak_abs_pitch_roll_rad,
            shape=(2,),
            name="peak_abs_pitch_roll_rad",
        )

    @property
    def peak_abs_pitch_roll_rate_rad_s(self) -> np.ndarray:
        """Return modeled angular-rate peaks over the held forecast horizon."""

        return _readonly_vector(
            self.rollout.post_origin_peak_abs_pitch_roll_rate_rad_s,
            shape=(2,),
            name="peak_abs_pitch_roll_rate_rad_s",
        )

    @property
    def transferred_volume_m3(self) -> float:
        """Return total modeled pump transfer over the held forecast horizon."""

        return float(
            sum(
                interval.physical_path.transferred_volume_m3
                for interval in self.rollout.intervals
            )
        )

    @property
    def aggregate_pump_active_time_s(self) -> float:
        """Return cumulative active pump seconds over the held horizon."""

        return float(
            sum(
                interval.physical_path.aggregate_pump_active_time_s
                for interval in self.rollout.intervals
            )
        )

    @property
    def pump_start_count(self) -> int:
        return int(
            sum(
                interval.physical_path.pump_start_count
                for interval in self.rollout.intervals
            )
        )

    @property
    def pump_stop_count(self) -> int:
        return int(
            sum(
                interval.physical_path.pump_stop_count
                for interval in self.rollout.intervals
            )
        )

    @property
    def pump_direction_switch_count(self) -> int:
        return int(
            sum(
                interval.physical_path.pump_direction_switch_count
                for interval in self.rollout.intervals
            )
        )

    @property
    def reached_final_tank_masses_kg(self) -> np.ndarray:
        """Return actual modeled tank masses at the held horizon end."""

        return _readonly_vector(
            self.rollout.final_execution_state.actual_masses_kg,
            shape=(3,),
            name="reached_final_tank_masses_kg",
        )


@dataclass(frozen=True)
class PhysicalLifecycleHorizonComparison:
    """Same-origin held-request outcomes for later, explicit decision work.

    This comparison only collects the horizon consequences of operations that
    were already available in ``current_block_comparison``.  It does not rank
    them, impose posture limits, or provide a state handoff to real rolling
    control.
    """

    current_block_comparison: PhysicalLifecycleComparison
    continue_existing: PhysicalLifecycleHorizonOutcome
    release_to_current: PhysicalLifecycleHorizonOutcome
    new_track: PhysicalLifecycleHorizonOutcome | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.current_block_comparison, PhysicalLifecycleComparison):
            raise TypeError("current_block_comparison must be PhysicalLifecycleComparison")
        self._validate_outcome(
            name="continue_existing",
            outcome=self.continue_existing,
            lifecycle=PhysicalTargetLifecycle.CONTINUE_EXISTING,
        )
        self._validate_outcome(
            name="release_to_current",
            outcome=self.release_to_current,
            lifecycle=PhysicalTargetLifecycle.RELEASE_TO_CURRENT,
        )
        if self.new_track is None:
            if self.current_block_comparison.new_track is not None:
                raise ValueError("new_track is required when current-block comparison provides it")
        else:
            self._validate_outcome(
                name="new_track",
                outcome=self.new_track,
                lifecycle=PhysicalTargetLifecycle.NEW_TRACK,
            )

    def _validate_outcome(
        self,
        *,
        name: str,
        outcome: PhysicalLifecycleHorizonOutcome,
        lifecycle: PhysicalTargetLifecycle,
    ) -> None:
        if not isinstance(outcome, PhysicalLifecycleHorizonOutcome):
            raise TypeError(f"{name} must be PhysicalLifecycleHorizonOutcome")
        if outcome.lifecycle is not lifecycle:
            raise ValueError(f"{name} must retain the {lifecycle.value} lifecycle")
        if outcome.current_block_outcome is not self.current_block_comparison.outcome_for(lifecycle):
            raise ValueError(f"{name} must retain the matching current-block outcome")

    @property
    def available_lifecycles(self) -> tuple[PhysicalTargetLifecycle, ...]:
        return tuple(outcome.lifecycle for outcome in self.outcomes)

    @property
    def outcomes(self) -> tuple[PhysicalLifecycleHorizonOutcome, ...]:
        result = [self.continue_existing, self.release_to_current]
        if self.new_track is not None:
            result.append(self.new_track)
        return tuple(result)

    def outcome_for(
        self,
        lifecycle: PhysicalTargetLifecycle | str,
    ) -> PhysicalLifecycleHorizonOutcome:
        """Return one held-request physical outcome without selecting it."""

        selected = PhysicalTargetLifecycle(lifecycle)
        if selected is PhysicalTargetLifecycle.CONTINUE_EXISTING:
            return self.continue_existing
        if selected is PhysicalTargetLifecycle.RELEASE_TO_CURRENT:
            return self.release_to_current
        if self.new_track is None:
            raise ValueError("new_track is not available for this forecast horizon")
        return self.new_track


def assemble_physical_lifecycle_comparison_for_current_block(
    *,
    decision_input: FirstIntervalPhysicalDecisionInput,
    forecast_admission: PhysicalForecastAdmission,
) -> PhysicalLifecycleComparison:
    """Replay available lifecycles once and expose their common factual outcomes.

    The function neither uses posture limits nor selects an operation.  It is
    the sole current-block assembly path for a future policy to consume the
    same source-bound paths used by the rest of the V2 chain.
    """

    if not isinstance(decision_input, FirstIntervalPhysicalDecisionInput):
        raise TypeError("decision_input must be FirstIntervalPhysicalDecisionInput")
    if not isinstance(forecast_admission, PhysicalForecastAdmission):
        raise TypeError("forecast_admission must be PhysicalForecastAdmission")
    if forecast_admission.decision_input is not decision_input:
        raise ValueError("forecast_admission must retain the decision_input")

    facts = decision_input.lifecycle_facts
    rollout_kwargs = {
        "trajectory": decision_input.trajectory,
        "runtime_assembly": decision_input.runtime_assembly,
    }
    continue_existing = PhysicalLifecycleOutcome(
        lifecycle=PhysicalTargetLifecycle.CONTINUE_EXISTING,
        rollout=rollout_current_execution_block_lifecycle(
            lifecycle_trace=facts.continue_existing,
            **rollout_kwargs,
        ),
    )
    release_to_current = PhysicalLifecycleOutcome(
        lifecycle=PhysicalTargetLifecycle.RELEASE_TO_CURRENT,
        rollout=rollout_current_execution_block_lifecycle(
            lifecycle_trace=facts.release_to_current,
            **rollout_kwargs,
        ),
    )
    new_track = None
    if forecast_admission.admitted and facts.new_track is not None:
        new_track = PhysicalLifecycleOutcome(
            lifecycle=PhysicalTargetLifecycle.NEW_TRACK,
            rollout=rollout_current_execution_block_lifecycle(
                lifecycle_trace=facts.new_track,
                **rollout_kwargs,
            ),
        )
    return PhysicalLifecycleComparison(
        decision_input=decision_input,
        forecast_admission=forecast_admission,
        continue_existing=continue_existing,
        release_to_current=release_to_current,
        new_track=new_track,
    )


def assemble_physical_lifecycle_horizon_comparison(
    *,
    current_block_comparison: PhysicalLifecycleComparison,
) -> PhysicalLifecycleHorizonComparison:
    """Hold each already available lifecycle through the same forecast horizon.

    The function consumes current-block outcomes rather than raw states or
    requests, so it cannot replay the first block with a different origin.  It
    returns factual counterfactuals only and deliberately has no selected
    lifecycle or numerical preference.
    """

    if not isinstance(current_block_comparison, PhysicalLifecycleComparison):
        raise TypeError("current_block_comparison must be PhysicalLifecycleComparison")

    def build(outcome: PhysicalLifecycleOutcome) -> PhysicalLifecycleHorizonOutcome:
        return PhysicalLifecycleHorizonOutcome(
            lifecycle=outcome.lifecycle,
            current_block_outcome=outcome,
            rollout=rollout_held_lifecycle_to_forecast_horizon(
                prefix=outcome.rollout,
            ),
        )

    continue_existing = build(current_block_comparison.continue_existing)
    release_to_current = build(current_block_comparison.release_to_current)
    new_track = (
        None
        if current_block_comparison.new_track is None
        else build(current_block_comparison.new_track)
    )
    return PhysicalLifecycleHorizonComparison(
        current_block_comparison=current_block_comparison,
        continue_existing=continue_existing,
        release_to_current=release_to_current,
        new_track=new_track,
    )


__all__ = [
    "PhysicalLifecycleComparison",
    "PhysicalLifecycleChangeFromContinuation",
    "PhysicalLifecycleHorizonComparison",
    "PhysicalLifecycleHorizonOutcome",
    "PhysicalLifecycleOutcome",
    "assemble_physical_lifecycle_comparison_for_current_block",
    "assemble_physical_lifecycle_horizon_comparison",
]
