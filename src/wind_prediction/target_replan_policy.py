"""Pure policy for deciding whether an unfinished ballast target is still useful."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, Sequence


RecoveryDirectionMapper = Callable[
    [tuple[float, float, float]], Sequence[float]
]

_DIRECTION_ZERO_EPS = 1e-12


@dataclass(frozen=True)
class TargetReplanConfig:
    """Physical thresholds used by :func:`evaluate_target_replan`.

    The envelope values are the existing pitch and roll control envelopes.  The
    target stop error must use the same per-tank tolerance as the pump executor.
    The rate tolerance represents measurement or numerical noise, not a
    case-specific tuning parameter.
    """

    pitch_envelope_deg: float
    roll_envelope_deg: float
    target_stop_error_kg: float
    rate_noise_tolerance_deg_s: float

    def __post_init__(self) -> None:
        values = {
            "pitch_envelope_deg": self.pitch_envelope_deg,
            "roll_envelope_deg": self.roll_envelope_deg,
            "target_stop_error_kg": self.target_stop_error_kg,
            "rate_noise_tolerance_deg_s": self.rate_noise_tolerance_deg_s,
        }
        for name, value in values.items():
            if not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite")
        if self.pitch_envelope_deg <= 0.0 or self.roll_envelope_deg <= 0.0:
            raise ValueError("pitch and roll envelopes must be positive")
        if self.target_stop_error_kg < 0.0:
            raise ValueError("target_stop_error_kg must be non-negative")
        if self.rate_noise_tolerance_deg_s < 0.0:
            raise ValueError("rate_noise_tolerance_deg_s must be non-negative")


@dataclass(frozen=True)
class TargetReplanResult:
    """Immutable explanation of one target-replanning decision."""

    target_completed: bool
    outside_envelope_axes: tuple[bool, bool]
    recovering_axes: tuple[bool, bool]
    target_direction_helpful: bool
    requires_replan: bool
    reason: str


def _finite_tuple(
    values: Sequence[float],
    *,
    size: int,
    name: str,
) -> tuple[float, ...]:
    try:
        result = tuple(float(value) for value in values)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain numeric values") from exc
    if len(result) != size:
        raise ValueError(f"{name} must contain exactly {size} values")
    if not all(math.isfinite(value) for value in result):
        raise ValueError(f"{name} must contain only finite values")
    return result


def _resolve_recovery_direction(
    debt_kg: tuple[float, float, float],
    *,
    target_recovery_direction: Sequence[float] | None,
    debt_to_recovery_direction: RecoveryDirectionMapper | None,
) -> tuple[float, float]:
    supplied_vector = target_recovery_direction is not None
    supplied_mapper = debt_to_recovery_direction is not None
    if supplied_vector == supplied_mapper:
        raise ValueError(
            "provide exactly one of target_recovery_direction or "
            "debt_to_recovery_direction"
        )
    if supplied_vector:
        raw_direction = target_recovery_direction
    else:
        assert debt_to_recovery_direction is not None
        raw_direction = debt_to_recovery_direction(debt_kg)
    direction = _finite_tuple(
        raw_direction,
        size=2,
        name="target recovery direction",
    )
    return float(direction[0]), float(direction[1])


def evaluate_target_replan(
    posture_deg: Sequence[float],
    posture_rate_deg_s: Sequence[float],
    current_masses_kg: Sequence[float],
    target_masses_kg: Sequence[float],
    config: TargetReplanConfig,
    *,
    target_recovery_direction: Sequence[float] | None = None,
    debt_to_recovery_direction: RecoveryDirectionMapper | None = None,
) -> TargetReplanResult:
    """Decide whether the existing ballast target must be replanned.

    Vector order is ``[pitch, roll]``.  A positive recovery-direction component
    follows the planner compensation convention: it counteracts a positive
    posture error on the same axis.  The mapper, when used, receives the
    three-tank debt ``target_masses_kg - current_masses_kg``.

    Replanning concerns only outstanding execution debt.  A completed target
    means the previous action has finished; it does not by itself authorize a
    new pump command.  New work remains the responsibility of the normal
    posture-and-forecast candidate evaluation.  An unfinished target is
    replanned only when at least one axis is outside its control envelope, not
    every outside axis is measurably recovering, and the remaining target
    motion cannot help the axes that still need correction.
    """

    posture = _finite_tuple(posture_deg, size=2, name="posture_deg")
    posture_rate = _finite_tuple(
        posture_rate_deg_s,
        size=2,
        name="posture_rate_deg_s",
    )
    current_masses = _finite_tuple(
        current_masses_kg,
        size=3,
        name="current_masses_kg",
    )
    target_masses = _finite_tuple(
        target_masses_kg,
        size=3,
        name="target_masses_kg",
    )

    debt_kg = (
        target_masses[0] - current_masses[0],
        target_masses[1] - current_masses[1],
        target_masses[2] - current_masses[2],
    )
    target_completed = all(
        abs(debt) <= config.target_stop_error_kg for debt in debt_kg
    )

    envelopes = (config.pitch_envelope_deg, config.roll_envelope_deg)
    outside_axes = tuple(
        abs(angle) > envelope
        for angle, envelope in zip(posture, envelopes)
    )
    recovering_axes = tuple(
        abs(rate) > config.rate_noise_tolerance_deg_s and angle * rate < 0.0
        for angle, rate in zip(posture, posture_rate)
    )
    outside_indices = tuple(
        index for index, outside in enumerate(outside_axes) if outside
    )
    nonrecovering_outside_indices = tuple(
        index for index in outside_indices if not recovering_axes[index]
    )

    recovery_direction = _resolve_recovery_direction(
        debt_kg,
        target_recovery_direction=target_recovery_direction,
        debt_to_recovery_direction=debt_to_recovery_direction,
    )
    direction_is_nonzero = math.hypot(*recovery_direction) > _DIRECTION_ZERO_EPS
    target_direction_helpful = bool(
        not target_completed
        and nonrecovering_outside_indices
        and direction_is_nonzero
        and all(
            posture[index] * recovery_direction[index] > 0.0
            for index in nonrecovering_outside_indices
        )
    )

    if not outside_indices:
        requires_replan = False
        reason = "inside_envelope"
    elif not nonrecovering_outside_indices:
        requires_replan = False
        reason = "outside_envelope_recovering"
    elif target_completed:
        requires_replan = False
        reason = "completed_target_has_no_execution_debt"
    elif target_direction_helpful:
        requires_replan = False
        reason = "unfinished_target_direction_helpful"
    else:
        requires_replan = True
        reason = "unfinished_target_direction_not_helpful"

    return TargetReplanResult(
        target_completed=target_completed,
        outside_envelope_axes=(bool(outside_axes[0]), bool(outside_axes[1])),
        recovering_axes=(bool(recovering_axes[0]), bool(recovering_axes[1])),
        target_direction_helpful=target_direction_helpful,
        requires_replan=requires_replan,
        reason=reason,
    )
