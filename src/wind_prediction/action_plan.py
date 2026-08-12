"""Immutable action identity and commit contract.

This module deliberately stays independent of provider execution.  It records
which action won candidate selection, which action supervision will actually
execute, and whether the evaluation still applies after any correction.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from math import isfinite
from typing import Any


class TargetOperation(str, Enum):
    """Lifecycle operation to apply to the ballast target."""

    SET_DELTA = "set_delta"
    CONTINUE = "continue"
    HOLD_CURRENT = "hold_current"
    PAUSE = "pause"
    RESUME = "resume"
    RELEASE = "release"


@dataclass(frozen=True)
class EvaluationSummary:
    """Small, immutable record of the action identity that was evaluated."""

    evaluated_action: str
    evaluated_target_operation: TargetOperation
    safety_passed: bool
    evaluated_action_vector: tuple[float, ...] = ()
    evaluated_target_masses_kg: tuple[float, ...] = ()
    score: float | None = None
    summary: str = ""


@dataclass(frozen=True)
class ActionPlan:
    """Candidate selection plus the final supervised action identity."""

    candidate_action: str
    final_action: str
    target_operation: TargetOperation
    candidate_action_vector: tuple[float, ...] = ()
    final_action_vector: tuple[float, ...] = ()
    candidate_target_masses_kg: tuple[float, ...] = ()
    final_target_masses_kg: tuple[float, ...] = ()
    needs_reevaluation: bool = True
    evaluation: EvaluationSummary | None = None
    correction_reason: str = ""


def build_candidate_plan(
    candidate_action: str,
    target_operation: TargetOperation | str,
    *,
    action_vector: Any = (),
    target_masses_kg: Any = (),
) -> ActionPlan:
    """Create an unevaluated plan from the selected candidate action."""

    action = _require_action(candidate_action)
    vector = _finite_tuple(action_vector, field="action_vector")
    target = _finite_tuple(target_masses_kg, field="target_masses_kg")
    return ActionPlan(
        candidate_action=action,
        final_action=action,
        target_operation=TargetOperation(target_operation),
        candidate_action_vector=vector,
        final_action_vector=vector,
        candidate_target_masses_kg=target,
        final_target_masses_kg=target,
    )


def apply_action_correction(
    plan: ActionPlan,
    final_action: str,
    *,
    target_operation: TargetOperation | str | None = None,
    action_vector: Any | None = None,
    target_masses_kg: Any | None = None,
    reason: str = "",
) -> ActionPlan:
    """Return a corrected plan and invalidate an evaluation of old identity."""

    action = _require_action(final_action)
    operation = (
        plan.target_operation
        if target_operation is None
        else TargetOperation(target_operation)
    )
    vector = (
        plan.final_action_vector
        if action_vector is None
        else _finite_tuple(action_vector, field="action_vector")
    )
    target = (
        plan.final_target_masses_kg
        if target_masses_kg is None
        else _finite_tuple(target_masses_kg, field="target_masses_kg")
    )
    identity_changed = (
        action != plan.final_action
        or operation != plan.target_operation
        or vector != plan.final_action_vector
        or target != plan.final_target_masses_kg
    )
    return replace(
        plan,
        final_action=action,
        target_operation=operation,
        final_action_vector=vector,
        final_target_masses_kg=target,
        needs_reevaluation=plan.needs_reevaluation or identity_changed,
        correction_reason=str(reason) if identity_changed else plan.correction_reason,
    )


def record_evaluation(
    plan: ActionPlan,
    evaluated_action: str,
    *,
    safety_passed: bool,
    action_vector: Any | None = None,
    target_masses_kg: Any | None = None,
    score: float | None = None,
    summary: str = "",
) -> ActionPlan:
    """Attach an evaluation and mark it current only for the final action."""

    action = _require_action(evaluated_action)
    if not isinstance(safety_passed, bool):
        raise TypeError("safety_passed must be a bool")

    normalized_score = None if score is None else float(score)
    if normalized_score is not None and not isfinite(normalized_score):
        raise ValueError("score must be finite when provided")

    vector = (
        plan.final_action_vector
        if action_vector is None
        else _finite_tuple(action_vector, field="action_vector")
    )
    target = (
        plan.final_target_masses_kg
        if target_masses_kg is None
        else _finite_tuple(target_masses_kg, field="target_masses_kg")
    )
    evaluation = EvaluationSummary(
        evaluated_action=action,
        evaluated_target_operation=plan.target_operation,
        evaluated_action_vector=vector,
        evaluated_target_masses_kg=target,
        safety_passed=safety_passed,
        score=normalized_score,
        summary=str(summary),
    )
    return replace(
        plan,
        evaluation=evaluation,
        needs_reevaluation=(
            action != plan.final_action
            or vector != plan.final_action_vector
            or target != plan.final_target_masses_kg
        ),
    )


def commit_action_plan(plan: ActionPlan) -> dict[str, Any]:
    """Validate the final identity and return a JSON-loggable snapshot."""

    evaluation = plan.evaluation
    if evaluation is None:
        raise ValueError("cannot commit an action plan without an evaluation")
    if evaluation.evaluated_action != plan.final_action:
        raise ValueError(
            "cannot commit: final action does not match the evaluated action"
        )
    if evaluation.evaluated_target_operation != plan.target_operation:
        raise ValueError(
            "cannot commit: final target operation does not match the evaluated operation"
        )
    if evaluation.evaluated_action_vector != plan.final_action_vector:
        raise ValueError(
            "cannot commit: final action vector does not match the evaluated vector"
        )
    if evaluation.evaluated_target_masses_kg != plan.final_target_masses_kg:
        raise ValueError(
            "cannot commit: final target masses do not match the evaluated target"
        )
    if plan.needs_reevaluation:
        raise ValueError("cannot commit an action plan that needs reevaluation")
    if not evaluation.safety_passed:
        raise ValueError("cannot commit an action plan that failed safety evaluation")

    return {
        "candidate_action": plan.candidate_action,
        "final_action": plan.final_action,
        "target_operation": plan.target_operation.value,
        "candidate_action_vector": list(plan.candidate_action_vector),
        "final_action_vector": list(plan.final_action_vector),
        "candidate_target_masses_kg": list(plan.candidate_target_masses_kg),
        "final_target_masses_kg": list(plan.final_target_masses_kg),
        "needs_reevaluation": plan.needs_reevaluation,
        "correction_reason": plan.correction_reason,
        "evaluation": {
            "evaluated_action": evaluation.evaluated_action,
            "evaluated_target_operation": evaluation.evaluated_target_operation.value,
            "evaluated_action_vector": list(evaluation.evaluated_action_vector),
            "evaluated_target_masses_kg": list(
                evaluation.evaluated_target_masses_kg
            ),
            "safety_passed": evaluation.safety_passed,
            "score": evaluation.score,
            "summary": evaluation.summary,
        },
    }


def _require_action(action: str) -> str:
    if not isinstance(action, str) or not action.strip():
        raise ValueError("action identity must be a non-empty string")
    return action


def _finite_tuple(values: Any, *, field: str) -> tuple[float, ...]:
    if values is None:
        return ()
    try:
        normalized = tuple(float(value) for value in values)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must contain numeric values") from exc
    if not all(isfinite(value) for value in normalized):
        raise ValueError(f"{field} must contain only finite values")
    return normalized


__all__ = [
    "ActionPlan",
    "EvaluationSummary",
    "TargetOperation",
    "apply_action_correction",
    "build_candidate_plan",
    "commit_action_plan",
    "record_evaluation",
]
