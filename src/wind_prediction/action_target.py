"""Action meaning, proxy-vector formation and execution-target requests.

This module is deliberately narrower than candidate evaluation.  It defines
what each action means, how the retained two-axis demand proxy becomes an
action vector, and which three-tank execution request that vector creates.
It does not form forecast demand, authorize actions, simulate pumps, score
candidates or propagate platform state.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np

from .action_plan import TargetOperation
from .ballast_allocation import compensation_to_mass_delta_kg
from .execution_rollout import ExecutionRolloutRequest, ExecutionRolloutState


_EPS = 1e-12


class ControlAction(str, Enum):
    """Controller decisions with explicit target-lifecycle meaning."""

    STRENGTHEN = "strengthen"
    NORMAL = "normal"
    REDUCED = "reduced"
    CONTINUE_TARGET = "continue_target"
    RELEASE_TARGET = "release_target"
    REVERSE = "reverse"


class ActionVectorSource(str, Enum):
    """State from which an action derives its target-update vector."""

    NONE = "none"
    RESIDUAL_DEMAND = "residual_demand"
    PREVIOUS_ACTION_VECTOR = "previous_action_vector"


@dataclass(frozen=True)
class ActionSemantics:
    """Stable lifecycle and vector meaning of one candidate action."""

    target_operation: TargetOperation
    vector_source: ActionVectorSource
    requires_nonzero_demand: bool


_ACTION_SEMANTICS: Mapping[ControlAction, ActionSemantics] = MappingProxyType(
    {
        ControlAction.CONTINUE_TARGET: ActionSemantics(
            target_operation=TargetOperation.CONTINUE,
            vector_source=ActionVectorSource.NONE,
            requires_nonzero_demand=False,
        ),
        ControlAction.RELEASE_TARGET: ActionSemantics(
            target_operation=TargetOperation.RELEASE,
            vector_source=ActionVectorSource.NONE,
            requires_nonzero_demand=False,
        ),
        ControlAction.REDUCED: ActionSemantics(
            target_operation=TargetOperation.SET_DELTA,
            vector_source=ActionVectorSource.RESIDUAL_DEMAND,
            requires_nonzero_demand=True,
        ),
        ControlAction.NORMAL: ActionSemantics(
            target_operation=TargetOperation.SET_DELTA,
            vector_source=ActionVectorSource.RESIDUAL_DEMAND,
            requires_nonzero_demand=True,
        ),
        ControlAction.STRENGTHEN: ActionSemantics(
            target_operation=TargetOperation.SET_DELTA,
            vector_source=ActionVectorSource.RESIDUAL_DEMAND,
            requires_nonzero_demand=True,
        ),
        ControlAction.REVERSE: ActionSemantics(
            target_operation=TargetOperation.SET_DELTA,
            vector_source=ActionVectorSource.PREVIOUS_ACTION_VECTOR,
            requires_nonzero_demand=True,
        ),
    }
)

ACTION_ORDER = tuple(_ACTION_SEMANTICS)


def action_semantics_for(action: ControlAction) -> ActionSemantics:
    """Return the target lifecycle and vector source for ``action``."""

    return _ACTION_SEMANTICS[action]


def _normalized_direction(
    demand_deg: Any,
    demand_axis_scale_deg: Any,
) -> np.ndarray:
    demand = np.asarray(demand_deg, dtype=float)
    scale = np.asarray(demand_axis_scale_deg, dtype=float)
    normalized = demand / scale
    magnitude = float(np.linalg.norm(normalized))
    if magnitude <= _EPS:
        return np.zeros(2, dtype=float)
    return normalized / magnitude * scale


def action_vector_for(
    action: ControlAction,
    *,
    demand_deg: Any,
    previous_action_vector_deg: Any,
    demand_axis_scale_deg: Any,
    reduced_ratio: float,
    normal_ratio: float,
    strengthen_ratio: float,
    reverse_ratio: float,
) -> np.ndarray:
    """Form one retained proxy-vector action from explicit configuration values.

    The vector still uses the legacy two-axis demand representation.  This
    function only centralizes its present action semantics; it does not claim
    the vector is a physical force, moment or ballast-mass demand.
    """

    semantics = action_semantics_for(action)
    if semantics.vector_source is ActionVectorSource.NONE:
        return np.zeros(2, dtype=float)
    demand_scale = np.asarray(demand_axis_scale_deg, dtype=float)
    if semantics.vector_source is ActionVectorSource.PREVIOUS_ACTION_VECTOR:
        reference = np.asarray(previous_action_vector_deg, dtype=float)
        if np.linalg.norm(reference / demand_scale) <= _EPS:
            return np.zeros(2, dtype=float)
        return -_normalized_direction(reference, demand_scale) * float(reverse_ratio)
    direction = _normalized_direction(demand_deg, demand_scale)
    ratio = {
        ControlAction.REDUCED: reduced_ratio,
        ControlAction.NORMAL: normal_ratio,
        ControlAction.STRENGTHEN: strengthen_ratio,
    }[action]
    return direction * float(ratio)


def execution_request_for_action(
    action: ControlAction,
    *,
    state: ExecutionRolloutState,
    action_vector_deg: Any,
    demand_axis_scale_deg: Any,
    action_mass_quantum_kg: float,
    tank_capacity_kg: float,
) -> ExecutionRolloutRequest:
    """Translate an action vector into the exact target evaluated by pumps."""

    operation = action_semantics_for(action).target_operation
    if operation is TargetOperation.CONTINUE:
        return ExecutionRolloutRequest.track(state.primary_target_masses_kg)
    if operation is TargetOperation.RELEASE:
        return ExecutionRolloutRequest.release_to_current()
    delta = compensation_to_mass_delta_kg(
        action_vector_deg,
        deadband_deg=demand_axis_scale_deg,
        action_mass_quantum_kg=action_mass_quantum_kg,
    )
    target = np.clip(
        state.actual_masses_kg + delta,
        0.0,
        float(tank_capacity_kg),
    )
    return ExecutionRolloutRequest.track(target)


__all__ = [
    "ACTION_ORDER",
    "ActionSemantics",
    "ActionVectorSource",
    "ControlAction",
    "action_semantics_for",
    "action_vector_for",
    "execution_request_for_action",
]
