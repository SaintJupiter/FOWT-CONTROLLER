"""Primary target lifecycle state helpers.

The provider still owns the control behavior. This module only centralizes the
shared default state used when a preview provider is created or reset.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class PrimaryTargetTransition:
    anchor_masses_kg: np.ndarray
    delta_kg: np.ndarray
    target_kg: np.ndarray
    initialized: bool
    refreshed: bool
    reused: bool
    resumed: bool
    last_reset_s: float


@dataclass(frozen=True)
class PrimaryTargetProposalMetrics:
    proposal_kg: np.ndarray
    target_delta_mean_kg: float
    target_delta_m3: float
    after_delta_mean_kg: float


def _anchor_masses(anchor_masses_kg: Any) -> np.ndarray:
    return np.asarray(anchor_masses_kg, dtype=float).copy()


def _three(values: Any) -> np.ndarray:
    arr = np.asarray(values, dtype=float).reshape(-1)
    if arr.size < 3:
        arr = np.pad(arr, (0, 3 - arr.size))
    return arr[:3].astype(float)


def plant_primary_masses(
    plant_info: dict[str, Any] | None,
    fallback_masses_kg: Any,
) -> np.ndarray:
    """Return the first three tank masses used by primary target lifecycle."""

    masses = np.asarray(
        (plant_info or {}).get("tank_masses", fallback_masses_kg),
        dtype=float,
    ).reshape(-1)
    if masses.size < 3:
        masses = np.asarray(fallback_masses_kg, dtype=float).reshape(-1)
    return masses[:3].astype(float)


def primary_target_from_delta(
    masses_kg: Any,
    delta_kg: Any,
    *,
    tank_capacity_kg: float,
    current_time_s: float,
    initialized: bool = True,
    refreshed: bool = True,
    reused: bool = False,
    resumed: bool = False,
) -> PrimaryTargetTransition:
    """Return a target transition from current masses plus a mass delta."""

    masses = _three(masses_kg)
    delta = _three(delta_kg)
    target = np.clip(masses + delta, 0.0, float(tank_capacity_kg))
    return PrimaryTargetTransition(
        anchor_masses_kg=masses.copy(),
        delta_kg=delta.copy(),
        target_kg=target.copy(),
        initialized=bool(initialized),
        refreshed=bool(refreshed),
        reused=bool(reused),
        resumed=bool(resumed),
        last_reset_s=float(current_time_s),
    )


def primary_target_at_current(
    masses_kg: Any,
    *,
    tank_capacity_kg: float,
    current_time_s: float,
    initialized: bool,
    clip_target: bool,
) -> PrimaryTargetTransition:
    """Return a transition whose target is the current plant masses."""

    masses = _three(masses_kg)
    target = (
        np.clip(masses.copy(), 0.0, float(tank_capacity_kg))
        if clip_target
        else masses.copy()
    )
    return PrimaryTargetTransition(
        anchor_masses_kg=masses.copy(),
        delta_kg=np.zeros(3, dtype=float),
        target_kg=target.copy(),
        initialized=bool(initialized),
        refreshed=True,
        reused=False,
        resumed=False,
        last_reset_s=float(current_time_s),
    )


def primary_target_from_paused(
    masses_kg: Any,
    paused_target_kg: Any,
    *,
    tank_capacity_kg: float,
    current_time_s: float,
) -> PrimaryTargetTransition:
    """Return a transition that resumes a paused target against current masses."""

    masses = _three(masses_kg)
    target = np.clip(
        _three(paused_target_kg),
        0.0,
        float(tank_capacity_kg),
    )
    return PrimaryTargetTransition(
        anchor_masses_kg=masses.copy(),
        delta_kg=(target - masses).copy(),
        target_kg=target.copy(),
        initialized=True,
        refreshed=False,
        reused=True,
        resumed=True,
        last_reset_s=float(current_time_s),
    )


def primary_target_capped_to_current(
    masses_kg: Any,
    target_kg: Any,
    *,
    cap_kg: float,
    tank_capacity_kg: float,
    current_time_s: float,
) -> PrimaryTargetTransition:
    """Return a transition that caps target distance from current masses."""

    masses = _three(masses_kg)
    target = _three(target_kg)
    cap = max(float(cap_kg), 0.0)
    delta = np.clip(target - masses, -cap, cap)
    capped_target = np.clip(masses + delta, 0.0, float(tank_capacity_kg))
    return PrimaryTargetTransition(
        anchor_masses_kg=masses.copy(),
        delta_kg=delta.copy(),
        target_kg=capped_target.copy(),
        initialized=True,
        refreshed=True,
        reused=False,
        resumed=False,
        last_reset_s=float(current_time_s),
    )


def primary_target_proposal_metrics(
    masses_kg: Any,
    proposal_delta_kg: Any,
    current_target_kg: Any,
    *,
    tank_capacity_kg: float,
    water_density_kg_m3: float = 1025.0,
) -> PrimaryTargetProposalMetrics:
    """Compare a candidate target proposal against current target state."""

    masses = _three(masses_kg)
    delta = _three(proposal_delta_kg)
    current_target = _three(current_target_kg)
    proposal = np.clip(masses + delta, 0.0, float(tank_capacity_kg))
    diff = proposal - current_target
    density = max(float(water_density_kg_m3), 1e-9)
    return PrimaryTargetProposalMetrics(
        proposal_kg=proposal.copy(),
        target_delta_mean_kg=float(np.mean(np.abs(diff))),
        target_delta_m3=float(np.sum(np.abs(diff)) / density),
        after_delta_mean_kg=float(np.mean(np.abs(proposal - masses))),
    )


def consume_primary_refresh_owner(
    target: Any,
    *,
    default: str = "unset",
) -> str:
    """Consume the pending refresh owner into the committed owner field."""

    owner = str(getattr(target, "_primary_refresh_owner_pending", default) or default)
    target._primary_refresh_owner = owner
    target._primary_refresh_owner_pending = "unset"
    return owner


def remember_paused_primary_target(
    target: Any,
    *,
    action: str,
    avec: Any,
) -> None:
    """Remember the current primary target for pause-mode resume."""

    target._paused_primary_target_kg = target._primary_target_kg.copy()
    target._paused_primary_delta_kg = target._primary_delta_kg.copy()
    target._paused_primary_avec = np.asarray(avec, dtype=float).reshape(2).copy()
    target._paused_primary_action = str(action)
    target._paused_primary_valid = True


def apply_primary_target_transition(
    target: Any,
    transition: PrimaryTargetTransition,
) -> None:
    """Apply a primary target transition to a provider-like object."""

    target._primary_anchor_masses_kg = transition.anchor_masses_kg.copy()
    target._primary_delta_kg = transition.delta_kg.copy()
    target._primary_target_kg = transition.target_kg.copy()
    target._primary_target_initialized = bool(transition.initialized)
    target._primary_target_refreshed = bool(transition.refreshed)
    target._primary_target_reused = bool(transition.reused)
    target._primary_target_resumed = bool(transition.resumed)
    target._primary_target_last_reset_s = float(transition.last_reset_s)


def mark_primary_target_reused(
    target: Any,
    *,
    refresh_owner: str,
) -> None:
    """Mark an existing primary target as reused for this planner bucket."""

    target._primary_target_refreshed = False
    target._primary_target_reused = True
    target._primary_target_resumed = False
    target._primary_refresh_owner_pending = str(refresh_owner)
    consume_primary_refresh_owner(target, default="reuse")


def primary_target_state_defaults(
    anchor_masses_kg: Any,
    *,
    refresh_owner: str,
) -> dict[str, Any]:
    """Return the provider attributes for a fresh primary target state."""

    anchor = _anchor_masses(anchor_masses_kg)
    zeros3 = np.zeros(3, dtype=float)
    return {
        "_primary_anchor_masses_kg": anchor.copy(),
        "_primary_delta_kg": zeros3.copy(),
        "_primary_target_kg": anchor.copy(),
        "_primary_target_initialized": False,
        "_primary_target_refreshed": False,
        "_primary_target_reused": False,
        "_primary_target_resumed": False,
        "_primary_target_last_reset_s": 0.0,
        "_primary_refresh_owner": str(refresh_owner),
        "_primary_refresh_owner_pending": "unset",
        "_paused_primary_target_kg": anchor.copy(),
        "_paused_primary_delta_kg": zeros3.copy(),
        "_paused_primary_avec": np.zeros(2, dtype=float),
        "_paused_primary_action": "hold",
        "_paused_primary_valid": False,
    }


def apply_primary_target_state_defaults(
    target: Any,
    anchor_masses_kg: Any,
    *,
    refresh_owner: str,
) -> None:
    """Apply fresh primary target attributes to a provider-like object."""

    for name, value in primary_target_state_defaults(
        anchor_masses_kg,
        refresh_owner=refresh_owner,
    ).items():
        setattr(target, name, value)
