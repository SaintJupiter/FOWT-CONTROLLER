"""Ballast planner core — A1.4 necessity-gate rolling planner.

All physics, gate logic, and comparator live here. Run scripts just
call run_planner_on_windows() and save the resulting DataFrames.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from functools import cmp_to_key
from typing import Any

import numpy as np

BLOCKS = (
    ("block1_0_20m", 0, 2),
    ("block2_20_40m", 2, 4),
    ("block3_40_60m", 4, 6),
)

ACTIONS = ("hold", "pump_saving", "active_small", "active_medium", "active_reverse_small")


@dataclass(frozen=True)
class PlannerConfig:
    deadband_pitch_deg: float = 1.0
    deadband_roll_deg: float = 0.8
    tank_capacity_kg: float = 1850.0 * 1025.0
    leak: float = 0.90
    pressure_sign_multiplier: float = -1.0
    pressure_norm_cap: float = 1.5
    action_mass_quantum_kg: float = 180000.0
    active_small_ratio: float = 0.15
    active_medium_ratio: float = 0.35
    pump_saving_ratio: float = 0.08
    reverse_small_ratio: float = 0.12
    upper_capacity_guard_ratio: float = 0.92
    lower_capacity_guard_ratio: float = 0.08
    fullspeed_guard: bool = True
    dwell_blocks_required_for_reverse: int = 1
    minimum_overcomp_ratio_for_reverse: float = 0.25
    terminal_band_abs: float = 0.05
    terminal_band_rel: float = 0.10
    active_medium_hold_term_min: float = 0.20
    active_medium_abs_improve_min: float = 0.05
    attitude_band_abs: float = 0.10
    attitude_band_rel: float = 0.10
    # Economic-form objective (zone-MPC style, see configs/planner_envelope.json).
    pitch_envelope_deg: float = 1.2
    roll_envelope_deg: float = 0.96
    w_pump_work: float = 1.0
    w_pump_duration: float = 0.4
    w_startstop: float = 0.3
    w_direction_switch: float = 0.6
    w_reverse_penalty: float = 0.5
    # State-feedback cost weights. With these non-zero, current attitude error
    # (after posture_state_residual injection) actually enters the scalar cost,
    # not just attitude_residual_cost being computed and ignored. Verified by
    # scripts/analysis/single_bucket_posture_cost_check.py.
    w_attitude_residual: float = 1.0
    w_terminal_residual: float = 1.0
    w_envelope_soft: float = 25.0
    w_terminal_envelope_soft: float = 25.0
    w_saturation_hard: float = 1000.0
    # Zone-MPC structural switches. Defaults preserve current production behaviour.
    envelope_use_discount: bool = True   # if False, envelope_norm uses raw (undiscounted) pressure
    envelope_barrier_active: bool = False  # if True, add barrier_const when max_env_norm>1 AND only weak actions used
    envelope_barrier_const: float = 50.0   # added to scalar cost when barrier triggers
    envelope_barrier_strong_actions: tuple = ("active_small", "active_medium", "active_reverse_small")
    # Posture-state forecast credit. This can discount the measured current
    # posture residual only when directional future relief is present.
    posture_hold_relief_margin_norm: float = 0.25
    # posture_hold_forecast_credit: weight by which forecast evidence is allowed
    # to discount the current-posture penalty. Default 0.0 enforces strict
    # fairness between learned/persistence/reactive: forecast can only affect
    # cost through the rollout pressure_vec (state-feedback MPC), NOT through
    # an extra credit gated on forecast_has_future. Set >0 only as a
    # learned-only diagnostic, never in a comparison main line.
    posture_hold_forecast_credit: float = 0.0
    # Hold-relief debt (default-off). This is a small planner-cost penalty for
    # weak/hold first actions after a previous high-posture hold was justified
    # by future relief or quiet pressure, but the next bucket did not actually
    # show attitude recovery. It does not force an action; it only makes
    # unfulfilled waiting less free in the scalar objective.
    hold_relief_debt_active: bool = False
    hold_relief_debt_weight: float = 1.0
    hold_relief_debt_pump_saving_factor: float = 0.6
    # Include the measured current posture as the initial residual state in
    # the planner rollout. This is closer to a state-feedback MPC than a pure
    # wind-pressure feedforward planner: a hold action must explain both future
    # wind pressure and the attitude error that already exists.
    # Default True so the planner sees current attitude in all forecast modes.
    posture_state_residual_active: bool = True
    posture_state_gain: float = 0.40
    posture_state_decay: float = 0.85
    posture_state_clip_norm: float = 6.0
    # Attitude cost zone form (objective-v4 prototype, default-off).
    # "quadratic" = current behavior: cost = residual_norm**2 everywhere.
    #   Bistable: with w_attitude=0 hold always wins (high-hold parked); with
    #   w_attitude>0 quadratic dominates pump, active over-fires.
    # "smooth_huber" = deadzone form: cost = max(0, residual_norm - delta)**2.
    #   Inside the zone (residual <= delta): zero penalty, hold wins on pump.
    #   Outside the zone: quadratic-from-boundary, pump can win at marginal
    #   excess and active wins at large excess. Forecast-aware via rollout
    #   without any extra credit/debt mechanism. Verified by
    #   scripts/analysis/cost_replay_attitude_zone_v4.py.
    attitude_zone_form: str = "quadratic"
    attitude_zone_delta_norm: float = 1.5
    attitude_zone_terminal_delta_norm: float = 1.5
    # Objective-mode switch (default-off prototype). When True, the provider
    # determines a mode per bucket based on (persistent_high_posture,
    # future_rising_pressure) and constructs an effective cfg that swaps
    # cost shape:
    #   economic mode (default, includes future_rising):
    #       quadratic attitude cost, w_attitude=1. Keeps preemptive response
    #       sensitivity when future risk is rising.
    #   recovery mode (persistent_high_posture AND NOT future_rising):
    #       smooth_huber attitude cost (deadzone), w_attitude=3. Prevents
    #       over-pumping when posture is parked at moderate-to-high values
    #       but no future risk is approaching.
    objective_mode_active: bool = False
    recovery_mode_posture_norm_threshold: float = 1.5
    # enter_buckets=1: latch recovery immediately on first high-posture bucket.
    # Higher values introduce warmup (first N buckets run in economic/quadratic
    # before recovery activates), which over-pumps lowrisk windows that start
    # at high posture (e.g., lowrisk_clean fr_relief at t=0).
    recovery_mode_enter_buckets: int = 1
    # exit_buckets=2: require 2 consecutive sub-threshold buckets to unlatch,
    # providing hysteresis against chattering.
    recovery_mode_exit_buckets: int = 2
    recovery_mode_future_rising_norm_diff: float = 0.3
    recovery_mode_w_attitude: float = 3.0
    recovery_mode_w_terminal: float = 3.0
    # Wider deadzone in recovery mode (2.5 vs 1.5 in economic) covers the
    # naturally stable high-posture residual (~1.98-2.5 in deadband-normalized
    # units, e.g. lowrisk_clean pitch=4.95deg). Inside this zone the smooth_huber
    # cost is zero, so the planner does not waste pump trying to push a
    # naturally-stable posture lower. The economic mode (when not in recovery)
    # uses the standard attitude_zone_delta_norm (1.5) for normal operation.
    recovery_mode_zone_delta_norm: float = 2.5

    # ------------------------------------------------------------------
    # Safety-floor hard constraint (default-off; safety_floor_v1 profile).
    # When ``safety_floor_active`` is True, any candidate sequence whose
    # rollout would push the unattenuated predicted attitude above
    # ``safety_floor_pitch_deg`` / ``safety_floor_roll_deg`` at ANY rollout
    # block is hard-rejected with reason ``safety_floor_violated``. The
    # safety residual is computed from raw posture (no posture_state_gain
    # attenuation, no forecast credit) so that a 5 deg actual pitch cannot
    # be hidden behind a 2 deg gain-attenuated planner residual.
    # ``safety_floor_*_deg`` are interpreted as deadband-normalized norms
    # (i.e. degrees because deadband_pitch_deg=1.0 and deadband_roll_deg=0.8
    # in the canonical config). 5.0 corresponds to the IEC envelope.
    # All-rejected case (no candidate sequence passes) is surfaced via
    # ``hard_reject_reason='safety_floor_violated_at_block_K'`` so the
    # provider can route the bucket through the reactive bridge.
    safety_floor_active: bool = False
    safety_floor_pitch_deg: float = 5.0
    safety_floor_roll_deg: float = 5.0
    # Current-state hard gate. When current measured pitch/roll exceeds
    # safety_floor_(pitch|roll)_deg - safety_floor_current_engage_margin_deg
    # the planner is forbidden from selecting a "hold" or "pump_saving" first
    # action (which would let posture drift). The engage margin defaults to
    # 1.0deg: with safety_floor=5.0deg, engagement starts at 4.0deg actual
    # pitch. This is independent of forecast — closes the gap where the LSTM
    # under-predicts continued high wind and the rollout-based safety check
    # is fooled (b_decay_strong bucket 9: pitch_at_t=5.54deg but planner
    # predicted residual stayed below 5deg under the optimistic forecast).
    safety_floor_current_engage_margin_deg: float = 1.0


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def norm_term(vec: np.ndarray, cfg: PlannerConfig) -> float:
    db = np.array([cfg.deadband_pitch_deg, cfg.deadband_roll_deg], dtype=float)
    return float(np.linalg.norm(vec / np.maximum(db, 1e-6)))


def axis_deadband_vec(vec: np.ndarray, cfg: PlannerConfig) -> np.ndarray:
    """Zero each attitude axis independently before vector normalization."""
    arr = np.asarray(vec, dtype=float).reshape(-1)
    if arr.size < 2:
        arr = np.pad(arr, (0, 2 - arr.size))
    arr = arr[:2].copy()
    db = np.array([cfg.deadband_pitch_deg, cfg.deadband_roll_deg], dtype=float)
    arr[np.abs(arr) < np.maximum(db, 1e-6)] = 0.0
    return arr


def envelope_norm(vec: np.ndarray, cfg: PlannerConfig) -> float:
    """Residual norm relative to the envelope, NOT the deadband. <=1.0 means inside the envelope."""
    env = np.array([cfg.pitch_envelope_deg, cfg.roll_envelope_deg], dtype=float)
    return float(np.linalg.norm(vec / np.maximum(env, 1e-6)))


def posture_state_vec(plant_info: dict[str, Any], cfg: PlannerConfig) -> np.ndarray:
    """Map measured pitch/roll attitude into planner-frame corrective demand."""
    if not cfg.posture_state_residual_active:
        return np.zeros(2, dtype=float)
    posture_vec = np.asarray(
        plant_info.get("posture_vec_deg", np.zeros(2, dtype=float)),
        dtype=float,
    ).reshape(-1)
    if posture_vec.size < 2:
        posture_vec = np.pad(posture_vec, (0, 2 - posture_vec.size))
    posture_vec = posture_vec[:2]
    db = np.array([cfg.deadband_pitch_deg, cfg.deadband_roll_deg], dtype=float)
    normalized = posture_vec / np.maximum(db, 1e-6)
    clipped = np.clip(
        normalized,
        -max(float(cfg.posture_state_clip_norm), 0.0),
        max(float(cfg.posture_state_clip_norm), 0.0),
    )
    # Same sign convention as the planner-frame posture correction path:
    # measured negative pitch should produce a negative planner-frame correction
    # vector. The planner's action_vec then maps that vector into the matching
    # tank target direction.
    return clipped * max(float(cfg.posture_state_gain), 0.0) * db


def posture_state_context(
    blocks: list[dict[str, Any]],
    plant_info: dict[str, Any],
    cfg: PlannerConfig,
) -> tuple[np.ndarray, np.ndarray, float, str]:
    """Return raw/effective current-posture residual vectors and preview credit."""
    raw_posture = posture_state_vec(plant_info, cfg)
    credit = 0.0
    credit_reason = "disabled"
    if cfg.posture_state_residual_active and norm_term(raw_posture, cfg) > 1e-9:
        credit, credit_reason = posture_hold_forecast_credit(blocks, plant_info, cfg)
    effective = raw_posture * max(0.0, 1.0 - float(credit))
    return raw_posture, effective, float(credit), str(credit_reason)


def apply_posture_state_to_blocks(
    blocks: list[dict[str, Any]],
    plant_info: dict[str, Any],
    cfg: PlannerConfig,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Add current-posture residual to planner rollout blocks.

    The returned blocks are used only by the planner's sequence evaluation and
    action-vector generation. The original forecast blocks should still be used
    for diagnostics and forecast-only gates.
    """
    raw_posture, posture, credit, credit_reason = posture_state_context(
        blocks,
        plant_info,
        cfg,
    )
    # Unattenuated posture (gain=1.0) for safety-floor hard check. Uses the
    # same clip as posture_state_vec so out-of-range readings still get
    # bounded, but does not apply ``posture_state_gain`` so the safety check
    # sees actual measured pitch/roll rather than the attenuated planner view.
    posture_vec_raw = np.asarray(
        plant_info.get("posture_vec_deg", np.zeros(2, dtype=float)),
        dtype=float,
    ).reshape(-1)
    if posture_vec_raw.size < 2:
        posture_vec_raw = np.pad(posture_vec_raw, (0, 2 - posture_vec_raw.size))
    posture_vec_raw = posture_vec_raw[:2]
    db = np.array([cfg.deadband_pitch_deg, cfg.deadband_roll_deg], dtype=float)
    posture_unattenuated_norm = np.clip(
        posture_vec_raw / np.maximum(db, 1e-6),
        -max(float(cfg.posture_state_clip_norm), 0.0),
        max(float(cfg.posture_state_clip_norm), 0.0),
    )
    posture_unattenuated = posture_unattenuated_norm * db
    adjusted: list[dict[str, Any]] = []
    for idx, block in enumerate(blocks):
        item = dict(block)
        decay = float(cfg.posture_state_decay) ** int(idx)
        residual = posture * decay
        raw_residual = raw_posture * decay
        safety_residual = posture_unattenuated * decay
        p = np.asarray(block.get("pressure_vec", np.zeros(2)), dtype=float)
        p_raw = np.asarray(block.get("pressure_vec_raw", p), dtype=float)
        item["pressure_vec"] = p + residual
        # Raw-envelope mode should still use the forecast-credited posture
        # residual. Otherwise the raw envelope path silently cancels the
        # forecast credit and makes learned/current-only behave the same.
        item["pressure_vec_raw"] = p_raw + residual
        # Safety view: forecast pressure + unattenuated posture, used ONLY by
        # the safety-floor hard reject. Does NOT enter the cost function.
        item["pressure_vec_safety"] = p_raw + safety_residual
        item["pressure_norm"] = norm_term(np.asarray(item["pressure_vec"], dtype=float), cfg)
        item["posture_state_residual_vec"] = residual
        item["posture_state_raw_residual_vec"] = raw_residual
        item["posture_state_safety_residual_vec"] = safety_residual
        adjusted.append(item)
    meta = {
        "posture_state_norm": norm_term(posture, cfg),
        "posture_state_raw_norm": norm_term(raw_posture, cfg),
        "posture_state_credit": float(credit),
        "posture_state_credit_reason": credit_reason,
    }
    return adjusted, meta


def posture_hold_forecast_credit(blocks: list[dict[str, Any]], plant_info: dict[str, Any],
                                 cfg: PlannerConfig) -> tuple[float, str]:
    """Credit a weak/hold first action only when forecast relieves current posture.

    A current-only baseline deliberately repeats current wind as an unknown
    future placeholder. Without the explicit ``forecast_has_future`` guard,
    repeated or modified placeholder blocks can look like future relief and
    incorrectly excuse holding a large posture.

    The credit is directional: a lower future pressure norm is not enough by
    itself. The future pressure must relax the component aligned with the
    measured pitch/roll error. This prevents the planner from treating a generic
    wind decay as permission to keep a persistent 3-6 deg attitude offset.
    """
    if not bool(plant_info.get("forecast_has_future", 0)):
        return 0.0, "no_future_forecast"
    if len(blocks) < 3:
        return 0.0, "missing_future_blocks"
    posture_vec = np.asarray(
        plant_info.get("posture_vec_deg", np.zeros(2, dtype=float)),
        dtype=float,
    ).reshape(-1)
    if posture_vec.size < 2:
        posture_vec = np.pad(posture_vec, (0, 2 - posture_vec.size))
    posture_vec = posture_vec[:2]
    posture_mag = float(np.linalg.norm(posture_vec))
    if posture_mag <= 1e-9:
        return 0.0, "no_current_posture"
    posture_dir = posture_vec / posture_mag
    raw_vecs = [
        np.asarray(b.get("pressure_vec_raw", b.get("pressure_vec", np.zeros(2))), dtype=float)
        for b in blocks[:3]
    ]
    norms = [norm_term(v, cfg) for v in raw_vecs]
    vec0 = raw_vecs[0]
    vec2 = raw_vecs[2]
    proj0 = float(np.dot(vec0, posture_dir))
    proj2 = float(np.dot(vec2, posture_dir))
    directional_relief = max(0.0, proj0 - proj2)
    relief_credit = min(
        1.0,
        directional_relief / max(float(cfg.posture_hold_relief_margin_norm), 1e-6),
    )
    # Sign reversals only count when they reverse the component that is aligned
    # with the measured posture, not merely any two forecast vectors.
    signflip_credit = 1.0 if (proj0 > 0.0 and proj2 < 0.0) else 0.0
    credit = max(relief_credit, signflip_credit) * float(cfg.posture_hold_forecast_credit)
    if credit <= 1e-9:
        return 0.0, "no_posture_relief_or_signflip"
    if signflip_credit >= relief_credit:
        return float(min(max(credit, 0.0), 1.0)), "future_posture_signflip_credit"
    return float(min(max(credit, 0.0), 1.0)), "future_posture_relief_credit"


def pressure_proxy_vec(uv_block: np.ndarray, cfg: PlannerConfig, wind_ref: float = 12.0) -> np.ndarray:
    speed = np.sqrt(uv_block[:, 0] ** 2 + uv_block[:, 1] ** 2)
    mean_u = float(np.mean(uv_block[:, 0]))
    mean_v = float(np.mean(uv_block[:, 1]))
    mean_speed = float(np.mean(speed))
    mean_dir = float((np.rad2deg(np.arctan2(-mean_u, -mean_v)) + 360.0) % 360.0)
    mag = float(
        np.clip(
            (max(mean_speed, 0.0) / max(wind_ref, 1.0)) ** 2,
            0.0,
            max(float(cfg.pressure_norm_cap), 1e-6),
        )
    )
    wd_rad = math.radians(mean_dir)
    raw = np.array(
        [-cfg.deadband_pitch_deg * mag * math.cos(wd_rad),
          cfg.deadband_roll_deg * mag * math.sin(wd_rad)],
        dtype=float,
    )
    return cfg.pressure_sign_multiplier * raw


def action_vec(name: str, pressure_vec: np.ndarray, cfg: PlannerConfig,
               previous_vec: np.ndarray | None = None) -> np.ndarray:
    db = np.array([cfg.deadband_pitch_deg, cfg.deadband_roll_deg], dtype=float)
    demand_vec = axis_deadband_vec(pressure_vec, cfg)
    active_axes = np.abs(demand_vec) > 0.0
    if float(np.linalg.norm(demand_vec / np.maximum(db, 1e-6))) <= 1e-9:
        direction = np.zeros(2, dtype=float)
    else:
        direction = demand_vec / max(float(np.linalg.norm(demand_vec)), 1e-9)
    if name == "hold":
        return np.zeros(2, dtype=float)
    if name == "pump_saving":
        if previous_vec is not None and float(np.linalg.norm(previous_vec)) > 1e-9:
            prev_dir = previous_vec / max(float(np.linalg.norm(previous_vec)), 1e-9)
            direction = 0.6 * direction + 0.4 * prev_dir
            direction[~active_axes] = 0.0
            n = float(np.linalg.norm(direction))
            if n > 1e-9:
                direction = direction / n
            else:
                direction = np.zeros(2, dtype=float)
        return direction * db * cfg.pump_saving_ratio
    if name == "active_small":
        return direction * db * cfg.active_small_ratio
    if name == "active_medium":
        return direction * db * cfg.active_medium_ratio
    if name == "active_reverse_small":
        return -direction * db * cfg.reverse_small_ratio
    raise KeyError(name)


def planner_action_vec(
    name: str,
    k: int,
    blocks: list[dict[str, Any]],
    plant_info: dict[str, Any],
    cfg: PlannerConfig,
    previous_vec: np.ndarray | None = None,
) -> tuple[np.ndarray, bool]:
    """Action vector used by both sequence scoring and the provider output."""
    p = np.asarray(blocks[k].get("pressure_vec", np.zeros(2)), dtype=float)
    return action_vec(name, p, cfg, previous_vec=previous_vec), False


PHYSICAL_ATTITUDE_ALLOC = np.array(
    [
        [1.0, 0.0],
        [-0.5, -1.0],
        [-0.5, 1.0],
    ],
    dtype=float,
)

COMPENSATION_ATTITUDE_ALLOC = -PHYSICAL_ATTITUDE_ALLOC


def tank_signal(avec: np.ndarray, cfg: PlannerConfig) -> np.ndarray:
    """Convert planner compensation vector to per-tank mass-change signal.

    `avec` lives in the planner compensation frame: positive components mean
    "move ballast to oppose positive physical pitch/roll". Therefore this
    matrix is the negative of the physical attitude allocation matrix used by
    the closed-loop MIMO controller.
    """
    alloc = COMPENSATION_ATTITUDE_ALLOC
    db = np.array([cfg.deadband_pitch_deg, cfg.deadband_roll_deg], dtype=float)
    return alloc @ (avec / np.maximum(db, 1e-6))


def update_tanks(masses: np.ndarray, avec: np.ndarray, cfg: PlannerConfig) -> tuple[np.ndarray, float]:
    sig = tank_signal(avec, cfg)
    delta = sig * cfg.action_mass_quantum_kg
    new = np.clip(masses + delta, 0.0, cfg.tank_capacity_kg)
    req_norm = float(np.linalg.norm(delta))
    exec_ratio = float(np.linalg.norm(new - masses) / req_norm) if req_norm > 1e-9 else 0.0
    return new, exec_ratio


def compute_pressure_blocks(uv: np.ndarray, discounts: list[float],
                             cfg: PlannerConfig) -> list[dict[str, Any]]:
    rows = []
    for idx, (name, s, e) in enumerate(BLOCKS):
        raw = pressure_proxy_vec(uv[s:e], cfg)
        pvec = raw * float(discounts[idx])
        rows.append({
            "block_name": name,
            "pressure_vec": pvec,
            "pressure_vec_raw": raw,
            "pressure_norm": norm_term(pvec, cfg),
        })
    return rows


# ---------------------------------------------------------------------------
# Hard constraint check
# ---------------------------------------------------------------------------

def hard_violation(masses: np.ndarray, avec: np.ndarray,
                   plant_info: dict[str, Any], cfg: PlannerConfig) -> tuple[bool, str]:
    sig = tank_signal(avec, cfg)
    ratio = masses / max(cfg.tank_capacity_kg, 1.0)
    if np.any((ratio >= cfg.upper_capacity_guard_ratio) & (sig > 0.0)):
        return True, "capacity_guard"
    if np.any((ratio <= cfg.lower_capacity_guard_ratio) & (sig < 0.0)):
        return True, "capacity_guard"
    if cfg.fullspeed_guard and int(plant_info.get("pump_fullspeed_any", 0)) and np.linalg.norm(avec) > 1e-9:
        return True, "fullspeed_guard"
    return False, ""


# ---------------------------------------------------------------------------
# Admission gates
# ---------------------------------------------------------------------------

def attitude_same_band(a: float, b: float, cfg: PlannerConfig) -> bool:
    tol = max(cfg.attitude_band_abs, cfg.attitude_band_rel * max(abs(a), abs(b), 1e-9))
    return abs(a - b) <= tol


def terminal_same_band(a: float, b: float, cfg: PlannerConfig) -> bool:
    tol = max(cfg.terminal_band_abs, cfg.terminal_band_rel * max(abs(a), abs(b), 1e-9))
    return abs(a - b) <= tol


def reverse_gate(k: int, blocks: list[dict[str, Any]], comp: np.ndarray,
                 dwell: int, cfg: PlannerConfig) -> tuple[bool, str]:
    p = blocks[k]["pressure_vec"]
    if float(np.linalg.norm(comp)) <= cfg.minimum_overcomp_ratio_for_reverse * max(blocks[k]["pressure_norm"], 1e-9):
        return False, "overcomp_too_small"
    if dwell < cfg.dwell_blocks_required_for_reverse:
        return False, "reverse_after_full_block_only"
    if k < len(blocks) - 1:
        p_next = blocks[k + 1]["pressure_vec"]
        if float(np.dot(p, p_next)) >= 0.0 and float(np.dot(comp, p)) >= 0.0:
            return False, "no_future_reversal_signal"
    r_hold = p - cfg.leak * comp
    r_rev = p - (cfg.leak * comp + action_vec("active_reverse_small", p, cfg))
    if norm_term(r_rev, cfg) + 0.05 >= norm_term(r_hold, cfg):
        return False, "terminal_not_improved"
    return True, "future_reversal_and_overcomp"


def medium_gate(k: int, blocks: list[dict[str, Any]], comp: np.ndarray,
                cfg: PlannerConfig) -> tuple[bool, str]:
    """A1.4 necessity gate: medium must be necessary relative to small, not just better than hold."""
    p = blocks[k]["pressure_vec"]
    hold_t = norm_term(p - cfg.leak * comp, cfg)
    small_t = norm_term(p - (cfg.leak * comp + action_vec("active_small", p, cfg)), cfg)
    med_t = norm_term(p - (cfg.leak * comp + action_vec("active_medium", p, cfg)), cfg)

    if hold_t < cfg.active_medium_hold_term_min:
        return False, "hold_terminal_too_low"
    if k == 0 and small_t <= 1.0:
        return False, "active_small_already_within_deadband"
    if attitude_same_band(small_t, med_t, cfg):
        return False, "small_and_medium_same_band"
    if (small_t - med_t) < cfg.active_medium_abs_improve_min:
        return False, "medium_not_significantly_better_than_small"
    return True, "active_small_insufficient_medium_required"


# ---------------------------------------------------------------------------
# Sequence evaluation
# ---------------------------------------------------------------------------

def evaluate_sequence(sequence: tuple[str, ...], blocks: list[dict[str, Any]],
                      plant_info: dict[str, Any], cfg: PlannerConfig) -> dict[str, Any]:
    comp = np.zeros(2, dtype=float)
    prev_avec = np.zeros(2, dtype=float)
    masses = np.asarray(plant_info.get("tank_masses", np.zeros(3)), dtype=float).reshape(-1)[:3]
    if masses.size < 3:
        masses = np.pad(masses, (0, 3 - masses.size))

    costs = dict(attitude_residual_cost=0.0, pump_work_cost=0.0, pump_duration_cost=0.0,
                 startstop_cost=0.0, direction_switch_cost=0.0, reverse_penalty=0.0,
                 saturation_penalty=0.0, terminal_residual_cost=0.0,
                 envelope_violation_cost=0.0, terminal_envelope_violation=0.0,
                 max_envelope_norm=0.0, envelope_barrier_triggered=0.0,
                 posture_hold_barrier_triggered=0.0, posture_hold_barrier_cost=0.0,
                 posture_hold_norm=0.0, posture_hold_forecast_credit=0.0,
                 hold_relief_debt_cost=0.0,
                 hold_relief_debt_level=float(plant_info.get("hold_relief_debt_level", 0.0)),
                 posture_hold_action_posture_directed=0.0,
                 posture_state_norm=float(plant_info.get("posture_state_norm", 0.0)),
                 posture_state_raw_norm=float(plant_info.get("posture_state_raw_norm", 0.0)),
                 posture_state_credit=float(plant_info.get("posture_state_credit", 0.0)))
    meta = dict(hard_reject_reason="", reverse_allowed=False, reverse_release_reason="",
                reverse_reject_reason="", reverse_after_full_block_only=False,
                active_medium_gate_reason="", posture_hold_barrier_reason="disabled",
                posture_state_credit_reason=str(
                    plant_info.get("posture_state_credit_reason", "disabled")
                ))
    dwell = 0
    last_sign = 0

    # Current-state hard gate (independent of forecast): if measured posture
    # already exceeds the engagement threshold, the planner cannot select
    # ``hold`` or ``pump_saving`` as the FIRST action. Closes the failure mode
    # where the LSTM under-predicts continued high wind and the rollout-based
    # safety check passes optimistic plans on a posture that is already at
    # the edge of the envelope. The engagement threshold is
    # safety_floor_(pitch|roll)_deg - safety_floor_current_engage_margin_deg.
    if cfg.safety_floor_active and len(sequence) > 0:
        posture_now = np.asarray(
            plant_info.get("posture_vec_deg", np.zeros(2, dtype=float)),
            dtype=float,
        ).reshape(-1)
        if posture_now.size >= 2:
            curr_pitch_abs = abs(float(posture_now[0]))
            curr_roll_abs = abs(float(posture_now[1]))
            engage_pitch = float(cfg.safety_floor_pitch_deg) - float(
                cfg.safety_floor_current_engage_margin_deg
            )
            engage_roll = float(cfg.safety_floor_roll_deg) - float(
                cfg.safety_floor_current_engage_margin_deg
            )
            first = str(sequence[0])
            if (
                (curr_pitch_abs > engage_pitch or curr_roll_abs > engage_roll)
                and first in ("hold", "pump_saving")
            ):
                return {
                    **meta, "sequence": sequence,
                    "hard_reject_reason": "safety_floor_current_state_engaged",
                    "costs": costs, "selection_reason": "hard_reject",
                    "safety_floor_predicted_pitch": curr_pitch_abs,
                    "safety_floor_predicted_roll": curr_roll_abs,
                }

    for k, name in enumerate(sequence):
        p = blocks[k]["pressure_vec"]

        if name == "active_reverse_small":
            ok, reason = reverse_gate(k, blocks, comp, dwell, cfg)
            meta["reverse_after_full_block_only"] = dwell >= cfg.dwell_blocks_required_for_reverse
            if not ok:
                return {**meta, "sequence": sequence,
                        "hard_reject_reason": "reverse_not_released",
                        "reverse_reject_reason": reason, "costs": costs,
                        "selection_reason": "hard_reject"}
            meta["reverse_allowed"] = True
            meta["reverse_release_reason"] = reason

        if name == "active_medium":
            ok, reason = medium_gate(k, blocks, comp, cfg)
            meta["active_medium_gate_reason"] = reason
            if not ok:
                return {**meta, "sequence": sequence,
                        "hard_reject_reason": "active_medium_gate",
                        "costs": costs, "selection_reason": "hard_reject"}

        avec, _ = planner_action_vec(
            name, k, blocks, plant_info, cfg, previous_vec=prev_avec
        )
        violated, reason = hard_violation(masses, avec, plant_info, cfg)
        if violated:
            return {**meta, "sequence": sequence, "hard_reject_reason": reason,
                    "costs": costs, "selection_reason": "hard_reject"}

        masses, exec_ratio = update_tanks(masses, avec, cfg)
        comp = cfg.leak * comp + exec_ratio * avec
        residual_vec = p - comp
        residual_norm = norm_term(residual_vec, cfg)
        # Safety-floor hard reject: check unattenuated predicted attitude
        # against the absolute pitch/roll floor. Uses pressure_vec_safety
        # (forecast pressure + raw posture, no gain) minus same comp so that
        # any planned compensation reduces the safety residual identically.
        # Independent of cost function — pure hard constraint.
        if cfg.safety_floor_active:
            p_safety = np.asarray(
                blocks[k].get("pressure_vec_safety",
                              blocks[k].get("pressure_vec_raw", p)),
                dtype=float,
            )
            safety_resid = p_safety - comp
            if (abs(float(safety_resid[0])) > float(cfg.safety_floor_pitch_deg)
                    or abs(float(safety_resid[1])) > float(cfg.safety_floor_roll_deg)):
                return {
                    **meta, "sequence": sequence,
                    "hard_reject_reason": f"safety_floor_violated_at_block_{k}",
                    "costs": costs, "selection_reason": "hard_reject",
                    "safety_floor_predicted_pitch": float(safety_resid[0]),
                    "safety_floor_predicted_roll": float(safety_resid[1]),
                }
        # envelope evaluated either on discounted or raw (undiscounted) residual.
        if cfg.envelope_use_discount:
            env_norm = envelope_norm(residual_vec, cfg)
        else:
            p_raw = blocks[k].get("pressure_vec_raw", p)
            env_norm = envelope_norm(p_raw - comp, cfg)
        if cfg.attitude_zone_form == "smooth_huber":
            zone_excess = max(0.0, residual_norm - float(cfg.attitude_zone_delta_norm))
            costs["attitude_residual_cost"] += zone_excess ** 2
        else:
            costs["attitude_residual_cost"] += residual_norm ** 2
        # Soft envelope penalty: zero when inside, quadratic when outside.
        env_excess = max(0.0, env_norm - 1.0)
        costs["envelope_violation_cost"] += env_excess ** 2
        if env_norm > costs["max_envelope_norm"]:
            costs["max_envelope_norm"] = env_norm
        db = np.array([cfg.deadband_pitch_deg, cfg.deadband_roll_deg], dtype=float)
        costs["pump_work_cost"] += float(np.linalg.norm(avec / db))
        if np.linalg.norm(avec) > 1e-9:
            costs["pump_duration_cost"] += 1.0
        if np.linalg.norm(prev_avec) <= 1e-9 and np.linalg.norm(avec) > 1e-9:
            costs["startstop_cost"] += 1.0
        if np.linalg.norm(prev_avec) > 1e-9 and np.linalg.norm(avec) > 1e-9 and float(np.dot(prev_avec, avec)) < 0.0:
            costs["direction_switch_cost"] += 1.0
        if name == "active_reverse_small":
            costs["reverse_penalty"] += 1.0
        ratio = masses / max(cfg.tank_capacity_kg, 1.0)
        costs["saturation_penalty"] += float(
            max(0.0, float(np.max(ratio - cfg.upper_capacity_guard_ratio)))
            + max(0.0, float(np.max(cfg.lower_capacity_guard_ratio - ratio)))
        )
        if np.linalg.norm(avec) > 1e-9:
            sign = 1 if float(np.dot(avec, p)) >= 0.0 else -1
            dwell = dwell + 1 if sign == last_sign else 1
            last_sign = sign
        prev_avec = avec

    terminal_vec = blocks[-1]["pressure_vec"] - comp
    terminal_norm_val = norm_term(terminal_vec, cfg)
    if cfg.attitude_zone_form == "smooth_huber":
        terminal_excess = max(0.0, terminal_norm_val - float(cfg.attitude_zone_terminal_delta_norm))
        costs["terminal_residual_cost"] = terminal_excess ** 2
    else:
        costs["terminal_residual_cost"] = terminal_norm_val ** 2
    if cfg.envelope_use_discount:
        terminal_env = envelope_norm(terminal_vec, cfg)
    else:
        terminal_raw = blocks[-1].get("pressure_vec_raw", blocks[-1]["pressure_vec"])
        terminal_env = envelope_norm(terminal_raw - comp, cfg)
    costs["terminal_envelope_violation"] = max(0.0, terminal_env - 1.0) ** 2
    if terminal_env > costs["max_envelope_norm"]:
        costs["max_envelope_norm"] = terminal_env
    # zone-MPC barrier flag: True if envelope violated AND no strong action used.
    has_strong = any(a in cfg.envelope_barrier_strong_actions for a in sequence)
    costs["envelope_barrier_triggered"] = float(
        cfg.envelope_barrier_active and costs["max_envelope_norm"] > 1.0 and not has_strong
    )
    if cfg.hold_relief_debt_active:
        first_action = str(sequence[0]) if sequence else "hold"
        debt_level = max(0.0, float(plant_info.get("hold_relief_debt_level", 0.0)))
        if debt_level > 0.0 and first_action in ("hold", "pump_saving"):
            factor = (
                1.0
                if first_action == "hold"
                else max(0.0, float(cfg.hold_relief_debt_pump_saving_factor))
            )
            costs["hold_relief_debt_cost"] = float(debt_level * factor)
    return {**meta, "sequence": sequence, "costs": costs, "selection_reason": "feasible"}


# ---------------------------------------------------------------------------
# Comparator
# ---------------------------------------------------------------------------

def compare_sequences(a: dict[str, Any], b: dict[str, Any],
                      cfg: PlannerConfig) -> tuple[int, str, int]:
    ca, cb = a["costs"], b["costs"]
    dims = [
        ("terminal_residual",          ca["terminal_residual_cost"],   cb["terminal_residual_cost"]),
        ("attitude_residual",          ca["attitude_residual_cost"],    cb["attitude_residual_cost"]),
        ("direction_switch_plus_reverse",
         ca["direction_switch_cost"] + ca["reverse_penalty"],
         cb["direction_switch_cost"] + cb["reverse_penalty"]),
        ("saturation_margin",          ca["saturation_penalty"],        cb["saturation_penalty"]),
        ("pump_work_proxy",            ca["pump_work_cost"],             cb["pump_work_cost"]),
        ("pump_duration_proxy",        ca["pump_duration_cost"],         cb["pump_duration_cost"]),
        ("startstop_proxy",            ca["startstop_cost"],             cb["startstop_cost"]),
    ]
    name, va, vb = dims[0]
    if not terminal_same_band(va, vb, cfg):
        return (-1 if va < vb else 1), name, 1
    name, va, vb = dims[1]
    if not attitude_same_band(va, vb, cfg):
        return (-1 if va < vb else 1), name, 2
    for depth, (name, va, vb) in enumerate(dims[2:], start=3):
        if abs(va - vb) > 1e-9:
            return (-1 if va < vb else 1), name, depth
    return 0, "tie", len(dims)


def _total_cost(row: dict[str, Any]) -> float:
    c = row["costs"]
    return sum(c[k] for k in ("terminal_residual_cost", "attitude_residual_cost",
                               "direction_switch_cost", "reverse_penalty",
                               "saturation_penalty", "pump_work_cost",
                               "pump_duration_cost", "startstop_cost"))


def economic_scalar_cost(row: dict[str, Any], cfg: PlannerConfig) -> float:
    """Zone-MPC scalar objective.

    J = w_pump_work * pump_work
        + w_pump_duration * pump_duration
        + w_startstop * startstop
        + w_direction_switch * direction_switch
        + w_reverse_penalty * reverse_penalty
        + w_envelope_soft * sum_k max(0, env_norm_k - 1)^2
        + w_terminal_envelope_soft * max(0, terminal_env_norm - 1)^2
        + w_saturation_hard * saturation_penalty

    Notes:
      - Inside the envelope (env_norm <= 1) the soft penalty is zero, so
        the planner is free to optimize for pump cost.
      - Outside the envelope the quadratic penalty grows fast (weight 25),
        so the planner will only accept envelope violation if pump savings
        are large enough to justify it (or no inside-envelope plan exists).
      - saturation is kept as a quasi-hard barrier with weight 1000.
    """
    c = row["costs"]
    # When attitude_zone_form == "smooth_huber", attitude_residual_cost and
    # terminal_residual_cost already encode a deadzone-quadratic penalty on
    # residual. The legacy envelope_violation_cost / terminal_envelope_violation
    # are also quadratic-above-threshold (threshold = 1.2deg envelope), so they
    # are redundant with the new zone term and would dominate via w=25. Disable
    # the envelope soft terms in this objective form to keep the cost shape
    # interpretable as a single zone-MPC penalty.
    if cfg.attitude_zone_form == "smooth_huber":
        envelope_soft_eff = 0.0
        terminal_envelope_soft_eff = 0.0
    else:
        envelope_soft_eff = cfg.w_envelope_soft
        terminal_envelope_soft_eff = cfg.w_terminal_envelope_soft
    return (
        cfg.w_pump_work * c["pump_work_cost"]
        + cfg.w_pump_duration * c["pump_duration_cost"]
        + cfg.w_startstop * c["startstop_cost"]
        + cfg.w_direction_switch * c["direction_switch_cost"]
        + cfg.w_reverse_penalty * c["reverse_penalty"]
        + cfg.w_attitude_residual * c["attitude_residual_cost"]
        + cfg.w_terminal_residual * c["terminal_residual_cost"]
        + envelope_soft_eff * c["envelope_violation_cost"]
        + terminal_envelope_soft_eff * c["terminal_envelope_violation"]
        + cfg.w_saturation_hard * c["saturation_penalty"]
        + cfg.envelope_barrier_const * c.get("envelope_barrier_triggered", 0.0)
        + cfg.hold_relief_debt_weight * c.get("hold_relief_debt_cost", 0.0)
    )


def select_best_economic(evaluated: list[dict[str, Any]], cfg: PlannerConfig
                         ) -> tuple[list[dict[str, Any]], list[float]]:
    """Sort feasible sequences by economic scalar cost and return (sorted, scalar_costs)."""
    feasible = [r for r in evaluated if not r["hard_reject_reason"]]
    scalars = [(economic_scalar_cost(r, cfg), r) for r in feasible]
    scalars.sort(key=lambda t: t[0])
    return [r for _, r in scalars], [s for s, _ in scalars]


def gap_pct(best: dict[str, Any], second: dict[str, Any]) -> float:
    va, vb = _total_cost(best), _total_cost(second)
    return float((vb - va) / max(abs(va), 1e-9) * 100.0)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

DEFAULT_PLANT_INFO: dict[str, Any] = {
    "tank_masses": np.array([1108000.0, 1362000.0, 1362000.0], dtype=float),
    "pump_fullspeed_any": 0,
    "pump_total_backlog_kg": 0.0,
    "pump_rate_cmd_m3_min": np.array([0.0, 0.0, 0.0], dtype=float),
}

_ALL_SEQUENCES = [(a, b, c) for a in ACTIONS for b in ACTIONS for c in ACTIONS]


def run_planner_on_windows(
    windows: Any,
    replay_dataset: Any,
    cfg: PlannerConfig,
    discount_profiles: dict[str, list[float]],
    plant_info: dict[str, Any] | None = None,
    timestamp_fmt: str = "%Y-%m-%d %H:%M:%S",
) -> Any:
    """Evaluate the A1.4 planner for every (window, discount_profile) pair.

    Returns a flat DataFrame with one row per (window × profile).
    """
    from datetime import datetime
    import pandas as pd

    if plant_info is None:
        plant_info = DEFAULT_PLANT_INFO

    rows = []
    for _, win in windows.iterrows():
        ts = str(win["prediction_timestamp"])
        sample = replay_dataset.sample_for_history_end(datetime.strptime(ts, timestamp_fmt))
        if sample is None:
            continue
        uv = np.asarray(sample.y_uv_raw, dtype=float)
        for profile_name, discounts in discount_profiles.items():
            blocks = compute_pressure_blocks(uv, discounts, cfg)
            evaluated = [evaluate_sequence(seq, blocks, plant_info, cfg) for seq in _ALL_SEQUENCES]
            feasible = sorted(
                [r for r in evaluated if not r["hard_reject_reason"]],
                key=cmp_to_key(lambda a, b: compare_sequences(a, b, cfg)[0]),
            )
            if not feasible:
                continue
            best = feasible[0]
            second = feasible[1] if len(feasible) > 1 else best
            win_dim, depth = compare_sequences(best, second, cfg)[1:]
            rows.append({
                "discount_profile": profile_name,
                "a1_group": str(win.get("a1_group", "")),
                "prediction_timestamp": ts,
                "best_sequence": ">".join(best["sequence"]),
                "second_best_sequence": ">".join(second["sequence"]),
                "first_action": best["sequence"][0],
                "best_vs_second_gap_pct": gap_pct(best, second),
                "winning_dimension": win_dim,
                "lexicographic_decision_depth": depth,
                **{k: float(v) for k, v in best["costs"].items()},
                "reverse_allowed": int(bool(best["reverse_allowed"])),
                "reverse_release_reason": str(best["reverse_release_reason"]),
                "reverse_reject_reason": str(best["reverse_reject_reason"]),
                "active_medium_gate_reason": str(best["active_medium_gate_reason"]),
            })
    return pd.DataFrame(rows)
