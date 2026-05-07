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
import pandas as pd

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
    w_envelope_soft: float = 25.0
    w_terminal_envelope_soft: float = 25.0
    w_saturation_hard: float = 1000.0
    # Zone-MPC structural switches. Defaults preserve current production behaviour.
    envelope_use_discount: bool = True   # if False, envelope_norm uses raw (undiscounted) pressure
    envelope_barrier_active: bool = False  # if True, add barrier_const when max_env_norm>1 AND only weak actions used
    envelope_barrier_const: float = 50.0   # added to scalar cost when barrier triggers
    envelope_barrier_strong_actions: tuple = ("active_small", "active_medium", "active_reverse_small")


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def norm_term(vec: np.ndarray, cfg: PlannerConfig) -> float:
    db = np.array([cfg.deadband_pitch_deg, cfg.deadband_roll_deg], dtype=float)
    return float(np.linalg.norm(vec / np.maximum(db, 1e-6)))


def envelope_norm(vec: np.ndarray, cfg: PlannerConfig) -> float:
    """Residual norm relative to the envelope, NOT the deadband. <=1.0 means inside the envelope."""
    env = np.array([cfg.pitch_envelope_deg, cfg.roll_envelope_deg], dtype=float)
    return float(np.linalg.norm(vec / np.maximum(env, 1e-6)))


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
    if float(np.linalg.norm(pressure_vec / np.maximum(db, 1e-6))) <= 1e-9:
        direction = np.zeros(2, dtype=float)
    else:
        direction = pressure_vec / max(float(np.linalg.norm(pressure_vec)), 1e-9)
    if name == "hold":
        return np.zeros(2, dtype=float)
    if name == "pump_saving":
        if previous_vec is not None and float(np.linalg.norm(previous_vec)) > 1e-9:
            prev_dir = previous_vec / max(float(np.linalg.norm(previous_vec)), 1e-9)
            direction = 0.6 * direction + 0.4 * prev_dir
            n = float(np.linalg.norm(direction))
            if n > 1e-9:
                direction = direction / n
        return direction * db * cfg.pump_saving_ratio
    if name == "active_small":
        return direction * db * cfg.active_small_ratio
    if name == "active_medium":
        return direction * db * cfg.active_medium_ratio
    if name == "active_reverse_small":
        return -direction * db * cfg.reverse_small_ratio
    raise KeyError(name)


def tank_signal(avec: np.ndarray, cfg: PlannerConfig) -> np.ndarray:
    alloc = np.array([[-1.0, 0.0], [0.5, 1.0], [0.5, -1.0]], dtype=float)
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
                 max_envelope_norm=0.0, envelope_barrier_triggered=0.0)
    meta = dict(hard_reject_reason="", reverse_allowed=False, reverse_release_reason="",
                reverse_reject_reason="", reverse_after_full_block_only=False,
                active_medium_gate_reason="")
    dwell = 0
    last_sign = 0

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

        avec = action_vec(name, p, cfg, previous_vec=prev_avec)
        violated, reason = hard_violation(masses, avec, plant_info, cfg)
        if violated:
            return {**meta, "sequence": sequence, "hard_reject_reason": reason,
                    "costs": costs, "selection_reason": "hard_reject"}

        masses, exec_ratio = update_tanks(masses, avec, cfg)
        comp = cfg.leak * comp + exec_ratio * avec
        residual_vec = p - comp
        residual_norm = norm_term(residual_vec, cfg)
        # envelope evaluated either on discounted or raw (undiscounted) residual.
        if cfg.envelope_use_discount:
            env_norm = envelope_norm(residual_vec, cfg)
        else:
            p_raw = blocks[k].get("pressure_vec_raw", p)
            env_norm = envelope_norm(p_raw - comp, cfg)
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
    costs["terminal_residual_cost"] = norm_term(terminal_vec, cfg) ** 2
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
    return (
        cfg.w_pump_work * c["pump_work_cost"]
        + cfg.w_pump_duration * c["pump_duration_cost"]
        + cfg.w_startstop * c["startstop_cost"]
        + cfg.w_direction_switch * c["direction_switch_cost"]
        + cfg.w_reverse_penalty * c["reverse_penalty"]
        + cfg.w_envelope_soft * c["envelope_violation_cost"]
        + cfg.w_terminal_envelope_soft * c["terminal_envelope_violation"]
        + cfg.w_saturation_hard * c["saturation_penalty"]
        + cfg.envelope_barrier_const * c.get("envelope_barrier_triggered", 0.0)
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
    windows: pd.DataFrame,
    replay_dataset: Any,
    cfg: PlannerConfig,
    discount_profiles: dict[str, list[float]],
    plant_info: dict[str, Any] | None = None,
    timestamp_fmt: str = "%Y-%m-%d %H:%M:%S",
) -> pd.DataFrame:
    """Evaluate the A1.4 planner for every (window, discount_profile) pair.

    Returns a flat DataFrame with one row per (window × profile).
    """
    from datetime import datetime

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
