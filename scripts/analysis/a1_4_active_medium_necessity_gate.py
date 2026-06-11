#!/usr/bin/env python3
"""A1.4 — active_medium necessity gate (minimal change vs A1.3).

Only the active_medium admission gate is updated. Compared with A1.3, the gate
now requires medium to be NECESSARY relative to active_small (not just better
than hold). Three interpretable rules replace the old "incremental improvement
vs the global median" check:

    A. hold_terminal_too_low                  (kept from A1.3)
    B. active_small_already_within_deadband   (NEW; small qualifies attitude)
    C. small_and_medium_same_band             (NEW; same risk tier → prefer small)
    D. medium_not_significantly_better_than_small  (kept threshold, retargeted)

Nothing else is touched: reverse gate, block discount, pressure-sign convention,
windows, sequence enumeration and the lexicographic comparator are all the same
as A1.3.
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from datetime import datetime
from functools import cmp_to_key
from pathlib import Path
from typing import Any
import sys

import numpy as np
import pandas as pd


TIMESTAMP_FMT = "%Y-%m-%d %H:%M:%S"
BLOCKS = (
    ("block1_0_20m", 0, 2),
    ("block2_20_40m", 2, 4),
    ("block3_40_60m", 4, 6),
)


@dataclass(frozen=True)
class PlannerConfig:
    deadband_pitch_deg: float = 1.0
    deadband_roll_deg: float = 0.8
    tank_capacity_kg: float = 1850.0 * 1025.0
    leak: float = 0.90
    pressure_sign_multiplier: float = -1.0
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


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def pressure_proxy_vec_from_uv_block(uv_block: np.ndarray, cfg: PlannerConfig, wind_reference_mps: float = 12.0) -> np.ndarray:
    speed = np.sqrt(uv_block[:, 0] ** 2 + uv_block[:, 1] ** 2)
    mean_u = float(np.mean(uv_block[:, 0]))
    mean_v = float(np.mean(uv_block[:, 1]))
    mean_speed = float(np.mean(speed))
    mean_dir = float((np.rad2deg(np.arctan2(-mean_u, -mean_v)) + 360.0) % 360.0)
    mag = float(np.clip((max(mean_speed, 0.0) / max(wind_reference_mps, 1.0)) ** 2, 0.0, 1.5))
    wd_rad = math.radians(mean_dir)
    raw = np.array(
        [
            -cfg.deadband_pitch_deg * mag * math.cos(wd_rad),
            cfg.deadband_roll_deg * mag * math.sin(wd_rad),
        ],
        dtype=float,
    )
    return cfg.pressure_sign_multiplier * raw


def action_vec(name: str, pressure_vec: np.ndarray, cfg: PlannerConfig, previous_vec: np.ndarray | None = None) -> np.ndarray:
    deadband = np.array([cfg.deadband_pitch_deg, cfg.deadband_roll_deg], dtype=float)
    if float(np.linalg.norm(pressure_vec / np.maximum(deadband, 1e-6))) <= 1e-9:
        direction = np.zeros(2, dtype=float)
    else:
        direction = pressure_vec / max(float(np.linalg.norm(pressure_vec)), 1e-9)
    if name == "hold":
        return np.zeros(2, dtype=float)
    if name == "pump_saving":
        if previous_vec is not None and float(np.linalg.norm(previous_vec)) > 1e-9:
            prev_dir = previous_vec / max(float(np.linalg.norm(previous_vec)), 1e-9)
            direction = 0.6 * direction + 0.4 * prev_dir
            if float(np.linalg.norm(direction)) > 1e-9:
                direction = direction / float(np.linalg.norm(direction))
        return direction * deadband * cfg.pump_saving_ratio
    if name == "active_small":
        return direction * deadband * cfg.active_small_ratio
    if name == "active_medium":
        return direction * deadband * cfg.active_medium_ratio
    if name == "active_reverse_small":
        return -direction * deadband * cfg.reverse_small_ratio
    raise KeyError(name)


def tank_signal_from_action(action_vec_: np.ndarray, cfg: PlannerConfig) -> np.ndarray:
    alloc = np.array([[-1.0, 0.0], [0.5, 1.0], [0.5, -1.0]], dtype=float)
    deadband = np.array([cfg.deadband_pitch_deg, cfg.deadband_roll_deg], dtype=float)
    return alloc @ (action_vec_ / np.maximum(deadband, 1e-6))


def compute_pressure_blocks(sample, discount_blocks: list[float], cfg: PlannerConfig) -> list[dict[str, Any]]:
    uv = np.asarray(sample.y_uv_raw, dtype=float)
    rows = []
    for idx, (name, start, end) in enumerate(BLOCKS):
        pvec = pressure_proxy_vec_from_uv_block(uv[start:end], cfg=cfg) * float(discount_blocks[idx])
        rows.append(
            {
                "block_name": name,
                "pressure_vec": pvec,
                "pressure_norm": float(np.linalg.norm(pvec / np.array([cfg.deadband_pitch_deg, cfg.deadband_roll_deg]))),
            }
        )
    return rows


def hard_violation(tank_masses: np.ndarray, action_vec_: np.ndarray, plant_info_prev: dict[str, Any], cfg: PlannerConfig) -> tuple[bool, str]:
    tank_signal = tank_signal_from_action(action_vec_, cfg)
    mass_ratio = tank_masses / max(cfg.tank_capacity_kg, 1.0)
    pushing_upper = np.any((mass_ratio >= cfg.upper_capacity_guard_ratio) & (tank_signal > 0.0))
    pulling_lower = np.any((mass_ratio <= cfg.lower_capacity_guard_ratio) & (tank_signal < 0.0))
    if pushing_upper or pulling_lower:
        return True, "capacity_guard"
    if cfg.fullspeed_guard and int(plant_info_prev.get("pump_fullspeed_any", 0)) and np.linalg.norm(action_vec_) > 1e-9:
        return True, "fullspeed_guard"
    return False, ""


def update_virtual_tanks(tank_masses: np.ndarray, action_vec_: np.ndarray, cfg: PlannerConfig) -> tuple[np.ndarray, float]:
    signal = tank_signal_from_action(action_vec_, cfg)
    delta_req = signal * cfg.action_mass_quantum_kg
    new_mass = np.clip(tank_masses + delta_req, 0.0, cfg.tank_capacity_kg)
    actual = new_mass - tank_masses
    req_norm = float(np.linalg.norm(delta_req))
    exec_ratio = float(np.linalg.norm(actual) / req_norm) if req_norm > 1e-9 else 0.0
    return new_mass, exec_ratio


def norm_term(vec: np.ndarray, cfg: PlannerConfig) -> float:
    deadband = np.array([cfg.deadband_pitch_deg, cfg.deadband_roll_deg], dtype=float)
    return float(np.linalg.norm(vec / np.maximum(deadband, 1e-6)))


def reverse_gate_allowed(
    k: int,
    pressure_blocks: list[dict[str, Any]],
    comp_vec: np.ndarray,
    dwell_blocks_same_direction: int,
    cfg: PlannerConfig,
) -> tuple[bool, str]:
    current_pressure = pressure_blocks[k]["pressure_vec"]
    if float(np.linalg.norm(comp_vec)) <= cfg.minimum_overcomp_ratio_for_reverse * max(pressure_blocks[k]["pressure_norm"], 1e-9):
        return False, "overcomp_too_small"
    if dwell_blocks_same_direction < cfg.dwell_blocks_required_for_reverse:
        return False, "reverse_after_full_block_only"
    if k < len(pressure_blocks) - 1:
        next_pressure = pressure_blocks[k + 1]["pressure_vec"]
        if float(np.dot(current_pressure, next_pressure)) >= 0.0 and float(np.dot(comp_vec, current_pressure)) >= 0.0:
            return False, "no_future_reversal_signal"
    residual_hold = current_pressure - (cfg.leak * comp_vec)
    residual_reverse = current_pressure - (cfg.leak * comp_vec + action_vec("active_reverse_small", current_pressure, cfg))
    if norm_term(residual_reverse, cfg) + 0.05 >= norm_term(residual_hold, cfg):
        return False, "terminal_not_improved"
    return True, "future_reversal_and_overcomp"


def attitude_same_band(a: float, b: float, cfg: PlannerConfig) -> bool:
    tol = max(cfg.attitude_band_abs, cfg.attitude_band_rel * max(abs(a), abs(b), 1e-9))
    return abs(a - b) <= tol


def terminal_same_band(a: float, b: float, cfg: PlannerConfig) -> bool:
    tol = max(cfg.terminal_band_abs, cfg.terminal_band_rel * max(abs(a), abs(b), 1e-9))
    return abs(a - b) <= tol


def active_medium_gate_allowed(
    k: int,
    pressure_blocks: list[dict[str, Any]],
    comp_vec: np.ndarray,
    cfg: PlannerConfig,
) -> tuple[bool, str]:
    """A1.4 necessity gate.

    Logic in plain words:
      - hold residual already small  → no action of any kind needed
      - active_small leaves residual within deadband (norm_term ≤ 1.0)
        → small qualifies attitude; medium has no engineering necessity
      - small and medium fall in the same risk tier (existing attitude band)
        → no tier improvement; prefer the smaller, smoother action
      - medium must beat small by at least the existing abs-improve threshold,
        otherwise it is just a slightly-smaller residual at strictly larger
        pump cost.
    """
    current_pressure = pressure_blocks[k]["pressure_vec"]
    hold_residual = current_pressure - (cfg.leak * comp_vec)
    small_residual = current_pressure - (cfg.leak * comp_vec + action_vec("active_small", current_pressure, cfg))
    med_residual = current_pressure - (cfg.leak * comp_vec + action_vec("active_medium", current_pressure, cfg))
    hold_term = norm_term(hold_residual, cfg)
    small_term = norm_term(small_residual, cfg)
    med_term = norm_term(med_residual, cfg)

    if hold_term < cfg.active_medium_hold_term_min:
        return False, "hold_terminal_too_low"
    # Rule B applies only at the first decision point: it asks whether the
    # planner should ESCALATE the first action to medium. For later blocks
    # (k>=1) earlier actions have already accumulated comp; using "small fits
    # within deadband" there would wrongly reject sequences like
    # active_small>hold>active_medium that legitimately need a late escalation
    # in genuinely high-pressure windows. norm_term is deadband-normalized,
    # so 1.0 == one deadband unit.
    if k == 0 and small_term <= 1.0:
        return False, "active_small_already_within_deadband"
    if attitude_same_band(small_term, med_term, cfg):
        return False, "small_and_medium_same_band"
    if (small_term - med_term) < cfg.active_medium_abs_improve_min:
        return False, "medium_not_significantly_better_than_small"
    return True, "active_small_insufficient_medium_required"


def compare_rows(a: dict[str, Any], b: dict[str, Any], cfg: PlannerConfig) -> tuple[int, str, int]:
    dims = [
        ("terminal_residual", float(a["costs"]["terminal_residual_cost"]), float(b["costs"]["terminal_residual_cost"])),
        ("attitude_residual", float(a["costs"]["attitude_residual_cost"]), float(b["costs"]["attitude_residual_cost"])),
        ("direction_switch_plus_reverse", float(a["costs"]["direction_switch_cost"] + a["costs"]["reverse_penalty"]), float(b["costs"]["direction_switch_cost"] + b["costs"]["reverse_penalty"])),
        ("saturation_margin", float(a["costs"]["saturation_penalty"]), float(b["costs"]["saturation_penalty"])),
        ("pump_work_proxy", float(a["costs"]["pump_work_cost"]), float(b["costs"]["pump_work_cost"])),
        ("pump_duration_proxy", float(a["costs"]["pump_duration_cost"]), float(b["costs"]["pump_duration_cost"])),
        ("startstop_proxy", float(a["costs"]["startstop_cost"]), float(b["costs"]["startstop_cost"])),
    ]
    name, va, vb = dims[0]
    if not terminal_same_band(va, vb, cfg):
        if va < vb:
            return -1, name, 1
        if va > vb:
            return 1, name, 1
    name, va, vb = dims[1]
    if not attitude_same_band(va, vb, cfg):
        if va < vb:
            return -1, name, 2
        if va > vb:
            return 1, name, 2
    for depth, (name, va, vb) in enumerate(dims[2:], start=3):
        if abs(va - vb) > 1e-9:
            return (-1 if va < vb else 1), name, depth
    return 0, "tie", len(dims)


def total_key_sum(row: dict[str, Any]) -> float:
    return float(
        row["costs"]["terminal_residual_cost"]
        + row["costs"]["attitude_residual_cost"]
        + row["costs"]["direction_switch_cost"]
        + row["costs"]["reverse_penalty"]
        + row["costs"]["saturation_penalty"]
        + row["costs"]["pump_work_cost"]
        + row["costs"]["pump_duration_cost"]
        + row["costs"]["startstop_cost"]
    )


def sequence_gap_pct(a: dict[str, Any], b: dict[str, Any]) -> float:
    va = total_key_sum(a)
    vb = total_key_sum(b)
    return float((vb - va) / max(abs(va), 1e-9) * 100.0)


def evaluate_sequence(
    sequence: tuple[str, str, str],
    pressure_blocks: list[dict[str, Any]],
    plant_info_prev: dict[str, Any],
    cfg: PlannerConfig,
    a1_group: str,
) -> dict[str, Any]:
    comp_vec = np.zeros(2, dtype=float)
    last_action_vec = np.zeros(2, dtype=float)
    tank_masses = np.asarray(plant_info_prev.get("tank_masses", np.zeros(3, dtype=float)), dtype=float).reshape(-1)[:3]
    if tank_masses.size < 3:
        tank_masses = np.pad(tank_masses, (0, 3 - tank_masses.size))
    costs = {
        "attitude_residual_cost": 0.0,
        "pump_work_cost": 0.0,
        "pump_duration_cost": 0.0,
        "startstop_cost": 0.0,
        "direction_switch_cost": 0.0,
        "reverse_penalty": 0.0,
        "saturation_penalty": 0.0,
        "terminal_residual_cost": 0.0,
    }
    reverse_allowed_any = False
    reverse_release_reason = ""
    reverse_reject_reason = ""
    reverse_after_full_block_only = False
    active_medium_gate_reason = ""
    dwell_blocks_same_direction = 0
    last_direction_sign = 0

    for k, action_name in enumerate(sequence):
        pressure_vec = pressure_blocks[k]["pressure_vec"]
        if action_name == "active_reverse_small":
            allowed, reason = reverse_gate_allowed(k, pressure_blocks, comp_vec, dwell_blocks_same_direction, cfg)
            reverse_after_full_block_only = dwell_blocks_same_direction >= cfg.dwell_blocks_required_for_reverse
            if not allowed:
                return {
                    "sequence": sequence,
                    "hard_reject_reason": "reverse_not_released",
                    "reverse_allowed": False,
                    "reverse_release_reason": "",
                    "reverse_reject_reason": reason,
                    "reverse_after_full_block_only": reverse_after_full_block_only,
                    "active_medium_gate_reason": active_medium_gate_reason,
                    "costs": costs,
                    "selection_reason": "hard_reject",
                }
            reverse_allowed_any = True
            reverse_release_reason = reason
        if action_name == "active_medium":
            allowed, reason = active_medium_gate_allowed(k, pressure_blocks, comp_vec, cfg)
            active_medium_gate_reason = reason
            if not allowed:
                return {
                    "sequence": sequence,
                    "hard_reject_reason": "active_medium_gate",
                    "reverse_allowed": reverse_allowed_any,
                    "reverse_release_reason": reverse_release_reason,
                    "reverse_reject_reason": reverse_reject_reason,
                    "reverse_after_full_block_only": reverse_after_full_block_only,
                    "active_medium_gate_reason": reason,
                    "costs": costs,
                    "selection_reason": "hard_reject",
                }

        avec = action_vec(action_name, pressure_vec, cfg, previous_vec=last_action_vec)
        violated, reason = hard_violation(tank_masses, avec, plant_info_prev, cfg)
        if violated:
            return {
                "sequence": sequence,
                "hard_reject_reason": reason,
                "reverse_allowed": reverse_allowed_any,
                "reverse_release_reason": reverse_release_reason,
                "reverse_reject_reason": reverse_reject_reason,
                "reverse_after_full_block_only": reverse_after_full_block_only,
                "active_medium_gate_reason": active_medium_gate_reason,
                "costs": costs,
                "selection_reason": "hard_reject",
            }

        tank_next, exec_ratio = update_virtual_tanks(tank_masses, avec, cfg)
        comp_next = cfg.leak * comp_vec + exec_ratio * avec
        residual = pressure_vec - comp_next
        residual_norm = norm_term(residual, cfg)
        costs["attitude_residual_cost"] += residual_norm**2
        costs["pump_work_cost"] += float(np.linalg.norm(avec / np.array([cfg.deadband_pitch_deg, cfg.deadband_roll_deg])))
        if np.linalg.norm(avec) > 1e-9:
            costs["pump_duration_cost"] += 1.0
        if np.linalg.norm(last_action_vec) <= 1e-9 and np.linalg.norm(avec) > 1e-9:
            costs["startstop_cost"] += 1.0
        if np.linalg.norm(last_action_vec) > 1e-9 and np.linalg.norm(avec) > 1e-9 and float(np.dot(last_action_vec, avec)) < 0.0:
            costs["direction_switch_cost"] += 1.0
        if action_name == "active_reverse_small":
            costs["reverse_penalty"] += 1.0

        mass_ratio = tank_next / max(cfg.tank_capacity_kg, 1.0)
        sat_proxy = float(
            max(0.0, np.max(mass_ratio - cfg.upper_capacity_guard_ratio))
            + max(0.0, np.max(cfg.lower_capacity_guard_ratio - mass_ratio))
        )
        costs["saturation_penalty"] += sat_proxy

        comp_vec = comp_next
        tank_masses = tank_next
        if np.linalg.norm(avec) > 1e-9:
            sign = 1 if float(np.dot(avec, pressure_vec)) >= 0.0 else -1
            if sign == last_direction_sign:
                dwell_blocks_same_direction += 1
            else:
                dwell_blocks_same_direction = 1
                last_direction_sign = sign
        last_action_vec = avec

    terminal_residual = pressure_blocks[-1]["pressure_vec"] - comp_vec
    costs["terminal_residual_cost"] = norm_term(terminal_residual, cfg) ** 2
    return {
        "sequence": sequence,
        "hard_reject_reason": "",
        "reverse_allowed": reverse_allowed_any,
        "reverse_release_reason": reverse_release_reason,
        "reverse_reject_reason": reverse_reject_reason,
        "reverse_after_full_block_only": reverse_after_full_block_only,
        "active_medium_gate_reason": active_medium_gate_reason,
        "costs": costs,
        "selection_reason": "feasible",
    }


def build_sequences() -> list[tuple[str, str, str]]:
    actions = ["hold", "pump_saving", "active_small", "active_medium", "active_reverse_small"]
    return [(a, b, c) for a in actions for b in actions for c in actions]


def main() -> None:
    t0 = time.perf_counter()
    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root / "src"))
    from wind_prediction.replay_dataset import Fino1ReplayDataset

    base_out = repo_root / "outputs" / "wind_prediction" / "planner_a1_dryrun"
    diag_dir = base_out / "diagnostics"
    ensure_dir(base_out)
    ensure_dir(diag_dir)

    windows = pd.read_csv(base_out / "window_selection_preview.csv")
    sign_cfg = load_json(base_out / "diagnostics" / "a01_pressure_vec_sign_convention.json")
    discount_cfg = load_json(base_out / "diagnostics" / "a1_block_discount_config.json")
    a1_3_action_summary = pd.read_csv(base_out / "a1_3_action_summary.csv")

    cfg = PlannerConfig(pressure_sign_multiplier=float(sign_cfg["planner_sign_multiplier"]))
    replay = Fino1ReplayDataset(
        dataset_dir=repo_root / "data" / "processed" / "wind_ml_10min" / "ballast_decision_fino1_meteo_aux_v1",
        split="test",
    )
    seqs = build_sequences()
    discount_profiles = {
        "default_discount": discount_cfg["default_discount_blocks"],
        "no_discount": discount_cfg["no_discount_blocks"],
    }

    results = []
    gate_rows = []
    medium_diag_rows = []  # raw small/med/hold terms when first action is active_medium

    for _, win in windows.iterrows():
        sample = replay.sample_for_history_end(datetime.strptime(str(win["prediction_timestamp"]), TIMESTAMP_FMT))
        if sample is None:
            continue
        plant_info_prev = {
            "tank_masses": np.array([1108000.0, 1362000.0, 1362000.0], dtype=float),
            "pump_fullspeed_any": 0,
            "pump_total_backlog_kg": 0.0,
            "pump_rate_cmd_m3_min": np.array([0.0, 0.0, 0.0], dtype=float),
        }
        for profile_name, discounts in discount_profiles.items():
            pblocks = compute_pressure_blocks(sample, discounts, cfg)
            seq_rows = [evaluate_sequence(seq, pblocks, plant_info_prev, cfg, a1_group=str(win["a1_group"])) for seq in seqs]
            feasible = [r for r in seq_rows if not r["hard_reject_reason"]]
            feasible_sorted = sorted(feasible, key=cmp_to_key(lambda a, b: compare_rows(a, b, cfg)[0]))
            best = feasible_sorted[0] if feasible_sorted else None
            second = feasible_sorted[1] if len(feasible_sorted) > 1 else best

            # Diagnostic: log block-0 small/med/hold terms for THIS window so we can
            # see why the gate fired (or didn't).
            cur_pressure = pblocks[0]["pressure_vec"]
            hold_term = norm_term(cur_pressure, cfg)
            small_term = norm_term(cur_pressure - action_vec("active_small", cur_pressure, cfg), cfg)
            med_term = norm_term(cur_pressure - action_vec("active_medium", cur_pressure, cfg), cfg)
            medium_diag_rows.append(
                {
                    "discount_profile": profile_name,
                    "a1_group": win["a1_group"],
                    "prediction_timestamp": win["prediction_timestamp"],
                    "block0_pressure_norm": float(pblocks[0]["pressure_norm"]),
                    "block0_hold_term": hold_term,
                    "block0_small_term": small_term,
                    "block0_med_term": med_term,
                    "small_within_deadband": int(small_term <= 1.0),
                    "small_med_same_band": int(attitude_same_band(small_term, med_term, cfg)),
                    "small_minus_med": small_term - med_term,
                }
            )

            if best is None:
                continue
            gate_rows.append(
                {
                    "discount_profile": profile_name,
                    "a1_group": win["a1_group"],
                    "prediction_timestamp": win["prediction_timestamp"],
                    "best_first_action": best["sequence"][0],
                    "active_medium_gate_reason": str(best["active_medium_gate_reason"]),
                }
            )
            winning_dim, depth = compare_rows(best, second, cfg)[1:]
            results.append(
                {
                    "discount_profile": profile_name,
                    "a1_group": win["a1_group"],
                    "prediction_timestamp": win["prediction_timestamp"],
                    "best_sequence": ">".join(best["sequence"]),
                    "second_best_sequence": ">".join(second["sequence"]) if second else "",
                    "first_action": best["sequence"][0],
                    "best_vs_second_gap_pct": sequence_gap_pct(best, second) if second else np.nan,
                    "winning_dimension": winning_dim,
                    "lexicographic_decision_depth": depth,
                    "terminal_residual_cost": float(best["costs"]["terminal_residual_cost"]),
                    "attitude_residual_cost": float(best["costs"]["attitude_residual_cost"]),
                    "pump_work_cost": float(best["costs"]["pump_work_cost"]),
                    "pump_duration_cost": float(best["costs"]["pump_duration_cost"]),
                    "startstop_cost": float(best["costs"]["startstop_cost"]),
                    "direction_switch_cost": float(best["costs"]["direction_switch_cost"]),
                    "reverse_penalty": float(best["costs"]["reverse_penalty"]),
                    "saturation_penalty": float(best["costs"]["saturation_penalty"]),
                    "reverse_allowed": int(bool(best["reverse_allowed"])),
                    "reverse_release_reason": str(best["reverse_release_reason"]),
                    "reverse_reject_reason": str(best["reverse_reject_reason"]),
                    "reverse_after_full_block_only": int(bool(best["reverse_after_full_block_only"])),
                    "active_medium_gate_reason": str(best["active_medium_gate_reason"]),
                }
            )

    results_df = pd.DataFrame(results)
    results_df.to_csv(base_out / "a1_4_window_results.csv", index=False)

    action_summary_rows = []
    for (profile, group), sub in results_df.groupby(["discount_profile", "a1_group"]):
        total = len(sub)
        for action, cnt in sub["first_action"].value_counts().items():
            action_summary_rows.append(
                {
                    "discount_profile": profile,
                    "a1_group": group,
                    "first_action": action,
                    "count": int(cnt),
                    "ratio": float(cnt / max(total, 1)),
                }
            )
    action_summary_df = pd.DataFrame(action_summary_rows)
    action_summary_df.to_csv(base_out / "a1_4_action_summary.csv", index=False)

    medium_diag_df = pd.DataFrame(medium_diag_rows)
    medium_diag_df.to_csv(diag_dir / "a1_4_medium_term_diagnostics.csv", index=False)

    lex_summary = (
        results_df.groupby(["discount_profile", "winning_dimension"])
        .size()
        .reset_index(name="count")
        .sort_values(["discount_profile", "count"], ascending=[True, False])
    )
    lex_summary.to_csv(diag_dir / "a1_4_lexicographic_dimension_summary.csv", index=False)

    pd.DataFrame(gate_rows).to_csv(diag_dir / "a1_4_active_medium_gate_check.csv", index=False)

    # ---- compare with A1.3 default-discount summary -----------------------
    def count_from(df: pd.DataFrame, profile: str, group: str, action: str) -> int:
        if df.empty:
            return 0
        m = (df["discount_profile"] == profile) & (df["a1_group"] == group) & (df["first_action"] == action)
        return int(df.loc[m, "count"].sum())

    prev_low_medium_default = count_from(a1_3_action_summary, "default_discount", "low_pressure_normal", "active_medium")
    prev_high_hold_default = count_from(a1_3_action_summary, "default_discount", "high_pressure_high_event", "hold")

    cur_default = results_df[results_df["discount_profile"] == "default_discount"]
    low_medium_now = int(np.sum((cur_default["a1_group"] == "low_pressure_normal") & (cur_default["first_action"] == "active_medium")))
    high_hold_now = int(np.sum((cur_default["a1_group"] == "high_pressure_high_event") & (cur_default["first_action"] == "hold")))
    total_default = len(cur_default)
    attitude_dom = int(np.sum(cur_default["winning_dimension"] == "attitude_residual"))
    terminal_dom = int(np.sum(cur_default["winning_dimension"] == "terminal_residual"))

    block_reasons = (
        pd.Series([r["active_medium_gate_reason"] for r in gate_rows if r["active_medium_gate_reason"] not in ("", "active_small_insufficient_medium_required")])
        .value_counts()
        .to_dict()
    )

    report_lines = [
        "# A1.4 Active-Medium Necessity Gate Report",
        "",
        f"- elapsed wall time: `{time.perf_counter() - t0:.1f} s`",
        "- only change vs A1.3: active_medium gate now compares against active_small "
        "(deadband-based qualification + same-band tier check), not just against hold.",
        "- not changed: reverse gate, block discount, sign convention, windows, "
        "closed-only controller, high_pressure logic.",
        "",
        "## Final Answers",
        "",
        f"1. low_pressure_normal 中 active_medium 是否降到 0–1：`{'YES' if low_medium_now <= 1 else 'NO'}`",
        f"   - A1.3 default count: `{prev_low_medium_default}`",
        f"   - A1.4 default count: `{low_medium_now}`",
        "",
        f"2. active_medium 被挡住的主要原因 (reasons among rejected):",
        f"   - `{block_reasons}`",
        "",
        f"3. high_pressure 行为是否没有明显恶化：`{'YES' if high_hold_now <= prev_high_hold_default else 'NO'}`",
        f"   - A1.3 default hold count: `{prev_high_hold_default}`",
        f"   - A1.4 default hold count: `{high_hold_now}`",
        "",
        f"4. winning_dimension 是否没有重新单维垄断：`{'YES' if max(attitude_dom, terminal_dom) < max(total_default - 2, 1) else 'NO'}`",
        f"   - attitude_residual dominant count: `{attitude_dom} / {total_default}`",
        f"   - terminal_residual dominant count: `{terminal_dom} / {total_default}`",
        "",
        f"5. recommended next step: `{'PROCEED_NO_FURTHER_GATE_FIX' if low_medium_now <= 1 and high_hold_now <= prev_high_hold_default else 'INSPECT_DIAGNOSTICS'}`",
    ]
    (base_out / "A1_4_active_medium_necessity_gate_report.md").write_text(
        "\n".join(report_lines).strip() + "\n", encoding="utf-8"
    )

    print(f"Saved A1.4 outputs under {base_out}")


if __name__ == "__main__":
    main()
