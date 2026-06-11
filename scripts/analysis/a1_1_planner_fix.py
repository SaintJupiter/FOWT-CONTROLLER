#!/usr/bin/env python3
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
    active_medium_rel_improve_min: float = 0.15


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


def active_medium_gate_allowed(
    k: int,
    pressure_blocks: list[dict[str, Any]],
    comp_vec: np.ndarray,
    cfg: PlannerConfig,
) -> tuple[bool, str]:
    current_pressure = pressure_blocks[k]["pressure_vec"]
    hold_residual = current_pressure - (cfg.leak * comp_vec)
    med_residual = current_pressure - (cfg.leak * comp_vec + action_vec("active_medium", current_pressure, cfg))
    hold_term = norm_term(hold_residual, cfg)
    med_term = norm_term(med_residual, cfg)
    abs_improve = hold_term - med_term
    rel_improve = abs_improve / max(hold_term, 1e-9)
    if hold_term < cfg.active_medium_hold_term_min:
        return False, "hold_terminal_too_low"
    if abs_improve < cfg.active_medium_abs_improve_min:
        return False, "medium_improve_too_small"
    if rel_improve < cfg.active_medium_rel_improve_min:
        return False, "medium_rel_improve_too_small"
    return True, "hold_terminal_high_and_medium_improves"


def terminal_same_band(a: float, b: float, cfg: PlannerConfig) -> bool:
    tol = max(cfg.terminal_band_abs, cfg.terminal_band_rel * max(abs(a), abs(b), 1e-9))
    return abs(a - b) <= tol


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
    for depth, (name, va, vb) in enumerate(dims[1:], start=2):
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
    hard_reject_reason = ""
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
    prev_action_summary = pd.read_csv(base_out / "a1_pilot_action_summary.csv")

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
    hold_vs_active_rows = []

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
            seq_rows = [evaluate_sequence(seq, pblocks, plant_info_prev, cfg) for seq in seqs]
            feasible = [r for r in seq_rows if not r["hard_reject_reason"]]
            feasible_sorted = sorted(feasible, key=cmp_to_key(lambda a, b: compare_rows(a, b, cfg)[0]))
            best = feasible_sorted[0] if feasible_sorted else None
            second = feasible_sorted[1] if len(feasible_sorted) > 1 else best

            best_hold = None
            hold_feasible = [r for r in feasible_sorted if r["sequence"][0] == "hold"]
            if hold_feasible:
                best_hold = hold_feasible[0]
            best_active = None
            active_feasible = [r for r in feasible_sorted if r["sequence"][0] in ("active_small", "active_medium")]
            if active_feasible:
                best_active = active_feasible[0]

            if win["a1_group"] == "high_pressure_high_event":
                hold_vs_active_rows.append(
                    {
                        "discount_profile": profile_name,
                        "prediction_timestamp": win["prediction_timestamp"],
                        "best_hold_sequence": ">".join(best_hold["sequence"]) if best_hold else "",
                        "best_hold_terminal_residual": float(best_hold["costs"]["terminal_residual_cost"]) if best_hold else np.nan,
                        "best_active_sequence": ">".join(best_active["sequence"]) if best_active else "",
                        "best_active_terminal_residual": float(best_active["costs"]["terminal_residual_cost"]) if best_active else np.nan,
                        "active_minus_hold_terminal_residual": (
                            float(best_active["costs"]["terminal_residual_cost"] - best_hold["costs"]["terminal_residual_cost"])
                            if best_active and best_hold
                            else np.nan
                        ),
                    }
                )

            if best is None:
                continue
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
    results_df.to_csv(base_out / "a1_1_window_results.csv", index=False)

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
    action_summary_df.to_csv(base_out / "a1_1_action_summary.csv", index=False)

    hold_vs_active_df = pd.DataFrame(hold_vs_active_rows)
    hold_vs_active_df.to_csv(diag_dir / "a1_1_hold_vs_active_comparison.csv", index=False)

    # Compare with A1 pilot
    prev_default_low_medium = 0
    prev_default_high_hold = 0
    if not prev_action_summary.empty:
        mask = (prev_action_summary["discount_profile"] == "default_discount") & (prev_action_summary["a1_group"] == "low_pressure_normal") & (prev_action_summary["first_action"] == "active_medium")
        prev_default_low_medium = int(prev_action_summary.loc[mask, "count"].sum())
        mask2 = (prev_action_summary["discount_profile"] == "default_discount") & (prev_action_summary["a1_group"] == "high_pressure_high_event") & (prev_action_summary["first_action"] == "hold")
        prev_default_high_hold = int(prev_action_summary.loc[mask2, "count"].sum())

    cur_default = results_df[results_df["discount_profile"] == "default_discount"]
    low_medium_now = int(np.sum((cur_default["a1_group"] == "low_pressure_normal") & (cur_default["first_action"] == "active_medium")))
    high_hold_now = int(np.sum((cur_default["a1_group"] == "high_pressure_high_event") & (cur_default["first_action"] == "hold")))
    terminal_dom_now = int(np.sum(cur_default["winning_dimension"] == "terminal_residual"))
    total_default = len(cur_default)

    report_lines = [
        "# A1.1 Planner Fix Report",
        "",
        f"- elapsed wall time: `{time.perf_counter() - t0:.1f} s`",
        "- fixed items in this round:",
        "  - terminal_residual tolerance band",
        "  - active_medium release gate",
        "  - high_pressure hold-vs-active terminal comparison output",
        "",
        "## Final Answers",
        "",
        f"1. low_pressure_normal 中 active_medium 是否减少：`{'YES' if low_medium_now < prev_default_low_medium else 'NO'}`",
        f"   - A1-pilot default count: `{prev_default_low_medium}`",
        f"   - A1.1 default count: `{low_medium_now}`",
        "",
        f"2. high_pressure/high_event 中 hold 是否仍然过多：`{'YES' if high_hold_now >= 4 else 'NO'}`",
        f"   - A1-pilot default hold count: `{prev_default_high_hold}`",
        f"   - A1.1 default hold count: `{high_hold_now}`",
        "",
        f"3. terminal_residual 是否还垄断排序：`{'YES' if terminal_dom_now >= max(total_default - 2, 1) else 'NO'}`",
        f"   - default windows decided by terminal_residual: `{terminal_dom_now}/{total_default}`",
        "",
        f"4. 是否可以进入下一轮 A1 dry-run 或需要继续修 planner：`{'ENTER_NEXT_A1_DRYRUN' if (low_medium_now < prev_default_low_medium and high_hold_now < prev_default_high_hold and terminal_dom_now < total_default) else 'NEED_MORE_PLANNER_FIX'}`",
    ]
    (base_out / "A1_1_planner_fix_report.md").write_text("\n".join(report_lines).strip() + "\n", encoding="utf-8")

    print(f"Saved A1.1 outputs under {base_out}")


if __name__ == "__main__":
    main()
