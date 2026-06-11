#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from datetime import datetime
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
    norm = float(np.linalg.norm(pressure_vec / np.maximum(deadband, 1e-6)))
    if norm <= 1e-9:
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
        return direction * np.array([cfg.deadband_pitch_deg, cfg.deadband_roll_deg]) * cfg.pump_saving_ratio
    if name == "active_small":
        return direction * np.array([cfg.deadband_pitch_deg, cfg.deadband_roll_deg]) * cfg.active_small_ratio
    if name == "active_medium":
        return direction * np.array([cfg.deadband_pitch_deg, cfg.deadband_roll_deg]) * cfg.active_medium_ratio
    if name == "active_reverse_small":
        return -direction * np.array([cfg.deadband_pitch_deg, cfg.deadband_roll_deg]) * cfg.reverse_small_ratio
    raise KeyError(name)


def tank_signal_from_action(action_vec_: np.ndarray, cfg: PlannerConfig) -> np.ndarray:
    alloc = np.array([[-1.0, 0.0], [0.5, 1.0], [0.5, -1.0]], dtype=float)
    deadband = np.array([cfg.deadband_pitch_deg, cfg.deadband_roll_deg], dtype=float)
    return alloc @ (action_vec_ / np.maximum(deadband, 1e-6))


def compute_pressure_blocks(sample, discount_blocks: list[float], cfg: PlannerConfig) -> list[dict[str, Any]]:
    uv = np.asarray(sample.y_uv_raw, dtype=float)
    rows = []
    for idx, (name, start, end) in enumerate(BLOCKS):
        pvec = pressure_proxy_vec_from_uv_block(uv[start:end], cfg=cfg)
        pvec = pvec * float(discount_blocks[idx])
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
    exec_ratio = 0.0
    req_norm = float(np.linalg.norm(delta_req))
    if req_norm > 1e-9:
        exec_ratio = float(np.linalg.norm(actual) / req_norm)
    return new_mass, exec_ratio


def reverse_gate_allowed(
    k: int,
    pressure_blocks: list[dict[str, Any]],
    comp_vec: np.ndarray,
    dwell_blocks_same_direction: int,
    cfg: PlannerConfig,
) -> tuple[bool, str]:
    if k >= len(pressure_blocks):
        return False, "no_block"
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
    deadband = np.array([cfg.deadband_pitch_deg, cfg.deadband_roll_deg], dtype=float)
    hold_term = float(np.linalg.norm(residual_hold / deadband))
    reverse_term = float(np.linalg.norm(residual_reverse / deadband))
    if reverse_term + 0.05 >= hold_term:
        return False, "terminal_not_improved"
    return True, "future_reversal_and_overcomp"


def lexicographic_key(costs: dict[str, float]) -> tuple[float, ...]:
    return (
        float(costs["terminal_residual_cost"]),
        float(costs["attitude_residual_cost"]),
        float(costs["direction_switch_cost"] + costs["reverse_penalty"]),
        float(costs["saturation_penalty"]),
        float(costs["pump_work_cost"]),
        float(costs["pump_duration_cost"]),
        float(costs["startstop_cost"]),
    )


def winning_dimension_name(costs_a: dict[str, float], costs_b: dict[str, float]) -> tuple[str, int]:
    dims = [
        ("terminal_residual", float(costs_a["terminal_residual_cost"]), float(costs_b["terminal_residual_cost"])),
        ("attitude_residual", float(costs_a["attitude_residual_cost"]), float(costs_b["attitude_residual_cost"])),
        ("direction_switch_plus_reverse", float(costs_a["direction_switch_cost"] + costs_a["reverse_penalty"]), float(costs_b["direction_switch_cost"] + costs_b["reverse_penalty"])),
        ("saturation_margin", float(costs_a["saturation_penalty"]), float(costs_b["saturation_penalty"])),
        ("pump_work_proxy", float(costs_a["pump_work_cost"]), float(costs_b["pump_work_cost"])),
        ("pump_duration_proxy", float(costs_a["pump_duration_cost"]), float(costs_b["pump_duration_cost"])),
        ("startstop_proxy", float(costs_a["startstop_cost"]), float(costs_b["startstop_cost"])),
    ]
    for idx, (name, a, b) in enumerate(dims, start=1):
        if abs(a - b) > 1e-9:
            return name, idx
    return "tie", len(dims)


def sequence_total_gap(best_key: tuple[float, ...], second_key: tuple[float, ...]) -> float:
    a = sum(best_key)
    b = sum(second_key)
    denom = max(abs(a), 1e-9)
    return float((b - a) / denom * 100.0)


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
    dwell_blocks_same_direction = 0
    last_direction_sign = 0

    deadband = np.array([cfg.deadband_pitch_deg, cfg.deadband_roll_deg], dtype=float)

    for k, action_name in enumerate(sequence):
        pressure_vec = pressure_blocks[k]["pressure_vec"]
        if action_name == "active_reverse_small":
            allowed, reason = reverse_gate_allowed(k, pressure_blocks, comp_vec, dwell_blocks_same_direction, cfg)
            reverse_after_full_block_only = dwell_blocks_same_direction >= cfg.dwell_blocks_required_for_reverse
            if not allowed:
                hard_reject_reason = "reverse_not_released"
                reverse_reject_reason = reason
                return {
                    "sequence": sequence,
                    "hard_reject_reason": hard_reject_reason,
                    "reverse_allowed": False,
                    "reverse_release_reason": "",
                    "reverse_reject_reason": reverse_reject_reason,
                    "reverse_after_full_block_only": reverse_after_full_block_only,
                    "costs": costs,
                    "lexicographic_key": lexicographic_key(costs),
                    "selection_reason": "hard_reject",
                }
            reverse_allowed_any = True
            reverse_release_reason = reason

        avec = action_vec(action_name, pressure_vec, cfg, previous_vec=last_action_vec)
        violated, vreason = hard_violation(tank_masses, avec, plant_info_prev, cfg)
        if violated:
            hard_reject_reason = vreason
            return {
                "sequence": sequence,
                "hard_reject_reason": hard_reject_reason,
                "reverse_allowed": reverse_allowed_any,
                "reverse_release_reason": reverse_release_reason,
                "reverse_reject_reason": reverse_reject_reason,
                "reverse_after_full_block_only": reverse_after_full_block_only,
                "costs": costs,
                "lexicographic_key": lexicographic_key(costs),
                "selection_reason": "hard_reject",
            }

        tank_next, exec_ratio = update_virtual_tanks(tank_masses, avec, cfg)
        comp_next = cfg.leak * comp_vec + exec_ratio * avec
        residual = pressure_vec - comp_next
        residual_norm = float(np.linalg.norm(residual / deadband))
        costs["attitude_residual_cost"] += residual_norm**2
        costs["pump_work_cost"] += float(np.linalg.norm(avec / deadband))
        if np.linalg.norm(avec) > 1e-9:
            costs["pump_duration_cost"] += 1.0

        if np.linalg.norm(last_action_vec) <= 1e-9 and np.linalg.norm(avec) > 1e-9:
            costs["startstop_cost"] += 1.0

        if np.linalg.norm(last_action_vec) > 1e-9 and np.linalg.norm(avec) > 1e-9:
            dotv = float(np.dot(last_action_vec, avec))
            if dotv < 0.0:
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
    costs["terminal_residual_cost"] = float(np.linalg.norm(terminal_residual / deadband) ** 2)
    return {
        "sequence": sequence,
        "hard_reject_reason": hard_reject_reason,
        "reverse_allowed": reverse_allowed_any,
        "reverse_release_reason": reverse_release_reason,
        "reverse_reject_reason": reverse_reject_reason,
        "reverse_after_full_block_only": reverse_after_full_block_only,
        "costs": costs,
        "lexicographic_key": lexicographic_key(costs),
        "selection_reason": "feasible",
    }


def build_sequences() -> list[tuple[str, str, str]]:
    actions = ["hold", "pump_saving", "active_small", "active_medium", "active_reverse_small"]
    out = []
    for a in actions:
        for b in actions:
            for c in actions:
                out.append((a, b, c))
    return out


def count_first_actions(results_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (discount_profile, a1_group), sub in results_df.groupby(["discount_profile", "a1_group"]):
        total = len(sub)
        for action, cnt in sub["first_action"].value_counts().items():
            rows.append(
                {
                    "discount_profile": discount_profile,
                    "a1_group": a1_group,
                    "first_action": action,
                    "count": int(cnt),
                    "ratio": float(cnt / max(total, 1)),
                }
            )
    return pd.DataFrame(rows)


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

    cfg = PlannerConfig(pressure_sign_multiplier=float(sign_cfg["planner_sign_multiplier"]))
    replay = Fino1ReplayDataset(
        dataset_dir=repo_root / "data" / "processed" / "wind_ml_10min" / "ballast_decision_fino1_meteo_aux_v1",
        split="test",
    )
    event_cols = list(replay.event_columns)
    seqs = build_sequences()

    results = []
    reverse_rows = []
    discount_rows = []

    discount_profiles = {
        "default_discount": discount_cfg["default_discount_blocks"],
        "no_discount": discount_cfg["no_discount_blocks"],
    }

    for _, win in windows.iterrows():
        ts = datetime.strptime(str(win["prediction_timestamp"]), TIMESTAMP_FMT)
        sample = replay.sample_for_history_end(ts)
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
            feasible.sort(key=lambda r: r["lexicographic_key"])
            if feasible:
                best = feasible[0]
                second = feasible[1] if len(feasible) > 1 else feasible[0]
                best_seq = ">".join(best["sequence"])
                second_seq = ">".join(second["sequence"])
                winning_dim, depth = winning_dimension_name(best["costs"], second["costs"])
                gap = sequence_total_gap(best["lexicographic_key"], second["lexicographic_key"])
                first_action = best["sequence"][0]
                hold_rows = [r for r in feasible if r["sequence"] == ("hold", "hold", "hold")]
                hold_cost = float(sum(hold_rows[0]["lexicographic_key"])) if hold_rows else np.nan
                active_rows = [r for r in feasible if r["sequence"][0] == "active_medium"]
                active_cost = float(sum(active_rows[0]["lexicographic_key"])) if active_rows else np.nan
                selection_reason = f"winning_dimension={winning_dim};depth={depth}"
                result_row = {
                    "discount_profile": profile_name,
                    "a1_group": win["a1_group"],
                    "prediction_timestamp": win["prediction_timestamp"],
                    "best_sequence": best_seq,
                    "second_best_sequence": second_seq,
                    "first_action": first_action,
                    "total_cost": float(sum(best["lexicographic_key"])),
                    "best_vs_second_gap_pct": gap,
                    "lexicographic_decision_depth": depth,
                    "winning_dimension": winning_dim,
                    "hard_reject_reason": "",
                    "reverse_allowed": int(bool(best["reverse_allowed"])),
                    "reverse_release_reason": str(best["reverse_release_reason"]),
                    "reverse_reject_reason": str(best["reverse_reject_reason"]),
                    "reverse_after_full_block_only": int(bool(best["reverse_after_full_block_only"])),
                    "terminal_residual_cost": float(best["costs"]["terminal_residual_cost"]),
                    "hold_sequence_cost": hold_cost,
                    "active_sequence_cost": active_cost,
                    "selection_reason": selection_reason,
                    "attitude_residual_cost": float(best["costs"]["attitude_residual_cost"]),
                    "pump_work_cost": float(best["costs"]["pump_work_cost"]),
                    "pump_duration_cost": float(best["costs"]["pump_duration_cost"]),
                    "startstop_cost": float(best["costs"]["startstop_cost"]),
                    "direction_switch_cost": float(best["costs"]["direction_switch_cost"]),
                    "reverse_penalty": float(best["costs"]["reverse_penalty"]),
                    "saturation_penalty": float(best["costs"]["saturation_penalty"]),
                }
                results.append(result_row)
            else:
                rejected = seq_rows[0]
                results.append(
                    {
                        "discount_profile": profile_name,
                        "a1_group": win["a1_group"],
                        "prediction_timestamp": win["prediction_timestamp"],
                        "best_sequence": "",
                        "second_best_sequence": "",
                        "first_action": "none",
                        "total_cost": np.nan,
                        "best_vs_second_gap_pct": np.nan,
                        "lexicographic_decision_depth": np.nan,
                        "winning_dimension": "none",
                        "hard_reject_reason": rejected["hard_reject_reason"],
                        "reverse_allowed": 0,
                        "reverse_release_reason": "",
                        "reverse_reject_reason": rejected["reverse_reject_reason"],
                        "reverse_after_full_block_only": int(bool(rejected["reverse_after_full_block_only"])),
                        "terminal_residual_cost": np.nan,
                        "hold_sequence_cost": np.nan,
                        "active_sequence_cost": np.nan,
                        "selection_reason": "all_rejected",
                        "attitude_residual_cost": np.nan,
                        "pump_work_cost": np.nan,
                        "pump_duration_cost": np.nan,
                        "startstop_cost": np.nan,
                        "direction_switch_cost": np.nan,
                        "reverse_penalty": np.nan,
                        "saturation_penalty": np.nan,
                    }
                )

            # Per-window diagnostics
            default_best = feasible[0]["sequence"][0] if profile_name == "default_discount" and feasible else None
            discount_rows.append(
                {
                    "discount_profile": profile_name,
                    "a1_group": win["a1_group"],
                    "prediction_timestamp": win["prediction_timestamp"],
                    "best_first_action": feasible[0]["sequence"][0] if feasible else "none",
                    "best_sequence": ">".join(feasible[0]["sequence"]) if feasible else "",
                    "terminal_residual_cost": float(feasible[0]["costs"]["terminal_residual_cost"]) if feasible else np.nan,
                }
            )
            for row in seq_rows:
                if "active_reverse_small" in row["sequence"]:
                    reverse_rows.append(
                        {
                            "discount_profile": profile_name,
                            "a1_group": win["a1_group"],
                            "prediction_timestamp": win["prediction_timestamp"],
                            "sequence": ">".join(row["sequence"]),
                            "hard_reject_reason": row["hard_reject_reason"],
                            "reverse_allowed": int(bool(row["reverse_allowed"])),
                            "reverse_release_reason": str(row["reverse_release_reason"]),
                            "reverse_reject_reason": str(row["reverse_reject_reason"]),
                            "reverse_after_full_block_only": int(bool(row["reverse_after_full_block_only"])),
                        }
                    )

    results_df = pd.DataFrame(results)
    results_df.to_csv(base_out / "a1_pilot_window_results.csv", index=False)

    action_summary_df = count_first_actions(results_df)
    action_summary_df.to_csv(base_out / "a1_pilot_action_summary.csv", index=False)

    discount_df = pd.DataFrame(discount_rows)
    default_df = discount_df[discount_df["discount_profile"] == "default_discount"][["prediction_timestamp", "best_first_action", "best_sequence"]].rename(
        columns={"best_first_action": "default_first_action", "best_sequence": "default_best_sequence"}
    )
    nodisc_df = discount_df[discount_df["discount_profile"] == "no_discount"][["prediction_timestamp", "best_first_action", "best_sequence"]].rename(
        columns={"best_first_action": "no_discount_first_action", "best_sequence": "no_discount_best_sequence"}
    )
    discount_sens = default_df.merge(nodisc_df, on="prediction_timestamp", how="outer")
    discount_sens["first_action_flipped"] = (discount_sens["default_first_action"] != discount_sens["no_discount_first_action"]).astype(int)
    discount_sens.to_csv(diag_dir / "a1_pilot_discount_sensitivity.csv", index=False)

    reverse_df = pd.DataFrame(reverse_rows)
    reverse_summary = (
        reverse_df.groupby(["discount_profile", "reverse_allowed", "reverse_release_reason", "reverse_reject_reason", "reverse_after_full_block_only"])
        .size()
        .reset_index(name="count")
    )
    reverse_summary.to_csv(diag_dir / "a1_pilot_reverse_gate_summary.csv", index=False)

    # Headline report
    total_windows = int(len(results_df[results_df["discount_profile"] == "default_discount"]))
    non_hold_high = int(np.sum((results_df["discount_profile"] == "default_discount") & (results_df["a1_group"] == "high_pressure_high_event") & (~results_df["first_action"].isin(["hold"])) ))
    hold_like_low = int(np.sum((results_df["discount_profile"] == "default_discount") & (results_df["a1_group"] == "low_pressure_normal") & (results_df["first_action"].isin(["hold", "pump_saving"])) ))
    reverse_allowed_count = int(np.sum(results_df["reverse_allowed"]))
    flip_count = int(np.sum(discount_sens["first_action_flipped"]))
    small_gap_count = int(np.sum(pd.to_numeric(results_df["best_vs_second_gap_pct"], errors="coerce").fillna(np.inf) < 5.0))

    report_lines = [
        "# A1 Pilot Dry-Run Report",
        "",
        f"- elapsed wall time: `{time.perf_counter() - t0:.1f} s`",
        f"- evaluated windows per discount profile: `{total_windows}`",
        "- planner convention: `pressure_vec_for_planner = - raw_pressure_vec`",
        "- residual definition: `future_pressure_vec - virtual_ballast_comp_vec`",
        "- default discount: `[1.00, 0.85, 0.70]`",
        "- no_discount control: `[1.00, 1.00, 1.00]`",
        "",
        "## Behavior Summary",
        "",
        f"- high_pressure/high_event non-hold count under default discount: `{non_hold_high}`",
        f"- low_pressure_normal hold-or-pump_saving count under default discount: `{hold_like_low}`",
        f"- reverse-allowed selected window count: `{reverse_allowed_count}`",
        f"- default vs no_discount first-action flips: `{flip_count}`",
        f"- windows with best-vs-second gap < 5%: `{small_gap_count}`",
        "",
        "## Recommendation",
        "",
        "- continue A1 if high-pressure windows are not all hold, reverse remains tightly gated, and discount sensitivity is not dominated by action flips.",
        "- pause and repair planner logic first if best vs second is almost always tiny or if first_action flips across most windows under no_discount.",
        "- later window-pool expansion can improve coverage, but current A1-pilot should use the existing 14-window preview set as requested.",
    ]
    (base_out / "A1_pilot_dryrun_report.md").write_text("\n".join(report_lines).strip() + "\n", encoding="utf-8")

    print(f"Saved A1 pilot outputs under {base_out}")


if __name__ == "__main__":
    main()
