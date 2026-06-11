#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib-cache")

import numpy as np

TANK_CAPACITY_KG = 1850.0 * 1025.0
PUMP_SIGN_EPS = 1e-3


@dataclass(frozen=True)
class SensitivityConfig:
    name: str
    weight_profile_name: str
    structure_profile_name: str
    active_ratio_low: float
    active_ratio_high: float
    pump_saving_ratio: float
    description: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "weight_profile_name": self.weight_profile_name,
            "structure_profile_name": self.structure_profile_name,
            "active_ratio_low": float(self.active_ratio_low),
            "active_ratio_high": float(self.active_ratio_high),
            "pump_saving_ratio": float(self.pump_saving_ratio),
            "description": self.description,
        }


HARD_GATE_THRESHOLDS = {
    "pitch_rms_rel": 0.05,
    "pitch_rms_abs_deg": 0.03,
    "roll_rms_rel": 0.05,
    "roll_rms_abs_deg": 0.03,
    "pitch_peak_rel": 0.05,
    "pitch_peak_abs_deg": 0.05,
    "roll_peak_rel": 0.05,
    "roll_peak_abs_deg": 0.05,
    "pose_exceed_time_rel": 0.15,
    "pose_exceed_time_abs_s": 60.0,
    "pump_work_rel": 0.10,
    "pump_work_abs_m3": 2.0,
    "switch_per_min_rel": 0.15,
    "switch_per_min_abs": 0.20,
    "direction_switch_rel": 0.15,
    "direction_switch_abs": 2.0,
    "sat_ratio_abs": 0.02,
    "capacity_margin_drop_abs": 0.015,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate the weight-tuning workflow for prediction-driven ballast decisions."
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("data/processed/wind_ml_10min/ballast_decision_fino1_meteo_aux_v1"),
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("outputs/wind_prediction/lstm_segmented_fino1_meteo_aux_v1"),
    )
    parser.add_argument("--split", default="test")
    parser.add_argument("--event-name", default="ballast_attention_event")
    parser.add_argument("--replay-rows", type=int, default=24)
    parser.add_argument("--dt", type=float, default=1.0)
    parser.add_argument("--window-repeats-per-label", type=int, default=2)
    parser.add_argument("--case-prefix", default="weight_sensitivity_v3")
    parser.add_argument(
        "--default-structure-profile",
        default="near_focus",
        help="Default structure profile used during weight discrimination.",
    )
    parser.add_argument(
        "--structure-profiles",
        default="near_focus,smooth_view,risk_focus",
        help="Comma-separated structure profiles for sensitivity validation.",
    )
    parser.add_argument(
        "--scale-modes",
        default="p95,2sigma",
        help="Comma-separated baseline scale modes to compare.",
    )
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _safe_float(row: dict, key: str) -> float:
    try:
        return float(row.get(key, 0.0))
    except Exception:
        return 0.0


def _percentile_scale(values: list[float], q: float, floor: float) -> float:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return float(floor)
    val = float(np.nanpercentile(np.abs(arr), q))
    return max(float(floor), val)


def _two_sigma_scale(values: list[float], floor: float) -> float:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return float(floor)
    val = float(2.0 * np.nanstd(arr))
    return max(float(floor), abs(val))


def estimate_baseline_scales(
    timeseries_rows: list[dict],
    deadband_pitch_deg: float,
    deadband_roll_deg: float,
    scale_mode: str,
):
    from wind_prediction import BaselineScales

    scale_fn = _percentile_scale if scale_mode == "p95" else _two_sigma_scale
    scale_arg = 95.0 if scale_mode == "p95" else None

    def calc(values: list[float], floor: float) -> float:
        if scale_mode == "p95":
            return scale_fn(values, scale_arg, floor)  # type: ignore[arg-type]
        return scale_fn(values, floor)  # type: ignore[misc]

    return BaselineScales(
        pitch_deg=calc([_safe_float(r, "pitch_deg") for r in timeseries_rows], 1e-3),
        roll_deg=calc([_safe_float(r, "roll_deg") for r in timeseries_rows], 1e-3),
        pump_rate_m3_min=calc([_safe_float(r, "pump_total_rate_m3_min") for r in timeseries_rows], 1.0),
        backlog_kg=calc([_safe_float(r, "pump_total_backlog_kg") for r in timeseries_rows], 1.0),
        wind_speed_mps=calc([_safe_float(r, "wind_speed") for r in timeseries_rows], 1.0),
        deadband_pitch_deg=float(deadband_pitch_deg),
        deadband_roll_deg=float(deadband_roll_deg),
        saturation_margin_ratio=0.08,
        eps=1e-6,
    )


def wrap_angle_diff_deg(a: float, b: float) -> float:
    diff = (float(b) - float(a) + 180.0) % 360.0 - 180.0
    return abs(diff)


def select_representative_windows(
    replay,
    event_name: str,
    row_count: int,
    repeats_per_label: int,
) -> list[dict[str, Any]]:
    event_idx = replay.event_columns.index(event_name)
    samples = replay.samples
    trace_rows = replay.trace_rows
    trace_idx = {row.timestamp: idx for idx, row in enumerate(trace_rows)}
    max_start_idx = len(samples) - row_count
    if max_start_idx < 0:
        max_start_idx = 0

    candidates: list[dict[str, Any]] = []
    for sample in samples[: max_start_idx + 1]:
        start_idx = trace_idx.get(sample.history_end)
        if start_idx is None:
            continue
        rows = trace_rows[start_idx : start_idx + row_count]
        if len(rows) < row_count:
            continue
        ws = np.array([row.wind_speed_ms for row in rows], dtype=float)
        wd = np.array([row.wind_dir_deg for row in rows], dtype=float)
        ws_jump = float(np.max(np.abs(np.diff(ws)))) if ws.size > 1 else 0.0
        wd_jump = (
            float(np.max([wrap_angle_diff_deg(wd[i], wd[i + 1]) for i in range(len(wd) - 1)]))
            if wd.size > 1
            else 0.0
        )
        ws_std = float(np.std(ws))
        wd_var = (
            float(np.mean([wrap_angle_diff_deg(wd[i], wd[i + 1]) for i in range(len(wd) - 1)]))
            if wd.size > 1
            else 0.0
        )
        event_score = float(sample.y_event[event_idx])
        combo_score = 1.5 * event_score + 0.2 * ws_jump + 0.02 * wd_jump + 0.15 * ws_std + 0.01 * wd_var
        candidates.append(
            {
                "timestamp": sample.history_end,
                "event_score": event_score,
                "ws_jump": ws_jump,
                "wd_jump": wd_jump,
                "combo_score": combo_score,
            }
        )

    label_specs = (
        ("high_event", "event_score"),
        ("speed_jump", "ws_jump"),
        ("direction_jump", "wd_jump"),
        ("combined_variability", "combo_score"),
    )
    min_gap_s = row_count * replay.update_interval_s // 2
    picked: list[datetime] = []
    selected: list[dict[str, Any]] = []
    for label, key in label_specs:
        rows = sorted(candidates, key=lambda x: float(x[key]), reverse=True)
        local_count = 0
        for row in rows:
            ts = row["timestamp"]
            if all(abs((ts - prev).total_seconds()) >= min_gap_s for prev in picked):
                picked.append(ts)
                split_role = "tuning" if local_count == 0 else "holdout"
                selected.append(
                    {
                        "window_label": label,
                        "split_role": split_role,
                        "start_timestamp": ts,
                        "rank_in_label": local_count + 1,
                    }
                )
                local_count += 1
            if local_count >= repeats_per_label:
                break
    return selected


def sign_switch_count(series: np.ndarray) -> int:
    signs: list[int] = []
    for val in series:
        if abs(float(val)) <= PUMP_SIGN_EPS:
            continue
        signs.append(1 if val > 0.0 else -1)
    if len(signs) <= 1:
        return 0
    return int(sum(1 for a, b in zip(signs[:-1], signs[1:]) if a != b))


def summarize_timeseries(
    timeseries_rows: list[dict],
    dt: float,
    deadband_pitch_deg: float,
    deadband_roll_deg: float,
    tank_capacity_kg: float,
) -> dict[str, float]:
    pitch = np.asarray([_safe_float(r, "pitch_deg") for r in timeseries_rows], dtype=float)
    roll = np.asarray([_safe_float(r, "roll_deg") for r in timeseries_rows], dtype=float)
    pump_total = np.asarray([_safe_float(r, "pump_total_rate_m3_min") for r in timeseries_rows], dtype=float)
    backlog = np.asarray([_safe_float(r, "pump_total_backlog_kg") for r in timeseries_rows], dtype=float)
    tank1 = np.asarray([_safe_float(r, "tank1_kg") for r in timeseries_rows], dtype=float)
    tank2 = np.asarray([_safe_float(r, "tank2_kg") for r in timeseries_rows], dtype=float)
    tank3 = np.asarray([_safe_float(r, "tank3_kg") for r in timeseries_rows], dtype=float)

    pitch_peak = float(np.max(np.abs(pitch))) if pitch.size else 0.0
    roll_peak = float(np.max(np.abs(roll))) if roll.size else 0.0
    pitch_exceed_time_s = float(np.sum(np.abs(pitch) > deadband_pitch_deg) * dt)
    roll_exceed_time_s = float(np.sum(np.abs(roll) > deadband_roll_deg) * dt)
    pose_exceed_time_s = float(
        np.sum((np.abs(pitch) > deadband_pitch_deg) | (np.abs(roll) > deadband_roll_deg)) * dt
    )
    pump_work_m3 = float(np.sum(np.abs(pump_total)) * dt / 60.0)
    direction_switch_count = float(sign_switch_count(pump_total))

    mass_ratio = np.vstack([tank1, tank2, tank3]) / max(tank_capacity_kg, 1e-6)
    lower_margin = np.min(mass_ratio, axis=0) if mass_ratio.size else np.array([1.0])
    upper_margin = np.min(1.0 - mass_ratio, axis=0) if mass_ratio.size else np.array([1.0])
    capacity_headroom_min_ratio = float(np.min(np.minimum(lower_margin, upper_margin))) if lower_margin.size else 1.0

    return {
        "pitch_peak_abs_deg": pitch_peak,
        "roll_peak_abs_deg": roll_peak,
        "pitch_exceed_time_s": pitch_exceed_time_s,
        "roll_exceed_time_s": roll_exceed_time_s,
        "pose_exceed_time_s": pose_exceed_time_s,
        "pump_work_m3": pump_work_m3,
        "direction_switch_count": direction_switch_count,
        "capacity_headroom_min_ratio": capacity_headroom_min_ratio,
        "pump_backlog_p95_kg": _percentile_scale(backlog.tolist(), 95.0, 0.0),
    }


def _nonworse_limit(closed_val: float, rel_tol: float, abs_tol: float) -> float:
    return float(closed_val + max(abs_tol, abs(closed_val) * rel_tol))


def _safe_rel_change(candidate: float, baseline: float, eps: float = 1e-6) -> float:
    denom = max(abs(float(baseline)), eps)
    return float((float(candidate) - float(baseline)) / denom)


def build_hard_gate_report(candidate_row: dict[str, float], closed_row: dict[str, float]) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if candidate_row["pitch_rms_deg"] > _nonworse_limit(
        closed_row["pitch_rms_deg"], HARD_GATE_THRESHOLDS["pitch_rms_rel"], HARD_GATE_THRESHOLDS["pitch_rms_abs_deg"]
    ):
        reasons.append("pitch_rms")
    if candidate_row["roll_rms_deg"] > _nonworse_limit(
        closed_row["roll_rms_deg"], HARD_GATE_THRESHOLDS["roll_rms_rel"], HARD_GATE_THRESHOLDS["roll_rms_abs_deg"]
    ):
        reasons.append("roll_rms")
    if candidate_row["pitch_peak_abs_deg"] > _nonworse_limit(
        closed_row["pitch_peak_abs_deg"], HARD_GATE_THRESHOLDS["pitch_peak_rel"], HARD_GATE_THRESHOLDS["pitch_peak_abs_deg"]
    ):
        reasons.append("pitch_peak")
    if candidate_row["roll_peak_abs_deg"] > _nonworse_limit(
        closed_row["roll_peak_abs_deg"], HARD_GATE_THRESHOLDS["roll_peak_rel"], HARD_GATE_THRESHOLDS["roll_peak_abs_deg"]
    ):
        reasons.append("roll_peak")
    if candidate_row["pose_exceed_time_s"] > _nonworse_limit(
        closed_row["pose_exceed_time_s"],
        HARD_GATE_THRESHOLDS["pose_exceed_time_rel"],
        HARD_GATE_THRESHOLDS["pose_exceed_time_abs_s"],
    ):
        reasons.append("pose_exceed_time")
    if candidate_row["pump_work_m3"] > _nonworse_limit(
        closed_row["pump_work_m3"], HARD_GATE_THRESHOLDS["pump_work_rel"], HARD_GATE_THRESHOLDS["pump_work_abs_m3"]
    ):
        reasons.append("pump_work")
    if candidate_row["switch_per_min"] > _nonworse_limit(
        closed_row["switch_per_min"],
        HARD_GATE_THRESHOLDS["switch_per_min_rel"],
        HARD_GATE_THRESHOLDS["switch_per_min_abs"],
    ):
        reasons.append("switch_per_min")
    if candidate_row["direction_switch_count"] > _nonworse_limit(
        closed_row["direction_switch_count"],
        HARD_GATE_THRESHOLDS["direction_switch_rel"],
        HARD_GATE_THRESHOLDS["direction_switch_abs"],
    ):
        reasons.append("direction_switch")
    if candidate_row["sat_ratio"] > closed_row["sat_ratio"] + HARD_GATE_THRESHOLDS["sat_ratio_abs"]:
        reasons.append("sat_ratio")
    if candidate_row["capacity_headroom_min_ratio"] < (
        closed_row["capacity_headroom_min_ratio"] - HARD_GATE_THRESHOLDS["capacity_margin_drop_abs"]
    ):
        reasons.append("capacity_margin")
    return (len(reasons) == 0), reasons


def build_weight_configs(default_structure_profile: str) -> list[SensitivityConfig]:
    return [
        SensitivityConfig("balanced", "balanced", default_structure_profile, 0.25, 0.50, 0.12, "temporary default"),
        SensitivityConfig(
            "attitude_priority", "attitude_priority", default_structure_profile, 0.25, 0.50, 0.12, "pose first"
        ),
        SensitivityConfig(
            "pump_priority", "pump_priority", default_structure_profile, 0.25, 0.50, 0.12, "pump burden first"
        ),
        SensitivityConfig(
            "switch_smooth", "switch_smooth", default_structure_profile, 0.24, 0.45, 0.12, "switch smoothing"
        ),
        SensitivityConfig(
            "risk_aggressive", "risk_aggressive", default_structure_profile, 0.26, 0.55, 0.12, "risk amplified"
        ),
        SensitivityConfig(
            "risk_conservative", "risk_conservative", default_structure_profile, 0.20, 0.40, 0.10, "risk conservative"
        ),
    ]


def build_structure_configs(base: SensitivityConfig, structure_profiles: list[str]) -> list[SensitivityConfig]:
    rows: list[SensitivityConfig] = []
    for structure_name in structure_profiles:
        rows.append(
            SensitivityConfig(
                name=f"{base.name}__{structure_name}",
                weight_profile_name=base.weight_profile_name,
                structure_profile_name=structure_name,
                active_ratio_low=base.active_ratio_low,
                active_ratio_high=base.active_ratio_high,
                pump_saving_ratio=base.pump_saving_ratio,
                description=f"{base.name} under {structure_name}",
            )
        )
    return rows


def summarize_candidate_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    selected_counts = Counter(str(r.get("selected_plan", "")) for r in records)
    rank_counts = Counter(str(r.get("rank_order", "")) for r in records if r.get("rank_order"))
    screen_counts = Counter()
    cost_totals: dict[str, list[float]] = defaultdict(list)
    cost_att: dict[str, list[float]] = defaultdict(list)
    cost_pump: dict[str, list[float]] = defaultdict(list)
    cost_switch: dict[str, list[float]] = defaultdict(list)
    cost_sat: dict[str, list[float]] = defaultdict(list)
    action_norms: dict[str, list[float]] = defaultdict(list)
    total = max(len(records), 1)
    for r in records:
        snap = str(r.get("screened_snapshot", ""))
        if snap:
            for item in snap.split(";"):
                if not item:
                    continue
                _name, reason = item.split(":", 1)
                screen_counts[reason] += 1
        for candidate in ("hold", "active_adjust", "pump_saving"):
            if f"{candidate}_cost_total" in r:
                cost_totals[candidate].append(float(r[f"{candidate}_cost_total"]))
                cost_att[candidate].append(float(r.get(f"{candidate}_cost_attitude", 0.0)))
                cost_pump[candidate].append(float(r.get(f"{candidate}_cost_pump", 0.0)))
                cost_switch[candidate].append(float(r.get(f"{candidate}_cost_switch", 0.0)))
                cost_sat[candidate].append(float(r.get(f"{candidate}_cost_saturation", 0.0)))
                action_norms[candidate].append(float(r.get(f"{candidate}_action_norm", 0.0)))
    return {
        "record_count": len(records),
        "selected_hold_ratio": float(selected_counts.get("hold", 0) / total),
        "selected_active_ratio": float(selected_counts.get("active_adjust", 0) / total),
        "selected_pump_saving_ratio": float(selected_counts.get("pump_saving", 0) / total),
        "selected_hold_count": int(selected_counts.get("hold", 0)),
        "selected_active_count": int(selected_counts.get("active_adjust", 0)),
        "selected_pump_saving_count": int(selected_counts.get("pump_saving", 0)),
        "unique_rank_count": int(len(rank_counts)),
        "top_rank_pattern": rank_counts.most_common(1)[0][0] if rank_counts else "",
        "screen_reason_counts": ";".join(f"{k}:{v}" for k, v in screen_counts.items()),
        "hold_cost_mean": float(np.mean(cost_totals["hold"])) if cost_totals["hold"] else np.nan,
        "active_cost_mean": float(np.mean(cost_totals["active_adjust"])) if cost_totals["active_adjust"] else np.nan,
        "pump_saving_cost_mean": float(np.mean(cost_totals["pump_saving"])) if cost_totals["pump_saving"] else np.nan,
        "hold_attitude_mean": float(np.mean(cost_att["hold"])) if cost_att["hold"] else np.nan,
        "active_attitude_mean": float(np.mean(cost_att["active_adjust"])) if cost_att["active_adjust"] else np.nan,
        "pump_saving_attitude_mean": float(np.mean(cost_att["pump_saving"])) if cost_att["pump_saving"] else np.nan,
        "hold_pump_mean": float(np.mean(cost_pump["hold"])) if cost_pump["hold"] else np.nan,
        "active_pump_mean": float(np.mean(cost_pump["active_adjust"])) if cost_pump["active_adjust"] else np.nan,
        "pump_saving_pump_mean": float(np.mean(cost_pump["pump_saving"])) if cost_pump["pump_saving"] else np.nan,
        "hold_switch_mean": float(np.mean(cost_switch["hold"])) if cost_switch["hold"] else np.nan,
        "active_switch_mean": float(np.mean(cost_switch["active_adjust"])) if cost_switch["active_adjust"] else np.nan,
        "pump_saving_switch_mean": float(np.mean(cost_switch["pump_saving"])) if cost_switch["pump_saving"] else np.nan,
        "hold_action_norm_mean": float(np.mean(action_norms["hold"])) if action_norms["hold"] else np.nan,
        "active_action_norm_mean": float(np.mean(action_norms["active_adjust"])) if action_norms["active_adjust"] else np.nan,
        "pump_saving_action_norm_mean": float(np.mean(action_norms["pump_saving"])) if action_norms["pump_saving"] else np.nan,
    }


def aggregate_config_summary(rows: list[dict[str, Any]], group_keys: list[str]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[k] for k in group_keys)].append(row)
    summaries: list[dict[str, Any]] = []
    for key, subset in grouped.items():
        summary = {k: v for k, v in zip(group_keys, key)}
        pass_count = int(sum(int(r["pass_hard_gate"]) for r in subset))
        summary.update(
            {
                "case_count": int(len(subset)),
                "pass_count": pass_count,
                "pass_ratio": float(pass_count / max(len(subset), 1)),
                "mean_delta_pitch_rms_vs_closed": float(np.mean([float(r["delta_pitch_rms_vs_closed"]) for r in subset])),
                "mean_delta_roll_rms_vs_closed": float(np.mean([float(r["delta_roll_rms_vs_closed"]) for r in subset])),
                "mean_delta_pitch_peak_vs_closed": float(np.mean([float(r["delta_pitch_peak_vs_closed"]) for r in subset])),
                "mean_delta_roll_peak_vs_closed": float(np.mean([float(r["delta_roll_peak_vs_closed"]) for r in subset])),
                "mean_delta_pose_exceed_time_s_vs_closed": float(
                    np.mean([float(r["delta_pose_exceed_time_s_vs_closed"]) for r in subset])
                ),
                "mean_delta_pump_work_m3_vs_closed": float(np.mean([float(r["delta_pump_work_m3_vs_closed"]) for r in subset])),
                "mean_delta_switch_vs_closed": float(np.mean([float(r["delta_switch_vs_closed"]) for r in subset])),
                "mean_delta_direction_switch_vs_closed": float(
                    np.mean([float(r["delta_direction_switch_vs_closed"]) for r in subset])
                ),
                "mean_delta_sat_vs_closed": float(np.mean([float(r["delta_sat_vs_closed"]) for r in subset])),
                "mean_delta_capacity_headroom_vs_closed": float(
                    np.mean([float(r["delta_capacity_headroom_vs_closed"]) for r in subset])
                ),
                "mean_selected_hold_ratio": float(np.mean([float(r["selected_hold_ratio"]) for r in subset])),
                "mean_selected_active_ratio": float(np.mean([float(r["selected_active_ratio"]) for r in subset])),
                "mean_selected_pump_saving_ratio": float(np.mean([float(r["selected_pump_saving_ratio"]) for r in subset])),
            }
        )
        pose_index = (
            summary["mean_delta_pitch_rms_vs_closed"]
            + summary["mean_delta_roll_rms_vs_closed"]
            + 0.5 * (summary["mean_delta_pitch_peak_vs_closed"] + summary["mean_delta_roll_peak_vs_closed"])
        )
        burden_index = (
            max(0.0, summary["mean_delta_pump_work_m3_vs_closed"])
            + max(0.0, summary["mean_delta_switch_vs_closed"])
            + max(0.0, summary["mean_delta_direction_switch_vs_closed"])
            + 10.0 * max(0.0, summary["mean_delta_sat_vs_closed"])
            + 10.0 * max(0.0, -summary["mean_delta_capacity_headroom_vs_closed"])
        )
        summary["pose_index"] = float(pose_index)
        summary["burden_index"] = float(burden_index)
        summaries.append(summary)
    return summaries


def choose_recommended(rows: list[dict[str, Any]], config_key: str = "config_name") -> dict[str, Any]:
    return sorted(
        rows,
        key=lambda r: (
            -int(r["pass_count"]),
            float(r["pose_index"]),
            float(r["burden_index"]),
            str(r[config_key]),
        ),
    )[0]


def compute_discrimination_rows(
    records_by_case: dict[tuple[str, str, str], list[dict[str, Any]]],
    windows: list[dict[str, Any]],
    config_names: list[str],
    scale_mode: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for window in windows:
        if window["split_role"] not in {"tuning", "holdout"}:
            continue
        series = []
        for config_name in config_names:
            key = (window["window_label"], window["split_role"], f"{config_name}|{scale_mode}|near_focus")
            records = records_by_case.get(key, [])
            if records:
                series.append(records)
        if len(series) < 2:
            continue
        n = min(len(s) for s in series)
        if n <= 0:
            continue
        selection_div, ranking_div, action_div = 0, 0, 0
        for idx in range(n):
            selected = {str(s[idx].get("selected_plan", "")) for s in series}
            rankings = {str(s[idx].get("rank_order", "")) for s in series}
            actions = {str(s[idx].get("selected_action_signature", "")) for s in series}
            if len(selected) > 1:
                selection_div += 1
            if len(rankings) > 1:
                ranking_div += 1
            if len(actions) > 1:
                action_div += 1
        rows.append(
            {
                "window_label": window["window_label"],
                "split_role": window["split_role"],
                "scale_mode": scale_mode,
                "compared_config_count": len(series),
                "steps_compared": n,
                "selection_divergence_rate": float(selection_div / n),
                "ranking_divergence_rate": float(ranking_div / n),
                "action_divergence_rate": float(action_div / n),
            }
        )
    return rows


def build_summary_markdown(
    *,
    actual_wall_seconds: float,
    tuning_recommended_p95: str,
    tuning_recommended_2sigma: str,
    operational_default: str,
    holdout_row: dict[str, Any],
    discrimination_rows: list[dict[str, Any]],
    structure_rows: list[dict[str, Any]],
    scale_rows: list[dict[str, Any]],
) -> str:
    tuning_discrimination = [r for r in discrimination_rows if r["split_role"] == "tuning"]
    max_sel_div = max((float(r["selection_divergence_rate"]) for r in tuning_discrimination), default=0.0)
    max_rank_div = max((float(r["ranking_divergence_rate"]) for r in tuning_discrimination), default=0.0)
    discrimination_pass = max_sel_div >= 0.05 or max_rank_div >= 0.05
    holdout_pass = (
        int(holdout_row.get("pass_count", 0)) == int(holdout_row.get("case_count", 0))
        and float(holdout_row.get("pose_index", 0.0)) <= 0.0
        and float(holdout_row.get("burden_index", 0.0)) <= 0.0
    )
    structure_pass = all(int(r.get("pass_count", 0)) == int(r.get("case_count", 0)) for r in structure_rows)
    scale_pass = tuning_recommended_p95 == tuning_recommended_2sigma
    lines = [
        "# Tuning Workflow Validation Summary",
        "",
        f"- actual wall time: `{actual_wall_seconds:.1f} s`",
        f"- operational temporary default: `{operational_default}`",
        f"- tuning-selected p95 candidate: `{tuning_recommended_p95}`",
        "- note: `the operational temporary default should not change unless hold-out and sensitivity checks also pass`",
        f"- discrimination check: `{'pass' if discrimination_pass else 'fail'}`",
        f"- hold-out validation: `{'pass' if holdout_pass else 'fail'}`",
        f"- structure-profile sensitivity check: `{'pass' if structure_pass else 'fail'}`",
        f"- scale-definition sensitivity check: `{'pass' if scale_pass else 'fail'}`",
        "",
        "## Recommendation",
        "",
        f"- current operational temporary default: `{operational_default}`",
        f"- p95 tuning recommended: `{tuning_recommended_p95}`",
        f"- 2sigma tuning recommended: `{tuning_recommended_2sigma}`",
        f"- hold-out summary for temporary default: pose_index=`{holdout_row.get('pose_index')}`, burden_index=`{holdout_row.get('burden_index')}`",
        "",
        "## Discrimination Highlights",
        "",
        f"- max tuning selection divergence rate: `{max_sel_div:.3f}`",
        f"- max tuning ranking divergence rate: `{max_rank_div:.3f}`",
        "- if these stay near zero in many windows, the candidate set or cost resolution is insufficient",
        "",
        "## Next-Step Gate",
        "",
        "- only continue small-range weight tuning if discrimination, hold-out, and sensitivity checks are at least directionally acceptable",
        "- otherwise adjust candidate envelope or scoring structure first, instead of expanding parameter grids",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root / "src"))
    sys.path.insert(0, str(repo_root / "archive" / "legacy_fowt_control"))

    from defaults import DEFAULT_CONTROLLER_CFG
    from run_validation import discover_stiffness_file, run_closed_loop_case
    from wind_prediction import (
        CandidatePlanEvaluator,
        DEFAULT_STRUCTURE_PROFILES,
        DEFAULT_WEIGHT_PROFILES,
        Fino1ReplayDataset,
        ForecastModelAdapter,
        ModelDecisionProvider,
    )

    t0 = time.perf_counter()
    replay = Fino1ReplayDataset(dataset_dir=args.dataset_dir, split=args.split)
    windows = select_representative_windows(
        replay=replay,
        event_name=args.event_name,
        row_count=args.replay_rows,
        repeats_per_label=max(int(args.window_repeats_per_label), 2),
    )

    adapter = ForecastModelAdapter(model_dir=args.model_dir, dataset_dir=args.dataset_dir, device="cpu")
    excel_path = discover_stiffness_file()
    if not excel_path:
        archive_candidate = repo_root / "archive" / "legacy_fowt_control" / "data" / "副本水平刚度曲线.xlsx"
        if archive_candidate.exists():
            excel_path = str(archive_candidate)
    if not excel_path:
        raise SystemExit("No stiffness file found under data/.")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = repo_root / "results" / f"{args.case_prefix}_{args.split}_{ts}"
    paper_dir = out_dir / "paper_ready"
    tables_dir = paper_dir / "tables"
    debug_dir = out_dir / "debug"
    candidate_scores_dir = debug_dir / "candidate_scores"
    paper_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    candidate_scores_dir.mkdir(parents=True, exist_ok=True)

    deadband_pitch_deg = float(DEFAULT_CONTROLLER_CFG["deadband_pitch"])
    deadband_roll_deg = float(DEFAULT_CONTROLLER_CFG["deadband_roll"])
    structure_profiles = [
        item.strip() for item in str(args.structure_profiles).split(",") if item.strip() in DEFAULT_STRUCTURE_PROFILES
    ]
    if not structure_profiles:
        structure_profiles = ["near_focus", "smooth_view", "risk_focus"]
    scale_modes = [item.strip() for item in str(args.scale_modes).split(",") if item.strip() in {"p95", "2sigma"}]
    if not scale_modes:
        scale_modes = ["p95", "2sigma"]

    weight_configs = build_weight_configs(args.default_structure_profile)
    all_rows: list[dict[str, Any]] = []
    baseline_rows: list[dict[str, Any]] = []
    choice_rows: list[dict[str, Any]] = []
    records_by_case: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    closed_baselines: dict[tuple[str, str], dict[str, Any]] = {}

    # Shared closed-only baselines per window.
    for window in windows:
        wind_trace = replay.build_wind_trace(
            start_timestamp=window["start_timestamp"],
            row_count=args.replay_rows,
            dt_s=args.dt,
        )
        n_steps = int(wind_trace["n_steps"])
        closed_row, closed_timeseries = run_closed_loop_case(
            excel_path=excel_path,
            case_name=f"{args.case_prefix}_{window['window_label']}_{window['split_role']}_closed_only",
            dt=args.dt,
            n_steps=n_steps,
            wind_trace=wind_trace,
            trim_cfg=None,
            control_enabled=True,
            record_timeseries=True,
            preview_trim_provider=None,
        )
        closed_metrics = {
            **{k: float(v) for k, v in closed_row.items() if isinstance(v, (int, float))},
            **summarize_timeseries(
                closed_timeseries,
                dt=args.dt,
                deadband_pitch_deg=deadband_pitch_deg,
                deadband_roll_deg=deadband_roll_deg,
                tank_capacity_kg=TANK_CAPACITY_KG,
            ),
        }
        closed_baselines[(window["window_label"], window["split_role"])] = {
            "metrics": closed_metrics,
            "timeseries": closed_timeseries,
            "wind_trace": wind_trace,
            "n_steps": n_steps,
        }
        for scale_mode in scale_modes:
            baseline_scales = estimate_baseline_scales(
                timeseries_rows=closed_timeseries,
                deadband_pitch_deg=deadband_pitch_deg,
                deadband_roll_deg=deadband_roll_deg,
                scale_mode=scale_mode,
            )
            baseline_rows.append(
                {
                    "window_label": window["window_label"],
                    "split_role": window["split_role"],
                    "start_timestamp": window["start_timestamp"].strftime("%Y-%m-%d %H:%M:%S"),
                    "scale_mode": scale_mode,
                    **baseline_scales.to_dict(),
                }
            )

    # Phase 1: weight discrimination under default structure and two scale modes.
    for window in windows:
        closed_blob = closed_baselines[(window["window_label"], window["split_role"])]
        closed_metrics = closed_blob["metrics"]
        wind_trace = closed_blob["wind_trace"]
        n_steps = closed_blob["n_steps"]
        for scale_mode in scale_modes:
            baseline_scales = estimate_baseline_scales(
                timeseries_rows=closed_blob["timeseries"],
                deadband_pitch_deg=deadband_pitch_deg,
                deadband_roll_deg=deadband_roll_deg,
                scale_mode=scale_mode,
            )
            for config in weight_configs:
                evaluator = CandidatePlanEvaluator(
                    baseline_scales=baseline_scales,
                    weight_profile=DEFAULT_WEIGHT_PROFILES[config.weight_profile_name],
                    structure_profile=DEFAULT_STRUCTURE_PROFILES[config.structure_profile_name],
                    active_ratio_low=config.active_ratio_low,
                    active_ratio_high=config.active_ratio_high,
                    pump_saving_ratio=config.pump_saving_ratio,
                )
                provider = ModelDecisionProvider(
                    replay_dataset=replay,
                    forecast_adapter=adapter,
                    start_timestamp=window["start_timestamp"],
                    evaluator=evaluator,
                )
                row, timeseries = run_closed_loop_case(
                    excel_path=excel_path,
                    case_name=f"{args.case_prefix}_{window['window_label']}_{window['split_role']}_{config.name}_{scale_mode}",
                    dt=args.dt,
                    n_steps=n_steps,
                    wind_trace=wind_trace,
                    trim_cfg=None,
                    control_enabled=True,
                    record_timeseries=True,
                    preview_trim_provider=provider,
                )
                extra = summarize_timeseries(
                    timeseries,
                    dt=args.dt,
                    deadband_pitch_deg=deadband_pitch_deg,
                    deadband_roll_deg=deadband_roll_deg,
                    tank_capacity_kg=TANK_CAPACITY_KG,
                )
                candidate_metrics = {**{k: float(v) for k, v in row.items() if isinstance(v, (int, float))}, **extra}
                pass_gate, fail_reasons = build_hard_gate_report(candidate_metrics, closed_metrics)
                records = list(provider.evaluator.records)
                records_by_case[(window["window_label"], window["split_role"], f"{config.name}|{scale_mode}|near_focus")] = records
                write_csv(
                    candidate_scores_dir
                    / f"weight_validation_{window['window_label']}_{window['split_role']}_{config.name}_{scale_mode}.csv",
                    records,
                )
                choice_blob = summarize_candidate_records(records)
                choice_rows.append(
                    {
                        "phase": "weight_validation",
                        "window_label": window["window_label"],
                        "split_role": window["split_role"],
                        "scale_mode": scale_mode,
                        "config_name": config.name,
                        **choice_blob,
                    }
                )
                all_rows.append(
                    {
                        "phase": "weight_validation",
                        "window_label": window["window_label"],
                        "split_role": window["split_role"],
                        "start_timestamp": window["start_timestamp"].strftime("%Y-%m-%d %H:%M:%S"),
                        "scale_mode": scale_mode,
                        "config_name": config.name,
                        "weight_profile_name": config.weight_profile_name,
                        "structure_profile_name": config.structure_profile_name,
                        "pitch_rms_deg": candidate_metrics["pitch_rms_deg"],
                        "roll_rms_deg": candidate_metrics["roll_rms_deg"],
                        "pitch_peak_abs_deg": candidate_metrics["pitch_peak_abs_deg"],
                        "roll_peak_abs_deg": candidate_metrics["roll_peak_abs_deg"],
                        "pose_exceed_time_s": candidate_metrics["pose_exceed_time_s"],
                        "pump_work_m3": candidate_metrics["pump_work_m3"],
                        "switch_per_min": candidate_metrics["switch_per_min"],
                        "direction_switch_count": candidate_metrics["direction_switch_count"],
                        "sat_ratio": candidate_metrics["sat_ratio"],
                        "capacity_headroom_min_ratio": candidate_metrics["capacity_headroom_min_ratio"],
                        "pass_hard_gate": int(pass_gate),
                        "fail_reasons": ";".join(fail_reasons),
                        "delta_pitch_rms_vs_closed": candidate_metrics["pitch_rms_deg"] - closed_metrics["pitch_rms_deg"],
                        "delta_roll_rms_vs_closed": candidate_metrics["roll_rms_deg"] - closed_metrics["roll_rms_deg"],
                        "delta_pitch_peak_vs_closed": candidate_metrics["pitch_peak_abs_deg"] - closed_metrics["pitch_peak_abs_deg"],
                        "delta_roll_peak_vs_closed": candidate_metrics["roll_peak_abs_deg"] - closed_metrics["roll_peak_abs_deg"],
                        "delta_pose_exceed_time_s_vs_closed": candidate_metrics["pose_exceed_time_s"] - closed_metrics["pose_exceed_time_s"],
                        "delta_pump_work_m3_vs_closed": candidate_metrics["pump_work_m3"] - closed_metrics["pump_work_m3"],
                        "delta_switch_vs_closed": candidate_metrics["switch_per_min"] - closed_metrics["switch_per_min"],
                        "delta_direction_switch_vs_closed": candidate_metrics["direction_switch_count"] - closed_metrics["direction_switch_count"],
                        "delta_sat_vs_closed": candidate_metrics["sat_ratio"] - closed_metrics["sat_ratio"],
                        "delta_capacity_headroom_vs_closed": candidate_metrics["capacity_headroom_min_ratio"] - closed_metrics["capacity_headroom_min_ratio"],
                        "selected_hold_ratio": choice_blob["selected_hold_ratio"],
                        "selected_active_ratio": choice_blob["selected_active_ratio"],
                        "selected_pump_saving_ratio": choice_blob["selected_pump_saving_ratio"],
                        "unique_rank_count": choice_blob["unique_rank_count"],
                    }
                )

    discrimination_rows = compute_discrimination_rows(
        records_by_case=records_by_case,
        windows=windows,
        config_names=[cfg.name for cfg in weight_configs],
        scale_mode="p95",
    )

    weight_rows = [r for r in all_rows if r["phase"] == "weight_validation"]
    tuning_summaries = aggregate_config_summary(
        [r for r in weight_rows if r["split_role"] == "tuning" and r["scale_mode"] == "p95"],
        ["config_name", "weight_profile_name", "structure_profile_name", "scale_mode"],
    )
    holdout_summaries = aggregate_config_summary(
        [r for r in weight_rows if r["split_role"] == "holdout" and r["scale_mode"] == "p95"],
        ["config_name", "weight_profile_name", "structure_profile_name", "scale_mode"],
    )
    tuning_summaries_2sigma = aggregate_config_summary(
        [r for r in weight_rows if r["split_role"] == "tuning" and r["scale_mode"] == "2sigma"],
        ["config_name", "weight_profile_name", "structure_profile_name", "scale_mode"],
    )
    recommended_p95 = choose_recommended(tuning_summaries)
    recommended_2sigma = choose_recommended(tuning_summaries_2sigma)
    holdout_map = {row["config_name"]: row for row in holdout_summaries}
    holdout_for_recommended = holdout_map[recommended_p95["config_name"]]
    temp_default_cfg = next(cfg for cfg in weight_configs if cfg.name == recommended_p95["config_name"])

    # Phase 2: structure sensitivity for the temporary default under p95.
    structure_rows: list[dict[str, Any]] = []
    structure_configs = build_structure_configs(temp_default_cfg, structure_profiles)
    for window in windows:
        closed_blob = closed_baselines[(window["window_label"], window["split_role"])]
        closed_metrics = closed_blob["metrics"]
        wind_trace = closed_blob["wind_trace"]
        n_steps = closed_blob["n_steps"]
        baseline_scales = estimate_baseline_scales(
            timeseries_rows=closed_blob["timeseries"],
            deadband_pitch_deg=deadband_pitch_deg,
            deadband_roll_deg=deadband_roll_deg,
            scale_mode="p95",
        )
        for config in structure_configs:
            evaluator = CandidatePlanEvaluator(
                baseline_scales=baseline_scales,
                weight_profile=DEFAULT_WEIGHT_PROFILES[config.weight_profile_name],
                structure_profile=DEFAULT_STRUCTURE_PROFILES[config.structure_profile_name],
                active_ratio_low=config.active_ratio_low,
                active_ratio_high=config.active_ratio_high,
                pump_saving_ratio=config.pump_saving_ratio,
            )
            provider = ModelDecisionProvider(
                replay_dataset=replay,
                forecast_adapter=adapter,
                start_timestamp=window["start_timestamp"],
                evaluator=evaluator,
            )
            row, timeseries = run_closed_loop_case(
                excel_path=excel_path,
                case_name=f"{args.case_prefix}_{window['window_label']}_{window['split_role']}_{config.name}_structure",
                dt=args.dt,
                n_steps=n_steps,
                wind_trace=wind_trace,
                trim_cfg=None,
                control_enabled=True,
                record_timeseries=True,
                preview_trim_provider=provider,
            )
            extra = summarize_timeseries(
                timeseries,
                dt=args.dt,
                deadband_pitch_deg=deadband_pitch_deg,
                deadband_roll_deg=deadband_roll_deg,
                tank_capacity_kg=TANK_CAPACITY_KG,
            )
            candidate_metrics = {**{k: float(v) for k, v in row.items() if isinstance(v, (int, float))}, **extra}
            pass_gate, fail_reasons = build_hard_gate_report(candidate_metrics, closed_metrics)
            records = list(provider.evaluator.records)
            write_csv(
                candidate_scores_dir
                / f"structure_validation_{window['window_label']}_{window['split_role']}_{config.name}.csv",
                records,
            )
            choice_blob = summarize_candidate_records(records)
            structure_rows.append(
                {
                    "phase": "structure_validation",
                    "window_label": window["window_label"],
                    "split_role": window["split_role"],
                    "structure_profile_name": config.structure_profile_name,
                    "config_name": config.name,
                    "pass_hard_gate": int(pass_gate),
                    "fail_reasons": ";".join(fail_reasons),
                    "delta_pitch_rms_vs_closed": candidate_metrics["pitch_rms_deg"] - closed_metrics["pitch_rms_deg"],
                    "delta_roll_rms_vs_closed": candidate_metrics["roll_rms_deg"] - closed_metrics["roll_rms_deg"],
                    "delta_pitch_peak_vs_closed": candidate_metrics["pitch_peak_abs_deg"] - closed_metrics["pitch_peak_abs_deg"],
                    "delta_roll_peak_vs_closed": candidate_metrics["roll_peak_abs_deg"] - closed_metrics["roll_peak_abs_deg"],
                    "delta_pose_exceed_time_s_vs_closed": candidate_metrics["pose_exceed_time_s"] - closed_metrics["pose_exceed_time_s"],
                    "delta_pump_work_m3_vs_closed": candidate_metrics["pump_work_m3"] - closed_metrics["pump_work_m3"],
                    "delta_switch_vs_closed": candidate_metrics["switch_per_min"] - closed_metrics["switch_per_min"],
                    "delta_direction_switch_vs_closed": candidate_metrics["direction_switch_count"] - closed_metrics["direction_switch_count"],
                    "delta_sat_vs_closed": candidate_metrics["sat_ratio"] - closed_metrics["sat_ratio"],
                    "delta_capacity_headroom_vs_closed": candidate_metrics["capacity_headroom_min_ratio"] - closed_metrics["capacity_headroom_min_ratio"],
                    "selected_hold_ratio": choice_blob["selected_hold_ratio"],
                    "selected_active_ratio": choice_blob["selected_active_ratio"],
                    "selected_pump_saving_ratio": choice_blob["selected_pump_saving_ratio"],
                }
            )

    structure_summary_rows = aggregate_config_summary(
        structure_rows,
        ["config_name", "structure_profile_name", "split_role"],
    )

    # Scale sensitivity summary from phase 1.
    scale_summary_rows = aggregate_config_summary(
        weight_rows,
        ["config_name", "scale_mode", "split_role"],
    )

    actual_wall = time.perf_counter() - t0
    write_csv(paper_dir / "summary.csv", all_rows)
    write_csv(tables_dir / "baseline_scales.csv", baseline_rows)
    write_csv(tables_dir / "choice_distribution.csv", choice_rows)
    write_csv(tables_dir / "discrimination_summary.csv", discrimination_rows)
    write_csv(tables_dir / "tuning_summary_p95.csv", tuning_summaries)
    write_csv(tables_dir / "holdout_summary_p95.csv", holdout_summaries)
    write_csv(tables_dir / "tuning_summary_2sigma.csv", tuning_summaries_2sigma)
    write_csv(tables_dir / "structure_summary.csv", structure_summary_rows)
    write_csv(tables_dir / "scale_summary.csv", scale_summary_rows)
    write_csv(tables_dir / "weight_profiles.csv", [{"name": k, **v.to_dict()} for k, v in DEFAULT_WEIGHT_PROFILES.items()])
    write_csv(tables_dir / "structure_profiles.csv", [{"name": k, **v.to_dict()} for k, v in DEFAULT_STRUCTURE_PROFILES.items()])
    write_csv(tables_dir / "weight_configs.csv", [cfg.to_dict() for cfg in weight_configs])
    write_csv(tables_dir / "structure_configs.csv", [cfg.to_dict() for cfg in structure_configs])
    write_csv(tables_dir / "window_plan.csv", [
        {
            "window_label": w["window_label"],
            "split_role": w["split_role"],
            "start_timestamp": w["start_timestamp"].strftime("%Y-%m-%d %H:%M:%S"),
            "rank_in_label": w["rank_in_label"],
        }
        for w in windows
    ])
    write_csv(
        tables_dir / "hard_gate_thresholds.csv",
        [{"threshold_name": k, "value": v} for k, v in HARD_GATE_THRESHOLDS.items()],
    )
    (out_dir / "run_summary.md").write_text(
        build_summary_markdown(
            actual_wall_seconds=actual_wall,
            tuning_recommended_p95=str(recommended_p95["config_name"]),
            tuning_recommended_2sigma=str(recommended_2sigma["config_name"]),
            operational_default="balanced",
            holdout_row=holdout_for_recommended,
            discrimination_rows=discrimination_rows,
            structure_rows=structure_summary_rows,
            scale_rows=scale_summary_rows,
        ),
        encoding="utf-8",
    )
    print(f"Saved outputs under {out_dir}")


if __name__ == "__main__":
    main()
