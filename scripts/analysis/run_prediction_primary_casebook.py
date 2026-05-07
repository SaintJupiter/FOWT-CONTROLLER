#!/usr/bin/env python3
"""Prediction-primary casebook: 10 diverse 1h visual comparisons.

Each case runs closed_only and prediction_primary_econ for one hour and writes
a pitch/roll/pump-rate comparison figure. This is a short visual audit, not a
long validation campaign.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ModuleNotFoundError:  # pragma: no cover - optional for --skip-figures runs
    matplotlib = None
    plt = None
import numpy as np
import pandas as pd

repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root / "src"))
sys.path.insert(0, str(repo_root / "archive" / "legacy_fowt_control"))

from wind_prediction.ballast_planner import PlannerConfig
from wind_prediction.ballast_planner_provider import BallastPlannerPreviewProvider
from wind_prediction.forecast_adapter import ForecastModelAdapter, PersistenceMeanForecastAdapter
from wind_prediction.replay_dataset import Fino1ReplayDataset


DT = 1.0
N_STEPS = 3600
TIMESTAMP_FMT = "%Y-%m-%d %H:%M:%S"


# Frozen non-preview reactive baselines for closed_only. The default
# engineered_minimal profile is the closed_baseline_v1 reference: ordinary PI
# feedback plus pump actuator constraints, with no forecast or planner access.
CLOSED_BASELINE_V1_PUMP_CFG = {
    "pump_stop_err_kg": 300.0,
    "pump_restart_err_kg": 500.0,
    "pump_min_on_s": 20.0,
    "pump_min_off_s": 12.0,
    "pump_hold_before_stop_s": 10.0,
    "pump_global_quiet_hold_s": 0.0,
    "pump_global_quiet_backlog_kg": float("inf"),
    "pump_target_quiet_rate_kg_s": float("inf"),
    "pump_ramp_up_m3_min_per_s": 2.0,
    "pump_ramp_down_m3_min_per_s": 3.0,
}

CLOSED_ACTUATOR_SMOOTHING_CANDIDATE_PUMP_CFG = {
    **CLOSED_BASELINE_V1_PUMP_CFG,
    "pump_min_off_s": 20.0,
    "pump_hold_before_stop_s": 15.0,
    "pump_ramp_up_m3_min_per_s": 0.5,
    "pump_ramp_down_m3_min_per_s": 0.75,
}

CLOSED_RELAXED_ECONOMY_RAMP1P5_PUMP_CFG = {
    **CLOSED_BASELINE_V1_PUMP_CFG,
    "pump_stop_err_kg": 700.0,
    "pump_restart_err_kg": 1100.0,
    "pump_min_off_s": 20.0,
    "pump_hold_before_stop_s": 15.0,
    "pump_ramp_up_m3_min_per_s": 1.5,
    "pump_ramp_down_m3_min_per_s": 2.25,
}

CLOSED_SMOOTH_ECON700_PUMP_CFG = {
    **CLOSED_ACTUATOR_SMOOTHING_CANDIDATE_PUMP_CFG,
    "pump_stop_err_kg": 700.0,
    "pump_restart_err_kg": 1100.0,
}

CLOSED_PUMP_PROFILES = {
    "raw": None,
    "formal": CLOSED_BASELINE_V1_PUMP_CFG,
    "engineered_minimal": CLOSED_BASELINE_V1_PUMP_CFG,
    # Display-friendly closed profile. It reduces pump-rate total variation by
    # about 41% in the 5-case check, with <=3.28% pump-work increase and nearly
    # unchanged attitude metrics. The formal paper baseline remains
    # engineered_minimal unless explicitly changed.
    "actuator_smoothed_v2": CLOSED_ACTUATOR_SMOOTHING_CANDIDATE_PUMP_CFG,
    "smooth": CLOSED_ACTUATOR_SMOOTHING_CANDIDATE_PUMP_CFG,
    # Supplemental non-preview references for baseline-fairness checks.
    "relaxed_economy_ramp1p5": CLOSED_RELAXED_ECONOMY_RAMP1P5_PUMP_CFG,
    "smooth_econ700": CLOSED_SMOOTH_ECON700_PUMP_CFG,
}

CLOSED_TARGET_SHAPE_PROFILES = {
    "raw": {},
    "formal": {},
    "engineered_minimal": {},
    "actuator_smoothed_v2": {},
    "smooth": {},
    "relaxed_economy_ramp1p5": {},
    "smooth_econ700": {},
}

PRIMARY_SAFETY_PROFILES = {
    "default": {},
    "bucket_guard": {
        "primary_safety_bucket_guard_enabled": True,
        "primary_safety_bucket_guard_bucket_s": 600.0,
        "primary_safety_bucket_guard_pitch_enter_deg": 4.0,
        "primary_safety_bucket_guard_roll_enter_deg": 4.0,
        "primary_safety_bucket_guard_pitch_exit_deg": 3.5,
        "primary_safety_bucket_guard_roll_exit_deg": 3.5,
        "primary_safety_bucket_guard_improve_tol_deg": 0.0,
        "primary_safety_bucket_guard_exit_required_windows": 1,
        "primary_safety_bucket_guard_max_active_windows": 1,
        "primary_safety_bucket_guard_hold_only": False,
        "primary_safety_bucket_guard_require_current_high": False,
    },
    "risk_gate": {
        "primary_safety_pitch_enter_deg": 4.0,
        "primary_safety_roll_enter_deg": 4.0,
        "primary_safety_pitch_exit_deg": 3.5,
        "primary_safety_roll_exit_deg": 3.5,
        "primary_safety_enter_hold_s": 0.0,
        "primary_safety_exit_hold_s": 0.0,
        "primary_safety_exit_required_windows": 1,
        "primary_safety_emergency_pitch_enter_deg": 8.0,
        "primary_safety_emergency_roll_enter_deg": 6.5,
        "primary_safety_bucket_guard_enabled": True,
        "primary_safety_bucket_guard_bucket_s": 600.0,
        "primary_safety_bucket_guard_pitch_enter_deg": 4.0,
        "primary_safety_bucket_guard_roll_enter_deg": 4.0,
        "primary_safety_bucket_guard_pitch_exit_deg": 3.5,
        "primary_safety_bucket_guard_roll_exit_deg": 3.5,
        "primary_safety_bucket_guard_improve_tol_deg": 0.0,
        "primary_safety_bucket_guard_exit_required_windows": 1,
        "primary_safety_bucket_guard_max_active_windows": 1,
        "primary_safety_bucket_guard_hold_only": False,
        "primary_safety_bucket_guard_require_current_high": False,
    },
    "hold_risk_gate": {
        "primary_safety_pitch_enter_deg": 4.0,
        "primary_safety_roll_enter_deg": 4.0,
        "primary_safety_pitch_exit_deg": 3.5,
        "primary_safety_roll_exit_deg": 3.5,
        "primary_safety_enter_hold_s": 0.0,
        "primary_safety_exit_hold_s": 0.0,
        "primary_safety_exit_required_windows": 1,
        "primary_safety_emergency_pitch_enter_deg": 8.0,
        "primary_safety_emergency_roll_enter_deg": 6.5,
        "primary_safety_hold_risk_gate_enabled": True,
        "primary_safety_hold_risk_gate_action": "hold",
        "primary_safety_bucket_guard_enabled": True,
        "primary_safety_bucket_guard_bucket_s": 600.0,
        "primary_safety_bucket_guard_pitch_enter_deg": 4.0,
        "primary_safety_bucket_guard_roll_enter_deg": 4.0,
        "primary_safety_bucket_guard_pitch_exit_deg": 3.5,
        "primary_safety_bucket_guard_roll_exit_deg": 3.5,
        "primary_safety_bucket_guard_improve_tol_deg": 0.0,
        "primary_safety_bucket_guard_exit_required_windows": 1,
        "primary_safety_bucket_guard_max_active_windows": 1,
        "primary_safety_bucket_guard_hold_only": True,
        "primary_safety_bucket_guard_require_current_high": False,
    },
    "strict": {
        "primary_safety_pitch_enter_deg": 5.5,
        "primary_safety_roll_enter_deg": 4.5,
        "primary_safety_pitch_exit_deg": 4.8,
        "primary_safety_roll_exit_deg": 3.8,
        "primary_safety_enter_hold_s": 10.0,
        "primary_safety_emergency_pitch_enter_deg": 7.5,
        "primary_safety_emergency_roll_enter_deg": 6.5,
    },
}


CASES = [
    ("01_onset_strong", "2024-11-27 19:40:00", "Strong onset: calm now, very strong future wind"),
    ("02_onset_signflip", "2023-10-31 06:20:00", "Onset with direction flip"),
    ("03_onset_moderate", "2024-10-10 03:50:00", "Moderate onset"),
    ("04_decay_strong", "2024-09-27 13:00:00", "Strong decay: high now, weak future wind"),
    ("05_decay_signflip", "2022-03-20 19:00:00", "Decay with direction flip"),
    ("06_signflip_high", "2023-10-03 06:30:00", "High-pressure sign-flip"),
    ("07_signflip_sustained", "2023-03-14 04:40:00", "Sustained high sign-flip"),
    ("08_lowrisk_quiet", "2021-12-20 14:30:00", "Low-risk quiet window"),
    ("09_high_pressure_event", "2022-02-04 11:00:00", "High-pressure high-event normal window"),
    ("10_residual_high", "2024-09-05 18:10:00", "Residual-high normal window"),
    # Fallbacks in case a timestamp is unavailable in the replay dataset.
    ("11_clean_low_pressure", "2023-09-11 03:40:00", "Clean low-pressure normal window"),
    ("12_onset_late", "2024-12-06 08:40:00", "Late onset fallback"),
]


def discover_excel() -> str:
    for d in (repo_root / "data", repo_root / "archive" / "legacy_fowt_control" / "data"):
        for x in d.rglob("*.xlsx"):
            if "stiffness" in x.name.lower() or "matrix" in x.name.lower():
                return str(x)
    raise FileNotFoundError("stiffness/matrix xlsx not found")


def pct_abs(x: np.ndarray, q: float) -> float:
    return float(np.percentile(np.abs(np.asarray(x, dtype=float)), q))


def work_m3(pump: np.ndarray) -> float:
    return float(np.trapezoid(np.abs(np.asarray(pump, dtype=float)), dx=DT) / 60.0)


def closed_pump_cfg(profile: str) -> dict | None:
    cfg = CLOSED_PUMP_PROFILES[str(profile)]
    return None if cfg is None else dict(cfg)


def closed_target_shape_override(profile: str) -> dict:
    return dict(CLOSED_TARGET_SHAPE_PROFILES[str(profile)])


def primary_safety_override(profile: str) -> dict:
    return dict(PRIMARY_SAFETY_PROFILES[str(profile)])


def smooth_wind_trace(trace: dict, transition_s: float) -> dict:
    transition_s = float(transition_s)
    if transition_s <= 0.0:
        return trace
    dt = float(trace.get("dt", DT))
    update_s = float(trace.get("update_interval_s", 0.0))
    if update_s <= 0.0:
        return trace
    hold_steps = int(round(update_s / dt))
    if hold_steps <= 1:
        return trace
    trans_steps = min(max(1, int(round(transition_s / dt))), hold_steps)
    ws = np.asarray(trace["ws"], dtype=float).copy()
    wd = np.asarray(trace["wd"], dtype=float).copy()
    n_bucket = int(np.ceil(len(ws) / hold_steps))
    for k in range(1, n_bucket):
        start = k * hold_steps
        if start >= len(ws):
            break
        end = min(start + trans_steps, len(ws))
        prev_ws = float(ws[start - 1])
        prev_wd = float(wd[start - 1])
        curr_ws = float(ws[start])
        curr_wd = float(wd[start])
        wd_delta = ((curr_wd - prev_wd + 180.0) % 360.0) - 180.0
        denom = max(end - start - 1, 1)
        for i, idx in enumerate(range(start, end)):
            frac = 0.5 * (1.0 - np.cos(np.pi * i / denom))
            ws[idx] = prev_ws + frac * (curr_ws - prev_ws)
            wd[idx] = (prev_wd + frac * wd_delta) % 360.0
    out = dict(trace)
    out["ws"] = ws
    out["wd"] = wd
    out["mean_lpf_tau_s"] = float(transition_s)
    out["transition_shape"] = "raised_cosine"
    return out


def summarize(df: pd.DataFrame) -> dict:
    pitch = df["pitch_deg"].to_numpy(dtype=float)
    roll = df["roll_deg"].to_numpy(dtype=float)
    pump = df["pump_total_rate_m3_min"].to_numpy(dtype=float)
    zeros = pd.Series(np.zeros(len(df)))
    return {
        "pitch_mean": float(np.mean(pitch)),
        "pitch_abs_p50": pct_abs(pitch, 50),
        "pitch_abs_p95": pct_abs(pitch, 95),
        "pitch_abs_max": pct_abs(pitch, 100),
        "roll_mean": float(np.mean(roll)),
        "roll_abs_p50": pct_abs(roll, 50),
        "roll_abs_p95": pct_abs(roll, 95),
        "roll_abs_max": pct_abs(roll, 100),
        "pump_work_m3": work_m3(pump),
        "pump_duty": float(np.mean(np.abs(pump) > 1e-6)),
        "latch_switches": int(df["pump_latch_switch_count"].iloc[-1])
        if "pump_latch_switch_count" in df.columns
        else 0,
        "primary_delta_mean_kg": float(
            np.mean(np.abs(df.get("preview_primary_delta_mean_kg", zeros).to_numpy(dtype=float)))
        ),
        "primary_candidate_ratio": float(
            np.mean(df.get("preview_primary_candidate_applied", zeros).to_numpy(dtype=float))
        ),
        "primary_applied_ratio": float(
            np.mean(df.get("preview_primary_applied", zeros).to_numpy(dtype=float))
        ),
        "primary_safety_fallback_ratio": float(
            np.mean(df.get("preview_primary_safety_fallback", zeros).to_numpy(dtype=float))
        ),
        "primary_safety_active_ratio": float(
            np.mean(df.get("preview_primary_safety_active", zeros).to_numpy(dtype=float))
        ),
        "primary_safety_transition_count": int(
            np.sum(
                np.abs(
                    np.diff(
                        df.get("preview_primary_safety_active", zeros).to_numpy(dtype=float),
                        prepend=0.0,
                    )
                )
                > 0.5
            )
        ),
        "primary_bucket_guard_active_ratio": float(
            np.mean(df.get("preview_primary_bucket_guard_active", zeros).to_numpy(dtype=float))
        ),
        "primary_bucket_guard_transition_count": int(
            np.sum(
                np.abs(
                    np.diff(
                        df.get("preview_primary_bucket_guard_active", zeros).to_numpy(dtype=float),
                        prepend=0.0,
                    )
                )
                > 0.5
            )
        ),
        "preview_pump_suppression_ratio": float(
            np.mean(df.get("preview_pump_suppression_active", zeros).to_numpy(dtype=float))
        ),
        "preview_suppression_delta_mean_kg": float(
            np.mean(df.get("preview_suppression_delta_mean_kg", zeros).to_numpy(dtype=float))
        ),
        "preview_event_risk_boost_ratio": float(
            np.mean(df.get("preview_event_risk_pressure_boost_enabled", zeros).to_numpy(dtype=float))
        ),
        "preview_event_risk_scale_mean": float(
            np.mean(
                df.get(
                    "preview_event_risk_scale_0_20m",
                    pd.Series(np.ones(len(df)), index=df.index),
                ).to_numpy(dtype=float)
            )
        ),
        "preview_event_risk_floor_ratio": float(
            np.mean(
                np.max(
                    np.column_stack(
                        [
                            df.get("preview_event_risk_floor_active_0_20m", zeros).to_numpy(dtype=float),
                            df.get("preview_event_risk_floor_active_20_40m", zeros).to_numpy(dtype=float),
                            df.get("preview_event_risk_floor_active_40_60m", zeros).to_numpy(dtype=float),
                        ]
                    ),
                    axis=1,
                )
            )
        ),
        "preview_event_risk_floor_norm_mean": float(
            np.mean(
                np.column_stack(
                    [
                        df.get("preview_event_risk_floor_norm_0_20m", zeros).to_numpy(dtype=float),
                        df.get("preview_event_risk_floor_norm_20_40m", zeros).to_numpy(dtype=float),
                        df.get("preview_event_risk_floor_norm_40_60m", zeros).to_numpy(dtype=float),
                    ]
                )
            )
        ),
        "preview_lead_action_ratio": float(
            np.mean(df.get("preview_lead_action_active", zeros).to_numpy(dtype=float))
        ),
        "preview_relief_medium_cap_ratio": float(
            np.mean(df.get("preview_relief_medium_cap_active", zeros).to_numpy(dtype=float))
        ),
        "primary_target_resumed_ratio": float(
            np.mean(df.get("preview_primary_target_resumed", zeros).to_numpy(dtype=float))
        ),
    }


def _planner_short_action(action: str) -> str:
    mapping = {
        "hold": "hold",
        "pump_saving": "save",
        "active_small": "small",
        "active_medium": "medium",
        "active_reverse_small": "reverse",
    }
    return mapping.get(str(action), str(action))


def _format_plan_note(row: pd.Series) -> str:
    action = _planner_short_action(str(row.get("first_action", "hold")))
    tp = float(row.get("target_pitch_deg", 0.0))
    tr = float(row.get("target_roll_deg", 0.0))
    b0 = float(row.get("pressure_block0_norm", 0.0))
    b1 = float(row.get("pressure_block1_norm", 0.0))
    b2 = float(row.get("pressure_block2_norm", 0.0))
    reused = int(float(row.get("prediction_primary_target_reused", 0.0))) if "prediction_primary_target_reused" in row else 0
    refresh = "reuse" if reused else "new"
    if action == "hold":
        return f"hold\nfuture {b0:.2f}/{b1:.2f}/{b2:.2f}"
    return (
        f"{action} {refresh}\n"
        f"target {tp:+.2f}/{tr:+.2f} deg\n"
        f"future {b0:.2f}/{b1:.2f}/{b2:.2f}"
    )


def _circular_mean_deg(values: np.ndarray) -> float:
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return float("nan")
    rad = np.deg2rad(vals)
    mean_sin = float(np.mean(np.sin(rad)))
    mean_cos = float(np.mean(np.cos(rad)))
    return float((np.rad2deg(np.arctan2(mean_sin, mean_cos)) + 360.0) % 360.0)


def _format_prediction_role(row: pd.Series | None) -> str:
    if row is None:
        return "Pred: n/a"
    action = str(row.get("first_action", "hold"))
    short = _planner_short_action(action)
    role_map = {
        "hold": "hold",
        "save": "pump-save",
        "small": "pre-small",
        "medium": "pre-med",
        "reverse": "release",
    }
    role = role_map.get(short, short)
    b0 = float(row.get("pressure_block0_norm", 0.0))
    b1 = float(row.get("pressure_block1_norm", 0.0))
    b2 = float(row.get("pressure_block2_norm", 0.0))
    dkg = float(row.get("prediction_primary_delta_abs_mean_kg", 0.0))
    refreshed = int(float(row.get("prediction_primary_target_refreshed", 0.0))) > 0
    reused = int(float(row.get("prediction_primary_target_reused", 0.0))) > 0
    mode = "new" if refreshed or not reused else "reuse"
    if role == "hold" or dkg <= 1.0:
        return f"Pred: hold\nrisk {b0:.2f}/{b1:.2f}/{b2:.2f}"
    return f"Pred: {role}\n{mode}, dM {dkg / 1000.0:.0f}t"


def _planner_row_for_interval(
    planner_log: pd.DataFrame | None,
    start_s: float,
    end_s: float,
) -> pd.Series | None:
    if planner_log is None or planner_log.empty or "current_time_s" not in planner_log.columns:
        return None
    times = planner_log["current_time_s"].to_numpy(dtype=float)
    mask = (times >= float(start_s) - 1e-6) & (times < float(end_s) - 1e-6)
    if np.any(mask):
        return planner_log.iloc[int(np.flatnonzero(mask)[0])]
    before = np.flatnonzero(times <= float(start_s) + 1e-6)
    if before.size > 0:
        return planner_log.iloc[int(before[-1])]
    return planner_log.iloc[0]


def _same_wind_block(a: dict, b: dict, ws_tol: float = 0.05, wd_tol: float = 0.5) -> bool:
    if not (np.isfinite(a["ws"]) and np.isfinite(b["ws"]) and np.isfinite(a["wd"]) and np.isfinite(b["wd"])):
        return False
    wd_delta = abs(((float(a["wd"]) - float(b["wd"]) + 180.0) % 360.0) - 180.0)
    return abs(float(a["ws"]) - float(b["ws"])) <= float(ws_tol) and wd_delta <= float(wd_tol)


def _annotate_wind_prediction_strip(
    ax,
    wind_df: pd.DataFrame,
    planner_log: pd.DataFrame | None,
    duration_min: float,
    block_min: float = 10.0,
) -> None:
    ax.set_ylim(0.0, 1.0)
    ax.set_yticks([])
    ax.set_ylabel("Wind\nPred", rotation=0, ha="right", va="center", labelpad=28)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color("#bdbdbd")
    ax.grid(False)

    t = wind_df["t_s"].to_numpy(dtype=float) / 60.0
    ws = wind_df.get("wind_speed", pd.Series(np.zeros(len(wind_df)))).to_numpy(dtype=float)
    wd = wind_df.get("wind_dir_deg", pd.Series(np.zeros(len(wind_df)))).to_numpy(dtype=float)
    cmap = plt.get_cmap("Blues")
    n_blocks = int(np.ceil(float(duration_min) / float(block_min)))
    blocks: list[dict] = []
    for i in range(n_blocks):
        start = i * float(block_min)
        end = min((i + 1) * float(block_min), float(duration_min))
        if end <= start:
            continue
        mask = (t >= start - 1e-9) & (t < end - 1e-9)
        if not np.any(mask):
            mask = np.abs(t - start) <= 0.5
        ws_val = float(np.nanmean(ws[mask])) if np.any(mask) else float("nan")
        wd_val = _circular_mean_deg(wd[mask]) if np.any(mask) else float("nan")
        blocks.append(
            {
                "start": start,
                "end": end,
                "mid": 0.5 * (start + end),
                "ws": ws_val,
                "wd": wd_val,
                "planner_row": _planner_row_for_interval(planner_log, start * 60.0, end * 60.0),
            }
        )

    groups: list[dict] = []
    for block in blocks:
        if groups and _same_wind_block(groups[-1], block):
            groups[-1]["end"] = block["end"]
        else:
            groups.append({"start": block["start"], "end": block["end"], "ws": block["ws"], "wd": block["wd"]})

    for group in groups:
        speed_norm = 0.0 if not np.isfinite(group["ws"]) else float(np.clip(group["ws"] / 25.0, 0.0, 1.0))
        face = cmap(0.20 + 0.70 * speed_norm)
        ax.axvspan(group["start"], group["end"], facecolor=face, alpha=0.72, lw=0.0)
        ax.axvline(group["start"], color="#ffffff", lw=0.9, alpha=0.85)
        group_mid = 0.5 * (group["start"] + group["end"])
        wind_label = (
            f"WS {group['ws']:.1f} m/s  WD {group['wd']:.0f} deg"
            if np.isfinite(group["ws"]) and np.isfinite(group["wd"])
            else "WS n/a  WD n/a"
        )
        ax.text(group_mid, 0.76, wind_label, ha="center", va="center", fontsize=7.0, color="#0f2438")

    for block in blocks:
        ax.text(
            block["mid"],
            0.25,
            _format_prediction_role(block["planner_row"]),
            ha="center",
            va="center",
            fontsize=6.4,
            color="#2d2d2d",
            linespacing=1.12,
        )
    ax.axvline(float(duration_min), color="#ffffff", lw=0.8, alpha=0.75)
    ax.text(
        0.0,
        1.06,
        "10-min replay blocks: current wind and prediction-primary role",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.5,
        color="#444444",
    )


def _annotate_planner(ax, planner_log: pd.DataFrame | None, duration_min: float) -> None:
    if planner_log is None or planner_log.empty:
        return
    y0, y1 = ax.get_ylim()
    span = max(y1 - y0, 1e-6)
    y = y1 + 0.045 * span
    for _, row in planner_log.iterrows():
        start_min = float(row.get("current_time_s", 0.0)) / 60.0
        if start_min >= duration_min:
            continue
        interval_min = 10.0
        if "current_time_s" in planner_log.columns and len(planner_log) > 1:
            times = np.sort(planner_log["current_time_s"].to_numpy(dtype=float))
            diffs = np.diff(times)
            diffs = diffs[diffs > 1e-6]
            if diffs.size > 0:
                interval_min = float(np.median(diffs) / 60.0)
        ax.text(
            min(start_min + 0.5 * interval_min, duration_min - 1.0),
            y,
            _format_plan_note(row),
            ha="center",
            va="bottom",
            fontsize=7.0,
            color="#3f3f3f",
            linespacing=1.1,
            clip_on=False,
        )
    ax.set_ylim(y0, y1 + 0.24 * span)


def plot_case(
    case_id: str,
    label: str,
    ts_str: str,
    closed: pd.DataFrame,
    primary: pd.DataFrame,
    out_dir: Path,
    primary_label: str,
    duration_min: float,
    planner_log: pd.DataFrame | None = None,
) -> Path:
    if plt is None:
        raise ModuleNotFoundError(
            "matplotlib is required for plotting; rerun without --skip-figures only in an environment that has matplotlib"
        )
    t_closed = closed["t_s"].to_numpy(dtype=float) / 60.0
    t_primary = primary["t_s"].to_numpy(dtype=float) / 60.0
    c = "#2468B2"
    p = "#D85C27"
    fig_w = max(12.5, min(24.0, 12.5 * duration_min / 60.0))
    fig, axes = plt.subplots(
        4,
        1,
        figsize=(fig_w, 9.8),
        sharex=True,
        constrained_layout=True,
        gridspec_kw={"height_ratios": [0.78, 1.0, 1.0, 1.0]},
    )
    strip_ax = axes[0]
    series_axes = axes[1:]
    wind_source = primary if {"wind_speed", "wind_dir_deg"}.issubset(primary.columns) else closed
    _annotate_wind_prediction_strip(strip_ax, wind_source, planner_log, duration_min)
    rows = [
        ("Pitch angle (deg)", "pitch_deg"),
        ("Roll angle (deg)", "roll_deg"),
        ("Total pump rate (m3/min)", "pump_total_rate_m3_min"),
    ]
    for ax, (ylabel, col) in zip(series_axes, rows):
        ax.plot(t_closed, closed[col].to_numpy(dtype=float), color=c, lw=1.45, label="closed_only")
        ax.plot(t_primary, primary[col].to_numpy(dtype=float), color=p, lw=1.45, label=primary_label)
        ax.axhline(0.0, color="#777777", lw=0.8, alpha=0.55)
        for x in np.arange(20.0, duration_min, 20.0):
            ax.axvline(x, color="#999999", lw=0.9, ls="--", alpha=0.55)
        ax.set_ylabel(ylabel)
        ax.grid(True, color="#d9d9d9", lw=0.7, alpha=0.75)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    series_axes[0].legend(loc="upper left", frameon=False, ncol=2)
    series_axes[2].set_xlabel("Time (min)")
    series_axes[2].set_xlim(0, duration_min)
    sc = summarize(closed)
    sp = summarize(primary)
    metric_text = (
        f"pump: {sc['pump_work_m3']:.1f} -> {sp['pump_work_m3']:.1f} m3 "
        f"({(sp['pump_work_m3'] - sc['pump_work_m3']) / max(sc['pump_work_m3'], 1e-9) * 100:+.1f}%)\n"
        f"pitch p95: {sc['pitch_abs_p95']:.2f} -> {sp['pitch_abs_p95']:.2f} deg; "
        f"roll p95: {sc['roll_abs_p95']:.2f} -> {sp['roll_abs_p95']:.2f} deg"
    )
    series_axes[0].text(
        0.995,
        0.03,
        metric_text,
        transform=series_axes[0].transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="#cccccc", alpha=0.92),
    )
    fig.suptitle(f"{case_id}: {label} ({ts_str})", fontsize=13, y=1.01)
    out_path = out_dir / f"{case_id}_closed_vs_primary.png"
    fig.savefig(out_path, dpi=165, bbox_inches="tight")
    plt.close(fig)
    return out_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        default="outputs/wind_prediction/prediction_primary_casebook",
        help="Output directory for figures, timeseries, planner logs, and report.",
    )
    parser.add_argument(
        "--primary-label",
        default="prediction_primary_econ",
        help="Label/name for the prediction-primary variant in outputs.",
    )
    parser.add_argument(
        "--duration-s",
        type=float,
        default=float(N_STEPS * DT),
        help="Simulation duration per case in seconds.",
    )
    parser.add_argument(
        "--case-ids",
        default="",
        help="Comma-separated canonical case ids to run, e.g. 01,04,05,09,10. Empty means first 10 cases.",
    )
    parser.add_argument(
        "--cases-csv",
        default="",
        help="Optional CSV with case_id,timestamp,label columns. Overrides the built-in case list.",
    )
    parser.add_argument(
        "--skip-figures",
        action="store_true",
        help=(
            "Skip PNG plotting during batch runs. Raw 1Hz timeseries, planner logs, "
            "summary CSV, and report are still written."
        ),
    )
    parser.add_argument(
        "--primary-only",
        action="store_true",
        help=(
            "Run only the prediction-primary variant. Useful for learned/persistence/oracle "
            "screening where the closed baseline would be identical and can be added later."
        ),
    )
    parser.add_argument(
        "--wind-transition-s",
        type=float,
        default=0.0,
        help="Raised-cosine transition duration for replay wind bucket changes. 0 keeps hard 10-min steps.",
    )
    parser.add_argument(
        "--closed-pump-profile",
        choices=sorted(CLOSED_PUMP_PROFILES),
        default="engineered_minimal",
        help="Pump-side profile for closed_only only. Default is the frozen non-preview baseline.",
    )
    parser.add_argument(
        "--primary-scale",
        type=float,
        default=1.0,
        help="Mass-target scale for prediction_primary only.",
    )
    parser.add_argument(
        "--primary-control-profile",
        choices=(
            "manual",
            "rawenv_holdpause_barrier_v1",
            "rawenv_holdpause_barrier_reliefcap030_v1",
        ),
        default="manual",
        help=(
            "Convenience profile for prediction-primary planner/execution switches. "
            "manual preserves explicit CLI flags; rawenv_holdpause_barrier_v1 applies "
            "the current guarded value profile: raw envelope, hold target pause, and "
            "the envelope barrier. rawenv_holdpause_barrier_reliefcap030_v1 additionally "
            "enables the event-risk-conditioned medium-action cap at ratio 0.30."
        ),
    )
    parser.add_argument(
        "--primary-safety-profile",
        choices=sorted(PRIMARY_SAFETY_PROFILES),
        default="default",
        help="Runtime fallback profile for prediction_primary only.",
    )
    parser.add_argument(
        "--forecast-source",
        choices=("oracle", "persistence_mean", "learned"),
        default="oracle",
        help="Forecast source for prediction_primary. Oracle uses replay future y_uv_raw.",
    )
    parser.add_argument(
        "--model-dir",
        default="outputs/wind_prediction/lstm_segmented_fino1_meteo_aux_v1",
        help="Trained forecast model directory used when --forecast-source=learned.",
    )
    parser.add_argument(
        "--dataset-dir",
        default="data/processed/wind_ml_10min/ballast_decision_fino1_meteo_aux_v1",
        help="Replay dataset directory.",
    )
    parser.add_argument(
        "--persistence-history-min",
        type=float,
        default=10.0,
        help="History mean window for --forecast-source=persistence_mean.",
    )
    parser.add_argument(
        "--event-reset-mode",
        choices=("action", "target_change", "active_bucket"),
        default="action",
        help="Prediction-primary target refresh policy. Default preserves current behavior.",
    )
    parser.add_argument(
        "--planner-envelope-raw",
        action="store_true",
        help=(
            "Evaluate planner envelope costs on raw undiscounted pressure blocks. "
            "Default preserves discounted-envelope behavior."
        ),
    )
    parser.add_argument(
        "--planner-envelope-barrier",
        action="store_true",
        help=(
            "Enable the existing zone-MPC envelope barrier: if a sequence violates "
            "the envelope and uses only weak actions, add a fixed scalar cost."
        ),
    )
    parser.add_argument(
        "--planner-envelope-barrier-const",
        type=float,
        default=50.0,
        help="Fixed scalar cost used by --planner-envelope-barrier.",
    )
    parser.add_argument(
        "--planner-pressure-norm-cap",
        type=float,
        default=1.5,
        help=(
            "Deadband-normalized cap for pressure_proxy_vec. Default 1.5 preserves "
            "the current planner; larger values test whether high-wind saturation "
            "is compressing learned/persistence forecast differences."
        ),
    )
    parser.add_argument(
        "--target-reset-tol-deg",
        type=float,
        default=0.02,
        help="Planner-frame target-change tolerance for --event-reset-mode=target_change.",
    )
    parser.add_argument(
        "--primary-hold-target-mode",
        choices=("current", "pause"),
        default="current",
        help=(
            "Prediction-primary hold target handling. current preserves existing behavior; "
            "pause temporarily anchors hold at current masses but can resume the prior "
            "same-direction active target instead of re-issuing a fresh target."
        ),
    )
    parser.add_argument(
        "--pump-suppression",
        action="store_true",
        help="Enable forecast-based preview pump suppression in prediction-primary.",
    )
    parser.add_argument(
        "--pump-suppression-restart-err-kg",
        type=float,
        default=2000.0,
        help="Restart-error threshold for forecast-based preview pump suppression.",
    )
    parser.add_argument(
        "--pump-suppression-relief-margin-norm",
        type=float,
        default=0.25,
        help="Required block0-to-block2 relief margin for preview pump suppression.",
    )
    parser.add_argument(
        "--pump-suppression-low-risk-norm",
        type=float,
        default=0.75,
        help="All-block low-risk threshold for preview pump suppression.",
    )
    parser.add_argument(
        "--event-risk-pressure-boost",
        action="store_true",
        help="Enable event-probability pressure boost in prediction-primary.",
    )
    parser.add_argument(
        "--event-risk-pressure-boost-gain",
        type=float,
        default=0.35,
        help="Event-risk pressure boost gain above the probability threshold.",
    )
    parser.add_argument(
        "--event-risk-pressure-boost-threshold",
        type=float,
        default=0.50,
        help="Event probability threshold before pressure boost starts.",
    )
    parser.add_argument(
        "--event-risk-pressure-boost-max",
        type=float,
        default=0.35,
        help="Maximum per-block pressure boost from event risk.",
    )
    parser.add_argument(
        "--event-risk-pressure-floor",
        action="store_true",
        help="Raise low learned/oracle pressure-block norms when segment event risk is high.",
    )
    parser.add_argument(
        "--event-risk-pressure-floor-threshold",
        type=float,
        default=0.70,
        help="Segment event probability threshold before the pressure floor can activate.",
    )
    parser.add_argument(
        "--event-risk-pressure-floor-norm",
        type=float,
        default=0.85,
        help="Target minimum deadband-normalized pressure norm when the event-risk floor activates.",
    )
    parser.add_argument(
        "--event-risk-pressure-floor-max-lift",
        type=float,
        default=0.50,
        help="Maximum per-block pressure-norm lift applied by the event-risk floor.",
    )
    parser.add_argument(
        "--preview-lead-action",
        action="store_true",
        help="Enable forecast-only prepump when low current pressure precedes high future event pressure.",
    )
    parser.add_argument(
        "--preview-lead-event-threshold",
        type=float,
        default=0.90,
        help="Segment event probability required for forecast lead-action.",
    )
    parser.add_argument(
        "--preview-lead-current-norm-max",
        type=float,
        default=0.65,
        help="Maximum current raw pressure norm that still qualifies as a low-pressure lead window.",
    )
    parser.add_argument(
        "--preview-lead-future-norm-min",
        type=float,
        default=1.00,
        help="Minimum future pressure norm required for forecast lead-action.",
    )
    parser.add_argument(
        "--preview-lead-current-wind-max-ms",
        type=float,
        default=8.0,
        help="Maximum current wind-speed proxy for forecast lead-action.",
    )
    parser.add_argument(
        "--preview-lead-pitch-abs-max-deg",
        type=float,
        default=2.0,
        help="Maximum current absolute pitch for forecast lead-action.",
    )
    parser.add_argument(
        "--preview-lead-roll-abs-max-deg",
        type=float,
        default=1.5,
        help="Maximum current absolute roll for forecast lead-action.",
    )
    parser.add_argument(
        "--preview-lead-action-name",
        choices=("pump_saving", "active_small"),
        default="active_small",
        help="Planner action used for a forecast lead-action.",
    )
    parser.add_argument(
        "--pump-suppression-event-risk-guard",
        action="store_true",
        help="Block preview pump suppression when learned/oracle event risk is high.",
    )
    parser.add_argument(
        "--pump-suppression-event-risk-guard-threshold",
        type=float,
        default=0.70,
        help="Maximum segment event probability that still permits preview pump suppression.",
    )
    parser.add_argument(
        "--relief-medium-cap",
        action="store_true",
        help="Cap active_medium to active_small in high-confidence future-relief windows.",
    )
    parser.add_argument(
        "--relief-medium-cap-event-threshold",
        type=float,
        default=0.70,
        help="Minimum segment event probability required for relief medium cap.",
    )
    parser.add_argument(
        "--relief-medium-cap-ratio",
        type=float,
        default=0.25,
        help="Action ratio used when capping active_medium in future-relief windows.",
    )
    return parser.parse_args()


def _load_indexed_cases(cases_csv: str) -> list[tuple[int, str, str, str]]:
    if not cases_csv:
        return [
            (idx, raw_case_id, ts_str, label)
            for idx, (raw_case_id, ts_str, label) in enumerate(CASES[:10], start=1)
        ]
    path = Path(cases_csv)
    if not path.is_absolute():
        path = repo_root / path
    df = pd.read_csv(path)
    required = {"timestamp"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"--cases-csv missing required columns: {sorted(missing)}")
    rows: list[tuple[int, str, str, str]] = []
    for idx, row in enumerate(df.to_dict("records"), start=1):
        raw_case_id = str(row.get("case_id", "") or row.get("group", "") or f"case_{idx:02d}")
        label = str(row.get("label", "") or row.get("selection_group", "") or raw_case_id)
        rows.append((idx, raw_case_id, str(row["timestamp"]), label))
    return rows


def _numbered_case_id(case_num: int, raw_case_id: str) -> str:
    clean = str(raw_case_id).strip() or f"case_{case_num:02d}"
    if len(clean) > 3 and clean[:2].isdigit() and clean[2] == "_":
        clean = clean[3:]
    elif "_" in clean and clean.split("_", 1)[0].isdigit():
        clean = clean.split("_", 1)[1]
    clean = "".join(ch if ch.isalnum() or ch in ("_", "-") else "_" for ch in clean)
    return f"{case_num:02d}_{clean}"


def _make_forecast_adapter(args: argparse.Namespace):
    source = str(args.forecast_source)
    dataset_dir = Path(args.dataset_dir)
    if not dataset_dir.is_absolute():
        dataset_dir = repo_root / dataset_dir
    if source == "oracle":
        return None
    if source == "persistence_mean":
        return PersistenceMeanForecastAdapter(
            dataset_dir=dataset_dir,
            history_minutes=float(args.persistence_history_min),
        )
    if source == "learned":
        model_dir = Path(args.model_dir)
        if not model_dir.is_absolute():
            model_dir = repo_root / model_dir
        return ForecastModelAdapter(
            model_dir=model_dir,
            dataset_dir=dataset_dir,
            device="cpu",
        )
    raise ValueError(f"unsupported forecast source: {source}")


def main() -> None:
    args = parse_args()
    primary_control_profile = str(args.primary_control_profile)
    if primary_control_profile in {
        "rawenv_holdpause_barrier_v1",
        "rawenv_holdpause_barrier_reliefcap030_v1",
    }:
        args.planner_envelope_raw = True
        args.planner_envelope_barrier = True
        args.planner_envelope_barrier_const = 50.0
        args.primary_hold_target_mode = "pause"
    if primary_control_profile == "rawenv_holdpause_barrier_reliefcap030_v1":
        args.relief_medium_cap = True
        args.relief_medium_cap_ratio = 0.30
        args.relief_medium_cap_event_threshold = 0.70
    t0 = time.perf_counter()
    n_steps = int(round(float(args.duration_s) / DT))
    if n_steps <= 0:
        raise ValueError("--duration-s must be positive")
    duration_min = float(n_steps * DT / 60.0)
    selected_ids = {s.strip() for s in str(args.case_ids).split(",") if s.strip()}
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = repo_root / out_dir
    primary_variant = str(args.primary_label)
    closed_profile = str(args.closed_pump_profile)
    closed_profile_cfg = closed_pump_cfg(closed_profile)
    closed_target_shape_cfg = closed_target_shape_override(closed_profile)
    primary_target_shape_cfg = primary_safety_override(str(args.primary_safety_profile))
    primary_scale = float(args.primary_scale)
    fig_dir = out_dir / "figures"
    ts_dir = out_dir / "timeseries"
    log_dir = out_dir / "planner_logs"
    for d in (fig_dir, ts_dir, log_dir):
        d.mkdir(parents=True, exist_ok=True)

    base_planner_out = repo_root / "outputs" / "wind_prediction" / "planner_a1_dryrun"
    sign_cfg = json.loads((base_planner_out / "diagnostics" / "a01_pressure_vec_sign_convention.json").read_text())
    discount_cfg = json.loads((base_planner_out / "diagnostics" / "a1_block_discount_config.json").read_text())
    cfg = PlannerConfig(
        pressure_sign_multiplier=float(sign_cfg["planner_sign_multiplier"]),
        pressure_norm_cap=float(args.planner_pressure_norm_cap),
        envelope_use_discount=not bool(args.planner_envelope_raw),
        envelope_barrier_active=bool(args.planner_envelope_barrier),
        envelope_barrier_const=float(args.planner_envelope_barrier_const),
    )
    discounts = discount_cfg["default_discount_blocks"]
    excel_path = discover_excel()
    dataset_dir = Path(args.dataset_dir)
    if not dataset_dir.is_absolute():
        dataset_dir = repo_root / dataset_dir
    replay = Fino1ReplayDataset(
        dataset_dir=dataset_dir,
        split="test",
    )
    forecast_adapter = _make_forecast_adapter(args)
    forecast_source_label = (
        "oracle_future" if forecast_adapter is None else str(forecast_adapter.model_version)
    )

    from defaults import clone_cfg  # noqa: F401
    from run_validation import run_closed_loop_case

    rows: list[dict] = []
    issues: list[str] = []
    indexed_cases = _load_indexed_cases(str(args.cases_csv))
    if selected_ids:
        indexed_cases = [
            item
            for item in indexed_cases
            if f"{item[0]:02d}" in selected_ids
            or str(item[0]) in selected_ids
            or item[1] in selected_ids
        ]
    completed = 0
    for case_num, raw_case_id, ts_str, label in indexed_cases:
        case_id = _numbered_case_id(case_num, raw_case_id)
        ts = datetime.strptime(ts_str, TIMESTAMP_FMT)
        try:
            row_count = int(np.ceil(n_steps * DT / replay.update_interval_s))
            wind_trace = replay.build_wind_trace(start_timestamp=ts, row_count=row_count, dt_s=DT)
            wind_trace = smooth_wind_trace(wind_trace, float(args.wind_transition_s))
        except Exception as e:
            issues.append(f"{raw_case_id} {ts_str}: wind_trace failed: {e}")
            continue

        dfs: dict[str, pd.DataFrame] = {}
        providers: dict[str, BallastPlannerPreviewProvider | None] = {}
        variants = (primary_variant,) if bool(args.primary_only) else ("closed_only", primary_variant)
        for variant in variants:
            provider = None
            if variant == primary_variant:
                provider = BallastPlannerPreviewProvider(
                    replay_dataset=replay,
                    start_timestamp=ts,
                    cfg=cfg,
                    block_discounts=discounts,
                    bias_shape="event_decay",
                    objective_mode="economic",
                    event_reset_mode=str(args.event_reset_mode),
                    target_reset_tol_deg=float(args.target_reset_tol_deg),
                    setpoint_channel_enabled=False,
                    prediction_primary_enabled=True,
                    prediction_primary_scale=primary_scale,
                    primary_hold_target_mode=str(args.primary_hold_target_mode),
                    ff_channel_enabled=False,
                    pump_suppression_enabled=bool(args.pump_suppression),
                    pump_suppression_restart_err_kg=float(
                        args.pump_suppression_restart_err_kg
                    ),
                    pump_suppression_relief_margin_norm=float(
                        args.pump_suppression_relief_margin_norm
                    ),
                    pump_suppression_low_risk_norm=float(
                        args.pump_suppression_low_risk_norm
                    ),
                    event_risk_pressure_boost_enabled=bool(
                        args.event_risk_pressure_boost
                    ),
                    event_risk_pressure_boost_gain=float(
                        args.event_risk_pressure_boost_gain
                    ),
                    event_risk_pressure_boost_threshold=float(
                        args.event_risk_pressure_boost_threshold
                    ),
                    event_risk_pressure_boost_max=float(
                        args.event_risk_pressure_boost_max
                    ),
                    event_risk_pressure_floor_enabled=bool(
                        args.event_risk_pressure_floor
                    ),
                    event_risk_pressure_floor_threshold=float(
                        args.event_risk_pressure_floor_threshold
                    ),
                    event_risk_pressure_floor_norm=float(
                        args.event_risk_pressure_floor_norm
                    ),
                    event_risk_pressure_floor_max_lift=float(
                        args.event_risk_pressure_floor_max_lift
                    ),
                    preview_lead_action_enabled=bool(args.preview_lead_action),
                    preview_lead_event_threshold=float(args.preview_lead_event_threshold),
                    preview_lead_current_norm_max=float(args.preview_lead_current_norm_max),
                    preview_lead_future_norm_min=float(args.preview_lead_future_norm_min),
                    preview_lead_current_wind_max_ms=float(
                        args.preview_lead_current_wind_max_ms
                    ),
                    preview_lead_pitch_abs_max_deg=float(
                        args.preview_lead_pitch_abs_max_deg
                    ),
                    preview_lead_roll_abs_max_deg=float(
                        args.preview_lead_roll_abs_max_deg
                    ),
                    preview_lead_action_name=str(args.preview_lead_action_name),
                    pump_suppression_event_risk_guard_enabled=bool(
                        args.pump_suppression_event_risk_guard
                    ),
                    pump_suppression_event_risk_guard_threshold=float(
                        args.pump_suppression_event_risk_guard_threshold
                    ),
                    relief_medium_cap_enabled=bool(args.relief_medium_cap),
                    relief_medium_cap_event_threshold=float(
                        args.relief_medium_cap_event_threshold
                    ),
                    relief_medium_cap_ratio=float(args.relief_medium_cap_ratio),
                    forecast_adapter=forecast_adapter,
                )
            providers[variant] = provider
            run_name = f"{case_id}_{ts_str.replace(':', '').replace(' ', '_')}_{variant}"
            print(f"--- {run_name} ---", flush=True)
            try:
                _, timeseries = run_closed_loop_case(
                    excel_path=excel_path,
                    case_name=run_name,
                    dt=DT,
                    n_steps=n_steps,
                    wind_trace=wind_trace,
                    control_enabled=True,
                    record_timeseries=True,
                    experiment_protocol="main",
                    platform_profile="default",
                    start_from_heave_equilibrium=True,
                    preview_trim_provider=provider,
                    pump_cfg=closed_profile_cfg if variant == "closed_only" else None,
                    target_shape_override_cfg=(
                        closed_target_shape_cfg if variant == "closed_only" else primary_target_shape_cfg
                    ),
                )
            except Exception as e:
                issues.append(f"{run_name}: {type(e).__name__}: {e}")
                dfs = {}
                break
            df = pd.DataFrame(timeseries)
            df.to_csv(ts_dir / f"{run_name}_timeseries.csv", index=False)
            dfs[variant] = df
            if provider is not None and provider.records:
                pd.DataFrame(provider.records).to_csv(log_dir / f"{run_name}_planner_log.csv", index=False)

        if primary_variant not in dfs:
            continue
        has_closed = "closed_only" in dfs
        if not has_closed and not bool(args.primary_only):
            continue

        planner_log = None
        provider = providers.get(primary_variant)
        if provider is not None and provider.records:
            planner_log = pd.DataFrame(provider.records)
        fig_path: Path | None = None
        if has_closed and not bool(args.skip_figures):
            fig_path = plot_case(
                case_id,
                label,
                ts_str,
                dfs["closed_only"],
                dfs[primary_variant],
                fig_dir,
                primary_variant,
                duration_min=duration_min,
                planner_log=planner_log,
            )
        sc = summarize(dfs["closed_only"]) if has_closed else {}
        sp = summarize(dfs[primary_variant])
        row = {
            "case_id": case_id,
            "timestamp": ts_str,
            "label": label,
            "figure": str(fig_path.relative_to(repo_root)) if fig_path is not None else "",
            "closed_pump_work_m3": sc.get("pump_work_m3", np.nan),
            "primary_pump_work_m3": sp["pump_work_m3"],
            "d_pump_work_pct": (
                (sp["pump_work_m3"] - sc["pump_work_m3"]) / max(sc["pump_work_m3"], 1e-9) * 100.0
                if has_closed
                else np.nan
            ),
            "closed_pitch_p95": sc.get("pitch_abs_p95", np.nan),
            "primary_pitch_p95": sp["pitch_abs_p95"],
            "d_pitch_p95": sp["pitch_abs_p95"] - sc["pitch_abs_p95"] if has_closed else np.nan,
            "closed_roll_p95": sc.get("roll_abs_p95", np.nan),
            "primary_roll_p95": sp["roll_abs_p95"],
            "d_roll_p95": sp["roll_abs_p95"] - sc["roll_abs_p95"] if has_closed else np.nan,
            "closed_latch_switches": sc.get("latch_switches", np.nan),
            "primary_latch_switches": sp["latch_switches"],
            "primary_delta_mean_kg": sp["primary_delta_mean_kg"],
            "primary_candidate_ratio": sp["primary_candidate_ratio"],
            "primary_applied_ratio": sp["primary_applied_ratio"],
            "primary_safety_fallback_ratio": sp["primary_safety_fallback_ratio"],
            "primary_safety_active_ratio": sp["primary_safety_active_ratio"],
            "primary_safety_transition_count": sp["primary_safety_transition_count"],
            "preview_pump_suppression_ratio": sp["preview_pump_suppression_ratio"],
            "preview_suppression_delta_mean_kg": sp["preview_suppression_delta_mean_kg"],
            "preview_event_risk_boost_ratio": sp["preview_event_risk_boost_ratio"],
            "preview_event_risk_scale_mean": sp["preview_event_risk_scale_mean"],
            "preview_event_risk_floor_ratio": sp["preview_event_risk_floor_ratio"],
            "preview_event_risk_floor_norm_mean": sp["preview_event_risk_floor_norm_mean"],
            "preview_lead_action_ratio": sp["preview_lead_action_ratio"],
            "preview_relief_medium_cap_ratio": sp["preview_relief_medium_cap_ratio"],
            "closed_pump_profile": closed_profile,
            "primary_scale": primary_scale,
            "primary_control_profile": str(args.primary_control_profile),
            "primary_safety_profile": str(args.primary_safety_profile),
            "forecast_source_requested": str(args.forecast_source),
            "forecast_source_effective": forecast_source_label,
            "event_reset_mode": str(args.event_reset_mode),
            "primary_hold_target_mode": str(args.primary_hold_target_mode),
            "primary_target_resumed_ratio": sp["primary_target_resumed_ratio"],
            "planner_envelope_use_discount": int(bool(cfg.envelope_use_discount)),
            "planner_envelope_mode": "discounted" if cfg.envelope_use_discount else "raw",
            "planner_envelope_barrier_active": int(bool(cfg.envelope_barrier_active)),
            "planner_envelope_barrier_const": float(cfg.envelope_barrier_const),
            "planner_pressure_norm_cap": float(cfg.pressure_norm_cap),
            "target_reset_tol_deg": float(args.target_reset_tol_deg),
            "pump_suppression_enabled": int(bool(args.pump_suppression)),
            "pump_suppression_restart_err_kg": float(
                args.pump_suppression_restart_err_kg
            ),
            "pump_suppression_relief_margin_norm": float(
                args.pump_suppression_relief_margin_norm
            ),
            "pump_suppression_low_risk_norm": float(
                args.pump_suppression_low_risk_norm
            ),
            "event_risk_pressure_boost_enabled": int(bool(args.event_risk_pressure_boost)),
            "event_risk_pressure_boost_gain": float(args.event_risk_pressure_boost_gain),
            "event_risk_pressure_boost_threshold": float(args.event_risk_pressure_boost_threshold),
            "event_risk_pressure_boost_max": float(args.event_risk_pressure_boost_max),
            "event_risk_pressure_floor_enabled": int(bool(args.event_risk_pressure_floor)),
            "event_risk_pressure_floor_threshold": float(args.event_risk_pressure_floor_threshold),
            "event_risk_pressure_floor_norm": float(args.event_risk_pressure_floor_norm),
            "event_risk_pressure_floor_max_lift": float(args.event_risk_pressure_floor_max_lift),
            "preview_lead_action_enabled": int(bool(args.preview_lead_action)),
            "preview_lead_event_threshold": float(args.preview_lead_event_threshold),
            "preview_lead_current_norm_max": float(args.preview_lead_current_norm_max),
            "preview_lead_future_norm_min": float(args.preview_lead_future_norm_min),
            "preview_lead_current_wind_max_ms": float(args.preview_lead_current_wind_max_ms),
            "preview_lead_pitch_abs_max_deg": float(args.preview_lead_pitch_abs_max_deg),
            "preview_lead_roll_abs_max_deg": float(args.preview_lead_roll_abs_max_deg),
            "preview_lead_action_name": str(args.preview_lead_action_name),
            "pump_suppression_event_risk_guard_enabled": int(
                bool(args.pump_suppression_event_risk_guard)
            ),
            "pump_suppression_event_risk_guard_threshold": float(
                args.pump_suppression_event_risk_guard_threshold
            ),
            "relief_medium_cap_enabled": int(bool(args.relief_medium_cap)),
            "relief_medium_cap_event_threshold": float(args.relief_medium_cap_event_threshold),
            "relief_medium_cap_ratio": float(args.relief_medium_cap_ratio),
        }
        rows.append(row)
        completed += 1
        saved_label = f"saved {fig_path.name}" if fig_path is not None else "figures skipped"
        if has_closed:
            print(
                f"   {saved_label}: pump {row['d_pump_work_pct']:+.1f}%, "
                f"d_pitch_p95={row['d_pitch_p95']:+.3f}, d_roll_p95={row['d_roll_p95']:+.3f}",
                flush=True,
            )
        else:
            print(
                f"   {saved_label}: primary pump {row['primary_pump_work_m3']:.2f} m3, "
                f"pitch_p95={row['primary_pitch_p95']:.3f}, roll_p95={row['primary_roll_p95']:.3f}",
                flush=True,
            )

    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / "casebook_summary.csv", index=False)
    lines = [
        "# Prediction-primary casebook",
        "",
        f"- elapsed: `{time.perf_counter() - t0:.1f}s`",
        f"- completed cases: `{len(summary)}`",
        f"- variants: `{'primary_only' if bool(args.primary_only) else 'closed_only vs ' + primary_variant}`",
        f"- closed-only pump profile: `{closed_profile}`",
        f"- prediction-primary scale: `{primary_scale:.3f}`",
        f"- prediction-primary control profile: `{args.primary_control_profile}`",
        f"- prediction-primary safety profile: `{args.primary_safety_profile}`",
        f"- forecast source requested: `{args.forecast_source}`",
        f"- forecast source effective: `{forecast_source_label}`",
        f"- event reset mode: `{args.event_reset_mode}`",
        f"- planner envelope mode: `{'discounted' if cfg.envelope_use_discount else 'raw'}`",
        f"- planner envelope barrier: `{int(bool(cfg.envelope_barrier_active))}`",
        f"- planner envelope barrier const: `{float(cfg.envelope_barrier_const):.3f}`",
        f"- planner pressure norm cap: `{float(cfg.pressure_norm_cap):.3f}`",
        f"- target reset tol: `{float(args.target_reset_tol_deg):.3f} deg`",
        f"- forecast pump suppression: `{int(bool(args.pump_suppression))}`",
        f"- event-risk pressure boost: `{int(bool(args.event_risk_pressure_boost))}`",
        f"- event-risk pressure floor: `{int(bool(args.event_risk_pressure_floor))}`",
        f"- preview lead action: `{int(bool(args.preview_lead_action))}`",
        f"- suppression event-risk guard: `{int(bool(args.pump_suppression_event_risk_guard))}`",
        f"- relief medium cap: `{int(bool(args.relief_medium_cap))}`",
        f"- relief medium cap ratio: `{float(args.relief_medium_cap_ratio):.3f}`",
        f"- replay dataset: `{dataset_dir.relative_to(repo_root) if dataset_dir.is_relative_to(repo_root) else dataset_dir}`",
        f"- duration per case: `{duration_min:.0f} min`",
        f"- replay wind transition: `{float(args.wind_transition_s):.1f}s raised-cosine`",
        f"- figures skipped: `{int(bool(args.skip_figures))}`",
        f"- primary-only screening: `{int(bool(args.primary_only))}`",
        "",
        "| case | label | primary pump | pump delta | d_pitch_p95 | d_roll_p95 | suppression | risk boost | risk floor | lead | relief cap | safety fallback | latch switches | figure |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for _, r in summary.iterrows():
        pump_txt = "n/a" if pd.isna(r["d_pump_work_pct"]) else f"{r['d_pump_work_pct']:+.1f}%"
        d_pitch_txt = "n/a" if pd.isna(r["d_pitch_p95"]) else f"{r['d_pitch_p95']:+.3f}"
        d_roll_txt = "n/a" if pd.isna(r["d_roll_p95"]) else f"{r['d_roll_p95']:+.3f}"
        latch_txt = (
            "n/a"
            if pd.isna(r["closed_latch_switches"])
            else f"{int(r['closed_latch_switches'])}->{int(r['primary_latch_switches'])}"
        )
        lines.append(
            f"| {r['case_id']} | {r['label']} | {r['primary_pump_work_m3']:.2f} | {pump_txt} | "
            f"{d_pitch_txt} | {d_roll_txt} | "
            f"{r['preview_pump_suppression_ratio'] * 100:.1f}% | "
            f"{r['preview_event_risk_scale_mean']:.2f}x | "
            f"{r['preview_event_risk_floor_ratio'] * 100:.1f}% | "
            f"{r['preview_lead_action_ratio'] * 100:.1f}% | "
            f"{r['preview_relief_medium_cap_ratio'] * 100:.1f}% | "
            f"{r['primary_safety_fallback_ratio'] * 100:.1f}% | "
            f"{latch_txt} | "
            f"`{r['figure']}` |"
        )
    lines += ["", "## Issues", ""]
    lines += [f"- {x}" for x in issues] if issues else ["None."]
    (out_dir / "casebook_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Report: {out_dir / 'casebook_report.md'}")


if __name__ == "__main__":
    main()
