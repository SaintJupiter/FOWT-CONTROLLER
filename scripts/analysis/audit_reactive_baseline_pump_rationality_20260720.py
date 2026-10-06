#!/usr/bin/env python3
"""Audit whether the reactive closed_only baseline pumps for traceable reasons.

This is a read-only audit of two frozen representative windows.  It separates
active pump volume into:

1. current attitude feedback outside the controller's hysteretic deadband;
2. completion of an already-issued tank target while both axes are in deadband;
3. the short actuator dwell/ramp tail after the target error is already small.

No controller is rerun and no frozen artifact is modified.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
FROZEN = ROOT / "outputs/wind_prediction/frozen_h_holdout150_20260710"
OUT = (
    ROOT
    / "outputs/paper_figures/20260720_abcd_absolute_attitude_context"
    / "qa"
)

PITCH_ENTER_DEG = 1.0
ROLL_ENTER_DEG = 0.8
PITCH_EXIT_DEG = 0.5
ROLL_EXIT_DEG = 0.4
PUMP_STOP_ERR_KG = 300.0
PUMP_RESTART_ERR_KG = 500.0
RATE_EPS = 1e-6

CASES = [
    {
        "short_name": "04_sel6h_50_t290",
        "case_name": "04_sel6h_50_2024-07-03_210000",
        "batch_task": "batch_3_paired/task_5",
        "event_min": 290.0,
    },
    {
        "short_name": "05_posadd6h_023_t140",
        "case_name": "05_posadd6h_023_2023-08-09_042000",
        "batch_task": "batch_3_paired/task_3",
        "event_min": 140.0,
    },
]

COLS = [
    "t_s",
    "pitch_deg",
    "roll_deg",
    "pitch_sp_deg",
    "roll_sp_deg",
    "ctrl_filtered_pitch_deg",
    "ctrl_filtered_roll_deg",
    "ctrl_pitch_err_raw_deg",
    "ctrl_roll_err_raw_deg",
    "ctrl_pitch_err_deg",
    "ctrl_roll_err_deg",
    "ctrl_pitch_in_deadband",
    "ctrl_roll_in_deadband",
    "ctrl_pitch_u_total",
    "ctrl_roll_u_total",
    "ctrl_pitch_i_term",
    "ctrl_roll_i_term",
    "ctrl_sign_warn",
    "target_tank1_kg",
    "target_tank2_kg",
    "target_tank3_kg",
    "tank1_kg",
    "tank2_kg",
    "tank3_kg",
    "err_tank1_kg",
    "err_tank2_kg",
    "err_tank3_kg",
    "pump_rate1_m3min",
    "pump_rate2_m3min",
    "pump_rate3_m3min",
    "pump_total_rate_m3_min",
    "pump_total_backlog_kg",
    "pump_target_motion_kg_s",
    "pump_latched1",
    "pump_latched2",
    "pump_latched3",
    "pump_fullspeed_any",
]


def pct(num: float, den: float) -> float:
    return 100.0 * float(num) / float(den) if den > 0 else 0.0


def load_case(spec: dict) -> pd.DataFrame:
    path = (
        FROZEN
        / spec["batch_task"]
        / "timeseries"
        / f"{spec['case_name']}_closed_only_timeseries.csv"
    )
    frame = pd.read_csv(path, usecols=COLS)
    start_s = (float(spec["event_min"]) - 20.0) * 60.0
    end_s = (float(spec["event_min"]) + 60.0) * 60.0
    frame = frame[frame.t_s.between(start_s, end_s)].copy()
    frame["relative_time_min"] = (
        frame["t_s"] / 60.0 - float(spec["event_min"])
    )
    return frame.reset_index(drop=True)


def episode_starts(active: np.ndarray) -> np.ndarray:
    return np.flatnonzero(active & ~np.r_[False, active[:-1]])


def classify_rows(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["pump_active"] = out.pump_total_rate_m3_min.abs() > RATE_EPS
    out["both_attitude_axes_in_deadband"] = (
        (out.ctrl_pitch_in_deadband > 0)
        & (out.ctrl_roll_in_deadband > 0)
    )
    tank_err = out[
        ["err_tank1_kg", "err_tank2_kg", "err_tank3_kg"]
    ].abs()
    out["max_abs_tank_error_kg"] = tank_err.max(axis=1)
    out["mean_abs_tank_error_kg"] = tank_err.mean(axis=1)
    out["theta_deg"] = np.maximum(
        out.pitch_deg.abs(), out.roll_deg.abs()
    )
    out["normalized_attitude_error"] = np.maximum(
        out.ctrl_pitch_err_raw_deg.abs() / PITCH_ENTER_DEG,
        out.ctrl_roll_err_raw_deg.abs() / ROLL_ENTER_DEG,
    )

    reason = np.full(len(out), "pump_off", dtype=object)
    active = out.pump_active.to_numpy(bool)
    both_in = out.both_attitude_axes_in_deadband.to_numpy(bool)
    max_err = out.max_abs_tank_error_kg.to_numpy(float)
    reason[active & ~both_in] = "current_attitude_feedback"
    reason[
        active & both_in & (max_err > PUMP_STOP_ERR_KG)
    ] = "existing_target_completion"
    reason[
        active & both_in & (max_err <= PUMP_STOP_ERR_KG)
    ] = "actuator_dwell_or_ramp_tail"
    out["pump_reason_class"] = reason
    return out


def summarize(spec: dict, frame: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    dt_s = float(np.median(np.diff(frame.t_s.to_numpy(float))))
    frame["incremental_pump_m3"] = (
        frame.pump_total_rate_m3_min.abs() * dt_s / 60.0
    )
    active = frame.pump_active.to_numpy(bool)
    total_volume = float(frame.incremental_pump_m3.sum())
    total_active_s = float(active.sum() * dt_s)

    breakdown = []
    for reason in [
        "current_attitude_feedback",
        "existing_target_completion",
        "actuator_dwell_or_ramp_tail",
    ]:
        mask = frame.pump_reason_class.eq(reason)
        volume = float(frame.loc[mask, "incremental_pump_m3"].sum())
        seconds = float(mask.sum() * dt_s)
        breakdown.append(
            {
                "reason": reason,
                "pump_volume_m3": volume,
                "pump_volume_share_pct": pct(volume, total_volume),
                "active_time_s": seconds,
                "active_time_share_pct": pct(seconds, total_active_s),
            }
        )

    starts = episode_starts(active)
    episode_rows = []
    for episode_id, idx in enumerate(starts, start=1):
        row = frame.iloc[int(idx)]
        future_time = float(row.t_s) + 300.0
        future_idx = int(
            np.argmin(
                np.abs(frame.t_s.to_numpy(float) - future_time)
            )
        )
        future = frame.iloc[future_idx]
        episode_rows.append(
            {
                "case": spec["short_name"],
                "episode": episode_id,
                "relative_start_min": float(row.relative_time_min),
                "pitch_deg": float(row.pitch_deg),
                "roll_deg": float(row.roll_deg),
                "filtered_pitch_deg": float(
                    row.ctrl_filtered_pitch_deg
                ),
                "filtered_roll_deg": float(row.ctrl_filtered_roll_deg),
                "pitch_in_deadband": int(row.ctrl_pitch_in_deadband),
                "roll_in_deadband": int(row.ctrl_roll_in_deadband),
                "max_abs_tank_error_kg": float(
                    row.max_abs_tank_error_kg
                ),
                "pump_reason_class": str(row.pump_reason_class),
                "normalized_attitude_error_at_start": float(
                    row.normalized_attitude_error
                ),
                "normalized_attitude_error_after_5min": float(
                    future.normalized_attitude_error
                ),
                "ctrl_sign_warn": int(row.ctrl_sign_warn),
            }
        )
    episodes = pd.DataFrame(episode_rows)

    active_frame = frame[frame.pump_active]
    immediate_threshold = (
        active_frame.normalized_attitude_error >= 1.0
    )
    hysteresis_continuation = (
        (active_frame.normalized_attitude_error < 1.0)
        & (~active_frame.both_attitude_axes_in_deadband)
    )
    immediate_volume = float(
        active_frame.loc[
            immediate_threshold, "incremental_pump_m3"
        ].sum()
    )
    hysteresis_volume = float(
        active_frame.loc[
            hysteresis_continuation, "incremental_pump_m3"
        ].sum()
    )
    sign_warn_volume = float(
        active_frame.loc[
            active_frame.ctrl_sign_warn.astype(bool),
            "incremental_pump_m3",
        ].sum()
    )
    pitch_error = active_frame.ctrl_pitch_err_raw_deg.to_numpy(float)
    roll_error = active_frame.ctrl_roll_err_raw_deg.to_numpy(float)
    pitch_effort = active_frame.ctrl_pitch_u_total.to_numpy(float)
    roll_effort = active_frame.ctrl_roll_u_total.to_numpy(float)
    dominant_is_pitch = (
        np.abs(pitch_error) / PITCH_ENTER_DEG
        >= np.abs(roll_error) / ROLL_ENTER_DEG
    )
    dominant_axis_sign_consistent = np.where(
        dominant_is_pitch,
        pitch_error * pitch_effort >= 0.0,
        roll_error * roll_effort >= 0.0,
    )
    active_volume_rows = active_frame.incremental_pump_m3.to_numpy(float)
    dominant_consistent_volume = float(
        active_volume_rows[dominant_axis_sign_consistent].sum()
    )

    episode_durations = []
    for start_idx in starts:
        end_idx = int(start_idx)
        while (
            end_idx + 1 < len(active)
            and bool(active[end_idx + 1])
        ):
            end_idx += 1
        episode_durations.append(
            (end_idx - int(start_idx) + 1) * dt_s
        )
    target_cols = [
        "target_tank1_kg",
        "target_tank2_kg",
        "target_tank3_kg",
    ]
    target_motion = (
        frame[target_cols].diff().abs().max(axis=1).fillna(0.0)
        > 1e-6
    )
    active_and_target_moving = active & target_motion.to_numpy(bool)

    summary = {
        "case": spec["short_name"],
        "window_relative_min": [-20.0, 60.0],
        "controller_thresholds": {
            "pitch_enter_deg": PITCH_ENTER_DEG,
            "roll_enter_deg": ROLL_ENTER_DEG,
            "pitch_exit_deg": PITCH_EXIT_DEG,
            "roll_exit_deg": ROLL_EXIT_DEG,
            "pump_stop_err_kg": PUMP_STOP_ERR_KG,
            "pump_restart_err_kg": PUMP_RESTART_ERR_KG,
        },
        "pump": {
            "total_volume_m3": total_volume,
            "active_time_s": total_active_s,
            "active_time_pct": pct(total_active_s, len(frame) * dt_s),
            "episode_count": int(len(starts)),
            "volume_breakdown": breakdown,
            "active_target_moving_time_pct_of_active": pct(
                active_and_target_moving.sum() * dt_s,
                total_active_s,
            ),
            "active_ctrl_sign_warn_time_s": float(
                active_frame.ctrl_sign_warn.astype(bool).sum() * dt_s
            ),
            "active_ctrl_sign_warn_share_pct": pct(
                active_frame.ctrl_sign_warn.astype(bool).sum() * dt_s,
                total_active_s,
            ),
            "active_ctrl_sign_warn_volume_m3": sign_warn_volume,
            "active_ctrl_sign_warn_volume_share_pct": pct(
                sign_warn_volume, total_volume
            ),
            "immediate_threshold_volume_m3": immediate_volume,
            "immediate_threshold_volume_share_pct": pct(
                immediate_volume, total_volume
            ),
            "hysteresis_continuation_volume_m3": hysteresis_volume,
            "hysteresis_continuation_volume_share_pct": pct(
                hysteresis_volume, total_volume
            ),
            "episode_duration_median_s": float(
                np.median(episode_durations)
            )
            if episode_durations
            else 0.0,
            "episode_duration_p95_s": float(
                np.quantile(episode_durations, 0.95)
            )
            if episode_durations
            else 0.0,
            "dominant_axis_command_sign_consistent_time_pct": pct(
                dominant_axis_sign_consistent.sum() * dt_s,
                total_active_s,
            ),
            "dominant_axis_command_sign_consistent_volume_pct": pct(
                dominant_consistent_volume, total_volume
            ),
            "dominant_axis_command_sign_mismatch_volume_pct": (
                100.0 - pct(dominant_consistent_volume, total_volume)
            ),
        },
        "attitude_during_pump": {
            "pitch_abs_mean_deg": float(
                active_frame.pitch_deg.abs().mean()
            ),
            "roll_abs_mean_deg": float(
                active_frame.roll_deg.abs().mean()
            ),
            "theta_p95_deg": float(
                active_frame.theta_deg.quantile(0.95)
            ),
            "theta_gt2_time_s": float(
                (active_frame.theta_deg > 2.0).sum() * dt_s
            ),
            "both_axes_in_deadband_time_share_pct": pct(
                active_frame.both_attitude_axes_in_deadband.sum() * dt_s,
                total_active_s,
            ),
        },
        "episode_response": {
            "starts_with_any_axis_outside_deadband": int(
                (
                    (episodes.pitch_in_deadband == 0)
                    | (episodes.roll_in_deadband == 0)
                ).sum()
            )
            if not episodes.empty
            else 0,
            "starts_with_target_error_above_restart_kg": int(
                (
                    episodes.max_abs_tank_error_kg
                    >= PUMP_RESTART_ERR_KG
                ).sum()
            )
            if not episodes.empty
            else 0,
            "five_min_normalized_error_lower_count": int(
                (
                    episodes.normalized_attitude_error_after_5min
                    < episodes.normalized_attitude_error_at_start
                ).sum()
            )
            if not episodes.empty
            else 0,
            "episode_count": int(len(episodes)),
        },
    }
    return summary, episodes


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    summaries = []
    episode_frames = []
    for spec in CASES:
        frame = classify_rows(load_case(spec))
        summary, episodes = summarize(spec, frame)
        summaries.append(summary)
        episode_frames.append(episodes)
        frame.to_csv(
            OUT / f"{spec['short_name']}_reactive_pump_audit_rows.csv",
            index=False,
        )

    with (OUT / "reactive_baseline_pump_rationality_audit.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump({"cases": summaries}, handle, ensure_ascii=False, indent=2)
    pd.concat(episode_frames, ignore_index=True).to_csv(
        OUT / "reactive_baseline_pump_start_episodes.csv", index=False
    )
    print(
        OUT / "reactive_baseline_pump_rationality_audit.json"
    )


if __name__ == "__main__":
    main()
