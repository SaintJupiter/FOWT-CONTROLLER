#!/usr/bin/env python3
"""Closed-loop no-preview strategy-layer candidate check.

This experiment keeps the forecast/planner disabled. It compares the frozen
closed baseline against a small set of transparent engineering candidates:

- actuator smoothing for a cleaner pump-rate trace,
- deadband target release, which drops stale mass targets once the PI controller
  says both pitch/roll are safely inside the deadband,
- a mild economy threshold profile, used only as a non-preview reference.

All metrics are computed from raw 1 Hz timeseries. Rolling medians are used only
for pump-rate visualization.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "archive" / "legacy_fowt_control"))

from defaults import clone_cfg
from run_prediction_primary_casebook import (
    CASES,
    CLOSED_ACTUATOR_SMOOTHING_CANDIDATE_PUMP_CFG,
    CLOSED_BASELINE_V1_PUMP_CFG,
    DT,
    TIMESTAMP_FMT,
    discover_excel,
    smooth_wind_trace,
)
from run_validation import run_closed_loop_case
from wind_prediction.replay_dataset import Fino1ReplayDataset


DEFAULT_CASE_IDS = "01,04,06,08,09"
DEFAULT_DURATION_S = 7200.0


DEADBAND_TARGET_RELEASE_CFG = {
    "deadband_target_release_enabled": True,
    "deadband_target_release_pitch_deg": 0.5,
    "deadband_target_release_roll_deg": 0.4,
    "deadband_target_release_near_zero_deg": 0.15,
    "deadband_target_release_rate_eps_deg_s": 0.002,
    "deadband_target_release_blend": 1.0,
    "deadband_target_release_require_both_axes": True,
    "deadband_target_release_reset_limiter": True,
    "deadband_target_release_exit_pitch_deg": 0.75,
    "deadband_target_release_exit_roll_deg": 0.6,
}


REACTIVE_PUMP_SUPPRESSION_CFG = {
    "reactive_pump_suppression_enabled": True,
    "reactive_pump_suppression_pitch_enter_deg": 1.2,
    "reactive_pump_suppression_roll_enter_deg": 1.0,
    "reactive_pump_suppression_pitch_exit_deg": 1.6,
    "reactive_pump_suppression_roll_exit_deg": 1.3,
    "reactive_pump_suppression_restart_err_kg": 1100.0,
    "reactive_pump_suppression_require_not_fullspeed": True,
}


REACTIVE_PUMP_SUPPRESSION_STRICT_CFG = {
    **REACTIVE_PUMP_SUPPRESSION_CFG,
    "reactive_pump_suppression_pitch_enter_deg": 0.8,
    "reactive_pump_suppression_roll_enter_deg": 0.65,
    "reactive_pump_suppression_pitch_exit_deg": 1.2,
    "reactive_pump_suppression_roll_exit_deg": 0.95,
    "reactive_pump_suppression_restart_err_kg": 700.0,
}


RELAXED_ECONOMY_RAMP1P5_PUMP_CFG = {
    **CLOSED_BASELINE_V1_PUMP_CFG,
    "pump_stop_err_kg": 700.0,
    "pump_restart_err_kg": 1100.0,
    "pump_min_off_s": 20.0,
    "pump_hold_before_stop_s": 15.0,
    "pump_ramp_up_m3_min_per_s": 1.5,
    "pump_ramp_down_m3_min_per_s": 2.25,
}


SMOOTH_RELAXED_ECONOMY_PUMP_CFG = {
    **CLOSED_ACTUATOR_SMOOTHING_CANDIDATE_PUMP_CFG,
    "pump_stop_err_kg": 700.0,
    "pump_restart_err_kg": 1100.0,
}


@dataclass(frozen=True)
class ClosedCandidate:
    name: str
    pump_cfg: dict
    target_shape_override: dict = field(default_factory=dict)
    controller_cfg: dict = field(default_factory=dict)
    note: str = ""


CANDIDATES: dict[str, ClosedCandidate] = {
    "v1_engineered_minimal": ClosedCandidate(
        name="v1_engineered_minimal",
        pump_cfg=CLOSED_BASELINE_V1_PUMP_CFG,
        note="Frozen no-preview PI + actuator constraints baseline.",
    ),
    "v2_actuator_smoothed_lb": ClosedCandidate(
        name="v2_actuator_smoothed_lb",
        pump_cfg=CLOSED_ACTUATOR_SMOOTHING_CANDIDATE_PUMP_CFG,
        note="Accepted strong actuator smoothing candidate.",
    ),
    "v1_target_release": ClosedCandidate(
        name="v1_target_release",
        pump_cfg=CLOSED_BASELINE_V1_PUMP_CFG,
        target_shape_override=DEADBAND_TARGET_RELEASE_CFG,
        note="Drops stale mass target inside pitch/roll deadband.",
    ),
    "v2_target_release": ClosedCandidate(
        name="v2_target_release",
        pump_cfg=CLOSED_ACTUATOR_SMOOTHING_CANDIDATE_PUMP_CFG,
        target_shape_override=DEADBAND_TARGET_RELEASE_CFG,
        note="Strong smoothing plus deadband target release.",
    ),
    "reactive_suppression_v1": ClosedCandidate(
        name="reactive_suppression_v1",
        pump_cfg=CLOSED_BASELINE_V1_PUMP_CFG,
        target_shape_override=REACTIVE_PUMP_SUPPRESSION_CFG,
        note="Closed-only safe-zone pump-start suppression for small unlatched target errors.",
    ),
    "reactive_suppression_strict_v1": ClosedCandidate(
        name="reactive_suppression_strict_v1",
        pump_cfg=CLOSED_BASELINE_V1_PUMP_CFG,
        target_shape_override=REACTIVE_PUMP_SUPPRESSION_STRICT_CFG,
        note="Stricter safe-zone suppression with smaller error threshold.",
    ),
    "smooth_reactive_suppression": ClosedCandidate(
        name="smooth_reactive_suppression",
        pump_cfg=CLOSED_ACTUATOR_SMOOTHING_CANDIDATE_PUMP_CFG,
        target_shape_override=REACTIVE_PUMP_SUPPRESSION_CFG,
        note="Accepted actuator smoothing plus closed-only safe-zone suppression.",
    ),
    "smooth_reactive_suppression_strict": ClosedCandidate(
        name="smooth_reactive_suppression_strict",
        pump_cfg=CLOSED_ACTUATOR_SMOOTHING_CANDIDATE_PUMP_CFG,
        target_shape_override=REACTIVE_PUMP_SUPPRESSION_STRICT_CFG,
        note="Accepted actuator smoothing plus strict safe-zone suppression.",
    ),
    "econ_ramp1p5": ClosedCandidate(
        name="econ_ramp1p5",
        pump_cfg=RELAXED_ECONOMY_RAMP1P5_PUMP_CFG,
        note="Previously selected mild no-preview economy reference.",
    ),
    "smooth_econ700": ClosedCandidate(
        name="smooth_econ700",
        pump_cfg=SMOOTH_RELAXED_ECONOMY_PUMP_CFG,
        note="Strong smoothing plus relaxed stop/restart thresholds.",
    ),
    "smooth_econ700_target_release": ClosedCandidate(
        name="smooth_econ700_target_release",
        pump_cfg=SMOOTH_RELAXED_ECONOMY_PUMP_CFG,
        target_shape_override=DEADBAND_TARGET_RELEASE_CFG,
        note="Strong smoothing, relaxed thresholds, and target release.",
    ),
    "deadband_exit0p3_v1": ClosedCandidate(
        name="deadband_exit0p3_v1",
        pump_cfg=CLOSED_BASELINE_V1_PUMP_CFG,
        controller_cfg={"deadband_exit_ratio": 0.3},
        note="Wider Schmitt-style deadband latch without changing PI gains.",
    ),
    "integral_decay_10min_v1": ClosedCandidate(
        name="integral_decay_10min_v1",
        pump_cfg=CLOSED_BASELINE_V1_PUMP_CFG,
        controller_cfg={
            "integral_deadband_decay_tau_s": 600.0,
            "integral_reversal_decay_tau_s": 300.0,
        },
        note="Leaks stale PI integral inside deadband and faster after error reversal.",
    ),
    "integral_decay_5min_v1": ClosedCandidate(
        name="integral_decay_5min_v1",
        pump_cfg=CLOSED_BASELINE_V1_PUMP_CFG,
        controller_cfg={
            "integral_deadband_decay_tau_s": 300.0,
            "integral_reversal_decay_tau_s": 180.0,
        },
        note="More aggressive stale-integral leakage candidate.",
    ),
    "smooth_integral_decay_10min": ClosedCandidate(
        name="smooth_integral_decay_10min",
        pump_cfg=CLOSED_ACTUATOR_SMOOTHING_CANDIDATE_PUMP_CFG,
        controller_cfg={
            "integral_deadband_decay_tau_s": 600.0,
            "integral_reversal_decay_tau_s": 300.0,
        },
        note="Display-friendly smoothing plus conservative integral leakage.",
    ),
}


COLORS = {
    "v1_engineered_minimal": "#2f6f9f",
    "v2_actuator_smoothed_lb": "#d85c27",
    "v1_target_release": "#4d908e",
    "v2_target_release": "#b45309",
    "reactive_suppression_v1": "#0f766e",
    "reactive_suppression_strict_v1": "#65a30d",
    "smooth_reactive_suppression": "#c2410c",
    "smooth_reactive_suppression_strict": "#ea580c",
    "econ_ramp1p5": "#2f855a",
    "smooth_econ700": "#7a5195",
    "smooth_econ700_target_release": "#8f2d56",
    "deadband_exit0p3_v1": "#5f6f52",
    "integral_decay_10min_v1": "#1f7a8c",
    "integral_decay_5min_v1": "#7c3aed",
    "smooth_integral_decay_10min": "#a83f39",
}


def _selected_cases(case_ids: str) -> list[tuple[str, str, str]]:
    selected = {s.strip() for s in str(case_ids).split(",") if s.strip()}
    rows = []
    for idx, (case_id, ts_str, label) in enumerate(CASES[:10], start=1):
        if selected and f"{idx:02d}" not in selected and str(idx) not in selected and case_id not in selected:
            continue
        rows.append((case_id, ts_str, label))
    return rows


def _work_m3(df: pd.DataFrame) -> float:
    pump = df["pump_total_rate_m3_min"].to_numpy(dtype=float)
    return float(np.trapezoid(np.abs(pump), dx=DT) / 60.0)


def _cumulative_work_m3(df: pd.DataFrame) -> np.ndarray:
    pump = np.abs(df["pump_total_rate_m3_min"].to_numpy(dtype=float))
    return np.cumsum(pump) * DT / 60.0


def _rolling_median(values: np.ndarray, window_s: float = 30.0) -> np.ndarray:
    window = max(1, int(round(float(window_s) / DT)))
    return pd.Series(values).rolling(window=window, center=True, min_periods=1).median().to_numpy(dtype=float)


def _visible_stage_changes(df: pd.DataFrame) -> int:
    count = 0
    for col in ("pump_stage_idx1", "pump_stage_idx2", "pump_stage_idx3"):
        if col in df.columns:
            vals = df[col].to_numpy(dtype=float)
            if vals.size > 1:
                count += int(np.sum(np.diff(vals) != 0.0))
    return count


def _pump_starts(df: pd.DataFrame) -> int:
    pump = np.abs(df["pump_total_rate_m3_min"].to_numpy(dtype=float)) > 1e-6
    if pump.size <= 1:
        return int(pump[0]) if pump.size else 0
    return int(np.sum(pump[1:] & ~pump[:-1]))


def _state_transitions(df: pd.DataFrame, col: str) -> int:
    if col not in df.columns:
        return 0
    vals = df[col].to_numpy(dtype=float)
    if vals.size <= 1:
        return 0
    return int(np.sum(np.abs(np.diff(vals)) > 0.5))


def _rate_tv_per_min(df: pd.DataFrame) -> float:
    pump = df["pump_total_rate_m3_min"].to_numpy(dtype=float)
    t = df["t_s"].to_numpy(dtype=float)
    duration_min = max(float(t[-1] - t[0]) / 60.0, 1e-9) if t.size > 1 else 1e-9
    return float(np.sum(np.abs(np.diff(pump))) / duration_min)


def _summary_row(case_id: str, profile: str, df: pd.DataFrame) -> dict:
    pitch = df["pitch_deg"].to_numpy(dtype=float)
    roll = df["roll_deg"].to_numpy(dtype=float)
    pump = df["pump_total_rate_m3_min"].to_numpy(dtype=float)
    zeros = pd.Series(np.zeros(len(df)))
    release_active = df.get("deadband_target_release_active", zeros).to_numpy(dtype=float)
    reactive_active = df.get("reactive_pump_suppression_active", zeros).to_numpy(dtype=float)
    return {
        "case_id": case_id,
        "profile": profile,
        "pump_work_m3": _work_m3(df),
        "pump_duty_pct": float(np.mean(np.abs(pump) > 1e-6) * 100.0),
        "latch_switches": int(df["pump_latch_switch_count"].iloc[-1])
        if "pump_latch_switch_count" in df.columns
        else 0,
        "visible_stage_idx_changes": _visible_stage_changes(df),
        "pump_total_starts": _pump_starts(df),
        "rate_tv_per_min": _rate_tv_per_min(df),
        "pitch_abs_p95": float(np.percentile(np.abs(pitch), 95)),
        "roll_abs_p95": float(np.percentile(np.abs(roll), 95)),
        "pitch_abs_max": float(np.max(np.abs(pitch))),
        "roll_abs_max": float(np.max(np.abs(roll))),
        "cmd_gap_p95_kg": float(np.percentile(df.get("cmd_gap_kg", zeros).to_numpy(dtype=float), 95)),
        "pump_backlog_p95_kg": float(np.percentile(df["pump_total_backlog_kg"], 95))
        if "pump_total_backlog_kg" in df.columns
        else np.nan,
        "deadband_release_active_pct": float(np.mean(release_active > 0.5) * 100.0),
        "deadband_release_transitions": _state_transitions(df, "deadband_target_release_active"),
        "deadband_release_delta_mean_kg": float(
            np.mean(df.get("deadband_target_release_delta_mean_kg", zeros).to_numpy(dtype=float))
        ),
        "reactive_suppression_active_pct": float(np.mean(reactive_active > 0.5) * 100.0),
        "reactive_suppression_transitions": _state_transitions(
            df, "reactive_pump_suppression_active"
        ),
        "reactive_suppression_blocked_tanks_sum": int(
            np.sum(df.get("reactive_suppression_blocked_tanks", zeros).to_numpy(dtype=float))
        ),
        "reactive_suppression_delta_mean_kg": float(
            np.mean(df.get("reactive_suppression_delta_mean_kg", zeros).to_numpy(dtype=float))
        ),
    }


def _delta_rows(summary: pd.DataFrame, reference_profile: str = "v1_engineered_minimal") -> pd.DataFrame:
    rows = []
    metrics = [
        "pump_work_m3",
        "pump_duty_pct",
        "latch_switches",
        "visible_stage_idx_changes",
        "pump_total_starts",
        "rate_tv_per_min",
        "pitch_abs_p95",
        "roll_abs_p95",
        "pitch_abs_max",
        "roll_abs_max",
        "cmd_gap_p95_kg",
        "pump_backlog_p95_kg",
        "deadband_release_active_pct",
        "deadband_release_transitions",
        "deadband_release_delta_mean_kg",
        "reactive_suppression_active_pct",
        "reactive_suppression_transitions",
        "reactive_suppression_blocked_tanks_sum",
        "reactive_suppression_delta_mean_kg",
    ]
    for case_id, group in summary.groupby("case_id", sort=False):
        base = group[group["profile"] == reference_profile]
        if base.empty:
            continue
        b = base.iloc[0]
        for _, r in group.iterrows():
            row = {
                "case_id": case_id,
                "profile": r["profile"],
                "reference_profile": reference_profile,
            }
            for key in metrics:
                a = float(b[key])
                c = float(r[key])
                row[f"{key}_ref"] = a
                row[f"{key}_profile"] = c
                row[f"d_{key}"] = c - a
                row[f"d_{key}_pct"] = (c - a) / max(abs(a), 1e-9) * 100.0
            row["pump_saving_pct"] = -row["d_pump_work_m3_pct"]
            rows.append(row)
    return pd.DataFrame(rows)


def _aggregate(delta: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for profile, group in delta.groupby("profile", sort=False):
        lowrisk = group[group["case_id"].str.contains("08_lowrisk", regex=False)]
        lowrisk_ok = True
        if not lowrisk.empty:
            lowrisk_ok = bool(
                float(lowrisk["pump_work_m3_profile"].iloc[0]) <= 1e-9
                and int(round(float(lowrisk["visible_stage_idx_changes_profile"].iloc[0]))) == 0
            )
        rows.append(
            {
                "profile": profile,
                "mean_pump_delta_pct": float(group["d_pump_work_m3_pct"].mean()),
                "max_pump_delta_pct": float(group["d_pump_work_m3_pct"].max()),
                "mean_pump_saving_pct": float(group["pump_saving_pct"].mean()),
                "mean_latch_delta_pct": float(group["d_latch_switches_pct"].mean()),
                "mean_visible_stage_delta_pct": float(group["d_visible_stage_idx_changes_pct"].mean()),
                "mean_rate_tv_delta_pct": float(group["d_rate_tv_per_min_pct"].mean()),
                "max_pitch_p95_delta_deg": float(group["d_pitch_abs_p95"].max()),
                "max_roll_p95_delta_deg": float(group["d_roll_abs_p95"].max()),
                "max_pitch_abs_delta_deg": float(group["d_pitch_abs_max"].max()),
                "max_roll_abs_delta_deg": float(group["d_roll_abs_max"].max()),
                "mean_deadband_release_active_pct": float(group["deadband_release_active_pct_profile"].mean()),
                "mean_deadband_release_transitions": float(group["deadband_release_transitions_profile"].mean()),
                "mean_reactive_suppression_active_pct": float(
                    group["reactive_suppression_active_pct_profile"].mean()
                ),
                "mean_reactive_suppression_transitions": float(
                    group["reactive_suppression_transitions_profile"].mean()
                ),
                "mean_reactive_suppression_delta_mean_kg": float(
                    group["reactive_suppression_delta_mean_kg_profile"].mean()
                ),
                "lowrisk_unchanged": int(lowrisk_ok),
                "acceptable_visual_baseline": int(
                    lowrisk_ok
                    and group["d_pump_work_m3_pct"].max() <= 4.0
                    and group["d_pitch_abs_p95"].max() <= 0.05
                    and group["d_roll_abs_p95"].max() <= 0.05
                    and group["d_pitch_abs_max"].max() <= 0.10
                    and group["d_roll_abs_max"].max() <= 0.10
                ),
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(
        ["acceptable_visual_baseline", "mean_rate_tv_delta_pct", "mean_pump_delta_pct"],
        ascending=[False, True, True],
    )


def _plot_case_compare(
    case_id: str,
    label: str,
    traces: dict[str, pd.DataFrame],
    profiles: list[str],
    out_dir: Path,
    suffix: str,
) -> Path:
    profiles = [p for p in profiles if p in traces]
    if len(profiles) < 2:
        raise ValueError("need at least two profiles to compare")
    ref = traces[profiles[0]]
    t = ref["t_s"].to_numpy(dtype=float) / 60.0
    fig, axes = plt.subplots(
        5,
        1,
        figsize=(14.2, 11.2),
        sharex=True,
        constrained_layout=True,
        gridspec_kw={"height_ratios": [0.75, 1.0, 1.15, 0.9, 0.8]},
    )

    ax = axes[0]
    ax.plot(t, ref["wind_speed"], color="#1d70b8", lw=1.3)
    ax.set_ylabel("Wind\n(m/s)")
    ax.grid(True, color="#d7dce2", lw=0.7)
    ax2 = ax.twinx()
    ax2.plot(t, ref["wind_dir_deg"], color="#45515f", lw=0.9, alpha=0.78)
    ax2.set_ylabel("Dir\n(deg)")

    ax = axes[1]
    for profile in profiles:
        df = traces[profile]
        color = COLORS.get(profile)
        ax.plot(t, df["pitch_deg"], color=color, lw=1.05, label=f"{profile} pitch")
        ax.plot(t, df["roll_deg"], color=color, lw=0.85, ls="--", alpha=0.78, label=f"{profile} roll")
    ax.axhline(0.0, color="#777", lw=0.7)
    ax.set_ylabel("Attitude\n(deg)")
    ax.legend(ncol=3, frameon=False, fontsize=7.4, loc="upper left")
    ax.grid(True, color="#d7dce2", lw=0.7)

    ax = axes[2]
    for profile in profiles:
        df = traces[profile]
        pump = df["pump_total_rate_m3_min"].to_numpy(dtype=float)
        color = COLORS.get(profile)
        ax.plot(t, pump, color=color, lw=0.45, alpha=0.16)
        ax.plot(t, _rolling_median(pump, 30.0), color=color, lw=1.45, label=profile)
    ax.set_ylabel("Pump rate\n30s median")
    ax.legend(ncol=3, frameon=False, fontsize=8, loc="upper left")
    ax.grid(True, color="#d7dce2", lw=0.7)

    ax = axes[3]
    for profile in profiles:
        df = traces[profile]
        color = COLORS.get(profile)
        ax.plot(t, _cumulative_work_m3(df), color=color, lw=1.55, label=f"{profile} ({_work_m3(df):.0f} m3)")
    ax.set_ylabel("Cumulative\npump work (m3)")
    ax.legend(ncol=3, frameon=False, fontsize=8, loc="upper left")
    ax.grid(True, color="#d7dce2", lw=0.7)

    ax = axes[4]
    for idx, profile in enumerate(profiles):
        df = traces[profile]
        active = np.abs(df["pump_total_rate_m3_min"].to_numpy(dtype=float)) > 1e-6
        y0 = idx * 1.0
        ax.fill_between(t, y0, y0 + 0.72, where=active, step="post", color=COLORS.get(profile), alpha=0.66)
        ax.text(t[0], y0 + 0.36, profile, va="center", ha="right", fontsize=7.7)
    ax.set_yticks([])
    ax.set_ylabel("Pump active")
    ax.set_xlabel("Time (min)")
    ax.grid(True, axis="x", color="#d7dce2", lw=0.7)

    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.suptitle(f"{case_id}: {label} | closed strategy candidate comparison", fontsize=13, y=1.01)
    out = out_dir / f"{case_id}_{suffix}_strategy_compare.png"
    fig.savefig(out, dpi=190, bbox_inches="tight")
    plt.close(fig)
    return out


def _plot_aggregate(aggregate: pd.DataFrame, out_dir: Path) -> Path | None:
    if aggregate.empty:
        return None
    data = aggregate[aggregate["profile"] != "v1_engineered_minimal"].copy()
    if data.empty:
        return None
    data = data.sort_values("mean_rate_tv_delta_pct")
    x = np.arange(len(data))
    labels = data["profile"].tolist()
    fig, axes = plt.subplots(4, 1, figsize=(14.4, 10.6), sharex=True, constrained_layout=True)
    axes[0].bar(x, data["mean_rate_tv_delta_pct"], color="#1b7f79")
    axes[0].set_ylabel("Rate TV/min\nchange (%)")
    axes[1].bar(x, data["mean_pump_delta_pct"], color="#d08c2f")
    axes[1].axhline(4.0, color="#9c2f2f", ls="--", lw=0.85)
    axes[1].set_ylabel("Pump work\nchange (%)")
    axes[2].bar(x, data["mean_visible_stage_delta_pct"], color="#7a5195")
    axes[2].set_ylabel("Visible stage\nchange (%)")
    axes[3].bar(x - 0.18, data["max_pitch_p95_delta_deg"], width=0.36, color="#2f6f9f", label="pitch p95")
    axes[3].bar(x + 0.18, data["max_roll_p95_delta_deg"], width=0.36, color="#d85c27", label="roll p95")
    axes[3].axhline(0.05, color="#9c2f2f", ls="--", lw=0.85)
    axes[3].set_ylabel("Worst p95\nchange (deg)")
    axes[3].legend(frameon=False, ncol=2, loc="upper left")
    axes[3].set_xticks(x)
    axes[3].set_xticklabels(labels, rotation=28, ha="right")
    for ax in axes:
        ax.grid(True, color="#d7dce2", lw=0.7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.suptitle("Closed strategy candidates: raw-metric deltas vs v1", fontsize=13, y=1.01)
    out = out_dir / "closed_strategy_candidate_aggregate.png"
    fig.savefig(out, dpi=190, bbox_inches="tight")
    plt.close(fig)
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default="outputs/wind_prediction/closed_strategy_layer_candidates_v1")
    parser.add_argument("--case-ids", default=DEFAULT_CASE_IDS)
    parser.add_argument("--duration-s", type=float, default=DEFAULT_DURATION_S)
    parser.add_argument("--wind-transition-s", type=float, default=0.0)
    parser.add_argument(
        "--profiles",
        default="",
        help="Comma-separated profile names. Empty means all candidates.",
    )
    parser.add_argument(
        "--dataset-dir",
        default="data/processed/wind_ml_10min/ballast_decision_fino1_meteo_aux_v1",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    t0 = time.perf_counter()
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = REPO_ROOT / out_dir
    ts_dir = out_dir / "timeseries"
    fig_dir = out_dir / "figures"
    for d in (out_dir, ts_dir, fig_dir):
        d.mkdir(parents=True, exist_ok=True)

    dataset_dir = Path(args.dataset_dir)
    if not dataset_dir.is_absolute():
        dataset_dir = REPO_ROOT / dataset_dir
    replay = Fino1ReplayDataset(dataset_dir=dataset_dir, split="test")
    excel_path = discover_excel()
    n_steps = int(round(float(args.duration_s) / DT))
    wanted_profiles = {p.strip() for p in str(args.profiles).split(",") if p.strip()}
    candidate_items = [
        (name, candidate)
        for name, candidate in CANDIDATES.items()
        if not wanted_profiles or name in wanted_profiles
    ]
    if "v1_engineered_minimal" not in {name for name, _ in candidate_items}:
        candidate_items.insert(0, ("v1_engineered_minimal", CANDIDATES["v1_engineered_minimal"]))

    summary_rows: list[dict] = []
    traces_by_case: dict[str, dict[str, pd.DataFrame]] = {}
    labels: dict[str, str] = {}
    issues: list[str] = []

    for case_id, ts_str, label in _selected_cases(args.case_ids):
        labels[case_id] = label
        ts = datetime.strptime(ts_str, TIMESTAMP_FMT)
        try:
            row_count = int(np.ceil(n_steps * DT / replay.update_interval_s))
            wind_trace = replay.build_wind_trace(start_timestamp=ts, row_count=row_count, dt_s=DT)
            wind_trace = smooth_wind_trace(wind_trace, float(args.wind_transition_s))
        except Exception as exc:
            issues.append(f"{case_id}: wind trace failed: {exc}")
            continue
        traces_by_case[case_id] = {}
        for profile, candidate in candidate_items:
            run_name = f"{case_id}_{ts_str.replace(':', '').replace(' ', '_')}_{profile}"
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
                    preview_trim_provider=None,
                    pump_cfg=clone_cfg(candidate.pump_cfg),
                    target_shape_override_cfg=clone_cfg(candidate.target_shape_override),
                    controller_cfg=clone_cfg(candidate.controller_cfg),
                )
            except Exception as exc:
                issues.append(f"{run_name}: simulation failed: {exc}")
                continue
            df = pd.DataFrame(timeseries)
            df.to_csv(ts_dir / f"{run_name}_timeseries.csv", index=False)
            traces_by_case[case_id][profile] = df
            summary_rows.append(_summary_row(case_id, profile, df))

    summary = pd.DataFrame(summary_rows)
    delta = _delta_rows(summary)
    aggregate = _aggregate(delta)
    fig_paths: list[Path] = []
    aggregate_fig = _plot_aggregate(aggregate, fig_dir)
    if aggregate_fig is not None:
        fig_paths.append(aggregate_fig)

    # Compare the practical display baseline options on the most visibly busy cases.
    for case_id in ("04_decay_strong", "06_signflip_high"):
        if case_id in traces_by_case:
            profiles = [
                "v1_engineered_minimal",
                "v2_actuator_smoothed_lb",
                "reactive_suppression_v1",
                "smooth_reactive_suppression",
                "v2_target_release",
                "smooth_econ700",
                "smooth_econ700_target_release",
                "integral_decay_10min_v1",
                "smooth_integral_decay_10min",
            ]
            try:
                fig_paths.append(
                    _plot_case_compare(
                        case_id,
                        labels.get(case_id, case_id),
                        traces_by_case[case_id],
                        profiles,
                        fig_dir,
                        "v1_v2_release_econ",
                    )
                )
            except Exception as exc:
                issues.append(f"{case_id}: compare plot failed: {exc}")

    summary.to_csv(out_dir / "closed_strategy_candidate_summary.csv", index=False)
    delta.to_csv(out_dir / "closed_strategy_candidate_deltas.csv", index=False)
    aggregate.to_csv(out_dir / "closed_strategy_candidate_aggregate.csv", index=False)
    meta = {
        "duration_s": float(args.duration_s),
        "case_ids": str(args.case_ids),
        "wind_transition_s": float(args.wind_transition_s),
        "profiles": {
            name: {
                "pump_cfg": candidate.pump_cfg,
                "target_shape_override": candidate.target_shape_override,
                "controller_cfg": candidate.controller_cfg,
                "note": candidate.note,
            }
            for name, candidate in CANDIDATES.items()
        },
        "metric_layer": "raw 1Hz timeseries only",
        "plot_layer": "30s rolling median is visualization only",
        "visual_acceptance_note": "User accepted <= about 3.3% pump-work increase for strong smoothing if the engineering trace is cleaner.",
        "issues": issues,
        "runtime_s": float(time.perf_counter() - t0),
        "figures": [str(p.relative_to(out_dir)) for p in fig_paths],
    }
    (out_dir / "closed_strategy_candidate_meta.json").write_text(
        json.dumps(meta, indent=2, allow_nan=True),
        encoding="utf-8",
    )

    lines = [
        "# Closed Strategy-Layer Candidates v1",
        "",
        "Purpose: diagnose whether the no-preview closed controller has a strategy-layer mismatch beyond actuator smoothing.",
        "",
        "Interpretation rule: pump work, stage changes, and duty use raw 1 Hz data. Rolling medians are visualization only.",
        "",
        "## Candidate Notes",
        "",
    ]
    for name, candidate in CANDIDATES.items():
        lines.append(f"- `{name}`: {candidate.note}")
    lines.extend(
        [
            "",
            "## Aggregate vs v1",
            "",
            "| profile | mean pump delta | max pump delta | rate TV delta | visible stage delta | max pitch p95 delta | max roll p95 delta | release active | reactive active | acceptable visual |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for _, r in aggregate.iterrows():
        lines.append(
            f"| {r['profile']} | {r['mean_pump_delta_pct']:+.2f}% | {r['max_pump_delta_pct']:+.2f}% | {r['mean_rate_tv_delta_pct']:+.2f}% | {r['mean_visible_stage_delta_pct']:+.2f}% | {r['max_pitch_p95_delta_deg']:+.4f} | {r['max_roll_p95_delta_deg']:+.4f} | {r['mean_deadband_release_active_pct']:.1f}% | {r['mean_reactive_suppression_active_pct']:.1f}% | {int(r['acceptable_visual_baseline'])} |"
        )
    lines.extend(["", "## Figures", ""])
    lines.extend([f"- `{p.relative_to(out_dir)}`" for p in fig_paths])
    if issues:
        lines.extend(["", "## Issues", "", *[f"- {item}" for item in issues]])
    (out_dir / "closed_strategy_candidate_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(out_dir)


if __name__ == "__main__":
    main()
