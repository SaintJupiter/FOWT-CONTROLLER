#!/usr/bin/env python3
"""Evaluate a light actuator-smoothing v2 profile for the no-preview closed baseline.

Metrics are always computed from raw 1 Hz timeseries. Plotting may add smoothed
visual overlays, but never feeds those overlays back into pump-work or switch
counts.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
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

from wind_prediction.replay_dataset import Fino1ReplayDataset

from run_prediction_primary_casebook import (
    CASES,
    CLOSED_BASELINE_V1_PUMP_CFG,
    CLOSED_ACTUATOR_SMOOTHING_CANDIDATE_PUMP_CFG,
    DT,
    TIMESTAMP_FMT,
    discover_excel,
    smooth_wind_trace,
)


DEFAULT_CASE_IDS = "01,04,06,08,09"
PROFILE_CFGS = {
    "v1_engineered_minimal": CLOSED_BASELINE_V1_PUMP_CFG,
    "v2_actuator_smoothed_lb": CLOSED_ACTUATOR_SMOOTHING_CANDIDATE_PUMP_CFG,
}

OAT_PROFILE_CFGS = {
    "v1_engineered_minimal": CLOSED_BASELINE_V1_PUMP_CFG,
    "release_tau5_only": {
        **CLOSED_BASELINE_V1_PUMP_CFG,
        "pump_rate_release_tau_s": 5.0,
    },
    "stage_h75_d10_only": {
        **CLOSED_BASELINE_V1_PUMP_CFG,
        "pump_stage_hysteresis_kg": 75.0,
        "pump_stage_min_dwell_s": 10.0,
        "pump_stage_allow_zero_rate_latched": True,
    },
    "stage_h25_only": {
        **CLOSED_BASELINE_V1_PUMP_CFG,
        "pump_stage_hysteresis_kg": 25.0,
        "pump_stage_allow_zero_rate_latched": True,
    },
    "stage_h50_only": {
        **CLOSED_BASELINE_V1_PUMP_CFG,
        "pump_stage_hysteresis_kg": 50.0,
        "pump_stage_allow_zero_rate_latched": True,
    },
    "stage_h75_only": {
        **CLOSED_BASELINE_V1_PUMP_CFG,
        "pump_stage_hysteresis_kg": 75.0,
        "pump_stage_allow_zero_rate_latched": True,
    },
    "stop_hold_only": {
        **CLOSED_BASELINE_V1_PUMP_CFG,
        "pump_min_off_s": 20.0,
        "pump_hold_before_stop_s": 15.0,
    },
    "ramp_0p5_0p75_only": {
        **CLOSED_BASELINE_V1_PUMP_CFG,
        "pump_ramp_up_m3_min_per_s": 0.5,
        "pump_ramp_down_m3_min_per_s": 0.75,
    },
    "ramp_0p25_0p5_only": {
        **CLOSED_BASELINE_V1_PUMP_CFG,
        "pump_ramp_up_m3_min_per_s": 0.25,
        "pump_ramp_down_m3_min_per_s": 0.5,
    },
    "stop_hold_ramp_0p5": {
        **CLOSED_BASELINE_V1_PUMP_CFG,
        "pump_min_off_s": 20.0,
        "pump_hold_before_stop_s": 15.0,
        "pump_ramp_up_m3_min_per_s": 0.5,
        "pump_ramp_down_m3_min_per_s": 0.75,
    },
    "stage_release_only": {
        **CLOSED_BASELINE_V1_PUMP_CFG,
        "pump_stage_hysteresis_kg": 75.0,
        "pump_stage_min_dwell_s": 10.0,
        "pump_rate_release_tau_s": 5.0,
        "pump_stage_allow_zero_rate_latched": True,
    },
    "v2_actuator_smoothed_lb": CLOSED_ACTUATOR_SMOOTHING_CANDIDATE_PUMP_CFG,
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


def _visible_stage_changes(df: pd.DataFrame) -> int:
    count = 0
    for col in ("pump_stage_idx1", "pump_stage_idx2", "pump_stage_idx3"):
        if col in df.columns:
            vals = df[col].to_numpy(dtype=float)
            if vals.size > 1:
                count += int(np.sum(np.diff(vals) != 0.0))
    return count


def _total_pump_starts(df: pd.DataFrame) -> int:
    pump = df["pump_total_rate_m3_min"].to_numpy(dtype=float)
    active = np.abs(pump) > 1e-6
    if active.size <= 1:
        return int(active[0]) if active.size else 0
    return int(np.sum(active[1:] & ~active[:-1]))


def _rate_tv_per_min(df: pd.DataFrame) -> float:
    pump = df["pump_total_rate_m3_min"].to_numpy(dtype=float)
    t = df["t_s"].to_numpy(dtype=float)
    duration_min = max(float(t[-1] - t[0]) / 60.0, 1e-9) if t.size > 1 else 1e-9
    return float(np.sum(np.abs(np.diff(pump))) / duration_min)


def _transition_count(series: pd.Series) -> int:
    vals = series.to_numpy(dtype=float)
    if vals.size <= 1:
        return 0
    return int(np.sum(np.diff(vals) != 0.0))


def _summarize(case_id: str, profile: str, df: pd.DataFrame) -> dict:
    pitch = df["pitch_deg"].to_numpy(dtype=float)
    roll = df["roll_deg"].to_numpy(dtype=float)
    pump = df["pump_total_rate_m3_min"].to_numpy(dtype=float)
    return {
        "case_id": case_id,
        "profile": profile,
        "pump_work_m3": _work_m3(df),
        "pump_duty_pct": float(np.mean(np.abs(pump) > 1e-6) * 100.0),
        "latch_switches": int(df["pump_latch_switch_count"].iloc[-1])
        if "pump_latch_switch_count" in df.columns
        else 0,
        "visible_stage_idx_changes": _visible_stage_changes(df),
        "pump_total_starts": _total_pump_starts(df),
        "rate_tv_per_min": _rate_tv_per_min(df),
        "pitch_abs_p95": float(np.percentile(np.abs(pitch), 95)),
        "roll_abs_p95": float(np.percentile(np.abs(roll), 95)),
        "pitch_abs_max": float(np.max(np.abs(pitch))),
        "roll_abs_max": float(np.max(np.abs(roll))),
        "pump_backlog_p95_kg": float(np.percentile(df["pump_total_backlog_kg"], 95))
        if "pump_total_backlog_kg" in df.columns
        else np.nan,
        "raw_stage_metric_note": "visible_stage_idx_changes uses raw pump_stage_idx columns, not smoothed plot data",
    }


def _delta_rows(summary: pd.DataFrame, reference_profile: str = "v1_engineered_minimal") -> pd.DataFrame:
    rows = []
    for (case_id, profile), group in summary.groupby(["case_id", "profile"], sort=True):
        case_group = summary[summary["case_id"] == case_id]
        if profile == reference_profile or reference_profile not in set(case_group["profile"]):
            continue
        v1 = case_group[case_group["profile"] == reference_profile].iloc[0]
        v2 = group.iloc[0]
        row = {"case_id": case_id, "profile": profile, "reference_profile": reference_profile}
        for key in (
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
            "pump_backlog_p95_kg",
        ):
            a = float(v1[key])
            b = float(v2[key])
            row[f"{key}_v1"] = a
            row[f"{key}_v2"] = b
            row[f"d_{key}"] = b - a
            denom = max(abs(a), 1e-9)
            row[f"d_{key}_pct"] = (b - a) / denom * 100.0
        rows.append(row)
    return pd.DataFrame(rows)


def _rolling_median(values: np.ndarray, dt_s: float, window_s: float) -> np.ndarray:
    window = max(1, int(round(float(window_s) / max(float(dt_s), 1e-9))))
    return pd.Series(values).rolling(window=window, center=True, min_periods=1).median().to_numpy(dtype=float)


def _plot_case(case_id: str, label: str, traces: dict[str, pd.DataFrame], out_dir: Path) -> Path | None:
    if not {"v1_engineered_minimal", "v2_actuator_smoothed_lb"}.issubset(traces):
        return None
    v1 = traces["v1_engineered_minimal"]
    v2 = traces["v2_actuator_smoothed_lb"]
    t = v1["t_s"].to_numpy(dtype=float) / 60.0
    dt_s = float(np.nanmedian(np.diff(v1["t_s"].to_numpy(dtype=float)))) if len(v1) > 1 else 1.0

    fig, axes = plt.subplots(
        5,
        1,
        figsize=(14.0, 10.8),
        sharex=True,
        constrained_layout=True,
        gridspec_kw={"height_ratios": [0.8, 1.0, 1.15, 0.9, 0.9]},
    )
    c1 = "#2f6f9f"
    c2 = "#d85c27"

    ax = axes[0]
    ax.plot(t, v1["wind_speed"], color="#1d70b8", lw=1.3)
    ax.set_ylabel("Wind\n(m/s)")
    ax.grid(True, color="#d7dce2", lw=0.7)
    ax2 = ax.twinx()
    ax2.plot(t, v1["wind_dir_deg"], color="#45515f", lw=0.9, alpha=0.75)
    ax2.set_ylabel("Dir\n(deg)")

    ax = axes[1]
    ax.plot(t, v1["pitch_deg"], color=c1, lw=1.1, label="v1 pitch")
    ax.plot(t, v2["pitch_deg"], color=c2, lw=1.1, label="v2 pitch")
    ax.plot(t, v1["roll_deg"], color=c1, lw=0.9, ls="--", alpha=0.82, label="v1 roll")
    ax.plot(t, v2["roll_deg"], color=c2, lw=0.9, ls="--", alpha=0.82, label="v2 roll")
    ax.axhline(0.0, color="#777", lw=0.7)
    ax.set_ylabel("Attitude\n(deg)")
    ax.legend(ncol=4, frameon=False, fontsize=8, loc="upper left")
    ax.grid(True, color="#d7dce2", lw=0.7)

    ax = axes[2]
    p1 = v1["pump_total_rate_m3_min"].to_numpy(dtype=float)
    p2 = v2["pump_total_rate_m3_min"].to_numpy(dtype=float)
    ax.plot(t, p1, color=c1, lw=0.55, alpha=0.23, label="v1 raw")
    ax.plot(t, p2, color=c2, lw=0.55, alpha=0.23, label="v2 raw")
    ax.plot(t, _rolling_median(p1, dt_s, 30.0), color=c1, lw=1.55, label="v1 30s median")
    ax.plot(t, _rolling_median(p2, dt_s, 30.0), color=c2, lw=1.55, label="v2 30s median")
    ax.set_ylabel("Pump total\n(m3/min)")
    ax.legend(ncol=4, frameon=False, fontsize=8, loc="upper left")
    ax.grid(True, color="#d7dce2", lw=0.7)

    ax = axes[3]
    for tank in (1, 2, 3):
        ax.step(t, v1[f"pump_stage_idx{tank}"], where="post", color=c1, lw=0.8, alpha=0.40)
        ax.step(t, v2[f"pump_stage_idx{tank}"], where="post", color=c2, lw=0.9, alpha=0.55)
    ax.set_ylabel("Visible\nstage idx")
    ax.grid(True, color="#d7dce2", lw=0.7)

    ax = axes[4]
    for idx, (profile, df, color) in enumerate(
        [("v1", v1, c1), ("v2", v2, c2)]
    ):
        y0 = idx * 1.2
        active = df["pump_total_rate_m3_min"].to_numpy(dtype=float) > 1e-6
        ax.fill_between(t, y0, y0 + 0.85, where=active, step="post", color=color, alpha=0.68)
        ax.text(t[0], y0 + 0.43, profile, va="center", ha="right", fontsize=8)
    ax.set_yticks([])
    ax.set_ylabel("Pump\nactive")
    ax.set_xlabel("Time (min)")
    ax.grid(True, axis="x", color="#d7dce2", lw=0.7)

    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle(f"{case_id}: {label} | v1 vs actuator-smoothed v2", fontsize=13, y=1.01)
    out = out_dir / f"{case_id}_v1_vs_v2_actuator_smoothing.png"
    fig.savefig(out, dpi=190, bbox_inches="tight")
    plt.close(fig)
    return out


def _plot_delta_overview(delta: pd.DataFrame, out_dir: Path) -> Path:
    if "profile" in delta.columns:
        delta = delta[delta["profile"] == "v2_actuator_smoothed_lb"].copy()
    delta = delta.sort_values("case_id")
    x = np.arange(len(delta))
    labels = delta["case_id"].tolist()
    fig, axes = plt.subplots(4, 1, figsize=(13.6, 9.8), sharex=True, constrained_layout=True)
    axes[0].bar(x, delta["d_visible_stage_idx_changes_pct"], color="#7a5195")
    axes[0].axhline(-80.0, color="#126b44", ls="--", lw=0.9)
    axes[0].set_ylabel("Visible stage\nchange (%)")
    axes[1].bar(x, delta["d_rate_tv_per_min_pct"], color="#1b7f79")
    axes[1].set_ylabel("Rate TV/min\nchange (%)")
    axes[2].bar(x, delta["d_pump_work_m3_pct"], color="#d08c2f")
    axes[2].axhline(2.0, color="#9c2f2f", ls="--", lw=0.8)
    axes[2].axhline(-2.0, color="#9c2f2f", ls="--", lw=0.8)
    axes[2].set_ylabel("Pump work\nchange (%)")
    axes[3].bar(x - 0.18, delta["d_pitch_abs_p95"], width=0.36, color="#2f6f9f", label="pitch p95")
    axes[3].bar(x + 0.18, delta["d_roll_abs_p95"], width=0.36, color="#d85c27", label="roll p95")
    axes[3].axhline(0.05, color="#9c2f2f", ls="--", lw=0.8)
    axes[3].axhline(-0.05, color="#9c2f2f", ls="--", lw=0.8)
    axes[3].set_ylabel("Attitude p95\nchange (deg)")
    axes[3].legend(frameon=False, ncol=2, loc="upper left")
    axes[3].set_xticks(x)
    axes[3].set_xticklabels(labels, rotation=32, ha="right")
    for ax in axes:
        ax.grid(True, color="#d7dce2", lw=0.7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.suptitle("Closed actuator smoothing v2: raw-metric deltas vs v1", fontsize=13, y=1.01)
    out = out_dir / "closed_actuator_smoothing_v2_delta_overview.png"
    fig.savefig(out, dpi=190, bbox_inches="tight")
    plt.close(fig)
    return out


def _verdict(delta: pd.DataFrame) -> dict:
    if delta.empty:
        return {"decision": "NO_DATA", "reason": "no paired v1/v2 rows"}
    if "profile" in delta.columns:
        delta = delta[delta["profile"] == "v2_actuator_smoothed_lb"].copy()
    if delta.empty:
        return {"decision": "NO_DATA", "reason": "no v2_actuator_smoothed_lb paired rows"}
    mean_stage = float(delta["d_visible_stage_idx_changes_pct"].mean())
    mean_latch = float(delta["d_latch_switches_pct"].mean())
    max_pitch_p95 = float(delta["d_pitch_abs_p95"].max())
    max_roll_p95 = float(delta["d_roll_abs_p95"].max())
    max_abs_pump = float(np.max(np.abs(delta["d_pump_work_m3_pct"].to_numpy(dtype=float))))
    mean_stage_abs_v2 = float(delta["visible_stage_idx_changes_v2"].mean())
    lowrisk_ok = True
    lowrisk = delta[delta["case_id"].str.contains("08_lowrisk", regex=False)]
    if not lowrisk.empty:
        lowrisk_ok = bool(
            float(lowrisk["pump_work_m3_v2"].iloc[0]) <= 1e-9
            and int(round(float(lowrisk["visible_stage_idx_changes_v2"].iloc[0]))) == 0
        )
    pass_core = (
        mean_stage <= -80.0
        and mean_stage_abs_v2 < 300.0
        and max_pitch_p95 <= 0.05
        and max_roll_p95 <= 0.05
        and max_abs_pump <= 2.0
        and lowrisk_ok
    )
    return {
        "decision": "PASS" if pass_core else "REVIEW",
        "mean_visible_stage_change_delta_pct": mean_stage,
        "mean_latch_switch_delta_pct": mean_latch,
        "mean_visible_stage_changes_v2": mean_stage_abs_v2,
        "max_pitch_p95_delta_deg": max_pitch_p95,
        "max_roll_p95_delta_deg": max_roll_p95,
        "max_abs_pump_work_delta_pct": max_abs_pump,
        "lowrisk_quiet_unchanged": bool(lowrisk_ok),
        "criteria": {
            "mean_visible_stage_change_delta_pct": "<= -80",
            "mean_visible_stage_changes_v2": "< 300 per 2h selected cases",
            "max_pitch_or_roll_p95_delta_deg": "<= +0.05",
            "max_abs_pump_work_delta_pct": "<= 2",
            "lowrisk_quiet": "zero pump and zero stage changes",
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        default="outputs/wind_prediction/closed_actuator_smoothing_v2",
        help="Output directory for timeseries, figures, and summary tables.",
    )
    parser.add_argument(
        "--case-ids",
        default=DEFAULT_CASE_IDS,
        help="Comma-separated case ids. Default runs 01,04,06,08,09.",
    )
    parser.add_argument("--duration-s", type=float, default=7200.0)
    parser.add_argument("--wind-transition-s", type=float, default=0.0)
    parser.add_argument(
        "--dataset-dir",
        default="data/processed/wind_ml_10min/ballast_decision_fino1_meteo_aux_v1",
    )
    parser.add_argument(
        "--variant-set",
        choices=("v2", "oat"),
        default="v2",
        help="v2 runs only reference plus v2; oat runs one-at-a-time actuator variants.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    start = time.perf_counter()
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = REPO_ROOT / out_dir
    ts_dir = out_dir / "timeseries"
    fig_dir = out_dir / "figures"
    for d in (out_dir, ts_dir, fig_dir):
        d.mkdir(parents=True, exist_ok=True)

    excel_path = discover_excel()
    dataset_dir = Path(args.dataset_dir)
    if not dataset_dir.is_absolute():
        dataset_dir = REPO_ROOT / dataset_dir
    replay = Fino1ReplayDataset(dataset_dir=dataset_dir, split="test")

    from defaults import clone_cfg
    from run_validation import run_closed_loop_case

    n_steps = int(round(float(args.duration_s) / DT))
    summary_rows: list[dict] = []
    fig_paths: list[Path] = []
    issues: list[str] = []
    profile_cfgs = PROFILE_CFGS if str(args.variant_set) == "v2" else OAT_PROFILE_CFGS
    for case_id, ts_str, label in _selected_cases(args.case_ids):
        ts = datetime.strptime(ts_str, TIMESTAMP_FMT)
        try:
            row_count = int(np.ceil(n_steps * DT / replay.update_interval_s))
            wind_trace = replay.build_wind_trace(start_timestamp=ts, row_count=row_count, dt_s=DT)
            wind_trace = smooth_wind_trace(wind_trace, float(args.wind_transition_s))
        except Exception as exc:
            issues.append(f"{case_id}: wind trace failed: {exc}")
            continue
        traces: dict[str, pd.DataFrame] = {}
        for profile, cfg in profile_cfgs.items():
            run_name = f"{case_id}_{ts_str.replace(':', '').replace(' ', '_')}_{profile}"
            print(f"--- {run_name} ---", flush=True)
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
                pump_cfg=clone_cfg(cfg),
                target_shape_override_cfg={},
            )
            df = pd.DataFrame(timeseries)
            df.to_csv(ts_dir / f"{run_name}_timeseries.csv", index=False)
            traces[profile] = df
            summary_rows.append(_summarize(case_id, profile, df))
        case_fig = _plot_case(case_id, label, traces, fig_dir)
        if case_fig is not None:
            fig_paths.append(case_fig)

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out_dir / "closed_actuator_smoothing_v2_summary.csv", index=False)
    delta = _delta_rows(summary)
    delta.to_csv(out_dir / "closed_actuator_smoothing_v2_deltas.csv", index=False)
    if not delta.empty:
        fig_paths.insert(0, _plot_delta_overview(delta, fig_dir))
    verdict = _verdict(delta)
    meta = {
        "duration_s": float(args.duration_s),
        "case_ids": str(args.case_ids),
        "variant_set": str(args.variant_set),
        "profiles": profile_cfgs,
        "metric_layer": "raw 1Hz timeseries only",
        "plot_layer": "30s rolling median overlay used only for visualization",
        "verdict": verdict,
        "issues": issues,
        "runtime_s": float(time.perf_counter() - start),
    }
    (out_dir / "closed_actuator_smoothing_v2_meta.json").write_text(
        json.dumps(meta, indent=2, allow_nan=True),
        encoding="utf-8",
    )
    lines = [
        "# Closed Actuator Smoothing v2",
        "",
        "Purpose: evaluate a normal actuator-layer anti-chatter profile without changing PI, forecast, or planner logic.",
        "",
        "Metric rule: all metrics use raw 1 Hz timeseries. Smoothed curves are visualization only.",
        "",
        "v2 lower-bound profile:",
        "",
        *[
            f"- `{key}` = `{value}`"
            for key, value in CLOSED_ACTUATOR_SMOOTHING_CANDIDATE_PUMP_CFG.items()
            if key not in CLOSED_BASELINE_V1_PUMP_CFG
            or CLOSED_BASELINE_V1_PUMP_CFG.get(key) != value
        ],
        "",
        "Verdict:",
        "",
        f"- decision: `{verdict['decision']}`",
        f"- mean visible stage change delta: `{verdict.get('mean_visible_stage_change_delta_pct', float('nan')):.2f}%`",
        f"- max |pump work delta|: `{verdict.get('max_abs_pump_work_delta_pct', float('nan')):.2f}%`",
        f"- max pitch p95 delta: `{verdict.get('max_pitch_p95_delta_deg', float('nan')):.4f} deg`",
        f"- max roll p95 delta: `{verdict.get('max_roll_p95_delta_deg', float('nan')):.4f} deg`",
        "",
        "Files:",
        "",
        f"- `closed_actuator_smoothing_v2_summary.csv`",
        f"- `closed_actuator_smoothing_v2_deltas.csv`",
        f"- `closed_actuator_smoothing_v2_meta.json`",
        *[f"- `{p.relative_to(out_dir)}`" for p in fig_paths],
    ]
    if issues:
        lines.extend(["", "Issues:", "", *[f"- {item}" for item in issues]])
    (out_dir / "closed_actuator_smoothing_v2_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(out_dir)


if __name__ == "__main__":
    main()
