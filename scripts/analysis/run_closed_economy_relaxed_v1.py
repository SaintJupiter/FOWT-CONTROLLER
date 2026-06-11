#!/usr/bin/env python3
"""Relaxed no-preview closed economy check.

This experiment keeps the reactive PI controller and all preview/planner logic
unchanged. It only relaxes actuator-layer pump stop/restart thresholds to test
whether a normal non-preview controller can trade a small attitude penalty for
less pump work and less visible chatter.

Metrics are computed from raw 1 Hz timeseries. Rolling medians are used only in
figures.
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


PROFILE_CFGS: dict[str, dict] = {
    "v1_engineered_minimal": CLOSED_BASELINE_V1_PUMP_CFG,
    "v2_actuator_smoothed_lb": CLOSED_ACTUATOR_SMOOTHING_CANDIDATE_PUMP_CFG,
    "econ_stop500_restart850": {
        **CLOSED_BASELINE_V1_PUMP_CFG,
        "pump_stop_err_kg": 500.0,
        "pump_restart_err_kg": 850.0,
        "pump_min_off_s": 20.0,
        "pump_hold_before_stop_s": 15.0,
    },
    "econ_stop700_restart1100": {
        **CLOSED_BASELINE_V1_PUMP_CFG,
        "pump_stop_err_kg": 700.0,
        "pump_restart_err_kg": 1100.0,
        "pump_min_off_s": 20.0,
        "pump_hold_before_stop_s": 15.0,
    },
    "econ_stop900_restart1400": {
        **CLOSED_BASELINE_V1_PUMP_CFG,
        "pump_stop_err_kg": 900.0,
        "pump_restart_err_kg": 1400.0,
        "pump_min_off_s": 25.0,
        "pump_hold_before_stop_s": 15.0,
    },
    "econ_stop1100_restart1700": {
        **CLOSED_BASELINE_V1_PUMP_CFG,
        "pump_stop_err_kg": 1100.0,
        "pump_restart_err_kg": 1700.0,
        "pump_min_off_s": 30.0,
        "pump_hold_before_stop_s": 20.0,
    },
    "econ_stop700_restart1100_ramp": {
        **CLOSED_BASELINE_V1_PUMP_CFG,
        "pump_stop_err_kg": 700.0,
        "pump_restart_err_kg": 1100.0,
        "pump_min_off_s": 20.0,
        "pump_hold_before_stop_s": 15.0,
        "pump_ramp_up_m3_min_per_s": 0.5,
        "pump_ramp_down_m3_min_per_s": 0.75,
    },
    "econ_stop700_restart1100_ramp1p0": {
        **CLOSED_BASELINE_V1_PUMP_CFG,
        "pump_stop_err_kg": 700.0,
        "pump_restart_err_kg": 1100.0,
        "pump_min_off_s": 20.0,
        "pump_hold_before_stop_s": 15.0,
        "pump_ramp_up_m3_min_per_s": 1.0,
        "pump_ramp_down_m3_min_per_s": 1.5,
    },
    "econ_stop700_restart1100_ramp1p5": {
        **CLOSED_BASELINE_V1_PUMP_CFG,
        "pump_stop_err_kg": 700.0,
        "pump_restart_err_kg": 1100.0,
        "pump_min_off_s": 20.0,
        "pump_hold_before_stop_s": 15.0,
        "pump_ramp_up_m3_min_per_s": 1.5,
        "pump_ramp_down_m3_min_per_s": 2.25,
    },
    "econ_stop900_restart1400_ramp": {
        **CLOSED_BASELINE_V1_PUMP_CFG,
        "pump_stop_err_kg": 900.0,
        "pump_restart_err_kg": 1400.0,
        "pump_min_off_s": 25.0,
        "pump_hold_before_stop_s": 15.0,
        "pump_ramp_up_m3_min_per_s": 0.5,
        "pump_ramp_down_m3_min_per_s": 0.75,
    },
    "econ_stop1100_restart1700_ramp": {
        **CLOSED_BASELINE_V1_PUMP_CFG,
        "pump_stop_err_kg": 1100.0,
        "pump_restart_err_kg": 1700.0,
        "pump_min_off_s": 30.0,
        "pump_hold_before_stop_s": 20.0,
        "pump_ramp_up_m3_min_per_s": 0.5,
        "pump_ramp_down_m3_min_per_s": 0.75,
    },
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


def _rate_tv_per_min(df: pd.DataFrame) -> float:
    pump = df["pump_total_rate_m3_min"].to_numpy(dtype=float)
    t = df["t_s"].to_numpy(dtype=float)
    duration_min = max(float(t[-1] - t[0]) / 60.0, 1e-9) if t.size > 1 else 1e-9
    return float(np.sum(np.abs(np.diff(pump))) / duration_min)


def _summary_row(case_id: str, profile: str, df: pd.DataFrame) -> dict:
    pump = df["pump_total_rate_m3_min"].to_numpy(dtype=float)
    pitch = df["pitch_deg"].to_numpy(dtype=float)
    roll = df["roll_deg"].to_numpy(dtype=float)
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
        "pump_backlog_p95_kg": float(np.percentile(df["pump_total_backlog_kg"], 95))
        if "pump_total_backlog_kg" in df.columns
        else np.nan,
    }


def _delta_rows(summary: pd.DataFrame, reference_profile: str = "v1_engineered_minimal") -> pd.DataFrame:
    rows = []
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
                a = float(b[key])
                c = float(r[key])
                row[f"{key}_v1"] = a
                row[f"{key}_profile"] = c
                row[f"d_{key}"] = c - a
                row[f"d_{key}_pct"] = (c - a) / max(abs(a), 1e-9) * 100.0
            row["pump_saving_pct"] = -row["d_pump_work_m3_pct"]
            rows.append(row)
    return pd.DataFrame(rows)


def _aggregate(delta: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for profile, group in delta.groupby("profile", sort=False):
        if profile == "v1_engineered_minimal":
            continue
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
                "mean_pump_saving_pct": float(group["pump_saving_pct"].mean()),
                "median_pump_saving_pct": float(group["pump_saving_pct"].median()),
                "min_case_pump_saving_pct": float(group["pump_saving_pct"].min()),
                "mean_latch_delta_pct": float(group["d_latch_switches_pct"].mean()),
                "mean_visible_stage_delta_pct": float(group["d_visible_stage_idx_changes_pct"].mean()),
                "mean_rate_tv_delta_pct": float(group["d_rate_tv_per_min_pct"].mean()),
                "max_pitch_p95_delta_deg": float(group["d_pitch_abs_p95"].max()),
                "max_roll_p95_delta_deg": float(group["d_roll_abs_p95"].max()),
                "max_pitch_abs_delta_deg": float(group["d_pitch_abs_max"].max()),
                "max_roll_abs_delta_deg": float(group["d_roll_abs_max"].max()),
                "lowrisk_unchanged": int(lowrisk_ok),
                "valid_relaxed": int(
                    lowrisk_ok
                    and group["d_pitch_abs_p95"].max() <= 0.50
                    and group["d_roll_abs_p95"].max() <= 0.35
                    and group["d_pitch_abs_max"].max() <= 1.00
                    and group["d_roll_abs_max"].max() <= 0.80
                ),
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(
        ["valid_relaxed", "mean_pump_saving_pct", "mean_rate_tv_delta_pct"],
        ascending=[False, False, True],
    )


def _choose_profile(aggregate: pd.DataFrame) -> str:
    valid = aggregate[(aggregate["valid_relaxed"] == 1) & (aggregate["mean_pump_saving_pct"] > 0.1)]
    if valid.empty:
        return "v1_engineered_minimal"
    return str(valid.iloc[0]["profile"])


def _plot_case_compare(
    case_id: str,
    label: str,
    traces: dict[str, pd.DataFrame],
    profiles: list[str],
    out_dir: Path,
    suffix: str,
) -> Path:
    colors = {
        "v1_engineered_minimal": "#2f6f9f",
        "v2_actuator_smoothed_lb": "#d85c27",
        "econ_stop500_restart850": "#3f8f4f",
        "econ_stop700_restart1100": "#126b44",
        "econ_stop900_restart1400": "#7a5195",
        "econ_stop1100_restart1700": "#8d4d2e",
        "econ_stop700_restart1100_ramp": "#0f766e",
        "econ_stop900_restart1400_ramp": "#8b5cf6",
        "econ_stop1100_restart1700_ramp": "#a16207",
    }
    profiles = [p for p in profiles if p in traces]
    if not profiles:
        raise ValueError("no profiles available for plot")
    ref = traces[profiles[0]]
    t = ref["t_s"].to_numpy(dtype=float) / 60.0
    fig, axes = plt.subplots(
        4,
        1,
        figsize=(14.0, 9.8),
        sharex=True,
        constrained_layout=True,
        gridspec_kw={"height_ratios": [1.1, 0.9, 0.95, 0.8]},
    )

    ax = axes[0]
    for profile in profiles:
        df = traces[profile]
        work = _work_m3(df)
        ax.plot(
            t,
            _cumulative_work_m3(df),
            lw=1.8,
            color=colors.get(profile, None),
            label=f"{profile} ({work:.0f} m3)",
        )
    ax.set_ylabel("Cumulative\npump work (m3)")
    ax.legend(frameon=False, ncol=2, fontsize=8, loc="upper left")
    ax.grid(True, color="#d7dce2", lw=0.7)

    ax = axes[1]
    for profile in profiles:
        df = traces[profile]
        pump = df["pump_total_rate_m3_min"].to_numpy(dtype=float)
        ax.plot(t, _rolling_median(pump, 30.0), lw=1.25, color=colors.get(profile, None), label=profile)
    ax.set_ylabel("Pump rate\n30s median")
    ax.grid(True, color="#d7dce2", lw=0.7)

    ax = axes[2]
    for profile in profiles:
        df = traces[profile]
        att = np.maximum(
            np.abs(df["pitch_deg"].to_numpy(dtype=float)),
            np.abs(df["roll_deg"].to_numpy(dtype=float)),
        )
        ax.plot(t, att, lw=1.25, color=colors.get(profile, None), label=profile)
    ax.axhline(4.0, color="#9c2f2f", lw=0.8, ls="--", alpha=0.7)
    ax.set_ylabel("Max(|pitch|,|roll|)\n(deg)")
    ax.grid(True, color="#d7dce2", lw=0.7)

    ax = axes[3]
    for idx, profile in enumerate(profiles):
        df = traces[profile]
        active = np.abs(df["pump_total_rate_m3_min"].to_numpy(dtype=float)) > 1e-6
        y0 = idx * 1.0
        ax.fill_between(
            t,
            y0,
            y0 + 0.72,
            where=active,
            step="post",
            color=colors.get(profile, None),
            alpha=0.72,
        )
        ax.text(t[0], y0 + 0.36, profile, va="center", ha="right", fontsize=8)
    ax.set_yticks([])
    ax.set_ylabel("Pump active")
    ax.set_xlabel("Time (min)")
    ax.grid(True, axis="x", color="#d7dce2", lw=0.7)

    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.suptitle(f"{case_id}: {label} | cumulative pump comparison", fontsize=13, y=1.01)
    out = out_dir / f"{case_id}_{suffix}_cumulative_pump_comparison.png"
    fig.savefig(out, dpi=190, bbox_inches="tight")
    plt.close(fig)
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        default="outputs/wind_prediction/closed_economy_relaxed_v1",
    )
    parser.add_argument("--case-ids", default=DEFAULT_CASE_IDS)
    parser.add_argument("--duration-s", type=float, default=DEFAULT_DURATION_S)
    parser.add_argument(
        "--profiles",
        default="",
        help="Comma-separated profile names. Empty means all profiles.",
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
    summary_rows: list[dict] = []
    traces_by_case: dict[str, dict[str, pd.DataFrame]] = {}
    labels: dict[str, str] = {}

    for case_id, ts_str, label in _selected_cases(args.case_ids):
        labels[case_id] = label
        ts = datetime.strptime(ts_str, TIMESTAMP_FMT)
        row_count = int(np.ceil(n_steps * DT / replay.update_interval_s))
        wind_trace = replay.build_wind_trace(start_timestamp=ts, row_count=row_count, dt_s=DT)
        wind_trace = smooth_wind_trace(wind_trace, 0.0)
        traces_by_case[case_id] = {}
        wanted_profiles = {p.strip() for p in str(args.profiles).split(",") if p.strip()}
        profile_items = [
            (profile, cfg)
            for profile, cfg in PROFILE_CFGS.items()
            if not wanted_profiles or profile in wanted_profiles
        ]
        for profile, cfg in profile_items:
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
            traces_by_case[case_id][profile] = df
            summary_rows.append(_summary_row(case_id, profile, df))

    summary = pd.DataFrame(summary_rows)
    delta = _delta_rows(summary)
    aggregate = _aggregate(delta)
    selected = _choose_profile(aggregate)
    fig_paths: list[Path] = []
    for case_id in ("01_onset_strong", "04_decay_strong"):
        if case_id in traces_by_case:
            fig_paths.append(
                _plot_case_compare(
                    case_id,
                    labels.get(case_id, case_id),
                    traces_by_case[case_id],
                    ["v1_engineered_minimal", "v2_actuator_smoothed_lb"],
                    fig_dir,
                    "v1_vs_v2",
                )
            )
            compare_profiles = ["v1_engineered_minimal", "v2_actuator_smoothed_lb"]
            if selected not in compare_profiles and selected in traces_by_case[case_id]:
                compare_profiles.append(selected)
            if len(compare_profiles) > 2:
                fig_paths.append(
                    _plot_case_compare(
                        case_id,
                        labels.get(case_id, case_id),
                        traces_by_case[case_id],
                        compare_profiles,
                        fig_dir,
                        "v1_v2_selected_economy",
                    )
                )

    summary.to_csv(out_dir / "closed_economy_relaxed_summary.csv", index=False)
    delta.to_csv(out_dir / "closed_economy_relaxed_deltas.csv", index=False)
    aggregate.to_csv(out_dir / "closed_economy_relaxed_aggregate.csv", index=False)
    meta = {
        "duration_s": float(args.duration_s),
        "case_ids": str(args.case_ids),
        "selected_profile": selected,
        "profiles": PROFILE_CFGS,
        "valid_relaxed_constraints": {
            "max_pitch_p95_delta_deg": 0.50,
            "max_roll_p95_delta_deg": 0.35,
            "max_pitch_abs_delta_deg": 1.00,
            "max_roll_abs_delta_deg": 0.80,
            "lowrisk": "unchanged",
        },
        "metric_layer": "raw 1Hz timeseries only",
        "plot_layer": "rolling median for pump-rate visualization only; cumulative pump uses raw integration",
        "runtime_s": float(time.perf_counter() - t0),
        "figures": [str(p.relative_to(out_dir)) for p in fig_paths],
    }
    (out_dir / "closed_economy_relaxed_meta.json").write_text(
        json.dumps(meta, indent=2, allow_nan=True),
        encoding="utf-8",
    )
    lines = [
        "# Closed Economy Relaxed v1",
        "",
        "Purpose: test whether a no-preview closed controller can trade a small attitude penalty for lower pump work without changing PI gains or using forecast/planner information.",
        "",
        f"- selected profile: `{selected}`",
        f"- duration: `{float(args.duration_s) / 3600.0:.1f} h`",
        f"- cases: `{args.case_ids}`",
        "- metrics: raw 1 Hz data",
        "",
        "## Aggregate",
        "",
        "| profile | mean saving | min case saving | mean rate TV delta | max pitch p95 delta | max roll p95 delta | valid |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for _, r in aggregate.iterrows():
        lines.append(
            f"| {r['profile']} | {r['mean_pump_saving_pct']:.2f}% | {r['min_case_pump_saving_pct']:.2f}% | {r['mean_rate_tv_delta_pct']:.2f}% | {r['max_pitch_p95_delta_deg']:.3f} | {r['max_roll_p95_delta_deg']:.3f} | {int(r['valid_relaxed'])} |"
        )
    lines.extend(["", "## Figures", ""])
    lines.extend([f"- `{p.relative_to(out_dir)}`" for p in fig_paths])
    (out_dir / "closed_economy_relaxed_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(out_dir)


if __name__ == "__main__":
    main()
