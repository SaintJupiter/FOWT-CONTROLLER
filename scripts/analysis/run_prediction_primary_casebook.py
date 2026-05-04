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

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root / "src"))
sys.path.insert(0, str(repo_root / "archive" / "legacy_fowt_control"))

from wind_prediction.ballast_planner import PlannerConfig
from wind_prediction.ballast_planner_provider import BallastPlannerPreviewProvider
from wind_prediction.replay_dataset import Fino1ReplayDataset


DT = 1.0
N_STEPS = 3600
TIMESTAMP_FMT = "%Y-%m-%d %H:%M:%S"


# Presentation-only reactive baseline refinements. These parameters reuse the
# already-implemented pump-side stop/start and stage hysteresis layers, while
# leaving the prediction-primary provider untouched.
CLOSED_PUMP_PROFILES = {
    "raw": None,
    "smooth": {
        "pump_stop_err_kg": 225.0,
        "pump_restart_err_kg": 725.0,
        "pump_min_on_s": 30.0,
        "pump_min_off_s": 12.0,
        "pump_hold_before_stop_s": 20.0,
        "pump_global_quiet_hold_s": 20.0,
        "pump_global_quiet_backlog_kg": 800.0,
        "pump_target_quiet_rate_kg_s": 20.0,
        "pump_ramp_up_m3_min_per_s": 2.0,
        "pump_ramp_down_m3_min_per_s": 3.0,
        "pump_stage_hysteresis_kg": 75.0,
        "pump_stage_min_dwell_s": 10.0,
        "pump_low_end_stage_hysteresis_kg": 125.0,
        "pump_low_end_stage_min_dwell_s": 20.0,
        "pump_low_end_stage_max_idx": 2,
    },
}

CLOSED_TARGET_SHAPE_PROFILES = {
    "raw": {},
    "smooth": {
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
    t_closed = closed["t_s"].to_numpy(dtype=float) / 60.0
    t_primary = primary["t_s"].to_numpy(dtype=float) / 60.0
    c = "#2468B2"
    p = "#D85C27"
    fig_w = max(12.5, min(24.0, 12.5 * duration_min / 60.0))
    fig, axes = plt.subplots(3, 1, figsize=(fig_w, 8.4), sharex=True, constrained_layout=True)
    rows = [
        ("Pitch angle (deg)", "pitch_deg"),
        ("Roll angle (deg)", "roll_deg"),
        ("Total pump rate (m3/min)", "pump_total_rate_m3_min"),
    ]
    for ax, (ylabel, col) in zip(axes, rows):
        ax.plot(t_closed, closed[col].to_numpy(dtype=float), color=c, lw=1.45, label="closed_only")
        ax.plot(t_primary, primary[col].to_numpy(dtype=float), color=p, lw=1.45, label=primary_label)
        ax.axhline(0.0, color="#777777", lw=0.8, alpha=0.55)
        for x in np.arange(20.0, duration_min, 20.0):
            ax.axvline(x, color="#999999", lw=0.9, ls="--", alpha=0.55)
        ax.set_ylabel(ylabel)
        ax.grid(True, color="#d9d9d9", lw=0.7, alpha=0.75)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].legend(loc="upper left", frameon=False, ncol=2)
    _annotate_planner(axes[0], planner_log, duration_min)
    axes[2].set_xlabel("Time (min)")
    axes[2].set_xlim(0, duration_min)
    sc = summarize(closed)
    sp = summarize(primary)
    metric_text = (
        f"pump: {sc['pump_work_m3']:.1f} -> {sp['pump_work_m3']:.1f} m3 "
        f"({(sp['pump_work_m3'] - sc['pump_work_m3']) / max(sc['pump_work_m3'], 1e-9) * 100:+.1f}%)\n"
        f"pitch p95: {sc['pitch_abs_p95']:.2f} -> {sp['pitch_abs_p95']:.2f} deg; "
        f"roll p95: {sc['roll_abs_p95']:.2f} -> {sp['roll_abs_p95']:.2f} deg"
    )
    axes[0].text(
        0.995,
        0.03,
        metric_text,
        transform=axes[0].transAxes,
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
        "--wind-transition-s",
        type=float,
        default=0.0,
        help="Raised-cosine transition duration for replay wind bucket changes. 0 keeps hard 10-min steps.",
    )
    parser.add_argument(
        "--closed-pump-profile",
        choices=sorted(CLOSED_PUMP_PROFILES),
        default="smooth",
        help="Pump-side profile for closed_only only. 'raw' preserves the old reactive baseline.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
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
    fig_dir = out_dir / "figures"
    ts_dir = out_dir / "timeseries"
    log_dir = out_dir / "planner_logs"
    for d in (fig_dir, ts_dir, log_dir):
        d.mkdir(parents=True, exist_ok=True)

    base_planner_out = repo_root / "outputs" / "wind_prediction" / "planner_a1_dryrun"
    sign_cfg = json.loads((base_planner_out / "diagnostics" / "a01_pressure_vec_sign_convention.json").read_text())
    discount_cfg = json.loads((base_planner_out / "diagnostics" / "a1_block_discount_config.json").read_text())
    cfg = PlannerConfig(pressure_sign_multiplier=float(sign_cfg["planner_sign_multiplier"]))
    discounts = discount_cfg["default_discount_blocks"]
    excel_path = discover_excel()
    replay = Fino1ReplayDataset(
        dataset_dir=repo_root / "data" / "processed" / "wind_ml_10min" / "ballast_decision_fino1_meteo_aux_v1",
        split="test",
    )

    from defaults import clone_cfg  # noqa: F401
    from run_validation import run_closed_loop_case

    rows: list[dict] = []
    issues: list[str] = []
    indexed_cases = [
        (idx, raw_case_id, ts_str, label)
        for idx, (raw_case_id, ts_str, label) in enumerate(CASES[:10], start=1)
    ]
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
        case_id = f"{case_num:02d}_{raw_case_id.split('_', 1)[1]}"
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
        for variant in ("closed_only", primary_variant):
            provider = None
            if variant == primary_variant:
                provider = BallastPlannerPreviewProvider(
                    replay_dataset=replay,
                    start_timestamp=ts,
                    cfg=cfg,
                    block_discounts=discounts,
                    bias_shape="event_decay",
                    objective_mode="economic",
                    setpoint_channel_enabled=False,
                    prediction_primary_enabled=True,
                    prediction_primary_scale=1.0,
                    ff_channel_enabled=False,
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
                    target_shape_override_cfg=closed_target_shape_cfg if variant == "closed_only" else None,
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

        if "closed_only" not in dfs or primary_variant not in dfs:
            continue

        planner_log = None
        provider = providers.get(primary_variant)
        if provider is not None and provider.records:
            planner_log = pd.DataFrame(provider.records)
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
        sc = summarize(dfs["closed_only"])
        sp = summarize(dfs[primary_variant])
        row = {
            "case_id": case_id,
            "timestamp": ts_str,
            "label": label,
            "figure": str(fig_path.relative_to(repo_root)),
            "closed_pump_work_m3": sc["pump_work_m3"],
            "primary_pump_work_m3": sp["pump_work_m3"],
            "d_pump_work_pct": (sp["pump_work_m3"] - sc["pump_work_m3"]) / max(sc["pump_work_m3"], 1e-9) * 100.0,
            "closed_pitch_p95": sc["pitch_abs_p95"],
            "primary_pitch_p95": sp["pitch_abs_p95"],
            "d_pitch_p95": sp["pitch_abs_p95"] - sc["pitch_abs_p95"],
            "closed_roll_p95": sc["roll_abs_p95"],
            "primary_roll_p95": sp["roll_abs_p95"],
            "d_roll_p95": sp["roll_abs_p95"] - sc["roll_abs_p95"],
            "closed_latch_switches": sc["latch_switches"],
            "primary_latch_switches": sp["latch_switches"],
            "primary_delta_mean_kg": sp["primary_delta_mean_kg"],
            "primary_candidate_ratio": sp["primary_candidate_ratio"],
            "primary_applied_ratio": sp["primary_applied_ratio"],
            "primary_safety_fallback_ratio": sp["primary_safety_fallback_ratio"],
            "primary_safety_active_ratio": sp["primary_safety_active_ratio"],
            "primary_safety_transition_count": sp["primary_safety_transition_count"],
            "closed_pump_profile": closed_profile,
        }
        rows.append(row)
        completed += 1
        print(
            f"   saved {fig_path.name}: pump {row['d_pump_work_pct']:+.1f}%, "
            f"d_pitch_p95={row['d_pitch_p95']:+.3f}, d_roll_p95={row['d_roll_p95']:+.3f}",
            flush=True,
        )

    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / "casebook_summary.csv", index=False)
    lines = [
        "# Prediction-primary casebook",
        "",
        f"- elapsed: `{time.perf_counter() - t0:.1f}s`",
        f"- completed cases: `{len(summary)}`",
        f"- variants: `closed_only` vs `{primary_variant}`",
        f"- closed-only pump profile: `{closed_profile}`",
        f"- duration per case: `{duration_min:.0f} min`",
        f"- replay wind transition: `{float(args.wind_transition_s):.1f}s raised-cosine`",
        "",
        "| case | label | pump delta | d_pitch_p95 | d_roll_p95 | safety fallback | latch switches | figure |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for _, r in summary.iterrows():
        lines.append(
            f"| {r['case_id']} | {r['label']} | {r['d_pump_work_pct']:+.1f}% | "
            f"{r['d_pitch_p95']:+.3f} | {r['d_roll_p95']:+.3f} | "
            f"{r['primary_safety_fallback_ratio'] * 100:.1f}% | "
            f"{int(r['closed_latch_switches'])}->{int(r['primary_latch_switches'])} | "
            f"`{r['figure']}` |"
        )
    lines += ["", "## Issues", ""]
    lines += [f"- {x}" for x in issues] if issues else ["None."]
    (out_dir / "casebook_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Report: {out_dir / 'casebook_report.md'}")


if __name__ == "__main__":
    main()
