#!/usr/bin/env python3
"""A2-onset: 4-mode comparison on transient-onset windows.

These are windows where block0_norm < 0.5 and block2_norm > 1.0 — i.e. the
60-min forecast says "calm now, ramp soon". This is the regime where a
preview planner can plausibly demonstrate predictive value over a purely
reactive closed_only controller.

Window selection: top 6 by growth_norm from
outputs/wind_prediction/transient_onset_scan/scan_top_onset.csv.
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root / "src"))
sys.path.insert(0, str(repo_root / "archive" / "legacy_fowt_control"))

from wind_prediction.ballast_planner import PlannerConfig
from wind_prediction.ballast_planner_provider import BallastPlannerPreviewProvider
from wind_prediction.replay_dataset import Fino1ReplayDataset


MODES = (
    ("closed_only",                  None,         None),
    ("closed_plus_lex_event_decay",  "event_decay", "lex"),
    ("closed_plus_econ_event_decay", "event_decay", "economic"),
)
TIMESTAMP_FMT = "%Y-%m-%d %H:%M:%S"
DT = 1.0
N_STEPS = 3600  # row_count=60 -> 10h
N_TOP_ONSET = 6


def discover_excel() -> str:
    for d in (repo_root / "data", repo_root / "archive" / "legacy_fowt_control" / "data"):
        for x in d.rglob("*.xlsx"):
            if "stiffness" in x.name.lower() or "matrix" in x.name.lower():
                return str(x)
    raise FileNotFoundError


def percentiles(x: np.ndarray) -> dict:
    a = np.abs(x)
    return {
        "abs_p50": float(np.percentile(a, 50)),
        "abs_p95": float(np.percentile(a, 95)),
        "abs_p99": float(np.percentile(a, 99)),
        "abs_max": float(a.max()),
        "rms": float(np.sqrt(np.mean(x ** 2))),
    }


def summarize_run(df: pd.DataFrame, dt: float) -> dict:
    n = len(df)
    pitch = df.get("pitch_deg", pd.Series(np.zeros(n))).values
    roll = df.get("roll_deg", pd.Series(np.zeros(n))).values
    pump_rate = df.get("pump_total_rate_m3_min", pd.Series(np.zeros(n))).abs().values
    out = {
        "n_steps": n,
        "any_nan": int(np.any(np.isnan(pitch)) or np.any(np.isnan(roll))),
        "pump_work_m3": float(np.trapezoid(pump_rate, dx=dt) / 60.0),
        "pump_duty_ratio": float(np.mean(pump_rate > 1e-6)),
    }
    out.update({f"pitch_{k}": v for k, v in percentiles(pitch).items()})
    out.update({f"roll_{k}": v for k, v in percentiles(roll).items()})
    out["envelope_pitch_violation_ratio"] = float(np.mean(np.abs(pitch) > 1.2))
    out["envelope_roll_violation_ratio"] = float(np.mean(np.abs(roll) > 0.96))
    # First-hour stats — onset is in [0, 60min].
    first_hour_n = int(min(3600 / dt, n))
    out["pitch_first_hour_p95"] = float(np.percentile(np.abs(pitch[:first_hour_n]), 95))
    out["pitch_first_hour_max"] = float(np.max(np.abs(pitch[:first_hour_n])))
    out["roll_first_hour_p95"] = float(np.percentile(np.abs(roll[:first_hour_n]), 95))
    out["roll_first_hour_max"] = float(np.max(np.abs(roll[:first_hour_n])))
    out["pump_work_first_hour"] = float(np.trapezoid(pump_rate[:first_hour_n], dx=dt) / 60.0)
    return out


def main() -> None:
    t0 = time.perf_counter()
    out_dir = repo_root / "outputs" / "wind_prediction" / "planner_a2_onset"
    out_dir.mkdir(parents=True, exist_ok=True)

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

    onset_df = pd.read_csv(repo_root / "outputs" / "wind_prediction" / "transient_onset_scan" / "scan_top_onset.csv")
    onset_df = onset_df.head(N_TOP_ONSET)
    print(f"Top {len(onset_df)} onset windows:\n{onset_df[['timestamp','block0_norm','block2_norm','speed_t0','speed_block2']]}")

    from defaults import clone_cfg  # noqa: F401
    from run_validation import run_closed_loop_case

    summary_rows: list[dict] = []
    issues: list[str] = []

    for _, win in onset_df.iterrows():
        ts_str = str(win["timestamp"])
        ts = datetime.strptime(ts_str, TIMESTAMP_FMT)
        try:
            wind_trace = replay.build_wind_trace(start_timestamp=ts, row_count=N_STEPS // 60, dt_s=DT)
        except ValueError as e:
            issues.append(f"{ts_str}: build_wind_trace failed ({e})")
            continue

        for mode_tag, bias_shape, obj_mode in MODES:
            tag = f"{ts_str.replace(':', '').replace(' ', '_')}_{mode_tag}"
            print(f"\n--- {ts_str} mode={mode_tag} ---")

            provider = None
            if bias_shape is not None and obj_mode is not None:
                provider = BallastPlannerPreviewProvider(
                    replay_dataset=replay,
                    start_timestamp=ts,
                    cfg=cfg,
                    block_discounts=discounts,
                    bias_shape=bias_shape,
                    objective_mode=obj_mode,
                )

            try:
                row, timeseries = run_closed_loop_case(
                    excel_path=excel_path,
                    case_name=tag,
                    dt=DT,
                    n_steps=int(wind_trace["n_steps"]),
                    wind_trace=wind_trace,
                    control_enabled=True,
                    record_timeseries=True,
                    experiment_protocol="main",
                    platform_profile="default",
                    start_from_heave_equilibrium=True,
                    preview_trim_provider=provider,
                )
            except Exception as e:
                issues.append(f"{tag}: {type(e).__name__}: {e}")
                print(f"   [FAIL] {e}")
                continue

            df = pd.DataFrame(timeseries)
            df.to_csv(out_dir / f"{tag}_timeseries.csv", index=False)

            row_summary = summarize_run(df, DT)
            row_summary.update({
                "window": ts_str, "mode": mode_tag, "bias_shape": str(bias_shape),
                "objective_mode": str(obj_mode), "case_name": tag,
                "block0_norm": float(win["block0_norm"]),
                "block2_norm": float(win["block2_norm"]),
                "speed_t0": float(win["speed_t0"]),
                "speed_block2": float(win["speed_block2"]),
            })
            if provider is not None and provider.records:
                pd.DataFrame(provider.records).to_csv(out_dir / f"{tag}_planner_log.csv", index=False)
                actions = [r["first_action"] for r in provider.records]
                row_summary["planner_first_actions_first10"] = "|".join(actions[:10])
                row_summary["planner_n_buckets"] = len(actions)
                row_summary["planner_active_ratio"] = float(np.mean([a != "hold" for a in actions]))
                # First-hour planner activity
                first6 = actions[:6]  # 60 min = 6 buckets of 10 min
                row_summary["planner_active_ratio_first_hour"] = float(np.mean([a != "hold" for a in first6])) if first6 else 0.0
            summary_rows.append(row_summary)
            print(f"   pitch_first_hour_p95={row_summary['pitch_first_hour_p95']:.3f} "
                  f"work_first_hour={row_summary['pump_work_first_hour']:.1f} "
                  f"work_total={row_summary['pump_work_m3']:.1f}")

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(out_dir / "A2_onset_summary.csv", index=False)

    # Paired delta vs closed_only, focus on FIRST HOUR (where onset matters).
    delta_rows = []
    for ts_str in summary_df["window"].unique():
        sub = summary_df[summary_df["window"] == ts_str]
        co_rows = sub[sub["mode"] == "closed_only"]
        if co_rows.empty:
            continue
        co = co_rows.iloc[0]
        for _, r in sub.iterrows():
            if r["mode"] == "closed_only":
                continue
            delta_rows.append({
                "window": ts_str, "mode": r["mode"],
                "block2_norm": r["block2_norm"], "speed_t0": r["speed_t0"], "speed_block2": r["speed_block2"],
                "d_pitch_first_hour_p95": r["pitch_first_hour_p95"] - co["pitch_first_hour_p95"],
                "d_pitch_first_hour_max": r["pitch_first_hour_max"] - co["pitch_first_hour_max"],
                "d_pump_work_first_hour": r["pump_work_first_hour"] - co["pump_work_first_hour"],
                "d_pump_work_first_hour_pct": (r["pump_work_first_hour"] - co["pump_work_first_hour"]) / max(co["pump_work_first_hour"], 1e-9) * 100.0,
                "d_pitch_p95_total": r["pitch_abs_p95"] - co["pitch_abs_p95"],
                "d_pump_work_total_pct": (r["pump_work_m3"] - co["pump_work_m3"]) / max(co["pump_work_m3"], 1e-9) * 100.0,
            })
    delta_df = pd.DataFrame(delta_rows)
    delta_df.to_csv(out_dir / "A2_onset_delta_paired.csv", index=False)

    # Aggregate per-mode.
    paired_agg_rows = []
    for mode_tag in delta_df["mode"].unique():
        sub = delta_df[delta_df["mode"] == mode_tag]
        paired_agg_rows.append({
            "mode": mode_tag, "n": len(sub),
            "d_pitch_first_hour_p95_mean": float(sub["d_pitch_first_hour_p95"].mean()),
            "d_pitch_first_hour_p95_std": float(sub["d_pitch_first_hour_p95"].std()),
            "d_pitch_first_hour_max_mean": float(sub["d_pitch_first_hour_max"].mean()),
            "d_pump_work_first_hour_mean": float(sub["d_pump_work_first_hour"].mean()),
            "d_pump_work_first_hour_pct_mean": float(sub["d_pump_work_first_hour_pct"].mean()),
            "d_pump_work_first_hour_pct_std": float(sub["d_pump_work_first_hour_pct"].std()),
            "d_pump_work_total_pct_mean": float(sub["d_pump_work_total_pct"].mean()),
            "d_pump_work_total_pct_std": float(sub["d_pump_work_total_pct"].std()),
        })
    paired_agg = pd.DataFrame(paired_agg_rows)
    paired_agg.to_csv(out_dir / "A2_onset_paired_aggregate.csv", index=False)

    # Build report.
    lines = [
        "# A2-onset: planner on transient-onset windows",
        "",
        f"- elapsed: `{time.perf_counter()-t0:.1f}s`",
        f"- onset windows: `{len(summary_df['window'].unique())}` x `{len(MODES)}` modes",
        "- focus metric: FIRST HOUR (the 60-min preview horizon where onset materializes)",
        "",
        "## Per-window first-hour metrics",
        "",
        "| window | speed_t0 | speed_block2 | mode | pitch_p95(1h) | pitch_max(1h) | pump_work(1h) | active_ratio(1h) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for ts_str in summary_df["window"].unique():
        for mode_tag, _, _ in MODES:
            r = summary_df[(summary_df["window"] == ts_str) & (summary_df["mode"] == mode_tag)]
            if r.empty:
                continue
            r = r.iloc[0]
            ar = f"{r.get('planner_active_ratio_first_hour', 0):.2f}" if mode_tag != "closed_only" else "-"
            lines.append(
                f"| {ts_str} | {r['speed_t0']:.1f} | {r['speed_block2']:.1f} | {mode_tag} | "
                f"{r['pitch_first_hour_p95']:.3f} | {r['pitch_first_hour_max']:.3f} | "
                f"{r['pump_work_first_hour']:.1f} | {ar} |"
            )

    lines += [
        "",
        "## Paired delta vs closed_only — FIRST HOUR (mean across onset windows)",
        "",
        "| mode | n | d_pitch_first_hour_p95 | d_pump_work_first_hour | d_pump_work_total_pct |",
        "|---|---|---|---|---|",
    ]
    for _, r in paired_agg.iterrows():
        lines.append(
            f"| {r['mode']} | {int(r['n'])} | "
            f"{r['d_pitch_first_hour_p95_mean']:+.4f}+/-{r['d_pitch_first_hour_p95_std']:.4f} | "
            f"{r['d_pump_work_first_hour_pct_mean']:+.2f}%+/-{r['d_pump_work_first_hour_pct_std']:.2f}% | "
            f"{r['d_pump_work_total_pct_mean']:+.2f}%+/-{r['d_pump_work_total_pct_std']:.2f}% |"
        )

    lines += ["", "## Issues", ""]
    lines += [f"- {x}" for x in issues] if issues else ["None."]

    (out_dir / "A2_onset_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\nReport: {out_dir / 'A2_onset_report.md'}")


if __name__ == "__main__":
    main()
