#!/usr/bin/env python3
"""A2-onset FF scale sweep: dose-response of the FF channel.

Round-1 FF result (planner_a2_onset_ff): ff_only @ scale=1.0 produced
d_pitch_1h_p95 = -0.028deg / d_pump_total = +15%. FF works but undersized:
planner's active_medium = 0.35deg pitch maps to ~63t/tank, which is small
relative to the 4deg actual onset pitch.

This sweep tests ff_scale = {1.0, 2.0, 4.0} on ff_only mode (cleanest signal).
ff_scale multiplies the per-tank mass delta directly, WITHOUT touching the
planner's economic cost calculus (which is what foiled round-2 amplitude
ablation when we bumped active_medium_ratio).

Goal: determine whether FF dose-response is monotonic and whether a useful
operating point exists below the +5% pump_total budget.
"""
from __future__ import annotations

import json
import sys
import time
from collections import Counter
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


TIMESTAMP_FMT = "%Y-%m-%d %H:%M:%S"
DT = 1.0
N_STEPS = 3600
N_TOP_ONSET = 6


def discover_excel() -> str:
    for d in (repo_root / "data", repo_root / "archive" / "legacy_fowt_control" / "data"):
        for x in d.rglob("*.xlsx"):
            if "stiffness" in x.name.lower() or "matrix" in x.name.lower():
                return str(x)
    raise FileNotFoundError


def percentiles(x: np.ndarray) -> dict:
    a = np.abs(x)
    return {"abs_p50": float(np.percentile(a, 50)),
            "abs_p95": float(np.percentile(a, 95)),
            "abs_max": float(a.max()),
            "rms": float(np.sqrt(np.mean(x ** 2)))}


def summarize_run(df: pd.DataFrame, dt: float) -> dict:
    n = len(df)
    pitch = df.get("pitch_deg", pd.Series(np.zeros(n))).values
    roll = df.get("roll_deg", pd.Series(np.zeros(n))).values
    pump_rate = df.get("pump_total_rate_m3_min", pd.Series(np.zeros(n))).abs().values
    out = {"n_steps": n,
           "pump_work_m3": float(np.trapezoid(pump_rate, dx=dt) / 60.0),
           "pump_duty_ratio": float(np.mean(pump_rate > 1e-6))}
    out.update({f"pitch_{k}": v for k, v in percentiles(pitch).items()})
    out.update({f"roll_{k}": v for k, v in percentiles(roll).items()})
    first_hour_n = int(min(3600 / dt, n))
    out["pitch_first_hour_p95"] = float(np.percentile(np.abs(pitch[:first_hour_n]), 95))
    out["pitch_first_hour_max"] = float(np.max(np.abs(pitch[:first_hour_n])))
    out["roll_first_hour_p95"] = float(np.percentile(np.abs(roll[:first_hour_n]), 95))
    out["pump_work_first_hour"] = float(np.trapezoid(pump_rate[:first_hour_n], dx=dt) / 60.0)
    return out


def main() -> None:
    t0 = time.perf_counter()
    out_dir = repo_root / "outputs" / "wind_prediction" / "planner_a2_onset_ff_scale"
    out_dir.mkdir(parents=True, exist_ok=True)

    base_planner_out = repo_root / "outputs" / "wind_prediction" / "planner_a1_dryrun"
    sign_cfg = json.loads((base_planner_out / "diagnostics" / "a01_pressure_vec_sign_convention.json").read_text())
    discount_cfg = json.loads((base_planner_out / "diagnostics" / "a1_block_discount_config.json").read_text())
    base_cfg = PlannerConfig(pressure_sign_multiplier=float(sign_cfg["planner_sign_multiplier"]))
    discounts = discount_cfg["default_discount_blocks"]

    excel_path = discover_excel()
    replay = Fino1ReplayDataset(
        dataset_dir=repo_root / "data" / "processed" / "wind_ml_10min" / "ballast_decision_fino1_meteo_aux_v1",
        split="test",
    )

    onset_df = pd.read_csv(repo_root / "outputs" / "wind_prediction" / "transient_onset_scan" / "scan_top_onset.csv")
    onset_df = onset_df.head(N_TOP_ONSET)

    from defaults import clone_cfg  # noqa: F401
    from run_validation import run_closed_loop_case

    variants = [
        {"tag": "closed_only",   "provider_kw": None},
        {"tag": "ff_scale_1",    "provider_kw": {"setpoint_channel_enabled": False, "ff_channel_enabled": True, "ff_scale": 1.0}},
        {"tag": "ff_scale_2",    "provider_kw": {"setpoint_channel_enabled": False, "ff_channel_enabled": True, "ff_scale": 2.0}},
        {"tag": "ff_scale_4",    "provider_kw": {"setpoint_channel_enabled": False, "ff_channel_enabled": True, "ff_scale": 4.0}},
    ]
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

        for var in variants:
            tag = f"{ts_str.replace(':', '').replace(' ', '_')}_{var['tag']}"
            print(f"\n--- {ts_str} variant={var['tag']} ---")
            provider = None
            if var["provider_kw"] is not None:
                provider = BallastPlannerPreviewProvider(
                    replay_dataset=replay, start_timestamp=ts, cfg=base_cfg,
                    block_discounts=discounts, bias_shape="event_decay",
                    objective_mode="economic", **var["provider_kw"],
                )
            try:
                row, timeseries = run_closed_loop_case(
                    excel_path=excel_path, case_name=tag, dt=DT,
                    n_steps=int(wind_trace["n_steps"]), wind_trace=wind_trace,
                    control_enabled=True, record_timeseries=True,
                    experiment_protocol="main", platform_profile="default",
                    start_from_heave_equilibrium=True, preview_trim_provider=provider,
                )
            except Exception as e:
                issues.append(f"{tag}: {type(e).__name__}: {e}")
                print(f"   [FAIL] {e}")
                continue
            df = pd.DataFrame(timeseries)
            df.to_csv(out_dir / f"{tag}_timeseries.csv", index=False)
            row_summary = summarize_run(df, DT)
            row_summary.update({"window": ts_str, "variant": var["tag"], "case_name": tag,
                                "speed_t0": float(win["speed_t0"]),
                                "speed_block2": float(win["speed_block2"])})
            summary_rows.append(row_summary)
            print(f"   pitch_1h_p95={row_summary['pitch_first_hour_p95']:.3f} "
                  f"pump(1h)={row_summary['pump_work_first_hour']:.1f} "
                  f"pump(total)={row_summary['pump_work_m3']:.1f}")

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(out_dir / "ff_scale_summary.csv", index=False)

    delta_rows = []
    for ts_str in summary_df["window"].unique():
        sub = summary_df[summary_df["window"] == ts_str]
        co_rows = sub[sub["variant"] == "closed_only"]
        if co_rows.empty:
            continue
        co = co_rows.iloc[0]
        for _, r in sub.iterrows():
            if r["variant"] == "closed_only":
                continue
            delta_rows.append({
                "window": ts_str, "variant": r["variant"],
                "speed_t0": r["speed_t0"], "speed_block2": r["speed_block2"],
                "d_pitch_1h_p95": r["pitch_first_hour_p95"] - co["pitch_first_hour_p95"],
                "d_roll_1h_p95": r["roll_first_hour_p95"] - co["roll_first_hour_p95"],
                "d_pump_work_1h_pct": (r["pump_work_first_hour"] - co["pump_work_first_hour"])
                                       / max(co["pump_work_first_hour"], 1e-9) * 100.0,
                "d_pitch_p95_total": r["pitch_abs_p95"] - co["pitch_abs_p95"],
                "d_pump_work_total_pct": (r["pump_work_m3"] - co["pump_work_m3"])
                                          / max(co["pump_work_m3"], 1e-9) * 100.0,
            })
    delta_df = pd.DataFrame(delta_rows)
    delta_df.to_csv(out_dir / "ff_scale_delta_paired.csv", index=False)

    agg_rows = []
    for var_tag in delta_df["variant"].unique():
        sub = delta_df[delta_df["variant"] == var_tag]
        agg_rows.append({
            "variant": var_tag, "n": len(sub),
            "d_pitch_1h_p95_mean": float(sub["d_pitch_1h_p95"].mean()),
            "d_pitch_1h_p95_std": float(sub["d_pitch_1h_p95"].std()),
            "d_roll_1h_p95_mean": float(sub["d_roll_1h_p95"].mean()),
            "d_pump_work_1h_pct_mean": float(sub["d_pump_work_1h_pct"].mean()),
            "d_pump_work_total_pct_mean": float(sub["d_pump_work_total_pct"].mean()),
            "d_pitch_p95_total_mean": float(sub["d_pitch_p95_total"].mean()),
        })
    agg_df = pd.DataFrame(agg_rows)
    agg_df.to_csv(out_dir / "ff_scale_aggregate.csv", index=False)

    lines = [
        "# A2-onset FF scale sweep",
        "",
        f"- elapsed: `{time.perf_counter()-t0:.1f}s`",
        f"- onset windows: `{len(summary_df['window'].unique())}` x 4 variants (closed + 3 scales)",
        "- ff_scale multiplies mass FF only; planner economics unchanged",
        "",
        "## Paired delta vs closed_only — FIRST HOUR + total",
        "",
        "| variant | n | d_pitch_1h_p95 | d_pitch_total_p95 | d_roll_1h_p95 | d_pump(1h) | d_pump(total) |",
        "|---|---|---|---|---|---|---|",
    ]
    for _, r in agg_df.iterrows():
        lines.append(
            f"| {r['variant']} | {int(r['n'])} | "
            f"{r['d_pitch_1h_p95_mean']:+.4f}+/-{r['d_pitch_1h_p95_std']:.4f} | "
            f"{r['d_pitch_p95_total_mean']:+.4f} | "
            f"{r['d_roll_1h_p95_mean']:+.4f} | "
            f"{r['d_pump_work_1h_pct_mean']:+.2f}% | "
            f"{r['d_pump_work_total_pct_mean']:+.2f}% |"
        )

    lines += ["", "## Per-window first-hour metrics", "",
              "| window | variant | pitch_1h_p95 | pump(1h) | pump(total) |",
              "|---|---|---|---|---|"]
    for ts_str in summary_df["window"].unique():
        for var in variants:
            r = summary_df[(summary_df["window"] == ts_str) & (summary_df["variant"] == var["tag"])]
            if r.empty:
                continue
            r = r.iloc[0]
            lines.append(
                f"| {ts_str} | {var['tag']} | {r['pitch_first_hour_p95']:.3f} | "
                f"{r['pump_work_first_hour']:.1f} | {r['pump_work_m3']:.1f} |"
            )

    lines += ["", "## Issues", ""]
    lines += [f"- {x}" for x in issues] if issues else ["None."]

    (out_dir / "ff_scale_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\nReport: {out_dir / 'ff_scale_report.md'}")


if __name__ == "__main__":
    main()
