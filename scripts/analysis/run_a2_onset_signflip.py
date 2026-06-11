#!/usr/bin/env python3
"""A2-onset sign-flip experiment: test whether the planner's pitch_bias is
being injected with the wrong sign for setpoint-shift FF.

Diagnosis (Codex 2026-05-04, after structural ablation round 3):
  - Round 3 confirmed: every variant (v0..v5) produces d_pitch_1h_p95 > 0
    on every onset window. Even v0 baseline shows +0.007deg on average.
  - Hypothesis: pressure_proxy_vec sign is calibrated for "compensation
    direction" (where ballast must move TO PUSH BACK against wind), but
    setpoint-shift FF requires the OPPOSITE sign: bias must be in the
    direction of expected pitch SO THAT error = sp - actual stays small.
  - When sign matches actual pitch direction, PI error shrinks toward zero,
    controller stops correcting -> pitch grows worse, pump saves a bit.

Minimal experiment: same v0_baseline planner, only difference is
setpoint_bias_sign = +1 vs -1. Three variants on the 6 onset windows:
  closed_only   -- baseline
  v0_signpos    -- current production (sign = +1)
  v0_signneg    -- flipped (sign = -1)

Success criterion: signneg shows d_pitch_1h_p95 < 0 (or at least
substantially smaller than signpos), confirming sign was the issue.
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
    return {
        "abs_p50": float(np.percentile(a, 50)),
        "abs_p95": float(np.percentile(a, 95)),
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
        "pump_work_m3": float(np.trapezoid(pump_rate, dx=dt) / 60.0),
        "pump_duty_ratio": float(np.mean(pump_rate > 1e-6)),
    }
    out.update({f"pitch_{k}": v for k, v in percentiles(pitch).items()})
    out.update({f"roll_{k}": v for k, v in percentiles(roll).items()})
    first_hour_n = int(min(3600 / dt, n))
    out["pitch_first_hour_p95"] = float(np.percentile(np.abs(pitch[:first_hour_n]), 95))
    out["pitch_first_hour_max"] = float(np.max(np.abs(pitch[:first_hour_n])))
    out["roll_first_hour_p95"] = float(np.percentile(np.abs(roll[:first_hour_n]), 95))
    out["roll_first_hour_max"] = float(np.max(np.abs(roll[:first_hour_n])))
    out["pump_work_first_hour"] = float(np.trapezoid(pump_rate[:first_hour_n], dx=dt) / 60.0)
    return out


def main() -> None:
    t0 = time.perf_counter()
    out_dir = repo_root / "outputs" / "wind_prediction" / "planner_a2_onset_signflip"
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
        {"tag": "closed_only", "provider": False},
        {"tag": "v0_signpos",  "provider": True, "sign": +1.0},
        {"tag": "v0_signneg",  "provider": True, "sign": -1.0},
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
            if var["provider"]:
                provider = BallastPlannerPreviewProvider(
                    replay_dataset=replay, start_timestamp=ts, cfg=base_cfg,
                    block_discounts=discounts, bias_shape="event_decay",
                    objective_mode="economic",
                    setpoint_bias_sign=float(var["sign"]),
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
            row_summary.update({
                "window": ts_str, "variant": var["tag"], "case_name": tag,
                "speed_t0": float(win["speed_t0"]), "speed_block2": float(win["speed_block2"]),
            })
            if provider is not None and provider.records:
                pd.DataFrame(provider.records).to_csv(out_dir / f"{tag}_planner_log.csv", index=False)
                actions = [r["first_action"] for r in provider.records]
                row_summary["planner_active_ratio_first_hour"] = (
                    float(np.mean([a != "hold" for a in actions[:6]])) if actions else 0.0
                )
                row_summary["planner_action_mix"] = "|".join(f"{k}:{v}" for k, v in Counter(actions).most_common())
                pitches = [r["target_pitch_deg"] for r in provider.records]
                rolls = [r["target_roll_deg"] for r in provider.records]
                row_summary["max_abs_bias_pitch_deg"] = float(max(map(abs, pitches)) if pitches else 0.0)
                row_summary["mean_signed_bias_pitch_deg_first_hour"] = float(np.mean(pitches[:6]) if pitches else 0.0)
                row_summary["mean_signed_bias_roll_deg_first_hour"] = float(np.mean(rolls[:6]) if rolls else 0.0)
            summary_rows.append(row_summary)
            print(f"   pitch_1h_p95={row_summary['pitch_first_hour_p95']:.3f} "
                  f"pump(1h)={row_summary['pump_work_first_hour']:.1f} "
                  f"pump(total)={row_summary['pump_work_m3']:.1f}")

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(out_dir / "signflip_summary.csv", index=False)

    delta_rows = []
    for ts_str in summary_df["window"].unique():
        sub = summary_df[summary_df["window"] == ts_str]
        co_rows = sub[sub["variant"] == "closed_only"]
        if co_rows.empty:
            continue
        co = co_rows.iloc[0]
        # also extract the sign of actual pitch first-hour for direction analysis
        pitch_co = pd.read_csv(out_dir / f"{co['case_name']}_timeseries.csv")["pitch_deg"].values
        first_h = pitch_co[:int(3600 / DT)]
        co_pitch_mean_signed_1h = float(np.mean(first_h))
        for _, r in sub.iterrows():
            if r["variant"] == "closed_only":
                continue
            delta_rows.append({
                "window": ts_str, "variant": r["variant"],
                "speed_t0": r["speed_t0"], "speed_block2": r["speed_block2"],
                "co_pitch_mean_signed_1h": co_pitch_mean_signed_1h,
                "bias_pitch_mean_signed_1h": r.get("mean_signed_bias_pitch_deg_first_hour", 0.0),
                "bias_pitch_max_abs": r.get("max_abs_bias_pitch_deg", 0.0),
                "d_pitch_1h_p95": r["pitch_first_hour_p95"] - co["pitch_first_hour_p95"],
                "d_pitch_1h_max": r["pitch_first_hour_max"] - co["pitch_first_hour_max"],
                "d_roll_1h_p95": r["roll_first_hour_p95"] - co["roll_first_hour_p95"],
                "d_pump_work_1h_pct": (r["pump_work_first_hour"] - co["pump_work_first_hour"])
                                       / max(co["pump_work_first_hour"], 1e-9) * 100.0,
                "d_pump_work_total_pct": (r["pump_work_m3"] - co["pump_work_m3"])
                                          / max(co["pump_work_m3"], 1e-9) * 100.0,
                "active_ratio_1h": r.get("planner_active_ratio_first_hour", 0.0),
                "action_mix": r.get("planner_action_mix", ""),
            })
    delta_df = pd.DataFrame(delta_rows)
    delta_df.to_csv(out_dir / "signflip_delta_paired.csv", index=False)

    agg_rows = []
    for var_tag in delta_df["variant"].unique():
        sub = delta_df[delta_df["variant"] == var_tag]
        agg_rows.append({
            "variant": var_tag, "n": len(sub),
            "d_pitch_1h_p95_mean": float(sub["d_pitch_1h_p95"].mean()),
            "d_pitch_1h_p95_std": float(sub["d_pitch_1h_p95"].std()),
            "d_pitch_1h_max_mean": float(sub["d_pitch_1h_max"].mean()),
            "d_roll_1h_p95_mean": float(sub["d_roll_1h_p95"].mean()),
            "d_pump_work_1h_pct_mean": float(sub["d_pump_work_1h_pct"].mean()),
            "d_pump_work_1h_pct_std": float(sub["d_pump_work_1h_pct"].std()),
            "d_pump_work_total_pct_mean": float(sub["d_pump_work_total_pct"].mean()),
        })
    agg_df = pd.DataFrame(agg_rows)
    agg_df.to_csv(out_dir / "signflip_aggregate.csv", index=False)

    lines = [
        "# A2-onset sign-flip experiment",
        "",
        f"- elapsed: `{time.perf_counter()-t0:.1f}s`",
        f"- onset windows: `{len(summary_df['window'].unique())}` x 3 variants",
        "- hypothesis: setpoint-shift FF needs OPPOSITE sign vs the compensation-direction sign currently in pressure_sign_multiplier",
        "- success: v0_signneg shows d_pitch_1h_p95 < 0 (or substantially smaller than v0_signpos)",
        "",
        "## Paired delta vs closed_only — FIRST HOUR (mean across onset windows)",
        "",
        "| variant | n | d_pitch_1h_p95 | d_roll_1h_p95 | d_pump_work(1h) | d_pump_work(total) |",
        "|---|---|---|---|---|---|",
    ]
    for _, r in agg_df.iterrows():
        lines.append(
            f"| {r['variant']} | {int(r['n'])} | "
            f"{r['d_pitch_1h_p95_mean']:+.4f}+/-{r['d_pitch_1h_p95_std']:.4f} | "
            f"{r['d_roll_1h_p95_mean']:+.4f} | "
            f"{r['d_pump_work_1h_pct_mean']:+.2f}%+/-{r['d_pump_work_1h_pct_std']:.2f}% | "
            f"{r['d_pump_work_total_pct_mean']:+.2f}% |"
        )

    lines += [
        "",
        "## Direction diagnostic per window",
        "",
        "| window | co_pitch_mean(1h) | signpos: bias_pitch_mean(1h) | signneg: bias_pitch_mean(1h) |",
        "|---|---|---|---|",
    ]
    for ts_str in summary_df["window"].unique():
        d_pos = delta_df[(delta_df["window"] == ts_str) & (delta_df["variant"] == "v0_signpos")]
        d_neg = delta_df[(delta_df["window"] == ts_str) & (delta_df["variant"] == "v0_signneg")]
        if d_pos.empty or d_neg.empty:
            continue
        co = float(d_pos.iloc[0]["co_pitch_mean_signed_1h"])
        bp = float(d_pos.iloc[0]["bias_pitch_mean_signed_1h"])
        bn = float(d_neg.iloc[0]["bias_pitch_mean_signed_1h"])
        lines.append(f"| {ts_str} | {co:+.3f} | {bp:+.3f} | {bn:+.3f} |")

    lines += [
        "",
        "## Per-window first-hour metrics",
        "",
        "| window | variant | pitch_p95(1h) | pump(1h) | bias_pitch(signed,1h) | mix |",
        "|---|---|---|---|---|---|",
    ]
    for ts_str in summary_df["window"].unique():
        for var in variants:
            r = summary_df[(summary_df["window"] == ts_str) & (summary_df["variant"] == var["tag"])]
            if r.empty:
                continue
            r = r.iloc[0]
            bp = (f"{r.get('mean_signed_bias_pitch_deg_first_hour', 0):+.3f}"
                  if var["tag"] != "closed_only" else "-")
            mix = r.get("planner_action_mix", "") if var["tag"] != "closed_only" else "-"
            lines.append(
                f"| {ts_str} | {var['tag']} | {r['pitch_first_hour_p95']:.3f} | "
                f"{r['pump_work_first_hour']:.1f} | {bp} | {mix} |"
            )

    lines += ["", "## Issues", ""]
    lines += [f"- {x}" for x in issues] if issues else ["None."]

    (out_dir / "signflip_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\nReport: {out_dir / 'signflip_report.md'}")


if __name__ == "__main__":
    main()
