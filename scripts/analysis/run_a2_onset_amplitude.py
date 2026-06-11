#!/usr/bin/env python3
"""A2-onset amplitude ablation: do bigger preview biases finally beat the deadband?

Diagnosis (2026-05-04):
  - The planner injects (pitch_bias, roll_bias) onto the closed-loop SETPOINT
    via _combined_trim_raw = current_trim_raw + preview_trim_bias
    (controllers_extras.py:764), at full scale (preview_scale_mode="unscaled",
    preview_scale=1.0). The injection works.
  - The closed-loop PI deadband is pitch=1.0deg / roll=0.8deg, applied to the
    ERROR signal (setpoint - actual) (controllers.py:192-208). On onset windows
    the actual pitch starts near 0deg, so a planner bias of 0.35deg shifts the
    setpoint by 0.35deg and produces an error of 0.35deg, which is INSIDE the
    1.0deg deadband -- the controller never leaves deadband, no pump fires.
  - This is the structural reason the ABL2/ABL4 ablations made things WORSE:
    they triggered active_medium more often, but each medium trigger could not
    push the controller out of deadband.

Fix in this experiment: bump action ratios so active_medium clears the deadband
with margin, active_small sits just below.
  - active_small_ratio:  0.15 -> 0.80  (= 0.80deg, just below pitch deadband)
  - active_medium_ratio: 0.35 -> 1.50  (= 1.50deg, deadband + 50%)
  - pump_saving_ratio:   0.08 -> 0.30  (= 0.30deg, gentle nudge)
  - reverse_small_ratio: 0.12 -> 0.60  (= 0.60deg, real reverse trim)

Two amplitude levels tested, on the same 6 onset windows, paired vs closed_only.
"""
from __future__ import annotations

import dataclasses
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


def make_variants(base_cfg: PlannerConfig) -> list[dict]:
    """closed_only + 3 amplitude levels (baseline / mid / aggressive)."""
    mid_cfg = dataclasses.replace(
        base_cfg,
        active_small_ratio=0.50,   # 0.50deg pitch (half deadband)
        active_medium_ratio=1.20,  # 1.20deg pitch (deadband + 20%)
        pump_saving_ratio=0.20,
        reverse_small_ratio=0.40,
    )
    aggressive_cfg = dataclasses.replace(
        base_cfg,
        active_small_ratio=0.80,
        active_medium_ratio=1.50,
        pump_saving_ratio=0.30,
        reverse_small_ratio=0.60,
    )
    return [
        {"tag": "closed_only",        "cfg": base_cfg,       "provider": False},
        {"tag": "amp_baseline_0.35",  "cfg": base_cfg,       "provider": True},
        {"tag": "amp_mid_1.20",       "cfg": mid_cfg,        "provider": True},
        {"tag": "amp_aggressive_1.50","cfg": aggressive_cfg, "provider": True},
    ]


def main() -> None:
    t0 = time.perf_counter()
    out_dir = repo_root / "outputs" / "wind_prediction" / "planner_a2_onset_amplitude"
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

    variants = make_variants(base_cfg)
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
                    replay_dataset=replay,
                    start_timestamp=ts,
                    cfg=var["cfg"],
                    block_discounts=discounts,
                    bias_shape="event_decay",
                    objective_mode="economic",
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
                "window": ts_str, "variant": var["tag"], "case_name": tag,
                "speed_t0": float(win["speed_t0"]),
                "speed_block2": float(win["speed_block2"]),
            })
            if provider is not None and provider.records:
                pd.DataFrame(provider.records).to_csv(out_dir / f"{tag}_planner_log.csv", index=False)
                actions = [r["first_action"] for r in provider.records]
                row_summary["planner_active_ratio_first_hour"] = (
                    float(np.mean([a != "hold" for a in actions[:6]])) if actions else 0.0
                )
                from collections import Counter
                row_summary["planner_action_mix"] = "|".join(f"{k}:{v}" for k, v in Counter(actions).most_common())
                # max bias actually issued
                pitches = [abs(r["target_pitch_deg"]) for r in provider.records]
                rolls = [abs(r["target_roll_deg"]) for r in provider.records]
                row_summary["max_bias_pitch_deg"] = float(max(pitches) if pitches else 0.0)
                row_summary["max_bias_roll_deg"] = float(max(rolls) if rolls else 0.0)
            summary_rows.append(row_summary)
            print(f"   pitch_first_hour_p95={row_summary['pitch_first_hour_p95']:.3f} "
                  f"pump_work(1h)={row_summary['pump_work_first_hour']:.1f} "
                  f"pump_work(total)={row_summary['pump_work_m3']:.1f}")

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(out_dir / "amplitude_summary.csv", index=False)

    # Paired delta vs closed_only.
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
                "max_bias_pitch_deg": r.get("max_bias_pitch_deg", 0.0),
                "d_pitch_1h_p95": r["pitch_first_hour_p95"] - co["pitch_first_hour_p95"],
                "d_pitch_1h_max": r["pitch_first_hour_max"] - co["pitch_first_hour_max"],
                "d_roll_1h_p95": r["roll_first_hour_p95"] - co["roll_first_hour_p95"],
                "d_pump_work_1h": r["pump_work_first_hour"] - co["pump_work_first_hour"],
                "d_pump_work_1h_pct": (r["pump_work_first_hour"] - co["pump_work_first_hour"])
                                       / max(co["pump_work_first_hour"], 1e-9) * 100.0,
                "d_pitch_p95_total": r["pitch_abs_p95"] - co["pitch_abs_p95"],
                "d_pump_work_total_pct": (r["pump_work_m3"] - co["pump_work_m3"])
                                          / max(co["pump_work_m3"], 1e-9) * 100.0,
                "active_ratio_1h": r.get("planner_active_ratio_first_hour", 0.0),
            })
    delta_df = pd.DataFrame(delta_rows)
    delta_df.to_csv(out_dir / "amplitude_delta_paired.csv", index=False)

    # Aggregate per variant.
    agg_rows = []
    for var_tag in delta_df["variant"].unique():
        sub = delta_df[delta_df["variant"] == var_tag]
        agg_rows.append({
            "variant": var_tag, "n": len(sub),
            "max_bias_pitch_deg_mean": float(sub["max_bias_pitch_deg"].mean()),
            "active_ratio_1h_mean": float(sub["active_ratio_1h"].mean()),
            "d_pitch_1h_p95_mean": float(sub["d_pitch_1h_p95"].mean()),
            "d_pitch_1h_p95_std": float(sub["d_pitch_1h_p95"].std()),
            "d_pitch_1h_max_mean": float(sub["d_pitch_1h_max"].mean()),
            "d_roll_1h_p95_mean": float(sub["d_roll_1h_p95"].mean()),
            "d_pump_work_1h_pct_mean": float(sub["d_pump_work_1h_pct"].mean()),
            "d_pump_work_1h_pct_std": float(sub["d_pump_work_1h_pct"].std()),
            "d_pump_work_total_pct_mean": float(sub["d_pump_work_total_pct"].mean()),
        })
    agg_df = pd.DataFrame(agg_rows)
    agg_df.to_csv(out_dir / "amplitude_aggregate.csv", index=False)

    lines = [
        "# A2-onset amplitude ablation",
        "",
        f"- elapsed: `{time.perf_counter()-t0:.1f}s`",
        f"- onset windows: `{len(summary_df['window'].unique())}` x `{len(variants)}` variants",
        "- core hypothesis: bias must clear closed-loop deadband (pitch=1.0deg / roll=0.8deg) for preview FF to fire pumps preemptively",
        "",
        "## Amplitude levels",
        "",
        "| tag | active_small | active_medium | reverse_small | bias_max(pitch) |",
        "|---|---|---|---|---|",
        "| amp_baseline_0.35 | 0.15 | 0.35 | 0.12 | 0.35deg (inside deadband) |",
        "| amp_mid_1.20 | 0.50 | 1.20 | 0.40 | 1.20deg (clears deadband 20%) |",
        "| amp_aggressive_1.50 | 0.80 | 1.50 | 0.60 | 1.50deg (clears deadband 50%) |",
        "",
        "## Paired delta vs closed_only — FIRST HOUR (mean across onset windows)",
        "",
        "| variant | n | max_bias | active_ratio(1h) | d_pitch_1h_p95 | d_roll_1h_p95 | d_pump_work(1h) | d_pump_work(total) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for _, r in agg_df.iterrows():
        lines.append(
            f"| {r['variant']} | {int(r['n'])} | "
            f"{r['max_bias_pitch_deg_mean']:.2f} | "
            f"{r['active_ratio_1h_mean']:.2f} | "
            f"{r['d_pitch_1h_p95_mean']:+.4f}+/-{r['d_pitch_1h_p95_std']:.4f} | "
            f"{r['d_roll_1h_p95_mean']:+.4f} | "
            f"{r['d_pump_work_1h_pct_mean']:+.2f}%+/-{r['d_pump_work_1h_pct_std']:.2f}% | "
            f"{r['d_pump_work_total_pct_mean']:+.2f}% |"
        )

    lines += ["", "## Per-window first-hour metrics", "",
              "| window | variant | pitch_p95(1h) | pump_work(1h) | active(1h) | max_bias | mix |",
              "|---|---|---|---|---|---|---|"]
    for ts_str in summary_df["window"].unique():
        for var in variants:
            r = summary_df[(summary_df["window"] == ts_str) & (summary_df["variant"] == var["tag"])]
            if r.empty:
                continue
            r = r.iloc[0]
            ar = f"{r.get('planner_active_ratio_first_hour', 0):.2f}" if var["tag"] != "closed_only" else "-"
            mb = f"{r.get('max_bias_pitch_deg', 0):.2f}" if var["tag"] != "closed_only" else "-"
            mix = r.get("planner_action_mix", "") if var["tag"] != "closed_only" else "-"
            lines.append(
                f"| {ts_str} | {var['tag']} | {r['pitch_first_hour_p95']:.3f} | "
                f"{r['pump_work_first_hour']:.1f} | {ar} | {mb} | {mix} |"
            )

    lines += ["", "## Issues", ""]
    lines += [f"- {x}" for x in issues] if issues else ["None."]

    (out_dir / "amplitude_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\nReport: {out_dir / 'amplitude_report.md'}")


if __name__ == "__main__":
    main()
