#!/usr/bin/env python3
"""A2-economic on the full 14-window expanded set with transient diagnostics.

Adds two things on top of run_a2_economic.py:
  (a) reads windows from outputs/.../window_selection_expanded_preview.csv
      so all 14 windows are included (4 groups);
  (b) a transient-onset diagnostic: for each window, computes
      pressure_norm(block_0) vs pressure_norm(block_2) growth,
      block_2_minus_block_0_norm, and a sign flip flag.

Run length kept at 10h/window; 14*4 = 56 sims expected.
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

from wind_prediction.ballast_planner import (
    PlannerConfig,
    compute_pressure_blocks,
    norm_term,
)
from wind_prediction.ballast_planner_provider import BallastPlannerPreviewProvider
from wind_prediction.replay_dataset import Fino1ReplayDataset


MODES = (
    ("closed_only",                       None,         None),
    ("closed_plus_lex_event_decay",       "event_decay", "lex"),
    ("closed_plus_econ_event_decay",      "event_decay", "economic"),
    ("closed_plus_econ_hold",             "hold",        "economic"),
)
TIMESTAMP_FMT = "%Y-%m-%d %H:%M:%S"
DT = 1.0
N_STEPS = 3600  # row_count=60 -> 10h


def discover_excel() -> str:
    for d in (repo_root / "data", repo_root / "archive" / "legacy_fowt_control" / "data"):
        for x in d.rglob("*.xlsx"):
            if "stiffness" in x.name.lower() or "matrix" in x.name.lower():
                return str(x)
    raise FileNotFoundError("No stiffness/matrix .xlsx found in data/")


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
    pump_active = (pump_rate > 1e-6)
    pump_work_m3 = float(np.trapezoid(pump_rate, dx=dt) / 60.0)
    pump_duty = float(np.mean(pump_active))
    if "pump_stage_switch_count" in df.columns:
        switches = float(df["pump_stage_switch_count"].iloc[-1] - df["pump_stage_switch_count"].iloc[0])
    else:
        switches = float("nan")
    fullspeed = float(np.mean(df["pump_fullspeed_any"].values)) if "pump_fullspeed_any" in df.columns else float("nan")
    bias_max = float(np.max(np.abs(df["preview_pitch_bias_deg"].values))) if "preview_pitch_bias_deg" in df.columns else 0.0

    out = {
        "n_steps": n,
        "any_nan": int(np.any(np.isnan(pitch)) or np.any(np.isnan(roll))),
        "pump_work_m3": pump_work_m3,
        "pump_duty_ratio": pump_duty,
        "pump_stage_switches": switches,
        "pump_fullspeed_ratio": fullspeed,
        "preview_bias_max_abs": bias_max,
    }
    out.update({f"pitch_{k}": v for k, v in percentiles(pitch).items()})
    out.update({f"roll_{k}": v for k, v in percentiles(roll).items()})
    out["envelope_pitch_violation_ratio"] = float(np.mean(np.abs(pitch) > 1.2))
    out["envelope_roll_violation_ratio"] = float(np.mean(np.abs(roll) > 0.96))
    return out


def transient_signature(replay: Fino1ReplayDataset, ts: datetime, cfg: PlannerConfig,
                        discounts: list[float]) -> dict:
    sample = replay.sample_for_history_end(ts)
    if sample is None:
        return {"block0_norm": float("nan"), "block2_norm": float("nan"),
                "block2_minus_block0_norm": float("nan"), "sign_flip": 0,
                "is_transient_onset": 0}
    uv = np.asarray(sample.y_uv_raw, dtype=float)
    blocks = compute_pressure_blocks(uv, discounts, cfg)
    p0 = blocks[0]["pressure_vec"]
    p2 = blocks[-1]["pressure_vec"]
    n0 = blocks[0]["pressure_norm"]
    n2 = blocks[-1]["pressure_norm"]
    sign_flip = int(float(np.dot(p0, p2)) < 0.0)
    is_onset = int((n0 < 0.5) and (n2 > 1.0))  # low now, high later
    return {
        "block0_norm": n0, "block2_norm": n2,
        "block2_minus_block0_norm": n2 - n0,
        "sign_flip": sign_flip, "is_transient_onset": is_onset,
    }


def main() -> None:
    t0 = time.perf_counter()
    out_dir = repo_root / "outputs" / "wind_prediction" / "planner_a2_economic_expanded"
    out_dir.mkdir(parents=True, exist_ok=True)

    base_planner_out = repo_root / "outputs" / "wind_prediction" / "planner_a1_dryrun"
    sign_cfg = json.loads((base_planner_out / "diagnostics" / "a01_pressure_vec_sign_convention.json").read_text())
    discount_cfg = json.loads((base_planner_out / "diagnostics" / "a1_block_discount_config.json").read_text())

    cfg = PlannerConfig(pressure_sign_multiplier=float(sign_cfg["planner_sign_multiplier"]))
    discounts = discount_cfg["default_discount_blocks"]

    excel_path = discover_excel()
    print(f"Stiffness file: {excel_path}")

    replay = Fino1ReplayDataset(
        dataset_dir=repo_root / "data" / "processed" / "wind_ml_10min" / "ballast_decision_fino1_meteo_aux_v1",
        split="test",
    )

    windows_df = pd.read_csv(base_planner_out / "window_selection_expanded_preview.csv")
    print(f"Loaded {len(windows_df)} windows from expanded selection.")

    from defaults import clone_cfg  # noqa: F401
    from run_validation import run_closed_loop_case

    summary_rows: list[dict] = []
    transient_rows: list[dict] = []
    issues: list[str] = []

    for _, win in windows_df.iterrows():
        ts_str = str(win["prediction_timestamp"])
        group = str(win.get("a1_group", ""))
        ts = datetime.strptime(ts_str, TIMESTAMP_FMT)
        try:
            wind_trace = replay.build_wind_trace(start_timestamp=ts, row_count=N_STEPS // 60, dt_s=DT)
        except ValueError as e:
            issues.append(f"{ts_str}: build_wind_trace failed ({e})")
            continue

        # transient signature (computed once per window, mode-independent)
        sig = transient_signature(replay, ts, cfg, discounts)
        transient_rows.append({"window": ts_str, "group": group, **sig})

        for mode_tag, bias_shape, obj_mode in MODES:
            tag = f"{ts_str.replace(':', '').replace(' ', '_')}_{mode_tag}"
            print(f"\n--- {ts_str} [{group}] mode={mode_tag} onset={sig['is_transient_onset']} ---")

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
            # only save timeseries for transient-onset windows or storm group to limit disk
            if sig["is_transient_onset"] or group == "high_pressure_high_event":
                df.to_csv(out_dir / f"{tag}_timeseries.csv", index=False)

            row_summary = summarize_run(df, DT)
            row_summary.update({
                "window": ts_str, "group": group,
                "mode": mode_tag, "bias_shape": str(bias_shape),
                "objective_mode": str(obj_mode), "case_name": tag,
                "is_transient_onset": sig["is_transient_onset"],
                "block0_norm": sig["block0_norm"],
                "block2_norm": sig["block2_norm"],
                "sign_flip": sig["sign_flip"],
            })
            if provider is not None and provider.records:
                pd.DataFrame(provider.records).to_csv(out_dir / f"{tag}_planner_log.csv", index=False)
                actions = [r["first_action"] for r in provider.records]
                row_summary["planner_first_actions"] = "|".join(actions)
                row_summary["planner_n_buckets"] = len(actions)
                row_summary["planner_active_ratio"] = float(np.mean([a != "hold" for a in actions]))
            summary_rows.append(row_summary)
            print(f"   pitch_p95={row_summary['pitch_abs_p95']:.3f} "
                  f"work={row_summary['pump_work_m3']:.1f} duty={row_summary['pump_duty_ratio']:.3f}")

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(out_dir / "A2_economic_expanded_summary.csv", index=False)
    pd.DataFrame(transient_rows).to_csv(out_dir / "A2_economic_expanded_transient_signatures.csv", index=False)

    # Aggregate per-mode statistics across all 14 windows.
    agg_rows = []
    for mode_tag, _, _ in MODES:
        sub = summary_df[summary_df["mode"] == mode_tag]
        if sub.empty:
            continue
        agg_rows.append({
            "mode": mode_tag,
            "n_windows": len(sub),
            "pitch_p95_mean": float(sub["pitch_abs_p95"].mean()),
            "pitch_p95_std": float(sub["pitch_abs_p95"].std()),
            "roll_p95_mean": float(sub["roll_abs_p95"].mean()),
            "roll_p95_std": float(sub["roll_abs_p95"].std()),
            "pump_work_mean": float(sub["pump_work_m3"].mean()),
            "pump_work_std": float(sub["pump_work_m3"].std()),
            "pump_duty_mean": float(sub["pump_duty_ratio"].mean()),
            "pump_duty_std": float(sub["pump_duty_ratio"].std()),
            "envelope_violation_pitch_mean": float(sub["envelope_pitch_violation_ratio"].mean()),
        })
    agg_df = pd.DataFrame(agg_rows)
    agg_df.to_csv(out_dir / "A2_economic_expanded_aggregate.csv", index=False)

    # Per-window paired delta vs closed_only.
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
            d_p95 = r["pitch_abs_p95"] - co["pitch_abs_p95"]
            d_work = (r["pump_work_m3"] - co["pump_work_m3"]) / max(co["pump_work_m3"], 1e-9) * 100.0
            delta_rows.append({
                "window": ts_str, "group": r["group"], "mode": r["mode"],
                "is_transient_onset": int(r["is_transient_onset"]),
                "d_pitch_p95": d_p95,
                "d_pump_work_pct": d_work,
                "d_pump_duty_pct": (r["pump_duty_ratio"] - co["pump_duty_ratio"]) * 100.0,
            })
    delta_df = pd.DataFrame(delta_rows)
    delta_df.to_csv(out_dir / "A2_economic_expanded_delta_paired.csv", index=False)

    # Aggregate paired delta per mode (paired means the variance from window-to-window is removed).
    paired_agg_rows = []
    for mode_tag in delta_df["mode"].unique():
        sub = delta_df[delta_df["mode"] == mode_tag]
        paired_agg_rows.append({
            "mode": mode_tag,
            "n": len(sub),
            "d_pitch_p95_mean": float(sub["d_pitch_p95"].mean()),
            "d_pitch_p95_std": float(sub["d_pitch_p95"].std()),
            "d_pump_work_pct_mean": float(sub["d_pump_work_pct"].mean()),
            "d_pump_work_pct_std": float(sub["d_pump_work_pct"].std()),
            "d_pump_duty_pct_mean": float(sub["d_pump_duty_pct"].mean()),
            "d_pump_duty_pct_std": float(sub["d_pump_duty_pct"].std()),
        })
    paired_agg = pd.DataFrame(paired_agg_rows)
    paired_agg.to_csv(out_dir / "A2_economic_expanded_paired_aggregate.csv", index=False)

    # Build markdown report.
    lines = [
        "# A2-economic on Expanded 14-window Set",
        "",
        f"- elapsed: `{time.perf_counter()-t0:.1f}s`",
        f"- windows: `{len(summary_df['window'].unique())}` x `{len(MODES)}` modes",
        "- envelope: pitch=1.2, roll=0.96 (1.2x deadband)",
        "",
        "## Aggregate per-mode (mean +/- std across 14 windows)",
        "",
        "| mode | pitch_p95 | roll_p95 | pump_work(m^3) | pump_duty | env_pitch_viol |",
        "|---|---|---|---|---|---|",
    ]
    for _, r in agg_df.iterrows():
        lines.append(
            f"| {r['mode']} | {r['pitch_p95_mean']:.3f}+/-{r['pitch_p95_std']:.3f} | "
            f"{r['roll_p95_mean']:.3f}+/-{r['roll_p95_std']:.3f} | "
            f"{r['pump_work_mean']:.1f}+/-{r['pump_work_std']:.1f} | "
            f"{r['pump_duty_mean']:.3f}+/-{r['pump_duty_std']:.3f} | "
            f"{r['envelope_violation_pitch_mean']*100:.2f}% |"
        )

    lines += [
        "",
        "## Paired delta vs closed_only (per-mode means across 14 windows)",
        "",
        "Paired = each window contributes a (mode - closed_only) delta, removing window-to-window variance.",
        "",
        "| mode | n | d_pitch_p95 (mean+/-std) | d_pump_work (mean%+/-%) | d_pump_duty (mean%+/-%) |",
        "|---|---|---|---|---|",
    ]
    for _, r in paired_agg.iterrows():
        lines.append(
            f"| {r['mode']} | {int(r['n'])} | "
            f"{r['d_pitch_p95_mean']:+.4f}+/-{r['d_pitch_p95_std']:.4f} | "
            f"{r['d_pump_work_pct_mean']:+.2f}%+/-{r['d_pump_work_pct_std']:.2f}% | "
            f"{r['d_pump_duty_pct_mean']:+.2f}%+/-{r['d_pump_duty_pct_std']:.2f}% |"
        )

    # Onset-only aggregate.
    onset_df = delta_df[delta_df["is_transient_onset"] == 1]
    lines += [
        "",
        f"## Transient-onset windows only (n={len(onset_df)//max(len(MODES)-1, 1)})",
        "",
    ]
    if not onset_df.empty:
        for mode_tag in onset_df["mode"].unique():
            sub = onset_df[onset_df["mode"] == mode_tag]
            lines.append(
                f"- {mode_tag}: n={len(sub)} d_pitch_p95={sub['d_pitch_p95'].mean():+.4f} d_pump_work={sub['d_pump_work_pct'].mean():+.2f}%"
            )
    else:
        lines.append("No transient-onset windows in expanded set (block0_norm<0.5 AND block2_norm>1.0).")

    lines += ["", "## Issues", ""]
    lines += [f"- {x}" for x in issues] if issues else ["None."]

    (out_dir / "A2_economic_expanded_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\nReport: {out_dir / 'A2_economic_expanded_report.md'}")


if __name__ == "__main__":
    main()
