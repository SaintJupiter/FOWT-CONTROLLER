#!/usr/bin/env python3
"""A2-onset structural ablation (round 3): zone-MPC envelope semantics.

Round-2 amplitude ablation showed that bumping active_medium_ratio to 1.2/1.5
caused the planner to abandon active_medium and pick pump_saving instead.
Diagnosis: economic scalar cost is dominated by pump_work_cost (linear in
||avec/db||), and envelope_violation_cost is effectively zero because the far
block is discounted to <= 1.05x deadband and the envelope is 1.2x deadband.
Bigger actions are punished without earning any envelope benefit.

This round tests the zone-MPC fix as a STRUCTURAL change, with each piece
isolated. Production defaults are not changed; everything is opt-in via
PlannerConfig overrides.

Variants (each paired vs closed_only):
  v0  baseline               -- current production planner config
  v1  +env_undiscounted      -- envelope evaluated on raw (undiscounted) pressure
  v2  v1 + tight_env (1.0/0.8)
  v3  v2 + amplitude bump (small=0.8, medium=1.5)
  v4  v3 + low pump_work weight (0.2)
  v5  v4 + barrier on (envelope-violated AND no strong action -> +50.0 cost)

Success criteria (per Codex 2026-05-04):
  - first-hour pitch p95 actually decreases (vs +0.005..+0.12 in earlier rounds)
  - action_mix returns to active_medium (not pump_saving fake-active)
  - pump_work delta stays negative or improves
"""
from __future__ import annotations

import dataclasses
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
    out["pump_work_first_hour"] = float(np.trapezoid(pump_rate[:first_hour_n], dx=dt) / 60.0)
    return out


def make_variants(base_cfg: PlannerConfig) -> list[dict]:
    v1 = dataclasses.replace(base_cfg, envelope_use_discount=False)
    v2 = dataclasses.replace(v1, pitch_envelope_deg=1.0, roll_envelope_deg=0.8)
    v3 = dataclasses.replace(v2, active_small_ratio=0.80, active_medium_ratio=1.50,
                             pump_saving_ratio=0.30, reverse_small_ratio=0.60)
    v4 = dataclasses.replace(v3, w_pump_work=0.2)
    v5 = dataclasses.replace(v4, envelope_barrier_active=True, envelope_barrier_const=50.0)
    return [
        {"tag": "closed_only",      "cfg": base_cfg, "provider": False},
        {"tag": "v0_baseline",      "cfg": base_cfg, "provider": True},
        {"tag": "v1_env_undisc",    "cfg": v1,       "provider": True},
        {"tag": "v2_env_tight",     "cfg": v2,       "provider": True},
        {"tag": "v3_amp_bump",      "cfg": v3,       "provider": True},
        {"tag": "v4_low_wpump",     "cfg": v4,       "provider": True},
        {"tag": "v5_barrier_on",    "cfg": v5,       "provider": True},
    ]


def main() -> None:
    t0 = time.perf_counter()
    out_dir = repo_root / "outputs" / "wind_prediction" / "planner_a2_onset_struct"
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
                    replay_dataset=replay, start_timestamp=ts, cfg=var["cfg"],
                    block_discounts=discounts, bias_shape="event_decay",
                    objective_mode="economic",
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
                row_summary["planner_n_active_medium"] = int(sum(a == "active_medium" for a in actions))
                row_summary["planner_action_mix"] = "|".join(f"{k}:{v}" for k, v in Counter(actions).most_common())
                pitches = [abs(r["target_pitch_deg"]) for r in provider.records]
                rolls = [abs(r["target_roll_deg"]) for r in provider.records]
                row_summary["max_bias_pitch_deg"] = float(max(pitches) if pitches else 0.0)
                row_summary["max_bias_roll_deg"] = float(max(rolls) if rolls else 0.0)
                row_summary["max_env_norm_logged"] = float(np.nanmax(
                    [r.get("max_envelope_norm", 0.0) for r in provider.records]
                ) if provider.records else 0.0)
            summary_rows.append(row_summary)
            print(f"   pitch_1h_p95={row_summary['pitch_first_hour_p95']:.3f} "
                  f"pump(1h)={row_summary['pump_work_first_hour']:.1f} "
                  f"pump(total)={row_summary['pump_work_m3']:.1f}")

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(out_dir / "struct_summary.csv", index=False)

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
                "n_active_medium": r.get("planner_n_active_medium", 0),
                "max_env_norm": r.get("max_env_norm_logged", 0.0),
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
                "action_mix": r.get("planner_action_mix", ""),
            })
    delta_df = pd.DataFrame(delta_rows)
    delta_df.to_csv(out_dir / "struct_delta_paired.csv", index=False)

    agg_rows = []
    for var_tag in delta_df["variant"].unique():
        sub = delta_df[delta_df["variant"] == var_tag]
        agg_rows.append({
            "variant": var_tag, "n": len(sub),
            "max_bias_pitch_deg_mean": float(sub["max_bias_pitch_deg"].mean()),
            "n_active_medium_mean": float(sub["n_active_medium"].mean()),
            "max_env_norm_mean": float(sub["max_env_norm"].mean()),
            "active_ratio_1h_mean": float(sub["active_ratio_1h"].mean()),
            "d_pitch_1h_p95_mean": float(sub["d_pitch_1h_p95"].mean()),
            "d_pitch_1h_p95_std": float(sub["d_pitch_1h_p95"].std()),
            "d_pitch_1h_max_mean": float(sub["d_pitch_1h_max"].mean()),
            "d_pump_work_1h_pct_mean": float(sub["d_pump_work_1h_pct"].mean()),
            "d_pump_work_1h_pct_std": float(sub["d_pump_work_1h_pct"].std()),
            "d_pump_work_total_pct_mean": float(sub["d_pump_work_total_pct"].mean()),
        })
    agg_df = pd.DataFrame(agg_rows)
    agg_df.to_csv(out_dir / "struct_aggregate.csv", index=False)

    lines = [
        "# A2-onset structural ablation (round 3): zone-MPC semantics",
        "",
        f"- elapsed: `{time.perf_counter()-t0:.1f}s`",
        f"- onset windows: `{len(summary_df['window'].unique())}` x `{len(variants)}` variants",
        "- success criteria: pitch_p95(1h) decreases AND action_mix returns to active_medium (not pump_saving)",
        "",
        "## Variants (cumulative; each adds on top of the previous)",
        "",
        "| tag | env_use_discount | envelope (pitch/roll) | medium_ratio | w_pump_work | barrier |",
        "|---|---|---|---|---|---|",
        "| closed_only | - | - | - | - | - |",
        "| v0_baseline | True | 1.2/0.96 | 0.35 | 1.0 | off |",
        "| v1_env_undisc | False | 1.2/0.96 | 0.35 | 1.0 | off |",
        "| v2_env_tight | False | 1.0/0.8 | 0.35 | 1.0 | off |",
        "| v3_amp_bump | False | 1.0/0.8 | 1.50 | 1.0 | off |",
        "| v4_low_wpump | False | 1.0/0.8 | 1.50 | 0.2 | off |",
        "| v5_barrier_on | False | 1.0/0.8 | 1.50 | 0.2 | on (+50) |",
        "",
        "## Paired delta vs closed_only — FIRST HOUR (mean across onset windows)",
        "",
        "| variant | n | max_bias | env_norm_max | n_active_medium | active_ratio | d_pitch_1h_p95 | d_roll_1h_p95 | d_pump_1h | d_pump_total |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for _, r in agg_df.iterrows():
        lines.append(
            f"| {r['variant']} | {int(r['n'])} | "
            f"{r['max_bias_pitch_deg_mean']:.2f} | "
            f"{r['max_env_norm_mean']:.2f} | "
            f"{r['n_active_medium_mean']:.1f} | "
            f"{r['active_ratio_1h_mean']:.2f} | "
            f"{r['d_pitch_1h_p95_mean']:+.4f}+/-{r['d_pitch_1h_p95_std']:.4f} | "
            f"{r['d_pitch_1h_max_mean']:+.4f} | "
            f"{r['d_pump_work_1h_pct_mean']:+.2f}%+/-{r['d_pump_work_1h_pct_std']:.2f}% | "
            f"{r['d_pump_work_total_pct_mean']:+.2f}% |"
        )

    lines += ["", "## Per-window first-hour metrics", "",
              "| window | variant | pitch_1h_p95 | pump(1h) | active(1h) | max_bias | env_max | mix |",
              "|---|---|---|---|---|---|---|---|"]
    for ts_str in summary_df["window"].unique():
        for var in variants:
            r = summary_df[(summary_df["window"] == ts_str) & (summary_df["variant"] == var["tag"])]
            if r.empty:
                continue
            r = r.iloc[0]
            ar = f"{r.get('planner_active_ratio_first_hour', 0):.2f}" if var["tag"] != "closed_only" else "-"
            mb = f"{r.get('max_bias_pitch_deg', 0):.2f}" if var["tag"] != "closed_only" else "-"
            em = f"{r.get('max_env_norm_logged', 0):.2f}" if var["tag"] != "closed_only" else "-"
            mix = r.get("planner_action_mix", "") if var["tag"] != "closed_only" else "-"
            lines.append(
                f"| {ts_str} | {var['tag']} | {r['pitch_first_hour_p95']:.3f} | "
                f"{r['pump_work_first_hour']:.1f} | {ar} | {mb} | {em} | {mix} |"
            )

    lines += ["", "## Issues", ""]
    lines += [f"- {x}" for x in issues] if issues else ["None."]

    (out_dir / "struct_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\nReport: {out_dir / 'struct_report.md'}")


if __name__ == "__main__":
    main()
