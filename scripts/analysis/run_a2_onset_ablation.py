#!/usr/bin/env python3
"""A2-onset ablation: untangle why economic planner is silent on onset windows.

Codex 2026-05-03 diagnosis:
  - pressure_proxy_vec saturates at 1.5x deadband; far block discount 0.7 brings
    the max to 1.05x deadband, which is always inside the 1.2x envelope (env_norm
    <= 0.875), so envelope_violation_cost is effectively zero on every onset
    window. The economic objective collapses to "minimize pump", and the planner
    holds for the first 30-40 min even when the future ramps to 14-23 m/s.
  - reverse_gate is similarly suppressed: A1 logs show the dominant reject reason
    is `overcomp_too_small` at minimum_overcomp_ratio_for_reverse=0.25.

Three ablations on the 6 onset windows:
  ABL1 flat_discount     -- block_discounts = [1.0, 1.0, 1.0] (was [1.0, 0.85, 0.7])
  ABL2 tight_envelope    -- pitch_envelope_deg=1.0, roll_envelope_deg=0.8 (was 1.2/0.96)
  ABL3 loose_reverse     -- minimum_overcomp_ratio_for_reverse=0.10, dwell=0
  ABL4 combined          -- ABL1 + ABL2 + ABL3

Each ablation is paired against closed_only on the same 6 onset windows.
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


def make_variants(base_cfg: PlannerConfig, base_discounts: list[float]) -> list[dict]:
    """Return list of variant specs: (tag, cfg, discounts, mode_label).
    closed_only is tag-only (no provider built).
    """
    flat_discounts = [1.0, 1.0, 1.0]
    tight_cfg = dataclasses.replace(base_cfg, pitch_envelope_deg=1.0, roll_envelope_deg=0.8)
    loose_cfg = dataclasses.replace(
        base_cfg,
        minimum_overcomp_ratio_for_reverse=0.10,
        dwell_blocks_required_for_reverse=0,
        active_medium_hold_term_min=0.10,
    )
    combined_cfg = dataclasses.replace(
        base_cfg,
        pitch_envelope_deg=1.0, roll_envelope_deg=0.8,
        minimum_overcomp_ratio_for_reverse=0.10,
        dwell_blocks_required_for_reverse=0,
        active_medium_hold_term_min=0.10,
    )
    return [
        # tag, cfg, discounts, build_provider
        {"tag": "closed_only",     "cfg": base_cfg,     "discounts": base_discounts, "provider": False},
        {"tag": "econ_baseline",   "cfg": base_cfg,     "discounts": base_discounts, "provider": True},
        {"tag": "abl1_flat_disc",  "cfg": base_cfg,     "discounts": flat_discounts, "provider": True},
        {"tag": "abl2_tight_env",  "cfg": tight_cfg,    "discounts": base_discounts, "provider": True},
        {"tag": "abl3_loose_rev",  "cfg": loose_cfg,    "discounts": base_discounts, "provider": True},
        {"tag": "abl4_combined",   "cfg": combined_cfg, "discounts": flat_discounts, "provider": True},
    ]


def main() -> None:
    t0 = time.perf_counter()
    out_dir = repo_root / "outputs" / "wind_prediction" / "planner_a2_onset_ablation"
    out_dir.mkdir(parents=True, exist_ok=True)

    base_planner_out = repo_root / "outputs" / "wind_prediction" / "planner_a1_dryrun"
    sign_cfg = json.loads((base_planner_out / "diagnostics" / "a01_pressure_vec_sign_convention.json").read_text())
    discount_cfg = json.loads((base_planner_out / "diagnostics" / "a1_block_discount_config.json").read_text())
    base_cfg = PlannerConfig(pressure_sign_multiplier=float(sign_cfg["planner_sign_multiplier"]))
    base_discounts = discount_cfg["default_discount_blocks"]

    excel_path = discover_excel()
    replay = Fino1ReplayDataset(
        dataset_dir=repo_root / "data" / "processed" / "wind_ml_10min" / "ballast_decision_fino1_meteo_aux_v1",
        split="test",
    )

    onset_df = pd.read_csv(repo_root / "outputs" / "wind_prediction" / "transient_onset_scan" / "scan_top_onset.csv")
    onset_df = onset_df.head(N_TOP_ONSET)

    from defaults import clone_cfg  # noqa: F401
    from run_validation import run_closed_loop_case

    variants = make_variants(base_cfg, base_discounts)
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
                    block_discounts=var["discounts"],
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
                first6 = actions[:6]
                row_summary["planner_active_ratio_first_hour"] = (
                    float(np.mean([a != "hold" for a in first6])) if first6 else 0.0
                )
                row_summary["planner_n_active_total"] = int(sum(a != "hold" for a in actions))
                # which non-hold actions were taken (rough mix indicator)
                from collections import Counter
                ctr = Counter(actions)
                row_summary["planner_action_mix"] = "|".join(f"{k}:{v}" for k, v in ctr.most_common())
            summary_rows.append(row_summary)
            print(f"   pitch_first_hour_p95={row_summary['pitch_first_hour_p95']:.3f} "
                  f"pump_work(1h)={row_summary['pump_work_first_hour']:.1f} "
                  f"pump_work(total)={row_summary['pump_work_m3']:.1f}")

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(out_dir / "ablation_summary.csv", index=False)

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
                "d_pitch_1h_p95": r["pitch_first_hour_p95"] - co["pitch_first_hour_p95"],
                "d_pitch_1h_max": r["pitch_first_hour_max"] - co["pitch_first_hour_max"],
                "d_pump_work_1h": r["pump_work_first_hour"] - co["pump_work_first_hour"],
                "d_pump_work_1h_pct": (r["pump_work_first_hour"] - co["pump_work_first_hour"])
                                       / max(co["pump_work_first_hour"], 1e-9) * 100.0,
                "d_pitch_p95_total": r["pitch_abs_p95"] - co["pitch_abs_p95"],
                "d_pump_work_total_pct": (r["pump_work_m3"] - co["pump_work_m3"])
                                          / max(co["pump_work_m3"], 1e-9) * 100.0,
                "active_ratio_1h": r.get("planner_active_ratio_first_hour", 0.0),
                "n_active_total": r.get("planner_n_active_total", 0),
                "action_mix": r.get("planner_action_mix", ""),
            })
    delta_df = pd.DataFrame(delta_rows)
    delta_df.to_csv(out_dir / "ablation_delta_paired.csv", index=False)

    # Aggregate per variant.
    agg_rows = []
    for var_tag in delta_df["variant"].unique():
        sub = delta_df[delta_df["variant"] == var_tag]
        agg_rows.append({
            "variant": var_tag, "n": len(sub),
            "active_ratio_1h_mean": float(sub["active_ratio_1h"].mean()),
            "n_active_total_mean": float(sub["n_active_total"].mean()),
            "d_pitch_1h_p95_mean": float(sub["d_pitch_1h_p95"].mean()),
            "d_pitch_1h_p95_std": float(sub["d_pitch_1h_p95"].std()),
            "d_pitch_1h_max_mean": float(sub["d_pitch_1h_max"].mean()),
            "d_pump_work_1h_pct_mean": float(sub["d_pump_work_1h_pct"].mean()),
            "d_pump_work_1h_pct_std": float(sub["d_pump_work_1h_pct"].std()),
            "d_pump_work_total_pct_mean": float(sub["d_pump_work_total_pct"].mean()),
        })
    agg_df = pd.DataFrame(agg_rows)
    agg_df.to_csv(out_dir / "ablation_aggregate.csv", index=False)

    # Build report.
    lines = [
        "# A2-onset ablation: relax discount / envelope / reverse-gate",
        "",
        f"- elapsed: `{time.perf_counter()-t0:.1f}s`",
        f"- onset windows: `{len(summary_df['window'].unique())}` x `{len(variants)}` variants",
        "- focus metric: FIRST HOUR (60-min preview horizon)",
        "",
        "## Variants",
        "",
        "| tag | discounts | envelope (pitch/roll) | reverse_overcomp | dwell |",
        "|---|---|---|---|---|",
        "| closed_only | - | - | - | - |",
        f"| econ_baseline | {base_discounts} | 1.2/0.96 | 0.25 | 1 |",
        "| abl1_flat_disc | [1.0, 1.0, 1.0] | 1.2/0.96 | 0.25 | 1 |",
        "| abl2_tight_env | base | 1.0/0.8 | 0.25 | 1 |",
        "| abl3_loose_rev | base | 1.2/0.96 | 0.10 | 0 |",
        "| abl4_combined | [1.0, 1.0, 1.0] | 1.0/0.8 | 0.10 | 0 |",
        "",
        "## Paired delta vs closed_only — FIRST HOUR (mean across 6 onset windows)",
        "",
        "| variant | n | active_ratio(1h) | d_pitch_1h_p95 | d_pump_work(1h) | d_pump_work(total) |",
        "|---|---|---|---|---|---|",
    ]
    for _, r in agg_df.iterrows():
        lines.append(
            f"| {r['variant']} | {int(r['n'])} | "
            f"{r['active_ratio_1h_mean']:.2f} | "
            f"{r['d_pitch_1h_p95_mean']:+.4f}+/-{r['d_pitch_1h_p95_std']:.4f} | "
            f"{r['d_pump_work_1h_pct_mean']:+.2f}%+/-{r['d_pump_work_1h_pct_std']:.2f}% | "
            f"{r['d_pump_work_total_pct_mean']:+.2f}% |"
        )

    lines += [
        "",
        "## Per-window first-hour metrics",
        "",
        "| window | variant | pitch_p95(1h) | pump_work(1h) | active_ratio(1h) | action_mix |",
        "|---|---|---|---|---|---|",
    ]
    for ts_str in summary_df["window"].unique():
        for var in variants:
            r = summary_df[(summary_df["window"] == ts_str) & (summary_df["variant"] == var["tag"])]
            if r.empty:
                continue
            r = r.iloc[0]
            ar = f"{r.get('planner_active_ratio_first_hour', 0):.2f}" if var["tag"] != "closed_only" else "-"
            mix = r.get("planner_action_mix", "") if var["tag"] != "closed_only" else "-"
            lines.append(
                f"| {ts_str} | {var['tag']} | "
                f"{r['pitch_first_hour_p95']:.3f} | "
                f"{r['pump_work_first_hour']:.1f} | {ar} | {mix} |"
            )

    lines += ["", "## Issues", ""]
    lines += [f"- {x}" for x in issues] if issues else ["None."]

    (out_dir / "ablation_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\nReport: {out_dir / 'ablation_report.md'}")


if __name__ == "__main__":
    main()
