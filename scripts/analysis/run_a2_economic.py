#!/usr/bin/env python3
"""A2-economic: closed-loop comparison of objective formulations.

Modes (per window):
  closed_only                          : no planner, no preview bias
  closed_plus_planner_lex_event_decay  : current lex-order planner + event_decay
  closed_plus_planner_econ_event_decay : economic-objective planner + event_decay
  closed_plus_planner_econ_hold        : economic-objective planner + hold (sanity)

The economic objective is a zone-MPC scalar:
  J = w_pump_work*work + w_pump_duration*dur + w_startstop*ss
      + w_direction_switch*dsw + w_reverse_penalty*rev
      + w_envelope_soft*sum_k max(0, env_norm_k - 1)^2
      + w_terminal_envelope_soft*max(0, terminal_env_norm - 1)^2
      + w_saturation_hard*saturation
with envelope (pitch=2.0 deg, roll=1.5 deg) defined in
configs/planner_envelope.json.

Reports primary metrics that the lex objective ignored:
  pump_work_m3, pump_duty_ratio, pump_stage_switches
plus attitude p95 / p99 / max so envelope compliance is visible.
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


WINDOWS = [
    ("2023-09-11 03:40:00", "clean_low_pressure_normal"),
    ("2024-09-05 18:10:00", "residual_high_normal"),
    ("2022-02-04 11:00:00", "high_pressure_high_event"),
]
MODES = (
    # (mode_tag, bias_shape, objective_mode)
    ("closed_only",                       None,         None),
    ("closed_plus_lex_event_decay",       "event_decay", "lex"),
    ("closed_plus_econ_event_decay",      "event_decay", "economic"),
    ("closed_plus_econ_hold",             "hold",        "economic"),
)
TIMESTAMP_FMT = "%Y-%m-%d %H:%M:%S"
DT = 1.0
N_STEPS = 3600  # row_count=60 (10-min buckets) -> 60*600 = 36000 ticks = 10h


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
    if "pump_latch_switch_count" in df.columns:
        latch_switches = float(df["pump_latch_switch_count"].iloc[-1] - df["pump_latch_switch_count"].iloc[0])
    else:
        latch_switches = float("nan")
    fullspeed = float(np.mean(df["pump_fullspeed_any"].values)) if "pump_fullspeed_any" in df.columns else float("nan")
    bias_max = float(np.max(np.abs(df["preview_pitch_bias_deg"].values))) if "preview_pitch_bias_deg" in df.columns else 0.0

    out = {
        "n_steps": n,
        "any_nan": int(np.any(np.isnan(pitch)) or np.any(np.isnan(roll))),
        "pump_work_m3": pump_work_m3,
        "pump_duty_ratio": pump_duty,
        "pump_stage_switches": switches,
        "pump_latch_switches": latch_switches,
        "pump_fullspeed_ratio": fullspeed,
        "preview_bias_max_abs": bias_max,
    }
    out.update({f"pitch_{k}": v for k, v in percentiles(pitch).items()})
    out.update({f"roll_{k}": v for k, v in percentiles(roll).items()})
    out["envelope_pitch_violation_ratio"] = float(np.mean(np.abs(pitch) > 1.2))
    out["envelope_roll_violation_ratio"] = float(np.mean(np.abs(roll) > 0.96))
    return out


def main() -> None:
    t0 = time.perf_counter()
    out_dir = repo_root / "outputs" / "wind_prediction" / "planner_a2_economic"
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

    from defaults import clone_cfg  # noqa: F401
    from run_validation import run_closed_loop_case

    summary_rows: list[dict] = []
    issues: list[str] = []

    for ts_str, group in WINDOWS:
        ts = datetime.strptime(ts_str, TIMESTAMP_FMT)
        try:
            wind_trace = replay.build_wind_trace(start_timestamp=ts, row_count=N_STEPS // 60, dt_s=DT)
        except ValueError as e:
            issues.append(f"{ts_str}: build_wind_trace failed ({e})")
            continue

        for mode_tag, bias_shape, obj_mode in MODES:
            tag = f"{ts_str.replace(':', '').replace(' ', '_')}_{mode_tag}"
            print(f"\n--- {ts_str} [{group}] mode={mode_tag} ---")

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
                "window": ts_str, "group": group,
                "mode": mode_tag, "bias_shape": str(bias_shape),
                "objective_mode": str(obj_mode), "case_name": tag,
            })
            if provider is not None and provider.records:
                pd.DataFrame(provider.records).to_csv(out_dir / f"{tag}_planner_log.csv", index=False)
                actions = [r["first_action"] for r in provider.records]
                row_summary["planner_first_actions"] = "|".join(actions)
                row_summary["planner_n_buckets"] = len(actions)
                env_norms = [r.get("max_envelope_norm", float("nan")) for r in provider.records]
                env_norms = [v for v in env_norms if not (v is None or (isinstance(v, float) and np.isnan(v)))]
                if env_norms:
                    row_summary["planner_max_env_norm"] = float(max(env_norms))
                    row_summary["planner_mean_env_norm"] = float(np.mean(env_norms))
            summary_rows.append(row_summary)
            print(f"   pitch_p95={row_summary['pitch_abs_p95']:.3f} "
                  f"pump_work={row_summary['pump_work_m3']:.1f} "
                  f"duty={row_summary['pump_duty_ratio']:.3f} "
                  f"sw={row_summary['pump_stage_switches']:.0f}")

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(out_dir / "A2_economic_summary.csv", index=False)

    # Build comparison report focused on pump cost vs envelope compliance.
    report_lines = [
        "# A2-economic Comparison Report",
        "",
        f"- elapsed: `{time.perf_counter()-t0:.1f}s`",
        f"- windows: `{len(WINDOWS)}` x `{len(MODES)}` modes",
        "- envelope (zone-MPC): pitch=1.2 deg, roll=0.96 deg (1.2x closed-loop deadband)",
        "",
        "## Per-window pump cost vs envelope compliance",
        "",
        "| window | mode | pitch_p95 | pitch_p99 | pitch_max | roll_p95 | pump_work(m^3) | pump_duty | switches | pitch>env(%) |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for _, r in summary_df.iterrows():
        report_lines.append(
            f"| {r['window']} | {r['mode']} | "
            f"{r['pitch_abs_p95']:.3f} | {r['pitch_abs_p99']:.3f} | {r['pitch_abs_max']:.3f} | "
            f"{r['roll_abs_p95']:.3f} | {r['pump_work_m3']:.1f} | "
            f"{r['pump_duty_ratio']:.3f} | {r['pump_stage_switches']:.0f} | "
            f"{r['envelope_pitch_violation_ratio']*100:.2f} |"
        )

    # Aggregate per-window deltas vs closed_only.
    report_lines += [
        "",
        "## Delta vs closed_only (per window)",
        "",
        "| window | mode | d_pitch_p95 | d_roll_p95 | d_pump_work_pct | d_pump_duty_pct |",
        "|---|---|---|---|---|---|",
    ]
    for ts_str, _ in WINDOWS:
        sub = summary_df[summary_df["window"] == ts_str]
        co = sub[sub["mode"] == "closed_only"]
        if co.empty:
            continue
        co0 = co.iloc[0]
        for _, r in sub.iterrows():
            if r["mode"] == "closed_only":
                continue
            d_p95 = r["pitch_abs_p95"] - co0["pitch_abs_p95"]
            d_r95 = r["roll_abs_p95"] - co0["roll_abs_p95"]
            d_work = (r["pump_work_m3"] - co0["pump_work_m3"]) / max(co0["pump_work_m3"], 1e-9) * 100.0
            d_duty = (r["pump_duty_ratio"] - co0["pump_duty_ratio"]) / max(co0["pump_duty_ratio"], 1e-9) * 100.0
            report_lines.append(
                f"| {ts_str} | {r['mode']} | {d_p95:+.3f} | {d_r95:+.3f} | {d_work:+.1f}% | {d_duty:+.1f}% |"
            )

    report_lines += ["", "## Issues", ""]
    report_lines += [f"- {x}" for x in issues] if issues else ["None."]

    (out_dir / "A2_economic_report.md").write_text("\n".join(report_lines), encoding="utf-8")
    print(f"\nReport: {out_dir / 'A2_economic_report.md'}")


if __name__ == "__main__":
    main()
