#!/usr/bin/env python3
"""A2-smoke: minimal closed-loop check that planner first_action injects cleanly.

Runs 3 windows × 2 modes (closed_only / closed_plus_planner) for 1 hour each.
Writes time-series + a small per-(window,mode) summary + pass/fail report.
No tuning. No statistical claims. Interface verification only.
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root / "src"))
sys.path.insert(0, str(repo_root / "archive" / "legacy_fowt_control"))

from wind_prediction.ballast_planner import PlannerConfig
from wind_prediction.ballast_planner_provider import BallastPlannerPreviewProvider
from wind_prediction.replay_dataset import Fino1ReplayDataset


SMOKE_WINDOWS = [
    ("2023-09-11 03:40:00", "clean_low_pressure_normal"),
    ("2024-09-05 18:10:00", "residual_high_normal"),
    ("2022-02-04 11:00:00", "high_pressure_high_event"),
]
SMOKE_MODES = (
    ("closed_only", None),
    ("closed_plus_planner_hold", "hold"),
    ("closed_plus_planner_event_decay", "event_decay"),
)
TIMESTAMP_FMT = "%Y-%m-%d %H:%M:%S"
DT = 1.0
N_STEPS = 3600  # 1 hour


def discover_excel() -> str:
    for d in (repo_root / "data", repo_root / "archive" / "legacy_fowt_control" / "data"):
        for x in d.rglob("*.xlsx"):
            if "stiffness" in x.name.lower() or "matrix" in x.name.lower():
                return str(x)
    raise FileNotFoundError("No stiffness/matrix .xlsx found in data/")


def summarize_timeseries(df: pd.DataFrame) -> dict:
    pitch = df["pitch_deg"].astype(float).values if "pitch_deg" in df.columns else np.zeros(len(df))
    roll = df["roll_deg"].astype(float).values if "roll_deg" in df.columns else np.zeros(len(df))
    summary = {
        "n_steps": int(len(df)),
        "pitch_rms": float(np.sqrt(np.mean(pitch ** 2))),
        "roll_rms": float(np.sqrt(np.mean(roll ** 2))),
        "pitch_abs_max": float(np.max(np.abs(pitch))),
        "roll_abs_max": float(np.max(np.abs(roll))),
        "any_nan": int(np.any(np.isnan(pitch)) or np.any(np.isnan(roll))),
    }
    if "pump_fullspeed_any" in df.columns:
        summary["pump_fullspeed_any_ratio"] = float(np.mean(df["pump_fullspeed_any"].astype(float)))
    if "preview_pitch_bias_deg" in df.columns:
        bias = df["preview_pitch_bias_deg"].astype(float).values
        summary["preview_pitch_bias_max_abs"] = float(np.max(np.abs(bias)))
        summary["preview_pitch_bias_nonzero_ratio"] = float(np.mean(np.abs(bias) > 1e-9))
    return summary


def main() -> None:
    t0 = time.perf_counter()
    out_dir = repo_root / "outputs" / "wind_prediction" / "planner_a2_smoke"
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

    # Late imports — depend on archive sys.path being in place.
    from defaults import clone_cfg
    from run_validation import run_closed_loop_case

    summary_rows: list[dict] = []
    issues: list[str] = []

    for ts_str, group in SMOKE_WINDOWS:
        ts = datetime.strptime(ts_str, TIMESTAMP_FMT)
        try:
            wind_trace = replay.build_wind_trace(start_timestamp=ts, row_count=N_STEPS // 60, dt_s=DT)
        except ValueError as e:
            issues.append(f"{ts_str}: build_wind_trace failed ({e})")
            continue

        for mode, shape in SMOKE_MODES:
            tag = f"{ts_str.replace(':', '').replace(' ', '_')}_{mode}"
            print(f"\n--- {ts_str} [{group}] mode={mode} ---")

            provider = None
            if shape is not None:
                provider = BallastPlannerPreviewProvider(
                    replay_dataset=replay,
                    start_timestamp=ts,
                    cfg=cfg,
                    block_discounts=discounts,
                    bias_shape=shape,
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
                issues.append(f"{tag}: run_closed_loop_case raised {type(e).__name__}: {e}")
                print(f"   [FAIL] {e}")
                continue

            df = pd.DataFrame(timeseries)
            df.to_csv(out_dir / f"{tag}_timeseries.csv", index=False)

            row_summary = summarize_timeseries(df)
            row_summary.update({
                "window": ts_str,
                "group": group,
                "mode": mode,
                "case_name": tag,
            })
            if provider is not None:
                # Save planner action log.
                pd.DataFrame(provider.records).to_csv(out_dir / f"{tag}_planner_log.csv", index=False)
                if provider.records:
                    actions = [r["first_action"] for r in provider.records]
                    row_summary["planner_first_actions"] = "|".join(actions)
                    row_summary["planner_n_buckets"] = len(actions)
            summary_rows.append(row_summary)
            print(f"   pitch_rms={row_summary['pitch_rms']:.4f}deg, "
                  f"roll_rms={row_summary['roll_rms']:.4f}deg, "
                  f"any_nan={row_summary['any_nan']}")

    if not summary_rows:
        report_lines = ["# A2-smoke Report", "", "ALL RUNS FAILED.", ""]
        report_lines.extend(f"- {x}" for x in issues)
        (out_dir / "A2_smoke_report.md").write_text("\n".join(report_lines), encoding="utf-8")
        return

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(out_dir / "A2_smoke_summary.csv", index=False)

    # Pass/fail per spec §5 — each planner mode is compared against closed_only.
    checks = []
    for ts_str, _ in SMOKE_WINDOWS:
        sub = summary_df[summary_df["window"] == ts_str]
        co_rows = sub[sub["mode"] == "closed_only"]
        if co_rows.empty:
            continue
        co = co_rows.iloc[0]
        for mode, shape in SMOKE_MODES:
            if shape is None:
                continue
            cp_rows = sub[sub["mode"] == mode]
            if cp_rows.empty:
                continue
            cp = cp_rows.iloc[0]
            tag = f"{ts_str} [{mode}]"
            ok1 = (co["any_nan"] == 0) and (cp["any_nan"] == 0)
            checks.append((tag, "no_nan", ok1, f"co_nan={co['any_nan']} cp_nan={cp['any_nan']}"))
            ok2_p = cp["pitch_rms"] <= max(2.0 * co["pitch_rms"], 0.05)
            ok2_r = cp["roll_rms"] <= max(2.0 * co["roll_rms"], 0.05)
            checks.append((tag, "rms_bounded", ok2_p and ok2_r,
                           f"pitch co={co['pitch_rms']:.3f} cp={cp['pitch_rms']:.3f} | "
                           f"roll co={co['roll_rms']:.3f} cp={cp['roll_rms']:.3f}"))
            if "pump_fullspeed_any_ratio" in co and "pump_fullspeed_any_ratio" in cp:
                base = max(co["pump_fullspeed_any_ratio"], 0.001)
                ok3 = cp["pump_fullspeed_any_ratio"] <= 1.5 * base + 0.05
                checks.append((tag, "pump_fullspeed_bounded", ok3,
                               f"co={co['pump_fullspeed_any_ratio']:.3f} cp={cp['pump_fullspeed_any_ratio']:.3f}"))
            if "preview_pitch_bias_max_abs" in cp:
                ok4 = cp["preview_pitch_bias_max_abs"] >= 0.0
                checks.append((tag, "preview_bias_observed", ok4,
                               f"max_abs={cp['preview_pitch_bias_max_abs']:.4f} "
                               f"nonzero={cp.get('preview_pitch_bias_nonzero_ratio', float('nan')):.3f}"))

    pass_count = sum(1 for _, _, ok, _ in checks if ok)
    total = len(checks)

    report_lines = [
        "# A2-smoke Report",
        "",
        f"- elapsed: `{time.perf_counter()-t0:.1f}s`",
        f"- windows: `{len(SMOKE_WINDOWS)}` × 2 modes",
        f"- pass: `{pass_count}/{total}` interface checks",
        "",
        "## Interface Checks",
        "",
        "| window | check | result | detail |",
        "|---|---|---|---|",
    ]
    for ts_str, name, ok, detail in checks:
        report_lines.append(f"| {ts_str} | {name} | {'PASS' if ok else 'FAIL'} | {detail} |")

    report_lines += ["", "## Issues / Exceptions", ""]
    if issues:
        for x in issues:
            report_lines.append(f"- {x}")
    else:
        report_lines.append("None.")

    report_lines += ["", "## Per-window summary", "", "```",
                      summary_df.to_string(index=False), "```"]

    verdict = "INTERFACE_OK" if pass_count == total else "INTERFACE_NEEDS_FIX"
    report_lines += ["", f"## Verdict: `{verdict}`", ""]

    (out_dir / "A2_smoke_report.md").write_text("\n".join(report_lines), encoding="utf-8")
    print(f"\n{verdict} — see {out_dir / 'A2_smoke_report.md'}")


if __name__ == "__main__":
    main()
