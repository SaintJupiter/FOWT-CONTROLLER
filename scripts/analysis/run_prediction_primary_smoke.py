#!/usr/bin/env python3
"""Smoke test for the default-off prediction-primary prototype.

This is intentionally tiny: one onset window, one hour, three variants. The
goal is only to verify that prediction-primary can bypass the PI/setpoint path
and directly set mass targets before the plant pump latch/rate layer.
"""
from __future__ import annotations

import json
import sys
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


DT = 1.0
N_STEPS = 3600
TIMESTAMP_FMT = "%Y-%m-%d %H:%M:%S"


def discover_excel() -> str:
    for d in (repo_root / "data", repo_root / "archive" / "legacy_fowt_control" / "data"):
        for x in d.rglob("*.xlsx"):
            if "stiffness" in x.name.lower() or "matrix" in x.name.lower():
                return str(x)
    raise FileNotFoundError("stiffness/matrix xlsx not found")


def summarize(df: pd.DataFrame) -> dict:
    pitch = df.get("pitch_deg", pd.Series(np.zeros(len(df)))).to_numpy(dtype=float)
    roll = df.get("roll_deg", pd.Series(np.zeros(len(df)))).to_numpy(dtype=float)
    pump = df.get("pump_total_rate_m3_min", pd.Series(np.zeros(len(df)))).abs().to_numpy(dtype=float)
    primary = df.get("preview_primary_applied", pd.Series(np.zeros(len(df)))).to_numpy(dtype=float)
    primary_delta = df.get("preview_primary_delta_mean_kg", pd.Series(np.zeros(len(df)))).to_numpy(dtype=float)
    return {
        "pitch_p95": float(np.percentile(np.abs(pitch), 95)),
        "roll_p95": float(np.percentile(np.abs(roll), 95)),
        "pump_work_m3": float(np.trapezoid(pump, dx=DT) / 60.0),
        "pump_duty_ratio": float(np.mean(pump > 1e-6)),
        "primary_applied_ratio": float(np.mean(primary > 0.5)),
        "primary_delta_mean_kg": float(np.mean(np.abs(primary_delta))),
        "primary_delta_max_kg": float(np.max(np.abs(primary_delta))) if len(primary_delta) else 0.0,
    }


def main() -> None:
    out_dir = repo_root / "outputs" / "wind_prediction" / "prediction_primary_smoke"
    out_dir.mkdir(parents=True, exist_ok=True)

    base_planner_out = repo_root / "outputs" / "wind_prediction" / "planner_a1_dryrun"
    sign_cfg = json.loads((base_planner_out / "diagnostics" / "a01_pressure_vec_sign_convention.json").read_text())
    discount_cfg = json.loads((base_planner_out / "diagnostics" / "a1_block_discount_config.json").read_text())
    cfg = PlannerConfig(pressure_sign_multiplier=float(sign_cfg["planner_sign_multiplier"]))
    discounts = discount_cfg["default_discount_blocks"]

    replay = Fino1ReplayDataset(
        dataset_dir=repo_root / "data" / "processed" / "wind_ml_10min" / "ballast_decision_fino1_meteo_aux_v1",
        split="test",
    )
    onset_df = pd.read_csv(repo_root / "outputs" / "wind_prediction" / "transient_onset_scan" / "scan_top_onset.csv")
    ts_str = str(onset_df.iloc[0]["timestamp"])
    ts = datetime.strptime(ts_str, TIMESTAMP_FMT)
    wind_trace = replay.build_wind_trace(start_timestamp=ts, row_count=N_STEPS // 60, dt_s=DT)

    from defaults import clone_cfg  # noqa: F401
    from run_validation import run_closed_loop_case

    variants = [
        ("closed_only", None),
        (
            "setpoint_econ",
            {
                "bias_shape": "event_decay",
                "objective_mode": "economic",
                "setpoint_channel_enabled": True,
                "prediction_primary_enabled": False,
                "ff_channel_enabled": False,
            },
        ),
        (
            "prediction_primary_econ",
            {
                "bias_shape": "event_decay",
                "objective_mode": "economic",
                "setpoint_channel_enabled": False,
                "prediction_primary_enabled": True,
                "prediction_primary_scale": 1.0,
                "ff_channel_enabled": False,
            },
        ),
    ]

    rows = []
    issues = []
    for tag, provider_kw in variants:
        provider = None
        if provider_kw is not None:
            provider = BallastPlannerPreviewProvider(
                replay_dataset=replay,
                start_timestamp=ts,
                cfg=cfg,
                block_discounts=discounts,
                **provider_kw,
            )
        case_name = f"{ts_str.replace(':', '').replace(' ', '_')}_{tag}"
        print(f"--- {case_name} ---")
        try:
            _, timeseries = run_closed_loop_case(
                excel_path=discover_excel(),
                case_name=case_name,
                dt=DT,
                n_steps=N_STEPS,
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
            print(f"[FAIL] {e}")
            continue
        df = pd.DataFrame(timeseries)
        df.to_csv(out_dir / f"{case_name}_timeseries.csv", index=False)
        row = {"variant": tag, "window": ts_str}
        row.update(summarize(df))
        if provider is not None and provider.records:
            pd.DataFrame(provider.records).to_csv(out_dir / f"{case_name}_planner_log.csv", index=False)
            row["planner_actions"] = "|".join([r["first_action"] for r in provider.records[:6]])
        rows.append(row)
        print(row)

    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / "prediction_primary_smoke_summary.csv", index=False)
    lines = [
        "# Prediction-primary smoke",
        "",
        f"- window: `{ts_str}`",
        "- duration: `1h`",
        "- purpose: verify mass-domain prediction-primary path is wired, not prove final savings",
        "",
        "| variant | pump_work | pitch_p95 | roll_p95 | primary_applied | primary_delta_mean | primary_delta_max |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for _, r in summary.iterrows():
        lines.append(
            f"| {r['variant']} | {r['pump_work_m3']:.2f} | {r['pitch_p95']:.3f} | "
            f"{r['roll_p95']:.3f} | {r['primary_applied_ratio']:.2f} | "
            f"{r['primary_delta_mean_kg']:.1f} | {r['primary_delta_max_kg']:.1f} |"
        )
    lines += ["", "## Issues", ""]
    lines += [f"- {x}" for x in issues] if issues else ["None."]
    (out_dir / "prediction_primary_smoke_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Report: {out_dir / 'prediction_primary_smoke_report.md'}")


if __name__ == "__main__":
    main()
