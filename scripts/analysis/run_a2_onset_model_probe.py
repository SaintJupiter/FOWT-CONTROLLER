#!/usr/bin/env python3
"""Probe onset planner variants before running a larger batch.

This script keeps the original A2-onset protocol but adds two checks:
  1. event_decay reset semantics: original action-only vs active-bucket pulse.
  2. forecast source: oracle future vs a trained LSTM/GRU forecast adapter.
"""
from __future__ import annotations

import argparse
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
from wind_prediction.forecast_adapter import ForecastModelAdapter
from wind_prediction.replay_dataset import Fino1ReplayDataset

TIMESTAMP_FMT = "%Y-%m-%d %H:%M:%S"
DT = 1.0
N_STEPS = 3600


def discover_excel() -> str:
    for d in (repo_root / "data", repo_root / "archive" / "legacy_fowt_control" / "data"):
        for x in d.rglob("*.xlsx"):
            if "stiffness" in x.name.lower() or "matrix" in x.name.lower():
                return str(x)
    raise FileNotFoundError("No stiffness/matrix .xlsx found")


def percentiles(x: np.ndarray) -> dict[str, float]:
    a = np.abs(x)
    return {
        "abs_p95": float(np.percentile(a, 95)),
        "abs_max": float(a.max()),
        "rms": float(np.sqrt(np.mean(x ** 2))),
    }


def summarize(df: pd.DataFrame) -> dict[str, float]:
    n = len(df)
    pitch = df.get("pitch_deg", pd.Series(np.zeros(n))).to_numpy(dtype=float)
    roll = df.get("roll_deg", pd.Series(np.zeros(n))).to_numpy(dtype=float)
    pump_rate = df.get("pump_total_rate_m3_min", pd.Series(np.zeros(n))).abs().to_numpy(dtype=float)
    first_n = min(3600, n)
    out = {
        "pump_work_m3": float(np.trapezoid(pump_rate, dx=DT) / 60.0),
        "pump_duty_ratio": float(np.mean(pump_rate > 1e-6)),
        "pump_work_first_hour": float(np.trapezoid(pump_rate[:first_n], dx=DT) / 60.0),
        "preview_pitch_bias_max": float(df.get("preview_pitch_bias_deg", pd.Series(np.zeros(n))).abs().max()),
        "pitch_sp_max": float(df.get("pitch_sp_deg", pd.Series(np.zeros(n))).abs().max()),
    }
    out.update({f"pitch_{k}": v for k, v in percentiles(pitch).items()})
    out.update({f"roll_{k}": v for k, v in percentiles(roll).items()})
    out["pitch_first_hour_p95"] = float(np.percentile(np.abs(pitch[:first_n]), 95))
    out["pitch_first_hour_max"] = float(np.max(np.abs(pitch[:first_n])))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-n", type=int, default=6)
    parser.add_argument(
        "--model-dirs",
        nargs="*",
        default=[
            "outputs/wind_prediction/lstm_segmented_fino1_meteo_aux_v1",
            "outputs/wind_prediction/gru_segmented_fino1_meteo_aux_v1",
        ],
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=repo_root / "outputs" / "wind_prediction" / "planner_a2_onset_model_probe",
    )
    parser.add_argument(
        "--variant-set",
        choices=["all", "action_only"],
        default="all",
        help="Use action_only for a faster top-N comparison without active-bucket diagnostics.",
    )
    args = parser.parse_args()

    t0 = time.perf_counter()
    out_dir = args.out_dir
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
    onset_df = onset_df.head(int(args.top_n))
    excel_path = discover_excel()

    from run_validation import run_closed_loop_case

    adapters: dict[str, ForecastModelAdapter | None] = {"oracle": None}
    for model_dir in args.model_dirs:
        model_path = repo_root / model_dir
        name = model_path.name
        adapters[name] = ForecastModelAdapter(model_dir=model_path, device="cpu")

    variants = [("closed_only", None, None, None, None)]
    variants.append(("econ_oracle_action_reset", "event_decay", "economic", "action", "oracle"))
    if args.variant_set == "all":
        variants.append(("econ_oracle_active_bucket", "event_decay", "economic", "active_bucket", "oracle"))
    for name in adapters:
        if name == "oracle":
            continue
        variants.append((f"econ_{name}_action_reset", "event_decay", "economic", "action", name))
        if args.variant_set == "all":
            variants.append((f"econ_{name}_active_bucket", "event_decay", "economic", "active_bucket", name))

    summary_rows: list[dict] = []
    for _, win in onset_df.iterrows():
        ts_str = str(win["timestamp"])
        ts = datetime.strptime(ts_str, TIMESTAMP_FMT)
        wind_trace = replay.build_wind_trace(start_timestamp=ts, row_count=N_STEPS // 60, dt_s=DT)
        for mode_tag, bias_shape, obj_mode, reset_mode, adapter_name in variants:
            provider = None
            if bias_shape is not None:
                provider = BallastPlannerPreviewProvider(
                    replay_dataset=replay,
                    start_timestamp=ts,
                    cfg=cfg,
                    block_discounts=discounts,
                    bias_shape=bias_shape,
                    objective_mode=obj_mode,
                    event_reset_mode=reset_mode,
                    forecast_adapter=adapters[adapter_name],
                )
            case_name = f"{ts_str.replace(':', '').replace(' ', '_')}_{mode_tag}"
            _, timeseries = run_closed_loop_case(
                excel_path=excel_path,
                case_name=case_name,
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
            df = pd.DataFrame(timeseries)
            row = summarize(df)
            row.update({
                "window": ts_str,
                "mode": mode_tag,
                "forecast_source": adapter_name or "none",
                "event_reset_mode": reset_mode or "none",
                "speed_t0": float(win["speed_t0"]),
                "speed_block2": float(win["speed_block2"]),
            })
            if provider is not None and provider.records:
                pd.DataFrame(provider.records).to_csv(out_dir / f"{case_name}_planner_log.csv", index=False)
                actions = [r["first_action"] for r in provider.records]
                row["planner_first_actions_first10"] = "|".join(actions[:10])
                row["planner_active_ratio_first_hour"] = float(np.mean([a != "hold" for a in actions[:6]]))
                row["planner_active_ratio"] = float(np.mean([a != "hold" for a in actions]))
            summary_rows.append(row)
            print(
                f"{ts_str} {mode_tag}: pitch1h_p95={row['pitch_first_hour_p95']:.3f} "
                f"work1h={row['pump_work_first_hour']:.1f} work={row['pump_work_m3']:.1f}"
            )

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out_dir / "A2_onset_model_probe_summary.csv", index=False)

    delta_rows = []
    for ts_str in summary["window"].unique():
        sub = summary[summary["window"] == ts_str]
        base = sub[sub["mode"] == "closed_only"].iloc[0]
        for _, r in sub.iterrows():
            if r["mode"] == "closed_only":
                continue
            delta_rows.append({
                "window": ts_str,
                "mode": r["mode"],
                "d_pitch_first_hour_p95": r["pitch_first_hour_p95"] - base["pitch_first_hour_p95"],
                "d_pitch_first_hour_max": r["pitch_first_hour_max"] - base["pitch_first_hour_max"],
                "d_pump_work_first_hour_pct": (r["pump_work_first_hour"] - base["pump_work_first_hour"]) / max(base["pump_work_first_hour"], 1e-9) * 100.0,
                "d_pump_work_total_pct": (r["pump_work_m3"] - base["pump_work_m3"]) / max(base["pump_work_m3"], 1e-9) * 100.0,
            })
    delta = pd.DataFrame(delta_rows)
    delta.to_csv(out_dir / "A2_onset_model_probe_delta.csv", index=False)
    metric_cols = [
        "d_pitch_first_hour_p95",
        "d_pitch_first_hour_max",
        "d_pump_work_first_hour_pct",
        "d_pump_work_total_pct",
    ]
    agg = delta.groupby("mode")[metric_cols].agg(["count", "mean", "std"])
    agg.to_csv(out_dir / "A2_onset_model_probe_aggregate.csv")
    print(f"elapsed={time.perf_counter() - t0:.1f}s")
    print(agg)


if __name__ == "__main__":
    main()
