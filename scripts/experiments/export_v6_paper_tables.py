#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Supplement frozen v6 paper tables.")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results/v6_paper_ready_test_20260501_221851"),
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("data/processed/wind_ml_10min/ballast_decision_fino1_meteo_aux_v1"),
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("outputs/wind_prediction/lstm_segmented_fino1_meteo_aux_v1"),
    )
    parser.add_argument(
        "--v6-results-dir",
        type=Path,
        default=Path("results/weight_sensitivity_v6_test_20260501_212519"),
    )
    parser.add_argument("--split", default="test")
    parser.add_argument("--dt", type=float, default=1.0)
    parser.add_argument("--replay-rows", type=int, default=24)
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _safe_float(row: dict[str, Any], key: str) -> float:
    try:
        return float(row.get(key, 0.0))
    except Exception:
        return 0.0


def summarize_timeseries(
    timeseries_rows: list[dict[str, Any]],
    dt: float,
    deadband_pitch_deg: float,
    deadband_roll_deg: float,
    tank_capacity_kg: float,
) -> dict[str, float]:
    pitch = np.asarray([_safe_float(r, "pitch_deg") for r in timeseries_rows], dtype=float)
    roll = np.asarray([_safe_float(r, "roll_deg") for r in timeseries_rows], dtype=float)
    pump_total = np.asarray([_safe_float(r, "pump_total_rate_m3_min") for r in timeseries_rows], dtype=float)
    tank1 = np.asarray([_safe_float(r, "tank1_kg") for r in timeseries_rows], dtype=float)
    tank2 = np.asarray([_safe_float(r, "tank2_kg") for r in timeseries_rows], dtype=float)
    tank3 = np.asarray([_safe_float(r, "tank3_kg") for r in timeseries_rows], dtype=float)
    latch_switch = np.asarray([_safe_float(r, "pump_latch_switch_count") for r in timeseries_rows], dtype=float)

    active = np.abs(pump_total) > 1e-6
    switch_count = 0 if active.size <= 1 else int(np.sum(active[1:] != active[:-1]))
    total_minutes = max(len(timeseries_rows) * dt / 60.0, 1e-9)

    pitch_peak = float(np.max(np.abs(pitch))) if pitch.size else 0.0
    roll_peak = float(np.max(np.abs(roll))) if roll.size else 0.0
    pose_exceed_time_s = float(
        np.sum((np.abs(pitch) > deadband_pitch_deg) | (np.abs(roll) > deadband_roll_deg)) * dt
    )
    pump_work_m3 = float(np.sum(np.abs(pump_total)) * dt / 60.0)
    mass_ratio = np.vstack([tank1, tank2, tank3]) / max(tank_capacity_kg, 1e-6)
    lower_margin = np.min(mass_ratio, axis=0) if mass_ratio.size else np.array([1.0])
    upper_margin = np.min(1.0 - mass_ratio, axis=0) if mass_ratio.size else np.array([1.0])
    capacity_headroom_min_ratio = float(np.min(np.minimum(lower_margin, upper_margin))) if lower_margin.size else 1.0

    return {
        "pitch_peak_abs_deg": pitch_peak,
        "pitch_rms_deg": float(np.sqrt(np.mean(np.square(pitch)))) if pitch.size else 0.0,
        "roll_peak_abs_deg": roll_peak,
        "roll_rms_deg": float(np.sqrt(np.mean(np.square(roll)))) if roll.size else 0.0,
        "pose_exceed_time_s": pose_exceed_time_s,
        "pump_work_m3": pump_work_m3,
        "startstop_count": float(np.max(latch_switch)) if latch_switch.size else 0.0,
        "switch_per_min": float(switch_count / total_minutes),
        "capacity_headroom_min_ratio": capacity_headroom_min_ratio,
    }


def relative_change(candidate: float, baseline: float, eps: float = 1e-6) -> float:
    denom = max(abs(float(baseline)), eps)
    return float((float(candidate) - float(baseline)) / denom)


def choose_conservative(v6_results_dir: Path) -> str:
    rows = read_csv_rows(v6_results_dir / "paper_ready" / "tables" / "holdout_summary_p95.csv")
    candidates = [r for r in rows if r["config_name"] in {"pump_priority", "switch_smooth"}]
    candidates.sort(
        key=lambda r: (
            float(r["burden_index"]),
            -float(r["pass_ratio"]),
            float(r["pose_index"]),
            0 if r["config_name"] == "pump_priority" else 1,
        )
    )
    return str(candidates[0]["config_name"])


def build_boundary_summary(
    grouped_rows: list[dict[str, Any]],
    v6_results_dir: Path,
) -> list[dict[str, Any]]:
    grouped_map = {(r["window_group"], r["strategy"]): r for r in grouped_rows}
    run_summary = (v6_results_dir / "run_summary.md").read_text(encoding="utf-8")
    holdout_status = "fail" if "hold-out validation: `fail`" in run_summary else "pass"
    profile_status = "fail" if "structure-profile sensitivity check: `fail`" in run_summary else "pass"
    scale_status = "fail" if "scale-definition sensitivity check: `fail`" in run_summary else "pass"
    out: list[dict[str, Any]] = []
    for strategy in ("v6_balanced", "v6_conservative"):
        row = grouped_map[("high_event", strategy)]
        out.append(
            {
                "topic": "high_event",
                "strategy": strategy,
                "status": "positive" if float(row["delta_pose_exceed_time_pct"]) < 0.0 else "mixed",
                "evidence": (
                    f"pitch_rms={row['delta_pitch_rms_pct']:.2f}%, "
                    f"roll_rms={row['delta_roll_rms_pct']:.2f}%, "
                    f"pose_exceed_time={row['delta_pose_exceed_time_pct']:.2f}%, "
                    f"pump_work={row['delta_pump_work_pct']:.2f}%"
                ),
            }
        )
    for strategy in ("v6_balanced", "v6_conservative"):
        row = grouped_map[("combined_variability", strategy)]
        out.append(
            {
                "topic": "combined_variability",
                "strategy": strategy,
                "status": "boundary" if float(row["delta_pump_work_pct"]) > 0.0 or float(row["delta_sat_ratio_pct"]) > 0.0 else "positive",
                "evidence": (
                    f"pitch_rms={row['delta_pitch_rms_pct']:.2f}%, "
                    f"roll_rms={row['delta_roll_rms_pct']:.2f}%, "
                    f"pose_exceed_time={row['delta_pose_exceed_time_pct']:.2f}%, "
                    f"pump_work={row['delta_pump_work_pct']:.2f}%, "
                    f"sat_ratio={row['delta_sat_ratio_pct']:.2f}%"
                ),
            }
        )
    out.extend(
        [
            {
                "topic": "holdout_stability",
                "strategy": "v6_global",
                "status": holdout_status,
                "evidence": "derived from weight_sensitivity_v6 run_summary.md",
            },
            {
                "topic": "profile_sensitivity",
                "strategy": "v6_global",
                "status": profile_status,
                "evidence": "derived from weight_sensitivity_v6 run_summary.md",
            },
            {
                "topic": "scale_sensitivity",
                "strategy": "v6_global",
                "status": scale_status,
                "evidence": "derived from weight_sensitivity_v6 run_summary.md",
            },
        ]
    )
    return out


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root / "src"))
    sys.path.insert(0, str(repo_root / "archive" / "legacy_fowt_control"))

    from defaults import DEFAULT_CONTROLLER_CFG
    from run_validation import discover_stiffness_file, run_closed_loop_case
    from wind_prediction import (
        CandidatePlanEvaluator,
        DEFAULT_STRUCTURE_PROFILES,
        DEFAULT_WEIGHT_PROFILES,
        Fino1ReplayDataset,
        ForecastModelAdapter,
        ModelDecisionProvider,
        BaselineScales,
    )

    excel_path = discover_stiffness_file()
    if not excel_path:
        archive_candidate = repo_root / "archive" / "legacy_fowt_control" / "data" / "副本水平刚度曲线.xlsx"
        if archive_candidate.exists():
            excel_path = str(archive_candidate)
    if not excel_path:
        raise SystemExit("No stiffness file found under data/.")

    tables_dir = args.results_dir / "paper_ready" / "tables"
    debug_dir = args.results_dir / "debug"
    tables_dir.mkdir(parents=True, exist_ok=True)

    replay = Fino1ReplayDataset(dataset_dir=args.dataset_dir, split=args.split)
    adapter = ForecastModelAdapter(model_dir=args.model_dir, dataset_dir=args.dataset_dir, device="cpu")
    conservative_name = choose_conservative(args.v6_results_dir)
    weight_cfg_rows = read_csv_rows(args.v6_results_dir / "paper_ready" / "tables" / "weight_configs.csv")
    cfg_by_name = {r["name"]: r for r in weight_cfg_rows}
    balanced_cfg = cfg_by_name["balanced"]
    conservative_cfg = cfg_by_name[conservative_name]

    windows = [
        row for row in read_csv_rows(args.v6_results_dir / "paper_ready" / "tables" / "window_plan.csv")
        if row["window_label"] in {"high_event", "combined_variability"}
    ]

    deadband_pitch_deg = float(DEFAULT_CONTROLLER_CFG["deadband_pitch"])
    deadband_roll_deg = float(DEFAULT_CONTROLLER_CFG["deadband_roll"])
    tank_capacity_kg = 1850.0 * 1025.0

    absolute_rows: list[dict[str, Any]] = []
    sign_records: list[dict[str, Any]] = []

    for window in windows:
        start_timestamp = datetime.strptime(window["start_timestamp"], "%Y-%m-%d %H:%M:%S")
        wind_trace = replay.build_wind_trace(
            start_timestamp=start_timestamp,
            row_count=args.replay_rows,
            dt_s=args.dt,
        )
        n_steps = int(wind_trace["n_steps"])

        closed_row, closed_timeseries = run_closed_loop_case(
            excel_path=excel_path,
            case_name=f"export_v6_{window['window_label']}_{window['split_role']}_closed_only",
            dt=args.dt,
            n_steps=n_steps,
            wind_trace=wind_trace,
            trim_cfg=None,
            control_enabled=True,
            record_timeseries=True,
            preview_trim_provider=None,
        )
        closed_metrics = {
            **closed_row,
            **summarize_timeseries(
                closed_timeseries,
                dt=args.dt,
                deadband_pitch_deg=deadband_pitch_deg,
                deadband_roll_deg=deadband_roll_deg,
                tank_capacity_kg=tank_capacity_kg,
            ),
        }
        absolute_rows.append(
            {
                "strategy": "closed_only",
                "window_id": f"{window['window_label']}__{window['split_role']}",
                "window_group": window["window_label"],
                "split_role": window["split_role"],
                "pitch_peak": float(closed_metrics["pitch_peak_abs_deg"]),
                "pitch_rms": float(closed_metrics["pitch_rms_deg"]),
                "roll_peak": float(closed_metrics["roll_peak_abs_deg"]),
                "roll_rms": float(closed_metrics["roll_rms_deg"]),
                "pose_exceed_time": float(closed_metrics["pose_exceed_time_s"]),
                "pump_work": float(closed_metrics["pump_work_m3"]),
                "startstop_count": float(closed_metrics["startstop_count"]),
                "switch_per_min": float(closed_metrics["switch_per_min"]),
                "sat_ratio": float(closed_metrics["sat_ratio"]),
            }
        )

        baseline_scales = BaselineScales(
            pitch_deg=max(deadband_pitch_deg, 1e-3),
            roll_deg=max(deadband_roll_deg, 1e-3),
            pump_rate_m3_min=max(float(np.nanpercentile(np.abs([_safe_float(r, "pump_total_rate_m3_min") for r in closed_timeseries]), 95.0)), 1.0),
            backlog_kg=max(float(np.nanpercentile(np.abs([_safe_float(r, "pump_total_backlog_kg") for r in closed_timeseries]), 95.0)), 1.0),
            wind_speed_mps=max(float(np.nanpercentile(np.abs([_safe_float(r, "wind_speed") for r in closed_timeseries]), 95.0)), 1.0),
            deadband_pitch_deg=deadband_pitch_deg,
            deadband_roll_deg=deadband_roll_deg,
            saturation_margin_ratio=0.08,
            eps=1e-6,
        )

        for strategy_name, cfg in [("v6_balanced", balanced_cfg), ("v6_conservative", conservative_cfg)]:
            evaluator = CandidatePlanEvaluator(
                baseline_scales=baseline_scales,
                weight_profile=DEFAULT_WEIGHT_PROFILES[cfg["weight_profile_name"]],
                structure_profile=DEFAULT_STRUCTURE_PROFILES[cfg["structure_profile_name"]],
                active_ratio_low=float(cfg["active_ratio_low"]),
                active_ratio_high=float(cfg["active_ratio_high"]),
                pump_saving_ratio=float(cfg["pump_saving_ratio"]),
            )
            provider = ModelDecisionProvider(
                replay_dataset=replay,
                forecast_adapter=adapter,
                start_timestamp=start_timestamp,
                evaluator=evaluator,
            )
            row, timeseries = run_closed_loop_case(
                excel_path=excel_path,
                case_name=f"export_v6_{window['window_label']}_{window['split_role']}_{strategy_name}",
                dt=args.dt,
                n_steps=n_steps,
                wind_trace=wind_trace,
                trim_cfg=None,
                control_enabled=True,
                record_timeseries=True,
                preview_trim_provider=provider,
            )
            metrics = {
                **row,
                **summarize_timeseries(
                    timeseries,
                    dt=args.dt,
                    deadband_pitch_deg=deadband_pitch_deg,
                    deadband_roll_deg=deadband_roll_deg,
                    tank_capacity_kg=tank_capacity_kg,
                ),
            }
            absolute_rows.append(
                {
                    "strategy": strategy_name,
                    "window_id": f"{window['window_label']}__{window['split_role']}",
                    "window_group": window["window_label"],
                    "split_role": window["split_role"],
                    "pitch_peak": float(metrics["pitch_peak_abs_deg"]),
                    "pitch_rms": float(metrics["pitch_rms_deg"]),
                    "roll_peak": float(metrics["roll_peak_abs_deg"]),
                    "roll_rms": float(metrics["roll_rms_deg"]),
                    "pose_exceed_time": float(metrics["pose_exceed_time_s"]),
                    "pump_work": float(metrics["pump_work_m3"]),
                    "startstop_count": float(metrics["startstop_count"]),
                    "switch_per_min": float(metrics["switch_per_min"]),
                    "sat_ratio": float(metrics["sat_ratio"]),
                }
            )

    write_csv(tables_dir / "absolute_window_metrics.csv", absolute_rows)

    metrics = ["pitch_peak", "pitch_rms", "roll_peak", "roll_rms", "pose_exceed_time", "pump_work", "startstop_count", "switch_per_min", "sat_ratio"]
    by_window = {}
    for row in absolute_rows:
        by_window.setdefault(row["window_id"], {})[row["strategy"]] = row
    for strategy in ("v6_balanced", "v6_conservative"):
        for window_group in ("high_event", "combined_variability"):
            subset = [w for w, rows in by_window.items() if rows["closed_only"]["window_group"] == window_group and strategy in rows]
            for metric in metrics:
                better = worse = equal = 0
                for window_id in subset:
                    base = float(by_window[window_id]["closed_only"][metric])
                    cand = float(by_window[window_id][strategy][metric])
                    delta = cand - base
                    if abs(delta) <= 1e-9:
                        equal += 1
                    elif delta < 0.0:
                        better += 1
                    else:
                        worse += 1
                sign_records.append(
                    {
                        "strategy": strategy,
                        "window_group": window_group,
                        "metric": metric,
                        "n_better": better,
                        "n_worse": worse,
                        "n_equal": equal,
                        "n_windows": len(subset),
                    }
                )
    write_csv(tables_dir / "sign_consistency_summary.csv", sign_records)

    grouped_rows = read_csv_rows(args.results_dir / "paper_ready" / "tables" / "grouped_change_rates.csv")
    grouped_for_boundary = []
    for row in grouped_rows:
        grouped_for_boundary.append(
            {
                "window_group": row["window_label"],
                "strategy": row["config_name"],
                "delta_pitch_rms_pct": float(row["delta_pitch_rms_pct"]),
                "delta_roll_rms_pct": float(row["delta_roll_rms_pct"]),
                "delta_pose_exceed_time_pct": float(row["delta_pose_exceed_time_pct"]),
                "delta_pump_work_pct": float(row["delta_pump_work_pct"]),
                "delta_sat_ratio_pct": float(row["delta_sat_ratio_pct"]),
            }
        )
    write_csv(
        tables_dir / "v6_sensitivity_boundary_summary.csv",
        build_boundary_summary(grouped_for_boundary, args.v6_results_dir),
    )

    print(f"Supplemented tables under {tables_dir}")


if __name__ == "__main__":
    main()
