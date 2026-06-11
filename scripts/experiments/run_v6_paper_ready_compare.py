#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib-cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build paper-ready fixed-window comparison for frozen v6 decision logic."
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
    parser.add_argument("--split", default="test")
    parser.add_argument("--dt", type=float, default=1.0)
    parser.add_argument("--replay-rows", type=int, default=24)
    parser.add_argument(
        "--v6-results-dir",
        type=Path,
        default=Path("results/weight_sensitivity_v6_test_20260501_212519"),
    )
    parser.add_argument("--case-prefix", default="v6_paper_ready_compare")
    parser.add_argument(
        "--conservative-candidate",
        choices=["auto", "pump_priority", "switch_smooth"],
        default="auto",
    )
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


def _percentile_scale(values: list[float], q: float, floor: float) -> float:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return float(floor)
    val = float(np.nanpercentile(np.abs(arr), q))
    return max(float(floor), val)


def estimate_baseline_scales(timeseries_rows: list[dict[str, Any]], deadband_pitch_deg: float, deadband_roll_deg: float):
    from wind_prediction import BaselineScales

    return BaselineScales(
        pitch_deg=_percentile_scale([_safe_float(r, "pitch_deg") for r in timeseries_rows], 95.0, 1e-3),
        roll_deg=_percentile_scale([_safe_float(r, "roll_deg") for r in timeseries_rows], 95.0, 1e-3),
        pump_rate_m3_min=_percentile_scale(
            [_safe_float(r, "pump_total_rate_m3_min") for r in timeseries_rows], 95.0, 1.0
        ),
        backlog_kg=_percentile_scale(
            [_safe_float(r, "pump_total_backlog_kg") for r in timeseries_rows], 95.0, 1.0
        ),
        wind_speed_mps=_percentile_scale([_safe_float(r, "wind_speed") for r in timeseries_rows], 95.0, 1.0),
        deadband_pitch_deg=float(deadband_pitch_deg),
        deadband_roll_deg=float(deadband_roll_deg),
        saturation_margin_ratio=0.08,
        eps=1e-6,
    )


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

    pitch_peak = float(np.max(np.abs(pitch))) if pitch.size else 0.0
    roll_peak = float(np.max(np.abs(roll))) if roll.size else 0.0
    pose_exceed_time_s = float(
        np.sum((np.abs(pitch) > deadband_pitch_deg) | (np.abs(roll) > deadband_roll_deg)) * dt
    )
    pump_work_m3 = float(np.sum(np.abs(pump_total)) * dt / 60.0)

    signs: list[int] = []
    for val in pump_total:
        if abs(float(val)) <= 1e-3:
            continue
        signs.append(1 if val > 0.0 else -1)
    direction_switch_count = 0 if len(signs) <= 1 else int(sum(1 for a, b in zip(signs[:-1], signs[1:]) if a != b))

    mass_ratio = np.vstack([tank1, tank2, tank3]) / max(tank_capacity_kg, 1e-6)
    lower_margin = np.min(mass_ratio, axis=0) if mass_ratio.size else np.array([1.0])
    upper_margin = np.min(1.0 - mass_ratio, axis=0) if mass_ratio.size else np.array([1.0])
    capacity_headroom_min_ratio = float(np.min(np.minimum(lower_margin, upper_margin))) if lower_margin.size else 1.0

    return {
        "pitch_peak_abs_deg": pitch_peak,
        "roll_peak_abs_deg": roll_peak,
        "pose_exceed_time_s": pose_exceed_time_s,
        "pump_work_m3": pump_work_m3,
        "direction_switch_count": float(direction_switch_count),
        "capacity_headroom_min_ratio": capacity_headroom_min_ratio,
    }


def relative_change(candidate: float, baseline: float, eps: float = 1e-6) -> float:
    denom = max(abs(float(baseline)), eps)
    return float((float(candidate) - float(baseline)) / denom)


def choose_conservative(v6_results_dir: Path, requested: str) -> str:
    if requested != "auto":
        return requested
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


def plot_window_compare(window_label: str, split_role: str, series_map: dict[str, list[dict[str, Any]]], output_path: Path) -> None:
    fig, axes = plt.subplots(4, 1, figsize=(11, 10), sharex=True)
    order = ["closed_only", "v6_balanced", "v6_conservative"]
    for label in order:
        rows = series_map[label]
        t_min = np.array([_safe_float(r, "t_s") / 60.0 for r in rows], dtype=float)
        axes[0].plot(t_min, [_safe_float(r, "wind_speed") for r in rows], label=label, linewidth=1.8)
        axes[1].plot(t_min, [_safe_float(r, "pitch_deg") for r in rows], label=label, linewidth=1.6)
        axes[2].plot(t_min, [_safe_float(r, "roll_deg") for r in rows], label=label, linewidth=1.6)
        axes[3].plot(t_min, [_safe_float(r, "pump_total_rate_m3_min") for r in rows], label=label, linewidth=1.6)
    axes[0].set_ylabel("Wind (m/s)")
    axes[1].set_ylabel("Pitch (deg)")
    axes[2].set_ylabel("Roll (deg)")
    axes[3].set_ylabel("Pump rate")
    axes[3].set_xlabel("Time (min)")
    axes[0].set_title(f"{window_label} / {split_role}")
    for ax in axes:
        ax.grid(True, alpha=0.25)
    axes[0].legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def classify_outcome(group_rows: list[dict[str, Any]], overall_rows: list[dict[str, Any]]) -> str:
    balanced_rows = [r for r in group_rows if r["config_name"] == "v6_balanced"]
    if not balanced_rows:
        return "边界性效果"

    pose_positive_count = 0
    strict_positive_count = 0
    for row in balanced_rows:
        pose_gain = (
            float(row["delta_pitch_rms_pct"]) < 0.0
            or float(row["delta_roll_rms_pct"]) < 0.0
            or float(row["delta_pose_exceed_time_pct"]) < 0.0
        )
        if pose_gain:
            pose_positive_count += 1
        burden_controlled = (
            float(row["delta_pump_work_pct"]) <= 1.0
            and float(row["delta_switch_per_min_pct"]) <= 1.0
            and float(row["delta_sat_ratio_pct"]) <= 2.0
            and float(row["delta_roll_peak_pct"]) <= 1.0
        )
        if pose_gain and burden_controlled:
            strict_positive_count += 1

    conservative_clear = any(
        float(row["balanced_minus_conservative_pump_work_pct"]) >= 0.5
        or float(row["balanced_minus_conservative_switch_pct"]) >= 0.5
        or float(row["balanced_minus_conservative_sat_pct"]) >= 0.5
        for row in overall_rows
    )

    if strict_positive_count == len(balanced_rows) and conservative_clear:
        return "正向效果"
    if pose_positive_count > 0:
        return "边界性效果"
    return "边界性效果"


def build_run_summary(
    conservative_name: str,
    actual_wall_seconds: float,
    grouped_rows: list[dict[str, Any]],
    overall_rows: list[dict[str, Any]],
) -> str:
    classification = classify_outcome(grouped_rows, overall_rows)
    lines = [
        "# v6 Paper-Ready Comparison Summary",
        "",
        f"- actual wall time: `{actual_wall_seconds:.1f} s`",
        "- frozen decision version: `v6`",
        "- note: `v6 is treated as the current paper-ready conservative feasible version, not a final optimal parameter set`",
        "- fixed baseline: `closed-only`",
        "- fixed balanced config: `balanced`",
        f"- fixed conservative config: `{conservative_name}`",
        "- fixed structure / scale for this comparison: `near_focus + p95`",
        "",
        "## Why We Stop Tuning Here",
        "",
        "- this round does not continue evaluator tuning or weight search",
        "- v5/v6 already showed the main trade-off: looser active triggering raises separation but increases pump burden; tighter triggering lowers pump burden but weakens separation",
        "- the current task is to report a stable, honest paper-ready comparison rather than continue local repairs",
        "",
        "## Fixed-Window Outcomes",
        "",
    ]
    for row in grouped_rows:
        lines.extend(
            [
                f"### {row['window_label']} / {row['config_name']}",
                f"- pitch_rms change vs closed-only: `{row['delta_pitch_rms_pct']:.2f}%`",
                f"- roll_rms change vs closed-only: `{row['delta_roll_rms_pct']:.2f}%`",
                f"- pitch_peak change vs closed-only: `{row['delta_pitch_peak_pct']:.2f}%`",
                f"- roll_peak change vs closed-only: `{row['delta_roll_peak_pct']:.2f}%`",
                f"- pose_exceed_time change vs closed-only: `{row['delta_pose_exceed_time_pct']:.2f}%`",
                f"- pump_work change vs closed-only: `{row['delta_pump_work_pct']:.2f}%`",
                f"- switch_per_min change vs closed-only: `{row['delta_switch_per_min_pct']:.2f}%`",
                f"- direction_switch_count change vs closed-only: `{row['delta_direction_switch_count_pct']:.2f}%`",
                f"- sat_ratio change vs closed-only: `{row['delta_sat_ratio_pct']:.2f}%`",
                "",
            ]
        )
    lines.extend(
        [
            "## Current Interpretation",
            "",
            f"- current conclusion type: `{classification}`",
            "- balanced vs conservative should be read as a posture-vs-burden trade-off under fixed v6 logic",
            "- if balanced and conservative remain very close, the current result should be framed as a conservative boundary-effect validation rather than a stable optimal controller",
            "- no further CandidatePlanEvaluator repair is performed in this validation round",
            "",
            "## Next Step",
            "",
            "- proceed to paper writing and figure/table organization",
            "- if the effect is modest, frame it as a conservative feasibility / boundary-effect result rather than a stable optimal controller",
        ]
    )
    return "\n".join(lines).strip() + "\n"


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
    )

    t0 = time.perf_counter()
    replay = Fino1ReplayDataset(dataset_dir=args.dataset_dir, split=args.split)
    adapter = ForecastModelAdapter(model_dir=args.model_dir, dataset_dir=args.dataset_dir, device="cpu")

    excel_path = discover_stiffness_file()
    if not excel_path:
        archive_candidate = repo_root / "archive" / "legacy_fowt_control" / "data" / "副本水平刚度曲线.xlsx"
        if archive_candidate.exists():
            excel_path = str(archive_candidate)
    if not excel_path:
        raise SystemExit("No stiffness file found under data/.")

    conservative_name = choose_conservative(args.v6_results_dir, args.conservative_candidate)

    weight_cfg_rows = read_csv_rows(args.v6_results_dir / "paper_ready" / "tables" / "weight_configs.csv")
    cfg_by_name = {r["name"]: r for r in weight_cfg_rows}
    balanced_cfg = cfg_by_name["balanced"]
    conservative_cfg = cfg_by_name[conservative_name]

    windows = [
        row for row in read_csv_rows(args.v6_results_dir / "paper_ready" / "tables" / "window_plan.csv")
        if row["window_label"] in {"high_event", "combined_variability"}
    ]

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = repo_root / "results" / f"{args.case_prefix}_{args.split}_{ts}"
    paper_dir = out_dir / "paper_ready"
    tables_dir = paper_dir / "tables"
    figures_dir = paper_dir / "figures"
    debug_dir = out_dir / "debug"
    candidate_scores_dir = debug_dir / "candidate_scores"
    for p in (paper_dir, tables_dir, figures_dir, debug_dir, candidate_scores_dir):
        p.mkdir(parents=True, exist_ok=True)

    deadband_pitch_deg = float(DEFAULT_CONTROLLER_CFG["deadband_pitch"])
    deadband_roll_deg = float(DEFAULT_CONTROLLER_CFG["deadband_roll"])
    tank_capacity_kg = 1850.0 * 1025.0

    comparison_rows: list[dict[str, Any]] = []
    grouped_rows: list[dict[str, Any]] = []
    overall_rows: list[dict[str, Any]] = []

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
            case_name=f"{args.case_prefix}_{window['window_label']}_{window['split_role']}_closed_only",
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
        baseline_scales = estimate_baseline_scales(
            timeseries_rows=closed_timeseries,
            deadband_pitch_deg=deadband_pitch_deg,
            deadband_roll_deg=deadband_roll_deg,
        )

        timeseries_map = {"closed_only": closed_timeseries}
        write_csv(debug_dir / f"{window['window_label']}_{window['split_role']}_closed_only_timeseries.csv", closed_timeseries)

        for public_name, cfg in [("v6_balanced", balanced_cfg), ("v6_conservative", conservative_cfg)]:
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
                case_name=f"{args.case_prefix}_{window['window_label']}_{window['split_role']}_{public_name}",
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
            rel_row = {
                "window_label": window["window_label"],
                "split_role": window["split_role"],
                "config_name": public_name,
                "source_config_name": cfg["name"],
                "delta_pitch_rms_pct": 100.0 * relative_change(metrics["pitch_rms_deg"], closed_metrics["pitch_rms_deg"]),
                "delta_roll_rms_pct": 100.0 * relative_change(metrics["roll_rms_deg"], closed_metrics["roll_rms_deg"]),
                "delta_pitch_peak_pct": 100.0 * relative_change(metrics["pitch_peak_abs_deg"], closed_metrics["pitch_peak_abs_deg"]),
                "delta_roll_peak_pct": 100.0 * relative_change(metrics["roll_peak_abs_deg"], closed_metrics["roll_peak_abs_deg"]),
                "delta_pose_exceed_time_pct": 100.0 * relative_change(metrics["pose_exceed_time_s"], closed_metrics["pose_exceed_time_s"]),
                "delta_pump_work_pct": 100.0 * relative_change(metrics["pump_work_m3"], closed_metrics["pump_work_m3"]),
                "delta_switch_per_min_pct": 100.0 * relative_change(metrics["switch_per_min"], closed_metrics["switch_per_min"]),
                "delta_direction_switch_count_pct": 100.0 * relative_change(
                    metrics["direction_switch_count"], closed_metrics["direction_switch_count"]
                ),
                "delta_sat_ratio_pct": 100.0 * relative_change(metrics["sat_ratio"], closed_metrics["sat_ratio"]),
                "delta_capacity_headroom_pct": 100.0 * relative_change(
                    metrics["capacity_headroom_min_ratio"], closed_metrics["capacity_headroom_min_ratio"]
                ),
            }
            comparison_rows.append(
                {
                    "window_label": window["window_label"],
                    "split_role": window["split_role"],
                    "config_name": public_name,
                    "source_config_name": cfg["name"],
                    "pitch_rms_deg": metrics["pitch_rms_deg"],
                    "roll_rms_deg": metrics["roll_rms_deg"],
                    "pitch_peak_abs_deg": metrics["pitch_peak_abs_deg"],
                    "roll_peak_abs_deg": metrics["roll_peak_abs_deg"],
                    "pose_exceed_time_s": metrics["pose_exceed_time_s"],
                    "pump_work_m3": metrics["pump_work_m3"],
                    "switch_per_min": metrics["switch_per_min"],
                    "direction_switch_count": metrics["direction_switch_count"],
                    "sat_ratio": metrics["sat_ratio"],
                    "capacity_headroom_min_ratio": metrics["capacity_headroom_min_ratio"],
                    **rel_row,
                }
            )
            timeseries_map[public_name] = timeseries
            write_csv(debug_dir / f"{window['window_label']}_{window['split_role']}_{public_name}_timeseries.csv", timeseries)
            write_csv(debug_dir / f"{window['window_label']}_{window['split_role']}_{public_name}_decision_records.csv", list(provider.records))
            write_csv(
                candidate_scores_dir / f"{window['window_label']}_{window['split_role']}_{public_name}_candidate_scores.csv",
                list(provider.evaluator.records),
            )

        if window["split_role"] == "holdout":
            plot_window_compare(
                window_label=window["window_label"],
                split_role=window["split_role"],
                series_map=timeseries_map,
                output_path=figures_dir / f"{window['window_label']}_{window['split_role']}_compare.png",
            )

    for label in ("high_event", "combined_variability"):
        for config_name in ("v6_balanced", "v6_conservative"):
            subset = [r for r in comparison_rows if r["window_label"] == label and r["config_name"] == config_name]
            grouped_rows.append(
                {
                    "window_label": label,
                    "config_name": config_name,
                    "delta_pitch_rms_pct": float(np.mean([r["delta_pitch_rms_pct"] for r in subset])),
                    "delta_roll_rms_pct": float(np.mean([r["delta_roll_rms_pct"] for r in subset])),
                    "delta_pitch_peak_pct": float(np.mean([r["delta_pitch_peak_pct"] for r in subset])),
                    "delta_roll_peak_pct": float(np.mean([r["delta_roll_peak_pct"] for r in subset])),
                    "delta_pose_exceed_time_pct": float(np.mean([r["delta_pose_exceed_time_pct"] for r in subset])),
                    "delta_pump_work_pct": float(np.mean([r["delta_pump_work_pct"] for r in subset])),
                    "delta_switch_per_min_pct": float(np.mean([r["delta_switch_per_min_pct"] for r in subset])),
                    "delta_direction_switch_count_pct": float(np.mean([r["delta_direction_switch_count_pct"] for r in subset])),
                    "delta_sat_ratio_pct": float(np.mean([r["delta_sat_ratio_pct"] for r in subset])),
                    "delta_capacity_headroom_pct": float(np.mean([r["delta_capacity_headroom_pct"] for r in subset])),
                }
            )

    grouped_map = {(r["window_label"], r["config_name"]): r for r in grouped_rows}
    for label in ("high_event", "combined_variability"):
        b = grouped_map[(label, "v6_balanced")]
        c = grouped_map[(label, "v6_conservative")]
        overall_rows.append(
            {
                "window_label": label,
                "balanced_minus_conservative_pitch_rms_pct": float(b["delta_pitch_rms_pct"] - c["delta_pitch_rms_pct"]),
                "balanced_minus_conservative_roll_rms_pct": float(b["delta_roll_rms_pct"] - c["delta_roll_rms_pct"]),
                "balanced_minus_conservative_pose_exceed_time_pct": float(b["delta_pose_exceed_time_pct"] - c["delta_pose_exceed_time_pct"]),
                "balanced_minus_conservative_pump_work_pct": float(b["delta_pump_work_pct"] - c["delta_pump_work_pct"]),
                "balanced_minus_conservative_switch_pct": float(b["delta_switch_per_min_pct"] - c["delta_switch_per_min_pct"]),
                "balanced_minus_conservative_sat_pct": float(b["delta_sat_ratio_pct"] - c["delta_sat_ratio_pct"]),
            }
        )

    write_csv(paper_dir / "summary.csv", comparison_rows)
    write_csv(tables_dir / "core_metrics_vs_closed.csv", comparison_rows)
    write_csv(tables_dir / "grouped_change_rates.csv", grouped_rows)
    write_csv(tables_dir / "balanced_vs_conservative.csv", overall_rows)

    actual_wall_seconds = time.perf_counter() - t0
    (out_dir / "run_summary.md").write_text(
        build_run_summary(
            conservative_name=conservative_name,
            actual_wall_seconds=actual_wall_seconds,
            grouped_rows=grouped_rows,
            overall_rows=overall_rows,
        ),
        encoding="utf-8",
    )
    print(f"Saved outputs under {out_dir}")


if __name__ == "__main__":
    main()
