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
    parser = argparse.ArgumentParser(description="Run fixed-window v6 vs constraint-first A/B.")
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
    parser.add_argument("--case-prefix", default="constraint_first_ab")
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


def percentile_scale(values: list[float], q: float, floor: float) -> float:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return float(floor)
    return max(float(floor), float(np.nanpercentile(np.abs(arr), q)))


def plot_window_compare(window_label: str, split_role: str, series_map: dict[str, list[dict[str, Any]]], output_path: Path) -> None:
    fig, axes = plt.subplots(4, 1, figsize=(11, 10), sharex=True)
    order = ["closed_only", "v6_balanced", "constraint_first_v1"]
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


def plot_candidate_funnel(rows: list[dict[str, Any]], output_path: Path) -> None:
    labels = ["hard_ok", "attitude_eligible", "selected"]
    strategies = ["v6_balanced", "constraint_first_v1"]
    values = []
    for strategy in strategies:
        sub = [r for r in rows if r["strategy"] == strategy]
        values.append(
            [
                sum(int(r["hard_ok"]) for r in sub),
                sum(int(r["attitude_eligible"]) for r in sub),
                sum(int(r["selected"]) for r in sub),
            ]
        )
    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for idx, strategy in enumerate(strategies):
        ax.bar(x + (idx - 0.5) * width, values[idx], width=width, label=strategy)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Count")
    ax.set_title("Candidate Layer Funnel")
    ax.legend(loc="best")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def build_run_summary(
    actual_wall_seconds: float,
    grouped_rows: list[dict[str, Any]],
    conservative_like: str,
    constraint_hold_ratio: float,
) -> str:
    rows = {(r["window_group"], r["strategy"]): r for r in grouped_rows}
    he_cf = rows[("high_event", "constraint_first_v1")]
    cv_cf = rows[("combined_variability", "constraint_first_v1")]
    he_v6 = rows[("high_event", "v6_balanced")]
    cv_v6 = rows[("combined_variability", "v6_balanced")]

    if (
        float(he_cf["delta_pose_exceed_time_pct"]) < 0.0
        and float(he_cf["delta_pump_work_pct"]) <= 1.0
        and float(cv_cf["delta_pump_work_pct"]) <= float(cv_v6["delta_pump_work_pct"])
        and float(cv_cf["delta_sat_ratio_pct"]) <= float(cv_v6["delta_sat_ratio_pct"])
    ):
        conclusion = "正向效果"
    else:
        conclusion = "边界性效果"

    lines = [
        "# constraint_first_v1 A/B Summary",
        "",
        f"- actual wall time: `{actual_wall_seconds:.1f} s`",
        "- note: `v6 remains the current paper-ready conservative feasible version; constraint_first_v1 is only a minimal A/B check`",
        "- compared methods: `closed-only`, `v6 balanced`, `constraint_first_v1`",
        f"- v6 conservative reference kept as: `{conservative_like}`",
        "",
        "## Main Findings",
        "",
        f"- high_event / v6 balanced: pitch_rms `{he_v6['delta_pitch_rms_pct']:.2f}%`, roll_rms `{he_v6['delta_roll_rms_pct']:.2f}%`, pose_exceed_time `{he_v6['delta_pose_exceed_time_pct']:.2f}%`, pump_work `{he_v6['delta_pump_work_pct']:.2f}%`",
        f"- high_event / constraint_first_v1: pitch_rms `{he_cf['delta_pitch_rms_pct']:.2f}%`, roll_rms `{he_cf['delta_roll_rms_pct']:.2f}%`, pose_exceed_time `{he_cf['delta_pose_exceed_time_pct']:.2f}%`, pump_work `{he_cf['delta_pump_work_pct']:.2f}%`",
        f"- combined_variability / v6 balanced: pose_exceed_time `{cv_v6['delta_pose_exceed_time_pct']:.2f}%`, pump_work `{cv_v6['delta_pump_work_pct']:.2f}%`, sat_ratio `{cv_v6['delta_sat_ratio_pct']:.2f}%`",
        f"- combined_variability / constraint_first_v1: pose_exceed_time `{cv_cf['delta_pose_exceed_time_pct']:.2f}%`, pump_work `{cv_cf['delta_pump_work_pct']:.2f}%`, sat_ratio `{cv_cf['delta_sat_ratio_pct']:.2f}%`",
        f"- constraint_first_v1 hold-selection ratio: `{constraint_hold_ratio:.2%}`",
        "",
        "## Interpretation",
        "",
        f"- current conclusion type: `{conclusion}`",
        "- this round does not tune v6 or modify candidate structure; it only checks whether a constraint-first selector gives a clearer pump-burden trade-off",
        "- if constraint_first_v1 mostly collapses to `hold`, it should be treated as a conservative explanatory variant rather than the paper mainline",
        "- if constraint_first_v1 keeps the high_event pose benefit and reduces combined_variability burden metrics, it can be written as a more conservative auxiliary decision variant",
        "- otherwise keep v6 as the main paper-ready version and describe constraint_first_v1 as an explanatory A/B check",
    ]
    return "\n".join(lines).strip() + "\n"


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root / "src"))
    sys.path.insert(0, str(repo_root / "archive" / "legacy_fowt_control"))

    from defaults import DEFAULT_CONTROLLER_CFG
    from run_validation import discover_stiffness_file, run_closed_loop_case
    from wind_prediction import (
        AttitudeBands,
        BaselineScales,
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

    conservative_name = choose_conservative(args.v6_results_dir)
    weight_cfg_rows = read_csv_rows(args.v6_results_dir / "paper_ready" / "tables" / "weight_configs.csv")
    cfg_by_name = {r["name"]: r for r in weight_cfg_rows}
    balanced_cfg = cfg_by_name["balanced"]

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
    candidate_scores_dir = debug_dir / "candidate_scores" / "constraint_first_v1"
    for p in (paper_dir, tables_dir, figures_dir, debug_dir, candidate_scores_dir):
        p.mkdir(parents=True, exist_ok=True)

    deadband_pitch_deg = float(DEFAULT_CONTROLLER_CFG["deadband_pitch"])
    deadband_roll_deg = float(DEFAULT_CONTROLLER_CFG["deadband_roll"])
    tank_capacity_kg = 1850.0 * 1025.0

    closed_abs_rows: list[dict[str, Any]] = []
    comparison_rows: list[dict[str, Any]] = []
    grouped_rows: list[dict[str, Any]] = []
    candidate_funnel_rows: list[dict[str, Any]] = []

    # First pass: closed-only bands and baselines
    closed_window_cache: dict[str, dict[str, Any]] = {}
    exceed_values: list[float] = []
    for window in windows:
        start_timestamp = datetime.strptime(window["start_timestamp"], "%Y-%m-%d %H:%M:%S")
        wind_trace = replay.build_wind_trace(start_timestamp=start_timestamp, row_count=args.replay_rows, dt_s=args.dt)
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
        closed_window_cache[f"{window['window_label']}__{window['split_role']}"] = {
            "window": window,
            "wind_trace": wind_trace,
            "n_steps": n_steps,
            "metrics": closed_metrics,
            "timeseries": closed_timeseries,
        }
        exceed_values.append(float(closed_metrics["pose_exceed_time_s"]))
        closed_abs_rows.append(
            {
                "strategy": "closed_only",
                "window_id": f"{window['window_label']}__{window['split_role']}",
                "window_group": window["window_label"],
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
        write_csv(debug_dir / f"{window['window_label']}_{window['split_role']}_closed_only_timeseries.csv", closed_timeseries)

    exceed_time_band = percentile_scale(exceed_values, 95.0, 60.0)
    attitude_bands = AttitudeBands(
        pitch_rms_band_deg=deadband_pitch_deg,
        roll_rms_band_deg=deadband_roll_deg,
        exceed_time_band_s=exceed_time_band,
        eps=1e-6,
    )
    write_csv(
        tables_dir / "closed_only_attitude_bands.csv",
        [
            {
                "pitch_rms_band_deg": deadband_pitch_deg,
                "roll_rms_band_deg": deadband_roll_deg,
                "exceed_time_band_s": exceed_time_band,
                "pitch_roll_source": "controller_deadband",
                "exceed_time_source": "closed_only_p95_fixed_windows",
            }
        ],
    )

    for key, bundle in closed_window_cache.items():
        window = bundle["window"]
        start_timestamp = datetime.strptime(window["start_timestamp"], "%Y-%m-%d %H:%M:%S")
        closed_metrics = bundle["metrics"]
        wind_trace = bundle["wind_trace"]
        n_steps = bundle["n_steps"]
        closed_timeseries = bundle["timeseries"]
        baseline_scales = BaselineScales(
            pitch_deg=max(deadband_pitch_deg, 1e-3),
            roll_deg=max(deadband_roll_deg, 1e-3),
            pump_rate_m3_min=percentile_scale([_safe_float(r, "pump_total_rate_m3_min") for r in closed_timeseries], 95.0, 1.0),
            backlog_kg=percentile_scale([_safe_float(r, "pump_total_backlog_kg") for r in closed_timeseries], 95.0, 1.0),
            wind_speed_mps=percentile_scale([_safe_float(r, "wind_speed") for r in closed_timeseries], 95.0, 1.0),
            deadband_pitch_deg=deadband_pitch_deg,
            deadband_roll_deg=deadband_roll_deg,
            saturation_margin_ratio=0.08,
            eps=1e-6,
        )
        timeseries_map = {"closed_only": closed_timeseries}
        for strategy_name, selection_mode in [("v6_balanced", "weighted_cost_v1"), ("constraint_first_v1", "constraint_first_v1")]:
            evaluator = CandidatePlanEvaluator(
                baseline_scales=baseline_scales,
                weight_profile=DEFAULT_WEIGHT_PROFILES[balanced_cfg["weight_profile_name"]],
                structure_profile=DEFAULT_STRUCTURE_PROFILES[balanced_cfg["structure_profile_name"]],
                selection_mode=selection_mode,
                attitude_bands=attitude_bands,
                active_ratio_low=float(balanced_cfg["active_ratio_low"]),
                active_ratio_high=float(balanced_cfg["active_ratio_high"]),
                pump_saving_ratio=float(balanced_cfg["pump_saving_ratio"]),
            )
            provider = ModelDecisionProvider(
                replay_dataset=replay,
                forecast_adapter=adapter,
                start_timestamp=start_timestamp,
                evaluator=evaluator,
            )
            row, timeseries = run_closed_loop_case(
                excel_path=excel_path,
                case_name=f"{args.case_prefix}_{window['window_label']}_{window['split_role']}_{strategy_name}",
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
            comparison_rows.append(
                {
                    "window_label": window["window_label"],
                    "split_role": window["split_role"],
                    "strategy": strategy_name,
                    "pitch_peak": float(metrics["pitch_peak_abs_deg"]),
                    "pitch_rms": float(metrics["pitch_rms_deg"]),
                    "roll_peak": float(metrics["roll_peak_abs_deg"]),
                    "roll_rms": float(metrics["roll_rms_deg"]),
                    "pose_exceed_time": float(metrics["pose_exceed_time_s"]),
                    "pump_work": float(metrics["pump_work_m3"]),
                    "startstop_count": float(metrics["startstop_count"]),
                    "switch_per_min": float(metrics["switch_per_min"]),
                    "sat_ratio": float(metrics["sat_ratio"]),
                    "delta_pitch_peak_pct": 100.0 * relative_change(metrics["pitch_peak_abs_deg"], closed_metrics["pitch_peak_abs_deg"]),
                    "delta_pitch_rms_pct": 100.0 * relative_change(metrics["pitch_rms_deg"], closed_metrics["pitch_rms_deg"]),
                    "delta_roll_peak_pct": 100.0 * relative_change(metrics["roll_peak_abs_deg"], closed_metrics["roll_peak_abs_deg"]),
                    "delta_roll_rms_pct": 100.0 * relative_change(metrics["roll_rms_deg"], closed_metrics["roll_rms_deg"]),
                    "delta_pose_exceed_time_pct": 100.0 * relative_change(metrics["pose_exceed_time_s"], closed_metrics["pose_exceed_time_s"]),
                    "delta_pump_work_pct": 100.0 * relative_change(metrics["pump_work_m3"], closed_metrics["pump_work_m3"]),
                    "delta_startstop_count_pct": 100.0 * relative_change(metrics["startstop_count"], closed_metrics["startstop_count"]),
                    "delta_switch_per_min_pct": 100.0 * relative_change(metrics["switch_per_min"], closed_metrics["switch_per_min"]),
                    "delta_sat_ratio_pct": 100.0 * relative_change(metrics["sat_ratio"], closed_metrics["sat_ratio"]),
                }
            )
            write_csv(debug_dir / f"{window['window_label']}_{window['split_role']}_{strategy_name}_timeseries.csv", timeseries)
            write_csv(debug_dir / f"{window['window_label']}_{window['split_role']}_{strategy_name}_decision_records.csv", list(provider.records))
            if strategy_name == "constraint_first_v1":
                write_csv(
                    candidate_scores_dir / f"{window['window_label']}_{window['split_role']}_{strategy_name}_candidate_scores.csv",
                    list(provider.evaluator.candidate_records),
                )
            for diag in provider.evaluator.candidate_records:
                diag_row = dict(diag)
                diag_row["window_group"] = window["window_label"]
                diag_row["split_role"] = window["split_role"]
                diag_row["strategy"] = strategy_name
                candidate_funnel_rows.append(diag_row)
            timeseries_map[strategy_name] = timeseries

        if window["split_role"] == "holdout":
            plot_window_compare(
                window_label=window["window_label"],
                split_role=window["split_role"],
                series_map=timeseries_map,
                output_path=figures_dir / f"{window['window_label']}_method_compare.png",
            )

    core_rows = []
    for row in comparison_rows:
        core_rows.append(row)
    write_csv(tables_dir / "constraint_first_core_metrics_vs_closed.csv", core_rows)

    for window_group in ("high_event", "combined_variability"):
        for strategy in ("v6_balanced", "constraint_first_v1"):
            subset = [r for r in comparison_rows if r["window_label"] == window_group and r["strategy"] == strategy]
            grouped_rows.append(
                {
                    "window_group": window_group,
                    "strategy": strategy,
                    "delta_pitch_peak_pct": float(np.mean([r["delta_pitch_peak_pct"] for r in subset])),
                    "delta_pitch_rms_pct": float(np.mean([r["delta_pitch_rms_pct"] for r in subset])),
                    "delta_roll_peak_pct": float(np.mean([r["delta_roll_peak_pct"] for r in subset])),
                    "delta_roll_rms_pct": float(np.mean([r["delta_roll_rms_pct"] for r in subset])),
                    "delta_pose_exceed_time_pct": float(np.mean([r["delta_pose_exceed_time_pct"] for r in subset])),
                    "delta_pump_work_pct": float(np.mean([r["delta_pump_work_pct"] for r in subset])),
                    "delta_startstop_count_pct": float(np.mean([r["delta_startstop_count_pct"] for r in subset])),
                    "delta_switch_per_min_pct": float(np.mean([r["delta_switch_per_min_pct"] for r in subset])),
                    "delta_sat_ratio_pct": float(np.mean([r["delta_sat_ratio_pct"] for r in subset])),
                }
            )
    write_csv(tables_dir / "constraint_first_grouped_change_rates.csv", grouped_rows)

    sign_rows = []
    metric_names = ["pitch_peak", "pitch_rms", "roll_peak", "roll_rms", "pose_exceed_time", "pump_work", "startstop_count", "switch_per_min", "sat_ratio"]
    for strategy in ("v6_balanced", "constraint_first_v1"):
        for window_group in ("high_event", "combined_variability"):
            subset = [r for r in comparison_rows if r["window_label"] == window_group and r["strategy"] == strategy]
            for metric in metric_names:
                better = sum(1 for r in subset if float(r[f"delta_{metric}_pct"]) < 0.0)
                worse = sum(1 for r in subset if float(r[f"delta_{metric}_pct"]) > 0.0)
                equal = len(subset) - better - worse
                sign_rows.append(
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
    write_csv(tables_dir / "sign_consistency_summary.csv", sign_rows)

    funnel_summary = []
    for window_group in ("high_event", "combined_variability"):
        for split_role in ("tuning", "holdout"):
            sub = [r for r in candidate_funnel_rows if r["window_group"] == window_group and r["split_role"] == split_role]
            for candidate_name in ("hold", "pump_saving", "active_adjust"):
                cand = [r for r in sub if r["candidate_name"] == candidate_name]
                if not cand:
                    continue
                funnel_summary.append(
                    {
                        "window_group": window_group,
                        "split_role": split_role,
                        "strategy": "constraint_first_v1",
                        "candidate_name": candidate_name,
                        "n_rows": len(cand),
                        "hard_ok_count": sum(int(r["hard_ok"]) for r in cand),
                        "attitude_eligible_count": sum(int(r["attitude_eligible"]) for r in cand),
                        "selected_count": sum(int(r["selected"]) for r in cand),
                        "mean_attitude_risk": float(np.mean([float(r["attitude_risk"]) for r in cand])),
                    }
                )
    write_csv(tables_dir / "candidate_layer_funnel.csv", funnel_summary)
    plot_candidate_funnel(candidate_funnel_rows, figures_dir / "candidate_layer_funnel.png")

    constraint_rows = [r for r in candidate_funnel_rows if r["strategy"] == "constraint_first_v1"]
    selected_rows = [r for r in constraint_rows if int(r["selected"]) == 1]
    hold_selected = sum(1 for r in selected_rows if r["candidate_name"] == "hold")
    constraint_hold_ratio = hold_selected / max(len(selected_rows), 1)

    actual_wall_seconds = time.perf_counter() - t0
    (out_dir / "run_summary.md").write_text(
        build_run_summary(
            actual_wall_seconds=actual_wall_seconds,
            grouped_rows=grouped_rows,
            conservative_like=conservative_name,
            constraint_hold_ratio=constraint_hold_ratio,
        ),
        encoding="utf-8",
    )
    print(f"Saved outputs under {out_dir}")


if __name__ == "__main__":
    main()
