#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from datetime import datetime
from pathlib import Path


os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib-cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run prediction-driven slow-layer decision on FINO1 replay against the closed-only baseline."
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
    parser.add_argument("--start-mode", choices=["first_sample", "first_positive"], default="first_positive")
    parser.add_argument("--event-name", default="ballast_attention_event")
    parser.add_argument("--replay-rows", type=int, default=24)
    parser.add_argument("--dt", type=float, default=1.0)
    parser.add_argument("--case-prefix", default="prediction_decision")
    parser.add_argument("--with-oracle-decision", action="store_true")
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict]) -> None:
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


def _safe_float(row: dict, key: str) -> float:
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


def estimate_baseline_scales(
    timeseries_rows: list[dict],
    deadband_pitch_deg: float,
    deadband_roll_deg: float,
):
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


def make_table_rows(mapping: dict[str, dict]) -> list[dict]:
    rows: list[dict] = []
    for name, blob in mapping.items():
        row = {"name": name}
        for key, value in blob.items():
            row[key] = value
        rows.append(row)
    return rows


def plot_compare(timeseries_map: dict[str, list[dict]], output_path: Path) -> None:
    labels = list(timeseries_map.keys())
    if not labels:
        return
    fig, axes = plt.subplots(5, 1, figsize=(11, 12), sharex=True)
    for label in labels:
        rows = timeseries_map[label]
        t_min = np.array([_safe_float(r, "t_s") / 60.0 for r in rows], dtype=float)
        axes[0].plot(t_min, [_safe_float(r, "wind_speed") for r in rows], label=label)
        axes[1].plot(t_min, [_safe_float(r, "pitch_deg") for r in rows], label=label)
        axes[2].plot(t_min, [_safe_float(r, "roll_deg") for r in rows], label=label)
        axes[3].plot(t_min, [_safe_float(r, "pump_total_rate_m3_min") for r in rows], label=label)
        axes[4].plot(t_min, [_safe_float(r, "pitch_sp_raw_deg") for r in rows], label=label)

    axes[0].set_ylabel("Wind (m/s)")
    axes[1].set_ylabel("Pitch (deg)")
    axes[2].set_ylabel("Roll (deg)")
    axes[3].set_ylabel("Pump rate")
    axes[4].set_ylabel("Pitch SP")
    axes[4].set_xlabel("Time (min)")
    for ax in axes:
        ax.grid(True, alpha=0.3)
    axes[0].legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def build_run_summary(
    summary_rows: list[dict],
    start_timestamp: datetime,
    replay_rows: int,
    dt: float,
    used_oracle: bool,
    estimated_wall_minutes: float,
    actual_wall_seconds: float,
    baseline_scales: dict[str, float],
    candidate_ratios: dict[str, float],
    weight_profiles: dict[str, dict],
) -> str:
    lines = [
        "# Run Summary",
        "",
        f"- replay start timestamp: `{start_timestamp.strftime('%Y-%m-%d %H:%M:%S')}`",
        f"- replay rows: `{int(replay_rows)}` x 10 min",
        f"- sim dt: `{float(dt):.3f} s`",
        f"- pre-run wall-time estimate: `~{estimated_wall_minutes:.1f} min`",
        f"- actual wall time: `{actual_wall_seconds:.1f} s`",
        f"- oracle decision included: `{int(bool(used_oracle))}`",
        "",
        "## Baseline Scales",
        "",
    ]
    for key, value in baseline_scales.items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(["", "## Candidate Ratios", ""])
    for key, value in candidate_ratios.items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(["", "## Weight Profiles", ""])
    for profile_name, profile_blob in weight_profiles.items():
        lines.append(f"### {profile_name}")
        for key, value in profile_blob.items():
            lines.append(f"- {key}: `{value}`")
        lines.append("")
    lines.extend(["## Outcomes", ""])
    for row in summary_rows:
        label = str(row.get("comparison_label", "unknown"))
        lines.extend(
            [
                f"### {label}",
                f"- pitch_rms_deg: `{row.get('pitch_rms_deg')}`",
                f"- roll_rms_deg: `{row.get('roll_rms_deg')}`",
                f"- pitch_abs_max_deg: `{row.get('pitch_abs_max_deg')}`",
                f"- roll_abs_max_deg: `{row.get('roll_abs_max_deg')}`",
                f"- switch_per_min: `{row.get('switch_per_min')}`",
                f"- sat_ratio: `{row.get('sat_ratio')}`",
                "",
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
        DEFAULT_WEIGHT_PROFILES,
        Fino1ReplayDataset,
        ForecastModelAdapter,
        ModelDecisionProvider,
        OracleDecisionProvider,
    )

    t0 = time.perf_counter()
    replay = Fino1ReplayDataset(dataset_dir=args.dataset_dir, split=args.split)
    if args.start_mode == "first_positive":
        start_timestamp = replay.first_positive_sample_timestamp(event_name=args.event_name)
    else:
        start_timestamp = replay.first_sample_timestamp()

    wind_trace = replay.build_wind_trace(
        start_timestamp=start_timestamp,
        row_count=args.replay_rows,
        dt_s=args.dt,
    )
    n_steps = int(wind_trace["n_steps"])
    run_count = 1 + 3 + int(bool(args.with_oracle_decision))
    estimated_wall_minutes = max(1.0, (run_count * n_steps) / 18000.0)

    adapter = ForecastModelAdapter(model_dir=args.model_dir, dataset_dir=args.dataset_dir, device="cpu")

    excel_path = discover_stiffness_file()
    if not excel_path:
        archive_candidate = (
            repo_root / "archive" / "legacy_fowt_control" / "data" / "副本水平刚度曲线.xlsx"
        )
        if archive_candidate.exists():
            excel_path = str(archive_candidate)
    if not excel_path:
        raise SystemExit("No stiffness file found under data/.")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = repo_root / "results" / f"{args.case_prefix}_{args.split}_{ts}"
    paper_dir = out_dir / "paper_ready"
    tables_dir = paper_dir / "tables"
    debug_dir = out_dir / "debug"
    candidate_scores_dir = debug_dir / "candidate_scores"
    paper_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    debug_dir.mkdir(parents=True, exist_ok=True)
    candidate_scores_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict] = []
    timeseries_map: dict[str, list[dict]] = {}

    closed_row, closed_timeseries = run_closed_loop_case(
        excel_path=excel_path,
        case_name=f"{args.case_prefix}_closed_only",
        dt=args.dt,
        n_steps=n_steps,
        wind_trace=wind_trace,
        trim_cfg=None,
        control_enabled=True,
        record_timeseries=True,
        preview_trim_provider=None,
    )
    closed_row["comparison_label"] = "closed_only"
    closed_row["replay_start_timestamp"] = start_timestamp.strftime("%Y-%m-%d %H:%M:%S")
    closed_row["replay_rows"] = int(args.replay_rows)
    closed_row["sim_dt_s"] = float(args.dt)
    summary_rows.append(closed_row)
    timeseries_map["closed_only"] = closed_timeseries
    write_csv(debug_dir / "closed_only_timeseries.csv", closed_timeseries)

    deadband_pitch_deg = float(DEFAULT_CONTROLLER_CFG["deadband_pitch"])
    deadband_roll_deg = float(DEFAULT_CONTROLLER_CFG["deadband_roll"])
    baseline_scales = estimate_baseline_scales(
        timeseries_rows=closed_timeseries,
        deadband_pitch_deg=deadband_pitch_deg,
        deadband_roll_deg=deadband_roll_deg,
    )

    weight_profiles = {
        "balanced": DEFAULT_WEIGHT_PROFILES["balanced"],
        "attitude_priority": DEFAULT_WEIGHT_PROFILES["attitude_priority"],
        "pump_priority": DEFAULT_WEIGHT_PROFILES["pump_priority"],
    }
    candidate_ratios: dict[str, float] | None = None

    for profile_name, profile in weight_profiles.items():
        evaluator = CandidatePlanEvaluator(
            baseline_scales=baseline_scales,
            weight_profile=profile,
        )
        provider = ModelDecisionProvider(
            replay_dataset=replay,
            forecast_adapter=adapter,
            start_timestamp=start_timestamp,
            evaluator=evaluator,
        )
        label = f"prediction_decision_{profile_name}"
        row, timeseries = run_closed_loop_case(
            excel_path=excel_path,
            case_name=f"{args.case_prefix}_{label}",
            dt=args.dt,
            n_steps=n_steps,
            wind_trace=wind_trace,
            trim_cfg=None,
            control_enabled=True,
            record_timeseries=True,
            preview_trim_provider=provider,
        )
        row["comparison_label"] = label
        row["weight_profile"] = profile.name
        row["replay_start_timestamp"] = start_timestamp.strftime("%Y-%m-%d %H:%M:%S")
        row["replay_rows"] = int(args.replay_rows)
        row["sim_dt_s"] = float(args.dt)
        summary_rows.append(row)
        timeseries_map[label] = timeseries
        write_csv(debug_dir / f"{label}_timeseries.csv", timeseries)
        write_csv(debug_dir / f"{label}_decision_records.csv", list(provider.records))
        write_csv(candidate_scores_dir / f"{label}_candidate_scores.csv", list(provider.evaluator.records))
        if candidate_ratios is None:
            candidate_ratios = dict(evaluator.metadata()["candidate_ratios"])

    if args.with_oracle_decision:
        evaluator = CandidatePlanEvaluator(
            baseline_scales=baseline_scales,
            weight_profile=weight_profiles["balanced"],
        )
        oracle_provider = OracleDecisionProvider(
            replay_dataset=replay,
            start_timestamp=start_timestamp,
            evaluator=evaluator,
        )
        label = "oracle_decision_balanced"
        row, timeseries = run_closed_loop_case(
            excel_path=excel_path,
            case_name=f"{args.case_prefix}_{label}",
            dt=args.dt,
            n_steps=n_steps,
            wind_trace=wind_trace,
            trim_cfg=None,
            control_enabled=True,
            record_timeseries=True,
            preview_trim_provider=oracle_provider,
        )
        row["comparison_label"] = label
        row["weight_profile"] = "balanced"
        row["replay_start_timestamp"] = start_timestamp.strftime("%Y-%m-%d %H:%M:%S")
        row["replay_rows"] = int(args.replay_rows)
        row["sim_dt_s"] = float(args.dt)
        summary_rows.append(row)
        timeseries_map[label] = timeseries
        write_csv(debug_dir / f"{label}_timeseries.csv", timeseries)
        write_csv(debug_dir / f"{label}_decision_records.csv", list(oracle_provider.records))
        write_csv(candidate_scores_dir / f"{label}_candidate_scores.csv", list(oracle_provider.evaluator.records))

    summary_rows.sort(key=lambda row: str(row["comparison_label"]))
    write_csv(paper_dir / "summary.csv", summary_rows)
    write_csv(paper_dir / "summary_table.csv", summary_rows)
    write_csv(tables_dir / "baseline_scales.csv", [baseline_scales.to_dict()])
    write_csv(tables_dir / "weight_profiles.csv", make_table_rows({name: p.to_dict() for name, p in weight_profiles.items()}))
    if candidate_ratios is not None:
        write_csv(tables_dir / "candidate_ratios.csv", [candidate_ratios])
    plot_compare(timeseries_map, paper_dir / "compare_pitch_roll_pump.png")

    actual_wall_seconds = time.perf_counter() - t0
    (out_dir / "run_summary.md").write_text(
        build_run_summary(
            summary_rows=summary_rows,
            start_timestamp=start_timestamp,
            replay_rows=args.replay_rows,
            dt=args.dt,
            used_oracle=bool(args.with_oracle_decision),
            estimated_wall_minutes=estimated_wall_minutes,
            actual_wall_seconds=actual_wall_seconds,
            baseline_scales=baseline_scales.to_dict(),
            candidate_ratios=candidate_ratios or {},
            weight_profiles={name: profile.to_dict() for name, profile in weight_profiles.items()},
        ),
        encoding="utf-8",
    )
    print(f"Saved outputs under {out_dir}")
    for row in summary_rows:
        print(
            f"{row['comparison_label']}: "
            f"pitch_rms_deg={row.get('pitch_rms_deg')}, "
            f"roll_rms_deg={row.get('roll_rms_deg')}, "
            f"switch_per_min={row.get('switch_per_min')}, "
            f"sat_ratio={row.get('sat_ratio')}"
        )


if __name__ == "__main__":
    main()
