#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib-cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Scenario:
    name: str
    title: str
    segments: list[tuple[int, float, float]]
    notes: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate current ballast controller behavior under 10-minute wind inputs."
    )
    parser.add_argument("--dt", type=float, default=1.0)
    parser.add_argument("--platform-profile", type=str, default="default")
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("data/processed/wind_ml_10min/ballast_decision_fino1_meteo_aux_v1"),
    )
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--event-name", type=str, default="ballast_attention_event")
    parser.add_argument("--real-window-rows", type=int, default=24)
    parser.add_argument("--transition-s", type=float, default=0.0)
    return parser.parse_args()


def _insert_repo_paths(repo_root: Path) -> None:
    sys.path.insert(0, str(repo_root / "src"))
    sys.path.insert(0, str(repo_root / "archive" / "legacy_fowt_control"))


def _discover_stiffness_path(repo_root: Path) -> str:
    from run_validation import discover_stiffness_file

    excel_path = discover_stiffness_file()
    if excel_path:
        return str(excel_path)
    archive_candidate = repo_root / "archive" / "legacy_fowt_control" / "data" / "副本水平刚度曲线.xlsx"
    if archive_candidate.exists():
        return str(archive_candidate)
    raise FileNotFoundError("No stiffness file found under data/ or archive/legacy_fowt_control/data/.")


def _linear_interp_angle_deg(start_deg: float, end_deg: float, frac: np.ndarray) -> np.ndarray:
    start = float(start_deg)
    end = float(end_deg)
    delta = ((end - start + 180.0) % 360.0) - 180.0
    return (start + delta * frac + 360.0) % 360.0


def build_piecewise_wind_trace(
    segments: list[tuple[int, float, float]],
    dt_s: float,
    transition_s: float = 0.0,
) -> tuple[dict, list[float]]:
    from wind_env import wind_speed_to_thrust_n

    if not segments:
        raise ValueError("segments must not be empty")
    dt_s = float(dt_s)
    transition_s = max(0.0, float(transition_s))
    ws_chunks: list[np.ndarray] = []
    wd_chunks: list[np.ndarray] = []
    change_times_s: list[float] = []
    elapsed_s = 0.0

    prev_ws = None
    prev_wd = None
    for idx, (duration_min, ws, wd) in enumerate(segments):
        duration_s = int(round(float(duration_min) * 60.0))
        steps = int(round(duration_s / dt_s))
        if steps <= 0:
            raise ValueError(f"segment {idx} duration too short for dt={dt_s}")

        ws_arr = np.full(steps, float(ws), dtype=float)
        wd_arr = np.full(steps, float(wd), dtype=float)
        if idx > 0 and transition_s > 0.0:
            transition_steps = min(steps, int(round(transition_s / dt_s)))
            if transition_steps > 0:
                frac = np.linspace(0.0, 1.0, transition_steps, endpoint=False, dtype=float)
                ws_arr[:transition_steps] = float(prev_ws) + (float(ws) - float(prev_ws)) * frac
                wd_arr[:transition_steps] = _linear_interp_angle_deg(float(prev_wd), float(wd), frac)

        if idx > 0:
            change_times_s.append(elapsed_s)
        elapsed_s += steps * dt_s
        ws_chunks.append(ws_arr)
        wd_chunks.append(wd_arr)
        prev_ws = float(ws)
        prev_wd = float(wd)

    ws_vals = np.concatenate(ws_chunks)
    wd_vals = np.concatenate(wd_chunks)
    thrust_vals = wind_speed_to_thrust_n(ws_vals)
    return {
        "ws": ws_vals,
        "wd": wd_vals,
        "thrust_n": np.asarray(thrust_vals, dtype=float),
        "seed": 0,
        "n_steps": int(ws_vals.size),
        "dt": float(dt_s),
        "update_interval_s": 600.0,
        "mean_lpf_tau_s": 0.0,
    }, change_times_s


def default_scenarios() -> list[Scenario]:
    return [
        Scenario(
            name="speed_step_8_to_20",
            title="A. Speed Step 8 -> 20 -> 8 m/s",
            segments=[(20, 8.0, 0.0), (40, 20.0, 0.0), (40, 8.0, 0.0)],
            notes="10 min bucket hold; no extra transition smoothing.",
        ),
        Scenario(
            name="speed_ramp_6_to_18",
            title="B. Speed Ramp 6 -> 18 -> 6 m/s",
            segments=[
                (10, 6.0, 0.0),
                (10, 8.0, 0.0),
                (10, 10.0, 0.0),
                (10, 12.0, 0.0),
                (10, 14.0, 0.0),
                (10, 16.0, 0.0),
                (10, 18.0, 0.0),
                (10, 16.0, 0.0),
                (10, 14.0, 0.0),
                (10, 12.0, 0.0),
                (10, 10.0, 0.0),
                (10, 8.0, 0.0),
                (10, 6.0, 0.0),
            ],
            notes="Trend-following check under repeated 10 min mean updates.",
        ),
        Scenario(
            name="direction_turn_0_90",
            title="C. Direction Turn 0 -> 45 -> 90 -> 45 -> 0 deg",
            segments=[(30, 12.0, 0.0), (20, 12.0, 45.0), (20, 12.0, 90.0), (20, 12.0, 45.0), (30, 12.0, 0.0)],
            notes="Pure direction change with rated-like wind speed.",
        ),
        Scenario(
            name="gust_return_10_25",
            title="D. Gust Return 10 -> 22 -> 25 -> 22 -> 10 m/s",
            segments=[(20, 10.0, 0.0), (10, 22.0, 0.0), (10, 25.0, 0.0), (10, 22.0, 0.0), (40, 10.0, 0.0)],
            notes="Strong thrust excursion and recovery.",
        ),
    ]


def _write_csv(path: Path, rows: list[dict]) -> None:
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


def _mode_specs():
    from defaults import TRIM_CFG_TEST_E, clone_cfg

    return [
        ("open", False, None),
        ("closed", True, None),
        ("trim_simple_default", True, clone_cfg(TRIM_CFG_TEST_E)),
    ]


def _prepare_timeseries_frame(rows: list[dict]) -> pd.DataFrame:
    frame = pd.DataFrame(rows).copy()
    if frame.empty:
        return frame
    frame["t_min"] = frame["t_s"] / 60.0
    return frame


def _response_metrics(df: pd.DataFrame, change_times_s: list[float], mode: str) -> dict:
    metrics = {
        "first_change_s": float(change_times_s[0]) if change_times_s else np.nan,
        "first_trim_response_lag_s": np.nan,
        "first_pump_response_lag_s": np.nan,
    }
    if df.empty or not change_times_s:
        return metrics
    change_t = float(change_times_s[0])
    after = df[df["t_s"] >= change_t].copy()
    if after.empty:
        return metrics

    pump_mask = after["pump_total_rate_m3_min"].abs() > 1e-6
    if pump_mask.any():
        metrics["first_pump_response_lag_s"] = float(after.loc[pump_mask, "t_s"].iloc[0] - change_t)

    if mode == "trim_simple_default":
        trim_mask = (
            after["current_pitch_trim_scaled_deg"].abs() + after["current_roll_trim_scaled_deg"].abs()
        ) > 1e-4
        if trim_mask.any():
            metrics["first_trim_response_lag_s"] = float(after.loc[trim_mask, "t_s"].iloc[0] - change_t)
    return metrics


def _plot_scenario(
    scenario: Scenario,
    out_path: Path,
    frames: dict[str, pd.DataFrame],
    change_times_s: list[float],
) -> None:
    colors = {
        "open": "#9e9e9e",
        "closed": "#1f77b4",
        "trim_simple_default": "#d62728",
    }

    fig, axes = plt.subplots(5, 1, figsize=(13, 14), sharex=True)
    fig.suptitle(scenario.title, fontsize=14)

    ref_df = next(iter(frames.values()))
    ax0 = axes[0]
    ax0.plot(ref_df["t_min"], ref_df["wind_speed"], color="black", linewidth=1.8, label="wind speed")
    ax0.set_ylabel("WS [m/s]")
    ax0b = ax0.twinx()
    ax0b.plot(ref_df["t_min"], ref_df["wind_dir_deg"], color="#ff7f0e", linestyle="--", linewidth=1.2, label="wind dir")
    ax0b.set_ylabel("WD [deg]")
    ax0.set_title(scenario.notes, fontsize=10)

    for mode, df in frames.items():
        axes[1].plot(df["t_min"], df["pitch_deg"], label=mode, color=colors[mode], linewidth=1.5)
        axes[2].plot(df["t_min"], df["roll_deg"], label=mode, color=colors[mode], linewidth=1.5)
        axes[3].plot(df["t_min"], df["pump_total_rate_m3_min"], label=mode, color=colors[mode], linewidth=1.3)
        axes[4].plot(df["t_min"], df["trim_scale"], label=f"{mode}: trim_scale", color=colors[mode], linewidth=1.3)
        if mode == "trim_simple_default":
            axes[4].plot(
                df["t_min"],
                df["current_pitch_trim_scaled_deg"],
                color="#9467bd",
                linewidth=1.0,
                alpha=0.9,
                label="trim pitch scaled",
            )

    axes[1].set_ylabel("Pitch [deg]")
    axes[2].set_ylabel("Roll [deg]")
    axes[3].set_ylabel("Pump [m3/min]")
    axes[4].set_ylabel("Trim")
    axes[4].set_xlabel("Time [min]")

    for ax in axes:
        for t_s in change_times_s:
            ax.axvline(float(t_s) / 60.0, color="#bdbdbd", linestyle=":", linewidth=0.9)
        ax.grid(alpha=0.25)

    axes[1].legend(loc="upper right", ncol=3, fontsize=9)
    axes[4].legend(loc="upper right", fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _build_real_fino_scenario(replay, window_rows: int) -> tuple[Scenario, dict, list[float], str]:
    event_idx = replay.event_columns.index("ballast_attention_event")
    sample_ts = [sample.history_end for sample in replay.samples]
    sample_flags = np.array([float(sample.y_event[event_idx]) >= 0.5 for sample in replay.samples], dtype=float)
    window_rows = int(window_rows)
    if sample_flags.size < window_rows:
        raise ValueError("not enough replay samples for requested real-window length")

    best_start = 0
    best_score = -np.inf
    for start in range(0, sample_flags.size - window_rows + 1):
        score = float(np.mean(sample_flags[start : start + window_rows]))
        if score > best_score:
            best_score = score
            best_start = start

    start_timestamp = sample_ts[best_start]
    rows = replay.trace_rows_from(start_timestamp=start_timestamp, row_count=window_rows)
    if len(rows) < window_rows:
        raise ValueError("real FINO1 window truncated unexpectedly")

    ws_vals = np.repeat(np.array([row.wind_speed_ms for row in rows], dtype=float), int(replay.update_interval_s))
    wd_vals = np.repeat(np.array([row.wind_dir_deg for row in rows], dtype=float), int(replay.update_interval_s))

    from wind_env import wind_speed_to_thrust_n

    trace = {
        "ws": ws_vals,
        "wd": wd_vals,
        "thrust_n": np.asarray(wind_speed_to_thrust_n(ws_vals), dtype=float),
        "seed": 0,
        "n_steps": int(ws_vals.size),
        "dt": 1.0,
        "update_interval_s": float(replay.update_interval_s),
        "mean_lpf_tau_s": 0.0,
    }
    change_times_s = [i * replay.update_interval_s for i in range(1, len(rows))]
    scenario = Scenario(
        name="fino1_real_high_variability",
        title="E. FINO1 Real High-Variability 4h Window",
        segments=[],
        notes=f"start={start_timestamp.strftime('%Y-%m-%d %H:%M')}, attention_rate={best_score:.2f}",
    )
    return scenario, trace, change_times_s, start_timestamp.strftime("%Y-%m-%d %H:%M:%S")


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    _insert_repo_paths(repo_root)

    from defaults import clone_cfg
    from run_validation import run_closed_loop_case
    from wind_prediction import Fino1ReplayDataset

    excel_path = _discover_stiffness_path(repo_root)
    results_dir = repo_root / "results" / f"controller_10min_validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    results_dir.mkdir(parents=True, exist_ok=True)

    replay = Fino1ReplayDataset(dataset_dir=args.dataset_dir, split=args.split)
    scenarios = list(default_scenarios())

    summary_rows: list[dict] = []
    scenario_index_rows: list[dict] = []

    for scenario in scenarios:
        trace, change_times_s = build_piecewise_wind_trace(
            segments=scenario.segments,
            dt_s=float(args.dt),
            transition_s=float(args.transition_s),
        )
        frames: dict[str, pd.DataFrame] = {}
        for mode, control_enabled, trim_cfg in _mode_specs():
            row, timeseries = run_closed_loop_case(
                excel_path=excel_path,
                case_name=f"{scenario.name}_{mode}",
                dt=float(args.dt),
                n_steps=int(trace["n_steps"]),
                wind_trace=trace,
                trim_cfg=clone_cfg(trim_cfg) if isinstance(trim_cfg, dict) else None,
                control_enabled=bool(control_enabled),
                record_timeseries=True,
                experiment_protocol="main",
                platform_profile=str(args.platform_profile),
                start_from_heave_equilibrium=True,
            )
            df = _prepare_timeseries_frame(timeseries)
            frames[mode] = df
            df.to_csv(results_dir / f"{scenario.name}_{mode}_timeseries.csv", index=False)
            metrics = _response_metrics(df, change_times_s=change_times_s, mode=mode)
            row.update(
                {
                    "scenario": scenario.name,
                    "scenario_title": scenario.title,
                    "mode": mode,
                    "transition_s": float(args.transition_s),
                    "platform_profile": str(args.platform_profile),
                    "first_change_s": metrics["first_change_s"],
                    "first_trim_response_lag_s": metrics["first_trim_response_lag_s"],
                    "first_pump_response_lag_s": metrics["first_pump_response_lag_s"],
                }
            )
            summary_rows.append(row)

        _plot_scenario(
            scenario=scenario,
            out_path=results_dir / f"{scenario.name}_compare.png",
            frames=frames,
            change_times_s=change_times_s,
        )
        scenario_index_rows.append(
            {
                "scenario": scenario.name,
                "title": scenario.title,
                "notes": scenario.notes,
                "change_times_min": ",".join(f"{t/60.0:.1f}" for t in change_times_s),
            }
        )

    real_scenario, real_trace, real_change_times_s, real_start = _build_real_fino_scenario(
        replay=replay,
        window_rows=int(args.real_window_rows),
    )
    real_frames: dict[str, pd.DataFrame] = {}
    for mode, control_enabled, trim_cfg in _mode_specs():
        row, timeseries = run_closed_loop_case(
            excel_path=excel_path,
            case_name=f"{real_scenario.name}_{mode}",
            dt=1.0,
            n_steps=int(real_trace["n_steps"]),
            wind_trace=real_trace,
            trim_cfg=clone_cfg(trim_cfg) if isinstance(trim_cfg, dict) else None,
            control_enabled=bool(control_enabled),
            record_timeseries=True,
            experiment_protocol="main",
            platform_profile=str(args.platform_profile),
            start_from_heave_equilibrium=True,
        )
        df = _prepare_timeseries_frame(timeseries)
        real_frames[mode] = df
        df.to_csv(results_dir / f"{real_scenario.name}_{mode}_timeseries.csv", index=False)
        metrics = _response_metrics(df, change_times_s=real_change_times_s, mode=mode)
        row.update(
            {
                "scenario": real_scenario.name,
                "scenario_title": real_scenario.title,
                "mode": mode,
                "transition_s": 0.0,
                "platform_profile": str(args.platform_profile),
                "first_change_s": metrics["first_change_s"],
                "first_trim_response_lag_s": metrics["first_trim_response_lag_s"],
                "first_pump_response_lag_s": metrics["first_pump_response_lag_s"],
                "replay_start_timestamp": real_start,
            }
        )
        summary_rows.append(row)

    _plot_scenario(
        scenario=real_scenario,
        out_path=results_dir / f"{real_scenario.name}_compare.png",
        frames=real_frames,
        change_times_s=real_change_times_s,
    )
    scenario_index_rows.append(
        {
            "scenario": real_scenario.name,
            "title": real_scenario.title,
            "notes": real_scenario.notes,
            "change_times_min": ",".join(f"{t/60.0:.1f}" for t in real_change_times_s),
            "replay_start_timestamp": real_start,
        }
    )

    _write_csv(results_dir / "scenario_index.csv", scenario_index_rows)
    _write_csv(results_dir / "summary.csv", summary_rows)
    print(f"Saved controller 10min validation outputs to {results_dir}")


if __name__ == "__main__":
    main()
