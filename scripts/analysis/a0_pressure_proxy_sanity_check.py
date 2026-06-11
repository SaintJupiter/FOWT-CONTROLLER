#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib-cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


TIMESTAMP_FMT = "%Y-%m-%d %H:%M:%S"
BLOCKS = (
    ("block1_0_20m", 0, 2),
    ("block2_20_40m", 2, 4),
    ("block3_40_60m", 4, 6),
)
RISK_LABELS = ("high_event", "combined_variability", "normal_low_risk")


@dataclass(frozen=True)
class SampleStats:
    timestamp: datetime
    current_ws: float
    current_wd_deg: float
    current_u: float
    current_v: float
    global_event: float
    seg_0_20: float
    seg_20_40: float
    seg_40_60: float
    ws_jump: float
    wd_jump: float
    ws_std: float
    wd_var: float
    combo_score: float
    normal_score: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="A0 sanity check for future_pressure_vec against closed-only block response."
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=None,
        help="Optional dataset directory override. If omitted, auto-discover FINO1 dataset.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/wind_prediction/planner_a0_pressure_proxy_sanity"),
    )
    parser.add_argument("--split", default="test")
    parser.add_argument("--dt", type=float, default=1.0)
    parser.add_argument("--window-count-per-type", type=int, default=5)
    parser.add_argument("--block-minutes", type=int, default=20)
    parser.add_argument("--future-rows", type=int, default=6)
    parser.add_argument("--wind-reference-mps", type=float, default=12.0)
    parser.add_argument("--deadband-pitch-deg", type=float, default=1.0)
    parser.add_argument("--deadband-roll-deg", type=float, default=0.8)
    return parser.parse_args()


def ensure_dirs(base_dir: Path) -> dict[str, Path]:
    paper_dir = base_dir / "paper_ready"
    figures_dir = paper_dir / "figures"
    diagnostics_dir = base_dir / "diagnostics"
    intermediate_dir = base_dir / "intermediate"
    for p in (base_dir, paper_dir, figures_dir, diagnostics_dir, intermediate_dir):
        p.mkdir(parents=True, exist_ok=True)
    return {
        "base": base_dir,
        "paper": paper_dir,
        "figures": figures_dir,
        "diagnostics": diagnostics_dir,
        "intermediate": intermediate_dir,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        pd.DataFrame().to_csv(path, index=False)
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        import csv

        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def wrap_angle_diff_deg(a: float, b: float) -> float:
    return abs((float(b) - float(a) + 180.0) % 360.0 - 180.0)


def mean_direction_deg_from_uv(u_vals: np.ndarray, v_vals: np.ndarray) -> float:
    mean_u = float(np.mean(u_vals))
    mean_v = float(np.mean(v_vals))
    return float((np.rad2deg(np.arctan2(-mean_u, -mean_v)) + 360.0) % 360.0)


def speed_dir_from_uv(uv: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    speed = np.sqrt(uv[:, 0] ** 2 + uv[:, 1] ** 2)
    direction = (np.rad2deg(np.arctan2(-uv[:, 0], -uv[:, 1])) + 360.0) % 360.0
    return speed.astype(float), direction.astype(float)


def discover_dataset_dir(repo_root: Path, override: Path | None) -> tuple[Path, list[str]]:
    scanned: list[str] = []
    if override is not None:
        scanned.append(str(override))
        return override, scanned

    base = repo_root / "data" / "processed" / "wind_ml_10min"
    candidates = [
        base / "ballast_decision_fino1_meteo_aux_v1",
        base / "ballast_decision_fino1_segmented_v1",
        base / "ballast_decision_fino1_vertical_shear_v1",
    ]
    for c in candidates:
        scanned.append(str(c))
        if (c / "metadata.json").exists():
            return c, scanned

    for meta in sorted(base.glob("**/metadata.json")):
        scanned.append(str(meta.parent))
        if "fino1" in meta.parent.name.lower():
            return meta.parent, scanned

    raise FileNotFoundError("No FINO1 dataset directory with metadata.json found.")


def discover_model_dirs(repo_root: Path) -> list[str]:
    dirs: list[str] = []
    base = repo_root / "outputs" / "wind_prediction"
    for cfg in sorted(base.glob("**/lstm_config.json")):
        dirs.append(str(cfg.parent))
    return dirs


def discover_existing_closed_only_sources(repo_root: Path) -> list[str]:
    out: list[str] = []
    for path in sorted((repo_root / "results").glob("**/*closed_only_timeseries.csv")):
        out.append(str(path))
    return out


def pick_nonoverlap(
    rows: list[SampleStats],
    count: int,
    min_gap_s: int,
    existing: list[datetime],
    predicate,
) -> list[SampleStats]:
    picked: list[SampleStats] = []
    for row in rows:
        if not predicate(row):
            continue
        ts = row.timestamp
        if all(abs((ts - prev).total_seconds()) >= min_gap_s for prev in existing):
            picked.append(row)
            existing.append(ts)
        if len(picked) >= count:
            break
    return picked


def build_sample_stats(replay) -> list[SampleStats]:
    event_cols = list(replay.event_columns)
    global_idx = event_cols.index("ballast_attention_event")
    seg0_idx = event_cols.index("attention_event_0_20m")
    seg1_idx = event_cols.index("attention_event_20_40m")
    seg2_idx = event_cols.index("attention_event_40_60m")

    rows: list[SampleStats] = []
    for sample in replay.samples:
        uv = np.asarray(sample.y_uv_raw, dtype=float)
        speed, direction = speed_dir_from_uv(uv)
        ws_jump = float(np.max(np.abs(np.diff(speed)))) if len(speed) > 1 else 0.0
        wd_jump = (
            float(np.max([wrap_angle_diff_deg(direction[i], direction[i + 1]) for i in range(len(direction) - 1)]))
            if len(direction) > 1
            else 0.0
        )
        ws_std = float(np.std(speed))
        wd_var = (
            float(np.mean([wrap_angle_diff_deg(direction[i], direction[i + 1]) for i in range(len(direction) - 1)]))
            if len(direction) > 1
            else 0.0
        )
        global_event = float(sample.y_event[global_idx])
        seg_0_20 = float(sample.y_event[seg0_idx])
        seg_20_40 = float(sample.y_event[seg1_idx])
        seg_40_60 = float(sample.y_event[seg2_idx])
        combo_score = 1.5 * global_event + 0.2 * ws_jump + 0.02 * wd_jump + 0.15 * ws_std + 0.01 * wd_var
        normal_score = combo_score + 0.5 * (seg_0_20 + seg_20_40 + seg_40_60)
        wd_rad = math.radians(float(sample.wind_obs["wd_deg"]))
        rows.append(
            SampleStats(
                timestamp=sample.history_end,
                current_ws=float(sample.wind_obs["ws"]),
                current_wd_deg=float(sample.wind_obs["wd_deg"]),
                current_u=float(-sample.wind_obs["ws"] * math.sin(wd_rad)),
                current_v=float(-sample.wind_obs["ws"] * math.cos(wd_rad)),
                global_event=global_event,
                seg_0_20=seg_0_20,
                seg_20_40=seg_20_40,
                seg_40_60=seg_40_60,
                ws_jump=ws_jump,
                wd_jump=wd_jump,
                ws_std=ws_std,
                wd_var=wd_var,
                combo_score=combo_score,
                normal_score=normal_score,
            )
        )
    return rows


def select_windows(replay, count_per_type: int) -> list[dict[str, Any]]:
    rows = build_sample_stats(replay)
    min_gap_s = int(replay.update_interval_s * 6)
    all_picked: list[datetime] = []

    high_sorted = sorted(rows, key=lambda r: (r.global_event, r.combo_score), reverse=True)
    high_rows = pick_nonoverlap(
        high_sorted,
        count=count_per_type,
        min_gap_s=min_gap_s,
        existing=all_picked,
        predicate=lambda r: r.global_event >= 0.5,
    )

    combo_sorted = sorted(rows, key=lambda r: r.combo_score, reverse=True)
    combo_rows = pick_nonoverlap(
        combo_sorted,
        count=count_per_type,
        min_gap_s=min_gap_s,
        existing=all_picked,
        predicate=lambda r: True,
    )

    normal_sorted = sorted(rows, key=lambda r: (r.normal_score, r.ws_std, r.wd_var))
    normal_rows = pick_nonoverlap(
        normal_sorted,
        count=count_per_type,
        min_gap_s=min_gap_s,
        existing=all_picked,
        predicate=lambda r: max(r.global_event, r.seg_0_20, r.seg_20_40, r.seg_40_60) < 0.5,
    )

    out: list[dict[str, Any]] = []
    for label, picked in (
        ("high_event", high_rows),
        ("combined_variability", combo_rows),
        ("normal_low_risk", normal_rows),
    ):
        for rank, row in enumerate(picked, start=1):
            out.append(
                {
                    "risk_type": label,
                    "prediction_timestamp": row.timestamp.strftime(TIMESTAMP_FMT),
                    "rank_in_type": rank,
                    "event_score": row.global_event,
                    "combo_score": row.combo_score,
                    "ws_jump": row.ws_jump,
                    "wd_jump": row.wd_jump,
                    "ws_std": row.ws_std,
                    "wd_var": row.wd_var,
                }
            )
    return out


def pressure_proxy_vec(ws: float, wd_deg: float, wind_reference_mps: float, deadband_pitch_deg: float, deadband_roll_deg: float) -> np.ndarray:
    magnitude = float(np.clip((max(float(ws), 0.0) / float(max(wind_reference_mps, 1.0))) ** 2, 0.0, 1.5))
    wd_rad = math.radians(float(wd_deg))
    return np.array(
        [
            -float(deadband_pitch_deg) * magnitude * math.cos(wd_rad),
            float(deadband_roll_deg) * magnitude * math.sin(wd_rad),
        ],
        dtype=float,
    )


def compute_pressure_features(sample, args) -> list[dict[str, Any]]:
    uv = np.asarray(sample.y_uv_raw, dtype=float)
    speed, direction = speed_dir_from_uv(uv)
    rows: list[dict[str, Any]] = []
    for block_name, start, end in BLOCKS:
        uv_block = uv[start:end]
        sp_block = speed[start:end]
        dir_block = direction[start:end]
        mean_u = float(np.mean(uv_block[:, 0]))
        mean_v = float(np.mean(uv_block[:, 1]))
        mean_speed = float(np.mean(sp_block))
        mean_dir = float((np.rad2deg(np.arctan2(-mean_u, -mean_v)) + 360.0) % 360.0)
        step_vecs = np.vstack(
            [
                pressure_proxy_vec(ws=float(sp), wd_deg=float(wd), wind_reference_mps=args.wind_reference_mps, deadband_pitch_deg=args.deadband_pitch_deg, deadband_roll_deg=args.deadband_roll_deg)
                for sp, wd in zip(sp_block, dir_block)
            ]
        )
        mean_vec = np.mean(step_vecs, axis=0)
        step_mags = np.linalg.norm(step_vecs, axis=1)
        rows.append(
            {
                "block_name": block_name,
                "pressure_pitch_component": float(mean_vec[0]),
                "pressure_roll_component": float(mean_vec[1]),
                "pressure_mag_mean": float(np.mean(step_mags)),
                "pressure_mag_max": float(np.max(step_mags)),
                "pressure_mag_vecnorm": float(np.linalg.norm(mean_vec)),
                "wind_speed_block_mean": mean_speed,
                "wind_speed_sq_block_mean": float(np.mean(np.square(sp_block))),
                "wind_speed_change_abs": float(abs(mean_speed - float(sample.wind_obs["ws"]))),
                "wind_dir_change_abs_deg": float(wrap_angle_diff_deg(float(sample.wind_obs["wd_deg"]), mean_dir)),
                "wind_vector_change_mag": float(np.linalg.norm(np.array([mean_u, mean_v]) - np.array([sample.wind_obs["ws"] * 0.0 + (-sample.wind_obs["ws"] * math.sin(math.radians(float(sample.wind_obs["wd_deg"])))) , -sample.wind_obs["ws"] * math.cos(math.radians(float(sample.wind_obs["wd_deg"])))], dtype=float))),
                "block_mean_speed": mean_speed,
                "block_mean_dir_deg": mean_dir,
                "current_ws": float(sample.wind_obs["ws"]),
                "current_wd_deg": float(sample.wind_obs["wd_deg"]),
            }
        )
    return rows


def find_sample_by_timestamp(replay, timestamp_str: str):
    ts = datetime.strptime(timestamp_str, TIMESTAMP_FMT)
    sample = replay.sample_for_history_end(ts)
    if sample is None:
        raise KeyError(f"missing replay sample for timestamp={timestamp_str}")
    return sample


def block_response_metrics(df: pd.DataFrame, start_s: float, end_s: float, dt: float, deadband_pitch_deg: float, deadband_roll_deg: float) -> dict[str, Any]:
    sub = df[(df["t_s"] >= start_s) & (df["t_s"] < end_s)].copy()
    pitch = pd.to_numeric(sub["pitch_deg"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    roll = pd.to_numeric(sub["roll_deg"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    if len(sub) == 0:
        return {
            "matched_closed_only_row_count": 0,
            "pitch_rms": np.nan,
            "roll_rms": np.nan,
            "pitch_peak_abs": np.nan,
            "roll_peak_abs": np.nan,
            "pose_exceed_time": np.nan,
            "pitch_signed_mean": np.nan,
            "roll_signed_mean": np.nan,
        }
    return {
        "matched_closed_only_row_count": int(len(sub)),
        "pitch_rms": float(np.sqrt(np.mean(np.square(pitch)))),
        "roll_rms": float(np.sqrt(np.mean(np.square(roll)))),
        "pitch_peak_abs": float(np.max(np.abs(pitch))),
        "roll_peak_abs": float(np.max(np.abs(roll))),
        "pose_exceed_time": float(np.sum((np.abs(pitch) > deadband_pitch_deg) | (np.abs(roll) > deadband_roll_deg)) * dt),
        "pitch_signed_mean": float(np.mean(pitch)),
        "roll_signed_mean": float(np.mean(roll)),
    }


def safe_corr(x: pd.Series, y: pd.Series, method: str) -> tuple[float | None, float | None, int]:
    mask = np.isfinite(x.to_numpy(dtype=float)) & np.isfinite(y.to_numpy(dtype=float))
    xs = x.to_numpy(dtype=float)[mask]
    ys = y.to_numpy(dtype=float)[mask]
    n = int(xs.size)
    if n < 3 or np.allclose(xs, xs[0]) or np.allclose(ys, ys[0]):
        return None, None, n
    if method == "pearson":
        corr, pval = pearsonr(xs, ys)
    else:
        corr, pval = spearmanr(xs, ys)
    return float(corr), float(pval), n


def summarize_distribution(series: pd.Series) -> dict[str, Any]:
    arr = pd.to_numeric(series, errors="coerce").dropna().to_numpy(dtype=float)
    if arr.size == 0:
        return {"count": 0, "median": np.nan, "p25": np.nan, "p75": np.nan, "p90": np.nan}
    return {
        "count": int(arr.size),
        "median": float(np.nanmedian(arr)),
        "p25": float(np.nanpercentile(arr, 25)),
        "p75": float(np.nanpercentile(arr, 75)),
        "p90": float(np.nanpercentile(arr, 90)),
    }


def plot_pressure_distribution(df: pd.DataFrame, out_path: Path) -> None:
    order = ["high_event", "combined_variability", "normal_low_risk"]
    block_names = [b[0] for b in BLOCKS]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
    for ax, block_name in zip(axes, block_names):
        data = [df[(df["risk_type"] == risk) & (df["block_name"] == block_name)]["pressure_mag_mean"].to_numpy(dtype=float) for risk in order]
        ax.boxplot(data, labels=order, showfliers=False)
        ax.set_title(block_name)
        ax.set_ylabel("pressure_mag_mean")
        ax.tick_params(axis="x", rotation=20)
        ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_scatter(df: pd.DataFrame, y_col: str, out_path: Path) -> None:
    block_names = [b[0] for b in BLOCKS]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
    for ax, block_name in zip(axes, block_names):
        sub = df[df["block_name"] == block_name].copy()
        ax.scatter(sub["pressure_mag_mean"], sub[y_col], alpha=0.8)
        ax.set_title(block_name)
        ax.set_xlabel("pressure_mag_mean")
        ax.set_ylabel(y_col)
        ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_baseline_comparison(df: pd.DataFrame, out_path: Path) -> pd.DataFrame:
    metrics = ["pitch_rms", "roll_rms", "pose_exceed_time"]
    features = ["pressure_mag_mean", "pressure_mag_max", "wind_speed_block_mean", "wind_speed_change_abs", "wind_dir_change_abs_deg"]
    rows: list[dict[str, Any]] = []
    for feature in features:
        vals: list[float] = []
        for block_name, _, _ in BLOCKS:
            sub = df[df["block_name"] == block_name]
            for metric in metrics:
                corr, _, n = safe_corr(sub[feature], sub[metric], "spearman")
                if corr is not None and n >= 3:
                    vals.append(abs(corr))
        rows.append({"feature": feature, "mean_abs_spearman": float(np.mean(vals)) if vals else np.nan})
    plot_df = pd.DataFrame(rows).sort_values("mean_abs_spearman", ascending=False)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(plot_df["feature"], plot_df["mean_abs_spearman"])
    ax.set_ylabel("mean |Spearman|")
    ax.set_title("Pressure proxy vs naive baseline correlation strength")
    ax.tick_params(axis="x", rotation=20)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return plot_df


def plot_extreme_coverage(df: pd.DataFrame, out_path: Path) -> None:
    rows: list[dict[str, Any]] = []
    for block_name, _, _ in BLOCKS:
        sub = df[df["block_name"] == block_name].copy()
        if sub.empty:
            continue
        threshold = float(np.nanpercentile(sub["pose_exceed_time"].to_numpy(dtype=float), 90))
        extreme = sub[sub["pose_exceed_time"] >= threshold]
        median_pressure = float(np.nanmedian(sub["pressure_mag_mean"].to_numpy(dtype=float)))
        rows.append(
            {
                "block_name": block_name,
                "coverage_above_median": float(np.mean(extreme["pressure_mag_mean"].to_numpy(dtype=float) >= median_pressure)) if len(extreme) else np.nan,
            }
        )
    plot_df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(plot_df["block_name"], plot_df["coverage_above_median"])
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("fraction of extreme windows above median pressure")
    ax.set_title("Extreme response coverage")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def classify_a0(
    corr_df: pd.DataFrame,
    risk_df: pd.DataFrame,
    baseline_df: pd.DataFrame,
    extreme_df: pd.DataFrame,
    flip_df: pd.DataFrame | None,
) -> tuple[str, list[str]]:
    notes: list[str] = []
    primary = corr_df[
        (corr_df["feature"].isin(["pressure_mag_mean", "pressure_mag_max"]))
        & (corr_df["response_metric"].isin(["pitch_rms", "roll_rms", "pose_exceed_time"]))
        & (corr_df["corr_type"] == "spearman")
    ].copy()

    block_scores: dict[str, tuple[int, int]] = {}
    for block_name, _, _ in BLOCKS:
        sub = primary[primary["block_name"] == block_name]
        pos = int(np.sum(pd.to_numeric(sub["corr_value"], errors="coerce").fillna(-99.0) > 0.0))
        total = int(len(sub))
        block_scores[block_name] = (pos, total)
        notes.append(f"{block_name}: positive primary correlations {pos}/{total}")

    summary_map = {(r["risk_type"], r["block_name"]): r for r in risk_df.to_dict("records")}
    separation_ok = True
    for block_name, _, _ in BLOCKS:
        hi = float(summary_map[("high_event", block_name)]["median"])
        lo = float(summary_map[("normal_low_risk", block_name)]["median"])
        if not (np.isfinite(hi) and np.isfinite(lo) and hi > lo):
            separation_ok = False
            notes.append(f"{block_name}: high_event median pressure not above normal")

    base_proxy = baseline_df[baseline_df["feature"] == "pressure_mag_mean"]
    if base_proxy.empty:
        proxy_strength = np.nan
        naive_best = np.nan
    else:
        proxy_strength = float(base_proxy["mean_abs_spearman"].iloc[0])
        naive_best = float(
            baseline_df[baseline_df["feature"] != "pressure_mag_mean"]["mean_abs_spearman"].max()
        )
    if np.isfinite(proxy_strength) and np.isfinite(naive_best):
        notes.append(f"pressure_mag_mean mean|rho|={proxy_strength:.3f}, best naive={naive_best:.3f}")

    extreme_pose = extreme_df[
        (extreme_df["response_metric"] == "pose_exceed_time") & (extreme_df["coverage_type"] == "above_overall_median")
    ]
    extreme_ok = bool(len(extreme_pose) > 0 and np.nanmean(pd.to_numeric(extreme_pose["fraction"], errors="coerce")) >= 0.5)
    if not extreme_ok:
        notes.append("extreme response coverage below 0.5 for pose_exceed_time")

    flip_warning = False
    if flip_df is not None and not flip_df.empty:
        overall = flip_df[flip_df["block_name"] == "overall"].copy()
        if not overall.empty:
            best_mode = str(overall.sort_values("median_cosine_similarity", ascending=False)["flip_mode"].iloc[0])
            orig_val = float(overall[overall["flip_mode"] == "none"]["median_cosine_similarity"].iloc[0])
            best_val = float(overall.sort_values("median_cosine_similarity", ascending=False)["median_cosine_similarity"].iloc[0])
            if best_mode != "none" and (best_val - orig_val) > 0.20:
                flip_warning = True
                notes.append(f"coordinate flip warning: {best_mode} improves median cosine by {best_val - orig_val:.3f}")

    b1_good = block_scores["block1_0_20m"][0] >= 4
    b2_good = block_scores["block2_20_40m"][0] >= 4
    b3_good = block_scores["block3_40_60m"][0] >= 3
    proxy_not_weaker = not (np.isfinite(proxy_strength) and np.isfinite(naive_best) and proxy_strength < 0.9 * naive_best)

    if b1_good and b2_good and separation_ok and proxy_not_weaker and extreme_ok and not flip_warning:
        if b3_good:
            return "PASS", notes
        return "WEAK_PASS", notes
    if b1_good and separation_ok:
        return "WEAK_PASS", notes
    return "FAIL", notes


def main() -> None:
    args = parse_args()
    t0 = time.perf_counter()
    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root / "src"))
    sys.path.insert(0, str(repo_root / "archive" / "legacy_fowt_control"))

    from defaults import DEFAULT_CONTROLLER_CFG
    from run_validation import discover_stiffness_file, run_closed_loop_case
    from wind_prediction import Fino1ReplayDataset

    estimated_minutes = "35-60 min"

    dirs = ensure_dirs(args.output_dir)
    dataset_dir, dataset_scan = discover_dataset_dir(repo_root, args.dataset_dir)
    model_dirs = discover_model_dirs(repo_root)
    existing_closed_only = discover_existing_closed_only_sources(repo_root)

    replay = Fino1ReplayDataset(dataset_dir=dataset_dir, split=args.split)
    event_cols = list(replay.event_columns)
    window_rows = select_windows(replay=replay, count_per_type=args.window_count_per_type)
    write_csv(dirs["intermediate"] / "a0_selected_windows.csv", window_rows)

    excel_path = discover_stiffness_file()
    if not excel_path:
        archive_candidate = repo_root / "archive" / "legacy_fowt_control" / "data" / "副本水平刚度曲线.xlsx"
        if archive_candidate.exists():
            excel_path = str(archive_candidate)
    if not excel_path:
        raise SystemExit("No stiffness file found under data/ or archive/legacy_fowt_control/data/.")

    source_summary = {
        "scanned_dataset_candidates": dataset_scan,
        "chosen_dataset_dir": str(dataset_dir),
        "scanned_model_dirs_count": len(model_dirs),
        "scanned_model_dirs": model_dirs[:20],
        "existing_closed_only_timeseries_count": len(existing_closed_only),
        "existing_closed_only_timeseries_examples": existing_closed_only[:20],
        "closed_only_response_source": "run_closed_loop_case executed inside A0 script",
        "pressure_proxy_source": "oracle future y_uv_raw from replay dataset to isolate proxy validity from forecast error",
        "split": args.split,
        "sample_count": len(replay.samples),
        "event_columns": event_cols,
        "history_steps": int(replay.history_steps),
        "future_steps": int(replay.future_steps),
        "future_rows_used_for_closed_only": int(args.future_rows),
        "dt_s": float(args.dt),
    }
    (dirs["diagnostics"] / "available_sources_summary.json").write_text(
        json.dumps(source_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    alignment_rows: list[dict[str, Any]] = []
    block_metric_rows: list[dict[str, Any]] = []

    deadband_pitch_deg = float(DEFAULT_CONTROLLER_CFG["deadband_pitch"])
    deadband_roll_deg = float(DEFAULT_CONTROLLER_CFG["deadband_roll"])

    for window in window_rows:
        sample = find_sample_by_timestamp(replay, window["prediction_timestamp"])
        pressure_rows = compute_pressure_features(sample, args)

        start_timestamp = sample.history_end
        wind_trace = replay.build_wind_trace(
            start_timestamp=start_timestamp,
            row_count=args.future_rows,
            dt_s=args.dt,
        )
        n_steps = int(wind_trace["n_steps"])
        closed_summary, closed_timeseries = run_closed_loop_case(
            excel_path=excel_path,
            case_name=f"a0_closed_only_{window['risk_type']}_{window['rank_in_type']}",
            dt=args.dt,
            n_steps=n_steps,
            wind_trace=wind_trace,
            trim_cfg=None,
            control_enabled=True,
            record_timeseries=True,
            preview_trim_provider=None,
        )
        df_ts = pd.DataFrame(closed_timeseries)

        for block_name, start_idx, end_idx in BLOCKS:
            block_start_s = start_idx * replay.update_interval_s
            block_end_s = end_idx * replay.update_interval_s
            resp = block_response_metrics(
                df=df_ts,
                start_s=block_start_s,
                end_s=block_end_s,
                dt=args.dt,
                deadband_pitch_deg=deadband_pitch_deg,
                deadband_roll_deg=deadband_roll_deg,
            )
            p_row = next(r for r in pressure_rows if r["block_name"] == block_name)
            merged = {
                **window,
                **p_row,
                **resp,
                "prediction_timestamp": window["prediction_timestamp"],
            }
            block_metric_rows.append(merged)
            alignment_rows.append(
                {
                    "risk_type": window["risk_type"],
                    "rank_in_type": window["rank_in_type"],
                    "prediction_timestamp": window["prediction_timestamp"],
                    "block_name": block_name,
                    "block_start_time": (start_timestamp + timedelta(seconds=block_start_s)).strftime(TIMESTAMP_FMT),
                    "block_end_time": (start_timestamp + timedelta(seconds=block_end_s)).strftime(TIMESTAMP_FMT),
                    "matched_closed_only_row_count": resp["matched_closed_only_row_count"],
                    "matched_t_s_min": float(block_start_s),
                    "matched_t_s_max": float(block_end_s),
                    "pitch_rms": resp["pitch_rms"],
                    "roll_rms": resp["roll_rms"],
                    "pose_exceed_time": resp["pose_exceed_time"],
                }
            )

    block_df = pd.DataFrame(block_metric_rows)
    block_df.to_csv(dirs["intermediate"] / "a0_block_metrics.csv", index=False)
    write_csv(dirs["diagnostics"] / "a0_alignment_check_examples.csv", alignment_rows[:45])

    corr_rows: list[dict[str, Any]] = []
    response_metrics = ["pitch_rms", "roll_rms", "pitch_peak_abs", "roll_peak_abs", "pose_exceed_time"]
    features = ["pressure_mag_mean", "pressure_mag_max"]
    for block_name, _, _ in BLOCKS:
        sub = block_df[block_df["block_name"] == block_name].copy()
        for feature in features:
            for metric in response_metrics:
                for corr_type in ("pearson", "spearman"):
                    corr_val, pval, n = safe_corr(sub[feature], sub[metric], corr_type)
                    corr_rows.append(
                        {
                            "block_name": block_name,
                            "feature": feature,
                            "response_metric": metric,
                            "corr_type": corr_type,
                            "corr_value": corr_val,
                            "p_value": pval,
                            "n": n,
                        }
                    )
    corr_df = pd.DataFrame(corr_rows)
    corr_df.to_csv(dirs["diagnostics"] / "a0_correlations_pressure_vs_response.csv", index=False)

    component_rows: list[dict[str, Any]] = []
    for block_name, _, _ in BLOCKS:
        sub = block_df[block_df["block_name"] == block_name].copy()
        specs = [
            ("abs_pressure_pitch_component", np.abs(sub["pressure_pitch_component"]), sub["pitch_rms"], "pitch_rms"),
            ("abs_pressure_roll_component", np.abs(sub["pressure_roll_component"]), sub["roll_rms"], "roll_rms"),
            ("pressure_pitch_component_signed", sub["pressure_pitch_component"], sub["pitch_signed_mean"], "pitch_signed_mean"),
            ("pressure_roll_component_signed", sub["pressure_roll_component"], sub["roll_signed_mean"], "roll_signed_mean"),
        ]
        for feature_name, x, y, metric_name in specs:
            for corr_type in ("pearson", "spearman"):
                corr_val, pval, n = safe_corr(pd.Series(x), pd.Series(y), corr_type)
                component_rows.append(
                    {
                        "block_name": block_name,
                        "feature": feature_name,
                        "response_metric": metric_name,
                        "corr_type": corr_type,
                        "corr_value": corr_val,
                        "p_value": pval,
                        "n": n,
                    }
                )
    component_df = pd.DataFrame(component_rows)
    component_df.to_csv(dirs["diagnostics"] / "a0_component_correlations.csv", index=False)

    risk_rows: list[dict[str, Any]] = []
    for block_name, _, _ in BLOCKS:
        for risk_type in RISK_LABELS:
            sub = block_df[(block_df["block_name"] == block_name) & (block_df["risk_type"] == risk_type)]
            stats = summarize_distribution(sub["pressure_mag_mean"])
            risk_rows.append({"block_name": block_name, "risk_type": risk_type, **stats})
    risk_df = pd.DataFrame(risk_rows)
    risk_df.to_csv(dirs["diagnostics"] / "a0_summary_by_risk_type.csv", index=False)

    baseline_rows: list[dict[str, Any]] = []
    naive_features = [
        "pressure_mag_mean",
        "pressure_mag_max",
        "wind_speed_block_mean",
        "wind_speed_change_abs",
        "wind_dir_change_abs_deg",
    ]
    for block_name, _, _ in BLOCKS:
        sub = block_df[block_df["block_name"] == block_name].copy()
        for feature in naive_features:
            for metric in response_metrics:
                for corr_type in ("pearson", "spearman"):
                    corr_val, pval, n = safe_corr(sub[feature], sub[metric], corr_type)
                    baseline_rows.append(
                        {
                            "block_name": block_name,
                            "feature": feature,
                            "response_metric": metric,
                            "corr_type": corr_type,
                            "corr_value": corr_val,
                            "p_value": pval,
                            "n": n,
                        }
                    )
    baseline_df = pd.DataFrame(baseline_rows)
    baseline_df.to_csv(dirs["diagnostics"] / "a0_naive_baseline_comparison.csv", index=False)

    direction_rows: list[dict[str, Any]] = []
    flip_rows: list[dict[str, Any]] = []
    valid_direction_rows: list[dict[str, Any]] = []
    for _, row in block_df.iterrows():
        pressure_vec = np.array([float(row["pressure_pitch_component"]), float(row["pressure_roll_component"])], dtype=float)
        response_vec = np.array([float(row["pitch_signed_mean"]), float(row["roll_signed_mean"])], dtype=float)
        pressure_norm = float(np.linalg.norm(pressure_vec))
        response_norm = float(np.linalg.norm(response_vec))
        reliable = bool(pressure_norm > 1e-6 and response_norm > 0.05)
        cosine = np.nan
        angle = np.nan
        if reliable:
            cosine = float(np.dot(pressure_vec, response_vec) / (pressure_norm * response_norm))
            cosine = float(np.clip(cosine, -1.0, 1.0))
            angle = float(np.degrees(np.arccos(cosine)))
            valid_direction_rows.append(
                {
                    "block_name": row["block_name"],
                    "pressure_vec": pressure_vec,
                    "response_vec": response_vec,
                }
            )
        direction_rows.append(
            {
                "risk_type": row["risk_type"],
                "block_name": row["block_name"],
                "prediction_timestamp": row["prediction_timestamp"],
                "pressure_pitch_component": float(row["pressure_pitch_component"]),
                "pressure_roll_component": float(row["pressure_roll_component"]),
                "pitch_signed_mean": float(row["pitch_signed_mean"]),
                "roll_signed_mean": float(row["roll_signed_mean"]),
                "pressure_norm": pressure_norm,
                "response_norm": response_norm,
                "reliable_signed_direction": int(reliable),
                "cosine_similarity": cosine,
                "angle_deg": angle,
            }
        )
    direction_df = pd.DataFrame(direction_rows)
    direction_df.to_csv(dirs["diagnostics"] / "a0_direction_consistency.csv", index=False)

    flip_modes = {
        "none": np.array([1.0, 1.0], dtype=float),
        "flip_pitch": np.array([-1.0, 1.0], dtype=float),
        "flip_roll": np.array([1.0, -1.0], dtype=float),
        "flip_both": np.array([-1.0, -1.0], dtype=float),
    }
    for block_name, _, _ in BLOCKS:
        sub = direction_df[(direction_df["block_name"] == block_name) & (direction_df["reliable_signed_direction"] == 1)]
        if sub.empty:
            for mode in flip_modes:
                flip_rows.append({"block_name": block_name, "flip_mode": mode, "n": 0, "median_cosine_similarity": np.nan})
            continue
        p = sub[["pressure_pitch_component", "pressure_roll_component"]].to_numpy(dtype=float)
        r = sub[["pitch_signed_mean", "roll_signed_mean"]].to_numpy(dtype=float)
        for mode, flip in flip_modes.items():
            p2 = p * flip.reshape(1, 2)
            cosines = np.sum(p2 * r, axis=1) / (
                np.linalg.norm(p2, axis=1) * np.linalg.norm(r, axis=1)
            )
            cosines = np.clip(cosines, -1.0, 1.0)
            flip_rows.append(
                {
                    "block_name": block_name,
                    "flip_mode": mode,
                    "n": int(len(cosines)),
                    "median_cosine_similarity": float(np.nanmedian(cosines)),
                }
            )
    overall = direction_df[direction_df["reliable_signed_direction"] == 1]
    if not overall.empty:
        p = overall[["pressure_pitch_component", "pressure_roll_component"]].to_numpy(dtype=float)
        r = overall[["pitch_signed_mean", "roll_signed_mean"]].to_numpy(dtype=float)
        for mode, flip in flip_modes.items():
            p2 = p * flip.reshape(1, 2)
            cosines = np.sum(p2 * r, axis=1) / (
                np.linalg.norm(p2, axis=1) * np.linalg.norm(r, axis=1)
            )
            cosines = np.clip(cosines, -1.0, 1.0)
            flip_rows.append(
                {
                    "block_name": "overall",
                    "flip_mode": mode,
                    "n": int(len(cosines)),
                    "median_cosine_similarity": float(np.nanmedian(cosines)),
                }
            )
    flip_df = pd.DataFrame(flip_rows)
    flip_df.to_csv(dirs["diagnostics"] / "a0_coordinate_flip_check.csv", index=False)

    extreme_rows: list[dict[str, Any]] = []
    for block_name, _, _ in BLOCKS:
        sub = block_df[block_df["block_name"] == block_name].copy()
        if sub.empty:
            continue
        overall_median = float(np.nanmedian(sub["pressure_mag_mean"].to_numpy(dtype=float)))
        overall_p75 = float(np.nanpercentile(sub["pressure_mag_mean"].to_numpy(dtype=float), 75))
        for metric in ["pitch_rms", "roll_rms", "pose_exceed_time"]:
            threshold = float(np.nanpercentile(sub[metric].to_numpy(dtype=float), 90))
            extreme = sub[sub[metric] >= threshold].copy()
            if extreme.empty:
                continue
            for coverage_name, cut in [("above_overall_median", overall_median), ("above_overall_p75", overall_p75)]:
                extreme_rows.append(
                    {
                        "block_name": block_name,
                        "response_metric": metric,
                        "threshold_top10pct": threshold,
                        "extreme_count": int(len(extreme)),
                        "pressure_cut": cut,
                        "coverage_type": coverage_name,
                        "fraction": float(np.mean(extreme["pressure_mag_mean"].to_numpy(dtype=float) >= cut)),
                        "median_pressure_extreme": float(np.nanmedian(extreme["pressure_mag_mean"].to_numpy(dtype=float))),
                        "median_pressure_all": overall_median,
                    }
                )
    extreme_df = pd.DataFrame(extreme_rows)
    extreme_df.to_csv(dirs["diagnostics"] / "a0_extreme_response_coverage.csv", index=False)

    plot_pressure_distribution(block_df, dirs["figures"] / "pressure_distribution_by_risk_type.png")
    plot_scatter(block_df, "pitch_rms", dirs["figures"] / "pressure_vs_pitch_rms_scatter_by_block.png")
    plot_scatter(block_df, "roll_rms", dirs["figures"] / "pressure_vs_roll_rms_scatter_by_block.png")
    baseline_strength_df = plot_baseline_comparison(block_df, dirs["figures"] / "pressure_proxy_vs_naive_correlation_comparison.png")
    plot_extreme_coverage(extreme_df, dirs["figures"] / "extreme_response_window_coverage.png")

    verdict, verdict_notes = classify_a0(
        corr_df=corr_df,
        risk_df=risk_df,
        baseline_df=baseline_strength_df,
        extreme_df=extreme_df,
        flip_df=flip_df,
    )

    actual_seconds = time.perf_counter() - t0
    report_lines = [
        "# A0 Pressure Proxy Sanity Report",
        "",
        f"- estimated total effort before execution: `{estimated_minutes}`",
        f"- actual elapsed wall time: `{actual_seconds:.1f} s`",
        f"- chosen dataset dir: `{dataset_dir}`",
        f"- split: `{args.split}`",
        f"- selected window counts: `{dict(pd.Series([r['risk_type'] for r in window_rows]).value_counts().sort_index())}`",
        f"- total evaluated windows: `{len(window_rows)}`",
        f"- total block samples: `{len(block_df)}`",
        f"- time alignment: `prediction_timestamp -> future 0-20 / 20-40 / 40-60 min blocks`, matched against closed-only rows by absolute block offset from the same prediction timestamp",
        "",
        "## Data Source Notes",
        "",
        "- Existing model output directories were auto-scanned and logged in `available_sources_summary.json`.",
        "- Primary A0 pressure proxy uses oracle future `y_uv_raw` from the FINO1 replay dataset so that proxy validity is not confounded by forecast model error.",
        "- Closed-only response was generated by running the existing baseline controller without preview injection on each selected A0 window.",
        "",
        "## Verdict",
        "",
        f"- A0 classification: `{verdict}`",
    ]
    for note in verdict_notes:
        report_lines.append(f"- {note}")
    report_lines.extend(
        [
            "",
            "## Recommendation",
            "",
            "- `PASS`: can enter A1 dry-run directly.",
            "- `WEAK_PASS`: can enter A1 dry-run, but keep remote-block discount and sensitivity checks enabled.",
            "- `FAIL`: do not implement planner yet; first revisit alignment, coordinate convention, scale, normalization, and pressure proxy definition.",
            "",
            f"- current recommendation based on A0: `{'enter A1 dry-run' if verdict != 'FAIL' else 'do not enter A1 dry-run yet'}`",
        ]
    )
    (dirs["base"] / "A0_pressure_proxy_sanity_report.md").write_text("\n".join(report_lines).strip() + "\n", encoding="utf-8")
    print(f"A0 completed with verdict={verdict}. Outputs saved under {dirs['base']}")


if __name__ == "__main__":
    main()
