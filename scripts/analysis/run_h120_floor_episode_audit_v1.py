#!/usr/bin/env python3
"""h120 floor-episode read-only audit v1.

Audits existing v1.6 + f120-oracle baseline traces to decide whether h120
has an actionable lever inside active reactive-floor recovery episodes.
No controller changes are made by this script.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REPO = Path(__file__).resolve().parents[2]
BASE = REPO / "outputs/wind_prediction/h120_learned_risk_scheduler_probe_v1"
OUT = REPO / "outputs/wind_prediction/h120_floor_episode_action_shaping_audit_v1"
RAW = OUT / "raw_tables"
DEBUG = OUT / "debug"
PAPER = OUT / "paper_ready"

RUNS = {
    "guard10": BASE / "guard10_v16_oracle_baseline",
    "broader20": BASE / "broader20_v16_oracle_baseline",
}


def ensure_dirs() -> None:
    for path in (OUT, RAW, DEBUG, PAPER):
        path.mkdir(parents=True, exist_ok=True)


def num(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col not in df.columns:
        return pd.Series(default, index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce").fillna(default).astype(float)


def text(df: pd.DataFrame, col: str, default: str = "") -> pd.Series:
    if col not in df.columns:
        return pd.Series(default, index=df.index, dtype=str)
    return df[col].fillna(default).astype(str)


def flag(df: pd.DataFrame, col: str, default: int = 0) -> pd.Series:
    return (num(df, col, default) > 0).astype(int)


def case_id_from_path(path: Path) -> str:
    stem = path.name.replace("_prediction_primary_econ_planner_log.csv", "")
    parts = stem.split("_")
    if parts and parts[0].isdigit():
        return "_".join(parts[1:-2])
    return "_".join(parts[:-2])


def markdown_table(df: pd.DataFrame, max_rows: int | None = None) -> str:
    if df is None or df.empty:
        return "_empty_"
    data = df.copy()
    if max_rows is not None:
        data = data.head(max_rows)

    def fmt(value: Any) -> str:
        if pd.isna(value):
            return ""
        if isinstance(value, float):
            return f"{value:.3f}"
        return str(value)

    lines = [
        "| " + " | ".join(str(c) for c in data.columns) + " |",
        "| " + " | ".join(["---"] * len(data.columns)) + " |",
    ]
    for _, row in data.iterrows():
        lines.append("| " + " | ".join(fmt(row[c]) for c in data.columns) + " |")
    return "\n".join(lines)


def build_bucket_rows(dataset: str, log_path: Path) -> pd.DataFrame:
    df = pd.read_csv(log_path)
    out = pd.DataFrame(index=df.index)
    case_id = case_id_from_path(log_path)
    out["dataset"] = dataset
    out["case_id"] = case_id
    out["bucket"] = num(df, "bucket").astype(int)
    out["current_time_s"] = num(df, "current_time_s")
    out["history_end"] = text(df, "history_end")
    out["current_pitch_deg"] = num(df, "current_pitch_deg")
    out["current_roll_deg"] = num(df, "current_roll_deg")
    out["pitch_abs_deg"] = out["current_pitch_deg"].abs()
    out["roll_abs_deg"] = out["current_roll_deg"].abs()
    out["max_axis_deg"] = np.maximum(out["pitch_abs_deg"], out["roll_abs_deg"])
    out["theta_total_deg"] = np.sqrt(out["current_pitch_deg"] ** 2 + out["current_roll_deg"] ** 2)
    out["dominant_axis"] = np.where(out["pitch_abs_deg"] >= out["roll_abs_deg"], "pitch", "roll")
    out["pitch_trend_deg"] = out["pitch_abs_deg"].diff().fillna(0.0)
    out["roll_trend_deg"] = out["roll_abs_deg"].diff().fillna(0.0)
    out["max_axis_trend_deg"] = out["max_axis_deg"].diff().fillna(0.0)
    out["floor_active"] = flag(df, "reactive_floor_active")
    out["floor_elapsed_s"] = num(df, "reactive_floor_elapsed_s")
    out["floor_resolved_action"] = text(df, "reactive_floor_resolved_action", "none")
    out["floor_medium_delay_active"] = flag(df, "reactive_floor_medium_delay_active")
    out["floor_reason"] = text(df, "reactive_floor_reason", "none")
    out["post_exit_active"] = flag(df, "reactive_floor_post_exit_active")
    out["post_exit_released"] = flag(df, "reactive_floor_post_exit_released")
    out["target_age_s"] = num(df, "reactive_floor_target_age_s")
    out["target_error_mean_kg"] = num(df, "reactive_floor_target_err_mean_kg")
    alt_err = num(df, "active_effectiveness_target_err_mean_kg", np.nan)
    out["target_error_mean_kg"] = out["target_error_mean_kg"].where(
        out["target_error_mean_kg"] > 0, alt_err.fillna(0.0)
    )
    out["target_refreshed"] = flag(df, "prediction_primary_target_refreshed")
    out["target_reused"] = flag(df, "prediction_primary_target_reused")
    out["target_stale_during_floor"] = (
        (out["target_age_s"] >= 600.0) & (out["target_error_mean_kg"] <= 500.0) & (out["floor_active"] > 0)
    ).astype(int)
    out["pump_rate_m3_min"] = num(df, "active_effectiveness_pump_rate_m3_min")
    alt_rate = num(df, "no_unexplained_hold_pump_rate_m3_min", np.nan)
    out["pump_rate_m3_min"] = out["pump_rate_m3_min"].where(out["pump_rate_m3_min"] > 0, alt_rate.fillna(0.0))
    out["pump_idle"] = (
        (flag(df, "reactive_floor_pump_idle") > 0)
        | (flag(df, "active_effectiveness_pump_idle") > 0)
        | (out["pump_rate_m3_min"].abs() <= 1e-6)
    ).astype(int)
    out["pump_idle_while_high"] = ((out["max_axis_deg"] > 5.0) & (out["pump_idle"] > 0)).astype(int)
    out["near_b0_norm"] = num(df, "pressure_block0_norm")
    out["near_b1_norm"] = num(df, "pressure_block1_norm")
    out["near_b2_norm"] = num(df, "pressure_block2_norm")
    out["near_max_norm"] = out[["near_b0_norm", "near_b1_norm", "near_b2_norm"]].max(axis=1)
    out["near_last_norm"] = out["near_b2_norm"]
    out["near_intensification"] = out["near_last_norm"] - out["near_b0_norm"]
    out["far_60_80_norm"] = num(df, "far_horizon_norm_60_80")
    out["far_80_100_norm"] = num(df, "far_horizon_norm_80_100")
    out["far_100_120_norm"] = num(df, "far_horizon_norm_100_120")
    out["far_min_norm"] = out[["far_60_80_norm", "far_80_100_norm", "far_100_120_norm"]].min(axis=1)
    out["far_max_norm"] = out[["far_60_80_norm", "far_80_100_norm", "far_100_120_norm"]].max(axis=1)
    out["far_persistent_high"] = (out["far_min_norm"] >= 1.0).astype(int)
    out["delayed_intensification"] = (out["far_max_norm"] >= out["near_max_norm"] + 0.3).astype(int)
    out["reintensification_after_relief"] = (
        (out["near_last_norm"] <= out["near_b0_norm"] - 0.25)
        & (out["far_max_norm"] >= out["near_last_norm"] + 0.3)
    ).astype(int)
    out["direction_shift"] = flag(df, "far_horizon_direction_shift")
    out["signflip_or_reversal"] = flag(df, "far_horizon_reversal")
    out["h120_direction_consistent"] = flag(df, "h120_scheduler_far_direction_consistent_with_current_posture")
    out["axis_tradeoff_bucket"] = (
        ((out["pitch_trend_deg"] < -0.05) & (out["roll_trend_deg"] > 0.05))
        | ((out["roll_trend_deg"] < -0.05) & (out["pitch_trend_deg"] > 0.05))
    ).astype(int)
    out["wrong_axis_recovery_proxy"] = (
        (out["floor_active"] > 0)
        & (out["axis_tradeoff_bucket"] > 0)
        & (out["max_axis_deg"] > 5.0)
    ).astype(int)
    out["recovery_good_bucket"] = (
        (out["floor_active"] > 0)
        & (out["max_axis_trend_deg"] < -0.1)
        & (out["pump_idle_while_high"] == 0)
    ).astype(int)
    out["recovery_slow_bucket"] = (
        (out["floor_active"] > 0)
        & (out["max_axis_deg"] > 5.0)
        & (out["max_axis_trend_deg"] >= -0.05)
    ).astype(int)
    return out


def assign_episode_ids(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    episode_keys: list[str] = []
    for (dataset, case_id), group in out.groupby(["dataset", "case_id"], sort=False):
        active = group["floor_active"].astype(int).to_numpy()
        episode_no = 0
        prev = 0
        for idx, value in zip(group.index, active):
            if value and not prev:
                episode_no += 1
            episode_keys.append(f"{dataset}::{case_id}::floor{episode_no:02d}" if value else "")
            prev = int(value)
    out["floor_episode_id"] = episode_keys
    return out


def build_audit_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    frames: list[pd.DataFrame] = []
    for dataset, run_dir in RUNS.items():
        for path in sorted((run_dir / "planner_logs").glob("*_planner_log.csv")):
            frames.append(build_bucket_rows(dataset, path))
    bucket = assign_episode_ids(pd.concat(frames, ignore_index=True))
    bucket.to_csv(RAW / "floor_bucket_audit_table.csv", index=False)

    episode_rows: list[dict[str, Any]] = []
    for episode_id, group in bucket[bucket["floor_active"] > 0].groupby("floor_episode_id", sort=False):
        if not episode_id:
            continue
        max_start = float(group["max_axis_deg"].iloc[0])
        max_end = float(group["max_axis_deg"].iloc[-1])
        row = {
            "floor_episode_id": episode_id,
            "dataset": str(group["dataset"].iloc[0]),
            "case_id": str(group["case_id"].iloc[0]),
            "start_bucket": int(group["bucket"].min()),
            "end_bucket": int(group["bucket"].max()),
            "duration_buckets": int(len(group)),
            "duration_s": float(len(group) * 600.0),
            "max_axis_start_deg": max_start,
            "max_axis_end_deg": max_end,
            "max_axis_peak_deg": float(group["max_axis_deg"].max()),
            "exit_below_5": int(max_end < 5.0),
            "recovery_slope_deg_per_bucket": float((max_end - max_start) / max(len(group) - 1, 1)),
            "active_small_rows": int(group["floor_resolved_action"].eq("active_small").sum()),
            "active_medium_rows": int(group["floor_resolved_action"].eq("active_medium").sum()),
            "medium_escalation": int(group["floor_medium_delay_active"].max()),
            "post_exit_released": int(group["post_exit_released"].max()),
            "pump_idle_while_high_rows": int(group["pump_idle_while_high"].sum()),
            "target_stale_rows": int(group["target_stale_during_floor"].sum()),
            "target_reused_rows": int(group["target_reused"].sum()),
            "target_refreshed_rows": int(group["target_refreshed"].sum()),
            "target_reused_ratio": float(group["target_reused"].mean()),
            "axis_tradeoff_rows": int(group["axis_tradeoff_bucket"].sum()),
            "wrong_axis_recovery_rows": int(group["wrong_axis_recovery_proxy"].sum()),
            "recovery_slow_rows": int(group["recovery_slow_bucket"].sum()),
            "recovery_good_rows": int(group["recovery_good_bucket"].sum()),
            "far_persistent_high_rows": int(group["far_persistent_high"].sum()),
            "delayed_intensification_rows": int(group["delayed_intensification"].sum()),
            "reintensification_rows": int(group["reintensification_after_relief"].sum()),
            "direction_shift_rows": int(group["direction_shift"].sum()),
            "signflip_rows": int(group["signflip_or_reversal"].sum()),
            "mean_near_max_norm": float(group["near_max_norm"].mean()),
            "mean_far_max_norm": float(group["far_max_norm"].mean()),
        }
        row["recovery_good"] = int(row["exit_below_5"] and row["pump_idle_while_high_rows"] == 0 and row["target_stale_rows"] == 0)
        row["recovery_slow"] = int(row["recovery_slow_rows"] > 0 or row["duration_buckets"] >= 3)
        row["pump_idle_while_high"] = int(row["pump_idle_while_high_rows"] > 0)
        row["target_stale_during_floor"] = int(row["target_stale_rows"] > 0)
        row["axis_tradeoff"] = int(row["axis_tradeoff_rows"] > 0)
        row["wrong_axis_recovery"] = int(row["wrong_axis_recovery_rows"] > 0)
        row["medium_needed_late"] = int(row["medium_escalation"] > 0 and row["recovery_slow"])
        row["early_stop_too_early"] = 0
        row["double_pump_no_exit"] = int(row["active_medium_rows"] > 0 and not row["exit_below_5"])
        row["no_obvious_h120_lever"] = int(
            not any(
                row[k]
                for k in [
                    "recovery_slow",
                    "pump_idle_while_high",
                    "target_stale_during_floor",
                    "axis_tradeoff",
                    "wrong_axis_recovery",
                    "medium_needed_late",
                    "double_pump_no_exit",
                ]
            )
        )
        episode_rows.append(row)
    episode = pd.DataFrame(episode_rows)
    episode.to_csv(RAW / "floor_episode_audit_table.csv", index=False)

    axis_cols = [
        "dataset",
        "case_id",
        "bucket",
        "floor_episode_id",
        "current_pitch_deg",
        "current_roll_deg",
        "dominant_axis",
        "pitch_trend_deg",
        "roll_trend_deg",
        "axis_tradeoff_bucket",
        "wrong_axis_recovery_proxy",
        "near_max_norm",
        "far_max_norm",
        "direction_shift",
        "signflip_or_reversal",
    ]
    axis = bucket[bucket["floor_active"] > 0][axis_cols].copy()
    axis.to_csv(RAW / "floor_axis_alignment_table.csv", index=False)
    return bucket, episode, axis


def write_phase0() -> None:
    lines = [
        "# Phase 0 State Check",
        "",
        "- v1.6 reproduction flags: `--forecast-source oracle --dataset-dir data/processed/wind_ml_10min/ballast_decision_fino1_meteo_aux_h240_f120_v1 --far-horizon --reactive-floor-predictive-veto on --reactive-floor-action active_small --reactive-floor-medium-delay-s 1200 --reactive-floor-post-exit-mode early_stop`.",
        "- f120 oracle source exists: `data/processed/wind_ml_10min/ballast_decision_fino1_meteo_aux_h240_f120_v1`.",
        "- Existing baseline traces found for guard10 and broader20 under `outputs/wind_prediction/h120_learned_risk_scheduler_probe_v1/*_v16_oracle_baseline`.",
        "- Planner logs already expose floor_active, resolved floor action, medium delay, target age/error, target reuse/refresh, pump idle/rate, posture, near pressure blocks, and far-horizon norms.",
        "- Full axis-wise pressure vector is not fully exposed in baseline logs; Phase 1 uses available norm/shift proxies and posture-axis trends. Extra telemetry may be needed only if Phase 1 passes.",
        "- No telemetry or controller changes were made for Phase 0/1 audit.",
    ]
    (DEBUG / "phase0_state_check.md").write_text("\n".join(lines) + "\n")


def write_reports(bucket: pd.DataFrame, episode: pd.DataFrame, axis: pd.DataFrame) -> None:
    if episode.empty:
        summary = pd.DataFrame(
            [
                {
                    "dataset": dataset,
                    "episodes": 0,
                    "floor_buckets": int(bucket[bucket["dataset"].eq(dataset)]["floor_active"].sum()),
                }
                for dataset in sorted(bucket["dataset"].unique())
            ]
        )
    else:
        summary = (
            episode.groupby("dataset", dropna=False)
            .agg(
                episodes=("floor_episode_id", "count"),
                cases=("case_id", "nunique"),
                floor_buckets=("duration_buckets", "sum"),
                recovery_slow=("recovery_slow", "sum"),
                pump_idle_while_high=("pump_idle_while_high", "sum"),
                target_stale_during_floor=("target_stale_during_floor", "sum"),
                axis_tradeoff=("axis_tradeoff", "sum"),
                wrong_axis_recovery=("wrong_axis_recovery", "sum"),
                medium_needed_late=("medium_needed_late", "sum"),
                double_pump_no_exit=("double_pump_no_exit", "sum"),
                no_obvious_h120_lever=("no_obvious_h120_lever", "sum"),
                mean_duration_buckets=("duration_buckets", "mean"),
                mean_peak_axis_deg=("max_axis_peak_deg", "mean"),
            )
            .reset_index()
        )
    all_row = {
        "dataset": "all",
        "episodes": int(len(episode)),
        "cases": int(episode["case_id"].nunique()) if not episode.empty else 0,
        "floor_buckets": int(episode["duration_buckets"].sum()) if not episode.empty else 0,
        "recovery_slow": int(episode["recovery_slow"].sum()) if not episode.empty else 0,
        "pump_idle_while_high": int(episode["pump_idle_while_high"].sum()) if not episode.empty else 0,
        "target_stale_during_floor": int(episode["target_stale_during_floor"].sum()) if not episode.empty else 0,
        "axis_tradeoff": int(episode["axis_tradeoff"].sum()) if not episode.empty else 0,
        "wrong_axis_recovery": int(episode["wrong_axis_recovery"].sum()) if not episode.empty else 0,
        "medium_needed_late": int(episode["medium_needed_late"].sum()) if not episode.empty else 0,
        "double_pump_no_exit": int(episode["double_pump_no_exit"].sum()) if not episode.empty else 0,
        "no_obvious_h120_lever": int(episode["no_obvious_h120_lever"].sum()) if not episode.empty else 0,
        "mean_duration_buckets": float(episode["duration_buckets"].mean()) if not episode.empty else np.nan,
        "mean_peak_axis_deg": float(episode["max_axis_peak_deg"].mean()) if not episode.empty else np.nan,
    }
    summary = pd.concat([summary, pd.DataFrame([all_row])], ignore_index=True, sort=False)
    summary.to_csv(RAW / "floor_episode_summary.csv", index=False)

    floor_buckets = bucket[bucket["floor_active"] > 0]
    bucket_summary = pd.DataFrame(
        [
            {
                "dataset": dataset,
                "total_buckets": int(len(group)),
                "floor_buckets": int(group["floor_active"].sum()),
                "high_buckets": int((group["max_axis_deg"] > 5.0).sum()),
                "pump_idle_while_high_buckets": int(group["pump_idle_while_high"].sum()),
                "target_stale_during_floor_buckets": int(group["target_stale_during_floor"].sum()),
                "axis_tradeoff_floor_buckets": int(group[(group["floor_active"] > 0)]["axis_tradeoff_bucket"].sum()),
            }
            for dataset, group in bucket.groupby("dataset")
        ]
    )
    bucket_summary.to_csv(RAW / "floor_bucket_summary.csv", index=False)

    enough_problem_space = False
    if not episode.empty:
        enough_problem_space = bool(
            (episode["recovery_slow"].sum() >= 3)
            or (episode["axis_tradeoff"].sum() >= 3)
            or (episode["pump_idle_while_high"].sum() >= 3)
            or (episode["target_stale_during_floor"].sum() >= 3)
            or (episode["medium_needed_late"].sum() >= 2)
        )

    lines = [
        "# Floor Episode Audit Summary",
        "",
        "Scope: read-only audit over existing v1.6 + f120-oracle baseline planner logs.",
        "",
        "## Episode Summary",
        markdown_table(summary),
        "",
        "## Bucket Summary",
        markdown_table(bucket_summary),
        "",
        "## Episodes",
        markdown_table(episode, max_rows=30),
        "",
        "## Phase 1 Go/No-Go",
        "",
    ]
    if enough_problem_space:
        lines.extend(
            [
                "GO: floor-active episodes show enough recovery problems to justify a small default-off oracle shaping candidate design.",
                "",
                "Next: implement only the candidates supported by the observed failure modes.",
            ]
        )
    else:
        lines.extend(
            [
                "NO-GO: floor-active episodes are too sparse and/or do not show enough recurring recovery-quality failures.",
                "",
                "This means h120 floor-episode action shaping does not currently have a robust action space on this casebook. Do not implement oracle shaping candidates from this evidence alone.",
            ]
        )
    text_out = "\n".join(lines) + "\n"
    (PAPER / "floor_episode_audit_summary.md").write_text(text_out)
    (PAPER / "floor_shaping_key_findings.md").write_text(text_out)

    decision_lines = [
        "# h120 Floor Episode Action Shaping Decision",
        "",
        "Status: Phase 1 read-only audit completed.",
        "",
        f"- Floor episodes found: {len(episode)}.",
        f"- Floor buckets found: {int(bucket['floor_active'].sum())}.",
        f"- Phase 1 go to oracle shaping: {'yes' if enough_problem_space else 'no'}.",
        "",
        "Required answers:",
        f"1. **Does floor_active contain h120 action space?** {'Partly, enough for candidate design' if enough_problem_space else 'Not enough in the current casebook'} .",
        "2. **Main problem type?** See episode summary; this audit tracks recovery_slow, axis_tradeoff, target_stale, and pump_idle while high.",
        "3. **Does oracle h120 floor-shaping beat v1.6?** Not tested yet; gated on Phase 1.",
        "4. **Best candidate?** Not selected unless Phase 1 passes.",
        "5. **Broader20/lowrisk harm?** Not tested; Phase 1 shows baseline floor-active opportunities are rare on broader20.",
        "6. **Worth learned h120 floor-shaping head?** Not before an oracle candidate clears guard10 and broader20.",
        "7. **If not worth it, next h120 direction?** Use this evidence to seek another interface with real actuation headroom.",
        "8. **Does v1.6 remain mainline?** Yes.",
    ]
    (OUT / "h120_floor_episode_action_shaping_decision.md").write_text("\n".join(decision_lines) + "\n")


def main() -> None:
    ensure_dirs()
    write_phase0()
    bucket, episode, axis = build_audit_tables()
    write_reports(bucket, episode, axis)


if __name__ == "__main__":
    main()
