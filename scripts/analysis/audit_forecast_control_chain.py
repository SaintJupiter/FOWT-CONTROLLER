#!/usr/bin/env python3
"""Audit how forecast-source differences propagate into the ballast planner.

This script is diagnostic only: it does not run the 1 Hz plant simulation and
does not change controller behavior. For each replay bucket it evaluates the
same planner with oracle, learned, and persistence forecasts, then reports where
the chain collapses to the same action/gate decision.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from functools import cmp_to_key
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root / "src"))

from wind_prediction import Fino1ReplayDataset, ForecastModelAdapter, PersistenceMeanForecastAdapter
from wind_prediction.ballast_planner import (
    PlannerConfig,
    _ALL_SEQUENCES,
    DEFAULT_PLANT_INFO,
    compare_sequences,
    compute_pressure_blocks,
    evaluate_sequence,
    select_best_economic,
)


TIMESTAMP_FMT = "%Y-%m-%d %H:%M:%S"
DEFAULT_CASES = [
    ("01_onset_strong", "2024-11-27 19:40:00", "Strong onset"),
    ("04_decay_strong", "2024-09-27 13:00:00", "Strong decay"),
    ("06_signflip_high", "2023-10-03 06:30:00", "High-pressure sign-flip"),
    ("09_high_pressure_event", "2022-02-04 11:00:00", "High-pressure event"),
]

EVENT_RISK_KEYS = (
    "attention_event_0_20m",
    "attention_event_20_40m",
    "attention_event_40_60m",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-dir",
        default="data/processed/wind_ml_10min/ballast_decision_fino1_meteo_aux_v1",
    )
    parser.add_argument(
        "--model-dir",
        default="outputs/wind_prediction/lstm_segmented_fino1_meteo_aux_v1",
    )
    parser.add_argument(
        "--out-dir",
        default="outputs/wind_prediction/forecast_control_chain_audit_4case_1h",
    )
    parser.add_argument("--duration-s", type=float, default=3600.0)
    parser.add_argument(
        "--case-spec",
        action="append",
        default=[],
        help="Case override as case_id|YYYY-MM-DD HH:MM:SS|label. May be repeated.",
    )
    parser.add_argument(
        "--case-csv",
        default="",
        help="Optional CSV with case_id,timestamp,label columns. Used after --case-spec entries.",
    )
    parser.add_argument(
        "--case-ids",
        default="",
        help="Optional comma-separated case ids to keep from --case-csv.",
    )
    parser.add_argument("--persistence-history-min", type=float, default=10.0)
    parser.add_argument("--relief-margin-norm", type=float, default=0.25)
    parser.add_argument("--low-risk-norm", type=float, default=0.75)
    parser.add_argument(
        "--planner-envelope-mode",
        choices=("discounted", "raw"),
        default="discounted",
        help="Whether planner envelope scoring uses discounted or raw pressure blocks.",
    )
    parser.add_argument("--planner-envelope-barrier", action="store_true")
    parser.add_argument("--planner-envelope-barrier-const", type=float, default=50.0)
    parser.add_argument("--relief-medium-cap", action="store_true")
    parser.add_argument("--relief-medium-cap-event-threshold", type=float, default=0.70)
    return parser.parse_args()


def _resolve(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else repo_root / p


def _load_cases(args: argparse.Namespace) -> list[tuple[str, str, str]]:
    cases: list[tuple[str, str, str]] = []
    for spec in args.case_spec:
        parts = str(spec).split("|", 2)
        if len(parts) != 3:
            raise ValueError(f"--case-spec must be case_id|timestamp|label, got: {spec!r}")
        cases.append((parts[0].strip(), parts[1].strip(), parts[2].strip()))
    if args.case_csv:
        case_csv = _resolve(str(args.case_csv))
        df = pd.read_csv(case_csv)
        required = {"case_id", "timestamp", "label"}
        missing = required.difference(df.columns)
        if missing:
            raise ValueError(f"{case_csv} is missing columns: {sorted(missing)}")
        keep_ids = {x.strip() for x in str(args.case_ids).split(",") if x.strip()}
        if keep_ids:
            df = df[df["case_id"].astype(str).isin(keep_ids)].copy()
        for _, row in df.iterrows():
            cases.append(
                (
                    str(row["case_id"]),
                    str(row["timestamp"]),
                    str(row["label"]),
                )
            )
    return cases or list(DEFAULT_CASES)


def _planner_config(args: argparse.Namespace) -> tuple[PlannerConfig, list[float]]:
    base_planner_out = repo_root / "outputs" / "wind_prediction" / "planner_a1_dryrun"
    sign_cfg = json.loads((base_planner_out / "diagnostics" / "a01_pressure_vec_sign_convention.json").read_text())
    discount_cfg = json.loads((base_planner_out / "diagnostics" / "a1_block_discount_config.json").read_text())
    cfg = PlannerConfig(
        pressure_sign_multiplier=float(sign_cfg["planner_sign_multiplier"]),
        envelope_use_discount=str(args.planner_envelope_mode) != "raw",
        envelope_barrier_active=bool(args.planner_envelope_barrier),
        envelope_barrier_const=float(args.planner_envelope_barrier_const),
    )
    return cfg, list(discount_cfg["default_discount_blocks"])


def _speed_dir(uv: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    speed = np.sqrt(uv[:, 0] ** 2 + uv[:, 1] ** 2)
    direction = (np.rad2deg(np.arctan2(-uv[:, 0], -uv[:, 1])) + 360.0) % 360.0
    return speed.astype(float), direction.astype(float)


def _planner_row(
    case_id: str,
    case_label: str,
    bucket: int,
    sample: Any,
    source: str,
    uv: np.ndarray,
    event_probs: dict[str, float],
    oracle_uv: np.ndarray,
    persist_uv: np.ndarray,
    cfg: PlannerConfig,
    discounts: list[float],
    relief_margin_norm: float,
    low_risk_norm: float,
    relief_medium_cap_enabled: bool,
    relief_medium_cap_event_threshold: float,
) -> dict[str, Any]:
    blocks = compute_pressure_blocks(uv, discounts, cfg)
    evaluated = [evaluate_sequence(seq, blocks, DEFAULT_PLANT_INFO, cfg) for seq in _ALL_SEQUENCES]
    feasible, scalars = select_best_economic(evaluated, cfg)
    if not feasible:
        feasible = sorted(
            [r for r in evaluated if not r["hard_reject_reason"]],
            key=cmp_to_key(lambda a, b: compare_sequences(a, b, cfg)[0]),
        )
        scalars = []
    best = feasible[0] if feasible else {"sequence": ("hold", "hold", "hold"), "costs": {}}
    speed, direction = _speed_dir(uv)
    oracle_speed, oracle_dir = _speed_dir(oracle_uv)
    persist_speed, persist_dir = _speed_dir(persist_uv)
    norms = [float(b.get("pressure_norm", 0.0)) for b in blocks[:3]]
    while len(norms) < 3:
        norms.append(0.0)
    vec0 = np.asarray(blocks[0].get("pressure_vec", np.zeros(2)), dtype=float)
    vec2 = np.asarray(blocks[2].get("pressure_vec", np.zeros(2)), dtype=float)
    dot02 = float(np.dot(vec0, vec2))
    relief_gate = int(norms[2] + float(relief_margin_norm) <= norms[0])
    signflip_gate = int(dot02 < 0.0)
    lowrisk_gate = int(max(norms) <= float(low_risk_norm))
    if relief_gate:
        suppression_reason = "forecast_relief_block2_below_block0"
    elif signflip_gate:
        suppression_reason = "forecast_sign_flip_block0_block2"
    elif lowrisk_gate:
        suppression_reason = "forecast_low_risk_all_blocks"
    else:
        suppression_reason = "forecast_not_safe_for_suppression"
    uv_err_oracle = uv - oracle_uv
    uv_diff_persist = uv - persist_uv
    dir_diff_oracle = (direction - oracle_dir + 180.0) % 360.0 - 180.0
    event_vec = [float(event_probs.get(key, 0.0)) for key in EVENT_RISK_KEYS]
    max_event = max(event_vec) if event_vec else 0.0
    relief_cap_margin = norms[0] - norms[2]
    relief_cap_reason = "disabled"
    relief_cap_eligible = False
    if relief_medium_cap_enabled:
        if str(best["sequence"][0]) != "active_medium":
            relief_cap_reason = "current_action_not_medium"
        elif max_event < relief_medium_cap_event_threshold:
            relief_cap_reason = "event_risk_not_confident"
        elif relief_cap_margin < relief_margin_norm:
            relief_cap_reason = "future_relief_too_small"
        elif norms[0] < 1.0:
            relief_cap_reason = "current_pressure_not_high"
        elif dot02 < 0.0:
            relief_cap_reason = "future_direction_flip"
        else:
            relief_cap_reason = "forecast_relief_caps_medium"
            relief_cap_eligible = True
    return {
        "case_id": case_id,
        "case_label": case_label,
        "bucket": int(bucket),
        "history_end": sample.history_end.strftime(TIMESTAMP_FMT),
        "source": source,
        "first_action": str(best["sequence"][0]),
        "best_sequence": ">".join(str(x) for x in best["sequence"]),
        "best_scalar_cost": float(scalars[0]) if scalars else np.nan,
        "max_envelope_norm": float(best.get("costs", {}).get("max_envelope_norm", np.nan)),
        "pump_work_cost": float(best.get("costs", {}).get("pump_work_cost", np.nan)),
        "envelope_violation_cost": float(best.get("costs", {}).get("envelope_violation_cost", np.nan)),
        "envelope_barrier_triggered": int(
            float(best.get("costs", {}).get("envelope_barrier_triggered", 0.0)) > 0.5
        ),
        "pressure_block0_norm": norms[0],
        "pressure_block1_norm": norms[1],
        "pressure_block2_norm": norms[2],
        "pressure_block02_dot": dot02,
        "event_prob_0_20m": event_vec[0] if len(event_vec) > 0 else 0.0,
        "event_prob_20_40m": event_vec[1] if len(event_vec) > 1 else 0.0,
        "event_prob_40_60m": event_vec[2] if len(event_vec) > 2 else 0.0,
        "event_prob_max": float(max_event),
        "relief_cap_eligible": int(relief_cap_eligible),
        "relief_cap_reason": relief_cap_reason,
        "relief_cap_margin_norm": float(relief_cap_margin),
        "suppression_gate_active": int(bool(relief_gate or signflip_gate or lowrisk_gate)),
        "suppression_reason": suppression_reason,
        "forecast_speed0_ms": float(speed[0]),
        "forecast_speed_mean_ms": float(np.mean(speed)),
        "forecast_speed_max_ms": float(np.max(speed)),
        "forecast_speed_block2_minus_block0_ms": float(np.mean(speed[4:6]) - np.mean(speed[0:2])),
        "forecast_dir0_deg": float(direction[0]),
        "oracle_speed_mean_ms": float(np.mean(oracle_speed)),
        "persistence_speed_mean_ms": float(np.mean(persist_speed)),
        "uv_rmse_vs_oracle_ms": float(np.sqrt(np.mean(uv_err_oracle**2))),
        "uv_mae_vs_oracle_ms": float(np.mean(np.abs(uv_err_oracle))),
        "uv_rmse_vs_persistence_ms": float(np.sqrt(np.mean(uv_diff_persist**2))),
        "dir_mae_vs_oracle_deg": float(np.mean(np.abs(dir_diff_oracle))),
    }


def main() -> None:
    args = parse_args()
    dataset_dir = _resolve(str(args.dataset_dir))
    model_dir = _resolve(str(args.model_dir))
    out_dir = _resolve(str(args.out_dir))
    out_dir.mkdir(parents=True, exist_ok=True)
    replay = Fino1ReplayDataset(dataset_dir=dataset_dir, split="test")
    learned = ForecastModelAdapter(model_dir=model_dir, dataset_dir=dataset_dir, device="cpu")
    persistence = PersistenceMeanForecastAdapter(
        dataset_dir=dataset_dir,
        history_minutes=float(args.persistence_history_min),
    )
    cfg, discounts = _planner_config(args)
    cases = _load_cases(args)

    buckets_per_case = int(np.ceil(float(args.duration_s) / replay.update_interval_s))
    rows: list[dict[str, Any]] = []
    for case_id, ts_str, label in cases:
        start = datetime.strptime(ts_str, TIMESTAMP_FMT)
        for bucket in range(buckets_per_case):
            ts = replay.simulation_timestamp(start, bucket * replay.update_interval_s)
            sample = replay.sample_for_history_end(ts)
            if sample is None:
                raise KeyError(f"missing sample for {case_id} at {ts.strftime(TIMESTAMP_FMT)}")
            oracle_uv = np.asarray(sample.y_uv_raw, dtype=float)
            learned_forecast = learned.predict_window(
                sample.x_window,
                timestamp=sample.history_end.strftime(TIMESTAMP_FMT),
            )
            persist_forecast = persistence.predict_window(
                sample.x_window,
                timestamp=sample.history_end.strftime(TIMESTAMP_FMT),
            )
            learned_uv = np.asarray(learned_forecast.wind_uv_raw, dtype=float)
            persist_uv = np.asarray(persist_forecast.wind_uv_raw, dtype=float)
            oracle_events = {
                name: float(sample.y_event[idx])
                for idx, name in enumerate(replay.event_columns)
            }
            for source, uv, event_probs in (
                ("oracle", oracle_uv, oracle_events),
                ("learned", learned_uv, dict(learned_forecast.event_probs)),
                ("persistence", persist_uv, dict(persist_forecast.event_probs)),
            ):
                rows.append(
                    _planner_row(
                        case_id=case_id,
                        case_label=label,
                        bucket=bucket,
                        sample=sample,
                        source=source,
                        uv=uv,
                        event_probs=event_probs,
                        oracle_uv=oracle_uv,
                        persist_uv=persist_uv,
                        cfg=cfg,
                        discounts=discounts,
                        relief_margin_norm=float(args.relief_margin_norm),
                        low_risk_norm=float(args.low_risk_norm),
                        relief_medium_cap_enabled=bool(args.relief_medium_cap),
                        relief_medium_cap_event_threshold=float(args.relief_medium_cap_event_threshold),
                    )
                )

    detail = pd.DataFrame(rows)
    detail_path = out_dir / "forecast_control_chain_detail.csv"
    detail.to_csv(detail_path, index=False)

    wide = detail.pivot(index=["case_id", "bucket", "history_end"], columns="source")
    summary_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    pairs = [("learned", "persistence"), ("learned", "oracle"), ("persistence", "oracle")]
    for (case_id, bucket, history_end), _ in wide.iterrows():
        rec = {"case_id": case_id, "bucket": int(bucket), "history_end": history_end}
        for a, b in pairs:
            rec[f"{a}_vs_{b}_action_match"] = int(
                wide.loc[(case_id, bucket, history_end), ("first_action", a)]
                == wide.loc[(case_id, bucket, history_end), ("first_action", b)]
            )
            rec[f"{a}_vs_{b}_suppression_match"] = int(
                wide.loc[(case_id, bucket, history_end), ("suppression_reason", a)]
                == wide.loc[(case_id, bucket, history_end), ("suppression_reason", b)]
            )
        rec["oracle_action"] = wide.loc[(case_id, bucket, history_end), ("first_action", "oracle")]
        rec["learned_action"] = wide.loc[(case_id, bucket, history_end), ("first_action", "learned")]
        rec["persistence_action"] = wide.loc[(case_id, bucket, history_end), ("first_action", "persistence")]
        rec["oracle_suppression_reason"] = wide.loc[(case_id, bucket, history_end), ("suppression_reason", "oracle")]
        rec["learned_suppression_reason"] = wide.loc[(case_id, bucket, history_end), ("suppression_reason", "learned")]
        rec["persistence_suppression_reason"] = wide.loc[(case_id, bucket, history_end), ("suppression_reason", "persistence")]
        for source in ("oracle", "learned", "persistence"):
            rec[f"{source}_event_prob_max"] = wide.loc[(case_id, bucket, history_end), ("event_prob_max", source)]
            rec[f"{source}_relief_cap_eligible"] = wide.loc[
                (case_id, bucket, history_end), ("relief_cap_eligible", source)
            ]
            rec[f"{source}_relief_cap_reason"] = wide.loc[
                (case_id, bucket, history_end), ("relief_cap_reason", source)
            ]
            rec[f"{source}_barrier"] = wide.loc[
                (case_id, bucket, history_end), ("envelope_barrier_triggered", source)
            ]
            rec[f"{source}_p0"] = wide.loc[(case_id, bucket, history_end), ("pressure_block0_norm", source)]
            rec[f"{source}_p2"] = wide.loc[(case_id, bucket, history_end), ("pressure_block2_norm", source)]
        rec["learned_uv_rmse_vs_oracle_ms"] = wide.loc[(case_id, bucket, history_end), ("uv_rmse_vs_oracle_ms", "learned")]
        rec["persistence_uv_rmse_vs_oracle_ms"] = wide.loc[(case_id, bucket, history_end), ("uv_rmse_vs_oracle_ms", "persistence")]
        rec["learned_uv_rmse_vs_persistence_ms"] = wide.loc[(case_id, bucket, history_end), ("uv_rmse_vs_persistence_ms", "learned")]
        summary_rows.append(rec)
    bucket_summary = pd.DataFrame(summary_rows)
    bucket_summary_path = out_dir / "forecast_control_chain_bucket_summary.csv"
    bucket_summary.to_csv(bucket_summary_path, index=False)

    for case_id, group in bucket_summary.groupby("case_id"):
        row: dict[str, Any] = {"case_id": case_id, "buckets": int(len(group))}
        for a, b in pairs:
            row[f"{a}_vs_{b}_action_match_ratio"] = float(group[f"{a}_vs_{b}_action_match"].mean())
            row[f"{a}_vs_{b}_suppression_match_ratio"] = float(group[f"{a}_vs_{b}_suppression_match"].mean())
        row["learned_uv_rmse_vs_oracle_ms"] = float(group["learned_uv_rmse_vs_oracle_ms"].mean())
        row["persistence_uv_rmse_vs_oracle_ms"] = float(group["persistence_uv_rmse_vs_oracle_ms"].mean())
        row["learned_uv_rmse_vs_persistence_ms"] = float(group["learned_uv_rmse_vs_persistence_ms"].mean())
        for source in ("oracle", "learned", "persistence"):
            row[f"{source}_barrier_ratio"] = float(group[f"{source}_barrier"].mean())
            row[f"{source}_relief_cap_eligible_ratio"] = float(group[f"{source}_relief_cap_eligible"].mean())
        pair_rows.append(row)
    pair_summary = pd.DataFrame(pair_rows)
    all_row = {"case_id": "ALL", "buckets": int(len(bucket_summary))}
    for a, b in pairs:
        all_row[f"{a}_vs_{b}_action_match_ratio"] = float(
            bucket_summary[f"{a}_vs_{b}_action_match"].mean()
        )
        all_row[f"{a}_vs_{b}_suppression_match_ratio"] = float(
            bucket_summary[f"{a}_vs_{b}_suppression_match"].mean()
        )
    all_row["learned_uv_rmse_vs_oracle_ms"] = float(bucket_summary["learned_uv_rmse_vs_oracle_ms"].mean())
    all_row["persistence_uv_rmse_vs_oracle_ms"] = float(bucket_summary["persistence_uv_rmse_vs_oracle_ms"].mean())
    all_row["learned_uv_rmse_vs_persistence_ms"] = float(bucket_summary["learned_uv_rmse_vs_persistence_ms"].mean())
    for source in ("oracle", "learned", "persistence"):
        all_row[f"{source}_barrier_ratio"] = float(bucket_summary[f"{source}_barrier"].mean())
        all_row[f"{source}_relief_cap_eligible_ratio"] = float(
            bucket_summary[f"{source}_relief_cap_eligible"].mean()
        )
    pair_summary = pd.concat([pair_summary, pd.DataFrame([all_row])], ignore_index=True)
    pair_summary_path = out_dir / "forecast_control_chain_pair_summary.csv"
    pair_summary.to_csv(pair_summary_path, index=False)

    diff_windows = bucket_summary[
        bucket_summary["learned_vs_persistence_action_match"].eq(1)
        & bucket_summary["learned_vs_oracle_action_match"].eq(0)
    ]
    lines = [
        "# Forecast-control chain audit",
        "",
        f"- dataset: `{dataset_dir.relative_to(repo_root) if dataset_dir.is_relative_to(repo_root) else dataset_dir}`",
        f"- model: `{model_dir.relative_to(repo_root) if model_dir.is_relative_to(repo_root) else model_dir}`",
        f"- duration per case: `{float(args.duration_s) / 60.0:.0f} min`",
        f"- buckets: `{len(bucket_summary)}`",
        f"- envelope mode: `{args.planner_envelope_mode}`",
        f"- envelope barrier: `{int(bool(args.planner_envelope_barrier))}`",
        f"- relief medium cap diagnostic: `{int(bool(args.relief_medium_cap))}`",
        "",
        "## Pair Summary",
        "",
        "| case | buckets | L/P action | L/O action | P/O action | L/P gate | L/O gate | L RMSE vs O | P RMSE vs O | L RMSE vs P |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, r in pair_summary.iterrows():
        lines.append(
            f"| {r['case_id']} | {int(r['buckets'])} | "
            f"{r['learned_vs_persistence_action_match_ratio'] * 100:.1f}% | "
            f"{r['learned_vs_oracle_action_match_ratio'] * 100:.1f}% | "
            f"{r['persistence_vs_oracle_action_match_ratio'] * 100:.1f}% | "
            f"{r['learned_vs_persistence_suppression_match_ratio'] * 100:.1f}% | "
            f"{r['learned_vs_oracle_suppression_match_ratio'] * 100:.1f}% | "
            f"{r['learned_uv_rmse_vs_oracle_ms']:.2f} | "
            f"{r['persistence_uv_rmse_vs_oracle_ms']:.2f} | "
            f"{r['learned_uv_rmse_vs_persistence_ms']:.2f} |"
        )
    lines.extend(
        [
            "",
            "## Barrier And Relief-Cap Eligibility",
            "",
            "| case | O barrier | L barrier | P barrier | O cap | L cap | P cap |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for _, r in pair_summary.iterrows():
        lines.append(
            f"| {r['case_id']} | "
            f"{r['oracle_barrier_ratio'] * 100:.1f}% | "
            f"{r['learned_barrier_ratio'] * 100:.1f}% | "
            f"{r['persistence_barrier_ratio'] * 100:.1f}% | "
            f"{r['oracle_relief_cap_eligible_ratio'] * 100:.1f}% | "
            f"{r['learned_relief_cap_eligible_ratio'] * 100:.1f}% | "
            f"{r['persistence_relief_cap_eligible_ratio'] * 100:.1f}% |"
        )
    lines.extend(
        [
            "",
            "## Buckets Where Learned Equals Persistence But Differs From Oracle",
            "",
            "| case | bucket | time | oracle | learned | persistence | oracle gate | learned gate | persistence gate |",
            "|---|---:|---|---|---|---|---|---|---|",
        ]
    )
    for _, r in diff_windows.iterrows():
        lines.append(
            f"| {r['case_id']} | {int(r['bucket'])} | {r['history_end']} | "
            f"{r['oracle_action']} | {r['learned_action']} | {r['persistence_action']} | "
            f"{r['oracle_suppression_reason']} | {r['learned_suppression_reason']} | "
            f"{r['persistence_suppression_reason']} |"
        )
    lines.extend(
        [
            "",
            "## Files",
            "",
            f"- detail: `{detail_path.relative_to(repo_root)}`",
            f"- bucket summary: `{bucket_summary_path.relative_to(repo_root)}`",
            f"- pair summary: `{pair_summary_path.relative_to(repo_root)}`",
        ]
    )
    report_path = out_dir / "forecast_control_chain_audit_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Report: {report_path}")
    print(pair_summary.to_string(index=False))


if __name__ == "__main__":
    main()
