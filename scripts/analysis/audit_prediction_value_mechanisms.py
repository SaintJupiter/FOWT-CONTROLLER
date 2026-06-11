#!/usr/bin/env python3
"""Map where forecast information can change prediction-primary decisions.

This is a planner-only audit. It does not run the 1 Hz plant simulation and it
does not change controller behavior. For fixed replay buckets it evaluates the
same economic planner under three forecast sources:

  - oracle future
  - learned forecast
  - persistence forecast

The output classifies each bucket into mechanism types such as future-risk
early action, future-relief/signflip suppression, all sources equal, or learned
extra action. The goal is to find where prediction value exists without mixing
in event-risk floors, preview-lead gates, or closed-loop tuning.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
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
    compute_pressure_blocks,
    evaluate_sequence,
    norm_term,
    select_best_economic,
)


TIMESTAMP_FMT = "%Y-%m-%d %H:%M:%S"
EVENT_RISK_KEYS = (
    "attention_event_0_20m",
    "attention_event_20_40m",
    "attention_event_40_60m",
)

CASES = [
    ("01_onset_strong", "2024-11-27 19:40:00", "Strong onset: calm now, very strong future wind"),
    ("02_onset_signflip", "2023-10-31 06:20:00", "Onset with direction flip"),
    ("03_onset_moderate", "2024-10-10 03:50:00", "Moderate onset"),
    ("04_decay_strong", "2024-09-27 13:00:00", "Strong decay: high now, weak future wind"),
    ("05_decay_signflip", "2022-03-20 19:00:00", "Decay with direction flip"),
    ("06_signflip_high", "2023-10-03 06:30:00", "High-pressure sign-flip"),
    ("07_signflip_sustained", "2023-03-14 04:40:00", "Sustained high sign-flip"),
    ("08_lowrisk_quiet", "2021-12-20 14:30:00", "Low-risk quiet window"),
    ("09_high_pressure_event", "2022-02-04 11:00:00", "High-pressure high-event normal window"),
    ("10_residual_high", "2024-09-05 18:10:00", "Residual-high normal window"),
]

ACTION_SEVERITY = {
    "hold": 0,
    "pump_saving": 1,
    "active_small": 2,
    "active_reverse_small": 2,
    "active_medium": 3,
}


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
        default="outputs/wind_prediction/prediction_value_mechanism_audit_10case_1h",
    )
    parser.add_argument("--duration-s", type=float, default=3600.0)
    parser.add_argument("--case-ids", default="", help="Comma-separated ids like 01,04,06. Empty = first 10 cases.")
    parser.add_argument("--persistence-history-min", type=float, default=10.0)
    parser.add_argument("--future-risk-margin-norm", type=float, default=0.25)
    parser.add_argument("--relief-margin-norm", type=float, default=0.25)
    parser.add_argument("--low-risk-norm", type=float, default=0.75)
    return parser.parse_args()


def _resolve(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else repo_root / p


def _planner_base() -> tuple[float, list[float]]:
    base = repo_root / "outputs" / "wind_prediction" / "planner_a1_dryrun"
    sign_cfg = json.loads((base / "diagnostics" / "a01_pressure_vec_sign_convention.json").read_text())
    discount_cfg = json.loads((base / "diagnostics" / "a1_block_discount_config.json").read_text())
    return float(sign_cfg["planner_sign_multiplier"]), list(discount_cfg["default_discount_blocks"])


def _speed_dir(uv: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    speed = np.sqrt(uv[:, 0] ** 2 + uv[:, 1] ** 2)
    direction = (np.rad2deg(np.arctan2(-uv[:, 0], -uv[:, 1])) + 360.0) % 360.0
    return speed.astype(float), direction.astype(float)


def _angle_mae_deg(a: np.ndarray, b: np.ndarray) -> float:
    delta = (np.asarray(a, dtype=float) - np.asarray(b, dtype=float) + 180.0) % 360.0 - 180.0
    return float(np.mean(np.abs(delta)))


def _event_probs_for_oracle(sample: Any, replay: Fino1ReplayDataset) -> dict[str, float]:
    return {
        name: float(sample.y_event[idx])
        for idx, name in enumerate(replay.event_columns)
    }


def _empty_event_probs() -> dict[str, float]:
    return {key: 0.0 for key in EVENT_RISK_KEYS}


def _regime_from_oracle(
    norms: list[float],
    dot02: float,
    future_risk_margin_norm: float,
    relief_margin_norm: float,
    low_risk_norm: float,
) -> str:
    max_norm = max(norms)
    if max_norm <= low_risk_norm:
        return "lowrisk"
    if dot02 < 0.0:
        return "signflip"
    if norms[2] + relief_margin_norm <= norms[0]:
        return "future_relief"
    if norms[2] >= norms[0] + future_risk_margin_norm:
        return "future_risk_onset"
    if max_norm >= 1.0:
        return "sustained_high"
    return "stable_moderate"


def _evaluate_source(
    *,
    case_id: str,
    case_label: str,
    bucket: int,
    sample: Any,
    replay: Fino1ReplayDataset,
    source: str,
    uv: np.ndarray,
    oracle_uv: np.ndarray,
    persistence_uv: np.ndarray,
    event_probs: dict[str, float],
    cfg: PlannerConfig,
    discounts: list[float],
    envelope_mode: str,
) -> dict[str, Any]:
    blocks = compute_pressure_blocks(uv, discounts, cfg)
    evaluated = [evaluate_sequence(seq, blocks, DEFAULT_PLANT_INFO, cfg) for seq in _ALL_SEQUENCES]
    feasible, scalar_costs = select_best_economic(evaluated, cfg)
    if feasible:
        best = feasible[0]
        scalar = float(scalar_costs[0])
    else:
        best = {"sequence": ("hold", "hold", "hold"), "costs": {}}
        scalar = float("nan")

    norms = [float(b.get("pressure_norm", 0.0)) for b in blocks[:3]]
    raws = [
        norm_term(np.asarray(b.get("pressure_vec_raw", b.get("pressure_vec")), dtype=float), cfg)
        for b in blocks[:3]
    ]
    while len(norms) < 3:
        norms.append(0.0)
        raws.append(0.0)
    vec0 = np.asarray(blocks[0].get("pressure_vec", np.zeros(2)), dtype=float)
    vec2 = np.asarray(blocks[2].get("pressure_vec", np.zeros(2)), dtype=float)
    dot02 = float(np.dot(vec0, vec2))
    speed, direction = _speed_dir(uv)
    oracle_speed, oracle_direction = _speed_dir(oracle_uv)
    persistence_speed, persistence_direction = _speed_dir(persistence_uv)
    costs = best.get("costs", {})
    first_action = str(best["sequence"][0])
    severity = int(ACTION_SEVERITY.get(first_action, 0))
    return {
        "case_id": case_id,
        "case_label": case_label,
        "bucket": int(bucket),
        "history_end": sample.history_end.strftime(TIMESTAMP_FMT),
        "envelope_mode": envelope_mode,
        "source": source,
        "first_action": first_action,
        "action_severity": severity,
        "best_sequence": ">".join(str(x) for x in best["sequence"]),
        "best_scalar_cost": scalar,
        "max_envelope_norm": float(costs.get("max_envelope_norm", np.nan)),
        "envelope_violation_cost": float(costs.get("envelope_violation_cost", np.nan)),
        "terminal_envelope_violation": float(costs.get("terminal_envelope_violation", np.nan)),
        "pump_work_cost": float(costs.get("pump_work_cost", np.nan)),
        "pump_duration_cost": float(costs.get("pump_duration_cost", np.nan)),
        "startstop_cost": float(costs.get("startstop_cost", np.nan)),
        "pressure_block0_norm": norms[0],
        "pressure_block1_norm": norms[1],
        "pressure_block2_norm": norms[2],
        "raw_pressure_block0_norm": raws[0],
        "raw_pressure_block1_norm": raws[1],
        "raw_pressure_block2_norm": raws[2],
        "pressure_block02_dot": dot02,
        "forecast_speed_mean_ms": float(np.mean(speed)),
        "forecast_speed_max_ms": float(np.max(speed)),
        "forecast_speed_block0_ms": float(np.mean(speed[:2])),
        "forecast_speed_block2_ms": float(np.mean(speed[4:6])),
        "forecast_speed_block2_minus_block0_ms": float(np.mean(speed[4:6]) - np.mean(speed[:2])),
        "forecast_dir_block0_deg": float(np.mean(direction[:2])),
        "forecast_dir_block2_deg": float(np.mean(direction[4:6])),
        "uv_rmse_vs_oracle_ms": float(np.sqrt(np.mean((uv - oracle_uv) ** 2))),
        "uv_rmse_vs_persistence_ms": float(np.sqrt(np.mean((uv - persistence_uv) ** 2))),
        "speed_rmse_vs_oracle_ms": float(np.sqrt(np.mean((speed - oracle_speed) ** 2))),
        "speed_rmse_vs_persistence_ms": float(np.sqrt(np.mean((speed - persistence_speed) ** 2))),
        "dir_mae_vs_oracle_deg": _angle_mae_deg(direction, oracle_direction),
        "dir_mae_vs_persistence_deg": _angle_mae_deg(direction, persistence_direction),
        "event_risk_prob_0_20m": float(event_probs.get(EVENT_RISK_KEYS[0], 0.0)),
        "event_risk_prob_20_40m": float(event_probs.get(EVENT_RISK_KEYS[1], 0.0)),
        "event_risk_prob_40_60m": float(event_probs.get(EVENT_RISK_KEYS[2], 0.0)),
    }


def _mechanism(row: pd.Series) -> str:
    o = int(row["oracle_severity"])
    l = int(row["learned_severity"])
    p = int(row["persistence_severity"])
    if o == l == p:
        return "all_sources_same_action"
    if o > p:
        if l > p:
            return "learned_more_active_than_persistence"
        return "oracle_more_active_upper_bound_only"
    if o < p:
        if l < p:
            return "learned_less_active_than_persistence"
        return "oracle_less_active_upper_bound_only"
    if l > p:
        return "learned_extra_action_without_oracle_support"
    if l < p:
        return "learned_extra_suppression_without_oracle_support"
    return "forecast_sources_differ_same_severity"


def _wide_bucket_summary(detail: pd.DataFrame, future_risk_margin: float, relief_margin: float, lowrisk: float) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    grouped = detail.groupby(["envelope_mode", "case_id", "bucket", "history_end"], sort=True)
    for (mode, case_id, bucket, history_end), g in grouped:
        by = {str(r["source"]): r for _, r in g.iterrows()}
        oracle = by["oracle"]
        learned = by["learned"]
        persistence = by["persistence"]
        oracle_norms = [
            float(oracle["pressure_block0_norm"]),
            float(oracle["pressure_block1_norm"]),
            float(oracle["pressure_block2_norm"]),
        ]
        regime = _regime_from_oracle(
            oracle_norms,
            float(oracle["pressure_block02_dot"]),
            future_risk_margin,
            relief_margin,
            lowrisk,
        )
        rec = {
            "envelope_mode": mode,
            "case_id": case_id,
            "case_label": str(oracle["case_label"]),
            "bucket": int(bucket),
            "history_end": history_end,
            "oracle_regime": regime,
        }
        for source, r in by.items():
            prefix = source
            rec[f"{prefix}_action"] = str(r["first_action"])
            rec[f"{prefix}_severity"] = int(r["action_severity"])
            rec[f"{prefix}_sequence"] = str(r["best_sequence"])
            rec[f"{prefix}_n0"] = float(r["pressure_block0_norm"])
            rec[f"{prefix}_n1"] = float(r["pressure_block1_norm"])
            rec[f"{prefix}_n2"] = float(r["pressure_block2_norm"])
            rec[f"{prefix}_raw_n0"] = float(r["raw_pressure_block0_norm"])
            rec[f"{prefix}_raw_n1"] = float(r["raw_pressure_block1_norm"])
            rec[f"{prefix}_raw_n2"] = float(r["raw_pressure_block2_norm"])
            rec[f"{prefix}_max_env"] = float(r["max_envelope_norm"])
            rec[f"{prefix}_scalar_cost"] = float(r["best_scalar_cost"])
            rec[f"{prefix}_pump_work_cost"] = float(r["pump_work_cost"])
            rec[f"{prefix}_event2"] = float(r["event_risk_prob_40_60m"])
        rec["learned_matches_persistence_action"] = int(rec["learned_action"] == rec["persistence_action"])
        rec["learned_matches_oracle_action"] = int(rec["learned_action"] == rec["oracle_action"])
        rec["persistence_matches_oracle_action"] = int(rec["persistence_action"] == rec["oracle_action"])
        rec["learned_uv_rmse_vs_oracle_ms"] = float(learned["uv_rmse_vs_oracle_ms"])
        rec["persistence_uv_rmse_vs_oracle_ms"] = float(persistence["uv_rmse_vs_oracle_ms"])
        rec["learned_uv_rmse_vs_persistence_ms"] = float(learned["uv_rmse_vs_persistence_ms"])
        rec["learned_speed_block2_minus_block0_ms"] = float(learned["forecast_speed_block2_minus_block0_ms"])
        rec["oracle_speed_block2_minus_block0_ms"] = float(oracle["forecast_speed_block2_minus_block0_ms"])
        rec["persistence_speed_block2_minus_block0_ms"] = float(persistence["forecast_speed_block2_minus_block0_ms"])
        rows.append(rec)
    out = pd.DataFrame(rows)
    out["mechanism"] = out.apply(_mechanism, axis=1)
    return out


def _fraction_table(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    counts = df.groupby(group_cols + ["mechanism"], dropna=False).size().reset_index(name="buckets")
    totals = df.groupby(group_cols, dropna=False).size().reset_index(name="total_buckets")
    merged = counts.merge(totals, on=group_cols, how="left")
    merged["bucket_ratio"] = merged["buckets"] / merged["total_buckets"].clip(lower=1)
    return merged.sort_values(group_cols + ["mechanism"]).reset_index(drop=True)


def _write_report(
    *,
    out_dir: Path,
    dataset_dir: Path,
    model_dir: Path,
    detail_path: Path,
    bucket_path: Path,
    mechanism_path: Path,
    case_path: Path,
    bucket_summary: pd.DataFrame,
    mechanism_summary: pd.DataFrame,
    case_summary: pd.DataFrame,
) -> None:
    lines = [
        "# Prediction Value Mechanism Audit",
        "",
        "Planner-only scan. No 1 Hz plant simulation; no event floor, preview lead, or pump suppression.",
        "",
        f"- dataset: `{dataset_dir.relative_to(repo_root) if dataset_dir.is_relative_to(repo_root) else dataset_dir}`",
        f"- learned model: `{model_dir.relative_to(repo_root) if model_dir.is_relative_to(repo_root) else model_dir}`",
        f"- buckets: `{len(bucket_summary)}`",
        "",
        "## Mechanism Summary",
        "",
        "| envelope | regime | mechanism | buckets | ratio |",
        "|---|---|---|---:|---:|",
    ]
    show = mechanism_summary[
        mechanism_summary["mechanism"].ne("all_sources_same_action")
        | (mechanism_summary["bucket_ratio"] > 0.0)
    ]
    for _, r in show.iterrows():
        lines.append(
            f"| {r['envelope_mode']} | {r['oracle_regime']} | {r['mechanism']} | "
            f"{int(r['buckets'])} | {float(r['bucket_ratio']) * 100:.1f}% |"
        )
    lines.extend(
        [
            "",
            "## Case Summary",
            "",
            "| envelope | case | buckets | L=P action | L=O action | P=O action | L RMSE vs O | P RMSE vs O |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for _, r in case_summary.iterrows():
        lines.append(
            f"| {r['envelope_mode']} | {r['case_id']} | {int(r['buckets'])} | "
            f"{float(r['learned_matches_persistence_action_ratio']) * 100:.1f}% | "
            f"{float(r['learned_matches_oracle_action_ratio']) * 100:.1f}% | "
            f"{float(r['persistence_matches_oracle_action_ratio']) * 100:.1f}% | "
            f"{float(r['learned_uv_rmse_vs_oracle_ms']):.2f} | "
            f"{float(r['persistence_uv_rmse_vs_oracle_ms']):.2f} |"
        )
    lines.extend(
        [
            "",
            "## High-Signal Buckets",
            "",
            "| envelope | case | bucket | regime | mechanism | O/L/P actions | O n0/n1/n2 | L n0/n1/n2 | P n0/n1/n2 |",
            "|---|---|---:|---|---|---|---|---|---|",
        ]
    )
    high_signal = bucket_summary[
        bucket_summary["mechanism"].ne("all_sources_same_action")
    ].copy()
    for _, r in high_signal.head(80).iterrows():
        lines.append(
            f"| {r['envelope_mode']} | {r['case_id']} | {int(r['bucket'])} | "
            f"{r['oracle_regime']} | {r['mechanism']} | "
            f"{r['oracle_action']}/{r['learned_action']}/{r['persistence_action']} | "
            f"{r['oracle_n0']:.2f}/{r['oracle_n1']:.2f}/{r['oracle_n2']:.2f} | "
            f"{r['learned_n0']:.2f}/{r['learned_n1']:.2f}/{r['learned_n2']:.2f} | "
            f"{r['persistence_n0']:.2f}/{r['persistence_n1']:.2f}/{r['persistence_n2']:.2f} |"
        )
    lines.extend(
        [
            "",
            "## Files",
            "",
            f"- detail: `{detail_path.relative_to(repo_root)}`",
            f"- bucket summary: `{bucket_path.relative_to(repo_root)}`",
            f"- mechanism summary: `{mechanism_path.relative_to(repo_root)}`",
            f"- case summary: `{case_path.relative_to(repo_root)}`",
        ]
    )
    report_path = out_dir / "prediction_value_mechanism_audit_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Report: {report_path}")


def main() -> None:
    args = parse_args()
    dataset_dir = _resolve(str(args.dataset_dir))
    model_dir = _resolve(str(args.model_dir))
    out_dir = _resolve(str(args.out_dir))
    out_dir.mkdir(parents=True, exist_ok=True)
    selected_ids = {x.strip() for x in str(args.case_ids).split(",") if x.strip()}
    cases = [
        c for c in CASES
        if not selected_ids or c[0][:2] in selected_ids or c[0] in selected_ids
    ]

    replay = Fino1ReplayDataset(dataset_dir=dataset_dir, split="test")
    learned_adapter = ForecastModelAdapter(model_dir=model_dir, dataset_dir=dataset_dir, device="cpu")
    persistence_adapter = PersistenceMeanForecastAdapter(
        dataset_dir=dataset_dir,
        history_minutes=float(args.persistence_history_min),
    )
    sign_multiplier, discounts = _planner_base()
    buckets_per_case = int(np.ceil(float(args.duration_s) / replay.update_interval_s))

    rows: list[dict[str, Any]] = []
    for case_id, ts_str, label in cases:
        start = datetime.strptime(ts_str, TIMESTAMP_FMT)
        for bucket in range(buckets_per_case):
            ts = replay.simulation_timestamp(start, bucket * replay.update_interval_s)
            sample = replay.sample_for_history_end(ts)
            if sample is None:
                print(f"[WARN] missing sample for {case_id} {ts.strftime(TIMESTAMP_FMT)}")
                continue
            oracle_uv = np.asarray(sample.y_uv_raw, dtype=float)
            oracle_events = _event_probs_for_oracle(sample, replay)
            learned_result = learned_adapter.predict_window(
                sample.x_window,
                timestamp=sample.history_end.strftime(TIMESTAMP_FMT),
            )
            learned_uv = np.asarray(learned_result.wind_uv_raw, dtype=float)
            learned_events = dict(learned_result.event_probs)
            persistence_result = persistence_adapter.predict_window(
                sample.x_window,
                timestamp=sample.history_end.strftime(TIMESTAMP_FMT),
            )
            persistence_uv = np.asarray(persistence_result.wind_uv_raw, dtype=float)
            for envelope_mode, envelope_use_discount in (("discounted", True), ("raw", False)):
                cfg = PlannerConfig(
                    pressure_sign_multiplier=sign_multiplier,
                    envelope_use_discount=envelope_use_discount,
                )
                for source, uv, events in (
                    ("oracle", oracle_uv, oracle_events),
                    ("learned", learned_uv, learned_events),
                    ("persistence", persistence_uv, _empty_event_probs()),
                ):
                    rows.append(
                        _evaluate_source(
                            case_id=case_id,
                            case_label=label,
                            bucket=bucket,
                            sample=sample,
                            replay=replay,
                            source=source,
                            uv=uv,
                            oracle_uv=oracle_uv,
                            persistence_uv=persistence_uv,
                            event_probs=events,
                            cfg=cfg,
                            discounts=discounts,
                            envelope_mode=envelope_mode,
                        )
                    )

    detail = pd.DataFrame(rows)
    bucket_summary = _wide_bucket_summary(
        detail,
        future_risk_margin=float(args.future_risk_margin_norm),
        relief_margin=float(args.relief_margin_norm),
        lowrisk=float(args.low_risk_norm),
    )
    mechanism_summary = _fraction_table(bucket_summary, ["envelope_mode", "oracle_regime"])
    case_summary = (
        bucket_summary
        .groupby(["envelope_mode", "case_id"], dropna=False)
        .agg(
            buckets=("bucket", "count"),
            learned_matches_persistence_action_ratio=("learned_matches_persistence_action", "mean"),
            learned_matches_oracle_action_ratio=("learned_matches_oracle_action", "mean"),
            persistence_matches_oracle_action_ratio=("persistence_matches_oracle_action", "mean"),
            learned_uv_rmse_vs_oracle_ms=("learned_uv_rmse_vs_oracle_ms", "mean"),
            persistence_uv_rmse_vs_oracle_ms=("persistence_uv_rmse_vs_oracle_ms", "mean"),
        )
        .reset_index()
    )

    detail_path = out_dir / "prediction_value_mechanism_detail.csv"
    bucket_path = out_dir / "prediction_value_mechanism_buckets.csv"
    mechanism_path = out_dir / "prediction_value_mechanism_summary.csv"
    case_path = out_dir / "prediction_value_mechanism_case_summary.csv"
    detail.to_csv(detail_path, index=False)
    bucket_summary.to_csv(bucket_path, index=False)
    mechanism_summary.to_csv(mechanism_path, index=False)
    case_summary.to_csv(case_path, index=False)
    _write_report(
        out_dir=out_dir,
        dataset_dir=dataset_dir,
        model_dir=model_dir,
        detail_path=detail_path,
        bucket_path=bucket_path,
        mechanism_path=mechanism_path,
        case_path=case_path,
        bucket_summary=bucket_summary,
        mechanism_summary=mechanism_summary,
        case_summary=case_summary,
    )
    print(case_summary.to_string(index=False))


if __name__ == "__main__":
    main()
