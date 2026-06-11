#!/usr/bin/env python3
"""Audit forecast signals that can mislead prediction-primary control.

This is an offline diagnostic. It loads replay windows, compares learned and
persistence forecasts against the oracle future, and scores the control-facing
gates that would justify pump saving, hold-current, or preview action.

It does not run the 1 Hz plant simulation and does not change controller
behavior.
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

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from wind_prediction import Fino1ReplayDataset, ForecastModelAdapter, PersistenceMeanForecastAdapter  # noqa: E402
from wind_prediction.ballast_planner import PlannerConfig, compute_pressure_blocks, norm_term  # noqa: E402


TIMESTAMP_FMT = "%Y-%m-%d %H:%M:%S"
EVENT_RISK_KEYS = (
    "attention_event_0_20m",
    "attention_event_20_40m",
    "attention_event_40_60m",
)
GATE_COLUMNS = (
    "relief_gate",
    "lowrisk_gate",
    "signflip_gate",
    "strong_signflip_gate",
    "suppression_gate",
    "future_risk_gate",
    "event_gate",
    "risky_gate",
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
        default="outputs/wind_prediction/forecast_signal_reliability_v1",
    )
    parser.add_argument("--split", default="test")
    parser.add_argument(
        "--case-csv",
        action="append",
        default=[],
        help=(
            "Optional CSV with timestamp plus optional case_id,label columns. "
            "May be repeated. If omitted, the split is sampled by stride/limit."
        ),
    )
    parser.add_argument(
        "--case-ids",
        default="",
        help="Optional comma-separated case ids to keep from all --case-csv files.",
    )
    parser.add_argument(
        "--duration-s",
        type=float,
        default=10800.0,
        help="Replay duration per case for case CSV inputs. Offline only.",
    )
    parser.add_argument("--stride", type=int, default=30)
    parser.add_argument(
        "--limit",
        type=int,
        default=600,
        help="Maximum sampled windows when --case-csv is omitted. 0 means all stride-selected windows.",
    )
    parser.add_argument("--persistence-history-min", type=float, default=10.0)
    parser.add_argument("--relief-margin-norm", type=float, default=0.25)
    parser.add_argument("--future-risk-margin-norm", type=float, default=0.25)
    parser.add_argument("--low-risk-norm", type=float, default=0.75)
    parser.add_argument("--high-risk-norm", type=float, default=1.20)
    parser.add_argument("--signflip-norm", type=float, default=1.20)
    parser.add_argument("--signflip-future-norm", type=float, default=0.70)
    parser.add_argument("--signflip-cos", type=float, default=-0.30)
    parser.add_argument("--event-risk-threshold", type=float, default=0.70)
    parser.add_argument("--pressure-underpredict-margin-norm", type=float, default=0.25)
    parser.add_argument("--pressure-agreement-tol-norm", type=float, default=0.20)
    return parser.parse_args()


def _resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p


def _display(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _safe_div(num: float, den: float) -> float:
    return float(num) / float(den) if abs(float(den)) > 1e-12 else 0.0


def _load_planner_base() -> tuple[PlannerConfig, list[float]]:
    base = REPO_ROOT / "outputs" / "wind_prediction" / "planner_a1_dryrun"
    sign_path = base / "diagnostics" / "a01_pressure_vec_sign_convention.json"
    discount_path = base / "diagnostics" / "a1_block_discount_config.json"
    sign_multiplier = -1.0
    discounts = [1.0, 0.85, 0.70]
    if sign_path.exists():
        sign_cfg = json.loads(sign_path.read_text(encoding="utf-8"))
        sign_multiplier = float(sign_cfg.get("planner_sign_multiplier", sign_multiplier))
    if discount_path.exists():
        discount_cfg = json.loads(discount_path.read_text(encoding="utf-8"))
        discounts = [float(v) for v in discount_cfg.get("default_discount_blocks", discounts)]
    return PlannerConfig(pressure_sign_multiplier=sign_multiplier), discounts


def _load_case_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    keep_ids = {x.strip() for x in str(args.case_ids).split(",") if x.strip()}
    rows: list[dict[str, Any]] = []
    for csv_arg in args.case_csv:
        csv_path = _resolve(csv_arg)
        frame = pd.read_csv(csv_path)
        if "timestamp" not in frame.columns:
            raise ValueError(f"{csv_path} is missing required column: timestamp")
        case_group = csv_path.stem
        if case_group.endswith("_cases"):
            case_group = case_group[:-6]
        for idx, row in enumerate(frame.to_dict("records"), start=1):
            case_id = str(row.get("case_id", "") or row.get("group", "") or f"case_{idx:02d}")
            if keep_ids and case_id not in keep_ids:
                continue
            rows.append(
                {
                    "case_group": case_group,
                    "case_id": case_id,
                    "case_label": str(row.get("label", "") or row.get("selection_group", "") or case_id),
                    "start_timestamp": str(row["timestamp"]),
                }
            )
    return rows


def _sample_plan(args: argparse.Namespace, replay: Fino1ReplayDataset) -> list[dict[str, Any]]:
    case_rows = _load_case_rows(args)
    if case_rows:
        buckets_per_case = int(np.ceil(float(args.duration_s) / float(replay.update_interval_s)))
        plan: list[dict[str, Any]] = []
        for case in case_rows:
            start = datetime.strptime(str(case["start_timestamp"]), TIMESTAMP_FMT)
            for bucket in range(buckets_per_case):
                plan.append(
                    {
                        "case_group": case["case_group"],
                        "case_id": case["case_id"],
                        "case_label": case["case_label"],
                        "bucket": int(bucket),
                        "timestamp": replay.simulation_timestamp(
                            start,
                            bucket * replay.update_interval_s,
                        ),
                    }
                )
        return plan

    stride = max(1, int(args.stride))
    limit = int(args.limit)
    plan = []
    for idx, ts in enumerate(replay.iter_sample_timestamps()):
        if idx % stride != 0:
            continue
        plan.append(
            {
                "case_group": f"{args.split}_stride_{stride}",
                "case_id": f"sample_{idx:06d}",
                "case_label": "stride sample",
                "bucket": 0,
                "timestamp": ts,
            }
        )
        if limit > 0 and len(plan) >= limit:
            break
    return plan


def _speed_dir(uv: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(uv, dtype=float)
    speed = np.sqrt(arr[:, 0] ** 2 + arr[:, 1] ** 2)
    direction = (np.rad2deg(np.arctan2(-arr[:, 0], -arr[:, 1])) + 360.0) % 360.0
    return speed.astype(float), direction.astype(float)


def _angle_diff_deg(a: np.ndarray | float, b: np.ndarray | float) -> np.ndarray:
    return (np.asarray(a, dtype=float) - np.asarray(b, dtype=float) + 180.0) % 360.0 - 180.0


def _circ_mean_deg(values: np.ndarray) -> float:
    rad = np.deg2rad(np.asarray(values, dtype=float))
    return float((np.rad2deg(np.arctan2(np.mean(np.sin(rad)), np.mean(np.cos(rad)))) + 360.0) % 360.0)


def _event_probs_for_oracle(sample: Any, replay: Fino1ReplayDataset) -> dict[str, float]:
    return {name: float(sample.y_event[idx]) for idx, name in enumerate(replay.event_columns)}


def _empty_event_probs() -> dict[str, float]:
    return {key: 0.0 for key in EVENT_RISK_KEYS}


def _source_features(
    *,
    uv: np.ndarray,
    event_probs: dict[str, float],
    cfg: PlannerConfig,
    discounts: list[float],
    args: argparse.Namespace,
) -> dict[str, float | int]:
    arr = np.asarray(uv, dtype=float)
    blocks = compute_pressure_blocks(arr, discounts, cfg)
    norms = [float(b.get("pressure_norm", 0.0)) for b in blocks[:3]]
    raw_norms = [
        norm_term(np.asarray(b.get("pressure_vec_raw", b.get("pressure_vec")), dtype=float), cfg)
        for b in blocks[:3]
    ]
    while len(norms) < 3:
        norms.append(0.0)
        raw_norms.append(0.0)

    vec0 = np.asarray(blocks[0].get("pressure_vec", np.zeros(2)), dtype=float)
    vec2 = np.asarray(blocks[2].get("pressure_vec", np.zeros(2)), dtype=float)
    dot02 = float(np.dot(vec0, vec2))
    denom = float(np.linalg.norm(vec0) * np.linalg.norm(vec2))
    cos02 = dot02 / denom if denom > 1e-12 else float("nan")

    speed, direction = _speed_dir(arr)
    block0_speed = float(np.mean(speed[:2]))
    block1_speed = float(np.mean(speed[2:4]))
    block2_speed = float(np.mean(speed[4:6]))
    dir_block0 = _circ_mean_deg(direction[:2])
    dir_block2 = _circ_mean_deg(direction[4:6])
    dir_shift_block02 = abs(float(_angle_diff_deg(dir_block2, dir_block0)))
    dir_shift_max = float(np.max(np.abs(_angle_diff_deg(direction, direction[0]))))

    event_vals = [float(event_probs.get(key, 0.0)) for key in EVENT_RISK_KEYS]
    event_max = max(event_vals) if event_vals else 0.0
    max_norm = max(norms)
    relief_margin = norms[0] - norms[2]
    future_rise = norms[2] - norms[0]
    relief_gate = norms[2] + float(args.relief_margin_norm) <= norms[0]
    lowrisk_gate = max_norm <= float(args.low_risk_norm)
    signflip_gate = dot02 < 0.0
    strong_signflip_gate = (
        norms[0] >= float(args.signflip_norm)
        and norms[2] >= float(args.signflip_future_norm)
        and np.isfinite(cos02)
        and cos02 <= float(args.signflip_cos)
    )
    future_risk_gate = (
        norms[2] >= float(args.high_risk_norm)
        and future_rise >= float(args.future_risk_margin_norm)
    )
    event_gate = event_max >= float(args.event_risk_threshold)
    risky_gate = max_norm >= float(args.high_risk_norm)
    suppression_gate = bool(relief_gate or lowrisk_gate or signflip_gate)

    return {
        "n0": norms[0],
        "n1": norms[1],
        "n2": norms[2],
        "raw_n0": raw_norms[0],
        "raw_n1": raw_norms[1],
        "raw_n2": raw_norms[2],
        "max_norm": float(max_norm),
        "relief_margin_norm": float(relief_margin),
        "future_rise_norm": float(future_rise),
        "pressure_dot02": float(dot02),
        "pressure_cos02": float(cos02),
        "speed_block0_ms": block0_speed,
        "speed_block1_ms": block1_speed,
        "speed_block2_ms": block2_speed,
        "speed_range_ms": float(np.max(speed) - np.min(speed)),
        "speed_drop02_ms": float(block0_speed - block2_speed),
        "speed_rise02_ms": float(block2_speed - block0_speed),
        "dir_block0_deg": dir_block0,
        "dir_block2_deg": dir_block2,
        "dir_shift_block02_deg": dir_shift_block02,
        "dir_shift_max_deg": dir_shift_max,
        "event_prob_0_20m": event_vals[0] if len(event_vals) > 0 else 0.0,
        "event_prob_20_40m": event_vals[1] if len(event_vals) > 1 else 0.0,
        "event_prob_40_60m": event_vals[2] if len(event_vals) > 2 else 0.0,
        "event_prob_max": float(event_max),
        "relief_gate": int(relief_gate),
        "lowrisk_gate": int(lowrisk_gate),
        "signflip_gate": int(signflip_gate),
        "strong_signflip_gate": int(strong_signflip_gate),
        "suppression_gate": int(suppression_gate),
        "future_risk_gate": int(future_risk_gate),
        "event_gate": int(event_gate),
        "risky_gate": int(risky_gate),
    }


def _forecast_errors(pred_uv: np.ndarray, oracle_uv: np.ndarray) -> dict[str, float]:
    pred_speed, pred_dir = _speed_dir(pred_uv)
    oracle_speed, oracle_dir = _speed_dir(oracle_uv)
    return {
        "uv_rmse_vs_oracle_ms": float(np.sqrt(np.mean((np.asarray(pred_uv) - np.asarray(oracle_uv)) ** 2))),
        "speed_rmse_vs_oracle_ms": float(np.sqrt(np.mean((pred_speed - oracle_speed) ** 2))),
        "dir_mae_vs_oracle_deg": float(np.mean(np.abs(_angle_diff_deg(pred_dir, oracle_dir)))),
    }


def _prefix_dict(prefix: str, values: dict[str, Any]) -> dict[str, Any]:
    return {f"{prefix}_{key}": value for key, value in values.items()}


def _add_pair_flags(row: dict[str, Any], source: str, args: argparse.Namespace) -> None:
    for gate in GATE_COLUMNS:
        pred = int(row[f"{source}_{gate}"])
        truth = int(row[f"oracle_{gate}"])
        row[f"{source}_{gate}_fp"] = int(pred == 1 and truth == 0)
        row[f"{source}_{gate}_fn"] = int(pred == 0 and truth == 1)
        row[f"{source}_{gate}_match"] = int(pred == truth)

    row[f"{source}_max_norm_underpredict"] = int(
        float(row[f"{source}_max_norm"]) + float(args.pressure_underpredict_margin_norm)
        < float(row["oracle_max_norm"])
    )
    row[f"{source}_far_norm_underpredict"] = int(
        float(row[f"{source}_n2"]) + float(args.pressure_underpredict_margin_norm)
        < float(row["oracle_n2"])
    )
    row[f"{source}_false_relief_rising"] = int(
        int(row[f"{source}_relief_gate"]) == 1 and float(row["oracle_future_rise_norm"]) > 0.0
    )
    row[f"{source}_event_gate_fn"] = int(
        int(row["oracle_event_gate"]) == 1 and int(row[f"{source}_event_gate"]) == 0
    )
    row[f"{source}_dangerous_optimism"] = int(
        int(row[f"{source}_suppression_gate_fp"]) == 1
        or int(row[f"{source}_max_norm_underpredict"]) == 1
        or int(row[f"{source}_event_gate_fn"]) == 1
    )


def _add_learned_confidence_flags(row: dict[str, Any], args: argparse.Namespace) -> None:
    pressure_delta = max(
        abs(float(row["learned_n0"]) - float(row["persistence_n0"])),
        abs(float(row["learned_n1"]) - float(row["persistence_n1"])),
        abs(float(row["learned_n2"]) - float(row["persistence_n2"])),
    )
    row["learned_persistence_pressure_delta_max"] = float(pressure_delta)
    row["learned_persistence_pressure_agree"] = int(
        pressure_delta <= float(args.pressure_agreement_tol_norm)
    )
    row["learned_persistence_suppression_agree"] = int(
        int(row["learned_suppression_gate"]) == int(row["persistence_suppression_gate"])
    )
    row["learned_persistence_relief_agree"] = int(
        int(row["learned_relief_gate"]) == int(row["persistence_relief_gate"])
    )
    row["learned_suppression_fp_when_persistence_agrees"] = int(
        int(row["learned_suppression_gate_fp"]) == 1
        and int(row["learned_persistence_suppression_agree"]) == 1
    )
    row["learned_suppression_fp_when_persistence_disagrees"] = int(
        int(row["learned_suppression_gate_fp"]) == 1
        and int(row["learned_persistence_suppression_agree"]) == 0
    )


def _confusion_table(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for source in ("learned", "persistence"):
        for gate in GATE_COLUMNS:
            truth = df[f"oracle_{gate}"].astype(bool)
            pred = df[f"{source}_{gate}"].astype(bool)
            tp = int((truth & pred).sum())
            fp = int((~truth & pred).sum())
            fn = int((truth & ~pred).sum())
            tn = int((~truth & ~pred).sum())
            precision = _safe_div(tp, tp + fp)
            recall = _safe_div(tp, tp + fn)
            f1 = _safe_div(2.0 * precision * recall, precision + recall)
            rows.append(
                {
                    "source": source,
                    "gate": gate,
                    "windows": int(len(df)),
                    "truth_rate": float(truth.mean()),
                    "pred_rate": float(pred.mean()),
                    "match_rate": float((truth == pred).mean()),
                    "tp": tp,
                    "fp": fp,
                    "fn": fn,
                    "tn": tn,
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                    "false_positive_rate": _safe_div(fp, fp + tn),
                    "false_negative_rate": _safe_div(fn, fn + tp),
                }
            )
    return pd.DataFrame(rows)


def _error_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for source in ("learned", "persistence"):
        row: dict[str, Any] = {
            "source": source,
            "windows": int(len(df)),
            "uv_rmse_vs_oracle_ms": float(df[f"{source}_uv_rmse_vs_oracle_ms"].mean()),
            "speed_rmse_vs_oracle_ms": float(df[f"{source}_speed_rmse_vs_oracle_ms"].mean()),
            "dir_mae_vs_oracle_deg": float(df[f"{source}_dir_mae_vs_oracle_deg"].mean()),
            "n0_abs_err": float((df[f"{source}_n0"] - df["oracle_n0"]).abs().mean()),
            "n1_abs_err": float((df[f"{source}_n1"] - df["oracle_n1"]).abs().mean()),
            "n2_abs_err": float((df[f"{source}_n2"] - df["oracle_n2"]).abs().mean()),
            "max_norm_underpredict_rate": float(df[f"{source}_max_norm_underpredict"].mean()),
            "far_norm_underpredict_rate": float(df[f"{source}_far_norm_underpredict"].mean()),
            "dangerous_optimism_rate": float(df[f"{source}_dangerous_optimism"].mean()),
        }
        rows.append(row)
    return pd.DataFrame(rows)


def _case_summary(df: pd.DataFrame) -> pd.DataFrame:
    agg = (
        df.groupby(["case_group", "case_id"], dropna=False)
        .agg(
            windows=("history_end", "count"),
            learned_uv_rmse_vs_oracle_ms=("learned_uv_rmse_vs_oracle_ms", "mean"),
            persistence_uv_rmse_vs_oracle_ms=("persistence_uv_rmse_vs_oracle_ms", "mean"),
            learned_suppression_fp_rate=("learned_suppression_gate_fp", "mean"),
            learned_relief_fp_rate=("learned_relief_gate_fp", "mean"),
            learned_lowrisk_fp_rate=("learned_lowrisk_gate_fp", "mean"),
            learned_event_fn_rate=("learned_event_gate_fn", "mean"),
            learned_dangerous_optimism_rate=("learned_dangerous_optimism", "mean"),
            learned_persistence_pressure_agree_rate=("learned_persistence_pressure_agree", "mean"),
        )
        .reset_index()
        .sort_values(["learned_dangerous_optimism_rate", "learned_suppression_fp_rate"], ascending=False)
    )
    return agg


def _markdown_table(df: pd.DataFrame, columns: list[str] | None = None, max_rows: int | None = None) -> str:
    if df.empty:
        return "_none_"
    show = df.copy()
    if columns is not None:
        show = show[columns]
    if max_rows is not None:
        show = show.head(max_rows)
    lines = [
        "| " + " | ".join(str(c) for c in show.columns) + " |",
        "| " + " | ".join(["---"] * len(show.columns)) + " |",
    ]
    for _, row in show.iterrows():
        vals: list[str] = []
        for col in show.columns:
            val = row[col]
            if isinstance(val, (float, np.floating)):
                vals.append(f"{float(val):.3f}")
            else:
                vals.append(str(val))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def _write_report(
    *,
    args: argparse.Namespace,
    dataset_dir: Path,
    model_dir: Path,
    out_dir: Path,
    detail: pd.DataFrame,
    error_summary: pd.DataFrame,
    confusion: pd.DataFrame,
    case_summary: pd.DataFrame,
) -> None:
    learned_gate = confusion[confusion["source"].eq("learned")].copy()
    high_fp = learned_gate.sort_values("false_positive_rate", ascending=False)
    high_fn = learned_gate.sort_values("false_negative_rate", ascending=False)
    worst = detail.sort_values(
        ["learned_dangerous_optimism", "learned_uv_rmse_vs_oracle_ms"],
        ascending=[False, False],
    )
    recs = [
        "1. Treat learned relief/low-risk/sign-flip as advisory unless the gate has low false-positive rate on the target casebook.",
        "2. Do not let learned event risk be the only safety unlock when event false-negative rate is high.",
        "3. Prefer fail-closed control: release holds or fall back to current/reactive behavior when learned and persistence disagree on pressure shape.",
        "4. Promote a forecast feature to closed-loop control only after this offline gate audit and a short smooth-switch smoke both pass.",
    ]
    suppression_fp = float(
        learned_gate.loc[learned_gate["gate"].eq("suppression_gate"), "false_positive_rate"].iloc[0]
    )
    event_fn = float(
        learned_gate.loc[learned_gate["gate"].eq("event_gate"), "false_negative_rate"].iloc[0]
    )
    if suppression_fp <= 0.05 and event_fn <= 0.30:
        verdict = "usable_for_guarded_control"
    elif suppression_fp <= 0.15:
        verdict = "advisory_with_veto"
    else:
        verdict = "diagnostic_only_until_retrained_or_gated"

    lines = [
        "# Forecast Signal Reliability Audit v1",
        "",
        "Offline forecast-control diagnostic. No 1 Hz plant simulation was run.",
        "",
        f"- dataset: `{_display(dataset_dir)}`",
        f"- learned model: `{_display(model_dir)}`",
        f"- windows: `{len(detail)}`",
        f"- case CSV inputs: `{len(args.case_csv)}`",
        f"- duration per case: `{float(args.duration_s) / 3600.0:.2f} h`",
        f"- verdict: `{verdict}`",
        "",
        "## Forecast Error Summary",
        "",
        _markdown_table(error_summary),
        "",
        "## Learned Gate False Positives",
        "",
        _markdown_table(
            high_fp,
            columns=[
                "gate",
                "truth_rate",
                "pred_rate",
                "match_rate",
                "fp",
                "false_positive_rate",
                "precision",
                "recall",
            ],
        ),
        "",
        "## Learned Gate False Negatives",
        "",
        _markdown_table(
            high_fn,
            columns=[
                "gate",
                "truth_rate",
                "pred_rate",
                "match_rate",
                "fn",
                "false_negative_rate",
                "precision",
                "recall",
            ],
        ),
        "",
        "## Casebook Hotspots",
        "",
        _markdown_table(case_summary, max_rows=30),
        "",
        "## Worst Optimistic Windows",
        "",
        _markdown_table(
            worst,
            columns=[
                "case_group",
                "case_id",
                "bucket",
                "history_end",
                "learned_dangerous_optimism",
                "learned_suppression_gate_fp",
                "learned_event_gate_fn",
                "learned_uv_rmse_vs_oracle_ms",
                "oracle_n0",
                "oracle_n2",
                "learned_n0",
                "learned_n2",
                "persistence_n0",
                "persistence_n2",
            ],
            max_rows=40,
        ),
        "",
        "## Control Plan",
        "",
        *recs,
        "",
        "## Files",
        "",
        f"- detail: `{_display(out_dir / 'forecast_signal_reliability_detail.csv')}`",
        f"- errors: `{_display(out_dir / 'forecast_signal_error_summary.csv')}`",
        f"- gate confusion: `{_display(out_dir / 'forecast_signal_gate_confusion.csv')}`",
        f"- case summary: `{_display(out_dir / 'forecast_signal_case_summary.csv')}`",
    ]
    (out_dir / "forecast_signal_reliability_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    dataset_dir = _resolve(args.dataset_dir)
    model_dir = _resolve(args.model_dir)
    out_dir = _resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    replay = Fino1ReplayDataset(dataset_dir=dataset_dir, split=str(args.split))
    learned = ForecastModelAdapter(model_dir=model_dir, dataset_dir=dataset_dir, device="cpu")
    persistence = PersistenceMeanForecastAdapter(
        dataset_dir=dataset_dir,
        history_minutes=float(args.persistence_history_min),
    )
    cfg, discounts = _load_planner_base()
    plan = _sample_plan(args, replay)
    if not plan:
        raise SystemExit("No windows selected")

    rows: list[dict[str, Any]] = []
    missing = 0
    for idx, item in enumerate(plan, start=1):
        sample = replay.sample_for_history_end(item["timestamp"])
        if sample is None:
            missing += 1
            continue
        oracle_uv = np.asarray(sample.y_uv_raw, dtype=float)
        learned_result = learned.predict_window(
            sample.x_window,
            timestamp=sample.history_end.strftime(TIMESTAMP_FMT),
        )
        learned_uv = np.asarray(learned_result.wind_uv_raw, dtype=float)
        persistence_result = persistence.predict_window(
            sample.x_window,
            timestamp=sample.history_end.strftime(TIMESTAMP_FMT),
        )
        persistence_uv = np.asarray(persistence_result.wind_uv_raw, dtype=float)

        source_features = {
            "oracle": _source_features(
                uv=oracle_uv,
                event_probs=_event_probs_for_oracle(sample, replay),
                cfg=cfg,
                discounts=discounts,
                args=args,
            ),
            "learned": _source_features(
                uv=learned_uv,
                event_probs=dict(learned_result.event_probs),
                cfg=cfg,
                discounts=discounts,
                args=args,
            ),
            "persistence": _source_features(
                uv=persistence_uv,
                event_probs=_empty_event_probs(),
                cfg=cfg,
                discounts=discounts,
                args=args,
            ),
        }
        row: dict[str, Any] = {
            "case_group": item["case_group"],
            "case_id": item["case_id"],
            "case_label": item["case_label"],
            "bucket": int(item["bucket"]),
            "history_end": sample.history_end.strftime(TIMESTAMP_FMT),
        }
        for source, feats in source_features.items():
            row.update(_prefix_dict(source, feats))
        row.update(_prefix_dict("learned", _forecast_errors(learned_uv, oracle_uv)))
        row.update(_prefix_dict("persistence", _forecast_errors(persistence_uv, oracle_uv)))
        row["learned_uv_rmse_advantage_ms"] = (
            float(row["persistence_uv_rmse_vs_oracle_ms"])
            - float(row["learned_uv_rmse_vs_oracle_ms"])
        )
        _add_pair_flags(row, "learned", args)
        _add_pair_flags(row, "persistence", args)
        _add_learned_confidence_flags(row, args)
        rows.append(row)
        if idx % 200 == 0:
            print(f"processed {idx}/{len(plan)} windows", flush=True)

    detail = pd.DataFrame(rows)
    if detail.empty:
        raise SystemExit(f"No selected windows could be loaded; missing={missing}")

    error_summary = _error_summary(detail)
    confusion = _confusion_table(detail)
    case_summary = _case_summary(detail)

    detail.to_csv(out_dir / "forecast_signal_reliability_detail.csv", index=False)
    error_summary.to_csv(out_dir / "forecast_signal_error_summary.csv", index=False)
    confusion.to_csv(out_dir / "forecast_signal_gate_confusion.csv", index=False)
    case_summary.to_csv(out_dir / "forecast_signal_case_summary.csv", index=False)
    _write_report(
        args=args,
        dataset_dir=dataset_dir,
        model_dir=model_dir,
        out_dir=out_dir,
        detail=detail,
        error_summary=error_summary,
        confusion=confusion,
        case_summary=case_summary,
    )
    print(f"Report: {out_dir / 'forecast_signal_reliability_report.md'}")
    print(error_summary.to_string(index=False))
    if missing:
        print(f"[WARN] skipped missing windows: {missing}")


if __name__ == "__main__":
    main()
