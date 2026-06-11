#!/usr/bin/env python3
"""Audit 60-120 minute far-horizon signals on the guard10 case set.

This is observability-only. It does not run the controller and does not change
planner behavior. The goal is to answer whether a longer forecast horizon
contains reliable relief/risk signals that could justify a later default-off
far-horizon weighting prototype.
"""
from __future__ import annotations

import argparse
import math
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from wind_prediction.ballast_planner import PlannerConfig, norm_term, pressure_proxy_vec  # noqa: E402
from wind_prediction.forecast_adapter import ForecastModelAdapter  # noqa: E402
from wind_prediction.replay_dataset import Fino1ReplayDataset  # noqa: E402

TIMESTAMP_FMT = "%Y-%m-%d %H:%M:%S"
BLOCKS_120 = (
    ("0_20", 0, 2),
    ("20_40", 2, 4),
    ("40_60", 4, 6),
    ("60_80", 6, 8),
    ("80_100", 8, 10),
    ("100_120", 10, 12),
)
NEAR_WEIGHTS = (1.00, 0.85, 0.70)
FAR_WEIGHTS = (0.50, 0.35, 0.25)


def _resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p


def _parse_ts(raw: str) -> datetime:
    return datetime.strptime(str(raw).strip(), TIMESTAMP_FMT)


def _direction_delta_deg(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na <= 1e-9 or nb <= 1e-9:
        return 0.0
    cosv = float(np.dot(a, b) / (na * nb))
    return float(math.degrees(math.acos(max(-1.0, min(1.0, cosv)))))


def _block_features(uv: np.ndarray, cfg: PlannerConfig) -> dict[str, Any]:
    uv = np.asarray(uv, dtype=float)
    if uv.shape[0] < 12:
        raise ValueError(f"need at least 12 future steps for 120min audit, got {uv.shape}")
    vecs = []
    norms = []
    for _name, start, end in BLOCKS_120:
        vec = pressure_proxy_vec(uv[start:end], cfg)
        vecs.append(np.asarray(vec, dtype=float))
        norms.append(norm_term(vec, cfg))
    norms_arr = np.asarray(norms, dtype=float)
    near = norms_arr[:3]
    far = norms_arr[3:6]
    near_w = np.asarray(NEAR_WEIGHTS, dtype=float)
    far_w = np.asarray(FAR_WEIGHTS, dtype=float)
    dir_shift = _direction_delta_deg(vecs[2], vecs[5])
    dot_60_120 = float(np.dot(vecs[2], vecs[5]))
    near_max = float(np.max(near))
    near_last = float(near[-1])
    far_min = float(np.min(far))
    far_max = float(np.max(far))
    far_mean = float(np.mean(far))
    hidden_relief = (
        near_max >= 1.0
        and far_min <= max(0.75, near_max - 0.35)
        and far_min <= near_last - 0.25
    )
    hidden_intensification = (
        near_max <= 0.85
        and far_max >= 1.10
        and far_max >= near_max + 0.35
    )
    far_reversal = near_max >= 0.55 and far_max >= 0.55 and dot_60_120 < 0.0
    far_direction_shift = near_max >= 0.55 and far_max >= 0.55 and dir_shift >= 60.0
    return {
        "norm_0_20": float(norms_arr[0]),
        "norm_20_40": float(norms_arr[1]),
        "norm_40_60": float(norms_arr[2]),
        "norm_60_80": float(norms_arr[3]),
        "norm_80_100": float(norms_arr[4]),
        "norm_100_120": float(norms_arr[5]),
        "near_max": near_max,
        "near_mean": float(np.mean(near)),
        "near_last": near_last,
        "far_min": far_min,
        "far_max": far_max,
        "far_mean": far_mean,
        "far_last": float(far[-1]),
        "near_weighted_sum": float(np.sum(near * near_w)),
        "far_weighted_sum": float(np.sum(far * far_w)),
        "far_over_near_weighted": float(np.sum(far * far_w) / max(np.sum(near * near_w), 1e-9)),
        "dir_shift_60_to_120_deg": dir_shift,
        "dot_60_to_120": dot_60_120,
        "hidden_relief_60_120": int(hidden_relief),
        "hidden_intensification_60_120": int(hidden_intensification),
        "far_reversal_60_120": int(far_reversal),
        "far_direction_shift_60_120": int(far_direction_shift),
        "far_hint_any": int(hidden_relief or hidden_intensification or far_reversal or far_direction_shift),
    }


def _prefixed(prefix: str, features: dict[str, Any]) -> dict[str, Any]:
    return {f"{prefix}_{key}": value for key, value in features.items()}


def _confusion(df: pd.DataFrame, signal: str, truth: str) -> dict[str, Any]:
    pred = df[signal].astype(bool)
    actual = df[truth].astype(bool)
    tp = int((pred & actual).sum())
    fp = int((pred & ~actual).sum())
    fn = int((~pred & actual).sum())
    tn = int((~pred & ~actual).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    return {
        "signal": signal,
        "truth": truth,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "pred_rate": float(pred.mean()),
        "truth_rate": float(actual.mean()),
    }


def _markdown_table(df: pd.DataFrame, max_rows: int | None = None) -> str:
    if df.empty:
        return "_none_"
    if max_rows is not None:
        df = df.head(max_rows)
    lines = [
        "| " + " | ".join(map(str, df.columns)) + " |",
        "| " + " | ".join(["---"] * len(df.columns)) + " |",
    ]
    for _, row in df.iterrows():
        vals = []
        for col in df.columns:
            val = row[col]
            if isinstance(val, float):
                vals.append(f"{val:.3f}")
            else:
                vals.append(str(val))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases-csv", default="outputs/wind_prediction/broad_optimization_work/guard10_broad_cases.csv")
    parser.add_argument("--dataset-f120", default="data/processed/wind_ml_10min/ballast_decision_fino1_meteo_aux_h240_f120_v1")
    parser.add_argument("--model-f120", default="outputs/wind_prediction/gru_fino1_h240_f120_segmented_head_v1")
    parser.add_argument("--dataset-f180", default="data/processed/wind_ml_10min/ballast_decision_fino1_meteo_aux_h240_f180_v1")
    parser.add_argument("--model-f180", default="outputs/wind_prediction/gru_fino1_h240_f180_weighted_segmented_head_v1")
    parser.add_argument("--duration-s", type=float, default=7200.0)
    parser.add_argument("--out-dir", default="outputs/wind_prediction/far_horizon_weighting_guard10_audit_v1")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = _resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cases = pd.read_csv(_resolve(args.cases_csv))
    cfg = PlannerConfig(pressure_sign_multiplier=-1.0, envelope_use_discount=False)
    replay_f120 = Fino1ReplayDataset(dataset_dir=_resolve(args.dataset_f120), split="test")
    replay_f180 = Fino1ReplayDataset(dataset_dir=_resolve(args.dataset_f180), split="test")
    learned_f120 = ForecastModelAdapter(model_dir=_resolve(args.model_f120), dataset_dir=_resolve(args.dataset_f120), device="cpu")
    learned_f180 = ForecastModelAdapter(model_dir=_resolve(args.model_f180), dataset_dir=_resolve(args.dataset_f180), device="cpu")

    bucket_count = int(round(float(args.duration_s) / replay_f120.update_interval_s))
    rows: list[dict[str, Any]] = []
    for _, case in cases.iterrows():
        start_ts = _parse_ts(case["timestamp"])
        for bucket in range(bucket_count):
            ts = start_ts + timedelta(seconds=bucket * replay_f120.update_interval_s)
            sample120 = replay_f120.sample_for_history_end(ts)
            sample180 = replay_f180.sample_for_history_end(ts)
            if sample120 is None or sample180 is None:
                rows.append(
                    {
                        "case_id": case["case_id"],
                        "label": case["label"],
                        "bucket": bucket,
                        "timestamp": ts.strftime(TIMESTAMP_FMT),
                        "missing_sample": 1,
                    }
                )
                continue
            oracle_feat = _block_features(sample120.y_uv_raw[:12], cfg)
            pred120 = learned_f120.predict_window(sample120.x_window, timestamp=ts.strftime(TIMESTAMP_FMT)).wind_uv_raw
            pred180 = learned_f180.predict_window(sample180.x_window, timestamp=ts.strftime(TIMESTAMP_FMT)).wind_uv_raw
            learned120_feat = _block_features(pred120[:12], cfg)
            learned180_feat = _block_features(pred180[:12], cfg)
            row: dict[str, Any] = {
                "case_id": case["case_id"],
                "label": case["label"],
                "bucket": bucket,
                "timestamp": ts.strftime(TIMESTAMP_FMT),
                "missing_sample": 0,
            }
            row.update(_prefixed("oracle", oracle_feat))
            row.update(_prefixed("learned_f120", learned120_feat))
            row.update(_prefixed("learned_f180", learned180_feat))
            rows.append(row)

    bucket_df = pd.DataFrame(rows)
    bucket_df.to_csv(out_dir / "far_horizon_bucket_log.csv", index=False)
    valid = bucket_df[bucket_df["missing_sample"].eq(0)].copy()

    signal_names = [
        "hidden_relief_60_120",
        "hidden_intensification_60_120",
        "far_reversal_60_120",
        "far_direction_shift_60_120",
        "far_hint_any",
    ]
    confusion_rows = []
    for name in signal_names:
        for source in ("learned_f120", "learned_f180"):
            confusion_rows.append(
                {
                    "source": source,
                    **_confusion(valid, f"{source}_{name}", f"oracle_{name}"),
                }
            )
    confusion_df = pd.DataFrame(confusion_rows)
    confusion_df.to_csv(out_dir / "far_horizon_signal_confusion.csv", index=False)

    case_summary = (
        valid.groupby(["case_id", "label"])
        .agg(
            buckets=("bucket", "count"),
            oracle_far_hint_buckets=("oracle_far_hint_any", "sum"),
            oracle_relief_buckets=("oracle_hidden_relief_60_120", "sum"),
            oracle_intensification_buckets=("oracle_hidden_intensification_60_120", "sum"),
            oracle_shift_buckets=("oracle_far_direction_shift_60_120", "sum"),
            learned_f120_hint_buckets=("learned_f120_far_hint_any", "sum"),
            learned_f180_hint_buckets=("learned_f180_far_hint_any", "sum"),
            oracle_far_over_near_weighted_mean=("oracle_far_over_near_weighted", "mean"),
            learned_f120_far_over_near_weighted_mean=("learned_f120_far_over_near_weighted", "mean"),
            learned_f180_far_over_near_weighted_mean=("learned_f180_far_over_near_weighted", "mean"),
        )
        .reset_index()
    )
    case_summary.to_csv(out_dir / "far_horizon_case_summary.csv", index=False)

    overall = {
        "valid_buckets": int(len(valid)),
        "oracle_far_hint_rate": float(valid["oracle_far_hint_any"].mean()),
        "oracle_hidden_relief_rate": float(valid["oracle_hidden_relief_60_120"].mean()),
        "oracle_hidden_intensification_rate": float(valid["oracle_hidden_intensification_60_120"].mean()),
        "oracle_far_direction_shift_rate": float(valid["oracle_far_direction_shift_60_120"].mean()),
        "learned_f120_far_hint_rate": float(valid["learned_f120_far_hint_any"].mean()),
        "learned_f180_far_hint_rate": float(valid["learned_f180_far_hint_any"].mean()),
    }

    selected_conf = confusion_df[
        confusion_df["truth"].isin(
            [
                "oracle_hidden_relief_60_120",
                "oracle_hidden_intensification_60_120",
                "oracle_far_hint_any",
            ]
        )
    ][["source", "truth", "tp", "fp", "fn", "precision", "recall", "f1", "pred_rate", "truth_rate"]]
    lines = [
        "# Far Horizon Weighting Guard10 Audit",
        "",
        "This is observability-only. No controller or planner behavior was changed.",
        "",
        "## Weights Under Review",
        "",
        f"- near 0-60min weights: `{list(NEAR_WEIGHTS)}`",
        f"- proposed far 60-120min weights: `{list(FAR_WEIGHTS)}`",
        "",
        "## Overall",
        "",
        _markdown_table(pd.DataFrame([overall])),
        "",
        "## Learned-vs-Oracle Signal Agreement",
        "",
        _markdown_table(selected_conf),
        "",
        "## Case Summary",
        "",
        _markdown_table(case_summary),
        "",
        "## Output Files",
        "",
        "- `far_horizon_bucket_log.csv`",
        "- `far_horizon_case_summary.csv`",
        "- `far_horizon_signal_confusion.csv`",
    ]
    (out_dir / "far_horizon_weighting_summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()
