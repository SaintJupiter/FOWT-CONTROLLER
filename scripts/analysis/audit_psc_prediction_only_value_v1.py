#!/usr/bin/env python3
"""Prediction-only audit for PSC signals, separated from deadband execution.

This answers: does the learned PSC model contain future information beyond a
current-only/persistence baseline before any controller or deadband is applied?
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


PSC_TO_SOURCE_EVENTS = {
    "event_ramp_onset": "speed_ramp_ge_3ms",
    "event_sustained_high": "future_speed_ge_train_p95",
    "event_attention_any": "ballast_attention_event",
    "event_attention_0_20m": "attention_event_0_20m",
    "event_attention_20_40m": "attention_event_20_40m",
    "event_attention_40_60m": "attention_event_40_60m",
    "event_attention_60_80m": "attention_event_60_80m",
    "event_attention_80_100m": "attention_event_80_100m",
    "event_attention_100_120m": "attention_event_100_120m",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-dataset-dir",
        type=Path,
        default=Path("data/processed/wind_ml_10min/ballast_decision_fino1_meteo_aux_h240_f120_v1"),
    )
    parser.add_argument(
        "--label-dir",
        type=Path,
        default=Path("outputs/wind_prediction/psc_decision_model_v1/dataset"),
    )
    parser.add_argument(
        "--signals-dir",
        type=Path,
        default=Path("outputs/wind_prediction/psc_decision_model_v1/decision_signals_gru_adapter_calibrated_v1"),
    )
    parser.add_argument(
        "--replay-label-dir",
        type=Path,
        default=Path("outputs/wind_prediction/psc_decision_model_v1/replay_value_labels_v1"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/wind_prediction/psc_prediction_only_audit_v1"),
    )
    parser.add_argument("--split", choices=["validation", "test"], default="test")
    return parser.parse_args()


def binary_metrics(y: np.ndarray, score: np.ndarray, threshold: float) -> dict[str, float]:
    pred = score >= threshold
    truth = y >= 0.5
    tp = int((pred & truth).sum())
    fp = int((pred & ~truth).sum())
    fn = int((~pred & truth).sum())
    tn = int((~pred & ~truth).sum())
    return {
        "threshold": float(threshold),
        "base_rate": float(truth.mean()),
        "pred_rate": float(pred.mean()),
        "precision": tp / max(tp + fp, 1),
        "recall": tp / max(tp + fn, 1),
        "specificity": tn / max(tn + fp, 1),
        "accuracy": (tp + tn) / max(tp + fp + fn + tn, 1),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def roc_auc(y: np.ndarray, score: np.ndarray) -> float:
    y = (y >= 0.5).astype(np.int8)
    score = np.asarray(score, dtype=np.float64)
    pos = int(y.sum())
    neg = int((1 - y).sum())
    if pos == 0 or neg == 0:
        return float("nan")
    order = np.argsort(score)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(score) + 1)
    # Tie correction by average rank.
    sorted_score = score[order]
    start = 0
    while start < len(score):
        end = start + 1
        while end < len(score) and sorted_score[end] == sorted_score[start]:
            end += 1
        if end - start > 1:
            ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    pos_rank_sum = float(ranks[y == 1].sum())
    return (pos_rank_sum - pos * (pos + 1) / 2.0) / (pos * neg)


def average_precision(y: np.ndarray, score: np.ndarray) -> float:
    y = (y >= 0.5).astype(np.int8)
    pos = int(y.sum())
    if pos == 0:
        return float("nan")
    order = np.argsort(-np.asarray(score, dtype=np.float64))
    sorted_y = y[order]
    tp = np.cumsum(sorted_y)
    precision = tp / np.arange(1, len(y) + 1)
    return float((precision * sorted_y).sum() / pos)


def top_lift(y: np.ndarray, score: np.ndarray, frac: float = 0.10) -> float:
    n = max(1, int(round(len(y) * frac)))
    idx = np.argsort(score)[-n:]
    base = float(np.mean(y))
    if base <= 1e-12:
        return float("nan")
    return float(np.mean(y[idx]) / base)


def current_only_event_scores(sample_index: pd.DataFrame, source_dataset_dir: Path, split: str) -> pd.DataFrame:
    x = np.load(source_dataset_dir / f"X_{split}.npy", mmap_mode="r")
    metadata = json.loads((source_dataset_dir / "metadata.json").read_text(encoding="utf-8"))
    feature_cols = metadata["feature_columns"]
    idx = {name: feature_cols.index(name) for name in feature_cols}
    last = np.asarray(x[:, -1, :], dtype=np.float32)

    current_speed = last[:, idx["wind_speed_ms"]]
    speed_delta = np.abs(last[:, idx["wind_speed_delta_10m"]])
    direction_delta = np.abs(last[:, idx["wind_dir_delta_10m_deg"]])
    current_std_60m = np.abs(last[:, idx["wind_speed_std_60m"]])
    future_speed_p95 = float(metadata["event_thresholds"]["future_speed_train_p95_ms"])
    vector_p90 = float(metadata["event_thresholds"]["vector_change_train_p90_ms"])

    # Current-only/persistence scores: no future shape is available, so these
    # are deliberately weak current-state proxies.
    out = pd.DataFrame({"split_row": np.arange(len(sample_index), dtype=np.int64)})
    out["current_score_event_ramp_onset"] = speed_delta + 0.25 * current_std_60m
    out["current_score_event_sustained_high"] = current_speed - future_speed_p95
    out["current_score_event_attention_any"] = np.maximum.reduce(
        [
            out["current_score_event_ramp_onset"].to_numpy(),
            direction_delta / 45.0,
            np.maximum(0.0, current_speed - future_speed_p95),
        ]
    )
    out["current_score_event_reversal_signflip"] = direction_delta
    out["current_score_event_reintensification"] = np.maximum(0.0, current_speed - future_speed_p95) + speed_delta
    out["current_score_event_transient_decay"] = current_std_60m - speed_delta
    out["current_score_event_vector_change"] = np.hypot(last[:, idx["wind_u_delta_10m"]], last[:, idx["wind_v_delta_10m"]]) - vector_p90
    for name in [
        "event_attention_0_20m",
        "event_attention_20_40m",
        "event_attention_40_60m",
        "event_attention_60_80m",
        "event_attention_80_100m",
        "event_attention_100_120m",
    ]:
        out[f"current_score_{name}"] = out["current_score_event_attention_any"]
    return out


def event_audit(signals: pd.DataFrame, sample_index: pd.DataFrame, current_scores: pd.DataFrame) -> pd.DataFrame:
    rows = []
    merged = signals.merge(current_scores, on="split_row", how="left")
    source_event_map = {
        **PSC_TO_SOURCE_EVENTS,
        "event_reversal_signflip": "direction_shift_ge_45deg",
        "event_reintensification": "vector_change_ge_train_p90",
        "event_transient_decay": "ballast_attention_event",
    }
    for psc_event, truth_col in source_event_map.items():
        if truth_col not in sample_index.columns:
            continue
        truth = sample_index[truth_col].to_numpy(dtype=np.float32)
        learned_score = merged[f"psc_prob_{psc_event}"].to_numpy(dtype=np.float32)
        current_score = merged[f"current_score_{psc_event}"].to_numpy(dtype=np.float32)
        row = {
            "event": psc_event,
            "truth_column": truth_col,
            "base_rate": float(truth.mean()),
            "learned_auc": roc_auc(truth, learned_score),
            "current_only_auc": roc_auc(truth, current_score),
            "learned_ap": average_precision(truth, learned_score),
            "current_only_ap": average_precision(truth, current_score),
            "learned_top10_lift": top_lift(truth, learned_score, 0.10),
            "current_only_top10_lift": top_lift(truth, current_score, 0.10),
        }
        if psc_event == "event_attention_any":
            row.update({f"learned_at_0p5_{k}": v for k, v in binary_metrics(truth, learned_score, 0.5).items()})
        rows.append(row)
    return pd.DataFrame(rows)


def regime_audit(signals: pd.DataFrame, label_dir: Path, split: str) -> pd.DataFrame:
    y_regime = np.load(label_dir / f"y_regime_{split}.npy", mmap_mode="r")
    manifest = json.loads((label_dir / "psc_decision_dataset_manifest.json").read_text(encoding="utf-8"))
    classes = manifest["regime_classes"]
    rows = []
    pred = signals["psc_regime_argmax"].to_numpy()
    for idx, cls in enumerate(classes):
        truth = np.asarray(y_regime[:] == idx, dtype=np.float32)
        score = signals[f"psc_regime_prob_{cls}"].to_numpy(dtype=np.float32)
        rows.append(
            {
                "regime": cls,
                "base_rate": float(truth.mean()),
                "learned_auc": roc_auc(truth, score),
                "learned_ap": average_precision(truth, score),
                "learned_top10_lift": top_lift(truth, score, 0.10),
                "argmax_precision": float(((pred == cls) & (truth > 0.5)).sum() / max((pred == cls).sum(), 1)),
                "argmax_recall": float(((pred == cls) & (truth > 0.5)).sum() / max((truth > 0.5).sum(), 1)),
            }
        )
    return pd.DataFrame(rows)


def scalar_audit(signals: pd.DataFrame, label_dir: Path, split: str) -> pd.DataFrame:
    diagnostics = pd.read_csv(label_dir / f"label_diagnostics_{split}.csv")
    rows = []
    scalar_cols = [
        "time_to_attention_min",
        "event_strength_ms",
        "max_direction_shift_deg",
        "min_vector_cosine",
        "peak_to_late_drop_ms",
        "closed_roundtrip_proxy",
    ]
    for col in scalar_cols:
        truth = diagnostics[col].to_numpy(dtype=np.float32)
        pred = signals[f"psc_scalar_{col}"].to_numpy(dtype=np.float32)
        mae = float(np.mean(np.abs(pred - truth)))
        corr = float(np.corrcoef(pred, truth)[0, 1]) if np.std(pred) > 1e-8 and np.std(truth) > 1e-8 else float("nan")
        rows.append({"scalar": col, "learned_mae": mae, "learned_pearson": corr, "truth_mean": float(truth.mean())})
    return pd.DataFrame(rows)


def opportunity_audit(signals: pd.DataFrame, replay_label_dir: Path, split: str) -> pd.DataFrame:
    rows_path = replay_label_dir / "replay_value_rows.csv"
    if not rows_path.exists():
        return pd.DataFrame()
    rows = pd.read_csv(rows_path)
    rows = rows[rows["split"].eq(split)].copy()
    if rows.empty:
        return pd.DataFrame()
    merged = rows.merge(signals, on=["split", "split_row"], how="left")
    out = []
    for col in ["closed_pump_waste_m3", "deadband_opportunity_m3", "relax_event_exposure_risk_s"]:
        truth = merged[col].to_numpy(dtype=np.float32)
        pred = merged[f"psc_value_{col}"].to_numpy(dtype=np.float32)
        out.append(
            {
                "target": col,
                "rows": int(len(merged)),
                "mae": float(np.mean(np.abs(pred - truth))),
                "pearson": float(np.corrcoef(pred, truth)[0, 1]) if np.std(pred) > 1e-8 and np.std(truth) > 1e-8 else float("nan"),
                "top20_lift": top_lift(truth, pred, 0.20),
            }
        )
    return pd.DataFrame(out)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    signals = pd.read_csv(args.signals_dir / f"psc_decision_signals_{args.split}.csv.gz")
    sample_index = pd.read_csv(args.source_dataset_dir / "sample_index.csv.gz")
    sample_index = sample_index[sample_index["split"].eq(args.split)].reset_index(drop=True)
    current_scores = current_only_event_scores(sample_index, args.source_dataset_dir, args.split)

    event = event_audit(signals, sample_index, current_scores)
    regime = regime_audit(signals, args.label_dir, args.split)
    scalar = scalar_audit(signals, args.label_dir, args.split)
    opportunity = opportunity_audit(signals, args.replay_label_dir, args.split)

    event.to_csv(args.output_dir / f"prediction_only_event_audit_{args.split}.csv", index=False)
    regime.to_csv(args.output_dir / f"prediction_only_regime_audit_{args.split}.csv", index=False)
    scalar.to_csv(args.output_dir / f"prediction_only_scalar_audit_{args.split}.csv", index=False)
    if not opportunity.empty:
        opportunity.to_csv(args.output_dir / f"prediction_only_opportunity_audit_{args.split}.csv", index=False)

    summary = {
        "split": args.split,
        "rows": int(len(signals)),
        "event_mean_learned_auc": float(event["learned_auc"].mean()),
        "event_mean_current_only_auc": float(event["current_only_auc"].mean()),
        "attention_any": event[event["event"].eq("event_attention_any")].to_dict(orient="records"),
        "regime_mean_learned_auc": float(regime["learned_auc"].mean()),
        "opportunity": opportunity.to_dict(orient="records") if not opportunity.empty else [],
        "outputs": {
            "event": str(args.output_dir / f"prediction_only_event_audit_{args.split}.csv"),
            "regime": str(args.output_dir / f"prediction_only_regime_audit_{args.split}.csv"),
            "scalar": str(args.output_dir / f"prediction_only_scalar_audit_{args.split}.csv"),
            "opportunity": str(args.output_dir / f"prediction_only_opportunity_audit_{args.split}.csv") if not opportunity.empty else None,
        },
    }
    (args.output_dir / f"prediction_only_summary_{args.split}.json").write_text(
        json.dumps(summary, indent=2, default=str),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, default=str))
    print("\nEvent audit")
    print(event.to_string(index=False))
    print("\nRegime audit")
    print(regime.to_string(index=False))
    if not opportunity.empty:
        print("\nOpportunity audit")
        print(opportunity.to_string(index=False))


if __name__ == "__main__":
    main()
