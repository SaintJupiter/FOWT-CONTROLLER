#!/usr/bin/env python3
"""Audit far-horizon event coverage and learnability for h240_f120.

This script is forecast-layer only. It does not run or patch the controller.

Outputs:
- far_horizon_event_coverage_train_val_guard10.csv
- far_horizon_event_coverage_summary.md
- far_hint_learnability_classifier_metrics.csv
- far_hint_guard10_predictions.csv
- far_hint_learnability_report.md
"""
from __future__ import annotations

import argparse
import gzip
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import (
    average_precision_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from wind_prediction.ballast_planner import PlannerConfig, norm_term, pressure_proxy_vec  # noqa: E402
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
FAR_BLOCK_NAMES = ("60_80", "80_100", "100_120")
LABELS = (
    "far_horizon_hint_any",
    "hidden_relief",
    "hidden_intensification",
    "direction_shift",
    "reversal",
    "sign_flip",
)


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


def _pressure_blocks(uv: np.ndarray, cfg: PlannerConfig) -> tuple[np.ndarray, list[np.ndarray]]:
    uv = np.asarray(uv, dtype=float)
    if uv.shape[0] < 12:
        raise ValueError(f"need 12 future steps, got {uv.shape}")
    norms: list[float] = []
    vecs: list[np.ndarray] = []
    for _name, start, end in BLOCKS_120:
        vec = np.asarray(pressure_proxy_vec(uv[start:end], cfg), dtype=float)
        vecs.append(vec)
        norms.append(norm_term(vec, cfg))
    return np.asarray(norms, dtype=float), vecs


def _event_features(uv: np.ndarray, cfg: PlannerConfig) -> dict[str, Any]:
    norms, vecs = _pressure_blocks(uv, cfg)
    near = norms[:3]
    far = norms[3:]
    near_max = float(np.max(near))
    near_last = float(near[-1])
    far_min = float(np.min(far))
    far_max = float(np.max(far))
    dir_shift = _direction_delta_deg(vecs[2], vecs[5])
    dot_60_120 = float(np.dot(vecs[2], vecs[5]))
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
    reversal = near_max >= 0.55 and far_max >= 0.55 and dot_60_120 < 0.0
    direction_shift = near_max >= 0.55 and far_max >= 0.55 and dir_shift >= 60.0

    # Pressure-curve sign flip: the far-horizon norm trend reverses relative to
    # the near-horizon trend. This is intentionally separate from vector reversal.
    near_trend = float(norms[2] - norms[0])
    far_trend = float(norms[5] - norms[2])
    sign_flip = (
        near_max >= 0.55
        and far_max >= 0.55
        and abs(near_trend) >= 0.15
        and abs(far_trend) >= 0.15
        and near_trend * far_trend < 0.0
    )

    relief_block = int(np.argmin(far))
    intens_block = int(np.argmax(far))
    pressure_change = far - near_last
    max_abs_change = float(np.max(np.abs(pressure_change)))
    return {
        "norm_0_20": float(norms[0]),
        "norm_20_40": float(norms[1]),
        "norm_40_60": float(norms[2]),
        "norm_60_80": float(norms[3]),
        "norm_80_100": float(norms[4]),
        "norm_100_120": float(norms[5]),
        "near_max": near_max,
        "near_last": near_last,
        "far_min": far_min,
        "far_max": far_max,
        "far_range": float(np.max(far) - np.min(far)),
        "near_range": float(np.max(near) - np.min(near)),
        "far_min_delta_vs_near_last": float(far_min - near_last),
        "far_max_delta_vs_near_last": float(far_max - near_last),
        "max_abs_far_delta_vs_near_last": max_abs_change,
        "dir_shift_60_to_120_deg": dir_shift,
        "dot_60_to_120": dot_60_120,
        "near_trend_norm": near_trend,
        "far_trend_norm": far_trend,
        "hidden_relief": int(hidden_relief),
        "hidden_intensification": int(hidden_intensification),
        "reversal": int(reversal),
        "direction_shift": int(direction_shift),
        "sign_flip": int(sign_flip),
        # Keep this aligned with the planner/audit far-horizon hint definition.
        # sign_flip is reported separately as an additional diagnostic label.
        "far_horizon_hint_any": int(hidden_relief or hidden_intensification or reversal or direction_shift),
        "relief_block": FAR_BLOCK_NAMES[relief_block],
        "intensification_block": FAR_BLOCK_NAMES[intens_block],
        "relief_delta_magnitude": float(max(0.0, near_last - far_min)),
        "intensification_delta_magnitude": float(max(0.0, far_max - near_max)),
        "sign_flip_magnitude": float(min(abs(near_trend), abs(far_trend))) if sign_flip else 0.0,
    }


def _load_split_arrays(dataset_dir: Path, split: str) -> tuple[np.ndarray, np.ndarray]:
    x = np.load(dataset_dir / f"X_{split}.npy", mmap_mode="r")
    y = np.load(dataset_dir / f"y_uv_raw_{split}.npy", mmap_mode="r")
    return x, y


def _load_split_index(dataset_dir: Path, split: str) -> pd.DataFrame:
    rows = []
    with gzip.open(dataset_dir / "sample_index.csv.gz", "rt", encoding="utf-8", newline="") as f:
        for chunk in pd.read_csv(f, chunksize=250_000):
            rows.append(chunk[chunk["split"] == split].copy())
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def _coverage_for_split(dataset_dir: Path, split: str, cfg: PlannerConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    _x, y = _load_split_arrays(dataset_dir, split)
    index = _load_split_index(dataset_dir, split)
    rows: list[dict[str, Any]] = []
    for i in range(y.shape[0]):
        feats = _event_features(np.asarray(y[i], dtype=float), cfg)
        row = {"source": split, "sample_idx": i}
        if i < len(index):
            row["history_end"] = index.loc[i, "history_end"]
        row.update(feats)
        rows.append(row)
    sample_df = pd.DataFrame(rows)
    return _summarize_coverage(sample_df, split), sample_df


def _guard10_samples(dataset_dir: Path, cases_csv: Path, cfg: PlannerConfig) -> pd.DataFrame:
    replay = Fino1ReplayDataset(dataset_dir=dataset_dir, split="test")
    cases = pd.read_csv(cases_csv)
    bucket_count = 12
    rows: list[dict[str, Any]] = []
    for _, case in cases.iterrows():
        start_ts = _parse_ts(str(case["timestamp"]))
        for bucket in range(bucket_count):
            ts = start_ts + timedelta(seconds=bucket * replay.update_interval_s)
            sample = replay.sample_for_history_end(ts)
            if sample is None:
                rows.append(
                    {
                        "source": "guard10",
                        "case_id": str(case["case_id"]),
                        "case_key": _case_key(str(case["case_id"])),
                        "bucket": bucket,
                        "history_end": ts.strftime(TIMESTAMP_FMT),
                        "missing_sample": 1,
                    }
                )
                continue
            feats = _event_features(sample.y_uv_raw, cfg)
            row = {
                "source": "guard10",
                "case_id": str(case["case_id"]),
                "case_key": _case_key(str(case["case_id"])),
                "bucket": bucket,
                "history_end": ts.strftime(TIMESTAMP_FMT),
                "missing_sample": 0,
            }
            row.update(feats)
            rows.append(row)
    return pd.DataFrame(rows)


def _case_key(case_id: str) -> str:
    # Strip numeric prefix used in guard10 csv while preserving descriptive name.
    text = str(case_id)
    parts = text.split("_", 1)
    return parts[1] if parts and parts[0].isdigit() and len(parts) > 1 else text


def _summarize_coverage(samples: pd.DataFrame, source: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    n = len(samples)
    for label in LABELS:
        if label not in samples.columns:
            continue
        mask = samples[label].astype(bool)
        subset = samples[mask]
        row: dict[str, Any] = {
            "source": source,
            "label": label,
            "sample_count": int(n),
            "positive_count": int(mask.sum()),
            "positive_rate": float(mask.mean()) if n else 0.0,
        }
        if len(subset):
            for metric in [
                "max_abs_far_delta_vs_near_last",
                "relief_delta_magnitude",
                "intensification_delta_magnitude",
                "sign_flip_magnitude",
                "far_range",
            ]:
                vals = subset[metric].astype(float)
                row[f"{metric}_p50"] = float(vals.quantile(0.50))
                row[f"{metric}_p90"] = float(vals.quantile(0.90))
                row[f"{metric}_max"] = float(vals.max())
            if label == "hidden_relief":
                counts = subset["relief_block"].value_counts().to_dict()
            elif label == "hidden_intensification":
                counts = subset["intensification_block"].value_counts().to_dict()
            else:
                # For general labels, report where the largest absolute far change occurs.
                far_cols = ["norm_60_80", "norm_80_100", "norm_100_120"]
                far = subset[far_cols].to_numpy(dtype=float)
                near_last = subset["near_last"].to_numpy(dtype=float).reshape(-1, 1)
                max_idx = np.argmax(np.abs(far - near_last), axis=1)
                counts = pd.Series([FAR_BLOCK_NAMES[int(i)] for i in max_idx]).value_counts().to_dict()
            for block in FAR_BLOCK_NAMES:
                row[f"block_{block}_count"] = int(counts.get(block, 0))
        else:
            for metric in [
                "max_abs_far_delta_vs_near_last",
                "relief_delta_magnitude",
                "intensification_delta_magnitude",
                "sign_flip_magnitude",
                "far_range",
            ]:
                row[f"{metric}_p50"] = np.nan
                row[f"{metric}_p90"] = np.nan
                row[f"{metric}_max"] = np.nan
            for block in FAR_BLOCK_NAMES:
                row[f"block_{block}_count"] = 0
        rows.append(row)
    return pd.DataFrame(rows)


def _feature_matrix(x: np.ndarray, sample_count: int | None = None) -> np.ndarray:
    n = x.shape[0] if sample_count is None else min(int(sample_count), x.shape[0])
    arr = np.asarray(x[:n], dtype=np.float32)
    last = arr[:, -1, :]
    mean = arr.mean(axis=1)
    std = arr.std(axis=1)
    delta = arr[:, -1, :] - arr[:, 0, :]
    # Compact but stronger than a raw last-row logistic baseline.
    return np.concatenate([last, mean, std, delta], axis=1).astype(np.float32)


@dataclass
class ClassifierResult:
    metrics: dict[str, Any]
    model: SGDClassifier | None
    scaler: StandardScaler | None


def _fit_classifier(
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray,
    val_y: np.ndarray,
    label: str,
) -> ClassifierResult:
    pos = int(train_y.sum())
    neg = int(len(train_y) - pos)
    if pos == 0 or int(val_y.sum()) == 0:
        return ClassifierResult(
            metrics={
                "label": label,
                "train_positive_count": pos,
                "validation_positive_count": int(val_y.sum()),
                "precision": np.nan,
                "recall": np.nan,
                "f1": np.nan,
                "auc": np.nan,
                "average_precision": np.nan,
                "threshold": np.nan,
                "note": "no_positive_in_train_or_validation",
            },
            model=None,
            scaler=None,
        )
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_x)
    val_scaled = scaler.transform(val_x)
    class_weight = {0: 1.0, 1: max(1.0, neg / max(pos, 1))}
    clf = SGDClassifier(
        loss="log_loss",
        penalty="elasticnet",
        alpha=1e-4,
        l1_ratio=0.05,
        class_weight=class_weight,
        max_iter=1000,
        tol=1e-4,
        random_state=17,
        n_jobs=1,
    )
    clf.fit(train_scaled, train_y)
    scores = clf.predict_proba(val_scaled)[:, 1]
    auc = float(roc_auc_score(val_y, scores)) if len(np.unique(val_y)) > 1 else np.nan
    ap = float(average_precision_score(val_y, scores)) if len(np.unique(val_y)) > 1 else np.nan
    thresholds = np.unique(np.quantile(scores, np.linspace(0.01, 0.99, 99)))
    best: dict[str, Any] | None = None
    for threshold in thresholds:
        pred = scores >= threshold
        precision, recall, f1, _ = precision_recall_fscore_support(
            val_y,
            pred.astype(int),
            average="binary",
            zero_division=0,
        )
        item = {
            "threshold": float(threshold),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
        }
        if best is None or item["f1"] > best["f1"]:
            best = item
    assert best is not None
    metrics = {
        "label": label,
        "train_positive_count": pos,
        "train_positive_rate": float(pos / max(len(train_y), 1)),
        "validation_positive_count": int(val_y.sum()),
        "validation_positive_rate": float(val_y.mean()),
        "precision": best["precision"],
        "recall": best["recall"],
        "f1": best["f1"],
        "auc": auc,
        "average_precision": ap,
        "threshold": best["threshold"],
        "note": "",
    }
    return ClassifierResult(metrics=metrics, model=clf, scaler=scaler)


def _guard10_feature_rows(dataset_dir: Path, cases_csv: Path) -> pd.DataFrame:
    replay = Fino1ReplayDataset(dataset_dir=dataset_dir, split="test")
    cases = pd.read_csv(cases_csv)
    rows: list[dict[str, Any]] = []
    for _, case in cases.iterrows():
        start_ts = _parse_ts(str(case["timestamp"]))
        for bucket in range(12):
            ts = start_ts + timedelta(seconds=bucket * replay.update_interval_s)
            sample = replay.sample_for_history_end(ts)
            if sample is None:
                continue
            rows.append(
                {
                    "case_id": str(case["case_id"]),
                    "case_key": _case_key(str(case["case_id"])),
                    "bucket": bucket,
                    "history_end": ts.strftime(TIMESTAMP_FMT),
                    "x_window": sample.x_window,
                }
            )
    return pd.DataFrame(rows)


def _load_guard10_oracle_key(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["case_key", "bucket"])
    d = pd.read_csv(path)
    d["case_key"] = d["case_id"].map(_case_key)
    d["bucket"] = d["bucket"].astype(int)
    return d


def _markdown_table(df: pd.DataFrame, max_rows: int | None = None) -> str:
    if df.empty:
        return "_none_"
    view = df if max_rows is None else df.head(max_rows)
    lines = [
        "| " + " | ".join(view.columns.astype(str)) + " |",
        "| " + " | ".join(["---"] * len(view.columns)) + " |",
    ]
    for _, row in view.iterrows():
        vals = []
        for value in row:
            if isinstance(value, float):
                vals.append(f"{value:.4f}")
            else:
                vals.append(str(value))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-dir",
        default="data/processed/wind_ml_10min/ballast_decision_fino1_meteo_aux_h240_f120_v1",
    )
    parser.add_argument("--cases-csv", default="outputs/wind_prediction/broad_optimization_work/guard10_broad_cases.csv")
    parser.add_argument(
        "--oracle-key",
        default="outputs/wind_prediction/h240_f120_oracle_learned_attribution_v1/oracle_learned_action_diff_buckets.csv",
    )
    parser.add_argument("--out-dir", default="outputs/wind_prediction/far_horizon_learnability_audit_v1")
    parser.add_argument("--train-subsample", type=int, default=250000)
    parser.add_argument("--validation-subsample", type=int, default=100000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_dir = _resolve(args.dataset_dir)
    cases_csv = _resolve(args.cases_csv)
    out_dir = _resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = PlannerConfig(pressure_sign_multiplier=-1.0, envelope_use_discount=False)

    coverage_tables: list[pd.DataFrame] = []
    sample_tables: dict[str, pd.DataFrame] = {}
    for split in ("train", "validation"):
        coverage, samples = _coverage_for_split(dataset_dir, split, cfg)
        coverage_tables.append(coverage)
        sample_tables[split] = samples
    guard10_samples = _guard10_samples(dataset_dir, cases_csv, cfg)
    sample_tables["guard10"] = guard10_samples
    coverage_tables.append(_summarize_coverage(guard10_samples[guard10_samples["missing_sample"] == 0], "guard10"))
    coverage_summary = pd.concat(coverage_tables, ignore_index=True)
    coverage_summary.to_csv(out_dir / "far_horizon_event_coverage_train_val_guard10.csv", index=False)

    # Classifier data: use chronological train/validation split from existing dataset.
    x_train, _ = _load_split_arrays(dataset_dir, "train")
    x_val, _ = _load_split_arrays(dataset_dir, "validation")
    n_train = min(args.train_subsample, x_train.shape[0])
    n_val = min(args.validation_subsample, x_val.shape[0])
    train_x = _feature_matrix(x_train, n_train)
    val_x = _feature_matrix(x_val, n_val)
    train_samples = sample_tables["train"].iloc[:n_train].reset_index(drop=True)
    val_samples = sample_tables["validation"].iloc[:n_val].reset_index(drop=True)

    classifier_results: dict[str, ClassifierResult] = {}
    metric_rows: list[dict[str, Any]] = []
    for label in LABELS:
        result = _fit_classifier(
            train_x,
            train_samples[label].astype(int).to_numpy(),
            val_x,
            val_samples[label].astype(int).to_numpy(),
            label,
        )
        classifier_results[label] = result
        metric_rows.append(result.metrics)
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(out_dir / "far_hint_learnability_classifier_metrics.csv", index=False)

    # Guard10 external evaluation, including oracle-action-diff bucket subset.
    guard10_features = _guard10_feature_rows(dataset_dir, cases_csv)
    guard_x = _feature_matrix(np.stack(guard10_features["x_window"].to_list(), axis=0))
    oracle_key = _load_guard10_oracle_key(_resolve(args.oracle_key))
    key_pairs = set(zip(oracle_key["case_key"], oracle_key["bucket"].astype(int)))
    guard_rows: list[dict[str, Any]] = []
    guard_labels = guard10_samples[guard10_samples["missing_sample"] == 0].copy().reset_index(drop=True)
    for label in LABELS:
        result = classifier_results[label]
        if result.model is None or result.scaler is None:
            scores = np.full(len(guard10_features), np.nan)
            pred = np.zeros(len(guard10_features), dtype=bool)
            threshold = np.nan
        else:
            scores = result.model.predict_proba(result.scaler.transform(guard_x))[:, 1]
            threshold = float(result.metrics["threshold"])
            pred = scores >= threshold
        for i, base_row in guard10_features.iterrows():
            truth = int(guard_labels.loc[i, label])
            pair = (str(base_row["case_key"]), int(base_row["bucket"]))
            guard_rows.append(
                {
                    "label": label,
                    "case_id": base_row["case_id"],
                    "case_key": base_row["case_key"],
                    "bucket": int(base_row["bucket"]),
                    "history_end": base_row["history_end"],
                    "truth": truth,
                    "score": float(scores[i]) if not np.isnan(scores[i]) else np.nan,
                    "pred": int(pred[i]),
                    "threshold": threshold,
                    "is_oracle_action_diff_bucket": int(pair in key_pairs),
                    "is_oracle_far_hint_bucket": int(
                        pair in set(
                            zip(
                                guard_labels.loc[guard_labels["far_horizon_hint_any"].astype(bool), "case_key"],
                                guard_labels.loc[guard_labels["far_horizon_hint_any"].astype(bool), "bucket"].astype(int),
                            )
                        )
                    ),
                }
            )
    guard_pred = pd.DataFrame(guard_rows)
    guard_pred.to_csv(out_dir / "far_hint_guard10_predictions.csv", index=False)

    guard_eval_rows: list[dict[str, Any]] = []
    for label in LABELS:
        sub = guard_pred[guard_pred["label"] == label]
        truth = sub["truth"].astype(bool)
        pred = sub["pred"].astype(bool)
        tp = int((truth & pred).sum())
        fp = int((~truth & pred).sum())
        fn = int((truth & ~pred).sum())
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        key = sub[sub["is_oracle_action_diff_bucket"].astype(bool)]
        hint_key = sub[sub["is_oracle_far_hint_bucket"].astype(bool)]
        guard_eval_rows.append(
            {
                "label": label,
                "guard10_positive_count": int(truth.sum()),
                "guard10_pred_count": int(pred.sum()),
                "guard10_tp": tp,
                "guard10_fp": fp,
                "guard10_fn": fn,
                "guard10_precision": precision,
                "guard10_recall": recall,
                "oracle_action_diff_bucket_count": int(len(key)),
                "oracle_action_diff_pred_count": int(key["pred"].sum()),
                "oracle_action_diff_true_positive_count": int((key["truth"].astype(bool) & key["pred"].astype(bool)).sum()),
                "oracle_far_hint_bucket_count": int(len(hint_key)),
                "oracle_far_hint_pred_count": int(hint_key["pred"].sum()),
            }
        )
    guard_eval = pd.DataFrame(guard_eval_rows)
    guard_eval.to_csv(out_dir / "far_hint_guard10_classifier_eval.csv", index=False)

    def _metric(label: str, col: str, default: float = float("nan")) -> float:
        row = metrics[metrics["label"] == label]
        if row.empty or col not in row.columns:
            return default
        return float(row.iloc[0][col])

    def _coverage(source: str, label: str, col: str) -> float:
        row = coverage_summary[(coverage_summary["source"] == source) & (coverage_summary["label"] == label)]
        if row.empty:
            return float("nan")
        return float(row.iloc[0][col])

    def _guard(label: str, col: str) -> float:
        row = guard_eval[guard_eval["label"] == label]
        if row.empty:
            return float("nan")
        return float(row.iloc[0][col])

    conclusion_lines = [
        "## Conclusions",
        "",
        "1. Event coverage is uneven, not absent.",
        (
            f"Full train positives: far_hint={int(_coverage('train', 'far_horizon_hint_any', 'positive_count'))} "
            f"({_coverage('train', 'far_horizon_hint_any', 'positive_rate'):.2%}), "
            f"hidden_relief={int(_coverage('train', 'hidden_relief', 'positive_count'))} "
            f"({_coverage('train', 'hidden_relief', 'positive_rate'):.2%}), "
            f"hidden_intensification={int(_coverage('train', 'hidden_intensification', 'positive_count'))} "
            f"({_coverage('train', 'hidden_intensification', 'positive_rate'):.2%}). "
            "Relief has enough samples for an auxiliary objective; intensification/direction/reversal are rare enough "
            "that ordinary regression or unbalanced losses are unlikely to preserve them."
        ),
        "",
        "2. A lightweight classifier can recover a coarse far-hint signal from the existing 240min history, but with low precision.",
        (
            f"Validation far_hint AUC={_metric('far_horizon_hint_any', 'auc'):.3f}, "
            f"recall={_metric('far_horizon_hint_any', 'recall'):.3f}, "
            f"precision={_metric('far_horizon_hint_any', 'precision'):.3f}; "
            f"on guard10 it hits {_guard('far_horizon_hint_any', 'guard10_tp'):.0f}/"
            f"{_guard('far_horizon_hint_any', 'guard10_positive_count'):.0f} true far-hint buckets, "
            f"including {_guard('far_horizon_hint_any', 'oracle_far_hint_pred_count'):.0f}/"
            f"{_guard('far_horizon_hint_any', 'oracle_far_hint_bucket_count'):.0f} oracle far-hint buckets."
        ),
        "",
        "3. Exact subtypes are the hard part.",
        (
            f"Hidden relief guard10 recall={_guard('hidden_relief', 'guard10_recall'):.3f}; "
            f"hidden intensification guard10 recall={_guard('hidden_intensification', 'guard10_recall'):.3f} "
            "but validation recall is low because the class is very rare. Direction shift and reversal are too sparse "
            "for the current simple classifier to be reliable."
        ),
        "",
        "4. This supports path C more than pure path A or B.",
        "The signal is not completely missing from the 240min input, and it is not merely a controller-use problem. "
        "The current LSTM pressure regression failed to express a signal that a simple auxiliary classifier can partly rank. "
        "The next forecast-layer step should therefore be multi-task / auxiliary-head training with event rebalancing, "
        "not controller tuning and not another plain regression-loss tweak.",
    ]

    coverage_focus = coverage_summary[
        coverage_summary["label"].isin(["far_horizon_hint_any", "hidden_relief", "hidden_intensification"])
    ][["source", "label", "sample_count", "positive_count", "positive_rate"]]
    report_lines = [
        "# Far-Horizon Learnability Audit",
        "",
        "Scope: h240_f120 forecast-layer only. No controller changes, no closed-loop model training.",
        "",
        "Classifier note: validation metrics below use a chronological training prefix subsample "
        f"(train n={n_train}, validation n={n_val}) for speed; event coverage above uses full split counts.",
        "",
        *conclusion_lines,
        "",
        "## Event Coverage",
        _markdown_table(coverage_focus),
        "",
        "Full coverage table: `far_horizon_event_coverage_train_val_guard10.csv`.",
        "",
        "## Lightweight Classifier Validation Metrics",
        _markdown_table(
            metrics[
                [
                    "label",
                    "train_positive_count",
                    "validation_positive_count",
                    "precision",
                    "recall",
                    "f1",
                    "auc",
                    "average_precision",
                ]
            ]
        ),
        "",
        "## Guard10 External Evaluation",
        _markdown_table(
            guard_eval[
                [
                    "label",
                    "guard10_positive_count",
                    "guard10_pred_count",
                    "guard10_tp",
                    "guard10_fp",
                    "guard10_fn",
                    "guard10_precision",
                    "guard10_recall",
                    "oracle_action_diff_pred_count",
                    "oracle_far_hint_pred_count",
                ]
            ]
        ),
        "",
        "## Decision Logic",
        "",
        "- If train/validation positives are scarce, ordinary regression losses are unlikely to learn the event without resampling or auxiliary labels.",
        "- If the lightweight classifier has weak validation recall/AUC, the current 240min history features may not contain enough signal for reliable 60-120min hints.",
        "- If the classifier is strong but the LSTM regression is weak, the next step is representation/loss redesign with auxiliary heads rather than controller patches.",
        "",
        "## Outputs",
        "- `far_horizon_event_coverage_train_val_guard10.csv`",
        "- `far_hint_learnability_classifier_metrics.csv`",
        "- `far_hint_guard10_predictions.csv`",
        "- `far_hint_guard10_classifier_eval.csv`",
    ]
    (out_dir / "far_horizon_event_coverage_summary.md").write_text("\n".join(report_lines), encoding="utf-8")
    (out_dir / "far_hint_learnability_report.md").write_text("\n".join(report_lines), encoding="utf-8")
    print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()
