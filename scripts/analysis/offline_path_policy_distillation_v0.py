#!/usr/bin/env python3
"""Offline active_small -> active_medium path-policy distillation v0.

This script does not run closed-loop simulations and does not modify the
controller. It builds a narrow offline classification dataset from existing
forced-prefix replay artifacts and tests whether a lightweight classifier can
identify when a learned active_small decision should be upgraded to
active_medium.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[2]
PATH_DATASET = REPO_ROOT / "outputs/wind_prediction/path_policy_distillation_dataset_v1"
FORCED_DIR = REPO_ROOT / "outputs/wind_prediction/forced_prefix_replay_v1"
LEARNED_RUN = REPO_ROOT / "outputs/wind_prediction/pp_h240_f120_near_block_eventbalanced_v2_guard10_learned_v1"
ORACLE_RUN = REPO_ROOT / "outputs/wind_prediction/pp_h240_f120_oracle_farlog_guard10_v1"
OUT_DIR = REPO_ROOT / "outputs/wind_prediction/offline_path_policy_distillation_v0"

RUN_RE = re.compile(
    r"^(?P<forecast>learned|oracle)_forecast_"
    r"(?P<prefix_source>learned|oracle)_prefix_len(?P<length>\d+)_"
    r"(?P<mode>final_action|raw_action|target_update)$"
)


def _safe_float(value: Any, default: float = np.nan) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if np.isfinite(out) else default


def _parse_action_sequence(seq: Any) -> dict[int, str]:
    out: dict[int, str] = {}
    text = "" if pd.isna(seq) else str(seq)
    for part in text.split(","):
        if ":" not in part:
            continue
        left, right = part.split(":", 1)
        try:
            bucket = int(left.strip())
        except ValueError:
            continue
        out[bucket] = right.strip()
    return out


def _parse_run_label(label: str) -> dict[str, Any]:
    match = RUN_RE.match(str(label))
    if not match:
        return {"forecast_source": "", "prefix_source": "", "prefix_length": 0, "forced_mode": ""}
    d = match.groupdict()
    return {
        "forecast_source": d["forecast"],
        "prefix_source": d["prefix_source"],
        "prefix_length": int(d["length"]),
        "forced_mode": d["mode"],
    }


def _case_file(run_dir: Path, case_id: str, kind: str) -> Path | None:
    sub = "planner_logs" if kind == "log" else "timeseries"
    suffix = "_planner_log.csv" if kind == "log" else "_timeseries.csv"
    return next((run_dir / sub).glob(f"*{case_id}*{suffix}"), None)


def _load_logs(run_dir: Path) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    log_dir = run_dir / "planner_logs"
    if not log_dir.exists():
        return out
    for path in sorted(log_dir.glob("*_planner_log.csv")):
        case_id = _case_from_filename(path.name)
        df = pd.read_csv(path, low_memory=False)
        if "bucket" in df.columns:
            df["bucket"] = pd.to_numeric(df["bucket"], errors="coerce").fillna(-1).astype(int)
        out[case_id] = df
    return out


def _case_from_filename(name: str) -> str:
    known = [
        "fr_relief_01",
        "fr_relief_09",
        "sf_holdout_02",
        "b_decay_strong",
        "b_signflip_fallback",
        "b_high_pressure_event",
        "b_residual_high",
        "lowrisk_quiet",
        "lowrisk_clean",
        "lowrisk_random_03",
    ]
    for case in known:
        if case in str(name):
            return case
    return str(name)


def _row_for_bucket(logs: dict[str, pd.DataFrame], case_id: str, bucket: int) -> pd.Series:
    df = logs.get(case_id)
    if df is None or df.empty or "bucket" not in df.columns:
        return pd.Series(dtype=object)
    hit = df[df["bucket"].eq(int(bucket))]
    return hit.iloc[0] if not hit.empty else pd.Series(dtype=object)


def _prev_action(logs: dict[str, pd.DataFrame], case_id: str, bucket: int) -> str:
    row = _row_for_bucket(logs, case_id, int(bucket) - 1)
    return str(row.get("first_action", "none")) if not row.empty else "none"


def _count_recent_action(logs: dict[str, pd.DataFrame], case_id: str, bucket: int, action: str, window: int = 3) -> int:
    df = logs.get(case_id)
    if df is None or df.empty or "bucket" not in df.columns:
        return 0
    lo = max(0, int(bucket) - int(window))
    sub = df[df["bucket"].between(lo, int(bucket) - 1)]
    return int(sub.get("first_action", pd.Series(dtype=str)).astype(str).eq(action).sum())


def _feature_row(
    *,
    sample: pd.Series,
    bucket: int,
    learned_action: str,
    oracle_action: str,
    learned_logs: dict[str, pd.DataFrame],
    oracle_logs: dict[str, pd.DataFrame],
) -> dict[str, Any]:
    case_id = str(sample["case_id"])
    row = _row_for_bucket(learned_logs, case_id, bucket)
    orow = _row_for_bucket(oracle_logs, case_id, bucket)
    b0 = _safe_float(row.get("raw_pressure_block0_norm"), _safe_float(row.get("pressure_block0_norm"), 0.0))
    b1 = _safe_float(row.get("raw_pressure_block1_norm"), _safe_float(row.get("pressure_block1_norm"), 0.0))
    b2 = _safe_float(row.get("raw_pressure_block2_norm"), _safe_float(row.get("pressure_block2_norm"), 0.0))
    ob0 = _safe_float(orow.get("raw_pressure_block0_norm"), _safe_float(orow.get("pressure_block0_norm"), np.nan))
    ob1 = _safe_float(orow.get("raw_pressure_block1_norm"), _safe_float(orow.get("pressure_block1_norm"), np.nan))
    ob2 = _safe_float(orow.get("raw_pressure_block2_norm"), _safe_float(orow.get("pressure_block2_norm"), np.nan))
    pressure_range = max(b0, b1, b2) - min(b0, b1, b2)
    future_change = max(abs(b1 - b0), abs(b2 - b1), abs(b2 - b0))
    target_vals = [
        _safe_float(row.get("prediction_primary_target_t1_kg")),
        _safe_float(row.get("prediction_primary_target_t2_kg")),
        _safe_float(row.get("prediction_primary_target_t3_kg")),
    ]
    return {
        "case_id": case_id,
        "bucket": int(bucket),
        "run_label": str(sample["run_label"]),
        "forecast_source": str(sample["forecast_source"]),
        "prefix_source": str(sample["prefix_source"]),
        "prefix_length": int(sample["prefix_length"]),
        "forced_mode": str(sample["forced_mode"]),
        "learned_action": learned_action,
        "oracle_action": oracle_action,
        "case_label": str(sample["case_label"]),
        "sample_class": str(sample["sample_class"]),
        "pump_delta": _safe_float(sample["pump_delta"]),
        "fallback_delta": _safe_float(sample["fallback_delta"]),
        "time_over_5_delta": _safe_float(sample["time_over_5_delta"]),
        "posture_distance_delta": _safe_float(sample["posture_distance_delta"]),
        "key_action_match_gain": _safe_float(sample["key_action_match_gain"]),
        "pitch_abs": abs(_safe_float(row.get("current_pitch_deg"), _safe_float(row.get("pitch_deg"), 0.0))),
        "roll_abs": abs(_safe_float(row.get("current_roll_deg"), _safe_float(row.get("roll_deg"), 0.0))),
        "b0_norm": b0,
        "b1_norm": b1,
        "b2_norm": b2,
        "b01_delta": b1 - b0,
        "b12_delta": b2 - b1,
        "b02_delta": b2 - b0,
        "pressure_range": pressure_range,
        "future_change_mag": future_change,
        "anti_persistence": pressure_range / max(abs(b0), 1e-6),
        "oracle_b0_norm": ob0,
        "oracle_b1_norm": ob1,
        "oracle_b2_norm": ob2,
        "oracle_b01_delta": ob1 - ob0 if np.isfinite(ob1) and np.isfinite(ob0) else np.nan,
        "oracle_b12_delta": ob2 - ob1 if np.isfinite(ob2) and np.isfinite(ob1) else np.nan,
        "direction_b01_agree_oracle": int(np.sign(b1 - b0) == np.sign(ob1 - ob0)) if np.isfinite(ob1) and np.isfinite(ob0) else 0,
        "direction_b12_agree_oracle": int(np.sign(b2 - b1) == np.sign(ob2 - ob1)) if np.isfinite(ob2) and np.isfinite(ob1) else 0,
        "target_abs_mean_kg": float(np.nanmean(np.abs(target_vals))),
        "target_abs_max_kg": float(np.nanmax(np.abs(target_vals))),
        "target_age_s": _safe_float(row.get("prediction_primary_target_age_s"), 0.0),
        "target_delta_abs_mean_kg": _safe_float(row.get("prediction_primary_delta_abs_mean_kg"), 0.0),
        "planner_action_pitch_deg": _safe_float(row.get("planner_action_pitch_deg"), 0.0),
        "planner_action_roll_deg": _safe_float(row.get("planner_action_roll_deg"), 0.0),
        "prev_action_active_small": int(_prev_action(learned_logs, case_id, bucket) == "active_small"),
        "prev_action_active_medium": int(_prev_action(learned_logs, case_id, bucket) == "active_medium"),
        "recent_small_count": _count_recent_action(learned_logs, case_id, bucket, "active_small", 3),
        "recent_medium_count": _count_recent_action(learned_logs, case_id, bucket, "active_medium", 3),
        "recent_hold_count": _count_recent_action(learned_logs, case_id, bucket, "hold", 3),
    }


def build_dedup_and_dataset(path_dataset: Path, forced_dir: Path, learned_run: Path, oracle_run: Path, out_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    samples = pd.read_csv(path_dataset / "path_policy_distillation_samples.csv", low_memory=False)
    action = pd.read_csv(forced_dir / "forced_prefix_action_trace.csv", low_memory=False)
    state = pd.read_csv(forced_dir / "forced_prefix_path_state_trace.csv", low_memory=False)
    labels = pd.read_csv(forced_dir / "forced_prefix_case_labels.csv", low_memory=False)
    learned_logs = _load_logs(learned_run)
    oracle_logs = _load_logs(oracle_run)

    # Deduplicate at the prefix-experiment level. This prevents adjacent prefix
    # lengths from being treated as independent proof when they force the same
    # action sequence on the same case.
    dedup_cols = ["case_id", "prefix_length", "forced_mode", "forecast_source", "prefix_source", "forced_sequence"]
    dedup = samples.drop_duplicates(dedup_cols).copy()
    dedup.to_csv(out_dir / "path_policy_distillation_dedup_samples.csv", index=False)

    rows: list[dict[str, Any]] = []
    candidate_samples = samples[
        samples["forced_mode"].isin(["final_action", "raw_action"])
        & samples["prefix_source"].eq("oracle")
    ].copy()
    for _, sample in candidate_samples.iterrows():
        learned_seq = _parse_action_sequence(sample.get("learned_original_sequence"))
        oracle_seq = _parse_action_sequence(sample.get("oracle_prefix_sequence"))
        for bucket, learned_action in learned_seq.items():
            oracle_action = oracle_seq.get(bucket, "")
            if learned_action != "active_small" or oracle_action != "active_medium":
                continue
            row = _feature_row(
                sample=sample,
                bucket=bucket,
                learned_action=learned_action,
                oracle_action=oracle_action,
                learned_logs=learned_logs,
                oracle_logs=oracle_logs,
            )
            if row["sample_class"] == "positive":
                label = 1
                split_class = "positive"
            elif row["sample_class"] == "negative":
                label = 0
                split_class = "negative"
            elif row["sample_class"] == "caution":
                label = -1
                split_class = "caution"
            else:
                label = -2
                split_class = "neutral"
            row["label"] = label
            row["split_class"] = split_class
            rows.append(row)
    dataset = pd.DataFrame(rows)
    # Remove identical evidence rows to avoid prefix-length inflation.
    evidence_cols = [
        "case_id",
        "bucket",
        "forced_mode",
        "learned_action",
        "oracle_action",
        "split_class",
        "case_label",
    ]
    dataset = dataset.drop_duplicates(evidence_cols).copy()
    dataset.to_csv(out_dir / "active_small_to_medium_dataset.csv", index=False)
    _write_independence_report(samples, dedup, dataset, action, state, labels, out_dir)
    _write_dataset_summary(dataset, out_dir)
    return dedup, dataset


def _write_independence_report(
    samples: pd.DataFrame,
    dedup: pd.DataFrame,
    dataset: pd.DataFrame,
    action: pd.DataFrame,
    state: pd.DataFrame,
    labels: pd.DataFrame,
    out_dir: Path,
) -> None:
    class_case = (
        samples.groupby(["sample_class", "case_id"], as_index=False)
        .size()
        .rename(columns={"size": "count"})
    )
    dedup_case = (
        dedup.groupby(["sample_class", "case_id"], as_index=False)
        .size()
        .rename(columns={"size": "dedup_count"})
    )
    dataset_case = (
        dataset.groupby(["split_class", "case_id"], as_index=False)
        .size()
        .rename(columns={"size": "candidate_count"})
        if not dataset.empty
        else pd.DataFrame()
    )
    repeated = (
        samples.groupby(["case_id", "forced_mode", "prefix_source", "forced_sequence"], as_index=False)
        .agg(n_prefix_lengths=("prefix_length", "nunique"), n_rows=("run_label", "count"))
        .query("n_prefix_lengths > 1")
        .sort_values(["n_prefix_lengths", "n_rows"], ascending=False)
    )
    lines = [
        "# Path Policy Distillation Sample Independence",
        "",
        "This report checks how much evidence is repeated across adjacent prefix lengths and cases.",
        "",
        f"Raw prefix samples: {len(samples)}",
        f"Deduplicated prefix samples: {len(dedup)}",
        f"Offline active_small->active_medium candidate rows: {len(dataset)}",
        "",
        "## Raw Sample Concentration by Class / Case",
        "",
        _md_table(class_case, 80),
        "",
        "## Deduplicated Sample Concentration by Class / Case",
        "",
        _md_table(dedup_case, 80),
        "",
        "## Candidate Classification Rows by Class / Case",
        "",
        _md_table(dataset_case, 80),
        "",
        "## Repeated Prefix Sequences Across Neighbor Lengths",
        "",
        _md_table(repeated[["case_id", "forced_mode", "prefix_source", "forced_sequence", "n_prefix_lengths", "n_rows"]], 40),
        "",
        "## Independence Rule",
        "",
        "The offline classifier uses case-level holdout. Adjacent prefix lengths from the same case are not treated as independent generalization evidence.",
    ]
    (out_dir / "path_policy_distillation_sample_independence.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def _write_dataset_summary(dataset: pd.DataFrame, out_dir: Path) -> None:
    if dataset.empty:
        text = "# Active Small To Medium Dataset\n\nNo candidate rows were generated.\n"
        (out_dir / "active_small_to_medium_dataset_summary.md").write_text(text, encoding="utf-8")
        return
    counts = (
        dataset.groupby(["split_class", "case_id"], as_index=False)
        .size()
        .rename(columns={"size": "count"})
    )
    feature_means = (
        dataset.groupby("split_class", as_index=False)
        .agg(
            n=("label", "count"),
            mean_pitch_abs=("pitch_abs", "mean"),
            mean_roll_abs=("roll_abs", "mean"),
            mean_b0=("b0_norm", "mean"),
            mean_b1=("b1_norm", "mean"),
            mean_b2=("b2_norm", "mean"),
            mean_pressure_range=("pressure_range", "mean"),
            mean_future_change=("future_change_mag", "mean"),
            mean_target_delta=("target_delta_abs_mean_kg", "mean"),
            mean_recent_small=("recent_small_count", "mean"),
            mean_recent_medium=("recent_medium_count", "mean"),
        )
    )
    lines = [
        "# Active Small To Medium Offline Dataset",
        "",
        "Candidate rule: learned action is `active_small`, oracle prefix/action is `active_medium`, forced mode is `final_action` or `raw_action`, and `target_update` is excluded.",
        "",
        "Labels: positive = beneficial without pump/fallback/time_over_5 worsening; negative = worse/target-lifecycle-dominated; caution = pump-tradeoff/inconclusive; neutral is not a primary training target.",
        "",
        "## Counts by Class and Case",
        "",
        _md_table(counts, 80),
        "",
        "## Feature Means by Class",
        "",
        _md_table(feature_means, 20),
    ]
    (out_dir / "active_small_to_medium_dataset_summary.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def _md_table(df: pd.DataFrame, max_rows: int = 50) -> str:
    if df.empty:
        return "_No rows._"
    show = df.head(max_rows).copy()
    cols = list(show.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in show.iterrows():
        vals = []
        for col in cols:
            value = row[col]
            if isinstance(value, float):
                vals.append(f"{value:.3f}" if np.isfinite(value) else "")
            else:
                vals.append(str(value))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


FEATURES = [
    "pitch_abs",
    "roll_abs",
    "b0_norm",
    "b1_norm",
    "b2_norm",
    "b01_delta",
    "b12_delta",
    "b02_delta",
    "pressure_range",
    "future_change_mag",
    "anti_persistence",
    "direction_b01_agree_oracle",
    "direction_b12_agree_oracle",
    "target_abs_mean_kg",
    "target_abs_max_kg",
    "target_age_s",
    "target_delta_abs_mean_kg",
    "planner_action_pitch_deg",
    "planner_action_roll_deg",
    "prev_action_active_small",
    "prev_action_active_medium",
    "recent_small_count",
    "recent_medium_count",
    "recent_hold_count",
]


def _eval_predictions(y_true: np.ndarray, score: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    out: dict[str, float] = {}
    if len(np.unique(y_true)) > 1:
        out["auc"] = float(roc_auc_score(y_true, score))
        out["pr_auc"] = float(average_precision_score(y_true, score))
    else:
        out["auc"] = np.nan
        out["pr_auc"] = np.nan
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, pred, average="binary", zero_division=0
    )
    out["precision"] = float(precision)
    out["recall"] = float(recall)
    out["f1"] = float(f1)
    out["n"] = int(len(y_true))
    out["positives"] = int(np.sum(y_true))
    out["predicted_positive"] = int(np.sum(pred))
    return out


def train_offline(dataset: pd.DataFrame, out_dir: Path) -> None:
    trainable = dataset[dataset["label"].isin([0, 1])].copy()
    if trainable.empty or trainable["label"].nunique() < 2:
        _write_no_model_report(out_dir, "Not enough positive/negative training rows.")
        return
    trainable[FEATURES] = trainable[FEATURES].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    cases = sorted(trainable["case_id"].unique())
    prediction_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    importance_rows: list[dict[str, Any]] = []

    for holdout in cases:
        train = trainable[~trainable["case_id"].eq(holdout)].copy()
        test = trainable[trainable["case_id"].eq(holdout)].copy()
        if train["label"].nunique() < 2 or test.empty:
            continue
        x_train = train[FEATURES].to_numpy(dtype=float)
        y_train = train["label"].to_numpy(dtype=int)
        x_test = test[FEATURES].to_numpy(dtype=float)
        y_test = test["label"].to_numpy(dtype=int)
        models = {
            "logistic_regression": make_pipeline(
                StandardScaler(),
                LogisticRegression(max_iter=2000, class_weight="balanced", random_state=7),
            ),
            "random_forest": RandomForestClassifier(
                n_estimators=200,
                max_depth=4,
                min_samples_leaf=2,
                class_weight="balanced_subsample",
                random_state=7,
            ),
        }
        for model_name, model in models.items():
            model.fit(x_train, y_train)
            score = model.predict_proba(x_test)[:, 1]
            for threshold in [0.5, 0.7, 0.85]:
                pred = (score >= threshold).astype(int)
                m = _eval_predictions(y_test, score, pred)
                m.update({"model": model_name, "holdout_case": holdout, "threshold": threshold})
                metric_rows.append(m)
            for row, s in zip(test.itertuples(index=False), score):
                prediction_rows.append(
                    {
                        "model": model_name,
                        "holdout_case": holdout,
                        "case_id": row.case_id,
                        "bucket": row.bucket,
                        "label": int(row.label),
                        "score": float(s),
                        "pred_0p5": int(s >= 0.5),
                        "pred_0p7": int(s >= 0.7),
                        "pred_0p85": int(s >= 0.85),
                        "split_class": row.split_class,
                        "case_label": row.case_label,
                        "pump_delta": float(row.pump_delta),
                        "fallback_delta": float(row.fallback_delta),
                        "time_over_5_delta": float(row.time_over_5_delta),
                        "forced_mode": row.forced_mode,
                        "prefix_length": int(row.prefix_length),
                    }
                )
            if model_name == "random_forest":
                for feature, value in zip(FEATURES, model.feature_importances_):
                    importance_rows.append(
                        {
                            "model": model_name,
                            "holdout_case": holdout,
                            "feature": feature,
                            "importance": float(value),
                        }
                    )
            else:
                clf = model.named_steps["logisticregression"]
                for feature, value in zip(FEATURES, clf.coef_.reshape(-1)):
                    importance_rows.append(
                        {
                            "model": model_name,
                            "holdout_case": holdout,
                            "feature": feature,
                            "importance": float(value),
                        }
                    )

    metrics = pd.DataFrame(metric_rows)
    preds = pd.DataFrame(prediction_rows)
    imp = pd.DataFrame(importance_rows)
    metrics.to_csv(out_dir / "active_small_to_medium_offline_metrics.csv", index=False)
    preds.to_csv(out_dir / "active_small_to_medium_predictions.csv", index=False)
    if not imp.empty:
        (
            imp.groupby(["model", "feature"], as_index=False)["importance"]
            .mean()
            .sort_values(["model", "importance"], ascending=[True, False])
            .to_csv(out_dir / "active_small_to_medium_feature_importance.csv", index=False)
        )
    else:
        pd.DataFrame(columns=["model", "feature", "importance"]).to_csv(
            out_dir / "active_small_to_medium_feature_importance.csv", index=False
        )
    _write_offline_report(trainable, metrics, preds, imp, out_dir)


def _write_no_model_report(out_dir: Path, reason: str) -> None:
    text = f"# Active Small To Medium Offline Report\n\nNo classifier trained: {reason}\n"
    (out_dir / "active_small_to_medium_offline_report.md").write_text(text, encoding="utf-8")


def _write_offline_report(
    trainable: pd.DataFrame,
    metrics: pd.DataFrame,
    preds: pd.DataFrame,
    imp: pd.DataFrame,
    out_dir: Path,
) -> None:
    if preds.empty:
        _write_no_model_report(out_dir, "No predictions generated.")
        return
    # Safety-focused diagnostics: false positives on known dangerous cases.
    safety_rows = []
    for model in sorted(preds["model"].unique()):
        sub = preds[preds["model"].eq(model)]
        for th_col, th in [("pred_0p5", 0.5), ("pred_0p7", 0.7), ("pred_0p85", 0.85)]:
            row = {"model": model, "threshold": th}
            for case in ["b_high_pressure_event", "fr_relief_01"]:
                c = sub[sub["case_id"].eq(case)]
                row[f"{case}_rows"] = int(len(c))
                row[f"{case}_false_positive"] = int(((c[th_col] == 1) & (c["label"] == 0)).sum())
                row[f"{case}_predicted_positive"] = int((c[th_col] == 1).sum())
            safety_rows.append(row)
    safety = pd.DataFrame(safety_rows)
    pos_cases = trainable[trainable["label"].eq(1)]["case_id"].value_counts().rename_axis("case_id").reset_index(name="positive_rows")
    neg_cases = trainable[trainable["label"].eq(0)]["case_id"].value_counts().rename_axis("case_id").reset_index(name="negative_rows")
    trainable_cases = (
        trainable.groupby(["case_id", "label"], as_index=False)
        .size()
        .rename(columns={"size": "rows"})
    )
    pivot_cases = (
        trainable_cases.pivot_table(index="case_id", columns="label", values="rows", fill_value=0)
        .rename(columns={0: "negative_rows", 1: "positive_rows"})
        .reset_index()
    )
    for col in ["negative_rows", "positive_rows"]:
        if col not in pivot_cases.columns:
            pivot_cases[col] = 0
    pivot_cases["can_be_strict_case_holdout_test"] = (
        (pivot_cases["negative_rows"] > 0) & (pivot_cases["positive_rows"] > 0)
    ).astype(int)
    imp_mean = (
        imp.groupby(["model", "feature"], as_index=False)["importance"].mean()
        .sort_values(["model", "importance"], ascending=[True, False])
        if not imp.empty
        else pd.DataFrame()
    )
    best_metric = (
        metrics.groupby(["model", "threshold"], as_index=False)
        .agg(
            n=("n", "sum"),
            positives=("positives", "sum"),
            precision=("precision", "mean"),
            recall=("recall", "mean"),
            f1=("f1", "mean"),
            auc=("auc", "mean"),
            pr_auc=("pr_auc", "mean"),
            predicted_positive=("predicted_positive", "sum"),
        )
        if not metrics.empty
        else pd.DataFrame()
    )
    # Conservative decision rule.
    unsafe = safety[
        (safety["threshold"].eq(0.7))
        & (
            (safety["b_high_pressure_event_predicted_positive"] > 0)
            | (safety["fr_relief_01_predicted_positive"] > 0)
        )
    ]
    enough_cases = trainable[trainable["label"].eq(1)]["case_id"].nunique() >= 3
    has_negative_holdout = trainable[trainable["label"].eq(0)]["case_id"].nunique() >= 2
    if not has_negative_holdout:
        decision = (
            "Do not enter closed-loop probe. Strict case-level rejection of dangerous "
            "cases is not testable because trainable negative evidence exists in only "
            "one case (`b_high_pressure_event`)."
        )
    elif not unsafe.empty or not enough_cases:
        decision = (
            "Do not enter closed-loop probe. Positive evidence is case-concentrated "
            "and/or the classifier predicts upgrades in dangerous/tradeoff cases."
        )
    else:
        decision = (
            "Potentially eligible for a very narrow default-off closed-loop probe, "
            "but only after manual threshold selection."
        )
    lines = [
        "# Active Small To Medium Offline Report",
        "",
        "This is an offline-only classifier. It is not connected to the controller and no closed-loop run was performed.",
        "",
        "## Trainable Evidence Concentration",
        "",
        "Positive rows by case:",
        "",
        _md_table(pos_cases, 20),
        "",
        "Negative rows by case:",
        "",
        _md_table(neg_cases, 20),
        "",
        "Strict case-level holdout coverage:",
        "",
        _md_table(pivot_cases, 20),
        "",
        "Note: if a holdout case contains only positives or only negatives, AUC/PR-AUC are undefined and safety rejection cannot be proven for that case.",
        "",
        "## Case-Level Holdout Metrics",
        "",
        _md_table(best_metric, 30),
        "",
        "## Dangerous Case False Positive Audit",
        "",
        _md_table(safety, 30),
        "",
        "## Feature Importance / Coefficients",
        "",
        _md_table(imp_mean, 40),
        "",
        "## Decision",
        "",
        decision,
    ]
    (out_dir / "active_small_to_medium_offline_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path-dataset-dir", default=str(PATH_DATASET))
    parser.add_argument("--forced-dir", default=str(FORCED_DIR))
    parser.add_argument("--learned-run", default=str(LEARNED_RUN))
    parser.add_argument("--oracle-run", default=str(ORACLE_RUN))
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    return parser.parse_args()


def _resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p


def main() -> None:
    args = parse_args()
    out_dir = _resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _, dataset = build_dedup_and_dataset(
        _resolve(args.path_dataset_dir),
        _resolve(args.forced_dir),
        _resolve(args.learned_run),
        _resolve(args.oracle_run),
        out_dir,
    )
    train_offline(dataset, out_dir)
    print(f"wrote {out_dir / 'active_small_to_medium_offline_report.md'}")
    print(f"dataset_rows={len(dataset)} trainable={int(dataset['label'].isin([0,1]).sum()) if not dataset.empty else 0}")


if __name__ == "__main__":
    main()
