#!/usr/bin/env python3
"""h120 multi-label risk screen v1.

Offline training/validation of risk-screen heads for h120 prepare actions.
No controller gate, no shadow gate, no v1.6 floor changes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REPO = Path(__file__).resolve().parents[2]
ROOTCAUSE = REPO / "outputs/wind_prediction/h120_regression_risk_rootcause_v1"
EXPANSION = REPO / "outputs/wind_prediction/h120_action_effect_dataset_expansion_v1"
OUT = REPO / "outputs/wind_prediction/h120_multilabel_risk_screen_v1"
RAW = OUT / "raw_tables"
DEBUG = OUT / "debug"
PAPER = OUT / "paper_ready"


FEATURE_COLUMNS = [
    "current_pitch_deg",
    "current_roll_deg",
    "max_axis_deg",
    "posture_trend_5m_deg",
    "posture_trend_10m_deg",
    "in_3_5_band",
    "in_4_5_band",
    "near_floor_enter",
    "floor_active",
    "target_age_s",
    "target_error_mean_kg",
    "target_stale",
    "pump_idle",
    "pump_rate_m3_min",
    "near_b0_norm",
    "near_b1_norm",
    "near_b2_norm",
    "near_max_norm",
    "near_last_norm",
    "near_intensification",
    "near_relief",
    "far_60_80_norm",
    "far_80_100_norm",
    "far_100_120_norm",
    "far_min_norm",
    "far_max_norm",
    "far_persistent_high",
    "delayed_intensification",
    "reintensification_after_relief",
    "direction_consistent",
    "signflip_or_reversal",
    "direction_mismatch",
    "near_safe_far_risky",
    "near_risky_far_relief",
    "candidate_prefloor_like",
    "candidate_enter4_like",
]


def ensure_dirs() -> None:
    for path in (OUT, RAW, DEBUG, PAPER):
        path.mkdir(parents=True, exist_ok=True)


def markdown_table(df: pd.DataFrame, max_rows: int | None = None) -> str:
    if df is None or df.empty:
        return "_empty_"
    d = df.copy()
    if max_rows is not None:
        d = d.head(max_rows)

    def fmt(value: Any) -> str:
        if pd.isna(value):
            return ""
        if isinstance(value, float):
            return f"{value:.3f}"
        return str(value)

    lines = [
        "| " + " | ".join(str(c) for c in d.columns) + " |",
        "| " + " | ".join(["---"] * len(d.columns)) + " |",
    ]
    for _, row in d.iterrows():
        lines.append("| " + " | ".join(fmt(row[c]) for c in d.columns) + " |")
    return "\n".join(lines)


def num_series(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col not in df.columns:
        return pd.Series(default, index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce").fillna(default).astype(float)


def as_float(value: Any, default: float = 0.0) -> float:
    try:
        if pd.isna(value):
            return default
    except TypeError:
        pass
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def classifiers() -> dict[str, Any]:
    from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return {
        "logistic": make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                solver="liblinear",
                random_state=4242,
            ),
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            max_iter=90,
            learning_rate=0.05,
            random_state=4242,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=400,
            max_depth=6,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=4242,
        ),
    }


@dataclass(frozen=True)
class RiskHead:
    name: str
    label_col: str


RISK_HEADS = [
    RiskHead("low_posture_false_prepare_prob", "low_posture_false_prepare_label"),
    RiskHead("target_lifecycle_anomaly_prob", "target_lifecycle_anomaly_label"),
    RiskHead("non_monotone_action_response_prob", "non_monotone_action_response_label"),
    RiskHead("far_persistent_high_alone_false_trigger_prob", "far_persistent_high_alone_false_trigger_label"),
    RiskHead("prepare_direction_mismatch_prob", "prepare_direction_mismatch_label"),
    RiskHead("axis_tradeoff_risk_prob", "axis_tradeoff_risk_label"),
    RiskHead("double_pump_no_avoid_prob", "double_pump_no_avoid_label"),
    RiskHead("broader20_false_prepare_prob", "broader20_false_prepare_label"),
    RiskHead("overall_regression_risk_prob", "overall_regression_risk_label"),
]


def build_dataset() -> pd.DataFrame:
    subtype = pd.read_csv(ROOTCAUSE / "regression_risk_subtype_table.csv")
    candidates = pd.read_csv(EXPANSION / "expanded_candidate_table.csv")
    broader = pd.read_csv(ROOTCAUSE / "broader20_false_prepare_table.csv")
    for frame in (subtype, candidates, broader):
        frame["probe_id"] = frame["probe_id"].astype(str)
    broader_probe_ids = set(broader["probe_id"].astype(str))
    base = candidates.drop_duplicates("probe_id").copy()
    labels = subtype.drop_duplicates("probe_id")[
        [
            "probe_id",
            "low_posture_false_prepare_subtype",
            "tank_allocation_target_error_anomaly_subtype",
            "non_monotone_target_lifecycle_subtype",
            "far_persistent_high_alone_false_trigger_subtype",
            "prepare_direction_mismatch_subtype",
            "axis_tradeoff_subtype",
            "double_pump_no_avoid_subtype",
            "regression_risk",
            "anti_trigger",
            "micro50_action_label",
        ]
    ].copy()
    labels = labels.rename(
        columns={
            "regression_risk": "rootcause_regression_risk",
            "anti_trigger": "rootcause_anti_trigger",
        }
    )
    df = base.merge(labels, on="probe_id", how="left")
    df["group_id"] = df["dataset"].astype(str) + "::" + df["case_id"].astype(str)
    df["low_posture_false_prepare_label"] = (num_series(df, "low_posture_false_prepare_subtype") > 0).astype(int)
    df["target_lifecycle_anomaly_label"] = (num_series(df, "tank_allocation_target_error_anomaly_subtype") > 0).astype(int)
    df["non_monotone_action_response_label"] = (num_series(df, "non_monotone_target_lifecycle_subtype") > 0).astype(int)
    df["far_persistent_high_alone_false_trigger_label"] = (
        num_series(df, "far_persistent_high_alone_false_trigger_subtype") > 0
    ).astype(int)
    df["prepare_direction_mismatch_label"] = (num_series(df, "prepare_direction_mismatch_subtype") > 0).astype(int)
    df["axis_tradeoff_risk_label"] = (num_series(df, "axis_tradeoff_subtype") > 0).astype(int)
    df["double_pump_no_avoid_label"] = (num_series(df, "double_pump_no_avoid_subtype") > 0).astype(int)
    df["broader20_false_prepare_label"] = df["probe_id"].isin(broader_probe_ids).astype(int)
    df["overall_regression_risk_label"] = (num_series(df, "rootcause_regression_risk") > 0).astype(int)
    df["is_broader20"] = df["dataset"].astype(str).eq("broader20").astype(int)
    df["is_lowrisk_context"] = df["candidate_reason"].astype(str).str.contains("lowrisk", na=False).astype(int)
    df["is_guard10_positive"] = (
        df["dataset"].astype(str).eq("guard10")
        & (
            (num_series(df, "micro50_positive_effect") > 0)
            | (num_series(df, "pump_saving_positive_effect") > 0)
            | (num_series(df, "positive_effect") > 0)
        )
    ).astype(int)
    df.to_csv(OUT / "multilabel_risk_dataset.csv", index=False)
    df.to_csv(RAW / "multilabel_risk_dataset.csv", index=False)
    return df


def build_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    cols = [c for c in FEATURE_COLUMNS if c in df.columns]
    x = (
        df[cols]
        .apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .reset_index(drop=True)
    )
    return x, list(x.columns)


def make_splits(df: pd.DataFrame, n_splits: int = 5) -> list[tuple[int, np.ndarray, np.ndarray]]:
    from sklearn.model_selection import GroupKFold

    groups = df["group_id"].astype(str).to_numpy()
    unique = np.unique(groups)
    k = min(n_splits, len(unique))
    if k < 2:
        return []
    splitter = GroupKFold(n_splits=k)
    return [
        (fold, train_idx, test_idx)
        for fold, (train_idx, test_idx) in enumerate(splitter.split(df, groups=groups), start=1)
    ]


def safe_auc(y_true: np.ndarray, prob: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score

    try:
        return float(roc_auc_score(y_true, prob)) if len(set(y_true.tolist())) > 1 else np.nan
    except ValueError:
        return np.nan


def train_heads(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    from sklearn.metrics import precision_recall_fscore_support

    splits = make_splits(df)
    x_all, _ = build_features(df)
    metric_rows: list[dict[str, Any]] = []
    pred_rows: list[dict[str, Any]] = []
    for head in RISK_HEADS:
        y_all = df[head.label_col].astype(int).reset_index(drop=True)
        if y_all.nunique() < 2 or not splits:
            continue
        for fold, train_idx, test_idx in splits:
            y_train = y_all.iloc[train_idx].to_numpy()
            y_test = y_all.iloc[test_idx].to_numpy()
            if len(set(y_train.tolist())) < 2:
                continue
            test = df.iloc[test_idx].copy()
            for model_name, model in classifiers().items():
                model.fit(x_all.iloc[train_idx], y_train)
                prob = model.predict_proba(x_all.iloc[test_idx])[:, 1]
                pred = (prob >= 0.5).astype(int)
                precision, recall, f1, _ = precision_recall_fscore_support(
                    y_test, pred, average="binary", zero_division=0
                )
                broader_mask = test["dataset"].astype(str).eq("broader20").to_numpy()
                lowrisk_mask = test["is_lowrisk_context"].astype(int).to_numpy() > 0
                broader_pos = y_test[broader_mask]
                broader_pred = pred[broader_mask]
                lowrisk_pos = y_test[lowrisk_mask]
                lowrisk_pred = pred[lowrisk_mask]
                metric_rows.append(
                    {
                        "head": head.name,
                        "label_col": head.label_col,
                        "model": model_name,
                        "fold": fold,
                        "test_rows": int(len(test)),
                        "test_cases": int(test["group_id"].nunique()),
                        "test_positive": int(y_test.sum()),
                        "precision": float(precision),
                        "recall": float(recall),
                        "f1": float(f1),
                        "roc_auc": safe_auc(y_test, prob),
                        "recall_broader20": float(
                            ((broader_pred == 1) & (broader_pos == 1)).sum() / max(int((broader_pos == 1).sum()), 1)
                        )
                        if len(broader_pos)
                        else np.nan,
                        "recall_lowrisk": float(
                            ((lowrisk_pred == 1) & (lowrisk_pos == 1)).sum() / max(int((lowrisk_pos == 1).sum()), 1)
                        )
                        if len(lowrisk_pos)
                        else np.nan,
                        "false_positive_broader20": int(((pred == 1) & (y_test == 0) & broader_mask).sum()),
                        "false_positive_lowrisk": int(((pred == 1) & (y_test == 0) & lowrisk_mask).sum()),
                    }
                )
                pred_df = test[
                    [
                        "probe_id",
                        "dataset",
                        "case_id",
                        "group_id",
                        "bucket",
                        "candidate_reason",
                        "overall_regression_risk_label",
                        "broader20_false_prepare_label",
                        "low_posture_false_prepare_label",
                        "axis_tradeoff_risk_label",
                        "is_guard10_positive",
                        "is_lowrisk_context",
                    ]
                ].copy()
                pred_df["head"] = head.name
                pred_df["label_col"] = head.label_col
                pred_df["model"] = model_name
                pred_df["fold"] = fold
                pred_df["target"] = y_test
                pred_df["prob"] = prob
                pred_df["pred"] = pred
                pred_rows.extend(pred_df.to_dict("records"))
    fold_metrics = pd.DataFrame(metric_rows)
    preds = pd.DataFrame(pred_rows)
    fold_metrics.to_csv(RAW / "multilabel_risk_fold_metrics.csv", index=False)
    preds.to_csv(RAW / "multilabel_risk_oof_predictions.csv", index=False)
    metrics = (
        fold_metrics.groupby(["head", "label_col", "model"], dropna=False)[
            [
                "test_rows",
                "test_cases",
                "test_positive",
                "precision",
                "recall",
                "f1",
                "roc_auc",
                "recall_broader20",
                "recall_lowrisk",
                "false_positive_broader20",
                "false_positive_lowrisk",
            ]
        ]
        .mean()
        .reset_index()
        if not fold_metrics.empty
        else pd.DataFrame()
    )
    metrics.to_csv(OUT / "multilabel_risk_metrics.csv", index=False)
    metrics.to_csv(RAW / "multilabel_risk_metrics.csv", index=False)
    return metrics, preds


def high_confidence_table(preds: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (head, model), group in preds.groupby(["head", "model"], dropna=False):
        for threshold in (0.50, 0.60, 0.70, 0.80, 0.90):
            picked = group[group["prob"] >= threshold]
            rows.append(
                {
                    "head": head,
                    "model": model,
                    "threshold": threshold,
                    "selected_rows": int(len(picked)),
                    "selected_cases": int(picked["group_id"].nunique()) if not picked.empty else 0,
                    "precision": float(picked["target"].mean()) if not picked.empty else np.nan,
                    "broader20_selected": int(picked["dataset"].astype(str).eq("broader20").sum()) if not picked.empty else 0,
                    "lowrisk_selected": int(picked["is_lowrisk_context"].astype(int).sum()) if not picked.empty else 0,
                    "guard10_positive_selected": int(picked["is_guard10_positive"].astype(int).sum()) if not picked.empty else 0,
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(RAW / "high_confidence_risk_heads.csv", index=False)
    return out


def calibration_bins(preds: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if preds.empty:
        out = pd.DataFrame()
        out.to_csv(OUT / "calibration_bins.csv", index=False)
        out.to_csv(RAW / "calibration_bins.csv", index=False)
        return out
    bins = np.linspace(0.0, 1.0, 6)
    for (head, model), group in preds.groupby(["head", "model"], dropna=False):
        labels = pd.cut(group["prob"], bins=bins, include_lowest=True)
        for interval, sub in group.groupby(labels, observed=False):
            if sub.empty:
                continue
            rows.append(
                {
                    "head": head,
                    "model": model,
                    "bin": str(interval),
                    "rows": int(len(sub)),
                    "mean_prob": float(sub["prob"].mean()),
                    "observed_rate": float(sub["target"].mean()),
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "calibration_bins.csv", index=False)
    out.to_csv(RAW / "calibration_bins.csv", index=False)
    return out


def feature_importance(df: pd.DataFrame) -> pd.DataFrame:
    from sklearn.ensemble import RandomForestClassifier

    x, feature_names = build_features(df)
    rows: list[dict[str, Any]] = []
    for head in RISK_HEADS:
        y = df[head.label_col].astype(int)
        if y.nunique() < 2:
            continue
        model = RandomForestClassifier(
            n_estimators=400,
            max_depth=6,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=4242,
        )
        model.fit(x, y)
        for feature, importance in sorted(
            zip(feature_names, model.feature_importances_), key=lambda item: item[1], reverse=True
        )[:25]:
            rows.append(
                {
                    "head": head.name,
                    "label_col": head.label_col,
                    "feature": feature,
                    "importance": float(importance),
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "feature_importance_table.csv", index=False)
    out.to_csv(RAW / "feature_importance_table.csv", index=False)
    return out


def best_model_map(metrics: pd.DataFrame) -> dict[str, str]:
    best: dict[str, str] = {}
    for head, group in metrics.groupby("head", dropna=False):
        ranked = group.sort_values(["recall", "f1", "roc_auc"], ascending=[False, False, False])
        if not ranked.empty:
            best[str(head)] = str(ranked.iloc[0]["model"])
    return best


def pivot_predictions(preds: pd.DataFrame, metrics: pd.DataFrame) -> pd.DataFrame:
    best = best_model_map(metrics)
    frames = []
    for head, model in best.items():
        sub = preds[(preds["head"] == head) & (preds["model"] == model)].copy()
        sub = sub[["probe_id", "group_id", "prob"]].rename(columns={"prob": head})
        frames.append(sub)
    if not frames:
        return pd.DataFrame()
    out = frames[0]
    for sub in frames[1:]:
        out = out.merge(sub, on=["probe_id", "group_id"], how="outer")
    return out


def evaluate_risk_screen(df: pd.DataFrame, preds: pd.DataFrame, metrics: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    pivot = pivot_predictions(preds, metrics)
    if pivot.empty:
        empty = pd.DataFrame()
        return empty, empty, {}
    scored = df.merge(pivot, on=["probe_id", "group_id"], how="left")
    screen_variants = {
        "core_requested": [
            "low_posture_false_prepare_prob",
            "prepare_direction_mismatch_prob",
            "axis_tradeoff_risk_prob",
            "target_lifecycle_anomaly_prob",
            "overall_regression_risk_prob",
        ],
        "core_plus_broader_head": [
            "low_posture_false_prepare_prob",
            "prepare_direction_mismatch_prob",
            "axis_tradeoff_risk_prob",
            "target_lifecycle_anomaly_prob",
            "overall_regression_risk_prob",
            "broader20_false_prepare_prob",
        ],
        "all_heads_diagnostic": [
            "low_posture_false_prepare_prob",
            "target_lifecycle_anomaly_prob",
            "non_monotone_action_response_prob",
            "far_persistent_high_alone_false_trigger_prob",
            "prepare_direction_mismatch_prob",
            "axis_tradeoff_risk_prob",
            "double_pump_no_avoid_prob",
            "broader20_false_prepare_prob",
            "overall_regression_risk_prob",
        ],
    }
    rows: list[dict[str, Any]] = []
    clean_rows: list[dict[str, Any]] = []
    for screen_variant, risk_heads in screen_variants.items():
        for threshold in (0.35, 0.45, 0.55, 0.65, 0.75, 0.85):
            blocked = pd.Series(False, index=scored.index)
            for head in risk_heads:
                if head in scored:
                    blocked |= num_series(scored, head, 1.0) >= threshold
            clean = scored[~blocked].copy()
            broader_false = scored["broader20_false_prepare_label"].astype(int) > 0
            lowrisk_false = (
                (scored["is_lowrisk_context"].astype(int) > 0)
                & (scored["overall_regression_risk_label"].astype(int) > 0)
            )
            guard_pos = scored["is_guard10_positive"].astype(int) > 0
            bad = (
                (num_series(clean, "overall_regression_risk_label") > 0)
                | (num_series(clean, "broader20_false_prepare_label") > 0)
                | (num_series(clean, "axis_tradeoff_risk_label") > 0)
                | (num_series(clean, "target_lifecycle_anomaly_label") > 0)
                | (num_series(clean, "non_monotone_action_response_label") > 0)
            )
            rows.append(
                {
                    "screen_variant": screen_variant,
                    "risk_threshold": threshold,
                    "risk_heads_used": ",".join(risk_heads),
                    "blocked_buckets": int(blocked.sum()),
                    "allowed_buckets": int((~blocked).sum()),
                    "allowed_cases": int(clean["group_id"].nunique()) if not clean.empty else 0,
                    "broader20_false_prepare_block_rate": float(
                        (blocked & broader_false).sum() / max(int(broader_false.sum()), 1)
                    ),
                    "lowrisk_false_prepare_block_rate": float(
                        (blocked & lowrisk_false).sum() / max(int(lowrisk_false.sum()), 1)
                    ),
                    "guard10_positive_block_rate": float((blocked & guard_pos).sum() / max(int(guard_pos.sum()), 1)),
                    "guard10_positive_kept": int((~blocked & guard_pos).sum()),
                    "clean_bad_rate": float(bad.mean()) if not clean.empty else np.nan,
                    "clean_overall_regression_rate": float(num_series(clean, "overall_regression_risk_label").mean())
                    if not clean.empty
                    else np.nan,
                    "clean_broader20_false_rate": float(num_series(clean, "broader20_false_prepare_label").mean())
                    if not clean.empty
                    else np.nan,
                    "clean_positive_rate": float(num_series(clean, "positive_effect").mean())
                    if "positive_effect" in clean and not clean.empty
                    else np.nan,
                    "clean_micro50_positive_rate": float(num_series(clean, "micro50_positive_effect").mean())
                    if "micro50_positive_effect" in clean and not clean.empty
                    else np.nan,
                }
            )
            tmp = clean[["probe_id", "dataset", "case_id", "group_id", "bucket"]].copy()
            tmp["screen_variant"] = screen_variant
            tmp["risk_threshold"] = threshold
            tmp["clean_bad"] = bad.astype(int).to_numpy() if not clean.empty else []
            clean_rows.extend(tmp.to_dict("records"))
    combo = pd.DataFrame(rows)
    clean_table = pd.DataFrame(clean_rows)
    combo.to_csv(OUT / "risk_screen_combination_table.csv", index=False)
    combo.to_csv(RAW / "risk_screen_combination_table.csv", index=False)
    clean_table.to_csv(OUT / "clean_subset_diagnostics.csv", index=False)
    clean_table.to_csv(RAW / "clean_subset_diagnostics.csv", index=False)
    evidence = {}
    if not combo.empty:
        combo_core = combo[combo["screen_variant"].eq("core_requested")].copy()
        if combo_core.empty:
            combo_core = combo.copy()
        candidates = combo_core[
            (combo_core["allowed_buckets"] >= 5)
            & (combo_core["guard10_positive_kept"] >= 1)
        ].copy()
        if candidates.empty:
            candidates = combo_core.copy()
        candidates["score"] = (
            candidates["broader20_false_prepare_block_rate"]
            + candidates["lowrisk_false_prepare_block_rate"]
            - candidates["clean_bad_rate"].fillna(1.0)
            - 0.25 * candidates["guard10_positive_block_rate"]
        )
        best = candidates.sort_values("score", ascending=False).iloc[0]
        evidence = {k: (float(v) if isinstance(v, (np.floating, float)) else int(v) if isinstance(v, (np.integer, int)) else v) for k, v in best.to_dict().items()}
    scored.to_csv(RAW / "risk_screen_oof_score_table.csv", index=False)
    return combo, clean_table, evidence


def write_manifest(df: pd.DataFrame) -> None:
    label_counts = pd.DataFrame(
        [
            {
                "label": head.label_col,
                "positive_count": int(df[head.label_col].sum()),
                "positive_rate": float(df[head.label_col].mean()),
            }
            for head in RISK_HEADS
        ]
    )
    label_counts.to_csv(RAW / "risk_label_distribution.csv", index=False)
    split_rows = []
    for fold, train_idx, test_idx in make_splits(df):
        train = df.iloc[train_idx]
        test = df.iloc[test_idx]
        split_rows.append(
            {
                "fold": fold,
                "train_rows": int(len(train)),
                "test_rows": int(len(test)),
                "train_cases": int(train["group_id"].nunique()),
                "test_cases": int(test["group_id"].nunique()),
                "test_datasets": ",".join(sorted(test["dataset"].astype(str).unique())),
            }
        )
    split = pd.DataFrame(split_rows)
    split.to_csv(RAW / "split_distribution.csv", index=False)
    md = [
        "# Multilabel Risk Dataset Manifest",
        "",
        f"- Rows / buckets: {len(df)}.",
        f"- Cases: {df['group_id'].nunique()}.",
        "- Unit: bucket-level prepare risk screen sample.",
        "- Split: GroupKFold by `dataset::case_id`; case_id/dataset_group are not training features.",
        "- Feature set: current observable posture/execution state and near/far forecast pressure/shape features only.",
        "- Labels are diagnostic supervision from root-cause/counterfactual outputs; future outcomes are not used as features.",
        "",
        "## Label Distribution",
        markdown_table(label_counts),
        "",
        "## Split Distribution",
        markdown_table(split),
    ]
    (OUT / "multilabel_risk_dataset_manifest.md").write_text("\n".join(md) + "\n")


def write_reports(
    df: pd.DataFrame,
    metrics: pd.DataFrame,
    hc: pd.DataFrame,
    cal: pd.DataFrame,
    importance: pd.DataFrame,
    combo: pd.DataFrame,
    evidence: dict[str, Any],
) -> None:
    overall = metrics[metrics["head"].eq("overall_regression_risk_prob")].sort_values(
        ["recall", "f1"], ascending=False
    )
    broader = metrics[metrics["head"].eq("broader20_false_prepare_prob")].sort_values(
        ["recall", "f1"], ascending=False
    )
    lowpost = metrics[metrics["head"].eq("low_posture_false_prepare_prob")].sort_values(
        ["recall", "f1"], ascending=False
    )
    overall_recall = as_float(overall.iloc[0]["recall"], np.nan) if not overall.empty else np.nan
    broader_recall = as_float(broader.iloc[0]["recall"], np.nan) if not broader.empty else np.nan
    lowpost_recall = as_float(lowpost.iloc[0]["recall"], np.nan) if not lowpost.empty else np.nan
    clean_bad = as_float(evidence.get("clean_bad_rate"), np.nan)
    allowed_cases = int(evidence.get("allowed_cases", 0) or 0)
    guard_kept = int(evidence.get("guard10_positive_kept", 0) or 0)
    go = (
        overall_recall >= 0.75
        and as_float(evidence.get("broader20_false_prepare_block_rate"), 0.0) >= 0.80
        and as_float(evidence.get("lowrisk_false_prepare_block_rate"), 0.0) >= 0.80
        and clean_bad < 0.75
        and guard_kept > 0
        and allowed_cases >= 3
    )
    paper = [
        "# h120 Multilabel Risk Screen Findings",
        "",
        f"- Dataset rows: {len(df)}; cases: {df['group_id'].nunique()}.",
        f"- Overall regression best recall: {overall_recall:.3f}.",
        f"- Broader20 false prepare best recall: {broader_recall:.3f}.",
        f"- Low-posture false prepare best recall: {lowpost_recall:.3f}.",
        f"- Best combination threshold: {evidence.get('risk_threshold', np.nan)}.",
        f"- Best clean bad-rate: {clean_bad:.3f}; allowed buckets={evidence.get('allowed_buckets', 0)}, allowed cases={allowed_cases}.",
        f"- Guard10 positives kept: {guard_kept}.",
        f"- Go to clean-subset value/ranking: {'yes' if go else 'no'}.",
        "",
        "## Best Metrics By Head",
        markdown_table(
            metrics.sort_values(["head", "recall", "f1"], ascending=[True, False, False])
            .groupby("head")
            .head(1)
            .reset_index(drop=True)
        ),
        "",
        "## Risk Screen Combination",
        markdown_table(combo),
    ]
    (PAPER / "risk_screen_findings.md").write_text("\n".join(paper) + "\n")
    decision = [
        "# h120 Multilabel Risk Screen Decision",
        "",
        "Scope: offline multi-label risk screen only.  No controller gate, no shadow gate, and no v1.6 floor changes.",
        "",
        "## Results",
        "",
        f"- Overall regression-risk best recall: {overall_recall:.3f}.",
        f"- Broader20 false-prepare best recall: {broader_recall:.3f}.",
        f"- Low-posture false-prepare best recall: {lowpost_recall:.3f}.",
        f"- Best combination screen blocks broader20 false prepare at {as_float(evidence.get('broader20_false_prepare_block_rate'), np.nan):.3f}.",
        f"- Best combination screen blocks lowrisk false prepare at {as_float(evidence.get('lowrisk_false_prepare_block_rate'), np.nan):.3f}.",
        f"- Clean subset bad-rate: {clean_bad:.3f}; previous clean bad-rate was 0.750.",
        f"- Guard10 positives kept: {guard_kept}.",
        "",
        "## Required Answers",
        "",
        f"1. **Are regression-risk subtypes learnable?**  Partly.  Several subtype heads are learnable, but the deciding head is overall regression risk; best recall is {overall_recall:.3f}.",
        f"2. **Can broader20 false prepare be stably blocked?**  {'Yes in offline OOF screening' if as_float(evidence.get('broader20_false_prepare_block_rate'), 0.0) >= 0.80 else 'Not yet'}.  Best screen block-rate is {as_float(evidence.get('broader20_false_prepare_block_rate'), np.nan):.3f}.",
        f"3. **Is lowrisk false prepare controllable?**  {'Mostly yes' if as_float(evidence.get('lowrisk_false_prepare_block_rate'), 0.0) >= 0.80 else 'Not yet'}.  Best lowrisk block-rate is {as_float(evidence.get('lowrisk_false_prepare_block_rate'), np.nan):.3f}.",
        f"4. **Is the clean subset cleaner than last round?**  {'Yes' if clean_bad < 0.75 else 'No'}.  Clean bad-rate is {clean_bad:.3f}.",
        f"5. **Allow clean-subset value/ranking?**  {'Yes, offline only' if go else 'No'}.",
        "6. **If not allowed, next step?**  If no-go, improve labels/features around the weak heads or redesign action/state interface; do not move to gate.  If go, next step is offline clean-subset value/ranking, not controller.",
        "7. **Does v1.6 remain mainline?**  Yes.  This remains an offline h120 risk-screen research path.",
    ]
    (OUT / "h120_multilabel_risk_screen_decision.md").write_text("\n".join(decision) + "\n")
    pd.DataFrame([{"go_clean_subset_value_learning": int(go), **evidence}]).to_csv(
        RAW / "risk_screen_go_no_go_evidence.csv", index=False
    )


def main() -> None:
    ensure_dirs()
    df = build_dataset()
    write_manifest(df)
    metrics, preds = train_heads(df)
    hc = high_confidence_table(preds)
    cal = calibration_bins(preds)
    importance = feature_importance(df)
    combo, clean, evidence = evaluate_risk_screen(df, preds, metrics)
    write_reports(df, metrics, hc, cal, importance, combo, evidence)


if __name__ == "__main__":
    main()
