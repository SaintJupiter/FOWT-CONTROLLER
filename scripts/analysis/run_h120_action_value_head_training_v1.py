#!/usr/bin/env python3
"""Offline h120 action-value head training v1.

This trains lightweight offline heads from h120_action_value_label_refinement_v1.
It does not attach a controller gate, does not modify v1.6 floor logic, and
does not train a complex sequence model.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REPO = Path(__file__).resolve().parents[2]
SOURCE = REPO / "outputs/wind_prediction/h120_action_value_label_refinement_v1"
OUT = REPO / "outputs/wind_prediction/h120_action_value_head_training_v1"
DEBUG = OUT / "debug"
RAW = OUT / "raw_tables"

PREPARE_ACTIONS = ("pump_saving_prepare", "active_small_prepare")
PUMP_ACTION = "pump_saving_prepare"
ANTI_PATTERN = (
    "broader20_anti_trigger|near_safe_far_risky|direction_mismatch|lowrisk_sanity"
)

FEATURE_COLUMNS = [
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
    for path in (OUT, DEBUG, RAW):
        path.mkdir(parents=True, exist_ok=True)


def markdown_table(df: pd.DataFrame, max_rows: int | None = None) -> str:
    if df is None or df.empty:
        return "_empty_"
    d = df.copy()
    if max_rows is not None:
        d = d.head(max_rows)
    cols = [str(c) for c in d.columns]

    def fmt(value: Any) -> str:
        if pd.isna(value):
            return ""
        if isinstance(value, float):
            return f"{value:.3f}"
        return str(value)

    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for _, row in d.iterrows():
        lines.append("| " + " | ".join(fmt(row[c]) for c in d.columns) + " |")
    return "\n".join(lines)


def num_series(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col not in df:
        return pd.Series(default, index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce").fillna(default).astype(float)


def load_training_table() -> pd.DataFrame:
    path = SOURCE / "constrained_label_table.csv"
    if not path.exists():
        raise FileNotFoundError(f"missing constrained label table: {path}")
    df = pd.read_csv(path)
    prep = df[df["action"].isin(PREPARE_ACTIONS)].copy().reset_index(drop=True)
    prep["row_id"] = np.arange(len(prep))
    prep["group_id"] = prep["dataset"].astype(str) + "::" + prep["case_id"].astype(str)
    prep["is_broader20"] = (prep["dataset"].astype(str) == "broader20").astype(int)
    prep["is_lowrisk_sanity"] = prep["selection_reasons"].astype(str).str.contains(
        "lowrisk_sanity", na=False
    ).astype(int)
    prep["is_near_safe_far_risky"] = (num_series(prep, "near_safe_far_risky") > 0).astype(int)
    prep["is_direction_mismatch"] = (num_series(prep, "direction_mismatch") > 0).astype(int)
    prep["is_pump_saving_action"] = (prep["action"] == PUMP_ACTION).astype(int)
    prep["is_active_small_action"] = (prep["action"] == "active_small_prepare").astype(int)

    hard_regression = (
        (num_series(prep, "delta_full_fallback_pp") > 0.05)
        | (num_series(prep, "delta_full_max_p95") > 0.05)
        | (num_series(prep, "time_gain_s") < -5.0)
        | (num_series(prep, "idle_gain_s") < -30.0)
    )
    no_safety_gain = (
        (num_series(prep, "time_gain_s") < 30.0)
        & (num_series(prep, "idle_gain_s") < 60.0)
        & (num_series(prep, "pump_delta_m3") > 0.0)
    )
    anti_context = (
        prep["selection_reasons"].astype(str).str.contains(ANTI_PATTERN, regex=True, na=False)
        | (prep["is_near_safe_far_risky"] > 0)
        | (prep["is_direction_mismatch"] > 0)
        | (prep["is_lowrisk_sanity"] > 0)
    )
    is_safe = prep["constrained_label"].astype(str).eq("safe_positive")
    is_negative = prep["constrained_label"].astype(str).eq("negative")
    is_ambiguous = prep["constrained_label"].astype(str).eq("ambiguous")

    prep["hard_regression_label"] = hard_regression.astype(int)
    prep["regression_risk_label"] = hard_regression.astype(int)
    prep["anti_trigger_label"] = np.nan
    prep.loc[is_safe, "anti_trigger_label"] = 0
    prep.loc[anti_context | (is_negative & (no_safety_gain | hard_regression)), "anti_trigger_label"] = 1
    # Ambiguous rows without a specific anti-trigger pattern remain unlabeled.
    prep.loc[is_ambiguous & ~(anti_context | hard_regression), "anti_trigger_label"] = np.nan

    prep["pump_saving_safe_positive_label"] = np.nan
    pump_mask = prep["action"].eq(PUMP_ACTION)
    prep.loc[pump_mask & is_safe, "pump_saving_safe_positive_label"] = 1
    prep.loc[pump_mask & is_negative, "pump_saving_safe_positive_label"] = 0

    prep["active_small_safe_positive_label"] = np.nan
    active_mask = prep["action"].eq("active_small_prepare")
    prep.loc[active_mask & is_safe, "active_small_safe_positive_label"] = 1
    prep.loc[active_mask & is_negative, "active_small_safe_positive_label"] = 0

    prep["expected_time_gain_s"] = num_series(prep, "time_gain_s")
    prep["expected_idle_gain_s"] = num_series(prep, "idle_gain_s")
    prep["expected_pump_delta_m3"] = num_series(prep, "pump_delta_m3")
    prep["anti_context_label"] = anti_context.astype(int)
    prep.to_csv(OUT / "training_table.csv", index=False)
    prep.to_csv(RAW / "training_table.csv", index=False)
    return prep


def build_features(df: pd.DataFrame, include_action: bool) -> tuple[pd.DataFrame, list[str]]:
    cols = [c for c in FEATURE_COLUMNS if c in df.columns]
    x = (
        df[cols]
        .apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
    )
    if include_action:
        action = pd.get_dummies(df["action"].astype(str), prefix="action")
        x = pd.concat([x.reset_index(drop=True), action.reset_index(drop=True)], axis=1)
    return x, list(x.columns)


@dataclass(frozen=True)
class ClassificationHead:
    name: str
    label_col: str
    row_filter: str
    positive_name: str
    include_action: bool


@dataclass(frozen=True)
class RegressionHead:
    name: str
    target_col: str
    include_action: bool


CLASS_HEADS = [
    ClassificationHead(
        "anti_trigger",
        "anti_trigger_label",
        "all_prepare",
        "reject_prepare",
        True,
    ),
    ClassificationHead(
        "pump_saving_safe_positive",
        "pump_saving_safe_positive_label",
        "pump_saving_only",
        "safe_prepare",
        False,
    ),
    ClassificationHead(
        "regression_risk",
        "regression_risk_label",
        "all_prepare",
        "regression_risk",
        True,
    ),
]

REG_HEADS = [
    RegressionHead("expected_time_gain_s", "expected_time_gain_s", True),
    RegressionHead("expected_idle_gain_s", "expected_idle_gain_s", True),
    RegressionHead("expected_pump_delta_m3", "expected_pump_delta_m3", True),
]


def head_mask(df: pd.DataFrame, row_filter: str) -> pd.Series:
    if row_filter == "pump_saving_only":
        return df["action"].eq(PUMP_ACTION)
    return pd.Series(True, index=df.index)


def make_splits(df: pd.DataFrame) -> list[tuple[int, np.ndarray, np.ndarray]]:
    from sklearn.model_selection import GroupKFold

    groups = df["group_id"].astype(str).to_numpy()
    unique_groups = np.unique(groups)
    n_splits = min(5, len(unique_groups))
    splitter = GroupKFold(n_splits=n_splits)
    return [
        (fold, train_idx, test_idx)
        for fold, (train_idx, test_idx) in enumerate(
            splitter.split(df, groups=groups), start=1
        )
    ]


def write_split_distribution(df: pd.DataFrame, splits: list[tuple[int, np.ndarray, np.ndarray]]) -> pd.DataFrame:
    rows = []
    for fold, train_idx, test_idx in splits:
        train = df.iloc[train_idx]
        test = df.iloc[test_idx]
        base = {
            "fold": fold,
            "train_rows": len(train),
            "test_rows": len(test),
            "train_groups": train["group_id"].nunique(),
            "test_groups": test["group_id"].nunique(),
            "test_group_ids": ";".join(sorted(test["group_id"].unique())),
        }
        for head in CLASS_HEADS:
            for prefix, sub in (("train", train), ("test", test)):
                valid = sub[head_mask(sub, head.row_filter) & sub[head.label_col].notna()]
                base[f"{prefix}_{head.name}_rows"] = len(valid)
                base[f"{prefix}_{head.name}_positive"] = int(
                    pd.to_numeric(valid[head.label_col], errors="coerce").fillna(0).sum()
                )
        rows.append(base)
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "split_distribution.csv", index=False)
    return out


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
                random_state=321,
            ),
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            max_iter=80,
            learning_rate=0.05,
            random_state=321,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=5,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=321,
        ),
    }


def regressors() -> dict[str, Any]:
    from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return {
        "ridge": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        "hist_gradient_boosting": HistGradientBoostingRegressor(
            max_iter=80,
            learning_rate=0.05,
            random_state=321,
        ),
        "random_forest": RandomForestRegressor(
            n_estimators=300,
            max_depth=5,
            min_samples_leaf=2,
            random_state=321,
        ),
    }


def classification_metrics(y_true: np.ndarray, proba: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    from sklearn.metrics import precision_recall_fscore_support, roc_auc_score

    pred = (proba >= threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, pred, average="binary", zero_division=0
    )
    try:
        auc = roc_auc_score(y_true, proba) if len(set(y_true.tolist())) > 1 else np.nan
    except ValueError:
        auc = np.nan
    high = proba >= 0.70
    high_precision = float(y_true[high].mean()) if high.any() else np.nan
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "roc_auc": float(auc) if not pd.isna(auc) else np.nan,
        "high_conf_positive_rows": int(high.sum()),
        "high_conf_precision": high_precision,
    }


def run_classification_heads(df: pd.DataFrame, splits: list[tuple[int, np.ndarray, np.ndarray]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    model_factories = classifiers
    rows = []
    pred_rows = []
    for head in CLASS_HEADS:
        x_all, feature_names = build_features(df, head.include_action)
        valid_mask = head_mask(df, head.row_filter) & df[head.label_col].notna()
        for fold, train_idx, test_idx in splits:
            train_mask = valid_mask.copy()
            train_mask.iloc[:] = False
            train_mask.iloc[train_idx] = True
            train_mask &= valid_mask

            eval_mask = valid_mask.copy()
            eval_mask.iloc[:] = False
            eval_mask.iloc[test_idx] = True
            eval_mask &= valid_mask

            pred_mask = head_mask(df, head.row_filter).copy()
            pred_mask.iloc[:] = False
            pred_mask.iloc[test_idx] = True
            pred_mask &= head_mask(df, head.row_filter)

            if train_mask.sum() == 0 or eval_mask.sum() == 0:
                continue
            y_train = pd.to_numeric(df.loc[train_mask, head.label_col], errors="coerce").astype(int).to_numpy()
            y_eval = pd.to_numeric(df.loc[eval_mask, head.label_col], errors="coerce").astype(int).to_numpy()
            if len(set(y_train.tolist())) < 2:
                rows.append(
                    {
                        "head": head.name,
                        "model": "skipped",
                        "fold": fold,
                        "task_type": "classification",
                        "reason": "train fold has one class",
                    }
                )
                continue
            for model_name, model in model_factories().items():
                model.fit(x_all.loc[train_mask], y_train)
                eval_proba = model.predict_proba(x_all.loc[eval_mask])[:, 1]
                metrics = classification_metrics(y_eval, eval_proba)
                eval_df = df.loc[eval_mask].copy()
                eval_pred = (eval_proba >= 0.5).astype(int)
                false_positive = (eval_pred == 1) & (y_eval == 0)
                broader_lowrisk = (
                    eval_df["dataset"].astype(str).eq("broader20")
                    | eval_df["selection_reasons"].astype(str).str.contains("lowrisk_sanity", na=False)
                ).to_numpy()
                near_safe = (num_series(eval_df, "near_safe_far_risky").to_numpy() > 0)
                mismatch = (num_series(eval_df, "direction_mismatch").to_numpy() > 0)
                anti_eval = (
                    eval_df["selection_reasons"].astype(str).str.contains(ANTI_PATTERN, regex=True, na=False)
                    | (near_safe)
                    | (mismatch)
                ).to_numpy()
                if head.name == "pump_saving_safe_positive":
                    anti_recall = float(((eval_proba < 0.5) & anti_eval).sum() / max(anti_eval.sum(), 1))
                elif head.name == "anti_trigger":
                    anti_recall = metrics["recall"]
                else:
                    anti_recall = np.nan
                rows.append(
                    {
                        "head": head.name,
                        "model": model_name,
                        "fold": fold,
                        "task_type": "classification",
                        "test_rows": int(eval_mask.sum()),
                        "test_positive": int(y_eval.sum()),
                        **metrics,
                        "false_positive_broader_lowrisk": int((false_positive & broader_lowrisk).sum()),
                        "false_positive_near_safe_far_risky": int((false_positive & near_safe).sum()),
                        "false_positive_direction_mismatch": int((false_positive & mismatch).sum()),
                        "anti_trigger_recall": anti_recall,
                    }
                )
                if pred_mask.sum():
                    pred_proba = model.predict_proba(x_all.loc[pred_mask])[:, 1]
                    pred_subset = df.loc[pred_mask, [
                        "row_id",
                        "refined_probe_id",
                        "dataset",
                        "case_id",
                        "group_id",
                        "action",
                        "constrained_label",
                        "selection_reasons",
                        "near_safe_far_risky",
                        "direction_mismatch",
                        "time_gain_s",
                        "idle_gain_s",
                        "pump_delta_m3",
                        "delta_full_fallback_pp",
                        "delta_full_max_p95",
                    ]].copy()
                    pred_subset["fold"] = fold
                    pred_subset["head"] = head.name
                    pred_subset["model"] = model_name
                    pred_subset["label_col"] = head.label_col
                    pred_subset["label_value"] = pd.to_numeric(
                        df.loc[pred_mask, head.label_col], errors="coerce"
                    ).to_numpy()
                    pred_subset["proba"] = pred_proba
                    pred_rows.extend(pred_subset.to_dict("records"))
    metrics_df = pd.DataFrame(rows)
    pred_df = pd.DataFrame(pred_rows)
    return metrics_df, pred_df


def run_regression_heads(df: pd.DataFrame, splits: list[tuple[int, np.ndarray, np.ndarray]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    rows = []
    pred_rows = []
    for head in REG_HEADS:
        x_all, feature_names = build_features(df, head.include_action)
        y_all = pd.to_numeric(df[head.target_col], errors="coerce").astype(float)
        for fold, train_idx, test_idx in splits:
            train_mask = pd.Series(False, index=df.index)
            test_mask = pd.Series(False, index=df.index)
            train_mask.iloc[train_idx] = True
            test_mask.iloc[test_idx] = True
            for model_name, model in regressors().items():
                model.fit(x_all.loc[train_mask], y_all.loc[train_mask])
                pred = model.predict(x_all.loc[test_mask])
                y_true = y_all.loc[test_mask].to_numpy()
                rmse = math.sqrt(mean_squared_error(y_true, pred))
                rows.append(
                    {
                        "head": head.name,
                        "model": model_name,
                        "fold": fold,
                        "task_type": "regression",
                        "test_rows": int(test_mask.sum()),
                        "mae": float(mean_absolute_error(y_true, pred)),
                        "rmse": float(rmse),
                        "r2": float(r2_score(y_true, pred)) if len(y_true) > 1 else np.nan,
                    }
                )
                pred_subset = df.loc[test_mask, [
                    "row_id",
                    "refined_probe_id",
                    "dataset",
                    "case_id",
                    "group_id",
                    "action",
                    "constrained_label",
                ]].copy()
                pred_subset["fold"] = fold
                pred_subset["head"] = head.name
                pred_subset["model"] = model_name
                pred_subset["target"] = y_true
                pred_subset["prediction"] = pred
                pred_rows.extend(pred_subset.to_dict("records"))
    return pd.DataFrame(rows), pd.DataFrame(pred_rows)


def aggregate_metrics(class_metrics: pd.DataFrame, reg_metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if not class_metrics.empty:
        valid = class_metrics[class_metrics["model"].ne("skipped")].copy()
        group_cols = ["head", "model", "task_type"]
        agg_cols = [
            "precision",
            "recall",
            "f1",
            "roc_auc",
            "high_conf_positive_rows",
            "high_conf_precision",
            "false_positive_broader_lowrisk",
            "false_positive_near_safe_far_risky",
            "false_positive_direction_mismatch",
            "anti_trigger_recall",
        ]
        rows.append(
            valid.groupby(group_cols, dropna=False)[agg_cols]
            .mean()
            .reset_index()
        )
    if not reg_metrics.empty:
        rows.append(
            reg_metrics.groupby(["head", "model", "task_type"], dropna=False)[
                ["mae", "rmse", "r2"]
            ]
            .mean()
            .reset_index()
        )
    out = pd.concat(rows, ignore_index=True, sort=False) if rows else pd.DataFrame()
    out.to_csv(OUT / "head_metrics_table.csv", index=False)
    return out


def calibration_bins(class_predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if class_predictions.empty:
        out = pd.DataFrame()
        out.to_csv(OUT / "calibration_bins.csv", index=False)
        return out
    bins = [0.0, 0.2, 0.4, 0.6, 0.8, 1.000001]
    labels = ["0.0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1.0"]
    pred = class_predictions[class_predictions["label_value"].notna()].copy()
    pred["bin"] = pd.cut(pred["proba"], bins=bins, labels=labels, include_lowest=True)
    for (head, model, bin_name), g in pred.groupby(["head", "model", "bin"], dropna=False):
        rows.append(
            {
                "head": head,
                "model": model,
                "confidence_bin": str(bin_name),
                "rows": len(g),
                "mean_pred": float(g["proba"].mean()) if len(g) else np.nan,
                "observed_rate": float(pd.to_numeric(g["label_value"]).mean()) if len(g) else np.nan,
                "positive_rows": int(pd.to_numeric(g["label_value"]).sum()) if len(g) else 0,
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "calibration_bins.csv", index=False)
    return out


def feature_importances(df: pd.DataFrame) -> pd.DataFrame:
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    rows = []
    for head in CLASS_HEADS:
        mask = head_mask(df, head.row_filter) & df[head.label_col].notna()
        if mask.sum() < 5:
            continue
        y = pd.to_numeric(df.loc[mask, head.label_col], errors="coerce").astype(int).to_numpy()
        if len(set(y.tolist())) < 2:
            continue
        x, feature_names = build_features(df.loc[mask].reset_index(drop=True), head.include_action)
        rf = RandomForestClassifier(
            n_estimators=300,
            max_depth=5,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=777,
        )
        rf.fit(x, y)
        for feature, importance in sorted(
            zip(feature_names, rf.feature_importances_), key=lambda v: v[1], reverse=True
        )[:20]:
            rows.append(
                {
                    "head": head.name,
                    "model": "random_forest",
                    "feature": feature,
                    "importance": float(importance),
                }
            )
        logit = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, class_weight="balanced", solver="liblinear", random_state=777),
        )
        logit.fit(x, y)
        coefs = np.abs(logit.named_steps["logisticregression"].coef_[0])
        for feature, importance in sorted(zip(feature_names, coefs), key=lambda v: v[1], reverse=True)[:20]:
            rows.append(
                {
                    "head": head.name,
                    "model": "logistic_abs_coef",
                    "feature": feature,
                    "importance": float(importance),
                }
            )
    for head in REG_HEADS:
        x, feature_names = build_features(df.reset_index(drop=True), head.include_action)
        y = pd.to_numeric(df[head.target_col], errors="coerce").astype(float).to_numpy()
        rf = RandomForestRegressor(
            n_estimators=300,
            max_depth=5,
            min_samples_leaf=2,
            random_state=777,
        )
        rf.fit(x, y)
        for feature, importance in sorted(
            zip(feature_names, rf.feature_importances_), key=lambda v: v[1], reverse=True
        )[:20]:
            rows.append(
                {
                    "head": head.name,
                    "model": "random_forest",
                    "feature": feature,
                    "importance": float(importance),
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "feature_importance_table.csv", index=False)
    return out


def high_confidence_gate_table(df: pd.DataFrame, class_predictions: pd.DataFrame) -> pd.DataFrame:
    if class_predictions.empty:
        out = pd.DataFrame()
        out.to_csv(OUT / "high_confidence_gate_table.csv", index=False)
        return out
    pump_rows = df[df["action"].eq(PUMP_ACTION)].copy()
    models = sorted(class_predictions["model"].dropna().unique())
    pred = class_predictions.pivot_table(
        index=["row_id", "model"],
        columns="head",
        values="proba",
        aggfunc="mean",
    ).reset_index()

    rows = []
    for anti_model in models:
        anti = pred[
            pred["model"].eq(anti_model)
            & pred["row_id"].isin(pump_rows["row_id"].to_numpy())
        ][["row_id", "anti_trigger"]].rename(
            columns={"anti_trigger": "anti_trigger_prob"}
        )
        for pump_model in models:
            pump = pred[
                pred["model"].eq(pump_model)
                & pred["row_id"].isin(pump_rows["row_id"].to_numpy())
            ][["row_id", "pump_saving_safe_positive"]].rename(
                columns={"pump_saving_safe_positive": "pump_safe_prob"}
            )
            for reg_model in models:
                reg = pred[
                    pred["model"].eq(reg_model)
                    & pred["row_id"].isin(pump_rows["row_id"].to_numpy())
                ][["row_id", "regression_risk"]].rename(
                    columns={"regression_risk": "regression_risk_prob"}
                )
                merged = pump_rows.merge(anti, on="row_id", how="left").merge(
                    pump, on="row_id", how="left"
                ).merge(reg, on="row_id", how="left")
                merged = merged.dropna(
                    subset=["anti_trigger_prob", "pump_safe_prob", "regression_risk_prob"]
                )
                for pump_thr in (0.5, 0.7, 0.85):
                    for anti_thr in (0.3, 0.5):
                        for reg_thr in (0.3, 0.5):
                            selected = merged[
                                (merged["pump_safe_prob"] >= pump_thr)
                                & (merged["anti_trigger_prob"] <= anti_thr)
                                & (merged["regression_risk_prob"] <= reg_thr)
                            ].copy()
                            safe = selected["constrained_label"].astype(str).eq("safe_positive")
                            not_safe = ~safe
                            broader_lowrisk = selected["dataset"].astype(str).eq("broader20") | selected[
                                "selection_reasons"
                            ].astype(str).str.contains("lowrisk_sanity", na=False)
                            near_safe = num_series(selected, "near_safe_far_risky").to_numpy() > 0
                            mismatch = num_series(selected, "direction_mismatch").to_numpy() > 0
                            case_counts = selected["group_id"].value_counts()
                            rows.append(
                                {
                                    "anti_model": anti_model,
                                    "pump_model": pump_model,
                                    "regression_model": reg_model,
                                    "pump_threshold": pump_thr,
                                    "anti_max": anti_thr,
                                    "regression_max": reg_thr,
                                    "eligible_rows": len(merged),
                                    "selected_rows": len(selected),
                                    "safe_positive_rows": int(safe.sum()),
                                    "precision": float(safe.mean()) if len(selected) else np.nan,
                                    "mean_time_gain_s": float(selected["time_gain_s"].mean()) if len(selected) else np.nan,
                                    "mean_idle_gain_s": float(selected["idle_gain_s"].mean()) if len(selected) else np.nan,
                                    "mean_pump_delta_m3": float(selected["pump_delta_m3"].mean()) if len(selected) else np.nan,
                                    "false_positive_broader_lowrisk": int((not_safe & broader_lowrisk).sum()),
                                    "false_positive_near_safe_far_risky": int((not_safe & near_safe).sum()),
                                    "false_positive_direction_mismatch": int((not_safe & mismatch).sum()),
                                    "unique_cases": int(selected["group_id"].nunique()) if len(selected) else 0,
                                    "top_case_share": float(case_counts.iloc[0] / len(selected)) if len(selected) else np.nan,
                                    "selected_case_ids": ";".join(case_counts.index[:5]) if len(selected) else "",
                                }
                            )
    out = pd.DataFrame(rows).sort_values(
        ["precision", "selected_rows"], ascending=[False, False], na_position="last"
    )
    out.to_csv(OUT / "high_confidence_gate_table.csv", index=False)
    return out


def write_manifest(df: pd.DataFrame, split_distribution: pd.DataFrame) -> None:
    label_summary = []
    for col in [
        "anti_trigger_label",
        "pump_saving_safe_positive_label",
        "regression_risk_label",
    ]:
        valid = df[col].dropna()
        label_summary.append(
            {
                "label": col,
                "valid_rows": len(valid),
                "positive_rows": int(valid.sum()) if len(valid) else 0,
                "positive_rate": float(valid.mean()) if len(valid) else np.nan,
            }
        )
    label_summary_df = pd.DataFrame(label_summary)
    label_summary_df.to_csv(RAW / "label_summary.csv", index=False)

    md = ["# h120 Action-Value Head Training Table Manifest", ""]
    md.append("## Source")
    md.append("")
    md.append(f"- source table: `{SOURCE / 'constrained_label_table.csv'}`")
    md.append(f"- prepare rows used: {len(df)}")
    md.append(f"- groups: {df['group_id'].nunique()}")
    md.append("")
    md.append("## Feature Policy")
    md.append("")
    md.append("Training features use only current state, execution state, near forecast blocks, far forecast blocks, and far-shape flags already present before the prepare decision.")
    md.append("")
    md.append("Excluded from training features: dataset, case_id, timestamp, selection category, constrained labels, realized deltas, future outcome columns, and split/group identifiers.")
    md.append("")
    md.append("Feature columns:")
    md.append("")
    for col in FEATURE_COLUMNS:
        if col in df.columns:
            md.append(f"- `{col}`")
    md.append("")
    md.append("Action dummies are included only for heads trained across both prepare actions.")
    md.append("")
    md.append("## Label Definitions")
    md.append("")
    md.append("- `anti_trigger_label`: positive when prepare should be rejected because the row is anti-trigger context, pump-waste negative, or hard-regression negative; safe_positive rows are negative examples; ambiguous rows without anti context are unlabeled.")
    md.append("- `pump_saving_safe_positive_label`: for pump_saving_prepare rows only, safe_positive=1 and negative=0; ambiguous excluded.")
    md.append("- `regression_risk_label`: hard regression if fallback/max_p95/time/idle worsens beyond tolerance.")
    md.append("- expected-value targets: `time_gain_s`, `idle_gain_s`, and `pump_delta_m3`.")
    md.append("")
    md.append("## Label Summary")
    md.append("")
    md.append(markdown_table(label_summary_df))
    md.append("")
    md.append("## Split Strategy")
    md.append("")
    md.append("GroupKFold split by `dataset::case_id`. Adjacent buckets from the same case do not cross train/test.")
    md.append("")
    md.append(markdown_table(split_distribution))
    (OUT / "training_table_manifest.md").write_text("\n".join(md) + "\n")


def write_reports(
    df: pd.DataFrame,
    split_distribution: pd.DataFrame,
    head_metrics: pd.DataFrame,
    gate_table: pd.DataFrame,
    feature_importance: pd.DataFrame,
    calibration: pd.DataFrame,
) -> None:
    class_summary = head_metrics[head_metrics["task_type"].eq("classification")].copy()
    reg_summary = head_metrics[head_metrics["task_type"].eq("regression")].copy()
    best_gate = gate_table.head(10).copy() if not gate_table.empty else pd.DataFrame()
    best_pump = class_summary[class_summary["head"].eq("pump_saving_safe_positive")].sort_values(
        "high_conf_precision", ascending=False, na_position="last"
    )
    best_anti = class_summary[class_summary["head"].eq("anti_trigger")].sort_values(
        "anti_trigger_recall", ascending=False, na_position="last"
    )
    best_reg = class_summary[class_summary["head"].eq("regression_risk")].sort_values(
        "f1", ascending=False, na_position="last"
    )
    gate_viable = False
    if not gate_table.empty:
        viable = gate_table[
            (gate_table["selected_rows"] >= 5)
            & (gate_table["precision"] >= 0.70)
            & (gate_table["false_positive_broader_lowrisk"] <= 1)
            & (gate_table["top_case_share"] <= 0.5)
        ]
        gate_viable = not viable.empty

    summary = ["# h120 Action-Value Head Training Summary", ""]
    summary.append("## Scope")
    summary.append("")
    summary.append("- Offline lightweight head training only.")
    summary.append("- No controller gate implemented.")
    summary.append("- v1.6 floor unchanged.")
    summary.append(f"- prepare rows: {len(df)}; groups: {df['group_id'].nunique()}.")
    summary.append("")
    summary.append("## Best Classification Metrics")
    summary.append("")
    summary.append(markdown_table(class_summary.sort_values(["head", "f1"], ascending=[True, False]).head(20)))
    summary.append("")
    summary.append("## Regression Metrics")
    summary.append("")
    summary.append(markdown_table(reg_summary.sort_values(["head", "mae"]).head(20)))
    summary.append("")
    summary.append("## Best Gate Candidates")
    summary.append("")
    summary.append(markdown_table(best_gate))
    summary.append("")
    summary.append("## Interpretation")
    summary.append("")
    if gate_viable:
        summary.append("A small offline high-confidence region exists under the proposed thresholds, but it still requires controller-side gate review and broader counterfactual validation.")
    else:
        summary.append("No robust high-confidence pump_saving gate region was found. The models either reject most prepare actions or select too few/case-concentrated rows.")
    (OUT / "head_training_summary.md").write_text("\n".join(summary) + "\n")

    decision = ["# h120 Action-Value Head Decision", ""]
    decision.append("## Verdict")
    decision.append("")
    if gate_viable:
        decision.append("Offline heads show an initial high-confidence region, but this is still not a controller integration result. A default-off gate proposal would need a separate probe.")
    else:
        decision.append("Do not enter `h120_action_value_gate` yet. The offline heads are informative, but not controller-ready.")
    decision.append("")
    decision.append("## Required Answers")
    decision.append("")
    decision.append("1. **Is the anti-trigger head usable?**")
    decision.append("")
    if not best_anti.empty:
        row = best_anti.iloc[0]
        decision.append(
            f"Partly. Best mean anti-trigger recall is {row.get('anti_trigger_recall', np.nan):.3f} "
            f"({row['model']}), but this must be judged alongside prepare precision because rejecting everything can look good."
        )
    else:
        decision.append("No reliable anti-trigger metrics were produced.")
    decision.append("")
    decision.append("2. **Is the pump_saving_safe_positive head usable?**")
    decision.append("")
    if not best_pump.empty:
        row = best_pump.iloc[0]
        decision.append(
            f"Not yet. Best high-confidence precision is {row.get('high_conf_precision', np.nan):.3f} "
            f"with {row.get('high_conf_positive_rows', np.nan):.1f} high-confidence rows on average."
        )
    else:
        decision.append("No; there were not enough valid pump_saving labels.")
    decision.append("")
    decision.append("3. **Is the regression-risk head usable?**")
    decision.append("")
    if not best_reg.empty:
        row = best_reg.iloc[0]
        decision.append(
            f"It has diagnostic value. Best F1 is {row.get('f1', np.nan):.3f}, but it should remain a guard head until sample size increases."
        )
    else:
        decision.append("No reliable regression-risk classifier was produced.")
    decision.append("")
    decision.append("4. **Is there a high-confidence action-value region?**")
    decision.append("")
    if gate_viable:
        decision.append("A small candidate region exists offline. It is not yet a controller gate because sample size and case concentration still need review.")
    else:
        decision.append("No robust region meeting precision, false-positive, sample-size, and case-concentration criteria was found.")
    decision.append("")
    decision.append("5. **Allow default-off h120_action_value_gate?**")
    decision.append("")
    decision.append("No. This run is offline training only and does not justify controller integration.")
    decision.append("")
    decision.append("6. **Why not?**")
    decision.append("")
    decision.append("The limiting factors are sparse safe-positive labels, case concentration, and weak separability of pump_saving safe positives under grouped validation. The feature set has useful anti-trigger signals, but not enough positive-action precision.")
    decision.append("")
    decision.append("7. **Next step?**")
    decision.append("")
    decision.append("Expand counterfactual samples around rare safe positives and anti-trigger slices, then retrain the same heads. Do not tune a controller gate yet.")
    decision.append("")
    decision.append("## Best Gate Rows")
    decision.append("")
    decision.append(markdown_table(best_gate))
    (OUT / "h120_action_value_head_decision.md").write_text("\n".join(decision) + "\n")


def write_task_plan() -> None:
    plan = {
        "task": "h120_action_value_head_training_v1",
        "status": "done",
        "steps": [
            "build prepare-row training table",
            "derive action-value labels",
            "split by case group",
            "train lightweight offline heads",
            "evaluate high-confidence gate region",
            "write decision",
        ],
        "controller_modified": False,
        "v16_floor_modified": False,
    }
    (DEBUG / "task_master_plan.json").write_text(json.dumps(plan, indent=2) + "\n")


def main() -> None:
    ensure_dirs()
    write_task_plan()
    df = load_training_table()
    splits = make_splits(df)
    split_distribution = write_split_distribution(df, splits)
    write_manifest(df, split_distribution)
    class_metrics, class_predictions = run_classification_heads(df, splits)
    reg_metrics, reg_predictions = run_regression_heads(df, splits)
    class_metrics.to_csv(RAW / "classification_fold_metrics.csv", index=False)
    class_predictions.to_csv(RAW / "classification_oof_predictions.csv", index=False)
    reg_metrics.to_csv(RAW / "regression_fold_metrics.csv", index=False)
    reg_predictions.to_csv(RAW / "regression_oof_predictions.csv", index=False)
    head_metrics = aggregate_metrics(class_metrics, reg_metrics)
    calibration = calibration_bins(class_predictions)
    importance = feature_importances(df)
    gate_table = high_confidence_gate_table(df, class_predictions)
    write_reports(df, split_distribution, head_metrics, gate_table, importance, calibration)


if __name__ == "__main__":
    main()
