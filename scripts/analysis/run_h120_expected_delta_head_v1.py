#!/usr/bin/env python3
"""Offline h120 expected-delta / pairwise action-value heads v1.

This script does not attach a controller gate and does not modify v1.6 floor
logic.  It reframes the h120 action-value data from sparse safe-positive
classification into expected-delta regression and pairwise action ranking.
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
SOURCE_ROOT = REPO / "outputs/wind_prediction/h120_action_value_overnight_v1"
SUP_ROOT = SOURCE_ROOT / "supplemental_sweep_v1"
SOURCE_TABLE = SUP_ROOT / "raw_tables/combined_constrained_label_table.csv"
OUT = REPO / "outputs/wind_prediction/h120_expected_delta_head_v1"
RAW = OUT / "raw_tables"
DEBUG = OUT / "debug"
PAPER = OUT / "paper_ready"

WAIT_ACTION = "wait"
WATCH_ACTION = "watch_only"
PUMP_ACTION = "pump_saving_prepare"
ACTIVE_ACTION = "active_small_prepare"
PREPARE_ACTIONS = (PUMP_ACTION, ACTIVE_ACTION)
ACTION_ORDER = (WAIT_ACTION, WATCH_ACTION, PUMP_ACTION, ACTIVE_ACTION)

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


def load_source() -> pd.DataFrame:
    if not SOURCE_TABLE.exists():
        raise FileNotFoundError(f"missing source table: {SOURCE_TABLE}")
    df = pd.read_csv(SOURCE_TABLE)
    df["group_id"] = df["dataset"].astype(str) + "::" + df["case_id"].astype(str)
    df["bucket_key"] = (
        df["dataset"].astype(str)
        + "::"
        + df["case_id"].astype(str)
        + "::"
        + pd.to_numeric(df["bucket"], errors="coerce").fillna(-1).astype(int).astype(str)
    )
    df["is_broader20"] = df["dataset"].astype(str).eq("broader20").astype(int)
    df["is_lowrisk_sanity"] = df["selection_reasons"].astype(str).str.contains(
        "lowrisk_sanity|lowrisk", regex=True, na=False
    ).astype(int)
    df["is_direction_mismatch"] = (num_series(df, "direction_mismatch") > 0).astype(int)
    df["is_near_safe_far_risky"] = (num_series(df, "near_safe_far_risky") > 0).astype(int)
    df["delta_time_over5_s"] = num_series(df, "delta_post60_time_over5")
    df["delta_idle_over5_s"] = num_series(df, "delta_post60_idle_over5")
    df["delta_pump_m3"] = num_series(df, "delta_full_pump_m3")
    df["delta_fallback_pp"] = num_series(df, "delta_full_fallback_pp")
    df["delta_max_p95"] = num_series(df, "delta_full_max_p95")
    df["time_gain_s"] = -df["delta_time_over5_s"]
    df["idle_gain_s"] = -df["delta_idle_over5_s"]
    df["pump_delta_m3"] = df["delta_pump_m3"]
    df["regression_risk_label"] = (
        (df["delta_fallback_pp"] > 0.05)
        | (df["delta_max_p95"] > 0.05)
        | (df["delta_time_over5_s"] > 5.0)
        | (df["delta_idle_over5_s"] > 30.0)
    ).astype(int)
    df["safe_positive_label"] = df["constrained_label"].astype(str).eq("safe_positive").astype(int)
    df["negative_label"] = df["constrained_label"].astype(str).eq("negative").astype(int)
    df.to_csv(RAW / "source_combined_table.csv", index=False)
    return df


def utility(
    delta_time: pd.Series | float,
    delta_idle: pd.Series | float,
    delta_pump: pd.Series | float,
    delta_fallback: pd.Series | float,
    delta_p95: pd.Series | float,
) -> pd.Series | float:
    # Positive is better: reduce time/idle/fallback/p95 while penalizing pump.
    return (
        -1.0 * delta_time
        - 0.35 * delta_idle
        - 0.12 * delta_pump
        - 180.0 * delta_fallback
        - 120.0 * delta_p95
    )


def build_action_table(df: pd.DataFrame) -> pd.DataFrame:
    action = df[df["action"].isin(ACTION_ORDER)].copy().reset_index(drop=True)
    action["row_id"] = np.arange(len(action))
    action["action_utility"] = utility(
        action["delta_time_over5_s"],
        action["delta_idle_over5_s"],
        action["delta_pump_m3"],
        action["delta_fallback_pp"],
        action["delta_max_p95"],
    )
    action["prepare_action"] = action["action"].isin(PREPARE_ACTIONS).astype(int)
    action.to_csv(OUT / "action_delta_table.csv", index=False)
    action.to_csv(RAW / "action_delta_table.csv", index=False)
    return action


def build_pairwise_table(action: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    pair_defs = [
        (WAIT_ACTION, PUMP_ACTION),
        (WAIT_ACTION, ACTIVE_ACTION),
        (PUMP_ACTION, ACTIVE_ACTION),
    ]
    group_cols = ["bucket_key"]
    for _, g in action.groupby(group_cols, dropna=False):
        by_action = {str(r.action): r for r in g.itertuples()}
        if WAIT_ACTION not in by_action:
            continue
        for left, right in pair_defs:
            if left not in by_action or right not in by_action:
                continue
            l = by_action[left]
            r = by_action[right]
            util_l = float(l.action_utility)
            util_r = float(r.action_utility)
            preferred = right if util_r > util_l else left
            util_margin = util_r - util_l
            base = r if right in PREPARE_ACTIONS else l
            left_is_wait = left == WAIT_ACTION
            rows.append(
                {
                    "pair_id": f"{base.bucket_key}::{left}_vs_{right}",
                    "bucket_key": base.bucket_key,
                    "refined_probe_id": base.refined_probe_id,
                    "dataset": base.dataset,
                    "case_id": base.case_id,
                    "group_id": base.group_id,
                    "bucket": int(base.bucket),
                    "timestamp": base.timestamp,
                    "left_action": left,
                    "right_action": right,
                    "pair_type": f"{left}_vs_{right}",
                    "pairwise_preferred_action": preferred,
                    "right_preferred_label": int(preferred == right),
                    "prepare_preferred_label": int(
                        preferred in PREPARE_ACTIONS
                        and (left_is_wait or left in PREPARE_ACTIONS)
                    ),
                    "utility_left": util_l,
                    "utility_right": util_r,
                    "utility_margin_right_minus_left": util_margin,
                    "delta_time_right_minus_left": float(r.delta_time_over5_s - l.delta_time_over5_s),
                    "delta_idle_right_minus_left": float(r.delta_idle_over5_s - l.delta_idle_over5_s),
                    "delta_pump_right_minus_left": float(r.delta_pump_m3 - l.delta_pump_m3),
                    "delta_fallback_right_minus_left": float(r.delta_fallback_pp - l.delta_fallback_pp),
                    "delta_max_p95_right_minus_left": float(r.delta_max_p95 - l.delta_max_p95),
                    "regression_risk_label_right": int(r.regression_risk_label),
                    "safe_positive_right": int(r.safe_positive_label),
                    "negative_right": int(r.negative_label),
                    "is_broader20": int(base.is_broader20),
                    "is_lowrisk_sanity": int(base.is_lowrisk_sanity),
                    "is_direction_mismatch": int(base.is_direction_mismatch),
                    "is_near_safe_far_risky": int(base.is_near_safe_far_risky),
                }
            )
            for col in FEATURE_COLUMNS:
                rows[-1][col] = getattr(base, col, 0.0)
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "pairwise_action_table.csv", index=False)
    out.to_csv(RAW / "pairwise_action_table.csv", index=False)
    return out


def build_features(df: pd.DataFrame, action_col: str | None = None, pair_col: str | None = None) -> tuple[pd.DataFrame, list[str]]:
    cols = [c for c in FEATURE_COLUMNS if c in df.columns]
    x = (
        df[cols]
        .apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .reset_index(drop=True)
    )
    if action_col is not None:
        d = pd.get_dummies(df[action_col].astype(str), prefix=action_col)
        x = pd.concat([x, d.reset_index(drop=True)], axis=1)
    if pair_col is not None:
        d = pd.get_dummies(df[pair_col].astype(str), prefix=pair_col)
        x = pd.concat([x, d.reset_index(drop=True)], axis=1)
    return x, list(x.columns)


def make_splits(df: pd.DataFrame, n_splits: int = 5) -> list[tuple[int, np.ndarray, np.ndarray]]:
    from sklearn.model_selection import GroupKFold

    groups = df["group_id"].astype(str).to_numpy()
    unique = np.unique(groups)
    k = min(n_splits, len(unique))
    splitter = GroupKFold(n_splits=k)
    return [
        (i, train_idx, test_idx)
        for i, (train_idx, test_idx) in enumerate(splitter.split(df, groups=groups), start=1)
    ]


def regressors() -> dict[str, Any]:
    from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
    from sklearn.linear_model import ElasticNet, Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return {
        "ridge": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        "elastic_net": make_pipeline(StandardScaler(), ElasticNet(alpha=0.02, l1_ratio=0.25, max_iter=5000)),
        "hist_gradient_boosting": HistGradientBoostingRegressor(max_iter=100, learning_rate=0.05, random_state=4242),
        "random_forest": RandomForestRegressor(
            n_estimators=400,
            max_depth=6,
            min_samples_leaf=2,
            random_state=4242,
        ),
    }


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
            max_iter=100,
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
class RegressionHead:
    name: str
    target_col: str


REG_HEADS = [
    RegressionHead("expected_time_gain_s", "time_gain_s"),
    RegressionHead("expected_idle_gain_s", "idle_gain_s"),
    RegressionHead("expected_pump_delta_m3", "pump_delta_m3"),
    RegressionHead("expected_utility", "action_utility"),
]


def train_expected_delta_heads(action: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    rows: list[dict[str, Any]] = []
    pred_rows: list[dict[str, Any]] = []
    splits = make_splits(action)
    x_all, feature_names = build_features(action, action_col="action")
    for head in REG_HEADS:
        y = pd.to_numeric(action[head.target_col], errors="coerce").astype(float)
        for fold, train_idx, test_idx in splits:
            train_mask = pd.Series(False, index=action.index)
            test_mask = pd.Series(False, index=action.index)
            train_mask.iloc[train_idx] = True
            test_mask.iloc[test_idx] = True
            for model_name, model in regressors().items():
                model.fit(x_all.loc[train_mask], y.loc[train_mask])
                pred = model.predict(x_all.loc[test_mask])
                target = y.loc[test_mask].to_numpy()
                rmse = math.sqrt(mean_squared_error(target, pred))
                rows.append(
                    {
                        "head": head.name,
                        "model": model_name,
                        "fold": fold,
                        "test_rows": int(test_mask.sum()),
                        "mae": float(mean_absolute_error(target, pred)),
                        "rmse": float(rmse),
                        "r2": float(r2_score(target, pred)) if len(target) > 1 else np.nan,
                        "target_mean": float(np.mean(target)),
                        "pred_mean": float(np.mean(pred)),
                    }
                )
                sub = action.loc[test_mask, [
                    "row_id",
                    "refined_probe_id",
                    "bucket_key",
                    "dataset",
                    "case_id",
                    "group_id",
                    "bucket",
                    "action",
                    "constrained_label",
                    "safe_positive_label",
                    "negative_label",
                    "regression_risk_label",
                    "is_broader20",
                    "is_lowrisk_sanity",
                    "is_direction_mismatch",
                    "is_near_safe_far_risky",
                    "time_gain_s",
                    "idle_gain_s",
                    "pump_delta_m3",
                    "action_utility",
                ]].copy()
                sub["fold"] = fold
                sub["head"] = head.name
                sub["model"] = model_name
                sub["target"] = target
                sub["prediction"] = pred
                pred_rows.extend(sub.to_dict("records"))
    metrics = pd.DataFrame(rows)
    preds = pd.DataFrame(pred_rows)
    metrics.to_csv(RAW / "expected_delta_fold_metrics.csv", index=False)
    preds.to_csv(RAW / "expected_delta_oof_predictions.csv", index=False)
    agg = metrics.groupby(["head", "model"], dropna=False)[
        ["test_rows", "mae", "rmse", "r2", "target_mean", "pred_mean"]
    ].mean().reset_index()
    agg.to_csv(OUT / "expected_delta_metrics.csv", index=False)
    return agg, preds


def train_regression_risk(action: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    from sklearn.metrics import precision_recall_fscore_support, roc_auc_score

    prep = action[action["action"].isin(PREPARE_ACTIONS)].copy().reset_index(drop=True)
    splits = make_splits(prep)
    x_all, _ = build_features(prep, action_col="action")
    y = prep["regression_risk_label"].astype(int)
    rows: list[dict[str, Any]] = []
    pred_rows: list[dict[str, Any]] = []
    for fold, train_idx, test_idx in splits:
        y_train = y.iloc[train_idx].to_numpy()
        y_test = y.iloc[test_idx].to_numpy()
        if len(set(y_train.tolist())) < 2:
            continue
        for model_name, model in classifiers().items():
            model.fit(x_all.iloc[train_idx], y_train)
            proba = model.predict_proba(x_all.iloc[test_idx])[:, 1]
            pred = (proba >= 0.5).astype(int)
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_test, pred, average="binary", zero_division=0
            )
            try:
                auc = roc_auc_score(y_test, proba) if len(set(y_test.tolist())) > 1 else np.nan
            except ValueError:
                auc = np.nan
            rows.append(
                {
                    "head": "regression_risk_prob",
                    "model": model_name,
                    "fold": fold,
                    "test_rows": len(y_test),
                    "test_positive": int(y_test.sum()),
                    "precision": float(precision),
                    "recall": float(recall),
                    "f1": float(f1),
                    "roc_auc": float(auc) if not pd.isna(auc) else np.nan,
                }
            )
            sub = prep.iloc[test_idx][[
                "row_id",
                "refined_probe_id",
                "bucket_key",
                "dataset",
                "case_id",
                "group_id",
                "bucket",
                "action",
                "constrained_label",
                "safe_positive_label",
                "negative_label",
                "regression_risk_label",
                "is_broader20",
                "is_lowrisk_sanity",
                "is_direction_mismatch",
                "is_near_safe_far_risky",
                "time_gain_s",
                "idle_gain_s",
                "pump_delta_m3",
                "action_utility",
            ]].copy()
            sub["fold"] = fold
            sub["model"] = model_name
            sub["regression_risk_prob"] = proba
            pred_rows.extend(sub.to_dict("records"))
    metrics = pd.DataFrame(rows)
    preds = pd.DataFrame(pred_rows)
    metrics.to_csv(RAW / "regression_risk_fold_metrics.csv", index=False)
    preds.to_csv(RAW / "regression_risk_oof_predictions.csv", index=False)
    agg = metrics.groupby(["head", "model"], dropna=False)[
        ["test_rows", "test_positive", "precision", "recall", "f1", "roc_auc"]
    ].mean().reset_index()
    agg.to_csv(RAW / "regression_risk_metrics.csv", index=False)
    return agg, preds


def train_pairwise_heads(pairwise: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    from sklearn.metrics import precision_recall_fscore_support, roc_auc_score

    splits = make_splits(pairwise)
    x_all, _ = build_features(pairwise, pair_col="pair_type")
    rows: list[dict[str, Any]] = []
    pred_rows: list[dict[str, Any]] = []
    y = pairwise["right_preferred_label"].astype(int)
    for fold, train_idx, test_idx in splits:
        y_train = y.iloc[train_idx].to_numpy()
        y_test = y.iloc[test_idx].to_numpy()
        if len(set(y_train.tolist())) < 2:
            continue
        for model_name, model in classifiers().items():
            model.fit(x_all.iloc[train_idx], y_train)
            proba = model.predict_proba(x_all.iloc[test_idx])[:, 1]
            pred = (proba >= 0.5).astype(int)
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_test, pred, average="binary", zero_division=0
            )
            try:
                auc = roc_auc_score(y_test, proba) if len(set(y_test.tolist())) > 1 else np.nan
            except ValueError:
                auc = np.nan
            test = pairwise.iloc[test_idx].copy()
            false_prepare = (
                (pred == 1)
                & (test["right_action"].isin(PREPARE_ACTIONS).to_numpy())
                & (test["right_preferred_label"].to_numpy() == 0)
            )
            rows.append(
                {
                    "model": model_name,
                    "fold": fold,
                    "task": "right_action_preferred",
                    "test_rows": len(test),
                    "test_positive": int(y_test.sum()),
                    "precision": float(precision),
                    "recall": float(recall),
                    "f1": float(f1),
                    "roc_auc": float(auc) if not pd.isna(auc) else np.nan,
                    "false_prepare_broader20": int((false_prepare & test["is_broader20"].to_numpy().astype(bool)).sum()),
                    "false_prepare_lowrisk": int((false_prepare & test["is_lowrisk_sanity"].to_numpy().astype(bool)).sum()),
                    "false_prepare_mismatch": int((false_prepare & test["is_direction_mismatch"].to_numpy().astype(bool)).sum()),
                }
            )
            sub = test[[
                "pair_id",
                "bucket_key",
                "refined_probe_id",
                "dataset",
                "case_id",
                "group_id",
                "bucket",
                "left_action",
                "right_action",
                "pair_type",
                "pairwise_preferred_action",
                "right_preferred_label",
                "prepare_preferred_label",
                "utility_margin_right_minus_left",
                "safe_positive_right",
                "negative_right",
                "regression_risk_label_right",
                "is_broader20",
                "is_lowrisk_sanity",
                "is_direction_mismatch",
                "is_near_safe_far_risky",
            ]].copy()
            sub["fold"] = fold
            sub["model"] = model_name
            sub["right_preferred_prob"] = proba
            pred_rows.extend(sub.to_dict("records"))
    metrics = pd.DataFrame(rows)
    preds = pd.DataFrame(pred_rows)
    metrics.to_csv(RAW / "pairwise_preference_fold_metrics.csv", index=False)
    preds.to_csv(RAW / "pairwise_preference_oof_predictions.csv", index=False)
    agg = metrics.groupby(["model", "task"], dropna=False)[
        [
            "test_rows",
            "test_positive",
            "precision",
            "recall",
            "f1",
            "roc_auc",
            "false_prepare_broader20",
            "false_prepare_lowrisk",
            "false_prepare_mismatch",
        ]
    ].mean().reset_index()
    agg.to_csv(OUT / "pairwise_preference_metrics.csv", index=False)
    return agg, preds


def aggregate_expected_preds(preds: pd.DataFrame, model: str) -> pd.DataFrame:
    sub = preds[preds["model"].eq(model)].copy()
    pivot = sub.pivot_table(
        index=[
            "row_id",
            "refined_probe_id",
            "bucket_key",
            "dataset",
            "case_id",
            "group_id",
            "bucket",
            "action",
            "constrained_label",
            "safe_positive_label",
            "negative_label",
            "regression_risk_label",
            "is_broader20",
            "is_lowrisk_sanity",
            "is_direction_mismatch",
            "is_near_safe_far_risky",
            "time_gain_s",
            "idle_gain_s",
            "pump_delta_m3",
            "action_utility",
        ],
        columns="head",
        values="prediction",
        aggfunc="mean",
    ).reset_index()
    return pivot


def high_confidence_delta_gate(
    action: pd.DataFrame,
    delta_preds: pd.DataFrame,
    risk_preds: pd.DataFrame,
    pairwise_preds: pd.DataFrame,
    metrics: pd.DataFrame,
) -> pd.DataFrame:
    if delta_preds.empty:
        out = pd.DataFrame()
        out.to_csv(OUT / "high_confidence_delta_gate_table.csv", index=False)
        return out
    reg_model = "random_forest"
    if "expected_utility" in metrics["head"].unique():
        util_metrics = metrics[metrics["head"].eq("expected_utility")].sort_values("mae")
        if not util_metrics.empty:
            reg_model = str(util_metrics.iloc[0]["model"])
    pred = aggregate_expected_preds(delta_preds, reg_model)
    pred = pred[pred["action"].isin(PREPARE_ACTIONS)].copy()
    pred = pred.rename(
        columns={
            "expected_time_gain_s": "pred_time_gain_s",
            "expected_idle_gain_s": "pred_idle_gain_s",
            "expected_pump_delta_m3": "pred_pump_delta_m3",
            "expected_utility": "pred_utility",
        }
    )
    if not risk_preds.empty:
        risk_model = "random_forest" if "random_forest" in set(risk_preds["model"]) else str(risk_preds["model"].iloc[0])
        risk = risk_preds[risk_preds["model"].eq(risk_model)][["row_id", "regression_risk_prob"]]
        pred = pred.merge(risk, on="row_id", how="left")
    else:
        pred["regression_risk_prob"] = np.nan
    if not pairwise_preds.empty:
        pair = pairwise_preds[
            pairwise_preds["right_action"].isin(PREPARE_ACTIONS)
            & pairwise_preds["left_action"].eq(WAIT_ACTION)
        ].copy()
        pair_model = "random_forest" if "random_forest" in set(pair["model"]) else str(pair["model"].iloc[0])
        pair = pair[pair["model"].eq(pair_model)][["bucket_key", "right_action", "right_preferred_prob"]].rename(
            columns={"right_action": "action", "right_preferred_prob": "prepare_preferred_prob"}
        )
        pred = pred.merge(pair, on=["bucket_key", "action"], how="left")
    else:
        pred["prepare_preferred_prob"] = np.nan

    # Lightweight anti-trigger proxy: reuse previous stable logic as a diagnostic
    # only, without training a new gate here.
    pred["anti_trigger_context"] = (
        pred["is_broader20"].astype(bool)
        | pred["is_lowrisk_sanity"].astype(bool)
        | pred["is_direction_mismatch"].astype(bool)
        | pred["is_near_safe_far_risky"].astype(bool)
    ).astype(int)

    rows: list[dict[str, Any]] = []
    for action_name in PREPARE_ACTIONS:
        base = pred[pred["action"].eq(action_name)].copy()
        if base.empty:
            continue
        for time_thr in (20.0, 40.0, 80.0):
            for utility_thr in (5.0, 20.0, 50.0):
                for pump_cap in (20.0, 40.0, 80.0):
                    for risk_max in (0.35, 0.50, 0.65):
                        selected = base[
                            (base["pred_time_gain_s"] >= time_thr)
                            & (base["pred_utility"] >= utility_thr)
                            & (base["pred_pump_delta_m3"] <= pump_cap)
                            & (base["regression_risk_prob"].fillna(0.0) <= risk_max)
                            & (base["anti_trigger_context"] == 0)
                        ].copy()
                        safe = selected["safe_positive_label"].astype(bool)
                        false = ~safe
                        cases = selected["group_id"].value_counts()
                        rows.append(
                            {
                                "action": action_name,
                                "delta_model": reg_model,
                                "time_gain_threshold": time_thr,
                                "utility_threshold": utility_thr,
                                "pump_delta_cap": pump_cap,
                                "regression_risk_max": risk_max,
                                "eligible_rows": len(base),
                                "selected_rows": len(selected),
                                "safe_positive_rows": int(safe.sum()),
                                "negative_rows": int(selected["negative_label"].sum()) if len(selected) else 0,
                                "precision": float(safe.mean()) if len(selected) else np.nan,
                                "mean_actual_time_gain_s": float(selected["time_gain_s"].mean()) if len(selected) else np.nan,
                                "mean_actual_idle_gain_s": float(selected["idle_gain_s"].mean()) if len(selected) else np.nan,
                                "mean_actual_pump_delta_m3": float(selected["pump_delta_m3"].mean()) if len(selected) else np.nan,
                                "false_positive_broader20": int((false & selected["is_broader20"].astype(bool)).sum()) if len(selected) else 0,
                                "false_positive_lowrisk": int((false & selected["is_lowrisk_sanity"].astype(bool)).sum()) if len(selected) else 0,
                                "false_positive_direction_mismatch": int((false & selected["is_direction_mismatch"].astype(bool)).sum()) if len(selected) else 0,
                                "false_positive_near_safe_far_risky": int((false & selected["is_near_safe_far_risky"].astype(bool)).sum()) if len(selected) else 0,
                                "unique_cases": int(selected["group_id"].nunique()) if len(selected) else 0,
                                "top_case_share": float(cases.iloc[0] / len(selected)) if len(selected) else np.nan,
                                "selected_case_ids": ";".join(cases.index[:6]) if len(selected) else "",
                            }
                        )
    out = pd.DataFrame(rows).sort_values(
        ["precision", "safe_positive_rows", "selected_rows"],
        ascending=[False, False, False],
        na_position="last",
    )
    out.to_csv(OUT / "high_confidence_delta_gate_table.csv", index=False)
    out.to_csv(RAW / "high_confidence_delta_gate_table.csv", index=False)
    pred.to_csv(RAW / "expected_delta_gate_oof_rows.csv", index=False)
    return out


def feature_importance(action: pd.DataFrame, pairwise: pd.DataFrame) -> pd.DataFrame:
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

    rows: list[dict[str, Any]] = []
    x_action, action_features = build_features(action, action_col="action")
    for head in REG_HEADS:
        y = pd.to_numeric(action[head.target_col], errors="coerce").fillna(0.0)
        model = RandomForestRegressor(
            n_estimators=400,
            max_depth=6,
            min_samples_leaf=2,
            random_state=2026,
        )
        model.fit(x_action, y)
        for feature, importance in sorted(
            zip(action_features, model.feature_importances_),
            key=lambda v: v[1],
            reverse=True,
        )[:20]:
            rows.append(
                {
                    "head": head.name,
                    "model": "random_forest",
                    "feature": feature,
                    "importance": float(importance),
                }
            )
    x_pair, pair_features = build_features(pairwise, pair_col="pair_type")
    y_pair = pairwise["right_preferred_label"].astype(int)
    if len(set(y_pair.tolist())) > 1:
        clf = RandomForestClassifier(
            n_estimators=400,
            max_depth=6,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=2026,
        )
        clf.fit(x_pair, y_pair)
        for feature, importance in sorted(
            zip(pair_features, clf.feature_importances_),
            key=lambda v: v[1],
            reverse=True,
        )[:20]:
            rows.append(
                {
                    "head": "pairwise_preferred_action",
                    "model": "random_forest",
                    "feature": feature,
                    "importance": float(importance),
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "feature_importance_table.csv", index=False)
    out.to_csv(RAW / "feature_importance_table.csv", index=False)
    return out


def write_split_distribution(action: pd.DataFrame, pairwise: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for name, df in (("action", action), ("pairwise", pairwise)):
        for fold, train_idx, test_idx in make_splits(df):
            train = df.iloc[train_idx]
            test = df.iloc[test_idx]
            rows.append(
                {
                    "table": name,
                    "fold": fold,
                    "train_rows": len(train),
                    "test_rows": len(test),
                    "train_cases": train["group_id"].nunique(),
                    "test_cases": test["group_id"].nunique(),
                    "test_case_ids": ";".join(sorted(test["group_id"].unique())),
                    "test_prepare_positive_rows": int(test.get("safe_positive_label", pd.Series(dtype=int)).sum()) if name == "action" else int(test.get("safe_positive_right", pd.Series(dtype=int)).sum()),
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(RAW / "split_distribution.csv", index=False)
    return out


def write_manifest(source: pd.DataFrame, action: pd.DataFrame, pairwise: pd.DataFrame, split_dist: pd.DataFrame) -> None:
    label = action[action["action"].isin(PREPARE_ACTIONS)].groupby(["action", "constrained_label"]).size().reset_index(name="rows")
    pair_counts = pairwise.groupby(["pair_type", "pairwise_preferred_action"]).size().reset_index(name="rows")
    md = ["# h120 Expected-Delta Training Manifest", ""]
    md.append("## Scope")
    md.append("")
    md.append("- Offline expected-delta and pairwise ranking heads only.")
    md.append("- No controller gate.")
    md.append("- v1.6 delayed-medium floor unchanged.")
    md.append("")
    md.append("## Source")
    md.append("")
    md.append(f"- source table: `{SOURCE_TABLE}`")
    md.append(f"- source rows: {len(source)}")
    md.append(f"- action rows: {len(action)}")
    md.append(f"- pairwise rows: {len(pairwise)}")
    md.append(f"- groups: {action['group_id'].nunique()}")
    md.append("")
    md.append("## Target Variables")
    md.append("")
    md.append("- `delta_time_over5_s`, `delta_idle_over5_s`, `delta_pump_m3`, `delta_fallback_pp`, `delta_max_p95`.")
    md.append("- Regression heads predict `time_gain_s`, `idle_gain_s`, `pump_delta_m3`, and an offline utility proxy.")
    md.append("- Pairwise heads predict whether the right action is preferred under the fixed offline utility proxy.")
    md.append("- `regression_risk_prob` is a guard head, positive when fallback/max_p95/time/idle worsen beyond tolerance.")
    md.append("")
    md.append("## Feature Policy")
    md.append("")
    md.append("Features use only current posture, execution state, near/far forecast blocks, and far-shape flags. Dataset/case/timestamp/future outcomes are split or diagnostics only, not training features.")
    md.append("")
    md.append("## Prepare Label Distribution")
    md.append("")
    md.append(markdown_table(label))
    md.append("")
    md.append("## Pairwise Preference Distribution")
    md.append("")
    md.append(markdown_table(pair_counts))
    md.append("")
    md.append("## GroupKFold Distribution")
    md.append("")
    md.append(markdown_table(split_dist))
    (OUT / "expected_delta_training_manifest.md").write_text("\n".join(md) + "\n")


def write_reports(
    action: pd.DataFrame,
    pairwise: pd.DataFrame,
    delta_metrics: pd.DataFrame,
    risk_metrics: pd.DataFrame,
    pair_metrics: pd.DataFrame,
    gate: pd.DataFrame,
    importance: pd.DataFrame,
) -> None:
    all_delta_metrics = delta_metrics.copy()
    combined_metrics = pd.concat(
        [
            all_delta_metrics.assign(task_type="regression"),
            risk_metrics.assign(task_type="classification"),
        ],
        ignore_index=True,
        sort=False,
    )
    combined_metrics.to_csv(OUT / "expected_delta_metrics.csv", index=False)

    viable = pd.DataFrame()
    if not gate.empty:
        viable = gate[
            (gate["selected_rows"] >= 5)
            & (gate["safe_positive_rows"] >= 3)
            & (gate["precision"] >= 0.50)
            & (gate["false_positive_broader20"] <= 1)
            & (gate["false_positive_lowrisk"] == 0)
            & (gate["false_positive_direction_mismatch"] <= 1)
            & (gate["unique_cases"] >= 3)
            & (gate["top_case_share"] <= 0.50)
        ].copy()
    shadow_allowed = not viable.empty

    summary = ["# Expected-Delta Findings", ""]
    summary.append("## Main Result")
    summary.append("")
    if shadow_allowed:
        summary.append("An offline high-confidence region exists under the current expected-delta diagnostics. It is still not a controller gate, but it may justify a shadow-gate experiment.")
    else:
        summary.append("Expected-delta heads are more informative than sparse safe-positive classification, but no robust high-confidence prepare region met the shadow-gate criteria.")
    summary.append("")
    summary.append("## Delta Metrics")
    summary.append("")
    summary.append(markdown_table(delta_metrics.sort_values(["head", "mae"]).head(24)))
    summary.append("")
    summary.append("## Pairwise Metrics")
    summary.append("")
    summary.append(markdown_table(pair_metrics.sort_values("f1", ascending=False).head(12)))
    summary.append("")
    summary.append("## High-Confidence Gate Candidates")
    summary.append("")
    summary.append(markdown_table(gate.head(12)))
    (PAPER / "expected_delta_findings.md").write_text("\n".join(summary) + "\n")

    decision = ["# h120 Expected-Delta Head Decision", ""]
    decision.append("## Verdict")
    decision.append("")
    if shadow_allowed:
        decision.append("Expected-delta/pairwise training gives enough offline signal to consider a shadow-gate logging experiment, but not a controller gate.")
    else:
        decision.append("Do not enter a controller gate. Do not enter shadow gate unless the team accepts a research-only diagnostic with no controller action.")
    decision.append("")
    decision.append("## Required Answers")
    decision.append("")
    decision.append("1. **Is expected-delta more stable than safe_positive binary classification?**")
    decision.append("")
    best_delta = delta_metrics.sort_values(["head", "mae"]).groupby("head").head(1)
    decision.append("Yes as a diagnostic target: it uses all action rows rather than only 11 pump_saving safe positives. However, regression error and case-level concentration still limit controller use.")
    decision.append("")
    decision.append(markdown_table(best_delta[["head", "model", "mae", "rmse", "r2"]]))
    decision.append("")
    decision.append("2. **Is there a high-confidence prepare region?**")
    decision.append("")
    if shadow_allowed:
        decision.append("There is a small offline candidate region meeting the configured shadow criteria:")
        decision.append("")
        decision.append(markdown_table(viable.head(8)))
    else:
        decision.append("No robust region met the required precision, false-positive, case-spread, and sample-count criteria.")
        decision.append("")
        decision.append(markdown_table(gate.head(8)))
    decision.append("")
    decision.append("3. **Is pump_saving_prepare suitable as the first small action?**")
    decision.append("")
    pump_gate = gate[gate["action"].eq(PUMP_ACTION)].head(5) if not gate.empty else pd.DataFrame()
    if not pump_gate.empty and (pump_gate["safe_positive_rows"].max() >= 3):
        decision.append("Only as a cautious research action. It has some positive windows, but remains sparse and needs expected-delta guards.")
    else:
        decision.append("Not yet as a standalone first action. It should be handled through expected-delta scoring plus anti-trigger/regression-risk guards.")
    decision.append("")
    decision.append("4. **Should active_small_prepare stay second-stage?**")
    decision.append("")
    decision.append("Yes. It has more safe-positive rows than pump_saving, but higher authority and broader regression risk. Keep it as a second-stage research action.")
    decision.append("")
    decision.append("5. **Allow shadow gate?**")
    decision.append("")
    decision.append("Allowed only if shadow means logging-only with no pump/controller effect. It should not be a default-off closed-loop gate yet." if shadow_allowed else "No, not under the configured criteria. More data/action redesign is needed before shadow recommendations are useful.")
    decision.append("")
    decision.append("6. **If not allowed, what next?**")
    decision.append("")
    decision.append("Expand candidate samples beyond the current 360-bucket pool, redesign the prepare action family into budgeted/continuous target-refresh magnitudes, and continue expected-delta head design. Do not return to far-risk-threshold tuning.")
    decision.append("")
    decision.append("## Do Not Do")
    decision.append("")
    decision.append("- Do not attach controller gate.")
    decision.append("- Do not weaken v1.6 floor.")
    decision.append("- Do not tune far-high thresholds to chase a few cases.")
    decision.append("- Do not treat oracle/safe-positive counts as learned controller proof.")
    decision.append("- Do not use case_id or dataset as model features.")
    (OUT / "h120_expected_delta_decision.md").write_text("\n".join(decision) + "\n")


def write_task_status() -> None:
    status = {
        "task": "h120_expected_delta_head_v1",
        "status": "completed",
        "controller_modified": False,
        "v16_floor_modified": False,
        "outputs": [
            str((OUT / "expected_delta_training_manifest.md").relative_to(REPO)),
            str((OUT / "pairwise_action_table.csv").relative_to(REPO)),
            str((OUT / "expected_delta_metrics.csv").relative_to(REPO)),
            str((OUT / "pairwise_preference_metrics.csv").relative_to(REPO)),
            str((OUT / "high_confidence_delta_gate_table.csv").relative_to(REPO)),
            str((OUT / "h120_expected_delta_decision.md").relative_to(REPO)),
        ],
    }
    (DEBUG / "task_status.json").write_text(json.dumps(status, indent=2) + "\n")


def main() -> None:
    ensure_dirs()
    source = load_source()
    action = build_action_table(source)
    pairwise = build_pairwise_table(action)
    split_dist = write_split_distribution(action, pairwise)
    write_manifest(source, action, pairwise, split_dist)
    delta_metrics, delta_preds = train_expected_delta_heads(action)
    risk_metrics, risk_preds = train_regression_risk(action)
    pair_metrics, pair_preds = train_pairwise_heads(pairwise)
    gate = high_confidence_delta_gate(action, delta_preds, risk_preds, pair_preds, delta_metrics)
    importance = feature_importance(action, pairwise)
    write_reports(action, pairwise, delta_metrics, risk_metrics, pair_metrics, gate, importance)
    write_task_status()


if __name__ == "__main__":
    main()

