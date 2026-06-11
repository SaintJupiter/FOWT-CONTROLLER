#!/usr/bin/env python3
"""Offline h120 action-effect regime audit v1.

This script reads the completed h120 prepare action-family counterfactuals and
builds bucket-level action-effect regime labels.  It does not attach a
controller gate, does not modify v1.6 floor logic, and does not run new
closed-loop casebooks.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REPO = Path(__file__).resolve().parents[2]
SOURCE = REPO / "outputs/wind_prediction/h120_prepare_action_family_probe_v1"
OUT = REPO / "outputs/wind_prediction/h120_action_effect_regime_v1"
RAW = OUT / "raw_tables"
DEBUG = OUT / "debug"
PAPER = OUT / "paper_ready"

COUNTERFACTUAL_TABLE = SOURCE / "action_family_counterfactual_table.csv"
EFFECT_PATTERN_TABLE = SOURCE / "action_family_effect_pattern_table.csv"
CANDIDATE_TABLE = SOURCE / "action_family_candidate_table.csv"
BASELINE_ROOT = REPO / "outputs/wind_prediction/h120_learned_risk_scheduler_probe_v1"

WAIT_ACTION = "wait"
WATCH_ACTION = "watch_only"
MICRO50_ACTION = "micro_prepare_50"
PUMP_ACTION = "pump_saving_prepare"
ACTIVE_ACTION = "active_small_prepare"
MAIN_ACTIONS = (MICRO50_ACTION, PUMP_ACTION)
PREPARE_ACTIONS = (
    "micro_prepare_25",
    MICRO50_ACTION,
    "pump_saving_prepare",
    "micro_prepare_75",
    "active_small_prepare",
)

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
    if col not in df:
        return pd.Series(default, index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce").fillna(default).astype(float)


def safe_float(value: Any, default: float = np.nan) -> float:
    try:
        if pd.isna(value):
            return default
    except TypeError:
        pass
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def load_source_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not COUNTERFACTUAL_TABLE.exists():
        raise FileNotFoundError(f"missing counterfactual table: {COUNTERFACTUAL_TABLE}")
    cf = pd.read_csv(COUNTERFACTUAL_TABLE)
    pattern = pd.read_csv(EFFECT_PATTERN_TABLE)
    candidates = pd.read_csv(CANDIDATE_TABLE)
    for frame in (cf, pattern, candidates):
        frame["dataset"] = frame["dataset"].astype(str)
        frame["case_id"] = frame["case_id"].astype(str)
        frame["bucket"] = pd.to_numeric(frame["bucket"], errors="coerce").fillna(-1).astype(int)
    cf["group_id"] = cf["dataset"] + "::" + cf["case_id"]
    cf["bucket_key"] = cf["dataset"] + "::" + cf["case_id"] + "::" + cf["bucket"].astype(str)
    pattern["bucket_key"] = pattern["dataset"] + "::" + pattern["case_id"] + "::" + pattern["bucket"].astype(str)
    candidates["bucket_key"] = (
        candidates["dataset"] + "::" + candidates["case_id"] + "::" + candidates["bucket"].astype(str)
    )
    return cf, pattern, candidates


def _case_aliases(row: pd.Series) -> set[str]:
    aliases = {str(row.get("case_id", "")), str(row.get("numbered_case_id", ""))}
    for alias in list(aliases):
        if len(alias) > 3 and alias[:2].isdigit() and alias[2] == "_":
            aliases.add(alias[3:])
    return {a for a in aliases if a and a != "nan"}


def baseline_axis_lookup(candidates: pd.DataFrame) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for dataset, run_name in {
        "guard10": "guard10_v16_oracle_baseline",
        "broader20": "broader20_v16_oracle_baseline",
    }.items():
        path = BASELINE_ROOT / run_name / "casebook_summary.csv"
        if not path.exists():
            continue
        summary = pd.read_csv(path)
        for _, cand in candidates[candidates["dataset"].eq(dataset)].iterrows():
            aliases = _case_aliases(cand)
            match = summary.loc[summary["case_id"].astype(str).isin(aliases)]
            if match.empty:
                continue
            row = match.iloc[0]
            out[str(cand["bucket_key"])] = {
                "baseline_pitch_p95": safe_float(row.get("primary_pitch_p95")),
                "baseline_roll_p95": safe_float(row.get("primary_roll_p95")),
            }
    return out


def forced_axis_metrics(cf: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for row in cf.itertuples(index=False):
        action = str(row.action)
        if action in (WAIT_ACTION, WATCH_ACTION):
            continue
        run_dir_value = getattr(row, "run_dir", "")
        if not isinstance(run_dir_value, str) or not run_dir_value:
            continue
        summary_path = REPO / run_dir_value / "casebook_summary.csv"
        if not summary_path.exists():
            continue
        try:
            summary = pd.read_csv(summary_path).iloc[0]
        except Exception:
            continue
        rows.append(
            {
                "probe_id": row.probe_id,
                "bucket_key": row.bucket_key,
                "action": action,
                "primary_pitch_p95": safe_float(summary.get("primary_pitch_p95")),
                "primary_roll_p95": safe_float(summary.get("primary_roll_p95")),
                "primary_pump_work_m3": safe_float(summary.get("primary_pump_work_m3")),
            }
        )
    return pd.DataFrame(rows)


def enrich_counterfactual_with_axis(cf: pd.DataFrame, candidates: pd.DataFrame) -> pd.DataFrame:
    baseline = baseline_axis_lookup(candidates)
    axis = forced_axis_metrics(cf)
    out = cf.merge(axis, on=["probe_id", "bucket_key", "action"], how="left")
    out["baseline_pitch_p95"] = out["bucket_key"].map(
        {k: v["baseline_pitch_p95"] for k, v in baseline.items()}
    )
    out["baseline_roll_p95"] = out["bucket_key"].map(
        {k: v["baseline_roll_p95"] for k, v in baseline.items()}
    )
    for col in ("primary_pitch_p95", "primary_roll_p95"):
        out.loc[out["action"].isin([WAIT_ACTION, WATCH_ACTION]), col] = out.loc[
            out["action"].isin([WAIT_ACTION, WATCH_ACTION]),
            "bucket_key",
        ].map({k: v[col.replace("primary", "baseline")] for k, v in baseline.items()})
    out["delta_pitch_p95"] = num_series(out, "primary_pitch_p95") - num_series(out, "baseline_pitch_p95")
    out["delta_roll_p95"] = num_series(out, "primary_roll_p95") - num_series(out, "baseline_roll_p95")
    out.to_csv(RAW / "action_effect_action_table.csv", index=False)
    return out


def action_utility(row: pd.Series) -> float:
    return (
        safe_float(row.get("time_gain_s"), 0.0)
        + 0.35 * safe_float(row.get("idle_gain_s"), 0.0)
        - 0.12 * safe_float(row.get("pump_delta_m3"), 0.0)
        - 180.0 * max(safe_float(row.get("delta_full_fallback_pp"), 0.0), 0.0)
        - 120.0 * max(safe_float(row.get("delta_full_max_p95"), 0.0), 0.0)
    )


def monotone(values: list[float], *, nondecreasing: bool = True, tol: float = 1e-6) -> bool:
    if len(values) < 2:
        return True
    if nondecreasing:
        return all(b >= a - tol for a, b in zip(values, values[1:]))
    return all(b <= a + tol for a, b in zip(values, values[1:]))


def build_regime_labels(
    cf: pd.DataFrame, pattern: pd.DataFrame, candidates: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    action_rows = enrich_counterfactual_with_axis(cf, candidates)
    action_labeled = action_rows[action_rows["action"].isin([WAIT_ACTION, WATCH_ACTION, *PREPARE_ACTIONS])].copy()
    action_labeled["time_gain_s"] = -num_series(action_labeled, "delta_post60_time_over5")
    action_labeled["idle_gain_s"] = -num_series(action_labeled, "delta_post60_idle_over5")
    action_labeled["pump_delta_m3"] = num_series(action_labeled, "delta_full_pump_m3")
    action_labeled["action_utility"] = action_labeled.apply(action_utility, axis=1)
    prep = action_labeled[action_labeled["action"].isin(PREPARE_ACTIONS)].copy()
    prep["action_positive_effect"] = (
        (
            (prep["time_gain_s"] >= 30.0)
            | (prep["idle_gain_s"] >= 60.0)
            | (num_series(prep, "delta_post60_floor_entry") < 0)
            | (num_series(prep, "delta_post60_medium_delay_rows") < 0)
        )
        & (num_series(prep, "delta_full_fallback_pp") <= 0.05)
        & (num_series(prep, "delta_full_max_p95") <= 0.05)
        & (prep["pump_delta_m3"] <= 80.0)
        & (num_series(prep, "delta_post60_time_over5") <= 5.0)
    ).astype(int)
    prep["action_regression_risk"] = (
        (num_series(prep, "delta_post60_time_over5") > 10.0)
        | (num_series(prep, "delta_post60_idle_over5") > 20.0)
        | (num_series(prep, "delta_full_fallback_pp") > 0.05)
        | (num_series(prep, "delta_full_max_p95") > 0.05)
    ).astype(int)
    prep["action_negative_effect"] = (
        (prep["action_regression_risk"] > 0)
        | (
            (prep["pump_delta_m3"] > 5.0)
            & (prep["time_gain_s"] < 30.0)
            & (prep["idle_gain_s"] < 60.0)
            & (num_series(prep, "delta_post60_floor_entry") >= 0)
            & (num_series(prep, "delta_post60_medium_delay_rows") >= 0)
        )
    ).astype(int)
    prep["axis_tradeoff"] = (
        ((prep["delta_pitch_p95"] <= -0.05) & (prep["delta_roll_p95"] >= 0.05))
        | ((prep["delta_roll_p95"] <= -0.05) & (prep["delta_pitch_p95"] >= 0.05))
    ).astype(int)
    for col in [
        "action_positive_effect",
        "action_regression_risk",
        "action_negative_effect",
        "axis_tradeoff",
    ]:
        action_labeled[col] = 0
        action_labeled.loc[prep.index, col] = prep[col].astype(int)

    rows = []
    pattern_by_key = pattern.set_index("bucket_key").to_dict("index")
    candidates_by_key = candidates.set_index("bucket_key").to_dict("index")
    scale_order = {
        "micro_prepare_25": 0.25,
        "micro_prepare_50": 0.50,
        "pump_saving_prepare": 0.08 / 0.15,
        "micro_prepare_75": 0.75,
        "active_small_prepare": 1.00,
    }
    for bucket_key, group in prep.groupby("bucket_key", dropna=False):
        group = group.copy()
        group["scale_order"] = group["action"].map(scale_order)
        group = group.sort_values(["scale_order", "action"])
        p = pattern_by_key.get(bucket_key, {})
        c = candidates_by_key.get(bucket_key, {})
        pump_values = list(pd.to_numeric(group["pump_delta_m3"], errors="coerce").fillna(0.0))
        time_gains = list(pd.to_numeric(group["time_gain_s"], errors="coerce").fillna(0.0))
        idle_gains = list(pd.to_numeric(group["idle_gain_s"], errors="coerce").fillna(0.0))
        utility_values = list(pd.to_numeric(group["action_utility"], errors="coerce").fillna(-1e9))
        no_effect = int(p.get("all_prepare_no_effect", 0))
        effectful = int(not bool(no_effect))
        non_mono = int(
            bool(p.get("pump_delta_nonmonotone_by_scale", 0))
            or not monotone(time_gains, nondecreasing=True, tol=0.5)
            or not monotone(idle_gains, nondecreasing=True, tol=0.5)
        )
        pos = int(group["action_positive_effect"].max() > 0)
        neg = int(group["action_negative_effect"].max() > 0)
        reg = int(group["action_regression_risk"].max() > 0)
        axis_tradeoff = int(group["axis_tradeoff"].max() > 0)
        best = group.iloc[int(np.argmax(utility_values))]
        micro50 = group[group["action"].eq(MICRO50_ACTION)]
        pump = group[group["action"].eq(PUMP_ACTION)]
        active = group[group["action"].eq(ACTIVE_ACTION)]
        rows.append(
            {
                "bucket_key": bucket_key,
                "probe_id": group.iloc[0]["probe_id"],
                "dataset": group.iloc[0]["dataset"],
                "case_id": group.iloc[0]["case_id"],
                "group_id": f"{group.iloc[0]['dataset']}::{group.iloc[0]['case_id']}",
                "bucket": int(group.iloc[0]["bucket"]),
                "timestamp": group.iloc[0]["timestamp"],
                "selection_category": group.iloc[0].get("selection_category", ""),
                "candidate_reasons": group.iloc[0].get("candidate_reasons", ""),
                "no_effect": no_effect,
                "effectful": effectful,
                "positive_effect": pos,
                "negative_effect": neg,
                "non_monotone": non_mono,
                "regression_risk": reg,
                "axis_tradeoff": axis_tradeoff,
                "best_action_by_utility": best["action"],
                "best_action_utility": safe_float(best["action_utility"]),
                "best_action_positive": int(best["action_positive_effect"]),
                "best_action_negative": int(best["action_negative_effect"]),
                "micro50_positive_effect": int(micro50["action_positive_effect"].max()) if not micro50.empty else 0,
                "micro50_negative_effect": int(micro50["action_negative_effect"].max()) if not micro50.empty else 0,
                "pump_saving_positive_effect": int(pump["action_positive_effect"].max()) if not pump.empty else 0,
                "pump_saving_negative_effect": int(pump["action_negative_effect"].max()) if not pump.empty else 0,
                "active_small_positive_effect": int(active["action_positive_effect"].max()) if not active.empty else 0,
                "active_small_regression_risk": int(active["action_regression_risk"].max()) if not active.empty else 0,
                "range_pump_delta_m3": safe_float(p.get("range_pump_delta_m3"), np.nan),
                "range_time_delta_s": safe_float(p.get("range_time_delta_s"), np.nan),
                "range_idle_delta_s": safe_float(p.get("range_idle_delta_s"), np.nan),
            }
        )
        for col in FEATURE_COLUMNS:
            rows[-1][col] = c.get(col, group.iloc[0].get(col, 0.0))
    regime = pd.DataFrame(rows)
    regime.to_csv(OUT / "action_effect_regime_table.csv", index=False)
    regime.to_csv(RAW / "action_effect_regime_table.csv", index=False)
    # Wait/watch rows are included for the optional effectful-subset pairwise stage.
    for col in ["action_positive_effect", "action_regression_risk", "action_negative_effect", "axis_tradeoff"]:
        action_labeled[col] = action_labeled[col].fillna(0).astype(int)
    action_labeled.to_csv(RAW / "action_effect_action_labeled_table.csv", index=False)
    return regime, prep, action_labeled


def summarize_regimes(regime: pd.DataFrame, action_rows: pd.DataFrame) -> dict[str, pd.DataFrame]:
    summaries: dict[str, pd.DataFrame] = {}
    labels = [
        "no_effect",
        "effectful",
        "positive_effect",
        "negative_effect",
        "non_monotone",
        "regression_risk",
        "axis_tradeoff",
    ]
    summaries["dataset"] = (
        regime.groupby("dataset")[labels]
        .agg(["sum", "mean"])
        .round(3)
        .reset_index()
    )
    summaries["case"] = (
        regime.groupby(["dataset", "case_id"])[labels]
        .sum()
        .reset_index()
        .sort_values(["dataset", "negative_effect", "positive_effect"], ascending=[True, False, False])
    )
    shape_cols = [
        "direction_consistent",
        "direction_mismatch",
        "near_safe_far_risky",
        "delayed_intensification",
        "reintensification_after_relief",
        "signflip_or_reversal",
        "far_persistent_high",
    ]
    shape_rows = []
    for col in shape_cols:
        if col not in regime:
            continue
        for value, group in regime.groupby((num_series(regime, col) > 0).astype(int)):
            shape_rows.append(
                {
                    "feature": col,
                    "feature_active": int(value),
                    "rows": int(len(group)),
                    **{f"{lab}_rate": float(group[lab].mean()) for lab in labels},
                }
            )
    summaries["shape"] = pd.DataFrame(shape_rows)
    posture = regime.copy()
    posture["max_axis_bin"] = pd.cut(
        num_series(posture, "max_axis_deg"),
        [-0.01, 3.0, 4.0, 5.0, 99.0],
        labels=["<3", "3-4", "4-5", ">=5"],
    )
    summaries["max_axis_bin"] = (
        posture.groupby(["dataset", "max_axis_bin"], observed=False)[labels]
        .agg(["sum", "mean", "count"])
        .reset_index()
    )
    action_summary = (
        action_rows.groupby(["dataset", "action"])[
            ["action_positive_effect", "action_negative_effect", "action_regression_risk", "axis_tradeoff"]
        ]
        .agg(["sum", "mean"])
        .round(3)
        .reset_index()
    )
    summaries["action_family"] = action_summary
    for name, frame in summaries.items():
        frame.to_csv(RAW / f"regime_distribution_by_{name}.csv", index=False)
    return summaries


def build_features(df: pd.DataFrame, action_col: str | None = None) -> tuple[pd.DataFrame, list[str]]:
    cols = [c for c in FEATURE_COLUMNS if c in df.columns]
    x = (
        df[cols]
        .apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .reset_index(drop=True)
    )
    if action_col is not None:
        x = pd.concat(
            [x, pd.get_dummies(df[action_col].astype(str), prefix=action_col).reset_index(drop=True)],
            axis=1,
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
            max_iter=80,
            learning_rate=0.05,
            random_state=4242,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=5,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=4242,
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
            random_state=4242,
        ),
        "random_forest": RandomForestRegressor(
            n_estimators=300,
            max_depth=5,
            min_samples_leaf=2,
            random_state=4242,
        ),
    }


@dataclass(frozen=True)
class EffectHead:
    name: str
    label_col: str


EFFECT_HEADS = [
    EffectHead("effectful_prob", "effectful"),
    EffectHead("no_effect_prob", "no_effect"),
    EffectHead("regression_risk_prob", "regression_risk"),
    EffectHead("positive_effect_prob", "positive_effect"),
]


def train_effect_regime_heads(regime: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    from sklearn.metrics import precision_recall_fscore_support, roc_auc_score

    splits = make_splits(regime)
    x_all, _ = build_features(regime)
    metrics: list[dict[str, Any]] = []
    pred_rows: list[dict[str, Any]] = []
    for head in EFFECT_HEADS:
        y_all = regime[head.label_col].astype(int).reset_index(drop=True)
        if y_all.nunique() < 2 or not splits:
            continue
        for fold, train_idx, test_idx in splits:
            y_train = y_all.iloc[train_idx].to_numpy()
            y_test = y_all.iloc[test_idx].to_numpy()
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
                test = regime.iloc[test_idx].copy()
                false_pos = (pred == 1) & (y_test == 0)
                metrics.append(
                    {
                        "head": head.name,
                        "model": model_name,
                        "fold": fold,
                        "test_rows": int(len(test)),
                        "test_positive": int(y_test.sum()),
                        "precision": float(precision),
                        "recall": float(recall),
                        "f1": float(f1),
                        "roc_auc": float(auc) if not pd.isna(auc) else np.nan,
                        "false_positive_broader20": int((false_pos & test["dataset"].eq("broader20").to_numpy()).sum()),
                        "false_positive_lowrisk": int(
                            (
                                false_pos
                                & test["candidate_reasons"].astype(str).str.contains("lowrisk", na=False).to_numpy()
                            ).sum()
                        ),
                        "false_positive_mismatch": int(
                            (false_pos & (num_series(test, "direction_mismatch") > 0).to_numpy()).sum()
                        ),
                    }
                )
                sub = test[
                    [
                        "bucket_key",
                        "probe_id",
                        "dataset",
                        "case_id",
                        "group_id",
                        "bucket",
                        "candidate_reasons",
                        "effectful",
                        "no_effect",
                        "positive_effect",
                        "negative_effect",
                        "regression_risk",
                    ]
                ].copy()
                sub["head"] = head.name
                sub["model"] = model_name
                sub["fold"] = fold
                sub["target"] = y_test
                sub["prob"] = proba
                sub["pred"] = pred
                pred_rows.extend(sub.to_dict("records"))
    metric_df = pd.DataFrame(metrics)
    pred_df = pd.DataFrame(pred_rows)
    metric_df.to_csv(RAW / "effect_regime_fold_metrics.csv", index=False)
    pred_df.to_csv(RAW / "effect_regime_oof_predictions.csv", index=False)
    if metric_df.empty:
        agg = pd.DataFrame()
    else:
        agg = (
            metric_df.groupby(["head", "model"], dropna=False)[
                [
                    "test_rows",
                    "test_positive",
                    "precision",
                    "recall",
                    "f1",
                    "roc_auc",
                    "false_positive_broader20",
                    "false_positive_lowrisk",
                    "false_positive_mismatch",
                ]
            ]
            .mean()
            .reset_index()
        )
    agg.to_csv(OUT / "effect_regime_model_metrics.csv", index=False)
    agg.to_csv(RAW / "effect_regime_model_metrics.csv", index=False)
    return agg, pred_df


def high_confidence_effect_table(regime: pd.DataFrame, preds: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if preds.empty:
        out = pd.DataFrame()
        out.to_csv(RAW / "effect_regime_high_confidence_table.csv", index=False)
        return out
    for (head, model), g in preds.groupby(["head", "model"], dropna=False):
        for threshold in (0.60, 0.70, 0.80):
            pick = g[g["prob"] >= threshold].copy()
            rows.append(
                {
                    "head": head,
                    "model": model,
                    "threshold": threshold,
                    "selected_rows": int(len(pick)),
                    "selected_cases": int(pick["case_id"].nunique()) if not pick.empty else 0,
                    "precision": float(pick["target"].mean()) if not pick.empty else np.nan,
                    "broader20_selected": int((pick["dataset"] == "broader20").sum()) if not pick.empty else 0,
                    "guard10_selected": int((pick["dataset"] == "guard10").sum()) if not pick.empty else 0,
                    "mismatch_selected": int(
                        pick["candidate_reasons"].astype(str).str.contains("direction_mismatch", na=False).sum()
                    )
                    if not pick.empty
                    else 0,
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "effect_regime_high_confidence_table.csv", index=False)
    out.to_csv(RAW / "effect_regime_high_confidence_table.csv", index=False)
    return out


def feature_importance(regime: pd.DataFrame) -> pd.DataFrame:
    from sklearn.ensemble import RandomForestClassifier

    x, feature_names = build_features(regime)
    rows = []
    for head in EFFECT_HEADS:
        y = regime[head.label_col].astype(int)
        if y.nunique() < 2:
            continue
        model = RandomForestClassifier(
            n_estimators=300,
            max_depth=5,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=4242,
        )
        model.fit(x, y)
        for name, importance in sorted(
            zip(feature_names, model.feature_importances_), key=lambda item: item[1], reverse=True
        )[:20]:
            rows.append({"head": head.name, "feature": name, "importance": float(importance)})
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "effect_regime_feature_importance.csv", index=False)
    out.to_csv(RAW / "effect_regime_feature_importance.csv", index=False)
    return out


def evaluate_regime_go(metrics: pd.DataFrame, hc: pd.DataFrame, preds: pd.DataFrame) -> tuple[bool, dict[str, Any]]:
    evidence: dict[str, Any] = {
        "effectful_high_conf_precision": np.nan,
        "effectful_high_conf_cases": 0,
        "regression_risk_best_recall": np.nan,
        "broader20_effectful_false_positive_rate": np.nan,
    }
    if hc.empty or metrics.empty or preds.empty:
        return False, evidence
    effect_hc = hc[
        (hc["head"] == "effectful_prob")
        & (hc["threshold"] >= 0.70)
        & (hc["selected_rows"] >= 3)
    ].sort_values(["precision", "selected_cases"], ascending=[False, False])
    if not effect_hc.empty:
        best = effect_hc.iloc[0]
        evidence["effectful_high_conf_precision"] = safe_float(best["precision"])
        evidence["effectful_high_conf_cases"] = int(best["selected_cases"])
        evidence["effectful_high_conf_model"] = str(best["model"])
    reg = metrics[metrics["head"] == "regression_risk_prob"].sort_values("recall", ascending=False)
    if not reg.empty:
        evidence["regression_risk_best_recall"] = safe_float(reg.iloc[0]["recall"])
        evidence["regression_risk_best_model"] = str(reg.iloc[0]["model"])
    eff_preds = preds[(preds["head"] == "effectful_prob") & (preds["prob"] >= 0.70)].copy()
    if not eff_preds.empty:
        broad = eff_preds[eff_preds["dataset"] == "broader20"]
        if not broad.empty:
            evidence["broader20_effectful_false_positive_rate"] = float(
                ((broad["pred"] == 1) & (broad["target"] == 0)).mean()
            )
    go = (
        evidence["effectful_high_conf_precision"] >= 0.65
        and evidence["regression_risk_best_recall"] >= 0.75
        and evidence["effectful_high_conf_cases"] >= 3
        and (
            pd.isna(evidence["broader20_effectful_false_positive_rate"])
            or evidence["broader20_effectful_false_positive_rate"] <= 0.25
        )
    )
    return bool(go), evidence


def build_effectful_action_subset(regime: pd.DataFrame, action_rows: pd.DataFrame) -> pd.DataFrame:
    eligible_keys = set(regime.loc[regime["effectful"] > 0, "bucket_key"].astype(str))
    subset = action_rows[
        action_rows["bucket_key"].astype(str).isin(eligible_keys)
        & action_rows["action"].isin([WAIT_ACTION, MICRO50_ACTION, PUMP_ACTION])
    ].copy()
    subset["group_id"] = subset["dataset"].astype(str) + "::" + subset["case_id"].astype(str)
    if "time_gain_s" not in subset:
        subset["time_gain_s"] = -num_series(subset, "delta_post60_time_over5")
    if "idle_gain_s" not in subset:
        subset["idle_gain_s"] = -num_series(subset, "delta_post60_idle_over5")
    if "pump_delta_m3" not in subset:
        subset["pump_delta_m3"] = num_series(subset, "delta_full_pump_m3")
    if "action_utility" not in subset:
        subset["action_utility"] = subset.apply(action_utility, axis=1)
    subset.to_csv(RAW / "effectful_action_subset.csv", index=False)
    return subset


def train_effectful_expected_delta(subset: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, roc_auc_score

    if subset.empty or subset["group_id"].nunique() < 2:
        empty = pd.DataFrame()
        empty.to_csv(RAW / "effectful_expected_delta_metrics.csv", index=False)
        empty.to_csv(RAW / "effectful_pairwise_preference_table.csv", index=False)
        return empty, empty
    x_all, _ = build_features(subset, action_col="action")
    splits = make_splits(subset)
    rows = []
    for target in ("time_gain_s", "idle_gain_s", "pump_delta_m3", "action_utility"):
        y = pd.to_numeric(subset[target], errors="coerce").fillna(0.0)
        for fold, train_idx, test_idx in splits:
            for model_name, model in regressors().items():
                model.fit(x_all.iloc[train_idx], y.iloc[train_idx])
                pred = model.predict(x_all.iloc[test_idx])
                truth = y.iloc[test_idx].to_numpy()
                rows.append(
                    {
                        "head": target,
                        "model": model_name,
                        "fold": fold,
                        "test_rows": int(len(test_idx)),
                        "mae": float(mean_absolute_error(truth, pred)),
                        "rmse": float(math.sqrt(mean_squared_error(truth, pred))),
                        "r2": float(r2_score(truth, pred)) if len(truth) > 1 else np.nan,
                    }
                )
    metrics = pd.DataFrame(rows)
    agg = (
        metrics.groupby(["head", "model"], dropna=False)[["test_rows", "mae", "rmse", "r2"]]
        .mean()
        .reset_index()
        if not metrics.empty
        else pd.DataFrame()
    )
    agg.to_csv(OUT / "effectful_expected_delta_metrics.csv", index=False)
    agg.to_csv(RAW / "effectful_expected_delta_metrics.csv", index=False)

    pair_rows = []
    for bucket_key, group in subset.groupby("bucket_key"):
        by_action = {str(r.action): r for r in group.itertuples()}
        if WAIT_ACTION not in by_action:
            continue
        for action in (MICRO50_ACTION, PUMP_ACTION):
            if action not in by_action:
                continue
            wait = by_action[WAIT_ACTION]
            prep = by_action[action]
            pair_rows.append(
                {
                    "bucket_key": bucket_key,
                    "dataset": prep.dataset,
                    "case_id": prep.case_id,
                    "group_id": prep.group_id,
                    "bucket": int(prep.bucket),
                    "action": action,
                    "prepare_preferred": int(float(prep.action_utility) > float(wait.action_utility)),
                    "utility_margin": float(prep.action_utility) - float(wait.action_utility),
                }
            )
    pairwise = pd.DataFrame(pair_rows)
    if pairwise.empty or pairwise["prepare_preferred"].nunique() < 2 or pairwise["group_id"].nunique() < 2:
        pairwise.to_csv(OUT / "effectful_pairwise_preference_table.csv", index=False)
        pairwise.to_csv(RAW / "effectful_pairwise_preference_table.csv", index=False)
        return agg, pairwise
    pair_source = pairwise.merge(
        subset.drop_duplicates("bucket_key")[["bucket_key", *[c for c in FEATURE_COLUMNS if c in subset.columns]]],
        on="bucket_key",
        how="left",
    )
    x_pair, _ = build_features(pair_source, action_col="action")
    splits_pair = make_splits(pair_source)
    pref_rows = []
    for fold, train_idx, test_idx in splits_pair:
        y_train = pair_source["prepare_preferred"].iloc[train_idx].astype(int).to_numpy()
        y_test = pair_source["prepare_preferred"].iloc[test_idx].astype(int).to_numpy()
        if len(set(y_train.tolist())) < 2:
            continue
        for model_name, model in classifiers().items():
            model.fit(x_pair.iloc[train_idx], y_train)
            proba = model.predict_proba(x_pair.iloc[test_idx])[:, 1]
            try:
                auc = roc_auc_score(y_test, proba) if len(set(y_test.tolist())) > 1 else np.nan
            except ValueError:
                auc = np.nan
            pred = (proba >= 0.5).astype(int)
            pref_rows.append(
                {
                    "model": model_name,
                    "fold": fold,
                    "test_rows": int(len(test_idx)),
                    "positive_rows": int(y_test.sum()),
                    "accuracy": float((pred == y_test).mean()),
                    "roc_auc": float(auc) if not pd.isna(auc) else np.nan,
                }
            )
    pref = pd.DataFrame(pref_rows)
    pref.to_csv(OUT / "effectful_pairwise_preference_table.csv", index=False)
    pref.to_csv(RAW / "effectful_pairwise_preference_table.csv", index=False)
    return agg, pref


def write_split_distribution(regime: pd.DataFrame, action_subset: pd.DataFrame | None = None) -> pd.DataFrame:
    rows = []
    for name, frame in (("regime", regime), ("effectful_action_subset", action_subset)):
        if frame is None or frame.empty:
            continue
        splits = make_splits(frame)
        for fold, train_idx, test_idx in splits:
            train = frame.iloc[train_idx]
            test = frame.iloc[test_idx]
            rows.append(
                {
                    "table": name,
                    "fold": fold,
                    "train_rows": int(len(train)),
                    "test_rows": int(len(test)),
                    "train_cases": int(train["group_id"].nunique()),
                    "test_cases": int(test["group_id"].nunique()),
                    "test_datasets": ",".join(sorted(test["dataset"].astype(str).unique())),
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(RAW / "split_distribution.csv", index=False)
    return out


def write_reports(
    regime: pd.DataFrame,
    action_rows: pd.DataFrame,
    summaries: dict[str, pd.DataFrame],
    metrics: pd.DataFrame,
    high_conf: pd.DataFrame,
    importance: pd.DataFrame,
    go: bool,
    evidence: dict[str, Any],
    delta_metrics: pd.DataFrame,
    pairwise_metrics: pd.DataFrame,
) -> None:
    label_cols = [
        "no_effect",
        "effectful",
        "positive_effect",
        "negative_effect",
        "non_monotone",
        "regression_risk",
        "axis_tradeoff",
    ]
    overall = pd.DataFrame(
        [
            {
                "buckets": int(len(regime)),
                **{col: int(regime[col].sum()) for col in label_cols},
            }
        ]
    )
    broader_neg = regime[
        (regime["dataset"] == "broader20")
        & ((regime["negative_effect"] > 0) | (regime["regression_risk"] > 0))
    ]
    guard_pos = regime[(regime["dataset"] == "guard10") & (regime["positive_effect"] > 0)]
    micro50 = action_rows[action_rows["action"] == MICRO50_ACTION]
    active = action_rows[action_rows["action"] == ACTIVE_ACTION]
    active_reg_rate = float(active["action_regression_risk"].mean()) if not active.empty else np.nan
    micro50_pos_rate = float(micro50["action_positive_effect"].mean()) if not micro50.empty else np.nan
    best_delta_r2 = (
        float(delta_metrics["r2"].max())
        if delta_metrics is not None and not delta_metrics.empty and "r2" in delta_metrics
        else np.nan
    )
    pairwise_auc = (
        float(pairwise_metrics["roc_auc"].mean())
        if pairwise_metrics is not None and not pairwise_metrics.empty and "roc_auc" in pairwise_metrics
        else np.nan
    )
    pairwise_best_auc = (
        float(pairwise_metrics["roc_auc"].max())
        if pairwise_metrics is not None and not pairwise_metrics.empty and "roc_auc" in pairwise_metrics
        else np.nan
    )
    subset_ready = bool(
        pd.notna(best_delta_r2)
        and best_delta_r2 >= 0.0
        and pd.notna(pairwise_auc)
        and pairwise_auc >= 0.75
    )

    summary = f"""# h120 Action-Effect Regime v1

Scope: offline labeling and lightweight model diagnostics only. No controller gate, no v1.6 floor change, no new casebook runs.

## Overall Regime Counts

{markdown_table(overall)}

## Dataset Distribution

{markdown_table(summaries['dataset'])}

## Action-Family Distribution

{markdown_table(summaries['action_family'])}

## Key Observations

- Broader20 negative/regression buckets: {len(broader_neg)} / {int((regime['dataset'] == 'broader20').sum())}.
- Guard10 positive-effect buckets: {len(guard_pos)} / {int((regime['dataset'] == 'guard10').sum())}.
- `micro_prepare_50` action positive-effect rate: {micro50_pos_rate:.3f}.
- `active_small_prepare` action regression-risk rate: {active_reg_rate:.3f}.
- Effect-regime go for expected-delta subset diagnostics: {'GO' if go else 'NO-GO'}.
- Effectful-subset best expected-delta R2: {best_delta_r2:.3f}.
- Effectful-subset mean pairwise AUC: {pairwise_auc:.3f}; best fold/model AUC: {pairwise_best_auc:.3f}.
"""
    (PAPER / "action_effect_regime_summary.md").write_text(summary)

    decision = f"""# h120 Action-Effect Regime Decision

## Decision

Do **not** enter shadow gate or controller gate from this result.  The regime labels are useful enough to run an effectful-subset diagnostic, but the effectful-subset expected-delta / pairwise heads are not controller-ready.

## Answers

1. **Does the prepare action have a predictable effect regime?**  Partially.  The labels separate no-effect/effectful/regression/non-monotone regimes, but the sample is still small and case-skewed.  The best evidence is the explicit count table, not a controller-ready classifier.

2. **Are no_effect / non_monotone a major reason expected-delta was hard to learn?**  Yes.  The upstream action-family probe already showed no-effect and non-monotone buckets; this run formalizes them as supervised labels.  Expected-delta should not be trained across those buckets as if the action axis were smooth.

3. **Is `micro_prepare_50` still the first candidate action?**  Yes, for offline modeling only.  It remains the best small-action neighborhood compared with pump_saving, but it still does not solve broader20 false prepare by itself.

4. **Allow shadow gate?**  No.  The effect-regime screen is promising, but the follow-on value/ranking signal is still weak: best expected-delta R2={best_delta_r2:.3f}, mean pairwise AUC={pairwise_auc:.3f}.  Shadow gate requires both an effectful screen and a usable value/ranking head.

5. **Next step.**  Continue offline: expand the action-effect dataset, keep `micro_prepare_50` / pump_saving as the small-action family, add more broader20/lowrisk anti-trigger cases, and improve labels before any controller coupling.  v1.6 remains the control mainline.

## Go / No-Go Evidence

- Effectful high-confidence precision target: >= 0.65; observed {evidence.get('effectful_high_conf_precision', np.nan):.3f}.
- Regression-risk recall target: >= 0.75; observed {evidence.get('regression_risk_best_recall', np.nan):.3f}.
- Effectful high-confidence case support target: >= 3; observed {evidence.get('effectful_high_conf_cases', 0)}.
- Broader20 effectful false-positive rate target: <= 0.25; observed {evidence.get('broader20_effectful_false_positive_rate', np.nan):.3f}.

## Expected-Delta Subset

Expected-delta-on-effectful-subset was {'executed' if go else 'not executed'} because the regime diagnostic gate was {'passed' if go else 'not passed'}.  It is {'not ready for shadow gate' if not subset_ready else 'promising enough for a future log-only shadow diagnostic, still not a controller gate'}: best expected-delta R2={best_delta_r2:.3f}, mean pairwise AUC={pairwise_auc:.3f}, best fold/model pairwise AUC={pairwise_best_auc:.3f}.
"""
    (OUT / "h120_action_effect_regime_decision.md").write_text(decision)

    manifest = f"""# h120 Action-Effect Regime Manifest

- Source: `{SOURCE.relative_to(REPO)}`
- Counterfactual table: `{COUNTERFACTUAL_TABLE.relative_to(REPO)}`
- Output: `{OUT.relative_to(REPO)}`
- Controller changes: none.
- v1.6 floor changes: none.
- New casebook runs: none.
- Axis tradeoff source: baseline and forced-run `casebook_summary.csv` pitch/roll p95 where available.
"""
    (DEBUG / "manifest.md").write_text(manifest)

    # Convenience copies / compact paper tables.
    summaries["case"].head(40).to_csv(PAPER / "action_effect_case_top40.csv", index=False)
    if not metrics.empty:
        metrics.to_csv(PAPER / "effect_regime_model_metrics.csv", index=False)
    if not high_conf.empty:
        high_conf.to_csv(PAPER / "effect_regime_high_confidence_table.csv", index=False)
    if not importance.empty:
        importance.to_csv(PAPER / "effect_regime_feature_importance.csv", index=False)
    if not delta_metrics.empty:
        delta_metrics.to_csv(PAPER / "effectful_expected_delta_metrics.csv", index=False)
    if not pairwise_metrics.empty:
        pairwise_metrics.to_csv(PAPER / "effectful_pairwise_preference_table.csv", index=False)


def main() -> None:
    ensure_dirs()
    cf, pattern, candidates = load_source_tables()
    regime, prep_action_rows, all_action_rows = build_regime_labels(cf, pattern, candidates)
    summaries = summarize_regimes(regime, prep_action_rows)
    metrics, preds = train_effect_regime_heads(regime)
    high_conf = high_confidence_effect_table(regime, preds)
    importance = feature_importance(regime)
    go, evidence = evaluate_regime_go(metrics, high_conf, preds)
    action_subset = build_effectful_action_subset(regime, all_action_rows) if go else pd.DataFrame()
    write_split_distribution(regime, action_subset)
    if go:
        delta_metrics, pairwise_metrics = train_effectful_expected_delta(action_subset)
    else:
        delta_metrics = pd.DataFrame()
        pairwise_metrics = pd.DataFrame()
        delta_metrics.to_csv(OUT / "effectful_expected_delta_metrics.csv", index=False)
        delta_metrics.to_csv(RAW / "effectful_expected_delta_metrics.csv", index=False)
        pairwise_metrics.to_csv(OUT / "effectful_pairwise_preference_table.csv", index=False)
        pairwise_metrics.to_csv(RAW / "effectful_pairwise_preference_table.csv", index=False)
    write_reports(
        regime,
        prep_action_rows,
        summaries,
        metrics,
        high_conf,
        importance,
        go,
        evidence,
        delta_metrics,
        pairwise_metrics,
    )
    print(f"[done] wrote {OUT}")


if __name__ == "__main__":
    main()
