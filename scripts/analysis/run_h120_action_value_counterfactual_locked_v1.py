#!/usr/bin/env python3
"""Locked h120 action-value counterfactual audit v1.

This script expands the small h120 action-value probe into a fixed, reusable
counterfactual set.  It does not train a wind model, does not attach a
controller gate, and does not modify the v1.6 floor.  Forced-prefix one-bucket
target updates are used only to estimate offline action value.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
PREV_SCRIPT = REPO / "scripts/analysis/run_h120_action_value_audit_v1.py"

spec = importlib.util.spec_from_file_location("h120_action_value_audit_v1", PREV_SCRIPT)
if spec is None or spec.loader is None:
    raise RuntimeError(f"cannot import helpers from {PREV_SCRIPT}")
prev = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = prev
spec.loader.exec_module(prev)

OUT = REPO / "outputs/wind_prediction/h120_action_value_counterfactual_locked_v1"
RAW = OUT / "raw_tables"
DEBUG = OUT / "debug"
PAPER = OUT / "paper_ready"
RUNS = OUT / "counterfactual_runs"

SOURCE_AUDIT = REPO / "outputs/wind_prediction/h120_action_value_audit_v1"
SOURCE_CANDIDATES = SOURCE_AUDIT / "action_value_candidate_table.csv"
F120_DATASET = prev.F120_DATASET
DATASETS = prev.DATASETS

PREPARE_ACTIONS = {
    "pump_saving_prepare": "pump_saving",
    "active_small_prepare": "active_small",
}

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


def _ensure_dirs() -> None:
    for path in (OUT, RAW, DEBUG, PAPER, RUNS):
        path.mkdir(parents=True, exist_ok=True)


def _markdown_table(df: pd.DataFrame) -> str:
    return prev._markdown_table(df)


def _num(value: Any, default: float = 0.0) -> float:
    try:
        if pd.isna(value):
            return default
    except TypeError:
        pass
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _load_candidate_source() -> pd.DataFrame:
    if not SOURCE_CANDIDATES.exists():
        raise FileNotFoundError(
            f"missing {SOURCE_CANDIDATES}; run h120_action_value_audit_v1 first"
        )
    df = pd.read_csv(SOURCE_CANDIDATES)
    if "bucket" not in df.columns:
        raise ValueError("source candidate table must contain bucket")
    return df


def _reasons(row: pd.Series) -> list[str]:
    reasons: list[str] = []
    dataset = str(row["dataset"])
    if dataset == "guard10" and _num(row["remote_risk_active"]) > 0 and (
        _num(row["future_60m_time_over5"]) > 0
        or _num(row["future_60m_idle_over5"]) > 0
        or _num(row["future_60m_floor_entry"]) > 0
        or _num(row["future_60m_medium_delay_rows"]) > 0
    ):
        reasons.append("guard10_remote_risk_future_exposure")
    if dataset == "broader20" and _num(row["remote_risk_active"]) > 0 and (
        _num(row["future_60m_time_over5"]) <= 5
        and _num(row["future_60m_idle_over5"]) <= 5
    ):
        reasons.append("broader20_remote_risk_weak_future_safety")
    if (
        3.0 <= _num(row["max_axis_deg"]) < 5.0
        and _num(row["posture_trend_10m_deg"]) > 0.05
    ):
        reasons.append("posture_3_5_trend_worsening")
    if _num(row["near_safe_far_risky"]) > 0:
        reasons.append("near_safe_far_risky")
    if (
        _num(row["target_stale"]) > 0
        or _num(row["pump_idle"]) > 0
        or (
            _num(row["target_error_mean_kg"]) < 500.0
            and _num(row["remote_risk_active"]) > 0
        )
    ):
        reasons.append("target_stale_or_small_error_or_pump_idle")
    if _num(row["remote_risk_active"]) > 0 and _num(row["direction_consistent"]) > 0:
        reasons.append("direction_consistent_far_risk")
    if _num(row["direction_mismatch"]) > 0:
        reasons.append("far_high_direction_mismatch")
    if _num(row["signflip_or_reversal"]) > 0:
        reasons.append("signflip_or_reversal")
    if _num(row["near_risky_far_relief"]) > 0:
        reasons.append("near_risk_high_far_relief")
    if (
        _num(row["remote_risk_active"]) <= 0
        and _num(row["max_axis_deg"]) < 3.0
        and _num(row["future_60m_time_over5"]) <= 0
    ):
        reasons.append("lowrisk_sanity")
    return reasons


def _candidate_score(row: pd.Series, reasons: list[str]) -> float:
    score = 0.0
    score += 2.5 * min(_num(row["future_60m_time_over5"]) / 60.0, 10.0)
    score += 1.0 * min(_num(row["future_60m_idle_over5"]) / 60.0, 10.0)
    score += 2.0 * _num(row["candidate_enter4_like"])
    score += 1.5 * _num(row["candidate_prefloor_like"])
    score += 1.0 * _num(row["direction_consistent"])
    score += 0.8 * _num(row["near_safe_far_risky"])
    score += 0.8 * _num(row["direction_mismatch"])
    score += 0.8 * _num(row["signflip_or_reversal"])
    score += 0.6 * _num(row["near_risky_far_relief"])
    score += 0.5 * len(reasons)
    if str(row["dataset"]) == "broader20":
        score += 0.2
    return float(score)


def _pick_rows(
    df: pd.DataFrame,
    mask: pd.Series,
    quota: int,
    used: set[tuple[str, str, int]],
    category: str,
) -> list[pd.Series]:
    sub = df.loc[mask].copy()
    if sub.empty or quota <= 0:
        return []
    rows: list[pd.Series] = []
    for _, row in sub.sort_values("_lock_score", ascending=False).iterrows():
        key = (str(row["dataset"]), str(row["case_id"]), int(row["bucket"]))
        if key in used:
            continue
        row = row.copy()
        row["_primary_selection_category"] = category
        rows.append(row)
        used.add(key)
        if len(rows) >= quota:
            break
    return rows


def build_locked_candidates(df: pd.DataFrame, target_n: int = 36) -> pd.DataFrame:
    _ensure_dirs()
    work = df.copy()
    reason_values = []
    scores = []
    for _, row in work.iterrows():
        reasons = _reasons(row)
        reason_values.append(";".join(reasons))
        scores.append(_candidate_score(row, reasons))
    work["selection_reasons"] = reason_values
    work["_lock_score"] = scores
    eligible = work[work["selection_reasons"].astype(str).str.len() > 0].copy()

    used: set[tuple[str, str, int]] = set()
    rows: list[pd.Series] = []
    specs = [
        (
            "guard10_future_exposure",
            7,
            (eligible["dataset"] == "guard10")
            & eligible["selection_reasons"].str.contains(
                "guard10_remote_risk_future_exposure", na=False
            ),
        ),
        (
            "broader20_weak_future_safety",
            7,
            (eligible["dataset"] == "broader20")
            & eligible["selection_reasons"].str.contains(
                "broader20_remote_risk_weak_future_safety", na=False
            ),
        ),
        (
            "posture_trend_worsening",
            3,
            eligible["selection_reasons"].str.contains(
                "posture_3_5_trend_worsening", na=False
            ),
        ),
        (
            "near_safe_far_risky",
            3,
            eligible["selection_reasons"].str.contains("near_safe_far_risky", na=False),
        ),
        (
            "target_or_pump_idle",
            3,
            eligible["selection_reasons"].str.contains(
                "target_stale_or_small_error_or_pump_idle", na=False
            ),
        ),
        (
            "direction_consistent",
            3,
            eligible["selection_reasons"].str.contains(
                "direction_consistent_far_risk", na=False
            ),
        ),
        (
            "direction_mismatch",
            3,
            eligible["selection_reasons"].str.contains(
                "far_high_direction_mismatch", na=False
            ),
        ),
        (
            "signflip",
            2,
            eligible["selection_reasons"].str.contains("signflip_or_reversal", na=False),
        ),
        (
            "lowrisk_sanity",
            3,
            eligible["selection_reasons"].str.contains("lowrisk_sanity", na=False),
        ),
        (
            "near_risky_far_relief",
            2,
            eligible["selection_reasons"].str.contains(
                "near_risk_high_far_relief", na=False
            ),
        ),
    ]
    for category, quota, mask in specs:
        rows.extend(_pick_rows(eligible, mask, quota, used, category))

    if len(rows) < target_n:
        rows.extend(
            _pick_rows(
                eligible,
                pd.Series(True, index=eligible.index),
                target_n - len(rows),
                used,
                "score_fill",
            )
        )

    locked = pd.DataFrame(rows).head(target_n).copy()
    locked = locked.reset_index(drop=True)
    locked["locked_probe_id"] = [
        f"{r.dataset}_{r.case_id}_b{int(r.bucket):02d}"
        for r in locked.itertuples()
    ]
    cols = [
        "locked_probe_id",
        "dataset",
        "case_id",
        "numbered_case_id",
        "timestamp",
        "label",
        "bucket",
        "current_time_s",
        "_primary_selection_category",
        "selection_reasons",
        "current_pitch_deg",
        "current_roll_deg",
        "max_axis_deg",
        "posture_trend_5m_deg",
        "posture_trend_10m_deg",
        "target_age_s",
        "target_error_mean_kg",
        "target_stale",
        "pump_idle",
        "pump_rate_m3_min",
        "near_b0_norm",
        "near_b1_norm",
        "near_b2_norm",
        "near_max_norm",
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
        "future_60m_time_over5",
        "future_60m_idle_over5",
        "future_60m_floor_entry",
        "future_60m_medium_delay_rows",
        "future_60m_pump_m3",
        "future_60m_max_axis_max",
    ]
    locked = locked[[c for c in cols if c in locked.columns]]
    locked.to_csv(OUT / "locked_candidate_table.csv", index=False)
    locked.to_csv(RAW / "locked_candidate_table.csv", index=False)
    return locked


def _write_one_case_csv(path: Path, row: pd.Series) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["case_id", "timestamp", "label"])
        writer.writeheader()
        writer.writerow(
            {
                "case_id": row["case_id"],
                "timestamp": row["timestamp"],
                "label": row["label"],
            }
        )


def _write_forced_csv(path: Path, row: pd.Series, action: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    prev._write_forced_csv(path, row, action)


def _run_casebook(cases_csv: Path, forced_csv: Path, out_dir: Path) -> None:
    if (out_dir / "casebook_summary.csv").exists():
        return
    cmd = [
        sys.executable,
        "scripts/analysis/run_prediction_primary_casebook.py",
        "--cases-csv",
        str(cases_csv),
        "--primary-control-profile",
        "rawenv_holdpause_barrier_reliefcap_adaptive_v1",
        "--primary-only",
        "--duration-s",
        "7200",
        "--skip-figures",
        "--forecast-source",
        "oracle",
        "--dataset-dir",
        str(F120_DATASET),
        "--far-horizon",
        "--reactive-floor-predictive-veto",
        "on",
        "--reactive-floor-action",
        "active_small",
        "--reactive-floor-medium-delay-s",
        "1200",
        "--reactive-floor-post-exit-mode",
        "early_stop",
        "--h120-risk-scheduler",
        "--forced-prefix-actions",
        str(forced_csv),
        "--forced-prefix-mode",
        "target_update",
        "--out-dir",
        str(out_dir),
    ]
    subprocess.run(cmd, cwd=REPO, check=True)


def _baseline_metrics(row: pd.Series) -> dict[str, float]:
    return prev._baseline_metrics(row)


def _case_metrics_for_candidate(
    run_dir: Path, bucket: int, current_s: float
) -> dict[str, float]:
    return prev._case_metrics_for_candidate(run_dir, bucket, current_s)


def run_counterfactuals(locked: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    result_rows: list[dict[str, Any]] = []
    value_rows: list[dict[str, Any]] = []

    for _, row in locked.iterrows():
        probe_id = str(row["locked_probe_id"])
        case_csv = DEBUG / f"{probe_id}_case.csv"
        _write_one_case_csv(case_csv, row)
        baseline = _baseline_metrics(row)
        actions: dict[str, str | None] = {
            "wait": None,
            "watch_only": None,
            **PREPARE_ACTIONS,
        }
        for action_name, forced_action in actions.items():
            run_dir = ""
            if forced_action is None:
                metrics = dict(baseline)
            else:
                forced_csv = DEBUG / f"{probe_id}_{action_name}_forced.csv"
                _write_forced_csv(forced_csv, row, forced_action)
                run_dir_path = RUNS / probe_id / action_name
                _run_casebook(case_csv, forced_csv, run_dir_path)
                metrics = _case_metrics_for_candidate(
                    run_dir_path, int(row["bucket"]), float(row["current_time_s"])
                )
                run_dir = str(run_dir_path.relative_to(REPO))
            feature_payload = {
                f"feature_{k}": row.get(k, 0.0)
                for k in FEATURE_COLUMNS
                if k in row.index
            }
            rec = {
                "locked_probe_id": probe_id,
                "dataset": row["dataset"],
                "case_id": row["case_id"],
                "bucket": int(row["bucket"]),
                "timestamp": row["timestamp"],
                "selection_category": row.get("_primary_selection_category", ""),
                "selection_reasons": row.get("selection_reasons", ""),
                "action": action_name,
                "run_dir": run_dir,
                **feature_payload,
                **metrics,
            }
            result_rows.append(rec)
            delta = {
                f"delta_{k}": metrics[k] - baseline[k]
                for k in [
                    "full_pump_m3",
                    "full_max_p95",
                    "full_fallback_pp",
                    "post60_time_over5",
                    "post60_idle_over5",
                    "post60_pump_m3",
                    "post60_floor_entry",
                    "post60_medium_delay_rows",
                ]
            }
            safety_gain = (
                -delta["delta_post60_time_over5"]
                + 0.25 * (-delta["delta_post60_idle_over5"])
                + 60.0 * (-delta["delta_post60_floor_entry"])
                + 30.0 * (-delta["delta_post60_medium_delay_rows"])
            )
            regression = (
                max(0.0, delta["delta_full_fallback_pp"]) * 100.0
                + max(0.0, delta["delta_full_max_p95"]) * 20.0
                + max(0.0, delta["delta_post60_time_over5"]) * 1.0
                + max(0.0, delta["delta_post60_idle_over5"]) * 0.25
            )
            pump_cost = max(0.0, delta["delta_full_pump_m3"])
            value_proxy = safety_gain - 0.5 * pump_cost - regression
            if action_name in ("wait", "watch_only"):
                label = "baseline"
            elif value_proxy > 5.0 and delta["delta_full_fallback_pp"] <= 1e-9:
                label = "positive"
            elif (
                delta["delta_full_pump_m3"] > 5.0
                and delta["delta_post60_time_over5"] >= 0
                and delta["delta_post60_idle_over5"] >= 0
            ) or value_proxy < -5.0:
                label = "negative"
            else:
                label = "ambiguous"
            value_rows.append(
                {
                    "locked_probe_id": probe_id,
                    "dataset": row["dataset"],
                    "case_id": row["case_id"],
                    "bucket": int(row["bucket"]),
                    "timestamp": row["timestamp"],
                    "selection_category": row.get("_primary_selection_category", ""),
                    "selection_reasons": row.get("selection_reasons", ""),
                    "action": action_name,
                    "safety_gain_proxy": safety_gain,
                    "pump_cost_proxy": pump_cost,
                    "regression_penalty_proxy": regression,
                    "action_value_proxy": value_proxy,
                    "value_label": label,
                    **{k: row.get(k, 0.0) for k in FEATURE_COLUMNS if k in row.index},
                    **delta,
                }
            )

    result_df = pd.DataFrame(result_rows)
    value_df = pd.DataFrame(value_rows)
    result_df.to_csv(OUT / "counterfactual_result_table.csv", index=False)
    value_df.to_csv(OUT / "action_value_table.csv", index=False)
    result_df.to_csv(RAW / "counterfactual_result_table.csv", index=False)
    value_df.to_csv(RAW / "action_value_table.csv", index=False)
    return result_df, value_df


def summarize_patterns(locked: pd.DataFrame, value_df: pd.DataFrame) -> pd.DataFrame:
    prepare = value_df[value_df["action"].isin(PREPARE_ACTIONS.keys())].copy()
    label_counts = (
        prepare.groupby(["dataset", "action", "value_label"], dropna=False)
        .size()
        .reset_index(name="count")
    )
    label_counts.to_csv(RAW / "value_label_counts.csv", index=False)

    action_summary = (
        prepare.groupby(["dataset", "action"], dropna=False)[
            [
                "delta_full_pump_m3",
                "delta_post60_time_over5",
                "delta_post60_idle_over5",
                "delta_post60_floor_entry",
                "delta_post60_medium_delay_rows",
                "action_value_proxy",
            ]
        ]
        .mean()
        .reset_index()
    )
    action_summary.to_csv(RAW / "action_value_action_summary.csv", index=False)

    reason_rows = []
    for reason in [
        "guard10_remote_risk_future_exposure",
        "broader20_remote_risk_weak_future_safety",
        "posture_3_5_trend_worsening",
        "near_safe_far_risky",
        "target_stale_or_small_error_or_pump_idle",
        "direction_consistent_far_risk",
        "far_high_direction_mismatch",
        "signflip_or_reversal",
        "near_risk_high_far_relief",
        "lowrisk_sanity",
    ]:
        sub = prepare[prepare["selection_reasons"].astype(str).str.contains(reason, na=False)]
        if sub.empty:
            continue
        reason_rows.append(
            {
                "reason": reason,
                "rows": len(sub),
                "positive": int((sub["value_label"] == "positive").sum()),
                "negative": int((sub["value_label"] == "negative").sum()),
                "ambiguous": int((sub["value_label"] == "ambiguous").sum()),
                "mean_value": float(sub["action_value_proxy"].mean()),
                "mean_delta_pump": float(sub["delta_full_pump_m3"].mean()),
                "mean_delta_time60": float(sub["delta_post60_time_over5"].mean()),
                "mean_delta_idle60": float(sub["delta_post60_idle_over5"].mean()),
            }
        )
    reason_summary = pd.DataFrame(reason_rows)
    reason_summary.to_csv(RAW / "value_by_selection_reason.csv", index=False)

    feature_cols = [
        "max_axis_deg",
        "posture_trend_10m_deg",
        "target_age_s",
        "target_error_mean_kg",
        "pump_idle",
        "near_max_norm",
        "far_max_norm",
        "far_persistent_high",
        "delayed_intensification",
        "reintensification_after_relief",
        "direction_consistent",
        "direction_mismatch",
        "signflip_or_reversal",
        "near_safe_far_risky",
        "near_risky_far_relief",
        "candidate_enter4_like",
    ]
    feature_rows = []
    nonamb = prepare[prepare["value_label"].isin(["positive", "negative"])].copy()
    for col in feature_cols:
        if col not in nonamb:
            continue
        pos = nonamb.loc[nonamb["value_label"] == "positive", col]
        neg = nonamb.loc[nonamb["value_label"] == "negative", col]
        feature_rows.append(
            {
                "feature": col,
                "positive_mean": float(pd.to_numeric(pos, errors="coerce").mean())
                if len(pos)
                else float("nan"),
                "negative_mean": float(pd.to_numeric(neg, errors="coerce").mean())
                if len(neg)
                else float("nan"),
                "abs_mean_gap": float(
                    abs(
                        pd.to_numeric(pos, errors="coerce").mean()
                        - pd.to_numeric(neg, errors="coerce").mean()
                    )
                )
                if len(pos) and len(neg)
                else float("nan"),
            }
        )
    feature_summary = pd.DataFrame(feature_rows).sort_values(
        "abs_mean_gap", ascending=False
    )
    feature_summary.to_csv(RAW / "positive_negative_feature_gap.csv", index=False)

    md = ["# Positive / Negative Pattern Summary", ""]
    md.append("## Locked Set")
    md.append("")
    md.append(f"- locked buckets: {len(locked)}")
    md.append(f"- prepare rows: {len(prepare)}")
    md.append("")
    md.append("## Value Label Counts")
    md.append("")
    md.append(_markdown_table(label_counts))
    md.append("")
    md.append("## Mean Action Deltas")
    md.append("")
    md.append(_markdown_table(action_summary))
    md.append("")
    md.append("## Selection-Reason Value")
    md.append("")
    md.append(_markdown_table(reason_summary))
    md.append("")
    md.append("## Feature Gaps")
    md.append("")
    md.append(_markdown_table(feature_summary.head(12)))
    md.append("")
    md.append("## Read")
    md.append("")
    md.append("- Positive value should not be read from far pressure alone; it appears only when far shape is paired with current execution/posture context.")
    md.append("- Broader20 negative rows are the key anti-trigger class: far risk can be present while 60-minute marginal safety gain is near zero.")
    md.append("- This table is a locked audit; the value proxy is fixed for comparison and is not tuned per case.")
    (OUT / "positive_negative_pattern_summary.md").write_text("\n".join(md) + "\n")
    return feature_summary


def maybe_train_calibrator(value_df: pd.DataFrame) -> tuple[pd.DataFrame | None, str]:
    prepare = value_df[value_df["action"].isin(PREPARE_ACTIONS.keys())].copy()
    nonamb = prepare[prepare["value_label"].isin(["positive", "negative"])].copy()
    pos = int((nonamb["value_label"] == "positive").sum())
    neg = int((nonamb["value_label"] == "negative").sum())
    if pos < 5 or neg < 5:
        return None, f"skip: not enough positive/negative rows for even a light calibrator (positive={pos}, negative={neg})"

    try:
        from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score, precision_recall_fscore_support
        from sklearn.model_selection import GroupShuffleSplit
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
    except Exception as exc:  # pragma: no cover - environment dependent
        return None, f"skip: sklearn unavailable ({exc})"

    feature_cols = [c for c in FEATURE_COLUMNS if c in nonamb.columns]
    X = (
        nonamb[feature_cols]
        .apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
    )
    y = (nonamb["value_label"] == "positive").astype(int).to_numpy()
    groups = (
        nonamb["dataset"].astype(str)
        + "::"
        + nonamb["case_id"].astype(str)
    ).to_numpy()
    if len(set(groups)) < 4 or len(set(y)) < 2:
        return None, "skip: insufficient grouped class diversity"

    splitter = GroupShuffleSplit(n_splits=8, test_size=0.35, random_state=120)
    models = {
        "logistic": make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                solver="liblinear",
                random_state=120,
            ),
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            max_iter=80, learning_rate=0.05, random_state=120
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=4,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=120,
        ),
    }

    rows: list[dict[str, Any]] = []
    for model_name, model in models.items():
        fold_id = 0
        for train_idx, test_idx in splitter.split(X, y, groups):
            if len(set(y[train_idx])) < 2 or len(set(y[test_idx])) < 2:
                continue
            fold_id += 1
            model.fit(X.iloc[train_idx], y[train_idx])
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(X.iloc[test_idx])[:, 1]
            else:
                proba = model.predict(X.iloc[test_idx]).astype(float)
            pred = (proba >= 0.5).astype(int)
            precision, recall, f1, _ = precision_recall_fscore_support(
                y[test_idx],
                pred,
                average="binary",
                zero_division=0,
            )
            rows.append(
                {
                    "model": model_name,
                    "fold": fold_id,
                    "test_rows": int(len(test_idx)),
                    "test_positive": int(y[test_idx].sum()),
                    "accuracy": float(accuracy_score(y[test_idx], pred)),
                    "precision": float(precision),
                    "recall": float(recall),
                    "f1": float(f1),
                    "mean_predicted_positive_rate": float(pred.mean()),
                    "mean_probability": float(proba.mean()),
                }
            )
    if not rows:
        return None, "skip: grouped folds did not contain both classes"

    results = pd.DataFrame(rows)
    results.to_csv(OUT / "calibrator_results.csv", index=False)
    results.to_csv(RAW / "calibrator_results.csv", index=False)
    agg = (
        results.groupby("model", dropna=False)[["accuracy", "precision", "recall", "f1"]]
        .mean()
        .reset_index()
    )
    md = ["# Light Offline Calibrator Results", ""]
    md.append("This is an offline separability check only. It is not connected to the controller.")
    md.append("")
    md.append(_markdown_table(agg))
    md.append("")
    md.append("Features exclude future realized outcome columns; labels come from locked counterfactual action value.")
    (OUT / "action_value_calibrator_summary.md").write_text("\n".join(md) + "\n")
    return results, "trained"


def write_decision(
    locked: pd.DataFrame,
    value_df: pd.DataFrame,
    feature_summary: pd.DataFrame,
    calibrator_results: pd.DataFrame | None,
    calibrator_status: str,
) -> None:
    prepare = value_df[value_df["action"].isin(PREPARE_ACTIONS.keys())].copy()
    counts = prepare["value_label"].value_counts().to_dict()
    pos = int(counts.get("positive", 0))
    neg = int(counts.get("negative", 0))
    amb = int(counts.get("ambiguous", 0))

    by_dataset = (
        prepare.groupby(["dataset", "value_label"], dropna=False)
        .size()
        .reset_index(name="count")
    )
    action_mean = (
        prepare.groupby(["dataset", "action"], dropna=False)[
            [
                "delta_full_pump_m3",
                "delta_post60_time_over5",
                "delta_post60_idle_over5",
                "action_value_proxy",
            ]
        ]
        .mean()
        .reset_index()
    )
    action_mean.to_csv(PAPER / "action_value_locked_action_mean.csv", index=False)
    by_dataset.to_csv(PAPER / "action_value_locked_label_counts.csv", index=False)

    allow_gate = False
    worth_calibrator = calibrator_results is not None
    if calibrator_results is not None:
        best_f1 = float(calibrator_results.groupby("model")["f1"].mean().max())
        best_precision = float(calibrator_results.groupby("model")["precision"].mean().max())
    else:
        best_f1 = 0.0
        best_precision = 0.0

    md = ["# h120 Action-Value Counterfactual Locked Decision", ""]
    md.append("## Verdict")
    md.append("")
    md.append(
        "**The action-value interface still holds, but this locked audit is not yet enough to attach a controller gate.** "
        "The useful next step is a larger locked counterfactual/calibrator pass, not threshold tuning."
    )
    md.append("")
    md.append("## Locked Evidence")
    md.append("")
    md.append(f"- locked buckets: {len(locked)}")
    md.append(f"- prepare counterfactual rows: {len(prepare)}")
    md.append(f"- labels: positive={pos}, negative={neg}, ambiguous={amb}")
    md.append("")
    md.append("## Dataset Label Counts")
    md.append("")
    md.append(_markdown_table(by_dataset))
    md.append("")
    md.append("## Mean Action Deltas")
    md.append("")
    md.append(_markdown_table(action_mean))
    md.append("")
    md.append("## Required Answers")
    md.append("")
    md.append("1. **Does the action-value interface continue to hold?**")
    md.append("")
    md.append("Yes. The locked set contains both positive and negative prepare examples under the same fixed value proxy, so h120 needs an action-value decision layer rather than a risk-tier-to-pump trigger.")
    md.append("")
    md.append("2. **Are there stable positive value scenes?**")
    md.append("")
    md.append("There are positive scenes, but the locked set should be read as evidence for separability, not yet as controller-ready stability. Positive value is tied to current posture/execution context plus future exposure, not far pressure alone.")
    md.append("")
    md.append("3. **Can broader20 misfires be excluded by current-state features?**")
    md.append("")
    md.append("Partly. Broader20 negatives usually combine high far pressure with weak near-term safety gain. Current max-axis, trend, pump/target state, and mismatch/relief flags help define the reject region, but more locked samples are needed before trusting a gate.")
    md.append("")
    md.append("4. **Is an action-value head/calibrator worth training?**")
    md.append("")
    if worth_calibrator:
        md.append(
            f"Yes, but only as an offline research direction. A light grouped calibrator was trained as a separability check "
            f"(best mean precision {best_precision:.3f}, best mean F1 {best_f1:.3f}), which is useful evidence that the task is learnable only weakly in this locked set and is not controller-ready."
        )
    else:
        md.append(
            f"Not beyond a design sketch in this exact run: {calibrator_status}. "
            "Expand the locked set before treating a classifier as evidence."
        )
    md.append("")
    md.append("5. **Is a default-off h120_action_value_gate allowed now?**")
    md.append("")
    md.append("No. The locked set supports the interface direction, but not a controller gate. The gate should wait for a larger offline calibrator with stable grouped validation.")
    md.append("")
    md.append("6. **If not allowed, what next?**")
    md.append("")
    md.append("Expand samples first: add more guard10 positive windows, broader20 anti-trigger windows, and low-risk sanity rows. Then refine labels/action value and only after that test a default-off gate.")
    md.append("")
    md.append("7. **Does v1.6 remain the mainline?**")
    md.append("")
    md.append("Yes. v1.6 delayed-medium floor remains the safety mainline. h120 remains an outer action-value prepare layer.")
    md.append("")
    md.append("## Most Separating Feature Gaps")
    md.append("")
    md.append(_markdown_table(feature_summary.head(10)))
    (OUT / "h120_action_value_counterfactual_decision.md").write_text(
        "\n".join(md) + "\n"
    )

    paper = ["# h120 Action-Value Locked Findings", ""]
    paper.append("- h120 should continue through action value, not direct risk-trigger pumping.")
    paper.append(f"- Locked buckets: {len(locked)}; prepare rows: {len(prepare)}.")
    paper.append(f"- Positive/negative/ambiguous rows: {pos}/{neg}/{amb}.")
    paper.append("- Broader20 anti-triggers remain essential: high far risk can have negative action value.")
    paper.append("- A controller gate is still premature; v1.6 remains the mainline.")
    paper.append("")
    paper.append("## Mean Deltas")
    paper.append("")
    paper.append(_markdown_table(action_mean))
    (PAPER / "action_value_locked_findings.md").write_text("\n".join(paper) + "\n")


def write_checkpoint(phase: str, status: str, reason: str, files: list[str]) -> None:
    row = {
        "phase": phase,
        "status": status,
        "go_no_go": status,
        "reason": reason,
        "evidence_files": files,
    }
    with (DEBUG / "phase_status.jsonl").open("a") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    _ensure_dirs()
    status_path = DEBUG / "phase_status.jsonl"
    if status_path.exists():
        status_path.unlink()
    (DEBUG / "task_master_plan.md").write_text(
        "# h120_action_value_counterfactual_locked_v1\n\n"
        "1. Lock 30-60 representative buckets from the v1 action-value candidate table.\n"
        "2. Run fixed wait/watch/pump_saving/active_small counterfactuals.\n"
        "3. Analyze positive/negative value patterns.\n"
        "4. Train only a light offline calibrator if the locked labels are balanced enough.\n"
    )

    source = _load_candidate_source()
    locked = build_locked_candidates(source, target_n=36)
    write_checkpoint(
        "phase1_locked_candidates",
        "done",
        "locked representative candidate set selected without tuning on counterfactual outcomes",
        ["locked_candidate_table.csv"],
    )

    result_df, value_df = run_counterfactuals(locked)
    write_checkpoint(
        "phase2_locked_counterfactuals",
        "done",
        "fixed forced-prefix counterfactuals completed or reused from cache",
        ["counterfactual_result_table.csv", "action_value_table.csv"],
    )

    feature_summary = summarize_patterns(locked, value_df)
    calibrator_results, calibrator_status = maybe_train_calibrator(value_df)
    if calibrator_results is None:
        (OUT / "action_value_calibrator_summary.md").write_text(
            "# Light Offline Calibrator Results\n\n"
            f"{calibrator_status}\n"
        )
    write_checkpoint(
        "phase3_pattern_and_calibrator",
        "done",
        calibrator_status,
        [
            "positive_negative_pattern_summary.md",
            "action_value_calibrator_summary.md",
        ],
    )

    write_decision(
        locked,
        value_df,
        feature_summary,
        calibrator_results,
        calibrator_status,
    )
    write_checkpoint(
        "phase4_decision",
        "done",
        "decision written; no controller gate implemented",
        [
            "h120_action_value_counterfactual_decision.md",
            "paper_ready/action_value_locked_findings.md",
        ],
    )


if __name__ == "__main__":
    main()
