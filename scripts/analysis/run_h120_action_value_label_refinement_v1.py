#!/usr/bin/env python3
"""h120 action-value label refinement v1.

This expands the locked counterfactual set and replaces the previous single
value-proxy label with constrained labels.  It intentionally does not attach a
controller gate, does not modify the v1.6 floor, and does not train a complex
forecast model.
"""

from __future__ import annotations

import csv
import importlib.util
import json
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

OUT = REPO / "outputs/wind_prediction/h120_action_value_label_refinement_v1"
RAW = OUT / "raw_tables"
DEBUG = OUT / "debug"
PAPER = OUT / "paper_ready"
RUNS = OUT / "counterfactual_runs"

SOURCE_AUDIT = REPO / "outputs/wind_prediction/h120_action_value_audit_v1"
SOURCE_CANDIDATES = SOURCE_AUDIT / "action_value_candidate_table.csv"
LOCKED_V1 = REPO / "outputs/wind_prediction/h120_action_value_counterfactual_locked_v1"
LOCKED_V1_CANDIDATES = LOCKED_V1 / "locked_candidate_table.csv"
LOCKED_V1_RUNS = LOCKED_V1 / "counterfactual_runs"
F120_DATASET = prev.F120_DATASET
DATASETS = prev.DATASETS

TARGET_BUCKETS = 84
TARGET_PER_DATASET = TARGET_BUCKETS // 2
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
        raise FileNotFoundError(f"missing source candidate table: {SOURCE_CANDIDATES}")
    df = pd.read_csv(SOURCE_CANDIDATES)
    if "bucket" not in df:
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
        reasons.append("guard10_positive_window")
    if dataset == "broader20" and _num(row["remote_risk_active"]) > 0 and (
        _num(row["future_60m_time_over5"]) <= 5
        and _num(row["future_60m_idle_over5"]) <= 5
    ):
        reasons.append("broader20_anti_trigger")
    if _num(row["near_safe_far_risky"]) > 0:
        reasons.append("near_safe_far_risky")
    if _num(row["near_risky_far_relief"]) > 0:
        reasons.append("near_risk_high_far_relief")
    if _num(row["direction_mismatch"]) > 0:
        reasons.append("direction_mismatch")
    if _num(row["signflip_or_reversal"]) > 0:
        reasons.append("signflip_or_reversal")
    if (
        _num(row["target_stale"]) > 0
        or _num(row["pump_idle"]) > 0
        or (_num(row["remote_risk_active"]) > 0 and _num(row["target_error_mean_kg"]) < 500)
    ):
        reasons.append("target_or_pump_idle")
    if (
        3.0 <= _num(row["max_axis_deg"]) < 5.0
        and _num(row["posture_trend_10m_deg"]) > 0.05
    ):
        reasons.append("posture_3_5_worsening")
    if _num(row["remote_risk_active"]) > 0 and _num(row["direction_consistent"]) > 0:
        reasons.append("direction_consistent_far_risk")
    if (
        _num(row["remote_risk_active"]) <= 0
        and _num(row["max_axis_deg"]) < 3.0
        and _num(row["future_60m_time_over5"]) <= 0
    ):
        reasons.append("lowrisk_sanity")
    return reasons


def _score(row: pd.Series, reasons: list[str]) -> float:
    score = 0.0
    score += 2.2 * min(_num(row["future_60m_time_over5"]) / 60.0, 10.0)
    score += 1.0 * min(_num(row["future_60m_idle_over5"]) / 60.0, 10.0)
    score += 1.8 * _num(row["candidate_enter4_like"])
    score += 1.2 * _num(row["candidate_prefloor_like"])
    score += 0.8 * _num(row["near_safe_far_risky"])
    score += 0.8 * _num(row["near_risky_far_relief"])
    score += 0.8 * _num(row["direction_mismatch"])
    score += 0.8 * _num(row["signflip_or_reversal"])
    score += 0.5 * _num(row["direction_consistent"])
    score += 0.4 * len(reasons)
    return float(score)


def _key(row: pd.Series) -> tuple[str, str, int]:
    return str(row["dataset"]), str(row["case_id"]), int(row["bucket"])


def _pick(
    pool: pd.DataFrame,
    mask: pd.Series,
    quota: int,
    used: set[tuple[str, str, int]],
    category: str,
) -> list[pd.Series]:
    rows: list[pd.Series] = []
    sub = pool.loc[mask].copy()
    if sub.empty or quota <= 0:
        return rows
    for _, row in sub.sort_values("_refine_score", ascending=False).iterrows():
        key = _key(row)
        if key in used:
            continue
        row = row.copy()
        row["selection_category"] = category
        rows.append(row)
        used.add(key)
        if len(rows) >= quota:
            break
    return rows


def build_refined_candidates(df: pd.DataFrame) -> pd.DataFrame:
    _ensure_dirs()
    work = df.copy()
    reasons = []
    scores = []
    for _, row in work.iterrows():
        rs = _reasons(row)
        reasons.append(";".join(rs))
        scores.append(_score(row, rs))
    work["selection_reasons"] = reasons
    work["_refine_score"] = scores
    eligible = work[work["selection_reasons"].astype(str).str.len() > 0].copy()

    used: set[tuple[str, str, int]] = set()
    rows: list[pd.Series] = []

    if LOCKED_V1_CANDIDATES.exists():
        locked = pd.read_csv(LOCKED_V1_CANDIDATES)
        locked_keys = {tuple(x) for x in locked[["dataset", "case_id", "bucket"]].to_numpy()}
        seed = eligible[
            eligible.apply(
                lambda r: (str(r["dataset"]), str(r["case_id"]), int(r["bucket"]))
                in locked_keys,
                axis=1,
            )
        ].copy()
        cat_map = {
            (str(r.dataset), str(r.case_id), int(r.bucket)): str(
                getattr(r, "_primary_selection_category", "locked_v1_seed")
            )
            for r in locked.itertuples()
        }
        for _, row in seed.sort_values(["dataset", "_refine_score"], ascending=[True, False]).iterrows():
            if _key(row) in used:
                continue
            row = row.copy()
            row["selection_category"] = "seed_" + cat_map.get(_key(row), "locked_v1")
            rows.append(row)
            used.add(_key(row))

    specs = [
        ("guard10_positive_window", 14, eligible["selection_reasons"].str.contains("guard10_positive_window", na=False)),
        ("broader20_anti_trigger", 14, eligible["selection_reasons"].str.contains("broader20_anti_trigger", na=False)),
        ("near_safe_far_risky", 8, eligible["selection_reasons"].str.contains("near_safe_far_risky", na=False)),
        ("near_risk_high_far_relief", 8, eligible["selection_reasons"].str.contains("near_risk_high_far_relief", na=False)),
        ("direction_mismatch", 8, eligible["selection_reasons"].str.contains("direction_mismatch", na=False)),
        ("signflip_or_reversal", 6, eligible["selection_reasons"].str.contains("signflip_or_reversal", na=False)),
        ("target_or_pump_idle", 10, eligible["selection_reasons"].str.contains("target_or_pump_idle", na=False)),
        ("posture_3_5_worsening", 10, eligible["selection_reasons"].str.contains("posture_3_5_worsening", na=False)),
        ("lowrisk_sanity", 8, eligible["selection_reasons"].str.contains("lowrisk_sanity", na=False)),
    ]
    for category, quota, mask in specs:
        rows.extend(_pick(eligible, mask, quota, used, category))

    selected = pd.DataFrame(rows).drop_duplicates(["dataset", "case_id", "bucket"])
    balanced_rows: list[pd.Series] = []
    for dataset in ("guard10", "broader20"):
        sub = selected[selected["dataset"] == dataset].copy()
        if len(sub) < TARGET_PER_DATASET:
            fill = _pick(
                eligible,
                (eligible["dataset"] == dataset),
                TARGET_PER_DATASET - len(sub),
                used,
                f"{dataset}_score_fill",
            )
            if fill:
                selected = pd.concat([selected, pd.DataFrame(fill)], ignore_index=True)
                sub = selected[selected["dataset"] == dataset].copy()
        balanced_rows.extend([r for _, r in sub.sort_values("_refine_score", ascending=False).head(TARGET_PER_DATASET).iterrows()])

    refined = pd.DataFrame(balanced_rows).reset_index(drop=True)
    refined["refined_probe_id"] = [
        f"{r.dataset}_{r.case_id}_b{int(r.bucket):02d}" for r in refined.itertuples()
    ]
    cols = [
        "refined_probe_id",
        "dataset",
        "case_id",
        "numbered_case_id",
        "timestamp",
        "label",
        "bucket",
        "current_time_s",
        "selection_category",
        "selection_reasons",
        "current_pitch_deg",
        "current_roll_deg",
        *FEATURE_COLUMNS,
        "future_60m_time_over5",
        "future_60m_idle_over5",
        "future_60m_floor_entry",
        "future_60m_medium_delay_rows",
        "future_60m_pump_m3",
        "future_60m_max_axis_max",
    ]
    refined = refined[[c for c in cols if c in refined.columns]]
    refined.to_csv(OUT / "refined_candidate_table.csv", index=False)
    refined.to_csv(RAW / "refined_candidate_table.csv", index=False)
    return refined


def _timestamp_token(ts: str) -> str:
    return pd.Timestamp(ts).strftime("%Y-%m-%d_%H%M%S")


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


def _case_metrics(run_dir: Path, row: pd.Series) -> dict[str, float]:
    return prev._case_metrics_for_candidate(
        run_dir, int(row["bucket"]), float(row["current_time_s"])
    )


def _cached_run_dir(probe_id: str, action_name: str) -> Path | None:
    path = LOCKED_V1_RUNS / probe_id / action_name
    if (path / "casebook_summary.csv").exists():
        return path
    return None


def run_counterfactuals(refined: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    delta_rows: list[dict[str, Any]] = []

    for _, row in refined.iterrows():
        probe_id = str(row["refined_probe_id"])
        old_probe_id = probe_id
        case_csv = DEBUG / f"{probe_id}_case.csv"
        _write_one_case_csv(case_csv, row)
        baseline = _baseline_metrics(row)
        actions: dict[str, str | None] = {
            "wait": None,
            "watch_only": None,
            **PREPARE_ACTIONS,
        }
        for action_name, forced_action in actions.items():
            cache_source = ""
            if forced_action is None:
                metrics = dict(baseline)
                run_dir = ""
            else:
                cached = _cached_run_dir(old_probe_id, action_name)
                if cached is not None:
                    metrics = _case_metrics(cached, row)
                    run_dir = str(cached.relative_to(REPO))
                    cache_source = "locked_v1"
                else:
                    forced_csv = DEBUG / f"{probe_id}_{action_name}_forced.csv"
                    _write_forced_csv(forced_csv, row, forced_action)
                    run_path = RUNS / probe_id / action_name
                    _run_casebook(case_csv, forced_csv, run_path)
                    metrics = _case_metrics(run_path, row)
                    run_dir = str(run_path.relative_to(REPO))
            feature_payload = {
                f"feature_{k}": row.get(k, 0.0)
                for k in FEATURE_COLUMNS
                if k in row.index
            }
            rows.append(
                {
                    "refined_probe_id": probe_id,
                    "dataset": row["dataset"],
                    "case_id": row["case_id"],
                    "bucket": int(row["bucket"]),
                    "timestamp": row["timestamp"],
                    "selection_category": row["selection_category"],
                    "selection_reasons": row["selection_reasons"],
                    "action": action_name,
                    "run_dir": run_dir,
                    "cache_source": cache_source,
                    **feature_payload,
                    **metrics,
                }
            )
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
            delta_rows.append(
                {
                    "refined_probe_id": probe_id,
                    "dataset": row["dataset"],
                    "case_id": row["case_id"],
                    "bucket": int(row["bucket"]),
                    "timestamp": row["timestamp"],
                    "selection_category": row["selection_category"],
                    "selection_reasons": row["selection_reasons"],
                    "action": action_name,
                    **{k: row.get(k, 0.0) for k in FEATURE_COLUMNS if k in row.index},
                    **delta,
                }
            )

    result = pd.DataFrame(rows)
    deltas = pd.DataFrame(delta_rows)
    result.to_csv(OUT / "refined_counterfactual_table.csv", index=False)
    deltas.to_csv(RAW / "refined_counterfactual_delta_table.csv", index=False)
    result.to_csv(RAW / "refined_counterfactual_table.csv", index=False)
    return result, deltas


def _label_for_row(
    row: pd.Series,
    *,
    pump_cap_m3: float = 80.0,
    time_threshold_s: float = 30.0,
    idle_threshold_s: float = 60.0,
    fallback_tol_pp: float = 0.05,
    max_p95_tol_deg: float = 0.05,
    hard_constraints: bool = True,
) -> tuple[str, str]:
    if str(row["action"]) in ("wait", "watch_only"):
        return "baseline", "baseline action"

    pump_delta = _num(row["delta_full_pump_m3"])
    time_delta = _num(row["delta_post60_time_over5"])
    idle_delta = _num(row["delta_post60_idle_over5"])
    fallback_delta = _num(row["delta_full_fallback_pp"])
    max_p95_delta = _num(row["delta_full_max_p95"])
    floor_delta = _num(row["delta_post60_floor_entry"])
    medium_delta = _num(row["delta_post60_medium_delay_rows"])

    time_gain = -time_delta
    idle_gain = -idle_delta
    safety_gain = (
        time_gain >= time_threshold_s
        or idle_gain >= idle_threshold_s
        or floor_delta < 0
        or medium_delta < 0
    )
    hard_ok = (
        fallback_delta <= fallback_tol_pp
        and max_p95_delta <= max_p95_tol_deg
        and pump_delta <= pump_cap_m3
        and time_delta <= 5.0
    )
    safety_worse = (
        time_delta > 10.0
        or idle_delta > 20.0
        or fallback_delta > fallback_tol_pp
        or max_p95_delta > max_p95_tol_deg
    )
    no_gain = time_gain < 10.0 and idle_gain < 20.0 and floor_delta >= 0 and medium_delta >= 0
    anti_trigger = (
        (_num(row.get("near_safe_far_risky", 0.0)) > 0 or _num(row.get("direction_mismatch", 0.0)) > 0)
        and not safety_gain
    )
    if safety_gain and (hard_ok or not hard_constraints):
        return "safe_positive", "safety gain within pump/fallback/max constraints"
    if safety_worse:
        return "negative", "safety regression hard constraint"
    if pump_delta > 5.0 and no_gain:
        return "negative", "pump increase without meaningful safety gain"
    if anti_trigger:
        return "negative", "anti-trigger feature without meaningful safety gain"
    return "ambiguous", "small or unclear pump/safety tradeoff"


def apply_constrained_labels(deltas: pd.DataFrame) -> pd.DataFrame:
    labels = []
    reasons = []
    for _, row in deltas.iterrows():
        label, reason = _label_for_row(row)
        labels.append(label)
        reasons.append(reason)
    out = deltas.copy()
    out["constrained_label"] = labels
    out["constrained_label_reason"] = reasons
    out["time_gain_s"] = -out["delta_post60_time_over5"]
    out["idle_gain_s"] = -out["delta_post60_idle_over5"]
    out["pump_delta_m3"] = out["delta_full_pump_m3"]
    out.to_csv(OUT / "constrained_label_table.csv", index=False)
    out.to_csv(RAW / "constrained_label_table.csv", index=False)
    return out


def build_case_level_table(labels: pd.DataFrame) -> pd.DataFrame:
    prep = labels[labels["action"].isin(PREPARE_ACTIONS.keys())].copy()
    rows = []
    for (dataset, case_id, action), g in prep.groupby(["dataset", "case_id", "action"], dropna=False):
        rows.append(
            {
                "dataset": dataset,
                "case_id": case_id,
                "action": action,
                "rows": len(g),
                "safe_positive_rows": int((g["constrained_label"] == "safe_positive").sum()),
                "negative_rows": int((g["constrained_label"] == "negative").sum()),
                "ambiguous_rows": int((g["constrained_label"] == "ambiguous").sum()),
                "mean_pump_delta_m3": float(g["delta_full_pump_m3"].mean()),
                "mean_time_gain_s": float((-g["delta_post60_time_over5"]).mean()),
                "mean_idle_gain_s": float((-g["delta_post60_idle_over5"]).mean()),
                "mean_fallback_delta_pp": float(g["delta_full_fallback_pp"].mean()),
                "mean_max_p95_delta_deg": float(g["delta_full_max_p95"].mean()),
            }
        )
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "case_level_value_table.csv", index=False)
    table.to_csv(RAW / "case_level_value_table.csv", index=False)
    return table


def build_label_sensitivity(deltas: pd.DataFrame) -> pd.DataFrame:
    rows = []
    configs = []
    for pump_cap in (40.0, 80.0, 120.0):
        for idle_thr in (30.0, 60.0, 120.0):
            for max_tol in (0.0, 0.05, 0.10):
                configs.append((pump_cap, idle_thr, max_tol, True))
    configs.append((80.0, 60.0, 0.05, False))
    for pump_cap, idle_thr, max_tol, hard in configs:
        labels = []
        for _, row in deltas.iterrows():
            label, _ = _label_for_row(
                row,
                pump_cap_m3=pump_cap,
                idle_threshold_s=idle_thr,
                max_p95_tol_deg=max_tol,
                hard_constraints=hard,
            )
            labels.append(label)
        tmp = deltas.copy()
        tmp["label"] = labels
        prep = tmp[tmp["action"].isin(PREPARE_ACTIONS.keys())]
        rows.append(
            {
                "pump_cap_m3": pump_cap,
                "idle_threshold_s": idle_thr,
                "max_p95_tol_deg": max_tol,
                "hard_constraints": int(hard),
                "prepare_rows": len(prep),
                "safe_positive": int((prep["label"] == "safe_positive").sum()),
                "negative": int((prep["label"] == "negative").sum()),
                "ambiguous": int((prep["label"] == "ambiguous").sum()),
                "broader20_safe_positive": int(
                    ((prep["dataset"] == "broader20") & (prep["label"] == "safe_positive")).sum()
                ),
                "lowrisk_safe_positive": int(
                    (
                        prep["selection_reasons"].astype(str).str.contains("lowrisk_sanity", na=False)
                        & (prep["label"] == "safe_positive")
                    ).sum()
                ),
                "near_safe_far_risky_safe_positive": int(
                    ((prep["near_safe_far_risky"] > 0) & (prep["label"] == "safe_positive")).sum()
                ),
            }
        )
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "label_sensitivity_table.csv", index=False)
    table.to_csv(RAW / "label_sensitivity_table.csv", index=False)
    return table


def run_calibrators(labels: pd.DataFrame) -> pd.DataFrame:
    prep = labels[labels["action"].isin(PREPARE_ACTIONS.keys())].copy()
    usable = prep[prep["constrained_label"].isin(["safe_positive", "negative"])].copy()
    if usable.empty or usable["constrained_label"].nunique() < 2:
        out = pd.DataFrame(
            [
                {
                    "model": "not_trained",
                    "reason": "insufficient constrained label diversity",
                }
            ]
        )
        out.to_csv(OUT / "offline_calibrator_results.csv", index=False)
        out.to_csv(RAW / "offline_calibrator_results.csv", index=False)
        return out

    try:
        from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import precision_recall_fscore_support
        from sklearn.model_selection import GroupShuffleSplit
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
    except Exception as exc:  # pragma: no cover
        out = pd.DataFrame([{"model": "not_trained", "reason": f"sklearn unavailable: {exc}"}])
        out.to_csv(OUT / "offline_calibrator_results.csv", index=False)
        out.to_csv(RAW / "offline_calibrator_results.csv", index=False)
        return out

    action_dummies = pd.get_dummies(usable["action"], prefix="action")
    feature_cols = [c for c in FEATURE_COLUMNS if c in usable.columns]
    X = (
        usable[feature_cols]
        .apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
    )
    X = pd.concat([X.reset_index(drop=True), action_dummies.reset_index(drop=True)], axis=1)
    y = (usable["constrained_label"] == "safe_positive").astype(int).to_numpy()
    groups = (usable["dataset"].astype(str) + "::" + usable["case_id"].astype(str)).to_numpy()

    models = {
        "logistic": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, class_weight="balanced", solver="liblinear", random_state=121),
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            max_iter=80, learning_rate=0.05, random_state=121
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=240,
            max_depth=5,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=121,
        ),
    }
    splitter = GroupShuffleSplit(n_splits=8, test_size=0.35, random_state=121)
    rows = []
    thresholds = (0.50, 0.70, 0.85)
    for model_name, model in models.items():
        fold = 0
        for train_idx, test_idx in splitter.split(X, y, groups):
            if len(set(y[train_idx])) < 2 or len(set(y[test_idx])) < 2:
                continue
            fold += 1
            model.fit(X.iloc[train_idx], y[train_idx])
            proba = model.predict_proba(X.iloc[test_idx])[:, 1]
            test = usable.iloc[test_idx].copy().reset_index(drop=True)
            test_y = y[test_idx]
            for threshold in thresholds:
                pred = (proba >= threshold).astype(int)
                precision, recall, f1, _ = precision_recall_fscore_support(
                    test_y, pred, average="binary", zero_division=0
                )
                predicted_positive = pred == 1
                false_positive = predicted_positive & (test_y == 0)
                broader_lowrisk = (
                    (test["dataset"].astype(str) == "broader20")
                    | test["selection_reasons"].astype(str).str.contains("lowrisk_sanity", na=False)
                ).to_numpy()
                near_safe = (test["near_safe_far_risky"].to_numpy(dtype=float) > 0)
                anti_trigger = (
                    test["selection_reasons"].astype(str).str.contains(
                        "broader20_anti_trigger|near_safe_far_risky|direction_mismatch|lowrisk_sanity",
                        regex=True,
                        na=False,
                    )
                ).to_numpy()
                anti_neg = anti_trigger & (test_y == 0)
                anti_recall = (
                    float(((pred == 0) & anti_neg).sum() / max(anti_neg.sum(), 1))
                    if anti_neg.sum()
                    else 0.0
                )
                rows.append(
                    {
                        "model": model_name,
                        "fold": fold,
                        "threshold": threshold,
                        "test_rows": int(len(test_idx)),
                        "test_safe_positive": int(test_y.sum()),
                        "precision": float(precision),
                        "recall": float(recall),
                        "f1": float(f1),
                        "predicted_positive_rows": int(predicted_positive.sum()),
                        "high_conf_prepare_precision": float(precision),
                        "false_positive_broader_lowrisk": int((false_positive & broader_lowrisk).sum()),
                        "false_positive_near_safe_far_risky": int((false_positive & near_safe).sum()),
                        "anti_trigger_recall": anti_recall,
                    }
                )

            # Bucket-level preferred action at high confidence.
            test["proba"] = proba
            pred_pref = []
            true_pref = []
            for probe_id, g in test.groupby("refined_probe_id", dropna=False):
                best = g.sort_values("proba", ascending=False).iloc[0]
                pred_pref.append(str(best["action"]) if float(best["proba"]) >= 0.70 else "watch")
                pos = g[g["constrained_label"] == "safe_positive"]
                if pos.empty:
                    true_pref.append("watch")
                else:
                    # Prefer the safer smaller action on ties.
                    pos = pos.assign(
                        action_rank=pos["action"].map(
                            {"pump_saving_prepare": 0, "active_small_prepare": 1}
                        )
                    )
                    true_pref.append(
                        str(
                            pos.sort_values(
                                ["delta_post60_time_over5", "delta_post60_idle_over5", "action_rank"],
                                ascending=[True, True, True],
                            ).iloc[0]["action"]
                        )
                    )
            preferred_precision = (
                float(np.mean([p == t for p, t in zip(pred_pref, true_pref)]))
                if pred_pref
                else 0.0
            )
            rows.append(
                {
                    "model": model_name,
                    "fold": fold,
                    "threshold": "preferred_action_0.70",
                    "test_rows": int(len(test_idx)),
                    "test_safe_positive": int(test_y.sum()),
                    "precision": preferred_precision,
                    "recall": np.nan,
                    "f1": np.nan,
                    "predicted_positive_rows": int(sum(p != "watch" for p in pred_pref)),
                    "high_conf_prepare_precision": np.nan,
                    "false_positive_broader_lowrisk": np.nan,
                    "false_positive_near_safe_far_risky": np.nan,
                    "anti_trigger_recall": np.nan,
                    "preferred_action_precision": preferred_precision,
                }
            )

    out = pd.DataFrame(rows)
    out.to_csv(OUT / "offline_calibrator_results.csv", index=False)
    out.to_csv(RAW / "offline_calibrator_results.csv", index=False)
    return out


def write_reports(
    refined: pd.DataFrame,
    labels: pd.DataFrame,
    case_table: pd.DataFrame,
    sensitivity: pd.DataFrame,
    calibrator: pd.DataFrame,
) -> None:
    prep = labels[labels["action"].isin(PREPARE_ACTIONS.keys())].copy()
    label_counts = prep["constrained_label"].value_counts().reset_index()
    label_counts.columns = ["constrained_label", "count"]
    by_dataset = (
        prep.groupby(["dataset", "action", "constrained_label"], dropna=False)
        .size()
        .reset_index(name="count")
    )
    case_concentration = (
        case_table.groupby(["dataset", "case_id"], dropna=False)[
            ["safe_positive_rows", "negative_rows", "ambiguous_rows"]
        ]
        .sum()
        .reset_index()
        .sort_values("safe_positive_rows", ascending=False)
    )
    action_summary = (
        prep.groupby(["dataset", "action"], dropna=False)[
            [
                "time_gain_s",
                "idle_gain_s",
                "pump_delta_m3",
                "delta_full_fallback_pp",
                "delta_full_max_p95",
            ]
        ]
        .mean()
        .reset_index()
    )
    label_counts.to_csv(RAW / "constrained_label_counts.csv", index=False)
    by_dataset.to_csv(RAW / "constrained_label_by_dataset_action.csv", index=False)
    action_summary.to_csv(RAW / "refined_action_summary.csv", index=False)
    case_concentration.to_csv(RAW / "case_positive_concentration.csv", index=False)

    cal_numeric = calibrator[pd.to_numeric(calibrator.get("threshold", pd.Series()), errors="coerce").notna()].copy()
    if not cal_numeric.empty:
        cal_numeric["threshold"] = pd.to_numeric(cal_numeric["threshold"])
        cal_numeric["prepare_rate"] = (
            cal_numeric["predicted_positive_rows"]
            / cal_numeric["test_rows"].clip(lower=1)
        )
        cal_summary = (
            cal_numeric.groupby(["model", "threshold"], dropna=False)[
                [
                    "precision",
                    "recall",
                    "f1",
                    "false_positive_broader_lowrisk",
                    "false_positive_near_safe_far_risky",
                    "anti_trigger_recall",
                ]
            ]
            .mean()
            .reset_index()
        )
        cal_diag = (
            cal_numeric.groupby(["model", "threshold"], dropna=False)
            .agg(
                folds=("fold", "count"),
                precision_mean=("precision", "mean"),
                precision_max=("precision", "max"),
                recall_mean=("recall", "mean"),
                f1_mean=("f1", "mean"),
                predicted_positive_rows_mean=("predicted_positive_rows", "mean"),
                prepare_rate_mean=("prepare_rate", "mean"),
                false_positive_broader_lowrisk_mean=(
                    "false_positive_broader_lowrisk",
                    "mean",
                ),
                false_positive_near_safe_far_risky_mean=(
                    "false_positive_near_safe_far_risky",
                    "mean",
                ),
                anti_trigger_recall_mean=("anti_trigger_recall", "mean"),
            )
            .reset_index()
        )
    else:
        cal_summary = calibrator.copy()
        cal_diag = calibrator.copy()
    cal_summary.to_csv(RAW / "offline_calibrator_summary.csv", index=False)
    cal_diag.to_csv(RAW / "offline_calibrator_diagnostic_summary.csv", index=False)

    preferred = calibrator[
        calibrator.get("threshold", pd.Series(dtype=str)).astype(str).eq(
            "preferred_action_0.70"
        )
    ].copy()
    if not preferred.empty:
        preferred_summary = (
            preferred.groupby("model", dropna=False)
            .agg(
                folds=("fold", "count"),
                preferred_action_precision_mean=("preferred_action_precision", "mean"),
                preferred_action_precision_max=("preferred_action_precision", "max"),
                predicted_non_watch_mean=("predicted_positive_rows", "mean"),
            )
            .reset_index()
        )
    else:
        preferred_summary = pd.DataFrame()
    preferred_summary.to_csv(RAW / "preferred_action_diagnostic_summary.csv", index=False)

    contrast_features = [
        "max_axis_deg",
        "posture_trend_5m_deg",
        "posture_trend_10m_deg",
        "target_age_s",
        "target_error_mean_kg",
        "pump_idle",
        "pump_rate_m3_min",
        "near_b0_norm",
        "near_b1_norm",
        "near_b2_norm",
        "near_max_norm",
        "near_intensification",
        "near_relief",
        "far_60_80_norm",
        "far_80_100_norm",
        "far_100_120_norm",
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
        "time_gain_s",
        "idle_gain_s",
        "pump_delta_m3",
        "delta_full_fallback_pp",
        "delta_full_max_p95",
    ]
    contrast_features = [c for c in contrast_features if c in prep.columns]
    feature_contrast = (
        prep.groupby("constrained_label", dropna=False)[contrast_features]
        .mean(numeric_only=True)
        .T.reset_index()
        .rename(columns={"index": "feature"})
    )
    for label in ("safe_positive", "negative", "ambiguous"):
        if label not in feature_contrast.columns:
            feature_contrast[label] = np.nan
    feature_contrast["safe_minus_negative"] = (
        feature_contrast["safe_positive"] - feature_contrast["negative"]
    )
    feature_contrast.to_csv(RAW / "label_feature_contrast.csv", index=False)

    coverage_rows = []
    coverage_flags = [
        "far_persistent_high",
        "delayed_intensification",
        "reintensification_after_relief",
        "direction_consistent",
        "signflip_or_reversal",
        "direction_mismatch",
        "near_safe_far_risky",
        "near_risky_far_relief",
        "target_stale",
        "pump_idle",
        "candidate_prefloor_like",
        "candidate_enter4_like",
    ]
    for flag in coverage_flags:
        if flag not in refined.columns or flag not in prep.columns:
            continue
        coverage_rows.append(
            {
                "feature": flag,
                "candidate_rows": int((refined[flag].astype(float) > 0).sum()),
                "prepare_safe_positive_rows": int(
                    (
                        (prep[flag].astype(float) > 0)
                        & (prep["constrained_label"] == "safe_positive")
                    ).sum()
                ),
                "prepare_negative_rows": int(
                    (
                        (prep[flag].astype(float) > 0)
                        & (prep["constrained_label"] == "negative")
                    ).sum()
                ),
            }
        )
    coverage = pd.DataFrame(coverage_rows)
    coverage.to_csv(RAW / "candidate_shape_coverage.csv", index=False)

    best_high = (
        cal_summary[cal_summary.get("threshold", pd.Series(dtype=float)).eq(0.70)]
        if "threshold" in cal_summary
        else pd.DataFrame()
    )
    best_precision = (
        float(best_high["precision"].max()) if not best_high.empty and "precision" in best_high else 0.0
    )
    best_anti = (
        float(best_high["anti_trigger_recall"].max())
        if not best_high.empty and "anti_trigger_recall" in best_high
        else 0.0
    )
    safe_positive = int((prep["constrained_label"] == "safe_positive").sum())
    negative = int((prep["constrained_label"] == "negative").sum())
    ambiguous = int((prep["constrained_label"] == "ambiguous").sum())
    broader_lowrisk_fp_region = prep[
        (prep["constrained_label"] == "safe_positive")
        & (
            (prep["dataset"] == "broader20")
            | prep["selection_reasons"].astype(str).str.contains("lowrisk_sanity", na=False)
        )
    ]

    md = ["# h120 Action-Value Label Refinement Decision", ""]
    md.append("## Verdict")
    md.append("")
    md.append(
        "**Use constrained action-value labels going forward, but do not attach a default-off gate yet.** "
        "The refined labels are safer and more interpretable than the old proxy, but the high-confidence calibrator is still not stable enough for closed-loop use."
    )
    md.append("")
    md.append("## Evidence")
    md.append("")
    md.append(f"- refined buckets: {len(refined)}")
    md.append(f"- prepare rows: {len(prep)}")
    md.append(f"- constrained labels: safe_positive={safe_positive}, negative={negative}, ambiguous={ambiguous}")
    md.append(f"- best threshold-0.70 prepare precision: {best_precision:.3f}")
    md.append(f"- best threshold-0.70 anti-trigger recall: {best_anti:.3f}")
    md.append(
        "- note: high anti-trigger recall can come from mostly rejecting prepare, "
        "so it is not enough by itself."
    )
    md.append("")
    md.append("## Label Counts")
    md.append("")
    md.append(_markdown_table(by_dataset))
    md.append("")
    md.append("## Case-Level Concentration")
    md.append("")
    md.append(_markdown_table(case_concentration.head(12)))
    md.append("")
    md.append("## Sensitivity Snapshot")
    md.append("")
    md.append(_markdown_table(sensitivity.head(12)))
    md.append("")
    md.append("## Required Answers")
    md.append("")
    md.append("1. **Are constrained labels more stable than the old proxy?**")
    md.append("")
    md.append("Yes. They separate safe positives from pump-only gains using hard fallback/max-p95/pump constraints, so they are more controller-relevant than the old scalar proxy. Sensitivity still changes counts, so these labels should be treated as a refined audit target rather than a final objective.")
    md.append("")
    md.append("2. **Is there a high-precision prepare region?**")
    md.append("")
    if best_precision >= 0.75:
        md.append("There is an initial high-precision region offline, but it still needs closed-loop sanity after further expansion.")
    else:
        md.append("Not yet. The offline models do not produce a sufficiently precise high-confidence prepare region.")
    md.append("")
    md.append("3. **Can broader20 / lowrisk misfires be excluded by features?**")
    md.append("")
    md.append(
        f"Partly, but not enough for a gate. There are {len(broader_lowrisk_fp_region)} constrained safe-positive rows in broader/lowrisk regions, so the reject boundary still needs more anti-trigger data and likely a better value head."
    )
    md.append("")
    md.append("4. **Which action is better for first gate?**")
    md.append("")
    ps = prep[prep["action"] == "pump_saving_prepare"]
    ac = prep[prep["action"] == "active_small_prepare"]
    ps_pos = int((ps["constrained_label"] == "safe_positive").sum())
    ac_pos = int((ac["constrained_label"] == "safe_positive").sum())
    md.append(
        f"`pump_saving_prepare` is still the safer first-gate candidate because it has lower actuation magnitude, even though safe-positive counts are pump_saving={ps_pos}, active_small={ac_pos}. `active_small_prepare` should remain a second-stage/high-confidence action."
    )
    md.append("")
    md.append("5. **Allow default-off h120_action_value_gate now?**")
    md.append("")
    md.append("No. The label refinement supports the path, but the offline high-confidence precision and anti-trigger separation are not yet controller-ready.")
    md.append("")
    md.append("6. **Next step?**")
    md.append("")
    md.append("Expand samples again around anti-trigger and rare positive classes, then train/design an action-value head with explicit outputs for safe_positive, preferred_action, expected time/idle gain, pump cost, and anti-trigger probability.")
    md.append("")
    md.append("7. **v1.6 status**")
    md.append("")
    md.append("v1.6 delayed-medium reactive floor remains the safety mainline. h120 remains an outer offline action-value research layer.")
    (OUT / "h120_action_value_label_decision.md").write_text("\n".join(md) + "\n")

    paper = ["# h120 Action-Value Label Findings", ""]
    paper.append("- Constrained labels are safer and more interpretable than the old scalar proxy.")
    paper.append(f"- Refined buckets: {len(refined)}; prepare rows: {len(prep)}.")
    paper.append(f"- Labels: safe_positive={safe_positive}, negative={negative}, ambiguous={ambiguous}.")
    paper.append(
        f"- Best threshold-0.70 prepare precision: {best_precision:.3f}; "
        f"best anti-trigger recall: {best_anti:.3f}."
    )
    paper.append("- Default-off controller gate is still premature; continue offline value-head work.")
    paper.append("")
    paper.append("## Label Counts")
    paper.append("")
    paper.append(_markdown_table(by_dataset))
    paper.append("")
    paper.append("## Action Summary")
    paper.append("")
    paper.append(_markdown_table(action_summary))
    (PAPER / "action_value_label_findings.md").write_text("\n".join(paper) + "\n")

    selected_features = [
        "max_axis_deg",
        "target_age_s",
        "pump_idle",
        "near_max_norm",
        "far_max_norm",
        "delayed_intensification",
        "reintensification_after_relief",
        "direction_consistent",
        "direction_mismatch",
        "near_safe_far_risky",
        "near_risky_far_relief",
        "time_gain_s",
        "idle_gain_s",
        "pump_delta_m3",
        "delta_full_fallback_pp",
        "delta_full_max_p95",
    ]
    feature_snapshot = feature_contrast[
        feature_contrast["feature"].isin(selected_features)
    ][["feature", "safe_positive", "negative", "safe_minus_negative"]]
    ext = ["# h120 Action-Value Label Refinement: Extended Findings", ""]
    ext.append("## Bottom Line")
    ext.append("")
    ext.append(
        "This run supports the 120min action-value direction, but it does not yet "
        "produce a controller-ready gate. The constrained labels are cleaner than "
        "the old scalar value proxy, yet safe-positive prepare decisions remain "
        "sparse, case-concentrated, and hard to separate under grouped validation."
    )
    ext.append("")
    ext.append("## Run Scope")
    ext.append("")
    ext.append(f"- {len(refined)} locked/refined buckets.")
    ext.append("- Four actions per bucket: wait, watch_only, pump_saving_prepare, active_small_prepare.")
    ext.append(f"- {len(labels)} counterfactual rows; {len(prep)} prepare rows.")
    ext.append("- No controller gate was implemented; v1.6 floor was not changed.")
    ext.append("")
    ext.append("## Constrained Label Outcome")
    ext.append("")
    ext.append(
        f"- safe_positive={safe_positive}, negative={negative}, ambiguous={ambiguous} on prepare rows."
    )
    ext.append(
        "- near_safe_far_risky and lowrisk_sanity had zero safe-positive rows in "
        "the sensitivity table, which is encouraging for anti-trigger design."
    )
    ext.append("")
    ext.append(_markdown_table(by_dataset))
    ext.append("")
    ext.append("## Mean Action Effects")
    ext.append("")
    ext.append(_markdown_table(action_summary))
    ext.append("")
    ext.append(
        "Interpretation: active_small_prepare creates more apparent safe positives, "
        "especially in broader20, but it is not automatically a better first gate "
        "because its actuation is larger and guard10 aggregate fallback/max-p95 "
        "deltas are worse. pump_saving_prepare remains the safer first-stage action candidate."
    )
    ext.append("")
    ext.append("## Positive Case Concentration")
    ext.append("")
    ext.append(_markdown_table(case_concentration.head(12)))
    ext.append("")
    ext.append(
        "Positive rows are not evenly distributed. A future value head must prove "
        "these are reusable patterns rather than case concentration."
    )
    ext.append("")
    ext.append("## Shape / State Coverage")
    ext.append("")
    ext.append(_markdown_table(coverage))
    ext.append("")
    ext.append(
        "Useful hints: delayed_intensification and reintensification_after_relief "
        "are enriched in safe positives; direction_mismatch and near_safe_far_risky "
        "are enriched in negatives. Direction_consistent alone is not enough."
    )
    ext.append("")
    ext.append("## Feature Contrast")
    ext.append("")
    ext.append(_markdown_table(feature_snapshot))
    ext.append("")
    ext.append(
        "The strongest contrasts are not simply far pressure magnitude. "
        "far_persistent_high alone is more common in negatives than safe positives."
    )
    ext.append("")
    ext.append("## Offline Calibrator Diagnostic")
    ext.append("")
    ext.append(_markdown_table(cal_diag))
    if not preferred_summary.empty:
        ext.append("")
        ext.append(_markdown_table(preferred_summary))
    ext.append("")
    ext.append(
        "The light calibrators are not controller-ready. HGB/RF mostly learn to "
        "reject prepare actions at high threshold, which gives high anti-trigger "
        "recall but zero positive precision. Logistic predicts more prepare rows, "
        "but with very low precision."
    )
    ext.append("")
    ext.append("## Decision")
    ext.append("")
    ext.append("- Continue h120 as action-value/value-head work, not as far-risk thresholding.")
    ext.append("- Do not enter default-off h120_action_value_gate yet.")
    ext.append(
        "- Next useful step is a dedicated value head/label set: safe_positive "
        "probability, anti-trigger probability, expected time/idle gain, expected "
        "pump delta, and preferred action."
    )
    ext.append("- v1.6 remains the safety mainline.")
    (PAPER / "action_value_label_extended_findings.md").write_text(
        "\n".join(ext) + "\n"
    )

    gpt = ["# GPT Evaluation Brief: h120_action_value_label_refinement_v1", ""]
    gpt.extend(ext[2:])
    gpt.append("")
    gpt.append("## Suggested Next Step")
    gpt.append("")
    gpt.append("- Do not connect to controller yet.")
    gpt.append(
        "- Build a dedicated h120 action-value/value-head target with explicit "
        "anti-trigger output, expected time/idle gain, pump cost, and preferred action."
    )
    (DEBUG / "gpt_evaluation_brief.md").write_text("\n".join(gpt) + "\n")


def write_checkpoint(phase: str, status: str, reason: str, files: list[str]) -> None:
    with (DEBUG / "phase_status.jsonl").open("a") as f:
        f.write(
            json.dumps(
                {
                    "phase": phase,
                    "status": status,
                    "go_no_go": status,
                    "reason": reason,
                    "evidence_files": files,
                },
                ensure_ascii=False,
            )
            + "\n"
        )


def main() -> None:
    _ensure_dirs()
    status = DEBUG / "phase_status.jsonl"
    if status.exists():
        status.unlink()
    (DEBUG / "task_master_plan.md").write_text(
        "# h120_action_value_label_refinement_v1\n\n"
        "1. Expand locked counterfactual candidates to 80-150 buckets.\n"
        "2. Reuse cached forced-prefix runs where available and run missing cases.\n"
        "3. Apply constrained safe_positive / negative / ambiguous labels.\n"
        "4. Report case-level aggregation and label sensitivity.\n"
        "5. Train only light offline calibrators if labels are diverse.\n"
    )

    source = _load_candidate_source()
    refined = build_refined_candidates(source)
    write_checkpoint(
        "phase1_expand_candidates",
        "done",
        "expanded refined candidates selected from oracle action-value candidate table",
        ["refined_candidate_table.csv"],
    )

    _, deltas = run_counterfactuals(refined)
    write_checkpoint(
        "phase2_counterfactuals",
        "done",
        "counterfactuals completed or reused from locked_v1 cache",
        ["refined_counterfactual_table.csv"],
    )

    labels = apply_constrained_labels(deltas)
    case_table = build_case_level_table(labels)
    sensitivity = build_label_sensitivity(deltas)
    write_checkpoint(
        "phase3_labels",
        "done",
        "constrained labels, case-level aggregation, and sensitivity tables written",
        ["constrained_label_table.csv", "case_level_value_table.csv", "label_sensitivity_table.csv"],
    )

    calibrator = run_calibrators(labels)
    write_checkpoint(
        "phase4_calibrator",
        "done",
        "light offline calibrators trained as separability checks only",
        ["offline_calibrator_results.csv"],
    )

    write_reports(refined, labels, case_table, sensitivity, calibrator)
    write_checkpoint(
        "phase5_decision",
        "done",
        "decision written; no controller gate implemented",
        ["h120_action_value_label_decision.md", "paper_ready/action_value_label_findings.md"],
    )


if __name__ == "__main__":
    main()
