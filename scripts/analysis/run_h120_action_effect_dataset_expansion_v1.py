#!/usr/bin/env python3
"""h120 action-effect dataset expansion v1.

Offline expansion of action-effect counterfactuals for the h120 action-value
route.  This script does not attach a controller gate, does not modify v1.6
floor logic, and only uses forced-prefix replay for a locked set of candidate
buckets.
"""

from __future__ import annotations

import csv
import importlib.util
import math
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REPO = Path(__file__).resolve().parents[2]

PREPARE_SCRIPT = REPO / "scripts/analysis/run_h120_prepare_action_family_probe_v1.py"
spec = importlib.util.spec_from_file_location("h120_prepare_action_family_probe_v1", PREPARE_SCRIPT)
if spec is None or spec.loader is None:
    raise RuntimeError(f"cannot import helpers from {PREPARE_SCRIPT}")
prepare = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = prepare
spec.loader.exec_module(prepare)

OUT = REPO / "outputs/wind_prediction/h120_action_effect_dataset_expansion_v1"
RAW = OUT / "raw_tables"
DEBUG = OUT / "debug"
PAPER = OUT / "paper_ready"
RUNS = OUT / "counterfactual_runs"

PREPARE_V1 = REPO / "outputs/wind_prediction/h120_prepare_action_family_probe_v1"
REGIME_V1 = REPO / "outputs/wind_prediction/h120_action_effect_regime_v1"
OVERNIGHT = REPO / "outputs/wind_prediction/h120_action_value_overnight_v1"
SUPPLEMENTAL = OVERNIGHT / "supplemental_sweep_v1"
BASELINE_ROOT = REPO / "outputs/wind_prediction/h120_learned_risk_scheduler_probe_v1"

TARGET_BUCKETS = 140
WAIT_ACTION = "wait"
MICRO50_ACTION = "micro_prepare_50"
PUMP_ACTION = "pump_saving_prepare"
ACTIVE_ACTION = "active_small_prepare"
CORE_ACTIONS = [
    {
        "action": WAIT_ACTION,
        "base_action": "none",
        "active_small_scale": 0.0,
        "run_forced": False,
        "description": "v1.6 oracle baseline; no prepare.",
    },
    {
        "action": MICRO50_ACTION,
        "base_action": "active_small",
        "active_small_scale": 0.50,
        "run_forced": True,
        "description": "Small prepare candidate, 50% of active_small target refresh.",
    },
    {
        "action": PUMP_ACTION,
        "base_action": "pump_saving",
        "active_small_scale": 0.08 / 0.15,
        "run_forced": True,
        "description": "Existing pump_saving prepare reference.",
    },
    {
        "action": ACTIVE_ACTION,
        "base_action": "active_small",
        "active_small_scale": 1.0,
        "run_forced": True,
        "description": "Upper reference only; not a first gate candidate.",
    },
]
ACTION_DEFS = {d["action"]: d for d in CORE_ACTIONS}
FORCED_ACTIONS = [MICRO50_ACTION, PUMP_ACTION, ACTIVE_ACTION]

METRIC_KEYS = list(prepare.METRIC_KEYS)
FEATURE_COLUMNS = [
    "current_pitch_deg",
    "current_roll_deg",
    *[c for c in prepare.FEATURE_COLUMNS if c not in ("current_pitch_deg", "current_roll_deg")],
]


def ensure_dirs() -> None:
    for path in (OUT, RAW, DEBUG, PAPER, RUNS):
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


def as_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def row_key(row: pd.Series) -> tuple[str, str, int]:
    return str(row["dataset"]), str(row["case_id"]), int(row["bucket"])


def token_set(value: Any) -> set[str]:
    return {tok.strip() for tok in str(value or "").split(";") if tok.strip()}


def case_aliases(row: pd.Series) -> set[str]:
    aliases = {str(row.get("case_id", "")), str(row.get("numbered_case_id", ""))}
    for alias in list(aliases):
        if len(alias) > 3 and alias[:2].isdigit() and alias[2] == "_":
            aliases.add(alias[3:])
    return {a for a in aliases if a and a != "nan"}


def load_candidate_pool() -> pd.DataFrame:
    pool = prepare.load_candidate_pool().copy()
    pool["dataset"] = pool["dataset"].astype(str)
    pool["case_id"] = pool["case_id"].astype(str)
    pool["bucket"] = pd.to_numeric(pool["bucket"], errors="coerce").fillna(-1).astype(int)
    pool["bucket_key"] = pool["dataset"] + "::" + pool["case_id"] + "::" + pool["bucket"].astype(str)
    if "action_family_candidate_reasons" not in pool:
        pool["action_family_candidate_reasons"] = [
            ";".join(prepare._candidate_reason(row)) for _, row in pool.iterrows()
        ]
    else:
        fill = [";".join(prepare._candidate_reason(row)) for _, row in pool.iterrows()]
        pool["action_family_candidate_reasons"] = pool["action_family_candidate_reasons"].fillna(pd.Series(fill))
    pool["_base_selection_score"] = [prepare._selection_score(row) for _, row in pool.iterrows()]

    prior_path = REGIME_V1 / "action_effect_regime_table.csv"
    if prior_path.exists():
        prior = pd.read_csv(prior_path)
        prior["bucket_key"] = prior["dataset"].astype(str) + "::" + prior["case_id"].astype(str) + "::" + pd.to_numeric(prior["bucket"], errors="coerce").fillna(-1).astype(int).astype(str)
        prior_cols = [
            "bucket_key",
            "no_effect",
            "effectful",
            "positive_effect",
            "negative_effect",
            "non_monotone",
            "regression_risk",
            "axis_tradeoff",
            "micro50_positive_effect",
            "micro50_negative_effect",
            "pump_saving_positive_effect",
            "pump_saving_negative_effect",
        ]
        pool = pool.merge(prior[[c for c in prior_cols if c in prior]], on="bucket_key", how="left", suffixes=("", "_prior"))
    for col in [
        "no_effect",
        "effectful",
        "positive_effect",
        "negative_effect",
        "non_monotone",
        "regression_risk",
        "axis_tradeoff",
        "micro50_positive_effect",
        "micro50_negative_effect",
        "pump_saving_positive_effect",
        "pump_saving_negative_effect",
    ]:
        if col not in pool:
            pool[col] = 0
        pool[col] = pd.to_numeric(pool[col], errors="coerce").fillna(0).astype(int)
    return pool


def candidate_reason(row: pd.Series) -> list[str]:
    reasons = token_set(row.get("action_family_candidate_reasons"))
    reasons |= token_set(row.get("selection_reasons"))
    dataset = str(row.get("dataset"))
    future_time = as_float(row.get("future_60m_time_over5"))
    future_idle = as_float(row.get("future_60m_idle_over5"))
    if dataset == "broader20" and as_int(row.get("known_prepare_negative_rows")) > 0:
        reasons.add("broader20_false_prepare")
    if dataset == "broader20" and future_time <= 5.0 and future_idle <= 20.0:
        reasons.add("broader20_low_future_exposure")
    if as_float(row.get("max_axis_deg")) < 3.0 and future_time <= 5.0 and future_idle <= 20.0:
        reasons.add("lowrisk_sanity")
    if as_int(row.get("direction_mismatch")) > 0:
        reasons.add("direction_mismatch")
    if as_int(row.get("near_safe_far_risky")) > 0:
        reasons.add("near_safe_far_risky")
    if as_int(row.get("far_persistent_high")) > 0 and not (
        as_int(row.get("delayed_intensification")) or as_int(row.get("reintensification_after_relief"))
    ):
        reasons.add("far_persistent_high_alone")
    if as_int(row.get("signflip_or_reversal")) > 0:
        reasons.add("signflip_or_reversal")
    if as_int(row.get("axis_tradeoff")) > 0:
        reasons.add("prior_axis_tradeoff")
    if as_int(row.get("regression_risk")) > 0:
        reasons.add("prior_regression_risk")
    if as_int(row.get("micro50_positive_effect")) > 0:
        reasons.add("prior_micro50_positive")
    if (
        as_float(row.get("posture_trend_10m_deg")) > 0.25
        and (as_int(row.get("pump_idle")) > 0 or as_float(row.get("target_error_mean_kg")) < 500.0)
        and as_float(row.get("near_max_norm")) >= 0.8
    ):
        reasons.add("trend_worsening_target_quiet_near_high")
    return sorted(reasons)


def selection_score(row: pd.Series) -> float:
    reasons = set(row.get("_candidate_reason_list", []))
    score = as_float(row.get("_base_selection_score"))
    weights = {
        "broader20_false_prepare": 16.0,
        "lowrisk_sanity": 14.0,
        "direction_mismatch": 11.0,
        "near_safe_far_risky": 10.0,
        "far_persistent_high_alone": 8.0,
        "signflip_or_reversal": 8.0,
        "prior_axis_tradeoff": 16.0,
        "prior_regression_risk": 16.0,
        "prior_micro50_positive": 18.0,
        "trend_worsening_target_quiet_near_high": 12.0,
        "guard10_future_exposure": 8.0,
        "delayed_intensification": 5.0,
        "reintensification_after_relief": 5.0,
    }
    for reason, weight in weights.items():
        if reason in reasons:
            score += weight
    if str(row.get("dataset")) == "guard10":
        score += min(as_float(row.get("future_60m_time_over5")) / 90.0, 8.0)
    return score


def pick(pool: pd.DataFrame, mask: pd.Series, quota: int, used: set[tuple[str, str, int]], category: str) -> list[pd.Series]:
    rows: list[pd.Series] = []
    sub = pool.loc[mask].copy()
    if quota <= 0 or sub.empty:
        return rows
    for _, row in sub.sort_values("_expansion_score", ascending=False).iterrows():
        key = row_key(row)
        if key in used:
            continue
        out = row.copy()
        out["expansion_selection_category"] = category
        rows.append(out)
        used.add(key)
        if len(rows) >= quota:
            break
    return rows


def select_candidates(pool: pd.DataFrame, target: int = TARGET_BUCKETS) -> pd.DataFrame:
    work = pool.copy()
    work["_candidate_reason_list"] = [candidate_reason(row) for _, row in work.iterrows()]
    work["candidate_reason"] = [";".join(row) for row in work["_candidate_reason_list"]]
    work["_expansion_score"] = [selection_score(row) for _, row in work.iterrows()]
    used: set[tuple[str, str, int]] = set()
    rows: list[pd.Series] = []
    specs = [
        ("prior_micro50_positive", 18, work["candidate_reason"].str.contains("prior_micro50_positive", na=False)),
        ("prior_regression_risk", 24, work["candidate_reason"].str.contains("prior_regression_risk", na=False)),
        ("prior_axis_tradeoff", 18, work["candidate_reason"].str.contains("prior_axis_tradeoff", na=False)),
        ("broader20_false_prepare", 28, work["candidate_reason"].str.contains("broader20_false_prepare|broader20_anti_trigger", regex=True, na=False)),
        ("lowrisk_sanity", 22, work["candidate_reason"].str.contains("lowrisk_sanity|future_low_exposure_remote", regex=True, na=False)),
        ("direction_mismatch", 18, work["candidate_reason"].str.contains("direction_mismatch", na=False)),
        ("near_safe_far_risky", 10, work["candidate_reason"].str.contains("near_safe_far_risky", na=False)),
        ("far_persistent_high_alone", 16, work["candidate_reason"].str.contains("far_persistent_high_alone", na=False)),
        ("signflip_or_reversal", 10, work["candidate_reason"].str.contains("signflip_or_reversal", na=False)),
        ("trend_target_quiet_near_high", 18, work["candidate_reason"].str.contains("trend_worsening_target_quiet_near_high", na=False)),
        (
            "guard10_positive_window",
            18,
            (work["dataset"] == "guard10")
            & ((work["future_60m_time_over5"] > 0) | (work["future_60m_idle_over5"] > 0)),
        ),
    ]
    for category, quota, mask in specs:
        rows.extend(pick(work, mask, quota, used, category))
    if len(rows) < target:
        rows.extend(pick(work, pd.Series(True, index=work.index), target - len(rows), used, "score_fill"))

    selected = pd.DataFrame(rows).drop_duplicates(["dataset", "case_id", "bucket"]).head(target).copy()
    selected["probe_id"] = [
        f"{r.dataset}_{r.case_id}_b{int(r.bucket):02d}" for r in selected.itertuples()
    ]
    cols = [
        "probe_id",
        "dataset",
        "case_id",
        "numbered_case_id",
        "timestamp",
        "label",
        "bucket",
        "current_time_s",
        "expansion_selection_category",
        "candidate_reason",
        "selection_category",
        "selection_reasons",
        "known_pump_saving_safe_positive",
        "known_active_small_safe_positive",
        "known_prepare_negative_rows",
        "known_prepare_safe_positive_rows",
        "current_pitch_deg",
        "current_roll_deg",
        *[c for c in FEATURE_COLUMNS if c not in ("current_pitch_deg", "current_roll_deg")],
        "future_60m_time_over5",
        "future_60m_idle_over5",
        "future_60m_floor_entry",
        "future_60m_medium_delay_rows",
        "future_60m_pump_m3",
        "future_60m_max_axis_max",
        "no_effect",
        "effectful",
        "positive_effect",
        "negative_effect",
        "non_monotone",
        "regression_risk",
        "axis_tradeoff",
        "micro50_positive_effect",
        "micro50_negative_effect",
        "pump_saving_positive_effect",
        "pump_saving_negative_effect",
        "_expansion_score",
    ]
    selected = selected[[c for c in cols if c in selected.columns]]
    selected.to_csv(OUT / "expanded_candidate_table.csv", index=False)
    selected.to_csv(RAW / "expanded_candidate_table.csv", index=False)
    report = (
        selected.groupby(["dataset", "expansion_selection_category"]).size().reset_index(name="buckets")
    )
    report.to_csv(DEBUG / "candidate_sampling_report.csv", index=False)
    return selected


def write_one_case_csv(path: Path, row: pd.Series) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["case_id", "timestamp", "label"])
        writer.writeheader()
        writer.writerow(
            {
                "case_id": row["case_id"],
                "timestamp": row["timestamp"],
                "label": row.get("label", row["case_id"]),
            }
        )


def scaled_vec(row: pd.Series, action_def: dict[str, Any]) -> tuple[float, float, float, str]:
    base_action = str(action_def["base_action"])
    if base_action == "pump_saving":
        pitch, roll, norm = prepare.prev._forced_vec(row, "pump_saving")
        return pitch, roll, norm, "pump_saving"
    pitch, roll, _ = prepare.prev._forced_vec(row, "active_small")
    scale = float(action_def["active_small_scale"])
    pitch *= scale
    roll *= scale
    return pitch, roll, float(math.hypot(pitch, roll)), "active_small"


def write_forced_csv(path: Path, row: pd.Series, action_def: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pitch, roll, norm, provider_action = scaled_vec(row, action_def)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "case_id",
                "bucket",
                "action",
                "forced_pitch_deg",
                "forced_roll_deg",
                "source",
                "forced_vec_norm",
                "action_family_name",
                "target_refresh_scale",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "case_id": row["case_id"],
                "bucket": int(row["bucket"]),
                "action": provider_action,
                "forced_pitch_deg": pitch,
                "forced_roll_deg": roll,
                "source": "h120_action_effect_dataset_expansion_v1",
                "forced_vec_norm": norm,
                "action_family_name": action_def["action"],
                "target_refresh_scale": action_def["active_small_scale"],
            }
        )


def run_casebook(cases_csv: Path, forced_csv: Path, out_dir: Path) -> None:
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
        str(prepare.F120_DATASET),
        "--far-horizon",
        "--reactive-floor-predictive-veto",
        "on",
        "--reactive-floor-action",
        "active_small",
        "--reactive-floor-medium-delay-s",
        "1200",
        "--reactive-floor-post-exit-mode",
        "early_stop",
        "--forced-prefix-actions",
        str(forced_csv),
        "--forced-prefix-mode",
        "target_update",
        "--out-dir",
        str(out_dir),
    ]
    subprocess.run(cmd, cwd=REPO, check=True)


def cached_run_dir(probe_id: str, action: str) -> Path | None:
    candidates = [
        RUNS / probe_id / action,
        PREPARE_V1 / "counterfactual_runs" / probe_id / action,
        OVERNIGHT / "counterfactual_runs" / probe_id / action,
        SUPPLEMENTAL / "counterfactual_runs" / probe_id / action,
    ]
    for path in candidates:
        if (path / "casebook_summary.csv").exists():
            return path
    return None


def baseline_metrics(row: pd.Series) -> dict[str, float]:
    return prepare._baseline_metrics(row)


def case_metrics(run_dir: Path, row: pd.Series) -> dict[str, float]:
    return prepare._case_metrics(run_dir, row)


def metrics_delta(metrics: dict[str, float], baseline: dict[str, float]) -> dict[str, float]:
    return {f"delta_{key}": float(metrics[key] - baseline[key]) for key in METRIC_KEYS}


def read_axis_from_summary(run_dir: Path) -> dict[str, float]:
    path = run_dir / "casebook_summary.csv"
    if not path.exists():
        return {}
    try:
        row = pd.read_csv(path).iloc[0]
    except Exception:
        return {}
    return {
        "primary_pitch_p95": as_float(row.get("primary_pitch_p95"), np.nan),
        "primary_roll_p95": as_float(row.get("primary_roll_p95"), np.nan),
    }


def baseline_axis_lookup(selected: pd.DataFrame) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for dataset, run_name in {
        "guard10": "guard10_v16_oracle_baseline",
        "broader20": "broader20_v16_oracle_baseline",
    }.items():
        summary_path = BASELINE_ROOT / run_name / "casebook_summary.csv"
        if not summary_path.exists():
            continue
        summary = pd.read_csv(summary_path)
        for _, row in selected[selected["dataset"].eq(dataset)].iterrows():
            aliases = case_aliases(row)
            match = summary.loc[summary["case_id"].astype(str).isin(aliases)]
            if match.empty:
                continue
            sr = match.iloc[0]
            key = f"{row['dataset']}::{row['case_id']}::{int(row['bucket'])}"
            out[key] = {
                "baseline_pitch_p95": as_float(sr.get("primary_pitch_p95"), np.nan),
                "baseline_roll_p95": as_float(sr.get("primary_roll_p95"), np.nan),
            }
    return out


def label_action(row: pd.Series) -> tuple[str, str]:
    action = str(row["action"])
    if action == WAIT_ACTION:
        return "baseline", "baseline action"
    pump_delta = as_float(row.get("delta_full_pump_m3"))
    time_delta = as_float(row.get("delta_post60_time_over5"))
    idle_delta = as_float(row.get("delta_post60_idle_over5"))
    fallback_delta = as_float(row.get("delta_full_fallback_pp"))
    p95_delta = as_float(row.get("delta_full_max_p95"))
    floor_delta = as_float(row.get("delta_post60_floor_entry"))
    medium_delta = as_float(row.get("delta_post60_medium_delay_rows"))
    time_gain = -time_delta
    idle_gain = -idle_delta
    safety_gain = time_gain >= 30.0 or idle_gain >= 60.0 or floor_delta < 0.0 or medium_delta < 0.0
    hard_ok = fallback_delta <= 0.05 and p95_delta <= 0.05 and pump_delta <= 80.0 and time_delta <= 5.0
    safety_worse = time_delta > 10.0 or idle_delta > 20.0 or fallback_delta > 0.05 or p95_delta > 0.05
    no_gain = time_gain < 10.0 and idle_gain < 20.0 and floor_delta >= 0.0 and medium_delta >= 0.0
    anti_trigger = (
        (as_int(row.get("near_safe_far_risky")) > 0 or as_int(row.get("direction_mismatch")) > 0)
        and not safety_gain
    )
    if safety_gain and hard_ok:
        return "safe_positive", "safety gain within pump/fallback/max constraints"
    if safety_worse:
        return "negative", "safety regression hard constraint"
    if pump_delta > 5.0 and no_gain:
        return "negative", "pump increase without meaningful safety gain"
    if anti_trigger:
        return "negative", "anti-trigger feature without meaningful safety gain"
    return "ambiguous", "small or unclear pump/safety tradeoff"


def run_counterfactuals(selected: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    axis_base = baseline_axis_lookup(selected)
    for idx, row in selected.iterrows():
        probe_id = str(row["probe_id"])
        case_csv = DEBUG / f"{probe_id}_case.csv"
        write_one_case_csv(case_csv, row)
        baseline = baseline_metrics(row)
        bucket_key = f"{row['dataset']}::{row['case_id']}::{int(row['bucket'])}"
        base_axis = axis_base.get(bucket_key, {})
        if idx % 10 == 0:
            print(f"[expansion] {idx + 1}/{len(selected)} {probe_id}", flush=True)
        for action_name in [WAIT_ACTION, MICRO50_ACTION, PUMP_ACTION, ACTIVE_ACTION]:
            action_def = ACTION_DEFS[action_name]
            cache_source = "baseline"
            run_dir = ""
            metrics = dict(baseline)
            axis = {
                "primary_pitch_p95": base_axis.get("baseline_pitch_p95", np.nan),
                "primary_roll_p95": base_axis.get("baseline_roll_p95", np.nan),
            }
            if bool(action_def["run_forced"]):
                cached = cached_run_dir(probe_id, action_name)
                if cached is None:
                    forced_csv = DEBUG / f"{probe_id}_{action_name}_forced.csv"
                    run_path = RUNS / probe_id / action_name
                    write_forced_csv(forced_csv, row, action_def)
                    run_casebook(case_csv, forced_csv, run_path)
                    cached = run_path
                    cache_source = "new_expansion_run"
                else:
                    cache_source = "cached"
                metrics = case_metrics(cached, row)
                run_dir = str(cached.relative_to(REPO))
                axis = read_axis_from_summary(cached)
            delta = metrics_delta(metrics, baseline)
            payload: dict[str, Any] = {
                "probe_id": probe_id,
                "bucket_key": bucket_key,
                "dataset": row["dataset"],
                "case_id": row["case_id"],
                "group_id": f"{row['dataset']}::{row['case_id']}",
                "bucket": int(row["bucket"]),
                "timestamp": row["timestamp"],
                "current_time_s": as_float(row.get("current_time_s")),
                "action": action_name,
                "base_action": action_def["base_action"],
                "target_refresh_scale": float(action_def["active_small_scale"]),
                "run_dir": run_dir,
                "cache_source": cache_source,
                "expansion_selection_category": row.get("expansion_selection_category", ""),
                "candidate_reason": row.get("candidate_reason", ""),
                "baseline_pitch_p95": base_axis.get("baseline_pitch_p95", np.nan),
                "baseline_roll_p95": base_axis.get("baseline_roll_p95", np.nan),
                **axis,
            }
            for col in FEATURE_COLUMNS:
                if col in row.index:
                    payload[col] = row.get(col, 0.0)
            payload.update(metrics)
            payload.update(delta)
            rows.append(payload)
    table = pd.DataFrame(rows)
    labels = [label_action(row) for _, row in table.iterrows()]
    table["action_label"] = [x[0] for x in labels]
    table["action_label_reason"] = [x[1] for x in labels]
    table["time_gain_s"] = -num_series(table, "delta_post60_time_over5")
    table["idle_gain_s"] = -num_series(table, "delta_post60_idle_over5")
    table["pump_delta_m3"] = num_series(table, "delta_full_pump_m3")
    table["delta_pitch_p95"] = num_series(table, "primary_pitch_p95") - num_series(table, "baseline_pitch_p95")
    table["delta_roll_p95"] = num_series(table, "primary_roll_p95") - num_series(table, "baseline_roll_p95")
    table["action_utility"] = (
        table["time_gain_s"]
        + 0.35 * table["idle_gain_s"]
        - 0.12 * table["pump_delta_m3"]
        - 180.0 * np.maximum(num_series(table, "delta_full_fallback_pp"), 0.0)
        - 120.0 * np.maximum(num_series(table, "delta_full_max_p95"), 0.0)
    )
    table.to_csv(OUT / "expanded_counterfactual_table.csv", index=False)
    table.to_csv(RAW / "expanded_counterfactual_table.csv", index=False)
    return table


def is_action_positive(df: pd.DataFrame) -> pd.Series:
    safety = (
        (num_series(df, "time_gain_s") >= 30.0)
        | (num_series(df, "idle_gain_s") >= 60.0)
        | (num_series(df, "delta_post60_floor_entry") < 0)
        | (num_series(df, "delta_post60_medium_delay_rows") < 0)
    )
    hard_ok = (
        (num_series(df, "delta_full_fallback_pp") <= 0.05)
        & (num_series(df, "delta_full_max_p95") <= 0.05)
        & (num_series(df, "pump_delta_m3") <= 80.0)
        & (num_series(df, "delta_post60_time_over5") <= 5.0)
    )
    return safety & hard_ok


def is_action_regression(df: pd.DataFrame) -> pd.Series:
    return (
        (num_series(df, "delta_post60_time_over5") > 10.0)
        | (num_series(df, "delta_post60_idle_over5") > 20.0)
        | (num_series(df, "delta_full_fallback_pp") > 0.05)
        | (num_series(df, "delta_full_max_p95") > 0.05)
    )


def is_action_negative(df: pd.DataFrame) -> pd.Series:
    no_gain = (
        (num_series(df, "time_gain_s") < 30.0)
        & (num_series(df, "idle_gain_s") < 60.0)
        & (num_series(df, "delta_post60_floor_entry") >= 0)
        & (num_series(df, "delta_post60_medium_delay_rows") >= 0)
    )
    return is_action_regression(df) | ((num_series(df, "pump_delta_m3") > 5.0) & no_gain)


def is_axis_tradeoff(df: pd.DataFrame) -> pd.Series:
    return (
        ((num_series(df, "delta_pitch_p95") <= -0.05) & (num_series(df, "delta_roll_p95") >= 0.05))
        | ((num_series(df, "delta_roll_p95") <= -0.05) & (num_series(df, "delta_pitch_p95") >= 0.05))
    )


def monotone(values: list[float], *, nondecreasing: bool = True, tol: float = 1e-6) -> bool:
    if len(values) < 2:
        return True
    if nondecreasing:
        return all(b >= a - tol for a, b in zip(values, values[1:]))
    return all(b <= a + tol for a, b in zip(values, values[1:]))


def build_regime(table: pd.DataFrame, selected: pd.DataFrame) -> pd.DataFrame:
    prep = table[table["action"].isin(FORCED_ACTIONS)].copy()
    prep["action_positive_effect"] = is_action_positive(prep).astype(int)
    prep["action_regression_risk"] = is_action_regression(prep).astype(int)
    prep["action_negative_effect"] = is_action_negative(prep).astype(int)
    prep["axis_tradeoff"] = is_axis_tradeoff(prep).astype(int)
    selected_by_key = selected.set_index("probe_id").to_dict("index")
    rows: list[dict[str, Any]] = []
    scale_order = {MICRO50_ACTION: 0.50, PUMP_ACTION: 0.08 / 0.15, ACTIVE_ACTION: 1.0}
    for probe_id, group in prep.groupby("probe_id", dropna=False):
        group = group.copy()
        group["scale_order"] = group["action"].map(scale_order)
        group = group.sort_values(["scale_order", "action"])
        sel = selected_by_key.get(probe_id, {})
        pump_delta = num_series(group, "pump_delta_m3")
        time_delta = num_series(group, "delta_post60_time_over5")
        idle_delta = num_series(group, "delta_post60_idle_over5")
        p95_delta = num_series(group, "delta_full_max_p95")
        fallback_delta = num_series(group, "delta_full_fallback_pp")
        all_no_effect = bool(
            pump_delta.abs().max() < 0.01
            and time_delta.abs().max() < 0.5
            and idle_delta.abs().max() < 0.5
            and p95_delta.abs().max() < 1e-6
            and fallback_delta.abs().max() < 1e-6
        )
        positive = int(group["action_positive_effect"].max() > 0)
        regression = int(group["action_regression_risk"].max() > 0)
        negative = int(group["action_negative_effect"].max() > 0)
        axis = int(group["axis_tradeoff"].max() > 0)
        non_mono = int(
            not monotone(list(pump_delta), nondecreasing=True)
            or not monotone(list(num_series(group, "time_gain_s")), nondecreasing=True, tol=0.5)
            or not monotone(list(num_series(group, "idle_gain_s")), nondecreasing=True, tol=0.5)
        )
        micro = group[group["action"].eq(MICRO50_ACTION)]
        pump = group[group["action"].eq(PUMP_ACTION)]
        active = group[group["action"].eq(ACTIVE_ACTION)]
        anti_context = (
            "lowrisk" in str(sel.get("candidate_reason", ""))
            or "broader20_false_prepare" in str(sel.get("candidate_reason", ""))
            or "broader20_anti_trigger" in str(sel.get("candidate_reason", ""))
            or "direction_mismatch" in str(sel.get("candidate_reason", ""))
            or "near_safe_far_risky" in str(sel.get("candidate_reason", ""))
            or "far_persistent_high_alone" in str(sel.get("candidate_reason", ""))
            or (
                str(sel.get("dataset")) == "broader20"
                and as_float(sel.get("future_60m_time_over5")) <= 5.0
                and as_float(sel.get("future_60m_idle_over5")) <= 20.0
            )
        )
        micro_pos = int(micro["action_positive_effect"].max()) if not micro.empty else 0
        pump_pos = int(pump["action_positive_effect"].max()) if not pump.empty else 0
        anti_trigger = int(
            anti_context
            and (all_no_effect or negative or regression or axis)
            and micro_pos == 0
            and pump_pos == 0
        )
        best = group.iloc[int(np.argmax(num_series(group, "action_utility").to_numpy()))]
        row: dict[str, Any] = {
            "probe_id": probe_id,
            "bucket_key": group.iloc[0]["bucket_key"],
            "dataset": group.iloc[0]["dataset"],
            "case_id": group.iloc[0]["case_id"],
            "group_id": group.iloc[0]["group_id"],
            "bucket": int(group.iloc[0]["bucket"]),
            "timestamp": group.iloc[0]["timestamp"],
            "expansion_selection_category": sel.get("expansion_selection_category", ""),
            "candidate_reason": sel.get("candidate_reason", ""),
            "no_effect": int(all_no_effect),
            "effectful": int(not all_no_effect),
            "positive_effect": positive,
            "negative_effect": negative,
            "regression_risk": regression,
            "non_monotone": non_mono,
            "axis_tradeoff": axis,
            "anti_trigger": anti_trigger,
            "micro50_positive_effect": micro_pos,
            "micro50_negative_effect": int(micro["action_negative_effect"].max()) if not micro.empty else 0,
            "pump_saving_positive_effect": pump_pos,
            "pump_saving_negative_effect": int(pump["action_negative_effect"].max()) if not pump.empty else 0,
            "active_small_positive_effect": int(active["action_positive_effect"].max()) if not active.empty else 0,
            "active_small_regression_risk": int(active["action_regression_risk"].max()) if not active.empty else 0,
            "best_action_by_utility": best["action"],
            "best_action_utility": as_float(best["action_utility"]),
            "best_action_positive": int(best["action_positive_effect"]),
            "best_action_negative": int(best["action_negative_effect"]),
            "range_pump_delta_m3": float(pump_delta.max() - pump_delta.min()),
            "range_time_delta_s": float(time_delta.max() - time_delta.min()),
            "range_idle_delta_s": float(idle_delta.max() - idle_delta.min()),
        }
        for col in FEATURE_COLUMNS:
            row[col] = sel.get(col, group.iloc[0].get(col, 0.0))
        rows.append(row)
    regime = pd.DataFrame(rows)
    regime.to_csv(OUT / "expanded_regime_table.csv", index=False)
    regime.to_csv(RAW / "expanded_regime_table.csv", index=False)
    prep.to_csv(RAW / "expanded_action_labeled_table.csv", index=False)
    return regime


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
            max_iter=90,
            learning_rate=0.05,
            random_state=4242,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=350,
            max_depth=6,
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
            max_iter=90,
            learning_rate=0.05,
            random_state=4242,
        ),
        "random_forest": RandomForestRegressor(
            n_estimators=350,
            max_depth=6,
            min_samples_leaf=2,
            random_state=4242,
        ),
    }


@dataclass(frozen=True)
class RegimeHead:
    name: str
    label_col: str


STAGE_A_HEADS = [
    RegimeHead("anti_trigger_prob", "anti_trigger"),
    RegimeHead("regression_risk_prob", "regression_risk"),
    RegimeHead("axis_tradeoff_prob", "axis_tradeoff"),
    RegimeHead("no_effect_prob", "no_effect"),
    RegimeHead("effectful_prob", "effectful"),
]


def safe_auc(y_true: np.ndarray, prob: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score

    try:
        return float(roc_auc_score(y_true, prob)) if len(set(y_true.tolist())) > 1 else np.nan
    except ValueError:
        return np.nan


def train_stage_a(regime: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    from sklearn.metrics import precision_recall_fscore_support

    splits = make_splits(regime)
    x_all, _ = build_features(regime)
    metrics: list[dict[str, Any]] = []
    pred_rows: list[dict[str, Any]] = []
    if not splits:
        empty = pd.DataFrame()
        for name in (
            "stageA_regime_model_metrics.csv",
            "stageA_regime_oof_predictions.csv",
            "stageA_regime_high_confidence_table.csv",
            "stageA_feature_importance_table.csv",
        ):
            empty.to_csv(RAW / name, index=False)
        return empty, empty, empty, empty

    for head in STAGE_A_HEADS:
        if head.label_col not in regime:
            continue
        y_all = regime[head.label_col].astype(int).reset_index(drop=True)
        if y_all.nunique() < 2:
            continue
        for fold, train_idx, test_idx in splits:
            y_train = y_all.iloc[train_idx].to_numpy()
            y_test = y_all.iloc[test_idx].to_numpy()
            if len(set(y_train.tolist())) < 2:
                continue
            for model_name, model in classifiers().items():
                model.fit(x_all.iloc[train_idx], y_train)
                prob = model.predict_proba(x_all.iloc[test_idx])[:, 1]
                pred = (prob >= 0.5).astype(int)
                precision, recall, f1, _ = precision_recall_fscore_support(
                    y_test, pred, average="binary", zero_division=0
                )
                test = regime.iloc[test_idx].copy()
                false_pos = (pred == 1) & (y_test == 0)
                metrics.append(
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
                        "false_positive_broader20": int(
                            (false_pos & test["dataset"].astype(str).eq("broader20").to_numpy()).sum()
                        ),
                        "false_positive_lowrisk": int(
                            (
                                false_pos
                                & test["candidate_reason"].astype(str).str.contains("lowrisk", na=False).to_numpy()
                            ).sum()
                        ),
                        "false_positive_direction_mismatch": int(
                            (false_pos & (num_series(test, "direction_mismatch") > 0).to_numpy()).sum()
                        ),
                    }
                )
                keep_cols = [
                    "probe_id",
                    "bucket_key",
                    "dataset",
                    "case_id",
                    "group_id",
                    "bucket",
                    "candidate_reason",
                    "anti_trigger",
                    "regression_risk",
                    "axis_tradeoff",
                    "no_effect",
                    "effectful",
                    "positive_effect",
                    "negative_effect",
                    "micro50_positive_effect",
                    "pump_saving_positive_effect",
                ]
                sub = test[[c for c in keep_cols if c in test.columns]].copy()
                sub["head"] = head.name
                sub["model"] = model_name
                sub["fold"] = fold
                sub["target"] = y_test
                sub["prob"] = prob
                sub["pred"] = pred
                pred_rows.extend(sub.to_dict("records"))

    fold_metrics = pd.DataFrame(metrics)
    preds = pd.DataFrame(pred_rows)
    fold_metrics.to_csv(RAW / "stageA_regime_fold_metrics.csv", index=False)
    preds.to_csv(RAW / "stageA_regime_oof_predictions.csv", index=False)
    if fold_metrics.empty:
        agg = pd.DataFrame()
    else:
        agg = (
            fold_metrics.groupby(["head", "label_col", "model"], dropna=False)[
                [
                    "test_rows",
                    "test_cases",
                    "test_positive",
                    "precision",
                    "recall",
                    "f1",
                    "roc_auc",
                    "false_positive_broader20",
                    "false_positive_lowrisk",
                    "false_positive_direction_mismatch",
                ]
            ]
            .mean()
            .reset_index()
        )
    agg.to_csv(OUT / "stageA_regime_model_metrics.csv", index=False)
    agg.to_csv(RAW / "stageA_regime_model_metrics.csv", index=False)
    high_conf = build_stage_a_high_confidence(preds)
    importance = stage_a_feature_importance(regime)
    return agg, preds, high_conf, importance


def build_stage_a_high_confidence(preds: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if preds.empty:
        out = pd.DataFrame()
        out.to_csv(OUT / "stageA_regime_high_confidence_table.csv", index=False)
        out.to_csv(RAW / "stageA_regime_high_confidence_table.csv", index=False)
        return out
    for (head, model), group in preds.groupby(["head", "model"], dropna=False):
        for threshold in (0.50, 0.60, 0.70, 0.80, 0.90):
            picked = group[group["prob"] >= threshold].copy()
            rows.append(
                {
                    "head": head,
                    "model": model,
                    "threshold": threshold,
                    "selected_rows": int(len(picked)),
                    "selected_cases": int(picked["group_id"].nunique()) if not picked.empty else 0,
                    "precision": float(picked["target"].mean()) if not picked.empty else np.nan,
                    "broader20_selected": int(picked["dataset"].astype(str).eq("broader20").sum()) if not picked.empty else 0,
                    "lowrisk_selected": int(
                        picked["candidate_reason"].astype(str).str.contains("lowrisk", na=False).sum()
                    )
                    if not picked.empty
                    else 0,
                    "direction_mismatch_selected": int(
                        (num_series(picked, "direction_mismatch") > 0).sum()
                    )
                    if not picked.empty
                    else 0,
                    "positive_effect_rate": float(picked["positive_effect"].mean()) if "positive_effect" in picked and not picked.empty else np.nan,
                    "regression_risk_rate": float(picked["regression_risk"].mean()) if "regression_risk" in picked and not picked.empty else np.nan,
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "stageA_regime_high_confidence_table.csv", index=False)
    out.to_csv(RAW / "stageA_regime_high_confidence_table.csv", index=False)
    return out


def stage_a_feature_importance(regime: pd.DataFrame) -> pd.DataFrame:
    from sklearn.ensemble import RandomForestClassifier

    x, feature_names = build_features(regime)
    rows: list[dict[str, Any]] = []
    for head in STAGE_A_HEADS:
        if head.label_col not in regime:
            continue
        y = regime[head.label_col].astype(int)
        if y.nunique() < 2:
            continue
        model = RandomForestClassifier(
            n_estimators=350,
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
    out.to_csv(OUT / "stageA_feature_importance_table.csv", index=False)
    out.to_csv(RAW / "stageA_feature_importance_table.csv", index=False)
    return out


def best_model_by_head(metrics: pd.DataFrame) -> dict[str, str]:
    best: dict[str, str] = {}
    if metrics.empty:
        return best
    for head, group in metrics.groupby("head", dropna=False):
        score_col = "recall" if head in {"anti_trigger_prob", "regression_risk_prob"} else "f1"
        ranked = group.sort_values([score_col, "roc_auc", "precision"], ascending=[False, False, False])
        if not ranked.empty:
            best[str(head)] = str(ranked.iloc[0]["model"])
    return best


def pivot_best_predictions(preds: pd.DataFrame, metrics: pd.DataFrame) -> pd.DataFrame:
    best = best_model_by_head(metrics)
    rows = []
    if preds.empty:
        return pd.DataFrame()
    for head, model in best.items():
        sub = preds[(preds["head"] == head) & (preds["model"] == model)].copy()
        sub = sub[["probe_id", "bucket_key", "group_id", "prob"]].rename(columns={"prob": head})
        rows.append(sub)
    if not rows:
        return pd.DataFrame()
    out = rows[0]
    for sub in rows[1:]:
        out = out.merge(sub, on=["probe_id", "bucket_key", "group_id"], how="outer")
    return out


def clean_subset_diagnostics(regime: pd.DataFrame, preds: pd.DataFrame, metrics: pd.DataFrame) -> pd.DataFrame:
    pivot = pivot_best_predictions(preds, metrics)
    if pivot.empty:
        out = pd.DataFrame()
        out.to_csv(RAW / "stageA_clean_subset_table.csv", index=False)
        return out
    scored = regime.merge(pivot, on=["probe_id", "bucket_key", "group_id"], how="left")
    rows: list[dict[str, Any]] = []
    for guard_t in (0.35, 0.45, 0.55):
        for effect_t in (0.55, 0.65, 0.75):
            mask = (
                (num_series(scored, "anti_trigger_prob", 1.0) <= guard_t)
                & (num_series(scored, "regression_risk_prob", 1.0) <= guard_t)
                & (num_series(scored, "axis_tradeoff_prob", 1.0) <= guard_t)
                & (num_series(scored, "no_effect_prob", 1.0) <= guard_t)
                & (num_series(scored, "effectful_prob", 0.0) >= effect_t)
            )
            picked = scored[mask].copy()
            rows.append(
                {
                    "guard_threshold_max": guard_t,
                    "effectful_threshold_min": effect_t,
                    "selected_buckets": int(len(picked)),
                    "selected_cases": int(picked["group_id"].nunique()) if not picked.empty else 0,
                    "positive_effect_rate": float(picked["positive_effect"].mean()) if not picked.empty else np.nan,
                    "micro50_positive_rate": float(picked["micro50_positive_effect"].mean()) if not picked.empty else np.nan,
                    "pump_saving_positive_rate": float(picked["pump_saving_positive_effect"].mean()) if not picked.empty else np.nan,
                    "anti_trigger_rate": float(picked["anti_trigger"].mean()) if not picked.empty else np.nan,
                    "regression_risk_rate": float(picked["regression_risk"].mean()) if not picked.empty else np.nan,
                    "axis_tradeoff_rate": float(picked["axis_tradeoff"].mean()) if not picked.empty else np.nan,
                    "no_effect_rate": float(picked["no_effect"].mean()) if not picked.empty else np.nan,
                    "broader20_selected": int(picked["dataset"].astype(str).eq("broader20").sum()) if not picked.empty else 0,
                    "lowrisk_selected": int(
                        picked["candidate_reason"].astype(str).str.contains("lowrisk", na=False).sum()
                    )
                    if not picked.empty
                    else 0,
                    "direction_mismatch_selected": int((num_series(picked, "direction_mismatch") > 0).sum())
                    if not picked.empty
                    else 0,
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "stageA_clean_subset_table.csv", index=False)
    out.to_csv(RAW / "stageA_clean_subset_table.csv", index=False)
    scored.to_csv(RAW / "stageA_bucket_oof_score_table.csv", index=False)
    return out


def evaluate_stage_a(
    metrics: pd.DataFrame, clean: pd.DataFrame, regime: pd.DataFrame
) -> tuple[bool, dict[str, Any]]:
    evidence: dict[str, Any] = {
        "anti_trigger_best_recall": np.nan,
        "regression_risk_best_recall": np.nan,
        "best_clean_selected_buckets": 0,
        "best_clean_selected_cases": 0,
        "best_clean_positive_effect_rate": np.nan,
        "best_clean_bad_rate": np.nan,
        "best_clean_broader20_selected": 0,
        "case_concentration_max_share": np.nan,
    }
    if not metrics.empty:
        anti = metrics[metrics["head"].eq("anti_trigger_prob")].sort_values(["recall", "precision"], ascending=False)
        reg = metrics[metrics["head"].eq("regression_risk_prob")].sort_values(["recall", "precision"], ascending=False)
        if not anti.empty:
            evidence["anti_trigger_best_recall"] = as_float(anti.iloc[0]["recall"], np.nan)
            evidence["anti_trigger_best_model"] = str(anti.iloc[0]["model"])
        if not reg.empty:
            evidence["regression_risk_best_recall"] = as_float(reg.iloc[0]["recall"], np.nan)
            evidence["regression_risk_best_model"] = str(reg.iloc[0]["model"])
    if not clean.empty:
        c = clean[clean["selected_buckets"] >= 8].copy()
        if not c.empty:
            c["bad_rate"] = c[["anti_trigger_rate", "regression_risk_rate", "axis_tradeoff_rate", "no_effect_rate"]].max(axis=1)
            ranked = c.sort_values(["bad_rate", "positive_effect_rate", "selected_cases"], ascending=[True, False, False])
            best = ranked.iloc[0]
            evidence["best_clean_guard_threshold_max"] = as_float(best["guard_threshold_max"])
            evidence["best_clean_effectful_threshold_min"] = as_float(best["effectful_threshold_min"])
            evidence["best_clean_selected_buckets"] = int(best["selected_buckets"])
            evidence["best_clean_selected_cases"] = int(best["selected_cases"])
            evidence["best_clean_positive_effect_rate"] = as_float(best["positive_effect_rate"], np.nan)
            evidence["best_clean_bad_rate"] = as_float(best["bad_rate"], np.nan)
            evidence["best_clean_broader20_selected"] = int(best["broader20_selected"])
    if not regime.empty:
        case_counts = regime.groupby("group_id").size()
        if case_counts.sum() > 0:
            evidence["case_concentration_max_share"] = float(case_counts.max() / case_counts.sum())
    go = (
        evidence["anti_trigger_best_recall"] >= 0.80
        and evidence["regression_risk_best_recall"] >= 0.75
        and evidence["best_clean_selected_cases"] >= 3
        and evidence["best_clean_selected_buckets"] >= 8
        and (
            pd.isna(evidence["best_clean_bad_rate"])
            or evidence["best_clean_bad_rate"] <= 0.35
        )
        and evidence["case_concentration_max_share"] <= 0.35
    )
    pd.DataFrame([evidence]).to_csv(RAW / "stageA_go_no_go_evidence.csv", index=False)
    return bool(go), evidence


def build_stage_b_subset(
    action_table: pd.DataFrame, regime: pd.DataFrame, metrics: pd.DataFrame, preds: pd.DataFrame, clean: pd.DataFrame
) -> pd.DataFrame:
    scored_path = RAW / "stageA_bucket_oof_score_table.csv"
    if not scored_path.exists() or clean.empty:
        subset = pd.DataFrame()
        subset.to_csv(RAW / "stageB_clean_action_subset.csv", index=False)
        return subset
    scored = pd.read_csv(scored_path)
    c = clean[clean["selected_buckets"] >= 8].copy()
    if c.empty:
        subset = pd.DataFrame()
        subset.to_csv(RAW / "stageB_clean_action_subset.csv", index=False)
        return subset
    c["bad_rate"] = c[["anti_trigger_rate", "regression_risk_rate", "axis_tradeoff_rate", "no_effect_rate"]].max(axis=1)
    chosen = c.sort_values(["bad_rate", "positive_effect_rate", "selected_cases"], ascending=[True, False, False]).iloc[0]
    mask = (
        (num_series(scored, "anti_trigger_prob", 1.0) <= as_float(chosen["guard_threshold_max"]))
        & (num_series(scored, "regression_risk_prob", 1.0) <= as_float(chosen["guard_threshold_max"]))
        & (num_series(scored, "axis_tradeoff_prob", 1.0) <= as_float(chosen["guard_threshold_max"]))
        & (num_series(scored, "no_effect_prob", 1.0) <= as_float(chosen["guard_threshold_max"]))
        & (num_series(scored, "effectful_prob", 0.0) >= as_float(chosen["effectful_threshold_min"]))
    )
    keys = set(scored.loc[mask, "bucket_key"].astype(str))
    subset = action_table[
        action_table["bucket_key"].astype(str).isin(keys)
        & action_table["action"].isin([WAIT_ACTION, MICRO50_ACTION, PUMP_ACTION])
    ].copy()
    subset["group_id"] = subset["dataset"].astype(str) + "::" + subset["case_id"].astype(str)
    subset.to_csv(RAW / "stageB_clean_action_subset.csv", index=False)
    return subset


def train_stage_b(subset: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from sklearn.metrics import accuracy_score

    if subset.empty or subset["group_id"].nunique() < 2:
        delta_empty = pd.DataFrame(
            columns=["head", "model", "test_rows", "mae", "rmse", "r2"]
        )
        pair_empty = pd.DataFrame(
            columns=[
                "model",
                "test_rows",
                "test_positive",
                "accuracy",
                "roc_auc",
                "high_conf_prepare_rows",
                "high_conf_prepare_precision",
                "broader20_high_conf",
            ]
        )
        delta_empty.to_csv(OUT / "stageB_clean_expected_delta_metrics.csv", index=False)
        delta_empty.to_csv(RAW / "stageB_clean_expected_delta_metrics.csv", index=False)
        pair_empty.to_csv(OUT / "stageB_pairwise_metrics.csv", index=False)
        pair_empty.to_csv(RAW / "stageB_pairwise_metrics.csv", index=False)
        return delta_empty, pair_empty
    x_all, _ = build_features(subset, action_col="action")
    splits = make_splits(subset)
    rows: list[dict[str, Any]] = []
    for target in ("time_gain_s", "idle_gain_s", "pump_delta_m3", "action_utility"):
        if target not in subset:
            continue
        y = pd.to_numeric(subset[target], errors="coerce").fillna(0.0).reset_index(drop=True)
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
    delta_fold = pd.DataFrame(rows)
    delta_fold.to_csv(RAW / "stageB_clean_expected_delta_fold_metrics.csv", index=False)
    delta = (
        delta_fold.groupby(["head", "model"], dropna=False)[["test_rows", "mae", "rmse", "r2"]]
        .mean()
        .reset_index()
        if not delta_fold.empty
        else pd.DataFrame()
    )
    delta.to_csv(OUT / "stageB_clean_expected_delta_metrics.csv", index=False)
    delta.to_csv(RAW / "stageB_clean_expected_delta_metrics.csv", index=False)

    pair_rows: list[dict[str, Any]] = []
    for bucket_key, group in subset.groupby("bucket_key", dropna=False):
        by_action = {str(r.action): r for r in group.itertuples(index=False)}
        if WAIT_ACTION not in by_action:
            continue
        for action in (MICRO50_ACTION, PUMP_ACTION):
            if action not in by_action:
                continue
            wait = by_action[WAIT_ACTION]
            prep = by_action[action]
            pair_rows.append(
                {
                    "pair_id": f"{bucket_key}::{action}",
                    "bucket_key": bucket_key,
                    "dataset": prep.dataset,
                    "case_id": prep.case_id,
                    "group_id": prep.group_id,
                    "bucket": int(prep.bucket),
                    "action": action,
                    "prepare_preferred": int(float(prep.action_utility) > float(wait.action_utility)),
                    "utility_margin": float(prep.action_utility) - float(wait.action_utility),
                    "time_gain_s": float(prep.time_gain_s),
                    "idle_gain_s": float(prep.idle_gain_s),
                    "pump_delta_m3": float(prep.pump_delta_m3),
                }
            )
            for col in FEATURE_COLUMNS:
                pair_rows[-1][col] = getattr(prep, col, 0.0)
    pair = pd.DataFrame(pair_rows)
    pair.to_csv(RAW / "stageB_pairwise_source_table.csv", index=False)
    if pair.empty or pair["prepare_preferred"].nunique() < 2 or pair["group_id"].nunique() < 2:
        pair.to_csv(OUT / "stageB_pairwise_metrics.csv", index=False)
        pair.to_csv(RAW / "stageB_pairwise_metrics.csv", index=False)
        return delta, pair
    x_pair, _ = build_features(pair, action_col="action")
    splits_pair = make_splits(pair)
    metrics: list[dict[str, Any]] = []
    for fold, train_idx, test_idx in splits_pair:
        y_train = pair["prepare_preferred"].iloc[train_idx].astype(int).to_numpy()
        y_test = pair["prepare_preferred"].iloc[test_idx].astype(int).to_numpy()
        if len(set(y_train.tolist())) < 2:
            continue
        for model_name, model in classifiers().items():
            model.fit(x_pair.iloc[train_idx], y_train)
            prob = model.predict_proba(x_pair.iloc[test_idx])[:, 1]
            pred = (prob >= 0.5).astype(int)
            metrics.append(
                {
                    "model": model_name,
                    "fold": fold,
                    "test_rows": int(len(test_idx)),
                    "test_positive": int(y_test.sum()),
                    "accuracy": float(accuracy_score(y_test, pred)),
                    "roc_auc": safe_auc(y_test, prob),
                    "high_conf_prepare_rows": int((prob >= 0.70).sum()),
                    "high_conf_prepare_precision": float(y_test[prob >= 0.70].mean()) if (prob >= 0.70).sum() else np.nan,
                    "broader20_high_conf": int(
                        ((prob >= 0.70) & pair.iloc[test_idx]["dataset"].astype(str).eq("broader20").to_numpy()).sum()
                    ),
                }
            )
    fold_pair = pd.DataFrame(metrics)
    fold_pair.to_csv(RAW / "stageB_pairwise_fold_metrics.csv", index=False)
    pair_metrics = (
        fold_pair.groupby("model", dropna=False)[
            ["test_rows", "test_positive", "accuracy", "roc_auc", "high_conf_prepare_rows", "high_conf_prepare_precision", "broader20_high_conf"]
        ]
        .mean()
        .reset_index()
        if not fold_pair.empty
        else pd.DataFrame()
    )
    pair_metrics.to_csv(OUT / "stageB_pairwise_metrics.csv", index=False)
    pair_metrics.to_csv(RAW / "stageB_pairwise_metrics.csv", index=False)
    return delta, pair_metrics


def case_concentration(regime: pd.DataFrame, action_table: pd.DataFrame) -> pd.DataFrame:
    prep = action_table[action_table["action"].isin(FORCED_ACTIONS)].copy()
    rows = []
    for group_id, group in regime.groupby("group_id", dropna=False):
        action_group = prep[prep["group_id"].eq(group_id)]
        rows.append(
            {
                "group_id": group_id,
                "dataset": group["dataset"].iloc[0],
                "case_id": group["case_id"].iloc[0],
                "buckets": int(len(group)),
                "positive_effect_buckets": int(group["positive_effect"].sum()),
                "negative_effect_buckets": int(group["negative_effect"].sum()),
                "regression_risk_buckets": int(group["regression_risk"].sum()),
                "anti_trigger_buckets": int(group["anti_trigger"].sum()),
                "micro50_positive_buckets": int(group["micro50_positive_effect"].sum()),
                "pump_saving_positive_buckets": int(group["pump_saving_positive_effect"].sum()),
                "forced_rows": int(len(action_group)),
                "forced_pump_delta_mean": float(num_series(action_group, "pump_delta_m3").mean()) if not action_group.empty else np.nan,
                "forced_time_gain_mean": float(num_series(action_group, "time_gain_s").mean()) if not action_group.empty else np.nan,
                "forced_idle_gain_mean": float(num_series(action_group, "idle_gain_s").mean()) if not action_group.empty else np.nan,
            }
        )
    out = pd.DataFrame(rows).sort_values(["buckets", "positive_effect_buckets"], ascending=False)
    out.to_csv(OUT / "case_concentration_table.csv", index=False)
    out.to_csv(RAW / "case_concentration_table.csv", index=False)
    return out


def write_phase_status(phase: str, status: str, go_no_go: str, files: list[Path], reason: str) -> None:
    import json

    payload = {
        "phase": phase,
        "status": status,
        "go_no_go": go_no_go,
        "evidence_files": [str(p.relative_to(REPO)) for p in files],
        "reason": reason,
    }
    with (DEBUG / "phase_status.jsonl").open("a") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def write_reports(
    selected: pd.DataFrame,
    action_table: pd.DataFrame,
    regime: pd.DataFrame,
    stage_a_metrics: pd.DataFrame,
    stage_a_clean: pd.DataFrame,
    stage_a_go: bool,
    stage_a_evidence: dict[str, Any],
    stage_b_delta: pd.DataFrame,
    stage_b_pair: pd.DataFrame,
    concentration: pd.DataFrame,
) -> None:
    regime_counts = pd.DataFrame(
        [
            {"label": col, "count": int(regime[col].sum()), "rate": float(regime[col].mean())}
            for col in [
                "no_effect",
                "effectful",
                "positive_effect",
                "negative_effect",
                "regression_risk",
                "non_monotone",
                "axis_tradeoff",
                "anti_trigger",
                "micro50_positive_effect",
                "pump_saving_positive_effect",
            ]
            if col in regime
        ]
    )
    dataset_counts = (
        regime.groupby("dataset")[
            [
                "no_effect",
                "effectful",
                "positive_effect",
                "negative_effect",
                "regression_risk",
                "axis_tradeoff",
                "anti_trigger",
                "micro50_positive_effect",
                "pump_saving_positive_effect",
            ]
        ]
        .agg(["sum", "mean"])
        .reset_index()
    )
    action_summary = (
        action_table[action_table["action"].isin(FORCED_ACTIONS)]
        .groupby(["dataset", "action"])
        .agg(
            rows=("probe_id", "count"),
            mean_pump_delta=("pump_delta_m3", "mean"),
            mean_time_gain=("time_gain_s", "mean"),
            mean_idle_gain=("idle_gain_s", "mean"),
            safe_positive_rate=("action_label", lambda s: float((s == "safe_positive").mean())),
            negative_rate=("action_label", lambda s: float((s == "negative").mean())),
        )
        .reset_index()
    )
    regime_counts.to_csv(PAPER / "action_effect_regime_counts.csv", index=False)
    action_summary.to_csv(PAPER / "action_effect_by_action_dataset.csv", index=False)
    if not stage_a_metrics.empty:
        stage_a_metrics.to_csv(PAPER / "stageA_regime_model_metrics.csv", index=False)
    if not stage_b_pair.empty:
        stage_b_pair.to_csv(PAPER / "stageB_pairwise_metrics.csv", index=False)

    best_delta_r2 = (
        float(stage_b_delta["r2"].max())
        if stage_b_delta is not None and not stage_b_delta.empty and "r2" in stage_b_delta
        else np.nan
    )
    pair_auc = (
        float(stage_b_pair["roc_auc"].mean())
        if stage_b_pair is not None and not stage_b_pair.empty and "roc_auc" in stage_b_pair
        else np.nan
    )
    pair_best_auc = (
        float(stage_b_pair["roc_auc"].max())
        if stage_b_pair is not None and not stage_b_pair.empty and "roc_auc" in stage_b_pair
        else np.nan
    )
    micro_guard = action_summary[
        action_summary["dataset"].eq("guard10") & action_summary["action"].eq(MICRO50_ACTION)
    ]
    micro_broad = action_summary[
        action_summary["dataset"].eq("broader20") & action_summary["action"].eq(MICRO50_ACTION)
    ]
    micro_guard_pos = as_float(micro_guard.iloc[0]["safe_positive_rate"], np.nan) if not micro_guard.empty else np.nan
    micro_broad_neg = as_float(micro_broad.iloc[0]["negative_rate"], np.nan) if not micro_broad.empty else np.nan

    finding = [
        "# h120 Action-Effect Dataset Expansion Findings",
        "",
        f"- Locked buckets: {len(selected)}.",
        f"- Counterfactual rows: {len(action_table)}; forced prepare rows: {int(action_table['action'].isin(FORCED_ACTIONS).sum())}.",
        f"- micro_prepare_50 guard10 safe-positive rate: {micro_guard_pos:.3f}.",
        f"- micro_prepare_50 broader20 negative rate: {micro_broad_neg:.3f}.",
        f"- Stage A go/no-go: {'go' if stage_a_go else 'no-go'}; anti_trigger recall={stage_a_evidence.get('anti_trigger_best_recall', np.nan):.3f}, regression recall={stage_a_evidence.get('regression_risk_best_recall', np.nan):.3f}.",
        f"- Stage B best expected-delta R2: {best_delta_r2:.3f}; mean pairwise AUC: {pair_auc:.3f}; best pairwise AUC: {pair_best_auc:.3f}.",
        "",
        "## Regime Counts",
        markdown_table(regime_counts),
        "",
        "## Action Summary",
        markdown_table(action_summary, max_rows=24),
        "",
        "## Stage A Evidence",
        markdown_table(pd.DataFrame([stage_a_evidence])),
        "",
        "## Case Concentration",
        markdown_table(concentration.head(12)),
    ]
    (PAPER / "action_effect_expansion_findings.md").write_text("\n".join(finding) + "\n")

    stage_b_ready = (
        pd.notna(pair_auc)
        and pair_auc >= 0.70
        and (pd.isna(best_delta_r2) or best_delta_r2 > -0.05)
    )
    allow_shadow = bool(stage_a_go and stage_b_ready)
    decision = [
        "# h120 Action-Effect Dataset Expansion Decision",
        "",
        "Scope: offline action-effect expansion only.  No controller gate was attached, v1.6 delayed-medium reactive floor was not changed, and active_small_prepare remains an upper reference rather than a first gate action.",
        "",
        "## Dataset",
        "",
        f"- Locked buckets: {len(selected)}.",
        f"- Counterfactual rows: {len(action_table)}.",
        f"- Forced prepare rows: {int(action_table['action'].isin(FORCED_ACTIONS).sum())}.",
        "",
        "## Regime Summary",
        "",
        markdown_table(regime_counts),
        "",
        "## Stage A",
        "",
        f"- anti_trigger best recall: {stage_a_evidence.get('anti_trigger_best_recall', np.nan):.3f}.",
        f"- regression_risk best recall: {stage_a_evidence.get('regression_risk_best_recall', np.nan):.3f}.",
        f"- best clean subset: {stage_a_evidence.get('best_clean_selected_buckets', 0)} buckets across {stage_a_evidence.get('best_clean_selected_cases', 0)} cases, bad-rate={stage_a_evidence.get('best_clean_bad_rate', np.nan):.3f}.",
        f"- Stage A decision: {'go' if stage_a_go else 'no-go'}.",
        "",
        "## Stage B",
        "",
        f"- Stage B {'was executed' if stage_a_go else 'was not executed'} because Stage A was {'go' if stage_a_go else 'no-go'}.",
        f"- Best expected-delta R2: {best_delta_r2:.3f}.",
        f"- Mean pairwise AUC: {pair_auc:.3f}; best pairwise AUC: {pair_best_auc:.3f}.",
        "",
        "## Required Answers",
        "",
        f"1. **After expansion, are regression_risk / anti_trigger more learnable?**  {'Partly yes' if stage_a_evidence.get('anti_trigger_best_recall', 0) >= 0.8 or stage_a_evidence.get('regression_risk_best_recall', 0) >= 0.75 else 'Not enough'}.  The key numbers are anti_trigger recall {stage_a_evidence.get('anti_trigger_best_recall', np.nan):.3f} and regression_risk recall {stage_a_evidence.get('regression_risk_best_recall', np.nan):.3f}.",
        f"2. **Is micro_prepare_50 still the first candidate?**  Yes, keep it as the first small-action candidate if any prepare action survives.  It remains more controlled than active_small_prepare, with guard10 safe-positive rate {micro_guard_pos:.3f} and broader20 negative rate {micro_broad_neg:.3f}; the broader20 rate is still the limiting risk.",
        f"3. **Can broader20 / lowrisk false prepare be stably blocked?**  {'Yes enough for a next offline shadow-style diagnostic' if stage_a_go else 'Not yet'}.  The clean subset bad-rate is {stage_a_evidence.get('best_clean_bad_rate', np.nan):.3f}; this must remain low before any future gate probe.",
        f"4. **Did clean-subset expected-delta / ranking improve?**  {'Yes enough for a future shadow gate diagnostic' if stage_b_ready else 'No, not enough yet'}.  Best R2={best_delta_r2:.3f}, mean pairwise AUC={pair_auc:.3f}.",
        f"5. **Allow shadow gate?**  {'Yes, only log-only shadow gate, not controller, if the user chooses to continue' if allow_shadow else 'No.  Do not enter shadow gate from this result.'}",
        "6. **If not allowed, next step?**  Continue expanding targeted anti-trigger/regression-risk and rare positive micro_prepare_50 samples, or redesign the prepare action/state interface.  Do not return to threshold tuning of far-risk labels.",
        "",
        "## Bottom Line",
        "",
        ("Expansion produced enough Stage A evidence to justify a next log-only shadow diagnostic, but the controller gate remains off."
         if allow_shadow else
         "Expansion is useful as an offline dataset, but it is still not enough to move into shadow gate or controller gate.  v1.6 remains the control mainline while h120 remains an offline action-effect research path."),
    ]
    (OUT / "h120_action_effect_dataset_expansion_decision.md").write_text("\n".join(decision) + "\n")


def main() -> None:
    ensure_dirs()
    write_phase_status(
        "phase0",
        "started",
        "go",
        [],
        "Using existing h120 action-value/action-family artifacts; no old task rerun.",
    )
    pool = load_candidate_pool()
    selected = select_candidates(pool)
    write_phase_status(
        "phase1_candidate_expansion",
        "completed",
        "go" if len(selected) >= 120 else "no-go",
        [OUT / "expanded_candidate_table.csv"],
        f"selected {len(selected)} locked buckets",
    )
    if len(selected) < 120:
        empty = pd.DataFrame()
        for name in [
            "expanded_counterfactual_table.csv",
            "expanded_regime_table.csv",
            "stageA_regime_model_metrics.csv",
            "stageB_clean_expected_delta_metrics.csv",
            "stageB_pairwise_metrics.csv",
            "case_concentration_table.csv",
        ]:
            empty.to_csv(OUT / name, index=False)
            empty.to_csv(RAW / name, index=False)
        (OUT / "h120_action_effect_dataset_expansion_decision.md").write_text(
            "# h120 Action-Effect Dataset Expansion Decision\n\nNo-go: fewer than 120 candidate buckets were available after stratified selection.\n"
        )
        return

    action_table = run_counterfactuals(selected)
    write_phase_status(
        "phase2_counterfactual",
        "completed",
        "go",
        [OUT / "expanded_counterfactual_table.csv"],
        f"built {len(action_table)} counterfactual rows",
    )
    regime = build_regime(action_table, selected)
    concentration = case_concentration(regime, action_table)
    stage_a_metrics, stage_a_preds, stage_a_hc, importance = train_stage_a(regime)
    stage_a_clean = clean_subset_diagnostics(regime, stage_a_preds, stage_a_metrics)
    stage_a_go, stage_a_evidence = evaluate_stage_a(stage_a_metrics, stage_a_clean, regime)
    write_phase_status(
        "phase3_stageA_regime_heads",
        "completed",
        "go" if stage_a_go else "no-go",
        [
            OUT / "stageA_regime_model_metrics.csv",
            OUT / "stageA_clean_subset_table.csv",
            OUT / "case_concentration_table.csv",
        ],
        "Stage A guards passed" if stage_a_go else "Stage A guards did not pass",
    )

    if stage_a_go:
        subset = build_stage_b_subset(action_table, regime, stage_a_metrics, stage_a_preds, stage_a_clean)
        stage_b_delta, stage_b_pair = train_stage_b(subset)
        write_phase_status(
            "phase4_stageB_clean_expected_delta",
            "completed",
            "go" if not stage_b_pair.empty else "no-go",
            [OUT / "stageB_clean_expected_delta_metrics.csv", OUT / "stageB_pairwise_metrics.csv"],
            f"stage B subset rows={len(subset)}",
        )
    else:
        stage_b_delta = pd.DataFrame(
            columns=["head", "model", "test_rows", "mae", "rmse", "r2"]
        )
        stage_b_pair = pd.DataFrame(
            columns=[
                "model",
                "test_rows",
                "test_positive",
                "accuracy",
                "roc_auc",
                "high_conf_prepare_rows",
                "high_conf_prepare_precision",
                "broader20_high_conf",
            ]
        )
        stage_b_delta.to_csv(OUT / "stageB_clean_expected_delta_metrics.csv", index=False)
        stage_b_delta.to_csv(RAW / "stageB_clean_expected_delta_metrics.csv", index=False)
        stage_b_pair.to_csv(OUT / "stageB_pairwise_metrics.csv", index=False)
        stage_b_pair.to_csv(RAW / "stageB_pairwise_metrics.csv", index=False)

    write_reports(
        selected,
        action_table,
        regime,
        stage_a_metrics,
        stage_a_clean,
        stage_a_go,
        stage_a_evidence,
        stage_b_delta,
        stage_b_pair,
        concentration,
    )
    write_phase_status(
        "phase5_decision",
        "completed",
        "go" if stage_a_go else "no-go",
        [OUT / "h120_action_effect_dataset_expansion_decision.md"],
        "final report written",
    )


if __name__ == "__main__":
    main()
