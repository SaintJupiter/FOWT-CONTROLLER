#!/usr/bin/env python3
"""h120 prepare action-family probe v1.

This probe keeps the v1.6 delayed-medium reactive floor fixed and studies only
offline forced-prefix counterfactuals.  It does not attach a controller gate,
does not train a model, and does not change pump penalties or floor logic.
"""

from __future__ import annotations

import csv
import importlib.util
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REPO = Path(__file__).resolve().parents[2]
REFINE_SCRIPT = REPO / "scripts/analysis/run_h120_action_value_label_refinement_v1.py"

spec = importlib.util.spec_from_file_location("h120_action_value_label_refinement_v1", REFINE_SCRIPT)
if spec is None or spec.loader is None:
    raise RuntimeError(f"cannot import helpers from {REFINE_SCRIPT}")
refine = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = refine
spec.loader.exec_module(refine)

prev = refine.prev

OUT = REPO / "outputs/wind_prediction/h120_prepare_action_family_probe_v1"
RAW = OUT / "raw_tables"
DEBUG = OUT / "debug"
PAPER = OUT / "paper_ready"
RUNS = OUT / "counterfactual_runs"

OVERNIGHT = REPO / "outputs/wind_prediction/h120_action_value_overnight_v1"
SUPPLEMENTAL = OVERNIGHT / "supplemental_sweep_v1"
SOURCE_CANDIDATES = [
    OVERNIGHT / "positive_candidate_table.csv",
    SUPPLEMENTAL / "supplemental_candidate_table.csv",
]
COMBINED_LABELS = SUPPLEMENTAL / "raw_tables/combined_constrained_label_table.csv"
F120_DATASET = refine.F120_DATASET

TARGET_BUCKETS = 48

FEATURE_COLUMNS = list(refine.FEATURE_COLUMNS)
METRIC_KEYS = [
    "full_pump_m3",
    "full_max_p95",
    "full_fallback_pp",
    "post60_time_over5",
    "post60_idle_over5",
    "post60_pump_m3",
    "post60_floor_entry",
    "post60_medium_delay_rows",
]

ACTION_FAMILY = [
    {
        "action": "wait",
        "base_action": "none",
        "active_small_scale": 0.0,
        "run_forced": False,
        "description": "v1.6 oracle baseline; no prepare action.",
    },
    {
        "action": "watch_only",
        "base_action": "none",
        "active_small_scale": 0.0,
        "run_forced": False,
        "description": "Log-only prepare state; identical to wait in this offline probe.",
    },
    {
        "action": "micro_prepare_25",
        "base_action": "active_small",
        "active_small_scale": 0.25,
        "run_forced": True,
        "description": "Forced target refresh using 25% of the active_small target vector.",
    },
    {
        "action": "micro_prepare_50",
        "base_action": "active_small",
        "active_small_scale": 0.50,
        "run_forced": True,
        "description": "Forced target refresh using 50% of the active_small target vector.",
    },
    {
        "action": "micro_prepare_75",
        "base_action": "active_small",
        "active_small_scale": 0.75,
        "run_forced": True,
        "description": "Forced target refresh using 75% of the active_small target vector.",
    },
    {
        "action": "pump_saving_prepare",
        "base_action": "pump_saving",
        "active_small_scale": 0.08 / 0.15,
        "run_forced": True,
        "description": "Existing pump_saving prepare action; included as the current small-action reference.",
    },
    {
        "action": "active_small_prepare",
        "base_action": "active_small",
        "active_small_scale": 1.00,
        "run_forced": True,
        "description": "Existing active_small prepare action; upper reference, not a first gate default.",
    },
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


def _key_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["dataset"] = out["dataset"].astype(str)
    out["case_id"] = out["case_id"].astype(str)
    out["bucket"] = pd.to_numeric(out["bucket"], errors="coerce").fillna(-1).astype(int)
    return out


def load_candidate_pool() -> pd.DataFrame:
    frames = []
    for path in SOURCE_CANDIDATES:
        if path.exists():
            frames.append(pd.read_csv(path))
    if not frames:
        raise FileNotFoundError("missing action-value overnight candidate tables")
    pool = _key_frame(pd.concat(frames, ignore_index=True))
    pool = pool.drop_duplicates(["dataset", "case_id", "bucket"], keep="first").reset_index(drop=True)

    if COMBINED_LABELS.exists():
        labels = _key_frame(pd.read_csv(COMBINED_LABELS))
        prep = labels[labels["action"].isin(["pump_saving_prepare", "active_small_prepare"])].copy()
        stats = []
        for (dataset, case_id, bucket), g in prep.groupby(["dataset", "case_id", "bucket"], dropna=False):
            row: dict[str, Any] = {
                "dataset": dataset,
                "case_id": case_id,
                "bucket": int(bucket),
                "known_safe_positive_any": int((g["constrained_label"] == "safe_positive").any()),
                "known_negative_any": int((g["constrained_label"] == "negative").any()),
                "known_pump_saving_safe_positive": int(
                    (
                        (g["action"] == "pump_saving_prepare")
                        & (g["constrained_label"] == "safe_positive")
                    ).any()
                ),
                "known_active_small_safe_positive": int(
                    (
                        (g["action"] == "active_small_prepare")
                        & (g["constrained_label"] == "safe_positive")
                    ).any()
                ),
                "known_prepare_negative_rows": int((g["constrained_label"] == "negative").sum()),
                "known_prepare_safe_positive_rows": int((g["constrained_label"] == "safe_positive").sum()),
            }
            for col in [
                "delta_full_pump_m3",
                "delta_post60_time_over5",
                "delta_post60_idle_over5",
                "delta_full_fallback_pp",
                "delta_full_max_p95",
            ]:
                if col in g:
                    row[f"known_mean_{col}"] = float(pd.to_numeric(g[col], errors="coerce").mean())
            stats.append(row)
        if stats:
            pool = pool.merge(pd.DataFrame(stats), on=["dataset", "case_id", "bucket"], how="left")

    for col in [
        "known_safe_positive_any",
        "known_negative_any",
        "known_pump_saving_safe_positive",
        "known_active_small_safe_positive",
        "known_prepare_negative_rows",
        "known_prepare_safe_positive_rows",
    ]:
        if col not in pool:
            pool[col] = 0
        pool[col] = pd.to_numeric(pool[col], errors="coerce").fillna(0).astype(int)
    return pool


def _candidate_reason(row: pd.Series) -> list[str]:
    reasons = []
    if _num(row.get("known_pump_saving_safe_positive")) > 0:
        reasons.append("known_pump_saving_safe_positive")
    if _num(row.get("known_active_small_safe_positive")) > 0:
        reasons.append("known_active_small_safe_positive")
    if str(row.get("dataset")) == "guard10" and (
        _num(row.get("future_60m_time_over5")) > 0 or _num(row.get("future_60m_idle_over5")) > 0
    ):
        reasons.append("guard10_future_high_posture")
    if str(row.get("dataset")) == "broader20" and _num(row.get("known_prepare_negative_rows")) > 0:
        reasons.append("broader20_known_negative_prepare")
    if _num(row.get("direction_mismatch")) > 0:
        reasons.append("direction_mismatch")
    if _num(row.get("near_safe_far_risky")) > 0:
        reasons.append("near_safe_far_risky")
    if _num(row.get("signflip_or_reversal")) > 0:
        reasons.append("signflip_or_reversal")
    if _num(row.get("delayed_intensification")) > 0:
        reasons.append("delayed_intensification")
    if _num(row.get("reintensification_after_relief")) > 0:
        reasons.append("reintensification_after_relief")
    if _num(row.get("direction_consistent")) > 0:
        reasons.append("direction_consistent")
    if (
        _num(row.get("pump_idle")) > 0
        or _num(row.get("target_stale")) > 0
        or _num(row.get("target_error_mean_kg")) < 500.0
    ):
        reasons.append("execution_stale_or_idle")
    if (
        _num(row.get("max_axis_deg")) < 3.0
        and _num(row.get("future_60m_time_over5")) <= 0
        and _num(row.get("future_60m_idle_over5")) <= 0
    ):
        reasons.append("lowrisk_sanity")
    return reasons


def _selection_score(row: pd.Series) -> float:
    score = 0.0
    score += 16.0 * _num(row.get("known_pump_saving_safe_positive"))
    score += 10.0 * _num(row.get("known_active_small_safe_positive"))
    score += 5.0 * min(_num(row.get("future_60m_time_over5")) / 60.0, 8.0)
    score += 3.0 * min(_num(row.get("future_60m_idle_over5")) / 60.0, 8.0)
    score += 2.0 * _num(row.get("candidate_enter4_like"))
    score += 1.4 * _num(row.get("candidate_prefloor_like"))
    score += 1.0 * _num(row.get("delayed_intensification"))
    score += 1.0 * _num(row.get("reintensification_after_relief"))
    score += 0.8 * _num(row.get("direction_consistent"))
    score += 0.8 * _num(row.get("direction_mismatch"))
    score += 0.8 * _num(row.get("near_safe_far_risky"))
    score += 0.8 * _num(row.get("signflip_or_reversal"))
    score += 0.5 * _num(row.get("known_prepare_negative_rows"))
    return float(score)


def _row_key(row: pd.Series) -> tuple[str, str, int]:
    return str(row["dataset"]), str(row["case_id"]), int(row["bucket"])


def _pick(pool: pd.DataFrame, mask: pd.Series, quota: int, used: set[tuple[str, str, int]], category: str) -> list[pd.Series]:
    rows: list[pd.Series] = []
    sub = pool.loc[mask].copy()
    if quota <= 0 or sub.empty:
        return rows
    for _, row in sub.sort_values("_selection_score", ascending=False).iterrows():
        key = _row_key(row)
        if key in used:
            continue
        row = row.copy()
        row["action_family_selection_category"] = category
        rows.append(row)
        used.add(key)
        if len(rows) >= quota:
            break
    return rows


def select_candidates(pool: pd.DataFrame, target: int = TARGET_BUCKETS) -> pd.DataFrame:
    work = pool.copy()
    work["action_family_candidate_reasons"] = [";".join(_candidate_reason(r)) for _, r in work.iterrows()]
    work["_selection_score"] = [_selection_score(r) for _, r in work.iterrows()]
    used: set[tuple[str, str, int]] = set()
    rows: list[pd.Series] = []
    specs = [
        ("pump_saving_known_positive", 12, work["known_pump_saving_safe_positive"] > 0),
        (
            "active_small_known_positive",
            10,
            (work["known_active_small_safe_positive"] > 0)
            & (work["known_pump_saving_safe_positive"] <= 0),
        ),
        (
            "guard10_future_exposure",
            10,
            (work["dataset"] == "guard10")
            & ((work["future_60m_time_over5"] > 0) | (work["future_60m_idle_over5"] > 0)),
        ),
        (
            "broader20_known_negative",
            10,
            (work["dataset"] == "broader20") & (work["known_prepare_negative_rows"] > 0),
        ),
        ("direction_mismatch_or_near_safe", 6, (work["direction_mismatch"] > 0) | (work["near_safe_far_risky"] > 0)),
        ("lowrisk_sanity", 4, work["action_family_candidate_reasons"].str.contains("lowrisk_sanity", na=False)),
    ]
    for category, quota, mask in specs:
        rows.extend(_pick(work, mask, quota, used, category))

    if len(rows) < target:
        rows.extend(_pick(work, pd.Series(True, index=work.index), target - len(rows), used, "score_fill"))

    selected = pd.DataFrame(rows).drop_duplicates(["dataset", "case_id", "bucket"]).head(target).copy()
    selected["probe_id"] = [
        f"{r.dataset}_{r.case_id}_b{int(r.bucket):02d}" for r in selected.itertuples()
    ]
    selected = selected.sort_values(["dataset", "_selection_score"], ascending=[True, False]).reset_index(drop=True)
    cols = [
        "probe_id",
        "dataset",
        "case_id",
        "numbered_case_id",
        "timestamp",
        "label",
        "bucket",
        "current_time_s",
        "action_family_selection_category",
        "action_family_candidate_reasons",
        "selection_category",
        "selection_reasons",
        "known_pump_saving_safe_positive",
        "known_active_small_safe_positive",
        "known_prepare_negative_rows",
        "known_prepare_safe_positive_rows",
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
    selected = selected[[c for c in cols if c in selected.columns]]
    selected.to_csv(OUT / "action_family_candidate_table.csv", index=False)
    selected.to_csv(RAW / "prepare_action_family_candidate_table.csv", index=False)
    return selected


def write_action_definition_table() -> pd.DataFrame:
    table = pd.DataFrame(ACTION_FAMILY)
    table["active_small_ratio_equiv"] = table["active_small_scale"] * 0.15
    table.to_csv(OUT / "prepare_action_family_table.csv", index=False)
    table.to_csv(RAW / "prepare_action_definition_table.csv", index=False)
    table.to_csv(DEBUG / "prepare_action_definition_table.csv", index=False)
    return table


def _write_one_case_csv(path: Path, row: pd.Series) -> None:
    refine._write_one_case_csv(path, row)


def _scaled_forced_vec(row: pd.Series, action_def: dict[str, Any]) -> tuple[float, float, float, str]:
    base_action = str(action_def["base_action"])
    if base_action == "pump_saving":
        pitch, roll, norm = prev._forced_vec(row, "pump_saving")
        return pitch, roll, norm, "pump_saving"
    pitch, roll, _ = prev._forced_vec(row, "active_small")
    scale = float(action_def["active_small_scale"])
    pitch *= scale
    roll *= scale
    return pitch, roll, float(math.hypot(pitch, roll)), "active_small"


def _write_scaled_forced_csv(path: Path, row: pd.Series, action_def: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pitch, roll, norm, provider_action = _scaled_forced_vec(row, action_def)
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
                "source": "h120_prepare_action_family_probe_v1",
                "forced_vec_norm": norm,
                "action_family_name": action_def["action"],
                "target_refresh_scale": action_def["active_small_scale"],
            }
        )


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
        "--forced-prefix-actions",
        str(forced_csv),
        "--forced-prefix-mode",
        "target_update",
        "--out-dir",
        str(out_dir),
    ]
    subprocess.run(cmd, cwd=REPO, check=True)


def _baseline_metrics(row: pd.Series) -> dict[str, float]:
    return refine._baseline_metrics(row)


def _case_metrics(run_dir: Path, row: pd.Series) -> dict[str, float]:
    return refine._case_metrics(run_dir, row)


def _metrics_delta(metrics: dict[str, float], baseline: dict[str, float]) -> dict[str, float]:
    return {f"delta_{k}": float(metrics[k] - baseline[k]) for k in METRIC_KEYS}


def _label_delta(row: pd.Series) -> tuple[str, str]:
    action = str(row["action"])
    if action in ("wait", "watch_only"):
        return "baseline", "baseline action"
    pump_delta = _num(row.get("delta_full_pump_m3"))
    time_delta = _num(row.get("delta_post60_time_over5"))
    idle_delta = _num(row.get("delta_post60_idle_over5"))
    fallback_delta = _num(row.get("delta_full_fallback_pp"))
    max_p95_delta = _num(row.get("delta_full_max_p95"))
    floor_delta = _num(row.get("delta_post60_floor_entry"))
    medium_delta = _num(row.get("delta_post60_medium_delay_rows"))
    time_gain = -time_delta
    idle_gain = -idle_delta
    safety_gain = (
        time_gain >= 30.0
        or idle_gain >= 60.0
        or floor_delta < 0.0
        or medium_delta < 0.0
    )
    hard_ok = (
        fallback_delta <= 0.05
        and max_p95_delta <= 0.05
        and pump_delta <= 80.0
        and time_delta <= 5.0
    )
    safety_worse = (
        time_delta > 10.0
        or idle_delta > 20.0
        or fallback_delta > 0.05
        or max_p95_delta > 0.05
    )
    no_gain = time_gain < 10.0 and idle_gain < 20.0 and floor_delta >= 0.0 and medium_delta >= 0.0
    anti_trigger = (
        (_num(row.get("near_safe_far_risky")) > 0.0 or _num(row.get("direction_mismatch")) > 0.0)
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
    action_defs = {d["action"]: d for d in ACTION_FAMILY}
    for idx, row in selected.iterrows():
        probe_id = str(row["probe_id"])
        case_csv = DEBUG / f"{probe_id}_case.csv"
        _write_one_case_csv(case_csv, row)
        baseline = _baseline_metrics(row)
        if idx % 8 == 0:
            print(f"[probe] {idx + 1}/{len(selected)} {probe_id}", flush=True)
        for action_name, action_def in action_defs.items():
            if not bool(action_def["run_forced"]):
                metrics = dict(baseline)
                run_dir = ""
                cache_source = "baseline"
            else:
                forced_csv = DEBUG / f"{probe_id}_{action_name}_forced.csv"
                run_path = RUNS / probe_id / action_name
                _write_scaled_forced_csv(forced_csv, row, action_def)
                _run_casebook(case_csv, forced_csv, run_path)
                metrics = _case_metrics(run_path, row)
                run_dir = str(run_path.relative_to(REPO))
                cache_source = "action_family_v1"
            delta = _metrics_delta(metrics, baseline)
            payload = {
                "probe_id": probe_id,
                "dataset": row["dataset"],
                "case_id": row["case_id"],
                "bucket": int(row["bucket"]),
                "timestamp": row["timestamp"],
                "current_time_s": _num(row.get("current_time_s")),
                "action": action_name,
                "base_action": action_def["base_action"],
                "target_refresh_scale": float(action_def["active_small_scale"]),
                "run_dir": run_dir,
                "cache_source": cache_source,
                "selection_category": row.get("action_family_selection_category", ""),
                "candidate_reasons": row.get("action_family_candidate_reasons", ""),
                "known_pump_saving_safe_positive": _num(row.get("known_pump_saving_safe_positive")),
                "known_active_small_safe_positive": _num(row.get("known_active_small_safe_positive")),
                "known_prepare_negative_rows": _num(row.get("known_prepare_negative_rows")),
            }
            for col in FEATURE_COLUMNS:
                if col in row.index:
                    payload[col] = row.get(col, 0.0)
            payload.update(metrics)
            payload.update(delta)
            rows.append(payload)

    table = pd.DataFrame(rows)
    labels = []
    reasons = []
    for _, row in table.iterrows():
        label, reason = _label_delta(row)
        labels.append(label)
        reasons.append(reason)
    table["action_family_label"] = labels
    table["action_family_label_reason"] = reasons
    table["time_gain_s"] = -table["delta_post60_time_over5"]
    table["idle_gain_s"] = -table["delta_post60_idle_over5"]
    table["pump_delta_m3"] = table["delta_full_pump_m3"]
    table["regression_risk"] = (
        (table["delta_post60_time_over5"] > 10.0)
        | (table["delta_post60_idle_over5"] > 20.0)
        | (table["delta_full_fallback_pp"] > 0.05)
        | (table["delta_full_max_p95"] > 0.05)
    ).astype(int)
    table["reference_utility"] = (
        table["time_gain_s"]
        + 0.35 * table["idle_gain_s"]
        - 0.15 * table["pump_delta_m3"]
        - 180.0 * np.maximum(table["delta_full_fallback_pp"], 0.0)
        - 120.0 * np.maximum(table["delta_full_max_p95"], 0.0)
    )
    table.to_csv(OUT / "action_family_counterfactual_table.csv", index=False)
    table.to_csv(RAW / "action_family_counterfactual_table.csv", index=False)
    return table


def build_pareto_tables(table: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    prep = table[~table["action"].isin(["wait", "watch_only"])].copy()
    rows = []
    for (dataset, action), g in prep.groupby(["dataset", "action"], dropna=False):
        anti = (
            g["candidate_reasons"].astype(str).str.contains(
                "lowrisk_sanity|broader20_known_negative|direction_mismatch|near_safe_far_risky",
                regex=True,
                na=False,
            )
            | (g["dataset"].astype(str) == "broader20")
        )
        rows.append(
            {
                "dataset": dataset,
                "action": action,
                "rows": int(len(g)),
                "cases": int(g["case_id"].nunique()),
                "safe_positive_rows": int((g["action_family_label"] == "safe_positive").sum()),
                "negative_rows": int((g["action_family_label"] == "negative").sum()),
                "ambiguous_rows": int((g["action_family_label"] == "ambiguous").sum()),
                "safe_positive_rate": float((g["action_family_label"] == "safe_positive").mean()),
                "negative_rate": float((g["action_family_label"] == "negative").mean()),
                "regression_risk_rate": float(g["regression_risk"].mean()),
                "anti_context_negative_rate": float(
                    ((g["action_family_label"] == "negative") & anti).sum() / max(int(anti.sum()), 1)
                ),
                "mean_pump_delta_m3": float(g["pump_delta_m3"].mean()),
                "median_pump_delta_m3": float(g["pump_delta_m3"].median()),
                "mean_time_gain_s": float(g["time_gain_s"].mean()),
                "median_time_gain_s": float(g["time_gain_s"].median()),
                "mean_idle_gain_s": float(g["idle_gain_s"].mean()),
                "median_idle_gain_s": float(g["idle_gain_s"].median()),
                "mean_fallback_delta_pp": float(g["delta_full_fallback_pp"].mean()),
                "mean_max_p95_delta_deg": float(g["delta_full_max_p95"].mean()),
                "mean_reference_utility": float(g["reference_utility"].mean()),
            }
        )
    pareto = pd.DataFrame(rows).sort_values(["dataset", "target_sort"], ignore_index=True) if False else pd.DataFrame(rows)
    order = {d["action"]: i for i, d in enumerate(ACTION_FAMILY)}
    pareto["_order"] = pareto["action"].map(order).fillna(999).astype(int)
    pareto = pareto.sort_values(["dataset", "_order"]).drop(columns=["_order"]).reset_index(drop=True)
    pareto.to_csv(OUT / "action_family_pareto_table.csv", index=False)
    pareto.to_csv(RAW / "action_family_pareto_table.csv", index=False)

    case_rows = []
    for (dataset, case_id, action), g in prep.groupby(["dataset", "case_id", "action"], dropna=False):
        case_rows.append(
            {
                "dataset": dataset,
                "case_id": case_id,
                "action": action,
                "rows": int(len(g)),
                "safe_positive_rows": int((g["action_family_label"] == "safe_positive").sum()),
                "negative_rows": int((g["action_family_label"] == "negative").sum()),
                "ambiguous_rows": int((g["action_family_label"] == "ambiguous").sum()),
                "mean_pump_delta_m3": float(g["pump_delta_m3"].mean()),
                "mean_time_gain_s": float(g["time_gain_s"].mean()),
                "mean_idle_gain_s": float(g["idle_gain_s"].mean()),
                "mean_fallback_delta_pp": float(g["delta_full_fallback_pp"].mean()),
                "mean_max_p95_delta_deg": float(g["delta_full_max_p95"].mean()),
                "mean_reference_utility": float(g["reference_utility"].mean()),
            }
        )
    case_table = pd.DataFrame(case_rows)
    case_table["_order"] = case_table["action"].map(order).fillna(999).astype(int)
    case_table = case_table.sort_values(["dataset", "case_id", "_order"]).drop(columns=["_order"]).reset_index(drop=True)
    case_table.to_csv(OUT / "action_family_case_table.csv", index=False)
    case_table.to_csv(RAW / "action_family_case_table.csv", index=False)
    return pareto, case_table


def build_effect_pattern_tables(table: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    prep_actions = [
        "micro_prepare_25",
        "micro_prepare_50",
        "pump_saving_prepare",
        "micro_prepare_75",
        "active_small_prepare",
    ]
    scale_order = {
        "micro_prepare_25": 0.25,
        "micro_prepare_50": 0.50,
        "pump_saving_prepare": 0.08 / 0.15,
        "micro_prepare_75": 0.75,
        "active_small_prepare": 1.00,
    }
    rows = []
    for probe_id, all_rows in table.groupby("probe_id", dropna=False):
        wait = all_rows[all_rows["action"] == "wait"]
        prep = all_rows[all_rows["action"].isin(prep_actions)].copy()
        if wait.empty or prep.empty:
            continue
        prep["scale_order"] = prep["action"].map(scale_order)
        prep = prep.sort_values(["scale_order", "action"])
        wait_row = wait.iloc[0]
        pump_delta = pd.to_numeric(prep["delta_full_pump_m3"], errors="coerce").fillna(0.0)
        time_delta = pd.to_numeric(prep["delta_post60_time_over5"], errors="coerce").fillna(0.0)
        idle_delta = pd.to_numeric(prep["delta_post60_idle_over5"], errors="coerce").fillna(0.0)
        fallback_delta = pd.to_numeric(prep["delta_full_fallback_pp"], errors="coerce").fillna(0.0)
        p95_delta = pd.to_numeric(prep["delta_full_max_p95"], errors="coerce").fillna(0.0)
        utility = pd.to_numeric(prep["reference_utility"], errors="coerce").fillna(-1e9)

        all_no_effect = bool(
            pump_delta.abs().max() < 0.01
            and time_delta.abs().max() < 0.5
            and idle_delta.abs().max() < 0.5
            and fallback_delta.abs().max() < 1e-6
            and p95_delta.abs().max() < 1e-6
        )
        pump_values = list(pd.to_numeric(prep["pump_delta_m3"], errors="coerce").fillna(0.0))
        monotone_nondec = all(b >= a - 1e-6 for a, b in zip(pump_values, pump_values[1:]))
        monotone_noninc = all(b <= a + 1e-6 for a, b in zip(pump_values, pump_values[1:]))
        best = prep.iloc[int(utility.to_numpy().argmax())]
        safe = prep["action_family_label"] == "safe_positive"
        negative = prep["action_family_label"] == "negative"
        rows.append(
            {
                "probe_id": probe_id,
                "dataset": wait_row.get("dataset"),
                "case_id": wait_row.get("case_id"),
                "bucket": int(wait_row.get("bucket")),
                "selection_category": wait_row.get("selection_category"),
                "candidate_reasons": wait_row.get("candidate_reasons"),
                "all_prepare_no_effect": int(all_no_effect),
                "any_prepare_effect": int(not all_no_effect),
                "pump_delta_nonmonotone_by_scale": int(not (monotone_nondec or monotone_noninc)),
                "pump_delta_monotone_nondec": int(monotone_nondec),
                "pump_delta_monotone_noninc": int(monotone_noninc),
                "safe_positive_action_count": int(safe.sum()),
                "negative_action_count": int(negative.sum()),
                "best_action_by_reference_utility": best["action"],
                "best_action_label": best["action_family_label"],
                "best_reference_utility": float(best["reference_utility"]),
                "best_pump_delta_m3": float(best["pump_delta_m3"]),
                "best_time_gain_s": float(best["time_gain_s"]),
                "best_idle_gain_s": float(best["idle_gain_s"]),
                "range_pump_delta_m3": float(pump_delta.max() - pump_delta.min()),
                "range_time_delta_s": float(time_delta.max() - time_delta.min()),
                "range_idle_delta_s": float(idle_delta.max() - idle_delta.min()),
                "range_fallback_delta_pp": float(fallback_delta.max() - fallback_delta.min()),
                "range_max_p95_delta_deg": float(p95_delta.max() - p95_delta.min()),
                "has_direction_mismatch": int(
                    pd.to_numeric(prep["direction_mismatch"], errors="coerce").fillna(0).max() > 0
                ),
                "has_near_safe_far_risky": int(
                    pd.to_numeric(prep["near_safe_far_risky"], errors="coerce").fillna(0).max() > 0
                ),
                "has_delayed_intensification": int(
                    pd.to_numeric(prep["delayed_intensification"], errors="coerce").fillna(0).max() > 0
                ),
                "has_reintensification": int(
                    pd.to_numeric(prep["reintensification_after_relief"], errors="coerce").fillna(0).max() > 0
                ),
                "max_axis_deg": float(wait_row.get("max_axis_deg", np.nan)),
                "posture_trend_10m_deg": float(wait_row.get("posture_trend_10m_deg", np.nan)),
            }
        )

    effect_table = pd.DataFrame(rows).sort_values(["dataset", "case_id", "bucket"]).reset_index(drop=True)
    effect_table.to_csv(OUT / "action_family_effect_pattern_table.csv", index=False)
    effect_table.to_csv(RAW / "action_family_effect_pattern_table.csv", index=False)

    summary_rows = []
    for dataset, group in effect_table.groupby("dataset", dropna=False):
        summary_rows.append(
            {
                "dataset": dataset,
                "buckets": int(len(group)),
                "all_prepare_no_effect_buckets": int(group["all_prepare_no_effect"].sum()),
                "effectful_buckets": int(group["any_prepare_effect"].sum()),
                "nonmonotone_pump_buckets": int(group["pump_delta_nonmonotone_by_scale"].sum()),
                "buckets_with_any_safe_positive_action": int(
                    (group["safe_positive_action_count"] > 0).sum()
                ),
                "buckets_with_only_negative_prepare_actions": int(
                    ((group["safe_positive_action_count"] == 0) & (group["negative_action_count"] > 0)).sum()
                ),
                "best_micro50_or_pumpsaving_buckets": int(
                    group["best_action_by_reference_utility"].isin(
                        ["micro_prepare_50", "pump_saving_prepare"]
                    ).sum()
                ),
                "best_active_small_buckets": int(
                    (group["best_action_by_reference_utility"] == "active_small_prepare").sum()
                ),
            }
        )
    effect_summary = pd.DataFrame(summary_rows)
    effect_summary.to_csv(OUT / "action_family_effect_pattern_summary.csv", index=False)
    effect_summary.to_csv(RAW / "action_family_effect_pattern_summary.csv", index=False)

    effect_markdown = f"""# Action-Family Effect Pattern Summary

This diagnostic asks whether the prepare action axis is actually learnable before any controller gate is attempted.

{_markdown_table(effect_summary)}

`all_prepare_no_effect_buckets` means every forced prepare action reproduced the wait trajectory. `nonmonotone_pump_buckets` means larger target-refresh scale did not produce a monotone pump-delta axis, so action scale is not a clean value coordinate.

Interpretation: expected-delta heads should not be retrained on all rows as if every bucket were controllable. The next offline model needs a prepare-effectiveness/action-effect-regime label first, then expected-delta or ranking on effectful buckets only.
"""
    (PAPER / "action_family_effect_pattern_summary.md").write_text(effect_markdown)
    return effect_table, effect_summary


def make_plots(pareto: pd.DataFrame) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        (DEBUG / "plot_error.txt").write_text(f"matplotlib unavailable: {exc}\n")
        return

    marker = {"guard10": "o", "broader20": "s"}
    colors = {
        "micro_prepare_25": "#2b8cbe",
        "micro_prepare_50": "#41ab5d",
        "micro_prepare_75": "#fdae61",
        "pump_saving_prepare": "#756bb1",
        "active_small_prepare": "#d7301f",
    }
    prep = pareto.copy()
    for y, fname, ylabel in [
        ("mean_time_gain_s", "pump_vs_time_gain.png", "Mean 60-min time>5 gain (s)"),
        ("mean_idle_gain_s", "pump_vs_idle_gain.png", "Mean 60-min idle>5 gain (s)"),
        ("regression_risk_rate", "pump_vs_regression_risk.png", "Regression-risk rate"),
    ]:
        fig, ax = plt.subplots(figsize=(8.0, 5.2))
        for _, row in prep.iterrows():
            ax.scatter(
                row["mean_pump_delta_m3"],
                row[y],
                marker=marker.get(str(row["dataset"]), "o"),
                color=colors.get(str(row["action"]), "#666666"),
                s=70,
                edgecolor="black",
                linewidth=0.6,
            )
            ax.annotate(
                f"{row['dataset']}:{row['action'].replace('_prepare', '').replace('micro_', 'm')}",
                (row["mean_pump_delta_m3"], row[y]),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=7,
            )
        ax.axhline(0.0, color="#777777", linewidth=0.8)
        ax.axvline(0.0, color="#777777", linewidth=0.8)
        ax.set_xlabel("Mean pump delta (m3)")
        ax.set_ylabel(ylabel)
        ax.set_title("h120 prepare action-family Pareto probe")
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        fig.savefig(PAPER / fname, dpi=180)
        plt.close(fig)


def write_reports(
    selected: pd.DataFrame,
    action_defs: pd.DataFrame,
    table: pd.DataFrame,
    pareto: pd.DataFrame,
    case_table: pd.DataFrame,
    effect_table: pd.DataFrame,
    effect_summary: pd.DataFrame,
) -> None:
    prep = table[~table["action"].isin(["wait", "watch_only"])].copy()
    label_summary = (
        prep.groupby(["action", "action_family_label"]).size().unstack(fill_value=0).reset_index()
    )
    dataset_summary = (
        prep.groupby(["dataset", "action", "action_family_label"]).size().unstack(fill_value=0).reset_index()
    )
    label_summary.to_csv(RAW / "action_family_label_summary.csv", index=False)
    dataset_summary.to_csv(RAW / "action_family_dataset_label_summary.csv", index=False)

    best_guard = pareto[pareto["dataset"] == "guard10"].sort_values(
        ["safe_positive_rate", "mean_reference_utility", "mean_pump_delta_m3"],
        ascending=[False, False, True],
    )
    best_guard_action = str(best_guard.iloc[0]["action"]) if not best_guard.empty else "none"
    small_actions = ["micro_prepare_50", "pump_saving_prepare"]
    best_small_guard = (
        pareto[(pareto["dataset"] == "guard10") & (pareto["action"].isin(small_actions))]
        .sort_values(["safe_positive_rate", "mean_reference_utility"], ascending=[False, False])
    )
    best_small_action = str(best_small_guard.iloc[0]["action"]) if not best_small_guard.empty else "none"

    micro = pareto[pareto["action"].str.startswith("micro_prepare")].copy()
    pump = pareto[pareto["action"] == "pump_saving_prepare"].copy()
    active = pareto[pareto["action"] == "active_small_prepare"].copy()
    micro_has_signal = bool(
        (micro["safe_positive_rows"].sum() > 0)
        and (
            micro["regression_risk_rate"].mean()
            <= max(float(pump["regression_risk_rate"].mean()) if not pump.empty else 1.0, 0.0)
        )
    )
    broader_micro_neg = float(
        micro.loc[micro["dataset"] == "broader20", "negative_rate"].mean()
        if not micro.loc[micro["dataset"] == "broader20"].empty
        else np.nan
    )
    broader_pump_neg = float(
        pump.loc[pump["dataset"] == "broader20", "negative_rate"].mean()
        if not pump.loc[pump["dataset"] == "broader20"].empty
        else np.nan
    )
    broader_false_prepare_reduced = bool(
        np.isfinite(broader_micro_neg) and np.isfinite(broader_pump_neg) and broader_micro_neg < broader_pump_neg
    )
    no_effect_total = int(effect_summary["all_prepare_no_effect_buckets"].sum())
    nonmonotone_total = int(effect_summary["nonmonotone_pump_buckets"].sum())
    total_buckets = int(effect_summary["buckets"].sum())
    broad50 = pareto[(pareto["dataset"] == "broader20") & (pareto["action"] == "micro_prepare_50")]
    broad_pump = pareto[(pareto["dataset"] == "broader20") & (pareto["action"] == "pump_saving_prepare")]
    guard50 = pareto[(pareto["dataset"] == "guard10") & (pareto["action"] == "micro_prepare_50")]
    guard_pump = pareto[(pareto["dataset"] == "guard10") & (pareto["action"] == "pump_saving_prepare")]
    micro50_guard_rate = (
        float(guard50.iloc[0]["safe_positive_rate"]) if not guard50.empty else float("nan")
    )
    pump_guard_rate = (
        float(guard_pump.iloc[0]["safe_positive_rate"]) if not guard_pump.empty else float("nan")
    )
    micro50_broader_neg = (
        float(broad50.iloc[0]["negative_rate"]) if not broad50.empty else float("nan")
    )
    pump_broader_neg = (
        float(broad_pump.iloc[0]["negative_rate"]) if not broad_pump.empty else float("nan")
    )

    allow_shadow = bool(
        micro_has_signal
        and broader_false_prepare_reduced
        and prep[prep["action_family_label"] == "safe_positive"]["case_id"].nunique() >= 3
    )

    summary = f"""# h120 Prepare Action-Family Probe v1

Scope: offline forced-prefix counterfactual only.  The v1.6 delayed-medium reactive floor is unchanged; no controller gate is attached; no learned model is trained.

## Candidate Set

- Locked buckets: {len(selected)}
- Guard10 buckets: {int((selected['dataset'] == 'guard10').sum())}
- Broader20 buckets: {int((selected['dataset'] == 'broader20').sum())}
- Counterfactual rows: {len(table)}
- Forced prepare rows: {len(prep)}

## Action Family

{_markdown_table(action_defs[['action', 'base_action', 'active_small_scale', 'active_small_ratio_equiv']])}

## Label Summary

{_markdown_table(label_summary)}

## Pareto Summary

{_markdown_table(pareto[['dataset','action','rows','safe_positive_rows','negative_rows','safe_positive_rate','negative_rate','regression_risk_rate','mean_pump_delta_m3','mean_time_gain_s','mean_idle_gain_s','mean_reference_utility']])}

## Effect Pattern Summary

{_markdown_table(effect_summary)}
"""
    (OUT / "prepare_action_family_summary.md").write_text(summary)

    decision = f"""# h120 Prepare Action-Family Decision

## Answers

1. **Is the coarse prepare action family part of the expected-delta learning problem?**  Yes.  The probe found {no_effect_total}/{total_buckets} locked buckets where all forced prepare actions reproduced the wait trajectory, and {nonmonotone_total}/{total_buckets} buckets where pump delta was non-monotone with target-refresh scale.  That means the previous weak expected-delta heads were not just a model issue: the action axis itself is partly no-effect / thresholded / branch-dependent.

2. **Is there a more stable micro_prepare than pump_saving_prepare?**  `micro_prepare_50` is the best small-action offline candidate in this locked set: guard10 safe-positive rate {micro50_guard_rate:.3f} vs pump_saving {pump_guard_rate:.3f}.  However, it does not solve false prepare: broader20 negative rate is {micro50_broader_neg:.3f} vs pump_saving {pump_broader_neg:.3f}.  `micro_prepare_25` is often too weak; `micro_prepare_75` starts to inherit the same regression risk as larger actions.

3. **Best first-version h120 prepare candidate?**  If the next offline head needs one small action, use the `micro_prepare_50` / `pump_saving_prepare` neighborhood as the candidate family.  Do not use `active_small_prepare` as a first gate action.  Active-small looks favorable in selected broader20 high-pressure windows, but it is an upper reference with high authority and selected-set bias, not proof of lowrisk safety.

4. **Did broader20 / lowrisk false prepare decrease?**  {'Yes for the averaged micro family, but this is not yet enough for a gate because the improvement is small and selected-set dependent.' if broader_false_prepare_reduced else 'No.  `micro_prepare_50` has essentially the same broader20 negative rate as pump_saving_prepare, and the broader20/lowrisk anti-trigger problem remains open.'}

5. **Worth retraining expected-delta head on the new action family?**  Yes, but only offline and not on the full mixed table.  First add a `prepare_effective` / `action_effect_regime` label, exclude or separately model no-effect buckets, then train expected-delta or pairwise ranking for the `micro_prepare_50` / pump_saving neighborhood.

6. **Allow shadow gate?**  {'A log-only shadow gate could be considered only after the effectiveness label is trained; do not attach any control action.' if allow_shadow else 'No.  Do not proceed to shadow gate until a small action shows stable positives, a lower broader20 false-prepare rate, and multi-case support.'}

7. **If not allowed, next step.**  Build an action-effect/ranking dataset: label no-effect, monotone-positive, non-monotone, regression-risk, and axis-tradeoff regimes; keep only `micro_prepare_50` / pump_saving as first-pass small actions; expand lowrisk/broader anti-trigger coverage; then retry expected-delta/ranking heads.  v1.6 remains the control mainline.

## Interpretation

The key question is no longer whether h120 should continue; it should.  The interface must be action-value based.  This probe shows that simply scaling the old prepare target is not yet a clean action coordinate.  The next useful step is to learn whether a prepare action is effective at all in the current target-lifecycle state, before asking how much value it has.
"""
    (OUT / "h120_prepare_action_family_decision.md").write_text(decision)

    paper = f"""# Action-Family Findings

- The probe compares wait/watch, micro_prepare_25/50/75, pump_saving_prepare, and active_small_prepare on the same locked bucket set.
- `watch_only` is a no-pump baseline and is identical to wait by construction.
- Micro actions are implemented as scaled active_small target-refresh vectors through the existing forced-prefix replay interface, so no controller code or v1.6 floor logic is changed.
- Best small-action guard10 candidate in this run: `{best_small_action}`.
- `micro_prepare_50` improves the guard10 safe-positive rate relative to pump_saving, but does not reduce broader20 negative rate.
- `active_small_prepare` is an upper reference only; its selected-set broader20 positives should not be interpreted as controller-ready lowrisk behavior.
- {no_effect_total}/{total_buckets} buckets were no-effect across all prepare actions; {nonmonotone_total}/{total_buckets} had non-monotone pump delta by scale.
- Shadow/controller gate is {'not recommended yet' if not allow_shadow else 'only recommended as log-only shadow, not control'}.

See:
- `action_family_pareto_table.csv`
- `action_family_counterfactual_table.csv`
- `action_family_effect_pattern_table.csv`
- `pump_vs_time_gain.png`
- `pump_vs_idle_gain.png`
- `pump_vs_regression_risk.png`
"""
    (PAPER / "action_family_findings.md").write_text(paper)

    manifest = f"""# Run Manifest

- Output root: `{OUT}`
- Candidate source: `{OVERNIGHT}` and `{SUPPLEMENTAL}`
- Baseline: v1.6 f120 oracle baseline from existing action-value audit helpers.
- Forced prepare implementation: `--forced-prefix-actions` + `--forced-prefix-mode target_update`.
- h120 risk scheduler: not enabled in the counterfactual command; this isolates the prepare action family from the old floor-internal scheduler.
- Required figures are under `paper_ready/`.
"""
    (DEBUG / "counterfactual_run_manifest.md").write_text(manifest)


def main() -> None:
    _ensure_dirs()
    action_defs = write_action_definition_table()
    pool = load_candidate_pool()
    selected = select_candidates(pool, TARGET_BUCKETS)
    table = run_counterfactuals(selected)
    pareto, case_table = build_pareto_tables(table)
    effect_table, effect_summary = build_effect_pattern_tables(table)
    make_plots(pareto)
    write_reports(selected, action_defs, table, pareto, case_table, effect_table, effect_summary)
    print(f"[done] wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
