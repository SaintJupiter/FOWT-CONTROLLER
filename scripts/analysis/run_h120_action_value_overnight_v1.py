#!/usr/bin/env python3
"""h120 action-value overnight v1.

This runner continues the h120 action-value line without attaching a controller
gate.  It expands the counterfactual sample around pump_saving_prepare positive
candidates, regenerates constrained labels, and retries lightweight offline
heads only if the labels pass the phase gates.
"""

from __future__ import annotations

import importlib.util
import json
import math
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "outputs/wind_prediction/h120_action_value_overnight_v1"
RAW = OUT / "raw_tables"
DEBUG = OUT / "debug"
PAPER = OUT / "paper_ready"
RUNS = OUT / "counterfactual_runs"

SOURCE_AUDIT = REPO / "outputs/wind_prediction/h120_action_value_audit_v1"
SOURCE_HEAD = REPO / "outputs/wind_prediction/h120_action_value_head_training_v1"
SOURCE_REFINEMENT = REPO / "outputs/wind_prediction/h120_action_value_label_refinement_v1"
SOURCE_LOCKED = REPO / "outputs/wind_prediction/h120_action_value_counterfactual_locked_v1"

SOURCE_CANDIDATES = SOURCE_AUDIT / "action_value_candidate_table.csv"
SOURCE_REFINED = SOURCE_REFINEMENT / "refined_candidate_table.csv"
SOURCE_LABELS = SOURCE_REFINEMENT / "constrained_label_table.csv"

LABEL_REFINEMENT_SCRIPT = REPO / "scripts/analysis/run_h120_action_value_label_refinement_v1.py"
HEAD_TRAINING_SCRIPT = REPO / "scripts/analysis/run_h120_action_value_head_training_v1.py"

TARGET_BUCKETS = 220
TARGET_PER_DATASET = TARGET_BUCKETS // 2
PREPARE_ACTIONS = ("pump_saving_prepare", "active_small_prepare")
PUMP_ACTION = "pump_saving_prepare"


def _import_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


refine = _import_module(LABEL_REFINEMENT_SCRIPT, "h120_label_refinement_v1_helpers")
head = _import_module(HEAD_TRAINING_SCRIPT, "h120_head_training_v1_helpers")

FEATURE_COLUMNS = list(refine.FEATURE_COLUMNS)


def ensure_dirs() -> None:
    for path in (OUT, RAW, DEBUG, PAPER, RUNS):
        path.mkdir(parents=True, exist_ok=True)


def num(value: Any, default: float = 0.0) -> float:
    try:
        if pd.isna(value):
            return default
    except TypeError:
        pass
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def nser(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col not in df.columns:
        return pd.Series(default, index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce").fillna(default).astype(float)


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


def append_phase_status(
    phase: str,
    status: str,
    go_no_go: str,
    evidence_files: list[Path],
    next_phase: str,
    reason: str,
) -> None:
    ensure_dirs()
    record = {
        "phase": phase,
        "status": status,
        "go_no_go": go_no_go,
        "evidence_files": [str(p.relative_to(REPO)) for p in evidence_files],
        "next_phase": next_phase,
        "reason": reason,
    }
    with (DEBUG / "phase_status.jsonl").open("a") as f:
        f.write(json.dumps(record, ensure_ascii=True) + "\n")


def write_handoff(
    completed_phase: str,
    unfinished_phase: str,
    key_results: list[str],
    generated_files: list[Path],
    next_steps: list[str],
) -> None:
    ensure_dirs()
    md = ["# h120 Action-Value Overnight Handoff", ""]
    md.append(f"- completed phase: `{completed_phase}`")
    md.append(f"- unfinished phase: `{unfinished_phase}`")
    md.append("")
    md.append("## Key Results")
    md.append("")
    for item in key_results:
        md.append(f"- {item}")
    md.append("")
    md.append("## Generated Files")
    md.append("")
    for path in generated_files:
        md.append(f"- `{path.relative_to(REPO)}`")
    md.append("")
    md.append("## Next Steps")
    md.append("")
    for step in next_steps:
        md.append(f"- {step}")
    md.append("")
    md.append("## Continue Command")
    md.append("")
    md.append("```bash")
    md.append(".venv312/bin/python scripts/analysis/run_h120_action_value_overnight_v1.py")
    md.append("```")
    (OUT / "handoff.md").write_text("\n".join(md) + "\n")


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def phase0_state_check() -> dict[str, Any]:
    ensure_dirs()
    head_metrics = _read_csv(SOURCE_HEAD / "head_metrics_table.csv")
    gate = _read_csv(SOURCE_HEAD / "high_confidence_gate_table.csv")
    labels = _read_csv(SOURCE_LABELS)
    candidates = _read_csv(SOURCE_CANDIDATES)

    prep = labels[labels.get("action", pd.Series(dtype=str)).isin(PREPARE_ACTIONS)].copy()
    label_counts = (
        prep["constrained_label"].value_counts(dropna=False).rename_axis("label").reset_index(name="rows")
        if not prep.empty
        else pd.DataFrame()
    )
    pump = prep[prep["action"].eq(PUMP_ACTION)].copy() if not prep.empty else pd.DataFrame()
    pump_counts = (
        pump["constrained_label"].value_counts(dropna=False).rename_axis("label").reset_index(name="pump_saving_rows")
        if not pump.empty
        else pd.DataFrame()
    )

    anti = head_metrics[head_metrics.get("head", pd.Series(dtype=str)).eq("anti_trigger")].copy()
    pump_head = head_metrics[
        head_metrics.get("head", pd.Series(dtype=str)).eq("pump_saving_safe_positive")
    ].copy()
    anti_best = anti.sort_values("anti_trigger_recall", ascending=False, na_position="last").head(1)
    pump_best = pump_head.sort_values("f1", ascending=False, na_position="last").head(1)

    md = ["# Phase 0 Current State Check", ""]
    md.append("## Scope")
    md.append("")
    md.append("- Continue h120 action-value work; do not rerun old tasks.")
    md.append("- No controller gate, no v1.6 floor modification, no relief suppression.")
    md.append("- v1.6 delayed-medium floor remains the control mainline.")
    md.append("")
    md.append("## Estimated Overnight Timing")
    md.append("")
    timing = pd.DataFrame(
        [
            {"stage": "data/status check", "estimated_time": "10-20 min"},
            {"stage": "candidate mining", "estimated_time": "10-30 min"},
            {"stage": "counterfactual expansion", "estimated_time": "2-6 h"},
            {"stage": "label regeneration", "estimated_time": "10-30 min"},
            {"stage": "offline head retry", "estimated_time": "20-60 min"},
            {"stage": "report/handoff", "estimated_time": "15-30 min"},
        ]
    )
    md.append(markdown_table(timing))
    md.append("")
    md.append("## Existing Label State")
    md.append("")
    md.append(f"- source candidate rows: {len(candidates)}")
    md.append(f"- previous prepare rows: {len(prep)}")
    md.append(markdown_table(label_counts))
    md.append("")
    md.append("## Previous pump_saving Label State")
    md.append("")
    md.append(markdown_table(pump_counts))
    md.append("")
    md.append("## Previous Head Signals")
    md.append("")
    if not anti_best.empty:
        row = anti_best.iloc[0]
        md.append(
            f"- anti-trigger best recall: {row.get('anti_trigger_recall', np.nan):.3f}, "
            f"F1: {row.get('f1', np.nan):.3f}, model: `{row.get('model')}`."
        )
    else:
        md.append("- anti-trigger metrics missing.")
    if not pump_best.empty:
        row = pump_best.iloc[0]
        md.append(
            f"- pump_saving_safe_positive best F1: {row.get('f1', np.nan):.3f}, "
            f"high-confidence precision: {row.get('high_conf_precision', np.nan):.3f}."
        )
    else:
        md.append("- pump_saving_safe_positive metrics missing.")
    md.append("")
    md.append("## Previous Gate Diagnostic")
    md.append("")
    if not gate.empty:
        best = gate.sort_values(["precision", "selected_rows"], ascending=[False, False], na_position="last").head(5)
        md.append(markdown_table(best))
    else:
        md.append("_missing_")
    md.append("")
    md.append("## Phase 0 Decision")
    md.append("")
    md.append("Go to Phase 1. The previous run confirms a useful reject signal, but pump_saving positive labels are too sparse and the previous gate selected no true safe-positive region.")
    path = OUT / "phase0_state_check.md"
    path.write_text("\n".join(md) + "\n")

    append_phase_status(
        "phase0",
        "completed",
        "go",
        [path],
        "phase1",
        "existing anti-trigger signal confirmed; pump_saving positives remain sparse",
    )
    write_handoff(
        "phase0",
        "phase1",
        [
            f"source candidates={len(candidates)}",
            f"previous prepare rows={len(prep)}",
            "previous pump_saving head was not controller-ready",
        ],
        [path],
        ["mine an expanded positive-biased locked candidate set"],
    )
    return {
        "previous_pump_safe_positive": int(
            ((pump.get("constrained_label", pd.Series(dtype=str)) == "safe_positive")).sum()
        ),
        "source_candidates": len(candidates),
    }


def candidate_key(row: pd.Series) -> tuple[str, str, int]:
    return str(row["dataset"]), str(row["case_id"]), int(row["bucket"])


def probe_id(row: pd.Series) -> str:
    raw = f"{row['dataset']}_{row['case_id']}_b{int(row['bucket']):02d}"
    return "".join(ch if ch.isalnum() or ch in ("_", "-", ".") else "_" for ch in raw)


def previous_positive_keys() -> set[tuple[str, str, int]]:
    labels = _read_csv(SOURCE_LABELS)
    if labels.empty:
        return set()
    pump = labels[
        labels["action"].eq(PUMP_ACTION) & labels["constrained_label"].astype(str).eq("safe_positive")
    ]
    return {(str(r.dataset), str(r.case_id), int(r.bucket)) for r in pump.itertuples()}


def previous_refined_keys() -> set[tuple[str, str, int]]:
    refined = _read_csv(SOURCE_REFINED)
    if refined.empty:
        return set()
    return {(str(r.dataset), str(r.case_id), int(r.bucket)) for r in refined.itertuples()}


def reason_list(row: pd.Series, prev_safe: set[tuple[str, str, int]]) -> list[str]:
    rs: list[str] = []
    dataset = str(row["dataset"])
    key = candidate_key(row)
    max_axis = num(row.get("max_axis_deg"))
    trend5 = num(row.get("posture_trend_5m_deg"))
    trend10 = num(row.get("posture_trend_10m_deg"))
    future_time = num(row.get("future_60m_time_over5"))
    future_idle = num(row.get("future_60m_idle_over5"))
    future_floor = num(row.get("future_60m_floor_entry"))
    future_medium = num(row.get("future_60m_medium_delay_rows"))
    future_exposure = future_time > 30 or future_idle > 60 or future_floor > 0 or future_medium > 0
    remote = num(row.get("remote_risk_active")) > 0
    near_high = max(num(row.get("near_b0_norm")), num(row.get("near_b1_norm")), num(row.get("near_b2_norm"))) >= 1.0
    stale_or_idle = (
        num(row.get("target_stale")) > 0
        or num(row.get("pump_idle")) > 0
        or num(row.get("target_error_mean_kg"), 9999.0) < 250.0
    )
    far_shape = (
        num(row.get("delayed_intensification")) > 0
        or num(row.get("reintensification_after_relief")) > 0
        or num(row.get("near_risky_far_relief")) > 0
    )

    if key in prev_safe:
        rs.append("prior_pump_saving_safe_positive")
    if dataset == "guard10" and future_exposure:
        rs.append("guard10_future_exposure")
    if 4.0 <= max_axis < 5.0:
        rs.append("max_axis_4_5_prefloor")
    if 3.0 <= max_axis < 5.0 and (trend5 > 0.02 or trend10 > 0.05):
        rs.append("posture_3_5_worsening")
    if stale_or_idle:
        rs.append("target_or_pump_idle")
    if near_high:
        rs.append("near_pressure_high")
    if num(row.get("delayed_intensification")) > 0:
        rs.append("delayed_intensification")
    if num(row.get("reintensification_after_relief")) > 0:
        rs.append("reintensification_after_relief")
    if remote and num(row.get("direction_consistent")) > 0:
        rs.append("direction_consistent_far_risk")
    if num(row.get("near_risky_far_relief")) > 0 and stale_or_idle:
        rs.append("near_risk_high_far_relief_stale")
    if dataset == "guard10" and future_exposure and (max_axis >= 3.5 or stale_or_idle or near_high or far_shape):
        rs.append("positive_likelihood")

    if dataset == "broader20" and remote and future_time <= 5 and future_idle <= 5:
        rs.append("broader20_anti_trigger")
    if num(row.get("near_safe_far_risky")) > 0 and max_axis < 4.0:
        rs.append("near_safe_far_risky_low_posture")
    if num(row.get("direction_mismatch")) > 0:
        rs.append("direction_mismatch")
    if num(row.get("signflip_or_reversal")) > 0:
        rs.append("signflip_or_reversal")
    if remote and num(row.get("far_persistent_high")) > 0 and not far_shape:
        rs.append("far_persistent_high_alone")
    if not remote and max_axis < 3.0 and future_time <= 0 and future_idle <= 0:
        rs.append("lowrisk_sanity")
    if future_time <= 5 and future_idle <= 5 and future_floor <= 0 and remote:
        rs.append("future_low_exposure_remote")
    return rs


def candidate_score(row: pd.Series, reasons: list[str], prev_refined: set[tuple[str, str, int]]) -> float:
    key = candidate_key(row)
    score = 0.0
    score += 20.0 if "prior_pump_saving_safe_positive" in reasons else 0.0
    score += 4.0 if "positive_likelihood" in reasons else 0.0
    score += 3.0 if "guard10_future_exposure" in reasons else 0.0
    score += 2.2 if "max_axis_4_5_prefloor" in reasons else 0.0
    score += 1.8 if "posture_3_5_worsening" in reasons else 0.0
    score += 1.4 if "target_or_pump_idle" in reasons else 0.0
    score += 1.2 if "near_pressure_high" in reasons else 0.0
    score += 1.2 if "delayed_intensification" in reasons else 0.0
    score += 1.5 if "reintensification_after_relief" in reasons else 0.0
    score += 1.0 if "direction_consistent_far_risk" in reasons else 0.0
    score += 1.0 if "near_risk_high_far_relief_stale" in reasons else 0.0
    score += 1.0 if "broader20_anti_trigger" in reasons else 0.0
    score += 0.8 if "direction_mismatch" in reasons else 0.0
    score += 0.8 if "near_safe_far_risky_low_posture" in reasons else 0.0
    score += 0.6 if "lowrisk_sanity" in reasons else 0.0
    score += min(num(row.get("future_60m_time_over5")) / 120.0, 5.0)
    score += min(num(row.get("future_60m_idle_over5")) / 180.0, 4.0)
    score += 0.3 * len(reasons)
    score += 0.2 if key in prev_refined else 0.0
    return float(score)


def pick_rows(
    pool: pd.DataFrame,
    mask: pd.Series,
    quota: int,
    used: set[tuple[str, str, int]],
    category: str,
) -> list[pd.Series]:
    out: list[pd.Series] = []
    sub = pool.loc[mask].copy()
    if quota <= 0 or sub.empty:
        return out
    for _, row in sub.sort_values("_overnight_score", ascending=False).iterrows():
        key = candidate_key(row)
        if key in used:
            continue
        row = row.copy()
        row["selection_category"] = category
        out.append(row)
        used.add(key)
        if len(out) >= quota:
            break
    return out


def build_positive_candidates() -> pd.DataFrame:
    ensure_dirs()
    if not SOURCE_CANDIDATES.exists():
        raise FileNotFoundError(f"missing source candidates: {SOURCE_CANDIDATES}")
    source = pd.read_csv(SOURCE_CANDIDATES)
    prev_safe = previous_positive_keys()
    prev_refined = previous_refined_keys()

    work = source.copy()
    reasons = []
    scores = []
    for _, row in work.iterrows():
        rs = reason_list(row, prev_safe)
        reasons.append(";".join(rs))
        scores.append(candidate_score(row, rs, prev_refined))
    work["selection_reasons"] = reasons
    work["_overnight_score"] = scores
    eligible = work[work["selection_reasons"].astype(str).str.len() > 0].copy()

    used: set[tuple[str, str, int]] = set()
    rows: list[pd.Series] = []

    if SOURCE_REFINED.exists():
        old = pd.read_csv(SOURCE_REFINED)
        old_keys = {(str(r.dataset), str(r.case_id), int(r.bucket)) for r in old.itertuples()}
        seed = eligible[
            eligible.apply(lambda r: candidate_key(r) in old_keys, axis=1)
        ].copy()
        for _, row in seed.sort_values("_overnight_score", ascending=False).iterrows():
            if candidate_key(row) in used:
                continue
            row = row.copy()
            row["selection_category"] = "seed_label_refinement_v1"
            rows.append(row)
            used.add(candidate_key(row))

    specs = [
        ("prior_pump_saving_safe_positive", 30, eligible["selection_reasons"].str.contains("prior_pump_saving_safe_positive", na=False)),
        ("positive_likelihood", 55, eligible["selection_reasons"].str.contains("positive_likelihood", na=False)),
        ("max_axis_4_5_prefloor", 25, eligible["selection_reasons"].str.contains("max_axis_4_5_prefloor", na=False)),
        ("posture_3_5_worsening", 25, eligible["selection_reasons"].str.contains("posture_3_5_worsening", na=False)),
        ("delayed_or_reintensification", 30, eligible["selection_reasons"].str.contains("delayed_intensification|reintensification_after_relief", regex=True, na=False)),
        ("near_risk_high_far_relief_stale", 12, eligible["selection_reasons"].str.contains("near_risk_high_far_relief_stale", na=False)),
        ("broader20_anti_trigger", 35, eligible["selection_reasons"].str.contains("broader20_anti_trigger", na=False)),
        ("future_low_exposure_remote", 25, eligible["selection_reasons"].str.contains("future_low_exposure_remote", na=False)),
        ("direction_mismatch", 18, eligible["selection_reasons"].str.contains("direction_mismatch", na=False)),
        ("near_safe_far_risky_low_posture", 18, eligible["selection_reasons"].str.contains("near_safe_far_risky_low_posture", na=False)),
        ("signflip_or_reversal", 12, eligible["selection_reasons"].str.contains("signflip_or_reversal", na=False)),
        ("lowrisk_sanity", 18, eligible["selection_reasons"].str.contains("lowrisk_sanity", na=False)),
    ]
    for category, quota, mask in specs:
        rows.extend(pick_rows(eligible, mask, quota, used, category))

    selected = pd.DataFrame(rows).drop_duplicates(["dataset", "case_id", "bucket"])
    for dataset in ("guard10", "broader20"):
        sub = selected[selected["dataset"].eq(dataset)]
        need = TARGET_PER_DATASET - len(sub)
        if need > 0:
            fill = pick_rows(
                eligible,
                eligible["dataset"].eq(dataset),
                need,
                used,
                f"{dataset}_score_fill",
            )
            if fill:
                selected = pd.concat([selected, pd.DataFrame(fill)], ignore_index=True)

    balanced: list[pd.Series] = []
    for dataset in ("guard10", "broader20"):
        sub = selected[selected["dataset"].eq(dataset)].copy()
        balanced.extend(
            r for _, r in sub.sort_values("_overnight_score", ascending=False).head(TARGET_PER_DATASET).iterrows()
        )
    out = pd.DataFrame(balanced).drop_duplicates(["dataset", "case_id", "bucket"]).reset_index(drop=True)
    out["refined_probe_id"] = [probe_id(r) for _, r in out.iterrows()]
    out["positive_candidate_flag"] = out["selection_reasons"].astype(str).str.contains(
        "prior_pump_saving_safe_positive|positive_likelihood|guard10_future_exposure|max_axis_4_5_prefloor|posture_3_5_worsening",
        regex=True,
        na=False,
    ).astype(int)
    out["anti_candidate_flag"] = out["selection_reasons"].astype(str).str.contains(
        "broader20_anti_trigger|future_low_exposure_remote|direction_mismatch|near_safe_far_risky_low_posture|lowrisk_sanity",
        regex=True,
        na=False,
    ).astype(int)

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
        "positive_candidate_flag",
        "anti_candidate_flag",
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
    out = out[[c for c in cols if c in out.columns]]
    out.to_csv(OUT / "positive_candidate_table.csv", index=False)
    out.to_csv(RAW / "positive_candidate_table.csv", index=False)

    summary_rows = []
    for dataset, group in out.groupby("dataset", dropna=False):
        summary_rows.append(
            {
                "dataset": dataset,
                "rows": len(group),
                "cases": group["case_id"].nunique(),
                "positive_candidate_rows": int(group["positive_candidate_flag"].sum()),
                "anti_candidate_rows": int(group["anti_candidate_flag"].sum()),
                "mean_future60_time_over5": float(group["future_60m_time_over5"].mean()),
                "mean_future60_idle_over5": float(group["future_60m_idle_over5"].mean()),
            }
        )
    by_category = out["selection_category"].value_counts().rename_axis("selection_category").reset_index(name="rows")
    by_reason = []
    for reason in [
        "prior_pump_saving_safe_positive",
        "positive_likelihood",
        "guard10_future_exposure",
        "max_axis_4_5_prefloor",
        "posture_3_5_worsening",
        "delayed_intensification",
        "reintensification_after_relief",
        "direction_consistent_far_risk",
        "broader20_anti_trigger",
        "future_low_exposure_remote",
        "direction_mismatch",
        "near_safe_far_risky_low_posture",
        "lowrisk_sanity",
    ]:
        by_reason.append({"reason": reason, "rows": int(out["selection_reasons"].str.contains(reason, na=False).sum())})

    md = ["# Positive Candidate Sampling Report", ""]
    md.append("## Sampling Scope")
    md.append("")
    md.append(f"- source rows: {len(source)}")
    md.append(f"- selected rows: {len(out)}")
    md.append(f"- target buckets: {TARGET_BUCKETS}")
    md.append("- Existing label-refinement candidates are seeded first to preserve prior evidence and cache reuse.")
    md.append("- Extra rows are selected toward guard10 positive-likelihood regions plus broader20/lowrisk anti-trigger controls.")
    md.append("")
    md.append("## Dataset Summary")
    md.append("")
    md.append(markdown_table(pd.DataFrame(summary_rows)))
    md.append("")
    md.append("## Selection Categories")
    md.append("")
    md.append(markdown_table(by_category, max_rows=30))
    md.append("")
    md.append("## Reason Coverage")
    md.append("")
    md.append(markdown_table(pd.DataFrame(by_reason)))
    md.append("")
    potential_positive = int(out["positive_candidate_flag"].sum())
    md.append("## Phase 1 Decision")
    md.append("")
    if potential_positive >= 30:
        md.append(f"Go to Phase 2. Potential-positive rows: {potential_positive}, above the 30-row minimum.")
    else:
        md.append(f"No-go. Potential-positive rows: {potential_positive}, below the 30-row minimum.")
    report = DEBUG / "positive_candidate_sampling_report.md"
    report.write_text("\n".join(md) + "\n")

    go = potential_positive >= 30
    append_phase_status(
        "phase1",
        "completed",
        "go" if go else "no-go",
        [OUT / "positive_candidate_table.csv", report],
        "phase2" if go else "final",
        f"potential-positive candidates={potential_positive}",
    )
    write_handoff(
        "phase1",
        "phase2" if go else "final",
        [
            f"selected buckets={len(out)}",
            f"potential-positive rows={potential_positive}",
            f"anti-candidate rows={int(out['anti_candidate_flag'].sum())}",
        ],
        [OUT / "positive_candidate_table.csv", report],
        ["run fixed wait/watch/pump_saving/active_small counterfactuals"] if go else ["write final no-go due to sample coverage"],
    )
    return out


def configure_refinement_helpers() -> None:
    refine.OUT = OUT
    refine.RAW = RAW
    refine.DEBUG = DEBUG
    refine.PAPER = PAPER
    refine.RUNS = RUNS
    refine._ensure_dirs()

    cache_roots = [
        ("overnight_v1", RUNS),
        ("label_refinement_v1", SOURCE_REFINEMENT / "counterfactual_runs"),
        ("locked_v1", SOURCE_LOCKED / "counterfactual_runs"),
        ("audit_v1", SOURCE_AUDIT / "counterfactual_runs"),
    ]

    def cached_run_dir(probe: str, action_name: str) -> Path | None:
        for _, root in cache_roots:
            path = root / probe / action_name
            if (path / "casebook_summary.csv").exists():
                return path
        return None

    refine._cached_run_dir = cached_run_dir


def run_counterfactual_phase(candidates: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    configure_refinement_helpers()
    result, deltas = refine.run_counterfactuals(candidates)
    result.to_csv(OUT / "positive_counterfactual_table.csv", index=False)
    result.to_csv(RAW / "positive_counterfactual_table.csv", index=False)
    prep = deltas[deltas["action"].isin(PREPARE_ACTIONS)].copy()
    prep.to_csv(OUT / "prepare_action_result_table.csv", index=False)
    prep.to_csv(RAW / "prepare_action_result_table.csv", index=False)

    cache_counts = (
        result["cache_source"].fillna("").replace("", "new_or_baseline").value_counts()
        .rename_axis("cache_source").reset_index(name="rows")
    )
    action_counts = result["action"].value_counts().rename_axis("action").reset_index(name="rows")
    md = ["# Counterfactual Run Manifest", ""]
    md.append("## Scope")
    md.append("")
    md.append(f"- candidate buckets: {len(candidates)}")
    md.append(f"- counterfactual rows: {len(result)}")
    md.append(f"- prepare delta rows: {len(prep)}")
    md.append("- Actions: wait, watch_only, pump_saving_prepare, active_small_prepare.")
    md.append("- No controller gate implemented; forced actions are offline counterfactual probes only.")
    md.append("")
    md.append("## Action Rows")
    md.append("")
    md.append(markdown_table(action_counts))
    md.append("")
    md.append("## Cache / Run Rows")
    md.append("")
    md.append(markdown_table(cache_counts))
    path = DEBUG / "counterfactual_run_manifest.md"
    path.write_text("\n".join(md) + "\n")

    append_phase_status(
        "phase2",
        "completed",
        "pending_phase3_labels",
        [OUT / "positive_counterfactual_table.csv", OUT / "prepare_action_result_table.csv", path],
        "phase3",
        "counterfactual rows generated; constrained labels still needed for go/no-go",
    )
    write_handoff(
        "phase2",
        "phase3",
        [
            f"candidate buckets={len(candidates)}",
            f"prepare rows={len(prep)}",
        ],
        [OUT / "positive_counterfactual_table.csv", OUT / "prepare_action_result_table.csv", path],
        ["regenerate constrained labels and label sensitivity"],
    )
    return result, deltas


def build_action_label_sensitivity(deltas: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    configs = []
    for pump_cap in (40.0, 80.0, 120.0):
        for time_thr in (20.0, 30.0, 60.0):
            for idle_thr in (40.0, 60.0, 120.0):
                for max_tol in (0.0, 0.05, 0.10):
                    configs.append((pump_cap, time_thr, idle_thr, max_tol, True))
    configs.append((80.0, 30.0, 60.0, 0.05, False))
    for pump_cap, time_thr, idle_thr, max_tol, hard in configs:
        labels = []
        for _, row in deltas.iterrows():
            label, _ = refine._label_for_row(
                row,
                pump_cap_m3=pump_cap,
                time_threshold_s=time_thr,
                idle_threshold_s=idle_thr,
                max_p95_tol_deg=max_tol,
                hard_constraints=hard,
            )
            labels.append(label)
        tmp = deltas.copy()
        tmp["label"] = labels
        prep = tmp[tmp["action"].isin(PREPARE_ACTIONS)].copy()
        for action, group in prep.groupby("action", dropna=False):
            rows.append(
                {
                    "action": action,
                    "pump_cap_m3": pump_cap,
                    "time_threshold_s": time_thr,
                    "idle_threshold_s": idle_thr,
                    "max_p95_tol_deg": max_tol,
                    "hard_constraints": int(hard),
                    "prepare_rows": len(group),
                    "safe_positive": int(group["label"].eq("safe_positive").sum()),
                    "negative": int(group["label"].eq("negative").sum()),
                    "ambiguous": int(group["label"].eq("ambiguous").sum()),
                    "broader20_safe_positive": int(
                        (group["dataset"].eq("broader20") & group["label"].eq("safe_positive")).sum()
                    ),
                    "lowrisk_safe_positive": int(
                        (
                            group["selection_reasons"].astype(str).str.contains("lowrisk_sanity", na=False)
                            & group["label"].eq("safe_positive")
                        ).sum()
                    ),
                    "near_safe_far_risky_safe_positive": int(
                        ((group["near_safe_far_risky"] > 0) & group["label"].eq("safe_positive")).sum()
                    ),
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(RAW / "action_label_sensitivity_table.csv", index=False)
    out.to_csv(OUT / "action_label_sensitivity_table.csv", index=False)
    return out


def phase3_labels(deltas: pd.DataFrame, previous_pump_safe: int) -> tuple[pd.DataFrame, bool, dict[str, Any]]:
    configure_refinement_helpers()
    labels = refine.apply_constrained_labels(deltas)
    case_table = refine.build_case_level_table(labels)
    sensitivity = refine.build_label_sensitivity(deltas)
    action_sensitivity = build_action_label_sensitivity(deltas)

    prep = labels[labels["action"].isin(PREPARE_ACTIONS)].copy()
    pump = prep[prep["action"].eq(PUMP_ACTION)].copy()
    pump_safe = int(pump["constrained_label"].eq("safe_positive").sum())
    pump_negative = int(pump["constrained_label"].eq("negative").sum())
    active_safe = int(
        prep[prep["action"].eq("active_small_prepare")]["constrained_label"].eq("safe_positive").sum()
    )
    label_counts = prep.groupby(["action", "constrained_label"]).size().reset_index(name="rows")

    pump_sens = action_sensitivity[
        (action_sensitivity["action"].eq(PUMP_ACTION)) & (action_sensitivity["hard_constraints"].eq(1))
    ].copy()
    min_safe = int(pump_sens["safe_positive"].min()) if not pump_sens.empty else 0
    max_safe = int(pump_sens["safe_positive"].max()) if not pump_sens.empty else 0
    stable_enough = max_safe > 0 and min_safe >= max(5, math.floor(0.25 * max_safe))
    increased = pump_safe > previous_pump_safe
    phase_go = pump_safe >= 25 and stable_enough

    label_counts.to_csv(RAW / "label_distribution_table.csv", index=False)
    sensitivity.to_csv(RAW / "label_sensitivity_table.csv", index=False)
    case_table.to_csv(RAW / "case_level_value_table.csv", index=False)

    md = ["# Label Distribution Summary", ""]
    md.append("## Constrained Labels")
    md.append("")
    md.append(markdown_table(label_counts))
    md.append("")
    md.append("## Key Counts")
    md.append("")
    md.append(f"- previous pump_saving safe_positive rows: {previous_pump_safe}")
    md.append(f"- overnight pump_saving safe_positive rows: {pump_safe}")
    md.append(f"- overnight pump_saving negative rows: {pump_negative}")
    md.append(f"- active_small safePositive rows: {active_safe}")
    md.append(f"- pump_saving hard-label sensitivity range: {min_safe} to {max_safe}")
    md.append("")
    md.append("## Sensitivity Snapshot")
    md.append("")
    md.append(markdown_table(action_sensitivity[action_sensitivity["action"].eq(PUMP_ACTION)].head(20)))
    md.append("")
    md.append("## Phase 3 Decision")
    md.append("")
    if phase_go:
        md.append("Go to Phase 4. pump_saving safe_positive rows reached the target band and are not wholly threshold-fragile.")
    elif not increased:
        md.append("No-go to Phase 4. pump_saving safe_positive rows did not increase over the previous label-refinement run.")
    elif pump_safe < 25:
        md.append("No-go to Phase 4. pump_saving safePositive rows increased but remain below the 25-row minimum.")
    else:
        md.append("No-go to Phase 4. pump_saving labels are too sensitive to constrained-label thresholds.")
    paper = PAPER / "label_distribution_summary.md"
    paper.write_text("\n".join(md) + "\n")

    append_phase_status(
        "phase3",
        "completed",
        "go" if phase_go else "no-go",
        [
            OUT / "constrained_label_table.csv",
            OUT / "label_sensitivity_table.csv",
            OUT / "action_label_sensitivity_table.csv",
            paper,
        ],
        "phase4" if phase_go else "final",
        f"pump_saving safePositive={pump_safe}; previous={previous_pump_safe}; hard sensitivity range={min_safe}-{max_safe}",
    )
    write_handoff(
        "phase3",
        "phase4" if phase_go else "final",
        [
            f"pump_saving safePositive={pump_safe}",
            f"previous pump_saving safePositive={previous_pump_safe}",
            f"active_small safePositive={active_safe}",
            f"pump_saving hard sensitivity range={min_safe}-{max_safe}",
        ],
        [
            OUT / "constrained_label_table.csv",
            OUT / "label_sensitivity_table.csv",
            OUT / "action_label_sensitivity_table.csv",
            paper,
        ],
        ["train offline action-value heads"] if phase_go else ["write final decision; do not train heads from weak labels"],
    )
    return labels, phase_go, {
        "pump_safe": pump_safe,
        "pump_negative": pump_negative,
        "active_safe": active_safe,
        "min_safe": min_safe,
        "max_safe": max_safe,
        "stable_enough": stable_enough,
    }


def configure_head_helpers() -> None:
    head.SOURCE = OUT
    head.OUT = OUT
    head.DEBUG = DEBUG
    head.RAW = RAW
    head.ensure_dirs()


def copy_head_outputs_to_raw() -> None:
    for name in [
        "head_metrics_table.csv",
        "high_confidence_gate_table.csv",
        "feature_importance_table.csv",
        "calibration_bins.csv",
        "split_distribution.csv",
        "training_table.csv",
    ]:
        src = OUT / name
        if src.exists():
            shutil.copyfile(src, RAW / name)


def phase4_train_heads() -> tuple[bool, dict[str, Any]]:
    configure_head_helpers()
    head.main()
    copy_head_outputs_to_raw()

    metrics = _read_csv(OUT / "head_metrics_table.csv")
    gate = _read_csv(OUT / "high_confidence_gate_table.csv")
    pump = metrics[metrics.get("head", pd.Series(dtype=str)).eq("pump_saving_safe_positive")].copy()
    anti = metrics[metrics.get("head", pd.Series(dtype=str)).eq("anti_trigger")].copy()
    best_pump = pump.sort_values("f1", ascending=False, na_position="last").head(1)
    best_anti = anti.sort_values("anti_trigger_recall", ascending=False, na_position="last").head(1)
    viable = pd.DataFrame()
    if not gate.empty:
        viable = gate[
            (gate["selected_rows"] >= 5)
            & (gate["safe_positive_rows"] >= 3)
            & (gate["precision"] >= 0.60)
            & (gate["false_positive_broader_lowrisk"] <= 1)
            & (gate["top_case_share"] <= 0.50)
        ].copy()
    pump_f1 = float(best_pump.iloc[0]["f1"]) if not best_pump.empty and not pd.isna(best_pump.iloc[0]["f1"]) else 0.0
    pump_hcp = (
        float(best_pump.iloc[0]["high_conf_precision"])
        if not best_pump.empty and not pd.isna(best_pump.iloc[0]["high_conf_precision"])
        else np.nan
    )
    phase_go = pump_f1 > 0.0 and not viable.empty

    md = ["# Head Training Findings", ""]
    md.append("## Best pump_saving Head")
    md.append("")
    md.append(markdown_table(best_pump))
    md.append("")
    md.append("## Best anti-trigger Head")
    md.append("")
    md.append(markdown_table(best_anti))
    md.append("")
    md.append("## Best Gate Rows")
    md.append("")
    md.append(markdown_table(gate.head(10) if not gate.empty else gate))
    md.append("")
    md.append("## Phase 4 Decision")
    md.append("")
    if phase_go:
        md.append("Go to Phase 5 shadow gate. A nonzero pump_saving head and a small high-confidence candidate region were found offline.")
    else:
        md.append("No-go to Phase 5. Offline heads still do not meet the combined precision, false-positive, and case-concentration criteria.")
    paper = PAPER / "head_training_findings.md"
    paper.write_text("\n".join(md) + "\n")

    append_phase_status(
        "phase4",
        "completed",
        "go" if phase_go else "no-go",
        [
            OUT / "head_metrics_table.csv",
            OUT / "high_confidence_gate_table.csv",
            OUT / "feature_importance_table.csv",
            OUT / "calibration_bins.csv",
            paper,
        ],
        "phase5" if phase_go else "final",
        f"pump_saving F1={pump_f1:.3f}; viable_gate_rows={len(viable)}",
    )
    write_handoff(
        "phase4",
        "phase5" if phase_go else "final",
        [
            f"pump_saving best F1={pump_f1:.3f}",
            f"pump_saving high-confidence precision={pump_hcp if not pd.isna(pump_hcp) else 'nan'}",
            f"viable gate rows={len(viable)}",
        ],
        [
            OUT / "head_metrics_table.csv",
            OUT / "high_confidence_gate_table.csv",
            paper,
        ],
        ["write shadow recommendations"] if phase_go else ["write final decision; do not enter controller gate"],
    )
    return phase_go, {"pump_f1": pump_f1, "pump_hcp": pump_hcp, "viable_gate_rows": len(viable)}


def phase5_shadow_gate() -> None:
    gate = _read_csv(OUT / "high_confidence_gate_table.csv")
    preds = _read_csv(RAW / "classification_oof_predictions.csv")
    train = _read_csv(OUT / "training_table.csv")
    if gate.empty or preds.empty or train.empty:
        return
    viable = gate[
        (gate["selected_rows"] >= 5)
        & (gate["safe_positive_rows"] >= 3)
        & (gate["precision"] >= 0.60)
        & (gate["false_positive_broader_lowrisk"] <= 1)
        & (gate["top_case_share"] <= 0.50)
    ].copy()
    if viable.empty:
        return
    best = viable.iloc[0]
    pump_rows = train[train["action"].eq(PUMP_ACTION)].copy()
    pred = preds.pivot_table(index=["row_id", "model"], columns="head", values="proba", aggfunc="mean").reset_index()
    anti = pred[pred["model"].eq(best["anti_model"])][["row_id", "anti_trigger"]].rename(columns={"anti_trigger": "anti_trigger_prob"})
    pump = pred[pred["model"].eq(best["pump_model"])][["row_id", "pump_saving_safe_positive"]].rename(columns={"pump_saving_safe_positive": "pump_safe_prob"})
    reg = pred[pred["model"].eq(best["regression_model"])][["row_id", "regression_risk"]].rename(columns={"regression_risk": "regression_risk_prob"})
    merged = pump_rows.merge(anti, on="row_id", how="left").merge(pump, on="row_id", how="left").merge(reg, on="row_id", how="left")
    merged["shadow_recommend_prepare"] = (
        (merged["pump_safe_prob"] >= best["pump_threshold"])
        & (merged["anti_trigger_prob"] <= best["anti_max"])
        & (merged["regression_risk_prob"] <= best["regression_max"])
    ).astype(int)
    merged.to_csv(RAW / "shadow_gate_recommendation_table.csv", index=False)
    selected = merged[merged["shadow_recommend_prepare"].eq(1)]
    safe = selected["constrained_label"].astype(str).eq("safe_positive")
    md = ["# Shadow Gate Summary", ""]
    md.append("This is a shadow-only diagnostic on offline counterfactual rows. It does not change the controller.")
    md.append("")
    md.append(f"- selected rows: {len(selected)}")
    md.append(f"- safe_positive rows: {int(safe.sum())}")
    md.append(f"- precision: {float(safe.mean()) if len(selected) else np.nan:.3f}")
    md.append(f"- unique cases: {selected['group_id'].nunique() if len(selected) else 0}")
    md.append("")
    md.append("## Selected Rows")
    md.append("")
    md.append(markdown_table(selected[["dataset", "case_id", "bucket", "constrained_label", "time_gain_s", "idle_gain_s", "pump_delta_m3"]].head(30)))
    paper = PAPER / "shadow_gate_summary.md"
    paper.write_text("\n".join(md) + "\n")
    append_phase_status(
        "phase5",
        "completed",
        "shadow_only",
        [RAW / "shadow_gate_recommendation_table.csv", paper],
        "final",
        "shadow gate diagnostics written; no controller gate implemented",
    )


def final_decision(label_info: dict[str, Any] | None, head_info: dict[str, Any] | None, stopped_at: str) -> None:
    labels = _read_csv(OUT / "constrained_label_table.csv")
    prep = labels[labels.get("action", pd.Series(dtype=str)).isin(PREPARE_ACTIONS)].copy() if not labels.empty else pd.DataFrame()
    counts = (
        prep.groupby(["action", "constrained_label"]).size().reset_index(name="rows")
        if not prep.empty
        else pd.DataFrame()
    )
    md = ["# h120 Action-Value Overnight Decision", ""]
    md.append("## Verdict")
    md.append("")
    if stopped_at == "phase5":
        md.append("The run reached shadow-gate diagnostics only. No controller gate was implemented; v1.6 remains the control mainline.")
    elif stopped_at == "phase4_no_go":
        md.append("The sample expansion and labels were good enough to retry offline heads, but the head/gate criteria were still not strong enough for a controller gate.")
    elif stopped_at == "phase3_no_go":
        md.append("The counterfactual expansion did not create a stable enough pump_saving safe-positive label set for retraining heads.")
    else:
        md.append("The run stopped before offline head training because an earlier phase gate did not pass.")
    md.append("")
    md.append("## Required Answers")
    md.append("")
    md.append("1. **Was pump_saving_safePositive expanded?**")
    md.append("")
    if label_info:
        md.append(f"pump_saving safePositive rows: {label_info.get('pump_safe')}; active_small safePositive rows: {label_info.get('active_safe')}.")
    else:
        md.append("No constrained label table was produced.")
    md.append("")
    md.append("2. **Are action-value labels stable?**")
    md.append("")
    if label_info:
        md.append(
            f"pump_saving hard-label sensitivity range: {label_info.get('min_safe')}-{label_info.get('max_safe')}. "
            f"Stable enough flag: {label_info.get('stable_enough')}."
        )
    else:
        md.append("Not evaluated.")
    md.append("")
    md.append("3. **Does anti-trigger remain useful?**")
    md.append("")
    if head_info:
        md.append("Anti-trigger was retrained as part of the offline heads; see `head_metrics_table.csv`.")
    else:
        md.append("Previous anti-trigger result remains the best evidence; this run did not pass into head training or did not complete it.")
    md.append("")
    md.append("4. **Does pump_saving_safePositive become learnable?**")
    md.append("")
    if head_info:
        md.append(f"Best pump_saving F1 after expansion: {head_info.get('pump_f1'):.3f}; viable gate rows: {head_info.get('viable_gate_rows')}.")
    else:
        md.append("Not yet. The run did not justify using the expanded labels for a controller-ready head.")
    md.append("")
    md.append("5. **Are broader20 / lowrisk false positives controlled?**")
    md.append("")
    md.append("They are explicitly included as anti-trigger controls in candidate selection and head diagnostics. Controller entry still requires low false-positive evidence under grouped validation.")
    md.append("")
    md.append("6. **Allow default-off h120_action_value_gate?**")
    md.append("")
    md.append("No. This overnight task is offline-only. A real controller gate remains blocked until a high-confidence prepare region is stable and not case-concentrated.")
    md.append("")
    md.append("7. **Next step?**")
    md.append("")
    if stopped_at == "phase4_no_go":
        md.append("Use this expanded set to redesign the positive head/label or mine a second targeted batch around true pump_saving positives; do not tune controller thresholds.")
    elif stopped_at == "phase3_no_go":
        md.append("Continue sample mining or revise the constrained safe-positive definition before model training.")
    else:
        md.append("Review the shadow diagnostics before any default-off closed-loop gate proposal.")
    md.append("")
    md.append("8. **v1.6 mainline?**")
    md.append("")
    md.append("Yes. v1.6 delayed-medium reactive floor remains the main control baseline.")
    md.append("")
    md.append("## Label Counts")
    md.append("")
    md.append(markdown_table(counts))
    path = OUT / "action_value_overnight_decision.md"
    path.write_text("\n".join(md) + "\n")
    write_handoff(
        stopped_at,
        "none",
        [
            f"stopped_at={stopped_at}",
            f"pump_saving safePositive={label_info.get('pump_safe') if label_info else 'n/a'}",
            f"head_info={head_info if head_info else 'n/a'}",
        ],
        [path],
        ["use decision file and raw_tables for the next planning step"],
    )


def write_task_plan() -> None:
    plan = {
        "task": "h120_action_value_overnight_v1",
        "controller_modified": False,
        "v16_floor_modified": False,
        "phases": [
            "phase0_state_check",
            "phase1_positive_candidate_mining",
            "phase2_locked_counterfactual_expansion",
            "phase3_constrained_label_regeneration",
            "phase4_offline_head_retry",
            "phase5_shadow_gate_if_allowed",
        ],
        "output_root": str(OUT.relative_to(REPO)),
    }
    (DEBUG / "task_master_plan.json").write_text(json.dumps(plan, indent=2) + "\n")


def main() -> None:
    ensure_dirs()
    write_task_plan()
    phase0 = phase0_state_check()
    candidates = build_positive_candidates()
    if int(candidates.get("positive_candidate_flag", pd.Series(dtype=int)).sum()) < 30:
        final_decision(None, None, "phase1_no_go")
        return
    _, deltas = run_counterfactual_phase(candidates)
    labels, phase3_go, label_info = phase3_labels(
        deltas, int(phase0.get("previous_pump_safe_positive", 0))
    )
    if not phase3_go:
        final_decision(label_info, None, "phase3_no_go")
        return
    phase4_go, head_info = phase4_train_heads()
    if not phase4_go:
        final_decision(label_info, head_info, "phase4_no_go")
        return
    phase5_shadow_gate()
    final_decision(label_info, head_info, "phase5")


if __name__ == "__main__":
    main()
