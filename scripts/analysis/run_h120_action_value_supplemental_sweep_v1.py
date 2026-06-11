#!/usr/bin/env python3
"""Supplemental h120 action-value sweep.

This script does not attach a controller gate.  It reuses the overnight v1
counterfactual machinery and fills the remaining observable candidate pool,
prioritizing neighborhoods around buckets that became pump_saving safe-positive.
"""

from __future__ import annotations

import importlib.util
import json
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

SUPPLEMENT = OUT / "supplemental_sweep_v1"
SUP_RAW = SUPPLEMENT / "raw_tables"
SUP_DEBUG = SUPPLEMENT / "debug"
SUP_PAPER = SUPPLEMENT / "paper_ready"
SUP_RUNS = SUPPLEMENT / "counterfactual_runs"

OVERNIGHT_SCRIPT = REPO / "scripts/analysis/run_h120_action_value_overnight_v1.py"


def _import_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


overnight = _import_module(OVERNIGHT_SCRIPT, "h120_action_value_overnight_helpers")
refine = overnight.refine

PREPARE_ACTIONS = overnight.PREPARE_ACTIONS
PUMP_ACTION = overnight.PUMP_ACTION
FEATURE_COLUMNS = overnight.FEATURE_COLUMNS


def ensure_dirs() -> None:
    for path in (SUPPLEMENT, SUP_RAW, SUP_DEBUG, SUP_PAPER, SUP_RUNS):
        path.mkdir(parents=True, exist_ok=True)


def markdown_table(df: pd.DataFrame, max_rows: int | None = None) -> str:
    return overnight.markdown_table(df, max_rows=max_rows)


def append_status(phase: str, status: str, reason: str, files: list[Path]) -> None:
    ensure_dirs()
    record = {
        "phase": phase,
        "status": status,
        "reason": reason,
        "evidence_files": [str(p.relative_to(REPO)) for p in files],
    }
    with (SUP_DEBUG / "supplemental_status.jsonl").open("a") as f:
        f.write(json.dumps(record, ensure_ascii=True) + "\n")


def probe_id(row: pd.Series) -> str:
    return f"{row['dataset']}_{row['case_id']}_b{int(row['bucket']):02d}"


def key_tuple(row: pd.Series | Any) -> tuple[str, str, int]:
    return (str(row.dataset), str(row.case_id), int(row.bucket))


def configure_helpers() -> None:
    refine.OUT = SUPPLEMENT
    refine.RAW = SUP_RAW
    refine.DEBUG = SUP_DEBUG
    refine.PAPER = SUP_PAPER
    refine.RUNS = SUP_RUNS
    refine._ensure_dirs()

    cache_roots = [
        ("supplemental_v1", SUP_RUNS),
        ("overnight_v1", OUT / "counterfactual_runs"),
        ("label_refinement_v1", overnight.SOURCE_REFINEMENT / "counterfactual_runs"),
        ("locked_v1", overnight.SOURCE_LOCKED / "counterfactual_runs"),
        ("audit_v1", overnight.SOURCE_AUDIT / "counterfactual_runs"),
    ]

    def cached_run_dir(probe: str, action_name: str) -> Path | None:
        for _, root in cache_roots:
            path = root / probe / action_name
            if (path / "casebook_summary.csv").exists():
                return path
        return None

    refine._cached_run_dir = cached_run_dir


def load_source_pool() -> pd.DataFrame:
    if not overnight.SOURCE_CANDIDATES.exists():
        raise FileNotFoundError(f"missing source candidates: {overnight.SOURCE_CANDIDATES}")
    pool = pd.read_csv(overnight.SOURCE_CANDIDATES)
    prev_safe = overnight.previous_positive_keys()
    prev_refined = overnight.previous_refined_keys()
    reasons = []
    scores = []
    for _, row in pool.iterrows():
        rs = overnight.reason_list(row, prev_safe)
        reasons.append(";".join(rs))
        scores.append(overnight.candidate_score(row, rs, prev_refined))
    pool["selection_reasons"] = reasons
    pool["_overnight_score"] = scores
    pool["refined_probe_id"] = [probe_id(r) for _, r in pool.iterrows()]
    pool["positive_candidate_flag"] = pool["selection_reasons"].astype(str).str.contains(
        "prior_pump_saving_safe_positive|positive_likelihood|guard10_future_exposure|max_axis_4_5_prefloor|posture_3_5_worsening",
        regex=True,
        na=False,
    ).astype(int)
    pool["anti_candidate_flag"] = pool["selection_reasons"].astype(str).str.contains(
        "broader20_anti_trigger|future_low_exposure_remote|direction_mismatch|near_safe_far_risky_low_posture|lowrisk_sanity",
        regex=True,
        na=False,
    ).astype(int)
    return pool


def selected_keys() -> set[tuple[str, str, int]]:
    selected = pd.read_csv(OUT / "positive_candidate_table.csv")
    return {(str(r.dataset), str(r.case_id), int(r.bucket)) for r in selected.itertuples()}


def positive_neighborhood_keys(labels: pd.DataFrame) -> set[tuple[str, str, int]]:
    pump = labels[
        labels["action"].eq(PUMP_ACTION)
        & labels["constrained_label"].astype(str).eq("safe_positive")
    ].copy()
    keys: set[tuple[str, str, int]] = set()
    for r in pump.itertuples():
        for b in range(max(0, int(r.bucket) - 2), min(11, int(r.bucket) + 2) + 1):
            keys.add((str(r.dataset), str(r.case_id), int(b)))
    return keys


def build_supplemental_candidates() -> pd.DataFrame:
    ensure_dirs()
    pool = load_source_pool()
    used = selected_keys()
    labels = pd.read_csv(OUT / "constrained_label_table.csv")
    neigh = positive_neighborhood_keys(labels)

    rows: list[pd.Series] = []
    seen: set[tuple[str, str, int]] = set()

    remaining = pool[
        ~pool.apply(lambda r: key_tuple(r) in used, axis=1)
        & pool["selection_reasons"].astype(str).str.len().gt(0)
    ].copy()

    def add(sub: pd.DataFrame, category: str, limit: int | None = None) -> None:
        count = 0
        for _, row in sub.sort_values("_overnight_score", ascending=False).iterrows():
            k = (str(row["dataset"]), str(row["case_id"]), int(row["bucket"]))
            if k in seen:
                continue
            r = row.copy()
            r["selection_category"] = category
            rows.append(r)
            seen.add(k)
            count += 1
            if limit is not None and count >= limit:
                break

    add(remaining[remaining.apply(lambda r: key_tuple(r) in neigh, axis=1)], "positive_safe_neighborhood", None)
    add(
        remaining[
            remaining["selection_reasons"].astype(str).str.contains(
                "positive_likelihood|guard10_future_exposure|max_axis_4_5_prefloor|posture_3_5_worsening|delayed_intensification|reintensification_after_relief|near_risk_high_far_relief_stale",
                regex=True,
                na=False,
            )
        ],
        "remaining_positive_likelihood",
        None,
    )
    add(
        remaining[
            remaining["selection_reasons"].astype(str).str.contains(
                "broader20_anti_trigger|future_low_exposure_remote|direction_mismatch|near_safe_far_risky_low_posture|lowrisk_sanity|signflip_or_reversal",
                regex=True,
                na=False,
            )
        ],
        "remaining_anti_trigger",
        None,
    )
    add(remaining, "remaining_score_fill", None)

    out = pd.DataFrame(rows).drop_duplicates(["dataset", "case_id", "bucket"]).reset_index(drop=True)
    if out.empty:
        out = remaining.copy().reset_index(drop=True)
    out["refined_probe_id"] = [probe_id(r) for _, r in out.iterrows()]

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
    out.to_csv(SUPPLEMENT / "supplemental_candidate_table.csv", index=False)
    out.to_csv(SUP_RAW / "supplemental_candidate_table.csv", index=False)

    report = ["# Supplemental Candidate Sweep", ""]
    report.append(f"- source observable candidates: {len(pool)}")
    report.append(f"- overnight selected buckets: {len(used)}")
    report.append(f"- supplemental buckets: {len(out)}")
    report.append(f"- positive-neighborhood target keys: {len(neigh)}")
    report.append("")
    report.append("## By Dataset")
    report.append("")
    by_dataset = out.groupby("dataset").size().reset_index(name="rows")
    report.append(markdown_table(by_dataset))
    report.append("")
    report.append("## By Selection Category")
    report.append("")
    by_cat = out.groupby("selection_category").size().reset_index(name="rows")
    report.append(markdown_table(by_cat))
    (SUP_DEBUG / "supplemental_candidate_report.md").write_text("\n".join(report) + "\n")
    append_status("supplemental_candidates", "completed", f"selected {len(out)} remaining buckets", [SUPPLEMENT / "supplemental_candidate_table.csv"])
    return out


def run_supplemental_counterfactuals(candidates: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    configure_helpers()
    result, deltas = refine.run_counterfactuals(candidates)
    result.to_csv(SUPPLEMENT / "supplemental_counterfactual_table.csv", index=False)
    deltas.to_csv(SUPPLEMENT / "supplemental_delta_table.csv", index=False)
    result.to_csv(SUP_RAW / "supplemental_counterfactual_table.csv", index=False)
    deltas.to_csv(SUP_RAW / "supplemental_delta_table.csv", index=False)
    prep = deltas[deltas["action"].isin(PREPARE_ACTIONS)].copy()
    prep.to_csv(SUPPLEMENT / "supplemental_prepare_delta_table.csv", index=False)
    prep.to_csv(SUP_RAW / "supplemental_prepare_delta_table.csv", index=False)
    append_status(
        "supplemental_counterfactual",
        "completed",
        f"counterfactual rows={len(result)}, prepare rows={len(prep)}",
        [SUPPLEMENT / "supplemental_counterfactual_table.csv", SUPPLEMENT / "supplemental_prepare_delta_table.csv"],
    )
    return result, deltas


def relabel_combined(supp_deltas: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    configure_helpers()
    overnight_deltas = pd.read_csv(RAW / "refined_counterfactual_delta_table.csv")
    combined = pd.concat([overnight_deltas, supp_deltas], ignore_index=True)
    combined = combined.drop_duplicates(["dataset", "case_id", "bucket", "action"], keep="last")
    combined.to_csv(SUPPLEMENT / "combined_delta_table.csv", index=False)
    combined.to_csv(SUP_RAW / "combined_delta_table.csv", index=False)

    labels = refine.apply_constrained_labels(combined)
    labels.to_csv(SUPPLEMENT / "combined_constrained_label_table.csv", index=False)
    labels.to_csv(SUP_RAW / "combined_constrained_label_table.csv", index=False)

    action_sens = overnight.build_action_label_sensitivity(combined)
    action_sens.to_csv(SUPPLEMENT / "combined_action_label_sensitivity_table.csv", index=False)
    action_sens.to_csv(SUP_RAW / "combined_action_label_sensitivity_table.csv", index=False)

    case_table = refine.build_case_level_table(labels)
    case_table.to_csv(SUPPLEMENT / "combined_case_level_value_table.csv", index=False)
    case_table.to_csv(SUP_RAW / "combined_case_level_value_table.csv", index=False)
    return combined, labels, action_sens


def write_analysis(labels: pd.DataFrame, action_sens: pd.DataFrame) -> dict[str, Any]:
    prep = labels[labels["action"].isin(PREPARE_ACTIONS)].copy()
    label_counts = prep.groupby(["action", "constrained_label"]).size().reset_index(name="rows")
    pump = prep[prep["action"].eq(PUMP_ACTION)].copy()
    pump_safe = int(pump["constrained_label"].eq("safe_positive").sum())
    active_safe = int(
        prep[prep["action"].eq("active_small_prepare")]["constrained_label"].eq("safe_positive").sum()
    )
    pump_negative = int(pump["constrained_label"].eq("negative").sum())
    pump_sens = action_sens[
        action_sens["action"].eq(PUMP_ACTION) & action_sens["hard_constraints"].eq(1)
    ].copy()
    min_safe = int(pump_sens["safe_positive"].min()) if not pump_sens.empty else 0
    max_safe = int(pump_sens["safe_positive"].max()) if not pump_sens.empty else 0

    pump_sp = pump[pump["constrained_label"].eq("safe_positive")].copy()
    by_case = (
        pump_sp.groupby(["dataset", "case_id"])
        .agg(
            safe_positive_rows=("constrained_label", "size"),
            mean_time_gain_s=("time_gain_s", "mean"),
            mean_idle_gain_s=("idle_gain_s", "mean"),
            mean_pump_delta_m3=("pump_delta_m3", "mean"),
            buckets=("bucket", lambda s: ",".join(str(int(x)) for x in sorted(s))),
        )
        .reset_index()
        .sort_values("safe_positive_rows", ascending=False)
    )
    by_case.to_csv(SUPPLEMENT / "pump_saving_safe_positive_by_case.csv", index=False)
    by_case.to_csv(SUP_RAW / "pump_saving_safe_positive_by_case.csv", index=False)

    feature_cols = [
        "max_axis_deg",
        "posture_trend_5m_deg",
        "posture_trend_10m_deg",
        "target_age_s",
        "target_error_mean_kg",
        "pump_idle",
        "near_max_norm",
        "near_last_norm",
        "far_max_norm",
        "far_min_norm",
        "far_persistent_high",
        "delayed_intensification",
        "reintensification_after_relief",
        "direction_consistent",
        "direction_mismatch",
        "near_safe_far_risky",
        "near_risky_far_relief",
        "time_gain_s",
        "idle_gain_s",
        "pump_delta_m3",
    ]
    feature_cols = [c for c in feature_cols if c in pump.columns]
    contrast = pump.groupby("constrained_label")[feature_cols].mean(numeric_only=True).reset_index()
    contrast.to_csv(SUPPLEMENT / "pump_saving_feature_contrast.csv", index=False)
    contrast.to_csv(SUP_RAW / "pump_saving_feature_contrast.csv", index=False)

    md = ["# Supplemental Sweep Findings", ""]
    md.append("## Label Counts")
    md.append("")
    md.append(markdown_table(label_counts))
    md.append("")
    md.append("## Pump Saving Key Counts")
    md.append("")
    md.append(f"- pump_saving safe_positive rows: {pump_safe}")
    md.append(f"- pump_saving negative rows: {pump_negative}")
    md.append(f"- active_small safe_positive rows: {active_safe}")
    md.append(f"- pump_saving hard-label sensitivity range: {min_safe} to {max_safe}")
    md.append("")
    md.append("## Pump Saving Safe Positives By Case")
    md.append("")
    md.append(markdown_table(by_case, max_rows=30))
    md.append("")
    md.append("## Feature Contrast")
    md.append("")
    md.append(markdown_table(contrast, max_rows=20))
    md.append("")
    if pump_safe >= 25 and min_safe >= max(5, int(0.25 * max_safe)):
        md.append("## Decision")
        md.append("")
        md.append("The supplemental sweep reaches the minimum pump_saving safe-positive count for an offline head retry.")
    else:
        md.append("## Decision")
        md.append("")
        md.append("The supplemental sweep still does not reach a stable enough pump_saving safe-positive set for head training.")
    findings = SUP_PAPER / "supplemental_sweep_findings.md"
    findings.write_text("\n".join(md) + "\n")

    decision = ["# h120 Action-Value Supplemental Sweep Decision", ""]
    decision.append("## Verdict")
    decision.append("")
    if pump_safe >= 25:
        decision.append("The broader observable pool contains enough pump_saving safe positives to justify a separate offline head retry.")
    else:
        decision.append("Even after filling the remaining observable candidate pool, pump_saving safe positives remain sparse.")
    decision.append("")
    decision.append("## Answers")
    decision.append("")
    decision.append(f"1. Expanded pump_saving safePositive: {pump_safe} rows total.")
    decision.append(f"2. Label sensitivity range: {min_safe}-{max_safe} under hard constraints.")
    decision.append("3. Controller gate remains blocked; this script only performs offline counterfactual analysis.")
    decision.append("4. v1.6 delayed-medium floor remains the mainline.")
    decision.append("5. Next useful direction is either a broader candidate generator beyond the existing 360 bucket pool, or a two-stage head where anti-trigger remains primary and prepare value is trained on richer positive windows.")
    (SUPPLEMENT / "supplemental_decision.md").write_text("\n".join(decision) + "\n")

    append_status(
        "supplemental_analysis",
        "completed",
        f"pump_saving safePositive={pump_safe}, sensitivity={min_safe}-{max_safe}",
        [findings, SUPPLEMENT / "supplemental_decision.md"],
    )
    return {
        "pump_safe": pump_safe,
        "pump_negative": pump_negative,
        "active_safe": active_safe,
        "min_safe": min_safe,
        "max_safe": max_safe,
    }


def write_handoff(info: dict[str, Any]) -> None:
    md = ["# Supplemental Sweep Handoff", ""]
    md.append("- completed phase: supplemental_sweep_v1")
    md.append("- unfinished phase: none")
    md.append("")
    md.append("## Key Results")
    md.append("")
    for k, v in info.items():
        md.append(f"- {k}: {v}")
    md.append("")
    md.append("## Generated Files")
    md.append("")
    for path in [
        SUPPLEMENT / "supplemental_candidate_table.csv",
        SUPPLEMENT / "combined_constrained_label_table.csv",
        SUP_PAPER / "supplemental_sweep_findings.md",
        SUPPLEMENT / "supplemental_decision.md",
    ]:
        md.append(f"- `{path.relative_to(REPO)}`")
    md.append("")
    md.append("## Continue Command")
    md.append("")
    md.append("```bash")
    md.append(".venv312/bin/python scripts/analysis/run_h120_action_value_supplemental_sweep_v1.py")
    md.append("```")
    (SUPPLEMENT / "handoff.md").write_text("\n".join(md) + "\n")


def main() -> None:
    ensure_dirs()
    candidates = build_supplemental_candidates()
    if candidates.empty:
        info = {"pump_safe": "n/a", "reason": "no supplemental candidates"}
        write_handoff(info)
        return
    _, deltas = run_supplemental_counterfactuals(candidates)
    _, labels, action_sens = relabel_combined(deltas)
    info = write_analysis(labels, action_sens)
    write_handoff(info)


if __name__ == "__main__":
    main()
