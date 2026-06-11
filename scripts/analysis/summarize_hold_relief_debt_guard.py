#!/usr/bin/env python3
"""Summarize the hold_relief_debt guard run with unified recovery metrics."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]


def _resolve(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p


def _glob_one(root: Path, case_id: str, suffix: str) -> Path | None:
    matches = sorted(root.glob(f"*{case_id}*{suffix}"))
    return matches[0] if matches else None


def _maybe_run_metrics(baseline_dir: Path, candidate_dir: Path, out_dir: Path) -> Path:
    eval_dir = out_dir / "_eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "analysis" / "evaluate_attitude_recovery_metrics.py"),
        "--baseline-dir",
        str(baseline_dir),
        "--candidate-dir",
        str(candidate_dir),
        "--out-dir",
        str(eval_dir),
    ]
    subprocess.run(cmd, check=True)
    return eval_dir


def _bucket_log_for_case(case_id: str, baseline_dir: Path, candidate_dir: Path) -> pd.DataFrame:
    cand_path = _glob_one(candidate_dir / "planner_logs", case_id, "_planner_log.csv")
    if cand_path is None:
        return pd.DataFrame()
    cand_log = pd.read_csv(cand_path, low_memory=False)
    base_path = _glob_one(baseline_dir / "planner_logs", case_id, "_planner_log.csv")
    base_log = pd.read_csv(base_path, low_memory=False) if base_path else pd.DataFrame()

    keep = [
        "bucket",
        "current_time_s",
        "first_action",
        "planner_first_action_raw",
        "current_pitch_deg",
        "current_roll_deg",
        "pressure_block0_norm",
        "pressure_block1_norm",
        "pressure_block2_norm",
        "raw_pressure_block0_norm",
        "raw_pressure_block1_norm",
        "raw_pressure_block2_norm",
        "hold_future_posture_evidence_active",
        "hold_future_posture_evidence_reason",
        "hold_relief_debt_enabled",
        "hold_relief_debt_applied",
        "hold_relief_debt_planner_level",
        "hold_relief_debt_level",
        "hold_relief_debt_cost",
        "hold_relief_debt_reason",
        "hold_relief_debt_wait_reason",
        "hold_relief_debt_pending",
        "hold_relief_debt_pending_reason",
        "hold_relief_debt_current_response_deg",
        "hold_relief_debt_lowrisk_safe",
        "hold_relief_debt_fallback_dominated",
        "best_scalar_cost",
        "prediction_primary_target_refreshed",
        "prediction_primary_target_reused",
        "prediction_primary_target_age_s",
    ]
    cand = cand_log[[c for c in keep if c in cand_log.columns]].copy()
    cand = cand.rename(columns={c: f"candidate_{c}" for c in cand.columns if c != "bucket"})
    cand.insert(0, "case_id", case_id)
    if base_log.empty:
        return cand

    base_keep = [
        "bucket",
        "first_action",
        "planner_first_action_raw",
        "hold_future_posture_evidence_active",
        "hold_future_posture_evidence_reason",
        "prediction_primary_target_refreshed",
        "prediction_primary_target_reused",
        "prediction_primary_target_age_s",
    ]
    base = base_log[[c for c in base_keep if c in base_log.columns]].copy()
    base = base.rename(columns={c: f"baseline_{c}" for c in base.columns if c != "bucket"})
    return cand.merge(base, on="bucket", how="left")


def _write_bucket_log(table: pd.DataFrame, baseline_dir: Path, candidate_dir: Path, out_path: Path) -> None:
    rows = []
    for case_id in table["case_id"].astype(str).tolist():
        row = _bucket_log_for_case(case_id, baseline_dir, candidate_dir)
        if not row.empty:
            rows.append(row)
    out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    out.to_csv(out_path, index=False)


def _write_summary(table: pd.DataFrame, bucket_log: pd.DataFrame, out_path: Path) -> None:
    num = lambda col: pd.to_numeric(table.get(col, 0.0), errors="coerce").fillna(0.0)
    extra_pump = float(num("extra_pump_m3").sum())
    fallback_delta = float(num("fallback_ratio_delta").sum())
    high_hold_reduced = int(num("high_hold_bucket_reduced").sum())
    time3_min = float(num("time_over_3deg_reduced_s").sum() / 60.0)
    cont3_min = float(num("max_continuous_over_3deg_reduced_s").sum() / 60.0)
    trigger_cases: list[str] = []
    trigger_buckets = 0
    reason_counts: dict[str, int] = {}
    if not bucket_log.empty and "candidate_hold_relief_debt_applied" in bucket_log.columns:
        applied = pd.to_numeric(
            bucket_log["candidate_hold_relief_debt_applied"],
            errors="coerce",
        ).fillna(0.0) > 0
        trigger_buckets = int(applied.sum())
        trigger_cases = sorted(bucket_log.loc[applied, "case_id"].astype(str).unique().tolist())
        if "candidate_hold_relief_debt_reason" in bucket_log.columns:
            reason_counts = (
                bucket_log.loc[applied, "candidate_hold_relief_debt_reason"]
                .astype(str)
                .value_counts()
                .to_dict()
            )
    lowrisk = table[table["case_type"].isin(["B", "C"])]
    lowrisk_extra = lowrisk[pd.to_numeric(lowrisk["extra_pump_m3"], errors="coerce").fillna(0.0) > 5.0]
    fallback_worse = table[pd.to_numeric(table["fallback_ratio_delta"], errors="coerce").fillna(0.0) > 0.0]
    pump_worse = table[pd.to_numeric(table["extra_pump_m3"], errors="coerce").fillna(0.0) > 50.0]
    improved = table[
        (pd.to_numeric(table["high_hold_bucket_reduced"], errors="coerce").fillna(0.0) > 0.0)
        | (pd.to_numeric(table["time_over_3deg_reduced_s"], errors="coerce").fillna(0.0) >= 300.0)
        | (pd.to_numeric(table["max_continuous_over_3deg_reduced_s"], errors="coerce").fillna(0.0) >= 300.0)
    ]

    verdict = "freeze"
    if (
        trigger_buckets > 0
        and len(improved) >= 2
        and fallback_delta <= 0.0
        and extra_pump <= 180.0
        and len(lowrisk_extra) == 0
    ):
        verdict = "keep_default_off_candidate"

    lines = [
        "# hold_relief_debt summary",
        "",
        f"- cases: {len(table)}",
        f"- debt trigger buckets: {trigger_buckets}",
        f"- debt trigger cases: {', '.join(trigger_cases) or 'none'}",
        f"- debt trigger reasons: {reason_counts or {}}",
        f"- total extra pump m3: {extra_pump:+.1f}",
        f"- fallback ratio delta sum: {fallback_delta:+.4f}",
        f"- high_hold_bucket_reduced total: {high_hold_reduced:+d}",
        f"- time_over_3deg_reduced total: {time3_min:+.1f} min",
        f"- max_continuous_over_3deg_reduced total: {cont3_min:+.1f} min",
        f"- lowrisk cases with extra pump >5 m3: {', '.join(lowrisk_extra['case_id'].astype(str).tolist()) or 'none'}",
        f"- fallback worse cases: {', '.join(fallback_worse['case_id'].astype(str).tolist()) or 'none'}",
        f"- pump +50m3 cases: {', '.join(pump_worse['case_id'].astype(str).tolist()) or 'none'}",
        f"- meaningful improvement cases: {', '.join(improved['case_id'].astype(str).tolist()) or 'none'}",
        f"- verdict: {verdict}",
        "",
        "## Requested answers",
        "",
        f"1. high_hold_bucket_count reduced: {'yes' if high_hold_reduced > 0 else 'no'}",
        f"2. max_continuous_over_3 / time_over_3 reduced: {'yes' if cont3_min > 0.0 or time3_min > 0.0 else 'no'}",
        f"3. fallback not rise: {'yes' if fallback_delta <= 0.0 and len(fallback_worse) == 0 else 'no'}",
        f"4. pump not systematically increase: {'yes' if extra_pump <= 180.0 and len(pump_worse) <= 2 else 'no'}",
        f"5. lowrisk safe not harmed: {'yes' if len(lowrisk_extra) == 0 else 'no'}",
        f"6. benefit spread beyond 09: {'yes' if len(improved[~improved['case_id'].astype(str).str.contains('09', regex=False)]) > 0 else 'no'}",
        f"7. keep default-off or freeze: {verdict}",
    ]
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", required=True)
    parser.add_argument("--candidate-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    baseline_dir = _resolve(args.baseline_dir)
    candidate_dir = _resolve(args.candidate_dir)
    out_dir = _resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    eval_dir = _maybe_run_metrics(baseline_dir, candidate_dir, out_dir)
    table = pd.read_csv(eval_dir / "attitude_recovery_metric_table.csv", low_memory=False)
    table.to_csv(out_dir / "hold_relief_debt_case_comparison.csv", index=False)
    bucket_path = out_dir / "hold_relief_debt_bucket_log.csv"
    _write_bucket_log(table, baseline_dir, candidate_dir, bucket_path)
    bucket_log = pd.read_csv(bucket_path, low_memory=False) if bucket_path.exists() else pd.DataFrame()
    _write_summary(table, bucket_log, out_dir / "hold_relief_debt_summary.md")
    print(out_dir / "hold_relief_debt_summary.md")


if __name__ == "__main__":
    main()
