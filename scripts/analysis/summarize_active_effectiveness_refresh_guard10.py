#!/usr/bin/env python3
"""Summarize the guard10 active_effectiveness_refresh probe."""

from __future__ import annotations

import argparse
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]


def _resolve(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path, low_memory=False)


def _glob_one(root: Path, case_id: str, suffix: str) -> Path | None:
    matches = sorted(root.glob(f"*{case_id}*{suffix}"))
    return matches[0] if matches else None


def _safe_float(value: object, default: float = np.nan) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _maybe_run_metrics(baseline_dir: Path, candidate_dir: Path, out_dir: Path, case_type_csv: str | None) -> Path:
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
    if case_type_csv:
        cmd.extend(["--case-type-csv", case_type_csv])
    subprocess.run(cmd, check=True)
    return eval_dir


def _load_log(path: Path | None) -> pd.DataFrame:
    if path is None or not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, low_memory=False)


def _bucket_log_for_case(case_id: str, baseline_dir: Path, candidate_dir: Path) -> pd.DataFrame:
    base_log = _load_log(_glob_one(baseline_dir / "planner_logs", case_id, "_planner_log.csv"))
    cand_log = _load_log(_glob_one(candidate_dir / "planner_logs", case_id, "_planner_log.csv"))
    if cand_log.empty:
        return pd.DataFrame()

    keep_cols = [
        "bucket",
        "current_time_s",
        "first_action",
        "planner_first_action_raw",
        "prediction_primary_target_refreshed",
        "prediction_primary_target_reused",
        "prediction_primary_target_age_s",
        "active_effectiveness_refresh_enabled",
        "active_effectiveness_refresh_active",
        "active_effectiveness_refresh_reason",
        "active_effectiveness_repeated_active_buckets",
        "active_effectiveness_pitch_abs_deg",
        "active_effectiveness_roll_abs_deg",
        "active_effectiveness_attitude_abs_deg",
        "active_effectiveness_current_response_deg",
        "active_effectiveness_future_relief_status",
        "active_effectiveness_target_err_mean_kg",
        "active_effectiveness_pump_rate_m3_min",
        "active_effectiveness_target_ready",
        "active_effectiveness_pump_idle",
        "active_effectiveness_proposal_delta_mean_kg",
        "active_effectiveness_proposal_delta_m3",
        "active_effectiveness_budget_used_m3",
        "active_effectiveness_lowrisk_safe",
        "active_effectiveness_fallback_dominated",
    ]
    cand = cand_log[[c for c in keep_cols if c in cand_log.columns]].copy()
    cand = cand.rename(columns={c: f"candidate_{c}" for c in cand.columns if c != "bucket"})
    cand.insert(0, "case_id", case_id)

    if base_log.empty:
        return cand

    base_keep = [
        "bucket",
        "first_action",
        "planner_first_action_raw",
        "prediction_primary_target_refreshed",
        "prediction_primary_target_reused",
        "prediction_primary_target_age_s",
    ]
    base = base_log[[c for c in base_keep if c in base_log.columns]].copy()
    base = base.rename(columns={c: f"baseline_{c}" for c in base.columns if c != "bucket"})
    merged = cand.merge(base, on="bucket", how="left")
    return merged


def _write_bucket_log(table: pd.DataFrame, baseline_dir: Path, candidate_dir: Path, out_path: Path) -> None:
    rows = []
    for case_id in table["case_id"].astype(str).tolist():
        b = _bucket_log_for_case(case_id, baseline_dir, candidate_dir)
        if not b.empty:
            rows.append(b)
    if rows:
        pd.concat(rows, ignore_index=True).to_csv(out_path, index=False)
    else:
        pd.DataFrame().to_csv(out_path, index=False)


def _write_summary(table: pd.DataFrame, out_path: Path) -> None:
    total_extra_pump = float(pd.to_numeric(table["extra_pump_m3"], errors="coerce").fillna(0.0).sum())
    fallback_delta_sum = float(pd.to_numeric(table["fallback_ratio_delta"], errors="coerce").fillna(0.0).sum())
    active_weak_sum = int(pd.to_numeric(table["active_but_weak_bucket_reduced"], errors="coerce").fillna(0.0).sum())
    high_hold_sum = int(pd.to_numeric(table["high_hold_bucket_reduced"], errors="coerce").fillna(0.0).sum())
    time3_min = float(pd.to_numeric(table["time_over_3deg_reduced_s"], errors="coerce").fillna(0.0).sum() / 60.0)
    cont3_min = float(pd.to_numeric(table["max_continuous_over_3deg_reduced_s"], errors="coerce").fillna(0.0).sum() / 60.0)
    pump_increase_cases = int((pd.to_numeric(table["extra_pump_m3"], errors="coerce").fillna(0.0) > 1.0).sum())
    fallback_worse_cases = int((pd.to_numeric(table["fallback_ratio_delta"], errors="coerce").fillna(0.0) > 0.0).sum())
    improved_cases = table[
        (pd.to_numeric(table["time_over_3deg_reduced_s"], errors="coerce").fillna(0.0) > 0.0)
        | (pd.to_numeric(table["max_continuous_over_3deg_reduced_s"], errors="coerce").fillna(0.0) > 0.0)
        | (pd.to_numeric(table["active_but_weak_bucket_reduced"], errors="coerce").fillna(0.0) > 0.0)
        | (pd.to_numeric(table["high_hold_bucket_reduced"], errors="coerce").fillna(0.0) > 0.0)
    ]
    lowrisk = table[table["case_type"].isin(["B", "C"])]
    lowrisk_extra = lowrisk[pd.to_numeric(lowrisk["extra_pump_m3"], errors="coerce").fillna(0.0) > 1.0]
    spread_cases = table[
        (pd.to_numeric(table["high_hold_bucket_reduced"], errors="coerce").fillna(0.0) > 0.0)
        | (pd.to_numeric(table["active_but_weak_bucket_reduced"], errors="coerce").fillna(0.0) > 0.0)
        | (pd.to_numeric(table["time_over_3deg_reduced_s"], errors="coerce").fillna(0.0) > 0.0)
    ]

    verdict = "freeze"
    if fallback_delta_sum <= 0.0 and total_extra_pump <= 180.0 and len(improved_cases) >= 2 and len(lowrisk_extra) == 0:
        verdict = "keep_default_off_candidate"

    lines = [
        "# active_effectiveness_refresh summary",
        "",
        f"- cases: {len(table)}",
        f"- cases with meaningful improvement: {len(improved_cases)}",
        f"- spread cases: {', '.join(spread_cases['case_id'].astype(str).tolist()) or 'none'}",
        f"- lowrisk cases with extra pump > 1 m3: {', '.join(lowrisk_extra['case_id'].astype(str).tolist()) or 'none'}",
        f"- total extra pump m3: {total_extra_pump:+.1f}",
        f"- fallback ratio delta sum: {fallback_delta_sum:+.4f}",
        f"- active_but_weak_bucket_reduced total: {active_weak_sum:+d}",
        f"- high_hold_bucket_reduced total: {high_hold_sum:+d}",
        f"- time_over_3deg_reduced total: {time3_min:+.1f} min",
        f"- max_continuous_over_3deg_reduced total: {cont3_min:+.1f} min",
        f"- pump increase cases (>1 m3): {pump_increase_cases}",
        f"- fallback worse cases: {fallback_worse_cases}",
        f"- verdict: {verdict}",
        "",
        "## Requested answers",
        "",
        f"1. active_but_weak_response reduced: {'yes' if active_weak_sum > 0 else 'no'}",
        f"2. time_over_3 / max_continuous_over_3 reduced: {'yes' if time3_min > 0.0 or cont3_min > 0.0 else 'no'}",
        f"3. fallback not rise: {'yes' if fallback_delta_sum <= 0.0 else 'no'}",
        f"4. pump not systematically increase: {'yes' if total_extra_pump <= 180.0 and pump_increase_cases <= 3 else 'no'}",
        f"5. lowrisk not harmed: {'yes' if len(lowrisk_extra) == 0 else 'no'}",
        f"6. benefits spread beyond 09: {'yes' if len(spread_cases[~spread_cases['case_id'].astype(str).str.contains('09', regex=False)]) > 0 else 'no'}",
        f"7. keep default-off candidate vs freeze: {verdict}",
    ]
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", required=True)
    parser.add_argument("--candidate-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--case-type-csv", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    baseline_dir = _resolve(args.baseline_dir)
    candidate_dir = _resolve(args.candidate_dir)
    out_dir = _resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    eval_dir = _maybe_run_metrics(baseline_dir, candidate_dir, out_dir, args.case_type_csv)
    table = _read_csv(eval_dir / "attitude_recovery_metric_table.csv")
    table.to_csv(out_dir / "active_effectiveness_refresh_case_comparison.csv", index=False)
    _write_bucket_log(table, baseline_dir, candidate_dir, out_dir / "active_effectiveness_refresh_bucket_log.csv")
    _write_summary(table, out_dir / "active_effectiveness_refresh_summary.md")
    print(out_dir / "active_effectiveness_refresh_summary.md")


if __name__ == "__main__":
    main()
