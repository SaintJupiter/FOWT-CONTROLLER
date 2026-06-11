#!/usr/bin/env python3
"""Summarize a small holdout probe for default-off medium escalation."""
from __future__ import annotations

import argparse
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
    return pd.read_csv(path)


def _case_ts_path(out_dir: Path, case_id: str) -> Path | None:
    matches = sorted((out_dir / "timeseries").glob(f"*{case_id}*_timeseries.csv"))
    return matches[0] if matches else None


def _case_log_path(out_dir: Path, case_id: str) -> Path | None:
    matches = sorted((out_dir / "planner_logs").glob(f"*{case_id}*_planner_log.csv"))
    return matches[0] if matches else None


def _high_pitch_ratio(df: pd.DataFrame, threshold: float = 5.0) -> float:
    if "pitch_deg" not in df.columns or df.empty:
        return float("nan")
    return float(np.mean(np.abs(df["pitch_deg"].to_numpy(dtype=float)) > threshold))


def _safe_mean(df: pd.DataFrame, column: str) -> float:
    if column not in df.columns or df.empty:
        return 0.0
    return float(pd.to_numeric(df[column], errors="coerce").fillna(0.0).mean())


def _safe_sum(df: pd.DataFrame, column: str) -> int:
    if column not in df.columns or df.empty:
        return 0
    return int(pd.to_numeric(df[column], errors="coerce").fillna(0.0).sum())


def _reason_summary(log: pd.DataFrame) -> str:
    if "medium_escalation_reason" not in log.columns or log.empty:
        return ""
    reasons = log["medium_escalation_reason"].fillna("").astype(str)
    counts = reasons.value_counts()
    return "; ".join(f"{reason}:{int(count)}" for reason, count in counts.items())


def _group_label(case_id: str, label: str) -> str:
    text = f"{case_id} {label}".lower()
    if "lowrisk" in text:
        return "lowrisk"
    if "future_relief" in text or "fr_relief" in text:
        return "future_relief"
    if "high" in text or "pressure" in text or "residual" in text:
        return "high_boundary"
    if "signflip" in text or "sf_" in text:
        return "signflip"
    return "other"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", required=True)
    parser.add_argument("--candidate-dir", required=True)
    parser.add_argument(
        "--out-dir",
        default="outputs/wind_prediction/medium_escalation_holdout10",
    )
    parser.add_argument(
        "--case-ids",
        default="",
        help="Optional comma-separated case ids. Defaults to candidate summary order.",
    )
    args = parser.parse_args()

    baseline_dir = _resolve(args.baseline_dir)
    candidate_dir = _resolve(args.candidate_dir)
    out_dir = _resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    base_summary = _read_csv(baseline_dir / "casebook_summary.csv")
    cand_summary = _read_csv(candidate_dir / "casebook_summary.csv")
    if args.case_ids.strip():
        case_ids = [s.strip() for s in args.case_ids.split(",") if s.strip()]
    else:
        case_ids = [str(x) for x in cand_summary["case_id"].tolist()]

    rows: list[dict] = []
    bucket_rows: list[pd.DataFrame] = []
    for case_id in case_ids:
        base_match = base_summary.loc[base_summary["case_id"].astype(str) == case_id]
        cand_match = cand_summary.loc[cand_summary["case_id"].astype(str) == case_id]
        if base_match.empty or cand_match.empty:
            raise ValueError(f"missing case_id={case_id}")
        base_row = base_match.iloc[0]
        cand_row = cand_match.iloc[0]

        base_ts_path = _case_ts_path(baseline_dir, case_id)
        cand_ts_path = _case_ts_path(candidate_dir, case_id)
        cand_log_path = _case_log_path(candidate_dir, case_id)
        base_ts = _read_csv(base_ts_path) if base_ts_path is not None else pd.DataFrame()
        cand_ts = _read_csv(cand_ts_path) if cand_ts_path is not None else pd.DataFrame()
        cand_log = _read_csv(cand_log_path) if cand_log_path is not None else pd.DataFrame()

        base_pump = float(base_row.get("primary_pump_work_m3", np.nan))
        cand_pump = float(cand_row.get("primary_pump_work_m3", np.nan))
        base_pitch_p95 = float(base_row.get("primary_pitch_p95", np.nan))
        cand_pitch_p95 = float(cand_row.get("primary_pitch_p95", np.nan))
        base_fallback = float(base_row.get("primary_safety_fallback_ratio", np.nan))
        cand_fallback = float(cand_row.get("primary_safety_fallback_ratio", np.nan))
        med_bucket_count = _safe_sum(cand_log, "medium_escalation_active")
        med_ratio = _safe_mean(cand_ts, "preview_medium_escalation_active")
        label = str(cand_row.get("label", ""))

        rows.append(
            {
                "case_id": case_id,
                "label": label,
                "case_group": _group_label(case_id, label),
                "baseline_pump_work_m3": base_pump,
                "candidate_pump_work_m3": cand_pump,
                "delta_pump_work_m3": cand_pump - base_pump,
                "baseline_pitch_p95_deg": base_pitch_p95,
                "candidate_pitch_p95_deg": cand_pitch_p95,
                "delta_pitch_p95_deg": cand_pitch_p95 - base_pitch_p95,
                "baseline_high_pitch_ratio": _high_pitch_ratio(base_ts),
                "candidate_high_pitch_ratio": _high_pitch_ratio(cand_ts),
                "delta_high_pitch_ratio": _high_pitch_ratio(cand_ts)
                - _high_pitch_ratio(base_ts),
                "baseline_fallback_ratio": base_fallback,
                "candidate_fallback_ratio": cand_fallback,
                "delta_fallback_ratio": cand_fallback - base_fallback,
                "medium_escalation_ratio": med_ratio,
                "medium_escalation_bucket_count": med_bucket_count,
                "medium_escalation_reasons": _reason_summary(cand_log),
                "triggered": int(med_bucket_count > 0 or med_ratio > 0.0),
            }
        )

        if not cand_log.empty:
            keep_cols = [
                "bucket",
                "current_time_s",
                "history_end",
                "planner_first_action_raw",
                "first_action",
                "current_pitch_deg",
                "current_roll_deg",
                "pressure_block0_norm",
                "pressure_block1_norm",
                "pressure_block2_norm",
                "medium_escalation_enabled",
                "medium_escalation_active",
                "medium_escalation_reason",
                "medium_escalation_candidate_action",
                "medium_escalation_pitch_abs_deg",
                "medium_escalation_roll_abs_deg",
                "medium_escalation_pressure_norm",
                "medium_escalation_future_pressure_norm",
                "medium_escalation_small_action_norm",
                "medium_escalation_medium_action_norm",
                "medium_escalation_small_pitch_residual_deg",
                "medium_escalation_medium_pitch_residual_deg",
                "medium_escalation_pitch_improve_deg",
                "medium_escalation_fallback_dominated",
                "medium_escalation_lowrisk_posture",
                "relief_medium_cap_active",
                "relief_medium_cap_reason",
            ]
            work = cand_log.copy()
            work.insert(0, "case_id", case_id)
            bucket_rows.append(work[[c for c in ["case_id", *keep_cols] if c in work.columns]])

    compare = pd.DataFrame(rows)
    compare.to_csv(out_dir / "medium_escalation_holdout_case_comparison.csv", index=False)
    bucket_log = pd.concat(bucket_rows, ignore_index=True) if bucket_rows else pd.DataFrame()
    bucket_log.to_csv(out_dir / "medium_escalation_holdout_bucket_log.csv", index=False)

    lowrisk = compare.loc[compare["case_group"] == "lowrisk"]
    triggered = compare.loc[compare["triggered"] == 1]
    high_risk = compare.loc[
        (compare["baseline_pitch_p95_deg"] >= 5.0)
        | (compare["baseline_fallback_ratio"] > 0.0)
        | (compare["baseline_high_pitch_ratio"] >= 0.10)
    ]
    pump_increases = compare.loc[compare["delta_pump_work_m3"] > 1.0]
    worsened_fallback = compare.loc[compare["delta_fallback_ratio"] > 1e-9]
    improved_high_risk = high_risk.loc[
        (high_risk["delta_high_pitch_ratio"] < -1e-9)
        | (high_risk["delta_fallback_ratio"] < -1e-9)
        | (high_risk["delta_pitch_p95_deg"] < -1e-9)
    ]

    verdict = (
        "keep_default_off_holdout_probe"
        if len(pump_increases) == 0
        and len(worsened_fallback) == 0
        and int(lowrisk["triggered"].sum()) == 0
        and len(triggered) > 0
        else "freeze_or_review_before_more_testing"
    )

    lines = [
        "# Medium Escalation Holdout Summary",
        "",
        f"- cases: {len(compare)}",
        f"- triggered cases: {len(triggered)} ({', '.join(triggered['case_id'].tolist()) or 'none'})",
        f"- high-risk or fallback baseline cases: {len(high_risk)}",
        f"- high-risk cases with pitch/fallback improvement: {len(improved_high_risk)}",
        f"- lowrisk triggered cases: {int(lowrisk['triggered'].sum())}",
        f"- pump increases > 1 m3: {len(pump_increases)}",
        f"- fallback worsened cases: {len(worsened_fallback)}",
        f"- max pump delta: {compare['delta_pump_work_m3'].max():+.2f} m3",
        f"- total pump delta: {compare['delta_pump_work_m3'].sum():+.2f} m3",
        f"- verdict: {verdict}",
        "",
        "## Triggered Cases",
    ]
    if triggered.empty:
        lines.append("- none")
    else:
        for _, row in triggered.iterrows():
            lines.append(
                "- "
                f"{row['case_id']}: med buckets {int(row['medium_escalation_bucket_count'])}, "
                f"pump {row['delta_pump_work_m3']:+.2f} m3, "
                f"pitch p95 {row['delta_pitch_p95_deg']:+.3f} deg, "
                f"high-pitch ratio {row['delta_high_pitch_ratio']:+.4f}, "
                f"fallback {row['delta_fallback_ratio']:+.4f}"
            )
    lines.extend(["", "## Non-Triggered High-Risk Cases"])
    non_triggered_high = high_risk.loc[high_risk["triggered"] == 0]
    if non_triggered_high.empty:
        lines.append("- none")
    else:
        for _, row in non_triggered_high.iterrows():
            lines.append(
                "- "
                f"{row['case_id']}: baseline pitch p95 {row['baseline_pitch_p95_deg']:.3f}, "
                f"fallback {row['baseline_fallback_ratio']:.4f}, "
                f"reason mix: {row['medium_escalation_reasons']}"
            )
    lines.extend(
        [
            "",
            "## Interpretation",
            "- This validates stability only for a small holdout; it is not mainline evidence.",
            "- No control logic was tuned in this holdout pass.",
            "- Continue only as default-off if the next question is mechanism coverage, not parameter tuning.",
        ]
    )
    (out_dir / "medium_escalation_holdout_summary.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    print(out_dir / "medium_escalation_holdout_summary.md")


if __name__ == "__main__":
    main()
