#!/usr/bin/env python3
"""Summarize the default-off medium escalation 3-case probe."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
CASE_ORDER = ("fr_relief_09", "lowrisk_clean", "fr_relief_01")


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def _match_case(df: pd.DataFrame, case_key: str) -> pd.DataFrame:
    if "case_id" not in df.columns:
        return df.iloc[0:0]
    mask = df["case_id"].astype(str).str.contains(case_key, regex=False)
    return df.loc[mask].copy()


def _work_m3(df: pd.DataFrame) -> float:
    if "pump_total_rate_m3_min" not in df.columns:
        return float("nan")
    return float(np.trapezoid(np.abs(df["pump_total_rate_m3_min"].to_numpy(dtype=float)), dx=1.0) / 60.0)


def _high_pitch_ratio(df: pd.DataFrame, threshold: float = 5.0) -> float:
    if "pitch_deg" not in df.columns or df.empty:
        return float("nan")
    return float(np.mean(np.abs(df["pitch_deg"].to_numpy(dtype=float)) > float(threshold)))


def _safe_mean(df: pd.DataFrame, column: str) -> float:
    if column not in df.columns or df.empty:
        return 0.0
    return float(pd.to_numeric(df[column], errors="coerce").fillna(0.0).mean())


def _case_log_path(out_dir: Path, case_id: str) -> Path | None:
    matches = sorted((out_dir / "planner_logs").glob(f"*{case_id}*_planner_log.csv"))
    return matches[0] if matches else None


def _case_ts_path(out_dir: Path, case_id: str) -> Path | None:
    matches = sorted((out_dir / "timeseries").glob(f"*{case_id}*_timeseries.csv"))
    return matches[0] if matches else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline-dir",
        required=True,
        help="Casebook output directory without --medium-escalation.",
    )
    parser.add_argument(
        "--candidate-dir",
        required=True,
        help="Casebook output directory with --medium-escalation.",
    )
    parser.add_argument(
        "--out-dir",
        default="outputs/wind_prediction/medium_escalation_3case",
        help="Directory for medium escalation summary artifacts.",
    )
    args = parser.parse_args()

    baseline_dir = Path(args.baseline_dir)
    candidate_dir = Path(args.candidate_dir)
    out_dir = Path(args.out_dir)
    if not baseline_dir.is_absolute():
        baseline_dir = REPO_ROOT / baseline_dir
    if not candidate_dir.is_absolute():
        candidate_dir = REPO_ROOT / candidate_dir
    if not out_dir.is_absolute():
        out_dir = REPO_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    base_summary = _read_csv(baseline_dir / "casebook_summary.csv")
    cand_summary = _read_csv(candidate_dir / "casebook_summary.csv")
    rows: list[dict] = []
    bucket_rows: list[dict] = []

    for case_key in CASE_ORDER:
        base_row_df = _match_case(base_summary, case_key)
        cand_row_df = _match_case(cand_summary, case_key)
        if base_row_df.empty or cand_row_df.empty:
            raise ValueError(f"Missing case in summaries: {case_key}")
        base_row = base_row_df.iloc[0]
        cand_row = cand_row_df.iloc[0]
        base_case_id = str(base_row["case_id"])
        cand_case_id = str(cand_row["case_id"])
        base_ts_path = _case_ts_path(baseline_dir, base_case_id)
        cand_ts_path = _case_ts_path(candidate_dir, cand_case_id)
        base_log_path = _case_log_path(baseline_dir, base_case_id)
        cand_log_path = _case_log_path(candidate_dir, cand_case_id)
        base_ts = _read_csv(base_ts_path) if base_ts_path is not None else pd.DataFrame()
        cand_ts = _read_csv(cand_ts_path) if cand_ts_path is not None else pd.DataFrame()
        base_log = _read_csv(base_log_path) if base_log_path is not None else pd.DataFrame()
        cand_log = _read_csv(cand_log_path) if cand_log_path is not None else pd.DataFrame()

        rows.append(
            {
                "case_key": case_key,
                "baseline_case_id": base_case_id,
                "candidate_case_id": cand_case_id,
                "baseline_primary_pump_work_m3": float(base_row.get("primary_pump_work_m3", _work_m3(base_ts))),
                "candidate_primary_pump_work_m3": float(cand_row.get("primary_pump_work_m3", _work_m3(cand_ts))),
                "delta_pump_work_m3": float(cand_row.get("primary_pump_work_m3", _work_m3(cand_ts)))
                - float(base_row.get("primary_pump_work_m3", _work_m3(base_ts))),
                "baseline_pitch_p95_deg": float(base_row.get("primary_pitch_p95", np.nan)),
                "candidate_pitch_p95_deg": float(cand_row.get("primary_pitch_p95", np.nan)),
                "delta_pitch_p95_deg": float(cand_row.get("primary_pitch_p95", np.nan))
                - float(base_row.get("primary_pitch_p95", np.nan)),
                "baseline_pitch_high_ratio": _high_pitch_ratio(base_ts),
                "candidate_pitch_high_ratio": _high_pitch_ratio(cand_ts),
                "delta_pitch_high_ratio": _high_pitch_ratio(cand_ts) - _high_pitch_ratio(base_ts),
                "baseline_fallback_ratio": float(base_row.get("primary_safety_fallback_ratio", np.nan)),
                "candidate_fallback_ratio": float(cand_row.get("primary_safety_fallback_ratio", np.nan)),
                "delta_fallback_ratio": float(cand_row.get("primary_safety_fallback_ratio", np.nan))
                - float(base_row.get("primary_safety_fallback_ratio", np.nan)),
                "candidate_medium_escalation_ratio": _safe_mean(
                    cand_ts,
                    "preview_medium_escalation_active",
                ),
                "candidate_medium_bucket_count": int(
                    pd.to_numeric(
                        cand_log.get("medium_escalation_active", pd.Series(dtype=float)),
                        errors="coerce",
                    ).fillna(0.0).sum()
                ),
                "candidate_medium_action_ratio": float(
                    np.mean(cand_log.get("first_action", pd.Series(dtype=str)).astype(str) == "active_medium")
                )
                if not cand_log.empty
                else 0.0,
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
            work.insert(0, "case_key", case_key)
            bucket_rows.append(work[[c for c in ["case_key", *keep_cols] if c in work.columns]])

    compare = pd.DataFrame(rows)
    compare.to_csv(out_dir / "medium_escalation_case_comparison.csv", index=False)
    bucket_log = pd.concat(bucket_rows, ignore_index=True) if bucket_rows else pd.DataFrame()
    bucket_log.to_csv(out_dir / "medium_escalation_bucket_log.csv", index=False)

    r09 = compare.loc[compare["case_key"] == "fr_relief_09"].iloc[0]
    low = compare.loc[compare["case_key"] == "lowrisk_clean"].iloc[0]
    fr01 = compare.loc[compare["case_key"] == "fr_relief_01"].iloc[0]
    max_pump_increase = float(compare["delta_pump_work_m3"].max())
    fr09_pump_delta = float(r09["delta_pump_work_m3"])
    lowrisk_extra_pump = float(low["delta_pump_work_m3"])
    fr09_reduced = bool(
        r09["delta_pitch_high_ratio"] < -1e-9 or r09["delta_fallback_ratio"] < -1e-9
    )
    lowrisk_ok = bool(lowrisk_extra_pump <= 1.0 and low["candidate_medium_bucket_count"] == 0)
    fr01_ok = bool(fr01["candidate_medium_bucket_count"] == 0)
    pump_ok = bool(max_pump_increase <= 25.0 and lowrisk_extra_pump <= 1.0)
    keep_default_off = bool(fr09_reduced and lowrisk_ok and fr01_ok and pump_ok)

    lines = [
        "# Medium Escalation Summary",
        "",
        f"1. fr09 high pitch / fallback reduction: {'yes' if fr09_reduced else 'no'} "
        f"(high-pitch delta {r09['delta_pitch_high_ratio']:+.4f}, fallback delta {r09['delta_fallback_ratio']:+.4f}).",
        f"2. lowrisk_clean no extra pumping: {'yes' if lowrisk_ok else 'no'} "
        f"(pump delta {lowrisk_extra_pump:+.2f} m3, medium buckets {int(low['candidate_medium_bucket_count'])}).",
        f"3. fr01 no meaningless medium: {'yes' if fr01_ok else 'no'} "
        f"(medium buckets {int(fr01['candidate_medium_bucket_count'])}).",
        f"4. pump volume acceptable: {'yes' if pump_ok else 'no'} "
        f"(max case pump increase {max_pump_increase:+.2f} m3, fr09 pump delta {fr09_pump_delta:+.2f} m3).",
        f"5. keep as default-off prototype: {'yes' if keep_default_off else 'no'}.",
        "6. cannot enter mainline: yes, this is a three-case default-off prototype only.",
    ]
    (out_dir / "medium_escalation_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(out_dir / "medium_escalation_summary.md")


if __name__ == "__main__":
    main()
