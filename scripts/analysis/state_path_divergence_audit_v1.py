#!/usr/bin/env python3
"""Case-level state-path divergence audit for oracle vs learned closed loops.

This analysis follows the target/tank/pump state across all buckets in the
guard10 runs and marks whether divergence appears before the oracle-vs-learned
key-bucket action differences.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "analysis"))

from key_bucket_failure_split_v1 import _md_table  # noqa: E402
from model_capability_map_v1 import RUNS, _case_from_name, _run_path, _safe_float  # noqa: E402


def _load_case(run_dir: Path, case: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    log_path = next((run_dir / "planner_logs").glob(f"*{case}*_planner_log.csv"), None)
    ts_path = next((run_dir / "timeseries").glob(f"*{case}*_timeseries.csv"), None)
    if log_path is None or ts_path is None:
        return pd.DataFrame(), pd.DataFrame()
    return pd.read_csv(log_path, low_memory=False), pd.read_csv(ts_path, low_memory=False)


def _nearest_ts(ts: pd.DataFrame, t_s: float) -> pd.Series:
    if ts.empty or "t_s" not in ts.columns:
        return pd.Series(dtype=object)
    idx = (pd.to_numeric(ts["t_s"], errors="coerce") - float(t_s)).abs().idxmin()
    return ts.loc[idx]


def _vec(row: pd.Series, cols: list[str], default: float = np.nan) -> np.ndarray:
    return np.array([_safe_float(row.get(c), default) for c in cols], dtype=float)


def _bucket_rows(
    learned_log: pd.DataFrame,
    learned_ts: pd.DataFrame,
    oracle_log: pd.DataFrame,
    oracle_ts: pd.DataFrame,
    case: str,
    key_buckets: set[int],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    buckets = sorted(set(learned_log["bucket"].astype(int)).intersection(set(oracle_log["bucket"].astype(int))))
    for bucket in buckets:
        lrow = learned_log[learned_log["bucket"].astype(int).eq(bucket)].iloc[0]
        orow = oracle_log[oracle_log["bucket"].astype(int).eq(bucket)].iloc[0]
        t_s = _safe_float(lrow.get("current_time_s"), bucket * 600.0)
        lts = _nearest_ts(learned_ts, t_s)
        ots = _nearest_ts(oracle_ts, t_s)
        target_l = _vec(lrow, ["prediction_primary_target_t1_kg", "prediction_primary_target_t2_kg", "prediction_primary_target_t3_kg"])
        target_o = _vec(orow, ["prediction_primary_target_t1_kg", "prediction_primary_target_t2_kg", "prediction_primary_target_t3_kg"])
        tank_l = _vec(lts, ["tank1_kg", "tank2_kg", "tank3_kg"])
        tank_o = _vec(ots, ["tank1_kg", "tank2_kg", "tank3_kg"])
        pump_l = _vec(lts, ["pump_rate1_m3min", "pump_rate2_m3min", "pump_rate3_m3min"], 0.0)
        pump_o = _vec(ots, ["pump_rate1_m3min", "pump_rate2_m3min", "pump_rate3_m3min"], 0.0)
        rec = {
            "case_id": case,
            "bucket": int(bucket),
            "is_key_bucket": int(bucket in key_buckets),
            "learned_action": str(lrow.get("first_action", "")),
            "oracle_action": str(orow.get("first_action", "")),
            "action_diff": int(str(lrow.get("first_action", "")) != str(orow.get("first_action", ""))),
            "learned_raw_action": str(lrow.get("planner_first_action_raw", lrow.get("first_action", ""))),
            "oracle_raw_action": str(orow.get("planner_first_action_raw", orow.get("first_action", ""))),
            "raw_action_diff": int(str(lrow.get("planner_first_action_raw", lrow.get("first_action", ""))) != str(orow.get("planner_first_action_raw", orow.get("first_action", "")))),
            "target_mean_abs_delta_kg": float(np.nanmean(np.abs(target_o - target_l))),
            "tank_mean_abs_delta_kg": float(np.nanmean(np.abs(tank_o - tank_l))),
            "pump_rate_mean_abs_delta_m3_min": float(np.nanmean(np.abs(pump_o - pump_l))),
            "backlog_delta_abs_kg": abs(_safe_float(ots.get("pump_total_backlog_kg"), 0.0) - _safe_float(lts.get("pump_total_backlog_kg"), 0.0)),
            "target_age_delta_abs_s": abs(_safe_float(orow.get("prediction_primary_target_age_s"), 0.0) - _safe_float(lrow.get("prediction_primary_target_age_s"), 0.0)),
            "learned_target_reused": int(_safe_float(lrow.get("prediction_primary_target_reused"), 0.0) > 0.5),
            "oracle_target_reused": int(_safe_float(orow.get("prediction_primary_target_reused"), 0.0) > 0.5),
            "learned_target_refreshed": int(_safe_float(lrow.get("prediction_primary_target_refreshed"), 0.0) > 0.5),
            "oracle_target_refreshed": int(_safe_float(orow.get("prediction_primary_target_refreshed"), 0.0) > 0.5),
            "learned_relief_medium_cap": int(_safe_float(lrow.get("relief_medium_cap_active"), 0.0) > 0.5),
            "oracle_relief_medium_cap": int(_safe_float(orow.get("relief_medium_cap_active"), 0.0) > 0.5),
            "learned_far_hint": int(_safe_float(lrow.get("far_horizon_hint_any"), 0.0) > 0.5),
            "oracle_far_hint": int(_safe_float(orow.get("far_horizon_hint_any"), 0.0) > 0.5),
            "learned_pitch_abs": abs(_safe_float(lrow.get("current_pitch_deg"), 0.0)),
            "oracle_pitch_abs": abs(_safe_float(orow.get("current_pitch_deg"), 0.0)),
        }
        rec["state_diverged"] = int(
            rec["target_mean_abs_delta_kg"] > 500.0
            or rec["tank_mean_abs_delta_kg"] > 500.0
            or rec["pump_rate_mean_abs_delta_m3_min"] > 0.1
            or rec["target_age_delta_abs_s"] >= 600.0
        )
        rows.append(rec)
    return rows


def run_audit(
    learned_run: str,
    out_dir: Path,
    key_buckets_path: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    key_df = pd.read_csv(key_buckets_path, low_memory=False)
    key_df["case_id"] = key_df["case_id"].map(_case_from_name)
    learned_dir = _run_path(RUNS[learned_run])
    oracle_dir = _run_path(RUNS["oracle"])
    rows: list[dict[str, Any]] = []
    for case, sub in key_df.groupby("case_id"):
        learned_log, learned_ts = _load_case(learned_dir, case)
        oracle_log, oracle_ts = _load_case(oracle_dir, case)
        if learned_log.empty or oracle_log.empty:
            continue
        rows.extend(
            _bucket_rows(
                learned_log,
                learned_ts,
                oracle_log,
                oracle_ts,
                case,
                set(int(x) for x in sub["bucket"].tolist()),
            )
        )
    out = pd.DataFrame(rows)
    out.to_csv(out_dir / "state_path_bucket_table.csv", index=False)

    case_rows: list[dict[str, Any]] = []
    for case, sub in out.groupby("case_id"):
        key_sub = sub[sub["is_key_bucket"].eq(1)]
        first_key = int(key_sub["bucket"].min()) if not key_sub.empty else -1
        first_state = sub[sub["state_diverged"].eq(1)]["bucket"]
        first_action = sub[sub["action_diff"].eq(1)]["bucket"]
        first_raw_action = sub[sub["raw_action_diff"].eq(1)]["bucket"]
        case_rows.append(
            {
                "case_id": case,
                "first_key_bucket": first_key,
                "first_state_divergence_bucket": int(first_state.min()) if not first_state.empty else -1,
                "first_action_diff_bucket": int(first_action.min()) if not first_action.empty else -1,
                "first_raw_action_diff_bucket": int(first_raw_action.min()) if not first_raw_action.empty else -1,
                "key_bucket_count": int(len(key_sub)),
                "state_diverged_before_first_key": int((not first_state.empty) and first_key >= 0 and int(first_state.min()) <= first_key),
                "max_target_delta_kg": float(sub["target_mean_abs_delta_kg"].max()),
                "max_tank_delta_kg": float(sub["tank_mean_abs_delta_kg"].max()),
                "max_pump_rate_delta_m3_min": float(sub["pump_rate_mean_abs_delta_m3_min"].max()),
                "max_target_age_delta_s": float(sub["target_age_delta_abs_s"].max()),
            }
        )
    cases = pd.DataFrame(case_rows)
    cases.to_csv(out_dir / "state_path_case_summary.csv", index=False)

    lines = [
        "# State-Path Divergence Audit v1",
        "",
        "Analysis-only. No model training and no controller changes.",
        "",
        "## Summary",
        "",
        f"- cases analyzed: {len(cases)}",
        f"- buckets analyzed: {len(out)}",
        f"- key buckets in scope: {int(out['is_key_bucket'].sum())}",
        f"- cases where state diverged before or at first key bucket: {int(cases['state_diverged_before_first_key'].sum())}/{len(cases)}",
        "",
        "## Case Summary",
        _md_table(cases),
        "",
        "## Interpretation",
        "",
    ]
    if int(cases["state_diverged_before_first_key"].sum()) >= max(1, len(cases) // 2):
        lines.append("- Most cases already have target/tank/pump state divergence before the key action differences. Bucket-local forecast/action attribution is therefore insufficient.")
    else:
        lines.append("- State divergence does not consistently precede key action differences; forecast/controller bucket-local attribution may be enough for some cases.")
    lines.extend(
        [
            "- The next useful intervention should be evaluated as a multi-bucket closed-loop path, not only a single-bucket action-rank flip.",
            "",
            "## Outputs",
            "- `state_path_bucket_table.csv`",
            "- `state_path_case_summary.csv`",
            "- `state_path_divergence_summary.md`",
        ]
    )
    (out_dir / "state_path_divergence_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {out_dir / 'state_path_divergence_summary.md'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--learned-run", default="near_block_eventbalanced_v2")
    parser.add_argument("--key-buckets", default="outputs/wind_prediction/key_bucket_failure_split_v1/key_bucket_failure_split.csv")
    parser.add_argument("--out-dir", default="outputs/wind_prediction/state_path_divergence_audit_v1")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_audit(
        learned_run=str(args.learned_run),
        out_dir=_run_path(args.out_dir),
        key_buckets_path=_run_path(args.key_buckets),
    )


if __name__ == "__main__":
    main()
