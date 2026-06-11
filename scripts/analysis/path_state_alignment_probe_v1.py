#!/usr/bin/env python3
"""Path-state alignment probe for learned vs oracle prediction-primary runs.

Analysis only. No model training and no controller mutation.

There is no existing forced-prefix-action hook in the casebook runner, so the
``prefix intervention`` table is a diagnostic proxy built from the two existing
closed-loop paths: learned prefix vs oracle prefix. It reports whether the
oracle-like prefix is associated with posture-state movement toward oracle at
the later key bucket, without claiming a new closed-loop counterfactual was
executed.
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


def _action_severity(action: str) -> int:
    return {
        "hold": 0,
        "pump_saving": 1,
        "active_small": 2,
        "active_medium": 3,
        "active_reverse_small": 2,
    }.get(str(action), 0)


def _vec(row: pd.Series, cols: list[str], default: float = np.nan) -> np.ndarray:
    return np.array([_safe_float(row.get(c), default) for c in cols], dtype=float)


def _planner_row(log: pd.DataFrame, bucket: int) -> pd.Series:
    hit = log[log["bucket"].astype(int).eq(bucket)]
    return hit.iloc[0] if not hit.empty else pd.Series(dtype=object)


def _posture_from_row(row: pd.Series) -> np.ndarray:
    return np.array(
        [
            _safe_float(row.get("current_pitch_deg"), 0.0),
            _safe_float(row.get("current_roll_deg"), 0.0),
        ],
        dtype=float,
    )


def _bucket_state_delta(
    lrow: pd.Series,
    orow: pd.Series,
    lts: pd.Series,
    ots: pd.Series,
) -> dict[str, float]:
    target_l = _vec(lrow, ["prediction_primary_target_t1_kg", "prediction_primary_target_t2_kg", "prediction_primary_target_t3_kg"])
    target_o = _vec(orow, ["prediction_primary_target_t1_kg", "prediction_primary_target_t2_kg", "prediction_primary_target_t3_kg"])
    tank_l = _vec(lts, ["tank1_kg", "tank2_kg", "tank3_kg"])
    tank_o = _vec(ots, ["tank1_kg", "tank2_kg", "tank3_kg"])
    pump_l = _vec(lts, ["pump_rate1_m3min", "pump_rate2_m3min", "pump_rate3_m3min"], 0.0)
    pump_o = _vec(ots, ["pump_rate1_m3min", "pump_rate2_m3min", "pump_rate3_m3min"], 0.0)
    posture_l = _posture_from_row(lrow)
    posture_o = _posture_from_row(orow)
    return {
        "posture_delta_norm_deg": float(np.linalg.norm(posture_o - posture_l)),
        "pitch_delta_oracle_minus_learned_deg": float(posture_o[0] - posture_l[0]),
        "roll_delta_oracle_minus_learned_deg": float(posture_o[1] - posture_l[1]),
        "target_mean_abs_delta_kg": float(np.nanmean(np.abs(target_o - target_l))),
        "tank_mean_abs_delta_kg": float(np.nanmean(np.abs(tank_o - tank_l))),
        "pump_rate_mean_abs_delta_m3_min": float(np.nanmean(np.abs(pump_o - pump_l))),
        "target_age_delta_abs_s": abs(_safe_float(orow.get("prediction_primary_target_age_s"), 0.0) - _safe_float(lrow.get("prediction_primary_target_age_s"), 0.0)),
    }


def _divergence_flags(delta: dict[str, float]) -> dict[str, int]:
    return {
        "platform_posture_diverged": int(delta["posture_delta_norm_deg"] > 0.25),
        "target_diverged": int(delta["target_mean_abs_delta_kg"] > 500.0),
        "tank_diverged": int(delta["tank_mean_abs_delta_kg"] > 500.0),
        "pump_diverged": int(delta["pump_rate_mean_abs_delta_m3_min"] > 0.1),
    }


def _prefix_window(first_key: int, width: int = 5) -> range:
    start = max(0, int(first_key) - width)
    return range(start, int(first_key))


def _seq(log: pd.DataFrame, buckets: list[int], col: str = "first_action") -> str:
    values = []
    for b in buckets:
        row = _planner_row(log, b)
        values.append(str(row.get(col, "")) if not row.empty else "")
    return ">".join(values)


def _dominant_prefix_pattern(lseq: list[str], oseq: list[str]) -> str:
    diffs = [(l, o) for l, o in zip(lseq, oseq) if l != o]
    if not diffs:
        return "same_prefix"
    if any(l == "active_medium" and o == "active_small" for l, o in diffs):
        return "learned_more_aggressive_medium_to_small_oracle"
    if any(l == "active_small" and o == "active_medium" for l, o in diffs):
        return "oracle_more_aggressive_small_to_medium"
    if any(l in ("hold", "pump_saving") and o.startswith("active") for l, o in diffs):
        return "oracle_earlier_active"
    if any(l.startswith("active") and o in ("hold", "pump_saving") for l, o in diffs):
        return "oracle_less_active_or_holds"
    return "mixed_prefix"


def run_probe(
    learned_run: str,
    out_dir: Path,
    key_buckets_path: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    key_df = pd.read_csv(key_buckets_path, low_memory=False)
    key_df["case_id"] = key_df["case_id"].map(_case_from_name)
    learned_dir = _run_path(RUNS[learned_run])
    oracle_dir = _run_path(RUNS["oracle"])
    timeline_rows: list[dict[str, Any]] = []
    prefix_rows: list[dict[str, Any]] = []
    intervention_rows: list[dict[str, Any]] = []

    for case, key_sub in key_df.groupby("case_id"):
        llog, lts = _load_case(learned_dir, case)
        olog, ots = _load_case(oracle_dir, case)
        if llog.empty or olog.empty:
            continue
        key_buckets = sorted(int(x) for x in key_sub["bucket"].tolist())
        buckets = sorted(set(llog["bucket"].astype(int)).intersection(set(olog["bucket"].astype(int))))
        # Task 1: divergence timeline.
        first = {
            "target": -1,
            "tank": -1,
            "pump": -1,
            "platform": -1,
            "raw_action": -1,
            "final_action": -1,
        }
        bucket_cache: dict[int, dict[str, Any]] = {}
        for b in buckets:
            lr = _planner_row(llog, b)
            orow = _planner_row(olog, b)
            t = _safe_float(lr.get("current_time_s"), b * 600.0)
            ltr = _nearest_ts(lts, t)
            otr = _nearest_ts(ots, t)
            delta = _bucket_state_delta(lr, orow, ltr, otr)
            flags = _divergence_flags(delta)
            raw_diff = int(str(lr.get("planner_first_action_raw", lr.get("first_action", ""))) != str(orow.get("planner_first_action_raw", orow.get("first_action", ""))))
            final_diff = int(str(lr.get("first_action", "")) != str(orow.get("first_action", "")))
            rec = {
                "case_id": case,
                "bucket": int(b),
                "is_key_bucket": int(b in key_buckets),
                "learned_action": str(lr.get("first_action", "")),
                "oracle_action": str(orow.get("first_action", "")),
                "learned_raw_action": str(lr.get("planner_first_action_raw", lr.get("first_action", ""))),
                "oracle_raw_action": str(orow.get("planner_first_action_raw", orow.get("first_action", ""))),
                "raw_action_diverged": raw_diff,
                "final_action_diverged": final_diff,
                **delta,
                **flags,
            }
            bucket_cache[b] = rec
            timeline_rows.append(rec)
            for name, col in [("target", "target_diverged"), ("tank", "tank_diverged"), ("pump", "pump_diverged"), ("platform", "platform_posture_diverged")]:
                if first[name] < 0 and rec[col]:
                    first[name] = int(b)
            if first["raw_action"] < 0 and raw_diff:
                first["raw_action"] = int(b)
            if first["final_action"] < 0 and final_diff:
                first["final_action"] = int(b)

        # Task 2/3 proxy: prefix attribution around each key bucket.
        for kb in key_buckets:
            prefix = list(_prefix_window(kb, 5))
            lseq = []
            oseq = []
            lraw = []
            oraw = []
            learned_pump_proxy = 0
            oracle_pump_proxy = 0
            oracle_more_aggressive_count = 0
            oracle_less_aggressive_count = 0
            for b in prefix:
                lr = _planner_row(llog, b)
                orow = _planner_row(olog, b)
                la = str(lr.get("first_action", "")) if not lr.empty else ""
                oa = str(orow.get("first_action", "")) if not orow.empty else ""
                lra = str(lr.get("planner_first_action_raw", la)) if not lr.empty else ""
                ora = str(orow.get("planner_first_action_raw", oa)) if not orow.empty else ""
                lseq.append(la)
                oseq.append(oa)
                lraw.append(lra)
                oraw.append(ora)
                learned_pump_proxy += int(_action_severity(la) > 0)
                oracle_pump_proxy += int(_action_severity(oa) > 0)
                if _action_severity(oa) > _action_severity(la):
                    oracle_more_aggressive_count += 1
                if _action_severity(oa) < _action_severity(la):
                    oracle_less_aggressive_count += 1
            key_state = bucket_cache.get(kb, {})
            before = bucket_cache.get(kb - 1, {})
            pattern = _dominant_prefix_pattern(lseq, oseq)
            prefix_rec = {
                "case_id": case,
                "key_bucket": kb,
                "prefix_buckets": ",".join(str(x) for x in prefix),
                "learned_prefix_actions": ">".join(lseq),
                "oracle_prefix_actions": ">".join(oseq),
                "learned_prefix_raw_actions": ">".join(lraw),
                "oracle_prefix_raw_actions": ">".join(oraw),
                "dominant_prefix_pattern": pattern,
                "oracle_more_aggressive_count": oracle_more_aggressive_count,
                "oracle_less_aggressive_count": oracle_less_aggressive_count,
                "learned_active_bucket_count": learned_pump_proxy,
                "oracle_active_bucket_count": oracle_pump_proxy,
                "oracle_minus_learned_active_count": oracle_pump_proxy - learned_pump_proxy,
                "posture_delta_at_key_norm_deg": key_state.get("posture_delta_norm_deg", np.nan),
                "posture_delta_before_key_norm_deg": before.get("posture_delta_norm_deg", np.nan),
                "posture_delta_growth_into_key_deg": key_state.get("posture_delta_norm_deg", np.nan) - before.get("posture_delta_norm_deg", np.nan) if before else np.nan,
                "target_delta_at_key_kg": key_state.get("target_mean_abs_delta_kg", np.nan),
                "tank_delta_at_key_kg": key_state.get("tank_mean_abs_delta_kg", np.nan),
                "pump_delta_at_key_m3_min": key_state.get("pump_rate_mean_abs_delta_m3_min", np.nan),
                "first_platform_divergence_bucket": first["platform"],
                "first_target_divergence_bucket": first["target"],
                "first_tank_divergence_bucket": first["tank"],
                "first_pump_divergence_bucket": first["pump"],
                "first_raw_action_divergence_bucket": first["raw_action"],
                "first_final_action_divergence_bucket": first["final_action"],
                "platform_divergence_lead_buckets": kb - first["platform"] if first["platform"] >= 0 else np.nan,
            }
            prefix_rows.append(prefix_rec)
            # Proxy intervention rows: not a forced closed-loop rerun.
            for mode, action_seq, active_count in [
                ("learned_forecast_learned_state_learned_prefix", lseq, learned_pump_proxy),
                ("learned_forecast_oracle_like_prefix_proxy", oseq, oracle_pump_proxy),
                ("oracle_forecast_learned_prefix_proxy", lseq, learned_pump_proxy),
                ("oracle_forecast_oracle_like_prefix_proxy", oseq, oracle_pump_proxy),
            ]:
                intervention_rows.append(
                    {
                        "case_id": case,
                        "key_bucket": kb,
                        "intervention_proxy": mode,
                        "prefix_actions": ">".join(action_seq),
                        "prefix_active_bucket_count": active_count,
                        "prefix_active_delta_vs_learned": active_count - learned_pump_proxy,
                        "observed_key_posture_delta_norm_deg": key_state.get("posture_delta_norm_deg", np.nan),
                        "observed_key_target_delta_kg": key_state.get("target_mean_abs_delta_kg", np.nan),
                        "observed_key_tank_delta_kg": key_state.get("tank_mean_abs_delta_kg", np.nan),
                        "observed_key_pump_delta_m3_min": key_state.get("pump_rate_mean_abs_delta_m3_min", np.nan),
                        "proxy_note": "diagnostic_proxy_not_forced_closed_loop_rerun",
                    }
                )

    timeline = pd.DataFrame(timeline_rows)
    prefix = pd.DataFrame(prefix_rows)
    intervention = pd.DataFrame(intervention_rows)
    timeline.to_csv(out_dir / "posture_divergence_timeline.csv", index=False)
    prefix.to_csv(out_dir / "prefix_action_attribution.csv", index=False)
    intervention.to_csv(out_dir / "prefix_intervention_replay.csv", index=False)

    case_summary_rows: list[dict[str, Any]] = []
    for case, sub in timeline.groupby("case_id"):
        key_first = int(sub[sub["is_key_bucket"].eq(1)]["bucket"].min()) if sub["is_key_bucket"].any() else -1
        def first_bucket(col: str) -> int:
            hit = sub[sub[col].eq(1)]["bucket"]
            return int(hit.min()) if not hit.empty else -1
        case_summary_rows.append(
            {
                "case_id": case,
                "first_key_bucket": key_first,
                "first_platform_posture_divergence_bucket": first_bucket("platform_posture_diverged"),
                "first_target_divergence_bucket": first_bucket("target_diverged"),
                "first_tank_divergence_bucket": first_bucket("tank_diverged"),
                "first_pump_divergence_bucket": first_bucket("pump_diverged"),
                "first_raw_action_divergence_bucket": first_bucket("raw_action_diverged"),
                "first_final_action_divergence_bucket": first_bucket("final_action_diverged"),
                "platform_lead_to_key_buckets": key_first - first_bucket("platform_posture_diverged") if first_bucket("platform_posture_diverged") >= 0 and key_first >= 0 else np.nan,
            }
        )
    case_summary = pd.DataFrame(case_summary_rows)
    pattern_counts = prefix["dominant_prefix_pattern"].value_counts().rename_axis("dominant_prefix_pattern").reset_index(name="count")
    active_delta = prefix.groupby("dominant_prefix_pattern")["oracle_minus_learned_active_count"].mean().reset_index(name="mean_oracle_minus_learned_active_count")
    pattern_summary = pattern_counts.merge(active_delta, on="dominant_prefix_pattern", how="left")
    stable_patterns = pattern_summary[pattern_summary["count"].ge(2)]
    lines = [
        "# Path-State Alignment Probe v1",
        "",
        "Analysis-only. No model training and no controller changes.",
        "",
        "## Scope Note",
        "",
        "No forced-prefix-action hook exists in the current runner, so `prefix_intervention_replay.csv` is a diagnostic proxy from existing learned/oracle paths, not a new closed-loop counterfactual run.",
        "",
        "## Case-Level Divergence Timeline",
        _md_table(case_summary),
        "",
        "## Prefix Pattern Counts",
        _md_table(pattern_summary),
        "",
        "## Stable Pattern Assessment",
        "",
    ]
    if stable_patterns.empty:
        lines.append("- No repeated prefix-action pattern appears at count >= 2; do not design a generalized path-level candidate from this alone.")
    else:
        for _, row in stable_patterns.iterrows():
            lines.append(f"- `{row['dominant_prefix_pattern']}` appears {int(row['count'])} times, mean oracle-minus-learned active bucket count {float(row['mean_oracle_minus_learned_active_count']):+.2f}.")
    if (prefix["oracle_minus_learned_active_count"] > 0).sum() > (prefix["oracle_minus_learned_active_count"] < 0).sum():
        lines.append("- Oracle-like prefixes are more often more active than learned prefixes; any path-level recovery candidate risks being a pump-for-posture trade unless verified against f60_relief_e15.")
    elif (prefix["oracle_minus_learned_active_count"] < 0).sum() > 0:
        lines.append("- Some oracle-like prefixes are less active than learned prefixes, so not all oracle alignment is simply more pump.")
    lines.extend(
        [
            "",
            "## Decision",
            "",
            "- Do not implement a path-level candidate unless a repeated pattern improves posture without obvious active-bucket increase.",
            "- If repeated patterns are weak or pump-heavy, freeze h240 path work and keep `f60_relief_e15` as main reference.",
            "",
            "## Outputs",
            "- `path_state_alignment_summary.md`",
            "- `posture_divergence_timeline.csv`",
            "- `prefix_action_attribution.csv`",
            "- `prefix_intervention_replay.csv`",
        ]
    )
    (out_dir / "path_state_alignment_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {out_dir / 'path_state_alignment_summary.md'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--learned-run", default="near_block_eventbalanced_v2")
    parser.add_argument("--key-buckets", default="outputs/wind_prediction/key_bucket_failure_split_v1/key_bucket_failure_split.csv")
    parser.add_argument("--out-dir", default="outputs/wind_prediction/path_state_alignment_probe_v1")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_probe(
        learned_run=str(args.learned_run),
        out_dir=_run_path(args.out_dir),
        key_buckets_path=_run_path(args.key_buckets),
    )


if __name__ == "__main__":
    main()
