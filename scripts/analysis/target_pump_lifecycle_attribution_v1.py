#!/usr/bin/env python3
"""Target/pump lifecycle attribution for oracle-vs-learned key buckets.

This is the follow-up to G001. It is analysis-only: no training and no
controller mutation. The key question is whether offline planner replay failed
because forecast/action cost attribution is wrong, or because oracle and learned
closed-loop states diverged before the bucket.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "analysis"))

from key_bucket_failure_split_v1 import (  # noqa: E402
    _action_reject_summary,
    _best_for_action,
    _bool_log,
    _evaluate_all,
    _float_log,
    _load_log_row,
    _load_ts_row,
    _md_table,
)
from model_capability_map_v1 import (  # noqa: E402
    DEFAULT_DATASET_DIR,
    KEY_BUCKETS,
    RUNS,
    TIMESTAMP_FMT,
    _case_from_name,
    _cfg_from_casebook,
    _forecast_uv_for_run,
    _plant_info,
    _run_path,
    _safe_float,
)
from wind_prediction.replay_dataset import Fino1ReplayDataset  # noqa: E402


def _replay_one(
    *,
    run_name: str,
    run_dir: Path,
    case: str,
    bucket: int,
    dataset_dir: Path,
    replay: Fino1ReplayDataset,
    adapters: dict[str, Any],
) -> dict[str, Any] | None:
    log_row = _load_log_row(run_dir, case, bucket)
    if log_row is None:
        return None
    ts_row = _load_ts_row(run_dir, case, bucket, log_row)
    hist = str(log_row.get("history_end", "")).strip()
    if not hist:
        return None
    sample = replay.sample_for_history_end(datetime.strptime(hist, TIMESTAMP_FMT))
    if sample is None:
        return None
    cfg, discounts = _cfg_from_casebook(run_dir)
    plant = _plant_info(log_row, ts_row)
    uv, has_future, version = _forecast_uv_for_run(
        run_name,
        sample,
        hist,
        dataset_dir,
        adapters,
    )
    plant["forecast_has_future"] = int(bool(has_future))
    effective_discounts = discounts if has_future else [1.0 for _ in discounts]
    raw_blocks, feasible, scalars, evaluated = _evaluate_all(
        uv,
        plant,
        cfg,
        effective_discounts,
    )
    replay_action = str(feasible[0]["sequence"][0]) if feasible else "no_feasible"
    replay_sequence = ">".join(feasible[0]["sequence"]) if feasible else ""
    raw_action = str(log_row.get("planner_first_action_raw", log_row.get("first_action", "")))
    final_action = str(log_row.get("first_action", ""))
    raw_tuple = _best_for_action(feasible, scalars, raw_action)
    final_tuple = _best_for_action(feasible, scalars, final_action)
    return {
        "log_row": log_row,
        "ts_row": ts_row,
        "cfg": cfg,
        "plant": plant,
        "raw_blocks": raw_blocks,
        "feasible": feasible,
        "scalars": scalars,
        "evaluated": evaluated,
        "forecast_version": version,
        "forecast_has_future": int(bool(has_future)),
        "raw_action": raw_action,
        "final_action": final_action,
        "replay_action": replay_action,
        "replay_sequence": replay_sequence,
        "raw_rank": raw_tuple[0] if raw_tuple else np.nan,
        "raw_gap": raw_tuple[1] - scalars[0] if raw_tuple and scalars else np.nan,
        "final_rank": final_tuple[0] if final_tuple else np.nan,
        "final_gap": final_tuple[1] - scalars[0] if final_tuple and scalars else np.nan,
        "raw_reject_summary": _action_reject_summary(evaluated, raw_action),
        "final_reject_summary": _action_reject_summary(evaluated, final_action),
    }


def _log_value(row: pd.Series, name: str, default: Any = "") -> Any:
    return row.get(name, default) if name in row.index else default


def _target_vec(row: pd.Series) -> np.ndarray:
    return np.array(
        [
            _safe_float(row.get("prediction_primary_target_t1_kg"), np.nan),
            _safe_float(row.get("prediction_primary_target_t2_kg"), np.nan),
            _safe_float(row.get("prediction_primary_target_t3_kg"), np.nan),
        ],
        dtype=float,
    )


def _tank_vec(ts_row: pd.Series) -> np.ndarray:
    return np.array(
        [
            _safe_float(ts_row.get("tank1_kg"), np.nan),
            _safe_float(ts_row.get("tank2_kg"), np.nan),
            _safe_float(ts_row.get("tank3_kg"), np.nan),
        ],
        dtype=float,
    )


def _pump_vec(ts_row: pd.Series) -> np.ndarray:
    return np.array(
        [
            _safe_float(ts_row.get("pump_rate1_m3min"), 0.0),
            _safe_float(ts_row.get("pump_rate2_m3min"), 0.0),
            _safe_float(ts_row.get("pump_rate3_m3min"), 0.0),
        ],
        dtype=float,
    )


def _state_delta(learned: dict[str, Any], oracle: dict[str, Any]) -> dict[str, float]:
    lrow, orow = learned["log_row"], oracle["log_row"]
    lts, ots = learned["ts_row"], oracle["ts_row"]
    target_l = _target_vec(lrow)
    target_o = _target_vec(orow)
    tank_l = _tank_vec(lts)
    tank_o = _tank_vec(ots)
    pump_l = _pump_vec(lts)
    pump_o = _pump_vec(ots)
    return {
        "state_pitch_delta_oracle_minus_learned": _float_log(orow, "current_pitch_deg") - _float_log(lrow, "current_pitch_deg"),
        "state_roll_delta_oracle_minus_learned": _float_log(orow, "current_roll_deg") - _float_log(lrow, "current_roll_deg"),
        "target_mean_abs_delta_kg": float(np.nanmean(np.abs(target_o - target_l))),
        "tank_mean_abs_delta_kg": float(np.nanmean(np.abs(tank_o - tank_l))),
        "pump_rate_mean_abs_delta_m3_min": float(np.nanmean(np.abs(pump_o - pump_l))),
        "backlog_delta_oracle_minus_learned_kg": _safe_float(ots.get("pump_total_backlog_kg"), 0.0) - _safe_float(lts.get("pump_total_backlog_kg"), 0.0),
        "target_age_delta_oracle_minus_learned_s": _float_log(orow, "prediction_primary_target_age_s", 0.0) - _float_log(lrow, "prediction_primary_target_age_s", 0.0),
    }


def _final_postprocess_tags(run: dict[str, Any], prefix: str) -> dict[str, Any]:
    row = run["log_row"]
    tags: list[str] = []
    if run["final_action"] != run["raw_action"]:
        tags.append("final_differs_from_raw")
    for col, tag in [
        ("relief_medium_cap_active", "relief_medium_cap"),
        ("far_horizon_relief_gate_active", "far_horizon_gate"),
        ("medium_escalation_active", "medium_escalation"),
        ("no_unexplained_hold_action_active", "no_unexplained_hold"),
        ("hold_risk_micro_action_active", "hold_risk_micro"),
        ("quiet_posture_action_active", "quiet_posture"),
        ("preview_lead_action_active", "preview_lead"),
        ("preview_sequence_lead_action_active", "sequence_lead"),
    ]:
        if _bool_log(row, col):
            tags.append(tag)
    return {
        f"{prefix}_postprocess_tags": ";".join(tags),
        f"{prefix}_relief_medium_cap_reason": _log_value(row, "relief_medium_cap_reason"),
        f"{prefix}_far_horizon_relief_gate_reason": _log_value(row, "far_horizon_relief_gate_reason"),
        f"{prefix}_medium_escalation_reason": _log_value(row, "medium_escalation_reason"),
        f"{prefix}_no_unexplained_hold_reason": _log_value(row, "no_unexplained_hold_action_reason"),
    }


def _classify_lifecycle(
    learned: dict[str, Any],
    oracle: dict[str, Any],
    g001_row: pd.Series,
    deltas: dict[str, float],
) -> tuple[str, str]:
    learned_own_ok = learned["replay_action"] == learned["raw_action"]
    oracle_own_ok = oracle["replay_action"] == oracle["raw_action"]
    tags: list[str] = []
    if learned_own_ok:
        tags.append("learned_own_replay_ok")
    if oracle_own_ok:
        tags.append("oracle_own_replay_ok")
    if learned["final_action"] != learned["raw_action"] or oracle["final_action"] != oracle["raw_action"]:
        tags.append("postprocess_final_action")
    if deltas["target_mean_abs_delta_kg"] > 500.0 or abs(deltas["target_age_delta_oracle_minus_learned_s"]) >= 600.0:
        tags.append("target_state_diverged")
    if deltas["tank_mean_abs_delta_kg"] > 500.0 or deltas["pump_rate_mean_abs_delta_m3_min"] > 0.1 or abs(deltas["backlog_delta_oracle_minus_learned_kg"]) > 500.0:
        tags.append("pump_or_tank_state_diverged")
    if str(g001_row.get("primary_cause", "")) == "replay-inconclusive" and learned_own_ok and oracle_own_ok:
        primary = "closed-loop-state-divergence"
    elif not learned_own_ok or not oracle_own_ok:
        primary = "unmodeled-replay-gap"
    elif "postprocess_final_action" in tags:
        primary = "postprocess-sensitive"
    elif "target_state_diverged" in tags:
        primary = "target-lifecycle-divergence"
    elif "pump_or_tank_state_diverged" in tags:
        primary = "pump-state-divergence"
    else:
        primary = "replay-clean"
    return primary, ";".join(tags)


def run_attribution(
    learned_run: str,
    out_dir: Path,
    dataset_dir: Path,
    g001_csv: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    g001 = pd.read_csv(g001_csv, low_memory=False)
    learned_dir = _run_path(RUNS[learned_run])
    oracle_dir = _run_path(RUNS["oracle"])
    replay = Fino1ReplayDataset(dataset_dir=dataset_dir, split="test")
    adapters: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []

    for _, g in g001.iterrows():
        case = _case_from_name(str(g["case_id"]))
        bucket = int(g["bucket"])
        learned = _replay_one(
            run_name=learned_run,
            run_dir=learned_dir,
            case=case,
            bucket=bucket,
            dataset_dir=dataset_dir,
            replay=replay,
            adapters=adapters,
        )
        oracle = _replay_one(
            run_name="oracle",
            run_dir=oracle_dir,
            case=case,
            bucket=bucket,
            dataset_dir=dataset_dir,
            replay=replay,
            adapters=adapters,
        )
        if learned is None or oracle is None:
            continue
        deltas = _state_delta(learned, oracle)
        primary, tags = _classify_lifecycle(learned, oracle, g, deltas)
        rec: dict[str, Any] = {
            "case_id": case,
            "bucket": bucket,
            "g001_primary_cause": g.get("primary_cause", ""),
            "g001_split_group": g.get("split_group", ""),
            "lifecycle_primary_cause": primary,
            "lifecycle_tags": tags,
            "learned_final_action": learned["final_action"],
            "learned_raw_action": learned["raw_action"],
            "learned_own_replay_action": learned["replay_action"],
            "learned_own_replay_matches_raw": int(learned["replay_action"] == learned["raw_action"]),
            "learned_raw_rank": learned["raw_rank"],
            "learned_raw_gap": learned["raw_gap"],
            "learned_raw_reject_summary": learned["raw_reject_summary"],
            "oracle_final_action": oracle["final_action"],
            "oracle_raw_action": oracle["raw_action"],
            "oracle_own_replay_action": oracle["replay_action"],
            "oracle_own_replay_matches_raw": int(oracle["replay_action"] == oracle["raw_action"]),
            "oracle_raw_rank": oracle["raw_rank"],
            "oracle_raw_gap": oracle["raw_gap"],
            "oracle_raw_reject_summary": oracle["raw_reject_summary"],
            "learned_target_reused": int(_bool_log(learned["log_row"], "prediction_primary_target_reused")),
            "oracle_target_reused": int(_bool_log(oracle["log_row"], "prediction_primary_target_reused")),
            "learned_target_refreshed": int(_bool_log(learned["log_row"], "prediction_primary_target_refreshed")),
            "oracle_target_refreshed": int(_bool_log(oracle["log_row"], "prediction_primary_target_refreshed")),
            "learned_target_age_s": _float_log(learned["log_row"], "prediction_primary_target_age_s", 0.0),
            "oracle_target_age_s": _float_log(oracle["log_row"], "prediction_primary_target_age_s", 0.0),
            "learned_delta_abs_mean_kg": _float_log(learned["log_row"], "prediction_primary_delta_abs_mean_kg", 0.0),
            "oracle_delta_abs_mean_kg": _float_log(oracle["log_row"], "prediction_primary_delta_abs_mean_kg", 0.0),
            "learned_active_effectiveness_target_err_kg": _float_log(learned["log_row"], "active_effectiveness_target_err_mean_kg", 0.0),
            "oracle_active_effectiveness_target_err_kg": _float_log(oracle["log_row"], "active_effectiveness_target_err_mean_kg", 0.0),
            "learned_active_effectiveness_pump_idle": int(_bool_log(learned["log_row"], "active_effectiveness_pump_idle")),
            "oracle_active_effectiveness_pump_idle": int(_bool_log(oracle["log_row"], "active_effectiveness_pump_idle")),
            "learned_backlog_kg": _safe_float(learned["ts_row"].get("pump_total_backlog_kg"), 0.0),
            "oracle_backlog_kg": _safe_float(oracle["ts_row"].get("pump_total_backlog_kg"), 0.0),
        }
        rec.update(deltas)
        rec.update(_final_postprocess_tags(learned, "learned"))
        rec.update(_final_postprocess_tags(oracle, "oracle"))
        rows.append(rec)

    out = pd.DataFrame(rows)
    out.to_csv(out_dir / "target_pump_lifecycle_attribution.csv", index=False)
    cause_counts = out["lifecycle_primary_cause"].value_counts().rename_axis("lifecycle_primary_cause").reset_index(name="count")
    g001_x_life = pd.crosstab(out["g001_primary_cause"], out["lifecycle_primary_cause"]).reset_index()
    own_replay = pd.DataFrame(
        [
            {
                "run": "learned",
                "own_replay_raw_match": int(out["learned_own_replay_matches_raw"].sum()),
                "total": int(len(out)),
            },
            {
                "run": "oracle",
                "own_replay_raw_match": int(out["oracle_own_replay_matches_raw"].sum()),
                "total": int(len(out)),
            },
        ]
    )
    replay_inconclusive = out[out["g001_primary_cause"].eq("replay-inconclusive")]
    lines = [
        "# Target/Pump Lifecycle Attribution v1",
        "",
        "Analysis-only. No model training and no controller changes.",
        "",
        "## Summary",
        "",
        f"- key buckets analyzed: {len(out)}",
        f"- learned own-state replay matches raw action: {int(out['learned_own_replay_matches_raw'].sum())}/{len(out)}",
        f"- oracle own-state replay matches raw action: {int(out['oracle_own_replay_matches_raw'].sum())}/{len(out)}",
        f"- G001 replay-inconclusive buckets: {len(replay_inconclusive)}",
        f"- of those, both own-state replays match raw: {int(((replay_inconclusive['learned_own_replay_matches_raw'] == 1) & (replay_inconclusive['oracle_own_replay_matches_raw'] == 1)).sum())}/{len(replay_inconclusive)}",
        "",
        "## Lifecycle Cause Counts",
        _md_table(cause_counts),
        "",
        "## G001 Cause vs Lifecycle Cause",
        _md_table(g001_x_life),
        "",
        "## Own-State Replay Fidelity",
        _md_table(own_replay),
        "",
        "## Decision",
        "",
    ]
    if len(replay_inconclusive) and int(((replay_inconclusive["learned_own_replay_matches_raw"] == 1) & (replay_inconclusive["oracle_own_replay_matches_raw"] == 1)).sum()) >= max(1, len(replay_inconclusive) // 2):
        lines.append("- Most G001 replay-inconclusive buckets become explainable when each run uses its own closed-loop state. The previous replay gap was mainly state-path dependence, not a planner surrogate bug.")
    else:
        lines.append("- Own-state replay still misses many raw actions. Improve replay state reconstruction before G002/G003.")
    if out["lifecycle_tags"].astype(str).str.contains("target_state_diverged", regex=False).sum() > 0:
        lines.append("- Target lifecycle divergence is material; forecast/controller candidates should be judged with target-state attribution, not first_action alone.")
    if out["lifecycle_tags"].astype(str).str.contains("postprocess_final_action", regex=False).sum() > 0:
        lines.append("- Final action post-processing is material; raw planner action and final action must stay separated in later G002/G003 metrics.")
    lines.extend(
        [
            "",
            "## Outputs",
            "- `target_pump_lifecycle_attribution.csv`",
            "- `target_pump_lifecycle_attribution_summary.md`",
        ]
    )
    (out_dir / "target_pump_lifecycle_attribution_summary.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {out_dir / 'target_pump_lifecycle_attribution_summary.md'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--learned-run", default="near_block_eventbalanced_v2")
    parser.add_argument("--dataset-dir", default=DEFAULT_DATASET_DIR)
    parser.add_argument(
        "--g001-csv",
        default="outputs/wind_prediction/key_bucket_failure_split_v1/key_bucket_failure_split.csv",
    )
    parser.add_argument(
        "--out-dir",
        default="outputs/wind_prediction/target_pump_lifecycle_attribution_v1",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_attribution(
        learned_run=str(args.learned_run),
        out_dir=_run_path(args.out_dir),
        dataset_dir=_run_path(args.dataset_dir),
        g001_csv=_run_path(args.g001_csv),
    )


if __name__ == "__main__":
    main()
