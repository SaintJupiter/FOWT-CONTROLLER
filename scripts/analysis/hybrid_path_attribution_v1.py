#!/usr/bin/env python3
"""Hybrid forecast/state attribution for key oracle-vs-learned buckets.

Analysis only. No model training and no controller mutation.

Scope note:
The raw planner cost consumes platform posture, tank masses, pump availability,
hold-relief debt/posture state metadata, and forecast pressure blocks. Target
lifecycle is logged here as closed-loop memory, but it does not directly enter
``evaluate_sequence`` except through target/tank/pump state that already exists
at the bucket. Therefore target lifecycle swap is reported as a diagnostic
state component, not injected into the raw planner surrogate as a hidden input.
"""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "analysis"))

from key_bucket_failure_split_v1 import (  # noqa: E402
    _action_reject_summary,
    _best_for_action,
    _evaluate_all,
    _gap_summary,
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
    _nearest_ts,
    _plant_info,
    _run_path,
    _safe_float,
)
from wind_prediction.replay_dataset import Fino1ReplayDataset  # noqa: E402


def _load_log_ts(run_dir: Path, case: str, bucket: int) -> tuple[pd.Series | None, pd.Series | None]:
    log_path = next((run_dir / "planner_logs").glob(f"*{case}*_planner_log.csv"), None)
    ts_path = next((run_dir / "timeseries").glob(f"*{case}*_timeseries.csv"), None)
    if log_path is None or ts_path is None:
        return None, None
    log = pd.read_csv(log_path, low_memory=False)
    hit = log[log["bucket"].astype(int).eq(bucket)]
    if hit.empty:
        return None, None
    row = hit.iloc[0]
    ts = pd.read_csv(ts_path, low_memory=False)
    return row, _nearest_ts(ts, _safe_float(row.get("current_time_s"), bucket * 600.0))


def _forecast_for(
    run_name: str,
    sample: Any,
    hist: str,
    dataset_dir: Path,
    adapters: dict[str, Any],
) -> tuple[np.ndarray, bool, str]:
    return _forecast_uv_for_run(run_name, sample, hist, dataset_dir, adapters)


def _combine_state(
    base: dict[str, Any],
    donor: dict[str, Any],
    mode: str,
) -> dict[str, Any]:
    out = dict(base)
    if mode in ("full", "platform"):
        for key in ("posture_vec_deg", "posture_state_norm", "posture_state_raw_norm", "posture_state_credit", "posture_state_credit_reason"):
            if key in donor:
                out[key] = donor[key]
    if mode in ("full", "tank"):
        if "tank_masses" in donor:
            out["tank_masses"] = np.asarray(donor["tank_masses"], dtype=float).copy()
    if mode in ("full", "pump"):
        for key in ("pump_fullspeed_any", "pump_total_backlog_kg", "pump_rate_cmd_m3_min"):
            if key in donor:
                value = donor[key]
                out[key] = np.asarray(value, dtype=float).copy() if isinstance(value, np.ndarray) else value
    if mode == "target_lifecycle":
        # Raw planner cost does not read target lifecycle directly. Keep this
        # as an explicit no-op diagnostic so the CSV shows that this component
        # cannot explain raw action changes inside evaluate_sequence.
        out["target_lifecycle_swap_noop"] = 1
    return out


def _target_state(row: pd.Series) -> dict[str, Any]:
    return {
        "target_t1_kg": _safe_float(row.get("prediction_primary_target_t1_kg")),
        "target_t2_kg": _safe_float(row.get("prediction_primary_target_t2_kg")),
        "target_t3_kg": _safe_float(row.get("prediction_primary_target_t3_kg")),
        "target_age_s": _safe_float(row.get("prediction_primary_target_age_s"), 0.0),
        "target_reused": int(_safe_float(row.get("prediction_primary_target_reused"), 0.0) > 0.5),
        "target_refreshed": int(_safe_float(row.get("prediction_primary_target_refreshed"), 0.0) > 0.5),
        "target_resumed": int(_safe_float(row.get("prediction_primary_target_resumed"), 0.0) > 0.5),
        "target_delta_abs_mean_kg": _safe_float(row.get("prediction_primary_delta_abs_mean_kg"), 0.0),
    }


def _eval_combo(
    *,
    label: str,
    state_label: str,
    forecast_label: str,
    plant: dict[str, Any],
    uv: np.ndarray,
    has_future: bool,
    cfg: Any,
    discounts: list[float],
    learned_final_action: str,
    oracle_final_action: str,
) -> dict[str, Any]:
    effective_discounts = discounts if has_future else [1.0 for _ in discounts]
    _, feasible, scalars, evaluated = _evaluate_all(uv, plant, cfg, effective_discounts)
    raw_action = str(feasible[0]["sequence"][0]) if feasible else "no_feasible"
    sequence = ">".join(feasible[0]["sequence"]) if feasible else ""
    best_cost = float(scalars[0]) if scalars else np.nan
    oracle_tuple = _best_for_action(feasible, scalars, oracle_final_action)
    learned_tuple = _best_for_action(feasible, scalars, learned_final_action)
    selected_tuple = _best_for_action(feasible, scalars, raw_action)
    oracle_gap_main, oracle_gap_value, oracle_gap_components = _gap_summary(
        selected_tuple[2] if selected_tuple else None,
        oracle_tuple[2] if oracle_tuple else None,
        cfg,
    )
    return {
        "combo": label,
        "state_label": state_label,
        "forecast_label": forecast_label,
        "raw_action": raw_action,
        "best_sequence": sequence,
        "selected_cost": best_cost,
        "oracle_action": oracle_final_action,
        "learned_action": learned_final_action,
        "matches_oracle_action": int(raw_action == oracle_final_action),
        "matches_learned_action": int(raw_action == learned_final_action),
        "oracle_action_rank": oracle_tuple[0] if oracle_tuple else np.nan,
        "oracle_action_cost_gap": oracle_tuple[1] - best_cost if oracle_tuple and np.isfinite(best_cost) else np.nan,
        "learned_action_rank": learned_tuple[0] if learned_tuple else np.nan,
        "learned_action_cost_gap": learned_tuple[1] - best_cost if learned_tuple and np.isfinite(best_cost) else np.nan,
        "oracle_action_reject_summary": _action_reject_summary(evaluated, oracle_final_action),
        "learned_action_reject_summary": _action_reject_summary(evaluated, learned_final_action),
        "oracle_gap_main_component": oracle_gap_main,
        "oracle_gap_main_value": oracle_gap_value,
        "oracle_gap_components": oracle_gap_components,
        "hard_reject_count": int(sum(1 for r in evaluated if r.get("hard_reject_reason"))),
        "active_medium_reject_count": int(sum(1 for r in evaluated if r.get("hard_reject_reason") == "active_medium_gate")),
        "capacity_reject_count": int(sum(1 for r in evaluated if r.get("hard_reject_reason") == "capacity_guard")),
        "fullspeed_reject_count": int(sum(1 for r in evaluated if r.get("hard_reject_reason") == "fullspeed_guard")),
        "predicted_pump_work_cost": float(feasible[0]["costs"].get("pump_work_cost", np.nan)) if feasible else np.nan,
        "predicted_pump_duration_cost": float(feasible[0]["costs"].get("pump_duration_cost", np.nan)) if feasible else np.nan,
        "selected_active_medium_gate_reason": str(feasible[0].get("active_medium_gate_reason", "")) if feasible else "",
    }


def _numeric_delta(a: Any, b: Any) -> float:
    try:
        aa = np.asarray(a, dtype=float)
        bb = np.asarray(b, dtype=float)
        return float(np.nanmean(np.abs(aa - bb)))
    except Exception:
        return float("nan")


def _case_timeline(
    learned_dir: Path,
    oracle_dir: Path,
    case: str,
    key_buckets: set[int],
) -> list[dict[str, Any]]:
    llog_path = next((learned_dir / "planner_logs").glob(f"*{case}*_planner_log.csv"), None)
    olog_path = next((oracle_dir / "planner_logs").glob(f"*{case}*_planner_log.csv"), None)
    lts_path = next((learned_dir / "timeseries").glob(f"*{case}*_timeseries.csv"), None)
    ots_path = next((oracle_dir / "timeseries").glob(f"*{case}*_timeseries.csv"), None)
    if not all([llog_path, olog_path, lts_path, ots_path]):
        return []
    llog = pd.read_csv(llog_path, low_memory=False)
    olog = pd.read_csv(olog_path, low_memory=False)
    lts = pd.read_csv(lts_path, low_memory=False)
    ots = pd.read_csv(ots_path, low_memory=False)
    rows: list[dict[str, Any]] = []
    for bucket in sorted(set(llog["bucket"].astype(int)).intersection(set(olog["bucket"].astype(int)))):
        lr = llog[llog["bucket"].astype(int).eq(bucket)].iloc[0]
        orow = olog[olog["bucket"].astype(int).eq(bucket)].iloc[0]
        t = _safe_float(lr.get("current_time_s"), bucket * 600.0)
        ltr = _nearest_ts(lts, t)
        otr = _nearest_ts(ots, t)
        target_delta = _numeric_delta(
            [lr.get("prediction_primary_target_t1_kg"), lr.get("prediction_primary_target_t2_kg"), lr.get("prediction_primary_target_t3_kg")],
            [orow.get("prediction_primary_target_t1_kg"), orow.get("prediction_primary_target_t2_kg"), orow.get("prediction_primary_target_t3_kg")],
        )
        tank_delta = _numeric_delta(
            [ltr.get("tank1_kg"), ltr.get("tank2_kg"), ltr.get("tank3_kg")],
            [otr.get("tank1_kg"), otr.get("tank2_kg"), otr.get("tank3_kg")],
        )
        pump_delta = _numeric_delta(
            [ltr.get("pump_rate1_m3min"), ltr.get("pump_rate2_m3min"), ltr.get("pump_rate3_m3min")],
            [otr.get("pump_rate1_m3min"), otr.get("pump_rate2_m3min"), otr.get("pump_rate3_m3min")],
        )
        rows.append(
            {
                "case_id": case,
                "bucket": int(bucket),
                "is_key_bucket": int(bucket in key_buckets),
                "target_diverged": int(target_delta > 500.0),
                "tank_diverged": int(tank_delta > 500.0),
                "pump_diverged": int(pump_delta > 0.1),
                "raw_action_diverged": int(str(lr.get("planner_first_action_raw", lr.get("first_action", ""))) != str(orow.get("planner_first_action_raw", orow.get("first_action", "")))),
                "final_action_diverged": int(str(lr.get("first_action", "")) != str(orow.get("first_action", ""))),
                "target_mean_abs_delta_kg": target_delta,
                "tank_mean_abs_delta_kg": tank_delta,
                "pump_rate_mean_abs_delta_m3_min": pump_delta,
            }
        )
    return rows


def run_attribution(
    learned_run: str,
    out_dir: Path,
    dataset_dir: Path,
    key_buckets_path: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    key_df = pd.read_csv(key_buckets_path, low_memory=False)
    key_df["case_id"] = key_df["case_id"].map(_case_from_name)
    learned_dir = _run_path(RUNS[learned_run])
    oracle_dir = _run_path(RUNS["oracle"])
    cfg, discounts = _cfg_from_casebook(learned_dir)
    replay = Fino1ReplayDataset(dataset_dir=dataset_dir, split="test")
    adapters: dict[str, Any] = {}
    cf_rows: list[dict[str, Any]] = []
    swap_rows: list[dict[str, Any]] = []
    timeline_rows: list[dict[str, Any]] = []

    for case, sub in key_df.groupby("case_id"):
        timeline_rows.extend(_case_timeline(learned_dir, oracle_dir, case, set(int(x) for x in sub["bucket"].tolist())))

    for _, key in key_df.iterrows():
        case = _case_from_name(str(key["case_id"]))
        bucket = int(key["bucket"])
        learned_row, learned_ts = _load_log_ts(learned_dir, case, bucket)
        oracle_row, oracle_ts = _load_log_ts(oracle_dir, case, bucket)
        if learned_row is None or oracle_row is None or learned_ts is None or oracle_ts is None:
            continue
        hist = str(learned_row.get("history_end", "")).strip()
        if not hist:
            continue
        sample = replay.sample_for_history_end(datetime.strptime(hist, TIMESTAMP_FMT))
        if sample is None:
            continue
        learned_plant = _plant_info(learned_row, learned_ts)
        oracle_plant = _plant_info(oracle_row, oracle_ts)
        learned_uv, learned_has_future, learned_version = _forecast_for(learned_run, sample, hist, dataset_dir, adapters)
        oracle_uv, oracle_has_future, oracle_version = _forecast_for("oracle", sample, hist, dataset_dir, adapters)
        learned_action = str(learned_row.get("first_action", ""))
        oracle_action = str(oracle_row.get("first_action", ""))

        combos = [
            ("S_L+F_L", "learned", "learned", learned_plant, learned_uv, learned_has_future),
            ("S_L+F_O", "learned", "oracle", learned_plant, oracle_uv, oracle_has_future),
            ("S_O+F_L", "oracle", "learned", oracle_plant, learned_uv, learned_has_future),
            ("S_O+F_O", "oracle", "oracle", oracle_plant, oracle_uv, oracle_has_future),
        ]
        for label, state_label, forecast_label, plant, uv, has_future in combos:
            rec = {
                "case_id": case,
                "bucket": bucket,
                "learned_log_action": learned_action,
                "oracle_log_action": oracle_action,
                "learned_raw_action": str(learned_row.get("planner_first_action_raw", learned_action)),
                "oracle_raw_action": str(oracle_row.get("planner_first_action_raw", oracle_action)),
                "learned_forecast_version": learned_version,
                "oracle_forecast_version": oracle_version,
            }
            rec.update(_eval_combo(
                label=label,
                state_label=state_label,
                forecast_label=forecast_label,
                plant=plant,
                uv=uv,
                has_future=has_future,
                cfg=cfg,
                discounts=discounts,
                learned_final_action=learned_action,
                oracle_final_action=oracle_action,
            ))
            rec.update({f"learned_{k}": v for k, v in _target_state(learned_row).items()})
            rec.update({f"oracle_{k}": v for k, v in _target_state(oracle_row).items()})
            cf_rows.append(rec)

        for component in ["platform", "tank", "pump", "target_lifecycle", "full"]:
            swapped = _combine_state(learned_plant, oracle_plant, component)
            rec = {
                "case_id": case,
                "bucket": bucket,
                "swap_component": component,
                "base_state": "learned",
                "donor_state": "oracle",
                "forecast_label": "learned",
                "learned_log_action": learned_action,
                "oracle_log_action": oracle_action,
            }
            rec.update(_eval_combo(
                label=f"S_L_swap_{component}+F_L",
                state_label=f"learned_swap_{component}",
                forecast_label="learned",
                plant=swapped,
                uv=learned_uv,
                has_future=learned_has_future,
                cfg=cfg,
                discounts=discounts,
                learned_final_action=learned_action,
                oracle_final_action=oracle_action,
            ))
            swap_rows.append(rec)

    cf = pd.DataFrame(cf_rows)
    swap = pd.DataFrame(swap_rows)
    timeline = pd.DataFrame(timeline_rows)
    cf.to_csv(out_dir / "hybrid_counterfactual_replay_table.csv", index=False)
    swap.to_csv(out_dir / "state_component_swap_table.csv", index=False)

    case_rows: list[dict[str, Any]] = []
    for case, sub in timeline.groupby("case_id"):
        key_bucket = int(sub[sub["is_key_bucket"].eq(1)]["bucket"].min()) if sub["is_key_bucket"].any() else -1
        def first(col: str) -> int:
            hit = sub[sub[col].eq(1)]["bucket"]
            return int(hit.min()) if not hit.empty else -1
        case_rows.append(
            {
                "case_id": case,
                "first_key_bucket": key_bucket,
                "first_target_divergence_bucket": first("target_diverged"),
                "first_tank_divergence_bucket": first("tank_diverged"),
                "first_pump_divergence_bucket": first("pump_diverged"),
                "first_raw_action_divergence_bucket": first("raw_action_diverged"),
                "first_final_action_divergence_bucket": first("final_action_diverged"),
                "target_before_or_at_key": int(first("target_diverged") >= 0 and first("target_diverged") <= key_bucket),
                "tank_before_or_at_key": int(first("tank_diverged") >= 0 and first("tank_diverged") <= key_bucket),
                "pump_before_or_at_key": int(first("pump_diverged") >= 0 and first("pump_diverged") <= key_bucket),
            }
        )
    cases = pd.DataFrame(case_rows)
    cases.to_csv(out_dir / "case_path_divergence_timeline.csv", index=False)

    pivot = (
        cf.groupby("combo")["matches_oracle_action"]
        .agg(["sum", "count", "mean"])
        .reset_index()
        .rename(columns={"sum": "oracle_action_matches", "count": "buckets", "mean": "match_rate"})
    )
    avg_gap = (
        cf.groupby("combo")["oracle_action_cost_gap"]
        .mean()
        .reset_index(name="mean_oracle_action_gap")
    )
    pivot = pivot.merge(avg_gap, on="combo", how="left")
    swap_summary = (
        swap.groupby("swap_component")["matches_oracle_action"]
        .agg(["sum", "count", "mean"])
        .reset_index()
        .rename(columns={"sum": "oracle_action_matches", "count": "buckets", "mean": "match_rate"})
    )
    lines = [
        "# Hybrid Path Attribution v1",
        "",
        "Analysis-only. No model training and no controller changes.",
        "",
        "## Four-Quadrant Counterfactual Replay",
        _md_table(pivot),
        "",
        "## State Component Swap Summary",
        _md_table(swap_summary),
        "",
        "## Case Path Divergence Timeline",
        _md_table(cases),
        "",
        "## Interpretation",
        "",
    ]
    sl_fl = cf[cf["combo"].eq("S_L+F_L")]["matches_oracle_action"].sum()
    sl_fo = cf[cf["combo"].eq("S_L+F_O")]["matches_oracle_action"].sum()
    so_fl = cf[cf["combo"].eq("S_O+F_L")]["matches_oracle_action"].sum()
    so_fo = cf[cf["combo"].eq("S_O+F_O")]["matches_oracle_action"].sum()
    if sl_fo > sl_fl:
        lines.append(f"- Oracle forecast on learned state improves oracle-action matches ({int(sl_fl)} -> {int(sl_fo)}), so forecast still contributes.")
    else:
        lines.append(f"- Oracle forecast on learned state does not improve oracle-action matches ({int(sl_fl)} -> {int(sl_fo)}).")
    if so_fl > sl_fl:
        lines.append(f"- Oracle state with learned forecast improves oracle-action matches ({int(sl_fl)} -> {int(so_fl)}), so path state contributes.")
    else:
        lines.append(f"- Oracle state with learned forecast does not improve oracle-action matches ({int(sl_fl)} -> {int(so_fl)}).")
    if so_fo > max(sl_fo, so_fl):
        lines.append("- Full oracle behavior needs interaction between forecast and state; neither alone recovers it.")
    if int(cases[["target_before_or_at_key", "tank_before_or_at_key", "pump_before_or_at_key"]].max(axis=1).sum()) == len(cases):
        lines.append("- Every affected case has target/tank/pump divergence before or at the first key bucket.")
    lines.extend(
        [
            "",
            "## Outputs",
            "- `hybrid_counterfactual_replay_table.csv`",
            "- `state_component_swap_table.csv`",
            "- `case_path_divergence_timeline.csv`",
            "- `hybrid_path_attribution_summary.md`",
        ]
    )
    (out_dir / "hybrid_path_attribution_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {out_dir / 'hybrid_path_attribution_summary.md'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--learned-run", default="near_block_eventbalanced_v2")
    parser.add_argument("--dataset-dir", default=DEFAULT_DATASET_DIR)
    parser.add_argument("--key-buckets", default=KEY_BUCKETS)
    parser.add_argument("--out-dir", default="outputs/wind_prediction/hybrid_path_attribution_v1")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_attribution(
        learned_run=str(args.learned_run),
        out_dir=_run_path(args.out_dir),
        dataset_dir=_run_path(args.dataset_dir),
        key_buckets_path=_run_path(args.key_buckets),
    )


if __name__ == "__main__":
    main()
