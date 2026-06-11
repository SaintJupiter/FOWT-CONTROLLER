#!/usr/bin/env python3
"""Split key oracle/action-diff buckets by failure mode.

This is G001 for the post-shape-loss optimization round. It is analysis-only:
no model training and no controller mutation.

Important terminology:
- ``forecast-flippable`` means the *same bucket initial state, candidate set,
  gates, and target/pump state reconstructed from the learned run* flips to the
  oracle final first_action when only the learned forecast UV is linearly
  interpolated toward oracle UV.
- ``planner-blocked`` means that this near-block interpolation does not produce
  the oracle final first_action by alpha=1.0. This does not mean full oracle
  closed-loop cannot choose that action; the input key buckets are already
  defined by full oracle-vs-learned first_action differences.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "analysis"))

from cost_margin_flip_alpha_v1 import _evaluate_uv, _interp_uv  # noqa: E402
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
from wind_prediction.ballast_planner import (  # noqa: E402
    _ALL_SEQUENCES,
    apply_posture_state_to_blocks,
    compute_pressure_blocks,
    economic_scalar_cost,
    evaluate_sequence,
    select_best_economic,
)
from wind_prediction.replay_dataset import Fino1ReplayDataset  # noqa: E402


def _load_log_row(run_dir: Path, case: str, bucket: int) -> pd.Series | None:
    path = next((run_dir / "planner_logs").glob(f"*{case}*_planner_log.csv"), None)
    if path is None:
        return None
    log = pd.read_csv(path, low_memory=False)
    hit = log[log["bucket"].astype(int).eq(bucket)]
    return None if hit.empty else hit.iloc[0]


def _load_ts_row(run_dir: Path, case: str, bucket: int, log_row: pd.Series) -> pd.Series:
    path = next((run_dir / "timeseries").glob(f"*{case}*_timeseries.csv"), None)
    if path is None:
        return pd.Series(dtype=object)
    ts = pd.read_csv(path, low_memory=False)
    return _nearest_ts(ts, _safe_float(log_row.get("current_time_s"), bucket * 600.0))


def _weighted_components(row: dict[str, Any], cfg: Any) -> dict[str, float]:
    c = row["costs"]
    envelope_soft = 0.0 if cfg.attitude_zone_form == "smooth_huber" else cfg.w_envelope_soft
    terminal_envelope_soft = 0.0 if cfg.attitude_zone_form == "smooth_huber" else cfg.w_terminal_envelope_soft
    return {
        "pump_work": cfg.w_pump_work * c["pump_work_cost"],
        "pump_duration": cfg.w_pump_duration * c["pump_duration_cost"],
        "startstop": cfg.w_startstop * c["startstop_cost"],
        "direction_switch": cfg.w_direction_switch * c["direction_switch_cost"],
        "reverse": cfg.w_reverse_penalty * c["reverse_penalty"],
        "attitude_residual": cfg.w_attitude_residual * c["attitude_residual_cost"],
        "terminal_residual": cfg.w_terminal_residual * c["terminal_residual_cost"],
        "envelope_soft": envelope_soft * c["envelope_violation_cost"],
        "terminal_envelope_soft": terminal_envelope_soft * c["terminal_envelope_violation"],
        "saturation": cfg.w_saturation_hard * c["saturation_penalty"],
        "envelope_barrier": cfg.envelope_barrier_const * c.get("envelope_barrier_triggered", 0.0),
        "posture_hold_barrier": cfg.posture_hold_barrier_const * c.get("posture_hold_barrier_cost", 0.0),
        "hold_relief_debt": cfg.hold_relief_debt_weight * c.get("hold_relief_debt_cost", 0.0),
    }


def _evaluate_all(
    uv: np.ndarray,
    plant: dict[str, Any],
    cfg: Any,
    discounts: list[float],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[float], list[dict[str, Any]]]:
    raw_blocks = compute_pressure_blocks(np.asarray(uv, dtype=float), discounts, cfg)
    planner_blocks, posture_meta = apply_posture_state_to_blocks(raw_blocks, plant, cfg)
    info = dict(plant)
    info.update(posture_meta)
    evaluated = [evaluate_sequence(seq, planner_blocks, info, cfg) for seq in _ALL_SEQUENCES]
    feasible, scalars = select_best_economic(evaluated, cfg)
    return raw_blocks, feasible, [float(s) for s in scalars], evaluated


def _bool_log(row: pd.Series | None, col: str) -> bool:
    if row is None or col not in row.index:
        return False
    value = row.get(col)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return _safe_float(value, 0.0) > 0.5


def _float_log(row: pd.Series | None, col: str, default: float = np.nan) -> float:
    if row is None or col not in row.index:
        return default
    return _safe_float(row.get(col), default)


def _classify_bucket(
    *,
    flip_alpha: float,
    learned_log_action: str,
    learned_raw_action: str,
    learned_replay_action: str,
    oracle_log_action: str,
    oracle_raw_action: str,
    oracle_replay_action: str,
    learned_oracle: tuple[int, float, dict[str, Any]] | None,
    learned_reject_summary: str,
    learned_gap_main: str,
    learned_row: pd.Series,
    oracle_row: pd.Series,
) -> tuple[str, str, str]:
    """Return split_group, primary_cause, and semicolon-separated secondary tags."""
    tags: list[str] = []
    learned_replay_ok = learned_replay_action == learned_raw_action
    oracle_replay_ok = oracle_replay_action == oracle_raw_action
    if not learned_replay_ok or not oracle_replay_ok:
        tags.append("replay_raw_mismatch")
    if learned_log_action != learned_raw_action:
        tags.append("learned_final_postprocess")
    if oracle_log_action != oracle_raw_action:
        tags.append("oracle_final_postprocess")
    if _bool_log(learned_row, "relief_medium_cap_active") or _bool_log(oracle_row, "relief_medium_cap_active"):
        tags.append("relief_medium_cap")
    if _bool_log(learned_row, "far_horizon_relief_gate_active") or _bool_log(oracle_row, "far_horizon_relief_gate_active"):
        tags.append("far_horizon_gate")
    if _bool_log(oracle_row, "far_horizon_hint_any") and not _bool_log(learned_row, "far_horizon_hint_any"):
        tags.append("far_horizon_missing")
    if _bool_log(oracle_row, "far_horizon_hidden_relief") and not _bool_log(learned_row, "far_horizon_hidden_relief"):
        tags.append("hidden_relief_missing")
    if _bool_log(oracle_row, "far_horizon_hidden_intensification") and not _bool_log(learned_row, "far_horizon_hidden_intensification"):
        tags.append("hidden_intensification_missing")
    if (
        _bool_log(learned_row, "prediction_primary_target_reused")
        or _float_log(learned_row, "prediction_primary_target_age_s", 0.0) >= 1200.0
        or _bool_log(learned_row, "sustained_active_recompute_active")
        or _bool_log(learned_row, "active_effectiveness_refresh_active")
    ):
        tags.append("target_lifecycle_sensitive")

    if np.isfinite(flip_alpha):
        split_group = "forecast-flippable"
        if flip_alpha <= 0.30:
            primary = "forecast-flippable"
        else:
            primary = "near-block insufficient"
        return split_group, primary, ";".join(tags)

    split_group = "planner-blocked"
    if not learned_replay_ok or not oracle_replay_ok:
        primary = "replay-inconclusive"
    elif learned_oracle is None or "hard_reject" in learned_reject_summary:
        primary = "candidate-rejected"
    elif oracle_log_action != oracle_raw_action or learned_log_action != learned_raw_action:
        primary = "gate-blocked"
    elif "far_horizon_missing" in tags:
        primary = "far-horizon missing"
    elif "target_lifecycle_sensitive" in tags:
        primary = "target-lifecycle-sensitive"
    elif learned_gap_main != "missing_action":
        primary = "cost-margin-blocked"
    else:
        primary = "replay-inconclusive"
    return split_group, primary, ";".join(tags)


def _best_for_action(feasible: list[dict[str, Any]], scalars: list[float], action: str) -> tuple[int, float, dict[str, Any]] | None:
    for rank, (row, cost) in enumerate(zip(feasible, scalars), start=1):
        if str(row["sequence"][0]) == str(action):
            return rank, float(cost), row
    return None


def _action_reject_summary(evaluated: list[dict[str, Any]], action: str) -> str:
    rows = [r for r in evaluated if str(r["sequence"][0]) == str(action)]
    if not rows:
        return "no_candidate_sequence"
    feasible = [r for r in rows if not r["hard_reject_reason"]]
    if feasible:
        return "feasible_not_selected"
    reasons = Counter(str(r.get("hard_reject_reason", "")) or "unknown" for r in rows)
    medium = Counter(str(r.get("active_medium_gate_reason", "")) for r in rows if r.get("active_medium_gate_reason"))
    reverse = Counter(str(r.get("reverse_reject_reason", "")) for r in rows if r.get("reverse_reject_reason"))
    parts = [f"hard_reject:{reasons.most_common(1)[0][0]}"]
    if medium:
        parts.append(f"medium_gate:{medium.most_common(1)[0][0]}")
    if reverse:
        parts.append(f"reverse_gate:{reverse.most_common(1)[0][0]}")
    return ";".join(parts)


def _gap_summary(
    best_row: dict[str, Any] | None,
    other_row: dict[str, Any] | None,
    cfg: Any,
) -> tuple[str, float, str]:
    if best_row is None or other_row is None:
        return "missing_action", float("nan"), "missing_action"
    best = _weighted_components(best_row, cfg)
    other = _weighted_components(other_row, cfg)
    deltas = {k: other.get(k, 0.0) - best.get(k, 0.0) for k in sorted(set(best) | set(other))}
    positive = {k: v for k, v in deltas.items() if np.isfinite(v) and v > 1e-9}
    if positive:
        main = max(positive, key=positive.get)
        return main, float(positive[main]), ";".join(f"{k}:{v:.3f}" for k, v in sorted(deltas.items()))
    if deltas:
        main = max(deltas, key=lambda k: abs(deltas[k]))
        return main, float(deltas[main]), ";".join(f"{k}:{v:.3f}" for k, v in sorted(deltas.items()))
    return "none", 0.0, ""


def _log_fields(prefix: str, row: pd.Series | None) -> dict[str, Any]:
    if row is None:
        return {}
    cols = [
        "first_action",
        "planner_first_action_raw",
        "best_sequence",
        "best_scalar_cost",
        "relief_medium_cap_active",
        "relief_medium_cap_reason",
        "far_horizon_relief_gate_active",
        "far_horizon_relief_gate_reason",
        "active_medium_gate_reason",
        "prediction_primary_event_reset",
        "prediction_primary_target_refreshed",
        "prediction_primary_target_reused",
        "prediction_primary_target_resumed",
        "prediction_primary_target_age_s",
        "prediction_primary_delta_abs_mean_kg",
        "active_effectiveness_refresh_active",
        "active_effectiveness_refresh_reason",
        "sustained_active_recompute_active",
        "sustained_active_recompute_reason",
        "current_pitch_deg",
        "current_roll_deg",
        "pressure_block0_norm",
        "pressure_block1_norm",
        "pressure_block2_norm",
        "far_horizon_hint_any",
        "far_horizon_hidden_relief",
        "far_horizon_hidden_intensification",
    ]
    out: dict[str, Any] = {}
    for col in cols:
        if col in row.index:
            out[f"{prefix}_{col}"] = row.get(col)
    return out


def _md_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_empty_"
    text = df.copy()
    for col in text.columns:
        text[col] = text[col].map(lambda v: "" if pd.isna(v) else str(v))
    headers = list(text.columns)
    rows = text.values.tolist()
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def run_split(
    learned_run: str,
    out_dir: Path,
    dataset_dir: Path,
    key_buckets: Path,
    flip_alpha_table: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    key_df = pd.read_csv(key_buckets, low_memory=False)
    key_df["case_id"] = key_df["case_id"].map(_case_from_name)
    flip_df = pd.read_csv(flip_alpha_table, low_memory=False)
    flip_lookup = {
        (_case_from_name(str(r["case_id"])), int(r["bucket"])): r
        for _, r in flip_df.iterrows()
    }

    learned_dir = _run_path(RUNS[learned_run])
    oracle_dir = _run_path(RUNS["oracle"])
    cfg, discounts = _cfg_from_casebook(learned_dir)
    replay = Fino1ReplayDataset(dataset_dir=dataset_dir, split="test")
    adapters: dict[str, Any] = {}

    rows: list[dict[str, Any]] = []
    for _, key in key_df.iterrows():
        case = _case_from_name(str(key["case_id"]))
        bucket = int(key["bucket"])
        learned_row = _load_log_row(learned_dir, case, bucket)
        oracle_row = _load_log_row(oracle_dir, case, bucket)
        if learned_row is None or oracle_row is None:
            continue
        ts_row = _load_ts_row(learned_dir, case, bucket, learned_row)
        hist = str(learned_row.get("history_end", "")).strip()
        if not hist:
            continue
        sample = replay.sample_for_history_end(datetime.strptime(hist, TIMESTAMP_FMT))
        if sample is None:
            continue
        plant = _plant_info(learned_row, ts_row)
        plant["forecast_has_future"] = 1
        learned_uv, learned_has_future, _ = _forecast_uv_for_run(
            learned_run,
            sample,
            hist,
            dataset_dir,
            adapters,
        )
        learned_discounts = discounts if learned_has_future else [1.0 for _ in discounts]
        oracle_uv = np.asarray(sample.y_uv_raw, dtype=float)

        _, learned_feasible, learned_scalars, learned_eval = _evaluate_all(
            learned_uv,
            plant,
            cfg,
            learned_discounts,
        )
        _, oracle_feasible, oracle_scalars, oracle_eval = _evaluate_all(
            oracle_uv,
            plant,
            cfg,
            discounts,
        )
        learned_best = learned_feasible[0] if learned_feasible else None
        oracle_best = oracle_feasible[0] if oracle_feasible else None
        learned_replay_action = str(learned_best["sequence"][0]) if learned_best else "no_feasible"
        oracle_replay_action = str(oracle_best["sequence"][0]) if oracle_best else "no_feasible"
        learned_log_action = str(learned_row.get("first_action", ""))
        oracle_log_action = str(oracle_row.get("first_action", ""))
        learned_raw_action = str(learned_row.get("planner_first_action_raw", learned_log_action))
        oracle_raw_action = str(oracle_row.get("planner_first_action_raw", oracle_log_action))

        learned_oracle = _best_for_action(learned_feasible, learned_scalars, oracle_log_action)
        learned_selected = _best_for_action(learned_feasible, learned_scalars, learned_replay_action)
        oracle_oracle = _best_for_action(oracle_feasible, oracle_scalars, oracle_log_action)
        oracle_learned = _best_for_action(oracle_feasible, oracle_scalars, learned_log_action)
        learned_gap_main, learned_gap_value, learned_gap_components = _gap_summary(
            learned_selected[2] if learned_selected else None,
            learned_oracle[2] if learned_oracle else None,
            cfg,
        )
        oracle_gap_main, oracle_gap_value, oracle_gap_components = _gap_summary(
            oracle_oracle[2] if oracle_oracle else None,
            oracle_learned[2] if oracle_learned else None,
            cfg,
        )

        frow = flip_lookup.get((case, bucket))
        flip_alpha = _safe_float(frow.get("flip_alpha")) if frow is not None else np.nan
        learned_reject = _action_reject_summary(learned_eval, oracle_log_action)
        oracle_reject = _action_reject_summary(oracle_eval, learned_log_action)
        split_group, primary_cause, secondary_tags = _classify_bucket(
            flip_alpha=flip_alpha,
            learned_log_action=learned_log_action,
            learned_raw_action=learned_raw_action,
            learned_replay_action=learned_replay_action,
            oracle_log_action=oracle_log_action,
            oracle_raw_action=oracle_raw_action,
            oracle_replay_action=oracle_replay_action,
            learned_oracle=learned_oracle,
            learned_reject_summary=learned_reject,
            learned_gap_main=learned_gap_main,
            learned_row=learned_row,
            oracle_row=oracle_row,
        )

        rec: dict[str, Any] = {
            "case_id": case,
            "bucket": bucket,
            "split_group": split_group,
            "primary_cause": primary_cause,
            "secondary_tags": secondary_tags,
            "flip_alpha": flip_alpha,
            "learned_log_action": learned_log_action,
            "learned_raw_action": learned_raw_action,
            "learned_replay_action": learned_replay_action,
            "oracle_log_action": oracle_log_action,
            "oracle_raw_action": oracle_raw_action,
            "oracle_replay_action": oracle_replay_action,
            "learned_replay_matches_raw": int(learned_replay_action == learned_raw_action),
            "oracle_replay_matches_raw": int(oracle_replay_action == oracle_raw_action),
            "oracle_final_is_postprocessed": int(oracle_log_action != oracle_raw_action),
            "learned_final_is_postprocessed": int(learned_log_action != learned_raw_action),
            "oracle_far_hint_only": int(_bool_log(oracle_row, "far_horizon_hint_any") and not _bool_log(learned_row, "far_horizon_hint_any")),
            "oracle_hidden_relief_only": int(_bool_log(oracle_row, "far_horizon_hidden_relief") and not _bool_log(learned_row, "far_horizon_hidden_relief")),
            "oracle_hidden_intensification_only": int(_bool_log(oracle_row, "far_horizon_hidden_intensification") and not _bool_log(learned_row, "far_horizon_hidden_intensification")),
            "learned_oracle_action_rank": learned_oracle[0] if learned_oracle else np.nan,
            "learned_oracle_action_cost_gap": (
                learned_oracle[1] - learned_scalars[0]
                if learned_oracle and learned_scalars
                else np.nan
            ),
            "oracle_learned_action_rank": oracle_learned[0] if oracle_learned else np.nan,
            "oracle_learned_action_cost_gap": (
                oracle_learned[1] - oracle_scalars[0]
                if oracle_learned and oracle_scalars
                else np.nan
            ),
            "learned_oracle_action_reject_summary": learned_reject,
            "oracle_learned_action_reject_summary": oracle_reject,
            "learned_why_oracle_action_loses_main_component": learned_gap_main,
            "learned_why_oracle_action_loses_component_value": learned_gap_value,
            "learned_why_oracle_action_loses_components": learned_gap_components,
            "oracle_why_learned_action_loses_main_component": oracle_gap_main,
            "oracle_why_learned_action_loses_component_value": oracle_gap_value,
            "oracle_why_learned_action_loses_components": oracle_gap_components,
            "is_target_reused_in_learned": int(_safe_float(learned_row.get("prediction_primary_target_reused"), 0.0) > 0.5),
            "is_target_refreshed_in_learned": int(_safe_float(learned_row.get("prediction_primary_target_refreshed"), 0.0) > 0.5),
            "learned_target_age_s": _safe_float(learned_row.get("prediction_primary_target_age_s")),
        }
        rec.update(_log_fields("learned", learned_row))
        rec.update(_log_fields("oracle", oracle_row))
        rows.append(rec)

    out = pd.DataFrame(rows)
    out.to_csv(out_dir / "key_bucket_failure_split.csv", index=False)

    split_counts = out.groupby(["split_group", "primary_cause"]).size().reset_index(name="count")
    case_counts = out.groupby(["case_id", "split_group"]).size().reset_index(name="count")
    main_components = (
        out.groupby(["split_group", "learned_why_oracle_action_loses_main_component"])
        .size()
        .reset_index(name="count")
    )
    flippable = out[out["split_group"].eq("forecast-flippable")]
    blocked = out[out["split_group"].eq("planner-blocked")]
    lines = [
        "# G001 Key Bucket Failure Split",
        "",
        "Analysis-only. No model training and no controller changes.",
        "",
        "## Classification Scope",
        "",
        "- The input 18 buckets are full oracle-vs-learned `first_action` disagreements.",
        "- `forecast-flippable` means: with the learned-run bucket state, candidate set, gates, and target/pump state held fixed, linearly interpolating learned UV toward oracle UV eventually selects the oracle final `first_action`.",
        "- `planner-blocked` means: the same interpolation still does not select the oracle final `first_action` by alpha=1.0. This is a near-block/forecast-path diagnosis, not a claim that full oracle closed-loop cannot choose that action.",
        "- `replay-inconclusive` marks buckets where the offline replay does not reproduce the logged raw planner action; cost component attribution for those buckets should be treated as low confidence.",
        "",
        "## Summary",
        "",
        f"- key buckets analyzed: {len(out)}",
        f"- forecast-flippable: {len(flippable)}",
        f"- planner-blocked: {len(blocked)}",
        f"- median flip_alpha among flippable: {flippable['flip_alpha'].median():.3f}" if not flippable.empty else "- median flip_alpha among flippable: n/a",
        "",
        "## Split Counts",
        _md_table(split_counts),
        "",
        "## Cause Counts",
        _md_table(out["primary_cause"].value_counts().rename_axis("primary_cause").reset_index(name="count")),
        "",
        "## Case Distribution",
        _md_table(case_counts),
        "",
        "## Main Cost Component Blocking Oracle Action Under Learned Forecast",
        _md_table(main_components),
        "",
        "## Interpretation",
    ]
    flippable_share = len(flippable) / max(len(out), 1)
    clean_forecast = int(out["primary_cause"].eq("forecast-flippable").sum())
    near_insufficient = int(out["primary_cause"].eq("near-block insufficient").sum())
    gate_like = int(out["primary_cause"].isin(["gate-blocked", "candidate-rejected", "cost-margin-blocked"]).sum())
    lifecycle_like = int(out["primary_cause"].eq("target-lifecycle-sensitive").sum())
    if flippable_share >= 0.5:
        lines.append("- Forecast-side work is still relevant, but only if it optimizes planner cost/rank; plain shape-MSE has already shown weak action transfer.")
    if gate_like >= max(clean_forecast + near_insufficient, 1):
        lines.append("- Gate/candidate/cost-interface effects are at least as large as clean forecast-flippable space; G003-style default-off probes are justified after this diagnosis.")
    if lifecycle_like:
        lines.append("- Some buckets are target-lifecycle-sensitive; if these dominate closed-loop gains/losses, stop training/probing and write the lifecycle conclusion instead.")
    if out["secondary_tags"].astype(str).str.contains("far_horizon_missing", regex=False).sum():
        lines.append("- Far-horizon missing appears as a secondary tag, but it should not be treated as sufficient until it overlaps with action-rank movement.")
    lines.extend(
        [
            "",
            "## Recommended Gate",
            "",
            f"- clean forecast-flippable buckets: {clean_forecast}",
            f"- near-block insufficient buckets: {near_insufficient}",
            f"- gate/candidate/cost-interface buckets: {gate_like}",
            f"- target-lifecycle-sensitive buckets: {lifecycle_like}",
            "- Do not enter G002 or G003 automatically; use the counts above and the per-bucket CSV to select at most one forecast-side and one controller-interface candidate.",
            "",
            "## Outputs",
            "- `key_bucket_failure_split.csv`",
            "- `key_bucket_failure_split.md`",
        ]
    )
    (out_dir / "key_bucket_failure_split.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {out_dir / 'key_bucket_failure_split.md'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--learned-run", default="near_block_eventbalanced_v2")
    parser.add_argument("--dataset-dir", default=DEFAULT_DATASET_DIR)
    parser.add_argument("--key-buckets", default=KEY_BUCKETS)
    parser.add_argument(
        "--flip-alpha-table",
        default="outputs/wind_prediction/cost_margin_flip_alpha_v1_near_block_eventbalanced_v2/cost_margin_flip_alpha_table.csv",
    )
    parser.add_argument("--out-dir", default="outputs/wind_prediction/key_bucket_failure_split_v1")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_split(
        learned_run=str(args.learned_run),
        out_dir=_run_path(args.out_dir),
        dataset_dir=_run_path(args.dataset_dir),
        key_buckets=_run_path(args.key_buckets),
        flip_alpha_table=_run_path(args.flip_alpha_table),
    )


if __name__ == "__main__":
    main()
