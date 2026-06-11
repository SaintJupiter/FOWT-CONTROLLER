#!/usr/bin/env python3
"""Cost-margin interpolation audit for oracle-vs-learned key buckets.

Evaluation-only. For each key bucket, replay planner candidate costs while
linearly interpolating the forecasted UV sequence from learned to oracle:

    uv(alpha) = (1 - alpha) * learned_uv + alpha * oracle_uv

The first alpha where the selected first_action equals the oracle first_action
is reported as flip_alpha. This answers whether forecast-side improvements are
close to crossing the planner's discrete action threshold.
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
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "analysis"))

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


def _best_by_action(feasible: list[dict[str, Any]], scalars: list[float]) -> dict[str, tuple[int, float, str]]:
    out: dict[str, tuple[int, float, str]] = {}
    for rank, (item, cost) in enumerate(zip(feasible, scalars), start=1):
        action = str(item["sequence"][0])
        out.setdefault(action, (rank, float(cost), ">".join(item["sequence"])))
    return out


def _evaluate_uv(
    uv: np.ndarray,
    plant: dict[str, Any],
    cfg: Any,
    discounts: list[float],
) -> tuple[str, float, dict[str, tuple[int, float, str]], list[dict[str, Any]], list[float]]:
    raw_blocks = compute_pressure_blocks(np.asarray(uv, dtype=float), discounts, cfg)
    planner_blocks, posture_meta = apply_posture_state_to_blocks(raw_blocks, plant, cfg)
    info = dict(plant)
    info.update(posture_meta)
    evaluated = [evaluate_sequence(seq, planner_blocks, info, cfg) for seq in _ALL_SEQUENCES]
    feasible, scalars = select_best_economic(evaluated, cfg)
    if not feasible:
        return "no_feasible", float("nan"), {}, feasible, scalars
    return str(feasible[0]["sequence"][0]), float(scalars[0]), _best_by_action(feasible, scalars), feasible, scalars


def _interp_uv(learned_uv: np.ndarray, oracle_uv: np.ndarray, alpha: float) -> np.ndarray:
    n = min(len(learned_uv), len(oracle_uv))
    return (1.0 - alpha) * learned_uv[:n] + alpha * oracle_uv[:n]


def run_audit(
    learned_run: str,
    out_dir: Path,
    alpha_step: float,
    dataset_dir: Path,
    key_buckets: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    key_df = pd.read_csv(key_buckets, low_memory=False)
    if key_df.empty:
        raise SystemExit(f"empty key bucket file: {key_buckets}")
    key_df["case_id"] = key_df["case_id"].map(_case_from_name)

    learned_dir = _run_path(RUNS[learned_run])
    oracle_dir = _run_path(RUNS["oracle"])
    cfg, discounts = _cfg_from_casebook(learned_dir)
    replay = Fino1ReplayDataset(dataset_dir=dataset_dir, split="test")
    adapters: dict[str, Any] = {}

    alpha_values = np.round(np.arange(0.0, 1.0 + 1e-9, alpha_step), 6)
    rows: list[dict[str, Any]] = []
    alpha_trace_rows: list[dict[str, Any]] = []

    for _, key in key_df.iterrows():
        case = _case_from_name(str(key["case_id"]))
        bucket = int(key["bucket"])
        learned_log_path = next((learned_dir / "planner_logs").glob(f"*{case}*_planner_log.csv"), None)
        learned_ts_path = next((learned_dir / "timeseries").glob(f"*{case}*_timeseries.csv"), None)
        oracle_log_path = next((oracle_dir / "planner_logs").glob(f"*{case}*_planner_log.csv"), None)
        if learned_log_path is None or learned_ts_path is None or oracle_log_path is None:
            continue
        learned_log = pd.read_csv(learned_log_path, low_memory=False)
        oracle_log = pd.read_csv(oracle_log_path, low_memory=False)
        learned_hit = learned_log[learned_log["bucket"].astype(int).eq(bucket)]
        oracle_hit = oracle_log[oracle_log["bucket"].astype(int).eq(bucket)]
        if learned_hit.empty or oracle_hit.empty:
            continue
        learned_row = learned_hit.iloc[0]
        oracle_row = oracle_hit.iloc[0]
        ts = pd.read_csv(learned_ts_path, low_memory=False)
        ts_row = _nearest_ts(ts, _safe_float(learned_row.get("current_time_s"), bucket * 600.0))
        hist = str(learned_row.get("history_end", "")).strip()
        if not hist:
            continue
        sample = replay.sample_for_history_end(datetime.strptime(hist, TIMESTAMP_FMT))
        if sample is None:
            continue
        plant = _plant_info(learned_row, ts_row)
        plant["forecast_has_future"] = 1
        learned_uv, learned_has_future, learned_version = _forecast_uv_for_run(
            learned_run,
            sample,
            hist,
            dataset_dir,
            adapters,
        )
        oracle_uv = np.asarray(sample.y_uv_raw, dtype=float)
        learned_discounts = discounts if learned_has_future else [1.0 for _ in discounts]

        learned_best, learned_best_cost, learned_by_action, _, _ = _evaluate_uv(
            learned_uv,
            plant,
            cfg,
            learned_discounts,
        )
        oracle_best, oracle_best_cost, oracle_by_action, _, _ = _evaluate_uv(
            oracle_uv,
            plant,
            cfg,
            discounts,
        )
        oracle_action = str(oracle_row.get("first_action", key.get("oracle_first_action", "")))
        learned_action = str(learned_row.get("first_action", key.get("learned_first_action", "")))
        learned_oracle_tuple = learned_by_action.get(oracle_action)
        oracle_oracle_tuple = oracle_by_action.get(oracle_action)
        flip_alpha = np.nan
        nearest_alpha = np.nan
        min_gap = np.inf
        last_action = ""
        first_flip_action = ""
        for alpha in alpha_values:
            action, best_cost, by_action, _, _ = _evaluate_uv(
                _interp_uv(learned_uv, oracle_uv, float(alpha)),
                plant,
                cfg,
                discounts,
            )
            oracle_tuple = by_action.get(oracle_action)
            gap = float(oracle_tuple[1] - best_cost) if oracle_tuple else np.nan
            rank = float(oracle_tuple[0]) if oracle_tuple else np.nan
            alpha_trace_rows.append(
                {
                    "case_id": case,
                    "bucket": bucket,
                    "alpha": float(alpha),
                    "selected_action": action,
                    "oracle_action": oracle_action,
                    "oracle_action_rank": rank,
                    "oracle_action_cost_gap": gap,
                }
            )
            if np.isfinite(gap) and abs(gap) < min_gap:
                min_gap = abs(gap)
                nearest_alpha = float(alpha)
            if np.isnan(flip_alpha) and action == oracle_action:
                flip_alpha = float(alpha)
                first_flip_action = action
            last_action = action

        rows.append(
            {
                "case_id": case,
                "bucket": bucket,
                "learned_run": learned_run,
                "learned_log_action": learned_action,
                "oracle_log_action": oracle_action,
                "learned_replay_best_action": learned_best,
                "oracle_replay_best_action": oracle_best,
                "flip_alpha": flip_alpha,
                "nearest_margin_alpha": nearest_alpha,
                "nearest_abs_gap": min_gap,
                "final_alpha_action": last_action,
                "first_flip_action": first_flip_action,
                "learned_oracle_action_rank": learned_oracle_tuple[0] if learned_oracle_tuple else np.nan,
                "learned_oracle_action_cost_gap": (
                    learned_oracle_tuple[1] - learned_best_cost
                    if learned_oracle_tuple and np.isfinite(learned_best_cost)
                    else np.nan
                ),
                "oracle_oracle_action_rank": oracle_oracle_tuple[0] if oracle_oracle_tuple else np.nan,
                "oracle_oracle_action_cost_gap": (
                    oracle_oracle_tuple[1] - oracle_best_cost
                    if oracle_oracle_tuple and np.isfinite(oracle_best_cost)
                    else np.nan
                ),
                "learned_forecast_version": learned_version,
                "learned_forecast_has_future": int(bool(learned_has_future)),
            }
        )

    result = pd.DataFrame(rows)
    trace = pd.DataFrame(alpha_trace_rows)
    result.to_csv(out_dir / "cost_margin_flip_alpha_table.csv", index=False)
    trace.to_csv(out_dir / "cost_margin_alpha_trace.csv", index=False)

    finite = result["flip_alpha"].dropna()
    no_flip = int(result["flip_alpha"].isna().sum()) if not result.empty else 0
    lines = [
        "# Cost-Margin Flip Alpha Audit v1",
        "",
        f"learned_run: `{learned_run}`",
        "",
        "## Summary",
        "",
        f"- key buckets: {len(result)}",
        f"- flipped to oracle action by alpha=1.0: {len(finite)}/{len(result)}",
        f"- no flip by alpha=1.0: {no_flip}/{len(result)}",
    ]
    if len(finite):
        lines.extend(
            [
                f"- median flip_alpha: {float(finite.median()):.3f}",
                f"- p75 flip_alpha: {float(finite.quantile(0.75)):.3f}",
                f"- p90 flip_alpha: {float(finite.quantile(0.90)):.3f}",
            ]
        )
    if not result.empty:
        nearest = result["nearest_abs_gap"].replace([np.inf, -np.inf], np.nan)
        lines.append(f"- mean nearest_abs_gap along interpolation: {float(nearest.mean()):.3f}")
    lines.extend(
        [
            "",
            "## Decision Heuristic",
            "",
            "- median flip_alpha < 0.3: forecast-side cost-aware loss may still have room.",
            "- median flip_alpha > 0.7 or many no-flip buckets: shape-only forecast is unlikely to cross planner thresholds.",
            "",
            "## Outputs",
            "",
            "- `cost_margin_flip_alpha_table.csv`",
            "- `cost_margin_alpha_trace.csv`",
        ]
    )
    (out_dir / "cost_margin_flip_alpha_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {out_dir / 'cost_margin_flip_alpha_summary.md'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--learned-run", default="near_block_eventbalanced_v2", choices=sorted(set(RUNS) - {"oracle"}))
    parser.add_argument("--out-dir", default="outputs/wind_prediction/cost_margin_flip_alpha_v1")
    parser.add_argument("--alpha-step", type=float, default=0.05)
    parser.add_argument("--dataset-dir", default=DEFAULT_DATASET_DIR)
    parser.add_argument("--key-buckets", default=KEY_BUCKETS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_audit(
        learned_run=str(args.learned_run),
        out_dir=_run_path(args.out_dir),
        alpha_step=float(args.alpha_step),
        dataset_dir=_run_path(args.dataset_dir),
        key_buckets=_run_path(args.key_buckets),
    )


if __name__ == "__main__":
    main()
