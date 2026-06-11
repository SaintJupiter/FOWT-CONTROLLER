#!/usr/bin/env python3
"""Summarize h240_f120 forecast-loss experiments.

This is an evaluation-only helper. It compares existing closed-loop run
directories, planner-log far-horizon hints, and oracle-action attribution
without changing controller code or planner parameters.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


CASES = [
    "fr_relief_01",
    "fr_relief_09",
    "sf_holdout_02",
    "b_decay_strong",
    "b_signflip_fallback",
    "b_high_pressure_event",
    "b_residual_high",
    "lowrisk_quiet",
    "lowrisk_clean",
    "lowrisk_random_03",
]


DEFAULT_RUNS = {
    "pp_f60_relief_e15": "outputs/wind_prediction/pp_learned_relief_e15_guard10",
    "h240_f120_baseline": "outputs/wind_prediction/horizon_guard10_fino1_h240_f120_learned_v1",
    "h240_f120_relief_e15": "outputs/wind_prediction/pp_h240_f120_relief_e15_farlog_guard10_learned_v1",
    "h240_f120_anti_persistence": "outputs/wind_prediction/pp_h240_f120_anti_persistence_guard10_learned_v1",
    "h240_f120_synth_intensify": "outputs/wind_prediction/pp_h240_f120_synth_intensify_guard10_learned_v1",
    "h240_f120_multitask_far_hint_v0": (
        "outputs/wind_prediction/pp_h240_f120_multitask_far_hint_v0_guard10_learned_v1"
    ),
    "h240_f120_pressure_shape_v1": (
        "outputs/wind_prediction/pp_h240_f120_pressure_shape_supervision_v1_guard10_learned_v1"
    ),
    "h240_f120_planner_block_shape_v1": (
        "outputs/wind_prediction/pp_h240_f120_planner_block_shape_supervision_v1_guard10_learned_v1"
    ),
    "h240_f120_oracle": "outputs/wind_prediction/pp_h240_f120_oracle_farlog_guard10_v1",
    "h240_f120_persistence": "outputs/wind_prediction/pp_h240_f120_persistence_farlog_guard10_v1",
    "h240_f120_current_only": "outputs/wind_prediction/pp_h240_f120_current_only_farlog_guard10_v1",
}


def _run_dir(root: Path, value: str) -> Path:
    p = Path(value)
    return p if p.is_absolute() else root / p


def _case_from_name(name: str) -> str:
    for case in CASES:
        if case in name:
            return case
    return name.split("_20", 1)[0]


def _case_key(value: Any) -> str:
    text = str(value)
    for case in CASES:
        if case in text:
            return case
    return text


def _fallback_col(ts: pd.DataFrame) -> str | None:
    return next(
        (c for c in ts.columns if "safety_fallback" in c.lower() or "fallback_active" in c.lower()),
        None,
    )


def _safe_nanmean(values: list[float]) -> float:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0 or np.all(np.isnan(arr)):
        return float("nan")
    return float(np.nanmean(arr))


def _nan_range(values: list[float]) -> float:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0 or np.all(np.isnan(arr)):
        return float("nan")
    return float(np.nanmax(arr) - np.nanmin(arr))


def _timeseries_metrics(run_dir: Path) -> pd.DataFrame:
    ts_dir = run_dir / "timeseries"
    rows: list[dict[str, Any]] = []
    if not ts_dir.exists():
        return pd.DataFrame(rows)
    for path in sorted(ts_dir.glob("*timeseries.csv")):
        case = _case_from_name(path.name)
        ts = pd.read_csv(path, low_memory=False)
        if "pitch_deg" not in ts.columns:
            continue
        pitch = ts["pitch_deg"].abs()
        roll = ts["roll_deg"].abs() if "roll_deg" in ts.columns else pd.Series(0.0, index=ts.index)
        pump_col = "pump_total_rate_m3_min"
        pump = float((ts[pump_col].abs() / 60.0).sum()) if pump_col in ts.columns else float("nan")
        fb_col = _fallback_col(ts)
        fb = float(ts[fb_col].mean()) if fb_col else 0.0
        rows.append(
            {
                "case_id": case,
                "pump_m3": pump,
                "pitch_p95": float(np.percentile(pitch, 95)),
                "roll_p95": float(np.percentile(roll, 95)),
                "fallback_ratio": fb,
                "time_over_3_s": int(((pitch > 3.0) | (roll > 3.0)).sum()),
                "time_over_5_s": int(((pitch > 5.0) | (roll > 5.0)).sum()),
            }
        )
    return pd.DataFrame(rows)


def _casebook_metrics(run_dir: Path) -> pd.DataFrame:
    path = run_dir / "casebook_summary.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    if "case_id" not in df.columns:
        return pd.DataFrame()
    cols = ["case_id"]
    for col in ["primary_pump_work_m3", "primary_pitch_p95", "primary_roll_p95", "primary_safety_fallback_ratio"]:
        if col in df.columns:
            cols.append(col)
    return df[cols].copy()


def _planner_log_metrics(run_dir: Path, oracle_key: pd.DataFrame | None = None) -> dict[str, Any]:
    log_dir = run_dir / "planner_logs"
    rows = []
    if not log_dir.exists():
        return {}
    key_pairs = set()
    if oracle_key is not None and not oracle_key.empty:
        key_pairs = set(zip(oracle_key["case_id"].map(_case_key), oracle_key["bucket"].astype(int)))
    key_total = len(key_pairs)
    key_match = 0
    key_available = 0
    for path in sorted(log_dir.glob("*planner_log.csv")):
        case = _case_from_name(path.name)
        d = pd.read_csv(path, low_memory=False)
        for _, row in d.iterrows():
            bucket = int(row.get("bucket", -1))
            first_action = str(row.get("first_action", ""))
            rows.append(
                {
                    "case_id": case,
                    "bucket": bucket,
                    "first_action": first_action,
                    "far_horizon_hint_any": int(row.get("far_horizon_hint_any", 0)),
                    "far_horizon_hidden_relief": int(row.get("far_horizon_hidden_relief", 0)),
                    "far_horizon_hidden_intensification": int(row.get("far_horizon_hidden_intensification", 0)),
                    "far_horizon_reversal": int(row.get("far_horizon_reversal", 0)),
                    "far_horizon_direction_shift": int(row.get("far_horizon_direction_shift", 0)),
                    "far_horizon_available": int(row.get("far_horizon_available", 0)),
                    "far_horizon_norm_0_20": float(row.get("far_horizon_norm_0_20", np.nan)),
                    "far_horizon_norm_20_40": float(row.get("far_horizon_norm_20_40", np.nan)),
                    "far_horizon_norm_40_60": float(row.get("far_horizon_norm_40_60", np.nan)),
                    "far_horizon_norm_60_80": float(row.get("far_horizon_norm_60_80", np.nan)),
                    "far_horizon_norm_80_100": float(row.get("far_horizon_norm_80_100", np.nan)),
                    "far_horizon_norm_100_120": float(row.get("far_horizon_norm_100_120", np.nan)),
                }
            )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return {}
    if key_pairs and oracle_key is not None:
        oracle_actions = {
            (_case_key(r.case_id), int(r.bucket)): str(r.oracle_first_action)
            for r in oracle_key.itertuples(index=False)
        }
        for r in frame.itertuples(index=False):
            pair = (str(r.case_id), int(r.bucket))
            if pair in key_pairs:
                key_available += 1
                if str(r.first_action) == oracle_actions[pair]:
                    key_match += 1
    near_cols = ["far_horizon_norm_0_20", "far_horizon_norm_20_40", "far_horizon_norm_40_60"]
    far_cols = ["far_horizon_norm_60_80", "far_horizon_norm_80_100", "far_horizon_norm_100_120"]
    current = frame["far_horizon_norm_0_20"].to_numpy(dtype=float)
    def _direction_agree(row: pd.Series) -> float:
        c = float(row["far_horizon_norm_0_20"])
        vals = [float(row[col]) for col in near_cols[1:] + far_cols]
        deltas = np.asarray(vals, dtype=float) - c
        return float(np.sign(deltas[-1]))
    return {
        "buckets": int(len(frame)),
        "far_horizon_available": int(frame["far_horizon_available"].sum()),
        "far_horizon_hint_any": int(frame["far_horizon_hint_any"].sum()),
        "hidden_relief": int(frame["far_horizon_hidden_relief"].sum()),
        "hidden_intensification": int(frame["far_horizon_hidden_intensification"].sum()),
        "reversal": int(frame["far_horizon_reversal"].sum()),
        "direction_shift": int(frame["far_horizon_direction_shift"].sum()),
        "oracle_diff_key_buckets": int(key_total),
        "oracle_diff_key_available": int(key_available),
        "oracle_diff_key_action_matches": int(key_match),
        "near_range_mean": float((frame[near_cols].max(axis=1) - frame[near_cols].min(axis=1)).mean()),
        "far_range_mean": float((frame[far_cols].max(axis=1) - frame[far_cols].min(axis=1)).mean()),
        "far_over_near_mean": float(
            (frame[far_cols].mean(axis=1) / frame[near_cols].mean(axis=1).clip(lower=1e-9)).mean()
        ),
        "last_delta_sign_mean": _safe_nanmean([_direction_agree(row) for _, row in frame.iterrows()]),
    }


def _collect_planner_blocks(run_dir: Path) -> pd.DataFrame:
    log_dir = run_dir / "planner_logs"
    rows: list[dict[str, Any]] = []
    if not log_dir.exists():
        return pd.DataFrame(rows)
    for path in sorted(log_dir.glob("*planner_log.csv")):
        case = _case_from_name(path.name)
        d = pd.read_csv(path, low_memory=False)
        for _, row in d.iterrows():
            rows.append(
                {
                    "case_id": case,
                    "bucket": int(row.get("bucket", -1)),
                    "first_action": str(row.get("first_action", "")),
                    "b0": float(row.get("pressure_block0_norm", np.nan)),
                    "b1": float(row.get("pressure_block1_norm", np.nan)),
                    "b2": float(row.get("pressure_block2_norm", np.nan)),
                    "raw_b0": float(row.get("raw_pressure_block0_norm", np.nan)),
                    "raw_b1": float(row.get("raw_pressure_block1_norm", np.nan)),
                    "raw_b2": float(row.get("raw_pressure_block2_norm", np.nan)),
                }
            )
    return pd.DataFrame(rows)


def _near_block_shape_metrics(
    run_name: str,
    run_dir: Path,
    oracle_dir: Path,
    baseline_dir: Path | None = None,
    oracle_key: pd.DataFrame | None = None,
) -> dict[str, Any]:
    run = _collect_planner_blocks(run_dir)
    oracle = _collect_planner_blocks(oracle_dir)
    if run.empty or oracle.empty:
        return {}
    merged = run.merge(
        oracle[["case_id", "bucket", "first_action", "b0", "b1", "b2"]].rename(
            columns={
                "first_action": "oracle_action",
                "b0": "oracle_b0",
                "b1": "oracle_b1",
                "b2": "oracle_b2",
            }
        ),
        on=["case_id", "bucket"],
        how="inner",
    )
    block_cols = ["b0", "b1", "b2"]
    oracle_cols = ["oracle_b0", "oracle_b1", "oracle_b2"]
    pred = merged[block_cols].to_numpy(dtype=float)
    true = merged[oracle_cols].to_numpy(dtype=float)
    valid = np.isfinite(pred).all(axis=1) & np.isfinite(true).all(axis=1)
    merged = merged.loc[valid].copy()
    pred = pred[valid]
    true = true[valid]
    if pred.size == 0:
        return {}
    err = pred - true
    pred_delta = np.diff(pred, axis=1)
    true_delta = np.diff(true, axis=1)
    true_delta_abs = np.abs(true_delta)
    direction_mask = true_delta_abs >= 0.05
    if direction_mask.any():
        direction_agreement = float(
            (np.sign(pred_delta[direction_mask]) == np.sign(true_delta[direction_mask])).mean()
        )
    else:
        direction_agreement = float("nan")
    pred_range = np.nanmax(pred, axis=1) - np.nanmin(pred, axis=1)
    true_range = np.nanmax(true, axis=1) - np.nanmin(true, axis=1)
    range_mae = float(np.nanmean(np.abs(pred_range - true_range)))
    range_ratio = float(
        np.nanmean(pred_range / np.clip(true_range, 1e-9, None))
    )
    key_shape_closer_than_baseline = np.nan
    key_action_matches = np.nan
    key_count = 0
    if oracle_key is not None and not oracle_key.empty:
        key_pairs = set(zip(oracle_key["case_id"].map(_case_key), oracle_key["bucket"].astype(int)))
        key_mask = merged.apply(lambda r: (str(r["case_id"]), int(r["bucket"])) in key_pairs, axis=1)
        key = merged.loc[key_mask].copy()
        key_count = int(len(key))
        if key_count:
            key_pred = key[block_cols].to_numpy(dtype=float)
            key_true = key[oracle_cols].to_numpy(dtype=float)
            key_dist = np.linalg.norm(key_pred - key_true, axis=1)
            if baseline_dir is not None:
                base = _collect_planner_blocks(baseline_dir)
                if not base.empty:
                    base = base.rename(columns={"b0": "base_b0", "b1": "base_b1", "b2": "base_b2"})
                    key2 = key.merge(
                        base[["case_id", "bucket", "base_b0", "base_b1", "base_b2"]],
                        on=["case_id", "bucket"],
                        how="left",
                    )
                    base_pred = key2[["base_b0", "base_b1", "base_b2"]].to_numpy(dtype=float)
                    finite = np.isfinite(base_pred).all(axis=1)
                    if finite.any():
                        base_dist = np.linalg.norm(base_pred[finite] - key_true[finite], axis=1)
                        key_shape_closer_than_baseline = float((key_dist[finite] < base_dist).sum())
            key_action_matches = float((key["first_action"].astype(str) == key["oracle_action"].astype(str)).sum())
    return {
        "run": run_name,
        "near_block_b012_mae": float(np.nanmean(np.abs(err))),
        "near_block_b012_rmse": float(np.sqrt(np.nanmean(err ** 2))),
        "near_direction_agreement": direction_agreement,
        "near_range_mae": range_mae,
        "near_range_ratio": range_ratio,
        "oracle_diff_key_buckets": int(key_count),
        "oracle_diff_key_shape_closer_than_baseline": key_shape_closer_than_baseline,
        "oracle_diff_key_action_matches": key_action_matches,
    }


def _planner_key_bucket_rows(run_name: str, run_dir: Path, oracle_key: pd.DataFrame) -> list[dict[str, Any]]:
    """Return one long-format row per oracle!=learned key bucket for a run."""
    log_dir = run_dir / "planner_logs"
    if not log_dir.exists() or oracle_key.empty:
        return []
    by_pair: dict[tuple[str, int], pd.Series] = {}
    for path in sorted(log_dir.glob("*planner_log.csv")):
        case = _case_from_name(path.name)
        d = pd.read_csv(path, low_memory=False)
        for _, row in d.iterrows():
            by_pair[(case, int(row.get("bucket", -1)))] = row

    out: list[dict[str, Any]] = []
    for row in oracle_key.itertuples(index=False):
        case = _case_key(getattr(row, "case_id"))
        bucket = int(getattr(row, "bucket"))
        oracle_action = str(getattr(row, "oracle_first_action"))
        r = by_pair.get((case, bucket))
        if r is None:
            continue
        near_vals = [
            float(r.get("far_horizon_norm_0_20", np.nan)),
            float(r.get("far_horizon_norm_20_40", np.nan)),
            float(r.get("far_horizon_norm_40_60", np.nan)),
        ]
        far_vals = [
            float(r.get("far_horizon_norm_60_80", np.nan)),
            float(r.get("far_horizon_norm_80_100", np.nan)),
            float(r.get("far_horizon_norm_100_120", np.nan)),
        ]
        first_action = str(r.get("first_action", ""))
        out.append(
            {
                "run": run_name,
                "case_id": case,
                "bucket": bucket,
                "oracle_first_action": oracle_action,
                "first_action": first_action,
                "matches_oracle_action": int(first_action == oracle_action),
                "pressure_block0_norm": float(r.get("pressure_block0_norm", np.nan)),
                "pressure_block1_norm": float(r.get("pressure_block1_norm", np.nan)),
                "pressure_block2_norm": float(r.get("pressure_block2_norm", np.nan)),
                "far_horizon_available": int(r.get("far_horizon_available", 0)),
                "far_horizon_hint_any": int(r.get("far_horizon_hint_any", 0)),
                "hidden_relief": int(r.get("far_horizon_hidden_relief", 0)),
                "hidden_intensification": int(r.get("far_horizon_hidden_intensification", 0)),
                "reversal": int(r.get("far_horizon_reversal", 0)),
                "direction_shift": int(r.get("far_horizon_direction_shift", 0)),
                "norm_0_20": near_vals[0],
                "norm_20_40": near_vals[1],
                "norm_40_60": near_vals[2],
                "norm_60_80": far_vals[0],
                "norm_80_100": far_vals[1],
                "norm_100_120": far_vals[2],
                "near_range": _nan_range(near_vals),
                "far_range": _nan_range(far_vals),
                "far_over_near_weighted": float(r.get("far_horizon_far_over_near_weighted", np.nan)),
            }
        )
    return out


def _aggregate_case_metrics(df: pd.DataFrame) -> dict[str, Any]:
    if df.empty:
        return {}
    return {
        "sum_pump": float(df["pump_m3"].sum()),
        "max_p95": float(df["pitch_p95"].max()),
        "sum_fb": float(df["fallback_ratio"].sum()),
        "time_over_3": int(df["time_over_3_s"].sum()),
        "time_over_5": int(df["time_over_5_s"].sum()),
    }


def _markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_none_"
    lines = [
        "| " + " | ".join(map(str, df.columns)) + " |",
        "| " + " | ".join(["---"] * len(df.columns)) + " |",
    ]
    for _, row in df.iterrows():
        vals = []
        for value in row:
            if isinstance(value, float):
                vals.append(f"{value:.3f}")
            else:
                vals.append(str(value))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default="outputs/wind_prediction/h240_f120_forecast_loss_experiment_v1")
    parser.add_argument(
        "--oracle-key",
        default="outputs/wind_prediction/h240_f120_oracle_learned_attribution_v1/oracle_learned_action_diff_buckets.csv",
    )
    for name, path in DEFAULT_RUNS.items():
        parser.add_argument(f"--run-{name}", default=path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[2]
    out_dir = _run_dir(root, args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    oracle_key_path = _run_dir(root, args.oracle_key)
    oracle_key = pd.read_csv(oracle_key_path) if oracle_key_path.exists() else pd.DataFrame()
    summary_rows = []
    case_rows = []
    attr_rows = []
    near_shape_rows = []
    key_bucket_rows: list[dict[str, Any]] = []
    oracle_run_dir = _run_dir(root, getattr(args, "run_h240_f120_oracle"))
    baseline_run_dir = _run_dir(root, getattr(args, "run_h240_f120_baseline"))
    for name in DEFAULT_RUNS:
        run_dir = _run_dir(root, getattr(args, f"run_{name}"))
        case_df = _timeseries_metrics(run_dir)
        if case_df.empty:
            cb = _casebook_metrics(run_dir)
            if not cb.empty:
                case_df = cb.rename(
                    columns={
                        "primary_pump_work_m3": "pump_m3",
                        "primary_pitch_p95": "pitch_p95",
                        "primary_roll_p95": "roll_p95",
                        "primary_safety_fallback_ratio": "fallback_ratio",
                    }
                )
                case_df["time_over_3_s"] = np.nan
                case_df["time_over_5_s"] = np.nan
        if not case_df.empty:
            case_df.insert(0, "run", name)
            case_rows.append(case_df)
            row = {"run": name, "run_dir": str(run_dir), **_aggregate_case_metrics(case_df)}
            summary_rows.append(row)
        attr = _planner_log_metrics(run_dir, oracle_key)
        if attr:
            attr_rows.append({"run": name, **attr})
        near_shape = _near_block_shape_metrics(
            name,
            run_dir,
            oracle_run_dir,
            baseline_run_dir,
            oracle_key,
        )
        if near_shape:
            near_shape_rows.append(near_shape)
        key_bucket_rows.extend(_planner_key_bucket_rows(name, run_dir, oracle_key))

    summary = pd.DataFrame(summary_rows)
    cases = pd.concat(case_rows, ignore_index=True) if case_rows else pd.DataFrame()
    attribution = pd.DataFrame(attr_rows)
    near_shape = pd.DataFrame(near_shape_rows)
    key_buckets = pd.DataFrame(key_bucket_rows)
    summary.to_csv(out_dir / "closed_loop_summary.csv", index=False)
    cases.to_csv(out_dir / "case_comparison.csv", index=False)
    attribution.to_csv(out_dir / "forecast_control_attribution_summary.csv", index=False)
    near_shape.to_csv(out_dir / "near_block_shape_summary.csv", index=False)
    key_buckets.to_csv(out_dir / "key_bucket_model_comparison.csv", index=False)

    verdict_lines: list[str] = []
    if not summary.empty:
        by_run = {str(r.run): r for r in summary.itertuples(index=False)}
        by_attr = {str(r.run): r for r in attribution.itertuples(index=False)} if not attribution.empty else {}
        base = by_run.get("h240_f120_baseline")
        relief = by_run.get("h240_f120_relief_e15")
        anti = by_run.get("h240_f120_anti_persistence")
        intens = by_run.get("h240_f120_synth_intensify")
        f60 = by_run.get("pp_f60_relief_e15")
        oracle = by_run.get("h240_f120_oracle")
        if anti is not None and base is not None:
            anti_attr = by_attr.get("h240_f120_anti_persistence")
            verdict_lines.append(
                "- Anti-persistence / shape-preservation did not recover usable far-horizon hints: "
                f"pump {anti.sum_pump:.1f} vs baseline {base.sum_pump:.1f}, "
                f"max_p95 {anti.max_p95:.3f} vs {base.max_p95:.3f}, "
                f"hint_any {getattr(anti_attr, 'far_horizon_hint_any', 'n/a')}."
            )
        if intens is not None and base is not None:
            intens_attr = by_attr.get("h240_f120_synth_intensify")
            verdict_lines.append(
                "- Synth-intensify slightly reduces max_p95/fallback but spends more pump and still produces no far-horizon hints: "
                f"pump {intens.sum_pump:.1f} vs baseline {base.sum_pump:.1f}, "
                f"max_p95 {intens.max_p95:.3f} vs {base.max_p95:.3f}, "
                f"sum_fb {intens.sum_fb:.3f} vs {base.sum_fb:.3f}, "
                f"hint_any {getattr(intens_attr, 'far_horizon_hint_any', 'n/a')}."
            )
        if relief is not None and base is not None:
            verdict_lines.append(
                "- h240_f120 relief_e15 remains the best h240 learned variant in this batch: "
                f"pump {relief.sum_pump:.1f}, max_p95 {relief.max_p95:.3f}, sum_fb {relief.sum_fb:.3f}."
            )
        if f60 is not None:
            verdict_lines.append(
                "- The current overall learned best is still the f60 relief_e15 controller run on aggregate metrics: "
                f"pump {f60.sum_pump:.1f}, max_p95 {f60.max_p95:.3f}, sum_fb {f60.sum_fb:.3f}."
            )
        if oracle is not None:
            verdict_lines.append(
                "- Oracle remains far ahead, so the 60-120 min value exists in the data path but was not recovered by these two isolated losses: "
                f"pump {oracle.sum_pump:.1f}, max_p95 {oracle.max_p95:.3f}, sum_fb {oracle.sum_fb:.3f}."
            )

    lines = [
        "# h240_f120 forecast-loss experiment summary",
        "",
        "Controller is fixed. This report only compares forecast-model outputs through existing planner logs.",
        "",
        "## Verdict",
        *(verdict_lines or ["- No verdict generated; missing comparison rows."]),
        "",
        "## Closed-loop aggregate",
        _markdown_table(summary[["run", "sum_pump", "max_p95", "sum_fb", "time_over_3", "time_over_5"]] if not summary.empty else summary),
        "",
        "## Forecast-control attribution",
        _markdown_table(
            attribution[
                [
                    "run",
                    "far_horizon_hint_any",
                    "hidden_relief",
                    "hidden_intensification",
                    "reversal",
                    "direction_shift",
                    "oracle_diff_key_buckets",
                    "oracle_diff_key_action_matches",
                    "near_range_mean",
                    "far_range_mean",
                    "far_over_near_mean",
                ]
            ]
            if not attribution.empty
            else attribution
        ),
        "",
        "## 0-60min planner-block shape",
        _markdown_table(
            near_shape[
                [
                    "run",
                    "near_block_b012_mae",
                    "near_block_b012_rmse",
                    "near_direction_agreement",
                    "near_range_mae",
                    "near_range_ratio",
                    "oracle_diff_key_buckets",
                    "oracle_diff_key_shape_closer_than_baseline",
                    "oracle_diff_key_action_matches",
                ]
            ]
            if not near_shape.empty
            else near_shape
        ),
        "",
        "## Outputs",
        "- `closed_loop_summary.csv`",
        "- `case_comparison.csv`",
        "- `forecast_control_attribution_summary.csv`",
        "- `near_block_shape_summary.csv`",
        "- `key_bucket_model_comparison.csv`",
    ]
    (out_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {out_dir / 'summary.md'}")


if __name__ == "__main__":
    main()
