#!/usr/bin/env python3
"""Forced-prefix closed-loop replay for path-level attribution.

This is a diagnostic runner. It does not train models and does not change the
default controller path. It generates forced-prefix action specs from existing
learned/oracle planner logs, runs full closed-loop cases from t=0, and compares
whether oracle-like prefixes move later path state toward oracle.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]

CASES_CSV = REPO_ROOT / "outputs/wind_prediction/broad_optimization_work/guard10_broad_cases.csv"
KEY_BUCKETS = REPO_ROOT / "outputs/wind_prediction/key_bucket_failure_split_v1/key_bucket_failure_split.csv"
LEARNED_RUN = REPO_ROOT / "outputs/wind_prediction/pp_h240_f120_near_block_eventbalanced_v2_guard10_learned_v1"
ORACLE_RUN = REPO_ROOT / "outputs/wind_prediction/pp_h240_f120_oracle_farlog_guard10_v1"
MODEL_DIR = REPO_ROOT / "outputs/wind_prediction/lstm_h240_f120_near_block_eventbalanced_v2"
DATASET_DIR = REPO_ROOT / "data/processed/wind_ml_10min/ballast_decision_fino1_meteo_aux_h240_f120_v1"


def _run_path(value: str | Path) -> Path:
    p = Path(value)
    return p if p.is_absolute() else REPO_ROOT / p


def _case_from_name(name: str) -> str:
    known = [
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
    for case in known:
        if case in str(name):
            return case
    text = str(name)
    return text.split("_", 1)[-1] if "_" in text else text


def _safe_float(value: Any, default: float = np.nan) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if np.isfinite(out) else default


def _nearest_ts(ts: pd.DataFrame, t_s: float) -> pd.Series:
    if ts.empty or "t_s" not in ts.columns:
        return pd.Series(dtype=object)
    idx = (pd.to_numeric(ts["t_s"], errors="coerce") - float(t_s)).abs().idxmin()
    return ts.loc[idx]


def _load_case(run_dir: Path, case: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    log_path = next((run_dir / "planner_logs").glob(f"*{case}*_planner_log.csv"), None)
    ts_path = next((run_dir / "timeseries").glob(f"*{case}*_timeseries.csv"), None)
    if log_path is None or ts_path is None:
        return pd.DataFrame(), pd.DataFrame()
    return pd.read_csv(log_path, low_memory=False), pd.read_csv(ts_path, low_memory=False)


def _planner_row(log: pd.DataFrame, bucket: int) -> pd.Series:
    if log.empty or "bucket" not in log.columns:
        return pd.Series(dtype=object)
    hit = log[log["bucket"].astype(int).eq(int(bucket))]
    return hit.iloc[0] if not hit.empty else pd.Series(dtype=object)


def _metrics_for_timeseries(ts: pd.DataFrame) -> dict[str, float]:
    if ts.empty or "pitch_deg" not in ts.columns:
        return {
            "pump_m3": np.nan,
            "pitch_p95": np.nan,
            "max_p95_proxy": np.nan,
            "fallback": np.nan,
            "time_over_3": np.nan,
            "time_over_5": np.nan,
        }
    pitch = ts["pitch_deg"].abs()
    roll = ts["roll_deg"].abs() if "roll_deg" in ts.columns else pd.Series(0.0, index=ts.index)
    pump = float((ts["pump_total_rate_m3_min"].abs() / 60.0).sum()) if "pump_total_rate_m3_min" in ts.columns else np.nan
    fallback_candidates = [
        "preview_primary_safety_fallback",
        "prediction_primary_safety_fallback",
        "safety_fallback_active",
        "fallback_active",
    ]
    fb_col = next((c for c in fallback_candidates if c in ts.columns), None)
    if fb_col is None:
        fb_col = next(
            (
                c
                for c in ts.columns
                if "fallback" in c.lower()
                and not pd.api.types.is_object_dtype(ts[c])
                and "reason" not in c.lower()
                and "lookup" not in c.lower()
            ),
            None,
        )
    fallback = float(ts[fb_col].mean()) if fb_col else 0.0
    return {
        "pump_m3": pump,
        "pitch_p95": float(np.percentile(pitch, 95)),
        "max_p95_proxy": float(max(np.percentile(pitch, 95), np.percentile(roll, 95))),
        "fallback": fallback,
        "time_over_3": float(((pitch > 3.0) | (roll > 3.0)).sum()),
        "time_over_5": float(((pitch > 5.0) | (roll > 5.0)).sum()),
    }


def _state_trace_row(
    *,
    run_label: str,
    case: str,
    bucket: int,
    log: pd.DataFrame,
    ts: pd.DataFrame,
    oracle_log: pd.DataFrame,
    oracle_ts: pd.DataFrame,
) -> dict[str, Any]:
    row = _planner_row(log, bucket)
    orow = _planner_row(oracle_log, bucket)
    t_s = _safe_float(row.get("current_time_s"), bucket * 600.0)
    tr = _nearest_ts(ts, t_s)
    otr = _nearest_ts(oracle_ts, t_s)
    pitch = _safe_float(row.get("current_pitch_deg"), _safe_float(tr.get("pitch_deg"), 0.0))
    roll = _safe_float(row.get("current_roll_deg"), _safe_float(tr.get("roll_deg"), 0.0))
    opitch = _safe_float(orow.get("current_pitch_deg"), _safe_float(otr.get("pitch_deg"), 0.0))
    oroll = _safe_float(orow.get("current_roll_deg"), _safe_float(otr.get("roll_deg"), 0.0))
    target = np.array([
        _safe_float(row.get("prediction_primary_target_t1_kg")),
        _safe_float(row.get("prediction_primary_target_t2_kg")),
        _safe_float(row.get("prediction_primary_target_t3_kg")),
    ])
    otarget = np.array([
        _safe_float(orow.get("prediction_primary_target_t1_kg")),
        _safe_float(orow.get("prediction_primary_target_t2_kg")),
        _safe_float(orow.get("prediction_primary_target_t3_kg")),
    ])
    tank = np.array([
        _safe_float(tr.get("tank1_kg")),
        _safe_float(tr.get("tank2_kg")),
        _safe_float(tr.get("tank3_kg")),
    ])
    otank = np.array([
        _safe_float(otr.get("tank1_kg")),
        _safe_float(otr.get("tank2_kg")),
        _safe_float(otr.get("tank3_kg")),
    ])
    pump = np.array([
        _safe_float(tr.get("pump_rate1_m3min"), 0.0),
        _safe_float(tr.get("pump_rate2_m3min"), 0.0),
        _safe_float(tr.get("pump_rate3_m3min"), 0.0),
    ])
    opump = np.array([
        _safe_float(otr.get("pump_rate1_m3min"), 0.0),
        _safe_float(otr.get("pump_rate2_m3min"), 0.0),
        _safe_float(otr.get("pump_rate3_m3min"), 0.0),
    ])
    return {
        "run_label": run_label,
        "case_id": case,
        "bucket": int(bucket),
        "first_action": str(row.get("first_action", "")),
        "raw_action": str(row.get("planner_first_action_raw", "")),
        "oracle_first_action": str(orow.get("first_action", "")),
        "matches_oracle_action": int(str(row.get("first_action", "")) == str(orow.get("first_action", ""))),
        "pitch_deg": pitch,
        "roll_deg": roll,
        "oracle_pitch_deg": opitch,
        "oracle_roll_deg": oroll,
        "posture_distance_to_oracle_deg": float(np.linalg.norm(np.array([pitch - opitch, roll - oroll]))),
        "target_mean_abs_delta_to_oracle_kg": float(np.nanmean(np.abs(target - otarget))),
        "tank_mean_abs_delta_to_oracle_kg": float(np.nanmean(np.abs(tank - otank))),
        "pump_mean_abs_delta_to_oracle_m3_min": float(np.nanmean(np.abs(pump - opump))),
        "forced_prefix_active": int(_safe_float(row.get("forced_prefix_active"), 0.0) > 0.5),
        "forced_prefix_mode": str(row.get("forced_prefix_mode", "")),
        "forced_action_label": str(row.get("forced_action_label", "")),
        "forced_vector_mode": str(row.get("forced_vector_mode", "")),
        "forced_source_label": str(row.get("forced_source_label", "")),
        "forced_target_update_active": int(_safe_float(row.get("forced_target_update_active"), 0.0) > 0.5),
    }


def _collect_run(
    run_label: str,
    run_dir: Path,
    oracle_dir: Path,
    key_df: pd.DataFrame,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    summary_rows: list[dict[str, Any]] = []
    state_rows: list[dict[str, Any]] = []
    action_rows: list[dict[str, Any]] = []
    for case, sub in key_df.groupby("case_id"):
        log, ts = _load_case(run_dir, case)
        olog, ots = _load_case(oracle_dir, case)
        if log.empty or ts.empty or olog.empty or ots.empty:
            continue
        m = _metrics_for_timeseries(ts)
        forced_count = int(log.get("forced_prefix_active", pd.Series(0, index=log.index)).fillna(0).astype(float).sum()) if "forced_prefix_active" in log.columns else 0
        key_matches = 0
        for bucket in sorted(set(int(x) for x in sub["bucket"].tolist())):
            for b in sorted({max(0, bucket - 1), bucket, min(int(log["bucket"].max()), bucket + 1)}):
                state_rows.append(
                    _state_trace_row(
                        run_label=run_label,
                        case=case,
                        bucket=b,
                        log=log,
                        ts=ts,
                        oracle_log=olog,
                        oracle_ts=ots,
                    )
                )
            row = _planner_row(log, bucket)
            orow = _planner_row(olog, bucket)
            match = int(str(row.get("first_action", "")) == str(orow.get("first_action", "")))
            key_matches += match
            action_rows.append(
                {
                    "run_label": run_label,
                    "case_id": case,
                    "bucket": int(bucket),
                    "first_action": str(row.get("first_action", "")),
                    "raw_action": str(row.get("planner_first_action_raw", "")),
                    "oracle_first_action": str(orow.get("first_action", "")),
                    "matches_oracle_action": match,
                    "forced_prefix_active": int(_safe_float(row.get("forced_prefix_active"), 0.0) > 0.5),
                    "forced_prefix_mode": str(row.get("forced_prefix_mode", "")),
                    "forced_action_label": str(row.get("forced_action_label", "")),
                    "forced_vector_mode": str(row.get("forced_vector_mode", "")),
                    "forced_source_label": str(row.get("forced_source_label", "")),
                    "forced_original_raw_action": str(row.get("forced_original_raw_action", "")),
                    "forced_original_final_action": str(row.get("forced_original_final_action", "")),
                    "forced_target_update_active": int(_safe_float(row.get("forced_target_update_active"), 0.0) > 0.5),
                }
            )
        summary_rows.append(
            {
                "run_label": run_label,
                "case_id": case,
                **m,
                "key_action_match_count": int(key_matches),
                "key_bucket_count": int(len(sub)),
                "forced_bucket_count": forced_count,
            }
        )
    return summary_rows, state_rows, action_rows


def _source_run_dir(source: str, learned_run: Path, oracle_run: Path) -> Path:
    if source == "oracle":
        return oracle_run
    if source == "learned":
        return learned_run
    raise ValueError(f"unsupported prefix source {source!r}")


def _write_forced_spec(
    *,
    source: str,
    length: int,
    mode: str,
    key_df: pd.DataFrame,
    learned_run: Path,
    oracle_run: Path,
    out_path: Path,
) -> None:
    rows: list[dict[str, Any]] = []
    source_dir = _source_run_dir(source, learned_run, oracle_run)
    for case, sub in key_df.groupby("case_id"):
        log, _ = _load_case(source_dir, case)
        if log.empty:
            continue
        prefix_buckets: set[int] = set()
        for key_bucket in sub["bucket"].astype(int):
            start = max(0, int(key_bucket) - int(length))
            prefix_buckets.update(range(start, int(key_bucket)))
        # If the key bucket is at 0, use the first length buckets as the prefix.
        if not prefix_buckets:
            prefix_buckets.update(range(0, int(length)))
        for bucket in sorted(prefix_buckets):
            row = _planner_row(log, bucket)
            if row.empty:
                continue
            rows.append(
                {
                    "case_id": case,
                    "bucket": int(bucket),
                    "action": str(row.get("first_action", "")),
                    "planner_action_pitch_deg": _safe_float(row.get("planner_action_pitch_deg"), 0.0),
                    "planner_action_roll_deg": _safe_float(row.get("planner_action_roll_deg"), 0.0),
                    "source_label": source,
                    "prefix_length": int(length),
                    "forced_prefix_mode": mode,
                }
            )
    pd.DataFrame(rows).to_csv(out_path, index=False)


def _run_casebook(
    *,
    out_dir: Path,
    forecast_source: str,
    model_dir: Path,
    dataset_dir: Path,
    forced_spec: Path | None,
    forced_mode: str,
    python_bin: str,
    cases_csv: Path,
    case_ids: str,
) -> None:
    cmd = [
        python_bin,
        "scripts/analysis/run_prediction_primary_casebook.py",
        "--cases-csv",
        str(cases_csv),
        "--case-ids",
        str(case_ids),
        "--primary-control-profile",
        "rawenv_holdpause_barrier_reliefcap_adaptive_v1",
        "--primary-only",
        "--duration-s",
        "7200",
        "--skip-figures",
        "--forecast-source",
        forecast_source,
        "--dataset-dir",
        str(dataset_dir),
        "--out-dir",
        str(out_dir),
    ]
    if forecast_source == "learned":
        cmd.extend(["--model-dir", str(model_dir)])
    if forced_spec is not None and forced_mode != "off":
        cmd.extend(
            [
                "--forced-prefix-actions",
                str(forced_spec),
                "--forced-prefix-mode",
                forced_mode,
            ]
        )
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)


def _case_label(
    pump_delta: float,
    fallback_delta: float,
    posture_delta: float,
    action_gain: float,
    run_label: str = "",
) -> str:
    if not np.isfinite(pump_delta):
        return "inconclusive"
    if "target_update" in str(run_label):
        target_sensitive = (
            pump_delta > 150.0
            or fallback_delta > 0.02
            or posture_delta > 0.25
            or abs(action_gain) >= 1.0
        )
        if target_sensitive:
            return "target-lifecycle-dominated"
    if fallback_delta > 0.01 or posture_delta > 0.25:
        return "prefix-worse"
    if action_gain > 0 and fallback_delta <= 0.0 and pump_delta <= 25.0:
        return "prefix-beneficial"
    if action_gain > 0 and pump_delta > 25.0:
        return "prefix-pump-tradeoff"
    if abs(action_gain) < 1e-9 and abs(fallback_delta) < 0.005 and abs(posture_delta) < 0.15:
        return "prefix-no-effect"
    return "inconclusive"


def _write_reports(
    out_dir: Path,
    closed: pd.DataFrame,
    state: pd.DataFrame,
    action: pd.DataFrame,
) -> None:
    baseline = closed[closed["run_label"].eq("learned_forecast_learned_prefix_len0_normal")]
    attrib_rows: list[dict[str, Any]] = []
    for _, row in closed.iterrows():
        base = baseline[baseline["case_id"].eq(row["case_id"])]
        if base.empty or row["run_label"] == "learned_forecast_learned_prefix_len0_normal":
            continue
        b = base.iloc[0]
        srow = state[
            state["run_label"].eq(row["run_label"])
            & state["case_id"].eq(row["case_id"])
            & state["matches_oracle_action"].notna()
        ]
        brow = state[
            state["run_label"].eq("learned_forecast_learned_prefix_len0_normal")
            & state["case_id"].eq(row["case_id"])
        ]
        posture = float(srow["posture_distance_to_oracle_deg"].mean() - brow["posture_distance_to_oracle_deg"].mean()) if not srow.empty and not brow.empty else np.nan
        action_gain = float(row["key_action_match_count"] - b["key_action_match_count"])
        pump_delta = float(row["pump_m3"] - b["pump_m3"])
        fallback_delta = float(row["fallback"] - b["fallback"])
        label = _case_label(pump_delta, fallback_delta, posture, action_gain, str(row["run_label"]))
        attrib_rows.append(
            {
                "run_label": row["run_label"],
                "case_id": row["case_id"],
                "pump_delta_vs_learned_prefix": pump_delta,
                "fallback_delta_vs_learned_prefix": fallback_delta,
                "mean_key_posture_distance_delta_vs_learned_prefix": posture,
                "key_action_match_gain_vs_learned_prefix": action_gain,
                "case_label": label,
            }
        )
    attrib = pd.DataFrame(attrib_rows)
    attrib.to_csv(out_dir / "forced_prefix_case_labels.csv", index=False)
    label_counts = (
        attrib.groupby(["run_label", "case_label"], as_index=False)
        .size()
        .rename(columns={"size": "count"})
        if not attrib.empty
        else pd.DataFrame()
    )
    lines = [
        "# Forced-Prefix Replay v1",
        "",
        "Default-off diagnostic. Full closed-loop cases start at t=0; forced prefixes are applied only at selected buckets, then the normal controller resumes.",
        "",
        "## Aggregate Closed-Loop Summary",
    ]
    agg = closed.groupby("run_label", as_index=False).agg(
        sum_pump=("pump_m3", "sum"),
        max_p95=("max_p95_proxy", "max"),
        sum_fb=("fallback", "sum"),
        time_over_3=("time_over_3", "sum"),
        time_over_5=("time_over_5", "sum"),
        key_action_matches=("key_action_match_count", "sum"),
        forced_buckets=("forced_bucket_count", "sum"),
    )
    lines.append(_md_table(agg))
    lines.extend(["", "## Case Labels"])
    lines.append(_md_table(label_counts) if not label_counts.empty else "_No labels generated._")
    lines.extend(
        [
            "",
            "## Interpretation Rules",
            "",
            "- `prefix-beneficial`: action/posture/fallback improved without obvious pump increase.",
            "- `prefix-pump-tradeoff`: action/posture improved but pump increased materially.",
            "- `prefix-no-effect`: no meaningful path change.",
            "- `prefix-worse`: fallback or posture distance worsened.",
            "- `target-lifecycle-dominated`: target-update-only forcing materially changed pump/fallback/posture/action, so target lifecycle is path-sensitive but not safe as a standalone intervention.",
            "- `inconclusive`: mixed or weak signal.",
        ]
    )
    (out_dir / "forced_prefix_replay_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    case_lines = ["# Forced-Prefix Case Attribution", ""]
    if attrib.empty:
        case_lines.append("No attribution rows generated.")
    else:
        case_lines.append(_md_table(attrib))
    (out_dir / "forced_prefix_case_attribution.md").write_text("\n".join(case_lines) + "\n", encoding="utf-8")


def _md_table(df: pd.DataFrame, max_rows: int = 80) -> str:
    if df.empty:
        return "_No rows._"
    show = df.head(max_rows).copy()
    cols = list(show.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in show.iterrows():
        values = []
        for col in cols:
            value = row[col]
            if isinstance(value, float):
                values.append(f"{value:.3f}" if np.isfinite(value) else "")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default="outputs/wind_prediction/forced_prefix_replay_v1")
    parser.add_argument("--cases-csv", default=str(CASES_CSV))
    parser.add_argument("--key-buckets", default=str(KEY_BUCKETS))
    parser.add_argument("--learned-run", default=str(LEARNED_RUN))
    parser.add_argument("--oracle-run", default=str(ORACLE_RUN))
    parser.add_argument("--model-dir", default=str(MODEL_DIR))
    parser.add_argument("--dataset-dir", default=str(DATASET_DIR))
    parser.add_argument("--python-bin", default=".venv312/bin/python")
    parser.add_argument("--max-prefix-length", type=int, default=5)
    parser.add_argument("--modes", default="final_action,raw_action,target_update")
    parser.add_argument(
        "--case-ids",
        default="01,02,03,04,05,06",
        help="Case ids passed to casebook; default is the 6 affected path-divergence cases.",
    )
    parser.add_argument("--skip-runs", action="store_true", help="Only summarize existing run outputs.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = _run_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    specs_dir = out_dir / "forced_specs"
    runs_dir = out_dir / "runs"
    specs_dir.mkdir(parents=True, exist_ok=True)
    runs_dir.mkdir(parents=True, exist_ok=True)
    key_df = pd.read_csv(_run_path(args.key_buckets), low_memory=False)
    key_df["case_id"] = key_df["case_id"].map(_case_from_name)
    key_df = key_df[key_df["case_id"].isin([
        "fr_relief_01",
        "fr_relief_09",
        "sf_holdout_02",
        "b_decay_strong",
        "b_signflip_fallback",
        "b_high_pressure_event",
    ])].copy()
    learned_run = _run_path(args.learned_run)
    oracle_run = _run_path(args.oracle_run)
    model_dir = _run_path(args.model_dir)
    dataset_dir = _run_path(args.dataset_dir)
    cases_csv = _run_path(args.cases_csv)
    modes = [m.strip() for m in str(args.modes).split(",") if m.strip()]

    run_specs: list[tuple[str, Path, str, str, Path | None]] = [
        ("learned_forecast_learned_prefix_len0_normal", learned_run, "learned", "off", None),
        ("oracle_forecast_oracle_prefix_len0_normal", oracle_run, "oracle", "off", None),
    ]
    for mode in modes:
        for length in range(1, int(args.max_prefix_length) + 1):
            for forecast_source, prefix_source in [
                ("learned", "learned"),
                ("learned", "oracle"),
                ("oracle", "learned"),
                ("oracle", "oracle"),
            ]:
                label = f"{forecast_source}_forecast_{prefix_source}_prefix_len{length}_{mode}"
                spec = specs_dir / f"{label}.csv"
                _write_forced_spec(
                    source=prefix_source,
                    length=length,
                    mode=mode,
                    key_df=key_df,
                    learned_run=learned_run,
                    oracle_run=oracle_run,
                    out_path=spec,
                )
                run_specs.append((label, runs_dir / label, forecast_source, mode, spec))

    if not args.skip_runs:
        for label, run_dir, forecast_source, mode, spec in run_specs[2:]:
            if (run_dir / "casebook_summary.csv").exists():
                continue
            _run_casebook(
                out_dir=run_dir,
                forecast_source=forecast_source,
                model_dir=model_dir,
                dataset_dir=dataset_dir,
                forced_spec=spec,
                forced_mode=mode,
                python_bin=str(args.python_bin),
                cases_csv=cases_csv,
                case_ids=str(args.case_ids),
            )

    closed_rows: list[dict[str, Any]] = []
    state_rows: list[dict[str, Any]] = []
    action_rows: list[dict[str, Any]] = []
    for label, run_dir, _, _, _ in run_specs:
        if not (run_dir / "planner_logs").exists() and label.startswith("learned_forecast"):
            # The normal learned baseline is an existing source run.
            run_dir = learned_run
        elif not (run_dir / "planner_logs").exists() and label.startswith("oracle_forecast"):
            run_dir = oracle_run
        s, st, a = _collect_run(label, run_dir, oracle_run, key_df)
        closed_rows.extend(s)
        state_rows.extend(st)
        action_rows.extend(a)

    closed = pd.DataFrame(closed_rows)
    state = pd.DataFrame(state_rows)
    action = pd.DataFrame(action_rows)
    closed.to_csv(out_dir / "forced_prefix_closed_loop_summary.csv", index=False)
    state.to_csv(out_dir / "forced_prefix_path_state_trace.csv", index=False)
    action.to_csv(out_dir / "forced_prefix_action_trace.csv", index=False)
    _write_reports(out_dir, closed, state, action)
    print(f"wrote {out_dir / 'forced_prefix_replay_summary.md'}")


if __name__ == "__main__":
    main()
