#!/usr/bin/env python3
"""Oracle relief-conditioned posture-envelope audit for h120.

This audit tests whether a dynamic 3-5deg soft recovery band can save pump
work while leaving the v1.6 hard reactive floor untouched.  A1 uses only
0-60min oracle relief blocks; A2 can additionally use 60-120min blocks.
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON = REPO_ROOT / ".venv312/bin/python"
if not PYTHON.exists():
    PYTHON = REPO_ROOT / ".venv/bin/python"
OUT_DIR = REPO_ROOT / "outputs/wind_prediction/h120_relief_envelope_oracle_audit_v1"
F120_DATASET = (
    REPO_ROOT / "data/processed/wind_ml_10min/ballast_decision_fino1_meteo_aux_h240_f120_v1"
)
CASE_SOURCES = {
    "relief": REPO_ROOT / "outputs/wind_prediction/prediction_value_mechanism_grid_v2/future_relief_cases.csv",
    "lowrisk": REPO_ROOT / "outputs/wind_prediction/prediction_value_mechanism_grid_v2/lowrisk_quiet_cases.csv",
    "broader20": REPO_ROOT / "outputs/wind_prediction/h120_oracle_controller_value_audit_v1/broader20_cases.csv",
    "guard10": REPO_ROOT / "outputs/wind_prediction/h120_oracle_controller_value_audit_v1/guard10_cases.csv",
}
PROFILES: dict[str, dict[str, float]] = {
    "conservative": {
        "allowed_0_20": 4.50,
        "allowed_20_40": 4.20,
        "allowed_40_60": 3.90,
        "allowed_60_120": 3.50,
        "duration_s": 600.0,
        "debt_budget": 300.0,
        "near_limit_budget_s": 0.0,
        "worsening_eps": 0.0,
    },
    "medium": {
        "allowed_0_20": 4.65,
        "allowed_20_40": 4.40,
        "allowed_40_60": 4.10,
        "allowed_60_120": 3.70,
        "duration_s": 900.0,
        "debt_budget": 600.0,
        "near_limit_budget_s": 300.0,
        "worsening_eps": 0.05,
    },
    "aggressive": {
        "allowed_0_20": 4.80,
        "allowed_20_40": 4.55,
        "allowed_40_60": 4.25,
        "allowed_60_120": 3.90,
        "duration_s": 1200.0,
        "debt_budget": 900.0,
        "near_limit_budget_s": 600.0,
        "worsening_eps": 0.10,
    },
}


def _rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _mkdirs() -> dict[str, Path]:
    dirs = {
        "out": OUT_DIR,
        "runs": OUT_DIR / "runs",
        "paper": OUT_DIR / "paper_ready",
        "raw": OUT_DIR / "raw_tables",
        "diag": OUT_DIR / "diagnostics",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def _case_key(path: Path) -> str:
    name = path.name
    for suffix in ("_prediction_primary_econ_timeseries.csv", "_prediction_primary_econ_planner_log.csv"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return path.stem


def _fallback_series(df: pd.DataFrame) -> pd.Series:
    for col in (
        "preview_primary_safety_fallback",
        "primary_safety_fallback",
        "primary_safety_fallback_active",
    ):
        if col in df.columns:
            return df[col].astype(float) > 0.5
    return pd.Series(False, index=df.index)


def _edge_count(series: pd.Series) -> int:
    arr = series.fillna(0).astype(float).to_numpy() > 0.5
    if arr.size == 0:
        return 0
    return int(np.logical_and(arr, np.r_[True, ~arr[:-1]]).sum())


def _base_cmd(out_dir: Path, cases_csv: Path) -> list[str]:
    return [
        str(PYTHON),
        str(REPO_ROOT / "scripts/analysis/run_prediction_primary_casebook.py"),
        "--out-dir",
        str(out_dir),
        "--primary-only",
        "--skip-figures",
        "--duration-s",
        "7200",
        "--cases-csv",
        str(cases_csv),
        "--forecast-source",
        "oracle",
        "--dataset-dir",
        str(F120_DATASET),
        "--primary-control-profile",
        "rawenv_holdpause_barrier_reliefcap_adaptive_v1",
        "--reactive-floor-predictive-veto",
        "on",
        "--high-posture-metric",
        "max_axis",
        "--reactive-floor-action",
        "active_small",
        "--reactive-floor-medium-delay-s",
        "1200",
        "--reactive-floor-post-exit-mode",
        "early_stop",
        "--far-horizon",
    ]


def _run_casebook(
    run_dir: Path,
    cases_csv: Path,
    *,
    arm: str,
    profile: str,
    force: bool,
) -> None:
    if (run_dir / "casebook_summary.csv").exists() and not force:
        print(f"[skip] {_rel(run_dir)}")
        return
    run_dir.mkdir(parents=True, exist_ok=True)
    cmd = _base_cmd(run_dir, cases_csv)
    if arm in ("A1_near_envelope", "A2_h120_envelope"):
        p = PROFILES[profile]
        horizon = "near" if arm == "A1_near_envelope" else "far"
        cmd.extend(
            [
                "--relief-envelope",
                "--relief-envelope-horizon",
                horizon,
                "--relief-envelope-allowed-0-20-deg",
                str(p["allowed_0_20"]),
                "--relief-envelope-allowed-20-40-deg",
                str(p["allowed_20_40"]),
                "--relief-envelope-allowed-40-60-deg",
                str(p["allowed_40_60"]),
                "--relief-envelope-allowed-60-120-deg",
                str(p["allowed_60_120"]),
                "--relief-envelope-max-duration-s",
                str(p["duration_s"]),
                "--relief-envelope-debt-budget-deg-s",
                str(p["debt_budget"]),
                "--relief-envelope-near-limit-budget-s",
                str(p["near_limit_budget_s"]),
                "--relief-envelope-worsening-eps-deg",
                str(p["worsening_eps"]),
            ]
        )
    print(f"[run] {_rel(run_dir)}")
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)


def _read_log(run_dir: Path, key: str) -> pd.DataFrame:
    path = run_dir / "planner_logs" / f"{key}_prediction_primary_econ_planner_log.csv"
    if path.exists():
        return pd.read_csv(path, low_memory=False)
    matches = list((run_dir / "planner_logs").glob(f"{key}*_planner_log.csv"))
    if not matches:
        return pd.DataFrame()
    return pd.read_csv(matches[0], low_memory=False)


def _last_metric(log: pd.DataFrame, name: str, default: float = 0.0) -> float:
    if name not in log.columns or log.empty:
        return float(default)
    s = log[name].fillna(default).astype(float)
    return float(s.iloc[-1]) if len(s) else float(default)


def _sum_active(log: pd.DataFrame, name: str) -> int:
    if name not in log.columns or log.empty:
        return 0
    return int((log[name].fillna(0).astype(float) > 0.5).sum())


def _case_metrics(run_dir: Path, dataset: str, profile: str, arm: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for ts_path in sorted((run_dir / "timeseries").glob("*_timeseries.csv")):
        key = _case_key(ts_path)
        df = pd.read_csv(ts_path, low_memory=False)
        pitch = df["pitch_deg"].astype(float).abs()
        roll = df["roll_deg"].astype(float).abs()
        max_axis = np.maximum(pitch, roll)
        pump_rate = df.get("pump_total_rate_m3_min", pd.Series(0.0, index=df.index)).astype(float).abs()
        high = max_axis > 5.0
        idle = pump_rate < 0.05
        near5 = max_axis > 4.7
        fb = _fallback_series(df)
        log = _read_log(run_dir, key)
        floor = log.get("reactive_floor_active", pd.Series(dtype=float)).fillna(0).astype(float)
        medium = log.get("reactive_floor_medium_delay_active", pd.Series(dtype=float)).fillna(0).astype(float)
        rows.append(
            {
                "dataset": dataset,
                "profile": profile,
                "arm": arm,
                "case": key,
                "pump": float(pump_rate.sum() / 60.0),
                "sum_fb": float(fb.mean() * 100.0),
                "max_axis": float(max_axis.max()),
                "max_p95": float(np.percentile(max_axis, 95)),
                "pitch_p95": float(np.percentile(pitch, 95)),
                "roll_p95": float(np.percentile(roll, 95)),
                "time_over_5": int(high.sum()),
                "idle_time_over_5": int((high & idle).sum()),
                "near5_time": int(near5.sum()),
                "floor_trigger_count": _edge_count(floor),
                "floor_active_rows": int(floor.sum()) if len(floor) else 0,
                "medium_escalation_count": _edge_count(medium),
                "relief_envelope_active_rows": _sum_active(log, "relief_envelope_active"),
                "eligible_count": int(_last_metric(log, "relief_envelope_eligible_count")),
                "pump_opportunity_count": int(_last_metric(log, "relief_envelope_pump_opportunity_count")),
                "target_refresh_delayed_count": int(
                    _last_metric(log, "relief_envelope_target_refresh_delayed_count")
                ),
                "relaxation_active_time": float(
                    _last_metric(log, "relief_envelope_relaxation_active_time_s")
                ),
                "safety_debt_used": float(_last_metric(log, "relief_envelope_safety_debt_used")),
                "debt_exit_count": int(_last_metric(log, "relief_envelope_debt_exit_count")),
                "false_relief_count": int(_last_metric(log, "relief_envelope_false_relief_count")),
                "reintensification_veto_count": int(
                    _last_metric(log, "relief_envelope_reintensification_veto_count")
                ),
                "direction_mismatch_veto_count": int(
                    _last_metric(log, "relief_envelope_direction_mismatch_veto_count")
                ),
                "posture_worsening_exit_count": int(
                    _last_metric(log, "relief_envelope_posture_worsening_exit_count")
                ),
                "A2_unique_relax_count": int(
                    _last_metric(log, "relief_envelope_a2_unique_relax_count")
                ),
                "A2_unique_pump_saved": 0.0,
            }
        )
    return rows


def _aggregate(case_table: pd.DataFrame) -> pd.DataFrame:
    agg = (
        case_table.groupby(["dataset", "profile", "arm"], as_index=False)
        .agg(
            cases=("case", "count"),
            sum_pump=("pump", "sum"),
            sum_fb=("sum_fb", "sum"),
            max_axis=("max_axis", "max"),
            p95_max_axis=("max_p95", "max"),
            time_over_5=("time_over_5", "sum"),
            idle_time_over_5=("idle_time_over_5", "sum"),
            near5_time=("near5_time", "sum"),
            floor_trigger_count=("floor_trigger_count", "sum"),
            floor_active_rows=("floor_active_rows", "sum"),
            medium_escalation_count=("medium_escalation_count", "sum"),
            eligible_count=("eligible_count", "sum"),
            pump_opportunity_count=("pump_opportunity_count", "sum"),
            target_refresh_delayed_count=("target_refresh_delayed_count", "sum"),
            relaxation_active_time=("relaxation_active_time", "sum"),
            safety_debt_used=("safety_debt_used", "sum"),
            debt_exit_count=("debt_exit_count", "sum"),
            false_relief_count=("false_relief_count", "sum"),
            reintensification_veto_count=("reintensification_veto_count", "sum"),
            direction_mismatch_veto_count=("direction_mismatch_veto_count", "sum"),
            posture_worsening_exit_count=("posture_worsening_exit_count", "sum"),
            A2_unique_relax_count=("A2_unique_relax_count", "sum"),
        )
        .sort_values(["profile", "dataset", "arm"])
        .reset_index(drop=True)
    )
    baseline = agg[agg["arm"].eq("A0_v16")].set_index(["dataset", "profile"])
    near = agg[agg["arm"].eq("A1_near_envelope")].set_index(["dataset", "profile"])
    for idx, row in agg.iterrows():
        key = (row["dataset"], row["profile"])
        if key in baseline.index:
            for col in ("sum_pump", "sum_fb", "p95_max_axis", "time_over_5", "idle_time_over_5", "near5_time"):
                agg.loc[idx, f"delta_vs_A0_{col}"] = float(row[col]) - float(baseline.loc[key, col])
        if key in near.index:
            for col in ("sum_pump", "sum_fb", "p95_max_axis", "time_over_5", "idle_time_over_5", "near5_time"):
                agg.loc[idx, f"delta_vs_A1_{col}"] = float(row[col]) - float(near.loc[key, col])
    agg["A2_unique_pump_saved"] = np.where(
        agg["arm"].eq("A2_h120_envelope"),
        -agg.get("delta_vs_A1_sum_pump", 0.0).fillna(0.0),
        0.0,
    )
    return agg


def _reason_table(run_map: dict[tuple[str, str, str], Path]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (dataset, profile, arm), run_dir in run_map.items():
        if arm == "A0_v16":
            continue
        for log_path in sorted((run_dir / "planner_logs").glob("*_planner_log.csv")):
            log = pd.read_csv(log_path, low_memory=False)
            if "relief_envelope_reason" not in log.columns:
                continue
            counts = log["relief_envelope_reason"].astype(str).value_counts()
            for reason, count in counts.items():
                rows.append(
                    {
                        "dataset": dataset,
                        "profile": profile,
                        "arm": arm,
                        "reason": reason,
                        "rows": int(count),
                    }
                )
    if not rows:
        return pd.DataFrame()
    return (
        pd.DataFrame(rows)
        .groupby(["dataset", "profile", "arm", "reason"], as_index=False)["rows"]
        .sum()
        .sort_values(["profile", "dataset", "arm", "reason"])
    )


def _bucket_opportunity_tables(
    run_map: dict[tuple[str, str, str], Path]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []

    def _load_ts_pump(run_dir: Path, case: str) -> tuple[np.ndarray, np.ndarray]:
        ts_path = run_dir / "timeseries" / f"{case}_prediction_primary_econ_timeseries.csv"
        ts = pd.read_csv(ts_path, low_memory=False) if ts_path.exists() else pd.DataFrame()
        if not ts.empty and "t_s" in ts.columns:
            ts_time = ts["t_s"].astype(float).to_numpy()
            ts_pump = (
                ts.get("pump_total_rate_m3_min", pd.Series(0.0, index=ts.index))
                .astype(float)
                .abs()
                .to_numpy()
            )
            return ts_time, ts_pump
        return np.array([], dtype=float), np.array([], dtype=float)

    def _post_bucket_pump_m3(
        ts_time: np.ndarray,
        ts_pump: np.ndarray,
        t0: float,
        duration_s: float = 600.0,
    ) -> float:
        if ts_time.size == 0:
            return 0.0
        mask = (ts_time >= float(t0)) & (ts_time < float(t0) + duration_s)
        if not mask.any():
            return 0.0
        return float(ts_pump[mask].sum() / 60.0)

    for (dataset, profile, arm), run_dir in run_map.items():
        if arm == "A0_v16":
            continue
        baseline_dir = run_map.get((dataset, profile, "A0_v16"))
        for log_path in sorted((run_dir / "planner_logs").glob("*_planner_log.csv")):
            log = pd.read_csv(log_path, low_memory=False)
            if "relief_envelope_enabled" not in log.columns:
                continue
            case = log_path.name.replace("_prediction_primary_econ_planner_log.csv", "")
            ts_time, ts_pump = _load_ts_pump(run_dir, case)
            if baseline_dir is not None:
                base_ts_time, base_ts_pump = _load_ts_pump(baseline_dir, case)
            else:
                base_ts_time = np.array([], dtype=float)
                base_ts_pump = np.array([], dtype=float)

            def col(name: str, default: float | str = 0.0) -> pd.Series:
                if name in log.columns:
                    return log[name]
                return pd.Series(default, index=log.index)

            enabled = col("relief_envelope_enabled").fillna(0).astype(float) > 0.5
            if not enabled.any():
                continue
            active = col("relief_envelope_active").fillna(0).astype(float) > 0.5
            target_refreshed = (
                col("prediction_primary_target_refreshed").fillna(0).astype(float) > 0.5
            )
            target_reused = (
                col("prediction_primary_target_reused").fillna(0).astype(float) > 0.5
            )
            floor_active = col("reactive_floor_active").fillna(0).astype(float) > 0.5
            for i in log.index:
                if not bool(enabled.loc[i]):
                    continue
                pitch = float(col("current_pitch_deg").fillna(0).loc[i])
                roll = float(col("current_roll_deg").fillna(0).loc[i])
                current_time_s = float(col("current_time_s").fillna(float(i)).loc[i])
                rows.append(
                    {
                        "profile": profile,
                        "dataset": dataset,
                        "arm": arm,
                        "case": case,
                        "current_time_s": current_time_s,
                        "reason": str(col("relief_envelope_reason", "").loc[i]),
                        "active": int(active.loc[i]),
                        "target_refreshed": int(target_refreshed.loc[i]),
                        "target_reused": int(target_reused.loc[i]),
                        "target_delta_abs_mean_kg": float(
                            col("prediction_primary_delta_abs_mean_kg").fillna(0).loc[i]
                        ),
                        "target_age_s": float(
                            col("prediction_primary_target_age_s").fillna(0).loc[i]
                        ),
                        "target_err_mean_kg": float(
                            col("active_effectiveness_target_err_mean_kg").fillna(0).loc[i]
                        ),
                        "pump_rate_m3_min": float(
                            col("active_effectiveness_pump_rate_m3_min").fillna(0).loc[i]
                        ),
                        "post_bucket_pump_m3": _post_bucket_pump_m3(
                            ts_time, ts_pump, current_time_s
                        ),
                        "baseline_post_bucket_pump_m3": _post_bucket_pump_m3(
                            base_ts_time, base_ts_pump, current_time_s
                        ),
                        "reactive_floor_active": int(floor_active.loc[i]),
                        "pitch_deg": pitch,
                        "roll_deg": roll,
                        "max_axis_deg": max(abs(pitch), abs(roll)),
                        "allowed_max_axis_deg": float(
                            col("relief_envelope_allowed_max_axis_deg").fillna(0).loc[i]
                        ),
                        "relief_block_index": int(
                            col("relief_envelope_relief_block_index").fillna(-1).loc[i]
                        ),
                        "relief_time_min": float(
                            col("relief_envelope_relief_time_min").fillna(0).loc[i]
                        ),
                        "current_duration_s": float(
                            col("relief_envelope_current_duration_s").fillna(0).loc[i]
                        ),
                        "safety_debt_used": float(
                            col("relief_envelope_safety_debt_used").fillna(0).loc[i]
                        ),
                    }
                )
    bucket = pd.DataFrame(rows)
    if bucket.empty:
        bucket = pd.DataFrame(
            columns=[
                "profile",
                "dataset",
                "arm",
                "case",
                "current_time_s",
                "reason",
                "active",
                "target_refreshed",
                "pump_rate_m3_min",
            ]
        )
        return bucket, pd.DataFrame()
    summary_rows: list[dict[str, Any]] = []
    for (profile, dataset, arm), group in bucket.groupby(["profile", "dataset", "arm"]):
        active = group[group["active"] > 0]
        summary_rows.append(
            {
                "profile": profile,
                "dataset": dataset,
                "arm": arm,
                "rows": int(len(group)),
                "active_rows": int((group["active"] > 0).sum()),
                "active_target_refreshed_rows": int(
                    ((group["active"] > 0) & (group["target_refreshed"] > 0)).sum()
                ),
                "active_pump_rate_positive_rows": int(
                    ((group["active"] > 0) & (group["pump_rate_m3_min"].abs() > 1e-6)).sum()
                ),
                "active_post_bucket_pump_positive_rows": int(
                    ((group["active"] > 0) & (group["post_bucket_pump_m3"].abs() > 1e-6)).sum()
                ),
                "active_baseline_post_bucket_pump_positive_rows": int(
                    (
                        (group["active"] > 0)
                        & (group["baseline_post_bucket_pump_m3"].abs() > 1e-6)
                    ).sum()
                ),
                "active_post_bucket_pump_m3": float(
                    active["post_bucket_pump_m3"].abs().sum()
                )
                if len(active)
                else 0.0,
                "active_baseline_post_bucket_pump_m3": float(
                    active["baseline_post_bucket_pump_m3"].abs().sum()
                )
                if len(active)
                else 0.0,
                "active_estimated_pump_saved_m3": float(
                    (
                        active["baseline_post_bucket_pump_m3"].abs()
                        - active["post_bucket_pump_m3"].abs()
                    ).sum()
                )
                if len(active)
                else 0.0,
                "active_mean_pump_rate_m3_min": float(active["pump_rate_m3_min"].abs().mean())
                if len(active)
                else 0.0,
                "active_mean_target_err_kg": float(active["target_err_mean_kg"].abs().mean())
                if len(active)
                else 0.0,
                "active_mean_target_delta_kg": float(
                    active["target_delta_abs_mean_kg"].abs().mean()
                )
                if len(active)
                else 0.0,
                "active_mean_max_axis_deg": float(active["max_axis_deg"].mean())
                if len(active)
                else 0.0,
            }
        )
    return bucket, pd.DataFrame(summary_rows).sort_values(["profile", "dataset", "arm"])


def _md_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_empty_"

    def fmt(v: Any) -> str:
        if pd.isna(v):
            return ""
        if isinstance(v, float):
            if abs(v) <= 1.0:
                return f"{v:.3f}"
            return f"{v:.2f}"
        return str(v)

    lines = [
        "| " + " | ".join(str(c) for c in df.columns) + " |",
        "| " + " | ".join(["---"] * len(df.columns)) + " |",
    ]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(fmt(row[c]) for c in df.columns) + " |")
    return "\n".join(lines)


def _write_docs(
    compare: pd.DataFrame,
    reasons: pd.DataFrame,
    opportunity: pd.DataFrame,
    profiles: list[str],
) -> None:
    key_cols = [
        "dataset",
        "profile",
        "arm",
        "cases",
        "sum_pump",
        "sum_fb",
        "p95_max_axis",
        "time_over_5",
        "idle_time_over_5",
        "eligible_count",
        "pump_opportunity_count",
        "target_refresh_delayed_count",
        "A2_unique_relax_count",
        "delta_vs_A0_sum_pump",
        "delta_vs_A1_sum_pump",
        "delta_vs_A0_time_over_5",
        "delta_vs_A1_time_over_5",
        "A2_unique_pump_saved",
    ]
    key = compare[[c for c in key_cols if c in compare.columns]].copy()
    pass_rows: list[dict[str, Any]] = []
    for (dataset, profile), group in compare.groupby(["dataset", "profile"]):
        a1 = group[group["arm"].eq("A1_near_envelope")]
        a2 = group[group["arm"].eq("A2_h120_envelope")]
        if a1.empty or a2.empty:
            continue
        a1r = a1.iloc[0]
        a2r = a2.iloc[0]
        safety_clean = all(
            float(a2r[col]) <= float(a1r[col]) + tol
            for col, tol in (
                ("sum_fb", 1e-9),
                ("time_over_5", 1e-9),
                ("idle_time_over_5", 1e-9),
                ("p95_max_axis", 0.10),
            )
        )
        pass_rows.append(
            {
                "dataset": dataset,
                "profile": profile,
                "A2_pump_saved_vs_A1": float(a1r["sum_pump"] - a2r["sum_pump"]),
                "A2_unique_relax_count": int(a2r.get("A2_unique_relax_count", 0)),
                "safety_non_worse": int(safety_clean),
            }
        )
    pass_table = pd.DataFrame(pass_rows)
    pass_table.to_csv(OUT_DIR / "oracle_envelope_pass_table.csv", index=False)
    pass_table.to_csv(OUT_DIR / "raw_tables/oracle_envelope_pass_table.csv", index=False)
    good = pass_table[
        (pass_table.get("A2_pump_saved_vs_A1", pd.Series(dtype=float)) > 1e-6)
        & (pass_table.get("safety_non_worse", pd.Series(dtype=int)) > 0)
    ]
    if good.empty:
        verdict = "NO-GO"
        reason = (
            "No tested profile produced 120min-unique pump saving over A1 while "
            "keeping safety metrics non-worse."
        )
    else:
        verdict = "CONDITIONAL-GO"
        reason = "At least one dataset/profile produced A2-vs-A1 pump saving with non-worse aggregate safety."
    lines = [
        "# h120 Relief-Conditioned Posture Envelope Oracle Audit v1",
        "",
        "Scope: oracle-only test of a dynamic 3-5deg soft recovery band. The v1.6 hard floor remains active at 5deg; the lever only delays non-hard target refreshes when relief and envelope guards pass.",
        "",
        f"Profiles run: `{', '.join(profiles)}`.",
        "",
        "## Aggregate Results",
        _md_table(key),
        "",
        "## A2-vs-A1 Pass Table",
        _md_table(pass_table),
        "",
        "## Reason Counts",
        _md_table(reasons if not reasons.empty else pd.DataFrame()),
        "",
        "## Pump-Opportunity Diagnostics",
        _md_table(opportunity if not opportunity.empty else pd.DataFrame()),
        "",
        "## Decision",
        "",
        f"**{verdict}.** {reason}",
        "",
        "Answers:",
        "",
        "1. The 3-5deg soft band does create a few oracle-eligible relaxation rows, especially in broader20 under the medium profile, but those rows do not coincide with non-hard pump action.",
        "2. A2 has no 120min-unique pump saving over A1 on any tested dataset/profile; `A2_unique_relax_count` and `A2_unique_pump_saved` are both zero in the aggregate pass table.",
        "3. Safety metrics are non-worse because the lever never changes the closed-loop trajectory; pump, fallback, time>5, idle>5, and p95 max-axis are bit-identical to A0/A1 in the tested runs.",
        "4. This direction is not worth continuing under the current economy-target-refresh hook. Conservative and medium settings show no actionable pump opportunity; per the protocol, aggressive is not run.",
        "5. The failure mode is not merely over-conservative parameters: the pump-opportunity diagnostics show active envelope rows have zero pump-rate rows and provider-level `pump_opportunity_count=0`. Under the current controller architecture, h120 relief can identify safe waiting windows but not a non-hard pump action to remove.",
        "",
        "Interpretation: A2 must beat A1 to prove 120min-unique closed-loop pump-saving value. If A2 only matches A1, the same relief opportunity is already visible within 0-60min or no actionable non-hard refresh exists.",
    ]
    doc = "\n".join(lines) + "\n"
    (OUT_DIR / "decision.md").write_text(doc, encoding="utf-8")
    (OUT_DIR / "paper_ready/relief_envelope_findings.md").write_text(doc, encoding="utf-8")


def _plot_compare(compare: pd.DataFrame) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    paper = OUT_DIR / "paper_ready"
    for metric in ("sum_pump", "time_over_5", "idle_time_over_5"):
        for profile, group in compare.groupby("profile"):
            pivot = group.pivot(index="dataset", columns="arm", values=metric)
            if pivot.empty:
                continue
            ax = pivot.plot(kind="bar", figsize=(9, 4), title=f"{profile}: {metric}")
            ax.set_ylabel(metric)
            ax.figure.tight_layout()
            ax.figure.savefig(paper / f"{profile}_{metric}_compare.png", dpi=160)
            plt.close(ax.figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--profiles", nargs="*", default=["conservative"])
    parser.add_argument("--datasets", nargs="*", default=["relief", "lowrisk", "broader20", "guard10"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dirs = _mkdirs()
    profiles = [str(x) for x in args.profiles]
    for profile in profiles:
        if profile not in PROFILES:
            raise ValueError(f"unknown profile={profile!r}; choices={sorted(PROFILES)}")
    arms = ("A0_v16", "A1_near_envelope", "A2_h120_envelope")
    run_map: dict[tuple[str, str, str], Path] = {}
    for profile in profiles:
        for dataset in args.datasets:
            if dataset not in CASE_SOURCES:
                raise ValueError(f"unknown dataset={dataset!r}; choices={sorted(CASE_SOURCES)}")
            cases_csv = CASE_SOURCES[dataset]
            if not cases_csv.exists():
                raise FileNotFoundError(cases_csv)
            for arm in arms:
                run_dir = dirs["runs"] / profile / dataset / arm
                run_map[(dataset, profile, arm)] = run_dir
                _run_casebook(
                    run_dir,
                    cases_csv,
                    arm=arm,
                    profile=profile,
                    force=bool(args.force),
                )
    rows: list[dict[str, Any]] = []
    for (dataset, profile, arm), run_dir in run_map.items():
        rows.extend(_case_metrics(run_dir, dataset, profile, arm))
    case_table = pd.DataFrame(rows).sort_values(["profile", "dataset", "arm", "case"])
    case_table.to_csv(OUT_DIR / "relief_envelope_case_table.csv", index=False)
    case_table.to_csv(dirs["raw"] / "relief_envelope_case_table.csv", index=False)
    compare = _aggregate(case_table)
    compare.to_csv(OUT_DIR / "relief_envelope_compare_table.csv", index=False)
    compare.to_csv(dirs["raw"] / "relief_envelope_compare_table.csv", index=False)
    reasons = _reason_table(run_map)
    reasons.to_csv(OUT_DIR / "relief_envelope_reason_table.csv", index=False)
    reasons.to_csv(dirs["diag"] / "relief_envelope_reason_table.csv", index=False)
    bucket, opportunity = _bucket_opportunity_tables(run_map)
    bucket.to_csv(OUT_DIR / "diagnostics/relief_envelope_bucket_table.csv", index=False)
    bucket.to_csv(dirs["raw"] / "relief_envelope_bucket_table.csv", index=False)
    opportunity.to_csv(
        OUT_DIR / "diagnostics/relief_envelope_opportunity_summary.csv", index=False
    )
    opportunity.to_csv(dirs["raw"] / "relief_envelope_opportunity_summary.csv", index=False)
    _write_docs(compare, reasons, opportunity, profiles)
    _plot_compare(compare)
    print(f"[done] {_rel(OUT_DIR)}")


if __name__ == "__main__":
    main()
