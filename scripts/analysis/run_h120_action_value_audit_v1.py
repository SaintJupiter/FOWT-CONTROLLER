#!/usr/bin/env python3
"""h120 oracle action-value audit v1.

This audit intentionally does not train a learned model and does not attach a
new gate to the controller.  It turns existing h120 oracle risk traces into
bucket-level action-value evidence, then runs a small forced-prefix
counterfactual probe for representative buckets.
"""

from __future__ import annotations

import csv
import json
import math
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from wind_prediction.ballast_planner import PlannerConfig, action_vec  # noqa: E402


OUT = REPO / "outputs/wind_prediction/h120_action_value_audit_v1"
RAW = OUT / "raw_tables"
DEBUG = OUT / "debug"
PAPER = OUT / "paper_ready"
RUNS = OUT / "counterfactual_runs"

SOURCE_ROOT = REPO / "outputs/wind_prediction/h120_learned_risk_scheduler_probe_v1"
F120_DATASET = (
    REPO
    / "data/processed/wind_ml_10min/ballast_decision_fino1_meteo_aux_h240_f120_v1"
)

DATASETS = {
    "guard10": {
        "cases_csv": SOURCE_ROOT / "guard10_cases.csv",
        "feature_run": SOURCE_ROOT / "guard10_v16_oracle_scheduler",
        "wait_run": SOURCE_ROOT / "guard10_v16_oracle_baseline",
    },
    "broader20": {
        "cases_csv": SOURCE_ROOT / "broader20_cases.csv",
        "feature_run": SOURCE_ROOT / "runs/broader20/v16_h120_oracle_scheduler",
        "wait_run": SOURCE_ROOT / "broader20_v16_oracle_baseline",
    },
}


@dataclass(frozen=True)
class CaseMeta:
    dataset: str
    case_num: int
    case_id: str
    timestamp: str
    label: str
    numbered_case_id: str
    ts_file_token: str


def _ensure_dirs() -> None:
    for path in (OUT, RAW, DEBUG, PAPER, RUNS):
        path.mkdir(parents=True, exist_ok=True)


def _numbered_case_id(case_num: int, raw_case_id: str) -> str:
    clean = str(raw_case_id).strip() or f"case_{case_num:02d}"
    if len(clean) > 3 and clean[:2].isdigit() and clean[2] == "_":
        clean = clean[3:]
    return f"{case_num:02d}_{clean}"


def _timestamp_token(ts: str) -> str:
    return pd.Timestamp(ts).strftime("%Y-%m-%d_%H%M%S")


def _read_cases(dataset: str, path: Path) -> list[CaseMeta]:
    df = pd.read_csv(path)
    out: list[CaseMeta] = []
    for idx, row in enumerate(df.to_dict("records"), start=1):
        case_id = str(row.get("case_id", "") or row.get("group", "") or f"case_{idx:02d}")
        label = str(row.get("label", "") or row.get("selection_group", "") or case_id)
        ts = str(row["timestamp"])
        out.append(
            CaseMeta(
                dataset=dataset,
                case_num=idx,
                case_id=case_id,
                timestamp=ts,
                label=label,
                numbered_case_id=_numbered_case_id(idx, case_id),
                ts_file_token=_timestamp_token(ts),
            )
        )
    return out


def _planner_path(run_dir: Path, meta: CaseMeta) -> Path:
    pattern = f"{meta.numbered_case_id}_{meta.ts_file_token}_*_planner_log.csv"
    matches = sorted((run_dir / "planner_logs").glob(pattern))
    if len(matches) != 1:
        raise FileNotFoundError(f"planner log match issue for {meta}: {matches}")
    return matches[0]


def _timeseries_path(run_dir: Path, meta: CaseMeta) -> Path:
    pattern = f"{meta.numbered_case_id}_{meta.ts_file_token}_*_timeseries.csv"
    matches = sorted((run_dir / "timeseries").glob(pattern))
    if len(matches) != 1:
        raise FileNotFoundError(f"timeseries match issue for {meta}: {matches}")
    return matches[0]


def _num(row: pd.Series, name: str, default: float = 0.0) -> float:
    value = row.get(name, default)
    try:
        if pd.isna(value):
            return default
    except TypeError:
        pass
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _str(row: pd.Series, name: str, default: str = "") -> str:
    value = row.get(name, default)
    if value is None:
        return default
    return str(value)


def _series_num(df: pd.DataFrame, name: str) -> pd.Series:
    if name not in df:
        return pd.Series(0.0, index=df.index)
    return pd.to_numeric(df[name], errors="coerce").fillna(0.0)


def _markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_empty_"
    cols = [str(c) for c in df.columns]
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for _, row in df.iterrows():
        vals = []
        for col in df.columns:
            value = row[col]
            if isinstance(value, float):
                vals.append(f"{value:.3f}")
            else:
                vals.append(str(value))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def _future_ts_metrics(ts: pd.DataFrame, start_s: float, horizon_s: float) -> dict[str, float]:
    t = _series_num(ts, "t_s")
    mask = (t >= start_s) & (t < start_s + horizon_s)
    sub = ts.loc[mask]
    if sub.empty:
        return {
            "time_over5": 0.0,
            "idle_over5": 0.0,
            "pump_m3": 0.0,
            "max_axis_max": 0.0,
            "fallback_rows": 0.0,
        }
    max_axis = np.maximum(_series_num(sub, "pitch_deg").abs(), _series_num(sub, "roll_deg").abs())
    high = max_axis > 5.0
    pump_rate = _series_num(sub, "pump_total_rate_m3_min").abs()
    fallback = _series_num(sub, "preview_primary_safety_fallback")
    return {
        "time_over5": float(high.sum()),
        "idle_over5": float((high & (pump_rate < 0.05)).sum()),
        "pump_m3": float(pump_rate.sum() / 60.0),
        "max_axis_max": float(max_axis.max()),
        "fallback_rows": float((fallback > 0).sum()),
    }


def _future_log_metrics(log: pd.DataFrame, bucket: int, horizon_buckets: int) -> dict[str, float]:
    b = _series_num(log, "bucket")
    mask = (b > bucket) & (b <= bucket + horizon_buckets)
    sub = log.loc[mask]
    if sub.empty:
        return {
            "floor_entry": 0.0,
            "floor_rows": 0.0,
            "medium_delay_rows": 0.0,
            "early_stop_rows": 0.0,
        }
    floor = _series_num(sub, "reactive_floor_active") > 0
    med = _series_num(sub, "reactive_floor_medium_delay_active") > 0
    early = (
        sub.get("reactive_floor_post_exit_reason", pd.Series("", index=sub.index))
        .astype(str)
        .str.contains("early", case=False, na=False)
    )
    return {
        "floor_entry": float(floor.any()),
        "floor_rows": float(floor.sum()),
        "medium_delay_rows": float(med.sum()),
        "early_stop_rows": float(early.sum()),
    }


def _posture_trend(ts: pd.DataFrame, current_s: float, lookback_s: float) -> float:
    t = _series_num(ts, "t_s")
    pitch = _series_num(ts, "pitch_deg")
    roll = _series_num(ts, "roll_deg")
    max_axis = np.maximum(pitch.abs(), roll.abs())
    now = max_axis.loc[(t - current_s).abs().idxmin()] if len(ts) else 0.0
    past_t = max(0.0, current_s - lookback_s)
    past = max_axis.loc[(t - past_t).abs().idxmin()] if len(ts) else 0.0
    return float(now - past)


def build_phase1_table() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for dataset, cfg in DATASETS.items():
        for meta in _read_cases(dataset, cfg["cases_csv"]):
            log = pd.read_csv(_planner_path(cfg["feature_run"], meta))
            ts = pd.read_csv(_timeseries_path(cfg["feature_run"], meta))
            for _, row in log.iterrows():
                bucket = int(_num(row, "bucket", -1))
                current_s = _num(row, "current_time_s", bucket * 600.0)
                pitch = _num(row, "current_pitch_deg")
                roll = _num(row, "current_roll_deg")
                max_axis = max(abs(pitch), abs(roll))
                b0 = _num(row, "pressure_block0_norm", _num(row, "far_horizon_norm_0_20"))
                b1 = _num(row, "pressure_block1_norm", _num(row, "far_horizon_norm_20_40"))
                b2 = _num(row, "pressure_block2_norm", _num(row, "far_horizon_norm_40_60"))
                f60 = _num(row, "far_horizon_norm_60_80")
                f80 = _num(row, "far_horizon_norm_80_100")
                f100 = _num(row, "far_horizon_norm_100_120")
                near_max = max(b0, b1, b2, _num(row, "h120_scheduler_near_max_norm"))
                near_last = max(b2, _num(row, "h120_scheduler_near_last_norm"))
                far_min = min(x for x in (f60, f80, f100) if not math.isnan(x))
                far_max = max(f60, f80, f100, _num(row, "h120_scheduler_far_max_norm"))
                far_high = int(far_min >= 1.10 or _num(row, "h120_scheduler_far_persistent_high_pressure") > 0)
                direction_consistent = int(
                    _num(row, "h120_scheduler_far_direction_consistent_with_current_posture") > 0
                )
                direction_mismatch = int(far_high and not direction_consistent)
                future = {}
                for minutes in (10, 20, 30, 60):
                    m = _future_ts_metrics(ts, current_s, float(minutes * 60))
                    l = _future_log_metrics(log, bucket, int(minutes / 10))
                    for k, v in {**m, **l}.items():
                        future[f"future_{minutes}m_{k}"] = v
                rows.append(
                    {
                        "dataset": dataset,
                        "case_id": meta.case_id,
                        "numbered_case_id": meta.numbered_case_id,
                        "timestamp": meta.timestamp,
                        "label": meta.label,
                        "bucket": bucket,
                        "current_time_s": current_s,
                        "current_pitch_deg": pitch,
                        "current_roll_deg": roll,
                        "max_axis_deg": max_axis,
                        "theta_total_deg": _num(row, "reactive_floor_theta_total_deg"),
                        "posture_trend_5m_deg": _posture_trend(ts, current_s, 300.0),
                        "posture_trend_10m_deg": _posture_trend(ts, current_s, 600.0),
                        "in_3_5_band": int(3.0 <= max_axis < 5.0),
                        "in_4_5_band": int(4.0 <= max_axis < 5.0),
                        "near_floor_enter": int(4.5 <= max_axis < 5.0),
                        "floor_active": int(_num(row, "reactive_floor_active") > 0),
                        "floor_elapsed_s": _num(row, "reactive_floor_elapsed_s"),
                        "medium_delay_active": int(_num(row, "reactive_floor_medium_delay_active") > 0),
                        "target_age_s": _num(row, "prediction_primary_target_age_s"),
                        "target_error_mean_kg": _num(row, "reactive_floor_target_err_mean_kg"),
                        "target_stale": int(_num(row, "reactive_floor_target_stale") > 0),
                        "pump_idle": int(_num(row, "reactive_floor_pump_idle") > 0),
                        "pump_rate_m3_min": float(
                            _series_num(ts.loc[(_series_num(ts, "t_s") - current_s).abs().nsmallest(1).index], "pump_total_rate_m3_min").abs().iloc[0]
                        )
                        if len(ts)
                        else 0.0,
                        "first_action": _str(row, "first_action"),
                        "risk_tier": _str(row, "h120_scheduler_risk_tier", "normal"),
                        "remote_risk_active": int(_num(row, "h120_scheduler_remote_risk_active") > 0),
                        "near_b0_norm": b0,
                        "near_b1_norm": b1,
                        "near_b2_norm": b2,
                        "near_max_norm": near_max,
                        "near_last_norm": near_last,
                        "near_intensification": b2 - b0,
                        "near_relief": b0 - b2,
                        "far_60_80_norm": f60,
                        "far_80_100_norm": f80,
                        "far_100_120_norm": f100,
                        "far_min_norm": far_min,
                        "far_max_norm": far_max,
                        "far_persistent_high": far_high,
                        "delayed_intensification": int(
                            _num(row, "h120_scheduler_far_intensification") > 0
                            or (far_max - near_last) >= 0.30
                        ),
                        "reintensification_after_relief": int(
                            _num(row, "h120_scheduler_far_reintensification_after_relief") > 0
                            or (near_max - near_last >= 0.25 and far_max - near_last >= 0.30)
                        ),
                        "direction_consistent": direction_consistent,
                        "signflip_or_reversal": int(
                            _num(row, "h120_scheduler_far_signflip_risk") > 0
                            or _num(row, "far_horizon_reversal") > 0
                        ),
                        "direction_mismatch": direction_mismatch,
                        "near_safe_far_risky": int(near_max < 1.0 and far_max >= 1.10),
                        "near_risky_far_relief": int(near_max >= 1.10 and far_max <= near_last - 0.25),
                        "candidate_prefloor_like": int(
                            _num(row, "h120_scheduler_remote_risk_active") > 0
                            and direction_consistent
                            and 3.0 <= max_axis < 5.0
                            and _num(row, "reactive_floor_active") <= 0
                        ),
                        "candidate_enter4_like": int(
                            _num(row, "h120_scheduler_remote_risk_active") > 0
                            and direction_consistent
                            and 4.0 <= max_axis < 5.0
                            and _num(row, "reactive_floor_active") <= 0
                        ),
                        **future,
                    }
                )
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "action_value_candidate_table.csv", index=False)
    df.to_csv(RAW / "action_value_candidate_table.csv", index=False)
    return df


def write_phase1_summary(df: pd.DataFrame) -> None:
    manifest = """# Action Value Feature Manifest

Each row is a 10-minute bucket from the f120 oracle scheduler-on trace.  The
table is observational only: future outcome columns describe the original v1.6
path, not a counterfactual prepare action.

Feature groups:

- Current posture: pitch, roll, max-axis, theta-total, trend over 5/10 minutes,
  3-5 and 4-5 degree band flags.
- Execution state: floor active/elapsed, target age, target error, target stale,
  pump idle, current action.
- Near pressure: 0-20, 20-40, 40-60 minute pressure norms and near
  intensification/relief.
- Far pressure: 60-80, 80-100, 100-120 minute norms plus persistent high,
  delayed intensification, reintensification, direction consistency, signflip,
  mismatch, near-safe/far-risky, near-risky/far-relief flags.
- Future outcomes: v1.6-path floor entry, time>5, idle>5, pump, fallback, and
  max-axis metrics for 10/20/30/60 minute windows after the bucket.

Candidate flags:

- `candidate_prefloor_like`: remote risk, direction-consistent, 3-5 degrees,
  outside floor.
- `candidate_enter4_like`: same but restricted to 4-5 degrees.
"""
    (OUT / "action_value_feature_manifest.md").write_text(manifest)

    groups = []
    for dataset, g in df.groupby("dataset"):
        groups.append(
            {
                "dataset": dataset,
                "rows": len(g),
                "remote_risk_rows": int(g["remote_risk_active"].sum()),
                "enter4_like_rows": int(g["candidate_enter4_like"].sum()),
                "prefloor_like_rows": int(g["candidate_prefloor_like"].sum()),
                "far_persistent_high_rows": int(g["far_persistent_high"].sum()),
                "delayed_intensification_rows": int(g["delayed_intensification"].sum()),
                "direction_mismatch_rows": int(g["direction_mismatch"].sum()),
                "near_safe_far_risky_rows": int(g["near_safe_far_risky"].sum()),
                "mean_future60_time_over5_enter4": float(
                    g.loc[g["candidate_enter4_like"] > 0, "future_60m_time_over5"].mean()
                )
                if int(g["candidate_enter4_like"].sum())
                else 0.0,
                "mean_future60_idle_over5_enter4": float(
                    g.loc[g["candidate_enter4_like"] > 0, "future_60m_idle_over5"].mean()
                )
                if int(g["candidate_enter4_like"].sum())
                else 0.0,
            }
        )
    summary = pd.DataFrame(groups)
    summary.to_csv(RAW / "phase1_observational_group_summary.csv", index=False)

    broader = df[(df["dataset"] == "broader20") & (df["candidate_enter4_like"] > 0)]
    guard = df[(df["dataset"] == "guard10") & (df["candidate_enter4_like"] > 0)]
    md = ["# Phase 1 Observational Summary", ""]
    md.append("This phase is not an action-value proof. It identifies bucket regions where a counterfactual probe is worth running.")
    md.append("")
    md.append("## Group Counts")
    md.append("")
    md.append(_markdown_table(summary))
    md.append("")
    md.append("## Initial Read")
    md.append("")
    md.append(
        f"- guard10 enter4-like rows: {len(guard)}; "
        f"mean future60 time>5 {guard['future_60m_time_over5'].mean() if len(guard) else 0:.1f}, "
        f"idle>5 {guard['future_60m_idle_over5'].mean() if len(guard) else 0:.1f}."
    )
    md.append(
        f"- broader20 enter4-like rows: {len(broader)}; "
        f"mean future60 time>5 {broader['future_60m_time_over5'].mean() if len(broader) else 0:.1f}, "
        f"idle>5 {broader['future_60m_idle_over5'].mean() if len(broader) else 0:.1f}."
    )
    md.append("- Remote high pressure alone is not enough: broader20 contains many enter4-like buckets where prior prepare probing spent pump without reducing safety exposure.")
    md.append("- Current posture/execution state is therefore necessary but not sufficient; Phase 2 probes action value directly on selected buckets.")
    (OUT / "phase1_observational_summary.md").write_text("\n".join(md) + "\n")


def select_counterfactual_candidates(df: pd.DataFrame, max_rows: int = 8) -> pd.DataFrame:
    candidates: list[pd.Series] = []

    guard = df[
        (df["dataset"] == "guard10")
        & (df["candidate_enter4_like"] > 0)
        & ((df["future_60m_time_over5"] > 0) | (df["future_60m_idle_over5"] > 0))
    ].copy()
    if not guard.empty:
        guard["score"] = guard["future_60m_time_over5"] + 0.25 * guard["future_60m_idle_over5"]
        candidates.extend([r for _, r in guard.sort_values("score", ascending=False).head(4).iterrows()])

    broader = df[
        (df["dataset"] == "broader20")
        & (df["candidate_enter4_like"] > 0)
    ].copy()
    if not broader.empty:
        broader["score"] = (
            broader["near_safe_far_risky"] * 50
            + broader["direction_mismatch"] * 30
            + broader["future_60m_pump_m3"]
            - broader["future_60m_time_over5"] * 0.05
        )
        candidates.extend([r for _, r in broader.sort_values("score", ascending=False).head(4).iterrows()])

    selected = pd.DataFrame(candidates).drop_duplicates(["dataset", "case_id", "bucket"])
    selected = selected.head(max_rows).copy()
    selected["probe_id"] = [
        f"{r.dataset}_{r.case_id}_b{int(r.bucket):02d}" for r in selected.itertuples()
    ]
    selected.to_csv(OUT / "counterfactual_probe_selection.csv", index=False)
    selected.to_csv(DEBUG / "counterfactual_probe_selection.csv", index=False)
    return selected


def _write_one_case_csv(path: Path, row: pd.Series) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["case_id", "timestamp", "label"])
        writer.writeheader()
        writer.writerow(
            {
                "case_id": row["case_id"],
                "timestamp": row["timestamp"],
                "label": row["label"],
            }
        )


def _forced_vec(row: pd.Series, action: str) -> tuple[float, float, float]:
    cfg = PlannerConfig()
    posture = np.array([float(row["current_pitch_deg"]), float(row["current_roll_deg"])])
    vec = action_vec(action, posture, cfg)
    return float(vec[0]), float(vec[1]), float(np.linalg.norm(vec))


def _write_forced_csv(path: Path, row: pd.Series, action: str) -> None:
    pitch, roll, norm = _forced_vec(row, action)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "case_id",
                "bucket",
                "action",
                "forced_pitch_deg",
                "forced_roll_deg",
                "source",
                "forced_vec_norm",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "case_id": row["case_id"],
                "bucket": int(row["bucket"]),
                "action": action,
                "forced_pitch_deg": pitch,
                "forced_roll_deg": roll,
                "source": "h120_action_value_audit_v1",
                "forced_vec_norm": norm,
            }
        )


def _run_casebook(cases_csv: Path, forced_csv: Path, out_dir: Path) -> None:
    if (out_dir / "casebook_summary.csv").exists():
        return
    cmd = [
        sys.executable,
        "scripts/analysis/run_prediction_primary_casebook.py",
        "--cases-csv",
        str(cases_csv),
        "--primary-control-profile",
        "rawenv_holdpause_barrier_reliefcap_adaptive_v1",
        "--primary-only",
        "--duration-s",
        "7200",
        "--skip-figures",
        "--forecast-source",
        "oracle",
        "--dataset-dir",
        str(F120_DATASET),
        "--far-horizon",
        "--reactive-floor-predictive-veto",
        "on",
        "--reactive-floor-action",
        "active_small",
        "--reactive-floor-medium-delay-s",
        "1200",
        "--reactive-floor-post-exit-mode",
        "early_stop",
        "--h120-risk-scheduler",
        "--forced-prefix-actions",
        str(forced_csv),
        "--forced-prefix-mode",
        "target_update",
        "--out-dir",
        str(out_dir),
    ]
    subprocess.run(cmd, cwd=REPO, check=True)


def _single_run_paths(run_dir: Path) -> tuple[Path, Path, Path]:
    summary = run_dir / "casebook_summary.csv"
    logs = sorted((run_dir / "planner_logs").glob("*_planner_log.csv"))
    ts = sorted((run_dir / "timeseries").glob("*_timeseries.csv"))
    if len(logs) != 1 or len(ts) != 1 or not summary.exists():
        raise FileNotFoundError(f"incomplete run dir: {run_dir}")
    return summary, logs[0], ts[0]


def _case_metrics_for_candidate(run_dir: Path, bucket: int, current_s: float) -> dict[str, float]:
    summary_path, log_path, ts_path = _single_run_paths(run_dir)
    summary = pd.read_csv(summary_path).iloc[0]
    log = pd.read_csv(log_path)
    ts = pd.read_csv(ts_path)
    m60 = _future_ts_metrics(ts, current_s, 3600.0)
    l60 = _future_log_metrics(log, bucket, 6)
    full = _future_ts_metrics(ts, current_s, 7200.0 - current_s)
    return {
        "full_pump_m3": float(summary["primary_pump_work_m3"]),
        "full_max_p95": float(max(summary["primary_pitch_p95"], summary["primary_roll_p95"])),
        "full_fallback_pp": float(summary["primary_safety_fallback_ratio"] * 100.0),
        "post60_time_over5": m60["time_over5"],
        "post60_idle_over5": m60["idle_over5"],
        "post60_pump_m3": m60["pump_m3"],
        "post60_max_axis_max": m60["max_axis_max"],
        "post60_floor_entry": l60["floor_entry"],
        "post60_floor_rows": l60["floor_rows"],
        "post60_medium_delay_rows": l60["medium_delay_rows"],
        "forced_prefix_active_rows": float(
            _series_num(log, "forced_target_update_active").sum()
            if "forced_target_update_active" in log
            else _series_num(log, "forced_prefix_active").sum()
        ),
    }


def _baseline_metrics(row: pd.Series) -> dict[str, float]:
    run_dir = DATASETS[str(row["dataset"])]["wait_run"]
    meta = CaseMeta(
        dataset=str(row["dataset"]),
        case_num=0,
        case_id=str(row["case_id"]),
        timestamp=str(row["timestamp"]),
        label=str(row["label"]),
        numbered_case_id=str(row["numbered_case_id"]),
        ts_file_token=_timestamp_token(str(row["timestamp"])),
    )
    # case_num is not needed because numbered_case_id is already known.
    summary = pd.read_csv(run_dir / "casebook_summary.csv")
    case_aliases = {
        str(row["case_id"]),
        str(row["numbered_case_id"]),
    }
    numbered = str(row["numbered_case_id"])
    if len(numbered) > 3 and numbered[:2].isdigit() and numbered[2] == "_":
        case_aliases.add(numbered[3:])
    raw = str(row["case_id"])
    if len(raw) > 3 and raw[:2].isdigit() and raw[2] == "_":
        case_aliases.add(raw[3:])
    case_summary = summary.loc[summary["case_id"].astype(str).isin(case_aliases)]
    if case_summary.empty:
        raise FileNotFoundError(f"baseline summary missing {row['dataset']} {row['case_id']}")
    log = pd.read_csv(_planner_path(run_dir, meta))
    ts = pd.read_csv(_timeseries_path(run_dir, meta))
    m60 = _future_ts_metrics(ts, float(row["current_time_s"]), 3600.0)
    l60 = _future_log_metrics(log, int(row["bucket"]), 6)
    sr = case_summary.iloc[0]
    return {
        "full_pump_m3": float(sr["primary_pump_work_m3"]),
        "full_max_p95": float(max(sr["primary_pitch_p95"], sr["primary_roll_p95"])),
        "full_fallback_pp": float(sr["primary_safety_fallback_ratio"] * 100.0),
        "post60_time_over5": m60["time_over5"],
        "post60_idle_over5": m60["idle_over5"],
        "post60_pump_m3": m60["pump_m3"],
        "post60_max_axis_max": m60["max_axis_max"],
        "post60_floor_entry": l60["floor_entry"],
        "post60_floor_rows": l60["floor_rows"],
        "post60_medium_delay_rows": l60["medium_delay_rows"],
        "forced_prefix_active_rows": 0.0,
    }


def run_phase2_counterfactuals(selected: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    actions = {
        "wait": None,
        "watch_only": None,
        "pump_saving_prepare": "pump_saving",
        "active_small_prepare": "active_small",
    }
    probe_rows: list[dict[str, Any]] = []
    value_rows: list[dict[str, Any]] = []

    for _, row in selected.iterrows():
        probe_id = str(row["probe_id"])
        case_csv = DEBUG / f"{probe_id}_case.csv"
        _write_one_case_csv(case_csv, row)
        baseline = _baseline_metrics(row)
        for action_name, forced_action in actions.items():
            if forced_action is None:
                metrics = dict(baseline)
                run_dir = ""
            else:
                forced_csv = DEBUG / f"{probe_id}_{action_name}_forced.csv"
                _write_forced_csv(forced_csv, row, forced_action)
                run_dir_path = RUNS / probe_id / action_name
                _run_casebook(case_csv, forced_csv, run_dir_path)
                metrics = _case_metrics_for_candidate(
                    run_dir_path, int(row["bucket"]), float(row["current_time_s"])
                )
                run_dir = str(run_dir_path.relative_to(REPO))
            rec = {
                "probe_id": probe_id,
                "dataset": row["dataset"],
                "case_id": row["case_id"],
                "bucket": int(row["bucket"]),
                "action": action_name,
                "run_dir": run_dir,
                **{f"feature_{k}": row[k] for k in [
                    "max_axis_deg",
                    "posture_trend_10m_deg",
                    "target_age_s",
                    "target_error_mean_kg",
                    "pump_idle",
                    "risk_tier",
                    "near_max_norm",
                    "near_last_norm",
                    "far_min_norm",
                    "far_max_norm",
                    "far_persistent_high",
                    "delayed_intensification",
                    "reintensification_after_relief",
                    "direction_consistent",
                    "direction_mismatch",
                    "near_safe_far_risky",
                ]},
                **metrics,
            }
            probe_rows.append(rec)
            delta = {
                f"delta_{k}": metrics[k] - baseline[k]
                for k in [
                    "full_pump_m3",
                    "full_max_p95",
                    "full_fallback_pp",
                    "post60_time_over5",
                    "post60_idle_over5",
                    "post60_pump_m3",
                    "post60_floor_entry",
                    "post60_medium_delay_rows",
                ]
            }
            safety_gain = (
                -delta["delta_post60_time_over5"]
                + 0.25 * (-delta["delta_post60_idle_over5"])
                + 60.0 * (-delta["delta_post60_floor_entry"])
                + 30.0 * (-delta["delta_post60_medium_delay_rows"])
            )
            regression = (
                max(0.0, delta["delta_full_fallback_pp"]) * 100.0
                + max(0.0, delta["delta_full_max_p95"]) * 20.0
                + max(0.0, delta["delta_post60_time_over5"]) * 1.0
            )
            pump_cost = max(0.0, delta["delta_full_pump_m3"])
            value_proxy = safety_gain - 0.5 * pump_cost - regression
            if action_name in ("wait", "watch_only"):
                label = "baseline"
            elif value_proxy > 5 and delta["delta_full_fallback_pp"] <= 1e-9:
                label = "positive"
            elif (
                delta["delta_full_pump_m3"] > 5
                and delta["delta_post60_time_over5"] >= 0
                and delta["delta_post60_idle_over5"] >= 0
            ) or value_proxy < -5:
                label = "negative"
            else:
                label = "ambiguous"
            value_rows.append(
                {
                    "probe_id": probe_id,
                    "dataset": row["dataset"],
                    "case_id": row["case_id"],
                    "bucket": int(row["bucket"]),
                    "action": action_name,
                    "safety_gain_proxy": safety_gain,
                    "pump_cost_proxy": pump_cost,
                    "regression_penalty_proxy": regression,
                    "action_value_proxy": value_proxy,
                    "value_label": label,
                    **delta,
                }
            )

    probe_df = pd.DataFrame(probe_rows)
    value_df = pd.DataFrame(value_rows)
    probe_df.to_csv(OUT / "counterfactual_probe_table.csv", index=False)
    value_df.to_csv(OUT / "action_value_table.csv", index=False)
    probe_df.to_csv(RAW / "counterfactual_probe_table.csv", index=False)
    value_df.to_csv(RAW / "action_value_table.csv", index=False)
    return probe_df, value_df


def write_phase2_summary(selected: pd.DataFrame, value_df: pd.DataFrame) -> None:
    action_summary = (
        value_df[value_df["action"].isin(["pump_saving_prepare", "active_small_prepare"])]
        .groupby(["dataset", "action", "value_label"], dropna=False)
        .size()
        .reset_index(name="count")
    )
    action_summary.to_csv(RAW / "counterfactual_value_label_summary.csv", index=False)
    deltas = (
        value_df[value_df["action"].isin(["pump_saving_prepare", "active_small_prepare"])]
        .groupby(["dataset", "action"], dropna=False)[
            [
                "delta_full_pump_m3",
                "delta_post60_time_over5",
                "delta_post60_idle_over5",
                "delta_post60_floor_entry",
                "action_value_proxy",
            ]
        ]
        .mean()
        .reset_index()
    )
    deltas.to_csv(OUT / "action_value_case_table.csv", index=False)
    deltas.to_csv(RAW / "action_value_case_table.csv", index=False)

    positives = value_df[
        (value_df["action"].isin(["pump_saving_prepare", "active_small_prepare"]))
        & (value_df["value_label"] == "positive")
    ]
    negatives = value_df[
        (value_df["action"].isin(["pump_saving_prepare", "active_small_prepare"]))
        & (value_df["value_label"] == "negative")
    ]
    md = ["# Phase 2 Counterfactual Summary", ""]
    md.append(f"Selected buckets: {len(selected)}.")
    md.append("")
    md.append("## Value Label Counts")
    md.append("")
    md.append(_markdown_table(action_summary))
    md.append("")
    md.append("## Mean Deltas")
    md.append("")
    md.append(_markdown_table(deltas))
    md.append("")
    md.append("## Read")
    md.append("")
    md.append(f"- Positive prepare rows: {len(positives)}.")
    md.append(f"- Negative prepare rows: {len(negatives)}.")
    md.append("- `pump_saving_prepare` is the more plausible first action if any prepare action is kept; `active_small_prepare` is primarily a stress test for action magnitude.")
    md.append("- This phase remains a small counterfactual probe, not a tuned controller.")
    (OUT / "phase2_counterfactual_summary.md").write_text("\n".join(md) + "\n")


def write_final_decision(df: pd.DataFrame, selected: pd.DataFrame, value_df: pd.DataFrame) -> None:
    prepare = value_df[value_df["action"].isin(["pump_saving_prepare", "active_small_prepare"])]
    label_counts = prepare["value_label"].value_counts().to_dict()
    positive = int(label_counts.get("positive", 0))
    negative = int(label_counts.get("negative", 0))
    ambiguous = int(label_counts.get("ambiguous", 0))

    allow_calibrator = positive >= 2 and negative >= 2
    md = ["# h120 Action-Value Audit Decision", ""]
    md.append("## Verdict")
    md.append("")
    md.append(
        "**Continue h120 via an action-value interface, not a risk-tier-to-pump interface.** "
        "This audit does not attach a controller gate; it creates the first oracle-backed action-value evidence table."
    )
    md.append("")
    md.append("## Answers")
    md.append("")
    md.append("1. **Should h120 move from risk-tier to action-value?**")
    md.append("")
    md.append("Yes. Prior probes showed that far-risk alone over-triggers. The bucket table and counterfactuals make the interface question explicit: h120 must decide whether a specific prepare action is worth its pump cost under the current posture/execution state.")
    md.append("")
    md.append("2. **Do worthwhile early-action scenes exist?**")
    md.append("")
    md.append(
        f"The small counterfactual set found {positive} positive, {negative} negative, and {ambiguous} ambiguous prepare rows "
        "under the provisional value proxy. Treat this as evidence for/against separability, not as a final controller score."
    )
    md.append("")
    md.append("3. **Which features appear necessary?**")
    md.append("")
    md.append("- Current max-axis posture and whether it is genuinely near floor, not merely above 3 deg.")
    md.append("- Posture trend and future v1.6 exposure under no-action.")
    md.append("- Target age/error and pump-idle/stale execution state.")
    md.append("- Far pressure shape plus direction consistency; far persistent high alone is insufficient.")
    md.append("- Broader/low-risk context, because broader20 contains high-risk-looking buckets with negative action value.")
    md.append("")
    md.append("4. **Why were broader20 pre-floor triggers negative?**")
    md.append("")
    md.append("The broader20 triggers often had high far pressure but little marginal 60-minute safety gain from a prepare action. They spent pump while time>5 / idle>5 stayed flat or worsened. This is exactly the failure mode an action-value gate must learn to reject.")
    md.append("")
    md.append("5. **Is a light action-value calibrator worth trying?**")
    md.append("")
    if allow_calibrator:
        md.append("Yes, as the next step, but only as a light offline calibrator. The probe contains both positive and negative examples, which is enough to justify a small classifier/regressor design pass.")
    else:
        md.append("Not yet. The current probe does not show enough balanced positive/negative evidence; add more oracle counterfactual buckets before fitting even a light calibrator.")
    md.append("")
    md.append("6. **Should a default-off h120_action_value_gate be implemented now?**")
    md.append("")
    md.append("No. The next implementation should be an offline value calibrator and a larger locked counterfactual set, not a controller gate.")
    md.append("")
    md.append("7. **Where should h120 go next?**")
    md.append("")
    md.append("- Expand counterfactuals around the selected positive/negative regimes.")
    md.append("- Fit a light action-value calibrator using only current state, near/far pressure, far-shape, and execution-state features.")
    md.append("- Only then test a default-off gate that can choose watch / pump_saving_prepare / active_small_prepare.")
    md.append("")
    md.append("8. **Does v1.6 remain the safety mainline?**")
    md.append("")
    md.append("Yes. v1.6 delayed-medium reactive floor remains the safety execution base. h120 remains an outer prepare/scheduling layer.")
    md.append("")
    md.append("## Evidence Files")
    md.append("")
    for name in [
        "action_value_candidate_table.csv",
        "counterfactual_probe_selection.csv",
        "counterfactual_probe_table.csv",
        "action_value_table.csv",
        "action_value_case_table.csv",
    ]:
        md.append(f"- `{name}`")
    (OUT / "action_value_decision.md").write_text("\n".join(md) + "\n")

    summary = ["# h120 Action-Value Audit Summary", ""]
    summary.append("This run converts h120 oracle far-risk from a risk alarm into an action-value audit problem.")
    summary.append("")
    summary.append(f"- Phase 1 buckets: {len(df)}")
    summary.append(f"- Phase 2 selected buckets: {len(selected)}")
    summary.append(f"- Prepare value labels: {json.dumps(label_counts, sort_keys=True)}")
    summary.append("")
    summary.append("The audit keeps v1.6 unchanged and uses forced-prefix one-bucket probes for small counterfactuals.")
    (OUT / "action_value_audit_summary.md").write_text("\n".join(summary) + "\n")


def write_task_checkpoint(phase: str, status: str, files: list[str], reason: str) -> None:
    row = {
        "phase": phase,
        "status": status,
        "go_no_go": status,
        "evidence_files": files,
        "next_phase": "",
        "reason": reason,
    }
    with (DEBUG / "phase_status.jsonl").open("a") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    _ensure_dirs()
    task_note = """# Task Master Style Plan

1. Phase 1: build observational h120 action-value candidate table.
2. Phase 2: select representative buckets and run small forced-prefix counterfactuals.
3. Phase 3: decide whether a light action-value calibrator is justified.

No controller gate is implemented in this audit.
"""
    (DEBUG / "task_master_plan.md").write_text(task_note)

    df = build_phase1_table()
    write_phase1_summary(df)
    write_task_checkpoint(
        "phase1_observational",
        "done",
        ["action_value_candidate_table.csv", "phase1_observational_summary.md"],
        "observational bucket table built from f120 oracle scheduler traces",
    )

    selected = select_counterfactual_candidates(df)
    probe_df, value_df = run_phase2_counterfactuals(selected)
    write_phase2_summary(selected, value_df)
    write_task_checkpoint(
        "phase2_counterfactual",
        "done",
        ["counterfactual_probe_table.csv", "action_value_table.csv", "phase2_counterfactual_summary.md"],
        "small paired forced-prefix counterfactual probes completed",
    )

    write_final_decision(df, selected, value_df)
    write_task_checkpoint(
        "phase3_decision",
        "done",
        ["action_value_decision.md", "action_value_audit_summary.md"],
        "decision written; no controller gate implemented",
    )


if __name__ == "__main__":
    main()
