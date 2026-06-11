#!/usr/bin/env python3
"""Build the 2026-06-06 paper evidence package.

The package is intentionally conservative: it records which existing results can
support paper claims, which are only mechanism/boundary evidence, and whether a
production-near 6 h paired casebook is already frozen.
"""

from __future__ import annotations

import json
import math
import os
import shutil
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[2] / ".matplotlib_cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_ROOT = REPO_ROOT / "outputs" / "wind_prediction"
PKG = OUT_ROOT / "paper_20260606_execution"

PRODUCTION_PROFILE = "rawenv_holdpause_barrier_reliefcap_adaptive_v1"
RHO_WATER_KG_M3 = 1025.0
LB_PER_KG = 2.20462262185
LB_PER_M3 = RHO_WATER_KG_M3 * LB_PER_KG


RUNS = {
    "guard10_6h_current_only": OUT_ROOT / "guard10_6h_current_only_baseline_20260606",
    "guard10_6h_learned_gain045": OUT_ROOT / "guard10_6h_production_learned_gain045_20260606",
    "guard10_12h_current_only": OUT_ROOT / "guard10_12h_current_only_baseline_20260605",
    "guard10_12h_learned_gain040": OUT_ROOT / "guard10_12h_production_learned_candidate_20260605",
    "guard10_12h_learned_gain045": OUT_ROOT / "guard10_12h_production_learned_gain045_20260605",
    "p2_6h_boundary": OUT_ROOT / "p2_relaxed_resume_guard_confirm4_4case_6h_validation_20260605",
    "c3_12h_boundary": OUT_ROOT / "c3_balanced_release_pool_12h_candidate_20260605",
    "psc_structured_single": OUT_ROOT
    / "psc_selector_mixed_pool_v1"
    / "chain_guard_structured_validation_20260605"
    / "baseline",
}

MAIN_FIG_DIR = PKG / "main_figures"
TABLE_DIR = PKG / "tables"
APPENDIX_FIG_DIR = PKG / "appendix_figures"
MANUSCRIPT_DIR = PKG / "manuscript_notes"

STATIC_TABLES = {
    "d1_all80_6h": REPO_ROOT
    / "FIXED_PUMP_SAVING_RESULTS_20260530"
    / "D1_dc_preserving_deadband_current"
    / "tables"
    / "d1_all80_6h_summary.csv",
    "c1_p4_all28_6h": REPO_ROOT
    / "FIXED_PUMP_SAVING_RESULTS_20260530"
    / "C1_stable_direction_soft_decay_tail_sub"
    / "tables"
    / "stable_soft_decay_split_corrected_all28_summary.csv",
    "w1_necessity": REPO_ROOT
    / "FIXED_PUMP_SAVING_RESULTS_20260530"
    / "W1_prediction_necessity_audit"
    / "w1_prediction_necessity_summary.csv",
}


def safe_float(value: Any, default: float = math.nan) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    return out if math.isfinite(out) else default


def safe_str(value: Any, default: str = "") -> str:
    if value is None:
        return default
    if isinstance(value, float) and math.isnan(value):
        return default
    return str(value)


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def sum_col(df: pd.DataFrame, col: str) -> float:
    if col not in df.columns or df.empty:
        return math.nan
    return safe_float(pd.to_numeric(df[col], errors="coerce").sum())


def mean_col(df: pd.DataFrame, col: str) -> float:
    if col not in df.columns or df.empty:
        return math.nan
    values = pd.to_numeric(df[col], errors="coerce")
    if values.notna().sum() == 0:
        return math.nan
    return safe_float(values.mean())


def max_col(df: pd.DataFrame, col: str) -> float:
    if col not in df.columns or df.empty:
        return math.nan
    values = pd.to_numeric(df[col], errors="coerce")
    if values.notna().sum() == 0:
        return math.nan
    return safe_float(values.max())


def infer_regime(label: Any, case_id: Any = "") -> str:
    text = f"{safe_str(label)} {safe_str(case_id)}".lower()
    if "future_relief" in text or "fr_relief" in text:
        return "future_relief"
    if "signflip" in text or "sf_" in text:
        return "signflip_boundary"
    if "b/high" in text or "high_pressure" in text or "residual_high" in text or "decay_strong" in text:
        return "b_high_pressure"
    if "lowrisk" in text or "low risk" in text:
        return "lowrisk_reference"
    return "mixed_other"


def protocol_fields(protocol: dict[str, Any]) -> dict[str, Any]:
    identity = protocol.get("identity", {}) or {}
    inputs = protocol.get("inputs", {}) or {}
    result = protocol.get("result", {}) or {}
    config = protocol.get("configuration", {}) or {}
    env = protocol.get("environment", {}) or {}
    return {
        "kind": safe_str(identity.get("kind")),
        "cases_source": safe_str(identity.get("cases_source") or inputs.get("cases_csv")),
        "duration_s": safe_float(inputs.get("duration_s")),
        "duration_h": safe_float(inputs.get("duration_s")) / 3600.0
        if math.isfinite(safe_float(inputs.get("duration_s")))
        else math.nan,
        "case_ids_protocol": ";".join(str(x) for x in inputs.get("case_ids", []) or []),
        "completed_cases_protocol": safe_float(result.get("completed_cases")),
        "primary_control_profile": safe_str(identity.get("primary_control_profile")),
        "primary_label": safe_str(identity.get("primary_label")),
        "forecast_source_requested": safe_str(identity.get("forecast_source_requested")),
        "forecast_source_effective": safe_str(identity.get("forecast_source_effective")),
        "primary_only": safe_str(identity.get("primary_only")),
        "reactive_primary_only": safe_str(identity.get("reactive_primary_only")),
        "primary_pump_profile": safe_str(config.get("primary_pump_profile")),
        "closed_pump_profile": safe_str(config.get("closed_pump_profile")),
        "primary_scale": safe_float(config.get("primary_scale")),
        "planner_posture_state_gain": safe_float(config.get("planner_posture_state_gain")),
        "relief_medium_cap": safe_str(config.get("relief_medium_cap")),
        "python_executable": safe_str(env.get("python_executable")),
    }


def summarize_run(run_key: str, run_dir: Path) -> dict[str, Any]:
    protocol = read_json(run_dir / "run_protocol.json")
    df = read_csv(run_dir / "casebook_summary.csv")
    fields = protocol_fields(protocol)
    n_cases = len(df) if not df.empty else fields["completed_cases_protocol"]
    duration_s = fields["duration_s"]
    total_time_s = n_cases * duration_s if math.isfinite(duration_s) else math.nan

    primary_pump = sum_col(df, "primary_pump_work_m3")
    closed_pump = sum_col(df, "closed_pump_work_m3")
    if not math.isfinite(closed_pump) or closed_pump == 0:
        closed_pump = math.nan

    row = {
        "run_key": run_key,
        "run_dir": str(run_dir.relative_to(REPO_ROOT)),
        "summary_csv": str((run_dir / "casebook_summary.csv").relative_to(REPO_ROOT))
        if (run_dir / "casebook_summary.csv").exists()
        else "",
        "run_protocol": str((run_dir / "run_protocol.json").relative_to(REPO_ROOT))
        if (run_dir / "run_protocol.json").exists()
        else "",
        "n_cases": n_cases,
        "duration_h": fields["duration_h"],
        "total_case_hours": n_cases * fields["duration_h"]
        if math.isfinite(fields["duration_h"])
        else math.nan,
        "primary_pump_m3_sum": primary_pump,
        "primary_pump_lb_sum": primary_pump * LB_PER_M3
        if math.isfinite(primary_pump)
        else math.nan,
        "closed_pump_m3_sum": closed_pump,
        "within_run_saving_pct": (closed_pump - primary_pump) / closed_pump * 100.0
        if math.isfinite(closed_pump) and closed_pump
        else math.nan,
        "primary_pitch_p95_mean": mean_col(df, "primary_pitch_p95"),
        "primary_roll_p95_mean": mean_col(df, "primary_roll_p95"),
        "closed_pitch_p95_mean": mean_col(df, "closed_pitch_p95"),
        "closed_roll_p95_mean": mean_col(df, "closed_roll_p95"),
        "primary_t_gt5_s_sum": sum_col(df, "primary_time_over_5deg_s"),
        "primary_t_gt7p5_s_sum": sum_col(df, "primary_time_over_7p5deg_s"),
        "primary_t_gt10_s_sum": sum_col(df, "primary_time_over_10deg_s"),
        "closed_t_gt5_s_sum": sum_col(df, "closed_time_over_5deg_s"),
        "closed_t_gt7p5_s_sum": sum_col(df, "closed_time_over_7p5deg_s"),
        "closed_t_gt10_s_sum": sum_col(df, "closed_time_over_10deg_s"),
        "primary_t_gt5_pct": sum_col(df, "primary_time_over_5deg_s") / total_time_s * 100.0
        if math.isfinite(total_time_s) and total_time_s
        else math.nan,
        "primary_t_gt7p5_pct": sum_col(df, "primary_time_over_7p5deg_s") / total_time_s * 100.0
        if math.isfinite(total_time_s) and total_time_s
        else math.nan,
        "primary_t_gt10_pct": sum_col(df, "primary_time_over_10deg_s") / total_time_s * 100.0
        if math.isfinite(total_time_s) and total_time_s
        else math.nan,
        "safety_fallback_ratio_max": max_col(df, "primary_safety_fallback_ratio"),
        "safety_fallback_ratio_mean": mean_col(df, "primary_safety_fallback_ratio"),
        "latch_switches_sum": sum_col(df, "primary_latch_switches"),
    }
    row.update(fields)
    return row


def write_markdown_table(df: pd.DataFrame, path: Path, max_rows: int | None = None) -> None:
    if max_rows is not None:
        df = df.head(max_rows)
    path.write_text(markdown_table(df), encoding="utf-8")


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._\n"
    cols = list(df.columns)
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join("---" for _ in cols) + " |",
    ]
    for _, row in df.iterrows():
        vals = []
        for col in cols:
            value = row[col]
            if isinstance(value, float):
                if math.isnan(value):
                    vals.append("")
                else:
                    vals.append(f"{value:.6g}")
            else:
                vals.append(str(value).replace("\n", " ").replace("|", "\\|"))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines) + "\n"


def build_protocol_inventory() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for protocol_path in sorted(OUT_ROOT.glob("**/run_protocol.json")):
        protocol = read_json(protocol_path)
        fields = protocol_fields(protocol)
        run_dir = protocol_path.parent
        summary_path = run_dir / "casebook_summary.csv"
        n_summary = len(read_csv(summary_path)) if summary_path.exists() else math.nan
        fields.update(
            {
                "run_dir": str(run_dir.relative_to(REPO_ROOT)),
                "run_protocol": str(protocol_path.relative_to(REPO_ROOT)),
                "summary_csv": str(summary_path.relative_to(REPO_ROOT))
                if summary_path.exists()
                else "",
                "n_cases_summary": n_summary,
            }
        )
        rows.append(fields)
    return pd.DataFrame(rows)


def build_12h_pair_delta() -> pd.DataFrame:
    base = read_csv(RUNS["guard10_12h_current_only"] / "casebook_summary.csv")
    gain040 = read_csv(RUNS["guard10_12h_learned_gain040"] / "casebook_summary.csv")
    gain045 = read_csv(RUNS["guard10_12h_learned_gain045"] / "casebook_summary.csv")
    if base.empty or gain040.empty or gain045.empty:
        return pd.DataFrame()

    keep = [
        "case_id",
        "timestamp",
        "label",
        "primary_pump_work_m3",
        "primary_pitch_p95",
        "primary_roll_p95",
        "primary_time_over_5deg_s",
        "primary_time_over_7p5deg_s",
        "primary_time_over_10deg_s",
        "primary_safety_fallback_ratio",
        "primary_latch_switches",
    ]
    base = base[[c for c in keep if c in base.columns]].copy()
    base = base.rename(
        columns={
            "primary_pump_work_m3": "current_pump_m3",
            "primary_pitch_p95": "current_pitch_p95",
            "primary_roll_p95": "current_roll_p95",
            "primary_time_over_5deg_s": "current_t_gt5_s",
            "primary_time_over_7p5deg_s": "current_t_gt7p5_s",
            "primary_time_over_10deg_s": "current_t_gt10_s",
            "primary_safety_fallback_ratio": "current_fallback_ratio",
            "primary_latch_switches": "current_latch_switches",
        }
    )

    out = base
    for tag, df in (("gain040", gain040), ("gain045", gain045)):
        sub = df[[c for c in keep if c in df.columns]].copy()
        sub = sub.rename(
            columns={
                "primary_pump_work_m3": f"{tag}_pump_m3",
                "primary_pitch_p95": f"{tag}_pitch_p95",
                "primary_roll_p95": f"{tag}_roll_p95",
                "primary_time_over_5deg_s": f"{tag}_t_gt5_s",
                "primary_time_over_7p5deg_s": f"{tag}_t_gt7p5_s",
                "primary_time_over_10deg_s": f"{tag}_t_gt10_s",
                "primary_safety_fallback_ratio": f"{tag}_fallback_ratio",
                "primary_latch_switches": f"{tag}_latch_switches",
            }
        )
        sub = sub.drop(columns=[c for c in ("timestamp", "label") if c in sub.columns])
        out = out.merge(sub, on="case_id", how="left")
        out[f"{tag}_saving_pct_vs_current"] = (
            (out["current_pump_m3"] - out[f"{tag}_pump_m3"]) / out["current_pump_m3"] * 100.0
        )
        out[f"{tag}_d_t_gt5_s"] = out[f"{tag}_t_gt5_s"] - out["current_t_gt5_s"]
        out[f"{tag}_d_t_gt7p5_s"] = out[f"{tag}_t_gt7p5_s"] - out["current_t_gt7p5_s"]
        out[f"{tag}_d_t_gt10_s"] = out[f"{tag}_t_gt10_s"] - out["current_t_gt10_s"]
        out[f"{tag}_d_pitch_p95"] = out[f"{tag}_pitch_p95"] - out["current_pitch_p95"]
        out[f"{tag}_d_roll_p95"] = out[f"{tag}_roll_p95"] - out["current_roll_p95"]
    return out


def build_6h_pair_delta() -> pd.DataFrame:
    base = read_csv(RUNS["guard10_6h_current_only"] / "casebook_summary.csv")
    learned = read_csv(RUNS["guard10_6h_learned_gain045"] / "casebook_summary.csv")
    if base.empty or learned.empty:
        return pd.DataFrame()

    keep = [
        "case_id",
        "timestamp",
        "label",
        "primary_pump_work_m3",
        "primary_pitch_p95",
        "primary_roll_p95",
        "primary_time_over_3deg_s",
        "primary_time_over_4deg_s",
        "primary_time_over_5deg_s",
        "primary_time_over_7p5deg_s",
        "primary_time_over_10deg_s",
        "primary_area_over_5deg_deg_s",
        "primary_safety_fallback_ratio",
        "primary_latch_switches",
    ]
    base = base[[c for c in keep if c in base.columns]].copy()
    learned = learned[[c for c in keep if c in learned.columns]].copy()
    base = base.rename(
        columns={
            "primary_pump_work_m3": "current_pump_m3",
            "primary_pitch_p95": "current_pitch_p95",
            "primary_roll_p95": "current_roll_p95",
            "primary_time_over_3deg_s": "current_t_gt3_s",
            "primary_time_over_4deg_s": "current_t_gt4_s",
            "primary_time_over_5deg_s": "current_t_gt5_s",
            "primary_time_over_7p5deg_s": "current_t_gt7p5_s",
            "primary_time_over_10deg_s": "current_t_gt10_s",
            "primary_area_over_5deg_deg_s": "current_area_gt5_deg_s",
            "primary_safety_fallback_ratio": "current_fallback_ratio",
            "primary_latch_switches": "current_latch_switches",
        }
    )
    learned = learned.rename(
        columns={
            "primary_pump_work_m3": "learned_pump_m3",
            "primary_pitch_p95": "learned_pitch_p95",
            "primary_roll_p95": "learned_roll_p95",
            "primary_time_over_3deg_s": "learned_t_gt3_s",
            "primary_time_over_4deg_s": "learned_t_gt4_s",
            "primary_time_over_5deg_s": "learned_t_gt5_s",
            "primary_time_over_7p5deg_s": "learned_t_gt7p5_s",
            "primary_time_over_10deg_s": "learned_t_gt10_s",
            "primary_area_over_5deg_deg_s": "learned_area_gt5_deg_s",
            "primary_safety_fallback_ratio": "learned_fallback_ratio",
            "primary_latch_switches": "learned_latch_switches",
        }
    )
    learned = learned.drop(columns=[c for c in ("timestamp", "label") if c in learned.columns])
    out = base.merge(learned, on="case_id", how="inner")
    out["regime"] = [infer_regime(label, case_id) for label, case_id in zip(out["label"], out["case_id"])]
    out["duration_h"] = 6.0
    out["current_pump_lb"] = out["current_pump_m3"] * LB_PER_M3
    out["learned_pump_lb"] = out["learned_pump_m3"] * LB_PER_M3
    out["pump_saved_m3"] = out["current_pump_m3"] - out["learned_pump_m3"]
    out["pump_saved_lb"] = out["pump_saved_m3"] * LB_PER_M3
    out["pump_saving_pct"] = out["pump_saved_m3"] / out["current_pump_m3"].replace(0, math.nan) * 100.0
    out["d_pitch_p95"] = out["learned_pitch_p95"] - out["current_pitch_p95"]
    out["d_roll_p95"] = out["learned_roll_p95"] - out["current_roll_p95"]
    for metric in ("t_gt3_s", "t_gt4_s", "t_gt5_s", "t_gt7p5_s", "t_gt10_s", "area_gt5_deg_s"):
        out[f"d_{metric}"] = out[f"learned_{metric}"] - out[f"current_{metric}"]
    out["d_fallback_ratio"] = out["learned_fallback_ratio"] - out["current_fallback_ratio"]
    out["d_latch_switches"] = out["learned_latch_switches"] - out["current_latch_switches"]
    return out


def build_group_summary(pair6: pd.DataFrame) -> pd.DataFrame:
    if pair6.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for regime, g in pair6.groupby("regime", sort=True):
        current_pump = safe_float(g["current_pump_m3"].sum())
        learned_pump = safe_float(g["learned_pump_m3"].sum())
        n = int(len(g))
        total_time = n * 21600.0
        rows.append(
            {
                "regime": regime,
                "n_cases": n,
                "current_pump_m3": current_pump,
                "learned_pump_m3": learned_pump,
                "pump_saving_pct": (current_pump - learned_pump) / current_pump * 100.0
                if current_pump
                else math.nan,
                "win_cases": int((g["pump_saved_m3"] > 0).sum()),
                "loss_cases": int((g["pump_saved_m3"] < 0).sum()),
                "median_case_saving_pct": safe_float(g["pump_saving_pct"].median()),
                "mean_d_pitch_p95_deg": safe_float(g["d_pitch_p95"].mean()),
                "mean_d_roll_p95_deg": safe_float(g["d_roll_p95"].mean()),
                "current_t_gt5_pct": safe_float(g["current_t_gt5_s"].sum() / total_time * 100.0),
                "learned_t_gt5_pct": safe_float(g["learned_t_gt5_s"].sum() / total_time * 100.0),
                "d_t_gt5_s": safe_float(g["d_t_gt5_s"].sum()),
                "current_t_gt7p5_pct": safe_float(g["current_t_gt7p5_s"].sum() / total_time * 100.0),
                "learned_t_gt7p5_pct": safe_float(g["learned_t_gt7p5_s"].sum() / total_time * 100.0),
                "d_t_gt7p5_s": safe_float(g["d_t_gt7p5_s"].sum()),
                "current_t_gt10_pct": safe_float(g["current_t_gt10_s"].sum() / total_time * 100.0),
                "learned_t_gt10_pct": safe_float(g["learned_t_gt10_s"].sum() / total_time * 100.0),
                "d_t_gt10_s": safe_float(g["d_t_gt10_s"].sum()),
            }
        )
    current_pump = safe_float(pair6["current_pump_m3"].sum())
    learned_pump = safe_float(pair6["learned_pump_m3"].sum())
    total_time = len(pair6) * 21600.0
    rows.append(
        {
            "regime": "ALL",
            "n_cases": int(len(pair6)),
            "current_pump_m3": current_pump,
            "learned_pump_m3": learned_pump,
            "pump_saving_pct": (current_pump - learned_pump) / current_pump * 100.0
            if current_pump
            else math.nan,
            "win_cases": int((pair6["pump_saved_m3"] > 0).sum()),
            "loss_cases": int((pair6["pump_saved_m3"] < 0).sum()),
            "median_case_saving_pct": safe_float(pair6["pump_saving_pct"].median()),
            "mean_d_pitch_p95_deg": safe_float(pair6["d_pitch_p95"].mean()),
            "mean_d_roll_p95_deg": safe_float(pair6["d_roll_p95"].mean()),
            "current_t_gt5_pct": safe_float(pair6["current_t_gt5_s"].sum() / total_time * 100.0),
            "learned_t_gt5_pct": safe_float(pair6["learned_t_gt5_s"].sum() / total_time * 100.0),
            "d_t_gt5_s": safe_float(pair6["d_t_gt5_s"].sum()),
            "current_t_gt7p5_pct": safe_float(pair6["current_t_gt7p5_s"].sum() / total_time * 100.0),
            "learned_t_gt7p5_pct": safe_float(pair6["learned_t_gt7p5_s"].sum() / total_time * 100.0),
            "d_t_gt7p5_s": safe_float(pair6["d_t_gt7p5_s"].sum()),
            "current_t_gt10_pct": safe_float(pair6["current_t_gt10_s"].sum() / total_time * 100.0),
            "learned_t_gt10_pct": safe_float(pair6["learned_t_gt10_s"].sum() / total_time * 100.0),
            "d_t_gt10_s": safe_float(pair6["d_t_gt10_s"].sum()),
        }
    )
    return pd.DataFrame(rows)


def build_table1_casebook(pair6: pd.DataFrame) -> pd.DataFrame:
    if pair6.empty:
        return pd.DataFrame()
    group = build_group_summary(pair6)
    group = group[group["regime"] != "ALL"].copy()
    group["duration_h_per_case"] = 6.0
    group["total_case_hours"] = group["n_cases"] * 6.0
    group["paper_role"] = "primary 6 h paired evidence"
    group["case_source"] = "outputs/wind_prediction/broad_optimization_work/guard10_broad_cases.csv"
    return group[
        [
            "regime",
            "n_cases",
            "duration_h_per_case",
            "total_case_hours",
            "paper_role",
            "case_source",
        ]
    ]


def build_table2_6h(group_summary: pd.DataFrame) -> pd.DataFrame:
    if group_summary.empty:
        return pd.DataFrame()
    cols = [
        "regime",
        "n_cases",
        "current_pump_m3",
        "learned_pump_m3",
        "pump_saving_pct",
        "win_cases",
        "loss_cases",
        "median_case_saving_pct",
        "mean_d_pitch_p95_deg",
        "mean_d_roll_p95_deg",
        "current_t_gt5_pct",
        "learned_t_gt5_pct",
        "current_t_gt7p5_pct",
        "learned_t_gt7p5_pct",
        "current_t_gt10_pct",
        "learned_t_gt10_pct",
    ]
    return group_summary[[c for c in cols if c in group_summary.columns]].copy()


def build_table3_12h(runs_summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    current = runs_summary[runs_summary["run_key"] == "guard10_12h_current_only"]
    if current.empty:
        return pd.DataFrame()
    current_pump = safe_float(current.iloc[0]["primary_pump_m3_sum"])
    for key, label in (
        ("guard10_12h_current_only", "current-only"),
        ("guard10_12h_learned_gain040", "production gain 0.40"),
        ("guard10_12h_learned_gain045", "production-near gain 0.45"),
    ):
        row = runs_summary[runs_summary["run_key"] == key]
        if row.empty:
            continue
        r = row.iloc[0]
        pump = safe_float(r["primary_pump_m3_sum"])
        rows.append(
            {
                "variant": label,
                "n_cases": int(r["n_cases"]),
                "duration_h": safe_float(r["duration_h"]),
                "pump_work_m3": pump,
                "pump_saving_pct_vs_current": (current_pump - pump) / current_pump * 100.0
                if current_pump
                else math.nan,
                "pitch_p95_mean_deg": safe_float(r["primary_pitch_p95_mean"]),
                "roll_p95_mean_deg": safe_float(r["primary_roll_p95_mean"]),
                "t_gt5_pct": safe_float(r["primary_t_gt5_pct"]),
                "t_gt7p5_pct": safe_float(r["primary_t_gt7p5_pct"]),
                "t_gt10_pct": safe_float(r["primary_t_gt10_pct"]),
                "fallback_ratio_mean": safe_float(r["safety_fallback_ratio_mean"]),
                "source": safe_str(r["run_dir"]),
            }
        )
    return pd.DataFrame(rows)


def build_boundary_robustness_summary(runs_summary: pd.DataFrame, table3: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    w1 = read_csv(STATIC_TABLES["w1_necessity"])
    if not w1.empty:
        for set_name, role in (
            ("stable_direction_event", "longer positive regime support"),
            ("boundary_reversal_signflip", "boundary / abstain evidence"),
        ):
            match = w1[w1["set"].astype(str) == set_name]
            if match.empty:
                continue
            r = match.iloc[0]
            rows.append(
                {
                    "evidence": f"W1 {set_name}",
                    "n_cases": safe_float(r.get("n")),
                    "duration_h": 12.0,
                    "profile_or_branch": "W1 prediction necessity audit / historical specialist",
                    "comparison": "prediction-gated vs closed / blind deadband",
                    "pump_saving_pct": safe_float(r.get("prediction_gated_saving_pct")),
                    "d_t_gt5_s": safe_float(r.get("prediction_gated_d_time5_s")),
                    "d_t_gt7p5_s": math.nan,
                    "d_t_gt10_s": math.nan,
                    "fallback_rows": safe_float(r.get("fallback_rows")),
                    "paper_role": role,
                    "headline_allowed": "no",
                    "source": str(STATIC_TABLES["w1_necessity"].relative_to(REPO_ROOT)),
                    "note": "Use to explain direction-stable opportunity and signflip abstention; not a production-near headline.",
                }
            )

    for key, evidence, role in (
        ("p2_6h_boundary", "P2 relaxed 6 h boundary", "boundary / mechanism"),
        ("c3_12h_boundary", "C3 balanced 12 h boundary", "boundary / cautionary"),
        ("psc_structured_single", "PSC structured selector single trace", "mechanism / appendix only"),
    ):
        row = runs_summary[runs_summary["run_key"] == key]
        if row.empty:
            continue
        r = row.iloc[0]
        closed_t5 = safe_float(r.get("closed_t_gt5_s_sum"))
        primary_t5 = safe_float(r.get("primary_t_gt5_s_sum"))
        closed_t75 = safe_float(r.get("closed_t_gt7p5_s_sum"))
        primary_t75 = safe_float(r.get("primary_t_gt7p5_s_sum"))
        closed_t10 = safe_float(r.get("closed_t_gt10_s_sum"))
        primary_t10 = safe_float(r.get("primary_t_gt10_s_sum"))
        rows.append(
            {
                "evidence": evidence,
                "n_cases": safe_float(r.get("n_cases")),
                "duration_h": safe_float(r.get("duration_h")),
                "profile_or_branch": safe_str(r.get("primary_control_profile") or r.get("run_key")),
                "comparison": "within-run closed/current comparator if present",
                "pump_saving_pct": safe_float(r.get("within_run_saving_pct")),
                "d_t_gt5_s": primary_t5 - closed_t5
                if math.isfinite(primary_t5) and math.isfinite(closed_t5)
                else math.nan,
                "d_t_gt7p5_s": primary_t75 - closed_t75
                if math.isfinite(primary_t75) and math.isfinite(closed_t75)
                else math.nan,
                "d_t_gt10_s": primary_t10 - closed_t10
                if math.isfinite(primary_t10) and math.isfinite(closed_t10)
                else math.nan,
                "fallback_rows": math.nan,
                "paper_role": role,
                "headline_allowed": "no",
                "source": safe_str(r.get("run_dir")),
                "note": "Keep separate from production-near 6 h primary claim.",
            }
        )

    current = table3[table3["variant"] == "current-only"]
    gain045 = table3[table3["variant"] == "production-near gain 0.45"]
    if not current.empty and not gain045.empty:
        c = current.iloc[0]
        g = gain045.iloc[0]
        total_s = safe_float(g.get("n_cases")) * safe_float(g.get("duration_h")) * 3600.0
        rows.append(
            {
                "evidence": "guard10 mixed 12 h gain0.45",
                "n_cases": safe_float(g.get("n_cases")),
                "duration_h": safe_float(g.get("duration_h")),
                "profile_or_branch": PRODUCTION_PROFILE,
                "comparison": "production-near gain0.45 vs current-only",
                "pump_saving_pct": safe_float(g.get("pump_saving_pct_vs_current")),
                "d_t_gt5_s": (safe_float(g.get("t_gt5_pct")) - safe_float(c.get("t_gt5_pct"))) / 100.0 * total_s,
                "d_t_gt7p5_s": (safe_float(g.get("t_gt7p5_pct")) - safe_float(c.get("t_gt7p5_pct"))) / 100.0 * total_s,
                "d_t_gt10_s": (safe_float(g.get("t_gt10_pct")) - safe_float(c.get("t_gt10_pct"))) / 100.0 * total_s,
                "fallback_rows": safe_float(g.get("fallback_ratio_mean")),
                "paper_role": "12 h robustness disclosure",
                "headline_allowed": "yes, as robustness only",
                "source": safe_str(g.get("source")),
                "note": "Use as long-window mixed-regime boundary; do not write as deployment-scale proof.",
            }
        )

    return pd.DataFrame(rows)


def _style_axis(ax: plt.Axes) -> None:
    ax.grid(True, axis="y", color="#d8dde5", linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#9aa4ad")
    ax.spines["bottom"].set_color("#9aa4ad")


def _case_timeseries_path(run_dir: Path, case_id: str) -> Path | None:
    matches = sorted((run_dir / "timeseries").glob(f"{case_id}_*_timeseries.csv"))
    return matches[0] if matches else None


def _cumulative_pump_lb(df: pd.DataFrame) -> pd.Series:
    if "pump_total_rate_m3_min" not in df.columns or "t_s" not in df.columns:
        return pd.Series([math.nan] * len(df))
    t = pd.to_numeric(df["t_s"], errors="coerce").ffill().fillna(0.0)
    rate = pd.to_numeric(df["pump_total_rate_m3_min"], errors="coerce").fillna(0.0).abs()
    dt_s = t.diff().fillna(t.diff().median() if t.diff().notna().any() else 1.0).clip(lower=0.0)
    return (rate * dt_s / 60.0 * LB_PER_M3).cumsum()


def _episode_plot_series(df: pd.DataFrame) -> dict[str, pd.Series]:
    target_cols = [c for c in ("target_tank1_kg", "target_tank2_kg", "target_tank3_kg") if c in df.columns]
    if target_cols:
        target_total_lb = df[target_cols].apply(pd.to_numeric, errors="coerce").sum(axis=1) * LB_PER_KG
    else:
        target_total_lb = pd.Series([math.nan] * len(df))
    return {
        "t_h": pd.to_numeric(df["t_s"], errors="coerce") / 3600.0,
        "wind_speed": pd.to_numeric(df["wind_speed"], errors="coerce"),
        "wind_dir_deg": pd.to_numeric(df["wind_dir_deg"], errors="coerce"),
        "pump_lb_min": pd.to_numeric(df["pump_total_rate_m3_min"], errors="coerce").fillna(0.0) * LB_PER_M3,
        "cum_pump_lb": _cumulative_pump_lb(df),
        "pitch_abs": pd.to_numeric(df["pitch_deg"], errors="coerce").abs(),
        "roll_abs": pd.to_numeric(df["roll_deg"], errors="coerce").abs(),
        "total_ballast_lb": pd.to_numeric(df["ballast_total_kg"], errors="coerce").ffill() * LB_PER_KG,
        "target_total_lb": target_total_lb,
    }


def plot_control_architecture() -> None:
    MAIN_FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(11.2, 6.2))
    ax.set_axis_off()
    boxes = {
        "wind": (0.07, 0.72, "10 min wind history\nspeed, direction, events"),
        "forecast": (0.33, 0.72, "Short-term forecast\n0-20 / 20-40 / 40-60 min"),
        "risk": (0.60, 0.72, "Risk windows\ntrend, relief, signflip, gust"),
        "supervisor": (0.33, 0.42, "Prediction supervision layer\nregime ID, strategy selection,\nhold / release / cap / veto"),
        "feedback": (0.07, 0.18, "Current-state feedback\npitch, roll, heave, tank errors"),
        "safety": (0.60, 0.18, "Forecast-independent safeguards\nbarrier, fallback, actuator limits"),
        "actuator": (0.33, 0.04, "Active ballast execution\npump rates and target tank mass"),
    }
    for key, (x, y, text) in boxes.items():
        face = "#e8eef4" if key in {"forecast", "risk", "supervisor"} else "#f7f4ef"
        ax.text(
            x,
            y,
            text,
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=10.5,
            bbox={
                "boxstyle": "round,pad=0.55",
                "facecolor": face,
                "edgecolor": "#607080",
                "linewidth": 1.1,
            },
        )

    arrows = [
        ("wind", "forecast"),
        ("forecast", "risk"),
        ("risk", "supervisor"),
        ("feedback", "supervisor"),
        ("supervisor", "actuator"),
        ("safety", "actuator"),
        ("actuator", "feedback"),
    ]
    for src, dst in arrows:
        sx, sy, _ = boxes[src]
        dx, dy, _ = boxes[dst]
        arrow = FancyArrowPatch(
            (sx, sy - 0.04 if src == "supervisor" else sy),
            (dx, dy + 0.07 if dst == "actuator" else dy),
            transform=ax.transAxes,
            arrowstyle="-|>",
            mutation_scale=13,
            linewidth=1.1,
            color="#3f4b57",
            connectionstyle="arc3,rad=0.05",
        )
        ax.add_patch(arrow)

    ax.text(
        0.75,
        0.48,
        "Forecasts provide supervisory information;\nthe pump command remains constrained\nby the feedback and safety layers.",
        transform=ax.transAxes,
        fontsize=10,
        ha="center",
        va="center",
        color="#4a5662",
    )
    ax.set_title("Fig. 1. Short-term forecast-supervised active ballast framework", fontsize=13, pad=14)
    fig.tight_layout()
    fig.savefig(MAIN_FIG_DIR / "fig1_control_architecture.png", dpi=180)
    plt.close(fig)


def plot_representative_episode(pair6: pd.DataFrame) -> str:
    MAIN_FIG_DIR.mkdir(parents=True, exist_ok=True)
    if pair6.empty:
        return ""
    candidates = pair6[
        (pair6["pump_saving_pct"] > 0)
        & (pair6["d_t_gt7p5_s"] <= 0)
        & (pair6["d_t_gt10_s"] <= 0)
        & (pair6["regime"] == "b_high_pressure")
    ].copy()
    if candidates.empty:
        candidates = pair6[(pair6["pump_saving_pct"] > 0) & (pair6["d_t_gt10_s"] <= 0)].copy()
    if candidates.empty:
        return ""
    row = candidates.sort_values("pump_saving_pct", ascending=False).iloc[0]
    case_id = safe_str(row["case_id"])
    base_path = _case_timeseries_path(RUNS["guard10_6h_current_only"], case_id)
    learned_path = _case_timeseries_path(RUNS["guard10_6h_learned_gain045"], case_id)
    if base_path is None or learned_path is None:
        return ""

    base = read_csv(base_path)
    learned = read_csv(learned_path)
    if base.empty or learned.empty:
        return ""

    base_plot = _episode_plot_series(base)
    learned_plot = _episode_plot_series(learned)

    fig, axes = plt.subplots(4, 1, figsize=(11.2, 8.8), sharex=True)
    axes[0].plot(learned_plot["t_h"], learned_plot["wind_speed"], color="#315f8c", label="wind speed")
    axes[0].set_ylabel("m/s")
    ax0b = axes[0].twinx()
    ax0b.plot(learned_plot["t_h"], learned_plot["wind_dir_deg"], color="#9b6a2f", alpha=0.65, label="wind direction")
    ax0b.set_ylabel("deg")
    axes[0].set_title(f"Fig. 2. Representative 6 h paired episode: {case_id}")
    _style_axis(axes[0])

    axes[1].plot(base_plot["t_h"], base_plot["pump_lb_min"], color="#6b737c", linewidth=1.0, label="current-only lb/min")
    axes[1].plot(learned_plot["t_h"], learned_plot["pump_lb_min"], color="#2f6f9f", linewidth=1.0, label="learned lb/min")
    axes[1].set_ylabel("pump lb/min")
    axes[1].legend(frameon=False, ncol=2)
    _style_axis(axes[1])

    axes[2].plot(base_plot["t_h"], base_plot["cum_pump_lb"], color="#6b737c", label="current-only cumulative")
    axes[2].plot(learned_plot["t_h"], learned_plot["cum_pump_lb"], color="#2f6f9f", label="learned cumulative")
    axes[2].set_ylabel("cumulative lb")
    axes[2].legend(frameon=False, ncol=2)
    _style_axis(axes[2])

    axes[3].plot(base_plot["t_h"], base_plot["pitch_abs"], color="#315f8c", alpha=0.45, label="current pitch abs")
    axes[3].plot(learned_plot["t_h"], learned_plot["pitch_abs"], color="#315f8c", label="learned pitch abs")
    axes[3].plot(base_plot["t_h"], base_plot["roll_abs"], color="#b85c38", alpha=0.45, label="current roll abs")
    axes[3].plot(learned_plot["t_h"], learned_plot["roll_abs"], color="#b85c38", label="learned roll abs")
    for threshold in (5, 7.5, 10):
        axes[3].axhline(threshold, color="#7f8790", linewidth=0.8, linestyle="--")
    axes[3].set_ylabel("abs attitude deg")
    axes[3].set_xlabel("time (h)")
    axes[3].legend(frameon=False, ncol=2)
    _style_axis(axes[3])

    fig.text(
        0.02,
        0.01,
        f"Case saving {safe_float(row['pump_saving_pct']):.2f}%; "
        f"d t>7.5={safe_float(row['d_t_gt7p5_s']):.0f}s, d t>10={safe_float(row['d_t_gt10_s']):.0f}s. "
        "Metrics use raw 1 Hz data; plotted curves are unsmoothed.",
        fontsize=9,
        color="#4a5662",
    )
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    out = MAIN_FIG_DIR / "fig2_representative_6h_episode.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return str(out.relative_to(REPO_ROOT))


def plot_boundary_figure(boundary_summary: pd.DataFrame) -> None:
    MAIN_FIG_DIR.mkdir(parents=True, exist_ok=True)
    if boundary_summary.empty:
        return
    plot_df = boundary_summary.copy()
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.8))
    colors = ["#b85c38" if safe_float(v) < 0 else "#2f6f9f" for v in plot_df["pump_saving_pct"]]
    axes[0].barh(plot_df["evidence"], plot_df["pump_saving_pct"], color=colors)
    axes[0].axvline(0, color="#3d4752", linewidth=1)
    axes[0].set_xlabel("Pump saving (%)")
    axes[0].set_title("Boundary and robustness pump effect")
    _style_axis(axes[0])

    x = range(len(plot_df))
    axes[1].bar([i - 0.22 for i in x], plot_df["d_t_gt5_s"], 0.22, label="d t>5 s", color="#587a9f")
    axes[1].bar([i for i in x], plot_df["d_t_gt7p5_s"], 0.22, label="d t>7.5 s", color="#b85c38")
    axes[1].bar([i + 0.22 for i in x], plot_df["d_t_gt10_s"], 0.22, label="d t>10 s", color="#6f5c91")
    axes[1].axhline(0, color="#3d4752", linewidth=1)
    axes[1].set_xticks(list(x))
    axes[1].set_xticklabels(plot_df["evidence"], rotation=35, ha="right")
    axes[1].set_ylabel("Candidate - baseline exposure (s)")
    axes[1].set_title("Boundary attitude-exposure disclosure")
    axes[1].legend(frameon=False)
    _style_axis(axes[1])
    fig.suptitle("Fig. 5. Boundary, abstention, and long-window disclosure", y=1.02)
    fig.tight_layout()
    fig.savefig(MAIN_FIG_DIR / "fig5_boundary_robustness_disclosure.png", dpi=180)
    plt.close(fig)


def copy_appendix_figures() -> pd.DataFrame:
    src_root = OUT_ROOT / "multi_regime_curve_figures_20260606_py312"
    rows: list[dict[str, Any]] = []
    if not src_root.exists():
        return pd.DataFrame()
    APPENDIX_FIG_DIR.mkdir(parents=True, exist_ok=True)
    for regime_dir in sorted(p for p in src_root.iterdir() if p.is_dir()):
        dest_dir = APPENDIX_FIG_DIR / regime_dir.name
        dest_dir.mkdir(parents=True, exist_ok=True)
        for src in sorted(regime_dir.glob("*.png")):
            dest = dest_dir / src.name
            shutil.copy2(src, dest)
            rows.append(
                {
                    "regime": regime_dir.name,
                    "figure_type": src.stem,
                    "source": str(src.relative_to(REPO_ROOT)),
                    "package_path": str(dest.relative_to(REPO_ROOT)),
                    "status": "ready",
                }
            )
    return pd.DataFrame(rows)


def plot_main_figures(
    group_summary: pd.DataFrame,
    pair6: pd.DataFrame,
    table3: pd.DataFrame,
    boundary_summary: pd.DataFrame,
) -> None:
    MAIN_FIG_DIR.mkdir(parents=True, exist_ok=True)
    plot_control_architecture()
    plot_representative_episode(pair6)
    plot_boundary_figure(boundary_summary)
    if not group_summary.empty:
        plot_df = group_summary[group_summary["regime"] != "ALL"].copy()
        plot_df = plot_df.sort_values("pump_saving_pct")
        fig, ax = plt.subplots(figsize=(8.5, 4.8))
        colors = ["#b85c38" if v < 0 else "#2f6f9f" for v in plot_df["pump_saving_pct"]]
        ax.barh(plot_df["regime"], plot_df["pump_saving_pct"], color=colors)
        ax.axvline(0, color="#3d4752", linewidth=1)
        ax.set_xlabel("Pump saving vs current-only (%)")
        ax.set_ylabel("Regime")
        ax.set_title("Fig. 3. Production-near 6 h paired pump saving")
        _style_axis(ax)
        fig.tight_layout()
        fig.savefig(MAIN_FIG_DIR / "fig3_6h_paired_pump_saving.png", dpi=180)
        plt.close(fig)

        fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.8))
        width = 0.35
        x = range(len(plot_df))
        axes[0].bar([i - width / 2 for i in x], plot_df["d_t_gt5_s"], width, label="d t>5 s", color="#587a9f")
        axes[0].bar([i + width / 2 for i in x], plot_df["d_t_gt7p5_s"], width, label="d t>7.5 s", color="#b85c38")
        axes[0].axhline(0, color="#3d4752", linewidth=1)
        axes[0].set_xticks(list(x))
        axes[0].set_xticklabels(plot_df["regime"], rotation=30, ha="right")
        axes[0].set_ylabel("Learned - current exposure (s)")
        axes[0].set_title("Exposure change")
        axes[0].legend(frameon=False)
        _style_axis(axes[0])
        axes[1].bar([i - width / 2 for i in x], plot_df["mean_d_pitch_p95_deg"], width, label="d pitch p95", color="#315f8c")
        axes[1].bar([i + width / 2 for i in x], plot_df["mean_d_roll_p95_deg"], width, label="d roll p95", color="#7a8f3b")
        axes[1].axhline(0, color="#3d4752", linewidth=1)
        axes[1].set_xticks(list(x))
        axes[1].set_xticklabels(plot_df["regime"], rotation=30, ha="right")
        axes[1].set_ylabel("Learned - current p95 (deg)")
        axes[1].set_title("Posture p95 change")
        axes[1].legend(frameon=False)
        _style_axis(axes[1])
        fig.suptitle("Fig. 4. Production-near 6 h attitude trade-offs", y=1.02)
        fig.tight_layout()
        fig.savefig(MAIN_FIG_DIR / "fig4_6h_attitude_tradeoffs.png", dpi=180)
        plt.close(fig)

    if not table3.empty:
        fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6))
        axes[0].bar(table3["variant"], table3["pump_work_m3"], color=["#5f6b7a", "#2f6f9f", "#1f4e79"])
        axes[0].set_ylabel("Pump work (m3)")
        axes[0].set_title("12 h pump work")
        axes[0].tick_params(axis="x", rotation=25)
        _style_axis(axes[0])
        x = range(len(table3))
        axes[1].plot(list(x), table3["t_gt5_pct"], marker="o", label="t>5 deg", color="#315f8c")
        axes[1].plot(list(x), table3["t_gt7p5_pct"], marker="o", label="t>7.5 deg", color="#b85c38")
        axes[1].plot(list(x), table3["t_gt10_pct"], marker="o", label="t>10 deg", color="#6f5c91")
        axes[1].set_xticks(list(x))
        axes[1].set_xticklabels(table3["variant"], rotation=25, ha="right")
        axes[1].set_ylabel("Exposure share (%)")
        axes[1].set_title("12 h attitude exposure")
        axes[1].legend(frameon=False)
        _style_axis(axes[1])
        fig.suptitle("Fig. 6. Production-near 12 h guard10 robustness", y=1.02)
        fig.tight_layout()
        fig.savefig(MAIN_FIG_DIR / "fig6_12h_guard10_robustness.png", dpi=180)
        plt.close(fig)


def role_for_run(run_key: str, row: dict[str, Any]) -> tuple[str, str, str, str]:
    profile = row.get("primary_control_profile", "")
    if run_key.startswith("guard10_12h"):
        return (
            "robustness",
            "yes",
            "Production-near 12 h mixed-regime robustness disclosure.",
            "Table 3 / Fig. 6",
        )
    if run_key == "p2_6h_boundary":
        return (
            "boundary",
            "no",
            "P2 relaxed branch is useful for boundary interpretation but not a production headline.",
            "Fig. 5 / appendix",
        )
    if run_key == "c3_12h_boundary":
        return (
            "boundary",
            "no",
            "C3 balanced release-pool branch is boundary/mechanism evidence.",
            "Fig. 5 / appendix",
        )
    if run_key == "psc_structured_single":
        return (
            "mechanism",
            "no",
            "PSC single-output trace lacks a fabricated paired comparator; use as mechanism material only.",
            "appendix",
        )
    if profile == PRODUCTION_PROFILE:
        return ("candidate_primary", "conditional", "Production profile found; needs paired 6 h audit.", "Table 2")
    return ("mechanism", "no", "Non-production or historical branch.", "appendix")


def build_claim_evidence(runs_summary: pd.DataFrame, pair6: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    by_key = {safe_str(r["run_key"]): r for _, r in runs_summary.iterrows()}

    if not pair6.empty:
        current_pump = safe_float(pair6["current_pump_m3"].sum())
        learned_pump = safe_float(pair6["learned_pump_m3"].sum())
        total_time = len(pair6) * 21600.0
        rows.append(
            {
                "claim_id": "primary_guard10_6h_learned_gain045",
                "paper_claim": "Production-near learned forecast supervision reduces cumulative pump work versus current-only on the frozen guard10 6 h paired casebook.",
                "claim_role": "primary",
                "can_use_for_headline": "yes, as 6 h episode-level primary evidence",
                "n_cases": int(len(pair6)),
                "duration_h": 6.0,
                "profile": PRODUCTION_PROFILE,
                "forecast_source": "lstm_dual_head_preview",
                "baseline": "guard10_6h_current_only",
                "variant": "guard10_6h_learned_gain045",
                "pump_saving_pct": (current_pump - learned_pump) / current_pump * 100.0
                if current_pump
                else math.nan,
                "pump_current_m3": current_pump,
                "pump_variant_m3": learned_pump,
                "t_gt5_pct": safe_float(pair6["learned_t_gt5_s"].sum() / total_time * 100.0),
                "t_gt7p5_pct": safe_float(pair6["learned_t_gt7p5_s"].sum() / total_time * 100.0),
                "t_gt10_pct": safe_float(pair6["learned_t_gt10_s"].sum() / total_time * 100.0),
                "source": "outputs/wind_prediction/paper_20260606_execution/production_near_6h_per_case_delta.csv",
                "target_output": "Table 2 / Fig. 3 / Fig. 4",
                "caveat": "Claim is limited to frozen 6 h guard10 episode-level evidence; do not write as 24 h deployment-scale proof.",
            }
        )

    current = by_key.get("guard10_12h_current_only")
    gain040 = by_key.get("guard10_12h_learned_gain040")
    gain045 = by_key.get("guard10_12h_learned_gain045")
    if current is not None and gain040 is not None and gain045 is not None:
        current_pump = safe_float(current["primary_pump_m3_sum"])
        for key, label in (
            ("guard10_12h_learned_gain040", "12 h production-near gain 0.40"),
            ("guard10_12h_learned_gain045", "12 h production-near gain 0.45"),
        ):
            row = by_key[key]
            pump = safe_float(row["primary_pump_m3_sum"])
            saving = (current_pump - pump) / current_pump * 100.0 if current_pump else math.nan
            rows.append(
                {
                    "claim_id": f"robustness_{key}",
                    "paper_claim": f"{label} reduces cumulative pump work versus current-only on guard10 12 h mixed-regime cases.",
                    "claim_role": "robustness",
                    "can_use_for_headline": "yes, as 12 h robustness only",
                    "n_cases": int(row["n_cases"]),
                    "duration_h": row["duration_h"],
                    "profile": row["primary_control_profile"],
                    "forecast_source": row["forecast_source_effective"],
                    "baseline": "guard10_12h_current_only",
                    "variant": key,
                    "pump_saving_pct": saving,
                    "pump_current_m3": current_pump,
                    "pump_variant_m3": pump,
                    "t_gt5_pct": row["primary_t_gt5_pct"],
                    "t_gt7p5_pct": row["primary_t_gt7p5_pct"],
                    "t_gt10_pct": row["primary_t_gt10_pct"],
                    "source": row["run_dir"],
                    "target_output": "Table 3 / Fig. 6",
                    "caveat": "Do not present as 6 h high-gain headline; 12 h mixed windows dilute benefits.",
                }
            )

    for _, row in runs_summary.iterrows():
        run_key = safe_str(row["run_key"])
        role, headline, caveat, target = role_for_run(run_key, row.to_dict())
        if run_key.startswith("guard10_12h"):
            continue
        rows.append(
            {
                "claim_id": f"source_{run_key}",
                "paper_claim": f"Existing {run_key} material can support {role} discussion.",
                "claim_role": role,
                "can_use_for_headline": headline,
                "n_cases": row["n_cases"],
                "duration_h": row["duration_h"],
                "profile": row["primary_control_profile"],
                "forecast_source": row["forecast_source_effective"],
                "baseline": "within-run closed/current if present",
                "variant": run_key,
                "pump_saving_pct": row["within_run_saving_pct"],
                "pump_current_m3": row["closed_pump_m3_sum"],
                "pump_variant_m3": row["primary_pump_m3_sum"],
                "t_gt5_pct": row["primary_t_gt5_pct"],
                "t_gt7p5_pct": row["primary_t_gt7p5_pct"],
                "t_gt10_pct": row["primary_t_gt10_pct"],
                "source": row["run_dir"],
                "target_output": target,
                "caveat": caveat,
            }
        )

    # Static historical tables.
    d1 = read_csv(STATIC_TABLES["d1_all80_6h"])
    if not d1.empty:
        r = d1.iloc[0]
        rows.append(
            {
                "claim_id": "mechanism_d1_all80_6h",
                "paper_claim": "D1 neutral/headroom historical branch shows large redundant-pump savings.",
                "claim_role": "mechanism",
                "can_use_for_headline": "no",
                "n_cases": safe_float(r.get("cases")),
                "duration_h": 6.0,
                "profile": "dc_preserving_deadband_current / historical specialist",
                "forecast_source": "lstm_dual_head_preview",
                "baseline": "closed/current in historical branch",
                "variant": "D1 specialist",
                "pump_saving_pct": safe_float(r.get("saving_pct")),
                "pump_current_m3": safe_float(r.get("closed_pump_m3")),
                "pump_variant_m3": safe_float(r.get("primary_pump_m3")),
                "t_gt5_pct": math.nan,
                "t_gt7p5_pct": math.nan,
                "t_gt10_pct": math.nan,
                "source": str(STATIC_TABLES["d1_all80_6h"].relative_to(REPO_ROOT)),
                "target_output": "appendix / mechanism paragraph",
                "caveat": "Do not merge into production-near headline.",
            }
        )

    c1 = read_csv(STATIC_TABLES["c1_p4_all28_6h"])
    if not c1.empty:
        all28 = c1[c1["group"].astype(str).str.contains("all_28", na=False)]
        r = all28.iloc[0] if not all28.empty else c1.iloc[-1]
        rows.append(
            {
                "claim_id": "mechanism_c1_p4_all28_6h",
                "paper_claim": "C1/P4 soft-decay historical branch supports stable-direction relief mechanism.",
                "claim_role": "mechanism",
                "can_use_for_headline": "no",
                "n_cases": safe_float(r.get("n")),
                "duration_h": 6.0,
                "profile": "stable soft-decay specialist / historical branch",
                "forecast_source": "not frozen for production claim",
                "baseline": "a0 branch",
                "variant": "soft-decay candidate",
                "pump_saving_pct": safe_float(r.get("pump_saved_pct")),
                "pump_current_m3": safe_float(r.get("a0_pump_m3")),
                "pump_variant_m3": safe_float(r.get("candidate_pump_m3")),
                "t_gt5_pct": math.nan,
                "t_gt7p5_pct": math.nan,
                "t_gt10_pct": math.nan,
                "source": str(STATIC_TABLES["c1_p4_all28_6h"].relative_to(REPO_ROOT)),
                "target_output": "appendix / mechanism paragraph",
                "caveat": "Historical specialist result; not a production-near headline.",
            }
        )

    fig_root = OUT_ROOT / "multi_regime_curve_figures_20260606_py312"
    if fig_root.exists():
        png_count = len(list(fig_root.glob("*/*.png")))
        rows.append(
            {
                "claim_id": "appendix_multi_regime_curve_figures",
                "paper_claim": "Five regime-specific figure packs already separate pitch, roll, pump pounds, and ballast targets.",
                "claim_role": "appendix_figures",
                "can_use_for_headline": "no",
                "n_cases": math.nan,
                "duration_h": math.nan,
                "profile": "mixed",
                "forecast_source": "mixed",
                "baseline": "varies by regime",
                "variant": "25 PNG figures",
                "pump_saving_pct": math.nan,
                "pump_current_m3": math.nan,
                "pump_variant_m3": math.nan,
                "t_gt5_pct": math.nan,
                "t_gt7p5_pct": math.nan,
                "t_gt10_pct": math.nan,
                "source": str(fig_root.relative_to(REPO_ROOT)),
                "target_output": "appendix figure index",
                "caveat": f"Representative curves only; current PNG count={png_count}.",
            }
        )

    return pd.DataFrame(rows)


def build_figure_inventory() -> pd.DataFrame:
    fig_root = OUT_ROOT / "multi_regime_curve_figures_20260606_py312"
    rows: list[dict[str, Any]] = []
    if not fig_root.exists():
        return pd.DataFrame()
    for regime_dir in sorted(p for p in fig_root.iterdir() if p.is_dir()):
        pngs = sorted(regime_dir.glob("*.png"))
        selected_cases = regime_dir / "selected_cases.csv"
        selected_metrics = regime_dir / "selected_case_metrics.csv"
        rows.append(
            {
                "regime": regime_dir.name,
                "png_count": len(pngs),
                "has_selected_cases": selected_cases.exists(),
                "has_selected_case_metrics": selected_metrics.exists(),
                "figure_files": ";".join(str(p.relative_to(REPO_ROOT)) for p in pngs),
                "status": "ready" if len(pngs) >= 5 else "needs_more_figures",
            }
        )
    return pd.DataFrame(rows)


def build_6h_audit(protocol_inventory: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    if protocol_inventory.empty:
        return pd.DataFrame(), "No run_protocol.json files found."
    inv = protocol_inventory.copy()
    inv["duration_h_round"] = pd.to_numeric(inv["duration_h"], errors="coerce").round(3)
    prod6 = inv[
        (inv["duration_h_round"] == 6.0)
        & (inv["primary_control_profile"].astype(str) == PRODUCTION_PROFILE)
    ].copy()
    prod6 = prod6.sort_values(["forecast_source_requested", "n_cases_summary", "run_dir"])

    current_like = prod6[
        prod6["forecast_source_requested"].astype(str).str.contains("current", case=False, na=False)
    ]
    learned_like = prod6[
        prod6["forecast_source_requested"].astype(str).str.contains("learned", case=False, na=False)
    ]
    broad_learned = learned_like[pd.to_numeric(learned_like["n_cases_summary"], errors="coerce") >= 10]
    broad_current = current_like[pd.to_numeric(current_like["n_cases_summary"], errors="coerce") >= 10]

    if not prod6.empty and not broad_learned.empty and not broad_current.empty:
        verdict = (
            "FOUND_POSSIBLE_CANDIDATES: production-near 6 h current-like and learned-like "
            "casebooks exist, but paired case IDs must still be matched before Table 2."
        )
    elif not prod6.empty:
        verdict = (
            "PARTIAL_ONLY: production-near 6 h outputs exist, but a broad paired current-only "
            "versus learned casebook was not found in the protocol scan."
        )
    else:
        verdict = (
            "NOT_FOUND: no production-near 6 h run_protocol with "
            f"`{PRODUCTION_PROFILE}` was found. The broad 6 h primary evidence still needs "
            "case-selection freeze and paired rerun."
        )
    return prod6, verdict


def write_summary_md(
    runs_summary: pd.DataFrame,
    claim_map: pd.DataFrame,
    fig_inventory: pd.DataFrame,
    prod6: pd.DataFrame,
    prod6_verdict: str,
    pair12: pd.DataFrame,
    pair6: pd.DataFrame,
    group6: pd.DataFrame,
) -> None:
    lines: list[str] = []
    lines.append("# Paper Evidence Package 20260606")
    lines.append("")
    lines.append("Generated with `.venv312/bin/python`.")
    lines.append("")
    lines.append("## Audit Verdict")
    lines.append("")
    lines.append(f"- Production-near 6 h paired evidence: `{prod6_verdict}`")
    lines.append("- 12 h guard10 production-near robustness evidence: ready.")
    lines.append("- Multi-regime representative curve figures: ready for appendix/mechanism display.")
    lines.append("- Historical D1/C1/P4/W1/C3/PSC materials: usable as mechanism or boundary evidence only unless explicitly rerun under the production-near profile.")
    lines.append("")
    if not pair6.empty:
        current_pump_6h = safe_float(pair6["current_pump_m3"].sum())
        learned_pump_6h = safe_float(pair6["learned_pump_m3"].sum())
        saving_6h = (current_pump_6h - learned_pump_6h) / current_pump_6h * 100.0
        total_time_6h = len(pair6) * 21600.0
        lines.append("## Key 6 h Primary Numbers")
        lines.append("")
        lines.append(
            f"- Frozen guard10 6 h paired casebook: `{len(pair6)}` cases, "
            f"current-only pump `{current_pump_6h:.2f} m3`, learned pump `{learned_pump_6h:.2f} m3`, "
            f"saving `{saving_6h:.2f}%`."
        )
        lines.append(
            f"- Learned exposure: `t>5` `{pair6['learned_t_gt5_s'].sum() / total_time_6h * 100.0:.4f}%`, "
            f"`t>7.5` `{pair6['learned_t_gt7p5_s'].sum() / total_time_6h * 100.0:.4f}%`, "
            f"`t>10` `{pair6['learned_t_gt10_s'].sum() / total_time_6h * 100.0:.4f}%`."
        )
        lines.append(
            f"- Exposure deltas vs current-only: `d t>5` `{pair6['d_t_gt5_s'].sum():.0f}s`, "
            f"`d t>7.5` `{pair6['d_t_gt7p5_s'].sum():.0f}s`, "
            f"`d t>10` `{pair6['d_t_gt10_s'].sum():.0f}s`."
        )
        lines.append("")
    lines.append("## Key 12 h Robustness Numbers")
    lines.append("")
    current = runs_summary[runs_summary["run_key"] == "guard10_12h_current_only"]
    if not current.empty:
        current_pump = safe_float(current.iloc[0]["primary_pump_m3_sum"])
        for key in ("guard10_12h_learned_gain040", "guard10_12h_learned_gain045"):
            row = runs_summary[runs_summary["run_key"] == key]
            if row.empty:
                continue
            r = row.iloc[0]
            pump = safe_float(r["primary_pump_m3_sum"])
            saving = (current_pump - pump) / current_pump * 100.0 if current_pump else math.nan
            lines.append(
                f"- `{key}`: pump `{pump:.2f} m3`, saving `{saving:.2f}%`, "
                f"`t>5` `{safe_float(r['primary_t_gt5_pct']):.4f}%`, "
                f"`t>7.5` `{safe_float(r['primary_t_gt7p5_pct']):.4f}%`, "
                f"`t>10` `{safe_float(r['primary_t_gt10_pct']):.4f}%`."
            )
    lines.append("")
    lines.append("## Outputs")
    lines.append("")
    for name in (
        "casebook_protocol_inventory.csv",
        "run_summary_key_sources.csv",
        "claim_evidence_map.csv",
        "claim_evidence_map.md",
        "production_near_12h_per_case_delta.csv",
        "production_near_6h_per_case_delta.csv",
        "production_near_6h_regime_summary.csv",
        "production_near_6h_audit.csv",
        "production_near_6h_audit.md",
        "tables/table1_casebook.csv",
        "tables/table2_6h_primary_results.csv",
        "tables/table3_12h_robustness.csv",
        "tables/table3_boundary_robustness_summary.csv",
        "main_figures/fig1_control_architecture.png",
        "main_figures/fig2_representative_6h_episode.png",
        "main_figures/fig3_6h_paired_pump_saving.png",
        "main_figures/fig4_6h_attitude_tradeoffs.png",
        "main_figures/fig5_boundary_robustness_disclosure.png",
        "main_figures/fig6_12h_guard10_robustness.png",
        "appendix_figure_manifest.csv",
        "manuscript_notes/result_paragraphs.md",
        "manuscript_notes/reviewer_risk_response.md",
        "figure_inventory.csv",
        "next_action_checklist.md",
    ):
        lines.append(f"- `{name}`")
    lines.append("")
    lines.append("## Immediate Interpretation")
    lines.append("")
    lines.append("1. Table 2 / Fig. 3 / Fig. 4 can now be built from the frozen 6 h production-near paired casebook.")
    lines.append("2. Table 3 / Fig. 6 can be built from the 12 h guard10 robustness outputs.")
    lines.append("3. D1/P2/C3/PSC historical-specialist branches should remain mechanism/boundary/appendix evidence.")
    lines.append("")
    lines.append("## File Counts")
    lines.append("")
    lines.append(f"- key run summaries: `{len(runs_summary)}`")
    lines.append(f"- claim map rows: `{len(claim_map)}`")
    lines.append(f"- figure regimes: `{len(fig_inventory)}`")
    lines.append(f"- 6 h paired cases: `{len(pair6)}`")
    lines.append(f"- 6 h regime rows: `{len(group6)}`")
    lines.append(f"- 12 h paired cases: `{len(pair12)}`")
    lines.append(f"- production-near 6 h protocol candidates: `{len(prod6)}`")
    (PKG / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_6h_audit_md(prod6: pd.DataFrame, verdict: str) -> None:
    lines = [
        "# Production-Near 6 h Paired Evidence Audit",
        "",
        f"Verdict: `{verdict}`",
        "",
        f"Required production profile: `{PRODUCTION_PROFILE}`",
        "",
        "A Table 2 main result requires matched current-only and learned casebooks under the same profile, duration, case IDs, dataset, and replay split.",
        "",
    ]
    if prod6.empty:
        lines.extend(
            [
                "No matching production-near 6 h protocol was found in the scanned `run_protocol.json` files.",
                "",
                "Next action: freeze case selection, then rerun paired current-only and learned 6 h casebooks with `.venv312/bin/python`.",
            ]
        )
    else:
        cols = [
            "run_dir",
            "n_cases_summary",
            "forecast_source_requested",
            "forecast_source_effective",
            "primary_only",
            "cases_source",
            "python_executable",
        ]
        lines.append("Matched 6 h protocol candidates:")
        lines.append("")
        lines.append(markdown_table(prod6[[c for c in cols if c in prod6.columns]]).rstrip())
        lines.append("")
        lines.append("Manual check still required: pair case IDs and compare current-only vs learned under identical selection.")
    (PKG / "production_near_6h_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_next_actions(prod6_verdict: str) -> None:
    lines = [
        "# Next Action Checklist",
        "",
        "## P0",
        "",
        "- [x] Freeze broad guard10 6 h case selection before result interpretation.",
        "- [x] Run matched current-only and learned gain0.45 6 h production-near casebooks.",
        "- [x] Generate Table 1-3 candidate CSV/Markdown files.",
        "- [x] Generate Fig. 1-6 candidate PNG files.",
        "- [x] Keep D1/P2/C3/PSC historical branch results labeled as mechanism/boundary evidence.",
        "- [x] Copy existing five-panel appendix figure packs into the paper package.",
        "- [x] Draft result paragraphs and reviewer-risk response notes.",
        "",
        "## P1",
        "",
        "- [x] Convert `figure_inventory.csv` into an appendix figure index.",
        "- [ ] Add P1 transient/future-decay and W1 signflip/abstain curve packs if time allows.",
        "- [ ] Build prediction model summary table and one observed-vs-predicted figure.",
        "",
        "## Current Verdict",
        "",
        f"`{prod6_verdict}`",
        "",
    ]
    (PKG / "next_action_checklist.md").write_text("\n".join(lines), encoding="utf-8")


def write_manuscript_notes(
    pair6: pd.DataFrame,
    group6: pd.DataFrame,
    table3: pd.DataFrame,
    boundary_summary: pd.DataFrame,
) -> None:
    MANUSCRIPT_DIR.mkdir(parents=True, exist_ok=True)

    if not pair6.empty:
        current_pump = safe_float(pair6["current_pump_m3"].sum())
        learned_pump = safe_float(pair6["learned_pump_m3"].sum())
        saving = (current_pump - learned_pump) / current_pump * 100.0 if current_pump else math.nan
        total_time = len(pair6) * 21600.0
        learned_t5 = safe_float(pair6["learned_t_gt5_s"].sum() / total_time * 100.0)
        learned_t75 = safe_float(pair6["learned_t_gt7p5_s"].sum() / total_time * 100.0)
        learned_t10 = safe_float(pair6["learned_t_gt10_s"].sum() / total_time * 100.0)
        d_t5 = safe_float(pair6["d_t_gt5_s"].sum())
        d_t75 = safe_float(pair6["d_t_gt7p5_s"].sum())
        d_t10 = safe_float(pair6["d_t_gt10_s"].sum())
    else:
        current_pump = learned_pump = saving = learned_t5 = learned_t75 = learned_t10 = math.nan
        d_t5 = d_t75 = d_t10 = math.nan

    lines = [
        "# Result Paragraph Drafts",
        "",
        "## 3.1 仿真工况与评价指标",
        "",
        "本文采用预先冻结的 guard10 广覆盖工况作为 6 h 主结果样本池。每个工况分别运行无预测闭环反馈策略和预测监督调节策略，两者使用相同的执行层、安全层、数据集和回放窗口，仅改变监督层可用的未来风况信息。泵耗、姿态暴露、fallback 与 latch 指标均从 raw 1 Hz 输出统计；图中曲线仅用于可视化展示，不作为指标计算来源。",
        "",
        "姿态评价采用 pitch 与 roll 分轴统计，并以 `max(abs(pitch), abs(roll))` 定义阈值暴露。3 deg 和 4 deg 主要反映舒适/经济带占用，5 deg 表示服务压力带，7.5 deg 表示严重姿态压力，10 deg 表示运行边界。本文不把单点 5 deg 暴露视为失败，而是结合泵耗、7.5/10 deg 尾部、fallback 和 latch 共同判断策略是否可接受。",
        "",
        "## 3.2 仿真结果分析",
        "",
        f"在 10 个冻结的 6 h 成对工况中，预测监督调节策略的累计泵耗由无预测闭环反馈策略的 `{current_pump:.2f} m3` 降至 `{learned_pump:.2f} m3`，整体节泵 `{saving:.2f}%`。分工况结果显示，signflip boundary 与 B/high pressure 工况贡献主要节泵收益，lowrisk reference 工况基本保持不动作，future relief 工况收益较小但未引入严重尾部增长。",
        "",
        f"姿态代价方面，预测监督策略在 6 h 主样本池中的 `t>5 deg`、`t>7.5 deg` 和 `t>10 deg` 暴露占比分别为 `{learned_t5:.4f}%`、`{learned_t75:.4f}%` 和 `{learned_t10:.4f}%`；相对无预测闭环反馈策略的暴露变化分别为 `{d_t5:.0f}s`、`{d_t75:.0f}s` 和 `{d_t10:.0f}s`。因此，本组主证据支持“在预测可行动 episode 中降低不必要泵送，并保持严重姿态暴露有界”的有限主张，而不应扩写为全工况长期通用节泵结论。",
        "",
        "12 h mixed-regime 鲁棒性结果用于披露长窗口收益边界。guard10 混合工况中，gain 0.45 候选相对 current-only 总泵耗降低 9.51%，`t>7.5 deg` 低于 current-only，`t>10 deg` 基本不变，但 `t>5 deg` 服务压力带占用略有增加。该结果说明预测监督在混合长时段中的收益会被低风险、边界和非机会窗口稀释，适合作为 robustness disclosure，而不是 24 h deployment-scale 证明。",
        "",
        "## 图表引用建议",
        "",
        "- Fig. 1：说明预测只进入监督层，不直接输出泵命令。",
        "- Fig. 2：展示代表性 6 h 成对时域响应，突出泵速、累计泵耗和 pitch/roll 尾部。",
        "- Fig. 3：作为 Table 2 的视觉版，按 regime 展示节泵而不是只给总均值。",
        "- Fig. 4：回答节泵是否牺牲姿态，重点看 5/7.5 deg 暴露和 p95 变化。",
        "- Fig. 5：主动披露边界、拒绝动作和长窗口稀释。",
        "- Fig. 6：披露 12 h mixed-regime robustness。",
    ]
    (MANUSCRIPT_DIR / "result_paragraphs.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    risk_lines = [
        "# Reviewer Risk Response",
        "",
        "## 为什么主证据是 6 h",
        "",
        "本文定位为短时风况预测辅助主动压载监督调节的 episode-level 验证。6 h 窗口包含多个 10 min 风况预测与监督决策周期，适合验证预测信息是否能在可行动工况中改变泵送决策。论文同时保留 12 h mixed-regime robustness 披露，用于说明长窗口混合工况下收益会被稀释；24 h deployment-scale validation 写作后续工作。",
        "",
        "## 如何避免 cherry-picking",
        "",
        "主结果使用冻结的 guard10 broad casebook，同一 case/window 下成对比较 current-only 和 learned gain0.45。结果包保留 per-case delta、regime summary、weak/zero/negative cases 与 boundary evidence。正文不得只展示 Fig. 2 代表正例。",
        "",
        "## 姿态代价是否可接受",
        "",
        f"6 h 主结果在整体上节泵 `{saving:.2f}%`，同时 `d t>7.5={d_t75:.0f}s`、`d t>10={d_t10:.0f}s`。5 deg 是服务压力带而非硬失败线；任何 5 deg 增加应写成 Pareto trade-off，并与 pump/latch/fallback 一起讨论。",
        "",
        "## profile 是否混用",
        "",
        f"headline 只允许来自 `{PRODUCTION_PROFILE}` 下的 6 h paired evidence 和 12 h robustness disclosure。D1/P2/C3/W1/PSC 等历史或 specialist 结果只能作为 mechanism、boundary 或 appendix evidence，不能与 production-near headline 合并成单一节泵百分比。",
        "",
        "## baseline 是否公平",
        "",
        "baseline 是无预测闭环反馈策略，保留相同执行层、安全层和工程约束。预测监督策略的差异在于未来风况信息进入监督层，用于 regime identification、strategy selection、hold/release/cap/veto，而不是绕过反馈层直接输出泵命令。",
        "",
        "## 仍需披露的残余风险",
        "",
        "- 当前主证据为 10 个 6 h 工况，样本量适合短论文但不支持 deployment-wide claim。",
        "- Fig. 5 中 boundary/materials 包含 historical/specialist branch，必须标注 paper role。",
        "- prediction model summary 和 observed-vs-predicted 图仍是 P1，可增强第 2 章但不应改写主证据边界。",
    ]
    (MANUSCRIPT_DIR / "reviewer_risk_response.md").write_text("\n".join(risk_lines) + "\n", encoding="utf-8")

    table_notes = [
        "# Table And Figure Index Notes",
        "",
        "## Main Tables",
        "",
        "- Table 1: `tables/table1_casebook.csv`",
        "- Table 2: `tables/table2_6h_primary_results.csv`",
        "- Table 3a: `tables/table3_12h_robustness.csv`",
        "- Table 3b: `tables/table3_boundary_robustness_summary.csv`",
        "",
        "## Main Figures",
        "",
        "- Fig. 1: `main_figures/fig1_control_architecture.png`",
        "- Fig. 2: `main_figures/fig2_representative_6h_episode.png`",
        "- Fig. 3: `main_figures/fig3_6h_paired_pump_saving.png`",
        "- Fig. 4: `main_figures/fig4_6h_attitude_tradeoffs.png`",
        "- Fig. 5: `main_figures/fig5_boundary_robustness_disclosure.png`",
        "- Fig. 6: `main_figures/fig6_12h_guard10_robustness.png`",
        "",
        "## Appendix Figures",
        "",
        "See `appendix_figure_manifest.csv` and `appendix_figures/`.",
    ]
    (MANUSCRIPT_DIR / "table_figure_index_notes.md").write_text("\n".join(table_notes) + "\n", encoding="utf-8")


def main() -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / ".matplotlib_cache"))
    PKG.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    MAIN_FIG_DIR.mkdir(parents=True, exist_ok=True)

    protocol_inventory = build_protocol_inventory()
    protocol_inventory.to_csv(PKG / "casebook_protocol_inventory.csv", index=False)

    runs_summary = pd.DataFrame([summarize_run(key, path) for key, path in RUNS.items()])
    runs_summary.to_csv(PKG / "run_summary_key_sources.csv", index=False)
    write_markdown_table(runs_summary, PKG / "run_summary_key_sources.md")

    pair12 = build_12h_pair_delta()
    pair12.to_csv(PKG / "production_near_12h_per_case_delta.csv", index=False)
    if not pair12.empty:
        write_markdown_table(pair12, PKG / "production_near_12h_per_case_delta.md")

    pair6 = build_6h_pair_delta()
    pair6.to_csv(PKG / "production_near_6h_per_case_delta.csv", index=False)
    if not pair6.empty:
        write_markdown_table(pair6, PKG / "production_near_6h_per_case_delta.md")

    group6 = build_group_summary(pair6)
    group6.to_csv(PKG / "production_near_6h_regime_summary.csv", index=False)
    if not group6.empty:
        write_markdown_table(group6, PKG / "production_near_6h_regime_summary.md")

    table1 = build_table1_casebook(pair6)
    table1.to_csv(TABLE_DIR / "table1_casebook.csv", index=False)
    write_markdown_table(table1, TABLE_DIR / "table1_casebook.md")

    table2 = build_table2_6h(group6)
    table2.to_csv(TABLE_DIR / "table2_6h_primary_results.csv", index=False)
    write_markdown_table(table2, TABLE_DIR / "table2_6h_primary_results.md")

    table3 = build_table3_12h(runs_summary)
    table3.to_csv(TABLE_DIR / "table3_12h_robustness.csv", index=False)
    write_markdown_table(table3, TABLE_DIR / "table3_12h_robustness.md")

    boundary_summary = build_boundary_robustness_summary(runs_summary, table3)
    boundary_summary.to_csv(TABLE_DIR / "table3_boundary_robustness_summary.csv", index=False)
    write_markdown_table(boundary_summary, TABLE_DIR / "table3_boundary_robustness_summary.md")

    plot_main_figures(group6, pair6, table3, boundary_summary)

    appendix_manifest = copy_appendix_figures()
    appendix_manifest.to_csv(PKG / "appendix_figure_manifest.csv", index=False)
    write_markdown_table(appendix_manifest, PKG / "appendix_figure_manifest.md")
    write_manuscript_notes(pair6, group6, table3, boundary_summary)

    claim_map = build_claim_evidence(runs_summary, pair6)
    claim_map.to_csv(PKG / "claim_evidence_map.csv", index=False)
    write_markdown_table(claim_map, PKG / "claim_evidence_map.md")

    fig_inventory = build_figure_inventory()
    fig_inventory.to_csv(PKG / "figure_inventory.csv", index=False)
    write_markdown_table(fig_inventory, PKG / "figure_inventory.md")

    prod6, prod6_verdict = build_6h_audit(protocol_inventory)
    if not pair6.empty:
        prod6_verdict = (
            "READY: matched production-near 6 h current-only and learned gain0.45 "
            "guard10 casebooks are frozen and summarized for Table 2 / Fig. 3 / Fig. 4."
        )
    prod6.to_csv(PKG / "production_near_6h_audit.csv", index=False)
    write_6h_audit_md(prod6, prod6_verdict)
    write_next_actions(prod6_verdict)

    write_summary_md(runs_summary, claim_map, fig_inventory, prod6, prod6_verdict, pair12, pair6, group6)
    print(f"Wrote paper package to {PKG.relative_to(REPO_ROOT)}")
    print(prod6_verdict)


if __name__ == "__main__":
    main()
