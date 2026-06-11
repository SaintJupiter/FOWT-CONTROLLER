#!/usr/bin/env python3
"""Summarize No.4 25% overnight mild-arm runs.

The script compares new primary-only run directories against the frozen 0hao
baseline timeseries from the 96-case degradation ladder.  It is intentionally
agnostic to the exact run set: every `smoke_*` / `expanded_*` / `full_*`
directory under the overnight output folder is summarized if it contains
timeseries.
"""

from __future__ import annotations

from pathlib import Path
import re

import numpy as np
import pandas as pd


REPO = Path(__file__).resolve().parents[2]
BASE = REPO / "outputs" / "wind_prediction" / "psc_selector_mixed_pool_v1"
OUT = BASE / "psc_4hao_25pct_overnight_v1"
BASELINE_DIR = BASE / "degradation_ladder_96case_pair" / "current_forecast_adaptive" / "timeseries"
BASELINE_METRICS = BASE / "degradation_ladder_96case_pair" / "degradation_ladder_case_metrics.csv"
SMOKE_CASES = BASE / "psc_4hao_moderate_arm_feasibility_v1" / "4hao_smoke_cases.csv"
CASE_TABLES = [
    SMOKE_CASES,
    BASE / "psc_4hao_v2_comfort_gate_projection_v1" / "4hao_v2_candidate_gate_table.csv",
    BASE / "psc_structural_selector_96case_pair_3hao_v2_learned_v1" / "structural_selector_case_decisions.csv",
]


def _load_case_map() -> dict[str, str]:
    mapping: dict[str, str] = {}
    for path in CASE_TABLES:
        if not path.exists():
            continue
        df = pd.read_csv(path)
        if "case_id" not in df.columns:
            continue
        for raw in df["case_id"].dropna().astype(str):
            clean = re.sub(r"^\d+_", "", raw)
            mapping[clean] = raw
            mapping[raw] = raw
    return mapping


def _canonical_from_path(path: Path, mapping: dict[str, str]) -> str:
    name = path.name
    stem = name.replace("_prediction_primary_econ_timeseries.csv", "")
    stem = stem.replace("_current_forecast_adaptive_timeseries.csv", "")
    stem = re.sub(r"_\d{4}-\d{2}-\d{2}_\d{6}.*$", "", stem)
    for clean, raw in sorted(mapping.items(), key=lambda x: len(x[0]), reverse=True):
        if stem == raw or stem.endswith(clean):
            return raw
    return stem


def _metrics(path: Path) -> dict[str, float]:
    df = pd.read_csv(path, low_memory=False)
    pitch = pd.to_numeric(df.get("pitch_deg"), errors="coerce").fillna(0.0).to_numpy(float)
    roll = pd.to_numeric(df.get("roll_deg"), errors="coerce").fillna(0.0).to_numpy(float)
    axis = np.maximum(np.abs(pitch), np.abs(roll))
    pump = pd.to_numeric(df.get("pump_total_rate_m3_min"), errors="coerce").fillna(0.0).to_numpy(float)
    fallback = pd.to_numeric(
        df.get("preview_primary_safety_fallback", pd.Series(np.zeros(len(df)))),
        errors="coerce",
    ).fillna(0.0).to_numpy(float)
    out = {
        "pump_m3": float(np.sum(np.abs(pump)) / 60.0),
        "time_gt3_s": float(np.sum(axis > 3.0)),
        "time_gt4_s": float(np.sum(axis > 4.0)),
        "time_gt5_s": float(np.sum(axis > 5.0)),
        "fallback_s": float(np.sum(fallback)),
        "p95_axis_deg": float(np.quantile(axis, 0.95)) if len(axis) else 0.0,
        "max_axis_deg": float(np.max(axis)) if len(axis) else 0.0,
        "partial_refresh_count": 0.0,
        "partial_refresh_fraction_mean": 0.0,
    }
    if "economy_pump_budget_partial_refresh_count" in df.columns:
        partial = pd.to_numeric(
            df["economy_pump_budget_partial_refresh_count"],
            errors="coerce",
        ).fillna(0.0)
        out["partial_refresh_count"] = float(partial.max())
    if "economy_pump_budget_partial_refresh_fraction" in df.columns:
        frac = pd.to_numeric(
            df["economy_pump_budget_partial_refresh_fraction"],
            errors="coerce",
        ).fillna(0.0)
        active = frac[frac > 0.0]
        out["partial_refresh_fraction_mean"] = (
            float(active.mean()) if len(active) else 0.0
        )
    return out


def _planner_metrics(timeseries_path: Path) -> dict[str, float | str]:
    log_path = (
        timeseries_path.parent.parent
        / "planner_logs"
        / timeseries_path.name.replace("_timeseries.csv", "_planner_log.csv")
    )
    out: dict[str, float | str] = {
        "partial_refresh_count": 0.0,
        "partial_refresh_fraction_mean": 0.0,
        "budget_hold_count": 0.0,
        "budget_refresh_count": 0.0,
        "refresh_owners": "",
    }
    if not log_path.exists():
        return out
    df = pd.read_csv(log_path, low_memory=False)
    if "economy_pump_budget_partial_refresh_count" in df.columns:
        out["partial_refresh_count"] = float(
            pd.to_numeric(
                df["economy_pump_budget_partial_refresh_count"],
                errors="coerce",
            )
            .fillna(0.0)
            .max()
        )
    if "economy_pump_budget_partial_refresh_fraction" in df.columns:
        frac = pd.to_numeric(
            df["economy_pump_budget_partial_refresh_fraction"],
            errors="coerce",
        ).fillna(0.0)
        active = frac[frac > 0.0]
        out["partial_refresh_fraction_mean"] = (
            float(active.mean()) if len(active) else 0.0
        )
    if "economy_pump_budget_hold_count" in df.columns:
        out["budget_hold_count"] = float(
            pd.to_numeric(df["economy_pump_budget_hold_count"], errors="coerce")
            .fillna(0.0)
            .max()
        )
    if "economy_pump_budget_refresh_count" in df.columns:
        out["budget_refresh_count"] = float(
            pd.to_numeric(df["economy_pump_budget_refresh_count"], errors="coerce")
            .fillna(0.0)
            .max()
        )
    if "prediction_primary_refresh_owner" in df.columns:
        owners = (
            df["prediction_primary_refresh_owner"]
            .fillna("")
            .astype(str)
            .value_counts()
            .head(6)
        )
        out["refresh_owners"] = ";".join(
            f"{name}:{count}" for name, count in owners.items() if name
        )
    return out


def _baseline_index(mapping: dict[str, str]) -> dict[str, dict[str, float]]:
    rows: dict[str, dict[str, float]] = {}
    for path in BASELINE_DIR.glob("*_timeseries.csv"):
        case_id = _canonical_from_path(path, mapping)
        rows[case_id] = _metrics(path)
    if BASELINE_METRICS.exists():
        m = pd.read_csv(BASELINE_METRICS)
        m = m[m["arm"].eq("current_forecast_adaptive")]
        for _, row in m.iterrows():
            case_id = str(row["case_id"])
            rows.setdefault(case_id, {})
            for col in ("pump_m3", "time_gt5_s", "fallback_s", "p95_axis_deg", "max_axis_deg"):
                if col in row and pd.notna(row[col]):
                    rows[case_id][col] = float(row[col])
    return rows


def _run_dirs() -> list[Path]:
    dirs = []
    for path in sorted(OUT.iterdir() if OUT.exists() else []):
        if path.is_dir() and (path / "timeseries").exists():
            dirs.append(path)
    return dirs


def _label_from_dir(path: Path) -> str:
    name = path.name
    m_frac = re.search(r"frac(\d+)", name)
    m_budget = re.search(r"budget(\d+)", name)
    m_rel = re.search(r"release(\d+)", name)
    parts = ["4号v2"]
    if m_frac:
        parts.append(f"温和追踪{int(m_frac.group(1))}%")
    if m_budget:
        parts.append(f"budget{m_budget.group(1)}")
    if m_rel:
        rel = m_rel.group(1)
        if len(rel) == 2:
            parts.append(f"release{rel[0]}.{rel[1]}")
        else:
            parts.append(f"release{rel}")
    return "-".join(parts)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    mapping = _load_case_map()
    baseline = _baseline_index(mapping)
    case_rows: list[dict[str, object]] = []
    for run_dir in _run_dirs():
        for ts in sorted((run_dir / "timeseries").glob("*_timeseries.csv")):
            case_id = _canonical_from_path(ts, mapping)
            if case_id not in baseline:
                continue
            metrics = _metrics(ts)
            planner = _planner_metrics(ts)
            base = baseline[case_id]
            saved = float(base["pump_m3"]) - float(metrics["pump_m3"])
            row: dict[str, object] = {
                "run_id": run_dir.name,
                "candidate_name": _label_from_dir(run_dir),
                "case_id": case_id,
                "saved_m3": saved,
                "saving_pct": 100.0 * saved / max(float(base["pump_m3"]), 1e-9),
                "baseline_pump_m3": float(base["pump_m3"]),
                "primary_pump_m3": float(metrics["pump_m3"]),
                "fallback_s": float(metrics["fallback_s"]),
                "d_fallback_s": float(metrics["fallback_s"]) - float(base.get("fallback_s", 0.0)),
                "time_gt3_s": float(metrics["time_gt3_s"]),
                "d_time_gt3_s": float(metrics["time_gt3_s"]) - float(base.get("time_gt3_s", 0.0)),
                "time_gt4_s": float(metrics["time_gt4_s"]),
                "d_time_gt4_s": float(metrics["time_gt4_s"]) - float(base.get("time_gt4_s", 0.0)),
                "time_gt5_s": float(metrics["time_gt5_s"]),
                "d_time_gt5_s": float(metrics["time_gt5_s"]) - float(base.get("time_gt5_s", 0.0)),
                "p95_axis_deg": float(metrics["p95_axis_deg"]),
                "d_p95_axis_deg": float(metrics["p95_axis_deg"]) - float(base.get("p95_axis_deg", 0.0)),
                "max_axis_deg": float(metrics["max_axis_deg"]),
                "5deg_p95_margin": 5.0 - float(metrics["p95_axis_deg"]),
                "partial_refresh_count": float(planner["partial_refresh_count"])
                or float(metrics["partial_refresh_count"]),
                "partial_refresh_fraction_mean": float(
                    planner["partial_refresh_fraction_mean"]
                )
                or float(metrics["partial_refresh_fraction_mean"]),
                "budget_hold_count": float(planner["budget_hold_count"]),
                "budget_refresh_count": float(planner["budget_refresh_count"]),
                "refresh_owners": str(planner["refresh_owners"]),
            }
            row["safety_fail"] = int(
                row["d_fallback_s"] > 0
                or row["d_time_gt5_s"] >= 60
                or row["p95_axis_deg"] >= 5.0
            )
            case_rows.append(row)
    case_df = pd.DataFrame(case_rows)
    case_df.to_csv(OUT / "case_metrics.csv", index=False)
    if case_df.empty:
        return
    frontier = (
        case_df.groupby(["run_id", "candidate_name"], sort=False)
        .agg(
            opened_case_count=("case_id", "count"),
            fail_case_count=("safety_fail", "sum"),
            saved_m3=("saved_m3", "sum"),
            baseline_pump_m3=("baseline_pump_m3", "sum"),
            fallback_s=("fallback_s", "sum"),
            d_fallback_s=("d_fallback_s", "sum"),
            time_gt3_s=("time_gt3_s", "sum"),
            d_time_gt3_s=("d_time_gt3_s", "sum"),
            time_gt4_s=("time_gt4_s", "sum"),
            d_time_gt4_s=("d_time_gt4_s", "sum"),
            time_gt5_s=("time_gt5_s", "sum"),
            d_time_gt5_s=("d_time_gt5_s", "sum"),
            p95_axis_deg=("p95_axis_deg", "max"),
            d_p95_axis_deg=("d_p95_axis_deg", "max"),
            max_axis_deg=("max_axis_deg", "max"),
            partial_refresh_count=("partial_refresh_count", "sum"),
            budget_hold_count=("budget_hold_count", "sum"),
            budget_refresh_count=("budget_refresh_count", "sum"),
        )
        .reset_index()
    )
    frontier["saving_pct"] = (
        100.0 * frontier["saved_m3"] / frontier["baseline_pump_m3"].clip(lower=1e-9)
    )
    frontier["5deg_p95_margin"] = 5.0 - frontier["p95_axis_deg"]
    case_ids = (
        case_df.groupby("run_id")["case_id"]
        .apply(lambda s: ";".join(s.astype(str).tolist()))
        .rename("case_ids")
        .reset_index()
    )
    frontier = frontier.merge(case_ids, on="run_id", how="left")
    frontier.to_csv(OUT / "pareto_frontier.csv", index=False)
    strict = case_df[
        (case_df["d_fallback_s"] <= 0)
        & (case_df["d_time_gt5_s"] <= 0)
        & (case_df["d_p95_axis_deg"] <= 0.50)
    ]
    appendix = case_df.drop(strict.index)
    strict.to_csv(OUT / "accepted_mainline_cases.csv", index=False)
    appendix.to_csv(OUT / "rejected_or_appendix_cases.csv", index=False)

    frontier_text = frontier.sort_values("saving_pct", ascending=False).to_string(
        index=False
    )
    lines = [
        "# 4hao 25% Overnight Interim Readout",
        "",
        "## Pareto Frontier",
        "",
        "```",
        frontier_text,
        "```",
        "",
        "## Notes",
        "",
        "- This readout is generated from completed run directories only.",
        "- Full 96-case totals should be interpreted only for runs that actually cover all 96 cases.",
    ]
    (OUT / "README_interim.md").write_text("\n".join(lines), encoding="utf-8")
    print(frontier_text)


if __name__ == "__main__":
    main()
