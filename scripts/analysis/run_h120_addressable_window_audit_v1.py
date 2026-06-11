#!/usr/bin/env python3
"""Read-only addressable-window audit for h120 prepare interface.

This script does not change the controller.  It measures how often the
proposed Layer-1 pre-floor eligibility window appears in existing v1.6 +
f120-oracle baseline traces.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REPO = Path(__file__).resolve().parents[2]
BASE = REPO / "outputs/wind_prediction/h120_learned_risk_scheduler_probe_v1"
OUT = REPO / "outputs/wind_prediction/h120_addressable_window_audit_v1"
RAW = OUT / "raw_tables"
PAPER = OUT / "paper_ready"
DEBUG = OUT / "debug"


GROUPS = {
    "guard10": BASE / "guard10_v16_oracle_baseline",
    "broader20": BASE / "broader20_v16_oracle_baseline",
}


def ensure_dirs() -> None:
    for path in (OUT, RAW, PAPER, DEBUG):
        path.mkdir(parents=True, exist_ok=True)


def num(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col not in df.columns:
        return pd.Series(default, index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce").fillna(default).astype(float)


def flag(df: pd.DataFrame, col: str, default: int = 0) -> pd.Series:
    return (num(df, col, default) > 0).astype(int)


def case_id_from_log(path: Path) -> str:
    stem = path.name.replace("_prediction_primary_econ_planner_log.csv", "")
    parts = stem.split("_")
    if parts and parts[0].isdigit():
        return "_".join(parts[1:-2])
    return "_".join(parts[:-2])


def markdown_table(df: pd.DataFrame, max_rows: int | None = None) -> str:
    if df is None or df.empty:
        return "_empty_"
    data = df.copy()
    if max_rows is not None:
        data = data.head(max_rows)

    def fmt(value: Any) -> str:
        if pd.isna(value):
            return ""
        if isinstance(value, float):
            return f"{value:.3f}"
        return str(value)

    lines = [
        "| " + " | ".join(str(c) for c in data.columns) + " |",
        "| " + " | ".join(["---"] * len(data.columns)) + " |",
    ]
    for _, row in data.iterrows():
        lines.append("| " + " | ".join(fmt(row[c]) for c in data.columns) + " |")
    return "\n".join(lines)


def load_timeseries_future(ts_path: Path, bucket_times: pd.Series) -> pd.DataFrame:
    if not ts_path.exists():
        return pd.DataFrame(
            {
                "current_time_s": bucket_times,
                "future60_time_over5_s": np.nan,
                "future60_idle_over5_s": np.nan,
                "future60_max_axis_max": np.nan,
                "future60_pump_m3": np.nan,
            }
        )
    ts = pd.read_csv(ts_path)
    t = num(ts, "t_s")
    max_axis = np.maximum(num(ts, "pitch_deg").abs(), num(ts, "roll_deg").abs())
    pump_rate = num(ts, "pump_total_rate_m3_min")
    dt = float(np.nanmedian(np.diff(t.to_numpy()))) if len(t) > 2 else 1.0
    if not np.isfinite(dt) or dt <= 0:
        dt = 1.0
    rows: list[dict[str, Any]] = []
    for current_time_s in bucket_times:
        start = float(current_time_s)
        end = start + 3600.0
        mask = (t > start) & (t <= end)
        sub_axis = max_axis[mask]
        sub_pump = pump_rate[mask]
        over5 = sub_axis > 5.0
        idle_over5 = over5 & (sub_pump.abs() <= 1e-6)
        rows.append(
            {
                "current_time_s": start,
                "future60_time_over5_s": float(over5.sum() * dt),
                "future60_idle_over5_s": float(idle_over5.sum() * dt),
                "future60_max_axis_max": float(sub_axis.max()) if len(sub_axis) else np.nan,
                "future60_pump_m3": float((sub_pump.clip(lower=0).sum() * dt) / 60.0)
                if len(sub_pump)
                else np.nan,
            }
        )
    return pd.DataFrame(rows)


def enrich_log(dataset: str, log_path: Path) -> pd.DataFrame:
    df = pd.read_csv(log_path)
    case_id = case_id_from_log(log_path)
    ts_path = (
        log_path.parent.parent
        / "timeseries"
        / log_path.name.replace("_planner_log.csv", "_timeseries.csv")
    )
    out = pd.DataFrame(index=df.index)
    out["dataset"] = dataset
    out["case_id"] = case_id
    out["bucket"] = num(df, "bucket").astype(int)
    out["current_time_s"] = num(df, "current_time_s")
    out["current_pitch_deg"] = num(df, "current_pitch_deg")
    out["current_roll_deg"] = num(df, "current_roll_deg")
    out["max_axis_deg"] = np.maximum(out["current_pitch_deg"].abs(), out["current_roll_deg"].abs())
    out["dominant_axis"] = np.where(out["current_pitch_deg"].abs() >= out["current_roll_deg"].abs(), "pitch", "roll")
    out["posture_trend_10m_deg"] = out["max_axis_deg"].diff().fillna(0.0)
    out["floor_active"] = flag(df, "reactive_floor_active")
    out["floor_medium_active"] = flag(df, "reactive_floor_medium_delay_active")
    out["target_age_s"] = num(df, "prediction_primary_target_age_s")
    out["target_error_mean_kg"] = num(df, "reactive_floor_target_err_mean_kg")
    alt_err = num(df, "active_effectiveness_target_err_mean_kg", np.nan)
    out["target_error_mean_kg"] = out["target_error_mean_kg"].where(out["target_error_mean_kg"] > 0, alt_err.fillna(0.0))
    out["pump_rate_m3_min"] = num(df, "active_effectiveness_pump_rate_m3_min")
    alt_rate = num(df, "no_unexplained_hold_pump_rate_m3_min", np.nan)
    out["pump_rate_m3_min"] = out["pump_rate_m3_min"].where(out["pump_rate_m3_min"] > 0, alt_rate.fillna(0.0))
    out["pump_idle"] = ((flag(df, "active_effectiveness_pump_idle") > 0) | (out["pump_rate_m3_min"].abs() <= 1e-6)).astype(int)
    out["near_b0_norm"] = num(df, "pressure_block0_norm")
    out["near_b1_norm"] = num(df, "pressure_block1_norm")
    out["near_b2_norm"] = num(df, "pressure_block2_norm")
    out["near_max_norm"] = out[["near_b0_norm", "near_b1_norm", "near_b2_norm"]].max(axis=1)
    out["near_last_norm"] = out["near_b2_norm"]
    out["near_intensification"] = out["near_last_norm"] - out["near_b0_norm"]
    out["far_60_80_norm"] = num(df, "far_horizon_norm_60_80")
    out["far_80_100_norm"] = num(df, "far_horizon_norm_80_100")
    out["far_100_120_norm"] = num(df, "far_horizon_norm_100_120")
    out["far_min_norm"] = out[["far_60_80_norm", "far_80_100_norm", "far_100_120_norm"]].min(axis=1)
    out["far_max_norm"] = out[["far_60_80_norm", "far_80_100_norm", "far_100_120_norm"]].max(axis=1)
    out["far_persistent_high"] = (out["far_min_norm"] >= 1.0).astype(int)
    out["far_hidden_intensification"] = flag(df, "far_horizon_hidden_intensification")
    out["far_reversal"] = flag(df, "far_horizon_reversal")
    future = load_timeseries_future(ts_path, out["current_time_s"])
    out = out.merge(future, on="current_time_s", how="left")
    return out


def build_bucket_table() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for dataset, root in GROUPS.items():
        for log_path in sorted((root / "planner_logs").glob("*_planner_log.csv")):
            frames.append(enrich_log(dataset, log_path))
    if not frames:
        raise FileNotFoundError("No baseline planner logs found for addressable-window audit.")
    df = pd.concat(frames, ignore_index=True)
    return df


def apply_window_flags(
    df: pd.DataFrame,
    *,
    trend_eps: float = 0.05,
    target_age_s: float = 600.0,
    target_error_kg: float = 500.0,
    near_norm: float = 1.0,
) -> pd.DataFrame:
    out = df.copy()
    out["posture_4_5"] = ((out["max_axis_deg"] >= 4.0) & (out["max_axis_deg"] < 5.0)).astype(int)
    out["prefloor"] = (out["floor_active"] == 0).astype(int)
    out["trend_worsening"] = (out["posture_trend_10m_deg"] > trend_eps).astype(int)
    out["target_quiet_stale"] = (
        (out["target_age_s"] >= target_age_s)
        & (out["target_error_mean_kg"].abs() <= target_error_kg)
    ).astype(int)
    out["near_pressure_support"] = (out["near_max_norm"] >= near_norm).astype(int)
    out["layer1_addressable"] = (
        (out["posture_4_5"] > 0)
        & (out["prefloor"] > 0)
        & (out["trend_worsening"] > 0)
        & (out["target_quiet_stale"] > 0)
        & (out["pump_idle"] > 0)
        & (out["near_pressure_support"] > 0)
    ).astype(int)
    out["near_miss_no_trend"] = (
        (out["posture_4_5"] > 0)
        & (out["prefloor"] > 0)
        & (out["target_quiet_stale"] > 0)
        & (out["pump_idle"] > 0)
        & (out["near_pressure_support"] > 0)
    ).astype(int)
    out["future60_has_over5"] = (out["future60_time_over5_s"] > 0).astype(int)
    out["future60_expensive"] = (
        (out["future60_time_over5_s"] >= 120.0)
        | (out["future60_idle_over5_s"] >= 120.0)
    ).astype(int)
    return out


def summarize(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    for dataset, group in list(df.groupby("dataset")) + [("all", df)]:
        total = len(group)
        addr = group[group["layer1_addressable"] > 0]
        near_miss = group[group["near_miss_no_trend"] > 0]
        rows.append(
            {
                "dataset": dataset,
                "cases": int(group["case_id"].nunique()),
                "total_buckets": int(total),
                "posture_4_5_buckets": int(group["posture_4_5"].sum()),
                "near_miss_no_trend_buckets": int(len(near_miss)),
                "addressable_buckets": int(len(addr)),
                "addressable_ratio": float(len(addr) / max(total, 1)),
                "addressable_cases": int(addr["case_id"].nunique()),
                "addressable_future60_time_over5_mean_s": float(addr["future60_time_over5_s"].mean()) if len(addr) else np.nan,
                "addressable_future60_idle_over5_mean_s": float(addr["future60_idle_over5_s"].mean()) if len(addr) else np.nan,
                "addressable_future60_expensive_rate": float(addr["future60_expensive"].mean()) if len(addr) else np.nan,
                "near_miss_future60_time_over5_mean_s": float(near_miss["future60_time_over5_s"].mean()) if len(near_miss) else np.nan,
                "near_miss_future60_idle_over5_mean_s": float(near_miss["future60_idle_over5_s"].mean()) if len(near_miss) else np.nan,
            }
        )
    summary = pd.DataFrame(rows)
    case_rows = []
    for (dataset, case_id), group in df.groupby(["dataset", "case_id"]):
        addr = group[group["layer1_addressable"] > 0]
        case_rows.append(
            {
                "dataset": dataset,
                "case_id": case_id,
                "total_buckets": int(len(group)),
                "posture_4_5_buckets": int(group["posture_4_5"].sum()),
                "addressable_buckets": int(len(addr)),
                "addressable_ratio": float(len(addr) / max(len(group), 1)),
                "addressable_future60_time_over5_sum_s": float(addr["future60_time_over5_s"].sum()) if len(addr) else 0.0,
                "addressable_future60_idle_over5_sum_s": float(addr["future60_idle_over5_s"].sum()) if len(addr) else 0.0,
                "future60_time_over5_sum_s_all_buckets": float(group["future60_time_over5_s"].sum()),
                "future60_idle_over5_sum_s_all_buckets": float(group["future60_idle_over5_s"].sum()),
            }
        )
    case_summary = pd.DataFrame(case_rows).sort_values(
        ["dataset", "addressable_buckets", "addressable_future60_time_over5_sum_s"],
        ascending=[True, False, False],
    )
    return summary, case_summary


def threshold_sensitivity(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for trend_eps in (0.0, 0.03, 0.05, 0.10):
        for target_age_s in (300.0, 600.0, 1200.0):
            for near_norm in (0.9, 1.0, 1.1):
                flagged = apply_window_flags(
                    df,
                    trend_eps=trend_eps,
                    target_age_s=target_age_s,
                    target_error_kg=500.0,
                    near_norm=near_norm,
                )
                for dataset, group in list(flagged.groupby("dataset")) + [("all", flagged)]:
                    addr = group[group["layer1_addressable"] > 0]
                    rows.append(
                        {
                            "dataset": dataset,
                            "trend_eps": trend_eps,
                            "target_age_s": target_age_s,
                            "target_error_kg": 500.0,
                            "near_norm": near_norm,
                            "addressable_buckets": int(len(addr)),
                            "addressable_ratio": float(len(addr) / max(len(group), 1)),
                            "addressable_cases": int(addr["case_id"].nunique()),
                            "future60_time_over5_mean_s": float(addr["future60_time_over5_s"].mean()) if len(addr) else np.nan,
                            "future60_idle_over5_mean_s": float(addr["future60_idle_over5_s"].mean()) if len(addr) else np.nan,
                        }
                    )
    return pd.DataFrame(rows)


def write_report(df: pd.DataFrame, summary: pd.DataFrame, case_summary: pd.DataFrame, sensitivity: pd.DataFrame) -> None:
    addr = df[df["layer1_addressable"] > 0].copy()
    by_condition = pd.DataFrame(
        [
            {"condition": c, "buckets": int(df[c].sum()), "ratio": float(df[c].mean())}
            for c in [
                "posture_4_5",
                "prefloor",
                "trend_worsening",
                "target_quiet_stale",
                "pump_idle",
                "near_pressure_support",
                "near_miss_no_trend",
                "layer1_addressable",
            ]
        ]
    )
    by_condition.to_csv(RAW / "addressable_condition_funnel.csv", index=False)
    top_addr = addr[
        [
            "dataset",
            "case_id",
            "bucket",
            "current_time_s",
            "current_pitch_deg",
            "current_roll_deg",
            "max_axis_deg",
            "posture_trend_10m_deg",
            "target_age_s",
            "target_error_mean_kg",
            "pump_idle",
            "near_max_norm",
            "far_max_norm",
            "future60_time_over5_s",
            "future60_idle_over5_s",
        ]
    ].sort_values(["dataset", "future60_time_over5_s"], ascending=[True, False])
    top_addr.to_csv(RAW / "addressable_positive_window_examples.csv", index=False)
    lines = [
        "# h120 Addressable Window Audit v1",
        "",
        "Scope: read-only audit over existing v1.6 + f120-oracle baseline traces. No controller changes.",
        "",
        "Layer-1 eligibility measured here:",
        "- `4.0 <= max_axis < 5.0`",
        "- floor not active",
        "- 10min max-axis trend worsening by more than 0.05 deg",
        "- target quiet/stale: target age >= 600s and mean target error <= 500kg",
        "- pump idle",
        "- near pressure support: max 0-60min pressure norm >= 1.0",
        "",
        "Important limitation: existing baseline logs expose near/far pressure norms but not the full axis-wise pressure vector needed for the proposed axis-aware direction-consistency gate. This audit therefore measures window size before the direction-consistency requirement.",
        "",
        "## Dataset Summary",
        markdown_table(summary),
        "",
        "## Condition Funnel",
        markdown_table(by_condition),
        "",
        "## Case Summary",
        markdown_table(case_summary, max_rows=30),
        "",
        "## Threshold Sensitivity: all rows",
        markdown_table(
            sensitivity[sensitivity["dataset"].eq("all")].sort_values(
                ["addressable_buckets", "future60_time_over5_mean_s"], ascending=[False, False]
            ).head(18)
        ),
        "",
        "## Reviewer Interpretation",
    ]
    all_summary = summary[summary["dataset"].eq("all")].iloc[0]
    addr_n = int(all_summary["addressable_buckets"])
    addr_ratio = float(all_summary["addressable_ratio"])
    guard = summary[summary["dataset"].eq("guard10")]
    broader = summary[summary["dataset"].eq("broader20")]
    guard_n = int(guard.iloc[0]["addressable_buckets"]) if not guard.empty else 0
    broader_n = int(broader.iloc[0]["addressable_buckets"]) if not broader.empty else 0
    if addr_n == 0:
        interpretation = (
            "The strict 4-5deg pre-floor eligibility window is absent under the current thresholds. "
            "Before implementing axis-aware micro actions, the interface would need either a wider posture band "
            "or a different state lever."
        )
    elif addr_ratio < 0.05:
        interpretation = (
            "The strict addressable window exists but is small (<5% of buckets). This does not kill the h120 route, "
            "but it sets a tight ROI ceiling: the next oracle run must show visible benefit from a small number of buckets."
        )
    else:
        interpretation = (
            "The strict addressable window is non-trivial. It is large enough to justify the next oracle-first "
            "axis-aware micro ceiling, provided broader/lowrisk windows remain controlled."
        )
    lines.extend(
        [
            interpretation,
            "",
            f"- Strict addressable buckets: {addr_n} / {len(df)} ({addr_ratio:.3%}).",
            f"- Guard10 strict addressable buckets: {guard_n}.",
            f"- Broader20 strict addressable buckets: {broader_n}.",
            "- Next step should be oracle-first axis-aware micro only if this window is accepted as sufficient ROI.",
            "- Do not train learned heads from this audit alone.",
        ]
    )
    text = "\n".join(lines) + "\n"
    (OUT / "h120_addressable_window_report.md").write_text(text)
    (PAPER / "addressable_window_findings.md").write_text(text)


def main() -> None:
    ensure_dirs()
    bucket_table = build_bucket_table()
    flagged = apply_window_flags(bucket_table)
    summary, case_summary = summarize(flagged)
    sensitivity = threshold_sensitivity(bucket_table)
    flagged.to_csv(OUT / "addressable_bucket_table.csv", index=False)
    flagged.to_csv(RAW / "addressable_bucket_table.csv", index=False)
    summary.to_csv(OUT / "addressable_summary.csv", index=False)
    summary.to_csv(RAW / "addressable_summary.csv", index=False)
    case_summary.to_csv(OUT / "addressable_case_table.csv", index=False)
    case_summary.to_csv(RAW / "addressable_case_table.csv", index=False)
    sensitivity.to_csv(OUT / "addressable_threshold_sensitivity.csv", index=False)
    sensitivity.to_csv(RAW / "addressable_threshold_sensitivity.csv", index=False)
    write_report(flagged, summary, case_summary, sensitivity)


if __name__ == "__main__":
    main()
