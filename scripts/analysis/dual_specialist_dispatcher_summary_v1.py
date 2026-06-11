#!/usr/bin/env python3
"""Summarize the dual-specialist dispatcher mixed-case validation."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


def _markdown_table(df: pd.DataFrame) -> str:
    view = df.copy()
    for col in view.columns:
        if pd.api.types.is_float_dtype(view[col]):
            view[col] = view[col].map(lambda x: f"{x:.3f}")
        else:
            view[col] = view[col].astype(str)
    header = "| " + " | ".join(view.columns) + " |"
    sep = "| " + " | ".join(["---"] * len(view.columns)) + " |"
    rows = ["| " + " | ".join(row) + " |" for row in view.astype(str).itertuples(index=False, name=None)]
    return "\n".join([header, sep, *rows])


def _case_key(case_id: object) -> str:
    text = re.sub(r"^\d+_", "", str(case_id))
    match = re.match(r"(dual_(?:relief|neutral|boundary)_\d+)", text)
    return match.group(1) if match else text


def _max_axis_p95(row: pd.Series, prefix: str = "") -> float:
    pitch = float(row.get(f"{prefix}primary_pitch_p95", 0.0) or 0.0)
    roll = float(row.get(f"{prefix}primary_roll_p95", 0.0) or 0.0)
    return max(abs(pitch), abs(roll))


def _case_stem(path: Path, label_suffix: str) -> str:
    name = path.name
    suffix = f"_{label_suffix}_planner_log.csv"
    if name.endswith(suffix):
        return name[: -len(suffix)]
    return name.replace("_planner_log.csv", "")


def _read_log_metrics(log_dir: Path, label_suffix: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for path in sorted(log_dir.glob("*_planner_log.csv")):
        case_id = _case_key(_case_stem(path, label_suffix))
        df = pd.read_csv(path)
        reason = df.get("economy_pump_budget_reason")
        reason_s = reason.astype(str) if reason is not None else pd.Series([], dtype=str)
        active = pd.to_numeric(df.get("economy_pump_budget_active", 0), errors="coerce").fillna(0)
        rows.append(
            {
                "case_id": case_id,
                "budget_active_rows": int((active > 0).sum()),
                "relief_hold_rows": int(reason_s.str.contains("dual_specialist_hold:relief_decay", na=False).sum()),
                "neutral_hold_rows": int(reason_s.str.contains("dual_specialist_hold:neutral_mhs", na=False).sum()),
                "release_rows": int(reason_s.str.contains("release", na=False).sum()),
                "fallback_release_rows": int(reason_s.str.contains("fallback", na=False).sum()),
                "floor_release_rows": int(reason_s.str.contains("floor", na=False).sum()),
                "reversal_veto_rows": int(reason_s.str.contains("reversal|direction|signflip", na=False).sum()),
                "top_reason": reason_s.value_counts().index[0] if len(reason_s) else "",
            }
        )
    return pd.DataFrame(rows)


def _read_timeseries_metrics(ts_dir: Path, label_suffix: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    suffix = f"_{label_suffix}_timeseries.csv"
    for path in sorted(ts_dir.glob("*_timeseries.csv")):
        name = path.name
        case_id = _case_key(name[: -len(suffix)] if name.endswith(suffix) else name.replace("_timeseries.csv", ""))
        df = pd.read_csv(path, usecols=lambda c: c in {"pitch_deg", "roll_deg", "preview_primary_safety_fallback"})
        pitch = pd.to_numeric(df.get("pitch_deg", 0), errors="coerce").fillna(0).abs()
        roll = pd.to_numeric(df.get("roll_deg", 0), errors="coerce").fillna(0).abs()
        fallback = pd.to_numeric(df.get("preview_primary_safety_fallback", 0), errors="coerce").fillna(0)
        max_axis = pd.concat([pitch, roll], axis=1).max(axis=1)
        rows.append(
            {
                "case_id": case_id,
                "time_gt5_s": int((max_axis > 5.0).sum()),
                "fallback_s": int((fallback > 0).sum()),
                "max_axis_deg": float(max_axis.max()),
                "p95_axis_deg": float(max_axis.quantile(0.95)),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("outputs/wind_prediction/regime_conditioned_policy_development_v1/dual_specialist_dispatcher_v1"),
    )
    args = parser.parse_args()

    root = args.root
    raw_dir = root / "raw_tables"
    paper_dir = root / "paper_ready"
    raw_dir.mkdir(parents=True, exist_ok=True)
    paper_dir.mkdir(parents=True, exist_ok=True)

    cases = pd.read_csv(root / "raw_tables" / "dual_specialist_mixed_case_table.csv")
    a0 = pd.read_csv(root / "a0_1h" / "casebook_summary.csv")
    dual = pd.read_csv(root / "dual_specialist_learned_1h" / "casebook_summary.csv")
    cases["case_id"] = cases["case_id"].map(_case_key)
    a0["case_id"] = a0["case_id"].map(_case_key)
    dual["case_id"] = dual["case_id"].map(_case_key)

    a0_ts = _read_timeseries_metrics(root / "a0_1h" / "timeseries", "a0_v16")
    dual_ts = _read_timeseries_metrics(root / "dual_specialist_learned_1h" / "timeseries", "dual_specialist_learned")
    dual_log = _read_log_metrics(root / "dual_specialist_learned_1h" / "planner_logs", "dual_specialist_learned")

    keep = [
        "case_id",
        "primary_pump_work_m3",
        "primary_pitch_p95",
        "primary_roll_p95",
        "primary_safety_fallback_ratio",
        "primary_latch_switches",
    ]
    a0 = a0[keep].rename(columns={c: f"a0_{c}" for c in keep if c != "case_id"})
    dual = dual[keep].rename(columns={c: f"dual_{c}" for c in keep if c != "case_id"})
    a0_ts = a0_ts.rename(columns={c: f"a0_{c}" for c in a0_ts.columns if c != "case_id"})
    dual_ts = dual_ts.rename(columns={c: f"dual_{c}" for c in dual_ts.columns if c != "case_id"})

    df = cases.merge(a0, on="case_id", how="left").merge(dual, on="case_id", how="left")
    df = df.merge(a0_ts, on="case_id", how="left").merge(dual_ts, on="case_id", how="left").merge(dual_log, on="case_id", how="left")
    df["pump_saved_m3"] = df["a0_primary_pump_work_m3"] - df["dual_primary_pump_work_m3"]
    df["pump_saving_pct"] = 100.0 * df["pump_saved_m3"] / df["a0_primary_pump_work_m3"].where(df["a0_primary_pump_work_m3"] != 0)
    df["delta_time_gt5_s"] = df["dual_time_gt5_s"] - df["a0_time_gt5_s"]
    df["delta_fallback_s"] = df["dual_fallback_s"] - df["a0_fallback_s"]
    df["delta_p95_axis_deg"] = df["dual_p95_axis_deg"] - df["a0_p95_axis_deg"]
    df["delta_max_axis_deg"] = df["dual_max_axis_deg"] - df["a0_max_axis_deg"]

    def summarize(group: pd.DataFrame, name: str) -> dict[str, object]:
        a0_pump = float(group["a0_primary_pump_work_m3"].sum())
        dual_pump = float(group["dual_primary_pump_work_m3"].sum())
        saved = a0_pump - dual_pump
        pos_saved = group.loc[group["pump_saved_m3"] > 0, "pump_saved_m3"].sum()
        top_share = float(group["pump_saved_m3"].max() / pos_saved) if pos_saved > 0 else 0.0
        return {
            "regime_group": name,
            "cases": int(len(group)),
            "a0_pump_m3": a0_pump,
            "dual_pump_m3": dual_pump,
            "pump_saved_m3": saved,
            "pump_saving_pct": 100.0 * saved / a0_pump if a0_pump else 0.0,
            "active_cases": int((group["budget_active_rows"].fillna(0) > 0).sum()),
            "relief_active_cases": int((group["relief_hold_rows"].fillna(0) > 0).sum()),
            "neutral_active_cases": int((group["neutral_hold_rows"].fillna(0) > 0).sum()),
            "delta_time_gt5_s": int(group["delta_time_gt5_s"].sum()),
            "delta_fallback_s": int(group["delta_fallback_s"].sum()),
            "max_delta_p95_axis_deg": float(group["delta_p95_axis_deg"].max()),
            "mean_delta_p95_axis_deg": float(group["delta_p95_axis_deg"].mean()),
            "max_delta_max_axis_deg": float(group["delta_max_axis_deg"].max()),
            "top_positive_case_share": top_share,
        }

    summary_rows = [summarize(df, "ALL")]
    for name, group in df.groupby("regime_group", sort=True):
        summary_rows.append(summarize(group, str(name)))
    summary = pd.DataFrame(summary_rows)

    df.to_csv(raw_dir / "dual_specialist_learned_delta.csv", index=False)
    summary.to_csv(raw_dir / "dual_specialist_learned_summary.csv", index=False)

    decision = [
        "# Dual Specialist Dispatcher Mixed-Set Validation",
        "",
        "This run compares v1.6 A0 against `dual_specialist_pump_saving_v1` on a 96-case mixed casebook:",
        "24 transient-peak/future-decay cases, 64 neutral moderate-high steady cases, and 8 direction-reversal boundary cases.",
        "",
        "## Summary Table",
        "",
        _markdown_table(summary),
        "",
        "## Interpretation",
        "",
    ]
    all_row = summary.iloc[0]
    boundary = summary[summary["regime_group"] == "direction_reversal_boundary"]
    if not boundary.empty and float(boundary.iloc[0]["pump_saved_m3"]) < 0:
        decision.append(
            "The combined dispatcher is **not yet clean enough to freeze as a global automatic policy**: "
            "the boundary group increases pump use, which means one of the specialists still leaks into catch-up/reversal-like conditions."
        )
    else:
        decision.append(
            "The boundary group does not show aggregate pump harm in this run, so the combined dispatcher is a viable automatic-policy candidate subject to the detailed safety deltas."
        )
    decision.extend(
        [
            "",
            "Keep the two individual specialists as the current sources of truth. Treat this mixed dispatcher run as an integration test: "
            "it is useful because it reveals whether the automatic dispatcher abstains outside its intended regimes.",
        ]
    )
    (root / "dual_specialist_learned_decision.md").write_text("\n".join(decision) + "\n")
    (paper_dir / "dual_specialist_mixed_validation_summary.md").write_text("\n".join(decision) + "\n")


if __name__ == "__main__":
    main()
