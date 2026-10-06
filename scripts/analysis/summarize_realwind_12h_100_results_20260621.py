#!/usr/bin/env python3
"""Summarize the 100-window real-wind 12 h continuous validation results."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/wind_prediction/realwind_12h_100_continuous_validation_20260621"
OLD40 = ROOT / (
    "outputs/wind_prediction/positive_filtered_valid50_12h_opportunity_20260608/"
    "combined_40_paired_delta.csv"
)
DECLARED = OUT / "selection_audit_declared100.csv"
BATCHES = ("batch01", "batch02", "batch03")
THRESHOLDS = ("5", "7p5", "10")


def _read(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def _num(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce").fillna(default)
    return pd.Series(default, index=df.index, dtype=float)


def _pair_batch(batch: str) -> pd.DataFrame:
    cur_path = OUT / f"{batch}_current_only" / "casebook_summary.csv"
    learned_path = OUT / f"{batch}_learned_gain045" / "casebook_summary.csv"
    current = _read(cur_path)
    learned = _read(learned_path)

    cols = ["case_id", "timestamp", "label"]
    current = current.copy()
    learned = learned.copy()
    for df in (current, learned):
        df["timestamp"] = pd.to_datetime(df["timestamp"])

    merged = current.merge(
        learned,
        on=cols,
        suffixes=("_current", "_learned"),
        validate="one_to_one",
    )
    out = merged[cols].copy()
    out["planned_batch"] = batch
    out["current_primary_pump_work_m3"] = _num(merged, "primary_pump_work_m3_current")
    out["learned_primary_pump_work_m3"] = _num(merged, "primary_pump_work_m3_learned")
    out["saved_m3"] = (
        out["current_primary_pump_work_m3"] - out["learned_primary_pump_work_m3"]
    )
    out["saving_pct"] = np.where(
        out["current_primary_pump_work_m3"] > 1e-9,
        100.0 * out["saved_m3"] / out["current_primary_pump_work_m3"],
        0.0,
    )

    out["current_primary_safety_fallback_ratio"] = _num(
        merged, "primary_safety_fallback_ratio_current"
    )
    out["learned_primary_safety_fallback_ratio"] = _num(
        merged, "primary_safety_fallback_ratio_learned"
    )
    out["d_fallback_ratio"] = (
        out["learned_primary_safety_fallback_ratio"]
        - out["current_primary_safety_fallback_ratio"]
    )
    out["current_primary_latch_switches"] = _num(
        merged, "primary_latch_switches_current"
    )
    out["learned_primary_latch_switches"] = _num(
        merged, "primary_latch_switches_learned"
    )
    out["current_primary_pitch_p95"] = _num(merged, "primary_pitch_p95_current")
    out["learned_primary_pitch_p95"] = _num(merged, "primary_pitch_p95_learned")
    out["current_primary_roll_p95"] = _num(merged, "primary_roll_p95_current")
    out["learned_primary_roll_p95"] = _num(merged, "primary_roll_p95_learned")

    for thr in THRESHOLDS:
        c = f"primary_time_over_{thr}deg_s_current"
        l = f"primary_time_over_{thr}deg_s_learned"
        out[f"current_primary_time_over_{thr}deg_s"] = _num(merged, c)
        out[f"learned_primary_time_over_{thr}deg_s"] = _num(merged, l)
        out[f"d_t_{thr}deg"] = (
            out[f"learned_primary_time_over_{thr}deg_s"]
            - out[f"current_primary_time_over_{thr}deg_s"]
        )
    return out


def _normalize_old40(old: pd.DataFrame) -> pd.DataFrame:
    old = old.copy()
    old["timestamp"] = pd.to_datetime(old["timestamp"])
    old["planned_batch"] = "existing40"
    keep = [
        "case_id",
        "timestamp",
        "label",
        "planned_batch",
        "current_primary_pump_work_m3",
        "learned_primary_pump_work_m3",
        "saved_m3",
        "saving_pct",
        "current_primary_safety_fallback_ratio",
        "learned_primary_safety_fallback_ratio",
        "d_fallback_ratio",
        "current_primary_latch_switches",
        "learned_primary_latch_switches",
        "current_primary_pitch_p95",
        "learned_primary_pitch_p95",
        "current_primary_roll_p95",
        "learned_primary_roll_p95",
    ]
    for thr in THRESHOLDS:
        keep.extend(
            [
                f"current_primary_time_over_{thr}deg_s",
                f"learned_primary_time_over_{thr}deg_s",
                f"d_t_{thr}deg",
            ]
        )
    for col in keep:
        if col not in old.columns:
            old[col] = np.nan
    return old[keep]


def _aggregate(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    rows = []
    grouped = [((), df)] if not group_cols else df.groupby(group_cols, dropna=False)
    for key, sub in grouped:
        if not isinstance(key, tuple):
            key = (key,)
        row = {c: v for c, v in zip(group_cols, key)}
        cur = float(sub["current_primary_pump_work_m3"].sum())
        learned = float(sub["learned_primary_pump_work_m3"].sum())
        row.update(
            {
                "windows": int(len(sub)),
                "current_pump_m3": cur,
                "learned_pump_m3": learned,
                "saved_m3": cur - learned,
                "saving_pct": 100.0 * (cur - learned) / cur if cur > 1e-9 else 0.0,
                "positive_saving_windows": int((sub["saved_m3"] > 0).sum()),
                "current_latch_switches": float(
                    sub["current_primary_latch_switches"].sum()
                ),
                "learned_latch_switches": float(
                    sub["learned_primary_latch_switches"].sum()
                ),
            }
        )
        for thr in THRESHOLDS:
            dcol = f"d_t_{thr}deg"
            ccol = f"current_primary_time_over_{thr}deg_s"
            lcol = f"learned_primary_time_over_{thr}deg_s"
            row[f"current_theta_d_gt_{thr}_s"] = float(sub[ccol].sum())
            row[f"learned_theta_d_gt_{thr}_s"] = float(sub[lcol].sum())
            row[f"d_theta_d_gt_{thr}_s"] = float(sub[dcol].sum())
            row[f"positive_d_theta_d_gt_{thr}_windows"] = int((sub[dcol] > 0).sum())
        rows.append(row)
    return pd.DataFrame(rows).round(6)


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
        for c in df.columns:
            v = row[c]
            if isinstance(v, float):
                vals.append(f"{v:.2f}")
            else:
                vals.append(str(v))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def main() -> None:
    declared = _read(DECLARED)
    declared["timestamp"] = pd.to_datetime(declared["timestamp"])
    declared_lookup = declared[
        ["case_id", "source_case_id", "selector_stratum", "set_role", "planned_batch"]
    ].copy()

    old40 = _normalize_old40(_read(OLD40))
    batches = []
    for batch in BATCHES:
        paired = _pair_batch(batch)
        paired.to_csv(OUT / f"{batch}_paired_delta.csv", index=False)
        batches.append(paired)

    new60 = pd.concat(batches, ignore_index=True)
    new60.to_csv(OUT / "add60_paired_delta.csv", index=False)

    all100 = pd.concat([old40, new60], ignore_index=True)
    all100 = all100.merge(declared_lookup, on=["case_id", "planned_batch"], how="left")
    all100.to_csv(OUT / "declared100_paired_delta.csv", index=False)

    total = _aggregate(all100, [])
    by_role = _aggregate(all100, ["set_role"])
    by_batch = _aggregate(all100, ["planned_batch"])
    by_stratum = _aggregate(all100, ["selector_stratum"])

    total.to_csv(OUT / "declared100_total.csv", index=False)
    by_role.to_csv(OUT / "declared100_by_role.csv", index=False)
    by_batch.to_csv(OUT / "declared100_by_batch.csv", index=False)
    by_stratum.to_csv(OUT / "declared100_by_stratum.csv", index=False)

    compact_cols = [
        "windows",
        "current_pump_m3",
        "learned_pump_m3",
        "saved_m3",
        "saving_pct",
        "positive_saving_windows",
        "d_theta_d_gt_5_s",
        "d_theta_d_gt_7p5_s",
        "d_theta_d_gt_10_s",
    ]
    readout = [
        "# 100-Window 12 h Real-Wind Continuous Validation Summary",
        "",
        "## Total",
        "",
        _markdown_table(total[compact_cols]),
        "",
        "## By Set Role",
        "",
        _markdown_table(by_role[["set_role", *compact_cols]]),
        "",
        "## By Wind-Condition Stratum",
        "",
        _markdown_table(by_stratum[["selector_stratum", *compact_cols]]),
        "",
        "## Files",
        "",
        "- `declared100_paired_delta.csv`",
        "- `declared100_total.csv`",
        "- `declared100_by_role.csv`",
        "- `declared100_by_stratum.csv`",
    ]
    (OUT / "declared100_readout.md").write_text("\n".join(readout) + "\n", encoding="utf-8")
    print((OUT / "declared100_readout.md").relative_to(ROOT))
    print(total[compact_cols].to_string(index=False))


if __name__ == "__main__":
    main()
