#!/usr/bin/env python3
"""Combine old selector-positive cases with the add40 positive extension."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_OUT_DIR = Path(
    "outputs/wind_prediction/selector_positive_add40_6h_v1/"
    "combined_positive_summary_170case_6h"
)
OLD_SUMMARY = Path(
    "outputs/wind_prediction/selector_mixed_6h_limit20_v1/"
    "blind_d1_engineered_101case_6h/casebook_summary.csv"
)
NEW_SUMMARY = Path(
    "outputs/wind_prediction/selector_positive_add40_6h_v1/"
    "blind_d1_engineered_add40_120case_6h/casebook_summary.csv"
)

POSITIVE_STRATA = {
    "p2_neutral_headroom",
    "c3_gusty_oscillatory",
    "w1_stable_direction_event",
}


def _extract_label_field(label: str, key: str, default: str = "") -> str:
    marker = f"{key}="
    for part in str(label).split("|"):
        text = part.strip()
        if text.startswith(marker):
            return text[len(marker) :].strip()
    return default


def _num(row: pd.Series, col: str) -> float:
    return float(pd.to_numeric(row.get(col, np.nan), errors="coerce"))


def _safe_sum(df: pd.DataFrame, col: str) -> float:
    return float(df[col].fillna(0.0).sum()) if col in df.columns else 0.0


def _load_positive(path: Path, batch: str) -> pd.DataFrame:
    df = pd.read_csv(path).copy()
    df["selector_stratum"] = df["label"].map(
        lambda x: _extract_label_field(x, "selector_stratum", "unknown")
    )
    df["validation_role"] = df["label"].map(
        lambda x: _extract_label_field(x, "validation_role", "unknown")
    )
    df["gate_decision"] = df["label"].map(
        lambda x: _extract_label_field(x, "gate_decision", "abstain_deadband")
    )
    df["run_batch"] = batch
    keep = (
        df["validation_role"].eq("positive_allow")
        & df["gate_decision"].eq("allow_deadband")
        & df["selector_stratum"].isin(POSITIVE_STRATA)
    )
    return df.loc[keep].copy()


def _per_case(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in df.iterrows():
        closed = _num(row, "closed_pump_work_m3")
        primary = _num(row, "primary_pump_work_m3")
        record: dict[str, float | str] = {
            "run_batch": str(row["run_batch"]),
            "case_id": str(row["case_id"]),
            "timestamp": str(row["timestamp"]),
            "selector_stratum": str(row["selector_stratum"]),
            "validation_role": str(row["validation_role"]),
            "gate_decision": str(row["gate_decision"]),
            "closed_pump_m3": closed,
            "primary_pump_m3": primary,
            "pump_saved_m3": closed - primary,
            "pump_saving_pct": 100.0 * (closed - primary) / closed if closed else 0.0,
            "d_pitch_p95_deg": _num(row, "d_pitch_p95"),
            "d_roll_p95_deg": _num(row, "d_roll_p95"),
            "primary_safety_fallback_ratio": _num(row, "primary_safety_fallback_ratio"),
            "closed_latch_switches": _num(row, "closed_latch_switches"),
            "primary_latch_switches": _num(row, "primary_latch_switches"),
        }
        record["d_attitude_p95_deg"] = max(
            float(record["d_pitch_p95_deg"]),
            float(record["d_roll_p95_deg"]),
        )
        for thr, col_thr in [
            ("5", "5"),
            ("7p5", "7p5"),
            ("10", "10"),
        ]:
            closed_col = f"closed_time_over_{col_thr}deg_s"
            primary_col = f"primary_time_over_{col_thr}deg_s"
            delta_col = f"d_time_over_{col_thr}deg_s"
            max_cont_col = f"primary_max_continuous_over_{col_thr}deg_s"
            delta_max_cont_col = f"d_max_continuous_over_{col_thr}deg_s"
            delta = _num(row, delta_col)
            record[f"closed_time_over_{thr}_s"] = _num(row, closed_col)
            record[f"primary_time_over_{thr}_s"] = _num(row, primary_col)
            record[f"d_time_over_{thr}_s"] = delta
            record[f"d_time_over_{thr}_positive_s"] = max(delta, 0.0)
            record[f"primary_max_continuous_over_{thr}_s"] = _num(row, max_cont_col)
            record[f"d_max_continuous_over_{thr}_s"] = _num(row, delta_max_cont_col)
        rows.append(record)
    return pd.DataFrame(rows)


def _aggregate(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    grouped = [(("all",), df)] if not group_cols else df.groupby(group_cols, dropna=False)
    rows = []
    for key, sub in grouped:
        if not isinstance(key, tuple):
            key = (key,)
        closed = _safe_sum(sub, "closed_pump_m3")
        primary = _safe_sum(sub, "primary_pump_m3")
        row: dict[str, float | int | str] = {
            col: val for col, val in zip(group_cols or ["scope"], key)
        }
        row.update(
            {
                "cases": int(len(sub)),
                "closed_pump_m3": closed,
                "primary_pump_m3": primary,
                "pump_saved_m3": closed - primary,
                "pump_saving_pct": 100.0 * (closed - primary) / closed if closed else 0.0,
                "mean_d_attitude_p95_deg": float(sub["d_attitude_p95_deg"].mean()),
                "max_d_attitude_p95_deg": float(sub["d_attitude_p95_deg"].max()),
                "mean_d_pitch_p95_deg": float(sub["d_pitch_p95_deg"].mean()),
                "mean_d_roll_p95_deg": float(sub["d_roll_p95_deg"].mean()),
                "max_d_pitch_p95_deg": float(sub["d_pitch_p95_deg"].max()),
                "max_d_roll_p95_deg": float(sub["d_roll_p95_deg"].max()),
                "fallback_case_count": int((sub["primary_safety_fallback_ratio"] > 0).sum()),
                "closed_latch_switches": int(round(_safe_sum(sub, "closed_latch_switches"))),
                "primary_latch_switches": int(round(_safe_sum(sub, "primary_latch_switches"))),
            }
        )
        closed_switches = float(row["closed_latch_switches"])
        primary_switches = float(row["primary_latch_switches"])
        row["latch_switch_reduction_pct"] = (
            100.0 * (closed_switches - primary_switches) / closed_switches
            if closed_switches
            else 0.0
        )
        for thr in ["5", "7p5", "10"]:
            row[f"closed_time_over_{thr}_s"] = _safe_sum(
                sub, f"closed_time_over_{thr}_s"
            )
            row[f"primary_time_over_{thr}_s"] = _safe_sum(
                sub, f"primary_time_over_{thr}_s"
            )
            row[f"d_time_over_{thr}_s"] = _safe_sum(sub, f"d_time_over_{thr}_s")
            row[f"d_time_over_{thr}_positive_s"] = _safe_sum(
                sub, f"d_time_over_{thr}_positive_s"
            )
            row[f"d_time_over_{thr}_positive_case_count"] = int(
                (sub[f"d_time_over_{thr}_s"] > 0).sum()
            )
            row[f"max_primary_continuous_over_{thr}_s"] = float(
                sub[f"primary_max_continuous_over_{thr}_s"].max()
            )
            row[f"max_d_continuous_over_{thr}_s"] = float(
                sub[f"d_max_continuous_over_{thr}_s"].max()
            )
        rows.append(row)
    return pd.DataFrame(rows)


def _format_value(value: object) -> str:
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.3f}"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    return str(value)


def _markdown_table(df: pd.DataFrame, columns: list[str] | None = None) -> str:
    if columns is not None:
        df = df[columns]
    if df.empty:
        return "_empty_"
    cols = list(df.columns)
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join("---" for _ in cols) + " |",
    ]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(_format_value(row[col]) for col in cols) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-summary", type=Path, default=OLD_SUMMARY)
    parser.add_argument("--new-summary", type=Path, default=NEW_SUMMARY)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    old = _load_positive(args.old_summary, "existing_positive")
    new = _load_positive(args.new_summary, "add40_positive")
    combined_raw = pd.concat([old, new], ignore_index=True)
    per_case = _per_case(combined_raw)

    by_batch_stratum = _aggregate(per_case, ["run_batch", "selector_stratum"])
    by_stratum = _aggregate(per_case, ["selector_stratum"])
    by_batch = _aggregate(per_case, ["run_batch"])
    total = _aggregate(per_case, [])

    args.out_dir.mkdir(parents=True, exist_ok=True)
    per_case.to_csv(args.out_dir / "positive_combined_per_case.csv", index=False)
    by_batch_stratum.to_csv(
        args.out_dir / "positive_combined_by_batch_stratum.csv", index=False
    )
    by_stratum.to_csv(args.out_dir / "positive_combined_by_stratum.csv", index=False)
    by_batch.to_csv(args.out_dir / "positive_combined_by_batch.csv", index=False)
    total.to_csv(args.out_dir / "positive_combined_total.csv", index=False)

    compact_cols = [
        "selector_stratum",
        "cases",
        "pump_saving_pct",
        "pump_saved_m3",
        "latch_switch_reduction_pct",
        "mean_d_attitude_p95_deg",
        "max_d_attitude_p95_deg",
        "d_time_over_5_s",
        "d_time_over_5_positive_s",
        "d_time_over_7p5_s",
        "d_time_over_7p5_positive_s",
        "d_time_over_10_s",
        "d_time_over_10_positive_s",
        "d_time_over_10_positive_case_count",
        "max_d_pitch_p95_deg",
        "max_d_roll_p95_deg",
        "fallback_case_count",
    ]
    batch_cols = ["run_batch"] + compact_cols
    total_cols = [
        "scope",
        "cases",
        "pump_saving_pct",
        "pump_saved_m3",
        "latch_switch_reduction_pct",
        "mean_d_attitude_p95_deg",
        "max_d_attitude_p95_deg",
        "d_time_over_5_s",
        "d_time_over_5_positive_s",
        "d_time_over_7p5_s",
        "d_time_over_7p5_positive_s",
        "d_time_over_10_s",
        "d_time_over_10_positive_s",
        "d_time_over_10_positive_case_count",
        "fallback_case_count",
    ]

    lines = [
        "# Positive Add40 Combined Readout",
        "",
        f"- old source: `{args.old_summary}`",
        f"- new source: `{args.new_summary}`",
        f"- old positive cases included: `{len(old)}`",
        f"- new positive cases included: `{len(new)}`",
        f"- combined positive cases: `{len(per_case)}`",
        "",
        "## Total",
        "",
        _markdown_table(total, total_cols),
        "",
        "## Combined By Stratum",
        "",
        _markdown_table(by_stratum.sort_values("selector_stratum"), compact_cols),
        "",
        "## Existing Vs Add40",
        "",
        _markdown_table(
            by_batch_stratum.sort_values(["selector_stratum", "run_batch"]),
            batch_cols,
        ),
        "",
        "Notes: `d_time_over_*_s` is the net primary-minus-closed exposure delta. "
        "`d_time_over_*_positive_s` clips negative deltas to zero before summing, "
        "so it isolates cases where the policy added exposure.",
    ]
    (args.out_dir / "positive_combined_readout.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )

    print(total.to_string(index=False))
    print(by_stratum.sort_values("selector_stratum").to_string(index=False))
    print(args.out_dir / "positive_combined_readout.md")


if __name__ == "__main__":
    main()
