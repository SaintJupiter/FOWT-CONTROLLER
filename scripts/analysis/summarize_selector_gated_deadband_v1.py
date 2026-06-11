#!/usr/bin/env python3
"""Summarize closed, blind-deadband, and selector-gated deadband metrics."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def _extract_label_field(label: str, key: str, default: str = "") -> str:
    marker = f"{key}="
    for part in str(label).split("|"):
        text = part.strip()
        if text.startswith(marker):
            return text[len(marker) :].strip()
    return default


def _safe_sum(df: pd.DataFrame, col: str) -> float:
    return float(df[col].fillna(0.0).sum()) if col in df.columns else 0.0


def _row_metrics(row: pd.Series) -> dict[str, float | str]:
    gate = str(row["gate_decision"])
    use_deadband = gate == "allow_deadband"
    closed_pump = float(row["closed_pump_work_m3"])
    blind_pump = float(row["primary_pump_work_m3"])
    gated_pump = blind_pump if use_deadband else closed_pump

    def gated_delta(col: str) -> float:
        return float(row[col]) if use_deadband and col in row.index else 0.0

    def gated_primary(col: str, closed_col: str) -> float:
        return float(row[col]) if use_deadband and col in row.index else float(row[closed_col])

    return {
        "case_id": str(row["case_id"]),
        "timestamp": str(row.get("timestamp", "")),
        "selector_stratum": str(row["selector_stratum"]),
        "validation_role": str(row["validation_role"]),
        "gate_decision": gate,
        "closed_pump_m3": closed_pump,
        "blind_pump_m3": blind_pump,
        "gated_pump_m3": gated_pump,
        "blind_saving_pct": 100.0 * (closed_pump - blind_pump) / closed_pump if closed_pump else 0.0,
        "gated_saving_pct": 100.0 * (closed_pump - gated_pump) / closed_pump if closed_pump else 0.0,
        "blind_d_pitch_p95_deg": float(row.get("d_pitch_p95", 0.0)),
        "blind_d_roll_p95_deg": float(row.get("d_roll_p95", 0.0)),
        "gated_d_pitch_p95_deg": gated_delta("d_pitch_p95"),
        "gated_d_roll_p95_deg": gated_delta("d_roll_p95"),
        "closed_time_over_5_s": float(row.get("closed_time_over_5deg_s", 0.0)),
        "blind_time_over_5_s": float(row.get("primary_time_over_5deg_s", 0.0)),
        "gated_time_over_5_s": gated_primary("primary_time_over_5deg_s", "closed_time_over_5deg_s"),
        "blind_d_time_over_5_s": float(row.get("d_time_over_5deg_s", 0.0)),
        "gated_d_time_over_5_s": gated_delta("d_time_over_5deg_s"),
        "closed_time_over_7p5_s": float(row.get("closed_time_over_7p5deg_s", 0.0)),
        "blind_time_over_7p5_s": float(row.get("primary_time_over_7p5deg_s", 0.0)),
        "gated_time_over_7p5_s": gated_primary("primary_time_over_7p5deg_s", "closed_time_over_7p5deg_s"),
        "blind_d_time_over_7p5_s": float(row.get("d_time_over_7p5deg_s", 0.0)),
        "gated_d_time_over_7p5_s": gated_delta("d_time_over_7p5deg_s"),
        "blind_fallback_ratio": float(row.get("primary_safety_fallback_ratio", 0.0)),
        "gated_fallback_ratio": float(row.get("primary_safety_fallback_ratio", 0.0)) if use_deadband else 0.0,
        "closed_latch_switches": float(row.get("closed_latch_switches", 0.0)),
        "blind_latch_switches": float(row.get("primary_latch_switches", 0.0)),
        "gated_latch_switches": float(row.get("primary_latch_switches", 0.0))
        if use_deadband
        else float(row.get("closed_latch_switches", 0.0)),
    }


def _aggregate(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    rows = []
    grouped = [(("all",), df)] if not group_cols else df.groupby(group_cols, dropna=False)
    for key, sub in grouped:
        if not isinstance(key, tuple):
            key = (key,)
        closed = _safe_sum(sub, "closed_pump_m3")
        blind = _safe_sum(sub, "blind_pump_m3")
        gated = _safe_sum(sub, "gated_pump_m3")
        row = {col: val for col, val in zip(group_cols or ["scope"], key)}
        row.update(
            {
                "cases": int(len(sub)),
                "closed_pump_m3": closed,
                "blind_pump_m3": blind,
                "gated_pump_m3": gated,
                "blind_saving_pct": 100.0 * (closed - blind) / closed if closed else 0.0,
                "gated_saving_pct": 100.0 * (closed - gated) / closed if closed else 0.0,
                "blind_d_time_over_5_s": _safe_sum(sub, "blind_d_time_over_5_s"),
                "gated_d_time_over_5_s": _safe_sum(sub, "gated_d_time_over_5_s"),
                "blind_d_time_over_7p5_s": _safe_sum(sub, "blind_d_time_over_7p5_s"),
                "gated_d_time_over_7p5_s": _safe_sum(sub, "gated_d_time_over_7p5_s"),
                "blind_max_d_pitch_p95_deg": float(sub["blind_d_pitch_p95_deg"].max()),
                "blind_max_d_roll_p95_deg": float(sub["blind_d_roll_p95_deg"].max()),
                "gated_max_d_pitch_p95_deg": float(sub["gated_d_pitch_p95_deg"].max()),
                "gated_max_d_roll_p95_deg": float(sub["gated_d_roll_p95_deg"].max()),
                "blind_fallback_case_count": int((sub["blind_fallback_ratio"] > 0).sum()),
                "gated_fallback_case_count": int((sub["gated_fallback_ratio"] > 0).sum()),
                "closed_latch_switches": int(round(_safe_sum(sub, "closed_latch_switches"))),
                "blind_latch_switches": int(round(_safe_sum(sub, "blind_latch_switches"))),
                "gated_latch_switches": int(round(_safe_sum(sub, "gated_latch_switches"))),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _policy_gate(row: pd.Series, policy: str) -> str:
    role = str(row["validation_role"])
    stratum = str(row["selector_stratum"])
    if policy == "closed_all":
        return "abstain_deadband"
    if policy == "blind_all":
        return "allow_deadband"
    if policy == "strict_positive_only":
        return "allow_deadband" if role == "positive_allow" else "abstain_deadband"
    if policy == "allow_reintensification_probe":
        if role == "positive_allow" or stratum == "reintensification_boundary":
            return "allow_deadband"
        return "abstain_deadband"
    raise ValueError(f"unknown policy: {policy}")


def _policy_sweep(raw: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for policy in [
        "closed_all",
        "blind_all",
        "strict_positive_only",
        "allow_reintensification_probe",
    ]:
        work = raw.copy()
        work["gate_decision"] = work.apply(lambda row: _policy_gate(row, policy), axis=1)
        per_case = pd.DataFrame([_row_metrics(row) for _, row in work.iterrows()])
        agg = _aggregate(per_case, []).iloc[0].to_dict()
        agg["policy"] = policy
        agg["allow_cases"] = int((per_case["gate_decision"] == "allow_deadband").sum())
        agg["abstain_cases"] = int((per_case["gate_decision"] == "abstain_deadband").sum())
        rows.append(agg)
    out = pd.DataFrame(rows)
    cols = ["policy", "allow_cases", "abstain_cases"] + [
        col for col in out.columns if col not in {"policy", "allow_cases", "abstain_cases"}
    ]
    return out[cols]


def _format_value(value: object) -> str:
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.3f}"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    return str(value)


def _markdown_table(df: pd.DataFrame) -> str:
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
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args()

    summary_path = args.run_dir / "casebook_summary.csv"
    raw = pd.read_csv(summary_path).copy()
    raw["selector_stratum"] = raw["label"].map(lambda x: _extract_label_field(x, "selector_stratum", "unknown"))
    raw["validation_role"] = raw["label"].map(lambda x: _extract_label_field(x, "validation_role", "unknown"))
    raw["gate_decision"] = raw["label"].map(lambda x: _extract_label_field(x, "gate_decision", "abstain_deadband"))

    per_case = pd.DataFrame([_row_metrics(row) for _, row in raw.iterrows()])
    by_role = _aggregate(per_case, ["validation_role"])
    by_stratum = _aggregate(per_case, ["validation_role", "selector_stratum", "gate_decision"])
    total = _aggregate(per_case, [])
    policy_sweep = _policy_sweep(raw)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    per_case.to_csv(args.out_dir / "selector_gated_per_case.csv", index=False)
    by_role.to_csv(args.out_dir / "selector_gated_by_role.csv", index=False)
    by_stratum.to_csv(args.out_dir / "selector_gated_by_stratum.csv", index=False)
    total.to_csv(args.out_dir / "selector_gated_total.csv", index=False)
    policy_sweep.to_csv(args.out_dir / "selector_policy_sweep.csv", index=False)

    lines = [
        "# Selector-Gated Deadband Summary",
        "",
        f"- source run: `{args.run_dir}`",
        f"- cases: `{len(per_case)}`",
        "",
        "## Total",
        "",
        _markdown_table(total),
        "",
        "## By Role",
        "",
        _markdown_table(by_role),
        "",
        "## By Stratum",
        "",
        _markdown_table(by_stratum),
        "",
        "## Policy Sweep",
        "",
        _markdown_table(policy_sweep),
        "",
        "Interpretation: `blind` applies D1 everywhere. `gated` applies D1 only where "
        "`gate_decision=allow_deadband`; all abstain rows fall back to closed-only.",
    ]
    (args.out_dir / "selector_gated_summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(total.to_string(index=False))
    print(args.out_dir / "selector_gated_summary.md")


if __name__ == "__main__":
    main()
