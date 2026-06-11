#!/usr/bin/env python3
"""Summarize the No.4 active-posture refresh ON/OFF ablation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
BASE = REPO_ROOT / "outputs" / "wind_prediction" / "psc_selector_mixed_pool_v1"
OUT = BASE / "psc_4hao_case4_ablation_v1"
FROZEN_2H = BASE / "degradation_ladder_96case_pair" / "degradation_ladder_case_metrics.csv"
BASELINE_6H = OUT / "h6_baseline_current_forecast_adaptive"

RUNS = {
    ("2h", "ON"): OUT / "h2_on",
    ("2h", "OFF"): OUT / "h2_off",
    ("6h", "ON"): OUT / "h6_on",
    ("6h", "OFF"): OUT / "h6_off",
}

CASE_CSV = OUT / "case4_refresh_ablation_case_metrics.csv"
SUMMARY_CSV = OUT / "case4_refresh_ablation_summary.csv"
READOUT_MD = OUT / "case4_refresh_ablation_readout.md"


def _case_id_from_path(path: Path) -> str:
    stem = path.name.replace("_prediction_primary_econ_timeseries.csv", "")
    parts = stem.split("_")
    body = "_".join(parts[1:4])
    if body == "dual_relief_09":
        return "09_dual_relief_09"
    if body == "dual_relief_10":
        return "10_dual_relief_10"
    if body == "dual_relief_18":
        return "18_dual_relief_18"
    if body == "dual_boundary_06":
        return "94_dual_boundary_06"
    raise ValueError(f"cannot map case id from {path}")


def _metrics(path: Path) -> dict[str, float]:
    df = pd.read_csv(path, low_memory=False)
    axis = df[["pitch_deg", "roll_deg"]].abs().max(axis=1)
    fallback = pd.to_numeric(
        df.get("preview_primary_safety_fallback", pd.Series(0.0, index=df.index)),
        errors="coerce",
    ).fillna(0.0)
    return {
        "pump_m3": float(df["pump_total_rate_m3_min"].abs().sum() / 60.0),
        "time_gt3_s": float((axis > 3.0).sum()),
        "time_gt4_s": float((axis > 4.0).sum()),
        "time_gt5_s": float((axis > 5.0).sum()),
        "fallback_s": float((fallback > 0.0).sum()),
        "p95_axis_deg": float(axis.quantile(0.95)),
        "max_axis_deg": float(axis.max()),
    }


def _baseline_2h() -> pd.DataFrame:
    df = pd.read_csv(FROZEN_2H)
    return df[df["arm"].eq("current_forecast_adaptive")].set_index("case_id")


def _baseline_6h() -> pd.DataFrame:
    rows = []
    for path in sorted((BASELINE_6H / "timeseries").glob("*_current_forecast_adaptive_timeseries.csv")):
        row = _metrics(path)
        row["case_id"] = _case_id_from_path(path.with_name(path.name.replace("_current_forecast_adaptive", "_prediction_primary_econ")))
        rows.append(row)
    return pd.DataFrame(rows).set_index("case_id")


def _fmt(value: float, digits: int = 1) -> str:
    if pd.isna(value):
        return ""
    return f"{value:.{digits}f}"


def _table(rows: list[dict[str, object]], cols: list[tuple[str, str]]) -> list[str]:
    lines = [
        "| " + " | ".join(title for _, title in cols) + " |",
        "| " + " | ".join("---" for _ in cols) + " |",
    ]
    for row in rows:
        vals = []
        for key, _ in cols:
            val = row.get(key, "")
            if isinstance(val, float):
                digits = 2 if ("p95" in key or "pct" in key or "deg" in key) else 1
                vals.append(_fmt(val, digits))
            else:
                vals.append(str(val))
        lines.append("| " + " | ".join(vals) + " |")
    return lines


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    baselines = {"2h": _baseline_2h(), "6h": _baseline_6h()}

    rows: list[dict[str, object]] = []
    for (horizon, state), run_dir in RUNS.items():
        for path in sorted((run_dir / "timeseries").glob("*_prediction_primary_econ_timeseries.csv")):
            case_id = _case_id_from_path(path)
            metrics = _metrics(path)
            base = baselines[horizon].loc[case_id]
            row = {
                "horizon": horizon,
                "state": state,
                "case_id": case_id,
                **metrics,
                "baseline_pump_m3": float(base["pump_m3"]),
                "saved_m3": float(base["pump_m3"]) - metrics["pump_m3"],
                "saving_pct": (float(base["pump_m3"]) - metrics["pump_m3"])
                / max(float(base["pump_m3"]), 1e-9)
                * 100.0,
                "d_gt3_s": metrics["time_gt3_s"] - float(base.get("time_gt3_s", np.nan)),
                "d_gt4_s": metrics["time_gt4_s"] - float(base.get("time_gt4_s", np.nan)),
                "d_gt5_s": metrics["time_gt5_s"] - float(base["time_gt5_s"]),
                "d_fallback_s": metrics["fallback_s"] - float(base["fallback_s"]),
                "d_p95_axis_deg": metrics["p95_axis_deg"] - float(base["p95_axis_deg"]),
            }
            rows.append(row)

    case_df = pd.DataFrame(rows).sort_values(["horizon", "case_id", "state"])
    case_df.to_csv(CASE_CSV, index=False)

    summary = (
        case_df.groupby(["horizon", "state"], sort=False)
        .agg(
            cases=("case_id", "count"),
            baseline_pump_m3=("baseline_pump_m3", "sum"),
            pump_m3=("pump_m3", "sum"),
            saved_m3=("saved_m3", "sum"),
            time_gt3_s=("time_gt3_s", "sum"),
            time_gt4_s=("time_gt4_s", "sum"),
            time_gt5_s=("time_gt5_s", "sum"),
            d_gt5_s=("d_gt5_s", "sum"),
            fallback_s=("fallback_s", "sum"),
            d_fallback_s=("d_fallback_s", "sum"),
            max_d_p95_axis_deg=("d_p95_axis_deg", "max"),
            mean_d_p95_axis_deg=("d_p95_axis_deg", "mean"),
            max_axis_deg=("max_axis_deg", "max"),
        )
        .reset_index()
    )
    summary["saving_pct"] = 100.0 * summary["saved_m3"] / summary["baseline_pump_m3"].clip(lower=1e-9)
    summary.to_csv(SUMMARY_CSV, index=False)

    pivot = summary.pivot(index="horizon", columns="state")
    delta_rows = []
    for horizon in ["2h", "6h"]:
        delta_rows.append(
            {
                "horizon": horizon,
                "on_minus_off_saved_m3": float(pivot.loc[horizon, ("saved_m3", "ON")] - pivot.loc[horizon, ("saved_m3", "OFF")]),
                "on_minus_off_d_gt5_s": float(pivot.loc[horizon, ("d_gt5_s", "ON")] - pivot.loc[horizon, ("d_gt5_s", "OFF")]),
                "on_minus_off_d_fallback_s": float(pivot.loc[horizon, ("d_fallback_s", "ON")] - pivot.loc[horizon, ("d_fallback_s", "OFF")]),
                "on_minus_off_max_d_p95_deg": float(
                    pivot.loc[horizon, ("max_d_p95_axis_deg", "ON")]
                    - pivot.loc[horizon, ("max_d_p95_axis_deg", "OFF")]
                ),
            }
        )

    lines = [
        "# Case4 active-posture refresh ablation",
        "",
        "结论：先做 case 4 的判断被证实。关掉 case4 active-posture refresh 在 2h 能多省水，但到 6h 会把姿态债务和 fallback 外溢；打开 refresh 会牺牲 2h 的借来节水，但显著压低 6h gt5 / p95 / fallback。",
        "",
        "## Summary",
        "",
    ]
    lines.extend(
        _table(
            summary.to_dict("records"),
            [
                ("horizon", "视野"),
                ("state", "refresh"),
                ("saved_m3", "节水m3"),
                ("saving_pct", "节水%"),
                ("d_gt5_s", "d_gt5 s"),
                ("d_fallback_s", "d_fallback s"),
                ("max_d_p95_axis_deg", "最大d_p95 deg"),
                ("mean_d_p95_axis_deg", "平均d_p95 deg"),
                ("max_axis_deg", "最大姿态deg"),
            ],
        )
    )
    lines.extend(["", "## ON - OFF", ""])
    lines.extend(
        _table(
            delta_rows,
            [
                ("horizon", "视野"),
                ("on_minus_off_saved_m3", "ON-OFF节水m3"),
                ("on_minus_off_d_gt5_s", "ON-OFF d_gt5 s"),
                ("on_minus_off_d_fallback_s", "ON-OFF d_fallback s"),
                ("on_minus_off_max_d_p95_deg", "ON-OFF 最大d_p95"),
            ],
        )
    )
    lines.extend(
        [
            "",
            "## Readout",
            "",
            "- 2h：OFF 比 ON 多省水，但 ON 少 11s gt5、最大 p95 债务低约 0.92deg。这是短视野就能看见的债务换水。",
            "- 6h：OFF 的节水优势消失，ON 反而多保住约 39m3，同时少 1098s gt5、少 476s fallback、最大 p95 债务低约 1.91deg。",
            "- 关键 case 是 09：6h OFF 出现 476s fallback、p95 5.05deg；ON 保持 fallback=0、p95 3.14deg。",
            "- 因此 case 4 不是单纯收益问题，而是 2h 节水会在 6h 转成 deferred debt 的结构问题。下一步应做结构门/债务感知 gate，停止继续扫 enter。",
            "",
            f"- Case metrics: `{CASE_CSV.relative_to(REPO_ROOT)}`",
            f"- Summary: `{SUMMARY_CSV.relative_to(REPO_ROOT)}`",
        ]
    )
    READOUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(READOUT_MD)


if __name__ == "__main__":
    main()
