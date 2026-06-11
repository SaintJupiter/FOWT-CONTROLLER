#!/usr/bin/env python3
"""Summarize No.4 v2 smoke attempts against the frozen 0hao baseline."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
BASE = REPO_ROOT / "outputs" / "wind_prediction" / "psc_selector_mixed_pool_v1"
FEASIBILITY = BASE / "psc_4hao_moderate_arm_feasibility_v1"
BASELINE_METRICS = BASE / "degradation_ladder_96case_pair" / "degradation_ladder_case_metrics.csv"
SMOKE_CASES = FEASIBILITY / "4hao_smoke_cases.csv"
OUT = BASE / "psc_4hao_v2_smoke_attempts_v1"
OUT_CSV = OUT / "4hao_v2_smoke_attempts_case_metrics.csv"
OUT_SUMMARY = OUT / "4hao_v2_smoke_attempts_profile_summary.csv"
OUT_MD = OUT / "4hao_v2_smoke_attempts_readout.md"

PROFILES = [
    {
        "profile_id": "old_regime_auto",
        "name": "旧4号regime-auto",
        "dir": BASE / "psc_4hao_smoke_120min_regime_auto_learned_v1",
        "role": "old_reference",
    },
    {
        "profile_id": "v2_budget30_comfort",
        "name": "v2尝试A budget30+comfort",
        "dir": BASE / "psc_4hao_v2_smoke_120min_budget30_comfort_v1",
        "role": "failed_too_aggressive",
    },
    {
        "profile_id": "v2_budget300_comfort",
        "name": "v2尝试B budget300+comfort",
        "dir": BASE / "psc_4hao_v2_smoke_120min_budget300_comfort_v1",
        "role": "failed_not_real_mild_knob",
    },
]


def _axis_metrics(path: Path) -> dict[str, float]:
    df = pd.read_csv(path, usecols=["pitch_deg", "roll_deg", "pump_total_rate_m3_min", "preview_primary_safety_fallback"])
    axis = df[["pitch_deg", "roll_deg"]].abs().max(axis=1).to_numpy(float)
    pump = df["pump_total_rate_m3_min"].to_numpy(float)
    return {
        "pump_m3": float(np.sum(np.abs(pump)) / 60.0),
        "time_gt3_s": float((axis > 3.0).sum()),
        "time_gt4_s": float((axis > 4.0).sum()),
        "time_gt5_s": float((axis > 5.0).sum()),
        "time_gt6_s": float((axis > 6.0).sum()),
        "fallback_s": float(pd.to_numeric(df["preview_primary_safety_fallback"], errors="coerce").fillna(0.0).sum()),
        "p95_axis_deg": float(np.quantile(axis, 0.95)),
        "max_axis_deg": float(np.max(axis)),
    }


def _run_case_id(case_id: str, cases: pd.DataFrame) -> str:
    matches = cases.index[cases["case_id"].astype(str).eq(case_id)].tolist()
    if not matches:
        raise KeyError(case_id)
    raw = str(cases.loc[matches[0], "case_id"])
    clean = raw[3:] if len(raw) > 3 and raw[:2].isdigit() and raw[2] == "_" else raw
    return f"{matches[0] + 1:02d}_{clean}"


def _find_ts(run_dir: Path, run_case_id: str) -> Path | None:
    matches = sorted((run_dir / "timeseries").glob(f"{run_case_id}_*_timeseries.csv"))
    if len(matches) == 0:
        return None
    if len(matches) != 1:
        raise FileNotFoundError(
            f"{run_dir}: expected one timeseries for {run_case_id}, found {len(matches)}"
        )
    return matches[0]


def _fmt(value: float, suffix: str = "") -> str:
    if pd.isna(value):
        return ""
    return f"{value:.2f}{suffix}"


def _table(rows: list[dict[str, object]], cols: list[tuple[str, str]]) -> list[str]:
    lines = [
        "| " + " | ".join(title for _, title in cols) + " |",
        "| " + " | ".join("---" for _ in cols) + " |",
    ]
    for row in rows:
        values: list[str] = []
        for key, _ in cols:
            value = row.get(key, "")
            if isinstance(value, float):
                if "pct" in key:
                    values.append(_fmt(value, "%"))
                elif "m3" in key:
                    values.append(f"{value:.1f}")
                elif "deg" in key:
                    values.append(f"{value:.2f}")
                else:
                    values.append(f"{value:.0f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return lines


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cases = pd.read_csv(SMOKE_CASES)
    baseline = pd.read_csv(BASELINE_METRICS)
    baseline = baseline[baseline["arm"].eq("current_forecast_adaptive")].set_index("case_id")
    rows: list[dict[str, object]] = []
    for profile in PROFILES:
        run_dir = Path(profile["dir"])
        if not (run_dir / "timeseries").exists():
            continue
        for case_id in cases["case_id"].astype(str):
            run_case_id = _run_case_id(case_id, cases)
            ts = _find_ts(run_dir, run_case_id)
            if ts is None:
                continue
            metrics = _axis_metrics(ts)
            base = baseline.loc[case_id]
            saved = float(base["pump_m3"]) - metrics["pump_m3"]
            row = {
                "profile_id": profile["profile_id"],
                "profile_name": profile["name"],
                "role": profile["role"],
                "case_id": case_id,
                "saved_m3": saved,
                "saving_pct": 100.0 * saved / max(float(base["pump_m3"]), 1e-9),
                "baseline_pump_m3": float(base["pump_m3"]),
                "primary_pump_m3": metrics["pump_m3"],
                "baseline_p95_axis_deg": float(base["p95_axis_deg"]),
                "primary_p95_axis_deg": metrics["p95_axis_deg"],
                "d_p95_axis_deg": metrics["p95_axis_deg"] - float(base["p95_axis_deg"]),
                "primary_max_axis_deg": metrics["max_axis_deg"],
                "d_time_gt3_s": metrics["time_gt3_s"] - float((0 if "time_gt3_s" not in base else base["time_gt3_s"])),
                "primary_time_gt4_s": metrics["time_gt4_s"],
                "primary_time_gt5_s": metrics["time_gt5_s"],
                "d_time_gt5_s": metrics["time_gt5_s"] - float(base["time_gt5_s"]),
                "primary_time_gt6_s": metrics["time_gt6_s"],
                "d_time_gt6_s": metrics["time_gt6_s"] - float(base["time_gt6_s"]),
                "primary_fallback_s": metrics["fallback_s"],
                "d_fallback_s": metrics["fallback_s"] - float(base["fallback_s"]),
            }
            row["safety_fail"] = int(row["d_fallback_s"] > 0 or row["d_time_gt5_s"] >= 60)
            rows.append(row)

    out = pd.DataFrame(rows)
    out.to_csv(OUT_CSV, index=False)

    summary = (
        out.groupby(["profile_id", "profile_name", "role"], sort=False)
        .agg(
            cases=("case_id", "count"),
            fail_cases=("safety_fail", "sum"),
            saved_m3=("saved_m3", "sum"),
            baseline_pump_m3=("baseline_pump_m3", "sum"),
            d_time_gt5_s=("d_time_gt5_s", "sum"),
            d_fallback_s=("d_fallback_s", "sum"),
            max_d_p95_axis_deg=("d_p95_axis_deg", "max"),
            max_primary_p95_axis_deg=("primary_p95_axis_deg", "max"),
            max_primary_axis_deg=("primary_max_axis_deg", "max"),
        )
        .reset_index()
    )
    summary["saving_pct"] = 100.0 * summary["saved_m3"] / summary["baseline_pump_m3"].clip(lower=1e-9)
    summary.to_csv(OUT_SUMMARY, index=False)

    lines = [
        "# 4号 v2 smoke attempt readout",
        "",
        "## Profile summary",
        "",
    ]
    lines.extend(
        _table(
            summary.to_dict("records"),
            [
                ("profile_name", "方案"),
                ("role", "定位"),
                ("cases", "case数"),
                ("fail_cases", "失败case"),
                ("saved_m3", "节水m3"),
                ("saving_pct", "节水%"),
                ("d_time_gt5_s", "d_gt5 s"),
                ("d_fallback_s", "d_fallback s"),
                ("max_d_p95_axis_deg", "最大d_p95 deg"),
                ("max_primary_axis_deg", "最大姿态deg"),
            ],
        )
    )
    lines.extend(["", "## Key cases", ""])
    focus = out[out["case_id"].isin(["09_dual_relief_09", "10_dual_relief_10", "11_dual_relief_11", "18_dual_relief_18", "94_dual_boundary_06"])].copy()
    lines.extend(
        _table(
            focus.to_dict("records"),
            [
                ("profile_name", "方案"),
                ("case_id", "case"),
                ("saved_m3", "节水m3"),
                ("saving_pct", "case节水%"),
                ("primary_p95_axis_deg", "p95姿态deg"),
                ("d_p95_axis_deg", "d_p95 deg"),
                ("primary_time_gt4_s", "gt4 s"),
                ("d_time_gt5_s", "d_gt5 s"),
                ("d_fallback_s", "d_fallback s"),
                ("safety_fail", "失败"),
            ],
        )
    )
    lines.extend(
        [
            "",
            "## Readout",
            "",
            "- budget30/300 were real closed-loop smoke runs, but they are not yet a successful 4号温和arm。",
            "- The current budget mechanism is mostly a refresh-hold gate: smaller budget means earlier hold and can be more aggressive, not a smooth 20%-40% action scale.",
            "- The next implementation should add a true mild-action knob that scales or caps the held refresh delta itself, then combine it with p95/gt4/gt5/fallback acceptance gates.",
            "",
            f"- Case metrics: `{OUT_CSV}`",
            f"- Profile summary: `{OUT_SUMMARY}`",
        ]
    )
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {OUT_MD}")


if __name__ == "__main__":
    main()
