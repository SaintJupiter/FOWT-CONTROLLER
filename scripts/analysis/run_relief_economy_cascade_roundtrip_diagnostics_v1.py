#!/usr/bin/env python3
"""Read-only cascade and round-trip diagnostics for relief-economy savings.

This script checks whether the learned A1 relief-economy saving is a local
600s-window effect or a closed-loop cascade, and estimates the A0 round-trip
pump/chatter pool that a relief-aware economy layer could plausibly attack.
It does not run the controller or modify any control logic.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
BASE = REPO_ROOT / "outputs/wind_prediction/h120_relief_envelope_learned_a1_locked_holdout_v1"
RUN_A0 = BASE / "runs/A0_learned_v16"
RUN_A1 = BASE / "runs/A1_learned_near_envelope_medium_axis_guard"
TRIGGERS = BASE / "learned_a1_trigger_diagnostics.csv"
POST_WINDOW = BASE / "diagnostics/trigger_post_window_pump_table.csv"
OUT = REPO_ROOT / "outputs/wind_prediction/relief_economy_cascade_roundtrip_diagnostics_v1"


def _ensure_dirs() -> dict[str, Path]:
    dirs = {
        "out": OUT,
        "raw": OUT / "raw_tables",
        "paper": OUT / "paper_ready",
        "debug": OUT / "debug",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs


def _case_id_from_path(path: Path) -> str:
    suffix = "_prediction_primary_econ_timeseries.csv"
    name = path.name
    return name[: -len(suffix)] if name.endswith(suffix) else path.stem


def _load_ts(run_dir: Path, case: str) -> pd.DataFrame:
    path = run_dir / "timeseries" / f"{case}_prediction_primary_econ_timeseries.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    cols = [
        "t_s",
        "pitch_deg",
        "roll_deg",
        "pump_total_rate_m3_min",
        "pump_rate1_m3min",
        "pump_rate2_m3min",
        "pump_rate3_m3min",
        "tank1_kg",
        "tank2_kg",
        "tank3_kg",
        "target_tank1_kg",
        "target_tank2_kg",
        "target_tank3_kg",
    ]
    available = pd.read_csv(path, nrows=1).columns
    usecols = [c for c in cols if c in available]
    return pd.read_csv(path, usecols=usecols)


def _pump_integral(df: pd.DataFrame, start: float, end: float) -> float:
    s = df[(df["t_s"] >= start) & (df["t_s"] < end)]
    if s.empty:
        return 0.0
    return float(s["pump_total_rate_m3_min"].abs().sum() / 60.0)


def _m3_from_kg(kg: float) -> float:
    return float(kg) / 1000.0


def _roundtrip_from_tank_series(df: pd.DataFrame, tank_col: str) -> dict[str, float]:
    """Estimate reversible tank motion.

    total_motion is sum(abs(delta tank)), net_motion is abs(final-start), and
    roundtrip_proxy is the reversible component: (total - net) / 2.
    This is not "bad pump" by itself, but it is a useful upper-bound proxy for
    add-then-remove chatter in a reversible ballast system.
    """
    if tank_col not in df:
        return {"total_motion_m3": 0.0, "net_motion_m3": 0.0, "roundtrip_proxy_m3": 0.0}
    x = pd.to_numeric(df[tank_col], errors="coerce").to_numpy(dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return {"total_motion_m3": 0.0, "net_motion_m3": 0.0, "roundtrip_proxy_m3": 0.0}
    total = _m3_from_kg(np.abs(np.diff(x)).sum())
    net = _m3_from_kg(abs(x[-1] - x[0]))
    return {
        "total_motion_m3": total,
        "net_motion_m3": net,
        "roundtrip_proxy_m3": max((total - net) / 2.0, 0.0),
    }


def _roundtrip_rows() -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    a0_paths = sorted((RUN_A0 / "timeseries").glob("*_prediction_primary_econ_timeseries.csv"))
    for path in a0_paths:
        case = _case_id_from_path(path)
        a0 = _load_ts(RUN_A0, case)
        try:
            a1 = _load_ts(RUN_A1, case)
        except FileNotFoundError:
            continue
        row: dict[str, float | str] = {"case": case}
        for label, df in (("A0", a0), ("A1", a1)):
            pump = float(df["pump_total_rate_m3_min"].abs().sum() / 60.0)
            row[f"{label}_pump_m3"] = pump
            total_motion = 0.0
            net_motion = 0.0
            roundtrip = 0.0
            for tank in ("tank1_kg", "tank2_kg", "tank3_kg"):
                rt = _roundtrip_from_tank_series(df, tank)
                total_motion += rt["total_motion_m3"]
                net_motion += rt["net_motion_m3"]
                roundtrip += rt["roundtrip_proxy_m3"]
            row[f"{label}_tank_total_motion_m3"] = total_motion
            row[f"{label}_tank_net_motion_m3"] = net_motion
            row[f"{label}_roundtrip_proxy_m3"] = roundtrip
        row["A1_minus_A0_pump_m3"] = float(row["A1_pump_m3"]) - float(row["A0_pump_m3"])
        row["A1_minus_A0_roundtrip_proxy_m3"] = float(row["A1_roundtrip_proxy_m3"]) - float(
            row["A0_roundtrip_proxy_m3"]
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _cascade_rows() -> pd.DataFrame:
    triggers = pd.read_csv(POST_WINDOW)
    diag = pd.read_csv(TRIGGERS).set_index("case")
    rows: list[dict[str, float | str]] = []
    windows = [600, 1200, 1800, 3600, 7200]
    for _, tr in triggers.iterrows():
        case = str(tr["case"])
        trigger_t = float(tr["trigger_t"])
        drow = diag.loc[case] if case in diag.index else tr
        a0 = _load_ts(RUN_A0, case)
        a1 = _load_ts(RUN_A1, case)
        row: dict[str, float | str] = {
            "case": case,
            "case_source": str(tr.get("case_source", "")),
            "trigger_t": trigger_t,
            "total_case_delta_pump_m3": float(drow["A1_minus_A0_pump_m3"]),
            "total_case_delta_time5_s": float(drow["A1_minus_A0_time5_s"]),
            "total_case_delta_idle5_s": float(drow["A1_minus_A0_idle5_s"]),
            "total_case_delta_fallback_pp": float(drow["A1_minus_A0_fallback_pp"]),
            "total_case_delta_p95_deg": float(drow["A1_minus_A0_p95_deg"]),
        }
        prev = 0.0
        for w in windows:
            end = trigger_t + float(w)
            p0 = _pump_integral(a0, trigger_t, end)
            p1 = _pump_integral(a1, trigger_t, end)
            delta = p1 - p0
            row[f"delta_pump_{w}s_m3"] = delta
            row[f"incremental_delta_pump_{prev:.0f}_{w}s_m3"] = delta - (
                row.get(f"delta_pump_{int(prev)}s_m3", 0.0) if prev > 0 else 0.0
            )
            prev = float(w)
        total = float(row["total_case_delta_pump_m3"])
        early = float(row["delta_pump_600s_m3"])
        row["post_600s_delta_pump_m3"] = total - early
        row["share_of_saving_after_600s"] = (
            max(-(total - early), 0.0) / max(-total, 1e-9) if total < 0 else 0.0
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _md_table(df: pd.DataFrame, max_rows: int | None = None) -> str:
    if df.empty:
        return "_No rows._"
    d = df if max_rows is None else df.head(max_rows)
    cols = list(d.columns)
    out = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in d.iterrows():
        vals = []
        for c in cols:
            v = row[c]
            if isinstance(v, float):
                if math.isfinite(v):
                    vals.append(f"{v:.3f}")
                else:
                    vals.append("")
            else:
                vals.append(str(v))
        out.append("| " + " | ".join(vals) + " |")
    return "\n".join(out)


def main() -> None:
    dirs = _ensure_dirs()
    cascade = _cascade_rows()
    rt = _roundtrip_rows()

    cascade.to_csv(dirs["raw"] / "cascade_attribution_table.csv", index=False)
    rt.to_csv(dirs["raw"] / "roundtrip_proxy_case_table.csv", index=False)

    total_saved = -float(cascade["total_case_delta_pump_m3"].sum())
    early_saved = -float(cascade["delta_pump_600s_m3"].sum())
    post600_saved = total_saved - early_saved
    share_post600 = post600_saved / total_saved if total_saved > 1e-9 else 0.0

    summary = pd.DataFrame(
        [
            {"metric": "total_saved_m3", "value": total_saved},
            {"metric": "saved_within_600s_m3", "value": early_saved},
            {"metric": "saved_after_600s_m3", "value": post600_saved},
            {"metric": "share_saved_after_600s", "value": share_post600},
            {"metric": "A0_roundtrip_proxy_m3", "value": float(rt["A0_roundtrip_proxy_m3"].sum())},
            {"metric": "A1_roundtrip_proxy_m3", "value": float(rt["A1_roundtrip_proxy_m3"].sum())},
            {
                "metric": "roundtrip_proxy_reduction_m3",
                "value": -float(rt["A1_minus_A0_roundtrip_proxy_m3"].sum()),
            },
            {
                "metric": "saving_as_fraction_of_A0_roundtrip_proxy",
                "value": total_saved / max(float(rt["A0_roundtrip_proxy_m3"].sum()), 1e-9),
            },
        ]
    )
    summary.to_csv(dirs["raw"] / "cascade_roundtrip_summary.csv", index=False)
    summary.to_csv(dirs["out"] / "cascade_roundtrip_summary.csv", index=False)

    top_rt = rt.sort_values("A0_roundtrip_proxy_m3", ascending=False).head(12)
    top_cascade = cascade.sort_values("post_600s_delta_pump_m3").copy()

    md = f"""# Relief Economy Cascade And Round-Trip Diagnostics

## Purpose

This read-only audit checks the reviewer challenge that the previous 600s
windowed ceiling was not a true upper bound. It measures whether the current
`875.94 m3` saving is local to the delayed-refresh window or comes from
closed-loop trajectory cascades, and estimates the A0 reversible pump-motion
pool.

## Summary

| metric | value |
| --- | ---: |
| total saving from delayed cases | {total_saved:.2f} m3 |
| saving within 600s of trigger | {early_saved:.2f} m3 |
| saving after 600s | {post600_saved:.2f} m3 |
| share after 600s | {share_post600:.1%} |
| A0 round-trip proxy | {float(rt["A0_roundtrip_proxy_m3"].sum()):.2f} m3 |
| A1 round-trip proxy | {float(rt["A1_roundtrip_proxy_m3"].sum()):.2f} m3 |
| round-trip proxy reduction | {-float(rt["A1_minus_A0_roundtrip_proxy_m3"].sum()):.2f} m3 |

## Cascade Attribution

{_md_table(cascade[["case", "case_source", "total_case_delta_pump_m3", "delta_pump_600s_m3", "post_600s_delta_pump_m3", "share_of_saving_after_600s"]])}

## Largest A0 Round-Trip Proxy Cases

{_md_table(top_rt[["case", "A0_pump_m3", "A0_roundtrip_proxy_m3", "A1_minus_A0_pump_m3", "A1_minus_A0_roundtrip_proxy_m3"]])}

## Interpretation

- The prior 600s addressable-window number is **not** a hard ceiling if a
  delayed refresh changes later target/pump trajectory.
- A positive `saving_after_600s` means the result includes cascade/trajectory
  effects beyond the immediate refresh window.
- The round-trip proxy is an upper-bound diagnostic for reversible ballast
  motion, not a direct claim that all such motion is avoidable.
"""
    for path in (dirs["out"] / "cascade_roundtrip_diagnostics.md", dirs["paper"] / "cascade_roundtrip_diagnostics.md"):
        path.write_text(md, encoding="utf-8")

    print(f"[done] {OUT}")


if __name__ == "__main__":
    main()
