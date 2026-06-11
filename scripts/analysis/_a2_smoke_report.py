#!/usr/bin/env python3
"""Re-build the A2-smoke pass/fail report from already-saved time-series CSVs.

Used because the original run_a2_smoke.py crashed in the markdown step after
all simulations had already completed and been saved.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

repo_root = Path(__file__).resolve().parents[2]
out_dir = repo_root / "outputs" / "wind_prediction" / "planner_a2_smoke"

WINDOWS = [
    ("2023-09-11 03:40:00", "clean_low_pressure_normal", "2023-09-11_034000"),
    ("2024-09-05 18:10:00", "residual_high_normal",     "2024-09-05_181000"),
    ("2022-02-04 11:00:00", "high_pressure_high_event", "2022-02-04_110000"),
]
MODES = ("closed_only", "closed_plus_planner")


def summarize(df: pd.DataFrame) -> dict:
    pitch = df["pitch_deg"].astype(float).to_numpy() if "pitch_deg" in df.columns else np.zeros(len(df))
    roll = df["roll_deg"].astype(float).to_numpy() if "roll_deg" in df.columns else np.zeros(len(df))
    s = {
        "n_steps": int(len(df)),
        "pitch_rms": float(np.sqrt(np.mean(pitch ** 2))),
        "roll_rms": float(np.sqrt(np.mean(roll ** 2))),
        "pitch_abs_max": float(np.max(np.abs(pitch))),
        "roll_abs_max": float(np.max(np.abs(roll))),
        "any_nan": int(np.any(np.isnan(pitch)) or np.any(np.isnan(roll))),
    }
    if "pump_fullspeed_any" in df.columns:
        s["pump_fullspeed_any_ratio"] = float(np.mean(df["pump_fullspeed_any"].astype(float)))
    if "preview_pitch_bias_deg" in df.columns:
        bias = df["preview_pitch_bias_deg"].astype(float).to_numpy()
        s["preview_pitch_bias_max_abs"] = float(np.max(np.abs(bias)))
        s["preview_pitch_bias_nonzero_ratio"] = float(np.mean(np.abs(bias) > 1e-9))
    return s


def main() -> None:
    rows = []
    for ts, group, slug in WINDOWS:
        for mode in MODES:
            path = out_dir / f"{slug}_{mode}_timeseries.csv"
            if not path.exists():
                continue
            df = pd.read_csv(path)
            row = summarize(df)
            row.update({"window": ts, "group": group, "mode": mode, "slug": slug})
            if mode == "closed_plus_planner":
                log_path = out_dir / f"{slug}_{mode}_planner_log.csv"
                if log_path.exists():
                    log = pd.read_csv(log_path)
                    row["planner_first_actions"] = "|".join(log["first_action"].astype(str))
                    row["planner_n_buckets"] = len(log)
            rows.append(row)

    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / "A2_smoke_summary.csv", index=False)

    # Pass/fail
    checks = []
    for ts, group, slug in WINDOWS:
        sub = summary[summary["window"] == ts]
        if len(sub) != 2:
            continue
        co = sub[sub["mode"] == "closed_only"].iloc[0]
        cp = sub[sub["mode"] == "closed_plus_planner"].iloc[0]
        ok1 = (co["any_nan"] == 0) and (cp["any_nan"] == 0)
        checks.append((ts, "no_nan", ok1, f"co_nan={co['any_nan']} cp_nan={cp['any_nan']}"))

        ok2_p = cp["pitch_rms"] <= max(2.0 * co["pitch_rms"], 0.05)
        ok2_r = cp["roll_rms"] <= max(2.0 * co["roll_rms"], 0.05)
        checks.append((ts, "rms_bounded", ok2_p and ok2_r,
                       f"pitch co={co['pitch_rms']:.3f} cp={cp['pitch_rms']:.3f} | "
                       f"roll co={co['roll_rms']:.3f} cp={cp['roll_rms']:.3f}"))

        if "pump_fullspeed_any_ratio" in co.index and "pump_fullspeed_any_ratio" in cp.index:
            base = max(float(co["pump_fullspeed_any_ratio"]), 0.001)
            ok3 = cp["pump_fullspeed_any_ratio"] <= 1.5 * base + 0.05
            checks.append((ts, "pump_fullspeed_bounded", ok3,
                           f"co={co['pump_fullspeed_any_ratio']:.3f} cp={cp['pump_fullspeed_any_ratio']:.3f}"))

        if "preview_pitch_bias_max_abs" in cp.index:
            ok4 = float(cp["preview_pitch_bias_max_abs"]) > 0.0 or "active" not in str(cp.get("planner_first_actions", ""))
            checks.append((ts, "preview_bias_observed", ok4,
                           f"max_abs={cp['preview_pitch_bias_max_abs']:.4f} "
                           f"nonzero_ratio={cp.get('preview_pitch_bias_nonzero_ratio', float('nan')):.3f} "
                           f"actions={cp.get('planner_first_actions','')}"))

    pass_n = sum(1 for *_, ok, _ in [(c[0], c[1], c[2], c[3]) for c in checks] if ok)
    total = len(checks)

    lines = [
        "# A2-smoke Report",
        "",
        f"- windows: {len(WINDOWS)} × 2 modes",
        f"- pass: {pass_n}/{total} interface checks",
        "",
        "## Interface checks",
        "",
        "| window | check | result | detail |",
        "|---|---|---|---|",
    ]
    for ts, name, ok, detail in checks:
        lines.append(f"| {ts} | {name} | {'PASS' if ok else 'FAIL'} | {detail} |")

    lines += ["", "## Summary table", "", "```", summary.to_string(index=False), "```"]

    verdict = "INTERFACE_OK" if pass_n == total else "INTERFACE_NEEDS_FIX"
    lines += ["", f"## Verdict: `{verdict}`", ""]

    (out_dir / "A2_smoke_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"{verdict}: {pass_n}/{total} pass")
    print(summary[["window", "mode", "pitch_rms", "roll_rms", "any_nan"]].to_string(index=False))


if __name__ == "__main__":
    main()
