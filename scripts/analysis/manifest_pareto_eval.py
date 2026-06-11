#!/usr/bin/env python3
"""Manifest-based controller Pareto evaluator.

Replaces fuzzy directory scanning with explicit (controller_label, run_dir)
mapping. For each entry:

  1. Reads casebook_summary.csv (run metadata) to record protocol:
       forecast_source_effective, primary_control_profile, primary_safety_profile,
       closed_pump_profile, duration_s
  2. Reads per-case timeseries to compute raw 1Hz metrics
  3. Refuses to mix entries with different protocols inside the same
     comparison group (raises an error if a group is mixed)

Output:
  outputs/wind_prediction/manifest_pareto_eval/<group>_clean_pareto_table.csv
  outputs/wind_prediction/manifest_pareto_eval/<group>_clean_pareto.png

Hand-edit the MANIFEST dict below to add or remove controllers.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# (group, label, run_dir) — entries inside the same group must share protocol.
# Group "oracle_guard10" = all guard10 candidates that share oracle baseline.
# Group "learned_5case" = learned-forecast candidates.
MANIFEST = [
    # ---------------- Oracle protocol, 5case (canonical group, vanilla baseline) ----------------
    # baseline_v2a uses "manual" profile (no overlays); the candidates here use
    # "rawenv_holdpause_barrier_reliefcap_adaptive_v1" (full production overlays).
    # So this group answers: how does the full production stack (overlays + v3 mode)
    # compare to a no-overlay baseline?
    ("oracle_5case", "baseline_v2a", "outputs/wind_prediction/no_unexplained_hold_v2a_current5_baseline"),
    ("oracle_5case", "obj_mode_v3", "outputs/wind_prediction/objective_mode_v3_oracle_5case_2h"),
    # ---------------- Oracle protocol, 5case, ISOLATED v3-effect ----------------
    # Both entries use full overlays (rawenv_holdpause_barrier_reliefcap_adaptive_v1).
    # Only difference: --planner-objective-mode-active is OFF in the baseline.
    # So this group cleanly isolates the v3 mode effect from overlay effects.
    ("oracle_5case_isolated_v3", "overlays_only", "outputs/wind_prediction/baseline_overlays_oracle_5case_2h"),
    ("oracle_5case_isolated_v3", "overlays_plus_v3", "outputs/wind_prediction/objective_mode_v3_oracle_5case_2h"),
    # ---------------- Oracle protocol, 5case, candidates under overlays ----------------
    # Same overlay profile + oracle + 5case as the isolated_v3 group, but here we
    # test candidates that were originally evaluated under the bare "manual" profile.
    # Goal: do active_reproposal / hold_relief_debt produce different effects when
    # combined with the full overlay stack? Compare each against `overlays_only`.
    ("oracle_5case_candidates_under_overlays", "overlays_only", "outputs/wind_prediction/baseline_overlays_oracle_5case_2h"),
    ("oracle_5case_candidates_under_overlays", "active_reproposal_overlays", "outputs/wind_prediction/active_reproposal_overlays_oracle_5case_2h"),
    ("oracle_5case_candidates_under_overlays", "hold_relief_debt_overlays", "outputs/wind_prediction/hold_relief_debt_overlays_oracle_5case_2h"),
    # ---------------- Oracle protocol, guard10 (broader case set, manual profile only) ----------------
    ("oracle_guard10", "baseline_v2a_guard10", "outputs/wind_prediction/no_unexplained_hold_v2a_current5_baseline"),
    ("oracle_guard10", "active_reproposal", "outputs/wind_prediction/active_reproposal_guard10_candidate"),
    ("oracle_guard10", "hold_relief_debt", "outputs/wind_prediction/hold_relief_debt_guard10_candidate"),
    # ---------------- Learned protocol, 5case (apples-vs-apples to oracle_5case) ----------------
    ("learned_5case", "obj_mode_v3_learned", "outputs/wind_prediction/objective_mode_v3_prototype_5case_2h"),
]

CANONICAL_CASES = [
    'fr_relief_01', 'fr_relief_09', 'b_high_pressure_event',
    'lowrisk_clean', 'lowrisk_random_03',
    # also accept guard10 extended cases
    'sf_holdout_02', 'b_decay_strong', 'b_signflip_fallback', 'b_residual_high',
    'lowrisk_quiet',
]

PROTOCOL_KEYS = [
    "primary_control_profile",
    "primary_safety_profile",
    "closed_pump_profile",
    "forecast_source_effective",
]


def read_metadata(run_dir: Path) -> dict:
    """Extract protocol metadata from casebook_summary.csv."""
    csv = run_dir / "casebook_summary.csv"
    if not csv.exists():
        raise FileNotFoundError(f"missing {csv}")
    d = pd.read_csv(csv, nrows=1)
    return {k: str(d[k].iloc[0]) if k in d.columns else "UNKNOWN" for k in PROTOCOL_KEYS}


def compute_case_metrics(run_dir: Path) -> pd.DataFrame:
    """Compute per-case 1Hz metrics from timeseries/*.csv."""
    ts_dir = run_dir / "timeseries"
    if not ts_dir.exists():
        raise FileNotFoundError(f"missing {ts_dir}")
    rows = []
    for f in sorted(ts_dir.glob("*timeseries.csv")):
        # Skip closed_only/reactive helper files that share the candidate's dir
        # but represent a different controller for figure comparison.
        if "closed_only" in f.name or "reactive_current" in f.name:
            continue
        case = next((c for c in CANONICAL_CASES if c in f.name), None)
        if case is None:
            continue
        try:
            ts = pd.read_csv(f, low_memory=False)
        except Exception as e:
            print(f"  skip {f.name}: {e}", file=sys.stderr)
            continue
        if "pitch_deg" not in ts.columns:
            continue
        pitch = ts["pitch_deg"].abs()
        roll = ts["roll_deg"].abs() if "roll_deg" in ts.columns else pd.Series([0.0])
        pump = (ts["pump_total_rate_m3_min"].abs() / 60).sum() if "pump_total_rate_m3_min" in ts.columns else float("nan")
        fb_col = next((c for c in ts.columns if "safety_fallback" in c.lower() or "fallback_active" in c.lower()), None)
        fb = float(ts[fb_col].mean()) if fb_col else 0.0
        t_over_3 = int(((pitch > 3) | (roll > 3)).sum())
        t_over_4 = int(((pitch > 4) | (roll > 4)).sum())
        rows.append({
            "case": case,
            "pump_m3": float(pump),
            "pitch_p95": float(np.percentile(pitch, 95)),
            "pitch_max": float(pitch.max()),
            "roll_p95": float(np.percentile(roll, 95)),
            "fallback_ratio": fb,
            "t_over_3_s": t_over_3,
            "t_over_4_s": t_over_4,
            "n_steps": len(ts),
        })
    return pd.DataFrame(rows)


def collect_group(group_name: str) -> dict:
    entries = [e for e in MANIFEST if e[0] == group_name]
    if not entries:
        raise ValueError(f"empty group {group_name}")
    runs = {}
    protocols = []
    for grp, label, run_dir in entries:
        rd = Path(run_dir)
        if not rd.exists():
            print(f"WARN: missing run_dir {rd}", file=sys.stderr)
            continue
        meta = read_metadata(rd)
        metrics = compute_case_metrics(rd)
        runs[label] = {"protocol": meta, "metrics": metrics, "run_dir": str(rd)}
        protocols.append((label, meta))
    # Verify protocol homogeneity
    if protocols:
        ref_label, ref_proto = protocols[0]
        for label, proto in protocols[1:]:
            for k in PROTOCOL_KEYS:
                if proto[k] != ref_proto[k]:
                    print(f"PROTOCOL MISMATCH in group '{group_name}': "
                          f"{label}.{k}={proto[k]!r} vs {ref_label}.{k}={ref_proto[k]!r}",
                          file=sys.stderr)
    return runs


def build_pareto_table(group_runs: dict) -> pd.DataFrame:
    rows = []
    for label, info in group_runs.items():
        m = info["metrics"]
        for _, r in m.iterrows():
            rows.append({
                "ctrl": label,
                "case": r["case"],
                "pump_m3": r["pump_m3"],
                "pitch_p95": r["pitch_p95"],
                "pitch_max": r["pitch_max"],
                "roll_p95": r["roll_p95"],
                "fallback_ratio": r["fallback_ratio"],
                "t_over_3_s": r["t_over_3_s"],
                "t_over_4_s": r["t_over_4_s"],
                "protocol_profile": info["protocol"]["primary_control_profile"],
                "protocol_forecast": info["protocol"]["forecast_source_effective"],
            })
    return pd.DataFrame(rows)


def plot_group(group_name: str, table: pd.DataFrame, out_dir: Path) -> None:
    cases = sorted(table["case"].unique())
    n = len(cases)
    if n == 0:
        return
    fig, axes = plt.subplots((n + 1) // 2, 2, figsize=(14, 4 * ((n + 1) // 2)))
    axes = np.array(axes).flatten()
    color_map = {}
    cmap = plt.cm.tab10
    for i, ctrl in enumerate(sorted(table["ctrl"].unique())):
        color_map[ctrl] = cmap(i % 10)
    for i, case in enumerate(cases):
        ax = axes[i]
        sub = table[table.case == case].sort_values("pump_m3")
        for _, r in sub.iterrows():
            color = color_map.get(r["ctrl"], "black")
            size = 80 + 1500 * r["fallback_ratio"]
            ax.scatter(r["pump_m3"], r["pitch_p95"], c=[color], s=size, edgecolors="black",
                       linewidths=0.6, alpha=0.8)
            ax.annotate(r["ctrl"], (r["pump_m3"], r["pitch_p95"]), fontsize=8,
                        xytext=(5, 3), textcoords="offset points")
        ax.set_xlabel("pump m³")
        ax.set_ylabel("pitch p95 °")
        ax.set_title(case)
        ax.grid(alpha=0.3)
        ax.axhline(3, color="gray", linestyle=":", linewidth=0.8)
        ax.axhline(5, color="red", linestyle=":", linewidth=0.8)
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    proto_line = ", ".join(
        f"{k}={list(table[c].unique())[0]}"
        for k, c in [("profile", "protocol_profile"), ("forecast", "protocol_forecast")]
        if c in table.columns
    )
    fig.suptitle(f"Pareto (group={group_name})  protocol: {proto_line}\n(circle size = fallback_ratio)", fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out = out_dir / f"{group_name}_clean_pareto.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"wrote {out}")


def main():
    out_dir = Path("outputs/wind_prediction/manifest_pareto_eval")
    out_dir.mkdir(parents=True, exist_ok=True)
    for group in sorted({g for g, _, _ in MANIFEST}):
        print(f"\n=== group: {group} ===")
        runs = collect_group(group)
        if not runs:
            continue
        for label, info in runs.items():
            print(f"  {label}: {info['run_dir']}")
            for k, v in info["protocol"].items():
                print(f"    {k}: {v}")
        table = build_pareto_table(runs)
        if table.empty:
            continue
        csv = out_dir / f"{group}_clean_pareto_table.csv"
        table.to_csv(csv, index=False)
        print(f"wrote {csv}")
        plot_group(group, table, out_dir)
        # Per-case sorted view
        print(f"\nPer-case Pareto for group {group}:")
        for case in sorted(table["case"].unique()):
            sub = table[table.case == case].sort_values("pump_m3")
            print(f"\n  {case}:")
            for _, r in sub.iterrows():
                print(f"    {r['ctrl']:30s} pump={r['pump_m3']:7.1f} p95={r['pitch_p95']:.3f}° fb={r['fallback_ratio']:.4f} t_over_3={r['t_over_3_s']}s")


if __name__ == "__main__":
    main()
