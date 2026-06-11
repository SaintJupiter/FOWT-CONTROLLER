#!/usr/bin/env python3
"""Virtual forecast-regime gate for aggressive economy-mode probes.

This is a fast, read-only decisive experiment.  It composes existing A0 and
budget-arm closed-loop outputs case-by-case:

* E-blind: every case uses the aggressive budget arm.
* E-gated: only cases classified as forecast opportunity regimes use the budget
  arm; all other cases stay at A0.
* E-scrambled/random: same number of budget cases, but selected without the
  forecast regime.

The goal is not to ship this virtual selector.  It answers whether learned
forecast regime selection is even directionally smarter than blind relaxation.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd


def case_id(path: Path) -> str:
    suffixes = (
        "_prediction_primary_econ_planner_log.csv",
        "_prediction_primary_econ_timeseries.csv",
    )
    for suffix in suffixes:
        if path.name.endswith(suffix):
            return path.name[: -len(suffix)]
    return path.stem


def metrics_from_timeseries(path: Path) -> dict[str, float | str]:
    df = pd.read_csv(path, low_memory=False)
    pitch = df.get("pitch_deg", pd.Series(np.zeros(len(df)))).to_numpy(dtype=float)
    roll = df.get("roll_deg", pd.Series(np.zeros(len(df)))).to_numpy(dtype=float)
    max_axis = np.maximum(np.abs(pitch), np.abs(roll))
    pump = df.get("pump_total_rate_m3_min", pd.Series(np.zeros(len(df)))).to_numpy(dtype=float)
    return {
        "pump_m3": float(np.nansum(np.abs(pump)) / 60.0),
        "time_over3_s": float(np.nansum(max_axis > 3.0)),
        "time_over4_s": float(np.nansum(max_axis > 4.0)),
        "time_over45_s": float(np.nansum(max_axis > 4.5)),
        "time_over5_s": float(np.nansum(max_axis > 5.0)),
        "idle_over5_s": float(np.nansum((max_axis > 5.0) & (np.abs(pump) < 0.5))),
        "p95_max_axis": float(np.nanpercentile(max_axis, 95)) if len(max_axis) else 0.0,
        "max_axis": float(np.nanmax(max_axis)) if len(max_axis) else 0.0,
    }


def load_arm_metrics(group_dir: Path, arm: str) -> pd.DataFrame:
    rows = []
    for p in sorted((group_dir / arm / "timeseries").glob("*_timeseries.csv")):
        r = metrics_from_timeseries(p)
        r["case"] = case_id(p)
        r["arm"] = arm
        rows.append(r)
    return pd.DataFrame(rows)


def forecast_features(group_dir: Path) -> pd.DataFrame:
    rows = []
    for p in sorted((group_dir / "a0_baseline" / "planner_logs").glob("*_planner_log.csv")):
        df = pd.read_csv(p, low_memory=False)
        b0 = df.get("raw_pressure_block0_norm", pd.Series(np.nan, index=df.index)).to_numpy(dtype=float)
        b1 = df.get("raw_pressure_block1_norm", pd.Series(np.nan, index=df.index)).to_numpy(dtype=float)
        b2 = df.get("raw_pressure_block2_norm", pd.Series(np.nan, index=df.index)).to_numpy(dtype=float)
        near_max = np.nanmax(np.vstack([b0, b1, b2]), axis=0)
        decay = b0 - b2
        relief = (b2 <= 0.7) & (decay >= 0.2)
        calm = near_max < 0.9
        high = near_max >= 0.9
        reintensify = (b2 >= 0.9) & (b2 > b0 + 0.05)
        # A deployable regime score: high when relief/decay is persistent and
        # low when re-intensification dominates.
        relief_frac = float(np.nanmean(relief)) if len(df) else 0.0
        calm_frac = float(np.nanmean(calm)) if len(df) else 0.0
        high_frac = float(np.nanmean(high)) if len(df) else 0.0
        reint_frac = float(np.nanmean(reintensify)) if len(df) else 0.0
        mean_decay = float(np.nanmean(decay)) if len(df) else 0.0
        score = relief_frac + 0.5 * max(mean_decay, 0.0) + 0.25 * calm_frac - 0.75 * reint_frac
        rows.append(
            {
                "case": case_id(p),
                "relief_frac": relief_frac,
                "calm_frac": calm_frac,
                "high_frac": high_frac,
                "reintensify_frac": reint_frac,
                "mean_decay": mean_decay,
                "regime_score": score,
                "case_label_hint": "relief" if "_fr_relief_" in p.name or p.name.startswith(("01_fr", "02_fr", "03_fr", "04_fr", "05_fr", "06_fr", "07_fr", "08_fr", "09_fr", "10_fr")) else "other",
            }
        )
    return pd.DataFrame(rows)


def aggregate(selected: pd.DataFrame, a0: pd.DataFrame) -> dict[str, float]:
    metrics = [
        "pump_m3",
        "time_over3_s",
        "time_over4_s",
        "time_over45_s",
        "time_over5_s",
        "idle_over5_s",
    ]
    out: dict[str, float] = {}
    for m in metrics:
        out[m] = float(selected[m].sum())
        out[f"delta_{m}"] = float(selected[m].sum() - a0[m].sum())
    out["p95_max_axis_mean"] = float(selected["p95_max_axis"].mean())
    out["delta_p95_max_axis_mean"] = float(selected["p95_max_axis"].mean() - a0["p95_max_axis"].mean())
    out["max_axis_max"] = float(selected["max_axis"].max())
    out["pump_saving_m3"] = float(a0["pump_m3"].sum() - selected["pump_m3"].sum())
    out["pump_saving_pct"] = 100.0 * out["pump_saving_m3"] / max(float(a0["pump_m3"].sum()), 1e-9)
    out["comfort4_per_pct"] = out["delta_time_over4_s"] / max(out["pump_saving_pct"], 1e-9)
    out["comfort3_per_pct"] = out["delta_time_over3_s"] / max(out["pump_saving_pct"], 1e-9)
    return out


def combine_casewise(a0: pd.DataFrame, budget: pd.DataFrame, selected_cases: set[str]) -> pd.DataFrame:
    bmap = budget.set_index("case")
    rows = []
    for _, r in a0.iterrows():
        c = str(r["case"])
        if c in selected_cases and c in bmap.index:
            rows.append(bmap.loc[c].to_dict())
        else:
            rows.append(r.to_dict())
    return pd.DataFrame(rows)


def stable_random_cases(cases: list[str], n: int, seed_text: str) -> set[str]:
    scores = []
    for c in cases:
        h = hashlib.sha256(f"{seed_text}:{c}".encode("utf-8")).hexdigest()
        scores.append((int(h[:12], 16), c))
    scores.sort()
    return {c for _, c in scores[:n]}


def evaluate_group(group_dir: Path, group: str, arm: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    a0 = load_arm_metrics(group_dir, "a0_baseline")
    budget = load_arm_metrics(group_dir, arm)
    feats = forecast_features(group_dir)
    merged = feats.merge(
        a0[["case", "pump_m3", "time_over5_s", "p95_max_axis"]].rename(
            columns={
                "pump_m3": "a0_pump_m3",
                "time_over5_s": "a0_time_over5_s",
                "p95_max_axis": "a0_p95_max_axis",
            }
        ),
        on="case",
        how="left",
    )
    budget_delta = budget[["case", "pump_m3", "time_over5_s", "p95_max_axis"]].merge(
        a0[["case", "pump_m3", "time_over5_s", "p95_max_axis"]],
        on="case",
        suffixes=("_budget", "_a0"),
    )
    budget_delta["budget_pump_saving_m3"] = budget_delta["pump_m3_a0"] - budget_delta["pump_m3_budget"]
    budget_delta["budget_delta_time5_s"] = budget_delta["time_over5_s_budget"] - budget_delta["time_over5_s_a0"]
    budget_delta["budget_delta_p95"] = budget_delta["p95_max_axis_budget"] - budget_delta["p95_max_axis_a0"]
    merged = merged.merge(
        budget_delta[["case", "budget_pump_saving_m3", "budget_delta_time5_s", "budget_delta_p95"]],
        on="case",
        how="left",
    )
    all_cases = list(a0["case"].astype(str))
    policies: dict[str, set[str]] = {
        "E_blind_all": set(all_cases),
        "E_gated_relief_any": set(merged.loc[merged["relief_frac"] > 0.0, "case"].astype(str)),
        "E_gated_relief_persistent": set(merged.loc[merged["relief_frac"] >= 0.10, "case"].astype(str)),
        "E_gated_decay_score_pos": set(merged.loc[merged["regime_score"] > 0.15, "case"].astype(str)),
        "E_gated_top25pct_score": set(
            merged.sort_values("regime_score", ascending=False).head(max(1, int(round(0.25 * len(merged)))))["case"].astype(str)
        ),
        "E_gated_top50pct_score": set(
            merged.sort_values("regime_score", ascending=False).head(max(1, int(round(0.50 * len(merged)))))["case"].astype(str)
        ),
    }
    rows = []
    for name, selected_cases in policies.items():
        selected = combine_casewise(a0, budget, selected_cases)
        row = aggregate(selected, a0)
        row.update(
            {
                "group": group,
                "arm": arm,
                "policy": name,
                "selected_cases": len(selected_cases),
                "selected_case_share": len(selected_cases) / max(len(all_cases), 1),
                "control_type": "forecast_gated" if name.startswith("E_gated") else "blind",
            }
        )
        rows.append(row)
        if name.startswith("E_gated"):
            n = len(selected_cases)
            rand_rows = []
            for i in range(50):
                rand_cases = stable_random_cases(all_cases, n, f"{group}:{arm}:{name}:{i}")
                rand_selected = combine_casewise(a0, budget, rand_cases)
                rr = aggregate(rand_selected, a0)
                rand_rows.append(rr)
            rand = pd.DataFrame(rand_rows)
            row = aggregate(combine_casewise(a0, budget, stable_random_cases(all_cases, n, f"{group}:{arm}:{name}:fixed")), a0)
            row.update(
                {
                    "group": group,
                    "arm": arm,
                    "policy": f"{name}_scrambled_fixed",
                    "selected_cases": n,
                    "selected_case_share": n / max(len(all_cases), 1),
                    "control_type": "scrambled",
                    "random_pump_saving_pct_mean": float(rand["pump_saving_pct"].mean()),
                    "random_delta_time_over4_s_mean": float(rand["delta_time_over4_s"].mean()),
                    "random_delta_time_over5_s_mean": float(rand["delta_time_over5_s"].mean()),
                    "random_delta_p95_mean": float(rand["delta_p95_max_axis_mean"].mean()),
                }
            )
            rows.append(row)
    return pd.DataFrame(rows), merged


def write_summary(out_dir: Path, policy: pd.DataFrame, feats: pd.DataFrame) -> None:
    paper = out_dir / "paper_ready"
    paper.mkdir(parents=True, exist_ok=True)
    raw = out_dir / "raw_tables"
    raw.mkdir(parents=True, exist_ok=True)
    policy.to_csv(raw / "virtual_forecast_regime_policy_table.csv", index=False)
    feats.to_csv(raw / "virtual_forecast_regime_case_features.csv", index=False)

    def compact_table(df: pd.DataFrame) -> str:
        cols = [
            "group",
            "arm",
            "policy",
            "selected_cases",
            "pump_saving_pct",
            "delta_time_over3_s",
            "delta_time_over4_s",
            "delta_time_over5_s",
            "delta_idle_over5_s",
            "delta_p95_max_axis_mean",
            "comfort4_per_pct",
        ]
        d = df[cols].copy()
        rows = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
        for _, r in d.iterrows():
            vals = []
            for c in cols:
                v = r[c]
                vals.append(f"{v:.3f}" if isinstance(v, float) else str(v))
            rows.append("| " + " | ".join(vals) + " |")
        return "\n".join(rows)

    best = (
        policy[policy["control_type"] == "forecast_gated"]
        .sort_values(["pump_saving_pct", "comfort4_per_pct"], ascending=[False, True])
        .head(8)
    )
    lines = [
        "# Virtual Forecast-Regime Gate Decisive Audit",
        "",
        "This read-only audit composes existing A0 and aggressive budget outputs case-by-case. It tests whether learned forecast regime selection is directionally smarter than blind aggressive relaxation before implementing a heavier controller gate.",
        "",
        "## Best Forecast-Gated Points",
        "",
        compact_table(best),
        "",
        "## All Policies",
        "",
        compact_table(policy),
        "",
        "## Interpretation Guide",
        "",
        "- If a forecast-gated policy reaches similar pump saving with lower `delta_time_over3/4/5` and lower `delta_p95` than blind or scrambled controls, forecast regime selection is useful.",
        "- If forecast-gated policies save far less pump or have similar comfort cost to scrambled controls, the 20% result remains a generic comfort relaxation result.",
        "",
    ]
    # Automated rough verdict.
    gated = policy[policy["control_type"] == "forecast_gated"].copy()
    blind = policy[policy["policy"] == "E_blind_all"].copy()
    verdict = "NO_GO"
    if not gated.empty and not blind.empty:
        # A useful point should clear 10% and have lower time>4 cost per pct than blind.
        for _, g in gated.iterrows():
            b = blind[(blind["group"] == g["group"]) & (blind["arm"] == g["arm"])]
            if b.empty:
                continue
            b0 = b.iloc[0]
            if (
                float(g["pump_saving_pct"]) >= 10.0
                and float(g["comfort4_per_pct"]) < float(b0["comfort4_per_pct"])
                and float(g["delta_time_over5_s"]) <= float(b0["delta_time_over5_s"])
            ):
                verdict = "CONDITIONAL_GO"
                break
    lines += [
        "## Automated Rough Verdict",
        "",
        f"`{verdict}`",
        "",
        "This is not a final paper verdict; it is a fast gate deciding whether to spend time on a real forecast-gated controller implementation.",
        "",
    ]
    text = "\n".join(lines)
    (paper / "virtual_forecast_regime_gate_summary.md").write_text(text)
    (out_dir / "decision.md").write_text(text)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--root",
        default="outputs/wind_prediction/economy_pump_budget_closed_loop_probe_v1",
    )
    ap.add_argument("--groups", nargs="+", default=["guard10_3600", "broader20_3600"])
    ap.add_argument("--arms", nargs="+", default=["budget120", "budget100"])
    ap.add_argument(
        "--out-dir",
        default="outputs/wind_prediction/economy_pump_budget_closed_loop_probe_v1/forecast_regime_gate_virtual_v1",
    )
    args = ap.parse_args()
    root = Path(args.root)
    policies = []
    features = []
    for group in args.groups:
        group_dir = root / group
        for arm in args.arms:
            if not (group_dir / arm).exists:
                continue
            if not (group_dir / arm).exists():
                continue
            p, f = evaluate_group(group_dir, group, arm)
            policies.append(p)
            f["group"] = group
            f["arm"] = arm
            features.append(f)
    policy = pd.concat(policies, ignore_index=True) if policies else pd.DataFrame()
    feats = pd.concat(features, ignore_index=True) if features else pd.DataFrame()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_summary(out, policy, feats)
    print(f"Wrote {len(policy)} policy rows to {out / 'raw_tables' / 'virtual_forecast_regime_policy_table.csv'}")
    print(f"Wrote summary to {out / 'decision.md'}")


if __name__ == "__main__":
    main()
