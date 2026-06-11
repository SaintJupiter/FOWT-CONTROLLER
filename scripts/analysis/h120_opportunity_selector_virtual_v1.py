#!/usr/bin/env python3
"""Read-only opportunity-selector audit for aggressive pump-saving mode.

The blind budget arm can reach ~20% pump saving, but simple forecast-regime
gating did not explain it.  This script asks a narrower and cheaper question:
can observable controller-waste features (early pump use, round-trip/chatter,
target chasing) identify a deployable opportunity subset where aggressive
economy mode is worth using?

No controller runs are launched. Existing A0/budget outputs are composed
case-by-case, with random controls of the same subset size.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd


def case_id(path: Path) -> str:
    for suffix in (
        "_prediction_primary_econ_timeseries.csv",
        "_prediction_primary_econ_planner_log.csv",
    ):
        if path.name.endswith(suffix):
            return path.name[: -len(suffix)]
    return path.stem


def metrics_from_timeseries(path: Path) -> dict[str, float]:
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


def signed_rate_cols(df: pd.DataFrame) -> list[str]:
    cols = [c for c in ["pump_rate1_m3min", "pump_rate2_m3min", "pump_rate3_m3min"] if c in df.columns]
    if cols:
        return cols
    return [c for c in ["pump_target_rate1_m3min", "pump_target_rate2_m3min", "pump_target_rate3_m3min"] if c in df.columns]


def reversal_features(ts_path: Path, feature_window_s: float) -> dict[str, float]:
    df = pd.read_csv(ts_path, low_memory=False)
    if "t_s" in df.columns:
        df = df[df["t_s"].to_numpy(dtype=float) < float(feature_window_s)].copy()
    if df.empty:
        return {
            "feature_pump_m3": 0.0,
            "feature_roundtrip_proxy_m3": 0.0,
            "feature_reversal_count": 0.0,
            "feature_target_motion": 0.0,
            "feature_time_over4_s": 0.0,
            "feature_time_over5_s": 0.0,
        }
    rates = signed_rate_cols(df)
    roundtrip = 0.0
    reversals = 0
    for col in rates:
        r = df[col].to_numpy(dtype=float)
        pos = float(np.nansum(np.maximum(r, 0.0)) / 60.0)
        neg = float(np.nansum(np.maximum(-r, 0.0)) / 60.0)
        roundtrip += min(pos, neg)
        sign = np.sign(np.where(np.abs(r) > 0.05, r, 0.0))
        nz = sign[sign != 0]
        if nz.size > 1:
            reversals += int(np.sum(nz[1:] != nz[:-1]))
    pitch = df.get("pitch_deg", pd.Series(np.zeros(len(df)))).to_numpy(dtype=float)
    roll = df.get("roll_deg", pd.Series(np.zeros(len(df)))).to_numpy(dtype=float)
    max_axis = np.maximum(np.abs(pitch), np.abs(roll))
    target_motion = df.get("pump_target_motion_kg_s", pd.Series(np.zeros(len(df)))).to_numpy(dtype=float)
    pump = df.get("pump_total_rate_m3_min", pd.Series(np.zeros(len(df)))).to_numpy(dtype=float)
    return {
        "feature_pump_m3": float(np.nansum(np.abs(pump)) / 60.0),
        "feature_roundtrip_proxy_m3": roundtrip,
        "feature_reversal_count": float(reversals),
        "feature_target_motion": float(np.nanmean(np.abs(target_motion))) if len(target_motion) else 0.0,
        "feature_time_over4_s": float(np.nansum(max_axis > 4.0)),
        "feature_time_over5_s": float(np.nansum(max_axis > 5.0)),
    }


def forecast_features(log_path: Path, feature_window_s: float) -> dict[str, float]:
    df = pd.read_csv(log_path, low_memory=False)
    if "current_time_s" in df.columns:
        df = df[df["current_time_s"].to_numpy(dtype=float) < float(feature_window_s)].copy()
    if df.empty:
        return {
            "relief_frac": 0.0,
            "calm_frac": 0.0,
            "high_pressure_frac": 0.0,
            "reintensify_frac": 0.0,
            "mean_decay": 0.0,
            "target_refresh_count": 0.0,
        }
    b0 = df.get("raw_pressure_block0_norm", pd.Series(np.nan, index=df.index)).to_numpy(dtype=float)
    b1 = df.get("raw_pressure_block1_norm", pd.Series(np.nan, index=df.index)).to_numpy(dtype=float)
    b2 = df.get("raw_pressure_block2_norm", pd.Series(np.nan, index=df.index)).to_numpy(dtype=float)
    near_max = np.nanmax(np.vstack([b0, b1, b2]), axis=0)
    decay = b0 - b2
    relief = (b2 <= 0.7) & (decay >= 0.2)
    calm = near_max < 0.9
    high = near_max >= 0.9
    reintensify = (b2 >= 0.9) & (b2 > b0 + 0.05)
    refresh = df.get("prediction_primary_target_refreshed", pd.Series(np.zeros(len(df)))).to_numpy(dtype=float)
    event_reset = df.get("prediction_primary_event_reset", pd.Series(np.zeros(len(df)))).to_numpy(dtype=float)
    sustained = df.get("prediction_primary_sustained_active_recompute", pd.Series(np.zeros(len(df)))).to_numpy(dtype=float)
    return {
        "relief_frac": float(np.nanmean(relief)),
        "calm_frac": float(np.nanmean(calm)),
        "high_pressure_frac": float(np.nanmean(high)),
        "reintensify_frac": float(np.nanmean(reintensify)),
        "mean_decay": float(np.nanmean(decay)),
        "target_refresh_count": float(np.nansum(refresh) + np.nansum(event_reset) + np.nansum(sustained)),
    }


def load_arm_metrics(group_dir: Path, arm: str) -> pd.DataFrame:
    rows = []
    for p in sorted((group_dir / arm / "timeseries").glob("*_timeseries.csv")):
        r = metrics_from_timeseries(p)
        r["case"] = case_id(p)
        rows.append(r)
    return pd.DataFrame(rows)


def rank01(s: pd.Series) -> pd.Series:
    if s.nunique(dropna=False) <= 1:
        return pd.Series(np.zeros(len(s)), index=s.index)
    return s.rank(pct=True).fillna(0.0)


def build_features(group_dir: Path, feature_window_s: float) -> pd.DataFrame:
    rows = []
    ts_map = {case_id(p): p for p in sorted((group_dir / "a0_baseline" / "timeseries").glob("*_timeseries.csv"))}
    log_map = {case_id(p): p for p in sorted((group_dir / "a0_baseline" / "planner_logs").glob("*_planner_log.csv"))}
    for c, ts_path in ts_map.items():
        row = {"case": c}
        row.update(reversal_features(ts_path, feature_window_s))
        if c in log_map:
            row.update(forecast_features(log_map[c], feature_window_s))
        rows.append(row)
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["waste_score"] = (
        rank01(df["feature_pump_m3"])
        + rank01(df["feature_roundtrip_proxy_m3"])
        + 0.5 * rank01(df["feature_reversal_count"])
        + 0.5 * rank01(df["target_refresh_count"])
        + 0.25 * rank01(df["feature_target_motion"])
    )
    df["forecast_veto"] = (
        (df["reintensify_frac"] >= 0.20)
        | (df["high_pressure_frac"] >= 0.75)
        | (df["feature_time_over5_s"] >= 300.0)
    ).astype(int)
    df["opportunity_score"] = (
        df["waste_score"]
        + 0.5 * df["relief_frac"]
        + 0.25 * df["calm_frac"]
        + 0.25 * np.maximum(df["mean_decay"], 0.0)
        - 0.75 * df["reintensify_frac"]
        - 0.50 * df["high_pressure_frac"]
        - 0.50 * (df["feature_time_over5_s"] > 0).astype(float)
    )
    return df


def combine(a0: pd.DataFrame, budget: pd.DataFrame, selected_cases: set[str]) -> pd.DataFrame:
    b = budget.set_index("case")
    rows = []
    for _, r in a0.iterrows():
        c = str(r["case"])
        if c in selected_cases and c in b.index:
            x = b.loc[c].to_dict()
            x["selected"] = 1
            rows.append(x)
        else:
            x = r.to_dict()
            x["selected"] = 0
            rows.append(x)
    return pd.DataFrame(rows)


def aggregate(sel: pd.DataFrame, a0: pd.DataFrame, selected_cases: set[str], budget: pd.DataFrame) -> dict[str, float]:
    metrics = ["pump_m3", "time_over3_s", "time_over4_s", "time_over45_s", "time_over5_s", "idle_over5_s"]
    out: dict[str, float] = {}
    for m in metrics:
        out[m] = float(sel[m].sum())
        out[f"delta_{m}"] = float(sel[m].sum() - a0[m].sum())
    out["p95_max_axis_mean"] = float(sel["p95_max_axis"].mean())
    out["delta_p95_max_axis_mean"] = float(sel["p95_max_axis"].mean() - a0["p95_max_axis"].mean())
    out["max_axis_max"] = float(sel["max_axis"].max())
    out["pump_saving_m3"] = float(a0["pump_m3"].sum() - sel["pump_m3"].sum())
    out["pump_saving_pct"] = 100.0 * out["pump_saving_m3"] / max(float(a0["pump_m3"].sum()), 1e-9)
    out["comfort4_per_pct"] = out["delta_time_over4_s"] / max(out["pump_saving_pct"], 1e-9) if out["pump_saving_pct"] > 0 else np.nan
    # Opportunity-subset-only effect: useful for paper framing.
    if selected_cases:
        a0s = a0[a0["case"].isin(selected_cases)]
        bs = budget[budget["case"].isin(selected_cases)]
        if not a0s.empty and not bs.empty:
            out["subset_cases"] = float(len(selected_cases))
            out["subset_a0_pump_m3"] = float(a0s["pump_m3"].sum())
            out["subset_pump_saving_m3"] = float(a0s["pump_m3"].sum() - bs["pump_m3"].sum())
            out["subset_pump_saving_pct"] = 100.0 * out["subset_pump_saving_m3"] / max(float(a0s["pump_m3"].sum()), 1e-9)
            out["subset_delta_time_over5_s"] = float(bs["time_over5_s"].sum() - a0s["time_over5_s"].sum())
            out["subset_delta_time_over4_s"] = float(bs["time_over4_s"].sum() - a0s["time_over4_s"].sum())
            out["subset_delta_p95_mean"] = float(bs["p95_max_axis"].mean() - a0s["p95_max_axis"].mean())
    return out


def stable_random(cases: list[str], n: int, key: str) -> set[str]:
    scored = []
    for c in cases:
        h = hashlib.sha256(f"{key}:{c}".encode()).hexdigest()
        scored.append((int(h[:12], 16), c))
    scored.sort()
    return {c for _, c in scored[:n]}


def evaluate(group_dir: Path, group: str, arm: str, feature_window_s: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    a0 = load_arm_metrics(group_dir, "a0_baseline")
    budget = load_arm_metrics(group_dir, arm)
    feats = build_features(group_dir, feature_window_s)
    cases = list(a0["case"].astype(str))
    policies: dict[str, set[str]] = {}
    policies["blind_all"] = set(cases)
    n25 = max(1, int(round(0.25 * len(feats))))
    n50 = max(1, int(round(0.50 * len(feats))))
    policies["waste_top25"] = set(feats.sort_values("waste_score", ascending=False).head(n25)["case"].astype(str))
    policies["waste_top50"] = set(feats.sort_values("waste_score", ascending=False).head(n50)["case"].astype(str))
    eligible = feats[feats["forecast_veto"] == 0]
    policies["waste_veto_top25"] = set(eligible.sort_values("waste_score", ascending=False).head(n25)["case"].astype(str))
    policies["waste_veto_top50"] = set(eligible.sort_values("waste_score", ascending=False).head(n50)["case"].astype(str))
    policies["opportunity_top25"] = set(feats.sort_values("opportunity_score", ascending=False).head(n25)["case"].astype(str))
    policies["opportunity_top50"] = set(feats.sort_values("opportunity_score", ascending=False).head(n50)["case"].astype(str))
    rows = []
    for name, selected in policies.items():
        sel = combine(a0, budget, selected)
        row = aggregate(sel, a0, selected, budget)
        row.update({"group": group, "arm": arm, "policy": name, "control_type": "selector" if name != "blind_all" else "blind", "selected_cases": len(selected)})
        rows.append(row)
        if name != "blind_all":
            rand_rows = []
            for i in range(50):
                rset = stable_random(cases, len(selected), f"{group}:{arm}:{name}:{i}")
                rand_rows.append(aggregate(combine(a0, budget, rset), a0, rset, budget))
            rand = pd.DataFrame(rand_rows)
            fixed = stable_random(cases, len(selected), f"{group}:{arm}:{name}:fixed")
            rr = aggregate(combine(a0, budget, fixed), a0, fixed, budget)
            rr.update(
                {
                    "group": group,
                    "arm": arm,
                    "policy": f"{name}_random_fixed",
                    "control_type": "random_control",
                    "selected_cases": len(selected),
                    "random_pump_saving_pct_mean": float(rand["pump_saving_pct"].mean()),
                    "random_subset_pump_saving_pct_mean": float(rand.get("subset_pump_saving_pct", pd.Series([np.nan])).mean()),
                    "random_delta_time_over5_s_mean": float(rand["delta_time_over5_s"].mean()),
                    "random_delta_time_over4_s_mean": float(rand["delta_time_over4_s"].mean()),
                }
            )
            rows.append(rr)
    return pd.DataFrame(rows), feats


def write(out_dir: Path, policy: pd.DataFrame, feats: pd.DataFrame) -> None:
    raw = out_dir / "raw_tables"
    paper = out_dir / "paper_ready"
    raw.mkdir(parents=True, exist_ok=True)
    paper.mkdir(parents=True, exist_ok=True)
    policy.to_csv(raw / "opportunity_selector_policy_table.csv", index=False)
    feats.to_csv(raw / "opportunity_selector_case_features.csv", index=False)

    cols = [
        "group",
        "arm",
        "policy",
        "control_type",
        "selected_cases",
        "pump_saving_pct",
        "subset_pump_saving_pct",
        "delta_time_over4_s",
        "delta_time_over5_s",
        "delta_p95_max_axis_mean",
        "subset_delta_time_over5_s",
        "subset_delta_p95_mean",
    ]
    def md(df: pd.DataFrame) -> str:
        d = df[[c for c in cols if c in df.columns]].copy()
        rows = ["| " + " | ".join(d.columns) + " |", "| " + " | ".join(["---"] * len(d.columns)) + " |"]
        for _, r in d.iterrows():
            vals = []
            for c in d.columns:
                v = r[c]
                vals.append(f"{v:.3f}" if isinstance(v, float) and np.isfinite(v) else str(v))
            rows.append("| " + " | ".join(vals) + " |")
        return "\n".join(rows)

    selector = policy[policy["control_type"] == "selector"].copy()
    best_subset = selector.sort_values("subset_pump_saving_pct", ascending=False).head(12)
    best_full = selector.sort_values("pump_saving_pct", ascending=False).head(12)
    verdict = "NO_GO"
    if ((selector["subset_pump_saving_pct"] >= 20.0) & (selector["subset_delta_time_over5_s"] <= 0.0)).any():
        verdict = "CONDITIONAL_GO_SUBSET"
    if ((selector["pump_saving_pct"] >= 10.0) & (selector["delta_time_over5_s"] <= 0.0)).any():
        verdict = "CONDITIONAL_GO_FULLBOOK"
    lines = [
        "# Opportunity Selector Virtual Audit",
        "",
        "This audit uses only existing A0 and budget-arm outputs. It tests whether early observable controller-waste features can identify a pump-saving opportunity subset before we spend time implementing a real controller selector.",
        "",
        "## Best Opportunity-Subset Points",
        "",
        md(best_subset),
        "",
        "## Best Full-Book Points",
        "",
        md(best_full),
        "",
        "## Verdict",
        "",
        f"`{verdict}`",
        "",
        "Interpretation: subset success is not enough for a full-book headline, but it can support a defensible opportunity-regime result if the subset rule is declared up front and compared against random controls.",
        "",
    ]
    (out_dir / "decision.md").write_text("\n".join(lines))
    (paper / "opportunity_selector_summary.md").write_text("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="outputs/wind_prediction/economy_pump_budget_closed_loop_probe_v1")
    ap.add_argument("--groups", nargs="+", default=["guard10_3600", "broader20_3600"])
    ap.add_argument("--arms", nargs="+", default=["budget120", "budget100"])
    ap.add_argument("--feature-window-s", type=float, default=1200.0)
    ap.add_argument("--out-dir", default="outputs/wind_prediction/economy_pump_budget_closed_loop_probe_v1/opportunity_selector_virtual_v1")
    args = ap.parse_args()
    root = Path(args.root)
    policies = []
    features = []
    for group in args.groups:
        gd = root / group
        for arm in args.arms:
            if not (gd / arm).exists():
                continue
            p, f = evaluate(gd, group, arm, args.feature_window_s)
            policies.append(p)
            f["group"] = group
            f["arm"] = arm
            features.append(f)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    write(out, pd.concat(policies, ignore_index=True), pd.concat(features, ignore_index=True))
    print(f"Wrote {out / 'decision.md'}")


if __name__ == "__main__":
    main()
