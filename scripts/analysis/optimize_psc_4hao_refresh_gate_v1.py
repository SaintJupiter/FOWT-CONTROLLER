#!/usr/bin/env python3
"""Optimize a strict case-level gate for PSC 4hao refresh_on.

This is a post-run analysis over the Stage-A 6h refresh validation outputs.  It
does not claim to be a deployable classifier.  It answers a narrower question:
if refresh_on is only allowed on cases where it saves pump without adding
fallback or time>5 debt versus the current-forecast-adaptive baseline, how much
safe value remains and which regimes should be pursued next?
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_ROOT = Path("outputs/wind_prediction/psc_selector_mixed_pool_v1/psc_4hao_refresh_validation_6h_v1")
POOLS = ("mixed24", "risky20")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    return parser.parse_args()


def _load(root: Path, pool: str) -> pd.DataFrame:
    path = root / pool / "stage_a_refresh_on_case_deltas.csv"
    df = pd.read_csv(path)
    df["pool"] = pool
    df["mixed_regime"] = df["label"].str.extract(r"mixed_regime=([^|,]+)")[0].str.strip()
    df["fine_regime"] = df["label"].str.extract(r"fine_regime=([^|,]+)")[0].str.strip()
    df["split"] = df["label"].str.extract(r"split=([^|,]+)")[0].str.strip()
    return df


def _policy_rows(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["accept_refresh_on"] = (
        (out["saved_m3"] > 0.0)
        & (out["d_time_gt5_s"] <= 0.0)
        & (out["d_fallback_s"] <= 0.0)
    )
    out["policy_arm"] = np.where(out["accept_refresh_on"], "refresh_on", "baseline")
    out["policy_pump_m3"] = np.where(out["accept_refresh_on"], out["pump_m3"], out["pump_m3_baseline"])
    out["policy_saved_m3"] = out["pump_m3_baseline"] - out["policy_pump_m3"]
    out["policy_d_time_gt3_s"] = np.where(out["accept_refresh_on"], out["d_time_gt3_s"], 0.0)
    out["policy_d_time_gt4_s"] = np.where(out["accept_refresh_on"], out["d_time_gt4_s"], 0.0)
    out["policy_d_time_gt5_s"] = np.where(out["accept_refresh_on"], out["d_time_gt5_s"], 0.0)
    out["policy_d_fallback_s"] = np.where(out["accept_refresh_on"], out["d_fallback_s"], 0.0)
    out["reject_reason"] = "accepted"
    out.loc[out["saved_m3"] <= 0.0, "reject_reason"] = "no_pump_saving"
    out.loc[(out["saved_m3"] > 0.0) & (out["d_time_gt5_s"] > 0.0), "reject_reason"] = "adds_time_gt5"
    out.loc[
        (out["saved_m3"] > 0.0) & (out["d_time_gt5_s"] <= 0.0) & (out["d_fallback_s"] > 0.0),
        "reject_reason",
    ] = "adds_fallback"
    return out


def _summary(policy: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for pool, part in policy.groupby("pool", sort=False):
        baseline = float(part["pump_m3_baseline"].sum())
        refresh = float(part["pump_m3"].sum())
        gated = float(part["policy_pump_m3"].sum())
        rows.append(
            {
                "pool": pool,
                "cases": int(len(part)),
                "accepted_cases": int(part["accept_refresh_on"].sum()),
                "baseline_pump_m3": baseline,
                "refresh_on_pump_m3": refresh,
                "gated_pump_m3": gated,
                "refresh_on_saving_pct": 100.0 * (baseline - refresh) / max(baseline, 1e-9),
                "gated_saving_pct": 100.0 * (baseline - gated) / max(baseline, 1e-9),
                "refresh_on_d_time_gt5_s": float(part["d_time_gt5_s"].sum()),
                "gated_d_time_gt5_s": float(part["policy_d_time_gt5_s"].sum()),
                "refresh_on_d_fallback_s": float(part["d_fallback_s"].sum()),
                "gated_d_fallback_s": float(part["policy_d_fallback_s"].sum()),
                "accepted_saved_m3": float(part["policy_saved_m3"].sum()),
            }
        )
    return pd.DataFrame(rows)


def _group_summary(policy: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["pool", "mixed_regime", "split"]
    rows = []
    for keys, part in policy.groupby(group_cols, dropna=False, sort=False):
        baseline = float(part["pump_m3_baseline"].sum())
        gated = float(part["policy_pump_m3"].sum())
        rows.append(
            {
                "pool": keys[0],
                "mixed_regime": keys[1],
                "split": "" if pd.isna(keys[2]) else keys[2],
                "cases": int(len(part)),
                "accepted_cases": int(part["accept_refresh_on"].sum()),
                "gated_saving_pct": 100.0 * (baseline - gated) / max(baseline, 1e-9),
                "gated_saved_m3": baseline - gated,
                "gated_d_time_gt5_s": float(part["policy_d_time_gt5_s"].sum()),
                "gated_d_fallback_s": float(part["policy_d_fallback_s"].sum()),
                "rejected_adds_time_gt5": int((part["reject_reason"] == "adds_time_gt5").sum()),
                "rejected_adds_fallback": int((part["reject_reason"] == "adds_fallback").sum()),
                "rejected_no_saving": int((part["reject_reason"] == "no_pump_saving").sum()),
            }
        )
    return pd.DataFrame(rows)


def _md_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_empty_"
    lines = [
        "| " + " | ".join(df.columns) + " |",
        "| " + " | ".join(["---"] * len(df.columns)) + " |",
    ]
    for rec in df.to_dict("records"):
        vals = []
        for col in df.columns:
            val = rec[col]
            if isinstance(val, (float, np.floating)):
                vals.append(f"{float(val):.4g}")
            else:
                vals.append(str(val))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    root = args.root
    root.mkdir(parents=True, exist_ok=True)
    policy = pd.concat([_policy_rows(_load(root, pool)) for pool in POOLS], ignore_index=True)
    summary = _summary(policy)
    groups = _group_summary(policy)

    policy.to_csv(root / "refresh_on_strict_gate_case_policy.csv", index=False)
    summary.to_csv(root / "refresh_on_strict_gate_summary.csv", index=False)
    groups.to_csv(root / "refresh_on_strict_gate_group_summary.csv", index=False)

    accepted = policy[policy["accept_refresh_on"]].copy()
    rejected = policy[~policy["accept_refresh_on"]].copy()

    lines = [
        "# PSC 4hao refresh_on strict gate optimization v1",
        "",
        "This is a post-run strict gate over the Stage-A 6h validation outputs.",
        "Accept `refresh_on` only when the case saves pump and does not add time>5 or fallback versus baseline; otherwise fall back to baseline.",
        "",
        "## Pool summary",
        "",
        _md_table(summary),
        "",
        "## Regime summary",
        "",
        _md_table(groups),
        "",
        "## Accepted cases",
        "",
        _md_table(
            accepted[
                [
                    "pool",
                    "case_id",
                    "mixed_regime",
                    "split",
                    "saved_m3",
                    "saving_pct",
                    "d_time_gt5_s",
                    "d_fallback_s",
                    "p95_axis_deg",
                ]
            ]
        ),
        "",
        "## Rejected case counts",
        "",
        _md_table(
            rejected.groupby(["pool", "reject_reason"], sort=False)
            .size()
            .reset_index(name="cases")
        ),
        "",
        "## Decision",
        "",
        "- Aggregate `refresh_on` dominates `refresh_off`, so OFF-gate work should stop.",
        "- Full `refresh_on` is still not safe as an automatic global controller because rejected cases add fallback or time>5 debt.",
        "- The strict gate preserves a no-added-time>5/no-added-fallback operating subset and gives the next classifier target.",
        "- Next deployable step: learn an ex-ante accept/reject rule for these accepted cases, then rerun it on fresh 6h pools.",
        "",
    ]
    (root / "refresh_on_strict_gate_readout.md").write_text("\n".join(lines), encoding="utf-8")
    print(summary.to_string(index=False))
    print(root / "refresh_on_strict_gate_readout.md")


if __name__ == "__main__":
    main()
