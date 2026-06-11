#!/usr/bin/env python3
"""Profile the remaining neutral_untyped pool without tuning controllers.

This is a cheap, read-only shape profile.  It is *not* an outcome-labeled regime
proof because no fixed action has been evaluated on these rows yet.  Its only
purpose is deciding whether a future action-conditioned screen is worth doing,
and which candidate mechanisms should be tried first if so.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path("outputs/wind_prediction/regime_conditioned_policy_development_v1")
V3 = ROOT / "regime_mining_v3" / "raw_tables"
OUT = ROOT / "neutral_untyped_profile_v1"
OUT_RAW = OUT / "raw_tables"
OUT_PAPER = OUT / "paper_ready"


def load_rows() -> pd.DataFrame:
    parts = []
    for split in ("test", "validation"):
        path = V3 / f"regime_mining_v3_{split}_all_rows.csv"
        if path.exists():
            parts.append(pd.read_csv(path))
    if not parts:
        return pd.DataFrame()
    df = pd.concat(parts, ignore_index=True)
    return df[df["primary_regime"] == "neutral_untyped"].copy()


def assign_subtype(row: pd.Series) -> str:
    early = float(row.get("early_max_ms", 0.0))
    near = float(row.get("near_max_ms", 0.0))
    far = float(row.get("far_max_ms", 0.0))
    drop = float(row.get("peak_to_late_drop_ms", 0.0))
    early_rise = float(row.get("early_rise_ms", 0.0))
    near_range = float(row.get("near_range_ms", 0.0))
    far_range = float(row.get("far_range_ms", 0.0))
    dir_shift = float(row.get("dir_shift_abs_max_deg", 0.0))
    ramp_event = int(row.get("speed_ramp_ge_3ms", 0)) == 1
    attention = int(row.get("ballast_attention_event", 0)) == 1

    if ramp_event or attention or early_rise >= 3.0:
        return "neutral_ramp_or_event_onset"
    if dir_shift >= 35.0:
        return "neutral_direction_shift"
    if far >= near + 1.5:
        return "neutral_far_rise_watch"
    if early >= 14.0 and near_range <= 1.5 and far_range <= 1.8 and abs(drop) <= 1.0:
        return "neutral_moderate_high_steady"
    if early >= 12.0 and drop >= 1.0:
        return "neutral_mild_decay"
    if early <= 10.0:
        return "neutral_low_moderate"
    return "neutral_mixed_steady"


def main() -> None:
    OUT_RAW.mkdir(parents=True, exist_ok=True)
    OUT_PAPER.mkdir(parents=True, exist_ok=True)
    df = load_rows()
    if df.empty:
        raise SystemExit("no neutral_untyped rows found")
    df["neutral_subtype"] = df.apply(assign_subtype, axis=1)

    agg = (
        df.groupby(["split", "neutral_subtype"], dropna=False)
        .agg(
            rows=("neutral_subtype", "size"),
            episode_like_series=("series_id", "nunique"),
            early_max_median_ms=("early_max_ms", "median"),
            near_max_median_ms=("near_max_ms", "median"),
            far_max_median_ms=("far_max_ms", "median"),
            peak_to_late_drop_median_ms=("peak_to_late_drop_ms", "median"),
            dir_shift_median_deg=("dir_shift_abs_max_deg", "median"),
            future_speed_ramp_event_rate=("speed_ramp_ge_3ms", "mean"),
            ballast_attention_event_rate=("ballast_attention_event", "mean"),
        )
        .reset_index()
    )
    totals = df.groupby("split").size().rename("split_rows").reset_index()
    agg = agg.merge(totals, on="split", how="left")
    agg["within_neutral_pct"] = 100.0 * agg["rows"] / agg["split_rows"]

    mechanism = {
        "neutral_ramp_or_event_onset": (
            "boundary/safety candidate",
            "Do not economize first; use as negative control for event-onset abstention.",
        ),
        "neutral_direction_shift": (
            "boundary/veto candidate",
            "Do not economize without direction-aware proof; likely catch-up risk.",
        ),
        "neutral_far_rise_watch": (
            "h120 advisory candidate",
            "Good for supervisory far-risk monitoring, not near-horizon pump saving yet.",
        ),
        "neutral_moderate_high_steady": (
            "possible plateau-like candidate",
            "Only worth a future fixed-action screen if material A0 pump is shown.",
        ),
        "neutral_mild_decay": (
            "possible weak relief candidate",
            "Could be merged into relief/decay only after action-outcome labeling.",
        ),
        "neutral_low_moderate": (
            "low opportunity",
            "Likely little pump to save; no branch unless pump telemetry contradicts this.",
        ),
        "neutral_mixed_steady": (
            "untyped background",
            "No strategy without a fixed-action outcome label.",
        ),
    }
    agg["candidate_role"] = agg["neutral_subtype"].map(lambda x: mechanism.get(x, ("unknown", ""))[0])
    agg["recommended_next_check"] = agg["neutral_subtype"].map(lambda x: mechanism.get(x, ("", "unknown"))[1])

    df.to_csv(OUT_RAW / "neutral_untyped_profile_rows.csv", index=False)
    agg.to_csv(OUT_RAW / "neutral_untyped_subtype_summary.csv", index=False)

    cols = [
        "split",
        "neutral_subtype",
        "rows",
        "within_neutral_pct",
        "early_max_median_ms",
        "peak_to_late_drop_median_ms",
        "dir_shift_median_deg",
        "candidate_role",
        "recommended_next_check",
    ]
    rows = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in agg.sort_values(["split", "rows"], ascending=[True, False])[cols].iterrows():
        vals = []
        for col in cols:
            val = row[col]
            vals.append(f"{float(val):.2f}" if isinstance(val, float) else str(val))
        rows.append("| " + " | ".join(vals) + " |")

    md = [
        "# Neutral Untyped Profile v1",
        "",
        "This is a read-only shape profile of the remaining `neutral_untyped` pool. It is not a controller result and it is not a new automatic regime.",
        "",
        "## Summary",
        "",
        "\n".join(rows),
        "",
        "## Decision",
        "",
        "- The neutral pool is broad, but most subtypes are boundary/advisory/background rather than immediate pump-saving targets.",
        "- The only subtypes worth a future action-conditioned screen are `neutral_moderate_high_steady` and `neutral_mild_decay`.",
        "- The next proof must be fixed-action outcome labeling, not another shape-only taxonomy split.",
        "",
    ]
    text = "\n".join(md)
    (OUT / "decision.md").write_text(text, encoding="utf-8")
    (OUT_PAPER / "neutral_untyped_profile_summary.md").write_text(text, encoding="utf-8")
    print(agg.sort_values(["split", "rows"], ascending=[True, False]).to_string(index=False))
    print(f"\nWrote {OUT / 'decision.md'}")


if __name__ == "__main__":
    main()
