#!/usr/bin/env python3
"""Mine forecast-shape regimes for regime-conditioned ballast pump saving.

This is a read-only data-mining pass.  It uses actual future wind sequences as
oracle labels to define candidate regimes, estimate prevalence, and export
representative casebooks.  It does not claim learned runtime recognition; that
must be validated separately for any promoted strategy.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


DATASET_DEFAULT = (
    "data/processed/wind_ml_10min/"
    "ballast_decision_fino1_meteo_aux_h240_f120_v1"
)


def circular_diff_deg(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (a - b + 180.0) % 360.0 - 180.0


def block_means(x: np.ndarray) -> np.ndarray:
    """Return six 20-minute block means from a 12-step 10-minute sequence."""
    return x.reshape(len(x), 6, 2).mean(axis=2)


def classify_features(speed: np.ndarray, direction: np.ndarray, uv: np.ndarray) -> pd.DataFrame:
    b = block_means(speed)
    u_b = block_means(uv[:, :, 0])
    v_b = block_means(uv[:, :, 1])
    early = speed[:, :4]
    near = speed[:, :6]
    far = speed[:, 6:]
    early_max = early.max(axis=1)
    near_max = near.max(axis=1)
    far_max = far.max(axis=1)
    late_mean = speed[:, 8:].mean(axis=1)
    first = speed[:, 0]
    early_rise = early_max - first
    peak_to_late_drop = early_max - late_mean
    near_range = near.max(axis=1) - near.min(axis=1)
    far_range = far.max(axis=1) - far.min(axis=1)
    b_range_0_60 = b[:, :3].max(axis=1) - b[:, :3].min(axis=1)
    b_slope_0_60 = b[:, 2] - b[:, 0]
    b_slope_60_120 = b[:, 5] - b[:, 3]
    min_speed = speed.min(axis=1)
    mean_speed = speed.mean(axis=1)
    max_speed = speed.max(axis=1)

    dir_shift = np.abs(circular_diff_deg(direction, direction[:, [0]])).max(axis=1)
    near_dir_shift = np.abs(circular_diff_deg(direction[:, :6], direction[:, [0]])).max(axis=1)
    far_dir_shift = np.abs(circular_diff_deg(direction[:, 6:], direction[:, [5]])).max(axis=1)
    dot_early_late = u_b[:, 0] * u_b[:, 5] + v_b[:, 0] * v_b[:, 5]
    vec_norm_early = np.sqrt(u_b[:, 0] ** 2 + v_b[:, 0] ** 2)
    vec_norm_late = np.sqrt(u_b[:, 5] ** 2 + v_b[:, 5] ** 2)
    reversal_score = dot_early_late / np.maximum(vec_norm_early * vec_norm_late, 1e-6)

    # Forecast-shape labels.  Thresholds are intentionally simple and
    # engineering-readable; they define a mining taxonomy, not a final learned
    # classifier.
    high = max_speed >= 19.25
    near_high = near_max >= 18.0
    far_high = far_max >= 19.25
    calm = max_speed < 12.0
    low_variation = (max_speed - min_speed) < 2.5
    reversal = (dir_shift >= 45.0) | (reversal_score < 0.35)
    reintensify = (b[:, :3].max(axis=1) >= 17.0) & (b[:, 2] < b[:, :2].max(axis=1) - 2.0) & (
        b[:, 4:].max(axis=1) >= b[:, 2] + 3.0
    )
    fast_decay = near_high & (peak_to_late_drop >= 5.0) & (dir_shift < 45.0) & ~reintensify
    soft_decay = near_high & (peak_to_late_drop >= 2.5) & (peak_to_late_drop < 5.0) & (dir_shift < 45.0) & ~reintensify
    decay_low_pump_proxy = (max_speed < 16.0) & (peak_to_late_drop >= 3.0) & (dir_shift < 45.0)
    plateau = (mean_speed >= 17.0) & (b_range_0_60 <= 2.0) & (abs(b_slope_0_60) <= 1.0) & (dir_shift < 35.0)
    slow_decay = (mean_speed >= 16.0) & (peak_to_late_drop >= 1.0) & (peak_to_late_drop < 3.5) & ~plateau
    lowrisk_redundant = (max_speed < 12.0) & (dir_shift < 25.0) & (near_range < 2.0)
    far_only = (near_max < 15.0) & far_high
    sustained_high = high & (peak_to_late_drop < 2.0) & ~plateau & ~reversal
    gusty_oscillation = (max_speed >= 15.0) & (
        ((speed[:, 1:-1] > speed[:, :-2]) & (speed[:, 1:-1] > speed[:, 2:])).sum(axis=1) >= 2
    )

    primary = np.full(len(speed), "neutral_untyped", dtype=object)
    priority: list[tuple[str, np.ndarray]] = [
        ("direction_reversal_boundary", reversal),
        ("reintensification_boundary", reintensify),
        ("transient_peak_fast_decay", fast_decay),
        ("transient_peak_soft_decay", soft_decay),
        ("residual_high_plateau", plateau),
        ("residual_high_slow_decay", slow_decay),
        ("far_only_h120_advisory", far_only),
        ("sustained_high_safety_event", sustained_high),
        ("lowrisk_stable_redundant_candidate", lowrisk_redundant),
        ("gusty_oscillation_candidate", gusty_oscillation),
        ("transient_decay_low_pump_proxy", decay_low_pump_proxy),
        ("quiet_low_opportunity", calm & low_variation),
    ]
    assigned = np.zeros(len(speed), dtype=bool)
    for name, mask in priority:
        take = mask & ~assigned
        primary[take] = name
        assigned |= take

    out = pd.DataFrame(
        {
            "primary_regime": primary,
            "future_speed_first_ms": first,
            "future_speed_max_ms": max_speed,
            "future_speed_mean_ms": mean_speed,
            "early_max_ms": early_max,
            "near_max_ms": near_max,
            "far_max_ms": far_max,
            "late_mean_ms": late_mean,
            "early_rise_ms": early_rise,
            "peak_to_late_drop_ms": peak_to_late_drop,
            "near_range_ms": near_range,
            "far_range_ms": far_range,
            "block_range_0_60_ms": b_range_0_60,
            "block_slope_0_60_ms": b_slope_0_60,
            "block_slope_60_120_ms": b_slope_60_120,
            "dir_shift_abs_max_deg": dir_shift,
            "near_dir_shift_abs_max_deg": near_dir_shift,
            "far_dir_shift_abs_max_deg": far_dir_shift,
            "vector_reversal_cosine": reversal_score,
            "flag_fast_decay": fast_decay,
            "flag_soft_decay": soft_decay,
            "flag_plateau": plateau,
            "flag_slow_decay": slow_decay,
            "flag_reversal": reversal,
            "flag_reintensification": reintensify,
            "flag_lowrisk": lowrisk_redundant,
            "flag_far_only": far_only,
            "flag_gusty_oscillation": gusty_oscillation,
        }
    )
    for i in range(6):
        out[f"speed_block_{i}_mean_ms"] = b[:, i]
    return out


def episode_summary(df: pd.DataFrame) -> pd.DataFrame:
    work = df.sort_values("future_start").copy()
    work["future_start"] = pd.to_datetime(work["future_start"])
    rows = []
    for regime, g in work.groupby("primary_regime", sort=False):
        g = g.sort_values("future_start")
        dt = g["future_start"].diff().dt.total_seconds().fillna(600)
        episode_id = (dt > 900).cumsum()
        sizes = g.groupby(episode_id).size()
        rows.append(
            {
                "primary_regime": regime,
                "episode_count": int(len(sizes)),
                "median_episode_rows": float(sizes.median()),
                "p90_episode_rows": float(sizes.quantile(0.9)),
                "max_episode_rows": int(sizes.max()),
                "median_episode_minutes": float(sizes.median() * 10),
                "max_episode_minutes": float(sizes.max() * 10),
            }
        )
    return pd.DataFrame(rows)


def strategy_matrix() -> pd.DataFrame:
    rows = [
        (
            "transient_peak_fast_decay",
            "Mature candidate",
            "episode_auto_v1",
            "Forecast sees a near high/rising load followed by decay; avoid chasing the temporary peak.",
            "Matched 24-case run: 18.13% pump saving, with disclosed safety-margin cost.",
            "Run boundary abstention once before final promotion.",
        ),
        (
            "transient_peak_soft_decay",
            "Support / conservative use",
            "episode_auto_v1 with stricter release",
            "Same mechanism as fast decay, but weaker drop so less confidence.",
            "Merge into fast-decay family unless more cases show separate behavior.",
            "Do not create a separate branch yet.",
        ),
        (
            "residual_high_plateau",
            "High-potential candidate",
            "plateau pursuit suppression or operator Pareto budget",
            "High but steady load can create target/pump chatter; avoid continuous small pursuit.",
            "Existing budget100 plateau evidence suggests large potential, but recognition is not mature.",
            "Do one separability/recognition pass before any controller branch.",
        ),
        (
            "residual_high_slow_decay",
            "Boundary candidate",
            "early-release economy hold, not automatic yet",
            "Some pump can be saved, but slow-changing high load risks catch-up.",
            "Treat as boundary until plateau-vs-boundary detection is better.",
            "No threshold tuning on current small set.",
        ),
        (
            "lowrisk_stable_redundant_candidate",
            "Sparse candidate",
            "refresh cooldown / refill suppression",
            "Low future pressure and safe posture may make refill redundant.",
            "Existing evidence is sparse and top-case dominated.",
            "Mine more samples; no automatic branch yet.",
        ),
        (
            "gusty_oscillation_candidate",
            "New candidate to screen",
            "anti-chatter target smoothing",
            "Multiple short peaks may induce repeated target chasing.",
            "No closed-loop evidence yet; promising only if A0 pump opportunity exists.",
            "Read-only casebook first; short smoke only if pump opportunity is high.",
        ),
        (
            "far_only_h120_advisory",
            "Advisory",
            "supervisory warning",
            "Risk is outside 0-60min execution window; h120 helps awareness, not pump control.",
            "Already consistent with horizon-boundary results.",
            "Keep in advisory layer.",
        ),
        (
            "direction_reversal_boundary",
            "Veto boundary",
            "release/abstain",
            "Holding through reversal creates catch-up/fallback risk.",
            "Use to test abstention.",
            "Not a saving regime.",
        ),
        (
            "reintensification_boundary",
            "Veto boundary",
            "release/abstain",
            "Initial relief is followed by a second event; do not stay relaxed.",
            "Use as boundary examples.",
            "Not a saving regime.",
        ),
        (
            "sustained_high_safety_event",
            "Safety",
            "v1.6 hard floor",
            "Real high load must be handled by safety recovery.",
            "Economy saving should not own this state.",
            "No economy branch.",
        ),
        (
            "quiet_low_opportunity",
            "No action",
            "baseline",
            "There is little pump to save.",
            "Not useful for 20% claim.",
            "No branch.",
        ),
        (
            "neutral_untyped",
            "Mine later",
            "none",
            "No clear shape/action mechanism yet.",
            "Needs pump-opportunity evidence before strategy design.",
            "Do not tune blindly.",
        ),
    ]
    return pd.DataFrame(
        rows,
        columns=[
            "primary_regime",
            "status",
            "proposed_strategy",
            "mechanism",
            "current_evidence",
            "next_action",
        ],
    )


def representative_cases(df: pd.DataFrame, regime: str, n: int) -> pd.DataFrame:
    g = df[df["primary_regime"] == regime].copy()
    if g.empty:
        return g
    # Prefer stronger, clearer examples but keep samples spaced by at least one
    # hour to avoid near-duplicate overlapping windows.
    if "decay" in regime:
        g["score"] = g["peak_to_late_drop_ms"] + 0.2 * g["near_max_ms"] - 0.03 * g["dir_shift_abs_max_deg"]
    elif "plateau" in regime:
        g["score"] = g["future_speed_mean_ms"] - g["block_range_0_60_ms"] - 0.03 * g["dir_shift_abs_max_deg"]
    elif "lowrisk" in regime or "quiet" in regime:
        g["score"] = -g["future_speed_max_ms"] - 0.03 * g["dir_shift_abs_max_deg"]
    elif "reversal" in regime:
        g["score"] = g["dir_shift_abs_max_deg"] - 20.0 * g["vector_reversal_cosine"]
    elif "reintensification" in regime:
        g["score"] = g["far_max_ms"] - g["speed_block_2_mean_ms"]
    elif "gusty" in regime:
        g["score"] = g["future_speed_max_ms"] + g["near_range_ms"]
    else:
        g["score"] = g["future_speed_max_ms"]
    g = g.sort_values("score", ascending=False)
    keep = []
    taken_times: list[pd.Timestamp] = []
    for _, row in g.iterrows():
        t = pd.to_datetime(row["future_start"])
        if all(abs((t - x).total_seconds()) >= 3600 for x in taken_times):
            keep.append(row)
            taken_times.append(t)
        if len(keep) >= n:
            break
    if not keep:
        return g.head(n)
    return pd.DataFrame(keep)


def write_casebook(df: pd.DataFrame, path: Path) -> None:
    out = pd.DataFrame(
        {
            "case_id": [f"{path.stem}_{i+1:02d}" for i in range(len(df))],
            "timestamp": df["future_start"].astype(str).values,
            "label": (
                df["primary_regime"].astype(str)
                + " | max="
                + df["future_speed_max_ms"].round(1).astype(str)
                + " | drop="
                + df["peak_to_late_drop_ms"].round(1).astype(str)
                + " | dir="
                + df["dir_shift_abs_max_deg"].round(0).astype(int).astype(str)
            ),
        }
    )
    out.to_csv(path, index=False)


def process_split(dataset: Path, split: str, out_dir: Path, max_cases: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    idx_all = pd.read_csv(dataset / "sample_index.csv.gz")
    idx = idx_all[idx_all["split"] == split].reset_index(drop=True)
    y_speed_dir = np.load(dataset / f"y_speed_dir_raw_{split}.npy", mmap_mode="r")
    y_uv = np.load(dataset / f"y_uv_raw_{split}.npy", mmap_mode="r")
    features = classify_features(y_speed_dir[:, :, 0], y_speed_dir[:, :, 1], y_uv)
    idx_base = idx.drop(columns=[c for c in idx.columns if c in features.columns], errors="ignore")
    full = pd.concat([idx_base, features], axis=1)

    summary = (
        full.groupby("primary_regime")
        .agg(
            rows=("primary_regime", "size"),
            future_speed_mean_ms=("future_speed_mean_ms", "mean"),
            future_speed_max_median_ms=("future_speed_max_ms", "median"),
            early_max_median_ms=("early_max_ms", "median"),
            peak_to_late_drop_median_ms=("peak_to_late_drop_ms", "median"),
            dir_shift_median_deg=("dir_shift_abs_max_deg", "median"),
        )
        .reset_index()
    )
    summary["row_pct"] = 100.0 * summary["rows"] / len(full)
    eps = episode_summary(full)
    summary = summary.merge(eps, on="primary_regime", how="left")
    summary = summary.sort_values("rows", ascending=False)

    raw = out_dir / "raw_tables"
    case_dir = raw / f"casebooks_{split}"
    case_dir.mkdir(parents=True, exist_ok=True)
    full.to_csv(raw / f"regime_mining_v3_{split}_all_rows.csv", index=False)
    summary.to_csv(raw / f"regime_mining_v3_{split}_prevalence.csv", index=False)
    for regime in summary["primary_regime"]:
        reps = representative_cases(full, regime, max_cases)
        reps.to_csv(case_dir / f"{regime}_features.csv", index=False)
        write_casebook(reps, case_dir / f"{regime}_cases.csv")
    return full, summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-dir", default=DATASET_DEFAULT)
    ap.add_argument("--out-dir", default="outputs/wind_prediction/regime_conditioned_policy_development_v1/regime_mining_v3")
    ap.add_argument("--splits", nargs="+", default=["test", "validation"])
    ap.add_argument("--max-cases", type=int, default=24)
    args = ap.parse_args()

    dataset = Path(args.dataset_dir)
    out_dir = Path(args.out_dir)
    raw = out_dir / "raw_tables"
    paper = out_dir / "paper_ready"
    raw.mkdir(parents=True, exist_ok=True)
    paper.mkdir(parents=True, exist_ok=True)

    all_summaries = []
    for split in args.splits:
        _, summary = process_split(dataset, split, out_dir, args.max_cases)
        summary.insert(0, "split", split)
        all_summaries.append(summary)
    combined = pd.concat(all_summaries, ignore_index=True)
    combined.to_csv(raw / "regime_mining_v3_prevalence_all_splits.csv", index=False)

    strat = strategy_matrix()
    strat.to_csv(raw / "regime_mining_v3_strategy_matrix.csv", index=False)

    # Markdown summary.
    lines = [
        "# Regime Mining v3 Summary",
        "",
        "This read-only scan defines oracle/actual future-shape regimes from the wind dataset.  It estimates prevalence and exports representative casebooks.  It does not by itself prove learned runtime recognition or closed-loop pump saving.",
        "",
        "## Prevalence by split",
        "",
        "| split | regime | rows | row % | episodes | median episode min | max episode min |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, r in combined.sort_values(["split", "rows"], ascending=[True, False]).iterrows():
        lines.append(
            f"| {r['split']} | {r['primary_regime']} | {int(r['rows'])} | {r['row_pct']:.2f}% | "
            f"{int(r.get('episode_count', 0) or 0)} | {r.get('median_episode_minutes', 0):.0f} | {r.get('max_episode_minutes', 0):.0f} |"
        )
    lines += [
        "",
        "## Strategy status",
        "",
        "| regime | status | proposed strategy | next action |",
        "| --- | --- | --- | --- |",
    ]
    for _, r in strat.iterrows():
        lines.append(f"| {r.primary_regime} | {r.status} | {r.proposed_strategy} | {r.next_action} |")
    lines += [
        "",
        "## Decision logic",
        "",
        "- Promote a regime only if it has runtime-observable forecast shape, material pump opportunity, a clear control mechanism, and a matched-horizon closed-loop result.",
        "- Treat reversal, re-intensification, and hard-floor/recovery states as veto or safety regimes, not pump-saving targets.",
        "- Use the exported casebooks for short smoke tests before any long casebook validation.",
    ]
    (paper / "regime_mining_v3_summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"WROTE {raw / 'regime_mining_v3_prevalence_all_splits.csv'}")
    print(f"WROTE {raw / 'regime_mining_v3_strategy_matrix.csv'}")
    print(f"WROTE {paper / 'regime_mining_v3_summary.md'}")


if __name__ == "__main__":
    main()
