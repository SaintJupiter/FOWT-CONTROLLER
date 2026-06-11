#!/usr/bin/env python3
"""Build a path-policy distillation dataset from forced-prefix replay results.

This script is deliberately read-only with respect to controller behavior. It
does not train a model and does not run new simulations. It converts the
forced-prefix replay matrix into prefix-level samples that say which oracle-like
prefixes are safe to imitate, which are tradeoffs, and which should be avoided.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FORCED_DIR = REPO_ROOT / "outputs/wind_prediction/forced_prefix_replay_v1"
DEFAULT_OUT_DIR = REPO_ROOT / "outputs/wind_prediction/path_policy_distillation_dataset_v1"


RUN_RE = re.compile(
    r"^(?P<forecast>learned|oracle)_forecast_"
    r"(?P<prefix_source>learned|oracle)_prefix_len(?P<length>\d+)_"
    r"(?P<mode>final_action|raw_action|target_update)$"
)


def _safe_float(value: Any, default: float = np.nan) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if np.isfinite(out) else default


def _parse_run_label(label: str) -> dict[str, Any]:
    match = RUN_RE.match(str(label))
    if not match:
        return {
            "forecast_source": "",
            "prefix_source": "",
            "prefix_length": 0,
            "forced_mode": "",
        }
    out = match.groupdict()
    return {
        "forecast_source": out["forecast"],
        "prefix_source": out["prefix_source"],
        "prefix_length": int(out["length"]),
        "forced_mode": out["mode"],
    }


def _is_active(action: Any) -> bool:
    text = str(action)
    return text.startswith("active") or text == "pump_saving"


def _is_medium(action: Any) -> bool:
    return str(action) == "active_medium"


def _sequence_string(df: pd.DataFrame) -> str:
    if df.empty:
        return ""
    return ",".join(
        f"{int(row.bucket)}:{row.action}"
        for row in df.sort_values("bucket").itertuples(index=False)
    )


def _first_active_bucket(df: pd.DataFrame) -> float:
    if df.empty:
        return np.nan
    active = df[df["action"].map(_is_active)]
    return float(active["bucket"].min()) if not active.empty else np.nan


def _max_active_streak(df: pd.DataFrame) -> int:
    if df.empty:
        return 0
    best = cur = 0
    prev_bucket: int | None = None
    for row in df.sort_values("bucket").itertuples(index=False):
        bucket = int(row.bucket)
        if _is_active(row.action) and (prev_bucket is None or bucket == prev_bucket + 1):
            cur += 1
        elif _is_active(row.action):
            cur = 1
        else:
            cur = 0
        best = max(best, cur)
        prev_bucket = bucket
    return int(best)


def _count_actions(df: pd.DataFrame, action: str) -> int:
    if df.empty:
        return 0
    return int(df["action"].astype(str).eq(action).sum())


def _load_specs(spec_dir: Path) -> dict[str, pd.DataFrame]:
    specs: dict[str, pd.DataFrame] = {}
    if not spec_dir.exists():
        return specs
    for path in sorted(spec_dir.glob("*.csv")):
        df = pd.read_csv(path, low_memory=False)
        if "case_id" not in df.columns or "bucket" not in df.columns:
            continue
        if "action" not in df.columns:
            continue
        df["case_id"] = df["case_id"].astype(str)
        df["bucket"] = pd.to_numeric(df["bucket"], errors="coerce").fillna(-1).astype(int)
        df["action"] = df["action"].astype(str)
        specs[path.stem] = df
    return specs


def _spec_case(specs: dict[str, pd.DataFrame], label: str, case_id: str) -> pd.DataFrame:
    df = specs.get(label)
    if df is None or df.empty:
        return pd.DataFrame(columns=["case_id", "bucket", "action"])
    return df[df["case_id"].eq(str(case_id))].copy()


def _sequence_features(learned: pd.DataFrame, oracle: pd.DataFrame) -> dict[str, Any]:
    learned_buckets = set(int(x) for x in learned.get("bucket", pd.Series(dtype=int)).tolist())
    oracle_buckets = set(int(x) for x in oracle.get("bucket", pd.Series(dtype=int)).tolist())
    buckets = sorted(learned_buckets | oracle_buckets)
    learned_map = {int(r.bucket): str(r.action) for r in learned.itertuples(index=False)}
    oracle_map = {int(r.bucket): str(r.action) for r in oracle.itertuples(index=False)}
    earlier_active = False
    learned_first = _first_active_bucket(learned)
    oracle_first = _first_active_bucket(oracle)
    if np.isfinite(oracle_first):
        earlier_active = not np.isfinite(learned_first) or oracle_first < learned_first
    medium_upgrade = 0
    active_to_hold = 0
    hold_to_active = 0
    same_action = 0
    action_diff = 0
    for bucket in buckets:
        la = learned_map.get(bucket, "")
        oa = oracle_map.get(bucket, "")
        same_action += int(la == oa)
        action_diff += int(la != oa)
        medium_upgrade += int(oa == "active_medium" and la in {"active_small", "pump_saving", "hold"})
        active_to_hold += int(la == "hold" and _is_active(oa))
        hold_to_active += int(_is_active(la) and oa == "hold")
    learned_active_streak = _max_active_streak(learned)
    oracle_active_streak = _max_active_streak(oracle)
    return {
        "oracle_earlier_active": int(earlier_active),
        "learned_first_active_bucket": learned_first,
        "oracle_first_active_bucket": oracle_first,
        "learned_active_count": int(learned["action"].map(_is_active).sum()) if not learned.empty else 0,
        "oracle_active_count": int(oracle["action"].map(_is_active).sum()) if not oracle.empty else 0,
        "learned_active_medium_count": _count_actions(learned, "active_medium"),
        "oracle_active_medium_count": _count_actions(oracle, "active_medium"),
        "learned_hold_count": _count_actions(learned, "hold"),
        "oracle_hold_count": _count_actions(oracle, "hold"),
        "learned_max_active_streak": learned_active_streak,
        "oracle_max_active_streak": oracle_active_streak,
        "oracle_keeps_more_continuous_active": int(oracle_active_streak > learned_active_streak),
        "oracle_medium_upgrade_count": int(medium_upgrade),
        "oracle_active_vs_learned_hold_count": int(active_to_hold),
        "oracle_hold_vs_learned_active_count": int(hold_to_active),
        "prefix_action_diff_count": int(action_diff),
        "prefix_same_action_count": int(same_action),
    }


def _sample_class(row: pd.Series) -> str:
    label = str(row.get("case_label", ""))
    pump_delta = _safe_float(row.get("pump_delta"))
    fb_delta = _safe_float(row.get("fallback_delta"))
    if label == "prefix-beneficial" and pump_delta <= 0.0 and fb_delta <= 1e-9:
        return "positive"
    if label in {"prefix-pump-tradeoff", "inconclusive"}:
        return "caution"
    if label in {"prefix-worse", "target-lifecycle-dominated"}:
        return "negative"
    return "neutral"


def _decision_text(samples: pd.DataFrame) -> str:
    candidate = samples[
        samples["run_label"].str.contains("learned_forecast_oracle_prefix", na=False)
        & samples["forced_mode"].isin(["final_action", "raw_action"])
    ]
    positive = candidate[candidate["sample_class"].eq("positive")]
    cases = sorted(positive["case_id"].unique().tolist())
    stable_cases = len(cases)
    negative = candidate[candidate["sample_class"].eq("negative")]
    if stable_cases >= 2 and len(positive) >= 8:
        return (
            "Worth a narrow research-only path-level policy distillation pass. "
            "Positive samples exist across multiple cases, but negative/tradeoff "
            "patterns are also present, so this is not a production candidate."
        )
    if len(positive) > 0:
        return (
            "Weak signal only. Positive samples exist, but coverage is too small "
            "or unstable for a distillation run without more evidence."
        )
    return (
        "Freeze h240/path. There are not enough clean positive samples to justify "
        "path-level policy distillation."
    )


def _md_table(df: pd.DataFrame, max_rows: int = 40) -> str:
    if df.empty:
        return "_No rows._"
    show = df.head(max_rows).copy()
    cols = list(show.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in show.iterrows():
        vals: list[str] = []
        for col in cols:
            val = row[col]
            if isinstance(val, float):
                vals.append(f"{val:.3f}" if np.isfinite(val) else "")
            else:
                vals.append(str(val))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def build_dataset(forced_dir: Path, out_dir: Path) -> pd.DataFrame:
    closed = pd.read_csv(forced_dir / "forced_prefix_closed_loop_summary.csv", low_memory=False)
    labels = pd.read_csv(forced_dir / "forced_prefix_case_labels.csv", low_memory=False)
    action = pd.read_csv(forced_dir / "forced_prefix_action_trace.csv", low_memory=False)
    state = pd.read_csv(forced_dir / "forced_prefix_path_state_trace.csv", low_memory=False)
    specs = _load_specs(forced_dir / "forced_specs")

    closed_cols = [
        "run_label",
        "case_id",
        "pump_m3",
        "max_p95_proxy",
        "fallback",
        "time_over_3",
        "time_over_5",
        "key_action_match_count",
        "key_bucket_count",
        "forced_bucket_count",
    ]
    merged = labels.merge(closed[closed_cols], on=["run_label", "case_id"], how="left")
    baseline = closed[closed["run_label"].eq("learned_forecast_learned_prefix_len0_normal")].copy()
    baseline = baseline.rename(
        columns={
            "pump_m3": "baseline_pump_m3",
            "max_p95_proxy": "baseline_max_p95",
            "fallback": "baseline_fallback",
            "time_over_3": "baseline_time_over_3",
            "time_over_5": "baseline_time_over_5",
            "key_action_match_count": "baseline_key_action_match_count",
        }
    )
    merged = merged.merge(
        baseline[
            [
                "case_id",
                "baseline_pump_m3",
                "baseline_max_p95",
                "baseline_fallback",
                "baseline_time_over_3",
                "baseline_time_over_5",
                "baseline_key_action_match_count",
            ]
        ],
        on="case_id",
        how="left",
    )

    rows: list[dict[str, Any]] = []
    for row in merged.itertuples(index=False):
        run_label = str(row.run_label)
        case_id = str(row.case_id)
        meta = _parse_run_label(run_label)
        if meta["prefix_length"] <= 0:
            continue
        length = int(meta["prefix_length"])
        mode = str(meta["forced_mode"])
        learned_label = f"learned_forecast_learned_prefix_len{length}_{mode}"
        oracle_label = f"learned_forecast_oracle_prefix_len{length}_{mode}"
        forced_seq = _spec_case(specs, run_label, case_id)
        learned_seq = _spec_case(specs, learned_label, case_id)
        oracle_seq = _spec_case(specs, oracle_label, case_id)
        features = _sequence_features(learned_seq, oracle_seq)
        state_slice = state[state["run_label"].eq(run_label) & state["case_id"].eq(case_id)]
        action_slice = action[action["run_label"].eq(run_label) & action["case_id"].eq(case_id)]
        pump_delta = _safe_float(row.pump_delta_vs_learned_prefix)
        fallback_delta = _safe_float(row.fallback_delta_vs_learned_prefix)
        max_p95_delta = _safe_float(row.max_p95_proxy) - _safe_float(row.baseline_max_p95)
        time_over_3_delta = _safe_float(row.time_over_3) - _safe_float(row.baseline_time_over_3)
        time_over_5_delta = _safe_float(row.time_over_5) - _safe_float(row.baseline_time_over_5)
        out = {
            "run_label": run_label,
            "case_id": case_id,
            **meta,
            "forced_sequence": _sequence_string(forced_seq),
            "learned_original_sequence": _sequence_string(learned_seq),
            "oracle_prefix_sequence": _sequence_string(oracle_seq),
            "pump_delta": pump_delta,
            "max_p95_delta": max_p95_delta,
            "fallback_delta": fallback_delta,
            "time_over_3_delta": time_over_3_delta,
            "time_over_5_delta": time_over_5_delta,
            "key_action_match_gain": _safe_float(row.key_action_match_gain_vs_learned_prefix),
            "key_action_match_count": int(_safe_float(row.key_action_match_count, 0.0)),
            "key_bucket_count": int(_safe_float(row.key_bucket_count, 0.0)),
            "posture_distance_delta": _safe_float(
                row.mean_key_posture_distance_delta_vs_learned_prefix
            ),
            "case_label": str(row.case_label),
            "forced_bucket_count": int(_safe_float(row.forced_bucket_count, 0.0)),
            "mean_target_delta_to_oracle_kg": float(state_slice["target_mean_abs_delta_to_oracle_kg"].mean()) if not state_slice.empty else np.nan,
            "mean_tank_delta_to_oracle_kg": float(state_slice["tank_mean_abs_delta_to_oracle_kg"].mean()) if not state_slice.empty else np.nan,
            "mean_pump_delta_to_oracle_m3_min": float(state_slice["pump_mean_abs_delta_to_oracle_m3_min"].mean()) if not state_slice.empty else np.nan,
            "action_trace_matches": int(action_slice["matches_oracle_action"].sum()) if not action_slice.empty else 0,
            **features,
        }
        out["reduces_fallback"] = int(fallback_delta < -1e-9)
        out["reduces_pump"] = int(pump_delta < -1e-9)
        out["reduces_time_over_5"] = int(time_over_5_delta < -1e-9)
        out["posture_moves_toward_oracle"] = int(_safe_float(out["posture_distance_delta"]) < -1e-9)
        out["sample_class"] = _sample_class(pd.Series(out))
        rows.append(out)

    samples = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    samples.to_csv(out_dir / "path_policy_distillation_samples.csv", index=False)
    _write_reports(samples, out_dir)
    return samples


def _write_reports(samples: pd.DataFrame, out_dir: Path) -> None:
    positive = samples[samples["sample_class"].eq("positive")].copy()
    caution = samples[samples["sample_class"].eq("caution")].copy()
    negative = samples[samples["sample_class"].eq("negative")].copy()
    neutral = samples[samples["sample_class"].eq("neutral")].copy()

    learned_oracle = samples[
        samples["run_label"].str.contains("learned_forecast_oracle_prefix", na=False)
    ].copy()
    stable = learned_oracle[learned_oracle["sample_class"].eq("positive")]

    feature_cols = [
        "oracle_earlier_active",
        "oracle_keeps_more_continuous_active",
        "oracle_medium_upgrade_count",
        "oracle_active_vs_learned_hold_count",
        "oracle_hold_vs_learned_active_count",
        "reduces_fallback",
        "reduces_pump",
        "reduces_time_over_5",
        "posture_moves_toward_oracle",
    ]
    pattern_rows = []
    for name, df in [
        ("positive", positive),
        ("caution", caution),
        ("negative", negative),
        ("neutral", neutral),
        ("learned_oracle_positive", stable),
    ]:
        if df.empty:
            continue
        row: dict[str, Any] = {"sample_class": name, "n": len(df), "cases": ",".join(sorted(df["case_id"].unique()))}
        for col in feature_cols:
            row[f"{col}_rate"] = float((df[col] > 0).mean())
        row["mean_pump_delta"] = float(df["pump_delta"].mean())
        row["mean_fallback_delta"] = float(df["fallback_delta"].mean())
        row["mean_time_over_5_delta"] = float(df["time_over_5_delta"].mean())
        row["mean_action_gain"] = float(df["key_action_match_gain"].mean())
        pattern_rows.append(row)
    patterns = pd.DataFrame(pattern_rows)

    class_counts = (
        samples.groupby(["sample_class", "case_label"], as_index=False)
        .size()
        .rename(columns={"size": "count"})
    )
    mode_counts = (
        samples.groupby(["sample_class", "forced_mode", "prefix_source"], as_index=False)
        .size()
        .rename(columns={"size": "count"})
    )
    decision = _decision_text(samples)

    lines = [
        "# Path Policy Distillation Dataset v1",
        "",
        "This dataset is built from the existing forced-prefix replay results only. It does not run simulations, train models, or change controller behavior.",
        "",
        "## Sample Counts",
        "",
        _md_table(class_counts, max_rows=80),
        "",
        "## Mode / Source Counts",
        "",
        _md_table(mode_counts, max_rows=80),
        "",
        "## Pattern Summary",
        "",
        _md_table(patterns, max_rows=20),
        "",
        "## Recommended Use",
        "",
        "- `positive`: safe imitation candidates. These are beneficial prefixes with no pump/fallback worsening.",
        "- `caution`: pump-tradeoff or inconclusive prefixes. Do not train as direct positives.",
        "- `negative`: worse or target-lifecycle-dominated prefixes. Use only as avoid/contrast examples.",
        "- `neutral`: no-effect prefixes. Exclude from direct imitation targets.",
        "",
        "## Decision",
        "",
        decision,
    ]
    (out_dir / "path_policy_distillation_dataset_summary.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )

    ben_lines = [
        "# Beneficial Prefix Patterns",
        "",
        "Positive samples are restricted to prefixes labelled beneficial with no pump or fallback worsening.",
        "",
        "## Positive Samples",
        "",
        _md_table(
            positive[
                [
                    "run_label",
                    "case_id",
                    "forced_mode",
                    "prefix_length",
                    "forced_sequence",
                    "learned_original_sequence",
                    "oracle_prefix_sequence",
                    "pump_delta",
                    "fallback_delta",
                    "time_over_5_delta",
                    "key_action_match_gain",
                    "oracle_earlier_active",
                    "oracle_keeps_more_continuous_active",
                    "oracle_medium_upgrade_count",
                ]
            ].sort_values(["case_id", "run_label"]),
            max_rows=80,
        ),
        "",
        "## Common Pattern Rates",
        "",
        _md_table(patterns[patterns["sample_class"].isin(["positive", "learned_oracle_positive"])], max_rows=10),
    ]
    (out_dir / "beneficial_prefix_patterns.md").write_text(
        "\n".join(ben_lines) + "\n", encoding="utf-8"
    )

    neg_lines = [
        "# Negative / Caution Prefix Patterns",
        "",
        "Negative samples include worse and target-lifecycle-dominated prefixes. Caution samples include pump-tradeoff and inconclusive prefixes.",
        "",
        "## Negative Samples",
        "",
        _md_table(
            negative[
                [
                    "run_label",
                    "case_id",
                    "forced_mode",
                    "prefix_length",
                    "forced_sequence",
                    "pump_delta",
                    "max_p95_delta",
                    "fallback_delta",
                    "time_over_5_delta",
                    "key_action_match_gain",
                    "case_label",
                    "oracle_medium_upgrade_count",
                    "oracle_active_vs_learned_hold_count",
                ]
            ].sort_values(["case_id", "run_label"]),
            max_rows=80,
        ),
        "",
        "## Caution Samples",
        "",
        _md_table(
            caution[
                [
                    "run_label",
                    "case_id",
                    "forced_mode",
                    "prefix_length",
                    "forced_sequence",
                    "pump_delta",
                    "fallback_delta",
                    "time_over_5_delta",
                    "key_action_match_gain",
                    "case_label",
                ]
            ].sort_values(["case_id", "run_label"]),
            max_rows=80,
        ),
        "",
        "## Common Pattern Rates",
        "",
        _md_table(patterns[patterns["sample_class"].isin(["negative", "caution"])], max_rows=10),
    ]
    (out_dir / "negative_prefix_patterns.md").write_text(
        "\n".join(neg_lines) + "\n", encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--forced-dir", default=str(DEFAULT_FORCED_DIR))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    forced_dir = Path(args.forced_dir)
    if not forced_dir.is_absolute():
        forced_dir = REPO_ROOT / forced_dir
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = REPO_ROOT / out_dir
    samples = build_dataset(forced_dir, out_dir)
    print(f"wrote {out_dir / 'path_policy_distillation_dataset_summary.md'}")
    print(f"samples={len(samples)} positive={(samples['sample_class'] == 'positive').sum()} negative={(samples['sample_class'] == 'negative').sum()}")


if __name__ == "__main__":
    main()
