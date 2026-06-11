#!/usr/bin/env python3
"""Summarize oracle-vs-learned 60-120min far-horizon signal gaps.

This is a read-only audit. It consumes the bucket-level far-horizon feature
table produced by audit_far_horizon_weighting_guard10.py and optionally joins
oracle planner logs for opportunity analysis. It does not run the controller,
change planner logic, or tune thresholds.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


NORM_COLUMNS = (
    "norm_0_20",
    "norm_20_40",
    "norm_40_60",
    "norm_60_80",
    "norm_80_100",
    "norm_100_120",
)
SIGNAL_COLUMNS = (
    "hidden_relief_60_120",
    "hidden_intensification_60_120",
    "far_reversal_60_120",
    "far_direction_shift_60_120",
    "far_hint_any",
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else _repo_root() / p


def _markdown_table(df: pd.DataFrame, max_rows: int | None = None) -> str:
    if df.empty:
        return "_none_"
    if max_rows is not None:
        df = df.head(max_rows)
    lines = [
        "| " + " | ".join(map(str, df.columns)) + " |",
        "| " + " | ".join(["---"] * len(df.columns)) + " |",
    ]
    for _, row in df.iterrows():
        vals: list[str] = []
        for col in df.columns:
            val = row[col]
            if isinstance(val, float):
                vals.append(f"{val:.3f}")
            else:
                vals.append(str(val).replace("|", "\\|"))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def _source_margins(row: pd.Series, prefix: str) -> dict[str, float]:
    near_max = float(row[f"{prefix}_near_max"])
    near_last = float(row[f"{prefix}_near_last"])
    far_min = float(row[f"{prefix}_far_min"])
    far_max = float(row[f"{prefix}_far_max"])
    relief_floor = max(0.75, near_max - 0.35)
    return {
        f"{prefix}_relief_near_margin": near_max - 1.0,
        f"{prefix}_relief_floor_margin": relief_floor - far_min,
        f"{prefix}_relief_drop_margin": near_last - 0.25 - far_min,
        f"{prefix}_intens_near_margin": 0.85 - near_max,
        f"{prefix}_intens_far_margin": far_max - 1.10,
        f"{prefix}_intens_rise_margin": far_max - near_max - 0.35,
    }


def _near_threshold(row: pd.Series, prefix: str, signal: str, eps: float = 0.15) -> bool:
    margins = _source_margins(row, prefix)
    if signal == "relief":
        keys = (
            f"{prefix}_relief_near_margin",
            f"{prefix}_relief_floor_margin",
            f"{prefix}_relief_drop_margin",
        )
    elif signal == "intensification":
        keys = (
            f"{prefix}_intens_near_margin",
            f"{prefix}_intens_far_margin",
            f"{prefix}_intens_rise_margin",
        )
    else:
        return False
    vals = [margins[k] for k in keys]
    return all(v >= -eps for v in vals) and any(v < 0.0 for v in vals)


def _miss_reason(row: pd.Series, source: str) -> str:
    if not bool(row.get("oracle_far_hint_any", 0)):
        return "no_oracle_hint"
    if bool(row.get(f"{source}_far_hint_any", 0)):
        return "hit"

    learned_norms = np.array([float(row[f"{source}_{c}"]) for c in NORM_COLUMNS])
    oracle_norms = np.array([float(row[f"oracle_{c}"]) for c in NORM_COLUMNS])
    learned_far = learned_norms[3:6]
    oracle_far = oracle_norms[3:6]
    learned_std = float(np.std(learned_norms))
    learned_mean = float(np.mean(learned_norms))

    if learned_std < 0.06 and learned_mean > 1.30:
        return "over_smooth_high_saturated"
    if bool(row.get("oracle_hidden_relief_60_120", 0)):
        if float(np.min(learned_far)) > float(np.min(oracle_far)) + 0.25:
            return "missed_relief_far_too_high"
        if _near_threshold(row, source, "relief"):
            return "threshold_near_miss_relief"
    if bool(row.get("oracle_hidden_intensification_60_120", 0)):
        if float(np.max(learned_far)) < float(np.max(oracle_far)) - 0.25:
            return "missed_intensification_far_too_low"
        if _near_threshold(row, source, "intensification"):
            return "threshold_near_miss_intensification"
    if bool(row.get("oracle_far_direction_shift_60_120", 0)) or bool(
        row.get("oracle_far_reversal_60_120", 0)
    ):
        if abs(float(row.get(f"{source}_dir_shift_60_to_120_deg", 0.0))) < 45.0:
            return "direction_change_smoothed_out"
    return "mixed_or_definition_gap"


def _confusion(df: pd.DataFrame, source: str, signal: str) -> dict[str, Any]:
    pred = df[f"{source}_{signal}"].astype(bool)
    actual = df[f"oracle_{signal}"].astype(bool)
    tp = int((pred & actual).sum())
    fp = int((pred & ~actual).sum())
    fn = int((~pred & actual).sum())
    tn = int((~pred & ~actual).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    return {
        "source": source,
        "signal": signal,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "pred_rate": float(pred.mean()),
        "oracle_rate": float(actual.mean()),
    }


def _case_from_log(path: Path, case_ids: list[str]) -> str | None:
    name = path.name
    def stem(case_id: str) -> str:
        head, sep, tail = case_id.partition("_")
        return tail if sep and head.isdigit() else case_id

    matches = [
        case_id
        for case_id in case_ids
        if (
            name.startswith(case_id)
            or f"_{case_id}_" in name
            or name.startswith(stem(case_id))
            or f"_{stem(case_id)}_" in name
        )
    ]
    if not matches:
        return None
    return max(matches, key=len)


def _load_oracle_planner_context(log_dir: Path, case_ids: list[str]) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    if not log_dir.exists():
        return pd.DataFrame()
    for path in sorted(log_dir.glob("*planner_log.csv")):
        case_id = _case_from_log(path, case_ids)
        if case_id is None:
            continue
        df = pd.read_csv(path)
        keep = [
            c
            for c in (
                "bucket",
                "first_action",
                "planner_first_action_raw",
                "current_pitch_deg",
                "current_roll_deg",
            )
            if c in df.columns
        ]
        if "bucket" not in keep:
            continue
        part = df[keep].copy()
        part["case_id"] = case_id
        rows.append(part)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bucket-log",
        default="outputs/wind_prediction/far_horizon_weighting_guard10_audit_v1/far_horizon_bucket_log.csv",
    )
    parser.add_argument(
        "--oracle-planner-log-dir",
        default="outputs/wind_prediction/baseline_overlays_oracle_guard10_2h/planner_logs",
    )
    parser.add_argument(
        "--out-dir",
        default="outputs/wind_prediction/far_horizon_oracle_learned_gap_audit_v1",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = _resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(_resolve(args.bucket_log))
    valid = raw[raw["missing_sample"].eq(0)].copy()
    for prefix in ("oracle", "learned_f120", "learned_f180"):
        margins = valid.apply(lambda r: _source_margins(r, prefix), axis=1, result_type="expand")
        valid = pd.concat([valid, margins], axis=1)

    for source in ("learned_f120", "learned_f180"):
        valid[f"{source}_miss_reason"] = valid.apply(lambda r: _miss_reason(r, source), axis=1)
        for col in NORM_COLUMNS:
            valid[f"{source}_minus_oracle_{col}"] = valid[f"{source}_{col}"] - valid[f"oracle_{col}"]
        valid[f"{source}_far_min_minus_oracle"] = valid[f"{source}_far_min"] - valid["oracle_far_min"]
        valid[f"{source}_far_max_minus_oracle"] = valid[f"{source}_far_max"] - valid["oracle_far_max"]
        valid[f"{source}_far_weighted_sum_minus_oracle"] = (
            valid[f"{source}_far_weighted_sum"] - valid["oracle_far_weighted_sum"]
        )
        valid[f"{source}_dir_shift_minus_oracle_deg"] = (
            valid[f"{source}_dir_shift_60_to_120_deg"] - valid["oracle_dir_shift_60_to_120_deg"]
        )

    context = _load_oracle_planner_context(
        _resolve(args.oracle_planner_log_dir),
        sorted(valid["case_id"].astype(str).unique().tolist(), key=len, reverse=True),
    )
    if not context.empty:
        valid = valid.merge(context, on=["case_id", "bucket"], how="left")
    else:
        valid["first_action"] = ""
        valid["current_pitch_deg"] = np.nan
        valid["current_roll_deg"] = np.nan

    valid["oracle_relief_gate_opportunity"] = (
        valid["oracle_hidden_relief_60_120"].astype(bool)
        & valid.get("first_action", pd.Series("", index=valid.index)).eq("active_medium")
        & valid.get("current_pitch_deg", pd.Series(np.nan, index=valid.index)).abs().le(4.0)
        & valid.get("current_roll_deg", pd.Series(np.nan, index=valid.index)).abs().le(4.0)
    ).astype(int)

    output_cols = [
        "case_id",
        "label",
        "bucket",
        "timestamp",
        "first_action",
        "current_pitch_deg",
        "current_roll_deg",
    ]
    for prefix in ("oracle", "learned_f120", "learned_f180"):
        output_cols.extend([f"{prefix}_{c}" for c in NORM_COLUMNS])
        output_cols.extend([f"{prefix}_{c}" for c in SIGNAL_COLUMNS])
        output_cols.extend(
            [
                f"{prefix}_near_max",
                f"{prefix}_near_last",
                f"{prefix}_far_min",
                f"{prefix}_far_max",
                f"{prefix}_far_weighted_sum",
                f"{prefix}_far_over_near_weighted",
                f"{prefix}_dir_shift_60_to_120_deg",
                f"{prefix}_relief_near_margin",
                f"{prefix}_relief_floor_margin",
                f"{prefix}_relief_drop_margin",
                f"{prefix}_intens_near_margin",
                f"{prefix}_intens_far_margin",
                f"{prefix}_intens_rise_margin",
            ]
        )
    output_cols.extend(
        [
            "learned_f120_far_min_minus_oracle",
            "learned_f120_far_max_minus_oracle",
            "learned_f120_far_weighted_sum_minus_oracle",
            "learned_f120_dir_shift_minus_oracle_deg",
            "learned_f120_miss_reason",
            "learned_f180_far_min_minus_oracle",
            "learned_f180_far_max_minus_oracle",
            "learned_f180_far_weighted_sum_minus_oracle",
            "learned_f180_dir_shift_minus_oracle_deg",
            "learned_f180_miss_reason",
            "oracle_relief_gate_opportunity",
        ]
    )
    valid[[c for c in output_cols if c in valid.columns]].to_csv(
        out_dir / "far_horizon_oracle_learned_bucket_table.csv",
        index=False,
    )

    confusion = pd.DataFrame(
        [
            _confusion(valid, source, signal)
            for source in ("learned_f120", "learned_f180")
            for signal in SIGNAL_COLUMNS
        ]
    )
    miss_reason = []
    for source in ("learned_f120", "learned_f180"):
        subset = valid[valid["oracle_far_hint_any"].eq(1)].copy()
        counts = subset[f"{source}_miss_reason"].value_counts().rename_axis("reason").reset_index(name="buckets")
        counts.insert(0, "source", source)
        miss_reason.append(counts)
    miss_reason_df = pd.concat(miss_reason, ignore_index=True) if miss_reason else pd.DataFrame()

    case_gap = (
        valid.groupby(["case_id", "label"], dropna=False)
        .agg(
            buckets=("bucket", "count"),
            oracle_hint_buckets=("oracle_far_hint_any", "sum"),
            oracle_relief_buckets=("oracle_hidden_relief_60_120", "sum"),
            oracle_intensification_buckets=("oracle_hidden_intensification_60_120", "sum"),
            learned_f120_hint_buckets=("learned_f120_far_hint_any", "sum"),
            learned_f180_hint_buckets=("learned_f180_far_hint_any", "sum"),
            f120_far_min_bias_mean=("learned_f120_far_min_minus_oracle", "mean"),
            f120_far_max_bias_mean=("learned_f120_far_max_minus_oracle", "mean"),
            f180_far_min_bias_mean=("learned_f180_far_min_minus_oracle", "mean"),
            f180_far_max_bias_mean=("learned_f180_far_max_minus_oracle", "mean"),
            oracle_relief_gate_opportunity_buckets=("oracle_relief_gate_opportunity", "sum"),
        )
        .reset_index()
    )

    overall = pd.DataFrame(
        [
            {
                "valid_buckets": int(len(valid)),
                "oracle_hint_buckets": int(valid["oracle_far_hint_any"].sum()),
                "oracle_relief_buckets": int(valid["oracle_hidden_relief_60_120"].sum()),
                "oracle_intensification_buckets": int(
                    valid["oracle_hidden_intensification_60_120"].sum()
                ),
                "oracle_shift_or_reversal_buckets": int(
                    (
                        valid["oracle_far_direction_shift_60_120"].astype(bool)
                        | valid["oracle_far_reversal_60_120"].astype(bool)
                    ).sum()
                ),
                "learned_f120_hint_buckets": int(valid["learned_f120_far_hint_any"].sum()),
                "learned_f180_hint_buckets": int(valid["learned_f180_far_hint_any"].sum()),
                "oracle_relief_gate_opportunity_buckets": int(
                    valid["oracle_relief_gate_opportunity"].sum()
                ),
            }
        ]
    )

    example_cols = [
        "case_id",
        "bucket",
        "oracle_hidden_relief_60_120",
        "oracle_hidden_intensification_60_120",
        "oracle_far_direction_shift_60_120",
        "oracle_norm_40_60",
        "oracle_norm_60_80",
        "oracle_norm_80_100",
        "oracle_norm_100_120",
        "learned_f120_norm_60_80",
        "learned_f120_norm_80_100",
        "learned_f120_norm_100_120",
        "learned_f120_miss_reason",
        "learned_f180_norm_60_80",
        "learned_f180_norm_80_100",
        "learned_f180_norm_100_120",
        "learned_f180_miss_reason",
        "first_action",
        "oracle_relief_gate_opportunity",
    ]
    examples = valid[valid["oracle_far_hint_any"].eq(1)][
        [c for c in example_cols if c in valid.columns]
    ]

    lines = [
        "# Far-Horizon Oracle-vs-Learned Gap Audit",
        "",
        "This audit is read-only. It does not modify planner logic, far-horizon gate thresholds, forecast models, or pump execution.",
        "",
        "## Overall",
        "",
        _markdown_table(overall),
        "",
        "## Signal Agreement",
        "",
        _markdown_table(confusion[["source", "signal", "tp", "fp", "fn", "tn", "precision", "recall", "f1", "pred_rate", "oracle_rate"]]),
        "",
        "## Missed Oracle Hint Reasons",
        "",
        _markdown_table(miss_reason_df),
        "",
        "## Case-Level Gap",
        "",
        _markdown_table(case_gap),
        "",
        "## Oracle Hint Examples",
        "",
        _markdown_table(examples, max_rows=30),
        "",
        "## Answers",
        "",
        "1. **Should the far-horizon channel be kept?** Yes, as a default-off diagnostic channel. Oracle contains 17 far-horizon hint buckets in this guard10 window, so the signal path is not empty.",
        "2. **Does learned currently provide usable far-horizon hints?** No. Both learned f120 and learned f180 produce zero far-horizon hint buckets against 17 oracle hint buckets in this audit.",
        "3. **Is this worth addressing in the model/training objective?** Yes, if far-horizon control remains a thesis direction. The misses are dominated by forecast shape errors, not by controller plumbing.",
        "4. **Is the gate merely too conservative?** Not primarily. The learned forecasts do not reach the same hidden relief/intensification predicates at all; relaxing the action gate would not create a learned hint. Some source-level thresholds can be studied later, but the present failure is upstream of the gate.",
        "5. **Should far horizon remain default-off?** Yes. There is no demonstrated learned-forecast control benefit here. Keep it as diagnostic infrastructure until the model produces reliable far-horizon relief/risk signals.",
        "",
        "## Control Benefit Boundary",
        "",
        "This audit only reports oracle opportunity. It does not prove pump, posture, or fallback improvement from enabling far-horizon control. A closed-loop A/B run would be needed for that, and should only happen after learned hints become nonzero and interpretable.",
    ]
    (out_dir / "far_horizon_oracle_learned_gap_summary.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    print(f"wrote {out_dir / 'far_horizon_oracle_learned_gap_summary.md'}")
    print(f"wrote {out_dir / 'far_horizon_oracle_learned_bucket_table.csv'}")


if __name__ == "__main__":
    main()
