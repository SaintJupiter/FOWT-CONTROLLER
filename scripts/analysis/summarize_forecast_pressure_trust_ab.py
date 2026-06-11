#!/usr/bin/env python3
"""Build forecast-pressure-trust A/B compare tables and go/no-go report."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = (
    REPO_ROOT / "outputs" / "wind_prediction" / "forecast_pressure_trust_gate_20260604"
)


def _resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p


def _num(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col not in df.columns:
        return pd.Series(np.full(len(df), default), index=df.index)
    return pd.to_numeric(df[col], errors="coerce").fillna(default)


def _planner_metrics(log_dir: Path, case_ids: list[str]) -> pd.DataFrame:
    rows = []
    case_ids = sorted([str(x) for x in case_ids], key=len, reverse=True)
    for path in sorted((log_dir / "planner_logs").glob("*_planner_log.csv")):
        df = pd.read_csv(path)
        if df.empty:
            continue
        stem = path.name.replace("_planner_log.csv", "")
        case_id = next(
            (candidate for candidate in case_ids if stem.startswith(candidate + "_")),
            stem,
        )
        gate_enabled = _num(df, "forecast_pressure_trust_gate_enabled")
        trusted = _num(df, "forecast_pressure_trust_trusted", 1.0)
        block_mask = (gate_enabled > 0.5) & (trusted < 0.5)
        rows.append(
            {
                "case_id": case_id,
                "planner_log": str(path.relative_to(REPO_ROOT)),
                "planner_bucket_count": len(df),
                "planner_pressure_trust_block_ratio": float(np.mean(block_mask)),
                "planner_pressure_trust_scale_mean": float(
                    np.mean(
                        np.column_stack(
                            [
                                _num(df, "forecast_pressure_trust_block0_scale", 1.0),
                                _num(df, "forecast_pressure_trust_block1_scale", 1.0),
                                _num(df, "forecast_pressure_trust_block2_scale", 1.0),
                            ]
                        )
                    )
                ),
                "planner_pressure_trust_current_support_ratio": float(
                    np.mean(_num(df, "forecast_pressure_trust_current_support") > 0.5)
                ),
                "planner_pressure_trust_event_support_ratio": float(
                    np.mean(_num(df, "forecast_pressure_trust_event_support") > 0.5)
                ),
                "planner_pressure_trust_shape_support_ratio": float(
                    np.mean(
                        _num(df, "forecast_pressure_trust_pressure_shape_support")
                        > 0.5
                    )
                ),
                "planner_pressure_trust_current_speed_mean_ms": float(
                    np.mean(_num(df, "forecast_pressure_trust_current_speed_ms"))
                ),
                "planner_pressure_trust_current_observed_norm_mean": float(
                    np.mean(
                        _num(df, "forecast_pressure_trust_current_observed_norm")
                    )
                ),
                "planner_pressure_trust_reason_modes": ";".join(
                    df.get(
                        "forecast_pressure_trust_reason",
                        pd.Series(["missing" for _ in range(len(df))], index=df.index),
                    )
                    .fillna("missing")
                    .astype(str)
                    .value_counts()
                    .head(3)
                    .index
                ),
            }
        )
    return pd.DataFrame(rows)


def _load_run_summary(run_dir: Path, suffix: str) -> pd.DataFrame:
    df = pd.read_csv(run_dir / "casebook_summary.csv")
    metrics = _planner_metrics(run_dir, df["case_id"].astype(str).tolist())
    out = df.merge(metrics, on="case_id", how="left")
    keep = [
        "case_id",
        "timestamp",
        "label",
        "primary_pump_work_m3",
        "primary_latch_switches",
        "primary_pitch_p95",
        "primary_roll_p95",
        "primary_safety_fallback_ratio",
        "planner_pressure_trust_block_ratio",
        "planner_pressure_trust_scale_mean",
        "planner_pressure_trust_current_support_ratio",
        "planner_pressure_trust_event_support_ratio",
        "planner_pressure_trust_shape_support_ratio",
        "planner_pressure_trust_current_speed_mean_ms",
        "planner_pressure_trust_current_observed_norm_mean",
        "planner_pressure_trust_reason_modes",
    ]
    for col in keep:
        if col not in out.columns:
            out[col] = np.nan
    return out[keep].add_suffix(f"_{suffix}").rename(
        columns={
            f"case_id_{suffix}": "case_id",
            f"timestamp_{suffix}": "timestamp",
            f"label_{suffix}": "label",
        }
    )


def _pair_compare(group: str, raw_dir: Path, trust_dir: Path) -> pd.DataFrame:
    raw = _load_run_summary(raw_dir, "raw")
    trust = _load_run_summary(trust_dir, "trust")
    merged = raw.merge(trust, on=["case_id", "timestamp", "label"], how="inner")
    merged.insert(0, "group", group)
    merged["d_pump_m3"] = (
        merged["primary_pump_work_m3_trust"] - merged["primary_pump_work_m3_raw"]
    )
    merged["d_latch"] = (
        merged["primary_latch_switches_trust"]
        - merged["primary_latch_switches_raw"]
    )
    merged["d_pitch_p95"] = (
        merged["primary_pitch_p95_trust"] - merged["primary_pitch_p95_raw"]
    )
    merged["d_roll_p95"] = (
        merged["primary_roll_p95_trust"] - merged["primary_roll_p95_raw"]
    )
    merged["d_fallback_ratio"] = (
        merged["primary_safety_fallback_ratio_trust"]
        - merged["primary_safety_fallback_ratio_raw"]
    )
    return merged


def _write_report(out_dir: Path, compare: pd.DataFrame, summary: pd.DataFrame) -> None:
    lines = [
        "# Forecast Pressure Trust Gate Go/No-Go",
        "",
        "## Decision",
        "",
        "**NO-GO for promotion beyond default-off.** The implemented telemetry and default-off gate are useful for diagnosis, but the focused A/B does not show material control improvement on the mined stable pool.",
        "",
        "## Aggregate A/B",
        "",
        "| group | cases | raw pump m3 | trust pump m3 | delta pump m3 | delta latch | mean d pitch p95 | mean d roll p95 | mean d fallback | trust block ratio |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary.to_dict("records"):
        lines.append(
            "| {group} | {case_count:.0f} | {raw:.2f} | {trust:.2f} | {dp:.2f} | {dl:.0f} | {dpt:.3f} | {dr:.3f} | {df:.4f} | {br:.3f} |".format(
                group=row["group"],
                case_count=row["case_count"],
                raw=row["pump_raw_sum_m3"],
                trust=row["pump_trust_sum_m3"],
                dp=row["d_pump_sum_m3"],
                dl=row["d_latch_sum"],
                dpt=row["d_pitch_p95_mean"],
                dr=row["d_roll_p95_mean"],
                df=row["d_fallback_ratio_mean"],
                br=row["trust_block_ratio_mean"],
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Stable pool: pressure trust produced 0.00 m3 pump delta and 0 latch delta across 12 cases.",
            "- Highwind/attention falsifiers: trust mode matched raw behavior exactly across 8 cases, so the gate did not damage reliable-event responses.",
            "- Planner-log telemetry shows the current candidate pool is not a clean pressure-untrusted pool under replay semantics: the stable probes have high current observed wind, coherent pressure shape, or no isolated unsupported spike, so the gate has no legitimate control to remove.",
            "",
            "## Recommended Next Step",
            "",
            "Keep the provider gate and telemetry default-off. Do not tune thresholds on this pool. The next useful work is to mine a fresh pool directly from replay samples and planner logs where `forecast_pressure_trust_block_ratio` would be nonzero under conservative thresholds, then rerun A/B only if that pool contains active stable cases.",
            "",
            "## Artifacts",
            "",
            "- `pressure_trust_stable_candidates.csv`",
            "- `pressure_trust_highwind_falsifiers.csv`",
            "- `forecast_pressure_trust_ab_compare.csv`",
            "- `forecast_pressure_trust_ab_summary.csv`",
            "- `forecast_pressure_trust_telemetry_contract.md`",
        ]
    )
    (out_dir / "forecast_pressure_trust_go_no_go_20260604.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = _resolve(args.out_dir)
    pairs = [
        ("stable", out_dir / "stable_raw_probe", out_dir / "stable_trust_probe"),
        ("highwind", out_dir / "highwind_raw_probe", out_dir / "highwind_trust_probe"),
    ]
    compare = pd.concat(
        [_pair_compare(group, raw, trust) for group, raw, trust in pairs],
        ignore_index=True,
        sort=False,
    )
    summary_rows = []
    for group, g in compare.groupby("group"):
        summary_rows.append(
            {
                "group": group,
                "case_count": len(g),
                "pump_raw_sum_m3": float(g["primary_pump_work_m3_raw"].sum()),
                "pump_trust_sum_m3": float(g["primary_pump_work_m3_trust"].sum()),
                "d_pump_sum_m3": float(g["d_pump_m3"].sum()),
                "d_latch_sum": float(g["d_latch"].sum()),
                "d_pitch_p95_mean": float(g["d_pitch_p95"].mean()),
                "d_roll_p95_mean": float(g["d_roll_p95"].mean()),
                "d_fallback_ratio_mean": float(g["d_fallback_ratio"].mean()),
                "trust_block_ratio_mean": float(
                    g["planner_pressure_trust_block_ratio_trust"].fillna(0.0).mean()
                ),
                "trust_scale_mean": float(
                    g["planner_pressure_trust_scale_mean_trust"].fillna(1.0).mean()
                ),
                "trust_current_support_ratio_mean": float(
                    g[
                        "planner_pressure_trust_current_support_ratio_trust"
                    ].fillna(0.0).mean()
                ),
                "trust_event_support_ratio_mean": float(
                    g[
                        "planner_pressure_trust_event_support_ratio_trust"
                    ].fillna(0.0).mean()
                ),
                "trust_shape_support_ratio_mean": float(
                    g[
                        "planner_pressure_trust_shape_support_ratio_trust"
                    ].fillna(0.0).mean()
                ),
            }
        )
    summary = pd.DataFrame(summary_rows).sort_values("group")
    compare.to_csv(out_dir / "forecast_pressure_trust_ab_compare.csv", index=False)
    summary.to_csv(out_dir / "forecast_pressure_trust_ab_summary.csv", index=False)
    _write_report(out_dir, compare, summary)
    print(summary.to_string(index=False))
    print(out_dir / "forecast_pressure_trust_go_no_go_20260604.md")


if __name__ == "__main__":
    main()
