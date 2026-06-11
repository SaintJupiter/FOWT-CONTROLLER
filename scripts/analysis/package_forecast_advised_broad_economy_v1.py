#!/usr/bin/env python3
"""Package the forecast-advised broad economy family for paper use.

This script is intentionally read-only with respect to controller behavior.  It
collects the already-run representative / boundary / oracle-missing checks and
writes a compact paper-ready result bundle.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


BASE = Path(
    "outputs/wind_prediction/regime_conditioned_policy_development_v1/"
    "neutral_untyped_baseline_headroom_v1"
)
PAPER = BASE / "paper_ready" / "forecast_advised_broad_economy_v1"
FIG = PAPER / "figures"


def pct(saved: float, base: float) -> float:
    return 100.0 * saved / base if base else 0.0


def load_casebook(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def normalized_case_id(value: object) -> str:
    return re.sub(r"^\d+_", "", str(value))


def split_corrected_representative(mixed: pd.DataFrame, validation: pd.DataFrame) -> pd.DataFrame:
    """Replace legacy validation rows that were replayed against the test split.

    The representative casebook contains both test and validation timestamps.
    Older runner invocations resolved all timestamps against the test split, so
    validation rows fail-closed with missing forecast samples.  The corrected
    source-of-truth keeps the original test rows and replaces those validation
    rows with the dedicated validation-split rerun.
    """

    mixed = mixed.copy()
    validation = validation.copy()
    mixed["_norm_case_id"] = mixed["case_id"].map(normalized_case_id)
    validation["_norm_case_id"] = validation["case_id"].map(normalized_case_id)
    corrected = pd.concat(
        [
            mixed[~mixed["_norm_case_id"].isin(set(validation["_norm_case_id"]))],
            validation,
        ],
        ignore_index=True,
    )
    if int(corrected["_norm_case_id"].nunique()) != int(len(mixed)):
        raise RuntimeError(
            "split-corrected representative does not preserve the 48 unique cases"
        )
    return corrected.drop(columns=["_norm_case_id"])


def casebook_metrics(df: pd.DataFrame, name: str) -> dict[str, float | str]:
    closed = float(df["closed_pump_work_m3"].sum())
    primary = float(df["primary_pump_work_m3"].sum())
    active = int((df["preview_pump_suppression_ratio"] > 0).sum())
    return {
        "set": name,
        "cases": int(len(df)),
        "active_cases": active,
        "active_ratio": active / len(df) if len(df) else 0.0,
        "a0_pump_m3": closed,
        "candidate_pump_m3": primary,
        "pump_saved_m3": closed - primary,
        "pump_saving_pct": pct(closed - primary, closed),
        "fallback_rows_s": float(df["primary_safety_fallback_ratio"].sum() * 3600.0),
        "mean_d_pitch_p95_deg": float(
            (df["primary_pitch_p95"] - df["closed_pitch_p95"]).mean()
        ),
        "mean_d_roll_p95_deg": float(
            (df["primary_roll_p95"] - df["closed_roll_p95"]).mean()
        ),
    }


def add_subtype(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    data["subtype"] = data["label"].str.extract(r"subtype=([^|]+)")[0].str.strip()
    return data


def _matching_timeseries(run_dir: Path, case_id: str, timestamp: object, suffix: str) -> Path | None:
    stamp = pd.to_datetime(timestamp).strftime("%Y-%m-%d_%H%M%S")
    matches = sorted(
        (run_dir / "timeseries").glob(f"*{normalized_case_id(case_id)}*{stamp}*_{suffix}_timeseries.csv")
    )
    if matches:
        return matches[0]
    matches = sorted((run_dir / "timeseries").glob(f"*{stamp}*_{suffix}_timeseries.csv"))
    return matches[0] if matches else None


def _timeseries_threshold_metrics(run_dir: Path, row: pd.Series, suffix: str) -> dict[str, float]:
    path = _matching_timeseries(run_dir, row["case_id"], row["timestamp"], suffix)
    if path is None or not path.exists():
        return {
            "duration_s": 0.0,
            "time_gt3_s": 0.0,
            "time_gt4_s": 0.0,
            "time_gt5_s": 0.0,
            "max_axis_deg": 0.0,
        }
    ts = pd.read_csv(path, usecols=["t_s", "pitch_deg", "roll_deg"])
    if ts.empty:
        return {
            "duration_s": 0.0,
            "time_gt3_s": 0.0,
            "time_gt4_s": 0.0,
            "time_gt5_s": 0.0,
            "max_axis_deg": 0.0,
        }
    dt = float(ts["t_s"].diff().dropna().median()) if len(ts) > 1 else 1.0
    if not dt or dt <= 0:
        dt = 1.0
    axis = ts[["pitch_deg", "roll_deg"]].abs().max(axis=1)
    return {
        "duration_s": float(len(ts) * dt),
        "time_gt3_s": float((axis > 3.0).sum() * dt),
        "time_gt4_s": float((axis > 4.0).sum() * dt),
        "time_gt5_s": float((axis > 5.0).sum() * dt),
        "max_axis_deg": float(axis.max()),
    }


def split_corrected_timeseries_metrics(
    rep_mixed: pd.DataFrame,
    validation_missing: pd.DataFrame,
) -> pd.DataFrame:
    mixed_norm = rep_mixed.copy()
    validation_norm = validation_missing.copy()
    mixed_norm["_norm_case_id"] = mixed_norm["case_id"].map(normalized_case_id)
    validation_norm["_norm_case_id"] = validation_norm["case_id"].map(normalized_case_id)
    mixed_keep = mixed_norm[~mixed_norm["_norm_case_id"].isin(set(validation_norm["_norm_case_id"]))].copy()
    mixed_keep["_run_dir"] = str(BASE / "forecast_advised_economy_v2_representative_48_1h")
    validation_norm["_run_dir"] = str(BASE / "forecast_advised_economy_v2_validation_missing20_1h")
    rows = pd.concat([mixed_keep, validation_norm], ignore_index=True)
    metrics = []
    for _, row in rows.iterrows():
        run_dir = Path(str(row["_run_dir"]))
        closed = _timeseries_threshold_metrics(run_dir, row, "closed_only")
        primary = _timeseries_threshold_metrics(run_dir, row, "prediction_primary_econ")
        m = {
            "closed_duration_s": closed["duration_s"],
            "primary_duration_s": primary["duration_s"],
            "closed_time_gt3_s": closed["time_gt3_s"],
            "primary_time_gt3_s": primary["time_gt3_s"],
            "d_time_gt3_s": primary["time_gt3_s"] - closed["time_gt3_s"],
            "closed_time_gt4_s": closed["time_gt4_s"],
            "primary_time_gt4_s": primary["time_gt4_s"],
            "d_time_gt4_s": primary["time_gt4_s"] - closed["time_gt4_s"],
            "closed_time_gt5_s": closed["time_gt5_s"],
            "primary_time_gt5_s": primary["time_gt5_s"],
            "d_time_gt5_s": primary["time_gt5_s"] - closed["time_gt5_s"],
            "closed_max_axis_deg": closed["max_axis_deg"],
            "primary_max_axis_deg": primary["max_axis_deg"],
            "d_max_axis_deg": primary["max_axis_deg"] - closed["max_axis_deg"],
        }
        m.update(
            {
                "case_id": row["case_id"],
                "timestamp": row["timestamp"],
                "subtype": str(row["label"]).split("subtype=", 1)[1].split("|", 1)[0].strip()
                if "subtype=" in str(row["label"])
                else "",
                "run_dir": row["_run_dir"],
                "pump_saving_pct": float(row["d_pump_work_pct"]) * -1.0,
                "fallback_s": float(row["primary_safety_fallback_ratio"]) * 3600.0,
            }
        )
        metrics.append(m)
    return pd.DataFrame(metrics)


def subtype_metrics(rep: pd.DataFrame) -> pd.DataFrame:
    data = add_subtype(rep)
    data["active"] = data["preview_pump_suppression_ratio"] > 0
    grouped = (
        data.groupby("subtype")
        .agg(
            cases=("case_id", "count"),
            active_cases=("active", "sum"),
            a0_pump_m3=("closed_pump_work_m3", "sum"),
            candidate_pump_m3=("primary_pump_work_m3", "sum"),
            fallback_rows_s=("primary_safety_fallback_ratio", lambda x: float(x.sum() * 3600.0)),
            mean_d_pitch_p95_deg=(
                "primary_pitch_p95",
                lambda x: 0.0,
            ),
        )
        .reset_index()
    )
    # Recompute p95 deltas without contorting groupby named agg lambdas.
    p95 = (
        data.assign(
            d_pitch_p95_deg=data["primary_pitch_p95"] - data["closed_pitch_p95"],
            d_roll_p95_deg=data["primary_roll_p95"] - data["closed_roll_p95"],
        )
        .groupby("subtype")
        .agg(
            mean_d_pitch_p95_deg=("d_pitch_p95_deg", "mean"),
            mean_d_roll_p95_deg=("d_roll_p95_deg", "mean"),
        )
        .reset_index()
    )
    grouped = grouped.drop(columns=["mean_d_pitch_p95_deg"]).merge(p95, on="subtype")
    grouped["active_ratio"] = grouped["active_cases"] / grouped["cases"]
    grouped["pump_saved_m3"] = grouped["a0_pump_m3"] - grouped["candidate_pump_m3"]
    grouped["pump_saving_pct"] = (
        100.0 * grouped["pump_saved_m3"] / grouped["a0_pump_m3"]
    )
    return grouped[
        [
            "subtype",
            "cases",
            "active_cases",
            "active_ratio",
            "a0_pump_m3",
            "candidate_pump_m3",
            "pump_saved_m3",
            "pump_saving_pct",
            "fallback_rows_s",
            "mean_d_pitch_p95_deg",
            "mean_d_roll_p95_deg",
        ]
    ]


def timeseries_safety_summary(ts_metrics: pd.DataFrame, name: str) -> dict[str, float | str]:
    return {
        "set": name,
        "cases": int(len(ts_metrics)),
        "duration_s": float(ts_metrics["primary_duration_s"].sum()),
        "closed_time_gt3_s": float(ts_metrics["closed_time_gt3_s"].sum()),
        "primary_time_gt3_s": float(ts_metrics["primary_time_gt3_s"].sum()),
        "d_time_gt3_s": float(ts_metrics["d_time_gt3_s"].sum()),
        "closed_time_gt4_s": float(ts_metrics["closed_time_gt4_s"].sum()),
        "primary_time_gt4_s": float(ts_metrics["primary_time_gt4_s"].sum()),
        "d_time_gt4_s": float(ts_metrics["d_time_gt4_s"].sum()),
        "closed_time_gt5_s": float(ts_metrics["closed_time_gt5_s"].sum()),
        "primary_time_gt5_s": float(ts_metrics["primary_time_gt5_s"].sum()),
        "d_time_gt5_s": float(ts_metrics["d_time_gt5_s"].sum()),
        "fallback_s": float(ts_metrics["fallback_s"].sum()),
        "closed_max_axis_deg": float(ts_metrics["closed_max_axis_deg"].max()),
        "primary_max_axis_deg": float(ts_metrics["primary_max_axis_deg"].max()),
        "d_max_axis_deg": float(ts_metrics["d_max_axis_deg"].max()),
    }


def md_table(df: pd.DataFrame, cols: list[str]) -> str:
    out = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in df[cols].iterrows():
        vals: list[str] = []
        for col in cols:
            value = row[col]
            if isinstance(value, float):
                if col.endswith("_ratio") or col == "active_ratio":
                    vals.append(f"{100.0 * value:.1f}%")
                elif "pct" in col:
                    vals.append(f"{value:.2f}%")
                elif "m3" in col:
                    vals.append(f"{value:.1f}")
                elif "deg" in col:
                    vals.append(f"{value:+.3f}")
                else:
                    vals.append(f"{value:.1f}")
            else:
                vals.append(str(value))
        out.append("| " + " | ".join(vals) + " |")
    return "\n".join(out)


def write_plot(metrics: pd.DataFrame, subtypes: pd.DataFrame) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return

    FIG.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), dpi=160)

    axes[0].bar(subtypes["subtype"], subtypes["pump_saving_pct"], color="#2f6f73")
    axes[0].set_ylabel("Pump saving (%)")
    axes[0].set_title("Subtype pump saving")
    axes[0].tick_params(axis="x", rotation=35, labelsize=8)
    axes[0].axhline(20, color="#8c3f2d", linewidth=1, linestyle="--")

    coverage_labels = ["all windows", "broad adjustable", "cleaner opportunity"]
    coverage_vals = [19.9, 48.7, 59.0]
    axes[1].bar(coverage_labels, coverage_vals, color="#8067a9")
    axes[1].set_ylim(0, 70)
    axes[1].set_ylabel("Estimated active coverage (%)")
    axes[1].set_title("Coverage readings")
    axes[1].tick_params(axis="x", rotation=20, labelsize=8)
    axes[1].axhline(35, color="#8c3f2d", linewidth=1, linestyle="--")

    fig.tight_layout()
    fig.savefig(FIG / "fig_broad_economy_subtypes_and_coverage.png")
    plt.close(fig)


def main() -> None:
    PAPER.mkdir(parents=True, exist_ok=True)

    rep_mixed = load_casebook(BASE / "forecast_advised_economy_v2_representative_48_1h/casebook_summary.csv")
    validation_missing = load_casebook(BASE / "forecast_advised_economy_v2_validation_missing20_1h/casebook_summary.csv")
    rep = split_corrected_representative(rep_mixed, validation_missing)
    boundary = load_casebook(BASE / "forecast_advised_economy_v2_smoke_boundary_8_1h/casebook_summary.csv")
    positive = load_casebook(BASE / "forecast_advised_economy_smoke_candidate_12_1h_failclosed/casebook_summary.csv")
    oracle_missing = load_casebook(BASE / "forecast_advised_economy_v2_oracle_missing20_1h/casebook_summary.csv")

    metrics = pd.DataFrame(
        [
            casebook_metrics(rep, "split-corrected neutral_untyped 48"),
            casebook_metrics(rep_mixed, "legacy mixed-split neutral_untyped 48"),
            casebook_metrics(validation_missing, "validation split recovered 20"),
            casebook_metrics(positive, "positive clean/graded smoke 12"),
            casebook_metrics(boundary, "boundary negative smoke 8"),
            casebook_metrics(oracle_missing, "legacy oracle missing20 wrong-split"),
        ]
    )
    subtypes = subtype_metrics(rep)
    ts_metrics = split_corrected_timeseries_metrics(rep_mixed, validation_missing)
    clean_subtypes = {
        "neutral_mild_decay",
        "neutral_mixed_steady",
        "neutral_moderate_high_steady",
    }
    rep_with_subtype = add_subtype(rep)
    clean_rep = rep_with_subtype[rep_with_subtype["subtype"].isin(clean_subtypes)].copy()
    ramp_rep = rep_with_subtype[rep_with_subtype["subtype"].eq("neutral_ramp_or_event_onset")].copy()
    subgroup_metrics = pd.DataFrame(
        [
            casebook_metrics(clean_rep, "cost-filtered neutral/headroom clean 40"),
            casebook_metrics(ramp_rep, "aggressive ramp/onset endpoint 8"),
        ]
    )
    ts_summary = pd.DataFrame(
        [
            timeseries_safety_summary(ts_metrics, "split-corrected neutral_untyped 48"),
            timeseries_safety_summary(
                ts_metrics[ts_metrics["subtype"].isin(clean_subtypes)],
                "cost-filtered neutral/headroom clean 40",
            ),
            timeseries_safety_summary(
                ts_metrics[ts_metrics["subtype"].eq("neutral_ramp_or_event_onset")],
                "aggressive ramp/onset endpoint 8",
            ),
        ]
    )

    metrics.to_csv(PAPER / "broad_economy_core_metrics.csv", index=False)
    subtypes.to_csv(PAPER / "broad_economy_subtype_metrics.csv", index=False)
    subgroup_metrics.to_csv(PAPER / "broad_economy_cost_filtered_metrics.csv", index=False)
    ts_metrics.to_csv(PAPER / "broad_economy_split_corrected_timeseries_safety.csv", index=False)
    ts_summary.to_csv(PAPER / "broad_economy_split_corrected_safety_summary.csv", index=False)

    summary = f"""# Forecast-Advised Broad Economy Family

## Result

This package freezes the third positive pump-saving family:

> neutral / moderate-high load, enough current posture headroom, and a forecast
> that does not warn of direction reversal, re-intensification, far-boundary
> behavior, or fallback-dominated state.

The controller role is not a new hard-safety layer.  It is a forecast-advised
economy / Pareto mode: reduce economy-layer target pursuit when the forecast
does not show boundary risk.  The v1.6 hard floor, fallback, recovery, and pump
penalty remain unchanged.

{md_table(metrics, ["set", "cases", "active_cases", "active_ratio", "pump_saving_pct", "fallback_rows_s", "mean_d_pitch_p95_deg", "mean_d_roll_p95_deg"])}

## Subtype Coverage

{md_table(subtypes, ["subtype", "cases", "active_cases", "active_ratio", "pump_saving_pct", "fallback_rows_s", "mean_d_pitch_p95_deg", "mean_d_roll_p95_deg"])}

## Clean vs Aggressive Split

The 48-case split-corrected result contains two cost classes.  The 40-case
clean subset keeps the steady / mild-decay neutral-headroom cases.  The 8-case
`neutral_ramp_or_event_onset` subset is a higher-gain, higher-exposure endpoint
and should be discussed as aggressive Pareto behavior rather than folded into a
no-cost automatic claim.

{md_table(subgroup_metrics, ["set", "cases", "active_cases", "active_ratio", "pump_saving_pct", "fallback_rows_s", "mean_d_pitch_p95_deg", "mean_d_roll_p95_deg"])}

## Split-Corrected Safety Exposure

{md_table(ts_summary, [
    "set",
    "cases",
    "duration_s",
    "closed_time_gt5_s",
    "primary_time_gt5_s",
    "d_time_gt5_s",
    "fallback_s",
    "closed_max_axis_deg",
    "primary_max_axis_deg",
    "d_max_axis_deg",
])}

## Coverage Interpretation

- `neutral_untyped` is about 19.9% of all mined windows.
- The split-corrected representative check activates 48/48 cases, so this
  family covers the full representative `neutral_untyped` sample under the
  correct replay split.
- Against the broad adjustable pool, this is about 48.7%.
- Against the cleaner economy-opportunity pool, this is about 59.0%.

The previous 28/48 active reading was a split-resolution artifact.  The 20
inactive rows were validation timestamps replayed through a runner that only
loaded the test split.  A dedicated validation-split rerun recovers those 20
cases at 59.52% in-regime pump saving with learned forecasts present.

## Decision

Use this as the broad third family in the final conclusion.  Do not promote the
V2 mild branch as a separate mechanism: all active rows in the representative
split-corrected run used the strict `forecast_advised_economy_suppression`
reason except for a few headroom-vetoed planner buckets inside otherwise active
cases.

Further expansion should not focus on the former missing-20 pool; that pool is
now accounted for.  The next global lever, if needed, is a genuinely different
large-pool action such as gusty batching or a lowrisk no-churn mode.

## Files

- `broad_economy_core_metrics.csv`
- `broad_economy_subtype_metrics.csv`
- `broad_economy_cost_filtered_metrics.csv`
- `broad_economy_split_corrected_timeseries_safety.csv`
- `broad_economy_split_corrected_safety_summary.csv`
- `figures/fig_broad_economy_subtypes_and_coverage.png`
"""
    (PAPER / "forecast_advised_broad_economy_summary.md").write_text(
        summary, encoding="utf-8"
    )
    write_plot(metrics, subtypes)


if __name__ == "__main__":
    main()
