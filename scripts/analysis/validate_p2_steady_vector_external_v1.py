"""Validate the P2 steady-vector rule on held-out P2-like runs.

This is a read-only audit. It reuses existing closed-only and
prediction-primary timeseries plus planner logs, then evaluates the
candidate P2-2 selectors discovered in the P0 bucket audit.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "outputs/wind_prediction/regime_conditioned_policy_development_v1"
CONT = BASE / "continuous_episode_validation_v1"
OUT = CONT / "p2_external_steady_vector_validation_v1"
BUCKET_S = 600.0


@dataclass(frozen=True)
class RunSpec:
    name: str
    run_dir: Path
    primary_token: str


RUNS = [
    RunSpec(
        name="top2_long_32h",
        run_dir=CONT / "p2_long_real_top2_guarded_stale_32h",
        primary_token="p2_long_real_guarded_stale",
    ),
    RunSpec(
        name="p2g_12x6h",
        run_dir=CONT
        / "p2_golden_opportunity_library_v1"
        / "p2g_smoke_6h_guarded_stale",
        primary_token="p2g_guarded_stale",
    ),
]


def _case_prefix(path: Path, suffix: str) -> str:
    name = path.name
    if not name.endswith(suffix):
        raise ValueError(f"Unexpected suffix for {path}")
    return name[: -len(suffix)]


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False)


def _time_step_seconds(df: pd.DataFrame) -> pd.Series:
    t = pd.to_numeric(df["t_s"], errors="coerce")
    dt = t.shift(-1) - t
    positive = dt[dt > 0]
    fallback = float(positive.median()) if not positive.empty else 20.0
    return dt.where(dt > 0, fallback).fillna(fallback)


def _bucket_timeseries(path: Path, prefix: str) -> pd.DataFrame:
    df = _read_csv(path)
    if "t_s" not in df.columns:
        raise ValueError(f"{path} has no t_s")
    t = pd.to_numeric(df["t_s"], errors="coerce").fillna(0.0)
    dt = _time_step_seconds(df)
    rate = pd.to_numeric(df.get("pump_total_rate_m3_min", 0.0), errors="coerce").fillna(0.0)
    pitch = pd.to_numeric(df.get("pitch_deg", 0.0), errors="coerce").fillna(0.0).abs()
    roll = pd.to_numeric(df.get("roll_deg", 0.0), errors="coerce").fillna(0.0).abs()
    max_axis = np.maximum(pitch, roll)
    latch = pd.to_numeric(df.get("pump_latch_switch_count", 0.0), errors="coerce").fillna(0.0)
    fallback_col = pd.to_numeric(
        df.get("preview_primary_safety_fallback", 0.0), errors="coerce"
    ).fillna(0.0)

    work = rate.clip(lower=0.0) * dt / 60.0
    bucket_s = np.floor(t / BUCKET_S) * BUCKET_S
    tmp = pd.DataFrame(
        {
            "case": prefix,
            "bucket_s": bucket_s,
            "pump_m3": work,
            "dt_s": dt,
            "time_gt3_s": np.where(max_axis > 3.0, dt, 0.0),
            "time_gt4_s": np.where(max_axis > 4.0, dt, 0.0),
            "time_gt5_s": np.where(max_axis > 5.0, dt, 0.0),
            "fallback_s": np.where(fallback_col > 0.5, dt, 0.0),
            "max_axis": max_axis,
            "latch": latch,
        }
    )
    grouped = (
        tmp.groupby(["case", "bucket_s"], as_index=False)
        .agg(
            pump_m3=("pump_m3", "sum"),
            duration_s=("dt_s", "sum"),
            time_gt3_s=("time_gt3_s", "sum"),
            time_gt4_s=("time_gt4_s", "sum"),
            time_gt5_s=("time_gt5_s", "sum"),
            fallback_s=("fallback_s", "sum"),
            max_axis_p95=("max_axis", lambda x: float(np.nanpercentile(x, 95))),
            max_axis_max=("max_axis", "max"),
            latch_first=("latch", "first"),
            latch_last=("latch", "last"),
        )
        .copy()
    )
    grouped["latch_delta"] = grouped["latch_last"] - grouped["latch_first"]
    return grouped


def _bucket_planner(path: Path, prefix: str) -> pd.DataFrame:
    df = _read_csv(path)
    if "current_time_s" not in df.columns:
        raise ValueError(f"{path} has no current_time_s")
    t = pd.to_numeric(df["current_time_s"], errors="coerce").fillna(0.0)
    out = pd.DataFrame(
        {
            "case": prefix,
            "bucket_s": np.floor(t / BUCKET_S) * BUCKET_S,
            "raw_pitch_abs_change_0_2": (
                pd.to_numeric(df.get("raw_pressure_block2_pitch", np.nan), errors="coerce")
                - pd.to_numeric(df.get("raw_pressure_block0_pitch", np.nan), errors="coerce")
            ).abs(),
            "raw_roll_abs_change_0_2": (
                pd.to_numeric(df.get("raw_pressure_block2_roll", np.nan), errors="coerce")
                - pd.to_numeric(df.get("raw_pressure_block0_roll", np.nan), errors="coerce")
            ).abs(),
            "event_risk_prob_0_20m": pd.to_numeric(
                df.get("event_risk_prob_0_20m", np.nan), errors="coerce"
            ),
            "event_risk_prob_20_40m": pd.to_numeric(
                df.get("event_risk_prob_20_40m", np.nan), errors="coerce"
            ),
            "event_risk_prob_40_60m": pd.to_numeric(
                df.get("event_risk_prob_40_60m", np.nan), errors="coerce"
            ),
            "forecast_advised_economy_candidate": pd.to_numeric(
                df.get("forecast_advised_economy_candidate", 0.0), errors="coerce"
            ).fillna(0.0),
            "forecast_advised_economy_confirmed": pd.to_numeric(
                df.get("forecast_advised_economy_confirmed", 0.0), errors="coerce"
            ).fillna(0.0),
            "forecast_advised_economy_boundary_veto": pd.to_numeric(
                df.get("forecast_advised_economy_boundary_veto", 0.0), errors="coerce"
            ).fillna(0.0),
            "forecast_advised_economy_headroom_deg": pd.to_numeric(
                df.get("forecast_advised_economy_headroom_deg", np.nan), errors="coerce"
            ),
            "forecast_advised_economy_speed_range_ms": pd.to_numeric(
                df.get("forecast_advised_economy_speed_range_ms", np.nan), errors="coerce"
            ),
        }
    )
    return out.groupby(["case", "bucket_s"], as_index=False).mean(numeric_only=True)


def _load_run(spec: RunSpec) -> pd.DataFrame:
    timeseries = spec.run_dir / "timeseries"
    logs = spec.run_dir / "planner_logs"
    rows: list[pd.DataFrame] = []
    for closed_path in sorted(timeseries.glob("*_closed_only_timeseries.csv")):
        prefix = _case_prefix(closed_path, "_closed_only_timeseries.csv")
        primary_path = timeseries / f"{prefix}_{spec.primary_token}_timeseries.csv"
        planner_path = logs / f"{prefix}_{spec.primary_token}_planner_log.csv"
        if not primary_path.exists() or not planner_path.exists():
            raise FileNotFoundError(
                f"Missing primary/log for {prefix}: {primary_path.exists()} {planner_path.exists()}"
            )
        closed = _bucket_timeseries(closed_path, prefix).add_prefix("closed_")
        primary = _bucket_timeseries(primary_path, prefix).add_prefix("primary_")
        features = _bucket_planner(planner_path, prefix)
        merged = closed.merge(
            primary,
            left_on=["closed_case", "closed_bucket_s"],
            right_on=["primary_case", "primary_bucket_s"],
            how="inner",
        )
        merged = merged.merge(
            features,
            left_on=["closed_case", "closed_bucket_s"],
            right_on=["case", "bucket_s"],
            how="left",
        )
        merged["run"] = spec.name
        merged["case"] = prefix
        merged["bucket_s"] = merged["closed_bucket_s"]
        rows.append(merged)
    return pd.concat(rows, ignore_index=True)


def _summarize_selector(df: pd.DataFrame, selector: str, mask: pd.Series) -> dict[str, float | str]:
    sub = df[mask.fillna(False)].copy()
    closed = float(sub["closed_pump_m3"].sum())
    primary = float(sub["primary_pump_m3"].sum())
    saving = closed - primary
    return {
        "selector": selector,
        "buckets": int(len(sub)),
        "cases": int(sub["case"].nunique()) if not sub.empty else 0,
        "closed_pump_m3": closed,
        "primary_pump_m3": primary,
        "saving_m3": saving,
        "saving_pct": (100.0 * saving / closed) if closed > 0 else np.nan,
        "closed_latch_delta": float(sub["closed_latch_delta"].sum()),
        "primary_latch_delta": float(sub["primary_latch_delta"].sum()),
        "d_time_gt3_s": float(sub["primary_time_gt3_s"].sum() - sub["closed_time_gt3_s"].sum()),
        "d_time_gt4_s": float(sub["primary_time_gt4_s"].sum() - sub["closed_time_gt4_s"].sum()),
        "d_time_gt5_s": float(sub["primary_time_gt5_s"].sum() - sub["closed_time_gt5_s"].sum()),
        "d_fallback_s": float(sub["primary_fallback_s"].sum() - sub["closed_fallback_s"].sum()),
        "max_primary_axis_p95": float(sub["primary_max_axis_p95"].max()) if not sub.empty else np.nan,
        "mean_raw_pitch_abs_change_0_2": float(sub["raw_pitch_abs_change_0_2"].mean())
        if not sub.empty
        else np.nan,
    }


def _summaries(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    confirmed = df["forecast_advised_economy_confirmed"] > 0.5
    candidate = df["forecast_advised_economy_candidate"] > 0.5
    no_boundary = df["forecast_advised_economy_boundary_veto"] < 0.5
    pitch006 = df["raw_pitch_abs_change_0_2"] <= 0.00603172366334895
    pitch008 = df["raw_pitch_abs_change_0_2"] <= 0.0082843397176601

    selectors = {
        "all_buckets": pd.Series(True, index=df.index),
        "p2_candidate": candidate,
        "p2_confirmed": confirmed,
        "p2_confirmed_no_boundary": confirmed & no_boundary,
        "p2_steady_pitch006": confirmed & no_boundary & pitch006,
        "p2_steady_pitch008": confirmed & no_boundary & pitch008,
    }
    summary_rows = []
    for run_name, run_df in df.groupby("run"):
        for name, mask in selectors.items():
            row = _summarize_selector(run_df, name, mask.loc[run_df.index])
            row["run"] = run_name
            summary_rows.append(row)
    for name, mask in selectors.items():
        row = _summarize_selector(df, name, mask)
        row["run"] = "combined_external"
        summary_rows.append(row)
    selector_summary = pd.DataFrame(summary_rows)

    case_rows = []
    for (run_name, case), case_df in df.groupby(["run", "case"]):
        for name, mask in selectors.items():
            row = _summarize_selector(case_df, name, mask.loc[case_df.index])
            row["run"] = run_name
            row["case"] = case
            case_rows.append(row)
    case_summary = pd.DataFrame(case_rows)
    return selector_summary, case_summary


def _write_decision(selector_summary: pd.DataFrame, case_summary: pd.DataFrame) -> None:
    def to_md(view: pd.DataFrame) -> str:
        cols = list(view.columns)
        lines = [
            "|" + "|".join(cols) + "|",
            "|" + "|".join(["---"] * len(cols)) + "|",
        ]
        for _, row in view.iterrows():
            lines.append("|" + "|".join(str(row[col]) for col in cols) + "|")
        return "\n".join(lines)

    def table_for(run: str) -> str:
        cols = [
            "selector",
            "buckets",
            "cases",
            "closed_pump_m3",
            "saving_m3",
            "saving_pct",
            "d_time_gt5_s",
            "d_fallback_s",
            "primary_latch_delta",
        ]
        view = selector_summary[selector_summary["run"] == run][cols].copy()
        for col in ["closed_pump_m3", "saving_m3", "saving_pct", "d_time_gt5_s", "d_fallback_s", "primary_latch_delta"]:
            view[col] = view[col].map(lambda x: f"{x:.3f}" if pd.notna(x) else "")
        return to_md(view)

    combined = selector_summary[selector_summary["run"] == "combined_external"].set_index("selector")
    p2 = combined.loc["p2_confirmed"]
    p22 = combined.loc["p2_steady_pitch008"]
    p22_strict = combined.loc["p2_steady_pitch006"]
    lines = [
        "# P2-2 steady-vector external validation",
        "",
        "This read-only audit applies the P0 steady-vector candidate rules to two external P2-like sets:",
        "",
        "- `top2_long_32h`: two long natural P2-rich scenes not used to discover the rule.",
        "- `p2g_12x6h`: twelve stricter P2-G scenes where wind-shape filtering previously failed.",
        "",
        "The tested rule is internal to P2: `forecast_advised_economy_confirmed` and no boundary veto, plus low raw pitch pressure change from block 0 to block 2.",
        "",
        "## Combined verdict",
        "",
        f"- Plain confirmed P2 external saving: `{p2['saving_pct']:.2f}%` over `{p2['closed_pump_m3']:.1f} m3` closed-only pump.",
        f"- P2-2 pitch<=0.008 saving: `{p22['saving_pct']:.2f}%` over `{p22['closed_pump_m3']:.1f} m3` closed-only pump.",
        f"- P2-2 pitch<=0.006 saving: `{p22_strict['saving_pct']:.2f}%` over `{p22_strict['closed_pump_m3']:.1f} m3` closed-only pump.",
        "",
    ]
    if p22["saving_pct"] >= p2["saving_pct"] + 5.0 and p22["saving_pct"] >= 10.0:
        lines.extend(
            [
                "**Decision:** P2-2 improves external selection and is worth one runtime-gate smoke test.",
                "Keep the rule narrow and default-off; do not present it as final until closed-loop validation confirms the same gain.",
            ]
        )
    else:
        lines.extend(
            [
                "**Decision:** P2-2 does not externally rescue P2. The top4 P0 gain was not robust enough to promote.",
                "Do not build a new controller gate from this rule without another independent signal.",
            ]
        )
    lines.extend(
        [
            "",
            "## Selector summary: combined external",
            "",
            table_for("combined_external"),
            "",
            "## Selector summary: top2 long scenes",
            "",
            table_for("top2_long_32h"),
            "",
            "## Selector summary: P2-G 12x6h",
            "",
            table_for("p2g_12x6h"),
            "",
            "## Notes",
            "",
            "- This is not a new simulation; it evaluates existing matched closed-only vs prediction-primary traces.",
            "- The numbers are bucket-subset estimates. A runtime gate still needs a closed-loop smoke run because changing activation changes later posture and catch-up.",
            "- The rule was discovered on top4 12h P2 buckets, so top2/P2-G are treated as external checks.",
        ]
    )
    (OUT / "decision.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    all_runs = pd.concat([_load_run(spec) for spec in RUNS], ignore_index=True)
    all_runs["saving_m3"] = all_runs["closed_pump_m3"] - all_runs["primary_pump_m3"]
    all_runs["saving_pct_bucket"] = np.where(
        all_runs["closed_pump_m3"] > 0,
        100.0 * all_runs["saving_m3"] / all_runs["closed_pump_m3"],
        np.nan,
    )
    selector_summary, case_summary = _summaries(all_runs)
    all_runs.to_csv(OUT / "bucket_level_with_p2_steady_vector_features.csv", index=False)
    selector_summary.to_csv(OUT / "selector_summary.csv", index=False)
    case_summary.to_csv(OUT / "case_selector_summary.csv", index=False)
    _write_decision(selector_summary, case_summary)
    print(OUT / "decision.md")


if __name__ == "__main__":
    main()
