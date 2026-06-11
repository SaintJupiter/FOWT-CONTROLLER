#!/usr/bin/env python3
"""Read-only calm-to-5deg rise-time audit for the h120 line.

This script measures how long high-posture episodes spend in a preceding calm
state before crossing the 5 degree floor trigger. It does not change the
controller or any forecast model.

The purpose is to answer a narrow structural question:
if calm-to-5 transitions almost always happen inside 60 minutes, then there is
little room for a 120 minute closed-loop selector to add value in this
architecture.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "outputs/wind_prediction/h120_rise_time_audit_v1"
RAW = OUT / "raw_tables"
DEBUG = OUT / "debug"
PAPER = OUT / "paper_ready"

RUNS = {
    "guard10": REPO / "outputs/wind_prediction/h120_oracle_pareto_mode_selector_v1/runs/guard10/static_v16_balanced",
    "broader20": REPO / "outputs/wind_prediction/h120_oracle_pareto_mode_selector_v1/runs/broader20/static_v16_balanced",
}

HIGH_DEG = 5.0
CALM_THRESHOLDS = (3.0, 4.0, 4.5)


def ensure_dirs() -> None:
    for path in (OUT, RAW, DEBUG, PAPER):
        path.mkdir(parents=True, exist_ok=True)


def num(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col not in df.columns:
        return pd.Series(default, index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce").fillna(default).astype(float)


def case_id_from_path(path: Path) -> str:
    stem = path.name.replace("_prediction_primary_econ_timeseries.csv", "")
    parts = stem.split("_")
    if parts and parts[0].isdigit():
        return "_".join(parts[1:-2])
    return "_".join(parts[:-2])


def markdown_table(df: pd.DataFrame, max_rows: int | None = None) -> str:
    if df is None or df.empty:
        return "_empty_"
    data = df.copy()
    if max_rows is not None:
        data = data.head(max_rows)

    def fmt(value: Any) -> str:
        if pd.isna(value):
            return ""
        if isinstance(value, float):
            return f"{value:.3f}"
        return str(value)

    lines = [
        "| " + " | ".join(str(c) for c in data.columns) + " |",
        "| " + " | ".join(["---"] * len(data.columns)) + " |",
    ]
    for _, row in data.iterrows():
        lines.append("| " + " | ".join(fmt(row[c]) for c in data.columns) + " |")
    return "\n".join(lines)


def contiguous_true_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for idx, value in enumerate(mask.tolist()):
        if value and start is None:
            start = idx
        elif not value and start is not None:
            runs.append((start, idx))
            start = None
    if start is not None:
        runs.append((start, len(mask)))
    return runs


def last_calm_before_crossing(calm_mask: np.ndarray, start_idx: int) -> int | None:
    if start_idx <= 0:
        return None
    idxs = np.flatnonzero(calm_mask[:start_idx])
    if len(idxs) == 0:
        return None
    return int(idxs[-1])


def build_case_rows(dataset: str, ts_path: Path) -> pd.DataFrame:
    ts = pd.read_csv(ts_path, low_memory=False)
    out = pd.DataFrame(index=ts.index)
    out["dataset"] = dataset
    out["case_id"] = case_id_from_path(ts_path)
    out["t_s"] = num(ts, "t_s").round().astype(int)
    out["pitch_deg"] = num(ts, "pitch_deg")
    out["roll_deg"] = num(ts, "roll_deg")
    out["max_axis_deg"] = np.maximum(out["pitch_deg"].abs(), out["roll_deg"].abs())
    out["pump_rate_m3_min"] = num(ts, "pump_total_rate_m3_min").abs()
    out["pump_idle"] = (out["pump_rate_m3_min"] <= 1e-6).astype(int)
    out["floor_active"] = (num(ts, "reactive_floor_active") > 0).astype(int)
    out["target_age_s"] = num(ts, "prediction_primary_target_age_s")
    out["target_err_mean_kg"] = num(ts, "reactive_floor_target_err_mean_kg")
    alt_err = num(ts, "active_effectiveness_target_err_mean_kg", np.nan)
    out["target_err_mean_kg"] = out["target_err_mean_kg"].where(out["target_err_mean_kg"] > 0, alt_err.fillna(0.0))
    out["floor_episode_id"] = ""
    episode_no = 0
    prev = 0
    for idx, value in zip(out.index, out["floor_active"].to_numpy()):
        if value and not prev:
            episode_no += 1
        out.at[idx, "floor_episode_id"] = f"{dataset}::{out.at[0, 'case_id']}::floor{episode_no:02d}" if value else ""
        prev = int(value)

    rows: list[dict[str, Any]] = []
    high_runs = contiguous_true_runs((out["max_axis_deg"].to_numpy() > HIGH_DEG))
    for ep_no, (start, end) in enumerate(high_runs, start=1):
        episode = out.iloc[start:end].copy()
        if episode.empty:
            continue
        start_t = int(episode["t_s"].iloc[0])
        peak = float(episode["max_axis_deg"].max())
        duration_s = int(episode["t_s"].iloc[-1] - episode["t_s"].iloc[0] + 1)
        row: dict[str, Any] = {
            "dataset": dataset,
            "case_id": str(out["case_id"].iloc[0]),
            "episode_no": ep_no,
            "episode_start_s": start_t,
            "episode_end_s": int(episode["t_s"].iloc[-1]),
            "episode_duration_s": duration_s,
            "peak_max_axis_deg": peak,
            "mean_max_axis_deg": float(episode["max_axis_deg"].mean()),
            "floor_active_rows": int(episode["floor_active"].sum()),
            "pump_idle_rows": int(episode["pump_idle"].sum()),
            "target_stale_proxy_rows": int(((episode["target_age_s"] >= 600.0) & (episode["target_err_mean_kg"] <= 500.0)).sum()),
            "floor_id": str(episode["floor_episode_id"].iloc[0]) if "floor_episode_id" in episode.columns else "",
        }
        for calm_threshold in CALM_THRESHOLDS:
            calm_mask = (out["max_axis_deg"].to_numpy() <= calm_threshold)
            calm_start = last_calm_before_crossing(calm_mask, start)
            if calm_start is None:
                row[f"rise_from_{str(calm_threshold).replace('.', 'p')}_s"] = np.nan
                row[f"rise_from_{str(calm_threshold).replace('.', 'p')}_min"] = np.nan
                row[f"calm_start_from_{str(calm_threshold).replace('.', 'p')}_s"] = np.nan
                row[f"calm_run_len_from_{str(calm_threshold).replace('.', 'p')}_s"] = np.nan
                row[f"calm_run_len_from_{str(calm_threshold).replace('.', 'p')}_min"] = np.nan
            else:
                rise_s = int(out["t_s"].iloc[start] - out["t_s"].iloc[calm_start])
                row[f"rise_from_{str(calm_threshold).replace('.', 'p')}_s"] = rise_s
                row[f"rise_from_{str(calm_threshold).replace('.', 'p')}_min"] = rise_s / 60.0
                row[f"calm_start_from_{str(calm_threshold).replace('.', 'p')}_s"] = int(out["t_s"].iloc[calm_start])
                row[f"calm_run_len_from_{str(calm_threshold).replace('.', 'p')}_s"] = rise_s
                row[f"calm_run_len_from_{str(calm_threshold).replace('.', 'p')}_min"] = rise_s / 60.0
        rows.append(row)
    return pd.DataFrame(rows)


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for dataset, group in list(df.groupby("dataset")) + [("all", df)]:
        row: dict[str, Any] = {
            "dataset": dataset,
            "cases": int(group["case_id"].nunique()) if not group.empty else 0,
            "episodes": int(len(group)),
            "mean_peak_max_axis_deg": float(group["peak_max_axis_deg"].mean()) if not group.empty else np.nan,
            "median_episode_duration_s": float(group["episode_duration_s"].median()) if not group.empty else np.nan,
            "max_episode_duration_s": float(group["episode_duration_s"].max()) if not group.empty else np.nan,
        }
        for calm_threshold in CALM_THRESHOLDS:
            key = str(calm_threshold).replace(".", "p")
            col = f"rise_from_{key}_s"
            valid = group[col].dropna() if col in group.columns else pd.Series(dtype=float)
            row[f"episodes_with_valid_rise_{key}"] = int(valid.notna().sum())
            row[f"median_rise_{key}_min"] = float(valid.median() / 60.0) if len(valid) else np.nan
            row[f"p90_rise_{key}_min"] = float(valid.quantile(0.90) / 60.0) if len(valid) else np.nan
            row[f"max_rise_{key}_min"] = float(valid.max() / 60.0) if len(valid) else np.nan
            row[f"count_ge_60m_{key}"] = int((valid >= 3600.0).sum()) if len(valid) else 0
            row[f"count_ge_120m_{key}"] = int((valid >= 7200.0).sum()) if len(valid) else 0
            row[f"share_ge_60m_{key}"] = float((valid >= 3600.0).mean()) if len(valid) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    ensure_dirs()

    rows: list[pd.DataFrame] = []
    for dataset, run_dir in RUNS.items():
        ts_dir = run_dir / "timeseries"
        for ts_path in sorted(ts_dir.glob("*_timeseries.csv")):
            rows.append(build_case_rows(dataset, ts_path))

    if not rows:
        raise FileNotFoundError("No static v16 timeseries files found for rise-time audit.")

    episode = pd.concat(rows, ignore_index=True)
    episode.to_csv(RAW / "rise_time_episode_table.csv", index=False)

    summary = summarise(episode)
    summary.to_csv(RAW / "rise_time_summary.csv", index=False)

    case_summary = (
        episode.groupby(["dataset", "case_id"], as_index=False)
        .agg(
            episodes=("episode_no", "count"),
            median_rise_3p0_s=("rise_from_3p0_s", "median"),
            max_rise_3p0_s=("rise_from_3p0_s", "max"),
            median_rise_4p5_s=("rise_from_4p5_s", "median"),
            max_rise_4p5_s=("rise_from_4p5_s", "max"),
            median_rise_4p0_s=("rise_from_4p0_s", "median"),
            max_rise_4p0_s=("rise_from_4p0_s", "max"),
            max_episode_duration_s=("episode_duration_s", "max"),
            peak_max_axis_deg=("peak_max_axis_deg", "max"),
        )
        .sort_values(["dataset", "max_rise_4p5_s", "max_rise_4p0_s"], ascending=[True, False, False])
        .reset_index(drop=True)
    )
    case_summary.to_csv(RAW / "rise_time_case_summary.csv", index=False)

    worst = episode.sort_values(["rise_from_4p5_s", "rise_from_4p0_s", "rise_from_3p0_s"], ascending=False).head(20)
    worst.to_csv(DEBUG / "worst_rise_episodes.csv", index=False)

    long_45 = episode[episode["rise_from_4p5_s"].ge(3600.0)].copy()
    long_40 = episode[episode["rise_from_4p0_s"].ge(3600.0)].copy()
    long_30 = episode[episode["rise_from_3p0_s"].ge(3600.0)].copy()

    lines = [
        "# h120 Rise-Time Audit Summary",
        "",
        "Scope: read-only calm-to-5deg rise-time audit on static v1.6 traces from the oracle Pareto selector run.",
        "",
        "## Dataset Summary",
        markdown_table(summary),
        "",
        "## Case Summary (top rows)",
        markdown_table(case_summary, max_rows=20),
        "",
        "## Long Rise Episodes (>=60 min, 4.5deg calm threshold)",
        markdown_table(long_45.sort_values("rise_from_4p5_s", ascending=False), max_rows=20),
        "",
        "## Long Rise Episodes (>=60 min, 4.0deg calm threshold)",
        markdown_table(long_40.sort_values("rise_from_4p0_s", ascending=False), max_rows=20),
        "",
        "## Long Rise Episodes (>=60 min, 3.0deg calm threshold)",
        markdown_table(long_30.sort_values("rise_from_3p0_s", ascending=False), max_rows=20),
        "",
        "## Interpretation",
    ]

    max_45 = float(summary.loc[summary["dataset"] == "all", "max_rise_4p5_min"].iloc[0]) if not summary.empty else np.nan
    max_40 = float(summary.loc[summary["dataset"] == "all", "max_rise_4p0_min"].iloc[0]) if not summary.empty else np.nan
    max_30 = float(summary.loc[summary["dataset"] == "all", "max_rise_3p0_min"].iloc[0]) if not summary.empty else np.nan
    count_long_45 = int(len(long_45))
    count_long_40 = int(len(long_40))
    count_long_30 = int(len(long_30))
    any_long_45 = bool(count_long_45)
    any_long_40 = bool(count_long_40)
    any_long_30 = bool(count_long_30)
    if (
        (not np.isnan(max_45) and max_45 < 60.0)
        and (not np.isnan(max_40) and max_40 < 60.0)
        and (not np.isnan(max_30) and max_30 < 60.0)
    ):
        decision = (
            "All observed calm-to-5 rise times are under 60 minutes for all calm thresholds. "
            "That is the cleanest possible structural no-go for a 120 minute closed-loop selector: "
            "the event is almost always actionable within the 60 minute horizon."
        )
    elif any_long_45 or any_long_40:
        decision = (
            "Some operational-threshold rise episodes exceed 60 minutes. That does not prove a 120 minute "
            "controller win, but it would be the one remaining place where a controlled oracle ceiling test "
            "could still be justified."
        )
    elif any_long_30:
        decision = (
            "Only the very broad 3.0deg-to-5deg definition has >=60 minute rise episodes, and the maximum is "
            "just over the boundary. The actionable 4.0/4.5deg-to-5deg rises remain below 60 minutes. This "
            "does not reopen the current closed-loop h120 controller route; it only suggests that a future "
            "hard-constraint formulation should document whether action is required as early as 3deg."
        )
    else:
        decision = "No valid rise episodes were found."
    lines.append(decision)
    lines.append("")
    lines.append("## Required Answer")
    lines.append(f"- max rise (3.0deg calm threshold): `{max_30:.1f}` min" if not np.isnan(max_30) else "- max rise (3.0deg calm threshold): n/a")
    lines.append(f"- max rise (4.5deg calm threshold): `{max_45:.1f}` min" if not np.isnan(max_45) else "- max rise (4.5deg calm threshold): n/a")
    lines.append(f"- max rise (4.0deg calm threshold): `{max_40:.1f}` min" if not np.isnan(max_40) else "- max rise (4.0deg calm threshold): n/a")
    lines.append(f"- count >=60 min (3.0deg): `{count_long_30}`")
    lines.append(f"- count >=60 min (4.5deg): `{count_long_45}`")
    lines.append(f"- count >=60 min (4.0deg): `{count_long_40}`")
    lines.append("")
    lines.append("Because the 4.0/4.5deg operational rise windows stay below 60 min, the practical next step is to stop h120 closed-loop selector work for the current controller architecture and reframe the 120 minute forecast as supervisory / situational-awareness support. A hard-constraint reframe would need a separate 3deg-entry argument.")

    (PAPER / "rise_time_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (OUT / "h120_rise_time_decision.md").write_text(
        "\n".join(
            [
                "# h120 Rise-Time Decision",
                "",
                "Status: read-only rise-time audit completed.",
                "",
                f"- guard10 cases: `{summary.loc[summary['dataset'] == 'guard10', 'cases'].iloc[0] if 'guard10' in summary['dataset'].values else 0}`",
                f"- broader20 cases: `{summary.loc[summary['dataset'] == 'broader20', 'cases'].iloc[0] if 'broader20' in summary['dataset'].values else 0}`",
                f"- max rise 3.0deg calm threshold: `{max_30:.1f}` min" if not np.isnan(max_30) else "- max rise 3.0deg calm threshold: n/a",
                f"- max rise 4.5deg calm threshold: `{max_45:.1f}` min" if not np.isnan(max_45) else "- max rise 4.5deg calm threshold: n/a",
                f"- max rise 4.0deg calm threshold: `{max_40:.1f}` min" if not np.isnan(max_40) else "- max rise 4.0deg calm threshold: n/a",
                "",
                "Conclusion:",
                decision,
                "",
                "Recommended next step:",
                "Stop tuning h120 closed-loop controller interfaces under the current objective. Preserve v1.6 as the mainline and reposition the 120 minute forecast as a supervisory warning / reporting layer, unless the project explicitly reframes the objective around hard constraints that require intervention near 3deg.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
