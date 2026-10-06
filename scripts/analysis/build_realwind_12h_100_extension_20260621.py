#!/usr/bin/env python3
"""Build a predeclared 100-window real-wind 12 h validation extension.

The script selects additional 12 h continuous windows from the existing 170
forecast-actionable 6 h window pool using only pre-12 h selection information:
6 h closed-loop pump burden, high-angle exposure guards, 12 h availability, and
minimum start-time separation. It does not inspect 12 h paired outcomes.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
SOURCE_170 = ROOT / (
    "outputs/wind_prediction/selector_positive_add40_6h_v1/"
    "combined_positive_summary_170case_6h/positive_combined_per_case.csv"
)
EXISTING_40 = ROOT / (
    "outputs/wind_prediction/positive_filtered_valid50_12h_opportunity_20260608/"
    "selection_audit_positive_filtered_valid50_12h.csv"
)
OUT = ROOT / "outputs/wind_prediction/realwind_12h_100_continuous_validation_20260621"
CASEBOOK_DIR = OUT / "casebooks"

END_GUARD = pd.Timestamp("2025-01-01 10:40:00")
MIN_CLOSED_PUMP_M3 = 500.0
MAX_CLOSED_T75_S = 30.0
MIN_START_SEPARATION_H = 12.0
N_EXISTING = 40
N_ADD = 60
BATCH_SIZE = 20


def _separated(t: pd.Timestamp, chosen: list[pd.Timestamp]) -> bool:
    min_gap_s = MIN_START_SEPARATION_H * 3600.0
    return all(abs((t - u).total_seconds()) >= min_gap_s for u in chosen)


def _casebook(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "case_id": df["new_case_id"],
            "timestamp": df["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S"),
            "label": df.apply(
                lambda r: (
                    "12h_realwind_continuous_validation"
                    f" | selector_stratum={r['selector_stratum']}"
                    f" | source_case_id={r['case_id']}"
                    f" | basis=closed_pump_ge500_t10zero_t75le30_endguard_12h_spacing"
                ),
                axis=1,
            ),
        }
    )


def _markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_empty_"
    cols = [str(c) for c in df.columns]
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[c]) for c in df.columns) + " |")
    return "\n".join(lines)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    CASEBOOK_DIR.mkdir(parents=True, exist_ok=True)

    source = pd.read_csv(SOURCE_170)
    source["timestamp"] = pd.to_datetime(source["timestamp"])

    existing = pd.read_csv(EXISTING_40).head(N_EXISTING).copy()
    existing["timestamp"] = pd.to_datetime(existing["timestamp"])
    existing_source_ids = set(existing["source_case_id"].astype(str))
    existing_times = list(existing["timestamp"])

    eligible = source.copy()
    eligible = eligible[~eligible["case_id"].astype(str).isin(existing_source_ids)]
    eligible = eligible[eligible["timestamp"] <= END_GUARD]
    eligible = eligible[eligible["closed_pump_m3"] >= MIN_CLOSED_PUMP_M3]
    eligible = eligible[eligible["closed_time_over_10_s"] == 0]
    eligible = eligible[eligible["closed_time_over_7p5_s"] <= MAX_CLOSED_T75_S]
    eligible = eligible.sort_values(
        ["closed_pump_m3", "timestamp"], ascending=[False, True]
    ).reset_index(drop=True)

    selected_rows = []
    chosen_times = list(existing_times)
    for _, row in eligible.iterrows():
        t = pd.Timestamp(row["timestamp"])
        if not _separated(t, chosen_times):
            continue
        selected_rows.append(row)
        chosen_times.append(t)
        if len(selected_rows) == N_ADD:
            break

    if len(selected_rows) < N_ADD:
        raise RuntimeError(f"Only selected {len(selected_rows)} additional windows")

    selected = pd.DataFrame(selected_rows).reset_index(drop=True)
    selected["extension_rank"] = range(1, len(selected) + 1)
    selected["new_case_id"] = [f"real12h_{N_EXISTING + i:03d}" for i in range(1, len(selected) + 1)]
    selected["planned_batch"] = [f"batch{i // BATCH_SIZE + 1:02d}" for i in range(len(selected))]

    existing_audit = existing.copy()
    existing_audit["set_role"] = "existing_completed_40"
    selected["set_role"] = "new_predeclared_add60"

    selected.to_csv(OUT / "selection_audit_add60.csv", index=False)
    eligible.to_csv(OUT / "eligible_remaining_pool.csv", index=False)

    all_declared = pd.concat(
        [
            pd.DataFrame(
                {
                    "set_role": "existing_completed_40",
                    "case_id": existing["case_id"],
                    "timestamp": existing["timestamp"],
                    "source_case_id": existing["source_case_id"],
                    "selector_stratum": existing["selector_stratum"],
                    "closed_pump_m3": existing["closed_pump_work_m3"],
                    "closed_time_over_7p5_s": existing["closed_time_over_7p5deg_s"],
                    "closed_time_over_10_s": existing["closed_time_over_10deg_s"],
                    "planned_batch": "existing40",
                }
            ),
            pd.DataFrame(
                {
                    "set_role": selected["set_role"],
                    "case_id": selected["new_case_id"],
                    "timestamp": selected["timestamp"],
                    "source_case_id": selected["case_id"],
                    "selector_stratum": selected["selector_stratum"],
                    "closed_pump_m3": selected["closed_pump_m3"],
                    "closed_time_over_7p5_s": selected["closed_time_over_7p5_s"],
                    "closed_time_over_10_s": selected["closed_time_over_10_s"],
                    "planned_batch": selected["planned_batch"],
                }
            ),
        ],
        ignore_index=True,
    )
    all_declared.to_csv(OUT / "selection_audit_declared100.csv", index=False)

    for batch_name, batch in selected.groupby("planned_batch", sort=True):
        _casebook(batch).to_csv(
            CASEBOOK_DIR / f"realwind_12h_add60_{batch_name}.csv", index=False
        )
    _casebook(selected).to_csv(CASEBOOK_DIR / "realwind_12h_add60_all.csv", index=False)

    summary = (
        all_declared.groupby(["set_role", "selector_stratum"], as_index=False)
        .agg(
            windows=("case_id", "count"),
            mean_closed_pump_m3=("closed_pump_m3", "mean"),
            min_closed_pump_m3=("closed_pump_m3", "min"),
            max_closed_pump_m3=("closed_pump_m3", "max"),
            max_closed_t75_s=("closed_time_over_7p5_s", "max"),
            max_closed_t10_s=("closed_time_over_10_s", "max"),
        )
        .round(2)
    )
    summary.to_csv(OUT / "selection_stratum_summary.csv", index=False)

    readout = [
        "# 12 h Real-Wind Continuous Validation 100-Window Extension",
        "",
        "## Selection Role",
        "",
        "This file predeclares a 100-window 12 h continuous validation set. The first 40 "
        "windows reuse the already completed paired 12 h runs. The additional 60 windows "
        "are selected from the existing 170 forecast-actionable 6 h pool before inspecting "
        "any 12 h paired outcomes.",
        "",
        "## Selection Rule for the Additional 60",
        "",
        f"- Source: `{SOURCE_170.relative_to(ROOT)}`.",
        f"- Exclude the first {N_EXISTING} completed 12 h source windows.",
        f"- 12 h availability end guard: timestamp <= {END_GUARD}.",
        f"- 6 h closed-loop cumulative pump volume >= {MIN_CLOSED_PUMP_M3:.0f} m3.",
        "- 6 h closed-loop dominant-attitude exposure at 10 deg equals 0 s.",
        f"- 6 h closed-loop dominant-attitude exposure at 7.5 deg <= {MAX_CLOSED_T75_S:.0f} s.",
        f"- Minimum start-time separation from all declared windows: {MIN_START_SEPARATION_H:.0f} h.",
        "- Rank remaining candidates by 6 h closed-loop cumulative pump volume.",
        "",
        "## Stratum Summary",
        "",
        _markdown_table(summary),
        "",
        "## Casebooks",
        "",
        f"- `{(CASEBOOK_DIR / 'realwind_12h_add60_batch01.csv').relative_to(ROOT)}`",
        f"- `{(CASEBOOK_DIR / 'realwind_12h_add60_batch02.csv').relative_to(ROOT)}`",
        f"- `{(CASEBOOK_DIR / 'realwind_12h_add60_batch03.csv').relative_to(ROOT)}`",
        "",
        "## Reporting Boundary",
        "",
        "This set supports a real-wind-sequence driven simulation claim for selected "
        "forecast-actionable continuous windows. It should not be described as a full "
        "real-sea-state or annual-frequency-weighted validation set.",
    ]
    (OUT / "selection_readout.md").write_text("\n".join(readout) + "\n", encoding="utf-8")

    print(f"wrote {OUT.relative_to(ROOT)}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
