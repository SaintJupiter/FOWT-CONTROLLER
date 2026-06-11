#!/usr/bin/env python3
"""Mine favorable real-scene C3/W1 continuous casebooks.

This is a selection-only utility. It reads pre-action historical forecast/state
features and writes real replay casebooks; it does not inspect controller
outcomes or tune a controller against pump results.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


REPO = Path(__file__).resolve().parents[2]
BASE = REPO / "outputs/wind_prediction/regime_conditioned_policy_development_v1"
SRC = BASE / "continuous_episode_validation_v1/episode_feature_timeline.csv"
OUT = BASE / "c3_w1_real_scene_optimization_v1"
CASEBOOK_DIR = OUT / "casebooks"
DT_MINUTES = 10


@dataclass(frozen=True)
class Pick:
    family: str
    start: pd.Timestamp
    end: pd.Timestamp
    windows: int
    score: float
    reason: str
    padding_h: float = 3.0
    duration_h: float = 12.0

    @property
    def replay_start(self) -> pd.Timestamp:
        return self.start - pd.Timedelta(hours=self.padding_h)


def _num(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col not in df.columns:
        return pd.Series(default, index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce").fillna(default)


def _bool(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(False, index=df.index, dtype=bool)
    return df[col].astype(str).str.lower().isin({"1", "true", "yes"})


def _runs(df: pd.DataFrame, mask: pd.Series, flag: str) -> pd.DataFrame:
    rows = df[mask.fillna(False)].sort_values("future_start").copy()
    if rows.empty:
        return pd.DataFrame()
    gap = rows["future_start"].diff().dt.total_seconds().fillna(DT_MINUTES * 60)
    groups = (gap > DT_MINUTES * 60 * 1.5).cumsum()
    out = []
    for _, g in rows.groupby(groups):
        out.append(
            {
                "flag": flag,
                "start": g["future_start"].min(),
                "end": g["future_start"].max(),
                "windows": int(len(g)),
                "duration_h": float(len(g) * DT_MINUTES / 60.0),
                "max_speed": float(_num(g, "future_speed_max_ms").max()),
                "mean_speed": float(_num(g, "future_speed_mean_ms").mean()),
                "near_range_mean": float(_num(g, "near_range_ms").mean()),
                "dir_shift_max": float(_num(g, "dir_shift_abs_max_deg").max()),
                "ramp_frac": float(_num(g, "speed_ramp_ge_3ms").mean()),
                "event_frac": float(_num(g, "ballast_attention_event").mean()),
            }
        )
    return pd.DataFrame(out)


def _pick(runs: pd.DataFrame, family: str, reason: str, n: int, min_windows: int) -> list[Pick]:
    if runs.empty:
        return []
    r = runs[runs["windows"] >= int(min_windows)].copy()
    if r.empty:
        return []
    r["score"] = (
        r["windows"].astype(float)
        + 2.0 * r["near_range_mean"].fillna(0.0)
        + 0.05 * r["dir_shift_max"].fillna(0.0)
        + 0.25 * r["max_speed"].fillna(0.0)
    )
    r = r.sort_values(["windows", "score"], ascending=False)
    picks: list[Pick] = []
    for _, row in r.iterrows():
        start = pd.Timestamp(row["start"])
        if any(abs((start - p.start).total_seconds()) < 24 * 3600 for p in picks):
            continue
        picks.append(
            Pick(
                family=family,
                start=start,
                end=pd.Timestamp(row["end"]),
                windows=int(row["windows"]),
                score=float(row["score"]),
                reason=reason,
            )
        )
        if len(picks) >= int(n):
            break
    return picks


def _valid_start(ts: pd.Timestamp, available: list[pd.Timestamp]) -> pd.Timestamp:
    if ts in set(available):
        return ts
    later = [x for x in available if x >= ts]
    return later[0] if later else available[-1]


def _casebook(picks: list[Pick], available: list[pd.Timestamp], prefix: str) -> pd.DataFrame:
    rows = []
    for i, pick in enumerate(picks, start=1):
        replay_start = _valid_start(pick.replay_start, available)
        rows.append(
            {
                "case_id": f"{i:02d}_{prefix}_{pick.family}",
                "timestamp": replay_start.strftime("%Y-%m-%d %H:%M:%S"),
                "label": (
                    f"{pick.family} | core={pick.start:%Y-%m-%d %H:%M:%S}"
                    f"..{pick.end:%Y-%m-%d %H:%M:%S} | windows={pick.windows} "
                    f"| {pick.reason}"
                ),
            }
        )
    return pd.DataFrame(rows)


def _md_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_empty_"
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[c]) for c in cols) + " |")
    return "\n".join(lines)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    CASEBOOK_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(SRC)
    df["future_start"] = pd.to_datetime(df["future_start"])
    df = df.sort_values("future_start").reset_index(drop=True)

    speed_max = _num(df, "future_speed_max_ms")
    near_range = _num(df, "near_range_ms")
    dir_shift = _num(df, "dir_shift_abs_max_deg")
    ramp = _num(df, "speed_ramp_ge_3ms")
    event = _num(df, "ballast_attention_event")
    reintensify = _bool(df, "flag_reintensification")

    c3_mask = (
        (speed_max >= 10.0)
        & (near_range >= 1.0)
        & (dir_shift <= 25.0)
        & (ramp == 0.0)
        & (event == 0.0)
        & (~reintensify)
    )
    w1_mask = (
        (speed_max >= 8.0)
        & (dir_shift >= 30.0)
        & (dir_shift <= 50.0)
        & (ramp == 0.0)
        & (event == 0.0)
    )
    ordinary_mask = (
        (speed_max <= 12.0)
        & (near_range <= 0.8)
        & (dir_shift <= 15.0)
        & (ramp == 0.0)
        & (event == 0.0)
    )
    ramp_boundary_mask = (ramp == 1.0) | (event == 1.0)

    run_tables = [
        _runs(df, c3_mask, "c3_real_gusty_favorable"),
        _runs(df, w1_mask, "w1_real_direction_shift_favorable"),
        _runs(df, ordinary_mask, "ordinary_low_false_activation"),
        _runs(df, ramp_boundary_mask, "ramp_event_boundary"),
    ]
    runs = pd.concat([x for x in run_tables if not x.empty], ignore_index=True)
    runs.to_csv(OUT / "real_scene_runs_by_flag.csv", index=False)

    c3_picks = _pick(
        runs[runs["flag"].eq("c3_real_gusty_favorable")],
        "c3_real_gusty_favorable",
        "real historical high short-window wind-speed variation with stable direction",
        n=6,
        min_windows=12,
    )
    w1_picks = _pick(
        runs[runs["flag"].eq("w1_real_direction_shift_favorable")],
        "w1_real_direction_shift_favorable",
        "real historical direction-shift/reversal-like window without ramp/event onset",
        n=6,
        min_windows=6,
    )
    boundary_picks = []
    boundary_picks += _pick(
        runs[runs["flag"].eq("ordinary_low_false_activation")],
        "ordinary_low_false_activation",
        "real ordinary low-variation background where C3/W1 should stay quiet",
        n=3,
        min_windows=24,
    )
    boundary_picks += _pick(
        runs[runs["flag"].eq("ramp_event_boundary")],
        "ramp_event_boundary",
        "real ramp/event boundary where pump-saving holds should release or abstain",
        n=3,
        min_windows=6,
    )

    available = list(pd.to_datetime(df["future_start"]))
    c3_cb = _casebook(c3_picks, available, "c3")
    w1_cb = _casebook(w1_picks, available, "w1")
    boundary_cb = _casebook(boundary_picks, available, "boundary")
    combined_cb = pd.concat([c3_cb, w1_cb, boundary_cb], ignore_index=True)

    c3_cb.to_csv(CASEBOOK_DIR / "c3_real_favorable_top6_12h_cases.csv", index=False)
    w1_cb.to_csv(CASEBOOK_DIR / "w1_real_favorable_top6_12h_cases.csv", index=False)
    boundary_cb.to_csv(CASEBOOK_DIR / "c3_w1_real_boundary_12h_cases.csv", index=False)
    combined_cb.to_csv(CASEBOOK_DIR / "c3_w1_real_combined_12h_cases.csv", index=False)

    pick_rows = []
    for group, picks in (("c3", c3_picks), ("w1", w1_picks), ("boundary", boundary_picks)):
        for p in picks:
            pick_rows.append(
                {
                    "group": group,
                    "family": p.family,
                    "core_start": p.start,
                    "core_end": p.end,
                    "windows": p.windows,
                    "core_duration_h": round(p.windows * DT_MINUTES / 60.0, 3),
                    "score": round(p.score, 3),
                    "reason": p.reason,
                }
            )
    picks_df = pd.DataFrame(pick_rows)
    picks_df.to_csv(OUT / "real_scene_selected_cases.csv", index=False)

    summary = [
        "# C3/W1 Real-Scene Case Mining v1",
        "",
        "Selection uses only real historical replay timestamps and pre-action forecast/state features.",
        "No controller outcome, pump saving, fallback, or posture result is used for selection.",
        "",
        "## Casebooks",
        "",
        f"- C3 favorable: `{CASEBOOK_DIR / 'c3_real_favorable_top6_12h_cases.csv'}`",
        f"- W1 favorable: `{CASEBOOK_DIR / 'w1_real_favorable_top6_12h_cases.csv'}`",
        f"- Boundary/ordinary: `{CASEBOOK_DIR / 'c3_w1_real_boundary_12h_cases.csv'}`",
        f"- Combined: `{CASEBOOK_DIR / 'c3_w1_real_combined_12h_cases.csv'}`",
        "",
        "## Selected Scenes",
        "",
        _md_table(picks_df),
        "",
        "## Interpretation",
        "",
        "These are favorable real scenes, not synthetic scenarios and not natural-base-rate deployment tests. "
        "They are appropriate for demonstrating how much C3/W1 can help when the real environment matches "
        "their intended mechanism. Boundary scenes must be run alongside them to prove the strategy does not "
        "fire in ordinary, ramp, or event-like contexts.",
    ]
    (OUT / "real_scene_selection_summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
    print(f"wrote {OUT}")
    print(picks_df.to_string(index=False))


if __name__ == "__main__":
    main()
