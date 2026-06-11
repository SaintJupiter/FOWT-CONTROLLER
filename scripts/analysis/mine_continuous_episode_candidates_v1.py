#!/usr/bin/env python3
"""Mine continuous validation episodes for regime-conditioned pump saving.

This script is intentionally read-only with respect to controller behavior. It
does not tune thresholds or inspect pump-saving outcomes. It builds long,
continuous episode candidates from pre-action forecast/state features so the
controller can later be tested on ordinary -> regime -> ordinary transitions.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


REPO = Path(__file__).resolve().parents[2]
BASE = REPO / "outputs/wind_prediction/regime_conditioned_policy_development_v1"
OUT = BASE / "continuous_episode_validation_v1"
CASEBOOK_DIR = OUT / "casebooks"
DATASET_INDEX = REPO / (
    "data/processed/wind_ml_10min/"
    "ballast_decision_fino1_meteo_aux_v1/sample_index.csv.gz"
)
NEUTRAL_ROWS = BASE / "neutral_untyped_profile_v1/raw_tables/neutral_untyped_profile_rows.csv"
MINING_TEST = BASE / "regime_mining_v3/raw_tables/casebooks_test"
DT_MINUTES = 10


@dataclass(frozen=True)
class EpisodePick:
    layer: str
    family: str
    role: str
    core_start: pd.Timestamp
    core_end: pd.Timestamp
    core_windows: int
    score: float
    selection_reason: str
    padding_before_h: float = 3.0
    duration_h: float = 12.0

    @property
    def episode_start(self) -> pd.Timestamp:
        return self.core_start - pd.Timedelta(hours=float(self.padding_before_h))

    @property
    def duration_s(self) -> int:
        return int(round(float(self.duration_h) * 3600.0))


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _as_bool(s: pd.Series) -> pd.Series:
    if s.empty:
        return s.astype(bool)
    return s.astype(str).str.strip().str.lower().isin({"1", "true", "yes", "y"})


def _num(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col not in df.columns:
        return pd.Series(default, index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce").fillna(default)


def _flag_set(feature_file: str) -> set[pd.Timestamp]:
    path = MINING_TEST / feature_file
    df = _read_csv(path)
    if df.empty or "future_start" not in df.columns:
        return set()
    if "split" in df.columns:
        df = df[df["split"].astype(str).eq("test")].copy()
    return set(pd.to_datetime(df["future_start"]))


def _load_timeline() -> pd.DataFrame:
    idx = pd.read_csv(DATASET_INDEX)
    idx = idx[idx["split"].astype(str).eq("test")].copy()
    for col in ("history_start", "history_end", "future_start", "future_end"):
        idx[col] = pd.to_datetime(idx[col])

    keep = [
        "split",
        "series_id",
        "history_start",
        "history_end",
        "future_start",
        "future_end",
        "future_speed_ramp_max_ms",
        "future_dir_shift_abs_max_deg",
        "future_vector_change_max_ms",
        "speed_ramp_ge_3ms",
        "direction_shift_ge_45deg",
        "future_speed_ge_train_p95",
        "ballast_attention_event",
        "attention_event_0_20m",
        "attention_event_20_40m",
        "attention_event_40_60m",
        "attention_event_60_80m",
        "attention_event_80_100m",
        "attention_event_100_120m",
    ]
    timeline = idx[[c for c in keep if c in idx.columns]].copy()
    timeline = timeline.sort_values("future_start").drop_duplicates("future_start")

    neutral = _read_csv(NEUTRAL_ROWS)
    if not neutral.empty:
        neutral = neutral[neutral["split"].astype(str).eq("test")].copy()
        for col in ("history_start", "history_end", "future_start", "future_end"):
            neutral[col] = pd.to_datetime(neutral[col])
        neutral_cols = [
            "future_start",
            "primary_regime",
            "future_speed_first_ms",
            "future_speed_max_ms",
            "future_speed_mean_ms",
            "early_max_ms",
            "near_max_ms",
            "far_max_ms",
            "late_mean_ms",
            "early_rise_ms",
            "peak_to_late_drop_ms",
            "near_range_ms",
            "far_range_ms",
            "block_range_0_60_ms",
            "block_slope_0_60_ms",
            "block_slope_60_120_ms",
            "dir_shift_abs_max_deg",
            "near_dir_shift_abs_max_deg",
            "far_dir_shift_abs_max_deg",
            "vector_reversal_cosine",
            "flag_fast_decay",
            "flag_soft_decay",
            "flag_plateau",
            "flag_slow_decay",
            "flag_reversal",
            "flag_reintensification",
            "flag_lowrisk",
            "flag_far_only",
            "flag_gusty_oscillation",
            "speed_block_0_mean_ms",
            "speed_block_1_mean_ms",
            "speed_block_2_mean_ms",
            "speed_block_3_mean_ms",
            "speed_block_4_mean_ms",
            "speed_block_5_mean_ms",
            "neutral_subtype",
        ]
        neutral = neutral[[c for c in neutral_cols if c in neutral.columns]].copy()
        neutral = neutral.drop_duplicates("future_start")
        timeline = timeline.merge(neutral, on="future_start", how="left", suffixes=("", "_neutral"))

    ts = timeline["future_start"]
    timeline["flag_p1_decay"] = ts.isin(_flag_set("transient_peak_fast_decay_features.csv"))
    timeline["flag_c3_gusty"] = ts.isin(_flag_set("gusty_oscillation_candidate_features.csv"))
    timeline["flag_w1_reversal"] = ts.isin(_flag_set("direction_reversal_boundary_features.csv"))
    timeline["flag_reintensification_boundary"] = ts.isin(
        _flag_set("reintensification_boundary_features.csv")
    )
    timeline["flag_sustained_high_safety"] = ts.isin(
        _flag_set("sustained_high_safety_event_features.csv")
    )

    subtype = timeline.get("neutral_subtype", pd.Series("", index=timeline.index)).fillna("")
    p2_subtype = subtype.isin(
        ["neutral_moderate_high_steady", "neutral_mild_decay", "neutral_mixed_steady"]
    )
    timeline["flag_p2_clean_headroom"] = (
        p2_subtype
        & (_num(timeline, "early_max_ms") >= 11.0)
        & (_num(timeline, "near_max_ms") >= 11.0)
        & (_num(timeline, "near_range_ms") <= 2.8)
        & (_num(timeline, "far_range_ms") <= 3.0)
        & (_num(timeline, "dir_shift_abs_max_deg") <= 25.0)
        & (_num(timeline, "early_rise_ms") < 3.0)
        & (_num(timeline, "speed_ramp_ge_3ms") == 0.0)
        & (_num(timeline, "ballast_attention_event") == 0.0)
        & (~_as_bool(timeline.get("flag_reversal", pd.Series(False, index=timeline.index))))
        & (~_as_bool(timeline.get("flag_reintensification", pd.Series(False, index=timeline.index))))
    )

    timeline["flag_p2_ramp_or_onset"] = (
        subtype.eq("neutral_ramp_or_event_onset")
        | (_num(timeline, "early_rise_ms") >= 3.0)
        | (_num(timeline, "speed_ramp_ge_3ms") == 1.0)
        | (_num(timeline, "ballast_attention_event") == 1.0)
        | _as_bool(timeline.get("flag_reintensification", pd.Series(False, index=timeline.index)))
    )
    timeline["flag_direction_shift_boundary"] = (
        subtype.eq("neutral_direction_shift")
        | (_num(timeline, "dir_shift_abs_max_deg") > 30.0)
        | (_num(timeline, "direction_shift_ge_45deg") == 1.0)
    )
    timeline["flag_far_rise_watch"] = subtype.eq("neutral_far_rise_watch") | (
        (_num(timeline, "speed_block_5_mean_ms") - _num(timeline, "speed_block_2_mean_ms")) >= 2.0
    )
    timeline["flag_warning_any"] = (
        timeline["flag_p2_ramp_or_onset"]
        | timeline["flag_direction_shift_boundary"]
        | timeline["flag_w1_reversal"]
        | timeline["flag_reintensification_boundary"]
        | timeline["flag_sustained_high_safety"]
        | timeline["flag_far_rise_watch"]
    )
    timeline["flag_saving_candidate_any"] = (
        timeline["flag_p1_decay"] | timeline["flag_p2_clean_headroom"] | timeline["flag_c3_gusty"]
    )
    timeline["flag_ordinary_background"] = (
        ~timeline["flag_saving_candidate_any"]
        & ~timeline["flag_warning_any"]
        & (_num(timeline, "future_speed_max_ms") <= 12.0)
    )
    return timeline.sort_values("future_start").reset_index(drop=True)


def _runs_for_flag(timeline: pd.DataFrame, flag: str) -> pd.DataFrame:
    rows = timeline[timeline[flag].fillna(False)].copy()
    if rows.empty:
        return pd.DataFrame(
            columns=["flag", "start", "end", "windows", "duration_h", "mean_speed", "max_speed"]
        )
    rows = rows.sort_values("future_start")
    gap = rows["future_start"].diff().dt.total_seconds().fillna(DT_MINUTES * 60)
    group_id = (gap > DT_MINUTES * 60 * 1.5).cumsum()
    out = []
    for _, g in rows.groupby(group_id):
        out.append(
            {
                "flag": flag,
                "start": g["future_start"].min(),
                "end": g["future_start"].max(),
                "windows": int(len(g)),
                "duration_h": float(len(g) * DT_MINUTES / 60.0),
                "mean_speed": float(_num(g, "future_speed_mean_ms").mean()),
                "max_speed": float(_num(g, "future_speed_max_ms").max()),
                "near_range_mean": float(_num(g, "near_range_ms").mean()),
                "dir_shift_max": float(_num(g, "dir_shift_abs_max_deg").max()),
            }
        )
    return pd.DataFrame(out)


def _score_runs(runs: pd.DataFrame, speed_weight: float = 1.0) -> pd.DataFrame:
    if runs.empty:
        return runs
    runs = runs.copy()
    runs["score"] = runs["windows"].astype(float) + speed_weight * runs["max_speed"].fillna(0.0)
    return runs.sort_values(["windows", "score"], ascending=False)


def _pick_spaced(
    runs: pd.DataFrame,
    layer: str,
    family: str,
    role: str,
    reason: str,
    n: int,
    min_windows: int,
    min_spacing_h: float = 24.0,
) -> list[EpisodePick]:
    if runs.empty:
        return []
    candidates = _score_runs(runs[runs["windows"] >= int(min_windows)].copy())
    picks: list[EpisodePick] = []
    for _, row in candidates.iterrows():
        start = pd.Timestamp(row["start"])
        if any(abs((start - p.core_start).total_seconds()) < min_spacing_h * 3600.0 for p in picks):
            continue
        picks.append(
            EpisodePick(
                layer=layer,
                family=family,
                role=role,
                core_start=start,
                core_end=pd.Timestamp(row["end"]),
                core_windows=int(row["windows"]),
                score=float(row.get("score", row["windows"])),
                selection_reason=reason,
            )
        )
        if len(picks) >= int(n):
            break
    return picks


def _valid_episode_start(ts: pd.Timestamp, available: set[pd.Timestamp]) -> pd.Timestamp:
    if ts in available:
        return ts
    later = sorted(t for t in available if t >= ts)
    if later:
        return later[0]
    return max(available)


def _to_episode_frame(picks: list[EpisodePick], available: set[pd.Timestamp]) -> pd.DataFrame:
    rows = []
    for i, pick in enumerate(picks, start=1):
        start = _valid_episode_start(pick.episode_start, available)
        rows.append(
            {
                "episode_id": f"{pick.layer}_{pick.family}_{i:02d}",
                "layer": pick.layer,
                "family": pick.family,
                "role": pick.role,
                "timestamp": start.strftime("%Y-%m-%d %H:%M:%S"),
                "core_start": pick.core_start.strftime("%Y-%m-%d %H:%M:%S"),
                "core_end": pick.core_end.strftime("%Y-%m-%d %H:%M:%S"),
                "duration_s": pick.duration_s,
                "duration_h": pick.duration_h,
                "core_windows": pick.core_windows,
                "core_duration_h": round(pick.core_windows * DT_MINUTES / 60.0, 3),
                "padding_before_h": pick.padding_before_h,
                "score": round(pick.score, 6),
                "selection_reason": pick.selection_reason,
            }
        )
    return pd.DataFrame(rows)


def _write_casebook(path: Path, episodes: pd.DataFrame) -> None:
    if episodes.empty:
        pd.DataFrame(columns=["case_id", "timestamp", "label"]).to_csv(path, index=False)
        return
    cb = pd.DataFrame(
        {
            "case_id": episodes["episode_id"],
            "timestamp": episodes["timestamp"],
            "label": episodes.apply(
                lambda r: (
                    f"{r['layer']} | {r['family']} | core={r['core_start']}..{r['core_end']} "
                    f"| {r['selection_reason']}"
                ),
                axis=1,
            ),
        }
    )
    cb.to_csv(path, index=False)


def _markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_empty_"
    cols = [str(c) for c in df.columns]
    rows = []
    rows.append("| " + " | ".join(cols) + " |")
    rows.append("| " + " | ".join(["---"] * len(cols)) + " |")
    for _, row in df.iterrows():
        vals = [str(row[c]) for c in df.columns]
        rows.append("| " + " | ".join(vals) + " |")
    return "\n".join(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    CASEBOOK_DIR.mkdir(parents=True, exist_ok=True)

    timeline = _load_timeline()
    timeline.to_csv(OUT / "episode_feature_timeline.csv", index=False)

    flags = [
        "flag_p2_clean_headroom",
        "flag_p2_ramp_or_onset",
        "flag_direction_shift_boundary",
        "flag_far_rise_watch",
        "flag_p1_decay",
        "flag_c3_gusty",
        "flag_w1_reversal",
        "flag_reintensification_boundary",
        "flag_sustained_high_safety",
        "flag_ordinary_background",
    ]
    run_tables = []
    for flag in flags:
        run_tables.append(_runs_for_flag(timeline, flag))
    all_runs = pd.concat(run_tables, ignore_index=True)
    all_runs.to_csv(OUT / "episode_runs_by_flag.csv", index=False)

    summary_rows = []
    total_windows = max(int(len(timeline)), 1)
    for flag in flags:
        n = int(timeline[flag].fillna(False).sum())
        runs = all_runs[all_runs["flag"].eq(flag)]
        summary_rows.append(
            {
                "flag": flag,
                "windows": n,
                "all_window_pct": round(100.0 * n / total_windows, 3),
                "runs": int(len(runs)),
                "runs_ge_30min": int((runs["windows"] >= 3).sum()) if not runs.empty else 0,
                "runs_ge_60min": int((runs["windows"] >= 6).sum()) if not runs.empty else 0,
                "max_run_h": round(float(runs["duration_h"].max()), 3) if not runs.empty else 0.0,
            }
        )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUT / "episode_flag_prevalence_summary.csv", index=False)

    picks: list[EpisodePick] = []
    picks += _pick_spaced(
        all_runs[all_runs["flag"].eq("flag_p2_clean_headroom")],
        "B",
        "p2_clean_headroom_transition",
        "pump_saving_entry_exit",
        "pre-action P2 clean/headroom run with continuous padding",
        n=4,
        min_windows=6,
    )
    picks += _pick_spaced(
        all_runs[all_runs["flag"].eq("flag_p2_ramp_or_onset")],
        "C",
        "p2_ramp_or_event_boundary",
        "warning_override_boundary",
        "ramp/event-onset lookalike where P2 should close or abstain",
        n=4,
        min_windows=3,
    )
    picks += _pick_spaced(
        all_runs[all_runs["flag"].eq("flag_direction_shift_boundary")],
        "C",
        "direction_shift_boundary",
        "warning_override_boundary",
        "direction shift/reversal-like boundary for economy saving",
        n=2,
        min_windows=3,
    )
    picks += _pick_spaced(
        all_runs[all_runs["flag"].eq("flag_p1_decay")],
        "B",
        "p1_future_relief_decay_transition",
        "pump_saving_entry_exit",
        "future relief/decay run with continuous padding",
        n=2,
        min_windows=3,
    )
    picks += _pick_spaced(
        all_runs[all_runs["flag"].eq("flag_c3_gusty")],
        "B",
        "c3_gusty_transition_candidate",
        "candidate_entry_exit",
        "gusty/chatter candidate run, currently Pareto/future-work",
        n=2,
        min_windows=3,
    )
    picks += _pick_spaced(
        all_runs[all_runs["flag"].eq("flag_w1_reversal")],
        "C",
        "w1_reversal_warning",
        "warning_abstention",
        "direction reversal warning run",
        n=2,
        min_windows=3,
    )
    picks += _pick_spaced(
        all_runs[all_runs["flag"].eq("flag_ordinary_background")],
        "A",
        "ordinary_background",
        "false_activation_cost",
        "ordinary background run at natural base rate",
        n=3,
        min_windows=12,
    )

    available = set(pd.to_datetime(timeline["future_start"]))
    episodes = _to_episode_frame(picks, available)
    episodes.to_csv(OUT / "episode_candidates.csv", index=False)

    all_smoke = episodes.groupby(["layer", "family"], as_index=False).head(1).copy()
    all_smoke = all_smoke.sort_values(["layer", "family", "timestamp"]).reset_index(drop=True)
    p2_smoke = all_smoke[
        all_smoke["family"].isin(
            ["p2_clean_headroom_transition", "p2_ramp_or_event_boundary", "ordinary_background"]
        )
    ].copy()

    _write_casebook(CASEBOOK_DIR / "continuous_episode_smoke_12h_cases.csv", all_smoke)
    _write_casebook(CASEBOOK_DIR / "p2_continuous_smoke_12h_cases.csv", p2_smoke)

    decision = [
        "# Continuous Episode Mining v1",
        "",
        "## Purpose",
        "",
        "Mine continuous 12h validation episodes from pre-action forecast/state features, "
        "so later runs can test autonomous entry, exit, false activation, and warning override. "
        "This step does not inspect pump-saving outcomes and does not tune controller thresholds.",
        "",
        "## Method",
        "",
        "- A natural/background layer: ordinary continuous windows for false-activation cost.",
        "- B transition-rich layer: saving-regime runs with 3h leading context and continuous exit context.",
        "- C boundary/override layer: ramp, direction-shift, reversal, or future-risk windows where saving should close or abstain.",
        "- Episode labels are sampling labels only. Paper-grade `should open` labels must still come from counterfactual closed-loop outcomes, not from these forecast-shape rules.",
        "",
        "## Flag Prevalence",
        "",
        _markdown_table(summary),
        "",
        "## Smoke Casebooks",
        "",
        f"- All-family 12h smoke: `{CASEBOOK_DIR / 'continuous_episode_smoke_12h_cases.csv'}`",
        f"- P2-only 12h smoke: `{CASEBOOK_DIR / 'p2_continuous_smoke_12h_cases.csv'}`",
        "",
        "## Recommended Immediate Smoke",
        "",
        "Run the P2-only 12h smoke first with `forecast_advised_economy_guarded_stale_v1`. "
        "It is intentionally small: one clean transition, one ramp/event boundary, and one ordinary background segment.",
        "",
        "```bash",
        ".venv312/bin/python scripts/analysis/run_prediction_primary_casebook.py \\",
        f"  --cases-csv {CASEBOOK_DIR / 'p2_continuous_smoke_12h_cases.csv'} \\",
        f"  --out-dir {OUT / 'p2_guarded_stale_continuous_smoke_12h'} \\",
        "  --primary-label p2_guarded_stale_continuous_smoke \\",
        "  --primary-control-profile forecast_advised_economy_guarded_stale_v1 \\",
        "  --forecast-source learned --duration-s 43200 --skip-figures",
        "```",
        "",
        "## Current Interpretation",
        "",
        "This miner moves validation from isolated 1h/2h regime windows toward realistic continuous episodes. "
        "It does not yet prove automatic deployment; it creates the episode set needed to test it.",
    ]
    (OUT / "decision.md").write_text("\n".join(decision) + "\n", encoding="utf-8")

    print(f"wrote {OUT}")
    print(summary.to_string(index=False))
    print(f"smoke rows: all={len(all_smoke)} p2={len(p2_smoke)}")


if __name__ == "__main__":
    main()
