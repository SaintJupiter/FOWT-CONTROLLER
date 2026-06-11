#!/usr/bin/env python3
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
import sys

import numpy as np
import pandas as pd


TIMESTAMP_FMT = "%Y-%m-%d %H:%M:%S"


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def summarize_flip(flip_df: pd.DataFrame) -> dict[str, object]:
    overall = flip_df[flip_df["block_name"] == "overall"].copy()
    overall = overall.sort_values("median_cosine_similarity", ascending=False)
    best = overall.iloc[0].to_dict()
    raw = overall[overall["flip_mode"] == "none"].iloc[0].to_dict()
    return {
        "raw_flip_mode": str(raw["flip_mode"]),
        "raw_median_cosine_similarity": float(raw["median_cosine_similarity"]),
        "best_flip_mode": str(best["flip_mode"]),
        "best_median_cosine_similarity": float(best["median_cosine_similarity"]),
        "improvement": float(best["median_cosine_similarity"] - raw["median_cosine_similarity"]),
    }


def pick_nonoverlap(df: pd.DataFrame, count: int, min_gap_s: int, chosen: list[datetime], predicate) -> list[pd.Series]:
    out: list[pd.Series] = []
    for _, row in df.iterrows():
        if not predicate(row):
            continue
        ts = datetime.strptime(str(row["prediction_timestamp"]), TIMESTAMP_FMT)
        if all(abs((ts - prev).total_seconds()) >= min_gap_s for prev in chosen):
            out.append(row)
            chosen.append(ts)
        if len(out) >= count:
            break
    return out


def pick_nonoverlap_relaxed(
    df: pd.DataFrame,
    count: int,
    min_gap_s: int,
    chosen: list[datetime],
    predicates,
) -> list[pd.Series]:
    out: list[pd.Series] = []
    local_chosen = list(chosen)
    for predicate in predicates:
        for _, row in df.iterrows():
            if any(str(r["prediction_timestamp"]) == str(row["prediction_timestamp"]) for r in out):
                continue
            if not predicate(row):
                continue
            ts = datetime.strptime(str(row["prediction_timestamp"]), TIMESTAMP_FMT)
            if all(abs((ts - prev).total_seconds()) >= min_gap_s for prev in local_chosen):
                out.append(row)
                local_chosen.append(ts)
            if len(out) >= count:
                chosen[:] = local_chosen
                return out
    chosen[:] = local_chosen
    return out


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root / "src"))

    from wind_prediction.replay_dataset import Fino1ReplayDataset

    a0_dir = repo_root / "outputs" / "wind_prediction" / "planner_a0_pressure_proxy_sanity"
    a1_dir = repo_root / "outputs" / "wind_prediction" / "planner_a1_dryrun"
    diagnostics_dir = a1_dir / "diagnostics"
    paper_ready_dir = a1_dir / "paper_ready"
    ensure_dir(a1_dir)
    ensure_dir(diagnostics_dir)
    ensure_dir(paper_ready_dir)

    block_df = pd.read_csv(a0_dir / "intermediate" / "a0_block_metrics.csv")
    direction_df = pd.read_csv(a0_dir / "diagnostics" / "a0_direction_consistency.csv")
    flip_df = pd.read_csv(a0_dir / "diagnostics" / "a0_coordinate_flip_check.csv")
    report_path = a0_dir / "A0_pressure_proxy_sanity_report.md"
    report_text = report_path.read_text(encoding="utf-8") if report_path.exists() else ""

    flip_summary = summarize_flip(flip_df)
    use_flipped = str(flip_summary["best_flip_mode"]) == "flip_both"

    # Planner-facing convention:
    # pressure_vec_for_planner is defined in the same compensation coordinate as action_vec and
    # virtual_ballast_comp_vec. Current A0 shows that the raw proxy is globally sign-reversed
    # relative to signed closed-only response, so the planner-facing vector uses -raw.
    sign_multiplier = -1.0 if use_flipped else 1.0
    direction_compare_rows = []
    reliable = direction_df[direction_df["reliable_signed_direction"] == 1].copy()
    for block_name, sub in reliable.groupby("block_name"):
        raw_med = float(np.nanmedian(sub["cosine_similarity"].to_numpy(dtype=float)))
        p_pitch = sign_multiplier * sub["pressure_pitch_component"].to_numpy(dtype=float)
        p_roll = sign_multiplier * sub["pressure_roll_component"].to_numpy(dtype=float)
        r_pitch = sub["pitch_signed_mean"].to_numpy(dtype=float)
        r_roll = sub["roll_signed_mean"].to_numpy(dtype=float)
        cos = (p_pitch * r_pitch + p_roll * r_roll) / (
            np.sqrt(p_pitch**2 + p_roll**2) * np.sqrt(r_pitch**2 + r_roll**2)
        )
        corrected_med = float(np.nanmedian(cos))
        direction_compare_rows.append(
            {
                "block_name": block_name,
                "raw_median_cosine_similarity": raw_med,
                "planner_sign_multiplier": sign_multiplier,
                "corrected_median_cosine_similarity": corrected_med,
            }
        )
    overall_raw = float(flip_summary["raw_median_cosine_similarity"])
    overall_corr = float(flip_summary["best_median_cosine_similarity"]) if use_flipped else overall_raw
    direction_compare_rows.append(
        {
            "block_name": "overall",
            "raw_median_cosine_similarity": overall_raw,
            "planner_sign_multiplier": sign_multiplier,
            "corrected_median_cosine_similarity": overall_corr,
        }
    )
    pd.DataFrame(direction_compare_rows).to_csv(
        diagnostics_dir / "a01_direction_consistency_before_after.csv",
        index=False,
    )

    planner_sign_note = {
        "pressure_vec_raw_definition": "raw pressure proxy from A0 using deadband-scaled ws^2 and wind direction projection",
        "pressure_vec_for_planner_definition": "pressure_vec_for_planner = - raw_pressure_vec",
        "planner_sign_multiplier": sign_multiplier,
        "rationale": "A0 coordinate flip check showed global near-perfect anti-alignment in signed direction; flipping both pitch and roll changes overall median cosine from negative to strongly positive.",
        "shared_coordinate_system": [
            "future_pressure_vec",
            "virtual_ballast_comp_vec",
            "residual_vec",
            "action_vec",
        ],
        "residual_definition": "residual_vec = future_pressure_vec - virtual_ballast_comp_vec",
    }
    (diagnostics_dir / "a01_pressure_vec_sign_convention.json").write_text(
        json.dumps(planner_sign_note, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # Build A1 window pool.
    # Use block1 pressure for near-term gating because A0 says 0-40 min is more trustworthy than block3.
    near = block_df[block_df["block_name"] == "block1_0_20m"].copy()
    near["pressure_mag_for_selection"] = near["pressure_mag_mean"]
    pressure_q75 = float(np.nanpercentile(near["pressure_mag_for_selection"], 75))
    pressure_q25 = float(np.nanpercentile(near["pressure_mag_for_selection"], 25))

    # Rebuild labels with current A0 findings:
    # - high_pressure/high_event: event positive and pressure high
    # - combined_variability: keep existing label, regardless of pure pressure
    # - low_pressure_normal: event negative and pressure low
    # - normal_but_high_pressure: event negative and pressure high
    near["is_event_high"] = pd.to_numeric(near["event_score"], errors="coerce").fillna(0.0) >= 0.5
    near["is_pressure_high"] = pd.to_numeric(near["pressure_mag_for_selection"], errors="coerce").fillna(0.0) >= pressure_q75
    near["is_pressure_low"] = pd.to_numeric(near["pressure_mag_for_selection"], errors="coerce").fillna(np.inf) <= pressure_q25

    replay = Fino1ReplayDataset(
        dataset_dir=repo_root / "data" / "processed" / "wind_ml_10min" / "ballast_decision_fino1_meteo_aux_v1",
        split="test",
    )
    min_gap_s = int(replay.update_interval_s * 6)
    chosen: list[datetime] = []

    high_sorted = near.sort_values(["pressure_mag_for_selection", "event_score"], ascending=False)
    high_pressure_high_event = pick_nonoverlap_relaxed(
        high_sorted,
        count=5,
        min_gap_s=min_gap_s,
        chosen=chosen,
        predicates=[
            lambda r: bool(r["is_event_high"]) and bool(r["is_pressure_high"]),
            lambda r: bool(r["is_event_high"]) and float(r["pressure_mag_for_selection"]) >= float(np.nanpercentile(near["pressure_mag_for_selection"], 60)),
            lambda r: bool(r["is_event_high"]),
        ],
    )
    combined_variability = pick_nonoverlap(
        near[near["risk_type"] == "combined_variability"].sort_values("combo_score", ascending=False),
        count=5,
        min_gap_s=min_gap_s,
        chosen=chosen,
        predicate=lambda r: True,
    )
    low_sorted = near.sort_values(["pressure_mag_for_selection", "event_score"], ascending=True)
    low_pressure_normal = pick_nonoverlap_relaxed(
        low_sorted,
        count=5,
        min_gap_s=min_gap_s,
        chosen=chosen,
        predicates=[
            lambda r: (not bool(r["is_event_high"])) and bool(r["is_pressure_low"]),
            lambda r: not bool(r["is_event_high"]),
        ],
    )
    normal_high_sorted = near.sort_values("pressure_mag_for_selection", ascending=False)
    normal_but_high_pressure = pick_nonoverlap_relaxed(
        normal_high_sorted,
        count=5,
        min_gap_s=min_gap_s,
        chosen=chosen,
        predicates=[
            lambda r: (not bool(r["is_event_high"])) and bool(r["is_pressure_high"]),
            lambda r: not bool(r["is_event_high"]),
        ],
    )

    rows = []
    for label, items in [
        ("high_pressure_high_event", high_pressure_high_event),
        ("combined_variability", combined_variability),
        ("low_pressure_normal", low_pressure_normal),
        ("normal_but_high_pressure", normal_but_high_pressure),
    ]:
        for rank, row in enumerate(items, start=1):
            rows.append(
                {
                    "a1_group": label,
                    "rank_in_group": rank,
                    "prediction_timestamp": row["prediction_timestamp"],
                    "source_risk_type": row["risk_type"],
                    "selection_strength": label,
                    "event_score": float(row["event_score"]),
                    "combo_score": float(row["combo_score"]),
                    "pressure_mag_block1": float(row["pressure_mag_for_selection"]),
                    "pitch_rms_block1": float(row["pitch_rms"]),
                    "roll_rms_block1": float(row["roll_rms"]),
                    "pose_exceed_time_block1": float(row["pose_exceed_time"]),
                }
            )
    window_df = pd.DataFrame(rows)
    window_df.to_csv(a1_dir / "window_selection_preview.csv", index=False)

    discount_cfg = {
        "default_discount_profile_name": "discounted_remote_v1",
        "default_discount_blocks": [1.00, 0.85, 0.70],
        "alternate_discount_profile_name": "discounted_remote_v2",
        "alternate_discount_blocks": [1.00, 0.85, 0.60],
        "no_discount_profile_name": "no_discount",
        "no_discount_blocks": [1.00, 1.00, 1.00],
        "note": "These are A1 dry-run diagnostic configurations only, not tuning conclusions.",
    }
    (diagnostics_dir / "a1_block_discount_config.json").write_text(
        json.dumps(discount_cfg, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    summary_lines = [
        "# A0.1 Sign Convention And A1 Input Prep",
        "",
        "## Pressure Sign Convention",
        "",
        f"- raw overall median cosine similarity: `{overall_raw:.3f}`",
        f"- corrected overall median cosine similarity: `{overall_corr:.3f}`",
        f"- chosen planner sign multiplier: `{sign_multiplier:+.0f}`",
        "- planner-facing definition: `pressure_vec_for_planner = - raw_pressure_vec`",
        "- semantic note: `future_pressure_vec`, `virtual_ballast_comp_vec`, `residual_vec`, and `action_vec` must all live in the same compensation coordinate system.",
        "- residual definition: `residual_vec = future_pressure_vec - virtual_ballast_comp_vec`",
        "",
        "## A1 Window Groups",
        "",
        "- `high_pressure_high_event`: future event-positive and near-block pressure high",
        "- `combined_variability`: keep variability-focused windows",
        "- `low_pressure_normal`: event-negative and near-block pressure low",
        "- `normal_but_high_pressure`: event-negative but near-block pressure high",
        "",
        f"- window count generated: `{len(window_df)}`",
        "- note: if strict quartile-based windows are insufficient, this preview falls back to the nearest available event/pressure-consistent windows and keeps the source label visible in `window_selection_preview.csv`.",
        "",
        "## A1 Discount Profiles",
        "",
        "- default: `[1.00, 0.85, 0.70]`",
        "- alternate: `[1.00, 0.85, 0.60]`",
        "- control: `[1.00, 1.00, 1.00]`",
        "",
        "## Recommendation",
        "",
        "- A0.1 conclusion: `can enter formal A1 dry-run`",
        "- caution 1: keep block3 discounted by default",
        "- caution 2: use planner-facing flipped sign convention, not the raw pressure sign",
        "- caution 3: do not treat `normal_low_risk` as equivalent to `low_pressure`",
    ]
    (a1_dir / "A0_1_sign_and_window_prep.md").write_text(
        "\n".join(summary_lines).strip() + "\n",
        encoding="utf-8",
    )

    print(f"Saved A0.1 outputs under {a1_dir}")


if __name__ == "__main__":
    main()
