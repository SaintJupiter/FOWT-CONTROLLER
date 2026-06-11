#!/usr/bin/env python3
"""Build replay-derived PSC control-value labels.

This script converts retained closed-vs-deadband replay timeseries into sparse
row-level labels aligned with the h240/f120 PSC dataset. It upgrades the old
`closed_roundtrip_proxy` idea into actual replay-derived control-value targets:

- closed_pump_waste_m3: closed-only pump work in the next horizon.
- deadband_opportunity_m3: closed pump work minus deadband pump work.
- relax_event_exposure_risk_s: additional posture/fallback exposure under
  deadband relative to closed-only.

The labels are intentionally sparse: only windows covered by retained replay
episodes are labeled. The training script can use them as an auxiliary masked
loss without inventing labels for the rest of the corpus.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


SPLITS = ("train", "validation", "test")
VALUE_COLUMNS = [
    "closed_pump_waste_m3",
    "deadband_opportunity_m3",
    "relax_event_exposure_risk_s",
]


@dataclass(frozen=True)
class ReplayFamily:
    name: str
    timeseries_dir: Path
    primary_pattern: str


DEFAULT_FAMILIES = [
    ReplayFamily(
        name="d1_deadband_all80_6h",
        timeseries_dir=Path(
            "outputs/wind_prediction/regime_conditioned_policy_development_v1/"
            "dc_preserving_deadband_v1/p2like_all80_pid_deadband_1p5_6h/timeseries"
        ),
        primary_pattern="pid_deadband_1p5_6h_timeseries.csv",
    ),
    ReplayFamily(
        name="c3_deadband_24case_6h",
        timeseries_dir=Path(
            "outputs/wind_prediction/regime_conditioned_policy_development_v1/"
            "hard_24h_pump_gate_v1/dc_deadband_c3_24case_6h_validation_v1/timeseries"
        ),
        primary_pattern="dc_deadband_c3_24case_6h_validation_v1_timeseries.csv",
    ),
    ReplayFamily(
        name="w1_stable_broad20_d1_12h",
        timeseries_dir=Path(
            "outputs/wind_prediction/regime_conditioned_policy_development_v1/"
            "hard_24h_pump_gate_v1/w1_hi_attn_stable_broad20_d1_12h_v1/timeseries"
        ),
        primary_pattern="w1_hi_attn_stable_broad20_d1_12h_v1_timeseries.csv",
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("data/processed/wind_ml_10min/ballast_decision_fino1_meteo_aux_h240_f120_v1"),
    )
    parser.add_argument(
        "--base-label-dir",
        type=Path,
        default=Path("outputs/wind_prediction/psc_decision_model_v1/dataset"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/wind_prediction/psc_decision_model_v1/replay_value_labels_v1"),
    )
    parser.add_argument("--horizon-minutes", type=int, default=120)
    parser.add_argument("--stride-minutes", type=int, default=10)
    parser.add_argument("--posture-threshold-deg", type=float, default=5.0)
    parser.add_argument("--fallback-penalty-s", type=float, default=600.0)
    return parser.parse_args()


def timestamp_from_closed_name(path: Path) -> pd.Timestamp | None:
    match = re.search(r"_(\d{4}-\d{2}-\d{2})_(\d{6})_closed_only_timeseries\.csv$", path.name)
    if not match:
        return None
    return pd.Timestamp(f"{match.group(1)} {match.group(2)[:2]}:{match.group(2)[2:4]}:{match.group(2)[4:6]}")


def primary_for_closed(path: Path, primary_pattern: str) -> Path:
    return path.with_name(path.name.replace("closed_only_timeseries.csv", primary_pattern))


def pump_work_m3(df: pd.DataFrame, mask: np.ndarray) -> float:
    if not mask.any() or "pump_total_rate_m3_min" not in df:
        return 0.0
    t = df.loc[mask, "t_s"].to_numpy(dtype="float64")
    rate = np.abs(df.loc[mask, "pump_total_rate_m3_min"].to_numpy(dtype="float64"))
    if len(t) < 2:
        return 0.0
    dt_min = np.diff(t, prepend=t[0]) / 60.0
    if len(dt_min) > 1:
        dt_min[0] = np.median(dt_min[1:])
    return float(np.nansum(rate * np.maximum(dt_min, 0.0)))


def time_over_threshold_s(df: pd.DataFrame, mask: np.ndarray, threshold: float) -> float:
    if not mask.any() or "pitch_deg" not in df or "roll_deg" not in df:
        return 0.0
    t = df.loc[mask, "t_s"].to_numpy(dtype="float64")
    axis = np.maximum(
        np.abs(df.loc[mask, "pitch_deg"].to_numpy(dtype="float64")),
        np.abs(df.loc[mask, "roll_deg"].to_numpy(dtype="float64")),
    )
    if len(t) < 2:
        return 0.0
    dt = np.diff(t, prepend=t[0])
    if len(dt) > 1:
        dt[0] = np.median(dt[1:])
    return float(np.nansum(np.maximum(dt, 0.0) * (axis > threshold)))


def fallback_exposure_s(df: pd.DataFrame, mask: np.ndarray, fallback_penalty_s: float) -> float:
    if not mask.any() or "preview_primary_safety_fallback" not in df:
        return 0.0
    active = pd.to_numeric(df.loc[mask, "preview_primary_safety_fallback"], errors="coerce").fillna(0.0)
    return float((active > 0.5).any()) * fallback_penalty_s


def load_sample_index(dataset_dir: Path) -> tuple[pd.DataFrame, dict[str, pd.Series]]:
    sample_index = pd.read_csv(dataset_dir / "sample_index.csv.gz")
    sample_index["future_start"] = pd.to_datetime(sample_index["future_start"])
    split_maps = {
        split: pd.Series(
            np.arange(int((sample_index["split"] == split).sum()), dtype=np.int64),
            index=sample_index.loc[sample_index["split"] == split, "future_start"],
        )
        for split in SPLITS
    }
    return sample_index, split_maps


def build_family_rows(
    family: ReplayFamily,
    split_maps: dict[str, pd.Series],
    horizon_minutes: int,
    stride_minutes: int,
    posture_threshold_deg: float,
    fallback_penalty_s: float,
) -> list[dict]:
    rows: list[dict] = []
    if not family.timeseries_dir.exists():
        return rows
    horizon_s = horizon_minutes * 60.0
    stride_s = stride_minutes * 60.0
    for closed_path in sorted(family.timeseries_dir.glob("*_closed_only_timeseries.csv")):
        start_ts = timestamp_from_closed_name(closed_path)
        primary_path = primary_for_closed(closed_path, family.primary_pattern)
        if start_ts is None or not primary_path.exists():
            continue
        closed = pd.read_csv(closed_path)
        primary = pd.read_csv(primary_path)
        max_t = min(float(closed["t_s"].max()), float(primary["t_s"].max()))
        offsets = np.arange(0.0, max(0.0, max_t - horizon_s) + 1.0, stride_s)
        for offset_s in offsets:
            future_start = start_ts + pd.to_timedelta(offset_s, unit="s")
            split = None
            split_row = None
            for candidate_split, mapping in split_maps.items():
                if future_start in mapping.index:
                    split = candidate_split
                    split_row = int(mapping.loc[future_start])
                    break
            if split is None or split_row is None:
                continue
            closed_mask = (closed["t_s"] >= offset_s) & (closed["t_s"] < offset_s + horizon_s)
            primary_mask = (primary["t_s"] >= offset_s) & (primary["t_s"] < offset_s + horizon_s)
            closed_work = pump_work_m3(closed, closed_mask.to_numpy())
            primary_work = pump_work_m3(primary, primary_mask.to_numpy())
            closed_time5 = time_over_threshold_s(closed, closed_mask.to_numpy(), posture_threshold_deg)
            primary_time5 = time_over_threshold_s(primary, primary_mask.to_numpy(), posture_threshold_deg)
            exposure_risk = max(0.0, primary_time5 - closed_time5) + fallback_exposure_s(
                primary, primary_mask.to_numpy(), fallback_penalty_s
            )
            rows.append(
                {
                    "split": split,
                    "split_row": split_row,
                    "future_start": future_start,
                    "family": family.name,
                    "episode_start": start_ts,
                    "offset_s": float(offset_s),
                    "closed_pump_waste_m3": closed_work,
                    "deadband_opportunity_m3": max(0.0, closed_work - primary_work),
                    "relax_event_exposure_risk_s": exposure_risk,
                    "closed_time_over5_s": closed_time5,
                    "primary_time_over5_s": primary_time5,
                    "primary_pump_work_m3": primary_work,
                }
            )
    return rows


def write_split_arrays(rows: pd.DataFrame, base_label_dir: Path, output_dir: Path) -> dict:
    manifest = json.loads((base_label_dir / "psc_decision_dataset_manifest.json").read_text(encoding="utf-8"))
    summary: dict[str, dict] = {}
    for split in SPLITS:
        n = int(np.load(base_label_dir / f"y_regime_{split}.npy", mmap_mode="r").shape[0])
        y = np.full((n, len(VALUE_COLUMNS)), np.nan, dtype=np.float32)
        mask = np.zeros(n, dtype=bool)
        split_rows = rows[rows["split"] == split].copy()
        if len(split_rows):
            # Duplicate timestamps can occur across families; average labels for the same split row.
            grouped = split_rows.groupby("split_row", as_index=False)[VALUE_COLUMNS].mean()
            idx = grouped["split_row"].to_numpy(dtype=np.int64)
            y[idx, :] = grouped[VALUE_COLUMNS].to_numpy(dtype=np.float32)
            mask[idx] = True
        np.save(output_dir / f"y_replay_value_{split}.npy", y)
        np.save(output_dir / f"y_replay_value_mask_{split}.npy", mask)
        summary[split] = {
            "dataset_rows": n,
            "labeled_rows": int(mask.sum()),
            "label_share": float(mask.mean()),
        }
    manifest["replay_value_columns"] = VALUE_COLUMNS
    manifest["source_base_label_dir"] = str(base_label_dir)
    (output_dir / "replay_value_label_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _, split_maps = load_sample_index(args.dataset_dir)

    rows = []
    for family in DEFAULT_FAMILIES:
        rows.extend(
            build_family_rows(
                family=family,
                split_maps=split_maps,
                horizon_minutes=args.horizon_minutes,
                stride_minutes=args.stride_minutes,
                posture_threshold_deg=args.posture_threshold_deg,
                fallback_penalty_s=args.fallback_penalty_s,
            )
        )
    row_df = pd.DataFrame(rows)
    if row_df.empty:
        raise SystemExit("No replay value labels could be aligned to the PSC dataset.")
    row_df.to_csv(args.output_dir / "replay_value_rows.csv", index=False)
    summary = write_split_arrays(row_df, args.base_label_dir, args.output_dir)

    family_summary = (
        row_df.groupby(["family", "split"], as_index=False)
        .agg(
            rows=("split_row", "count"),
            closed_pump_waste_m3_mean=("closed_pump_waste_m3", "mean"),
            deadband_opportunity_m3_mean=("deadband_opportunity_m3", "mean"),
            relax_event_exposure_risk_s_mean=("relax_event_exposure_risk_s", "mean"),
        )
        .sort_values(["family", "split"])
    )
    family_summary.to_csv(args.output_dir / "replay_value_family_summary.csv", index=False)
    (args.output_dir / "replay_value_split_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"rows": len(row_df), "summary": summary}, indent=2))


if __name__ == "__main__":
    main()
