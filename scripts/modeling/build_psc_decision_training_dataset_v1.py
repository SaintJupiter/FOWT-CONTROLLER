#!/usr/bin/env python3
"""Build a structured PSC decision-model training set.

The source wind dataset already has chronological train/validation/test splits
and future speed/direction targets. This script converts those future targets
into the regime/event/severity labels a predictive supervisory controller
selector needs. It does not copy the large X arrays; it stores split-local row
indices plus compact labels so the training script can memory-map the original
dataset.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


SPLITS = ("train", "validation", "test")

REGIME_CLASSES = [
    "stable",
    "transient_decay",
    "ramp_onset",
    "reversal_signflip",
    "sustained_high",
    "reintensification",
]

MULTILABEL_COLUMNS = [
    "event_reversal_signflip",
    "event_reintensification",
    "event_transient_decay",
    "event_sustained_high",
    "event_ramp_onset",
    "event_attention_any",
    "event_attention_0_20m",
    "event_attention_20_40m",
    "event_attention_40_60m",
    "event_attention_60_80m",
    "event_attention_80_100m",
    "event_attention_100_120m",
]

SCALAR_COLUMNS = [
    "time_to_attention_min",
    "event_strength_ms",
    "max_direction_shift_deg",
    "min_vector_cosine",
    "peak_to_late_drop_ms",
    "closed_roundtrip_proxy",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("data/processed/wind_ml_10min/ballast_decision_fino1_meteo_aux_h240_f120_v1"),
        help="Prepared h240/f120 ballast-decision wind dataset.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/wind_prediction/psc_decision_model_v1/dataset"),
        help="Directory for compact PSC decision labels.",
    )
    parser.add_argument("--chunk-size", type=int, default=50000)
    return parser.parse_args()


def wrap_angle_diff_deg(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Return absolute circular difference in degrees."""
    return np.abs((a - b + 180.0) % 360.0 - 180.0)


def unit_vectors(speed: np.ndarray, direction_deg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    theta = np.deg2rad(direction_deg)
    return speed * np.cos(theta), speed * np.sin(theta)


def derive_chunk_labels(
    speed_dir: np.ndarray,
    thresholds: dict,
    resolution_minutes: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    speed = speed_dir[:, :, 0].astype("float32")
    direction = speed_dir[:, :, 1].astype("float32")
    n, horizon = speed.shape

    first_speed = speed[:, 0]
    max_speed = np.nanmax(speed, axis=1)
    mean_speed = np.nanmean(speed, axis=1)
    late_mean = np.nanmean(speed[:, max(horizon - 3, 0) :], axis=1)
    early_max = np.nanmax(speed[:, : min(3, horizon)], axis=1)
    near_max = np.nanmax(speed[:, : min(6, horizon)], axis=1)
    far_max = np.nanmax(speed[:, min(6, horizon) :], axis=1) if horizon > 6 else near_max
    far_min = np.nanmin(speed[:, min(6, horizon) :], axis=1) if horizon > 6 else np.nanmin(speed, axis=1)

    ramp_from_first = max_speed - first_speed
    peak_to_late_drop = max_speed - late_mean
    far_reintensify = far_max - np.minimum(near_max, far_min)

    dir_shift = wrap_angle_diff_deg(direction, direction[:, [0]])
    max_dir_shift = np.nanmax(dir_shift, axis=1)

    u, v = unit_vectors(speed, direction)
    u0 = u[:, [0]]
    v0 = v[:, [0]]
    norm = np.sqrt(u * u + v * v) * np.sqrt(u0 * u0 + v0 * v0)
    cosine = np.divide(u * u0 + v * v0, np.maximum(norm, 1e-6))
    min_vector_cos = np.nanmin(cosine, axis=1)
    vector_change = np.sqrt((u - u0) ** 2 + (v - v0) ** 2)
    max_vector_change = np.nanmax(vector_change, axis=1)

    speed_ramp_threshold = float(thresholds.get("speed_ramp_ms", 3.0))
    direction_shift_threshold = float(thresholds.get("direction_shift_deg", 45.0))
    high_speed_threshold = float(thresholds.get("future_speed_train_p95_ms", 19.25))
    vector_change_threshold = float(thresholds.get("vector_change_train_p90_ms", 5.35))

    reversal = (max_dir_shift >= direction_shift_threshold) | (min_vector_cos <= 0.25)
    reintensification = (far_reintensify >= 2.5) & (far_max >= 0.85 * high_speed_threshold)
    transient_decay = (peak_to_late_drop >= 3.0) & (early_max >= 0.75 * max_speed) & (late_mean <= max_speed - 2.0)
    sustained_high = (max_speed >= high_speed_threshold) | (mean_speed >= 0.85 * high_speed_threshold)
    ramp_onset = (ramp_from_first >= speed_ramp_threshold) | (max_vector_change >= vector_change_threshold)

    regime = np.zeros(n, dtype=np.int64)
    regime[transient_decay] = REGIME_CLASSES.index("transient_decay")
    regime[ramp_onset] = REGIME_CLASSES.index("ramp_onset")
    regime[sustained_high] = REGIME_CLASSES.index("sustained_high")
    regime[reintensification] = REGIME_CLASSES.index("reintensification")
    regime[reversal] = REGIME_CLASSES.index("reversal_signflip")

    attention_step_mask = (
        (speed - first_speed[:, None] >= speed_ramp_threshold)
        | (dir_shift >= direction_shift_threshold)
        | (speed >= high_speed_threshold)
        | (vector_change >= vector_change_threshold)
    )
    has_attention = attention_step_mask.any(axis=1)
    first_attention_step = np.argmax(attention_step_mask, axis=1)
    time_to_attention_min = np.where(has_attention, (first_attention_step + 1) * resolution_minutes, horizon * resolution_minutes + resolution_minutes)

    bins = []
    for start in range(0, horizon, 2):
        stop = min(start + 2, horizon)
        bins.append(attention_step_mask[:, start:stop].any(axis=1))
    while len(bins) < 6:
        bins.append(np.zeros(n, dtype=bool))

    event_strength = np.maximum.reduce(
        [
            np.maximum(ramp_from_first, 0.0),
            max_vector_change,
            np.maximum(max_speed - high_speed_threshold, 0.0),
            max_dir_shift / 20.0,
        ]
    ).astype("float32")

    pressure = np.clip((speed / 12.0) ** 2, 0.0, 2.5)
    pressure_churn = np.nansum(np.abs(np.diff(pressure, axis=1)), axis=1)
    vector_churn = np.nansum(np.sqrt(np.diff(u, axis=1) ** 2 + np.diff(v, axis=1) ** 2), axis=1)
    closed_roundtrip_proxy = pressure_churn + 0.05 * vector_churn

    multilabel = np.stack(
        [
            reversal,
            reintensification,
            transient_decay,
            sustained_high,
            ramp_onset,
            has_attention,
            bins[0],
            bins[1],
            bins[2],
            bins[3],
            bins[4],
            bins[5],
        ],
        axis=1,
    ).astype("float32")

    scalars = np.stack(
        [
            time_to_attention_min.astype("float32"),
            event_strength,
            max_dir_shift.astype("float32"),
            min_vector_cos.astype("float32"),
            peak_to_late_drop.astype("float32"),
            closed_roundtrip_proxy.astype("float32"),
        ],
        axis=1,
    )

    diagnostics = pd.DataFrame(
        {
            "regime": [REGIME_CLASSES[i] for i in regime],
            "time_to_attention_min": time_to_attention_min,
            "event_strength_ms": event_strength,
            "max_direction_shift_deg": max_dir_shift,
            "min_vector_cosine": min_vector_cos,
            "peak_to_late_drop_ms": peak_to_late_drop,
            "closed_roundtrip_proxy": closed_roundtrip_proxy,
        }
    )
    return regime, multilabel, scalars.astype("float32"), diagnostics


def build_split(
    dataset_dir: Path,
    split: str,
    output_dir: Path,
    thresholds: dict,
    resolution_minutes: int,
    chunk_size: int,
) -> dict:
    speed_dir = np.load(dataset_dir / f"y_speed_dir_raw_{split}.npy", mmap_mode="r")
    n = int(speed_dir.shape[0])
    regime_out = np.empty(n, dtype=np.int64)
    multilabel_out = np.empty((n, len(MULTILABEL_COLUMNS)), dtype=np.float32)
    scalar_out = np.empty((n, len(SCALAR_COLUMNS)), dtype=np.float32)
    diag_frames: list[pd.DataFrame] = []

    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        regime, multilabel, scalars, diagnostics = derive_chunk_labels(
            np.asarray(speed_dir[start:end]),
            thresholds=thresholds,
            resolution_minutes=resolution_minutes,
        )
        regime_out[start:end] = regime
        multilabel_out[start:end] = multilabel
        scalar_out[start:end] = scalars
        diagnostics.insert(0, "split_row", np.arange(start, end))
        diag_frames.append(diagnostics)

    np.save(output_dir / f"row_indices_{split}.npy", np.arange(n, dtype=np.int64))
    np.save(output_dir / f"y_regime_{split}.npy", regime_out)
    np.save(output_dir / f"y_multilabel_{split}.npy", multilabel_out)
    np.save(output_dir / f"y_scalar_{split}.npy", scalar_out)

    diagnostics_df = pd.concat(diag_frames, ignore_index=True)
    diagnostics_df.to_csv(output_dir / f"label_diagnostics_{split}.csv", index=False)

    regime_counts = diagnostics_df["regime"].value_counts().reindex(REGIME_CLASSES, fill_value=0)
    return {
        "rows": n,
        "regime_counts": {k: int(v) for k, v in regime_counts.items()},
        "multilabel_rates": {
            name: float(multilabel_out[:, i].mean()) for i, name in enumerate(MULTILABEL_COLUMNS)
        },
        "scalar_mean": {name: float(np.nanmean(scalar_out[:, i])) for i, name in enumerate(SCALAR_COLUMNS)},
        "scalar_std": {name: float(np.nanstd(scalar_out[:, i])) for i, name in enumerate(SCALAR_COLUMNS)},
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    metadata = json.loads((args.dataset_dir / "metadata.json").read_text(encoding="utf-8"))
    thresholds = json.loads((args.dataset_dir / "event_thresholds.json").read_text(encoding="utf-8"))
    resolution_minutes = int(metadata.get("input_resolution_minutes", 10))

    split_summaries = {}
    for split in SPLITS:
        split_summaries[split] = build_split(
            dataset_dir=args.dataset_dir,
            split=split,
            output_dir=args.output_dir,
            thresholds=thresholds,
            resolution_minutes=resolution_minutes,
            chunk_size=args.chunk_size,
        )

    manifest = {
        "dataset_source": str(args.dataset_dir),
        "source_metadata": str(args.dataset_dir / "metadata.json"),
        "source_thresholds": thresholds,
        "regime_classes": REGIME_CLASSES,
        "multilabel_columns": MULTILABEL_COLUMNS,
        "scalar_columns": SCALAR_COLUMNS,
        "label_rule_priority": [
            "reversal_signflip overrides all safety/economy classes",
            "reintensification overrides sustained_high/ramp/stable",
            "sustained_high overrides ramp/stable unless reversal/reintensification",
            "ramp_onset overrides stable",
            "transient_decay is retained when no higher-risk safety class overrides it",
        ],
        "closed_roundtrip_proxy_note": (
            "Proxy from future pressure/vector churn, not a measured closed-only pump label. "
            "Replace with plant-derived closed-only roundtrip density when available."
        ),
        "split_summaries": split_summaries,
    }
    (args.output_dir / "psc_decision_dataset_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    rows = []
    for split, summary in split_summaries.items():
        for regime, count in summary["regime_counts"].items():
            rows.append({"split": split, "regime": regime, "count": count, "share": count / summary["rows"]})
    pd.DataFrame(rows).to_csv(args.output_dir / "regime_distribution.csv", index=False)

    print(f"Wrote PSC decision labels to {args.output_dir}")
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
