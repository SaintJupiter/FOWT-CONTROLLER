#!/usr/bin/env python3
"""Prepare a FINO1 single-station 10-minute wind dataset from BSH CSV exports.

This script builds two reusable layers:

1. canonical observations for one fused FINO1 wind series;
2. a supervised-learning table aligned with the existing repo conventions.

The fused series is intentionally conservative:

- wind speed uses 102 m cup measurements, preferring mast-corrected values
  when they pass QC and falling back to the raw 102 m cup signal otherwise;
- wind direction uses the 91 m vane signal when it passes QC and falls back
  to the 82 m ultrasonic direction only to reduce long gaps;
- auxiliary atmospheric variables are retained for future experiments but are
  not forced into the current base feature set.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


SOURCE_NAME = "BSH_FINO1"
SITE_NAME = "FINO1_Platform"
SENSOR_NAME = "fused_102m_speed_91m_dir_v1"
SERIES_ID = f"{SOURCE_NAME}::{SITE_NAME}::{SENSOR_NAME}"

GOOD_QC_FLAGS = {"2", "3", "4"}
BAD_QC_FLAGS = {"1", "5", "6", "7", "8", "9", "0"}

SPEED_BINS_MS = [-np.inf, 3, 6, 9, 12, 15, 20, 25, np.inf]
SPEED_BIN_LABELS = list(range(len(SPEED_BINS_MS) - 1))
DIR_SECTORS = 16
HORIZONS_STEPS = {"10m": 1, "30m": 3, "60m": 6}
ROLLING_WINDOWS = {"30m": "30min", "60m": "60min", "120m": "120min"}


def circular_diff_deg(target: pd.Series | np.ndarray, base: pd.Series | np.ndarray) -> np.ndarray:
    return (np.asarray(target, dtype="float64") - np.asarray(base, dtype="float64") + 180.0) % 360.0 - 180.0


def normalize_direction_deg(values: pd.Series) -> pd.Series:
    direction = pd.to_numeric(values, errors="coerce") % 360.0
    direction = direction.mask(direction < 0.0, direction + 360.0)
    return direction


def meteorological_uv(speed_ms: pd.Series, direction_deg: pd.Series) -> tuple[pd.Series, pd.Series]:
    radians = np.deg2rad(direction_deg.astype("float64"))
    u = -speed_ms.astype("float64") * np.sin(radians)
    v = -speed_ms.astype("float64") * np.cos(radians)
    return u, v


def parse_bsh_csv(path: Path) -> pd.DataFrame:
    with path.open("r", encoding="utf-8") as handle:
        lines = handle.readlines()
    header_idx = next(i for i, line in enumerate(lines) if line.startswith("#Time;"))
    columns = [col.strip() for col in lines[header_idx][1:].strip().split(";")]
    df = pd.read_csv(
        path,
        sep=";",
        names=columns,
        skiprows=header_idx + 1,
        header=None,
        dtype=str,
        low_memory=False,
    )
    for column in df.columns:
        df[column] = df[column].astype(str).str.strip()
        df[column] = df[column].replace({"nan": np.nan, "_": np.nan, "-": np.nan, "": np.nan})
    df["Time"] = pd.to_datetime(df["Time"], errors="coerce")
    return df


def numeric_column(df: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(df[column].str.replace(",", ".", regex=False), errors="coerce")


def qc_mask(df: pd.DataFrame, column: str) -> pd.Series:
    return df[column].fillna("").astype(str).str.strip().isin(GOOD_QC_FLAGS)


def load_fino1_wind(paths: list[Path]) -> pd.DataFrame:
    frames = []
    for path in paths:
        df = parse_bsh_csv(path)

        speed_raw = numeric_column(df, "WSPD_CUP.Cup Anemometer.102.0 [m/s]").where(
            qc_mask(df, "WSPD_CUP_FOR.Cup Anemometer.102.0 [WG_cup_FOR]")
        )
        speed_mc = numeric_column(df, "WSPD_CUP_MC.Cup Anemometer.102.0 [m/s]").where(
            qc_mask(df, "WSPD_CUP_MC_FOR.Cup Anemometer.102.0 [WG_cup_mc_FOR]")
        )
        dir_vane = numeric_column(df, "WDIR_VANE_315deg.Wind Vane.91.0 [deg]").where(
            qc_mask(df, "WDIR_VANE_315deg_FOR.Wind Vane.91.0 [WR_vane_315deg_FOR]")
        )
        dir_usa = numeric_column(df, "WDIR_USA_311deg.Ultrasonic Anemometer.82.0 [deg]").where(
            qc_mask(df, "WDIR_USA_311deg_FOR.Ultrasonic Anemometer.82.0 [WR_usa_311deg_FOR]")
        )

        fused_speed = speed_mc.combine_first(speed_raw)
        fused_dir = dir_vane.combine_first(dir_usa)

        speed_source = np.select(
            [speed_mc.notna(), speed_raw.notna()],
            ["cup_102m_mast_corrected", "cup_102m_raw"],
            default="missing",
        )
        dir_source = np.select(
            [dir_vane.notna(), dir_usa.notna()],
            ["vane_91m", "usa_82m"],
            default="missing",
        )

        out = pd.DataFrame(
            {
                "timestamp": df["Time"],
                "series_id": SERIES_ID,
                "source": SOURCE_NAME,
                "site": SITE_NAME,
                "sensor": SENSOR_NAME,
                "station_id": "6201065",
                "height_m": 102.0,
                "wind_speed_height_m": 102.0,
                "wind_dir_height_m": np.where(dir_vane.notna(), 91.0, np.where(dir_usa.notna(), 82.0, np.nan)),
                "wind_speed_ms": fused_speed,
                "wind_dir_deg": fused_dir,
                "wind_speed_ms_raw_102m": speed_raw,
                "wind_speed_ms_mc_102m": speed_mc,
                "wind_dir_deg_vane_91m": dir_vane,
                "wind_dir_deg_usa_82m": dir_usa,
                "wind_speed_source": speed_source,
                "wind_dir_source": dir_source,
                "wind_speed_qc_raw": df["WSPD_CUP_FOR.Cup Anemometer.102.0 [WG_cup_FOR]"],
                "wind_speed_qc_mc": df["WSPD_CUP_MC_FOR.Cup Anemometer.102.0 [WG_cup_mc_FOR]"],
                "wind_dir_qc_vane": df["WDIR_VANE_315deg_FOR.Wind Vane.91.0 [WR_vane_315deg_FOR]"],
                "wind_dir_qc_usa": df["WDIR_USA_311deg_FOR.Ultrasonic Anemometer.82.0 [WR_usa_311deg_FOR]"],
                "raw_file": path.name,
            }
        )
        frames.append(out)

    merged = pd.concat(frames, ignore_index=True).sort_values("timestamp")
    merged = merged.drop_duplicates(["series_id", "timestamp"], keep="last").reset_index(drop=True)
    return merged


def load_fino1_aux(path: Path) -> pd.DataFrame:
    df = parse_bsh_csv(path)
    temperature = numeric_column(df, "DRYT.Thermometer.101.0 [degC]").where(
        qc_mask(df, "DRYT_FOR.Thermometer.101.0 [LT_FOR]")
    )
    pressure = numeric_column(df, "ATMP.Barometer.92.0 [hPa]").where(
        qc_mask(df, "ATMP_FOR.Barometer.92.0 [LD_FOR]")
    )
    humidity = numeric_column(df, "RELH.Hygrometer.101.0 [%]").where(
        qc_mask(df, "RELH_FOR.Hygrometer.101.0 [FL_FOR]")
    )
    return pd.DataFrame(
        {
            "timestamp": df["Time"],
            "air_temp_c": temperature,
            "air_pressure_hpa": pressure,
            "rel_humidity_pct": humidity,
            "air_temp_qc": df["DRYT_FOR.Thermometer.101.0 [LT_FOR]"],
            "air_pressure_qc": df["ATMP_FOR.Barometer.92.0 [LD_FOR]"],
            "rel_humidity_qc": df["RELH_FOR.Hygrometer.101.0 [FL_FOR]"],
            "aux_raw_file": path.name,
        }
    )


def clean_records(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["wind_speed_ms"] = pd.to_numeric(df["wind_speed_ms"], errors="coerce")
    df["wind_dir_deg"] = normalize_direction_deg(df["wind_dir_deg"])

    keep = (
        df["timestamp"].notna()
        & df["wind_speed_ms"].between(0, 80, inclusive="both")
        & df["wind_dir_deg"].between(0, 360, inclusive="left")
    )
    df = df.loc[keep].copy()
    df = df.drop_duplicates(["series_id", "timestamp"]).sort_values(["series_id", "timestamp"])
    return df.reset_index(drop=True)


def add_model_features(df: pd.DataFrame) -> pd.DataFrame:
    df = clean_records(df)
    radians = np.deg2rad(df["wind_dir_deg"].astype("float64"))
    df["wind_dir_sin"] = np.sin(radians)
    df["wind_dir_cos"] = np.cos(radians)
    df["wind_u_ms"], df["wind_v_ms"] = meteorological_uv(df["wind_speed_ms"], df["wind_dir_deg"])
    df["speed_bin_3ms"] = pd.cut(
        df["wind_speed_ms"],
        bins=SPEED_BINS_MS,
        labels=SPEED_BIN_LABELS,
        right=False,
    ).astype("Int64")
    df["dir_sector_16"] = np.floor(((df["wind_dir_deg"] + 360.0 / DIR_SECTORS / 2) % 360.0) / (360.0 / DIR_SECTORS))
    df["dir_sector_16"] = df["dir_sector_16"].astype("Int64")

    group = df.groupby("series_id", group_keys=False, sort=False)
    prev_ts = group["timestamp"].shift(1)
    prev_is_exact_10m = (df["timestamp"] - prev_ts).dt.total_seconds().eq(600)
    df["wind_speed_delta_10m"] = group["wind_speed_ms"].diff().where(prev_is_exact_10m)
    df["wind_dir_delta_10m_deg"] = group["wind_dir_deg"].transform(
        lambda s: pd.Series(circular_diff_deg(s, s.shift(1)), index=s.index)
    ).where(prev_is_exact_10m)
    df["wind_u_delta_10m"] = group["wind_u_ms"].diff().where(prev_is_exact_10m)
    df["wind_v_delta_10m"] = group["wind_v_ms"].diff().where(prev_is_exact_10m)

    def add_time_window_features(series: pd.DataFrame) -> pd.DataFrame:
        series = series.sort_values("timestamp").copy()
        indexed = series.set_index("timestamp")
        for label, window in ROLLING_WINDOWS.items():
            min_periods = 2 if label == "30m" else 3
            series[f"wind_speed_mean_{label}"] = indexed["wind_speed_ms"].rolling(window, min_periods=min_periods).mean().values
            series[f"wind_speed_std_{label}"] = indexed["wind_speed_ms"].rolling(window, min_periods=min_periods).std().values
            series[f"wind_u_mean_{label}"] = indexed["wind_u_ms"].rolling(window, min_periods=min_periods).mean().values
            series[f"wind_v_mean_{label}"] = indexed["wind_v_ms"].rolling(window, min_periods=min_periods).mean().values
            series[f"wind_dir_sin_mean_{label}"] = indexed["wind_dir_sin"].rolling(window, min_periods=min_periods).mean().values
            series[f"wind_dir_cos_mean_{label}"] = indexed["wind_dir_cos"].rolling(window, min_periods=min_periods).mean().values
        return series

    parts = [add_time_window_features(series) for _series_id, series in group]
    df = pd.concat(parts, ignore_index=True) if parts else df.iloc[0:0].copy()
    group = df.groupby("series_id", group_keys=False, sort=False)

    hour = df["timestamp"].dt.hour + df["timestamp"].dt.minute / 60.0
    day_of_year = df["timestamp"].dt.dayofyear.astype("float64")
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
    df["dayofyear_sin"] = np.sin(2 * np.pi * day_of_year / 366.0)
    df["dayofyear_cos"] = np.cos(2 * np.pi * day_of_year / 366.0)

    pos = group.cumcount()
    n = group["timestamp"].transform("size").clip(lower=2)
    ratio = pos / (n - 1)
    df["split"] = np.select([ratio < 0.70, ratio < 0.85], ["train", "validation"], default="test")

    target_base = df[
        [
            "series_id",
            "timestamp",
            "wind_speed_ms",
            "wind_dir_deg",
            "wind_dir_sin",
            "wind_dir_cos",
            "wind_u_ms",
            "wind_v_ms",
            "speed_bin_3ms",
            "dir_sector_16",
        ]
    ].copy()
    for label, steps in HORIZONS_STEPS.items():
        horizon = pd.Timedelta(minutes=steps * 10)
        future = target_base.copy()
        future["timestamp"] = future["timestamp"] - horizon
        future = future.rename(
            columns={
                "wind_speed_ms": f"target_wind_speed_ms_tplus_{label}",
                "wind_dir_deg": f"target_wind_dir_deg_tplus_{label}",
                "wind_dir_sin": f"target_wind_dir_sin_tplus_{label}",
                "wind_dir_cos": f"target_wind_dir_cos_tplus_{label}",
                "wind_u_ms": f"target_wind_u_ms_tplus_{label}",
                "wind_v_ms": f"target_wind_v_ms_tplus_{label}",
                "speed_bin_3ms": f"target_speed_bin_3ms_tplus_{label}",
                "dir_sector_16": f"target_dir_sector_16_tplus_{label}",
            }
        )
        df = df.merge(future, on=["series_id", "timestamp"], how="left", sort=False)
        df[f"target_speed_delta_ms_tplus_{label}"] = df[f"target_wind_speed_ms_tplus_{label}"] - df["wind_speed_ms"]
        df[f"target_dir_delta_deg_tplus_{label}"] = circular_diff_deg(
            df[f"target_wind_dir_deg_tplus_{label}"], df["wind_dir_deg"]
        )
    return df


def build_transition_table(df: pd.DataFrame, horizon_label: str) -> pd.DataFrame:
    next_speed = f"target_speed_bin_3ms_tplus_{horizon_label}"
    next_dir = f"target_dir_sector_16_tplus_{horizon_label}"
    subset = df.dropna(subset=["speed_bin_3ms", "dir_sector_16", next_speed, next_dir]).copy()
    subset[next_speed] = subset[next_speed].astype("int64")
    subset[next_dir] = subset[next_dir].astype("int64")
    group_cols = ["speed_bin_3ms", "dir_sector_16", next_speed, next_dir]
    counts = subset.groupby(group_cols, observed=True).size().reset_index(name="count")
    denom = counts.groupby(["speed_bin_3ms", "dir_sector_16"], observed=True)["count"].transform("sum")
    counts["probability"] = counts["count"] / denom
    counts = counts.rename(
        columns={
            "speed_bin_3ms": "current_speed_bin_3ms",
            "dir_sector_16": "current_dir_sector_16",
            next_speed: f"next_speed_bin_3ms_{horizon_label}",
            next_dir: f"next_dir_sector_16_{horizon_label}",
        }
    )
    counts["horizon"] = horizon_label
    return counts.sort_values(
        ["current_speed_bin_3ms", "current_dir_sector_16", f"next_speed_bin_3ms_{horizon_label}", f"next_dir_sector_16_{horizon_label}"]
    )


def write_summary(df: pd.DataFrame, out_dir: Path) -> None:
    summary = (
        df.groupby(["source", "site", "sensor"], dropna=False)
        .agg(
            rows=("timestamp", "size"),
            start_time=("timestamp", "min"),
            end_time=("timestamp", "max"),
            mean_speed_ms=("wind_speed_ms", "mean"),
            p95_speed_ms=("wind_speed_ms", lambda s: s.quantile(0.95)),
            wind_speed_missing_rate=("wind_speed_ms", lambda s: s.isna().mean()),
            wind_dir_missing_rate=("wind_dir_deg", lambda s: s.isna().mean()),
            air_temp_missing_rate=("air_temp_c", lambda s: s.isna().mean()),
            air_pressure_missing_rate=("air_pressure_hpa", lambda s: s.isna().mean()),
            rel_humidity_missing_rate=("rel_humidity_pct", lambda s: s.isna().mean()),
        )
        .reset_index()
    )
    summary.to_csv(out_dir / "dataset_summary.csv", index=False)


def write_schema(out_dir: Path) -> None:
    schema = {
        "time_resolution": "10 minutes",
        "primary_wind_speed_rule": "prefer 102m mast-corrected cup speed when QC in {2,3,4}; fallback to raw 102m cup speed",
        "primary_wind_direction_rule": "prefer 91m vane direction when QC in {2,3,4}; fallback to 82m ultrasonic direction",
        "direction_encoding": ["wind_dir_sin", "wind_dir_cos"],
        "meteorological_vector_components": {
            "wind_u_ms": "-wind_speed_ms * sin(direction_rad)",
            "wind_v_ms": "-wind_speed_ms * cos(direction_rad)",
        },
        "speed_bins_ms": ["-inf", 3, 6, 9, 12, 15, 20, 25, "inf"],
        "direction_sectors": DIR_SECTORS,
        "rolling_windows": ROLLING_WINDOWS,
        "forecast_horizon_steps": HORIZONS_STEPS,
        "split_policy": "chronological 70/15/15 within the fused FINO1 series",
        "retained_auxiliary_columns": ["air_temp_c", "air_pressure_hpa", "rel_humidity_pct"],
        "accepted_qc_flags": sorted(GOOD_QC_FLAGS),
        "rejected_qc_flags": sorted(BAD_QC_FLAGS),
    }
    (out_dir / "feature_schema.json").write_text(json.dumps(schema, indent=2), encoding="utf-8")


def write_readme(out_dir: Path, featured: pd.DataFrame, supervised_rows: int) -> None:
    fused_dir_fill_rate = float(featured["wind_dir_source"].eq("usa_82m").mean())
    fused_speed_mc_rate = float(featured["wind_speed_source"].eq("cup_102m_mast_corrected").mean())
    text = f"""# FINO1 fused 10-minute wind ML dataset

Generated by `scripts/data_preparation/prepare_fino1_bsh_wind_dataset.py`.

## Purpose

This folder provides a single-station FINO1 wind dataset aligned with the repo's
existing forecasting pipeline while keeping extra columns for future model
upgrades.

## Fusion rules

- Primary wind speed: 102 m cup speed, preferring mast-corrected values when QC
  passes and falling back to raw 102 m cup speed otherwise.
- Primary wind direction: 91 m wind-vane direction, falling back to 82 m
  ultrasonic direction when the vane signal is missing or fails QC.
- Accepted quality flags: `2, 3, 4`.
- Atmospheric auxiliaries retained for later experiments: `air_temp_c`,
  `air_pressure_hpa`, `rel_humidity_pct`.

## Coverage

- Series id: `{SERIES_ID}`
- Canonical rows: `{len(featured)}`
- Supervised rows with a 30-minute target: `{supervised_rows}`
- Time span: `{featured['timestamp'].min()}` to `{featured['timestamp'].max()}`
- Share of rows using mast-corrected 102 m speed: `{fused_speed_mc_rate:.2%}`
- Share of rows using 82 m ultrasonic direction fallback: `{fused_dir_fill_rate:.2%}`

## Files

- `canonical_observations_10min.csv.gz`: cleaned FINO1 time series with fused
  wind variables and retained auxiliary measurements.
- `supervised_learning_table_10min.csv.gz`: forecasting-ready rows with rolling
  features, chronological splits, and 10/30/60 minute targets.
- `transition_probabilities_10m.csv`, `transition_probabilities_30m.csv`,
  `transition_probabilities_60m.csv`: state-transition tables derived from the
  fused wind series.
- `dataset_summary.csv`: coverage and missingness summary.
- `feature_schema.json`: processing and QC rules.

## Why this structure exists

- It is directly compatible with the current recurrent and LightGBM training
  scripts through the same processed-table conventions.
- It preserves enough provenance columns to support later feature changes,
  ablations, or model replacement without redoing raw parsing.
"""
    (out_dir / "README.md").write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--wind-files",
        nargs="+",
        type=Path,
        required=True,
        help="BSH FINO1 wind export CSV files.",
    )
    parser.add_argument(
        "--aux-file",
        type=Path,
        required=True,
        help="BSH FINO1 auxiliary meteorology CSV file.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/processed/wind_ml_10min/fino1_platform_10min"),
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    wind = load_fino1_wind(sorted(args.wind_files))
    aux = load_fino1_aux(args.aux_file)
    df = wind.merge(aux, on="timestamp", how="left", sort=True)
    featured = add_model_features(df)

    canonical_cols = [
        "timestamp",
        "series_id",
        "source",
        "site",
        "sensor",
        "station_id",
        "height_m",
        "wind_speed_height_m",
        "wind_dir_height_m",
        "wind_speed_ms",
        "wind_dir_deg",
        "wind_dir_sin",
        "wind_dir_cos",
        "wind_u_ms",
        "wind_v_ms",
        "wind_speed_ms_raw_102m",
        "wind_speed_ms_mc_102m",
        "wind_dir_deg_vane_91m",
        "wind_dir_deg_usa_82m",
        "wind_speed_source",
        "wind_dir_source",
        "wind_speed_qc_raw",
        "wind_speed_qc_mc",
        "wind_dir_qc_vane",
        "wind_dir_qc_usa",
        "air_temp_c",
        "air_pressure_hpa",
        "rel_humidity_pct",
        "air_temp_qc",
        "air_pressure_qc",
        "rel_humidity_qc",
        "wind_speed_delta_10m",
        "wind_dir_delta_10m_deg",
        "wind_u_delta_10m",
        "wind_v_delta_10m",
        "wind_speed_mean_30m",
        "wind_speed_std_30m",
        "wind_u_mean_30m",
        "wind_v_mean_30m",
        "wind_dir_sin_mean_30m",
        "wind_dir_cos_mean_30m",
        "wind_speed_mean_60m",
        "wind_speed_std_60m",
        "wind_u_mean_60m",
        "wind_v_mean_60m",
        "wind_dir_sin_mean_60m",
        "wind_dir_cos_mean_60m",
        "wind_speed_mean_120m",
        "wind_speed_std_120m",
        "wind_u_mean_120m",
        "wind_v_mean_120m",
        "wind_dir_sin_mean_120m",
        "wind_dir_cos_mean_120m",
        "hour_sin",
        "hour_cos",
        "dayofyear_sin",
        "dayofyear_cos",
        "speed_bin_3ms",
        "dir_sector_16",
        "raw_file",
        "aux_raw_file",
        "split",
    ]
    featured[canonical_cols].to_csv(args.out_dir / "canonical_observations_10min.csv.gz", index=False, compression="gzip")

    supervised = featured.dropna(subset=["target_wind_speed_ms_tplus_30m", "target_wind_dir_deg_tplus_30m"]).copy()
    supervised.to_csv(args.out_dir / "supervised_learning_table_10min.csv.gz", index=False, compression="gzip")

    for horizon in HORIZONS_STEPS:
        transitions = build_transition_table(featured, horizon)
        transitions.to_csv(args.out_dir / f"transition_probabilities_{horizon}.csv", index=False)

    write_summary(featured, args.out_dir)
    write_schema(args.out_dir)
    write_readme(args.out_dir, featured, len(supervised))

    print(f"Wrote FINO1 processed dataset to {args.out_dir}")
    print(f"Series id: {SERIES_ID}")
    print(f"Canonical rows: {len(featured)}")
    print(f"Supervised rows: {len(supervised)}")


if __name__ == "__main__":
    main()
