#!/usr/bin/env python3
"""Prepare 10-minute wind speed/direction data for forecasting models.

The output keeps the native 10-minute resolution and adds physically meaningful
representations for circular wind direction and wind-vector learning.
"""

from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


SPEED_BINS_MS = [-np.inf, 3, 6, 9, 12, 15, 20, 25, np.inf]
SPEED_BIN_LABELS = list(range(len(SPEED_BINS_MS) - 1))
DIR_SECTORS = 16
HORIZONS_STEPS = {"10m": 1, "30m": 3, "60m": 6}
ROLLING_WINDOWS = {"30m": "30min", "60m": "60min", "120m": "120min"}


def circular_diff_deg(target: pd.Series | np.ndarray, base: pd.Series | np.ndarray) -> np.ndarray:
    """Smallest signed angular change from base to target in degrees."""
    return (np.asarray(target, dtype="float64") - np.asarray(base, dtype="float64") + 180.0) % 360.0 - 180.0


def normalize_direction_deg(values: pd.Series) -> pd.Series:
    direction = pd.to_numeric(values, errors="coerce") % 360.0
    direction = direction.mask(direction < 0.0, direction + 360.0)
    return direction


def meteorological_uv(speed_ms: pd.Series, direction_deg: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Convert wind-from direction to meteorological u/v components.

    u is positive eastward and v is positive northward. Meteorological wind
    direction is the direction the wind comes from, so both components are
    negated relative to the unit vector pointing toward the direction angle.
    """
    radians = np.deg2rad(direction_deg.astype("float64"))
    u = -speed_ms.astype("float64") * np.sin(radians)
    v = -speed_ms.astype("float64") * np.cos(radians)
    return u, v


def make_series_id(df: pd.DataFrame) -> pd.Series:
    return df["source"].astype(str) + "::" + df["site"].astype(str) + "::" + df["sensor"].astype(str)


def clean_records(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["wind_speed_ms"] = pd.to_numeric(df["wind_speed_ms"], errors="coerce")
    df["wind_dir_deg"] = normalize_direction_deg(df["wind_dir_deg"])
    df["height_m"] = pd.to_numeric(df["height_m"], errors="coerce")
    if "ti_pct" not in df:
        df["ti_pct"] = np.nan
    df["ti_pct"] = pd.to_numeric(df["ti_pct"], errors="coerce")

    keep = (
        df["timestamp"].notna()
        & df["wind_speed_ms"].between(0, 80, inclusive="both")
        & df["wind_dir_deg"].between(0, 360, inclusive="left")
    )
    df = df.loc[keep].copy()
    df["series_id"] = make_series_id(df)
    df = df.drop_duplicates(["series_id", "timestamp"]).sort_values(["series_id", "timestamp"])
    return df.reset_index(drop=True)


def load_dwd(raw_dir: Path) -> pd.DataFrame:
    rows = []
    dwd_dir = raw_dir / "DWD_10min_wind"
    for path in sorted(dwd_dir.glob("10minutenwerte_wind_*.zip")):
        with zipfile.ZipFile(path) as archive:
            txt_names = [name for name in archive.namelist() if name.endswith(".txt")]
            if not txt_names:
                continue
            with archive.open(txt_names[0]) as handle:
                part = pd.read_csv(handle, sep=";", dtype=str)
        part.columns = [col.strip() for col in part.columns]
        station = part["STATIONS_ID"].astype(str).str.strip()
        out = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(
                    part["MESS_DATUM"].astype(str).str.strip(), format="%Y%m%d%H%M", errors="coerce"
                ),
                "source": "DWD",
                "site": "02115_Helgoland",
                "sensor": "10min_wind",
                "height_m": np.nan,
                "wind_speed_ms": pd.to_numeric(part["FF_10"], errors="coerce"),
                "wind_dir_deg": pd.to_numeric(part["DD_10"], errors="coerce"),
                "ti_pct": np.nan,
                "quality_flag": "QN=" + part["QN"].astype(str).str.strip(),
                "raw_file": path.name,
                "station_id": station,
            }
        )
        rows.append(out)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def load_ndbc(raw_dir: Path) -> pd.DataFrame:
    rows = []
    ndbc_dir = raw_dir / "NDBC_continuous_winds"
    names = [
        "year",
        "month",
        "day",
        "hour",
        "minute",
        "wind_dir_deg",
        "wind_speed_ms",
        "gust_dir_deg",
        "gust_speed_ms",
        "gust_time",
    ]
    for path in sorted(ndbc_dir.glob("*c20*.txt.gz")):
        part = pd.read_csv(path, sep=r"\s+", comment="#", names=names, compression="gzip")
        station = path.name.split("c", 1)[0]
        timestamp = pd.to_datetime(
            dict(
                year=pd.to_numeric(part["year"], errors="coerce"),
                month=pd.to_numeric(part["month"], errors="coerce"),
                day=pd.to_numeric(part["day"], errors="coerce"),
                hour=pd.to_numeric(part["hour"], errors="coerce"),
                minute=pd.to_numeric(part["minute"], errors="coerce"),
            ),
            errors="coerce",
        )
        out = pd.DataFrame(
            {
                "timestamp": timestamp,
                "source": "NOAA_NDBC",
                "site": station,
                "sensor": "continuous_winds",
                "height_m": np.nan,
                "wind_speed_ms": pd.to_numeric(part["wind_speed_ms"], errors="coerce").replace(99.0, np.nan),
                "wind_dir_deg": pd.to_numeric(part["wind_dir_deg"], errors="coerce").replace(999.0, np.nan),
                "ti_pct": np.nan,
                "quality_flag": "raw_continuous",
                "raw_file": path.name,
                "station_id": station,
            }
        )
        rows.append(out)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def dtu_column(ds: xr.Dataset, var: str) -> pd.Series:
    da = ds[var]
    if "statistics" in da.dims:
        da = da.sel(statistics="mean")
    return pd.Series(np.asarray(da.values, dtype="float64"))


def load_dtu_series(
    nc_path: Path,
    source: str,
    site: str,
    specs: list[dict[str, str]],
) -> pd.DataFrame:
    ds = xr.open_dataset(nc_path)
    timestamps = pd.Series(pd.to_datetime(ds["time"].values))
    rows = []
    try:
        for spec in specs:
            speed_var = spec["speed_var"]
            dir_var = spec["dir_var"]
            speed = dtu_column(ds, speed_var)
            direction = dtu_column(ds, dir_var)
            mask = speed.notna() & direction.notna()
            quality_bits = []
            for var in (speed_var, dir_var):
                qc_var = f"{var}_qc"
                if qc_var in ds:
                    qc = dtu_column(ds, qc_var)
                    mask &= qc.eq(0)
                    quality_bits.append(f"{qc_var}=0")
            ti = np.nan
            ti_var = spec.get("ti_var")
            if ti_var and ti_var in ds:
                ti = dtu_column(ds, ti_var)
            height = spec.get("height_m")
            if height is None:
                height = ds[speed_var].attrs.get("height", np.nan)
            out = pd.DataFrame(
                {
                    "timestamp": timestamps,
                    "source": source,
                    "site": site,
                    "sensor": spec["sensor"],
                    "height_m": float(height) if pd.notna(height) else np.nan,
                    "wind_speed_ms": speed,
                    "wind_dir_deg": direction,
                    "ti_pct": ti,
                    "quality_flag": ";".join(quality_bits) if quality_bits else "finite_only",
                    "raw_file": nc_path.name,
                    "station_id": site,
                }
            )
            rows.append(out.loc[mask].copy())
    finally:
        ds.close()
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def load_dtu(raw_dir: Path) -> pd.DataFrame:
    dtu_dir = raw_dir / "DTU_wind_farm"
    rows = [
        load_dtu_series(
            dtu_dir / "norre_m2_all.nc",
            "DTU",
            "norre_m2",
            [
                {"sensor": "mast1_31m", "speed_var": "s31_1", "dir_var": "d31_1", "height_m": 31},
                {"sensor": "mast2_31m_34m_dir", "speed_var": "s31_2", "dir_var": "d34_2", "height_m": 31},
            ],
        ),
        load_dtu_series(
            dtu_dir / "hornsrev_all.nc",
            "DTU",
            "hornsrev_offshore_mast",
            [
                {
                    "sensor": "sonic_50m",
                    "speed_var": "ws40",
                    "dir_var": "wd40",
                    "ti_var": "TI_ws40",
                    "height_m": 50,
                }
            ],
        ),
        load_dtu_series(
            dtu_dir / "delabole_all.nc",
            "DTU",
            "delabole",
            [
                {"sensor": "mast2_33m", "speed_var": "ws33_m2", "dir_var": "wd33_m2", "ti_var": "TI_ws33_m2"},
                {"sensor": "mast1_44m", "speed_var": "ws44_m1", "dir_var": "wd44_m1", "ti_var": "TI_ws44_m1"},
            ],
        ),
    ]
    return pd.concat(rows, ignore_index=True)


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
        df[f"target_speed_delta_ms_tplus_{label}"] = (
            df[f"target_wind_speed_ms_tplus_{label}"] - df["wind_speed_ms"]
        )
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


def write_normalization_stats(df: pd.DataFrame, out_dir: Path) -> None:
    feature_cols = [
        "wind_speed_ms",
        "wind_dir_sin",
        "wind_dir_cos",
        "wind_u_ms",
        "wind_v_ms",
        "wind_speed_delta_10m",
        "wind_dir_delta_10m_deg",
        "wind_u_delta_10m",
        "wind_v_delta_10m",
        "ti_pct",
        "wind_speed_mean_30m",
        "wind_speed_std_30m",
        "wind_speed_mean_60m",
        "wind_speed_std_60m",
        "wind_speed_mean_120m",
        "wind_speed_std_120m",
    ]
    train = df[df["split"].eq("train")]
    rows = []
    for col in feature_cols:
        if col in train:
            s = pd.to_numeric(train[col], errors="coerce")
            rows.append(
                {
                    "feature": col,
                    "mean_train": s.mean(),
                    "std_train": s.std(),
                    "min_train": s.min(),
                    "max_train": s.max(),
                    "missing_rate_train": s.isna().mean(),
                }
            )
    pd.DataFrame(rows).to_csv(out_dir / "normalization_stats_train.csv", index=False)


def write_summary(df: pd.DataFrame, out_dir: Path) -> None:
    summary = (
        df.groupby(["source", "site", "sensor"], dropna=False)
        .agg(
            rows=("timestamp", "size"),
            start_time=("timestamp", "min"),
            end_time=("timestamp", "max"),
            mean_speed_ms=("wind_speed_ms", "mean"),
            p95_speed_ms=("wind_speed_ms", lambda s: s.quantile(0.95)),
            missing_ti_rate=("ti_pct", lambda s: s.isna().mean()),
        )
        .reset_index()
    )
    summary.to_csv(out_dir / "dataset_summary.csv", index=False)


def write_readme(out_dir: Path, total_rows: int, supervised_rows: int) -> None:
    text = f"""# 10-minute wind ML processed dataset

This folder is generated by `scripts/data_preparation/prepare_wind_ml_dataset.py`.

## Files

- `canonical_observations_10min.csv.gz`: cleaned 10-minute wind observations from DTU, DWD and NOAA/NDBC.
- `supervised_learning_table_10min.csv.gz`: forecasting-ready rows with circular direction features, wind-vector components, rolling history features, chronological split labels and 10/30/60 minute targets.
- `transition_probabilities_10m.csv`, `transition_probabilities_30m.csv`, `transition_probabilities_60m.csv`: 4D wind-state transition probabilities.
- `dataset_summary.csv`: row counts and time coverage by source/site/sensor.
- `normalization_stats_train.csv`: train-split statistics for model scaling without validation/test leakage.
- `feature_schema.json`: column groups and bin definitions.

## Processing choices

- Native 10-minute observations are preserved because several wind-turbine/SCADA datasets and short-term forecasting studies use this scale, and it keeps sub-hour wind-change detail available.
- Wind direction is not used as a plain linear number. It is encoded as `wind_dir_sin` and `wind_dir_cos` to avoid the 359-to-0 degree discontinuity.
- Meteorological wind vector components are added as `wind_u_ms` and `wind_v_ms`, using wind-from direction: `u = -speed * sin(direction)`, `v = -speed * cos(direction)`.
- Direction changes use signed circular deltas in [-180, 180) degrees.
- Future labels are matched by exact timestamps, so the 30-minute target must come from `t+30min`; missing intervals are not silently treated as valid targets.
- The main ballast-control-relevant target is the 30-minute horizon, with 10-minute and 60-minute horizons retained for comparison.
- State transitions use 3 m/s speed bins and 16 wind-direction sectors, producing the same conceptual 4D matrix as speed-state x direction-state -> next-speed-state x next-direction-state.
- Train/validation/test splits are chronological within each station/sensor series to reduce time leakage.

## Row counts

- Canonical observations: {total_rows}
- Supervised rows with a 30-minute future target: {supervised_rows}
"""
    (out_dir / "README.md").write_text(text, encoding="utf-8")


def write_schema(out_dir: Path) -> None:
    schema = {
        "time_resolution": "10 minutes",
        "direction_encoding": ["wind_dir_sin", "wind_dir_cos"],
        "meteorological_vector_components": {
            "wind_u_ms": "-wind_speed_ms * sin(direction_rad)",
            "wind_v_ms": "-wind_speed_ms * cos(direction_rad)",
        },
        "speed_bins_ms": ["-inf", 3, 6, 9, 12, 15, 20, 25, "inf"],
        "direction_sectors": DIR_SECTORS,
        "rolling_windows": ROLLING_WINDOWS,
        "forecast_horizon_steps": HORIZONS_STEPS,
        "preferred_first_model_target": "target_speed_delta_ms_tplus_30m + target_dir_delta_deg_tplus_30m or target wind_u/v tplus_30m",
        "split_policy": "chronological 70/15/15 within each source-site-sensor series",
    }
    (out_dir / "feature_schema.json").write_text(json.dumps(schema, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", default="data/raw/wind_10min", type=Path)
    parser.add_argument("--out-dir", default="data/processed/wind_ml_10min", type=Path)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    raw_frames = [load_dtu(args.raw_dir), load_dwd(args.raw_dir), load_ndbc(args.raw_dir)]
    raw = pd.concat([frame for frame in raw_frames if not frame.empty], ignore_index=True)
    featured = add_model_features(raw)

    canonical_cols = [
        "timestamp",
        "series_id",
        "source",
        "site",
        "sensor",
        "station_id",
        "height_m",
        "wind_speed_ms",
        "wind_dir_deg",
        "wind_dir_sin",
        "wind_dir_cos",
        "wind_u_ms",
        "wind_v_ms",
        "ti_pct",
        "speed_bin_3ms",
        "dir_sector_16",
        "quality_flag",
        "raw_file",
        "split",
    ]
    featured[canonical_cols].to_csv(args.out_dir / "canonical_observations_10min.csv.gz", index=False, compression="gzip")

    supervised = featured.dropna(subset=["target_wind_speed_ms_tplus_30m", "target_wind_dir_deg_tplus_30m"]).copy()
    supervised.to_csv(args.out_dir / "supervised_learning_table_10min.csv.gz", index=False, compression="gzip")

    for horizon in HORIZONS_STEPS:
        transitions = build_transition_table(featured, horizon)
        transitions.to_csv(args.out_dir / f"transition_probabilities_{horizon}.csv", index=False)

    write_summary(featured, args.out_dir)
    write_normalization_stats(featured, args.out_dir)
    write_schema(args.out_dir)
    write_readme(args.out_dir, total_rows=len(featured), supervised_rows=len(supervised))
    print(f"Wrote processed wind ML dataset to {args.out_dir}")
    print(f"Canonical rows: {len(featured):,}")
    print(f"Supervised rows with 30m target: {len(supervised):,}")


if __name__ == "__main__":
    main()
