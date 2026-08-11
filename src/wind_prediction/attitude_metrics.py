"""Shared attitude exposure metric definitions.

This module is intentionally small: it gives analysis scripts one vocabulary
for posture exposure without changing controller behavior.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


STARTUP_SPLIT_S = 300.0


@dataclass(frozen=True)
class AttitudeMetricBand:
    key: str
    threshold_deg: float
    label: str
    meaning: str
    control_use: str


ATTITUDE_METRIC_BANDS: tuple[AttitudeMetricBand, ...] = (
    AttitudeMetricBand(
        key="1p5deg",
        threshold_deg=1.5,
        label="fine_posture_exposure",
        meaning="fine posture-deviation occupancy",
        control_use="early posture-cost accounting",
    ),
    AttitudeMetricBand(
        key="2deg",
        threshold_deg=2.0,
        label="moderate_posture_exposure",
        meaning="moderate posture-deviation occupancy",
        control_use="economy-versus-posture acceptance accounting",
    ),
    AttitudeMetricBand(
        key="3deg",
        threshold_deg=3.0,
        label="comfort_outside",
        meaning="comfort/economy band occupancy",
        control_use="low-posture tracking cost accounting",
    ),
    AttitudeMetricBand(
        key="4deg",
        threshold_deg=4.0,
        label="near_service_pressure",
        meaning="service-margin occupancy",
        control_use="economy caution and Pareto cost accounting",
    ),
    AttitudeMetricBand(
        key="5deg",
        threshold_deg=5.0,
        label="service_pressure",
        meaning="service-pressure occupancy, not a failure line",
        control_use="duration/area budget and release-veto audit",
    ),
    AttitudeMetricBand(
        key="7p5deg",
        threshold_deg=7.5,
        label="severe_pressure",
        meaning="strong posture-pressure occupancy",
        control_use="strong warning if sustained or worsening",
    ),
    AttitudeMetricBand(
        key="10deg",
        threshold_deg=10.0,
        label="operating_limit",
        meaning="operating-limit occupancy",
        control_use="hard-protection review if sustained or increased",
    ),
)


def _exposure_metric_keys() -> tuple[str, ...]:
    keys: list[str] = []
    scope_prefixes = ("", "startup_0_300s_", "steady_after_300s_")
    metric_prefixes = ("time", "max_continuous", "area")
    for scope in scope_prefixes:
        for band in ATTITUDE_METRIC_BANDS:
            for metric in metric_prefixes:
                suffix = "deg_s" if metric == "area" else "s"
                keys.append(f"{scope}{metric}_over_{band.key}_{suffix}")
    return tuple(keys)


ATTITUDE_EXPOSURE_METRIC_KEYS: tuple[str, ...] = _exposure_metric_keys()


def _as_1d_float(values: Iterable[float] | np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float).reshape(-1)
    return arr


def _dt_s(time_s: np.ndarray | None, fallback_dt_s: float) -> float:
    fallback = max(float(fallback_dt_s), 1e-9)
    if time_s is None or time_s.size < 2:
        return fallback
    diffs = np.diff(time_s.astype(float))
    diffs = diffs[np.isfinite(diffs) & (diffs > 0.0)]
    if diffs.size == 0:
        return fallback
    return float(np.median(diffs))


def _max_run_s(mask: np.ndarray, dt_s: float) -> float:
    if mask.size == 0:
        return 0.0
    best = 0
    current = 0
    for value in mask.astype(bool):
        if value:
            current += 1
            best = max(best, current)
        else:
            current = 0
    return float(best) * float(dt_s)


def compute_attitude_exposure_metrics(
    pitch_deg: Iterable[float] | np.ndarray,
    roll_deg: Iterable[float] | np.ndarray,
    *,
    time_s: Iterable[float] | np.ndarray | None = None,
    fallback_dt_s: float = 1.0,
    startup_split_s: float = STARTUP_SPLIT_S,
) -> dict[str, float]:
    """Return threshold exposure metrics for max-axis platform attitude.

    Exposure is computed from ``max(abs(pitch), abs(roll))``.  ``area`` metrics
    are degree-seconds above the threshold.
    """
    pitch = _as_1d_float(pitch_deg)
    roll = _as_1d_float(roll_deg)
    n = min(pitch.size, roll.size)
    out = {key: 0.0 for key in ATTITUDE_EXPOSURE_METRIC_KEYS}
    if n == 0:
        return out
    pitch = pitch[:n]
    roll = roll[:n]
    valid = np.isfinite(pitch) & np.isfinite(roll)
    if time_s is not None:
        time_arr = _as_1d_float(time_s)[:n]
        if time_arr.size != n:
            time_arr = None
        else:
            valid &= np.isfinite(time_arr)
    else:
        time_arr = None
    if not bool(np.any(valid)):
        return out
    pitch = pitch[valid]
    roll = roll[valid]
    if time_arr is not None:
        time_arr = time_arr[valid]
    n = pitch.size
    dt = _dt_s(time_arr, fallback_dt_s)
    max_axis = np.maximum(np.abs(pitch), np.abs(roll))

    if time_arr is None:
        rel_time = np.arange(n, dtype=float) * dt
    else:
        rel_time = time_arr.astype(float) - float(time_arr[0])
    scopes = (
        ("", np.ones(n, dtype=bool)),
        ("startup_0_300s_", rel_time < float(startup_split_s)),
        ("steady_after_300s_", rel_time >= float(startup_split_s)),
    )

    for scope_prefix, scope_mask in scopes:
        scoped_axis = max_axis[scope_mask]
        for band in ATTITUDE_METRIC_BANDS:
            over = scoped_axis > float(band.threshold_deg)
            excess = np.clip(scoped_axis - float(band.threshold_deg), 0.0, None)
            out[f"{scope_prefix}time_over_{band.key}_s"] = float(np.sum(over)) * dt
            out[f"{scope_prefix}max_continuous_over_{band.key}_s"] = _max_run_s(
                over,
                dt,
            )
            out[f"{scope_prefix}area_over_{band.key}_deg_s"] = (
                float(np.sum(excess)) * dt
            )
    return out


def attitude_metric_semantics_markdown_rows() -> list[str]:
    rows = []
    for band in ATTITUDE_METRIC_BANDS:
        rows.append(
            f"| `{band.threshold_deg:g} deg` | `{band.label}` | "
            f"{band.meaning} | {band.control_use} |"
        )
    return rows
