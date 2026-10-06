"""Canonical grouping of discrete forecast leads into controller stages.

The controller, policy trace and physical forecast diagnostics all receive
discrete future records.  This module makes their time grouping explicit: it
does not interpolate a forecast, invent a value at the origin, or decide an
action.  A grid is only a shared index-and-time contract.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any


def _positive_finite(name: str, value: Any) -> float:
    try:
        scalar = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be positive and finite") from exc
    if not math.isfinite(scalar) or scalar <= 0.0:
        raise ValueError(f"{name} must be positive and finite")
    return scalar


def _positive_int(name: str, value: Any) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer")
    try:
        integer = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive integer") from exc
    if integer != value or integer <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return integer


@dataclass(frozen=True)
class ForecastStageWindow:
    """One right-closed planning stage over discrete future records.

    ``point_start_index`` is inclusive and ``point_end_index`` is exclusive.
    ``lead_start_s`` and ``lead_end_s`` describe the first and final discrete
    forecast records in the window, both strictly after the forecast origin.
    """

    index: int
    point_start_index: int
    point_end_index: int
    lead_start_s: float
    lead_end_s: float

    @property
    def point_indices(self) -> tuple[int, ...]:
        return tuple(range(self.point_start_index, self.point_end_index))

    @property
    def terminal_point_index(self) -> int:
        """Return the final discrete point representing this stage endpoint."""

        return self.point_end_index - 1


@dataclass(frozen=True)
class ForecastStageGrid:
    """Shared time and indexing contract for a fixed controller horizon."""

    sample_period_s: float
    stage_duration_s: float
    stage_count: int
    points_per_stage: int
    windows: tuple[ForecastStageWindow, ...]

    @property
    def required_point_count(self) -> int:
        return self.stage_count * self.points_per_stage


def build_forecast_stage_grid(
    *,
    sample_period_s: float,
    stage_duration_s: float,
    stage_count: int,
    available_point_count: int | None = None,
) -> ForecastStageGrid:
    """Build a stage contract without altering the supplied forecast records.

    The stage duration must be an integer number of samples.  For example,
    with 600 s samples and 1200 s stages, stage 0 contains forecast points
    0 and 1, whose leads are +600 and +1200 s.  The origin itself is never
    inserted into a forecast stage.
    """

    sample_period = _positive_finite("sample_period_s", sample_period_s)
    stage_duration = _positive_finite("stage_duration_s", stage_duration_s)
    count = _positive_int("stage_count", stage_count)
    points_per_stage_float = stage_duration / sample_period
    points_per_stage = int(round(points_per_stage_float))
    if points_per_stage <= 0 or not math.isclose(
        points_per_stage_float,
        points_per_stage,
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise ValueError(
            "stage_duration_s must be an integer multiple of sample_period_s"
        )

    required = count * points_per_stage
    if available_point_count is not None:
        available = _positive_int("available_point_count", available_point_count)
        if available < required:
            raise ValueError(
                "forecast sequence does not cover the configured control horizon: "
                f"need {required} points, got {available}"
            )

    windows = tuple(
        ForecastStageWindow(
            index=index,
            point_start_index=index * points_per_stage,
            point_end_index=(index + 1) * points_per_stage,
            lead_start_s=float((index * points_per_stage + 1) * sample_period),
            lead_end_s=float((index + 1) * points_per_stage * sample_period),
        )
        for index in range(count)
    )
    return ForecastStageGrid(
        sample_period_s=sample_period,
        stage_duration_s=stage_duration,
        stage_count=count,
        points_per_stage=points_per_stage,
        windows=windows,
    )


__all__ = [
    "ForecastStageGrid",
    "ForecastStageWindow",
    "build_forecast_stage_grid",
]
