"""Explicit time contract for already-converted platform load forecasts.

``GeneralizedLoadForecast`` stores one current generalized load and a finite
sequence of future generalized-load points.  Every load uses the frozen
platform convention ``[X, Y, Z, K, M, N]`` about one reference point.

It is intentionally not a wind-forecast adapter.  In particular, it does not
convert ``ForecastEvidence`` or wind vectors, choose aerodynamic inputs,
interpolate point forecasts, schedule physical substeps, or enter a controller
or pump model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


def _finite_load(name: str, value: Any) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must have shape (6,) and contain finite values") from exc
    if array.shape != (6,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must have shape (6,) and contain finite values")
    result = np.array(array, dtype=float, copy=True)
    result.setflags(write=False)
    return result


def _finite_loads(value: Any) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "future_generalized_loads must have shape (H, 6) with H > 0 and finite values"
        ) from exc
    if array.ndim != 2 or array.shape[0] == 0 or array.shape[1] != 6:
        raise ValueError(
            "future_generalized_loads must have shape (H, 6) with H > 0 and finite values"
        )
    if not np.all(np.isfinite(array)):
        raise ValueError(
            "future_generalized_loads must have shape (H, 6) with H > 0 and finite values"
        )
    result = np.array(array, dtype=float, copy=True)
    result.setflags(write=False)
    return result


def _lead_times(value: Any, *, horizon_steps: int) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "lead_times_s must contain one finite positive time per future load"
        ) from exc
    if array.shape != (horizon_steps,) or not np.all(np.isfinite(array)):
        raise ValueError(
            "lead_times_s must contain one finite positive time per future load"
        )
    if np.any(array <= 0.0) or np.any(np.diff(array) <= 0.0):
        raise ValueError("lead_times_s must be positive and strictly increasing")
    result = np.array(array, dtype=float, copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class GeneralizedLoadForecast:
    """Current and future physical load points in a shared platform convention.

    Future samples are discrete prediction points at ``lead_times_s``.  They
    are not interval averages and do not prescribe how a later plant model
    should hold or interpolate a load between points.
    """

    current_generalized_load: Any
    future_generalized_loads: Any
    lead_times_s: Any

    def __post_init__(self) -> None:
        current = _finite_load("current_generalized_load", self.current_generalized_load)
        future = _finite_loads(self.future_generalized_loads)
        leads = _lead_times(self.lead_times_s, horizon_steps=int(future.shape[0]))
        object.__setattr__(self, "current_generalized_load", current)
        object.__setattr__(self, "future_generalized_loads", future)
        object.__setattr__(self, "lead_times_s", leads)

    @property
    def horizon_steps(self) -> int:
        """Return the number of discrete future physical-load points."""

        return int(self.future_generalized_loads.shape[0])

    def future_load_at(self, index: int) -> np.ndarray:
        """Return an isolated copy of the future load at one valid lead index."""

        if not isinstance(index, int) or isinstance(index, bool):
            raise TypeError("index must be an integer")
        if index < 0 or index >= self.horizon_steps:
            raise IndexError("future load index is outside the forecast horizon")
        return np.array(self.future_generalized_loads[index], dtype=float, copy=True)

    def lead_time_at(self, index: int) -> float:
        """Return the positive lead time of one valid future load point."""

        if not isinstance(index, int) or isinstance(index, bool):
            raise TypeError("index must be an integer")
        if index < 0 or index >= self.horizon_steps:
            raise IndexError("future load index is outside the forecast horizon")
        return float(self.lead_times_s[index])


__all__ = ["GeneralizedLoadForecast"]
