"""Validated forecast evidence retained until candidate evaluation.

The trained model already returns a full vector trajectory.  This module keeps
that trajectory, its lead-time reliability, and event probabilities together
so planning code does not have to reconstruct control evidence from unrelated
provider attributes.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .forecast_adapter import ForecastResult


@dataclass(frozen=True)
class ForecastEvidence:
    """One forecast origin and all evidence available at that origin.

    ``uv_ms`` uses the forecasting data convention: an ENU downwind velocity
    with component order ``[east, north]`` and unit m/s.  It is not yet
    resolved into the platform axes used by the low-order load model.

    For the current FINO1 sequence dataset, ``uv_ms[i]`` is the dataset record
    at ``origin + (i + 1) * sample_period_s``.  Thus the first element is a
    future ``+sample_period_s`` record, not a value at the forecast origin and
    not an interval average.  A later plant-coupling layer must define its own
    scheduling rule before treating these records as physical substep inputs.
    """

    source: str
    model_version: str
    origin_time: str | None
    sample_period_s: float
    uv_ms: np.ndarray
    event_probs: Mapping[str, float] = field(default_factory=dict)
    lead_reliability: np.ndarray = field(default_factory=lambda: np.ones(0, dtype=float))
    provides_future_preview: bool = True
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def horizon_steps(self) -> int:
        return int(np.asarray(self.uv_ms).shape[0])

    @property
    def horizon_minutes(self) -> float:
        return float(self.horizon_steps * self.sample_period_s / 60.0)


def validate_forecast_evidence(evidence: ForecastEvidence) -> ForecastEvidence:
    """Reject malformed evidence before it can influence a control decision."""

    uv = np.asarray(evidence.uv_ms, dtype=float)
    reliability = np.asarray(evidence.lead_reliability, dtype=float).reshape(-1)
    if uv.ndim != 2 or uv.shape[1] != 2 or uv.shape[0] == 0:
        raise ValueError("ForecastEvidence.uv_ms must have shape (H, 2)")
    if not np.all(np.isfinite(uv)):
        raise ValueError("ForecastEvidence.uv_ms must be finite")
    if not np.isfinite(evidence.sample_period_s) or evidence.sample_period_s <= 0.0:
        raise ValueError("ForecastEvidence.sample_period_s must be positive")
    if reliability.shape != (uv.shape[0],):
        raise ValueError("ForecastEvidence.lead_reliability must contain one value per lead")
    if not np.all(np.isfinite(reliability)):
        raise ValueError("ForecastEvidence.lead_reliability must be finite")
    if np.any(reliability < 0.0) or np.any(reliability > 1.0):
        raise ValueError("ForecastEvidence.lead_reliability must lie in [0, 1]")
    for name, value in evidence.event_probs.items():
        probability = float(value)
        if not np.isfinite(probability) or probability < 0.0 or probability > 1.0:
            raise ValueError(f"event probability {name} must lie in [0, 1]")
    return evidence


def evidence_from_result(
    result: ForecastResult,
    *,
    source: str,
    sample_period_s: float,
    provides_future_preview: bool,
    lead_reliability: np.ndarray | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> ForecastEvidence:
    """Convert a numeric model result into immutable control evidence."""

    uv = np.array(result.wind_uv_raw, dtype=float, copy=True)
    if lead_reliability is None:
        reliability = np.ones(uv.shape[0], dtype=float)
    else:
        reliability = np.array(lead_reliability, dtype=float, copy=True).reshape(-1)
    evidence = ForecastEvidence(
        source=str(source),
        model_version=str(result.model_version),
        origin_time=result.timestamp,
        sample_period_s=float(sample_period_s),
        uv_ms=uv,
        event_probs=dict(result.event_probs),
        lead_reliability=reliability,
        provides_future_preview=bool(provides_future_preview),
        metadata=dict(metadata or {}),
    )
    return validate_forecast_evidence(evidence)


def lead_reliability_from_metrics(
    metrics_csv: str | Path,
    *,
    split: str = "validation",
    metric: str = "vector_mae_ms",
    minimum: float = 0.45,
) -> np.ndarray:
    """Derive relative lead reliability from held-out per-lead forecast error.

    The first lead is the reference.  Later leads receive ``MAE_1 / MAE_k``
    clipped to a conservative floor.  This yields a dimensionless, auditable
    reliability curve without adding tuned control thresholds.
    """

    path = Path(metrics_csv)
    if not path.is_file():
        raise FileNotFoundError(path)
    rows: list[tuple[int, float]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if str(row.get("split", "")) != str(split):
                continue
            scope = str(row.get("scope", ""))
            if not scope.startswith("tplus_") or not scope.endswith("m"):
                continue
            try:
                lead = int(scope.removeprefix("tplus_").removesuffix("m"))
                value = float(row[metric])
            except (KeyError, TypeError, ValueError):
                continue
            if np.isfinite(value) and value > 0.0:
                rows.append((lead, value))
    if not rows:
        raise ValueError(f"no per-lead {metric} rows for split={split!r} in {path}")
    rows.sort(key=lambda item: item[0])
    reference = rows[0][1]
    floor = float(np.clip(minimum, 0.0, 1.0))
    return np.asarray(
        [np.clip(reference / value, floor, 1.0) for _, value in rows],
        dtype=float,
    )
