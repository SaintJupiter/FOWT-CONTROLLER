"""Shared source binding for physical endpoint previews.

Two factual sources can currently provide a new physical tank target: the
older relative-load endpoint diagnostic and the newer endpoint state-equation
diagnostic.  Both expose the same small lifecycle surface, while this module
keeps their distinct provenance checks in one place.
"""

from __future__ import annotations

from typing import TypeAlias

import numpy as np

from .forecast_platform_trajectory import ForecastPlatformTrajectory
from .forecast_rhs_ballast_endpoint_preview import ForecastRhsBallastEndpointPreview
from .physical_forecast_endpoint_preview import PhysicalForecastEndpointPreview


PhysicalEndpointPreview: TypeAlias = (
    PhysicalForecastEndpointPreview | ForecastRhsBallastEndpointPreview
)


def is_physical_endpoint_preview(value: object) -> bool:
    """Return whether value is one supported factual endpoint source."""

    return isinstance(
        value,
        (PhysicalForecastEndpointPreview, ForecastRhsBallastEndpointPreview),
    )


def validate_endpoint_preview_at_trajectory_lead(
    *,
    preview: PhysicalEndpointPreview,
    trajectory: ForecastPlatformTrajectory,
    lead_index: int,
    lead_time_s: float,
    first_lead_only: bool,
) -> None:
    """Require one preview to retain an exact forecast-trajectory endpoint."""

    if not isinstance(trajectory, ForecastPlatformTrajectory):
        raise TypeError("trajectory must be ForecastPlatformTrajectory")
    if not isinstance(lead_index, int) or isinstance(lead_index, bool):
        raise TypeError("lead_index must be an integer")
    if not 0 <= lead_index < len(trajectory.steps):
        raise ValueError("lead_index must refer to one forecast trajectory step")
    expected_lead_time_s = float(lead_time_s)
    if not np.isfinite(expected_lead_time_s) or not np.isclose(
        expected_lead_time_s,
        trajectory.steps[lead_index].end_time_s,
        rtol=0.0,
        atol=1.0e-9,
    ):
        raise ValueError("lead_time_s must match the forecast trajectory endpoint")
    if not is_physical_endpoint_preview(preview):
        raise TypeError("preview must be a supported physical endpoint preview")

    if isinstance(preview, PhysicalForecastEndpointPreview):
        if preview.trajectory is not trajectory:
            raise ValueError(
                "relative-load endpoint preview must use the matching forecast trajectory"
            )
    else:
        if preview.rhs_diagnostic.rhs_point.trajectory is not trajectory:
            raise ValueError(
                "rhs endpoint preview must use the matching rhs trajectory"
            )
    if preview.lead_index != lead_index or not np.isclose(
        preview.lead_time_s,
        expected_lead_time_s,
        rtol=0.0,
        atol=1.0e-9,
    ):
        raise ValueError("endpoint preview must use the matching rhs forecast lead")
    if first_lead_only and preview.lead_index != 0:
        raise ValueError("endpoint preview must use the first forecast interval")


__all__ = [
    "PhysicalEndpointPreview",
    "is_physical_endpoint_preview",
    "validate_endpoint_preview_at_trajectory_lead",
]
