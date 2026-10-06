"""Small, controller-independent wind-observation conversions.

The replay dataset exposes meteorological wind observations as speed and
wind-from direction.  Physical load assembly instead expects an ENU downwind
velocity ``[east, north]``.  Keeping this conversion below both the legacy
controller adapter and the physical shadow path avoids two slightly different
interpretations of the same observation record.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any


def wind_observation_to_enu_downwind_ms(
    wind_observation: Mapping[str, Any],
) -> tuple[float, float]:
    """Convert a meteorological wind-from observation to ENU downwind m/s.

    The observation must use ``ws`` in m/s and ``wd_deg`` clockwise from true
    north.  This function only converts the recorded convention; it does not
    resolve platform heading, apply a height correction, or infer an airspeed
    at the rotor.
    """

    try:
        speed = float(wind_observation["ws"])
        direction_deg = float(wind_observation["wd_deg"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("wind observation must contain numeric ws and wd_deg") from exc
    if not math.isfinite(speed) or speed < 0.0 or not math.isfinite(direction_deg):
        raise ValueError("wind speed and direction must be finite and speed non-negative")
    direction_rad = math.radians(direction_deg)
    return (
        -speed * math.sin(direction_rad),
        -speed * math.cos(direction_rad),
    )


__all__ = ["wind_observation_to_enu_downwind_ms"]
