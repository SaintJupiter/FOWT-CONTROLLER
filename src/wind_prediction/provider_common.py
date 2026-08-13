"""Shared constants and small validation helpers for the ballast provider."""

from __future__ import annotations

import numpy as np


def complete_event_probabilities_available(
    event_probs: dict[str, float] | None,
    expected_columns,
) -> bool:
    """Return whether all expected event probabilities are finite probabilities."""

    expected = tuple(str(name) for name in expected_columns)
    if not expected or not isinstance(event_probs, dict):
        return False
    try:
        values = [float(event_probs[name]) for name in expected]
    except (KeyError, TypeError, ValueError):
        return False
    return bool(np.all(np.isfinite(values))) and all(
        0.0 <= value <= 1.0 for value in values
    )


HOLD_RELEASE_ENTER_PITCH_DEG = 2.5
HOLD_RELEASE_ENTER_ROLL_DEG = 2.5
HOLD_FORECAST_POSTURE_RELIEF_NORM = 0.20
FORECAST_RELIEF_MARGIN_NORM = 0.25
FORECAST_REINTENSIFY_HIGH_NORM = 1.05
FORECAST_REINTENSIFY_RISE_NORM = 0.10
