#!/usr/bin/env python3
"""Run the compact controller through three deterministic control cycles."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from wind_prediction.controller import (
    ControllerMeasurements,
    ControllerRuntimeState,
    ForecastAssistedBallastController,
    ForecastEvidence,
    load_controller_config,
)


ROOT = Path(__file__).resolve().parents[2]


def _forecast(origin: str, wind_speed_ms: float) -> ForecastEvidence:
    vectors = np.array(
        [(0.0, -(wind_speed_ms + offset)) for offset in (0, 1, 2, 2, 1, 0)],
        dtype=float,
    )
    return ForecastEvidence(
        source="controller_core_smoke",
        model_version="deterministic-smoke-v1",
        origin_time=origin,
        sample_period_s=600.0,
        uv_ms=vectors,
        lead_reliability=np.full(6, 0.9),
        event_probs={
            "attention_event_0_20m": 0.9,
            "attention_event_20_40m": 0.9,
            "attention_event_40_60m": 0.9,
        },
        provides_future_preview=True,
    )


def main() -> None:
    loaded = load_controller_config(ROOT / "configs/controller_core_v2.json")
    controller = ForecastAssistedBallastController(loaded.config)
    state = ControllerRuntimeState.initialize((900_000.0,) * 3)
    rows = []
    for index, posture in enumerate(((2.4, 0.2), (1.9, 0.4), (1.2, 0.1))):
        time_s = float(index * controller.config.stage_duration_s)
        result = controller.step(
            ControllerMeasurements(
                time_s=time_s,
                posture_deg=posture,
                posture_rate_deg_s=(0.0, 0.0),
                current_wind_uv_ms=(0.0, -10.0),
            ),
            state,
            _forecast(f"2026-08-12T00:{index * 20:02d}:00", 10.0),
        )
        state = result.state
        rows.append(dict(result.trace))
    payload = {
        "schema_version": "controller_core_smoke.v2",
        "controller_config": {
            "path": str(loaded.source_path.relative_to(ROOT)),
            "sha256": loaded.sha256,
        },
        "forecast_source": "controller_core_smoke",
        "forecast_model_version": "deterministic-smoke-v1",
        "completed_cycles": len(rows),
        "traces": rows,
    }
    print(json.dumps(payload, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
