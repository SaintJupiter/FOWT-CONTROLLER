#!/usr/bin/env python3
"""Verify that the reactive reference consumes only the current wind sample.

This is an input-semantics check, not a controller-performance experiment.
Future LSTM values may change without changing the load sequence exposed to
the current-observation reference. Changing the current observation must
remain visible after the common rotor-load conversion.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SRC_DIRECTORY = ROOT / "src"
VALIDATION_DIRECTORY = Path(__file__).resolve().parent
for directory in (SRC_DIRECTORY, VALIDATION_DIRECTORY):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from fowt_platform import IncrementalState  # noqa: E402
from preview_mpc_experiment_runtime import (  # noqa: E402
    assemble_planner_load_forecast,
    bind_source_identity,
)
from real_lstm_preview_fixture import (  # noqa: E402
    RealLstmPreviewResources,
    RealLstmWindRecord,
    assemble_planner_rotor_preview_from_record,
    infer_real_lstm_wind_record,
    prepare_real_lstm_preview_resources,
)
from wind_prediction.replay_dataset import TIMESTAMP_FMT  # noqa: E402


DEFAULT_ORIGIN = "2022-12-09 20:40:00"


def _visible_current_observation_input(
    *,
    resources: RealLstmPreviewResources,
    source_record: RealLstmWindRecord,
) -> tuple[np.ndarray, np.ndarray, str, bool]:
    preview = assemble_planner_rotor_preview_from_record(
        resources=resources,
        source_record=source_record,
        platform_state=IncrementalState.zeros(),
        planner_forecast_mode="current_observation",
        horizon_blocks=1,
    )
    forecast = assemble_planner_load_forecast(
        source_preview=preview,
        planner_forecast_mode="current_observation",
        planner_horizon_blocks=1,
    )
    identity = bind_source_identity(
        source_preview=preview,
        planner_forecast_mode="current_observation",
        lead_times_s=forecast.lead_times_s,
        oracle_records=None,
    )
    return (
        forecast.current_generalized_load,
        forecast.future_generalized_loads,
        identity.source_record_sha256,
        identity.uses_future_information,
    )


def run_check(*, origin: datetime, device: str = "cpu") -> dict[str, Any]:
    resources = prepare_real_lstm_preview_resources(device=device)
    source_record = infer_real_lstm_wind_record(
        origin=origin,
        resources=resources,
    )
    baseline = _visible_current_observation_input(
        resources=resources,
        source_record=source_record,
    )

    changed_future = replace(
        source_record,
        forecast=replace(
            source_record.forecast,
            uv_ms=np.full_like(source_record.forecast.uv_ms, 1.0e6),
        ),
    )
    future_changed = _visible_current_observation_input(
        resources=resources,
        source_record=changed_future,
    )
    future_isolated = bool(
        np.array_equal(baseline[0], future_changed[0])
        and np.array_equal(baseline[1], future_changed[1])
        and baseline[2] == future_changed[2]
    )
    if not future_isolated:
        raise RuntimeError("future LSTM values leaked into current-observation input")

    changed_current = replace(
        source_record,
        current_enu_downwind_wind_mps=(
            1.05 * source_record.current_enu_downwind_wind_mps
        ),
    )
    current_changed = _visible_current_observation_input(
        resources=resources,
        source_record=changed_current,
    )
    current_visible = bool(
        not np.allclose(baseline[0], current_changed[0], rtol=0.0, atol=1.0e-9)
        and not np.allclose(baseline[1], current_changed[1], rtol=0.0, atol=1.0e-9)
        and baseline[2] != current_changed[2]
    )
    if not current_visible:
        raise RuntimeError("current wind change did not reach the reactive input")
    if baseline[3] or future_changed[3] or current_changed[3]:
        raise RuntimeError("current-observation input was mislabeled as future information")

    return {
        "purpose": "current_observation_input_future_isolation_check",
        "origin": origin.strftime(TIMESTAMP_FMT),
        "future_lstm_values_isolated": future_isolated,
        "current_observation_visible": current_visible,
        "uses_future_information": False,
        "is_controller_performance_result": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--origin", default=DEFAULT_ORIGIN)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()
    result = run_check(
        origin=datetime.strptime(args.origin, TIMESTAMP_FMT),
        device=args.device,
    )
    payload = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
