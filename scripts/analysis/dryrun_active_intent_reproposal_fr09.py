#!/usr/bin/env python3
"""Dry-run active-intent target reproposal on fr_relief_09 bucket 8-9 logs.

This is not a closed-loop simulation. It replays the target-manager branch with
real logged masses/action vectors to verify that repeated `active_small` with a
changed action vector would refresh the target only when the default-off
active-intent reproposal switch is enabled.
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from wind_prediction.ballast_planner import PlannerConfig  # noqa: E402
from wind_prediction.ballast_planner_provider import BallastPlannerPreviewProvider  # noqa: E402


class _ReplayStub:
    update_interval_s = 600


def _provider(enabled: bool) -> BallastPlannerPreviewProvider:
    return BallastPlannerPreviewProvider(
        replay_dataset=_ReplayStub(),
        start_timestamp=datetime(2024, 12, 19, 6, 0, 0),
        cfg=PlannerConfig(),
        block_discounts=[1.0, 0.85, 0.7],
        prediction_primary_enabled=True,
        primary_hold_target_mode="pause",
        event_reset_mode="action",
        active_intent_reproposal_enabled=enabled,
        active_intent_reproposal_min_delta_kg=1500.0,
    )


def main() -> None:
    log_path = REPO / (
        "outputs/wind_prediction/postureaware_mainline_2case_learned_2h_v2/"
        "planner_logs/01_fr_relief_09_2024-12-19_060000_"
        "prediction_primary_econ_planner_log.csv"
    )
    ts_path = REPO / (
        "outputs/wind_prediction/postureaware_mainline_2case_learned_2h_v2/"
        "timeseries/01_fr_relief_09_2024-12-19_060000_"
        "prediction_primary_econ_timeseries.csv"
    )
    log = pd.read_csv(log_path)
    ts = pd.read_csv(ts_path)

    rows = []
    for enabled in (False, True):
        provider = _provider(enabled)
        prev_action = "active_small"
        provider._target_action = prev_action
        provider._primary_target_initialized = True
        provider._primary_target_kg = log.loc[log["bucket"] == 7, [
            "prediction_primary_target_t1_kg",
            "prediction_primary_target_t2_kg",
            "prediction_primary_target_t3_kg",
        ]].iloc[0].to_numpy(dtype=float)
        provider._primary_anchor_masses_kg = provider._primary_target_kg.copy()

        for bucket_id in (8, 9):
            plan = log.loc[log["bucket"] == bucket_id].iloc[0]
            start = float(plan["current_time_s"])
            first = ts[(ts["t_s"] >= start) & (ts["t_s"] < start + 600)].iloc[0]
            masses = np.array(
                [
                    float(first["tank1_kg"]),
                    float(first["tank2_kg"]),
                    float(first["tank3_kg"]),
                ],
                dtype=float,
            )
            plant_info = {"tank_masses": masses}
            avec = np.array(
                [
                    float(plan["planner_action_pitch_deg"]),
                    float(plan["planner_action_roll_deg"]),
                ],
                dtype=float,
            )
            event_reset = False
            needed, reason = provider._active_intent_reproposal_needed(
                plant_info,
                avec,
                "active_small",
            )
            if event_reset or not provider._primary_target_initialized:
                branch = "update_event_reset"
                provider._update_primary_target(plant_info, avec, start)
            elif needed:
                branch = "update_reproposal"
                provider._update_primary_target(plant_info, avec, start)
            else:
                branch = "reuse"
                provider._reuse_primary_target()
            rows.append(
                {
                    "enabled": int(enabled),
                    "bucket": bucket_id,
                    "needed": int(needed),
                    "reason": reason,
                    "branch": branch,
                    "delta_mean_kg": provider._active_intent_reproposal_delta_mean_kg,
                    "target_refreshed": int(provider._primary_target_refreshed),
                    "target_reused": int(provider._primary_target_reused),
                    "target_t1_kg": provider._primary_target_kg[0],
                    "target_t2_kg": provider._primary_target_kg[1],
                    "target_t3_kg": provider._primary_target_kg[2],
                }
            )

    out = pd.DataFrame(rows)
    print(out.to_csv(index=False))


if __name__ == "__main__":
    main()
