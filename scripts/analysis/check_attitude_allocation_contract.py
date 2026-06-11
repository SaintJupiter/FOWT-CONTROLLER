#!/usr/bin/env python3
"""Check the pitch/roll-to-tank allocation sign contract.

This is a structural check, not a controller tuning experiment. The closed-loop
MIMO controller uses the physical attitude allocation matrix: positive pitch
mass pattern produces positive physical pitch moment, and positive roll mass
pattern produces positive physical roll moment. The preview planner uses the
opposite matrix because its action vector is a compensation direction.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "archive" / "legacy_fowt_control"))

from controllers import build_attitude_allocation_matrix  # noqa: E402
from wind_prediction.ballast_planner import (  # noqa: E402
    COMPENSATION_ATTITUDE_ALLOC,
    PHYSICAL_ATTITUDE_ALLOC,
    PlannerConfig,
    action_vec,
    tank_signal,
)


def _canonical_tank_pos() -> np.ndarray:
    r = 46.2
    return np.array(
        [
            [r, 0.0, -10.0],
            [-r / 2.0, r * 0.866, -10.0],
            [-r / 2.0, -r * 0.866, -10.0],
        ],
        dtype=float,
    )


def main() -> int:
    cfg = PlannerConfig()
    controller_alloc = build_attitude_allocation_matrix(_canonical_tank_pos())
    planner_basis = np.column_stack(
        [
            tank_signal(np.array([cfg.deadband_pitch_deg, 0.0]), cfg),
            tank_signal(np.array([0.0, cfg.deadband_roll_deg]), cfg),
        ]
    )

    checks = [
        ("controller_matches_physical", controller_alloc, PHYSICAL_ATTITUDE_ALLOC),
        ("planner_basis_matches_compensation", planner_basis, COMPENSATION_ATTITUDE_ALLOC),
        ("compensation_is_negative_physical", COMPENSATION_ATTITUDE_ALLOC, -PHYSICAL_ATTITUDE_ALLOC),
        ("planner_is_negative_controller", planner_basis, -controller_alloc),
    ]
    static_response = PHYSICAL_ATTITUDE_ALLOC.T @ COMPENSATION_ATTITUDE_ALLOC
    expected_static_response = np.array([[-1.5, 0.0], [0.0, -2.0]], dtype=float)
    checks.append(
        (
            "pure_axis_compensation_has_no_static_cross_coupling",
            static_response,
            expected_static_response,
        )
    )

    pitch_only_vec = action_vec(
        "active_small",
        np.array([4.0 * cfg.deadband_pitch_deg, 0.5 * cfg.deadband_roll_deg]),
        cfg,
    )
    pitch_only_response = PHYSICAL_ATTITUDE_ALLOC.T @ tank_signal(pitch_only_vec, cfg)
    checks.append(
        (
            "small_roll_inside_deadband_is_not_carried_by_pitch_action",
            pitch_only_response[1:2],
            np.zeros(1, dtype=float),
        )
    )

    stale_roll_vec = action_vec(
        "pump_saving",
        np.array([4.0 * cfg.deadband_pitch_deg, 0.5 * cfg.deadband_roll_deg]),
        cfg,
        previous_vec=np.array([0.0, cfg.deadband_roll_deg], dtype=float),
    )
    stale_roll_response = PHYSICAL_ATTITUDE_ALLOC.T @ tank_signal(stale_roll_vec, cfg)
    checks.append(
        (
            "pump_saving_previous_roll_does_not_leak_inside_deadband",
            stale_roll_response[1:2],
            np.zeros(1, dtype=float),
        )
    )
    ok = True
    for name, actual, expected in checks:
        passed = bool(np.allclose(actual, expected, atol=1e-9))
        print(f"{name}: {'PASS' if passed else 'FAIL'}")
        if not passed:
            print("actual=")
            print(actual)
            print("expected=")
            print(expected)
            ok = False

    pitch_pattern = planner_basis[:, 0]
    roll_pattern = planner_basis[:, 1]
    print("planner_pitch_pattern=", pitch_pattern.tolist())
    print("planner_roll_pattern=", roll_pattern.tolist())
    print("controller_physical_alloc=")
    print(controller_alloc)
    print("physical_response_to_planner_basis=")
    print(static_response)
    print("pitch_only_action_vec=", pitch_only_vec.tolist())
    print("pitch_only_physical_response=", pitch_only_response.tolist())
    print("stale_roll_action_vec=", stale_roll_vec.tolist())
    print("stale_roll_physical_response=", stale_roll_response.tolist())
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
