import csv
import json
import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
LEGACY_DIR = ROOT / "archive" / "legacy_fowt_control"
sys.path.insert(0, str(LEGACY_DIR))

from defaults import (  # noqa: E402
    DEFAULT_MAIN_EXPERIMENT_PROTOCOL,
    DEFAULT_PUMP_CFG,
    DEFAULT_TARGET_SHAPE_CFG,
)
from target_execution import TargetSlewLimiter  # noqa: E402


EXPECTED_PUMP_CFG = {
    "pump_stop_err_kg": 300.0,
    "pump_restart_err_kg": 500.0,
    "pump_min_on_s": 20.0,
    "pump_min_off_s": 12.0,
    "pump_hold_before_stop_s": 10.0,
    "pump_global_quiet_backlog_kg": float("inf"),
    "pump_target_quiet_rate_kg_s": float("inf"),
    "pump_global_quiet_hold_s": 0.0,
    "pump_ramp_up_m3_min_per_s": 2.0,
    "pump_ramp_down_m3_min_per_s": 3.0,
}

EXPECTED_RATE_SCHEDULE = [
    {"error_kg": 0.0, "rate_m3_min": 0.0},
    {"error_kg": 200.0, "rate_m3_min": 4.0},
    {"error_kg": 300.0, "rate_m3_min": 6.0},
    {"error_kg": 500.0, "rate_m3_min": 8.0},
    {"error_kg": 700.0, "rate_m3_min": 10.0},
    {"error_kg": 1000.0, "rate_m3_min": 12.0},
    {"error_kg": 2000.0, "rate_m3_min": 14.0},
    {"error_kg": 3000.0, "rate_m3_min": 15.0},
]


class FrozenHPhysicalManifestTests(unittest.TestCase):
    def test_all_150_frozen_cases_use_matched_engineered_minimal_pumps(self):
        protocol_dir = (
            ROOT / "paper_snapshot" / "2026-08-11" / "evidence" / "run_protocols"
        )
        protocol_paths = sorted(protocol_dir.glob("*_run_protocol.json"))
        self.assertEqual(
            len(protocol_paths),
            15,
            "the frozen 150-case evidence must contain 15 ten-case protocols",
        )

        completed_cases = 0
        case_sources = set()
        for path in protocol_paths:
            protocol = json.loads(path.read_text(encoding="utf-8"))
            with self.subTest(protocol=path.name):
                self.assertEqual(protocol["schema_version"], "run_protocol.v1")
                self.assertEqual(protocol["inputs"]["duration_s"], 21600.0)
                self.assertEqual(
                    protocol["configuration"]["closed_pump_profile"],
                    "engineered_minimal",
                )
                self.assertEqual(
                    protocol["configuration"]["primary_pump_profile"],
                    "engineered_minimal",
                )
                self.assertEqual(protocol["result"]["completed_cases"], 10)
                self.assertEqual(protocol["result"]["issue_count"], 0)
            completed_cases += int(protocol["result"]["completed_cases"])
            case_sources.add(protocol["identity"]["cases_source"])

        self.assertEqual(completed_cases, 150)
        self.assertEqual(len(case_sources), 15)

        summary_path = (
            ROOT
            / "paper_snapshot"
            / "2026-08-11"
            / "evidence"
            / "full_prediction"
            / "combined_casebook_summary.csv"
        )
        with summary_path.open(newline="", encoding="utf-8-sig") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 150)
        self.assertEqual(
            {row["closed_pump_profile"] for row in rows},
            {"engineered_minimal"},
        )
        self.assertEqual(
            {row["primary_pump_profile"] for row in rows},
            {"engineered_minimal"},
        )

    def test_frozen_pump_and_per_tank_target_slew_parameters_are_locked(self):
        for key, expected in EXPECTED_PUMP_CFG.items():
            with self.subTest(parameter=key):
                actual = DEFAULT_PUMP_CFG[key]
                if np.isinf(expected):
                    self.assertTrue(np.isinf(actual))
                else:
                    self.assertEqual(actual, expected)

        for target_shape in (
            DEFAULT_TARGET_SHAPE_CFG,
            DEFAULT_MAIN_EXPERIMENT_PROTOCOL["target_shape_cfg"],
        ):
            self.assertEqual(target_shape["actuator_authority"], "plant")
            self.assertTrue(target_shape["enable_target_slew_limit"])
            self.assertEqual(target_shape["target_slew_rate_m3_min"], 10.0)

        limiter = TargetSlewLimiter(
            dt=1.0,
            rho=1025.0,
            max_capacity=1850.0 * 1025.0,
            target_slew_rate_m3_min=10.0,
            enabled=True,
        )
        initial = np.array([900000.0, 1000000.0, 1100000.0])
        limiter.reset(initial)
        applied, _ = limiter.update(initial + 100000.0)
        expected_per_tank_delta = 10.0 * 1025.0 / 60.0
        np.testing.assert_allclose(
            applied - initial,
            np.full(3, expected_per_tank_delta),
            rtol=0.0,
            atol=1e-10,
            err_msg="10 m3/min target slew must be applied independently to each tank",
        )

    def test_execution_rollout_manifests_match_the_frozen_actuator(self):
        for config_name in (
            "planner_runtime_v2_execution_only.json",
            "planner_runtime_v2_forecast_execution.json",
        ):
            path = ROOT / "configs" / config_name
            runtime = json.loads(path.read_text(encoding="utf-8"))
            rollout = runtime["execution_rollout"]
            with self.subTest(config=config_name):
                self.assertTrue(rollout["enabled"])
                self.assertEqual(rollout["internal_step_s"], 1.0)
                self.assertEqual(rollout["water_density_kg_m3"], 1025.0)
                self.assertTrue(rollout["target_slew_enabled"])
                self.assertEqual(rollout["target_slew_rate_m3_min"], 10.0)
                self.assertEqual(rollout["stop_error_kg"], 300.0)
                self.assertEqual(rollout["restart_error_kg"], 500.0)
                self.assertEqual(rollout["min_on_s"], 20.0)
                self.assertEqual(rollout["min_off_s"], 12.0)
                self.assertEqual(rollout["near_target_hold_s"], 10.0)
                self.assertEqual(rollout["ramp_up_m3_min_per_s"], 2.0)
                self.assertEqual(rollout["ramp_down_m3_min_per_s"], 3.0)
                self.assertEqual(rollout["max_pump_rate_m3_min"], 15.0)
                self.assertEqual(rollout["tank_capacity_kg"], 1850.0 * 1025.0)
                self.assertEqual(
                    rollout["pump_rate_schedule_m3_min"],
                    EXPECTED_RATE_SCHEDULE,
                )


if __name__ == "__main__":
    unittest.main()
