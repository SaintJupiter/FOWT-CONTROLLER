import unittest

import numpy as np

from wind_prediction.controller_chain_runner import run_controller_plant_chain
from wind_prediction.controller_core import ControlCoreConfig
from wind_prediction.controller_plant_adapter import CompactControllerPlantAdapter
from wind_prediction.execution_rollout import ExecutionRolloutConfig
from wind_prediction.forecast_action_policy import ForecastActionPolicyConfig


def _fake_platform_identity(*, status="runtime", purpose="control"):
    return {
        "schema_version": "floating_platform_identity.v1",
        "profile": {
            "requested_name": "fake_platform",
            "base_name": "fake_platform",
            "status": status,
            "purpose": purpose,
        },
        "model_modes": {"dynamics": "fake"},
        "constants": {"seawater_density_kg_m3": 1025.0},
        "structure": {"dry_mass_kg": 7_300_000.0},
        "ballast_system": {
            "tank_count": 3,
            "tank_capacity_kg": 1_896_250.0,
            "pump_rate_schedule_m3_min": [
                [0.0, 0.0],
                [200.0, 4.0],
                [300.0, 6.0],
                [500.0, 8.0],
                [700.0, 10.0],
                [1000.0, 12.0],
                [2000.0, 14.0],
                [3000.0, 15.0],
            ],
            "pump_stop_error_kg": 300.0,
            "pump_restart_error_kg": 500.0,
            "pump_minimum_on_s": 20.0,
            "pump_minimum_off_s": 12.0,
            "pump_hold_before_stop_s": 10.0,
            "pump_ramp_up_m3_min_per_s": 2.0,
            "pump_ramp_down_m3_min_per_s": 3.0,
        },
        "hydrodynamics": {"model": "fake"},
        "mooring": {"model": "fake"},
        "reference_mass_properties": {"total_mass_kg": 10_000_000.0},
    }


class FakePlant:
    def __init__(self):
        self.state = np.zeros(12, dtype=float)
        self.current_ballast_mass = np.full(3, 900_000.0)
        self.target_ballast_mass = self.current_ballast_mass.copy()
        self.tank_capacity = 1_896_250.0
        self._latched = np.zeros(3, dtype=bool)

    @staticmethod
    def resolved_platform_identity():
        return _fake_platform_identity()

    def set_ballast_target(self, *masses_kg):
        self.target_ballast_mass = np.asarray(masses_kg, dtype=float)

    def step(self, thrust_n, wind_direction_deg, dt_s, current_time_s):
        del wind_direction_deg, current_time_s
        error = self.target_ballast_mass - self.current_ballast_mass
        change = np.clip(error, -100.0 * dt_s, 100.0 * dt_s)
        self.current_ballast_mass += change
        self._latched = np.abs(error) > 1.0
        self.state[4] = float(thrust_n) * 1e-9
        rates = change * 60.0 / (1025.0 * dt_s)
        info = {
            "tank_masses": self.current_ballast_mass.copy(),
            "target_ballast_mass": self.target_ballast_mass.copy(),
            "pump_net_rate_m3_min": rates,
            "pump_rate_smoothed_m3_min": np.abs(rates),
            "pump_latched": self._latched.copy(),
        }
        return self.state, info


class FakeMassTelemetryPlant(FakePlant):
    class _MassProperties:
        total_mass_kg = 10_000_000.0
        center_of_mass_m = np.array([0.0, 0.0, -10.0])
        inertia_about_reference_kg_m2 = np.diag([1.0e9, 1.1e9, 1.2e9])

    def __init__(self):
        super().__init__()
        self.mass_properties = self._MassProperties()
        self.target_ballast_mass = self.current_ballast_mass + np.array(
            [2000.0, 0.0, 0.0]
        )

    def step(self, thrust_n, wind_direction_deg, dt_s, current_time_s):
        state, info = super().step(thrust_n, wind_direction_deg, dt_s, current_time_s)
        delta = float(np.sum(self.current_ballast_mass) - 2_700_000.0)
        total_mass = 10_000_000.0 + delta
        center = np.array([delta * 1e-8, 0.0, -10.0])
        inertia = np.diag([1.0e9 + delta, 1.1e9, 1.2e9])
        effective = np.diag([total_mass] * 3 + np.diag(inertia).tolist())
        info.update(
            {
                "platform_total_mass_kg": total_mass,
                "platform_center_of_mass_m": center,
                "platform_inertia_about_reference_kg_m2": inertia,
                "platform_effective_mass_matrix": effective,
            }
        )
        return state, info


class AuditOnlyFakePlant(FakePlant):
    @staticmethod
    def resolved_platform_identity():
        return _fake_platform_identity(
            status="audit_only_not_for_control_validation",
            purpose="audit",
        )


class FrameworkSmokeFakePlant(FakePlant):
    @staticmethod
    def resolved_platform_identity():
        return _fake_platform_identity(
            status="framework_only_not_for_performance_validation",
            purpose="framework_smoke",
        )


class ShallowIdentityFakePlant(FakePlant):
    @staticmethod
    def resolved_platform_identity():
        return {
            "schema_version": "floating_platform_identity.v1",
            "profile": {
                "requested_name": "shallow",
                "base_name": "shallow",
                "status": "runtime",
                "purpose": "control",
            },
        }


class UnidentifiedFakePlant(FakePlant):
    resolved_platform_identity = None


class DensityMismatchFakePlant(FakePlant):
    @staticmethod
    def resolved_platform_identity():
        identity = _fake_platform_identity()
        identity["constants"]["seawater_density_kg_m3"] = 1030.0
        return identity


def compact_test_config(*, target_slew_enabled=False):
    execution = ExecutionRolloutConfig(
        block_duration_s=20.0,
        internal_step_s=1.0,
        target_slew_enabled=target_slew_enabled,
    )
    policy = ForecastActionPolicyConfig(
        enabled=True,
        stage_duration_s=20.0,
    )
    return ControlCoreConfig(
        stage_duration_s=20.0,
        stage_count=3,
        execution=execution,
        forecast_policy=policy,
    )


class ControllerChainRunnerTests(unittest.TestCase):
    def test_connected_adapter_rejects_target_slew_until_command_path_owns_it(self):
        plant = FakePlant()
        with self.assertRaisesRegex(ValueError, "does not support target slew"):
            CompactControllerPlantAdapter(
                initial_tank_masses_kg=plant.current_ballast_mass,
                forecast_source=None,
                config=compact_test_config(target_slew_enabled=True),
            )

    def test_feedback_decision_target_execution_and_feedback_run_in_order(self):
        plant = FakePlant()
        adapter = CompactControllerPlantAdapter(
            initial_tank_masses_kg=plant.current_ballast_mass,
            forecast_source=None,
            config=compact_test_config(),
        )
        result = run_controller_plant_chain(
            plant=plant,
            controller=adapter,
            wind_trace={"ws": np.full(40, 10.0), "wd": np.zeros(40)},
            thrust_model=lambda speed: speed * 1000.0,
            duration_s=40.0,
            dt_s=1.0,
            sample_interval_s=10.0,
        )

        self.assertTrue(result.summary.completed)
        self.assertTrue(result.summary.finite_state)
        self.assertTrue(result.summary.capacity_respected)
        self.assertEqual(result.summary.decision_count, 2)
        self.assertEqual(result.summary.forecast_decision_count, 0)
        self.assertEqual(result.summary.max_command_transfer_error_kg, 0.0)
        self.assertEqual(result.summary.platform_profile_requested_name, "fake_platform")
        self.assertEqual(result.summary.platform_model_status, "runtime")
        self.assertEqual(result.summary.platform_run_purpose, "control")
        self.assertGreaterEqual(len(result.sampled_timeseries), 4)
        self.assertEqual(len(result.decisions), 2)
        self.assertIn("stages", result.decisions[0]["trace"])
        self.assertFalse(result.summary.platform_mass_telemetry_available)

    def test_optional_platform_mass_telemetry_is_recorded(self):
        plant = FakeMassTelemetryPlant()
        plant.state[4] = np.radians(3.0)
        adapter = CompactControllerPlantAdapter(
            initial_tank_masses_kg=plant.current_ballast_mass,
            forecast_source=None,
            config=compact_test_config(),
        )
        result = run_controller_plant_chain(
            plant=plant,
            controller=adapter,
            wind_trace={"ws": np.full(40, 10.0), "wd": np.zeros(40)},
            thrust_model=lambda speed: speed * 1000.0,
            duration_s=40.0,
            dt_s=1.0,
            sample_interval_s=10.0,
        )

        self.assertTrue(result.summary.platform_mass_telemetry_available)
        self.assertEqual(result.summary.initial_platform_total_mass_kg, 10_000_000.0)
        self.assertGreater(result.summary.max_abs_tank_mass_change_kg, 0.0)
        self.assertGreater(result.summary.max_abs_platform_mass_change_kg, 0.0)
        self.assertEqual(result.summary.max_platform_mass_balance_error_kg, 0.0)
        self.assertIn("platform_effective_mass_matrix", result.sampled_timeseries[0])

    def test_invalid_wind_trace_is_rejected_before_running(self):
        plant = FakePlant()
        adapter = CompactControllerPlantAdapter(
            initial_tank_masses_kg=plant.current_ballast_mass,
            forecast_source=None,
            config=compact_test_config(),
        )
        with self.assertRaisesRegex(ValueError, "shorter"):
            run_controller_plant_chain(
                plant=plant,
                controller=adapter,
                wind_trace={"ws": [10.0], "wd": [0.0]},
                thrust_model=lambda speed: speed,
                duration_s=10.0,
                dt_s=1.0,
            )

    def test_audit_only_plant_is_rejected_by_control_runner(self):
        plant = AuditOnlyFakePlant()
        adapter = CompactControllerPlantAdapter(
            initial_tank_masses_kg=plant.current_ballast_mass,
            forecast_source=None,
            config=compact_test_config(),
        )
        with self.assertRaisesRegex(ValueError, "audit-only"):
            run_controller_plant_chain(
                plant=plant,
                controller=adapter,
                wind_trace={"ws": np.full(10, 10.0), "wd": np.zeros(10)},
                thrust_model=lambda speed: speed,
                duration_s=10.0,
                dt_s=1.0,
            )

    def test_framework_smoke_summary_retains_platform_use_identity(self):
        plant = FrameworkSmokeFakePlant()
        adapter = CompactControllerPlantAdapter(
            initial_tank_masses_kg=plant.current_ballast_mass,
            forecast_source=None,
            config=compact_test_config(),
        )
        result = run_controller_plant_chain(
            plant=plant,
            controller=adapter,
            wind_trace={"ws": np.full(10, 10.0), "wd": np.zeros(10)},
            thrust_model=lambda speed: speed,
            duration_s=10.0,
            dt_s=1.0,
        )

        self.assertEqual(
            result.summary.platform_model_status,
            "framework_only_not_for_performance_validation",
        )
        self.assertEqual(result.summary.platform_run_purpose, "framework_smoke")
        self.assertEqual(
            result.summary.to_dict()["platform_model_status"],
            "framework_only_not_for_performance_validation",
        )

    def test_unidentified_plant_is_rejected_by_control_runner(self):
        plant = UnidentifiedFakePlant()
        adapter = CompactControllerPlantAdapter(
            initial_tank_masses_kg=plant.current_ballast_mass,
            forecast_source=None,
            config=compact_test_config(),
        )
        with self.assertRaisesRegex(ValueError, "resolved platform identity"):
            run_controller_plant_chain(
                plant=plant,
                controller=adapter,
                wind_trace={"ws": np.full(10, 10.0), "wd": np.zeros(10)},
                thrust_model=lambda speed: speed,
                duration_s=10.0,
                dt_s=1.0,
            )

    def test_semantically_shallow_platform_identity_is_rejected(self):
        plant = ShallowIdentityFakePlant()
        adapter = CompactControllerPlantAdapter(
            initial_tank_masses_kg=plant.current_ballast_mass,
            forecast_source=None,
            config=compact_test_config(),
        )
        with self.assertRaisesRegex(ValueError, "non-empty 'model_modes'"):
            run_controller_plant_chain(
                plant=plant,
                controller=adapter,
                wind_trace={"ws": np.full(10, 10.0), "wd": np.zeros(10)},
                thrust_model=lambda speed: speed,
                duration_s=10.0,
                dt_s=1.0,
            )

    def test_execution_configuration_must_match_platform_identity(self):
        plant = DensityMismatchFakePlant()
        adapter = CompactControllerPlantAdapter(
            initial_tank_masses_kg=plant.current_ballast_mass,
            forecast_source=None,
            config=compact_test_config(),
        )
        with self.assertRaisesRegex(ValueError, "seawater_density_kg_m3"):
            run_controller_plant_chain(
                plant=plant,
                controller=adapter,
                wind_trace={"ws": np.full(10, 10.0), "wd": np.zeros(10)},
                thrust_model=lambda speed: speed,
                duration_s=10.0,
                dt_s=1.0,
            )


if __name__ == "__main__":
    unittest.main()
