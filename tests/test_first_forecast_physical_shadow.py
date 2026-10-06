import importlib.util
import math
from pathlib import Path
import sys
import unittest

import numpy as np

from fowt_platform import BallastModelSnapshot, IncrementalState, PlatformMatrices
from wind_prediction.controller_core import (
    ControlCoreConfig,
    ControlObservation,
    decide_control_cycle,
)
from wind_prediction.execution_rollout import (
    ExecutionRolloutConfig,
    ExecutionRolloutState,
)
from wind_prediction.forecast_action_policy import ForecastActionPolicyConfig
from wind_prediction.forecast_evidence import ForecastEvidence
from wind_prediction.forecast_physical_load import (
    ForecastRotorLoadParameters,
    ForecastRotorOperatingState,
)


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "validation" / "trace_first_forecast_physical_shadow.py"
SPEC = importlib.util.spec_from_file_location("first_forecast_physical_shadow", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _execution_config() -> ExecutionRolloutConfig:
    return ExecutionRolloutConfig(
        block_duration_s=1_200.0,
        water_density_kg_m3=1_000.0,
        max_pump_rate_m3_min=1.0,
        tank_capacity_kg=1.0e9,
        internal_step_s=30.0,
        stop_error_kg=0.0,
        restart_error_kg=0.0,
        min_on_s=0.0,
        min_off_s=0.0,
        near_target_hold_s=0.0,
        ramp_up_m3_min_per_s=np.inf,
        ramp_down_m3_min_per_s=np.inf,
        pump_rate_schedule_m3_min=((0.0, 1.0), (1.0e9, 1.0)),
    )


def _controller_config() -> ControlCoreConfig:
    return ControlCoreConfig(
        stage_duration_s=1_200.0,
        stage_count=1,
        stage_discounts=(1.0,),
        execution=_execution_config(),
        forecast_policy=ForecastActionPolicyConfig(
            enabled=False,
            stage_duration_s=1_200.0,
        ),
    )


def _evidence(*, origin_time="2026-08-22T00:00:00") -> ForecastEvidence:
    return ForecastEvidence(
        source="shadow_forecast",
        model_version="shadow_model",
        origin_time=origin_time,
        sample_period_s=600.0,
        uv_ms=np.array([[0.0, -10.0], [0.0, -12.0]], dtype=float),
        lead_reliability=np.array([0.9, 0.8]),
        event_probs={},
    )


def _observation() -> ControlObservation:
    actual = np.full(3, 5.0e8)
    execution = ExecutionRolloutState(
        masses_kg=actual,
        target_masses_kg=actual,
        primary_target_kg=actual,
        pump_rates_m3_min=[0.0, 0.0, 0.0],
        pump_latched=[False, False, False],
        pump_off_elapsed_s=[1_000.0, 1_000.0, 1_000.0],
    )
    return ControlObservation(
        time_s=0.0,
        posture_deg=(0.0, 0.0),
        posture_rate_deg_s=(0.0, 0.0),
        current_wind_uv_ms=(0.0, -8.0),
        execution_state=execution,
    )


def _snapshot(
    actual: np.ndarray,
    *,
    tank_capacities_kg: np.ndarray | None = None,
) -> BallastModelSnapshot:
    restoring = np.eye(6)
    restoring[4, 4] = 1.0e10
    restoring[3, 3] = 1.0e10
    matrices = PlatformMatrices(
        mass=np.eye(6),
        damping=np.zeros((6, 6)),
        hydrostatic_stiffness=restoring,
        mooring_stiffness=np.zeros((6, 6)),
    )
    return BallastModelSnapshot(
        actual_tank_masses_kg=actual,
        reference_tank_masses_kg=actual,
        tank_mass_deltas_kg=np.zeros(3),
        tank_capacities_kg=(
            np.full(3, 1.0e9)
            if tank_capacities_kg is None
            else tank_capacities_kg
        ),
        tank_coordinates_m=[
            [-20.0, 0.0, -1.0],
            [10.0, 17.0, -1.0],
            [10.0, -17.0, -1.0],
        ],
        gravity_m_s2=9.81,
        matrices=matrices,
        incremental_ballast_load=np.zeros(6),
        runtime_provenance="test_frozen_snapshot",
    )


def _rotor_parameters() -> ForecastRotorLoadParameters:
    return ForecastRotorLoadParameters(
        air_density_kg_m3=1.225,
        rotor_radius_m=100.0,
        wind_reference_height_m=100.0,
        rotor_plane_height_m=100.0,
        power_law_shear_exponent=0.0,
        frozen_equilibrium_heading_rad=np.pi,
        rotor_force_application_point_from_reference_m=[0.0, 0.0, 120.0],
        downwind_rotor_normal_platform=[1.0, 0.0, 0.0],
    )


def _rotor_operating_state() -> ForecastRotorOperatingState:
    return ForecastRotorOperatingState(
        current_thrust_coefficient=0.8,
        future_thrust_coefficients=[0.8, 0.8],
        current_platform_reference_velocity_platform_mps=[0.0, 0.0, 0.0],
        current_platform_angular_velocity_platform_radps=[0.0, 0.0, 0.0],
        future_platform_reference_velocities_platform_mps=np.zeros((2, 3)),
        future_platform_angular_velocities_platform_radps=np.zeros((2, 3)),
        future_platform_kinematics_mode="frozen_zero",
        future_platform_kinematics_source="shadow_trace_frozen_platform",
    )


def _trace(**overrides):
    observation = overrides.pop("observation", _observation())
    values = {
        "observation": observation,
        "evidence": _evidence(),
        "controller_config": _controller_config(),
        "current_wind_source": "same_origin_observation",
        "current_wind_observation_time": "2026-08-22T00:00:00",
        "rotor_load_parameters": _rotor_parameters(),
        "rotor_operating_state": _rotor_operating_state(),
        "platform_snapshot": _snapshot(observation.execution_state.actual_masses_kg),
        "platform_state": IncrementalState.zeros(),
    }
    values.update(overrides)
    return MODULE.trace_first_forecast_physical_shadow(**values)


class FirstForecastPhysicalShadowTests(unittest.TestCase):
    def test_traces_unmodified_legacy_decision_beside_the_plus_600_second_endpoint(self):
        observation = _observation()
        config = _controller_config()
        evidence = _evidence()
        expected = decide_control_cycle(observation, evidence, config)

        record = _trace(
            observation=observation,
            controller_config=config,
            evidence=evidence,
        )

        self.assertTrue(record.is_shadow_only)
        self.assertEqual(record.legacy_decision.action, expected.action)
        self.assertEqual(record.legacy_decision.target_operation, expected.target_operation)
        self.assertEqual(record.legacy_decision.score, expected.score)
        self.assertEqual(record.physical_preview.lead_index, 0)
        self.assertEqual(record.physical_preview.lead_time_s, 600.0)
        self.assertEqual(
            record.physical_preview.reachability.execution_duration_s,
            600.0,
        )
        self.assertEqual(record.legacy_decision.context.stages[0].lead_start_s, 600.0)
        self.assertEqual(record.legacy_decision.context.stages[0].lead_end_s, 1_200.0)
        self.assertNotEqual(
            record.physical_preview.lead_time_s,
            record.legacy_decision.context.stages[0].lead_end_s,
        )
        self.assertEqual(record.physical_preview.endpoint_sample.fraction, 1.0)
        self.assertIs(
            record.physical_execution_compensation.preview,
            record.physical_preview,
        )
        self.assertIs(
            record.current_posture_passive_load.platform_snapshot,
            record.physical_execution_compensation.platform_snapshot,
        )
        np.testing.assert_allclose(
            record.current_posture_passive_load.current_passive_pitch_roll_load_nm,
            [0.0, 0.0],
        )
        np.testing.assert_allclose(
            record.physical_execution_compensation.relative_load_residual_after_execution_nm,
            record.physical_execution_compensation.relative_external_pitch_roll_load_nm
            + record.physical_execution_compensation.executed_ballast_pitch_roll_moment_nm,
        )
        self.assertEqual(
            record.physical_preview.input_identity.forecast_source,
            "shadow_forecast",
        )
        np.testing.assert_allclose(
            record.observed_current_enu_downwind_air_velocity_mps,
            observation.current_wind_uv_ms,
        )
        self.assertEqual(record.platform_runtime_provenance, "test_frozen_snapshot")
        self.assertTrue(
            np.isclose(
                record.legacy_target_relation.physical_endpoint_net_mass_delta_kg,
                0.0,
            )
        )
        np.testing.assert_allclose(
            record.legacy_target_relation.proposed_target_tank_masses_kg,
            record.legacy_decision.target_masses_kg,
        )

    def test_does_not_mutate_the_observation_execution_state(self):
        base_observation = _observation()
        before = {
            "actual": base_observation.execution_state.actual_masses_kg.copy(),
            "target": base_observation.execution_state.target_masses_kg.copy(),
            "primary": base_observation.execution_state.primary_target_masses_kg.copy(),
            "rate": base_observation.execution_state.signed_flow_m3_min.copy(),
            "latched": base_observation.execution_state.pump_latched.copy(),
        }
        platform_state = IncrementalState(
            position=[0.0, 0.0, 0.0, 0.01, -0.02, 0.0],
            velocity=[0.0, 0.0, 0.0, 0.001, -0.002, 0.0],
        )
        state_before = {
            "position": platform_state.position.copy(),
            "velocity": platform_state.velocity.copy(),
        }
        observation = ControlObservation(
            time_s=base_observation.time_s,
            posture_deg=(
                math.degrees(platform_state.position[4]),
                math.degrees(platform_state.position[3]),
            ),
            posture_rate_deg_s=(
                math.degrees(platform_state.velocity[4]),
                math.degrees(platform_state.velocity[3]),
            ),
            current_wind_uv_ms=base_observation.current_wind_uv_ms,
            execution_state=base_observation.execution_state,
        )

        _trace(observation=observation, platform_state=platform_state)

        np.testing.assert_allclose(observation.execution_state.actual_masses_kg, before["actual"])
        np.testing.assert_allclose(observation.execution_state.target_masses_kg, before["target"])
        np.testing.assert_allclose(observation.execution_state.primary_target_masses_kg, before["primary"])
        np.testing.assert_allclose(observation.execution_state.signed_flow_m3_min, before["rate"])
        np.testing.assert_array_equal(observation.execution_state.pump_latched, before["latched"])
        np.testing.assert_allclose(platform_state.position, state_before["position"])
        np.testing.assert_allclose(platform_state.velocity, state_before["velocity"])

    def test_rejects_a_platform_state_that_describes_different_posture_than_observation(self):
        with self.assertRaisesRegex(ValueError, "platform_state pitch-roll posture"):
            _trace(
                platform_state=IncrementalState(
                    position=[0.0, 0.0, 0.0, 0.0, 0.01, 0.0],
                    velocity=np.zeros(6),
                )
            )

    def test_rejects_mismatched_snapshot_capacity_or_time_identity(self):
        observation = _observation()
        wrong_snapshot = _snapshot(observation.execution_state.actual_masses_kg + [1.0, 0.0, 0.0])
        with self.assertRaisesRegex(ValueError, "platform_snapshot actual"):
            _trace(observation=observation, platform_snapshot=wrong_snapshot)

        mismatched_capacity_snapshot = _snapshot(
            observation.execution_state.actual_masses_kg,
            tank_capacities_kg=np.array([1.0e9, 9.0e8, 1.0e9]),
        )
        with self.assertRaisesRegex(ValueError, "tank capacities"):
            _trace(
                observation=observation,
                platform_snapshot=mismatched_capacity_snapshot,
            )

        with self.assertRaisesRegex(ValueError, "must match forecast_origin_time"):
            _trace(current_wind_observation_time="2026-08-22T00:10:00")

    def test_rejects_missing_source_bound_forecast_time(self):
        with self.assertRaisesRegex(ValueError, "evidence.origin_time"):
            _trace(evidence=_evidence(origin_time=None))


if __name__ == "__main__":
    unittest.main()
