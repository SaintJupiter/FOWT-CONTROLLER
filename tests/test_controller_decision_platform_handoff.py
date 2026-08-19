import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

from fowt_platform import (
    IncrementalLoads,
    IncrementalPlatformModel,
    IncrementalState,
    assemble_ballast_model_snapshot,
    load_volturnus_reference_components,
)
from wind_prediction.controller_core import (
    ControlAction,
    ControlCoreConfig,
    ControlObservation,
    decide_control_cycle,
)
from wind_prediction.execution_rollout import (
    ExecutionRolloutConfig,
    ExecutionRolloutState,
    ExecutionTargetOperation,
    simulate_execution_step,
)
from wind_prediction.forecast_action_policy import ForecastActionPolicyConfig
from wind_prediction.forecast_evidence import ForecastEvidence


ROOT = Path(__file__).resolve().parents[1]
REFERENCE_MANIFEST = (
    ROOT
    / "configs"
    / "reference_platforms"
    / "volturnus_s_openfast_v1_1_16.json"
)
REFERENCE_TANK_MASSES_KG = np.array([1_108_000.0, 1_362_000.0, 1_362_000.0])
TANK_CAPACITY_KG = 1_896_250.0
TANK_COORDINATES_M = np.array(
    [
        [46.2, 0.0, -10.0],
        [-23.1, 46.2 * 0.866, -10.0],
        [-23.1, -46.2 * 0.866, -10.0],
    ]
)


def _controller_config() -> ControlCoreConfig:
    execution = ExecutionRolloutConfig(
        block_duration_s=1200.0,
        internal_step_s=1.0,
        water_density_kg_m3=1025.0,
        max_pump_rate_m3_min=1.0,
        target_slew_enabled=False,
        stop_error_kg=0.0,
        restart_error_kg=0.0,
        min_on_s=0.0,
        min_off_s=0.0,
        near_target_hold_s=0.0,
        ramp_up_m3_min_per_s=np.inf,
        ramp_down_m3_min_per_s=np.inf,
        pump_rate_schedule_m3_min=((0.0, 1.0), (20_000.0, 1.0)),
        tank_capacity_kg=TANK_CAPACITY_KG,
    )
    return ControlCoreConfig(
        stage_duration_s=1200.0,
        stage_count=3,
        execution=execution,
        posture_priority_envelope_deg=(3.0, 3.0),
        w_residual=80.0,
        w_terminal_residual=80.0,
        w_pump_volume=0.2,
        forecast_policy=ForecastActionPolicyConfig(
            enabled=True,
            stage_duration_s=1200.0,
            high_impact_reliability_min=0.65,
            high_impact_event_probability_min=0.60,
        ),
    )


def _forecast() -> ForecastEvidence:
    return ForecastEvidence(
        source="handoff_test",
        model_version="test",
        origin_time="2026-08-19T00:00:00",
        sample_period_s=600.0,
        uv_ms=np.array([(0.0, -8.0)] * 6, dtype=float),
        lead_reliability=np.full(6, 0.9),
        event_probs={
            "attention_event_0_20m": 0.9,
            "attention_event_20_40m": 0.9,
            "attention_event_40_60m": 0.9,
        },
        provides_future_preview=True,
    )


class ControllerDecisionPlatformHandoffTests(unittest.TestCase):
    def test_decision_request_drives_short_execution_without_leaking_preview_state(self):
        config = _controller_config()
        execution_state = ExecutionRolloutState(
            masses_kg=REFERENCE_TANK_MASSES_KG,
            target_masses_kg=REFERENCE_TANK_MASSES_KG,
            primary_target_kg=REFERENCE_TANK_MASSES_KG,
            pump_rates_m3_min=np.zeros(3),
            pump_latched=np.zeros(3, dtype=bool),
        )
        observation = ControlObservation(
            time_s=0.0,
            posture_deg=(4.5, 0.0),
            posture_rate_deg_s=(0.0, 0.0),
            current_wind_uv_ms=(0.0, -8.0),
            execution_state=execution_state,
        )

        decision = decide_control_cycle(observation, _forecast(), config)

        self.assertIn(
            decision.action,
            {ControlAction.NORMAL, ControlAction.STRENGTHEN},
        )
        self.assertEqual(
            decision.execution_request.operation,
            ExecutionTargetOperation.TRACK,
        )
        np.testing.assert_allclose(
            decision.execution_request.target_masses_kg,
            decision.target_masses_kg,
        )

        short_execution = simulate_execution_step(
            execution_state,
            decision.execution_request,
            replace(config.execution, block_duration_s=1.0),
        )
        self.assertLess(
            np.linalg.norm(short_execution.mass_delta_kg),
            np.linalg.norm(decision.first_execution.mass_delta_kg),
        )

        reference = load_volturnus_reference_components(REFERENCE_MANIFEST)
        damping = np.zeros((6, 6))
        zero_loads = lambda snapshot: IncrementalLoads(
            wind=np.zeros(6),
            wave=np.zeros(6),
            ballast=snapshot.incremental_ballast_load,
            other=np.zeros(6),
        )
        snapshot_n = assemble_ballast_model_snapshot(
            reference=reference,
            damping=damping,
            actual_tank_masses_kg=execution_state.actual_masses_kg,
            reference_tank_masses_kg=REFERENCE_TANK_MASSES_KG,
            tank_capacities_kg=np.full(3, TANK_CAPACITY_KG),
            tank_coordinates_m=TANK_COORDINATES_M,
        )
        state_n1 = IncrementalPlatformModel(snapshot_n.matrices).advance_frozen_step(
            IncrementalState.zeros(), zero_loads(snapshot_n), duration_s=1.0
        )
        snapshot_n1 = assemble_ballast_model_snapshot(
            reference=reference,
            damping=damping,
            actual_tank_masses_kg=short_execution.state.actual_masses_kg,
            reference_tank_masses_kg=REFERENCE_TANK_MASSES_KG,
            tank_capacities_kg=np.full(3, TANK_CAPACITY_KG),
            tank_coordinates_m=TANK_COORDINATES_M,
        )
        state_n2 = IncrementalPlatformModel(snapshot_n1.matrices).advance_frozen_step(
            state_n1, zero_loads(snapshot_n1), duration_s=1.0
        )

        np.testing.assert_allclose(state_n1.position, np.zeros(6))
        np.testing.assert_allclose(state_n1.velocity, np.zeros(6))
        self.assertGreater(np.linalg.norm(state_n2.velocity[3:5]), 0.0)


if __name__ == "__main__":
    unittest.main()
