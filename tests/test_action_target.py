import unittest

import numpy as np

from wind_prediction.action_target import (
    ActionVectorSource,
    ControlAction,
    action_semantics_for,
    action_vector_for,
    execution_request_for_action,
)
from wind_prediction.controller_core import (
    ControlAction as CoreControlAction,
    ControlCoreConfig,
    _action_vector,
    execution_request_for_action as core_execution_request_for_action,
)
from wind_prediction.execution_rollout import (
    ExecutionRolloutConfig,
    ExecutionRolloutState,
    ExecutionTargetOperation,
)


def _state():
    masses = np.array([900_000.0, 900_000.0, 900_000.0])
    return ExecutionRolloutState(
        masses_kg=masses,
        target_masses_kg=masses + np.array([10_000.0, -5_000.0, -5_000.0]),
        primary_target_kg=masses + np.array([10_000.0, -5_000.0, -5_000.0]),
        pump_rates_m3_min=np.zeros(3),
        pump_latched=np.zeros(3, dtype=bool),
    )


def _config():
    return ControlCoreConfig(
        stage_duration_s=1200.0,
        execution=ExecutionRolloutConfig(
            block_duration_s=1200.0,
            internal_step_s=60.0,
            target_slew_enabled=False,
            tank_capacity_kg=1_896_250.0,
        ),
    )


class ActionTargetTests(unittest.TestCase):
    def test_controller_core_reexports_the_same_action_enum(self):
        self.assertIs(CoreControlAction, ControlAction)

    def test_direct_action_vector_matches_legacy_controller_wrapper(self):
        config = _config()
        demand = np.array([2.4, -1.2])
        previous = np.array([0.3, -0.5])

        direct = action_vector_for(
            ControlAction.STRENGTHEN,
            demand_deg=demand,
            previous_action_vector_deg=previous,
            demand_axis_scale_deg=config.resolved_legacy_demand_axis_scale_deg,
            reduced_ratio=config.reduced_ratio,
            normal_ratio=config.normal_ratio,
            strengthen_ratio=config.strengthen_ratio,
            reverse_ratio=config.reverse_ratio,
        )
        wrapped = _action_vector(ControlAction.STRENGTHEN, demand, previous, config)

        np.testing.assert_allclose(direct, wrapped)

    def test_action_semantics_keep_target_lifecycle_meaning(self):
        self.assertEqual(
            action_semantics_for(ControlAction.CONTINUE_TARGET).vector_source,
            ActionVectorSource.NONE,
        )
        self.assertEqual(
            action_semantics_for(ControlAction.REVERSE).vector_source,
            ActionVectorSource.PREVIOUS_ACTION_VECTOR,
        )

    def test_direct_execution_request_matches_core_compatibility_wrapper(self):
        config = _config()
        state = _state()
        action_vector = np.array([0.2, -0.1])

        direct = execution_request_for_action(
            ControlAction.NORMAL,
            state=state,
            action_vector_deg=action_vector,
            demand_axis_scale_deg=config.resolved_legacy_demand_axis_scale_deg,
            action_mass_quantum_kg=config.action_mass_quantum_kg,
            tank_capacity_kg=config.execution.tank_capacity_kg,
        )
        wrapped = core_execution_request_for_action(
            ControlAction.NORMAL,
            state=state,
            action_vector_deg=action_vector,
            config=config,
        )

        self.assertEqual(direct.operation, wrapped.operation)
        np.testing.assert_allclose(direct.target_masses_kg, wrapped.target_masses_kg)

    def test_continue_and_release_remain_distinct_direct_requests(self):
        config = _config()
        state = _state()
        common = dict(
            state=state,
            action_vector_deg=np.zeros(2),
            demand_axis_scale_deg=config.resolved_legacy_demand_axis_scale_deg,
            action_mass_quantum_kg=config.action_mass_quantum_kg,
            tank_capacity_kg=config.execution.tank_capacity_kg,
        )

        continuing = execution_request_for_action(
            ControlAction.CONTINUE_TARGET,
            **common,
        )
        releasing = execution_request_for_action(
            ControlAction.RELEASE_TARGET,
            **common,
        )

        self.assertEqual(continuing.operation, ExecutionTargetOperation.TRACK)
        np.testing.assert_allclose(continuing.target_masses_kg, state.primary_target_kg)
        self.assertEqual(releasing.operation, ExecutionTargetOperation.RELEASE_TO_CURRENT)
        self.assertIsNone(releasing.target_masses_kg)


if __name__ == "__main__":
    unittest.main()
