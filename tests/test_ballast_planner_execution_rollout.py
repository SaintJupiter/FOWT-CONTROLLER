import unittest

import numpy as np

from wind_prediction.ballast_planner import (
    STOP_EXECUTION_ACTION,
    PlannerConfig,
    candidate_actions,
    candidate_sequences,
    compensation_vec_from_mass_delta,
    evaluate_sequence,
)
from wind_prediction.execution_rollout import ExecutionRolloutConfig


class BallastPlannerExecutionRolloutTests(unittest.TestCase):
    @staticmethod
    def block(pressure=(2.0, 0.0)):
        vector = np.asarray(pressure, dtype=float)
        return {
            "pressure_vec": vector,
            "pressure_vec_raw": vector.copy(),
            "pressure_norm": float(np.linalg.norm(vector)),
        }

    @staticmethod
    def plant(*, off_elapsed_s=120.0):
        masses = np.asarray([50000.0, 50000.0, 50000.0], dtype=float)
        return {
            "tank_masses": masses,
            "target_ballast_mass": masses.copy(),
            "primary_target_kg": masses.copy(),
            "prediction_primary_scale": 1.0,
            "pump_latched": np.zeros(3, dtype=bool),
            "pump_rate_cmd_m3_min": np.zeros(3, dtype=float),
            "pump_net_rate_m3_min": np.zeros(3, dtype=float),
            "pump_on_elapsed_s": np.zeros(3, dtype=float),
            "pump_off_elapsed_s": np.full(3, off_elapsed_s, dtype=float),
            "pump_near_target_s": np.zeros(3, dtype=float),
        }

    @staticmethod
    def execution_config():
        return ExecutionRolloutConfig(
            block_duration_s=60.0,
            internal_step_s=1.0,
            water_density_kg_m3=1000.0,
            max_pump_rate_m3_min=1.0,
            target_slew_enabled=False,
            stop_error_kg=1.0,
            restart_error_kg=2.0,
            min_on_s=0.0,
            min_off_s=120.0,
            near_target_hold_s=0.0,
            ramp_up_m3_min_per_s=np.inf,
            ramp_down_m3_min_per_s=np.inf,
            tank_capacity_kg=100000.0,
            pump_rate_schedule_m3_min=((0.0, 1.0), (100000.0, 1.0)),
        )

    def test_legacy_path_keeps_nominal_action_cost(self):
        cfg = PlannerConfig(
            action_mass_quantum_kg=60000.0,
            tank_capacity_kg=100000.0,
        )

        result = evaluate_sequence(
            ("active_small",),
            [self.block()],
            self.plant(),
            cfg,
        )

        self.assertEqual(result["execution_rollout_active"], 0)
        self.assertAlmostEqual(result["costs"]["pump_work_cost"], 0.15)
        self.assertEqual(result["costs"]["actual_pump_volume_m3"], 0.0)

    def test_identical_masses_but_different_pump_state_change_execution_cost(self):
        cfg = PlannerConfig(
            action_mass_quantum_kg=60000.0,
            tank_capacity_kg=100000.0,
            execution_rollout_active=True,
            execution_rollout=self.execution_config(),
        )

        waiting = evaluate_sequence(
            ("active_small",),
            [self.block()],
            self.plant(off_elapsed_s=0.0),
            cfg,
        )
        ready = evaluate_sequence(
            ("active_small",),
            [self.block()],
            self.plant(off_elapsed_s=120.0),
            cfg,
        )

        self.assertEqual(waiting["costs"]["actual_pump_volume_m3"], 0.0)
        self.assertGreater(ready["costs"]["actual_pump_volume_m3"], 0.0)
        self.assertGreater(ready["costs"]["actual_pump_starts"], 0.0)
        self.assertLess(
            ready["costs"]["terminal_residual_cost"],
            waiting["costs"]["terminal_residual_cost"],
        )

    def test_execution_rollout_exposes_exact_first_target_for_runtime_commit(self):
        cfg = PlannerConfig(
            action_mass_quantum_kg=60000.0,
            tank_capacity_kg=100000.0,
            execution_rollout_active=True,
            execution_rollout=self.execution_config(),
        )
        plant = self.plant(off_elapsed_s=120.0)

        result = evaluate_sequence(
            ("active_small",),
            [self.block()],
            plant,
            cfg,
        )

        requested = np.asarray(
            result["first_execution_requested_target_kg"],
            dtype=float,
        )
        self.assertEqual(requested.shape, (3,))
        self.assertGreater(np.max(np.abs(requested - plant["tank_masses"])), 0.0)

    def test_v2_action_library_separates_target_maintenance_from_pump_stop(self):
        legacy = PlannerConfig()
        explicit = PlannerConfig(
            candidate_action_mode="explicit_target_lifecycle",
        )

        self.assertNotIn("maintain_target", candidate_actions(legacy))
        self.assertIn("maintain_target", candidate_actions(explicit))
        self.assertIn(STOP_EXECUTION_ACTION, candidate_actions(explicit))
        self.assertNotIn("hold", candidate_actions(explicit))
        self.assertEqual(len(candidate_sequences(legacy)), 5 ** 3)
        self.assertEqual(len(candidate_sequences(explicit)), 6 ** 3)

    def test_maintain_target_continues_existing_pump_demand_while_stop_stops(self):
        cfg = PlannerConfig(
            action_mass_quantum_kg=60000.0,
            tank_capacity_kg=100000.0,
            candidate_action_mode="explicit_target_lifecycle",
            execution_rollout_active=True,
            execution_rollout=self.execution_config(),
        )
        plant = self.plant(off_elapsed_s=120.0)
        existing_target = np.asarray([65000.0, 35000.0, 50000.0])
        plant["primary_target_kg"] = existing_target.copy()
        plant["target_ballast_mass"] = existing_target.copy()

        maintain = evaluate_sequence(
            ("maintain_target",),
            [self.block()],
            plant,
            cfg,
        )
        stop = evaluate_sequence(
            (STOP_EXECUTION_ACTION,),
            [self.block()],
            plant,
            cfg,
        )

        np.testing.assert_allclose(
            maintain["first_execution_requested_target_kg"],
            existing_target,
        )
        np.testing.assert_allclose(
            stop["first_execution_requested_target_kg"],
            plant["tank_masses"],
        )
        self.assertGreater(maintain["costs"]["actual_pump_volume_m3"], 0.0)
        self.assertEqual(stop["costs"]["actual_pump_volume_m3"], 0.0)

    def test_completed_target_outside_envelope_returns_decision_to_planner(self):
        cfg = PlannerConfig(
            action_mass_quantum_kg=60000.0,
            tank_capacity_kg=100000.0,
            candidate_action_mode="explicit_target_lifecycle",
            execution_rollout_active=True,
            execution_rollout=self.execution_config(),
            target_replan_policy_mode="enforce",
        )
        plant = self.plant(off_elapsed_s=120.0)
        plant["posture_vec_deg"] = np.asarray([2.0, 0.0])
        plant["posture_rate_vec_deg_s"] = np.asarray([0.01, 0.0])

        maintain = evaluate_sequence(
            ("maintain_target",),
            [self.block()],
            plant,
            cfg,
        )
        stop = evaluate_sequence(
            (STOP_EXECUTION_ACTION,),
            [self.block()],
            plant,
            cfg,
        )
        active = evaluate_sequence(
            ("active_small",),
            [self.block()],
            plant,
            cfg,
        )

        self.assertEqual(maintain["hard_reject_reason"], "")
        self.assertEqual(stop["hard_reject_reason"], "")
        self.assertEqual(active["hard_reject_reason"], "")
        self.assertEqual(active["target_replan_required"], 0)

    def test_explicit_lifecycle_rejects_zero_effect_active_action(self):
        cfg = PlannerConfig(
            action_mass_quantum_kg=60000.0,
            tank_capacity_kg=100000.0,
            candidate_action_mode="explicit_target_lifecycle",
            execution_rollout_active=True,
            execution_rollout=self.execution_config(),
            target_replan_policy_mode="enforce",
        )
        plant = self.plant(off_elapsed_s=120.0)
        plant["posture_vec_deg"] = np.asarray([0.0, -2.0])
        plant["posture_rate_vec_deg_s"] = np.asarray([0.0, -0.01])

        result = evaluate_sequence(
            ("pump_saving",),
            [self.block((0.0, 0.0))],
            plant,
            cfg,
        )

        self.assertEqual(
            result["hard_reject_reason"],
            "active_action_has_zero_effect",
        )
        self.assertEqual(result["target_replan_required"], 0)

    def test_replan_rejects_final_action_override_that_does_not_correct_posture(self):
        cfg = PlannerConfig(
            action_mass_quantum_kg=60000.0,
            tank_capacity_kg=100000.0,
            candidate_action_mode="explicit_target_lifecycle",
            execution_rollout_active=True,
            execution_rollout=self.execution_config(),
            target_replan_policy_mode="enforce",
        )
        plant = self.plant(off_elapsed_s=120.0)
        plant["posture_vec_deg"] = np.asarray([0.0, -2.0])
        plant["posture_rate_vec_deg_s"] = np.asarray([0.0, -0.01])
        harmful_target = np.asarray([50000.0, 56000.0, 44000.0])
        plant["primary_target_kg"] = harmful_target.copy()
        plant["target_ballast_mass"] = harmful_target.copy()

        result = evaluate_sequence(
            ("active_small",),
            [self.block((0.0, 0.0))],
            plant,
            cfg,
            action_vec_overrides={0: np.asarray([0.1, 0.1])},
        )

        self.assertEqual(result["hard_reject_reason"], "replan_action_not_corrective")
        self.assertEqual(result["target_replan_action_corrective"], 0)

    def test_unhelpful_unfinished_target_can_be_stopped_before_replanning(self):
        cfg = PlannerConfig(
            action_mass_quantum_kg=60000.0,
            tank_capacity_kg=100000.0,
            candidate_action_mode="explicit_target_lifecycle",
            execution_rollout_active=True,
            execution_rollout=self.execution_config(),
            target_replan_policy_mode="enforce",
        )
        plant = self.plant(off_elapsed_s=120.0)
        plant["posture_vec_deg"] = np.asarray([2.0, 0.0])
        plant["posture_rate_vec_deg_s"] = np.asarray([0.01, 0.0])
        harmful_target = np.asarray([56000.0, 47000.0, 47000.0])
        plant["primary_target_kg"] = harmful_target.copy()
        plant["target_ballast_mass"] = harmful_target.copy()

        result = evaluate_sequence(
            (STOP_EXECUTION_ACTION,),
            [self.block((0.0, 0.0))],
            plant,
            cfg,
        )

        requested = np.asarray(result["first_execution_requested_target_kg"])
        self.assertEqual(result["target_replan_reason"], "unfinished_target_direction_not_helpful")
        self.assertEqual(result["hard_reject_reason"], "")
        np.testing.assert_allclose(requested, plant["tank_masses"])

    def test_inside_envelope_also_rejects_ambiguous_zero_active_action(self):
        cfg = PlannerConfig(
            action_mass_quantum_kg=60000.0,
            tank_capacity_kg=100000.0,
            candidate_action_mode="explicit_target_lifecycle",
            execution_rollout_active=True,
            execution_rollout=self.execution_config(),
            target_replan_policy_mode="enforce",
        )
        plant = self.plant(off_elapsed_s=120.0)
        plant["posture_vec_deg"] = np.asarray([0.5, 0.0])
        plant["posture_rate_vec_deg_s"] = np.asarray([0.0, 0.0])

        result = evaluate_sequence(
            ("pump_saving",),
            [self.block((0.0, 0.0))],
            plant,
            cfg,
        )

        self.assertEqual(
            result["hard_reject_reason"],
            "active_action_has_zero_effect",
        )
        self.assertEqual(result["target_replan_required"], 0)
        self.assertEqual(result["target_replan_action_redirected"], 0)

    def test_completed_target_can_wait_while_posture_is_recovering(self):
        cfg = PlannerConfig(
            action_mass_quantum_kg=60000.0,
            tank_capacity_kg=100000.0,
            candidate_action_mode="explicit_target_lifecycle",
            execution_rollout_active=True,
            execution_rollout=self.execution_config(),
            target_replan_policy_mode="enforce",
        )
        plant = self.plant(off_elapsed_s=120.0)
        plant["posture_vec_deg"] = np.asarray([2.0, 0.0])
        plant["posture_rate_vec_deg_s"] = np.asarray([-0.01, 0.0])

        maintain = evaluate_sequence(
            ("maintain_target",),
            [self.block()],
            plant,
            cfg,
        )

        self.assertEqual(maintain["hard_reject_reason"], "")
        self.assertEqual(maintain["target_replan_required"], 0)
        self.assertEqual(
            maintain["target_replan_reason"],
            "outside_envelope_recovering",
        )

    def test_target_replan_shadow_records_without_rejecting(self):
        cfg = PlannerConfig(
            action_mass_quantum_kg=60000.0,
            tank_capacity_kg=100000.0,
            candidate_action_mode="explicit_target_lifecycle",
            execution_rollout_active=True,
            execution_rollout=self.execution_config(),
            target_replan_policy_mode="shadow",
        )
        plant = self.plant(off_elapsed_s=120.0)
        plant["posture_vec_deg"] = np.asarray([2.0, 0.0])
        plant["posture_rate_vec_deg_s"] = np.asarray([0.0, 0.0])

        maintain = evaluate_sequence(
            ("maintain_target",),
            [self.block()],
            plant,
            cfg,
        )

        self.assertEqual(maintain["hard_reject_reason"], "")
        self.assertEqual(maintain["target_replan_required"], 0)

    def test_common_mode_mass_change_has_no_attitude_compensation(self):
        cfg = PlannerConfig(action_mass_quantum_kg=60000.0)

        compensation = compensation_vec_from_mass_delta(
            np.asarray([1000.0, 1000.0, 1000.0]),
            cfg,
        )

        np.testing.assert_allclose(compensation, np.zeros(2), atol=1e-12)

    def test_final_vector_override_is_used_by_execution_rollout(self):
        cfg = PlannerConfig(
            action_mass_quantum_kg=60000.0,
            tank_capacity_kg=100000.0,
            execution_rollout_active=True,
            execution_rollout=self.execution_config(),
        )

        nominal = evaluate_sequence(
            ("active_small",),
            [self.block()],
            self.plant(off_elapsed_s=120.0),
            cfg,
        )
        corrected = evaluate_sequence(
            ("active_small",),
            [self.block()],
            self.plant(off_elapsed_s=120.0),
            cfg,
            action_vec_overrides={0: np.zeros(2, dtype=float)},
        )

        self.assertGreater(nominal["costs"]["actual_pump_volume_m3"], 0.0)
        self.assertEqual(corrected["costs"]["actual_pump_volume_m3"], 0.0)

    def test_exact_target_override_is_used_by_execution_rollout(self):
        cfg = PlannerConfig(
            action_mass_quantum_kg=60000.0,
            tank_capacity_kg=100000.0,
            execution_rollout_active=True,
            execution_rollout=self.execution_config(),
        )
        plant = self.plant(off_elapsed_s=120.0)

        corrected = evaluate_sequence(
            ("active_small",),
            [self.block()],
            plant,
            cfg,
            requested_target_overrides={0: plant["tank_masses"].copy()},
        )

        self.assertEqual(corrected["costs"]["actual_pump_volume_m3"], 0.0)

    def test_execution_switch_costs_are_normalized_by_pump_count(self):
        cfg = PlannerConfig(
            action_mass_quantum_kg=60000.0,
            tank_capacity_kg=100000.0,
            execution_rollout_active=True,
            execution_rollout=self.execution_config(),
        )

        result = evaluate_sequence(
            ("active_small",),
            [self.block()],
            self.plant(off_elapsed_s=120.0),
            cfg,
        )

        raw_cycles = (
            result["costs"]["actual_pump_starts"]
            + result["costs"]["actual_pump_stops"]
        )
        self.assertAlmostEqual(
            result["costs"]["startstop_cost"],
            raw_cycles / 6.0,
        )
        self.assertAlmostEqual(
            result["costs"]["direction_switch_cost"],
            result["costs"]["actual_direction_switches"] / 3.0,
        )

    def test_future_strong_action_does_not_clear_current_hold_barrier(self):
        cfg = PlannerConfig(
            action_mass_quantum_kg=60000.0,
            tank_capacity_kg=100000.0,
            execution_rollout_active=True,
            execution_rollout=self.execution_config(),
            envelope_barrier_active=True,
            envelope_barrier_strong_actions=("active_small", "active_medium"),
        )
        blocks = [self.block((2.0, 0.0)) for _ in range(3)]

        result = evaluate_sequence(
            ("hold", "hold", "active_small"),
            blocks,
            self.plant(off_elapsed_s=120.0),
            cfg,
        )

        self.assertGreaterEqual(
            result["costs"]["envelope_barrier_triggered"],
            1.0,
        )

    def test_medium_gate_uses_actual_execution_not_nominal_request(self):
        cfg = PlannerConfig(
            action_mass_quantum_kg=60000.0,
            tank_capacity_kg=100000.0,
            execution_rollout_active=True,
            execution_rollout=self.execution_config(),
        )

        result = evaluate_sequence(
            ("active_medium",),
            [self.block()],
            self.plant(off_elapsed_s=0.0),
            cfg,
        )

        self.assertEqual(result["hard_reject_reason"], "active_medium_gate")
        self.assertEqual(
            result["active_medium_gate_reason"],
            "small_and_medium_same_band",
        )

    def test_current_fullspeed_flag_does_not_reject_stateful_rollout(self):
        cfg = PlannerConfig(
            action_mass_quantum_kg=60000.0,
            tank_capacity_kg=100000.0,
            execution_rollout_active=True,
            execution_rollout=self.execution_config(),
        )
        plant = self.plant(off_elapsed_s=120.0)
        plant["pump_fullspeed_any"] = 1

        result = evaluate_sequence(
            ("active_small",),
            [self.block()],
            plant,
            cfg,
        )

        self.assertNotEqual(result["hard_reject_reason"], "fullspeed_guard")
        self.assertGreater(result["costs"]["actual_pump_volume_m3"], 0.0)


if __name__ == "__main__":
    unittest.main()
