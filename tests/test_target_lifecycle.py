import unittest

import numpy as np

from wind_prediction.target_lifecycle import (
    apply_planner_authoritative_target,
    apply_primary_target_state_defaults,
    apply_primary_target_transition,
    mark_primary_target_reused,
    primary_target_at_current,
    primary_target_blended_to_current,
    primary_target_capped_to_current,
    primary_target_from_delta,
    primary_target_from_paused,
    remember_paused_primary_target,
)


class _TargetState:
    pass


class TargetLifecycleTests(unittest.TestCase):
    def test_planner_authoritative_commit_uses_exact_evaluated_target(self):
        state = _TargetState()
        apply_primary_target_state_defaults(
            state,
            [100.0, 200.0, 300.0],
            refresh_owner="initial",
        )
        state._paused_primary_valid = True

        transition = apply_planner_authoritative_target(
            state,
            operation="set_delta",
            current_masses_kg=[110.0, 190.0, 300.0],
            requested_target_kg=[160.0, 170.0, 270.0],
            tank_capacity_kg=1000.0,
            current_time_s=20.0,
        )

        np.testing.assert_allclose(transition.target_kg, [160.0, 170.0, 270.0])
        np.testing.assert_allclose(state._primary_delta_kg, [50.0, -20.0, -30.0])
        self.assertFalse(state._paused_primary_valid)
        self.assertEqual(state._primary_refresh_owner, "planner_authoritative")

    def test_planner_authoritative_hold_releases_outstanding_target(self):
        state = _TargetState()
        apply_primary_target_state_defaults(
            state,
            [100.0, 200.0, 300.0],
            refresh_owner="initial",
        )

        apply_planner_authoritative_target(
            state,
            operation="hold_current",
            current_masses_kg=[120.0, 180.0, 300.0],
            requested_target_kg=None,
            tank_capacity_kg=1000.0,
            current_time_s=30.0,
        )

        np.testing.assert_allclose(state._primary_target_kg, [120.0, 180.0, 300.0])
        np.testing.assert_allclose(state._primary_delta_kg, np.zeros(3))

    def test_generate_pause_resume_release_sequence(self):
        state = _TargetState()
        apply_primary_target_state_defaults(
            state,
            [100.0, 200.0, 300.0],
            refresh_owner="initial",
        )

        generated = primary_target_from_delta(
            [100.0, 200.0, 300.0],
            [50.0, -25.0, -25.0],
            tank_capacity_kg=1000.0,
            current_time_s=10.0,
        )
        apply_primary_target_transition(state, generated)
        np.testing.assert_allclose(state._primary_target_kg, [150.0, 175.0, 275.0])

        remember_paused_primary_target(
            state,
            action="active_small",
            avec=[0.1, -0.2],
        )
        held = primary_target_at_current(
            [110.0, 190.0, 300.0],
            tank_capacity_kg=1000.0,
            current_time_s=20.0,
            initialized=True,
            clip_target=True,
        )
        apply_primary_target_transition(state, held)
        np.testing.assert_allclose(state._primary_target_kg, [110.0, 190.0, 300.0])

        resumed = primary_target_from_paused(
            [110.0, 190.0, 300.0],
            state._paused_primary_target_kg,
            tank_capacity_kg=1000.0,
            current_time_s=30.0,
        )
        apply_primary_target_transition(state, resumed)
        self.assertTrue(state._primary_target_resumed)
        np.testing.assert_allclose(state._primary_target_kg, [150.0, 175.0, 275.0])

        released = primary_target_blended_to_current(
            [120.0, 185.0, 295.0],
            state._primary_target_kg,
            blend=1.0,
            tank_capacity_kg=1000.0,
            current_time_s=40.0,
        )
        apply_primary_target_transition(state, released)
        np.testing.assert_allclose(state._primary_delta_kg, [0.0, 0.0, 0.0])

    def test_cap_and_reuse_preserve_explicit_state_flags(self):
        state = _TargetState()
        apply_primary_target_state_defaults(state, [100.0] * 3, refresh_owner="initial")
        capped = primary_target_capped_to_current(
            [100.0] * 3,
            [500.0, -100.0, 150.0],
            cap_kg=40.0,
            tank_capacity_kg=1000.0,
            current_time_s=5.0,
        )
        apply_primary_target_transition(state, capped)
        np.testing.assert_allclose(state._primary_delta_kg, [40.0, -40.0, 40.0])
        mark_primary_target_reused(state, refresh_owner="test_reuse")
        self.assertTrue(state._primary_target_reused)
        self.assertFalse(state._primary_target_refreshed)
        self.assertEqual(state._primary_refresh_owner, "test_reuse")
    def test_blend_zero_keeps_pending_target(self):
        transition = primary_target_blended_to_current(
            [100, 200, 300],
            [400, 500, 600],
            blend=0.0,
            tank_capacity_kg=1000.0,
            current_time_s=10.0,
        )
        np.testing.assert_allclose(transition.target_kg, [400, 500, 600])
        np.testing.assert_allclose(transition.delta_kg, [300, 300, 300])

    def test_blend_one_releases_target_to_current_masses(self):
        transition = primary_target_blended_to_current(
            [100, 200, 300],
            [400, 500, 600],
            blend=1.0,
            tank_capacity_kg=1000.0,
            current_time_s=20.0,
        )
        np.testing.assert_allclose(transition.target_kg, [100, 200, 300])
        np.testing.assert_allclose(transition.delta_kg, [0, 0, 0])

    def test_blend_is_clamped_and_target_respects_capacity(self):
        transition = primary_target_blended_to_current(
            [1200, -100, 300],
            [900, 500, 600],
            blend=2.0,
            tank_capacity_kg=1000.0,
            current_time_s=30.0,
        )
        np.testing.assert_allclose(transition.target_kg, [1000, 0, 300])


if __name__ == "__main__":
    unittest.main()
