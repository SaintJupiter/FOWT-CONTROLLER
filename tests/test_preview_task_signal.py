import unittest
from dataclasses import FrozenInstanceError

import numpy as np

from wind_prediction.preview_task_signal import (
    PreviewPostureFilter,
    PreviewPostureFilterState,
    filter_affine_task_signal,
)


class PreviewTaskSignalTests(unittest.TestCase):
    def setUp(self):
        # Synthetic test values, not physical tuning or formal task parameters.
        self.config = PreviewPostureFilter(time_constant_s=17.0)
        self.root = PreviewPostureFilterState(
            time_s=120.5, roll_rad=0.03, pitch_rad=-0.02
        )
        self.times = np.array([121.0, 124.5, 131.0, 155.0, 161.5, 190.0])
        self.posture = np.array(
            [
                [0.1, -0.05],
                [-0.03, 0.02],
                [0.04, 0.08],
                [-0.06, -0.01],
                [0.02, -0.04],
                [0.07, 0.03],
            ]
        )

    def advance(self, state=None, times=None, posture=None):
        return self.config.advance(
            initial_state=self.root if state is None else state,
            sample_times_s=self.times if times is None else times,
            posture_rad=self.posture if posture is None else posture,
        )

    def test_irregular_absolute_times_follow_the_requested_recurrence(self):
        trajectory = self.advance()
        self.assertEqual(trajectory.initial_state, self.root)
        self.assertEqual(trajectory.end_state.time_s, self.times[-1])
        expected = np.empty((len(self.times) + 1, 2))
        expected[0] = [self.root.roll_rad, self.root.pitch_rad]
        previous_time = self.root.time_s
        for index, time in enumerate(self.times):
            a = np.exp(-(time - previous_time) / self.config.time_constant_s)
            expected[index + 1] = (
                a * expected[index] + (1.0 - a) * self.posture[index]
            )
            previous_time = time
        np.testing.assert_array_equal(trajectory.posture_rad, expected)
        np.testing.assert_array_equal(
            trajectory.time_s, np.concatenate(([self.root.time_s], self.times))
        )

    def test_segmented_and_single_sample_advances_equal_whole_batch(self):
        whole = self.advance()
        for split in range(1, len(self.times)):
            with self.subTest(split=split):
                first = self.advance(
                    times=self.times[:split], posture=self.posture[:split]
                )
                second = self.advance(
                    state=first.end_state,
                    times=self.times[split:],
                    posture=self.posture[split:],
                )
                self.assertEqual(second.initial_state, first.end_state)
                np.testing.assert_array_equal(
                    np.concatenate((first.posture_rad, second.posture_rad[1:])),
                    whole.posture_rad,
                )
                np.testing.assert_array_equal(
                    np.concatenate((first.time_s, second.time_s[1:])), whole.time_s
                )
                self.assertEqual(second.end_state, whole.end_state)

        state = self.root
        rows = [[state.roll_rad, state.pitch_rad]]
        for time, posture in zip(self.times, self.posture):
            single = self.advance(
                state=state, times=np.array([time]), posture=posture[None, :]
            )
            state = single.end_state
            rows.append(single.posture_rad[-1])
        np.testing.assert_array_equal(rows, whole.posture_rad)
        self.assertEqual(state, whole.end_state)

    def test_changing_future_samples_cannot_change_past_filtered_samples(self):
        prefix = self.advance(times=self.times[:3], posture=self.posture[:3])
        changed = self.posture.copy()
        changed[3:] *= -100.0
        future = self.advance(posture=changed)
        np.testing.assert_array_equal(future.posture_rad[:4], prefix.posture_rad)
        np.testing.assert_array_equal(future.time_s[:4], prefix.time_s)

    def test_branches_from_same_root_are_isolated_and_state_is_immutable(self):
        root_before = PreviewPostureFilterState(120.5, 0.03, -0.02)
        first = self.advance()
        first_before = first.posture_rad.copy()
        other = self.advance(posture=-self.posture)
        self.assertEqual(self.root, root_before)
        np.testing.assert_array_equal(first.posture_rad, first_before)
        np.testing.assert_array_equal(self.advance().posture_rad, first_before)
        self.assertFalse(np.array_equal(first.posture_rad[1:], other.posture_rad[1:]))
        with self.assertRaises(FrozenInstanceError):
            self.root.roll_rad = 1.0
        with self.assertRaises(FrozenInstanceError):
            self.config.time_constant_s = 1.0

    def test_affine_filter_matches_direct_replay_for_random_decisions(self):
        rng = np.random.default_rng(405)
        times = self.root.time_s + np.cumsum(rng.uniform(0.1, 12.0, size=19))
        constant = rng.normal(scale=0.03, size=(19, 2))
        response = rng.normal(scale=0.01, size=(19, 2, 5))
        affine = filter_affine_task_signal(
            config=self.config,
            initial_state=self.root,
            sample_times_s=times,
            constant_rad=constant,
            response_rad=response,
        )
        np.testing.assert_array_equal(
            affine.constant_rad[0], [self.root.roll_rad, self.root.pitch_rad]
        )
        np.testing.assert_array_equal(affine.response_rad[0], np.zeros((2, 5)))
        self.assertEqual(affine.initial_state, self.root)
        for _ in range(12):
            decision = rng.normal(size=5)
            actual = self.advance(
                times=times,
                posture=constant + np.einsum("nad,d->na", response, decision),
            )
            mapped = affine.constant_rad + np.einsum(
                "nad,d->na", affine.response_rad, decision
            )
            np.testing.assert_allclose(
                mapped, actual.posture_rad, atol=1e-15, rtol=1e-13
            )
            np.testing.assert_array_equal(actual.time_s, affine.time_s)
            np.testing.assert_array_equal(mapped[0], affine.constant_rad[0])
            evaluated = affine.evaluate(decision)
            np.testing.assert_array_equal(evaluated.posture_rad, mapped)
            self.assertEqual(evaluated.initial_state, self.root)
            np.testing.assert_allclose(
                [evaluated.end_state.roll_rad, evaluated.end_state.pitch_rad],
                actual.posture_rad[-1],
                atol=1e-15,
                rtol=1e-13,
            )

    def test_affine_filter_does_not_use_future_constants_or_responses(self):
        rng = np.random.default_rng(23)
        response = rng.normal(size=(len(self.times), 2, 3))
        baseline = filter_affine_task_signal(
            config=self.config,
            initial_state=self.root,
            sample_times_s=self.times,
            constant_rad=self.posture,
            response_rad=response,
        )
        changed_constant = self.posture.copy()
        changed_response = response.copy()
        changed_constant[3:] += 100.0
        changed_response[3:] -= 100.0
        changed = filter_affine_task_signal(
            config=self.config,
            initial_state=self.root,
            sample_times_s=self.times,
            constant_rad=changed_constant,
            response_rad=changed_response,
        )
        np.testing.assert_array_equal(
            changed.constant_rad[:4], baseline.constant_rad[:4]
        )
        np.testing.assert_array_equal(
            changed.response_rad[:4], baseline.response_rad[:4]
        )

    def test_outputs_are_readonly_and_do_not_alias_caller_arrays(self):
        times = self.times.copy()
        posture = self.posture.copy()
        response = np.ones((len(times), 2, 3))
        actual = self.advance(times=times, posture=posture)
        affine = filter_affine_task_signal(
            config=self.config,
            initial_state=self.root,
            sample_times_s=times,
            constant_rad=posture,
            response_rad=response,
        )
        actual_before = actual.posture_rad.copy()
        constant_before = affine.constant_rad.copy()
        response_before = affine.response_rad.copy()
        times[:] = -1.0
        posture[:] = 100.0
        response[:] = -100.0
        np.testing.assert_array_equal(actual.posture_rad, actual_before)
        np.testing.assert_array_equal(affine.constant_rad, constant_before)
        np.testing.assert_array_equal(affine.response_rad, response_before)
        np.testing.assert_array_equal(actual.time_s[1:], self.times)
        np.testing.assert_array_equal(affine.time_s[1:], self.times)
        for array in (
            actual.time_s,
            actual.posture_rad,
            actual.roll_rad,
            actual.pitch_rad,
            affine.time_s,
            affine.constant_rad,
            affine.response_rad,
        ):
            with self.subTest(shape=array.shape):
                self.assertFalse(array.flags.writeable)
                with self.assertRaises(ValueError):
                    array.flat[0] = 0.0

    def test_empty_future_retains_root_without_consuming_it_again(self):
        actual = self.advance(times=np.empty(0), posture=np.empty((0, 2)))
        self.assertEqual(actual.initial_state, self.root)
        self.assertEqual(actual.end_state, self.root)
        np.testing.assert_array_equal(actual.time_s, [self.root.time_s])
        affine = filter_affine_task_signal(
            config=self.config,
            initial_state=self.root,
            sample_times_s=np.empty(0),
            constant_rad=np.empty((0, 2)),
            response_rad=np.empty((0, 2, 3)),
        )
        np.testing.assert_array_equal(affine.constant_rad, actual.posture_rad)
        np.testing.assert_array_equal(affine.response_rad, np.zeros((1, 2, 3)))

    def test_invalid_or_repeated_absolute_times_are_explicitly_rejected(self):
        bad_times = (
            [120.5],
            [120.0],
            [121.0, 121.0],
            [122.0, 121.0],
            [121.0, np.nan],
            [np.inf],
            [-np.inf],
            [[121.0]],
            121.0,
        )
        for times in bad_times:
            n = np.asarray(times).size
            with self.subTest(times=times):
                with self.assertRaisesRegex(ValueError, "times|strictly"):
                    self.advance(times=times, posture=np.zeros((n, 2)))
                with self.assertRaisesRegex(ValueError, "times|strictly"):
                    filter_affine_task_signal(
                        config=self.config,
                        initial_state=self.root,
                        sample_times_s=times,
                        constant_rad=np.zeros((n, 2)),
                        response_rad=np.zeros((n, 2, 3)),
                    )
        consumed = self.advance()
        with self.assertRaisesRegex(ValueError, "strictly after"):
            self.advance(
                state=consumed.end_state,
                times=self.times[-1:],
                posture=self.posture[-1:],
            )

    def test_time_constant_is_required_positive_and_finite(self):
        with self.assertRaises(TypeError):
            PreviewPostureFilter()
        for value in (0.0, -1.0, np.nan, np.inf, -np.inf):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "positive and finite"):
                    PreviewPostureFilter(time_constant_s=value)
        for name in ("time_s", "roll_rad", "pitch_rad"):
            values = dict(time_s=0.0, roll_rad=0.0, pitch_rad=0.0)
            values[name] = np.nan
            with self.subTest(name=name):
                with self.assertRaisesRegex(ValueError, name):
                    PreviewPostureFilterState(**values)

    def test_posture_and_affine_shapes_and_finiteness_are_checked(self):
        for posture in (
            np.zeros((6, 3)),
            np.zeros((5, 2)),
            np.zeros(12),
            np.full((6, 2), np.nan),
        ):
            with self.subTest(shape=posture.shape):
                with self.assertRaises(ValueError):
                    self.advance(posture=posture)
        for response in (
            np.zeros((6, 2)),
            np.zeros((5, 2, 3)),
            np.zeros((6, 3, 3)),
            np.full((6, 2, 3), np.inf),
        ):
            with self.subTest(shape=response.shape):
                with self.assertRaisesRegex(ValueError, "response_rad"):
                    filter_affine_task_signal(
                        config=self.config,
                        initial_state=self.root,
                        sample_times_s=self.times,
                        constant_rad=self.posture,
                        response_rad=response,
                    )

        affine = filter_affine_task_signal(
            config=self.config,
            initial_state=self.root,
            sample_times_s=self.times,
            constant_rad=self.posture,
            response_rad=np.zeros((6, 2, 3)),
        )
        for decision in (np.zeros(2), np.zeros((3, 1)), [0.0, np.nan, 0.0]):
            with self.subTest(decision=decision):
                with self.assertRaisesRegex(ValueError, "decision"):
                    affine.evaluate(decision)


if __name__ == "__main__":
    unittest.main()
