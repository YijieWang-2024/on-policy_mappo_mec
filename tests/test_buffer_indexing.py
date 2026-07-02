import unittest
from types import SimpleNamespace
from unittest.mock import patch

import gymnasium as gym
import numpy as np
import torch

from onpolicy.utils.separated_buffer import SeparatedReplayBuffer
from onpolicy.utils.shared_buffer import SharedReplayBuffer


def make_args(*, episode_length=4, n_rollout_threads=1, use_gae=True):
    return SimpleNamespace(
        episode_length=episode_length,
        n_rollout_threads=n_rollout_threads,
        hidden_size=1,
        recurrent_N=1,
        gamma=0.9,
        gae_lambda=0.8,
        use_gae=use_gae,
        use_popart=False,
        use_valuenorm=False,
        algorithm_name="mappo",
    )


def make_buffers(*, episode_length=4, n_rollout_threads=1, use_gae=True):
    args = make_args(
        episode_length=episode_length,
        n_rollout_threads=n_rollout_threads,
        use_gae=use_gae,
    )
    obs_space = gym.spaces.Box(-np.inf, np.inf, shape=(1,), dtype=np.float32)
    act_space = gym.spaces.Box(-1, 1, shape=(1,), dtype=np.float32)
    return (
        SharedReplayBuffer(args, 1, obs_space, obs_space, act_space),
        SeparatedReplayBuffer(args, obs_space, obs_space, act_space),
    )


class BufferIndexingTest(unittest.TestCase):
    def test_returns_use_transition_next_values_and_next_state_masks(self):
        current_values = np.array([1, 2, 3, 4], dtype=np.float32)
        next_values = np.array([2, 10, 4, 8], dtype=np.float32)
        next_masks = np.array([1, 0, 1, 0], dtype=np.float32)
        bootstrap_masks = np.array([1, 1, 1, 0], dtype=np.float32)

        for use_gae, expected in (
            (True, [8.56, 10.0, 2.44, 1.0]),
            (False, [10.0, 10.0, 1.9, 1.0]),
        ):
            for buffer in make_buffers(use_gae=use_gae):
                with self.subTest(
                    use_gae=use_gae, buffer=buffer.__class__.__name__
                ):
                    buffer.value_preds[:-1].reshape(4, -1)[:, 0] = current_values
                    buffer.next_value_preds.reshape(4, -1)[:, 0] = next_values
                    buffer.rewards.fill(1)
                    buffer.masks[1:].reshape(4, -1)[:, 0] = next_masks
                    buffer.bootstrap_masks.reshape(4, -1)[:, 0] = bootstrap_masks

                    buffer.compute_returns()

                    actual = buffer.returns[:-1].reshape(4, -1)[:, 0]
                    np.testing.assert_allclose(actual, expected, atol=1e-6)
                    if isinstance(buffer, SharedReplayBuffer):
                        np.testing.assert_allclose(
                            buffer.advantages.reshape(4, -1)[:, 0],
                            actual - current_values,
                            atol=1e-6,
                        )

    def test_insert_stores_transition_at_t_and_next_state_at_t_plus_one(self):
        for buffer in make_buffers(episode_length=2):
            with self.subTest(buffer=buffer.__class__.__name__):
                shared = isinstance(buffer, SharedReplayBuffer)
                state_shape = (1, 1, 1, 1) if shared else (1, 1, 1)
                obs_shape = (1, 1, 1) if shared else (1, 1)
                transition_shape = (1, 1, 1) if shared else (1, 1)

                for step in range(2):
                    marker = float(step + 1)
                    buffer.insert(
                        np.full(obs_shape, marker + 10, dtype=np.float32),
                        np.full(obs_shape, marker + 20, dtype=np.float32),
                        np.full(state_shape, marker + 30, dtype=np.float32),
                        np.full(state_shape, marker + 40, dtype=np.float32),
                        np.full(transition_shape, marker + 50, dtype=np.float32),
                        np.full(transition_shape, marker + 60, dtype=np.float32),
                        np.full(transition_shape, marker + 70, dtype=np.float32),
                        np.full(transition_shape, marker + 80, dtype=np.float32),
                        np.full(transition_shape, marker + 90, dtype=np.float32),
                        np.full(transition_shape, marker + 100, dtype=np.float32),
                        np.full(transition_shape, marker + 110, dtype=np.float32),
                    )

                    self.assertEqual(float(buffer.value_preds[step].flat[0]), marker + 70)
                    self.assertEqual(
                        float(buffer.next_value_preds[step].flat[0]), marker + 80
                    )
                    self.assertEqual(float(buffer.rewards[step].flat[0]), marker + 90)
                    self.assertEqual(float(buffer.obs[step + 1].flat[0]), marker + 20)
                    self.assertEqual(float(buffer.masks[step + 1].flat[0]), marker + 100)
                    self.assertEqual(
                        float(buffer.bootstrap_masks[step].flat[0]), marker + 110
                    )

                buffer.after_update()
                np.testing.assert_array_equal(buffer.obs[0], buffer.obs[-1])
                np.testing.assert_array_equal(buffer.masks[0], buffer.masks[-1])
                np.testing.assert_array_equal(
                    buffer.rnn_states_critic[0], buffer.rnn_states_critic[-1]
                )

    def test_separated_recurrent_generator_flattens_time_before_chunks(self):
        _, buffer = make_buffers(episode_length=4, n_rollout_threads=2)
        for env_id in range(2):
            for step in range(4):
                marker = env_id * 10 + step
                buffer.obs[step, env_id, 0] = marker
                buffer.share_obs[step, env_id, 0] = marker
                buffer.rnn_states[step, env_id, 0, 0] = marker

        advantages = np.zeros_like(buffer.rewards)
        with patch.object(torch, "randperm", return_value=torch.arange(4)):
            batch = next(buffer.recurrent_generator(advantages, 1, 2))

        obs_batch = batch[1][:, 0]
        rnn_states_batch = batch[2][:, 0, 0]
        np.testing.assert_array_equal(obs_batch, [0, 2, 10, 12, 1, 3, 11, 13])
        np.testing.assert_array_equal(rnn_states_batch, [0, 2, 10, 12])

    def test_feed_forward_generators_do_not_drop_remainder_samples(self):
        for buffer in make_buffers(episode_length=5):
            with self.subTest(buffer=buffer.__class__.__name__):
                buffer.obs[:-1].reshape(5, -1)[:, 0] = np.arange(5)
                advantages = np.zeros_like(buffer.rewards)
                with patch.object(torch, "randperm", return_value=torch.arange(5)):
                    batches = list(buffer.feed_forward_generator(advantages, 2))

                observed = np.concatenate([batch[1][:, 0] for batch in batches])
                np.testing.assert_array_equal(observed, np.arange(5))
                self.assertEqual(sum(len(batch[1]) for batch in batches), 5)

                with patch.object(torch, "randperm", return_value=torch.arange(5)):
                    explicit_batches = list(
                        buffer.feed_forward_generator(
                            advantages, mini_batch_size=2
                        )
                    )
                self.assertEqual([len(batch[1]) for batch in explicit_batches], [2, 2, 1])

    def test_recurrent_chunks_cannot_cross_trajectory_boundaries(self):
        for buffer in make_buffers(episode_length=5):
            with self.subTest(buffer=buffer.__class__.__name__):
                with self.assertRaisesRegex(AssertionError, "episode_length"):
                    next(
                        buffer.recurrent_generator(
                            np.zeros_like(buffer.rewards), 1, 2
                        )
                    )


if __name__ == "__main__":
    unittest.main()
