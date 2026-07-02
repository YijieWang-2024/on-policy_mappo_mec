import unittest
from types import SimpleNamespace

import gymnasium as gym
import numpy as np

from onpolicy.envs.env_wrappers import DummyVecEnv
from onpolicy.envs.mpe.MPE_env import MPEEnv
from onpolicy.utils.shared_buffer import SharedReplayBuffer


class OneStepTruncationEnv:
    def __init__(self):
        space = gym.spaces.Box(-np.inf, np.inf, shape=(1,), dtype=np.float32)
        self.observation_space = [space]
        self.share_observation_space = [space]
        self.action_space = [gym.spaces.Discrete(1)]

    def reset(self, *, seed=None, options=None):
        return [np.array([0], dtype=np.float32)], {"seed": seed}

    def step(self, action):
        return (
            [np.array([9], dtype=np.float32)],
            [[1.0]],
            [False],
            [True],
            [{"step": "final"}],
        )

    def close(self):
        pass


class GymnasiumMigrationTest(unittest.TestCase):
    def test_all_retained_mpe_scenarios_reset_and_step(self):
        scenarios = {
            "simple_adversary": dict(num_agents=3, num_landmarks=2),
            "simple_attack": dict(
                num_good_agents=1, num_adversaries=3, num_landmarks=4
            ),
            "simple_crypto": dict(num_agents=3, num_landmarks=2),
            "simple_crypto_display": dict(num_agents=3, num_landmarks=2),
            "simple_push": dict(num_agents=2, num_landmarks=2),
            "simple_reference": dict(num_agents=2, num_landmarks=3),
            "simple_speaker_listener": dict(num_agents=2, num_landmarks=3),
            "simple_spread": dict(num_agents=3, num_landmarks=3),
            "simple_tag": dict(
                num_good_agents=1, num_adversaries=3, num_landmarks=2
            ),
            "simple_world_comm": dict(
                num_good_agents=2, num_adversaries=4, num_landmarks=1
            ),
        }

        for scenario_name, scenario_args in scenarios.items():
            with self.subTest(scenario=scenario_name):
                args = SimpleNamespace(
                    scenario_name=scenario_name,
                    episode_length=2,
                    **scenario_args,
                )
                env = MPEEnv(args)
                obs, _ = env.reset(seed=5)
                actions = []
                for action_space in env.action_space:
                    if action_space.__class__.__name__ == "Discrete":
                        actions.append(np.eye(action_space.n, dtype=np.float32)[0])
                    elif action_space.__class__.__name__ == "MultiDiscrete":
                        actions.append(
                            np.concatenate(
                                [
                                    np.eye(high + 1, dtype=np.float32)[0]
                                    for high in action_space.high
                                ]
                            )
                        )
                    else:
                        actions.append(np.zeros(action_space.shape, dtype=np.float32))
                result = env.step(actions)
                self.assertEqual(len(result), 5)
                self.assertEqual(len(result[0]), len(obs))

    def test_mpe_time_limit_is_truncation_and_seeded_reset_is_reproducible(self):
        args = SimpleNamespace(
            scenario_name="simple_spread",
            num_agents=3,
            num_landmarks=3,
            episode_length=1,
        )
        env = MPEEnv(args)
        first_obs, _ = env.reset(seed=7)
        second_obs, _ = env.reset(seed=7)
        np.testing.assert_allclose(first_obs, second_obs)

        actions = [np.eye(5, dtype=np.float32)[0] for _ in range(3)]
        _, _, terminated, truncated, _ = env.step(actions)
        self.assertFalse(any(terminated))
        self.assertTrue(all(truncated))

    def test_vector_autoreset_preserves_final_observation(self):
        envs = DummyVecEnv([OneStepTruncationEnv])
        obs, _ = envs.reset(seed=3)
        self.assertEqual(obs[0, 0, 0], 0)

        obs, _, terminated, truncated, infos = envs.step([[0]])
        self.assertFalse(terminated[0, 0])
        self.assertTrue(truncated[0, 0])
        self.assertEqual(obs[0, 0, 0], 0)
        self.assertEqual(infos[0]["final_observation"][0, 0], 9)
        self.assertEqual(infos[0]["final_info"][0]["step"], "final")

    def test_truncation_bootstraps_but_termination_does_not(self):
        args = SimpleNamespace(
            episode_length=1,
            n_rollout_threads=1,
            hidden_size=1,
            recurrent_N=1,
            gamma=0.9,
            gae_lambda=0.95,
            use_gae=True,
            use_popart=False,
            use_valuenorm=False,
            algorithm_name="mappo",
        )
        obs_space = gym.spaces.Box(-np.inf, np.inf, shape=(1,), dtype=np.float32)
        action_space = gym.spaces.Discrete(2)
        buffer = SharedReplayBuffer(args, 1, obs_space, obs_space, action_space)
        buffer.value_preds[0] = 2
        buffer.next_value_preds[0] = 10
        buffer.rewards[0] = 1
        buffer.masks[1] = 0

        buffer.bootstrap_masks[0] = 1
        buffer.compute_returns()
        self.assertAlmostEqual(float(buffer.returns[0, 0, 0, 0]), 10.0)

        buffer.bootstrap_masks[0] = 0
        buffer.compute_returns()
        self.assertAlmostEqual(float(buffer.returns[0, 0, 0, 0]), 1.0)

    def test_discounted_returns_stop_at_reset_but_bootstrap_truncation(self):
        args = SimpleNamespace(
            episode_length=2,
            n_rollout_threads=1,
            hidden_size=1,
            recurrent_N=1,
            gamma=0.9,
            gae_lambda=0.95,
            use_gae=False,
            use_popart=False,
            use_valuenorm=False,
            algorithm_name="mappo",
        )
        obs_space = gym.spaces.Box(-np.inf, np.inf, shape=(1,), dtype=np.float32)
        buffer = SharedReplayBuffer(
            args, 1, obs_space, obs_space, gym.spaces.Discrete(2)
        )
        buffer.rewards[:] = 1
        buffer.next_value_preds[1] = 10
        buffer.masks[1] = 1
        buffer.masks[2] = 0
        buffer.bootstrap_masks[:] = 1
        buffer.compute_returns()

        self.assertAlmostEqual(float(buffer.returns[1, 0, 0, 0]), 10.0)
        self.assertAlmostEqual(float(buffer.returns[0, 0, 0, 0]), 10.0)


if __name__ == "__main__":
    unittest.main()
