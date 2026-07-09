from __future__ import annotations

from tempfile import TemporaryDirectory

import gymnasium as gym
import numpy as np
import pytest

from onpolicy.envs.env_wrappers import ShareDummyVecEnv
from onpolicy.envs.mec.normalization import (
    MECComponentNormalizer,
    MECComponentVecNormalize,
)
from onpolicy.envs.mec.observation import (
    AGENT_OBS_DIM,
    OWN_SLICE,
    PHYSICAL_PUBLIC_STATE_DIM,
    PHYSICAL_PUBLIC_SLICE,
    PUBLIC_SLICE,
    RESOURCE_CONTEXT_SLICE,
    team_state_dim,
    repeat_team_state,
)


NUM_AGENTS = 3
PUBLIC = np.arange(1, PHYSICAL_PUBLIC_STATE_DIM + 1, dtype=np.float32) * 10 + 2
RESOURCE = np.asarray([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], dtype=np.float32)
UAVS = np.asarray([[2, 4, 6], [0, 0, 0]], dtype=np.float32)


def _obs(public=PUBLIC, uavs=UAVS):
    rows = np.zeros((NUM_AGENTS, AGENT_OBS_DIM), dtype=np.float32)
    rows[0, 0] = 1.0
    rows[0, OWN_SLICE] = public[:3]
    rows[:, PHYSICAL_PUBLIC_SLICE] = public
    rows[:, RESOURCE_CONTEXT_SLICE] = RESOURCE
    rows[1:, OWN_SLICE] = uavs
    return rows


def _seeded_normalizer(num_agents=NUM_AGENTS):
    normalizer = MECComponentNormalizer(num_agents, clip_obs=100.0)
    normalizer.physical_public_rms.mean = np.arange(
        1, PHYSICAL_PUBLIC_STATE_DIM + 1, dtype=np.float64
    ) * 10
    normalizer.physical_public_rms.var = np.full(PHYSICAL_PUBLIC_STATE_DIM, 4.0, dtype=np.float64)
    normalizer.physical_public_rms.count = 10.0
    normalizer.uav_state_rms.mean = np.asarray([1, 2, 3], dtype=np.float64)
    normalizer.uav_state_rms.var = np.asarray([1, 4, 9], dtype=np.float64)
    normalizer.uav_state_rms.count = 10.0
    return normalizer


def test_component_normalizer_reconstructs_obs_and_share_obs():
    normalizer = _seeded_normalizer()
    obs = _obs()
    share_obs = repeat_team_state(obs)

    norm_obs = normalizer.normalize_obs(obs)
    norm_share = normalizer.normalize_share_obs(share_obs)

    public_expected = np.ones(PHYSICAL_PUBLIC_STATE_DIM, dtype=np.float32)
    uav_expected = np.asarray([[1, 1, 1], [-1, -1, -1]], dtype=np.float32)

    assert norm_obs.shape == obs.shape
    assert norm_share.shape == share_obs.shape
    np.testing.assert_allclose(norm_obs[:, 0], obs[:, 0])
    np.testing.assert_allclose(
        norm_obs[:, RESOURCE_CONTEXT_SLICE],
        np.repeat(RESOURCE[None, :], NUM_AGENTS, axis=0),
    )
    np.testing.assert_allclose(norm_obs[0, OWN_SLICE], public_expected[:3])
    np.testing.assert_allclose(
        norm_obs[:, PHYSICAL_PUBLIC_SLICE],
        np.repeat(public_expected[None, :], NUM_AGENTS, axis=0),
    )
    np.testing.assert_allclose(norm_obs[1:, OWN_SLICE], uav_expected)

    np.testing.assert_allclose(norm_share[0], norm_share[-1])
    np.testing.assert_allclose(norm_share[0, :PHYSICAL_PUBLIC_STATE_DIM], public_expected)
    np.testing.assert_allclose(
        norm_share[0, PHYSICAL_PUBLIC_STATE_DIM:PHYSICAL_PUBLIC_STATE_DIM + 6],
        RESOURCE,
    )
    np.testing.assert_allclose(norm_share[0, PHYSICAL_PUBLIC_STATE_DIM + 6:].reshape(2, 3), uav_expected)


def test_component_update_uses_one_public_sample_and_all_uavs():
    normalizer = MECComponentNormalizer(NUM_AGENTS)
    obs = np.stack([_obs(), _obs(public=PUBLIC + 10, uavs=UAVS + 10)])

    normalizer.update(obs, repeat_team_state(obs))

    assert normalizer.physical_public_rms.count == pytest.approx(2.0001)
    assert normalizer.uav_state_rms.count == pytest.approx(4.0001)


class OneStepMECEnv:
    def __init__(self):
        obs_space = gym.spaces.Box(-np.inf, np.inf, shape=(AGENT_OBS_DIM,), dtype=np.float32)
        share_space = gym.spaces.Box(-np.inf, np.inf, shape=(team_state_dim(NUM_AGENTS),), dtype=np.float32)
        self.observation_space = [obs_space for _ in range(NUM_AGENTS)]
        self.share_observation_space = [share_space for _ in range(NUM_AGENTS)]
        self.action_space = [gym.spaces.Box(-1, 1, shape=(1,), dtype=np.float32) for _ in range(NUM_AGENTS)]

    def reset(self, *, seed=None, options=None):
        obs = _obs()
        return obs, repeat_team_state(obs), {"seed": seed}

    def step(self, action):
        obs = _obs(public=PUBLIC + 5, uavs=UAVS + 5)
        return (
            obs,
            repeat_team_state(obs),
            [[1.0] for _ in range(NUM_AGENTS)],
            [False] * NUM_AGENTS,
            [True] * NUM_AGENTS,
            [{"step": "final"} for _ in range(NUM_AGENTS)],
        )

    def close(self):
        pass


def test_component_vec_normalize_final_obs_save_load_and_mismatch():
    envs = MECComponentVecNormalize(ShareDummyVecEnv([OneStepMECEnv]))
    obs, share_obs, _ = envs.reset(seed=1)
    assert obs.shape == (1, NUM_AGENTS, AGENT_OBS_DIM)
    np.testing.assert_allclose(
        obs[0, :, RESOURCE_CONTEXT_SLICE],
        np.repeat(RESOURCE[None, :], NUM_AGENTS, axis=0),
    )

    _, _, _, _, _, infos = envs.step(np.zeros((1, NUM_AGENTS, 1), dtype=np.float32))
    assert "final_observation" in infos[0]
    assert "final_share_observation" in infos[0]
    np.testing.assert_allclose(
        infos[0]["final_observation"][:, RESOURCE_CONTEXT_SLICE],
        np.repeat(RESOURCE[None, :], NUM_AGENTS, axis=0),
    )
    np.testing.assert_allclose(
        infos[0]["final_share_observation"][0],
        infos[0]["final_share_observation"][-1],
    )

    raw_obs = _obs(public=PUBLIC + 2, uavs=UAVS + 2)
    with TemporaryDirectory() as tmp:
        path = f"{tmp}/vec_normalize.npz"
        envs.save_vec_normalize(path)
        loaded = MECComponentNormalizer(NUM_AGENTS)
        loaded.load(path, expected_num_agents=NUM_AGENTS)
        np.testing.assert_allclose(
            loaded.normalize_obs(raw_obs),
            envs.normalizer.normalize_obs(raw_obs),
        )
        with pytest.raises(ValueError, match="num_agents mismatch"):
            MECComponentNormalizer(NUM_AGENTS + 1).load(
                path,
                expected_num_agents=NUM_AGENTS + 1,
            )
