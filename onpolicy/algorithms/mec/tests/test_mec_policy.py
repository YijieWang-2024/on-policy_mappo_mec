from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch

from onpolicy.algorithms.mec.mec_actor_critic import _anchor_pool, _pooled_rep, _pool_dim
from onpolicy.algorithms.mec.mec_policy import MECPolicy
from onpolicy.algorithms.r_mappo.r_mappo import R_MAPPO
from onpolicy.config import get_config
from onpolicy.envs.mec.MEC_env import ACT_DIM, MECEnv
from onpolicy.envs.mec.observation import (
    AGENT_OBS_DIM,
    OWN_SLICE,
    PUBLIC_STATE_DIM,
    PUBLIC_SLICE,
    repeat_team_state,
    team_state_dim,
)
from onpolicy.utils.shared_buffer import SharedReplayBuffer


def _args(**overrides):
    args = get_config().parse_args([])
    args.env_name = "MEC"
    args.algorithm_name = "mappo"
    args.use_recurrent_policy = False
    args.use_naive_recurrent_policy = False
    args.mec_policy_arch = "mean"
    args.mec_scenario = "v6_hap_loadbearing"
    args.mec_fleet_size = 4
    args.mec_episode_horizon = 2
    args.episode_length = 2
    args.n_rollout_threads = 2
    args.hidden_size = 16
    args.layer_N = 1
    args.recurrent_N = 1
    args.num_mini_batch = 1
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def test_mec_env_adapter_and_team_state_shapes():
    env = MECEnv(_args())
    obs, share, _ = env.reset(seed=1)
    assert obs.shape == (5, AGENT_OBS_DIM)
    assert env.action_space[0].shape == (ACT_DIM,)

    expected_share = repeat_team_state(obs)
    assert share.shape == (5, team_state_dim(env.num_agents))
    np.testing.assert_allclose(share, expected_share)
    np.testing.assert_allclose(share[0], share[-1])

    action = np.zeros((env.num_agents, ACT_DIM), dtype=np.float32)
    next_obs, next_share, rewards, terminated, truncated, infos = env.step(action)
    assert next_obs.shape == obs.shape
    assert next_share.shape == share.shape
    assert len(rewards) == env.num_agents
    assert terminated == [False] * env.num_agents
    assert isinstance(truncated[0], bool)
    assert "accepted" in infos[0]


def test_mec_policy_arch_forward_shapes():
    for arch in ("mean", "flat", "hotspot_pool", "anchor_pool", "grid_pool", "csd_pool", "slot_query", "mhd_deepsets"):
        args = _args(mec_policy_arch=arch)
        env = MECEnv(args)
        args.num_agents = env.num_agents
        obs, share, _ = env.reset(seed=2)
        policy = MECPolicy(
            args,
            env.observation_space[0],
            env.share_observation_space[0],
            env.action_space[0],
            torch.device("cpu"),
            num_agents=env.num_agents,
        )
        rnn = np.zeros((env.num_agents, args.recurrent_N, args.hidden_size), dtype=np.float32)
        masks = np.ones((env.num_agents, 1), dtype=np.float32)
        values, actions, log_probs, _, _ = policy.get_actions(share, obs, rnn, rnn, masks)
        assert values.shape == (env.num_agents, 1)
        assert actions.shape == (env.num_agents, ACT_DIM)
        assert log_probs.shape == (env.num_agents, 1)

        eval_values, eval_log_probs, entropy = policy.evaluate_actions(
            share, obs, rnn, rnn, actions.detach().numpy(), masks
        )
        assert eval_values.shape == (env.num_agents, 1)
        assert eval_log_probs.shape == (env.num_agents, 1)
        assert torch.isfinite(entropy)


def test_slot_query_policy_is_permutation_equivariant():
    args = _args(mec_policy_arch="slot_query", mec_slot_actor_share_encoder=True)
    env = MECEnv(args)
    args.num_agents = env.num_agents
    obs, share, _ = env.reset(seed=7)
    policy = MECPolicy(
        args,
        env.observation_space[0],
        env.share_observation_space[0],
        env.action_space[0],
        torch.device("cpu"),
        num_agents=env.num_agents,
    )
    rnn = np.zeros((env.num_agents, args.recurrent_N, args.hidden_size), dtype=np.float32)
    masks = np.ones((env.num_agents, 1), dtype=np.float32)

    perm = np.array([2, 0, 3, 1])
    obs_perm = obs.copy()
    obs_perm[1:] = obs[1:][perm]
    share_perm = share.copy()
    uavs = share[0, PUBLIC_STATE_DIM:].reshape(env.num_agents - 1, -1)
    share_perm[:, PUBLIC_STATE_DIM:] = uavs[perm].reshape(1, -1)

    values, actions, _, _, _ = policy.get_actions(share, obs, rnn, rnn, masks, deterministic=True)
    values_perm, actions_perm, _, _, _ = policy.get_actions(
        share_perm, obs_perm, rnn, rnn, masks, deterministic=True
    )
    torch.testing.assert_close(actions_perm[0], actions[0], atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(actions_perm[1:], actions[1:][perm], atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(values_perm, values, atol=1e-5, rtol=1e-5)


def test_mhd_deepsets_policy_is_permutation_equivariant():
    args = _args(
        mec_policy_arch="mhd_deepsets",
        mec_critic_arch="mean",
        mec_deepsets_features="ref",
        mec_scenario="v7_random_split_hotspots",
    )
    env = MECEnv(args)
    args.num_agents = env.num_agents
    obs, share, _ = env.reset(seed=17)
    policy = MECPolicy(
        args,
        env.observation_space[0],
        env.share_observation_space[0],
        env.action_space[0],
        torch.device("cpu"),
        num_agents=env.num_agents,
    )
    rnn = np.zeros((env.num_agents, args.recurrent_N, args.hidden_size), dtype=np.float32)
    masks = np.ones((env.num_agents, 1), dtype=np.float32)

    perm = np.array([2, 0, 3, 1])
    obs_perm = obs.copy()
    obs_perm[1:] = obs[1:][perm]
    share_perm = share.copy()
    uavs = share[0, PUBLIC_STATE_DIM:].reshape(env.num_agents - 1, -1)
    share_perm[:, PUBLIC_STATE_DIM:] = uavs[perm].reshape(1, -1)

    values, actions, _, _, _ = policy.get_actions(share, obs, rnn, rnn, masks, deterministic=True)
    values_perm, actions_perm, _, _, _ = policy.get_actions(
        share_perm, obs_perm, rnn, rnn, masks, deterministic=True
    )
    torch.testing.assert_close(actions_perm[0], actions[0], atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(actions_perm[1:], actions[1:][perm], atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(values_perm, values, atol=1e-5, rtol=1e-5)


def test_slot_query_separate_critic_does_not_backprop_to_actor_encoder():
    args = _args(mec_policy_arch="slot_query", mec_slot_critic_encoder="separate")
    env = MECEnv(args)
    policy = MECPolicy(
        args,
        env.observation_space[0],
        env.share_observation_space[0],
        env.action_space[0],
        torch.device("cpu"),
        num_agents=env.num_agents,
    )
    _, share, _ = env.reset(seed=8)
    rnn = np.zeros((env.num_agents, args.recurrent_N, args.hidden_size), dtype=np.float32)
    masks = np.ones((env.num_agents, 1), dtype=np.float32)
    values = policy.get_values(share, rnn, masks)
    values.sum().backward()
    assert all(p.grad is None for p in policy.actor.minor_encoder.parameters())


def test_slot_query_reconstruction_loss_is_trainable():
    args = _args(mec_policy_arch="slot_query", mec_slot_actor_share_encoder=True)
    env = MECEnv(args)
    policy = MECPolicy(
        args,
        env.observation_space[0],
        env.share_observation_space[0],
        env.action_space[0],
        torch.device("cpu"),
        num_agents=env.num_agents,
    )
    _, share, _ = env.reset(seed=9)
    loss = policy.mec_set_reconstruction_loss(share)
    assert torch.isfinite(loss)
    loss.backward()
    assert any(p.grad is not None for p in policy.actor.reconstruction_decoder.parameters())


def test_anchor_pool_descriptor_shapes_padding_and_permutation_invariance():
    args = _args(mec_scenario="v7_random_split_hotspots", mec_policy_arch="hotspot_pool")
    env = MECEnv(args)
    obs, _, _ = env.reset(seed=5)
    public = torch.as_tensor(obs[0:1, PUBLIC_SLICE], dtype=torch.float32)
    uavs = torch.as_tensor(obs[1:, OWN_SLICE][None, :, :], dtype=torch.float32)

    hotspot_desc = _anchor_pool(public, uavs, "hotspot_pool")
    anchor_desc = _anchor_pool(public, uavs, "anchor_pool")
    grid_desc = _anchor_pool(public, uavs, "grid_pool")
    csd_desc = _pooled_rep(public, uavs, "csd_pool")
    assert hotspot_desc.shape[-1] == _pool_dim("hotspot_pool") == 16
    assert anchor_desc.shape[-1] == _pool_dim("anchor_pool") == 36
    assert grid_desc.shape[-1] == _pool_dim("grid_pool") == 52
    assert csd_desc.shape[-1] == _pool_dim("csd_pool") == 42
    torch.testing.assert_close(hotspot_desc.reshape(1, 4, 4)[:, 2:], torch.zeros(1, 2, 4))

    perm = torch.tensor([2, 0, 3, 1])
    torch.testing.assert_close(hotspot_desc, _anchor_pool(public, uavs[:, perm], "hotspot_pool"))
    torch.testing.assert_close(anchor_desc, _anchor_pool(public, uavs[:, perm], "anchor_pool"))
    torch.testing.assert_close(grid_desc, _anchor_pool(public, uavs[:, perm], "grid_pool"))
    torch.testing.assert_close(csd_desc, _pooled_rep(public, uavs[:, perm], "csd_pool"))


def test_beta_head_initializes_to_requested_distribution():
    args = _args(mec_beta_alpha_init=2.0, mec_beta_eta_init=2.0)
    env = MECEnv(args)
    policy = MECPolicy(
        args,
        env.observation_space[0],
        env.share_observation_space[0],
        env.action_space[0],
        torch.device("cpu"),
        num_agents=env.num_agents,
    )
    obs, _, _ = env.reset(seed=3)
    features = policy.actor._features(obs)
    _, _, beta_dist = policy.actor._dists(features)
    alpha = beta_dist.concentration1[1:]
    eta = beta_dist.concentration0[1:]
    torch.testing.assert_close(alpha, torch.full_like(alpha, 2.0), atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(eta, torch.full_like(eta, 2.0), atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(beta_dist.mean[1:], torch.full_like(beta_dist.mean[1:], 0.5))


def test_grouped_generator_preserves_complete_team_rows():
    args = _args()
    env = MECEnv(args)
    buffer = SharedReplayBuffer(
        args,
        env.num_agents,
        env.observation_space[0],
        env.share_observation_space[0],
        env.action_space[0],
    )
    for t in range(args.episode_length):
        for e in range(args.n_rollout_threads):
            marker = 10 * t + e
            buffer.obs[t, e, :, 4] = marker
            buffer.share_obs[t, e, :, 0] = marker

    advantages = np.ones_like(buffer.rewards)
    batches = list(buffer.feed_forward_generator_transformer(advantages, num_mini_batch=1))
    obs_batch = batches[0][1].reshape(-1, env.num_agents, AGENT_OBS_DIM)
    for team in obs_batch:
        assert np.unique(team[:, 4]).size == 1


def test_rolewise_policy_loss_handles_imbalanced_roles():
    args = _args(mec_rolewise_loss=True, use_valuenorm=False, use_popart=False)
    dummy_policy = SimpleNamespace(uses_grouped_batches=False)
    trainer = R_MAPPO(args, dummy_policy, torch.device("cpu"))

    surrogate = torch.ones(10, 1)
    active_masks = torch.ones(10, 1)
    obs = np.zeros((10, AGENT_OBS_DIM), dtype=np.float32)
    obs[0, 0] = 1.0

    loss, info = trainer._policy_action_loss(surrogate, active_masks, obs)
    assert torch.isfinite(loss)
    assert info["major_policy_loss"] == 1.0
    assert info["minor_policy_loss"] == 1.0
