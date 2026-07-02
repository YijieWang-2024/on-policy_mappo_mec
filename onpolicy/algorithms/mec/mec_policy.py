"""Minimal major-minor MEC policy for the first MAPPO port.

The environment emits canonical local rows. Population descriptors are built
here so env dynamics stay independent of algorithm choices.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch.distributions import Beta, Normal

try:
    from gym import spaces
except Exception:  # pragma: no cover
    from gymnasium import spaces

from onpolicy.algorithms.utils.mlp import MLPBase
from onpolicy.algorithms.utils.popart import PopArt
from onpolicy.algorithms.utils.util import check, init
from onpolicy.envs.mec.observation import (
    AGENT_OBS_DIM,
    OWN_SLICE,
    PUBLIC_SLICE,
    PUBLIC_STATE_DIM,
    UAV_STATE_DIM,
    team_state_dim,
)
from onpolicy.utils.util import get_shape_from_obs_space, update_linear_schedule

_EPS = 1e-6
_BETA_EPS = 1e-4


def _mlp(args, input_dim):
    return MLPBase(args, (int(input_dim),))


def _softplus_inverse(x: float) -> float:
    if x <= 0.0:
        raise ValueError("softplus inverse input must be positive")
    return math.log(math.expm1(x))


def _init_beta_head(layer, target_alpha: float, target_beta: float):
    nn.init.constant_(layer.weight, 0.0)
    nn.init.constant_(layer.bias[0], _softplus_inverse(target_alpha - _BETA_EPS))
    nn.init.constant_(layer.bias[1], _softplus_inverse(target_beta - _BETA_EPS))
    return layer


class MECActor(nn.Module):
    VEL_DIM = 2
    ACT_DIM = 3

    def __init__(self, args, obs_space, action_space, num_agents, device):
        super().__init__()
        self.num_agents = int(num_agents)
        self.num_uavs = self.num_agents - 1
        self.obs_dim = int(get_shape_from_obs_space(obs_space)[0])
        self.arch = getattr(args, "mec_policy_arch", "mean")
        self.hidden_size = args.hidden_size
        self._use_policy_active_masks = args.use_policy_active_masks
        self._use_rolewise_loss = bool(getattr(args, "mec_rolewise_loss", False))
        self.tpdv = dict(dtype=torch.float32, device=device)

        if self.obs_dim != AGENT_OBS_DIM:
            raise ValueError(f"MEC obs dim must be {AGENT_OBS_DIM}, got {self.obs_dim}")
        if args.use_recurrent_policy or args.use_naive_recurrent_policy:
            raise NotImplementedError("MEC mean/flat policy is feed-forward only")
        if self.arch not in {"mean", "flat"}:
            raise ValueError("first MEC port supports --mec_policy_arch mean|flat")

        if self.arch == "mean":
            major_dim = PUBLIC_STATE_DIM + UAV_STATE_DIM
            minor_dim = UAV_STATE_DIM + PUBLIC_STATE_DIM + UAV_STATE_DIM
        else:
            flat_dim = PUBLIC_STATE_DIM + self.num_uavs * UAV_STATE_DIM
            major_dim = flat_dim
            minor_dim = UAV_STATE_DIM + flat_dim

        self.major_base = _mlp(args, major_dim)
        self.minor_base = _mlp(args, minor_dim)

        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][args.use_orthogonal]

        def init_(module, gain=0.01):
            return init(module, init_method, lambda x: nn.init.constant_(x, 0), gain=gain)

        logstd_init = float(getattr(args, "mec_logstd_init", -1.9))
        self.major_mean = init_(nn.Linear(self.hidden_size, self.VEL_DIM))
        self.major_logstd = nn.Parameter(torch.full((self.VEL_DIM,), logstd_init))
        self.minor_mean = init_(nn.Linear(self.hidden_size, self.VEL_DIM))
        self.minor_logstd = nn.Parameter(torch.full((self.VEL_DIM,), logstd_init))

        beta_alpha_init = float(getattr(args, "mec_beta_alpha_init", 2.0))
        beta_eta_init = float(getattr(args, "mec_beta_eta_init", 2.0))
        self.minor_beta = _init_beta_head(
            nn.Linear(self.hidden_size, 2),
            target_alpha=beta_alpha_init,
            target_beta=beta_eta_init,
        )
        self.to(device)

    def _team(self, obs):
        obs = check(obs).to(**self.tpdv)
        if obs.shape[-1] != self.obs_dim or obs.shape[0] % self.num_agents != 0:
            raise ValueError("MEC actor requires complete K+1 canonical obs rows")
        return obs.reshape(-1, self.num_agents, self.obs_dim)

    def _features(self, obs):
        team = self._team(obs)
        public = team[:, 0, PUBLIC_SLICE]
        uavs = team[:, 1:, OWN_SLICE]
        if self.arch == "mean":
            rep = uavs.mean(dim=1)
        else:
            rep = uavs.reshape(uavs.shape[0], -1)

        major_in = torch.cat([public, rep], dim=-1)
        minor_in = torch.cat(
            [
                uavs,
                public[:, None, :].expand(-1, self.num_uavs, -1),
                rep[:, None, :].expand(-1, self.num_uavs, -1),
            ],
            dim=-1,
        )
        major_feat = self.major_base(major_in).unsqueeze(1)
        minor_feat = self.minor_base(minor_in.reshape(-1, minor_in.shape[-1]))
        minor_feat = minor_feat.reshape(-1, self.num_uavs, self.hidden_size)
        return torch.cat([major_feat, minor_feat], dim=1).reshape(-1, self.hidden_size)

    def _dists(self, features):
        major_mean = self.major_mean(features)
        major_std = torch.exp(self.major_logstd).expand_as(major_mean)
        minor_mean = self.minor_mean(features)
        minor_std = torch.exp(self.minor_logstd).expand_as(minor_mean)
        ab = torch.nn.functional.softplus(self.minor_beta(features)) + _BETA_EPS
        return (
            Normal(major_mean, major_std),
            Normal(minor_mean, minor_std),
            Beta(ab[:, 0:1], ab[:, 1:2]),
        )

    def forward(self, obs, rnn_states, masks, available_actions=None, deterministic=False):
        rnn_states = check(rnn_states).to(**self.tpdv)
        features = self._features(obs)
        is_major = (check(obs).to(**self.tpdv)[:, 0:1] > 0.5).float()
        major, minor_v, minor_b = self._dists(features)

        v_major = major.mean if deterministic else major.rsample()
        v_minor = minor_v.mean if deterministic else minor_v.rsample()
        b_minor = minor_b.mean if deterministic else minor_b.rsample()

        vel = is_major * v_major + (1.0 - is_major) * v_minor
        beta = (1.0 - is_major) * b_minor
        actions = torch.cat([vel, beta], dim=-1)

        lp_major = major.log_prob(v_major).sum(-1, keepdim=True)
        lp_minor = minor_v.log_prob(v_minor).sum(-1, keepdim=True) + minor_b.log_prob(
            b_minor.clamp(_EPS, 1.0 - _EPS)
        )
        return actions, is_major * lp_major + (1.0 - is_major) * lp_minor, rnn_states

    def evaluate_actions(self, obs, rnn_states, action, masks, available_actions=None, active_masks=None):
        obs_t = check(obs).to(**self.tpdv)
        action = check(action).to(**self.tpdv)
        features = self._features(obs_t)
        is_major = (obs_t[:, 0:1] > 0.5).float()
        major, minor_v, minor_b = self._dists(features)

        vel = action[:, : self.VEL_DIM]
        beta = action[:, self.VEL_DIM : self.VEL_DIM + 1].clamp(_EPS, 1.0 - _EPS)
        lp_major = major.log_prob(vel).sum(-1, keepdim=True)
        lp_minor = minor_v.log_prob(vel).sum(-1, keepdim=True) + minor_b.log_prob(beta)
        ent_major = major.entropy().sum(-1, keepdim=True)
        ent_minor = minor_v.entropy().sum(-1, keepdim=True) + minor_b.entropy()
        action_log_probs = is_major * lp_major + (1.0 - is_major) * lp_minor
        entropy = is_major * ent_major + (1.0 - is_major) * ent_minor

        if active_masks is not None:
            active_masks = check(active_masks).to(**self.tpdv)
        if self._use_rolewise_loss:
            weights = active_masks if self._use_policy_active_masks and active_masks is not None else torch.ones_like(entropy)
            major_w = weights * is_major
            minor_w = weights * (1.0 - is_major)
            dist_entropy = 0.5 * (
                (entropy * major_w).sum() / major_w.sum().clamp_min(1.0)
                + (entropy * minor_w).sum() / minor_w.sum().clamp_min(1.0)
            )
        elif self._use_policy_active_masks and active_masks is not None:
            dist_entropy = (entropy * active_masks).sum() / active_masks.sum()
        else:
            dist_entropy = entropy.mean()
        return action_log_probs, dist_entropy


class MECCritic(nn.Module):
    def __init__(self, args, cent_obs_space, num_agents, device):
        super().__init__()
        self.num_agents = int(num_agents)
        self.num_uavs = self.num_agents - 1
        self.state_dim = int(get_shape_from_obs_space(cent_obs_space)[0])
        self.arch = getattr(args, "mec_policy_arch", "mean")
        self.hidden_size = args.hidden_size
        self._use_popart = args.use_popart
        self.tpdv = dict(dtype=torch.float32, device=device)

        if self.state_dim != team_state_dim(self.num_agents):
            raise ValueError("MEC critic requires canonical team state")
        input_dim = PUBLIC_STATE_DIM + (UAV_STATE_DIM if self.arch == "mean" else self.num_uavs * UAV_STATE_DIM)
        self.base = _mlp(args, input_dim)

        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][args.use_orthogonal]

        def init_(module):
            return init(module, init_method, lambda x: nn.init.constant_(x, 0))

        self.v_out = init_(PopArt(self.hidden_size, 1, device=device) if self._use_popart else nn.Linear(self.hidden_size, 1))
        self.to(device)

    def _features(self, cent_obs):
        cent_obs = check(cent_obs).to(**self.tpdv)
        public = cent_obs[:, :PUBLIC_STATE_DIM]
        uavs = cent_obs[:, PUBLIC_STATE_DIM:].reshape(-1, self.num_uavs, UAV_STATE_DIM)
        rep = uavs.mean(dim=1) if self.arch == "mean" else uavs.reshape(uavs.shape[0], -1)
        return self.base(torch.cat([public, rep], dim=-1))

    def forward(self, cent_obs, rnn_states, masks):
        rnn_states = check(rnn_states).to(**self.tpdv)
        return self.v_out(self._features(cent_obs)), rnn_states


class MECPolicy:
    uses_grouped_batches = True

    def __init__(self, args, obs_space, cent_obs_space, act_space, device=torch.device("cpu"), num_agents=None):
        if num_agents is None:
            raise ValueError("MECPolicy requires num_agents")
        self.device = device
        self.lr = args.lr
        self.critic_lr = args.critic_lr
        self.actor = MECActor(args, obs_space, act_space, num_agents, device)
        self.critic = MECCritic(args, cent_obs_space, num_agents, device)
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=self.lr, eps=args.opti_eps, weight_decay=args.weight_decay
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=self.critic_lr, eps=args.opti_eps, weight_decay=args.weight_decay
        )

    def lr_decay(self, episode, episodes):
        update_linear_schedule(self.actor_optimizer, episode, episodes, self.lr)
        update_linear_schedule(self.critic_optimizer, episode, episodes, self.critic_lr)

    def get_actions(self, cent_obs, obs, rnn_states_actor, rnn_states_critic, masks, available_actions=None, deterministic=False):
        actions, action_log_probs, rnn_states_actor = self.actor(
            obs, rnn_states_actor, masks, available_actions, deterministic
        )
        values, rnn_states_critic = self.critic(cent_obs, rnn_states_critic, masks)
        return values, actions, action_log_probs, rnn_states_actor, rnn_states_critic

    def get_values(self, cent_obs, rnn_states_critic, masks):
        values, _ = self.critic(cent_obs, rnn_states_critic, masks)
        return values

    def evaluate_actions(self, cent_obs, obs, rnn_states_actor, rnn_states_critic, action, masks, available_actions=None, active_masks=None):
        action_log_probs, dist_entropy = self.actor.evaluate_actions(
            obs, rnn_states_actor, action, masks, available_actions, active_masks
        )
        values, _ = self.critic(cent_obs, rnn_states_critic, masks)
        return values, action_log_probs, dist_entropy

    def act(self, obs, rnn_states_actor, masks, available_actions=None, deterministic=False):
        actions, _, rnn_states_actor = self.actor(
            obs, rnn_states_actor, masks, available_actions, deterministic
        )
        return actions, rnn_states_actor
