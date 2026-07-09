"""MAPPO policy wrapper for the MEC actor/critic networks."""

from __future__ import annotations

import torch

from onpolicy.algorithms.mec.mec_actor_critic import MECActor, MECCritic
from onpolicy.utils.util import update_linear_schedule


class MECPolicy:
    """MEC policy with the same public API expected by ``R_MAPPO``."""

    uses_grouped_batches = True

    def __init__(self, args, obs_space, cent_obs_space, act_space, device=torch.device("cpu"), num_agents=None):
        if num_agents is None:
            raise ValueError("MECPolicy requires num_agents")
        self.device = device
        self.lr = args.lr
        self.critic_lr = args.critic_lr
        self.actor = MECActor(args, obs_space, act_space, num_agents, device)
        actor_encoder = getattr(self.actor, "minor_encoder", None)
        self.critic = MECCritic(args, cent_obs_space, num_agents, device, actor_encoder=actor_encoder)
        self.recompute_values_after_actor_update = (
            getattr(args, "mec_policy_arch", None) == "slot_query"
            and getattr(args, "mec_slot_critic_encoder", "separate") == "shared_grad"
        )
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

    def mec_set_reconstruction_loss(self, cent_obs):
        return self.actor.reconstruction_loss(cent_obs)
