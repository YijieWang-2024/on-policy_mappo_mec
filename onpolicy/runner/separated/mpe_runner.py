import time
from itertools import chain

import imageio
import numpy as np
import torch

from onpolicy.runner.separated.base_runner import Runner


def _t2n(x):
    return x.detach().cpu().numpy()


class MPERunner(Runner):
    """Separated-policy MPE runner with Gymnasium termination semantics."""

    def _central_obs(self, obs):
        return np.asarray([list(chain(*env_obs)) for env_obs in obs])

    def _actions_to_env(self, actions, envs):
        actions_env = [[] for _ in range(actions.shape[0])]
        for agent_id in range(self.num_agents):
            action = actions[:, agent_id]
            action_space = envs.action_space[agent_id]
            if action_space.__class__.__name__ == "MultiDiscrete":
                parts = [
                    np.eye(action_space.high[i] + 1)[action[:, i].astype(int)]
                    for i in range(action_space.shape)
                ]
                agent_actions = np.concatenate(parts, axis=1)
            elif action_space.__class__.__name__ == "Discrete":
                agent_actions = np.squeeze(
                    np.eye(action_space.n)[action.astype(int)], axis=1
                )
            else:
                raise NotImplementedError
            for env_id in range(actions.shape[0]):
                actions_env[env_id].append(agent_actions[env_id])
        return actions_env

    def _actual_next_obs(self, obs, infos):
        next_obs = obs.copy()
        for env_id, info in enumerate(infos):
            if "final_observation" in info:
                next_obs[env_id] = info["final_observation"]
        return next_obs

    @torch.no_grad()
    def _next_values(self, next_obs, rnn_states_critic):
        central_obs = self._central_obs(next_obs)
        masks = np.ones((self.n_rollout_threads, 1), dtype=np.float32)
        values = []
        for agent_id in range(self.num_agents):
            self.trainer[agent_id].prep_rollout()
            critic_obs = (
                central_obs if self.use_centralized_V else next_obs[:, agent_id]
            )
            value = self.trainer[agent_id].policy.get_values(
                critic_obs, rnn_states_critic[:, agent_id], masks
            )
            values.append(_t2n(value))
        return np.asarray(values).transpose(1, 0, 2)

    def run(self):
        self.warmup()
        start = time.time()
        episodes = self.num_env_steps // self.episode_length // self.n_rollout_threads

        for episode in range(episodes):
            if self.use_linear_lr_decay:
                for agent_id in range(self.num_agents):
                    self.trainer[agent_id].policy.lr_decay(episode, episodes)

            for step in range(self.episode_length):
                rollout = self.collect(step)
                obs, rewards, terminated, truncated, infos = self.envs.step(rollout[-1])
                self.insert((obs, rewards, terminated, truncated, infos, *rollout[:-1]))

            self.compute()
            train_infos = self.train()
            total_num_steps = (
                (episode + 1) * self.episode_length * self.n_rollout_threads
            )

            if episode % self.save_interval == 0 or episode == episodes - 1:
                self.save()

            if episode % self.log_interval == 0:
                elapsed = time.time() - start
                print(
                    f"\n Scenario {self.all_args.scenario_name} Algo {self.algorithm_name} "
                    f"Exp {self.experiment_name} updates {episode}/{episodes} episodes, "
                    f"total num timesteps {total_num_steps}/{self.num_env_steps}, "
                    f"FPS {int(total_num_steps / elapsed)}.\n"
                )
                for agent_id in range(self.num_agents):
                    individual_rewards = [
                        info["agent_infos"][agent_id].get("individual_reward", 0)
                        for info in infos
                    ]
                    train_infos[agent_id]["individual_rewards"] = np.mean(
                        individual_rewards
                    )
                    train_infos[agent_id]["average_episode_rewards"] = (
                        np.mean(self.buffer[agent_id].rewards) * self.episode_length
                    )
                    print(
                        f"average episode rewards of agent{agent_id} is "
                        f"{train_infos[agent_id]['average_episode_rewards']}"
                    )
                self.log_train(train_infos, total_num_steps)

            if episode % self.eval_interval == 0 and self.use_eval:
                self.eval(total_num_steps)

    def warmup(self):
        obs, _ = self.envs.reset(seed=self.all_args.seed)
        central_obs = self._central_obs(obs)
        for agent_id in range(self.num_agents):
            share_obs = central_obs if self.use_centralized_V else obs[:, agent_id]
            self.buffer[agent_id].share_obs[0] = share_obs
            self.buffer[agent_id].obs[0] = obs[:, agent_id]

    @torch.no_grad()
    def collect(self, step):
        values, actions, action_log_probs = [], [], []
        rnn_states, rnn_states_critic = [], []

        for agent_id in range(self.num_agents):
            self.trainer[agent_id].prep_rollout()
            value, action, action_log_prob, rnn_state, rnn_state_critic = (
                self.trainer[agent_id].policy.get_actions(
                    self.buffer[agent_id].share_obs[step],
                    self.buffer[agent_id].obs[step],
                    self.buffer[agent_id].rnn_states[step],
                    self.buffer[agent_id].rnn_states_critic[step],
                    self.buffer[agent_id].masks[step],
                )
            )
            values.append(_t2n(value))
            actions.append(_t2n(action))
            action_log_probs.append(_t2n(action_log_prob))
            rnn_states.append(_t2n(rnn_state))
            rnn_states_critic.append(_t2n(rnn_state_critic))

        values = np.asarray(values).transpose(1, 0, 2)
        actions = np.asarray(actions).transpose(1, 0, 2)
        action_log_probs = np.asarray(action_log_probs).transpose(1, 0, 2)
        rnn_states = np.asarray(rnn_states).transpose(1, 0, 2, 3)
        rnn_states_critic = np.asarray(rnn_states_critic).transpose(1, 0, 2, 3)
        return (
            values,
            actions,
            action_log_probs,
            rnn_states,
            rnn_states_critic,
            self._actions_to_env(actions, self.envs),
        )

    def insert(self, data):
        (
            obs,
            rewards,
            terminated,
            truncated,
            infos,
            values,
            actions,
            action_log_probs,
            rnn_states,
            rnn_states_critic,
        ) = data

        actual_next_obs = self._actual_next_obs(obs, infos)
        next_values = self._next_values(actual_next_obs, rnn_states_critic)
        episode_dones = terminated | truncated
        rnn_states[episode_dones] = 0
        rnn_states_critic[episode_dones] = 0
        masks = (~episode_dones)[..., None].astype(np.float32)
        bootstrap_masks = (~terminated)[..., None].astype(np.float32)
        central_obs = self._central_obs(obs)

        for agent_id in range(self.num_agents):
            share_obs = central_obs if self.use_centralized_V else obs[:, agent_id]
            self.buffer[agent_id].insert(
                share_obs,
                obs[:, agent_id],
                rnn_states[:, agent_id],
                rnn_states_critic[:, agent_id],
                actions[:, agent_id],
                action_log_probs[:, agent_id],
                values[:, agent_id],
                next_values[:, agent_id],
                rewards[:, agent_id],
                masks[:, agent_id],
                bootstrap_masks[:, agent_id],
            )

    @torch.no_grad()
    def _eval_actions(self, obs, rnn_states, masks, envs):
        actions = []
        for agent_id in range(self.num_agents):
            self.trainer[agent_id].prep_rollout()
            action, rnn_state = self.trainer[agent_id].policy.act(
                obs[:, agent_id],
                rnn_states[:, agent_id],
                masks[:, agent_id],
                deterministic=True,
            )
            actions.append(_t2n(action))
            rnn_states[:, agent_id] = _t2n(rnn_state)
        actions = np.asarray(actions).transpose(1, 0, 2)
        return rnn_states, self._actions_to_env(actions, envs)

    @torch.no_grad()
    def eval(self, total_num_steps):
        obs, _ = self.eval_envs.reset(seed=self.all_args.seed * 50000)
        rnn_states = np.zeros(
            (
                self.n_eval_rollout_threads,
                self.num_agents,
                self.recurrent_N,
                self.hidden_size,
            ),
            dtype=np.float32,
        )
        masks = np.ones(
            (self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32
        )
        episode_rewards = []

        for _ in range(self.episode_length):
            rnn_states, actions_env = self._eval_actions(
                obs, rnn_states, masks, self.eval_envs
            )
            obs, rewards, terminated, truncated, _ = self.eval_envs.step(actions_env)
            episode_rewards.append(rewards)
            episode_dones = terminated | truncated
            rnn_states[episode_dones] = 0
            masks = (~episode_dones)[..., None].astype(np.float32)

        episode_rewards = np.asarray(episode_rewards)
        train_infos = []
        for agent_id in range(self.num_agents):
            average_reward = np.mean(
                np.sum(episode_rewards[:, :, agent_id], axis=0)
            )
            train_infos.append({"eval_average_episode_rewards": average_reward})
            print(f"eval average episode rewards of agent{agent_id}: {average_reward}")
        self.log_train(train_infos, total_num_steps)

    @torch.no_grad()
    def render(self):
        all_frames = []
        for episode in range(self.all_args.render_episodes):
            obs, _ = self.envs.reset(seed=self.all_args.seed + episode)
            rnn_states = np.zeros(
                (
                    self.n_rollout_threads,
                    self.num_agents,
                    self.recurrent_N,
                    self.hidden_size,
                ),
                dtype=np.float32,
            )
            masks = np.ones(
                (self.n_rollout_threads, self.num_agents, 1), dtype=np.float32
            )
            episode_rewards = []

            for _ in range(self.episode_length):
                started = time.time()
                rnn_states, actions_env = self._eval_actions(
                    obs, rnn_states, masks, self.envs
                )
                obs, rewards, terminated, truncated, _ = self.envs.step(actions_env)
                episode_rewards.append(rewards)
                episode_dones = terminated | truncated
                rnn_states[episode_dones] = 0
                masks = (~episode_dones)[..., None].astype(np.float32)

                if self.all_args.save_gifs:
                    all_frames.append(self.envs.render("rgb_array")[0][0])
                    time.sleep(max(0, self.all_args.ifi - (time.time() - started)))

            episode_rewards = np.asarray(episode_rewards)
            for agent_id in range(self.num_agents):
                average_reward = np.mean(
                    np.sum(episode_rewards[:, :, agent_id], axis=0)
                )
                print(
                    f"eval average episode rewards of agent{agent_id}: {average_reward}"
                )

        if self.all_args.save_gifs:
            imageio.mimsave(
                str(self.gif_dir) + "/render.gif",
                all_frames,
                duration=self.all_args.ifi,
            )
