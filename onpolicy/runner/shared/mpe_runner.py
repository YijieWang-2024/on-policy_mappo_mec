import time

import imageio
import numpy as np
import torch

from onpolicy.runner.shared.base_runner import Runner


def _t2n(x):
    return x.detach().cpu().numpy()


class MPERunner(Runner):
    """Shared-policy MPE runner with Gymnasium termination semantics."""

    def _share_obs(self, obs):
        if not self.use_centralized_V:
            return obs
        share_obs = obs.reshape(obs.shape[0], -1)
        return np.expand_dims(share_obs, 1).repeat(self.num_agents, axis=1)

    def _actions_to_env(self, actions, envs):
        action_space = envs.action_space[0]
        if action_space.__class__.__name__ == "MultiDiscrete":
            parts = [
                np.eye(action_space.high[i] + 1)[actions[:, :, i].astype(int)]
                for i in range(action_space.shape)
            ]
            return np.concatenate(parts, axis=2)
        if action_space.__class__.__name__ == "Discrete":
            return np.squeeze(
                np.eye(action_space.n)[actions.astype(int)], axis=2
            )
        raise NotImplementedError

    def _actual_next_obs(self, obs, infos):
        next_obs = obs.copy()
        for env_id, info in enumerate(infos):
            if "final_observation" in info:
                next_obs[env_id] = info["final_observation"]
        return next_obs

    @torch.no_grad()
    def _next_values(self, next_obs, rnn_states_critic):
        self.trainer.prep_rollout()
        next_share_obs = self._share_obs(next_obs)
        masks = np.ones(
            (self.n_rollout_threads, self.num_agents, 1), dtype=np.float32
        )
        if self.algorithm_name in ("mat", "mat_dec"):
            values = self.trainer.policy.get_values(
                np.concatenate(next_share_obs),
                np.concatenate(next_obs),
                np.concatenate(rnn_states_critic),
                np.concatenate(masks),
            )
        else:
            values = self.trainer.policy.get_values(
                np.concatenate(next_share_obs),
                np.concatenate(rnn_states_critic),
                np.concatenate(masks),
            )
        return np.array(np.split(_t2n(values), self.n_rollout_threads))

    def run(self):
        self.warmup()
        start = time.time()
        episodes = self.num_env_steps // self.episode_length // self.n_rollout_threads

        for episode in range(episodes):
            if self.use_linear_lr_decay:
                self.trainer.policy.lr_decay(episode, episodes)

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
                env_infos = {}
                for agent_id in range(self.num_agents):
                    rewards_for_agent = [
                        info["agent_infos"][agent_id]["individual_reward"]
                        for info in infos
                        if "individual_reward" in info["agent_infos"][agent_id]
                    ]
                    env_infos[f"agent{agent_id}/individual_rewards"] = rewards_for_agent
                train_infos["average_episode_rewards"] = (
                    np.mean(self.buffer.rewards) * self.episode_length
                )
                print(
                    "average episode rewards is "
                    f"{train_infos['average_episode_rewards']}"
                )
                self.log_train(train_infos, total_num_steps)
                self.log_env(env_infos, total_num_steps)

            if episode % self.eval_interval == 0 and self.use_eval:
                self.eval(total_num_steps)

    def warmup(self):
        obs, _ = self.envs.reset(seed=self.all_args.seed)
        self.buffer.share_obs[0] = self._share_obs(obs)
        self.buffer.obs[0] = obs

    @torch.no_grad()
    def collect(self, step):
        self.trainer.prep_rollout()
        value, action, action_log_prob, rnn_states, rnn_states_critic = (
            self.trainer.policy.get_actions(
                np.concatenate(self.buffer.share_obs[step]),
                np.concatenate(self.buffer.obs[step]),
                np.concatenate(self.buffer.rnn_states[step]),
                np.concatenate(self.buffer.rnn_states_critic[step]),
                np.concatenate(self.buffer.masks[step]),
            )
        )
        values = np.array(np.split(_t2n(value), self.n_rollout_threads))
        actions = np.array(np.split(_t2n(action), self.n_rollout_threads))
        action_log_probs = np.array(
            np.split(_t2n(action_log_prob), self.n_rollout_threads)
        )
        rnn_states = np.array(np.split(_t2n(rnn_states), self.n_rollout_threads))
        rnn_states_critic = np.array(
            np.split(_t2n(rnn_states_critic), self.n_rollout_threads)
        )
        actions_env = self._actions_to_env(actions, self.envs)
        return (
            values,
            actions,
            action_log_probs,
            rnn_states,
            rnn_states_critic,
            actions_env,
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

        self.buffer.insert(
            self._share_obs(obs),
            obs,
            rnn_states,
            rnn_states_critic,
            actions,
            action_log_probs,
            values,
            next_values,
            rewards,
            masks,
            bootstrap_masks,
        )

    @torch.no_grad()
    def _eval_actions(self, obs, rnn_states, masks, envs):
        self.trainer.prep_rollout()
        if self.algorithm_name in ("mat", "mat_dec"):
            action, rnn_states = self.trainer.policy.act(
                np.concatenate(self._share_obs(obs)),
                np.concatenate(obs),
                np.concatenate(rnn_states),
                np.concatenate(masks),
                deterministic=True,
            )
        else:
            action, rnn_states = self.trainer.policy.act(
                np.concatenate(obs),
                np.concatenate(rnn_states),
                np.concatenate(masks),
                deterministic=True,
            )
        actions = np.array(np.split(_t2n(action), obs.shape[0]))
        rnn_states = np.array(np.split(_t2n(rnn_states), obs.shape[0]))
        return rnn_states, self._actions_to_env(actions, envs)

    @torch.no_grad()
    def eval(self, total_num_steps):
        obs, _ = self.eval_envs.reset(seed=self.all_args.seed * 50000)
        rnn_states = np.zeros(
            (self.n_eval_rollout_threads, *self.buffer.rnn_states.shape[2:]),
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

        average_reward = np.mean(np.sum(np.asarray(episode_rewards), axis=0))
        print(f"eval average episode rewards of agent: {average_reward}")
        self.log_env({"eval_average_episode_rewards": [average_reward]}, total_num_steps)

    @torch.no_grad()
    def render(self):
        all_frames = []
        for episode in range(self.all_args.render_episodes):
            obs, _ = self.envs.reset(seed=self.all_args.seed + episode)
            rnn_states = np.zeros(
                (self.n_rollout_threads, *self.buffer.rnn_states.shape[2:]),
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
                else:
                    self.envs.render("human")

            print(
                "average episode rewards is: "
                + str(np.mean(np.sum(np.asarray(episode_rewards), axis=0)))
            )

        if self.all_args.save_gifs:
            imageio.mimsave(
                str(self.gif_dir) + "/render.gif",
                all_frames,
                duration=self.all_args.ifi,
            )
