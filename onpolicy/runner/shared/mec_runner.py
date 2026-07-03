"""Shared-policy runner for the MEC environment."""

from __future__ import annotations

import time

import imageio
import numpy as np
import torch

from onpolicy.runner.shared.base_runner import Runner


def _t2n(x):
    return x.detach().cpu().numpy()


class MECRunner(Runner):
    """MEC runner with share_obs supplied by the environment."""

    def _critic_obs(self, obs, share_obs):
        return share_obs if self.use_centralized_V else obs

    def _actions_to_env(self, actions, envs):
        action_space = envs.action_space[0]
        if action_space.__class__.__name__ == "Box":
            return actions
        raise NotImplementedError("MEC expects continuous Box actions")

    def _actual_next_obs(self, obs, share_obs, infos):
        next_obs = obs.copy()
        next_share_obs = share_obs.copy()
        for env_id, info in enumerate(infos):
            if "final_observation" in info:
                next_obs[env_id] = info["final_observation"]
            if "final_share_observation" in info:
                next_share_obs[env_id] = info["final_share_observation"]
        return next_obs, next_share_obs

    @torch.no_grad()
    def _next_values(self, next_obs, next_share_obs, rnn_states_critic):
        self.trainer.prep_rollout()
        critic_obs = self._critic_obs(next_obs, next_share_obs)
        masks = np.ones(
            (self.n_rollout_threads, self.num_agents, 1), dtype=np.float32
        )
        if self.algorithm_name in ("mat", "mat_dec"):
            values = self.trainer.policy.get_values(
                np.concatenate(critic_obs),
                np.concatenate(next_obs),
                np.concatenate(rnn_states_critic),
                np.concatenate(masks),
            )
        else:
            values = self.trainer.policy.get_values(
                np.concatenate(critic_obs),
                np.concatenate(rnn_states_critic),
                np.concatenate(masks),
            )
        return np.array(np.split(_t2n(values), self.n_rollout_threads))

    def run(self):
        self.warmup()
        start = time.time()
        episodes = self.num_env_steps // self.episode_length // self.n_rollout_threads
        freeze_encoder_updates = int(
            getattr(
                self.all_args,
                "mec_set_freeze_pretrained_encoder_updates",
                0,
            )
            or 0
        )
        encoder_frozen = None

        for episode in range(episodes):
            if freeze_encoder_updates > 0 and hasattr(
                self.trainer.policy, "set_set_encoder_trainable"
            ):
                should_train_encoder = episode >= freeze_encoder_updates
                if encoder_frozen is not (not should_train_encoder):
                    changed = self.trainer.policy.set_set_encoder_trainable(
                        should_train_encoder
                    )
                    if changed:
                        state = "unfrozen" if should_train_encoder else "frozen"
                        print(
                            "MEC Set actor population encoder "
                            f"{state} at PPO update {episode}"
                        )
                    encoder_frozen = not should_train_encoder
            if self.use_linear_lr_decay:
                self.trainer.policy.lr_decay(episode, episodes)
            if getattr(self.all_args, "use_entropy_anneal", False):
                frac = max(0.0, 1.0 - episode / max(1, episodes))
                emin = getattr(self.all_args, "entropy_coef_min", 0.0)
                self.trainer.entropy_coef = (
                    emin + (self.all_args.entropy_coef - emin) * frac
                )

            for step in range(self.episode_length):
                rollout = self.collect(step)
                obs, share_obs, rewards, terminated, truncated, infos = self.envs.step(
                    rollout[-1]
                )
                self.insert(
                    (obs, share_obs, rewards, terminated, truncated, infos, *rollout[:-1])
                )

            self.compute()
            train_infos = self.train()
            total_num_steps = (
                (episode + 1) * self.episode_length * self.n_rollout_threads
            )

            update = episode + 1
            if update % self.save_interval == 0 or episode == episodes - 1:
                self.save(episode=episode, total_num_steps=total_num_steps)

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

            if (
                self.use_eval
                and (update % self.eval_interval == 0 or episode == episodes - 1)
            ):
                eval_reward = self.eval(total_num_steps)
                if self.maybe_save_best(eval_reward, total_num_steps):
                    print(
                        "new best fixed-validation reward "
                        f"{eval_reward:.6f} at {total_num_steps} steps"
                    )

    def warmup(self):
        obs, share_obs, _ = self.envs.reset(seed=self.all_args.seed)
        self.buffer.share_obs[0] = self._critic_obs(obs, share_obs)
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
            share_obs,
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

        self._last_infos = infos
        actual_next_obs, actual_next_share_obs = self._actual_next_obs(
            obs, share_obs, infos
        )
        next_values = self._next_values(
            actual_next_obs, actual_next_share_obs, rnn_states_critic
        )
        episode_dones = terminated | truncated

        rnn_states[episode_dones] = 0
        rnn_states_critic[episode_dones] = 0
        masks = (~episode_dones)[..., None].astype(np.float32)
        bootstrap_masks = (~terminated)[..., None].astype(np.float32)

        self.buffer.insert(
            self._critic_obs(obs, share_obs),
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
    def _eval_actions(self, obs, share_obs, rnn_states, masks, envs):
        self.trainer.prep_rollout()
        if self.algorithm_name in ("mat", "mat_dec"):
            action, rnn_states = self.trainer.policy.act(
                np.concatenate(self._critic_obs(obs, share_obs)),
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
        if hasattr(self.eval_envs, "copy_vec_normalize_from") and hasattr(
            self.envs, "obs_rms"
        ):
            self.eval_envs.copy_vec_normalize_from(self.envs, training=False)
        target_episodes = int(self.all_args.eval_episodes)
        completed_rewards = []
        batch = 0
        while len(completed_rewards) < target_episodes:
            batch_seed = (
                int(getattr(self.all_args, "eval_seed", 1000))
                + batch * self.n_eval_rollout_threads * 1000
            )
            obs, share_obs, _ = self.eval_envs.reset(seed=batch_seed)
            rnn_states = np.zeros(
                (
                    self.n_eval_rollout_threads,
                    *self.buffer.rnn_states.shape[2:],
                ),
                dtype=np.float32,
            )
            masks = np.ones(
                (
                    self.n_eval_rollout_threads,
                    self.num_agents,
                    1,
                ),
                dtype=np.float32,
            )
            episode_rewards = []

            for _ in range(self.episode_length):
                rnn_states, actions_env = self._eval_actions(
                    obs, share_obs, rnn_states, masks, self.eval_envs
                )
                obs, share_obs, rewards, terminated, truncated, _ = self.eval_envs.step(
                    actions_env
                )
                episode_rewards.append(rewards)
                episode_dones = terminated | truncated
                rnn_states[episode_dones] = 0
                masks = (~episode_dones)[..., None].astype(np.float32)

            batch_rewards = np.sum(np.asarray(episode_rewards), axis=0)
            completed_rewards.extend(np.mean(batch_rewards, axis=1).tolist())
            batch += 1

        average_reward = float(np.mean(completed_rewards[:target_episodes]))
        print(f"eval average episode rewards of agent: {average_reward}")
        self.log_env({"eval_average_episode_rewards": [average_reward]}, total_num_steps)
        return average_reward

    @torch.no_grad()
    def render(self):
        all_frames = []
        for episode in range(self.all_args.render_episodes):
            obs, share_obs, _ = self.envs.reset(seed=self.all_args.seed + episode)
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
                    obs, share_obs, rnn_states, masks, self.envs
                )
                obs, share_obs, rewards, terminated, truncated, _ = self.envs.step(
                    actions_env
                )
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

    def log_env(self, env_infos, total_num_steps):
        """Also surface team MEC metrics carried on the major agent's info slot."""
        is_training_snapshot = any(key.startswith("agent") for key in env_infos)
        if not is_training_snapshot:
            return super().log_env(env_infos, total_num_steps)

        keys = (
            "training_cost", "src_cost", "ovf_cost", "queue_cost",
            "energy_cost", "accepted", "offloaded", "overflow", "U_src",
            "access_utilization", "backhaul_utilization",
            "uav_compute_utilization", "hub_compute_utilization",
            "source_outside", "source_capacity",
            "hotspot_offered", "hotspot_accepted", "hotspot_source",
            "background_offered", "background_accepted", "background_source",
            "eta_p05", "eta_p50", "eta_p95", "eta_served", "eta_all",
            "n_hotspot_uav", "n_background_uav", "hub_to_hotspot", "w1",
        )
        for key in keys:
            vals = []
            for info in getattr(self, "_last_infos", []):
                major = info.get("agent_infos", [{}])[0]
                if key in major and major[key] is not None:
                    vals.append(major[key])
            if vals:
                env_infos[f"mec/{key}"] = vals
        self._print_mec_summary(env_infos)
        super().log_env(env_infos, total_num_steps)

    @staticmethod
    def _print_mec_summary(env_infos):
        def m(key):
            vals = env_infos.get(f"mec/{key}")
            return float(np.mean(vals)) if vals else float("nan")

        acc, usrc, total = m("accepted"), m("U_src"), m("training_cost")
        denom = acc + usrc
        accept_rate = 100.0 * acc / denom if denom > 0 else float("nan")
        share = lambda k: 100.0 * m(k) / total if total else float("nan")
        print(
            f"  [mec] accept={accept_rate:4.1f}%  accepted={acc/1e6:5.1f}  "
            f"ovf={m('overflow')/1e6:4.1f}  U_src={usrc/1e6:5.1f} "
            f"(out={m('source_outside')/1e6:4.1f}, cap={m('source_capacity')/1e6:4.1f})  "
            f"n_hot={m('n_hotspot_uav'):4.1f}  hub={m('hub_to_hotspot'):5.0f}m  "
            f"W1={m('w1'):6.0f}m  | shares src={share('src_cost'):3.0f}% "
            f"ovf={share('ovf_cost'):3.0f}% q={share('queue_cost'):3.0f}% "
            f"e={share('energy_cost'):3.0f}%"
        )
