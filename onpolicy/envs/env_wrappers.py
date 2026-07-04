"""Gymnasium-compatible vector environments for multi-agent tasks."""

from __future__ import annotations

from abc import ABC, abstractmethod
from multiprocessing import Pipe, Process
from pathlib import Path

import cloudpickle
import numpy as np


def _expand_seeds(seed, num_envs):
    if seed is None:
        return [None] * num_envs
    if np.isscalar(seed):
        return [int(seed) + rank * 1000 for rank in range(num_envs)]
    seeds = list(seed)
    if len(seeds) != num_envs:
        raise ValueError(f"Expected {num_envs} seeds, got {len(seeds)}")
    return seeds


def _reset_obs(env, seed=None, options=None):
    result = env.reset(seed=seed, options=options)
    if isinstance(result, tuple) and len(result) == 2:
        return result
    return result, {}


def _reset_share(env, seed=None, options=None):
    result = env.reset(seed=seed, options=options)
    if not (isinstance(result, tuple) and len(result) == 3):
        raise ValueError("share-aware env reset must return obs, share_obs, info")
    return result


def _autoreset_obs(env, result):
    obs, rewards, terminated, truncated, agent_infos = result
    obs = np.asarray(obs)
    rewards = np.asarray(rewards)
    terminated = np.asarray(terminated, dtype=bool)
    truncated = np.asarray(truncated, dtype=bool)
    info = {"agent_infos": agent_infos}

    if np.all(terminated | truncated):
        final_observation = obs.copy()
        final_info = agent_infos
        obs, reset_info = _reset_obs(env)
        obs = np.asarray(obs)
        info.update(
            final_observation=final_observation,
            final_info=final_info,
            reset_info=reset_info,
        )

    return obs, rewards, terminated, truncated, info


def _autoreset_share(env, result):
    obs, share_obs, rewards, terminated, truncated, agent_infos = result
    obs = np.asarray(obs)
    share_obs = np.asarray(share_obs)
    rewards = np.asarray(rewards)
    terminated = np.asarray(terminated, dtype=bool)
    truncated = np.asarray(truncated, dtype=bool)
    info = {"agent_infos": agent_infos}

    if np.all(terminated | truncated):
        final_observation = obs.copy()
        final_share_observation = share_obs.copy()
        final_info = agent_infos
        obs, share_obs, reset_info = _reset_share(env)
        obs = np.asarray(obs)
        share_obs = np.asarray(share_obs)
        info.update(
            final_observation=final_observation,
            final_share_observation=final_share_observation,
            final_info=final_info,
            reset_info=reset_info,
        )

    return obs, share_obs, rewards, terminated, truncated, info


class CloudpickleWrapper:
    """Serialize environment factories for subprocess workers."""

    def __init__(self, value):
        self.value = value

    def __getstate__(self):
        return cloudpickle.dumps(self.value)

    def __setstate__(self, value):
        import pickle

        self.value = pickle.loads(value)


class ShareVecEnv(ABC):
    """Base vector env with per-agent obs and optional centralized state."""

    def __init__(self, num_envs, observation_space, share_observation_space, action_space):
        self.num_envs = num_envs
        self.observation_space = observation_space
        self.share_observation_space = share_observation_space
        self.action_space = action_space
        self.closed = False

    @abstractmethod
    def reset(self, seed=None, options=None):
        pass

    @abstractmethod
    def step_async(self, actions):
        pass

    @abstractmethod
    def step_wait(self):
        pass

    def step(self, actions):
        self.step_async(actions)
        return self.step_wait()

    def render(self, mode="human"):
        return [env.render(mode=mode) for env in self.envs]

    @abstractmethod
    def close(self):
        pass


def _worker(remote, parent_remote, env_fn_wrapper):
    parent_remote.close()
    env = env_fn_wrapper.value()
    try:
        while True:
            command, data = remote.recv()
            if command == "step":
                remote.send(_autoreset_obs(env, env.step(data)))
            elif command == "reset":
                remote.send(_reset_obs(env, **data))
            elif command == "render":
                remote.send(env.render(mode=data))
            elif command == "get_spaces":
                remote.send(
                    (env.observation_space, env.share_observation_space, env.action_space)
                )
            elif command == "close":
                env.close()
                remote.close()
                break
            else:
                raise NotImplementedError(command)
    except EOFError:
        env.close()


def shareworker(remote, parent_remote, env_fn_wrapper):
    parent_remote.close()
    env = env_fn_wrapper.value()
    try:
        while True:
            command, data = remote.recv()
            if command == "step":
                remote.send(_autoreset_share(env, env.step(data)))
            elif command == "reset":
                remote.send(_reset_share(env, **data))
            elif command == "render":
                remote.send(env.render(mode=data))
            elif command == "get_spaces":
                remote.send(
                    (env.observation_space, env.share_observation_space, env.action_space)
                )
            elif command == "close":
                env.close()
                remote.close()
                break
            else:
                raise NotImplementedError(command)
    except EOFError:
        env.close()


class SubprocVecEnv(ShareVecEnv):
    """Run obs-only environments in subprocesses."""

    def __init__(self, env_fns):
        self.waiting = False
        self.remotes, work_remotes = zip(*[Pipe() for _ in env_fns])
        self.processes = [
            Process(target=_worker, args=(work_remote, remote, CloudpickleWrapper(env_fn)))
            for work_remote, remote, env_fn in zip(work_remotes, self.remotes, env_fns)
        ]
        for process in self.processes:
            process.daemon = True
            process.start()
        for work_remote in work_remotes:
            work_remote.close()

        self.remotes[0].send(("get_spaces", None))
        observation_space, share_observation_space, action_space = self.remotes[0].recv()
        super().__init__(len(env_fns), observation_space, share_observation_space, action_space)

    def step_async(self, actions):
        for remote, action in zip(self.remotes, actions):
            remote.send(("step", action))
        self.waiting = True

    def step_wait(self):
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        obs, rewards, terminated, truncated, infos = zip(*results)
        return np.stack(obs), np.stack(rewards), np.stack(terminated), np.stack(truncated), infos

    def reset(self, seed=None, options=None):
        seeds = _expand_seeds(seed, self.num_envs)
        for remote, env_seed in zip(self.remotes, seeds):
            remote.send(("reset", {"seed": env_seed, "options": options}))
        results = [remote.recv() for remote in self.remotes]
        obs, infos = zip(*results)
        return np.stack(obs), infos

    def render(self, mode="human"):
        for remote in self.remotes:
            remote.send(("render", mode))
        frames = [remote.recv() for remote in self.remotes]
        if mode == "rgb_array":
            return np.asarray(frames)
        return frames

    def close(self):
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(("close", None))
        for process in self.processes:
            process.join()
        self.closed = True


class ShareSubprocVecEnv(SubprocVecEnv):
    """Run share-aware environments in subprocesses."""

    def __init__(self, env_fns):
        self.waiting = False
        self.remotes, work_remotes = zip(*[Pipe() for _ in env_fns])
        self.processes = [
            Process(target=shareworker, args=(work_remote, remote, CloudpickleWrapper(env_fn)))
            for work_remote, remote, env_fn in zip(work_remotes, self.remotes, env_fns)
        ]
        for process in self.processes:
            process.daemon = True
            process.start()
        for work_remote in work_remotes:
            work_remote.close()

        self.remotes[0].send(("get_spaces", None))
        observation_space, share_observation_space, action_space = self.remotes[0].recv()
        ShareVecEnv.__init__(
            self, len(env_fns), observation_space, share_observation_space, action_space
        )

    def step_wait(self):
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        obs, share_obs, rewards, terminated, truncated, infos = zip(*results)
        return (
            np.stack(obs),
            np.stack(share_obs),
            np.stack(rewards),
            np.stack(terminated),
            np.stack(truncated),
            infos,
        )

    def reset(self, seed=None, options=None):
        seeds = _expand_seeds(seed, self.num_envs)
        for remote, env_seed in zip(self.remotes, seeds):
            remote.send(("reset", {"seed": env_seed, "options": options}))
        results = [remote.recv() for remote in self.remotes]
        obs, share_obs, infos = zip(*results)
        return np.stack(obs), np.stack(share_obs), infos


class DummyVecEnv(ShareVecEnv):
    """Run obs-only environments sequentially in the current process."""

    def __init__(self, env_fns):
        self.envs = [env_fn() for env_fn in env_fns]
        env = self.envs[0]
        super().__init__(
            len(env_fns), env.observation_space, env.share_observation_space, env.action_space
        )
        self.actions = None

    def step_async(self, actions):
        self.actions = actions

    def step_wait(self):
        results = [
            _autoreset_obs(env, env.step(action))
            for action, env in zip(self.actions, self.envs)
        ]
        self.actions = None
        obs, rewards, terminated, truncated, infos = zip(*results)
        return np.stack(obs), np.stack(rewards), np.stack(terminated), np.stack(truncated), infos

    def reset(self, seed=None, options=None):
        seeds = _expand_seeds(seed, self.num_envs)
        results = [
            _reset_obs(env, seed=env_seed, options=options)
            for env, env_seed in zip(self.envs, seeds)
        ]
        obs, infos = zip(*results)
        return np.stack(obs), infos

    def render(self, mode="human"):
        frames = [env.render(mode=mode) for env in self.envs]
        if mode == "rgb_array":
            return np.asarray(frames)
        return frames

    def close(self):
        if self.closed:
            return
        for env in self.envs:
            env.close()
        self.closed = True


class ShareDummyVecEnv(DummyVecEnv):
    """Run share-aware environments sequentially in the current process."""

    def step_wait(self):
        results = [
            _autoreset_share(env, env.step(action))
            for action, env in zip(self.actions, self.envs)
        ]
        self.actions = None
        obs, share_obs, rewards, terminated, truncated, infos = zip(*results)
        return (
            np.stack(obs),
            np.stack(share_obs),
            np.stack(rewards),
            np.stack(terminated),
            np.stack(truncated),
            infos,
        )

    def reset(self, seed=None, options=None):
        seeds = _expand_seeds(seed, self.num_envs)
        results = [
            _reset_share(env, seed=env_seed, options=options)
            for env, env_seed in zip(self.envs, seeds)
        ]
        obs, share_obs, infos = zip(*results)
        return np.stack(obs), np.stack(share_obs), infos


class RunningMeanStd:
    """Numpy running mean/std with the same state shape as an observation."""

    def __init__(self, shape, epsilon=1e-4):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = float(epsilon)

    def update(self, values):
        values = np.asarray(values, dtype=np.float64)
        if values.size == 0:
            return
        batch_mean = values.mean(axis=0)
        batch_var = values.var(axis=0)
        batch_count = values.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        total_count = self.count + batch_count
        new_mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        correction = np.square(delta) * self.count * batch_count / total_count
        new_var = (m_a + m_b + correction) / total_count
        self.mean = new_mean
        self.var = np.maximum(new_var, 1e-12)
        self.count = float(total_count)

    def state_dict(self, prefix):
        return {
            f"{prefix}_mean": self.mean,
            f"{prefix}_var": self.var,
            f"{prefix}_count": np.asarray(self.count, dtype=np.float64),
        }

    def load_state_dict(self, data, prefix):
        self.mean = np.asarray(data[f"{prefix}_mean"], dtype=np.float64)
        self.var = np.asarray(data[f"{prefix}_var"], dtype=np.float64)
        self.count = float(np.asarray(data[f"{prefix}_count"]))


class ShareVecNormalize(ShareVecEnv):
    """Normalize obs and share_obs before they enter the runner/buffer."""

    def __init__(
        self,
        venv,
        *,
        training=True,
        clip_obs=10.0,
        epsilon=1e-8,
        obs_mask=None,
        share_obs_mask=None,
        share_obs_unique=False,
    ):
        super().__init__(
            venv.num_envs,
            venv.observation_space,
            venv.share_observation_space,
            venv.action_space,
        )
        self.venv = venv
        self.training = bool(training)
        self.clip_obs = float(clip_obs)
        self.epsilon = float(epsilon)
        self.share_obs_unique = bool(share_obs_unique)
        self.obs_rms = RunningMeanStd(self.observation_space[0].shape)
        self.share_obs_rms = RunningMeanStd(self.share_observation_space[0].shape)
        self.obs_mask = self._mask(obs_mask, self.observation_space[0].shape)
        self.share_obs_mask = self._mask(share_obs_mask, self.share_observation_space[0].shape)

    @staticmethod
    def _mask(mask, shape):
        if mask is None:
            return np.ones(shape, dtype=bool)
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != tuple(shape):
            raise ValueError(f"normalization mask shape {mask.shape} != {shape}")
        return mask

    def _obs_samples(self, obs):
        return np.asarray(obs).reshape(-1, *self.observation_space[0].shape)

    def _share_obs_samples(self, share_obs):
        share_obs = np.asarray(share_obs)
        if self.share_obs_unique and share_obs.ndim >= 3:
            share_obs = share_obs[:, 0, :]
        return share_obs.reshape(-1, *self.share_observation_space[0].shape)

    def _update(self, obs, share_obs):
        if not self.training:
            return
        self.obs_rms.update(self._obs_samples(obs))
        self.share_obs_rms.update(self._share_obs_samples(share_obs))

    def _normalize_array(self, values, rms, mask):
        values = np.asarray(values, dtype=np.float32)
        out = values.copy()
        normed = (values - rms.mean) / np.sqrt(rms.var + self.epsilon)
        normed = np.clip(normed, -self.clip_obs, self.clip_obs)
        out[..., mask] = normed[..., mask]
        return out.astype(np.float32)

    def normalize_obs(self, obs):
        return self._normalize_array(obs, self.obs_rms, self.obs_mask)

    def normalize_share_obs(self, share_obs):
        return self._normalize_array(share_obs, self.share_obs_rms, self.share_obs_mask)

    def _normalize_infos(self, infos):
        normalized = []
        for info in infos:
            info = dict(info)
            if "final_observation" in info:
                info["final_observation"] = self.normalize_obs(info["final_observation"])
            if "final_share_observation" in info:
                info["final_share_observation"] = self.normalize_share_obs(
                    info["final_share_observation"]
                )
            normalized.append(info)
        return tuple(normalized)

    def reset(self, seed=None, options=None):
        obs, share_obs, infos = self.venv.reset(seed=seed, options=options)
        self._update(obs, share_obs)
        return self.normalize_obs(obs), self.normalize_share_obs(share_obs), infos

    def step_async(self, actions):
        return self.venv.step_async(actions)

    def step_wait(self):
        obs, share_obs, rewards, terminated, truncated, infos = self.venv.step_wait()
        self._update(obs, share_obs)
        return (
            self.normalize_obs(obs),
            self.normalize_share_obs(share_obs),
            rewards,
            terminated,
            truncated,
            self._normalize_infos(infos),
        )

    def render(self, mode="human"):
        return self.venv.render(mode=mode)

    def close(self):
        self.venv.close()
        self.closed = True

    def copy_vec_normalize_from(self, other, *, training=None):
        source = other
        if not hasattr(source, "obs_rms") and hasattr(source, "venv"):
            source = source.venv
        if not hasattr(source, "obs_rms"):
            raise ValueError("source env has no vec-normalize statistics")
        self.obs_rms.mean = source.obs_rms.mean.copy()
        self.obs_rms.var = source.obs_rms.var.copy()
        self.obs_rms.count = float(source.obs_rms.count)
        self.share_obs_rms.mean = source.share_obs_rms.mean.copy()
        self.share_obs_rms.var = source.share_obs_rms.var.copy()
        self.share_obs_rms.count = float(source.share_obs_rms.count)
        self.clip_obs = float(source.clip_obs)
        self.epsilon = float(source.epsilon)
        self.obs_mask = source.obs_mask.copy()
        self.share_obs_mask = source.share_obs_mask.copy()
        self.share_obs_unique = bool(source.share_obs_unique)
        if training is not None:
            self.training = bool(training)

    def save_vec_normalize(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "norm_mode": np.asarray("flat"),
            "clip_obs": np.asarray(self.clip_obs, dtype=np.float64),
            "epsilon": np.asarray(self.epsilon, dtype=np.float64),
            "obs_mask": self.obs_mask,
            "share_obs_mask": self.share_obs_mask,
            "share_obs_unique": np.asarray(self.share_obs_unique, dtype=bool),
        }
        payload.update(self.obs_rms.state_dict("obs"))
        payload.update(self.share_obs_rms.state_dict("share_obs"))
        np.savez(path, **payload)

    def load_vec_normalize(self, path, *, training=None):
        data = np.load(Path(path), allow_pickle=False)
        self.obs_rms.load_state_dict(data, "obs")
        self.share_obs_rms.load_state_dict(data, "share_obs")
        self.clip_obs = float(np.asarray(data["clip_obs"]))
        self.epsilon = float(np.asarray(data["epsilon"]))
        if "obs_mask" in data:
            self.obs_mask = np.asarray(data["obs_mask"], dtype=bool)
        if "share_obs_mask" in data:
            self.share_obs_mask = np.asarray(data["share_obs_mask"], dtype=bool)
        if "share_obs_unique" in data:
            self.share_obs_unique = bool(np.asarray(data["share_obs_unique"]))
        if training is not None:
            self.training = bool(training)
