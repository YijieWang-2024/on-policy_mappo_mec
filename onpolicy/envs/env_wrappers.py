"""Minimal Gymnasium-compatible vector environments for MPE."""

from abc import ABC, abstractmethod
from multiprocessing import Pipe, Process

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


def _autoreset(env, result):
    obs, rewards, terminated, truncated, agent_infos = result
    obs = np.asarray(obs)
    rewards = np.asarray(rewards)
    terminated = np.asarray(terminated, dtype=bool)
    truncated = np.asarray(truncated, dtype=bool)
    info = {"agent_infos": agent_infos}

    if np.all(terminated | truncated):
        final_observation = obs.copy()
        final_info = agent_infos
        obs, reset_info = env.reset()
        obs = np.asarray(obs)
        info.update(
            final_observation=final_observation,
            final_info=final_info,
            reset_info=reset_info,
        )

    return obs, rewards, terminated, truncated, info


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
    """Vector environment for a batch of multi-agent MPE environments."""

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
                remote.send(_autoreset(env, env.step(data)))
            elif command == "reset":
                remote.send(env.reset(**data))
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
    """Run each MPE environment in its own subprocess."""

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
        super().__init__(
            len(env_fns), observation_space, share_observation_space, action_space
        )

    def step_async(self, actions):
        for remote, action in zip(self.remotes, actions):
            remote.send(("step", action))
        self.waiting = True

    def step_wait(self):
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        obs, rewards, terminated, truncated, infos = zip(*results)
        return (
            np.stack(obs),
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


class DummyVecEnv(ShareVecEnv):
    """Run MPE environments sequentially in the current process."""

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
            _autoreset(env, env.step(action))
            for action, env in zip(self.actions, self.envs)
        ]
        self.actions = None
        obs, rewards, terminated, truncated, infos = zip(*results)
        return (
            np.stack(obs),
            np.stack(rewards),
            np.stack(terminated),
            np.stack(truncated),
            infos,
        )

    def reset(self, seed=None, options=None):
        seeds = _expand_seeds(seed, self.num_envs)
        results = [
            env.reset(seed=env_seed, options=options)
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
