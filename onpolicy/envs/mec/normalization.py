"""MEC component-level obs/share_obs normalization."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from onpolicy.envs.env_wrappers import RunningMeanStd, ShareVecEnv
from onpolicy.envs.mec.observation import (
    OWN_SLICE,
    PHYSICAL_PUBLIC_SLICE,
    PHYSICAL_PUBLIC_STATE_DIM,
    PUBLIC_STATE_DIM,
    UAV_STATE_DIM,
)


class MECComponentNormalizer:
    """Normalize MEC physical components without changing obs/share_obs shape."""

    norm_mode = "component"

    def __init__(self, num_agents: int, *, clip_obs=10.0, epsilon=1e-8):
        self.num_agents = int(num_agents)
        self.num_uavs = self.num_agents - 1
        if self.num_uavs <= 0:
            raise ValueError("MEC component normalization requires at least one UAV")
        self.clip_obs = float(clip_obs)
        self.epsilon = float(epsilon)
        self.physical_public_rms = RunningMeanStd((PHYSICAL_PUBLIC_STATE_DIM,))
        self.uav_state_rms = RunningMeanStd((UAV_STATE_DIM,))

    def update(self, obs, share_obs=None):
        obs_b, _ = self._obs_batch(obs)
        self.physical_public_rms.update(obs_b[:, 0, PHYSICAL_PUBLIC_SLICE])
        self.uav_state_rms.update(obs_b[:, 1:, OWN_SLICE].reshape(-1, UAV_STATE_DIM))

    def normalize_obs(self, obs):
        obs_b, squeezed = self._obs_batch(obs)
        out = obs_b.copy()
        public_raw = obs_b[:, 0, PHYSICAL_PUBLIC_SLICE]
        public_norm = self._normalize(public_raw, self.physical_public_rms)
        out[:, :, PHYSICAL_PUBLIC_SLICE] = public_norm[:, None, :]
        out[:, 0, OWN_SLICE] = public_norm[:, :UAV_STATE_DIM]
        uav_norm = self._normalize(obs_b[:, 1:, OWN_SLICE], self.uav_state_rms)
        out[:, 1:, OWN_SLICE] = uav_norm
        return out[0] if squeezed else out

    def normalize_share_obs(self, share_obs):
        share_b, squeezed = self._share_batch(share_obs)
        state = share_b[:, 0, :].copy()
        public_norm = self._normalize(
            state[:, :PHYSICAL_PUBLIC_STATE_DIM], self.physical_public_rms
        )
        uav_raw = state[:, PUBLIC_STATE_DIM:].reshape(-1, self.num_uavs, UAV_STATE_DIM)
        uav_norm = self._normalize(uav_raw, self.uav_state_rms)
        norm_state = state.copy()
        norm_state[:, :PHYSICAL_PUBLIC_STATE_DIM] = public_norm
        norm_state[:, PUBLIC_STATE_DIM:] = uav_norm.reshape(-1, self.num_uavs * UAV_STATE_DIM)
        out = np.repeat(norm_state[:, None, :], self.num_agents, axis=1)
        return out[0] if squeezed else out.astype(np.float32)

    def copy_from(self, other):
        self._check_num_agents(other.num_agents)
        self.clip_obs = float(other.clip_obs)
        self.epsilon = float(other.epsilon)
        self.physical_public_rms.mean = other.physical_public_rms.mean.copy()
        self.physical_public_rms.var = other.physical_public_rms.var.copy()
        self.physical_public_rms.count = float(other.physical_public_rms.count)
        self.uav_state_rms.mean = other.uav_state_rms.mean.copy()
        self.uav_state_rms.var = other.uav_state_rms.var.copy()
        self.uav_state_rms.count = float(other.uav_state_rms.count)

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "norm_mode": np.asarray(self.norm_mode),
            "num_agents": np.asarray(self.num_agents, dtype=np.int64),
            "clip_obs": np.asarray(self.clip_obs, dtype=np.float64),
            "epsilon": np.asarray(self.epsilon, dtype=np.float64),
        }
        payload.update(self.physical_public_rms.state_dict("physical_public"))
        payload.update(self.uav_state_rms.state_dict("uav_state"))
        np.savez(path, **payload)

    def load(self, path, *, expected_num_agents=None):
        data = np.load(Path(path), allow_pickle=False)
        mode = str(np.asarray(data["norm_mode"])) if "norm_mode" in data else "flat"
        if mode != self.norm_mode:
            raise ValueError(f"expected component vec-normalize stats, got {mode!r}")
        saved_num_agents = int(np.asarray(data["num_agents"]))
        if expected_num_agents is not None and saved_num_agents != int(expected_num_agents):
            raise ValueError(
                "component vec-normalize stats num_agents mismatch: "
                f"saved={saved_num_agents}, current={expected_num_agents}"
            )
        self._check_num_agents(saved_num_agents)
        self.clip_obs = float(np.asarray(data["clip_obs"]))
        self.epsilon = float(np.asarray(data["epsilon"]))
        self.physical_public_rms.load_state_dict(data, "physical_public")
        self.uav_state_rms.load_state_dict(data, "uav_state")

    def _check_num_agents(self, num_agents):
        if int(num_agents) != self.num_agents:
            raise ValueError(
                f"MEC component normalizer num_agents mismatch: "
                f"expected {self.num_agents}, got {num_agents}"
            )

    def _normalize(self, values, rms):
        values = np.asarray(values, dtype=np.float32)
        normed = (values - rms.mean) / np.sqrt(rms.var + self.epsilon)
        return np.clip(normed, -self.clip_obs, self.clip_obs).astype(np.float32)

    def _obs_batch(self, obs):
        obs = np.asarray(obs, dtype=np.float32)
        if obs.ndim == 2:
            obs = obs[None, ...]
            squeezed = True
        elif obs.ndim == 3:
            squeezed = False
        else:
            raise ValueError(f"expected obs shape [N,D] or [E,N,D], got {obs.shape}")
        self._check_num_agents(obs.shape[-2])
        return obs, squeezed

    def _share_batch(self, share_obs):
        share_obs = np.asarray(share_obs, dtype=np.float32)
        if share_obs.ndim == 2:
            share_obs = share_obs[None, ...]
            squeezed = True
        elif share_obs.ndim == 3:
            squeezed = False
        else:
            raise ValueError(
                f"expected share_obs shape [N,D] or [E,N,D], got {share_obs.shape}"
            )
        self._check_num_agents(share_obs.shape[-2])
        expected_dim = PUBLIC_STATE_DIM + self.num_uavs * UAV_STATE_DIM
        if share_obs.shape[-1] != expected_dim:
            raise ValueError(
                f"expected share_obs dim {expected_dim}, got {share_obs.shape[-1]}"
            )
        return share_obs, squeezed


class MECComponentVecNormalize(ShareVecEnv):
    """ShareVecEnv wrapper for MEC component-level normalization."""

    norm_mode = "component"

    def __init__(self, venv, *, training=True, clip_obs=10.0, epsilon=1e-8):
        super().__init__(
            venv.num_envs,
            venv.observation_space,
            venv.share_observation_space,
            venv.action_space,
        )
        self.venv = venv
        self.training = bool(training)
        self.normalizer = MECComponentNormalizer(
            len(self.observation_space),
            clip_obs=clip_obs,
            epsilon=epsilon,
        )

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

    def normalize_obs(self, obs):
        return self.normalizer.normalize_obs(obs)

    def normalize_share_obs(self, share_obs):
        return self.normalizer.normalize_share_obs(share_obs)

    def copy_vec_normalize_from(self, other, *, training=None):
        source = other
        if hasattr(source, "normalizer"):
            self.normalizer.copy_from(source.normalizer)
        elif hasattr(source, "venv") and hasattr(source.venv, "normalizer"):
            self.normalizer.copy_from(source.venv.normalizer)
        else:
            raise ValueError("source env has no component vec-normalize stats")
        if training is not None:
            self.training = bool(training)

    def save_vec_normalize(self, path):
        self.normalizer.save(path)

    def load_vec_normalize(self, path, *, training=None):
        self.normalizer.load(path, expected_num_agents=len(self.observation_space))
        if training is not None:
            self.training = bool(training)

    def _update(self, obs, share_obs):
        if self.training:
            self.normalizer.update(obs, share_obs)

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
