from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from onpolicy.config import get_config
from onpolicy.envs.mec.MEC_env import ACT_DIM, MECEnv
from onpolicy.envs.mec.normalization import MECComponentNormalizer
from onpolicy.algorithms.mec.mec_policy import MECPolicy
from onpolicy.utils.run_config import apply_saved_args, explicit_option_names, load_model_config


def _parse(argv):
    parser = get_config()
    parser.add_argument("--mec_scenario", type=str, default="v6_hap_loadbearing")
    parser.add_argument("--mec_fleet_size", type=int, default=None)
    parser.add_argument("--mec_episode_horizon", type=int, default=None)
    parser.add_argument("--mec_eval_controller", choices=["policy", "random", "hover", "heuristic"], default="policy")
    parser.add_argument("--mec_eval_seed", type=int, default=None)
    parser.add_argument("--mec_eval_seed_stride", type=int, default=None)
    parser.add_argument("--mec_eval_episodes", type=int, default=None)
    parser.add_argument("--mec_obs_norm_mode", choices=["flat", "component"], default="flat")
    args = parser.parse_known_args(argv)[0]
    saved = load_model_config(args.model_dir)
    if saved:
        apply_saved_args(args, saved, explicit_option_names(argv), skip={"model_dir"})
    if args.mec_eval_seed is None:
        args.mec_eval_seed = args.test_seed
    if args.mec_eval_seed_stride is None:
        args.mec_eval_seed_stride = args.test_seed_stride
    if args.mec_eval_episodes is None:
        args.mec_eval_episodes = args.test_episodes
    args.env_name = "MEC"
    args.use_recurrent_policy = False
    args.use_naive_recurrent_policy = False
    return args


def _load_policy(args, env):
    policy = MECPolicy(
        args,
        env.observation_space[0],
        env.share_observation_space[0],
        env.action_space[0],
        torch.device("cpu"),
        num_agents=env.num_agents,
    )
    if args.model_dir:
        model_dir = Path(args.model_dir)
        policy.actor.load_state_dict(torch.load(str(model_dir / "actor.pt"), map_location="cpu"))
        critic = model_dir / "critic.pt"
        if critic.exists():
            policy.critic.load_state_dict(torch.load(str(critic), map_location="cpu"))
    return policy


def _load_obs_norm(args):
    if not getattr(args, "use_obs_norm", False):
        return None
    stats_path = Path(args.model_dir or "") / "vec_normalize.npz"
    if not stats_path.exists():
        raise FileNotFoundError(
            f"--use_obs_norm is enabled but {stats_path} was not found"
        )
    data = np.load(stats_path, allow_pickle=False)
    mode = str(np.asarray(data["norm_mode"])) if "norm_mode" in data else "flat"
    if mode == "component":
        normalizer = MECComponentNormalizer(
            args.num_agents,
            clip_obs=args.obs_norm_clip,
            epsilon=args.obs_norm_epsilon,
        )
        normalizer.load(stats_path, expected_num_agents=args.num_agents)
        return normalizer
    if mode != "flat":
        raise ValueError(f"unknown vec_normalize norm_mode: {mode}")
    return {
        "mode": "flat",
        "mean": np.asarray(data["obs_mean"], dtype=np.float64),
        "var": np.asarray(data["obs_var"], dtype=np.float64),
        "mask": np.asarray(data["obs_mask"], dtype=bool),
        "clip": float(np.asarray(data["clip_obs"])),
        "epsilon": float(np.asarray(data["epsilon"])),
    }


def _normalize_obs(obs, norm):
    if norm is None:
        return obs
    if hasattr(norm, "normalize_obs"):
        return norm.normalize_obs(obs)
    obs = np.asarray(obs, dtype=np.float32)
    out = obs.copy()
    normed = (obs - norm["mean"]) / np.sqrt(norm["var"] + norm["epsilon"])
    normed = np.clip(normed, -norm["clip"], norm["clip"])
    out[..., norm["mask"]] = normed[..., norm["mask"]]
    return out.astype(np.float32)


def _heuristic_action(env):
    st = env.env.state
    center = st.demand_center_m

    def unit_to(target, current):
        vec = np.asarray(target, dtype=float) - np.asarray(current, dtype=float)
        norm = np.linalg.norm(vec)
        return np.zeros(2) if norm <= 1e-12 else vec / norm

    action = np.zeros((env.num_agents, ACT_DIM), dtype=np.float32)
    action[0, :2] = unit_to(center, st.hap_xy_m)
    action[1:, :2] = np.asarray([unit_to(center, xy) for xy in st.uav_xy_m])
    action[1:, 2] = 0.5
    return action


@torch.no_grad()
def _policy_action(policy, env, obs, rnn_states, masks):
    actions, rnn_states = policy.act(
        obs,
        rnn_states,
        masks,
        deterministic=True,
    )
    return actions.cpu().numpy(), rnn_states


def _episode(args, policy, seed, obs_norm=None):
    env = MECEnv(args)
    obs, _, _ = env.reset(seed=seed)
    policy_obs = _normalize_obs(obs, obs_norm)
    rnn_states = np.zeros((env.num_agents, args.recurrent_N, args.hidden_size), dtype=np.float32)
    masks = np.ones((env.num_agents, 1), dtype=np.float32)
    total_reward = 0.0
    last_info = None

    for _ in range(env.env.horizon):
        if args.mec_eval_controller == "policy":
            action, rnn_states = _policy_action(policy, env, policy_obs, rnn_states, masks)
        elif args.mec_eval_controller == "random":
            action = np.random.default_rng(seed).uniform(-1, 1, (env.num_agents, ACT_DIM))
        elif args.mec_eval_controller == "hover":
            action = np.zeros((env.num_agents, ACT_DIM), dtype=np.float32)
            action[1:, 2] = 0.5
        else:
            action = _heuristic_action(env)
        obs, _, rewards, terminated, truncated, infos = env.step(action)
        policy_obs = _normalize_obs(obs, obs_norm)
        total_reward += float(rewards[0][0])
        last_info = infos[0]
        done = np.asarray(terminated) | np.asarray(truncated)
        rnn_states[done] = 0
        masks = (~done)[:, None].astype(np.float32)
        if all(done):
            break

    return {
        "episode_reward": total_reward,
        "cost_per_slot": -total_reward / env.env.horizon,
        "accept_rate": last_info["accepted"] / max(last_info["accepted"] + last_info["U_src"], 1e-12),
        "accepted_mbit": last_info["accepted"] / 1e6,
        "overflow_mbit": last_info["overflow"] / 1e6,
        "w1": last_info["w1"],
    }


def main(argv=None):
    args = _parse(sys.argv[1:] if argv is None else argv)
    probe = MECEnv(args)
    args.num_agents = probe.num_agents
    if args.mec_episode_horizon is None:
        args.mec_episode_horizon = probe.env.horizon
    probe.close()
    policy = _load_policy(args, MECEnv(args)) if args.mec_eval_controller == "policy" else None
    obs_norm = _load_obs_norm(args) if args.mec_eval_controller == "policy" else None

    rows = [
        _episode(args, policy, args.mec_eval_seed + i * args.mec_eval_seed_stride, obs_norm)
        for i in range(args.mec_eval_episodes)
    ]
    mean = {k: float(np.mean([row[k] for row in rows])) for k in rows[0]}
    print(json.dumps({"controller": args.mec_eval_controller, "episodes": len(rows), "mean": mean}, indent=2))


if __name__ == "__main__":
    main()
