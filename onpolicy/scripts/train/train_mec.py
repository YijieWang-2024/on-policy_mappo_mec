#!/usr/bin/env python
"""Train script for the finite-K HAP/UAV MEC environment (v2 port).

Mirrors train_mpe.py. num_agents = K+1 (major hub + K minor UAVs) is probed from
the scenario rather than passed on the CLI (D6). Continuous Box actions are
handled by MECRunner (D4) + ACTLayer's DiagGaussian.
"""

import os
import random
import socket
import sys
from pathlib import Path

import numpy as np
import setproctitle
import torch

from onpolicy.config import get_config
from onpolicy.envs.env_wrappers import (
    ShareDummyVecEnv,
    ShareSubprocVecEnv,
    ShareVecNormalize,
)
from onpolicy.envs.mec.MEC_env import MECEnv
from onpolicy.envs.mec.normalization import MECComponentVecNormalize
from onpolicy.envs.mec.observation import (
    PHYSICAL_PUBLIC_STATE_DIM,
    PUBLIC_STATE_DIM,
    RESOURCE_CONTEXT_SLICE,
)


def _make_env_fns(all_args, base_seed):
    def get_env_fn(rank):
        def init_env():
            if all_args.env_name != "MEC":
                raise NotImplementedError(f"unsupported env: {all_args.env_name}")
            env = MECEnv(all_args)
            env.seed(base_seed + rank * 1000)
            return env
        return init_env
    return get_env_fn


def make_train_env(all_args):
    fn = _make_env_fns(all_args, all_args.seed)
    if all_args.n_rollout_threads == 1:
        envs = ShareDummyVecEnv([fn(0)])
    else:
        envs = ShareSubprocVecEnv([fn(i) for i in range(all_args.n_rollout_threads)])
    return _maybe_wrap_obs_norm(all_args, envs, training=True)


def make_eval_env(all_args):
    fn = _make_env_fns(all_args, all_args.seed * 50000)
    if all_args.n_eval_rollout_threads == 1:
        envs = ShareDummyVecEnv([fn(0)])
    else:
        envs = ShareSubprocVecEnv([fn(i) for i in range(all_args.n_eval_rollout_threads)])
    return _maybe_wrap_obs_norm(all_args, envs, training=False)


def _mec_norm_masks(envs):
    obs_mask = np.ones(envs.observation_space[0].shape, dtype=bool)
    obs_mask[0] = False
    obs_mask[RESOURCE_CONTEXT_SLICE] = False

    share_obs_mask = np.ones(envs.share_observation_space[0].shape, dtype=bool)
    share_obs_mask[PHYSICAL_PUBLIC_STATE_DIM:PUBLIC_STATE_DIM] = False
    return obs_mask, share_obs_mask


def _maybe_wrap_obs_norm(all_args, envs, *, training):
    if not getattr(all_args, "use_obs_norm", False):
        return envs
    mode = getattr(all_args, "mec_obs_norm_mode", "flat")
    if mode == "component":
        envs = MECComponentVecNormalize(
            envs,
            training=training,
            clip_obs=all_args.obs_norm_clip,
            epsilon=all_args.obs_norm_epsilon,
        )
    elif mode == "flat":
        obs_mask, share_obs_mask = _mec_norm_masks(envs)
        envs = ShareVecNormalize(
            envs,
            training=training,
            clip_obs=all_args.obs_norm_clip,
            epsilon=all_args.obs_norm_epsilon,
            obs_mask=obs_mask,
            share_obs_mask=share_obs_mask,
            share_obs_unique=True,
        )
    else:
        raise ValueError(f"unknown MEC obs norm mode: {mode}")
    if all_args.model_dir:
        stats_path = Path(all_args.model_dir) / "vec_normalize.npz"
        if stats_path.exists():
            envs.load_vec_normalize(stats_path, training=training)
        else:
            print(f"obs normalization requested but {stats_path} was not found")
    return envs


def parse_args(args, parser):
    parser.add_argument("--mec_scenario", type=str, default="v6_hap_loadbearing",
                        help="scenario yaml name under onpolicy/envs/mec/scenarios")
    parser.add_argument("--mec_fleet_size", type=int, default=None,
                        help="override K (UAV count); num_agents becomes K+1")
    parser.add_argument(
        "--mec_episode_horizon",
        type=int,
        default=None,
        help=(
            "override the MEC scenario horizon; must equal --episode_length "
            "so each rollout is one complete environment episode"
        ),
    )
    parser.add_argument(
        "--mec_obs_norm_mode",
        choices=["flat", "component"],
        default="flat",
        help="MEC obs/share_obs RMS mode used when --use_obs_norm is enabled",
    )
    return parser.parse_known_args(args)[0]


def _probe_env_contract(all_args):
    probe = MECEnv(all_args)
    n = probe.num_agents
    horizon = probe.env.horizon
    probe.close()
    return n, horizon


def main(args):
    parser = get_config()
    all_args = parse_args(args, parser)

    if all_args.algorithm_name == "rmappo":
        all_args.use_recurrent_policy = True
        all_args.use_naive_recurrent_policy = False
    elif all_args.algorithm_name in ("mappo", "ippo"):
        all_args.use_recurrent_policy = False
        all_args.use_naive_recurrent_policy = False
        if all_args.algorithm_name == "ippo":
            all_args.use_centralized_V = False
    elif all_args.algorithm_name not in ("mat", "mat_dec"):
        raise NotImplementedError

    device = torch.device("cuda:0") if (all_args.cuda and torch.cuda.is_available()) else torch.device("cpu")
    torch.set_num_threads(all_args.n_training_threads)

    run_dir = (Path(os.path.dirname(os.path.abspath(__file__))).parents[0]
               / "results" / all_args.env_name / all_args.mec_scenario
               / all_args.algorithm_name / all_args.experiment_name)
    if all_args.use_wandb:
        import wandb
        run = wandb.init(config=all_args, project=all_args.env_name, entity=all_args.user_name,
                         notes=socket.gethostname(),
                         name=f"{all_args.algorithm_name}_{all_args.experiment_name}_seed{all_args.seed}",
                         group=all_args.mec_scenario, dir=str(run_dir), reinit=True)
    else:
        run = None
        if not run_dir.exists():
            os.makedirs(str(run_dir))
        existing = [int(f.name[3:]) for f in run_dir.iterdir() if f.name.startswith("run") and f.name[3:].isdigit()]
        run_dir = run_dir / f"run{(max(existing) + 1) if existing else 1}"
        if not run_dir.exists():
            os.makedirs(str(run_dir))

    setproctitle.setproctitle(
        f"{all_args.algorithm_name}-{all_args.env_name}-{all_args.experiment_name}@{all_args.user_name}")

    torch.manual_seed(all_args.seed)
    torch.cuda.manual_seed_all(all_args.seed)
    np.random.seed(all_args.seed)
    random.seed(all_args.seed)

    num_agents, env_horizon = _probe_env_contract(all_args)
    if int(all_args.episode_length) != int(env_horizon):
        raise ValueError(
            "--episode_length must match the MEC environment horizon: "
            f"episode_length={all_args.episode_length}, "
            f"environment_horizon={env_horizon}. Pass both "
            "--episode_length and --mec_episode_horizon when changing it."
        )
    all_args.num_agents = num_agents
    all_args.scenario_name = all_args.mec_scenario   # base runner logs scenario_name

    envs = make_train_env(all_args)
    eval_envs = make_eval_env(all_args) if all_args.use_eval else None

    config = {"all_args": all_args, "envs": envs, "eval_envs": eval_envs,
              "num_agents": num_agents, "device": device, "run_dir": run_dir}

    from onpolicy.runner.shared.mec_runner import MECRunner as Runner
    runner = Runner(config)
    runner.run()

    envs.close()
    if all_args.use_eval and eval_envs is not envs:
        eval_envs.close()
    if run is not None:
        run.finish()
    else:
        runner.writter.export_scalars_to_json(str(runner.log_dir + "/summary.json"))
        runner.writter.close()


if __name__ == "__main__":
    main(sys.argv[1:])
