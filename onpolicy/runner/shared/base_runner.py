import wandb
import os
import json
import numpy as np
import torch
from pathlib import Path
from tensorboardX import SummaryWriter
from onpolicy.utils.shared_buffer import SharedReplayBuffer
from onpolicy.utils.run_config import save_run_config

def _t2n(x):
    """Convert torch tensor to a numpy array."""
    return x.detach().cpu().numpy()

class Runner(object):
    """
    Base class for training recurrent policies.
    :param config: (dict) Config dictionary containing parameters for training.
    """
    def __init__(self, config):

        self.all_args = config['all_args']
        self.envs = config['envs']
        self.eval_envs = config['eval_envs']
        self.device = config['device']
        self.num_agents = config['num_agents']
        if config.__contains__("render_envs"):
            self.render_envs = config['render_envs']       

        # parameters
        self.env_name = self.all_args.env_name
        self.algorithm_name = self.all_args.algorithm_name
        self.experiment_name = self.all_args.experiment_name
        self.use_centralized_V = self.all_args.use_centralized_V
        self.use_obs_instead_of_state = self.all_args.use_obs_instead_of_state
        self.num_env_steps = self.all_args.num_env_steps
        self.episode_length = self.all_args.episode_length
        self.n_rollout_threads = self.all_args.n_rollout_threads
        self.n_eval_rollout_threads = self.all_args.n_eval_rollout_threads
        self.n_render_rollout_threads = self.all_args.n_render_rollout_threads
        self.use_linear_lr_decay = self.all_args.use_linear_lr_decay
        self.hidden_size = self.all_args.hidden_size
        self.use_wandb = self.all_args.use_wandb
        self.use_render = self.all_args.use_render
        self.recurrent_N = self.all_args.recurrent_N

        # interval
        self.save_interval = self.all_args.save_interval
        self.use_eval = self.all_args.use_eval
        self.eval_interval = self.all_args.eval_interval
        self.log_interval = self.all_args.log_interval
        self.save_step_checkpoints = getattr(
            self.all_args, "save_step_checkpoints", False
        )
        self.best_eval_reward = -np.inf

        # dir
        self.model_dir = self.all_args.model_dir

        if self.use_render:
            self.run_dir = config["run_dir"]
            self.gif_dir = str(self.run_dir / "gifs")
            if not os.path.exists(self.gif_dir):
                os.makedirs(self.gif_dir)
        elif self.use_wandb:
            self.save_dir = str(wandb.run.dir)
            self.run_dir = str(wandb.run.dir)
        else:
            self.run_dir = config["run_dir"]
            self.log_dir = str(self.run_dir / 'logs')
            if not os.path.exists(self.log_dir):
                os.makedirs(self.log_dir)
            self.writter = SummaryWriter(self.log_dir)
            self.save_dir = str(self.run_dir / 'models')
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)

        if not self.use_render:
            save_run_config(
                self.all_args,
                self.run_dir,
                self.save_dir,
                num_agents=self.num_agents,
            )

        if self.algorithm_name == "mat" or self.algorithm_name == "mat_dec":
            from onpolicy.algorithms.mat.mat_trainer import MATTrainer as TrainAlgo
            from onpolicy.algorithms.mat.algorithm.transformer_policy import TransformerPolicy as Policy
        else:
            from onpolicy.algorithms.r_mappo.r_mappo import R_MAPPO as TrainAlgo
            from onpolicy.algorithms.r_mappo.algorithm.rMAPPOPolicy import R_MAPPOPolicy as Policy
            if self.env_name == "MEC":
                # major-minor shared-parameter actor (hub + K shared UAVs); same R_MAPPO trainer
                from onpolicy.algorithms.mec.mec_policy import MECPolicy as Policy

        share_observation_space = self.envs.share_observation_space[0] if self.use_centralized_V else self.envs.observation_space[0]

        print("obs_space: ", self.envs.observation_space)
        print("share_obs_space: ", self.envs.share_observation_space)
        print("act_space: ", self.envs.action_space)
        
        # policy network
        if self.algorithm_name == "mat" or self.algorithm_name == "mat_dec":
            self.policy = Policy(self.all_args, self.envs.observation_space[0], share_observation_space, self.envs.action_space[0], self.num_agents, device = self.device)
        elif self.env_name == "MEC":
            self.policy = Policy(
                self.all_args,
                self.envs.observation_space[0],
                share_observation_space,
                self.envs.action_space[0],
                device=self.device,
                num_agents=self.num_agents,
            )
        else:
            self.policy = Policy(self.all_args, self.envs.observation_space[0], share_observation_space, self.envs.action_space[0], device = self.device)

        pretrained_actor = getattr(
            self.all_args, "mec_set_pretrained_actor", None
        )
        if self.env_name == "MEC" and pretrained_actor:
            actor_state = torch.load(
                str(pretrained_actor), map_location=self.device
            )
            loaded_keys = self.policy.load_set_pretrained_actor(
                actor_state
            )
            print(
                "loaded MEC Set pretrained representation "
                f"from {pretrained_actor} ({len(loaded_keys)} tensors)"
            )

        # algorithm
        if self.algorithm_name == "mat" or self.algorithm_name == "mat_dec":
            self.trainer = TrainAlgo(self.all_args, self.policy, self.num_agents, device = self.device)
        else:
            self.trainer = TrainAlgo(self.all_args, self.policy, device = self.device)

        if self.model_dir is not None:
            self.restore(self.model_dir)
        
        # buffer
        self.buffer = SharedReplayBuffer(self.all_args,
                                        self.num_agents,
                                        self.envs.observation_space[0],
                                        share_observation_space,
                                        self.envs.action_space[0])

    def run(self):
        """Collect training data, perform training updates, and evaluate policy."""
        raise NotImplementedError

    def warmup(self):
        """Collect warmup pre-training data."""
        raise NotImplementedError

    def collect(self, step):
        """Collect rollouts for training."""
        raise NotImplementedError

    def insert(self, data):
        """
        Insert data into buffer.
        :param data: (Tuple) data to insert into training buffer.
        """
        raise NotImplementedError
    
    @torch.no_grad()
    def compute(self):
        """Calculate returns for the collected data."""
        self.buffer.compute_returns(self.trainer.value_normalizer)
    
    def train(self):
        """Train policies with data in buffer. """
        self.trainer.prep_training()
        train_infos = self.trainer.train(self.buffer)      
        self.buffer.after_update()
        return train_infos

    def _save_training_state(self, target_dir, total_num_steps=None):
        """Save optimizer/value-normalizer state for resumable checkpoints."""
        if self.algorithm_name in ("mat", "mat_dec"):
            return
        state = {
            "total_num_steps": total_num_steps,
            "actor_optimizer": self.trainer.policy.actor_optimizer.state_dict(),
            "critic_optimizer": self.trainer.policy.critic_optimizer.state_dict(),
        }
        if self.trainer.value_normalizer is not None:
            state["value_normalizer"] = (
                self.trainer.value_normalizer.state_dict()
            )
        torch.save(state, str(Path(target_dir) / "trainer_state.pt"))

    def _save_models_to(self, target_dir, episode=0, total_num_steps=None):
        target_dir = Path(target_dir)
        target_dir.mkdir(parents=True, exist_ok=True)
        if self.algorithm_name == "mat" or self.algorithm_name == "mat_dec":
            self.policy.save(str(target_dir), episode)
        else:
            policy_actor = self.trainer.policy.actor
            torch.save(
                policy_actor.state_dict(), str(target_dir / "actor.pt")
            )
            policy_critic = self.trainer.policy.critic
            torch.save(
                policy_critic.state_dict(), str(target_dir / "critic.pt")
            )
            self._save_training_state(target_dir, total_num_steps)
        save_run_config(
            self.all_args,
            self.run_dir,
            target_dir,
            num_agents=self.num_agents,
        )

    def save(self, episode=0, total_num_steps=None):
        """Save latest models and optionally retain a numbered checkpoint."""
        self._save_models_to(
            self.save_dir, episode=episode, total_num_steps=total_num_steps
        )
        if self.save_step_checkpoints and total_num_steps is not None:
            checkpoint_dir = (
                Path(self.save_dir)
                / "checkpoints"
                / f"step_{int(total_num_steps):012d}"
            )
            self._save_models_to(
                checkpoint_dir,
                episode=episode,
                total_num_steps=total_num_steps,
            )

    def maybe_save_best(self, eval_reward, total_num_steps):
        """Keep the checkpoint with the highest fixed-validation reward."""
        if eval_reward is None or not np.isfinite(eval_reward):
            return False
        if eval_reward <= self.best_eval_reward:
            return False
        self.best_eval_reward = float(eval_reward)
        best_dir = Path(self.save_dir) / "best"
        self._save_models_to(best_dir, total_num_steps=total_num_steps)
        metadata = {
            "selection_metric": "eval_average_episode_rewards",
            "selection_mode": "max",
            "selection_split": "validation",
            "eval_reward": self.best_eval_reward,
            "total_num_steps": int(total_num_steps),
            "validation_seed": int(
                getattr(self.all_args, "eval_seed", 1000)
            ),
            "validation_episodes": int(self.all_args.eval_episodes),
        }
        with (best_dir / "best_checkpoint.json").open(
            "w", encoding="utf-8"
        ) as f:
            json.dump(metadata, f, indent=2, sort_keys=True)
            f.write("\n")
        return True

    def restore(self, model_dir):
        """Restore policy's networks from a saved model."""
        model_dir = Path(model_dir)
        if self.algorithm_name == "mat" or self.algorithm_name == "mat_dec":
            self.policy.restore(str(model_dir))
        else:
            policy_actor_state_dict = torch.load(
                str(model_dir / "actor.pt"), map_location=self.device
            )
            self.policy.actor.load_state_dict(policy_actor_state_dict)
            if not self.all_args.use_render:
                policy_critic_state_dict = torch.load(
                    str(model_dir / "critic.pt"), map_location=self.device
                )
                self.policy.critic.load_state_dict(policy_critic_state_dict)
            trainer_state_path = model_dir / "trainer_state.pt"
            if trainer_state_path.exists() and not self.all_args.use_render:
                state = torch.load(
                    str(trainer_state_path), map_location=self.device
                )
                if "actor_optimizer" in state:
                    self.trainer.policy.actor_optimizer.load_state_dict(
                        state["actor_optimizer"]
                    )
                if "critic_optimizer" in state:
                    self.trainer.policy.critic_optimizer.load_state_dict(
                        state["critic_optimizer"]
                    )
                if (
                    self.trainer.value_normalizer is not None
                    and "value_normalizer" in state
                ):
                    self.trainer.value_normalizer.load_state_dict(
                        state["value_normalizer"]
                    )

    def log_train(self, train_infos, total_num_steps):
        """
        Log training info.
        :param train_infos: (dict) information about training update.
        :param total_num_steps: (int) total number of training env steps.
        """
        for k, v in train_infos.items():
            if self.use_wandb:
                wandb.log({k: v}, step=total_num_steps)
            else:
                self.writter.add_scalar(k, v, total_num_steps)

    def log_env(self, env_infos, total_num_steps):
        """
        Log env info.
        :param env_infos: (dict) information about env state.
        :param total_num_steps: (int) total number of training env steps.
        """
        for k, v in env_infos.items():
            if len(v)>0:
                if self.use_wandb:
                    wandb.log({k: np.mean(v)}, step=total_num_steps)
                else:
                    self.writter.add_scalar(k, np.mean(v), total_num_steps)
