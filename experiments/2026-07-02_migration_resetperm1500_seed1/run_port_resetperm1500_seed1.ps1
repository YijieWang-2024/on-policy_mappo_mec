$ErrorActionPreference = "Stop"

$common = @(
  "-m", "onpolicy.scripts.train.train_mec",
  "--env_name", "MEC",
  "--algorithm_name", "mappo",
  "--mec_scenario", "v6_hap_loadbearing",
  "--seed", "1",
  "--n_rollout_threads", "16",
  "--n_eval_rollout_threads", "8",
  "--episode_length", "350",
  "--mec_episode_horizon", "350",
  "--num_env_steps", "1500000",
  "--ppo_epoch", "5",
  "--num_mini_batch", "1",
  "--hidden_size", "128",
  "--layer_N", "2",
  "--mec_logstd_init", "-1.2",
  "--entropy_coef", "0.003",
  "--use_entropy_anneal",
  "--mec_rolewise_loss",
  "--use_eval",
  "--eval_interval", "18",
  "--eval_episodes", "24",
  "--eval_seed", "1000",
  "--save_step_checkpoints",
  "--use_wandb"
)

conda run -n marl python @common `
  --experiment_name "port_resetperm1500_mean_seed1" `
  --mec_policy_arch "mean"

conda run -n marl python @common `
  --experiment_name "port_resetperm1500_flat_seed1" `
  --mec_policy_arch "flat"
