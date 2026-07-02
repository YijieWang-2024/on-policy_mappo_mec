#!/bin/sh
script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
project_root=$(CDPATH= cd -- "${script_dir}/../../.." && pwd)
cd "${project_root}" || exit 1

env="MPE"
scenario="simple_reference"
num_landmarks=3
num_agents=2
algo="rmappo" # "mappo" "ippo" "mat" "mat_dec"
exp="check"
seed_max=1

echo "env is ${env}, scenario is ${scenario}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"
for seed in `seq ${seed_max}`;
do
    echo "seed is ${seed}:"
    CUDA_VISIBLE_DEVICES=0 python -m onpolicy.scripts.train.train_mpe --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} \
    --scenario_name ${scenario} --num_agents ${num_agents} --num_landmarks ${num_landmarks} --seed ${seed} \
    --n_training_threads 1 --n_rollout_threads 128 --num_mini_batch 1 --episode_length 25 --num_env_steps 3000000 \
    --ppo_epoch 15 --gain 0.01 --lr 7e-4 --critic_lr 7e-4 --user_name "yuchao"
done
