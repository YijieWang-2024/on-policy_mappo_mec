#!/bin/sh
script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
project_root=$(CDPATH= cd -- "${script_dir}/../.." && pwd)
cd "${project_root}" || exit 1

env="MPE"
scenario="simple_speaker_listener"
num_landmarks=3
num_agents=2
algo="rmappo"
exp="check"
seed_max=1

echo "env is ${env}"
for seed in `seq ${seed_max}`
do
    CUDA_VISIBLE_DEVICES=0 python -m onpolicy.scripts.render.render_mpe --save_gifs --share_policy --env_name ${env} --algorithm_name ${algo} \
    --experiment_name ${exp} --scenario_name ${scenario} --num_agents ${num_agents} --num_landmarks ${num_landmarks} --seed ${seed} \
    --n_training_threads 1 --n_rollout_threads 1 --use_render --episode_length 25 --render_episodes 5 \
    --model_dir "xxx"
done
