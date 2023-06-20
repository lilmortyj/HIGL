GPU=$1
SEED=$2

CUDA_VISIBLE_DEVICES=${GPU} python main.py \
--env_name "AntMazeT-v0" \
--reward_shaping dense \
--algo hiro \
--version "dense" \
--seed ${SEED} \
--max_timesteps 10e5 \
--landmark_sampling none \
--save_models