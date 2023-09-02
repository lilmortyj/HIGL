GPU=$1

CUDA_VISIBLE_DEVICES=${GPU} python main.py \
--env_name "AntReacher-v0" \
--reward_shaping dense \
--algo hrac \
--version "dense" \
--max_timesteps 10e5 \
--landmark_sampling none \
--save_models \
--rand_eval