GPU=$1

CUDA_VISIBLE_DEVICES=${GPU} python main.py \
--env_name "AntReacher-v0" \
--reward_shaping dense \
--algo hrac-ft \
--version "dense" \
--max_timesteps 10e5 \
--landmark_sampling none \
--load \
--load_dir "./models/20230901133750" \
--save_models \
--ME_train \
--ME_eval \
--per_timestep_collect