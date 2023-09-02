GPU=$1
SEED=$2

CUDA_VISIBLE_DEVICES=${GPU} python main.py \
--env_name "AntReacher-v0" \
--reward_shaping dense \
--algo hrac-ft \
--version "dense" \
--seed ${SEED} \
--max_timesteps 10e5 \
--landmark_sampling none \
--load \
--load_dir "./models/20230901133750" \
--save_models \
--ME