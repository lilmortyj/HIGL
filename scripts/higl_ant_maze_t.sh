GPU=$1
SEED=$2

CUDA_VISIBLE_DEVICES=${GPU} python main.py \
--env_name "AntMazeT-v0" \
--reward_shaping dense \
--algo higl \
--version "dense" \
--seed ${SEED} \
--max_timesteps 10e5 \
--landmark_sampling fps \
--n_landmark_coverage 20 \
--use_novelty_landmark \
--novelty_algo rnd \
--n_landmark_novelty 20 \
--save_models