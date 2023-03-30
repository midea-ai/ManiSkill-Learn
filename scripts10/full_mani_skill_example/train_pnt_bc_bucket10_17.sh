#!/bin/bash
# change the environment name to OpenCabinetDoor, OpenCabinetDrawer, PushChair, or MoveBucket
# change the network config
# increase eval_cfg.num_procs for parallel evaluation

model_list=$(python -c "import mani_skill, os, os.path as osp; print(osp.abspath(osp.join(osp.dirname(mani_skill.__file__), 'assets', 'config_files', 'bucket_models.yml')))")
echo ${model_list}

python -m tools.run_rl configs/bc/mani_skill_point_cloud_transformer10.py --gpu-ids=3 \
	--work-dir=./work_dirs/bc_pointnet_transformer_bucket10_17/ \
	--cfg-options "train_mfrl_cfg.total_steps=150000" "train_mfrl_cfg.init_replay_buffers=" \
	"train_mfrl_cfg.init_replay_with_split=[\"./full_mani_skill_data/MoveBucket/\",\"$model_list\"]" \
	"env_cfg.env_name=MoveBucket-v0" "eval_cfg.num=100" "eval_cfg.num_procs=1" "train_mfrl_cfg.n_eval=10000" \
	"agent.policy_cfg.nn_cfg.matrix_index=17"


# python -m tools.run_rl configs/bc/mani_skill_point_cloud_transformer2.py --gpu-ids=0 \
# 	--work-dir=./work_dirs/bc_pointnet_transformer_door2_0/ \
# 	--cfg-options "train_mfrl_cfg.total_steps=150000" "train_mfrl_cfg.init_replay_buffers=" \
# 	"train_mfrl_cfg.init_replay_with_split=[\"./full_mani_skill_data/OpenCabinetDoor/\",\"$model_list\"]" \
# 	"env_cfg.env_name=OpenCabinetDoor-v0" "eval_cfg.num=100" "eval_cfg.num_procs=1" "train_mfrl_cfg.n_eval=10000" \
# 	"agent.policy_cfg.nn_cfg.matrix_index=0"
