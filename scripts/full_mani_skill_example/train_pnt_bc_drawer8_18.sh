#!/bin/bash
# change the environment name to OpenCabinetDoor, OpenCabinetDrawer, PushChair, or MoveBucket
# change the network config
# increase eval_cfg.num_procs for parallel evaluation

model_list=$(python -c "import mani_skill, os, os.path as osp; print(osp.abspath(osp.join(osp.dirname(mani_skill.__file__), 'assets', 'config_files', 'cabinet_models_drawer.yml')))")


#python -m tools.run_rl configs/bc/mani_skill_point_cloud_transformer.py --gpu-ids=0 \
#	--work-dir=./work_dirs/bc_transformer_drawer/ \
#	--cfg-options "train_mfrl_cfg.total_steps=150000" "train_mfrl_cfg.init_replay_buffers=" \
#	"train_mfrl_cfg.init_replay_with_split=[\"full_mani_skill_data/openCabinetDrawer/\",\"$model_list\"]" \
#	"env_cfg.env_name=OpenCabinetDrawer-v0" "eval_cfg.num=300" "eval_cfg.num_procs=1" "train_mfrl_cfg.n_eval=150000"

python -m tools.run_rl configs/bc/mani_skill_point_cloud_transformer8.py --gpu-ids=1 \
	--work-dir=./work_dirs/bc_pointnet_transformer_drawer8_18/ \
	--cfg-options "train_mfrl_cfg.total_steps=150000" "train_mfrl_cfg.init_replay_buffers=" \
	"train_mfrl_cfg.init_replay_with_split=[\"./full_mani_skill_data/OpenCabinetDrawer\",\"$model_list\"]" \
	"env_cfg.env_name=OpenCabinetDrawer-v0" "eval_cfg.num=100" "eval_cfg.num_procs=1" "train_mfrl_cfg.n_eval=10000" \
	"agent.policy_cfg.nn_cfg.matrix_index=18"

sleep 10

# python -m tools.run_rl configs/bc/mani_skill_point_cloud_transformer8.py --gpu-ids=2 \
# 	--work-dir=./work_dirs/bc_pointnet_transformer_drawer8_1/ \
# 	--cfg-options "train_mfrl_cfg.total_steps=150000" "train_mfrl_cfg.init_replay_buffers=" \
# 	"train_mfrl_cfg.init_replay_with_split=[\"./full_mani_skill_data/OpenCabinetDrawer\",\"$model_list\"]" \
# 	"env_cfg.env_name=OpenCabinetDrawer-v0" "eval_cfg.num=100" "eval_cfg.num_procs=1" "train_mfrl_cfg.n_eval=10000" \
# 	"agent.policy_cfg.nn_cfg.matrix_index=1"

# sleep 10

# python -m tools.run_rl configs/bc/mani_skill_point_cloud_transformer8.py --gpu-ids=3 \
# 	--work-dir=./work_dirs/bc_pointnet_transformer_drawer8_2/ \
# 	--cfg-options "train_mfrl_cfg.total_steps=150000" "train_mfrl_cfg.init_replay_buffers=" \
# 	"train_mfrl_cfg.init_replay_with_split=[\"./full_mani_skill_data/OpenCabinetDrawer\",\"$model_list\"]" \
# 	"env_cfg.env_name=OpenCabinetDrawer-v0" "eval_cfg.num=100" "eval_cfg.num_procs=1" "train_mfrl_cfg.n_eval=10000" \
# 	"agent.policy_cfg.nn_cfg.matrix_index=2"

# sleep 10

# python -m tools.run_rl configs/bc/mani_skill_point_cloud_transformer8.py --gpu-ids=1 \
# 	--work-dir=./work_dirs/bc_pointnet_transformer_drawer8_3/ \
# 	--cfg-options "train_mfrl_cfg.total_steps=150000" "train_mfrl_cfg.init_replay_buffers=" \
# 	"train_mfrl_cfg.init_replay_with_split=[\"./full_mani_skill_data/OpenCabinetDrawer\",\"$model_list\"]" \
# 	"env_cfg.env_name=OpenCabinetDrawer-v0" "eval_cfg.num=100" "eval_cfg.num_procs=1" "train_mfrl_cfg.n_eval=10000" \
# 	"agent.policy_cfg.nn_cfg.matrix_index=3"

# sleep 10

# python -m tools.run_rl configs/bc/mani_skill_point_cloud_transformer8.py --gpu-ids=2 \
# 	--work-dir=./work_dirs/bc_pointnet_transformer_drawer8_4/ \
# 	--cfg-options "train_mfrl_cfg.total_steps=150000" "train_mfrl_cfg.init_replay_buffers=" \
# 	"train_mfrl_cfg.init_replay_with_split=[\"./full_mani_skill_data/OpenCabinetDrawer\",\"$model_list\"]" \
# 	"env_cfg.env_name=OpenCabinetDrawer-v0" "eval_cfg.num=100" "eval_cfg.num_procs=1" "train_mfrl_cfg.n_eval=10000" \
# 	"agent.policy_cfg.nn_cfg.matrix_index=4"

# sleep 10
